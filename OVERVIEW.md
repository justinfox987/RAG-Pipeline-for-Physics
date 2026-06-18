# RAG Pipeline for Physics — System Overview

Physics research assistant built on CBorg (LBL's hosted LLM API at `api.cborg.lbl.gov`). Users ingest PDFs into per-topic FAISS vector indexes; at query time the system retrieves the most relevant pages and sends them with the question to an LLM for a grounded answer.

---

## File Map

```
config.py              — all constants (single source of truth)
ingest.py              — PDF → FAISS ingestion pipeline
query.py               — retrieval + reasoning pipeline
session_store.py       — chat session persistence (JSON files)
app.py                 — Streamlit main page (chat UI)
pages/1_Library.py     — browse indexed topics/files
pages/2_Ingest.py      — upload PDFs via UI and trigger ingestion
providers/
  __init__.py          — get_provider() factory (reads LLM_PROVIDER env var)
  base.py              — LLMProvider protocol definition
  cborg_provider.py    — CBorg/OpenAI-compatible implementation
cost_estimator.py      — dry-run cost estimate before ingesting
build_base.py          — build + package a distributable base index
find_max_workers.py    — binary-search for max safe concurrent vision workers
reset.py               — wipe indexes (all or per-topic)
test.py                — scratch pad for one-off API calls
```

---

## Provider Layer

All API calls go through `providers/cborg_provider.py` via an OpenAI-compatible client pointed at `https://api.cborg.lbl.gov`. The `LLMProvider` protocol (base.py) has four methods:

- `embed_texts(texts, model)` → `(embeddings, usage)`
- `transcribe_image(data_uri, prompt, model, ...)` → `(text, usage)`
- `reason(system_prompt, user_messages, model, ...)` → `(text, usage)`
- `get_budget_info()` → dict with `spend`, `max_budget`, `budget_reset_at`

**Models in use:**

| Role | Model | Notes |
|---|---|---|
| Chat/reasoning | `gemma-4` (default) | Configurable per query |
| Query routing | `cborg-mini-fast` | Gemma 4 E2B non-thinking; classifies YES/NO |
| Vision transcription | `gemini-3.1-flash-lite` | Routed through Bedrock |
| Embedding | `cohere-embed-v4` | 1536-dim MRL, routed through Bedrock |

**Rate limits:** Cohere embed-v4 via Bedrock throttles by tokens/minute. Safe concurrency is `EMBED_WORKERS=2` (32 pages per burst ≈ 8,000 tokens). Vision is more tolerant; `MAX_WORKERS=24` is configured but the actual safe value should be confirmed with `find_max_workers.py`.

**Pricing (CBorg, $/1M tokens):**

| Model | Input | Output |
|---|---|---|
| gemini-3.1-flash-lite | $0.25 | $1.50 |
| cohere-embed-v4 | $0.12 | — |
| gemma-4 | unknown | unknown |

---

## Ingestion Pipeline (`ingest.py`)

### Routing: paper vs. textbook path

Documents ≤ `TEXTBOOK_PAGE_THRESHOLD` (25) pages → **paper path** (full vision on every page).  
Documents > 25 pages → **textbook path** (selective vision).

### Textbook path

1. **Classify all pages** — extract text layer, count math indicators (Greek, operators, `_^` patterns, equation numbers), check for problem/solution sections.
2. **Auto-threshold** — compute a per-document score cutoff that routes `VISION_TARGET` (15%) or `BASE_VISION_TARGET` (25%) of pages to vision. Scanned pages always go to vision; problem-section pages are penalized by +160 score.
3. **Text-layer pages** (`clean=False`) — raw text extracted immediately, stored for lazy upgrade later.
4. **Vision pages** (`clean=True`) — rendered to PNG at `PAGE_DPI` (170 DPI), sent to Gemini with a LaTeX-extraction prompt (`MAX_WORKERS=24` concurrent, `max_tokens=4000`), descriptions stored.
5. **Embed all pages** — descriptions truncated to `MAX_EMBED_CHARS` (4000 chars), batched (`EMBED_BATCH=16`), submitted with `EMBED_STAGGER=2.0s` between batches to avoid Bedrock token burst, embedded with `EMBED_WORKERS=2` concurrent workers, L2-normalized, added to FAISS `IndexFlatIP`.

### Checkpointing

`index.save()` is called after each completed PDF. On restart, files are identified by SHA-256 hash (16-char prefix) and skipped if already indexed.

### On-disk format

```
indexes/<topic>/
  index.faiss       — FAISS IndexFlatIP, 1536-dim float32, L2-normalized
  metadata.json     — pages[], files{}, embedding_model, vision_model, page_dpi, etc.
  base_manifest.json  — only present if this is a base (read-only) index
```

**Page record in `metadata.json`:**
```json
{
  "source":      "Sakurai.pdf",
  "page_num":    42,
  "description": "...",
  "image_path":  "",
  "clean":       true,
  "math_score":  37
}
```

`clean=true` = vision-transcribed (LaTeX-rich). `clean=false` = raw text layer (may be upgraded lazily at query time).

---

## Query Pipeline (`query.py`)

### Step 1 — Routing
A fast `cborg-mini-fast` (Gemma 4 E2B) call decides YES/NO whether to retrieve documents. The prompt is biased toward NO — it requires an *unambiguously* physics/math research question to trigger retrieval. Borderline and conversational messages skip retrieval entirely, avoiding the latency cost of retrieval + large-context reasoning.

### Step 2 — Retrieval
Query is embedded with cohere-embed-v4, L2-normalized, then searched against each selected topic's FAISS index (inner product = cosine similarity after normalization). Results from all topics are merged and sorted by score; top `top_k` (default 5) are returned.

### Step 3 — Lazy upgrade
Retrieved pages with `clean=false`, `math_score >= 5`, and retrieval `score >= 0.50` are re-transcribed with the vision model on the fly (up to `MAX_UPGRADES_PER_QUERY=2`). The upgraded description + new embedding vector are persisted back to disk.

### Step 4 — Reasoning
Retrieved page descriptions (with source/page metadata) are bundled into a single multi-content message and sent to the reasoning model with a physics-expert system prompt. If no pages were retrieved, a shorter conversational system prompt is used instead.

### Return type — `QueryResult`
```python
@dataclass
class QueryResult:
    question:       str
    topics:         list[str]
    model:          str
    top_k:          int
    pages:          list[dict]   # retrieved pages, each has score, topic, source, page_num, clean, _idx
    response_text:  str
    in_tok:         int
    out_tok:        int
    query_cost:     float | None  # None if model not in pricing table
    budget_data:    dict          # {"month": "2026-06", "spent": 0.14, "queries": 7}
    monthly_budget: float
```

### CBorgBudget (fetched separately, after the query)
```python
@dataclass
class CBorgBudget:
    spent:    float | None
    budget:   float | str
    reset_at: str
    raw_keys: list[str]     # non-empty = unexpected API schema
    # properties: .remaining, .reset_str
```

---

## Session Storage (`session_store.py`)

Sessions stored as `sessions/<8-char-uuid>.json`. Listed most-recently-modified first.

```python
{
  "id":       "a3f1bc20",
  "title":    "What is spin?",   # first message[:60], or "New Chat"
  "messages": [
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "...", "meta": { ... }},
  ]
}
```

**`meta` dict** (on every assistant message):
```python
{
  "topics":      ["heisenberg"],
  "model":       "gemma-4",
  "pages":       [{"source": "Sakurai.pdf", "page_num": 42, "score": 0.91,
                   "topic": "heisenberg", "clean": True}],
  "in_tok":      1840,
  "out_tok":     312,
  "cost_str":    "$0.0021",
  "budget_data": {"month": "2026-06", "spent": 0.14, "queries": 7},
  "cborg":       {"spent": 3.21, "budget": "50.0",
                  "remaining": "$46.79", "reset_str": "12d 4h 3m"},
}
```

---

## Configuration (`config.py`)

Key constants relevant to performance:

| Constant | Value | What it controls |
|---|---|---|
| `PAGE_DPI` | 170 | Image resolution for vision; higher = more tokens |
| `EMBED_BATCH` | 16 | Texts per embedding API call |
| `EMBED_WORKERS` | 2 | Concurrent embed batches (Bedrock limit) |
| `MAX_WORKERS` | 24 | Concurrent vision calls |
| `WORKER_STAGGER` | 1.0s | Delay between vision worker batches |
| `EMBED_STAGGER` | 2.0s | Delay between embedding batch submissions |
| `MAX_RETRIES` | 4 | Retry attempts on 429/error |
| `RETRY_BACKOFF` | 10s | Seconds per retry attempt (×attempt number) |
| `MAX_EMBED_CHARS` | 4000 | Description chars sent to embedding |
| `VISION_TARGET` | 15% | Fraction of textbook pages sent to vision |
| `BASE_VISION_TARGET` | 25% | Higher target for base index builds |
| `TEXTBOOK_PAGE_THRESHOLD` | 25 | Pages threshold for paper vs. textbook routing |
| `MATH_DENSITY_THRESHOLD` | 0.08 | Math indicators / total chars (density path) |
| `PROBLEM_SECTION_PENALTY` | 160 | Added to vision threshold for problem pages |
| `UPGRADE_MATH_THRESHOLD` | 5 | Min math_score to bother upgrading a raw page |
| `UPGRADE_MIN_SCORE` | 0.50 | Min retrieval score for lazy upgrade |
| `MAX_UPGRADES_PER_QUERY` | 2 | Cap on concurrent vision upgrades per query |

---

## Utility Scripts

- **`cost_estimator.py`** — runs full page classification (no API calls) then optionally samples N pages through vision to measure actual token usage. `--base` flag uses `BASE_VISION_TARGET`. `SAMPLE_WORKERS=6`.
- **`find_max_workers.py`** — binary-searches `[--lo, --hi]` worker counts by firing simultaneous vision probes; uses `max_retries=0` to catch 429s rather than silently retry them.
- **`build_base.py`** — builds a base index at `BASE_VISION_TARGET`, writes `base_manifest.json`, packages `indexes/<topic>/` into `dist/base-<topic>-v1.tar.gz` for distribution. Base indexes are read-only and skip lazy upgrades.
- **`reset.py`** — deletes `indexes/<topic>/` or all of `indexes/`.

---

## Performance Bottlenecks (known)

1. **Embedding rate limit** — cohere-embed-v4 via Bedrock throttles to ~8,000 tokens/burst. `EMBED_WORKERS=2` is the safe ceiling, giving ~2-3 pg/s sustained.
2. **Vision throughput** — Gemini 3.1 flash lite at 170 DPI produces ~6 image tiles × 258 tokens/tile ≈ 1,500+ input tokens/page. With 24 workers and ~2s/page, expect ~12 vision pages/second at full concurrency.
3. **Lazy upgrade cost** — each query can trigger up to 2 synchronous vision calls before responding; budget ~2-4s of extra latency when upgrades fire.
4. **FAISS rebuild on upgrade** — `persist_upgrades` calls `index.reconstruct_n(0, ntotal)` to read all vectors, modifies in-place, and rebuilds the full index. For large indexes this is O(n) on every upgrade.
