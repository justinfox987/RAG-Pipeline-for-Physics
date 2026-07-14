# RAG Pipeline for Physics (v3)

A retrieval-augmented generation (RAG) tool for querying a personal physics research library. Ask a question; the pipeline retrieves the most relevant pages from your indexed documents, runs an LLM reasoning step over them, and returns a cited, LaTeX-typeset response.

Built for [CBorg](https://cborg.lbl.gov) (LBL's hosted LLM/embedding API), which exposes an OpenAI-compatible endpoint. The provider layer is abstracted, so adapting to another OpenAI-compatible provider requires only editing `providers/`.

> **v3** replaces the Streamlit UI with a [Gradio 6](https://gradio.app) interface: native streaming, proper LaTeX rendering (KaTeX), and clipboard image paste built in. The Streamlit app (`app.py`) is legacy and slated for removal.

<span style='color: red;'>
WARNING: Significant portions of this code were written using AI. I've checked through the vast majority of it, but I may have missed bugs or nonsense code. If you think something is nonsense, it probably is.
</span>

---

## Features

- **Math-aware ingestion** — textbooks are routed page-by-page: dense math pages go through a vision model for LaTeX-rich transcription; prose pages use the free text layer and are upgraded lazily at query time
- **Multi-topic retrieval** — query across one or more indexed topic collections simultaneously
- **Query routing** — a fast classifier decides whether a message warrants retrieval; conversational messages get a direct response without pulling random pages
- **Streaming responses** — assistant replies stream token-by-token in the Gradio UI
- **Native LaTeX rendering** — KaTeX renders `$...$`, `$$...$$`, `\(...\)`, and `\[...\]` inline in the chat
- **Image input** — attach images via the file button or paste from clipboard with Ctrl+V; images are sent to the model and used to extract a retrieval query when no text is provided
- **Persistent chat sessions** — full ChatGPT-style session history saved to disk, accessible from the sidebar
- **Library browser** — inspect indexed topics, files, and page-level vision/raw ratios
- **In-app ingestion** — upload PDFs and ingest directly from the web UI without leaving the browser
- **Cost tracking** — per-query token counts, estimated cost, local month-to-date spend, and live CBorg budget display

---

## Requirements

- Python 3.11+
- A [CBorg](https://cborg.lbl.gov) API key (LBL affiliation required)

```bash
pip install gradio openai numpy faiss-cpu pymupdf tqdm streamlit
```

---

## Setup

**1. Set your API key**

```bash
export CBORG_API_KEY="your-key-here"
```

Add to your shell profile (`.bashrc`, `.zshrc`, etc.) to persist it.

**2. Organize your PDFs**

Place PDFs into topic folders under `curr_resources/`:

```
curr_resources/
  heisenberg/
    Sandratskii2017.pdf
    HeisenbergModel.pdf
  dmi/
    Dzyaloshinsky1958.pdf
```

Each folder becomes a separate searchable topic.

**3. Ingest**

```bash
# Ingest all topics
python ingest.py

# Ingest specific topics
python ingest.py heisenberg dmi

# Force re-ingest already-indexed files
python ingest.py --reindex heisenberg
```

Ingestion renders each page, routes it (vision for math-heavy pages, text layer for prose), embeds descriptions with Cohere embed-v4, and stores everything in a per-topic FAISS index under `indexes/`.

**4. Run the Gradio web app**

```bash
python app_gradio.py
```

Opens at `http://localhost:7860`.

---

## Web App (Gradio)

Three tabs:

| Tab | Description |
|-----|-------------|
| **💬 Chat** | Streaming chat interface. Attach images via file button or paste with Ctrl+V. The sidebar holds session history, topic multi-select, model name, and top-k. Each response includes a collapsible Details panel with retrieved pages, scores, token counts, cost, and CBorg budget. |
| **📚 Library** | Browse all indexed topics and their files. Shows total pages, vision-transcribed vs. raw-text counts per file. |
| **⬆ Ingest** | Upload PDFs and ingest them into a new or existing topic. |

Chat sessions are stored as JSON files in `sessions/` and persist across server restarts.

---

## CLI

```bash
python query.py "What is the Dzyaloshinskii-Moriya interaction?"
python query.py "Derive the spin-wave dispersion" --model gpt-5.1 --top-k 8
python query.py "What does Sandratskii say about noncollinear magnetism?" --topic heisenberg
```

Output is printed to the terminal and saved to `last_response.md`.

---

## Project Structure

```
.
├── app_gradio.py           # Gradio entry point (primary UI — v3)
├── app.py                  # Streamlit entry point (legacy; slated for removal)
├── query.py                # Query pipeline: routing, retrieval, reasoning, streaming
├── ingest.py               # Ingestion pipeline: vision routing, embedding, indexing
├── session_store.py        # Chat session persistence
├── config.py               # All model names, paths, and tuning parameters
├── build_base.py           # Builds shared base indexes
├── reset.py                # Resets local spend tracking
├── providers/
│   ├── __init__.py         # Provider factory (reads LLM_PROVIDER env var)
│   ├── cborg_provider.py   # CBorg/OpenAI-compatible provider (includes streaming)
│   └── base.py             # Provider protocol definition
├── pages/
│   ├── 1_Library.py        # Streamlit library page (legacy)
│   └── 2_Ingest.py         # Streamlit ingest page (legacy)
├── curr_resources/         # Source PDFs (gitignored)
├── indexes/                # Generated FAISS indexes (gitignored)
└── sessions/               # Saved chat sessions (gitignored)
```

---

## Pipeline Overview

```
PDF
 └─ ingest.py
     ├─ Short doc  → vision transcription (every page)
     └─ Textbook   → classify pages by math density
                      ├─ dense math  → vision transcription (upfront)
                      └─ prose       → text layer + lazy upgrade at query time
                      └─ embed with Cohere embed-v4 → FAISS IndexFlatIP

Query
 └─ query.py / app_gradio.py
     ├─ Image attached + short text → OCR image for retrieval query
     ├─ Route: does this need retrieval? (fast LLM classifier)
     │   ├─ No  → direct LLM response
     │   └─ Yes → cosine search in FAISS (per-topic min-score threshold)
     │             └─ lazy upgrade: vision-transcribe raw math pages on the fly
     └─ stream_reason_generator() → streaming LLM response with citations
```

---

## Configuration

All tuning parameters live in `config.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `DEFAULT_MODEL` | `gemini-3.1-pro` | LLM used for reasoning |
| `ROUTING_MODEL` | `cborg-mini-fast` | Fast model for query classification |
| `VISION_MODEL` | `cborg-ocr-fast` | Vision model for page transcription |
| `EMBEDDING_MODEL` | `cohere-embed-v4` | Embedding model |
| `EMBEDDING_DIM` | `1536` | Must match the embedding model output dimension |
| `DEFAULT_TOP_K` | `5` | Pages retrieved per query |
| `RETRIEVAL_MIN_SCORE` | `0.35` | Cosine similarity threshold for retrieval |
| `MONTHLY_BUDGET` | `50.00` | Local spend tracking limit (USD) |

Per-topic overrides (e.g. `retrieval_min_score`) can be set in `indexes/<topic>/metadata.json`.

---

## Notes

- Re-ingesting is only needed if you change `EMBEDDING_MODEL` (vectors are model-specific) or want updated vision transcriptions. Page descriptions are cached in `metadata.json`, so re-embedding from cache is fast and does not rerun vision.
- The `LLM_PROVIDER` environment variable selects the provider (default: `cborg`). Add a new provider by implementing the protocol in `providers/base.py`.
- Session images are stored as base64 data URIs in the session JSON. In the Gradio UI, images from the current session render inline; images from reloaded sessions show as a text note.
