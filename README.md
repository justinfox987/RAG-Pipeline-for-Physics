# RAG Pipeline for Physics

A retrieval-augmented generation (RAG) tool for querying a personal physics research library. Ask a question; the pipeline retrieves the most relevant pages from your indexed documents, runs an LLM reasoning step over them, and returns a cited, LaTeX-typeset response. Includes a full-featured web GUI with persistent chat sessions, a library browser, and in-app document ingestion.

Built for [CBorg](https://cborg.lbl.gov) (LBL's hosted LLM/embedding API), which exposes an OpenAI-compatible endpoint. The provider layer is abstracted, so adapting to another OpenAI-compatible provider requires only editing `providers/`.

<span style='color: red;'>
WARNING: Significant portions of this code were written using AI. I've checked through the vast majority of it, but I may have missed bugs or nonsense code. If you think something is nonsense, it probably is.
</span>

---

## Features

- **Math-aware ingestion** — textbooks are routed page-by-page: dense math pages go through a vision model for LaTeX-rich transcription; prose pages use the free text layer and are upgraded lazily at query time
- **Multi-topic retrieval** — query across one or more indexed topic collections simultaneously
- **Query routing** — a fast classifier decides whether a message warrants retrieval; conversational messages get a direct response without pulling random pages
- **Persistent chat sessions** — full ChatGPT-style session history saved to disk, accessible from the sidebar
- **Library browser** — inspect indexed topics, files, and page-level vision/raw ratios
- **In-app ingestion** — upload PDFs and ingest directly from the web UI
- **Cost tracking** — per-query token counts, estimated cost, local month-to-date spend, and live CBorg budget display

---

## Requirements

- Python 3.11+
- A [CBorg](https://cborg.lbl.gov) API key (LBL affiliation required)
- The following packages (install via pip):

```
pip install streamlit openai numpy faiss-cpu pymupdf tqdm
```

---

## Setup

**1. Set your API key**

```bash
export CBORG_API_KEY="your-key-here"
```

Add this to your shell profile (`.bashrc`, `.zshrc`, etc.) to persist it.

**2. Organize your PDFs**

Place PDFs into topic folders under `papers/`:

```
papers/
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

**4. Run the web app**

```bash
streamlit run app.py
```

---

## Web App

Three pages accessible from the sidebar:

| Page | Description |
|------|-------------|
| **App** (Chat) | Chat interface with persistent sessions. Topic, model, and top-k controls in the sidebar. Each response includes a collapsible Details panel with retrieved pages, token counts, cost, and CBorg budget. |
| **Library** | Browse all indexed topics and their files. Shows total pages, vision-transcribed vs. raw-text counts per file. |
| **Ingest** | Upload PDFs and ingest them into a topic without leaving the browser. |

Chat sessions are saved as JSON files in `sessions/` and persist across server restarts.

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
├── app.py                  # Streamlit entry point (Chat page)
├── query.py                # Query pipeline: routing, retrieval, reasoning
├── ingest.py               # Ingestion pipeline: vision routing, embedding, indexing
├── session_store.py        # Chat session persistence
├── build_base.py           # Builds shared base indexes
├── reset.py                # Utility to reset local spend tracking
├── providers/
│   ├── __init__.py         # Provider factory (reads LLM_PROVIDER env var)
│   ├── cborg_provider.py   # CBorg/OpenAI-compatible provider implementation
│   └── base.py             # Provider protocol definition
├── pages/
│   ├── 1_Library.py        # Library browser page
│   └── 2_Ingest.py         # Document upload/ingest page
├── papers/                 # Your source PDFs (gitignored)
└── indexes/                # Generated FAISS indexes (gitignored)
```

---

## Pipeline Overview

```
PDF
 └─ ingest.py
     ├─ Short doc  → vision transcription (every page)
     └─ Textbook   → classify pages by math density
                      ├─ dense math  → vision transcription (upfront)
                      └─ prose       → text layer (lazy upgrade at query time)
                      └─ embed with Cohere embed-v4 → FAISS index

Query
 └─ query.py / app.py
     ├─ Route: does this need retrieval? (fast classifier)
     │   ├─ No  → direct LLM response
     │   └─ Yes → retrieve top-k pages from FAISS
     │             └─ lazy upgrade: vision-transcribe raw math pages on the fly
     └─ reason() → LLM response with page citations
```

---

## Configuration

Key constants at the top of `query.py` and `ingest.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `DEFAULT_MODEL` | `gemma-4` | LLM used for reasoning |
| `ROUTING_MODEL` | `cborg-mini-fast` | Fast model for query classification |
| `EMBEDDING_MODEL` | `cohere-embed-v4` | Embedding model for indexing and retrieval |
| `EMBEDDING_DIM` | `1024` | Must match the embedding model's output dimension |
| `DEFAULT_TOP_K` | `5` | Pages retrieved per query |
| `VISION_TARGET` | `15%` | Target fraction of textbook pages sent to vision at ingest |

---

## Notes

- Re-ingesting is only necessary if you change `EMBEDDING_MODEL` (vectors are model-specific) or want to apply updated vision transcriptions. Page descriptions are cached in `metadata.json`, so re-embedding from cached descriptions is fast and does not require re-running vision.
- The `LLM_PROVIDER` environment variable selects the provider (default: `cborg`). A different provider can be added by implementing the protocol in `providers/base.py`.
