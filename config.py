"""
Central configuration for the RAG pipeline.

All model names, directory paths, and tuning parameters live here.
ingest.py, query.py, and utility scripts import from this module.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PAPERS_DIR = SCRIPT_DIR / "curr_resources"
INDEXES_DIR = SCRIPT_DIR / "indexes"
BUDGET_FILE = SCRIPT_DIR / "budget.json"

# ── API keys ────────────────────────────────────────────────────────────────────
CBORG_API_KEY = os.environ.get("CBORG_API_KEY")
if not CBORG_API_KEY:
    raise SystemExit("CBORG_API_KEY env var is not set.")

# ── Models ─────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "gemma-4"                # LLM for reasoning
ROUTING_MODEL = "cborg-mini-fast"        # fast classifier for query routing
VISION_MODEL = "gemini-3.1-flash-lite"   # vision model for math-page transcription
EMBEDDING_MODEL = "cohere-embed-v4"      # embedding model
EMBEDDING_DIM = 1536                     # cohere-embed-v4 output dimension

# ── Query settings ─────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 5
MONTHLY_BUDGET = 50.00
UPGRADE_MATH_THRESHOLD = 5     # min math_score to bother upgrading a raw page
UPGRADE_MIN_SCORE = 0.50  # min retrieval score to qualify for lazy upgrade
MAX_UPGRADES_PER_QUERY = 2     # cap concurrent vision calls per query

# ── Ingestion ──────────────────────────────────────────────────────────────────
PAGE_DPI = 170    # lower = smaller images = faster transfer
MAX_EMBED_CHARS = 4000   # trim descriptions before embedding (cohere-embed-v4 effective limit)
EMBED_BATCH = 16     # texts per embedding API call
EMBED_WORKERS = 2      # concurrent embedding batch workers
EMBED_STAGGER = 2.0    # seconds between batch submissions to avoid token burst
MAX_RETRIES = 4      # embedding retry attempts on failure
RETRY_BACKOFF = 10     # seconds between retries
MAX_WORKERS = 24     # concurrent vision workers
WORKER_STAGGER = 1.0    # seconds between worker starts to avoid burst 429s

# ── Textbook routing ──────────────────────────────────────────────────────────
TEXTBOOK_PAGE_THRESHOLD = 25    # docs longer than this use the textbook path
VISION_TARGET = 0.15  # target fraction of textbook pages sent to vision
BASE_VISION_TARGET = 0.25  # higher target for building shared base indexes

MATH_DENSITY_THRESHOLD = 0.08  # math indicators / total chars
MATH_DENSITY_MIN_COUNT = 20    # minimum absolute indicator count for density path
MIN_PAGE_CHARS = 100   # pages below this char count are skipped
PROBLEM_SECTION_PENALTY = 160   # score penalty for problem/solution section pages
