"""
PDF ingestion for the research pipeline.

For each PDF in papers/<topic>/:
  1. Render each page to an image (pymupdf)
  2. Send page image to gemini-3.1-flash-lite (vision) for LaTeX-rich description
  3. Embed description with cohere-embed-v4 (1536-dim)
  4. Store in per-topic FAISS index with metadata

Usage:
    python ingest.py                     # ingest all topics
    python ingest.py heisenberg dmi      # ingest specific topics
    python ingest.py --reindex heisenberg  # force re-ingest even if already indexed
"""
import os
import sys
import json
import hashlib
import base64
import time
from pathlib import Path
import threading

import numpy as np
import faiss
import fitz  # pymupdf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from providers import get_provider

from config import (
    SCRIPT_DIR,
    PAPERS_DIR,
    INDEXES_DIR,
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    MONTHLY_BUDGET,
    VISION_MODEL,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    PAGE_DPI,
    MAX_EMBED_CHARS,
    EMBED_BATCH,
    EMBED_WORKERS,
    EMBED_STAGGER,
    MAX_RETRIES,
    RETRY_BACKOFF,
    MAX_WORKERS,
    WORKER_STAGGER,
    TEXTBOOK_PAGE_THRESHOLD,
    VISION_TARGET,
    BASE_VISION_TARGET,
    MATH_DENSITY_THRESHOLD,
    MATH_DENSITY_MIN_COUNT,
    MIN_PAGE_CHARS,
    PROBLEM_SECTION_PENALTY,
)

provider = get_provider()

# ── Page rendering ──────────────────────────────────────────────────────


def render_page(page: fitz.Page, dpi: int = PAGE_DPI) -> str:
    """Render a PDF page to a base64-encoded PNG data URI."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img_bytes = pix.tobytes("png")
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ── Token / cost accounting ─────────────────────────────────────────────

class _UsageTracker:
    """Thread-safe accumulator for real API token usage during a build."""

    _BUCKETS = ["0", "1-4", "5-9", "10-19", "20-39", "40-79", "80+"]

    @staticmethod
    def _bucket(s: int) -> str:
        if s == 0:
            return "0"
        if s < 5:
            return "1-4"
        if s < 10:
            return "5-9"
        if s < 20:
            return "10-19"
        if s < 40:
            return "20-39"
        if s < 80:
            return "40-79"
        return "80+"

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock if hasattr(self, "_lock") else threading.Lock():
            self.vision_calls = 0
            self.vision_in = 0
            self.vision_out = 0
            self.vision_failed = 0
            self.embed_calls = 0
            self.embed_in = 0
            self.vision_pages = 0
            self.text_pages = 0
            self.skipped_pages = 0
            self.score_buckets = {b: 0 for b in self._BUCKETS}

    def add_vision(self, prompt_tokens, completion_tokens):
        with self._lock:
            self.vision_calls += 1
            self.vision_in += prompt_tokens
            self.vision_out += completion_tokens

    def add_vision_failed(self):
        with self._lock:
            self.vision_failed += 1

    def add_embed(self, prompt_tokens):
        with self._lock:
            self.embed_calls += 1
            self.embed_in += prompt_tokens

    def add_routing(self, vision, text, skipped):
        with self._lock:
            self.vision_pages += vision
            self.text_pages += text
            self.skipped_pages += skipped

    def add_scores(self, scores: list[int]):
        """Record math scores for a batch of pages (called after routing)."""
        with self._lock:
            for s in scores:
                self.score_buckets[self._bucket(s)] += 1

    def print_ingestion_summary(self, vision_target: float = VISION_TARGET):
        """Print the same histogram + routing breakdown shown by --classify-only."""
        with self._lock:
            total = self.vision_pages + self.text_pages + self.skipped_pages
            if total == 0:
                return
            buckets = dict(self.score_buckets)
            v_pages = self.vision_pages
            t_pages = self.text_pages
            sk_pages = self.skipped_pages

        print(f"\n{'=' * 56}")
        print(f"Ingestion summary across {total} textbook pages:")
        for label in self._BUCKETS:
            count = buckets[label]
            bar = "█" * int(40 * count / max(total, 1))
            print(f"  score {label:>6} | {count:>5} {bar}")
        print(f"\nRouting breakdown (target: {vision_target:.0%} vision):")
        print(
            f"  vision-now (upfront cost): {v_pages}  ({v_pages/max(total,1):.1%})")
        print(f"  text-layer (lazy upgrade): {t_pages}")
        print(f"  skipped (blank):           {sk_pages}")
        print(f"{'=' * 56}")

    def snapshot(self):
        with self._lock:
            return dict(
                vision_calls=self.vision_calls,
                vision_in=self.vision_in,
                vision_out=self.vision_out,
                vision_failed=self.vision_failed,
                embed_calls=self.embed_calls,
                embed_in=self.embed_in,
                vision_pages=self.vision_pages,
                text_pages=self.text_pages,
                skipped_pages=self.skipped_pages,
            )


USAGE = _UsageTracker()


# ── Page classification (textbook routing) ──────────────────────────────


# Characters and patterns that signal mathematical content.
_MATH_CHARS = set("=±∓×÷∑∏∫∂∇√∞≈≠≤≥∈∉⊂⊃∪∩→←↔⟨⟩·∝ℏℵ°")
_GREEK = set("αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ")
# Equation-number patterns like (3.14) or (2.7a)
_EQ_NUM_RE = re.compile(r"\(\d+(\.\d+)?[a-z]?\)")
# Superscript/subscript-ish runs common in flattened math, e.g. "x_i" "S^2"
_SUBSUP_RE = re.compile(r"[A-Za-z]\s*[_^]\s*[A-Za-z0-9]")


def extract_page_text(page: fitz.Page) -> str:
    """Pull the embedded text layer from a page (free, local, instant)."""
    try:
        return page.get_text("text") or ""
    except Exception:
        return ""


def count_math_indicators(text: str) -> int:
    """Heuristic count of mathematical-content signals on a page."""
    count = 0
    count += sum(1 for ch in text if ch in _MATH_CHARS)
    count += sum(1 for ch in text if ch in _GREEK)
    count += len(_EQ_NUM_RE.findall(text))
    count += len(_SUBSUP_RE.findall(text))
    return count


# ── Problem/solution section detection (for the demotion rule) ──────────

_PROBLEM_KEYWORDS = ("problem", "exercise", "solution", "answer", "solutions")

# A heading line that names a problem/solution section, optionally numbered:
#   "Problems"  "4.8 Exercises"  "Chapter 3 Solutions"  "Answers to Problems"
_HEADING_PROBLEM_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\s+|chapter\s+\d+\s+)?"
    r"(?:problems?|exercises?|solutions?|answers?(?:\s+to\s+[\w\s]+)?)\s*$",
    re.IGNORECASE,
)
# A normal numbered section/chapter heading (used to END problem-section
# state):
_HEADING_SECTION_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\s+\S|chapter\s+\d+\b)",
    re.IGNORECASE,
)


def _heading_lines(page: fitz.Page, n: int = 6) -> list[str]:
    """First few non-empty lines of a page — candidate headings."""
    text = page.get_text("text") or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[:n]


def compute_problem_pages(doc: fitz.Document) -> set[int]:
    """
    Return the set of 1-indexed page numbers that fall within problem/solution/
    exercise sections, using two signals:
      1. The PDF table of contents (reliable when present).
      2. Inline heading detection, carried forward across pages until a normal
         (non-problem) section heading ends the run.

    Sanity cap: if the inline state machine marks > 40% of the book as problem
    pages, it has almost certainly misfired (e.g. one keyword triggered early
    and chapter headings didn't match the exit regex). In that case, the inline
    results are discarded and only TOC-based ranges are used.
    """
    n = len(doc)

    # 1) TOC-based ranges (reliable)
    toc_problem_pages: set[int] = set()
    try:
        toc = doc.get_toc()
    except Exception:
        toc = []

    for idx, entry in enumerate(toc):
        level, title, start = entry[0], entry[1], entry[2]
        if any(k in title.lower() for k in _PROBLEM_KEYWORDS):
            end = n
            for nxt in toc[idx + 1:]:
                if nxt[0] <= level:
                    end = nxt[2] - 1
                    break
            for p in range(max(start, 1), min(end, n) + 1):
                toc_problem_pages.add(p)

    # 2) Inline heading state machine
    inline_problem_pages: set[int] = set()
    in_problem = False
    for i in range(n):
        heads = _heading_lines(doc[i])
        started = False
        for ln in heads:
            if _HEADING_PROBLEM_RE.match(ln):
                in_problem = True
                started = True
                break
        if not started and in_problem:
            for ln in heads:
                if _HEADING_SECTION_RE.match(ln) and not any(
                    k in ln.lower() for k in _PROBLEM_KEYWORDS
                ):
                    in_problem = False
                    break
        if in_problem:
            inline_problem_pages.add(i + 1)

    # Sanity cap: if inline detection flags > 40% of the book it has misfired.
    # Fall back to TOC-only in that case.
    if len(inline_problem_pages) > n * 0.40:
        return toc_problem_pages

    return toc_problem_pages | inline_problem_pages


_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def score_page(page: fitz.Page) -> tuple[str, int]:
    """
    Lightweight pre-pass: extract text quality and math score without routing.
    Returns ('scanned', 0) for image-only/broken-text pages, or ('ok', score).
    Used to build the score distribution before computing the auto-threshold.
    """
    raw = extract_page_text(page)
    ctrl_count = sum(1 for c in raw if c in "\x00" or _CTRL.match(c))
    cleaned = _CTRL.sub("", raw)
    ctrl_fraction = ctrl_count / max(len(raw), 1)

    if ctrl_fraction > 0.2 or len(cleaned.strip()) < MIN_PAGE_CHARS:
        if page.get_images() or len(raw.strip()) > 0 or page.get_drawings():
            return "scanned", 0
        return "blank", 0

    math_score = count_math_indicators(cleaned)
    return "ok", math_score


def compute_auto_threshold(doc: fitz.Document,
                           problem_pages: set[int],
                           target: float) -> int:
    """
    Find the absolute math-score threshold that routes approximately `target`
    fraction of the document's pages to vision.

    Only non-scanned, non-blank, non-problem pages are used to set the
    threshold — scanned pages always go to vision regardless, and problem
    pages are always demoted. Returns a threshold >= 1 so isolated symbols
    on otherwise-prose pages don't accidentally trigger vision.
    """
    quality: list[tuple[str, int]] = []
    for i, page in enumerate(doc):
        q, score = score_page(page)
        # (quality, score, 1-indexed page_num)
        quality.append((q, score, i + 1))

    n_total = len(doc)
    n_scanned = sum(1 for q, _, _ in quality if q == "scanned")

    # Content pages: non-scanned, non-blank, not in a problem section.
    content_scores = [
        score for q, score, pn in quality
        if q == "ok" and pn not in problem_pages
    ]

    # Target: how many content pages should go to vision?
    target_vision_total = round(target * n_total)
    target_vision_content = max(0, target_vision_total - n_scanned)

    if not content_scores or target_vision_content <= 0:
        return 999_999   # nothing goes to vision

    if target_vision_content >= len(content_scores):
        return 1         # everything goes to vision

    sorted_scores = sorted(content_scores, reverse=True)
    threshold = sorted_scores[target_vision_content - 1]
    # floor of 1 — never route score-0 prose via count path
    return max(threshold, 1)


def classify_page(page: fitz.Page,
                  threshold: int | None = None,
                  threshold_penalty: int = 0) -> tuple[str, str, int]:
    """
    Decide how to process a page.
    Returns (route, text, math_score) where route is one of:
      'skip'   — blank/near-blank, do not index
      'text'   — no/low math, use the free text layer (may be upgraded later)
      'vision' — dense math, send to the vision model now

    threshold: absolute score cutoff for vision (auto-computed per document).
               Falls back to a very high value if None (effectively no vision).
    threshold_penalty: added to both paths for problem-section pages.
    """
    if threshold is None:
        threshold = 999_999

    raw = extract_page_text(page)
    ctrl_count = sum(1 for c in raw if c in "\x00" or _CTRL.match(c))
    cleaned = _CTRL.sub("", raw)
    ctrl_fraction = ctrl_count / max(len(raw), 1)

    if ctrl_fraction > 0.2 or len(cleaned.strip()) < MIN_PAGE_CHARS:
        if page.get_images() or len(raw.strip()) > 0 or page.get_drawings():
            return "vision", cleaned, 0
        return "skip", cleaned, 0

    text = cleaned
    math_score = count_math_indicators(text)
    total_chars = max(len(text), 1)
    density = math_score / total_chars

    # Path 1 — absolute count (catches long, equation-heavy pages)
    if math_score >= threshold + threshold_penalty:
        return "vision", text, math_score

    # Path 2 — density (catches short pages that are almost entirely math)
    density_floor = MATH_DENSITY_MIN_COUNT + threshold_penalty
    if density >= MATH_DENSITY_THRESHOLD and math_score >= density_floor:
        return "vision", text, math_score

    return "text", text, math_score


# ── Page description via vision model ───────────────────────────────────

PAGE_DESCRIPTION_PROMPT = """You are indexing a physics/mathematics textbook or paper page for semantic search.

Extract and output:
- All mathematical expressions in LaTeX ($...$ inline, $$...$$ display)
- Section headings and equation numbers/labels
- Key definitions, theorems, and conceptual statements
- Figure/graph descriptions: axes, curves, key values, and what is shown
- Essential prose connecting the math

Be concise but complete on math and figures. Skip decorative text, page numbers, and headers.
Output only the extracted content with no preamble.
Prioritize mathematical content and figure data over prose; aim for completeness on equations rather than exhaustive narration.
Be as faithful to the notation in the text as possible, do not interpret or change any notation used, transcribe it exactly as it appears."""


def describe_page(data_uri, page_num, filename):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            text, usage = provider.transcribe_image(
                data_uri, PAGE_DESCRIPTION_PROMPT, VISION_MODEL,
                temperature=0.1, max_tokens=4000, timeout=60,
            )
            if not text:
                print(f"\n    [debug] empty response on page {page_num}: "
                      f"finish_reason={usage.get('finish_reason')}")
                raise ValueError("Empty response content from vision model")
            USAGE.add_vision(
                usage.get(
                    "prompt_tokens", 0), usage.get(
                    "completion_tokens", 0))
            if usage.get("finish_reason") == "length":
                print(
                    f"  ! page {page_num}: response truncated at max_tokens — partial saved")
            return text
        except Exception as e:
            msg = str(e)
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                print(
                    f"    ! page {page_num} attempt {attempt}/{MAX_RETRIES}: {msg[:80]} — retrying in {wait}s")
                time.sleep(wait)
            else:
                print(
                    f"    ! page {page_num} failed after {MAX_RETRIES} attempts: {msg[:120]}")
                USAGE.add_vision_failed()   # ---- NEW ----
                return f"[Page {page_num} of {filename} — description failed]"


# ── Embedding ───────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> np.ndarray:
    texts = [(_CTRL.sub("", t)[:MAX_EMBED_CHARS]) or "[no content]" for t in texts]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            embeddings, usage = provider.embed_texts(texts, EMBEDDING_MODEL)
            USAGE.add_embed(usage.get("prompt_tokens", 0))
            vecs = np.array(embeddings, dtype=np.float32)
            if vecs.ndim == 2 and vecs.shape[1] != EMBEDDING_DIM:
                raise SystemExit(
                    f"\nEmbedding dimension mismatch:\n"
                    f"  '{EMBEDDING_MODEL}' returned {vecs.shape[1]}-dim vectors\n"
                    f"  but EMBEDDING_DIM = {EMBEDDING_DIM}\n"
                    f"Fix: set EMBEDDING_DIM = {vecs.shape[1]} in ingest.py and query.py, then re-run.")
            return vecs
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                print(
                    f"    ! embedding attempt {attempt}/{MAX_RETRIES}: {str(e)} — retrying in {wait}s")
                time.sleep(wait)
            else:
                print("=== FULL EMBEDDING ERROR ===")
                print(repr(e))
                print(f"batch size: {len(texts)}")
                for i, t in enumerate(texts):
                    print(f"  item {i}: {len(t)} chars, repr={repr(t[:60])}")
                raise


def embed_all(
        descriptions: list[str],
        sources: list[str] | None = None) -> np.ndarray:
    batch_list = [descriptions[i: i + EMBED_BATCH]
                  for i in range(0, len(descriptions), EMBED_BATCH)]
    results: list[np.ndarray | None] = [None] * len(batch_list)

    def _run(batch_idx: int, batch: list[str]) -> tuple[int, np.ndarray]:
        return batch_idx, embed_texts(batch)

    with tqdm(total=len(descriptions), desc="  embedding", unit="pg", leave=False) as pbar:
        with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as ex:
            futures = {}
            for i, batch in enumerate(batch_list):
                if i > 0:
                    time.sleep(EMBED_STAGGER)
                futures[ex.submit(_run, i, batch)] = (i, batch)
            for fut in as_completed(futures):
                batch_idx, batch = futures[fut]
                try:
                    res_idx, vecs = fut.result()
                    results[res_idx] = vecs
                except Exception:
                    start = batch_idx * EMBED_BATCH
                    for j, d in enumerate(batch):
                        label = sources[start + j] if sources else f"index {start + j}"
                        print(f"    [embed] {label}: {len(d)} chars")
                    raise
                pbar.update(len(batch))

    return np.vstack(results) if results else np.zeros(
        (0, EMBEDDING_DIM), dtype=np.float32)


# ── FAISS index management ──────────────────────────────────────────────

class TopicIndex:
    """FAISS index + sidecar metadata for one topic."""

    def __init__(self, topic: str):
        self.topic = topic
        self.dir = INDEXES_DIR / topic
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.faiss"
        self.meta_path = self.dir / "metadata.json"
        self.read_only = False
        self.lazy_upgrade = True

        # Load existing metadata first (so we can derive embedding_dim)
        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8", errors="replace") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "pages": [],
                "files": {},
            }

        # Ensure required config keys exist
        self.metadata.setdefault("embedding_model", EMBEDDING_MODEL)
        self.metadata.setdefault("embedding_dim", EMBEDDING_DIM)
        self.metadata.setdefault("vision_model", VISION_MODEL)
        self.metadata.setdefault("page_dpi", PAGE_DPI)
        self.metadata.setdefault(
            "llm_provider", os.environ.get(
                "LLM_PROVIDER", "cborg"))

        # Defaults consumed by query.py (only if you changed query to read
        # these)
        self.metadata.setdefault("default_model", DEFAULT_MODEL)
        self.metadata.setdefault("default_top_k", DEFAULT_TOP_K)
        self.metadata.setdefault("monthly_budget", MONTHLY_BUDGET)

        # Now create/load FAISS index using the metadata embedding_dim
        embedding_dim = int(self.metadata["embedding_dim"])

        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            if hasattr(self.index, "d") and self.index.d != embedding_dim:
                raise SystemExit(
                    f"FAISS dim mismatch for topic '{topic}': "
                    f"index.d={self.index.d} vs metadata.embedding_dim={embedding_dim}. Reindex needed.")
        else:
            self.index = faiss.IndexFlatIP(embedding_dim)

    def has_file(self, fname: str, fhash: str) -> bool:
        return self.metadata["files"].get(fname) == fhash

    def add_pages(
            self,
            fname: str,
            fhash: str,
            pages: list[dict],
            vectors: np.ndarray):
        """
        pages: list of {page_num, description, image_path, clean, math_score}
        vectors: (n, EMBEDDING_DIM) float32, one per page
        """
        if self.read_only:
            raise RuntimeError(
                f"Topic '{self.topic}' is a read-only base index; refusing to add pages."
            )
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        for page in pages:
            self.metadata["pages"].append({
                "source": fname,
                "page_num": page["page_num"],
                "description": page["description"],
                "image_path": page.get("image_path", ""),
                "clean": page.get("clean", False),
                "math_score": page.get("math_score", 0),
            })
        self.metadata["files"][fname] = fhash

    def save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    @property
    def total_pages(self) -> int:
        return self.index.ntotal


# ── Per-PDF image cache ─────────────────────────────────────────────────


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:16]


# ── Topic ingestion ─────────────────────────────────────────────────────

def ingest_pdf(
        pdf_path: Path,
        index: TopicIndex,
        force: bool = False,
        vision_target: float | None = None) -> bool:
    """
    Ingest one PDF. Routes to the textbook path (cheap text + selective vision)
    for long documents, or the paper path (full vision) for short ones.
    Returns True if anything was added, False if skipped.
    """
    if index.read_only:
        print(
            f"  · {pdf_path.name} (skipped — '{index.topic}' is a read-only base index)")
        return False
    fhash = file_hash(pdf_path)
    if not force and index.has_file(pdf_path.name, fhash):
        print(f"  · {pdf_path.name} (already indexed)")
        return False

    doc = fitz.open(str(pdf_path))
    num_pages = len(doc)

    if num_pages > TEXTBOOK_PAGE_THRESHOLD:
        print(f"  + {pdf_path.name}  [textbook path — {num_pages} pages]")
        target = vision_target if vision_target is not None else VISION_TARGET
        result = ingest_textbook(
            pdf_path, doc, index, fhash, vision_target=target)
    else:
        print(f"  + {pdf_path.name}  [paper path — {num_pages} pages]")
        result = ingest_paper(pdf_path, doc, index, fhash)

    doc.close()
    return result


def ingest_textbook(pdf_path: Path, doc: fitz.Document, index: TopicIndex,
                    fhash: str, vision_target: float = VISION_TARGET) -> bool:
    """
    Textbook ingestion: auto-computes the score threshold to hit vision_target
    fraction of pages as vision, then classifies each page accordingly.
    Dense-math pages (clean=True) are vision-processed now; all others are
    stored as raw text (clean=False) for lazy upgrade at query time.
    """

    # ── Step 1: classify all pages ───────────────────────────────────────────
    print(f"    classifying pages (target: {vision_target:.0%} vision)...")
    problem_pages = compute_problem_pages(doc)
    threshold = compute_auto_threshold(doc, problem_pages, vision_target)
    print(
        f"    auto-threshold: {threshold} (to hit {vision_target:.0%} of {len(doc)} pages)")

    routes = {}   # page_num -> (route, text, math_score)
    n_demoted = 0
    for i, page in enumerate(doc):
        page_num = i + 1
        penalty = PROBLEM_SECTION_PENALTY if page_num in problem_pages else 0
        route, text, math_score = classify_page(
            page, threshold=threshold, threshold_penalty=penalty)
        if penalty and route == "text" and math_score >= threshold:
            n_demoted += 1
        routes[page_num] = (route, text, math_score)

    n_skip = sum(1 for r, _, _ in routes.values() if r == "skip")
    n_text = sum(1 for r, _, _ in routes.values() if r == "text")
    n_vision = sum(1 for r, _, _ in routes.values() if r == "vision")
    actual_pct = n_vision / max(len(doc), 1)
    print(
        f"    routing: {n_vision} vision-now ({actual_pct:.0%})  |  {n_text} text-layer  |  {n_skip} skipped")
    if problem_pages:
        print(f"    ({len(problem_pages)} pages in problem/solution sections; "
              f"{n_demoted} math pages demoted to text by penalty)")

    # ── Warn if >50% of pages need vision (likely a scanned PDF) ─────────────
    if actual_pct > 0.50:
        print(
            f"\n  ⚠  WARNING: {actual_pct:.0%} of pages in '{pdf_path.name}' need vision.")
        print(f"     This is likely a scanned/image-only PDF.")
        print(
            f"     Vision-processing {n_vision} pages will be slow and will cost ~"
            f"${n_vision * 0.0012:.2f} (estimate).")
        if sys.stdin.isatty():
            answer = input(
                "     Ingest this book anyway? [y/N]: ").strip().lower()
        else:
            answer = "n"
            print("     (non-interactive — skipping automatically)")
        if answer != "y":
            print(f"     Skipping '{pdf_path.name}'.")
            return False

    # ── Track scores for the end-of-run summary ─────────────────────────────
    USAGE.add_routing(n_vision, n_text, n_skip)
    USAGE.add_scores([math_score for _, _, math_score in routes.values()])

    pages_data = []

    # ── Step 2: free text-layer pages (instant, no API; clean=False) ─────────
    for page_num, (route, text, math_score) in routes.items():
        if route == "text":
            pages_data.append({
                "page_num": page_num,
                "description": text.strip(),
                "image_path": "",        # rendered lazily at query time if upgraded
                "clean": False,     # raw text layer — may be upgraded later
                "math_score": math_score,
            })

    # ── Step 3: dense-math pages — render then describe in parallel ──────────
    vision_pages = [pn for pn, (r, _, _) in routes.items() if r == "vision"]

    if vision_pages:
        print(f"    rendering {len(vision_pages)} dense-math pages...")
        mat = fitz.Matrix(PAGE_DPI / 72, PAGE_DPI / 72)
        render_args = []
        for page_num in vision_pages:
            page = doc[page_num - 1]
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            png_bytes = pix.tobytes("png")
            render_args.append((page_num, png_bytes))

        print(
            f"    describing {len(render_args)} pages with {MAX_WORKERS} workers...")

        def describe_one(args):
            page_num, png_bytes = args
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            data_uri = f"data:image/png;base64,{b64}"
            description = describe_page(data_uri, page_num, pdf_path.name)
            return page_num, description

        results = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for i, args in enumerate(render_args):
                if i > 0 and i % MAX_WORKERS == 0:
                    time.sleep(WORKER_STAGGER)
                futures[executor.submit(describe_one, args)] = args[0]
            with tqdm(total=len(futures), desc="    describing", leave=False) as pbar:
                for future in as_completed(futures):
                    page_num, description = future.result()
                    results[page_num] = {
                        "page_num": page_num,
                        "description": description,
                        "image_path": "",
                        "clean": True,
                        "math_score": routes[page_num][2],
                    }
                    pbar.update(1)

        pages_data.extend(results.values())

    # ── Step 4: embed everything ─────────────────────────────────────────────
    pages_data.sort(key=lambda p: p["page_num"])
    if not pages_data:
        print(f"    ! no indexable pages")
        return False

    print(f"    embedding {len(pages_data)} pages...")
    descriptions = [p["description"][:MAX_EMBED_CHARS] for p in pages_data]
    vectors = embed_all(descriptions)

    index.add_pages(pdf_path.name, fhash, pages_data, vectors)
    return True


def ingest_paper(
        pdf_path: Path,
        doc: fitz.Document,
        index: TopicIndex,
        fhash: str) -> bool:
    """
    Paper ingestion: full vision on every page (papers are short and
    figure-heavy, so per-page vision is worth it).
    """

    # Render all pages
    print(f"    rendering pages...")
    mat = fitz.Matrix(PAGE_DPI / 72, PAGE_DPI / 72)
    render_args = []
    for i, page in enumerate(tqdm(doc, desc="    rendering", leave=False)):
        page_num = i + 1
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        png_bytes = pix.tobytes("png")
        render_args.append((page_num, png_bytes))
    USAGE.add_routing(len(render_args), 0, 0)
    print(
        f"    describing {len(render_args)} pages with {MAX_WORKERS} workers...")

    def describe_one(args):
        page_num, png_bytes = args
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        data_uri = f"data:image/png;base64,{b64}"
        description = describe_page(data_uri, page_num, pdf_path.name)
        return page_num, description

    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for i, args in enumerate(render_args):
            if i > 0 and i % MAX_WORKERS == 0:
                time.sleep(WORKER_STAGGER)
            futures[executor.submit(describe_one, args)] = args[0]
        with tqdm(total=len(futures), desc="    describing", leave=False) as pbar:
            for future in as_completed(futures):
                page_num, description = future.result()
                results[page_num] = {
                    "page_num": page_num,
                    "description": description,
                    "image_path": "",
                    "clean": True,
                    "math_score": 0}
                pbar.update(1)

    pages_data = [results[pn] for pn in sorted(results.keys())]

    print(f"    embedding {len(pages_data)} page descriptions...")
    descriptions = [p["description"][:MAX_EMBED_CHARS] for p in pages_data]
    vectors = embed_all(descriptions)

    index.add_pages(pdf_path.name, fhash, pages_data, vectors)
    return True


def ingest_topic(topic_dir: Path, force: bool = False,
                 vision_target: float | None = None):
    """Ingest all PDFs in one topic folder."""
    topic = topic_dir.name
    pdfs = sorted(topic_dir.rglob("*.pdf"))

    if not pdfs:
        print(f"[{topic}] no PDFs found")
        return {}

    print(f"\n[{topic}] {len(pdfs)} PDF(s)")
    index = TopicIndex(topic)
    new_pages = 0

    for pdf in pdfs:
        if ingest_pdf(pdf, index, force=force, vision_target=vision_target):
            new_pages += index.total_pages
            index.save()  # checkpoint after each PDF so a crash only loses the current file

    if new_pages:
        print(f"[{topic}] complete — {index.total_pages:,} total pages indexed")
    else:
        print(f"[{topic}] nothing new")

    return USAGE.snapshot()


# ── Classify-only dry run (threshold calibration) ───────────────────────

def inspect_band(
        topic_dirs: list[Path],
        lo: int,
        hi: int,
        max_pages: int = 20):
    """
    Render pages whose math score falls in [lo, hi] to an inspection folder
    so you can visually eyeball whether the threshold is slicing through prose
    (safe) or dense math (unsafe). Filenames embed the score for easy sorting.
    """
    out_dir = SCRIPT_DIR / "inspect_band"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear previous run
    for old in out_dir.glob("*.png"):
        old.unlink()

    print(f"\nRendering pages with math score in [{lo}, {hi}] to: {out_dir}")
    print("Filenames are: score<NNN>__<book>__p<page>.png  (sort by name to group by score)")
    print("=" * 70)

    shown = 0
    for topic_dir in topic_dirs:
        for pdf in sorted(topic_dir.rglob("*.pdf")):
            doc = fitz.open(str(pdf))
            if len(doc) <= TEXTBOOK_PAGE_THRESHOLD:
                doc.close()
                continue
            problem_pages = compute_problem_pages(doc)
            for i, page in enumerate(doc):
                if shown >= max_pages:
                    doc.close()
                    print(
                        f"\n(stopped at {max_pages} pages — raise max_pages to see more)")
                    print(f"Rendered {shown} pages to {out_dir}")
                    return
                _, _, score = classify_page(page)
                if lo <= score <= hi:
                    safe_book = pdf.stem[:40].replace(" ", "_")
                    prob_tag = "PROB__" if (i + 1) in problem_pages else ""
                    fname = f"score{score:04d}__{prob_tag}{safe_book}__p{i+1:04d}.png"
                    mat = fitz.Matrix(PAGE_DPI / 72, PAGE_DPI / 72)
                    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                    pix.save(str(out_dir / fname))
                    shown += 1
            doc.close()

    if shown == 0:
        print("  (no pages in this band)")
    else:
        print(f"\nRendered {shown} pages to {out_dir}")
        print(
            "Open that folder and flip through — sorted by filename, lowest scores first.")


def classify_only(topic_dirs: list[Path]):
    """
    Run classification on all textbook PDFs WITHOUT processing anything.
    For each book, shows the auto-computed threshold for both VISION_TARGET
    and BASE_VISION_TARGET, plus the full score distribution and routing breakdown.
    """
    score_buckets = {
        "0": 0,
        "1-4": 0,
        "5-9": 0,
        "10-19": 0,
        "20-39": 0,
        "40-79": 0,
        "80+": 0,
    }

    total_pages = 0
    would_vision = would_text = would_skip = 0
    demoted = 0

    for topic_dir in topic_dirs:
        for pdf in sorted(topic_dir.rglob("*.pdf")):
            doc = fitz.open(str(pdf))
            if len(doc) <= TEXTBOOK_PAGE_THRESHOLD:
                doc.close()
                continue
            problem_pages = compute_problem_pages(doc)

            # Compute auto-thresholds for both targets
            t_ingest = compute_auto_threshold(
                doc, problem_pages, VISION_TARGET)
            t_base = compute_auto_threshold(
                doc, problem_pages, BASE_VISION_TARGET)

            print(
                f"\n  {pdf.name} ({len(doc)} pages, {len(problem_pages)} in problem sections)")
            print(
                f"    auto-threshold: {t_ingest} for ingest ({VISION_TARGET:.0%} vision)  |  "
                f"{t_base} for base build ({BASE_VISION_TARGET:.0%} vision)")

            sample_high = []
            density_path_hits = 0
            scanned_pages = 0
            for i, page in enumerate(doc):
                penalty = PROBLEM_SECTION_PENALTY if (
                    i + 1) in problem_pages else 0
                route, text, score = classify_page(
                    page, threshold=t_ingest, threshold_penalty=penalty)
                total_pages += 1
                score_buckets[_UsageTracker._bucket(score)] += 1
                if route == "vision":
                    would_vision += 1
                elif route == "text":
                    would_text += 1
                else:
                    would_skip += 1
                if penalty and route == "text" and score >= t_ingest:
                    demoted += 1
                if route == "vision" and score == 0:
                    scanned_pages += 1
                elif route == "vision" and score < t_ingest + penalty:
                    density_path_hits += 1
                if route == "vision" and score > 0 and len(sample_high) < 5:
                    total_chars = max(len(text), 1)
                    d = score / total_chars
                    sample_high.append((i + 1, score, f"{d:.3f}"))
            if sample_high:
                preview = ", ".join(
                    f"p{pn}=count{s}/density{d}" for pn, s, d in sample_high)
                print(f"    sample vision pages: {preview}")
            if scanned_pages:
                print(
                    f"    ! {scanned_pages} scanned/image-only pages → all going to vision (expensive)")
            if density_path_hits:
                print(
                    f"    density-path hits (short dense pages): {density_path_hits}")
            doc.close()

    print(f"\n{'=' * 56}")
    print(f"Math-score distribution across {total_pages} textbook pages:")
    for label, count in score_buckets.items():
        bar = "█" * int(40 * count / max(total_pages, 1))
        print(f"  score {label:>6} | {count:>5} {bar}")
    print(f"\nRouting breakdown (at {VISION_TARGET:.0%} ingest target):")
    print(f"  vision-now (upfront cost): {would_vision}")
    print(f"  demoted by problem penalty: {demoted}")
    print(f"  text-layer (lazy upgrade): {would_text}")
    print(f"  skipped (blank):           {would_skip}")
    print(
        f"  overall vision percent:    {round((would_vision / max(total_pages,1)), 3)}")
    print(f"\nThresholds are auto-computed per book from the distribution.")
    print(f"Adjust VISION_TARGET / BASE_VISION_TARGET to shift the split.")
    print(f"{'=' * 56}")


# ── Main ────────────────────────────────────────────────────────────────

def _print_ingestion_cost():
    snap = USAGE.snapshot()
    vision_cost = provider.estimate_cost(
        VISION_MODEL, snap["vision_in"], snap["vision_out"])
    embed_cost = provider.estimate_cost(EMBEDDING_MODEL, snap["embed_in"], 0)

    def cost_str(cost, model):
        return f"${cost:.4f}" if cost is not None else f"(unknown — no pricing for '{model}')"

    print(f"\n── Ingestion cost ────────────────────────────────────────")
    print(f"  Vision  ({VISION_MODEL})")
    print(
        f"    tokens:  {snap['vision_in']:,} in / {snap['vision_out']:,} out  ({snap['vision_calls']} calls, {snap['vision_failed']} failed)")
    print(f"    cost:    {cost_str(vision_cost, VISION_MODEL)}")
    print(f"  Embed   ({EMBEDDING_MODEL})")
    print(
        f"    tokens:  {snap['embed_in']:,} in  ({snap['embed_calls']} calls)")
    print(f"    cost:    {cost_str(embed_cost, EMBEDDING_MODEL)}")
    total = (vision_cost or 0.0) + (embed_cost or 0.0)
    if vision_cost is not None or embed_cost is not None:
        print(f"  Total:   ${total:.4f}")
    print(f"──────────────────────────────────────────────────────────")


def main(argv: list[str]):
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)

    force = "--reindex" in argv
    classify_dry = "--classify-only" in argv
    inspect = "--inspect-band" in argv
    base_build = "--base" in argv
    topics_requested = [
        a for a in argv if not a.startswith("--") and not a.isdigit()]

    vision_target = BASE_VISION_TARGET if base_build else VISION_TARGET

    if topics_requested:
        topic_dirs = []
        for t in topics_requested:
            td = PAPERS_DIR / t
            if not td.is_dir():
                print(f"! topic folder not found: {td}")
                sys.exit(1)
            topic_dirs.append(td)
    else:
        topic_dirs = sorted(p for p in PAPERS_DIR.iterdir() if p.is_dir())
        if not topic_dirs:
            print(f"No topic folders found under {PAPERS_DIR}")
            print("Create e.g. papers/heisenberg/ and drop PDFs in it.")
            sys.exit(0)

    if inspect:
        # parse the two integers after --inspect-band
        nums = [int(a) for a in argv if a.isdigit()]
        lo, hi = (nums + [50, 70])[:2] if len(nums) >= 2 else (50, 70)
        inspect_band(topic_dirs, lo, hi)
        return

    if classify_dry:
        classify_only(topic_dirs)
        return

    if base_build:
        print(f"[base build] vision target: {BASE_VISION_TARGET:.0%}")

    total_new = 0
    for td in topic_dirs:
        snap = ingest_topic(td, force=force, vision_target=vision_target)
        total_new += snap.get("vision_pages", 0) + snap.get("text_pages", 0)

    USAGE.print_ingestion_summary()
    _print_ingestion_cost()
    print(
        f"\nDone. {total_new} new page(s) indexed across {len(topic_dirs)} topic(s).")


if __name__ == "__main__":
    main(sys.argv[1:])
