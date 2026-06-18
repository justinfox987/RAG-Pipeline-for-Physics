"""
Research query pipeline.

Usage:
    python query.py "<question>"
    python query.py "<question>" --model gpt-5.4-pro
    python query.py "<question>" --top-k 8
    python query.py "<question>" --topic heisenberg dmi
"""
import sys
import json
import base64
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import faiss
import fitz
from providers import get_provider

from config import (
    SCRIPT_DIR, PAPERS_DIR, INDEXES_DIR, BUDGET_FILE,
    DEFAULT_MODEL, DEFAULT_TOP_K, ROUTING_MODEL,
    EMBEDDING_MODEL, EMBEDDING_DIM,
    UPGRADE_MATH_THRESHOLD, UPGRADE_MIN_SCORE, MAX_UPGRADES_PER_QUERY,
)


def load_topic_metadata(topic: str) -> dict:
    p = INDEXES_DIR / topic / "metadata.json"
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


provider = get_provider()


TRANSCRIPTION_PROMPT = """You are indexing a physics/mathematics textbook or paper page for semantic search.

Extract and output:
- All mathematical expressions in LaTeX ($...$ inline, $$...$$ display)
- Section headings and equation numbers/labels
- Key definitions, theorems, and conceptual statements
- Figure/graph descriptions: axes, curves, key values, and what is shown
- Essential prose connecting the math

Be concise but complete on math and figures. Skip decorative text, page numbers, and headers.
Output only the extracted content with no preamble."""


# ── Budget tracking ────────────────────────────────────────────────────────────

def record_spend(cost: float):
    """Track this query's cost locally AND show real CBorg spend if available."""
    from datetime import datetime
    if BUDGET_FILE.exists():
        with open(BUDGET_FILE) as f:
            data = json.load(f)
    else:
        data = {"month": "", "spent": 0.0, "queries": 0}
    current_month = datetime.now().strftime("%Y-%m")
    if data["month"] != current_month:
        data = {"month": current_month, "spent": 0.0, "queries": 0}
    data["spent"] += cost
    data["queries"] += 1
    with open(BUDGET_FILE, "w") as f:
        json.dump(data, f, indent=2)
    return data

# ── Embedding ──────────────────────────────────────────────────────────────────


def embed_query(query: str, embedding_model: str) -> np.ndarray:
    embeddings, _ = provider.embed_texts([query], embedding_model)
    vec = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(vec)
    return vec


def embed_texts(texts: list[str]) -> np.ndarray:
    embeddings, _ = provider.embed_texts(texts, EMBEDDING_MODEL)
    vecs = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(vecs)
    return vecs

# ── Lazy vision upgrade ────────────────────────────────────────────────────────


def is_base_topic(topic: str) -> bool:
    return (INDEXES_DIR / topic / "base_manifest.json").exists()


def transcribe_page(pdf_path, page_num, *, vision_model, page_dpi):
    doc = fitz.open(str(pdf_path))
    page = doc[page_num - 1]
    mat = fitz.Matrix(page_dpi / 72, page_dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img_bytes = pix.tobytes("png")
    doc.close()
    data_uri = f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
    text, _ = provider.transcribe_image(
        data_uri, TRANSCRIPTION_PROMPT, vision_model,
        temperature=0.1, max_tokens=4000, timeout=60,
    )
    return text


def persist_upgrades(topic: str, updates: list[tuple[int, str]]):
    """Batch-update descriptions + vectors for upgraded pages in one FAISS rebuild."""
    index_dir = INDEXES_DIR / topic
    index_path = index_dir / "index.faiss"
    meta_path = index_dir / "metadata.json"

    index = faiss.read_index(str(index_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    all_vecs = index.reconstruct_n(0, index.ntotal)
    new_vecs = embed_texts([desc for _, desc in updates])

    for (page_idx, new_desc), vec in zip(updates, new_vecs):
        metadata["pages"][page_idx]["description"] = new_desc
        metadata["pages"][page_idx]["clean"] = True
        all_vecs[page_idx] = vec

    new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    new_index.add(all_vecs)
    faiss.write_index(new_index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def upgrade_raw_math_pages(pages: list[dict]) -> int:
    """
    Upgrade retrieved raw math pages to clean vision transcriptions.
    Uses per-topic metadata.json for vision_model / page_dpi.
    """
    candidates = [
        p for p in pages
        if not p.get("clean", True)
        and not is_base_topic(p["topic"])
        and p.get("math_score", 0) >= UPGRADE_MATH_THRESHOLD
        and p.get("score", 0.0) >= UPGRADE_MIN_SCORE
    ]

    candidates.sort(key=lambda p: p["score"], reverse=True)
    candidates = candidates[:MAX_UPGRADES_PER_QUERY]
    if not candidates:
        return 0

    # ---- NEW: load per-topic metadata (only for topics we need) ----
    topics_needed = sorted({p["topic"] for p in candidates})
    metadata_by_topic: dict[str, dict] = {}
    for t in topics_needed:
        mp = INDEXES_DIR / t / "metadata.json"
        if not mp.exists():
            raise SystemExit(
                f"Missing metadata.json for topic '{t}' (needed for upgrades).")
        with open(mp, "r", encoding="utf-8", errors="replace") as f:
            metadata_by_topic[t] = json.load(f)

    def resolve_pdf(page: dict) -> Path | None:
        direct = PAPERS_DIR / page["topic"] / page["source"]
        if direct.exists():
            return direct
        matches = list((PAPERS_DIR / page["topic"]).rglob(page["source"]))
        return matches[0] if matches else None

    def upgrade_one(page: dict) -> tuple[dict, str] | None:
        pdf_path = resolve_pdf(page)
        if pdf_path is None:
            print(f"  ! PDF not found for {page['source']} — skipping upgrade")
            return None

        meta = metadata_by_topic[page["topic"]]
        vision_model = meta["vision_model"]
        page_dpi = int(meta.get("page_dpi", 120))

        try:
            # IMPORTANT: update transcribe_page signature accordingly (see note below)
            desc = transcribe_page(
                pdf_path,
                page["page_num"],
                vision_model=vision_model,
                page_dpi=page_dpi,
            )
            if not desc:
                return None
            return page, desc
        except Exception as e:
            print(
                f"  ! upgrade failed for p.{page['page_num']}: {str(e)[:80]}")
            return None

    print(f"  upgrading {len(candidates)} raw page(s) in parallel...")
    pending: dict[str, list[tuple[int, str]]] = {}

    with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
        futures = {executor.submit(upgrade_one, p): p for p in candidates}
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            page, desc = result
            page["description"] = desc
            page["clean"] = True
            pending.setdefault(page["topic"], []).append((page["_idx"], desc))
            print(f"    ✓ upgraded {page['source']} p.{page['page_num']}")

    upgraded = 0
    for topic, updates in pending.items():
        try:
            persist_upgrades(topic, updates)
            upgraded += len(updates)
        except Exception as e:
            print(
                f"  ! failed to persist upgrades for '{topic}': {str(e)[:80]}")

    return upgraded

# ── Retrieval ──────────────────────────────────────────────────────────────────


def get_all_topics() -> list[str]:
    if not INDEXES_DIR.exists():
        return []
    return sorted(
        p.name for p in INDEXES_DIR.iterdir()
        if p.is_dir() and (p / "index.faiss").exists()
    )


def retrieve_pages(topics: list[str], query: str, top_k: int,
                   query_vec: np.ndarray | None = None) -> list[dict]:
    all_pages: list[dict] = []

    # Load metadata for each selected topic (so we can derive embedding model/dim)
    metadata_by_topic: dict[str, dict] = {}
    embedding_models: set[str] = set()
    embedding_dims: set[int] = set()

    for topic in topics:
        meta_path = INDEXES_DIR / topic / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        metadata_by_topic[topic] = metadata

        if metadata.get("embedding_model") is not None:
            embedding_models.add(metadata["embedding_model"])
        if metadata.get("embedding_dim") is not None:
            embedding_dims.add(int(metadata["embedding_dim"]))

    if not metadata_by_topic:
        return []

    if len(embedding_models) > 1:
        raise SystemExit(
            f"Embedding model mismatch across selected topics: {embedding_models}")
    if len(embedding_dims) > 1:
        raise SystemExit(
            f"Embedding dim mismatch across selected topics: {embedding_dims}")

    if len(embedding_models) != 1:
        raise SystemExit(
            "No embedding_model found in metadata for the selected topics.")
    if len(embedding_dims) != 1:
        raise SystemExit(
            "No embedding_dim found in metadata for the selected topics.")

    effective_embedding_model = next(iter(embedding_models))
    effective_embedding_dim = next(iter(embedding_dims))

    # Embed query once using the shared effective embedding model (skip if pre-computed)
    if query_vec is None:
        query_vec = embed_query(query, effective_embedding_model)

    for topic in topics:
        if topic not in metadata_by_topic:
            print(f"  ! no metadata for topic '{topic}', skipping")
            continue

        index_path = INDEXES_DIR / topic / "index.faiss"
        if not index_path.exists():
            print(f"  ! no index for topic '{topic}', skipping")
            continue

        index = faiss.read_index(str(index_path))
        metadata = metadata_by_topic[topic]

        # Validate metadata consistency (older indexes may not have these keys)
        meta_dim = metadata.get("embedding_dim")
        meta_model = metadata.get("embedding_model")

        if meta_dim is not None and int(meta_dim) != effective_embedding_dim:
            raise SystemExit(
                f"Embedding dim mismatch for topic '{topic}': "
                f"index metadata expects {meta_dim}, query uses {effective_embedding_dim}. Reindex needed."
            )

        if meta_model is not None and meta_model != effective_embedding_model:
            print(
                f"  ! Warning: embedding_model mismatch for topic '{topic}': "
                f"index has '{meta_model}', query uses '{effective_embedding_model}'. Consider reindexing."
            )

        # Validate FAISS dimension if available
        if hasattr(index, "d") and index.d != effective_embedding_dim:
            raise SystemExit(
                f"FAISS dim mismatch for topic '{topic}': index.d={index.d} metadata_embedding_dim={effective_embedding_dim}. "
                f"Reindex needed."
            )

        k = min(top_k * 2, index.ntotal)
        scores, indices = index.search(query_vec, k)

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            page = metadata["pages"][idx].copy()
            page["score"] = float(score)
            page["topic"] = topic
            page["_idx"] = int(idx)
            all_pages.append(page)

    all_pages.sort(key=lambda p: p["score"], reverse=True)
    return all_pages[:top_k]

# ── Reasoning ─────────────────────────────────────────────────────────────────


SYSTEM_PROMPT = """You are a theoretical physics research assistant with deep expertise in condensed matter physics, quantum mechanics, and mathematical physics.

You will receive a research question and relevant pages from a personal research library, provided as transcribed text with LaTeX math.

Guidelines:
- Use LaTeX for all math
- Cite specific pages (e.g. "As shown on page 3 of Sandratskii 2017...")
- Show derivation steps clearly
- If the pages lack sufficient information, say so explicitly
- Be rigorous — this is research-level physics"""

DIRECT_SYSTEM_PROMPT = """You are a helpful physics research assistant. Respond conversationally and concisely."""


def reason(question, pages, model, extra_system: str | None = None):
    if pages:
        system = SYSTEM_PROMPT
        content = [
            {"type": "text", "text": f"Research question: {question}\n\nRelevant pages:"}]
        for page in pages:
            header = f"\n[{page['source']} — Page {page['page_num']} (score: {page['score']:.3f})]"
            desc = page.get("description", "").strip()
            content.append({"type": "text", "text": f"{header}\n{desc}"})
        print(f"  sending {len(pages)} pages to {model}...")
    else:
        system = DIRECT_SYSTEM_PROMPT
        content = [{"type": "text", "text": question}]
        print(f"  direct query (no retrieval) to {model}...")
    if extra_system:
        system = system + extra_system
    text, usage = provider.reason(
        system_prompt=system,
        user_messages=[{"role": "user", "content": content}],
        model=model,
        temperature=0.2,
        max_tokens=16384,
        timeout=None,
    )
    return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

# ── Main ───────────────────────────────────────────────────────────────────────


@dataclass
class QueryResult:
    question: str
    topics: list[str]
    model: str
    top_k: int
    pages: list[dict]
    response_text: str
    in_tok: int
    out_tok: int
    query_cost: float | None        # None = no pricing configured for this model
    budget_data: dict
    monthly_budget: float           # fallback max budget for the CBorg section

    @property
    def cost_str(self) -> str:
        """One source of truth for the human-readable cost string."""
        if self.query_cost is not None:
            return f"${self.query_cost:.4f}"
        return f"(unknown — no pricing for '{self.model}')"


@dataclass
class CBorgBudget:
    spent: float | None        # None = unexpected schema
    budget: float | str        # float if known, "?" if not
    reset_at: str              # raw ISO string or "?"
    raw_keys: list[str] = field(default_factory=list)

    @property
    def remaining(self) -> float | str:
        if self.spent is not None and isinstance(self.budget, (int, float)):
            return self.budget - self.spent
        return "?"

    @property
    def reset_str(self) -> str:
        if not self.reset_at or self.reset_at == "?":
            return "?"
        try:
            from datetime import datetime, timezone
            reset_dt = datetime.fromisoformat(
                self.reset_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            delta = reset_dt - now
            if delta.total_seconds() > 0:
                total_s = int(delta.total_seconds())
                days = total_s // 86400
                hours = (total_s % 86400) // 3600
                minutes = (total_s % 3600) // 60
                return f"{days}d {hours}h {minutes}m"
            return "imminent"
        except Exception:
            return self.reset_at


def _should_retrieve(question: str) -> bool:
    """Quick LLM call to decide whether the query warrants document retrieval.
    Defaults to True on any failure so retrieval is never silently skipped."""
    try:
        text, _ = provider.reason(
            system_prompt=(
                "You are a query router for a physics research assistant. "
                "Reply with only YES or NO. When in doubt, reply NO.\n"
                "Reply YES only if the message is unambiguously a physics or mathematics research question: "
                "a derivation, a concept explanation, a question about a specific paper, theorem, or equation.\n"
                "Reply NO for everything else: greetings, thanks, meta-questions, vague or conversational messages, "
                "and any message that does not clearly require searching a research library."
            ),
            user_messages=[{"role": "user", "content": question}],
            model=ROUTING_MODEL,
            temperature=0.0,
            max_tokens=5,
            timeout=10,
        )
        return text.strip().upper().startswith("Y")
    except Exception:
        return True


def run_query(
    question,
    topics=None,
    model=DEFAULT_MODEL,
    top_k=DEFAULT_TOP_K,
    force_retrieval: bool = False,
    disable_upgrades: bool = False,
    extra_system: str | None = None,
) -> QueryResult:
    """Pure logic: route, retrieve, reason, record spend. No printing, no file I/O.

    Raises ValueError (not SystemExit) so a GUI can catch and display it.
    force_retrieval: skip the router and always retrieve.
    disable_upgrades: skip lazy vision upgrades (use during eval to avoid index mutation).
    extra_system: appended to the active system prompt (eval uses this for \\boxed{} instruction).
    """
    topics = topics or get_all_topics()
    if not topics:
        raise ValueError("No indexes found. Run ingest.py first.")

    metadata_by_topic = {t: load_topic_metadata(t) for t in topics}
    first_meta = metadata_by_topic[topics[0]]

    monthly_budget = float(first_meta.get("monthly_budget", 50.0))

    if force_retrieval or _should_retrieve(question):
        pages = retrieve_pages(topics, question, top_k)
        if not disable_upgrades:
            upgrade_raw_math_pages(pages)
    else:
        pages = []

    response_text, in_tok, out_tok = reason(question, pages, model, extra_system=extra_system)

    query_cost = provider.estimate_cost(model, in_tok, out_tok)
    budget_data = record_spend(query_cost if query_cost is not None else 0.0)

    return QueryResult(
        question=question,
        topics=topics,
        model=model,
        top_k=top_k,
        pages=pages,
        response_text=response_text,
        in_tok=in_tok,
        out_tok=out_tok,
        query_cost=query_cost,
        budget_data=budget_data,
        monthly_budget=monthly_budget,
    )


def fetch_cborg_budget(fallback_budget: float, *, wait: bool = True) -> "CBorgBudget | None":
    import time
    if wait:
        time.sleep(5)
    cborg = provider.get_budget_info()
    if not cborg:
        return None
    spent = cborg.get("spend")
    if spent is None:
        return CBorgBudget(spent=None, budget="?", reset_at="?",
                           raw_keys=list(cborg.get("_raw", {}).keys()))
    budget = cborg.get("max_budget", fallback_budget)
    reset_at = cborg.get("budget_reset_at") or "?"
    return CBorgBudget(spent=spent, budget=budget, reset_at=reset_at)


def print_cborg_budget(cborg: "CBorgBudget | None"):
    if cborg is None:
        print(
            f"  CBorg actual:  (unavailable — check https://cborg.lbl.gov/api_spendcheck/)")
    elif cborg.raw_keys:
        print(
            f"  CBorg actual:  (field names unexpected — raw keys: {cborg.raw_keys})")
        print(f"                 check https://cborg.lbl.gov/api_spendcheck/")
    else:
        print(
            f"  CBorg actual:  ${cborg.spent:.4f} spent / ${cborg.budget} budget  (resets in {cborg.reset_str})")
        rem = cborg.remaining
        if isinstance(rem, float):
            print(f"  CBorg left:    ${rem:.4f}")
        else:
            print(f"  CBorg left:    {rem}")
    print(f"──────────────────────────────────────────────────────────")


def print_result_to_terminal(result: QueryResult):
    """All the terminal-facing output. CLI-only; a GUI ignores this entirely."""
    topics = result.topics

    print(f"\nQuery: {result.question}")
    print(
        f"Topics: {', '.join(topics)}  |  Model: {result.model}  |  Top-K: {result.top_k}")
    print("=" * 60)

    print(f"\n[1] Retrieved {len(result.pages)} page(s):")
    for p in result.pages:
        tag = f"[{p['topic']}] " if len(topics) > 1 else ""
        flag = "" if p.get("clean", True) else "  (raw)"
        print(
            f"    {tag}{p['source']} — p.{p['page_num']}  (score: {p['score']:.4f}){flag}")

    print(f"\n{'=' * 60}")
    print(result.response_text)
    print("=" * 60)

    print(f"\n── Cost ──────────────────────────────────────────────────")
    print(f"  Tokens:        {result.in_tok:,} in / {result.out_tok:,} out")
    print(f"  This query:    {result.cost_str}")
    print(
        f"  Local MTD:     ${result.budget_data['spent']:.4f}  ({result.budget_data['queries']} queries this session)")


def write_markdown(result: QueryResult):
    """Persist the last response to disk. CLI-only for now."""
    out_path = SCRIPT_DIR / "last_response.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Query\n\n{result.question}\n\n")
        f.write(
            f"**Topics:** {', '.join(result.topics)}  |  **Model:** {result.model}\n\n")
        f.write("**Retrieved pages:**\n")
        for p in result.pages:
            tag = f"[{p['topic']}] " if len(result.topics) > 1 else ""
            f.write(
                f"- {tag}{p['source']} — p.{p['page_num']} (score: {p['score']:.4f})\n")
        f.write(
            f"\n**Cost:** {result.cost_str}  |  **MTD:** ${result.budget_data['spent']:.4f}\n\n")
        f.write(f"# Response\n\n{result.response_text}\n")
    print(f"Saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--topic", nargs="*", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    try:
        result = run_query(args.question, args.topic, args.model, args.top_k)
    except ValueError as e:
        raise SystemExit(str(e))

    print_result_to_terminal(result)
    cborg = fetch_cborg_budget(result.monthly_budget)
    print_cborg_budget(cborg)
    write_markdown(result)


if __name__ == "__main__":
    main()
