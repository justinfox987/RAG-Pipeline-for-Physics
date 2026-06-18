"""
Estimate ingestion cost before running ingest.py.

Runs the real page classification (same logic as ingest.py) to get accurate
routing counts, then optionally samples a few pages through the vision model
to measure actual token usage and extrapolate cost + wall-clock time.

Usage:
    python cost_estimator.py                        # standard ingest, all topics
    python cost_estimator.py heisenberg dmi         # specific topics
    python cost_estimator.py --base                 # base build (25% vision target)
    python cost_estimator.py heisenberg --sample 5  # live token sampling
    python cost_estimator.py heisenberg --sample 0  # counts only, no API calls
"""
import time
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz

from config import (
    PAPERS_DIR, TEXTBOOK_PAGE_THRESHOLD, PROBLEM_SECTION_PENALTY,
    VISION_MODEL, MAX_WORKERS, PAGE_DPI, VISION_TARGET, BASE_VISION_TARGET,
    MAX_EMBED_CHARS,
)
from ingest import (
    classify_page, compute_problem_pages, compute_auto_threshold,
    render_page, provider,
)

# ── Pricing ─────────────────────────────────────────────────────────────

VISION_PRICE_IN, VISION_PRICE_OUT = provider._PRICING.get(
    VISION_MODEL, (0.0, 0.0))
EMBED_PRICE_PER_M = 0.12   # cohere-embed-v4: $0.12 / 1M input tokens

AVG_EMBED_TOKENS_VISION = 500
AVG_EMBED_TOKENS_TEXT = 300


# ── Classification ──────────────────────────────────────────────────────

def classify_topic(topic: str, vision_target: float) -> dict:
    """Run routing classification across all PDFs in curr_resources/<topic>/."""
    src = PAPERS_DIR / topic
    if not src.is_dir():
        raise SystemExit(f"  ! {PAPERS_DIR.name}/{topic}/ not found")

    n_vision = n_text = n_skip = 0
    vision_targets: list[tuple[Path, int]] = []
    text_chars = 0

    for pdf in sorted(src.rglob("*.pdf")):
        doc = fitz.open(str(pdf))
        n = len(doc)
        print(f"    {pdf.name}  ({n} pages)", end="")

        if n <= TEXTBOOK_PAGE_THRESHOLD:
            for i in range(n):
                vision_targets.append((pdf, i + 1))
            n_vision += n
            print(f"  [paper — all {n} pages → vision]")
            doc.close()
            continue

        problem_pages = compute_problem_pages(doc)
        threshold = compute_auto_threshold(doc, problem_pages, vision_target)

        fv = ft = fs = 0
        for i, page in enumerate(doc):
            penalty = PROBLEM_SECTION_PENALTY if (
                i + 1) in problem_pages else 0
            route, text, _ = classify_page(
                page, threshold=threshold, threshold_penalty=penalty)
            if route == "vision":
                vision_targets.append((pdf, i + 1))
                fv += 1
            elif route == "text":
                text_chars += len(text[:MAX_EMBED_CHARS])
                ft += 1
            else:
                fs += 1
        doc.close()

        pct = fv / max(n, 1)
        print(f"  [textbook — {fv} vision ({pct:.0%}), {ft} text, {fs} skip]")
        n_vision += fv
        n_text += ft
        n_skip += fs

    return {
        "vision": n_vision,
        "text": n_text,
        "skip": n_skip,
        "targets": vision_targets,
        "text_chars": text_chars,
    }


# ── Live sampling ───────────────────────────────────────────────────────

SAMPLE_WORKERS = 6


def _sample_one(pdf: Path, page_num: int) -> tuple:
    doc = fitz.open(str(pdf))
    data_uri = render_page(doc[page_num - 1], dpi=PAGE_DPI)
    doc.close()
    t0 = time.time()
    _, usage = provider.transcribe_image(
        data_uri, "Transcribe this page.", VISION_MODEL,
        temperature=0.1, max_tokens=4000, timeout=60,
    )
    return (
        pdf.name, page_num,
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
        time.time() - t0,
    )


def sample_vision_tokens(targets: list, sample_n: int):
    """Call the vision model on a random sample of pages and record token usage."""
    if sample_n <= 0 or not targets:
        return None

    sample = random.sample(targets, min(sample_n, len(targets)))
    in_toks, out_toks, secs = [], [], []

    print(
        f"\n  sampling {len(sample)} page(s) through {VISION_MODEL} ({SAMPLE_WORKERS} workers)...")
    with ThreadPoolExecutor(max_workers=SAMPLE_WORKERS) as ex:
        futures = {
            ex.submit(
                _sample_one,
                pdf,
                pg): (
                pdf.name,
                pg) for pdf,
            pg in sample}
        for fut in as_completed(futures):
            try:
                name, pg, inp, out, dt = fut.result()
                in_toks.append(inp)
                out_toks.append(out)
                secs.append(dt)
                print(
                    f"    {name} p.{pg}: {inp:,} in / {out:,} out  ({dt:.1f}s)")
            except Exception as e:
                name, pg = futures[fut]
                print(f"    {name} p.{pg}: failed — {e}")

    if not in_toks:
        return None
    return (
        sum(in_toks) / len(in_toks),
        sum(out_toks) / len(out_toks),
        sum(secs) / len(secs),
    )


# ── Reporting ───────────────────────────────────────────────────────────

def report(counts: dict, measured, vision_target: float, is_base: bool):
    n_vision = counts["vision"]
    n_text = counts["text"]
    n_skip = counts["skip"]
    total = n_vision + n_text + n_skip
    mode = "base build" if is_base else "standard ingest"

    print(
        f"\n  ── Routing summary  [{mode}, target: {vision_target:.0%} vision] ──────")
    print(f"  total pages:   {total}")
    print(f"  vision now:    {n_vision}  ({n_vision/max(total,1):.1%})")
    print(f"  text layer:    {n_text}")
    print(f"  skipped:       {n_skip}")

    # ── Vision cost ──────────────────────────────────────────────────────────
    print(
        f"\n  ── Vision cost  ({VISION_MODEL}) ──────────────────────────────────────")
    if measured:
        avg_in, avg_out, avg_sec = measured
        print(
            f"  measured avg:  {avg_in:,.0f} in / {avg_out:,.0f} out tokens  ({avg_sec:.1f}s/page)")
    else:
        avg_in, avg_out, avg_sec = 1200, 600, 8.0
        print(
            f"  estimated avg: {avg_in:,} in / {avg_out:,} out tokens  (run --sample N to measure)")

    tot_in = avg_in * n_vision
    tot_out = avg_out * n_vision
    cost_vis = tot_in / 1e6 * VISION_PRICE_IN + tot_out / 1e6 * VISION_PRICE_OUT
    serial_s = avg_sec * n_vision
    par_s = serial_s / max(MAX_WORKERS, 1)

    print(f"  total tokens:  {tot_in:,.0f} in / {tot_out:,.0f} out")
    if VISION_PRICE_IN == 0.0 and VISION_PRICE_OUT == 0.0:
        print(f"  vision cost:   $0.00  (model is free on CBorg)")
    else:
        print(f"  vision cost:   ${cost_vis:.4f}")
    print(
        f"  est. time:     ~{par_s/60:.0f} min  ({MAX_WORKERS} workers, no rate-limit slack)")

    # ── Embedding cost ───────────────────────────────────────────────────────
    text_tokens = counts["text_chars"] / 4
    vis_tokens = n_vision * (avg_out if measured else AVG_EMBED_TOKENS_VISION)
    total_embed = text_tokens + vis_tokens
    cost_embed = total_embed / 1e6 * EMBED_PRICE_PER_M

    print(
        f"\n  ── Embedding cost  (cohere-embed-v4 @ ${EMBED_PRICE_PER_M}/1M) ─────────")
    print(f"  est. tokens:   {total_embed:,.0f}")
    print(f"  embed cost:    ${cost_embed:.4f}")

    # ── Total ────────────────────────────────────────────────────────────────
    print(f"\n  ── Total estimate ─────────────────────────────────────────────────────")
    print(f"  cost:          ${cost_vis + cost_embed:.4f}")
    print(f"  time:          ~{par_s/60:.0f} min  (vision dominates)")
    if not measured:
        print(f"  (re-run with --sample N to replace the fallback token averages)")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "topics",
        nargs="*",
        help="Topic names to estimate (default: all in curr_resources/)")
    ap.add_argument(
        "--base",
        action="store_true",
        help=f"Estimate for base build ({BASE_VISION_TARGET:.0%} vision target)")
    ap.add_argument(
        "--sample",
        type=int,
        default=30,
        help="Pages to sample through vision model (0 = counts only, default: 30)")
    ap.add_argument("--vision-target", type=float, default=None,
                    help="Override vision fraction target (overrides --base)")
    args = ap.parse_args()

    if args.vision_target is not None:
        vision_target = args.vision_target
        is_base = False
    elif args.base:
        vision_target = BASE_VISION_TARGET
        is_base = True
    else:
        vision_target = VISION_TARGET
        is_base = False

    if args.topics:
        topics = args.topics
    else:
        if not PAPERS_DIR.exists():
            raise SystemExit(f"Document directory not found: {PAPERS_DIR}")
        topics = sorted(p.name for p in PAPERS_DIR.iterdir() if p.is_dir())
        if not topics:
            raise SystemExit(f"No topic folders found in {PAPERS_DIR}")

    # ── Classify all topics, accumulate counts ──────────────────────────────
    combined = {
        "vision": 0,
        "text": 0,
        "skip": 0,
        "targets": [],
        "text_chars": 0}
    for topic in topics:
        print(f"\n{'=' * 60}")
        print(f"  Topic: {topic}")
        print(f"{'=' * 60}")
        counts = classify_topic(topic, vision_target)
        combined["vision"] += counts["vision"]
        combined["text"] += counts["text"]
        combined["skip"] += counts["skip"]
        combined["targets"] += counts["targets"]
        combined["text_chars"] += counts["text_chars"]

    # ── Sample globally across all topics ────────────────────────────────────
    measured = sample_vision_tokens(combined["targets"], args.sample)

    # ── One combined cost report ────────────────────────────────────────────
    label = f"{'All topics' if len(topics) > 1 else topics[0]} — combined"
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    report(combined, measured, vision_target, is_base)

    print()


if __name__ == "__main__":
    main()
