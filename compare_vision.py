"""
Compare two vision models on the same PDF pages to evaluate transcription quality.

Renders each page once, calls both models in parallel, then uses an LLM judge
to score each transcription. Writes a markdown report.

Usage:
    python compare_vision.py paper.pdf
    python compare_vision.py textbook.pdf --pages 10-20
    python compare_vision.py textbook.pdf --sample 8
    python compare_vision.py textbook.pdf --model-a gemini-3.1-flash-lite --model-b gpt-5-nano
    python compare_vision.py textbook.pdf --no-judge
    python compare_vision.py textbook.pdf --output results.md
"""
import argparse
import base64
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import fitz

from providers import get_provider
from ingest import TRANSCRIPTION_PROMPT, render_page
from config import PAGE_DPI

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_MODEL_A = "cborg-ocr-fast"
DEFAULT_MODEL_B = "gpt-5-nano"
DEFAULT_JUDGE   = "gpt-5.1"
DEFAULT_SAMPLE  = 10
MAX_TOKENS      = 4000
TIMEOUT         = 90

JUDGE_SYSTEM = (
    "You are evaluating vision-model transcriptions of physics/math textbook pages. "
    "You will receive two transcriptions of the same page and must score each on four criteria. "
    "Respond ONLY with valid JSON — no markdown fences, no commentary."
)

JUDGE_PROMPT = """\
You are assessing whether Model B is a viable substitute for Model A at transcribing physics/math pages for a research retrieval system.

Score each transcription on these criteria (integer 1–5, where 5 = excellent):

1. math_completeness  — Are all visible equations captured in LaTeX?
2. latex_fidelity     — Is the notation accurate and faithful (no hallucinated or altered symbols)?
3. figure_coverage    — Are figures/graphs described with axes, curves, key values?
                        Score 3 if the page has no figures.
4. structure          — Are section headers, equation numbers, theorem/definition labels captured?

Then give an overall verdict on whether B can do the job of A:
- "acceptable"   — gaps are negligible; B would serve equally well for retrieval
- "marginal"     — B has minor gaps that may occasionally miss a detail but is broadly usable
- "unacceptable" — B has critical gaps (wrong equations, missing derivations, hallucinated notation) that would hurt retrieval quality

Transcription A ({model_a}):
{text_a}

---

Transcription B ({model_b}):
{text_b}

---

Respond with this exact JSON shape (integer scores only):
{{
  "A": {{"math_completeness": 0, "latex_fidelity": 0, "figure_coverage": 0, "structure": 0}},
  "B": {{"math_completeness": 0, "latex_fidelity": 0, "figure_coverage": 0, "structure": 0}},
  "verdict": "acceptable",
  "gaps": "one sentence describing specifically what B misses or gets wrong, or 'none' if negligible"
}}\
"""

CRITERIA = ["math_completeness", "latex_fidelity", "figure_coverage", "structure"]

provider = get_provider()


# ── Core helpers ───────────────────────────────────────────────────────────────

def transcribe(data_uri: str, model: str, page_num: int) -> str:
    try:
        text, _ = provider.transcribe_image(
            data_uri, TRANSCRIPTION_PROMPT, model,
            temperature=0.1, max_tokens=MAX_TOKENS, timeout=TIMEOUT,
        )
        return text or "[empty response]"
    except Exception as e:
        return f"[FAILED: {e}]"


def judge(text_a: str, text_b: str, model_a: str, model_b: str, judge_model: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        model_a=model_a, model_b=model_b,
        text_a=text_a[:3000], text_b=text_b[:3000],
    )
    try:
        response, _ = provider.reason(
            system_prompt=JUDGE_SYSTEM,
            user_messages=[{"role": "user", "content": prompt}],
            model=judge_model,
            temperature=0.0,
            max_tokens=512,
            timeout=60,
        )
        raw = response.strip()
        # Strip markdown fences if the model ignores instructions
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}


# ── Page range parsing ─────────────────────────────────────────────────────────

def parse_pages(spec: str, n_pages: int) -> list[int]:
    """Parse '3-7' or '1,4,9' into a sorted list of 1-indexed page numbers."""
    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            pages.update(range(int(lo), int(hi) + 1))
        else:
            pages.add(int(part))
    return sorted(p for p in pages if 1 <= p <= n_pages)


# ── Markdown report ────────────────────────────────────────────────────────────

def avg_scores(results: list[dict], label: str) -> dict[str, float]:
    totals = {c: 0 for c in CRITERIA}
    count = 0
    for r in results:
        scores = r.get("judgment", {}).get(label)
        if isinstance(scores, dict):
            for c in CRITERIA:
                totals[c] += scores.get(c, 0)
            count += 1
    if count == 0:
        return {c: 0.0 for c in CRITERIA}
    return {c: totals[c] / count for c in CRITERIA}


def write_report(
    pdf_path: Path,
    model_a: str,
    model_b: str,
    judge_model: str | None,
    results: list[dict],
    output: Path,
):
    lines = []
    lines.append(f"# Vision Model Comparison\n")
    lines.append(f"**PDF:** `{pdf_path.name}`  ")
    lines.append(f"**Model A (baseline):** `{model_a}`  ")
    lines.append(f"**Model B (candidate):** `{model_b}`  ")
    if judge_model:
        lines.append(f"**Judge:** `{judge_model}`  ")
    lines.append(f"**Date:** {date.today()}  ")
    lines.append(f"**Pages compared:** {len(results)}\n")

    if judge_model:
        avgs_a = avg_scores(results, "A")
        avgs_b = avg_scores(results, "B")
        overall_a = sum(avgs_a.values()) / len(CRITERIA)
        overall_b = sum(avgs_b.values()) / len(CRITERIA)

        verdicts = {"acceptable": 0, "marginal": 0, "unacceptable": 0}
        for r in results:
            v = r.get("judgment", {}).get("verdict", "").lower()
            if v in verdicts:
                verdicts[v] += 1

        n_judged = sum(verdicts.values())
        pct_ok = (verdicts["acceptable"] + verdicts["marginal"]) / max(n_judged, 1) * 100
        if verdicts["unacceptable"] == 0:
            conclusion = "B appears to be a **viable substitute** for A."
        elif verdicts["acceptable"] + verdicts["marginal"] > verdicts["unacceptable"]:
            conclusion = "B is a **marginal substitute** — usable but with meaningful gaps."
        else:
            conclusion = "B is **not a reliable substitute** — too many critical gaps."

        lines.append("---\n")
        lines.append("## Summary\n")
        lines.append(f"| Criterion | `{model_a}` | `{model_b}` | delta |")
        lines.append("|---|---|---|---|")
        for c in CRITERIA:
            delta = avgs_a[c] - avgs_b[c]
            sign = "+" if delta > 0 else ""
            lines.append(f"| {c} | {avgs_a[c]:.2f} | {avgs_b[c]:.2f} | {sign}{delta:.2f} |")
        delta_overall = overall_a - overall_b
        sign = "+" if delta_overall > 0 else ""
        lines.append(
            f"| **overall** | **{overall_a:.2f}** | **{overall_b:.2f}** | **{sign}{delta_overall:.2f}** |"
        )
        lines.append("")
        lines.append(
            f"**Verdicts:** {verdicts['acceptable']} acceptable, "
            f"{verdicts['marginal']} marginal, {verdicts['unacceptable']} unacceptable "
            f"({pct_ok:.0f}% usable) — out of {n_judged} pages\n"
        )
        lines.append(f"**Conclusion:** {conclusion}\n")

    lines.append("---\n")

    for r in results:
        pg = r["page"]
        lines.append(f"## Page {pg}\n")

        if judge_model:
            j = r.get("judgment", {})
            if "error" in j:
                lines.append(f"> **Judge error:** {j['error']}\n")
            else:
                verdict = j.get("verdict", "?")
                gaps = j.get("gaps", "")
                lines.append(f"> **Verdict: {verdict}** — {gaps}\n")

                scores_a = j.get("A", {})
                scores_b = j.get("B", {})
                lines.append(f"| Criterion | `{model_a}` | `{model_b}` | delta |")
                lines.append("|---|---|---|---|")
                for c in CRITERIA:
                    sa = scores_a.get(c, 0)
                    sb = scores_b.get(c, 0)
                    d = sa - sb
                    sign = "+" if d > 0 else ""
                    lines.append(f"| {c} | {sa} | {sb} | {sign}{d} |")
                lines.append("")

        lines.append(f"### Model A — `{model_a}`\n")
        lines.append("<details><summary>Transcription</summary>\n")
        lines.append(f"\n{r['text_a']}\n")
        lines.append("</details>\n")

        lines.append(f"### Model B — `{model_b}`\n")
        lines.append("<details><summary>Transcription</summary>\n")
        lines.append(f"\n{r['text_b']}\n")
        lines.append("</details>\n")

        lines.append("---\n")

    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to: {output}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("pdf", help="Path to the PDF file")
    ap.add_argument("--model-a", default=DEFAULT_MODEL_A,
                    help=f"Baseline model (default: {DEFAULT_MODEL_A})")
    ap.add_argument("--model-b", default=DEFAULT_MODEL_B,
                    help=f"Candidate model (default: {DEFAULT_MODEL_B})")
    ap.add_argument("--judge", default=DEFAULT_JUDGE,
                    help=f"LLM judge model (default: {DEFAULT_JUDGE})")
    ap.add_argument("--no-judge", action="store_true",
                    help="Skip LLM judging, output transcriptions only")
    ap.add_argument("--pages",
                    help="Page range to compare, e.g. '5-15' or '3,7,12'")
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                    help=f"Random pages to sample if --pages not given (default: {DEFAULT_SAMPLE})")
    ap.add_argument("--output",
                    help="Output markdown path (default: <pdf_stem>_vision_comparison.md)")
    args = ap.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        sys.exit(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    n = len(doc)
    print(f"PDF: {pdf_path.name}  ({n} pages)")

    if args.pages:
        page_nums = parse_pages(args.pages, n)
    else:
        k = min(args.sample, n)
        page_nums = sorted(random.sample(range(1, n + 1), k))

    print(f"Pages to compare: {page_nums}")
    print(f"Model A: {args.model_a}")
    print(f"Model B: {args.model_b}")
    judge_model = None if args.no_judge else args.judge
    if judge_model:
        print(f"Judge:   {judge_model}")
    print()

    output_path = Path(args.output) if args.output else \
        pdf_path.parent / f"{pdf_path.stem}_vision_comparison.md"

    results = []
    for pg in page_nums:
        print(f"  Page {pg}/{n} ...", end="  ", flush=True)
        data_uri = render_page(doc[pg - 1])

        # Call both models in parallel
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_a = ex.submit(transcribe, data_uri, args.model_a, pg)
            fut_b = ex.submit(transcribe, data_uri, args.model_b, pg)
            text_a = fut_a.result()
            text_b = fut_b.result()

        result = {
            "page": pg,
            "text_a": text_a,
            "text_b": text_b,
        }

        if judge_model:
            j = judge(text_a, text_b, args.model_a, args.model_b, judge_model)
            result["judgment"] = j
            verdict = j.get("verdict", "err")
            print(f"  →  {verdict}", end="")

        print()
        results.append(result)

    doc.close()
    write_report(pdf_path, args.model_a, args.model_b, judge_model, results, output_path)


if __name__ == "__main__":
    main()
