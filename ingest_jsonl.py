"""
JSONL dataset ingestion for the RAG pipeline.

Each line in the JSONL file becomes one indexed record (problem + solution pair).
No vision transcription needed — text is already clean.

Usage:
    python ingest_jsonl.py dataset.jsonl --topic physics-problems
    python ingest_jsonl.py dataset.jsonl --topic physics-problems --problem-field question --solution-field answer
    python ingest_jsonl.py dataset.jsonl --topic physics-problems --reindex
    python ingest_jsonl.py dataset.jsonl --topic physics-problems --preview 5
"""
import argparse
import json
import sys
from pathlib import Path

# Force UTF-8 output on Windows so Chinese/math characters don't crash prints.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time

import numpy as np
from tqdm import tqdm

from ingest import (
    TopicIndex,
    embed_texts,
    count_math_indicators,
    file_hash,
)
from config import EMBED_BATCH, EMBED_STAGGER, MAX_EMBED_CHARS

# Ordered candidate field names for auto-detection.
_PROBLEM_CANDIDATES = ("problem", "question", "query", "context", "input", "prompt")
_SOLUTION_CANDIDATES = ("solution", "answer", "response", "output", "explanation")


def detect_fields(sample: dict) -> tuple[str, str]:
    """Return (problem_field, solution_field) by scanning a sample record."""
    keys = set(sample.keys())
    prob = next((f for f in _PROBLEM_CANDIDATES if f in keys), None)
    sol = next((f for f in _SOLUTION_CANDIDATES if f in keys), None)
    if prob is None or sol is None:
        available = ", ".join(sorted(keys))
        sys.exit(
            f"Could not auto-detect problem/solution fields from: {available}\n"
            f"Use --problem-field and --solution-field to specify them explicitly."
        )
    return prob, sol


def embed_sequential(descriptions: list[str]) -> np.ndarray:
    """Embed in sequential batches with a progress bar that updates after each batch."""
    batches = [descriptions[i: i + EMBED_BATCH]
               for i in range(0, len(descriptions), EMBED_BATCH)]
    results = []
    with tqdm(total=len(descriptions), desc="  embedding", unit="rec") as pbar:
        for i, batch in enumerate(batches):
            vecs = embed_texts(batch)
            results.append(vecs)
            pbar.update(len(batch))
            if i < len(batches) - 1:
                time.sleep(EMBED_STAGGER)
    return np.vstack(results)


def build_description(record: dict, prob_field: str, sol_field: str) -> str:
    problem = str(record.get(prob_field, "")).strip()
    solution = str(record.get(sol_field, "")).strip()
    return f"PROBLEM:\n{problem}\n\nSOLUTION:\n{solution}"


def parse_filters(filter_args: list[str]) -> list[tuple[str, str]]:
    """Parse ['language=en', 'domain=Thermodynamics'] into [('language', 'en'), ...]."""
    parsed = []
    for f in filter_args:
        if "=" not in f:
            sys.exit(f"Invalid --filter format: '{f}' (expected field=value)")
        k, v = f.split("=", 1)
        parsed.append((k.strip(), v.strip()))
    return parsed


def _dir_hash(dir_path: Path) -> str:
    """Stable hash for a PhysReason directory based on all problem.json paths and sizes."""
    import hashlib
    h = hashlib.sha256()
    for p in sorted(dir_path.rglob("problem.json")):
        h.update(str(p).encode())
        h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:16]


def build_physreason_description(data: dict) -> str:
    """Build a description from a PhysReason problem.json structure."""
    qs = data.get("question_structure", {})
    parts = []

    context = qs.get("context", "").strip()
    if context:
        parts.append(f"PROBLEM:\n{context}")

    sub_q_keys = sorted(k for k in qs if k.startswith("sub_question_"))
    for key in sub_q_keys:
        label = key.replace("_", " ").title()
        parts.append(f"{label}: {qs[key].strip()}")

    captions = data.get("image_captions", "").strip()
    if captions:
        parts.append(f"Figure: {captions}")

    steps_by_subq = data.get("explanation_steps", {})
    solution_lines = []
    for subq_key in sorted(steps_by_subq):
        steps = steps_by_subq[subq_key]
        for step_key in sorted(steps):
            solution_lines.append(steps[step_key].strip())

    if solution_lines:
        parts.append("SOLUTION:\n" + "\n".join(solution_lines))

    return "\n\n".join(parts)


def load_physreason_dir(dir_path: Path, filters: list[tuple[str, str]]) -> list[tuple[str, dict]]:
    """Read all problem.json files from a PhysReason-style directory.
    Returns list of (folder_name, data) pairs."""
    records = []
    subdirs = sorted(p for p in dir_path.iterdir() if p.is_dir())
    for subdir in subdirs:
        json_file = subdir / "problem.json"
        if not json_file.exists():
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ! skipping {subdir.name}: {e}")
            continue
        if all(str(data.get(k, "")) == v for k, v in filters):
            records.append((subdir.name, data))
    return records


def ingest_jsonl(
    jsonl_path: Path,
    topic: str,
    prob_field: str | None,
    sol_field: str | None,
    force: bool,
    preview: int,
    filters: list[tuple[str, str]],
) -> None:
    is_dir = jsonl_path.is_dir()

    if is_dir:
        raw_records = load_physreason_dir(jsonl_path, filters)
        if not raw_records:
            sys.exit("No problem.json files found after applying filters.")
        print(f"  format: PhysReason directory ({len(raw_records)} problems)")
        # Use folder name as identifier, sequential index as page_num
        records = [(i + 1, folder, data) for i, (folder, data) in enumerate(raw_records)]
    else:
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        raw = []
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if all(str(rec.get(k, "")) == v for k, v in filters):
                    raw.append((i, rec))
            except json.JSONDecodeError as e:
                print(f"  ! skipping line {i}: {e}")
        if not raw:
            sys.exit("No valid JSON records found after applying filters.")

        _, sample = raw[0]
        if prob_field is None or sol_field is None:
            p, s = detect_fields(sample)
            prob_field = prob_field or p
            sol_field = sol_field or s

        if filters:
            print("  filters: " + ", ".join(f"{k}={v}" for k, v in filters))
        print(f"  fields: problem='{prob_field}'  solution='{sol_field}'")
        print(f"  records: {len(raw)}")
        records = [(line_num, None, rec) for line_num, rec in raw]

    if preview > 0:
        print(f"\n  -- preview ({preview} records) --")
        for entry in records[:preview]:
            if is_dir:
                idx, folder, data = entry
                desc = build_physreason_description(data)
                label = folder
            else:
                idx, _, rec = entry
                desc = build_description(rec, prob_field, sol_field)
                label = f"line {idx}"
            print(f"\n  [{label}] {desc[:300]}{'...' if len(desc) > 300 else ''}")
        print()
        return

    fname = jsonl_path.name
    fhash = file_hash(jsonl_path) if not is_dir else _dir_hash(jsonl_path)

    index = TopicIndex(topic)
    if not force and index.has_file(fname, fhash):
        print(f"  · {fname} (already indexed — use --reindex to force)")
        return

    pages = []
    descriptions = []
    for entry in records:
        if is_dir:
            idx, folder, data = entry
            desc = build_physreason_description(data)
        else:
            idx, _, rec = entry
            desc = build_description(rec, prob_field, sol_field)

        desc = desc[:MAX_EMBED_CHARS]
        math_score = count_math_indicators(desc)
        pages.append({
            "page_num": idx,
            "description": desc,
            "image_path": "",
            "clean": True,
            "math_score": math_score,
        })
        descriptions.append(desc)

    print(f"  embedding {len(descriptions)} records...")
    vectors = embed_sequential(descriptions)

    index.add_pages(fname, fhash, pages, vectors)
    index.save()

    print(f"  + indexed {len(pages)} records into topic '{topic}'")
    print(f"    index now has {index.total_pages} total entries")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a JSONL problem/solution dataset into the RAG pipeline."
    )
    parser.add_argument("file", type=Path, help="Path to the .jsonl file")
    parser.add_argument("--topic", required=True, help="Topic name (index to add to)")
    parser.add_argument("--problem-field", default=None,
                        help="JSON field name for the problem text (auto-detected if omitted)")
    parser.add_argument("--solution-field", default=None,
                        help="JSON field name for the solution text (auto-detected if omitted)")
    parser.add_argument("--reindex", action="store_true",
                        help="Re-ingest even if this file is already indexed")
    parser.add_argument("--preview", type=int, default=0, metavar="N",
                        help="Print the first N formatted records and exit (no indexing)")
    parser.add_argument("--filter", dest="filters", action="append", default=[],
                        metavar="FIELD=VALUE",
                        help="Only ingest records where FIELD equals VALUE (repeatable). "
                             "Example: --filter language=en --filter domain=Thermodynamics")
    args = parser.parse_args()

    if not args.file.exists():
        sys.exit(f"File not found: {args.file}")

    filters = parse_filters(args.filters)
    print(f"\nIngesting: {args.file.name}  ->  topic '{args.topic}'")
    ingest_jsonl(
        jsonl_path=args.file,
        topic=args.topic,
        prob_field=args.problem_field,
        sol_field=args.solution_field,
        force=args.reindex,
        preview=args.preview,
        filters=filters,
    )


if __name__ == "__main__":
    main()
