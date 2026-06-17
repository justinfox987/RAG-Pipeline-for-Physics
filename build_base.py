"""
Build a distributable BASE index for a topic.

Usage:
    python build_base.py heisenberg
    python build_base.py heisenberg --reindex

This runs your normal ingestion against papers/<topic>/, then writes a
base_manifest.json describing what was built, then packages everything
EXCEPT page_images/ into dist/base-<topic>.tar.gz for sharing.
"""
import sys
import json
import time
import tarfile
from pathlib import Path
from ingest import USAGE

# Reuse the real ingestion code — no duplication.
from ingest import (
    TopicIndex, ingest_topic, file_hash,
    PAPERS_DIR, INDEXES_DIR,
    EMBEDDING_MODEL, VISION_MODEL, EMBEDDING_DIM,
    BASE_VISION_TARGET, PROBLEM_SECTION_PENALTY,
)

BASE_INDEX_VERSION = 1
DIST_DIR = Path(__file__).resolve().parent / "dist"

# (input $/1M, output $/1M) — keyed by the model name ingest actually uses
VISION_PRICING = {
    "gemini-3.1-flash-lite": (0.25, 1.50),
}


def report_cost(topic, snap):
    print(f"\n=== cost report: {topic} ===")
    if not snap:
        print("  ! no usage recorded.")
        return None

    in_tok, out_tok = snap["vision_in"], snap["vision_out"]
    calls = snap["vision_calls"]
    print(f"  pages: vision={snap['vision_pages']}  text-layer={snap['text_pages']}  skipped={snap['skipped_pages']}")
    print(f"  vision calls: {calls}  (failed: {snap['vision_failed']})")
    print(f"  vision tokens: in={in_tok:,}  out={out_tok:,}")

    price = VISION_PRICING.get(VISION_MODEL)
    if price is None:
        print(f"  ! no pricing for '{VISION_MODEL}'.")
        return None
    in_price, out_price = price
    cost = (in_tok * in_price + out_tok * out_price) / 1_000_000
    print(f"  COST: ${cost:.4f}  (model={VISION_MODEL}, ${in_price}/${out_price} per 1M)")

    if calls:
        per = cost / calls
        in_per = in_tok / calls
        out_per = out_tok / calls
        print(f"  per vision page: ${per:.5f}  (avg in={in_per:.0f}, out={out_per:.0f} tokens)")
        for target in (4000, 10000):
            print(f"  --> {target:,} vision pages ≈ ${per * target:,.2f}")
    return cost


def write_manifest(topic):
    # ... unchanged ...
    topic_dir = INDEXES_DIR / topic
    index = TopicIndex(topic)  # loads from disk

    source_books = []
    src = PAPERS_DIR / topic
    for pdf in sorted(src.rglob("*.pdf")):
        source_books.append({"name": pdf.name, "sha256": file_hash(pdf)})

    manifest = {
        "base_index_version": BASE_INDEX_VERSION,
        "topic": topic,
        "embedding_model": EMBEDDING_MODEL,
        "vision_model": VISION_MODEL,
        "embedding_dim": index.index.d,
        "config_embedding_dim": EMBEDDING_DIM,
        "vision_target":   BASE_VISION_TARGET,
        "problem_penalty": PROBLEM_SECTION_PENALTY,
        "build_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_pages": index.total_pages,
        "source_books": source_books,
        "read_only": True,
        "lazy_upgrade": False,
    }
    manifest_path = topic_dir / "base_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  wrote {manifest_path}")
    return manifest


def package(topic, manifest):
    src = INDEXES_DIR / topic
    DIST_DIR.mkdir(exist_ok=True)
    archive = DIST_DIR / f"base-{topic}-v{manifest['base_index_version']}.tar.gz"

    # The lazy-upgrade page-image cache lives at top-level images/<topic>/, not
    # under indexes/<topic>/, so it's already outside this walk — nothing to
    # exclude. Base indexes ship no images by design (read-only, no PDFs).
    members = [p for p in src.rglob("*") if p.is_file()]

    with tarfile.open(archive, "w:gz") as tar:
        for p in members:
            tar.add(p, arcname=str(p.relative_to(INDEXES_DIR)))

    size_mb = archive.stat().st_size / 1e6
    print(f"  packaged {len(members)} files -> {archive}  ({size_mb:.1f} MB)")
    return archive


def main(argv):
    force = "--reindex" in argv
    topics = [a for a in argv if not a.startswith("--")]
    if not topics:
        print("Usage: python build_base.py <topic> [<topic> ...] [--reindex]")
        sys.exit(1)

    for topic in topics:
        topic_dir = PAPERS_DIR / topic
        if not topic_dir.is_dir():
            print(f"! no papers/{topic}/ folder — skipping")
            continue

        print(f"\n=== building base index: {topic} ===")
        USAGE.reset()
        snap = ingest_topic(topic_dir, force=force, vision_target=BASE_VISION_TARGET)
        report_cost(topic, snap)
        USAGE.print_ingestion_summary(vision_target=BASE_VISION_TARGET)
        manifest = write_manifest(topic)
        package(topic, manifest)

    print("\nDone. Upload the dist/*.tar.gz files to Google Drive and share the links.")


if __name__ == "__main__":
    main(sys.argv[1:])