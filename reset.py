"""Reset indexes (and previously, image caches — now removed).

    python reset.py            # wipe everything
    python reset.py --topic X  # wipe just topic X
"""
import argparse
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INDEXES_DIR = SCRIPT_DIR / "indexes"


def clear_topic(topic: str) -> None:
    """Remove the index for a single topic."""
    target = INDEXES_DIR / topic
    if target.exists():
        shutil.rmtree(target)
        print(f"  removed {target}")
    else:
        print(f"  nothing to remove for topic '{topic}'")


def clear_all() -> None:
    """Wipe every index. Full reset."""
    if INDEXES_DIR.exists():
        shutil.rmtree(INDEXES_DIR)
        print(f"  removed {INDEXES_DIR}/")
    else:
        print("  nothing to remove (indexes/ does not exist)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", help="only reset this topic; omit to reset all")
    args = ap.parse_args()
    if args.topic:
        clear_topic(args.topic)
        print(f"reset topic '{args.topic}'.")
    else:
        clear_all()
        print("reset all indexes.")
