"""
Session persistence for the Streamlit chat UI.

Sessions are stored as JSON files under sessions/<id>.json.
Each session: {id, title, messages: [{role, content, meta?}]}
"""
import json
import uuid
from pathlib import Path

from config import SCRIPT_DIR

SESSIONS_DIR = SCRIPT_DIR / "sessions"


def _ensure_dir():
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _path(sid: str) -> Path:
    return SESSIONS_DIR / f"{sid}.json"


def _write(session: dict):
    with open(_path(session["id"]), "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)


def create_session() -> dict:
    _ensure_dir()
    session = {"id": str(uuid.uuid4())[:8], "title": "New Chat", "messages": []}
    _write(session)
    return session


def load_session(sid: str) -> dict | None:
    p = _path(sid)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_session(session: dict):
    _ensure_dir()
    _write(session)


def list_sessions() -> list[dict]:
    _ensure_dir()
    stubs = []
    for p in sorted(SESSIONS_DIR.glob("*.json"),
                    key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            with open(p, "r", encoding="utf-8") as f:
                s = json.load(f)
            stubs.append({"id": s["id"], "title": s.get("title", "Untitled")})
        except Exception:
            pass
    return stubs


def delete_session(sid: str):
    p = _path(sid)
    if p.exists():
        p.unlink()
