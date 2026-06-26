import base64
import hashlib
import io
import re
import streamlit as st
from streamlit_paste_button import paste_image_button
from query import (
    run_query, fetch_cborg_budget,
    get_all_topics, DEFAULT_MODEL, DEFAULT_TOP_K,
)
from session_store import (
    create_session, load_session, save_session,
    list_sessions, delete_session,
)


def _encode_image(uploaded_file) -> str:
    """Return a base64 data URI for an uploaded Streamlit file."""
    mime = uploaded_file.type or "image/png"
    b64 = base64.b64encode(uploaded_file.read()).decode()
    return f"data:{mime};base64,{b64}"


def _show_images(images: list[str]):
    """Display a list of base64 data-URI images in the chat."""
    if not images:
        return
    cols = st.columns(min(len(images), 4))
    for col, uri in zip(cols, images):
        _, b64 = uri.split(",", 1)
        col.image(base64.b64decode(b64))

# ── LaTeX ──────────────────────────────────────────────────────────────────────


def _render_latex(text: str) -> str:
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$',   text, flags=re.DOTALL)
    return text

# ── Meta helpers ───────────────────────────────────────────────────────────────


def _build_meta(result, cborg) -> dict:
    """Build a JSON-serializable metadata dict from a QueryResult + CBorgBudget."""
    pages_slim = [
        {
            "source":   p["source"],
            "page_num": p["page_num"],
            "score":    p["score"],
            "topic":    p["topic"],
            "clean":    p.get("clean", True),
        }
        for p in result.pages
    ]
    retrieval_best_score = result.retrieval_best_score
    cborg_data = None
    if cborg and not cborg.raw_keys and cborg.spent is not None:
        rem = cborg.remaining
        cborg_data = {
            "spent":     cborg.spent,
            "budget":    str(cborg.budget),
            "remaining": f"${rem:.4f}" if isinstance(rem, float) else str(rem),
            "reset_str": cborg.reset_str,
        }
    elif cborg and cborg.raw_keys:
        cborg_data = {"raw_keys": cborg.raw_keys}
    return {
        "topics":                result.topics,
        "model":                 result.model,
        "pages":                 pages_slim,
        "retrieval_best_score":  retrieval_best_score,
        "in_tok":                result.in_tok,
        "out_tok":               result.out_tok,
        "cost_str":              result.cost_str,
        "budget_data":           result.budget_data,
        "cborg":                 cborg_data,
    }


def _render_meta(meta: dict):
    with st.expander("Details"):
        pages = meta.get("pages", [])
        best = meta.get("retrieval_best_score")
        if pages:
            multi = len({p["topic"] for p in pages}) > 1
            for p in pages:
                tag = f"[{p['topic']}] " if multi else ""
                flag = " *(raw)*" if not p.get("clean", True) else ""
                st.write(
                    f"- {tag}**{p['source']}** — p.{p['page_num']}  (score: {p['score']:.4f}){flag}")
        elif best is not None:
            st.warning(
                f"No pages passed the retrieval threshold. "
                f"Best candidate score: **{best:.4f}** — try lowering `retrieval_min_score` "
                f"in this topic's metadata, or broaden the query."
            )
        else:
            st.caption("No retrieval attempted for this query.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Tokens in",  f"{meta['in_tok']:,}")
        col2.metric("Tokens out", f"{meta['out_tok']:,}")
        col3.metric("Cost",       meta["cost_str"])
        queries = meta["budget_data"]["queries"]
        st.caption(
            f"Local MTD: ${meta['budget_data']['spent']:.4f}"
            f"  ({queries} quer{'y' if queries == 1 else 'ies'} this month)"
        )

        cborg = meta.get("cborg")
        if not cborg:
            st.caption("CBorg budget unavailable")
        elif "raw_keys" in cborg:
            st.warning(f"CBorg: unexpected fields — {cborg['raw_keys']}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("CBorg spent", f"${cborg['spent']:.4f}")
            c2.metric("CBorg left",  cborg["remaining"])
            c3.metric("Resets in",   cborg["reset_str"])

# ── Page config ────────────────────────────────────────────────────────────────


st.set_page_config(page_title="Research Assistant", layout="wide")

# ── Session bootstrap ─────────────────────────────────────────────────────────


def _ensure_session():
    sid = st.session_state.get("current_session_id")
    sess = st.session_state.get("current_session")

    if sid is None:
        stubs = list_sessions()
        if stubs:
            sid = stubs[0]["id"]
        else:
            new = create_session()
            st.session_state.current_session_id = new["id"]
            st.session_state.current_session = new
            return

    if sess is None or sess.get("id") != sid:
        loaded = load_session(sid)
        if loaded is None:
            loaded = create_session()
            sid = loaded["id"]
        st.session_state.current_session_id = sid
        st.session_state.current_session = loaded


_ensure_session()
session = st.session_state.current_session

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    if st.button("＋  New Chat", use_container_width=True, type="primary"):
        new = create_session()
        st.session_state.current_session_id = new["id"]
        st.session_state.current_session = new
        st.rerun()

    st.divider()

    for s in list_sessions():
        is_active = s["id"] == st.session_state.current_session_id
        label = s["title"] if len(s["title"]) <= 28 else s["title"][:26] + "…"
        if is_active:
            label = "▶ " + label

        col_t, col_x = st.columns([5, 1])
        if col_t.button(label, key=f"sess_{s['id']}", use_container_width=True):
            st.session_state.current_session_id = s["id"]
            st.session_state.current_session = load_session(s["id"])
            st.rerun()
        if col_x.button("✕", key=f"del_{s['id']}"):
            delete_session(s["id"])
            if st.session_state.current_session_id == s["id"]:
                remaining = [x for x in list_sessions()]
                if remaining:
                    st.session_state.current_session_id = remaining[0]["id"]
                    st.session_state.current_session = load_session(
                        remaining[0]["id"])
                else:
                    new = create_session()
                    st.session_state.current_session_id = new["id"]
                    st.session_state.current_session = new
            st.rerun()

    st.divider()

    st.caption("Query Settings")
    all_topics = get_all_topics()
    topics = st.multiselect("Topics", all_topics,
                            default=all_topics, key="topics")
    model = st.text_input("Model",  value=DEFAULT_MODEL, key="model_input")
    top_k = st.number_input(
        "Top-K", min_value=1, max_value=20, value=DEFAULT_TOP_K, key="top_k_input")

# ── Chat area ─────────────────────────────────────────────────────────────────

st.title(session["title"] if session["messages"] else "New Chat")

for msg in session["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and msg.get("images"):
            _show_images(msg["images"])
        st.markdown(_render_latex(msg["content"]))
        if msg["role"] == "assistant" and msg.get("meta"):
            _render_meta(msg["meta"])

# ── Input ─────────────────────────────────────────────────────────────────────

paste_result = paste_image_button("📋 Paste image from clipboard", key="paste_btn")
if paste_result.image_data is not None:
    buf = io.BytesIO()
    paste_result.image_data.save(buf, format="PNG")
    raw = buf.getvalue()
    img_hash = hashlib.md5(raw).hexdigest()
    # Only add if not already staged or previously sent (component value persists across reruns)
    if img_hash not in st.session_state.get("_paste_hashes", set()):
        uri = "data:image/png;base64," + base64.b64encode(raw).decode()
        st.session_state.setdefault("_pending_pastes", []).append(uri)
        st.session_state.setdefault("_paste_hashes", set()).add(img_hash)

if pending := st.session_state.get("_pending_pastes"):
    _show_images(pending)
    if st.button("✕ Clear pasted images", key="clear_paste"):
        st.session_state.pop("_pending_pastes", None)
        st.session_state.pop("_paste_hashes", None)
        st.rerun()

if prompt := st.chat_input("Ask a research question…", accept_file="multiple"):
    cur_topics = st.session_state.get("topics") or None
    cur_model = st.session_state.get("model_input",  DEFAULT_MODEL)
    cur_top_k = int(st.session_state.get("top_k_input", DEFAULT_TOP_K))

    question = prompt.text or ""
    images = [_encode_image(f) for f in (prompt.files or [])]
    images += st.session_state.pop("_pending_pastes", [])
    # _paste_hashes intentionally kept — prevents the persisted component value
    # from re-staging the same image on the next rerun after sending

    # Show and save user message
    with st.chat_message("user"):
        if images:
            _show_images(images)
        st.markdown(question)

    session["messages"].append({
        "role": "user",
        "content": question,
        "images": images,
    })
    if len(session["messages"]) == 1:
        session["title"] = question[:60] + ("…" if len(question) > 60 else "")
    save_session(session)
    st.session_state.current_session = session

    # Run query and show assistant response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Querying…"):
                result = run_query(
                    question, cur_topics, cur_model, cur_top_k,
                    images=images or None,
                )
        except (ValueError, SystemExit) as e:
            st.error(str(e))
            session["messages"].append(
                {"role": "assistant", "content": f"⚠ {e}", "meta": None})
        else:
            cborg = fetch_cborg_budget(result.monthly_budget, wait=False)
            st.markdown(_render_latex(result.response_text))
            meta = _build_meta(result, cborg)
            _render_meta(meta)
            session["messages"].append({
                "role":    "assistant",
                "content": result.response_text,
                "meta":    meta,
            })

    save_session(session)
    st.session_state.current_session = session
