"""Gradio UI for the Physics RAG Research Assistant (v3)."""
import base64
import json
import mimetypes
from pathlib import Path

import gradio as gr

from query import (
    get_all_topics,
    retrieve_pages,
    upgrade_raw_math_pages,
    _should_retrieve,
    record_spend,
    fetch_cborg_budget,
    stream_reason_generator,
    provider,
    INDEXES_DIR,
)
from session_store import (
    create_session,
    load_session,
    save_session,
    list_sessions,
    delete_session,
)
from config import (
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    RETRIEVAL_MIN_SCORE,
    VISION_MODEL,
    MONTHLY_BUDGET,
    PAPERS_DIR,
)
from ingest import ingest_pdf, TopicIndex


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_path(file_input) -> str:
    """Extract a local file path from whatever gr.MultimodalTextbox / gr.File returns."""
    if isinstance(file_input, str):
        return file_input
    if isinstance(file_input, dict):
        return file_input.get("path", file_input.get("name", ""))
    return getattr(file_input, "path", getattr(file_input, "name", ""))


def encode_file(file_input) -> tuple[str, str, str]:
    """Return (local_path, base64_data_uri, mime) for an uploaded file."""
    path = _get_path(file_input)
    p = Path(path)
    suffix = p.suffix.lower()
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        mime = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp", ".gif": "image/gif",
        }.get(suffix, mime)
    uri = f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
    return path, uri, mime


def _file_data(path: str, mime: str) -> dict:
    """Gradio 6 FileDataDict for displaying a local file in the chatbot."""
    p = Path(path)
    return {
        "path": path,
        "orig_name": p.name,
        "mime_type": mime,
        "meta": {"_type": "gradio.FileData"},
    }


def session_choices() -> list[tuple[str, str]]:
    sessions = list_sessions()
    choices = []
    for s in sessions:
        title = s["title"]
        label = title[:44] + "…" if len(title) > 44 else title
        choices.append((label, s["id"]))
    return choices


def msgs_to_history(messages: list[dict]) -> list[dict]:
    """Convert our session message format to Gradio 6 chatbot messages format.
    Historical images/files are shown as a text note (data URIs can't be served by Gradio)."""
    history = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        images = msg.get("images", [])   # stored as base64 data URIs
        files = msg.get("files", [])     # stored as dicts with name/mime
        if role == "user" and (images or files):
            parts = []
            if content:
                parts.append(content)
            if images:
                n = len(images)
                parts.append(f"*[{n} image{'s' if n > 1 else ''} attached]*")
            if files:
                names = ", ".join(f.get("name", "(file)") for f in files)
                parts.append(f"*[{len(files)} file{'s' if len(files) > 1 else ''}: {names}]*")
            history.append({"role": "user", "content": "\n\n".join(parts).strip()})
        else:
            history.append({"role": role, "content": content})
    return history


def build_details_md(meta: dict) -> str:
    lines = []
    pages = meta.get("pages", [])
    best = meta.get("retrieval_best_score")

    if pages:
        multi = len({p["topic"] for p in pages}) > 1
        lines.append("**Retrieved pages:**\n")
        for p in pages:
            tag = f"[{p['topic']}] " if multi else ""
            flag = " *(raw)*" if not p.get("clean", True) else ""
            lines.append(
                f"- {tag}**{p['source']}** — p.{p['page_num']}  "
                f"(score: {p['score']:.4f}){flag}"
            )
    elif best is not None:
        lines.append(
            f"⚠ No pages passed the retrieval threshold. "
            f"Best candidate score: **{best:.4f}**"
        )
    else:
        lines.append("*No retrieval attempted for this query.*")

    lines.append("")
    in_tok = meta.get("in_tok", 0)
    out_tok = meta.get("out_tok", 0)
    cost = meta.get("cost_str", "?")
    bd = meta.get("budget_data", {})
    spent = bd.get("spent", 0)
    queries = bd.get("queries", 0)
    lines.append(f"**Tokens:** {in_tok:,} in / {out_tok:,} out  |  **Cost:** {cost}")
    lines.append(
        f"**Local MTD:** ${spent:.4f} "
        f"({queries} quer{'y' if queries == 1 else 'ies'})"
    )

    cborg = meta.get("cborg")
    if cborg and "raw_keys" not in cborg and cborg.get("spent") is not None:
        lines.append(
            f"**CBorg:** ${cborg['spent']:.4f} spent / {cborg['budget']}  "
            f"•  resets in {cborg['reset_str']}"
        )

    return "\n".join(lines)


def build_library_md() -> str:
    if not INDEXES_DIR.exists():
        return "*No indexes found. Ingest some documents first.*"
    topics = sorted(
        p.name for p in INDEXES_DIR.iterdir()
        if p.is_dir() and (p / "index.faiss").exists()
    )
    if not topics:
        return "*No topics indexed yet. Go to the Ingest tab to add documents.*"

    lines = []
    for topic in topics:
        meta_path = INDEXES_DIR / topic / "metadata.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            lines.append(f"**{topic}** — *(could not read metadata: {e})*\n")
            continue

        all_pages = meta.get("pages", [])
        file_index = meta.get("files", {})
        n_pages = len(all_pages)
        n_vision = sum(1 for p in all_pages if p.get("clean", False))
        n_raw = n_pages - n_vision
        n_files = len(file_index)

        lines.append(f"### {topic}")
        lines.append(
            f"**{n_files}** file{'s' if n_files != 1 else ''}, "
            f"**{n_pages}** pages ({n_vision} vision · {n_raw} raw)"
        )
        lines.append(
            f"*Embedding:* `{meta.get('embedding_model', '?')}`  "
            f"*Vision:* `{meta.get('vision_model', '?')}`  "
            f"*Budget:* ${meta.get('monthly_budget', '?')}/mo"
        )
        for fname in sorted(file_index.keys()):
            fp = [p for p in all_pages if p.get("source") == fname]
            fv = sum(1 for p in fp if p.get("clean", False))
            lines.append(
                f"- 📄 **{fname}** — {len(fp)} pages "
                f"({fv} vision · {len(fp) - fv} raw)"
            )
        lines.append("")

    return "\n".join(lines)


# ── Session management ────────────────────────────────────────────────────────

def load_initial_state():
    sessions = list_sessions()
    if sessions:
        sid = sessions[0]["id"]
        session = load_session(sid) or create_session()
        sid = session["id"]
    else:
        session = create_session()
        sid = session["id"]
    history = msgs_to_history(session.get("messages", []))
    choices = session_choices()
    model_choices = provider.list_models()
    selected_model = DEFAULT_MODEL if DEFAULT_MODEL in model_choices else (model_choices[0] if model_choices else DEFAULT_MODEL)
    return (
        history,
        sid,
        gr.update(choices=choices, value=sid),
        gr.update(visible=False),
        "",
        gr.update(value={"text": "", "files": []}),
        gr.update(choices=model_choices, value=selected_model),
    )


def switch_session(sid, _current_sid):
    if not sid:
        return gr.update(), _current_sid, gr.update(visible=False), "", gr.update(value={"text": "", "files": []})
    session = load_session(sid)
    if session is None:
        return gr.update(), _current_sid, gr.update(visible=False), "", gr.update(value={"text": "", "files": []})
    history = msgs_to_history(session.get("messages", []))
    return history, sid, gr.update(visible=False), "", gr.update(value={"text": "", "files": []})


def new_chat(_current_sid):
    session = create_session()
    sid = session["id"]
    choices = session_choices()
    return (
        [],
        sid,
        gr.update(choices=choices, value=sid),
        gr.update(visible=False),
        "",
        gr.update(value={"text": "", "files": []}),
    )


def delete_session_handler(sid):
    if sid:
        delete_session(sid)
    sessions = list_sessions()
    if sessions:
        new_sid = sessions[0]["id"]
        session = load_session(new_sid) or create_session()
        new_sid = session["id"]
    else:
        session = create_session()
        new_sid = session["id"]
    history = msgs_to_history(session.get("messages", []))
    choices = session_choices()
    return (
        history,
        new_sid,
        gr.update(choices=choices, value=new_sid),
        gr.update(visible=False),
        "",
        gr.update(value={"text": "", "files": []}),
    )


# ── Chat handler ──────────────────────────────────────────────────────────────

def handle_submit(msg_data, history, session_id, topics, model, top_k):
    """Streaming generator: retrieve → stream response → save session."""
    question = (msg_data.get("text") or "").strip()
    file_list = msg_data.get("files") or []

    image_paths = []   # local temp paths (for chatbot display this session)
    image_mimes = []
    images = []        # base64 data URIs (for the model and session storage)
    attachments = []   # data URIs for non-image attachments
    attachments_meta = []

    for f in file_list:
        try:
            path, uri, mime = encode_file(f)
            if not path:
                continue
            if mime.startswith("image/"):
                image_paths.append(path)
                image_mimes.append(mime)
                images.append(uri)
            else:
                attachments.append(uri)
                attachments_meta.append({"name": Path(path).name, "mime": mime})
        except Exception:
            pass

    if not question and not images and not attachments:
        yield history, gr.update(visible=False), "", session_id, gr.update(), gr.update()
        return

    # ── Show user message ─────────────────────────────────────────────
    if image_paths or attachments_meta:
        # Mix text + FileDataDicts so attachments render inline in the chatbot
        user_content: list = []
        if question:
            user_content.append(question)
        for path, mime in zip(image_paths, image_mimes):
            user_content.append(_file_data(path, mime))
        for meta in attachments_meta:
            user_content.append(f"*Attached file:* {meta.get('name')} ({meta.get('mime')})")
        history = history + [{"role": "user", "content": user_content}]
    else:
        history = history + [{"role": "user", "content": question}]

    history = history + [{"role": "assistant", "content": ""}]
    yield (
        history,
        gr.update(visible=False),
        "",
        session_id,
        gr.update(),
        gr.update(value={"text": "", "files": []}),
    )

    # ── Retrieval ─────────────────────────────────────────────────────
    cur_topics = list(topics) if topics else get_all_topics()
    retrieval_query = question

    if images and len(question.strip()) < 20:
        try:
            extracted, _ = provider.transcribe_image(
                images[0],
                "Extract the physics problem or question from this image. "
                "Output only the problem text, no commentary.",
                VISION_MODEL, temperature=0.1, max_tokens=512, timeout=30,
            )
            if extracted and extracted.strip():
                retrieval_query = extracted.strip()
        except Exception:
            pass

    pages = []
    best_score = None
    try:
        if _should_retrieve(retrieval_query):
            pages, best_score = retrieve_pages(
                cur_topics, retrieval_query, int(top_k),
                min_score=RETRIEVAL_MIN_SCORE,
            )
            upgrade_raw_math_pages(pages)
    except Exception as e:
        print(f"  retrieval error: {e}")

    # ── Stream response ───────────────────────────────────────────────
    partial = ""
    in_tok = out_tok = 0
    streamed = False
    try:
        for chunk, usage in stream_reason_generator(
            question, pages, model, images=(images + attachments) or None
        ):
            streamed = True
            partial += chunk
            if usage is not None:
                in_tok = usage.get("prompt_tokens", 0)
                out_tok = usage.get("completion_tokens", 0)
            else:
                history[-1] = {"role": "assistant", "content": partial}
                yield (
                    history,
                    gr.update(visible=False),
                    "",
                    session_id,
                    gr.update(),
                    gr.update(),
                )
    except Exception as e:
        if attachments and not streamed:
            warning = (
                "⚠ Attached files could not be sent to the model. "
                "Proceeding with images only.\n\n"
            )
            partial = warning
            history[-1] = {"role": "assistant", "content": partial}
            yield (
                history,
                gr.update(visible=False),
                "",
                session_id,
                gr.update(),
                gr.update(),
            )
            try:
                for chunk, usage in stream_reason_generator(
                    question, pages, model, images=images or None
                ):
                    partial += chunk
                    if usage is not None:
                        in_tok = usage.get("prompt_tokens", 0)
                        out_tok = usage.get("completion_tokens", 0)
                    else:
                        history[-1] = {"role": "assistant", "content": partial}
                        yield (
                            history,
                            gr.update(visible=False),
                            "",
                            session_id,
                            gr.update(),
                            gr.update(),
                        )
            except Exception as retry_err:
                partial = f"⚠ Error: {retry_err}"
        else:
            partial = f"⚠ Error: {e}"

    history[-1] = {"role": "assistant", "content": partial}

    # ── Save session ──────────────────────────────────────────────────
    session = load_session(session_id) if session_id else None
    if session is None:
        session = create_session()
        session_id = session["id"]

    pages_slim = [
        {
            "source": p["source"], "page_num": p["page_num"],
            "score": p["score"], "topic": p["topic"],
            "clean": p.get("clean", True),
        }
        for p in pages
    ]

    query_cost = provider.estimate_cost(model, in_tok, out_tok)
    budget_data = record_spend(query_cost or 0.0)
    cborg = fetch_cborg_budget(MONTHLY_BUDGET, wait=False)
    cborg_data = None
    if cborg and not cborg.raw_keys and cborg.spent is not None:
        rem = cborg.remaining
        cborg_data = {
            "spent": cborg.spent, "budget": str(cborg.budget),
            "remaining": f"${rem:.4f}" if isinstance(rem, float) else str(rem),
            "reset_str": cborg.reset_str,
        }

    meta = {
        "topics": cur_topics, "model": model,
        "pages": pages_slim, "retrieval_best_score": best_score,
        "in_tok": in_tok, "out_tok": out_tok,
        "cost_str": (
            f"${query_cost:.4f}" if query_cost is not None
            else f"(unknown — no pricing for '{model}')"
        ),
        "budget_data": budget_data, "cborg": cborg_data,
    }

    if not session["messages"]:
        title_src = question or retrieval_query or "(image)"
        session["title"] = title_src[:60] + ("…" if len(title_src) > 60 else "")

    session["messages"].extend([
        {"role": "user", "content": question, "images": images, "files": attachments_meta},
        {"role": "assistant", "content": partial, "meta": meta},
    ])
    save_session(session)

    new_choices = session_choices()
    details_md = build_details_md(meta)
    yield (
        history,
        gr.update(visible=True, open=False),
        details_md,
        session_id,
        gr.update(choices=new_choices, value=session_id),
        gr.update(value={"text": "", "files": []}),
    )


# ── Ingest handler ────────────────────────────────────────────────────────────

def run_ingest(topic_choice, new_topic_name, files, force):
    topic = new_topic_name.strip() if topic_choice == "(new topic…)" else topic_choice
    if not topic:
        return "⚠ Enter a topic name."
    if not files:
        return "⚠ Upload at least one PDF."

    topic_dir = PAPERS_DIR / topic
    topic_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        src = Path(_get_path(f))
        dest = topic_dir / src.name
        dest.write_bytes(src.read_bytes())
        saved.append(dest)

    index = TopicIndex(topic)
    lines = []
    try:
        for pdf_path in saved:
            added = ingest_pdf(pdf_path, index, force=force)
            icon = "✓" if added else "—"
            note = "indexed" if added else "skipped (already indexed or no indexable pages)"
            lines.append(f"{icon} **{pdf_path.name}** — {note}")
        index.save()
        lines.append("\n**Done.**")
    except Exception as e:
        lines.append(f"\n⚠ **Ingestion failed:** {e}")

    return "\n".join(lines)


def toggle_new_topic(choice):
    return gr.update(visible=choice == "(new topic…)")


def refresh_ingest_topics():
    existing = get_all_topics()
    return gr.update(choices=["(new topic…)"] + existing)


# ── Layout ────────────────────────────────────────────────────────────────────

_all_topics = get_all_topics()
_existing_ingest = get_all_topics()
_model_choices = provider.list_models()
_default_model = DEFAULT_MODEL if DEFAULT_MODEL in _model_choices else (_model_choices[0] if _model_choices else DEFAULT_MODEL)

with gr.Blocks(title="Physics Research Assistant") as demo:

    session_id_state = gr.State(None)

    with gr.Tabs():

        # ── Chat ─────────────────────────────────────────────────────────
        with gr.Tab("💬 Chat"):
            with gr.Row(equal_height=False):

                # Sidebar
                with gr.Column(scale=1, min_width=240):
                    new_chat_btn = gr.Button("＋  New Chat", variant="primary")
                    session_radio = gr.Radio(
                        choices=[],
                        label="Sessions",
                        interactive=True,
                        container=False,
                    )
                    del_btn = gr.Button("✕ Delete session", size="sm", variant="stop")
                    gr.Markdown("---")
                    topics_dd = gr.Dropdown(
                        choices=_all_topics,
                        value=_all_topics,
                        multiselect=True,
                        label="Topics",
                    )
                    model_dd = gr.Dropdown(
                        choices=_model_choices,
                        value=_default_model,
                        label="Model",
                        interactive=True,
                        filterable=True,
                    )
                    topk_sl = gr.Slider(
                        minimum=1, maximum=20,
                        value=DEFAULT_TOP_K, step=1, label="Top-K",
                    )

                # Main chat
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        height=560,
                        latex_delimiters=[
                            {"left": "$$",   "right": "$$",   "display": True},
                            {"left": "$",    "right": "$",    "display": False},
                            {"left": "\\[",  "right": "\\]",  "display": True},
                            {"left": "\\(",  "right": "\\)",  "display": False},
                        ],
                    )
                    with gr.Accordion("Details", open=False, visible=False) as details_acc:
                        details_md = gr.Markdown()
                    msg_box = gr.MultimodalTextbox(
                        file_types=[
                            "image",
                            ".pdf", ".txt", ".md", ".json", ".csv",
                            ".doc", ".docx", ".rtf",
                        ],
                        file_count="multiple",
                        placeholder="Ask a research question…  (Ctrl+V to paste)",
                        show_label=False,
                        submit_btn=True,
                    )

        # ── Library ───────────────────────────────────────────────────────
        with gr.Tab("📚 Library"):
            refresh_lib_btn = gr.Button("Refresh", size="sm")
            library_md = gr.Markdown(build_library_md())

        # ── Ingest ────────────────────────────────────────────────────────
        with gr.Tab("⬆ Ingest"):
            with gr.Row():
                topic_choice_dd = gr.Dropdown(
                    choices=["(new topic…)"] + _existing_ingest,
                    value="(new topic…)" if not _existing_ingest else _existing_ingest[0],
                    label="Topic",
                    interactive=True,
                    scale=2,
                )
                new_topic_tb = gr.Textbox(
                    placeholder="e.g. heisenberg",
                    label="New topic name",
                    visible=not _existing_ingest,
                    scale=2,
                )
            pdf_upload = gr.File(
                file_types=[".pdf"],
                file_count="multiple",
                label="Upload PDF(s)",
            )
            force_cb = gr.Checkbox(
                label="Re-ingest already-indexed files", value=False)
            ingest_btn = gr.Button("Ingest", variant="primary")
            ingest_status = gr.Markdown()

    # ── Events ────────────────────────────────────────────────────────────────

    _session_outputs = [chatbot, session_id_state, session_radio, details_acc, details_md, msg_box, model_dd]

    demo.load(fn=load_initial_state, inputs=[], outputs=_session_outputs)

    # Chat
    _submit_outputs = [chatbot, details_acc, details_md, session_id_state, session_radio, msg_box]
    msg_box.submit(
        fn=handle_submit,
        inputs=[msg_box, chatbot, session_id_state, topics_dd, model_dd, topk_sl],
        outputs=_submit_outputs,
    )

    # Session sidebar
    session_radio.change(
        fn=switch_session,
        inputs=[session_radio, session_id_state],
        outputs=[chatbot, session_id_state, details_acc, details_md, msg_box],
    )
    new_chat_btn.click(
        fn=new_chat,
        inputs=[session_id_state],
        outputs=_session_outputs,
    )
    del_btn.click(
        fn=delete_session_handler,
        inputs=[session_id_state],
        outputs=_session_outputs,
    )

    # Library
    refresh_lib_btn.click(fn=build_library_md, inputs=[], outputs=[library_md])

    # Ingest
    topic_choice_dd.change(
        fn=toggle_new_topic,
        inputs=[topic_choice_dd],
        outputs=[new_topic_tb],
    )
    ingest_btn.click(
        fn=run_ingest,
        inputs=[topic_choice_dd, new_topic_tb, pdf_upload, force_cb],
        outputs=[ingest_status],
    ).then(fn=refresh_ingest_topics, inputs=[], outputs=[topic_choice_dd])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        share=False,
        theme=gr.themes.Soft(),
    )
