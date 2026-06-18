import streamlit as st
from query import get_all_topics, PAPERS_DIR
from ingest import ingest_pdf, TopicIndex

st.set_page_config(page_title="Ingest", layout="wide")
st.title("Ingest Documents")

# ── Topic selection ────────────────────────────────────────────────────────────

existing = get_all_topics()
col1, col2 = st.columns([2, 2])

with col1:
    choice = st.selectbox(
        "Topic",
        ["(new topic…)"] + existing,
        help="Pick an existing topic to add files, or create a new one.",
    )

with col2:
    if choice == "(new topic…)":
        topic_name = st.text_input(
            "New topic name", placeholder="e.g. heisenberg")
    else:
        topic_name = choice
        st.text_input("Topic", value=topic_name, disabled=True,
                      label_visibility="visible")

# ── Upload ─────────────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload PDF(s)",
    type="pdf",
    accept_multiple_files=True,
    help="Upload one or more PDFs. Large/math-heavy PDFs may take several minutes.",
)

force = st.checkbox(
    "Re-ingest already-indexed files",
    help="By default, files with an unchanged hash are skipped.",
)

# ── Run ────────────────────────────────────────────────────────────────────────

if st.button("Ingest", type="primary"):
    if not topic_name or topic_name == "(new topic…)":
        st.warning("Enter a topic name.")
    elif not uploaded:
        st.warning("Upload at least one PDF.")
    else:
        topic_dir = PAPERS_DIR / topic_name
        topic_dir.mkdir(parents=True, exist_ok=True)

        saved = []
        for f in uploaded:
            dest = topic_dir / f.name
            dest.write_bytes(f.getbuffer())
            saved.append(dest)

        index = TopicIndex(topic_name)
        results = []

        try:
            with st.status(f"Ingesting into '{topic_name}'…", expanded=True) as status:
                for pdf_path in saved:
                    st.write(f"Processing **{pdf_path.name}**…")
                    added = ingest_pdf(pdf_path, index, force=force)
                    results.append((pdf_path.name, added))

                if any(a for _, a in results):
                    st.write("Saving index…")
                    index.save()

                status.update(label="Done!", state="complete", expanded=False)
        except (SystemExit, Exception) as e:
            st.error(f"Ingestion failed: {e}")
        else:
            st.divider()
            for name, added in results:
                if added:
                    st.success(f"✓ **{name}** — indexed")
                else:
                    st.info(
                        f"— **{name}** — skipped  "
                        f"*(already indexed, no indexable pages, or scanned PDF in non-interactive mode)*"
                    )
