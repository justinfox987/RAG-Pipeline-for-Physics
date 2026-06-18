import json
import streamlit as st
from query import INDEXES_DIR

st.set_page_config(page_title="Library", layout="wide")
st.title("Library")

if not INDEXES_DIR.exists():
    st.info("No indexes found. Ingest some documents first.")
    st.stop()

topics = sorted(
    p.name for p in INDEXES_DIR.iterdir()
    if p.is_dir() and (p / "index.faiss").exists()
)

if not topics:
    st.info("No topics indexed yet. Go to Ingest to add documents.")
    st.stop()

for topic in topics:
    meta_path = INDEXES_DIR / topic / "metadata.json"
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        st.warning(f"{topic}: could not read metadata — {e}")
        continue

    all_pages = meta.get("pages", [])
    file_index = meta.get("files", {})
    n_files = len(file_index)
    n_pages = len(all_pages)
    n_vision = sum(1 for p in all_pages if p.get("clean", False))
    n_raw = n_pages - n_vision

    header = f"**{topic}** — {n_files} file{'s' if n_files != 1 else ''}, {n_pages} pages  ({n_vision} vision · {n_raw} raw)"
    with st.expander(header, expanded=False):
        st.caption(
            f"Embedding: `{meta.get('embedding_model', '?')}`  ·  "
            f"Vision: `{meta.get('vision_model', '?')}`  ·  "
            f"Budget: ${meta.get('monthly_budget', '?')}/mo"
        )
        st.divider()

        if not file_index:
            st.write("No files indexed.")
            continue

        for fname in sorted(file_index.keys()):
            fp = [p for p in all_pages if p.get("source") == fname]
            fv = sum(1 for p in fp if p.get("clean", False))
            fr = len(fp) - fv
            cols = st.columns([6, 1, 1, 1])
            cols[0].write(f"📄 **{fname}**")
            cols[1].metric("Pages",  len(fp), label_visibility="collapsed")
            cols[2].metric("Vision", fv,       label_visibility="collapsed")
            cols[3].metric("Raw",    fr,        label_visibility="collapsed")
