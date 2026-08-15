"""
Phase 6: Streamlit demo UI for the MultiModalRAG pipeline.
Wires together retrieval.hybrid_search, generation.llm.generate_answer,
and generation.citations.validate_and_format into an interactive demo.
"""

import streamlit as st
from PIL import Image

from retrieval.hybrid import hybrid_search
from generation.llm import generate_answer
from generation.citations import validate_and_format

import re

def _clean_display_text(text):
    """Collapse repeated whitespace from PDF extraction artifacts for display."""
    return re.sub(r"\s+", " ", text).strip()

st.set_page_config(page_title="MultiModalRAG Demo", layout="wide")

st.title("MultiModalRAG")
st.caption("Ask a question about the ingested PDFs — retrieval is hybrid (dense + BM25 + image), answers are grounded with citations.")

import os

from ingestion.pdf_extractor import extract_from_pdf
from ingestion.chunker import chunk_page
from ingestion.image_embedder import embed_image
from ingestion.vector_store import get_text_collection, get_image_collection, add_text_chunks, add_image_embedding


def _get_available_pdfs():
    collection = get_text_collection()
    stored = collection.get()
    pdfs = sorted(set(m.get("source_pdf", "unknown") for m in stored["metadatas"]))
    return pdfs


with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K sources", min_value=1, max_value=10, value=5)

    st.divider()
    available_pdfs = _get_available_pdfs()
    pdf_filter = st.selectbox("Search within", ["All documents"] + available_pdfs)
    selected_pdf = None if pdf_filter == "All documents" else pdf_filter

    st.divider()

    st.header("Upload a PDF")
    uploaded_file = st.file_uploader("Add a new document to the index", type="pdf")
    if uploaded_file is not None:
        if st.button("Ingest this PDF"):
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                try:
                    # save to a real path so pdf_extractor (fitz/pdfplumber) can open it
                    os.makedirs("data/uploaded", exist_ok=True)
                    save_path = os.path.join("data/uploaded", uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    pdf_basename = os.path.basename(save_path)

                    text_collection = get_text_collection()
                    image_collection = get_image_collection()

                    pages = extract_from_pdf(save_path)
                    total_chunks = 0
                    total_images = 0
                    for page in pages:
                        chunks = chunk_page(save_path, page)
                        add_text_chunks(text_collection, chunks)
                        total_chunks += len(chunks)

                        for img_path in page["image_paths"]:
                            img_emb = embed_image(img_path)
                            add_image_embedding(image_collection, img_path, img_emb, pdf_basename, page["page_num"])
                            total_images += 1

                    st.success(f"Ingested {uploaded_file.name}: {total_chunks} text chunks, {total_images} images added.")
                    st.info("Refresh or re-open the sidebar dropdown to see it in 'Search within'.")
                except Exception as e:
                    st.error(f"Ingestion failed (possibly already ingested?): {e}")

    st.divider()
    st.caption("Retrieval: dense embeddings + BM25 + CLIP image search")
    st.caption("Generation: llama3.2:3b via Ollama, citations enforced by prompt")

query = st.text_input("Ask a question", placeholder="e.g. What authentication method is used for genuineness?")
submitted = st.button("Ask", type="primary")

if submitted and query.strip():
    with st.spinner("Retrieving sources..."):
        results = hybrid_search(query, top_k=top_k, source_pdf=selected_pdf)

    with st.spinner("Generating answer..."):
        raw_answer = generate_answer(query, results)
        formatted_answer, warnings = validate_and_format(raw_answer, results)

    st.subheader("Answer")
    st.write(formatted_answer)

    if warnings:
        st.warning("\n".join(warnings))

    text_results = results.get("text", [])
    image_results = results.get("images", [])

    if text_results:
        with st.expander(f"Retrieved text sources ({len(text_results)})", expanded=False):
            for i, r in enumerate(text_results, start=1):
                meta = r["metadata"]
                st.markdown(f"**[{i}]** `{meta.get('source_pdf', 'unknown')}`, page {meta.get('page_num', '?')}")
                st.text(_clean_display_text(r["document"]))
                st.divider()

    if image_results:
        with st.expander(f"Retrieved images ({len(image_results)})", expanded=False):
            cols = st.columns(min(len(image_results), 4))
            for i, r in enumerate(image_results):
                meta = r["metadata"]
                image_path = meta.get("image_path")
                with cols[i % len(cols)]:
                    if image_path:
                        try:
                            st.image(Image.open(image_path), caption=f"{meta.get('source_pdf', 'unknown')}, page {meta.get('page_num', '?')}", use_container_width=True)
                        except Exception:
                            st.caption(f"(couldn't load {image_path})")

elif submitted:
    st.info("Enter a question first.")