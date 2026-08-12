"""
Phase 5 end-to-end test: PDF in -> question -> cited answer out.
Runs the full pipeline: ingestion -> chunking/storage -> hybrid
retrieval -> generation -> citation validation.
"""

from ingestion.pdf_extractor import extract_from_pdf
from ingestion.chunker import chunk_page
from ingestion.image_embedder import embed_image
from ingestion.vector_store import (
    get_text_collection,
    get_image_collection,
    add_text_chunks,
    add_image_embedding,
)
from retrieval.hybrid import hybrid_search
from generation.llm import generate_answer
from generation.citations import validate_and_format


def ingest_pdf(pdf_path):
    """Runs extraction, chunking, embedding, and storage for one PDF."""
    text_collection = get_text_collection()
    image_collection = get_image_collection()

    pages = extract_from_pdf(pdf_path)
    for page in pages:
        chunks = chunk_page(pdf_path, page)
        add_text_chunks(text_collection, chunks)

        for img_path in page["image_paths"]:
            img_emb = embed_image(img_path)
            add_image_embedding(image_collection, img_path, img_emb, pdf_path, page["page_num"])

    print(f"Ingested {pdf_path}: {text_collection.count()} text items, {image_collection.count()} image items total in store")


def ask(query):
    """Runs retrieval -> generation -> citation validation for one question."""
    results = hybrid_search(query)
    answer = generate_answer(query, results)
    formatted, warnings = validate_and_format(answer, results)

    print("\n=== ANSWER ===")
    print(formatted)

    if warnings:
        print("\n=== WARNINGS ===")
        for w in warnings:
            print(w)


if __name__ == "__main__":
    pdf_path = "data/sample2.pdf"
    query = "civic issue reporting"

    ingest_pdf(pdf_path)
    ask(query)