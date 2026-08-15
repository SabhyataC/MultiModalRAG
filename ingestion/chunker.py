"""
Splits extracted PDF page text into overlapping chunks suitable for
embedding and retrieval, tagged with metadata for traceability.
"""

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_text(text):
    """Splits a single string of text into a list of chunk strings."""
    if not text or not text.strip():
        return []
    return _splitter.split_text(text)


def chunk_page(pdf_path, page):
    """
    Chunks a single page's combined text (PDF text + OCR text) and
    attaches metadata to every chunk.

    Returns a list of dicts:
        {
            "text": str,
            "source_pdf": str,
            "page_num": int,
            "image_paths": [str, ...],   # images present on this page, if any
            "chunk_index": int,
        }
    """
    pdf_name = os.path.basename(pdf_path)

    combined_text = page["text"]
    if page.get("ocr_text"):
        combined_text += "\n\n" + page["ocr_text"]

    raw_chunks = chunk_text(combined_text)
    raw_chunks = [c for c in raw_chunks if not _is_garbled_text(c)]

    return [
        {
            "text": chunk,
            "source_pdf": pdf_name,
            "page_num": page["page_num"],
            "image_paths": page.get("image_paths", []),
            "chunk_index": i,
        }
        for i, chunk in enumerate(raw_chunks)
    ]

_COMMON_ENGLISH_WORDS = {
    "the", "and", "of", "in", "for", "to", "is", "are", "with", "on",
    "by", "as", "that", "this", "a", "an", "or", "from", "at", "be",
    "which", "it", "these", "such", "can", "will", "their", "its",
}


def _is_garbled_text(text, min_stopword_ratio=0.08, max_long_word_ratio=0.15):
    words = [w.strip(".,;:()[]\"'") for w in text.lower().split()]
    words = [w for w in words if w]
    if len(words) < 5:
        return False

    stopword_ratio = sum(1 for w in words if w in _COMMON_ENGLISH_WORDS) / len(words)
    long_word_ratio = sum(1 for w in words if len(w) > 12) / len(words)

    return stopword_ratio < min_stopword_ratio or long_word_ratio > max_long_word_ratio


if __name__ == "__main__":
    from ingestion.pdf_extractor import extract_from_pdf

    pdf_path = "data/sample2.pdf"
    pages = extract_from_pdf(pdf_path)

    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_page(pdf_path, page))

    print(f"Total chunks: {len(all_chunks)}\n")
    for c in all_chunks:
        print(c)
        print()