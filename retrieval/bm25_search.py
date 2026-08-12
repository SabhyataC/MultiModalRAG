"""
BM25 keyword search over the text collection.

Builds an in-memory BM25 index from every document currently stored
in Chroma's text collection. Rebuilt on every call — simple, correct,
fine at this project's current scale. Revisit with caching if it
becomes a bottleneck.
"""

import re
from rank_bm25 import BM25Okapi

from ingestion.vector_store import get_text_collection


def _tokenize(text):
    """Lowercase, strip punctuation, split on whitespace."""
    return re.findall(r"\b\w+\b", text.lower())


def _build_bm25_index():
    collection = get_text_collection()
    stored = collection.get()  # fetches all documents + metadata + ids

    ids = stored["ids"]
    documents = stored["documents"]
    metadatas = stored["metadatas"]

    tokenized_corpus = [_tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, ids, documents, metadatas


def bm25_search_text(query, top_k=5):
    """
    Keyword search over the text collection using BM25.
    """
    bm25, ids, documents, metadatas = _build_bm25_index()

    if not ids:
        return []

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {
            "id": ids[i],
            "document": documents[i],
            "metadata": metadatas[i],
            "score": scores[i],  # higher is more relevant
        }
        for i in ranked
    ]


if __name__ == "__main__":
    query = "civic issue reporting"
    for r in bm25_search_text(query):
        print(f"{r['score']:.4f}  {r['document'][:80]}")