"""
Cross-encoder reranker. Dense search and BM25 both score candidates
independently against the query (fast, shallower). A cross-encoder
scores the query and a candidate document jointly in one pass, which
is far more accurate — but too slow to run over the whole corpus. So
it's used as a second pass over a small shortlist, not initial
retrieval.
"""

from sentence_transformers import CrossEncoder

_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query, candidates, top_k=5):
    """
    candidates: list of result dicts (from dense_search_text or
    bm25_search_text), each must have a "document" key.

    Returns the same dicts, re-sorted by cross-encoder relevance
    score, with a "rerank_score" key added.
    """
    if not candidates:
        return []

    pairs = [(query, c["document"]) for c in candidates]
    scores = _reranker.predict(pairs)

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]


if __name__ == "__main__":
    from retrieval.dense_search import dense_search_text
    from retrieval.bm25_search import bm25_search_text

    query = "civic issue reporting"

    dense_results = dense_search_text(query, top_k=10)
    bm25_results = bm25_search_text(query, top_k=10)

    # dedupe by id, keeping first occurrence, before reranking
    seen = set()
    candidates = []
    for r in dense_results + bm25_results:
        if r["id"] not in seen:
            seen.add(r["id"])
            candidates.append(r)

    reranked = rerank(query, candidates)
    for r in reranked:
        print(f"{r['rerank_score']:.4f}  {r['document'][:80]}")