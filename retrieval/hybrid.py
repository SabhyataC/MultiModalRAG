"""
Query fan-out and merge: combines dense + BM25 text search via
Reciprocal Rank Fusion, reranks the fused candidates with the
cross-encoder, and separately fans out to the image collection.

Text and image results are kept separate rather than merged into one
ranked list — the cross-encoder can't score images, and CLIP distances
aren't on a comparable scale to cross-encoder relevance scores, so a
single combined ranking would be misleading.
"""

from retrieval.dense_search import dense_search_text, dense_search_images
from retrieval.bm25_search import bm25_search_text
from retrieval.reranker import rerank

RRF_K = 60  # standard damping constant for reciprocal rank fusion


def _reciprocal_rank_fusion(*ranked_lists):
    """
    Merges multiple ranked lists of result dicts (each already sorted
    best-first) by rank position rather than raw score, using RRF:
    score(doc) = sum(1 / (RRF_K + rank)) across every list it appears in.
    """
    fused_scores = {}
    doc_lookup = {}

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list):
            doc_id = result["id"]
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (RRF_K + rank)
            doc_lookup[doc_id] = result  # last write wins; fine, content is identical

    fused_ids = sorted(fused_scores, key=lambda doc_id: fused_scores[doc_id], reverse=True)
    return [doc_lookup[doc_id] for doc_id in fused_ids]


def _filter_by_source(results, source_pdf):
    if not source_pdf:
        return results
    return [r for r in results if r["metadata"].get("source_pdf") == source_pdf]


def hybrid_search(query, top_k=5, fanout_k=15, source_pdf=None):
    """
    Full Phase 4 pipeline: fan out to dense text, BM25 text, and dense
    image search, fuse + rerank the text results, and return text and
    image results separately.

    source_pdf: if given, restricts results to only this document
    (matched against metadata["source_pdf"], e.g. "sample2.pdf").
    """
    dense_results = _filter_by_source(dense_search_text(query, top_k=fanout_k), source_pdf)
    bm25_results = _filter_by_source(bm25_search_text(query, top_k=fanout_k), source_pdf)
    image_results = _filter_by_source(dense_search_images(query, top_k=top_k), source_pdf)

    fused_text = _reciprocal_rank_fusion(dense_results, bm25_results)
    reranked_text = rerank(query, fused_text, top_k=top_k)

    return {
        "text": reranked_text,
        "images": image_results,
    }


if __name__ == "__main__":
    query = "civic issue reporting"
    results = hybrid_search(query)

    print("=== TEXT ===")
    for r in results["text"]:
        print(f"{r['rerank_score']:.4f}  {r['document'][:80]}")

    print("\n=== IMAGES ===")
    for r in results["images"]:
        print(f"{r['distance']:.4f}  {r['metadata']['image_path']}")