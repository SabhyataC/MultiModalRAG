"""
Dense (embedding-based) search over the text and image Chroma collections.
"""

from ingestion.vector_store import get_text_collection, get_image_collection, embed_text_chunk
from ingestion.image_embedder import embed_text as embed_text_clip


def dense_search_text(query, top_k=5):
    """
    Dense search over the text collection using the sentence-transformer
    embedding space (same one used to embed stored text chunks).
    """
    collection = get_text_collection()
    query_embedding = embed_text_chunk(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    return _format_results(results)


def dense_search_images(query, top_k=5):
    """
    Dense search over the image collection using CLIP's shared
    text/image embedding space.
    """
    collection = get_image_collection()
    query_embedding = embed_text_clip(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    return _format_results(results)


def _format_results(chroma_results):
    """
    Flattens Chroma's batched query result (lists-of-lists, since a
    single call can take multiple query_embeddings) into a simple
    list of dicts for one query.
    """
    ids = chroma_results["ids"][0]
    documents = chroma_results["documents"][0]
    metadatas = chroma_results["metadatas"][0]
    distances = chroma_results["distances"][0]

    return [
        {
            "id": ids[i],
            "document": documents[i],
            "metadata": metadatas[i],
            "distance": distances[i],  # raw L2 — lower is more similar
        }
        for i in range(len(ids))
    ]


if __name__ == "__main__":
    query = "your test query here"

    print("=== TEXT ===")
    for r in dense_search_text(query):
        print(f"{r['distance']:.4f}  {r['document'][:80]}")

    print("\n=== IMAGES ===")
    for r in dense_search_images(query):
        print(f"{r['distance']:.4f}  {r['metadata']['image_path']}")