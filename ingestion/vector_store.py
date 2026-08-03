"""
Vector store setup using Chroma. Provides a persistent local
collection to store and query text-chunk and image embeddings.
"""

import chromadb
from sentence_transformers import SentenceTransformer

_client = chromadb.PersistentClient(path="chroma_db")
_text_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_text_collection():
    return _client.get_or_create_collection(name="multimodal_rag_text")


def get_image_collection():
    return _client.get_or_create_collection(name="multimodal_rag_images")


def embed_text_chunk(text):
    """Embeds a text chunk using a lightweight sentence-transformer model."""
    return _text_model.encode(text).tolist()


def add_text_chunks(collection, chunks):
    """
    Adds a list of chunk dicts (from chunker.chunk_page) to the
    collection as text-type entries.
    """
    if not chunks:
        return

    ids = [f"{c['source_pdf']}_p{c['page_num']}_c{c['chunk_index']}" for c in chunks]
    embeddings = [embed_text_chunk(c["text"]) for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "type": "text",
            "source_pdf": c["source_pdf"],
            "page_num": c["page_num"],
            "image_paths": ",".join(c["image_paths"]) if c["image_paths"] else "",
        }
        for c in chunks
    ]

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def add_image_embedding(collection, image_path, embedding, source_pdf, page_num):
    """
    Adds a single image's CLIP embedding to the collection as an
    image-type entry.
    """
    import os
    item_id = f"{source_pdf}_p{page_num}_img_{os.path.basename(image_path)}"

    collection.add(
        ids=[item_id],
        embeddings=[embedding],
        documents=[image_path],  # store the path as the "document" for images
        metadatas=[{
            "type": "image",
            "source_pdf": source_pdf,
            "page_num": page_num,
            "image_path": image_path,
        }],
    )


if __name__ == "__main__":
    from ingestion.pdf_extractor import extract_from_pdf
    from ingestion.chunker import chunk_page
    from ingestion.image_embedder import embed_image

    pdf_path = "data/sample2.pdf"
    text_collection = get_text_collection()
image_collection = get_image_collection()

pages = extract_from_pdf(pdf_path)
for page in pages:
    chunks = chunk_page(pdf_path, page)
    add_text_chunks(text_collection, chunks)

    for img_path in page["image_paths"]:
        img_emb = embed_image(img_path)
        add_image_embedding(image_collection, img_path, img_emb, pdf_path, page["page_num"])

    print(f"Text items: {text_collection.count()}")
    print(f"Image items: {image_collection.count()}")