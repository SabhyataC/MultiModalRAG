"""
Wires Phase 4's hybrid retrieval into an Ollama LLM call, with
citation of page/source enforced via prompt structure.
"""

import ollama

MODEL_NAME = "llama3.2:3b"


def _build_context(hybrid_results):
    """
    Formats retrieved text chunks into a numbered context block, each
    tagged with a source id the model must cite by. Image results are
    included by filename/page reference only (no pixel input yet).
    """
    lines = []

    for i, chunk in enumerate(hybrid_results["text"], start=1):
        source_pdf = chunk["metadata"].get("source_pdf", "unknown")
        page_num = chunk["metadata"].get("page_num", "?")
        lines.append(
            f"[{i}] (source: {source_pdf}, page {page_num})\n{chunk['document']}"
        )

    for i, img in enumerate(hybrid_results["images"], start=len(hybrid_results["text"]) + 1):
        source_pdf = img["metadata"].get("source_pdf", "unknown")
        page_num = img["metadata"].get("page_num", "?")
        image_path = img["metadata"].get("image_path", "unknown")
        lines.append(
            f"[{i}] (source: {source_pdf}, page {page_num}, image: {image_path})\n"
            f"[This is an image on this page — no extracted description available]"
        )

    return "\n\n".join(lines)


def _build_prompt(query, context):
    return f"""You are answering a question using only the numbered sources below. \
Every claim in your answer must be followed by a citation like [1] or [2] pointing \
to the source number it came from. If the sources don't contain the answer, say so \
— do not make anything up.

Sources:
{context}

Question: {query}

Answer (with citations):"""


def generate_answer(query, hybrid_results):
    """
    hybrid_results: the dict returned by retrieval.hybrid.hybrid_search()
    Returns the raw LLM answer text (citations enforced via prompt).
    """
    context = _build_context(hybrid_results)
    prompt = _build_prompt(query, context)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"]


if __name__ == "__main__":
    from retrieval.hybrid import hybrid_search

    query = "civic issue reporting"
    results = hybrid_search(query)
    answer = generate_answer(query, results)

    print(answer)