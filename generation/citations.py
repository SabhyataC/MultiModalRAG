"""
Post-generation citation enforcement: validates that every [n] marker
in the LLM's answer maps to a real source, flags hallucinated
citations, and resolves numbers into human-readable source references.
"""

import re


def _extract_citation_numbers(answer_text):
    return sorted(set(int(n) for n in re.findall(r"\[(\d+)\]", answer_text)))


def _source_label(index, hybrid_results):
    """Maps a 1-indexed citation number back to its source metadata."""
    text_results = hybrid_results["text"]
    image_results = hybrid_results["images"]

    if index <= len(text_results):
        meta = text_results[index - 1]["metadata"]
        source_pdf = meta.get("source_pdf", "unknown")
        page_num = meta.get("page_num", "?")
        return f"[{index}] {source_pdf}, page {page_num} (text)"

    img_index = index - len(text_results) - 1
    if 0 <= img_index < len(image_results):
        meta = image_results[img_index]["metadata"]
        source_pdf = meta.get("source_pdf", "unknown")
        page_num = meta.get("page_num", "?")
        image_path = meta.get("image_path", "unknown")
        return f"[{index}] {source_pdf}, page {page_num}, image: {image_path}"

    return None  # citation number doesn't map to any real source


def validate_and_format(answer_text, hybrid_results):
    """
    Returns (formatted_answer, warnings).
    formatted_answer has a resolved Sources section appended.
    warnings lists any hallucinated citation numbers found.
    """
    cited_numbers = _extract_citation_numbers(answer_text)
    warnings = []
    source_lines = []

    for n in cited_numbers:
        label = _source_label(n, hybrid_results)
        if label is None:
            warnings.append(f"Citation [{n}] does not match any retrieved source (possible hallucination)")
        else:
            source_lines.append(label)

    if not cited_numbers:
        warnings.append("Answer contains no citations at all")

    formatted = answer_text
    if source_lines:
        formatted += "\n\nSources:\n" + "\n".join(source_lines)

    return formatted, warnings


if __name__ == "__main__":
    from retrieval.hybrid import hybrid_search
    from generation.llm import generate_answer

    query = "civic issue reporting"
    results = hybrid_search(query)
    answer = generate_answer(query, results)

    formatted, warnings = validate_and_format(answer, results)

    print(formatted)
    if warnings:
        print("\n--- WARNINGS ---")
        for w in warnings:
            print(w)