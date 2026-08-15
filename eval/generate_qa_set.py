"""
Phase 6 item 1: auto-generates a QA eval set grounded in known
source chunks. Each question is generated from a single chunk, so
we know the correct page/source for it upfront -- needed later to
score retrieval precision/recall, not just faithfulness.
"""

import json
import re
import ollama

from ingestion.pdf_extractor import extract_from_pdf
from ingestion.chunker import chunk_page

MODEL_NAME = "llama3.2:3b"
SOURCE_PDFS = ["data/sample2.pdf"]
QUESTIONS_PER_CHUNK = 2  # tune to land in the 20-30 range depending on chunk count

QA_PROMPT_TEMPLATE = """Based ONLY on the following text, write one factual question and \
its answer. The question must be answerable using only this text. Respond in strict JSON \
with keys "question" and "answer", nothing else.

Text:
{chunk_text}
"""


def _generate_qa_for_chunk(chunk_text):
    prompt = QA_PROMPT_TEMPLATE.format(chunk_text=chunk_text)
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response["message"]["content"]

    # model sometimes wraps JSON in markdown fences -- strip if present
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def build_eval_set():
    eval_set = []

    for pdf_path in SOURCE_PDFS:
        pages = extract_from_pdf(pdf_path)
        for page in pages:
            chunks = chunk_page(pdf_path, page)
            for chunk in chunks:
                for _ in range(QUESTIONS_PER_CHUNK):
                    qa = _generate_qa_for_chunk(chunk["text"])
                    if qa is None or "question" not in qa or "answer" not in qa:
                        continue
                    eval_set.append({
                        "question": qa["question"],
                        "expected_answer": qa["answer"],
                        "source_pdf": pdf_path,
                        "page_num": page["page_num"],
                        "source_chunk": chunk["text"],
                    })

    return eval_set


if __name__ == "__main__":
    eval_set = build_eval_set()

    with open("eval/qa_set.json", "w") as f:
        json.dump(eval_set, f, indent=2)

    print(f"Generated {len(eval_set)} QA pairs -> eval/qa_set.json")
    