"""
Phase 6 item 2: custom eval for retrieval (recall@k, precision@k)
and faithfulness (LLM-as-judge), scored against eval/qa_set.json.
"""

import json
import ollama

from retrieval.hybrid import hybrid_search
from generation.llm import generate_answer, MODEL_NAME
import os
import re

JUDGE_MODEL_NAME = MODEL_NAME  

TOP_K = 5

FAITHFULNESS_PROMPT_TEMPLATE = """You are judging whether an AI-generated answer is faithful \
to the given context. The answer is FAITHFUL if every factual claim in it is supported by \
the context, even if worded differently or paraphrased. It is UNFAITHFUL only if it states \
something the context does not support, or contradicts the context.

Context:
{context}

Answer to judge:
{answer}

First, on a single line, write your verdict: "VERDICT: YES" or "VERDICT: NO".
Then, below it, briefly explain your reasoning by listing each claim and whether it's supported.
"""


def _load_eval_set(path="eval/qa_set.json"):
    with open(path) as f:
        return json.load(f)


def _retrieval_hit_rank(text_results, expected_pdf, expected_page):
    """
    1-indexed rank of the first retrieved chunk matching the expected
    source (by pdf basename + page, not exact chunk text -- duplicate
    ingestion can create multiple chunk ids for identical content).
    None if not found in the retrieved set.
    """
    expected_basename = os.path.basename(expected_pdf)
    for rank, result in enumerate(text_results, start=1):
        meta = result["metadata"]
        if meta.get("source_pdf") == expected_basename and meta.get("page_num") == expected_page:
            return rank
    return None


def _judge_faithfulness(answer, hybrid_results, debug=False):
    context = "\n\n".join(c["document"] for c in hybrid_results["text"])
    if debug:
        print(f"    [judge context]: {context[:500]!r}")
    prompt = FAITHFULNESS_PROMPT_TEMPLATE.format(context=context, answer=answer)

    response = ollama.chat(
        model=JUDGE_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 512},
    )
    
    raw = response["message"]["content"].strip()
    if debug:
        print(f"    [judge raw]: {raw!r}")

    match = re.search(r"VERDICT:\s*(YES|NO)", raw.upper())
    if not match:
        if debug:
            print("    [judge parse warning]: no VERDICT line found, defaulting to False")
        return False
    return match.group(1) == "YES"


def run_eval(path="eval/qa_set.json"):
    eval_set = _load_eval_set(path)

    recall_hits = 0
    precision_sum = 0
    faithful_count = 0
    per_question_results = []

    for item in eval_set:
        question = item["question"]
        expected_pdf = item["source_pdf"]
        expected_page = item["page_num"]

        results = hybrid_search(question, top_k=TOP_K)
        rank = _retrieval_hit_rank(results["text"], expected_pdf, expected_page)

        hit = rank is not None
        recall_hits += int(hit)
        precision_sum += (1 / TOP_K) if hit else 0

        answer = generate_answer(question, results)
        faithful = _judge_faithfulness(answer, results, debug=True)
        faithful_count += int(faithful)

        per_question_results.append({
            "question": question,
            "retrieval_hit": hit,
            "hit_rank": rank,
            "faithful": faithful,
            "generated_answer": answer,
        })

        print(f"[{'HIT' if hit else 'MISS'}] rank={rank} faithful={faithful}  {question[:60]}")

    n = len(eval_set)
    summary = {
        "num_questions": n,
        "recall_at_k": recall_hits / n,
        "precision_at_k": precision_sum / n,
        "faithfulness_rate": faithful_count / n,
    }

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    with open("eval/results.json", "w") as f:
        json.dump({"summary": summary, "per_question": per_question_results}, f, indent=2)

    return summary


if __name__ == "__main__":
    run_eval()