import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import json
import time
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

from rag_pipeline import run_pipeline
from bm25_retriever import load_bm25_index
from embeddings import load_vectorstore
from hybrid_retriever import hybrid_retrieve
from reranker import rerank

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"

# ── LOAD GOLDEN DATASET ───────────────────────────────────────

def load_golden_dataset():
    path = os.path.join(os.path.dirname(__file__), 'golden_dataset.json')
    with open(path, 'r') as f:
        return json.load(f)

# ── METRIC 1: FAITHFULNESS ────────────────────────────────────
# Are the claims in the answer supported by the retrieved chunks?

def measure_faithfulness(answer: str, contexts: list) -> float:
    if "cannot find sufficient information" in answer.lower():
        return 1.0  # refused answers are perfectly faithful

    context_text = "\n".join([f"[{i+1}]: {c[:400]}" for i, c in enumerate(contexts)])

    prompt = f"""You are evaluating whether an answer is faithful to source documents.

Source chunks:
{context_text}

Answer to evaluate:
{answer}

Score the answer from 0.0 to 1.0 where:
1.0 = every claim in the answer is supported by the chunks
0.5 = some claims are supported, some are not
0.0 = the answer contains claims not found in the chunks at all

Respond with ONLY a number between 0.0 and 1.0. No explanation."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10
    )

    try:
        return float(response.choices[0].message.content.strip())
    except Exception:
        return 0.5

# ── METRIC 2: ANSWER RELEVANCY ────────────────────────────────
# Does the answer actually address the question?

def measure_answer_relevancy(question: str, answer: str) -> float:
    if "cannot find sufficient information" in answer.lower():
        return 0.5  # neutral for refused answers

    prompt = f"""You are evaluating whether an answer is relevant to a question.

Question: {question}

Answer: {answer}

Score from 0.0 to 1.0 where:
1.0 = the answer directly and completely addresses the question
0.5 = the answer is partially relevant
0.0 = the answer does not address the question at all

Respond with ONLY a number between 0.0 and 1.0. No explanation."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10
    )

    try:
        return float(response.choices[0].message.content.strip())
    except Exception:
        return 0.5

# ── METRIC 3: CONTEXT PRECISION ───────────────────────────────
# Are the retrieved chunks actually useful for answering?

def measure_context_precision(question: str, contexts: list, ground_truth: str) -> float:
    useful = 0
    for ctx in contexts:
        prompt = f"""Is this chunk useful for answering the question?

Question: {question}
Expected answer: {ground_truth}
Chunk: {ctx[:400]}

Respond with ONLY yes or no."""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5
        )

        answer = response.choices[0].message.content.strip().lower()
        if "yes" in answer:
            useful += 1
        time.sleep(0.3)

    return useful / len(contexts) if contexts else 0.0

# ── METRIC 4: CONTEXT RECALL ──────────────────────────────────
# Did retrieval find the chunks needed to answer correctly?

def measure_context_recall(contexts: list, ground_truth: str) -> float:
    context_text = "\n".join([f"[{i+1}]: {c[:400]}" for i, c in enumerate(contexts)])

    prompt = f"""You are checking if retrieved chunks contain enough information to produce a correct answer.

Expected correct answer: {ground_truth}

Retrieved chunks:
{context_text}

Score from 0.0 to 1.0 where:
1.0 = the chunks contain all information needed to produce the correct answer
0.5 = the chunks contain some relevant information
0.0 = the chunks do not contain the information needed

Respond with ONLY a number between 0.0 and 1.0. No explanation."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10
    )

    try:
        return float(response.choices[0].message.content.strip())
    except Exception:
        return 0.5

# ── COLLECT PIPELINE OUTPUTS ──────────────────────────────────

def collect_and_evaluate(golden_dataset, bm25, chunks, vs):
    results = []

    print(f"Evaluating {len(golden_dataset)} questions...")
    print("-" * 55)

    for i, item in enumerate(golden_dataset):
        query        = item['question']
        ground_truth = item['ground_truth']

        print(f"[{i+1}/{len(golden_dataset)}] {query[:55]}")

        result           = run_pipeline(query)
        hybrid_results   = hybrid_retrieve(query, bm25, chunks, vs, k=10)
        reranked_results = rerank(query, hybrid_results, top_k=5)
        chunk_texts      = [c['text'] for c in reranked_results]
        answer           = result['answer']

        faithfulness    = measure_faithfulness(answer, chunk_texts)
        relevancy       = measure_answer_relevancy(query, answer)
        precision       = measure_context_precision(query, chunk_texts, ground_truth)
        recall          = measure_context_recall(chunk_texts, ground_truth)

        results.append({
            "question":          query,
            "answer":            answer,
            "status":            result['status'],
            "faithfulness":      faithfulness,
            "answer_relevancy":  relevancy,
            "context_precision": precision,
            "context_recall":    recall,
        })

        print(f"  faith={faithfulness:.2f} rel={relevancy:.2f} prec={precision:.2f} rec={recall:.2f}")
        time.sleep(1)

    return results

# ── AGGREGATE + SAVE REPORT ───────────────────────────────────

def save_report(results, output_path: str):
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    report = {
        m: round(sum(r[m] for r in results) / len(results), 4)
        for m in metrics
    }

    report["total_questions"] = len(results)
    report["refused_count"]   = sum(1 for r in results if r["status"] == "REFUSED")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    return report

# ── PRINT RESULTS ─────────────────────────────────────────────

def print_results(report: dict):
    threshold = 0.75
    passed    = 0
    metrics   = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    print("\n" + "="*55)
    print("EVALUATION RESULTS")
    print("="*55)
    print(f"{'Metric':<30} {'Score':<10} {'Status'}")
    print("-"*55)

    for metric in metrics:
        score  = report[metric]
        status = "PASS" if score >= threshold else "FAIL"
        if score >= threshold:
            passed += 1
        print(f"{metric:<30} {score:.4f}     {status}")

    print("="*55)
    print(f"Questions evaluated: {report['total_questions']}")
    print(f"Refused answers:     {report['refused_count']}")
    print(f"Metrics passing ({threshold}): {passed}/{len(metrics)}")
    print("\nReport saved to eval/ragas_report.json")

# ── MAIN ──────────────────────────────────────────────────────

if __name__ == "__main__":
    bm25, chunks = load_bm25_index()
    vs           = load_vectorstore()

    golden  = load_golden_dataset()
    results = collect_and_evaluate(golden, bm25, chunks, vs)
    report  = save_report(results, "eval/ragas_report.json")

    print_results(report)