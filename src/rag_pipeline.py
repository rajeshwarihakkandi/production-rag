import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import time
import yaml

from bm25_retriever import load_bm25_index
from embeddings import load_vectorstore
from hybrid_retriever import hybrid_retrieve
from reranker import rerank
from citation_enforcer import answer_with_enforcement, load_config

# ── LOAD EVERYTHING ONCE ──────────────────────────────────────

print("Initializing RAG pipeline...")
config       = load_config()
bm25, chunks = load_bm25_index()
vs           = load_vectorstore()
print("Pipeline ready.\n")


# ── FULL PIPELINE ─────────────────────────────────────────────

def run_pipeline(query: str, verbose: bool = False) -> dict:
    start = time.time()

    # Step 1: hybrid retrieval
    hybrid_results   = hybrid_retrieve(query, bm25, chunks, vs,
                                       k=config['retrieval']['k'])

    # Step 2: re-rank
    reranked_results = rerank(query, hybrid_results,
                              top_k=config['retrieval']['top_k_reranked'])

    # Step 3: generate cited answer with enforcement
    result = answer_with_enforcement(query, reranked_results)

    result['latency_seconds'] = round(time.time() - start, 2)

    if verbose:
        print(f"Hybrid results:  {len(hybrid_results)} chunks")
        print(f"Reranked to:     {len(reranked_results)} chunks")
        print(f"Status:          {result['status']}")
        print(f"Latency:         {result['latency_seconds']}s")

    return result


def print_result(result: dict):
    print(f"\n{'='*65}")
    print(f"QUERY:   {result['query']}")
    print(f"STATUS:  {result['status']}")
    print(f"LATENCY: {result['latency_seconds']}s")
    print(f"SOURCES: {', '.join(result['sources'])}")
    print(f"\nANSWER:\n{result['answer']}")
    print(f"{'='*65}")


# ── EDGE CASE TESTS ───────────────────────────────────────────

def run_edge_case_tests():
    edge_cases = [
        # normal queries
        ("What is retrieval augmented generation?",          "normal"),
        ("How does multi-head attention work?",              "normal"),
        # out of domain — should refuse
        ("What is the capital of France?",                   "out_of_domain"),
        ("Who won the FIFA World Cup in 2022?",              "out_of_domain"),
        # very short query
        ("RAG",                                              "short"),
        # very long query
        ("Can you explain in detail how the transformer architecture uses self attention mechanisms and positional encoding to process sequential data without recurrence?", "long"),
    ]

    print(f"\n{'='*65}")
    print("EDGE CASE TEST RESULTS")
    print(f"{'='*65}")
    print(f"{'Query':<50} {'Status':<12} {'Latency'}")
    print(f"{'-'*65}")

    passed = 0
    for query, query_type in edge_cases:
        result = run_pipeline(query)
        latency = result['latency_seconds']

        # out_of_domain should be refused
        if query_type == "out_of_domain":
            correct = result['status'] == "REFUSED" or "cannot find" in result['answer'].lower()
        else:
            correct = True  # just check it doesn't crash

        status_mark = "PASS" if correct else "FAIL"
        if correct:
            passed += 1

        short_query = query[:48] + ".." if len(query) > 48 else query
        print(f"{short_query:<50} {status_mark} {result['status']:<10} {latency}s")

    print(f"\nEdge case score: {passed}/{len(edge_cases)}")


# ── MAIN ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # run a normal query with full output
    result = run_pipeline("What is retrieval augmented generation?", verbose=True)
    print_result(result)

    result = run_pipeline("How does BERT use bidirectional training?", verbose=True)
    print_result(result)

    # run edge case tests
    run_edge_case_tests()