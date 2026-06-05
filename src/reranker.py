from hybrid_retriever import hybrid_retrieve
from embeddings import load_vectorstore
from bm25_retriever import load_bm25_index, bm25_retrieve
from sentence_transformers import CrossEncoder
import time
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))


# ── LOAD MODEL ────────────────────────────────────────────────
# Downloads once (~85MB), cached locally after that
print("Loading cross-encoder model...")
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("  Model loaded")


# ── RERANK ────────────────────────────────────────────────────

def rerank(query: str, chunks: list, top_k: int = 5) -> list:
    """
    Takes hybrid retrieval results and re-scores each (query, chunk) pair.
    Cross-encoder reads both together, much more accurate than vector similarity alone.
    Returns top_k chunks sorted by rerank score.
    """
    start = time.time()

    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = model.predict(pairs)

    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    elapsed = time.time() - start
    print(f"  Re-ranker: scored {len(chunks)} chunks in {elapsed:.2f}s")

    return reranked[:top_k]


# ── PRETTY PRINT ──────────────────────────────────────────────

def print_reranked_results(query, results):
    print(f"\n{'='*65}")
    print(f"QUERY: {query}")
    print(f"{'='*65}")
    print(f"{'Rank':<5} {'Rerank Score':<15} {'Source'}")
    print(f"{'-'*65}")

    for i, r in enumerate(results):
        source = f"{r['source_file']} p.{r['page_number']}"
        print(f"{i+1:<5} {r['rerank_score']:<15.4f} {source}")

    print(f"\nTop result preview:")
    print(f"  {results[0]['text'][:200]}...")


# ── VALIDATE: COMPARE HYBRID VS RERANKED ─────────────────────

def validate_reranker(bm25, chunks, vectorstore):
    test_queries = [
        ("what is retrieval augmented generation", "rag"),
        ("how does self attention work",           "attention_is_all_you_need"),
        ("chain of thought step by step",          "self_consistency"),
        ("Vaswani transformer architecture 2017",  "attention_is_all_you_need"),
        ("BERT bidirectional language model",      "bert"),
    ]

    print(f"\n{'='*65}")
    print("RERANKER VALIDATION — Hybrid vs Reranked (top-1 accuracy)")
    print(f"{'='*65}")
    print(f"{'Query':<45} {'Hybrid #1':<30} {'Reranked #1'}")
    print(f"{'-'*65}")

    hybrid_score = 0
    reranked_score = 0

    for query, expected in test_queries:
        hybrid_results = hybrid_retrieve(
            query, bm25, chunks, vectorstore, k=10)
        reranked_results = rerank(query, hybrid_results, top_k=5)

        hybrid_top = hybrid_results[0]['source_file']
        reranked_top = reranked_results[0]['source_file']

        hybrid_hit = expected in hybrid_top
        reranked_hit = expected in reranked_top

        if hybrid_hit:
            hybrid_score += 1
        if reranked_hit:
            reranked_score += 1

        h_mark = "PASS" if hybrid_hit else "FAIL"
        r_mark = "PASS" if reranked_hit else "FAIL"

        print(
            f"{query[:44]:<45} {h_mark} {hybrid_top[:25]:<28} {r_mark} {reranked_top}")

    print(f"\nHybrid top-1 score:   {hybrid_score}/5")
    print(f"Reranked top-1 score: {reranked_score}/5")


# ── MAIN ──────────────────────────────────────────────────────

if __name__ == "__main__":
    bm25, chunks = load_bm25_index()
    vs = load_vectorstore()

    test_queries = [
        "What is retrieval augmented generation?",
        "Vaswani multi-head attention transformer architecture",
        "chain of thought reasoning step by step",
    ]

    print("\n-- Reranked Results --")
    for query in test_queries:
        hybrid_results = hybrid_retrieve(query, bm25, chunks, vs, k=10)
        reranked_results = rerank(query, hybrid_results, top_k=5)
        print_reranked_results(query, reranked_results)

    validate_reranker(bm25, chunks, vs)
