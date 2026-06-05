from embeddings import load_vectorstore
from bm25_retriever import load_bm25_index, bm25_retrieve
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))


# ── SETTINGS ──────────────────────────────────────────────────
RRF_K = 60

# ── STEP 1: RRF ALGORITHM ─────────────────────────────────────


def reciprocal_rank_fusion(bm25_results, vector_results, k=RRF_K):
    scores = {}

    # score BM25 results
    for rank, chunk in enumerate(bm25_results):
        chunk_id = f"{chunk['source_file']}_p{chunk['page_number']}_c{chunk['chunk_index']}"
        if chunk_id not in scores:
            scores[chunk_id] = {
                "rrf_score":   0.0,
                "in_bm25":     False,
                "in_vector":   False,
                "bm25_rank":   None,
                "vector_rank": None,
                "text":        chunk["text"],
                "source_file": chunk["source_file"],
                "page_number": chunk["page_number"],
            }
        scores[chunk_id]["rrf_score"] += 1.0 / (k + rank + 1)
        scores[chunk_id]["in_bm25"] = True
        scores[chunk_id]["bm25_rank"] = rank + 1

    # score vector results
    for rank, doc in enumerate(vector_results):
        chunk_id = (
            f"{doc.metadata['source_file']}_"
            f"p{doc.metadata['page_number']}_"
            f"c{doc.metadata.get('chunk_index', 0)}"
        )
        if chunk_id not in scores:
            scores[chunk_id] = {
                "rrf_score":   0.0,
                "in_bm25":     False,
                "in_vector":   False,
                "bm25_rank":   None,
                "vector_rank": None,
                "text":        doc.page_content,
                "source_file": doc.metadata["source_file"],
                "page_number": doc.metadata["page_number"],
            }
        scores[chunk_id]["rrf_score"] += 1.0 / (k + rank + 1)
        scores[chunk_id]["in_vector"] = True
        scores[chunk_id]["vector_rank"] = rank + 1

    ranked = sorted(scores.values(),
                    key=lambda x: x["rrf_score"], reverse=True)
    return ranked


# ── STEP 2: HYBRID RETRIEVE ───────────────────────────────────

def hybrid_retrieve(query, bm25, chunks, vectorstore, k=10):
    bm25_results = bm25_retrieve(query, bm25, chunks, k=k)
    vector_results = vectorstore.similarity_search(query, k=k)
    fused = reciprocal_rank_fusion(bm25_results, vector_results)
    return fused


# ── STEP 3: PRETTY PRINT ──────────────────────────────────────

def print_hybrid_results(query, results, top_n=5):
    print(f"\n{'='*65}")
    print(f"QUERY: {query}")
    print(f"{'='*65}")
    print(f"{'Rank':<5} {'RRF Score':<12} {'BM25':<6} {'Vec':<5} {'Source'}")
    print(f"{'-'*65}")

    for i, r in enumerate(results[:top_n]):
        bm25_rank = f"#{r['bm25_rank']}" if r['in_bm25'] else "-"
        vec_rank = f"#{r['vector_rank']}" if r['in_vector'] else "-"
        source = f"{r['source_file']} p.{r['page_number']}"
        print(
            f"{i+1:<5} {r['rrf_score']:.6f}   {bm25_rank:<6} {vec_rank:<5} {source}")

    print(f"\nTop result preview:")
    print(f"  {results[0]['text'][:200]}...")


# ── STEP 4: VALIDATE QUALITY ──────────────────────────────────

def validate_quality(bm25, chunks, vectorstore):
    test_queries = [
        ("how does self attention work",           "attention_is_all_you_need"),
        ("what is retrieval augmented generation", "rag"),
        ("chain of thought step by step",          "chain_of_thought"),
        ("Vaswani transformer architecture 2017",  "attention_is_all_you_need"),
        ("BERT bidirectional language model",      "bert"),
    ]

    print(f"\n{'='*65}")
    print("QUALITY VALIDATION")
    print(f"{'='*65}")

    passed = 0
    for query, expected in test_queries:
        results = hybrid_retrieve(query, bm25, chunks, vectorstore, k=10)
        top_sources = [r['source_file'] for r in results[:3]]
        hit = any(expected in src for src in top_sources)
        status = "PASS" if hit else "FAIL"
        top_result = results[0]['source_file'] if results else "no results"
        if hit:
            passed += 1
        print(f"{status} | {query[:45]:<45} -> {top_result}")

    print(f"\nScore: {passed}/5 queries returned expected paper in top 3")


# ── MAIN ──────────────────────────────────────────────────────

if __name__ == "__main__":
    bm25, chunks = load_bm25_index()
    vs = load_vectorstore()

    test_queries = [
        "What is retrieval augmented generation?",
        "Vaswani multi-head attention transformer architecture",
        "chain of thought reasoning step by step",
    ]

    print("\n-- Hybrid RRF Results --")
    for query in test_queries:
        results = hybrid_retrieve(query, bm25, chunks, vs, k=10)
        print_hybrid_results(query, results, top_n=5)

    validate_quality(bm25, chunks, vs)
