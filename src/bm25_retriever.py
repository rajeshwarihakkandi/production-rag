import os
import json
import pickle
from rank_bm25 import BM25Okapi

# ── SETTINGS ──────────────────────────────────────────────
CHUNKS_PATH = os.path.join("data", "chunks", "all_chunks.json")
BM25_INDEX_PATH = os.path.join("data", "bm25_index.pkl")
# .pkl = pickle file — saves Python objects directly to disk
# so we don't have to rebuild the index every time

# ── STEP 1: TOKENIZER ─────────────────────────────────────


def tokenize(text):
    # BM25 works on lists of words, not raw text
    # so we split text into individual words first
    # lowercasing ensures "Attention" and "attention" match
    return text.lower().split()

# ── STEP 2: BUILD BM25 INDEX ──────────────────────────────


def build_bm25_index(chunks_path=CHUNKS_PATH):
    print("Loading chunks...")
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  Loaded {len(chunks)} chunks")

    print("Tokenizing chunks...")
    # tokenize every chunk's text into a list of words
    tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_chunks)
    # BM25Okapi is the most popular variant of BM25
    # It takes a list of tokenized documents and builds
    # an inverted index — a map of word → which chunks contain it

    # save the index + original chunks to disk
    print("Saving index to disk...")
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({
            "bm25":   bm25,
            "chunks": chunks
        }, f)

    print(f"  BM25 index saved to: {BM25_INDEX_PATH}")
    print(f"  Index covers {len(chunks)} chunks")
    return bm25, chunks

# ── STEP 3: LOAD EXISTING INDEX ───────────────────────────


def load_bm25_index():
    if not os.path.exists(BM25_INDEX_PATH):
        print("No existing index found — building fresh...")
        return build_bm25_index()

    print("Loading BM25 index from disk...")
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"  Loaded index covering {len(data['chunks'])} chunks")
    return data["bm25"], data["chunks"]

# ── STEP 4: BM25 RETRIEVE ─────────────────────────────────


def bm25_retrieve(query, bm25, chunks, k=5):
    # tokenize the query the same way we tokenized chunks
    tokenized_query = tokenize(query)

    # get BM25 scores for every chunk
    scores = bm25.get_scores(tokenized_query)
    # scores is a list of numbers — one score per chunk
    # higher score = more keyword overlap with the query

    # get the indices of top k chunks sorted by score
    top_k_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    # return the actual chunks with their scores attached
    results = []
    for idx in top_k_indices:
        chunk = chunks[idx].copy()
        chunk["bm25_score"] = round(float(scores[idx]), 4)
        results.append(chunk)

    return results

# ── STEP 5: COMPARE BM25 VS VECTOR ───────────────────────


def compare_results(query, bm25_results, vector_results):
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")

    print(f"\n── BM25 Results (keyword search) ──")
    for i, chunk in enumerate(bm25_results):
        print(f"  {i+1}. [{chunk['bm25_score']}] "
              f"{chunk['source_file']} p.{chunk['page_number']}")
        print(f"     {chunk['text'][:120]}...")

    print(f"\n── Vector Results (semantic search) ──")
    for i, doc in enumerate(vector_results):
        print(f"  {i+1}. {doc.metadata['source_file']} "
              f"p.{doc.metadata['page_number']}")
        print(f"     {doc.page_content[:120]}...")

    # find chunks that appear in both results
    bm25_sources = set(
        f"{c['source_file']}_{c['page_number']}"
        for c in bm25_results
    )
    vector_sources = set(
        f"{d.metadata['source_file']}_{d.metadata['page_number']}"
        for d in vector_results
    )
    overlap = bm25_sources & vector_sources

    print(f"\n── Overlap (in both results) ──")
    if overlap:
        for o in overlap:
            print(f"  ✓ {o.replace('_', ' p.')}")
    else:
        print("  No overlap — BM25 and vector found different chunks")
        print("  This is why we need BOTH — they complement each other!")


# ── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    from embeddings import load_vectorstore

    # build or load BM25 index
    bm25, chunks = load_bm25_index()

    # load vector store
    print("\nLoading vectorstore...")
    vs = load_vectorstore()

    # test queries — mix of semantic and keyword-heavy
    test_queries = [
        "What is retrieval augmented generation?",
        "Vaswani multi-head attention transformer architecture",
        "chain of thought reasoning step by step",
    ]

    for query in test_queries:
        # BM25 results
        bm25_results = bm25_retrieve(query, bm25, chunks, k=5)

        # Vector results
        vector_results = vs.similarity_search(query, k=5)

        # compare side by side
        compare_results(query, bm25_results, vector_results)
