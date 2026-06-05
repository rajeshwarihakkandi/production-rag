import os
import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ── SETTINGS ──────────────────────────────────────────────
CHUNKS_PATH = os.path.join("data", "chunks", "all_chunks.json")
VECTORSTORE_DIR = "vectorstore"
EMBED_MODEL = "all-MiniLM-L6-v2"
# all-MiniLM-L6-v2 is a free, fast, local embedding model
# runs entirely on your machine — no API key needed
# converts text into 384 numbers that capture meaning

# ── STEP 1: LOAD ALL CHUNKS ───────────────────────────────


def load_chunks(path):
    print("Loading chunks from JSON...")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"  Loaded {len(raw)} chunks")
    return raw

# ── STEP 2: CONVERT TO LANGCHAIN DOCUMENTS ───────────────


def to_documents(chunks):
    print("Converting chunks to LangChain Documents...")
    docs = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata={
                "source_file":  chunk["source_file"],
                "page_number":  chunk["page_number"],
                "chunk_index":  chunk["chunk_index"],
                "total_pages":  chunk["total_pages"],
            }
        )
        docs.append(doc)
    print(f"  Converted {len(docs)} documents")
    return docs

# ── STEP 3: CREATE EMBEDDINGS + STORE IN CHROMADB ────────


def build_vectorstore(docs):
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    print("  (This downloads ~90MB the first time — please wait...)")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    print(f"\nEmbedding {len(docs)} chunks and storing in ChromaDB...")
    print("  This will take 3–6 minutes for 3,736 chunks...")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=VECTORSTORE_DIR
    )

    print(f"\nVectorstore built and saved to: {VECTORSTORE_DIR}")
    return vectorstore

# ── STEP 4: SMOKE TEST — SEARCH BY MEANING ───────────────


def test_search(vectorstore):
    print("\n── Smoke test: searching by meaning ──\n")

    test_queries = [
        "how does self attention work in transformers",
        "what is retrieval augmented generation",
        "how do large language models learn from instructions",
    ]

    for query in test_queries:
        print(f"Query: '{query}'")
        results = vectorstore.similarity_search(query, k=3)
        for i, doc in enumerate(results):
            print(f"  Result {i+1}: {doc.metadata['source_file']} "
                  f"(page {doc.metadata['page_number']})")
            print(f"    Preview: {doc.page_content[:120]}...")
        print()

# ── LOAD EXISTING VECTORSTORE (no re-embedding) ───────────


def load_vectorstore():
    print("Loading existing vectorstore from disk...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embedding_model
    )
    print("  Vectorstore loaded")
    return vectorstore


# ── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH)
    docs = to_documents(chunks)
    vs = build_vectorstore(docs)
    test_search(vs)
