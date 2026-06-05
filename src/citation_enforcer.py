from dotenv import load_dotenv
from groq import Groq
import yaml
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))


load_dotenv()

# ── LOAD CONFIG ───────────────────────────────────────────────


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


config = load_config()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── STEP 1: GENERATE ANSWER WITH CITATION PROMPT ──────────────

def generate_cited_answer(query: str, chunks: list) -> str:
    """
    Generates an answer using only the provided chunks.
    Instructs the LLM to cite chunk numbers for every claim.
    """
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n[Chunk {i+1}] Source: {chunk['source_file']} p.{chunk['page_number']}\n"
        context += chunk['text'] + "\n"

    messages = [
        {"role": "system", "content": config['prompts']['system']},
        {"role": "user",   "content": f"Question: {query}\n\nContext:\n{context}"}
    ]

    response = client.chat.completions.create(
        model=config['models']['llm'],
        messages=messages,
        temperature=0.1,
        max_tokens=1000
    )

    return response.choices[0].message.content


# ── STEP 2: CHECK IF ANSWER IS GROUNDED ───────────────────────
def answer_with_enforcement(query: str, chunks: list) -> dict:
    """
    Single LLM call that generates AND self-checks grounding.
    Faster than two separate calls.
    """
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n[Chunk {i+1}] Source: {chunk['source_file']} p.{chunk['page_number']}\n"
        context += chunk['text'] + "\n"

    system_prompt = """You are a research assistant with strict grounding rules.

RULES:
1. Answer using ONLY the context chunks provided. Do not use outside knowledge.
2. Cite every claim with [Chunk X].
3. If the chunks do not contain relevant information to answer the question, you MUST respond with exactly: "INSUFFICIENT_CONTEXT"
4. If the question is about topics not covered in the chunks (geography, sports, general knowledge etc), respond with: "INSUFFICIENT_CONTEXT"

Respond with either a cited answer or the exact string "INSUFFICIENT_CONTEXT". Nothing else."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Question: {query}\n\nContext:\n{context}"}
    ]

    response = client.chat.completions.create(
        model=config['models']['llm'],
        messages=messages,
        temperature=0.1,
        max_tokens=1000
    )

    answer = response.choices[0].message.content.strip()

    if "INSUFFICIENT_CONTEXT" in answer:
        return {
            "query":    query,
            "answer":   "I cannot find sufficient information in the provided documents to answer this question.",
            "status":   "REFUSED",
            "grounded": False,
            "unsupported": [],
            "sources":  list(set([f"{c['source_file']} p.{c['page_number']}" for c in chunks]))
        }

    return {
        "query":    query,
        "answer":   answer,
        "status":   "GROUNDED",
        "grounded": True,
        "unsupported": [],
        "sources":  list(set([f"{c['source_file']} p.{c['page_number']}" for c in chunks]))
    }

# ── MAIN ──────────────────────────────────────────────────────


if __name__ == "__main__":
    from bm25_retriever import load_bm25_index
    from embeddings import load_vectorstore
    from hybrid_retriever import hybrid_retrieve
    from reranker import rerank

    bm25, chunks = load_bm25_index()
    vs = load_vectorstore()

    test_queries = [
        "What is retrieval augmented generation?",
        "How does multi-head attention work in transformers?",
        # this should get refused — not in any paper
        "What is the boiling point of mercury?",
    ]

    for query in test_queries:
        print(f"\n{'='*65}")
        print(f"QUERY: {query}")
        print(f"{'='*65}")

        hybrid_results = hybrid_retrieve(query, bm25, chunks, vs, k=10)
        reranked_results = rerank(query, hybrid_results, top_k=5)
        result = answer_with_enforcement(query, reranked_results)

        print(f"Status: {result['status']}")
        print(f"Sources: {result['sources']}")
        print(f"\nAnswer:\n{result['answer']}")
