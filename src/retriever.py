import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ── SETTINGS ──────────────────────────────────────────────
VECTORSTORE_DIR = "vectorstore"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
TOP_K = 5

# ── STEP 1: LOAD EXISTING VECTORSTORE ─────────────────────


def load_vectorstore():
    print("Loading vectorstore from disk...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embedding_model
    )
    print("  Vectorstore loaded")
    return vectorstore

# ── STEP 2: RETRIEVE RELEVANT CHUNKS ──────────────────────


def retrieve(vectorstore, query, k=TOP_K):
    results = vectorstore.similarity_search(query, k=k)
    return results

# ── STEP 3: FORMAT CHUNKS FOR THE LLM ─────────────────────


def format_chunks(docs):
    formatted = []
    for i, doc in enumerate(docs):
        chunk_text = (
            f"[Chunk {i+1}] "
            f"Source: {doc.metadata['source_file']}, "
            f"Page: {doc.metadata['page_number']}\n"
            f"{doc.page_content}"
        )
        formatted.append(chunk_text)
    return "\n\n".join(formatted)


# ── STEP 4: THE CITATION PROMPT ───────────────────────────
SYSTEM_PROMPT = """You are a research assistant that answers questions \
strictly based on the provided document chunks.

STRICT RULES you must always follow:
1. Answer ONLY using information from the chunks provided below.
2. For every claim you make, cite the chunk like this: [Chunk X]
3. At the end of your answer, list all sources used:
   Sources:
   - Chunk X: <filename>, Page <number>
4. If the chunks do not contain enough information to answer,
   say exactly: "I cannot answer this question based on the provided documents."
   Do NOT make up or infer anything not present in the chunks.

Here are the retrieved chunks:
{context}"""

HUMAN_PROMPT = "Question: {question}"

# ── STEP 5: GENERATE ANSWER WITH CITATIONS ────────────────


def answer(query, vectorstore):
    docs = retrieve(vectorstore, query)
    context = format_chunks(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  HUMAN_PROMPT),
    ])

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    chain = prompt | llm
    response = chain.invoke({
        "context":  context,
        "question": query,
    })

    return response.content, docs

# ── STEP 6: PRETTY PRINT THE RESULT ──────────────────────


def print_result(query, answer_text, docs):
    print("\n" + "="*60)
    print(f"QUESTION:\n{query}")
    print("="*60)
    print(f"\nANSWER:\n{answer_text}")
    print("\n" + "-"*60)
    print("RETRIEVED CHUNKS USED:")
    for i, doc in enumerate(docs):
        print(f"  Chunk {i+1}: {doc.metadata['source_file']} "
              f"(page {doc.metadata['page_number']})")
    print("="*60)


# ── MAIN: TEST WITH 3 QUESTIONS ───────────────────────────
if __name__ == "__main__":
    vs = load_vectorstore()

    questions = [
        "What is the core idea behind retrieval augmented generation?",
        "How does the attention mechanism work in transformers?",
        "What makes chain of thought prompting effective?",
    ]

    for question in questions:
        answer_text, docs = answer(question, vs)
        print_result(question, answer_text, docs)
        print("\n")
