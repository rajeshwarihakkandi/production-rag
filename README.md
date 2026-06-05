Production-Grade Retrieval Augmented Generation (RAG) System

Bridging the gap between a RAG demo and a RAG system that actually works in production.


What This Project Is
Most RAG tutorials show you how to ask questions to a PDF. This project goes further — it builds a system that retrieves accurately, refuses to hallucinate, measures its own quality, and enforces that quality on every code change through a CI pipeline.
Built on 20 AI research papers (~3,700 chunks), this system answers questions with exact citations back to the source document and page number. If the retrieved documents don't support an answer, the system says so rather than making something up.

Results
MetricScoreStatusFaithfulness0.80PASSContext Recall0.83PASSAnswer Relevancy0.68—Context Precision0.48—
Evaluated on a manually curated 10-question golden dataset. Faithfulness and context recall are the critical metrics enforced by the CI pipeline.

Architecture
Query
  │
  ▼
┌─────────────────────────────────────┐
│         Hybrid Retrieval            │
│  BM25 keyword search (rank-bm25)    │
│       +                             │
│  Vector semantic search (ChromaDB)  │
│       =                             │
│  Reciprocal Rank Fusion (RRF)       │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│         Cross-Encoder Re-Ranker     │
│  ms-marco-MiniLM-L-6-v2             │
│  Re-scores top-10 as (query,chunk)  │
│  pairs → returns top-5              │
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│         Citation Enforcement        │
│  LLM answers using ONLY chunks      │
│  Cites [Chunk X] for every claim    │
│  Returns INSUFFICIENT_CONTEXT       │
│  if chunks don't support the answer │
└─────────────────────────────────────┘
  │
  ▼
Cited Answer with Source + Page Number

Project Structure
production-rag/
│
├── src/
│   ├── chunker.py            # PDF/MD/web ingestion + chunking
│   ├── embeddings.py         # Embedding model + ChromaDB setup
│   ├── bm25_retriever.py     # BM25 keyword index
│   ├── hybrid_retriever.py   # RRF fusion of BM25 + vector results
│   ├── reranker.py           # Cross-encoder re-ranking
│   ├── citation_enforcer.py  # Grounded answer generation
│   └── rag_pipeline.py       # Full end-to-end pipeline entry point
│
├── eval/
│   ├── golden_dataset.json   # 10 manually verified QA pairs
│   ├── evaluate.py           # Evaluation script (faithfulness, recall, etc.)
│   └── ragas_report.json     # Latest evaluation scores
│
├── data/
│   ├── raw/                  # 20 AI research papers (PDFs)
│   ├── chunks/               # All 3,736 chunks with metadata
│   └── bm25_index.pkl        # Persisted BM25 index
│
├── vectorstore/              # ChromaDB persisted vector store
├── config.yaml               # All settings + prompts versioned here
└── .github/workflows/
    └── eval.yml              # CI pipeline — enforces quality on every PR

How It Was Built — Phase by Phase
Phase 1: Foundation
Goal: Ingest documents, chunk them, store embeddings, retrieve and answer with citations.

Loaded 20 AI research papers using LangChain's PyPDFLoader
Chunked using RecursiveCharacterTextSplitter with chunk_size=700, chunk_overlap=100
Each chunk tagged with source_file, page_number, chunk_index
Embedded using all-MiniLM-L6-v2 (local, free, 90MB)
Stored 3,736 vectors in ChromaDB (persisted to disk)
Built retrieval pipeline pulling top-k chunks and generating cited answers via Groq (LLaMA 3.1)

Why chunk overlap matters: Without overlap, an important sentence cut at a boundary loses context. 100-token overlap ensures continuity across chunks.

Phase 2: Production Quality
Goal: Improve retrieval precision, prevent hallucination, version all configuration.
Hybrid Retrieval

BM25 excels at exact keyword and phrase matching
Vector search captures semantic meaning and intent
Reciprocal Rank Fusion (RRF) combines both: score = 1 / (60 + rank)
Chunks appearing in both results get a consensus boost

Cross-Encoder Re-Ranker

Takes top-10 hybrid results
Re-scores each (query, chunk) pair together using ms-marco-MiniLM-L-6-v2
Returns top-5 after re-ranking
Latency: under 1.5 seconds for 10-20 chunks

Citation Enforcement

Single LLM call with strict system prompt
LLM must cite [Chunk X] for every claim
If chunks don't contain the answer → returns INSUFFICIENT_CONTEXT rather than hallucinating
Tested with out-of-domain queries (geography, sports) — all correctly refused

Config Versioning

All prompts, model names, thresholds stored in config.yaml
Every prompt change is a git diff — full audit trail of system behavior


Phase 3: Measurement
Goal: Prove quality with numbers and enforce it automatically.
Golden Dataset

10 manually written and verified QA pairs
Covers RAG, transformers, BERT, chain-of-thought, self-consistency, LoftQ, positional encoding

Evaluation Metrics

Faithfulness: Are claims in the answer supported by retrieved chunks?
Answer Relevancy: Does the answer address the question?
Context Precision: Are retrieved chunks actually useful?
Context Recall: Did retrieval find the chunks needed to answer correctly?

CI Pipeline

Every push to main triggers eval.yml via GitHub Actions
Quality gate checks faithfulness ≥ 0.75 and context recall ≥ 0.75
Build fails automatically if scores drop below threshold
Evaluation report saved as a build artifact


Tech Stack
ComponentLibraryOrchestrationLangChainVector storeChromaDBEmbedding modelall-MiniLM-L6-v2 (sentence-transformers)BM25 indexrank-bm25Re-rankercross-encoder/ms-marco-MiniLM-L-6-v2LLMLLaMA 3.1 8B via Groq (free tier)EvaluationCustom evaluator using GroqCI/CDGitHub Actions

Setup and Running Locally
Prerequisites

Python 3.12
Groq API key (free at console.groq.com)

Installation
bashgit clone https://github.com/rajeshwarihakkandi/production-rag.git
cd production-rag

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

pip install langchain langchain-chroma langchain-huggingface langchain-groq
pip install chromadb sentence-transformers rank-bm25 pypdf
pip install python-dotenv groq pyyaml datasets
Environment Setup
Create a .env file in the project root:
GROQ_API_KEY=your_groq_api_key_here
Running the Pipeline
bash# Ask a question
python src/rag_pipeline.py

# Run evaluation
python eval/evaluate.py

Example Output
QUERY:   What is retrieval augmented generation?
STATUS:  GROUNDED
LATENCY: 2.5s
SOURCES: rag_survey.pdf p.1, rag_survey.pdf p.11

ANSWER:
Retrieval-Augmented Generation (RAG) has emerged as a promising
solution by incorporating knowledge from external databases,
enhancing the accuracy and credibility of generation, particularly
for knowledge-intensive tasks [Chunk 1]. RAG addresses challenges
faced by LLMs such as hallucination, outdated knowledge, and
non-transparent reasoning processes [Chunk 2].
QUERY:   What is the capital of France?
STATUS:  REFUSED

ANSWER:
I cannot find sufficient information in the provided documents
to answer this question.

CI Pipeline
Every pull request automatically triggers an evaluation run:
yamlon:
  push:
    branches: [main]
  pull_request:
    branches: [main]
The quality gate checks two critical metrics:
faithfulness    >= 0.75   → PASS
context_recall  >= 0.75   → PASS
If either drops below threshold, the build fails and the PR cannot be merged. This makes quality a hard requirement rather than an afterthought.

Key Design Decisions
Why Groq instead of OpenAI?
Groq's free tier works without billing setup and has no India region restrictions. LLaMA 3.1 8B via Groq gives fast, reliable responses at zero cost.
Why local embeddings?
all-MiniLM-L6-v2 runs entirely on your machine. No API calls, no cost, no rate limits for embedding 3,736 chunks.
Why RRF over weighted scoring?
RRF uses rank position rather than raw scores, making it robust across different score scales from BM25 and vector search. The constant k=60 is proven to work well across domains.
Why a single LLM call for citation enforcement?
Early versions used two calls — one to generate, one to verify. This doubled latency to 20+ seconds. The single-call approach with INSUFFICIENT_CONTEXT as a structured refusal reduced latency to 2-3 seconds with the same quality.

Author
Basava Rajeshwari M Hakkandi