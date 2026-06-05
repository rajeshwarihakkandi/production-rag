<div align="center">

# Production-Grade RAG System

### Bridging the gap between a RAG demo and a RAG system that actually works in production

[![CI](https://github.com/rajeshwarihakkandi/production-rag/actions/workflows/eval.yml/badge.svg)](https://github.com/rajeshwarihakkandi/production-rag/actions)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-latest-orange)
![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.1-purple)

</div>

---

## What This Project Is

Most RAG tutorials show you how to ask questions to a PDF. This project goes further. It builds a system that retrieves accurately, refuses to hallucinate, measures its own quality, and enforces that quality on every code change through a CI pipeline.

Built on **20 AI research papers** (~3,700 chunks), this system answers questions with exact citations back to the source document and page number. If the retrieved documents do not support an answer, the system says so rather than making something up.

---

## Evaluation Results

| Metric | Score | Threshold | Status |
|:---|:---:|:---:|:---:|
| Faithfulness | **0.80** | 0.75 | ✅ PASS |
| Context Recall | **0.83** | 0.75 | ✅ PASS |
| Answer Relevancy | 0.68 | 0.75 | ❌ FAIL |
| Context Precision | 0.48 | 0.75 | ❌ FAIL |

> Evaluated on a manually curated 10-question golden dataset. Faithfulness and context recall are the two critical metrics enforced by the CI pipeline.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                          USER QUERY                          │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID RETRIEVAL                          │
│                                                              │
│   ┌─────────────────────┐    ┌──────────────────────────┐   │
│   │   BM25 Keyword      │    │   Vector Semantic        │   │
│   │   Search            │    │   Search (ChromaDB)      │   │
│   │                     │    │                          │   │
│   │  Great for exact    │    │  Great for meaning       │   │
│   │  terms and phrases  │    │  and intent              │   │
│   └──────────┬──────────┘    └────────────┬─────────────┘   │
│              │                            │                  │
│              └────────────┬───────────────┘                  │
│                           ▼                                  │
│              Reciprocal Rank Fusion (RRF)                    │
│              score = 1 / (60 + rank)                         │
│              Top-10 fused results                            │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   CROSS-ENCODER RE-RANKER                    │
│                                                              │
│   Model: cross-encoder/ms-marco-MiniLM-L-6-v2               │
│   Scores each (query, chunk) pair together                   │
│   Returns top-5 after re-ranking                             │
│   Latency: under 1.5 seconds                                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   CITATION ENFORCEMENT                       │
│                                                              │
│   LLM answers using ONLY the retrieved chunks                │
│   Every claim must cite [Chunk X]                            │
│   If chunks do not support the answer:                       │
│   → Returns INSUFFICIENT_CONTEXT (no hallucination)          │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            CITED ANSWER + SOURCE FILE + PAGE NUMBER          │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
production-rag/
│
├── 📁 src/
│   ├── chunker.py              # PDF/MD/web ingestion and chunking
│   ├── embeddings.py           # Embedding model + ChromaDB setup
│   ├── bm25_retriever.py       # BM25 keyword index
│   ├── hybrid_retriever.py     # RRF fusion of BM25 + vector results
│   ├── reranker.py             # Cross-encoder re-ranking
│   ├── citation_enforcer.py    # Grounded answer generation
│   └── rag_pipeline.py         # Full end-to-end pipeline entry point
│
├── 📁 eval/
│   ├── golden_dataset.json     # 10 manually verified QA pairs
│   ├── evaluate.py             # Evaluation script
│   └── ragas_report.json       # Latest evaluation scores
│
├── 📁 data/
│   ├── raw/                    # 20 AI research papers (PDFs)
│   ├── chunks/                 # 3,736 chunks with metadata
│   └── bm25_index.pkl          # Persisted BM25 index
│
├── 📁 vectorstore/             # ChromaDB persisted vector store
├── 📁 .github/workflows/
│   └── eval.yml                # CI pipeline quality gate
│
└── config.yaml                 # All settings and prompts versioned here
```

---

## How It Was Built

### Phase 1 — Foundation

**Goal:** Ingest documents, chunk them, store embeddings, retrieve and answer with citations.

- Loaded 20 AI research papers using LangChain's `PyPDFLoader`
- Chunked using `RecursiveCharacterTextSplitter` — `chunk_size=700`, `chunk_overlap=100`
- Each chunk tagged with `source_file`, `page_number`, `chunk_index`
- Embedded with `all-MiniLM-L6-v2` (local, free, no API needed)
- Stored 3,736 vectors in ChromaDB persisted to disk
- Built retrieval pipeline generating cited answers via Groq (LLaMA 3.1 8B)

> **Why overlap matters:** Without overlap, an important sentence cut at a chunk boundary loses context. 100-token overlap ensures continuity.

---

### Phase 2 — Production Quality

**Goal:** Improve retrieval precision, prevent hallucination, version all configuration.

**Hybrid Retrieval**
- BM25 handles exact keyword and phrase matching
- Vector search captures semantic meaning and intent
- RRF combines both — chunks in both results get a consensus boost

**Re-Ranking**
- Top-10 hybrid results passed to a cross-encoder
- Model reads query and chunk together as a pair and scores relevance
- Returns top-5 after re-ranking — consistently outperforms hybrid alone

**Citation Enforcement**
- Strict system prompt forces the LLM to cite every claim
- Out-of-domain queries (geography, sports, science facts) correctly refused
- All prompts stored in `config.yaml` — every change is a git diff

---

### Phase 3 — Measurement

**Goal:** Prove quality with numbers and enforce it automatically.

**Golden Dataset**
- 10 manually written and verified QA pairs
- Covers RAG, transformers, BERT, chain-of-thought, self-consistency, LoftQ

**CI Pipeline**

```
Every push to main
       │
       ▼
GitHub Actions triggers eval.yml
       │
       ▼
Reads eval/ragas_report.json
       │
       ├── faithfulness >= 0.75 ?  ──→ PASS
       │                          ──→ FAIL → build fails, PR blocked
       │
       └── context_recall >= 0.75 ? ──→ PASS
                                    ──→ FAIL → build fails, PR blocked
```

---

## Tech Stack

| Component | Tool |
|:---|:---|
| Orchestration | LangChain |
| Vector Store | ChromaDB |
| Embedding Model | all-MiniLM-L6-v2 (sentence-transformers) |
| Keyword Search | rank-bm25 (BM25Okapi) |
| Re-Ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | LLaMA 3.1 8B via Groq (free tier) |
| Evaluation | Custom evaluator using Groq |
| CI/CD | GitHub Actions |

---

## Setup

### Prerequisites
- Python 3.12
- Groq API key — free at [console.groq.com](https://console.groq.com)

### Install

```bash
git clone https://github.com/rajeshwarihakkandi/production-rag.git
cd production-rag

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

pip install langchain langchain-chroma langchain-huggingface langchain-groq
pip install chromadb sentence-transformers rank-bm25 pypdf
pip install python-dotenv groq pyyaml datasets
```

### Environment

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### Run

```bash
# Run the full pipeline
python src/rag_pipeline.py

# Run evaluation
python eval/evaluate.py
```

---

## Example Output

**Grounded answer:**
```
QUERY:   What is retrieval augmented generation?
STATUS:  GROUNDED
LATENCY: 2.5s
SOURCES: rag_survey.pdf p.1, rag_survey.pdf p.11

ANSWER:
Retrieval-Augmented Generation (RAG) has emerged as a promising solution
by incorporating knowledge from external databases, enhancing the accuracy
and credibility of generation [Chunk 1]. RAG addresses challenges faced by
LLMs such as hallucination, outdated knowledge, and non-transparent
reasoning processes [Chunk 2].
```

**Out-of-domain refusal:**
```
QUERY:   What is the capital of France?
STATUS:  REFUSED

ANSWER:
I cannot find sufficient information in the provided documents
to answer this question.
```

---

## Key Design Decisions

**Why Groq instead of OpenAI?**
Groq's free tier works without billing setup and has no India region restrictions on the free plan. LLaMA 3.1 8B via Groq gives fast, reliable responses at zero cost.

**Why local embeddings?**
`all-MiniLM-L6-v2` runs entirely on your machine. No API calls, no cost, no rate limits for embedding 3,736 chunks.

**Why RRF over weighted scoring?**
RRF uses rank position rather than raw scores, making it robust across different score scales from BM25 and vector search. The constant k=60 is proven to work well across domains.

**Why a single LLM call for citation enforcement?**
Early versions used two calls — one to generate, one to verify. This doubled latency to 20+ seconds. The single-call approach with `INSUFFICIENT_CONTEXT` as a structured refusal cut latency to 2 to 3 seconds with the same quality.

---

## Author

**Basava Rajeshwari M Hakkandi**
B.Tech Computer Engineering (AI & ML) — Presidency University, Bengaluru

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/rajeshwari-hakkandi)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/rajeshwarihakkandi)
