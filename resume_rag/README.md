# ResumeRAG

> **Grounded resume screening and candidate ranking with Hybrid RAG, Parent-Child retrieval, CRAG, Self-RAG, and LangGraph orchestration.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Orchestration-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![Chroma](https://img.shields.io/badge/Chroma-Vector%20Store-FF6B35)](https://www.trychroma.com/)
[![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-6B46C1)](https://docs.ragas.io/)

ResumeRAG is a project for answering recruiter questions over a resume corpus and producing **candidate fit scores backed by resume evidence**.

The implementation combines classical information retrieval, vector retrieval, cross-encoder reranking, corrective retrieval, self-verification, semantic caching, and graph-based orchestration.

The core design principle is:

**retrieve precisely → recover context → score candidates → verify evidence → return grounded results**

---

## Why ResumeRAG?

A basic resume search system can retrieve relevant text, but candidate screening needs more than keyword or semantic similarity.

ResumeRAG is designed to address several common RAG failure modes:

- **Semantic retrieval alone can miss exact skills** → hybrid vector + BM25 retrieval.
- **The best small chunk is not always the best generation context** → Parent-Child RAG.
- **Weak retrieval should not automatically become a confident answer** → CRAG-style grading and query rewriting.
- **LLM-generated candidate claims can be unsupported** → Self-RAG-style verification against the full resume.
- **Different recruiter requests require different workflows** → LangGraph intent routing for rank vs. compare.
- **RAG quality should be measurable** → RAGAS evaluation for faithfulness, context precision, and context recall.

---

## Architecture

```text
                         Recruiter Query / JD
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ LangGraph Router│
                         │ rank / compare  │
                         └───────┬─────────┘
                                 │
                    ┌────────────┴─────────────┐
                    │                          │
                 RANK PATH                 COMPARE PATH
                    │                          │
                    ▼                          ▼
              ┌──────────┐              Resolve 2 names
              │   CRAG   │                     │
              │ Retrieve │                     ▼
              │ Grade    │                Self-RAG × 2
              │ Rewrite  │                     │
              └────┬─────┘                     │
                   │                           │
             candidate pool                    │
                   │                           │
                   ▼                           │
          ┌─────────────────┐                  │
          │ Self-RAG Score  │◄─────────────────┘
          │ + Verification │
          └────────┬────────┘
                   │
              bounded loop
                   │
                   ▼
            ┌────────────┐
            │ Rank / Diff│
            └─────┬──────┘
                  │
                  ▼
             Grounded Result

 Retrieval internals
 ─────────────────────────────────────────────────────────

 Query
   │
   ├──────────────► Chroma vector search ─────┐
   │                                          │
   └──────────────► BM25 lexical search ──────┤
                                              ▼
                                      Child candidate pool
                                              │
                                              ▼
                                       Cohere Reranker
                                      (or RRF fallback)
                                              │
                                              ▼
                                         parent_id
                                              │
                                              ▼
                                      Parent DocStore
                                              │
                                              ▼
                                       Parent Documents
```

---

## Core Retrieval Design: Parent-Child RAG

ResumeRAG deliberately separates **retrieval granularity** from **generation context**.

```text
Resume
  │
  ├── Parent P001 ──┬── Child C001
  │                 ├── Child C002
  │                 └── Child C003
  │
  └── Parent P002 ──┬── Child C004
                    └── Child C005
```

### Children

Small chunks are embedded and searched because they provide better retrieval precision.

### Parents

The larger parent chunk is fetched after a child match and becomes the contextual unit returned downstream.

### Storage model

- **Child chunks:** ChromaDB
- **Parent chunks:** LangChain `LocalFileStore` wrapped by `create_kv_docstore`
- **Relationship:** child metadata contains `parent_id`
- **Retrieval pattern:** child retrieval → parent resolution → deduplicated parent context

This separation lets the system optimize the vector index for retrieval while preserving richer context for downstream reasoning.

---

## Retrieval Pipeline

ResumeRAG uses a layered retrieval pipeline rather than relying on a single retriever.

### 1. Dense retrieval

Child chunks are embedded with:

**`BAAI/bge-small-en-v1.5`**

The embedding model runs locally on CPU, so embeddings do not require a hosted embedding API.

### 2. Sparse retrieval

BM25 independently retrieves child chunks using lexical matching.

The implementation normalizes text and tokenizes terms so lexical variations such as `SQL`, `sql`, and punctuation differences can still match.

### 3. Candidate fusion / reranking

The vector and BM25 results are merged and deduplicated, then reranked with **Cohere Rerank** by default.

A **Reciprocal Rank Fusion (RRF)** path is also implemented as a fallback when reranking cost, quota, or latency is a concern.

```text
Query
 │
 ├──► Dense / Chroma ──┐
 │                     │
 └──► BM25 ────────────┤
                       ▼
                 Merge + Deduplicate
                       │
                       ▼
                 Cohere Rerank
                       │
                OR RRF fallback
                       │
                       ▼
                  Top children
                       │
                       ▼
                  parent_id
                       │
                       ▼
                Parent documents
```

---

## Corrective RAG (CRAG)

CRAG is used as a **retrieval quality gate** before candidate scoring.

```text
Query
  │
  ▼
Hybrid Retrieval
  │
  ▼
Grade retrieved documents
  │
  ├── enough relevant docs ──► continue
  │
  └── weak retrieval
          │
          ▼
      Rewrite query
          │
          ▼
    Hybrid retrieval again
          │
          ▼
        Re-grade
          │
     ┌────┴────┐
     │         │
  strong     weak
     │         │
 continue   no_strong_match
```

The query rewriter can expand abbreviations, introduce useful synonyms, and make implicit recruiter requirements more explicit before a second retrieval round.

The current implementation uses a **bounded retry strategy** rather than an unbounded loop.

---

## Self-RAG: Evidence-Grounded Candidate Scoring

The scoring stage is designed around a stronger rule than:

> "The LLM thinks this candidate is a match."

For each candidate, the LLM returns structured output containing:

- an overall **0–100 fit score**
- a concise summary
- **strength** and **gap** points
- a claim for each point
- an **exact verbatim resume quote** supporting each strength

The verifier then checks the evidence in code by normalizing case and whitespace and confirming that the quoted text actually exists in the full resume.

There is intentionally **no fuzzy evidence match** in the current verifier.

### Grounding contract

> **The model proposes evidence; deterministic code decides whether the evidence is actually present.**

If unsupported claims remain, the system performs a bounded regeneration cycle. After retry exhaustion, ungrounded claims are dropped instead of being returned as trusted evidence.

```text
Candidate Resume
       │
       ▼
   LLM Scoring
       │
       ▼
Structured claims + evidence
       │
       ▼
Deterministic verifier
       │
   ┌───┴────┐
   │        │
valid     invalid
   │        │
   ▼        ▼
accept    regenerate
              │
              ▼
         verify again
```

---

## Agentic Orchestration with LangGraph

The agent graph supports two primary workflows:

### Rank

Used for questions such as:

```text
Find the best GenAI engineers with RAG experience.
Rank candidates for this Data Engineer JD.
Who has Kubernetes and Docker experience?
```

Flow:

```text
Router
  ↓
CRAG retrieval
  ↓
Candidate pool
  ↓
Self-RAG score candidate 1
  ↓
Self-RAG score candidate 2
  ↓
...
  ↓
Sort by verified score
```

The graph uses a bounded scoring loop: candidates are scored until the pool is exhausted, then the graph transitions to ranking.

### Compare

Used for a direct two-candidate head-to-head comparison:

```text
Compare Alice vs Bob for this backend role.
```

The router identifies the compare intent, resolves the two names, scores both against the query, and produces a structured comparison containing score, summary, grounded points, and a winner.

---

## Resume Ingestion

The ingestion layer supports:

- **PDF** via `pypdf`
- **DOCX** via `docx2txt`

Each resume receives a stable candidate ID derived from its filename.

The content is then split hierarchically:

```text
Resume
  │
  ▼
Parent chunks
~1000 chars / 200 overlap
  │
  ▼
Child chunks
~200 chars / 20 overlap
```

Each child stores the `parent_id` needed to resolve its larger context later.

```text
PDF / DOCX
    │
    ▼
Document Loader
    │
    ▼
Parent Chunking
    │
    ├──────────────► Parent Store
    │
    ▼
Child Chunking
    │
    ▼
Embedding
    │
    ▼
ChromaDB
```

---

## Candidate Grounding Store

In addition to the parent docstore used by retrieval, the ingestion process maintains candidate-level full text for Self-RAG evidence verification.

This matters because scoring verifies evidence against the **full resume**, not only against a retrieved child chunk.

That gives the system three distinct representations:

```text
Child chunks
   ↓
optimized for retrieval

Parent chunks
   ↓
optimized for contextual retrieval

Full candidate text
   ↓
optimized for evidence verification
```

---

## Semantic Cache Design

The repository contains an explicit cache abstraction:

```text
BaseCache
   │
   └── InMemoryCache
```

Cache entries include:

- original query
- query embedding
- answer
- sources
- timestamp
- knowledge-base version
- hit count

The current backend is an in-memory semantic cache using cosine similarity against normalized BGE embeddings.

Cache validity is controlled by:

- semantic similarity threshold
- TTL
- knowledge-base version

The cache is hidden behind a `BaseCache` interface so another backend can be introduced without changing business logic.

### Knowledge-base versioning

The project computes a deterministic `kb_version` from the resume corpus and indexed child count.

```text
Resume corpus changes
       │
       ▼
New kb_version
       │
       ▼
Old cache entries become stale
```

This prevents answers generated against an older resume corpus from remaining valid after re-indexing.

---

## Evaluation

The project includes a hand-built evaluation dataset covering multiple recruiting scenarios, including positive matching and explicit refusal cases where no matching candidate should be returned.

Example evaluation domains include:

- PyTorch
- Kubernetes + Docker
- LangGraph + RAG
- UiPath
- OpenCV + YOLO
- SQL + statistics
- Spark + Kafka
- Redis
- networking
- Prometheus + Grafana

### RAGAS metrics

The project evaluates:

- **Faithfulness**
- **Context Precision**
- **Context Recall**

Evaluation results are written to CSV for inspection and comparison.

### Evaluation flow

```text
Evaluation Case
      │
      ▼
Hybrid Retrieval
      │
      ▼
LangGraph Agent
      │
      ▼
Candidate scoring
      │
      ▼
Verified response
      │
      ▼
RAGAS
 ┌────┼───────────────┐
 ▼    ▼               ▼
Faithfulness   Context Precision   Context Recall
```

---

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python 3.11+ |
| LLM | Anthropic Claude via `langchain-anthropic` |
| Orchestration | LangChain + LangGraph |
| Dense embeddings | `BAAI/bge-small-en-v1.5` |
| Vector store | ChromaDB |
| Sparse retrieval | BM25 |
| Reranking | Cohere Rerank |
| Fusion fallback | Reciprocal Rank Fusion |
| Parent store | LangChain `LocalFileStore` |
| PDF parsing | `pypdf` |
| DOCX parsing | `docx2txt` |
| Structured outputs | Pydantic |
| Evaluation | RAGAS |
| Observability | LangSmith support |
| Configuration | Pydantic Settings |
| Dependency management | `uv` |
| Testing | pytest |

---

## Project Structure

```text
ResumeRAG/
│
├── resume_rag/
│   ├── cache/
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── kb_version.py
│   │   └── memory.py
│   │
│   ├── components/
│   │   ├── embeddings.py
│   │   ├── llm.py
│   │   ├── reranker.py
│   │   ├── retriever.py
│   │   └── vectorstore.py
│   │
│   ├── crag/
│   │   ├── grader.py
│   │   ├── pipeline.py
│   │   └── rewriter.py
│   │
│   ├── eval/
│   │   ├── dataset.py
│   │   ├── ragas_eval.py
│   │   └── runner.py
│   │
│   ├── graph/
│   │   ├── builder.py
│   │   ├── edges.py
│   │   ├── resolve.py
│   │   ├── router.py
│   │   └── state.py
│   │
│   ├── ingestion/
│   │   ├── candidate_store.py
│   │   ├── chunker.py
│   │   ├── indexer.py
│   │   └── loader.py
│   │
│   ├── selfrag/
│   │   ├── pipeline.py
│   │   ├── scorer.py
│   │   └── verifier.py
│   │
│   ├── config.py
│   ├── logger.py
│   └── _compat.py
│
├── api/
├── data/
│   └── resumes/
├── scripts/
├── tests/
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

### Runtime-generated artifacts

The following are generated during local execution and should generally **not** be committed to source control:

```text
chroma_db/
parent_docstore/
candidate_texts.json
eval_results/
```

They can be recreated from the source resume corpus and ingestion pipeline.

---

## Configuration

Configuration is loaded using Pydantic Settings.

Typical credentials include:

```env
ANTHROPIC_API_KEY=...
COHERE_API_KEY=...
```

Optional LangSmith configuration:

```env
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=resume-rag
LANGSMITH_TRACING=false
```

### Security

Never commit:

```text
.env
API keys
tokens
credentials
private resume data
```

The `.gitignore` is configured to exclude environment files and generated RAG artifacts.

---

## Installation

The project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/RudraMoharana-99/ResumeRAG.git
cd ResumeRAG

uv sync
```

Python **3.11+** is required.

### Verify the environment

Use the project environment through `uv`:

```bash
uv run python -c "import resume_rag; print('resume_rag import: OK')"
```

Verify the retriever:

```bash
uv run python -c "from resume_rag.components.retriever import HybridRetriever; print('retriever import: OK')"
```

Run tests:

```bash
uv run python -m pytest
```

---

## Running the Project

The repository currently contains the underlying ingestion, retrieval, CRAG, Self-RAG, and LangGraph components.

A typical development workflow is:

```text
1. Add PDF/DOCX resumes under data/resumes/
2. Load and parse resumes
3. Create parent and child chunks
4. Index child chunks into ChromaDB
5. Store parent documents in the parent docstore
6. Persist candidate full text for evidence verification
7. Run recruiter queries through the LangGraph agent
8. Evaluate retrieval and generation quality
9. Inspect RAGAS results
```

The project is currently structured primarily as an engineering/reference implementation rather than a polished one-command production service.

---

## Example Queries

### Candidate ranking

```text
Find the top candidates for a GenAI engineer role with LangGraph and RAG experience.
```

### Skill search

```text
Who has experience with Kubernetes, Docker and Prometheus?
```

### Head-to-head comparison

```text
Compare Candidate A vs Candidate B for a backend engineering role.
```

### Negative / refusal case

```text
Find candidates with Solidity and blockchain experience.
```

If the corpus does not support the requirement strongly enough, the corrective retrieval path can return a `no_strong_match` outcome instead of blindly ranking candidates.

---

## Engineering Highlights

### 1. Retrieval and context are deliberately separated

Children optimize retrieval precision.

Parents optimize contextual completeness.

Full candidate text supports evidence verification.

This separation is the central Parent-Child RAG design.

### 2. Retrieval is deliberately layered

Instead of choosing between BM25 and dense search, the system uses both and then applies reranking.

This addresses both lexical and semantic matching patterns common in resumes and job descriptions.

### 3. Grounding is deterministic where it matters

The LLM proposes claims and evidence, but evidence verification is implemented as deterministic application logic.

### 4. Failure paths are explicit

The graph and retrieval pipelines carry statuses such as:

```text
ok
no_strong_match
not_found
```

This reduces the chance that an empty or weak retrieval silently becomes a confident answer.

### 5. Evaluation is part of the architecture

The project includes:

- a corpus-specific evaluation dataset
- expected candidate IDs
- retrieval bookkeeping
- grounded response synthesis
- RAGAS metrics
- CSV evaluation artifacts

---

## Design Principles

### Precision before generation

The system invests significant computation in retrieval and reranking before invoking candidate scoring.

### Evidence before confidence

A candidate score without supporting resume evidence is not treated as a trustworthy result.

### Bounded loops

CRAG and Self-RAG use bounded retry/regeneration behavior instead of unrestricted agent loops.

### Swappable infrastructure

The cache is abstracted behind `BaseCache`, and retrieval components are isolated behind dedicated interfaces/modules.

### Local-first where practical

Embeddings use a local BGE model, while Claude and Cohere are used where hosted model capabilities provide value.

---

## Current Limitations

This repository is an engineering project and is not yet a turnkey production service.

Natural next steps include:

- expose the agent behind a production FastAPI service
- add a clean ingestion CLI
- expand automated tests around retrieval, CRAG, Self-RAG, and graph routing
- replace local parent storage with a production object/document store where required
- add a persistent distributed semantic-cache backend such as Valkey/Redis
- add CI for linting, tests, and evaluation regression checks
- add richer observability around retrieval scores, reranking decisions, grounding failures, and latency
- add dataset/version management for production resume corpora
- add authentication, authorization, and tenant isolation for enterprise deployment
- add structured API contracts and deployment configuration

---

## What This Project Demonstrates

ResumeRAG is primarily a **RAG / Agentic AI engineering reference implementation** demonstrating how to combine:

```text
Advanced Retrieval
      +
Parent-Child RAG
      +
Hybrid Search
      +
Reranking
      +
CRAG
      +
Self-RAG
      +
LangGraph Routing & Loops
      +
Deterministic Evidence Verification
      +
Semantic Caching
      +
RAG Evaluation
      │
      ▼
Grounded Candidate Screening
```

The goal is not simply to put an LLM on top of a vector database.

The project demonstrates how retrieval, context management, verification, orchestration, caching, and evaluation can work together to build a more reliable GenAI application.

---

## License

No repository license is currently declared.

Add an appropriate license before distributing the project for reuse.

---

## Repository

**GitHub:** https://github.com/RudraMoharana-99/ResumeRAG

Built by **Rudra Moharana** as an exploration of production-oriented RAG and Agentic AI engineering.
