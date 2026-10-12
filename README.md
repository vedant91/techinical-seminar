# 🤖 Agentic RAG — 6-Agent, Self-Validating Hybrid Retrieval Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA%20NIM-Nemotron%20%2B%20Llama-76B900?logo=nvidia&logoColor=white)](https://build.nvidia.com)
[![pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-336791?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, multi-agent Retrieval-Augmented Generation (RAG) system engineered for high-stakes technical domains. Every stage operates behind an explicit quality gate, failures trigger targeted retry loops with actionable feedback, and ungrounded queries are rejected with honest gap reports rather than fabricated answers.

---

## 📌 Architecture Overview

```text
                 +-----------------------+
   document ---> | 1. INGESTION AGENT    | ---> Structured Document + Vectorless Outline Tree
                 +-----------------------+      (Preserves headings, lists, tables, code blocks)
                             |
                 +-----------------------+
                 | 2. CHUNKING AGENT     | ---> Semantic Chunks (structure-aware, never fixed-char splits)
                 +-----------------------+
                             |
                 +-----------------------+
                 | 3. CHUNK VALIDATOR    | <--- Gate: Coherence / Boundaries / Completeness
                 +-----------------------+      Fail -> Actionable feedback loop to Agent 2 (retry <= 2)
                             | Pass             Exhausted -> Quarantine failing chunks for human audit
                 +-----------------------+
   query ------> | 4. RETRIEVAL AGENT    | ---> HYBRID RETRIEVAL:
                 +-----------------------+      * Dense Vector (all-MiniLM-L6-v2 + pgvector)
                             |                  * Vectorless Reasoning (LLM document outline traversal)
                 +-----------------------+
                 | 5. RETRIEVAL VALIDATOR| <--- Gate: Relevance / Sufficiency / Information Gaps
                 +-----------------------+      Fail -> Feedback to Agent 4 ("lean on vectorless path")
                             | Pass             Exhausted -> Explicit "insufficient_context" refusal
                 +-----------------------+
                 | 6. ANSWER AGENT       | ---> Synthesized response with per-claim citation mapping
                 +-----------------------+      + Unanswered gaps breakdown (never hallucinates)
```

An **Orchestrator** coordinates state machine transitions across all six specialist agents, manages retry budgets, writes an observable audit trail (`logs/stage_log.jsonl`), and quarantines malformed data (`logs/quarantine.jsonl`).

---

## 💡 Why Hybrid Retrieval & Validation?

1. **Dense Vector Search Alone Fails on Structural Questions**:
   - Vector embeddings excel at semantic similarity ("*What encryption algorithm is used?*").
   - But vector search fails on structural or relational questions ("*What does Section 4 conclude?*" or "*Compare Q2 financial revenue tables with the security audit findings*").
   - **Our Vectorless Path** inspects the parsed outline tree directly like an intelligent table-of-contents, retrieving entire cohesive sections and merging them with vector hits.

2. **Deterministic Processing + LLM Quality Judges**:
   - Ingestion and chunking preserve document hierarchies deterministically.
   - Quality gates are evaluated by high-speed instruction judges (`meta/llama-3.2-11b-vision-instruct`).
   - Answer synthesis is powered by a high-capacity reasoning model (`nvidia/nemotron-3-ultra-550b-a55b`) with an internal reasoning budget.
   - **Zero-Dependency Fallback**: If no API key is provided, the pipeline switches to deterministic heuristic judges, allowing full offline testing without external dependencies.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Answer Synthesis** | NVIDIA NIM (`nvidia/nemotron-3-ultra-550b-a55b`) | Deep reasoning & citation mapping |
| **Quality Gate Judges** | NVIDIA NIM (`meta/llama-3.2-11b-vision-instruct`) | Low-latency validation & retry feedback |
| **Offline Brain** | Python Heuristic Rule Engine | Stand-in judges when API key is omitted |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Fast 384-dimensional local dense embeddings |
| **Primary Vector Store** | PostgreSQL 16 + `pgvector` (Docker) | Production cosine similarity vector search |
| **Fallback Vector Store**| NumPy In-Memory Cosine Store | Automatic zero-configuration fallback |
| **PDF Ingestion** | PyMuPDF (`fitz`) | Layout analysis & font-size heading extraction |
| **Backend API** | FastAPI + Uvicorn | Async REST endpoints & static file hosting |
| **Frontend UI** | Vanilla HTML5 / CSS3 / JavaScript | Modern glassmorphism dashboard & live pipeline viz |

---

## 🚀 Getting Started End-to-End

Follow these steps to set up and run the technical seminar project from scratch.

### 1. Prerequisites

- **Python 3.10, 3.11, or 3.12** installed on your machine.
- **Git** installed.
- *(Optional)* **Docker Desktop** installed (if you want PostgreSQL + pgvector). If Docker is not running, the system automatically uses the built-in in-memory vector store.
- *(Optional)* **NVIDIA NIM API Key** (free tier available at [build.nvidia.com](https://build.nvidia.com)). If omitted, the heuristic offline judges run seamlessly.

### 2. Clone the Repository

```bash
git clone https://github.com/vedant91/techinical-seminar.git
cd techinical-seminar
```

### 3. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Copy the sample configuration file:

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**macOS / Linux:**
```bash
cp .env.example .env
```

Open `.env` in your editor and configure your variables:
```env
# Get a free key at https://build.nvidia.com (leave empty for offline heuristic mode)
NVIDIA_API_KEY=your_key_here
NIM_BASE_URL=https://integrate.api.nvidia.com/v1
NIM_MODEL=nvidia/nemotron-3-ultra-550b-a55b
NIM_JUDGE_MODEL=meta/llama-3.2-11b-vision-instruct

# Reasoning mode for the answer model (1 = enabled, 0 = disabled)
NIM_ENABLE_THINKING=1
NIM_REASONING_BUDGET=16384

# Vector database (default is local pgvector container)
PG_DSN=postgresql://rag:rag@localhost:5432/ragdb
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_RETRIES=2
RETRIEVAL_TOP_K=8
LOG_DIR=logs
```

### 6. (Optional) Start pgvector with Docker

If you have Docker Desktop installed and want to run the PostgreSQL vector database:

```bash
docker compose up -d
```

> **Note**: If Docker is not running, you will see a non-blocking `[warn] pgvector unreachable ... Falling back to in-memory store`. The pipeline will execute with 100% functionality using the in-memory store!

---

## 🖥️ Running the Project

### Option A: Interactive Web Visualizer (Recommended)

Start the unified FastAPI server:

```powershell
python server.py
```

Now open your browser and navigate to:
```
http://localhost:8080
```

#### What You Can Do in the Web Dashboard:
1. **Document Upload**: Drag-and-drop or upload `.md`, `.txt`, or `.pdf` files (try `ML.pdf` or files in `demo/sample_docs/`).
2. **Ingestion & Outline**: Inspect the document metadata and real-time JSON outline tree produced by Agent 1.
3. **Chunking & Validation**: Review the semantic chunks generated by Agent 2, with detailed coherence, boundary quality, and coverage metrics from Agent 3.
4. **Hybrid Search**: Query your document and watch the side-by-side retrieval paths (Vector vs Vectorless) merge and synthesize into an answer with citations.

---

### Option B: Automated Seminar Demo Script

Run the automated seminar demo script which processes three sample enterprise documents and answers four diverse query types:

```powershell
python demo/run_demo.py
```

This runs through:
- **Vector Query**: Dense similarity matching across technical security specs.
- **Structural Query**: Outline-guided section retrieval ("*What does section 4 say?*").
- **Multi-Hop Query**: Merging financial figures with security audit findings across multiple documents.
- **Honest Refusal Query**: Intentionally asking about information not in the docs to verify that the pipeline refuses ungrounded answers instead of hallucinating.

---

### Option C: Custom CLI Ingestion & Querying

Ingest any document and ask arbitrary questions directly via the CLI:

```powershell
# Ingest a custom file and ask a question:
python demo/run_demo.py --ingest demo/sample_docs/acme_security_audit.md --ask "What critical vulnerabilities were identified?" --verbose
```

---

### Option D: Programmatic Python API

You can embed the self-validating pipeline into any Python application:

```python
from agentic_rag import AgenticRAG

# Initialize pipeline (loads embedder, brain, and store)
rag = AgenticRAG()

# Ingest any Markdown, plain text, or PDF document
rag.ingest_document("demo/sample_docs/acme_ai_platform.md")

# Ask a query
result = rag.ask("What encryption standards does the platform use?")

# Print formatted output with stage-by-stage confidence audit
AgenticRAG.print_result(result, verbose=True)
```

---

## 🧪 Verifying the Pipeline (Diagnostic Smoke Test)

Run the included smoke test to verify API keys, judge latency, and reasoning budgets:

```powershell
python scratch_smoke.py
```

Example expected output:
```text
key set: True
answer model: nvidia/nemotron-3-ultra-550b-a55b  thinking: True  budget: 16384
judge model : meta/llama-3.2-11b-vision-instruct
JUDGE path OK in 1.3s: {'status': 'pass', 'confidence': 0.95, 'reason': 'smoke ok'}
ANSWER path OK in 2.2s: {'stage': 'answer_generation', 'status': 'pass', 'confidence': 0.99, 'answer': '2+2 equals 4.', 'citations': [], 'unanswered_parts': [], 'reason': 'smoke ok'}
```

---

## 🛡️ The 6-Agent Quality Gate & Escalation System

1. **Agent 1: Ingestion (`ingestion.py`)**
   - Parses Markdown, text, and PDF files.
   - Extracts document hierarchy (H1-H6) and builds a table of contents outline.
   - Preserves tables and list structures.

2. **Agent 2: Chunking (`chunking.py`)**
   - Produces semantic chunks; never splits tables or lists mid-boundary.
   - Can receive structured feedback from the validator to merge or split boundaries.

3. **Agent 3: Chunk Validator (`chunk_validator.py`)**
   - Evaluates coherence, boundary quality, and completeness.
   - If quality fails, sends actionable feedback back to Agent 2 (up to `MAX_RETRIES`).
   - If retries fail, non-passing chunks are logged in `logs/quarantine.jsonl` while valid chunks proceed.

4. **Agent 4: Retrieval (`retrieval.py`)**
   - **Dense Path**: Generates query embeddings with `all-MiniLM-L6-v2` and performs cosine similarity search.
   - **Vectorless Path**: Uses outline reasoning to identify entire relevant sections.
   - Merges results and tags chunks with source provenance (`vector`, `vectorless`, or `both`).

5. **Agent 5: Retrieval Validator (`retrieval_validator.py`)**
   - Checks if retrieved context is sufficient and relevant for the query.
   - Can trigger a retry instructing Agent 4 to increase weights or look at specific outline sections.
   - If context cannot be validated, halts execution and returns an honest `insufficient_context` error rather than proceeding to guess.

6. **Agent 6: Answer Generator (`answer.py`)**
   - Produces synthesized answers strictly grounded in validated context.
   - Outputs precise per-claim citations.
   - Flags any unanswered aspects in `unanswered_parts`.

---

## 📂 Project Structure

```text
techinical-seminar/
├── .env.example             # Template for environment configuration
├── .gitignore               # Comprehensive Git exclusion rules
├── docker-compose.yml       # PostgreSQL 16 + pgvector container definition
├── requirements.txt         # Pinned project dependencies
├── server.py                # FastAPI server (API routes + frontend static server)
├── scratch_smoke.py         # End-to-end diagnostic & latency smoke test
├── ML.pdf                   # Sample technical seminar research PDF
├── README.md                # Comprehensive documentation
├── agentic_rag/             # Core Python package
│   ├── __init__.py          # Exports AgenticRAG and Config
│   ├── config.py            # Environment-driven configuration dataclass
│   ├── contracts.py         # Strict typed dataclass schemas for stage boundaries
│   ├── embeddings.py        # Local SentenceTransformer embedding wrapper
│   ├── llm.py               # NIMBrain (NVIDIA API) & HeuristicBrain fallbacks
│   ├── orchestrator.py      # State machine router, retries, quality gates, logs
│   ├── pipeline.py          # High-level facade for ingestion and querying
│   ├── store.py             # PgVectorStore and InMemoryStore implementations
│   └── agents/              # The 6 specialist pipeline agents
│       ├── ingestion.py            # Agent 1: Parse and build outline tree
│       ├── chunking.py             # Agent 2: Semantic boundary chunker
│       ├── chunk_validator.py      # Agent 3: Chunk quality gate
│       ├── retrieval.py            # Agent 4: Hybrid vector + outline retrieval
│       ├── retrieval_validator.py  # Agent 5: Retrieval sufficiency gate
│       └── answer.py               # Agent 6: Cited answer generator
├── demo/                    # Seminar demonstration scripts
│   ├── run_demo.py          # Interactive CLI demo runner
│   └── sample_docs/         # Sample domain documents
│       ├── acme_ai_platform.md     # Technical product specification
│       ├── acme_finance_q2.md      # Financial quarterly report with tables
│       └── acme_security_audit.md  # Security audit and compliance report
└── hybrid_search_viz/       # Modern web dashboard frontend
    ├── index.html           # 4-stage pipeline visualizer UI
    ├── style.css            # Dark mode glassmorphism design system
    └── script.js            # Live WebSocket/HTTP pipeline orchestrator
```

---

## 📡 REST API Reference

The FastAPI server provides the following endpoints:

| Endpoint | Method | Parameters | Description |
|---|---|---|---|
| `/api/health` | `GET` | None | Returns `{"status": "ok"}` for liveness checks |
| `/api/status` | `GET` | None | Returns active vector store type, indexed chunk count, and readiness |
| `/api/ingest` | `POST` | `multipart/form-data` (`file`) | Ingests `.md`, `.txt`, or `.pdf`, runs Agent 1-3, and returns the document tree |
| `/api/ask` | `POST` | `application/x-www-form-urlencoded` (`query`) | Runs Agent 4-6 on the indexed store and returns answer, citations, and path provenance |

---

## 🤝 Contributing & License

Created for technical seminars, research demonstrations, and production RAG reference implementations. Distributed under the MIT License.
