# RAG-I

> PDF-powered RAG pipeline with Inngest orchestration, Nomic embeddings, Qdrant vector search, and multi-LLM fallback (OpenAI → Gemini → Groq).

![Python](https://img.shields.io/badge/python-3.13-blue)
![uv](https://img.shields.io/badge/package%20manager-uv-purple)
![Tests](https://img.shields.io/badge/tests-18%20passing-brightgreen)

---

## What is this?

RAG-I is an event-driven Retrieval-Augmented Generation (RAG) system. Upload PDFs through a Streamlit UI, index their content as vector embeddings, and ask natural language questions. The system retrieves the most relevant chunks and uses an LLM to generate a grounded answer.

---


## Architecture

```
Streamlit UI
     │
     ▼ fire Inngest events
┌─────────────────────────────────────┐
│         FastAPI + Uvicorn           │
│                                     │
│  rag/ingest_pdf                     │
│    step 1: load + chunk PDF         │
│    step 2: embed (Nomic) + upsert   │──► Qdrant
│                                     │
│  rag/query_pdf_ai                   │
│    step 1: embed query + search     │◄── Qdrant
│    step 2: LLM answer               │──► OpenAI / Gemini / Groq
└─────────────────────────────────────┘
     │
     ▼ poll for result
Streamlit UI
```

---

## Features

* **PDF Ingestion** — upload any PDF; chunked (1024 tokens, 200 overlap) and indexed automatically
* **Semantic Search** — cosine similarity over embeddings
* **Multi-LLM Fallback** — OpenAI GPT-4o-mini → Gemini → Groq Llama 3.3
* **Event-Driven Steps** — retryable, observable Inngest steps
* **Structured Logging** — `structlog` with JSON (prod) and pretty console (dev)
* **18 Unit Tests** — external services mocked via `pytest`

---

## Tech Stack

| Layer           | Technology                                | Why This Choice                                                                 |
| --------------- | ----------------------------------------- | ------------------------------------------------------------------------------- |
| API             | FastAPI + Uvicorn                         | High-performance async framework with minimal boilerplate and automatic docs    |
| Orchestration   | Inngest                                   | Event-driven workflows with built-in retries, observability, and step isolation |
| UI              | Streamlit                                 | Rapid prototyping of interactive UIs without frontend overhead                  |
| Embeddings      | Nomic (`nomic-embed-text-v1.5`, 768 dims) | High-quality embeddings with good semantic performance and cost efficiency      |
| Vector DB       | Qdrant                                    | Fast vector similarity search with filtering and easy local + cloud setup       |
| PDF Parsing     | LlamaIndex                                | Simplifies document ingestion, chunking, and preprocessing pipelines            |
| LLM             | OpenAI → Gemini → Groq                    | Multi-provider fallback ensures reliability and avoids downtime/quota limits    |
| Logging         | structlog                                 | Structured, production-ready logging with flexible output formats               |
| Testing         | pytest + pytest-mock                      | Simple, powerful testing with easy mocking of external dependencies             |
| Package Manager | uv                                        | Extremely fast dependency resolution and modern Python packaging                |
| Language        | Python 3.13                               | Latest features and performance improvements for modern backend development     

---

## 🚧 Future Improvements

These features are planned but were not implemented due to time constraints. They will be prioritized when updating this project to a production-ready SaaS application.
### Architecture

* **REST Endpoints:**
We need to add two FastAPI routes `/ingest` and `/query` alongside the existing Inngest workflows. This way our system won't just rely on events.

* **Async Embedding:**
The embed.text()` calls are blocking the event loop. We should use `asyncio.run_in_executor` to run these calls in the background. This will help improve the systems throughput.

* **Retry & Backoff Configuration:**
We should make the LLM provider order and retry behavior configurable. This way we can control how the system handles retries and errors. We can use environment variables to make these settings configurable.

### Security

* **Authentication:**
To secure our API we need to add a middleware that checks for API keys. This will ensure that authorized users can access our FastAPI endpoints.

### Scalability

* **Upload Storage:**
Instead of storing uploads locally in the `uploads/` folder we should use a cloud storage solution like S3 or GCS. This will make our storage more durable and scalable.

* **Multi-Tenancy:**
We need to separate collections for each user. This will ensure that each users data is isolated and secure.

### Observability

* **Health Check Endpoint:**
We should add a `GET /health` endpoint. This will allow us to check if our service and Qdrant are connected and working properly.

* **Metrics & Monitoring:**
We need to track some metrics, such as how many ingestions we're processing how long queries are taking and how often we're using LLM providers. We can use tools, like Prometheus. Structured logging to monitor these metrics.

---

## Prerequisites

* [uv](https://docs.astral.sh/uv/)
* [Docker](https://www.docker.com/)
* API keys: Nomic + at least one LLM provider

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/AryanshTripathi/RAG-I.git
cd RAG-I
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Environment variables

Create `.env`:

```env
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...
NOMIC_API_KEY=nk-...
```

### 4. Nomic login

```bash
uv run nomic login <your-nomic-api-key>
```

### 5. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

## Running the App

Run in three terminals:

**FastAPI**

```bash
uv run uvicorn main:app --reload
```

**Inngest**

```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

**Streamlit**

```bash
uv run streamlit run streamlit_app.py
```

Open: [http://localhost:8501](http://localhost:8501)

---

## How It Works

### Ingesting a PDF

1. Upload PDF
2. Trigger `rag/ingest_pdf`
3. Chunk into 1024 tokens
4. Embed + store in Qdrant

### Querying

1. Ask question
2. Trigger `rag/query_pdf_ai`
3. Retrieve top-k chunks
4. LLM generates answer (with fallback)

---

## Project Structure

```
RAG-I/
├── main.py
├── data_loader.py
├── vector_db.py
├── custom_types.py
├── logger.py
├── streamlit_app.py
├── tests/
├── pyproject.toml
└── uv.lock
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

No external services required.

---

## Environment Variables

| Variable       | Required  | Description            |
| -------------- | --------- | ---------------------- |
| OPENAI_API_KEY | Optional* | Primary LLM            |
| GEMINI_API_KEY | Optional* | Fallback               |
| GROQ_API_KEY   | Optional* | Fallback               |
| NOMIC_API_KEY  | Yes       | Embeddings             |
| LOG_LEVEL      | No        | Default: debug         |
| ENV            | No        | production = JSON logs |

* At least one LLM key required
