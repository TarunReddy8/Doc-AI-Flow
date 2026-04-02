# DocAI — AI Document Intelligence Platform

> A production-grade, end-to-end document extraction pipeline that converts unstructured invoices, contracts, and reports into structured JSON — powered by OCR, LLMs, ChromaDB vector search, and full MLOps lifecycle management.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)
![MLflow](https://img.shields.io/badge/MLflow-3.x-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5+-purple)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## What is DocAI?

DocAI is an AI-powered document intelligence platform that automates the extraction of structured data from business documents. You upload a scanned invoice, a contract PDF, or any image-based document — DocAI runs it through a multi-stage pipeline involving Optical Character Recognition (OCR), LLM-based field extraction, semantic vector storage, and real-time experiment tracking — and returns a clean, validated JSON object in milliseconds.

The entire system is built around a **FastAPI** backend with async endpoints, a **Streamlit** frontend for visual interaction, and **MLflow** for tracking every extraction run as a reproducible experiment. All service interactions are observable via **Prometheus** metrics, and similarity search across processed documents is powered by **ChromaDB** embeddings.

---

## Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| API Gateway | FastAPI + Uvicorn | Async HTTP server, auto-generated Swagger docs |
| OCR Engine | DocTR / Tesseract | Converts document images/PDFs to raw text |
| LLM Extraction | LangChain + OpenAI / Anthropic | Prompts LLMs to extract structured fields |
| Mock Mode | Built-in mock extractor | Runs full pipeline with no API key |
| Vector Store | ChromaDB (PersistentClient) | Stores document embeddings for semantic search |
| Experiment Tracking | MLflow | Logs every extraction run, metrics, prompt versions |
| Monitoring | Prometheus client | Counters, histograms for latency, confidence, errors |
| Frontend | Streamlit | Visual UI for upload, search, drift monitoring |
| Schemas | Pydantic v2 | Strict request/response validation |
| Logging | structlog | Structured JSON log output |
| Containerization | Docker Compose | One-command full deployment |

---

## End-to-End Architecture

```
User (Browser / curl)
        |
        | HTTP multipart/form-data (file + document_type)
        v
+------------------+
|   Streamlit UI   |  ← frontend/app.py
|  localhost:8501  |
+--------+---------+
         |
         | POST /api/v1/extract
         v
+--------------------------------------+
|         FastAPI Gateway              |  ← app/main.py + app/api/routes.py
|          localhost:8000              |
|                                      |
|  1. File validation (type, size)     |
|  2. Prometheus: ACTIVE_EXTRACTIONS++ |
+----------+---------------------------+
           |
           v
+---------------------+
|    OCR Service      |  ← app/services/ocr_service.py
|                     |
|  DocTR (primary)    |
|  Tesseract (fallback|
|  Demo text (no OCR) |
|                     |
|  Output: raw_text   |
|          confidence |
|          engine_used|
+----------+----------+
           |
           v
+---------------------+
|  Classification     |  ← extraction_service.classify_document()
|                     |
|  Keyword match      |
|  or LLM call        |
|                     |
|  Output: invoice /  |
|  contract / report  |
+----------+----------+
           |
           v
+---------------------+
|  LLM Extraction     |  ← app/services/extraction_service.py
|                     |      OR
|  Prompt Registry:   |  ← app/services/mock_extraction.py
|  invoice_v1/v2      |
|  contract_v1        |
|                     |
|  LangChain → OpenAI |
|  or Anthropic       |
|                     |
|  Output: structured |
|  JSON + confidence  |
+----------+----------+
           |
     +-----+------+
     |            |
     v            v
+---------+  +------------------+
| ChromaDB|  |     MLflow       |
| Vector  |  |  Tracking Server |
| Store   |  |  localhost:5000  |
|         |  |                  |
| Stores: |  | Logs:            |
| OCR text|  | ocr_confidence   |
| metadata|  | extr_confidence  |
| embedgs |  | fields_extracted |
|         |  | processing_time  |
| Powers: |  | prompt_version   |
| /search |  | ocr_engine       |
+---------+  +------------------+
           |
           v
+---------------------+
|  ExtractionResult   |  ← app/schemas/extraction.py
|  (Pydantic model)   |
|                     |
|  document_id        |
|  status: completed  |
|  document_type      |
|  ocr_result         |
|  extracted_data {}  |
|  confidence_score   |
|  prompt_version     |
|  mlflow_run_id      |
|  processing_time_ms |
|  warnings []        |
+---------------------+
           |
           v
     JSON Response → Streamlit UI / API caller
```

---

## How the Workflow Works — Step by Step

### Step 1 — Configuration (`app/core/config.py`)

When the application starts, `pydantic-settings` reads `.env` and populates a typed `Settings` singleton via `get_settings()`. This controls everything: which OCR engine to use (`doctr` or `tesseract`), which LLM provider (`mock`, `openai`, `anthropic`), ChromaDB persist path, MLflow tracking URI, max file size, and accuracy thresholds. The `@lru_cache` decorator ensures the settings object is loaded only once per process lifetime.

---

### Step 2 — Application Startup (`app/main.py`)

FastAPI boots via an `asynccontextmanager` lifespan handler. At startup, `setup_logging()` configures `structlog` for structured JSON output (or pretty-print in DEBUG mode). The `APP_INFO` Prometheus `Info` gauge is populated with the current OCR engine and LLM config. A CORS middleware allows all origins (configurable for production). The `/metrics` endpoint is mounted as a separate ASGI sub-application serving Prometheus text format. All business routes are registered under the `/api/v1` prefix.

---

### Step 3 — Document Upload (`app/api/routes.py → POST /api/v1/extract`)

The client sends a `multipart/form-data` HTTP request with:
- `file` — the document binary (PDF, PNG, JPG, TIFF, BMP, WebP — max 50 MB)
- `document_type` — optional enum (`invoice`, `contract`, `report`, `receipt`, `unknown`)
- `store_in_vectordb` — whether to persist embeddings to ChromaDB
- `prompt_version` — optional string for A/B testing specific prompt versions

The route handler generates a UUID `document_id`, increments the `ACTIVE_EXTRACTIONS` Prometheus gauge, validates the file extension and size, and reads the full binary content into memory with `await file.read()`.

---

### Step 4 — OCR Extraction (`app/services/ocr_service.py`)

`OCRService._init_engines()` attempts to initialize two OCR engines at startup:
- **DocTR** (primary): a deep-learning OCR model from `python-doctr[torch]`. Loads a pretrained `ocr_predictor` using a CRNN architecture. High accuracy on printed text.
- **Tesseract** (fallback): calls `pytesseract.image_to_data()` with `Output.DICT` to get per-word confidence scores, averages them to produce a page-level confidence.

`extract_text()` converts the uploaded bytes to a list of `PIL.Image` objects (via `pdf2image.convert_from_bytes` for PDFs, or `Image.open` for images). It tries the primary engine, and if confidence falls below `OCR_CONFIDENCE_THRESHOLD` (default 0.7), it attempts the alternate engine and keeps whichever produced higher confidence. If neither engine is installed, a `_demo_text_fallback()` returns realistic sample invoice text so the full pipeline can run for demonstration.

The result is an `OCRResult` Pydantic model containing: `raw_text`, `confidence`, `engine_used`, `page_count`, and `processing_time_ms`.

---

### Step 5 — Document Classification (`app/services/extraction_service.py`)

If `document_type=unknown` was passed, `classify_document(ocr_text)` is called. In mock mode, `mock_classify()` does keyword matching — checking for words like "invoice", "total", "contract", "agreement", "receipt" in the lowercased OCR text. In real LLM mode, it invokes the `classify_v1` prompt via `LangChain` with a `SystemMessage` + `HumanMessage`, sending the first 500 characters of OCR text, and parses the single-word response into a `DocumentType` enum.

---

### Step 6 — LLM Structured Extraction (`app/services/extraction_service.py` + `mock_extraction.py`)

This is the core intelligence stage. The **Prompt Registry** (`PROMPT_REGISTRY`) is a versioned dictionary of system prompts and user templates:
- `invoice_v1` — basic field list extraction
- `invoice_v2` — chain-of-thought prompt with step-by-step instructions and explicit JSON schema
- `contract_v1` — legal metadata extraction
- `classify_v1` — single-word classification

`_select_prompt()` picks the highest-version prompt for the given document type, or uses the `prompt_version` override for A/B testing.

**In mock mode** (`LLM_PROVIDER=mock`): `mock_extract()` uses pool-based randomization to generate realistic invoice/contract data — pulling from `MOCK_VENDORS`, `MOCK_CUSTOMERS`, `MOCK_LINE_ITEMS` arrays. It simulates confidence scoring based on OCR text length.

**In real LLM mode**: `LangChain` invokes either `ChatOpenAI` or `ChatAnthropic` with the versioned prompt, parses the response through `_parse_json_output()` which strips markdown code fences and extracts JSON from the raw string. `_calculate_confidence()` computes a heuristic score weighted 70% on critical fields (invoice number, vendor, total) and 30% on important fields (line items, tax, currency).

The output is `(extracted_data: dict, prompt_version: str, confidence: float)`.

---

### Step 7 — Vector Storage (`app/services/vector_service.py`)

`VectorService` wraps a `chromadb.PersistentClient` pointing to `./data/chroma`. On first run it creates a collection named `docai_documents` with cosine similarity (`hnsw:space: cosine`). `store_document()` calls `collection.upsert()` with:
- `ids` — the document UUID
- `documents` — the raw OCR text (ChromaDB embeds this automatically using its default embedding function)
- `metadatas` — `document_type`, `extracted_at` timestamp, `field_count`, `confidence`, `filename`, `prompt_version`

This enables the `GET /api/v1/search?query=...` endpoint, which calls `collection.query()` with the search string, returns the top-N most semantically similar documents with cosine distance scores.

---

### Step 8 — Experiment Tracking (`app/services/mlflow_service.py`)

`MLflowService._init_mlflow()` calls `mlflow.set_tracking_uri()` pointing to the MLflow server on port 5000, and creates/selects the `docai-extraction` experiment. Every successful extraction triggers `log_extraction_run()` which opens an MLflow run and logs:
- **Parameters**: `document_id`, `document_type`, `prompt_version`, `ocr_engine`, `llm_model`
- **Metrics**: `ocr_confidence`, `extraction_confidence`, `fields_extracted`, `total_fields`, `field_completeness` (ratio), `processing_time_ms`
- **Tags**: `pipeline=docai-extraction`, `has_warnings` if applicable

The returned `run_id` is included in the API response so you can deep-link directly to that run in the MLflow UI.

`check_drift()` queries the last `2×window` runs, splits them into recent vs baseline halves, computes the mean `extraction_confidence` for each, and flags drift if the delta exceeds 0.05. `get_prompt_comparison()` groups runs by `prompt_version` and returns average confidence, processing time, and field completeness per version — the foundation for data-driven prompt A/B testing decisions.

---

### Step 9 — Prometheus Metrics (`monitoring/metrics.py`)

Every request updates a set of Prometheus instruments exposed at `GET /metrics`:
- `docai_requests_total` (Counter) — labelled by `document_type` and `status`
- `docai_request_duration_seconds` (Histogram) — end-to-end latency buckets
- `docai_ocr_confidence` (Histogram) — OCR quality distribution per engine
- `docai_extraction_confidence` (Histogram) — LLM quality per doc type and prompt version
- `docai_fields_extracted` (Histogram) — how many fields were populated
- `docai_extraction_errors_total` (Counter) — labelled by error type
- `docai_active_extractions` (Gauge) — current in-flight requests
- `docai_app` (Info) — static build metadata

In a Docker deployment these are scraped by a Prometheus container and visualized in Grafana.

---

### Step 10 — API Response

The route assembles and returns an `ExtractionResult` Pydantic model:

```json
{
  "document_id": "uuid-v4",
  "status": "completed",
  "document_type": "invoice",
  "ocr_result": {
    "raw_text": "...",
    "confidence": 0.75,
    "engine_used": "tesseract",
    "page_count": 1,
    "processing_time_ms": 340.2
  },
  "extracted_data": {
    "invoice_number": "INV-2024-0847",
    "vendor_name": "Acme Cloud Services",
    "total_amount": 3634.75,
    "line_items": [...],
    ...
  },
  "confidence_score": 0.9424,
  "prompt_version": "mock_v1.0",
  "mlflow_run_id": "abc123...",
  "processing_time_ms": 1680.72,
  "warnings": []
}
```

---

### Step 11 — Streamlit Frontend (`frontend/app.py`)

The Streamlit UI connects to the FastAPI backend via `requests`. It has three tabs:

- **Extract Document** — file uploader widget sends the binary to `POST /api/v1/extract`, renders the JSON result alongside confidence metrics, OCR text preview, and a link to the MLflow run.
- **Semantic Search** — text input calls `GET /api/v1/search`, renders matching documents with cosine distance scores.
- **Monitoring** — buttons call `GET /api/v1/monitoring/drift` and `GET /api/v1/prompts/compare`, displaying drift status and prompt A/B metrics.

The sidebar shows a live health check against `GET /api/v1/health`, displaying OCR engine, LLM provider, ChromaDB status, and MLflow connectivity.

---

### Step 12 — Offline Evaluation (`ml/pipelines/evaluation.py`)

`run_evaluation()` loads `GROUND_TRUTH_SAMPLES` — hardcoded reference extractions with known correct field values. For each sample, `calculate_field_accuracy()` compares extracted values against ground truth using exact string match for text fields, float tolerance (`< 0.01`) for numeric fields, and count comparison for arrays. It prints a per-sample accuracy report and an overall score. This is designed to run as a scheduled CI job (or manually) to detect prompt regression before it reaches production.

---

## Project Structure

```
docai/
│
├── .env                              # Active config (LLM_PROVIDER, API keys, ports)
├── .env.example                      # Template for new developers
├── requirements.txt                  # All Python dependencies
├── start.sh                          # One-command launcher (all 3 services)
├── docker-compose.yml                # Docker: API + MLflow + Prometheus + Grafana
│
├── app/                              # FastAPI backend
│   ├── main.py                       # App factory: CORS, metrics mount, lifespan
│   ├── api/
│   │   └── routes.py                 # /extract  /search  /health  /drift  /prompts
│   ├── core/
│   │   ├── config.py                 # pydantic-settings: reads .env into Settings
│   │   └── logging.py                # structlog JSON logger setup
│   ├── schemas/
│   │   └── extraction.py             # Pydantic models: ExtractionResult, OCRResult…
│   └── services/
│       ├── ocr_service.py            # DocTR + Tesseract OCR, confidence scoring
│       ├── extraction_service.py     # Versioned prompts, LangChain LLM calls
│       ├── mock_extraction.py        # Mock LLM — runs without any API key
│       ├── vector_service.py         # ChromaDB upsert + cosine similarity search
│       └── mlflow_service.py         # Experiment logging, drift detection, A/B compare
│
├── frontend/
│   └── app.py                        # Streamlit: upload, search, monitoring tabs
│
├── monitoring/
│   ├── metrics.py                    # Prometheus counters, histograms, gauges
│   └── prometheus.yml                # Scrape config for Docker deployment
│
├── ml/
│   └── pipelines/
│       └── evaluation.py             # Ground truth accuracy evaluation
│
├── data/
│   ├── generate_samples.py           # Generates sample_invoice.png + sample_contract.png
│   ├── sample_docs/                  # Test documents
│   ├── chroma/                       # ChromaDB SQLite + HNSW index (auto-managed)
│   └── mlflow.db                     # MLflow run history (auto-managed)
│
├── tests/
│   └── test_extraction.py            # pytest unit tests
│
└── docker/
    └── Dockerfile                    # Production container image
```

---

## Quick Start (Local)

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.10+ | [python.org](https://python.org) |
| Tesseract OCR | any | Windows: [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) |

> **No API key required** — runs in `mock` mode out of the box.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

The default `.env` already sets `LLM_PROVIDER=mock` — no changes needed to run locally.

To use a real LLM, edit `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
# or
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 3. Generate sample documents

```bash
python data/generate_samples.py
```

This creates `data/sample_docs/sample_invoice.png` and `sample_contract.png`.

### 4. Start all services

**Option A — single command:**
```bash
bash start.sh
```

**Option B — manually (3 terminals):**

```bash
# Terminal 1 — FastAPI backend
uvicorn app.main:app --port 8000 --host 0.0.0.0

# Terminal 2 — MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///data/mlflow.db \
  --default-artifact-root ./data/mlflow_artifacts

# Terminal 3 — Streamlit frontend
streamlit run frontend/app.py --server.port 8501
```

### 5. Open the interfaces

| Interface | URL |
|-----------|-----|
| Streamlit Dashboard | http://localhost:8501 |
| Swagger API Docs | http://localhost:8000/docs |
| MLflow Experiments | http://localhost:5000 |
| Prometheus Metrics | http://localhost:8000/metrics |

---

## Quick Start (Docker)

```bash
docker-compose up --build
```

Starts: FastAPI (8000), MLflow (5000), Prometheus (9090), Grafana (3000).

---

## API Reference

### POST `/api/v1/extract`

Upload a document and extract structured data.

```bash
curl -X POST http://localhost:8000/api/v1/extract \
  -F "file=@data/sample_docs/sample_invoice.png" \
  -F "document_type=invoice"
```

Query parameters:
- `document_type` — `invoice | contract | report | receipt | unknown` (default: `unknown`)
- `store_in_vectordb` — `true | false` (default: `true`)
- `prompt_version` — `v1 | v2` (optional, for A/B testing)

### GET `/api/v1/search`

Semantic search across all processed documents.

```bash
curl "http://localhost:8000/api/v1/search?query=cloud+hosting+invoice&n_results=5"
```

### GET `/api/v1/health`

System health check.

```bash
curl http://localhost:8000/api/v1/health
```

### GET `/api/v1/monitoring/drift`

Check extraction quality drift against historical baseline.

```bash
curl "http://localhost:8000/api/v1/monitoring/drift?document_type=invoice&window=50"
```

### GET `/api/v1/prompts/compare`

Compare prompt version performance for A/B testing decisions.

```bash
curl "http://localhost:8000/api/v1/prompts/compare?document_type=invoice"
```

---

## Supported Document Types

| Type | Extracted Fields |
|------|-----------------|
| `invoice` | invoice_number, invoice_date, due_date, vendor_name, vendor_address, customer_name, customer_address, line_items[], subtotal, tax, total_amount, currency, payment_terms |
| `contract` | contract_title, parties[], effective_date, expiration_date, contract_value, key_terms[], governing_law, termination_clause |
| `report` | Falls back to invoice extraction schema |
| `receipt` | Falls back to invoice extraction schema |

---

## Running Tests

```bash
pytest tests/ -v --cov=app
```

## Running the Evaluation Pipeline

```bash
python -m ml.pipelines.evaluation --doc-type invoice
```

This runs ground truth accuracy evaluation and prints per-field match rates.

---

## MLOps Features

| Feature | How it works |
|---------|-------------|
| Prompt versioning | Every prompt in `PROMPT_REGISTRY` has a version key (`invoice_v1`, `invoice_v2`). The version used is logged to MLflow per run. |
| A/B testing | Pass `?prompt_version=v1` or `v2` to force a specific version. Compare results via `GET /prompts/compare`. |
| Drift detection | `GET /monitoring/drift` splits the last N runs into recent vs baseline, compares mean confidence. Flags drift if delta > 0.05. |
| Experiment tracking | Every extraction run logs OCR confidence, extraction confidence, field completeness, processing time, and warnings to MLflow. |
| Accuracy evaluation | `ml/pipelines/evaluation.py` compares extractions against hardcoded ground truth samples with field-level accuracy scoring. |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
