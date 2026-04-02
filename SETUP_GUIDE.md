# 🚀 DocAI — Local Setup Guide (VS Code + Claude Code)

This guide walks you through running DocAI **end-to-end on your machine** —
from zero to seeing extractions in your browser.

---

## 📋 Prerequisites

| Tool | Why | Install |
|------|-----|---------|
| Python 3.11+ | Runtime | [python.org](https://python.org) |
| VS Code | Editor | [code.visualstudio.com](https://code.visualstudio.com) |
| Claude Code extension | AI pair-programming | VS Code Extensions → "Claude Code" |
| Tesseract OCR | Text extraction engine | See below |
| poppler-utils | PDF → image conversion | See below |

### Install Tesseract + Poppler

**macOS:**
```bash
brew install tesseract poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils
```

**Windows:**
- Tesseract: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
- Poppler: Download from [poppler releases](https://github.com/oseelon/poppler-win/releases)
- Add both to your system PATH

---

## 🏗️ Step-by-Step Setup

### Step 1: Create project folder
```bash
mkdir docai && cd docai
```

### Step 2: Create virtual environment
```bash
python -m venv venv

# Activate it:
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Create your .env file
```bash
cp .env.example .env
```

**Two modes to run DocAI:**

**Mode A — FREE (no API key needed):**
Edit `.env` and set:
```
LLM_PROVIDER=mock
```
This uses the built-in mock extractor that returns realistic fake data.
Perfect for learning the architecture without spending money.

**Mode B — Real LLM extraction:**
Edit `.env` and set either:
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
```
or:
```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Step 5: Create data directories
```bash
mkdir -p data/chroma data/sample_docs
```

### Step 6: Run the API server
```bash
uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000/docs — you should see the Swagger UI.

### Step 7: Run the Streamlit frontend (new terminal)
```bash
streamlit run frontend/app.py --server.port 8501
```
Open http://localhost:8501 — this is the visual dashboard.

### Step 8: Run MLflow tracking (new terminal)
```bash
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///data/mlflow.db \
  --default-artifact-root ./data/mlflow_artifacts
```
Open http://localhost:5000 — see every extraction logged as an experiment.

---

## 🧪 Test It

### Quick API test with curl:
```bash
curl -X POST http://localhost:8000/api/v1/extract \
  -F "file=@data/sample_docs/sample_invoice.png" \
  -F "document_type=invoice"
```

### Run unit tests:
```bash
pytest tests/ -v --cov=app
```

### Run evaluation pipeline:
```bash
python -m ml.pipelines.evaluation --doc-type invoice
```

---

## 🗂️ What to Explore in VS Code

Open the project in VS Code and explore in this order:

```
1. app/main.py              → Application entry point
2. app/api/routes.py        → All API endpoints (start here!)
3. app/services/ocr_service.py    → Dual OCR engine logic
4. app/services/extraction_service.py → LLM prompts + versioning
5. app/services/mock_extraction.py    → Mock mode (no API key)
6. app/services/vector_service.py     → ChromaDB semantic search
7. app/services/mlflow_service.py     → Experiment tracking
8. monitoring/metrics.py              → Prometheus counters
9. ml/pipelines/evaluation.py        → Ground truth evaluation
10. frontend/app.py                   → Streamlit dashboard
```

---

## 💡 Using Claude Code in VS Code

With the Claude Code extension installed, try these prompts:

- `"Explain what extract_document in routes.py does step by step"`
- `"Add a new document type 'receipt' with its own extraction prompt"`
- `"Write a test for the OCR fallback when DocTR is unavailable"`
- `"Add a /batch endpoint that processes multiple files"`
- `"Explain the drift detection logic in mlflow_service.py"`

Claude Code can see all your files and help you understand + extend the project.

---

## 🐳 Docker (Alternative to Local Setup)

If you prefer Docker:
```bash
docker-compose up --build
```
This starts: API (8000), MLflow (5000), Prometheus (9090), Grafana (3000).
