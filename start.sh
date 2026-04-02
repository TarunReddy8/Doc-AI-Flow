#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
# DocAI — Start all services locally
# Usage: bash start.sh
# ─────────────────────────────────────────────────────────

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════╗"
echo "║     DocAI — Document Intelligence Platform     ║"
echo "╚═══════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Check prerequisites ──────────────────────────────────
echo -e "${YELLOW}[1/6] Checking prerequisites...${NC}"

if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed."
    exit 1
fi

if ! command -v tesseract &> /dev/null; then
    echo -e "${YELLOW}WARNING: Tesseract not found. Install with:${NC}"
    echo "  macOS:  brew install tesseract"
    echo "  Ubuntu: sudo apt-get install tesseract-ocr"
    echo ""
    echo "Continuing anyway (OCR will be limited)..."
fi

echo -e "${GREEN}  Python: $(python3 --version)${NC}"

# ── Setup virtual environment ────────────────────────────
echo -e "${YELLOW}[2/6] Setting up virtual environment...${NC}"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}  Created virtual environment${NC}"
fi

source venv/bin/activate
echo -e "${GREEN}  Activated venv${NC}"

# ── Install dependencies ─────────────────────────────────
echo -e "${YELLOW}[3/6] Installing dependencies...${NC}"
pip install -q -r requirements.txt
echo -e "${GREEN}  Dependencies installed${NC}"

# ── Setup .env ───────────────────────────────────────────
echo -e "${YELLOW}[4/6] Checking configuration...${NC}"

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${GREEN}  Created .env from template (mock mode — no API key needed)${NC}"
else
    echo -e "${GREEN}  .env already exists${NC}"
fi

mkdir -p data/chroma data/sample_docs data/mlflow_artifacts

# ── Generate sample documents ────────────────────────────
echo -e "${YELLOW}[5/6] Generating sample documents...${NC}"
python3 data/generate_samples.py 2>/dev/null || echo "  (Sample generation skipped — run manually if needed)"

# ── Launch services ──────────────────────────────────────
echo -e "${YELLOW}[6/6] Starting services...${NC}"
echo ""

# Kill any existing processes on our ports
for port in 8000 8501 5000; do
    lsof -ti:$port 2>/dev/null | xargs kill -9 2>/dev/null || true
done

# Start FastAPI backend
echo -e "${BLUE}Starting API server on http://localhost:8000 ...${NC}"
uvicorn app.main:app --reload --port 8000 --host 0.0.0.0 &
API_PID=$!
sleep 2

# Start MLflow
echo -e "${BLUE}Starting MLflow on http://localhost:5000 ...${NC}"
mlflow server --host 0.0.0.0 --port 5000 \
    --backend-store-uri sqlite:///data/mlflow.db \
    --default-artifact-root ./data/mlflow_artifacts \
    > /dev/null 2>&1 &
MLFLOW_PID=$!
sleep 1

# Start Streamlit frontend
echo -e "${BLUE}Starting frontend on http://localhost:8501 ...${NC}"
streamlit run frontend/app.py --server.port 8501 --server.headless true &
STREAMLIT_PID=$!
sleep 2

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           All services running!                ║${NC}"
echo -e "${GREEN}╠═══════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║                                               ║${NC}"
echo -e "${GREEN}║  Frontend:  http://localhost:8501              ║${NC}"
echo -e "${GREEN}║  API Docs:  http://localhost:8000/docs         ║${NC}"
echo -e "${GREEN}║  MLflow:    http://localhost:5000               ║${NC}"
echo -e "${GREEN}║                                               ║${NC}"
echo -e "${GREEN}║  Mode: $(grep LLM_PROVIDER .env | cut -d= -f2 | head -1 | xargs)                           ║${NC}"
echo -e "${GREEN}║                                               ║${NC}"
echo -e "${GREEN}║  Press Ctrl+C to stop all services            ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════╝${NC}"
echo ""

# Trap Ctrl+C to kill all background processes
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down all services...${NC}"
    kill $API_PID $MLFLOW_PID $STREAMLIT_PID 2>/dev/null || true
    echo -e "${GREEN}Done.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for all background processes
wait
