"""
DocAI — AI Document Intelligence Platform
Main application entry point with middleware, CORS, and metrics.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.api.routes import router
from monitoring.metrics import APP_INFO


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    setup_logging()
    log = get_logger("docai")
    settings = get_settings()

    APP_INFO.info({
        "version": "1.0.0",
        "ocr_engine": settings.ocr_engine,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
    })

    log.info(
        "application_started",
        ocr=settings.ocr_engine,
        llm=f"{settings.llm_provider}/{settings.llm_model}",
    )
    yield
    log.info("application_shutdown")


app = FastAPI(
    title="DocAI — Document Intelligence Platform",
    description=(
        "AI-powered document extraction pipeline with OCR, LLM extraction, "
        "vector search, and full MLOps lifecycle management."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# API routes
app.include_router(router, prefix="/api/v1", tags=["extraction"])


@app.get("/")
async def root():
    return {
        "name": "DocAI — Document Intelligence Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
