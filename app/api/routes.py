"""
API Routes — endpoints for document upload, extraction, search, and monitoring.
Production-grade with proper error handling, metrics, and logging.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Depends
from fastapi.responses import JSONResponse

from app.core.config import get_settings, Settings
from app.core.logging import get_logger
from app.schemas.extraction import (
    ExtractionRequest,
    ExtractionResult,
    ExtractionStatus,
    DocumentType,
    HealthResponse,
    ExtractionMetrics,
)
from app.services.ocr_service import get_ocr_service
from app.services.extraction_service import get_extraction_service
from app.services.vector_service import get_vector_service
from app.services.mlflow_service import get_mlflow_service
from monitoring.metrics import (
    REQUESTS_TOTAL,
    REQUEST_DURATION,
    OCR_CONFIDENCE,
    OCR_DURATION,
    EXTRACTION_CONFIDENCE,
    FIELDS_EXTRACTED,
    EXTRACTION_ERRORS,
    DOCUMENTS_PROCESSED,
    ACTIVE_EXTRACTIONS,
)

logger = get_logger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}


def get_config() -> Settings:
    return get_settings()


# ── Health Check ─────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check(config: Settings = Depends(get_config)):
    """System health check — verifies all services are connected."""
    vector_svc = get_vector_service()
    mlflow_svc = get_mlflow_service()
    chroma_stats = await vector_svc.get_stats()

    return HealthResponse(
        status="healthy",
        ocr_engine=config.ocr_engine,
        llm_provider=config.llm_provider,
        mlflow_connected=mlflow_svc.is_connected,
        chroma_connected=chroma_stats.get("status") == "connected",
    )


# ── Document Extraction (Main Endpoint) ─────────────────────────────────

@router.post("/extract", response_model=ExtractionResult)
async def extract_document(
    file: UploadFile = File(..., description="Document file (PDF, PNG, JPG)"),
    document_type: DocumentType = Query(
        default=DocumentType.UNKNOWN,
        description="Document type (auto-detected if unknown)",
    ),
    store_in_vectordb: bool = Query(default=True),
    prompt_version: str | None = Query(
        default=None, description="Prompt version for A/B testing"
    ),
    config: Settings = Depends(get_config),
):
    """
    Upload a document and extract structured data.

    Pipeline: Upload → OCR → Classification → LLM Extraction → Storage → MLflow Tracking

    This is the core endpoint. Every run is tracked in MLflow for experiment comparison.
    """
    document_id = str(uuid.uuid4())
    start_time = time.time()
    warnings: list[str] = []

    ACTIVE_EXTRACTIONS.inc()

    try:
        # ── Step 1: Validate file ────────────────────────────────────
        _validate_file(file, config)

        # ── Step 2: Read file ────────────────────────────────────────
        content = await file.read()
        logger.info(
            "file_received",
            doc_id=document_id,
            filename=file.filename,
            size_bytes=len(content),
        )

        # ── Step 3: OCR ──────────────────────────────────────────────
        ocr_service = get_ocr_service()
        ocr_result = await ocr_service.extract_text(content, file.filename)

        OCR_CONFIDENCE.labels(engine=ocr_result.engine_used).observe(
            ocr_result.confidence
        )
        OCR_DURATION.labels(engine=ocr_result.engine_used).observe(
            ocr_result.processing_time_ms
        )

        if not ocr_result.raw_text.strip():
            raise HTTPException(
                status_code=422,
                detail="OCR produced no text. The document may be blank or unreadable.",
            )

        if ocr_result.confidence < config.ocr_confidence_threshold:
            warnings.append(
                f"Low OCR confidence ({ocr_result.confidence:.2f}). "
                "Results may be inaccurate."
            )

        # ── Step 4: Classify (if needed) ─────────────────────────────
        extraction_service = get_extraction_service()

        if document_type == DocumentType.UNKNOWN:
            document_type = await extraction_service.classify_document(
                ocr_result.raw_text
            )
            logger.info("document_classified", type=document_type.value)

        # ── Step 5: LLM Extraction ───────────────────────────────────
        extracted_data, version_used, confidence = await extraction_service.extract(
            ocr_text=ocr_result.raw_text,
            document_type=document_type,
            prompt_version=prompt_version,
        )

        EXTRACTION_CONFIDENCE.labels(
            document_type=document_type.value,
            prompt_version=version_used,
        ).observe(confidence)

        fields_count = len([v for v in extracted_data.values() if v is not None])
        FIELDS_EXTRACTED.labels(document_type=document_type.value).observe(
            fields_count
        )

        # ── Step 6: Store in ChromaDB ────────────────────────────────
        if store_in_vectordb:
            vector_service = get_vector_service()
            stored = await vector_service.store_document(
                document_id=document_id,
                ocr_text=ocr_result.raw_text,
                extracted_data=extracted_data,
                document_type=document_type.value,
                metadata={
                    "filename": file.filename,
                    "confidence": confidence,
                    "prompt_version": version_used,
                },
            )
            if not stored:
                warnings.append("Failed to store in vector database.")

        # ── Step 7: Log to MLflow ────────────────────────────────────
        total_time = (time.time() - start_time) * 1000
        mlflow_service = get_mlflow_service()
        run_id = await mlflow_service.log_extraction_run(
            document_id=document_id,
            document_type=document_type.value,
            prompt_version=version_used,
            ocr_confidence=ocr_result.confidence,
            extraction_confidence=confidence,
            fields_extracted=fields_count,
            total_fields=len(extracted_data),
            processing_time_ms=total_time,
            ocr_engine=ocr_result.engine_used,
            llm_model=config.llm_model,
            warnings=warnings,
        )

        # ── Build response ───────────────────────────────────────────
        REQUESTS_TOTAL.labels(
            document_type=document_type.value, status="success"
        ).inc()
        DOCUMENTS_PROCESSED.inc()
        REQUEST_DURATION.labels(document_type=document_type.value).observe(
            total_time / 1000
        )

        return ExtractionResult(
            document_id=document_id,
            status=ExtractionStatus.COMPLETED,
            document_type=document_type,
            ocr_result=ocr_result,
            extracted_data=extracted_data,
            confidence_score=confidence,
            prompt_version=version_used,
            mlflow_run_id=run_id,
            processing_time_ms=round(total_time, 2),
            warnings=warnings,
        )

    except HTTPException:
        raise
    except Exception as e:
        REQUESTS_TOTAL.labels(
            document_type=document_type.value, status="error"
        ).inc()
        EXTRACTION_ERRORS.labels(
            document_type=document_type.value,
            error_type=type(e).__name__,
        ).inc()
        logger.error("extraction_pipeline_error", doc_id=document_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    finally:
        ACTIVE_EXTRACTIONS.dec()


# ── Semantic Search ──────────────────────────────────────────────────────

@router.get("/search")
async def search_documents(
    query: str = Query(..., description="Search query"),
    n_results: int = Query(default=5, ge=1, le=20),
    document_type: str | None = Query(default=None),
):
    """Search processed documents using semantic similarity."""
    vector_service = get_vector_service()
    results = await vector_service.search_similar(
        query=query,
        n_results=n_results,
        document_type=document_type,
    )
    return {"query": query, "results": results, "count": len(results)}


# ── Prompt A/B Testing ───────────────────────────────────────────────────

@router.get("/prompts/compare")
async def compare_prompts(
    document_type: str = Query(
        default="invoice", description="Document type to compare"
    ),
):
    """Compare prompt version performance for A/B testing decisions."""
    mlflow_service = get_mlflow_service()
    comparison = await mlflow_service.get_prompt_comparison(document_type)
    return {"document_type": document_type, "versions": comparison}


# ── Drift Detection ─────────────────────────────────────────────────────

@router.get("/monitoring/drift")
async def check_drift(
    document_type: str = Query(default="invoice"),
    window: int = Query(default=50, ge=10, le=500),
):
    """Check for extraction quality drift against baseline."""
    mlflow_service = get_mlflow_service()
    result = await mlflow_service.check_drift(document_type, window)
    return result


# ── Metrics ──────────────────────────────────────────────────────────────

@router.get("/metrics/summary", response_model=ExtractionMetrics)
async def get_metrics_summary():
    """Get high-level extraction pipeline metrics."""
    vector_service = get_vector_service()
    stats = await vector_service.get_stats()
    return ExtractionMetrics(
        total_documents=stats.get("total_documents", 0),
    )


# ── Helpers ──────────────────────────────────────────────────────────────

def _validate_file(file: UploadFile, config: Settings) -> None:
    """Validate uploaded file type and size."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}",
        )

    if file.size and file.size > config.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {config.max_file_size_mb}MB",
        )
