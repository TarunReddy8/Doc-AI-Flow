"""
Prometheus metrics for monitoring extraction pipeline health.
Tracks latency, throughput, errors, and extraction quality in real-time.
"""

from prometheus_client import Counter, Histogram, Gauge, Info


# ── Request metrics ──────────────────────────────────────────────────────

REQUESTS_TOTAL = Counter(
    "docai_requests_total",
    "Total extraction requests",
    ["document_type", "status"],
)

REQUEST_DURATION = Histogram(
    "docai_request_duration_seconds",
    "Request processing time",
    ["document_type"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)


# ── OCR metrics ──────────────────────────────────────────────────────────

OCR_CONFIDENCE = Histogram(
    "docai_ocr_confidence",
    "OCR confidence scores",
    ["engine"],
    buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
)

OCR_DURATION = Histogram(
    "docai_ocr_duration_ms",
    "OCR processing time in milliseconds",
    ["engine"],
    buckets=[100, 500, 1000, 2000, 5000, 10000],
)


# ── Extraction metrics ───────────────────────────────────────────────────

EXTRACTION_CONFIDENCE = Histogram(
    "docai_extraction_confidence",
    "LLM extraction confidence scores",
    ["document_type", "prompt_version"],
    buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
)

FIELDS_EXTRACTED = Histogram(
    "docai_fields_extracted",
    "Number of fields successfully extracted",
    ["document_type"],
    buckets=[1, 3, 5, 8, 10, 15, 20],
)

EXTRACTION_ERRORS = Counter(
    "docai_extraction_errors_total",
    "Total extraction errors",
    ["document_type", "error_type"],
)


# ── System health ────────────────────────────────────────────────────────

DOCUMENTS_PROCESSED = Gauge(
    "docai_documents_processed_total",
    "Total documents processed since startup",
)

ACTIVE_EXTRACTIONS = Gauge(
    "docai_active_extractions",
    "Currently processing extractions",
)

VECTOR_STORE_SIZE = Gauge(
    "docai_vector_store_documents",
    "Number of documents in vector store",
)

APP_INFO = Info(
    "docai_app",
    "Application information",
)
