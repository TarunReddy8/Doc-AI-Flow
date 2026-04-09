"""
Pydantic schemas for document extraction — request/response models
with strict validation for production use.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ---------- Enums ----------


class DocumentType(str, Enum):
    INVOICE = "invoice"
    CONTRACT = "contract"
    REPORT = "report"
    RECEIPT = "receipt"
    UNKNOWN = "unknown"


class ExtractionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------- Invoice Schema ----------


class LineItem(BaseModel):
    description: str = Field(..., description="Item description")
    quantity: float | None = Field(None, description="Quantity")
    unit_price: float | None = Field(None, description="Unit price")
    total: float | None = Field(None, description="Line total")


class InvoiceData(BaseModel):
    invoice_number: str | None = Field(None, description="Invoice number/ID")
    invoice_date: str | None = Field(None, description="Invoice date")
    due_date: str | None = Field(None, description="Payment due date")
    vendor_name: str | None = Field(None, description="Vendor/supplier name")
    vendor_address: str | None = Field(None, description="Vendor address")
    customer_name: str | None = Field(None, description="Customer/bill-to name")
    customer_address: str | None = Field(None, description="Customer address")
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: float | None = None
    tax: float | None = None
    total_amount: float | None = None
    currency: str | None = Field(None, description="Currency code (USD, EUR, etc)")
    payment_terms: str | None = None


# ---------- Contract Schema ----------


class ContractData(BaseModel):
    contract_title: str | None = None
    parties: list[str] = Field(default_factory=list)
    effective_date: str | None = None
    expiration_date: str | None = None
    contract_value: float | None = None
    key_terms: list[str] = Field(default_factory=list)
    governing_law: str | None = None
    termination_clause: str | None = None


# ---------- API Schemas ----------


class ExtractionRequest(BaseModel):
    document_type: DocumentType = Field(
        default=DocumentType.UNKNOWN,
        description="Type of document (auto-detected if unknown)",
    )
    extract_tables: bool = Field(default=True)
    store_in_vectordb: bool = Field(default=True)
    prompt_version: str | None = Field(
        None, description="Specific prompt version to use (for A/B testing)"
    )


class OCRResult(BaseModel):
    raw_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    engine_used: str
    page_count: int
    processing_time_ms: float


class ExtractionResult(BaseModel):
    document_id: str
    status: ExtractionStatus
    document_type: DocumentType
    ocr_result: OCRResult | None = None
    extracted_data: dict[str, Any] = Field(default_factory=dict)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    prompt_version: str | None = None
    mlflow_run_id: str | None = None
    processing_time_ms: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    ocr_engine: str
    llm_provider: str
    mlflow_connected: bool
    chroma_connected: bool


class ExtractionMetrics(BaseModel):
    total_documents: int = 0
    avg_confidence: float = 0.0
    avg_processing_time_ms: float = 0.0
    documents_by_type: dict[str, int] = Field(default_factory=dict)
    extraction_success_rate: float = 0.0
