"""
Tests for DocAI extraction pipeline.
Covers OCR validation, extraction parsing, API endpoints, and schema validation.
"""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.extraction import (
    DocumentType,
    ExtractionStatus,
    InvoiceData,
    ContractData,
    ExtractionResult,
    OCRResult,
)
from app.services.extraction_service import ExtractionService
from ml.pipelines.evaluation import calculate_field_accuracy


client = TestClient(app)


# ── Schema Tests ─────────────────────────────────────────────────────────

class TestSchemas:
    def test_invoice_data_valid(self):
        data = InvoiceData(
            invoice_number="INV-001",
            vendor_name="Acme Corp",
            total_amount=1500.00,
            currency="USD",
            line_items=[],
        )
        assert data.invoice_number == "INV-001"
        assert data.total_amount == 1500.00

    def test_invoice_data_nullable_fields(self):
        data = InvoiceData()
        assert data.invoice_number is None
        assert data.line_items == []

    def test_contract_data_valid(self):
        data = ContractData(
            contract_title="Service Agreement",
            parties=["Company A", "Company B"],
            effective_date="2024-01-01",
        )
        assert len(data.parties) == 2

    def test_ocr_result_confidence_bounds(self):
        result = OCRResult(
            raw_text="test",
            confidence=0.95,
            engine_used="tesseract",
            page_count=1,
            processing_time_ms=100.0,
        )
        assert 0 <= result.confidence <= 1

    def test_extraction_result_complete(self):
        result = ExtractionResult(
            document_id="test-123",
            status=ExtractionStatus.COMPLETED,
            document_type=DocumentType.INVOICE,
            extracted_data={"invoice_number": "INV-001"},
            confidence_score=0.85,
        )
        assert result.status == ExtractionStatus.COMPLETED
        assert result.document_type == DocumentType.INVOICE


# ── Extraction Service Tests ─────────────────────────────────────────────

class TestExtractionService:
    def test_json_parsing_clean(self):
        service = ExtractionService.__new__(ExtractionService)
        raw = '{"invoice_number": "INV-001", "total_amount": 1500.00}'
        result = service._parse_json_output(raw)
        assert result["invoice_number"] == "INV-001"

    def test_json_parsing_with_markdown(self):
        service = ExtractionService.__new__(ExtractionService)
        raw = '```json\n{"invoice_number": "INV-002"}\n```'
        result = service._parse_json_output(raw)
        assert result["invoice_number"] == "INV-002"

    def test_json_parsing_with_preamble(self):
        service = ExtractionService.__new__(ExtractionService)
        raw = 'Here is the extracted data:\n{"vendor_name": "Acme"}'
        result = service._parse_json_output(raw)
        assert result["vendor_name"] == "Acme"

    def test_json_parsing_failure_returns_raw(self):
        service = ExtractionService.__new__(ExtractionService)
        raw = "This is not JSON at all"
        result = service._parse_json_output(raw)
        assert "raw_output" in result

    def test_confidence_calculation_invoice_full(self):
        service = ExtractionService.__new__(ExtractionService)
        data = {
            "invoice_number": "INV-001",
            "vendor_name": "Acme",
            "total_amount": 1500,
            "invoice_date": "2024-01-01",
            "customer_name": "Widget Inc",
            "line_items": [{"desc": "item"}],
            "subtotal": 1400,
            "tax": 100,
            "currency": "USD",
        }
        conf = service._calculate_confidence(data, DocumentType.INVOICE)
        assert conf > 0.9  # All critical and most important fields filled

    def test_confidence_calculation_invoice_partial(self):
        service = ExtractionService.__new__(ExtractionService)
        data = {
            "invoice_number": "INV-001",
            "vendor_name": None,
            "total_amount": None,
            "invoice_date": None,
        }
        conf = service._calculate_confidence(data, DocumentType.INVOICE)
        assert conf < 0.5  # Only 1 of 4 critical fields


# ── Evaluation Pipeline Tests ────────────────────────────────────────────

class TestEvaluation:
    def test_field_accuracy_perfect_match(self):
        extracted = {"invoice_number": "INV-001", "total_amount": 1500.00}
        expected = {"invoice_number": "INV-001", "total_amount": 1500.00}
        result = calculate_field_accuracy(extracted, expected)
        assert result["_summary"]["accuracy"] == 1.0

    def test_field_accuracy_partial_match(self):
        extracted = {"invoice_number": "INV-001", "total_amount": 999.99}
        expected = {"invoice_number": "INV-001", "total_amount": 1500.00}
        result = calculate_field_accuracy(extracted, expected)
        assert result["_summary"]["accuracy"] == 0.5

    def test_field_accuracy_case_insensitive(self):
        extracted = {"vendor_name": "acme corp"}
        expected = {"vendor_name": "Acme Corp"}
        result = calculate_field_accuracy(extracted, expected)
        assert result["vendor_name"]["match"] is True

    def test_field_accuracy_missing_field(self):
        extracted = {}
        expected = {"invoice_number": "INV-001"}
        result = calculate_field_accuracy(extracted, expected)
        assert result["invoice_number"]["match"] is False

    def test_line_items_count(self):
        extracted = {"line_items": [{"d": "a"}, {"d": "b"}, {"d": "c"}]}
        expected = {"line_items_count": 3}
        result = calculate_field_accuracy(extracted, expected)
        assert result["line_items_count"]["match"] is True


# ── API Endpoint Tests ───────────────────────────────────────────────────

class TestAPI:
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "DocAI" in data["name"]

    def test_health_endpoint(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "ocr_engine" in data
        assert "llm_provider" in data

    def test_extract_no_file(self):
        response = client.post("/api/v1/extract")
        assert response.status_code == 422  # Validation error

    def test_extract_wrong_file_type(self):
        from io import BytesIO
        response = client.post(
            "/api/v1/extract",
            files={"file": ("test.exe", BytesIO(b"fake"), "application/octet-stream")},
        )
        assert response.status_code == 400

    def test_search_endpoint(self):
        response = client.get("/api/v1/search?query=invoice+total")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    def test_drift_endpoint(self):
        response = client.get("/api/v1/monitoring/drift?document_type=invoice")
        assert response.status_code == 200
