"""
Mock Extraction Service — returns realistic fake extraction data
so you can run the full pipeline WITHOUT an API key.

Set LLM_PROVIDER=mock in your .env to use this.
"""

from __future__ import annotations

import random
from typing import Any

from app.core.logging import get_logger
from app.schemas.extraction import DocumentType

logger = get_logger(__name__)


# Realistic mock data pools
MOCK_VENDORS = [
    "Acme Cloud Services", "TechParts GmbH", "DataFlow Solutions",
    "Quantum Computing Inc", "NeuralNet Labs", "ServerStack LLC",
]

MOCK_CUSTOMERS = [
    "Widget Corp", "StartupAI Inc", "MegaTech Industries",
    "Innovation Labs", "CloudFirst Solutions", "DevOps Masters",
]

MOCK_LINE_ITEMS = [
    {"description": "Cloud Hosting (monthly)", "quantity": 1, "unit_price": 2400.00, "total": 2400.00},
    {"description": "API Calls (10K bundle)", "quantity": 3, "unit_price": 150.00, "total": 450.00},
    {"description": "Premium Support", "quantity": 1, "unit_price": 500.00, "total": 500.00},
    {"description": "GPU Compute (hourly)", "quantity": 120, "unit_price": 3.50, "total": 420.00},
    {"description": "Data Storage (TB)", "quantity": 5, "unit_price": 80.00, "total": 400.00},
    {"description": "SSL Certificate (annual)", "quantity": 2, "unit_price": 75.00, "total": 150.00},
    {"description": "Load Balancer", "quantity": 1, "unit_price": 350.00, "total": 350.00},
    {"description": "CDN Bandwidth (TB)", "quantity": 10, "unit_price": 45.00, "total": 450.00},
]

MOCK_CONTRACT_TERMS = [
    "30-day termination notice required",
    "Quarterly performance reviews",
    "Data must be encrypted at rest and in transit",
    "SLA: 99.95% uptime guarantee",
    "Intellectual property remains with originator",
    "Non-compete clause for 12 months post-termination",
]


def mock_classify(ocr_text: str) -> DocumentType:
    """Classify document based on keyword matching."""
    text_lower = ocr_text.lower()
    if any(w in text_lower for w in ["invoice", "bill", "amount due", "total"]):
        return DocumentType.INVOICE
    elif any(w in text_lower for w in ["contract", "agreement", "party", "terms"]):
        return DocumentType.CONTRACT
    elif any(w in text_lower for w in ["report", "summary", "analysis", "findings"]):
        return DocumentType.REPORT
    elif any(w in text_lower for w in ["receipt", "paid", "transaction"]):
        return DocumentType.RECEIPT
    return DocumentType.INVOICE  # default for demo


def mock_extract_invoice(ocr_text: str) -> dict[str, Any]:
    """Generate realistic invoice extraction from OCR text."""
    # Try to pull real data from the OCR text first
    lines = ocr_text.strip().split("\n")
    inv_number = None
    for line in lines:
        if any(kw in line.lower() for kw in ["invoice", "inv#", "inv-", "number"]):
            parts = line.split(":")
            if len(parts) > 1:
                inv_number = parts[-1].strip()
                break

    selected_items = random.sample(MOCK_LINE_ITEMS, k=min(3, len(MOCK_LINE_ITEMS)))
    subtotal = sum(item["total"] for item in selected_items)
    tax = round(subtotal * 0.085, 2)
    total = round(subtotal + tax, 2)

    return {
        "invoice_number": inv_number or f"INV-2024-{random.randint(1000, 9999)}",
        "invoice_date": "2024-03-15",
        "due_date": "2024-04-15",
        "vendor_name": random.choice(MOCK_VENDORS),
        "vendor_address": "123 Tech Boulevard, San Francisco, CA 94105",
        "customer_name": random.choice(MOCK_CUSTOMERS),
        "customer_address": "456 Innovation Drive, Austin, TX 78701",
        "line_items": selected_items,
        "subtotal": subtotal,
        "tax": tax,
        "total_amount": total,
        "currency": "USD",
        "payment_terms": "Net 30",
    }


def mock_extract_contract(ocr_text: str) -> dict[str, Any]:
    """Generate realistic contract extraction."""
    return {
        "contract_title": "Master Service Agreement",
        "parties": [random.choice(MOCK_VENDORS), random.choice(MOCK_CUSTOMERS)],
        "effective_date": "2024-01-01",
        "expiration_date": "2025-12-31",
        "contract_value": round(random.uniform(50000, 500000), 2),
        "key_terms": random.sample(MOCK_CONTRACT_TERMS, k=3),
        "governing_law": "State of California",
        "termination_clause": "Either party may terminate with 30 days written notice",
    }


def mock_extract(
    ocr_text: str, document_type: DocumentType
) -> tuple[dict[str, Any], str, float]:
    """
    Main mock extraction entry point.
    Returns: (extracted_data, prompt_version, confidence_score)
    """
    logger.info("mock_extraction_running", doc_type=document_type.value)

    if document_type == DocumentType.INVOICE:
        data = mock_extract_invoice(ocr_text)
    elif document_type == DocumentType.CONTRACT:
        data = mock_extract_contract(ocr_text)
    else:
        data = mock_extract_invoice(ocr_text)  # fallback

    # Simulate realistic confidence based on OCR text quality
    text_length = len(ocr_text.strip())
    if text_length > 500:
        confidence = round(random.uniform(0.82, 0.96), 4)
    elif text_length > 100:
        confidence = round(random.uniform(0.65, 0.85), 4)
    else:
        confidence = round(random.uniform(0.40, 0.65), 4)

    return data, "mock_v1.0", confidence
