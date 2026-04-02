"""
Evaluation Pipeline — compares extraction results against ground truth
to measure prompt quality and trigger retraining decisions.

Usage:
    python -m ml.pipelines.evaluation --doc-type invoice --window 100
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

# Ground truth samples for evaluation (in production, load from a database)
GROUND_TRUTH_SAMPLES = {
    "invoice": [
        {
            "ocr_text": """
INVOICE #INV-2024-0847
Date: March 15, 2024
Due: April 15, 2024

FROM: Acme Corp, 123 Business Ave, New York, NY 10001
TO: Widget Inc, 456 Commerce St, San Francisco, CA 94102

Item                    Qty    Unit Price    Total
Cloud Hosting (monthly)  1     $2,400.00    $2,400.00
API Calls (10K bundle)   3       $150.00      $450.00
Support Premium          1       $500.00      $500.00

Subtotal: $3,350.00
Tax (8.5%): $284.75
TOTAL: $3,634.75
Payment Terms: Net 30
            """,
            "expected": {
                "invoice_number": "INV-2024-0847",
                "invoice_date": "March 15, 2024",
                "due_date": "April 15, 2024",
                "vendor_name": "Acme Corp",
                "customer_name": "Widget Inc",
                "total_amount": 3634.75,
                "subtotal": 3350.00,
                "tax": 284.75,
                "currency": "USD",
                "line_items_count": 3,
            },
        },
        {
            "ocr_text": """
Invoice Number: 20240322-A
Invoice Date: 22/03/2024
Payment Due: 22/04/2024

Seller: TechParts GmbH
Hauptstrasse 42, 10115 Berlin, Germany

Buyer: DataFlow Ltd
10 Downing Tech Park, London, EC1A 1BB, UK

Description                 Qty   Price      Amount
GPU Server Rack (Dell)       2    €8,500.00  €17,000.00
Cooling Unit Installation    1    €3,200.00   €3,200.00

Net Total: €20,200.00
VAT (19%): €3,838.00
Grand Total: €24,038.00
Terms: Net 30 days
            """,
            "expected": {
                "invoice_number": "20240322-A",
                "invoice_date": "22/03/2024",
                "due_date": "22/04/2024",
                "vendor_name": "TechParts GmbH",
                "customer_name": "DataFlow Ltd",
                "total_amount": 24038.00,
                "subtotal": 20200.00,
                "tax": 3838.00,
                "currency": "EUR",
                "line_items_count": 2,
            },
        },
    ],
}


def calculate_field_accuracy(
    extracted: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    """
    Compare extracted fields against ground truth.
    Returns per-field and overall accuracy.
    """
    results = {}
    correct = 0
    total = 0

    for field, expected_value in expected.items():
        total += 1
        extracted_value = extracted.get(field)

        if field == "line_items_count":
            items = extracted.get("line_items", [])
            actual = len(items) if isinstance(items, list) else 0
            match = actual == expected_value
        elif isinstance(expected_value, (int, float)):
            try:
                match = abs(float(extracted_value or 0) - expected_value) < 0.01
            except (TypeError, ValueError):
                match = False
        else:
            match = (
                str(extracted_value or "").strip().lower()
                == str(expected_value).strip().lower()
            )

        if match:
            correct += 1

        results[field] = {
            "expected": expected_value,
            "extracted": extracted_value,
            "match": match,
        }

    results["_summary"] = {
        "correct": correct,
        "total": total,
        "accuracy": round(correct / max(total, 1), 4),
    }

    return results


def run_evaluation(
    document_type: str = "invoice",
    prompt_versions: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run evaluation across all ground truth samples for a document type.
    Compare multiple prompt versions if specified.
    """
    samples = GROUND_TRUTH_SAMPLES.get(document_type, [])
    if not samples:
        return {"error": f"No ground truth for type: {document_type}"}

    print(f"\n{'='*60}")
    print(f"  DocAI Evaluation Pipeline — {document_type.upper()}")
    print(f"  Samples: {len(samples)}")
    print(f"{'='*60}\n")

    # In production, this would call the extraction service
    # Here we demonstrate the evaluation framework
    report = {
        "document_type": document_type,
        "total_samples": len(samples),
        "samples": [],
    }

    for i, sample in enumerate(samples):
        print(f"Sample {i+1}/{len(samples)}:")
        print(f"  Expected fields: {list(sample['expected'].keys())}")

        # Placeholder — in production, call extraction_service.extract()
        # extracted = await extraction_service.extract(sample["ocr_text"], ...)
        # For demo, we simulate a result
        simulated_extraction = sample["expected"].copy()  # Perfect extraction
        accuracy = calculate_field_accuracy(simulated_extraction, sample["expected"])

        print(f"  Accuracy: {accuracy['_summary']['accuracy']*100:.1f}%")
        print(f"  Fields: {accuracy['_summary']['correct']}/{accuracy['_summary']['total']}")
        report["samples"].append(accuracy)

    # Overall accuracy
    all_correct = sum(s["_summary"]["correct"] for s in report["samples"])
    all_total = sum(s["_summary"]["total"] for s in report["samples"])
    report["overall_accuracy"] = round(all_correct / max(all_total, 1), 4)

    print(f"\n{'='*60}")
    print(f"  Overall Accuracy: {report['overall_accuracy']*100:.1f}%")
    print(f"  Total Fields Evaluated: {all_total}")
    print(f"{'='*60}\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DocAI Evaluation Pipeline")
    parser.add_argument("--doc-type", default="invoice", help="Document type")
    parser.add_argument("--window", type=int, default=100, help="Evaluation window")
    args = parser.parse_args()

    results = run_evaluation(document_type=args.doc_type)
    print(json.dumps(results, indent=2, default=str))
