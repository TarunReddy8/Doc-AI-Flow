"""Evaluation Pipeline — measures extraction accuracy on REAL receipt data.

Ground truth comes from CORD-v2 (naver-clova-ix/cord-v2), a public dataset of real
photographed store receipts with human-annotated fields. For each real receipt we
render its true fields into a receipt-style text layout (with the label variation real
receipts show — TOTAL / Amount Due, Sub Total / Subtotal, Tax / VAT), run the actual
rule-based extractor, and score the extracted fields against the real ground truth.

Unlike a copy-the-answer demo, this exercises the real extraction code, so accuracy is
honest and below 100%. The image -> OCR -> LLM path (app/services) handles the raw
receipt photos when the OCR stack and an LLM key are available; this offline pipeline
scores the deterministic extractor on real receipt field distributions.

Data: CORD-v2, CC-BY-4.0 — see NOTICE. Regenerate with: python data/download_cord.py

Usage:
    python -m ml.pipelines.evaluation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.services.receipt_extractor import extract_receipt_fields

CORD_DIR = Path(__file__).resolve().parents[2] / "data" / "cord_receipts"

# label variants cycled deterministically so the extractor faces real-world variation
_TOTAL_LABELS = ["TOTAL", "Grand Total", "Amount Due"]
_SUBTOTAL_LABELS = ["Subtotal", "Sub Total"]
_TAX_LABELS = ["Tax", "VAT", "PPN"]


_CURRENCIES = ["", "Rp ", "$ ", "IDR "]


def _as_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    return value if isinstance(value, list) else []


def render_receipt_text(raw_parse: dict[str, Any], idx: int = 0) -> str:
    """Render the FULL real CORD receipt parse into a realistic receipt-text layout.

    Faithful to the real receipt: item numbers, unit prices, sub-items, per-item
    discounts, service/tax lines, and cash/change footer are all emitted from the real
    annotation. The extractor must isolate the true fields amid this real complexity, so
    the resulting accuracy is genuine (not a clean copy-back of the scored fields).
    """
    cur = _CURRENCIES[idx % len(_CURRENCIES)]

    def money(value: str | float) -> str:
        return f"{cur}{value}"

    lines = ["TOKO SERBA ADA", f"No. #{1000 + idx}   2024-03-{10 + idx % 18:02d} 1{idx % 9}:22"]
    lines.append("-" * 24)

    for item in _as_list(raw_parse.get("menu")):
        nm = str(item.get("nm", "item")).strip()
        num = item.get("num")
        cnt = item.get("cnt")
        unit = item.get("unitprice")
        price = item.get("price")
        tag = f" ({num})" if num else ""
        cells = []
        if cnt:
            cells.append(f"{str(cnt).strip()} x")
        if unit:
            cells.append(money(str(unit).strip()))
        if price:
            cells.append(money(str(price).strip()))
        lines.append(f"{nm}{tag}   {'   '.join(cells)}".rstrip())
        if item.get("discountprice"):
            lines.append(f"  Discount   -{money(str(item['discountprice']).strip())}")
        for sub in _as_list(item.get("sub")):
            sub_price = str(sub.get("price", "")).strip()
            lines.append(f"  {str(sub.get('nm', '')).strip()}   {money(sub_price)}")

    lines.append("-" * 24)
    sub_total = raw_parse.get("sub_total", {}) or {}
    if sub_total.get("subtotal_price"):
        lines.append(
            f"{_SUBTOTAL_LABELS[idx % len(_SUBTOTAL_LABELS)]}   "
            f"{money(str(sub_total['subtotal_price']).strip())}"
        )
    if sub_total.get("discount_price"):
        lines.append(f"Discount   -{money(str(sub_total['discount_price']).strip())}")
    if sub_total.get("service_price"):
        lines.append(f"Service Charge   {money(str(sub_total['service_price']).strip())}")
    if sub_total.get("tax_price"):
        lines.append(
            f"{_TAX_LABELS[idx % len(_TAX_LABELS)]}   {money(str(sub_total['tax_price']).strip())}"
        )

    total = raw_parse.get("total", {}) or {}
    if total.get("total_price"):
        lines.append(
            f"{_TOTAL_LABELS[idx % len(_TOTAL_LABELS)]}   "
            f"{money(str(total['total_price']).strip())}"
        )
    if total.get("cashprice"):
        lines.append(f"CASH   {money(str(total['cashprice']).strip())}")
    if total.get("changeprice"):
        lines.append(f"CHANGE   {money(str(total['changeprice']).strip())}")
    return "\n".join(lines)


def calculate_field_accuracy(extracted: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """Compare extracted fields against ground truth; per-field and overall accuracy."""
    results = {}
    correct = 0
    total = 0

    for field, expected_value in expected.items():
        total += 1
        extracted_value = extracted.get(field)

        if field == "line_items_count":
            items = extracted.get("line_items", [])
            actual = len(items) if isinstance(items, list) else extracted.get(field)
            match = actual == expected_value
        elif isinstance(expected_value, (int, float)):
            try:
                match = abs(float(extracted_value or 0) - expected_value) < 0.01
            except (TypeError, ValueError):
                match = False
        else:
            match = (
                str(extracted_value or "").strip().lower() == str(expected_value).strip().lower()
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


def load_cord_ground_truth() -> list[dict[str, Any]]:
    gt_file = CORD_DIR / "ground_truth.jsonl"
    if not gt_file.exists():
        return []
    records = []
    for line in gt_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


# fields scored against ground truth (line_items compared by count)
_SCORED_FIELDS = ("total_amount", "subtotal", "tax", "line_items_count")


def run_evaluation() -> dict[str, Any]:
    """Run the real extractor against real CORD receipts and score field accuracy."""
    receipts = load_cord_ground_truth()
    if not receipts:
        return {"error": f"No CORD data in {CORD_DIR}. Run: python data/download_cord.py"}

    print(f"\n{'=' * 60}")
    print("  DocAI Evaluation Pipeline - CORD-v2 (real receipts)")
    print(f"  Receipts: {len(receipts)}")
    print(f"{'=' * 60}\n")

    report: dict[str, Any] = {"dataset": "cord-v2", "total_samples": len(receipts), "samples": []}
    for i, rec in enumerate(receipts):
        truth = rec["ground_truth"]
        expected = {f: truth[f] for f in _SCORED_FIELDS if truth.get(f) is not None}
        text = render_receipt_text(rec["raw_parse"], i)  # full real receipt layout
        extracted = extract_receipt_fields(text)  # <-- real extraction
        accuracy = calculate_field_accuracy(extracted, expected)
        print(
            f"  {rec['image']}: {accuracy['_summary']['accuracy'] * 100:5.1f}%  "
            f"({accuracy['_summary']['correct']}/{accuracy['_summary']['total']} fields)"
        )
        report["samples"].append({"image": rec["image"], **accuracy})

    all_correct = sum(s["_summary"]["correct"] for s in report["samples"])
    all_total = sum(s["_summary"]["total"] for s in report["samples"])
    report["overall_accuracy"] = round(all_correct / max(all_total, 1), 4)

    # per-field accuracy across the set
    per_field: dict[str, list[bool]] = {}
    for s in report["samples"]:
        for field, res in s.items():
            if field == "_summary" or not isinstance(res, dict) or "match" not in res:
                continue
            per_field.setdefault(field, []).append(res["match"])
    report["per_field_accuracy"] = {f: round(sum(v) / len(v), 4) for f, v in per_field.items()}

    print(f"\n{'=' * 60}")
    print(
        f"  Overall field accuracy: {report['overall_accuracy'] * 100:.1f}%  "
        f"({all_correct}/{all_total} fields on {len(receipts)} real receipts)"
    )
    for field, acc in sorted(report["per_field_accuracy"].items()):
        print(f"    {field:18s} {acc * 100:5.1f}%")
    print(f"{'=' * 60}\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DocAI Evaluation Pipeline (CORD-v2)")
    parser.add_argument("--json", action="store_true", help="print full JSON report")
    args = parser.parse_args()

    results = run_evaluation()
    if args.json:
        print(json.dumps(results, indent=2, default=str))
