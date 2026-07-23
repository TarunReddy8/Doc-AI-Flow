"""Rule-based receipt field extractor.

Parses receipt-style OCR text into structured fields — line items, subtotal, tax,
and grand total — using label heuristics and number parsing that tolerate the label
variation real receipts exhibit (TOTAL / Amount Due / Grand Total; Sub Total /
Subtotal; Tax / VAT / PPN). This is the deterministic, offline extraction path; the
LLM path (app/services/extraction_service.py) handles messier layouts when a key is set.

It is exercised on real CORD receipt data by ml/pipelines/evaluation.py.
"""

from __future__ import annotations

import re
from typing import Any

_NUMBER = re.compile(r"[-+]?\d[\d.,]*")

TOTAL_LABELS = ("grand total", "amount due", "total")
SUBTOTAL_LABELS = ("sub total", "subtotal", "sub-total")
TAX_LABELS = ("tax", "vat", "ppn", "gst")
# labels that mark a line as a summary row, not a purchasable line item
_SUMMARY_LABELS = TOTAL_LABELS + SUBTOTAL_LABELS + TAX_LABELS + (
    "change", "cash", "credit", "card", "payment", "qty", "count", "discount")


def _parse_number(token: str) -> float | None:
    """Parse a receipt number, handling both 1,234.50 and 1.234,50 groupings."""
    token = token.strip()
    if not token:
        return None
    if "," in token and "." in token:
        # the last separator is the decimal point
        if token.rfind(",") > token.rfind("."):
            token = token.replace(".", "").replace(",", ".")
        else:
            token = token.replace(",", "")
    else:
        # a lone comma is a thousands separator on these receipts
        token = token.replace(",", "")
    try:
        return round(float(token), 2)
    except ValueError:
        return None


def _last_number(line: str) -> float | None:
    matches = _NUMBER.findall(line)
    return _parse_number(matches[-1]) if matches else None


def _find_labeled_amount(lines: list[str], labels: tuple[str, ...]) -> float | None:
    """Return the amount on the last line whose label matches (last wins on receipts)."""
    found: float | None = None
    for line in lines:
        low = line.lower()
        if any(lbl in low for lbl in labels):
            amount = _last_number(line)
            if amount is not None:
                found = amount
    return found


_DIVIDER = re.compile(r"^[-=_*.\s]+$")
_DATE = re.compile(r"\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}")
_TIME = re.compile(r"\d{1,2}:\d{2}")


def _is_meta_line(line: str) -> bool:
    """Header/footer noise that is not a purchasable line item (dates, receipt #, phone)."""
    low = line.lower()
    if _DATE.search(line) or _TIME.search(line):
        return True
    if "#" in line or "tel" in low or "no." in low or "receipt" in low:
        return True
    return False


def _first_summary_index(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        low = line.lower()
        if any(lbl in low for lbl in TOTAL_LABELS + SUBTOTAL_LABELS + TAX_LABELS):
            return i
    return len(lines)


def extract_receipt_fields(text: str) -> dict[str, Any]:
    """Extract {line_items, subtotal, tax, total_amount} from receipt OCR text."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    total = _find_labeled_amount(lines, TOTAL_LABELS)
    # subtotal must not be mistaken for total; check subtotal labels independently
    subtotal = _find_labeled_amount(lines, SUBTOTAL_LABELS)
    tax = _find_labeled_amount(lines, TAX_LABELS)

    # Line items live in the region between the header and the totals block. Anchoring on
    # the totals block keeps payment/change footer lines and header meta out of the items.
    summary_start = _first_summary_index(lines)
    body = lines[1:summary_start] if summary_start > 1 else lines[:summary_start]

    line_items: list[dict[str, Any]] = []
    for line in body:
        if _DIVIDER.match(line) or _is_meta_line(line):
            continue
        low = line.lower()
        if any(lbl in low for lbl in _SUMMARY_LABELS):
            continue
        amount = _last_number(line)
        if amount is None:
            continue
        desc = _NUMBER.split(line, 1)[0].strip(" .:-x@")
        if not desc:
            continue
        qty = None
        qty_match = re.search(r"(\d+)\s*(?:x|@)", low)
        if qty_match:
            qty = float(qty_match.group(1))
        line_items.append({"description": desc, "quantity": qty, "total": amount})

    return {
        "line_items": line_items,
        "line_items_count": len(line_items),
        "subtotal": subtotal,
        "tax": tax,
        "total_amount": total,
    }
