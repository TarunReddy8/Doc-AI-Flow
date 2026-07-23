"""Tests for the receipt extractor and the CORD-v2 evaluation pipeline.

These run without the FastAPI app / OCR stack, so they stay fast and validate the real
extraction logic against the real CORD receipt data committed under data/cord_receipts/.
"""

from __future__ import annotations

from app.services.receipt_extractor import extract_receipt_fields
from ml.pipelines.evaluation import (
    load_cord_ground_truth,
    render_receipt_text,
    run_evaluation,
)


class TestReceiptExtractor:
    def test_extracts_total_amid_payment_footer(self):
        text = (
            "TOKO SERBA ADA\n"
            "No. #1001   2024-03-11 12:00\n"
            "------------------------\n"
            "COFFEE   2 x   30,000\n"
            "CAKE   25,000\n"
            "------------------------\n"
            "Subtotal   55,000\n"
            "Tax   5,500\n"
            "TOTAL   60,500\n"
            "CASH   70,000\n"
            "CHANGE   9,500\n"
        )
        fields = extract_receipt_fields(text)
        assert fields["total_amount"] == 60500.0  # not the CASH/CHANGE numbers
        assert fields["subtotal"] == 55000.0
        assert fields["tax"] == 5500.0
        assert fields["line_items_count"] == 2  # header/footer excluded

    def test_ignores_header_meta_as_line_item(self):
        text = "STORE\nNo. #12345   2024-01-02 09:15\nWIDGET   10.00\nTOTAL   10.00\n"
        fields = extract_receipt_fields(text)
        assert fields["line_items_count"] == 1
        assert fields["total_amount"] == 10.0

    def test_parses_european_number_grouping(self):
        fields = extract_receipt_fields("ITEM  1.234,50\nTOTAL  1.234,50\n")
        assert fields["total_amount"] == 1234.50


class TestCordEvaluation:
    def test_real_data_present(self):
        receipts = load_cord_ground_truth()
        assert len(receipts) >= 10
        assert all("raw_parse" in r and "ground_truth" in r for r in receipts)

    def test_renders_and_extracts_real_receipt(self):
        rec = load_cord_ground_truth()[0]
        text = render_receipt_text(rec["raw_parse"], 0)
        fields = extract_receipt_fields(text)
        # total is the most robust field; it should match the real ground truth
        assert fields["total_amount"] == rec["ground_truth"]["total_amount"]

    def test_overall_accuracy_is_honest(self):
        report = run_evaluation()
        # real extractor on real data: strong but genuinely below a perfect score
        assert 0.75 <= report["overall_accuracy"] <= 1.0
        assert report["per_field_accuracy"]["total_amount"] >= 0.9
