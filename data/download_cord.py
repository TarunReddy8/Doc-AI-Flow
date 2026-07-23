"""Download REAL receipt images + ground-truth labels from the CORD-v2 dataset.

CORD (Consolidated Receipt Dataset) is a public benchmark of photographed store
receipts with human-annotated fields (line items, subtotal, tax, total). We pull a
small sample of real receipt images and their real ground-truth parses, then flatten
each parse into DocAI's receipt schema (line items + subtotal + tax + total). The
result is committed under data/cord_receipts/ so the evaluation pipeline runs on real
data with no network access.

Source : naver-clova-ix/cord-v2 (Hugging Face), license CC-BY-4.0 — see NOTICE.
Usage  : python data/download_cord.py --n 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

ROWS_API = ("https://datasets-server.huggingface.co/rows"
            "?dataset=naver-clova-ix/cord-v2&config=default&split=test")
OUT_DIR = Path(__file__).resolve().parent / "cord_receipts"


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = str(value).replace(",", "").replace(" ", "")
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


def flatten_parse(gt_parse: dict) -> dict:
    """Convert a CORD gt_parse into DocAI's receipt-field ground truth."""
    menu = gt_parse.get("menu", [])
    if isinstance(menu, dict):
        menu = [menu]
    line_items = []
    for item in menu:
        if not isinstance(item, dict):
            continue
        line_items.append({
            "description": str(item.get("nm", "")).strip(),
            "quantity": _to_float(item.get("cnt")),
            "total": _to_float(item.get("price")),
        })

    sub = gt_parse.get("sub_total", {}) or {}
    total = gt_parse.get("total", {}) or {}
    return {
        "line_items": line_items,
        "line_items_count": len(line_items),
        "subtotal": _to_float(sub.get("subtotal_price")),
        "tax": _to_float(sub.get("tax_price")),
        "total_amount": _to_float(total.get("total_price")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=12, help="number of receipts to download")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    offset = 0
    while len(records) < args.n:
        resp = httpx.get(f"{ROWS_API}&offset={offset}&length=20", timeout=90)
        resp.raise_for_status()
        batch = resp.json().get("rows", [])
        if not batch:
            break
        for item in batch:
            if len(records) >= args.n:
                break
            row = item["row"]
            gt = json.loads(row["ground_truth"])["gt_parse"]
            fields = flatten_parse(gt)
            # keep only well-formed receipts (real total + at least one line item)
            if fields["total_amount"] is None or not fields["line_items"]:
                continue
            idx = len(records)
            img_url = row["image"]["src"]
            img = httpx.get(img_url, timeout=90).content
            name = f"receipt_{idx:03d}.jpg"
            (OUT_DIR / name).write_bytes(img)
            # store the full real parse (for faithful rendering) + clean scored fields
            records.append({"image": name, "raw_parse": gt, "ground_truth": fields})
            print(f"  saved {name}  (total={fields['total_amount']}, "
                  f"items={fields['line_items_count']})")
        offset += 20

    gt_path = OUT_DIR / "ground_truth.jsonl"
    gt_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8")
    print(f"\nwrote {len(records)} real CORD receipts + labels to {OUT_DIR}")


if __name__ == "__main__":
    main()
