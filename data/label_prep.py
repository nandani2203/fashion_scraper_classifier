"""
data/label_prep.py

Reads raw_products.csv produced by the scraper.
Filters unknowns, encodes labels as integers, checks class balance,
writes labeled_products.csv ready for FashionDataset.
"""

import csv
import logging
from pathlib import Path
from collections import Counter


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAW_CSV, LABELED_CSV, GENDER_CLASSES, SLEEVE_CLASSES

log = logging.getLogger(__name__)

GENDER_TO_IDX = {label: idx for idx, label in enumerate(GENDER_CLASSES)}
SLEEVE_TO_IDX = {label: idx for idx, label in enumerate(SLEEVE_CLASSES)}

OUTPUT_FIELDS = [
    "title", "image_url", "product_url",
    "gender", "sleeve_type",
    "gender_label", "sleeve_label",
    "local_image", "source",
]


def load_raw(path: Path = RAW_CSV) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at {path}.\n"
            f"Run the scraper first:  python scraper/flipkart_scraper.py"
        )
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_labeled(rows: list[dict], path: Path = LABELED_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Saved {len(rows)} labeled rows → {path}")


def prepare(raw_path: Path = RAW_CSV, out_path: Path = LABELED_CSV) -> list[dict]:
    """Full pipeline: load → filter → encode → balance check → save."""
    rows = load_raw(raw_path)
    log.info(f"Raw rows loaded: {len(rows)}")

    # ── filter ────────────────────────────────────────────────────────────
    kept    = []
    dropped = {"unknown_gender": 0, "unknown_sleeve": 0, "missing_image": 0}

    for row in rows:
        gender = row.get("gender", "").strip()
        sleeve = row.get("sleeve_type", "").strip()
        img    = row.get("local_image", "").strip()

        if gender not in GENDER_CLASSES:
            dropped["unknown_gender"] += 1
            continue
        if sleeve not in SLEEVE_CLASSES:
            dropped["unknown_sleeve"] += 1
            continue
        if not img or not Path(img).exists():   # empty OR missing file
            dropped["missing_image"] += 1
            continue

        kept.append(row)

    log.info(
        f"Kept: {len(kept)} | dropped → "
        f"unknown_gender={dropped['unknown_gender']}, "
        f"unknown_sleeve={dropped['unknown_sleeve']}, "
        f"missing_image={dropped['missing_image']}"
    )

    if not kept:
        raise ValueError(
            "No rows remain after filtering. "
            "Check images were downloaded and labels extracted correctly."
        )

    # ── encode ────────────────────────────────────────────────────────────
    for row in kept:
        row["gender_label"] = GENDER_TO_IDX[row["gender"]]
        row["sleeve_label"] = SLEEVE_TO_IDX[row["sleeve_type"]]

    # ── balance check ─────────────────────────────────────────────────────
    for attr, counter in [
        ("gender",     Counter(r["gender"]      for r in kept)),
        ("sleeve",     Counter(r["sleeve_type"] for r in kept)),
    ]:
        for cls, n in counter.items():
            pct = 100 * n / len(kept)
            log.info(f"  {attr:<8} {cls:<15} {n:>4} ({pct:.1f}%)")
            if pct < 20.0:
                log.warning(
                    f"Class imbalance: '{cls}' is only {pct:.1f}% of data. "
                    f"Consider scraping more '{cls}' products."
                )

    save_labeled(kept, out_path)
    return kept


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    prepare()
    print("\nLabel prep complete. Next: python model/train.py")