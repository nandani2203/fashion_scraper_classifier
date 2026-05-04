"""
data/augment.py

Generates augmented copies of minority-class images to balance training data.
Targets full_sleeve (155) to match half_sleeve (245).

Run after label_prep.py, before train.py:
    python data/augment.py
"""

import csv
import sys
import random
import logging
from pathlib import Path

from PIL import Image, ImageOps, ImageEnhance

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LABELED_CSV, IMAGES_DIR

log = logging.getLogger(__name__)


def augment_image(img: Image.Image, seed: int) -> Image.Image:
    random.seed(seed)
    ops = [
        lambda i: ImageOps.mirror(i),
        lambda i: i.rotate(random.uniform(-20, 20), expand=False, fillcolor=(200, 200, 200)),
        lambda i: ImageEnhance.Brightness(i).enhance(random.uniform(0.75, 1.25)),
        lambda i: ImageEnhance.Color(i).enhance(random.uniform(0.75, 1.25)),
        lambda i: ImageEnhance.Contrast(i).enhance(random.uniform(0.85, 1.15)),
    ]
    chosen = random.sample(ops, k=3)
    for op in chosen:
        img = op(img)
    return img


def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(LABELED_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    full_rows = [r for r in rows if r["sleeve_type"] == "full_sleeve"]
    half_count = sum(1 for r in rows if r["sleeve_type"] == "half_sleeve")

    log.info(f"Current — full_sleeve: {len(full_rows)}, half_sleeve: {half_count}")

    new_rows = []
    skipped = 0

    for i, row in enumerate(full_rows):
        src = Path(row["local_image"])
        if not src.exists():
            skipped += 1
            continue

        try:
            img = Image.open(src).convert("RGB")
            aug = augment_image(img, seed=i)

            aug_name = f"aug_{src.name}"
            aug_path = IMAGES_DIR / aug_name
            aug.save(aug_path, "JPEG", quality=85)

            new_row = dict(row)
            new_row["local_image"] = str(aug_path)
            new_row["image_url"]   = row["image_url"] + "__aug"
            new_rows.append(new_row)

        except Exception as e:
            log.warning(f"Skipped {src.name}: {e}")
            skipped += 1

    all_rows = rows + new_rows

    with open(LABELED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    new_full   = sum(1 for r in all_rows if r["sleeve_type"] == "full_sleeve")
    new_male   = sum(1 for r in all_rows if r["gender"] == "male")
    new_female = sum(1 for r in all_rows if r["gender"] == "female")

    log.info(f"Generated {len(new_rows)} augmented images ({skipped} skipped)")
    log.info(f"After — full_sleeve: {new_full}, half_sleeve: {half_count}")
    log.info(f"After — male: {new_male}, female: {new_female}")
    log.info(f"Total samples: {len(all_rows)}")
    log.info("Next: python model/train.py")


if __name__ == "__main__":
    run()
