"""
data/dataset.py

PyTorch Dataset and DataLoaders for fashion classification.
"""

import csv
import logging
from pathlib import Path
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    LABELED_CSV, IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD,
    BATCH_SIZE, VAL_SPLIT,
)

log = logging.getLogger(__name__)


def get_train_transforms() -> T.Compose:
    return T.Compose([
        T.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
        T.RandomCrop(IMAGE_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])


def get_val_transforms() -> T.Compose:
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])


class FashionDataset(Dataset):
    def __init__(self, csv_path: Path = LABELED_CSV, transform=None):
        self.transform = transform or get_val_transforms()
        self.samples   = self._load_csv(csv_path)

    def _load_csv(self, path: Path) -> list[dict]:
        if not path.exists():
            raise FileNotFoundError(
                f"Labeled CSV not found at {path}. Run: python data/label_prep.py"
            )
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        valid   = [r for r in rows if r.get("local_image") and Path(r["local_image"]).exists()]
        skipped = len(rows) - len(valid)
        if skipped:
            log.warning(f"Skipped {skipped} rows — local image missing.")
        log.info(f"FashionDataset: {len(valid)} samples from {path}")
        return valid

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        try:
            img = Image.open(row["local_image"]).convert("RGB")
            return self.transform(img), int(row["gender_label"]), int(row["sleeve_label"])
        except (UnidentifiedImageError, OSError) as e:
            log.warning(f"Corrupt image {row['local_image']}: {e}")
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), int(row["gender_label"]), int(row["sleeve_label"])


def get_dataloaders(
    csv_path:    Path  = LABELED_CSV,
    batch_size:  int   = BATCH_SIZE,
    val_split:   float = VAL_SPLIT,
    num_workers: int   = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Two separate Dataset instances — one per split — so train and val
    each get their own transforms with no cross-contamination.
    Indices are shuffled with a fixed seed for reproducibility.
    """
    train_ds = FashionDataset(csv_path, transform=get_train_transforms())
    val_ds   = FashionDataset(csv_path, transform=get_val_transforms())
    test_ds  = FashionDataset(csv_path, transform=get_val_transforms())

    n_total = len(train_ds)

    n_train = int(0.7 * n_total)
    n_val   = int(0.2 * n_total)
    n_test  = n_total - n_train - n_val

    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(42)).tolist()

    train_indices = indices[:n_train]
    val_indices   = indices[n_train:n_train + n_val]
    test_indices  = indices[n_train + n_val:]

    train_loader = DataLoader(
        Subset(train_ds, train_indices),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(val_ds, val_indices),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        Subset(test_ds, test_indices),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    log.info(f"DataLoaders ready — train: {n_train}, val: {n_val}, test: {n_test}")
    return train_loader, val_loader, test_loader