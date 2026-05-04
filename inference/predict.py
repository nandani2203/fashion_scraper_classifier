"""
inference/predict.py

Predictor class — loads a trained FashionClassifier checkpoint
and runs single or batch predictions on image URLs or local paths.
"""

import uuid
import logging
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

from config import GENDER_CLASSES, SLEEVE_CLASSES, MODEL_NAME, MODEL_VERSION, CHECKPOINTS_DIR
from data.dataset import get_val_transforms
from model.model import FashionClassifier

log = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    image_url:         str
    run_type:          str    # "single" | "batch"
    run_id:            str    # UUID — shared across all items in a batch
    predicted_gender:  str    # "male" | "female"
    predicted_sleeve:  str    # "full_sleeve" | "half_sleeve"
    confidence_gender: float
    confidence_sleeve: float
    model_name:        str = MODEL_NAME
    model_version:     str = MODEL_VERSION
    timestamp:         str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status:            str = "success"
    error_message:     str = ""

    def as_dict(self) -> dict:
        return {
            "image_url":         self.image_url,
            "run_type":          self.run_type,
            "run_id":            self.run_id,
            "predicted_gender":  self.predicted_gender,
            "predicted_sleeve":  self.predicted_sleeve,
            "confidence_gender": round(self.confidence_gender, 4),
            "confidence_sleeve": round(self.confidence_sleeve, 4),
            "model_name":        self.model_name,
            "model_version":     self.model_version,
            "timestamp":         self.timestamp,
            "status":            self.status,
            "error_message":     self.error_message,
        }


# ── Image loading ─────────────────────────────────────────────────────────

def _load_image(source: str) -> Image.Image:
    """Load from URL or local path. Raises on failure — caller handles the error."""
    if source.startswith("http://") or source.startswith("https://"):
        import requests
        resp = requests.get(source, timeout=15)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(source).convert("RGB")


# ── Predictor ─────────────────────────────────────────────────────────────

class Predictor:
    """
    Wraps a loaded FashionClassifier.
    Exposes predict_single() and predict_batch().
    """

    def __init__(self, checkpoint_path: Path = None, device: str = None):
        self.device = torch.device(
            device if device else
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )

        if checkpoint_path is None:
            checkpoint_path = self._find_best_checkpoint()

        # read metadata and weights in one pass
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model_name    = ckpt.get("model_name",    MODEL_NAME)
        self.model_version = ckpt.get("model_version", MODEL_VERSION)

        self.model = FashionClassifier(pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = get_val_transforms()
        log.info(f"Predictor ready — {self.model_name} {self.model_version} on {self.device}")

    @staticmethod
    def _find_best_checkpoint() -> Path:
        """Pick checkpoint with highest acc value in filename."""
        ckpts = list(CHECKPOINTS_DIR.glob("*.pth"))
        if not ckpts:
            raise FileNotFoundError(
                f"No checkpoints in {CHECKPOINTS_DIR}. "
                f"Run training first: python model/train.py"
            )
        def acc_from_name(p: Path) -> float:
            try:
                return float(p.stem.split("_acc")[-1])
            except (ValueError, IndexError):
                return 0.0
        return max(ckpts, key=acc_from_name)

    def _predict_one(self, source: str, run_type: str, run_id: str) -> PredictionResult:
        """Core inference for a single image. Used by both public methods."""
        try:
            image  = _load_image(source)
            tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                g_logits, s_logits = self.model(tensor)

            g_probs = F.softmax(g_logits, dim=1)[0]
            s_probs = F.softmax(s_logits, dim=1)[0]
            g_idx   = g_probs.argmax().item()
            s_idx   = s_probs.argmax().item()

            return PredictionResult(
                image_url         = source,
                run_type          = run_type,
                run_id            = run_id,
                predicted_gender  = GENDER_CLASSES[g_idx],
                predicted_sleeve  = SLEEVE_CLASSES[s_idx],
                confidence_gender = g_probs[g_idx].item(),
                confidence_sleeve = s_probs[s_idx].item(),
                model_name        = self.model_name,
                model_version     = self.model_version,
            )
        except (UnidentifiedImageError, OSError, Exception) as e:
            log.warning(f"Prediction failed for {source}: {e}")
            return PredictionResult(
                image_url         = source,
                run_type          = run_type,
                run_id            = run_id,
                predicted_gender  = "unknown",
                predicted_sleeve  = "unknown",
                confidence_gender = 0.0,
                confidence_sleeve = 0.0,
                status            = "error",
                error_message     = str(e),
            )

    def predict_single(self, image_source: str) -> PredictionResult:
        result = self._predict_one(image_source, run_type="single", run_id=str(uuid.uuid4()))
        log.info(
            f"[single] gender={result.predicted_gender} ({result.confidence_gender:.2%})  "
            f"sleeve={result.predicted_sleeve} ({result.confidence_sleeve:.2%})"
        )
        return result

    def predict_batch(self, image_sources: list[str]) -> list[PredictionResult]:
        if not image_sources:
            return []
        batch_id = str(uuid.uuid4())
        results  = [self._predict_one(src, "batch", batch_id) for src in image_sources]
        success  = sum(1 for r in results if r.status == "success")
        log.info(f"[batch] {success}/{len(results)} succeeded  batch_id={batch_id[:8]}…")
        return results


if __name__ == "__main__":
    import argparse, json, logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Run fashion attribute predictions")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--single", type=str)
    group.add_argument("--batch",  type=str, nargs="+")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device",     type=str, default=None)
    args = parser.parse_args()

    predictor = Predictor(
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else None,
        device          = args.device,
    )
    results = (
        [predictor.predict_single(args.single)] if args.single
        else predictor.predict_batch(args.batch)
    )
    for r in results:
        print(json.dumps(r.as_dict(), indent=2))