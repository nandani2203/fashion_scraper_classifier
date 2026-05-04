"""
tests/test_core.py

Core test suite for fashion_scraper_and_classifier.
Covers the logic that actually matters — label extraction,
metrics, DB helpers, and a full pipeline integration test.

Run with:
    pytest tests/test_core.py -v
"""

import csv
import sys
import uuid
import tempfile
import logging
from pathlib import Path
from datetime import datetime, timezone

import pytest
import torch

# ── path setup ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))


# ══════════════════════════════════════════════════════════════════════════
# UNIT TESTS — pure logic, no files, no network, no DB
# ══════════════════════════════════════════════════════════════════════════

class TestLabelExtraction:
    """
    extract_gender and extract_sleeve are the foundation of dataset quality.
    If these are wrong the model trains on bad labels — highest priority to test.
    """

    from scraper.snapdeal_scraper import extract_gender, extract_sleeve

    @pytest.mark.parametrize("title, expected", [
        ("Peter England Men Full Sleeve Formal Shirt",  "male"),
        ("BIBA Women Half Sleeve Cotton Kurta",         "female"),
        ("Jockey Mens Half Sleeve T-Shirt",             "male"),
        ("Global Desi Ladies Full Sleeve Kurta",        "female"),
        ("W for Woman Short Sleeve Printed Top",        "female"),
        ("Roadster Boy Slim Fit Shirt",                 "male"),
        # 'men' must not match inside 'women'
        ("FabIndia Women Full Sleeve Kurta",            "female"),
        # unknown — no keywords
        ("Classic Cotton Comfort Wear",                 "unknown"),
    ])
    def test_extract_gender(self, title, expected):
        from scraper.snapdeal_scraper import extract_gender
        assert extract_gender(title) == expected, \
            f"extract_gender({title!r}) → got {extract_gender(title)!r}, expected {expected!r}"

    @pytest.mark.parametrize("title, expected", [
        ("Men Full Sleeve Formal Shirt",       "full_sleeve"),
        ("Women Half Sleeve Cotton Kurta",     "half_sleeve"),
        ("Long Sleeve Casual Top",             "full_sleeve"),
        ("Short Sleeve Summer Dress",          "half_sleeve"),
        ("Men Polo T-Shirt",                   "half_sleeve"),
        ("Women Tshirt Printed",               "half_sleeve"),
        # unknown — no keywords
        ("Classic Comfort Wear",               "unknown"),
    ])
    def test_extract_sleeve(self, title, expected):
        from scraper.snapdeal_scraper import extract_sleeve
        assert extract_sleeve(title) == expected, \
            f"extract_sleeve({title!r}) → got {extract_sleeve(title)!r}, expected {expected!r}"

    def test_gender_hint_fallback(self):
        """When title has no keywords, hint from category URL should be used."""
        from scraper.snapdeal_scraper import extract_gender
        assert extract_gender("Classic Comfort Wear", hint="male")   == "male"
        assert extract_gender("Classic Comfort Wear", hint="female") == "female"
        assert extract_gender("Classic Comfort Wear", hint="")       == "unknown"

    def test_sleeve_hint_fallback(self):
        from scraper.snapdeal_scraper import extract_sleeve
        assert extract_sleeve("Classic Comfort Wear", hint="full_sleeve") == "full_sleeve"
        assert extract_sleeve("Classic Comfort Wear", hint="")            == "unknown"


class TestRunningMetrics:
    """
    RunningMetrics accumulates loss and accuracy across batches.
    Wrong math here silently produces misleading training logs.
    """

    def test_correct_predictions(self):
        from model.train import RunningMetrics
        m = RunningMetrics()
        # batch: both gender and sleeve predicted correctly
        g_logits = torch.tensor([[2.0, 0.5]])   # argmax=0 → male
        s_logits = torch.tensor([[0.5, 2.0]])   # argmax=1 → half_sleeve
        m.update(0.4, g_logits, s_logits, torch.tensor([0]), torch.tensor([1]))

        assert m.gender_correct == 1
        assert m.sleeve_correct == 1
        assert m.avg_loss       == pytest.approx(0.4)
        assert m.gender_acc     == pytest.approx(1.0)
        assert m.sleeve_acc     == pytest.approx(1.0)
        assert m.combined_acc   == pytest.approx(1.0)

    def test_wrong_predictions(self):
        from model.train import RunningMetrics
        m = RunningMetrics()
        g_logits = torch.tensor([[0.5, 2.0]])   # argmax=1, label=0 → wrong
        s_logits = torch.tensor([[2.0, 0.5]])   # argmax=0, label=1 → wrong
        m.update(1.2, g_logits, s_logits, torch.tensor([0]), torch.tensor([1]))

        assert m.gender_correct == 0
        assert m.sleeve_correct == 0
        assert m.gender_acc     == pytest.approx(0.0)
        assert m.combined_acc   == pytest.approx(0.0)

    def test_accumulates_across_batches(self):
        from model.train import RunningMetrics
        m = RunningMetrics()
        # batch 1: correct
        m.update(0.3, torch.tensor([[2.0,0.5]]), torch.tensor([[0.5,2.0]]),
                 torch.tensor([0]), torch.tensor([1]))
        # batch 2: wrong
        m.update(0.7, torch.tensor([[0.5,2.0]]), torch.tensor([[2.0,0.5]]),
                 torch.tensor([0]), torch.tensor([1]))

        assert m.n_samples      == 2
        assert m.gender_correct == 1
        assert m.avg_loss       == pytest.approx(0.5)   # (0.3+0.7)/2
        assert m.gender_acc     == pytest.approx(0.5)

    def test_reset(self):
        from model.train import RunningMetrics
        m = RunningMetrics()
        m.update(1.0, torch.tensor([[2.0,0.5]]), torch.tensor([[2.0,0.5]]),
                 torch.tensor([0]), torch.tensor([0]))
        m.reset()
        assert m.n_samples  == 0
        assert m.total_loss == 0.0


class TestParseTimestamp:
    """_parse_ts must handle all input types without raising."""

    def test_none_returns_datetime(self):
        from database.db import _parse_ts
        result = _parse_ts(None)
        assert isinstance(result, datetime)

    def test_valid_iso_string(self):
        from database.db import _parse_ts
        result = _parse_ts("2024-06-01T12:00:00")
        assert isinstance(result, datetime)
        assert result.year == 2024

    def test_invalid_string_returns_datetime(self):
        from database.db import _parse_ts
        result = _parse_ts("not-a-date")
        assert isinstance(result, datetime)

    def test_datetime_passthrough(self):
        from database.db import _parse_ts
        dt     = datetime(2024, 1, 15, tzinfo=timezone.utc)
        result = _parse_ts(dt)
        assert result == dt


# ══════════════════════════════════════════════════════════════════════════
# INTEGRATION TEST — full pipeline: CSV → label_prep → dataset → predict → DB
# ══════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """
    Verifies all modules connect correctly end-to-end.
    Uses temp directories so nothing touches the real project data.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        """Create temp dirs and a small labeled CSV with real images."""
        from PIL import Image

        # ── directories ───────────────────────────────────────────────────
        raw_dir     = tmp_path / "raw"
        labeled_dir = tmp_path / "labeled"
        images_dir  = tmp_path / "images"
        ckpt_dir    = tmp_path / "checkpoints"
        for d in [raw_dir, labeled_dir, images_dir, ckpt_dir]:
            d.mkdir()

        # ── patch config paths ────────────────────────────────────────────
        import config as cfg
        monkeypatch.setattr(cfg, "RAW_CSV",         raw_dir     / "raw_products.csv")
        monkeypatch.setattr(cfg, "LABELED_CSV",     labeled_dir / "labeled_products.csv")
        monkeypatch.setattr(cfg, "IMAGES_DIR",      images_dir)
        monkeypatch.setattr(cfg, "CHECKPOINTS_DIR", ckpt_dir)
        monkeypatch.setattr(cfg, "DB_PATH",         tmp_path / "test.db")
        monkeypatch.setattr(cfg, "DATABASE_URL",    f"sqlite:///{tmp_path / 'test.db'}")

        # ── write raw CSV ─────────────────────────────────────────────────
        samples = [
            ("Men Full Sleeve Shirt",    "male",   "full_sleeve",  0, 0),
            ("Women Half Sleeve Kurta",  "female", "half_sleeve",  1, 1),
            ("Men Half Sleeve T-Shirt",  "male",   "half_sleeve",  0, 1),
            ("Women Full Sleeve Top",    "female", "full_sleeve",  1, 0),
        ] * 5   # 20 rows total for a valid train/val split

        img_paths = []
        for i, (title, gender, sleeve, g_lbl, s_lbl) in enumerate(samples):
            p = images_dir / f"product_{i:04d}.jpg"
            Image.new("RGB", (224, 224), color=(i * 10, 80, 160)).save(p)
            img_paths.append(str(p))

        fields = ["title","image_url","product_url","gender","sleeve_type","local_image","source"]
        rows   = [
            {
                "title": s[0], "image_url": "", "product_url": "",
                "gender": s[1], "sleeve_type": s[2],
                "local_image": img_paths[i], "source": "test",
            }
            for i, s in enumerate(samples)
        ]
        with open(cfg.RAW_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)

        # ── save fake checkpoint ──────────────────────────────────────────
        import model.model as mm
        orig_init = mm.FashionClassifier.__init__
        def patched(self, **kw): orig_init(self, pretrained=False)
        monkeypatch.setattr(mm.FashionClassifier, "__init__", patched)

        from model.model import FashionClassifier
        m         = FashionClassifier()
        ckpt_path = ckpt_dir / "efficientnet_b0_v1.0_epoch05_acc0.850.pth"
        torch.save({
            "epoch": 5, "val_acc": 0.85,
            "model_name": "efficientnet_b0", "model_version": "v1.0",
            "state_dict": m.state_dict(),
        }, ckpt_path)

        self.tmp_path = tmp_path
        self.cfg      = cfg

        # reset db module singleton so it uses patched config
        import database.db as db_mod
        db_mod._engine  = None
        db_mod._Session = None

    def test_label_prep_produces_labeled_csv(self):
        """label_prep reads raw CSV and writes a clean labeled CSV."""
        import importlib
        import data.label_prep as lp
        importlib.reload(lp)

        rows = lp.prepare()
        assert len(rows) == 20
        assert all("gender_label" in r for r in rows)
        assert all("sleeve_label" in r for r in rows)
        assert all(r["gender_label"] in (0, 1) for r in rows)

    def test_dataset_loads_from_labeled_csv(self):
        """FashionDataset reads labeled CSV and returns correct tensor shapes."""
        import importlib
        import data.label_prep as lp; importlib.reload(lp)
        import data.dataset    as ds; importlib.reload(ds)

        lp.prepare()
        dataset = ds.FashionDataset()
        assert len(dataset) == 20

        img, gender, sleeve = dataset[0]
        assert img.shape == (3, 224, 224)
        assert gender in (0, 1)
        assert sleeve in (0, 1)

    def test_predictor_single_and_batch(self):
        """Predictor loads checkpoint, runs single + batch, returns correct types."""
        import importlib
        import inference.predict as ip; importlib.reload(ip)

        from PIL import Image
        img_path = str(self.tmp_path / "test_img.jpg")
        Image.new("RGB", (224, 224)).save(img_path)

        predictor = ip.Predictor(device="cpu")

        # single
        result = predictor.predict_single(img_path)
        assert result.status            == "success"
        assert result.run_type          == "single"
        assert result.predicted_gender  in ("male", "female")
        assert result.predicted_sleeve  in ("full_sleeve", "half_sleeve")
        assert 0.0 <= result.confidence_gender <= 1.0
        assert len(result.run_id) == 36

        # batch
        results  = predictor.predict_batch([img_path, img_path])
        run_ids  = set(r.run_id for r in results)
        assert len(results) == 2
        assert len(run_ids) == 1        # all share one batch_id
        assert all(r.run_type == "batch" for r in results)

    def test_predict_saves_to_db_and_history_retrieves(self):
        """PredictionResult → save_prediction → get_history → correct row returned."""
        import importlib
        import database.db as db; importlib.reload(db)
        import inference.predict as ip; importlib.reload(ip)

        db.init_db()

        result = ip.PredictionResult(
            image_url         = "http://example.com/shirt.jpg",
            run_type          = "single",
            run_id            = str(uuid.uuid4()),
            predicted_gender  = "male",
            predicted_sleeve  = "full_sleeve",
            confidence_gender = 0.91,
            confidence_sleeve = 0.87,
        )
        saved   = db.save_prediction(result)
        assert saved.id is not None

        history = db.get_history(limit=10)
        assert len(history) == 1
        assert history[0]["predicted_gender"]  == "male"
        assert history[0]["predicted_sleeve"]  == "full_sleeve"
        assert history[0]["confidence_gender"] == pytest.approx(0.91, abs=1e-3)

    def test_full_pipeline_end_to_end(self):
        """
        Full chain: label_prep → FashionDataset → Predictor → DB → get_stats.
        This is the test that proves all modules connect correctly.
        """
        import importlib
        import data.label_prep  as lp; importlib.reload(lp)
        import data.dataset     as ds; importlib.reload(ds)
        import database.db      as db; importlib.reload(db)
        import inference.predict as ip; importlib.reload(ip)

        from PIL import Image

        # step 1: label prep
        rows = lp.prepare()
        assert len(rows) == 20

        # step 2: dataset loads correctly
        dataset = ds.FashionDataset()
        assert len(dataset) == 20

        # step 3: predict
        img_path  = str(self.tmp_path / "pipeline_img.jpg")
        Image.new("RGB", (224, 224), color=(100, 150, 200)).save(img_path)

        predictor = ip.Predictor(device="cpu")
        results   = predictor.predict_batch([img_path] * 3)
        assert all(r.status == "success" for r in results)

        # step 4: save to DB
        db.init_db()
        db.save_predictions(results)

        # step 5: verify stats
        stats = db.get_stats()
        assert stats["total"]       == 3
        assert stats["total_batch"] == 3
        assert sum(stats["gender_counts"].values()) == 3
        assert sum(stats["sleeve_counts"].values()) == 3