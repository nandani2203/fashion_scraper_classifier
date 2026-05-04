# Fashion Attribute Classifier

An end-to-end ML application that scrapes fashion product images from Snapdeal, trains an EfficientNet-B0 classifier to predict **gender** (male/female) and **sleeve type** (full/half sleeve), and serves predictions via a Streamlit UI with full history tracking in SQLite.

---

## Pre-trained Checkpoint & Data

To skip scraping and training, download the files from Google Drive:
**[Google Drive →](https://drive.google.com/drive/folders/1y9MJn9kc5-nngfabPPKPwt2R5u_wTOvx)**

| File | Place at |
|------|----------|
| `efficientnet_b0_v1.0_epoch11_acc0.718.pth` | `model/checkpoints/` |
| `labeled_products.csv` | `data/labeled/` |
| `raw_products.csv` | `data/raw/` |

With these files you can skip steps 1–4 and go straight to `streamlit run app/streamlit_app.py`.

---

## Setup

### Requirements
- Python 3.10+
- ~2 GB disk space (images + model checkpoint)

### Install

```bash
git clone <repo-url>
cd fashion_scraper_and_classifier

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

---

## How to Run

Run these steps in order:

### 1. Scrape products
```bash
python scraper/snapdeal_scraper.py --pages 5
```
Downloads product images and metadata from Snapdeal into `data/raw/raw_products.csv`.

### 2. Prepare labels
```bash
python data/label_prep.py
```
Encodes gender and sleeve labels as integers, outputs `data/labeled/labeled_products.csv`.

### 3. (Optional) Augment minority class
```bash
python data/augment.py
```
Generates flipped/rotated copies of full_sleeve images to balance the dataset. Run before training if sleeve class imbalance is high.

### 4. Train the model
```bash
python model/train.py --epochs 20 --batch 32 --lr 1e-4 --freeze 5 --device cpu
```
Trains EfficientNet-B0 with two-phase fine-tuning. Best checkpoint saved to `model/checkpoints/`.

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 20 | Total training epochs |
| `--batch` | 32 | Batch size |
| `--lr` | 1e-4 | Learning rate (phase 1) |
| `--freeze` | 5 | Epochs to keep backbone frozen |
| `--device` | auto | `cpu`, `cuda`, `mps`, or `auto` |

### 5. Launch the UI
```bash
streamlit run app/streamlit_app.py
```
Opens at `http://localhost:8501`. Navigate using the sidebar.

### 6. Predict via CLI (optional)
```bash
# Single image
python inference/predict.py --single <image-url>

# Batch
python inference/predict.py --batch <url1> <url2> <url3>
```

### 7. Run tests
```bash
pytest tests/
```

---

## Project Structure

```
fashion_scraper_and_classifier/
├── scraper/          # Snapdeal scraper
├── data/             # label_prep, augment, dataset loader, images/
├── model/            # EfficientNet-B0, training loop, checkpoints/
├── inference/        # Predictor class, CLI
├── database/         # SQLAlchemy schema, query API
├── app/              # Streamlit UI (multi-page)
│   └── pages/        # Products, Train, Predict, History
├── tests/            # Unit + integration tests
└── config.py         # Single source of truth for all paths/hyperparams
```

---

## Approach

### Data Collection
Scraped clothing product images from **Snapdeal** using `requests` + `BeautifulSoup`. Used Snapdeal's built-in sleeve-length filter URLs (e.g. `?q=SleevesLength_s%3AFull%20Sleeves`) to get pre-filtered categories, which gave reliable labels without any manual annotation. Labels are extracted from both the URL filter hint and product title keywords, with the URL hint used as a fallback.

### Labeling Strategy
- **Gender**: keyword matching on product titles (`"men"`, `"women"`, etc.) with query-hint fallback
- **Sleeve**: keyword matching (`"full sleeve"`, `"half sleeve"`, `"t-shirt"` → half, etc.) with URL filter as ground truth fallback
- Zero unknown labels in the final dataset — reliable enough for training

### Model
**EfficientNet-B0** pretrained on ImageNet, with two independent classification heads appended to the pooled features:
- `gender_head` → 2 classes (male, female)
- `sleeve_head` → 2 classes (full_sleeve, half_sleeve)

Chose EfficientNet-B0 because:
- State-of-the-art accuracy/efficiency trade-off for image classification
- Pretrained ImageNet weights transfer well to fashion images
- Compact enough to train on CPU in reasonable time

### Training
Two-phase fine-tuning strategy:
1. **Phase 1** (epochs 1–5): Backbone frozen, only heads trained at full LR
2. **Phase 2** (epochs 6–20): Last 3 backbone blocks unfrozen, fine-tuned at 0.1× LR

Loss functions:
- Gender: `CrossEntropyLoss` with weight `[1.3, 1.0]` (boosts male class recall)
- Sleeve: `CrossEntropyLoss` with weight `[1.5, 1.0]` (boosts full_sleeve recall)
- Scheduler: CosineAnnealingLR

Class imbalance (full_sleeve underrepresented) addressed by:
- Weighted loss during training
- Offline augmentation (`data/augment.py`) generating flipped/rotated copies of full_sleeve images

### Database
All predictions stored in SQLite via SQLAlchemy ORM. Each prediction record includes image URL, run type (single/batch), UUID run_id, predicted labels, confidence scores, model version, timestamp, and status. Batch runs share a common `run_id` for group inspection.

---

## Limitations

- **Small dataset**: ~400–555 images is too few for a robust vision model. Performance is limited (~62–65% combined accuracy).
- **Snapdeal data ceiling**: Snapdeal returns ~20 unique products per filtered category page with heavy cross-page duplication, making large-scale scraping difficult from this source alone.
- **CPU training**: Training on CPU is slow (~1–2 min/epoch with fine-tuning). No GPU acceleration used.
- **Label quality**: Labels are auto-generated from URL filters and title keywords — no human verification. Edge cases (e.g. men's kurtas that look like women's tops) may be mislabeled.
- **Gender confusion on ethnic wear**: Kurtas appear in both male and female categories with similar visual styles, making gender classification harder for those categories.

---

## Improvements with More Time

- **More data**: Scrape from multiple sources (Flipkart, AJIO) or use a public fashion dataset (DeepFashion, iMaterialist) to reach 2000+ samples per class
- **GPU training**: Would allow larger batch sizes, more epochs, and larger model variants (EfficientNet-B3/B5)
- **Better augmentation**: MixUp, CutOut, or AutoAugment policies instead of basic transforms
- **Human-verified labels**: Manual review of 10–15% of labels to catch mislabeled ethnic wear
- **Confidence calibration**: Temperature scaling to make softmax confidence scores more meaningful
- **REST API**: Wrap inference in a FastAPI endpoint for production use instead of Streamlit-only access
