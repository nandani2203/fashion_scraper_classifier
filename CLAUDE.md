# Fashion Attribute Classifier

Assignment project: scrape fashion images from Snapdeal → train EfficientNet-B0 → classify **gender** (male/female) and **sleeve type** (full/half) → serve via Streamlit UI with prediction history in SQLite.

## Stack
- **Scraper**: requests + BeautifulSoup → Snapdeal
- **Model**: EfficientNet-B0 (torchvision), two output heads, two-phase training
- **DB**: SQLAlchemy + SQLite (`predictions` table)
- **UI**: Streamlit multi-page app

## Run Commands
```bash
# 1. Scrape
python scraper/snapdeal_scraper.py [--pages 5] [--no-images]

# 2. Label
python data/label_prep.py

# 3. Train
python model/train.py [--epochs 20] [--batch 32] [--lr 1e-4] [--freeze 5] [--device auto]

# 4. Predict (CLI)
python inference/predict.py --single <URL>
python inference/predict.py --batch <URL1> <URL2> ...

# 5. UI
streamlit run app/streamlit_app.py

# Tests
pytest tests/
```

## Key Architecture
- `config.py` — single source of truth for all paths, hyperparams, class maps
- `model/train.py` — Phase 1: freeze backbone, train heads. Phase 2: unfreeze last 3 blocks, fine-tune at 0.1×LR
- `inference/predict.py` — Predictor loads checkpoint, handles URL or local path, returns `PredictionResult` dataclass with UUID run_id
- `database/db.py` — singleton engine; `save_prediction`, `get_history`, `get_batch`, `get_stats`
- `app/state.py` — owns all `st.session_state`; lazy-loads Predictor; call `clear_predictor()` after training

## Data Flow
```
Snapdeal → raw_products.csv → labeled_products.csv → FashionDataset → train → checkpoint → Predictor → SQLite → UI
```

## Labels
- Gender: `male→0`, `female→1` (from product title keywords + query hint fallback)
- Sleeve: `full_sleeve→0`, `half_sleeve→1`
- Sleeve uses weighted loss `[1.3, 1.0]` to handle class imbalance
