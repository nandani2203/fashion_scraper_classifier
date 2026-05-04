from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
RAW_CSV         = DATA_DIR / "raw" / "raw_products.csv"
LABELED_CSV     = DATA_DIR / "labeled" / "labeled_products.csv"
IMAGES_DIR      = DATA_DIR / "images"
CHECKPOINTS_DIR = BASE_DIR / "model" / "checkpoints"
DB_PATH         = BASE_DIR / "database" / "fashion.db"

# ── Scraper ────────────────────────────────────────────────────────────────
SCRAPE_DELAY_MIN = 1.5   # seconds between requests (min)
SCRAPE_DELAY_MAX = 3.5   # seconds between requests (max)
MAX_PAGES        = 5     # pages per query — 5 × 20 × 8 queries = 800 raw products

# Search-based queries, stable across site restructures.
# Labels come directly from the query itself, no URL filter params to maintain.
# To change site: update SNAPDEAL_SEARCH_URL only.
# To add a category: add one tuple here.
SEARCH_QUERIES = [

    # MEN SHIRTS 
    ("men full sleeve shirt", "male", "full_sleeve"),
    ("men half sleeve shirt", "male", "half_sleeve"),

    # MEN T-SHIRTS 
    ("men full sleeve tshirt", "male", "full_sleeve"),
    ("men half sleeve tshirt", "male", "half_sleeve"),

    # WOMEN T-SHIRTS 
    (
        "https://www.snapdeal.com/products/women-apparel-tees?sort=plrty&q=SleevesLength_s%3AFull%20Sleeves%7C",
        "female",
        "full_sleeve"
    ),
    (
        "https://www.snapdeal.com/products/women-apparel-tees?sort=plrty&q=SleevesLength_s%3AHalf%20Sleeves%7C",
        "female",
        "half_sleeve"
    ),

    # WOMEN SHIRTS
    (
        "https://www.snapdeal.com/products/women-apparel-shirts-blouses?sort=plrty&q=SleevesLength_s%3AFull%20Sleeves%7C",
        "female",
        "full_sleeve"
    ),
    (
        "https://www.snapdeal.com/products/women-apparel-shirts-blouses?sort=plrty&q=SleevesLength_s%3AHalf%20Sleeves%7C",
        "female",
        "half_sleeve"
    ),

    # WOMEN TOPS & TUNICS 
    (
        "https://www.snapdeal.com/products/women-apparel-tops-tunics?sort=plrty&q=SleevesLength_s%3AFull%20Sleeves%7C",
        "female",
        "full_sleeve"
    ),
    (
        "https://www.snapdeal.com/products/women-apparel-tops-tunics?sort=plrty&q=SleevesLength_s%3AHalf%20Sleeves%7C",
        "female",
        "half_sleeve"
    ),
    # MEN KURTAS
    (
        "https://www.snapdeal.com/products/men-ethnic-wear-kurtas?sort=plrty&q=SleevesLength_s%3AFull%20Sleeves%7C",
        "male",
        "full_sleeve"
    ),
    (
        "https://www.snapdeal.com/products/men-ethnic-wear-kurtas?sort=plrty&q=SleevesLength_s%3AHalf%20Sleeves%7C",
        "male",
        "half_sleeve"
    ),

    # MEN SPORTS T-SHIRTS (were discussed but never added)
    (
        "https://www.snapdeal.com/products/men-sports-tshirts-polos?sort=plrty&q=SleevesLength_s%3AFull%20Sleeves%7C",
        "male",
        "full_sleeve"
    ),
    (
        "https://www.snapdeal.com/products/men-sports-tshirts-polos?sort=plrty&q=SleevesLength_s%3AHalf%20Sleeves%7C",
        "male",
        "half_sleeve"
    ),

    # MEN OUTERWEAR & WINTER WEAR (were discussed but never added)
    (
        "https://www.snapdeal.com/products/men-apparel-outerwear-jackets?sort=plrty&q=SleevesLength_s%3AFull%20Sleeves%7C",
        "male",
        "full_sleeve"
    ),
    (
        "https://www.snapdeal.com/products/men-apparel-sweatshirts?sort=plrty&q=SleevesLength_s%3AFull%20Sleeves%7C",
        "male",
        "full_sleeve"
    ),
    (
        "https://www.snapdeal.com/products/men-apparel-sweaters?sort=plrty&q=SleevesLength_s%3AFull%20Sleeves%7C",
        "male",
        "full_sleeve"
    ),

    # 👩 WOMEN KURTI SETS (NEW)
    (
        "https://www.snapdeal.com/search?keyword=kurti%20set&catId=0&categoryId=46139427&suggested=false&vertical=&noOfResults=20&SRPID=trendingSearch&clickSrc=CatTrending&sort=rlvncy&q=SleevesLength_s%3AHalf%20Sleeves%7C",
        "female",
        "half_sleeve"
    ),
    (
        "https://www.snapdeal.com/search?keyword=kurti%20set&catId=0&categoryId=46139427&suggested=false&vertical=&noOfResults=20&SRPID=trendingSearch&clickSrc=CatTrending&sort=rlvncy&q=SleevesLength_s%3AFull%20Sleeves%7C",
        "female",
        "full_sleeve"
    ),

    # WOMEN SWEATSHIRTS (full sleeve only — half sleeve sweatshirts don't exist as a category)
    (
        "https://www.snapdeal.com/products/women-apparel-sweatshirts?sort=plrty&q=SleevesLength_s%3AFull%20Sleeves%7C",
        "female",
        "full_sleeve"
    ),
]

SNAPDEAL_SEARCH_URL = "https://www.snapdeal.com/search?keyword={query}&sort=plrty"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Referer": "https://www.snapdeal.com/",
}

# ── Image / Model ──────────────────────────────────────────────────────────
IMAGE_SIZE = 224
IMAGE_MEAN = [0.485, 0.456, 0.406]   # ImageNet mean
IMAGE_STD  = [0.229, 0.224, 0.225]   # ImageNet std

GENDER_CLASSES = ["male", "female"]
SLEEVE_CLASSES = ["full_sleeve", "half_sleeve"]

NUM_GENDER_CLASSES = len(GENDER_CLASSES)
NUM_SLEEVE_CLASSES = len(SLEEVE_CLASSES)

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
NUM_EPOCHS    = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-5
VAL_SPLIT     = 0.2
FREEZE_EPOCHS = 5
MODEL_NAME    = "efficientnet_b0"
MODEL_VERSION = "v1.0"

# ── Database ───────────────────────────────────────────────────────────────
DATABASE_URL = f"sqlite:///{DB_PATH}"