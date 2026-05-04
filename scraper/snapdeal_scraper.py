"""
scraper/snapdeal_scraper.py

Scrapes clothing products from Snapdeal using search queries.
Search-based approach is stable — search URLs never change across
site restructures, unlike category filter URLs.

Labels come directly from the search query itself:
    "men full sleeve shirt" → gender=male, sleeve=full_sleeve
Title-based extraction is used as a secondary validator.

Confirmed working selectors (verified May 2025):
    Product card : div.product-tuple-listing  (20 per page)
    Title        : p.product-title
    Image        : img.product-image  (src populated directly, no lazy loading)
    Pagination   : &page=N
"""

import csv
import time
import random
import logging
import requests
from pathlib import Path
from dataclasses import dataclass, fields, asdict

from bs4 import BeautifulSoup

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    RAW_CSV, IMAGES_DIR,
    SEARCH_QUERIES, SNAPDEAL_SEARCH_URL,
    HEADERS, SCRAPE_DELAY_MIN, SCRAPE_DELAY_MAX, MAX_PAGES,
    GENDER_CLASSES, SLEEVE_CLASSES,
)

log = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────

@dataclass
class Product:
    title:       str
    image_url:   str
    product_url: str
    gender:      str        # "male" | "female" | "unknown"
    sleeve_type: str        # "full_sleeve" | "half_sleeve" | "unknown"
    local_image: str = ""   # filled after download
    source:      str = "snapdeal"


# ── Label extraction ──────────────────────────────────────────────────────

GENDER_KEYWORDS = {
    # female checked first — "men" is a substring of "women"
    "female": ["women", "woman", "female", "girl", "ladies", "lady", "womens"],
    "male":   [" men ", " men's", "mens", " man ", "male", "boy", "gents"],
}

SLEEVE_KEYWORDS = {
    "full_sleeve": ["full sleeve", "full-sleeve", "long sleeve", "long-sleeve", "full slv"],
    "half_sleeve": ["half sleeve", "half-sleeve", "short sleeve", "short-sleeve",
                    "half slv", "t-shirt", "tshirt", "polo"],
}


def extract_gender(title: str, hint: str = "") -> str:
    """
    Infer gender from title. Female checked first — 'men' is a substring of 'women'.
    Falls back to query hint if title has no keywords.
    """
    t = f" {title.lower()} "
    for gender, keywords in GENDER_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return gender
    return hint if hint in GENDER_CLASSES else "unknown"


def extract_sleeve(title: str, hint: str = "") -> str:
    t = title.lower()

    for sleeve, keywords in SLEEVE_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return sleeve

    # 👇 IMPORTANT FIX
    # If nothing found, trust query hint (already controlled)
    return hint if hint in SLEEVE_CLASSES else "unknown"

# ── HTTP helpers ──────────────────────────────────────────────────────────

def get_page(url: str, session: requests.Session) -> BeautifulSoup | None:
    """Fetch a URL and return BeautifulSoup, or None on failure."""
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        log.warning(f"Failed to fetch {url}: {e}")
        return None


def polite_sleep():
    time.sleep(random.uniform(SCRAPE_DELAY_MIN, SCRAPE_DELAY_MAX))


# ── URL builder ───────────────────────────────────────────────────────────

def build_search_url(query: str, page: int = 1) -> str:
    """
    Build a Snapdeal search URL from a plain English query.
    Page param is appended for pages > 1.

    Examples:
        build_search_url("men full sleeve shirt")
        → https://www.snapdeal.com/search?keyword=men+full+sleeve+shirt&sort=plrty

        build_search_url("men full sleeve shirt", page=2)
        → https://www.snapdeal.com/search?keyword=men+full+sleeve+shirt&sort=plrty&page=2
    """
    encoded = query.replace(" ", "+")
    base    = SNAPDEAL_SEARCH_URL.format(query=encoded)
    return base if page == 1 else f"{base}&page={page}"


# ── Parsing ───────────────────────────────────────────────────────────────

def parse_products_from_page(
    soup: BeautifulSoup,
    gender_hint: str,
    sleeve_hint: str,
) -> list[Product]:
    """
    Parse product cards from a Snapdeal search results page.
    Query hint is used as fallback label when title extraction fails.
    Since we searched for e.g. 'men full sleeve shirt', the hint is
    already highly reliable — most products will match it.
    """
    cards = soup.select("div.product-tuple-listing")
    log.info(f"  Found {len(cards)} cards on page")
    products = []

    for card in cards:
        try:
            title_tag = card.select_one("p.product-title")
            if not title_tag:
                continue
            title = title_tag.get_text(strip=True)
            if not title:
                continue

            img_tag   = card.select_one("img.product-image")
            if not img_tag:
                continue
            image_url = (img_tag.get("src") or img_tag.get("data-src") or "").strip()
            if not image_url.startswith("http"):
                continue

            a_tag       = card.select_one("a.dp-widget-link")
            product_url = a_tag["href"] if a_tag and a_tag.get("href") else ""

            sleeve = extract_sleeve(title, hint=sleeve_hint)

            # filter BEFORE creating object
            if sleeve not in SLEEVE_CLASSES:
                continue

            products.append(Product(
                title       = title,
                image_url   = image_url,
                 product_url = product_url,
                gender      = extract_gender(title, hint=gender_hint),
                sleeve_type = extract_sleeve(title, hint=sleeve_hint),
            ))

        except Exception as e:
            log.debug(f"Skipping card: {e}")

    return products


# ── Image download ────────────────────────────────────────────────────────

def download_image(image_url: str, dest_dir: Path, filename: str) -> str:
    """Download image to dest_dir/filename. Returns local path or '' on failure."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    if dest_path.exists():
        return str(dest_path)

    try:
        resp = requests.get(image_url, headers=HEADERS, timeout=15, stream=True)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return str(dest_path)
    except Exception as e:
        log.warning(f"Image download failed ({image_url}): {e}")
        return ""


# ── Scraper class ─────────────────────────────────────────────────────────

class SnapdealScraper:
    def __init__(self, max_pages: int = MAX_PAGES, download_images: bool = True):
        self.all_products = []
        self.global_seen_urls = set()
        self.max_pages       = max_pages
        self.download_images = download_images
        self.session         = requests.Session()
        self.session.headers.update(HEADERS)
        # self.all_products: list[Product] = []

    def scrape_query(self, query: str, gender_hint: str, sleeve_hint: str) -> list[Product]:
        """Scrape multiple pages for one search query."""
        products = []

        max_pages = 1 if not query.startswith("http") else self.max_pages

        for page in range(1, self.max_pages + 1):
            if query.startswith("http"):
                url = f"{query}&page={page}"
            else:
                url = build_search_url(query, page)
            log.info(f"  Page {page}: {url}")
            soup = get_page(url, self.session)

            if soup is None:
                log.warning(f"Could not fetch page {page}, stopping.")
                break

            page_products = parse_products_from_page(soup, gender_hint, sleeve_hint)
            log.info(f"  Parsed {len(page_products)} products")

            titles = [p.title for p in page_products]
            unique_titles = len(set(titles))

            log.info(f"Page {page}: total={len(titles)}, unique={unique_titles}")
            if not page_products:
                break

            # products.extend(page_products)
            new_count = 0

            for p in page_products:
                clean_url = p.product_url.split("?")[0]
                if clean_url not in self.global_seen_urls:
                    self.global_seen_urls.add(clean_url)
                    products.append(p)
                    new_count += 1

            log.info(f"Page {page}: new={new_count}")

            if page > 1 and new_count < 5:
                log.info("Stopping early → low new data")
                break
            polite_sleep()

        return products

    def scrape_all(self) -> list[Product]:
        """Scrape all queries defined in SEARCH_QUERIES."""
        for query, gender_hint, sleeve_hint in SEARCH_QUERIES:
            log.info(f"\nQuery: '{query}'  (gender={gender_hint}, sleeve={sleeve_hint})")
            self.all_products.extend(
                self.scrape_query(query, gender_hint, sleeve_hint)
            )

        log.info(f"\nTotal scraped: {len(self.all_products)}")
        self._deduplicate()
        return self.all_products

    def _deduplicate(self):
        """Remove duplicate products by image URL."""
        seen, unique = set(), []
        for p in self.all_products:
            if p.image_url not in seen:
                seen.add(p.image_url)
                unique.append(p)
        removed = len(self.all_products) - len(unique)
        if removed:
            log.info(f"Removed {removed} duplicates. Unique: {len(unique)}")
        self.all_products = unique

    def download_all_images(self):
        """Download images for all scraped products."""
        log.info(f"Downloading {len(self.all_products)} images...")
        for i, product in enumerate(self.all_products):
            product.local_image = download_image(
                product.image_url, IMAGES_DIR, f"product_{i:04d}.jpg"
            )
            if (i + 1) % 50 == 0:
                log.info(f"  {i + 1}/{len(self.all_products)} downloaded")
            polite_sleep()

    def save_to_csv(self, path: Path = RAW_CSV):
        """Save all products to CSV."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[f.name for f in fields(Product)])
            writer.writeheader()
            writer.writerows(asdict(p) for p in self.all_products)
        log.info(f"Saved {len(self.all_products)} products → {path}")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Snapdeal fashion scraper")
    parser.add_argument("--pages",     type=int, default=MAX_PAGES)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--output",    type=str, default=str(RAW_CSV))
    args = parser.parse_args()

    scraper = SnapdealScraper(max_pages=args.pages, download_images=not args.no_images)
    scraper.scrape_all()

    if not args.no_images:
        scraper.download_all_images()

    scraper.save_to_csv(Path(args.output))

    products      = scraper.all_products
    gender_counts = {g: sum(1 for p in products if p.gender == g)
                     for g in GENDER_CLASSES + ["unknown"]}
    sleeve_counts = {s: sum(1 for p in products if p.sleeve_type == s)
                     for s in SLEEVE_CLASSES + ["unknown"]}
    print(f"\nTotal : {len(products)}")
    print(f"Gender: {gender_counts}")
    print(f"Sleeve: {sleeve_counts}")
    print(f"Output: {args.output}")