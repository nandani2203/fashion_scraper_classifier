self.all_products = []
self.global_seen_urls = set()

from .snapdeal_scraper import SnapdealScraper, Product, extract_gender, extract_sleeve