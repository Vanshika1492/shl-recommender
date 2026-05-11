"""
SHL catalog scraper — Individual Test Solutions only.
Writes data/catalog.json with fields:
  name, url, description, test_type, test_type_label,
  remote_testing, adaptive, duration, languages
"""

import json, time, re
from pathlib import Path
import requests
from bs4 import BeautifulSoup

BASE = "https://www.shl.com"
CATALOG_URL = f"{BASE}/solutions/products/product-catalog/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; SHLBot/1.0; research)"
    )
}

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgment",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "M": "Motivation",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def fetch(url: str, retries: int = 3) -> BeautifulSoup:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser")
        except Exception as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def parse_test_types(cell_text: str) -> list[str]:
    """Extract single-letter test type codes from a table cell."""
    return [ch for ch in cell_text.upper() if ch in TEST_TYPE_MAP]


def scrape_page(url: str) -> list[dict]:
    soup = fetch(url)
    rows = []

    # The catalog renders as a filterable table; rows are <tr> inside
    # div.custom-select-wrapper or inside a <table>
    # Fallback: look for any <a> whose href contains /solutions/products/
    table = soup.find("table")
    if not table:
        # Some pages use divs
        links = soup.select("a[href*='/solutions/products/']")
        for a in links:
            href = a.get("href", "")
            if "/product-catalog/" in href:
                continue
            rows.append({
                "name": a.get_text(strip=True),
                "url": BASE + href if href.startswith("/") else href,
                "description": "",
                "test_type": [],
                "remote_testing": None,
                "adaptive": None,
                "duration": None,
                "languages": [],
            })
        return rows

    headers_row = table.find("tr")
    headers = [th.get_text(strip=True).lower() for th in headers_row.find_all(["th", "td"])]

    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue

        row: dict = {
            "name": "",
            "url": "",
            "description": "",
            "test_type": [],
            "remote_testing": None,
            "adaptive": None,
            "duration": None,
            "languages": [],
        }

        for i, cell in enumerate(cells):
            hdr = headers[i] if i < len(headers) else ""
            text = cell.get_text(strip=True)
            a = cell.find("a")

            if i == 0 or "name" in hdr or "assessment" in hdr:
                row["name"] = text
                if a:
                    href = a.get("href", "")
                    row["url"] = BASE + href if href.startswith("/") else href
            elif "remote" in hdr:
                row["remote_testing"] = bool(cell.find("img") or text.lower() in ("yes", "✓", "●"))
            elif "adaptive" in hdr or "irt" in hdr:
                row["adaptive"] = bool(cell.find("img") or text.lower() in ("yes", "✓", "●"))
            elif "duration" in hdr or "time" in hdr:
                row["duration"] = text
            elif "type" in hdr or "test type" in hdr:
                row["test_type"] = parse_test_types(text)
                # Also capture the raw icons / dots
                imgs = cell.find_all("img")
                for img in imgs:
                    alt = img.get("alt", "").strip().upper()
                    if alt and alt[0] in TEST_TYPE_MAP:
                        if alt[0] not in row["test_type"]:
                            row["test_type"].append(alt[0])
            elif "language" in hdr:
                row["languages"] = [lang.strip() for lang in text.split(",") if lang.strip()]

        if row["name"]:
            rows.append(row)

    return rows


def scrape_detail(item: dict) -> dict:
    """Visit the product page and enrich description + test_type."""
    if not item["url"]:
        return item
    try:
        soup = fetch(item["url"])
        # Description: first <p> in main content
        main = soup.find("main") or soup.find("article") or soup
        paras = main.find_all("p")
        texts = [p.get_text(" ", strip=True) for p in paras if len(p.get_text(strip=True)) > 60]
        if texts:
            item["description"] = texts[0][:500]

        # Test type from meta or badges
        for el in soup.find_all(["span", "div", "td", "li"]):
            t = el.get_text(strip=True)
            if re.match(r"^[ABCDEKMPRS]$", t):
                if t not in item["test_type"]:
                    item["test_type"].append(t)

        # Remote testing / adaptive from structured facts
        page_text = soup.get_text(" ", strip=True).lower()
        if item["remote_testing"] is None:
            item["remote_testing"] = "remote" in page_text or "online" in page_text
        if item["adaptive"] is None:
            item["adaptive"] = "adaptive" in page_text or "irt" in page_text

    except Exception as exc:
        print(f"  ⚠ detail fetch failed for {item['url']}: {exc}")

    return item


def paginate_and_scrape() -> list[dict]:
    """Handle pagination on the catalog page."""
    all_items: list[dict] = []
    seen_urls: set[str] = set()

    # Try paginated URLs: ?start=0, ?start=12, ...
    page = 0
    page_size = 12
    consecutive_empty = 0

    while consecutive_empty < 2:
        url = f"{CATALOG_URL}?type=1&start={page * page_size}"
        print(f"  Fetching page {page}: {url}")
        try:
            items = scrape_page(url)
        except Exception as exc:
            print(f"  ⚠ page {page} failed: {exc}")
            break

        new = [i for i in items if i["url"] not in seen_urls and i["name"]]
        if not new:
            consecutive_empty += 1
        else:
            consecutive_empty = 0
            for i in new:
                seen_urls.add(i["url"])
            all_items.extend(new)
            print(f"    +{len(new)} items (total {len(all_items)})")

        page += 1
        time.sleep(0.5)

    return all_items


def main():
    out_path = Path("data/catalog.json")
    out_path.parent.mkdir(exist_ok=True)

    print("=== Scraping SHL catalog (Individual Test Solutions) ===")
    items = paginate_and_scrape()

    if not items:
        print("⚠ Pagination scrape found nothing. Trying single-page scrape…")
        items = scrape_page(CATALOG_URL)

    print(f"\nEnriching {len(items)} items with detail pages…")
    enriched = []
    for idx, item in enumerate(items, 1):
        print(f"  [{idx}/{len(items)}] {item['name']}")
        enriched.append(scrape_detail(item))
        time.sleep(0.3)

    out_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False))
    print(f"\n✓ Saved {len(enriched)} items → {out_path}")


if __name__ == "__main__":
    main()
