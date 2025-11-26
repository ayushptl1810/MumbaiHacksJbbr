"""
End-to-end scraper for PTI Fact Check listings.

The script first attempts to fetch and parse the HTML with Requests + BeautifulSoup.
If the static HTML does not contain the fact-check cards (the site is an Angular
single-page app), it transparently falls back to driving a headless browser via
Selenium to obtain the rendered markup. Usage:

    python pti_fact_check_scraper.py --max-items 5 --prefer-selenium

Environment variables:
    PTI_FACT_CHECK_URL: Override the target URL if needed.
    SELENIUM_DRIVER_PATH: Optional path to a ChromeDriver binary to reuse.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from html import unescape
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

try:
    from selenium.webdriver import Chrome, ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
except ImportError:  # pragma: no cover - optional dependency
    Chrome = ChromeOptions = ChromeService = By = WebDriverWait = None

try:
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError:  # pragma: no cover - optional dependency
    ChromeDriverManager = None


DEFAULT_URL = os.environ.get("PTI_FACT_CHECK_URL", "https://www.ptinews.com/fact-check")
API_URL = os.environ.get(
    "PTI_FACT_CHECK_API", "https://api.ptivideos.com/pti/week-fact-check"
)
DETAIL_BASE_URL = os.environ.get(
    "PTI_FACT_DETAIL_BASE", "https://www.ptinews.com/fact-detail/"
)
DEFAULT_OUTPUT_PATH = os.environ.get(
    "PTI_FACT_CHECK_OUTPUT", "pti_fact_checks.json"
)
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
DATE_PATTERN = re.compile(
    r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),",
    re.IGNORECASE,
)


@dataclass
class FactCheckEntry:
    title: Optional[str]
    summary: Optional[str]
    url: str
    published_at: Optional[str]
    image_url: Optional[str]


class ScrapeError(RuntimeError):
    """Raised when neither HTTP parsing nor Selenium rendering succeeds."""


def fetch_html_via_requests(url: str, timeout: int = 20) -> str:
    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    response.raise_for_status()
    return response.text


def _collect_card_nodes(soup: BeautifulSoup) -> Iterable[BeautifulSoup]:
    primary = soup.select("div.news-content, div.fact-check-card, div[class*=newsCard]")
    if primary:
        return primary
    # Fallback: capture parent containers that hold fact-detail links.
    anchors = soup.select("a[href*='/fact-detail/']")
    nodes = []
    seen = set()
    for anchor in anchors:
        parent = anchor.find_parent(["article", "div", "li"])
        if not parent:
            continue
        key = id(parent)
        if key in seen:
            continue
        seen.add(key)
        nodes.append(parent)
    return nodes


def _extract_text(node: BeautifulSoup, selector: str) -> Optional[str]:
    target = node.select_one(selector)
    if not target:
        return None
    text = target.get_text(strip=True)
    return text or None


def _extract_date_text(node: BeautifulSoup) -> Optional[str]:
    candidates = node.select("p, span, div")
    for candidate in candidates:
        text = candidate.get_text(strip=True)
        if not text:
            continue
        if DATE_PATTERN.search(text):
            return text
    return None


def _extract_image_url(node: BeautifulSoup, base_url: str) -> Optional[str]:
    image = node.select_one("img")
    if not image:
        return None
    src = image.get("src") or image.get("data-src")
    if not src:
        return None
    return urljoin(base_url, src)


def parse_fact_checks(html: str, base_url: str) -> List[FactCheckEntry]:
    soup = BeautifulSoup(html, "html.parser")
    entries: List[FactCheckEntry] = []
    seen_urls = set()

    for node in _collect_card_nodes(soup):
        anchor = node.select_one("a[href*='/fact-detail/']")
        if not anchor:
            continue
        href = anchor.get("href")
        if not href:
            continue
        full_url = urljoin(base_url, href)
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        title = _extract_text(node, "a[href*='/fact-detail/'] h1, "
                                     "a[href*='/fact-detail/'] h2, "
                                     "a[href*='/fact-detail/'] h3, "
                                     "a[href*='/fact-detail/'] h4, "
                                     "a[href*='/fact-detail/'] h5, "
                                     "a[href*='/fact-detail/'] h6")
        if not title:
            title = anchor.get_text(strip=True) or None

        summary = _extract_text(node, "a[href*='/fact-detail/'] p, p.summary, div.summary")
        published_at = _extract_date_text(node)
        image_url = _extract_image_url(node, base_url)

        entries.append(
            FactCheckEntry(
                title=title,
                summary=summary,
                url=full_url,
                published_at=published_at,
                image_url=image_url,
            )
        )
    return entries


def _strip_html(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    fragment = BeautifulSoup(value, "html.parser")
    text = fragment.get_text(" ", strip=True)
    return unescape(text) or None


def _extract_entry_id(entry_url: str) -> Optional[int]:
    match = re.search(r"/(\d+)(?:/)?$", entry_url)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _resolve_chromedriver_service() -> ChromeService:
    driver_path = os.environ.get("SELENIUM_DRIVER_PATH")
    if driver_path:
        return ChromeService(executable_path=driver_path)
    if ChromeDriverManager is None:
        raise ImportError(
            "webdriver-manager is not installed and SELENIUM_DRIVER_PATH was not provided."
        )
    return ChromeService(executable_path=ChromeDriverManager().install())


def render_html_via_selenium(url: str, wait_seconds: int = 20) -> str:
    if Chrome is None or ChromeOptions is None or ChromeService is None or WebDriverWait is None:
        raise ImportError("selenium is required for browser rendering fallback.")

    options = ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"--user-agent={USER_AGENT}")

    service = _resolve_chromedriver_service()
    driver = Chrome(service=service, options=options)
    try:
        driver.get(url)
        WebDriverWait(driver, wait_seconds).until(
            lambda d: len(d.find_elements(By.CSS_SELECTOR, "a[href*='/fact-detail/']")) > 0
        )
        time.sleep(1.0)  # Allow lazy-loaded images and excerpts to settle.
        return driver.page_source
    finally:
        driver.quit()


def fetch_api_fact_checks(
    length: int,
    start: int = 0,
    verify: bool = True,
    timeout: int = 20,
) -> Dict[int, dict]:
    params = {"start": start, "length": length}
    response = requests.get(
        API_URL,
        params=params,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        verify=verify,
    )
    response.raise_for_status()
    payload = response.json()
    items = payload.get("week_fact_check", [])
    results: Dict[int, dict] = {}
    for item in items:
        entry_id = item.get("id")
        if isinstance(entry_id, int):
            results[entry_id] = item
    return results


def _api_record_to_entry(record: dict) -> FactCheckEntry:
    entry_id = record.get("id")
    url = f"{DETAIL_BASE_URL}{entry_id}"
    image_url = None
    images = record.get("newimage") or []
    if images and isinstance(images, list):
        first = images[0] or {}
        image_url = first.get("imgurl")
    summary = (
        record.get("story_data")
        or " ".join(record.get("textstory") or [])
        or None
    )
    return FactCheckEntry(
        title=_strip_html(record.get("headline")),
        summary=_strip_html(summary),
        url=url,
        published_at=record.get("releasetime") or record.get("date"),
        image_url=image_url,
    )


def _enrich_entries_with_api(
    entries: List[FactCheckEntry],
    max_items: Optional[int],
    api_verify: bool,
) -> List[FactCheckEntry]:
    if not entries:
        return entries

    needs_enrichment = any(
        not (entry.title and entry.summary and entry.published_at) for entry in entries
    )
    if not needs_enrichment:
        return entries

    try:
        api_data = fetch_api_fact_checks(
            length=max(len(entries), max_items or len(entries)),
            verify=api_verify,
        )
    except requests.exceptions.SSLError as exc:
        print(
            "Warning: API enrichment skipped due to TLS error. "
            "Set PTI_API_INSECURE=1 or pass --api-insecure to bypass verification.",
            file=sys.stderr,
        )
        return entries
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Warning: API enrichment failed: {exc}", file=sys.stderr)
        return entries

    for entry in entries:
        record = None
        entry_id = _extract_entry_id(entry.url)
        if entry_id and entry_id in api_data:
            record = api_data[entry_id]
        elif api_data:
            # Fallback to matching by image URL if IDs were missing.
            for candidate in api_data.values():
                images = candidate.get("newimage") or []
                if images and entry.image_url == images[0].get("imgurl"):
                    record = candidate
                    break
        if not record:
            continue
        entry.title = entry.title or _strip_html(record.get("headline"))
        summary_text = record.get("story_data") or " ".join(record.get("textstory") or [])
        entry.summary = entry.summary or _strip_html(summary_text)
        entry.published_at = entry.published_at or record.get("releasetime") or record.get(
            "date"
        )
        if not entry.image_url:
            images = record.get("newimage") or []
            if images and isinstance(images, list):
                entry.image_url = images[0].get("imgurl")
    return entries


def scrape_pti_fact_checks(
    url: str = DEFAULT_URL,
    max_items: Optional[int] = None,
    prefer_selenium: bool = False,
    api_insecure: bool = False,
) -> List[FactCheckEntry]:
    errors = []
    html = ""

    if not prefer_selenium:
        try:
            html = fetch_html_via_requests(url)
            entries = parse_fact_checks(html, url)
            if entries:
                trimmed = entries[:max_items] if max_items else entries
                return _enrich_entries_with_api(trimmed, max_items, not api_insecure)
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"Requests/BeautifulSoup attempt failed: {exc}")
        else:
            errors.append("Requests/BeautifulSoup attempt returned no fact-check cards.")

    try:
        html = render_html_via_selenium(url)
        entries = parse_fact_checks(html, url)
        if entries:
            trimmed = entries[:max_items] if max_items else entries
            return _enrich_entries_with_api(trimmed, max_items, not api_insecure)
        errors.append("Selenium rendering succeeded but produced zero cards.")
    except Exception as exc:  # pylint: disable=broad-except
        errors.append(f"Selenium fallback failed: {exc}")

    try:
        api_records = fetch_api_fact_checks(
            length=max_items or 20,
            verify=not api_insecure,
        )
        if api_records:
            entries = [
                _api_record_to_entry(record) for record in api_records.values()
            ]
            entries.sort(key=lambda entry: entry.published_at or "", reverse=True)
            return entries[:max_items] if max_items else entries
    except Exception as exc:  # pylint: disable=broad-except
        errors.append(f"API fallback failed: {exc}")

    raise ScrapeError("; ".join(errors))


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError:
        return None


def main() -> int:
    max_items = _env_int("PTI_MAX_ITEMS")
    prefer_selenium = _env_flag("PTI_PREFER_SELENIUM")
    api_insecure = _env_flag("PTI_API_INSECURE")

    try:
        entries = scrape_pti_fact_checks(
            url=DEFAULT_URL,
            max_items=max_items,
            prefer_selenium=prefer_selenium,
            api_insecure=api_insecure,
        )
    except ScrapeError as exc:
        print(f"Scrape failed: {exc}", file=sys.stderr)
        return 1

    payload = [asdict(entry) for entry in entries]
    json_dump = json.dumps(payload, indent=2, ensure_ascii=False)
    with open(DEFAULT_OUTPUT_PATH, "w", encoding="utf-8") as handle:
        handle.write(json_dump + "\n")
    print(f"Wrote {len(payload)} entries to {DEFAULT_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

