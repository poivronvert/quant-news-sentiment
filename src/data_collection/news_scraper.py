"""Financial news collector for the Taiwan market.

Pulls headlines from a small set of public RSS feeds and persists them
in a normalized long-format schema. Per-ticker linking is best-effort
and lives in `link_tickers` — Phase 3 will refine this with a proper
alias map.

Schema (one row per article):
    article_id : sha1 of canonical URL
    source     : feed shortname (e.g. "cnyes_tw_stock")
    title      : headline
    summary    : short description / first paragraph (may be empty)
    url        : canonical link
    published_at : timezone-aware UTC timestamp
    tickers    : comma-joined list of detected tickers (may be empty)

Use as a library (`fetch_all_feeds(...)`) or CLI:
    uv run python -m src.data_collection.news_scraper --out data/raw/news/
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import feedparser
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw" / "news"


@dataclass(frozen=True)
class FeedSource:
    name: str
    url: str
    # Optional default ticker hints — applied if no ticker is detected in
    # the title/summary (e.g. a TSMC-only feed would set hints={"2330.TW"}).
    hints: frozenset[str] = field(default_factory=frozenset)


# A deliberately small starter set. Add more in PRs once the schema is stable.
DEFAULT_FEEDS: tuple[FeedSource, ...] = (
    FeedSource("cnyes_headline", "https://news.cnyes.com/rss/v1/news/category/headline"),
    FeedSource("cnyes_tw_stock", "https://news.cnyes.com/rss/v1/news/category/tw_stock"),
    FeedSource("cnyes_tw_stock_news", "https://news.cnyes.com/rss/v1/news/category/tw_stock_news"),
)

# Minimal seed alias map: Chinese company name → TWSE code (with .TW suffix).
# Phase 3 will replace this with a generated map sourced from TWSE.
TICKER_ALIASES: dict[str, str] = {
    "台積電": "2330.TW",
    "TSMC": "2330.TW",
    "鴻海": "2317.TW",
    "聯發科": "2454.TW",
    "台達電": "2308.TW",
    "中華電": "2412.TW",
    "中華電信": "2412.TW",
    "國泰金": "2882.TW",
    "富邦金": "2881.TW",
    "中信金": "2891.TW",
    "聯電": "2303.TW",
    "南亞": "1303.TW",
    "台塑": "1301.TW",
    "中鋼": "2002.TW",
    "日月光": "3711.TW",
    "兆豐金": "2886.TW",
    "玉山金": "2884.TW",
}

# Match raw `2330` / `2330.TW` / `(2330)` patterns directly.
RAW_CODE_RE = re.compile(r"\b(\d{4})(?:\.TW)?\b")


def _canonical_url(url: str) -> str:
    return url.split("#", 1)[0].split("?utm_", 1)[0].strip()


def _hash_id(url: str) -> str:
    return hashlib.sha1(_canonical_url(url).encode("utf-8")).hexdigest()


def _parse_published(entry: feedparser.FeedParserDict) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        ts = entry.get(key)
        if ts:
            try:
                return datetime(*ts[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
    return None


def link_tickers(text: str, *, hints: Iterable[str] = ()) -> list[str]:
    """Best-effort ticker extraction from a headline+summary blob."""
    found: set[str] = set()
    for alias, code in TICKER_ALIASES.items():
        if alias in text:
            found.add(code)
    for match in RAW_CODE_RE.findall(text):
        # Only treat 4-digit codes that look like TWSE listed stocks (1xxx-9xxx).
        if "1000" <= match <= "9999":
            found.add(f"{match}.TW")
    if not found:
        found.update(hints)
    return sorted(found)


def fetch_feed(source: FeedSource) -> list[dict]:
    logger.info("Fetching %s", source.name)
    parsed = feedparser.parse(source.url)
    if parsed.bozo:
        logger.warning("Feed %s reported bozo: %s", source.name, parsed.bozo_exception)
    rows: list[dict] = []
    for entry in parsed.entries:
        url = entry.get("link") or ""
        if not url:
            continue
        title = (entry.get("title") or "").strip()
        summary = (entry.get("summary") or entry.get("description") or "").strip()
        published = _parse_published(entry)
        tickers = link_tickers(f"{title} {summary}", hints=source.hints)
        rows.append(
            {
                "article_id": _hash_id(url),
                "source": source.name,
                "title": title,
                "summary": summary,
                "url": _canonical_url(url),
                "published_at": published,
                "tickers": ",".join(tickers),
            }
        )
    logger.info("  %d entries from %s", len(rows), source.name)
    return rows


def fetch_all_feeds(feeds: Iterable[FeedSource] = DEFAULT_FEEDS) -> pd.DataFrame:
    all_rows: list[dict] = []
    for src in feeds:
        try:
            all_rows.extend(fetch_feed(src))
        except Exception as exc:  # noqa: BLE001 — best-effort RSS poll
            logger.error("Failed to fetch %s: %s", src.name, exc)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["article_id"]).reset_index(drop=True)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.sort_values("published_at", ascending=False, na_position="last")
    return df.reset_index(drop=True)


def write_snapshot(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"news_{stamp}.parquet"
    df.to_parquet(path, index=False)
    return path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch financial news RSS feeds.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--no-write", action="store_true", help="Skip writing the snapshot")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    df = fetch_all_feeds()
    if df.empty:
        logger.error("No articles fetched")
        return 1
    if not args.no_write:
        path = write_snapshot(df, args.out)
        logger.info("Wrote %d articles to %s", len(df), path)
    print(f"OK: {len(df):,} articles, "
          f"{(df['tickers'] != '').sum()} with tickers, "
          f"sources: {sorted(df['source'].unique().tolist())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
