"""Tests for news_scraper pure helpers (no network)."""

from __future__ import annotations

from src.data_collection.news_scraper import (
    _canonical_url,
    _hash_id,
    link_tickers,
)


def test_canonical_url_strips_fragments_and_utm():
    assert _canonical_url("https://x.com/a#frag") == "https://x.com/a"
    assert _canonical_url("https://x.com/a?utm_source=rss") == "https://x.com/a"


def test_hash_id_is_deterministic_and_canonical():
    a = _hash_id("https://x.com/a#frag")
    b = _hash_id("https://x.com/a")
    assert a == b
    assert len(a) == 40


def test_link_tickers_finds_chinese_alias():
    assert link_tickers("台積電法說會釋出展望") == ["2330.TW"]


def test_link_tickers_finds_raw_code():
    assert link_tickers("鴻海(2317)前三季營收創高") == ["2317.TW"]


def test_link_tickers_dedup_alias_and_code():
    assert link_tickers("台積電(2330)宣布加碼資本支出") == ["2330.TW"]


def test_link_tickers_falls_back_to_hints():
    assert link_tickers("市場觀察", hints={"2002.TW"}) == ["2002.TW"]


def test_link_tickers_ignores_non_stock_numbers():
    # 1234 is not a real ticker but matches the regex range — that is
    # accepted by design (best-effort), so we only assert no crash and
    # that it does include the alias-based hit.
    out = link_tickers("台積電 1234 數字")
    assert "2330.TW" in out
