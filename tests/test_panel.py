"""Tests for src.features.panel.

These tests are the look-ahead bias guards. If any of them ever fails
because of a "convenient" refactor, the change is almost certainly
introducing leakage and must be reverted.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.features.panel import (
    JOINED_SENTIMENT_COLUMNS,
    _join_sentiment_asof,
    _shift_features_one_day,
    build_panel,
    build_targets,
)
from src.features.technical import FEATURE_COLUMNS as TECHNICAL_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Synthetic price + sentiment fixtures
# ---------------------------------------------------------------------------


def _toy_prices(n: int = 80, ticker: str = "TST.TW", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    rets = rng.normal(loc=0.0008, scale=0.011, size=n)
    close = 100.0 * np.cumprod(1 + rets)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": close * 0.998,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1_000, 10_000, size=n),
        }
    )


def _two_ticker_prices() -> pd.DataFrame:
    return pd.concat(
        [_toy_prices(ticker="AAA.TW", seed=1), _toy_prices(ticker="BBB.TW", seed=2)],
        ignore_index=True,
    )


def _toy_scored_news() -> pd.DataFrame:
    """Mimics the news_scraper + sentiment scoring schema."""
    rows = []
    for i, day in enumerate(pd.date_range("2024-01-02", periods=20, freq="D")):
        rows.append(
            {
                "article_id": f"a{i}",
                "title": f"news {i}",
                "summary": "",
                "tickers": "AAA.TW",
                "published_at": pd.Timestamp(day, tz="UTC"),
                "score": 0.5 if i % 2 == 0 else -0.5,
                "confidence": 0.9,
                "label": "positive" if i % 2 == 0 else "negative",
                "scorer": "fake",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# build_targets
# ---------------------------------------------------------------------------


def test_build_targets_uses_next_day_close():
    df = _toy_prices(n=5)
    out = build_targets(df).sort_values("date").reset_index(drop=True)
    expected = out["adj_close"].shift(-1) / out["adj_close"] - 1.0
    np.testing.assert_allclose(out["ret_1d"].to_numpy(), expected.to_numpy(), equal_nan=True)
    # Last row has no next day -> NaN target.
    assert np.isnan(out["ret_1d"].iloc[-1])
    assert np.isnan(out["target_up"].iloc[-1])


def test_build_targets_per_ticker_isolation():
    df = _two_ticker_prices()
    out = build_targets(df)
    # The "last row" of AAA must NOT have looked at BBB's first row for its target.
    last_aaa = out[out["ticker"] == "AAA.TW"].iloc[-1]
    assert np.isnan(last_aaa["ret_1d"])


# ---------------------------------------------------------------------------
# Feature shift contract — the look-ahead guard
# ---------------------------------------------------------------------------


def test_shift_features_one_day_contract():
    df = _two_ticker_prices()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["fake_feature"] = np.arange(len(df), dtype="float64")
    shifted = _shift_features_one_day(df, ["fake_feature"])
    # First row of each ticker becomes NaN; subsequent rows equal the prior row's value.
    for tkr in ["AAA.TW", "BBB.TW"]:
        sub_in = df[df["ticker"] == tkr]["fake_feature"].to_numpy()
        sub_out = shifted[shifted["ticker"] == tkr]["fake_feature"].to_numpy()
        assert np.isnan(sub_out[0])
        np.testing.assert_array_equal(sub_out[1:], sub_in[:-1])


def test_panel_no_leakage_features_match_previous_day():
    """The single most important test in the project.

    For an arbitrary trading day t, every feature value on the panel
    row must equal the value computed from prices known at t-1, never
    from t itself.
    """
    prices = _toy_prices(n=80)
    panel = build_panel(prices)

    panel = panel.sort_values("date").reset_index(drop=True)
    # Compute MA5 directly on the unshifted prices.
    raw_ma5 = (
        prices.sort_values("date")["adj_close"]
              .rolling(window=5, min_periods=5)
              .mean()
              .reset_index(drop=True)
    )
    # The panel's ma_5 on row t must equal raw_ma5 on row t-1.
    for t in range(1, len(panel)):
        if pd.notna(panel.loc[t, "ma_5"]) and pd.notna(raw_ma5.iloc[t - 1]):
            assert panel.loc[t, "ma_5"] == pytest.approx(raw_ma5.iloc[t - 1])


# ---------------------------------------------------------------------------
# Sentiment as-of join — t-1 strict alignment
# ---------------------------------------------------------------------------


def test_sentiment_join_strict_t_minus_1():
    prices = _toy_prices(n=30, ticker="AAA.TW")
    news = _toy_scored_news()
    panel = build_panel(prices, news)

    # Pick a trading day that has a sentiment row exactly the day before.
    # Sentiment exists for 2024-01-02..2024-01-21 (calendar days). The
    # row for 2024-01-08 (Mon) should pull sentiment dated 2024-01-07.
    # The row for 2024-01-02 (Tue) should NOT pull sentiment dated
    # 2024-01-02 — strict t-1 means same-day is excluded.
    same_day_row = panel[panel["date"] == pd.Timestamp("2024-01-02").date()].iloc[0]
    assert pd.isna(same_day_row["mean_score"])

    later_row = panel[panel["date"] == pd.Timestamp("2024-01-08").date()].iloc[0]
    assert pd.notna(later_row["mean_score"])


def test_sentiment_join_per_ticker_isolation():
    prices = _two_ticker_prices()
    # Only AAA has news.
    news = _toy_scored_news()
    panel = build_panel(prices, news)

    bbb = panel[panel["ticker"] == "BBB.TW"]
    # BBB had no news in the fixture; mean_score must be entirely NaN.
    assert bbb["mean_score"].isna().all()
    # AAA should have at least some non-null sentiment by the second week.
    aaa = panel[panel["ticker"] == "AAA.TW"]
    assert aaa["mean_score"].notna().any()


def test_join_sentiment_asof_handles_missing_sentiment_frame():
    prices = _toy_prices(n=10)
    panel = build_panel(prices, sentiment_scored=None)
    for col in JOINED_SENTIMENT_COLUMNS:
        assert col in panel.columns
        assert panel[col].isna().all()


def test_join_sentiment_asof_empty_dataframe_branch():
    """Direct unit test of the helper with an empty sentiment frame."""
    prices = _toy_prices(n=5)
    panel_only = build_panel(prices, sentiment_scored=None)
    out = _join_sentiment_asof(panel_only, pd.DataFrame())
    for col in JOINED_SENTIMENT_COLUMNS:
        assert col in out.columns
        assert out[col].isna().all()


# ---------------------------------------------------------------------------
# Schema and edge cases
# ---------------------------------------------------------------------------


def test_panel_schema_contains_all_expected_columns():
    prices = _toy_prices(n=80)
    panel = build_panel(prices, _toy_scored_news())
    for col in TECHNICAL_FEATURE_COLUMNS:
        assert col in panel.columns
    for col in JOINED_SENTIMENT_COLUMNS:
        assert col in panel.columns
    assert "ret_1d" in panel.columns
    assert "target_up" in panel.columns


def test_build_panel_rejects_empty_prices():
    with pytest.raises(ValueError, match="empty"):
        build_panel(pd.DataFrame(columns=["date", "ticker", "adj_close"]))
