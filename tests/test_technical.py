"""Tests for src.features.technical."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.technical import (
    FEATURE_COLUMNS,
    REQUIRED_PRICE_COLS,
    TRADING_DAYS,
    add_technical_features,
    compute_indicators,
)


def _synthetic_prices(ticker: str = "TEST.TW", n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Deterministic mean-reverting price series long enough for all warmups."""
    rng = np.random.default_rng(seed)
    base = 100.0
    rets = rng.normal(loc=0.0005, scale=0.012, size=n)
    close = base * np.cumprod(1 + rets)
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
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


def test_add_technical_features_appends_all_columns():
    df = _synthetic_prices()
    out = add_technical_features(df)
    for col in FEATURE_COLUMNS:
        assert col in out.columns
    # Row count is preserved.
    assert len(out) == len(df)


def test_warmup_period_yields_nan_then_populated_values():
    df = _synthetic_prices(n=120)
    out = add_technical_features(df)
    # MA60 needs 60 prior closes -> first 59 rows should be NaN.
    assert out["ma_60"].iloc[:59].isna().all()
    assert out["ma_60"].iloc[59:].notna().all()
    # 20-day annualized vol needs 20 returns; the first 20 rows are NaN.
    assert out["hv_20"].iloc[:20].isna().all()
    assert out["hv_20"].iloc[20:].notna().all()


def test_rsi_within_zero_to_hundred():
    df = _synthetic_prices(n=120)
    out = add_technical_features(df)
    rsi = out["rsi_14"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_bollinger_bands_ordering():
    df = _synthetic_prices(n=120)
    out = add_technical_features(df).dropna(subset=["bb_lower", "bb_middle", "bb_upper"])
    assert (out["bb_lower"] <= out["bb_middle"]).all()
    assert (out["bb_middle"] <= out["bb_upper"]).all()


def test_macd_hist_equals_macd_minus_signal():
    df = _synthetic_prices(n=120)
    out = add_technical_features(df).dropna(subset=["macd", "macd_signal", "macd_hist"])
    diff = out["macd"] - out["macd_signal"]
    np.testing.assert_allclose(diff.to_numpy(), out["macd_hist"].to_numpy(), atol=1e-9)


def test_hv_20_uses_log_returns_and_annualization():
    df = _synthetic_prices(n=120, seed=42)
    out = add_technical_features(df).dropna(subset=["hv_20"])
    # Recompute manually for the last row and compare.
    log_ret = np.log(df["adj_close"] / df["adj_close"].shift(1))
    expected_last = log_ret.iloc[-20:].std() * np.sqrt(TRADING_DAYS)
    assert out["hv_20"].iloc[-1] == pytest.approx(expected_last, rel=1e-9)


def test_panel_grouping_does_not_leak_between_tickers():
    a = _synthetic_prices("AAA.TW", n=120, seed=1)
    b = _synthetic_prices("BBB.TW", n=120, seed=2)
    panel = pd.concat([a, b], ignore_index=True)
    out = add_technical_features(panel)

    # Indicator on ticker A computed in isolation must equal the panel result.
    only_a = compute_indicators(a)["ma_20"].to_numpy()
    panel_a = out[out["ticker"] == "AAA.TW"]["ma_20"].to_numpy()
    np.testing.assert_allclose(only_a, panel_a, equal_nan=True)


def test_missing_required_columns_raises():
    df = pd.DataFrame({"date": [], "ticker": [], "close": []})
    with pytest.raises(ValueError, match="missing columns"):
        add_technical_features(df)


def test_empty_input_returns_empty_with_feature_columns():
    df = pd.DataFrame(columns=REQUIRED_PRICE_COLS)
    out = add_technical_features(df)
    assert out.empty
    for col in FEATURE_COLUMNS:
        assert col in out.columns
