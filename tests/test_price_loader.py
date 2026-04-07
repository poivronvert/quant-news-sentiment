"""Lightweight tests for price_loader.

Network calls to yfinance are intentionally not mocked here — Phase 8
will introduce a proper test fixture. For now we test the pure helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_collection.price_loader import LONG_COLUMNS, add_returns


def _fake_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    rows = []
    for tkr, base in [("2330.TW", 100.0), ("2317.TW", 50.0)]:
        for i, d in enumerate(dates):
            # Compounding +1% per step keeps the per-step return constant.
            px = base * (1.01 ** i)
            rows.append([d, tkr, px, px, px, px, px, 1000])
    return pd.DataFrame(rows, columns=LONG_COLUMNS)


def test_add_returns_per_ticker():
    df = add_returns(_fake_panel())
    # First row of each ticker should have NaN return.
    firsts = df.groupby("ticker").head(1)
    assert firsts["ret_1d"].isna().all()
    # All non-NaN returns should be exactly 1%, computed independently per ticker.
    np.testing.assert_allclose(df["ret_1d"].dropna().to_numpy(), 0.01, atol=1e-12)
    # log_ret should equal log1p(ret_1d).
    np.testing.assert_allclose(
        df["log_ret"].dropna().to_numpy(),
        np.log1p(df["ret_1d"].dropna().to_numpy()),
    )


def test_long_columns_contract():
    df = add_returns(_fake_panel())
    for col in LONG_COLUMNS + ["ret_1d", "log_ret"]:
        assert col in df.columns
