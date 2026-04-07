"""Technical indicators on the long-format OHLCV panel.

Thin wrapper around ``pandas_ta`` that operates on the per-ticker price
frame produced by :mod:`src.data_collection.price_loader`. The output is
a long frame keyed on ``(date, ticker)`` carrying every indicator the
strategy needs in Phase 5.

Indicator menu (matches README §3 Feature Engineering):

==============  =================================================
column          definition
==============  =================================================
ma_5            Simple moving average, 5-day close
ma_20           Simple moving average, 20-day close
ma_60           Simple moving average, 60-day close
rsi_14          Relative Strength Index, 14-day
macd            MACD line (12, 26)
macd_signal     Signal line (9-day EMA of MACD)
macd_hist       MACD histogram
bb_lower        Bollinger lower band (20, 2σ)
bb_middle       Bollinger middle band (20-day SMA)
bb_upper        Bollinger upper band (20, 2σ)
bb_pct          Bollinger %B (price location inside the band)
hv_20           20-day annualized historical volatility (log-ret)
==============  =================================================

CLI::

    uv run python -m src.features.technical \\
        --input  data/raw/prices/tw50_seed_2020_2024.parquet \\
        --output data/processed/technical.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "prices" / "tw50_seed_2020_2024.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "technical.parquet"

REQUIRED_PRICE_COLS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]

FEATURE_COLUMNS: list[str] = [
    "ma_5", "ma_20", "ma_60",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_lower", "bb_middle", "bb_upper", "bb_pct",
    "hv_20",
]

# Trading days per year, used to annualize the rolling volatility.
TRADING_DAYS = 252


def compute_indicators(group: pd.DataFrame) -> pd.DataFrame:
    """Compute the full feature set for a single ticker.

    The input must be sorted by date ascending. All indicators use
    ``adj_close`` so that splits and dividends do not introduce
    artificial jumps in moving averages or returns.
    """
    g = group.sort_values("date").copy()
    close = g["adj_close"]

    g["ma_5"] = ta.sma(close, length=5)
    g["ma_20"] = ta.sma(close, length=20)
    g["ma_60"] = ta.sma(close, length=60)

    g["rsi_14"] = ta.rsi(close, length=14)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        g["macd"] = macd["MACD_12_26_9"].to_numpy()
        g["macd_signal"] = macd["MACDs_12_26_9"].to_numpy()
        g["macd_hist"] = macd["MACDh_12_26_9"].to_numpy()
    else:
        g["macd"] = g["macd_signal"] = g["macd_hist"] = np.nan

    bb = ta.bbands(close, length=20, std=2)
    if bb is not None and not bb.empty:
        # pandas_ta column naming includes the std twice (BBL_20_2.0_2.0).
        bb_cols = bb.columns
        lower = next(c for c in bb_cols if c.startswith("BBL_"))
        middle = next(c for c in bb_cols if c.startswith("BBM_"))
        upper = next(c for c in bb_cols if c.startswith("BBU_"))
        pct_candidates = [c for c in bb_cols if c.startswith("BBP_")]
        g["bb_lower"] = bb[lower].to_numpy()
        g["bb_middle"] = bb[middle].to_numpy()
        g["bb_upper"] = bb[upper].to_numpy()
        g["bb_pct"] = bb[pct_candidates[0]].to_numpy() if pct_candidates else np.nan
    else:
        for col in ("bb_lower", "bb_middle", "bb_upper", "bb_pct"):
            g[col] = np.nan

    # Annualized 20-day historical volatility from log returns.
    log_ret = np.log(close / close.shift(1))
    g["hv_20"] = log_ret.rolling(window=20, min_periods=20).std() * np.sqrt(TRADING_DAYS)

    return g


def add_technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`compute_indicators` to every ticker in a long-format frame.

    Returns the same rows as the input, sorted by ``(ticker, date)``,
    with the :data:`FEATURE_COLUMNS` appended.
    """
    missing = [c for c in REQUIRED_PRICE_COLS if c not in prices.columns]
    if missing:
        raise ValueError(f"prices frame missing columns: {missing}")
    if prices.empty:
        out = prices.copy()
        for c in FEATURE_COLUMNS:
            out[c] = pd.Series(dtype="float64")
        return out

    pieces = [
        compute_indicators(g)
        for _, g in prices.groupby("ticker", sort=False)
    ]
    return (
        pd.concat(pieces, ignore_index=True)
          .sort_values(["ticker", "date"])
          .reset_index(drop=True)
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute technical indicators on a price panel.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    prices = pd.read_parquet(args.input)
    logger.info("loaded %d rows × %d tickers from %s",
                len(prices), prices["ticker"].nunique(), args.input)

    feats = add_technical_features(prices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(args.output, index=False)
    logger.info("wrote %d rows to %s", len(feats), args.output)

    # Quick non-NaN coverage report so the user can sanity-check warmup losses.
    coverage = (feats[FEATURE_COLUMNS].notna().mean() * 100).round(1)
    print("OK: feature coverage (% non-NaN)")
    print(coverage.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
