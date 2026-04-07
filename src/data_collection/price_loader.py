"""TWSE OHLCV loader backed by yfinance.

Extracted from notebooks/01_data_exploration.ipynb. Use as a library
(`load_prices(...)`) or as a CLI (`uv run python -m src.data_collection.price_loader`).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw" / "prices"

# TW50 seed list (subset). Phase 7 will replace with the historical
# constituent panel for survivorship-bias handling.
TW50_SEED: list[str] = [
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2412.TW",
    "2882.TW", "2881.TW", "2891.TW", "2303.TW", "1303.TW",
    "1301.TW", "2002.TW", "3711.TW", "2886.TW", "2884.TW",
]

LONG_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


def download_ohlcv(
    tickers: list[str],
    start: str,
    end: str,
    *,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """Download OHLCV from yfinance and return a long-format DataFrame.

    Columns: date, ticker, open, high, low, close, adj_close, volume.
    Rows where adj_close is NaN are dropped (non-trading days for that symbol).
    """
    if not tickers:
        raise ValueError("tickers must be non-empty")

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise RuntimeError(f"yfinance returned empty frame for {tickers}")

    long = (
        raw.stack(level=0, future_stack=True)
           .rename_axis(index=["date", "ticker"])
           .reset_index()
    )
    long.columns = [c.lower().replace(" ", "_") for c in long.columns]
    long = long.dropna(subset=["adj_close"])
    long = long.sort_values(["ticker", "date"]).reset_index(drop=True)

    missing = [c for c in LONG_COLUMNS if c not in long.columns]
    if missing:
        raise RuntimeError(f"yfinance response missing expected columns: {missing}")
    return long[LONG_COLUMNS]


def add_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Add ret_1d and log_ret columns, computed per ticker."""
    out = prices.copy()
    out["ret_1d"] = out.groupby("ticker")["adj_close"].pct_change()
    out["log_ret"] = np.log1p(out["ret_1d"])
    return out


def load_prices(
    tickers: list[str] | None = None,
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    *,
    out_dir: Path | None = None,
    filename: str = "tw50_seed_2020_2024.parquet",
    write: bool = True,
) -> pd.DataFrame:
    """High-level entrypoint: download, add returns, optionally persist Parquet."""
    tickers = tickers or TW50_SEED
    out_dir = out_dir or DEFAULT_OUT_DIR

    logger.info("Downloading %d tickers from %s to %s", len(tickers), start, end)
    df = add_returns(download_ohlcv(tickers, start, end))

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        df.to_parquet(path, index=False)
        logger.info("Wrote %d rows to %s", len(df), path)
    return df


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch TWSE OHLCV via yfinance.")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--out", type=Path, default=None, help="Output directory")
    p.add_argument("--filename", default="tw50_seed_2020_2024.parquet")
    p.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Override seed list (e.g. 2330.TW 2317.TW)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    df = load_prices(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        out_dir=args.out,
        filename=args.filename,
    )
    print(f"OK: {len(df):,} rows × {df['ticker'].nunique()} tickers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
