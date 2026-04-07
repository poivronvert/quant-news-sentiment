"""Assemble the unified (date, ticker) feature panel for modeling.

This is the join point where Phase 4 hands off to Phase 5. The output
is a long DataFrame keyed on ``(date, ticker)`` containing:

- raw forward-looking labels (``ret_1d``, ``target_up``)  ← only on the
  feature row whose features are AS-OF the previous day
- all technical indicators from :mod:`src.features.technical`
- all sentiment features from :mod:`src.features.sentiment`

**Look-ahead bias protection** is the entire point of this module.
Every feature column is shifted forward by one trading day before it
gets joined to the row for which it acts as a predictor. In other
words: the row labeled ``2024-06-05`` for ``2330.TW`` carries the
indicators that were computable using only information available at
the close of ``2024-06-04``. The forward target on that same row is
the ``2024-06-05 -> 2024-06-06`` return, which is what the strategy
will attempt to predict.

Sentiment alignment uses the same convention: the LAST sentiment row
on or before the previous calendar day is used. Same-day news are
intentionally excluded because they may have arrived after the open.

CLI::

    uv run python -m src.features.panel \\
        --prices    data/raw/prices/tw50_seed_2020_2024.parquet \\
        --sentiment data/sentiment/scored.parquet \\
        --output    data/processed/panel.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.sentiment import (
    SENTIMENT_FEATURE_COLUMNS,
    add_rolling_sentiment_features,
    aggregate_daily,
)
from src.features.technical import FEATURE_COLUMNS as TECHNICAL_FEATURE_COLUMNS
from src.features.technical import add_technical_features

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRICES = REPO_ROOT / "data" / "raw" / "prices" / "tw50_seed_2020_2024.parquet"
DEFAULT_SENTIMENT = REPO_ROOT / "data" / "sentiment" / "scored.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "panel.parquet"

# Sentiment columns that get joined to the price panel. ``mean_score`` is
# included so the model has access to the raw daily score in addition to
# the rolling derivatives.
JOINED_SENTIMENT_COLUMNS: list[str] = [
    "mean_score",
    "n_articles",
    *SENTIMENT_FEATURE_COLUMNS,
]


def build_targets(prices: pd.DataFrame) -> pd.DataFrame:
    """Add forward-looking ``ret_1d`` and ``target_up`` columns.

    ``ret_1d`` on row ``t`` is the close-to-close return realized on
    ``t -> t+1``. ``target_up`` is the binary direction (``1`` if
    positive, ``0`` otherwise). Both are NaN on the last row of each
    ticker because there is no next day to look up.
    """
    df = prices.sort_values(["ticker", "date"]).copy()
    next_close = df.groupby("ticker")["adj_close"].shift(-1)
    df["ret_1d"] = next_close / df["adj_close"] - 1.0
    df["target_up"] = (df["ret_1d"] > 0).astype("float64")
    df.loc[df["ret_1d"].isna(), "target_up"] = np.nan
    return df


def _shift_features_one_day(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Lag every feature column by one row per ticker.

    The frame must already be sorted ``(ticker, date)`` ascending. After
    this transformation, the feature row aligned with date ``t`` carries
    values that were observable at the close of date ``t-1``.
    """
    out = df.copy()
    out[feature_cols] = (
        out.groupby("ticker", sort=False)[feature_cols].shift(1)
    )
    return out


def _join_sentiment_asof(
    panel: pd.DataFrame,
    sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """As-of join sentiment onto a price panel using strict ``<`` matching.

    For each ``(ticker, trading_date)`` we attach the most recent
    sentiment row whose calendar date is **strictly before** the trading
    date. This guarantees no same-day leakage even if news happened to
    be published intraday.
    """
    if sentiment.empty:
        out = panel.copy()
        for col in JOINED_SENTIMENT_COLUMNS:
            out[col] = np.nan
        return out

    # Force both sides to identical [ns] datetime resolution. merge_asof
    # is strict about dtype equality and silently mixed [us]/[s]/[ns]
    # frames blow up with a confusing MergeError.
    sent = sentiment.copy()
    sent["date"] = pd.to_datetime(sent["date"]).astype("datetime64[ns]")
    sent = sent.sort_values(["ticker", "date"]).reset_index(drop=True)

    panel_sorted = panel.copy()
    panel_sorted["date"] = pd.to_datetime(panel_sorted["date"]).astype("datetime64[ns]")
    panel_sorted = panel_sorted.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Per-ticker merge_asof avoids accidentally pulling sentiment from
    # a different ticker when dates collide.
    pieces: list[pd.DataFrame] = []
    sent_groups = dict(tuple(sent.groupby("ticker", sort=False)))
    for ticker, g in panel_sorted.groupby("ticker", sort=False):
        sent_g = sent_groups.get(ticker)
        if sent_g is None or sent_g.empty:
            tmp = g.copy()
            for col in JOINED_SENTIMENT_COLUMNS:
                tmp[col] = np.nan
            pieces.append(tmp)
            continue
        # `allow_exact_matches=False` enforces strict t-1 alignment: a
        # sentiment row dated 2024-06-05 will NOT be attached to the
        # 2024-06-05 trading row, only to 2024-06-06 onward.
        merged = pd.merge_asof(
            g.sort_values("date"),
            sent_g[["date", *JOINED_SENTIMENT_COLUMNS]].sort_values("date"),
            on="date",
            direction="backward",
            allow_exact_matches=False,
        )
        pieces.append(merged)

    return pd.concat(pieces, ignore_index=True)


def build_panel(
    prices: pd.DataFrame,
    sentiment_scored: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the modeling panel from raw prices and (optional) scored news.

    Parameters
    ----------
    prices
        Long-format OHLCV from :mod:`src.data_collection.price_loader`.
    sentiment_scored
        Optional output of :mod:`src.features.sentiment`. If ``None`` or
        empty the resulting panel still contains technical features and
        targets, with the sentiment columns filled with NaN.
    """
    if prices.empty:
        raise ValueError("prices frame is empty")

    # Step 1: technical indicators on the raw price panel.
    with_tech = add_technical_features(prices)

    # Step 2: forward target.
    with_targets = build_targets(with_tech)

    # Step 3: shift every technical feature by one day so that the row
    # for date `t` carries indicators known at the close of `t-1`.
    panel = _shift_features_one_day(with_targets, TECHNICAL_FEATURE_COLUMNS)

    # Step 4: as-of join sentiment with strict t-1 alignment.
    if sentiment_scored is not None and not sentiment_scored.empty:
        daily = aggregate_daily(sentiment_scored)
        rolled = add_rolling_sentiment_features(daily)
        panel = _join_sentiment_asof(panel, rolled)
    else:
        for col in JOINED_SENTIMENT_COLUMNS:
            panel[col] = np.nan

    # Final tidy: keep date as a python date for parquet readability,
    # drop rows that have no target (last day per ticker).
    panel["date"] = pd.to_datetime(panel["date"]).dt.date
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Assemble the (date, ticker) modeling panel.")
    p.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    p.add_argument(
        "--sentiment",
        type=Path,
        default=DEFAULT_SENTIMENT,
        help="Optional scored news Parquet. Missing file is treated as no sentiment.",
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    prices = pd.read_parquet(args.prices)
    logger.info("loaded %d price rows × %d tickers", len(prices), prices["ticker"].nunique())

    sentiment_df: pd.DataFrame | None = None
    if args.sentiment.exists():
        sentiment_df = pd.read_parquet(args.sentiment)
        logger.info("loaded %d scored articles from %s", len(sentiment_df), args.sentiment)
    else:
        logger.warning("sentiment file %s missing — building tech-only panel", args.sentiment)

    panel = build_panel(prices, sentiment_df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.output, index=False)
    logger.info("wrote %d panel rows to %s", len(panel), args.output)

    coverage = (
        panel[TECHNICAL_FEATURE_COLUMNS + JOINED_SENTIMENT_COLUMNS].notna().mean() * 100
    ).round(1)
    print("OK: panel feature coverage (% non-NaN)")
    print(coverage.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
