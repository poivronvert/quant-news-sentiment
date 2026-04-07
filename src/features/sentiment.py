"""Sentiment scoring for financial news.

Production-grade extraction of the prototype in
`notebooks/02_sentiment_analysis.ipynb`. Two interchangeable backends:

- ``GemmaScorer`` — calls a self-hosted vLLM (or any OpenAI-compatible)
  endpoint with a 5-class JSON prompt. Suitable for the headline LLM
  feature track. Uses ``LLM_MODEL`` / ``OPENAI_BASE_URL`` /
  ``OPENAI_API_KEY`` from the environment.
- ``HFScorer`` — runs a HuggingFace ``sentiment-analysis`` pipeline
  locally on CPU. Default model
  (``tabularisai/multilingual-sentiment-analysis``) is general-purpose
  and intentionally used as a non-LLM baseline / cross-check.

Both backends share the ``[-1, -0.5, 0, +0.5, +1]`` score scale so the
downstream feature engineering does not need to know which one ran.

CLI::

    uv run python -m src.features.sentiment \\
        --input  data/raw/news/news_*.parquet \\
        --output data/sentiment/scored.parquet \\
        --backend gemma   # or: hf

Re-running with the same ``--output`` is idempotent: any
``article_id`` already present in the output frame is skipped, so a
crashed long run can simply be restarted.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from textwrap import dedent
from typing import Iterable

import numpy as np
import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "sentiment" / "scored.parquet"

# Shared 5-class scale. Both Gemma and the HF baseline normalize their
# raw labels (lowercased, underscored) into one of these keys before
# the score lookup happens.
LABEL_TO_SCORE: dict[str, float] = {
    "very_negative": -1.0,
    "negative": -0.5,
    "neutral": 0.0,
    "positive": 0.5,
    "very_positive": 1.0,
}


@dataclass(frozen=True)
class ScoreResult:
    label: str        # one of the LABEL_TO_SCORE keys
    score: float      # in [-1, 1]
    confidence: float  # in [0, 1]; backend-specific semantics


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class Scorer(ABC):
    """Stateless single-text sentiment scorer."""

    name: str = "scorer"

    @abstractmethod
    def score(self, title: str, summary: str = "") -> ScoreResult:
        ...


_GEMMA_SYSTEM_PROMPT = dedent("""\
    You are a financial news sentiment classifier for the Taiwan stock market.
    Given a single headline (and optional summary), judge its likely short-term
    impact on the mentioned company's stock price.

    Reply with a single JSON object and nothing else:
      {"label": <one of: very_negative, negative, neutral, positive, very_positive>,
       "confidence": <float between 0 and 1>}

    Rules:
    - Use "neutral" for macro/political headlines with no clear company-level read.
    - Earnings beats, capacity expansion, large orders, upgrades -> positive/very_positive.
    - Earnings misses, layoffs, regulatory probes, downgrades -> negative/very_negative.
    - Do NOT explain. Do NOT add markdown fences. Just the JSON object.
""")


class GemmaScorer(Scorer):
    """OpenAI-compatible LLM scorer (vLLM-hosted Gemma by default).

    Reads endpoint config from the environment so swapping models means
    only changing ``.env``. Never hardcodes openai.com.
    """

    name = "gemma"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 40,
        max_retries: int = 3,
        backoff_seconds: float = 1.5,
    ) -> None:
        # Lazy import keeps tests that use FakeScorer free of the openai dep tree.
        from openai import OpenAI  # noqa: PLC0415

        self.model = model or os.environ["LLM_MODEL"]
        self.client = OpenAI(
            base_url=base_url or os.environ["OPENAI_BASE_URL"],
            api_key=api_key or os.environ["OPENAI_API_KEY"],
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def score(self, title: str, summary: str = "") -> ScoreResult:
        user_msg = title if not summary else f"標題：{title}\n摘要：{summary[:300]}"
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _GEMMA_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return _parse_gemma_response(resp.choices[0].message.content or "")
            except Exception as exc:  # noqa: BLE001 — vLLM can raise many things
                last_err = exc
                wait = self.backoff_seconds * (2 ** attempt)
                logger.warning(
                    "Gemma call failed (attempt %d/%d): %s; sleeping %.1fs",
                    attempt + 1, self.max_retries, exc, wait,
                )
                time.sleep(wait)
        logger.error("Gemma scoring failed after %d retries: %s", self.max_retries, last_err)
        return ScoreResult(label="neutral", score=0.0, confidence=0.0)


def _parse_gemma_response(raw: str) -> ScoreResult:
    """Parse Gemma's JSON reply, tolerating ```json fences and stray prose."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        obj = json.loads(text)
        label = str(obj.get("label", "neutral")).strip().lower()
        confidence = float(obj.get("confidence", 0.0))
    except (json.JSONDecodeError, TypeError, ValueError):
        label, confidence = "neutral", 0.0
    if label not in LABEL_TO_SCORE:
        label = "neutral"
    return ScoreResult(label=label, score=LABEL_TO_SCORE[label], confidence=confidence)


class HFScorer(Scorer):
    """Local HuggingFace pipeline scorer (CPU by default)."""

    name = "hf"

    def __init__(
        self,
        model_name: str = "tabularisai/multilingual-sentiment-analysis",
        *,
        device: int = -1,
        max_length: int = 256,
    ) -> None:
        from transformers import pipeline  # noqa: PLC0415 — heavy import, lazy

        self.model_name = model_name
        self.pipe = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=device,
            truncation=True,
            max_length=max_length,
        )

    def score(self, title: str, summary: str = "") -> ScoreResult:
        text = title if not summary else f"{title}。{summary[:300]}"
        out = self.pipe(text)[0]
        label = str(out["label"]).strip().lower().replace(" ", "_")
        if label not in LABEL_TO_SCORE:
            label = "neutral"
        return ScoreResult(
            label=label,
            score=LABEL_TO_SCORE[label],
            confidence=float(out.get("score", 0.0)),
        )


def make_scorer(backend: str) -> Scorer:
    """Factory for CLI use."""
    if backend == "gemma":
        return GemmaScorer()
    if backend == "hf":
        return HFScorer()
    raise ValueError(f"unknown backend: {backend!r}")


# ---------------------------------------------------------------------------
# DataFrame-level pipeline
# ---------------------------------------------------------------------------


SCORE_COLUMNS = ["label", "score", "confidence", "scorer"]


def score_dataframe(
    df: pd.DataFrame,
    scorer: Scorer,
    *,
    log_every: int = 25,
) -> pd.DataFrame:
    """Score every row in ``df`` and return ``df`` plus the four score columns.

    The input is expected to follow the news_scraper schema: at minimum
    ``title`` and ``summary`` columns must exist. The function does not
    mutate ``df``.
    """
    if df.empty:
        empty = df.copy()
        for col in SCORE_COLUMNS:
            empty[col] = pd.Series(dtype="object" if col in ("label", "scorer") else "float64")
        return empty

    records: list[dict] = []
    n = len(df)
    start = time.monotonic()
    for i, row in enumerate(df.itertuples(index=False), start=1):
        title = getattr(row, "title", "") or ""
        summary = getattr(row, "summary", "") or ""
        result = scorer.score(title, summary)
        records.append({
            "label": result.label,
            "score": result.score,
            "confidence": result.confidence,
            "scorer": scorer.name,
        })
        if i % log_every == 0 or i == n:
            elapsed = time.monotonic() - start
            rate = i / elapsed if elapsed > 0 else float("inf")
            logger.info("scored %d/%d (%.1f rows/s)", i, n, rate)

    out = pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(records)],
        axis=1,
    )
    return out


def aggregate_daily(
    scored: pd.DataFrame,
    *,
    tz: str = "Asia/Taipei",
) -> pd.DataFrame:
    """Per-ticker, per-day aggregation of an exploded scored frame.

    Input columns required: ``published_at``, ``tickers`` (comma-joined),
    ``score``, ``confidence``. Articles with no linked ticker are dropped.
    """
    if scored.empty:
        return pd.DataFrame(
            columns=["date", "ticker", "mean_score", "mean_conf", "n_articles"]
        )

    df = scored.copy()
    df = df[df["tickers"].fillna("") != ""]
    df["ticker"] = df["tickers"].str.split(",")
    df = df.explode("ticker")
    df["ticker"] = df["ticker"].str.strip()
    df = df[df["ticker"] != ""]

    published = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["date"] = published.dt.tz_convert(tz).dt.date

    daily = (
        df.groupby(["date", "ticker"], dropna=True)
          .agg(
              mean_score=("score", "mean"),
              mean_conf=("confidence", "mean"),
              n_articles=("score", "size"),
          )
          .reset_index()
          .sort_values(["date", "ticker"])
          .reset_index(drop=True)
    )
    return daily


SENTIMENT_FEATURE_COLUMNS: list[str] = [
    "sent_ma_3",
    "sent_ma_7",
    "sent_change_1",
    "news_count_7",
    "log_news_count_7",
]


def add_rolling_sentiment_features(
    daily: pd.DataFrame,
    *,
    business_days_only: bool = False,
) -> pd.DataFrame:
    """Add rolling sentiment features per ticker on a contiguous calendar.

    The output of :func:`aggregate_daily` is sparse — a ticker only has
    rows on days where it actually appeared in the news. Computing
    ``.rolling(3).mean()`` directly on that sparse frame would average
    across uneven time gaps and silently misrepresent the signal. This
    function reindexes each ticker to a contiguous date range first
    (calendar days by default; pass ``business_days_only=True`` to
    align to a Mon–Fri index instead) and *then* computes:

    - ``sent_ma_3``      — 3-day moving average of ``mean_score``
    - ``sent_ma_7``      — 7-day moving average of ``mean_score``
    - ``sent_change_1``  — day-over-day diff of ``mean_score``
    - ``news_count_7``   — rolling 7-day sum of ``n_articles``
    - ``log_news_count_7`` — ``log1p(news_count_7)``

    On no-news days, ``mean_score`` is left as NaN (we genuinely don't
    know) and ``n_articles`` is filled with 0. Rolling means use
    ``min_periods=1`` so a window with one valid score still produces
    a value; downstream code can decide how to handle the NaN tail.
    """
    expected = {"date", "ticker", "mean_score", "n_articles"}
    missing = expected - set(daily.columns)
    if missing:
        raise ValueError(f"daily frame missing columns: {sorted(missing)}")

    if daily.empty:
        out = daily.copy()
        for col in SENTIMENT_FEATURE_COLUMNS:
            out[col] = pd.Series(dtype="float64")
        return out

    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])

    pieces: list[pd.DataFrame] = []
    freq = "B" if business_days_only else "D"
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date")
        full_index = pd.date_range(g["date"].min(), g["date"].max(), freq=freq)
        dense = (
            g.set_index("date")
             .reindex(full_index)
             .rename_axis("date")
             .reset_index()
        )
        dense["ticker"] = ticker
        # n_articles: missing day means zero coverage.
        dense["n_articles"] = dense["n_articles"].fillna(0).astype(int)
        # mean_score / mean_conf stay NaN — they will be skipped by .mean().
        dense["sent_ma_3"] = dense["mean_score"].rolling(window=3, min_periods=1).mean()
        dense["sent_ma_7"] = dense["mean_score"].rolling(window=7, min_periods=1).mean()
        dense["sent_change_1"] = dense["mean_score"].diff()
        dense["news_count_7"] = dense["n_articles"].rolling(window=7, min_periods=1).sum()
        dense["log_news_count_7"] = np.log1p(dense["news_count_7"])
        pieces.append(dense)

    enriched = pd.concat(pieces, ignore_index=True)
    enriched = enriched[
        ["date", "ticker", "mean_score", "mean_conf", "n_articles", *SENTIMENT_FEATURE_COLUMNS]
    ]
    enriched["date"] = enriched["date"].dt.date
    return enriched.sort_values(["ticker", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# I/O helpers and CLI
# ---------------------------------------------------------------------------


def load_news(input_paths: Iterable[str | Path]) -> pd.DataFrame:
    """Read one or more news Parquet files (glob patterns OK) and concatenate."""
    files: list[Path] = []
    for raw in input_paths:
        matches = [Path(p) for p in glob(str(raw))]
        if not matches:
            raise FileNotFoundError(f"no files matched: {raw}")
        files.extend(matches)
    frames = [pd.read_parquet(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["article_id"]).reset_index(drop=True)
    return df


def merge_with_existing(
    new_scored: pd.DataFrame,
    existing_path: Path,
) -> pd.DataFrame:
    """Append new rows to an existing Parquet output, deduping by article_id."""
    if not existing_path.exists():
        return new_scored
    prior = pd.read_parquet(existing_path)
    combined = pd.concat([prior, new_scored], ignore_index=True)
    combined = combined.drop_duplicates(subset=["article_id"], keep="last")
    return combined.reset_index(drop=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score financial news with a sentiment backend.")
    p.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more Parquet paths or glob patterns from news_scraper.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination Parquet for the scored frame (default: %(default)s).",
    )
    p.add_argument(
        "--backend",
        choices=["gemma", "hf"],
        default="gemma",
        help="Which scorer to use.",
    )
    p.add_argument("--limit", type=int, default=None, help="Score only the first N rows (debug).")
    p.add_argument(
        "--daily",
        type=Path,
        default=None,
        help="If set, also write the per-ticker daily aggregation to this path.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(REPO_ROOT / ".env")
    args = _build_parser().parse_args(argv)

    news = load_news(args.input)
    logger.info("loaded %d unique articles from %s", len(news), args.input)

    # Skip articles already present in the output (idempotent resume).
    if args.output.exists():
        prior = pd.read_parquet(args.output)
        already = set(prior["article_id"].tolist())
        before = len(news)
        news = news[~news["article_id"].isin(already)].reset_index(drop=True)
        logger.info("skipping %d already-scored articles", before - len(news))

    if args.limit is not None:
        news = news.head(args.limit).reset_index(drop=True)
        logger.info("limit applied: scoring %d rows", len(news))

    if news.empty:
        logger.info("nothing to score")
        return 0

    scorer = make_scorer(args.backend)
    scored = score_dataframe(news, scorer)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged = merge_with_existing(scored, args.output)
    merged.to_parquet(args.output, index=False)
    logger.info("wrote %d scored rows to %s", len(merged), args.output)

    if args.daily is not None:
        daily = aggregate_daily(merged)
        args.daily.parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(args.daily, index=False)
        logger.info("wrote %d daily rows to %s", len(daily), args.daily)

    print(
        f"OK: scored={len(scored)} new, total={len(merged)}, "
        f"backend={scorer.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
