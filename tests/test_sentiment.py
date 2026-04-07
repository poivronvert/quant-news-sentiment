"""Tests for src.features.sentiment.

These deliberately avoid hitting the live vLLM endpoint and downloading
HuggingFace weights — both backends are exercised in the notebook.
Here we use a FakeScorer that returns deterministic results based on
keywords in the headline.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.features.sentiment import (
    LABEL_TO_SCORE,
    ScoreResult,
    Scorer,
    SCORE_COLUMNS,
    _parse_gemma_response,
    aggregate_daily,
    make_scorer,
    merge_with_existing,
    score_dataframe,
)


# ---------------------------------------------------------------------------
# Fake backend
# ---------------------------------------------------------------------------


class FakeScorer(Scorer):
    name = "fake"

    def score(self, title: str, summary: str = "") -> ScoreResult:
        text = f"{title} {summary}"
        if "好" in text or "創新高" in text:
            label = "very_positive"
        elif "壞" in text or "踩雷" in text:
            label = "very_negative"
        elif "穩" in text:
            label = "neutral"
        else:
            label = "positive"
        return ScoreResult(label=label, score=LABEL_TO_SCORE[label], confidence=0.9)


# ---------------------------------------------------------------------------
# Gemma response parser
# ---------------------------------------------------------------------------


def test_parse_gemma_plain_json():
    r = _parse_gemma_response('{"label": "positive", "confidence": 0.8}')
    assert r.label == "positive"
    assert r.score == 0.5
    assert r.confidence == pytest.approx(0.8)


def test_parse_gemma_strips_markdown_fence():
    raw = '```json\n{"label": "very_negative", "confidence": 0.95}\n```'
    r = _parse_gemma_response(raw)
    assert r.label == "very_negative"
    assert r.score == -1.0


def test_parse_gemma_unknown_label_falls_back_to_neutral():
    r = _parse_gemma_response('{"label": "wildly bullish", "confidence": 0.7}')
    assert r.label == "neutral"
    assert r.score == 0.0


def test_parse_gemma_invalid_json_falls_back():
    r = _parse_gemma_response("not json at all")
    assert r.label == "neutral"
    assert r.score == 0.0
    assert r.confidence == 0.0


# ---------------------------------------------------------------------------
# score_dataframe
# ---------------------------------------------------------------------------


def _news_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "article_id": "a1",
                "title": "台積電創新高",
                "summary": "好消息",
                "tickers": "2330.TW",
                "published_at": datetime(2024, 6, 3, 5, 30, tzinfo=timezone.utc),
            },
            {
                "article_id": "a2",
                "title": "金控海外踩雷",
                "summary": "",
                "tickers": "2882.TW",
                "published_at": datetime(2024, 6, 3, 6, 0, tzinfo=timezone.utc),
            },
            {
                "article_id": "a3",
                "title": "穩健經營",
                "summary": "",
                "tickers": "2330.TW,2882.TW",
                "published_at": datetime(2024, 6, 3, 7, 0, tzinfo=timezone.utc),
            },
        ]
    )


def test_score_dataframe_appends_columns_in_order():
    df = score_dataframe(_news_frame(), FakeScorer())
    for col in SCORE_COLUMNS:
        assert col in df.columns
    assert df["scorer"].unique().tolist() == ["fake"]


def test_score_dataframe_assigns_expected_scores():
    df = score_dataframe(_news_frame(), FakeScorer())
    by_id = df.set_index("article_id")
    assert by_id.loc["a1", "label"] == "very_positive"
    assert by_id.loc["a1", "score"] == 1.0
    assert by_id.loc["a2", "label"] == "very_negative"
    assert by_id.loc["a2", "score"] == -1.0
    assert by_id.loc["a3", "label"] == "neutral"
    assert by_id.loc["a3", "score"] == 0.0


def test_score_dataframe_empty_input():
    empty = pd.DataFrame(columns=["title", "summary", "article_id", "tickers", "published_at"])
    out = score_dataframe(empty, FakeScorer())
    assert out.empty
    for col in SCORE_COLUMNS:
        assert col in out.columns


# ---------------------------------------------------------------------------
# aggregate_daily
# ---------------------------------------------------------------------------


def test_aggregate_daily_explodes_multi_ticker_articles():
    scored = score_dataframe(_news_frame(), FakeScorer())
    daily = aggregate_daily(scored)
    # 2024-06-03 in UTC is still 2024-06-03 in Asia/Taipei since we
    # used early UTC times. 2330.TW gets a1 (+1) and a3 (0) → mean 0.5.
    # 2882.TW gets a2 (-1) and a3 (0) → mean -0.5.
    by_ticker = daily.set_index("ticker")
    assert by_ticker.loc["2330.TW", "n_articles"] == 2
    assert by_ticker.loc["2330.TW", "mean_score"] == pytest.approx(0.5)
    assert by_ticker.loc["2882.TW", "n_articles"] == 2
    assert by_ticker.loc["2882.TW", "mean_score"] == pytest.approx(-0.5)


def test_aggregate_daily_drops_untagged_articles():
    df = _news_frame()
    df.loc[len(df)] = {
        "article_id": "a4",
        "title": "macro story",
        "summary": "",
        "tickers": "",
        "published_at": datetime(2024, 6, 3, 8, 0, tzinfo=timezone.utc),
    }
    scored = score_dataframe(df, FakeScorer())
    daily = aggregate_daily(scored)
    # No row should reference an empty ticker.
    assert "" not in daily["ticker"].tolist()


def test_aggregate_daily_empty_input():
    out = aggregate_daily(pd.DataFrame())
    assert list(out.columns) == ["date", "ticker", "mean_score", "mean_conf", "n_articles"]
    assert out.empty


# ---------------------------------------------------------------------------
# Idempotent merge / CLI factory
# ---------------------------------------------------------------------------


def test_merge_with_existing_dedupes_by_article_id(tmp_path):
    first = score_dataframe(_news_frame(), FakeScorer())
    out = tmp_path / "scored.parquet"
    first.to_parquet(out, index=False)

    # Re-score the same input. The merged frame must not double up.
    second = score_dataframe(_news_frame(), FakeScorer())
    merged = merge_with_existing(second, out)
    assert len(merged) == len(first)
    assert merged["article_id"].is_unique


def test_make_scorer_rejects_unknown_backend():
    with pytest.raises(ValueError):
        make_scorer("magic")
