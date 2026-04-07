"""Tests for src.features.sentiment.

These deliberately avoid hitting the live vLLM endpoint and downloading
HuggingFace weights — both backends are exercised in the notebook.
Here we use a FakeScorer that returns deterministic results based on
keywords in the headline.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.features.sentiment import (
    LABEL_TO_SCORE,
    SCORE_COLUMNS,
    SENTIMENT_FEATURE_COLUMNS,
    ScoreResult,
    Scorer,
    _parse_gemma_response,
    add_rolling_sentiment_features,
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


# ---------------------------------------------------------------------------
# Rolling sentiment features
# ---------------------------------------------------------------------------


def _sparse_daily_frame() -> pd.DataFrame:
    """Two tickers with deliberately sparse coverage to exercise reindexing."""
    return pd.DataFrame(
        [
            # 2330.TW: news on day 1, day 3, day 5 — gaps are intentional
            {"date": pd.Timestamp("2024-06-03").date(), "ticker": "2330.TW",
             "mean_score": 1.0, "mean_conf": 0.9, "n_articles": 2},
            {"date": pd.Timestamp("2024-06-05").date(), "ticker": "2330.TW",
             "mean_score": -0.5, "mean_conf": 0.8, "n_articles": 1},
            {"date": pd.Timestamp("2024-06-07").date(), "ticker": "2330.TW",
             "mean_score": 0.5, "mean_conf": 0.85, "n_articles": 3},
            # 2882.TW: contiguous days
            {"date": pd.Timestamp("2024-06-03").date(), "ticker": "2882.TW",
             "mean_score": 0.0, "mean_conf": 0.7, "n_articles": 1},
            {"date": pd.Timestamp("2024-06-04").date(), "ticker": "2882.TW",
             "mean_score": 0.5, "mean_conf": 0.75, "n_articles": 2},
        ]
    )


def test_rolling_features_appends_columns():
    out = add_rolling_sentiment_features(_sparse_daily_frame())
    for col in SENTIMENT_FEATURE_COLUMNS:
        assert col in out.columns


def test_rolling_features_reindex_to_dense_calendar():
    out = add_rolling_sentiment_features(_sparse_daily_frame())
    # 2330.TW had news on 06-03, 06-05, 06-07. Dense calendar should
    # contain 06-03 .. 06-07 = 5 rows for that ticker.
    sub = out[out["ticker"] == "2330.TW"]
    assert len(sub) == 5
    # Filled days should have n_articles = 0 and mean_score NaN.
    no_news_days = sub[sub["date"].isin([pd.Timestamp("2024-06-04").date(),
                                          pd.Timestamp("2024-06-06").date()])]
    assert (no_news_days["n_articles"] == 0).all()
    assert no_news_days["mean_score"].isna().all()


def test_rolling_news_count_uses_summed_window():
    out = add_rolling_sentiment_features(_sparse_daily_frame())
    sub = out[out["ticker"] == "2330.TW"].sort_values("date").reset_index(drop=True)
    # 7-day rolling sum at the last row should equal 2 + 0 + 1 + 0 + 3 = 6
    # because the window covers all 5 dense rows (less than 7 days span).
    assert sub["news_count_7"].iloc[-1] == 6
    assert sub["log_news_count_7"].iloc[-1] == pytest.approx(np.log1p(6))


def test_rolling_ma_skips_nan_days():
    out = add_rolling_sentiment_features(_sparse_daily_frame())
    sub = out[out["ticker"] == "2330.TW"].sort_values("date").reset_index(drop=True)
    # 3-day rolling on 06-05 row covers 06-03 (1.0), 06-04 (NaN), 06-05 (-0.5)
    # Pandas .rolling().mean() skips NaN, so the value should be (1.0 + -0.5)/2 = 0.25
    row_06_05 = sub[sub["date"] == pd.Timestamp("2024-06-05").date()].iloc[0]
    assert row_06_05["sent_ma_3"] == pytest.approx(0.25)


def test_rolling_features_respect_per_ticker_isolation():
    out = add_rolling_sentiment_features(_sparse_daily_frame())
    # 2882.TW only spans 06-03 .. 06-04. The ma_7 on 06-04 should be (0.0+0.5)/2.
    sub = out[out["ticker"] == "2882.TW"].sort_values("date").reset_index(drop=True)
    assert sub.iloc[-1]["sent_ma_7"] == pytest.approx(0.25)
    # And it must not include any 2330 data leakage.
    assert len(sub) == 2


def test_rolling_features_empty_input():
    empty = pd.DataFrame(columns=["date", "ticker", "mean_score", "mean_conf", "n_articles"])
    out = add_rolling_sentiment_features(empty)
    assert out.empty
    for col in SENTIMENT_FEATURE_COLUMNS:
        assert col in out.columns


def test_rolling_features_missing_columns_raises():
    bad = pd.DataFrame({"date": [], "ticker": []})
    with pytest.raises(ValueError, match="missing columns"):
        add_rolling_sentiment_features(bad)
