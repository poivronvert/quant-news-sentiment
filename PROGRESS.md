# PROGRESS

專案執行進度追蹤。依照 README.md 所規劃的 pipeline 分階段推進，每階段完成後提交 commit。

---

## Execution Strategy

整體採「notebook 先行、模組化後補」策略：先在 notebooks/ 中做資料探索與原型開發，確認流程可行後再將穩定邏輯抽取到 `src/` 下的模組中，最後由 script / CLI 串接整條 pipeline。

### Phase 0 — Project Bootstrap
- [x] 閱讀 README.md，規劃執行策略
- [x] 以 `uv` 初始化 Python 專案（`pyproject.toml`, `uv.lock`）
- [x] 建立目錄骨架（data/, src/, notebooks/, results/, tests/）
- [x] 加入 `.gitignore`、`.env.example`
- [x] 撰寫 PROGRESS.md

### Phase 1 — Data Exploration (notebooks/01_data_exploration.ipynb)
- [x] 使用 yfinance 抓取台灣 50 成分股 OHLCV（2020/01/01–2024/12/31）— 目前先用 15 檔種子清單
- [x] 檢視缺失值、停牌、除權息等資料品質問題
- [x] 畫出價格 / 報酬率 / 波動率基本統計
- [x] 將原始價格資料存入 `data/raw/prices/tw50_seed_2020_2024.parquet`（18,210 rows）

### Phase 2 — Data Collection Modules
- [x] `src/data_collection/price_loader.py`：把 notebook 中驗證過的抓取邏輯模組化（library + CLI + pytest）
- [x] `src/data_collection/news_scraper.py`：cnyes RSS 三條 feed，含 ticker linker
- [x] 設計 news schema（article_id, source, title, summary, url, published_at, tickers）
- [ ] *Deferred to Phase 7：* 使用歷史 TW50 成分股清單取代 seed list
- [ ] *Deferred：* 擴充其他來源（聯合報、經濟日報等）

### Phase 3 — Sentiment Analysis
- [x] **3a** notebook prototype：Gemma 5-class JSON prompt → score in [-1,1]
- [x] **3b** cross-check baseline：`tabularisai/multilingual-sentiment-analysis`（CPU，非 finance-tuned，當對照組）
- [x] **3c** 抽出 `src/features/sentiment.py`（Scorer ABC + GemmaScorer + HFScorer + CLI + idempotent resume + daily aggregation）
- [x] 改用本機 vLLM 而非 OpenAI cloud（cost-driven decision，存進 memory）
- [x] 輸出至 `data/sentiment/`（CLI 已驗證）
- [ ] *Deferred to Phase 7：* 情緒時間衰減權重、bias 量化（已有 r=-0.273 / sign agreement 20% 證據）

### Phase 4 — Feature Engineering
- [x] **4a** `src/features/technical.py`：12 個技術指標（MA5/20/60、RSI、MACD、BBands、hv_20）via pandas-ta
- [x] **4b** `src/features/sentiment.py` 加 rolling features：sent_ma_3/7、sent_change_1、news_count_7、log_news_count_7（含 sparse-day reindex 修正）
- [x] **4c** `src/features/panel.py` 對齊到 (date, ticker) 主表，**嚴格 t-1 反 look-ahead**（features shift + merge_asof allow_exact_matches=False）
- [x] 輸出至 `data/processed/panel.parquet`
- [x] Bumped requires-python to >=3.12（pandas-ta 限制）
- [ ] *Deferred to Phase 7：* 歷史新聞回填 — 目前 sentiment 欄位在 2024 panel 上 0% 覆蓋是因為只有今天的新聞

### Phase 5 — Modeling (notebooks/03_model_training.ipynb)
- [ ] Baseline：僅技術指標 XGBoost 分類器
- [ ] Enhanced：技術 + 情緒 XGBoost
- [ ] LSTM 序列模型
- [ ] TimeSeriesSplit CV、嚴格時序切分 70/15/15
- [ ] 模組化至 `src/models/`

### Phase 6 — Backtesting (notebooks/04_backtest_results.ipynb)
- [ ] Backtrader 策略實作（`src/backtest/strategy.py`）
- [ ] 含手續費 0.1425% + 證交稅 0.3% + 滑價 0.05%
- [ ] 績效指標：Annualized Return / Sharpe / MDD / Win Rate / PF
- [ ] 圖表輸出至 `results/figures/`
- [ ] 回填 README.md 的 Results 表格

### Phase 7 — Bias & Validation
- [ ] Look-ahead bias 檢查（特徵時序嚴謹性）
- [ ] Survivorship bias（使用歷史成分股清單）
- [ ] Out-of-sample 驗證報告

### Phase 8 — Packaging
- [ ] `tests/` 加入關鍵模組單元測試
- [ ] Dockerfile
- [ ] 最終報告於 `results/reports/`

---

## Commit Log

| Phase | Commit | Description |
|---|---|---|
| 0 | `chore: bootstrap project with uv and directory scaffold` | 初始化 |
| 0 | `docs: add PROGRESS.md execution plan` | 進度計畫 |
| 1 | `feat(data): add TW50 price exploration notebook` | 資料探索 (5aa6dfe) |
| 2a | `feat(data): extract price_loader module from notebook` | 抓價模組 + CLI + tests |
| 2b | `feat(data): add cnyes RSS news scraper` | 138 篇 / 84 帶 ticker |
| — | `chore: configure self-hosted vLLM endpoint` | 改用本機 Gemma，省 OpenAI API 錢 |
| 3a | `feat(sentiment): prototype Gemma sentiment scoring notebook` | 5-class JSON prompt |
| 3b | `feat(sentiment): add multilingual baseline cross-check` | bias 對照組，r=-0.273 |
| 3c | `feat(sentiment): extract sentiment module with dual backend` | Scorer ABC + CLI + 21 tests |
| 4a | `feat(features): add technical indicators module` | 12 個 indicators via pandas-ta |
| 4b | `feat(features): add rolling sentiment features` | 5 個 rolling sentiment 特徵 |
| 4c | `feat(features): assemble (date, ticker) panel with t-1 leakage guards` | 47 tests passed，含 leakage 反偷看測試 |

---

## Notes

- Python 版本固定 3.10（README 指定）
- 套件管理使用 `uv`，不使用 `pip install` 直接裝
- 每個 phase 完成後提交 commit，避免大 commit
- API 金鑰一律從 `.env` 讀取，不進版控
