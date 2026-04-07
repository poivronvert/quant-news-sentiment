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

### Phase 3 — Sentiment Analysis (notebooks/02_sentiment_analysis.ipynb)
- [ ] FinBERT zero-shot baseline
- [ ] OpenAI GPT-4o-mini prompt-based scoring（需 `OPENAI_API_KEY`）
- [ ] 每日情緒聚合（加權平均 + 時間衰減）
- [ ] 輸出至 `data/sentiment/`

### Phase 4 — Feature Engineering
- [ ] `src/features/technical.py`：MA / RSI / MACD / BBands / HV
- [ ] `src/features/sentiment.py`：情緒 MA、變化率、新聞量
- [ ] 對齊技術指標與情緒特徵至同一交易日索引
- [ ] 輸出至 `data/processed/`

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

---

## Notes

- Python 版本固定 3.10（README 指定）
- 套件管理使用 `uv`，不使用 `pip install` 直接裝
- 每個 phase 完成後提交 commit，避免大 commit
- API 金鑰一律從 `.env` 讀取，不進版控
