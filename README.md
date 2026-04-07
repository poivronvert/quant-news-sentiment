# News Sentiment-Driven Quantitative Trading Strategy

利用 LLM 對財經新聞進行情緒分析，結合技術指標與機器學習模型，建構並回測台股量化交易策略。

---

## Project Overview

本專案結合自然語言處理（NLP）與傳統量化分析方法，探討「新聞情緒」是否能作為有效的交易訊號。透過 LLM 對每日財經新聞進行情緒打分，並與技術指標一同作為機器學習模型的輸入特徵，預測個股短期走勢，最後以 Backtrader 框架進行歷史回測，驗證策略績效。

### Key Features

- 多來源新聞資料自動爬取與清理
- 基於 LLM 的金融新聞情緒分析
- 技術指標特徵工程（RSI、MACD、Bollinger Bands 等）
- XGBoost / LSTM 雙模型對照實驗
- 完整回測框架與績效評估指標
- Look-ahead bias 與 survivorship bias 處理

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10 |
| Data | yfinance, FinMind, pandas, NumPy |
| NLP | OpenAI API / FinBERT, LangChain |
| ML | scikit-learn, XGBoost, PyTorch |
| Backtest | Backtrader, vectorbt |
| Visualization | matplotlib, seaborn, plotly |
| Environment | Docker, Linux |

---

## Project Structure

```
quant-news-sentiment/
├── data/
│   ├── raw/                  # 原始價格與新聞資料
│   ├── processed/            # 特徵工程後的資料
│   └── sentiment/            # LLM 情緒分數結果
├── src/
│   ├── data_collection/      # 資料爬取模組
│   │   ├── price_loader.py
│   │   └── news_scraper.py
│   ├── features/             # 特徵工程
│   │   ├── technical.py
│   │   └── sentiment.py
│   ├── models/               # 機器學習模型
│   │   ├── xgboost_model.py
│   │   └── lstm_model.py
│   ├── backtest/             # 回測引擎
│   │   └── strategy.py
│   └── utils/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_backtest_results.ipynb
├── results/
│   ├── figures/
│   └── reports/
├── tests/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Methodology

### 1. Data Collection

**價格資料：** 使用 yfinance 抓取台股 50 成分股的歷史 OHLCV 資料，期間為 2020/01/01 至 2024/12/31。

**新聞資料：** 透過 RSS 與公開財經 API 收集對應期間的個股相關新聞，每則新聞保留標題、內文摘要、發布時間與標的股票。

### 2. Sentiment Analysis

採用 LLM 對每則新聞進行情緒打分：

- **方法 A：** 使用 OpenAI GPT-4o-mini 進行 prompt-based 分類，輸出 -1（極負面）至 +1（極正面）的連續分數
- **方法 B：** 使用 FinBERT 進行 zero-shot 分類作為對照組

每日情緒特徵以加權平均彙整（權重依新聞來源可信度與發布時間衰減）。

### 3. Feature Engineering

**技術指標：**
- 移動平均線（MA5、MA20、MA60）
- 相對強弱指標 RSI(14)
- MACD（12, 26, 9）
- Bollinger Bands（20, 2）
- 歷史波動率（20 日）

**情緒指標：**
- 當日情緒分數
- 3 日 / 7 日情緒移動平均
- 情緒變化率
- 新聞數量（市場關注度代理變數）

### 4. Modeling

| Model | Purpose |
|---|---|
| **Baseline** | 僅使用技術指標的 XGBoost 分類器 |
| **Enhanced** | 技術指標 + 情緒特徵的 XGBoost 分類器 |
| **Deep Learning** | LSTM 時間序列模型，輸入完整特徵序列 |

預測目標為次日收盤報酬的方向（漲 / 跌），訓練 / 驗證 / 測試切分為 70 / 15 / 15，**嚴格按時間順序切分以避免 look-ahead bias**。

### 5. Backtesting

使用 Backtrader 框架進行回測，交易規則如下：

- 模型預測上漲訊號時持有，下跌訊號時平倉
- 單筆交易成本設為 0.1425%（券商手續費）+ 0.3%（證交稅）
- 滑價假設 0.05%
- 初始資金 NTD 1,000,000

---

## Results

| Metric | Baseline (Technical Only) | Enhanced (+ Sentiment) | LSTM |
|---|---|---|---|
| Annualized Return | TBD | TBD | TBD |
| Sharpe Ratio | TBD | TBD | TBD |
| Max Drawdown | TBD | TBD | TBD |
| Win Rate | TBD | TBD | TBD |
| Profit Factor | TBD | TBD | TBD |

*詳細結果請見 `notebooks/04_backtest_results.ipynb`*

---

## Key Considerations

### Bias Handling

**Look-ahead Bias：** 所有特徵嚴格使用 t 時點之前的資料計算，新聞情緒分數僅使用發布時間早於收盤前的新聞。

**Survivorship Bias：** 使用回測期間的歷史成分股清單，而非當前清單，避免只納入存活公司。

**Overfitting：** 採用時間序列交叉驗證（TimeSeriesSplit），並在 out-of-sample 測試集驗證最終模型。

### Limitations

- 樣本期間僅涵蓋 5 年，未包含完整景氣循環
- 新聞情緒可能存在發布延遲，實務中需考量資訊效率
- 未納入交易量限制與市場衝擊成本
- 策略基於日頻資料，不適用於高頻交易場景

---

## How to Run

### Setup

```bash
git clone https://github.com/your-username/quant-news-sentiment.git
cd quant-news-sentiment

# 建立虛擬環境
python -m venv venv
source venv/bin/activate

# 安裝依賴
pip install -r requirements.txt

# 設定 API 金鑰
cp .env.example .env
# 編輯 .env 填入 OPENAI_API_KEY
```

### Run with Docker

```bash
docker build -t quant-news .
docker run -v $(pwd)/data:/app/data quant-news
```

### Pipeline Execution

```bash
# 1. 資料收集
python src/data_collection/price_loader.py
python src/data_collection/news_scraper.py

# 2. 情緒分析
python src/features/sentiment.py

# 3. 特徵工程
python src/features/technical.py

# 4. 模型訓練
python src/models/xgboost_model.py

# 5. 回測
python src/backtest/strategy.py
```

---

## Future Work

- 加入財報資料（EPS、營收成長率）作為基本面特徵
- 嘗試 Transformer-based 時間序列模型
- 擴展至期貨與加密貨幣市場
- 實作 walk-forward 滾動回測
- 引入投資組合最佳化（Markowitz、Risk Parity）

---

## References

1. Tetlock, P. C. (2007). *Giving Content to Investor Sentiment: The Role of Media in the Stock Market*. Journal of Finance.
2. Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*.
3. López de Prado, M. (2018). *Advances in Financial Machine Learning*.

---

## Author

郭明潔 (Alice Kuo)
AI Engineer | Financial Data Intelligence
