# Financial News Sentiment and Stock Market Correlation Analysis

## Overview
This project analyzes the Financial News and Stock Price Integration Dataset (FNSPID) to enhance Nova Financial Solutions' predictive analytics. It focuses on quantifying news headline sentiment, correlating it with stock price movements, and proposing investment strategies. The analysis leverages NLP, time series analysis, and technical indicators to derive actionable insights.

## Objectives
- **Sentiment Analysis**: Quantify tone of financial news headlines using NLP.
- **Correlation Analysis**: Establish statistical links between news sentiment and stock price movements.
- **Investment Strategies**: Recommend predictive trading strategies based on sentiment insights.

## Dataset
- **Source**: FNSPID, combining news and stock price data.
- **Structure**:
  - `headline`: News article title (e.g., "AAPL hits all-time high").
  - `url`: Link to the full article.
  - `publisher`: Author or organization.
  - `date`: Publication date and time (UTC-4).
  - `stock`: Stock ticker symbol (e.g., AAPL).

## Project Structure
```
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
├── notebooks/
│   ├── __init__.py
│   └── README.md
├── tests/
│   ├── __init__.py
└── scripts/
    ├── __init__.py
    └── README.md
```

## Tasks
### Task 1: Data Engineering & Exploratory Data Analysis (EDA)
- **GitHub Setup**: Created repository with `task-1` branch; minimum 3 daily commits.
- **Descriptive Statistics**:
  - Headline length: Mean 73 chars, std 40 chars.
  - Top publishers: Paul Quintaro (228,373 articles), Lisa Levin (186,979), Benzinga Newsdesk (150,484).
  - Publication trends: Spikes in 2020, weekdays (Thu, Wed, Mon).
- **Text Analysis**: LDA identified topics like "higher cramer spikes," keywords: "stocks," "earnings."
- **Time Series**: 60% of articles published 9 AM–12 PM UTC-4.
- **Publisher Analysis**: Financial outlets dominate; benzinga.com leads email domains.

### Task 2: Quantitative Analysis
- **Branch**: `task-2`, merged `task-1` via PR.
- **Data**: Stock prices (Open, High, Low, Close, Volume) in pandas DataFrame.
- **Indicators**: SMA (20, 50, 200-day), RSI, MACD, Bollinger Bands, Stochastic Oscillator using TA-Lib and PyNance.
- **Visualizations**: Plotted price trends with SMA, RSI, MACD, and volume spikes.

### Task 3: Sentiment & Stock Correlation
- **Branch**: `task-3`, merged `task-2` via PR.
- **Data Alignment**: Matched news and stock dates.
- **Sentiment Analysis**: TextBlob scored headlines (45% neutral, 24% positive, 6% negative).
- **Correlation**: Pearson coefficient ~0.02, indicating weak positive correlation.
- **Challenges & Solutions**:
  - **Data Misalignment**: Standardized timestamps, used daily averages.
  - **Noisy Sentiment**: Applied polarity thresholds (>0.1 positive, <-0.1 negative), used TextBlob+VADER.
  - **High Dimensionality**: Optimized LDA to 30 topics.
  - **Computational Load**: Implemented batch processing and multiprocessing.

## Key Findings
- Weak positive correlation (~0.02) between news sentiment and daily stock returns.
- Positive sentiment from key publishers (e.g., Benzinga) linked to short-term price increases (1-2 days).

## Recommendations
- Develop real-time news sentiment pipeline.
- Integrate social media sentiment (e.g., X posts).
- Train ML models (LSTM, Random Forest) for price prediction.
- Implement sentiment-based trading triggers (buy on positive, sell on negative).
- Weight publishers by historical impact for trading algorithms.

## Requirements
- Python 3.8+
- Libraries: pandas, NLTK, TextBlob, TA-Lib, PyNance, matplotlib, seaborn, glob
- Install via `pip install -r requirements.txt`

## Usage
1. Clone the repository: `git clone <repository-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Run EDA notebooks in `notebooks/` for analysis.
4. Execute scripts in `scripts/` for data processing and visualizations.
