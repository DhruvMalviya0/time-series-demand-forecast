# Time Series Demand Forecasting

Predicting weekly retail product sales using classical and modern time series models — SARIMA and Prophet — with a full EDA and decomposition pipeline.

---

## Problem Statement

Retail demand forecasting is one of the most business-critical data science problems. Inaccurate forecasts lead to stockouts, overstock, and poor inventory planning. This project builds an end-to-end forecasting pipeline that:

- Identifies seasonal demand patterns across product families
- Tests and achieves stationarity before modeling
- Compares multiple forecasting approaches objectively
- Produces an 8-week forward forecast with business-interpretable metrics

---

## Applicable Domains

Retail · Supply Chain · E-commerce · FMCG · Home Improvement

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data wrangling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Time series | `statsmodels`, `pmdarima` |
| Forecasting | `prophet` |
| Evaluation | `scikit-learn`, `scipy` |

---

## Project Structure

```
time-series-demand-forecast/
├── data/
│   └── processed/            # Cleaned weekly aggregated data
├── notebooks/
│   ├── 01_eda.ipynb           # Phase 1 — EDA & decomposition
│   ├── 02_stationarity.ipynb  # Phase 2 — Stationarity & baselines
│   └── 03_modeling.ipynb      # Phase 3 — SARIMA & Prophet
├── src/
│   └── utils.py               # Reusable helper functions
├── reports/
│   └── figures/               # All saved plots (PNG)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Roadmap

### Phase 1 — Data Preparation & EDA

> Goal: Understand the data deeply before touching any model.

- Load and merge raw datasets (transactions, stores, oil prices, holidays)
- Handle missing values — forward-fill oil price gaps
- Engineer time features: year, month, week, quarter, is_weekend, is_holiday
- Aggregate to weekly store-family level
- EDA visualizations: trend, monthly seasonality, year-over-year, correlation heatmap, outlier detection
- Decompose series using STL (robust=True, period=52)
- Calculate seasonal strength score — confirms SARIMA/Prophet needed
- Generate ACF & PACF plots — confirms non-stationarity

**Key output:** `data/processed/weekly_sales.csv`, 8 saved figures, seasonal strength score

---

### Phase 2 — Baseline & Stationarity

> Goal: Formally prove non-stationarity and establish a performance floor to beat.

- ADF test (Augmented Dickey-Fuller) on original series
- KPSS test as complementary confirmation
- Apply first-order differencing `diff(1)` and seasonal differencing `diff(52)`
- Re-run ADF on differenced series → confirm stationarity
- Read ACF/PACF on differenced series → determine SARIMA `(p,d,q)(P,D,Q,52)` starting order
- Build 4 baseline models: Naive, Seasonal Naive, Rolling Mean, Holt-Winters
- Score all baselines on MAE, RMSE, MAPE over 12-week test horizon
- Save baseline scorecard as target for Phase 3

**Key output:** `data/processed/baseline_scorecard.csv`, SARIMA starting order, target RMSE

---

### Phase 3 — Model Building & Tuning

> Goal: Beat the best baseline with a properly tuned forecasting model.

- Fit SARIMA using `pmdarima.auto_arima` for automated order selection
- Fit Prophet with yearly/weekly seasonality and holiday regressors
- Tune hyperparameters — SARIMA via AIC, Prophet via cross-validation
- Compare all models on the same 12-week test set
- Select best model based on RMSE and MAPE

**Key output:** Trained SARIMA and Prophet models, model comparison table

---

### Phase 4 — Forecast & Visual Report

> Goal: Deliver a production-ready 8-week forecast with business context.

- Generate 8-week forward forecast with confidence intervals
- Build Seaborn visual report: forecast vs actuals, residual diagnostics, component plots
- Translate RMSE into business terms (inventory planning variance in units/revenue)
- Push final report figures to `reports/figures/`
- Clean up repo, finalize README with results table

**Key output:** 8-week forecast, full visual report, updated results table below

---

## Results

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| Naive | — | — | — |
| Seasonal Naive | — | — | — |
| Holt-Winters | — | — | — |
| SARIMA | — | — | — |
| Prophet | — | — | — |

> Results will be updated after Phase 3 is complete.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/DhruvMalviya0/time-series-demand-forecast.git
cd time-series-demand-forecast

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset

**Kaggle — Store Sales: Time Series Forecasting**
https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

Download and place the raw CSV files in `data/raw/` before running the notebooks.
The `data/` folder is excluded from this repo via `.gitignore` — raw files must be downloaded locally.

---

## Commit History

| Commit | Description |
|---|---|
| `phase 1` | Data loading, EDA, seasonality decomposition — 16/16 validation passed |
| `phase 2` | ADF/KPSS stationarity tests, differencing, baseline models scored |
| `phase 3` | SARIMA + Prophet models, hyperparameter tuning, comparison |
| `phase 4` | 8-week forecast, visual report, final README |

---

## Author

**Dhruv Malviya**
[github.com/DhruvMalviya0](https://github.com/DhruvMalviya0)
