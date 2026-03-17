# Lowe's Store Sales — Time Series Forecasting

End-to-end time series forecasting project using the Kaggle Store Sales dataset.  
Applies seasonality decomposition, SARIMA, and Prophet to weekly retail sales.

---

## Project Structure

```
lowes_forecasting/
├── data/
│   ├── raw/              ← Place Kaggle CSV files here
│   └── processed/        ← Cleaned/aggregated outputs
├── notebooks/
│   ├── 01_eda.ipynb      ← Phase 1: Data prep & EDA
│   ├── 02_stationarity.ipynb
│   └── 03_modeling.ipynb
├── src/
│   └── utils.py          ← Shared helper functions
├── reports/
│   └── figures/          ← Saved plots
├── requirements.txt
└── README.md
```

---

## Dataset

Download from: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

Place these files in `data/raw/`:
- `train.csv`
- `stores.csv`
- `oil.csv`
- `holidays_events.csv`
- `transactions.csv`

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Phases

| Phase | Notebook | Topics |
|-------|----------|--------|
| 1 | `01_eda.ipynb` | Data cleaning, EDA, seasonality decomposition, ACF/PACF |
| 2 | `02_stationarity.ipynb` | Stationarity tests (ADF, KPSS), differencing |
| 3 | `03_modeling.ipynb` | SARIMA, Prophet, evaluation (RMSE, MAPE) |

---

## Key Findings (Phase 1)

- Dataset: 54 product families × 54 stores × ~1700 days (weekly aggregated)
- Strong annual seasonality (seasonal strength ≈ 0.74, period = 52 weeks)
- Sales peak Nov–Dec; dip Jan–Feb post-holiday
- Oil price forward-filled (43 gaps)
- ACF shows slow decay → non-stationary, differencing required in Phase 2
