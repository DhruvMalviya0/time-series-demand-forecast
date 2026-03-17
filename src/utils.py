"""
utils.py — Shared helper functions for the Lowe's forecasting project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss


# ── Data helpers ─────────────────────────────────────────────────────────────

def load_and_merge(raw_dir: str = "data/raw") -> pd.DataFrame:
    """Load all raw CSVs and return the merged master DataFrame."""
    train    = pd.read_csv(f"{raw_dir}/train.csv",            parse_dates=["date"])
    stores   = pd.read_csv(f"{raw_dir}/stores.csv")
    oil      = pd.read_csv(f"{raw_dir}/oil.csv",              parse_dates=["date"])
    holidays = pd.read_csv(f"{raw_dir}/holidays_events.csv",  parse_dates=["date"])

    df = train.merge(stores, on="store_nbr", how="left")
    df = df.merge(oil, on="date", how="left")
    df = df.merge(
        holidays[holidays["transferred"] == False][["date", "type"]],
        on="date", how="left"
    )
    df.rename(columns={"type_x": "store_type", "type_y": "holiday_type"}, inplace=True)

    # Fill oil price gaps
    df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()

    # Time features
    df["year"]       = df["date"].dt.year
    df["month"]      = df["date"].dt.month
    df["week"]       = df["date"].dt.isocalendar().week.astype(int)
    df["dayofweek"]  = df["date"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_holiday"] = df["holiday_type"].notna().astype(int)
    df["quarter"]    = df["date"].dt.quarter

    return df


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the master DataFrame to weekly store-family level."""
    weekly = (
        df.groupby(["store_nbr", "family", pd.Grouper(key="date", freq="W")])
        .agg(
            sales=("sales", "sum"),
            onpromotion=("onpromotion", "sum"),
            avg_oil=("dcoilwtico", "mean"),
            is_holiday=("is_holiday", "max"),
        )
        .reset_index()
    )
    return weekly


# ── Quality checks ───────────────────────────────────────────────────────────

def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarising missing values and dtypes."""
    report = pd.DataFrame({
        "dtype":        df.dtypes,
        "missing_n":    df.isnull().sum(),
        "missing_pct":  (df.isnull().mean() * 100).round(2),
        "unique":       df.nunique(),
    })
    return report


def check_date_continuity(df: pd.DataFrame, date_col: str = "date", freq: str = "D") -> None:
    """Print any gaps in the date column."""
    full_range = pd.date_range(df[date_col].min(), df[date_col].max(), freq=freq)
    actual     = pd.DatetimeIndex(df[date_col].unique())
    gaps       = full_range.difference(actual)
    if len(gaps) == 0:
        print("No date gaps found.")
    else:
        print(f"{len(gaps)} date gaps found:")
        print(gaps)


# ── Stationarity tests ───────────────────────────────────────────────────────

def adf_test(series: pd.Series, label: str = "") -> dict:
    """Augmented Dickey-Fuller test. Returns result dict and prints summary."""
    result = adfuller(series.dropna(), autolag="AIC")
    out = {
        "label":      label,
        "adf_stat":   round(result[0], 4),
        "p_value":    round(result[1], 4),
        "n_lags":     result[2],
        "n_obs":      result[3],
        "stationary": result[1] < 0.05,
    }
    print(f"[ADF] {label}  stat={out['adf_stat']}  p={out['p_value']}  "
          f"→ {'Stationary' if out['stationary'] else 'Non-stationary'}")
    return out


def kpss_test(series: pd.Series, label: str = "") -> dict:
    """KPSS test for stationarity. Returns result dict and prints summary."""
    result = kpss(series.dropna(), regression="c", nlags="auto")
    out = {
        "label":      label,
        "kpss_stat":  round(result[0], 4),
        "p_value":    round(result[1], 4),
        "stationary": result[1] > 0.05,
    }
    print(f"[KPSS] {label}  stat={out['kpss_stat']}  p={out['p_value']}  "
          f"→ {'Stationary' if out['stationary'] else 'Non-stationary'}")
    return out


# ── Seasonality metrics ──────────────────────────────────────────────────────

def seasonal_strength(stl_result) -> float:
    """Compute STL seasonal strength (0 = no seasonality, 1 = pure seasonal)."""
    var_resid    = np.var(stl_result.resid)
    var_seas_res = np.var(stl_result.seasonal + stl_result.resid)
    return float(max(0, 1 - var_resid / var_seas_res))


def trend_strength(stl_result) -> float:
    """Compute STL trend strength."""
    var_resid      = np.var(stl_result.resid)
    var_trend_res  = np.var(stl_result.trend + stl_result.resid)
    return float(max(0, 1 - var_resid / var_trend_res))


# ── Evaluation metrics ───────────────────────────────────────────────────────

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Skips zeros in actual."""
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric MAPE — handles zero actuals more gracefully."""
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    mask  = denom != 0
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)
