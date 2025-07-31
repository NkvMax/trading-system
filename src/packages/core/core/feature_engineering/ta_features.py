"""
TA indicators and lags for the final set of features.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import ta


def add_ta_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and append core technical indicators to the DataFrame:
      - log return and 14-day return volatility
      - RSI (14), MACD diff, CCI (20), Stochastic %K, ATR (14)
    Returns a new DataFrame with these columns added.
    """
    out = df.copy()
    # log-yield and moving average
    out["log_ret"] = np.log(out["Close"]).diff()
    out["ret_vol_14"] = out["log_ret"].rolling(14).std()

    # RSI, MACD, CCI, Stochastic, ATR (ta-lib)
    out["rsi_14"] = ta.momentum.rsi(out["Close"], window=14)
    macd = ta.trend.macd_diff(out["Close"])
    out["macd"] = macd
    out["cci_20"] = ta.trend.cci(out["High"], out["Low"], out["Close"], window=20)
    out["stoch_k"] = ta.momentum.stoch(out["High"], out["Low"], out["Close"])
    out["atr_14"] = ta.volatility.average_true_range(
        out["High"], out["Low"], out["Close"], window=14
    )
    return out


def add_lags(df: pd.DataFrame, n_lags: int = 3) -> pd.DataFrame:
    """
    Append lagged versions of the 'Close' column:
      - Close_lag_1, Close_lag_2, ..., Close_lag_{n_lags}
    Returns a new DataFrame with lagged columns.
    """
    out = df.copy()
    for i in range(1, n_lags + 1):
        out[f"Close_lag_{i}"] = out["Close"].shift(i)
    return out
