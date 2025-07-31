"""
Low-level wrapper functions for `ta` / pandas-rolling.
Each function ***returns a DataFrame*** (dtype float32) with
one or more indicator columns.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import ta


def sma(close: pd.Series, window: int) -> pd.DataFrame:
    """
    Compute Simple Moving Average (SMA) over `window` periods for the `close` series.

    Returns:
        DataFrame with a single column `SMA_<window>` of dtype float32.
    """
    return pd.DataFrame({f"SMA_{window}": close.rolling(window).mean().astype("float32")})


def ema(close: pd.Series, window: int) -> pd.DataFrame:
    """
    Compute Exponential Moving Average (EMA) over `window` periods for the `close` series.

    Returns:
        DataFrame with a single column `EMA_<window>` of dtype float32.
    """
    return pd.DataFrame({f"EMA_{window}": close.ewm(span=window, adjust=False).mean().astype("float32")})


def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """
    Compute Bollinger Bands (upper, lower, and %B) over `window` periods and `n_std` standard deviations.

    Returns:
        DataFrame with columns:
          - `BB_upper_<window>`
          - `BB_lower_<window>`
          - `BB_%B_<window>`
        All columns are dtype float32.
    """
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    pct_b = (close - lower) / (upper - lower)
    return pd.DataFrame({
        f"BB_upper_{window}": upper.astype("float32"),
        f"BB_lower_{window}": lower.astype("float32"),
        f"BB_%B_{window}": pct_b.astype("float32")
    })


def donchian(high: pd.Series, low: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Compute Donchian channel high, low, and width over `window` periods.

    Returns:
        DataFrame with columns:
          - `Donchian_high_<window>`
          - `Donchian_low_<window>`
          - `Donchian_width_<window>`
        All columns are dtype float32.
    """
    high_n = high.rolling(window).max()
    low_n = low.rolling(window).min()
    width = high_n - low_n
    return pd.DataFrame({
        f"Donchian_high_{window}": high_n.astype("float32"),
        f"Donchian_low_{window}": low_n.astype("float32"),
        f"Donchian_width_{window}": width.astype("float32"),
    })


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Compute MACD line, signal line, and histogram.

    Args:
        fast: window for the fast EMA
        slow: window for the slow EMA
        signal: window for the signal line

    Returns:
        DataFrame with columns:
          - `MACD_<fast>_<slow>`
          - `MACDs_<signal>`
          - `MACD_hist`
        All columns are dtype float32.
    """
    macd_line = ta.trend.macd(close, window_fast=fast, window_slow=slow, fillna=False)
    macd_signal = ta.trend.macd_signal(close, window_fast=fast, window_slow=slow,
                                       window_sign=signal, fillna=False)
    macd_hist = macd_line - macd_signal
    return pd.DataFrame({
        f"MACD_{fast}_{slow}": macd_line.astype("float32"),
        f"MACDs_{signal}": macd_signal.astype("float32"),
        f"MACD_hist": macd_hist.astype("float32")
    })


def rsi(close: pd.Series, window: int = 14) -> pd.DataFrame:
    """
    Compute Relative Strength Index (RSI) over `window` periods for the `close` series.

    Returns:
        DataFrame with a single column `RSI_<window>` of dtype float32.
    """
    return pd.DataFrame({f"RSI_{window}": ta.momentum.rsi(close, window=window).astype("float32")})


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.DataFrame:
    """
    Compute Williams %R over `window` periods.

    Returns:
        DataFrame with a single column `WILLR_<window>` of dtype float32.
    """
    w = ta.momentum.williams_r(high, low, close, lbp=window)
    return pd.DataFrame({f"WILLR_{window}": w.astype("float32")})


def stoch(high: pd.Series, low: pd.Series, close: pd.Series,
          window: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
    """
    Compute Stochastic oscillator %K and %D.

    Args:
        window: lookback period
        smooth_k: smoothing window for %K
        smooth_d: smoothing window for %D

    Returns:
        DataFrame with columns `StochK_<window>` and `StochD_<window>`, dtype float32.
    """
    k = ta.momentum.stoch(high, low, close, window, smooth_k)
    d = ta.momentum.stoch_signal(high, low, close, window, smooth_k, smooth_d)
    return pd.DataFrame({
        f"StochK_{window}": k.astype("float32"),
        f"StochD_{window}": d.astype("float32")
    })


def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Compute Commodity Channel Index (CCI) over `window` periods.

    Returns:
        DataFrame with a single column `CCI_<window>` of dtype float32.
    """
    return pd.DataFrame({f"CCI_{window}": ta.trend.cci(high, low, close, window=window).astype("float32")})


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.DataFrame:
    """
    Compute Average Directional Index (ADX) over `window` periods.

    Returns:
        DataFrame with a single column `ADX_<window>` of dtype float32.
    """
    adx_val = ta.trend.adx(high, low, close, window=window)
    return pd.DataFrame({f"ADX_{window}": adx_val.astype("float32")})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.DataFrame:
    """
    Compute Average True Range (ATR) over `window` periods.

    Returns:
        DataFrame with a single column `ATR_<window>` of dtype float32.
    """
    atr_val = ta.volatility.average_true_range(high, low, close, window=window)
    return pd.DataFrame({f"ATR_{window}": atr_val.astype("float32")})


def roc(close: pd.Series, window: int = 10) -> pd.DataFrame:
    """
    Compute Rate of Change (ROC) over `window` periods for the `close` series.

    Returns:
        DataFrame with a single column `ROC_<window>` of dtype float32.
    """
    roc_val = close.pct_change(window)
    return pd.DataFrame({f"ROC_{window}": roc_val.astype("float32")})


def momentum(close: pd.Series, window: int = 10) -> pd.DataFrame:
    """
    Compute momentum (difference) over `window` periods for the `close` series.

    Returns:
        DataFrame with a single column `MOM_<window>` of dtype float32.
    """
    mom = close.diff(window)
    return pd.DataFrame({f"MOM_{window}": mom.astype("float32")})
