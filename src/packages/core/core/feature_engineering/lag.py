from __future__ import annotations
import pandas as pd
from typing import Sequence


def _extract_series(df: pd.DataFrame, column: str) -> pd.Series:
    """
    If df[column] returned a DataFrame with one column, we extract the Series, from it,
    otherwise we return it as is (Series).
    """
    s = df[column]
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 1:
            # one column - we take it
            return s.iloc[:, 0]
        else:
            raise ValueError(f"Column '{column}' yielded a DataFrame with multiple columns: {s.columns.tolist()}")
    return s  # already Series


def add_lags(df: pd.DataFrame, column: str, periods: Sequence[int]) -> pd.DataFrame:
    """
    Generate lag features for a specified column.

    Parameters:
        df: Input DataFrame.
        column: Name of the column to lag.
        periods: Sequence of lag periods.

    Returns:
        DataFrame with lag features, indexed like the input DataFrame.
    """
    s = _extract_series(df, column)
    out: dict[str, pd.Series] = {}
    for p in periods:
        out[f"{column}_lag{p}"] = s.shift(p)
    # Collect DataFrame and cast type
    return pd.DataFrame(
        {k: v.astype("float32") for k, v in out.items()},
        index=df.index
    )


def add_rolling_stats(
        df: pd.DataFrame,
        column: str,
        windows: Sequence[int],
        stats: Sequence[str]
) -> pd.DataFrame:
    """
    Compute rolling window statistics for a specified column.

    Parameters:
        df: Input DataFrame.
        column: Name of the column to compute rolling stats on.
        windows: Sequence of window sizes.
        stats: Sequence of statistics to compute (e.g., 'mean', 'std', 'zscore').

    Returns:
        DataFrame with rolling statistics, indexed like the input DataFrame.
    """
    s = _extract_series(df, column)
    out: dict[str, pd.Series] = {}
    for w in windows:
        roll = s.rolling(w)
        if "mean" in stats:
            out[f"{column}_roll{w}_mean"] = roll.mean()
        if "std" in stats:
            out[f"{column}_roll{w}_std"] = roll.std()
        if "zscore" in stats:
            out[f"{column}_roll{w}_z"] = (s - roll.mean()) / roll.std()
    return pd.DataFrame(
        {k: v.astype("float32") for k, v in out.items()},
        index=df.index
    )


def generate_lag_list(max_lag: int = 30) -> list[int]:
    """
    Helper to generate a list of lag periods up to max_lag.

    This can be used in YAML configs as:
        lags: [{column: Close, periods: !lag 3}]
    """
    return list(range(1, max_lag + 1))
