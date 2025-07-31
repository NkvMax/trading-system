import pandas as pd

__all__ = ["clean_ohlcv"]


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: sort, forward-fill missing data, ensure continuous business day index, cast to float32."""
    # Ensure index is DateTimeIndex
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # Sort and forward-fill
    df = df.sort_index()
    # Reindex to business day frequency to fill missing trading days
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B', tz=df.index.tz)
    df = df.reindex(full_idx)
    df = df.ffill()
    # Drop any fully NaN rows (e.g. before first data)
    df = df.dropna(how='all')
    # Cast to float32
    return df.astype('float32')
