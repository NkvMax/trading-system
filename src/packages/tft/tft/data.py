from __future__ import annotations
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet


def _add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute additional features for the dataset:
    - log_return: logarithmic return of 'Close' price
    """
    df = df.copy()
    df["log_return"] = np.log(df["Close"]).diff().fillna(0.0)
    return df


def make_dataset(
        path: Union[str, Path],
        encoder_length: int,
        horizon: int,
        phase: str | None = None,
) -> TimeSeriesDataSet:
    """
    Build a TimeSeriesDataSet for TFT training:
    1. Load raw data from Feather or Parquet.
    2. Filter by market phase if provided.
    3. Sort by time and add time_idx & asset_id.
    4. Add extra features (e.g., log_return).
    5. Instantiate TimeSeriesDataSet with appropriate fields:
       - time_idx: index for sequence order
       - target: 'Close' price
       - known and unknown real covariates
       - allow missing timesteps for irregular data
    6. Attach the processed dataframe for inspection.
    """
    path = Path(path)
    df = pd.read_feather(path) if path.suffix == ".feather" else pd.read_parquet(path)
    if phase is not None and "phase" in df.columns:
        df = df[df["phase"] == phase]
        if df.empty:
            raise ValueError(f"No data for phase '{phase}' in {path.name}")
    df = df.sort_values("time").reset_index(drop=True)
    df["time_idx"] = np.arange(len(df))
    df["asset_id"] = path.stem
    df = _add_extra_features(df)

    ts_dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="Close",
        group_ids=["asset_id"],
        max_encoder_length=encoder_length,
        max_prediction_length=horizon,
        static_categoricals=["asset_id"],
        time_varying_known_reals=["time_idx", "log_return"],
        time_varying_unknown_reals=["Close", "Volume"],
        add_relative_time_idx=True,
        add_target_scales=True,
        allow_missing_timesteps=True,
    )

    ts_dataset.dataframe = df.copy()

    return ts_dataset
