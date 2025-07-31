from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from pytorch_forecasting import TimeSeriesDataSet

from .tft import TFTModel


# Helper function for creating windows
def create_windowed_dataset(df, features, target, encoder_length, horizon):
    X, y = [], []
    for i in range(len(df) - encoder_length - horizon + 1):
        X_window = df[features].iloc[i:i + encoder_length].values.flatten()
        y_window = df[target].iloc[i + encoder_length:i + encoder_length + horizon].values
        X.append(X_window)
        y.append(y_window)
    return np.array(X), np.array(y)


class HybridXGBTFT:
    """
    1) XGB learns from windows (similar to TFT): encoder_length -> horizon
    2) XGB forecasts are added to the dataset as known-real
    3) TFT refines forecast (including XGB)
    4) Total = xgb_ratio , pred_xgb + (1-xgb_ratio) , pred_tft
    """

    def __init__(
            self,
            dataset: TimeSeriesDataSet,
            *,
            xgb_ratio: float = 0.7,
            xgb_params: dict | None = None,
            tft_params: dict | None = None,
            learning_rate: float = 1e-3,
            hidden_size: int = 64,
            dropout: float = 0.1,
            max_epochs: int = 30,
    ):
        """
        Initialize the hybrid model with XGBoost and TFT hyperparameters.
        """
        self.dataset = dataset

        self.xgb_ratio = xgb_ratio
        self.xgb_params = xgb_params or {}
        self.tft_params = tft_params or {}

        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.max_epochs = max_epochs

        self.xgb_model: xgb.XGBRegressor | None = None
        self.tft_model: TFTModel | None = None
        self.features: list[str] = []

    @staticmethod
    def _raw_df(ds: TimeSeriesDataSet) -> pd.DataFrame:
        if hasattr(ds, "dataframe"):
            return ds.dataframe.copy()
        if isinstance(getattr(ds, "data", None), pd.DataFrame):
            return ds.data.copy()
        raise RuntimeError(
            "Unable to extract source DataFrame from TimeSeriesDataSet."
            "Update pytorch-forecasting or pass the DataFrame manually."
        )

    def fit(self, *, gpus: int = 0, batch_size: int = 64):
        df = self._raw_df(self.dataset)
        exclude = {"Close", "time", "phase", "time_idx", "asset_id"}
        self.features = [c for c in df.columns if c not in exclude]

        encoder_length = self.dataset.max_encoder_length
        horizon = self.dataset.max_prediction_length

        # IMPORTANT: Target normalization for XGB
        target_mean = float(self.dataset.target_normalizer.center_)
        target_std = float(self.dataset.target_normalizer.scale_)
        df["Close_norm"] = (df["Close"] - target_mean) / target_std

        # Windows for XGB by normalized target
        X, y = create_windowed_dataset(df, self.features, "Close_norm", encoder_length, horizon)
        self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
        self.xgb_model.fit(X, y)

        xgb_preds = self.xgb_model.predict(X)

        start_idx = encoder_length
        stop_idx = encoder_length + len(xgb_preds)
        df_tft = df.iloc[start_idx:stop_idx].copy().reset_index(drop=True)
        df_tft["time_idx"] = np.arange(len(df_tft))

        for h in range(horizon):
            df_tft[f"xgb_pred_{h}"] = xgb_preds[:, h]

        time_varying_known_reals = ["time_idx"] + [f"xgb_pred_{h}" for h in range(horizon)]
        hybrid_ds = TimeSeriesDataSet(
            df_tft,
            time_idx="time_idx",
            target="Close",
            group_ids=["asset_id"],
            max_encoder_length=encoder_length,
            max_prediction_length=horizon,
            static_categoricals=["asset_id"],
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=["Close", "Volume"],
            add_relative_time_idx=True,
            add_target_scales=True,
            allow_missing_timesteps=True,
        )
        hybrid_ds.dataframe = df_tft.copy()

        for key in ["learning_rate", "hidden_size", "dropout", "max_epochs"]:
            self.tft_params.pop(key, None)

        self.tft_model = TFTModel(
            hybrid_ds,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            max_epochs=self.max_epochs,
            **self.tft_params,
        )
        self.tft_model.fit(gpus=gpus, batch_size=batch_size)
        self.dataset = hybrid_ds
        return self

    def predict(self, dataset: TimeSeriesDataSet | None = None, **tft_kwargs):
        """
        Generate predictions from both TFT and XGBoost, combine them by the configured ratio,
        and return the blended forecast.
        """
        ds = dataset or self.dataset

        # Check: .dataframe is required for XGB
        if not hasattr(ds, "dataframe"):
            raise RuntimeError(
                "\n[HybridXGBTFT] For predict, be sure to use self.dataset, "
                "or make sure that the TimeSeriesDataSet object has a .dataframe attribute.\n"
                "This usually means calling predict as:\n"
                "hybrid_model.predict(hybrid_model.dataset)\n"
                "or add the original DataFrame manually:\n"
                " js dataset.dataframe = original df.copy()\n"
            )

        # TFT Prediction
        tft_pred = self.tft_model.predict(ds, **tft_kwargs)
        if hasattr(tft_pred, "detach"):
            tft_pred = tft_pred.detach().cpu().numpy()
        if tft_pred.ndim == 3 and tft_pred.shape[2] == 1:
            tft_pred = tft_pred.squeeze(-1)  # (samples, horizon)

        # XGB Prediction
        df_pred = self._raw_df(ds)
        encoder_length = self.dataset.max_encoder_length
        horizon = self.dataset.max_prediction_length
        X, _ = create_windowed_dataset(df_pred, self.features, "Close", encoder_length, horizon)
        xgb_pred = self.xgb_model.predict(X)

        # DEBUG
        print(f"[HybridXGBTFT] xgb_pred.shape: {xgb_pred.shape}")
        print(f"[HybridXGBTFT] tft_pred.shape: {tft_pred.shape}")

        if xgb_pred.shape != tft_pred.shape:
            raise RuntimeError(
                f"[HybridXGBTFT] Prediction dimensions do not match: "
                f"xgb_pred {xgb_pred.shape} vs tft_pred {tft_pred.shape}.\n"
                f"Check the logic of windows and horizon!"
            )

        # gluing
        result = self.xgb_ratio * xgb_pred + (1.0 - self.xgb_ratio) * tft_pred
        return result


try:
    from .safe_hybrid import SafeHybridXGBTFT

    HybridXGBTFT = SafeHybridXGBTFT
    __all__ = ["HybridXGBTFT", "SafeHybridXGBTFT"]
except ImportError as e:
    print(f"Import error for SafeHybridXGBTFT: {e}")
    __all__ = ["HybridXGBTFT"]
