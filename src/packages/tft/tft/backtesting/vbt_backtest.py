import torch
import joblib
from tft.model.tft import TFTModel
from tft.model.hybrid_xgb_tft import HybridXGBTFT
from tft.data import make_dataset
import numpy as np
import pandas as pd
import vectorbt as vbt
from pathlib import Path
import os


def load_my_model(ds, phase, model_type="TFT", save_dir="models/saved"):
    """
    Load a pretrained model for a given phase and type.

    For model_type == "TFT":
      - Expects a .pth file at {save_dir}/tft_{phase}.pth.
      - Initializes a TFTModel wrapper and loads state_dict.
    For model_type == "Hybrid":
      - Expects a .joblib file at {save_dir}/hybrid_model_{phase}.joblib.
      - Loads and returns the XGBoost model.
    Raises:
      FileNotFoundError if the expected file is missing.
      ValueError for unsupported model_type.
    """

    if model_type == "TFT":
        pth_path = os.path.join(save_dir, f"tft_{phase}.pth")
        if not os.path.exists(pth_path):
            raise FileNotFoundError(f"File {pth_path} not found!")
        model = TFTModel(
            ds,
            learning_rate=1e-3,
            hidden_size=64,
            dropout=0.1,
            max_epochs=1
        )
        # Initialize the model "quickly" if there is no build_model method
        try:
            if model.model is None:
                # If the model has not yet been built, we build a minimal fit
                model.fit(gpus=0, batch_size=8)
        except AttributeError:
            # If there is no model field at all, it will appear after fit
            model.fit(gpus=0, batch_size=8)
        # Now model.model is definitely created, you can load weights
        model.model.load_state_dict(torch.load(pth_path, map_location="cpu"))
        return model

    elif model_type == "Hybrid":
        joblib_path = os.path.join(save_dir, f"hybrid_model_{phase}.joblib")
        if not os.path.exists(joblib_path):
            raise FileNotFoundError(f"File {joblib_path} not found!")
        model = joblib.load(joblib_path)
        return model

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_vbt_backtest_for_phase(
        *,
        asset: str,
        phase: str,
        threshold: float = 0.005,
        init_cash: float = 10_000.0,
        fee: float = 0.0005,
        encoder_length: int = 90,
        horizon: int = 7,
) -> dict:
    """
    Execute a vectorbt backtest for a single market phase using TFT or hybrid model.

    Parameters:
      asset (str): Ticker symbol, e.g. "ETH-USD".
      phase (str): One of the predefined market phases.
      threshold (float): Entry/exit threshold for expected return.
      init_cash (float): Starting capital for the backtest.
      fee (float): Trading fee rate per trade.
      encoder_length (int): Lookback window size for dataset.
      horizon (int): Forecast horizon for model predictions.

    Workflow:
      1. Load the feather file for the given asset and phase.
      2. Build a TimeSeriesDataSet for TFT.
      3. Determine model type (TFT vs Hybrid) based on saved files.
      4. Load the appropriate model via load_my_model().
      5. Generate predictions and convert to entry/exit signals.
      6. Create two Portfolios: one for the model, one for buy-and-hold.
      7. Collect and return stats and portfolio objects.

    Returns:
      dict: {
        "pf_tft": vectorbt Portfolio for model signals,
        "pf_hold": vectorbt Portfolio for buy-and-hold,
        "stats": DataFrame of performance metrics
      }
    """
    BASE = Path(__file__).resolve().parents[5]
    feather_path = BASE / f"data/regimes/{asset}_{phase}_fixed.feather"
    save_dir = BASE / "models/saved"

    # Loading dataset
    if not feather_path.exists():
        raise FileNotFoundError(f"Dataset {feather_path} not found!")

    df = pd.read_feather(feather_path)
    ds = make_dataset(
        feather_path,
        encoder_length=encoder_length,
        horizon=horizon,
        phase=phase,
    )
    ds.dataframe = df.copy()

    # Determine the model type (either by phase or by argument)
    model_type = "TFT"
    if "hybrid" in str(save_dir / f"hybrid_model_{phase}.joblib"):
        if (save_dir / f"hybrid_model_{phase}.joblib").exists():
            model_type = "Hybrid"
    model = load_my_model(ds, phase, model_type=model_type, save_dir=str(save_dir))

    # We receive predictions
    pred_loader = ds.to_dataloader(train=False, batch_size=256, num_workers=0)
    preds = model.predict(pred_loader).mean(axis=1)

    pred_series = pd.Series(preds, index=df["time"].iloc[-len(preds):])
    price_series = df.set_index("time")["Close"].loc[pred_series.index].ffill()

    def _gen_signals(pred, price, thr=0.005):
        exp_ret = pred / price - 1.0
        return exp_ret > thr, exp_ret < -thr

    entries, exits = _gen_signals(pred_series.values, price_series.values, threshold)

    pf_tft = vbt.Portfolio.from_signals(
        price_series,
        entries,
        exits,
        direction="longonly",
        init_cash=init_cash,
        fees=fee,
        freq="D",
    )
    pf_hold = vbt.Portfolio.from_holding(price_series, init_cash=init_cash, freq="D")

    cols = ["Total Return [%]", "Max Drawdown [%]", "Sharpe Ratio"]
    stats = pd.concat([pf_tft.stats()[cols], pf_hold.stats()[cols]], axis=1)
    stats.columns = [f"TFT-{phase}", f"Buy&Hold-{phase}"]

    return {"pf_tft": pf_tft, "pf_hold": pf_hold, "stats": stats.round(2)}
