import typer
from pathlib import Path
import pandas as pd
from tft.config import TrainParams, XGBParams, YAMLConfig
from tft.model.hybrid_xgb_tft import HybridXGBTFT
from tft.utils import ensure_dir

app = typer.Typer()


@app.command()
def train_xgb_tft(
        asset: str = typer.Argument(...),
        phase: str | None = typer.Option(None),
):
    """
    CLI command to train the Hybrid XGBoost + TFT model:
    1. Initialize training and XGB parameters (paths, encoder/horizon lengths).
    2. Load the full dataset from Feather.
    3. Create HybridXGBTFT with parameters from YAMLConfig or defaults.
    4. Fit the hybrid model:
       - XGBoost is trained on flattened features.
       - TFT is trained on time-series windows.
    5. Ensure the output directory exists.
    6. Save the trained XGBoost model to the configured path.
    """
    tp = TrainParams(asset=asset, phase=phase)
    xp = XGBParams(asset=asset, phase=phase)
    cfg = YAMLConfig.load(asset)
    df = pd.read_feather(Path(tp.asset + ".feather").parent / f"{asset}.feather")
    hy = HybridXGBTFT(
        df,
        xgb_params=cfg.raw.get("xgb_params", xp.params),
        tft_params=cfg.raw.get("tft_params", {})
    )
    hy.fit(
        encoder_length=tp.encoder_length,
        horizon=tp.horizon,
        gpus=tp.gpus,
        batch_size=tp.batch_size
    )
    ensure_dir(xp.model_path().parent)
    hy.xgb_model.save_model(str(xp.model_path()))
    typer.echo(f"Saved XGB ->{xp.model_path()}")
