import typer
from pathlib import Path
from tft.config import TrainParams, YAMLConfig
from tft.data import make_dataset
from tft.model.tft import TFTModel
from tft.utils import ensure_dir

app = typer.Typer()


@app.command()
def train_tft(
        asset: str = typer.Argument(...),
        phase: str | None = typer.Option(None),
        epochs: int = typer.Option(30),
        gpu: bool = typer.Option(False),
):
    """
    CLI command to train a TFT model for a specific market phase:
    1. Load training parameters from TrainParams (asset, phase, epochs, gpus).
    2. Load model hyperparameters from YAMLConfig (overrides defaults).
    3. Build the dataset (encoder_length + horizon) filtered by phase.
    4. Initialize TFTModel with configured hyperparameters.
    5. Ensure model output directory exists.
    6. Fit the model (CPU or GPU) with given batch size.
    7. Save checkpoint to the path defined by TrainParams.
    """
    params = TrainParams(asset=asset, phase=phase, epochs=epochs, gpus=int(gpu))
    cfg = YAMLConfig.load(asset).model_params(phase)
    ds = make_dataset(
        Path(params.asset + ".feather").parent / f"{asset}.feather",
        encoder_length=params.encoder_length,
        horizon=params.horizon,
        phase=phase
    )
    model = TFTModel(
        ds,
        learning_rate=cfg.get("learning_rate", params.learning_rate),
        hidden_size=cfg.get("hidden_size", params.hidden_size),
        dropout=cfg.get("dropout", params.dropout),
        max_epochs=params.epochs
    )
    ensure_dir(params.model_path().parent)
    model.fit(gpus=params.gpus, batch_size=params.batch_size)
    model.trainer.save_checkpoint(str(params.model_path()))
    typer.echo(f"Saved TFT -> {params.model_path()}")


if __name__ == "__main__":
    app()
