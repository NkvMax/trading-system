import typer
from pathlib import Path
from tft.config import TrainParams, YAMLConfig
from tft.data import make_dataset
from tft.model.inverted_tft import InvertedTFT
from tft.utils import ensure_dir

app = typer.Typer()


@app.command()
def train_inverted_tft(
        asset: str = typer.Argument(...),
        phase: str | None = typer.Option(None),
        epochs: int = typer.Option(30),
        gpu: bool = typer.Option(False),
):
    """
    CLI command to train an Inverted TFT model:
    1. Builds dataset with past <encoder_length> inputs and <horizon> targets.
    2. Wraps dataset in InvertedTFT (price series flipped).
    3. Fits the model for <epochs> epochs on CPU or GPU.
    4. Saves checkpoint with `_inv.ckpt` suffix.
    """
    tp = TrainParams(asset=asset, phase=phase, epochs=epochs, gpus=int(gpu))
    ds = make_dataset(
        Path(tp.asset + ".feather").parent / f"{asset}.feather",
        encoder_length=tp.encoder_length,
        horizon=tp.horizon,
        phase=phase
    )
    inv = InvertedTFT(ds, learning_rate=tp.learning_rate,
                      hidden_size=tp.hidden_size, dropout=tp.dropout,
                      max_epochs=tp.epochs)
    inv.fit(gpus=tp.gpus, batch_size=tp.batch_size)
    path = tp.model_path().with_name(tp.model_path().stem + "_inv.ckpt")
    ensure_dir(path.parent)
    inv.inner.trainer.save_checkpoint(str(path))
    typer.echo(f"Saved Inverted TFT -> {path}")


if __name__ == "__main__":
    app()
