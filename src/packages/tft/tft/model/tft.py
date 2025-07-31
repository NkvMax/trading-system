from __future__ import annotations
import torch
from pytorch_lightning import Trainer
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
from pathlib import Path


class TFTModel:
    def __init__(
            self,
            dataset,
            learning_rate: float = 1e-3,
            hidden_size: int = 64,
            dropout: float = 0.1,
            max_epochs: int = 30,
            **kwargs
    ):
        """
        Initialize the TFTModel wrapper:
        - Store the TimeSeriesDataSet and hyperparameters for TemporalFusionTransformer.
        - Prepare placeholders for the model and trainer.
        """
        self.dataset = dataset
        self.hparams = dict(
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            dropout=dropout,
            **kwargs
        )
        self.max_epochs = max_epochs
        self.model: TemporalFusionTransformer | None = None
        self.trainer: Trainer | None = None

    def fit(self, gpus: int = 0, batch_size: int = 64, callbacks: list = None):
        """
        Train the TFT:
        1. Create a TemporalFusionTransformer from the dataset with QuantileLoss.
        2. Configure a PyTorch Lightning Trainer (GPU or CPU).
        3. Convert dataset into train/validation DataLoaders.
        4. Run trainer.fit(model, train_loader, val_loader).
        Returns self after training.
        """
        self.model = TemporalFusionTransformer.from_dataset(
            self.dataset,
            loss=QuantileLoss(),
            **self.hparams
        )
        self.trainer = Trainer(
            max_epochs=self.max_epochs,
            accelerator="gpu" if gpus else "cpu",
            devices=gpus or 1,
            enable_checkpointing=False,
            logger=False,
            callbacks=callbacks or [],
        )
        train_loader = self.dataset.to_dataloader(train=True, batch_size=batch_size)
        val_loader = self.dataset.to_dataloader(train=False, batch_size=batch_size)
        self.trainer.fit(self.model, train_loader, val_loader)
        return self

    def predict(self, dataset=None):
        """
        Generate forecasts:
        - Use the trained TemporalFusionTransformer to predict on the provided dataset
          or the original dataset if none is passed.
        - Returns model predictions in ‘prediction’ mode.
        """
        ds = dataset or self.dataset
        return self.model.predict(ds, mode="prediction")


def load_tft(dataset, checkpoint_path: str | Path):
    """
    Load a pre-trained TFTModel:
    - Accepts either a full checkpoint (.ckpt) or a state_dict file (.pt).
    - For .ckpt: uses load_from_checkpoint.
    - For .pt: builds a new model instance then loads state_dict.
    - Raises ValueError for unsupported file extensions.
    """
    ckpt = Path(checkpoint_path)

    wrapper = TFTModel(dataset)

    if ckpt.suffix == ".ckpt":
        wrapper.model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
    elif ckpt.suffix == ".pt":
        state = torch.load(ckpt, map_location="cpu")
        wrapper.model = TemporalFusionTransformer.from_dataset(dataset)
        wrapper.model.load_state_dict(state)
    else:
        raise ValueError(f"Unsupported file type: {ckpt.suffix}")

    return wrapper
