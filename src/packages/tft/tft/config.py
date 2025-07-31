from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CFG_DIR = REPO_ROOT / "configs"
MODELS_DIR = REPO_ROOT / "models" / "tft"
DATA_DIR = REPO_ROOT / "data" / "features_full"


@dataclass
class TrainParams:
    """
    Holds default training parameters and paths for TFT models.
    """
    asset: str
    phase: str | None = None
    encoder_length: int = 90
    horizon: int = 7
    epochs: int = 30
    learning_rate: float = 1e-3
    hidden_size: int = 64
    dropout: float = 0.1
    batch_size: int = 64
    gpus: int = 0

    def model_path(self) -> Path:
        """
        Construct the filesystem path where the trained checkpoint will be saved.
        Format: models/tft/{asset}_{phase or 'all'}.ckpt
        """
        name = f"{self.asset}_{self.phase or 'all'}.ckpt"
        return MODELS_DIR / name


@dataclass
class XGBParams:
    """
    Holds default XGBoost parameters and paths for hybrid models.
    """
    asset: str
    phase: str | None = None
    params: dict = field(default_factory=lambda: {"n_estimators": 100, "max_depth": 3})

    def model_path(self) -> Path:
        """
        Construct the filesystem path for saving the XGBoost model JSON.
        Format: models/xgb/{asset}_{phase or 'all'}.json
        """
        name = f"{self.asset}_{self.phase or 'all'}.json"
        return MODELS_DIR.parent / "xgb" / name


@dataclass
class YAMLConfig:
    """
    Loads and provides access to optional YAML configuration for each asset.
    """
    raw: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, asset: str) -> "YAMLConfig":
        """
        Read `<configs>/{asset}.yaml` if it exists, otherwise return empty config.
        """
        path = CFG_DIR / f"{asset}.yaml"
        if not path.exists():
            return cls({})
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(data)

    def model_params(self, phase: str | None) -> dict:
        """
        Retrieve model-specific hyperparameters for the given phase.
        Falls back to 'all' if no phase-specific section is defined.
        """
        mp = self.raw.get("model_params", {})
        return mp.get(phase or "all", {})
