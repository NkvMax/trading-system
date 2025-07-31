from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd
import yaml

from .registry import FEATURE_REGISTRY
from .lag import add_lags, add_rolling_stats

PROJECT_ROOT = Path(__file__).resolve().parents[5]
CONFIG_DIR = PROJECT_ROOT / "configs"

PARAM_TO_COLUMN = {
    "close": "Close",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "volume": "Volume",
}


def _ensure_series(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError("Expected 1-column DataFrame, got " f"{obj.shape[1]} columns.")
        return obj.iloc[:, 0].astype("float32")
    elif isinstance(obj, pd.Series):
        return obj.astype("float32")
    else:
        return obj


class FeatureBuilder:
    FEATURE_COLUMNS: List[str] = []

    def __init__(
            self,
            raw_df: pd.DataFrame,
            *,
            asset: str,
            cfg_path: str | Path | None = None,
    ) -> None:
        """
        Initialize with raw OHLCV DataFrame and load the feature-engineering config
        (indicators, lags, rolling windows) from the YAML file for <asset>.
        """
        self.asset = asset
        self.df = raw_df[["Open", "High", "Low", "Close", "Volume"]].astype("float32")

        if cfg_path is None:
            cfg_path = CONFIG_DIR / f"{asset}.yaml"

        cfg_path = Path(cfg_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")

        with cfg_path.open("r", encoding="utf-8") as fh:
            full_cfg: Dict = yaml.safe_load(fh) or {}
        self.cfg = full_cfg.get("feature_engineering", {})

        self.parts: List[pd.DataFrame] = []

    def build(self) -> pd.DataFrame:
        """
        Execute the full feature pipeline:
          1) Add low-level indicators via FEATURE_REGISTRY.
          2) Add lagged versions of selected columns.
          3) Add rolling-window statistics.
        Concatenate results, drop rows with NaNs, reset index, and return final DataFrame.
        """
        self.parts.clear()
        self._add_indicators()
        self._add_lags()
        self._add_rolling()

        feats = pd.concat(self.parts, axis=1).astype("float32").dropna()
        final_df = pd.concat([self.df.loc[feats.index], feats], axis=1)
        final_df.index.name = "time"
        final_df.reset_index(inplace=True)

        FeatureBuilder.FEATURE_COLUMNS = feats.columns.tolist()
        return final_df

    def _add_indicators(self) -> None:
        """
        Iterate over configured indicator specs, call the corresponding function
        from FEATURE_REGISTRY, and append each result as a DataFrame.
        """
        for spec in self.cfg.get("indicators", []):
            spec = spec.copy()
            name: str = spec.pop("name")
            func = FEATURE_REGISTRY[name]
            sig = inspect.signature(func)
            params = list(sig.parameters)

            pos_args: list = spec.pop("args", [])
            kw_args: Dict = spec.pop("params", {}) | spec

            if {"close", "series"} & set(params):
                key = "close" if "close" in params else "series"
                kw_args.setdefault(key, self.df["Close"])

            if "high" in params and "high" not in kw_args and len(pos_args) <= params.index("high"):
                kw_args["high"] = self.df["High"]
            if "low" in params and "low" not in kw_args and len(pos_args) <= params.index("low"):
                kw_args["low"] = self.df["Low"]

            if params[:2] == ["high", "low"] and not pos_args:
                pos_args = [kw_args.pop("high"), kw_args.pop("low")]

            pos_args = [_ensure_series(a) for a in pos_args]
            kw_args = {k: _ensure_series(v) for k, v in kw_args.items()}

            feat = func(*pos_args, **kw_args)
            self.parts.append(feat if isinstance(feat, pd.DataFrame) else feat.to_frame())

    def _add_lags(self) -> None:
        """
        For each lag spec in the config, compute lagged columns on the combined DataFrame
        (raw + current features) and append to parts.
        """
        for lg in self.cfg.get("lags", []):
            col, periods = lg["column"], lg["periods"]
            merged = pd.concat([self.df] + self.parts, axis=1)
            if col not in merged.columns:
                print(f"[FeatureBuilder] skip lag for '{col}' – column not found")
                continue
            self.parts.append(add_lags(merged, col, periods))

    def _add_rolling(self) -> None:
        """
        For each rolling-spec in the config, compute rolling statistics (mean, std, etc.)
        on the raw DataFrame and append to parts.
        """
        for rl in self.cfg.get("rolling", []):
            self.parts.append(
                add_rolling_stats(
                    self.df,
                    rl["column"],
                    rl["windows"],
                    rl["stats"],
                )
            )
