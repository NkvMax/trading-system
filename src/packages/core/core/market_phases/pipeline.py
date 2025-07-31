"""
Deterministic market-phase labelling controlled by configs/<asset>.yaml.

No auto-calibration, no optimisation loops — only thresholds that you set
manually in the YAML (section ``market_phases``).

Output
------
The column `phase` is written back to data/features_full/<asset>.feather
with values:
    bull_strong, bull_weak, bear_strong, bear_weak, sideways

Additionally, separate .feather files are created for each phase in data/regimes/.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yaml

#  Paths
ROOT = Path(__file__).resolve().parents[5]
FEAT_DIR = ROOT / "data" / "features_full"
CFG_DIR = ROOT / "configs"
REGIME_DIR = ROOT / "data" / "regimes"

# Fallbacks - used if the key is not in YAML
FALLBACK = {
    "slope_days": 7,
    "slope_strong": 0.08,
    "slope_weak": 0.02,
    "vol_window": 30,
    "vol_norm_window": 90,
    "vol_high": 1.35,
}


def _load_cfg(asset: str) -> dict:
    """
    Load market-phase thresholds for a given asset:
    - Reads `<configs>/{asset}.yaml` under the `market_phases` section.
    - Merges with hard-coded FALLBACK defaults.
    - Returns a dict with keys:
        slope_days, slope_strong, slope_weak,
        vol_window, vol_norm_window, vol_high.
    """
    path = CFG_DIR / f"{asset}.yaml"
    if not path.exists():
        print(f"{path} not found — using hard-coded defaults.")
        return FALLBACK.copy()

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    mp = raw.get("market_phases", {})
    cfg = {**FALLBACK, **mp}  # fill the gaps
    return cfg


def _phase(
        slope: float,
        vol_ratio: float,
        cfg: dict,
) -> str:
    """
    Map numeric metrics -> text label.

    Rules (totally deterministic)
    ------------------------------
    • |slope|   < slope_weak      -> sideways
    • slope    ≥ slope_strong     -> bull_strong  (unless volatility is 'high')
    • slope    ≥ slope_weak       -> bull_weak    ('')

    • slope    ≤ -slope_strong    -> bear_strong  (unless volatility is 'high')
    • slope    ≤ -slope_weak      -> bear_weak    ('')
    • else                        -> sideways
    • If vol_ratio > vol_high     -> downgrade *strong* -> *weak*
    """
    strong, weak = cfg["slope_strong"], cfg["slope_weak"]
    high_vol = vol_ratio > cfg["vol_high"]

    # Flat market first
    if abs(slope) < weak:
        return "sideways"

    # Trending up
    if slope > 0:
        if slope >= strong and not high_vol:
            return "bull_strong"
        return "bull_weak"

    # Trending down
    if slope <= -strong and not high_vol:
        return "bear_strong"
    if slope <= -weak:
        return "bear_weak"

    return "sideways"


def add_phases(asset: str) -> Path:
    """
    Label each datapoint of <asset> with a market phase and save results:
    1. Load `data/features_full/{asset}.feather`, compute slope and vol_ratio.
    2. Call `_phase` row-by-row to assign one of 5 labels.
    3. Overwrite the original feather with a new 'phase' column.
    4. Split the DataFrame by phase and write out files to `data/regimes/`
       named `{asset}_{phase}_fixed.feather`.
    Returns the path to the updated feature file.
    """
    cfg = _load_cfg(asset)

    feat_path = FEAT_DIR / f"{asset}.feather"
    if not feat_path.exists():
        raise FileNotFoundError(feat_path)

    df = pd.read_feather(feat_path).sort_values("time").reset_index(drop=True)

    # Metrics
    close = df["Close"].astype("float64")
    log_ret = np.log(close).diff()

    d = cfg["slope_days"]
    slope = (close / close.shift(d) - 1).fillna(0)

    vol_short = log_ret.rolling(cfg["vol_window"]).std()
    vol_long = log_ret.rolling(cfg["vol_norm_window"]).std()
    vol_ratio = (vol_short / vol_long).replace({0: np.nan}).bfill()

    # Labels
    labels = [
        _phase(slope[i], vol_ratio[i], cfg) if i >= cfg["slope_days"] else "sideways"
        for i in range(len(df))
    ]

    df["phase"] = labels
    df.to_feather(feat_path)

    print(
        f"phase column written to {feat_path.name} "
        f"(slope_days={cfg['slope_days']}, strong={cfg['slope_strong']}, "
        f"weak={cfg['slope_weak']}, vol_high={cfg['vol_high']})"
    )

    # Add the required column asset_id
    df["asset_id"] = asset

    # Create files by phases with asset_id
    REGIME_DIR.mkdir(parents=True, exist_ok=True)
    for phase in df["phase"].unique():
        phase_df = df[df["phase"] == phase].reset_index(drop=True)
        phase_df["asset_id"] = asset # We guarantee the presence of asset_id
        out_path = REGIME_DIR / f"{asset}_{phase}_fixed.feather"
        phase_df.to_feather(out_path)
        print(f"Saved {out_path.name} shape={phase_df.shape}")

    return feat_path


#  CLI helper
if __name__ == "__main__":
    import typer

    typer.run(add_phases)
