"""
Map (slope, volatility) -> 5-phase market regime.

* bull_strong
* bull_weak
* sideways
* bear_weak
* bear_strong
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

# helpers
LABELS = ("bull_strong", "bull_weak", "sideways", "bear_weak", "bear_strong")


def _load_thresholds(asset: str) -> Dict:
    """
    Read `market_phases` section from configs/<asset>.yaml.
    If the YAML contains `phase3_overrides` - they are overlaid.
    """
    cfg_path = Path(__file__).resolve().parents[3] / "config" / f"{asset}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)["market_phases"]

    overrides = cfg.get("phase3_overrides", {})
    return {**cfg, **overrides}


# public API
def assign_regime(
        slope: pd.Series,
        vol_ratio: pd.Series,
        *,
        asset: str,
) -> pd.Series:
    """
    Convert numeric (slope, vol_ratio) into categorical market regime.

    Parameters
    ----------
    slope : pd.Series
        N-day % change (already pre-computed).
    vol_ratio : pd.Series
        Normalized rolling volatility.
    asset : str
        Ticker, e.g. ``"ETH-USD"`` – used to locate YAML thresholds.

    Returns
    -------
    pd.Series[Categorical] of shape (len(slope), )
        One of LABELS for every timestamp.
    """
    thr = _load_thresholds(asset)

    strong, weak, high = (
        thr["slope_strong"],
        thr["slope_weak"],
        thr["vol_high"],
    )

    cond_bs1 = (slope >= strong) & (vol_ratio < high)
    cond_bw = (slope >= weak) & ~cond_bs1
    cond_sw = slope.abs() < weak
    cond_bw2 = (slope <= -weak) & (slope > -strong)
    cond_bs2 = (slope <= -strong) & (vol_ratio >= high)

    regime = pd.Series("sideways", index=slope.index, dtype="object")
    regime.loc[cond_bs1] = "bull_strong"
    regime.loc[cond_bw] = "bull_weak"
    regime.loc[cond_bw2] = "bear_weak"
    regime.loc[cond_bs2] = "bear_strong"

    return regime.astype(pd.CategoricalDtype(categories=LABELS, ordered=False))
