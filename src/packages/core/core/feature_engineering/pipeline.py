"""
5-regime market phase detector (bull/bear – strong/weak + sideways).

Thresholds are loaded from `configs/<ASSET>.yaml` so you can fine-tune
them per symbol.

Output:
---
1. Column `phase` appended to the main features feather.
2. Two-column feather (date, phase) saved under `data/market_phases/`.

CLI:
---
poetry run python -m core.market_phases.pipeline \
       --asset ETH-USD \
       --src   data/features_full \
       --out   data/market_phases
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
import typer

app = typer.Typer(help="Tag each bar with market regime")


def _load_cfg(asset: str, cfg_path: Optional[Path] = None) -> dict:
    """
    Load YAML configuration for the given asset.

    If cfg_path is not provided, looks under `<repo_root>/config/{asset}.yaml`.
    Exits the CLI with error if the file is missing.
    """
    if cfg_path is None:
        cfg_path = Path(__file__).resolve().parents[3] / "config" / f"{asset}.yaml"
    if not cfg_path.exists():
        typer.echo(f"config not found: {cfg_path}")
        raise typer.Exit(1)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _classify(row, thr):
    """
    Classify a single DataFrame row into one of 5 market regimes based on:
      - row['slope'] vs. thr['slope_strong'] / thr['slope_weak']
      - row['vol_norm'] for volatility-based sideways detection

    Returns an integer code:
      0: strong up, 1: weak up, 2: sideways, 3: weak down, 4: strong down
    """
    slope = row["slope"]
    vol = row["vol_norm"]

    if slope >= thr["slope_strong"]:
        return 0  # strong up
    if thr["slope_weak"] <= slope < thr["slope_strong"]:
        return 1  # weak  up
    if -thr["slope_weak"] <= slope <= thr["slope_weak"]:
        return 2  # sideways
    if -thr["slope_strong"] < slope < -thr["slope_weak"]:
        return 3  # weak  down
    return 4  # strong down


@app.command()
def main(
        asset: str = typer.Argument(..., help="Ticker, e.g. ETH-USD"),
        src: Path = typer.Option(Path("data/features_full"), help="Where <asset>.feather lives"),
        out: Path = typer.Option(Path("data/market_phases"), help="Output directory"),
        cfg: Path | None = typer.Option(None, help="Explicit YAML; auto-detected otherwise"),
        force: bool = typer.Option(False, help="Overwrite existing file"),
) -> None:
    """
    Tag each data point of <asset> with a market regime label:
      1. Load features from `src/{asset}.feather`.
      2. Compute slope and normalized volatility.
      3. Apply `_classify` using thresholds from the YAML config.
      4. Save full feather with 'phase' column and a compact feather in `out/`.
    """
    src_file = src / f"{asset}.feather"
    if not src_file.exists():
        typer.echo(f"features file not found: {src_file}")
        raise typer.Exit(1)

    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"{asset}.feather"
    if out_file.exists() and not force:
        typer.echo(f"phases already exist: {out_file} (use --force to overwrite)")
        raise typer.Exit(0)

    cfg_dct = _load_cfg(asset, cfg)
    thr = cfg_dct["market_phases"]

    df = pd.read_feather(src_file)

    # pre-compute slope & normalized volatility
    df["slope"] = df["Close"].pct_change(thr["slope_days"])
    df["vol_norm"] = (
            df["Close"].pct_change().rolling(thr["vol_window"]).std()
            / df["Close"].pct_change().rolling(thr["vol_norm_window"]).std()
    )

    # allow phase-specific overrides
    overrides = thr.get("phase3_overrides", {})
    thr_effective = {**thr, **overrides}

    df["phase"] = df.apply(_classify, axis=1, thr=thr_effective).astype("int8")
    df[["Date", "phase"]].to_feather(out_file)

    typer.echo(f"phases saved -> {out_file}")


if __name__ == "__main__":
    app()
