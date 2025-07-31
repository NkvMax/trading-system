from __future__ import annotations

"""CLI utility for generating engineered features from raw OHLC candles.

Launch:
poetry run python -m packages.core.core.feature_engineering.make_features ETH-USD \
       --src data/raw \
       --out data/features_full

The file preserves **all** original OHLCV columns + explicit `time`,
so that downstream pipelines can sort and join without dancing.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .feature_builder import FeatureBuilder

app = typer.Typer(help="Generate engineered features and save to Feather")


@app.command()
def main(
        asset: str = typer.Argument(..., help="Ticker, e.g. ETH-USD"),
        src: Path = typer.Option(Path("data/raw"), help="Directory with <asset>.parquet"),
        out: Path = typer.Option(Path("data/features_full"), help="Output directory"),
        cfg: Optional[Path] = typer.Option(
            None, help="Explicit YAML config; auto‑detected otherwise"
        ),
        force: bool = typer.Option(False, help="Overwrite existing file if present"),
) -> None:
    """Collect features for *asset* and save them in Feather.
    If the file already exists and `--force` is not specified, the utility will exit silently.
    """

    src_file = src / f"{asset}.parquet"
    if not src_file.exists():
        typer.secho(f"raw file not found: {src_file}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"{asset}.feather"
    if out_file.exists() and not force:
        typer.secho(
            f"features already exist: {out_file} (use --force to overwrite)",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(0)

    raw_df = pd.read_parquet(src_file)
    # SIMPLE fix for multi-index
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = [col[0] for col in raw_df.columns]

    builder = FeatureBuilder(raw_df, asset=asset, cfg_path=cfg)
    feat_df = builder.build()
    feat_df.to_feather(out_file)

    typer.secho(f"features saved -> {out_file}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
