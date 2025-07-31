"""
Collects the FINAL dataset:
    1) read features_full/<asset>.feather
    2) clear NaN/inf, ffill/bfill
    3) select numeric columns
    4) save to features_final/<asset>.feather
CLI:
    poetry run tradingbot-make-features --asset ETH-USD
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_features(
        asset: str,
        features_full_dir: str | Path = "data/features_full",
        out_dir: str | Path = "data/features_final",
) -> Path:
    """
    Read the full feature set for <asset>, clean NaN/inf values (ffill -> bfill -> drop),
    select numeric columns, and save the cleaned DataFrame to the final features directory.
    Returns the path to the saved .feather file.
    """
    src = Path(features_full_dir) / f"{asset}.feather"
    if not src.exists():
        raise FileNotFoundError(f"[build_features] нет файла {src}\n"
                                "First run tradingbot-pipeline")

    df = pd.read_feather(src).set_index("time")

    # we leave only numerical signs
    num_df = df.select_dtypes(include=["number"]).astype("float32")

    # clean inf и NaN (ffill -> bfill -> drop)
    num_df = (num_df.replace([np.inf, -np.inf], np.nan)
              .ffill()
              .bfill()
              .dropna())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{asset}.feather"
    num_df.reset_index().to_feather(dst)
    return dst


# CLI
def _cli() -> None:
    p = argparse.ArgumentParser(prog="tradingbot-make-features")
    p.add_argument("--asset", default="ETH-USD")
    p.add_argument("--src", default="data/features_full")
    p.add_argument("--out", default="data/features_final")
    args = p.parse_args()
    fpath = build_features(args.asset, args.src, args.out)
    print("saved ->", fpath)


if __name__ == "__main__":
    _cli()
