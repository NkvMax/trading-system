from __future__ import annotations

import glob
import os
import subprocess
from pathlib import Path
from typing import List

import typer

app = typer.Typer(help="End-to-end pipeline runner")

# paths / helpers
BASE = Path(__file__).resolve().parents[2]
SRC = BASE / "src"


def _run(cmd: List[str], rel_cwd: str = "src") -> None:
    """
    Helper: spawn subprocess with PYTHONPATH set to our src/
    so that all `packages.*` imports resolve correctly.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC}:{env.get('PYTHONPATH', '')}"
    subprocess.run(cmd, cwd=BASE / rel_cwd, env=env, check=True)


def _existing_phases(asset: str) -> list[str]:
    """
    Helper: detect which market phases have both:
      1) a saved TFT checkpoint under models/tft/{asset}_<phase>.pt
      2) a fixed regime file under data/regimes/{asset}_<phase>_fixed.feather

    Raises if none found (i.e. run market-phases + train-tft first).
    """
    model_phases = {
        Path(p).stem.replace(f"{asset}_", "", 1)
        for p in glob.glob(str(BASE / f"models/tft/{asset}_*.pt"))
    }
    data_phases = {
        Path(p).stem.replace(f"{asset}_", "", 1).replace("_fixed", "", 1)
        for p in glob.glob(str(BASE / f"data/regimes/{asset}_*_fixed.feather"))
    }
    common = sorted(model_phases & data_phases)
    if not common:
        raise RuntimeError(
            f"No matching phases for {asset}. Run market-phases + train-tft first."
        )
    return common


# data stage
@app.command("build-dataset")
def build_dataset(
        symbol: str = typer.Argument(..., help="Ticker, e.g. BTC-USD"),
        start: str = typer.Option("2018-01-01", help="YYYY-MM-DD"),
) -> None:
    """
    CLI: download raw price data for <symbol> since <start>,
    and save to data/raw via core.data_preprocessing.pipelines.
    """
    _run([
        "poetry", "run", "python", "-m",
        "packages.core.core.data_preprocessing.pipelines",
        symbol, "--start", start
    ])


@app.command("make-features")
def make_features(
        asset: str = typer.Argument(..., help="Ticker, e.g. BTC-USD"),
        src: Path = typer.Option(Path("data/raw")),
        out: Path = typer.Option(Path("data/features_full")),
) -> None:
    """
    CLI: generate ~50 technical indicators for <asset>,
    reading from <src> and writing to <out>.
    """
    _run([
        "poetry", "run", "python", "-m",
        "packages.core.core.feature_engineering.make_features",
        asset,
        "--src", str((BASE / src).resolve()),
        "--out", str((BASE / out).resolve())
    ])


@app.command("market-phases")
def market_phases(asset: str = typer.Argument(...)) -> None:
    """
    CLI: label each day of <asset> as one of 5 regimes
    using the core.market_phases.pipeline module.
    """
    _run([
        "poetry", "run", "python", "-m",
        "packages.core.core.market_phases.pipeline",
        asset
    ])


# training
@app.command("train-tft")
def train_tft(
        asset: str = typer.Argument(...),
        epochs: int = typer.Option(20),
        gpu: bool = typer.Option(False),
) -> None:
    """
    CLI: train TFT model on <asset> for <epochs>.
    Pass --gpu to enable GPU training.
    """
    _run([
        "poetry", "run", "tft-train",
        asset, "--epochs", str(epochs), *(["--gpu"] if gpu else [])
    ], rel_cwd="src/packages/tft")


@app.command("train-all")
def train_all(
        asset: str = typer.Argument(...),
        epochs: int = typer.Option(30),
) -> None:
    """
    CLI: train all phase-specific TFT models for <asset>
    (delegates to tft-train-all script).
    """
    _run([
        "poetry", "run", "tft-train-all", asset, "--epochs", str(epochs)
    ], rel_cwd="src/packages/tft")


# back-tests
@app.command("backtest-tft")
def backtest_tft(
        asset: str = typer.Argument(...),
        phase: str | None = typer.Option(None),
        threshold: float = typer.Option(0.005),
) -> None:
    """
    CLI: for each (or a single) <phase>, run vectorbt backtest
    on the TFT model using threshold filter.
    """
    phases = [phase] if phase else _existing_phases(asset)
    for ph in phases:
        print(f"\n=== Back-test phase: {ph} ===")
        try:
            _run([
                "poetry", "run", "python", "-m",
                "tft.backtesting.vbt_backtest",
                asset, "--phase", ph, "--threshold", str(threshold)
            ], rel_cwd="src/packages/tft")
        except Exception as exc:
            print(f"skipped {ph}: {exc}")


@app.command("batch-backtest-tft")
def batch_backtest_tft(
        asset: str = typer.Argument(...),
        threshold: float = typer.Option(0.005),
) -> None:
    """
    CLI: run backtests for all existing phases,
    collect stats into a Pandas summary and print.
    """
    from packages.tft.tft.backtesting.vbt_backtest import run_vbt_backtest_for_phase
    import pandas as pd

    results = {}
    for ph in _existing_phases(asset):
        print(f"\n=== Back-test phase: {ph} ===")
        try:
            res = run_vbt_backtest_for_phase(asset, ph, threshold)
            print(res["stats"])
            results[ph] = res
        except Exception as exc:
            print(f"skipped {ph}: {exc}")

    if results:
        summary = pd.concat([r["stats"].iloc[[0]] for r in results.values()])
        print("\n====== SUMMARY ======")
        print(summary.to_string())


if __name__ == "__main__":
    app()
