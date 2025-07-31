"""Dataset builder for raw OHLCV data (daily candles).

Why this script exists
----------------------
* Keep the *download logic* in one place – no more duplicated yfinance snippets.
* Load asset‑specific defaults (date range, storage path, etc.) from `configs/{asset}.yaml`.
* Allow power‑users to override everything through CLI flags.

Typical usage
-------------
```bash
poetry run python -m core.data_preprocessing.pipelines BTC-USD \
    --start 2020-01-01 --end 2023-12-31

# or: grab the *entire* history, letting yfinance decide the earliest date
poetry run python -m core.data_preprocessing.pipelines ETH-USD
```
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
import yaml
import pandas as pd
import yfinance as yf

from .ohlcv import clean_ohlcv  # basic forward‑fill & dtype optimisation

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[5]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"

# CLI
app = typer.Typer(add_help_option=True, help="Download, clean and store raw OHLCV data as Parquet.")


# Helper
def _load_config(symbol: str) -> dict:
    """Read `configs/{symbol}.yaml` and return a dict. Exit with code 1 if missing."""
    cfg_path = CONFIG_DIR / f"{symbol}.yaml"
    if not cfg_path.exists():
        typer.secho(f"[ERROR] Config not found: {cfg_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# Core function
def build_dataset(
        symbol: str,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        output_dir: Optional[Path] = None,
) -> Path:
    """Download daily candles for *symbol* and save them as Parquet.

    Parameters
    ----------
    symbol: str
        Ticker like ``BTC-USD`` or ``ETH-USD``.
    start / end: str | None
        ISO‑8601 dates (``YYYY-MM-DD``). If *both* are ``None``, the entire
        history is fetched via ``period="max"``.
    output_dir: Path | None
        Target directory. When omitted, value from YAML config is used; if the
        config also omits it – fallback to ``data/raw``.
    Returns
    -------
    Path to the written Parquet file.
    """

    # Load YAML configuration
    cfg = _load_config(symbol)
    cfg_data = cfg.get("data", {})

    cfg_start = cfg_data.get("start")
    cfg_end = cfg_data.get("end")
    cfg_out = Path(cfg_data.get("output_dir", DEFAULT_OUTPUT_DIR))

    start_date: Optional[str] = start or cfg_start  # CLI overrides YAML
    end_date: Optional[str] = end or cfg_end

    # Download via yfinance
    yf_kwargs = dict(interval="1d", auto_adjust=False)
    if start_date is None and end_date is None:
        # Fetch "all" available history.
        hist = yf.download(symbol, period="max", **yf_kwargs)
    else:
        hist = yf.download(symbol, start=start_date, end=end_date, **yf_kwargs)

    if hist.empty:
        typer.secho("[ERROR] Download returned empty DataFrame.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    # Clean & store
    hist = clean_ohlcv(hist)

    out_dir = output_dir or cfg_out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}.parquet"
    hist.to_parquet(out_path)

    return out_path


# CLI entry point
@app.command()
def main(
        symbol: str = typer.Argument(..., help="Ticker, e.g. ETH-USD"),
        start: Optional[str] = typer.Option(None, help="Start date YYYY-MM-DD. Omit to download full history."),
        end: Optional[str] = typer.Option(None, help="End date YYYY-MM-DD."),
        output_dir: Optional[Path] = typer.Option(None, help="Custom output directory."),
) -> None:
    """CLI wrapper around :func:`build_dataset`."""
    out_path = build_dataset(symbol, start=start, end=end, output_dir=output_dir)
    typer.secho(f"Saved {out_path}", fg=typer.colors.GREEN)


if __name__ == "__main__":  # pragma: no cover
    # Allow running as `python pipelines.py BTC-USD`
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        # Direct call without Typer runner
        app()  # type: ignore[arg-type]
    else:
        app()
