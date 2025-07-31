from __future__ import annotations
import pandas as pd
from tft.backtesting.vbt_backtest import run_vbt_backtest_for_phase

PHASES = ["bull_strong", "bull_weak", "sideways", "bear_weak", "bear_strong"]


def batch_backtest(
        *,
        asset: str = "ETH-USD",
        threshold: float = 0.005,
        verbose: bool = True,
) -> dict:
    """
    Run backtests across all predefined market phases for a given asset.

    Parameters:
      asset (str): Ticker symbol, e.g. "ETH-USD".
      threshold (float): Minimum expected return threshold for signal generation.
      verbose (bool): If True, print progress and per-phase stats.

    Returns:
      dict: {
        "summary": DataFrame of stats per phase,
        "entry_counts": dict of number of trades per phase,
        "results": raw result dict per phase
      }
    """
    summary, entry_counts, results = {}, {}, {}

    for ph in PHASES:
        try:
            if verbose:
                print(f"\n=== Back-test {asset} / {ph} ===")
            res = run_vbt_backtest_for_phase(asset=asset, phase=ph, threshold=threshold)
        except FileNotFoundError as e:
            if verbose:
                print(f"[SKIP] {e}")
            continue

        results[ph] = res
        summary[ph] = res["stats"].iloc[:, 0]  # TFT-metrics
        entry_counts[ph] = len(res["pf_tft"].trades) if hasattr(res["pf_tft"], "trades") else 0

        if verbose:
            print(res["stats"].round(2))
            print("-" * 40)

    df_summary = pd.DataFrame(summary).T
    if verbose:
        print("\n==== FULL SUMMARY ====")
        print(df_summary.round(3))
        print("\nMarket transactions:", entry_counts)

    return {"summary": df_summary, "entry_counts": entry_counts, "results": results}


if __name__ == "__main__":
    import typer

    typer.run(lambda asset="ETH-USD", thr=0.005: batch_backtest(asset=asset, threshold=thr))
