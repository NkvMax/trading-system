# Phase‑Adaptive ML Trading System for Ethereum (ETH)

> A modular trading pipeline that automatically detects the current market regime (bull, bear, sideways) and switches to
> a specialised machine‑learning model optimised for that phase. Built with Python3.11, PyTorch Lightning, XGBoost and
> rich CLI tooling.

---


---

## Overview

Cryptocurrency markets transition through distinct regimes: exuberant bull runs, choppy sideways ranges, and brutal bear
declines. A single model often fails to perform consistently across these very different dynamics.

This project addresses that challenge by:

- **Automatically labelling market phases** using deterministic, forward‑secure rules based on 7‑day price slope and
  volatility.

- **Training five specialised models**—one for each phase (Bull_Strong, Bull_Weak, Sideways, Bear_Weak, Bear_Strong).

- **Switching models on‑the‑fly** during back‑tests or live trading, so the strategy always uses the most appropriate
  predictor.

The result is a robust system that captures upside during rallies while protecting capital in downturns.

---

## Key features

| Feature                    | Description                                                                                                                                    |
|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| **Phase classifier**       | Labels every day since2018 into one of5 regimes via YAML‑defined thresholds (no look‑ahead).                                                   |
| **Model zoo**              | TemporalFusionTransformer (TFT), hybrid **XGBoost+TFT** ensemble, and an _inverted_ TFT for gentle bear markets.                               |
| **~50 technical features** | Moving averages, RSI, MACD, ATR, Bollinger Bands, Donchian channels, ADX, and more.                                                            |
| **Rich CLI**               | Single entry‑point (`python -m tsroot.cli`) to run the full pipeline:download -> feature_engineering -> phase labelling -> train -> back‑test. |
| **Back‑tester**            | Walk‑forward simulation with realistic model hand‑off, PnL accounting and benchmark buy‑&‑hold comparison.                                     |
| **Modular codebase**       | Easily plug in new assets, features or models.                                                                                                 |

---

## Market‑phase logic

```yaml
# config/ETH-USD.yml (excerpt)
base_window: 30          # minimum phase duration, days
slope_period: 7          # price delta, days
thresholds:
  bull_strong: 0.08      # +8% and more in 7 days
  bull_weak: 0.02        # +2% ... +8%
  bear_weak: -0.02       # –2% ... –8%
  bear_strong: -0.08     # below -8%
volatility_filter: 0.015 # >1.5% avg. ATR -> calculate "sideways" phase
```

The rules are **fixed prior to training** to avoid data leakage. You can adjust thresholds globally in the YAML.

---

## Models

| Phase       | Model                   | Notes                                          |
|-------------|-------------------------|------------------------------------------------|
| Bull_Strong | **TFT**                 | Captures long‑horizon upward momentum.         |
| Bull_Weak   | **30% XGBoost+70% TFT** | Ensemble balances noise and trend.             |
| Sideways    | **TFT**                 | Learns short oscillatory patterns.             |
| Bear_Weak   | **Inverted TFT**        | Train on inverted prices, then flip sign.      |
| Bear_Strong | **TFT**                 | Mirrors bull‑strong logic for fast downtrends. |

All deep models are implemented with **PyTorch Lightning 2.x**; tree ensembles use **XGBoost 1.7+**.

---

### Per-Phase Performance Summary

| Phase       | Model                         | Total Return [%] | Max DD [%] | Sharpe   | Buy & Hold Return [%] | Buy & Hold DD [%] | Buy & Hold Sharpe |
|-------------|-------------------------------|------------------|------------|----------|-----------------------|-------------------|-------------------|
| bull_strong | **TFT**                       | **1354.45**      | **38.06**  | **3.33** | 1292.09               | 74.81             | 2.56              |
| bull_weak   | Hybrid: 70% TFT + 30% XGBoost | –                | –          | –        | –                     | –                 | –                 |
| sideways    | **TFT**                       | **146.14**       | 40.10      | 1.91     | –                     | –                 | –                 |
| bear_weak   | Inverted TFT                  | –                | –          | –        | –                     | –                 | –                 |
| bear_strong | **TFT**                       | **639.20**       | 41.74      | **2.36** | –                     | –                 | –                 |

---

## Project structure

```
trading-system-root/
├── tsroot/                # main package
│   ├── cli.py             # CLI entry point
│   ├── data/              # loading and caching prices
│   ├── features/          # generation of 50 technical features
│   ├── phase_classifier.py
│   ├── models/
│   │   ├── tft_model.py
│   │   ├── hybrid_model.py
│   │   └── …
│   ├── backtest.py        # backtesting engine
│   └── utils/
├── scripts/               # high-level startup scripts
├── tests/                 # pytest‑tests
├── config/phase_rules.yml
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-handle/trading-system.git
cd trading-system

# 2. Install poetry if not installed
pip install poetry  # or: pipx install poetry

# 3. Create and activate the virtual environment
poetry install

# 4. Activate shell (optional, or use poetry run ...)
poetry shell
```

## Quick start

```bash
# 1. Download raw historical data (e.g. from Yahoo Finance)
poetry run python -m tsroot.cli build-dataset ETH-USD --start 2018-01-01

# 2. Generate ~50 technical features
poetry run python -m tsroot.cli make-features ETH-USD

# 3. Label market regimes using deterministic rules
poetry run python -m tsroot.cli market-phases ETH-USD

# 4. Train all models for each market phase
poetry run python -m tsroot.cli train-all ETH-USD --epochs 30

# 5. Run backtest with automatic model switching by phase
poetry run python -m tsroot.cli backtest-tft ETH-USD

# 6. (Optional) Generate summary table across all regimes
poetry run python -m tsroot.cli batch-backtest-tft ETH-USD

```

---

## Training / retraining models

Each phase‑specific model can be retrained individually:

```bash
# Example: Retrain only the Bull Strong model
tsroot.cli train --phase bull_strong --epochs 50 --save-checkpoint
```

Hyper‑parameters (learning rate, hidden dims, ensemble weights) live in `config/model_params.yml` and can be overridden
via CLI.

---

## License

Distributed under the **MIT License**. See `LICENSE` for details.

---

> ©2025 Nikolaev Maxim. Built with ❤️ and a lot of caffeine.