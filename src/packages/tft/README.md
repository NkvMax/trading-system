# TFT Package

Implements Temporal Fusion Transformer (TFT) models and hybrid ensembles for time-series forecasting.

## Contents

- **data**:  
  `make_dataset` builds a `TimeSeriesDataSet` for training and inference.  
- **model**:  
  - `tft.py`: `TFTModel` wrapper around PyTorch Forecasting’s TemporalFusionTransformer.  
  - `inverted_tft.py`: `InvertedTFT` flips a series to train on bear-weak regimes.  
  - `hybrid_xgb_tft.py`: `HybridXGBTFT` combines XGBoost and TFT in a two-stage workflow.  
- **config**:  
  - `TrainParams`, `XGBParams`: dataclasses for default hyperparameters and model paths.  
  - `YAMLConfig`: loads asset-specific overrides from `configs/{asset}.yaml`.  
- **backtesting**:  
  - `vbt_backtest.py`: run a VectorBT backtest for one phase.  
  - `batch_backtest.py`: iterate over all phases and compile summary stats.

## Installation

```bash
cd tft
poetry install

# Train a TFT model for one phase
poetry run python -m tft.cli train-tft ETH-USD --phase bull_strong --epochs 30

# Train an inverted TFT model
poetry run python -m tft.cli train-inverted-tft ETH-USD --phase bear_weak

# Train the hybrid XGBoost+TFT model
poetry run python -m tft.cli train-xgb-tft ETH-USD --phase bull_weak

# Run backtest for one phase
poetry run python -m tft.cli backtest-tft ETH-USD

# Run batch backtest across all phases
poetry run python -m tft.cli batch-backtest-tft ETH-USD
