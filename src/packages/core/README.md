# Core Package

Provides data ingestion, feature engineering, and market-phase labeling pipelines.

## Contents

- **data_preprocessing**:  
  Download and clean raw OHLCV price data into `data/raw/{asset}.feather`.  
- **feature_engineering**:  
  Generate ~50 technical indicators and rolling/window features into `data/features_full/{asset}.feather`.  
- **market_phases**:  
  Label each row with one of five regimes (`bull_strong`, `bull_weak`, `sideways`, `bear_weak`, `bear_strong`) and write per-phase files to `data/regimes/{asset}_{phase}_fixed.feather`.

## Installation

```bash
cd core
poetry install

# Download raw price data
poetry run python -m packages.core.core.data_preprocessing.pipelines BTC-USD --start 2018-01-01

# Generate technical features
poetry run python -m packages.core.core.feature_engineering.make_features BTC-USD \
  --src data/raw --out data/features_full

# Label market phases
poetry run python -m packages.core.core.market_phases.pipeline BTC-USD
