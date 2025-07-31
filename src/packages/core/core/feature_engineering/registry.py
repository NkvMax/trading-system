"""
Single FEATURE_REGISTRY: name -> callable, kwargs by default.

We add new indicators in one line - and they immediately
available in FeatureBuilder and YAML config.
"""
from __future__ import annotations
from typing import Callable, Any, Dict
from . import indicators as I

FEATURE_REGISTRY: Dict[str, Callable[..., Any]] = {
    # sliding
    "sma": I.sma,
    "ema": I.ema,
    # price bands / channels
    "bollinger": I.bollinger,
    "donchian": I.donchian,
    # oscillators
    "macd": I.macd,
    "rsi": I.rsi,
    "williams_r": I.williams_r,
    "stoch": I.stoch,
    "cci": I.cci,
    "adx": I.adx,
    # volatility / range
    "atr": I.atr,
    # momentum
    "roc": I.roc,
    "momentum": I.momentum,
}
