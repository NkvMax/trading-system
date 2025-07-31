"""
Abstract interface for any OHLCV data loader used in the project.

All concrete loaders (Yahoo Finance, Binance, CSV, ...) must implement
`download()` with the same signature so they can be swapped freely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class BaseLoader(ABC):
    @abstractmethod
    def download(
            self,
            symbol: str,
            *,
            start: Optional[str] = None,
            end: Optional[str] = None,
            interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Return raw OHLCV dataframe indexed by UTC DatetimeIndex.

        Parameters
        ----------
        symbol : str
            Ticker, e.g. `"ETH-USD"`.
        start : str | None
            ISO date (`YYYY-MM-DD`). If `None` we’ll fetch the full history
            available at the data provider.
        end : str | None
            ISO date. `None` -> today.
        interval : str
            Candle resolution supported by the provider (`"1d"`, `"1h"`...).
        """
        raise NotImplementedError
