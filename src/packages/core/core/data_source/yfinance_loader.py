"""
Yahoo Finance implementation of :class:`BaseLoader`.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import yfinance as yf

from .base import BaseLoader


class YFinanceLoader(BaseLoader):
    def download(
            self,
            symbol: str,
            *,
            start: Optional[str] = None,
            end: Optional[str] = None,
            interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Download OHLCV data from Yahoo Finance.

        If *start* is `None` the entire available history is requested
        via `period=\"max\"`. This is useful when the user does not specify
        the date range on the CLI.
        """
        if start is None:
            raw = yf.download(
                tickers=symbol,
                period="max",
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="column",
            )
        else:
            raw = yf.download(
                tickers=symbol,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="column",
            )

        # Yahoo returns MultiIndex columns for multiple tickers
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)

        raw = raw.rename(columns=str.capitalize)
        raw.index = pd.to_datetime(raw.index, utc=True)

        return raw[["Open", "High", "Low", "Close", "Volume"]]
