from tft.model.hybrid_xgb_tft import HybridXGBTFT
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet


class SafeHybridXGBTFT(HybridXGBTFT):
    @staticmethod
    def _raw_df(ds: TimeSeriesDataSet) -> pd.DataFrame:
        """
        Ensure a pandas DataFrame is available from the dataset (dataframe or data attribute),
        otherwise raise a clear, descriptive error.
        """
        if hasattr(ds, "dataframe"):
            return ds.dataframe.copy()
        if isinstance(getattr(ds, "data", None), pd.DataFrame):
            return ds.data.copy()
        raise RuntimeError(
            "TimeSeriesDataSet does not contain a DataFrame (.dataframe / .data)."
        )
