import numpy as np
from tft.model.tft import TFTModel
from pytorch_forecasting import TimeSeriesDataSet


class InvertedTFT:
    def __init__(self, dataset, **tft_kwargs):
        """
        Invert the target series (multiply by -1), reconstruct the TimeSeriesDataSet,
        and initialize the underlying TFT model.
        """
        df = dataset.dataframe.copy()
        target_col = dataset.target
        df[target_col] = -df[target_col]

        # We will restore the missing features
        required_cols = (
                dataset.static_categoricals +
                dataset.static_reals +
                dataset.time_varying_known_categoricals +
                dataset.time_varying_known_reals +
                dataset.time_varying_unknown_categoricals +
                dataset.time_varying_unknown_reals
        )
        for col in required_cols:
            if col not in df.columns:
                if col == 'log_return':
                    df[col] = np.log(df['Close']).diff().fillna(0.0)
                else:
                    df[col] = 0

        time_idx_name = dataset.time_idx if isinstance(dataset.time_idx, str) else "time_idx"
        if time_idx_name not in df.columns:
            df[time_idx_name] = np.arange(len(df))
        df[time_idx_name] = df[time_idx_name].astype(int)
        group_ids = dataset.group_ids if isinstance(dataset.group_ids, list) else [dataset.group_ids]

        # Remove ALL protected columns from pytorch-forecasting!
        # - Manual: most frequent
        PROTECTED_COLUMNS = [
            "relative_time_idx", "target_scale", "encoder_length"
        ]
        # - Dynamically: all <target>_center and <target>_scale
        for col in list(df.columns):
            if any([
                col in PROTECTED_COLUMNS,
                col.endswith("_center"),
                col.endswith("_scale")
            ]):
                df = df.drop(columns=[col])

        # Now we create a dataset
        inverted_ds = TimeSeriesDataSet(
            df,
            time_idx=time_idx_name,
            target=target_col,
            group_ids=group_ids,
            max_encoder_length=dataset.max_encoder_length,
            max_prediction_length=dataset.max_prediction_length,
            static_categoricals=dataset.static_categoricals,
            static_reals=dataset.static_reals,
            time_varying_known_categoricals=dataset.time_varying_known_categoricals,
            time_varying_known_reals=dataset.time_varying_known_reals,
            time_varying_unknown_categoricals=dataset.time_varying_unknown_categoricals,
            time_varying_unknown_reals=dataset.time_varying_unknown_reals,
            add_relative_time_idx=dataset.add_relative_time_idx,
            add_target_scales=dataset.add_target_scales,
            add_encoder_length=dataset.add_encoder_length,
            allow_missing_timesteps=dataset.allow_missing_timesteps,
        )

        self.model = TFTModel(inverted_ds, **tft_kwargs)
        self.inverted = True
        self.dataset = inverted_ds

    def fit(self, *args, **kwargs):
        """
        Train the underlying TFT model on the inverted data.
        """
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """
        Ensure a pandas DataFrame is available from the dataset (dataframe or data attribute),
        otherwise raise a clear, descriptive error.
        """
        pred = self.model.predict(*args, **kwargs)
        return -pred
