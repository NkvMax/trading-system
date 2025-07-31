from __future__ import annotations

"""Data validation helper built on Great Expectations.

This module exposes a single :class: `GEValidator` that can be dropped into any
ETL / preprocessing step.  It loads an Expectation Suite from a JSON file and
runs it against a Pandas `DataFrame`.  On failure it raises `ValueError` so
that upstream orchestrators (Airflow / Prefect / custom CLI) can halt the
pipeline early.

Usage:
- validator = GEValidator("./expectations/ohlcv_suite.json")
- validator.validate(df)

Parameters are intentionally minimal; logging is delegated to the standard
`logging` module so that the host application decides where logs go.
"""

import json
import logging
from pathlib import Path
from typing import Any, Union

import pandas as pd
import great_expectations as gx
from great_expectations.core import ExpectationSuite
from great_expectations.dataset import PandasDataset

__all__ = ["GEValidator"]

logger = logging.getLogger(__name__)


class GEValidator:
    """Validate a DataFrame against a *Great Expectations* JSON suite.

    Parameters
    ----------
    expectation_path : str | Path
        Path to a JSON *Expectation Suite* exported from GX (v0.17+).
    """

    def __init__(self, expectation_path: Union[str, Path]):
        self.expectation_path = Path(expectation_path)
        if not self.expectation_path.exists():
            raise FileNotFoundError(
                f"Expectation file '{self.expectation_path}' does not exist."
            )
        self.suite: ExpectationSuite = self._load_suite()

    # public helpers
    def validate(
        self, df: pd.DataFrame, *, raise_on_failure: bool = True, **kwargs: Any
    ) -> dict[str, Any]:
        """Run validation.

        Parameters
        ----------
        df : pd.DataFrame
            Target dataframe.
        raise_on_failure : bool, default `True`
            Whether to raise `ValueError` when *any* expectation fails.
        **kwargs
            Passed straight to `dataset.validate` — e.g. `result_format`.

        Returns
        -------
        dict
            Validation result dictionary produced by Great Expectations.
        """

        dataset = PandasDataset(df.copy())
        logger.debug("Running Great Expectations validation: %s", self.suite.expectation_suite_name)
        result: dict[str, Any] = dataset.validate(
            expectation_suite=self.suite, result_format="SUMMARY", **kwargs
        )

        if not result.get("success", False):
            msg = "Great Expectations validation failed."
            if raise_on_failure:
                logger.error("%s Result: %s", msg, result)
                raise ValueError(msg)
            logger.warning(msg)
        else:
            logger.info("Validation succeeded: %s", self.suite.expectation_suite_name)

        return result

    # internal
    def _load_suite(self) -> ExpectationSuite:
        """Deserialize *Expectation Suite* from JSON."""
        logger.debug("Loading expectation suite from %s", self.expectation_path)
        with self.expectation_path.open("r", encoding="utf-8") as fp:
            suite_dict = json.load(fp)

        # Both signatures exist across GX versions; try modern API first.
        try:
            return ExpectationSuite(**suite_dict)  # type: ignore[arg-type]
        except TypeError:  # fallback for older GX
            return ExpectationSuite.from_json_dict(suite_dict)
