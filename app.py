from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic.dataclasses import dataclass
from pyfedappwrap.learning.app_types import BasePrePostProcessApp
from pyfedappwrap.learning.run_runfig import AppConfig, AppInputConfig, AppOutputConfig
from typing_extensions import override


@dataclass
class OneHotEncodingAppConfig(AppConfig):
    columns: str = "-"  # "-" oder "" => auto
    drop_first: bool = False
    dummy_na: bool = False
    prefix_sep: str = "__"
    bool_to_number: bool = True


@dataclass
class OneHotEncodingAppInputConfig(AppInputConfig):
    features: Any = None


@dataclass
class OneHotEncodingAppOutputConfig(AppOutputConfig):
    encoded: Any | None = None


class OneHotEncodingAPP(
    BasePrePostProcessApp[
        OneHotEncodingAppConfig,
        OneHotEncodingAppInputConfig,
        OneHotEncodingAppOutputConfig,
    ]
):
    def _get_columns_to_encode(self, df: pd.DataFrame) -> list[str]:
        raw = self.config.columns

        if raw in (None, "", "-"):
            cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            if not self.config.bool_to_number:
                cols += df.select_dtypes(include=["bool"]).columns.tolist()
            return cols

        cols = [c.strip() for c in str(raw).split(",") if c.strip()]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Requested columns not found in input: {missing}")
        return cols

    def run_process(self, data: OneHotEncodingAppInputConfig) -> OneHotEncodingAppOutputConfig:
        df = data.features
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'features' must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Input dataframe is empty.")

        cols_to_encode = self._get_columns_to_encode(df)

        encoded_df = pd.get_dummies(
            df,
            columns=cols_to_encode,
            drop_first=self.config.drop_first,
            dummy_na=self.config.dummy_na,
            prefix_sep=self.config.prefix_sep,
        )

        if self.config.bool_to_number:
            encoded_df = encoded_df*1

        return OneHotEncodingAppOutputConfig(encoded=encoded_df)

    @override
    def get_test_data(self) -> Optional[dict[str, Path]]:
        return {"features": Path("data/iris.csv")}

    @override
    def get_test_hyperparams(self) -> Optional[dict[str, Any]]:
        return {"columns": "-"}
