import numpy as np
import pandas as pd


def add_lag_features(df: pd.DataFrame, stat_cols: list[str], lags: int = 1) -> pd.DataFrame:
    raise NotImplementedError


def add_per90_ratios(df: pd.DataFrame, count_cols: list[str], minutes_col: str = "minutes") -> pd.DataFrame:
    raise NotImplementedError


def encode_position_groups(df: pd.DataFrame, position_col: str = "position") -> pd.DataFrame:
    raise NotImplementedError


def log_transform_valuation(df: pd.DataFrame, valuation_col: str = "market_value_eur") -> pd.DataFrame:
    df = df.copy()
    df["log_valuation"] = np.log(df[valuation_col].clip(lower=1e3))
    return df
