import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


CONTROL_FEATURES = [
    "age", "age_sq", "position_group", "log_valuation_lag1",
    "goals_per90", "assists_per90", "xg_per90", "progressive_carries_per90",
    "league", "club_league_position",
]


def ols_english_tax(df: pd.DataFrame) -> smf.OLS:
    """Fit OLS with is_english indicator; coefficient gives log-point premium."""
    raise NotImplementedError


def partial_dependence_nationality(model, X: pd.DataFrame) -> None:
    raise NotImplementedError


def residual_nationality_plot(df: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series) -> None:
    raise NotImplementedError
