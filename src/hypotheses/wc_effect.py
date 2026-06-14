import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

WC_MINUTES_THRESHOLD = 300


def build_did_dataset(
    df: pd.DataFrame,
    wc_year: int,
    wc_minutes_col: str = "wc_minutes",
) -> pd.DataFrame:
    """Construct pre/post panel with treatment indicator for DiD analysis."""
    raise NotImplementedError


def did_estimate(df: pd.DataFrame) -> smf.OLS:
    """
    DiD regression:
        log_valuation ~ post + treated + post*treated + controls
    Coefficient on post*treated is the WC valuation boost estimate.
    """
    raise NotImplementedError
