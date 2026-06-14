import pandas as pd


def compute_relative_value_score(
    df: pd.DataFrame,
    valuation_col: str = "market_value_eur",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Divide each valuation by the median for its position group × season."""
    if group_cols is None:
        group_cols = ["position_group", "season"]
    df = df.copy()
    medians = df.groupby(group_cols)[valuation_col].transform("median")
    df["relative_value_score"] = df[valuation_col] / medians.clip(lower=1)
    return df
