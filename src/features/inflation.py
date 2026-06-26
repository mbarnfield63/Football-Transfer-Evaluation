"""
Relative value scores — remove market inflation and position-group effects.

Adds two columns to the feature matrix:
  relative_value_score  — market_value_in_eur / median(market_value_in_eur)
                          within position_group × season
  log_relative_value    — log_valuation - median(log_valuation)
                          within position_group × season
                          (equivalent to log of relative_value_score; use this
                          as the model target when comparing across seasons)

Run:
    uv run python -m src.features.inflation
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parents[2] / "data" / "transfermarkt.duckdb"
FEATURES_DIR = Path(__file__).parents[2] / "data" / "features"
PARQUET_PATH = FEATURES_DIR / "feature_matrix.parquet"

_GROUP_COLS = ["position_group", "season"]


def add_relative_value(
    df: pd.DataFrame,
    group_cols: list[str] = _GROUP_COLS,
) -> pd.DataFrame:
    """
    Return df with two new columns: relative_value_score and log_relative_value.

    Rows where market_value_in_eur is null are assigned NaN for both columns.
    """
    df = df.copy()

    # raw ratio: player value / group-season median
    mv = df["market_value_in_eur"].astype(float)
    median_mv = mv.groupby([df[c] for c in group_cols]).transform("median")
    df["relative_value_score"] = mv / median_mv.clip(lower=1)

    # log-space equivalent: log(player) - median(log)
    lv = df["log_valuation"].astype(float)
    median_lv = lv.groupby([df[c] for c in group_cols]).transform("median")
    df["log_relative_value"] = lv - median_lv

    return df


# ---------------------------------------------------------------------------
# Transfer fee inflation index
# ---------------------------------------------------------------------------

_COMP_IDS = "'GB1','L1','ES1','IT1','FR1','PO1','NL1','BE1','SC1','TR1'"

_FEE_INDEX_SQL = f"""
SELECT
    (2000 + CAST(SPLIT_PART(t.transfer_season, '/', 1) AS INTEGER)) AS season_int,
    MEDIAN(t.transfer_fee) AS median_fee
FROM transfers t
JOIN clubs fc ON fc.club_id = t.from_club_id
JOIN clubs tc ON tc.club_id = t.to_club_id
WHERE t.transfer_fee > 0
  AND (fc.domestic_competition_id IN ({_COMP_IDS})
       OR tc.domestic_competition_id IN ({_COMP_IDS}))
GROUP BY season_int
ORDER BY season_int
"""


def build_fee_inflation_index(con: duckdb.DuckDBPyConnection) -> dict[int, float]:
    """Median top-5 transfer fee per season. Returns {{season_int: median_fee}}."""
    rows = con.execute(_FEE_INDEX_SQL).fetchall()
    return {int(season): float(median) for season, median in rows}


def inflation_adjust_fee(fee: float, season_int: int, index: dict[int, float]) -> float:
    """Return fee / index[season_int]."""
    return fee / index[season_int]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {PARQUET_PATH}. "
            "Run: uv run python -m src.features.engineering"
        )

    df = pd.read_parquet(PARQUET_PATH)
    df = add_relative_value(df)
    df.to_parquet(PARQUET_PATH, index=False)

    print(f"Rows: {len(df)}")
    print(f"\nRelative value score — describe (rows with valuation):")
    print(df.loc[df["relative_value_score"].notna(), "relative_value_score"].describe().round(3).to_string())
    print(f"\nLog relative value — describe:")
    print(df.loc[df["log_relative_value"].notna(), "log_relative_value"].describe().round(3).to_string())

    print(f"\nSample (position_group=FWD, season=2023, sorted by relative_value_score desc):")
    sample = (
        df.loc[(df["position_group"] == "FWD") & (df["season"] == 2023)]
        .dropna(subset=["relative_value_score"])
        .sort_values("relative_value_score", ascending=False)
        [["player_name", "competition_id", "goals_per90", "market_value_in_eur",
          "relative_value_score", "log_relative_value"]]
        .head(8)
    )
    print(sample.to_string(index=False))
    print(f"\nUpdated {PARQUET_PATH}")
