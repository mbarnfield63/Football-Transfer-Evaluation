"""
Assemble the transfer-level training dataset.

Season alignment:
    Transfer season_int = S  →  prior stats from games.season = S-1
                             →  selling club team quality from games.season = S-1
                             →  club spending from transfer window S-1
                             →  league spending from transfer window S (current)
                             →  inflation index from transfer window S

Run:
    uv run python -m src.features.transfer_features

Output:
    data/features/transfer_dataset.parquet
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import duckdb
import numpy as np
import pandas as pd

from src.features.engineering import POSITION_MAP
from src.features.inflation import DB_PATH

FEATURES_DIR = Path(__file__).parents[2] / "data" / "features"
OUT_PATH = FEATURES_DIR / "transfer_dataset.parquet"

SEASON_INT_MIN = 2017
SEASON_INT_MAX = 2024

LEAGUE_TIER = {
    "GB1": 1, "ES1": 2, "IT1": 3, "L1": 4, "FR1": 5,
    "NL1": 6, "PO1": 7, "TR1": 8, "BE1": 9, "SC1": 10,
}

_COMP_IDS = "'GB1','L1','ES1','IT1','FR1','PO1','NL1','BE1','SC1','TR1'"

_TRANSFER_SQL = f"""
WITH
-- qualifying transfers: seller in top-5, known fee, 2017-2024
top5_transfers AS (
    SELECT
        t.player_id,
        t.player_name,
        t.from_club_id,
        t.transfer_fee,
        t.transfer_date,
        (2000 + CAST(SPLIT_PART(t.transfer_season, '/', 1) AS INTEGER)) AS season_int,
        fc.domestic_competition_id AS from_competition_id
    FROM transfers t
    JOIN clubs fc ON fc.club_id = t.from_club_id
    WHERE t.transfer_fee > 0
      AND fc.domestic_competition_id IN ({_COMP_IDS})
      AND (2000 + CAST(SPLIT_PART(t.transfer_season, '/', 1) AS INTEGER))
          BETWEEN {SEASON_INT_MIN} AND {SEASON_INT_MAX}
),

-- static player attributes
player_attrs AS (
    SELECT
        player_id,
        country_of_citizenship AS nationality,
        date_of_birth,
        position,
        sub_position,
        height_in_cm,
        international_caps
    FROM players
),

-- prior-season aggregate stats across all top-5 competitions
prior_stats AS (
    SELECT
        a.player_id,
        g.season,
        SUM(a.goals)          AS goals,
        SUM(a.assists)        AS assists,
        SUM(a.minutes_played) AS minutes_played,
        SUM(a.yellow_cards)   AS yellow_cards,
        SUM(a.red_cards)      AS red_cards,
        COUNT(*)              AS appearances
    FROM appearances a
    JOIN games g ON g.game_id = a.game_id
    WHERE g.competition_id IN ({_COMP_IDS})
      AND g.season BETWEEN {SEASON_INT_MIN - 1} AND {SEASON_INT_MAX - 1}
    GROUP BY a.player_id, g.season
),

-- selling club's domestic goals in prior season (home + away)
team_goals AS (
    WITH home_g AS (
        SELECT home_club_id AS club_id, competition_id, season,
               SUM(home_club_goals) AS goals
        FROM games
        WHERE competition_id IN ({_COMP_IDS})
          AND season BETWEEN {SEASON_INT_MIN - 1} AND {SEASON_INT_MAX - 1}
        GROUP BY home_club_id, competition_id, season
    ),
    away_g AS (
        SELECT away_club_id AS club_id, competition_id, season,
               SUM(away_club_goals) AS goals
        FROM games
        WHERE competition_id IN ({_COMP_IDS})
          AND season BETWEEN {SEASON_INT_MIN - 1} AND {SEASON_INT_MAX - 1}
        GROUP BY away_club_id, competition_id, season
    )
    SELECT
        COALESCE(h.club_id, a.club_id)               AS club_id,
        COALESCE(h.competition_id, a.competition_id) AS competition_id,
        COALESCE(h.season, a.season)                 AS season,
        COALESCE(h.goals, 0) + COALESCE(a.goals, 0) AS team_goals_scored
    FROM home_g h
    FULL OUTER JOIN away_g a
        ON h.club_id = a.club_id
        AND h.competition_id = a.competition_id
        AND h.season = a.season
),

-- selling club's player-share totals in prior season
team_totals AS (
    SELECT
        a.player_club_id AS club_id,
        g.competition_id,
        g.season,
        SUM(a.minutes_played) AS team_total_minutes,
        SUM(a.goals)          AS team_total_goals
    FROM appearances a
    JOIN games g ON g.game_id = a.game_id
    WHERE g.competition_id IN ({_COMP_IDS})
      AND g.season BETWEEN {SEASON_INT_MIN - 1} AND {SEASON_INT_MAX - 1}
    GROUP BY a.player_club_id, g.competition_id, g.season
),

-- selling club's incoming transfer spend in prior window (S-1)
club_spending AS (
    SELECT
        to_club_id AS club_id,
        (2000 + CAST(SPLIT_PART(transfer_season, '/', 1) AS INTEGER)) AS season_int,
        SUM(transfer_fee) AS club_transfer_spending
    FROM transfers
    WHERE transfer_fee > 0
    GROUP BY to_club_id, season_int
),

-- from-league total incoming spend in current window (S)
league_spending AS (
    SELECT
        cl.domestic_competition_id AS competition_id,
        (2000 + CAST(SPLIT_PART(t.transfer_season, '/', 1) AS INTEGER)) AS season_int,
        SUM(t.transfer_fee) AS league_transfer_spending
    FROM transfers t
    JOIN clubs cl ON cl.club_id = t.to_club_id
    WHERE cl.domestic_competition_id IN ({_COMP_IDS})
      AND t.transfer_fee > 0
    GROUP BY cl.domestic_competition_id, season_int
),

-- seasonal median fee (inflation denominator)
inflation_index AS (
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
),

-- FIFA contract lookup (one row per player-season; populated by fifa_loader)
contract_data AS (
    SELECT player_id, season_int, contract_valid_until
    FROM fifa_contract_lookup
)

SELECT
    tr.player_id,
    tr.player_name,
    tr.transfer_date,
    tr.season_int,
    tr.from_competition_id,
    tr.transfer_fee,

    pa.nationality,
    pa.date_of_birth,
    pa.position,
    pa.sub_position,
    pa.height_in_cm,
    pa.international_caps,

    -- prior-season performance (across all top-5 comps)
    ps.goals,
    ps.assists,
    ps.minutes_played,
    ps.yellow_cards,
    ps.appearances,

    -- selling club context (prior season, domestic only)
    COALESCE(tg.team_goals_scored, 0)                            AS team_goals_scored,
    COALESCE(tt.team_total_minutes, 0)                           AS team_total_minutes,
    COALESCE(tt.team_total_goals, 0)                             AS team_total_goals,
    ps.minutes_played * 100.0 / NULLIF(tt.team_total_minutes, 0) AS pct_team_minutes,
    ps.goals * 100.0 / NULLIF(tt.team_total_goals, 0)            AS pct_team_goals,
    COALESCE(cs.club_transfer_spending, 0)                       AS club_transfer_spending_prior,

    -- market context (current window)
    COALESCE(ls.league_transfer_spending, 0)                     AS from_league_spending,
    ii.median_fee                                                AS inflation_median_fee,

    -- contract at time of transfer (FIFA dataset; NULL if player not matched)
    CASE
        WHEN cd.contract_valid_until IS NOT NULL
        THEN DATEDIFF('month', tr.transfer_date, MAKE_DATE(cd.contract_valid_until, 6, 30))
        ELSE NULL
    END AS contract_months_remaining_raw,
    CASE WHEN cd.contract_valid_until IS NOT NULL THEN 1 ELSE 0 END AS has_contract_data

FROM top5_transfers tr
JOIN player_attrs pa ON pa.player_id = tr.player_id
LEFT JOIN prior_stats ps
    ON ps.player_id = tr.player_id
    AND ps.season = tr.season_int - 1
LEFT JOIN team_goals tg
    ON tg.club_id = tr.from_club_id
    AND tg.competition_id = tr.from_competition_id
    AND tg.season = tr.season_int - 1
LEFT JOIN team_totals tt
    ON tt.club_id = tr.from_club_id
    AND tt.competition_id = tr.from_competition_id
    AND tt.season = tr.season_int - 1
LEFT JOIN club_spending cs
    ON cs.club_id = tr.from_club_id
    AND cs.season_int = tr.season_int - 1
LEFT JOIN league_spending ls
    ON ls.competition_id = tr.from_competition_id
    AND ls.season_int = tr.season_int
LEFT JOIN inflation_index ii
    ON ii.season_int = tr.season_int
LEFT JOIN contract_data cd
    ON cd.player_id  = tr.player_id
    AND cd.season_int = tr.season_int
WHERE ps.player_id IS NOT NULL
ORDER BY tr.season_int, tr.player_id
"""


def _add_python_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # age at start of transfer window
    df["age"] = df["season_int"] - pd.to_datetime(df["date_of_birth"]).dt.year
    df["age_sq"] = df["age"] ** 2

    # per-90 ratios (guard zero minutes)
    p90 = df["minutes_played"] / 90
    df["goals_per90"]              = df["goals"]   / p90
    df["assists_per90"]            = df["assists"] / p90
    df["goal_contributions_per90"] = (df["goals"] + df["assists"]) / p90
    df["yellows_per90"]            = df["yellow_cards"] / p90

    # position group + dummies
    df["position_group"] = (
        df["position"].map(POSITION_MAP)
        .fillna(df["sub_position"].map(POSITION_MAP))
        .fillna("UNK")
    )
    df["pos_DEF"] = (df["position_group"] == "DEF").astype(int)
    df["pos_MID"] = (df["position_group"] == "MID").astype(int)
    df["pos_FWD"] = (df["position_group"] == "FWD").astype(int)

    # nationality flag
    df["is_english"] = (df["nationality"] == "England").astype(int)

    # league tier (fixed constant — not data-driven)
    df["league_tier"] = df["from_competition_id"].map(LEAGUE_TIER)

    # targets
    df["log_transfer_fee"] = np.log(df["transfer_fee"])
    df["log_inflation_adjusted_fee"] = np.log(
        df["transfer_fee"] / df["inflation_median_fee"]
    )

    # contract feature — clip to [0, 60] months; NaN where scraper hasn't run
    df["contract_months_remaining"] = df["contract_months_remaining_raw"].clip(lower=0, upper=60)

    return df


def build_transfer_dataset(
    db_path: Path = DB_PATH,
    out_path: Path = OUT_PATH,
) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. "
            "Run: uv run python -m src.scraping.tm_scraper"
        )

    con = duckdb.connect(str(db_path))
    # Ensure FIFA contract lookup table exists so the CTE works even before fifa_loader runs.
    # If empty, contract columns will be all-NaN (XGBoost handles this natively).
    con.execute("""
        CREATE TABLE IF NOT EXISTS fifa_contract_lookup (
            player_id            INTEGER NOT NULL,
            season_int           INTEGER NOT NULL,
            contract_valid_until INTEGER NOT NULL,
            match_score          REAL,
            fifa_player_name     VARCHAR,
            PRIMARY KEY (player_id, season_int)
        )
    """)
    df = con.execute(_TRANSFER_SQL).fetchdf()
    con.close()

    df = _add_python_features(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building transfer dataset...")
    df = build_transfer_dataset()

    print(f"\nShape: {df.shape}")
    print(f"Season range: {df['season_int'].min()} – {df['season_int'].max()}")
    print(f"Rows by season:\n{df['season_int'].value_counts().sort_index().to_string()}")

    print(f"\nTarget — log_inflation_adjusted_fee:")
    print(df["log_inflation_adjusted_fee"].describe().round(3).to_string())

    print(f"\nNull check (key columns):")
    key_cols = [
        "goals_per90", "pct_team_minutes", "pct_team_goals",
        "team_goals_scored", "from_league_spending", "league_tier",
        "log_inflation_adjusted_fee",
    ]
    for col in key_cols:
        nulls = df[col].isna().sum()
        print(f"  {col}: {nulls} nulls")

    print(f"\nPosition group counts:\n{df['position_group'].value_counts().to_string()}")

    print(f"\nCorrelation with log_inflation_adjusted_fee (top 10):")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    corr = df[num_cols].corr()["log_inflation_adjusted_fee"].drop("log_inflation_adjusted_fee")
    corr = corr.drop("log_transfer_fee", errors="ignore")
    print(corr.abs().sort_values(ascending=False).head(10).round(3).to_string())

    print(f"\nSample (5 largest transfers):")
    sample_cols = [
        "player_name", "season_int", "from_competition_id", "position_group",
        "age", "goals_per90", "transfer_fee", "log_inflation_adjusted_fee",
    ]
    print(
        df.nlargest(5, "transfer_fee")[sample_cols]
        .to_string(index=False)
    )

    print(f"\nSaved to {OUT_PATH}")
