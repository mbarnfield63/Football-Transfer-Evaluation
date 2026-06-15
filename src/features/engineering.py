"""
Build the feature matrix from data/transfermarkt.duckdb.

Run:
    uv run python -m src.features.engineering

Output:
    data/features/feature_matrix.parquet  — one row per (player_id, season)
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parents[2] / "data" / "transfermarkt.duckdb"
FEATURES_DIR = Path(__file__).parents[2] / "data" / "features"

TOP5_COMPETITION_IDS = ("GB1", "L1", "ES1", "IT1", "FR1")

SEASON_MIN = 2017
SEASON_MAX = 2025
MIN_MINUTES = 90  # drop player-seasons below this to avoid noise


# ---------------------------------------------------------------------------
# SQL: step 1 — per-player-season stats from appearances + games
# ---------------------------------------------------------------------------

_TOP5_SQL = ', '.join(f"'{c}'" for c in TOP5_COMPETITION_IDS)

_STATS_SQL = f"""
WITH season_stats AS (
    SELECT
        a.player_id,
        g.season,
        a.competition_id,
        SUM(a.goals)           AS goals,
        SUM(a.assists)         AS assists,
        SUM(a.minutes_played)  AS minutes_played,
        SUM(a.yellow_cards)    AS yellow_cards,
        SUM(a.red_cards)       AS red_cards,
        COUNT(*)               AS appearances
    FROM appearances a
    JOIN games g ON g.game_id = a.game_id
    WHERE
        a.competition_id IN ({_TOP5_SQL})
        AND g.season BETWEEN {SEASON_MIN} AND {SEASON_MAX}
        AND a.minutes_played > 0
    GROUP BY a.player_id, g.season, a.competition_id
),

player_attrs AS (
    SELECT
        player_id,
        name                                    AS player_name,
        country_of_citizenship                  AS nationality,
        date_of_birth,
        position,
        sub_position,
        height_in_cm,
        foot,
        international_caps,
        current_club_domestic_competition_id    AS current_competition_id
    FROM players
),

-- end-of-season valuation: last valuation update in each Jul–Jun window
season_valuations AS (
    SELECT
        player_id,
        CASE
            WHEN MONTH(date) >= 7 THEN YEAR(date)
            ELSE YEAR(date) - 1
        END                    AS season,
        LAST(market_value_in_eur ORDER BY date) AS market_value_in_eur
    FROM player_valuations
    WHERE market_value_in_eur IS NOT NULL
    GROUP BY player_id, season
),

-- primary club per player-season: most minutes wins (handles Jan movers)
primary_club AS (
    SELECT a.player_id, g.competition_id, g.season, a.player_club_id AS primary_club_id
    FROM appearances a
    JOIN games g ON g.game_id = a.game_id
    WHERE g.competition_id IN ({_TOP5_SQL})
      AND g.season BETWEEN {SEASON_MIN - 1} AND {SEASON_MAX}
    GROUP BY a.player_id, g.competition_id, g.season, a.player_club_id
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY a.player_id, g.competition_id, g.season
        ORDER BY SUM(a.minutes_played) DESC
    ) = 1
),

-- club-season total goals (home + away combined)
team_goals AS (
    WITH home_g AS (
        SELECT home_club_id AS club_id, competition_id, season,
               SUM(home_club_goals) AS goals
        FROM games
        WHERE competition_id IN ({_TOP5_SQL})
          AND season BETWEEN {SEASON_MIN - 1} AND {SEASON_MAX}
        GROUP BY home_club_id, competition_id, season
    ),
    away_g AS (
        SELECT away_club_id AS club_id, competition_id, season,
               SUM(away_club_goals) AS goals
        FROM games
        WHERE competition_id IN ({_TOP5_SQL})
          AND season BETWEEN {SEASON_MIN - 1} AND {SEASON_MAX}
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

-- club-season player-share totals (for pct_team_minutes / pct_team_goals)
team_totals AS (
    SELECT
        a.player_club_id  AS club_id,
        g.competition_id,
        g.season,
        SUM(a.minutes_played) AS team_total_minutes,
        SUM(a.goals)          AS team_total_goals,
        SUM(a.assists)        AS team_total_assists
    FROM appearances a
    JOIN games g ON g.game_id = a.game_id
    WHERE g.competition_id IN ({_TOP5_SQL})
      AND g.season BETWEEN {SEASON_MIN - 1} AND {SEASON_MAX}
    GROUP BY a.player_club_id, g.competition_id, g.season
),

-- club incoming transfer spend per season
club_spending AS (
    SELECT
        to_club_id AS club_id,
        (2000 + CAST(SPLIT_PART(transfer_season, '/', 1) AS INTEGER)) AS season,
        SUM(transfer_fee) AS club_transfer_spending
    FROM transfers
    WHERE transfer_fee > 0
    GROUP BY to_club_id, season
),

-- league incoming transfer spend per season
league_spending AS (
    SELECT
        cl.domestic_competition_id AS competition_id,
        (2000 + CAST(SPLIT_PART(t.transfer_season, '/', 1) AS INTEGER)) AS season,
        SUM(t.transfer_fee) AS league_transfer_spending
    FROM transfers t
    JOIN clubs cl ON cl.club_id = t.to_club_id
    WHERE cl.domestic_competition_id IN ({_TOP5_SQL})
      AND t.transfer_fee > 0
    GROUP BY cl.domestic_competition_id, season
)

SELECT
    s.player_id,
    s.season,
    s.competition_id,
    pa.player_name,
    pa.nationality,
    pa.position,
    pa.sub_position,
    pa.height_in_cm,
    pa.foot,
    pa.international_caps,

    s.season - YEAR(pa.date_of_birth) AS age,

    s.appearances,
    s.goals,
    s.assists,
    s.minutes_played,
    s.yellow_cards,
    s.red_cards,

    pc.primary_club_id,
    COALESCE(tg.team_goals_scored, 0)                            AS team_goals_scored,
    COALESCE(tt.team_total_minutes, 0)                           AS team_total_minutes,
    COALESCE(tt.team_total_goals, 0)                             AS team_total_goals,
    s.minutes_played * 100.0 / NULLIF(tt.team_total_minutes, 0) AS pct_team_minutes,
    s.goals * 100.0 / NULLIF(tt.team_total_goals, 0)            AS pct_team_goals,
    COALESCE(cs.club_transfer_spending, 0)                       AS club_transfer_spending,
    COALESCE(ls.league_transfer_spending, 0)                     AS league_transfer_spending,

    sv.market_value_in_eur
FROM season_stats s
JOIN player_attrs pa ON pa.player_id = s.player_id
LEFT JOIN season_valuations sv
    ON sv.player_id = s.player_id AND sv.season = s.season
LEFT JOIN primary_club pc
    ON pc.player_id = s.player_id
    AND pc.competition_id = s.competition_id
    AND pc.season = s.season
LEFT JOIN team_goals tg
    ON tg.club_id = pc.primary_club_id
    AND tg.competition_id = s.competition_id
    AND tg.season = s.season
LEFT JOIN team_totals tt
    ON tt.club_id = pc.primary_club_id
    AND tt.competition_id = s.competition_id
    AND tt.season = s.season
LEFT JOIN club_spending cs
    ON cs.club_id = pc.primary_club_id
    AND cs.season = s.season
LEFT JOIN league_spending ls
    ON ls.competition_id = s.competition_id
    AND ls.season = s.season
WHERE s.minutes_played >= {MIN_MINUTES}
ORDER BY s.player_id, s.season
"""


# ---------------------------------------------------------------------------
# Python: step 2 — derived features on top of the SQL result
# ---------------------------------------------------------------------------

POSITION_MAP = {
    "Goalkeeper":  "GK",
    "Defender":    "DEF",
    "Midfield":    "MID",
    "Attack":      "FWD",
    # sub-positions that arrive without a parent position
    "Centre-Forward":   "FWD",
    "Left Winger":      "FWD",
    "Right Winger":     "FWD",
    "Second Striker":   "FWD",
    "Attacking Midfield": "MID",
    "Central Midfield": "MID",
    "Defensive Midfield": "MID",
    "Left Midfield":    "MID",
    "Right Midfield":   "MID",
    "Centre-Back":      "DEF",
    "Left-Back":        "DEF",
    "Right-Back":       "DEF",
}


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # per-90 ratios (avoid division by zero)
    p90 = df["minutes_played"] / 90
    df["goals_per90"]   = df["goals"]   / p90
    df["assists_per90"] = df["assists"] / p90
    df["goal_contributions_per90"] = (df["goals"] + df["assists"]) / p90
    df["yellows_per90"] = df["yellow_cards"] / p90

    # position group
    df["position_group"] = (
        df["position"].map(POSITION_MAP)
        .fillna(df["sub_position"].map(POSITION_MAP))
        .fillna("UNK")
    )

    # age squared for non-linear age curve
    df["age_sq"] = df["age"] ** 2

    # English player flag (for english_tax hypothesis)
    df["is_english"] = (df["nationality"] == "England").astype(int)

    # log valuation target (only for rows with a known valuation)
    mv = pd.to_numeric(df["market_value_in_eur"], errors="coerce").astype(float)
    df["log_valuation"] = np.log(mv.where(mv > 0))

    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 1-season lag of log_valuation per player."""
    df = df.sort_values(["player_id", "season"])
    df["lag1_log_valuation"] = (
        df.groupby("player_id")["log_valuation"].shift(1)
    )
    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_feature_matrix(db_path: Path = DB_PATH, out_dir: Path = FEATURES_DIR) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. "
            "Run: uv run python -m src.scraping.tm_scraper"
        )

    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute(_STATS_SQL).fetchdf()
    con.close()

    df = _add_derived_features(df)
    df = _add_lag_features(df)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feature_matrix.parquet"
    df.to_parquet(out_path, index=False)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building feature matrix...")
    df = build_feature_matrix()

    print(f"\nShape: {df.shape}")
    print(f"\nColumns:\n{list(df.columns)}")
    print(f"\nPosition group counts:\n{df['position_group'].value_counts().to_string()}")
    print(f"\nSeason range: {df['season'].min()} – {df['season'].max()}")
    print(f"\nRows with log_valuation: {df['log_valuation'].notna().sum()} / {len(df)}")
    print(f"\nRows with lag1_log_valuation: {df['lag1_log_valuation'].notna().sum()} / {len(df)}")

    print(f"\nSample (5 rows):")
    sample_cols = [
        "player_name", "season", "competition_id", "position_group",
        "age", "goals_per90", "assists_per90", "minutes_played",
        "is_english", "log_valuation", "lag1_log_valuation",
    ]
    print(df[sample_cols].dropna(subset=["log_valuation"]).head(5).to_string())
    print(f"\nSaved to data/features/feature_matrix.parquet")
