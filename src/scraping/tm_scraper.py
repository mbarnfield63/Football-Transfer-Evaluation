"""
Transfermarkt data via the davidcariboo/player-scores Kaggle dataset.

Setup (one-time):
  1. kaggle.com → Account → Create New Token → download kaggle.json
  2. Place it at C:/Users/<you>/.kaggle/kaggle.json
  3. Run: python -m src.scraping.tm_scraper   (downloads + builds DB)

After that, scrape_valuations() / scrape_transfers() query the local DuckDB.
"""

import zipfile
from pathlib import Path

import duckdb
import pandas as pd

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "transfermarkt"
DB_PATH = Path(__file__).parents[2] / "data" / "transfermarkt.duckdb"

KAGGLE_DATASET = "davidcariboo/player-scores"

# Top-5 league competition IDs as used by Transfermarkt / this dataset
COMPETITION_IDS = {
    "ENG-Premier League":       "GB1",
    "GER-Bundesliga":           "L1",
    "ESP-La Liga":              "ES1",
    "ITA-Serie A":              "IT1",
    "FRA-Ligue 1":              "FR1",
    "POR-Liga Portugal":        "PO1",
    "NED-Eredivisie":           "NL1",
    "BEL-Jupiler Pro League":   "BE1",
    "SCO-Scottish Premiership": "SC1",
    "TUR-Süper Lig":            "TR1",
}

LEAGUES = list(COMPETITION_IDS.keys())
SEASONS = list(range(2017, 2026))


# ---------------------------------------------------------------------------
# Step 1: download CSVs from Kaggle
# ---------------------------------------------------------------------------

def download_dataset(dest: Path = RAW_DIR) -> None:
    """Download and unzip the Kaggle dataset into dest/."""
    import kaggle  # noqa: PLC0415 — imported here so missing creds give a clear error

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} → {dest}")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(KAGGLE_DATASET, path=str(dest), unzip=True)
    print("Download complete.")


# ---------------------------------------------------------------------------
# Step 2: load CSVs into DuckDB
# ---------------------------------------------------------------------------

def build_db(csv_dir: Path = RAW_DIR, db_path: Path = DB_PATH) -> None:
    """
    Read the downloaded CSVs into a persistent DuckDB database.

    Creates one table per CSV file, named after the file stem.
    Safe to re-run — each table is replaced.
    """
    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSVs found in {csv_dir}. Run download_dataset() first."
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    for csv_path in sorted(csv_files):
        table = csv_path.stem
        print(f"  Loading {csv_path.name} → table '{table}'")
        con.execute(f"""
            CREATE OR REPLACE TABLE {table} AS
            SELECT * FROM read_csv_auto(
                '{csv_path.as_posix()}',
                header=True,
                sample_size=-1
            )
        """)

    con.close()
    print(f"Database built at {db_path}")
    _print_schema(db_path)


def _print_schema(db_path: Path) -> None:
    con = duckdb.connect(str(db_path), read_only=True)
    tables = con.execute("SHOW TABLES").fetchdf()
    for t in tables["name"]:
        cols = con.execute(f"PRAGMA table_info('{t}')").fetchdf()
        print(f"\n[{t}]")
        print(cols[["name", "type"]].to_string(index=False))
    con.close()


# ---------------------------------------------------------------------------
# Step 3: query helpers
# ---------------------------------------------------------------------------

def _connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. "
            "Run: python -m src.scraping.tm_scraper"
        )
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def scrape_valuations(leagues: list[str] = LEAGUES, seasons: list[int] = SEASONS) -> pd.DataFrame:
    """
    Return one row per (player, valuation date) for the given leagues/seasons.

    Key columns: player_id, player_name, competition_id, season, club_id,
                 market_value_in_eur, date, nationality, position, age
    """
    comp_ids = [COMPETITION_IDS[l] for l in leagues if l in COMPETITION_IDS]
    comp_ids_sql = ", ".join(f"'{c}'" for c in comp_ids)

    season_min, season_max = min(seasons), max(seasons)

    query = f"""
        SELECT
            pv.player_id,
            p.name                  AS player_name,
            p.current_club_domestic_competition_id AS competition_id,
            p.position,
            p.country_of_citizenship AS nationality,
            pv.date,
            -- derive season year from valuation date (season = year of Aug–Jun window)
            CASE
                WHEN MONTH(pv.date) >= 7 THEN YEAR(pv.date)
                ELSE YEAR(pv.date) - 1
            END                     AS season,
            pv.market_value_in_eur,
            pv.current_club_id      AS club_id
        FROM player_valuations pv
        JOIN players p ON p.player_id = pv.player_id
        WHERE
            p.current_club_domestic_competition_id IN ({comp_ids_sql})
            AND CASE
                    WHEN MONTH(pv.date) >= 7 THEN YEAR(pv.date)
                    ELSE YEAR(pv.date) - 1
                END BETWEEN {season_min} AND {season_max}
            AND pv.market_value_in_eur IS NOT NULL
        ORDER BY pv.player_id, pv.date
    """

    con = _connect()
    df = con.execute(query).fetchdf()
    con.close()
    return df


def scrape_transfers(leagues: list[str] = LEAGUES, seasons: list[int] = SEASONS) -> pd.DataFrame:
    """
    Return one row per transfer involving a club in the target leagues/seasons.

    Key columns: player_id, player_name, season, transfer_fee,
                 from_club_id, to_club_id, from_competition_id, to_competition_id
    """
    comp_ids = [COMPETITION_IDS[l] for l in leagues if l in COMPETITION_IDS]
    comp_ids_sql = ", ".join(f"'{c}'" for c in comp_ids)

    season_min, season_max = min(seasons), max(seasons)

    query = f"""
        SELECT
            t.player_id,
            t.player_name,
            t.transfer_season                      AS season,
            t.transfer_fee,
            t.market_value_in_eur,
            t.from_club_id,
            t.to_club_id,
            fc.domestic_competition_id             AS from_competition_id,
            tc.domestic_competition_id             AS to_competition_id
        FROM transfers t
        JOIN clubs fc ON fc.club_id = t.from_club_id
        JOIN clubs tc ON tc.club_id = t.to_club_id
        WHERE
            (
                fc.domestic_competition_id IN ({comp_ids_sql})
                OR tc.domestic_competition_id IN ({comp_ids_sql})
            )
            -- season is stored as "17/18"; add 2000 to get the start year
            AND (2000 + CAST(SPLIT_PART(t.transfer_season, '/', 1) AS INTEGER))
                BETWEEN {season_min} AND {season_max}
            AND t.transfer_fee IS NOT NULL
        ORDER BY t.player_id, t.transfer_season
    """

    con = _connect()
    df = con.execute(query).fetchdf()
    con.close()
    return df


# ---------------------------------------------------------------------------
# CLI: python -m src.scraping.tm_scraper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if "--skip-download" in sys.argv:
        print("Skipping download, building DB from existing CSVs...")
    else:
        download_dataset()

    build_db()

    print("\n--- Valuation sample (ENG-Premier League, 2023) ---")
    df_val = scrape_valuations(leagues=["ENG-Premier League"], seasons=[2023])
    print(df_val.head(10).to_string())
    print(f"\nShape: {df_val.shape}")

    print("\n--- Transfer sample (ENG-Premier League, 2023) ---")
    df_tr = scrape_transfers(leagues=["ENG-Premier League"], seasons=[2023])
    print(df_tr.head(10).to_string())
    print(f"\nShape: {df_tr.shape}")
