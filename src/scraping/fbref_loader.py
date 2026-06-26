"""
Load FBref Big-5 standard stats CSVs and fuzzy-match to TM player_ids.

Manual download steps (one-time per season):
  1. Go to https://fbref.com/en/comps/Big5/YYYY-{YYYY+1}/stats/players/
           YYYY-{YYYY+1}-Big-5-European-Leagues-Stats
  2. Click "Share & Export" → "Get table as CSV"
  3. Save as data/raw/fbref/YYYY.csv  (e.g. 2023.csv for the 2023-24 season)
  4. Repeat for each season 2017–2024

Then run:
    uv run python -m src.scraping.fbref_loader
    uv run python -m src.scraping.fbref_loader --skip-match  # reload CSVs only
"""

import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import duckdb
import pandas as pd
from rapidfuzz.fuzz import token_set_ratio, token_sort_ratio

from src.features.inflation import DB_PATH

FBREF_DIR = Path(__file__).parents[2] / "data" / "raw" / "fbref"

MATCH_THRESHOLD = 80

# FBref "Comp" column → TM competition_id
COMP_MAP = {
    "Premier League":  "GB1",
    "Bundesliga":      "L1",
    "La Liga":         "ES1",
    "Serie A":         "IT1",
    "Ligue 1":         "FR1",
}

# TM player universe query — one row per (player, season, competition)
_TM_PLAYERS_SQL = """
SELECT
    a.player_id,
    ANY_VALUE(a.player_name)              AS player_name,
    ANY_VALUE(p.country_of_citizenship)   AS nationality,
    ANY_VALUE(cl.name)                    AS club_name,
    g.season,
    g.competition_id
FROM appearances a
JOIN games   g  ON g.game_id   = a.game_id
JOIN players p  ON p.player_id = a.player_id
JOIN clubs   cl ON cl.club_id  = a.player_club_id
WHERE g.competition_id IN ('GB1','L1','ES1','IT1','FR1')
  AND g.season BETWEEN 2017 AND 2025
GROUP BY a.player_id, g.season, g.competition_id
"""


# ---------------------------------------------------------------------------
# Load + normalise FBref CSVs
# ---------------------------------------------------------------------------

def _infer_season(path: Path) -> int | None:
    m = re.search(r"(\d{4})", path.stem)
    return int(m.group(1)) if m else None


def load_fbref_csvs(fbref_dir: Path = FBREF_DIR) -> pd.DataFrame:
    csvs = sorted(fbref_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSVs found in {fbref_dir}.\n"
            "Download from FBref Big-5 standard stats pages — see module docstring."
        )

    frames = []
    for path in csvs:
        season = _infer_season(path)
        if season is None:
            print(f"  Warning: cannot infer season from {path.name} — skipped")
            continue

        df = pd.read_csv(path, skiprows=0)

        # FBref repeats header rows every 25 rows
        df = df[df["Player"] != "Player"].copy()

        # Nation column is "eng ENG" or just "ENG" — take last token
        if "Nation" in df.columns:
            df["Nation"] = df["Nation"].astype(str).str.split().str[-1].str.upper()

        # Comp → competition_id
        if "Comp" in df.columns:
            df["competition_id"] = df["Comp"].map(COMP_MAP)
        else:
            df["competition_id"] = None

        # keep only what we need
        keep = ["Player", "Nation", "Squad", "competition_id"]
        for col in ("xG", "PrgC"):
            if col in df.columns:
                keep.append(col)
            else:
                df[col] = None
                keep.append(col)

        df = df[keep].copy()
        df["season"] = season
        df["xG"] = pd.to_numeric(df["xG"], errors="coerce")
        df["PrgC"] = pd.to_numeric(df["PrgC"], errors="coerce")

        # aggregate players who moved mid-season (appear twice with different Squad)
        df = (
            df.groupby(["Player", "Nation", "competition_id", "season"], dropna=False)
            .agg(Squad=("Squad", "first"), xG=("xG", "sum"), PrgC=("PrgC", "sum"))
            .reset_index()
        )

        frames.append(df)
        print(f"  Loaded {path.name}: {len(df)} player-season rows (season={season})")

    if not frames:
        raise ValueError("No usable FBref CSVs after loading.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"Total FBref rows: {len(combined):,}")
    return combined


# ---------------------------------------------------------------------------
# Fuzzy match FBref player → TM player_id
# ---------------------------------------------------------------------------

def _match_player(
    fbref_name: str,
    fbref_club: str,
    candidates: pd.DataFrame,
    threshold: int = MATCH_THRESHOLD,
) -> int | None:
    if candidates.empty:
        return None

    scores = candidates.apply(
        lambda r: max(
            token_sort_ratio(fbref_name, str(r["player_name"])),
            token_set_ratio(fbref_name, str(r["player_name"])),
        ),
        axis=1,
    )

    best = scores.max()
    if best < threshold:
        return None

    near_best = candidates[scores >= best - 2]
    if len(near_best) == 1:
        return int(near_best.iloc[0]["player_id"])

    # tiebreak on club name
    club_scores = near_best["club_name"].apply(
        lambda c: token_sort_ratio(fbref_club or "", str(c))
    )
    return int(near_best.iloc[club_scores.argmax()]["player_id"])


def match_and_write(
    fbref_df: pd.DataFrame,
    db_path: Path = DB_PATH,
) -> None:
    con = duckdb.connect(str(db_path))
    tm = con.execute(_TM_PLAYERS_SQL).fetchdf()
    con.close()

    records: list[tuple] = []
    unmatched = 0

    for _, row in fbref_df.iterrows():
        comp = row["competition_id"]
        season = int(row["season"])

        # narrow TM candidates to same league + season
        mask = (tm["season"] == season)
        if pd.notna(comp):
            mask &= (tm["competition_id"] == comp)
        candidates = tm[mask]

        player_id = _match_player(str(row["Player"]), str(row.get("Squad", "")), candidates)

        if player_id is not None:
            records.append((
                player_id,
                season,
                float(row["xG"]) if pd.notna(row["xG"]) else None,
                float(row["PrgC"]) if pd.notna(row["PrgC"]) else None,
                str(row["Player"]),
            ))
        else:
            unmatched += 1

    matched = len(records)
    total = len(fbref_df)
    print(f"Matched: {matched}/{total} ({100*matched/total:.1f}%)  Unmatched: {unmatched}")

    con = duckdb.connect(str(db_path))
    con.execute("DROP TABLE IF EXISTS fbref_stats")
    con.execute("""
        CREATE TABLE fbref_stats (
            player_id        INTEGER NOT NULL,
            season           INTEGER NOT NULL,
            xg               DOUBLE,
            progressive_carries DOUBLE,
            fbref_player_name VARCHAR,
            PRIMARY KEY (player_id, season)
        )
    """)
    if records:
        con.executemany("INSERT INTO fbref_stats VALUES (?, ?, ?, ?, ?)", records)
    con.close()
    print(f"Written {matched} rows → fbref_stats in {db_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    skip_match = "--skip-match" in sys.argv

    fbref_df = load_fbref_csvs()

    if not skip_match:
        match_and_write(fbref_df)
    else:
        print("--skip-match: CSV load only, not writing to DB")
