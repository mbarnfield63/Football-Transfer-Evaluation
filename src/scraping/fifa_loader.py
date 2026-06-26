"""
Download FIFA annual datasets from Kaggle and fuzzy-match players to TM player_ids.

Creates one DuckDB table:
  fifa_contract_lookup — (player_id, season_int, contract_valid_until)

Year alignment:
  season_int = Y  ->  FIFA year Y data
  FIFA Y is collected ~Aug Y-1, capturing the contract state before the summer Y window.

Run:
    uv run python -m src.scraping.fifa_loader
    uv run python -m src.scraping.fifa_loader --skip-download  # re-match without re-downloading
"""

import re
import sys
from pathlib import Path

# Force UTF-8 output on Windows consoles that default to cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import duckdb
import pandas as pd
from rapidfuzz.fuzz import token_set_ratio, token_sort_ratio

from src.features.inflation import DB_PATH

FIFA_RAW_DIR    = Path(__file__).parents[2] / "data" / "raw" / "fifa"
FIFA_YEARS      = list(range(2017, 2025))   # season_int 2017-2024 inclusive
MATCH_THRESHOLD = 85                        # min rapidfuzz score; reject below this

# fifa-22 slug contains players_15.csv through players_22.csv (multi-year)
# fifa-23 slug contains players_23.csv (single year)
# ea-sports-fc-24 slug contains male_players.csv (FC24 = season_int 2024)
KAGGLE_SLUGS = [
    "stefanoleone992/fifa-22-complete-player-dataset",
    "stefanoleone992/fifa-23-complete-player-dataset",
    "stefanoleone992/ea-sports-fc-24-complete-player-dataset",
]

_COMP_IDS = "'GB1','L1','ES1','IT1','FR1','PO1','NL1','BE1','SC1','TR1'"

# Known TM -> FIFA nationality string differences
_NATIONALITY_MAP: dict[str, str] = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Türkiye":            "Turkey",
    "Ireland":            "Republic of Ireland",
    "Cape Verde Islands": "Cape Verde",
    "DR Congo":           "Congo DR",
    "Guinea-Bissau":      "Guinea Bissau",
    "The Gambia":         "Gambia",
    # Cote d'Ivoire / French overseas territories not in FIFA — will remain unmatched
}

# Canonical column names with possible aliases across FIFA CSV versions.
# Order matters — first alias match wins.
_COL_ALIASES: dict[str, list[str]] = {
    "sofifa_id":            ["sofifa_id", "player_id", "id"],
    "short_name":           ["short_name", "name"],
    "long_name":            ["long_name"],
    "nationality":          ["nationality_name", "nationality"],
    "club_name":            ["club_name", "club"],
    # FC24 renamed to club_contract_valid_until_year; older CSVs use club_contract_valid_until
    "contract_valid_until": [
        "club_contract_valid_until_year",
        "club_contract_valid_until",
        "contract_valid_until",
        "contract valid until",
    ],
}

_TRANSFERS_QUERY = f"""
SELECT
    t.player_id,
    ANY_VALUE(t.player_name)                                                  AS player_name,
    ANY_VALUE(p.country_of_citizenship)                                       AS nationality,
    ANY_VALUE(t.from_club_name)                                               AS from_club_name,
    (2000 + CAST(SPLIT_PART(t.transfer_season, '/', 1) AS INTEGER))          AS season_int
FROM transfers t
JOIN players p   ON p.player_id = t.player_id
JOIN clubs   fc  ON fc.club_id  = t.from_club_id
WHERE t.transfer_fee > 0
  AND fc.domestic_competition_id IN ({_COMP_IDS})
  AND (2000 + CAST(SPLIT_PART(t.transfer_season, '/', 1) AS INTEGER)) BETWEEN 2017 AND 2024
GROUP BY t.player_id, season_int
ORDER BY season_int, t.player_id
"""


# ---------------------------------------------------------------------------
# Kaggle download
# ---------------------------------------------------------------------------

def download_fifa_data(raw_dir: Path = FIFA_RAW_DIR) -> None:
    import kaggle  # noqa: PLC0415

    kaggle.api.authenticate()
    for slug in KAGGLE_SLUGS:
        dest = raw_dir / slug.split("/")[1]
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {slug} -> {dest}")
        kaggle.api.dataset_download_files(slug, path=str(dest), unzip=True)
    print("Downloads complete.")


# ---------------------------------------------------------------------------
# Column normalisation
# ---------------------------------------------------------------------------

def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame | None:
    """Return df with canonical column names, or None if required cols are absent."""
    cols_lower = {c.lower(): c for c in df.columns}
    rename: dict[str, str] = {}
    for canonical, aliases in _COL_ALIASES.items():
        for alias in aliases:
            if alias in cols_lower:
                rename[cols_lower[alias]] = canonical
                break
    df = df.rename(columns=rename)
    required = list(_COL_ALIASES.keys())
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None
    return df[required].copy()


def _infer_year_from_path(path: Path) -> int | None:
    """Extract the FIFA year from filename or parent directory name."""
    # Pattern 1: players_YY.csv  (e.g. players_22.csv -> 2022)
    m = re.search(r"players_(\d{2})\.csv$", path.name, re.IGNORECASE)
    if m:
        return 2000 + int(m.group(1))

    # Pattern 2: male_players.csv — infer from directory (e.g. ea-sports-fc-24-... -> 2024)
    if "male_players" in path.name.lower():
        dm = re.search(r"(?:fifa|fc)-?(\d{2})", path.parent.name, re.IGNORECASE)
        if dm:
            return 2000 + int(dm.group(1))

    return None


# ---------------------------------------------------------------------------
# Load all FIFA CSVs into a single DataFrame
# ---------------------------------------------------------------------------

def load_fifa_dataframe(raw_dir: Path = FIFA_RAW_DIR) -> pd.DataFrame:
    csvs = (
        list(raw_dir.glob("**/players_*.csv"))
        + list(raw_dir.glob("**/male_players.csv"))
    )
    if not csvs:
        raise FileNotFoundError(
            f"No FIFA CSVs found under {raw_dir}. "
            "Run without --skip-download to fetch them first."
        )

    frames: list[pd.DataFrame] = []
    found_years: set[int] = set()

    for path in csvs:
        year = _infer_year_from_path(path)
        if year is None or year not in FIFA_YEARS:
            continue

        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            print(f"  Warning: could not read {path}: {exc}")
            continue

        # FC24 (and potentially others) has multiple update snapshots per release.
        # Keep only the latest update per player so each sofifa_id appears once.
        if "fifa_update" in df.columns and "sofifa_id" in df.columns:
            df = (
                df.sort_values("fifa_update")
                .drop_duplicates(subset=["sofifa_id"], keep="last")
            )

        df = _standardise_columns(df)
        if df is None:
            print(f"  Warning: {path.name} (year={year}) missing required columns — skipped")
            continue

        df["contract_valid_until"] = pd.to_numeric(
            df["contract_valid_until"], errors="coerce"
        )
        df = df.dropna(subset=["contract_valid_until"])
        df = df[df["contract_valid_until"] > 0]
        df["contract_valid_until"] = df["contract_valid_until"].astype(int)

        df["fifa_year"] = year
        df = df.drop_duplicates(subset=["sofifa_id"])
        frames.append(df)
        found_years.add(year)

    missing_years = set(FIFA_YEARS) - found_years
    for y in sorted(missing_years):
        print(f"  Warning: FIFA {y} data not found — season_int={y} will have no contract data")

    if not frames:
        raise ValueError("No usable FIFA CSV data found after standardisation.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(combined):,} FIFA player-year rows across years: {sorted(found_years)}")
    return combined


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

def _normalise_nationality(nat: str | None) -> str | None:
    if not nat:
        return None
    return _NATIONALITY_MAP.get(nat, nat)


def match_player(
    player_name: str,
    nationality: str | None,
    from_club_name: str | None,
    fifa_df: pd.DataFrame,
    season_int: int,
    threshold: int = MATCH_THRESHOLD,
) -> tuple[int | None, float, str | None]:
    """Return (contract_valid_until, score, fifa_short_name) or (None, 0.0, None)."""
    norm_nat = _normalise_nationality(nationality)

    mask = fifa_df["fifa_year"] == season_int
    if norm_nat:
        mask &= fifa_df["nationality"] == norm_nat
    candidates = fifa_df[mask]

    if candidates.empty:
        return None, 0.0, None

    # token_sort_ratio handles abbreviated short names ("K. Mbappe")
    # token_set_ratio handles long legal names where TM name is a subset
    # ("Philippe Coutinho" is a subset of "Philippe Coutinho Correia" -> 100%)
    scores = candidates.apply(
        lambda row: max(
            token_sort_ratio(player_name, str(row["short_name"])),
            token_set_ratio(player_name, str(row["long_name"])),
        ),
        axis=1,
    )

    best_score = scores.max()
    if best_score < threshold:
        return None, 0.0, None

    near_best = candidates[scores >= best_score - 2]
    if len(near_best) == 1:
        winner = near_best.iloc[0]
    else:
        # Tiebreak on club name similarity
        club_scores = near_best["club_name"].apply(
            lambda c: token_sort_ratio(from_club_name or "", str(c))
        )
        winner = near_best.iloc[club_scores.argmax()]

    return int(winner["contract_valid_until"]), float(best_score), str(winner["short_name"])


# ---------------------------------------------------------------------------
# Build lookup table
# ---------------------------------------------------------------------------

def build_fifa_contract_lookup(
    db_path: Path = DB_PATH,
    raw_dir: Path = FIFA_RAW_DIR,
) -> None:
    con = duckdb.connect(str(db_path))
    transfers = con.execute(_TRANSFERS_QUERY).fetchdf()
    con.close()

    total = len(transfers)
    print(f"\nMatching {total} (player_id, season_int) pairs against FIFA data...")

    fifa_df = load_fifa_dataframe(raw_dir)

    records: list[tuple] = []
    unmatched: list[str] = []

    for row in transfers.itertuples(index=False):
        contract_year, score, fifa_name = match_player(
            player_name=str(row.player_name),
            nationality=row.nationality,
            from_club_name=row.from_club_name,
            fifa_df=fifa_df,
            season_int=int(row.season_int),
        )
        if contract_year is not None:
            records.append((
                int(row.player_id),
                int(row.season_int),
                contract_year,
                round(score, 1),
                fifa_name,
            ))
        else:
            unmatched.append(f"{row.player_name} ({row.nationality}, {row.season_int})")

    matched = len(records)
    print(f"Matched: {matched}/{total} ({100*matched/total:.1f}%)")
    if unmatched:
        print(f"Unmatched ({len(unmatched)} total), first 15:")
        for name in unmatched[:15]:
            print(f"  - {name}")

    con = duckdb.connect(str(db_path))
    con.execute("DROP TABLE IF EXISTS fifa_contract_lookup")
    con.execute("""
        CREATE TABLE fifa_contract_lookup (
            player_id            INTEGER NOT NULL,
            season_int           INTEGER NOT NULL,
            contract_valid_until INTEGER NOT NULL,
            match_score          REAL,
            fifa_player_name     VARCHAR,
            PRIMARY KEY (player_id, season_int)
        )
    """)
    if records:
        con.executemany(
            "INSERT INTO fifa_contract_lookup VALUES (?, ?, ?, ?, ?)",
            records,
        )
    con.close()
    print(f"Written {matched} rows to fifa_contract_lookup in {db_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--skip-download" not in sys.argv:
        download_fifa_data()
    build_fifa_contract_lookup()
