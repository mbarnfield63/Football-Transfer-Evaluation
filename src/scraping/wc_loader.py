"""
Pull WC 2018 + 2022 player minutes from FBref via soccerdata.
Fuzzy-match to TM player_ids and write wc_squads table to DuckDB.

Run:
    uv run python -m src.scraping.wc_loader
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import duckdb
import pandas as pd
from rapidfuzz.fuzz import token_set_ratio, token_sort_ratio

from src.features.inflation import DB_PATH

WC_YEARS = [2018, 2022]

SOCCERDATA_CACHE = Path.home() / "soccerdata" / "data" / "FBref"

# FBref national team names → TM country_of_citizenship where they differ
_TEAM_NAME_MAP: dict[str, str] = {
    "South Korea":          "Korea, South",
    "Korea Republic":       "Korea, South",
    "IR Iran":              "Iran",
    "Ivory Coast":          "Ivory Coast",
    "Côte d'Ivoire":        "Ivory Coast",
    "United States":        "United States",
    "USA":                  "United States",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Türkiye":              "Türkiye",
    "Turkey":               "Türkiye",
    "Czech Republic":       "Czech Republic",
    "DR Congo":             "DR Congo",
    "Cape Verde":           "Cape Verde Islands",
}

# All TM players — match against full player universe (not just top-10 leagues)
_TM_PLAYERS_SQL = """
SELECT player_id, name AS player_name, country_of_citizenship AS nationality
FROM players
WHERE name IS NOT NULL
"""

MATCH_THRESHOLD = 78


# ---------------------------------------------------------------------------
# Parse cached FBref HTML directly — no Selenium, no network
# ---------------------------------------------------------------------------

def _parse_wc_html(year: int, cache_dir: Path = SOCCERDATA_CACHE) -> pd.DataFrame:
    from lxml import etree, html as lhtml  # noqa: PLC0415

    path = cache_dir / f"players_INT-World Cup_{year}_playing_time.html"
    if not path.exists():
        raise FileNotFoundError(
            f"Cache file not found: {path}\n"
            "Run once with soccerdata to populate cache:\n"
            "  uv run python -c \"import soccerdata as sd, warnings; warnings.filterwarnings('ignore'); "
            f"sd.FBref(['INT-World Cup'], [{year}]).read_player_season_stats('playing_time')\""
        )

    with open(path, "rb") as f:
        content = f.read()

    tree = lhtml.fromstring(content)

    # FBref hides the data table inside an HTML comment
    for comment in tree.xpath("//comment()"):
        if "stats_playing_time" in (comment.text or ""):
            parser = etree.HTMLParser(recover=True)
            subtree = etree.fromstring(comment.text, parser)
            tables = subtree.xpath("//table[contains(@id,'stats_playing_time')]")
            if tables:
                rows = []
                for tr in tables[0].xpath(".//tbody/tr[not(contains(@class,'thead'))]"):
                    cells = {td.get("data-stat"): "".join(td.itertext()).strip() for td in tr.xpath("td|th")}
                    if cells.get("player"):
                        rows.append(cells)
                df = pd.DataFrame(rows)
                df["wc_year"] = year
                df["minutes_played"] = pd.to_numeric(df.get("minutes", pd.Series(dtype=str)), errors="coerce").fillna(0).astype(int)
                # "ar Argentina" → "Argentina" (strip leading FIFA 2-letter code)
                df["team"] = df["team"].str.replace(r"^[a-z]{2,3}\s+", "", regex=True).str.strip()
                return df[["player", "team", "wc_year", "minutes_played"]].copy()

    raise ValueError(f"Could not find stats_playing_time table in {path}")


def fetch_wc_minutes(wc_years: list[int] = WC_YEARS) -> pd.DataFrame:
    frames = [_parse_wc_html(yr) for yr in wc_years]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["player"] != ""]  # drop empty rows

    print(f"Loaded {len(df)} WC player-tournament rows from cache")
    for yr in sorted(df["wc_year"].unique()):
        sub = df[df["wc_year"] == yr]
        played = (sub["minutes_played"] > 0).sum()
        print(f"  {yr}: {len(sub)} squad members, {played} played >0 min")

    return df


# ---------------------------------------------------------------------------
# Fuzzy match to TM player_ids
# ---------------------------------------------------------------------------

def _normalise_team(team: str) -> str:
    return _TEAM_NAME_MAP.get(team, team)


def _match_player(
    fbref_name: str,
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
    return int(near_best.iloc[0]["player_id"])


def match_and_write(wc_df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    con = duckdb.connect(str(db_path))
    tm = con.execute(_TM_PLAYERS_SQL).fetchdf()
    con.close()

    # pre-index by nationality so each lookup is O(small) not O(all players)
    tm_by_nat: dict[str, pd.DataFrame] = {
        nat: grp for nat, grp in tm.groupby("nationality")
    }

    records: list[tuple] = []
    unmatched: list[str] = []

    for _, row in wc_df.iterrows():
        nat = _normalise_team(str(row["team"]))
        candidates = tm_by_nat.get(nat, tm)  # fall back to full table if no nat match

        player_id = _match_player(str(row["player"]), candidates)

        if player_id is not None:
            records.append((
                player_id,
                int(row["wc_year"]),
                int(row["minutes_played"]),
                str(row["team"]),
                str(row["player"]),
            ))
        else:
            unmatched.append(f"{row['player']} ({row['team']}, {row['wc_year']})")

    # deduplicate: same player matched twice (e.g. listed under two clubs in FBref)
    seen: dict[tuple, tuple] = {}
    for rec in records:
        key = (rec[0], rec[1])  # player_id, wc_year
        if key not in seen or rec[2] > seen[key][2]:  # keep higher minutes
            seen[key] = rec
    records = list(seen.values())

    matched = len(records)
    total = len(wc_df)
    print(f"\nMatched: {matched}/{total} ({100*matched/total:.1f}%)")
    if unmatched:
        print(f"Unmatched ({len(unmatched)}), first 15:")
        for s in unmatched[:15]:
            print(f"  - {s}")

    con = duckdb.connect(str(db_path))
    con.execute("DROP TABLE IF EXISTS wc_squads")
    con.execute("""
        CREATE TABLE wc_squads (
            player_id        INTEGER NOT NULL,
            wc_year          INTEGER NOT NULL,
            minutes_played   INTEGER NOT NULL,
            national_team    VARCHAR,
            fbref_name       VARCHAR,
            PRIMARY KEY (player_id, wc_year)
        )
    """)
    if records:
        con.executemany("INSERT INTO wc_squads VALUES (?, ?, ?, ?, ?)", records)
    con.close()
    print(f"Written {matched} rows → wc_squads in {db_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    wc_df = fetch_wc_minutes()
    match_and_write(wc_df)

    # quick sanity check
    con = duckdb.connect(str(DB_PATH))
    summary = con.execute("""
        SELECT wc_year,
               COUNT(*) AS squad_members,
               SUM(CASE WHEN minutes_played >= 200 THEN 1 ELSE 0 END) AS treatment,
               SUM(CASE WHEN minutes_played < 200 THEN 1 ELSE 0 END)  AS control
        FROM wc_squads
        GROUP BY wc_year ORDER BY wc_year
    """).fetchdf()
    con.close()
    print("\nTreatment/control split (200-min threshold):")
    print(summary.to_string(index=False))
