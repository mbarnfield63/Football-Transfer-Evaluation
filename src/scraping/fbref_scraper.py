import pandas as pd
import soccerdata as sd
from pathlib import Path

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"

LEAGUES = ["ENG-Premier League", "GER-Bundesliga", "ESP-La Liga", "ITA-Serie A", "FRA-Ligue 1"]
SEASONS = list(range(2017, 2026))

STAT_TYPES = ["standard", "shooting", "passing", "defense", "possession", "misc"]


def scrape_player_stats(leagues=LEAGUES, seasons=SEASONS) -> pd.DataFrame:
    raise NotImplementedError
