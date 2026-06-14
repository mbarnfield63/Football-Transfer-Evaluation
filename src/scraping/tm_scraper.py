import pandas as pd
import soccerdata as sd
from pathlib import Path

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"

LEAGUES = ["ENG-Premier League", "GER-Bundesliga", "ESP-La Liga", "ITA-Serie A", "FRA-Ligue 1"]
SEASONS = list(range(2017, 2026))


def scrape_valuations(leagues=LEAGUES, seasons=SEASONS) -> pd.DataFrame:
    raise NotImplementedError


def scrape_transfers(leagues=LEAGUES, seasons=SEASONS) -> pd.DataFrame:
    raise NotImplementedError
