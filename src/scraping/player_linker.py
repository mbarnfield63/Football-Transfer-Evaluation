import pandas as pd
from rapidfuzz import process, fuzz
from pathlib import Path

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
ID_MAP_PATH = PROCESSED_DIR / "player_id_map.csv"


def build_id_map(tm_df: pd.DataFrame, fbref_df: pd.DataFrame) -> pd.DataFrame:
    """Fuzzy-match players across TM and FBRef using name + birth year + position."""
    raise NotImplementedError


def load_id_map() -> pd.DataFrame:
    return pd.read_csv(ID_MAP_PATH)
