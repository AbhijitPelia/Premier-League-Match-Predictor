"""Download Premier League match CSVs from football-data.co.uk into data/raw/.

Run from the project root:
    python -m src.ingest
"""
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

# Season codes: "1617" means 2016-17. The last entry is the season in progress.
SEASONS = [
    "1617", "1718", "1819", "1920", "2021",
    "2122", "2223", "2324", "2425", "2526", "2627",
]
CURRENT_SEASON = SEASONS[-1]


def raw_path(season: str) -> Path:
    return RAW_DIR / f"E0_{season}.csv"


def download_season(season: str, force: bool = False) -> Path:
    """Download one season's CSV, skipping it if already cached (unless force=True)."""
    path = raw_path(season)
    if path.exists() and not force:
        print(f"{season}: cached")
        return path

    url = BASE_URL.format(season=season)
    resp = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "pl-predictor (personal project)"},
    )
    resp.raise_for_status()
    path.write_bytes(resp.content)
    print(f"{season}: downloaded ({len(resp.content) / 1024:.0f} KB)")
    time.sleep(1)  # be polite to the server
    return path


def load_season(season: str) -> pd.DataFrame:
    """Read one raw season file into a DataFrame (no cleaning yet)."""
    df = pd.read_csv(raw_path(season), encoding="latin-1")
    df = df.dropna(subset=["HomeTeam"]).copy()  # blank trailing rows; copy defragments
    df["season"] = season
    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for season in SEASONS:
        # The current season's file updates every matchweek, so always refresh it.
        download_season(season, force=(season == CURRENT_SEASON))

    print("\nRows per season:")
    for season in SEASONS:
        df = load_season(season)
        print(f"  {season}: {len(df):>3} matches, {df.shape[1]} columns")


if __name__ == "__main__":
    main()