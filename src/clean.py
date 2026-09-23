"""Build the clean matches table from the raw Football-Data.co.uk CSVs.

Run from the project root:
    python -m src.clean
"""
import pandas as pd

from src.ingest import ROOT, SEASONS, load_season
from src.teams import canonical_name

PROCESSED_DIR = ROOT / "data" / "processed"
MATCHES_PATH = PROCESSED_DIR / "matches.csv"

def load_matches() -> pd.DataFrame:
    """Read the saved matches table back with the correct dtypes.

    Season codes must stay strings: read_csv would otherwise turn "2627" into
    the integer 2627, breaking any comparison against a season code.
    """
    return pd.read_csv(MATCHES_PATH, dtype={"season": str}, parse_dates=["date"])

# Raw column -> clean column. Every season must have these, or we stop.
REQUIRED = {
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "HC": "home_corners",
    "AC": "away_corners",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
}

# Closing odds, in priority order: Pinnacle (sharpest), market average, Bet365.
# Not every season carries every bookmaker, so take the first one available.
ODDS_SOURCES = {
    "odds_home_close": ["PSCH", "AvgCH", "B365CH"],
    "odds_draw_close": ["PSCD", "AvgCD", "B365CD"],
    "odds_away_close": ["PSCA", "AvgCA", "B365CA"],
    "odds_over25_close": ["PC>2.5", "AvgC>2.5", "B365C>2.5"],
    "odds_under25_close": ["PC<2.5", "AvgC<2.5", "B365C<2.5"],
}

def first_available(raw: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Take the first bookmaker column that exists, filling gaps from the next."""
    series = None
    for col in candidates:
        if col not in raw.columns:
            continue
        values = pd.to_numeric(raw[col], errors="coerce")
        series = values if series is None else series.fillna(values)
    if series is None:
        return pd.Series(float("nan"), index=raw.index)
    return series


def clean_season(season: str) -> pd.DataFrame:
    """Turn one raw season file into rows of the clean matches table."""
    raw = load_season(season)

    needed = ["Date", "HomeTeam", "AwayTeam", *REQUIRED]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"Season {season} is missing required columns: {missing}")

    # Drop any rows without a final score (postponed or abandoned matches).
    raw = raw.dropna(subset=["FTHG", "FTAG"])

    # Older files use dd/mm/yy, newer ones dd/mm/yyyy; "mixed" handles both.
    date = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed")

    # Kickoff times only exist in newer seasons, and are UK local time.
    if "Time" in raw.columns:
        local = date + pd.to_timedelta(raw["Time"].astype(str) + ":00", errors="coerce")
        kickoff_utc = local.dt.tz_localize("Europe/London").dt.tz_convert("UTC")
    else:
        kickoff_utc = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]")

    home_id = raw["HomeTeam"].map(canonical_name)
    away_id = raw["AwayTeam"].map(canonical_name)

    columns = {
        "match_id": season + "_" + home_id + "_" + away_id,
        "season": season,
        "date": date,
        "kickoff_utc": kickoff_utc,
        "home_id": home_id,
        "away_id": away_id,
    }
    for old, new in REQUIRED.items():
        columns[new] = pd.to_numeric(raw[old], errors="coerce").astype("Int64")
    for new, candidates in ODDS_SOURCES.items():
        columns[new] = first_available(raw, candidates)

    return pd.DataFrame(columns)


def build_matches() -> pd.DataFrame:
    """Clean every season and combine them into one table, oldest first."""
    matches = pd.concat([clean_season(s) for s in SEASONS], ignore_index=True)
    matches = matches.sort_values(["date", "kickoff_utc", "match_id"])
    matches = matches.reset_index(drop=True)

    dupes = matches["match_id"][matches["match_id"].duplicated()]
    if not dupes.empty:
        raise ValueError(f"Duplicate match_ids found: {dupes.tolist()[:5]}")
    return matches


def main() -> None:
    matches = build_matches()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_csv(MATCHES_PATH, index=False)
    print(f"Saved {len(matches)} matches to {MATCHES_PATH.relative_to(ROOT)}\n")

    # Quick per-season summary: match count, average goals and corners,
    # and what share of matches have each closing-odds column filled in.
    summary = pd.DataFrame({
        "matches": matches.groupby("season").size(),
        "avg_goals": (matches["home_goals"] + matches["away_goals"])
            .groupby(matches["season"]).mean().round(2),
        "avg_corners": (matches["home_corners"] + matches["away_corners"])
            .groupby(matches["season"]).mean().round(2),
    })
    odds_cov = matches[list(ODDS_SOURCES)].notna().groupby(matches["season"]).mean()
    summary = summary.join(odds_cov.round(2))
    print(summary.to_string())


if __name__ == "__main__":
    main()