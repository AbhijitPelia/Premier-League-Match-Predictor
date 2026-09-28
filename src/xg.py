"""Fetch per-match expected goals from Understat.

Writes data/processed/xg.csv keyed by match_id, which load_matches() merges
into the matches table automatically.

Run from the project root:
    python -m src.xg
"""
import pandas as pd
import soccerdata as sd

from src.clean import XG_PATH, load_matches
from src.ingest import SEASONS
from src.teams import canonical_name_or_none

LEAGUE = "ENG-Premier League"


def fetch_schedule() -> pd.DataFrame:
    """Download the Understat schedule for every season, with xG per match."""
    understat = sd.Understat(leagues=LEAGUE, seasons=SEASONS)
    schedule = understat.read_schedule().reset_index()
    return schedule[schedule["is_result"]]


def to_match_ids(schedule: pd.DataFrame) -> pd.DataFrame:
    """Map Understat team names onto canonical IDs and build match_id.

    Joining on match_id rather than date avoids any disagreement between the
    two sources about which day a match was played on.
    """
    home_id = schedule["home_team"].map(canonical_name_or_none)
    away_id = schedule["away_team"].map(canonical_name_or_none)

    unknown = sorted(
        set(schedule.loc[home_id.isna(), "home_team"])
        | set(schedule.loc[away_id.isna(), "away_team"])
    )
    if unknown:
        raise KeyError(
            "Understat team names not in TEAM_ALIASES: "
            + ", ".join(repr(name) for name in unknown)
        )

    return pd.DataFrame({
        "match_id": schedule["season"].astype(str) + "_" + home_id + "_" + away_id,
        "home_xg": pd.to_numeric(schedule["home_xg"], errors="coerce"),
        "away_xg": pd.to_numeric(schedule["away_xg"], errors="coerce"),
    })


def main() -> None:
    schedule = fetch_schedule()
    print(f"Fetched {len(schedule)} played matches from Understat")

    xg = to_match_ids(schedule).dropna(subset=["home_xg", "away_xg"])
    duplicates = xg["match_id"][xg["match_id"].duplicated()]
    if not duplicates.empty:
        raise ValueError(f"Duplicate match_ids from Understat: {duplicates.tolist()[:5]}")

    xg.to_csv(XG_PATH, index=False)
    print(f"Saved {len(xg)} rows to {XG_PATH.name}\n")

    # How much of the matches table did we manage to cover, season by season?
    matches = load_matches()
    covered = matches["home_xg"].notna() if "home_xg" in matches else pd.Series(False, index=matches.index)
    summary = pd.DataFrame({
        "matches": matches.groupby("season").size(),
        "with_xg": covered.groupby(matches["season"]).sum(),
        "coverage": covered.groupby(matches["season"]).mean().round(3),
    })

    # xG should average close to actual goals; a large gap means a bad join.
    both = matches.dropna(subset=["home_xg", "away_xg"])
    summary["avg_goals"] = (both["home_goals"] + both["away_goals"]) \
        .groupby(both["season"]).mean().round(2)
    summary["avg_xg"] = (both["home_xg"] + both["away_xg"]) \
        .groupby(both["season"]).mean().round(2)
    print(summary.to_string())


if __name__ == "__main__":
    main()