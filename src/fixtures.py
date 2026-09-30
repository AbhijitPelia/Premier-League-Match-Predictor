"""Fetch upcoming Premier League fixtures from football-data.org and predict them.

Needs FOOTBALL_DATA_API_KEY in a .env file at the project root. Get a free key
at https://www.football-data.org/client/register.

Run from the project root:
    python -m src.fixtures
    python -m src.fixtures --days 14
"""
import argparse
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from src.clean import load_matches
from src.corners import CORNER_LINES
from src.corners import corner_probabilities
from src.corners import fit as fit_corners
from src.dixon_coles import expected_goals, fit, predict
from src.teams import canonical_name_or_none

BASE_URL = "https://api.football-data.org/v4"
COMPETITION = "PL"


def load_key() -> str:
    """Read the API key from .env, failing with a useful message if absent."""
    load_dotenv()
    key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not key:
        raise RuntimeError(
            "FOOTBALL_DATA_API_KEY not set. Put it in a .env file at the "
            "project root: FOOTBALL_DATA_API_KEY=your_key_here"
        )
    return key


def _throttle(response: requests.Response) -> None:
    """Sleep until the request counter resets, if this response used it up.

    football-data.org reports remaining requests in X-RequestsAvailable and
    seconds until reset in X-RequestCounter-Reset. Waiting when the budget is
    spent is cheaper than being blocked.
    """
    try:
        remaining = int(response.headers.get("X-RequestsAvailable", 1))
        reset_in = int(response.headers.get("X-RequestCounter-Reset", 0))
    except ValueError:
        return  # headers malformed; nothing reliable to act on

    if remaining <= 0 and reset_in > 0:
        print(f"  rate limit reached, waiting {reset_in}s...")
        time.sleep(reset_in + 1)


def get(path: str, key: str, **params) -> dict:
    """GET one endpoint, respecting the rate limit reported in the headers."""
    response = requests.get(
        f"{BASE_URL}/{path}",
        headers={"X-Auth-Token": key},
        params=params,
        timeout=30,
    )
    _throttle(response)

    if response.status_code == 429:
        raise RuntimeError("Rate limited despite throttling; wait a minute and retry")
    response.raise_for_status()
    return response.json()


def to_canonical(team: dict) -> str | None:
    """Map a football-data.org team object onto a canonical team ID.

    The API gives three spellings per team. Trying all of them means most
    already match aliases added for the other two data sources.
    """
    for spelling in (team.get("shortName"), team.get("name"), team.get("tla")):
        if spelling:
            canonical = canonical_name_or_none(spelling)
            if canonical:
                return canonical
    return None


def fetch_fixtures(key: str, days: int = 10) -> pd.DataFrame:
    """Upcoming scheduled matches within the next `days` days."""
    today = pd.Timestamp.now("UTC").normalize()
    payload = get(
        f"competitions/{COMPETITION}/matches",
        key,
        status="SCHEDULED",
        dateFrom=today.strftime("%Y-%m-%d"),
        dateTo=(today + pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
    )

    rows, unknown = [], set()
    for match in payload.get("matches", []):
        home_id = to_canonical(match["homeTeam"])
        away_id = to_canonical(match["awayTeam"])
        if home_id is None:
            unknown.add(match["homeTeam"].get("name"))
        if away_id is None:
            unknown.add(match["awayTeam"].get("name"))
        if home_id is None or away_id is None:
            continue

        rows.append({
            "kickoff_utc": pd.Timestamp(match["utcDate"]),
            "matchday": match.get("matchday"),
            "home_id": home_id,
            "away_id": away_id,
        })

    if unknown:
        raise KeyError(
            "football-data.org team names not in TEAM_ALIASES: "
            + ", ".join(repr(name) for name in sorted(unknown))
        )

    return pd.DataFrame(rows).sort_values("kickoff_utc").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict upcoming fixtures")
    parser.add_argument("--days", type=int, default=10,
                        help="how far ahead to look (default 10)")
    args = parser.parse_args()

    key = load_key()
    fixtures = fetch_fixtures(key, args.days)
    if fixtures.empty:
        print(f"No scheduled fixtures in the next {args.days} days.")
        return

    matches = load_matches()
    goals_model = fit(matches)
    corners_model = fit_corners(matches)
    print(f"Models fitted through {goals_model.fitted_through.date()} "f"on {goals_model.n_matches} matches\n")

    for fixture in fixtures.itertuples():
        home, away = fixture.home_id, fixture.away_id
        markets = predict(goals_model, home, away)
        corners = corner_probabilities(corners_model, home, away)
        lambda_home, lambda_away = expected_goals(goals_model, home, away)

        kickoff = fixture.kickoff_utc.tz_convert("Europe/London")
        print(f"{home} vs {away}   {kickoff:%a %d %b %H:%M} UK "f"(matchday {fixture.matchday})")
        print(f"  expected goals    {lambda_home:.2f} - {lambda_away:.2f}")
        print(f"  result            home {markets['home']:.1%}   "f"draw {markets['draw']:.1%}   away {markets['away']:.1%}")
        print(f"  over/under 2.5    over {markets['over2.5']:.1%}   "f"under {markets['under2.5']:.1%}")
        print(f"  over/under 10.5c  over {corners['over10.5']:.1%}   "f"under {corners['under10.5']:.1%}")
        print(f"  both teams score  {markets['btts']:.1%}\n")


if __name__ == "__main__":
    main()