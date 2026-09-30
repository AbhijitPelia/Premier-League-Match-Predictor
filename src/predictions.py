"""Log predictions before kickoff, and score them once matches are played.

Predictions are written in long format, one row per fixture per market per
selection, each stamped with when it was generated and which model made it.
Nothing is ever overwritten: re-running appends a new generation, and scoring
uses the most recent generation made strictly before kickoff.

That ordering is the point of this file. Without a log written ahead of time,
any accuracy figure is unfalsifiable — the model could have seen the result.

Run from the project root:
    python -m src.predictions            # predict and log upcoming fixtures
    python -m src.predictions --score    # score logged predictions against results
"""
import argparse

import numpy as np
import pandas as pd

from src.clean import PROCESSED_DIR, load_matches
from src.corners import corner_probabilities
from src.corners import fit as fit_corners
from src.dixon_coles import fit, predict
from src.fixtures import fetch_fixtures, load_key

PREDICTIONS_PATH = PROCESSED_DIR / "predictions.csv"

GOALS_LINE = 2.5
CORNER_LINE = 10.5

# Columns identifying one prediction. A re-run adds a new generated_at rather
# than replacing these.
KEY = ["match_id", "market", "line", "selection"]


def season_code(date: pd.Timestamp) -> str:
    """Season code for a date, matching the format used in the matches table.

    Seasons run July to June, so a October 2026 fixture belongs to "2627".
    """
    start = date.year if date.month >= 7 else date.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def model_version(goals_model, corners_model) -> str:
    """Compact description of the settings that produced a prediction.

    Stored alongside every row so a later change to the model is visible in the
    log rather than silently mixed in with earlier predictions.
    """
    return (f"dc(xi={goals_model.xi},sd={goals_model.prior_sd},"
            f"blend={goals_model.blend})+nb(sd={corners_model.prior_sd})")


def prediction_rows(fixture, goals_model, corners_model,
                    generated_at: pd.Timestamp) -> list[dict]:
    """Every market's probabilities for one fixture, as rows."""
    home, away = fixture.home_id, fixture.away_id
    markets = predict(goals_model, home, away)
    corners = corner_probabilities(corners_model, home, away, lines=(CORNER_LINE,))

    season = season_code(fixture.kickoff_utc)
    selections = [
        ("1x2", None, "home", markets["home"]),
        ("1x2", None, "draw", markets["draw"]),
        ("1x2", None, "away", markets["away"]),
        ("ou_goals", GOALS_LINE, "over", markets[f"over{GOALS_LINE}"]),
        ("ou_goals", GOALS_LINE, "under", markets[f"under{GOALS_LINE}"]),
        ("ou_corners", CORNER_LINE, "over", corners[f"over{CORNER_LINE}"]),
        ("ou_corners", CORNER_LINE, "under", corners[f"under{CORNER_LINE}"]),
        ("btts", None, "yes", markets["btts"]),
        ("btts", None, "no", 1 - markets["btts"]),
    ]

    return [{
        "match_id": f"{season}_{home}_{away}",
        "season": season,
        "kickoff_utc": fixture.kickoff_utc,
        "home_id": home,
        "away_id": away,
        "market": market,
        "line": line,
        "selection": selection,
        "probability": probability,
        "generated_at": generated_at,
        "fitted_through": goals_model.fitted_through,
        "model_version": model_version(goals_model, corners_model),
    } for market, line, selection, probability in selections]


def check_probabilities_sum(rows: list[dict]) -> None:
    """Every market's selections must sum to 1. Cheap, and catches real bugs."""
    frame = pd.DataFrame(rows)
    totals = frame.groupby(["match_id", "market", "line"], dropna=False)["probability"].sum()
    bad = totals[(totals - 1).abs() > 1e-6]
    if not bad.empty:
        raise ValueError(f"Probabilities do not sum to 1:\n{bad}")


def load_predictions() -> pd.DataFrame:
    """Read the log back, with timestamps as real UTC datetimes."""
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    frame = pd.read_csv(PREDICTIONS_PATH, dtype={"season": str})
    for column in ("kickoff_utc", "generated_at", "fitted_through"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def append(rows: list[dict]) -> int:
    """Add rows to the log, skipping exact duplicates of an earlier run."""
    new = pd.DataFrame(rows)
    existing = load_predictions()

    combined = pd.concat([existing, new], ignore_index=True) if not existing.empty else new
    combined = combined.drop_duplicates(subset=KEY + ["generated_at"], keep="last")
    combined = combined.sort_values(["kickoff_utc"] + KEY + ["generated_at"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(PREDICTIONS_PATH, index=False)
    return len(combined) - len(existing)


def latest_before_kickoff(predictions: pd.DataFrame) -> pd.DataFrame:
    """The most recent generation of each prediction made before its kickoff.

    Anything generated at or after kickoff is dropped outright — it could have
    been informed by the match itself, so it does not count.
    """
    valid = predictions[predictions["generated_at"] < predictions["kickoff_utc"]]
    return (valid.sort_values("generated_at")
            .groupby(KEY, dropna=False)
            .tail(1))


def actual_selection(match) -> dict[str, str]:
    """What actually happened, in the same vocabulary as the stored selections."""
    total_goals = match.home_goals + match.away_goals
    total_corners = match.home_corners + match.away_corners
    return {
        "1x2": ("home" if match.home_goals > match.away_goals
                else "away" if match.home_goals < match.away_goals else "draw"),
        "ou_goals": "over" if total_goals > GOALS_LINE else "under",
        "ou_corners": "over" if total_corners > CORNER_LINE else "under",
        "btts": "yes" if match.home_goals > 0 and match.away_goals > 0 else "no",
    }


def score() -> None:
    """Join logged predictions to played matches and report log loss per market."""
    predictions = load_predictions()
    if predictions.empty:
        print("No predictions logged yet.")
        return

    predictions = latest_before_kickoff(predictions)
    matches = load_matches().dropna(subset=["home_goals", "home_corners"])

    outcomes = {match.match_id: actual_selection(match)
                for match in matches.itertuples()}
    played = predictions[predictions["match_id"].isin(outcomes)]
    if played.empty:
        pending = predictions["match_id"].nunique()
        print(f"No logged predictions have been played yet ({pending} fixtures waiting).")
        return

    # Keep only the row naming what actually happened; its probability is what
    # log loss scores.
    hit = played[[
        outcomes[match_id][market] == selection
        for match_id, market, selection
        in zip(played["match_id"], played["market"], played["selection"])
    ]]

    summary = []
    for market, group in hit.groupby("market"):
        probabilities = np.clip(group["probability"].to_numpy(), 1e-15, 1)
        summary.append({
            "market": market,
            "matches": len(group),
            "log_loss": float(-np.log(probabilities).mean()),
            "mean_prob": float(probabilities.mean()),
        })

    print(f"Scored {hit['match_id'].nunique()} played fixtures\n")
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    print("\nBacktest marks for comparison: 1x2 0.9831, ou_goals 0.6773, "
          "ou_corners 0.6861")


def generate(days: int) -> None:
    """Fetch upcoming fixtures, predict them, and append to the log."""
    fixtures = fetch_fixtures(load_key(), days)
    now = pd.Timestamp.now("UTC")

    upcoming = fixtures[fixtures["kickoff_utc"] > now]
    if upcoming.empty:
        print(f"No fixtures kicking off in the next {days} days.")
        return

    matches = load_matches()
    goals_model = fit(matches)
    corners_model = fit_corners(matches)

    rows = []
    for fixture in upcoming.itertuples():
        rows.extend(prediction_rows(fixture, goals_model, corners_model, now))
    check_probabilities_sum(rows)

    added = append(rows)
    print(f"Logged {len(rows)} rows for {len(upcoming)} fixtures "
          f"({added} new) to {PREDICTIONS_PATH.name}")
    print(f"Model: {model_version(goals_model, corners_model)}, "
          f"fitted through {goals_model.fitted_through.date()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Log and score predictions")
    parser.add_argument("--score", action="store_true",
                        help="score logged predictions instead of generating new ones")
    parser.add_argument("--days", type=int, default=10,
                        help="how far ahead to predict (default 10)")
    args = parser.parse_args()

    score() if args.score else generate(args.days)


if __name__ == "__main__":
    main()