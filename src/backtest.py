"""Walk-forward backtest for the Dixon-Coles model.

For each matchweek, refits using only matches played before that date, then
predicts the upcoming fixtures. Scores the model against the closing odds and
a naive base-rate baseline using log loss.

Run from the project root:
    python -m src.backtest
    python -m src.backtest --xi 0.003 --prior-sd 0.25
    python -m src.backtest --start-season 2223 --refit-days 7
"""
import argparse

import numpy as np
import pandas as pd

from src import corners
from src.clean import load_matches
from src.dixon_coles import DEFAULT_BLEND, DEFAULT_PRIOR_SD, DEFAULT_XI, fit, predict

from src.corners import DEFAULT_PRIOR_SD as CORNER_PRIOR_SD
from src.corners import DEFAULT_XI as CORNER_XI
from src.corners import corner_probabilities
from src.corners import fit as fit_corners

CORNER_LINE = 10.5  # closest line to the league average total

EPSILON = 1e-15  # keeps log loss finite if a model ever says 0%


def log_loss(probabilities: np.ndarray) -> float:
    """Mean negative log of the probability assigned to what actually happened."""
    return float(-np.log(np.clip(probabilities, EPSILON, 1)).mean())


def strip_overround(odds: np.ndarray) -> np.ndarray:
    """Turn decimal odds into probabilities that sum to 1.

    Bookmakers price so the implied probabilities sum to more than 1 (their
    margin). Dividing by the total removes it, giving the market's actual view.
    """
    implied = 1 / odds
    return implied / implied.sum(axis=1, keepdims=True)


def run_backtest(matches: pd.DataFrame, start_season: str, xi: float,
                prior_sd: float, blend: float, refit_days: int,
                corners: bool = False, corner_xi: float = CORNER_XI,
                corner_prior_sd: float = CORNER_PRIOR_SD) -> pd.DataFrame:
    """Predict every match from `start_season` onward, refitting as we go."""
    matches = matches.sort_values(["date", "match_id"]).reset_index(drop=True)
    test = matches[matches["season"] >= start_season]

    model = None
    corner_model = None
    next_refit = None
    rows = []

    for season, season_matches in test.groupby("season", sort=True):
        print(f"  {season}...", end="", flush=True)
        for match in season_matches.itertuples():
            if model is None or match.date >= next_refit:
                history = matches[matches["date"] < match.date]
                model = fit(history, xi=xi, prior_sd=prior_sd, blend=blend)
                if corners:
                    corner_model = fit_corners(history, xi=corner_xi, prior_sd=corner_prior_sd)
                next_refit = match.date + pd.Timedelta(days=refit_days)

            probabilities = predict(model, match.home_id, match.away_id)
            total_goals = match.home_goals + match.away_goals
            corner_probs = (corner_probabilities(corner_model, match.home_id, match.away_id, lines=(CORNER_LINE,))
                            if corners else {})
            rows.append({
                "match_id": match.match_id,
                "season": match.season,
                "date": match.date,
                "result": ("home" if match.home_goals > match.away_goals
                           else "away" if match.home_goals < match.away_goals
                           else "draw"),
                "over25": total_goals > 2.5,
                "p_home": probabilities["home"],
                "p_draw": probabilities["draw"],
                "p_away": probabilities["away"],
                "p_over25": probabilities["over2.5"],
                "odds_home": match.odds_home_close,
                "odds_draw": match.odds_draw_close,
                "odds_away": match.odds_away_close,
                "odds_over25": match.odds_over25_close,
                "odds_under25": match.odds_under25_close,
                "over_corners": (match.home_corners + match.away_corners) > CORNER_LINE,
                "p_over_corners": corner_probs.get(f"over{CORNER_LINE}"),
            })
        print(f" {len(season_matches)} matches")

    return pd.DataFrame(rows)


def score_1x2(results: pd.DataFrame) -> pd.Series:
    """Log loss for the model, the market and a naive baseline on 1X2."""
    outcomes = results["result"].to_numpy()
    model_probs = results[["p_home", "p_draw", "p_away"]].to_numpy()
    market_probs = strip_overround(
        results[["odds_home", "odds_draw", "odds_away"]].to_numpy()
    )

    # Naive baseline: the base rate of each outcome over the test set.
    base = np.array([(outcomes == o).mean() for o in ("home", "draw", "away")])
    base_probs = np.tile(base, (len(results), 1))

    picked = np.array([["home", "draw", "away"].index(o) for o in outcomes])
    rows = np.arange(len(results))

    return pd.Series({
        "model": log_loss(model_probs[rows, picked]),
        "market": log_loss(market_probs[rows, picked]),
        "baseline": log_loss(base_probs[rows, picked]),
        "matches": len(results),
    })


def score_over_under(results: pd.DataFrame) -> pd.Series:
    """Log loss for the model, the market and a naive baseline on over/under 2.5."""
    usable = results.dropna(subset=["odds_over25", "odds_under25"])
    if usable.empty:
        return pd.Series({"model": np.nan, "market": np.nan, "baseline": np.nan, "matches": 0})

    went_over = usable["over25"].to_numpy()
    model_over = usable["p_over25"].to_numpy()
    market_over = strip_overround(
        usable[["odds_over25", "odds_under25"]].to_numpy()
    )[:, 0]
    base_over = np.full(len(usable), went_over.mean())

    def picked(p_over):
        return np.where(went_over, p_over, 1 - p_over)

    return pd.Series({
        "model": log_loss(picked(model_over)),
        "market": log_loss(picked(market_over)),
        "baseline": log_loss(picked(base_over)),
        "matches": len(usable),
    })

def score_corners(results: pd.DataFrame) -> pd.Series:
    """Log loss for the model and a naive baseline on over/under corners.

    There is no market column: Football-Data.co.uk carries no corner odds, so
    the only benchmark is predicting the league base rate every time.
    """
    usable = results.dropna(subset=["p_over_corners"])
    if usable.empty:
        return pd.Series({"model": np.nan, "baseline": np.nan, "matches": 0})

    went_over = usable["over_corners"].to_numpy()
    model_over = usable["p_over_corners"].to_numpy()
    base_over = np.full(len(usable), went_over.mean())

    def picked(p_over):
        return np.where(went_over, p_over, 1 - p_over)

    return pd.Series({
        "model": log_loss(picked(model_over)),
        "baseline": log_loss(picked(base_over)),
        "matches": len(usable),
    })

def calibration(probabilities: pd.Series, happened: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Group predictions into bands and compare predicted rate to actual rate."""
    band = pd.cut(probabilities, np.linspace(0, 1, bins + 1))
    table = pd.DataFrame({
        "n": happened.groupby(band, observed=True).size(),
        "predicted": probabilities.groupby(band, observed=True).mean(),
        "actual": happened.groupby(band, observed=True).mean(),
    })
    table["gap"] = table["actual"] - table["predicted"]
    return table.round(3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument("--start-season", default="1920",
                        help="first season to predict (default 1920, where O/U odds begin)")
    parser.add_argument("--xi", type=float, default=DEFAULT_XI)
    parser.add_argument("--prior-sd", type=float, default=DEFAULT_PRIOR_SD)
    parser.add_argument("--blend", type=float, default=DEFAULT_BLEND,
                        help="weight on xG vs goals, 0 to 1 (default 0)")
    parser.add_argument("--corners", action="store_true",
                        help="also fit and score the corners model (slower)")
    parser.add_argument("--corner-xi", type=float, default=CORNER_XI)
    parser.add_argument("--corner-prior-sd", type=float, default=CORNER_PRIOR_SD)
    parser.add_argument("--refit-days", type=int, default=7,
                        help="how often to refit, in days (default 7)")
    args = parser.parse_args()

    matches = load_matches()
    print(f"Backtesting from {args.start_season} "f"(xi={args.xi}, prior_sd={args.prior_sd}, blend={args.blend}, "f"refit every {args.refit_days}d)")
    results = run_backtest(matches, args.start_season, args.xi, args.prior_sd, args.blend, args.refit_days, args.corners, args.corner_xi, args.corner_prior_sd)

    print("\n1X2 log loss (lower is better)")
    print(score_1x2(results).round(4).to_string())

    print("\nOver/under 2.5 log loss")
    print(score_over_under(results).round(4).to_string())

    if args.corners:
        print(f"\nOver/under {CORNER_LINE} corners log loss")
        print(score_corners(results).round(4).to_string())

        print(f"\nCalibration: over {CORNER_LINE} corners")
        print(calibration(results["p_over_corners"], results["over_corners"]).to_string())

    print("\n1X2 log loss by season")
    by_season = results.groupby("season").apply(score_1x2, include_groups=False)
    print(by_season.round(4).to_string())

    print("\nCalibration: home win")
    print(calibration(results["p_home"], results["result"] == "home").to_string())

    print("\nCalibration: over 2.5 goals")
    print(calibration(results["p_over25"], results["over25"]).to_string())


if __name__ == "__main__":
    main()