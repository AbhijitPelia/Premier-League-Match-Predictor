"""Dixon-Coles model for Premier League match outcomes.

Fits an attack and defence strength per team, plus league-wide home advantage
and a low-score correction (rho). Produces a scoreline probability matrix, from
which 1X2 and over/under goals markets are derived.

Run from the project root:
    python -m src.dixon_coles
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from src.clean import MATCHES_PATH

MAX_GOALS = 10          # scoreline matrix covers 0-0 up to 10-10
DEFAULT_XI = 0.0018     # time decay per day; half-life of roughly one season


@dataclass
class DixonColesFit:
    """Fitted parameters, and the information needed to reproduce the fit."""
    attack: dict[str, float]
    defence: dict[str, float]
    home_advantage: float
    rho: float
    xi: float
    n_matches: int
    fitted_through: pd.Timestamp

    @property
    def teams(self) -> list[str]:
        return sorted(self.attack)


def _tau(home_goals, away_goals, lambda_home, lambda_away, rho):
    """Dixon-Coles correction applied to the four lowest scorelines."""
    tau = np.ones_like(lambda_home, dtype=float)
    tau = np.where((home_goals == 0) & (away_goals == 0),
                   1 - lambda_home * lambda_away * rho, tau)
    tau = np.where((home_goals == 0) & (away_goals == 1), 1 + lambda_home * rho, tau)
    tau = np.where((home_goals == 1) & (away_goals == 0), 1 + lambda_away * rho, tau)
    tau = np.where((home_goals == 1) & (away_goals == 1), 1 - rho, tau)
    return tau


def _neg_log_likelihood(params, home_idx, away_idx, home_goals, away_goals, weights, n_teams):
    """Weighted negative log-likelihood of the observed scorelines."""
    attack = params[:n_teams]
    defence = params[n_teams:2 * n_teams]
    home_advantage, rho = params[-2], params[-1]

    # Centre the attack ratings so the parameters are identifiable: without
    # this, adding a constant to every attack and subtracting it from every
    # defence gives an identical fit.
    attack = attack - attack.mean()

    lambda_home = np.exp(attack[home_idx] + defence[away_idx] + home_advantage)
    lambda_away = np.exp(attack[away_idx] + defence[home_idx])

    tau = _tau(home_goals, away_goals, lambda_home, lambda_away, rho)
    tau = np.clip(tau, 1e-10, None)  # keep the log finite if rho strays too far

    log_likelihood = (
        np.log(tau)
        + poisson.logpmf(home_goals, lambda_home)
        + poisson.logpmf(away_goals, lambda_away)
    )
    return -np.sum(weights * log_likelihood)


def fit(matches: pd.DataFrame, xi: float = DEFAULT_XI,
        as_of: pd.Timestamp | None = None) -> DixonColesFit:
    """Fit the model on every match played strictly before `as_of`.

    The `as_of` cutoff is what keeps backtests honest: a fit for gameweek N
    must never see results from gameweek N or later.
    """
    if as_of is not None:
        matches = matches[matches["date"] < as_of]
    if matches.empty:
        raise ValueError("No matches available to fit on")

    teams = sorted(set(matches["home_id"]) | set(matches["away_id"]))
    index = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    home_idx = matches["home_id"].map(index).to_numpy()
    away_idx = matches["away_id"].map(index).to_numpy()
    home_goals = matches["home_goals"].to_numpy(dtype=float)
    away_goals = matches["away_goals"].to_numpy(dtype=float)

    # Recent matches count for more: weight decays exponentially with age.
    latest = matches["date"].max()
    days_ago = (latest - matches["date"]).dt.days.to_numpy()
    weights = np.exp(-xi * days_ago)

    initial = np.concatenate([
        np.zeros(n_teams),      # attack
        np.zeros(n_teams),      # defence
        [0.25, -0.05],          # home advantage, rho
    ])
    bounds = [(-3, 3)] * (2 * n_teams) + [(-1, 1), (-0.5, 0.5)]

    result = minimize(
        _neg_log_likelihood,
        initial,
        args=(home_idx, away_idx, home_goals, away_goals, weights, n_teams),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"Fit did not converge: {result.message}")

    attack = result.x[:n_teams] - result.x[:n_teams].mean()
    defence = result.x[n_teams:2 * n_teams]

    return DixonColesFit(
        attack=dict(zip(teams, attack)),
        defence=dict(zip(teams, defence)),
        home_advantage=float(result.x[-2]),
        rho=float(result.x[-1]),
        xi=xi,
        n_matches=len(matches),
        fitted_through=latest,
    )


def score_matrix(model: DixonColesFit, home_id: str, away_id: str) -> np.ndarray:
    """Probability of every scoreline, as a (MAX_GOALS+1) x (MAX_GOALS+1) grid.

    Row i, column j is the probability of the match finishing i-j.
    """
    for team in (home_id, away_id):
        if team not in model.attack:
            raise KeyError(f"{team!r} has no fitted rating (no matches in the training window)")

    lambda_home = np.exp(model.attack[home_id] + model.defence[away_id] + model.home_advantage)
    lambda_away = np.exp(model.attack[away_id] + model.defence[home_id])

    goals = np.arange(MAX_GOALS + 1)
    home_probs = poisson.pmf(goals, lambda_home)
    away_probs = poisson.pmf(goals, lambda_away)
    matrix = np.outer(home_probs, away_probs)

    # Apply the low-score correction, then renormalise so the grid sums to 1.
    grid_home, grid_away = np.meshgrid(goals, goals, indexing="ij")
    matrix = matrix * _tau(grid_home, grid_away, lambda_home, lambda_away, model.rho)
    return matrix / matrix.sum()


def market_probabilities(matrix: np.ndarray, goal_lines=(1.5, 2.5, 3.5)) -> dict[str, float]:
    """Read every market off the scoreline matrix by summing the right cells."""
    goals = np.arange(matrix.shape[0])
    grid_home, grid_away = np.meshgrid(goals, goals, indexing="ij")
    totals = grid_home + grid_away

    probabilities = {
        "home": float(matrix[grid_home > grid_away].sum()),
        "draw": float(matrix[grid_home == grid_away].sum()),
        "away": float(matrix[grid_home < grid_away].sum()),
        "btts": float(matrix[(grid_home > 0) & (grid_away > 0)].sum()),
    }
    for line in goal_lines:
        over = float(matrix[totals > line].sum())
        probabilities[f"over{line}"] = over
        probabilities[f"under{line}"] = 1 - over
    return probabilities


def predict(model: DixonColesFit, home_id: str, away_id: str) -> dict[str, float]:
    """Convenience wrapper: fixture in, market probabilities out."""
    return market_probabilities(score_matrix(model, home_id, away_id))


def main() -> None:
    matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
    model = fit(matches)

    print(f"Fitted on {model.n_matches} matches through "f"{model.fitted_through.date()}")
    print(f"Home advantage: {model.home_advantage:.3f}   rho: {model.rho:.3f}\n")

    # Current-season teams only, strongest attack first.
    current = matches[matches["season"] == matches["season"].max()]
    current_teams = sorted(set(current["home_id"]) | set(current["away_id"]))
    ratings = pd.DataFrame({
        "attack": {t: model.attack[t] for t in current_teams},
        "defence": {t: model.defence[t] for t in current_teams},
    }).sort_values("attack", ascending=False).round(3)
    print(ratings.to_string())

    home, away = "arsenal", "chelsea"
    print(f"\nExample fixture: {home} vs {away}")
    for market, probability in predict(model, home, away).items():
        print(f"  {market:<10} {probability:6.1%}")


if __name__ == "__main__":
    main()