"""Negative binomial model for Premier League corner counts.

Fits a corners-won and corners-conceded strength per team, plus a league
baseline, home advantage, and a dispersion parameter. Produces the distribution
of total corners in a match, from which over/under lines are derived.

Corners are overdispersed relative to Poisson — the spread of counts is wider
than a Poisson with the same mean — so the marginals are negative binomial.
The dispersion parameter alpha controls this: var = mu + alpha * mu^2, so
alpha = 0 is exactly Poisson and larger alpha means more spread.

Run from the project root:
    python -m src.corners
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom

from src.clean import load_matches

MAX_CORNERS = 30          # per team; totals run to 60
DEFAULT_XI = 0.0018       # time decay per day — UNTUNED, copied from the goals model
DEFAULT_PRIOR_SD = 0.10   # UNTUNED starting point; corner ratings vary less than goal ratings
CORNER_LINES = (9.5, 10.5, 11.5)


@dataclass
class CornersFit:
    """Fitted parameters, and the information needed to reproduce the fit."""
    attack: dict[str, float]    # propensity to win corners
    defence: dict[str, float]   # propensity to concede them; positive concedes more
    baseline: float
    home_advantage: float
    alpha: float                # dispersion; 0 would be Poisson
    xi: float
    prior_sd: float
    n_matches: int
    fitted_through: pd.Timestamp

    @property
    def teams(self) -> list[str]:
        return sorted(self.attack)


def _nbinom_params(mu, alpha):
    """Convert (mean, dispersion) into the (n, p) scipy's nbinom expects."""
    n = 1.0 / alpha
    return n, n / (n + mu)


def _unpack(params, n_teams):
    """Split the flat parameter vector, centring attack and defence at zero."""
    attack = params[:n_teams]
    defence = params[n_teams:2 * n_teams]
    baseline, home_advantage, log_alpha = params[-3], params[-2], params[-1]
    return (attack - attack.mean(), defence - defence.mean(),
            baseline, home_advantage, log_alpha)


def _neg_log_posterior(params, home_idx, away_idx, home_corners, away_corners,
                       weights, n_teams, prior_sd):
    """Weighted negative log-likelihood, plus the shrinkage penalty."""
    attack, defence, baseline, home_advantage, log_alpha = _unpack(params, n_teams)

    mu_home = np.exp(baseline + attack[home_idx] + defence[away_idx] + home_advantage)
    mu_away = np.exp(baseline + attack[away_idx] + defence[home_idx])

    # alpha is fitted on a log scale so it can never go negative.
    alpha = np.exp(log_alpha)
    n, p_home = _nbinom_params(mu_home, alpha)
    _, p_away = _nbinom_params(mu_away, alpha)

    log_likelihood = (
        nbinom.logpmf(home_corners, n, p_home)
        + nbinom.logpmf(away_corners, n, p_away)
    )

    penalty = (np.sum(attack ** 2) + np.sum(defence ** 2)) / (2 * prior_sd ** 2)
    return -np.sum(weights * log_likelihood) + penalty


def fit(matches: pd.DataFrame, xi: float = DEFAULT_XI,
        prior_sd: float = DEFAULT_PRIOR_SD,
        as_of: pd.Timestamp | None = None) -> CornersFit:
    """Fit the model on every match played strictly before `as_of`."""
    if as_of is not None:
        matches = matches[matches["date"] < as_of]
    matches = matches.dropna(subset=["home_corners", "away_corners"])
    if matches.empty:
        raise ValueError("No matches with corner counts available to fit on")

    teams = sorted(set(matches["home_id"]) | set(matches["away_id"]))
    index = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    home_idx = matches["home_id"].map(index).to_numpy()
    away_idx = matches["away_id"].map(index).to_numpy()
    home_corners = matches["home_corners"].to_numpy(dtype=float)
    away_corners = matches["away_corners"].to_numpy(dtype=float)

    latest = matches["date"].max()
    days_ago = (latest - matches["date"]).dt.days.to_numpy()
    weights = np.exp(-xi * days_ago)

    initial = np.concatenate([
        np.zeros(n_teams),                            # attack
        np.zeros(n_teams),                            # defence
        [np.log(5.15), 0.15, np.log(0.07)],           # baseline, home adv, log alpha
    ])
    bounds = [(-2, 2)] * (2 * n_teams) + [(0, 3), (-1, 1), (-8, 2)]

    result = minimize(
        _neg_log_posterior,
        initial,
        args=(home_idx, away_idx, home_corners, away_corners,
              weights, n_teams, prior_sd),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000},
    )
    if not result.success:
        raise RuntimeError(f"Fit did not converge: {result.message}")

    attack, defence, baseline, home_advantage, log_alpha = _unpack(result.x, n_teams)

    return CornersFit(
        attack=dict(zip(teams, attack)),
        defence=dict(zip(teams, defence)),
        baseline=float(baseline),
        home_advantage=float(home_advantage),
        alpha=float(np.exp(log_alpha)),
        xi=xi,
        prior_sd=prior_sd,
        n_matches=len(matches),
        fitted_through=latest,
    )


def expected_corners(model: CornersFit, home_id: str, away_id: str) -> tuple[float, float]:
    """Expected corners for each side. Unknown teams fall back to league average."""
    mu_home = np.exp(
        model.baseline
        + model.attack.get(home_id, 0.0)
        + model.defence.get(away_id, 0.0)
        + model.home_advantage
    )
    mu_away = np.exp(
        model.baseline
        + model.attack.get(away_id, 0.0)
        + model.defence.get(home_id, 0.0)
    )
    return float(mu_home), float(mu_away)


def total_distribution(model: CornersFit, home_id: str, away_id: str) -> np.ndarray:
    """Probability of each possible total corner count in the match.

    The two sides' counts are treated as independent, so the distribution of
    their sum is the convolution of the two marginals. (Independence is an
    approximation: trailing teams chase the game and win more corners, which
    ties corner counts to the scoreline. See the note in docs/.)
    """
    mu_home, mu_away = expected_corners(model, home_id, away_id)
    counts = np.arange(MAX_CORNERS + 1)

    n, p_home = _nbinom_params(mu_home, model.alpha)
    _, p_away = _nbinom_params(mu_away, model.alpha)
    home_probs = nbinom.pmf(counts, n, p_home)
    away_probs = nbinom.pmf(counts, n, p_away)

    totals = np.convolve(home_probs, away_probs)
    return totals / totals.sum()  # recover the tail lost past MAX_CORNERS


def corner_probabilities(model: CornersFit, home_id: str, away_id: str,
                         lines=CORNER_LINES) -> dict[str, float]:
    """Over/under probabilities for each corner line, plus the expected total."""
    totals = total_distribution(model, home_id, away_id)
    counts = np.arange(len(totals))

    probabilities = {"expected_total": float((counts * totals).sum())}
    for line in lines:
        over = float(totals[counts > line].sum())
        probabilities[f"over{line}"] = over
        probabilities[f"under{line}"] = 1 - over
    return probabilities


def main() -> None:
    matches = load_matches()
    model = fit(matches)

    mean_corners = np.exp(model.baseline)
    print(f"Fitted on {model.n_matches} matches through {model.fitted_through.date()}")
    print(f"Baseline: {model.baseline:.3f} ({mean_corners:.2f} corners)   "
          f"Home advantage: {model.home_advantage:.3f}")
    print(f"alpha: {model.alpha:.4f}   implied var/mean at league average: "
          f"{1 + model.alpha * mean_corners:.2f}   (Poisson would be 1.00)\n")

    current = matches[matches["season"] == matches["season"].max()]
    current_teams = sorted(set(current["home_id"]) | set(current["away_id"]))
    ratings = pd.DataFrame({
        "attack": {t: model.attack[t] for t in current_teams},
        "defence": {t: model.defence[t] for t in current_teams},
    }).sort_values("attack", ascending=False).round(3)
    print(ratings.to_string())

    home, away = "arsenal", "chelsea"
    mu_home, mu_away = expected_corners(model, home, away)
    print(f"\nExample fixture: {home} vs {away}")
    print(f"  expected corners: {mu_home:.2f} - {mu_away:.2f}")
    for market, value in corner_probabilities(model, home, away).items():
        if market == "expected_total":
            print(f"  {market:<16} {value:6.2f}")
        else:
            print(f"  {market:<16} {value:6.1%}")


if __name__ == "__main__":
    main()