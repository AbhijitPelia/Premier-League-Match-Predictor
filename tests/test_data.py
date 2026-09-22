"""Sanity checks on the cleaned matches table.

Run from the project root:
    pytest
"""
import pandas as pd
import pytest

from src.clean import ODDS_SOURCES, build_matches
from src.ingest import SEASONS

CURRENT_SEASON = SEASONS[-1]
COMPLETED_SEASONS = SEASONS[:-1]


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    """Build the table once and share it across every test in this file."""
    return build_matches()


def test_completed_seasons_have_380_matches(matches):
    counts = matches.groupby("season").size()
    for season in COMPLETED_SEASONS:
        assert counts[season] == 380, f"{season} has {counts[season]} matches, expected 380"


def test_current_season_is_plausible(matches):
    count = matches.groupby("season").size()[CURRENT_SEASON]
    assert 0 < count <= 380, f"{CURRENT_SEASON} has {count} matches"


def test_match_ids_are_unique(matches):
    assert not matches["match_id"].duplicated().any()


def test_no_missing_core_values(matches):
    core = [
        "date", "home_id", "away_id",
        "home_goals", "away_goals", "home_corners", "away_corners",
    ]
    missing = matches[core].isna().sum()
    assert missing.sum() == 0, f"Missing values found:\n{missing[missing > 0]}"


def test_goals_and_corners_are_in_sane_ranges(matches):
    assert matches["home_goals"].between(0, 15).all()
    assert matches["away_goals"].between(0, 15).all()
    assert matches["home_corners"].between(0, 30).all()
    assert matches["away_corners"].between(0, 30).all()


def test_season_averages_are_plausible(matches):
    """Catches a shifted or mismatched column, which would skew these badly."""
    totals = matches["home_goals"] + matches["away_goals"]
    goals = totals.groupby(matches["season"]).mean()
    assert goals.between(2.3, 3.4).all(), f"Implausible goal averages:\n{goals}"

    corner_totals = matches["home_corners"] + matches["away_corners"]
    corners = corner_totals.groupby(matches["season"]).mean()
    assert corners.between(8.5, 12.5).all(), f"Implausible corner averages:\n{corners}"


def test_home_advantage_exists(matches):
    """Home teams have outscored away teams every season in PL history."""
    home_wins = (matches["home_goals"] > matches["away_goals"]).mean()
    away_wins = (matches["home_goals"] < matches["away_goals"]).mean()
    assert home_wins > away_wins, "No home advantage: home/away columns may be swapped"


def test_no_team_plays_itself(matches):
    assert (matches["home_id"] != matches["away_id"]).all()


def test_matches_are_in_chronological_order(matches):
    assert matches["date"].is_monotonic_increasing


def test_odds_are_valid_prices(matches):
    """Decimal odds are always above 1.0; anything else means a bad column."""
    for column in ODDS_SOURCES:
        values = matches[column].dropna()
        assert (values > 1.0).all(), f"{column} has prices at or below 1.0"


def test_1x2_odds_cover_every_season(matches):
    coverage = matches["odds_home_close"].notna().groupby(matches["season"]).mean()
    assert (coverage == 1.0).all(), f"Gaps in 1X2 closing odds:\n{coverage[coverage < 1]}"


def test_over_under_odds_cover_recent_seasons(matches):
    """Over/under prices only start in 2019-20; earlier seasons are expected gaps."""
    recent = matches[matches["season"] >= "1920"]
    coverage = recent["odds_over25_close"].notna().groupby(recent["season"]).mean()
    assert (coverage == 1.0).all(), f"Gaps in over/under odds:\n{coverage[coverage < 1]}"