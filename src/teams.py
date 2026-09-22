"""Canonical team IDs and name mapping across data sources.

Every team gets one internal ID (snake_case) that the rest of the code uses.
Each data source's spelling of that team is listed as an alias. When you add
a new data source, or a newly promoted team appears, add its spellings here.
"""

TEAM_ALIASES: dict[str, list[str]] = {
    "arsenal": ["Arsenal"],
    "aston_villa": ["Aston Villa"],
    "bournemouth": ["Bournemouth"],
    "brentford": ["Brentford"],
    "brighton": ["Brighton"],
    "burnley": ["Burnley"],
    "cardiff": ["Cardiff"],
    "chelsea": ["Chelsea"],
    "coventry": ["Coventry"],
    "crystal_palace": ["Crystal Palace"],
    "everton": ["Everton"],
    "fulham": ["Fulham"],
    "huddersfield": ["Huddersfield"],
    "hull": ["Hull"],
    "ipswich": ["Ipswich"],
    "leeds": ["Leeds"],
    "leicester": ["Leicester"],
    "liverpool": ["Liverpool"],
    "luton": ["Luton"],
    "man_city": ["Man City"],
    "man_united": ["Man United"],
    "middlesbrough": ["Middlesbrough"],
    "newcastle": ["Newcastle"],
    "norwich": ["Norwich"],
    "nottingham_forest": ["Nott'm Forest"],
    "sheffield_united": ["Sheffield United"],
    "southampton": ["Southampton"],
    "stoke": ["Stoke"],
    "sunderland": ["Sunderland"],
    "swansea": ["Swansea"],
    "tottenham": ["Tottenham"],
    "watford": ["Watford"],
    "west_brom": ["West Brom"],
    "west_ham": ["West Ham"],
    "wolves": ["Wolves"],
}


def _build_lookup() -> dict[str, str]:
    """Flip TEAM_ALIASES into {alias: team_id}, refusing duplicate aliases."""
    lookup: dict[str, str] = {}
    for team_id, aliases in TEAM_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                raise ValueError(
                    f"Alias {alias!r} is assigned to both "
                    f"{lookup[alias]!r} and {team_id!r}"
                )
            lookup[alias] = team_id
    return lookup


_LOOKUP = _build_lookup()


def canonical_name(name: str) -> str:
    """Return the canonical team ID for any known spelling of a team name."""
    try:
        return _LOOKUP[name.strip()]
    except KeyError:
        raise KeyError(
            f"Unknown team name {name!r}. Add it to TEAM_ALIASES in src/teams.py."
        ) from None