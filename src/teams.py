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
    "brighton": ["Brighton", "Brighton & Hove Albion", "Brighton & Hove Albion FC", "Brighton and Hove Albion"],
    "burnley": ["Burnley"],
    "cardiff": ["Cardiff"],
    "chelsea": ["Chelsea"],
    "coventry": ["Coventry"],
    "crystal_palace": ["Crystal Palace"],
    "everton": ["Everton"],
    "fulham": ["Fulham"],
    "huddersfield": ["Huddersfield"],
    "hull": ["Hull"],
    "ipswich": ["Ipswich", "Ipswich Town", "Ipswich Town FC"],
    "leeds": ["Leeds", "Leeds United", "Leeds United FC"],
    "leicester": ["Leicester"],
    "liverpool": ["Liverpool"],
    "luton": ["Luton"],
    "man_city": ["Man City", "Manchester City"],
    "man_united": ["Man United", "Manchester United"],
    "middlesbrough": ["Middlesbrough"],
    "newcastle": ["Newcastle", "Newcastle United"],
    "norwich": ["Norwich"],
    "nottingham_forest": ["Nott'm Forest", "Nottingham Forest"],
    "sheffield_united": ["Sheffield United"],
    "southampton": ["Southampton"],
    "stoke": ["Stoke"],
    "sunderland": ["Sunderland"],
    "swansea": ["Swansea"],
    "tottenham": ["Tottenham"],
    "watford": ["Watford"],
    "west_brom": ["West Brom", "West Bromwich Albion"],
    "west_ham": ["West Ham", "West Ham United"],
    "wolves": ["Wolves", "Wolverhampton Wanderers"],
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
        
def canonical_name_or_none(name: str) -> str | None:
    """Return the canonical_name for any known spelling of a team name, or None if unknown."""
    return _LOOKUP.get(name.strip())