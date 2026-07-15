"""Research-grade poker arena, policies, and statistical evaluation tools."""

from poker_research.arena import ArenaConfig, HandResult, play_hand
from poker_research.catalog import policy_factories
from poker_research.league import LeagueConfig, LeagueResult, run_league
from poker_research.policies import CallingStationPolicy, EquityValuePolicy

__all__ = [
    "ArenaConfig",
    "CallingStationPolicy",
    "EquityValuePolicy",
    "HandResult",
    "LeagueConfig",
    "LeagueResult",
    "play_hand",
    "policy_factories",
    "run_league",
]

__version__ = "0.3.0"
