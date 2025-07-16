"""connect-4 agents package

this package contains all agent implementations for the connect-4 tournament.
"""

from .base import Agent
from .random_agent import build_agent as build_random
from .minimax_agent import build_agent as build_minimax
from .mcts_agent import build_agent as build_mcts
from .tabular_q import build_agent as build_tabular_q
from .selfplay_pg import build_agent as build_selfplay_pg
from .grpo_agent import build_agent as build_grpo
from .ppo_agent import build_agent as build_ppo
from .human_cli import build_agent as build_human

__all__ = [
    "Agent",
    "build_random",
    "build_minimax", 
    "build_mcts",
    "build_tabular_q",
    "build_selfplay_pg",
    "build_grpo",
    "build_ppo",
    "build_human"
] 