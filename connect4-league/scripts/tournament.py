#!/usr/bin/env python3
"""Tournament Engine

Runs round-robin tournaments between all Connect-4 agents with Elo rating system.
"""

import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import yaml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from env.connect4 import Connect4Env
from agents.random_agent import build_agent as build_random
from agents.minimax_agent import build_agent as build_minimax
from agents.mcts_agent import build_agent as build_mcts
from agents.tabular_q import build_agent as build_tabular_q
from agents.selfplay_pg import build_agent as build_selfplay_pg
from agents.grpo_agent import build_agent as build_grpo
from agents.ppo_agent import build_agent as build_ppo


def play_game(args: Tuple[str, str, Path, Path, int, bool]) -> Dict[str, Any]:
    """Play a single game between two agents."""
    agent1_name, agent2_name, agent1_path, agent2_path, game_id, agent1_starts = args
    
    # Build agents
    agent_builders = {
        'random': build_random,
        'minimax': build_minimax,
        'mcts': build_mcts,
        'tabular_q': build_tabular_q,
        'selfplay_pg': build_selfplay_pg,
        'grpo': build_grpo,
        'ppo': build_ppo
    }
    
    agent1 = agent_builders[agent1_name](agent1_path)
    agent2 = agent_builders[agent2_name](agent2_path)
    
    # Setup environment
    env = Connect4Env()
    env.reset()
    
    # Reset agents
    agent1.reset()
    agent2.reset()
    
    # Determine starting order
    if agent1_starts:
        current_agent = agent1
        current_name = agent1_name
        other_agent = agent2
        other_name = agent2_name
        current_player = 1
    else:
        current_agent = agent2
        current_name = agent2_name
        other_agent = agent1
        other_name = agent1_name
        current_player = 2
    
    move_count = 0
    max_moves = 42  # Maximum possible moves in Connect-4
    
    # Game loop
    while move_count < max_moves:
        # Get observation for current player
        obs = env._get_observation()
        
        # Select action
        try:
            action = current_agent.act(obs)
        except Exception as e:
            # If agent fails, forfeit
            winner = 3 - current_player  # Other player wins
            break
        
        # Take step
        try:
            next_obs, reward, done, _, info = env.step(action)
            move_count += 1
            
            if done:
                winner = env.winner
                break
                
        except Exception as e:
            # Invalid move, forfeit
            winner = 3 - current_player  # Other player wins
            break
        
        # Switch players
        current_agent, other_agent = other_agent, current_agent
        current_name, other_name = other_name, current_name
        current_player = 3 - current_player
    else:
        # Maximum moves reached (shouldn't happen in Connect-4, but safety check)
        winner = 0  # Draw
    
    # Determine result
    if winner == 0:
        result = 'draw'
        agent1_score = 0.5
        agent2_score = 0.5
    elif (winner == 1 and agent1_starts) or (winner == 2 and not agent1_starts):
        result = 'agent1_win'
        agent1_score = 1.0
        agent2_score = 0.0
    else:
        result = 'agent2_win'
        agent1_score = 0.0
        agent2_score = 1.0
    
    return {
        'game_id': game_id,
        'agent1': agent1_name,
        'agent2': agent2_name,
        'agent1_starts': agent1_starts,
        'result': result,
        'winner': winner,
        'moves': move_count,
        'agent1_score': agent1_score,
        'agent2_score': agent2_score
    }


def calculate_elo_update(rating1: float, rating2: float, score1: float, k: float = 20.0) -> Tuple[float, float]:
    """Calculate Elo rating updates."""
    expected1 = 1.0 / (1.0 + 10**((rating2 - rating1) / 400))
    expected2 = 1.0 - expected1
    
    new_rating1 = rating1 + k * (score1 - expected1)
    new_rating2 = rating2 + k * ((1.0 - score1) - expected2)
    
    return new_rating1, new_rating2


def run_tournament(config: Dict[str, Any]) -> None:
    """Run the full tournament."""
    print("Connect-4 Tournament Engine")
    print("=" * 60)
    
    # Setup
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Available agents and their model paths
    agents = {
        'random': None,
        'minimax': None,
        'mcts': None,
        'tabular_q': data_dir / "tabular_q.pkl" if (data_dir / "tabular_q.pkl").exists() else None,
        'selfplay_pg': data_dir / "selfplaypg.pt" if (data_dir / "selfplaypg.pt").exists() else None,
        'grpo': data_dir / "grpo.pt" if (data_dir / "grpo.pt").exists() else None,
        'ppo': data_dir / "ppo.pt" if (data_dir / "ppo.pt").exists() else None
    }
    
    # Filter to only available agents
    available_agents = {name: path for name, path in agents.items() 
                       if path is None or path.exists() or name in ['random', 'minimax', 'mcts']}
    
    if len(available_agents) < 2:
        print("Error: Need at least 2 agents to run tournament")
        print(f"Available agents: {list(available_agents.keys())}")
        return
    
    print(f"Tournament participants: {list(available_agents.keys())}")
    print(f"Games per matchup: {config['games_per_pair']}")
    print(f"Using {config.get('num_processes', mp.cpu_count())} processes")
    
    # Initialize Elo ratings
    elo_ratings = {name: 1200.0 for name in available_agents.keys()}
    elo_history = []
    
    # Generate all matchups
    agent_names = list(available_agents.keys())
    all_games = []
    
    for i, agent1 in enumerate(agent_names):
        for j, agent2 in enumerate(agent_names):
            if i != j:  # Don't play against self
                for game_num in range(config['games_per_pair']):
                    # Alternate who starts
                    agent1_starts = game_num % 2 == 0
                    
                    game_args = (
                        agent1, agent2,
                        available_agents[agent1], available_agents[agent2],
                        len(all_games), agent1_starts
                    )
                    all_games.append(game_args)
    
    print(f"Total games to play: {len(all_games)}")
    
    # Play games using multiprocessing
    start_time = time.time()
    
    with mp.Pool(processes=config.get('num_processes', mp.cpu_count())) as pool:
        results = []
        
        # Use tqdm for progress tracking
        with tqdm(total=len(all_games), desc="Playing games", unit="game") as pbar:
            for result in pool.imap_unordered(play_game, all_games):
                results.append(result)
                pbar.update(1)
                
                # Update progress info
                if len(results) % 100 == 0:
                    elapsed = time.time() - start_time
                    games_per_sec = len(results) / elapsed
                    eta = (len(all_games) - len(results)) / games_per_sec
                    pbar.set_postfix({
                        'Games/sec': f"{games_per_sec:.1f}",
                        'ETA': f"{eta/60:.1f}min"
                    })
    
    tournament_time = time.time() - start_time
    print(f"\nTournament completed in {tournament_time:.1f} seconds")
    print(f"Average: {len(all_games)/tournament_time:.1f} games/second")
    
    # Process results
    print("\nProcessing results...")
    
    # Aggregate results by matchup
    matchup_results = {}
    game_results = []
    
    for result in results:
        agent1, agent2 = result['agent1'], result['agent2']
        key = (agent1, agent2)
        
        if key not in matchup_results:
            matchup_results[key] = {
                'agent1': agent1,
                'agent2': agent2,
                'wins': 0,
                'losses': 0,
                'draws': 0,
                'total_games': 0
            }
        
        matchup_results[key]['total_games'] += 1
        
        if result['result'] == 'agent1_win':
            matchup_results[key]['wins'] += 1
        elif result['result'] == 'agent2_win':
            matchup_results[key]['losses'] += 1
        else:
            matchup_results[key]['draws'] += 1
        
        game_results.append(result)
    
    # Calculate final Elo ratings
    print("Calculating Elo ratings...")
    
    # Sort games by game_id to ensure consistent order
    game_results.sort(key=lambda x: x['game_id'])
    
    for result in game_results:
        agent1, agent2 = result['agent1'], result['agent2']
        score1 = result['agent1_score']
        
        old_rating1 = elo_ratings[agent1]
        old_rating2 = elo_ratings[agent2]
        
        new_rating1, new_rating2 = calculate_elo_update(
            old_rating1, old_rating2, score1, k=config.get('elo_k', 20)
        )
        
        elo_ratings[agent1] = new_rating1
        elo_ratings[agent2] = new_rating2
        
        # Record history periodically
        if result['game_id'] % 100 == 0:
            elo_history.append({
                'game': result['game_id'],
                'ratings': elo_ratings.copy()
            })
    
    # Final Elo snapshot
    elo_history.append({
        'game': len(game_results),
        'ratings': elo_ratings.copy()
    })
    
    # Save results
    results_file = data_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(list(matchup_results.values()), f, indent=2)
    
    # Save Elo history
    elo_file = data_dir / "elo.csv"
    with open(elo_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['game'] + sorted(agent_names)
        writer.writerow(header)
        
        # Data
        for entry in elo_history:
            row = [entry['game']]
            for agent in sorted(agent_names):
                row.append(entry['ratings'][agent])
            writer.writerow(row)
    
    # Print final results
    print("\n" + "=" * 60)
    print("TOURNAMENT RESULTS")
    print("=" * 60)
    
    # Sort agents by Elo rating
    sorted_agents = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    
    print("\nFinal Elo Ratings:")
    print("-" * 30)
    for rank, (agent, rating) in enumerate(sorted_agents, 1):
        print(f"{rank:2d}. {agent:<12} {rating:7.1f}")
    
    # Print head-to-head results
    print("\nHead-to-Head Results:")
    print("-" * 40)
    for key, stats in matchup_results.items():
        agent1, agent2 = stats['agent1'], stats['agent2']
        wins, losses, draws = stats['wins'], stats['losses'], stats['draws']
        total = stats['total_games']
        win_rate = wins / total if total > 0 else 0
        
        print(f"{agent1} vs {agent2}: {wins}-{losses}-{draws} ({win_rate:.3f})")
    
    print(f"\nResults saved to: {results_file}")
    print(f"Elo history saved to: {elo_file}")


def load_config() -> Dict[str, Any]:
    """Load tournament configuration."""
    config_file = Path("configs/tournament.yaml")
    
    # Default configuration
    default_config = {
        'games_per_pair': 100,
        'num_processes': mp.cpu_count(),
        'elo_k': 20,
        'time_limit_ms': 500
    }
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
            default_config.update(file_config)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            print("Using default configuration")
    else:
        print(f"Config file {config_file} not found, using defaults")
    
    return default_config


def main() -> None:
    """Main tournament function."""
    # Load configuration
    config = load_config()
    
    try:
        run_tournament(config)
    except KeyboardInterrupt:
        print("\nTournament interrupted by user")
    except Exception as e:
        print(f"\nError running tournament: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 