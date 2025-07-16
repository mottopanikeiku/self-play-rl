#!/usr/bin/env python3
"""Training Script for RL Agents

Trains SelfPlayPG, GRPO, and PPO agents using self-play with progress tracking
and logging.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from env.connect4 import Connect4Env
from agents.selfplay_pg import SelfPlayPGAgent
from agents.grpo_agent import GRPOAgent
from agents.ppo_agent import PPOAgent


def self_play_episode(agent, env: Connect4Env) -> Dict[str, Any]:
    """Play one self-play episode and return statistics."""
    env.reset()
    agent.reset()
    
    episode_length = 0
    total_reward = 0.0
    
    while True:
        # Current player makes a move
        obs = env._get_observation()
        action = agent.act(obs)
        
        # Take step in environment
        next_obs, reward, done, _, info = env.step(action)
        episode_length += 1
        
        # Add reward to agent
        if hasattr(agent, 'add_reward'):
            agent.add_reward(reward, done)
        
        total_reward += reward
        
        if done:
            break
    
    # End episode for agent
    if hasattr(agent, 'end_episode'):
        agent.end_episode()
    
    return {
        'episode_length': episode_length,
        'total_reward': total_reward,
        'winner': env.winner,
        'moves': episode_length
    }


def train_agent(
    agent_name: str,
    agent_class,
    num_episodes: int = 50000,
    eval_interval: int = 1000,
    save_interval: int = 5000
) -> None:
    """Train a single agent using self-play."""
    print(f"\n{'='*60}")
    print(f"Training {agent_name}")
    print(f"{'='*60}")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    agent = agent_class(device=device)
    agent.set_training_mode(True)
    
    env = Connect4Env()
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Logging
    log_file = data_dir / "training_logs.jsonl"
    
    # Training loop
    stats = []
    episode_rewards = []
    episode_lengths = []
    win_counts = [0, 0, 0]  # [draws, player1_wins, player2_wins]
    
    with tqdm(total=num_episodes, desc=f"Training {agent_name}", unit="episode") as pbar:
        for episode in range(num_episodes):
            start_time = time.time()
            
            # Play episode
            episode_stats = self_play_episode(agent, env)
            
            # Track statistics
            episode_rewards.append(episode_stats['total_reward'])
            episode_lengths.append(episode_stats['episode_length'])
            
            winner = episode_stats['winner']
            if winner == 0:
                win_counts[0] += 1  # Draw
            elif winner == 1:
                win_counts[1] += 1  # Player 1
            else:
                win_counts[2] += 1  # Player 2
            
            episode_time = time.time() - start_time
            
            # Update progress bar
            pbar.set_postfix({
                'Avg_Reward': f"{np.mean(episode_rewards[-100:]):.3f}",
                'Avg_Length': f"{np.mean(episode_lengths[-100:]):.1f}",
                'Win_Rate': f"{win_counts[1]/(episode+1):.3f}",
                'EPS': f"{1.0/episode_time:.1f}"
            })
            pbar.update(1)
            
            # Evaluation and logging
            if (episode + 1) % eval_interval == 0:
                # Calculate recent performance
                recent_rewards = episode_rewards[-eval_interval:]
                recent_lengths = episode_lengths[-eval_interval:]
                recent_wins = win_counts[1] - (0 if episode < eval_interval else stats[-1]['player1_wins'])
                
                eval_stats = {
                    'agent': agent_name,
                    'episode': episode + 1,
                    'avg_reward': float(np.mean(recent_rewards)),
                    'avg_length': float(np.mean(recent_lengths)),
                    'win_rate_p1': recent_wins / eval_interval,
                    'total_wins_p1': win_counts[1],
                    'total_draws': win_counts[0],
                    'total_wins_p2': win_counts[2],
                    'player1_wins': win_counts[1],
                    'device': device,
                    'timestamp': time.time()
                }
                
                stats.append(eval_stats)
                
                # Log to file
                with open(log_file, 'a') as f:
                    f.write(json.dumps(eval_stats) + '\n')
                
                # Print evaluation
                print(f"\nEvaluation at episode {episode + 1}:")
                print(f"  Avg Reward: {eval_stats['avg_reward']:.4f}")
                print(f"  Avg Length: {eval_stats['avg_length']:.1f}")
                print(f"  Win Rate P1: {eval_stats['win_rate_p1']:.3f}")
                print(f"  Total Games: {episode + 1}")
            
            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                model_path = data_dir / f"{agent_name.lower()}.pt"
                agent.save_model(model_path)
                print(f"\nSaved checkpoint: {model_path}")
    
    # Final save
    final_model_path = data_dir / f"{agent_name.lower()}.pt"
    agent.save_model(final_model_path)
    print(f"\nTraining completed! Final model saved: {final_model_path}")
    
    # Final statistics
    print(f"\nFinal Statistics for {agent_name}:")
    print(f"  Total Episodes: {num_episodes}")
    print(f"  Final Win Rate P1: {win_counts[1]/num_episodes:.3f}")
    print(f"  Final Avg Reward: {np.mean(episode_rewards[-1000:]):.4f}")
    print(f"  Final Avg Length: {np.mean(episode_lengths[-1000:]):.1f}")


def main() -> None:
    """Main training function."""
    print("Connect-4 RL Agent Training")
    print("="*60)
    
    # Check for CUDA
    if torch.cuda.is_available():
        print(f"CUDA available! Using GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA not available. Using CPU.")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    # Training configuration
    training_config = {
        'num_episodes': 50000,
        'eval_interval': 1000,
        'save_interval': 5000
    }
    
    # Train agents in sequence
    agents_to_train = [
        ("SelfPlayPG", SelfPlayPGAgent),
        ("GRPO", GRPOAgent),
        ("PPO", PPOAgent)
    ]
    
    start_time = time.time()
    
    for agent_name, agent_class in agents_to_train:
        agent_start = time.time()
        
        try:
            train_agent(
                agent_name=agent_name,
                agent_class=agent_class,
                **training_config
            )
            
            agent_time = time.time() - agent_start
            print(f"\n{agent_name} training completed in {agent_time/3600:.2f} hours")
            
        except KeyboardInterrupt:
            print(f"\nTraining interrupted for {agent_name}")
            break
        except Exception as e:
            print(f"\nError training {agent_name}: {e}")
            continue
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"All training completed in {total_time/3600:.2f} hours")
    print(f"Models saved in: {Path('data').absolute()}")
    print(f"Training logs: {Path('data/training_logs.jsonl').absolute()}")


if __name__ == "__main__":
    main() 