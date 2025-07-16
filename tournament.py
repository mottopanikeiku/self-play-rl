import torch
import numpy as np
from pettingzoo.classic import connect_four_v3
from connect4_ppo import PPONet
from connect4_grpo import GRPONet
import os
import argparse

def load_ppo_model(checkpoint_path):
    model = PPONet()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def load_grpo_model(checkpoint_path):
    model = GRPONet()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def get_action_ppo(model, state, action_mask):
    with torch.no_grad():
        logits, _ = model(state, action_mask)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
    return action

def get_action_grpo(model, state, action_mask):
    with torch.no_grad():
        logits = model(state, action_mask)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
    return action

def play_match(model1, model2, model1_type, model2_type, num_games=100, verbose=False):
    model1_wins = 0
    model2_wins = 0
    draws = 0
    
    for game in range(num_games):
        env = connect_four_v3.env(render_mode=None)
        env.reset()
        
        if verbose and game % 10 == 0:
            print(f"Game {game}/{num_games}")
        
        for agent in env.agent_iter():
            obs, reward, term, trunc, _ = env.last()
            
            if term or trunc:
                action = None
            else:
                action_mask = torch.tensor(obs['action_mask']).unsqueeze(0)
                state = torch.tensor(obs['observation']).unsqueeze(0)
                
                if (game % 2 == 0 and agent == 'player_0') or (game % 2 == 1 and agent == 'player_1'):
                    if model1_type == 'ppo':
                        action = get_action_ppo(model1, state, action_mask)
                    else:
                        action = get_action_grpo(model1, state, action_mask)
                else:
                    if model2_type == 'ppo':
                        action = get_action_ppo(model2, state, action_mask)
                    else:
                        action = get_action_grpo(model2, state, action_mask)
            
            env.step(action)
        
        final_rewards = env.rewards
        
        if game % 2 == 0:
            if final_rewards['player_0'] == 1:
                model1_wins += 1
            elif final_rewards['player_0'] == -1:
                model2_wins += 1
            else:
                draws += 1
        else:
            if final_rewards['player_1'] == 1:
                model1_wins += 1
            elif final_rewards['player_1'] == -1:
                model2_wins += 1
            else:
                draws += 1
    
    return model1_wins, draws, model2_wins

def find_latest_checkpoint(directory, prefix):
    if not os.path.exists(directory):
        return None
    
    checkpoints = []
    for file in os.listdir(directory):
        if file.startswith(prefix) and file.endswith('.pt'):
            try:
                epoch = int(file.split('_')[-1].split('.')[0])
                checkpoints.append((epoch, os.path.join(directory, file)))
            except ValueError:
                continue
    
    if not checkpoints:
        return None
    
    checkpoints.sort(reverse=True)
    return checkpoints[0][1]

def run_tournament():
    ppo_checkpoint = find_latest_checkpoint('checkpoints_ppo', 'ppo_ckpt')
    grpo_checkpoint = find_latest_checkpoint('checkpoints_grpo', 'grpo_ckpt')
    
    if ppo_checkpoint is None:
        print("No PPO checkpoint found! Please train PPO first.")
        return
    
    if grpo_checkpoint is None:
        print("No GRPO checkpoint found! Please train GRPO first.")
        return
    
    print("Loading models...")
    ppo_model = load_ppo_model(ppo_checkpoint)
    grpo_model = load_grpo_model(grpo_checkpoint)
    
    print(f"PPO model: {ppo_checkpoint}")
    print(f"GRPO model: {grpo_checkpoint}")
    print("=" * 60)
    
    print("Running PPO vs GRPO tournament...")
    ppo_wins, draws, grpo_wins = play_match(
        ppo_model, grpo_model, 'ppo', 'grpo', 
        num_games=100, verbose=True
    )
    
    total_games = ppo_wins + draws + grpo_wins
    ppo_winrate = ppo_wins / total_games
    grpo_winrate = grpo_wins / total_games
    draw_rate = draws / total_games
    
    print("\nTournament Results:")
    print("=" * 60)
    print(f"PPO Wins:     {ppo_wins:3d} ({ppo_winrate:.1%})")
    print(f"GRPO Wins:    {grpo_wins:3d} ({grpo_winrate:.1%})")
    print(f"Draws:        {draws:3d} ({draw_rate:.1%})")
    print(f"Total Games:  {total_games}")
    
    if ppo_wins > grpo_wins:
        print(f"\nWinner: PPO (margin: {ppo_wins - grpo_wins} games)")
    elif grpo_wins > ppo_wins:
        print(f"\nWinner: GRPO (margin: {grpo_wins - ppo_wins} games)")
    else:
        print(f"\nResult: Tie!")

def interactive_match():
    ppo_checkpoint = find_latest_checkpoint('checkpoints_ppo', 'ppo_ckpt')
    grpo_checkpoint = find_latest_checkpoint('checkpoints_grpo', 'grpo_ckpt')
    
    models = {}
    
    if ppo_checkpoint:
        models['ppo'] = load_ppo_model(ppo_checkpoint)
        print(f"Loaded PPO model: {ppo_checkpoint}")
    
    if grpo_checkpoint:
        models['grpo'] = load_grpo_model(grpo_checkpoint)
        print(f"Loaded GRPO model: {grpo_checkpoint}")
    
    if len(models) < 2:
        print("Need at least 2 trained models for interactive match!")
        return
    
    model_names = list(models.keys())
    print(f"\nAvailable models: {model_names}")
    
    print("\nSelect models for match:")
    model1_name = input(f"Model 1 ({'/'.join(model_names)}): ").strip().lower()
    model2_name = input(f"Model 2 ({'/'.join(model_names)}): ").strip().lower()
    
    if model1_name not in models or model2_name not in models:
        print("Invalid model selection!")
        return
    
    num_games = int(input("Number of games (default 20): ") or 20)
    
    print(f"\nRunning {model1_name.upper()} vs {model2_name.upper()}...")
    
    model1_wins, draws, model2_wins = play_match(
        models[model1_name], models[model2_name], 
        model1_name, model2_name, 
        num_games=num_games, verbose=True
    )
    
    total_games = model1_wins + draws + model2_wins
    
    print(f"\nMatch Results:")
    print("=" * 40)
    print(f"{model1_name.upper()} Wins: {model1_wins:3d} ({model1_wins/total_games:.1%})")
    print(f"{model2_name.upper()} Wins: {model2_wins:3d} ({model2_wins/total_games:.1%})")
    print(f"Draws:       {draws:3d} ({draws/total_games:.1%})")

def main():
    parser = argparse.ArgumentParser(description='Connect-4 Model Tournament')
    parser.add_argument('--mode', choices=['tournament', 'interactive'], 
                       default='tournament', help='Tournament mode')
    
    args = parser.parse_args()
    
    if args.mode == 'tournament':
        run_tournament()
    else:
        interactive_match()

if __name__ == "__main__":
    main() 