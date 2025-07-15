import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
from pettingzoo.classic import connect_four_v3
import os
from collections import deque

class GRPONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(84, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 7)
        )
    
    def forward(self, x, action_mask=None):
        logits = self.body(x.flatten(1).float())
        
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)
        
        return logits

class PopulationBuffer:
    def __init__(self, size=1000):
        self.buffer = deque(maxlen=size)
        self.size = size
    
    def add_episode(self, episode_return):
        self.buffer.append(episode_return)
    
    def get_baseline(self):
        if len(self.buffer) == 0:
            return 0.0
        return np.mean(self.buffer)
    
    def get_relative_reward(self, episode_return):
        baseline = self.get_baseline()
        return episode_return - baseline

def selfplay_episode(model, population_buffer=None):
    env = connect_four_v3.env(render_mode=None)
    env.reset()
    memory = []
    
    for agent in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        
        if term or trunc:
            action = None
        else:
            action_mask = torch.tensor(obs['action_mask']).unsqueeze(0)
            state = torch.tensor(obs['observation']).unsqueeze(0)
            
            with torch.no_grad():
                logits = model(state, action_mask)
                probs = torch.softmax(logits, dim=-1)
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample().item()
                log_prob = action_dist.log_prob(torch.tensor(action))
            
            memory.append((state, action, log_prob, agent))
        
        env.step(action)
    
    final_rewards = env.rewards
    
    episode_returns = {}
    for agent in ['player_0', 'player_1']:
        if agent in final_rewards:
            if final_rewards[agent] == 1:
                episode_returns[agent] = 1.0
            elif final_rewards[agent] == -1:
                episode_returns[agent] = -1.0
            else:
                episode_returns[agent] = 0.0
        else:
            episode_returns[agent] = 0.0
    
    if population_buffer is not None:
        for agent_return in episode_returns.values():
            population_buffer.add_episode(agent_return)
    
    processed_memory = []
    for state, action, log_prob, player in memory:
        episode_return = episode_returns[player]
        
        if population_buffer is not None:
            relative_reward = population_buffer.get_relative_reward(episode_return)
        else:
            relative_reward = episode_return
        
        processed_memory.append((state, action, log_prob, relative_reward))
    
    return processed_memory

def grpo_update(model, optimizer, batch, entropy_coef=0.01, baseline_coef=0.1):
    if not batch:
        return 0.0
    
    states, actions, old_log_probs, relative_rewards = zip(*batch)
    
    states = torch.cat(states)
    actions = torch.tensor(actions)
    old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
    relative_rewards = torch.tensor(relative_rewards, dtype=torch.float32)
    
    relative_rewards = (relative_rewards - relative_rewards.mean()) / (relative_rewards.std() + 1e-8)
    
    logits = model(states)
    log_probs = torch.log_softmax(logits, dim=-1)
    action_log_probs = log_probs[range(len(actions)), actions]
    
    policy_loss = -(action_log_probs * relative_rewards).mean()
    
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    entropy_loss = -entropy_coef * entropy
    
    baseline_loss = baseline_coef * torch.mean(relative_rewards ** 2)
    
    total_loss = policy_loss + entropy_loss + baseline_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    
    return total_loss.item()

def tournament_evaluation(model1, model2, num_games=20):
    wins_model1 = 0
    wins_model2 = 0
    draws = 0
    
    for game in range(num_games):
        env = connect_four_v3.env(render_mode=None)
        env.reset()
        
        for agent in env.agent_iter():
            obs, reward, term, trunc, _ = env.last()
            
            if term or trunc:
                action = None
            else:
                action_mask = torch.tensor(obs['action_mask']).unsqueeze(0)
                state = torch.tensor(obs['observation']).unsqueeze(0)
                
                if (game % 2 == 0 and agent == 'player_0') or (game % 2 == 1 and agent == 'player_1'):
                    current_model = model1
                else:
                    current_model = model2
                
                with torch.no_grad():
                    logits = current_model(state, action_mask)
                    probs = torch.softmax(logits, dim=-1)
                    action = torch.multinomial(probs, 1).item()
            
            env.step(action)
        
        final_rewards = env.rewards
        
        if game % 2 == 0:
            if final_rewards['player_0'] == 1:
                wins_model1 += 1
            elif final_rewards['player_0'] == -1:
                wins_model2 += 1
            else:
                draws += 1
        else:
            if final_rewards['player_1'] == 1:
                wins_model1 += 1
            elif final_rewards['player_1'] == -1:
                wins_model2 += 1
            else:
                draws += 1
    
    return wins_model1, draws, wins_model2

def evaluate_vs_random(model, num_games=10):
    wins = 0
    draws = 0
    losses = 0
    
    for _ in range(num_games):
        env = connect_four_v3.env(render_mode=None)
        env.reset()
        
        for agent in env.agent_iter():
            obs, reward, term, trunc, _ = env.last()
            
            if term or trunc:
                action = None
            else:
                if agent == 'player_0':
                    action_mask = torch.tensor(obs['action_mask']).unsqueeze(0)
                    state = torch.tensor(obs['observation']).unsqueeze(0)
                    
                    with torch.no_grad():
                        logits = model(state, action_mask)
                        probs = torch.softmax(logits, dim=-1)
                        action = torch.multinomial(probs, 1).item()
                else:
                    valid_actions = [i for i, valid in enumerate(obs['action_mask']) if valid]
                    action = np.random.choice(valid_actions)
            
            env.step(action)
        
        if env.rewards['player_0'] == 1:
            wins += 1
        elif env.rewards['player_0'] == -1:
            losses += 1
        else:
            draws += 1
    
    return wins, draws, losses

def main():
    model = GRPONet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    population_buffer = PopulationBuffer(size=1000)
    
    os.makedirs('checkpoints_grpo', exist_ok=True)
    
    print("Starting Connect-4 GRPO Training...")
    print("=" * 50)
    
    for epoch in trange(1000, desc="GRPO Training"):
        batch = []
        
        for _ in range(32):
            episode_data = selfplay_episode(model, population_buffer)
            batch.extend(episode_data)
        
        loss = grpo_update(model, optimizer, batch)
        
        if epoch % 100 == 0:
            torch.save(model.state_dict(), f'checkpoints_grpo/grpo_ckpt_{epoch}.pt')
            
            wins, draws, losses = evaluate_vs_random(model, num_games=20)
            win_rate = wins / (wins + draws + losses)
            baseline = population_buffer.get_baseline()
            
            print(f"\nEpoch {epoch}:")
            print(f"  Loss: {loss:.4f}")
            print(f"  Population Baseline: {baseline:.3f}")
            print(f"  vs Random - Wins: {wins}, Draws: {draws}, Losses: {losses}")
            print(f"  Win Rate: {win_rate:.3f}")
            print("-" * 30)
    
    print("\nGRPO Training completed!")
    print("Final model saved as 'checkpoints_grpo/grpo_ckpt_999.pt'")

if __name__ == "__main__":
    main() 