import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
from pettingzoo.classic import connect_four_v3
import os

class Net(nn.Module):
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

def selfplay_episode(model):
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
                action = torch.multinomial(probs, 1).item()
            
            memory.append((state, action, agent))
        
        env.step(action)
    
    final_rewards = env.rewards
    
    processed_memory = []
    for state, action, player in memory:
        if player in final_rewards:
            if final_rewards[player] == 1:
                reward = 1.0
            elif final_rewards[player] == -1:
                reward = -1.0
            else:
                reward = 0.0
        else:
            reward = 0.0
        
        processed_memory.append((state, action, reward))
    
    return processed_memory

def train_step(batch, policy, optimizer):
    if not batch:
        return 0.0
    
    states, actions, rewards = zip(*batch)
    
    states = torch.cat(states)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    
    logits = policy(states)
    log_probs = torch.log_softmax(logits, dim=-1)
    
    policy_loss = -(log_probs[range(len(actions)), actions] * rewards).mean()
    
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    
    return policy_loss.item()

def evaluate_policy(model, num_games=10):
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
    policy = Net()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    os.makedirs('checkpoints', exist_ok=True)
    
    print("Starting Connect-4 Self-Play Training...")
    print("=" * 50)
    
    for epoch in trange(1000, desc="Training"):
        batch = []
        for _ in range(32):
            episode_data = selfplay_episode(policy)
            batch.extend(episode_data)
        
        loss = train_step(batch, policy, optimizer)
        
        if epoch % 100 == 0:
            torch.save(policy.state_dict(), f'checkpoints/ckpt_{epoch}.pt')
            
            wins, draws, losses = evaluate_policy(policy, num_games=20)
            win_rate = wins / (wins + draws + losses)
            
            print(f"\nEpoch {epoch}:")
            print(f"  Loss: {loss:.4f}")
            print(f"  vs Random - Wins: {wins}, Draws: {draws}, Losses: {losses}")
            print(f"  Win Rate: {win_rate:.3f}")
            print("-" * 30)
    
    print("\nTraining completed!")
    print("Final model saved as 'checkpoints/ckpt_999.pt'")

if __name__ == "__main__":
    main() 