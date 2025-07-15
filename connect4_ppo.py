import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange
from pettingzoo.classic import connect_four_v3
import os

class PPONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(84, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.policy_head = nn.Linear(256, 7)
        self.value_head = nn.Linear(256, 1)
    
    def forward(self, x, action_mask=None):
        shared_features = self.shared(x.flatten(1).float())
        
        logits = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)
        
        return logits, value.squeeze(-1)

def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    last_advantage = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_values[t]
            next_non_terminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t]
        
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = last_advantage = delta + gamma * lam * next_non_terminal * last_advantage
    
    returns = advantages + values
    return advantages, returns

def selfplay_episode(model):
    env = connect_four_v3.env(render_mode=None)
    env.reset()
    
    states = []
    actions = []
    rewards = []
    values = []
    log_probs = []
    action_masks = []
    dones = []
    agents = []
    
    for agent in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        
        if term or trunc:
            action = None
            dones.append(True)
        else:
            action_mask = torch.tensor(obs['action_mask']).unsqueeze(0)
            state = torch.tensor(obs['observation']).unsqueeze(0)
            
            with torch.no_grad():
                logits, value = model(state, action_mask)
                probs = torch.softmax(logits, dim=-1)
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample().item()
                log_prob = action_dist.log_prob(torch.tensor(action))
            
            states.append(state)
            actions.append(action)
            values.append(value.item())
            log_probs.append(log_prob.item())
            action_masks.append(action_mask)
            agents.append(agent)
            dones.append(False)
        
        env.step(action)
    
    final_rewards = env.rewards
    
    episode_rewards = []
    for i, agent in enumerate(agents):
        if agent in final_rewards:
            if final_rewards[agent] == 1:
                episode_rewards.append(1.0)
            elif final_rewards[agent] == -1:
                episode_rewards.append(-1.0)
            else:
                episode_rewards.append(0.0)
        else:
            episode_rewards.append(0.0)
    
    if len(states) == 0:
        return [], [], [], [], [], []
    
    states = torch.cat(states)
    actions = torch.tensor(actions)
    rewards = torch.tensor(episode_rewards, dtype=torch.float32)
    values = torch.tensor(values, dtype=torch.float32)
    log_probs = torch.tensor(log_probs, dtype=torch.float32)
    action_masks = torch.cat(action_masks)
    dones = torch.tensor(dones[:-1], dtype=torch.float32)
    
    next_values = torch.zeros_like(values)
    if len(values) > 1:
        next_values[:-1] = values[1:]
    next_values[-1] = 0.0
    
    advantages, returns = compute_gae(rewards, values, next_values, dones)
    
    return states, actions, log_probs, returns, advantages, action_masks

def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages, action_masks, 
               clip_eps=0.2, value_coef=0.5, entropy_coef=0.01, epochs=4):
    
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    total_loss = 0
    for _ in range(epochs):
        logits, values = model(states, action_masks)
        
        log_probs = torch.log_softmax(logits, dim=-1)
        action_log_probs = log_probs[range(len(actions)), actions]
        
        ratio = torch.exp(action_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = F.mse_loss(values, returns)
        
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        entropy_loss = -entropy_coef * entropy
        
        total_loss_step = policy_loss + value_coef * value_loss + entropy_loss
        
        optimizer.zero_grad()
        total_loss_step.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        total_loss += total_loss_step.item()
    
    return total_loss / epochs

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
                        logits, _ = model(state, action_mask)
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
    model = PPONet()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    os.makedirs('checkpoints_ppo', exist_ok=True)
    
    print("Starting Connect-4 PPO Training...")
    print("=" * 50)
    
    for epoch in trange(1000, desc="PPO Training"):
        batch_states = []
        batch_actions = []
        batch_log_probs = []
        batch_returns = []
        batch_advantages = []
        batch_masks = []
        
        for _ in range(16):
            states, actions, log_probs, returns, advantages, masks = selfplay_episode(model)
            if len(states) > 0:
                batch_states.append(states)
                batch_actions.append(actions)
                batch_log_probs.append(log_probs)
                batch_returns.append(returns)
                batch_advantages.append(advantages)
                batch_masks.append(masks)
        
        if batch_states:
            all_states = torch.cat(batch_states)
            all_actions = torch.cat(batch_actions)
            all_log_probs = torch.cat(batch_log_probs)
            all_returns = torch.cat(batch_returns)
            all_advantages = torch.cat(batch_advantages)
            all_masks = torch.cat(batch_masks)
            
            loss = ppo_update(model, optimizer, all_states, all_actions, all_log_probs, 
                            all_returns, all_advantages, all_masks)
        else:
            loss = 0.0
        
        if epoch % 100 == 0:
            torch.save(model.state_dict(), f'checkpoints_ppo/ppo_ckpt_{epoch}.pt')
            
            wins, draws, losses = evaluate_policy(model, num_games=20)
            win_rate = wins / (wins + draws + losses)
            
            print(f"\nEpoch {epoch}:")
            print(f"  Loss: {loss:.4f}")
            print(f"  vs Random - Wins: {wins}, Draws: {draws}, Losses: {losses}")
            print(f"  Win Rate: {win_rate:.3f}")
            print("-" * 30)
    
    print("\nPPO Training completed!")
    print("Final model saved as 'checkpoints_ppo/ppo_ckpt_999.pt'")

if __name__ == "__main__":
    main() 