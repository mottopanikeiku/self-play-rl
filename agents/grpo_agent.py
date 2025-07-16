"""grpo agent (group relative policy optimization)

Implementation of GRPO for competitive multi-agent environments with population-based
training and relative reward computation for Connect-4.
"""

from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base import Agent


class PolicyNetwork(nn.Module):
    """policy network for grpo agent."""
    
    def __init__(self, input_size: int = 42, hidden_size: int = 256, output_size: int = 7):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PopulationBuffer:
    """maintains population statistics for group baseline computation."""
    
    def __init__(self, maxlen: int = 100):
        self.returns = deque(maxlen=maxlen)
    
    def add_return(self, episode_return: float):
        self.returns.append(episode_return)
    
    def get_baseline(self) -> float:
        return float(np.mean(self.returns)) if self.returns else 0.0


class GRPOAgent(Agent):
    """grpo agent with population-based relative reward training."""
    
    def __init__(self, name: str = "GRPO", weights_path: Optional[Path] = None):
        super().__init__(name)
        
        self.lr = 1e-3
        self.clip_eps = 0.2
        self.entropy_coef = 0.01
        self.baseline_coef = 0.5
        self.group_size = 16
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.input_size = 42
        
        self.policy = PolicyNetwork(self.input_size).to(self.device)
        self.old_policy = PolicyNetwork(self.input_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        self.population_buffer = PopulationBuffer(maxlen=100)
        
        self.episode_states = []
        self.episode_actions = []
        self.episode_log_probs = []
        self.episode_rewards = []
        
        self.group_episodes = []
        
        self.training_mode = True
        
        if weights_path:
            self.load(weights_path)
    
    def _state_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """convert observation to tensor."""
        flattened = obs.flatten().astype(np.float32)
        return torch.FloatTensor(flattened).unsqueeze(0).to(self.device)
    
    def _get_valid_actions_mask(self, obs: np.ndarray) -> torch.Tensor:
        """generate mask for valid actions."""
        mask = torch.zeros(7, device=self.device)
        for col in range(7):
            if obs[0, col] == 0:
                mask[col] = 1.0
        return mask
    
    def act(self, obs: np.ndarray) -> int:
        """select action using current policy."""
        if not self.training_mode:
            return self._act_greedy(obs)
        
        try:
            state_tensor = self._state_to_tensor(obs)
            
            with torch.no_grad():
                logits = self.policy(state_tensor)
            
            valid_mask = self._get_valid_actions_mask(obs)
            logits = logits.squeeze(0)
            logits = logits + (valid_mask - 1) * 1e9
            
            if torch.all(valid_mask == 0):
                return 0
            
            probs = F.softmax(logits, dim=0)
            
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            if self.training_mode:
                self.episode_states.append(obs.copy())
                self.episode_actions.append(action.item())
                self.episode_log_probs.append(log_prob.item())
            
            return action.item()
        
        except Exception:
            valid_actions = [col for col in range(7) if obs[0, col] == 0]
            return np.random.choice(valid_actions) if valid_actions else 0
    
    def _act_greedy(self, obs: np.ndarray) -> int:
        """select action greedily during evaluation."""
        state_tensor = self._state_to_tensor(obs)
        
        with torch.no_grad():
            logits = self.policy(state_tensor)
        
        valid_mask = self._get_valid_actions_mask(obs)
        logits = logits.squeeze(0) + (valid_mask - 1) * 1e9
        
        return torch.argmax(logits).item()
    
    def add_reward(self, reward: float, done: bool):
        """store reward for current episode."""
        self.episode_rewards.append(reward)
        
        if done:
            episode_return = sum(self.episode_rewards)
            
            episode_data = {
                'states': self.episode_states.copy(),
                'actions': self.episode_actions.copy(),
                'log_probs': self.episode_log_probs.copy(),
                'rewards': self.episode_rewards.copy(),
                'return': episode_return
            }
            
            self.group_episodes.append(episode_data)
            
            self.reset()
            
            if len(self.group_episodes) >= self.group_size:
                self._train_on_group()
    
    def _train_on_group(self):
        """train policy on collected group episodes."""
        if len(self.group_episodes) == 0:
            return
        
        group_returns = [ep['return'] for ep in self.group_episodes]
        for ret in group_returns:
            self.population_buffer.add_return(ret)
        
        baseline = self.population_buffer.get_baseline()
        
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_relative_rewards = []
        
        for episode in self.group_episodes:
            episode_length = len(episode['states'])
            
            relative_reward = episode['return'] - baseline
            
            for i in range(episode_length):
                all_states.append(episode['states'][i])
                all_actions.append(episode['actions'][i])
                all_old_log_probs.append(episode['log_probs'][i])
                all_relative_rewards.append(relative_reward)
        
        if not all_states:
            self.group_episodes.clear()
            return
        
        states_tensor = torch.FloatTensor(np.array([s.flatten() for s in all_states])).to(self.device)
        actions_tensor = torch.LongTensor(all_actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(all_old_log_probs).to(self.device)
        relative_rewards_tensor = torch.FloatTensor(all_relative_rewards).to(self.device)
        
        relative_rewards_tensor = (relative_rewards_tensor - relative_rewards_tensor.mean()) / (relative_rewards_tensor.std() + 1e-8)
        
        logits = self.policy(states_tensor)
        
        action_masks = []
        for state in all_states:
            mask = torch.zeros(7, device=self.device)
            for col in range(7):
                if state[0, col] == 0:
                    mask[col] = 1.0
                else:
                    mask[col] = -1e9
            action_masks.append(mask)
        
        action_masks_tensor = torch.stack(action_masks)
        masked_logits = logits + action_masks_tensor
        
        probs = F.softmax(masked_logits, dim=1)
        action_dist = torch.distributions.Categorical(probs)
        new_log_probs = action_dist.log_prob(actions_tensor)
        
        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
        
        surr1 = ratio * relative_rewards_tensor
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * relative_rewards_tensor
        policy_loss = -torch.min(surr1, surr2).mean()
        
        entropy = action_dist.entropy().mean()
        entropy_bonus = self.entropy_coef * entropy
        
        total_loss = policy_loss - entropy_bonus
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        self.group_episodes.clear()
    
    def reset(self):
        """reset episode state."""
        self.episode_states.clear()
        self.episode_actions.clear() 
        self.episode_log_probs.clear()
        self.episode_rewards.clear()
    
    def set_training_mode(self, training: bool):
        """set training mode."""
        self.training_mode = training
        self.policy.train(training)
    
    def save(self, filepath: Path):
        """save model state."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr': self.lr,
            'clip_eps': self.clip_eps,
            'entropy_coef': self.entropy_coef,
            'baseline_coef': self.baseline_coef,
            'group_size': self.group_size,
            'population_returns': list(self.population_buffer.returns)
        }
        
        torch.save(checkpoint, filepath)
    
    def load(self, filepath: Path):
        """load model state."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.old_policy.load_state_dict(checkpoint['policy_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            self.lr = checkpoint.get('lr', self.lr)
            self.clip_eps = checkpoint.get('clip_eps', self.clip_eps)
            self.entropy_coef = checkpoint.get('entropy_coef', self.entropy_coef)
            self.baseline_coef = checkpoint.get('baseline_coef', self.baseline_coef)
            self.group_size = checkpoint.get('group_size', self.group_size)
            
            if 'population_returns' in checkpoint:
                self.population_buffer.returns = deque(checkpoint['population_returns'], maxlen=100)
                
        except Exception as e:
            raise RuntimeError(f"Failed to load GRPO model: {e}") from e


def build_agent(weights_path: Optional[Path] = None) -> Agent:
    """build a grpo agent."""
    agent = GRPOAgent(weights_path=weights_path)
    agent.set_training_mode(False)
    return agent 