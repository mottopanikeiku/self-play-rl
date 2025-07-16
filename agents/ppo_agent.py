"""ppo agent (proximal policy optimization)

Implementation of PPO with actor-critic architecture, GAE, and clipped objective
for Connect-4.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base import Agent


class ActorCriticNetwork(nn.Module):
    """actor-critic network with shared backbone."""
    
    def __init__(self, input_size: int = 42, hidden_size: int = 256, output_size: int = 7):
        super().__init__()
        
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Policy head
        self.policy_head = nn.Linear(hidden_size, output_size)
        
        # Value head
        self.value_head = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """forward pass returning policy logits and state values."""
        shared_features = self.shared(x)
        policy_logits = self.policy_head(shared_features)
        state_values = self.value_head(shared_features)
        return policy_logits, state_values


class PPOAgent(Agent):
    """proximal policy optimization agent with gae."""
    
    def __init__(
        self,
        name: str = "PPO",
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        epochs: int = 4,
        batch_size: int = 64,
        device: Optional[str] = None
    ) -> None:
        super().__init__(name)
        
        # Hyperparameters
        self.lr = lr
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Networks
        self.actor_critic = ActorCriticNetwork().to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr)
        
        # Episode storage
        self.episode_states: List[torch.Tensor] = []
        self.episode_actions: List[int] = []
        self.episode_rewards: List[float] = []
        self.episode_values: List[torch.Tensor] = []
        self.episode_log_probs: List[torch.Tensor] = []
        self.episode_dones: List[bool] = []
        
        # Training mode
        self.training_mode = True
    
    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """convert observation to tensor."""
        flat_obs = obs.flatten().astype(np.float32)
        return torch.from_numpy(flat_obs).to(self.device)
    
    def _get_valid_actions(self, obs: np.ndarray) -> List[int]:
        """get valid actions from observation."""
        return [col for col in range(7) if obs[0, col] == 0]
    
    def act(self, obs: np.ndarray) -> int:
        """select action using current policy."""
        state_tensor = self._obs_to_tensor(obs)
        valid_actions = self._get_valid_actions(obs)
        
        if not valid_actions:
            return 0  # Fallback
        
        with torch.no_grad():
            policy_logits, state_value = self.actor_critic(state_tensor.unsqueeze(0))
            
            # Mask invalid actions
            action_mask = torch.full((7,), -float('inf'), device=self.device)
            for action in valid_actions:
                action_mask[action] = 0.0
            
            masked_logits = policy_logits.squeeze(0) + action_mask
            
            # Sample action
            if self.training_mode:
                # Sample from distribution during training
                probs = F.softmax(masked_logits, dim=0)
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                
                # Store for PPO
                self.episode_states.append(state_tensor)
                self.episode_actions.append(action.item())
                self.episode_values.append(state_value.squeeze())
                self.episode_log_probs.append(log_prob)
                
                return action.item()
            else:
                # Greedy action during evaluation
                return torch.argmax(masked_logits).item()
    
    def add_reward(self, reward: float, done: bool = False) -> None:
        """add reward and done flag for current step."""
        if self.training_mode:
            self.episode_rewards.append(reward)
            self.episode_dones.append(done)
    
    def end_episode(self, final_value: float = 0.0) -> None:
        """end episode and update using ppo."""
        if not self.training_mode or len(self.episode_log_probs) == 0:
            self._reset_episode()
            return
        
        # Calculate advantages using GAE
        advantages, returns = self._compute_gae(final_value)
        
        # Convert to tensors
        states = torch.stack(self.episode_states)
        actions = torch.tensor(self.episode_actions, device=self.device)
        old_log_probs = torch.stack(self.episode_log_probs)
        old_values = torch.stack(self.episode_values)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        self._update_ppo(states, actions, old_log_probs, old_values, advantages, returns)
        
        self._reset_episode()
    
    def _compute_gae(self, final_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """compute generalized advantage estimation."""
        values = torch.stack(self.episode_values + [torch.tensor(final_value, device=self.device)])
        rewards = torch.tensor(self.episode_rewards, device=self.device, dtype=torch.float32)
        dones = torch.tensor(self.episode_dones, device=self.device, dtype=torch.bool)
        
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - float(dones[t])
                next_value = final_value
            else:
                next_non_terminal = 1.0 - float(dones[t])
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        
        returns = advantages + values[:-1]
        
        return advantages, returns
    
    def _update_ppo(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> None:
        """update policy using ppo."""
        dataset_size = len(states)
        
        for _ in range(self.epochs):
            # Shuffle data
            indices = torch.randperm(dataset_size, device=self.device)
            
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_values = old_values[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current policy and values
                policy_logits, values = self.actor_critic(batch_states)
                
                # Create action masks for each state in batch
                action_masks = []
                for state in batch_states:
                    obs = state.cpu().numpy().reshape(6, 7)
                    valid_actions = self._get_valid_actions(obs)
                    mask = torch.full((7,), -float('inf'), device=self.device)
                    for action in valid_actions:
                        mask[action] = 0.0
                    action_masks.append(mask)
                
                action_masks_tensor = torch.stack(action_masks)
                masked_logits = policy_logits + action_masks_tensor
                
                # Calculate new log probabilities
                new_log_probs = F.log_softmax(masked_logits, dim=1)
                action_log_probs = new_log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                
                # Calculate ratio
                ratio = torch.exp(action_log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                value_pred_clipped = batch_old_values + torch.clamp(
                    values.squeeze() - batch_old_values, -self.clip_eps, self.clip_eps
                )
                value_loss1 = (values.squeeze() - batch_returns) ** 2
                value_loss2 = (value_pred_clipped - batch_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                
                # Entropy loss
                probs = F.softmax(masked_logits, dim=1)
                log_probs = F.log_softmax(masked_logits, dim=1)
                entropy = -(probs * log_probs).sum(dim=1).mean()
                entropy_loss = -self.entropy_coef * entropy
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss + entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1.0)
                self.optimizer.step()
    
    def _reset_episode(self) -> None:
        """reset episode storage."""
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_values.clear()
        self.episode_log_probs.clear()
        self.episode_dones.clear()
    
    def reset(self) -> None:
        """reset agent state."""
        self._reset_episode()
    
    def set_training_mode(self, training: bool) -> None:
        """set training mode."""
        self.training_mode = training
        if training:
            self.actor_critic.train()
        else:
            self.actor_critic.eval()
    
    def save_model(self, filepath: Path) -> None:
        """save model weights."""
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr': self.lr,
            'clip_eps': self.clip_eps,
            'value_coef': self.value_coef,
            'entropy_coef': self.entropy_coef,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }, filepath)
    
    def load_model(self, filepath: Path) -> None:
        """load model weights."""
        if not filepath.exists():
            return
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load hyperparameters
            self.lr = checkpoint.get('lr', self.lr)
            self.clip_eps = checkpoint.get('clip_eps', self.clip_eps)
            self.value_coef = checkpoint.get('value_coef', self.value_coef)
            self.entropy_coef = checkpoint.get('entropy_coef', self.entropy_coef)
            self.gamma = checkpoint.get('gamma', self.gamma)
            self.gae_lambda = checkpoint.get('gae_lambda', self.gae_lambda)
            self.epochs = checkpoint.get('epochs', self.epochs)
            self.batch_size = checkpoint.get('batch_size', self.batch_size)
            
            print(f"Loaded PPO model from {filepath}")
            
        except Exception as e:
            print(f"Failed to load PPO model: {e}")


def build_agent(weights_path: Optional[Path] = None) -> Agent:
    """build a ppo agent.
    
    Args:
        weights_path: Path to saved model weights (.pt file)
        
    Returns:
        PPOAgent instance
    """
    agent = PPOAgent()
    
    if weights_path and weights_path.exists():
        agent.load_model(weights_path)
        agent.set_training_mode(False)  # Evaluation mode
    
    return agent 