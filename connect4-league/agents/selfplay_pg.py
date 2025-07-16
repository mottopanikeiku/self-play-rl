"""self-play policy gradient agent

implementation of reinforce (vanilla policy gradient) with self-play training
for connect-4.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base import Agent


class PolicyNetwork(nn.Module):
    """3-layer MLP for policy approximation."""
    
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
        """forward pass through network."""
        return self.network(x)


class SelfPlayPGAgent(Agent):
    """self-play policy gradient (reinforce) agent."""
    
    def __init__(
        self,
        name: str = "SelfPlayPG",
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        device: Optional[str] = None
    ) -> None:
        super().__init__(name)
        
        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Network
        self.policy_net = PolicyNetwork().to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Episode storage for REINFORCE
        self.episode_states: List[torch.Tensor] = []
        self.episode_actions: List[int] = []
        self.episode_rewards: List[float] = []
        self.episode_log_probs: List[torch.Tensor] = []
        
        # Training mode
        self.training_mode = True
    
    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """convert observation to tensor."""
        # Flatten the 6x7 board to 42-dimensional vector
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
            logits = self.policy_net(state_tensor.unsqueeze(0))  # Add batch dimension
            
            # Mask invalid actions
            action_mask = torch.full((7,), -float('inf'), device=self.device)
            for action in valid_actions:
                action_mask[action] = 0.0
            
            masked_logits = logits.squeeze(0) + action_mask
            
            # Sample action
            if self.training_mode:
                # Sample from distribution during training
                probs = F.softmax(masked_logits, dim=0)
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                
                # Store for REINFORCE
                self.episode_states.append(state_tensor)
                self.episode_actions.append(action.item())
                self.episode_log_probs.append(log_prob)
                
                return action.item()
            else:
                # Greedy action during evaluation
                return torch.argmax(masked_logits).item()
    
    def add_reward(self, reward: float) -> None:
        """add reward for current step."""
        if self.training_mode:
            self.episode_rewards.append(reward)
    
    def end_episode(self) -> None:
        """end episode and update policy using reinforce."""
        if not self.training_mode or len(self.episode_log_probs) == 0:
            self._reset_episode()
            return
        
        # Calculate returns (discounted rewards)
        returns = []
        G = 0.0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, device=self.device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = 0.0
        entropy_loss = 0.0
        
        for log_prob, G in zip(self.episode_log_probs, returns):
            policy_loss -= log_prob * G
            entropy_loss -= log_prob * log_prob.exp()  # Simple entropy approximation
        
        # Total loss
        loss = policy_loss + self.entropy_coef * entropy_loss
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self._reset_episode()
    
    def _reset_episode(self) -> None:
        """reset episode storage."""
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_log_probs.clear()
    
    def reset(self) -> None:
        """reset agent state."""
        self._reset_episode()
    
    def set_training_mode(self, training: bool) -> None:
        """set training mode."""
        self.training_mode = training
        if training:
            self.policy_net.train()
        else:
            self.policy_net.eval()
    
    def save_model(self, filepath: Path) -> None:
        """save model weights."""
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr': self.lr,
            'gamma': self.gamma,
            'entropy_coef': self.entropy_coef
        }, filepath)
    
    def load_model(self, filepath: Path) -> None:
        """load model weights."""
        if not filepath.exists():
            return
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load hyperparameters if available
            self.lr = checkpoint.get('lr', self.lr)
            self.gamma = checkpoint.get('gamma', self.gamma)
            self.entropy_coef = checkpoint.get('entropy_coef', self.entropy_coef)
            
            print(f"Loaded SelfPlayPG model from {filepath}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")


def build_agent(weights_path: Optional[Path] = None) -> Agent:
    """build a self-play policy gradient agent.
    
    Args:
        weights_path: Path to saved model weights (.pt file)
        
    Returns:
        SelfPlayPGAgent instance
    """
    agent = SelfPlayPGAgent()
    
    if weights_path and weights_path.exists():
        agent.load_model(weights_path)
        agent.set_training_mode(False)  # Evaluation mode
    
    return agent 