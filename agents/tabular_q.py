"""tabular q-learning agent

implementation of q-learning with tabular q-values, epsilon-greedy exploration,
and state space compression for connect-4.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pickle

from .base import Agent


class TabularQAgent(Agent):
    """tabular q-learning agent with epsilon-greedy policy."""
    
    def __init__(
        self, 
        name: str = "TabularQ",
        alpha: float = 0.1,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        gamma: float = 0.99
    ) -> None:
        super().__init__(name)
        self.alpha = alpha  # learning rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma  # discount factor
        
        # q-table: state_hash -> [q_values for 7 actions]
        self.q_table: Dict[int, np.ndarray] = {}
        
        # episode tracking
        self.episode_count = 0
        self.training_mode = True
        
        # for experience replay during training
        self.last_state: Optional[int] = None
        self.last_action: Optional[int] = None
    
    def _compress_state(self, obs: np.ndarray) -> int:
        """compress board state to a hash for q-table indexing.
        
        uses a simple hash of the board configuration. in practice,
        better compression methods could be used to reduce state space.
        """
        # convert board to tuple for hashing
        # use after-state representation to reduce state space
        flat_board = obs.flatten()
        
        # simple hash - in practice would use better compression
        # to handle the massive 3^42 state space
        return hash(tuple(flat_board))
    
    def _get_q_values(self, state_hash: int) -> np.ndarray:
        """get q-values for state, initializing if necessary."""
        if state_hash not in self.q_table:
            # initialize q-values optimistically
            self.q_table[state_hash] = np.zeros(7, dtype=np.float32)
        
        return self.q_table[state_hash]
    
    def _get_valid_actions(self, obs: np.ndarray) -> list[int]:
        """get list of valid actions from current state."""
        return [col for col in range(7) if obs[0, col] == 0]
    
    def act(self, obs: np.ndarray) -> int:
        """select action using epsilon-greedy policy."""
        state_hash = self._compress_state(obs)
        q_values = self._get_q_values(state_hash)
        valid_actions = self._get_valid_actions(obs)
        
        if not valid_actions:
            return 0  # fallback
        
        # epsilon-greedy action selection
        if self.training_mode and np.random.random() < self.epsilon:
            # explore: random valid action
            action = np.random.choice(valid_actions)
        else:
            # exploit: best valid action
            # mask invalid actions with -inf
            masked_q = q_values.copy()
            for i in range(7):
                if i not in valid_actions:
                    masked_q[i] = -np.inf
            
            action = np.argmax(masked_q)
            
            # fallback if no valid action found
            if action not in valid_actions:
                action = valid_actions[0]
        
        # store for learning
        self.last_state = state_hash
        self.last_action = action
        
        return action
    
    def learn(self, reward: float, next_obs: Optional[np.ndarray] = None, done: bool = False) -> None:
        """update q-values based on experience.
        
        args:
            reward: reward received
            next_obs: next state observation (none if terminal)
            done: whether episode is finished
        """
        if not self.training_mode or self.last_state is None or self.last_action is None:
            return
        
        current_q = self.q_table[self.last_state][self.last_action]
        
        if done or next_obs is None:
            # terminal state
            target_q = reward
        else:
            # get max q-value for next state
            next_state_hash = self._compress_state(next_obs)
            next_q_values = self._get_q_values(next_state_hash)
            next_valid_actions = self._get_valid_actions(next_obs)
            
            if next_valid_actions:
                max_next_q = max(next_q_values[action] for action in next_valid_actions)
            else:
                max_next_q = 0.0
            
            target_q = reward + self.gamma * max_next_q
        
        # q-learning update
        self.q_table[self.last_state][self.last_action] = current_q + self.alpha * (target_q - current_q)
    
    def end_episode(self, final_reward: float) -> None:
        """called at the end of an episode."""
        if self.training_mode:
            # final learning step
            self.learn(final_reward, done=True)
            
            # decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.episode_count += 1
    
    def set_training_mode(self, training: bool) -> None:
        """set whether agent is in training mode."""
        self.training_mode = training
    
    def reset(self) -> None:
        """reset agent state for new game."""
        self.last_state = None
        self.last_action = None
    
    def save_q_table(self, filepath: Path) -> None:
        """save q-table to file."""
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'alpha': self.alpha,
            'gamma': self.gamma
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_q_table(self, filepath: Path) -> None:
        """load q-table from file."""
        if not filepath.exists():
            return
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.q_table = data['q_table']
            self.epsilon = data.get('epsilon', self.epsilon)
            self.episode_count = data.get('episode_count', 0)
            
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Q-table: {e}") from e
    
    def get_stats(self) -> Dict[str, float]:
        """get training statistics."""
        return {
            'num_states': len(self.q_table),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'avg_q_value': np.mean([np.mean(q_vals) for q_vals in self.q_table.values()]) if self.q_table else 0.0
        }


def build_agent(weights_path: Optional[Path] = None) -> Agent:
    """build a tabular q-learning agent.
    
    args:
        weights_path: path to saved q-table (.pkl file)
        
    returns:
        tabularqagent instance
    """
    agent = TabularQAgent()
    
    if weights_path and weights_path.exists():
        agent.load_q_table(weights_path)
        agent.set_training_mode(False)  # evaluation mode
    
    return agent 