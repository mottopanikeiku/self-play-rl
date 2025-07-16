"""Fast Connect-4 Environment Implementation

A gymnasium-compatible Connect-4 environment optimized for performance.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import InvalidAction


class Connect4Env(gym.Env):
    """Fast Connect-4 Environment with canonical board representation.
    
    Observation Space: Box(0, 2, (6, 7), uint8)
        - 0: empty cell
        - 1: current player's piece
        - 2: opponent's piece
    
    Action Space: Discrete(7) - column to drop piece
    
    Rewards:
        - +10: win
        - -10: loss  
        - 0: draw
        - -0.01: step cost
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}
    
    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        
        # Environment constants
        self.rows = 6
        self.cols = 7
        self.connect_length = 4
        
        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.rows, self.cols), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(self.cols)
        
        # State
        self.board = np.zeros((self.rows, self.cols), dtype=np.uint8)
        self.current_player = 1
        self.move_count = 0
        self.game_over = False
        self.winner = 0
        
        # For deterministic behavior
        self.np_random: np.random.Generator
        
        # Zobrist hashing for transposition tables
        self._init_zobrist()
        
        self.render_mode = render_mode
    
    def _init_zobrist(self) -> None:
        """Initialize Zobrist hash values for fast board hashing."""
        # Fixed seed for consistent hashing across runs
        rng = np.random.RandomState(42)
        self.zobrist_table = rng.randint(
            0, 2**63, size=(self.rows, self.cols, 3), dtype=np.int64
        )
        self.zobrist_player = rng.randint(0, 2**63, dtype=np.int64)
    
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.board.fill(0)
        self.current_player = 1
        self.move_count = 0
        self.game_over = False
        self.winner = 0
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if self.game_over:
            raise InvalidAction("Game is already over")
        
        if not (0 <= action < self.cols):
            raise InvalidAction(f"Invalid action {action}, must be 0-{self.cols-1}")
        
        if not self._is_valid_action(action):
            raise InvalidAction(f"Column {action} is full")
        
        # Place piece
        row = self._drop_piece(action, self.current_player)
        self.move_count += 1
        
        # Check for win
        if self._check_win(row, action, self.current_player):
            self.game_over = True
            self.winner = self.current_player
            reward = 10.0 if self.current_player == 1 else -10.0
        elif self.move_count >= self.rows * self.cols:
            # Draw
            self.game_over = True
            reward = 0.0
        else:
            # Continue game with step cost
            reward = -0.01
            self.current_player = 3 - self.current_player  # Switch between 1 and 2
        
        observation = self._get_observation()
        
        return observation, reward, self.game_over, False, {}
    
    def _is_valid_action(self, action: int) -> bool:
        """Check if action (column) is valid."""
        return self.board[0, action] == 0
    
    def legal_actions(self) -> List[int]:
        """Get list of legal actions (non-full columns)."""
        return [col for col in range(self.cols) if self._is_valid_action(col)]
    
    def _drop_piece(self, col: int, player: int) -> int:
        """Drop piece in column and return the row it landed in."""
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = player
                return row
        raise InvalidAction(f"Column {col} is full")
    
    def _check_win(self, row: int, col: int, player: int) -> bool:
        """Check if the last move resulted in a win."""
        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal /
            (1, -1),  # diagonal \
        ]
        
        for dr, dc in directions:
            count = 1  # Count the piece just placed
            
            # Check in positive direction
            r, c = row + dr, col + dc
            while (0 <= r < self.rows and 0 <= c < self.cols and 
                   self.board[r, c] == player):
                count += 1
                r, c = r + dr, c + dc
            
            # Check in negative direction
            r, c = row - dr, col - dc
            while (0 <= r < self.rows and 0 <= c < self.cols and 
                   self.board[r, c] == player):
                count += 1
                r, c = r - dr, c - dc
            
            if count >= self.connect_length:
                return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get canonical board observation.
        
        Returns board from current player's perspective:
        - 0: empty
        - 1: current player's pieces
        - 2: opponent's pieces
        """
        if self.current_player == 1:
            return self.board.copy()
        else:
            # Swap perspective for player 2
            obs = self.board.copy()
            obs[obs == 1] = 3  # Temporary value
            obs[obs == 2] = 1
            obs[obs == 3] = 2
            return obs
    
    def hash(self) -> int:
        """Fast Zobrist hash of current board state."""
        hash_val = 0
        for row in range(self.rows):
            for col in range(self.cols):
                piece = self.board[row, col]
                if piece != 0:
                    hash_val ^= self.zobrist_table[row, col, piece]
        
        if self.current_player == 2:
            hash_val ^= self.zobrist_player
            
        return hash_val
    
    def render(self) -> Optional[str]:
        """Render the current board state."""
        if self.render_mode == "ansi" or self.render_mode == "human":
            board_str = "\n"
            board_str += " " + " ".join(str(i) for i in range(self.cols)) + "\n"
            board_str += "+" + "-" * (self.cols * 2 - 1) + "+\n"
            
            for row in range(self.rows):
                board_str += "|"
                for col in range(self.cols):
                    piece = self.board[row, col]
                    if piece == 0:
                        board_str += " "
                    elif piece == 1:
                        board_str += "X"
                    else:
                        board_str += "O"
                    if col < self.cols - 1:
                        board_str += " "
                board_str += "|\n"
            
            board_str += "+" + "-" * (self.cols * 2 - 1) + "+\n"
            
            if self.game_over:
                if self.winner == 0:
                    board_str += "Game Over: Draw!\n"
                else:
                    winner_symbol = "X" if self.winner == 1 else "O"
                    board_str += f"Game Over: {winner_symbol} wins!\n"
            else:
                current_symbol = "X" if self.current_player == 1 else "O"
                board_str += f"Current player: {current_symbol}\n"
            
            if self.render_mode == "human":
                print(board_str)
            else:
                return board_str
        
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        pass 