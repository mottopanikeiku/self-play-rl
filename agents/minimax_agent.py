"""minimax agent

alpha-beta pruning search with transposition table and position evaluation
for connect-4.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

from .base import Agent


class MinimaxAgent(Agent):
    """minimax agent with alpha-beta pruning and transposition table."""
    
    def __init__(self, name: str = "Minimax", depth: int = 4):
        super().__init__(name)
        self.depth = depth
        self.transposition_table: Dict[int, Tuple[int, int]] = {}
        
        np.random.seed(42)
        self.zobrist_table = np.random.randint(0, 2**63, size=(6, 7, 3), dtype=np.int64)
    
    def _hash_board(self, board: np.ndarray) -> int:
        """compute zobrist hash of board position."""
        hash_value = 0
        for row in range(6):
            for col in range(7):
                piece = int(board[row, col])
                if piece != 0:
                    hash_value ^= self.zobrist_table[row, col, piece]
        return hash_value
    
    def _obs_to_board(self, obs: np.ndarray) -> np.ndarray:
        """convert observation to standard board representation."""
        board = np.zeros((6, 7), dtype=int)
        board[obs == 1] = 1
        board[obs == 2] = 2
        return board
    
    def act(self, obs: np.ndarray) -> int:
        """select best move using minimax with alpha-beta pruning."""
        board = self._obs_to_board(obs)
        _, best_move = self._minimax(board, self.depth, -float('inf'), float('inf'), True)
        return best_move if best_move != -1 else 0
    
    def _minimax(self, board: np.ndarray, depth: int, alpha: float, beta: float, is_maximizing: bool) -> Tuple[float, int]:
        """minimax with alpha-beta pruning and transposition table."""
        board_hash = self._hash_board(board)
        
        if board_hash in self.transposition_table:
            cached_depth, cached_score = self.transposition_table[board_hash]
            if cached_depth >= depth:
                return cached_score, -1
        
        winner = self._check_winner(board)
        if winner == 1:
            score = 1000 + depth
        elif winner == 2:
            score = -1000 - depth
        elif self._is_full(board):
            score = 0
        elif depth == 0:
            score = self._evaluate_position(board)
        else:
            valid_moves = [col for col in range(7) if board[0, col] == 0]
            best_move = -1
            
            if is_maximizing:
                max_eval = -float('inf')
                for move in valid_moves:
                    new_board = self._make_move(board, move, 1)
                    eval_score, _ = self._minimax(new_board, depth - 1, alpha, beta, False)
                    if eval_score > max_eval:
                        max_eval = eval_score
                        best_move = move
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
                score = max_eval
            else:
                min_eval = float('inf')
                for move in valid_moves:
                    new_board = self._make_move(board, move, 2)
                    eval_score, _ = self._minimax(new_board, depth - 1, alpha, beta, True)
                    if eval_score < min_eval:
                        min_eval = eval_score
                        best_move = move
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
                score = min_eval
            
            self.transposition_table[board_hash] = (depth, score)
            return score, best_move
        
        self.transposition_table[board_hash] = (depth, score)
        return score, -1
    
    def _check_winner(self, board: np.ndarray) -> int:
        """check for winning condition."""
        for player in [1, 2]:
            # horizontal
            for row in range(6):
                for col in range(4):
                    if all(board[row, col + i] == player for i in range(4)):
                        return player
            
            # vertical
            for row in range(3):
                for col in range(7):
                    if all(board[row + i, col] == player for i in range(4)):
                        return player
            
            # diagonal (top-left to bottom-right)
            for row in range(3):
                for col in range(4):
                    if all(board[row + i, col + i] == player for i in range(4)):
                        return player
            
            # diagonal (bottom-left to top-right)
            for row in range(3, 6):
                for col in range(4):
                    if all(board[row - i, col + i] == player for i in range(4)):
                        return player
        return 0
    
    def _is_full(self, board: np.ndarray) -> bool:
        """check if board is completely filled."""
        return np.all(board[0, :] != 0)
    
    def _make_move(self, board: np.ndarray, col: int, player: int) -> np.ndarray:
        """apply move to board and return new board state."""
        new_board = board.copy()
        for row in range(5, -1, -1):
            if new_board[row, col] == 0:
                new_board[row, col] = player
                break
        return new_board
    
    def _evaluate_position(self, board: np.ndarray) -> float:
        """evaluate board position using heuristics."""
        score = 0
        
        # horizontal windows
        for row in range(6):
            for col in range(4):
                score += self._evaluate_window(board[row, col:col+4])
        
        # vertical windows
        for row in range(3):
            for col in range(7):
                score += self._evaluate_window(board[row:row+4, col])
        
        # diagonal windows (top-left to bottom-right)
        for row in range(3):
            for col in range(4):
                score += self._evaluate_window([board[row+i, col+i] for i in range(4)])
        
        # diagonal windows (bottom-left to top-right)
        for row in range(3, 6):
            for col in range(4):
                score += self._evaluate_window([board[row-i, col+i] for i in range(4)])
        
        return score
    
    def _evaluate_window(self, window) -> float:
        """evaluate a window of 4 positions."""
        window = np.array(window)
        player_count = np.sum(window == 1)
        opponent_count = np.sum(window == 2)
        empty_count = np.sum(window == 0)
        
        if opponent_count > 0:
            return 0
        
        if player_count == 4:
            return 1000
        elif player_count == 3 and empty_count == 1:
            return 100
        elif player_count == 2 and empty_count == 2:
            return 10
        else:
            return 0
    
    def reset(self):
        """reset agent state."""
        self.transposition_table.clear()


def build_agent() -> Agent:
    """build minimax agent."""
    return MinimaxAgent() 