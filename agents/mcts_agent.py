"""mcts agent

monte carlo tree search implementation with uct selection for connect-4.
"""

import math
import random
from typing import Dict, List, Optional
import numpy as np

from .base import Agent


class MCTSNode:
    """node in mcts tree."""
    
    def __init__(self, board: np.ndarray, parent=None, action: Optional[int] = None):
        self.board = board.copy()
        self.parent = parent
        self.action = action
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visits = 0
        self.wins = 0.0
        self.untried_actions = self._get_legal_actions()
    
    def _get_legal_actions(self) -> List[int]:
        """get valid column indices."""
        return [col for col in range(7) if self.board[0, col] == 0]
    
    def _check_win(self, board: np.ndarray, player: int) -> bool:
        """check if player has won."""
        rows, cols = board.shape
        
        # horizontal
        for row in range(rows):
            for col in range(cols - 3):
                if all(board[row, col + i] == player for i in range(4)):
                    return True
        
        # vertical
        for col in range(cols):
            for row in range(rows - 3):
                if all(board[row + i, col] == player for i in range(4)):
                    return True
        
        # diagonal (top-left to bottom-right)
        for row in range(rows - 3):
            for col in range(cols - 3):
                if all(board[row + i, col + i] == player for i in range(4)):
                    return True
        
        # diagonal (bottom-left to top-right)
        for row in range(3, rows):
            for col in range(cols - 3):
                if all(board[row - i, col + i] == player for i in range(4)):
                    return True
        
        return False
    
    def is_terminal(self) -> bool:
        """check if node represents terminal state."""
        return (self._check_win(self.board, 1) or 
                self._check_win(self.board, 2) or 
                len(self.untried_actions) == 0 and len(self.children) == 0)
    
    def is_fully_expanded(self) -> bool:
        """check if all children have been expanded."""
        return len(self.untried_actions) == 0
    
    def ucb1_value(self, exploration_constant: float = math.sqrt(2)) -> float:
        """calculate ucb1 value for selection."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.wins / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def expand(self) -> 'MCTSNode':
        """expand node by adding child."""
        action = self.untried_actions.pop()
        new_board = self._make_move(self.board, action)
        child_node = MCTSNode(new_board, parent=self, action=action)
        self.children[action] = child_node
        return child_node
    
    def _make_move(self, board: np.ndarray, action: int, player: int = 1) -> np.ndarray:
        """apply move to board."""
        new_board = board.copy()
        for row in range(5, -1, -1):
            if new_board[row, action] == 0:
                new_board[row, action] = player
                break
        return new_board
    
    def simulate(self) -> float:
        """random rollout simulation."""
        current_board = self.board.copy()
        current_player = 1
        
        while True:
            if self._check_win(current_board, 3 - current_player):
                return 1.0 if current_player == 2 else 0.0
            
            legal_actions = [col for col in range(7) if current_board[0, col] == 0]
            if not legal_actions:
                return 0.5  # draw
            
            action = random.choice(legal_actions)
            current_board = self._make_move(current_board, action, current_player)
            current_player = 3 - current_player
    
    def backpropagate(self, result: float):
        """update statistics up the tree."""
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(1.0 - result)


class MCTSAgent(Agent):
    """monte carlo tree search agent."""
    
    def __init__(self, name: str = "MCTS", num_simulations: int = 100):
        super().__init__(name)
        self.num_simulations = num_simulations
        self.root = None
    
    def _obs_to_board(self, obs: np.ndarray) -> np.ndarray:
        """convert observation to standard board representation."""
        board = np.zeros((6, 7), dtype=int)
        board[obs == 1] = 1
        board[obs == 2] = 2
        return board
    
    def act(self, obs: np.ndarray) -> int:
        """select action using mcts."""
        board = self._obs_to_board(obs)
        
        # reuse tree if board matches
        if self.root and not np.array_equal(self.root.board, board):
            self.root = None
        
        if self.root is None:
            self.root = MCTSNode(board)
        
        # mcts iterations
        for _ in range(self.num_simulations):
            node = self._select(self.root)
            child = self._expand(node)
            result = child.simulate()
            child.backpropagate(result)
        
        # handle case with no children
        if not self.root.children:
            legal_actions = [col for col in range(7) if board[0, col] == 0]
            return random.choice(legal_actions) if legal_actions else 0
        
        # select most visited child
        best_action = max(self.root.children.keys(), 
                         key=lambda a: self.root.children[a].visits)
        
        # update root for next turn
        if best_action in self.root.children:
            self.root = self.root.children[best_action]
            self.root.parent = None
        
        return best_action
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """select leaf node using ucb1."""
        while not node.is_terminal() and node.is_fully_expanded():
            node = max(node.children.values(), key=lambda c: c.ucb1_value())
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """expand node or return itself if terminal."""
        if node.is_terminal():
            return node
        return node.expand()
    
    def reset(self):
        """reset agent state."""
        self.root = None


def build_agent() -> Agent:
    """build mcts agent."""
    return MCTSAgent() 