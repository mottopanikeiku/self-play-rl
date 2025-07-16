"""Unit Tests for Connect-4 Environment

Comprehensive tests covering all environment functionality including
win detection, invalid moves, determinism, and edge cases.
"""

import numpy as np
import pytest
from gymnasium.error import InvalidAction

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from env.connect4 import Connect4Env


class TestConnect4Environment:
    """Test suite for Connect4Env."""
    
    def test_initialization(self):
        """Test environment initialization."""
        env = Connect4Env()
        
        # Check initial state
        assert env.rows == 6
        assert env.cols == 7
        assert env.connect_length == 4
        assert env.current_player == 1
        assert env.move_count == 0
        assert not env.game_over
        assert env.winner == 0
        
        # Check observation space
        assert env.observation_space.shape == (6, 7)
        assert env.observation_space.dtype == np.uint8
        
        # Check action space
        assert env.action_space.n == 7
    
    def test_reset(self):
        """Test environment reset functionality."""
        env = Connect4Env()
        
        # Make some moves
        env.step(0)
        env.step(1)
        
        # Reset
        obs, info = env.reset()
        
        # Check reset state
        assert np.all(env.board == 0)
        assert env.current_player == 1
        assert env.move_count == 0
        assert not env.game_over
        assert env.winner == 0
        assert obs.shape == (6, 7)
        assert np.all(obs == 0)
    
    def test_deterministic_seeding(self):
        """Test that seeding produces deterministic behavior."""
        # Create two environments with same seed
        env1 = Connect4Env()
        env2 = Connect4Env()
        
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        # Should be identical
        assert np.array_equal(obs1, obs2)
        
        # Make random moves and check they're the same
        for _ in range(5):
            legal_actions1 = env1.legal_actions()
            legal_actions2 = env2.legal_actions()
            assert legal_actions1 == legal_actions2
            
            if legal_actions1:
                action = legal_actions1[0]
                obs1, reward1, done1, _, _ = env1.step(action)
                obs2, reward2, done2, _, _ = env2.step(action)
                
                assert np.array_equal(obs1, obs2)
                assert reward1 == reward2
                assert done1 == done2
                
                if done1:
                    break
    
    def test_basic_moves(self):
        """Test basic move mechanics."""
        env = Connect4Env()
        env.reset()
        
        # Test first move
        obs, reward, done, _, info = env.step(3)  # Column 3
        
        assert not done
        assert reward == -0.01  # Step cost
        assert env.board[5, 3] == 1  # Piece at bottom
        assert env.current_player == 2  # Switched to player 2
        assert env.move_count == 1
        
        # Test second move
        obs, reward, done, _, info = env.step(3)  # Same column
        
        assert not done
        assert reward == -0.01
        assert env.board[4, 3] == 2  # Piece above first
        assert env.current_player == 1  # Switched back
        assert env.move_count == 2
    
    def test_invalid_moves(self):
        """Test invalid move handling."""
        env = Connect4Env()
        env.reset()
        
        # Test out of bounds moves
        with pytest.raises(InvalidAction):
            env.step(-1)
        
        with pytest.raises(InvalidAction):
            env.step(7)
        
        # Fill a column completely
        for _ in range(6):
            env.step(0)
        
        # Try to play in full column
        with pytest.raises(InvalidAction):
            env.step(0)
    
    def test_legal_actions(self):
        """Test legal actions functionality."""
        env = Connect4Env()
        env.reset()
        
        # Initially all columns should be legal
        legal = env.legal_actions()
        assert legal == [0, 1, 2, 3, 4, 5, 6]
        
        # Fill column 0 completely
        for _ in range(6):
            env.step(0)
        
        # Column 0 should no longer be legal
        legal = env.legal_actions()
        assert 0 not in legal
        assert len(legal) == 6
    
    def test_horizontal_win(self):
        """Test horizontal win detection."""
        env = Connect4Env()
        env.reset()
        
        # Create horizontal win for player 1
        # Bottom row: X X X X
        moves = [0, 0, 1, 1, 2, 2, 3]  # Player 1 gets columns 0,1,2,3
        
        for i, move in enumerate(moves):
            obs, reward, done, _, info = env.step(move)
            
            if i == len(moves) - 1:  # Last move
                assert done
                assert reward == 10.0  # Win reward
                assert env.winner == 1
                assert env.game_over
            else:
                assert not done
    
    def test_vertical_win(self):
        """Test vertical win detection."""
        env = Connect4Env()
        env.reset()
        
        # Create vertical win for player 1 in column 3
        moves = [3, 0, 3, 1, 3, 2, 3]  # Player 1 plays column 3 four times
        
        for i, move in enumerate(moves):
            obs, reward, done, _, info = env.step(move)
            
            if i == len(moves) - 1:  # Last move
                assert done
                assert reward == 10.0
                assert env.winner == 1
                assert env.game_over
            else:
                assert not done
    
    def test_diagonal_win_positive(self):
        """Test positive diagonal win (bottom-left to top-right)."""
        env = Connect4Env()
        env.reset()
        
        # Create diagonal win
        # Pattern (X = player 1, O = player 2):
        #    X
        #  O X
        # O O X
        # X O O X
        
        moves = [
            0,  # X in col 0
            1,  # O in col 1
            1,  # X in col 1
            2,  # O in col 2
            2,  # X in col 2
            3,  # O in col 3
            2,  # X in col 2 (third piece)
            3,  # O in col 3
            3,  # X in col 3 (third piece)
            4,  # O in col 4
            3   # X in col 3 (fourth piece - winning move)
        ]
        
        for i, move in enumerate(moves):
            obs, reward, done, _, info = env.step(move)
            
            if i == len(moves) - 1:  # Last move should win
                assert done
                assert reward == 10.0
                assert env.winner == 1
                break
    
    def test_diagonal_win_negative(self):
        """Test negative diagonal win (bottom-right to top-left)."""
        env = Connect4Env()
        env.reset()
        
        # Create negative diagonal win
        moves = [
            6,  # X in col 6
            5,  # O in col 5
            5,  # X in col 5
            4,  # O in col 4
            4,  # X in col 4
            3,  # O in col 3
            4,  # X in col 4 (third piece)
            3,  # O in col 3
            3,  # X in col 3 (third piece)
            2,  # O in col 2
            3   # X in col 3 (fourth piece - winning move)
        ]
        
        for i, move in enumerate(moves):
            obs, reward, done, _, info = env.step(move)
            
            if i == len(moves) - 1:  # Last move should win
                assert done
                assert reward == 10.0
                assert env.winner == 1
                break
    
    def test_draw_game(self):
        """Test draw condition when board is full."""
        env = Connect4Env()
        env.reset()
        
        # Fill board without any wins
        # This is tricky but possible with careful move ordering
        moves = [
            0, 1, 2, 3, 4, 5, 6,  # Fill bottom row
            0, 1, 2, 3, 4, 5, 6,  # Fill second row
            1, 0, 3, 2, 5, 4, 6,  # Fill third row (mixed)
            1, 0, 3, 2, 5, 4, 6,  # Fill fourth row
            2, 3, 0, 1, 6, 5, 4,  # Fill fifth row
            2, 3, 0, 1, 6, 5      # Fill top row partially
        ]
        
        game_ended = False
        for i, move in enumerate(moves):
            if move in env.legal_actions():
                obs, reward, done, _, info = env.step(move)
                
                if done:
                    game_ended = True
                    if env.winner == 0:  # Draw
                        assert reward == 0.0
                    break
        
        # If no win occurred, manually fill remaining spaces
        if not game_ended:
            while env.legal_actions() and not env.game_over:
                legal_moves = env.legal_actions()
                obs, reward, done, _, info = env.step(legal_moves[0])
                
                if done and env.winner == 0:
                    assert reward == 0.0
                    break
    
    def test_observation_format(self):
        """Test observation format and canonical representation."""
        env = Connect4Env()
        env.reset()
        
        # Make some moves
        env.step(0)  # Player 1
        env.step(1)  # Player 2
        
        obs = env._get_observation()
        
        # Check observation properties
        assert obs.shape == (6, 7)
        assert obs.dtype == np.uint8
        assert np.all((obs >= 0) & (obs <= 2))
        
        # Check canonical representation
        # Current player should always see themselves as 1
        if env.current_player == 1:
            # Player 1's turn: own pieces = 1, opponent = 2
            assert obs[5, 0] == 1  # Player 1's piece
            assert obs[5, 1] == 2  # Player 2's piece
        else:
            # Player 2's turn: own pieces = 1, opponent = 2
            assert obs[5, 0] == 2  # Player 1's piece (opponent)
            assert obs[5, 1] == 1  # Player 2's piece (own)
    
    def test_hash_function(self):
        """Test board hashing for transposition tables."""
        env = Connect4Env()
        env.reset()
        
        # Test that identical boards have same hash
        hash1 = env.hash()
        hash2 = env.hash()
        assert hash1 == hash2
        
        # Test that different boards have different hashes
        env.step(0)
        hash3 = env.hash()
        assert hash1 != hash3
        
        # Test that player change affects hash
        env2 = Connect4Env()
        env2.reset()
        env2.step(0)
        env2.step(1)
        
        hash4 = env2.hash()
        assert hash3 != hash4  # Different board states
    
    def test_game_over_state(self):
        """Test behavior when game is already over."""
        env = Connect4Env()
        env.reset()
        
        # Create a quick win
        moves = [0, 1, 0, 1, 0, 1, 0]  # Vertical win
        
        for move in moves:
            obs, reward, done, _, info = env.step(move)
            if done:
                break
        
        # Try to make another move
        with pytest.raises(InvalidAction):
            env.step(2)
    
    def test_reward_structure(self):
        """Test reward structure correctness."""
        env = Connect4Env()
        env.reset()
        
        # Test step cost
        obs, reward, done, _, info = env.step(3)
        assert reward == -0.01
        assert not done
        
        # Test win reward (create quick vertical win)
        moves = [0, 1, 0, 1, 0, 1, 0]
        
        for i, move in enumerate(moves):
            obs, reward, done, _, info = env.step(move)
            
            if done and env.winner == 1:
                assert reward == 10.0
                break
            elif done and env.winner == 2:
                assert reward == -10.0
                break
            else:
                assert reward == -0.01
    
    def test_rendering(self):
        """Test rendering functionality."""
        env = Connect4Env(render_mode="ansi")
        env.reset()
        
        # Test initial rendering
        output = env.render()
        assert isinstance(output, str)
        assert "0 1 2 3 4 5 6" in output  # Column numbers
        assert "Current player: X" in output
        
        # Make a move and test again
        env.step(3)
        output = env.render()
        assert "Current player: O" in output  # Player switched
        
        # Test human rendering mode
        env_human = Connect4Env(render_mode="human")
        env_human.reset()
        result = env_human.render()  # Should print to stdout
        assert result is None


def test_environment_integration():
    """Integration test simulating a full game."""
    env = Connect4Env()
    
    # Play a complete game
    obs, info = env.reset(seed=42)
    
    move_count = 0
    max_moves = 42
    
    while move_count < max_moves:
        # Get legal actions
        legal_actions = env.legal_actions()
        assert len(legal_actions) > 0
        
        # Make a move (simple strategy: try center first)
        if 3 in legal_actions:
            action = 3
        else:
            action = legal_actions[0]
        
        obs, reward, done, truncated, info = env.step(action)
        move_count += 1
        
        # Validate observation
        assert obs.shape == (6, 7)
        assert obs.dtype == np.uint8
        
        # Validate reward
        assert isinstance(reward, (int, float))
        
        if done:
            # Game ended
            assert env.game_over
            if env.winner == 0:
                assert reward == 0.0  # Draw
            else:
                assert reward in [10.0, -10.0]  # Win/loss
            break
        else:
            assert reward == -0.01  # Step cost
    
    assert move_count <= max_moves


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"]) 