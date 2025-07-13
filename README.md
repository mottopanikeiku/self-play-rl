# Connect-4 Self-Play Reinforcement Learning

A clean implementation of self-play reinforcement learning for Connect-4 using PyTorch and REINFORCE algorithm.

## Features

- **Simple Neural Network**: 3-layer MLP that maps board states to action probabilities
- **Self-Play Training**: Agents learn by playing against themselves
- **REINFORCE Algorithm**: Policy gradient method for training
- **Action Masking**: Prevents invalid moves during gameplay
- **Evaluation**: Track performance against random opponents
- **Interactive Demo**: Play against trained models or watch them compete

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training

```bash
python connect4_selfplay.py
```

This will:
- Train for 1000 epochs (32 games per epoch)
- Save checkpoints every 100 epochs to `checkpoints/`
- Evaluate against random opponents every 100 epochs
- Display training progress and win rates

### 3. Demo the Trained Model

```bash
python demo.py
```

Choose from:
- **Play against the model**: Interactive human vs AI gameplay
- **Watch model vs model**: Observe two models competing
- **Exit**: Quit the demo

## Architecture

### Neural Network
```
Input: 42 values (6×7 board state, flattened)
Hidden: 256 → ReLU → 256 → ReLU
Output: 7 values (action logits for each column)
```

### Training Process
1. **Self-Play**: Generate games by having the model play against itself
2. **Experience Collection**: Store (state, action, reward) tuples
3. **Policy Update**: Use REINFORCE to update the network weights
4. **Evaluation**: Test against random opponents to measure progress

### Key Components

- **Action Masking**: Prevents invalid moves by setting logits to -∞
- **Reward Shaping**: +1 for wins, -1 for losses, 0 for draws
- **Stochastic Policy**: Uses softmax + multinomial sampling for exploration

## Files

- `connect4_selfplay.py`: Main training script
- `demo.py`: Interactive demo for testing trained models
- `requirements.txt`: Python dependencies
- `checkpoints/`: Directory for saved model checkpoints

## Expected Results

After training, you should see:
- **Win rate vs random**: Should improve from ~50% to 80%+ over time
- **Loss convergence**: Training loss should stabilize
- **Strategic play**: Model learns to block opponent wins and create own win conditions

## Customization

### Training Parameters
- **Learning rate**: Adjust `lr` in optimizer (default: 1e-3)
- **Batch size**: Change games per epoch (default: 32)
- **Network size**: Modify hidden layer dimensions in `Net` class
- **Training length**: Change number of epochs (default: 1000)

### Advanced Features
- **Opponent curriculum**: Start with random, gradually increase difficulty
- **Value function**: Add critic network for Actor-Critic methods
- **MCTS integration**: Combine with Monte Carlo Tree Search
- **Temperature annealing**: Reduce exploration over time

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure all dependencies are installed
2. **No checkpoints**: Run training script first before demo
3. **Pygame issues**: Install pygame for visual rendering
4. **Memory problems**: Reduce batch size or network dimensions

### Performance Tips
- Use GPU if available: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- Increase batch size for more stable gradients
- Add baseline subtraction to reduce variance
- Implement experience replay for better sample efficiency

## License

This project is open source and available under the MIT License. 