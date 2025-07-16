# connect-4 reinforcement learning algorithms

implementation of multiple reinforcement learning algorithms for connect-4 with self-play training and head-to-head evaluation.

## algorithms implemented

### reinforce (policy gradient)
- file: `connect4_selfplay.py`
- monte carlo policy gradient with baseline
- 256x256 neural network architecture
- entropy regularization for exploration

### ppo (proximal policy optimization)  
- file: `connect4_ppo.py`
- actor-critic architecture with shared backbone
- clipped objective function and gae advantage estimation
- multiple training epochs per batch

### grpo (group relative policy optimization)
- file: `connect4_grpo.py`
- population-based training with relative reward computation
- group baseline normalization
- designed for competitive environments

## setup

### install dependencies
```bash
pip install -r requirements.txt
```

### train individual algorithms
```bash
python connect4_selfplay.py    # reinforce
python connect4_ppo.py         # ppo  
python connect4_grpo.py        # grpo
```

### run tournament
```bash
python tournament.py           # automatic mode
python tournament.py --mode interactive  # select models
```

### demo interface
```bash
python demo.py
```

## implementation details

### neural network architecture
```
input: 84 values (6x7x2 flattened board state)
hidden: linear(84, 256) -> relu -> linear(256, 256) -> relu
output: linear(256, 7) action logits
```

### training configuration
- episodes: 1000 per algorithm
- learning rates: reinforce (1e-3), ppo (3e-4), grpo (1e-3)
- discount factor: 0.99
- checkpoint frequency: every 100 episodes

### tournament system
- round-robin format between all trained models
- configurable number of games per matchup
- win/loss/draw statistics

## file structure
```
connect4_selfplay.py    # reinforce implementation
connect4_ppo.py         # ppo implementation
connect4_grpo.py        # grpo implementation
tournament.py           # tournament engine
demo.py                 # human interface
requirements.txt        # dependencies
checkpoints/            # reinforce model saves
checkpoints_ppo/        # ppo model saves
checkpoints_grpo/       # grpo model saves
```

## algorithm comparison

| algorithm | policy type | value function | stability |
|-----------|-------------|----------------|-----------| 