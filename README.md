# connect-4 reinforcement learning

this repository contains a comprehensive connect-4 ai tournament system with multiple reinforcement learning and classical algorithms.

## main project

the active implementation is in the `connect4-league/` directory, which features:

- 6 different ai algorithms (reinforcement learning + classical)
- professional tournament system with elo ratings
- automated training and evaluation pipeline
- comprehensive reporting and visualization

## getting started

```bash
cd connect4-league/
make all
```

this will build the environment, train agents, run tournaments, and generate reports.

## documentation

see `connect4-league/README.md` for complete documentation, including:
- algorithm descriptions
- installation instructions  
- usage examples
- configuration options
- performance benchmarks

## algorithms included

- self-play policy gradient (reinforce)
- group relative policy optimization (grpo)
- proximal policy optimization (ppo)
- minimax with alpha-beta pruning
- monte carlo tree search (mcts)
- tabular q-learning 