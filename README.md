# connect-4 algorithm league

tournament system for connect-4 featuring six ai algorithms competing in round-robin tournaments with elo ratings and automated reporting.

## quick start

```bash
make all
```

builds virtual environment, installs dependencies, trains reinforcement learning agents, runs tournament, and generates report.

## algorithms

| algorithm | type | description |
|-----------|------|-------------|
| self-play pg | reinforcement learning | reinforce policy gradient with monte carlo returns |
| grpo | reinforcement learning | group relative policy optimization with population training |
| ppo | reinforcement learning | proximal policy optimization with actor-critic architecture |
| minimax | classical | alpha-beta pruning search to depth 4 with transposition table |
| mcts | classical | monte carlo tree search with uct selection and 100 rollouts |
| tabular q | reinforcement learning | q-learning with state compression and experience replay |

## requirements

- python 3.11+
- 4gb+ ram for tournament execution
- multi-core cpu recommended for parallel processing
- optional: cuda-compatible gpu for accelerated rl training

## installation

### automatic
```bash
git clone <repository>
cd connect4-league
make all
```

### manual
```bash
python3.11 -m venv .venv
source .venv/bin/activate  # linux/mac
# .venv\Scripts\activate   # windows

pip install --upgrade pip
pip install -r requirements.txt

python scripts/train_agents.py
python scripts/tournament.py  
python scripts/report.py
```

## project structure

```
connect4-league/
├── makefile               # build automation
├── requirements.txt       # dependency specifications with version pins
├── pyproject.toml         # code quality configuration
├── readme.md             # documentation
├── data/                 # generated outputs
│   ├── *.pt              # pytorch model weights
│   ├── *.pkl             # serialized q-tables
│   ├── results.json      # tournament outcomes
│   ├── elo.csv          # rating evolution
│   └── training_logs.jsonl # training metrics
├── agents/               # algorithm implementations
│   ├── base.py          # abstract interface
│   ├── random_agent.py  # baseline random policy
│   ├── selfplay_pg.py   # reinforce implementation
│   ├── grpo_agent.py    # group relative policy optimization
│   ├── ppo_agent.py     # proximal policy optimization
│   ├── minimax_agent.py # alpha-beta search
│   ├── mcts_agent.py    # monte carlo tree search
│   ├── tabular_q.py     # q-learning with tables
│   └── human_cli.py     # command-line interface
├── env/                  # game environment
│   └── connect4.py      # gymnasium-compatible implementation
├── scripts/              # execution pipeline
│   ├── train_agents.py  # reinforcement learning training
│   ├── tournament.py    # round-robin competition
│   └── report.py        # visualization generation
├── configs/              # parameter files
│   ├── tournament.yaml  # competition settings
│   └── training.yaml    # learning hyperparameters
├── tests/               # unit tests
│   └── test_env.py      # environment validation
└── report/              # generated output
    └── index.html       # competition results
```

## configuration

### tournament settings (configs/tournament.yaml)
```yaml
games_per_pair: 100      # matches between each algorithm pair
num_processes: null      # cpu cores (null = detect automatically)
elo_k: 20               # rating volatility parameter
time_limit_ms: 500      # maximum decision time per move
```

### training settings (configs/training.yaml)
```yaml
num_episodes: 50000     # training episodes per reinforcement learning agent
selfplay_pg:
  lr: 3e-4             # learning rate
  gamma: 0.99          # discount factor
grpo:
  group_size: 16       # population size for group baseline
ppo:
  epochs: 4            # training epochs per data batch
  batch_size: 64       # mini-batch size for updates
```

## usage

### complete pipeline
```bash
make all                 # full build and execution
```

### individual components
```bash
python scripts/train_agents.py    # train reinforcement learning agents
python scripts/tournament.py      # run competition with existing models
python scripts/report.py          # generate visualization from results
make test                          # execute unit tests
```

### custom configuration
edit configs/tournament.yaml and configs/training.yaml, then run individual scripts.

## output formats

### training logs (data/training_logs.jsonl)
```json
{"agent": "ppo", "episode": 1000, "avg_reward": 0.145, "win_rate_p1": 0.523}
```

### tournament results (data/results.json)
```json
[
  {"agent1": "ppo", "agent2": "minimax", "wins": 67, "losses": 31, "draws": 2}
]
```

### elo evolution (data/elo.csv)
```csv
game,grpo,minimax,mcts,ppo,random,selfplay_pg
0,1200.0,1200.0,1200.0,1200.0,1200.0,1200.0
100,1234.5,1189.2,1205.7,1287.3,1098.4,1184.9
```

### html report (report/index.html)
- final leaderboard with elo ratings
- win-rate matrix heatmap
- elo evolution plots
- sample game animations
- detailed match statistics

## algorithm implementation details

### neural networks
- architecture: 3-layer mlp (42 → 256 → 256 → 7)
- input: flattened 6×7 board state (42 values)
- output: action probability distribution over 7 columns
- activation: relu for hidden layers

### self-play training
- all reinforcement learning agents train against themselves
- canonical board representation (current player always 1)
- action masking prevents selection of invalid moves
- gradient clipping with maximum norm 1.0

### classical algorithms
- minimax: 4-ply search depth, alpha-beta pruning, zobrist hashing for transposition table
- mcts: uct selection with exploration constant √2, 100 simulations per move, tree reuse
- q-learning: ε-greedy exploration (0.1→0.01 decay), learning rate 0.1, compressed state representation

## performance characteristics

### training time (8-core cpu)
- self-play pg: approximately 2 hours (50k episodes)
- grpo: approximately 3 hours (group batching overhead)
- ppo: approximately 4 hours (actor-critic complexity)
- complete pipeline: approximately 10 hours including tournament

### algorithm benchmarks vs random baseline
- ppo: 75-85% win rate
- grpo: 70-80% win rate  
- minimax: 85-95% win rate
- mcts: 70-80% win rate
- self-play pg: 60-70% win rate
- tabular q: 55-65% win rate

### tournament rankings
typical performance order (subject to training variation):
1. minimax (deterministic search advantage)
2. ppo (sample efficiency)
3. grpo (competitive optimization)
4. mcts (simulation-based planning)
5. self-play pg (baseline reinforcement learning)
6. tabular q (state space limitations)

## testing

```bash
make test                # execute all tests
python -m pytest tests/ -v    # detailed test output
python tests/test_env.py      # environment-specific tests
```

test coverage includes:
- connect-4 environment correctness
- win detection (horizontal, vertical, diagonal)
- invalid move handling
- deterministic seeding
- game termination conditions

## troubleshooting

### cuda errors during training
```bash
export CUDA_VISIBLE_DEVICES=""
python scripts/train_agents.py
```

### tournament finds no trained agents
```bash
ls data/*.pt data/*.pkl         # verify model files exist
python scripts/train_agents.py  # train agents if missing
```

### memory issues
reduce batch_size in configs/training.yaml or train agents individually.

### permission errors (linux)
```bash
chmod +x scripts/*.py
```

## customization

### adding new agents
1. implement agent class in agents/ inheriting from base.agent
2. provide build_agent() factory function
3. import in scripts/tournament.py
4. add to agent_builders dictionary

### modifying parameters
edit configs/training.yaml and configs/tournament.yaml configuration files.

## reproducibility

system designed for deterministic results:
- pinned dependency versions with integrity hashes
- fixed random seeds in all algorithms
- deterministic environment implementation
- consistent model serialization
- isolated virtual environment

identical hardware and commands produce identical outcomes.

## license

mit license 