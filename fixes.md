1. **Installation & Quickstart**

```bash
# Create install.sh script:
#!/bin/bash
set -e

# Create and activate virtual environment
python -m venv drl-env
source drl-env/bin/activate

# Install dependencies
pip install -e .  # Install package in editable mode
pip install -r requirements.txt
pip install "gymnasium[accept-rom-license,atari]"

# Verify installation
agent-arcade --version

# Setup NEAR CLI if not installed
if ! command -v near &> /dev/null; then
    echo "Installing NEAR CLI..."
    npm install -g near-cli
fi
```

2. **CLI Structure Fixes**

```python
cli/
├── core/
│   ├── __init__.py
│   ├── near.py        # Existing NEAR integration
│   ├── config.py      # Configuration management
│   └── utils.py       # Common utilities
├── games/
│   ├── __init__.py
│   ├── base.py        # Base game interface
│   ├── pong/
│   │   ├── __init__.py
│   │   └── game.py    # Rename from pong_arcade.py
│   └── space_invaders/
│       ├── __init__.py
│       └── game.py
└── main.py            # CLI entry point
```

3. **CLI Command Implementation**

```python
# cli/main.py
import click
from cli.core.near import NEARWallet
from cli.games import get_game

@click.group()
def cli():
    """Agent Arcade CLI"""
    pass

@cli.command()
@click.option('--network', default='testnet')
def login(network):
    """Login to NEAR wallet"""
    wallet = NEARWallet(network)
    wallet.login_with_cli()

@cli.command()
@click.argument('game')
@click.option('--render/--no-render', default=False)
@click.option('--config', type=click.Path())
def train(game, render, config):
    """Train an agent for a specific game"""
    game_instance = get_game(game)
    game_instance.train(render=render, config_path=config)

@cli.command()
@click.argument('game')
@click.option('--model', required=True, type=click.Path())
@click.option('--episodes', default=10)
def evaluate(game, model, episodes):
    """Evaluate a trained model"""
    game_instance = get_game(game)
    game_instance.evaluate(model_path=model, episodes=episodes)

@cli.command()
@click.argument('game')
@click.option('--model', required=True)
@click.option('--amount', required=True, type=float)
@click.option('--target-score', required=True, type=int)
def stake(game, model, amount, target_score):
    """Stake NEAR on agent performance"""
    wallet = NEARWallet()
    game_instance = get_game(game)
    game_instance.stake(wallet, model, amount, target_score)
```

4. **Package Setup Fix**

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="agent-arcade",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click',
        'stable-baselines3',
        'gymnasium[atari]',
        'py-near',
        'torch',
        'numpy',
    ],
    entry_points={
        'console_scripts': [
            'agent-arcade=cli.main:cli',
        ],
    },
)
```

5. **Documentation Updates**

```markdown
# docs/getting-started.md updates:

## Quick Start

1. **One-Line Install**
```bash
curl -sSL https://raw.githubusercontent.com/your-repo/agent-arcade/main/install.sh | bash
```

2. **Manual Installation**
```bash
# Clone repository
git clone https://github.com/your-repo/agent-arcade.git
cd agent-arcade

# Install package and dependencies
python -m venv drl-env
source drl-env/bin/activate  # On Windows: drl-env\Scripts\activate
pip install -e .
pip install -r requirements.txt
pip install "gymnasium[accept-rom-license,atari]"
```

3. **Verify Installation**
```bash
agent-arcade --version
```

## Basic Usage

1. **Login to NEAR**
```bash
agent-arcade login
```

2. **Train an Agent**
```bash
# Train Pong agent
agent-arcade train pong --render

# Train Space Invaders
agent-arcade train space-invaders --config configs/space_invaders_optimized_sb3_config.yaml
```

3. **Evaluate & Stake**
```bash
# Evaluate agent
agent-arcade evaluate pong --model models/pong_final.zip --episodes 10

# Stake NEAR
agent-arcade stake pong --model models/pong_final.zip --amount 10 --target-score 15
```
```

6. **Implementation Plan**:

1. **Phase 1: Installation & Setup**
   - Create `install.sh` script
   - Update `setup.py` with correct dependencies
   - Fix package structure and imports
   - Add version checking

2. **Phase 2: CLI Framework**
   - Implement core CLI structure
   - Add game loading mechanism
   - Create unified game interface
   - Add configuration management

3. **Phase 3: NEAR Integration**
   - Complete wallet integration
   - Implement staking flow
   - Add leaderboard updates
   - Create evaluation pipeline

4. **Phase 4: Testing & Documentation**
   - Add end-to-end tests
   - Update all documentation
   - Create example workflows
   - Add troubleshooting guide


1. **Current Assets**:
   - Game implementations (Pong & Space Invaders) ✅
   - Configuration files for both games ✅
   - Basic CLI structure with NEAR commands ✅
   - Documentation for metrics and NEAR integration ✅

2. **Phase 3 Requirements**:
   ```markdown
   - Complete wallet integration
   - Implement staking flow
   - Add leaderboard updates
   - Create evaluation pipeline
   ```

3. **What We Have for NEAR Integration**:
   - NEAR CLI tools and documentation
   - Basic wallet integration structure
   - Smart contract directory (`/contract`)
   - Integration documentation

4. **Missing Components**:
   - Smart contract implementation for:
     - Staking pool management
     - Leaderboard storage
     - Reward distribution
   - Contract deployment and initialization
   - Initial staking pool funding

5. **Dependencies Analysis**:
   ```
   Wallet Integration → Staking Flow → Leaderboard → Evaluation
         ↓                    ↓            ↓             ↓
   [Can Do Now]     [Needs Contract]  [Needs DB]   [Can Do Now]
   ```

**Recommendation**:

We can split Phase 3 into two sub-phases:

1. **Phase 3A - Client-Side** (Can do now):
   - Complete wallet integration
   - Build evaluation pipeline
   - Implement local leaderboard
   - Add staking flow structure

2. **Phase 3B - Contract & Pool** (Needs prerequisites):
   - Smart contract development
   - Pool initialization
   - Contract deployment
   - Fund staking pool
