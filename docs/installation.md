# Installation Guide

This guide provides detailed installation instructions and troubleshooting steps for Agent Arcade.

## System Requirements

- **Python**: Version 3.8 - 3.12 (3.13 not supported)
- **Operating System**: Linux, macOS, or WSL2 on Windows
- **Storage**: At least 2GB free space
- **Memory**: At least 4GB RAM recommended
- **Node.js**: Version 14 or higher (for NEAR CLI)

## Step-by-Step Installation

1. **Prepare Your Environment**

   ```bash
   # Create and activate virtual environment
   python3 -m venv drl-env
   source drl-env/bin/activate  # On Unix/macOS
   # or
   .\drl-env\Scripts\activate  # On Windows
   ```

2. **Install Core Dependencies**

   ```bash
   # Upgrade pip
   python3 -m pip install --upgrade pip

   # Install in this specific order
   pip install "ale-py==0.8.1"
   pip install "shimmy[atari]==0.2.1"
   pip install "gymnasium[atari]==0.28.1"
   ```

3. **Install Agent Arcade**

   ```bash
   pip install -e .
   ```

## Common Issues and Solutions

### 1. Dependency Conflicts

If you see errors about conflicting dependencies:

```bash
# Remove all related packages
pip uninstall -y ale-py shimmy gymnasium

# Clear pip cache
pip cache purge

# Reinstall in correct order
pip install "ale-py==0.8.1"
pip install "shimmy[atari]==0.2.1"
pip install "gymnasium[atari]==0.28.1"
```

### 2. ROM Installation Issues

If ROM installation fails:

```bash
# Check ROM directory
python3 -c "import ale_py; print(ale_py.get_roms_path())"

# Manual ROM installation
ROMS_DIR=$(python3 -c "import ale_py; print(ale_py.get_roms_path())")
mkdir -p "$ROMS_DIR"

# Download ROMs manually if needed
wget https://github.com/openai/atari-py/raw/master/atari_py/atari_roms/pong.bin -P "$ROMS_DIR"
wget https://github.com/openai/atari-py/raw/master/atari_py/atari_roms/space_invaders.bin -P "$ROMS_DIR"
chmod 644 "$ROMS_DIR"/*.bin
```

### 3. Python Version Issues

If you encounter Python version errors:

```bash
# Check current version
python3 --version

# Install correct version
# On macOS (using Homebrew):
brew install python@3.12
brew link python@3.12

# On Ubuntu/Debian:
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.12 python3.12-venv
```

### 4. Virtual Environment Issues

If virtual environment activation fails:

```bash
# Remove existing environment
rm -rf drl-env

# Create new environment
python3 -m venv drl-env
source drl-env/bin/activate

# Verify activation
which python
```

### 5. Import Errors

If you see import errors:

```bash
# Verify installations
pip list | grep -E "gymnasium|stable-baselines3|ale-py"

# Check package locations
python3 -c "import gymnasium; print(gymnasium.__file__)"
python3 -c "import ale_py; print(ale_py.__file__)"
```

### 6. NEAR CLI Issues

If NEAR CLI installation fails:

```bash
# Remove existing installation
npm uninstall -g near-cli

# Clear npm cache
npm cache clean --force

# Reinstall NEAR CLI
npm install -g near-cli
```

## Verification Steps

After installation, verify your setup:

```bash
# Check CLI installation
agent-arcade --version

# Verify environment
python3 -c "
import gymnasium
import stable_baselines3
import torch
import ale_py
print('Gymnasium:', gymnasium.__version__)
print('Stable-Baselines3:', stable_baselines3.__version__)
print('PyTorch:', torch.__version__)
print('ALE-py:', ale_py.__version__)
"

# Test ROM access
python3 -c "
import gymnasium as gym
env = gym.make('ALE/Pong-v5')
env.reset()
env.close()
print('ROM test successful!')
"
```

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/jbarnes850/agent-arcade/issues)
2. Join our [Discord Community](https://discord.gg/your-invite)
3. Create a new issue with:
   - Your system information
   - Error message
   - Steps to reproduce
   - Logs from `install.sh` 