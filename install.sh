#!/bin/bash
set -e

echo "🎮 Installing Agent Arcade..."

# Function to handle errors
handle_error() {
    echo "❌ Error occurred in install.sh:"
    echo "  Line: $1"
    echo "  Exit code: $2"
    echo "Please check the error message above and try again."
    exit 1
}

# Set up error handling
trap 'handle_error ${LINENO} $?' ERR

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check Python version is >= 3.8 and < 3.13
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [ "$(printf '%s\n' "3.8" "$python_version" | sort -V | head -n1)" != "3.8" ] || [ "$(printf '%s\n' "3.13" "$python_version" | sort -V | head -n1)" != "$python_version" ]; then
    echo "❌ Python version must be between 3.8 and 3.12. Found version: $python_version"
    exit 1
fi

# Check disk space before starting
echo "🔍 Checking system requirements..."
required_space=2048  # 2GB in MB
available_space=$(df -m . | awk 'NR==2 {print $4}')
if [ "$available_space" -lt "$required_space" ]; then
    echo "❌ Insufficient disk space. Required: 2GB, Available: $((available_space/1024))GB"
    exit 1
fi

# Check memory
total_memory=$(sysctl -n hw.memsize 2>/dev/null || free -b | awk '/^Mem:/{print $2}')
total_memory_gb=$((total_memory/1024/1024/1024))
if [ "$total_memory_gb" -lt 4 ]; then
    echo "⚠️  Warning: Less than 4GB RAM detected. Training performance may be impacted."
fi

# Check if virtual environment exists
if [ ! -d "drl-env" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv drl-env
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source drl-env/bin/activate || {
    echo "❌ Failed to activate virtual environment."
    exit 1
}

# Verify pip installation
echo "🔍 Verifying pip installation..."
if ! command -v pip &> /dev/null; then
    echo "❌ pip not found. Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py --user
    rm get-pip.py
fi

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip

# Clean install - remove existing package if present
pip uninstall -y agent-arcade || true

# Install core dependencies first
echo "📥 Installing core dependencies..."
pip install "gymnasium[atari,other]==0.28.1" || {
    echo "❌ Failed to install Gymnasium with Atari support."
    exit 1
}

echo "📥 Installing ALE interface..."
pip install "ale-py==0.8.1" || {
    echo "❌ Failed to install ALE interface."
    exit 1
}

# Verify ALE interface
echo "🎮 Verifying ALE interface..."
python3 -c "
import ale_py
from ale_py import ALEInterface

ale = ALEInterface()
print(f'ALE interface version: {ale_py.__version__}')
" || {
    echo "❌ ALE interface verification failed."
    exit 1
}

# Install AutoROM with license acceptance
echo "🎲 Installing AutoROM..."
pip install "AutoROM[accept-rom-license]==0.6.1" || {
    echo "❌ Failed to install AutoROM."
    exit 1
}

# Install Atari ROMs
echo "🎲 Installing Atari ROMs..."

# Get ROM directory
ROM_DIR=$(python3 -c "import ale_py; from pathlib import Path; print(Path(ale_py.__file__).parent / 'roms')")
mkdir -p "$ROM_DIR"

# Try installing ROMs using AutoROM first
echo "Attempting to install ROMs using AutoROM..."
python3 -m AutoROM.cli --accept-rom-license --install-dir "$ROM_DIR" || {
    echo "AutoROM installation failed. Checking for local ROMs..."
    
    # Check if ROMs exist in local directory
    if [ -f "roms/pong/pong.bin" ] && [ -f "roms/space_invaders/space_invaders.bin" ]; then
        echo "Found local ROMs. Copying to ALE directory..."
        # Copy ROMs from local directory
        python3 -c "
import os
import sys
import shutil
from pathlib import Path

try:
    # Source ROM directories
    source_dir = Path('roms')
    target_dir = Path('$ROM_DIR')
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy Pong ROM
    pong_source = source_dir / 'pong' / 'pong.bin'
    shutil.copy2(pong_source, target_dir / 'pong.bin')
    print('✅ Copied pong.bin')
        
    # Copy Space Invaders ROM
    space_source = source_dir / 'space_invaders' / 'space_invaders.bin'
    shutil.copy2(space_source, target_dir / 'space_invaders.bin')
    print('✅ Copied space_invaders.bin')
        
except Exception as e:
    print(f'❌ Failed to copy ROMs: {e}', file=sys.stderr)
    sys.exit(1)
"
    else
        echo "❌ ROM installation failed."
        echo ""
        echo "Please install the required ROMs manually using one of these methods:"
        echo ""
        echo "Method 1: Using AutoROM (Recommended)"
        echo "  python3 -m AutoROM.cli --accept-rom-license"
        echo ""
        echo "Method 2: Manual ROM Installation"
        echo "1. Create the ROM directory:"
        echo "   mkdir -p \"$ROM_DIR\""
        echo ""
        echo "2. Download the required ROMs:"
        echo "   - Pong ROM: https://github.com/openai/atari-py/raw/master/atari_py/atari_roms/pong.bin"
        echo "   - Space Invaders ROM: https://github.com/openai/atari-py/raw/master/atari_py/atari_roms/space_invaders.bin"
        echo ""
        echo "3. Move the downloaded ROMs to:"
        echo "   $ROM_DIR/pong.bin"
        echo "   $ROM_DIR/space_invaders.bin"
        echo ""
        echo "4. Set correct permissions:"
        echo "   chmod 644 \"$ROM_DIR\"/*.bin"
        echo ""
        echo "After installing the ROMs, run this script again."
        exit 1
    fi
}

# Verify ROM installation
echo "✅ Verifying ROMs..."
python3 -c "
import os
from pathlib import Path
import ale_py

rom_dir = Path(ale_py.__file__).parent / 'roms'
required_roms = ['pong.bin', 'space_invaders.bin']
missing_roms = [rom for rom in required_roms if not (rom_dir / rom).exists()]

if missing_roms:
    print('❌ Missing required ROMs:')
    for rom in missing_roms:
        print(f'  - {rom}')
    exit(1)
else:
    print('✅ All required ROMs are installed!')
    print(f'ROM directory: {rom_dir}')
    print('Available ROMs:')
    for rom in sorted(rom_dir.glob('*.bin')):
        print(f'  - {rom.name}')
" || {
    echo "❌ ROM verification failed."
    echo "Please ensure ROMs are installed correctly using the instructions above."
    exit 1
}

# Test ROM functionality
echo "🎮 Testing ROM functionality..."
python3 -c "
import gymnasium as gym
import numpy as np

def test_env(game_name):
    print(f'Testing {game_name}...')
    env = gym.make(f'ALE/{game_name}-v5', render_mode='rgb_array')
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray), f'{game_name}: Invalid observation type'
    assert obs.shape == (210, 160, 3), f'{game_name}: Invalid observation shape'
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float), f'{game_name}: Invalid reward type'
    env.close()
    print(f'✅ {game_name} ROM functional')

try:
    test_env('Pong')
    test_env('SpaceInvaders')
except Exception as e:
    print(f'❌ ROM functionality test failed: {e}')
    sys.exit(1)
" || {
    echo "❌ ROM functionality test failed."
    echo "Please ensure ROMs are correctly installed and compatible."
    exit 1
}

# Install the agent-arcade package
echo "📥 Installing Agent Arcade..."
pip install -e . || {
    echo "❌ Failed to install Agent Arcade package."
    exit 1
}

# Verify core dependencies
echo "🔍 Verifying core dependencies..."
python3 -c "
import gymnasium
import stable_baselines3
import torch
import numpy
import ale_py
print('✅ Core dependencies verified:')
print(f'  - Gymnasium version: {gymnasium.__version__}')
print(f'  - Stable-Baselines3 version: {stable_baselines3.__version__}')
print(f'  - PyTorch version: {torch.__version__}')
print(f'  - NumPy version: {numpy.__version__}')
print(f'  - ALE-py version: {ale_py.__version__}')
" || {
    echo "❌ Core dependency verification failed."
    exit 1
}

# Verify environments
echo "🎮 Verifying Atari environments..."
python3 -c "
import gymnasium as gym
for game in ['Pong', 'SpaceInvaders']:
    print(f'Testing {game}...')
    env = gym.make(f'ALE/{game}-v5', render_mode='rgb_array')
    env.reset()
    env.close()
    print(f'✅ {game} environment verified')
" || {
    echo "❌ Environment verification failed."
    exit 1
}

# Setup NEAR CLI if not installed
if ! command -v near &> /dev/null; then
    if ! command -v npm &> /dev/null; then
        echo "❌ npm is required for NEAR CLI but not installed."
        echo "Please install Node.js and npm first: https://nodejs.org/"
        exit 1
    fi
    echo "🌐 Installing NEAR CLI..."
    npm install -g near-cli || {
        echo "❌ NEAR CLI installation failed."
        exit 1
    }
fi

# Create necessary directories
mkdir -p models tensorboard videos

# Verify Agent Arcade CLI installation
echo "✅ Verifying Agent Arcade CLI..."
agent-arcade --version || {
    echo "❌ Agent Arcade CLI verification failed."
    exit 1
}

echo "🎉 Installation complete! Get started with: agent-arcade --help"
echo ""
echo "📚 Available commands:"
echo "  agent-arcade train         - Train an agent for a game"
echo "  agent-arcade evaluate      - Evaluate a trained model"
echo "  agent-arcade leaderboard   - View game leaderboards"
echo "  agent-arcade stake         - Manage stakes and evaluations"
echo "  agent-arcade wallet-cmd    - Manage NEAR wallet"
echo ""
echo "🎮 Try training your first agent:"
echo "  agent-arcade train pong --render"
echo ""
echo "📊 Monitor training progress:"
echo "  tensorboard --logdir ./tensorboard"

# Print system information
echo "📊 System Information:"
echo "  - OS: $(uname -s) $(uname -r)"
echo "  - Python: $(python3 --version)"
echo "  - Pip: $(pip --version)"
echo "  - CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || cat /proc/cpuinfo | grep 'model name' | head -n1 | cut -d':' -f2)"
echo "  - Memory: ${total_memory_gb}GB"
echo "  - Disk Space: $((available_space/1024))GB available"

# Print installation summary
echo "📝 Installation Summary:"
echo "  - Virtual Environment: drl-env"
echo "  - ROM Directory: $ROM_DIR"
echo "  - Models Directory: $(pwd)/models"
echo "  - Tensorboard Logs: $(pwd)/tensorboard"
echo "  - Video Recordings: $(pwd)/videos" 