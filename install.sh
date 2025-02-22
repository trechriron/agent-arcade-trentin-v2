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

# Check Python version is >= 3.9 and < 3.13
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [ "$(printf '%s\n' "3.9" "$python_version" | sort -V | head -n1)" != "3.9" ] || [ "$(printf '%s\n' "3.13" "$python_version" | sort -V | head -n1)" != "$python_version" ]; then
    echo "❌ Python version must be between 3.9 and 3.12. Found version: $python_version"
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

# Clean install - remove existing packages if present
echo "🧹 Cleaning existing installations..."
pip uninstall -y agent-arcade ale-py shimmy gymnasium || true

# Install dependencies in correct order with error handling
echo "📥 Installing core dependencies..."

# Install PyTorch first
echo "Installing PyTorch..."
if ! pip install "torch>=2.3.0"; then
    echo "❌ Failed to install PyTorch."
    exit 1
fi

# Install Gymnasium with Atari support first
echo "Installing Gymnasium with Atari support..."
if ! pip install "gymnasium[atari]>=0.29.1" "gymnasium[accept-rom-license]>=0.29.1"; then
    echo "❌ Failed to install Gymnasium with Atari support."
    exit 1
fi

# Install Gymnasium other dependencies (for video recording)
echo "Installing Gymnasium video recording support..."
if ! pip install "gymnasium[other]>=0.29.1"; then
    echo "❌ Failed to install Gymnasium video recording support."
    exit 1
fi

# Install ALE interface
echo "Installing latest ALE-py..."
if ! pip install "ale-py==0.10.1"; then
    echo "❌ Failed to install ALE interface."
    exit 1
fi

# Install Shimmy for environment compatibility
echo "Installing Shimmy..."
if ! pip install "shimmy[atari]>=2.0.0"; then
    echo "❌ Failed to install Shimmy."
    exit 1
fi

# Install Stable-Baselines3 after environment dependencies
echo "Installing Stable-Baselines3..."
if ! pip install "stable-baselines3[extra]>=2.5.0"; then
    echo "❌ Failed to install Stable-Baselines3."
    exit 1
fi

# Install standard-imghdr for TensorBoard compatibility
echo "Installing standard-imghdr for TensorBoard..."
if ! pip install "standard-imghdr>=3.13.0"; then
    echo "❌ Failed to install standard-imghdr."
    exit 1
fi

# Install AutoROM for ROM management
echo "🎲 Installing AutoROM..."
if ! pip install "autorom>=0.6.1"; then
    echo "❌ Failed to install AutoROM."
    exit 1
fi

# Install Atari ROMs using AutoROM
echo "🎲 Installing Atari ROMs..."
if ! AutoROM --accept-license; then
    echo "❌ Failed to install ROMs using AutoROM."
    exit 1
fi

# Verify AutoROM installation
echo "🔍 Verifying AutoROM installation..."
python3 -c "
import os
from pathlib import Path
import AutoROM

autorom_path = Path(AutoROM.__file__).parent / 'roms'
if not autorom_path.exists():
    print(f'❌ AutoROM directory not found: {autorom_path}')
    exit(1)

rom_files = list(autorom_path.glob('*.bin'))
if not rom_files:
    print(f'❌ No ROM files found in {autorom_path}')
    exit(1)

print(f'✅ Found {len(rom_files)} ROMs in {autorom_path}')
"

# Copy ROMs from AutoROM to ALE-py
echo "📂 Copying ROMs to ALE-py directory..."
python3 -c "
import os
import shutil
from pathlib import Path
import AutoROM
import ale_py

autorom_path = Path(AutoROM.__file__).parent / 'roms'
ale_path = Path(ale_py.__file__).parent / 'roms'

# Create ALE-py roms directory if it doesn't exist
ale_path.mkdir(parents=True, exist_ok=True)

# Copy all ROMs
for rom in autorom_path.glob('*.bin'):
    target = ale_path / rom.name
    print(f'Copying {rom.name} to {target}')
    shutil.copy2(rom, target)

print(f'✅ Copied ROMs from {autorom_path} to {ale_path}')
"

# Verify ROM paths and permissions
echo "🔍 Verifying ROM paths and permissions..."
python3 -c "
import os
from pathlib import Path
import ale_py

ale_path = Path(ale_py.__file__).parent / 'roms'
print(f'ALE ROM path: {ale_path}')

if not ale_path.exists():
    print(f'❌ ALE ROM directory not found')
    exit(1)

rom_files = list(ale_path.glob('*.bin'))
if not rom_files:
    print(f'❌ No ROM files found in ALE directory')
    exit(1)

print(f'Found {len(rom_files)} ROMs:')
for rom in rom_files:
    print(f'  - {rom.name} ({oct(rom.stat().st_mode)[-3:]})')
"

# Standardize ROM names
echo "📝 Standardizing ROM names..."
python3 -c "
import os
from pathlib import Path
import ale_py

ale_path = Path(ale_py.__file__).parent / 'roms'
name_mapping = {
    'pong.bin': 'pong.bin',
    'space_invaders.bin': 'space_invaders.bin',
    # Add more mappings if needed
}

for rom in ale_path.glob('*.bin'):
    lower_name = rom.name.lower()
    if lower_name in name_mapping and lower_name != rom.name:
        target = rom.parent / name_mapping[lower_name]
        print(f'Renaming {rom.name} to {target.name}')
        rom.rename(target)

# Ensure proper permissions
for rom in ale_path.glob('*.bin'):
    rom.chmod(0o644)
    print(f'Set permissions for {rom.name}')
"

# Verify ALE interface with environment registration
echo "🎮 Verifying ALE interface..."
python3 -c "
import gymnasium as gym
import ale_py
from ale_py import ALEInterface
from pathlib import Path
import sys

# Check ROM paths
ale_rom_path = Path(ale_py.__file__).parent / 'roms'
required_roms = ['pong.bin', 'space_invaders.bin']

print(f'Checking ROMs in: {ale_rom_path}')

missing_roms = [rom for rom in required_roms if not (ale_rom_path / rom).exists()]
if missing_roms:
    print('❌ Missing required ROMs in ALE-py directory:')
    for rom in missing_roms:
        print(f'  - {rom}')
    sys.exit(1)

# Register environments
gym.register_envs(ale_py)

ale = ALEInterface()
print(f'A.L.E: Arcade Learning Environment (version {ale_py.__version__})')
print('✅ Environment registration successful')

# Test environment creation with proper wrappers
print('Testing Pong environment...')
env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayscaleObservation(env)
env = gym.wrappers.FrameStackObservation(env, 16)

# Test a step to verify everything works
obs, _ = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
env.close()

print('✅ Environment test successful')
" || {
    echo "❌ ALE interface verification failed."
    echo "Please check the error message above and try again."
    exit 1
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

# Test ROM functionality with proper wrappers
echo "🎮 Testing ROM functionality..."
python3 -c "
import gymnasium as gym
import numpy as np
import sys
import ale_py

# Register ALE environments
gym.register_envs(ale_py)

def test_env(game_name):
    print(f'Testing {game_name}...')
    env = gym.make(f'ALE/{game_name}-v5', render_mode='rgb_array')
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.FrameStackObservation(env, 4)
    
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray), f'{game_name}: Invalid observation type'
    print(f'Observation shape: {obs.shape}')  # Debug print
    assert obs.shape == (4, 84, 84), f'{game_name}: Invalid observation shape, got {obs.shape}'
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
if ! pip install -e .; then
    echo "❌ Failed to install Agent Arcade package."
    exit 1
fi

# Check Node.js installation for NEAR CLI
echo "🔍 Checking Node.js installation..."
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required for NEAR CLI but not installed."
    echo "Please install Node.js from https://nodejs.org/"
    echo "Recommended version: 14.x or higher"
    exit 1
fi

# Verify Node.js version
node_version=$(node -v | cut -d'v' -f2)
if [ "$(printf '%s\n' "14.0.0" "$node_version" | sort -V | head -n1)" != "14.0.0" ]; then
    echo "❌ Node.js version must be 14.0.0 or higher. Found version: $node_version"
    exit 1
fi

# Install NEAR CLI if not present
if ! command -v near &> /dev/null; then
    echo "🌐 Installing NEAR CLI..."
    
    # Try installing without sudo first
    npm install -g near-cli 2>/dev/null || {
        echo "⚠️ Permission denied. Trying with sudo..."
        
        # Retry with sudo if the first attempt fails
        sudo npm install -g near-cli || {
            echo "❌ NEAR CLI installation failed."
            exit 1
        }
    }

    # Verify installation
    if command -v near &> /dev/null; then
        echo "✅ NEAR CLI installed successfully."
    else
        echo "❌ NEAR CLI installation failed even after sudo attempt."
        exit 1
    fi
else
    echo "✅ NEAR CLI is already installed."
fi

# Install staking dependencies
echo "📥 Installing staking dependencies..."
if ! pip install -e ".[staking]"; then
    echo "❌ Failed to install staking dependencies."
    exit 1
fi

# Verify NEAR CLI installation
echo "✅ Verifying NEAR CLI..."
if ! near --version; then
    echo "❌ NEAR CLI verification failed."
    exit 1
fi

# Ask if user wants to install NEAR integration
echo ""
echo "🌐 Would you like to install NEAR integration for staking? (y/N)"
read -r install_near

if [[ $install_near =~ ^[Yy]$ ]]; then
    echo "Installing NEAR integration..."
    if ! pip install -e ".[staking]"; then
        echo "❌ Failed to install NEAR integration."
        exit 1
    fi

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
if [[ $install_near =~ ^[Yy]$ ]]; then
    echo "  agent-arcade stake         - Manage stakes and evaluations"
    echo "  agent-arcade wallet-cmd    - Manage NEAR wallet"
fi
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
echo "  - ROM Directory: $(python3 -c "import ale_py; from pathlib import Path; print(Path(ale_py.__file__).parent / 'roms')")"
echo "  - Models Directory: $(pwd)/models"
echo "  - Tensorboard Logs: $(pwd)/tensorboard"
echo "  - Video Recordings: $(pwd)/videos" 