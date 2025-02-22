#!/bin/bash
set -e

echo "🎮 Installing Agent Arcade GPU dependencies..."

# Create virtual environment if it doesn't exist
if [ ! -d "drl-env" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv drl-env
fi

# Activate virtual environment
source drl-env/bin/activate

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install "torch>=2.3.0"
pip install "gymnasium[atari]>=0.29.1" "gymnasium[accept-rom-license]>=0.29.1"
pip install "ale-py==0.10.1"
pip install "shimmy[atari]>=2.0.0"
pip install "stable-baselines3[extra]>=2.5.0"

# Install AutoROM
echo "🎲 Installing AutoROM..."
pip install "autorom>=0.6.1"
AutoROM --accept-license

# Copy ROMs to ALE-py directory
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

# Create directories for outputs
mkdir -p models tensorboard videos

echo "✅ GPU Installation complete!"
echo ""
echo "📊 Available directories:"
echo "  - Models: $(pwd)/models"
echo "  - Tensorboard Logs: $(pwd)/tensorboard"
echo "  - Video Recordings: $(pwd)/videos"
echo ""
echo "🎮 Try training your first agent:"
echo "  agent-arcade train pong --steps 1000000" 