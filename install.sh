#!/bin/bash
set -e

echo "🎮 Installing Agent Arcade..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check Python version is >= 3.8
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [ "$(printf '%s\n' "3.8" "$python_version" | sort -V | head -n1)" != "3.8" ]; then
    echo "❌ Python 3.8 or higher is required. Found version: $python_version"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "drl-env" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv drl-env
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source drl-env/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install package in editable mode with all dependencies
echo "📥 Installing Agent Arcade..."
pip install -e .

# Verify core dependencies
echo "🔍 Verifying core dependencies..."
python3 -c "import gymnasium; import stable_baselines3; import torch; import numpy; import py_near" || {
    echo "❌ Core dependency verification failed."
    exit 1
}

# Install and verify Atari ROMs
echo "🎯 Installing Atari ROMs..."
pip install "AutoROM[accept-rom-license]"
python3 -m AutoROM --accept-license

# Verify Atari environment
echo "🎮 Verifying Atari environment..."
python3 -c "import gymnasium; gymnasium.make('ALE/Pong-v5', render_mode='rgb_array')" || {
    echo "❌ Atari environment verification failed."
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
    npm install -g near-cli
fi

# Verify NEAR CLI installation
if ! command -v near &> /dev/null; then
    echo "❌ NEAR CLI installation failed."
    exit 1
fi

# Verify Agent Arcade CLI installation
echo "✅ Verifying Agent Arcade CLI..."
agent-arcade --version

echo "🎉 Installation complete! Get started with: agent-arcade --help"
echo ""
echo "📚 Available commands:"
echo "  agent-arcade list-games    - List available games"
echo "  agent-arcade train         - Train an agent"
echo "  agent-arcade evaluate      - Evaluate a trained agent"
echo "  agent-arcade login         - Login to NEAR wallet" 