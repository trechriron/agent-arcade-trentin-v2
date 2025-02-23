#!/bin/bash
set -e

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Virtual environment not activated. Please run:"
    echo "   source drl-env/bin/activate"
    exit 1
fi

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

# Install ALE-py first
echo "Installing ALE-py..."
if ! pip install "ale-py==0.10.1"; then
    echo "❌ Failed to install ALE interface."
    exit 1
fi

# Install Shimmy for environment compatibility
echo "Installing Shimmy..."
if ! pip install "shimmy[atari]==0.2.1"; then
    echo "❌ Failed to install Shimmy."
    exit 1
fi

# Install Gymnasium with specific version
echo "Installing Gymnasium..."
if ! pip install "gymnasium[atari]==0.28.1" "gymnasium[accept-rom-license]==0.28.1" "gymnasium[other]==0.28.1"; then
    echo "❌ Failed to install Gymnasium."
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

# Create necessary directories
mkdir -p models tensorboard videos

# Print installation summary
echo "📝 Installation Summary:"
echo "  - Virtual Environment: $VIRTUAL_ENV"
echo "  - Python: $(python3 --version)"
echo "  - Pip: $(pip --version)"
echo "  - Node.js: $(node --version)"
echo "  - NEAR CLI: $(near --version)"

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