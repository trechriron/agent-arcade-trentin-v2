#!/bin/bash
set -e

# M3 Max specific optimizations
echo "🍎 Configuring for M3 Max..."
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8  # Allow more GPU memory usage
export PYTORCH_MPS_ALLOCATOR_POLICY=garbage_collection  # Enable better memory management
export OMP_NUM_THREADS=8  # Optimize CPU thread usage
export MKL_NUM_THREADS=8  # Optimize MKL thread usage
DEVICE="mps"

# Create necessary directories
mkdir -p models/{pong,space_invaders,river_raid}/{checkpoints,videos}
mkdir -p tensorboard

echo "🎮 Starting Agent Arcade Training Pipeline"
echo "Device: $DEVICE"
echo "Start time: $(date)"

# Kill any existing tensorboard processes
pkill -f tensorboard || true

# Start tensorboard in background
tensorboard --logdir ./tensorboard --port 6006 &
sleep 2  # Give tensorboard time to start

echo "Starting training pipeline..."
echo "Monitor progress at http://localhost:6006"

# Function to run training with proper error handling
train_game() {
    local game=$1
    local expected_time=$2
    echo "🎯 Training $game..."
    echo "Expected training time: $expected_time"
    
    if ! agent-arcade train $game; then
        echo "❌ Training failed for $game"
        return 1
    fi
    echo "✅ Training completed for $game"
}

# Train each game with estimated times
echo "🎮 Training Schedule:"
echo "1. Pong (~30 mins)"
echo "2. Space Invaders (~1 hour)"
echo "3. River Raid (~2 hours)"
echo ""

# Train Pong (500k steps, ~30 mins)
train_game "pong" "30 minutes"

# Train Space Invaders (1M steps, ~1 hour)
train_game "space_invaders" "1 hour"

# Train River Raid (2M steps, ~2 hours)
train_game "riverraid" "2 hours"

echo "Training complete! Models saved in models/ directory"

# Comprehensive evaluation
echo "🔍 Running evaluation..."
for game in pong space_invaders river_raid; do
    echo "Evaluating $game..."
    # More episodes for reliable metrics
    agent-arcade evaluate $game "models/${game}/${game}_final.zip" --episodes 100 --no-render
done

echo "📊 Training Summary:"
echo "Check tensorboard logs at: ./tensorboard"
echo "Models saved in: ./models/"

# Print environment information
echo "🔧 Environment Information:"
echo "- Hardware: Apple M3 Max"
echo "- Device: $DEVICE"
echo "- PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "- Gymnasium: $(python3 -c 'import gymnasium; print(gymnasium.__version__)')"
echo "- ALE: $(python3 -c 'import ale_py; print(ale_py.__version__)')"

# Print target metrics from training guide
echo "🎯 Target Metrics:"
echo "- Pong: Score > 15 (500k steps)"
echo "- Space Invaders: Score > 500 (1M steps)"
echo "- River Raid: Score > 10000 (2M steps)"

# Print ALE settings used
echo "🎮 ALE Settings:"
echo "- Frame Skip: 4"
echo "- Sticky Actions: 25% (v5 default)"
echo "- Observation: Grayscale (84x84)"
echo "- Frame Stack: 4 frames" 