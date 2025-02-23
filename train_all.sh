#!/bin/bash
set -e

# GH200 GPU specific optimizations
echo "🚀 Configuring for Lambda Labs GH200..."
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

# Create necessary directories
mkdir -p models/{pong,space_invaders,river_raid}/{checkpoints,videos}
mkdir -p tensorboard

echo "🎮 Starting Agent Arcade Training Pipeline"
echo "Device: cuda"
echo "Start time: $(date)"

# Kill any existing tensorboard processes
pkill -f tensorboard || true

# Start tensorboard in background
tensorboard --logdir ./tensorboard --port 6006 --bind_all &
sleep 2  # Give tensorboard time to start

echo "Starting training pipeline..."
echo "Monitor progress at http://localhost:6006"

# Function to run training with proper error handling and checkpointing
train_game() {
    local game=$1
    local expected_time=$2
    echo "🎯 Training $game..."
    echo "Expected training time: $expected_time"
    
    # Create checkpoint directory
    mkdir -p "models/${game}/checkpoints"
    
    if ! agent-arcade train $game \
        --config "models/${game}/config.yaml" \
        --output-dir "models/${game}" \
        --checkpoint-freq 100000; then
        echo "❌ Training failed for $game"
        return 1
    fi
    
    # Run comprehensive evaluation
    echo "Evaluating ${game}..."
    agent-arcade evaluate $game "models/${game}/${game}_final.zip" \
        --episodes 200 --no-render
    
    echo "✅ Training completed for $game"
}

# Print GPU information
echo "🔧 GPU Information:"
nvidia-smi
python3 -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name()}')"

# Train each game with estimated times
echo "🎮 Training Schedule:"
echo "1. Pong (~1 hour with parallel envs)"
echo "2. Space Invaders (~2 hours with parallel envs)"
echo "3. River Raid (~3 hours with parallel envs)"
echo ""

# Train games sequentially with proper GPU utilization
train_game "pong" "1 hour"
train_game "space_invaders" "2 hours"
train_game "riverraid" "3 hours"

echo "Training complete! Models saved in models/ directory"

# Print final metrics
echo "📊 Training Summary:"
echo "Check tensorboard logs at: ./tensorboard"
echo "Models saved in: ./models/"

# Print environment information
echo "🔧 Environment Information:"
echo "- Hardware: Lambda Labs GH200"
echo "- PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "- Gymnasium: $(python3 -c 'import gymnasium; print(gymnasium.__version__)')"
echo "- ALE: $(python3 -c 'import ale_py; print(ale_py.__version__)')"

# Print target metrics
echo "🎯 Target Metrics:"
echo "- Pong: Score > 19 (95% win rate)"
echo "- Space Invaders: Score > 1000"
echo "- River Raid: Score > 12000"

# Print ALE settings
echo "🎮 ALE Settings:"
echo "- Frame Skip: 4"
echo "- Sticky Actions: 25% (v5 default)"
echo "- Observation: Grayscale (84x84)"
echo "- Frame Stack: 16 frames" 