#!/bin/bash
set -e

# Function to check if running on Apple Silicon
check_m3() {
    if [[ $(uname -m) == "arm64" ]]; then
        return 0  # true
    else
        return 1  # false
    fi
}

# Function to check if CUDA is available
check_cuda() {
    if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        return 0  # true
    else
        return 1  # false
    fi
}

# Configure environment based on hardware
if check_m3; then
    echo "🍎 Detected Apple Silicon (M3 Max)"
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    DEVICE="mps"
    # M3 Max optimized batch sizes and parallel environments
    PONG_BATCH_SIZE=512
    PONG_N_ENVS=8
    INVADERS_BATCH_SIZE=1024
    INVADERS_N_ENVS=8
    RAID_BATCH_SIZE=1024
    RAID_N_ENVS=16
elif check_cuda; then
    echo "🚀 Detected CUDA GPU (GH200)"
    DEVICE="cuda"
    # GH200 optimized batch sizes and parallel environments
    PONG_BATCH_SIZE=1024
    PONG_N_ENVS=16
    INVADERS_BATCH_SIZE=2048
    INVADERS_N_ENVS=32
    RAID_BATCH_SIZE=4096
    RAID_N_ENVS=64
else
    echo "⚠️  No GPU detected, using CPU (not recommended for training)"
    DEVICE="cpu"
    # Conservative CPU settings
    PONG_BATCH_SIZE=32
    PONG_N_ENVS=4
    INVADERS_BATCH_SIZE=32
    INVADERS_N_ENVS=4
    RAID_BATCH_SIZE=32
    RAID_N_ENVS=4
fi

# Create output directories
mkdir -p models/{pong,space_invaders,river_raid}/baseline/checkpoints
mkdir -p tensorboard

echo "🎮 Starting Agent Arcade Training Pipeline"
echo "Device: $DEVICE"
echo "Start time: $(date)"

# Start TensorBoard in background
tensorboard --logdir ./tensorboard --port 6006 &
TENSORBOARD_PID=$!

# Function to update config file
update_config() {
    local game=$1
    local batch_size=$2
    local n_envs=$3
    local config_file="models/$game/config.yaml"
    
    # Create temporary config with updated values
    sed -i.bak "
        s/batch_size:.*/batch_size: $batch_size/g;
        s/n_envs:.*/n_envs: $n_envs/g;
        s/device:.*/device: \"$DEVICE\"/g;
    " "$config_file"
}

# Pong Training
echo "🏓 Training Pong..."
echo "Configuration: batch_size=$PONG_BATCH_SIZE, n_envs=$PONG_N_ENVS"
update_config "pong" $PONG_BATCH_SIZE $PONG_N_ENVS
agent-arcade train pong \
    --config models/pong/config.yaml \
    --checkpoint-freq 100000 \
    --output-dir models/pong/baseline \
    || { echo "❌ Pong training failed"; exit 1; }

# Space Invaders Training
echo "👾 Training Space Invaders..."
echo "Configuration: batch_size=$INVADERS_BATCH_SIZE, n_envs=$INVADERS_N_ENVS"
update_config "space_invaders" $INVADERS_BATCH_SIZE $INVADERS_N_ENVS
agent-arcade train space-invaders \
    --config models/space_invaders/config.yaml \
    --checkpoint-freq 200000 \
    --output-dir models/space_invaders/baseline \
    || { echo "❌ Space Invaders training failed"; exit 1; }

# River Raid Training
echo "🚁 Training River Raid..."
echo "Configuration: batch_size=$RAID_BATCH_SIZE, n_envs=$RAID_N_ENVS"
update_config "river_raid" $RAID_BATCH_SIZE $RAID_N_ENVS
agent-arcade train river-raid \
    --config models/river_raid/config.yaml \
    --checkpoint-freq 300000 \
    --output-dir models/river_raid/baseline \
    || { echo "❌ River Raid training failed"; exit 1; }

# Kill TensorBoard
kill $TENSORBOARD_PID

# Restore original configs
for game in pong space_invaders river_raid; do
    mv "models/$game/config.yaml.bak" "models/$game/config.yaml"
done

echo "✅ Training complete!"
echo "End time: $(date)"

# Run quick evaluation of all models
echo "🔍 Running evaluation..."
for game in pong space_invaders river_raid; do
    echo "Evaluating $game..."
    agent-arcade test-evaluate $game "models/$game/baseline/final_model.zip" --episodes 20 --no-render
done

echo "📊 Training Summary:"
echo "Check tensorboard logs at: ./tensorboard"
echo "Models saved in: ./models/*/baseline" 