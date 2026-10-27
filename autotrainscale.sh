#!/usr/bin/env bash
# ==============================================================================
# autotrainscale.sh - Dynamic Auto-Scaling Training Harness for Octo VLA
# Tuned for NVIDIA L40S (48GB Ada Lovelace VRAM), A100 (40GB/80GB), H100, RTX 4090
# Dynamically configures VRAM allocation, JAX memory preallocation, batch size,
# thread pooling, and system resources.
# ==============================================================================

set -eo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
RESET='\033[0m'

echo -e "${BOLD}${BLUE}======================================================================${RESET}"
echo -e "${BOLD}${CYAN}        OCTO VLA AUTOTRAINSCALE: GPU & RESOURCE AUTO-TUNER            ${RESET}"
echo -e "${BOLD}${BLUE}======================================================================${RESET}"

# 0. Automatic Dependency Check & Auto-Installation
echo -e "${CYAN}[0/5] Checking Required Packages (Octo, dlimp, JAX GPU, etc.)...${RESET}"
python -c "import octo; import dlimp; import jax" 2>/dev/null || {
    echo -e "${YELLOW}Missing packages detected. Installing/upgrading requirements...${RESET}"
    pip install -r requirements.txt
}

# 1. Inspect System Architecture & Disk Space
AVAILABLE_DISK_GB=$(df -BG . | awk 'NR==2 {gsub(/G/,""); print $4}')
echo -e "${CYAN}[1/5] Checking Disk & Storage:${RESET} Available Workspace Disk: ${BOLD}${AVAILABLE_DISK_GB} GB${RESET}"
if [ "$AVAILABLE_DISK_GB" -lt 5 ]; then
    echo -e "${YELLOW}Warning: Low disk space (<5GB). Checkpoint saving might fail if space is exhausted.${RESET}"
fi

# 2. Inspect GPU & VRAM via nvidia-smi
HAS_NVIDIA=false
TOTAL_VRAM_MB=0
GPU_NAME="Unknown"

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || echo "Unknown")
    TOTAL_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 || echo 0)
    TOTAL_VRAM_MB=$(echo "$TOTAL_VRAM_MB" | tr -d ' ')
    if [ "$TOTAL_VRAM_MB" -gt 0 ]; then
        HAS_NVIDIA=true
    fi
fi

TOTAL_VRAM_GB=$((TOTAL_VRAM_MB / 1024))

echo -e "${CYAN}[2/5] Inspecting GPU Hardware:${RESET}"
if [ "$HAS_NVIDIA" = true ]; then
    echo -e "      Found GPU: ${BOLD}${GREEN}${GPU_NAME}${RESET}"
    echo -e "      Dedicated VRAM: ${BOLD}${GREEN}${TOTAL_VRAM_GB} GB (${TOTAL_VRAM_MB} MB)${RESET}"
else
    echo -e "      ${YELLOW}No active NVIDIA GPU detected via nvidia-smi. Falling back to CPU mode.${RESET}"
fi

# 3. Dynamic Scaling Decision Engine based on VRAM
# Octo Small (135M ViT) + 4-step Action Horizon + Diffusion Denoiser
# In VRAM: 
#   VRAM >= 40GB (L40S 48GB, A100 40/80GB, H100 80GB) -> Batch size 64, LR 4e-4, Prealloc 0.85
#   VRAM >= 24GB (RTX 4090, RTX 3090, RTX 6000, A10G) -> Batch size 32, LR 3.5e-4, Prealloc 0.80
#   VRAM >= 16GB (V100, T4, RTX 4080)                 -> Batch size 16, LR 3e-4, Prealloc 0.75
#   VRAM < 16GB / CPU                                 -> Batch size 8,  LR 2.5e-4, Prealloc 0.60

BATCH_SIZE=8
EPOCHS=35
LEARNING_RATE="3e-4"
JAX_MEM_PREALLOC="0.80"

if [ "$TOTAL_VRAM_GB" -ge 40 ]; then
    echo -e "      ${BOLD}${GREEN}High-Tier Workstation / Datacenter Profile Detected (L40S / A100 / H100)!${RESET}"
    BATCH_SIZE=64
    EPOCHS=40
    LEARNING_RATE="5e-4"
    JAX_MEM_PREALLOC="0.85"
elif [ "$TOTAL_VRAM_GB" -ge 22 ]; then
    echo -e "      ${BOLD}${GREEN}Prosumer High-VRAM Profile Detected (RTX 3090 / 4090 / 24GB)!${RESET}"
    BATCH_SIZE=32
    EPOCHS=35
    LEARNING_RATE="3.5e-4"
    JAX_MEM_PREALLOC="0.80"
elif [ "$TOTAL_VRAM_GB" -ge 14 ]; then
    echo -e "      ${BOLD}${GREEN}Mid-VRAM Profile Detected (T4 / V100 / 16GB)!${RESET}"
    BATCH_SIZE=16
    EPOCHS=30
    LEARNING_RATE="3e-4"
    JAX_MEM_PREALLOC="0.75"
else
    echo -e "      ${YELLOW}Low-VRAM / CPU Fallback Profile Detected.${RESET}"
    BATCH_SIZE=8
    EPOCHS=20
    LEARNING_RATE="2e-4"
    JAX_MEM_PREALLOC="0.50"
fi

# Allow CLI overrides if arguments provided: e.g. ./autotrainscale.sh --batch_size 128
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --lr) LEARNING_RATE="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo -e "${CYAN}[3/5] Auto-Scaled Configuration Parameters:${RESET}"
echo -e "      • Batch Size:       ${BOLD}${GREEN}${BATCH_SIZE}${RESET}"
echo -e "      • Training Epochs:  ${BOLD}${GREEN}${EPOCHS}${RESET}"
echo -e "      • Learning Rate:    ${BOLD}${GREEN}${LEARNING_RATE}${RESET}"
echo -e "      • XLA/JAX Prealloc: ${BOLD}${GREEN}${JAX_MEM_PREALLOC}${RESET}"

# 4. Environment Variables for JAX / XLA & PyTorch CUDA
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=${JAX_MEM_PREALLOC}
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=$(nproc || echo 4)

# 5. Dependency & GPU Driver Verification
echo -e "${CYAN}[4/5] Checking JAX Accelerator Detection:${RESET}"
python -c "
import jax
devices = jax.devices()
print(f'      JAX Device Platform: {jax.default_backend().upper()}')
print(f'      Available Devices:   {devices}')
assert any(d.platform == 'gpu' for d in devices) or '$TOTAL_VRAM_GB' == '0', 'Warning: JAX cannot see CUDA GPU!'
" || {
    echo -e "${YELLOW}Notice: If JAX is currently running in CPU mode on a CUDA machine, run:${RESET}"
    echo -e "${BOLD}pip install --upgrade \"jax[cuda12_pip]\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html${RESET}"
}

# 6. Execute Scaled Training Run
echo -e "${CYAN}[5/5] Launching High-Throughput Octo Action Head Training...${RESET}"
echo -e "${BOLD}${BLUE}======================================================================${RESET}"

CMD="python train_dobot_octo_head.py \
    --dataset dobot_front_demonstrations.pkl \
    --checkpoint_dir octo_dobot_front_checkpoint \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LEARNING_RATE}"

echo -e "Executing: ${CYAN}${CMD}${RESET}\n"
exec $CMD
