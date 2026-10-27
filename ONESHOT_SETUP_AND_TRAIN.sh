#!/usr/bin/env bash
# ==============================================================================
# ONESHOT_SETUP_AND_TRAIN.sh
# Complete self-healing Linux GPU setup & Octo VLA training pipeline
# Run once on your VM. Automatically cleans up version conflicts, fixes Flax/
# Transformers/TFP compatibility, and launches training on your L40S GPU.
# ==============================================================================

set -eo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RESET='\033[0m'

echo -e "${BOLD}${CYAN}======================================================================${RESET}"
echo -e "${BOLD}${GREEN}   DOBOT OCTO VLA: ONE-SHOT GPU SETUP & TRAINING PIPELINE             ${RESET}"
echo -e "${BOLD}${CYAN}======================================================================${RESET}"

# Step 1: Enforce clean package versions for Linux CUDA environment
echo -e "\n${CYAN}[1/5] Aligning Python package versions (NumPy 1.x, TF-Keras, JAX CUDA)...${RESET}"
pip install --upgrade "numpy<2.0.0" tf-keras
pip install --upgrade "jax[cuda12_pip]>=0.4.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Step 2: Ensure Berkeley dlimp and Octo are installed
echo -e "\n${CYAN}[2/5] Checking Octo and Berkeley dlimp packages...${RESET}"
python -c "import octo; import dlimp" 2>/dev/null || {
    echo -e "${YELLOW}Installing Octo & dlimp from source repositories...${RESET}"
    pip install git+https://github.com/kvablack/dlimp.git@92e3eca97af3b14d0b6aa15182c0dc240407698d
    pip install git+https://github.com/octo-models/octo.git@241fb3514b7c40957a86d869fecb7c7fc353f540
}

# Step 3: Check and Download Pretrained Octo Weights
echo -e "\n${CYAN}[3/5] Verifying Pretrained Octo Small Checkpoint...${RESET}"
if [ ! -f "octo_small_checkpoint/config.json" ] || ([ ! -f "octo_small_checkpoint/270000/default/checkpoint" ] && [ ! -f "octo_small_checkpoint/300000/default/checkpoint" ]); then
    echo -e "${YELLOW}Downloading Octo Small checkpoint (rail-berkeley/octo-small)...${RESET}"
    python download_octo_checkpoint.py
fi

# Step 4: Validate GPU Detection
echo -e "\n${CYAN}[4/5] Checking JAX CUDA GPU Acceleration...${RESET}"
python -c "
import jax
devices = jax.devices()
print(f'  Active Backend: {jax.default_backend().upper()}')
print(f'  Available Devices: {devices}')
assert any(d.platform == 'gpu' for d in devices), 'JAX is not detecting GPU! Check CUDA driver.'
"

# Step 5: Launch Training with Dynamic GPU VRAM Tuning
echo -e "\n${CYAN}[5/5] Launching High-Throughput Octo Action Head Training...${RESET}"
echo -e "${BOLD}${CYAN}======================================================================${RESET}\n"

chmod +x autotrainscale.sh
./autotrainscale.sh
