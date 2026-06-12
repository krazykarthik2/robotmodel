#!/bin/bash
# Script to delete only model checkpoints

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CKPT_DIR="$PROJECT_ROOT/robotmodel/models/checkpoints"

echo "=== Deleting Model Checkpoints ==="
if [ -d "$CKPT_DIR" ]; then
    rm -rf "$CKPT_DIR"/*
    echo "Checkpoints in $CKPT_DIR cleared."
else
    echo "Checkpoint directory not found: $CKPT_DIR"
fi
