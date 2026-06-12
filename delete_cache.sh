#!/bin/bash
# Script to delete EVERYTHING: Checkpoints, Processed Data, and Raw Data

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CKPT_DIR="$PROJECT_ROOT/robotmodel/models/checkpoints"
PROCESSED_DIR="$PROJECT_ROOT/data/processed"
PROCESSED_FILE="$PROJECT_ROOT/data/processed_bridge.parquet"
RAW_DATA_DIR="/home/jupyter-238w1a5447/bridge_v2_data"

echo "=== Full Cache Cleanup ==="

echo "1. Deleting Model Checkpoints..."
rm -rf "$CKPT_DIR"/*

echo "2. Deleting Processed Dataset..."
rm -rf "$PROCESSED_DIR"/*
rm -f "$PROCESSED_FILE"

echo "3. Deleting Raw Dataset (BridgeData V2)..."
rm -rf "$RAW_DATA_DIR"

echo "=== Cleanup Complete ==="
