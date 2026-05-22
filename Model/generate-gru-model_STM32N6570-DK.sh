#!/bin/bash

set -eu # Exit on any error, Exit on unset variable

# P27-vm0: GRU(128,64) + 2×Conv1D(64,k=5,causal), kp7 27-feature, 40-step window
# Model: results/phase27_seed_sweep/P27-vm0/model_stedgeai_compat.keras
#
# INT8 workflow (current):
#   1. Convert float32 stateful TFLite → STEdgeAI ONNX (export-onnx)
#   2. Quantize STEdgeAI ONNX → INT8 QDQ ONNX using ORT (scripts/quantize_p27_stedgeai_int8.py)
#   3. Generate C code from INT8 ONNX (this script)
#
#   Result: weights 138 KB (vs 537 KB float32, -74%), activations 7.5 KB
#   I/O: c1_window=float32(1,5,27), h1/h2/c2=int8 states, softmax out=int8
#
# Step 1: Export STEdgeAI ONNX (run once, from st_ai_ws/ directory):
#   stedgeai export-onnx \
#       --model p27_stateful_rebuilt.tflite \
#       --type tflite \
#       --target stm32n6 \
#       --name gru_network \
#       --output st_ai_output_onnx
#
# Step 2: Quantize (run from repo root):
#   python3 scripts/quantize_p27_stedgeai_int8.py
#
# Step 3: Generate C code (this script, run from st_ai_ws/ directory):

STEDGEAI="/home/min/app/ST/STEdgeAI/4.0/Utilities/linux/stedgeai"
INT8_ONNX="p27_stateful_stedgeai_int8.onnx"
OUTPUT_DIR="st_ai_output_int8"
DEST_DIR="STM32N6570-DK/GRU"

if [ ! -f "$INT8_ONNX" ]; then
    echo "Error: $INT8_ONNX not found."
    echo "Run: python3 scripts/quantize_p27_stedgeai_int8.py"
    exit 1
fi

echo "Generating INT8 GRU model from $INT8_ONNX..."

"$STEDGEAI" generate \
    --model "$INT8_ONNX" \
    --type onnx \
    --target stm32n6 \
    --optimization time \
    --name gru_network \
    --output "$OUTPUT_DIR"

mkdir -p "$DEST_DIR"
cp "$OUTPUT_DIR/gru_network.c"          "$DEST_DIR/"
cp "$OUTPUT_DIR/gru_network.h"          "$DEST_DIR/"
cp "$OUTPUT_DIR/gru_network_data.c"     "$DEST_DIR/"
cp "$OUTPUT_DIR/gru_network_data.h"     "$DEST_DIR/"
cp "$OUTPUT_DIR/gru_network_details.h"  "$DEST_DIR/"

echo "INT8 GRU model generated in $DEST_DIR/"
echo "Weights are compiled into gru_network_data.c (no external flash needed)."
