#!/bin/bash

set -eu

# P38-nv-a65-gru: GRU(128,64) + 2×Conv1D(64,k=5,causal), kp13 45-feature (no velocity), 40-step window
# Source:   results/phase36_window_ablation/P38-nv-a65-gru/
# Metrics:  win MinPR=0.9345, event_vote MinPR=0.9079, FP_vid=21, FN_vid=8, threshold=0.725
# STedgeAI: weights=590 KiB, activations=33 KiB, MACC=5.84M
#
# Normalization and post-processing params are in p38_nv_a65_kp13/:
#   normalization.json  — MinMax per-feature (45 values each)
#   feature_columns.json — feature order
#
# Post-processing (firmware): vote_window=5, vote_k=3, threshold=0.725
#   → 5연속 창 중 3개 이상 fall score ≥ 0.725 → alarm

MODEL_FILE="p38_nv_a65_kp13_compat.keras"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: $MODEL_FILE not found in Model/ directory."
    echo "Copy from: Falling-Model-Development/results/phase36_window_ablation/P38-nv-a65-gru/model_stedgeai_compat.keras"
    exit 1
fi

echo "Generating GRU model from $MODEL_FILE..."

stedgeai generate \
    --model "$MODEL_FILE" \
    --type keras \
    --target stm32n6 \
    --input-data-type float32 \
    --output-data-type float32 \
    --compression high \
    --optimization time \
    --name gru_network \
    --output st_ai_output_gru_p38nv

mkdir -p STM32N6570-DK/GRU
cp st_ai_output_gru_p38nv/gru_network.c         STM32N6570-DK/GRU/
cp st_ai_output_gru_p38nv/gru_network.h         STM32N6570-DK/GRU/
cp st_ai_output_gru_p38nv/gru_network_data.c    STM32N6570-DK/GRU/
cp st_ai_output_gru_p38nv/gru_network_data.h    STM32N6570-DK/GRU/
cp st_ai_output_gru_p38nv/gru_network_details.h STM32N6570-DK/GRU/

echo "Done. GRU model generated in STM32N6570-DK/GRU/"
echo "Next steps:"
echo "  cd ../Application/STM32N6570-DK"
echo "  make FALL_DETECTION_MODEL=GRU GCC_PATH=<gcc_bin>"
echo "  make sign"
echo "  make ../../Model/STM32N6570-DK/GRU/gru_network_data.hex"
echo "  make flash_gru"
