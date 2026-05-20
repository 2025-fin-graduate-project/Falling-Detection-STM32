#!/bin/bash

set -eu

# P37-pure-h64-kp13-w40-gru: GRU(64,32) + 2×Conv1D(64,k=5,causal), kp13 45-feature, 40-step window
# Source:   results/phase36_window_ablation/P37-pure-h64-kp13-w40-gru/
# Metrics:  win MinPR=0.9327, event_vote MinPR=0.8947, FP=24, FN=7, threshold=0.725
# STedgeAI: weights=275 KiB, activations=21.5 KiB, MACC=2.76M
#
# Post-processing (firmware): vote_window=5, vote_k=3, threshold=0.725

MODEL_FILE="p37_pure_h64_kp13_compat.keras"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: $MODEL_FILE not found in Model/ directory."
    echo "Copy from: Falling-Model-Development/results/phase36_window_ablation/P37-pure-h64-kp13-w40-gru/model_stedgeai_compat.keras"
    exit 1
fi

echo "Generating GRU model from $MODEL_FILE..."

stedgeai generate \
    --model "$MODEL_FILE" \
    --type keras \
    --target stm32n6 \
    --input-data-type float32 \
    --output-data-type float32 \
    --optimization time \
    --name gru_network \
    --output st_ai_output_gru_p37h64

mkdir -p STM32N6570-DK/GRU
cp st_ai_output_gru_p37h64/gru_network.c         STM32N6570-DK/GRU/
cp st_ai_output_gru_p37h64/gru_network.h         STM32N6570-DK/GRU/
cp st_ai_output_gru_p37h64/gru_network_data.c    STM32N6570-DK/GRU/
cp st_ai_output_gru_p37h64/gru_network_data.h    STM32N6570-DK/GRU/
cp st_ai_output_gru_p37h64/gru_network_details.h STM32N6570-DK/GRU/

echo "Done. Files copied to STM32N6570-DK/GRU/"
