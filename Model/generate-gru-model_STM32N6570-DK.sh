#!/bin/bash

set -eu # Exit on any error, Exit on unset variable

# P37-pure-vel-kp13-w40-gru: GRU(128,64) + 2×Conv1D(64,k=5,causal), kp13+velocity 74-feature, 40-step window
# Model: results/phase36_window_ablation/P37-pure-vel-kp13-w40-gru/model_best.keras
# window MinPR=0.9350, event_vote MinPR=0.8991, FN=6 (best), threshold=0.50
# Compat keras: p37_vel_kp13_compat.keras (strip quantization_config fields)
#
# Weights (641 KB) are placed in XIP flash at 0x70680000 via linker .gru_weights section.
# gru_network_data.hex is generated from the build ELF after "make FALL_DETECTION_MODEL=GRU":
#   make ../../Model/STM32N6570-DK/GRU/gru_network_data.hex
# or simply use "make flash_gru" which runs all steps.
#
# Note: stedgeai 4.0 with --type keras. Do NOT use --st-neural-art (incompatible with
#       address/binary args) and do NOT use --address (NeuralArt E102 error).

MODEL_FILE="p37_vel_kp13_compat.keras"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: $MODEL_FILE not found in Model/ directory."
    echo "Generate from: results/phase36_window_ablation/P37-pure-vel-kp13-w40-gru/model_best.keras"
    echo "  python3 scripts/util/export_stedgeai.py  # strips quantization_config"
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
    --output st_ai_output_gru_p37vel

# Copy generated C files to application model directory
mkdir -p STM32N6570-DK/GRU
cp st_ai_output_gru_p37vel/gru_network.c         STM32N6570-DK/GRU/
cp st_ai_output_gru_p37vel/gru_network.h         STM32N6570-DK/GRU/
cp st_ai_output_gru_p37vel/gru_network_data.c    STM32N6570-DK/GRU/
cp st_ai_output_gru_p37vel/gru_network_data.h    STM32N6570-DK/GRU/
cp st_ai_output_gru_p37vel/gru_network_details.h STM32N6570-DK/GRU/

echo "GRU model generated in STM32N6570-DK/GRU/"
echo "Next steps:"
echo "  cd ../Application/STM32N6570-DK"
echo "  make FALL_DETECTION_MODEL=GRU GCC_PATH=<gcc_bin>"
echo "  make sign"
echo "  make ../../Model/STM32N6570-DK/GRU/gru_network_data.hex  # extract weights hex"
echo "  make flash_gru                                            # flash app + MoveNet + GRU weights"
