#!/bin/bash

set -eu # Exit on any error, Exit on unset variable

# P27-vm0: GRU(128,64) + 2×Conv1D(64,k=5,causal), kp7 27-feature, 40-step window
# Model: results/phase27_seed_sweep/P27-vm0/model_stedgeai_compat.keras
# INT8 MinP=0.9258, threshold=0.50, min_consecutive=1
#
# Note: stedgeai 4.0 with --type keras. Do NOT use --st-neural-art (incompatible with
#       address/binary args) and do NOT use --address (NeuralArt E102 error).

MODEL_FILE="p27_vm0_compat.keras"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: $MODEL_FILE not found in Model/ directory."
    echo "Copy from: results/phase27_seed_sweep/P27-vm0/model_stedgeai_compat.keras"
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
    --output st_ai_output_gru_p27

# Copy generated C files to application model directory
mkdir -p STM32N6570-DK/GRU
cp st_ai_output_gru_p27/gru_network.c       STM32N6570-DK/GRU/
cp st_ai_output_gru_p27/gru_network.h       STM32N6570-DK/GRU/
cp st_ai_output_gru_p27/gru_network_data.c  STM32N6570-DK/GRU/
cp st_ai_output_gru_p27/gru_network_data.h  STM32N6570-DK/GRU/
cp st_ai_output_gru_p27/gru_network_details.h STM32N6570-DK/GRU/

# Convert weights to HEX for xSPI2 flashing at 0x70680000
# MoveNet occupies 0x70380000..0x70634000; GRU weights start at 0x70680000
if [ -f "st_ai_output_gru_p27/gru_network_data.xSPI2.raw" ]; then
    cp st_ai_output_gru_p27/gru_network_data.xSPI2.raw STM32N6570-DK/GRU/gru_network_data.xSPI2.bin
    arm-none-eabi-objcopy -I binary STM32N6570-DK/GRU/gru_network_data.xSPI2.bin \
        --change-addresses 0x70680000 -O ihex STM32N6570-DK/GRU/gru_network_data.hex
    echo "Weight HEX generated at 0x70680000."
else
    echo "Note: no xSPI2.raw found — weights are compiled into gru_network_data.c directly."
fi

echo "GRU model generated in STM32N6570-DK/GRU/"
