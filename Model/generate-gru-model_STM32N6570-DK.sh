#!/bin/bash

set -eu # Exit on any error, Exit on unset variable

# Note: stedgeai v3.0.0+ is required.
# GRU models are best supported in ONNX format.
# If using TFLite, ensure it doesn't use "Select TF ops" (Variables).

MODEL_FILE="gru_v26_int8.tflite"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: $MODEL_FILE not found in Model/ directory."
    exit 1
fi

echo "Generating GRU model from $MODEL_FILE..."

# Generate using stedgeai
# --name gru_network will produce gru_network.c and stai_gru_network.c
# --address 0x70680000 ensures weights are referenced at this offset in xSPI2
stedgeai generate --model "$MODEL_FILE" \
    --target stm32n6 \
    --st-neural-art "default@user_neuralart_gru_STM32N6570-DK.json" \
    --input-data-type float32 \
    --output-data-type float32 \
    --optimization time \
    --name gru_network \
    --address 0x70680000 \
    --output st_ai_output_gru

# Copy files to the application model directory
mkdir -p STM32N6570-DK/GRU
cp st_ai_output_gru/gru_network.c STM32N6570-DK/GRU/
cp st_ai_output_gru/gru_network.h STM32N6570-DK/GRU/
cp st_ai_output_gru/stai_gru_network.c STM32N6570-DK/GRU/
cp st_ai_output_gru/stai_gru_network.h STM32N6570-DK/GRU/
cp st_ai_output_gru/gru_network_atonbuf.xSPI2.raw STM32N6570-DK/GRU/gru_network_data.xSPI2.bin

# Convert weights to HEX for flashing
# TCN weights were at 0x70680000. Let's use the same or check for overlap.
# MoveNet: 0x70380000..0x70634000
# TCN/GRU: 0x70680000 (enough margin)
arm-none-eabi-objcopy -I binary STM32N6570-DK/GRU/gru_network_data.xSPI2.bin --change-addresses 0x70680000 -O ihex STM32N6570-DK/GRU/gru_network_data.hex

echo "GRU model generated in STM32N6570-DK/GRU/"
