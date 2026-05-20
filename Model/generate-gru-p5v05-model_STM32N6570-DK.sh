#!/bin/bash
# P5-v05: GRU(64,32) + 2×Conv1D(64,k=5,causal), kp7 27-feature, 40-step window
#
# Metrics:  test_video MinP=0.9231  (no ev_vote eval)
# STedgeAI: weights=221 KiB, activations=21.5 KiB, MACC=2.2M
#
# Feature set (27): kp0,kp5,kp6,kp7,kp8,kp11,kp12 × (y,x,s) + HSSC_y/x,RWHC,VHSSC,AHSSC,AHSSC_x
# → Current kp13(45feat) 벡터에서 인덱스: [0-8, 9-14, 21-26, 39-44]
#
# Firmware 변경 사항:
#   Inc/app_config.h : POSE_FEATURE_COUNT 45→27, POSE_KP_SUBSET 정의
#   Src/pose_pipeline.c : kp13_indices 배열 수정 (7개 keypoint만)
#   Src/fall_detection.c: norm_min/norm_scale 배열 교체

set -eu
cd "$(dirname "$0")"

MODEL_FILE="p5_v05_kp7_compat.keras"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: $MODEL_FILE not found"
    echo "Run: cp Falling-Model-Development/results/training/gru_phase5_compact/P5-v05/model_stedgeai_compat.keras Model/p5_v05_kp7_compat.keras"
    exit 1
fi

echo "Generating GRU model (P5-v05, kp7/27feat) from $MODEL_FILE..."

stedgeai generate \
    --model "$MODEL_FILE" \
    --type keras \
    --target stm32n6 \
    --input-data-type float32 \
    --output-data-type float32 \
    --optimization time \
    --name gru_network \
    --output st_ai_output_gru_p5v05

mkdir -p STM32N6570-DK/GRU
cp st_ai_output_gru_p5v05/gru_network.c         STM32N6570-DK/GRU/
cp st_ai_output_gru_p5v05/gru_network.h         STM32N6570-DK/GRU/
cp st_ai_output_gru_p5v05/gru_network_data.c    STM32N6570-DK/GRU/
cp st_ai_output_gru_p5v05/gru_network_data.h    STM32N6570-DK/GRU/
cp st_ai_output_gru_p5v05/gru_network_details.h STM32N6570-DK/GRU/

echo "Done. GRU model generated in STM32N6570-DK/GRU/"
