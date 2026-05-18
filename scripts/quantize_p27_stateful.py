#!/usr/bin/env python3
"""p27_stateful.tflite → int8 TFLite 양자화.

float32 stateful 모델에서 real 시퀀스를 실행해 GRU 상태 분포를 수집하고,
TFLite 전체 정수 양자화(full integer PTQ)를 수행한다.

Usage:
    python3 scripts/quantize_p27_stateful.py

Output:
    Model/st_ai_ws/p27_stateful_int8.tflite
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# ── 경로 ────────────────────────────────────────────────────────────────────
REPO   = Path(__file__).resolve().parents[1]
TFLITE = REPO / "Model/st_ai_ws/p27_stateful.tflite"
OUT    = REPO / "Model/st_ai_ws/p27_stateful_int8.tflite"
VAL_CSV = Path("/home/min/Workspace/Graduate-Project/Falling-Model-Development"
               "/dataset/splits_v2_class_balanced_filtered/val.csv")
FEAT_JSON = Path("/home/min/Workspace/Graduate-Project/Falling-Model-Development"
                 "/results/phase27_seed_sweep/P27-vm0/feature_columns.json")
NORM_JSON = Path("/home/min/Workspace/Graduate-Project/Falling-Model-Development"
                 "/results/phase27_seed_sweep/P27-vm0/normalization.json")

MAX_CALIB_SAMPLES = 2000  # 캘리브레이션에 사용할 최대 샘플 수

# ── 정규화 로드 ──────────────────────────────────────────────────────────────
feat_cols = json.loads(FEAT_JSON.read_text())
norm      = json.loads(NORM_JSON.read_text())
norm_min   = np.array(norm["min"],   dtype=np.float32)
norm_scale = np.array(norm["scale"], dtype=np.float32)

print(f"Feature count: {len(feat_cols)}")
assert len(feat_cols) == 27, f"Expected 27 features, got {len(feat_cols)}"


def load_sequences(csv_path: Path) -> dict[str, np.ndarray]:
    """CSV → video_id별 정규화된 프레임 시퀀스 딕셔너리.

    Returns: {video_id: (T, 27) float32}
    """
    videos: dict[str, list[np.ndarray]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video_id"]
            try:
                frame = np.array([float(row[c]) for c in feat_cols], dtype=np.float32)
            except (KeyError, ValueError):
                continue
            # MinMax 정규화
            frame = (frame - norm_min) / norm_scale
            frame = np.clip(frame, 0.0, 1.0)
            videos.setdefault(vid, []).append(frame)

    return {vid: np.stack(frames) for vid, frames in videos.items()
            if len(frames) > 2}


print("Loading validation sequences …")
sequences = load_sequences(VAL_CSV)
print(f"  {len(sequences)} videos, "
      f"{sum(len(v) for v in sequences.values())} frames total")


# ── float32 stateful 모델 실행 → 캘리브레이션 샘플 수집 ──────────────────────
print("Running float32 model to collect calibration samples …")

interp_f32 = tf.lite.Interpreter(model_path=str(TFLITE))
interp_f32.allocate_tensors()

inp_details = interp_f32.get_input_details()
out_details = interp_f32.get_output_details()

# 입력/출력 인덱스 이름으로 매핑
def _idx(details, name_substr):
    for d in details:
        if name_substr in d["name"]:
            return d["index"]
    raise KeyError(f"{name_substr!r} not found in {[d['name'] for d in details]}")

# 인덱스 직접 사용 (텐서 이름으로 확인)
# Inputs:  h1(0,[1,128])  c2_state(1,[1,2,64])  h2(2,[1,64])  pose_seq(3,[1,1,27])  c1_state(4,[1,4,27])
# Outputs: c2_state(52,[1,2,64])  h2(102,[1,64])  logits(105,[1,2])  h1(72,[1,128])  c1_state(47,[1,4,27])
in_h1  = 0
in_c2  = 1
in_h2  = 2
in_seq = 3
in_c1  = 4

out_c2  = 52   # new c2_state (1,2,64)
out_h2  = 102  # new h2       (1,64)
out_h1  = 72   # new h1       (1,128)
out_c1  = 47   # new c1_state (1,4,27)

# 캘리브레이션 샘플 누적
calib_h1  = []
calib_h2  = []
calib_c1  = []
calib_seq = []
calib_c2  = []

for vid, frames in sequences.items():
    h1 = np.zeros((1, 128), dtype=np.float32)
    h2 = np.zeros((1, 64),  dtype=np.float32)
    c1 = np.zeros((1, 4, 27), dtype=np.float32)
    c2 = np.zeros((1, 2, 64), dtype=np.float32)

    for t in range(len(frames)):
        pose = frames[t:t+1, np.newaxis, :]   # (1,1,27)

        # 현재 입력 기록
        calib_h1.append(h1.copy())
        calib_h2.append(h2.copy())
        calib_c1.append(c1.copy())
        calib_seq.append(pose.copy())
        calib_c2.append(c2.copy())

        # 추론
        interp_f32.set_tensor(in_h1,  h1)
        interp_f32.set_tensor(in_h2,  h2)
        interp_f32.set_tensor(in_c1,  c1)
        interp_f32.set_tensor(in_seq, pose)
        interp_f32.set_tensor(in_c2,  c2)
        interp_f32.invoke()

        h1 = interp_f32.get_tensor(out_h1).copy()
        h2 = interp_f32.get_tensor(out_h2).copy()
        c1 = interp_f32.get_tensor(out_c1).copy()
        c2 = interp_f32.get_tensor(out_c2).copy()

    if len(calib_h1) >= MAX_CALIB_SAMPLES:
        break

calib_h1  = np.concatenate(calib_h1[:MAX_CALIB_SAMPLES])
calib_h2  = np.concatenate(calib_h2[:MAX_CALIB_SAMPLES])
calib_c1  = np.concatenate(calib_c1[:MAX_CALIB_SAMPLES])
calib_seq = np.concatenate(calib_seq[:MAX_CALIB_SAMPLES])
calib_c2  = np.concatenate(calib_c2[:MAX_CALIB_SAMPLES])

print(f"  Collected {len(calib_h1)} calibration samples")
print(f"  h1  range: [{calib_h1.min():.3f}, {calib_h1.max():.3f}]")
print(f"  h2  range: [{calib_h2.min():.3f}, {calib_h2.max():.3f}]")
print(f"  seq range: [{calib_seq.min():.3f}, {calib_seq.max():.3f}]")


# ── int8 TFLite 변환 ─────────────────────────────────────────────────────────
print("\nConverting to int8 TFLite …")

converter = tf.lite.TFLiteConverter.from_saved_model  # placeholder — replaced below
# TFLite → TFLite int8: load tflite content and use experimental_from_model_content
tflite_content = TFLITE.read_bytes()
converter = tf.lite.TFLiteConverter.experimental_from_model_content(tflite_content)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.float32   # 입/출력은 float 유지 (STM32 드라이버 호환)
converter.inference_output_type = tf.float32

def representative_dataset():
    # 입력 순서: h1(0), c2_state(1), h2(2), pose_seq(3), c1_state(4)
    for i in range(len(calib_h1)):
        yield [
            calib_h1[i:i+1],   # idx 0: h1  (1,128)
            calib_c2[i:i+1],   # idx 1: c2_state (1,2,64)
            calib_h2[i:i+1],   # idx 2: h2  (1,64)
            calib_seq[i:i+1],  # idx 3: pose_seq (1,1,27)
            calib_c1[i:i+1],   # idx 4: c1_state (1,4,27)
        ]

converter.representative_dataset = representative_dataset

try:
    tflite_int8 = converter.convert()
    OUT.write_bytes(tflite_int8)
    print(f"  Saved {OUT} ({len(tflite_int8)/1024:.1f} KB)")
except Exception as e:
    print(f"  Full int8 failed: {e}", file=sys.stderr)
    print("  Falling back to dynamic-range quantization …", file=sys.stderr)

    converter2 = tf.lite.TFLiteConverter.from_file(str(TFLITE))
    converter2.optimizations = [tf.lite.Optimize.DEFAULT]
    converter2.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_dyn = converter2.convert()
    out_dyn = OUT.parent / "p27_stateful_dynrange_int8.tflite"
    out_dyn.write_bytes(tflite_dyn)
    print(f"  Saved dynamic-range: {out_dyn} ({len(tflite_dyn)/1024:.1f} KB)")
    sys.exit(1)

print("\nDone.")
