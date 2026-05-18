#!/usr/bin/env python3
"""P27-vm0 stateful int8 TFLite 생성.

4-input 설계: c1_window (5×27)를 C코드에서 미리 빌드해서 입력으로 전달.
모델 내부에는 64-dim concat 1개만 존재 → TFLite int8 quantizer의 27/64 혼동 버그 회피.

모델 입력 (4개):
  h1        (1, 128)   — GRU1 hidden state
  h2        (1,  64)   — GRU2 hidden state
  c1_window (1, 5, 27) — [c1_state(4,27), pose(1,27)] — C에서 concat 후 전달
  c2_in     (1, 2, 64) — c2_state: 최근 2 c1_last 값

모델 출력 (4개):
  new_h1    (1, 128)
  new_h2    (1,  64)
  c1_last   (1,  64)   — C에서 c2_state 갱신에 사용
  logits    (1,   2)

C 코드의 state 갱신:
  c1_state[new] = [c1_state[1:], pose]       — 단순 메모리 shift
  c2_state[new] = [c2_state[1:], c1_last]    — 단순 메모리 shift

Usage:
    python3 scripts/export_p27_stateful_int8.py
"""

import csv
import json
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Lambda

# ── 경로 ────────────────────────────────────────────────────────────────────
REPO      = Path(__file__).resolve().parents[1]
KERAS     = Path("/home/min/Workspace/Graduate-Project/Falling-Model-Development"
                 "/results/phase27_seed_sweep/P27-vm0/model_stedgeai_compat.keras")
OUT_DIR   = REPO / "Model/st_ai_ws"
OUT_F32   = OUT_DIR / "p27_stateful_rebuilt.tflite"
OUT_INT8  = OUT_DIR / "p27_stateful_int8.tflite"
OUT_KERAS = OUT_DIR / "p27_stateful_4in_f32.keras"
VAL_CSV   = Path("/home/min/Workspace/Graduate-Project/Falling-Model-Development"
                 "/dataset/splits_v2_class_balanced_filtered/val.csv")
FEAT_JSON = Path("/home/min/Workspace/Graduate-Project/Falling-Model-Development"
                 "/results/phase27_seed_sweep/P27-vm0/feature_columns.json")
NORM_JSON = Path("/home/min/Workspace/Graduate-Project/Falling-Model-Development"
                 "/results/phase27_seed_sweep/P27-vm0/normalization.json")

MAX_CALIB = 2000

# ── 원래 모델 로드 ───────────────────────────────────────────────────────────
print("Loading window-based keras model …")
orig = tf.keras.models.load_model(str(KERAS), compile=False)

# 레이어 참조
conv1 = orig.get_layer("conv_pre_1")
bn1   = orig.get_layer("conv_pre_bn_1")
relu1 = orig.get_layer("conv_pre_relu_1")
conv2 = orig.get_layer("conv_pre_2")
bn2   = orig.get_layer("conv_pre_bn_2")
relu2 = orig.get_layer("conv_pre_relu_2")
ln1   = orig.get_layer("ln_1")
head  = orig.get_layer("head_dense")
cls   = orig.get_layer("classifier")
gru1_orig = orig.get_layer("gru_1")
gru2_orig = orig.get_layer("gru_2")


# ── 4-input Stateful 모델 재건 ────────────────────────────────────────────────
# 설계 원칙:
#   - c1_window = [c1_state(4,27), pose(1,27)] 를 C가 미리 concat해서 전달
#   - 모델 내부 concat은 [c2_in(2,64), c1_last(1,64)] 1개뿐 (64-dim only)
#   - 27-dim concat이 모델 내부에 없으므로 TFLite quantizer의 27!=64 버그 회피
print("Building 4-input stateful model …")

gru1_step = tf.keras.layers.GRU(
    128, return_sequences=True, return_state=True, unroll=True,
    activation="tanh", recurrent_activation="sigmoid",
    reset_after=True, use_bias=True, name="gru_1_step"
)
gru2_step = tf.keras.layers.GRU(
    64, return_sequences=False, return_state=True, unroll=True,
    activation="tanh", recurrent_activation="sigmoid",
    reset_after=True, use_bias=True, name="gru_2_step"
)

h1_in     = tf.keras.Input(shape=(128,),    batch_size=1, name="h1_in")
h2_in     = tf.keras.Input(shape=(64,),     batch_size=1, name="h2_in")
c1_window = tf.keras.Input(shape=(5, 27),   batch_size=1, name="c1_window")   # [c1_state(4)+pose(1)]
c2_in     = tf.keras.Input(shape=(2, 64),   batch_size=1, name="c2_in")       # c2_state: 2 past c1_last

# conv_pre_1: (B,5,27) → (B,5,64)
c1_feat = relu1(bn1(conv1(c1_window), training=False))
# Use rank-3 slice ([:,-1:,:]) for the concat input — NOT shared with any output tensor.
# If the same tensor is a model output AND a Reshape input, the int8 quantizer
# mis-propagates its rank, causing "3 != 2" CONCATENATION error.
# Keeping concat input (3D) and model output (2D) as separate STRIDED_SLICEs avoids this.
c1_last_3d = c1_feat[:, -1:, :]     # (B,1,64) rank-3 — concat input ONLY
c1_last_2d = c1_last_3d[:, 0, :]   # (B,64)   rank-2 — model output ONLY

# conv_pre_2: concat c2_in + c1_last_3d → (B,3,64), both rank-3 → no rank mismatch
x3      = Concatenate(axis=1, name="c2_concat")([c2_in, c1_last_3d])
c2_feat = relu2(bn2(conv2(x3), training=False))
c2_last_3d = c2_feat[:, -1:, :]     # (B,1,64) rank-3 for GRU input
c2_last_2d = c2_last_3d[:, 0, :]   # (B,64)   unused but avoids output sharing

# GRU1 1-step (c2_last_3d is (B,1,64) rank-3 — valid GRU sequence input)
gru1_seq, new_h1 = gru1_step(c2_last_3d, initial_state=h1_in)
ln_out = ln1(gru1_seq)          # (B,1,128)

# GRU2 1-step
_, new_h2 = gru2_step(ln_out, initial_state=h2_in)

# Head
logits = cls(head(new_h2))      # (B,2)

# c1_last_2d (B,64) 출력: C에서 c2_state 갱신에 사용.
# 별도 STRIDED_SLICE로 파생 — concat 입력(c1_last_3d)과 공유하지 않음 (quantizer 버그 회피)
c1_last_out = c1_last_2d

stateful_model = tf.keras.Model(
    inputs=[h1_in, h2_in, c1_window, c2_in],
    outputs=[new_h1, new_h2, c1_last_out, logits],
    name="p27_stateful_4in"
)

# 가중치 이전
print("Transferring weights …")
gru1_step.set_weights(gru1_orig.get_weights())
gru2_step.set_weights(gru2_orig.get_weights())


# ── 등가성 검증 (float32 window model vs 4-input stateful) ───────────────────
print("Verifying equivalence …")
np.random.seed(42)
T = 40
seq = np.random.rand(1, T, 27).astype(np.float32)

orig_out = orig(seq, training=False).numpy()

h1       = np.zeros((1, 128), dtype=np.float32)
h2       = np.zeros((1, 64),  dtype=np.float32)
c1_state = np.zeros((1, 4, 27), dtype=np.float32)
c2_state = np.zeros((1, 2, 64), dtype=np.float32)

for t in range(T):
    pose = seq[:, t:t+1, :]   # (1,1,27)

    # C에서 수행: c1_window = [c1_state(4), pose(1)]
    c1_win = np.concatenate([c1_state, pose], axis=1)  # (1,5,27)

    new_h1_np, new_h2_np, c1_last_np, logits_np = stateful_model(
        [h1, h2, c1_win, c2_state], training=False
    )

    h1 = new_h1_np.numpy()
    h2 = new_h2_np.numpy()

    # C에서 수행하는 state update
    c1_state = np.concatenate([c1_state[:, 1:, :], pose], axis=1)           # (1,4,27)
    c2_state = np.concatenate([c2_state[:, 1:, :],
                               c1_last_np.numpy()[:, np.newaxis, :]], axis=1)  # (1,2,64)

max_diff = np.max(np.abs(orig_out - logits_np.numpy()))
status   = "OK" if max_diff < 1e-4 else "WARNING: large diff!"
print(f"  orig={orig_out.squeeze()}, stateful={logits_np.numpy().squeeze()}")
print(f"  max_diff={max_diff:.6f}  {status}")

if max_diff > 1e-2:
    print("  FATAL: equivalence check failed — aborting.", file=sys.stderr)
    sys.exit(1)


# ── Float32 TFLite 변환 (캘리브레이션 실행용) ───────────────────────────────
print("\nConverting to float32 TFLite …")
conv_f32 = tf.lite.TFLiteConverter.from_keras_model(stateful_model)
conv_f32.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_f32 = conv_f32.convert()
OUT_F32.write_bytes(tflite_f32)
print(f"  Saved {OUT_F32} ({len(tflite_f32)/1024:.1f} KB)")


# ── 캘리브레이션 데이터 수집 ────────────────────────────────────────────────
print("\nCollecting calibration samples …")

feat_cols  = json.loads(FEAT_JSON.read_text())
norm       = json.loads(NORM_JSON.read_text())
norm_min   = np.array(norm["min"],   dtype=np.float32)
norm_scale = np.array(norm["scale"], dtype=np.float32)

def load_sequences(csv_path):
    videos = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video_id"]
            try:
                frame = np.array([float(row[c]) for c in feat_cols], dtype=np.float32)
            except (KeyError, ValueError):
                continue
            frame = np.clip((frame - norm_min) / norm_scale, 0.0, 1.0)
            videos.setdefault(vid, []).append(frame)
    return {v: np.stack(fs) for v, fs in videos.items() if len(fs) > 2}

sequences = load_sequences(VAL_CSV)
print(f"  {len(sequences)} videos loaded")

# float32 TFLite 인터프리터로 실제 상태 수집 (분포 수집용)
interp = tf.lite.Interpreter(model_content=tflite_f32)
interp.allocate_tensors()
inp_d = {d["name"]: d for d in interp.get_input_details()}
out_d_list = interp.get_output_details()

# 이름으로 입력 인덱스 찾기
def find_inp(sub):
    for name, d in inp_d.items():
        if sub in name:
            return d["index"]
    raise KeyError(f"'{sub}' not found in inputs: {list(inp_d.keys())}")

in_h1  = find_inp("h1_in")
in_h2  = find_inp("h2_in")
in_c1w = find_inp("c1_window")
in_c2  = find_inp("c2_in")

# 출력: shape로 찾기
shape_map = {tuple(d["shape"].tolist()): d["index"] for d in out_d_list}
out_h1  = shape_map[(1, 128)]
out_h2  = shape_map[(1, 64)]
# c1_last (1,64) conflicts with h2 (1,64) if same shape — use name instead
out_d_by_name = {d["name"]: d for d in out_d_list}
# Identify outputs by order: [new_h1(128), new_h2(64), c1_last(64), logits(2)]
# Use shape + position for disambiguation
out_128 = [d["index"] for d in out_d_list if tuple(d["shape"]) == (1, 128)]
out_64  = [d["index"] for d in out_d_list if tuple(d["shape"]) == (1, 64)]
out_2   = [d["index"] for d in out_d_list if tuple(d["shape"]) == (1, 2)]
assert len(out_128) == 1 and len(out_64) == 2 and len(out_2) == 1
out_h1  = out_128[0]

# Disambiguate new_h2 vs c1_last using names
def pick_idx(indices, name_substr):
    for idx in indices:
        for d in out_d_list:
            if d["index"] == idx and name_substr in d["name"]:
                return idx
    return None

out_h2     = pick_idx(out_64, "gru_2_step")
out_c1last = pick_idx(out_64, "c1_last_out")
if out_h2 is None or out_c1last is None:
    # fallback: first is new_h2 (GRU2 output), second is c1_last
    out_h2, out_c1last = out_64[0], out_64[1]
out_logits = out_2[0]

print(f"  Input indices: h1={in_h1}, h2={in_h2}, c1_window={in_c1w}, c2_in={in_c2}")
print(f"  Output indices: h1={out_h1}, h2={out_h2}, c1_last={out_c1last}, logits={out_logits}")

calib_h1  = []
calib_h2  = []
calib_c1w = []
calib_c2  = []

for vid, frames in sequences.items():
    h1 = np.zeros((1, 128), dtype=np.float32)
    h2 = np.zeros((1, 64),  dtype=np.float32)
    c1_state = np.zeros((1, 4, 27), dtype=np.float32)
    c2_state = np.zeros((1, 2, 64), dtype=np.float32)

    for t in range(len(frames)):
        pose = frames[t:t+1, np.newaxis, :]  # (1,1,27)
        c1_win = np.concatenate([c1_state, pose], axis=1)  # (1,5,27)

        calib_h1.append(h1.copy())
        calib_h2.append(h2.copy())
        calib_c1w.append(c1_win.copy())
        calib_c2.append(c2_state.copy())

        interp.set_tensor(in_h1,  h1)
        interp.set_tensor(in_h2,  h2)
        interp.set_tensor(in_c1w, c1_win)
        interp.set_tensor(in_c2,  c2_state)
        interp.invoke()

        h1        = interp.get_tensor(out_h1).copy()
        h2        = interp.get_tensor(out_h2).copy()
        c1_last   = interp.get_tensor(out_c1last).copy()   # (1,64)
        c1_state  = np.concatenate([c1_state[:, 1:, :], pose], axis=1)
        c2_state  = np.concatenate([c2_state[:, 1:, :],
                                    c1_last[:, np.newaxis, :]], axis=1)

    if len(calib_h1) >= MAX_CALIB:
        break

n = min(MAX_CALIB, len(calib_h1))
calib_h1  = np.concatenate(calib_h1[:n])
calib_h2  = np.concatenate(calib_h2[:n])
calib_c1w = np.concatenate(calib_c1w[:n])
calib_c2  = np.concatenate(calib_c2[:n])
print(f"  {n} calibration samples collected")
print(f"  h1  range: [{calib_h1.min():.3f}, {calib_h1.max():.3f}]")
print(f"  h2  range: [{calib_h2.min():.3f}, {calib_h2.max():.3f}]")
print(f"  c1w range: [{calib_c1w.min():.3f}, {calib_c1w.max():.3f}]")
print(f"  c2  range: [{calib_c2.min():.3f}, {calib_c2.max():.3f}]")


# ── Int8 TFLite 변환 ─────────────────────────────────────────────────────────
print("\nConverting to int8 TFLite …")

def representative_dataset():
    # 순서: stateful_model.inputs 순서 — h1, h2, c1_window, c2_in
    for i in range(n):
        yield [
            calib_h1[i:i+1],
            calib_h2[i:i+1],
            calib_c1w[i:i+1],
            calib_c2[i:i+1],
        ]

def try_int8(label, use_mlir=False):
    conv = tf.lite.TFLiteConverter.from_keras_model(stateful_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type  = tf.float32
    conv.inference_output_type = tf.float32
    conv.representative_dataset = representative_dataset
    if use_mlir:
        conv.experimental_new_quantizer = True
    try:
        data = conv.convert()
        print(f"  [{label}] OK — {len(data)/1024:.1f} KB")
        return data
    except Exception as e:
        print(f"  [{label}] Failed: {e}", file=sys.stderr)
        return None

tflite_int8 = try_int8("full-int8 f32-io")
if tflite_int8 is None:
    tflite_int8 = try_int8("full-int8 mlir f32-io", use_mlir=True)

if tflite_int8 is not None:
    OUT_INT8.write_bytes(tflite_int8)
    print(f"  Saved {OUT_INT8} ({len(tflite_int8)/1024:.1f} KB)")
else:
    print("  Full int8 failed — falling back to dynamic-range …", file=sys.stderr)
    conv_dyn = tf.lite.TFLiteConverter.from_keras_model(stateful_model)
    conv_dyn.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dyn = conv_dyn.convert()
    out_dyn = OUT_INT8.parent / "p27_stateful_dynrange.tflite"
    out_dyn.write_bytes(tflite_dyn)
    print(f"  Saved dynamic-range: {out_dyn} ({len(tflite_dyn)/1024:.1f} KB)")
    sys.exit(1)

# ── Keras 저장 (STEdgeAI generate용) ─────────────────────────────────────────
print("\nSaving float32 Keras model for STEdgeAI …")
stateful_model.save(str(OUT_KERAS))
print(f"  Saved {OUT_KERAS}")

print("\nDone. Next steps:")
print(f"  1. stedgeai generate --model {OUT_INT8} --type tflite --target stm32n6 ...")
print(f"     OR: stedgeai generate --model {OUT_KERAS} --type keras --target stm32n6 ...")
print(f"  2. stedgeai validate --model {OUT_INT8} --target stm32n6 --mode host ...")
