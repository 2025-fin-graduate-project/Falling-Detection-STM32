#!/usr/bin/env python3
"""P27-vm0 INT8 양자화 — STEdgeAI ONNX + ORT static quantization.

워크플로:
  1. float32 4-input 재구성 TFLite → STEdgeAI export-onnx → p27_stateful_stedgeai.onnx
  2. 이 스크립트: ORT quantize_static (QDQ, INT8) → p27_stateful_stedgeai_int8.onnx
  3. stedgeai generate (generate-gru-model_STM32N6570-DK.sh) → C 코드

STEdgeAI 자체 ONNX를 사용하는 이유:
  tf2onnx가 생성하는 ONNX는 Conv1D를 Reshape+Conv2D+Reshape 패턴으로 변환하므로
  STEdgeAI ONNX 파서가 "Unknown dimensions: W" 오류로 거부한다.
  STEdgeAI export-onnx는 STEdgeAI 파서가 지원하는 opset으로 ONNX를 생성한다.

사전 준비 (한 번만, st_ai_ws/ 디렉토리에서):
  stedgeai export-onnx \\
      --model p27_stateful_rebuilt.tflite \\
      --type tflite --target stm32n6 \\
      --name gru_network --output st_ai_output_onnx_f32

Usage:
    cd /path/to/Falling-Detection-STM32
    python3 scripts/quantize_p27_stedgeai_int8.py

Output:
    Model/st_ai_ws/p27_stateful_stedgeai_int8.onnx
"""

import csv
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

# ── 경로 ────────────────────────────────────────────────────────────────────
REPO     = Path(__file__).resolve().parents[1]
WS       = REPO / "Model/st_ai_ws"
ONNX_F32 = WS / "p27_stateful_stedgeai.onnx"
ONNX_INT8 = WS / "p27_stateful_stedgeai_int8.onnx"
TFLITE_F32 = WS / "p27_stateful_rebuilt.tflite"

VAL_CSV   = Path("/home/min/Workspace/Graduate-Project/Falling-Model-Development"
                 "/dataset/splits_v2_class_balanced_filtered/val.csv")
FEAT_JSON = Path("/home/min/Workspace/Graduate-Project/Falling-Model-Development"
                 "/results/phase27_seed_sweep/P27-vm0/feature_columns.json")
NORM_JSON = Path("/home/min/Workspace/Graduate-Project/Falling-Model-Development"
                 "/results/phase27_seed_sweep/P27-vm0/normalization.json")

MAX_CALIB = 2000

# ── 정규화 로드 ──────────────────────────────────────────────────────────────
feat_cols  = json.loads(FEAT_JSON.read_text())
norm       = json.loads(NORM_JSON.read_text())
norm_min   = np.array(norm["min"],   dtype=np.float32)
norm_scale = np.array(norm["scale"], dtype=np.float32)
assert len(feat_cols) == 27


def load_sequences(csv_path: Path) -> dict[str, np.ndarray]:
    videos: dict[str, list] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            vid = row["video_id"]
            try:
                frame = np.array([float(row[c]) for c in feat_cols], np.float32)
            except (KeyError, ValueError):
                continue
            frame = np.clip((frame - norm_min) / norm_scale, 0.0, 1.0)
            videos.setdefault(vid, []).append(frame)
    return {v: np.stack(fs) for v, fs in videos.items() if len(fs) > 2}


def collect_calibration(onnx_path: Path, sequences: dict) -> list[dict]:
    """float32 ORT 세션으로 실제 상태 시퀀스를 실행해 캘리브레이션 샘플 수집."""
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_names = [i.name for i in sess.get_inputs()]
    print(f"  Model inputs: {inp_names}")

    # STEdgeAI ONNX 입력 이름 매핑
    # 이름이 세션 실행마다 다를 수 있으므로 shape로 매핑
    inp_shapes = {i.name: i.shape for i in sess.get_inputs()}
    name_c1w = next(n for n, s in inp_shapes.items() if s == [1, 5, 27])
    name_c2  = next(n for n, s in inp_shapes.items() if s == [1, 2, 64])
    name_h1  = next(n for n, s in inp_shapes.items() if s == [1, 128])
    name_h2  = next(n for n, s in inp_shapes.items() if s == [1, 64])
    print(f"  c1_window={name_c1w}, c2={name_c2}, h1={name_h1}, h2={name_h2}")

    out_names = [o.name for o in sess.get_outputs()]
    out_shapes = {o.name: o.shape for o in sess.get_outputs()}
    name_h2_new = next(n for n, s in out_shapes.items() if s == [1, 64]
                       and "Add_82" in n or "h2" in n.lower())
    name_h1_new = next(n for n, s in out_shapes.items() if s == [1, 128])
    name_c1last = [n for n, s in out_shapes.items() if s in ([1, 64], [1, 64, 1])
                   and n != name_h2_new][0]
    print(f"  Outputs: h2_new={name_h2_new}, h1_new={name_h1_new}, c1_last={name_c1last}")

    samples = []
    for vid, frames in sequences.items():
        h1 = np.zeros((1, 128), np.float32)
        h2 = np.zeros((1,  64), np.float32)
        c1_state = np.zeros((1, 4, 27), np.float32)
        c2_state = np.zeros((1, 2, 64), np.float32)

        for t in range(len(frames)):
            pose  = frames[t:t+1, np.newaxis, :]           # (1,1,27)
            c1_win = np.concatenate([c1_state, pose], 1)   # (1,5,27)

            samples.append({
                name_c1w: c1_win.copy(),
                name_c2:  c2_state.copy(),
                name_h1:  h1.copy(),
                name_h2:  h2.copy(),
            })

            outs = sess.run(out_names, {
                name_c1w: c1_win, name_c2: c2_state,
                name_h1: h1, name_h2: h2,
            })
            out_map = dict(zip(out_names, outs))

            h2       = out_map[name_h2_new]
            h1       = out_map[name_h1_new]
            c1_last  = out_map[name_c1last].reshape(1, 64)
            c1_state = np.concatenate([c1_state[:, 1:], pose], 1)
            c2_state = np.concatenate([c2_state[:, 1:], c1_last[:, np.newaxis]], 1)

        if len(samples) >= MAX_CALIB:
            break

    return samples[:MAX_CALIB]


class GRUCalibReader(CalibrationDataReader):
    def __init__(self, samples: list[dict]):
        self._samples = samples
        self._idx = 0

    def get_next(self):
        if self._idx >= len(self._samples):
            return None
        s = self._samples[self._idx]
        self._idx += 1
        return s


# ── 메인 ─────────────────────────────────────────────────────────────────────
if not ONNX_F32.exists():
    raise FileNotFoundError(
        f"{ONNX_F32} not found.\n"
        "Run from Model/st_ai_ws/:\n"
        "  stedgeai export-onnx --model p27_stateful_rebuilt.tflite "
        "--type tflite --target stm32n6 --name gru_network --output st_ai_output_onnx_f32\n"
        "  cp st_ai_output_onnx_f32/gru_network.onnx p27_stateful_stedgeai.onnx"
    )

print("Loading validation sequences …")
sequences = load_sequences(VAL_CSV)
print(f"  {len(sequences)} videos, {sum(len(v) for v in sequences.values())} frames total")

print("\nCollecting calibration samples …")
calib_data = collect_calibration(ONNX_F32, sequences)
print(f"  {len(calib_data)} samples collected")

print("\nQuantizing to INT8 (QDQ) …")
quantize_static(
    str(ONNX_F32),
    str(ONNX_INT8),
    GRUCalibReader(calib_data),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=False,
    reduce_range=False,
)
size_kb = ONNX_INT8.stat().st_size / 1024
print(f"  Saved {ONNX_INT8} ({size_kb:.1f} KB)")

print("\nDone. Next step:")
print(f"  cd Model/st_ai_ws && bash ../../Model/generate-gru-model_STM32N6570-DK.sh")
