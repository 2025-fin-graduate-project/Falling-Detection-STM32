"""
export_gru_onnx.py
------------------
Exports the FallRNN (GRU) model to ONNX for ST Edge AI.
Compatible with pipeline/rnn_v1_260318.ipynb

Run:
  uv run --with torch --with onnx --with onnxruntime python export_gru_onnx.py
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import os

# ── Config ──────────────────────────────────────────────────────────
PTH_PATH   = "fall_detection_rnn_v1_gru.pth" # Ensure this exists in Model/ or pipeline/
ONNX_PATH  = "fall_detection_gru.onnx"
N_FEATURES = 55
WINDOW     = 60
HIDDEN     = 64
LAYERS     = 1
CLASSES    = 2
MODEL_TYPE = 'GRU'
OPSET      = 13
# ────────────────────────────────────────────────────────────────────

class FallRNN(nn.Module):
    def __init__(self, n_features=55, n_classes=2, hidden_size=64, num_layers=1,
                 dropout=0.2, model_type='GRU', bidirectional=False):
        super().__init__()
        self.model_type    = model_type.upper()
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        directions         = 2 if bidirectional else 1

        rnn_cls = nn.LSTM if self.model_type == 'LSTM' else nn.GRU
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,       # (B, T, F)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * directions, n_classes)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.rnn(x)          # out: (B, T, hidden*directions)
        last   = out[:, -1, :]        # take last timestep: (B, hidden*directions)
        last   = self.dropout(last)
        return self.classifier(last)  # (B, 2)

def main():
    device = torch.device('cpu')

    # 1. Build model
    print("Building GRU model...")
    model = FallRNN(
        n_features=N_FEATURES,
        n_classes=CLASSES,
        hidden_size=HIDDEN,
        num_layers=LAYERS,
        model_type=MODEL_TYPE
    ).to(device)

    # 2. Load weights
    if not os.path.exists(PTH_PATH):
        # Check in pipeline/ if not in current dir
        alt_path = os.path.join("..", "pipeline", PTH_PATH)
        if os.path.exists(alt_path):
            PTH_PATH = alt_path
        else:
            print(f"Error: {PTH_PATH} not found.")
            return

    print(f"Loading weights from {PTH_PATH}...")
    state_dict = torch.load(PTH_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Weights loaded.")

    # 3. Export to ONNX
    print(f"\nExporting to ONNX (opset {OPSET})...")
    dummy_input = torch.randn(1, WINDOW, N_FEATURES) # (1, 60, 55)
    
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        opset_version=OPSET,
        input_names=['input'],
        output_names=['output'],
        # Use static shapes for ST Edge AI
        export_params=True,
        do_constant_folding=True,
    )
    print(f"Saved: {ONNX_PATH}")

    # 4. Verify
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX checker: OK")

    with torch.no_grad():
        pt_out = model(dummy_input).numpy()

    sess    = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    ort_out = sess.run(None, {'input': dummy_input.numpy()})[0]

    max_diff = float(np.abs(pt_out - ort_out).max())
    print(f"\nPyTorch vs ONNX Runtime max diff: {max_diff:.2e}")
    assert max_diff < 1e-4, f"Output mismatch: {max_diff}"
    print("Numerical match: OK")

    print(f"\n✓ Export complete → {ONNX_PATH}")
    print(f"  Input  shape : (1, {WINDOW}, {N_FEATURES}) — (Batch, Time, Features)")
    print(f"  Output shape : (1, {CLASSES}) — logits [normal, fall]")

if __name__ == '__main__':
    main()
