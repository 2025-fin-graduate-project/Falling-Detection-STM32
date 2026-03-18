"""
export_tcn_onnx.py
------------------
fall_detection_model.pth (pytorch-tcn, causal, weight-normed)
→ fall_detection_tcn.onnx  (STEdgeAI-compatible, static shape, batch=1)

Problems solved:
  1. padder.buffer  : causal streaming buffer → zero for batch inference, buffers zeroed
  2. weight_norm    : parametrizations.weight.original0/1 fused into single weight tensor
  3. dynamic_axes   : removed (STEdgeAI needs fully static shapes)
  4. output shape   : (1, 2, T) → take last timestep → (1, 2)  via wrapper

Run:
  uv run --with pytorch-tcn --with onnx --with onnxruntime python export_tcn_onnx.py
"""

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from pytorch_tcn import TCN
import onnx
import onnxruntime as ort
import numpy as np

# ── Config ──────────────────────────────────────────────────────────
PTH_PATH   = "fall_detection_model.pth"
ONNX_PATH  = "fall_detection_tcn.onnx"
N_FEATURES = 55
WINDOW     = 60
CHANNELS   = [64, 64, 128, 2]
KERNEL     = 3
DROPOUT    = 0.2
OPSET      = 13
# ────────────────────────────────────────────────────────────────────


class TCNLastTimestep(nn.Module):
    """Wrapper that returns only the last timestep: (B,2,T) → (B,2)."""
    def __init__(self, tcn: nn.Module):
        super().__init__()
        self.tcn = tcn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.tcn(x)   # (B, 2, T)
        return out[:, :, -1]  # (B, 2)


def zero_padder_buffers(model: nn.Module):
    """Zero all causal-padding buffers so export represents zero-context start."""
    for name, module in model.named_modules():
        if hasattr(module, 'buffer') and isinstance(module.buffer, torch.Tensor):
            print(f"  zeroing padder buffer: {name}  shape={tuple(module.buffer.shape)}")
            module.buffer.zero_()


def fuse_weight_norm(model: nn.Module):
    """Remove weight_norm parametrizations — fuses g*v/||v|| into a plain weight."""
    fused = 0
    for name, module in model.named_modules():
        if parametrize.is_parametrized(module, 'weight'):
            parametrize.remove_parametrizations(module, 'weight', leave_parametrized=True)
            fused += 1
            print(f"  fused weight_norm: {name}")
    print(f"  total fused: {fused}")


def main():
    device = torch.device('cpu')

    # 1. Build model and load weights
    print("Loading model...")
    model = TCN(
        num_inputs=N_FEATURES,
        num_channels=CHANNELS,
        kernel_size=KERNEL,
        dropout=DROPOUT,
        causal=True,
    ).to(device)

    state_dict = torch.load(PTH_PATH, map_location=device)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    print("Weights loaded.")

    # 2. Zero all padder buffers (batch inference: no prior context)
    print("\nZeroing causal padder buffers...")
    zero_padder_buffers(model)

    # 3. Fuse weight normalization → plain weight tensors
    print("\nFusing weight normalization...")
    fuse_weight_norm(model)

    # 4. Wrap to output last timestep
    wrapped = TCNLastTimestep(model).eval()

    # 5. Verify forward pass
    dummy = torch.zeros(1, N_FEATURES, WINDOW)
    with torch.no_grad():
        out = wrapped(dummy)
    print(f"\nForward pass OK: input {tuple(dummy.shape)} → output {tuple(out.shape)}")

    # 6. Export to ONNX (static shapes, no dynamic axes)
    print(f"\nExporting to ONNX (opset {OPSET})...")
    torch.onnx.export(
        wrapped,
        dummy,
        ONNX_PATH,
        opset_version=OPSET,
        input_names=['input'],    # (1, 55, 60)
        output_names=['output'],  # (1, 2)
        # NO dynamic_axes → fully static
        export_params=True,
        do_constant_folding=True,
        dynamo=False,             # force legacy tracing exporter (opset 13)
    )
    print(f"Saved: {ONNX_PATH}")

    # 7. ONNX model check
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX checker: OK")

    # 8. Operator summary
    used_ops = sorted({n.op_type for n in onnx_model.graph.node})
    print(f"\nONNX ops ({len(used_ops)}): {used_ops}")

    # 9. Compare PyTorch vs ONNX Runtime outputs
    test_input = torch.randn(1, N_FEATURES, WINDOW)
    with torch.no_grad():
        pt_out = wrapped(test_input).numpy()

    sess    = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    ort_out = sess.run(None, {'input': test_input.numpy()})[0]

    max_diff = float(np.abs(pt_out - ort_out).max())
    print(f"\nPyTorch vs ONNX Runtime max diff: {max_diff:.2e}")
    assert max_diff < 1e-3, f"Output mismatch: {max_diff}"
    print("Numerical match: OK")

    print(f"\n✓ Export complete → {ONNX_PATH}")
    print(f"  Input  shape : (1, {N_FEATURES}, {WINDOW})")
    print(f"  Output shape : (1, 2)  — logits [normal, fall]")
    print(f"\nNext step: stedgeai analyze --model {ONNX_PATH} --target stm32n6")


if __name__ == '__main__':
    main()
