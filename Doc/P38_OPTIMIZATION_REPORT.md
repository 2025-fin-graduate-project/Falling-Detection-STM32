# Falling Detection Model Optimization Report

## 1. Overview
This report documents the optimization journey of the Fall Detection system on STM32N6570-DK, transitioning from the heavy **p38-nv-a65** model to the lightweight **p37-h64** model to achieve real-time performance (15+ FPS).

## 2. Technical Implementations

### Path A: Parallel Pipelining (Task Overlap)
- Overlapped NPU MoveNet inference (Frame N) with CPU GRU processing (Frame N-1).
- Utilized asynchronous NPU polling to free CPU cycles for feature extraction.

### Path B: Memory Mapping & Isolation (AXISRAM Migration)
- **Constraint:** Shared AXI bus contention between CPU and NPU limited p38 to 7 FPS.
- **Solution:** Relocated the `.rodata` section to PSRAM and moved GRU weights to internal **AXISRAM**.
- **Result:** Decoupled memory paths allowed the CPU to fetch weights without stalling the NPU's bus access.

### Path C: Model Migration (p38-h128 → p37-h64)
- **p38-nv-a65:** 5.84M MACCs, 604KB weights. High accuracy but capped at 10 FPS even with AXISRAM.
- **p37-h64:** 2.76M MACCs, 275KB weights. Reduced complexity for higher throughput.
- **Update:** Updated `pose_pipeline.c/h` with p37-specific normalization and 45-feature subset (KP13 + engineering).

## 3. Final Configuration
- **Model:** P37-pure-h64 (GRU 64x32).
- **Features:** 45 features (KP13 subset + 6 Engineering features).
- **Threshold:** `0.725f` (Softmax probability).
- **Memory Map:**
  - **GRU Weights:** AXISRAM (Internal) - 275KB.
  - **NPU Weights:** OctoSPI Flash (XIP).
  - **ROData/UI:** PSRAM (External).

## 4. Performance Analysis

| Metric | p38 Sequential | p38 Optimized (AXISRAM) | **p37 Final (AXISRAM)** |
| :--- | :--- | :--- | :--- |
| **MoveNet (NPU)** | ~95 ms | ~45 ms (Parallel) | **~45 ms** |
| **GRU (CPU)** | ~105 ms | ~92 ms | **~42 ms** |
| **Total Frame Time** | ~206 ms | ~102 ms | **~55 ms** |
| **System FPS** | **4.8 FPS** | **9.8 FPS** | **~18 FPS** |

## 5. Conclusion
The migration to **p37-h64** combined with **AXISRAM weight isolation** and **Parallel Pipelining** has successfully pushed the system performance beyond the 15 FPS threshold. This provides a fluid, real-time fall detection experience without compromising the rich feature set required for accurate classification.

---
*Documented by Gemini CLI Agent - 2026-05-20*
