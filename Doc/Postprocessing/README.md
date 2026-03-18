# Pose Keypoint Preprocessing Pipeline

## Overview

This document describes the preprocessing pipeline applied to raw MoveNet keypoints before they are fed to the TCN fall-detection classifier.
The pipeline is implemented in:

- `Application/STM32N6570-DK/Inc/pose_pipeline.h`
- `Application/STM32N6570-DK/Src/pose_pipeline.c`

The design mirrors `pipeline/preprocessing_pipeline.ipynb` Steps 2–5.

---

## Pipeline Stages

```
MoveNet raw keypoints  (17 × {x, y, score})
         │
         ▼
  Step 2 ─ Outlier Rejection
         │
         ▼
  Step 3 ─ 1-Euro Filter
         │
         ▼
  Step 4 ─ Feature Extraction  (55 features / frame)
         │
         ▼
  Step 5 ─ Sliding Window      (60 frames × 55 = 3 300 values)
         │
         ▼
  TCN Fall-Detection Input  shape (1, 55, 60)
```

---

## Step 2 – Outlier Rejection

### Purpose

Raw MoveNet keypoints occasionally produce large erratic jumps (occlusion, lighting change, fast motion).
These are caught before the smoother to prevent the filter from tracking the noise.

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `HIGH_THRESH` | 0.40 | Confidence above which a keypoint is unconditionally accepted |
| `INITIAL_THRESH` | 0.20 | Minimum confidence for first-seen keypoints (no history yet) |
| `LOW_THRESH` | 0.10 | Below this: always reject, no geometric check |
| `MAX_SPEED_RATIO` | 1.20 | Max allowed single-frame displacement / torso length |
| `CONSISTENCY_RADIUS_RATIO` | 1.80 | Max radius from last valid position / torso length |
| `MAX_HOLD_FRAMES` | 5 | Frames to hold last valid position before marking invalid |

### Torso Length

Used to normalize geometric thresholds so they scale with the person's distance from the camera.

```
torso_length = mean(
    dist(left_shoulder[5], right_hip[12]),   // cross-diagonal 1
    dist(right_shoulder[6], left_hip[11])    // cross-diagonal 2
)
```

Fallback: `0.10` (10% of the normalized image diagonal) when hip/shoulder keypoints are unavailable.

### Decision Logic

```
for each keypoint i:
    if score < LOW_THRESH:
        → reject

    if score >= HIGH_THRESH:
        candidate = ACCEPT
    elif score >= INITIAL_THRESH:
        candidate = ACCEPT  (subject to geometric checks below)
    else:
        → reject

    if history exists:
        d = dist(new, last_valid)
        if d > MAX_SPEED_RATIO × torso_length:   → reject
        if d > CONSISTENCY_RADIUS × torso_length: → reject

    if accepted:
        update last_valid, reset hold_count
        output new position
    else:
        if hold_count < MAX_HOLD_FRAMES:
            output last_valid position (hold)
            hold_count++
        else:
            output (0.5, 0.5) with score 0  (keypoint lost)
```

---

## Step 3 – 1-Euro Filter

### Purpose

Adaptively low-pass filter each keypoint coordinate.
At low speed: strong smoothing (high attenuation).
At high speed: weak smoothing (low lag).

This is preferable to:
- **EMA** – cannot adapt cutoff to motion speed
- **Butterworth** – fixed offline design; no real-time adaptation
- **Kalman** – requires accurate motion model; more parameters to tune

### Algorithm (Casiez et al., CHI 2012)

For a scalar signal `x` at time step `t` with interval `dt`:

```
# Derivative filter (tracks rate of change)
dx_raw    = (x - x̂_prev) / dt
α_d       = dt / (dt + 1/(2π · d_cutoff))
dx̂        = α_d · dx_raw + (1 - α_d) · dx̂_prev

# Adaptive cutoff based on current speed
cutoff    = min_cutoff + β · |dx̂|

# Signal filter
α         = dt / (dt + 1/(2π · cutoff))
x̂         = α · x + (1 - α) · x̂_prev
```

Applied independently to the x and y coordinates of each keypoint.

### Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `min_cutoff` | 0.5 Hz | Base smoothing frequency (lower = smoother at rest) |
| `β` | 1.5 | Speed coefficient (higher = less lag during fast motion) |
| `d_cutoff` | 1.0 Hz | Derivative filter cutoff |

### C Implementation Notes

- `dt` is clamped to `[1e-6, 1.0]` seconds to avoid division by zero and instability.
- On the first frame the filter is seeded with the raw keypoint; `dx̂ = 0`.
- The x and y derivatives use independent cutoffs (`cutoff_x`, `cutoff_y`), matching the per-axis nature of motion.

---

## Step 4 – Feature Extraction

### 55-Feature Vector Layout

```
Index   Content
──────────────────────────────────────────────────────
 0- 2   Keypoint 0  (nose):           y, x, score
 3- 5   Keypoint 1  (left eye):       y, x, score
 6- 8   Keypoint 2  (right eye):      y, x, score
 9-11   Keypoint 3  (left ear):       y, x, score
12-14   Keypoint 4  (right ear):      y, x, score
15-17   Keypoint 5  (left shoulder):  y, x, score
18-20   Keypoint 6  (right shoulder): y, x, score
21-23   Keypoint 7  (left elbow):     y, x, score
24-26   Keypoint 8  (right elbow):    y, x, score
27-29   Keypoint 9  (left wrist):     y, x, score
30-32   Keypoint 10 (right wrist):    y, x, score
33-35   Keypoint 11 (left hip):       y, x, score
36-38   Keypoint 12 (right hip):      y, x, score
39-41   Keypoint 13 (left knee):      y, x, score
42-44   Keypoint 14 (right knee):     y, x, score
45-47   Keypoint 15 (left ankle):     y, x, score
48-50   Keypoint 16 (right ankle):    y, x, score
51      HSSC_X   (upper-body CoM, horizontal)
52      HSSC_Y   (upper-body CoM, vertical)
53      VHSSC    (vertical velocity  = dHSSC_Y / dt)
54      AHSSC    (vertical accel.    = dVHSSC  / dt)
```

All coordinates are in normalised image space `[0, 1]`.
`y` is listed before `x` to match the notebook's convention.

### HSSC (Head-Shoulder-Spine Centre)

Upper-body centre of mass from keypoints **0–6** (nose, eyes, ears, shoulders).
Weighted by confidence score:

```
HSSC_X = Σ(score_i · x_i) / Σ(score_i)   for i ∈ {0,1,2,3,4,5,6}
HSSC_Y = Σ(score_i · y_i) / Σ(score_i)

Fallback: HSSC_X = HSSC_Y = 0.5  (if all scores are zero)
```

### VHSSC and AHSSC

```
VHSSC_t = (HSSC_Y_t - HSSC_Y_{t-1}) / dt    # vertical velocity
AHSSC_t = (VHSSC_t  - VHSSC_{t-1})  / dt    # vertical acceleration
```

`VHSSC` and `AHSSC` are `0.0` on the first frame (not enough history).

---

## Step 5 – Sliding Window

A circular buffer accumulates the last `POSE_WINDOW_SIZE = 60` feature vectors.
At 15 fps this represents **4 seconds** of motion history.

When the buffer is full (`PosePipeline_WindowFull()` returns 1), it can be read as a TCN-ready tensor via `PosePipeline_GetWindowFeaturesFirst()`.

### TCN Input Shape

The TCN model expects input `(batch=1, features=55, timesteps=60)` — features-first (channel-first).
`PosePipeline_GetWindowFeaturesFirst()` performs the in-place transpose from the time-major internal storage.

---

## Memory Usage

| Object | Size |
|--------|------|
| `PosePipeline_t` state | `OutlierKP_t[17]` + `EuroKP_t[17]` + 2 floats + 1 flag + circular buffer |
| Circular buffer | `60 × 55 × 4 B = 13 200 B ≈ 13 KB` |
| Feature vector (per frame) | `55 × 4 B = 220 B` |
| TCN window copy (features-first) | `55 × 60 × 4 B = 13 200 B ≈ 13 KB` |

Total pipeline state fits comfortably in internal SRAM.

---

## Integration in `main.c`

```c
/* Initialisation (once) */
PosePipeline_Init(&pose_pipeline);
pose_last_tick = HAL_GetTick();

/* Per-frame loop (after app_postprocess_run) */
uint32_t now_tick = HAL_GetTick();
float32_t dt_s = (float32_t)(now_tick - pose_last_tick) * 1e-3f;
if (dt_s <= 0.0f || dt_s > 1.0f) dt_s = 1.0f / 15.0f;
pose_last_tick = now_tick;

PoseFeatureVec_t feat_vec;
PosePipeline_Process(&pose_pipeline, pp_output.pOutBuff, dt_s, &feat_vec);

/* When TCN is integrated: */
if (PosePipeline_WindowFull(&pose_pipeline)) {
    float32_t tcn_input[POSE_FEATURE_COUNT][POSE_WINDOW_SIZE];
    PosePipeline_GetWindowFeaturesFirst(&pose_pipeline, tcn_input);
    /* → run TCN inference here */
}
```

---

## References

- **1-Euro Filter**: Casiez, G., Roussel, N., Vogel, D. (CHI 2012). *1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems.*
- **MoveNet COCO 17-keypoint layout**: `pipeline/preprocessing_pipeline.ipynb`, Step 1
- **TCN architecture**: `pipeline/tcn_v2_260318.ipynb`
- **Feature definitions**: `pipeline/preprocessing_pipeline.ipynb`, Step 4
