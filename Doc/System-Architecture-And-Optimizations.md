# System Architecture & Optimization Report

This document details the final, stable architecture for the real-time fall detection system on the STM32N6570-DK, along with the key optimizations and bug fixes implemented to achieve robust performance.

## 1. Final Architecture

The system employs a dual-model pipeline to balance performance and accuracy:

-   **MoveNet Single-Pose (NPU):** A lightweight pose estimation model (`st_movenet_lightning_a100_heatmaps_256_int8`) runs on the NPU for fast keypoint extraction from the 30fps camera feed.
-   **GRU Fall Detector (CPU):** A stateful GRU model (`p21_v01_stateful_int8`) runs exclusively on the Cortex-M55 CPU. It analyzes the time-series data of keypoint kinematics at 15fps to detect falls.

This hybrid approach avoids resource contention and allows both models to operate in parallel, maximizing throughput.

## 2. Critical Bug Fixes & Optimizations

The path to a stable system involved solving several critical, intertwined issues.

### Problem 1: System Freezes & HardFaults
-   **Root Cause:**
    1.  **NPU Resource Contention:** Initially, both MoveNet and the GRU model were configured to use the NPU, causing resource conflicts that froze the MoveNet visualization.
    2.  **Internal RAM Exhaustion:** The GRU model's large activation buffer (~7KB) and the LCD display buffers were placed in Internal RAM, leading to stack overflows and `HardFault` crashes.
-   **Solution:**
    1.  **CPU/NPU Isolation:** The GRU model was re-compiled using the **legacy Cube.AI C-API** (`--c-api legacy`), forcing it to run exclusively on the CPU.
    2.  **Memory Relocation:** All large, non-critical buffers were moved to PSRAM using the `__attribute__((section(".psram_bss")))` directive. The GRU model's weights were placed in external flash via `__attribute__((section(".weights_section")))`. This reduced Internal RAM usage from >95% to a stable **~42%**.

### Problem 2: Poor Detection Performance (Stuck Scores)
-   **Root Cause:**
    1.  **Incorrect Data Pipeline:** The MoveNet post-processor's output had its X and Y coordinates swapped. This fed the GRU incorrect spatial data, preventing it from detecting vertical motion.
    2.  **Broken Temporal Context:** The stateful GRU model's recurrent state buffers (`c1_state`, `c2_state`) were not being updated between frames. The model had no "memory" and was only seeing one frame at a time.
    3.  **Frequent State Resets:** Minor keypoint occlusions caused the GRU's state to be invalidated and reset, preventing the temporal analysis required to detect a fall.
-   **Solution:**
    1.  **Coordinate Correction:** The `pose_pipeline.c` logic was corrected to use `y = kp_raw[i].y_center` and `x = kp_raw[i].x_center`.
    2.  **State Feedback Loop:** The `memmove` and `memcpy` logic was restored in `main.c` to correctly shift the pose history and feature buffers, giving the GRU a 40-frame memory.
    3.  **State Persistence:** The reset counter (`FALL_PERSON_MISSING_RESET_COUNT`) was increased to 60 frames (2 seconds), making the GRU's memory robust to temporary keypoint loss.

### Problem 3: Slow Performance & Unresponsive Warmup
-   **Root Cause:**
    1.  **UART Bottleneck:** Excessive `printf` logging in the main loop caused UART stalls, artificially locking the main loop to a fixed 10fps (100ms per frame).
    2.  **Inefficient Post-Processing:** The MoveNet heatmap decoding was performed twice (for both Channel-First and Channel-Last layouts), wasting CPU cycles.
-   **Solution:**
    1.  **Silent Operation:** All per-frame `printf` calls were removed from the main loop, eliminating the bottleneck and restoring the target hardware frame rate.
    2.  **Post-Processing Optimization:** The post-processing function was modified to only perform the required Channel-Last decoding, halving its execution time.
    3.  **Hybrid Frame Rate:** A 15fps stride (`% 2 == 0`) was re-introduced for the GRU model to match its training environment, while the MoveNet visualization continues to run at the maximum possible frame rate (~30fps) for a smooth UI.

## 3. Final Configuration
-   **Threshold:** `GRU_FALL_SCORE_THRESHOLD` set to `0.31f` for high sensitivity.
-   **Filter:** `POSE_EURO_BETA` set to `1.5f` to ensure fall dynamics are not overly smoothed.
-   **Confidence Mask:** `POSE_CONF_MASK_THRESHOLD` lowered to `0.05f` to allow low-confidence keypoints to contribute to the velocity calculation.
