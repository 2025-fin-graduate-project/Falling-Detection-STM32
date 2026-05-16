/**
 ******************************************************************************
 * @file    pose_pipeline.h
 * @brief   Pose keypoint feature window builder for fall detection.
 *
 *          Pipeline D — matches the offline preprocessing in
 *          scripts/build_filtered_dataset.py (Python, model dev repo).
 *
 *          Feature layout per frame (57 floats):
 *            [0..50]  kp0..kp16 × (y, x, conf)           — 51
 *            [51]     HSSC_y  — upper-body centre y        — 1
 *            [52]     HSSC_x  — upper-body centre x        — 1
 *            [53]     RWHC    — bounding-box aspect ratio  — 1
 *            [54]     VHSSC   — EMA-smoothed vertical vel  — 1
 *            [55]     AHSSC   — vertical acceleration      — 1  (NEW)
 *            [56]     AHSSC_x — horizontal acceleration    — 1  (NEW)
 *
 *  NOTE: POSE_FEATURE_COUNT changed 55 → 57.
 *        Requires GRU model retrained on the filtered 57-feature dataset.
 ******************************************************************************
 */
#ifndef POSE_PIPELINE_H
#define POSE_PIPELINE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "arm_math.h"
#include "spe_pp_output_if.h"

/* ------------------------------------------------------------------ */
/* Compile-time constants                                               */
/* ------------------------------------------------------------------ */

#define POSE_KP_COUNT        17     /* COCO 17-keypoint MoveNet (all used for HSSC/RWHC) */
#define POSE_KP7_COUNT        7     /* kp7 subset stored in feature vector */
#define POSE_ENG_FEAT_COUNT   6     /* HSSC_Y, HSSC_X, RWHC, VHSSC, AHSSC, AHSSC_x */
#define POSE_FEATURE_COUNT   (POSE_KP7_COUNT * 3 + POSE_ENG_FEAT_COUNT)  /* 27 */
#define POSE_WINDOW_SIZE     40     /* timesteps (15 fps × ~2.7 s) */

/* --- One-Euro Filter (Pipeline D) -----------------------------------
 * Tuned for training-data quality; softer than STM32 v1 (1.0 / 0.5).
 * min_cutoff=0.5: α=0.136 at rest → holds 86% of previous (Frozen KP
 *   artefacts are naturally suppressed).
 * beta=0.3: at fall speed (~3 /s) cutoff rises to 1.4 Hz — enough to
 *   follow the descent without amplifying noise. */
#define POSE_EURO_MIN_CUTOFF  (0.5f)
#define POSE_EURO_BETA        (0.3f)
#define POSE_EURO_D_CUTOFF    (1.0f)

/* --- Low-confidence masking -----------------------------------------
 * Keypoints with smoothed confidence below this threshold are treated
 * as missing: the One-Euro filter is NOT updated and the previous
 * filtered coordinate is held.  This mirrors the offline mask→interpolate
 * step in Pipeline D (conf_thr=0.15). */
#define POSE_CONF_MASK_THRESHOLD  (0.15f)

/* --- Confidence EMA -------------------------------------------------
 * Matches Python Pipeline D exactly (alpha=0.5).
 * Previous asymmetric version (0.8/0.5) removed for better alignment. */
#define POSE_CONF_EMA_ALPHA       (0.5f)

/* --- VHSSC EMA (Pipeline D) ----------------------------------------
 * Applied to vertical velocity (VHSSC) before deriving AHSSC.
 * Matches Python Pipeline D (ema_deriv_alpha=0.4). */
#define POSE_VHSSC_EMA_ALPHA      (0.4f)

/* ------------------------------------------------------------------ */
/* MinMax normalization — P27-vm0 (kp7 + filtered, 27 features)        */
/* Applied after computing raw features, before storing to window buf. */
/* feat_norm = (feat_raw - min) / scale                                */
/* ------------------------------------------------------------------ */
#define POSE_NORM_MIN { \
    0.000000f, 0.000000f, 0.002383f, \
    0.000002f, 0.000000f, 0.004055f, \
    0.000000f, 0.000000f, 0.002157f, \
    0.001145f, 0.000001f, 0.001560f, \
    0.001290f, 0.002396f, 0.004167f, \
    0.008567f, 0.002751f, 0.008049f, \
    0.002279f, 0.000000f, 0.006607f, \
    0.000137f, 0.000158f, 0.024745f, \
    -2.220820f, -18.393295f, -67.749054f }

#define POSE_NORM_SCALE { \
    0.999908f, 0.999997f, 0.870511f, \
    0.997730f, 0.999995f, 0.968784f, \
    0.983443f, 0.999999f, 0.962983f, \
    0.997804f, 0.999956f, 0.972651f, \
    0.986192f, 0.994499f, 0.963977f, \
    0.991426f, 0.997080f, 0.941503f, \
    0.997647f, 0.999300f, 0.938715f, \
    0.991539f, 0.998045f, 21.949295f, \
    5.239275f, 45.194557f, 125.179825f }

/* ------------------------------------------------------------------ */
/* HSSC upper-body keypoint indices                                     */
/* nose=0, l_eye=1, r_eye=2, l_ear=3, r_ear=4, l_sho=5, r_sho=6      */
/* (Same as original 17-kp pipeline — HSSC computed from kp0-kp6)     */
/* ------------------------------------------------------------------ */
#define POSE_HSSC_INDICES_COUNT  7

/* COCO keypoint indices */
#define POSE_KP_NOSE           0
#define POSE_KP_LEFT_EYE       1
#define POSE_KP_RIGHT_EYE      2
#define POSE_KP_LEFT_EAR       3
#define POSE_KP_RIGHT_EAR      4
#define POSE_KP_LEFT_SHOULDER  5
#define POSE_KP_RIGHT_SHOULDER 6
#define POSE_KP_LEFT_ELBOW     7
#define POSE_KP_RIGHT_ELBOW    8
#define POSE_KP_LEFT_WRIST     9
#define POSE_KP_RIGHT_WRIST    10
#define POSE_KP_LEFT_HIP       11
#define POSE_KP_RIGHT_HIP      12
#define POSE_KP_LEFT_KNEE      13
#define POSE_KP_RIGHT_KNEE     14
#define POSE_KP_LEFT_ANKLE     15
#define POSE_KP_RIGHT_ANKLE    16

/* ------------------------------------------------------------------ */
/* State structures                                                     */
/* ------------------------------------------------------------------ */

/** Output of one processed frame (57 features) */
typedef struct {
    float32_t f[POSE_FEATURE_COUNT];
    uint8_t   valid;
} PoseFeatureVec_t;

typedef struct {
    float32_t x_prev;
    float32_t dx_prev;
    uint8_t   initialized;
} PoseEuroFilter_t;

typedef struct {
    float32_t x_prev;
    uint8_t   initialized;
} PoseEmaFilter_t;

/** Full pipeline state — allocate once, keep alive across frames. */
typedef struct {
    PoseEuroFilter_t y_filter[POSE_KP_COUNT];
    PoseEuroFilter_t x_filter[POSE_KP_COUNT];
    PoseEmaFilter_t  conf_filter[POSE_KP_COUNT];

    /* Vertical kinematics */
    float32_t hssc_y_prev;    /* previous HSSC_y for VHSSC */
    float32_t vhssc_ema;      /* EMA-smoothed VHSSC (used for AHSSC and output) */

    /* Horizontal kinematics (AHSSC_x) */
    float32_t hssc_x_prev;    /* previous HSSC_x */
    float32_t vhssc_x_ema;    /* EMA-smoothed horizontal velocity */

    uint8_t   deriv_initialized;

    /* Circular sliding window for GRU/TCN */
    float32_t win_buf[POSE_WINDOW_SIZE][POSE_FEATURE_COUNT];
    uint32_t  win_head;
    uint32_t  win_count;
} PosePipeline_t;

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

void PosePipeline_Init(PosePipeline_t *s);

/**
 * @brief  Process one frame through Pipeline D.
 *
 *  1. Clamp raw coordinates to [0, 1].
 *  2. If KP confidence < POSE_CONF_MASK_THRESHOLD: hold last filtered value
 *     (real-time equivalent of offline mask→interpolate).
 *  3. Apply One-Euro filter (min_cutoff=0.5, beta=0.3) to coordinates.
 *  4. Apply asymmetric EMA to confidence.
 *  5. Compute HSSC_y/x, RWHC, VHSSC.
 *  6. Apply EMA (alpha=0.4) to VHSSC → stored as VHSSC feature.
 *  7. Derive AHSSC = d(VHSSC_ema)/dt and AHSSC_x = d(VHSSC_x_ema)/dt.
 *  8. Append 57-float vector to circular window.
 */
void PosePipeline_Process(PosePipeline_t *s,
                          const spe_pp_outBuffer_t *kp_raw,
                          float32_t dt,
                          PoseFeatureVec_t *out);

void PosePipeline_GetWindowFeaturesFirst(const PosePipeline_t *s,
                                         float32_t dst[POSE_FEATURE_COUNT][POSE_WINDOW_SIZE]);

void PosePipeline_GetWindowTimeFirst(const PosePipeline_t *s,
                                     float32_t dst[POSE_WINDOW_SIZE][POSE_FEATURE_COUNT]);

uint8_t PosePipeline_WindowFull(const PosePipeline_t *s);

void PosePipeline_GetLatestFeature(const PosePipeline_t *s,
                                   float32_t dst[POSE_FEATURE_COUNT]);

#ifdef __cplusplus
}
#endif

#endif /* POSE_PIPELINE_H */
