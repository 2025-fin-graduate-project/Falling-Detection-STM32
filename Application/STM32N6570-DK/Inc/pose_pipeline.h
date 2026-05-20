/**
 ******************************************************************************
 * @file    pose_pipeline.h
 * @brief   Pose keypoint feature window builder for fall detection.
 *
 *          Pipeline D — kp13 subset (nose, all limbs).
 *
 *          Feature layout per frame (45 floats):
 *            [0..38]  kp0, 5-16 × (y, x, conf)     — 39
 *            [39]     HSSC_y  — upper-body centre y          — 1
 *            [40]     HSSC_x  — upper-body centre x          — 1
 *            [41]     RWHC    — bounding-box aspect ratio    — 1
 *            [42]     VHSSC   — EMA-smoothed vertical vel    — 1
 *            [43]     AHSSC   — vertical acceleration        — 1
 *            [44]     AHSSC_x — horizontal acceleration      — 1
 *
 *  Normalization: P37-pure-h64 45-feature training statistics.
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

#define POSE_KP_COUNT         17    /* COCO 17-keypoint MoveNet (all used for HSSC/RWHC) */
#define POSE_KP13_COUNT       13    /* kp0, kp5~kp16 */
#define POSE_ENG_FEAT_COUNT    6    /* HSSC_y, HSSC_x, RWHC, VHSSC, AHSSC, AHSSC_x */
#define POSE_FEATURE_COUNT    45    /* POSE_KP13_COUNT * 3 + POSE_ENG_FEAT_COUNT */
#define POSE_WINDOW_SIZE      40    /* timesteps (15 fps × ~2.7 s) */

/* --- One-Euro Filter (Pipeline D) ----------------------------------- */
#define POSE_EURO_MIN_CUTOFF  (0.5f)
#define POSE_EURO_BETA        (0.3f)
#define POSE_EURO_D_CUTOFF    (1.0f)

#define POSE_CONF_MASK_THRESHOLD  (0.15f)
#define POSE_CONF_EMA_ALPHA       (0.5f)
#define POSE_VHSSC_EMA_ALPHA      (0.4f)

/* ------------------------------------------------------------------ */
/* MinMax normalization — P37-pure-h64 training stats                  */
/* ------------------------------------------------------------------ */
#define POSE_NORM_MIN { \
    /* kp0  (y,x,s) */  0.000000f, 0.000000f, 0.002383f, \
    /* kp5  (y,x,s) */  0.000000f, 0.000000f, 0.004055f, \
    /* kp6  (y,x,s) */  0.000000f, 0.000000f, 0.001079f, \
    /* kp7  (y,x,s) */  0.001145f, 0.000000f, 0.001560f, \
    /* kp8  (y,x,s) */  0.001290f, 0.000724f, 0.004059f, \
    /* kp9  (y,x,s) */  0.005291f, 0.000000f, 0.004168f, \
    /* kp10 (y,x,s) */  0.002378f, 0.000000f, 0.004263f, \
    /* kp11 (y,x,s) */  0.008567f, 0.000359f, 0.008049f, \
    /* kp12 (y,x,s) */  0.002279f, 0.000000f, 0.006607f, \
    /* kp13 (y,x,s) */  0.021298f, 0.000248f, 0.005414f, \
    /* kp14 (y,x,s) */  0.017672f, 0.000000f, 0.004014f, \
    /* kp15 (y,x,s) */  0.021160f, 0.001553f, 0.000135f, \
    /* kp16 (y,x,s) */  0.028301f, 0.000000f, 0.000181f, \
    /* HSSC_y, HSSC_x, RWHC, VHSSC, AHSSC, AHSSC_x */ \
    0.000088f, 0.000158f, 0.024745f, -2.220820f, -18.393295f, -67.749054f }

#define POSE_NORM_SCALE { \
    /* kp0  (y,x,s) */  0.999908f, 0.999997f, 0.880732f, \
    /* kp5  (y,x,s) */  0.997732f, 0.999995f, 0.968784f, \
    /* kp6  (y,x,s) */  0.984889f, 0.999999f, 0.965219f, \
    /* kp7  (y,x,s) */  0.998121f, 0.999957f, 0.981908f, \
    /* kp8  (y,x,s) */  0.989397f, 0.996170f, 0.967367f, \
    /* kp9  (y,x,s) */  0.994656f, 1.000000f, 0.957223f, \
    /* kp10 (y,x,s) */  0.997243f, 0.998899f, 0.948190f, \
    /* kp11 (y,x,s) */  0.991431f, 0.999545f, 0.944041f, \
    /* kp12 (y,x,s) */  0.997647f, 0.999300f, 0.938715f, \
    /* kp13 (y,x,s) */  0.978090f, 0.998411f, 0.965195f, \
    /* kp14 (y,x,s) */  0.981398f, 0.995599f, 0.969497f, \
    /* kp15 (y,x,s) */  0.978840f, 0.996987f, 0.965499f, \
    /* kp16 (y,x,s) */  0.971697f, 0.999525f, 0.968101f, \
    /* HSSC_y, HSSC_x, RWHC, VHSSC, AHSSC, AHSSC_x */ \
    0.991588f, 0.998811f, 21.949295f, 5.239275f, 45.194557f, 125.428223f }

/* ------------------------------------------------------------------ */
/* COCO keypoint indices                                              */
/* ------------------------------------------------------------------ */
#define POSE_KP_NOSE           0
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

typedef struct {
    PoseEuroFilter_t y_filter[POSE_KP_COUNT];
    PoseEuroFilter_t x_filter[POSE_KP_COUNT];
    PoseEmaFilter_t  conf_filter[POSE_KP_COUNT];
    float32_t hssc_y_prev;
    float32_t vhssc_ema;
    float32_t hssc_x_prev;
    float32_t vhssc_x_ema;
    uint8_t   deriv_initialized;
    float32_t win_buf[POSE_WINDOW_SIZE][POSE_FEATURE_COUNT];
    uint32_t  win_head;
    uint32_t  win_count;
} PosePipeline_t;

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

void PosePipeline_Init(PosePipeline_t *s);
void PosePipeline_Process(PosePipeline_t *s, const spe_pp_outBuffer_t *kp_raw, float32_t dt, PoseFeatureVec_t *out);
void PosePipeline_GetWindowTimeFirst(const PosePipeline_t *s, float32_t dst[POSE_WINDOW_SIZE][POSE_FEATURE_COUNT]);
uint8_t PosePipeline_WindowFull(const PosePipeline_t *s);

#ifdef __cplusplus
}
#endif

#endif /* POSE_PIPELINE_H */
