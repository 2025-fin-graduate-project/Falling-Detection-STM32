/**
 ******************************************************************************
 * @file    pose_pipeline.h
 * @brief   Pose keypoint feature window builder for fall detection.
 *
 *          The fall models are fed from MoveNet postprocessing output:
 *          17 keypoints x (y, x, score) + HSSC_Y/X + RWHC + VHSSC.
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

#define POSE_KP_COUNT       17      /* COCO 17-keypoint MoveNet */
#define POSE_ENG_FEAT_COUNT  4      /* HSSC_Y, HSSC_X, RWHC, VHSSC */
#define POSE_FEATURE_COUNT  (POSE_KP_COUNT * 3 + POSE_ENG_FEAT_COUNT) /* 55 */
#define POSE_WINDOW_SIZE    60      /* timesteps (15 fps x 4 s) */

/* Match scripts/apply_euro_filter_and_recompute_features.py defaults.
 * Only keypoint coordinates are filtered; keypoint confidence stays raw. */
#define POSE_EURO_MIN_CUTOFF  (1.0f)
#define POSE_EURO_BETA        (0.5f)    /* Increased significantly to follow fast fall movements */
#define POSE_EURO_D_CUTOFF    (1.0f)

/* Confidence Smoothing (EMA): Higher alpha = faster response, less smoothing. */
#define POSE_CONF_EMA_ALPHA   (0.5f)    /* Increased from 0.3f for snappier disappearance */    /* Increased from 0.2f for even faster cleanup */

/* ------------------------------------------------------------------ */
/* HSSC upper-body keypoint indices  (Step 4)                           */
/* nose=0, left_eye=1, right_eye=2, left_ear=3, right_ear=4,           */
/* left_shoulder=5, right_shoulder=6                                    */
/* ------------------------------------------------------------------ */
#define POSE_HSSC_INDICES_COUNT  7

/* COCO 17-kp landmark indices used internally */
#define POSE_KP_NOSE          0
#define POSE_KP_LEFT_EYE      1
#define POSE_KP_RIGHT_EYE     2
#define POSE_KP_LEFT_EAR      3
#define POSE_KP_RIGHT_EAR     4
#define POSE_KP_LEFT_SHOULDER 5
#define POSE_KP_RIGHT_SHOULDER 6
#define POSE_KP_LEFT_ELBOW    7
#define POSE_KP_RIGHT_ELBOW   8
#define POSE_KP_LEFT_WRIST    9
#define POSE_KP_RIGHT_WRIST   10
#define POSE_KP_LEFT_HIP      11
#define POSE_KP_RIGHT_HIP     12
#define POSE_KP_LEFT_KNEE     13
#define POSE_KP_RIGHT_KNEE    14
#define POSE_KP_LEFT_ANKLE    15
#define POSE_KP_RIGHT_ANKLE   16

/* ------------------------------------------------------------------ */
/* State structures                                                     */
/* ------------------------------------------------------------------ */

/** Output of one processed frame (55 features) */
typedef struct {
    float32_t f[POSE_FEATURE_COUNT]; /* [y0,x0,s0, y1,x1,s1,..., HSSC_Y,HSSC_X,RWHC,VHSSC] */
    uint8_t   valid;                  /* 0 if frame could not be produced */
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

/** Full pipeline state – allocate once, keep alive across frames */
typedef struct {
    PoseEuroFilter_t y_filter[POSE_KP_COUNT];
    PoseEuroFilter_t x_filter[POSE_KP_COUNT];
    PoseEmaFilter_t  conf_filter[POSE_KP_COUNT]; /* Smoothed confidence */

    /* HSSC derivative history for VHSSC */
    float32_t hssc_y_prev;
    float32_t vhssc_prev;
    uint8_t   deriv_initialized;

    /* Circular sliding window for TCN */
    float32_t win_buf[POSE_WINDOW_SIZE][POSE_FEATURE_COUNT];
    uint32_t  win_head;         /* next write index                    */
    uint32_t  win_count;        /* number of valid frames stored so far */
} PosePipeline_t;

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

/**
 * @brief  Initialise (or reset) a pipeline state object.
 * @param  s  Pointer to pipeline state.
 */
void PosePipeline_Init(PosePipeline_t *s);

/**
 * @brief  Process one frame of raw keypoints through the full pipeline.
 *
 *         Applies One Euro filter to keypoint coordinates, copies raw
 *         confidence scores into the feature layout, computes
 *         HSSC/RWHC/VHSSC, then appends the vector to the window.
 *
 * @param  s         Pipeline state (updated in-place).
 * @param  kp_raw    Array of POSE_KP_COUNT raw keypoints from postprocess.
 * @param  dt        Elapsed time since last frame in seconds.
 * @param  out       Output feature vector for this frame (may be NULL).
 */
void PosePipeline_Process(PosePipeline_t *s,
                          const spe_pp_outBuffer_t *kp_raw,
                          float32_t dt,
                          PoseFeatureVec_t *out);

/**
 * @brief  Copy the sliding window into a flat TCN-ready buffer.
 *
 *         Fills dst[feature][timestep] (features-first, matching TCN input
 *         shape (1, 55, 60)). Only valid when PosePipeline_WindowFull().
 *
 * @param  s    Pipeline state.
 * @param  dst  Destination array [POSE_FEATURE_COUNT][POSE_WINDOW_SIZE].
 */
void PosePipeline_GetWindowFeaturesFirst(const PosePipeline_t *s,
                                         float32_t dst[POSE_FEATURE_COUNT][POSE_WINDOW_SIZE]);

/**
 * @brief  Copy the sliding window into a GRU-ready buffer.
 *
 *         Fills dst[timestep][feature] (time-first, matching GRU input
 *         shape (1, 60, 55)). Only valid when PosePipeline_WindowFull().
 *
 * @param  s    Pipeline state.
 * @param  dst  Destination array [POSE_WINDOW_SIZE][POSE_FEATURE_COUNT].
 */
void PosePipeline_GetWindowTimeFirst(const PosePipeline_t *s,
                                     float32_t dst[POSE_WINDOW_SIZE][POSE_FEATURE_COUNT]);

/**
 * @brief  Returns 1 when POSE_WINDOW_SIZE frames have been accumulated.
 */
uint8_t PosePipeline_WindowFull(const PosePipeline_t *s);

/**
 * @brief  Copy the most recent feature vector into dst (55 floats).
 *         Only valid when win_count >= 1.
 */
void PosePipeline_GetLatestFeature(const PosePipeline_t *s,
                                   float32_t dst[POSE_FEATURE_COUNT]);

#ifdef __cplusplus
}
#endif

#endif /* POSE_PIPELINE_H */
