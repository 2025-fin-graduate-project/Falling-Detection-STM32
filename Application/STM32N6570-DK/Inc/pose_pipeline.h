/**
 ******************************************************************************
 * @file    pose_pipeline.h
 * @brief   Pose keypoint post-processing pipeline for fall detection.
 *
 *          Pipeline stages (matching preprocessing_pipeline.ipynb):
 *          Step 2 - Outlier rejection  (torso-length-based speed + consistency)
 *          Step 3 - 1-Euro filter      (adaptive low-pass smoothing)
 *          Step 4 - Feature extraction (HSSC_X/Y, VHSSC, AHSSC)
 *          Step 5 - Sliding window     (60-frame buffer for TCN input)
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

#define POSE_KP_COUNT       17      /* COCO 17-keypoint MoveNet     */
#define POSE_ENG_FEAT_COUNT  4      /* HSSC_X, HSSC_Y, VHSSC, AHSSC */
#define POSE_FEATURE_COUNT  (POSE_KP_COUNT * 3 + POSE_ENG_FEAT_COUNT) /* 55 */
#define POSE_WINDOW_SIZE    60      /* timesteps for TCN (15 fps x 4 s) */

/* ------------------------------------------------------------------ */
/* Person presence gate  (Step 1)                                       */
/* ------------------------------------------------------------------ */
#define POSE_MIN_VISIBLE_KP         5   /* 노트북 centroid consistency 최소 후보 수 */
#define POSE_NO_PERSON_RESET_FRAMES 15  /* 연속 미감지 프레임 수 (15fps → 1초) */

/* ------------------------------------------------------------------ */
/* Outlier rejection parameters  (Step 2)                               */
/* ------------------------------------------------------------------ */
#define POSE_OUTLIER_HIGH_THRESH        0.40f
#define POSE_OUTLIER_INITIAL_THRESH     0.20f
#define POSE_OUTLIER_LOW_THRESH         0.10f
#define POSE_OUTLIER_MAX_SPEED_RATIO    1.20f
#define POSE_OUTLIER_CONSISTENCY_RATIO  1.80f
#define POSE_OUTLIER_MAX_HOLD_FRAMES    5

/* ------------------------------------------------------------------ */
/* 1-Euro filter parameters  (Step 3)                                   */
/* ------------------------------------------------------------------ */
#define POSE_EURO_MIN_CUTOFF    0.5f
#define POSE_EURO_BETA          1.5f
#define POSE_EURO_D_CUTOFF      1.0f

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
#define POSE_KP_LEFT_HIP      11
#define POSE_KP_RIGHT_HIP     12

/* ------------------------------------------------------------------ */
/* State structures                                                     */
/* ------------------------------------------------------------------ */

/** Per-keypoint 1-Euro filter state */
typedef struct {
    float32_t x_hat;        /* filtered x position         */
    float32_t y_hat;        /* filtered y position         */
    float32_t dx_hat;       /* filtered x derivative       */
    float32_t dy_hat;       /* filtered y derivative       */
    uint8_t   initialized;
} EuroKP_t;

/** Per-keypoint outlier rejection state */
typedef struct {
    float32_t last_x;       /* last accepted x             */
    float32_t last_y;       /* last accepted y             */
    float32_t last_proba;
    uint8_t   last_valid;   /* 1 = last_x/y are usable     */
    uint8_t   hold_count;   /* frames held since rejection */
} OutlierKP_t;

/** Output of one processed frame (55 features) */
typedef struct {
    float32_t f[POSE_FEATURE_COUNT]; /* [y0,x0,s0, y1,x1,s1,..., HSSC_X,HSSC_Y,VHSSC,AHSSC] */
    uint8_t   valid;                  /* 0 if frame could not be produced */
} PoseFeatureVec_t;

/** Full pipeline state – allocate once, keep alive across frames */
typedef struct {
    OutlierKP_t outlier[POSE_KP_COUNT];
    EuroKP_t    euro[POSE_KP_COUNT];

    /* HSSC derivative history for VHSSC / AHSSC */
    float32_t hssc_y_prev;
    float32_t vhssc_prev;
    float32_t torso_length;
    uint8_t   deriv_initialized;

    /* Circular sliding window for TCN */
    float32_t win_buf[POSE_WINDOW_SIZE][POSE_FEATURE_COUNT];
    uint32_t  win_head;         /* next write index                    */
    uint32_t  win_count;        /* number of valid frames stored so far */

    /* Person presence gate */
    uint32_t  no_person_count;  /* 연속으로 사람 미감지된 프레임 수 */
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
 *         Runs outlier rejection → 1-Euro filter → feature extraction,
 *         then appends the 55-feature vector to the sliding window.
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
 *         shape (1, 55, 60)).  Only valid when PosePipeline_WindowFull().
 *
 * @param  s    Pipeline state.
 * @param  dst  Destination array [POSE_FEATURE_COUNT][POSE_WINDOW_SIZE].
 */
void PosePipeline_GetWindowFeaturesFirst(const PosePipeline_t *s,
                                         float32_t dst[POSE_FEATURE_COUNT][POSE_WINDOW_SIZE]);

/**
 * @brief  Returns 1 when POSE_WINDOW_SIZE frames have been accumulated.
 */
uint8_t PosePipeline_WindowFull(const PosePipeline_t *s);

#ifdef __cplusplus
}
#endif

#endif /* POSE_PIPELINE_H */
