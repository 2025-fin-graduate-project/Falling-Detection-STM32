/**
 ******************************************************************************
 * @file    pose_pipeline.c
 * @brief   Pose keypoint post-processing pipeline.
 *
 *          Ported from pipeline/preprocessing_pipeline.ipynb Steps 2-4.
 *
 *          Step 2  Outlier rejection
 *                  - confidence gating  (HIGH / INITIAL / LOW thresholds)
 *                  - torso-length-normalised speed check
 *                  - consistency-radius check
 *                  - hold last valid position for up to MAX_HOLD_FRAMES
 *
 *          Step 3  1-Euro filter  (Casiez et al., CHI 2012)
 *                  - adaptive low-pass smoothing, selected over EMA / Kalman
 *                    / Butterworth for its good speed-responsiveness trade-off
 *
 *          Step 4  Feature extraction
 *                  - HSSC_X, HSSC_Y  (upper-body centre of mass, kp 0-6)
 *                  - VHSSC           (vertical velocity  = dHSSC_Y / dt)
 *                  - AHSSC           (vertical accel.   = dVHSSC  / dt)
 *
 *          Step 5  Circular sliding window (POSE_WINDOW_SIZE frames × 55 feat)
 ******************************************************************************
 */
#include "pose_pipeline.h"
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Internal helpers                                                     */
/* ------------------------------------------------------------------ */

/** 2-D Euclidean distance (normalised coordinates). */
static inline float32_t kp_dist(float32_t x0, float32_t y0,
                                 float32_t x1, float32_t y1)
{
    float32_t dx = x1 - x0;
    float32_t dy = y1 - y0;
    return sqrtf(dx * dx + dy * dy);
}

/**
 * Compute torso length from shoulder and hip keypoints.
 * Uses average of two cross-diagonal distances (left_shoulder↔right_hip,
 * right_shoulder↔left_hip) for robustness.
 * Falls back to a fixed fraction of the image diagonal if keypoints are
 * unavailable.
 */
static float32_t compute_torso_length(const spe_pp_outBuffer_t *kp)
{
    float32_t len = 0.0f;
    uint32_t n = 0;

    /* cross-diagonal 1: left_shoulder (5) → right_hip (12) */
    if (kp[POSE_KP_LEFT_SHOULDER].proba  >= POSE_OUTLIER_LOW_THRESH &&
        kp[POSE_KP_RIGHT_HIP].proba      >= POSE_OUTLIER_LOW_THRESH)
    {
        len += kp_dist(kp[POSE_KP_LEFT_SHOULDER].x_center,
                       kp[POSE_KP_LEFT_SHOULDER].y_center,
                       kp[POSE_KP_RIGHT_HIP].x_center,
                       kp[POSE_KP_RIGHT_HIP].y_center);
        n++;
    }
    /* cross-diagonal 2: right_shoulder (6) → left_hip (11) */
    if (kp[POSE_KP_RIGHT_SHOULDER].proba >= POSE_OUTLIER_LOW_THRESH &&
        kp[POSE_KP_LEFT_HIP].proba       >= POSE_OUTLIER_LOW_THRESH)
    {
        len += kp_dist(kp[POSE_KP_RIGHT_SHOULDER].x_center,
                       kp[POSE_KP_RIGHT_SHOULDER].y_center,
                       kp[POSE_KP_LEFT_HIP].x_center,
                       kp[POSE_KP_LEFT_HIP].y_center);
        n++;
    }

    if (n > 0)
        return len / (float32_t)n;

    /* Fallback: ~10% of normalised diagonal */
    return 0.10f;
}

/* ------------------------------------------------------------------ */
/* Step 2 – Outlier rejection                                           */
/* ------------------------------------------------------------------ */

/**
 * Process one keypoint through the outlier rejection stage.
 *
 * @param st       Per-keypoint state (updated).
 * @param raw      Raw keypoint from postprocess.
 * @param torso_l  Current torso length (normalised).
 * @param out_x    Accepted x output.
 * @param out_y    Accepted y output.
 * @param out_p    Accepted confidence output.
 * @return         1 if output is valid, 0 if invalid (no hold available).
 */
static uint8_t outlier_process(OutlierKP_t *st,
                                const spe_pp_outBuffer_t *raw,
                                float32_t torso_l,
                                float32_t *out_x,
                                float32_t *out_y,
                                float32_t *out_p)
{
    float32_t x = raw->x_center;
    float32_t y = raw->y_center;
    float32_t p = raw->proba;

    uint8_t accepted = 0;

    /* --- confidence gating ---------------------------------------- */
    if (p >= POSE_OUTLIER_HIGH_THRESH)
    {
        /* Definitely visible: apply geometric checks */
        accepted = 1;
    }
    else if (p >= POSE_OUTLIER_INITIAL_THRESH)
    {
        /* Borderline: accept only if geometric checks pass */
        accepted = 1; /* will be revoked below if checks fail */
    }
    else
    {
        /* Below minimum threshold – reject immediately */
        goto reject;
    }

    /* --- geometric checks (only when there is history) ------------- */
    if (st->last_valid)
    {
        float32_t d = kp_dist(x, y, st->last_x, st->last_y);

        /* Speed check */
        if (d > POSE_OUTLIER_MAX_SPEED_RATIO * torso_l)
        {
            accepted = 0;
        }

        /* Consistency / radius check */
        if (d > POSE_OUTLIER_CONSISTENCY_RATIO * torso_l)
        {
            accepted = 0;
        }
    }

    if (accepted)
    {
        st->last_x     = x;
        st->last_y     = y;
        st->last_proba = p;
        st->last_valid  = 1;
        st->hold_count  = 0;
        *out_x = x;
        *out_y = y;
        *out_p = p;
        return 1;
    }

reject:
    /* Hold last valid position */
    if (st->last_valid && st->hold_count < POSE_OUTLIER_MAX_HOLD_FRAMES)
    {
        st->hold_count++;
        *out_x = st->last_x;
        *out_y = st->last_y;
        *out_p = st->last_proba;
        return 1;
    }

    /* Exceed hold window or never had a valid position */
    *out_x = 0.5f;
    *out_y = 0.5f;
    *out_p = 0.0f;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Step 3 – 1-Euro filter                                               */
/* ------------------------------------------------------------------ */

/**
 * Compute the 1-Euro low-pass filter alpha coefficient.
 *
 * alpha = (2*pi*cutoff*dt) / (2*pi*cutoff*dt + 1)
 */
static inline float32_t euro_alpha(float32_t cutoff, float32_t dt)
{
    float32_t tau = 1.0f / (2.0f * (float32_t)M_PI * cutoff);
    return dt / (dt + tau);
}

/**
 * Run one 1-Euro filter step for a single keypoint.
 *
 * @param st      Filter state (updated).
 * @param x_raw   Outlier-rejected x input.
 * @param y_raw   Outlier-rejected y input.
 * @param dt      Time step in seconds.
 * @param out_x   Filtered x.
 * @param out_y   Filtered y.
 */
static void euro_process(EuroKP_t *st,
                          float32_t x_raw, float32_t y_raw,
                          float32_t dt,
                          float32_t *out_x, float32_t *out_y)
{
    if (!st->initialized)
    {
        st->x_hat  = x_raw;
        st->y_hat  = y_raw;
        st->dx_hat = 0.0f;
        st->dy_hat = 0.0f;
        st->initialized = 1;
        *out_x = x_raw;
        *out_y = y_raw;
        return;
    }

    /* Clamp dt to a reasonable range to avoid instability */
    if (dt < 1e-6f) dt = 1e-6f;
    if (dt > 1.0f)  dt = 1.0f;

    /* --- Derivative filter ---------------------------------------- */
    float32_t a_d = euro_alpha(POSE_EURO_D_CUTOFF, dt);

    float32_t dx_raw = (x_raw - st->x_hat) / dt;
    float32_t dy_raw = (y_raw - st->y_hat) / dt;

    float32_t dx_hat = a_d * dx_raw + (1.0f - a_d) * st->dx_hat;
    float32_t dy_hat = a_d * dy_raw + (1.0f - a_d) * st->dy_hat;

    st->dx_hat = dx_hat;
    st->dy_hat = dy_hat;

    /* --- Signal filter  ------------------------------------------- */
    float32_t speed = sqrtf(dx_hat * dx_hat + dy_hat * dy_hat);

    float32_t cutoff_x = POSE_EURO_MIN_CUTOFF + POSE_EURO_BETA * fabsf(dx_hat);
    float32_t cutoff_y = POSE_EURO_MIN_CUTOFF + POSE_EURO_BETA * fabsf(dy_hat);
    (void)speed; /* retained for documentation; individual cutoffs used */

    float32_t a_x = euro_alpha(cutoff_x, dt);
    float32_t a_y = euro_alpha(cutoff_y, dt);

    st->x_hat = a_x * x_raw + (1.0f - a_x) * st->x_hat;
    st->y_hat = a_y * y_raw + (1.0f - a_y) * st->y_hat;

    *out_x = st->x_hat;
    *out_y = st->y_hat;
}

/* ------------------------------------------------------------------ */
/* Step 4 – Feature extraction                                          */
/* ------------------------------------------------------------------ */

/* Upper-body keypoint indices for HSSC */
static const uint8_t hssc_indices[POSE_HSSC_INDICES_COUNT] = {
    POSE_KP_NOSE,
    POSE_KP_LEFT_EYE, POSE_KP_RIGHT_EYE,
    POSE_KP_LEFT_EAR, POSE_KP_RIGHT_EAR,
    POSE_KP_LEFT_SHOULDER, POSE_KP_RIGHT_SHOULDER
};

/**
 * Compute upper-body horizontal/vertical centre of mass (HSSC).
 *
 * Each contributing keypoint is weighted by its confidence score.
 * If no keypoint has positive confidence, the centre defaults to 0.5.
 *
 * @param filt_x   Filtered x positions [POSE_KP_COUNT].
 * @param filt_y   Filtered y positions [POSE_KP_COUNT].
 * @param filt_p   Filtered confidence  [POSE_KP_COUNT].
 * @param hssc_x   Output: horizontal centre of mass.
 * @param hssc_y   Output: vertical centre of mass.
 */
static void compute_hssc(const float32_t filt_x[POSE_KP_COUNT],
                          const float32_t filt_y[POSE_KP_COUNT],
                          const float32_t filt_p[POSE_KP_COUNT],
                          float32_t *hssc_x,
                          float32_t *hssc_y)
{
    float32_t sum_wx = 0.0f;
    float32_t sum_wy = 0.0f;
    float32_t sum_w  = 0.0f;

    for (uint32_t i = 0; i < POSE_HSSC_INDICES_COUNT; i++)
    {
        uint8_t idx = hssc_indices[i];
        float32_t w = filt_p[idx];
        if (w > 0.0f)
        {
            sum_wx += w * filt_x[idx];
            sum_wy += w * filt_y[idx];
            sum_w  += w;
        }
    }

    if (sum_w > 0.0f)
    {
        *hssc_x = sum_wx / sum_w;
        *hssc_y = sum_wy / sum_w;
    }
    else
    {
        *hssc_x = 0.5f;
        *hssc_y = 0.5f;
    }
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

void PosePipeline_Init(PosePipeline_t *s)
{
    memset(s, 0, sizeof(PosePipeline_t));
    /* Zero-initialisation covers all flags (initialized=0, last_valid=0,
       deriv_initialized=0, win_head=0, win_count=0). */
}

void PosePipeline_Process(PosePipeline_t *s,
                           const spe_pp_outBuffer_t *kp_raw,
                           float32_t dt,
                           PoseFeatureVec_t *out)
{
    /* --- Working buffers ------------------------------------------ */
    float32_t filt_x[POSE_KP_COUNT];
    float32_t filt_y[POSE_KP_COUNT];
    float32_t filt_p[POSE_KP_COUNT];

    /* --- Step 2: Outlier rejection -------------------------------- */
    float32_t torso_l = compute_torso_length(kp_raw);

    for (uint32_t i = 0; i < POSE_KP_COUNT; i++)
    {
        outlier_process(&s->outlier[i], &kp_raw[i], torso_l,
                        &filt_x[i], &filt_y[i], &filt_p[i]);
    }

    /* --- Step 3: 1-Euro filter ------------------------------------ */
    float32_t smooth_x[POSE_KP_COUNT];
    float32_t smooth_y[POSE_KP_COUNT];

    for (uint32_t i = 0; i < POSE_KP_COUNT; i++)
    {
        euro_process(&s->euro[i],
                     filt_x[i], filt_y[i],
                     dt,
                     &smooth_x[i], &smooth_y[i]);
    }

    /* --- Step 4: Feature extraction ------------------------------- */

    /* HSSC */
    float32_t hssc_x, hssc_y;
    compute_hssc(smooth_x, smooth_y, filt_p, &hssc_x, &hssc_y);

    /* VHSSC = d(HSSC_Y)/dt */
    float32_t vhssc = 0.0f;
    float32_t ahssc = 0.0f;

    if (s->deriv_initialized && dt > 1e-6f)
    {
        vhssc = (hssc_y - s->hssc_y_prev) / dt;
        ahssc = (vhssc - s->vhssc_prev)   / dt;
    }

    s->hssc_y_prev       = hssc_y;
    s->vhssc_prev        = vhssc;
    s->deriv_initialized = 1;

    /* --- Assemble 55-feature vector ------------------------------- */
    /*     Layout: [y0,x0,s0, y1,x1,s1, ..., y16,x16,s16,             */
    /*              HSSC_X, HSSC_Y, VHSSC, AHSSC]                      */
    float32_t features[POSE_FEATURE_COUNT];
    for (uint32_t i = 0; i < POSE_KP_COUNT; i++)
    {
        features[i * 3 + 0] = smooth_y[i];   /* y first (matches notebook) */
        features[i * 3 + 1] = smooth_x[i];
        features[i * 3 + 2] = filt_p[i];
    }
    features[POSE_KP_COUNT * 3 + 0] = hssc_x;
    features[POSE_KP_COUNT * 3 + 1] = hssc_y;
    features[POSE_KP_COUNT * 3 + 2] = vhssc;
    features[POSE_KP_COUNT * 3 + 3] = ahssc;

    /* --- Copy to caller output ------------------------------------ */
    if (out != NULL)
    {
        memcpy(out->f, features, sizeof(features));
        out->valid = 1;
    }

    /* --- Step 5: Append to sliding window ------------------------- */
    memcpy(s->win_buf[s->win_head], features, sizeof(features));
    s->win_head = (s->win_head + 1) % POSE_WINDOW_SIZE;
    if (s->win_count < POSE_WINDOW_SIZE)
        s->win_count++;
}

void PosePipeline_GetWindowFeaturesFirst(const PosePipeline_t *s,
                                          float32_t dst[POSE_FEATURE_COUNT][POSE_WINDOW_SIZE])
{
    /* win_buf is stored time-major [timestep][feature].
       Transpose to features-first [feature][timestep] for TCN input (1,55,60). */
    uint32_t oldest = (s->win_count < POSE_WINDOW_SIZE)
                      ? 0
                      : s->win_head;  /* circular buffer oldest entry */

    for (uint32_t t = 0; t < POSE_WINDOW_SIZE; t++)
    {
        uint32_t src_t = (oldest + t) % POSE_WINDOW_SIZE;
        for (uint32_t f = 0; f < POSE_FEATURE_COUNT; f++)
        {
            dst[f][t] = s->win_buf[src_t][f];
        }
    }
}

uint8_t PosePipeline_WindowFull(const PosePipeline_t *s)
{
    return (s->win_count >= POSE_WINDOW_SIZE) ? 1u : 0u;
}
