/**
 ******************************************************************************
 * @file    pose_pipeline.c
 * @brief   Pose feature window builder — Pipeline D (kp13 + velocity)
 *
 *  Implements the same preprocessing as train_window_phase37.py:
 *
 *  Step 1  Coordinate clamp to [0, 1]
 *  Step 2  Low-confidence hold  (conf < POSE_CONF_MASK_THRESHOLD)
 *  Step 3  One-Euro filter on coordinates  (min_cutoff=0.5, beta=0.3)
 *  Step 4  EMA on confidence  (alpha=0.5)
 *  Step 5  HSSC_y/x, RWHC, raw VHSSC
 *  Step 6  EMA on VHSSC  (alpha=0.4) → VHSSC output feature
 *  Step 7  AHSSC = d(VHSSC_ema)/dt,  AHSSC_x = d(VHSSC_x)/dt
 *  Step 8  Assemble raw 45-feat base vector
 *  Step 9  Velocity (29): Δ of raw base at vel_src_indices (first frame = 0)
 *  Step 10 Concatenate [base(45), vel(29)] = 74 features
 *  Step 11 Normalize + clip [0,1]  (P37-vel training stats)
 ******************************************************************************
 */
#include "pose_pipeline.h"
#include <string.h>
#include <math.h>

#define POSE_TWO_PI  (6.2831853071795864769f)

/* ------------------------------------------------------------------ */
/* Internal helpers                                                     */
/* ------------------------------------------------------------------ */

static inline float32_t clamp01(float32_t v)
{
    if (!isfinite(v)) return 0.0f;
    if (v < 0.0f)     return 0.0f;
    if (v > 1.0f)     return 1.0f;
    return v;
}

static inline float32_t euro_alpha(float32_t dt, float32_t cutoff)
{
    float32_t r = POSE_TWO_PI * cutoff * dt;
    return r / (r + 1.0f);
}

static float32_t euro_filter_apply(PoseEuroFilter_t *f, float32_t x, float32_t dt)
{
    if (!isfinite(x)) return f->initialized ? f->x_prev : 0.0f;

    if (!f->initialized)
    {
        f->x_prev      = x;
        f->dx_prev     = 0.0f;
        f->initialized = 1u;
        return x;
    }

    float32_t dx     = (x - f->x_prev) / dt;
    float32_t ad     = euro_alpha(dt, POSE_EURO_D_CUTOFF);
    float32_t dx_hat = ad * dx + (1.0f - ad) * f->dx_prev;
    float32_t cutoff = POSE_EURO_MIN_CUTOFF + POSE_EURO_BETA * fabsf(dx_hat);
    float32_t a      = euro_alpha(dt, cutoff);
    float32_t x_hat  = a * x + (1.0f - a) * f->x_prev;

    f->x_prev  = x_hat;
    f->dx_prev = dx_hat;
    return x_hat;
}

static float32_t ema_filter_apply(PoseEmaFilter_t *f, float32_t x, float32_t alpha)
{
    if (!f->initialized)
    {
        f->x_prev      = x;
        f->initialized = 1u;
        return x;
    }
    float32_t x_hat = alpha * x + (1.0f - alpha) * f->x_prev;
    f->x_prev = x_hat;
    return x_hat;
}

/* ------------------------------------------------------------------ */
/* Feature computation helpers                                          */
/* ------------------------------------------------------------------ */

/* kp13 subset: COCO indices [nose, L/R shoulder, L/R elbow, L/R wrist,
 *                             L/R hip, L/R knee, L/R ankle]           */
static const uint8_t kp13_indices[POSE_KP13_COUNT] = {
    POSE_KP_NOSE,
    POSE_KP_LEFT_SHOULDER,  POSE_KP_RIGHT_SHOULDER,
    POSE_KP_LEFT_ELBOW,     POSE_KP_RIGHT_ELBOW,
    POSE_KP_LEFT_WRIST,     POSE_KP_RIGHT_WRIST,
    POSE_KP_LEFT_HIP,       POSE_KP_RIGHT_HIP,
    POSE_KP_LEFT_KNEE,      POSE_KP_RIGHT_KNEE,
    POSE_KP_LEFT_ANKLE,     POSE_KP_RIGHT_ANKLE,
};

/* HSSC uses kp0-kp6 (nose, eyes, ears, shoulders) */
static const uint8_t hssc_indices[POSE_HSSC_INDICES_COUNT] = {
    POSE_KP_NOSE,
    POSE_KP_LEFT_EYE,  POSE_KP_RIGHT_EYE,
    POSE_KP_LEFT_EAR,  POSE_KP_RIGHT_EAR,
    POSE_KP_LEFT_SHOULDER, POSE_KP_RIGHT_SHOULDER
};

/* Velocity source indices into the 45-feat base array.
 * Matches KP13_VEL_BASE = columns ending in _y or _x from KP13_COLS.
 * Order: kp0_y,kp0_x, kp5_y,kp5_x, ..., kp16_y,kp16_x, HSSC_y, HSSC_x, AHSSC_x */
static const uint8_t vel_src_indices[POSE_VEL_COUNT] = {
     0,  1,   /* kp0  y, x */
     3,  4,   /* kp5  y, x */
     6,  7,   /* kp6  y, x */
     9, 10,   /* kp7  y, x */
    12, 13,   /* kp8  y, x */
    15, 16,   /* kp9  y, x */
    18, 19,   /* kp10 y, x */
    21, 22,   /* kp11 y, x */
    24, 25,   /* kp12 y, x */
    27, 28,   /* kp13 y, x */
    30, 31,   /* kp14 y, x */
    33, 34,   /* kp15 y, x */
    36, 37,   /* kp16 y, x */
    39, 40,   /* HSSC_y, HSSC_x */
    44        /* AHSSC_x */
};

/* MinMax normalization parameters — P37-pure-vel-kp13-w40-gru (74 features) */
static const float32_t norm_min[POSE_FEATURE_COUNT]   = POSE_NORM_MIN;
static const float32_t norm_scale[POSE_FEATURE_COUNT] = POSE_NORM_SCALE;

static void compute_hssc(const float32_t all_kp[POSE_KP_COUNT * 3u],
                         float32_t *hssc_x, float32_t *hssc_y)
{
    float32_t sum_x = 0.0f;
    float32_t sum_y = 0.0f;
    for (uint32_t i = 0u; i < POSE_HSSC_INDICES_COUNT; i++)
    {
        uint32_t idx = hssc_indices[i];
        sum_y += all_kp[idx * 3u + 0u];
        sum_x += all_kp[idx * 3u + 1u];
    }
    *hssc_y = sum_y / (float32_t)POSE_HSSC_INDICES_COUNT;
    *hssc_x = sum_x / (float32_t)POSE_HSSC_INDICES_COUNT;
}

static float32_t compute_rwhc(const float32_t all_kp[POSE_KP_COUNT * 3u])
{
    float32_t min_x = all_kp[1u], max_x = all_kp[1u];
    float32_t min_y = all_kp[0u], max_y = all_kp[0u];
    for (uint32_t i = 1u; i < POSE_KP_COUNT; i++)
    {
        float32_t y = all_kp[i * 3u + 0u];
        float32_t x = all_kp[i * 3u + 1u];
        if (x < min_x) min_x = x;
        if (x > max_x) max_x = x;
        if (y < min_y) min_y = y;
        if (y > max_y) max_y = y;
    }
    float32_t height = max_y - min_y;
    if (height <= 0.0f) height = 0.001f;
    return (max_x - min_x) / height;
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

void PosePipeline_Init(PosePipeline_t *s)
{
    memset(s, 0, sizeof(PosePipeline_t));
}

void PosePipeline_Process(PosePipeline_t *s,
                          const spe_pp_outBuffer_t *kp_raw,
                          float32_t dt,
                          PoseFeatureVec_t *out)
{
    if (out != NULL) out->valid = 0u;
    if ((s == NULL) || (kp_raw == NULL)) return;

    if (!isfinite(dt) || (dt <= 1e-6f)) dt = 1.0f / 15.0f;

    /* Internal full-17kp buffer for HSSC/RWHC */
    float32_t all_kp[POSE_KP_COUNT * 3u];

    /* ------------------------------------------------------------------
     * Steps 1–4 : per-keypoint filtering (all 17 kp)
     * ------------------------------------------------------------------ */
    for (uint32_t i = 0u; i < POSE_KP_COUNT; i++)
    {
        float32_t raw_y = clamp01(kp_raw[i].y_center);
        float32_t raw_x = clamp01(kp_raw[i].x_center);
        float32_t conf  = kp_raw[i].proba;

        float32_t filt_y;
        float32_t filt_x;

        if (conf >= POSE_CONF_MASK_THRESHOLD)
        {
            filt_y = clamp01(euro_filter_apply(&s->y_filter[i], raw_y, dt));
            filt_x = clamp01(euro_filter_apply(&s->x_filter[i], raw_x, dt));
        }
        else
        {
            filt_y = s->y_filter[i].initialized ? clamp01(s->y_filter[i].x_prev) : raw_y;
            filt_x = s->x_filter[i].initialized ? clamp01(s->x_filter[i].x_prev) : raw_x;
        }

        all_kp[i * 3u + 0u] = filt_y;
        all_kp[i * 3u + 1u] = filt_x;
        all_kp[i * 3u + 2u] = clamp01(ema_filter_apply(&s->conf_filter[i], conf, POSE_CONF_EMA_ALPHA));
    }

    /* ------------------------------------------------------------------
     * Step 5 : HSSC_y/x and RWHC
     * ------------------------------------------------------------------ */
    float32_t hssc_x;
    float32_t hssc_y;
    compute_hssc(all_kp, &hssc_x, &hssc_y);
    float32_t rwhc = compute_rwhc(all_kp);

    /* ------------------------------------------------------------------
     * Steps 6–7 : VHSSC EMA → AHSSC / AHSSC_x
     * ------------------------------------------------------------------ */
    float32_t vhssc_raw   = 0.0f;
    float32_t vhssc_x_raw = 0.0f;

    if (s->deriv_initialized)
    {
        vhssc_raw   = (hssc_y - s->hssc_y_prev) / dt;
        vhssc_x_raw = (hssc_x - s->hssc_x_prev) / dt;
    }

    float32_t vhssc_ema;
    if (s->deriv_initialized)
    {
        vhssc_ema = POSE_VHSSC_EMA_ALPHA * vhssc_raw
                    + (1.0f - POSE_VHSSC_EMA_ALPHA) * s->vhssc_ema;
    }
    else
    {
        vhssc_ema = 0.0f;
    }

    float32_t ahssc   = 0.0f;
    float32_t ahssc_x = 0.0f;

    if (s->deriv_initialized)
    {
        ahssc   = (vhssc_ema   - s->vhssc_ema)   / dt;
        ahssc_x = (vhssc_x_raw - s->vhssc_x_ema) / dt;
    }

    /* ------------------------------------------------------------------
     * Step 8 : Assemble raw 45-feat base vector
     *   [0..38]  kp13 × (y, x, conf)  — 39 features
     *   [39..44] HSSC_y, HSSC_x, RWHC, VHSSC, AHSSC, AHSSC_x
     * ------------------------------------------------------------------ */
    float32_t raw_base[POSE_BASE_FEAT_COUNT];

    for (uint32_t k = 0u; k < POSE_KP13_COUNT; k++)
    {
        uint32_t src = kp13_indices[k];
        raw_base[k * 3u + 0u] = all_kp[src * 3u + 0u];
        raw_base[k * 3u + 1u] = all_kp[src * 3u + 1u];
        raw_base[k * 3u + 2u] = all_kp[src * 3u + 2u];
    }

    raw_base[39u] = hssc_y;
    raw_base[40u] = hssc_x;
    raw_base[41u] = rwhc;
    raw_base[42u] = vhssc_ema;
    raw_base[43u] = ahssc;
    raw_base[44u] = ahssc_x;

    /* ------------------------------------------------------------------
     * Step 9 : Velocity — forward diff of raw base at vel_src_indices
     *   Matches Python: add_velocity(frames, vel_idx) with delta[0] = 0
     *   Computed BEFORE normalization (same as training pipeline).
     * ------------------------------------------------------------------ */
    float32_t vel[POSE_VEL_COUNT];

    if (s->deriv_initialized)
    {
        for (uint32_t i = 0u; i < POSE_VEL_COUNT; i++)
        {
            vel[i] = raw_base[vel_src_indices[i]] - s->prev_raw_vel[i];
        }
    }
    else
    {
        memset(vel, 0, sizeof(vel));
    }

    /* Update state */
    for (uint32_t i = 0u; i < POSE_VEL_COUNT; i++)
    {
        s->prev_raw_vel[i] = raw_base[vel_src_indices[i]];
    }
    s->hssc_y_prev       = hssc_y;
    s->hssc_x_prev       = hssc_x;
    s->vhssc_ema         = vhssc_ema;
    s->vhssc_x_ema       = vhssc_x_raw;
    s->deriv_initialized = 1u;

    /* ------------------------------------------------------------------
     * Step 10 : Assemble full 74-feat vector [base(45) | vel(29)]
     * Step 11 : Normalize + clip [0, 1]
     * ------------------------------------------------------------------ */
    float32_t features[POSE_FEATURE_COUNT];

    memcpy(features,                  raw_base, POSE_BASE_FEAT_COUNT * sizeof(float32_t));
    memcpy(features + POSE_BASE_FEAT_COUNT, vel, POSE_VEL_COUNT      * sizeof(float32_t));

    for (uint32_t f = 0u; f < POSE_FEATURE_COUNT; f++)
    {
        float32_t v = (features[f] - norm_min[f]) / norm_scale[f];
        if      (v < 0.0f) v = 0.0f;
        else if (v > 1.0f) v = 1.0f;
        features[f] = v;
    }

    if (out != NULL)
    {
        memcpy(out->f, features, sizeof(features));
        out->valid = 1u;
    }

    memcpy(s->win_buf[s->win_head], features, sizeof(features));
    s->win_head = (s->win_head + 1u) % POSE_WINDOW_SIZE;
    if (s->win_count < POSE_WINDOW_SIZE) s->win_count++;
}

void PosePipeline_GetWindowFeaturesFirst(const PosePipeline_t *s,
                                         float32_t dst[POSE_FEATURE_COUNT][POSE_WINDOW_SIZE])
{
    uint32_t oldest = (s->win_count < POSE_WINDOW_SIZE) ? 0u : s->win_head;
    for (uint32_t t = 0u; t < POSE_WINDOW_SIZE; t++)
    {
        uint32_t src_t = (oldest + t) % POSE_WINDOW_SIZE;
        for (uint32_t f = 0u; f < POSE_FEATURE_COUNT; f++)
            dst[f][t] = s->win_buf[src_t][f];
    }
}

void PosePipeline_GetWindowTimeFirst(const PosePipeline_t *s,
                                     float32_t dst[POSE_WINDOW_SIZE][POSE_FEATURE_COUNT])
{
    uint32_t oldest = (s->win_count < POSE_WINDOW_SIZE) ? 0u : s->win_head;
    for (uint32_t t = 0u; t < POSE_WINDOW_SIZE; t++)
    {
        uint32_t src_t = (oldest + t) % POSE_WINDOW_SIZE;
        memcpy(dst[t], s->win_buf[src_t], sizeof(dst[t]));
    }
}

uint8_t PosePipeline_WindowFull(const PosePipeline_t *s)
{
    return (s->win_count >= POSE_WINDOW_SIZE) ? 1u : 0u;
}

void PosePipeline_GetLatestFeature(const PosePipeline_t *s,
                                   float32_t dst[POSE_FEATURE_COUNT])
{
    uint32_t last = (s->win_head == 0u) ? (POSE_WINDOW_SIZE - 1u) : (s->win_head - 1u);
    memcpy(dst, s->win_buf[last], POSE_FEATURE_COUNT * sizeof(float32_t));
}
