/**
 ******************************************************************************
 * @file    pose_pipeline.c
 * @brief   Pose feature window builder for fall detection.
 *
 *          The fall models were trained from the MoveNet postprocessing output,
 *          so this file keeps the same raw keypoint layout:
 *          [y0,x0,score0, ..., y16,x16,score16, HSSC_Y,HSSC_X,RWHC,VHSSC].
 ******************************************************************************
 */
#include "pose_pipeline.h"
#include <string.h>
#include <math.h>

#define POSE_TWO_PI  (6.2831853071795864769f)

static inline float32_t clamp01(float32_t v)
{
  if (!isfinite(v))
  {
    return 0.0f;
  }
  if (v < 0.0f)
  {
    return 0.0f;
  }
  if (v > 1.0f)
  {
    return 1.0f;
  }
  return v;
}

static float32_t euro_alpha(float32_t dt, float32_t cutoff)
{
  float32_t r = POSE_TWO_PI * cutoff * dt;
  return r / (r + 1.0f);
}

static float32_t euro_filter_apply(PoseEuroFilter_t *filter, float32_t x, float32_t dt)
{
  if ((filter == NULL) || !isfinite(x))
  {
    return x;
  }

  if (!filter->initialized)
  {
    filter->x_prev = x;
    filter->dx_prev = 0.0f;
    filter->initialized = 1u;
    return x;
  }

  /* Velocity Cap: Prevent any single-frame jump larger than 0.25 (25% of screen).
   * 0.25 is enough to capture a very fast fall at 15fps, but still blocks 
   * the extreme coordinate 'explosions' caused by model noise. */
  float32_t max_dist = 0.25f;
  if (x > filter->x_prev + max_dist) x = filter->x_prev + max_dist;
  if (x < filter->x_prev - max_dist) x = filter->x_prev - max_dist;

  float32_t dx = (x - filter->x_prev) / dt;
  float32_t dx_alpha = euro_alpha(dt, POSE_EURO_D_CUTOFF);
  float32_t dx_hat = (dx_alpha * dx) + ((1.0f - dx_alpha) * filter->dx_prev);
  float32_t cutoff = POSE_EURO_MIN_CUTOFF + (POSE_EURO_BETA * fabsf(dx_hat));
  float32_t x_alpha = euro_alpha(dt, cutoff);
  float32_t x_hat = (x_alpha * x) + ((1.0f - x_alpha) * filter->x_prev);

  filter->x_prev = x_hat;
  filter->dx_prev = dx_hat;
  return x_hat;
}

static const uint8_t hssc_indices[POSE_HSSC_INDICES_COUNT] = {
  POSE_KP_NOSE,
  POSE_KP_LEFT_EYE,
  POSE_KP_RIGHT_EYE,
  POSE_KP_LEFT_EAR,
  POSE_KP_RIGHT_EAR,
  POSE_KP_LEFT_SHOULDER,
  POSE_KP_RIGHT_SHOULDER
};

static void compute_hssc(const float32_t features[POSE_FEATURE_COUNT],
                         float32_t *hssc_x,
                         float32_t *hssc_y)
{
  float32_t sum_x = 0.0f;
  float32_t sum_y = 0.0f;

  for (uint32_t i = 0; i < POSE_HSSC_INDICES_COUNT; i++)
  {
    uint32_t idx = hssc_indices[i];
    sum_y += features[idx * 3 + 0];
    sum_x += features[idx * 3 + 1];
  }

  *hssc_x = sum_x / (float32_t)POSE_HSSC_INDICES_COUNT;
  *hssc_y = sum_y / (float32_t)POSE_HSSC_INDICES_COUNT;
}

static float32_t compute_rwhc(const float32_t features[POSE_FEATURE_COUNT])
{
  float32_t min_x = features[1];
  float32_t max_x = features[1];
  float32_t min_y = features[0];
  float32_t max_y = features[0];

  for (uint32_t i = 1; i < POSE_KP_COUNT; i++)
  {
    float32_t y = features[i * 3 + 0];
    float32_t x = features[i * 3 + 1];

    if (x < min_x)
    {
      min_x = x;
    }
    if (x > max_x)
    {
      max_x = x;
    }
    if (y < min_y)
    {
      min_y = y;
    }
    if (y > max_y)
    {
      max_y = y;
    }
  }

  float32_t height = max_y - min_y;
  if (height <= 0.0f)
  {
    height = 0.001f;
  }

  return (max_x - min_x) / height;
}

void PosePipeline_Init(PosePipeline_t *s)
{
  memset(s, 0, sizeof(PosePipeline_t));
}

static float32_t ema_filter_apply(PoseEmaFilter_t *filter, float32_t x, float32_t alpha)
{
  if (filter == NULL) return x;
  if (!filter->initialized) {
    filter->x_prev = x;
    filter->initialized = 1u;
    return x;
  }
  float32_t x_hat = (alpha * x) + ((1.0f - alpha) * filter->x_prev);
  filter->x_prev = x_hat;
  return x_hat;
}

void PosePipeline_Process(PosePipeline_t *s,
                          const spe_pp_outBuffer_t *kp_raw,
                          float32_t dt,
                          PoseFeatureVec_t *out)
{
  if (out != NULL)
  {
    out->valid = 0u;
  }

  if ((s == NULL) || (kp_raw == NULL))
  {
    return;
  }

  if (!isfinite(dt) || (dt <= 1e-6f))
  {
    dt = 1.0f / 15.0f;
  }

  float32_t features[POSE_FEATURE_COUNT];

  for (uint32_t i = 0; i < POSE_KP_COUNT; i++)
  {
    float32_t y = clamp01(kp_raw[i].y_center);
    float32_t x = clamp01(kp_raw[i].x_center);

    features[i * 3 + 0] = clamp01(euro_filter_apply(&s->y_filter[i], y, dt));
    features[i * 3 + 1] = clamp01(euro_filter_apply(&s->x_filter[i], x, dt));

    /* Confidence Smoothing (EMA) with asymmetrical response:
     * - Fast Attack (0.8): React almost instantly when a keypoint is detected.
     * - Slow Decay (POSE_CONF_EMA_ALPHA): Hold slightly but not as strong as before. */
    float32_t current_conf = kp_raw[i].proba;
    float32_t alpha = (current_conf > s->conf_filter[i].x_prev) ? 0.8f : POSE_CONF_EMA_ALPHA;
    features[i * 3 + 2] = clamp01(ema_filter_apply(&s->conf_filter[i], current_conf, alpha));
  }

  float32_t hssc_x;
  float32_t hssc_y;
  compute_hssc(features, &hssc_x, &hssc_y);
  float32_t rwhc = compute_rwhc(features);

  float32_t vhssc = 0.0f;

  if (s->deriv_initialized)
  {
    vhssc = (hssc_y - s->hssc_y_prev) / dt;
  }

  s->hssc_y_prev = hssc_y;
  s->vhssc_prev = vhssc;
  s->deriv_initialized = 1u;

  features[POSE_KP_COUNT * 3 + 0] = hssc_y;
  features[POSE_KP_COUNT * 3 + 1] = hssc_x;
  features[POSE_KP_COUNT * 3 + 2] = rwhc;
  features[POSE_KP_COUNT * 3 + 3] = vhssc;

  if (out != NULL)
  {
    memcpy(out->f, features, sizeof(features));
    out->valid = 1u;
  }

  memcpy(s->win_buf[s->win_head], features, sizeof(features));
  s->win_head = (s->win_head + 1u) % POSE_WINDOW_SIZE;
  if (s->win_count < POSE_WINDOW_SIZE)
  {
    s->win_count++;
  }
}

void PosePipeline_GetWindowFeaturesFirst(const PosePipeline_t *s,
                                         float32_t dst[POSE_FEATURE_COUNT][POSE_WINDOW_SIZE])
{
  uint32_t oldest = (s->win_count < POSE_WINDOW_SIZE) ? 0u : s->win_head;

  for (uint32_t t = 0; t < POSE_WINDOW_SIZE; t++)
  {
    uint32_t src_t = (oldest + t) % POSE_WINDOW_SIZE;
    for (uint32_t f = 0; f < POSE_FEATURE_COUNT; f++)
    {
      dst[f][t] = s->win_buf[src_t][f];
    }
  }
}

void PosePipeline_GetWindowTimeFirst(const PosePipeline_t *s,
                                     float32_t dst[POSE_WINDOW_SIZE][POSE_FEATURE_COUNT])
{
  uint32_t oldest = (s->win_count < POSE_WINDOW_SIZE) ? 0u : s->win_head;

  for (uint32_t t = 0; t < POSE_WINDOW_SIZE; t++)
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
  uint32_t last = (s->win_head == 0) ? (POSE_WINDOW_SIZE - 1u) : (s->win_head - 1u);
  memcpy(dst, s->win_buf[last], POSE_FEATURE_COUNT * sizeof(float32_t));
}
