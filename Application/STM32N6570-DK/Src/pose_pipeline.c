/**
 ******************************************************************************
 * @file    pose_pipeline.c
 * @brief   Pose feature window builder for fall detection.
 *
 *          The fall models were trained from the MoveNet postprocessing output,
 *          so this file keeps the same raw keypoint layout:
 *          [y0,x0,score0, ..., y16,x16,score16, HSSC_X,HSSC_Y,VHSSC,AHSSC].
 ******************************************************************************
 */
#include "pose_pipeline.h"
#include <string.h>
#include <math.h>

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

void PosePipeline_Init(PosePipeline_t *s)
{
  memset(s, 0, sizeof(PosePipeline_t));
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
    features[i * 3 + 0] = clamp01(kp_raw[i].y_center);
    features[i * 3 + 1] = clamp01(kp_raw[i].x_center);
    features[i * 3 + 2] = clamp01(kp_raw[i].proba);
  }

  float32_t hssc_x;
  float32_t hssc_y;
  compute_hssc(features, &hssc_x, &hssc_y);

  float32_t vhssc = 0.0f;
  float32_t ahssc = 0.0f;

  if (s->deriv_initialized)
  {
    vhssc = (hssc_y - s->hssc_y_prev) / dt;
    ahssc = (vhssc - s->vhssc_prev) / dt;
  }

  s->hssc_y_prev = hssc_y;
  s->vhssc_prev = vhssc;
  s->deriv_initialized = 1u;

  features[POSE_KP_COUNT * 3 + 0] = hssc_x;
  features[POSE_KP_COUNT * 3 + 1] = hssc_y;
  features[POSE_KP_COUNT * 3 + 2] = vhssc;
  features[POSE_KP_COUNT * 3 + 3] = ahssc;

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
