/**
 ******************************************************************************
 * @file    fall_detection.h
 * @brief   High-level fall detection application logic.
 ******************************************************************************
 */

#ifndef FALL_DETECTION_H
#define FALL_DETECTION_H

#include "pose_pipeline.h"
#include "app_config.h"
#include "stai.h"

typedef struct
{
  uint8_t window_ready;
  uint8_t fall_detected;
  uint32_t frame_count;
  uint32_t inference_ms;
  float32_t normal_logit;
  float32_t fall_logit;
  float32_t normal_score;
  float32_t fall_score;
  uint8_t fall_vote_buf[GRU_FALL_VOTE_WINDOW];
  uint32_t fall_vote_idx;
  uint32_t fall_latch_tick;   /* HAL tick of last confirmed fall; 0 if never */
} FallDetectionState_t;

void FallDetection_Init(void);
void FallDetection_Update(PosePipeline_t *pipeline);
void FallDetection_Invalidate(PosePipeline_t *pipeline);
void FallDetection_RunInference(void);

/* Shared state access */
extern FallDetectionState_t fall_state;

#endif /* FALL_DETECTION_H */
