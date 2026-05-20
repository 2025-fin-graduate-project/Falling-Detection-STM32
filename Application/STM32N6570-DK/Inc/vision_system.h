/**
 ******************************************************************************
 * @file    vision_system.h
 * @brief   Vision system module for camera, NPU inference, and postprocessing.
 ******************************************************************************
 */

#ifndef VISION_SYSTEM_H
#define VISION_SYSTEM_H

#include <stdint.h>
#include "stai.h"
#include "app_postprocess.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
  int8_t raw_min;
  int8_t raw_max;
  float32_t raw_min_dequant;
  float32_t raw_max_dequant;
  float32_t max_keypoint_proba;
  uint32_t visible_keypoints;
  uint32_t person_keypoints;
  uint8_t person_present;
  uint8_t postprocess_valid;
  uint8_t draw_keypoints;
  uint8_t keypoints_too_spread;
  uint32_t output_width;
  uint32_t output_height;
  uint32_t output_channels;
  int32_t postprocess_status;
  uint32_t inference_ms;
} VisionMetrics_t;

/**
 * @brief  Initialize Vision System (Runtime, Network, Postprocess, Camera)
 * @param  bg_width Pointer to receive background width (from camera)
 * @param  bg_height Pointer to receive background height (from camera)
 * @retval 0 on success, negative on error
 */
int32_t Vision_Init(uint32_t *bg_width, uint32_t *bg_height);

/**
 * @brief  Start camera display stream to buffer
 * @param  buffer Address of the background buffer
 */
void Vision_StartCamera(uint8_t *buffer);

/**
 * @brief  Execute one iteration of the vision pipeline (Capture -> Inference -> Postprocess)
 * @param  pp_out Pointer to receive postprocessing output results
 * @param  metrics Pointer to receive vision debug metrics
 * @retval 0 on success, negative if no frame or error
 */
int32_t Vision_Process(spe_pp_out_t **pp_out, VisionMetrics_t *metrics);

/* Granular functions for pipelining */
void Vision_CaptureStart(void);
int32_t Vision_CaptureWait(void);
void Vision_InferenceStart(void);
void Vision_InferenceWait(void);
int32_t Vision_PostProcess(spe_pp_out_t **pp_out, VisionMetrics_t *metrics);

#ifdef __cplusplus
}
#endif

#endif /* VISION_SYSTEM_H */
