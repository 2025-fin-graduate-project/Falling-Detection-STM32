 /**
 ******************************************************************************
 * @file    main.c
 * @author  GPM Application Team
 *
 ******************************************************************************
 */
#include <stdio.h>
#include <assert.h>

#include "system_init.h"
#include "vision_system.h"
#include "presence_manager.h"
#include "ui_system.h"
#include "fall_detection.h"
#include "pose_pipeline.h"
#include "app_config.h"
#include "profiler.h"

BINDINGS;

#ifndef APP_GIT_SHA1_STRING
#define APP_GIT_SHA1_STRING "dev"
#endif
#ifndef APP_VERSION_STRING
#define APP_VERSION_STRING "unversioned"
#endif

static PosePipeline_t pose_pipeline;
static uint32_t pose_last_tick;

/**
  * @brief  Main program
  */
int main(void)
{
  uint32_t bg_w, bg_h;

  System_Init();
  Profiler_Init();

  printf("========================================\n");
  printf("STM32N6 Fall Detection System %s (%s)\n", APP_VERSION_STRING, APP_GIT_SHA1_STRING);
  printf("Build date & time: %s %s\n", __DATE__, __TIME__);
  printf("NN model: %s\n", STAI_NETWORK_ORIGIN_MODEL_NAME);
  printf("Fall model: GRU (p38-nv-a65 Pipelined)\n");
  printf("========================================\n");

  /* Initialize modules */
  Vision_Init(&bg_w, &bg_h);
  Presence_Init();
  FallDetection_Init();
  PosePipeline_Init(&pose_pipeline);
  UI_Init(bg_w, bg_h);

  /* Start camera display stream */
  Vision_StartCamera(UI_GetBgBuffer());

  pose_last_tick = HAL_GetTick();

  /* Pipelining state */
  static spe_pp_out_t *pp_output_prev = NULL;
  static uint8_t pp_prev_valid = 0;

  while (1)
  {
    spe_pp_out_t *pp_output = NULL;
    VisionMetrics_t v_metrics = {0};

    /* 1. Start Vision Capture */
    Vision_CaptureStart();
    
    /* 2. Wait for Capture Completion */
    if (Vision_CaptureWait() != 0) continue;

    /* 3. Start NPU Inference (Async) */
    Vision_InferenceStart();

    /* 4. Parallel Step: GRU for previous frame results */
    if (pp_prev_valid)
    {
      uint32_t now_tick = HAL_GetTick();
      float32_t dt_s = (float32_t)(now_tick - pose_last_tick) * 1e-3f;
      if (dt_s <= 0.0f || dt_s > 1.0f) dt_s = 1.0f / 15.0f;
      pose_last_tick = now_tick;

      PoseFeatureVec_t feat_vec;
      PosePipeline_Process(&pose_pipeline, pp_output_prev->pOutBuff, dt_s, &feat_vec);
      FallDetection_Update(&pose_pipeline);
    }

    /* 5. Wait for NPU Inference Completion */
    Vision_InferenceWait();

    /* 6. Postprocess NPU results */
    if (Vision_PostProcess(&pp_output, &v_metrics) != 0)
    {
      pp_prev_valid = 0;
      continue;
    }

    /* 7. Presence Logic (State Machine) */
    uint8_t pose_valid = Presence_Update(&v_metrics);
    
    /* decide whether to render the skeleton */
    v_metrics.draw_keypoints = (pose_valid && v_metrics.visible_keypoints > 0 && !v_metrics.keypoints_too_spread);

    if (!pose_valid)
    {
      FallDetection_Invalidate(&pose_pipeline);
    }

    /* Update pipelining state for next iteration */
    pp_output_prev = pp_output;
    pp_prev_valid = pose_valid;

    /* 8. UI Update (Rendering & Alarm) */
    UI_Update(pp_output, &v_metrics, Presence_GetStatus(), &fall_state);
    Alarm_Update();

    Profiler_PrintReport(3000);
  }
}
