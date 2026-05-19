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

  printf("========================================\n");
  printf("STM32N6 Fall Detection System %s (%s)\n", APP_VERSION_STRING, APP_GIT_SHA1_STRING);
  printf("Build date & time: %s %s\n", __DATE__, __TIME__);
  printf("NN model: %s\n", STAI_NETWORK_ORIGIN_MODEL_NAME);
  printf("Fall model: GRU (p38-nv-a65)\n");
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

  while (1)
  {
    spe_pp_out_t *pp_output = NULL;
    VisionMetrics_t v_metrics = {0};

    /* 1. Vision Pipeline (Capture -> Inference -> Postprocess) */
    if (Vision_Process(&pp_output, &v_metrics) != 0) continue;

    /* 2. Presence Logic (State Machine) */
    uint8_t pose_valid = Presence_Update(&v_metrics);
    
    /* decid whether to render the skeleton */
    v_metrics.draw_keypoints = (pose_valid && v_metrics.visible_keypoints > 0 && !v_metrics.keypoints_too_spread);

    /* 3. Pose Processing & Fall Detection */
    if (pose_valid)
    {
      uint32_t now_tick = HAL_GetTick();
      float32_t dt_s = (float32_t)(now_tick - pose_last_tick) * 1e-3f;
      if (dt_s <= 0.0f || dt_s > 1.0f) dt_s = 1.0f / 15.0f;
      pose_last_tick = now_tick;

      PoseFeatureVec_t feat_vec;
      PosePipeline_Process(&pose_pipeline, pp_output->pOutBuff, dt_s, &feat_vec);
      FallDetection_Update(&pose_pipeline);
    }
    else
    {
      FallDetection_Invalidate(&pose_pipeline);
    }

    /* 4. UI Update (Rendering & Alarm) */
    UI_Update(pp_output, &v_metrics, Presence_GetStatus(), &fall_state);
    Alarm_Update();
  }
}
