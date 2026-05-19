/**
 ******************************************************************************
 * @file    vision_system.c
 * @brief   Vision system implementation for MoveNet and camera pipeline.
 ******************************************************************************
 */

#include "vision_system.h"
#include "cmw_camera.h"
#include "app_camerapipeline.h"
#include "stai.h"
#include "stai_network.h"
#include "pose_pipeline.h"
#include "app_config.h"
#include "crop_img.h"
#include "utils.h"
#include <string.h>
#include <stdio.h>
#include <assert.h>

/* Alignment macros for DCMIPP hardware (must be multiple of 16) */
#define ALIGN_TO_16(value) (((value) + 15) & ~15)

#if (STAI_NETWORK_IN_1_WIDTH * STAI_NETWORK_IN_1_CHANNEL) != ALIGN_TO_16(STAI_NETWORK_IN_1_WIDTH * STAI_NETWORK_IN_1_CHANNEL)
#define DCMIPP_OUT_NN_LEN (ALIGN_TO_16(STAI_NETWORK_IN_1_WIDTH * STAI_NETWORK_IN_1_CHANNEL) * STAI_NETWORK_IN_1_HEIGHT)
#define DCMIPP_OUT_NN_BUFF_LEN (DCMIPP_OUT_NN_LEN + 32 - (DCMIPP_OUT_NN_LEN % 32))
__attribute__ ((aligned (32)))
static uint8_t dcmipp_out_nn_buf[DCMIPP_OUT_NN_BUFF_LEN];
#else
static uint8_t *dcmipp_out_nn_buf = NULL;
#endif

/* Vision context */
STAI_NETWORK_CONTEXT_DECLARE(network_context, STAI_NETWORK_CONTEXT_SIZE)

static uint32_t nn_in_len = 0;
static stai_ptr nn_in = NULL;
static stai_ptr nn_out[STAI_NETWORK_OUT_NUM] = {0};
static int32_t  nn_out_len[STAI_NETWORK_OUT_NUM] = {0};
static stai_size number_output = 0;
static uint32_t pitch_nn = 0;

/* Postprocessing */
#if POSTPROCESS_TYPE == POSTPROCESS_SPE_MOVENET_UI
static spe_movenet_pp_static_param_t pp_params;
static spe_pp_out_t pp_output;
#else
#error "Only SPE_MOVENET_UI is supported in this module for now"
#endif

/* Sync with camera callback */
volatile int32_t cameraFrameReceived = 0;

/* Internal helpers */
static void NeuralNetwork_Init(void);
static void Run_Inference(stai_network *network_instance);
static void Update_VisionMetrics(VisionMetrics_t *metrics);

int32_t Vision_Init(uint32_t *bg_width, uint32_t *bg_height)
{
  NeuralNetwork_Init();

  stai_network_info info;
  if (stai_network_get_info(network_context, &info) != STAI_SUCCESS) return -1;
  app_postprocess_init(&pp_params, &info);

  CameraPipeline_Init(bg_width, bg_height, &pitch_nn);

  return 0;
}

void Vision_StartCamera(uint8_t *buffer)
{
  CameraPipeline_DisplayPipe_Start(buffer, CMW_MODE_CONTINUOUS);
}

int32_t Vision_Process(spe_pp_out_t **pp_out, VisionMetrics_t *metrics)
{
  CameraPipeline_IspUpdate();

  uint8_t *nn_capture_dst;
  if (pitch_nn != (STAI_NETWORK_IN_1_WIDTH * STAI_NETWORK_IN_1_CHANNEL)) {
    nn_capture_dst = dcmipp_out_nn_buf;
  } else {
    nn_capture_dst = (uint8_t *)nn_in;
    SCB_InvalidateDCache_by_Addr(nn_in, nn_in_len);
  }

  /* Trigger Snapshot */
  CameraPipeline_NNPipe_Start(nn_capture_dst, CMW_MODE_SNAPSHOT);

  /* Wait for frame */
  uint32_t t_start = HAL_GetTick();
  while (cameraFrameReceived == 0) {
    if (HAL_GetTick() - t_start > 1000) return -2; // Timeout
  };
  cameraFrameReceived = 0;

  /* Handle Cropping if hardware pitch != network width */
  if (pitch_nn != (STAI_NETWORK_IN_1_WIDTH * STAI_NETWORK_IN_1_CHANNEL)) {
#if (STAI_NETWORK_IN_1_WIDTH * STAI_NETWORK_IN_1_CHANNEL) != ALIGN_TO_16(STAI_NETWORK_IN_1_WIDTH * STAI_NETWORK_IN_1_CHANNEL)
    SCB_InvalidateDCache_by_Addr(dcmipp_out_nn_buf, sizeof(dcmipp_out_nn_buf));
    img_crop(dcmipp_out_nn_buf, nn_in, pitch_nn, STAI_NETWORK_IN_1_WIDTH, STAI_NETWORK_IN_1_HEIGHT, STAI_NETWORK_IN_1_CHANNEL);
    SCB_CleanInvalidateDCache_by_Addr(nn_in, nn_in_len);
#endif
  } else {
    SCB_InvalidateDCache_by_Addr(nn_in, nn_in_len);
  }

  /* Inference */
  uint32_t ts0 = HAL_GetTick();
  Run_Inference(network_context);
  metrics->inference_ms = HAL_GetTick() - ts0;

  /* Invalidate output buffers for CPU postprocessing */
  for (int i = 0; i < (int)number_output; i++) {
    SCB_InvalidateDCache_by_Addr(nn_out[i], nn_out_len[i]);
  }

  /* Postprocessing */
  metrics->postprocess_status = app_postprocess_run((void **)nn_out, number_output, &pp_output, &pp_params);
  
  Update_VisionMetrics(metrics);
  
  *pp_out = &pp_output;
  return 0;
}

static void NeuralNetwork_Init(void)
{
  stai_network_info info;
  
  assert(stai_runtime_init() == STAI_SUCCESS);
  assert(stai_network_init(network_context) == STAI_SUCCESS);
  assert(stai_network_get_info(network_context, &info) == STAI_SUCCESS);
  
  number_output = STAI_NETWORK_OUT_NUM;
  nn_in_len = info.inputs[0].size_bytes;
  
  assert(stai_network_get_inputs(network_context, &nn_in, (stai_size *)&info.n_inputs) == STAI_SUCCESS);
  assert(stai_network_get_outputs(network_context, nn_out, &number_output) == STAI_SUCCESS);
  
  for (int i = 0; i < (int)number_output; i++) {
    nn_out_len[i] = info.outputs[i].size_bytes;
  }
}

static void Run_Inference(stai_network *network_instance)
{
  stai_return_code ret;
  do {
    ret = stai_network_run(network_instance, STAI_MODE_ASYNC);
    if (ret == STAI_RUNNING_WFE) LL_ATON_OSAL_WFE();
  } while (ret == STAI_RUNNING_WFE || ret == STAI_RUNNING_NO_WFE);

  assert(stai_ext_network_new_inference(network_instance) == STAI_SUCCESS);
}

static void Update_VisionMetrics(VisionMetrics_t *metrics)
{
  metrics->postprocess_valid = ((metrics->postprocess_status == AI_SPE_POSTPROCESS_ERROR_NO) && 
                                (pp_output.pOutBuff != NULL)) ? 1u : 0u;

  metrics->raw_min = 0;
  metrics->raw_max = 0;
  metrics->raw_min_dequant = 0.0f;
  metrics->raw_max_dequant = 0.0f;
  metrics->max_keypoint_proba = 0.0f;
  metrics->visible_keypoints = 0;
  metrics->person_keypoints = 0;
  metrics->person_present = 0u;
  metrics->draw_keypoints = 0u;
  metrics->keypoints_too_spread = 0u;
  metrics->output_width = STAI_NETWORK_OUT_1_WIDTH;
  metrics->output_height = STAI_NETWORK_OUT_1_HEIGHT;
  metrics->output_channels = STAI_NETWORK_OUT_1_CHANNEL;

  if (metrics->postprocess_valid && (nn_out[0] != NULL)) {
    int8_t *raw_output = (int8_t *)nn_out[0];
    int32_t raw_len = nn_out_len[0];
    int8_t r_min = raw_output[0];
    int8_t r_max = raw_output[0];

    for (int32_t i = 1; i < raw_len; i++) {
      if (raw_output[i] < r_min) r_min = raw_output[i];
      if (raw_output[i] > r_max) r_max = raw_output[i];
    }
    metrics->raw_min = r_min;
    metrics->raw_max = r_max;
    metrics->raw_min_dequant = pp_params.raw_scale * (float32_t)((int32_t)r_min - (int32_t)pp_params.raw_zero_point);
    metrics->raw_max_dequant = pp_params.raw_scale * (float32_t)((int32_t)r_max - (int32_t)pp_params.raw_zero_point);

    /* Robust Presence: Top-5 Average Confidence */
    float32_t scores[POSE_KP_COUNT];
    for (uint32_t i = 0; i < POSE_KP_COUNT; i++) {
      float32_t proba = pp_output.pOutBuff[i].proba;
      scores[i] = proba;
      if (proba > metrics->max_keypoint_proba) metrics->max_keypoint_proba = proba;
      if (proba >= AI_POSE_PP_CONF_THRESHOLD) metrics->visible_keypoints++;
      if (proba >= FALL_PERSON_CONF_THRESHOLD) metrics->person_keypoints++;
    }

    /* Sort top 5 scores */
    for (int i = 0; i < 5; i++) {
      for (int j = i + 1; j < POSE_KP_COUNT; j++) {
        if (scores[j] > scores[i]) {
          float32_t tmp = scores[i];
          scores[i] = scores[j];
          scores[j] = tmp;
        }
      }
    }
    float32_t top5_avg = (scores[0] + scores[1] + scores[2] + scores[3] + scores[4]) / 5.0f;
    metrics->person_present = (top5_avg >= 0.2f) ? 1u : 0u;

    /* Keypoint spread check */
    float32_t x_min = 1.0f, x_max = 0.0f, y_min = 1.0f, y_max = 0.0f;
    uint32_t n_spread = 0u;
    for (uint32_t i = 0; i < POSE_KP_COUNT; i++) {
      if (pp_output.pOutBuff[i].proba >= AI_POSE_PP_CONF_THRESHOLD) {
        float32_t x = pp_output.pOutBuff[i].x_center;
        float32_t y = pp_output.pOutBuff[i].y_center;
        if (x < x_min) x_min = x; 
        if (x > x_max) x_max = x;
        if (y < y_min) y_min = y; 
        if (y > y_max) y_max = y;
        n_spread++;
      }
    }
    if (n_spread >= 2u) {
      float32_t span_x = x_max - x_min;
      float32_t span_y = y_max - y_min;
      metrics->keypoints_too_spread = ((span_x > FALL_KP_SPREAD_MAX) || (span_y > FALL_KP_SPREAD_MAX)) ? 1u : 0u;
    }
  }
}
