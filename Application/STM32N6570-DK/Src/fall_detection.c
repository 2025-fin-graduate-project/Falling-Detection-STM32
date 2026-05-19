/**
 ******************************************************************************
 * @file    fall_detection.c
 * @brief   Fall detection application implementation.
 ******************************************************************************
 */

#include "fall_detection.h"
#include "gru_network.h"
#include "stm32n6xx_hal.h"
#include "stm32n6570_discovery_xspi.h"
#include "stm32n6570_discovery.h"
#include <string.h>
#include <stdio.h>

/* Fall detection model context */
STAI_NETWORK_CONTEXT_DECLARE(gru_network_context, STAI_GRU_NETWORK_CONTEXT_SIZE)

/* Activation buffer */
__attribute__((aligned(32))) static uint8_t gru_activation_buf[STAI_GRU_NETWORK_ACTIVATION_1_SIZE];

/* Input/Output pointers */
static stai_ptr gru_in[STAI_GRU_NETWORK_IN_NUM]   = {0};
static stai_ptr gru_out[STAI_GRU_NETWORK_OUT_NUM]  = {0};
static int32_t  gru_in_len[STAI_GRU_NETWORK_IN_NUM]  = {0};
static int32_t  gru_out_len[STAI_GRU_NETWORK_OUT_NUM] = {0};

/* Global state */
FallDetectionState_t fall_state;

void FallDetection_Init(void)
{
  int ret;
  memset(&fall_state, 0, sizeof(fall_state));

  stai_size number_input  = STAI_GRU_NETWORK_IN_NUM;
  stai_size number_output = STAI_GRU_NETWORK_OUT_NUM;

  ret = stai_gru_network_init(gru_network_context);
  if (ret != STAI_SUCCESS) {
      printf("Error: stai_gru_network_init failed (%d)\n", ret);
      return;
  }

  /* Connect input/output pointers into the activation buffer */
  stai_ptr act_bufs[STAI_GRU_NETWORK_ACTIVATIONS_NUM] = { (stai_ptr)gru_activation_buf };
  stai_size n_act = STAI_GRU_NETWORK_ACTIVATIONS_NUM;
  ret = stai_gru_network_set_activations(gru_network_context, act_bufs, n_act);
  if (ret != STAI_SUCCESS) return;

  ret = stai_gru_network_get_inputs(gru_network_context, gru_in, &number_input);
  if (ret != STAI_SUCCESS) return;

  ret = stai_gru_network_get_outputs(gru_network_context, gru_out, &number_output);
  if (ret != STAI_SUCCESS) return;

  gru_in_len[0]  = STAI_GRU_NETWORK_IN_1_SIZE_BYTES;
  gru_out_len[0] = STAI_GRU_NETWORK_OUT_1_SIZE_BYTES;

  BSP_LED_Init(LED_RED);
  BSP_LED_Off(LED_RED);
}

void Alarm_Update(void)
{
  uint8_t latch_active = (fall_state.fall_latch_tick != 0u) &&
                         ((HAL_GetTick() - fall_state.fall_latch_tick) < GRU_FALL_LATCH_MS);

  if (latch_active)
  {
    /* Blink LED_RED at ~2Hz while alarm is active */
    if ((HAL_GetTick() % (2u * GRU_ALARM_BLINK_PERIOD_MS)) < GRU_ALARM_BLINK_PERIOD_MS)
      BSP_LED_On(LED_RED);
    else
      BSP_LED_Off(LED_RED);
  }
  else
  {
    BSP_LED_Off(LED_RED);
  }
}


void FallDetection_RunInference(void)
{
  /* OctoSPI2 Fix: ll_aton leaves it in indirect mode. Force to memory-mapped. */
  {
    extern XSPI_HandleTypeDef hxspi_nor[];
    CLEAR_BIT(hxspi_nor[0].Instance->CR, XSPI_CR_DMAEN);
    if (HAL_XSPI_GET_FLAG(&hxspi_nor[0], HAL_XSPI_FLAG_BUSY))
    {
      SET_BIT(hxspi_nor[0].Instance->CR, XSPI_CR_ABORT);
      uint32_t t0 = HAL_GetTick();
      while (!HAL_XSPI_GET_FLAG(&hxspi_nor[0], HAL_XSPI_FLAG_TC) && (HAL_GetTick() - t0) < 100U);
      HAL_XSPI_CLEAR_FLAG(&hxspi_nor[0], HAL_XSPI_FLAG_TC);
    }
    hxspi_nor[0].State = HAL_XSPI_STATE_READY;
  }
  BSP_XSPI_NOR_EnableMemoryMappedMode(0);

  /* Invalidate D-Cache for the weight region to avoid stale data from indirect mode period */
  extern const uint64_t g_gru_network_weights_array[];
  SCB_InvalidateDCache_by_Addr((void*)g_gru_network_weights_array, STAI_GRU_NETWORK_WEIGHTS_SIZE_BYTES);

  stai_gru_network_run(gru_network_context, STAI_MODE_SYNC);
}

void FallDetection_Update(PosePipeline_t *pipeline)
{
  fall_state.frame_count = pipeline->win_count;
  fall_state.window_ready = PosePipeline_WindowFull(pipeline);

  if (!fall_state.window_ready)
  {
    fall_state.fall_detected = 0;
    fall_state.fall_score    = 0.0f;
    fall_state.normal_score  = 0.0f;
    return;
  }

  /* IN[0]: time-first window [40][45] */
  PosePipeline_GetWindowTimeFirst(pipeline, (float32_t (*)[POSE_FEATURE_COUNT])gru_in[0]);

  uint32_t t0 = HAL_GetTick();
  FallDetection_RunInference();
  fall_state.inference_ms = HAL_GetTick() - t0;

  float32_t *logits = (float32_t *)gru_out[0];

  fall_state.normal_logit = logits[0];
  fall_state.fall_logit   = logits[1];
  fall_state.normal_score = logits[0];
  fall_state.fall_score   = logits[1];

  fall_state.fall_detected = (fall_state.fall_score >= GRU_FALL_SCORE_THRESHOLD) ? 1u : 0u;

  /* Logging (1Hz) */
  static uint32_t last_log = 0;
  if (HAL_GetTick() - last_log >= 1000u)
  {
      printf("[GRU] %lums %s Fall %.2f Normal %.2f\r\n",
             fall_state.inference_ms,
             fall_state.fall_detected ? "FALL" : "NORMAL",
             (double)fall_state.fall_score,
             (double)fall_state.normal_score);
      last_log = HAL_GetTick();
  }

  /* Majority Vote Logic */
  fall_state.fall_vote_buf[fall_state.fall_vote_idx % GRU_FALL_VOTE_WINDOW] = fall_state.fall_detected;
  fall_state.fall_vote_idx++;

  uint32_t votes = 0;
  for (int i = 0; i < GRU_FALL_VOTE_WINDOW; i++) votes += fall_state.fall_vote_buf[i];

  if (votes >= GRU_FALL_VOTE_K && fall_state.fall_vote_idx >= GRU_FALL_VOTE_WINDOW)
  {
    printf("[ALARM] FALL CONFIRMED (%lu/%d votes) - resetting pipeline\r\n", votes, GRU_FALL_VOTE_WINDOW);
    fall_state.fall_latch_tick = HAL_GetTick();
    PosePipeline_Init(pipeline);
    memset(fall_state.fall_vote_buf, 0, sizeof(fall_state.fall_vote_buf));
    fall_state.fall_vote_idx = 0;
    fall_state.frame_count    = 0;
    fall_state.window_ready   = 0;
  }
}

void FallDetection_Invalidate(PosePipeline_t *pipeline)
{
  if (pipeline != NULL)
  {
    pipeline->win_count = 0u;
    pipeline->win_head  = 0u;
    pipeline->deriv_initialized = 0u;
  }

  fall_state.window_ready = 0u;
  fall_state.fall_detected = 0u;
  fall_state.frame_count = 0u;
  fall_state.inference_ms = 0u;
  fall_state.normal_logit = 0.0f;
  fall_state.fall_logit = 0.0f;
  fall_state.normal_score = 0.0f;
  fall_state.fall_score = 0.0f;
  memset(fall_state.fall_vote_buf, 0, sizeof(fall_state.fall_vote_buf));
  fall_state.fall_vote_idx = 0u;
}
