/**
 ******************************************************************************
 * @file    ui_system.h
 * @brief   UI system module for LCD rendering and alarm visualization.
 ******************************************************************************
 */

#ifndef UI_SYSTEM_H
#define UI_SYSTEM_H

#include <stdint.h>
#include "vision_system.h"
#include "presence_manager.h"
#include "fall_detection.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Initialize UI System (LCD, Layers, Fonts, Bindings)
 * @param  bg_width Background area width (from vision system)
 * @param  bg_height Background area height (from vision system)
 */
void UI_Init(uint32_t bg_width, uint32_t bg_height);

/**
 * @brief  Get the background buffer address for camera DMA
 * @return Address of the background framebuffer
 */
uint8_t* UI_GetBgBuffer(void);

/**
 * @brief  Update display with latest inference results and system state
 * @param  pp_out Postprocessing output (skeleton)
 * @param  v_metrics Vision performance and debug metrics
 * @param  p_status Presence state machine status
 * @param  f_state Fall detection state
 */
void UI_Update(spe_pp_out_t *pp_out, 
               VisionMetrics_t *v_metrics, 
               const PresenceStatus_t *p_status,
               const FallDetectionState_t *f_state);

#ifdef __cplusplus
}
#endif

#endif /* UI_SYSTEM_H */
