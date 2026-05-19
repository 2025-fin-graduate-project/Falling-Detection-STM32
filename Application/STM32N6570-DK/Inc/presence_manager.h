/**
 ******************************************************************************
 * @file    presence_manager.h
 * @brief   Presence manager module for person detection state machine.
 ******************************************************************************
 */

#ifndef PRESENCE_MANAGER_H
#define PRESENCE_MANAGER_H

#include <stdint.h>
#include "vision_system.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint32_t person_missing_count;
  uint32_t person_entry_count;
  uint8_t person_present_confirmed;
  uint8_t person_missing_confirmed;
} PresenceStatus_t;

/**
 * @brief  Initialize Presence Manager
 */
void Presence_Init(void);

/**
 * @brief  Update presence state based on vision metrics
 * @param  v_metrics Pointer to vision metrics from the latest frame
 * @return 1 if person is confirmed present and pose is valid for processing, 0 otherwise
 */
uint8_t Presence_Update(VisionMetrics_t *v_metrics);

/**
 * @brief  Get current presence status for UI/Logging
 * @return Pointer to current presence status
 */
const PresenceStatus_t* Presence_GetStatus(void);

#ifdef __cplusplus
}
#endif

#endif /* PRESENCE_MANAGER_H */
