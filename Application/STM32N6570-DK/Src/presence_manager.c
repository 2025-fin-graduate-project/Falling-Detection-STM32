/**
 ******************************************************************************
 * @file    presence_manager.c
 * @brief   Presence manager implementation for person detection state machine.
 ******************************************************************************
 */

#include "presence_manager.h"
#include "app_config.h"
#include <string.h>

static PresenceStatus_t presence_state;

void Presence_Init(void)
{
  memset(&presence_state, 0, sizeof(PresenceStatus_t));
  presence_state.person_missing_confirmed = 1u;
}

uint8_t Presence_Update(VisionMetrics_t *v_metrics)
{
  if (v_metrics->postprocess_valid)
  {
    if (v_metrics->person_present)
    {
      /* Entry logic: Need consecutive frames to confirm */
      presence_state.person_missing_count = 0u;
      if (!presence_state.person_present_confirmed)
      {
        presence_state.person_entry_count++;
        if (presence_state.person_entry_count >= FALL_PERSON_ENTRY_CONF_COUNT)
        {
          presence_state.person_present_confirmed = 1u;
          presence_state.person_missing_confirmed = 0u;
        }
      }
    }
    else
    {
      /* Exit logic: Need consecutive frames to confirm missing */
      presence_state.person_entry_count = 0u;
      if (presence_state.person_present_confirmed)
      {
        presence_state.person_missing_count++;
        if (presence_state.person_missing_count >= FALL_PERSON_MISSING_RESET_COUNT)
        {
          presence_state.person_present_confirmed = 0u;
          presence_state.person_missing_confirmed = 1u;
        }
      }
      else
      {
        presence_state.person_missing_confirmed = 1u;
      }
    }
  }

  /* Consolidated presence flag */
  return (v_metrics->postprocess_valid && presence_state.person_present_confirmed);
}

const PresenceStatus_t* Presence_GetStatus(void)
{
  return &presence_state;
}
