/**
  ******************************************************************************
  * @file    gru_network_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-05-18T02:32:06+0900
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef GRU_NETWORK_DATA_PARAMS_H
#define GRU_NETWORK_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_GRU_NETWORK_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_gru_network_data_weights_params[1]))
*/

#define AI_GRU_NETWORK_DATA_CONFIG               (NULL)


#define AI_GRU_NETWORK_DATA_ACTIVATIONS_SIZES \
  { 6916, }
#define AI_GRU_NETWORK_DATA_ACTIVATIONS_SIZE     (6916)
#define AI_GRU_NETWORK_DATA_ACTIVATIONS_COUNT    (1)
#define AI_GRU_NETWORK_DATA_ACTIVATION_1_SIZE    (6916)



#define AI_GRU_NETWORK_DATA_WEIGHTS_SIZES \
  { 549920, }
#define AI_GRU_NETWORK_DATA_WEIGHTS_SIZE         (549920)
#define AI_GRU_NETWORK_DATA_WEIGHTS_COUNT        (1)
#define AI_GRU_NETWORK_DATA_WEIGHT_1_SIZE        (549920)



#define AI_GRU_NETWORK_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_gru_network_activations_table[1])

extern ai_handle g_gru_network_activations_table[1 + 2];



#define AI_GRU_NETWORK_DATA_WEIGHTS_TABLE_GET() \
  (&g_gru_network_weights_table[1])

extern ai_handle g_gru_network_weights_table[1 + 2];


#endif    /* GRU_NETWORK_DATA_PARAMS_H */
