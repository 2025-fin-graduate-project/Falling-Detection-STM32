/**
  ******************************************************************************
  * @file    gru_network.h
  * @date    2026-05-17T02:21:08+0900
  * @brief   ST.AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */
#ifndef STAI_GRU_NETWORK_DETAILS_H
#define STAI_GRU_NETWORK_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_gru_network_details = {
  .tensors = (const stai_tensor[21]) {
   { .size_bytes = 4320, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 27}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "pose_sequence_output" },
   { .size_bytes = 10240, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv_pre_1_output" },
   { .size_bytes = 10240, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv_pre_relu_1_output" },
   { .size_bytes = 10240, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv_pre_2_output" },
   { .size_bytes = 10240, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv_pre_relu_2_output" },
   { .size_bytes = 20480, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "gru_1_output0" },
   { .size_bytes = 160, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "ln_1_Reduce_output" },
   { .size_bytes = 160, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "ln_1_Reduce_Mul_output" },
   { .size_bytes = 20480, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "ln_1_Sub_output" },
   { .size_bytes = 20480, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "ln_1_Mul_output" },
   { .size_bytes = 160, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "ln_1_Reduce_1_output" },
   { .size_bytes = 160, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "ln_1_Reduce_1_Mul_output" },
   { .size_bytes = 160, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "ln_1_Sqrt_output" },
   { .size_bytes = 160, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "ln_1_Reciprocal_output" },
   { .size_bytes = 20480, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "ln_1_Mul_1_output" },
   { .size_bytes = 20480, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 40, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "ln_1_Mul_2_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "gru_2_output0" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "head_dense_dense_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "head_dense_output" },
   { .size_bytes = 8, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 2}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "classifier_dense_output" },
   { .size_bytes = 8, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 2}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "classifier_output" }
  },
  .nodes = (const stai_node_details[20]){
    {.id = 2, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* conv_pre_1 */
    {.id = 3, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* conv_pre_relu_1 */
    {.id = 5, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* conv_pre_2 */
    {.id = 6, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* conv_pre_relu_2 */
    {.id = 8, .type = AI_LAYER_GRU_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* gru_1 */
    {.id = 9, .type = AI_LAYER_REDUCE_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* ln_1_Reduce */
    {.id = 9, .type = AI_LAYER_BN_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* ln_1_Reduce_Mul */
    {.id = 9, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){5, 7}}, .output_tensors = {1, (const int32_t[1]){8}} }, /* ln_1_Sub */
    {.id = 9, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){8, 8}}, .output_tensors = {1, (const int32_t[1]){9}} }, /* ln_1_Mul */
    {.id = 9, .type = AI_LAYER_REDUCE_TYPE, .input_tensors = {1, (const int32_t[1]){9}}, .output_tensors = {1, (const int32_t[1]){10}} }, /* ln_1_Reduce_1 */
    {.id = 9, .type = AI_LAYER_BN_TYPE, .input_tensors = {1, (const int32_t[1]){10}}, .output_tensors = {1, (const int32_t[1]){11}} }, /* ln_1_Reduce_1_Mul */
    {.id = 9, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){11}}, .output_tensors = {1, (const int32_t[1]){12}} }, /* ln_1_Sqrt */
    {.id = 9, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){12}}, .output_tensors = {1, (const int32_t[1]){13}} }, /* ln_1_Reciprocal */
    {.id = 9, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){8, 13}}, .output_tensors = {1, (const int32_t[1]){14}} }, /* ln_1_Mul_1 */
    {.id = 9, .type = AI_LAYER_BN_TYPE, .input_tensors = {1, (const int32_t[1]){14}}, .output_tensors = {1, (const int32_t[1]){15}} }, /* ln_1_Mul_2 */
    {.id = 10, .type = AI_LAYER_GRU_TYPE, .input_tensors = {1, (const int32_t[1]){15}}, .output_tensors = {1, (const int32_t[1]){16}} }, /* gru_2 */
    {.id = 11, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){16}}, .output_tensors = {1, (const int32_t[1]){17}} }, /* head_dense_dense */
    {.id = 11, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){17}}, .output_tensors = {1, (const int32_t[1]){18}} }, /* head_dense */
    {.id = 13, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){18}}, .output_tensors = {1, (const int32_t[1]){19}} }, /* classifier_dense */
    {.id = 13, .type = AI_LAYER_SM_TYPE, .input_tensors = {1, (const int32_t[1]){19}}, .output_tensors = {1, (const int32_t[1]){20}} } /* classifier */
  },
  .n_nodes = 20
};
#endif

