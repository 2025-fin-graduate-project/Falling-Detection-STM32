/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-03-18T23:54:33+0900
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
#ifndef STAI_NETWORK_DETAILS_H
#define STAI_NETWORK_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_network_details = {
  .tensors = (const stai_tensor[30]) {
   { .size_bytes = 13200, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 55, 60}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "input_output" },
   { .size_bytes = 13200, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 60, 55}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "input_Transpose_output" },
   { .size_bytes = 15360, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 60, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv1d_2_output" },
   { .size_bytes = 15360, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 60, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv1d_output" },
   { .size_bytes = 15360, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 60, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "relu_output" },
   { .size_bytes = 15360, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 60, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv1d_1_output" },
   { .size_bytes = 15360, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 60, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "relu_1_output" },
   { .size_bytes = 15360, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 60, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "add_output" },
   { .size_bytes = 15360, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 60, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "relu_2_output" },
   { .size_bytes = 15872, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 62, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv1d_3_output" },
   { .size_bytes = 15872, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 62, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "relu_3_output" },
   { .size_bytes = 16384, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 64, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv1d_4_output" },
   { .size_bytes = 16384, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 64, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "relu_4_output" },
   { .size_bytes = 16384, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 64, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "add_1_output" },
   { .size_bytes = 16384, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 64, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "relu_5_output" },
   { .size_bytes = 32768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 64, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv1d_7_output" },
   { .size_bytes = 35840, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 70, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv1d_5_output" },
   { .size_bytes = 35840, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 70, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "relu_6_output" },
   { .size_bytes = 38912, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 76, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv1d_6_output" },
   { .size_bytes = 38912, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 76, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "relu_7_output" },
   { .size_bytes = 38912, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 76, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "add_2_output" },
   { .size_bytes = 38912, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 76, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "relu_8_output" },
   { .size_bytes = 608, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 76, 2}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv1d_10_output" },
   { .size_bytes = 720, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 90, 2}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv1d_8_output" },
   { .size_bytes = 720, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 90, 2}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "relu_9_output" },
   { .size_bytes = 832, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 104, 2}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "conv1d_9_output" },
   { .size_bytes = 832, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 104, 2}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "relu_10_output" },
   { .size_bytes = 832, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 104, 2}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "add_3_output" },
   { .size_bytes = 832, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 104, 2}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "relu_11_output" },
   { .size_bytes = 8, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 2}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "output_output" }
  },
  .nodes = (const stai_node_details[29]){
    {.id = 2, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* input_Transpose */
    {.id = 5, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* conv1d_2 */
    {.id = 1, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* conv1d */
    {.id = 2, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* relu */
    {.id = 3, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* conv1d_1 */
    {.id = 4, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* relu_1 */
    {.id = 6, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){6, 2}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* add */
    {.id = 7, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){7}}, .output_tensors = {1, (const int32_t[1]){8}} }, /* relu_2 */
    {.id = 8, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){8}}, .output_tensors = {1, (const int32_t[1]){9}} }, /* conv1d_3 */
    {.id = 9, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){9}}, .output_tensors = {1, (const int32_t[1]){10}} }, /* relu_3 */
    {.id = 10, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){10}}, .output_tensors = {1, (const int32_t[1]){11}} }, /* conv1d_4 */
    {.id = 11, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){11}}, .output_tensors = {1, (const int32_t[1]){12}} }, /* relu_4 */
    {.id = 12, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){12, 8}}, .output_tensors = {1, (const int32_t[1]){13}} }, /* add_1 */
    {.id = 13, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){13}}, .output_tensors = {1, (const int32_t[1]){14}} }, /* relu_5 */
    {.id = 18, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){14}}, .output_tensors = {1, (const int32_t[1]){15}} }, /* conv1d_7 */
    {.id = 14, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){14}}, .output_tensors = {1, (const int32_t[1]){16}} }, /* conv1d_5 */
    {.id = 15, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){16}}, .output_tensors = {1, (const int32_t[1]){17}} }, /* relu_6 */
    {.id = 16, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){17}}, .output_tensors = {1, (const int32_t[1]){18}} }, /* conv1d_6 */
    {.id = 17, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){18}}, .output_tensors = {1, (const int32_t[1]){19}} }, /* relu_7 */
    {.id = 19, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){19, 15}}, .output_tensors = {1, (const int32_t[1]){20}} }, /* add_2 */
    {.id = 20, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){20}}, .output_tensors = {1, (const int32_t[1]){21}} }, /* relu_8 */
    {.id = 25, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){21}}, .output_tensors = {1, (const int32_t[1]){22}} }, /* conv1d_10 */
    {.id = 21, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){21}}, .output_tensors = {1, (const int32_t[1]){23}} }, /* conv1d_8 */
    {.id = 22, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){23}}, .output_tensors = {1, (const int32_t[1]){24}} }, /* relu_9 */
    {.id = 23, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){24}}, .output_tensors = {1, (const int32_t[1]){25}} }, /* conv1d_9 */
    {.id = 24, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){25}}, .output_tensors = {1, (const int32_t[1]){26}} }, /* relu_10 */
    {.id = 26, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){26, 22}}, .output_tensors = {1, (const int32_t[1]){27}} }, /* add_3 */
    {.id = 27, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){27}}, .output_tensors = {1, (const int32_t[1]){28}} }, /* relu_11 */
    {.id = 28, .type = AI_LAYER_GATHER_TYPE, .input_tensors = {1, (const int32_t[1]){28}}, .output_tensors = {1, (const int32_t[1]){29}} } /* output */
  },
  .n_nodes = 29
};
#endif

