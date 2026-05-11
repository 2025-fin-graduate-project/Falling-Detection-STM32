/**
  ******************************************************************************
  * @file    gru_network.h
  * @date    2026-05-10T22:50:57+0900
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
  .tensors = (const stai_tensor[47]) {
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "serving_default_gru1_h_in0_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "serving_default_gru2_h_in0_output" },
   { .size_bytes = 220, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 55}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "serving_default_pose_sequence0_output" },
   { .size_bytes = 220, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 55}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "unpack_9_output0" },
   { .size_bytes = 768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "gemm_10_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "split_11_output0" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "split_11_output1" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "split_11_output2" },
   { .size_bytes = 384, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 96}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "gemm_4_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "slice_7_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "slice_6_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "slice_5_output" },
   { .size_bytes = 768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 192}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "gemm_0_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "slice_3_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "slice_2_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_16_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "nl_17_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_18_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_19_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "nl_20_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "slice_1_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_12_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "nl_13_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_15_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_21_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_14_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_22_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "unpack_24_output0" },
   { .size_bytes = 384, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 96}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "gemm_25_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "split_26_output0" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "split_26_output1" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "split_26_output2" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_31_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "nl_32_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_33_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_34_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "nl_35_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_27_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "nl_28_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_30_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_36_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_29_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "eltwise_37_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "gemm_38_output" },
   { .size_bytes = 128, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "nl_38_nl_output" },
   { .size_bytes = 8, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 2}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "gemm_39_output" },
   { .size_bytes = 8, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 2}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "nl_40_output" }
  },
  .nodes = (const stai_node_details[40]){
    {.id = 9, .type = AI_LAYER_UNPACK_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* unpack_9 */
    {.id = 10, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* gemm_10 */
    {.id = 11, .type = AI_LAYER_SPLIT_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {3, (const int32_t[3]){5, 6, 7}} }, /* split_11 */
    {.id = 4, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){8}} }, /* gemm_4 */
    {.id = 7, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){8}}, .output_tensors = {1, (const int32_t[1]){9}} }, /* slice_7 */
    {.id = 6, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){8}}, .output_tensors = {1, (const int32_t[1]){10}} }, /* slice_6 */
    {.id = 5, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){8}}, .output_tensors = {1, (const int32_t[1]){11}} }, /* slice_5 */
    {.id = 0, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){12}} }, /* gemm_0 */
    {.id = 3, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){12}}, .output_tensors = {1, (const int32_t[1]){13}} }, /* slice_3 */
    {.id = 2, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){12}}, .output_tensors = {1, (const int32_t[1]){14}} }, /* slice_2 */
    {.id = 16, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){6, 14}}, .output_tensors = {1, (const int32_t[1]){15}} }, /* eltwise_16 */
    {.id = 17, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){15}}, .output_tensors = {1, (const int32_t[1]){16}} }, /* nl_17 */
    {.id = 18, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){16, 13}}, .output_tensors = {1, (const int32_t[1]){17}} }, /* eltwise_18 */
    {.id = 19, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){7, 17}}, .output_tensors = {1, (const int32_t[1]){18}} }, /* eltwise_19 */
    {.id = 20, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){18}}, .output_tensors = {1, (const int32_t[1]){19}} }, /* nl_20 */
    {.id = 1, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){12}}, .output_tensors = {1, (const int32_t[1]){20}} }, /* slice_1 */
    {.id = 12, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){5, 20}}, .output_tensors = {1, (const int32_t[1]){21}} }, /* eltwise_12 */
    {.id = 13, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){21}}, .output_tensors = {1, (const int32_t[1]){22}} }, /* nl_13 */
    {.id = 15, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {1, (const int32_t[1]){22}}, .output_tensors = {1, (const int32_t[1]){23}} }, /* eltwise_15 */
    {.id = 21, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){23, 19}}, .output_tensors = {1, (const int32_t[1]){24}} }, /* eltwise_21 */
    {.id = 14, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){22, 0}}, .output_tensors = {1, (const int32_t[1]){25}} }, /* eltwise_14 */
    {.id = 22, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){25, 24}}, .output_tensors = {1, (const int32_t[1]){26}} }, /* eltwise_22 */
    {.id = 24, .type = AI_LAYER_UNPACK_TYPE, .input_tensors = {1, (const int32_t[1]){26}}, .output_tensors = {1, (const int32_t[1]){27}} }, /* unpack_24 */
    {.id = 25, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){27}}, .output_tensors = {1, (const int32_t[1]){28}} }, /* gemm_25 */
    {.id = 26, .type = AI_LAYER_SPLIT_TYPE, .input_tensors = {1, (const int32_t[1]){28}}, .output_tensors = {3, (const int32_t[3]){29, 30, 31}} }, /* split_26 */
    {.id = 31, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){30, 10}}, .output_tensors = {1, (const int32_t[1]){32}} }, /* eltwise_31 */
    {.id = 32, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){32}}, .output_tensors = {1, (const int32_t[1]){33}} }, /* nl_32 */
    {.id = 33, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){33, 9}}, .output_tensors = {1, (const int32_t[1]){34}} }, /* eltwise_33 */
    {.id = 34, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){31, 34}}, .output_tensors = {1, (const int32_t[1]){35}} }, /* eltwise_34 */
    {.id = 35, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){35}}, .output_tensors = {1, (const int32_t[1]){36}} }, /* nl_35 */
    {.id = 27, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){29, 11}}, .output_tensors = {1, (const int32_t[1]){37}} }, /* eltwise_27 */
    {.id = 28, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){37}}, .output_tensors = {1, (const int32_t[1]){38}} }, /* nl_28 */
    {.id = 30, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {1, (const int32_t[1]){38}}, .output_tensors = {1, (const int32_t[1]){39}} }, /* eltwise_30 */
    {.id = 36, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){39, 36}}, .output_tensors = {1, (const int32_t[1]){40}} }, /* eltwise_36 */
    {.id = 29, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){38, 1}}, .output_tensors = {1, (const int32_t[1]){41}} }, /* eltwise_29 */
    {.id = 37, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){41, 40}}, .output_tensors = {1, (const int32_t[1]){42}} }, /* eltwise_37 */
    {.id = 38, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){42}}, .output_tensors = {1, (const int32_t[1]){43}} }, /* gemm_38 */
    {.id = 38, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){43}}, .output_tensors = {1, (const int32_t[1]){44}} }, /* nl_38_nl */
    {.id = 39, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){44}}, .output_tensors = {1, (const int32_t[1]){45}} }, /* gemm_39 */
    {.id = 40, .type = AI_LAYER_SM_TYPE, .input_tensors = {1, (const int32_t[1]){45}}, .output_tensors = {1, (const int32_t[1]){46}} } /* nl_40 */
  },
  .n_nodes = 40
};
#endif

