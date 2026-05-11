/**
  ******************************************************************************
  * @file    gru_network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-05-10T22:50:57+0900
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
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

#include "ai_lite_inspect.h"
#include "ai_platform_interface.h"
#include "layers.h"
#include "core_convert.h"
#include "gru_network.h"
#include "gru_network_details.h"
#include "gru_network_data.h"
#include "stai_events.h"

#include "ai_lite_inspect.h"

#include "lite_operators.h"
/*****************************************************************************/
#define STAI_INTERNAL_API_MAJOR               (1)
#define STAI_INTERNAL_API_MINOR               (0)
#define STAI_INTERNAL_API_MICRO               (0)

#define STAI_MAGIC                            (0xB1C00100)

/*****************************************************************************/
#define _STAI_CONCAT_ARG(a, b)     a ## b
#define STAI_CONCAT(a, b)         _STAI_CONCAT_ARG(a, b)

/*!  STAI_CAST SECTION                       *********************************/
#define STAI_CAST(type, expr) \
  ((type)(expr))


/*****************************************************************************/
#define STAI_SIZE(_size) \
  ((stai_size)(_size))

/*****************************************************************************/
#define STAI_INIT_BUFFER(_flags, _size, _address) \
  { \
    .size = (_size), \
    .address = (uintptr_t)(_address), \
    .flags = (_flags), \
  }

#define STAI_INIT_TENSOR(_name, _flags, _fmt, _size_bytes, _shape, _scale, _zeropoint) \
  { \
    .size_bytes = (_size_bytes), \
    .flags = (_flags), \
    .format = (stai_format)(_fmt), \
    .shape = STAI_PACK(_shape), \
    .scale = STAI_PACK(_scale), \
    .zeropoint = STAI_PACK(_zeropoint), \
    .name = (_name) \
  }

#define STAI_INIT_ARRAY(_size, _ptr) \
  { .size = STAI_SIZE(_size), .data = STAI_PACK(_ptr) }


#define STAI_CAST_ARRAY(_type, _size, _ptr) \
  { .size = STAI_SIZE(_size), .data = (_type)STAI_PACK(_ptr) }


#define STAI_DECLARE_ARRAY(_type, _size, ...) \
  { .size = STAI_SIZE(_size), .data = (_type[_size]) { STAI_PACK(__VA_ARGS__) } }


#define STAI_EMPTY_ARRAY() \
  { .size = 0, .data = NULL }


#define STAI_INIT_VERSION(_major, _minor, _micro) \
  { .major = (_major), .minor = (_minor), .micro = (_micro), .reserved = 0x0 }

/*****************************************************************************/
/**  Getters and setters  **/

#define STAI_GET_ARRAY_SIZE(nd_array) \
  (nd_array.size)


#define STAI_GET_ARRAY_ELEM(nd_array, pos) \
  (nd_array.data[(pos)])

#define _STAI_SET_ERROR(net_ctx, cond, value, exit) { \
  if (!(net_ctx)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE; } \
  if (((uintptr_t)net_ctx) & (_STAI_CONTEXT_ALIGNMENT-1)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_ALIGNMENT; } \
  if (((value) >= STAI_ERROR_GENERIC) && (cond)) { \
    if ((net_ctx)->_return_code == STAI_SUCCESS) { \
      (net_ctx)->_return_code = (value); \
    } \
    return (exit); \
  } \
}

/*****************************************************************************/
/* TODO REMOVE THESE TWO MACROS */
#define STAI_EVENT_NODE_START_CB
#define STAI_EVENT_NODE_STOP_CB

#ifdef STAI_EVENT_NODE_START_CB
#ifndef _STAI_GRU_NETWORK_EVENT_NODE_START_CB
  #define _STAI_GRU_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _start_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(const stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_START, (const void*)&_start_event); \
  }
#endif
#else
  #define _STAI_GRU_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_GRU_NETWORK_EVENT_NODE_START_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_START_CB */

#ifdef STAI_EVENT_NODE_STOP_CB
#ifndef _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB
  #define _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _stop_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_STOP, (const void*)&_stop_event); \
  }
#endif
#else
  #define _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_STOP_CB */


/*****************************************************************************/
#define _STAI_GRU_NETWORK_MODEL_SIGNATURE     "0xf1a6e22d7d8396b7b23684c598ecbb64"
#define _STAI_GRU_NETWORK_DATETIME            "2026-05-10T22:50:57+0900"
#define _STAI_GRU_NETWORK_COMPILE_DATETIME    __DATE__ " " __TIME__

#define _STAI_CONTEXT_ALIGNMENT        STAI_GRU_NETWORK_CONTEXT_ALIGNMENT

/*****************************************************************************/
#define g_gru_network_activations_1     (NULL)




#if defined(HAVE_GRU_NETWORK_INFO)
/*****************************************************************************/
static const stai_network_info g_gru_network_info = {
  .model_signature = _STAI_GRU_NETWORK_MODEL_SIGNATURE,
  .c_compile_datetime = _STAI_GRU_NETWORK_COMPILE_DATETIME,
  .c_model_name = STAI_GRU_NETWORK_MODEL_NAME,
  .c_model_datetime = _STAI_GRU_NETWORK_DATETIME,
  .c_model_signature = 0x0,
  .runtime_version = STAI_INIT_VERSION(11, 0, 0),
  .tool_version = STAI_INIT_VERSION(3, 0, 0),
  .api_version = STAI_INIT_VERSION(1, 0, 0),
  .n_macc = STAI_GRU_NETWORK_MACC_NUM,
  .n_nodes = STAI_GRU_NETWORK_NODES_NUM,
  .flags = STAI_GRU_NETWORK_FLAGS,
  .n_inputs = STAI_GRU_NETWORK_IN_NUM,
  .n_outputs = STAI_GRU_NETWORK_OUT_NUM,
  .n_activations = STAI_GRU_NETWORK_ACTIVATIONS_NUM,
  .n_weights = STAI_GRU_NETWORK_WEIGHTS_NUM,
  .n_states = STAI_GRU_NETWORK_STATES_NUM,
  .inputs = (stai_tensor[STAI_GRU_NETWORK_IN_NUM]) {
    STAI_INIT_TENSOR(
      STAI_GRU_NETWORK_IN_1_NAME,
      STAI_GRU_NETWORK_IN_1_FLAGS,
      STAI_GRU_NETWORK_IN_1_FORMAT,
      STAI_GRU_NETWORK_IN_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 2, 1, 64),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      STAI_GRU_NETWORK_IN_2_NAME,
      STAI_GRU_NETWORK_IN_2_FLAGS,
      STAI_GRU_NETWORK_IN_2_FORMAT,
      STAI_GRU_NETWORK_IN_2_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 2, 1, 32),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      STAI_GRU_NETWORK_IN_3_NAME,
      STAI_GRU_NETWORK_IN_3_FLAGS,
      STAI_GRU_NETWORK_IN_3_FORMAT,
      STAI_GRU_NETWORK_IN_3_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 3, 1, 1, 55),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
    .outputs = (stai_tensor[STAI_GRU_NETWORK_OUT_NUM]) {
    STAI_INIT_TENSOR(
      STAI_GRU_NETWORK_OUT_1_NAME,
      STAI_GRU_NETWORK_OUT_1_FLAGS,
      STAI_GRU_NETWORK_OUT_1_FORMAT,
      STAI_GRU_NETWORK_OUT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 2, 1, 64),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      STAI_GRU_NETWORK_OUT_2_NAME,
      STAI_GRU_NETWORK_OUT_2_FLAGS,
      STAI_GRU_NETWORK_OUT_2_FORMAT,
      STAI_GRU_NETWORK_OUT_2_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 2, 1, 2),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      STAI_GRU_NETWORK_OUT_3_NAME,
      STAI_GRU_NETWORK_OUT_3_FLAGS,
      STAI_GRU_NETWORK_OUT_3_FORMAT,
      STAI_GRU_NETWORK_OUT_3_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 2, 1, 32),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .activations = (stai_tensor[STAI_GRU_NETWORK_ACTIVATIONS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_GRU_NETWORK_ACTIVATION_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_GRU_NETWORK_ACTIVATION_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 2816),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .weights = (stai_tensor[STAI_GRU_NETWORK_WEIGHTS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_GRU_NETWORK_WEIGHT_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_GRU_NETWORK_WEIGHT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 135060),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },

  .states = NULL
};
#endif

#define _STAI_CONTEXT_ACQUIRE(_net_ctx, _net_handle) \
  _stai_gru_network_context* _net_ctx = (_stai_gru_network_context*)(_net_handle); \
  STAI_ASSERT(_net_ctx != NULL) \
  _STAI_SET_ERROR(_net_ctx, _net_ctx->_magic != STAI_MAGIC, \
                  STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE, _net_ctx->_return_code)


/*****************************************************************************/
static
void _stai_gru_network_check(_stai_gru_network_context* net_ctx)
{
  stai_size idx;

// Check activations status
  for (idx=0; idx<STAI_GRU_NETWORK_ACTIVATIONS_NUM; idx++) {
    if (net_ctx->_activations[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_GRU_NETWORK_ACTIVATIONS_NUM) ? STAI_FLAG_ACTIVATIONS : STAI_FLAG_NONE;
// Check inputs status
  for (idx=0; idx<STAI_GRU_NETWORK_IN_NUM; idx++) {
    if (net_ctx->_inputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_GRU_NETWORK_IN_NUM) ? STAI_FLAG_INPUTS : STAI_FLAG_NONE;

  // Check outputs status
  for (idx=0; idx<STAI_GRU_NETWORK_OUT_NUM; idx++) {
    if (net_ctx->_outputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_GRU_NETWORK_OUT_NUM) ? STAI_FLAG_OUTPUTS : STAI_FLAG_NONE;

// Check weights status
  for (idx=0; idx<STAI_GRU_NETWORK_WEIGHTS_NUM; idx++) {
    if (net_ctx->_weights[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_GRU_NETWORK_WEIGHTS_NUM) ? STAI_FLAG_WEIGHTS : STAI_FLAG_NONE;
STAI_PRINT("  [_stai_network_check] flags: 0x%08x\n", net_ctx->_flags)
}


/*****************************************************************************/
STAI_API_ENTRY
stai_return_code stai_gru_network_init(
  stai_network* network)
{
  /* Memory where to store internal context is provided by applications as a raw byte buffer */
  _stai_gru_network_context* net_ctx = (_stai_gru_network_context*)(network);
  net_ctx->_return_code = STAI_SUCCESS;
  STAI_PRINT("[Entering Network Init] network(%p) context_size(%d)\n", net_ctx, (int32_t)sizeof(_stai_gru_network_context))

  _STAI_SET_ERROR(net_ctx, STAI_GRU_NETWORK_CONTEXT_SIZE != sizeof(_stai_gru_network_context),
                 STAI_ERROR_NETWORK_INVALID_CONTEXT_SIZE, net_ctx->_return_code)

  {
    const _stai_gru_network_context _gru_network_context = {
      ._magic = STAI_MAGIC,
      ._signature = STAI_GRU_NETWORK_MODEL_SIGNATURE,
      ._flags = STAI_GRU_NETWORK_FLAGS,
      ._return_code = STAI_SUCCESS,
      ._callback = NULL,
      ._callback_cookie = NULL,
      ._activations = {
      (stai_ptr)g_gru_network_activations_1
      },
      ._weights = {
      (stai_ptr)g_gru_network_weights_array
      },
      ._inputs = {
    NULL,NULL,NULL},
      ._outputs = {
    NULL,NULL,NULL},
    };

    // Deep copy of internal context to opaque buffer provided by app
    *net_ctx = _gru_network_context;

    _stai_gru_network_check(net_ctx);
  }

  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_gru_network_deinit(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /*  Reset flags to initial state  */
  net_ctx->_flags = STAI_GRU_NETWORK_FLAGS;
  return net_ctx->_return_code;
}

/*****************************************************************************/





/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_pose_sequence0_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 55, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  unpack_9_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 55, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  gemm_10_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  split_11_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  split_11_output1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  split_11_output2_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  split_11_num_or_size_splits_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  gemm_4_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 96, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  slice_7_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  slice_6_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  slice_5_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  gemm_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  slice_3_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  slice_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_16_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  nl_17_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_18_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_19_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  slice_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_12_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  nl_13_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_15_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  arith_constant5_2D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  nl_20_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_21_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_gru1_h_in0_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 64, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_14_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_22_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 64, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  unpack_24_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  gemm_25_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 96, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  split_26_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  split_26_output1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  split_26_output2_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  split_26_num_or_size_splits_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_31_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  nl_32_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_33_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_34_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_27_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  nl_28_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_30_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  nl_35_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_36_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_gru2_h_in0_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 32, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_29_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_37_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 32, AI_STATIC)



/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_pose_sequence0_output, AI_STATIC,
  45, 0x0,
  AI_SHAPE_INIT(4, 1, 55, 1, 1), AI_STRIDE_INIT(4, 4, 4, 220, 220),
  1, &serving_default_pose_sequence0_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  unpack_9_output0, AI_STATIC,
  61, 0x0,
  AI_SHAPE_INIT(4, 1, 55, 1, 1), AI_STRIDE_INIT(4, 4, 4, 220, 220),
  1, &unpack_9_output0_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  gemm_10_output, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 1, 1), AI_STRIDE_INIT(4, 4, 4, 768, 768),
  1, &gemm_10_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  split_11_num_or_size_splits, AI_STATIC,
  52, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &split_11_num_or_size_splits_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  split_11_output0, AI_STATIC,
  53, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &split_11_output0_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  split_11_output1, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &split_11_output1_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  split_11_output2, AI_STATIC,
  55, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &split_11_output2_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  gemm_4_output, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &gemm_4_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  slice_7_output, AI_STATIC,
  51, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &slice_7_output_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  slice_6_output, AI_STATIC,
  50, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &slice_6_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  slice_5_output, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &slice_5_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  gemm_0_output, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 1, 1), AI_STRIDE_INIT(4, 4, 4, 768, 768),
  1, &gemm_0_output_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  slice_3_output, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &slice_3_output_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  slice_2_output, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &slice_2_output_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_16_output, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &eltwise_16_output_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_18_output, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &eltwise_18_output_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  nl_17_output, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &nl_17_output_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_19_output, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &eltwise_19_output_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  slice_1_output, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &slice_1_output_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_12_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &eltwise_12_output_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  arith_constant5_2D, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &arith_constant5_2D_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_15_output, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &eltwise_15_output_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  nl_13_output, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &nl_13_output_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_21_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &eltwise_21_output_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  nl_20_output, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &nl_20_output_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_14_output, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &eltwise_14_output_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_gru1_h_in0_output, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &serving_default_gru1_h_in0_output_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_22_output, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &eltwise_22_output_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  unpack_24_output0, AI_STATIC,
  60, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &unpack_24_output0_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  gemm_25_output, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &gemm_25_output_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  split_26_num_or_size_splits, AI_STATIC,
  56, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &split_26_num_or_size_splits_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  split_26_output0, AI_STATIC,
  57, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &split_26_output0_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  split_26_output1, AI_STATIC,
  58, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &split_26_output1_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  split_26_output2, AI_STATIC,
  59, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &split_26_output2_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_31_output, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &eltwise_31_output_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_33_output, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &eltwise_33_output_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  nl_32_output, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &nl_32_output_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_34_output, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &eltwise_34_output_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_27_output, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &eltwise_27_output_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_30_output, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &eltwise_30_output_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  nl_28_output, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &nl_28_output_array, NULL)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_36_output, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &eltwise_36_output_array, NULL)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  nl_35_output, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &nl_35_output_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_29_output, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &eltwise_29_output_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_gru2_h_in0_output, AI_STATIC,
  44, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &serving_default_gru2_h_in0_output_array, NULL)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_37_output, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &eltwise_37_output_array, NULL)


AI_TENSOR_CHAIN_OBJ_DECLARE(
  unpack_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_pose_sequence0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_9_output0),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  unpack_9_layer, 9,
  UNPACK_TYPE, 0x0, NULL,
  unpack, forward_unpack,
  &unpack_9_chain,
  NULL, &unpack_9_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  split_11_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &split_11_output0, &split_11_output1, &split_11_output2),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &split_11_num_or_size_splits),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  split_11_layer, 11,
  SPLIT_TYPE, 0x0, NULL,
  split, forward_split,
  &split_11_chain,
  NULL, &split_11_layer, AI_STATIC, 
  .outer_elems = 1, 
  .outer_elems_stride = 768, 
)


AI_STATIC_CONST ai_u8 slice_7_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_7_axes, AI_ARRAY_FORMAT_U8,
    slice_7_axes_data, slice_7_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_7_starts_data[] = { 64 };
AI_ARRAY_OBJ_DECLARE(
    slice_7_starts, AI_ARRAY_FORMAT_S16,
    slice_7_starts_data, slice_7_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_7_ends_data[] = { 96 };
AI_ARRAY_OBJ_DECLARE(
    slice_7_ends, AI_ARRAY_FORMAT_S16,
    slice_7_ends_data, slice_7_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_7_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_7_layer, 7,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_7_chain,
  NULL, &slice_7_layer, AI_STATIC, 
  .axes = &slice_7_axes, 
  .starts = &slice_7_starts, 
  .ends = &slice_7_ends, 
)


AI_STATIC_CONST ai_u8 slice_6_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_6_axes, AI_ARRAY_FORMAT_U8,
    slice_6_axes_data, slice_6_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_6_starts_data[] = { 32 };
AI_ARRAY_OBJ_DECLARE(
    slice_6_starts, AI_ARRAY_FORMAT_S16,
    slice_6_starts_data, slice_6_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_6_ends_data[] = { 64 };
AI_ARRAY_OBJ_DECLARE(
    slice_6_ends, AI_ARRAY_FORMAT_S16,
    slice_6_ends_data, slice_6_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_6_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_6_layer, 6,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_6_chain,
  NULL, &slice_6_layer, AI_STATIC, 
  .axes = &slice_6_axes, 
  .starts = &slice_6_starts, 
  .ends = &slice_6_ends, 
)


AI_STATIC_CONST ai_u8 slice_5_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_5_axes, AI_ARRAY_FORMAT_U8,
    slice_5_axes_data, slice_5_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_5_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_5_starts, AI_ARRAY_FORMAT_S16,
    slice_5_starts_data, slice_5_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_5_ends_data[] = { 32 };
AI_ARRAY_OBJ_DECLARE(
    slice_5_ends, AI_ARRAY_FORMAT_S16,
    slice_5_ends_data, slice_5_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_5_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_5_layer, 5,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_5_chain,
  NULL, &slice_5_layer, AI_STATIC, 
  .axes = &slice_5_axes, 
  .starts = &slice_5_starts, 
  .ends = &slice_5_ends, 
)


AI_STATIC_CONST ai_u8 slice_3_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_3_axes, AI_ARRAY_FORMAT_U8,
    slice_3_axes_data, slice_3_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_3_starts_data[] = { 128 };
AI_ARRAY_OBJ_DECLARE(
    slice_3_starts, AI_ARRAY_FORMAT_S16,
    slice_3_starts_data, slice_3_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_3_ends_data[] = { 192 };
AI_ARRAY_OBJ_DECLARE(
    slice_3_ends, AI_ARRAY_FORMAT_S16,
    slice_3_ends_data, slice_3_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_3_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_3_layer, 3,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_3_chain,
  NULL, &slice_3_layer, AI_STATIC, 
  .axes = &slice_3_axes, 
  .starts = &slice_3_starts, 
  .ends = &slice_3_ends, 
)


AI_STATIC_CONST ai_u8 slice_2_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_2_axes, AI_ARRAY_FORMAT_U8,
    slice_2_axes_data, slice_2_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_2_starts_data[] = { 64 };
AI_ARRAY_OBJ_DECLARE(
    slice_2_starts, AI_ARRAY_FORMAT_S16,
    slice_2_starts_data, slice_2_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_2_ends_data[] = { 128 };
AI_ARRAY_OBJ_DECLARE(
    slice_2_ends, AI_ARRAY_FORMAT_S16,
    slice_2_ends_data, slice_2_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_2_layer, 2,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_2_chain,
  NULL, &slice_2_layer, AI_STATIC, 
  .axes = &slice_2_axes, 
  .starts = &slice_2_starts, 
  .ends = &slice_2_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_16_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &split_11_output1, &slice_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_16_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_16_layer, 16,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_16_chain,
  NULL, &eltwise_16_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_18_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_17_output, &slice_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_18_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_18_layer, 18,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_18_chain,
  NULL, &eltwise_18_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_19_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &split_11_output2, &eltwise_18_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_19_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_19_layer, 19,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_19_chain,
  NULL, &eltwise_19_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)


AI_STATIC_CONST ai_u8 slice_1_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_1_axes, AI_ARRAY_FORMAT_U8,
    slice_1_axes_data, slice_1_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_1_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_1_starts, AI_ARRAY_FORMAT_S16,
    slice_1_starts_data, slice_1_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_1_ends_data[] = { 64 };
AI_ARRAY_OBJ_DECLARE(
    slice_1_ends, AI_ARRAY_FORMAT_S16,
    slice_1_ends_data, slice_1_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_1_layer, 1,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_1_chain,
  NULL, &slice_1_layer, AI_STATIC, 
  .axes = &slice_1_axes, 
  .starts = &slice_1_starts, 
  .ends = &slice_1_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_12_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &split_11_output0, &slice_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_12_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_12_layer, 12,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_12_chain,
  NULL, &eltwise_12_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &arith_constant5_2D, &nl_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_15_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_15_layer, 15,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_15_chain,
  NULL, &eltwise_15_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_21_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_15_output, &nl_20_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_21_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_21_layer, 21,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_21_chain,
  NULL, &eltwise_21_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_13_output, &serving_default_gru1_h_in0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_14_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_14_layer, 14,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_14_chain,
  NULL, &eltwise_14_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_22_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_14_output, &eltwise_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_22_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_22_layer, 22,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_22_chain,
  NULL, &eltwise_22_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  unpack_24_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_22_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &unpack_24_output0),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  unpack_24_layer, 24,
  UNPACK_TYPE, 0x0, NULL,
  unpack, forward_unpack,
  &unpack_24_chain,
  NULL, &unpack_24_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  split_26_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_25_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &split_26_output0, &split_26_output1, &split_26_output2),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &split_26_num_or_size_splits),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  split_26_layer, 26,
  SPLIT_TYPE, 0x0, NULL,
  split, forward_split,
  &split_26_chain,
  NULL, &split_26_layer, AI_STATIC, 
  .outer_elems = 1, 
  .outer_elems_stride = 384, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_31_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &split_26_output1, &slice_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_31_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_31_layer, 31,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_31_chain,
  NULL, &eltwise_31_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_33_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_32_output, &slice_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_33_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_33_layer, 33,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_33_chain,
  NULL, &eltwise_33_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_34_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &split_26_output2, &eltwise_33_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_34_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_34_layer, 34,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_34_chain,
  NULL, &eltwise_34_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_27_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &split_26_output0, &slice_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_27_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_27_layer, 27,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_27_chain,
  NULL, &eltwise_27_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_30_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &arith_constant5_2D, &nl_28_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_30_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_30_layer, 30,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_30_chain,
  NULL, &eltwise_30_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_36_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_30_output, &nl_35_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_36_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_36_layer, 36,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_36_chain,
  NULL, &eltwise_36_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_29_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &nl_28_output, &serving_default_gru2_h_in0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_29_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_29_layer, 29,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_29_chain,
  NULL, &eltwise_29_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_37_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_29_output, &eltwise_36_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_37_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_37_layer, 37,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &eltwise_37_chain,
  NULL, &eltwise_37_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)
/**  Hybrid layers declarations section  *************************************/
void forward_lite_unpack_9(_stai_gru_network_context* net_ctx)
{
  serving_default_pose_sequence0_output_array.data = AI_PTR(net_ctx->_inputs[2] + 0);
  serving_default_pose_sequence0_output_array.data_start = AI_PTR(net_ctx->_inputs[2] + 0);
  unpack_9_output0_array.data = AI_PTR(net_ctx->_activations[0] + 548);
  unpack_9_output0_array.data_start = AI_PTR(net_ctx->_activations[0] + 548);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(9, 1, { serving_default_pose_sequence0_output.data->data});
  forward_unpack(&unpack_9_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(9, 1, { unpack_9_output0.data->data});
}
void forward_lite_split_11(_stai_gru_network_context* net_ctx)
{
  gemm_10_output_array.data = AI_PTR(net_ctx->_activations[0] + 1152);
  gemm_10_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1152);
  split_11_num_or_size_splits_array.data = AI_PTR(net_ctx->_weights[0] + 43012);
  split_11_num_or_size_splits_array.data_start = AI_PTR(net_ctx->_weights[0] + 43012);
  split_11_output0_array.data = AI_PTR(net_ctx->_activations[0] + 512);
  split_11_output0_array.data_start = AI_PTR(net_ctx->_activations[0] + 512);
  split_11_output1_array.data = AI_PTR(net_ctx->_activations[0] + 256);
  split_11_output1_array.data_start = AI_PTR(net_ctx->_activations[0] + 256);
  split_11_output2_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  split_11_output2_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(11, 1, { gemm_10_output.data->data});
  forward_split(&split_11_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(11, 3, { split_11_output0.data->data,split_11_output1.data->data,split_11_output2.data->data});
}
void forward_lite_slice_7(_stai_gru_network_context* net_ctx)
{
  gemm_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 1536);
  gemm_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1536);
  slice_7_output_array.data = AI_PTR(net_ctx->_activations[0] + 1152);
  slice_7_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1152);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(7, 1, { gemm_4_output.data->data});
  forward_slice(&slice_7_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(7, 1, { slice_7_output.data->data});
}
void forward_lite_slice_6(_stai_gru_network_context* net_ctx)
{
  gemm_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 1536);
  gemm_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1536);
  slice_6_output_array.data = AI_PTR(net_ctx->_activations[0] + 1280);
  slice_6_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1280);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(6, 1, { gemm_4_output.data->data});
  forward_slice(&slice_6_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(6, 1, { slice_6_output.data->data});
}
void forward_lite_slice_5(_stai_gru_network_context* net_ctx)
{
  gemm_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 1536);
  gemm_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1536);
  slice_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 1920);
  slice_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1920);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(5, 1, { gemm_4_output.data->data});
  forward_slice(&slice_5_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(5, 1, { slice_5_output.data->data});
}
void forward_lite_slice_3(_stai_gru_network_context* net_ctx)
{
  gemm_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 2048);
  gemm_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2048);
  slice_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 1408);
  slice_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1408);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(3, 1, { gemm_0_output.data->data});
  forward_slice(&slice_3_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(3, 1, { slice_3_output.data->data});
}
void forward_lite_slice_2(_stai_gru_network_context* net_ctx)
{
  gemm_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 2048);
  gemm_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2048);
  slice_2_output_array.data = AI_PTR(net_ctx->_activations[0] + 1664);
  slice_2_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1664);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(2, 1, { gemm_0_output.data->data});
  forward_slice(&slice_2_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(2, 1, { slice_2_output.data->data});
}
void forward_lite_eltwise_16(_stai_gru_network_context* net_ctx)
{
  split_11_output1_array.data = AI_PTR(net_ctx->_activations[0] + 256);
  split_11_output1_array.data_start = AI_PTR(net_ctx->_activations[0] + 256);
  slice_2_output_array.data = AI_PTR(net_ctx->_activations[0] + 1664);
  slice_2_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1664);
  eltwise_16_output_array.data = AI_PTR(net_ctx->_activations[0] + 256);
  eltwise_16_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 256);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(16, 2, { split_11_output1.data->data,slice_2_output.data->data});
  forward_eltwise(&eltwise_16_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(16, 1, { eltwise_16_output.data->data});
}
void forward_lite_eltwise_18(_stai_gru_network_context* net_ctx)
{
  nl_17_output_array.data = AI_PTR(net_ctx->_activations[0] + 1664);
  nl_17_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1664);
  slice_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 1408);
  slice_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1408);
  eltwise_18_output_array.data = AI_PTR(net_ctx->_activations[0] + 256);
  eltwise_18_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 256);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(18, 2, { nl_17_output.data->data,slice_3_output.data->data});
  forward_eltwise(&eltwise_18_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(18, 1, { eltwise_18_output.data->data});
}
void forward_lite_eltwise_19(_stai_gru_network_context* net_ctx)
{
  split_11_output2_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  split_11_output2_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_18_output_array.data = AI_PTR(net_ctx->_activations[0] + 256);
  eltwise_18_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 256);
  eltwise_19_output_array.data = AI_PTR(net_ctx->_activations[0] + 1408);
  eltwise_19_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1408);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(19, 2, { split_11_output2.data->data,eltwise_18_output.data->data});
  forward_eltwise(&eltwise_19_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(19, 1, { eltwise_19_output.data->data});
}
void forward_lite_slice_1(_stai_gru_network_context* net_ctx)
{
  gemm_0_output_array.data = AI_PTR(net_ctx->_activations[0] + 2048);
  gemm_0_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2048);
  slice_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 256);
  slice_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 256);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(1, 1, { gemm_0_output.data->data});
  forward_slice(&slice_1_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(1, 1, { slice_1_output.data->data});
}
void forward_lite_eltwise_12(_stai_gru_network_context* net_ctx)
{
  split_11_output0_array.data = AI_PTR(net_ctx->_activations[0] + 512);
  split_11_output0_array.data_start = AI_PTR(net_ctx->_activations[0] + 512);
  slice_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 256);
  slice_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 256);
  eltwise_12_output_array.data = AI_PTR(net_ctx->_activations[0] + 1408);
  eltwise_12_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1408);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(12, 2, { split_11_output0.data->data,slice_1_output.data->data});
  forward_eltwise(&eltwise_12_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(12, 1, { eltwise_12_output.data->data});
}
void forward_lite_eltwise_15(_stai_gru_network_context* net_ctx)
{
  arith_constant5_2D_array.data = AI_PTR(net_ctx->_weights[0] + 0);
  arith_constant5_2D_array.data_start = AI_PTR(net_ctx->_weights[0] + 0);
  nl_13_output_array.data = AI_PTR(net_ctx->_activations[0] + 256);
  nl_13_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 256);
  eltwise_15_output_array.data = AI_PTR(net_ctx->_activations[0] + 512);
  eltwise_15_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 512);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(15, 2, { arith_constant5_2D.data->data,nl_13_output.data->data});
  forward_eltwise(&eltwise_15_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(15, 1, { eltwise_15_output.data->data});
}
void forward_lite_eltwise_21(_stai_gru_network_context* net_ctx)
{
  eltwise_15_output_array.data = AI_PTR(net_ctx->_activations[0] + 512);
  eltwise_15_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 512);
  nl_20_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  nl_20_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_21_output_array.data = AI_PTR(net_ctx->_activations[0] + 1408);
  eltwise_21_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1408);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(21, 2, { eltwise_15_output.data->data,nl_20_output.data->data});
  forward_eltwise(&eltwise_21_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(21, 1, { eltwise_21_output.data->data});
}
void forward_lite_eltwise_14(_stai_gru_network_context* net_ctx)
{
  nl_13_output_array.data = AI_PTR(net_ctx->_activations[0] + 256);
  nl_13_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 256);
  serving_default_gru1_h_in0_output_array.data = AI_PTR(net_ctx->_inputs[0] + 0);
  serving_default_gru1_h_in0_output_array.data_start = AI_PTR(net_ctx->_inputs[0] + 0);
  eltwise_14_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_14_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(14, 2, { nl_13_output.data->data,serving_default_gru1_h_in0_output.data->data});
  forward_eltwise(&eltwise_14_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(14, 1, { eltwise_14_output.data->data});
}
void forward_lite_eltwise_22(_stai_gru_network_context* net_ctx)
{
  eltwise_14_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_14_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_21_output_array.data = AI_PTR(net_ctx->_activations[0] + 1408);
  eltwise_21_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1408);
  eltwise_22_output_array.data = AI_PTR(net_ctx->_outputs[0] + 0);
  eltwise_22_output_array.data_start = AI_PTR(net_ctx->_outputs[0] + 0);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(22, 2, { eltwise_14_output.data->data,eltwise_21_output.data->data});
  forward_eltwise(&eltwise_22_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(22, 1, { eltwise_22_output.data->data});
}
void forward_lite_unpack_24(_stai_gru_network_context* net_ctx)
{
  eltwise_22_output_array.data = AI_PTR(net_ctx->_outputs[0] + 0);
  eltwise_22_output_array.data_start = AI_PTR(net_ctx->_outputs[0] + 0);
  unpack_24_output0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  unpack_24_output0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(24, 1, { eltwise_22_output.data->data});
  forward_unpack(&unpack_24_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(24, 1, { unpack_24_output0.data->data});
}
void forward_lite_split_26(_stai_gru_network_context* net_ctx)
{
  gemm_25_output_array.data = AI_PTR(net_ctx->_activations[0] + 1408);
  gemm_25_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1408);
  split_26_num_or_size_splits_array.data = AI_PTR(net_ctx->_weights[0] + 130568);
  split_26_num_or_size_splits_array.data_start = AI_PTR(net_ctx->_weights[0] + 130568);
  split_26_output0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  split_26_output0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  split_26_output1_array.data = AI_PTR(net_ctx->_activations[0] + 128);
  split_26_output1_array.data_start = AI_PTR(net_ctx->_activations[0] + 128);
  split_26_output2_array.data = AI_PTR(net_ctx->_activations[0] + 512);
  split_26_output2_array.data_start = AI_PTR(net_ctx->_activations[0] + 512);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(26, 1, { gemm_25_output.data->data});
  forward_split(&split_26_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(26, 3, { split_26_output0.data->data,split_26_output1.data->data,split_26_output2.data->data});
}
void forward_lite_eltwise_31(_stai_gru_network_context* net_ctx)
{
  split_26_output1_array.data = AI_PTR(net_ctx->_activations[0] + 128);
  split_26_output1_array.data_start = AI_PTR(net_ctx->_activations[0] + 128);
  slice_6_output_array.data = AI_PTR(net_ctx->_activations[0] + 1280);
  slice_6_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1280);
  eltwise_31_output_array.data = AI_PTR(net_ctx->_activations[0] + 640);
  eltwise_31_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 640);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(31, 2, { split_26_output1.data->data,slice_6_output.data->data});
  forward_eltwise(&eltwise_31_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(31, 1, { eltwise_31_output.data->data});
}
void forward_lite_eltwise_33(_stai_gru_network_context* net_ctx)
{
  nl_32_output_array.data = AI_PTR(net_ctx->_activations[0] + 128);
  nl_32_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 128);
  slice_7_output_array.data = AI_PTR(net_ctx->_activations[0] + 1152);
  slice_7_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1152);
  eltwise_33_output_array.data = AI_PTR(net_ctx->_activations[0] + 640);
  eltwise_33_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 640);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(33, 2, { nl_32_output.data->data,slice_7_output.data->data});
  forward_eltwise(&eltwise_33_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(33, 1, { eltwise_33_output.data->data});
}
void forward_lite_eltwise_34(_stai_gru_network_context* net_ctx)
{
  split_26_output2_array.data = AI_PTR(net_ctx->_activations[0] + 512);
  split_26_output2_array.data_start = AI_PTR(net_ctx->_activations[0] + 512);
  eltwise_33_output_array.data = AI_PTR(net_ctx->_activations[0] + 640);
  eltwise_33_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 640);
  eltwise_34_output_array.data = AI_PTR(net_ctx->_activations[0] + 128);
  eltwise_34_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 128);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(34, 2, { split_26_output2.data->data,eltwise_33_output.data->data});
  forward_eltwise(&eltwise_34_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(34, 1, { eltwise_34_output.data->data});
}
void forward_lite_eltwise_27(_stai_gru_network_context* net_ctx)
{
  split_26_output0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  split_26_output0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  slice_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 1920);
  slice_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1920);
  eltwise_27_output_array.data = AI_PTR(net_ctx->_activations[0] + 128);
  eltwise_27_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 128);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(27, 2, { split_26_output0.data->data,slice_5_output.data->data});
  forward_eltwise(&eltwise_27_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(27, 1, { eltwise_27_output.data->data});
}
void forward_lite_eltwise_30(_stai_gru_network_context* net_ctx)
{
  arith_constant5_2D_array.data = AI_PTR(net_ctx->_weights[0] + 0);
  arith_constant5_2D_array.data_start = AI_PTR(net_ctx->_weights[0] + 0);
  nl_28_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  nl_28_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  eltwise_30_output_array.data = AI_PTR(net_ctx->_activations[0] + 128);
  eltwise_30_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 128);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(30, 2, { arith_constant5_2D.data->data,nl_28_output.data->data});
  forward_eltwise(&eltwise_30_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(30, 1, { eltwise_30_output.data->data});
}
void forward_lite_eltwise_36(_stai_gru_network_context* net_ctx)
{
  eltwise_30_output_array.data = AI_PTR(net_ctx->_activations[0] + 128);
  eltwise_30_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 128);
  nl_35_output_array.data = AI_PTR(net_ctx->_activations[0] + 512);
  nl_35_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 512);
  eltwise_36_output_array.data = AI_PTR(net_ctx->_activations[0] + 640);
  eltwise_36_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 640);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(36, 2, { eltwise_30_output.data->data,nl_35_output.data->data});
  forward_eltwise(&eltwise_36_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(36, 1, { eltwise_36_output.data->data});
}
void forward_lite_eltwise_29(_stai_gru_network_context* net_ctx)
{
  nl_28_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  nl_28_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  serving_default_gru2_h_in0_output_array.data = AI_PTR(net_ctx->_inputs[1] + 0);
  serving_default_gru2_h_in0_output_array.data_start = AI_PTR(net_ctx->_inputs[1] + 0);
  eltwise_29_output_array.data = AI_PTR(net_ctx->_activations[0] + 128);
  eltwise_29_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 128);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(29, 2, { nl_28_output.data->data,serving_default_gru2_h_in0_output.data->data});
  forward_eltwise(&eltwise_29_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(29, 1, { eltwise_29_output.data->data});
}
void forward_lite_eltwise_37(_stai_gru_network_context* net_ctx)
{
  eltwise_29_output_array.data = AI_PTR(net_ctx->_activations[0] + 128);
  eltwise_29_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 128);
  eltwise_36_output_array.data = AI_PTR(net_ctx->_activations[0] + 640);
  eltwise_36_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 640);
  eltwise_37_output_array.data = AI_PTR(net_ctx->_outputs[2] + 0);
  eltwise_37_output_array.data_start = AI_PTR(net_ctx->_outputs[2] + 0);
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(37, 2, { eltwise_29_output.data->data,eltwise_36_output.data->data});
  forward_eltwise(&eltwise_37_layer);
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(37, 1, { eltwise_37_output.data->data});
}

/*****************************************************************************/













static const ai_i32 nl_17_t_in_0_shape_ch_prod_const_s32 = 64;



static const ai_i32 nl_20_t_in_0_shape_ch_prod_const_s32 = 64;



static const ai_i32 nl_13_t_in_0_shape_ch_prod_const_s32 = 64;









static const ai_i32 nl_32_t_in_0_shape_ch_prod_const_s32 = 32;



static const ai_i32 nl_35_t_in_0_shape_ch_prod_const_s32 = 32;


static const ai_i32 nl_28_t_in_0_shape_ch_prod_const_s32 = 32;






static const ai_i32 nl_38_nl_t_in_0_shape_ch_prod_const_s32 = 32;


static const ai_i32 nl_40_t_in_0_shape_ch_prod_const_s32 = 2;
STAI_API_ENTRY
stai_return_code stai_gru_network_run(
  stai_network* network,
  const stai_run_mode mode)
{
   STAI_UNUSED(mode)
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_ACTIVATIONS) != STAI_FLAG_ACTIVATIONS,
        STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_INPUTS) != STAI_FLAG_INPUTS,
                  STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_OUTPUTS) != STAI_FLAG_OUTPUTS,
                  STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_WEIGHTS) != STAI_FLAG_WEIGHTS,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)


  /* LITE_KERNEL_SECTION BEGIN unpack_9 */
  {
    
  forward_lite_unpack_9(net_ctx);
  }
  /* LITE_KERNEL_SECTION END unpack_9 */
  /* LITE_KERNEL_SECTION BEGIN gemm_10 */
  {
      forward_lite_dense_if32of32wf32_args arg_30f51e = {
      .output = (float*)(net_ctx->_activations[0] + 1152),
      .input = (float*)(net_ctx->_activations[0] + 548),
      .weights = (float*)(net_ctx->_weights[0] + 4),
      .bias = (float*)(net_ctx->_weights[0] + 42244),
      .n_channel_in = 55,
      .n_channel_out = 192,
      .n_elements = 1,
    };
  
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(10, 1, {(stai_ptr) (float*)(net_ctx->_activations[0] + 548)});
    
  forward_lite_dense_if32of32wf32((forward_lite_dense_if32of32wf32_args*)&arg_30f51e);
    
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(10, 1, {(stai_ptr) (float*)(net_ctx->_activations[0] + 1152)});
  }
  /* LITE_KERNEL_SECTION END gemm_10 */
  /* LITE_KERNEL_SECTION BEGIN split_11 */
  {
    
  forward_lite_split_11(net_ctx);
  }
  /* LITE_KERNEL_SECTION END split_11 */
  /* LITE_KERNEL_SECTION BEGIN gemm_4 */
  {
      forward_lite_dense_if32of32wf32_args arg_30f51e = {
      .output = (float*)(net_ctx->_activations[0] + 1536),
      .input = (float*)(net_ctx->_inputs[1] + 0),
      .weights = (float*)(net_ctx->_weights[0] + 43016),
      .bias = (float*)(net_ctx->_weights[0] + 55304),
      .n_channel_in = 32,
      .n_channel_out = 96,
      .n_elements = 1,
    };
  
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(4, 1, {(stai_ptr) (float*)(net_ctx->_inputs[1] + 0)});
    
  forward_lite_dense_if32of32wf32((forward_lite_dense_if32of32wf32_args*)&arg_30f51e);
    
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(4, 1, {(stai_ptr) (float*)(net_ctx->_activations[0] + 1536)});
  }
  /* LITE_KERNEL_SECTION END gemm_4 */
  /* LITE_KERNEL_SECTION BEGIN slice_7 */
  {
    
  forward_lite_slice_7(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_7 */
  /* LITE_KERNEL_SECTION BEGIN slice_6 */
  {
    
  forward_lite_slice_6(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_6 */
  /* LITE_KERNEL_SECTION BEGIN slice_5 */
  {
    
  forward_lite_slice_5(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_5 */
  /* LITE_KERNEL_SECTION BEGIN gemm_0 */
  {
      forward_lite_dense_if32of32wf32_args arg_30f51e = {
      .output = (float*)(net_ctx->_activations[0] + 2048),
      .input = (float*)(net_ctx->_inputs[0] + 0),
      .weights = (float*)(net_ctx->_weights[0] + 55688),
      .bias = (float*)(net_ctx->_weights[0] + 104840),
      .n_channel_in = 64,
      .n_channel_out = 192,
      .n_elements = 1,
    };
  
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(0, 1, {(stai_ptr) (float*)(net_ctx->_inputs[0] + 0)});
    
  forward_lite_dense_if32of32wf32((forward_lite_dense_if32of32wf32_args*)&arg_30f51e);
    
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(0, 1, {(stai_ptr) (float*)(net_ctx->_activations[0] + 2048)});
  }
  /* LITE_KERNEL_SECTION END gemm_0 */
  /* LITE_KERNEL_SECTION BEGIN slice_3 */
  {
    
  forward_lite_slice_3(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_3 */
  /* LITE_KERNEL_SECTION BEGIN slice_2 */
  {
    
  forward_lite_slice_2(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_2 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_16 */
  {
    
  forward_lite_eltwise_16(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_16 */
  /* LITE_KERNEL_SECTION BEGIN nl_17 */
  {
      ai_handle nl_17_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 1664);
    const ai_handle nl_17_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 256);
  
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(17, 1, {(stai_ptr) nl_17_t_in_0_ptr_const_handle});
    
  forward_lite_nl_sigmoid_if32of32(nl_17_t_out_0_ptr_handle, nl_17_t_in_0_ptr_const_handle, nl_17_t_in_0_shape_ch_prod_const_s32, NULL);
    
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(17, 1, {(stai_ptr) nl_17_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END nl_17 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_18 */
  {
    
  forward_lite_eltwise_18(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_18 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_19 */
  {
    
  forward_lite_eltwise_19(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_19 */
  /* LITE_KERNEL_SECTION BEGIN nl_20 */
  {
      ai_handle nl_20_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 0);
    const ai_handle nl_20_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 1408);
  
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(20, 1, {(stai_ptr) nl_20_t_in_0_ptr_const_handle});
    
  forward_lite_nl_tanh_if32of32(nl_20_t_out_0_ptr_handle, nl_20_t_in_0_ptr_const_handle, nl_20_t_in_0_shape_ch_prod_const_s32, NULL);
    
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(20, 1, {(stai_ptr) nl_20_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END nl_20 */
  /* LITE_KERNEL_SECTION BEGIN slice_1 */
  {
    
  forward_lite_slice_1(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_1 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_12 */
  {
    
  forward_lite_eltwise_12(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_12 */
  /* LITE_KERNEL_SECTION BEGIN nl_13 */
  {
      ai_handle nl_13_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 256);
    const ai_handle nl_13_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 1408);
  
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(13, 1, {(stai_ptr) nl_13_t_in_0_ptr_const_handle});
    
  forward_lite_nl_sigmoid_if32of32(nl_13_t_out_0_ptr_handle, nl_13_t_in_0_ptr_const_handle, nl_13_t_in_0_shape_ch_prod_const_s32, NULL);
    
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(13, 1, {(stai_ptr) nl_13_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END nl_13 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_15 */
  {
    
  forward_lite_eltwise_15(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_15 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_21 */
  {
    
  forward_lite_eltwise_21(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_21 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_14 */
  {
    
  forward_lite_eltwise_14(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_14 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_22 */
  {
    
  forward_lite_eltwise_22(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_22 */
  /* LITE_KERNEL_SECTION BEGIN unpack_24 */
  {
    
  forward_lite_unpack_24(net_ctx);
  }
  /* LITE_KERNEL_SECTION END unpack_24 */
  /* LITE_KERNEL_SECTION BEGIN gemm_25 */
  {
      forward_lite_dense_if32of32wf32_args arg_30f51e = {
      .output = (float*)(net_ctx->_activations[0] + 1408),
      .input = (float*)(net_ctx->_activations[0] + 0),
      .weights = (float*)(net_ctx->_weights[0] + 105608),
      .bias = (float*)(net_ctx->_weights[0] + 130184),
      .n_channel_in = 64,
      .n_channel_out = 96,
      .n_elements = 1,
    };
  
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(25, 1, {(stai_ptr) (float*)(net_ctx->_activations[0] + 0)});
    
  forward_lite_dense_if32of32wf32((forward_lite_dense_if32of32wf32_args*)&arg_30f51e);
    
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(25, 1, {(stai_ptr) (float*)(net_ctx->_activations[0] + 1408)});
  }
  /* LITE_KERNEL_SECTION END gemm_25 */
  /* LITE_KERNEL_SECTION BEGIN split_26 */
  {
    
  forward_lite_split_26(net_ctx);
  }
  /* LITE_KERNEL_SECTION END split_26 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_31 */
  {
    
  forward_lite_eltwise_31(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_31 */
  /* LITE_KERNEL_SECTION BEGIN nl_32 */
  {
      ai_handle nl_32_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 128);
    const ai_handle nl_32_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 640);
  
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(32, 1, {(stai_ptr) nl_32_t_in_0_ptr_const_handle});
    
  forward_lite_nl_sigmoid_if32of32(nl_32_t_out_0_ptr_handle, nl_32_t_in_0_ptr_const_handle, nl_32_t_in_0_shape_ch_prod_const_s32, NULL);
    
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(32, 1, {(stai_ptr) nl_32_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END nl_32 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_33 */
  {
    
  forward_lite_eltwise_33(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_33 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_34 */
  {
    
  forward_lite_eltwise_34(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_34 */
  /* LITE_KERNEL_SECTION BEGIN nl_35 */
  {
      ai_handle nl_35_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 512);
    const ai_handle nl_35_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 128);
  
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(35, 1, {(stai_ptr) nl_35_t_in_0_ptr_const_handle});
    
  forward_lite_nl_tanh_if32of32(nl_35_t_out_0_ptr_handle, nl_35_t_in_0_ptr_const_handle, nl_35_t_in_0_shape_ch_prod_const_s32, NULL);
    
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(35, 1, {(stai_ptr) nl_35_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END nl_35 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_27 */
  {
    
  forward_lite_eltwise_27(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_27 */
  /* LITE_KERNEL_SECTION BEGIN nl_28 */
  {
      ai_handle nl_28_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 0);
    const ai_handle nl_28_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 128);
  
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(28, 1, {(stai_ptr) nl_28_t_in_0_ptr_const_handle});
    
  forward_lite_nl_sigmoid_if32of32(nl_28_t_out_0_ptr_handle, nl_28_t_in_0_ptr_const_handle, nl_28_t_in_0_shape_ch_prod_const_s32, NULL);
    
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(28, 1, {(stai_ptr) nl_28_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END nl_28 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_30 */
  {
    
  forward_lite_eltwise_30(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_30 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_36 */
  {
    
  forward_lite_eltwise_36(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_36 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_29 */
  {
    
  forward_lite_eltwise_29(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_29 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_37 */
  {
    
  forward_lite_eltwise_37(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_37 */
  /* LITE_KERNEL_SECTION BEGIN gemm_38 */
  {
      forward_lite_dense_if32of32wf32_args arg_30f51e = {
      .output = (float*)(net_ctx->_activations[0] + 128),
      .input = (float*)(net_ctx->_outputs[2] + 0),
      .weights = (float*)(net_ctx->_weights[0] + 130572),
      .bias = (float*)(net_ctx->_weights[0] + 134668),
      .n_channel_in = 32,
      .n_channel_out = 32,
      .n_elements = 1,
    };
  
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(38, 1, {(stai_ptr) (float*)(net_ctx->_outputs[2] + 0)});
    
  forward_lite_dense_if32of32wf32((forward_lite_dense_if32of32wf32_args*)&arg_30f51e);
    
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(38, 1, {(stai_ptr) (float*)(net_ctx->_activations[0] + 128)});
  }
  /* LITE_KERNEL_SECTION END gemm_38 */
  /* LITE_KERNEL_SECTION BEGIN nl_38_nl */
  {
      ai_handle nl_38_nl_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 512);
    const ai_handle nl_38_nl_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 128);
  
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(38, 1, {(stai_ptr) nl_38_nl_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(nl_38_nl_t_out_0_ptr_handle, nl_38_nl_t_in_0_ptr_const_handle, nl_38_nl_t_in_0_shape_ch_prod_const_s32, NULL);
    
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(38, 1, {(stai_ptr) nl_38_nl_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END nl_38_nl */
  /* LITE_KERNEL_SECTION BEGIN gemm_39 */
  {
      forward_lite_dense_if32of32wf32_args arg_30f51e = {
      .output = (float*)(net_ctx->_activations[0] + 128),
      .input = (float*)(net_ctx->_activations[0] + 512),
      .weights = (float*)(net_ctx->_weights[0] + 134796),
      .bias = (float*)(net_ctx->_weights[0] + 135052),
      .n_channel_in = 32,
      .n_channel_out = 2,
      .n_elements = 1,
    };
  
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(39, 1, {(stai_ptr) (float*)(net_ctx->_activations[0] + 512)});
    
  forward_lite_dense_if32of32wf32((forward_lite_dense_if32of32wf32_args*)&arg_30f51e);
    
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(39, 1, {(stai_ptr) (float*)(net_ctx->_activations[0] + 128)});
  }
  /* LITE_KERNEL_SECTION END gemm_39 */
  /* LITE_KERNEL_SECTION BEGIN nl_40 */
  {
      ai_handle nl_40_t_out_0_ptr_handle = (ai_handle)(net_ctx->_outputs[1] + 0);
    const ai_handle nl_40_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 128);
  
  _STAI_GRU_NETWORK_EVENT_NODE_START_CB(40, 1, {(stai_ptr) nl_40_t_in_0_ptr_const_handle});
    
  forward_lite_nl_softmax_if32of32(nl_40_t_out_0_ptr_handle, nl_40_t_in_0_ptr_const_handle, nl_40_t_in_0_shape_ch_prod_const_s32, 1, 2);
    
  _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB(40, 1, {(stai_ptr) nl_40_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END nl_40 */
  return net_ctx->_return_code;
}

/*****************************************************************************/
/*  Getters APIs Section  */
STAI_API_ENTRY
stai_size stai_gru_network_get_context_size()
{
  return (stai_size)STAI_GRU_NETWORK_CONTEXT_SIZE;
}

#if defined(HAVE_GRU_NETWORK_INFO)
STAI_API_ENTRY
stai_return_code stai_gru_network_get_info(
  stai_network* network,
  stai_network_info* info)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, info==NULL, STAI_ERROR_NETWORK_INVALID_INFO, net_ctx->_return_code)

  // Copy of network info struct
  *info = g_gru_network_info;

  return STAI_SUCCESS;
}
#endif


STAI_API_ENTRY
stai_return_code stai_gru_network_get_activations(
  stai_network* network, stai_ptr* activations, stai_size* n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, !n_activations, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_activations = STAI_GRU_NETWORK_ACTIVATIONS_NUM;
for (stai_size idx=0; activations && (idx<STAI_GRU_NETWORK_ACTIVATIONS_NUM); idx++) {
    // get address of the activations buffers
    activations[idx] = net_ctx->_activations[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_gru_network_get_weights(
  stai_network* network, stai_ptr* weights, stai_size* n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_weights, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_weights = STAI_GRU_NETWORK_WEIGHTS_NUM;
for (stai_size idx=0; weights && (idx<STAI_GRU_NETWORK_WEIGHTS_NUM); idx++) {
    // get address of the weights buffers
    weights[idx] = net_ctx->_weights[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_gru_network_get_inputs(
  stai_network* network, stai_ptr* inputs, stai_size* n_inputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_inputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_inputs = STAI_GRU_NETWORK_IN_NUM;
  for (stai_size idx=0; inputs && (idx<STAI_GRU_NETWORK_IN_NUM); idx++) {
    inputs[idx] = net_ctx->_inputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_gru_network_get_outputs(
  stai_network* network, stai_ptr* outputs, stai_size* n_outputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_outputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_outputs = STAI_GRU_NETWORK_OUT_NUM;
  for (stai_size idx=0; outputs && (idx<STAI_GRU_NETWORK_OUT_NUM); idx++) {
    outputs[idx] = net_ctx->_outputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_gru_network_get_error(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /* return 1st generated error or STAI_SUCCESS if no errors so far */
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_gru_network_get_states(
  stai_network* network, stai_ptr* states, stai_size* n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_states, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  /* get the number of internals states (supporting multi-heap also for internal states) */
  *n_states = STAI_GRU_NETWORK_STATES_NUM;

  STAI_UNUSED(states)
return net_ctx->_return_code;
}


/*****************************************************************************/
/*  Setters APIs Section  */

STAI_API_ENTRY
stai_return_code stai_gru_network_set_activations(
  stai_network* network,
  const stai_ptr* activations,
  const stai_size n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _activations_alignment[] = STAI_GRU_NETWORK_ACTIVATIONS_ALIGNMENTS;
  STAI_PRINT("  [stai_gru_network_set_activations] network(%p) activations[%d]: %p\n\n", net_ctx, n_activations, activations)
  _STAI_SET_ERROR(net_ctx, !activations,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_activations!=STAI_GRU_NETWORK_ACTIVATIONS_NUM,
                  STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_NUM, net_ctx->_return_code)

  for (stai_size idx=0; activations && idx<STAI_GRU_NETWORK_ACTIVATIONS_NUM; idx++) {
    STAI_PRINT("  activation[%d]: %p\n", idx, activations[idx])
    _STAI_SET_ERROR(net_ctx, activations[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)activations[idx]) & (_activations_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_activations[idx] = activations[idx];
  }
  net_ctx->_inputs[0] = activations[0] + 896;

  net_ctx->_inputs[1] = activations[0] + 768;

  net_ctx->_inputs[2] = activations[0] + 1152;

  net_ctx->_outputs[0] = activations[0] + 256;

  net_ctx->_outputs[1] = activations[0] + 136;

  net_ctx->_outputs[2] = activations[0] + 0;
_stai_gru_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_gru_network_set_weights(
  stai_network* network,
  const stai_ptr* weights,
  const stai_size n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _weights_alignment[] = STAI_GRU_NETWORK_WEIGHTS_ALIGNMENTS;
  _STAI_SET_ERROR(net_ctx, !weights,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_weights!=STAI_GRU_NETWORK_WEIGHTS_NUM,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_NUM, net_ctx->_return_code)
  for (stai_size idx=0; weights && idx<STAI_GRU_NETWORK_WEIGHTS_NUM; idx++) {
    STAI_PRINT("  weight[%d]: %p\n", idx, weights[idx])
    _STAI_SET_ERROR(net_ctx, weights[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)weights[idx]) & (_weights_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_weights[idx] = weights[idx];
  }_stai_gru_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_gru_network_set_inputs(
  stai_network* network,
  const stai_ptr* inputs,
  const stai_size n_inputs)
{
  const uintptr_t _inputs_alignment[] = STAI_GRU_NETWORK_IN_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !inputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_inputs!=STAI_GRU_NETWORK_IN_NUM,
                  STAI_ERROR_NETWORK_INVALID_IN_NUM, net_ctx->_return_code)

  for (stai_size idx=0; inputs && idx<STAI_GRU_NETWORK_IN_NUM; idx++) {
    STAI_PRINT("  input[%d]: %p\n", idx, inputs[idx])
    _STAI_SET_ERROR(net_ctx, inputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)inputs[idx]) & (_inputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_inputs[idx] = inputs[idx];
  }

  _stai_gru_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_gru_network_set_outputs(
  stai_network* network,
  const stai_ptr* outputs,
  const stai_size n_outputs)
{
  const uintptr_t _outputs_alignment[] = STAI_GRU_NETWORK_OUT_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !outputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_outputs!=STAI_GRU_NETWORK_OUT_NUM,
                  STAI_ERROR_NETWORK_INVALID_OUT_NUM, net_ctx->_return_code)

  for (stai_size idx=0; outputs && idx<n_outputs; idx++) {
    STAI_PRINT("  output[%d]: %p\n", idx, outputs[idx])
    _STAI_SET_ERROR(net_ctx, outputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)outputs[idx]) & (_outputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_outputs[idx] = outputs[idx];
  }

  _stai_gru_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_gru_network_set_states(
  stai_network* network,
  const stai_ptr* states,
  const stai_size n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  STAI_UNUSED(states)
  STAI_UNUSED(n_states)
_stai_gru_network_check(net_ctx);
  return net_ctx->_return_code;
}

STAI_API_ENTRY
stai_return_code stai_gru_network_set_callback(
  stai_network* network, const stai_event_cb cb, void* cb_cookie)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  STAI_PRINT("  set_callback %p cb %p cookie %p\n", net_ctx, cb, cb_cookie)
  // _STAI_SET_ERROR(net_ctx, cb==NULL, STAI_ERROR_NETWORK_INVALID_CALLBACK, net_ctx->_return_code)
  net_ctx->_callback = cb;
  net_ctx->_callback_cookie = cb_cookie;
  return net_ctx->_return_code;
}

#undef _STAI_SET_ERROR
#undef _STAI_CONTEXT_ALIGNMENT
#undef _STAI_CONTEXT_ACQUIRE
#undef _STAI_GRU_NETWORK_EVENT_NODE_START_CB
#undef _STAI_GRU_NETWORK_EVENT_NODE_STOP_CB
#undef _STAI_GRU_NETWORK_MODEL_SIGNATURE
#undef _STAI_GRU_NETWORK_DATETIME
#undef _STAI_GRU_NETWORK_COMPILE_DATETIME

