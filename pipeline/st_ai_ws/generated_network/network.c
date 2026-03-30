/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-03-18T23:54:33+0900
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
#include "network.h"
#include "network_details.h"
#include "network_data.h"
#include "stai_events.h"

#include "lite_operators.h"

#include "ai_lite_inspect.h"
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
#ifndef _STAI_NETWORK_EVENT_NODE_START_CB
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
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
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_START_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_START_CB */

#ifdef STAI_EVENT_NODE_STOP_CB
#ifndef _STAI_NETWORK_EVENT_NODE_STOP_CB
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
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
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_STOP_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_STOP_CB */


/*****************************************************************************/
#define _STAI_NETWORK_MODEL_SIGNATURE     "0x32857d1871785c054637edf01bb4098e"
#define _STAI_NETWORK_DATETIME            "2026-03-18T23:54:33+0900"
#define _STAI_NETWORK_COMPILE_DATETIME    __DATE__ " " __TIME__

#define _STAI_CONTEXT_ALIGNMENT        STAI_NETWORK_CONTEXT_ALIGNMENT

/*****************************************************************************/
#define g_network_activations_1     (NULL)




#if defined(HAVE_NETWORK_INFO)
/*****************************************************************************/
static const stai_network_info g_network_info = {
  .model_signature = _STAI_NETWORK_MODEL_SIGNATURE,
  .c_compile_datetime = _STAI_NETWORK_COMPILE_DATETIME,
  .c_model_name = STAI_NETWORK_MODEL_NAME,
  .c_model_datetime = _STAI_NETWORK_DATETIME,
  .c_model_signature = 0x0,
  .runtime_version = STAI_INIT_VERSION(11, 0, 0),
  .tool_version = STAI_INIT_VERSION(3, 0, 0),
  .api_version = STAI_INIT_VERSION(1, 0, 0),
  .n_macc = STAI_NETWORK_MACC_NUM,
  .n_nodes = STAI_NETWORK_NODES_NUM,
  .flags = STAI_NETWORK_FLAGS,
  .n_inputs = STAI_NETWORK_IN_NUM,
  .n_outputs = STAI_NETWORK_OUT_NUM,
  .n_activations = STAI_NETWORK_ACTIVATIONS_NUM,
  .n_weights = STAI_NETWORK_WEIGHTS_NUM,
  .n_states = STAI_NETWORK_STATES_NUM,
  .inputs = (stai_tensor[STAI_NETWORK_IN_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_IN_1_NAME,
      STAI_NETWORK_IN_1_FLAGS,
      STAI_NETWORK_IN_1_FORMAT,
      STAI_NETWORK_IN_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 3, 1, 55, 60),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
    .outputs = (stai_tensor[STAI_NETWORK_OUT_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_1_NAME,
      STAI_NETWORK_OUT_1_FLAGS,
      STAI_NETWORK_OUT_1_FORMAT,
      STAI_NETWORK_OUT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 3, 1, 2, 1),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .activations = (stai_tensor[STAI_NETWORK_ACTIVATIONS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_ACTIVATION_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_ACTIVATION_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 82944),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .weights = (stai_tensor[STAI_NETWORK_WEIGHTS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 538444),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },

  .states = NULL
};
#endif

#define _STAI_CONTEXT_ACQUIRE(_net_ctx, _net_handle) \
  _stai_network_context* _net_ctx = (_stai_network_context*)(_net_handle); \
  STAI_ASSERT(_net_ctx != NULL) \
  _STAI_SET_ERROR(_net_ctx, _net_ctx->_magic != STAI_MAGIC, \
                  STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE, _net_ctx->_return_code)


/*****************************************************************************/
static
void _stai_network_check(_stai_network_context* net_ctx)
{
  stai_size idx;

// Check activations status
  for (idx=0; idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    if (net_ctx->_activations[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_ACTIVATIONS_NUM) ? STAI_FLAG_ACTIVATIONS : STAI_FLAG_NONE;
// Check inputs status
  for (idx=0; idx<STAI_NETWORK_IN_NUM; idx++) {
    if (net_ctx->_inputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_IN_NUM) ? STAI_FLAG_INPUTS : STAI_FLAG_NONE;

  // Check outputs status
  for (idx=0; idx<STAI_NETWORK_OUT_NUM; idx++) {
    if (net_ctx->_outputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_OUT_NUM) ? STAI_FLAG_OUTPUTS : STAI_FLAG_NONE;

// Check weights status
  for (idx=0; idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    if (net_ctx->_weights[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_WEIGHTS_NUM) ? STAI_FLAG_WEIGHTS : STAI_FLAG_NONE;
STAI_PRINT("  [_stai_network_check] flags: 0x%08x\n", net_ctx->_flags)
}


/*****************************************************************************/
STAI_API_ENTRY
stai_return_code stai_network_init(
  stai_network* network)
{
  /* Memory where to store internal context is provided by applications as a raw byte buffer */
  _stai_network_context* net_ctx = (_stai_network_context*)(network);
  net_ctx->_return_code = STAI_SUCCESS;
  STAI_PRINT("[Entering Network Init] network(%p) context_size(%d)\n", net_ctx, (int32_t)sizeof(_stai_network_context))

  _STAI_SET_ERROR(net_ctx, STAI_NETWORK_CONTEXT_SIZE != sizeof(_stai_network_context),
                 STAI_ERROR_NETWORK_INVALID_CONTEXT_SIZE, net_ctx->_return_code)

  {
    const _stai_network_context _network_context = {
      ._magic = STAI_MAGIC,
      ._signature = STAI_NETWORK_MODEL_SIGNATURE,
      ._flags = STAI_NETWORK_FLAGS,
      ._return_code = STAI_SUCCESS,
      ._callback = NULL,
      ._callback_cookie = NULL,
      ._activations = {
      (stai_ptr)g_network_activations_1
      },
      ._weights = {
      (stai_ptr)g_network_weights_array
      },
      ._inputs = {
    NULL},
      ._outputs = {
    NULL},
    };

    // Deep copy of internal context to opaque buffer provided by app
    *net_ctx = _network_context;

    _stai_network_check(net_ctx);
  }

  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_deinit(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /*  Reset flags to initial state  */
  net_ctx->_flags = STAI_NETWORK_FLAGS;
  return net_ctx->_return_code;
}

/*****************************************************************************/





/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  input_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 3300, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  input_Transpose_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3300, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conv1d_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3840, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  relu_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3840, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  add_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3840, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  relu_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3840, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  relu_4_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  add_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  conv1d_7_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  relu_7_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9728, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  add_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9728, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  conv1d_10_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 152, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  relu_10_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 208, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  add_3_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 208, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  relu_11_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 208, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  output_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 2, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  val_16_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)



/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  input_Transpose_output, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 55, 1, 60), AI_STRIDE_INIT(4, 4, 4, 220, 220),
  1, &input_Transpose_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  input_output, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 1, 60, 1, 55), AI_STRIDE_INIT(4, 4, 4, 240, 240),
  1, &input_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  add_output, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 60), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &add_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conv1d_2_output, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 60), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv1d_2_output_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  relu_1_output, AI_STATIC,
  53, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 60), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &relu_1_output_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  add_1_output, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &add_1_output_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  relu_2_output, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 60), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &relu_2_output_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  relu_4_output, AI_STATIC,
  56, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 64), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &relu_4_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  add_2_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 76), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &add_2_output_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conv1d_7_output, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 64), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv1d_7_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  relu_7_output, AI_STATIC,
  59, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 76), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &relu_7_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  add_3_output, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 104), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &add_3_output_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  conv1d_10_output, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 76), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &conv1d_10_output_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  relu_10_output, AI_STATIC,
  51, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 104), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &relu_10_output_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  output_output, AI_STATIC,
  50, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &output_output_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  relu_11_output, AI_STATIC,
  52, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 104), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &relu_11_output_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  val_16, AI_STATIC,
  63, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &val_16_array, NULL)


AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_Transpose_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_Transpose_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_Transpose_layer, 2,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &input_Transpose_chain,
  NULL, &input_Transpose_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  add_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &relu_1_output, &conv1d_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  add_layer, 6,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &add_chain,
  NULL, &add_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  add_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &relu_4_output, &relu_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  add_1_layer, 12,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &add_1_chain,
  NULL, &add_1_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  add_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &relu_7_output, &conv1d_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  add_2_layer, 19,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &add_2_chain,
  NULL, &add_2_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  add_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &relu_10_output, &conv1d_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &add_3_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  add_3_layer, 26,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &add_3_chain,
  NULL, &add_3_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  output_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &relu_11_output, &val_16),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &output_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  output_layer, 28,
  GATHER_TYPE, 0x0, NULL,
  gather, forward_gather,
  &output_chain,
  NULL, &output_layer, AI_STATIC, 
  .axis = AI_SHAPE_HEIGHT, 
)
/**  Hybrid layers declarations section  *************************************/
void forward_lite_input_Transpose(_stai_network_context* net_ctx)
{
  input_output_array.data = AI_PTR(net_ctx->_inputs[0] + 0);
  input_output_array.data_start = AI_PTR(net_ctx->_inputs[0] + 0);
  input_Transpose_output_array.data = AI_PTR(net_ctx->_activations[0] + 38548);
  input_Transpose_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 38548);
  _STAI_NETWORK_EVENT_NODE_START_CB(2, 1, { input_output.data->data});
  forward_transpose(&input_Transpose_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(2, 1, { input_Transpose_output.data->data});
}
void forward_lite_add(_stai_network_context* net_ctx)
{
  relu_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 34304);
  relu_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 34304);
  conv1d_2_output_array.data = AI_PTR(net_ctx->_activations[0] + 51748);
  conv1d_2_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 51748);
  add_output_array.data = AI_PTR(net_ctx->_activations[0] + 34304);
  add_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 34304);
  _STAI_NETWORK_EVENT_NODE_START_CB(6, 2, { relu_1_output.data->data,conv1d_2_output.data->data});
  forward_eltwise(&add_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(6, 1, { add_output.data->data});
}
void forward_lite_add_1(_stai_network_context* net_ctx)
{
  relu_4_output_array.data = AI_PTR(net_ctx->_activations[0] + 65024);
  relu_4_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 65024);
  relu_2_output_array.data = AI_PTR(net_ctx->_activations[0] + 49664);
  relu_2_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 49664);
  add_1_output_array.data = AI_PTR(net_ctx->_activations[0] + 65024);
  add_1_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 65024);
  _STAI_NETWORK_EVENT_NODE_START_CB(12, 2, { relu_4_output.data->data,relu_2_output.data->data});
  forward_eltwise(&add_1_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(12, 1, { add_1_output.data->data});
}
void forward_lite_add_2(_stai_network_context* net_ctx)
{
  relu_7_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  relu_7_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  conv1d_7_output_array.data = AI_PTR(net_ctx->_activations[0] + 49408);
  conv1d_7_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 49408);
  add_2_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  add_2_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(19, 2, { relu_7_output.data->data,conv1d_7_output.data->data});
  forward_eltwise(&add_2_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(19, 1, { add_2_output.data->data});
}
void forward_lite_add_3(_stai_network_context* net_ctx)
{
  relu_10_output_array.data = AI_PTR(net_ctx->_activations[0] + 2672);
  relu_10_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2672);
  conv1d_10_output_array.data = AI_PTR(net_ctx->_activations[0] + 512);
  conv1d_10_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 512);
  add_3_output_array.data = AI_PTR(net_ctx->_activations[0] + 1120);
  add_3_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1120);
  _STAI_NETWORK_EVENT_NODE_START_CB(26, 2, { relu_10_output.data->data,conv1d_10_output.data->data});
  forward_eltwise(&add_3_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(26, 1, { add_3_output.data->data});
}
void forward_lite_output(_stai_network_context* net_ctx)
{
  relu_11_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  relu_11_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  val_16_array.data = AI_PTR(net_ctx->_weights[0] + 538440);
  val_16_array.data_start = AI_PTR(net_ctx->_weights[0] + 538440);
  output_output_array.data = AI_PTR(net_ctx->_outputs[0] + 0);
  output_output_array.data_start = AI_PTR(net_ctx->_outputs[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(28, 2, { relu_11_output.data->data,val_16.data->data});
  forward_gather(&output_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(28, 1, { output_output.data->data});
}

/*****************************************************************************/



static const ai_u32 conv1d_2_t_in_0_shape_ch_const_u32 = 55;
static const ai_u32 conv1d_2_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 conv1d_2_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_2_t_in_0_shape_h_const_u32 = 60;
static const ai_u32 conv1d_2_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_2_t_out_0_shape_h_const_u32 = 60;
static const ai_u32 conv1d_2_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_2_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 conv1d_2_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv1d_2_l_pad_H_0_const_s32 = 0;
static const ai_u16 conv1d_2_l_stride_1_const_u16 = 1;
static const ai_u16 conv1d_2_l_stride_0_const_u16 = 1;
static const ai_u16 conv1d_2_l_dilation_H_const_u16 = 1;
static const ai_u16 conv1d_2_l_dilation_W_const_u16 = 1;

static const ai_u32 conv1d_t_in_0_shape_ch_const_u32 = 55;
static const ai_u32 conv1d_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 conv1d_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_t_in_0_shape_h_const_u32 = 60;
static const ai_u32 conv1d_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_t_out_0_shape_h_const_u32 = 60;
static const ai_u32 conv1d_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 conv1d_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv1d_l_pad_H_0_const_s32 = 2;
static const ai_u16 conv1d_l_stride_1_const_u16 = 1;
static const ai_u16 conv1d_l_stride_0_const_u16 = 1;
static const ai_u16 conv1d_l_dilation_H_const_u16 = 1;
static const ai_u16 conv1d_l_dilation_W_const_u16 = 1;

static const ai_i32 relu_t_in_0_shape_ch_h_prod_const_s32 = 3840;

static const ai_u32 conv1d_1_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 conv1d_1_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 conv1d_1_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_1_t_in_0_shape_h_const_u32 = 60;
static const ai_u32 conv1d_1_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_1_t_out_0_shape_h_const_u32 = 60;
static const ai_u32 conv1d_1_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_1_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 conv1d_1_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv1d_1_l_pad_H_0_const_s32 = 2;
static const ai_u16 conv1d_1_l_stride_1_const_u16 = 1;
static const ai_u16 conv1d_1_l_stride_0_const_u16 = 1;
static const ai_u16 conv1d_1_l_dilation_H_const_u16 = 1;
static const ai_u16 conv1d_1_l_dilation_W_const_u16 = 1;

static const ai_i32 relu_1_t_in_0_shape_ch_h_prod_const_s32 = 3840;


static const ai_i32 relu_2_t_in_0_shape_ch_h_prod_const_s32 = 3840;

static const ai_u32 conv1d_3_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 conv1d_3_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 conv1d_3_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_3_t_in_0_shape_h_const_u32 = 60;
static const ai_u32 conv1d_3_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_3_t_out_0_shape_h_const_u32 = 62;
static const ai_u32 conv1d_3_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_3_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 conv1d_3_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv1d_3_l_pad_H_0_const_s32 = 4;
static const ai_u16 conv1d_3_l_stride_1_const_u16 = 1;
static const ai_u16 conv1d_3_l_stride_0_const_u16 = 1;
static const ai_u16 conv1d_3_l_dilation_H_const_u16 = 1;
static const ai_u16 conv1d_3_l_dilation_W_const_u16 = 1;

static const ai_i32 relu_3_t_in_0_shape_ch_h_prod_const_s32 = 3968;

static const ai_u32 conv1d_4_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 conv1d_4_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 conv1d_4_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_4_t_in_0_shape_h_const_u32 = 62;
static const ai_u32 conv1d_4_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_4_t_out_0_shape_h_const_u32 = 64;
static const ai_u32 conv1d_4_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_4_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 conv1d_4_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv1d_4_l_pad_H_0_const_s32 = 4;
static const ai_u16 conv1d_4_l_stride_1_const_u16 = 1;
static const ai_u16 conv1d_4_l_stride_0_const_u16 = 1;
static const ai_u16 conv1d_4_l_dilation_H_const_u16 = 1;
static const ai_u16 conv1d_4_l_dilation_W_const_u16 = 1;

static const ai_i32 relu_4_t_in_0_shape_ch_h_prod_const_s32 = 4096;


static const ai_i32 relu_5_t_in_0_shape_ch_h_prod_const_s32 = 4096;

static const ai_u32 conv1d_7_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 conv1d_7_t_out_0_shape_ch_const_u32 = 128;
static const ai_u32 conv1d_7_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_7_t_in_0_shape_h_const_u32 = 64;
static const ai_u32 conv1d_7_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_7_t_out_0_shape_h_const_u32 = 64;
static const ai_u32 conv1d_7_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_7_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 conv1d_7_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv1d_7_l_pad_H_0_const_s32 = 0;
static const ai_u16 conv1d_7_l_stride_1_const_u16 = 1;
static const ai_u16 conv1d_7_l_stride_0_const_u16 = 1;
static const ai_u16 conv1d_7_l_dilation_H_const_u16 = 1;
static const ai_u16 conv1d_7_l_dilation_W_const_u16 = 1;

static const ai_u32 conv1d_5_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 conv1d_5_t_out_0_shape_ch_const_u32 = 128;
static const ai_u32 conv1d_5_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_5_t_in_0_shape_h_const_u32 = 64;
static const ai_u32 conv1d_5_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_5_t_out_0_shape_h_const_u32 = 70;
static const ai_u32 conv1d_5_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_5_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 conv1d_5_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv1d_5_l_pad_H_0_const_s32 = 8;
static const ai_u16 conv1d_5_l_stride_1_const_u16 = 1;
static const ai_u16 conv1d_5_l_stride_0_const_u16 = 1;
static const ai_u16 conv1d_5_l_dilation_H_const_u16 = 1;
static const ai_u16 conv1d_5_l_dilation_W_const_u16 = 1;

static const ai_i32 relu_6_t_in_0_shape_ch_h_prod_const_s32 = 8960;

static const ai_u32 conv1d_6_t_in_0_shape_ch_const_u32 = 128;
static const ai_u32 conv1d_6_t_out_0_shape_ch_const_u32 = 128;
static const ai_u32 conv1d_6_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_6_t_in_0_shape_h_const_u32 = 70;
static const ai_u32 conv1d_6_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_6_t_out_0_shape_h_const_u32 = 76;
static const ai_u32 conv1d_6_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_6_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 conv1d_6_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv1d_6_l_pad_H_0_const_s32 = 8;
static const ai_u16 conv1d_6_l_stride_1_const_u16 = 1;
static const ai_u16 conv1d_6_l_stride_0_const_u16 = 1;
static const ai_u16 conv1d_6_l_dilation_H_const_u16 = 1;
static const ai_u16 conv1d_6_l_dilation_W_const_u16 = 1;

static const ai_i32 relu_7_t_in_0_shape_ch_h_prod_const_s32 = 9728;


static const ai_i32 relu_8_t_in_0_shape_ch_h_prod_const_s32 = 9728;

static const ai_u32 conv1d_10_t_in_0_shape_ch_const_u32 = 128;
static const ai_u32 conv1d_10_t_out_0_shape_ch_const_u32 = 2;
static const ai_u32 conv1d_10_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_10_t_in_0_shape_h_const_u32 = 76;
static const ai_u32 conv1d_10_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_10_t_out_0_shape_h_const_u32 = 76;
static const ai_u32 conv1d_10_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_10_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 conv1d_10_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv1d_10_l_pad_H_0_const_s32 = 0;
static const ai_u16 conv1d_10_l_stride_1_const_u16 = 1;
static const ai_u16 conv1d_10_l_stride_0_const_u16 = 1;
static const ai_u16 conv1d_10_l_dilation_H_const_u16 = 1;
static const ai_u16 conv1d_10_l_dilation_W_const_u16 = 1;

static const ai_u32 conv1d_8_t_in_0_shape_ch_const_u32 = 128;
static const ai_u32 conv1d_8_t_out_0_shape_ch_const_u32 = 2;
static const ai_u32 conv1d_8_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_8_t_in_0_shape_h_const_u32 = 76;
static const ai_u32 conv1d_8_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_8_t_out_0_shape_h_const_u32 = 90;
static const ai_u32 conv1d_8_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_8_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 conv1d_8_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv1d_8_l_pad_H_0_const_s32 = 16;
static const ai_u16 conv1d_8_l_stride_1_const_u16 = 1;
static const ai_u16 conv1d_8_l_stride_0_const_u16 = 1;
static const ai_u16 conv1d_8_l_dilation_H_const_u16 = 1;
static const ai_u16 conv1d_8_l_dilation_W_const_u16 = 1;

static const ai_i32 relu_9_t_in_0_shape_ch_h_prod_const_s32 = 180;

static const ai_u32 conv1d_9_t_in_0_shape_ch_const_u32 = 2;
static const ai_u32 conv1d_9_t_out_0_shape_ch_const_u32 = 2;
static const ai_u32 conv1d_9_t_in_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_9_t_in_0_shape_h_const_u32 = 90;
static const ai_u32 conv1d_9_t_out_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_9_t_out_0_shape_h_const_u32 = 104;
static const ai_u32 conv1d_9_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 conv1d_9_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 conv1d_9_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv1d_9_l_pad_H_0_const_s32 = 16;
static const ai_u16 conv1d_9_l_stride_1_const_u16 = 1;
static const ai_u16 conv1d_9_l_stride_0_const_u16 = 1;
static const ai_u16 conv1d_9_l_dilation_H_const_u16 = 1;
static const ai_u16 conv1d_9_l_dilation_W_const_u16 = 1;

static const ai_i32 relu_10_t_in_0_shape_ch_h_prod_const_s32 = 208;


static const ai_i32 relu_11_t_in_0_shape_ch_h_prod_const_s32 = 208;

STAI_API_ENTRY
stai_return_code stai_network_run(
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


  /* LITE_KERNEL_SECTION BEGIN input_Transpose */
  {
    
  forward_lite_input_Transpose(net_ctx);
  }
  /* LITE_KERNEL_SECTION END input_Transpose */
  /* LITE_KERNEL_SECTION BEGIN conv1d_2 */
  {
      const ai_float* conv1d_2_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 38548);
    ai_float* conv1d_2_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 51748);
    const ai_u8* conv1d_2_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 0);
    const ai_u8* conv1d_2_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 14080);
    ai_float* conv1d_2_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 38328);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(5, 1, {(stai_ptr) conv1d_2_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(conv1d_2_t_in_0_ptr_const_f32, conv1d_2_t_out_0_ptr_f32, conv1d_2_t_weight_0_ptr_const_u8, conv1d_2_t_weight_1_ptr_const_u8, conv1d_2_t_scratch_0_ptr_f32, conv1d_2_t_in_0_shape_ch_const_u32, conv1d_2_t_out_0_shape_ch_const_u32, conv1d_2_t_in_0_shape_w_const_u32, conv1d_2_t_in_0_shape_h_const_u32, conv1d_2_t_out_0_shape_w_const_u32, conv1d_2_t_out_0_shape_h_const_u32, conv1d_2_t_weight_0_shape_w_const_u32, conv1d_2_t_weight_0_shape_h_const_u32, conv1d_2_l_pad_W_0_const_s32, conv1d_2_l_pad_H_0_const_s32, conv1d_2_l_stride_1_const_u16, conv1d_2_l_stride_0_const_u16, 1, 1, conv1d_2_l_dilation_H_const_u16, conv1d_2_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(5, 1, {(stai_ptr) conv1d_2_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv1d_2 */
  /* LITE_KERNEL_SECTION BEGIN conv1d */
  {
      const ai_float* conv1d_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 38548);
    ai_float* conv1d_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 35328);
    const ai_u8* conv1d_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 14336);
    const ai_u8* conv1d_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 56576);
    ai_float* conv1d_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 67108);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(1, 1, {(stai_ptr) conv1d_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(conv1d_t_in_0_ptr_const_f32, conv1d_t_out_0_ptr_f32, conv1d_t_weight_0_ptr_const_u8, conv1d_t_weight_1_ptr_const_u8, conv1d_t_scratch_0_ptr_f32, conv1d_t_in_0_shape_ch_const_u32, conv1d_t_out_0_shape_ch_const_u32, conv1d_t_in_0_shape_w_const_u32, conv1d_t_in_0_shape_h_const_u32, conv1d_t_out_0_shape_w_const_u32, conv1d_t_out_0_shape_h_const_u32, conv1d_t_weight_0_shape_w_const_u32, conv1d_t_weight_0_shape_h_const_u32, conv1d_l_pad_W_0_const_s32, conv1d_l_pad_H_0_const_s32, conv1d_l_stride_1_const_u16, conv1d_l_stride_0_const_u16, 3, 1, conv1d_l_dilation_H_const_u16, conv1d_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(1, 1, {(stai_ptr) conv1d_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv1d */
  /* LITE_KERNEL_SECTION BEGIN relu */
  {
      ai_handle relu_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 35328);
    const ai_handle relu_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 35328);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(2, 1, {(stai_ptr) relu_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(relu_t_out_0_ptr_handle, relu_t_in_0_ptr_const_handle, relu_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(2, 1, {(stai_ptr) relu_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END relu */
  /* LITE_KERNEL_SECTION BEGIN conv1d_1 */
  {
      const ai_float* conv1d_1_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 35328);
    ai_float* conv1d_1_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 34304);
    const ai_u8* conv1d_1_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 56832);
    const ai_u8* conv1d_1_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 105984);
    ai_float* conv1d_1_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 67108);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(3, 1, {(stai_ptr) conv1d_1_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(conv1d_1_t_in_0_ptr_const_f32, conv1d_1_t_out_0_ptr_f32, conv1d_1_t_weight_0_ptr_const_u8, conv1d_1_t_weight_1_ptr_const_u8, conv1d_1_t_scratch_0_ptr_f32, conv1d_1_t_in_0_shape_ch_const_u32, conv1d_1_t_out_0_shape_ch_const_u32, conv1d_1_t_in_0_shape_w_const_u32, conv1d_1_t_in_0_shape_h_const_u32, conv1d_1_t_out_0_shape_w_const_u32, conv1d_1_t_out_0_shape_h_const_u32, conv1d_1_t_weight_0_shape_w_const_u32, conv1d_1_t_weight_0_shape_h_const_u32, conv1d_1_l_pad_W_0_const_s32, conv1d_1_l_pad_H_0_const_s32, conv1d_1_l_stride_1_const_u16, conv1d_1_l_stride_0_const_u16, 3, 1, conv1d_1_l_dilation_H_const_u16, conv1d_1_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(3, 1, {(stai_ptr) conv1d_1_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv1d_1 */
  /* LITE_KERNEL_SECTION BEGIN relu_1 */
  {
      ai_handle relu_1_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 34304);
    const ai_handle relu_1_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 34304);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(4, 1, {(stai_ptr) relu_1_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(relu_1_t_out_0_ptr_handle, relu_1_t_in_0_ptr_const_handle, relu_1_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(4, 1, {(stai_ptr) relu_1_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END relu_1 */
  /* LITE_KERNEL_SECTION BEGIN add */
  {
    
  forward_lite_add(net_ctx);
  }
  /* LITE_KERNEL_SECTION END add */
  /* LITE_KERNEL_SECTION BEGIN relu_2 */
  {
      ai_handle relu_2_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 49664);
    const ai_handle relu_2_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 34304);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(7, 1, {(stai_ptr) relu_2_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(relu_2_t_out_0_ptr_handle, relu_2_t_in_0_ptr_const_handle, relu_2_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(7, 1, {(stai_ptr) relu_2_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END relu_2 */
  /* LITE_KERNEL_SECTION BEGIN conv1d_3 */
  {
      const ai_float* conv1d_3_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 49664);
    ai_float* conv1d_3_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 33792);
    const ai_u8* conv1d_3_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 106240);
    const ai_u8* conv1d_3_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 155392);
    ai_float* conv1d_3_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 65024);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(8, 1, {(stai_ptr) conv1d_3_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(conv1d_3_t_in_0_ptr_const_f32, conv1d_3_t_out_0_ptr_f32, conv1d_3_t_weight_0_ptr_const_u8, conv1d_3_t_weight_1_ptr_const_u8, conv1d_3_t_scratch_0_ptr_f32, conv1d_3_t_in_0_shape_ch_const_u32, conv1d_3_t_out_0_shape_ch_const_u32, conv1d_3_t_in_0_shape_w_const_u32, conv1d_3_t_in_0_shape_h_const_u32, conv1d_3_t_out_0_shape_w_const_u32, conv1d_3_t_out_0_shape_h_const_u32, conv1d_3_t_weight_0_shape_w_const_u32, conv1d_3_t_weight_0_shape_h_const_u32, conv1d_3_l_pad_W_0_const_s32, conv1d_3_l_pad_H_0_const_s32, conv1d_3_l_stride_1_const_u16, conv1d_3_l_stride_0_const_u16, 3, 1, conv1d_3_l_dilation_H_const_u16, conv1d_3_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(8, 1, {(stai_ptr) conv1d_3_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv1d_3 */
  /* LITE_KERNEL_SECTION BEGIN relu_3 */
  {
      ai_handle relu_3_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 33792);
    const ai_handle relu_3_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 33792);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(9, 1, {(stai_ptr) relu_3_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(relu_3_t_out_0_ptr_handle, relu_3_t_in_0_ptr_const_handle, relu_3_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(9, 1, {(stai_ptr) relu_3_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END relu_3 */
  /* LITE_KERNEL_SECTION BEGIN conv1d_4 */
  {
      const ai_float* conv1d_4_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 33792);
    ai_float* conv1d_4_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 65024);
    const ai_u8* conv1d_4_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 155648);
    const ai_u8* conv1d_4_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 204800);
    ai_float* conv1d_4_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 33024);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(10, 1, {(stai_ptr) conv1d_4_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(conv1d_4_t_in_0_ptr_const_f32, conv1d_4_t_out_0_ptr_f32, conv1d_4_t_weight_0_ptr_const_u8, conv1d_4_t_weight_1_ptr_const_u8, conv1d_4_t_scratch_0_ptr_f32, conv1d_4_t_in_0_shape_ch_const_u32, conv1d_4_t_out_0_shape_ch_const_u32, conv1d_4_t_in_0_shape_w_const_u32, conv1d_4_t_in_0_shape_h_const_u32, conv1d_4_t_out_0_shape_w_const_u32, conv1d_4_t_out_0_shape_h_const_u32, conv1d_4_t_weight_0_shape_w_const_u32, conv1d_4_t_weight_0_shape_h_const_u32, conv1d_4_l_pad_W_0_const_s32, conv1d_4_l_pad_H_0_const_s32, conv1d_4_l_stride_1_const_u16, conv1d_4_l_stride_0_const_u16, 3, 1, conv1d_4_l_dilation_H_const_u16, conv1d_4_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(10, 1, {(stai_ptr) conv1d_4_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv1d_4 */
  /* LITE_KERNEL_SECTION BEGIN relu_4 */
  {
      ai_handle relu_4_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 65024);
    const ai_handle relu_4_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 65024);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(11, 1, {(stai_ptr) relu_4_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(relu_4_t_out_0_ptr_handle, relu_4_t_in_0_ptr_const_handle, relu_4_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(11, 1, {(stai_ptr) relu_4_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END relu_4 */
  /* LITE_KERNEL_SECTION BEGIN add_1 */
  {
    
  forward_lite_add_1(net_ctx);
  }
  /* LITE_KERNEL_SECTION END add_1 */
  /* LITE_KERNEL_SECTION BEGIN relu_5 */
  {
      ai_handle relu_5_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 33024);
    const ai_handle relu_5_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 65024);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(13, 1, {(stai_ptr) relu_5_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(relu_5_t_out_0_ptr_handle, relu_5_t_in_0_ptr_const_handle, relu_5_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(13, 1, {(stai_ptr) relu_5_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END relu_5 */
  /* LITE_KERNEL_SECTION BEGIN conv1d_7 */
  {
      const ai_float* conv1d_7_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 33024);
    ai_float* conv1d_7_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 49408);
    const ai_u8* conv1d_7_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 205056);
    const ai_u8* conv1d_7_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 237824);
    ai_float* conv1d_7_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 32768);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(18, 1, {(stai_ptr) conv1d_7_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(conv1d_7_t_in_0_ptr_const_f32, conv1d_7_t_out_0_ptr_f32, conv1d_7_t_weight_0_ptr_const_u8, conv1d_7_t_weight_1_ptr_const_u8, conv1d_7_t_scratch_0_ptr_f32, conv1d_7_t_in_0_shape_ch_const_u32, conv1d_7_t_out_0_shape_ch_const_u32, conv1d_7_t_in_0_shape_w_const_u32, conv1d_7_t_in_0_shape_h_const_u32, conv1d_7_t_out_0_shape_w_const_u32, conv1d_7_t_out_0_shape_h_const_u32, conv1d_7_t_weight_0_shape_w_const_u32, conv1d_7_t_weight_0_shape_h_const_u32, conv1d_7_l_pad_W_0_const_s32, conv1d_7_l_pad_H_0_const_s32, conv1d_7_l_stride_1_const_u16, conv1d_7_l_stride_0_const_u16, 1, 1, conv1d_7_l_dilation_H_const_u16, conv1d_7_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(18, 1, {(stai_ptr) conv1d_7_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv1d_7 */
  /* LITE_KERNEL_SECTION BEGIN conv1d_5 */
  {
      const ai_float* conv1d_5_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 33024);
    ai_float* conv1d_5_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 8192);
    const ai_u8* conv1d_5_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 238336);
    const ai_u8* conv1d_5_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 336640);
    ai_float* conv1d_5_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 82176);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(14, 1, {(stai_ptr) conv1d_5_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(conv1d_5_t_in_0_ptr_const_f32, conv1d_5_t_out_0_ptr_f32, conv1d_5_t_weight_0_ptr_const_u8, conv1d_5_t_weight_1_ptr_const_u8, conv1d_5_t_scratch_0_ptr_f32, conv1d_5_t_in_0_shape_ch_const_u32, conv1d_5_t_out_0_shape_ch_const_u32, conv1d_5_t_in_0_shape_w_const_u32, conv1d_5_t_in_0_shape_h_const_u32, conv1d_5_t_out_0_shape_w_const_u32, conv1d_5_t_out_0_shape_h_const_u32, conv1d_5_t_weight_0_shape_w_const_u32, conv1d_5_t_weight_0_shape_h_const_u32, conv1d_5_l_pad_W_0_const_s32, conv1d_5_l_pad_H_0_const_s32, conv1d_5_l_stride_1_const_u16, conv1d_5_l_stride_0_const_u16, 3, 1, conv1d_5_l_dilation_H_const_u16, conv1d_5_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(14, 1, {(stai_ptr) conv1d_5_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv1d_5 */
  /* LITE_KERNEL_SECTION BEGIN relu_6 */
  {
      ai_handle relu_6_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 8192);
    const ai_handle relu_6_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 8192);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(15, 1, {(stai_ptr) relu_6_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(relu_6_t_out_0_ptr_handle, relu_6_t_in_0_ptr_const_handle, relu_6_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(15, 1, {(stai_ptr) relu_6_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END relu_6 */
  /* LITE_KERNEL_SECTION BEGIN conv1d_6 */
  {
      const ai_float* conv1d_6_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 8192);
    ai_float* conv1d_6_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
    const ai_u8* conv1d_6_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 337152);
    const ai_u8* conv1d_6_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 533760);
    ai_float* conv1d_6_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 47872);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(16, 1, {(stai_ptr) conv1d_6_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(conv1d_6_t_in_0_ptr_const_f32, conv1d_6_t_out_0_ptr_f32, conv1d_6_t_weight_0_ptr_const_u8, conv1d_6_t_weight_1_ptr_const_u8, conv1d_6_t_scratch_0_ptr_f32, conv1d_6_t_in_0_shape_ch_const_u32, conv1d_6_t_out_0_shape_ch_const_u32, conv1d_6_t_in_0_shape_w_const_u32, conv1d_6_t_in_0_shape_h_const_u32, conv1d_6_t_out_0_shape_w_const_u32, conv1d_6_t_out_0_shape_h_const_u32, conv1d_6_t_weight_0_shape_w_const_u32, conv1d_6_t_weight_0_shape_h_const_u32, conv1d_6_l_pad_W_0_const_s32, conv1d_6_l_pad_H_0_const_s32, conv1d_6_l_stride_1_const_u16, conv1d_6_l_stride_0_const_u16, 3, 1, conv1d_6_l_dilation_H_const_u16, conv1d_6_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(16, 1, {(stai_ptr) conv1d_6_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv1d_6 */
  /* LITE_KERNEL_SECTION BEGIN relu_7 */
  {
      ai_handle relu_7_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 0);
    const ai_handle relu_7_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(17, 1, {(stai_ptr) relu_7_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(relu_7_t_out_0_ptr_handle, relu_7_t_in_0_ptr_const_handle, relu_7_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(17, 1, {(stai_ptr) relu_7_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END relu_7 */
  /* LITE_KERNEL_SECTION BEGIN add_2 */
  {
    
  forward_lite_add_2(net_ctx);
  }
  /* LITE_KERNEL_SECTION END add_2 */
  /* LITE_KERNEL_SECTION BEGIN relu_8 */
  {
      ai_handle relu_8_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 38912);
    const ai_handle relu_8_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(20, 1, {(stai_ptr) relu_8_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(relu_8_t_out_0_ptr_handle, relu_8_t_in_0_ptr_const_handle, relu_8_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(20, 1, {(stai_ptr) relu_8_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END relu_8 */
  /* LITE_KERNEL_SECTION BEGIN conv1d_10 */
  {
      const ai_float* conv1d_10_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 38912);
    ai_float* conv1d_10_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 512);
    const ai_u8* conv1d_10_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 534272);
    const ai_u8* conv1d_10_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 535296);
    ai_float* conv1d_10_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(25, 1, {(stai_ptr) conv1d_10_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(conv1d_10_t_in_0_ptr_const_f32, conv1d_10_t_out_0_ptr_f32, conv1d_10_t_weight_0_ptr_const_u8, conv1d_10_t_weight_1_ptr_const_u8, conv1d_10_t_scratch_0_ptr_f32, conv1d_10_t_in_0_shape_ch_const_u32, conv1d_10_t_out_0_shape_ch_const_u32, conv1d_10_t_in_0_shape_w_const_u32, conv1d_10_t_in_0_shape_h_const_u32, conv1d_10_t_out_0_shape_w_const_u32, conv1d_10_t_out_0_shape_h_const_u32, conv1d_10_t_weight_0_shape_w_const_u32, conv1d_10_t_weight_0_shape_h_const_u32, conv1d_10_l_pad_W_0_const_s32, conv1d_10_l_pad_H_0_const_s32, conv1d_10_l_stride_1_const_u16, conv1d_10_l_stride_0_const_u16, 1, 1, conv1d_10_l_dilation_H_const_u16, conv1d_10_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(25, 1, {(stai_ptr) conv1d_10_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv1d_10 */
  /* LITE_KERNEL_SECTION BEGIN conv1d_8 */
  {
      const ai_float* conv1d_8_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 38912);
    ai_float* conv1d_8_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 2656);
    const ai_u8* conv1d_8_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 535304);
    const ai_u8* conv1d_8_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 538376);
    ai_float* conv1d_8_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 1120);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(21, 1, {(stai_ptr) conv1d_8_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(conv1d_8_t_in_0_ptr_const_f32, conv1d_8_t_out_0_ptr_f32, conv1d_8_t_weight_0_ptr_const_u8, conv1d_8_t_weight_1_ptr_const_u8, conv1d_8_t_scratch_0_ptr_f32, conv1d_8_t_in_0_shape_ch_const_u32, conv1d_8_t_out_0_shape_ch_const_u32, conv1d_8_t_in_0_shape_w_const_u32, conv1d_8_t_in_0_shape_h_const_u32, conv1d_8_t_out_0_shape_w_const_u32, conv1d_8_t_out_0_shape_h_const_u32, conv1d_8_t_weight_0_shape_w_const_u32, conv1d_8_t_weight_0_shape_h_const_u32, conv1d_8_l_pad_W_0_const_s32, conv1d_8_l_pad_H_0_const_s32, conv1d_8_l_stride_1_const_u16, conv1d_8_l_stride_0_const_u16, 3, 1, conv1d_8_l_dilation_H_const_u16, conv1d_8_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(21, 1, {(stai_ptr) conv1d_8_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv1d_8 */
  /* LITE_KERNEL_SECTION BEGIN relu_9 */
  {
      ai_handle relu_9_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 1120);
    const ai_handle relu_9_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 2656);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(22, 1, {(stai_ptr) relu_9_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(relu_9_t_out_0_ptr_handle, relu_9_t_in_0_ptr_const_handle, relu_9_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(22, 1, {(stai_ptr) relu_9_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END relu_9 */
  /* LITE_KERNEL_SECTION BEGIN conv1d_9 */
  {
      const ai_float* conv1d_9_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 1120);
    ai_float* conv1d_9_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 1840);
    const ai_u8* conv1d_9_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 538384);
    const ai_u8* conv1d_9_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 538432);
    ai_float* conv1d_9_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(23, 1, {(stai_ptr) conv1d_9_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(conv1d_9_t_in_0_ptr_const_f32, conv1d_9_t_out_0_ptr_f32, conv1d_9_t_weight_0_ptr_const_u8, conv1d_9_t_weight_1_ptr_const_u8, conv1d_9_t_scratch_0_ptr_f32, conv1d_9_t_in_0_shape_ch_const_u32, conv1d_9_t_out_0_shape_ch_const_u32, conv1d_9_t_in_0_shape_w_const_u32, conv1d_9_t_in_0_shape_h_const_u32, conv1d_9_t_out_0_shape_w_const_u32, conv1d_9_t_out_0_shape_h_const_u32, conv1d_9_t_weight_0_shape_w_const_u32, conv1d_9_t_weight_0_shape_h_const_u32, conv1d_9_l_pad_W_0_const_s32, conv1d_9_l_pad_H_0_const_s32, conv1d_9_l_stride_1_const_u16, conv1d_9_l_stride_0_const_u16, 3, 1, conv1d_9_l_dilation_H_const_u16, conv1d_9_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(23, 1, {(stai_ptr) conv1d_9_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conv1d_9 */
  /* LITE_KERNEL_SECTION BEGIN relu_10 */
  {
      ai_handle relu_10_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 2672);
    const ai_handle relu_10_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 1840);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(24, 1, {(stai_ptr) relu_10_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(relu_10_t_out_0_ptr_handle, relu_10_t_in_0_ptr_const_handle, relu_10_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(24, 1, {(stai_ptr) relu_10_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END relu_10 */
  /* LITE_KERNEL_SECTION BEGIN add_3 */
  {
    
  forward_lite_add_3(net_ctx);
  }
  /* LITE_KERNEL_SECTION END add_3 */
  /* LITE_KERNEL_SECTION BEGIN relu_11 */
  {
      ai_handle relu_11_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 0);
    const ai_handle relu_11_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 1120);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(27, 1, {(stai_ptr) relu_11_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(relu_11_t_out_0_ptr_handle, relu_11_t_in_0_ptr_const_handle, relu_11_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(27, 1, {(stai_ptr) relu_11_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END relu_11 */
  /* LITE_KERNEL_SECTION BEGIN output */
  {
    
  forward_lite_output(net_ctx);
  }
  /* LITE_KERNEL_SECTION END output */
  return net_ctx->_return_code;
}

/*****************************************************************************/
/*  Getters APIs Section  */
STAI_API_ENTRY
stai_size stai_network_get_context_size()
{
  return (stai_size)STAI_NETWORK_CONTEXT_SIZE;
}

#if defined(HAVE_NETWORK_INFO)
STAI_API_ENTRY
stai_return_code stai_network_get_info(
  stai_network* network,
  stai_network_info* info)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, info==NULL, STAI_ERROR_NETWORK_INVALID_INFO, net_ctx->_return_code)

  // Copy of network info struct
  *info = g_network_info;

  return STAI_SUCCESS;
}
#endif


STAI_API_ENTRY
stai_return_code stai_network_get_activations(
  stai_network* network, stai_ptr* activations, stai_size* n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, !n_activations, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_activations = STAI_NETWORK_ACTIVATIONS_NUM;
for (stai_size idx=0; activations && (idx<STAI_NETWORK_ACTIVATIONS_NUM); idx++) {
    // get address of the activations buffers
    activations[idx] = net_ctx->_activations[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_weights(
  stai_network* network, stai_ptr* weights, stai_size* n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_weights, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_weights = STAI_NETWORK_WEIGHTS_NUM;
for (stai_size idx=0; weights && (idx<STAI_NETWORK_WEIGHTS_NUM); idx++) {
    // get address of the weights buffers
    weights[idx] = net_ctx->_weights[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_inputs(
  stai_network* network, stai_ptr* inputs, stai_size* n_inputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_inputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_inputs = STAI_NETWORK_IN_NUM;
  for (stai_size idx=0; inputs && (idx<STAI_NETWORK_IN_NUM); idx++) {
    inputs[idx] = net_ctx->_inputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_outputs(
  stai_network* network, stai_ptr* outputs, stai_size* n_outputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_outputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_outputs = STAI_NETWORK_OUT_NUM;
  for (stai_size idx=0; outputs && (idx<STAI_NETWORK_OUT_NUM); idx++) {
    outputs[idx] = net_ctx->_outputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_error(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /* return 1st generated error or STAI_SUCCESS if no errors so far */
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_states(
  stai_network* network, stai_ptr* states, stai_size* n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_states, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  /* get the number of internals states (supporting multi-heap also for internal states) */
  *n_states = STAI_NETWORK_STATES_NUM;

  STAI_UNUSED(states)
return net_ctx->_return_code;
}


/*****************************************************************************/
/*  Setters APIs Section  */

STAI_API_ENTRY
stai_return_code stai_network_set_activations(
  stai_network* network,
  const stai_ptr* activations,
  const stai_size n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _activations_alignment[] = STAI_NETWORK_ACTIVATIONS_ALIGNMENTS;
  STAI_PRINT("  [stai_network_set_activations] network(%p) activations[%d]: %p\n\n", net_ctx, n_activations, activations)
  _STAI_SET_ERROR(net_ctx, !activations,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_activations!=STAI_NETWORK_ACTIVATIONS_NUM,
                  STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_NUM, net_ctx->_return_code)

  for (stai_size idx=0; activations && idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    STAI_PRINT("  activation[%d]: %p\n", idx, activations[idx])
    _STAI_SET_ERROR(net_ctx, activations[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)activations[idx]) & (_activations_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_activations[idx] = activations[idx];
  }
  net_ctx->_inputs[0] = activations[0] + 51748;

  net_ctx->_outputs[0] = activations[0] + 832;
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_weights(
  stai_network* network,
  const stai_ptr* weights,
  const stai_size n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _weights_alignment[] = STAI_NETWORK_WEIGHTS_ALIGNMENTS;
  _STAI_SET_ERROR(net_ctx, !weights,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_weights!=STAI_NETWORK_WEIGHTS_NUM,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_NUM, net_ctx->_return_code)
  for (stai_size idx=0; weights && idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    STAI_PRINT("  weight[%d]: %p\n", idx, weights[idx])
    _STAI_SET_ERROR(net_ctx, weights[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)weights[idx]) & (_weights_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_weights[idx] = weights[idx];
  }_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_inputs(
  stai_network* network,
  const stai_ptr* inputs,
  const stai_size n_inputs)
{
  const uintptr_t _inputs_alignment[] = STAI_NETWORK_IN_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !inputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_inputs!=STAI_NETWORK_IN_NUM,
                  STAI_ERROR_NETWORK_INVALID_IN_NUM, net_ctx->_return_code)

  for (stai_size idx=0; inputs && idx<STAI_NETWORK_IN_NUM; idx++) {
    STAI_PRINT("  input[%d]: %p\n", idx, inputs[idx])
    _STAI_SET_ERROR(net_ctx, inputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)inputs[idx]) & (_inputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_inputs[idx] = inputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_outputs(
  stai_network* network,
  const stai_ptr* outputs,
  const stai_size n_outputs)
{
  const uintptr_t _outputs_alignment[] = STAI_NETWORK_OUT_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !outputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_outputs!=STAI_NETWORK_OUT_NUM,
                  STAI_ERROR_NETWORK_INVALID_OUT_NUM, net_ctx->_return_code)

  for (stai_size idx=0; outputs && idx<n_outputs; idx++) {
    STAI_PRINT("  output[%d]: %p\n", idx, outputs[idx])
    _STAI_SET_ERROR(net_ctx, outputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)outputs[idx]) & (_outputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_outputs[idx] = outputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_states(
  stai_network* network,
  const stai_ptr* states,
  const stai_size n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  STAI_UNUSED(states)
  STAI_UNUSED(n_states)
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}

STAI_API_ENTRY
stai_return_code stai_network_set_callback(
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
#undef _STAI_NETWORK_EVENT_NODE_START_CB
#undef _STAI_NETWORK_EVENT_NODE_STOP_CB
#undef _STAI_NETWORK_MODEL_SIGNATURE
#undef _STAI_NETWORK_DATETIME
#undef _STAI_NETWORK_COMPILE_DATETIME

