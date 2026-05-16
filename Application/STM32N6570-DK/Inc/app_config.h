 /**
 ******************************************************************************
 * @file    app_config.h
 * @author  GPM Application Team
 *
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2023 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */

#ifndef APP_CONFIG
#define APP_CONFIG

#include "stai_network.h"
#include "arm_math.h"
#include "stm32_lcd.h"

#define USE_DCACHE

/*Defines: CMW_MIRRORFLIP_NONE; CMW_MIRRORFLIP_FLIP; CMW_MIRRORFLIP_MIRROR; CMW_MIRRORFLIP_FLIP_MIRROR;*/
#define CAMERA_FLIP CMW_MIRRORFLIP_NONE

#define ASPECT_RATIO_CROP       (1) /* Crop both pipes to nn input aspect ratio; Original aspect ratio kept */
#define ASPECT_RATIO_FIT        (2) /* Resize both pipe to NN input aspect ratio; Original aspect ratio not kept */
#define ASPECT_RATIO_FULLSCREEN (3) /* Resize camera image to NN input size and display a maximized image. See Doc/Build-Options.md#aspect-ratio-mode */
#define ASPECT_RATIO_MODE ASPECT_RATIO_CROP

/* Fall detection model selection: change this line to switch between TCN and GRU */
#define FALL_MODEL_TCN  0
#define FALL_MODEL_GRU  1
#define FALL_DETECTION_MODEL  FALL_MODEL_GRU

/* GRU window-based (1×40×27): window fills before inference begins (= POSE_WINDOW_SIZE = 40) */
#define GRU_WARMUP_FRAMES       40
/* Softmax fall score must reach this to count as a fall frame — P27-vm0 INT8 val reselection */
#define GRU_FALL_SCORE_THRESHOLD  0.50f
/* Consecutive fall windows required to trigger alarm + pipeline reset — P27-vm0 min_consecutive=1 */
#define GRU_FALL_RESET_COUNT    1
/* Keep showing FALL on display and sounding alarm for this many ms after confirmation */
#define GRU_FALL_LATCH_MS       5000
/* LED_RED blink half-period during alarm (ms) */
#define GRU_ALARM_BLINK_PERIOD_MS  250
/* Minimum postprocessed keypoint confidence required to run fall detection */
#define FALL_PERSON_CONF_THRESHOLD  (0.08f)
/* Max keypoint bounding-box span (normalized 0-1) before skipping visualization */
#define FALL_KP_SPREAD_MAX          (0.85f)
/* Consecutive postprocessed frames below the person threshold before reset */
#define FALL_PERSON_MISSING_RESET_COUNT  45
/* Consecutive postprocessed frames above the person threshold before confirming entry */
#define FALL_PERSON_ENTRY_CONF_COUNT     3

/* Model Related Info */
#define POSTPROCESS_TYPE    POSTPROCESS_SPE_MOVENET_UI

#define COLOR_BGR (0)
#define COLOR_RGB (1)
#define COLOR_MODE    COLOR_RGB

/* I/O configuration */
#define AI_SPE_MOVENET_POSTPROC_HEATMAP_WIDTH        (STAI_NETWORK_IN_1_WIDTH/4)
#define AI_SPE_MOVENET_POSTPROC_HEATMAP_HEIGHT       (STAI_NETWORK_IN_1_HEIGHT/4)

/* Post processing values */
#define AI_POSE_PP_CONF_THRESHOLD           (0.10f)
#define AI_POSE_PP_POSE_KEYPOINTS_NB        (17)

#define USE_BINDINGS
#define BINDINGS_NB (18)
#define BINDINGS const int bindings[BINDINGS_NB][3] = {\
    { 15, 13, UTIL_LCD_COLOR_ORANGE },\
    { 13, 11, UTIL_LCD_COLOR_ORANGE },\
    { 16, 14, UTIL_LCD_COLOR_ORANGE },\
    { 14, 12, UTIL_LCD_COLOR_ORANGE },\
    { 11, 12, UTIL_LCD_COLOR_MAGENTA },\
    { 5, 11, UTIL_LCD_COLOR_MAGENTA },\
    { 6, 12, UTIL_LCD_COLOR_MAGENTA },\
    { 5, 6, UTIL_LCD_COLOR_GREEN },\
    { 5, 7, UTIL_LCD_COLOR_BLUE },\
    { 7, 9, UTIL_LCD_COLOR_BLUE },\
    { 6, 8, UTIL_LCD_COLOR_BLUE },\
    { 8, 10, UTIL_LCD_COLOR_BLUE },\
    { 0, 1, UTIL_LCD_COLOR_GREEN },\
    { 0, 2, UTIL_LCD_COLOR_GREEN },\
    { 1, 3, UTIL_LCD_COLOR_BLUE },\
    { 2, 4, UTIL_LCD_COLOR_BLUE },\
    { 0, 5, UTIL_LCD_COLOR_MAGENTA },\
    { 0, 6, UTIL_LCD_COLOR_MAGENTA },\
}\

extern const int bindings[BINDINGS_NB][3];

/* Display */
#define WELCOME_MSG_1         "st_movenet_lightning_a100_heatmaps_256_int8.tflite"
#define WELCOME_MSG_2         "Model Running in STM32 MCU internal memory"

#endif
