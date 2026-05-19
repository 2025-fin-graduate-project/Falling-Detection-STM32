/**
 ******************************************************************************
 * @file    ui_system.c
 * @brief   UI system implementation for LCD rendering and visualization.
 ******************************************************************************
 */

#include "ui_system.h"
#include "app_config.h"
#include "app_camerapipeline.h"
#include "stm32n6570_discovery_lcd.h"
#include "stm32_lcd.h"
#include "stm32_lcd_ex.h"
#include "display_spe.h"
#include "stlogo.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>

#define LCD_FG_WIDTH  SCREEN_WIDTH
#define LCD_FG_HEIGHT SCREEN_HEIGHT
#define LCD_FG_FRAMEBUFFER_SIZE  (LCD_FG_WIDTH * LCD_FG_HEIGHT * 2)

typedef struct
{
  uint32_t X0;
  uint32_t Y0;
  uint32_t XSize;
  uint32_t YSize;
} Rectangle_t;

/* LCD State */
static Rectangle_t lcd_bg_area;
static Rectangle_t lcd_fg_area = { 0, 0, LCD_FG_WIDTH, LCD_FG_HEIGHT };

__attribute__ ((section (".psram_bss"))) __attribute__ ((aligned (32)))
static uint8_t lcd_bg_buffer[800 * 480 * 2];

__attribute__ ((section (".psram_bss"))) __attribute__ ((aligned (32)))
static uint8_t lcd_fg_buffer[2][LCD_FG_WIDTH * LCD_FG_HEIGHT * 2];
static int lcd_fg_buffer_rd_idx = 0;

/* Internal helpers */
static void Display_WelcomeScreen(void);
static int clamp_point(int *x, int *y);
static void convert_length(float32_t wi, float32_t hi, int *wo, int *ho);
static void convert_point(float32_t xi, float32_t yi, int *xo, int *yo);
static void Display_binding_line(int x0, int y0, int x1, int y1, uint32_t color);

void UI_Init(uint32_t bg_width, uint32_t bg_height)
{
  lcd_bg_area.XSize = bg_width;
  lcd_bg_area.YSize = bg_height;
#if ASPECT_RATIO_MODE == ASPECT_RATIO_CROP || ASPECT_RATIO_MODE == ASPECT_RATIO_FIT
  lcd_bg_area.X0 = (LCD_FG_WIDTH - bg_width) / 2;
#else
  lcd_bg_area.X0 = 0;
#endif
  lcd_bg_area.Y0 = 0;

  BSP_LCD_Init(0, LCD_ORIENTATION_LANDSCAPE);
  BSP_LCD_LayerConfig_t LConfig = {0};

  /* Layer 1: Background (Camera) */
  LConfig.X0 = lcd_bg_area.X0; LConfig.Y0 = lcd_bg_area.Y0;
  LConfig.X1 = lcd_bg_area.X0 + lcd_bg_area.XSize; LConfig.Y1 = lcd_bg_area.Y0 + lcd_bg_area.YSize;
  LConfig.PixelFormat = LCD_PIXEL_FORMAT_RGB565;
  LConfig.Address = (uint32_t) lcd_bg_buffer;
  BSP_LCD_ConfigLayer(0, LTDC_LAYER_1, &LConfig);

  /* Layer 2: Foreground (UI/Skeleton) */
  LConfig.X0 = lcd_fg_area.X0; LConfig.Y0 = lcd_fg_area.Y0;
  LConfig.X1 = lcd_fg_area.X0 + lcd_fg_area.XSize; LConfig.Y1 = lcd_fg_area.Y0 + lcd_fg_area.YSize;
  LConfig.PixelFormat = LCD_PIXEL_FORMAT_ARGB4444;
  LConfig.Address = (uint32_t) lcd_fg_buffer;
  BSP_LCD_ConfigLayer(0, LTDC_LAYER_2, &LConfig);

  UTIL_LCD_SetFuncDriver(&LCD_Driver);
  UTIL_LCD_SetLayer(LTDC_LAYER_2);
  UTIL_LCD_Clear(0x00000000);
  UTIL_LCD_SetFont(&Font20);
  UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_WHITE);

  Display_spe_InitFunctions(clamp_point, convert_length, convert_point, Display_binding_line);
}

uint8_t* UI_GetBgBuffer(void)
{
  return lcd_bg_buffer;
}

void UI_Update(spe_pp_out_t *pp_out, 
               VisionMetrics_t *v_metrics, 
               const PresenceStatus_t *p_status,
               const FallDetectionState_t *f_state)
{
  HAL_LTDC_SetAddress_NoReload(&hlcd_ltdc, (uint32_t) lcd_fg_buffer[lcd_fg_buffer_rd_idx], LTDC_LAYER_2);
  UTIL_LCD_FillRect(lcd_fg_area.X0, lcd_fg_area.Y0, lcd_fg_area.XSize, lcd_fg_area.YSize, 0x00000000);

  /* Draw skeleton if valid */
  if (pp_out != NULL && v_metrics->draw_keypoints) {
    Display_spe_Detection(pp_out->pOutBuff);
  }

  UTIL_LCD_SetBackColor(0x40000000);

  if (!v_metrics->postprocess_valid) {
    UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_GRAY);
    UTIL_LCDEx_PrintfAt(0, LINE(1), CENTER_MODE, "postprocess error PP %ld", (long)v_metrics->postprocess_status);
  }
  else if (p_status->person_missing_confirmed) {
    UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_GRAY);
    UTIL_LCDEx_PrintfAt(0, LINE(1), CENTER_MODE, "no person");
  }
  else {
    uint8_t latch_active = (f_state->fall_latch_tick != 0u) &&
                           ((HAL_GetTick() - f_state->fall_latch_tick) < GRU_FALL_LATCH_MS);

    if (f_state->window_ready || latch_active) {
      uint8_t show_fall = f_state->fall_detected || latch_active;
      UTIL_LCD_SetTextColor(show_fall ? UTIL_LCD_COLOR_RED : UTIL_LCD_COLOR_GREEN);
      UTIL_LCDEx_PrintfAt(0, LINE(1), CENTER_MODE, "%s Fall %.2f Normal %.2f %s %lums",
                          show_fall ? "FALL" : "NORMAL",
                          (double)f_state->fall_score,
                          (double)f_state->normal_score,
                          "GRU",
                          f_state->inference_ms);

      if (latch_active) {
        UTIL_LCD_FillRect(0, 185, LCD_FG_WIDTH, 110, 0xC0DD0000);
        UTIL_LCD_SetFont(&Font24);
        UTIL_LCD_SetBackColor(0x00000000);
        UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_WHITE);
        UTIL_LCDEx_PrintfAt(0, 228, CENTER_MODE, "!! FALL DETECTED !!");
        UTIL_LCD_SetFont(&Font20);
        UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_YELLOW);
        UTIL_LCDEx_PrintfAt(0, 258, CENTER_MODE, "Score %.0f%%  (threshold %.0f%%)",
                            (double)(f_state->fall_score * 100.0f),
                            (double)(GRU_FALL_SCORE_THRESHOLD * 100.0f));
      }
    }
    else {
      UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_YELLOW);
      UTIL_LCDEx_PrintfAt(0, LINE(1), CENTER_MODE, "Fall detector warming %lu/%u",
                          f_state->frame_count, POSE_WINDOW_SIZE);
    }
  }

  /* Performance and Debug Info */
  UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_WHITE);
  UTIL_LCDEx_PrintfAt(0, LINE(2), CENTER_MODE, "KP %lu/%u P%lu M%lu Max %.2f",
                      v_metrics->visible_keypoints, (uint32_t)AI_POSE_PP_POSE_KEYPOINTS_NB,
                      v_metrics->person_keypoints, p_status->person_missing_count,
                      (double)v_metrics->max_keypoint_proba);
  
  UTIL_LCDEx_PrintfAt(0, LINE(3), CENTER_MODE, "Raw[%d,%d] Deq[%.2f,%.2f]",
                      v_metrics->raw_min, v_metrics->raw_max,
                      (double)v_metrics->raw_min_dequant, (double)v_metrics->raw_max_dequant);
  
  UTIL_LCDEx_PrintfAt(0, LINE(4), CENTER_MODE, "Out %lux%lux%lu PP %ld",
                      v_metrics->output_width, v_metrics->output_height, v_metrics->output_channels,
                      (long)v_metrics->postprocess_status);
  
  UTIL_LCDEx_PrintfAt(0, LINE(20), CENTER_MODE, "Inference: %ums", v_metrics->inference_ms);

  Display_WelcomeScreen();

  SCB_CleanDCache_by_Addr(lcd_fg_buffer[lcd_fg_buffer_rd_idx], LCD_FG_FRAMEBUFFER_SIZE);
  HAL_LTDC_ReloadLayer(&hlcd_ltdc, LTDC_RELOAD_VERTICAL_BLANKING, LTDC_LAYER_2);
  lcd_fg_buffer_rd_idx = 1 - lcd_fg_buffer_rd_idx;
}

static void Display_WelcomeScreen(void)
{
  static uint32_t t0 = 0;
  if (t0 == 0) t0 = HAL_GetTick();
  if (HAL_GetTick() - t0 < 4000)
  {
    UTIL_LCD_FillRGBRect(300, 100, (uint8_t *) stlogo, 200, 107);
    UTIL_LCD_SetBackColor(0x40000000);
    UTIL_LCDEx_PrintfAt(0, LINE(16), CENTER_MODE, "Pose Estimation");
    UTIL_LCDEx_PrintfAt(0, LINE(17), CENTER_MODE, WELCOME_MSG_1);
    UTIL_LCDEx_PrintfAt(0, LINE(18), CENTER_MODE, WELCOME_MSG_2);
    UTIL_LCD_SetBackColor(0);
  }
}

/* Coordinate conversion bindings for spe_display */
static int clamp_point(int *x, int *y) {
  if (*x < (int)lcd_bg_area.X0) *x = lcd_bg_area.X0;
  if (*y < (int)lcd_bg_area.Y0) *y = lcd_bg_area.Y0;
  if (*x >= (int)(lcd_bg_area.X0 + lcd_bg_area.XSize)) *x = lcd_bg_area.X0 + lcd_bg_area.XSize - 1;
  if (*y >= (int)(lcd_bg_area.Y0 + lcd_bg_area.YSize)) *y = lcd_bg_area.Y0 + lcd_bg_area.YSize - 1;
  return 0;
}
static void convert_length(float32_t wi, float32_t hi, int *wo, int *ho) {
  *wo = lcd_bg_area.XSize * wi; *ho = lcd_bg_area.YSize * hi;
}
static void convert_point(float32_t xi, float32_t yi, int *xo, int *yo) {
  *xo = lcd_bg_area.XSize * xi + lcd_bg_area.X0; *yo = lcd_bg_area.YSize * yi + lcd_bg_area.Y0;
}
static void Display_binding_line(int x0, int y0, int x1, int y1, uint32_t color) {
  clamp_point(&x0, &y0); clamp_point(&x1, &y1);
  UTIL_LCD_DrawLine(x0, y0, x1, y1, color);
}
