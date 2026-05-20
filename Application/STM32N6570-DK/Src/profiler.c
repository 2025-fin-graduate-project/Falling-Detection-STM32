#include "profiler.h"
#include <stdio.h>

Profiler_t g_profiler = {0};

void Profiler_Init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

uint32_t Profiler_GetCycles(void)
{
    return DWT->CYCCNT;
}

uint32_t Profiler_CyclesToUs(uint32_t cycles)
{
    /* STM32N6 CPU runs at 800MHz (PLL1) */
    return cycles / 800U;
}

void Profiler_PrintReport(uint32_t interval_ms)
{
    static uint32_t last_tick = 0;
    uint32_t now = HAL_GetTick();
    if (now - last_tick >= interval_ms) {
        printf("[PROF] NPU=%luus Post=%luus xSPI=%luus GRU=%luus\r\n",
               g_profiler.npu_run_us, g_profiler.postproc_us,
               g_profiler.xspi_fix_us, g_profiler.gru_run_us);
        last_tick = now;
    }
}
