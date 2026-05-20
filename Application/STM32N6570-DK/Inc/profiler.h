#ifndef PROFILER_H
#define PROFILER_H

#include "stm32n6xx_hal.h"

typedef struct {
    uint32_t xspi_fix_us;
    uint32_t gru_run_us;
    uint32_t npu_run_us;
    uint32_t postproc_us;
} Profiler_t;

extern Profiler_t g_profiler;

void Profiler_Init(void);
uint32_t Profiler_GetCycles(void);
uint32_t Profiler_CyclesToUs(uint32_t cycles);
void Profiler_PrintReport(uint32_t interval_ms);

#endif /* PROFILER_H */
