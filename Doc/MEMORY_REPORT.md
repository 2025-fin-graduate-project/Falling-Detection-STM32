# STM32N6570-DK Fall Detection Memory Analysis

| Region | Component | Size (Bytes) | Size (KB) | % of Region |
| :--- | :--- | ---: | ---: | ---: |
| AXISRAM | ISR Vector | 844 | 0.82 | 0.1% |
| AXISRAM | Text (Code) | 169,276 | 165.31 | 29.3% |
| AXISRAM | RO Data | 353,088 | 344.81 | 61.1% |
| AXISRAM | Data (Init) | 12,796 | 12.50 | 2.2% |
| AXISRAM | BSS (Global) | 24,800 | 24.22 | 4.3% |
| AXISRAM | Heap/Stack | 16,896 | 16.50 | 2.9% |
| **Total** | **Internal SRAM** | **577,700** | **564.16** | **100%** |
| PSRAM | AI Buffers | 2,304,000 | 2250.00 | 100% |

### Memory Usage Visualization (Internal SRAM)

ISR Vector      [--------------------]    0.1%
Text (Code)     [#####---------------]    29.3%
RO Data         [############--------]    61.1%
Data (Init)     [--------------------]    2.2%
BSS (Global)    [--------------------]    4.3%
Heap/Stack      [--------------------]    2.9%

### 핵심 요약
- **AXISRAM 점유율**: 564.2 KB / 1,023 KB (55.1%)
- **Pipeline D (전처리)**: 14,252 Bytes (2.5% of Internal SRAM)
- **PSRAM (AI 모델 전용)**: 2,250 KB (Activations & Buffers)
