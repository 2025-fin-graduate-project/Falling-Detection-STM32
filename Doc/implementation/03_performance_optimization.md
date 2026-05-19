# Implementation 03: GRU 추론 지연 최적화 (95ms → 목표 ≤46ms)

## 1. 현황 및 문제

### 1.1 측정값
- MoveNet (NPU/ATON): **46ms**
- GRU 추론 (CPU): **95ms**
- 현재 총 프레임 주기: 141ms → **7.1fps** (목표 15fps 미달)

### 1.2 병목 원인

`fall_detection.c`의 `FallDetection_RunInference()`를 기준으로:

| 구간 | 원인 |
|------|------|
| xSPI2 memory-mapped 복구 (lines 62–75) | ll_aton이 OctoSPI2를 indirect 모드로 남김 |
| `stai_gru_network_run()` 본 계산 | GRU 가중치 604 KiB가 xSPI NOR Flash(XIP)에 상주 — D-cache(32 KiB) 대비 19× 초과로 캐시 스레싱 발생 |

```
g_gru_network_weights_array[75521]  /* 604 KiB — const in NOR Flash */
gru_activation_buf[33 KiB]         /* .bss → AXI SRAM (이미 빠름, 문제 없음) */
```

추론 1회당 가중치 전체를 플래시에서 읽어야 하며, D-cache 재사용률이 5% 미만.

---

## 2. Step 0: 구간별 프로파일링 추가 (필수 선행)

최적화 전 xSPI 복구와 GRU 계산의 실제 비중을 측정한다.

**`fall_detection.c` — `FallDetection_RunInference()` 수정:**

```c
void FallDetection_RunInference(void)
{
  /* DWT 사이클 카운터 활성화 (최초 1회) */
  static uint8_t dwt_init = 0;
  if (!dwt_init) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
    dwt_init = 1;
  }
  uint32_t cpu_hz = HAL_RCC_GetSysClockFreq();

  uint32_t c0 = DWT->CYCCNT;

  /* OctoSPI2 Fix */
  {
    extern XSPI_HandleTypeDef hxspi_nor[];
    CLEAR_BIT(hxspi_nor[0].Instance->CR, XSPI_CR_DMAEN);
    if (HAL_XSPI_GET_FLAG(&hxspi_nor[0], HAL_XSPI_FLAG_BUSY)) {
      SET_BIT(hxspi_nor[0].Instance->CR, XSPI_CR_ABORT);
      uint32_t t0 = HAL_GetTick();
      while (!HAL_XSPI_GET_FLAG(&hxspi_nor[0], HAL_XSPI_FLAG_TC) && (HAL_GetTick()-t0) < 100U);
      HAL_XSPI_CLEAR_FLAG(&hxspi_nor[0], HAL_XSPI_FLAG_TC);
    }
    hxspi_nor[0].State = HAL_XSPI_STATE_READY;
  }
  BSP_XSPI_NOR_EnableMemoryMappedMode(0);

  uint32_t c1 = DWT->CYCCNT;
  stai_gru_network_run(gru_network_context, STAI_MODE_SYNC);
  uint32_t c2 = DWT->CYCCNT;

  static uint32_t prof_last = 0;
  if (HAL_GetTick() - prof_last >= 3000u) {
    printf("[PROF] xSPI_fix=%luus  GRU_run=%luus  total=%luus\r\n",
           (c1-c0)/(cpu_hz/1000000U),
           (c2-c1)/(cpu_hz/1000000U),
           (c2-c0)/(cpu_hz/1000000U));
    prof_last = HAL_GetTick();
  }
}
```

- xSPI 복구 비중이 크면 → **Path A** 우선
- GRU 계산 자체가 크면 → **Path B** 우선

---

## 3. Path A: NPU+CPU 파이프라이닝

### 개요

MoveNet(NPU)과 GRU(CPU)를 비동기로 겹쳐 실행한다.

```
현재: [cam wait] ─ [NPU 46ms] ─ [GRU 95ms] = 141ms
변경: [cam wait] ─ [NPU start async] ─ [GRU(prev) 95ms] ─ [NPU result read] = ~95ms
```

**기대 효과**: 141ms → **95ms(10.5fps)**, 모델·가중치 변경 없음.

**전제**: ATON NPU와 Cortex-M55는 독립 하드웨어로 동시 실행 가능.

### main.c 변경 구조

```c
/* 전역 추가 */
static spe_pp_out_t pp_output_prev   = {0};
static uint8_t      pp_prev_valid    = 0;
static uint32_t     pose_prev_tick   = 0;

/* 메인 루프 재구성 */
while (1) {
    CameraPipeline_IspUpdate();

    /* 1. NPU 비동기 시작 */
    CameraPipeline_NNPipe_Start(nn_in, CMW_MODE_SNAPSHOT);

    /* 2. 이전 프레임 GRU 실행 (NPU와 병렬) */
    if (pp_prev_valid) {
        float32_t dt_s = (float32_t)(HAL_GetTick() - pose_prev_tick) * 1e-3f;
        if (dt_s <= 0.0f || dt_s > 1.0f) dt_s = 1.0f / 15.0f;
        pose_prev_tick = HAL_GetTick();

        PosePipeline_Process(&pose_pipeline, pp_output_prev.pOutBuff, dt_s, &feat_vec);
        FallDetection_Update(&pose_pipeline);  /* GRU 95ms — NPU와 겹침 */
    }

    /* 3. NPU 완료 대기 및 postprocess */
    while (cameraFrameReceived == 0) {}
    cameraFrameReceived = 0;
    /* ... cache invalidate, postprocess, presence state machine ... */

    /* 4. 결과를 다음 프레임용으로 저장 */
    pp_output_prev = pp_output;
    pp_prev_valid  = (postprocess_ok && pp_output.pOutBuff != NULL) ? 1 : 0;

    Display_NetworkOutput(&pp_output, movenet_ms);
    Alarm_Update();
}
```

> **주의**: STedgeAI `STAI_MODE_ASYNC` 지원 여부를 먼저 확인할 것. 미지원 시 FreeRTOS task 또는 DMA 완료 인터럽트로 NPU 비동기화를 구현해야 한다.

---

## 4. Path B: GRU 가중치 AXI SRAM 복사

### 개요

부팅 시 NOR Flash의 가중치(604 KiB)를 AXI SRAM으로 복사해 이후 접근 속도를 높인다.

| 메모리 | 접근 시간 | 가중치 접근 후 GRU 예상 |
|--------|----------|----------------------|
| xSPI NOR Flash (현재) | ~80ns | 95ms |
| AXI SRAM | ~2ns | ~20ms (4–5× 단축 예상) |

Path A + B 조합 시: max(46, 20) = **46ms → 21.7fps**

### 구현 방법

**1. 링커 스크립트에 SRAM 섹션 추가** (NPU 미사용 AXI SRAM5 권장):

```ld
/* STM32N6570_DK.ld */
.axisram5_data (NOLOAD) :
{
    *(.axisram5_bss)
} >AXISRAM5
```

`NPURam_enable()`에서 활성화되는 SRAM 범위와 충돌하지 않는 영역을 확인한다.

**2. `fall_detection.c` — 복사 버퍼 및 초기화:**

```c
/* 604 KiB SRAM 버퍼 — AXI SRAM5 배치 */
#define GRU_WEIGHTS_SIZE  (75521U * sizeof(uint64_t))   /* 604,168 bytes */

__attribute__((section(".axisram5_bss"), aligned(32)))
static uint8_t gru_weights_sram[GRU_WEIGHTS_SIZE];

void FallDetection_Init(void)
{
  /* 부팅 시 1회: Flash → SRAM 복사 */
  extern const uint64_t g_gru_network_weights_array[75521];
  memcpy(gru_weights_sram, g_gru_network_weights_array, GRU_WEIGHTS_SIZE);
  SCB_CleanDCache_by_Addr(gru_weights_sram, GRU_WEIGHTS_SIZE);
  printf("[GRU] Weights copied to SRAM (%u bytes)\r\n", GRU_WEIGHTS_SIZE);

  /* 이후 기존 초기화 진행 */
  stai_gru_network_init(gru_network_context);
  /* STedgeAI가 stai_network_set_weights() API를 제공하는 경우:
   * stai_gru_network_set_weights(gru_network_context,
   *                              gru_weights_sram, GRU_WEIGHTS_SIZE); */
  ...
}
```

> **확인 필요**: STedgeAI 4.0이 `stai_network_set_weights()` API를 제공하지 않는 경우, `gru_network_data.c`의 `g_gru_network_weights_array` 선언을 `const` → 일반 배열로 변경하고 `.axisram5_bss` 섹션으로 직접 배치해야 한다. STedgeAI `generate` 재실행 시 해당 파일이 덮어써지므로, 패치 스크립트를 통해 자동화할 것.

---

## 5. Path C: 경량 모델 교체 (ev 손실 허용 시)

| 모델 | Flash | GRU 추정 시간 | ev_vote MinPR | 비고 |
|------|-------|-------------|--------------|------|
| P38-nv-a65 GRU(128,64) | 590 KiB | 95ms | 0.9079 | 현재 |
| P39-nv-a65-h256 GRU(256,128) | ~1.1 MB | ~180ms | **0.9123** | 최고 정확도, 더 느림 |
| P37-pure-h64 GRU(64,32) | 275 KiB | **~24ms** | 0.8947 | Path A만으로 15fps 가능 |

GRU(64,32) 교체 시 `generate-gru-p38nv-model_STM32N6570-DK.sh`의 모델 경로를 교체하고 STedgeAI 재생성 필요. ev MinPR 0.013 하락 허용 여부를 사전에 확인한다.

---

## 6. 권장 적용 순서

```
1단계 (즉시, 30분)
  └─ Step 0 프로파일링 추가 → [PROF] 로그로 구간 비중 확인

2단계 (~2일, 모델 변경 없음)
  └─ Path A: NPU 비동기 + CPU 파이프라이닝 → 10.5fps

3단계 (~3일, 링커 작업)
  └─ Path B: 가중치 SRAM 복사 → GRU ≤20ms
  └─ (A+B 조합) → ~21fps

대안 (즉시, 정확도 허용 시)
  └─ Path C: GRU(64,32) 교체 → Path A만으로 15fps
```
