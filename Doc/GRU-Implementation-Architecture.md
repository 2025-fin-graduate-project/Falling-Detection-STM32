# STM32N6570-DK 기반 실시간 낙상 감지 시스템  
## MoveNet–GRU 이중 모델 아키텍처 구현 상세 보고서

---

> **플랫폼**: STM32N6570-DK (ARM Cortex-M55 @ 800 MHz + NPU 1 GHz)  
> **모델**: MoveNet Lightning 256×256 (NPU) + GRU v26 Stateful FP32 (CPU)  
> **빌드 환경**: arm-none-eabi-gcc, `-Os -g3`, STEdgeAI 런타임

---

## 목차

1. [시스템 개요](#1-시스템-개요)  
2. [하드웨어 플랫폼](#2-하드웨어-플랫폼)  
3. [MoveNet Lightning 포즈 추정 모델](#3-movenet-lightning-포즈-추정-모델)  
4. [포즈 후처리 및 특징 추출 파이프라인](#4-포즈-후처리-및-특징-추출-파이프라인)  
5. [GRU 낙상 감지 모델](#5-gru-낙상-감지-모델)  
6. [Stateless → Stateful 변환 기법](#6-stateless--stateful-변환-기법)  
7. [메모리 아키텍처](#7-메모리-아키텍처)  
8. [End-to-End 추론 파이프라인](#8-end-to-end-추론-파이프라인)  
9. [낙상 판정 알고리즘](#9-낙상-판정-알고리즘)  
10. [성능 분석](#10-성능-분석)

---

## 1. 시스템 개요

본 시스템은 STM32N6570-DK 임베디드 보드에서 카메라 입력으로부터 실시간 포즈를 추정하고, 그 결과를 시계열 특징 벡터로 변환하여 GRU 신경망으로 낙상을 감지한다. 두 모델의 역할은 명확하게 분리된다.

| 역할 | 모델 | 실행 장치 | 입력 | 출력 |
|---|---|---|---|---|
| 포즈 추정 | MoveNet Lightning 256 | NPU (ATON) | 256×256 RGB 영상 | 17개 키포인트 (x, y, confidence) |
| 낙상 감지 | GRU v26 Stateful FP32 | CPU (Cortex-M55) | 55-dim 특징 벡터 (1 frame) | 낙상/정상 로짓 2값 |

MoveNet은 매 프레임(30 fps) 추론하고, GRU는 2프레임마다 1회(15 fps)로 구동한다.

---

## 2. 하드웨어 플랫폼

### 2.1 프로세서 사양

| 항목 | 값 |
|---|---|
| 코어 | ARM Cortex-M55 (ARMv8.1-M + Helium MVE) |
| CPU 클록 | 800 MHz (PLL1) |
| NPU 클록 | 1,000 MHz (PLL2) |
| AXI 버스 클록 | 400 MHz (IC2) |
| AXISRAM3–6 클록 | 900 MHz (PLL3) |

### 2.2 메모리 구성

| 메모리 | 베이스 주소 | 크기 | 용도 |
|---|---|---|---|
| AXISRAM1_S | `0x34000400` | 1,023 KB | 코드, 데이터, BSS, 스택 |
| External PSRAM | `0x91000000` | 16 MB | LCD 프레임버퍼 |
| External NOR Flash | `0x70000000` | (다단계) | FSBL + App + Model 가중치 |

### 2.3 플래시 주소 공간 레이아웃

```
0x70000000  ┌─────────────────────────┐
            │  FSBL (ai_fsbl.hex)     │  1 MB
0x70100000  ├─────────────────────────┤
            │  Application (App.bin)  │  ~2.8 MB
0x70380000  ├─────────────────────────┤
            │  Model Weights          │
            │  - MoveNet ~2.7 MB      │
            │  - GRU     ~1.2 MB      │
            └─────────────────────────┘
```

---

## 3. MoveNet Lightning 포즈 추정 모델

### 3.1 모델 사양

| 항목 | 값 |
|---|---|
| 원본 모델명 | `st_movenet_lightning_a100_heatmaps_256_int8` |
| 입력 해상도 | 256 × 256 × 3 (RGB) |
| 입력 포맷 | STAI_FORMAT_U8 (uint8, 정규화 scale=0.00784, offset=127) |
| 출력 해상도 | 64 × 64 × 17 (히트맵) |
| 출력 포맷 | STAI_FORMAT_S8 (int8, scale=0.003599, offset=-128) |
| 키포인트 수 | 17개 (COCO keypoint 규격) |
| 가중치 수 | 263개 텐서 |
| 실행 장치 | STEdgeAI ATON NPU |

### 3.2 입출력 버퍼 규격

```
입력 버퍼
  형상 : {1, 256, 256, 3}  (NHWC)
  크기 : 256 × 256 × 3 = 196,608 bytes  (192 KB)
  정렬 : 32-byte aligned
  배치 : 1

출력 버퍼
  형상 : {1, 64, 64, 17}  (NHWC)
  크기 : 64 × 64 × 17 = 69,632 bytes  (68 KB)
  정렬 : 32-byte aligned
  역양자화 : val_float = scale × (raw_int8 − offset)
            = 0.003599 × (raw + 128)
```

### 3.3 COCO 17 키포인트 인덱스

| 인덱스 | 부위 | 인덱스 | 부위 |
|---|---|---|---|
| 0 | nose | 9 | left_wrist |
| 1 | left_eye | 10 | right_wrist |
| 2 | right_eye | 11 | left_hip |
| 3 | left_ear | 12 | right_hip |
| 4 | right_ear | 13 | left_knee |
| 5 | left_shoulder | 14 | right_knee |
| 6 | right_shoulder | 15 | left_ankle |
| 7 | left_elbow | 16 | right_ankle |
| 8 | right_elbow | | |

### 3.4 NPU 추론 흐름

MoveNet은 STEdgeAI ATON NPU 위에서 비동기(`STAI_MODE_ASYNC`) + WFE(Wait For Event) 방식으로 실행된다.

```c
// Run_Inference() — main.c
do {
    ret = stai_network_run(network_context, STAI_MODE_ASYNC);
    if (ret == STAI_RUNNING_WFE)
        LL_ATON_OSAL_WFE();      // CPU를 슬립, NPU 완료 인터럽트로 깨어남
} while (ret == STAI_RUNNING_WFE || ret == STAI_RUNNING_NO_WFE);
ret = stai_ext_network_new_inference(network_context);
```

NPU 추론 중 CPU는 WFE 슬립 상태로 전력 효율을 최대화한다.

---

## 4. 포즈 후처리 및 특징 추출 파이프라인

### 4.1 히트맵 → 키포인트 변환

MoveNet 출력 히트맵(64×64×17)은 `app_postprocess_spe_movenet_ui` 모듈이 처리한다.  
각 채널의 최대값 위치를 취하고 역양자화하여 정규화된 좌표와 신뢰도를 산출한다.

```
출력: spe_pp_outBuffer_t[17]
  .x_center  : float32, 0.0–1.0 (정규화된 수평 위치)
  .y_center  : float32, 0.0–1.0 (정규화된 수직 위치)
  .proba     : float32, 0.0–1.0 (키포인트 신뢰도)
```

신뢰도 임계값 `AI_POSE_PP_CONF_THRESHOLD = 0.10`을 넘는 키포인트만 가시 키포인트로 집계된다.

### 4.2 인물 존재 판정 (Robust Presence)

단순 임계값 대신 **상위 5개 키포인트 평균 신뢰도**로 인물 존재를 판정한다.

```
top5_avg = mean(상위 5개 키포인트 신뢰도)
person_present = (top5_avg ≥ 0.2)
```

연속적 상태 전환을 방지하기 위해 히스테리시스를 적용한다.

| 이벤트 | 임계 프레임 | 매크로 |
|---|---|---|
| 인물 진입 확정 | 연속 3프레임 | `FALL_PERSON_ENTRY_CONF_COUNT = 3` |
| 인물 소실 확정 | 연속 45프레임 | `FALL_PERSON_MISSING_RESET_COUNT = 45` |

### 4.3 키포인트 필터링 (One-Euro Filter)

Raw 키포인트 좌표는 One-Euro Filter를 통해 고주파 잡음을 제거하면서도 빠른 동작에 반응한다.

**One-Euro Filter 수식**:

```
α_d  = 2π f_c_d / (2π f_c_d + f_s)       # 미분 저역통과
x̂_d  = α_d (x − x̂_prev) / Δt + (1−α_d) x̂_d_prev
f_c  = f_c_min + β |x̂_d|                  # 컷오프 주파수 적응
α    = 2π f_c / (2π f_c + f_s)            # 위치 저역통과
x̂    = α x + (1−α) x̂_prev
```

**필터 파라미터**:

| 파라미터 | 값 | 역할 |
|---|---|---|
| `POSE_EURO_MIN_CUTOFF` | 1.0 Hz | 정적 상태 스무딩 강도 |
| `POSE_EURO_BETA` | 0.5 | 속도 민감도 (빠른 동작 지연 감소) |
| `POSE_EURO_D_CUTOFF` | 1.0 Hz | 미분 필터 컷오프 |

신뢰도는 EMA(지수 이동 평균)로 별도 스무딩한다:

```
conf_smooth = α × conf_raw + (1−α) × conf_prev
POSE_CONF_EMA_ALPHA = 0.5
```

**`PoseEuroFilter_t` 구조체** (12 bytes):
```c
typedef struct {
    float32_t x_prev;      // 이전 필터 출력 (4 bytes)
    float32_t dx_prev;     // 이전 미분 추정값 (4 bytes)
    uint8_t   initialized; // 초기화 플래그 (1 byte + 3 padding)
} PoseEuroFilter_t;        // 총 12 bytes
```

**`PoseEmaFilter_t` 구조체** (8 bytes):
```c
typedef struct {
    float32_t x_prev;      // 이전 EMA 출력 (4 bytes)
    uint8_t   initialized; // 초기화 플래그 (1 byte + 3 padding)
} PoseEmaFilter_t;         // 총 8 bytes
```

### 4.4 특징 벡터 구성 (55-dim)

`PosePipeline_Process()`는 매 유효 프레임마다 55차원 특징 벡터를 생성한다.

```
특징 벡터 구성 (POSE_FEATURE_COUNT = 55)

  [0..50]  : 17개 키포인트 × 3 = 51개
              각 키포인트: (y_filtered, x_filtered, conf_filtered)

  [51]     : HSSC_Y  — 상체 중심의 Y 속도 추정값
  [52]     : HSSC_X  — 상체 중심의 X 속도 추정값
  [53]     : RWHC    — 신체 종횡비 변화율
  [54]     : VHSSC   — 수직 상체 속도 (1차 미분)
```

**엔지니어링 특징 (Engineering Features)**:

HSSC(Head-Shoulder-Center)는 코(0), 눈(1,2), 귀(3,4), 어깨(5,6) 7개 키포인트의 가중 평균으로 상체 중심을 계산한다.

```
HSSC_Y(t) = mean(y_filtered[0..6])
HSSC_X(t) = mean(x_filtered[0..6])
VHSSC(t)  = (HSSC_Y(t) − HSSC_Y(t−1)) / Δt   [dt 정규화 포함]
```

### 4.5 시계열 원형 버퍼

```
win_buf[60][55]  — 원형 버퍼 (60 timestep × 55 features)
win_head         — 다음 쓰기 인덱스
win_count        — 저장된 유효 프레임 수
```

15 fps 기준으로 60프레임 = **4초**의 시간 맥락을 유지한다.  
GRU가 Stateful로 동작하므로 실제로는 버퍼 전체를 한 번에 GRU에 주입하지 않고,  
매 프레임 최신 특징 벡터 1개씩 GRU에 입력한다.

---

## 5. GRU 낙상 감지 모델

### 5.1 모델 사양

| 항목 | 값 |
|---|---|
| 원본 모델명 | `gru_v26_stateful_fp32` |
| 아키텍처 | 2-layer Stacked GRU + Dense |
| 제1 GRU 은닉 유닛 | 64 |
| 제2 GRU 은닉 유닛 | 32 |
| 출력 클래스 수 | 2 (정상, 낙상) |
| 데이터 타입 | FP32 (비양자화) |
| 총 노드 수 | 40 |
| MACC 수 | 37,378 |
| 가중치 크기 | 135,060 bytes (131.9 KB) |
| 실행 장치 | CPU (Cortex-M55), 동기 모드 |

### 5.2 GRU 레이어 구조

```
입력: pose_feature [1, 1, 55]  ← 현재 프레임 특징 벡터
      h1 [1, 64]               ← 1층 GRU 이전 은닉 상태
      h2 [1, 32]               ← 2층 GRU 이전 은닉 상태

┌──────────────────────────────────────────────────────┐
│  GRU Layer 1 (64 units)                              │
│   z₁ = σ(W_z·[h1, x] + b_z)   # Update Gate        │
│   r₁ = σ(W_r·[h1, x] + b_r)   # Reset Gate         │
│   ñ₁ = tanh(W_n·[r₁⊙h1, x] + b_n)  # New Gate      │
│   new_h1 = (1−z₁)⊙h1 + z₁⊙ñ₁                       │
└──────────────────┬───────────────────────────────────┘
                   │ new_h1 [1, 64]
┌──────────────────▼───────────────────────────────────┐
│  GRU Layer 2 (32 units)                              │
│   z₂ = σ(W_z·[h2, new_h1] + b_z)                    │
│   r₂ = σ(W_r·[h2, new_h1] + b_r)                    │
│   ñ₂ = tanh(W_n·[r₂⊙h2, new_h1] + b_n)             │
│   new_h2 = (1−z₂)⊙h2 + z₂⊙ñ₂                       │
└──────────────────┬───────────────────────────────────┘
                   │ new_h2 [1, 32]
┌──────────────────▼───────────────────────────────────┐
│  Dense + Softmax                                     │
│   logits = W_d · new_h2 + b_d   [1, 2]              │
│   p = softmax(logits)           [p_normal, p_fall]   │
└──────────────────────────────────────────────────────┘

출력: new_h1 [1, 64]  ← 다음 프레임으로 전달
      logits [1, 2]   ← 낙상 판정에 사용
      new_h2 [1, 32]  ← 다음 프레임으로 전달
```

### 5.3 노드 유형 분포 (총 40개)

| 노드 유형 | 수 | 역할 |
|---|---|---|
| UNPACK | 1 | 입력 텐서 언패킹 |
| DENSE | 4 | 선형 변환 (GRU 게이트, Dense 출력) |
| SPLIT | 2 | 게이트별 텐서 분리 |
| SLICE | 7 | 은닉 상태 추출 |
| ELTWISE | 18 | 요소별 연산 (게이트 연산, 상태 갱신) |
| NL | 8 | 비선형 활성화 (σ, tanh) |
| SM | 1 | Softmax (최종 확률 정규화) |

---

## 6. Stateless → Stateful 변환 기법

### 6.1 변환 배경

원본 GRU 모델은 stateless로 학습되었다 — 추론 시 T=60 타임스텝 전체를 한 번에 입력한다.  
임베디드 환경에서는 새로운 프레임이 생길 때마다 실시간으로 추론해야 하고,  
메모리에 60×55=3,300개 float를 누적 후 재입력하는 방식은 레이턴시가 크다.

### 6.2 Stateful 변환 원리

GRU 특성상 `h(t) = GRU(h(t−1), x(t))`가 성립한다. 이를 이용해:

```
기존 (Stateless):  전체 시퀀스 [x₁, x₂, ..., x₆₀] → 한 번에 추론

변환 후 (Stateful): 
  h₀ = 0 (초기화)
  h₁ = GRU(h₀, x₁)
  h₂ = GRU(h₁, x₂)
  ...
  h_t = GRU(h_{t-1}, x_t)  ← 매 프레임 1회 호출
```

이를 구현하기 위해 STEdgeAI의 stateful 변환 도구를 사용하여  
은닉 상태(h1, h2)를 **외부 입출력 텐서**로 노출시킨 모델을 생성했다.

### 6.3 입출력 인터페이스 상세

| 포트 | 방향 | 형상 | 크기 | 내용 |
|---|---|---|---|---|
| IN[0] | 입력 | {1, 64} | 256 bytes | 이전 h1 (GRU Layer 1 은닉 상태) |
| IN[1] | 입력 | {1, 32} | 128 bytes | 이전 h2 (GRU Layer 2 은닉 상태) |
| IN[2] | 입력 | {1, 1, 55} | 220 bytes | 현재 프레임 특징 벡터 |
| OUT[0] | 출력 | {1, 64} | 256 bytes | 갱신된 h1 |
| OUT[1] | 출력 | {1, 2} | 8 bytes | 낙상/정상 로짓 |
| OUT[2] | 출력 | {1, 32} | 128 bytes | 갱신된 h2 |

### 6.4 은닉 상태 관리 코드

```c
// FallDetection_Update() — main.c

/* 입력 준비: 이전 은닉 상태 + 현재 특징 벡터 */
memcpy(gru_in[0], gru_h1, 256);                           // h1 → IN[0]
memcpy(gru_in[1], gru_h2, 128);                           // h2 → IN[1]
PosePipeline_GetLatestFeature(pipeline, gru_in[2]);        // feat → IN[2]

/* CPU 동기 추론 (37,378 MACC, 비NPU) */
ret = stai_gru_network_run(gru_network_context, STAI_MODE_SYNC);

/* 은닉 상태 갱신: 다음 프레임을 위해 보존 */
memcpy(gru_h1, gru_out[0], 256);                          // OUT[0] → h1
memcpy(gru_h2, gru_out[2], 128);                          // OUT[2] → h2

/* 로짓 읽기 */
float32_t *logits = (float32_t *)gru_out[1];              // OUT[1]
```

### 6.5 은닉 상태 초기화 조건

| 조건 | 초기화 동작 |
|---|---|
| 낙상 확정 (5 cumulative frames) | `memset(gru_h1, 0)` + `memset(gru_h2, 0)` + PosePipeline 초기화 |
| 인물 소실 (`pose_valid == 0`) | `memset(gru_h1, 0)` + `memset(gru_h2, 0)` |
| 시스템 시작 | `FallModel_init()` 내 `memset` 초기화 |

### 6.6 Warmup 메커니즘

초기화 직후 은닉 상태가 모두 0이므로 출력이 신뢰하기 어렵다.  
`GRU_WARMUP_FRAMES = 15` (약 1초) 프레임 동안은 추론 결과를 무시한다.

```c
fall_state.window_ready = (pipeline->win_count >= GRU_WARMUP_FRAMES) ? 1u : 0u;
```

---

## 7. 메모리 아키텍처

### 7.1 전체 메모리 지도 (측정값 기준)

```
AXISRAM1_S (1,023 KB 총용량, 577 KB 사용 — 56.4%)
┌────────────────────────────────────────┐ 0x34000400
│  .isr_vector           844 B           │
│  .text              169,156 B (165 KB) │  ← 코드 + STEdgeAI 런타임
│  .gnu.sgstubs            32 B           │
│  .rodata            353,088 B (345 KB) │  ← GRU 가중치(rodata에 포함)
│  .init/.fini_array        8 B           │
│  .data               12,796 B  (12 KB) │  ← 초기화된 전역 변수
│  .bss                24,312 B  (24 KB) │  ← 비초기화 전역 변수 (표 참조)
│  ._user_heap_stack   16,896 B  (17 KB) │  ← 힙 512 B + 스택 16 KB
└────────────────────────────────────────┘ 0x340FFFFF

PSRAM (16 MB 총용량, 2,250 KB 사용 — 13.7%)
┌────────────────────────────────────────┐ 0x91000000
│  lcd_bg_buffer    768,000 B (750 KB)   │  ← 카메라 디스플레이 스트림
│  lcd_fg_buffer×2 1,536,000 B (1,500 KB)│  ← 더블버퍼 오버레이 레이어
└────────────────────────────────────────┘ 0x91232800
```

### 7.2 BSS 섹션 주요 심볼 상세 (측정값)

| 심볼 | 주소 | 크기 | 설명 |
|---|---|---|---|
| `gru_h2` | `0x34083CE0` | 128 B (0x80) | GRU Layer 2 은닉 상태 |
| `gru_h1` | `0x34083D60` | 256 B (0x100) | GRU Layer 1 은닉 상태 |
| `gru_out[3]` | `0x34083E60` | 12 B (0x0C) | 출력 포인터 배열 |
| `gru_in[3]` | `0x34083E6C` | 12 B (0x0C) | 입력 포인터 배열 |
| `gru_activation_buf` | (BSS) | 2,816 B | GRU 추론 스크래치 메모리 |
| `pose_pipeline` | (BSS) | 13,764 B | 키포인트 필터 + 원형 버퍼 |
| `fall_state` | (BSS) | ~44 B | 낙상 감지 상태 구조체 |
| `pose_debug_metrics` | (BSS) | ~52 B | 디버그 메트릭 구조체 |
| `network_context` | (BSS) | sizeof(ATON ctx) | MoveNet NPU 컨텍스트 |
| `gru_network_context` | (BSS) | sizeof(GRU ctx) | GRU CPU 컨텍스트 |

### 7.3 컴포넌트별 메모리 예산 (런타임)

#### MoveNet 관련

| 항목 | 위치 | 크기 |
|---|---|---|
| 입력 버퍼 (256×256×3) | AXISRAM (DMA target) | 196,608 B (192 KB) |
| 출력 버퍼 (64×64×17) | AXISRAM | 69,632 B (68 KB) |
| 가중치 (263 텐서) | External NOR Flash (xSPI) | ~2,715,788 B (2.6 MB) |
| NPU 활성화 스크래치 | AXISRAM3–6 (NPU SRAM) | 모델 내 자동 할당 |
| ATON 컨텍스트 | AXISRAM1_S .bss | sizeof(_stai_aton_context) |

#### GRU 관련

| 항목 | 위치 | 크기 |
|---|---|---|
| 가중치 | External NOR Flash (xSPI) | 135,060 B (131.9 KB) |
| 활성화 스크래치 | AXISRAM1_S .bss | 2,816 B |
| 은닉 상태 h1 (persistent) | AXISRAM1_S .bss | 256 B |
| 은닉 상태 h2 (persistent) | AXISRAM1_S .bss | 128 B |
| 입력 버퍼 (h1+h2+pose) | AXISRAM1_S (포인터 매핑) | 604 B |
| 출력 버퍼 (h1+logits+h2) | AXISRAM1_S (포인터 매핑) | 392 B |
| **GRU 런타임 합계** (가중치 제외) | AXISRAM1_S | **4,196 B (4.1 KB)** |

#### Pose Pipeline 구조체 상세

```
PosePipeline_t 총 크기: 13,764 bytes (13.4 KB)

  PoseEuroFilter_t y_filter[17]:   17 × 12 =   204 B  (y좌표 필터)
  PoseEuroFilter_t x_filter[17]:   17 × 12 =   204 B  (x좌표 필터)
  PoseEmaFilter_t  conf_filter[17]: 17 × 8  =   136 B  (신뢰도 EMA 필터)
  float32_t hssc_y_prev:                          4 B
  float32_t vhssc_prev:                           4 B
  uint8_t   deriv_initialized (+3 pad):           4 B
  float32_t win_buf[60][55]:     60×55×4 = 13,200 B  (시계열 원형 버퍼)
  uint32_t  win_head:                             4 B
  uint32_t  win_count:                            4 B
```

#### LCD 프레임버퍼 (PSRAM)

```
lcd_bg_buffer[800×480×2]:     768,000 B (750 KB)  — 카메라 RGB565
lcd_fg_buffer[2][800×480×2]: 1,536,000 B (1,500 KB)  — 더블버퍼 ARGB4444
                                                        (2 × 750 KB)
합계:                          2,304,000 B (2,250 KB)
```

---

## 8. End-to-End 추론 파이프라인

### 8.1 카메라 이중 파이프라인 (DCMIPP)

```
카메라 센서 (CSI)
      │
      ▼
DCMIPP ISP
      │
      ├─── Pipe 1 (연속 스트림, CMW_MODE_CONTINUOUS)
      │         │
      │         ▼
      │    lcd_bg_buffer  [800×480 RGB565, 750 KB]
      │         │
      │         ▼
      │    LTDC Layer 1 → 디스플레이 출력
      │
      └─── Pipe 2 (단일 스냅샷, CMW_MODE_SNAPSHOT)
                │
                ▼
           dcmipp_out_nn  [256×256+α RGB888, 캐시 정렬]
                │
                ▼ (필요시 img_crop: stride 패딩 제거)
           nn_in (MoveNet 입력 DMA 버퍼)
```

### 8.2 프레임 처리 메인 루프 상세 흐름

```
[메인 루프, 매 프레임]
│
├─ 1. CameraPipeline_IspUpdate()
│       ISP 파라미터 갱신 (자동 노출, 화이트밸런스)
│
├─ 2. CameraPipeline_NNPipe_Start(nn_in, SNAPSHOT)
│       DCMIPP Pipe2: 256×256 단일 프레임 캡처
│
├─ 3. while (cameraFrameReceived == 0) {}
│       DMA 완료 대기
│
├─ 4. [옵션] img_crop(dcmipp_out_nn, nn_in, ...)
│       stride 패딩 제거 (256×256 비배수 모델 사용 시)
│
├─ 5. Run_Inference(network_context)   [ts0 ~ ts1]
│       NPU ASYNC + WFE: MoveNet 추론
│       입력: nn_in [196,608 B], 출력: nn_out [69,632 B]
│
├─ 6. SCB_InvalidateDCache_by_Addr(nn_out)
│       출력 캐시 무효화 (NPU DMA → CPU 가시성 확보)
│
├─ 7. app_postprocess_run()
│       히트맵 → spe_pp_out_t[17]
│       (x_center, y_center, proba) × 17 키포인트
│
├─ 8. Update_PoseDebugMetrics()
│       ├─ 가시 키포인트 집계
│       ├─ Top-5 신뢰도 → 인물 존재 판정
│       └─ Keypoint bounding box → 분산 판정
│
├─ 9. 인물 존재 상태 기계
│       ├─ person_present → person_entry_count++
│       ├─ entry_count ≥ 3 → person_present_confirmed = 1
│       ├─ missing_count ≥ 45 → person_present_confirmed = 0
│       └─ draw_keypoints 갱신
│
├─ 10. [짝수 프레임 && pose_valid] PosePipeline_Process()
│       ├─ One-Euro 필터 (17 KP × x, y)
│       ├─ EMA 필터 (17 KP confidence)
│       ├─ 엔지니어링 특징 계산 (HSSC_Y, HSSC_X, RWHC, VHSSC)
│       ├─ 55-dim 특징 벡터 생성
│       └─ win_buf[win_head] = feat; win_head = (win_head+1) % 60
│
├─ 11. [짝수 프레임 && pose_valid] FallDetection_Update()
│       GRU 추론 (매 유효 짝수 프레임)
│
└─ 12. Display_NetworkOutput() + Alarm_Update()
        LCD 렌더링 + LED 제어
```

### 8.3 GRU 추론 상세 흐름 (FallDetection_Update 내부)

```
FallDetection_Update()
│
├─ 워밍업 체크: win_count < 15 → 조기 종료 (결과 신뢰 불가)
│
├─ 입력 준비
│   memcpy(gru_in[0], gru_h1, 256)          h1 [64 floats] → IN[0]
│   memcpy(gru_in[1], gru_h2, 128)          h2 [32 floats] → IN[1]
│   PosePipeline_GetLatestFeature(→gru_in[2])  feat [55 floats] → IN[2]
│
├─ stai_gru_network_run(STAI_MODE_SYNC)
│   CPU 동기 추론 (37,378 MACC)
│   활성화 버퍼: gru_activation_buf [2,816 B]
│
├─ 은닉 상태 갱신
│   memcpy(gru_h1, gru_out[0], 256)         OUT[0] → h1 보존
│   memcpy(gru_h2, gru_out[2], 128)         OUT[2] → h2 보존
│
├─ Softmax 확률 변환
│   max_logit = max(logits[0], logits[1])
│   p_normal  = exp(logits[0] − max_logit) / denom
│   p_fall    = exp(logits[1] − max_logit) / denom
│   (numerically stable log-sum-exp trick 적용)
│
└─ 낙상 판정
    fall_detected = (p_fall ≥ GRU_FALL_SCORE_THRESHOLD=0.65)
```

---

## 9. 낙상 판정 알고리즘

### 9.1 Cumulative 확정 카운터

단일 프레임의 낙상 판정은 오탐 가능성이 높으므로, 연속 감지 횟수로 확정한다.

```c
if (fall_detected)
    fall_consec_count++;     // 연속 낙상 카운터 증가
else
    fall_consec_count = 0;   // 정상 판정 시 리셋

if (fall_consec_count >= GRU_FALL_RESET_COUNT)  // 5회 연속
{
    // 낙상 확정 → 알람 트리거 + 상태 초기화
    fall_latch_tick = HAL_GetTick();
    memset(gru_h1, 0, 256);
    memset(gru_h2, 0, 128);
    PosePipeline_Init(pipeline);
    fall_consec_count = 0;
}
```

### 9.2 알람 판정 파라미터 요약

| 파라미터 | 값 | 의미 |
|---|---|---|
| `GRU_FALL_SCORE_THRESHOLD` | 0.65 | 낙상 확률 임계값 (Softmax 출력) |
| `GRU_FALL_RESET_COUNT` | 5 | 알람 확정에 필요한 연속 프레임 수 |
| `GRU_WARMUP_FRAMES` | 15 | 은닉 상태 초기화 후 무시 구간 |
| `GRU_FALL_LATCH_MS` | 5,000 ms | 알람 유지 시간 |
| `GRU_ALARM_BLINK_PERIOD_MS` | 250 ms | LED_RED 블링크 반주기 |

### 9.3 알람 출력 채널

낙상 확정 시 다음 3개 채널을 동시에 활성화한다.

```
1. 직렬 콘솔 (UART1 @115200)
   "[ALARM] FALL CONFIRMED (5 cumulative frames) - resetting GRU state\r\n"

2. LCD 오버레이 (5초 Latch)
   ┌──────────────────────────────────┐  y=185
   │   !! FALL DETECTED !!  (Font24)  │  y=228  ← 반투명 빨간 배너
   │  Score 65%  (threshold 65%)      │  y=258  ← Font20 노란색
   └──────────────────────────────────┘  y=295

3. LED_RED (GPIOG, PIN_10)
   Latch 유효 기간 중 2 Hz 블링크
   (HAL_GetTick() % 500 < 250) ? ON : OFF
```

### 9.4 인물 소실 처리

인물이 사라지면 GRU 은닉 상태도 즉시 초기화한다 (`FallDetection_Invalidate`).  
이는 카메라 시야에서 이탈 후 재진입 시 이전 동작의 맥락이 오염되지 않도록 보장한다.

---

## 10. 성능 분석

### 10.1 연산량 비교

| 모델 | MACC | 실행 장치 | 병렬성 |
|---|---|---|---|
| MoveNet Lightning 256 | 측정 불가 (수백M 이상 추정) | NPU 1 GHz | ATON 하드웨어 파이프라인 |
| GRU v26 Stateful | 37,378 | CPU 800 MHz | 단일 스레드 |

GRU의 37,378 MACC는 NPU 워크로드 대비 극히 경미하여, CPU에서 실행해도 전체 파이프라인에 병목을 초래하지 않는다.

### 10.2 메모리 효율 요약

| 영역 | 총용량 | 사용량 | 사용률 |
|---|---|---|---|
| AXISRAM1_S | 1,023 KB | 577 KB | 56.4% |
| PSRAM | 16,384 KB | 2,250 KB | 13.7% |
| GRU 런타임 메모리 (가중치 제외) | — | 4.1 KB | 극소 |
| GRU 가중치 (NOR Flash) | — | 131.9 KB | — |

GRU 추가로 인한 AXISRAM 증가는 주로 은닉 상태 버퍼(384 B)와 활성화 스크래치(2.8 KB)로,  
기존 MoveNet 전용 시스템 대비 메모리 증가량은 **~4 KB 미만**이다.

### 10.3 시간 분석

| 단계 | 주기 | 설명 |
|---|---|---|
| 카메라 캡처 | 30 fps | DCMIPP Pipe2 스냅샷 |
| MoveNet 추론 | 30 fps | NPU 비동기 추론 |
| 포즈 파이프라인 | 15 fps | 2프레임마다 1회 |
| GRU 추론 | 15 fps | 매 유효 짝수 프레임 |
| 낙상 확정 지연 | ~333 ms | 5프레임 × (1/15) s |
| Warmup 지연 | ~1 s | 15프레임 × (1/15) s |

---

*문서 생성 기준: STM32N6570-DK build, 2026-05-12*  
*측정값 출처: `arm-none-eabi-size -A Project.elf`, `Project.map`, `gru_network.h`, `stai_network.h`*
