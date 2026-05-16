# P27-vm0 낙상 감지 모델 명세

> **실험 ID**: P27-vm0  
> **소스 레포**: Falling-Model-Development · `results/phase27_seed_sweep/P27-vm0/`  
> **포팅 대상**: STM32N6570-DK (Cortex-M55 CPU)  
> **상태**: STedgeAI 4.0 generate 완료 · 온디바이스 성능 평가 미실시

---

## 1. 아키텍처

| 항목 | 값 |
|---|---|
| 기본 구조 | 2× Conv1D(64, k=5, causal) → GRU(128) → GRU(64) → Dense(2, softmax) |
| 입력 형태 | `(1, 40, 27)` — 배치=1, 시간=40 step, 특징=27 |
| 출력 형태 | `(1, 2)` — [정상 확률, 낙상 확률] (softmax) |
| 파라미터 수 | — |
| 추론 모드 | 윈도우 단위 (stateless) — 매 프레임 40-step 윈도우 전체 입력 |

### 레이어 구성

```
Input (1, 40, 27)
  └─ Conv1D(64, k=5, padding=causal, activation=relu)
  └─ Conv1D(64, k=5, padding=causal, activation=relu)
  └─ GRU(128, return_sequences=True)
  └─ GRU(64, return_sequences=False)
  └─ Dense(2, activation=softmax)
Output (1, 2)
```

---

## 2. 특징 벡터 (27차원)

### 2.1 키포인트 서브셋 (kp7, 21차원)

COCO 17-keypoint 중 7개 선택 × (y, x, confidence) = 21:

| 인덱스 | COCO 번호 | 부위 |
|---|---|---|
| 0–2 | 0 | nose |
| 3–5 | 5 | left shoulder |
| 6–8 | 6 | right shoulder |
| 9–11 | 7 | left elbow |
| 12–14 | 8 | right elbow |
| 15–17 | 11 | left hip |
| 18–20 | 12 | right hip |

> 좌표 순서: `(y, x, conf)`. HSSC/RWHC 계산은 전체 17 keypoint 기준.

### 2.2 엔지니어링 특징 (6차원)

| 인덱스 | 특징 | 설명 |
|---|---|---|
| 21 | HSSC_y | 상체 중심 y (kp0–kp6 평균) |
| 22 | HSSC_x | 상체 중심 x (kp0–kp6 평균) |
| 23 | RWHC | 전신 바운딩박스 가로/세로 비율 |
| 24 | VHSSC | EMA(α=0.4) 적용 수직 속도 |
| 25 | AHSSC | d(VHSSC_ema)/dt — 수직 가속도 |
| 26 | AHSSC_x | d(VHSSC_x_raw)/dt — 수평 가속도 |

### 2.3 전처리 파이프라인 (Pipeline D)

1. 좌표 클램프 `[0, 1]`
2. 저신뢰 홀드: `conf < 0.15` → One-Euro 필터 업데이트 건너뛰고 이전값 유지
3. One-Euro 필터 (min_cutoff=0.5, beta=0.3, d_cutoff=1.0)
4. Confidence EMA (α=0.5)
5. HSSC_y/x, RWHC, VHSSC_raw 계산
6. VHSSC EMA (α=0.4) → VHSSC feature
7. AHSSC = Δ(VHSSC_ema)/dt, AHSSC_x = Δ(VHSSC_x_raw)/dt

### 2.4 MinMax 정규화

학습 데이터(P27-vm0)에서 산출한 파라미터. `feat_norm = (feat_raw − min) / scale`.

```c
// norm_min[27]
{ 0.000000, 0.000000, 0.002383,   // nose y,x,c
  0.000002, 0.000000, 0.004055,   // l_sho y,x,c
  0.000000, 0.000000, 0.002157,   // r_sho y,x,c
  0.001145, 0.000001, 0.001560,   // l_elb y,x,c
  0.001290, 0.002396, 0.004167,   // r_elb y,x,c
  0.008567, 0.002751, 0.008049,   // l_hip y,x,c
  0.002279, 0.000000, 0.006607,   // r_hip y,x,c
  0.000137, 0.000158, 0.024745,   // HSSC_y, HSSC_x, RWHC
 -2.220820,-18.393295,-67.749054} // VHSSC, AHSSC, AHSSC_x

// norm_scale[27]
{ 0.999908, 0.999997, 0.870511,
  0.997730, 0.999995, 0.968784,
  0.983443, 0.999999, 0.962983,
  0.997804, 0.999956, 0.972651,
  0.986192, 0.994499, 0.963977,
  0.991426, 0.997080, 0.941503,
  0.997647, 0.999300, 0.938715,
  0.991539, 0.998045, 21.949295,
  5.239275, 45.194557,125.179825}
```

---

## 3. 학습 설정

| 파라미터 | 값 |
|---|---|
| 데이터셋 | `dataset/splits_v2_filtered/` (kp7, filtered) |
| 전처리 | `filtered` (Pipeline D) |
| 레이블 | `label` (LB-2) |
| 윈도우 크기 | 40 step (15 fps × 2.67 s) |
| 윈도우 범위 | 3.0 s – 9.0 s (낙상 이벤트 기준) |
| 손실 함수 | Focal Loss (γ=2.0, α=0.25) |
| 클래스 가중치 | True (class_weight=True) |
| Dropout | 0.3 |
| Noise std | 0.02 |
| Negative stride | 2 |
| Seed | 42 |
| Early stop patience | 5 |
| Max epochs | 30 |

---

## 4. 성능 지표

### 4.1 Float 추론 (val threshold 재선택)

| 지표 | 값 |
|---|---|
| Threshold | 0.525 |
| min_consecutive | 3 |
| **MinPR (test_video)** | **0.9280** |
| FallPrecision | 0.9789 |
| NFallPrecision | 0.9280 |
| FallRecall | 0.9726 |
| NFallRecall | 0.9440 |
| F1 | 0.9758 |
| AUC-ROC | 0.9913 |
| TN / FP / FN / TP | 219 / 13 / 17 / 604 |

### 4.2 INT8 추론 (STedgeAI host eval, stm32h7 proxy)

| 지표 | 값 |
|---|---|
| Threshold | **0.50** |
| min_consecutive | **1** |
| **MinPR (INT8)** | **0.9258** |
| FallPrecision | 0.9679 |
| NFallPrecision | 0.9258 |
| FallRecall | 0.9726 |
| NFallRecall | 0.9138 |
| TN / FP / FN / TP | 212 / 20 / 17 / 604 |

> INT8 threshold는 val 세트에서 재선택. Float보다 threshold 낮고 mc=1.  
> host eval은 stm32h7 proxy로 실행 (stm32n6 미지원) — 실제 온디바이스 결과와 다를 수 있음.

---

## 5. STedgeAI 분석 결과

| 항목 | 값 |
|---|---|
| Flash (weights) | **536.52 KiB** |
| 활성화 버퍼 | **40.16 KiB** (41,120 bytes) |
| MACC/추론 | **5,322,472** |
| xSPI2 주소 | `0x70680000` |
| analyze 성공 | True |

### Flash 레이아웃 (xSPI2 NOR 64 MB)

```
0x70000000  FSBL
0x70100000  Application code
0x70380000  MoveNet 256×256 (2.64 MB)  →  0x70634000
0x70680000  P27-vm0 GRU weights (537 KiB)  →  약 0x70706000
            여유 ~57 MB
```

---

## 6. STM32 I/O 상수 (생성된 헤더 기준)

```c
// gru_network.h (STedgeAI generate 출력)
#define STAI_GRU_NETWORK_IN_NUM              (1)
#define STAI_GRU_NETWORK_IN_1_SHAPE          {1, 40, 27}
#define STAI_GRU_NETWORK_IN_1_SIZE           (1080)       // float32 원소 수
#define STAI_GRU_NETWORK_IN_1_SIZE_BYTES     (4320)       // 4320 = 1080 × 4

#define STAI_GRU_NETWORK_OUT_NUM             (1)
#define STAI_GRU_NETWORK_OUT_1_SHAPE         {1, 2}
#define STAI_GRU_NETWORK_OUT_1_SIZE          (2)
#define STAI_GRU_NETWORK_OUT_1_SIZE_BYTES    (8)

#define STAI_GRU_NETWORK_ACTIVATION_1_SIZE_BYTES  (41120)
```

---

## 7. 온디바이스 동작 방식

### 7.1 추론 흐름

```
[MoveNet NPU] → 17 keypoints (y, x, conf)
      ↓
[PosePipeline_Process]
  · 좌표 클램프 · One-Euro filter (all 17 kp)
  · HSSC(kp0-6), RWHC(all 17), VHSSC/AHSSC 계산
  · kp7 추출 → 27-dim · MinMax 정규화
  · 40-step 원형 버퍼에 적재
      ↓  (윈도우 가득 찬 경우에만)
[PosePipeline_GetWindowTimeFirst] → float[40][27]
      ↓
[stai_gru_network_run] (CPU, Cortex-M55)
      ↓
gru_out[0] → [normal_score, fall_score]  (softmax 완료)
      ↓
[낙상 판정]
  fall_score ≥ 0.50  →  fall_consec_count++
  fall_consec_count ≥ 1  →  [ALARM] + PosePipeline_Init 리셋
```

### 7.2 후처리 파라미터

| 파라미터 | 값 | 위치 |
|---|---|---|
| `GRU_FALL_SCORE_THRESHOLD` | **0.50** | `app_config.h` |
| `GRU_FALL_RESET_COUNT` | **1** | `app_config.h` |
| `GRU_FALL_LATCH_MS` | 5000 ms | `app_config.h` |

### 7.3 리셋 정책

- **낙상 확정 시**: `PosePipeline_Init()` 전체 리셋 (필터 상태 포함, 윈도우 비움)  
- **인물 미감지 45프레임 이상**: `FallDetection_Invalidate()` — 윈도우만 비움, 필터 유지

---

## 8. 모델 파일 경로

| 파일 | 경로 |
|---|---|
| 원본 keras | `Falling-Model-Development/results/phase27_seed_sweep/P27-vm0/model.keras` |
| STedgeAI 호환 keras | `Falling-Model-Development/results/phase27_seed_sweep/P27-vm0/model_stedgeai_compat.keras` |
| generate 입력 | `Model/p27_vm0_compat.keras` (위 파일 복사) |
| 생성 C 파일 | `Model/STM32N6570-DK/GRU/gru_network.{c,h}` |
| 가중치 데이터 | `Model/STM32N6570-DK/GRU/gru_network_data.{c,h}` |
| generate 스크립트 | `Model/generate-gru-model_STM32N6570-DK.sh` |

---

## 9. 주의사항

- `model.keras`는 Keras 3.13 형식 — STedgeAI 4.0(Keras 3.7)에서 직접 로드 불가.  
  `model_stedgeai_compat.keras`는 `quantization_config` 필드를 제거한 호환 버전.
- STedgeAI generate 시 `--st-neural-art` 및 `--address` 플래그 사용 금지 (E102 오류).
- 생성된 모델은 **윈도우 단위 (1×40×27) stateless** — v26의 stateful 1-step과 다름.  
  온디바이스에서 hidden state 버퍼(gru_h1/h2) 없음.
- INT8 host eval threshold(0.50, mc=1)와 float threshold(0.525, mc=3)는 다르므로  
  실제 배포는 INT8 기준 사용.
