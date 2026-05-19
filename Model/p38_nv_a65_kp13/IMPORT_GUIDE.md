# P38-nv-a65-gru 모델 임포트 가이드

**출처**: Falling-Model-Development / Phase 38 Wave2-NV  
**날짜**: 2026-05-19  
**이전 모델 대비**: P37-vel-kp13 (velocity 74-feature) → **45-feature (velocity 제거)로 event MinPR 향상**

---

## 성능 요약

| 지표 | 값 |
|------|----|
| test_window MinPR | 0.9345 |
| **test_event_vote MinPR** | **0.9079** (최고) |
| Fall Precision / Recall | 0.9671 / 0.9872 |
| nFall Precision / Recall | 0.9628 / 0.9079 |
| FP 비디오 (228개 중) | **21** |
| FN 비디오 (625개 중) | **8** |
| 추론 임계값 | **0.725** |
| 포스트프로세싱 | vote_window=5, vote_k=3 |

**STedgeAI (stm32n6 analyze)**

| | 값 |
|-|----|
| Weights | 590 KiB |
| Activations | 33 KiB |
| MACC/window | 5,843,872 |

---

## 모델 아키텍처

```
Input: (1, 40, 45) float32   ← 40 프레임, 45 피처
  → Conv1D(64, kernel=5, causal, relu)
  → Conv1D(64, kernel=5, causal, relu)
  → GRU(128, return_sequences=True, reset_after=True, unroll=True)
  → GRU(64, reset_after=True, unroll=True)
  → Dense(64, relu)
  → Dropout(0.3)             ← 추론 시 비활성
  → Dense(2, softmax)
Output: [p_nfall, p_fall]    ← index 1이 낙상 확률
```

- **unroll=True**: TFLite/STedgeAI 필수 (TensorListReserve 회피)
- **reset_after=True**: CUDA 호환 GRU 구현
- 온디바이스 stateful 사용 시: `network_reset()` 호출로 hidden state 초기화

---

## 피처 파이프라인 (45개, 프레임당)

### 입력: MoveNet 17-keypoint 출력
`kp[i] = (y, x, score)`, 좌표는 이미지 높이/너비로 정규화된 [0,1] 값

### Step 1: EMA 스무딩 (온디바이스 구현 필요)
각 키포인트 y, x, score에 EMA 필터 적용:
```
ema_out = alpha * new_val + (1 - alpha) * prev_ema
alpha_kp = 0.5   (키포인트 좌표 및 confidence 스무딩)
```

### Step 2: 파생 피처 계산

```c
// HSSC: 머리-어깨-척추 중심 (kp0~6 평균)
// kp0=nose, kp1=left_eye, kp2=right_eye, kp3=left_ear, kp4=right_ear
// kp5=left_shoulder, kp6=right_shoulder
HSSC_y = mean(kp0_y, kp1_y, kp2_y, kp3_y, kp4_y, kp5_y, kp6_y)
HSSC_x = mean(kp0_x, kp1_x, kp2_x, kp3_x, kp4_x, kp5_x, kp6_x)

// RWHC: 전신 바운딩박스 가로/세로 비율
bbox_width  = max(kp0_x..kp16_x) - min(kp0_x..kp16_x)
bbox_height = max(kp0_y..kp16_y) - min(kp0_y..kp16_y)
RWHC = bbox_width / max(bbox_height, 1e-4)

// VHSSC: HSSC_y 1차 미분 (수직 이동 속도) + EMA 스무딩
dt = 1.0 / 15.0   // 15fps
VHSSC_raw = (HSSC_y[t] - HSSC_y[t-1]) / dt
VHSSC     = 0.4 * VHSSC_raw + 0.6 * VHSSC_prev   // alpha_deriv=0.4

// AHSSC: VHSSC 2차 미분 (수직 가속도)
AHSSC = (VHSSC[t] - VHSSC[t-1]) / dt

// AHSSC_x: HSSC_x 2차 미분 (수평 가속도)
VHSSC_x = (HSSC_x[t] - HSSC_x[t-1]) / dt
AHSSC_x = (VHSSC_x[t] - VHSSC_x[t-1]) / dt
```

### Step 3: 피처 벡터 조립 (순서 엄수)

```
index  feature      설명
  0    kp0_y        nose y
  1    kp0_x        nose x
  2    kp0_s        nose confidence
  3    kp5_y        left_shoulder y
  4    kp5_x        left_shoulder x
  5    kp5_s        left_shoulder confidence
  6    kp6_y        right_shoulder y
  7    kp6_x        right_shoulder x
  8    kp6_s        right_shoulder confidence
  9    kp7_y        left_elbow y
 10    kp7_x        left_elbow x
 11    kp7_s        left_elbow confidence
 12    kp8_y        right_elbow y
 13    kp8_x        right_elbow x
 14    kp8_s        right_elbow confidence
 15    kp9_y        left_wrist y
 16    kp9_x        left_wrist x
 17    kp9_s        left_wrist confidence
 18    kp10_y       right_wrist y
 19    kp10_x       right_wrist x
 20    kp10_s       right_wrist confidence
 21    kp11_y       left_hip y
 22    kp11_x       left_hip x
 23    kp11_s       left_hip confidence
 24    kp12_y       right_hip y
 25    kp12_x       right_hip x
 26    kp12_s       right_hip confidence
 27    kp13_y       left_knee y
 28    kp13_x       left_knee x
 29    kp13_s       left_knee confidence
 30    kp14_y       right_knee y
 31    kp14_x       right_knee x
 32    kp14_s       right_knee confidence
 33    kp15_y       left_ankle y
 34    kp15_x       left_ankle x
 35    kp15_s       left_ankle confidence
 36    kp16_y       right_ankle y
 37    kp16_x       right_ankle x
 38    kp16_s       right_ankle confidence
 39    HSSC_y       상체 중심 y
 40    HSSC_x       상체 중심 x
 41    RWHC         바운딩박스 가로/세로 비율
 42    VHSSC        HSSC_y 속도 (EMA 스무딩)
 43    AHSSC        HSSC_y 가속도
 44    AHSSC_x      HSSC_x 가속도
```
전체 45개. **이전 모델(P37-vel)의 velocity 29개 피처(index 45-73) 제거됨.**

### Step 4: MinMax 정규화

`normalization.json`의 min/scale 사용:
```c
feat_norm[i] = clamp((feat_raw[i] - norm_min[i]) / norm_scale[i], 0.0f, 1.0f)
```
- 45쌍의 (min, scale) 값 — `normalization.json` 참조
- 파생 피처(index 39-44)는 scale이 크게 다름 (AHSSC scale≈45, AHSSC_x scale≈125)

---

## 추론 흐름

```
매 프레임 (15fps):
  1. MoveNet → kp[17]
  2. EMA 스무딩 → kp_ema[17]
  3. 파생 피처 계산 → feat[45]
  4. MinMax 정규화 → feat_norm[45]
  5. 원형 버퍼에 push (40프레임 윈도우)
  6. 버퍼 풀 차면: GRU 추론 → score = output[1]
  7. score >= 0.725 → window_fall = 1, else 0
  8. 포스트프로세싱 (5창 투표):
       vote_buf[5] 에 window_fall push
       if sum(vote_buf) >= 3 → 낙상 알람!
       알람 시: hidden_state reset, vote_buf clear
```

**window stride**: 학습은 1프레임 stride, 온디바이스는 매 프레임 추론 권장

---

## 이전 모델(P37-vel)과의 차이점

| 항목 | P37-vel-kp13 | **P38-nv-a65** |
|------|-------------|----------------|
| 피처 수 | 74 (kp13 + velocity) | **45** (kp13만) |
| 임계값 | 0.50 | **0.725** |
| event MinPR | 0.8991 | **0.9079** |
| FP 비디오 | 23 | **21** |
| FN 비디오 | 6 | 8 |
| 펌웨어 복잡도 | velocity 계산 필요 | **단순** |

**펌웨어 변경 사항**:
- `PosePipeline_t` 구조체에서 velocity 관련 필드 제거 불필요 (기존 구조 유지)
- 입력 텐서 크기: `(1, 40, 74)` → `(1, 40, 45)` 로 변경
- 정규화 테이블: 74쌍 → **45쌍**
- 임계값: 0.50 → **0.725** 로 변경

---

## STedgeAI 변환 경로

```bash
# 1. Model/ 디렉토리에서 실행
cd Falling-Detection-STM32/Model

# 2. generate 스크립트 실행 (stedgeai가 PATH에 있어야 함)
bash generate-gru-p38nv-model_STM32N6570-DK.sh

# 또는 수동으로:
stedgeai generate \
    --model p38_nv_a65_kp13_compat.keras \
    --type keras \
    --target stm32n6 \
    --input-data-type float32 \
    --output-data-type float32 \
    --optimization time \
    --name gru_network \
    --output st_ai_output_gru_p38nv
```

compat keras (`model_stedgeai_compat.keras`)는 Keras 3.10+에서 추가된
`quantization_config` 필드를 제거한 버전 (STedgeAI 4.0 / Keras 3.7 호환).

---

## 파일 목록

```
p38_nv_a65_kp13/
├── model_best.keras          학습 중 val MinPR 최고 체크포인트
├── model.keras               최종 에폭 모델
├── model_stedgeai_compat.keras  STedgeAI 변환용 (quantization_config 제거)
├── normalization.json        MinMax 정규화 파라미터 (min: 45, scale: 45)
├── feature_columns.json      피처 순서 (45개 컬럼명)
├── metrics.json              전체 성능 지표 + STedgeAI analyze 결과
└── IMPORT_GUIDE.md           이 문서
p38_nv_a65_kp13_compat.keras  generate 스크립트에서 직접 참조하는 compat 파일
generate-gru-p38nv-model_STM32N6570-DK.sh  STedgeAI C코드 생성 스크립트
```
