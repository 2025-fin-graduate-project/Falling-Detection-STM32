# P38-nv-a65-gru 마이그레이션 가이드

**이전 모델**: P37-pure-vel-kp13-w40-gru (74 features, velocity 포함)  
**신규 모델**: P38-nv-a65-gru (45 features, velocity 없음)  
**목적**: event_vote MinPR 0.8991 → **0.9079** 향상, FP 23 → **21** 감소, 펌웨어 단순화

---

## 성능 비교

| 지표 | P37-vel (현재) | **P38-nv (신규)** | 변화 |
|------|--------------|-----------------|------|
| test_event_vote MinPR | 0.8991 | **0.9079** | +0.009 |
| FP 비디오 (228) | 23 | **21** | −2 |
| FN 비디오 (625) | 6 | 8 | +2 |
| Fall Precision | 0.9671 | 0.9671 | = |
| Fall Recall | 0.9904 | **0.9872** | −0.003 |
| nFall Precision | 0.9716 | **0.9628** | (FP 감소) |
| nFall Recall | 0.8991 | **0.9079** | +0.009 ✅ |
| 피처 수 | 74 | **45** | −29 |
| Weights | - | **590 KiB** | ← Flash 여유 |
| Activations | - | **33 KiB** | |
| MACC/window | - | **5,843,872** | |
| 추론 임계값 | 0.50 | **0.725** | ↑ |

---

## 모델 선정 근거 (가벼운 모델 검토 포함)

| 모델 | ev_vote | FP | arch | weights | 판정 |
|------|---------|----|------|---------|------|
| **P38-nv-a65-gru** | **0.9079** | **21** | GRU(128,64) | 590 KiB | ✅ **채택** |
| P37-pure-h64-kp13 | 0.8947 | 24 | GRU(64,32) | **275 KiB** | ev 0.013↓, FP+3 — 미채택 |
| P37-pure-kp7 | 0.8860 | 26 | GRU(128,64) | ~450 KiB | ev 0.022↓ — 미채택 |

Flash 여유가 ~60 MB이므로 590 KiB는 제약 없음. 성능 차이가 크기 때문에 가벼운 모델보다 P38-nv 채택.

---

## 아키텍처 변경 없음

```
Input (1, 40, 45) → Conv1D(64,k=5,causal) × 2 → GRU(128) → GRU(64)
                  → Dense(64,relu) → Dropout(0.3) → Dense(2,softmax)
Output: [p_nfall, p_fall]
```
아키텍처 동일, **입력 차원만 74 → 45**.

---

## 필요한 펌웨어 변경 사항

### 1. `pose_pipeline.h` — 4곳 수정

#### 1-A. 피처 개수 상수

```c
// 변경 전
#define POSE_VEL_COUNT        29
#define POSE_FEATURE_COUNT    (POSE_BASE_FEAT_COUNT + POSE_VEL_COUNT)  /* 74 */

// 변경 후 (velocity 제거)
// POSE_VEL_COUNT 삭제
#define POSE_FEATURE_COUNT    POSE_BASE_FEAT_COUNT   /* 45 */
```

#### 1-B. 정규화 테이블 교체

`POSE_NORM_MIN` / `POSE_NORM_SCALE` 74개짜리 테이블을 **45개로 교체**.  
`gru_norm_p38nv.h` 의 `p38nv_norm_min[]` / `p38nv_norm_scale[]` 사용.

```c
// pose_pipeline.h에서 직접 정의하는 경우:
#define POSE_NORM_MIN   { /* gru_norm_p38nv.h 의 45개 값 그대로 */ }
#define POSE_NORM_SCALE { /* gru_norm_p38nv.h 의 45개 값 그대로 */ }

// 또는 헤더 include 후 배열 참조:
#include "gru_norm_p38nv.h"
// p38nv_norm_min[], p38nv_norm_scale[] 사용
```

#### 1-C. `PosePipeline_t` 구조체 — velocity 필드 제거

```c
// 제거할 필드
float32_t prev_raw_vel[POSE_VEL_COUNT];  /* 삭제 */

// 유지 (AHSSC_x 계산에 여전히 필요)
float32_t hssc_x_prev;
float32_t vhssc_x_ema;   /* AHSSC_x 계산용으로 유지 */
```

`PoseFeatureVec_t.f[]` 배열도 74 → 45로 줄어들어 RAM 절약.

#### 1-D. `PosePipeline_Process()` 주석 업데이트

Step 9 (velocity 계산), Step 10 (concatenate) 제거:

```c
// 변경 후 파이프라인:
// 1~8: 동일 (EMA → HSSC → RWHC → VHSSC → AHSSC → 45-feat 조립)
// 9. Normalize 45 features using p38nv stats
// 10. Append 45-float vector to circular window
// (velocity 계산 없음)
```

---

### 2. `app_config.h` — 3곳 수정

```c
// 변경 전
#define GRU_FALL_SCORE_THRESHOLD  0.50f
#define GRU_FALL_RESET_COUNT    1
#define GRU_WARMUP_FRAMES       40

// 변경 후
#define GRU_FALL_SCORE_THRESHOLD  0.725f   /* P38-nv val-reselected threshold */
#define GRU_FALL_VOTE_WINDOW      5        /* 연속 5창 투표 */
#define GRU_FALL_VOTE_K           3        /* 5창 중 3개 이상 → alarm */
#define GRU_WARMUP_FRAMES         40       /* 변경 없음 */
```

> ⚠️ `GRU_FALL_RESET_COUNT=1` (단순 연속) → `vote_window=5, vote_k=3` (다수결)으로  
> 로직이 변경됨. 애플리케이션 코드에서 vote 버퍼 관리 추가 필요.

**포스트프로세싱 C 의사코드:**
```c
static uint8_t vote_buf[5] = {0};
static uint8_t vote_head = 0;

// 매 window 추론 후:
uint8_t window_fall = (gru_output[1] >= GRU_FALL_SCORE_THRESHOLD) ? 1 : 0;
vote_buf[vote_head % GRU_FALL_VOTE_WINDOW] = window_fall;
vote_head++;

uint8_t votes = 0;
for (int i = 0; i < GRU_FALL_VOTE_WINDOW; i++) votes += vote_buf[i];
if (votes >= GRU_FALL_VOTE_K && vote_head >= GRU_FALL_VOTE_WINDOW) {
    trigger_alarm();
    memset(vote_buf, 0, sizeof(vote_buf));
    vote_head = 0;
    network_reset();  /* GRU hidden state 초기화 */
}
```

---

### 3. `pose_pipeline.c` — velocity 블록 제거

`PosePipeline_Process()` 내 velocity 계산 블록 삭제:

```c
/* 삭제할 블록 (대략):
 * --- Step 9: velocity features ---
 * for (int i = 0; i < POSE_VEL_COUNT; i++) {
 *     float raw = base_feat[vel_src_idx[i]];
 *     feat[45 + i] = raw - s->prev_raw_vel[i];
 *     s->prev_raw_vel[i] = raw;
 * }
 */

/* 삭제할 초기화 코드:
 * memset(s->prev_raw_vel, 0, sizeof(s->prev_raw_vel));
 */
```

---

### 4. STedgeAI 재생성

```bash
cd Falling-Detection-STM32/Model
bash generate-gru-p38nv-model_STM32N6570-DK.sh
# → STM32N6570-DK/GRU/ 아래 C 파일 5개 갱신
```

---

## 변경이 필요 없는 것

| 항목 | 이유 |
|------|------|
| One-Euro 필터 파라미터 | 동일 (min_cutoff=0.5, beta=0.3) |
| HSSC 계산 (kp0-6 평균) | 동일 |
| RWHC 계산 | 동일 |
| VHSSC EMA (alpha=0.4) | 동일 |
| AHSSC, AHSSC_x | 동일 |
| 창 크기 (40 프레임) | 동일 |
| 15fps 프레임 레이트 | 동일 |
| FALL_PERSON_MISSING_RESET_COUNT | 동일 (45f) |
| MoveNet 설정 전체 | 변경 없음 |

---

## 마이그레이션 체크리스트

```
□ pose_pipeline.h
  □ POSE_FEATURE_COUNT → 45 (POSE_VEL_COUNT 제거)
  □ PosePipeline_t.prev_raw_vel 필드 제거
  □ POSE_NORM_MIN/SCALE → 45개 값으로 교체 (gru_norm_p38nv.h 참조)
  □ PoseFeatureVec_t.f[74] → f[45]

□ pose_pipeline.c
  □ velocity 계산 블록 제거 (~80 LOC)
  □ prev_raw_vel 초기화 제거

□ app_config.h
  □ GRU_FALL_SCORE_THRESHOLD: 0.50f → 0.725f
  □ GRU_FALL_VOTE_WINDOW, GRU_FALL_VOTE_K 상수 추가

□ fall_detection.c (또는 main loop)
  □ vote_buf[5] 관리 로직 추가
  □ GRU_FALL_RESET_COUNT 방식 → vote 방식 전환

□ STedgeAI
  □ bash generate-gru-p38nv-model_STM32N6570-DK.sh 실행
  □ STM32N6570-DK/GRU/ C파일 갱신 확인

□ 빌드 검증
  □ 입력 텐서 크기 확인: (1, 40, 45)
  □ Flash 크기 확인: ~590 KiB (여유 충분)
  □ RAM 확인: activations 33 KiB
```

---

## 파일 위치 요약

```
Falling-Detection-STM32/Model/
├── p38_nv_a65_kp13/
│   ├── model_best.keras              학습 최고 체크포인트
│   ├── model.keras                   최종 에폭
│   ├── model_stedgeai_compat.keras   STedgeAI 변환용
│   ├── normalization.json            Python용 파라미터 (45쌍)
│   ├── feature_columns.json          피처 순서 (45개)
│   ├── metrics.json                  전체 성능 + STedgeAI analyze
│   ├── gru_norm_p38nv.h              ← C 헤더 (펌웨어에 직접 포함)
│   ├── IMPORT_GUIDE.md               피처 파이프라인 상세
│   └── MIGRATION_GUIDE.md            이 문서
├── p38_nv_a65_kp13_compat.keras      generate 스크립트용 참조
└── generate-gru-p38nv-model_STM32N6570-DK.sh
```
