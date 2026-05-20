# P37-pure-h64 포팅 가이드

**대상 모델**: `P37-pure-h64-kp13-w40-gru`
**교체 대상**: P38-nv-a65-gru (GRU 128,64 → GRU 64,32)

## 모델 사양

| 항목 | P38-nv-a65 (이전) | P37-pure-h64 (신규) |
|------|-------------------|---------------------|
| GRU units | 128, 64 | **64, 32** |
| Feature set | kp13, 45f | kp13, 45f |
| Window | 40f | 40f |
| Threshold | 0.725 | 0.725 |
| Vote | 5창 중 3개 | 5창 중 3개 |
| Weights | 590 KiB | **275 KiB** |
| Activations | 33 KiB | **21.5 KiB** |
| MACC/frame | 5.84M | **2.76M** |
| event MinP (float) | 0.9079 | 0.8947 |

**교체 이유**: GRU weights 604 KiB → AXI 버스 포화로 7 FPS 제한. 275 KiB로 축소하여 FPS 향상 기대.

---

## 사전 완료 작업

- `Model/STM32N6570-DK/GRU/` — STedgeAI generate 결과 복사 완료
- `Model/p37_pure_h64_kp13_compat.keras` — compat keras 복사 완료
- `Model/p37_pure_h64_kp13/normalization.json` — 정규화 파라미터 복사 완료
- `Model/generate-gru-p37h64-model_STM32N6570-DK.sh` — generate 스크립트 생성 완료

---

## 포팅 절차

### 1. `Application/STM32N6570-DK/Inc/pose_pipeline.h`

#### 1-1. 상수 변경

```c
// 변경 전
#define POSE_KP13_COUNT        7
#define POSE_FEATURE_COUNT    27

// 변경 후
#define POSE_KP13_COUNT       13    /* kp0, kp5~kp16 */
#define POSE_FEATURE_COUNT    45    /* 13×3 + 6 */
```

#### 1-2. `POSE_NORM_MIN` 교체 (P37-pure-h64 학습 통계)

```c
#define POSE_NORM_MIN { \
    /* kp0  (y,x,s) */  0.000000f, 0.000000f, 0.002383f, \
    /* kp5  (y,x,s) */  0.000000f, 0.000000f, 0.004055f, \
    /* kp6  (y,x,s) */  0.000000f, 0.000000f, 0.001079f, \
    /* kp7  (y,x,s) */  0.001145f, 0.000000f, 0.001560f, \
    /* kp8  (y,x,s) */  0.001290f, 0.000724f, 0.004059f, \
    /* kp9  (y,x,s) */  0.005291f, 0.000000f, 0.004168f, \
    /* kp10 (y,x,s) */  0.002378f, 0.000000f, 0.004263f, \
    /* kp11 (y,x,s) */  0.008567f, 0.000359f, 0.008049f, \
    /* kp12 (y,x,s) */  0.002279f, 0.000000f, 0.006607f, \
    /* kp13 (y,x,s) */  0.021298f, 0.000248f, 0.005414f, \
    /* kp14 (y,x,s) */  0.017672f, 0.000000f, 0.004014f, \
    /* kp15 (y,x,s) */  0.021160f, 0.001553f, 0.000135f, \
    /* kp16 (y,x,s) */  0.028301f, 0.000000f, 0.000181f, \
    /* HSSC_y, HSSC_x, RWHC, VHSSC, AHSSC, AHSSC_x */ \
    0.000088f, 0.000158f, 0.024745f, -2.220820f, -18.393295f, -67.749054f }
```

#### 1-3. `POSE_NORM_SCALE` 교체

```c
#define POSE_NORM_SCALE { \
    /* kp0  (y,x,s) */  0.999908f, 0.999997f, 0.880732f, \
    /* kp5  (y,x,s) */  0.997732f, 0.999995f, 0.968784f, \
    /* kp6  (y,x,s) */  0.984889f, 0.999999f, 0.965219f, \
    /* kp7  (y,x,s) */  0.998121f, 0.999957f, 0.981908f, \
    /* kp8  (y,x,s) */  0.989397f, 0.996170f, 0.967367f, \
    /* kp9  (y,x,s) */  0.994656f, 1.000000f, 0.957223f, \
    /* kp10 (y,x,s) */  0.997243f, 0.998899f, 0.948190f, \
    /* kp11 (y,x,s) */  0.991431f, 0.999545f, 0.944041f, \
    /* kp12 (y,x,s) */  0.997647f, 0.999300f, 0.938715f, \
    /* kp13 (y,x,s) */  0.978090f, 0.998411f, 0.965195f, \
    /* kp14 (y,x,s) */  0.981398f, 0.995599f, 0.969497f, \
    /* kp15 (y,x,s) */  0.978840f, 0.996987f, 0.965499f, \
    /* kp16 (y,x,s) */  0.971697f, 0.999525f, 0.968101f, \
    /* HSSC_y, HSSC_x, RWHC, VHSSC, AHSSC, AHSSC_x */ \
    0.991588f, 0.998811f, 21.949295f, 5.239275f, 45.194557f, 125.428223f }
```

---

### 2. `Application/STM32N6570-DK/Src/pose_pipeline.c`

#### 2-1. `kp13_indices` — 7개 → 13개

```c
static const uint8_t kp13_indices[POSE_KP13_COUNT] = {
    POSE_KP_NOSE,
    POSE_KP_LEFT_SHOULDER,  POSE_KP_RIGHT_SHOULDER,
    POSE_KP_LEFT_ELBOW,     POSE_KP_RIGHT_ELBOW,
    POSE_KP_LEFT_WRIST,     POSE_KP_RIGHT_WRIST,
    POSE_KP_LEFT_HIP,       POSE_KP_RIGHT_HIP,
    POSE_KP_LEFT_KNEE,      POSE_KP_RIGHT_KNEE,
    POSE_KP_LEFT_ANKLE,     POSE_KP_RIGHT_ANKLE,
};
```

#### 2-2. engineering feature 인덱스 — `[21..26]` → `[39..44]`

```c
// 변경 전
features[21u] = hssc_y;
features[22u] = hssc_x;
features[23u] = rwhc;
features[24u] = vhssc_ema;
features[25u] = ahssc;
features[26u] = ahssc_x;

// 변경 후
features[39u] = hssc_y;
features[40u] = hssc_x;
features[41u] = rwhc;
features[42u] = vhssc_ema;
features[43u] = ahssc;
features[44u] = ahssc_x;
```

---

### 3. `Application/STM32N6570-DK/Src/app_config.h` 확인

```c
#define GRU_FALL_SCORE_THRESHOLD  0.725f
#define GRU_VOTE_WINDOW           5
#define GRU_VOTE_K                3
#define GRU_WARMUP_FRAMES         40
```

P38과 동일하므로 변경 불필요.

---

### 4. 빌드 및 플래시

```bash
cd Application/STM32N6570-DK

# 빌드
make GCC_PATH=/opt/st/stm32cubeide_2.1.1/plugins/\
com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.linux64_1.0.100.202602081740/tools/bin

# 서명
make sign

# 앱 플래시 + GRU weights 플래시
make flash_gru
```

---

## GRU weights 재생성이 필요한 경우

```bash
cd Model
bash generate-gru-p37h64-model_STM32N6570-DK.sh
```

---

## 주의사항

- `POSE_FEATURE_COUNT`가 27→45로 바뀌므로, `win_buf[40][45]`가 커짐
  - 메모리: `PosePipeline_t` 구조체 크기 증가 (40×27×4 = 4.3 KB → 40×45×4 = 7.2 KB)
  - PSRAM에 할당된 경우 문제없음; AXISRAM이면 `.bss` 사용량 확인 필요
- `--compression high` 옵션이 P38 generate 스크립트에 추가됨 — P37 스크립트에는 미적용 (필요 시 `generate-gru-p37h64-model_STM32N6570-DK.sh`에 추가 후 재생성)
