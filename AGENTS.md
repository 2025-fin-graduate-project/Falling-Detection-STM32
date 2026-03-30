# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

STM32N6570-DK 및 NUCLEO-N657X0-Q 보드에서 동작하는 실시간 포즈 추정 애플리케이션입니다. NPU 가속 MoveNet Lightning 모델을 사용하여 카메라 입력에서 13개 신체 키포인트를 실시간으로 검출하고 디스플레이에 스켈레톤을 오버레이합니다. 이 졸업 프로젝트는 낙상 감지(Falling Detection) 기능을 STM32N6 플랫폼에 구현하는 것을 목표로 합니다.

## 빌드 명령어

두 개의 타겟 보드가 있으며, 각 보드의 `Application/<보드명>/` 디렉토리에서 실행:

```bash
# Discovery 보드
cd Application/STM32N6570-DK
make -j8           # 빌드 (디버그)
make sign          # 플래시용 서명
make flash         # 외부 플래시에 프로그래밍

# Nucleo 보드
cd Application/NUCLEO-N657X0-Q
make -j8
make sign
make flash

make clean         # 빌드 결과물 정리
```

**빌드 환경**: `arm-none-eabi-gcc`, ARM Cortex-M55 타겟, `-Os -g3` 최적화

**플래시 메모리 레이아웃**:
- `0x70000000`: FSBL (`ai_fsbl.hex`) — 먼저 프로그래밍
- `0x70100000`: 애플리케이션
- `0x70380000`: 모델 가중치

## 아키텍처

### 부팅 방식

STM32N6은 내부 플래시가 없어 두 가지 동작 모드를 사용:
- **개발 모드**: SWD 디버거로 SRAM에 직접 로드
- **외부 플래시 부팅**: FSBL이 외부 xSPI NOR 플래시에서 애플리케이션을 RAM으로 로드 후 실행

### 핵심 데이터 흐름

```
카메라 → DCMIPP ISP → 이중 파이프라인
                        ├── Pipe 1: 디스플레이용 연속 스트림 (800×480 RGB565)
                        └── Pipe 2: NN 추론용 단일 스냅샷 (192×192 RGB888)
                                          ↓
                              STEdgeAI NPU 추론 (MoveNet Lightning)
                                          ↓
                              포즈 키포인트 후처리 (신뢰도 임계값 0.5)
                                          ↓
                              LTDC 이중 레이어 디스플레이
                              (배경: 카메라 피드 / 전경: 스켈레톤 오버레이)
```

### 주요 소스 파일

- `Application/<보드>/Src/main.c`: 하드웨어 초기화, NN 추론 루프, 메인 로직
- `Application/<보드>/Src/app_camerapipeline.c`: DCMIPP 이중 파이프 카메라 제어
- `Application/<보드>/Src/display_spe.c`: LTDC 디스플레이 렌더링, 스켈레톤 시각화
- `Application/<보드>/Inc/app_config.h`: 주요 설정값 (카메라, 디스플레이, AI 파라미터)

### 미들웨어 구조

- `Middlewares/stedgeai-lib/`: STEdgeAI NN 런타임 (ll_aton) — `NetworkRuntime1100_CM55_GCC.a` 사전 컴파일 라이브러리
- `Middlewares/ai-postprocessing-wrapper/`: 모델 히트맵 → 키포인트 좌표 변환
- `Middlewares/stm32-mw-camera/`: 카메라 센서 드라이버 및 DCMIPP 제어
- `Middlewares/screenl/`: LTDC/USB/SPI 디스플레이 드라이버
- `STM32Cube_FW_N6/`: STM32 HAL/LL 드라이버

## 주요 설정 (`app_config.h`)

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `POSTPROCESS_TYPE` | `POSTPROCESS_SPE_MOVENET_UI` | 포즈 추정 후처리 방식 |
| `ASPECT_RATIO_MODE` | `ASPECT_RATIO_CROP` | 카메라 화면 비율 처리 |
| `AI_POSE_PP_CONF_THRESHOLD` | `0.5f` | 키포인트 신뢰도 임계값 |
| `AI_POSE_PP_POSE_KEYPOINTS_NB` | `13` | 감지 키포인트 수 |
| `CAMERA_FLIP` | `CMW_MIRRORFLIP_NONE` | 카메라 미러/플립 설정 |

## 모델

- **기본 모델**: `Model/STM32N6570-DK/st_movenet_lightning_heatmaps_192_int8_pc.tflite`
- **입력**: 192×192 RGB 히트맵
- **출력**: 13개 신체 키포인트 (머리, 어깨, 팔꿈치, 손목, 엉덩이, 무릎, 발목)
- 커스텀 모델 교체 방법: `Doc/Deploy-your-Quantized-Model.md` 참고

## 배포 설정

`stmaic_STM32N6570-DK.conf` / `stmaic_NUCLEO-N657X0-Q.conf`: ModelZoo 통합용 JSON 설정 파일. 모델 경로, FSBL 위치, 서명/플래시 명령을 정의.

## 참고 문서

- `Doc/Application-Overview.md`: 시스템 아키텍처 및 데이터 흐름
- `Doc/Boot-Overview.md`: 부팅 모드 및 메모리 레이아웃
- `Doc/Build-Options.md`: 카메라/디스플레이/화면비 옵션
- `Doc/Deploy-your-Quantized-Model.md`: 커스텀 모델 통합 절차
