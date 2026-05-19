# Implementation 02: Code Refactoring & Modularization

## 1. 목적
- `main.c`에 집중된 낙상 감지 로직을 별도 모듈로 분리하여 가독성 및 유지보수성 향상.
- 확정된 GRU 모델 외의 레거시(TCN 등) 코드 제거로 소스코드 경량화.

## 2. 분리 방안

### 2.1 `fall_detection.h / .c` 신설
- **역할**: GRU 모델 컨텍스트 관리, 추론 실행, 다수결 투표(Majority Vote) 로직 처리.
- **주요 함수**:
    - `FallDetection_Init()`: GRU 네트워크 및 버퍼 초기화.
    - `FallDetection_Update()`: 새로운 포즈 데이터 입력 시 추론 및 상태 갱신.
    - `FallDetection_Invalidate()`: 사람 미탐지 시 상태 초기화.
    - `FallDetection_RunInference()`: NPU 간섭 해결을 포함한 실제 추론 루틴.

### 2.2 `main.c` 정리
- MoveNet(NPU) 관련 초기화 및 메인 루프만 유지.
- 낙상 관련 전역 변수 및 매크로를 `fall_detection` 모듈로 이관.
- `#if FALL_DETECTION_MODEL` 등 조건부 컴파일 분기 제거 (GRU 전용).

### 2.3 `Makefile` 업데이트
- `Src/fall_detection.c`를 빌드 대상에 추가.
- TCN 관련 소스 및 인클루드 경로 제거.

## 3. 후임자 가이드
- 새로운 모델 도입 시 `fall_detection.c` 내의 `gru_network_context` 및 `stai_gru_network_*` 함수군만 모델 이름에 맞게 수정하면 됨.
- 전처리 피처 순서 변경 시 `pose_pipeline.c` 수정 필요.
