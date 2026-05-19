# Implementation 01: P38-nv-a65-gru Model Migration

## 1. 개요
기존 P37-vel 모델(74 features)에서 성능과 안정성이 개선된 P38-nv 모델(45 features)로의 전환을 수행함.

## 2. 주요 수정 사항

### 2.1 Feature Pipeline (`pose_pipeline.h/c`)
- **피처 수 조정**: `POSE_FEATURE_COUNT`를 74에서 45로 변경.
- **Velocity 제거**: 모델 설계 의도에 따라 프레임 간 차분(delta) 피처 계산 로직 삭제.
- **정규화 테이블**: `POSE_NORM_MIN`, `POSE_NORM_SCALE`을 P38-nv 전용 값으로 교체.

### 2.2 Threshold & Logic (`app_config.h`)
- **임계값**: `GRU_FALL_SCORE_THRESHOLD`를 0.50f에서 0.725f로 상향.
- **다수결 로직**: `GRU_FALL_VOTE_WINDOW` (5), `GRU_FALL_VOTE_K` (3) 정의.

## 3. 모델 생성
- `Model/generate-gru-p38nv-model_STM32N6570-DK.sh` 실행을 통해 STedgeAI C 코드 생성.
- 입력 텐서 형상: `(1, 40, 45)`.

## 4. 결과
- FP(False Positive) 감소 및 전반적인 탐지 신뢰도 향상.
- 피처 수 감소에 따른 메모리 점유 및 연산 부하 최적화.
