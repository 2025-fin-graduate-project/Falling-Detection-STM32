# Falling Detection System Technical Analysis (P38-nv-a65-gru)

본 문서는 STM32N6570-DK 플랫폼 기반의 낙상 감지 시스템 파이프라인과 기술적 고려사항을 정리합니다.

## 1. 전체 파이프라인 구조

```mermaid
graph LR
    A[Camera/DCMIPP] --> B[MoveNet INT8/NPU]
    B --> C[Pose Pipeline / Feature Eng]
    C --> D[GRU Float32/M55]
    D --> E[Majority Vote / Postproc]
    E --> F[LCD/Alarm Output]
```

### 1.1 MoveNet (Pose Estimation)
- **역할**: 입력 영상에서 17개의 신체 주요 키포인트 좌표와 신뢰도 추출.
- **최적화**: `INT8` 양자화 모델을 사용하여 NPU(ATON) 가속기에서 고속 추론 수행.
- **성능**: 시스템 전체 루프의 가장 큰 부하를 담당하며, NPU 하드웨어 가속을 통해 실시간성 확보.

### 1.2 Pose Pipeline (Feature Engineering)
- **역할**: MoveNet의 원시 좌표를 시계열 모델 입력에 적합한 45차원 피처로 가공.
- **핵심 알고리즘**:
    - **One-Euro Filter**: 수치 떨림(Jitter) 제거 및 급격한 움직임 추적.
    - **Engineering Features**: 상체 중심(HSSC), 수직 속도(VHSSC), 수직 가속도(AHSSC) 등 물리적 지표 산출.
    - **Normalization**: P38-nv 데이터셋 통계 기반의 MinMax 스케일링 (`[0, 1]`).

### 1.3 GRU (Fall Detection Model)
- **모델 사양**: P38-nv-a65-gru (45 inputs, 40 timesteps).
- **연산 모드**: `Float32`. M55의 **Helium(MVE)** 기술을 활용한 가속.
- **특이사항**: RNN 계열의 수치 안정성을 위해 부동소수점 연산 유지. 약 5.8M MACC 규모로 CPU 단독으로도 0.1ms 내외의 초고속 추론 가능.

---

## 2. 하드웨어 및 소프트웨어 최적화

### 2.1 이종 컴퓨팅 자원 배분
- **NPU**: 합성곱 연산 위주의 MoveNet 가속.
- **CPU (M55)**: 로직 제어, 전처리 파이프라인 연산, 시계열 모델(GRU) 가속.
- **병목 관리**: NPU 추론 후 OctoSPI 버스 점유권 회수를 위한 하드웨어 Abort 및 상태 강제 초기화 로직 적용.

### 2.2 메모리 및 캐시 관리
- **가중치 배치**: MoveNet(NPU 전용), GRU(내부/외부 Flash 분할 배치).
- **데이터 일관성**: CPU와 NPU/DCMIPP 간 데이터 교환 시 `D-Cache Clean/Invalidate`를 엄격히 수행하여 일관성 보장.

### 2.3 안정성 강화 (Robustness)
- **Majority Vote (3/5)**: 5개 창(Window) 중 3개 이상의 긍정 판정 시 알람 발생. 단발성 포즈 추정 오류에 의한 오탐 방지.
- **Latching Mechanism**: 낙상 확정 후 5초간 알람 상태를 유지하여 사용자 가시성 확보.

---

## 3. 양자화 및 성능 요약

| 단계 | 모델/로직 | 데이터 타입 | 가속기 | 주요 최적화 |
|:---:|:---:|:---:|:---:|:---|
| **Pose** | MoveNet | INT8 | NPU (ATON) | 하드웨어 가속 |
| **Filter** | One-Euro | Float32 | M55 | FPU 연산 |
| **Fall** | GRU | Float32 | M55 | Helium (MVE) |
| **Judge** | Vote | Integer | M55 | 로직 제어 |

---
*Last Updated: 2026-05-19*
