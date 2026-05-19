# Implementation 04: Spaghetti Code Analysis & Modularization Plan

## 1. 개요
현재 `main.c`는 하드웨어 초기화, 카메라/NPU 파이프라인 관리, 추론, 후처리, Presence 상태 머신, 낙상 감지 로직, 그리고 복잡한 LCD 디스플레이 로직이 한데 섞여 있는 "신(God) 클래스" 형태임. 이를 기능별로 분리하여 가독성을 높이고 유지보수를 용이하게 하기 위한 분석을 수행함.

## 2. 현황 분석 (Spaghetti Point)

### 2.1 `main.c`의 과도한 책임
- **Hardware/System Init**: Clock, NPU, RAM, Cache, Security, IAC 초기화가 모두 포함됨.
- **Inference Flow**: MoveNet(NPU) 실행과 GRU(CPU) 실행이 서로 다른 방식으로 관리됨.
- **Presence Logic**: 사람이 화면에 있는지, 신뢰할 수 있는지 판단하는 State Machine이 `main()` 루프 내에 인라인으로 구현되어 있음.
- **Display Interaction**: `Display_NetworkOutput` 함수가 300라인에 달하며, 수많은 조건문(`#if`, `if`)과 전역 변수에 의존하여 렌더링을 수행함.

### 2.2 밀접한 결합 (Tight Coupling)
- **Post-processing & UI**: 후처리 결과(`pp_output`)가 디스플레이 함수에 직접 전달되며, 디스플레이 함수 내에서 결과 데이터의 유효성을 다시 체크함.
- **Global States**: `pose_debug_metrics`, `fall_state`, `pose_pipeline` 등 상태 변수들이 여러 파일과 `main.c`에 파편화되어 있음.

---

## 3. 모듈화 분리 방안

### 3.1 `system_init.h / .c` (System Infrastructure)
- **역할**: 저수준 하드웨어 설정 및 시스템 서비스 관리.
- **이관 항목**: 
    - `Hardware_init()`, `SystemClock_Config()`, `Security_Config()` 등.
    - `CONSOLE_Config()`, `NPURam_enable()`, `NPUCache_config()`.

### 3.2 `vision_system.h / .c` (Vision Provider)
- **역할**: 카메라 입력부터 NPU 추론 및 후처리까지의 "Pose Detection" 과정을 캡슐화.
- **주요 함수**:
    - `Vision_Init()`: 카메라 파이프라인 및 NPU 모델 초기화.
    - `Vision_Step()`: 카메라 프레임 획득 및 NPU 추론 실행.
    - `Vision_GetPoseOutput()`: 정제된 Keypoint 데이터 반환.

### 3.3 `presence_manager.h / .c` (Context Manager)
- **역할**: 검출된 Pose의 신뢰도를 바탕으로 "사람 존재 여부" 및 "유효성" 판단.
- **이관 항목**: 
    - `Update_PoseDebugMetrics()`의 핵심 로직 (Top-5 confidence, spread check).
    - Presence State Machine (`person_present_confirmed`, `person_missing_count`).

### 3.4 `app_ui.h / .c` (Display & UX)
- **역할**: LCD 출력 및 사용자 피드백(LED, 알람) 관리.
- **주요 함수**:
    - `UI_Init()`: LCD 드라이버 및 레이어 설정.
    - `UI_Refresh(AppStatus_t *status)`: 현재 앱 상태를 받아 화면 갱신. `main.c`의 `Display_NetworkOutput` 대체.
    - `UI_ShowWelcome()`: 초기 로고 및 환영 메시지.

---

## 4. 리팩토링 후 `main.c` 예상 구조

```c
int main(void) {
    // 1. 초기화
    System_Init();
    Vision_Init();
    Presence_Init();
    FallDetection_Init();
    UI_Init();

    while (1) {
        // 2. 입력 및 추론 (Vision)
        PoseData_t pose;
        if (Vision_Step(&pose) == SUCCESS) {
            
            // 3. 상태 업데이트 (Presence & Fall)
            if (Presence_Update(&pose)) {
                FallDetection_Update(&pose);
            } else {
                FallDetection_Invalidate();
            }
        }

        // 4. 출력 (UI & Alarm)
        UI_Refresh(&current_app_state);
        Alarm_Update();
    }
}
```

## 5. 기대 효과
1. **가독성 향상**: `main.c`가 1200라인에서 200라인 이하로 축소됨.
2. **테스트 용이성**: Presence 로직이나 Fall 로직을 카메라 없이 단위 테스트(Mocking)하기 쉬워짐.
3. **확장성**: 추후 다른 AI 모델(예: 행동 인식)을 추가할 때 `Vision_Step`의 인터페이스만 확장하면 됨.
