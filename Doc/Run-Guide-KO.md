# 실행 가이드

이 문서는 STM32N6570-DK 또는 NUCLEO-N657X0-Q 보드에서 이 프로젝트를 빌드하고 실제 하드웨어에서 실행하는 절차를 정리한다.

## 1. 준비물

- 지원 보드
  - STM32N6570-DK
  - NUCLEO-N657X0-Q
- 카메라 모듈
  - 기본 IMX335 모듈 또는 README에 명시된 지원 카메라
- NUCLEO-N657X0-Q 사용 시 표시 장치
  - USB/UVC 출력: PC의 카메라 앱 사용
  - SPI 출력: X-NUCLEO-GFX01M2 디스플레이 사용
- 설치 도구
  - `arm-none-eabi-gcc`
  - `make`
  - STM32CubeProgrammer (`STM32_Programmer_CLI`)
  - STM32 Signing Tool (`STM32_SigningTool_CLI`)

`arm-none-eabi-gcc`, `STM32_Programmer_CLI`, `STM32_SigningTool_CLI`는 터미널 `PATH`에서 실행 가능해야 한다.

## 2. 부팅 모드

STM32N6는 내부 플래시가 없으므로 외부 플래시에 프로그램을 올려 실행한다.

- Development mode: 빌드 결과물을 외부 플래시에 쓰거나 디버거로 SRAM 실행할 때 사용
- Boot from flash mode: 외부 플래시에 기록된 펌웨어를 실제로 부팅할 때 사용

보드별 스위치 위치는 [README Boot Modes](../README.md#boot-modes)의 이미지를 따른다.

## 3. 가장 빠른 실행: 사전 빌드 바이너리 사용

소스 수정 없이 보드에서 바로 실행하려면 사전 빌드된 `.hex` 파일을 사용한다.

### STM32N6570-DK

1. 보드를 Development mode로 설정한다.
2. STM32CubeProgrammer로 다음 파일을 프로그램한다.

   ```text
   Binary/STM32N6570-DK/STM32N6570-DK_GettingStarted_PoseEstimation.hex
   ```

3. 보드를 Boot from flash mode로 변경한다.
4. 보드 전원을 껐다 켠다.
5. 카메라 앞에 사람이 보이면 화면에 포즈 스켈레톤이 표시된다.

CLI로 플래시하려면 다음을 실행한다.

```bash
export DKEL="<STM32CubeProgrammer 설치 경로>/bin/ExternalLoader/MX66UW1G45G_STM32N6570-DK.stldr"
STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -el "$DKEL" -hardRst -w Binary/STM32N6570-DK/STM32N6570-DK_GettingStarted_PoseEstimation.hex
```

### NUCLEO-N657X0-Q: USB/UVC 출력

1. 보드를 Development mode로 설정한다.
2. STM32CubeProgrammer로 다음 파일을 프로그램한다.

   ```text
   Binary/NUCLEO-N657X0-Q/USB-UVC-Display/NUCLEO-N657X0-Q_GettingStarted_PoseEstimation-uvc.hex
   ```

3. 보드를 Boot from flash mode로 변경한다.
4. USB OTG 포트 CN8을 PC 또는 USB 호스트에 연결한다.
5. 보드 전원을 껐다 켠다.
6. PC에서 카메라 앱을 실행한다.
7. 카메라 앞에 사람이 보이면 USB 카메라 화면에 포즈 스켈레톤이 표시된다.

CLI로 플래시하려면 다음을 실행한다.

```bash
export NUEL="<STM32CubeProgrammer 설치 경로>/bin/ExternalLoader/MX25UM51245G_STM32N6570-NUCLEO.stldr"
STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -el "$NUEL" -hardRst -w Binary/NUCLEO-N657X0-Q/USB-UVC-Display/NUCLEO-N657X0-Q_GettingStarted_PoseEstimation-uvc.hex
```

### NUCLEO-N657X0-Q: SPI 디스플레이 출력

1. 보드를 Development mode로 설정한다.
2. STM32CubeProgrammer로 다음 파일을 프로그램한다.

   ```text
   Binary/NUCLEO-N657X0-Q/SPI-Display/NUCLEO-N657X0-Q_GettingStarted_PoseEstimation-spi.hex
   ```

3. 보드를 Boot from flash mode로 변경한다.
4. 보드 전원을 껐다 켠다.
5. 카메라 앞에 사람이 보이면 SPI 디스플레이에 포즈 스켈레톤이 표시된다.

CLI로 플래시하려면 다음을 실행한다.

```bash
export NUEL="<STM32CubeProgrammer 설치 경로>/bin/ExternalLoader/MX25UM51245G_STM32N6570-NUCLEO.stldr"
STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -el "$NUEL" -hardRst -w Binary/NUCLEO-N657X0-Q/SPI-Display/NUCLEO-N657X0-Q_GettingStarted_PoseEstimation-spi.hex
```

## 4. 소스에서 빌드 후 실행

소스를 수정한 뒤 실행하려면 보드별 `Application/<보드명>` 디렉토리에서 빌드, 서명, 플래시를 수행한다.

### STM32N6570-DK

```bash
cd Application/STM32N6570-DK
make -j8
make sign
make flash_weights
make flash_tcn_weights
make flash
```

실행 순서:

1. 보드를 Development mode로 설정한다.
2. 위 명령을 실행한다.
3. 보드를 Boot from flash mode로 변경한다.
4. 보드 전원을 껐다 켠다.

`make flash_weights`는 MoveNet 가중치 `Model/STM32N6570-DK/network_data.hex`를 외부 플래시에 기록한다. `make flash_tcn_weights`는 낙상 감지 TCN 가중치 `Model/STM32N6570-DK/TCN/tcn_network_data.hex`를 외부 플래시에 기록한다. 모델을 바꾸지 않았다면 각 가중치는 최초 1회만 실행하면 된다.

정상 실행 시 처음 60프레임 동안 화면에 `Fall detector warming <n>/60`이 표시된다. 이후 낙상 감지 TCN이 실행되며 `NORMAL` 또는 `FALL`, fall/normal score, TCN inference time이 표시된다.

### NUCLEO-N657X0-Q: USB/UVC 출력

USB/UVC 출력은 기본 빌드 옵션이다.

```bash
cd Application/NUCLEO-N657X0-Q
make -j8
make sign
make flash_weights
make flash
```

실행 순서:

1. 보드를 Development mode로 설정한다.
2. 위 명령을 실행한다.
3. USB OTG 포트 CN8을 PC 또는 USB 호스트에 연결한다.
4. 보드를 Boot from flash mode로 변경한다.
5. 보드 전원을 껐다 켠다.
6. PC에서 카메라 앱을 실행한다.

### NUCLEO-N657X0-Q: SPI 디스플레이 출력

SPI 디스플레이를 사용하려면 `SCR_LIB_SCREEN_ITF=SPI`를 지정해서 빌드와 플래시를 같은 옵션으로 실행한다.

```bash
cd Application/NUCLEO-N657X0-Q
make -j8 SCR_LIB_SCREEN_ITF=SPI
make sign SCR_LIB_SCREEN_ITF=SPI
make flash_weights SCR_LIB_SCREEN_ITF=SPI
make flash SCR_LIB_SCREEN_ITF=SPI
```

실행 순서:

1. 보드를 Development mode로 설정한다.
2. 위 명령을 실행한다.
3. 보드를 Boot from flash mode로 변경한다.
4. 보드 전원을 껐다 켠다.

## 5. 플래시에 기록되는 항목

외부 플래시에는 다음 데이터가 사용된다.

| 주소 | 항목 | 파일 |
| --- | --- | --- |
| `0x70000000` | FSBL | `FSBL/ai_fsbl.hex` |
| `0x70100000` | 애플리케이션 | `Application/<보드명>/build/Application/<보드명>/Project_sign.bin` |
| `0x70380000` | 모델 가중치 | `Model/<보드명>/network_data.hex` |
| `0x70680000` | 낙상 감지 TCN 가중치 | `Model/STM32N6570-DK/TCN/tcn_network_data.hex` |

`make flash`는 애플리케이션만 `0x70100000`에 기록한다. MoveNet 가중치는 `make flash_weights`, STM32N6570-DK 낙상 감지 TCN 가중치는 `make flash_tcn_weights`로 별도 기록한다. FSBL을 새로 기록해야 하는 경우에는 사전 빌드 통합 `.hex`를 사용하거나 [STM32CubeProgrammer 수동 플래시 문서](Program-Hex-Files-STM32CubeProgrammer.md)를 따른다.

## 6. 자주 쓰는 명령

```bash
# 빌드 결과 삭제
make clean

# 상세 빌드 로그 출력
make V=1 -j8

# NUCLEO SPI 디스플레이 빌드
make SCR_LIB_SCREEN_ITF=SPI -j8

# STM32N6570-DK 낙상 감지 TCN 가중치 플래시
make flash_tcn_weights
```

## 7. 실행 확인

정상 실행 시 카메라 영상 위에 사람의 주요 관절과 스켈레톤 선이 표시된다.

- STM32N6570-DK: 보드 디스플레이 확인
- NUCLEO USB/UVC: PC 카메라 앱 확인
- NUCLEO SPI: X-NUCLEO-GFX01M2 디스플레이 확인

화면이 나오지 않으면 다음을 먼저 확인한다.

- 보드가 Boot from flash mode인지 확인
- 플래시 후 전원을 완전히 껐다 켰는지 확인
- 카메라 모듈과 디스플레이 연결 확인
- STM32CubeProgrammer의 External Loader가 보드에 맞는지 확인
- 모델을 바꾼 경우 `make flash_weights`를 다시 실행했는지 확인
- STM32N6570-DK에서 낙상 감지 점수가 갱신되지 않으면 `make flash_tcn_weights`를 실행했는지 확인
