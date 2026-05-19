# STEdgeAI Memory Check

## 목적

초기 배포본 `192x192 MoveNet`과 현재 수정본 `256x256 MoveNet`을 `STEdgeAI`로 같은 조건에서 정식 비교했다.

- 보드: `STM32N6570-DK`
- 타겟: `stm32n6`
- memory pool: `Model/user_neuralart_STM32N6570-DK.json`
- 주요 옵션: `--optimization time`, `--enable-virtual-mem-pools`

## 비교 대상

- 초기 모델: `Model/st_movenet_lightning_heatmaps_192_int8_pc.tflite`
- 현재 모델: `Model/st_movenet_lightning_a100_heatmaps_256_int8.tflite`

## 생성 명령

```bash
/home/min/app/ST/STEdgeAI/3.0/Utilities/linux/stedgeai generate \
  --model Model/st_movenet_lightning_heatmaps_192_int8_pc.tflite \
  --target stm32n6 \
  --st-neural-art default@Model/user_neuralart_STM32N6570-DK.json \
  --input-data-type uint8 \
  --output-data-type int8 \
  --optimization time \
  --workspace /tmp/stedgeai_cmp_192_ws \
  --output /tmp/stedgeai_cmp_192_out

/home/min/app/ST/STEdgeAI/3.0/Utilities/linux/stedgeai generate \
  --model Model/st_movenet_lightning_a100_heatmaps_256_int8.tflite \
  --target stm32n6 \
  --st-neural-art default@Model/user_neuralart_STM32N6570-DK.json \
  --input-data-type uint8 \
  --output-data-type int8 \
  --optimization time \
  --workspace /tmp/stedgeai_cmp_256_ws \
  --output /tmp/stedgeai_cmp_256_out
```

## 요약 비교

| 항목 | 초기 192 모델 | 현재 256 모델 | 차이 |
|---|---:|---:|---:|
| Input tensor | 192x192x3 | 256x256x3 | 증가 |
| Output tensor | 48x48x13 | 64x64x17 | 증가 |
| Params | 2,355,908 | 2,356,308 | +400 |
| Weights (Flash) | 2.536 MB | 2.639 MB | +0.103 MB |
| Activations (RAM) | 1.635 MB | 2.297 MB | +0.662 MB |
| Total footprint | 4.171 MB | 4.936 MB | +0.765 MB |
| hyperRAM usage | 0 B | 0 B | 동일 |

## 메모리 배치 비교

### 초기 192 모델

- `cpuRAM2`: `864 KB / 1 MB` (`84.38%`)
- `npuRAM3`: `0 B / 448 KB`
- `npuRAM4`: `378 KB / 448 KB` (`84.38%`)
- `npuRAM5`: `432 KB / 448 KB` (`96.43%`)
- `npuRAM6`: `0 B / 448 KB`
- `octoFlash`: `2.536 MB / 61 MB`
- `hyperRAM`: `0 B / 16 MB`

Used ranges:

- `cpuRAM2`: `0x34100000-0x341D8000`
- `npuRAM4`: `0x34270000-0x342CE800`
- `npuRAM5`: `0x342E0000-0x3434C000`
- `octoFlash`: `0x70380000-0x706094A0`

### 현재 256 모델

- `cpuRAM2`: `1.000 MB / 1 MB` (`100.00%`)
- `npuRAM3`: `448 KB / 448 KB` (`100.00%`)
- `npuRAM4`: `448 KB / 448 KB` (`100.00%`)
- `npuRAM5`: `432 KB / 448 KB` (`96.43%`)
- `npuRAM6`: `0 B / 448 KB`
- `octoFlash`: `2.639 MB / 61 MB`
- `hyperRAM`: `0 B / 16 MB`

Used ranges:

- `cpuRAM2`: `0x34100000-0x34200000`
- `npuRAM3`: `0x34200000-0x34270000`
- `npuRAM4`: `0x34270000-0x342E0000`
- `npuRAM5`: `0x342E0000-0x3434C000`
- `octoFlash`: `0x70380000-0x706238D0`

## 해석

- 현재 `256` 모델은 초기 `192` 모델보다 메모리를 확실히 더 많이 사용한다.
- 증가의 핵심은 `weights`보다 `activations`다.
- `256` 모델은 `cpuRAM2`를 전부 사용하고, 추가로 `npuRAM3`까지 점유한다.
- 초기 `192` 모델은 `npuRAM3`를 사용하지 않고 `cpuRAM2 + npuRAM4 + npuRAM5` 조합으로 수용된다.
- 두 모델 모두 현재 설정에서는 `hyperRAM`으로 spill되지 않는다.
- 따라서 현재 설정은 성능상 중요한 `hyperRAM spill` 문제는 없는 상태다.

## 참고 리포트

- 192 report: `/tmp/stedgeai_cmp_192_out/network_generate_report.txt`
- 256 report: `/tmp/stedgeai_cmp_256_out/network_generate_report.txt`
- project report: `Model/st_ai_output/network_generate_report.txt`

## 비고

- `192` 모델 생성 시 마지막 `AI RT data/code size` 계산 단계에서 비치명적 실패가 있었지만,
  메모리 사용량 리포트와 메모리 배치 정보는 정상 생성되었다.
- 이번 비교는 `STEdgeAI` 기준의 모델 메모리 비교이며, 전체 펌웨어 `.text/.data/.bss/heap/stack`
  비교와는 별개다.

## 참고: 초기 통합 Hex 프로그래밍 로그

```text
     -------------------------------------------------------------------
                        STM32CubeProgrammer v2.22.0
      -------------------------------------------------------------------

ST-LINK SN  : 003900343234511733353533
ST-LINK FW  : V3J17M10
Board       : STM32N6570-DK
Voltage     : 3.28V
SWD freq    : 8000 KHz
Connect mode: Hot Plug
Reset mode  : Software reset
Device ID   : 0x486
Revision ID : Rev B
Device name : ST32N657
Device type : MCU
Device CPU  : Cortex-M55
BL Version  : --

Hard reset is performed

Opening and parsing file: STM32N6570-DK_GettingStarted_PoseEstimation.hex

Memory Programming ...
  File          : STM32N6570-DK_GettingStarted_PoseEstimation.hex
  Size          : 2.97 MB
  Address       : 0x70000000

Erasing memory corresponding to segment 0:
Erasing external memory sector 0
Erasing memory corresponding to segment 1:
Erasing external memory sectors [16 22]
Erasing memory corresponding to segment 2:
Erasing external memory sectors [56 96]
Download in Progress:
[==================================================] 100%
```

  lsusb에도 ST-LINK 장치가 보이지 않습니다. 보드를 STM32N6570-DK Development mode로 두고 ST-LINK USB-C 포트에 연결한 뒤 아래 순서로 다시 실행하면 됩니다:

  cd Application/STM32N6570-DK
  make flash_weights
  make flash_tcn_weights
  make flash

  현재 코드/서명 쪽 문제는 아니고, PC에서 보드/ST-LINK가 감지되지 않는 상태입니다.

