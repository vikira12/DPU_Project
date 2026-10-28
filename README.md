# DPU — Zybo Z7-20 INT8 CNN 가속기

Zynq-7020 (xc7z020clg400-1) PL에 올린 디스크립터 기반 INT8 CNN 가속기.
ARM이 DDR에 레이어 디스크립터 테이블을 써 두고 start를 한 번 주면, DPU가 디스크립터를 차례로 읽어
네트워크 전체를 끝까지 실행한다. 레이어 구성은 하드웨어에 고정되어 있지 않아서
**같은 비트스트림으로 MNIST CNN과 CIFAR-10 CNN(Depthwise-Separable 포함)을 모두 실행**한다.

| 항목 | 내용 |
|---|---|
| 보드 | Digilent Zybo Z7-20, PL 100MHz, 베어메탈 (Cortex-A9 667MHz) |
| 도구 | Vitis HLS / Vivado / Vitis 2025.2, PyTorch |
| 연산 | Conv3×3, Depthwise 3×3, Pointwise 1×1 (FC 포함), stride 1·2, ReLU, 2×2 max pool |
| 정밀도 | INT8 활성값·가중치, INT32 누산, 출력 채널별 재양자화 |
| 처리량 | 64 MAC/cycle (최대 6.4 GMAC/s) |
| 검증 | PyTorch 정수 참조 ↔ HLS csim ↔ ARM 참조 ↔ 보드 출력 **비트 단위 일치** |

## 결과 요약

| 모델 | 레이어 | MAC/추론 | float | INT8 | DPU 연산 사이클 | MAC 이용률 |
|---|---|---|---|---|---|---|
| MNIST | 6 | 2.05M | 99.02% | 99.02% | 44,368 (0.44ms) | 72% |
| CIFAR-10 | 11 | 23.3M | 86.05% | 86.01% | 399,376 (3.99ms) | 91% |

- 정확도는 각 데이터셋 시험셋 전체(10,000장) 기준, INT8은 하드웨어와 같은 정수 연산으로 계산
- 연산 사이클은 DPU perf 카운터 값 (DDR 적재·저장 제외). MAC 이용률 = MAC ÷ (사이클 × 64)

### 보드 측정 (베어메탈, PL 100MHz, ARM Cortex-A9 667MHz)

| 모델 | 비트 일치 | 정확도 (테스트 벡터) | DPU / 이미지 | ARM / 이미지 | 가속 | DPU 실효 |
|---|---|---|---|---|---|---|
| MNIST | 6/6 레이어, 100/100장 | 100/100 | 2.83ms (353 fps) | 38.2ms (26 fps) | **×13.5** | 1.45 GOPS |
| CIFAR-10 | 11/11 레이어, 200/200장 | 170/200 | 11.0ms (91 fps) | 435.2ms (2.3 fps) | **×39.6** | 4.24 GOPS |

- DPU 시간은 입력 기록·flush, 레이어별 가중치·파라미터 적재, 결과 invalidate까지 포함한 start→done 전체
- ARM은 같은 정수 연산의 C 구현 (`-O2`, NEON 미사용)
- 시간은 글로벌 타이머로 재고, 시작할 때 UART 전송 시간(115200 baud)과 비교해 타이머 환산을 검증한다

### 구현 결과 (Vivado, 100MHz)

| LUT | FF | BRAM | DSP | WNS |
|---|---|---|---|---|
| 14,777 (27.8%) | 17,649 (16.6%) | 67 (47.9%) | 117 (53.2%) | +0.682ns |

## 시스템 구조

```
              ┌───────────── PS (Cortex-A9) ─────────────┐
              │ 디스크립터·가중치·파라미터·입력 → DDR     │
              │ flush → start → done 폴링 → invalidate   │
              └──────┬────────────────────────┬──────────┘
                 GP0 │ (s_axilite CTRL)        │ HP0 (64bit)
                     ▼                         ▲
┌──────────────────────────── PL: dpu_top ─────┴─────────────────────┐
│ LAYER_LOOP: 디스크립터 8워드 읽기 (gmem1)                            │
│   ├─ 가중치 [oc][ic][3][3], bias·mult·shift 적재 (gmem1)            │
│   ├─ 입력 band (TH+2 = 10행) 적재 (gmem0)                           │
│   ├─ COMPUTE (II=1): TIC 8 입력채널 × TOC 8 출력채널 = 64 MAC/cycle │
│   ├─ requant → ReLU → 2×2 max pool                                   │
│   └─ 출력 band 저장 (gmem0)  → 다음 레이어는 다른 칸에서 입력        │
│ perf: cycle_count, mac_count, layer_count                           │
└──────────────────────────────────────────────────────────────────────┘
```

- **블록 디자인**: PS7 + SmartConnect (GP0 → CTRL 0x4000_0000), SmartConnect (gmem0·gmem1 → HP0), 완료 인터럽트 → IRQ_F2P
- **특징맵 메모리 (fmem)**: [64][64][64] int8 칸 3개. 칸 0 = 입력, 칸 1·2 = 레이어 출력 핑퐁
- **가중치 메모리 (wmem)**: 출력 채널마다 576바이트 칸 (Conv: [ic][3][3], DW: [3][3], PW: [ic])
- **파라미터 메모리 (pmem)**: 레이어마다 bias[64], mult[64], shift[64]

### 레이어 디스크립터 (int32 × 8)

| 워드 | 내용 |
|---|---|
| d0 | 입력 특징맵 오프셋 (fmem, 64bit 워드) |
| d1 | 출력 특징맵 오프셋 |
| d2 | 가중치 오프셋 (wmem 워드) |
| d3 | 파라미터 오프셋 (pmem 워드) |
| d4 | `in_h \| in_w<<8 \| in_ch<<16 \| out_ch<<24` |
| d5 | `type \| stride<<2 \| relu<<4 \| pool<<5` (type 0 Conv3×3, 1 DW3×3, 2 PW1×1) |
| d6, d7 | 예약 |

모델을 바꿀 때는 디스크립터 테이블과 가중치만 바꾸면 되고 HLS·Vivado를 다시 돌릴 필요가 없다.
한도: 레이어당 입력 64×64, 채널 64.

## 양자화

- 대칭 INT8, zero point 0. 가중치는 출력 채널별 scale, 활성값은 레이어 출력마다 scale 1개
- BatchNorm은 앞 Conv에 합친 뒤 양자화 (CIFAR-10)
- 활성값 scale: 학습 이미지 2,000장의 레이어 출력 99.99% 분위수로 보정
- 재양자화 (하드웨어·ARM·Python 동일):

  `y = sat8( round_away( acc × mult / 2^shift ) )`, `mult < 2^31`, acc = INT32 누산 + bias

  부동소수 없이 곱셈 1회와 시프트로 처리하고 반올림 방향까지 맞춰 세 구현의 출력이 비트 단위로 같다.

## 모델

### MNIST (입력 1×28×28)

| L | 연산 | 출력 |
|---|---|---|
| 0 | Conv3×3 1→16, ReLU, pool | 16×14×14 |
| 1 | Conv3×3 16→32, ReLU, pool | 32×7×7 |
| 2 | Conv3×3 32→64 s2, ReLU | 64×4×4 |
| 3 | Conv3×3 64→64, ReLU, pool | 64×2×2 |
| 4 | Conv3×3 64→64, ReLU, pool | 64×1×1 |
| 5 | PW1×1 64→10 (FC) | 10 |

### CIFAR-10 (입력 3×32×32, BN, 데이터 증강, AdamW + OneCycle 20 에폭)

| L | 연산 | 출력 | 연산 사이클 | MAC/cycle | 보드 DPU | 보드 ARM | 가속 |
|---|---|---|---|---|---|---|---|
| 0 | Conv3×3 3→32, ReLU | 32×32×32 | 36,864 | 24 | 718us | 20.1ms | ×28.0 |
| 1 | Conv3×3 32→32, ReLU, pool | 32×16×16 | 147,456 | 64 | 2,804us | 182.5ms | ×65.1 |
| 2 | DW3×3 32, ReLU | 32×16×16 | 9,216 | 8 | 514us | 2.2ms | ×4.3 |
| 3 | PW1×1 32→64, ReLU | 64×16×16 | 8,192 | 64 | 625us | 5.1ms | ×8.2 |
| 4 | Conv3×3 64→64, ReLU, pool | 64×8×8 | 147,456 | 64 | 2,830us | 174.5ms | ×61.6 |
| 5 | DW3×3 64, ReLU | 64×8×8 | 4,608 | 8 | 430us | 1.1ms | ×2.6 |
| 6 | PW1×1 64→64, ReLU | 64×8×8 | 4,096 | 64 | 464us | 2.1ms | ×4.6 |
| 7 | Conv3×3 64→64, ReLU, pool | 64×4×4 | 36,864 | 64 | 1,245us | 42.6ms | ×34.3 |
| 8 | Conv3×3 64→64 s2, ReLU | 64×2×2 | 2,304 | 64 | 661us | 2.6ms | ×3.9 |
| 9 | Conv3×3 64→64, ReLU, pool | 64×1×1 | 2,304 | 64 | 620us | 2.3ms | ×3.7 |
| 10 | PW1×1 64→10 (FC) | 10 | 16 | 40 | 86us | 0.008ms | ×0.1 |
| 합 | | | 399,376 | 58.4 | 11.0ms | 435.2ms | ×39.6 |

보드 열은 레이어 하나씩 start한 시간이다.

- **연산 이용률**: 일반 Conv·PW 레이어는 64 MAC/cycle을 다 쓴다. 첫 레이어는 입력 채널이 3개라 TIC 8 중 3개만 쓰고(37.5%),
  Depthwise는 출력 채널마다 입력 채널이 하나라 TIC 방향 병렬성이 없다(12.5%). DW 레이어는 MAC 수 자체가 작아 전체 이용률은 91%다.
- **메모리 병목**: 계산 루프 시간(3.99ms)은 전체 11.0ms의 36%이고, 나머지는 band·가중치 DDR 적재와 레이어 시작 비용이다.
  연산이 거의 없는 FC 레이어도 86us가 걸려 레이어당 고정 비용이 약 80us임을 알 수 있다.
  연산량이 큰 L1·L4는 ×60 이상 빨라지지만 작은 레이어는 ×3~4에 그친다.

## 검증

1. **Python 정수 참조** (`model/dpu_model.py`): 하드웨어와 같은 정수 연산으로 시험셋 정확도 계산, 테스트 벡터(이미지·레이어별 출력·로짓) 생성
2. **HLS csim** (`hls/dpu_tb.cpp`): 모델마다
   - 레이어 하나씩 실행해 레이어별 출력 비교
   - 네트워크 전체를 start 한 번으로 실행해 테스트 벡터 전체(MNIST 100장, CIFAR-10 200장) 로짓 비교
   - ARM 참조 구현(`sw/src/dpu_net.c`)도 같은 기준값과 비교
3. **버그 주입**: 반올림 방향, stride 2 열 위치, pool 최솟값 오류를 각각 넣어 csim이 모두 검출하는지 확인
4. **보드** (`sw/src/dpu_test.c`): 같은 비교를 실제 DPU로 수행하고 레이어별 DPU/ARM 시간, 전체 추론 시간, 가속 배율, GOPS 출력

csim과 보드 모두 MNIST 6/6 레이어·100/100장, CIFAR-10 11/11 레이어·200/200장 비트 일치.

보드 측정 중 DPU 시간이 계산 루프 사이클(100MHz 기준 최소 시간)보다 짧게 나오는 모순을 발견했다.
UART 전송 시간을 기준으로 타이머를 재 보니 환산이 정확히 4배 어긋나 있었고,
원인은 BSP의 `COUNTS_PER_SECOND`가 괄호 없는 매크로(`XPAR_CPU_CORE_CLOCK_FREQ_HZ/2`)라 `x / (double)COUNTS_PER_SECOND`가
`x / 667M / 2`로 풀린 것이었다. 괄호로 감싸 고치고 UART 기준 점검을 보드 프로그램 시작에 남겨 두었다.

## 폴더

| 경로 | 내용 |
|---|---|
| `model/dpu_model.py` | 학습, BN 합치기, 양자화, 정수 참조 추론, 헤더 생성 |
| `generated/` | 모델 헤더 `{mnist,cifar10}_model.h`, 테스트 벡터 `{mnist,cifar10}_testvec.h` |
| `hls/` | `dpu_top` HLS 소스, 테스트벤치 |
| `sw/src/dpu_net.[ch]` | 디스크립터 생성, 특징맵 배치, ARM 참조 구현 (csim·보드 공용) |
| `sw/src/models.c` | 생성된 헤더를 모델 목록 `DPU_MODELS[]`로 묶음 |
| `sw/src/dpu_test.c` | 보드 검증 프로그램 |
| `vitis/dpu_hls/` | HLS 컴포넌트 설정 |
| `vitis/build_app.py` | 플랫폼·보드 앱 생성/빌드 |
| `vivado/build_bd.tcl` | 블록 디자인 → 비트스트림 → xsa |

## 빌드

```
python model/dpu_model.py --model mnist                  # 1. 학습 + 헤더 (--skip-train: 저장된 가중치로 헤더만)
python model/dpu_model.py --model cifar10 --epochs 20
cd vitis/dpu_hls
vitis-run --mode hls --csim --config hls_config.cfg --work_dir dpu_top   # 2. csim
v++ -c --mode hls --config hls_config.cfg --work_dir dpu_top             # 3. 합성 + IP 패키징
cd ../..
vivado -mode batch -source vivado/build_bd.tcl   # 4. BD → 비트스트림 → vivado/dpu_bd_wrapper.xsa
vitis -s vitis/build_app.py                      # 5. 플랫폼·앱 → vitis/dpu_app/build/dpu_app.elf
```

6. 보드: Vitis IDE에서 `vitis/` 작업 공간을 열고 dpu_app 실행 (UART 115200).
   부트 모드 점퍼(JP5)를 JTAG으로 둘 것 (SD 부팅이면 리셋 때 SD 이미지가 올라옴).

모델만 바꾼 경우 1 → 5 → 6만 하면 된다.

## 한계와 개선 방향

- **레이어마다 DDR 왕복**: 특징맵과 가중치를 레이어마다 DDR에서 다시 읽고 적재·계산·저장이 겹치지 않아 계산 비중이 36%다.
  load·compute·store를 이중 버퍼로 겹치고 버스트 폭을 늘리면 계산 시간(3.99ms, 약 250 fps)에 가까워질 수 있다
- **Depthwise 이용률 12.5%**: DW 전용으로 공간 방향(여러 출력 화소)을 병렬화하는 경로를 두면 개선된다
- **채널 64 한도**: 128채널 이상은 band 버퍼 BRAM이 늘어난다 (현재 BRAM 47.9%). 채널 타일링을 디스크립터에 넣는 방식이 필요
- **잔차 연결·평균 풀링 없음**: ResNet류는 add 모드와 3번째 입력 오프셋을 디스크립터에 추가해야 함
- **베어메탈 전용**: PetaLinux + UIO 드라이버로 옮기면 카메라 입력 등 응용과 연결 가능
