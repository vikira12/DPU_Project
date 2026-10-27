# DPU

Zybo Z7-20 (xc7z020clg400-1)용 INT8 CNN 가속기. 레이어 디스크립터 테이블을 DDR에서 읽어
start 한 번으로 네트워크 전체를 실행한다. 예제 모델은 MNIST 소형 CNN (레이어 6개 모두 DPU에서 실행).

## 구조

```
PS (ARM)                          PL (dpu_top, 100MHz)
 디스크립터·가중치·파라미터 ─ DDR ─▶ m_axi gmem1 ─▶ 온칩 가중치·파라미터
 입력 이미지 ─────────────── DDR ◀▶ m_axi gmem0 ◀▶ band 버퍼 → 64 MAC/cycle → requant·ReLU·pool
 start/done ─────────────── GP0 ─▶ s_axilite CTRL
```

- 엔진: Conv3×3 / DW3×3 / PW1×1, stride 1·2, 출력 채널별 재양자화(INT32 mult, shift), ReLU, 2×2 max pool
- 최대 크기: 64×64×64 → 64채널. FC는 1×1 공간에서 PW1×1로 처리
- 디스크립터 (int32 × 8): 입력·출력 특징맵 오프셋, 가중치 오프셋, 파라미터 오프셋, 크기, 종류 ([hls/dpu_types.h](hls/dpu_types.h))

## 모델

| 레이어 | 연산 | 출력 |
|---|---|---|
| L0 | Conv3×3 1→16, ReLU, pool | 16×14×14 |
| L1 | Conv3×3 16→32, ReLU, pool | 32×7×7 |
| L2 | Conv3×3 32→64 stride 2, ReLU | 64×4×4 |
| L3 | Conv3×3 64→64, ReLU, pool | 64×2×2 |
| L4 | Conv3×3 64→64, ReLU, pool | 64×1×1 |
| L5 | PW1×1 64→10 (FC) | 10 |

MNIST 시험셋 정확도: float 99.02%, INT8 99.02%

## 폴더

| 폴더 | 내용 |
|---|---|
| `model/` | 학습·양자화·정수 참조 추론·헤더 생성 (`dpu_model.py`) |
| `generated/` | 모델 헤더 `mnist_model.h`, 테스트 벡터 `mnist_testvec.h` |
| `hls/` | HLS 소스 `dpu_top`, 테스트벤치 |
| `sw/src/` | PS 쪽 공용 코드(디스크립터·특징맵 배치·ARM 참조 구현), 보드 검증 프로그램 |
| `vitis/dpu_hls/` | HLS 컴포넌트 설정 |
| `vitis/build_app.py` | 플랫폼·보드 앱 생성/빌드 |
| `vivado/build_bd.tcl` | 블록 디자인 생성 → 비트스트림 → xsa |

## 빌드 순서

```
python model/dpu_model.py                     # 1. 학습 + 헤더 (generated/), --skip-train: 양자화·헤더만
cd vitis/dpu_hls
vitis-run --mode hls --csim --config hls_config.cfg --work_dir dpu_top   # 2. csim
v++ -c --mode hls --config hls_config.cfg --work_dir dpu_top             # 3. 합성 + IP 패키징
cd ../..
vivado -mode batch -source vivado/build_bd.tcl   # 4. BD → 비트스트림 → vivado/dpu_bd_wrapper.xsa
vitis -s vitis/build_app.py                      # 5. 플랫폼·앱 → vitis/dpu_app/build/dpu_app.elf
```

6. 보드: Vitis IDE에서 `vitis/` 작업 공간을 열고 dpu_app 실행 (UART 115200).
   레이어별 비트 비교, 이미지 100장 비트 비교·정확도, DPU/ARM 시간과 가속 배율을 출력한다.

## 검증 현황

- PC csim (Vitis 번들 g++): 레이어 6개 비트 일치, 이미지 100장 로짓 비트 일치, ARM 참조 구현 일치.
  버그 주입 3종(반올림, stride 2 열 위치, pool) 모두 검출
- 추론 1회 연산 사이클 44,368 (100MHz에서 0.44ms, 로드·저장 제외), MAC 2,048,896
- Vivado 100MHz: WNS +0.682ns, LUT 27.8%, BRAM 47.9%
- 보드 (베어메탈, JTAG): 레이어 6개 비트 일치, 이미지 100장 비트 일치·정확도 100/100.
  추론 1회 DPU 707.6us (입력 기록·flush, 가중치 적재, 결과 invalidate 포함) vs ARM 단독 10,798.9us (×15.3)

보드 실행 시 부트 모드 점퍼(JP5)를 JTAG으로 둘 것 (SD 부팅이면 리셋 때 SD 이미지가 올라옴)
