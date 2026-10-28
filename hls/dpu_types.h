#ifndef DPU_TYPES_H
#define DPU_TYPES_H

#include <ap_int.h>

// 데이터 타입
typedef ap_int<8>   pixel_t;    // 특징맵
typedef ap_int<8>   weight_t;   // 가중치
typedef ap_int<32>  acc_t;      // 누산기

// DDR 워드 (INT8 × 8, 리틀엔디언)
typedef ap_uint<64> word_t;
#define WORD_BYTES    8

// 최대 크기
//   fmem: 특징맵 int8 [MAX_IN_CH][MAX_IMG_H][MAX_IMG_W]
//   wmem: 출력 채널당 576바이트 (CONV3X3 [IC][3][3], DW3X3 [3][3], PW1X1 [IC])
//   pmem: 레이어당 bias[64], mult[64], shift[64]
#define MAX_IMG_H     64
#define MAX_IMG_W     64
#define MAX_IN_CH     64
#define MAX_OUT_CH    64
#define KSIZE         3
#define KK            (KSIZE * KSIZE)

#define ROW_WORDS     (MAX_IMG_W / WORD_BYTES)
#define FMAP_WORDS    (MAX_IN_CH * MAX_IMG_H * ROW_WORDS)
#define WEIGHT_WORDS  (MAX_OUT_CH * MAX_IN_CH * KK / WORD_BYTES)
#define OC_WEIGHT_WORDS (MAX_IN_CH * KK / WORD_BYTES)

// 병렬도: TIC × TOC = 64 MAC/cycle
#define TIC           8
#define TOC           8
#define IC_GROUPS     (MAX_IN_CH / TIC)
#define OC_TILES      (MAX_OUT_CH / TOC)

// band 출력 행 수 (짝수), 입력은 위아래 패딩 포함 TH + 2행
#define TH            8
#define BAND_ROWS     (TH + 2)

// 레이어 종류
#define LAYER_CONV3X3 0
#define LAYER_DW3X3   1
#define LAYER_PW1X1   2

// 레이어 디스크립터 (int32 × 8)
//   [0] 입력 오프셋  [1] 출력 오프셋  [2] 가중치 오프셋  [3] 파라미터 오프셋
//   [4] in_h | in_w << 8 | in_ch << 16 | out_ch << 24
//   [5] type | stride << 2 | relu << 4 | pool << 5
//   [6], [7] 예약
#define DESC_WORDS    8
#define MAX_LAYERS    32
#define P_LAYER_WORDS (3 * MAX_OUT_CH)
#define WMEM_WORDS    (8 * MAX_OUT_CH * OC_WEIGHT_WORDS)  // 코시뮬레이션용 깊이

// 성능 카운터
struct perf_counters_t {
    ap_uint<32> cycle_count;   // 연산 루프 사이클
    ap_uint<32> mac_count;     // MAC 수
    ap_uint<32> layer_count;   // 실행한 레이어 수
};

#endif
