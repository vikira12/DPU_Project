#ifndef DPU_H
#define DPU_H

#include "dpu_types.h"

// 가속기 최상위 모듈
//   desc 테이블의 레이어 n_layers개를 차례로 실행
//   레이어: conv → requant → [ReLU] → [2×2 maxpool]
void dpu_top(
    const acc_t  *desc,
    word_t       *fmem,
    const word_t *wmem,
    const acc_t  *pmem,
    ap_uint<8>   n_layers,
    perf_counters_t &perf
);

#endif
