#ifndef DPU_H
#define DPU_H

#include "dpu_types.h"

// ----------------------------------------------------------------------
// 최상위 함수 (Vitis HLS가 이 함수 하나를 IP로 합성함). 한 번 호출 = 디스크립터 n_layers개 실행
//
// 레이어마다: out = [maxpool2x2]( [relu]( requant( layer(in, weight) + bias ) ) )
//   layer   : CONV3X3 / DW3X3 (padding 1) 또는 PW1X1 (padding 0), stride 1 또는 2
//   requant : 출력 채널별 sat8( round_away( acc × mult / 2^shift ) ), acc × mult는 INT64
//   출력 크기: out_h = (in_h − 1) / stride + 1 (풀링 시 / 2), out_w도 같음
//
// 인터페이스 요약
//  - desc : 레이어 디스크립터 테이블 (dpu_types.h)
//  - fmem : 특징맵 영역. 레이어의 입력과 출력은 서로 다른 칸이어야 함 (보통 핑퐁)
//  - wmem, pmem : 가중치, bias·mult·shift
//  - perf : 성능 카운터
//  모든 주소와 n_layers는 AXI4-Lite 번들 하나(CTRL)에 모여 있음
// ----------------------------------------------------------------------
void dpu_top(
    const acc_t  *desc,
    word_t       *fmem,
    const word_t *wmem,
    const acc_t  *pmem,
    ap_uint<8>   n_layers,
    perf_counters_t &perf
);

#endif
