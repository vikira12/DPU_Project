#ifndef DPU_NET_H
#define DPU_NET_H

#include <stdint.h>

// ----------------------------------------------------------------------
// DPU 메모리 배치 (hls/dpu_types.h와 같아야 함)
//   특징맵 칸 하나 = int8 [64][64][64], 행 하나 64바이트
//   fmem = 칸 3개: 0 = 입력, 1·2 = 레이어 출력 핑퐁
// ----------------------------------------------------------------------
#define DPU_MAX_CH        64
#define DPU_MAX_HW        64
#define DPU_FMAP_BYTES    (DPU_MAX_CH * DPU_MAX_HW * DPU_MAX_HW)
#define DPU_FMAP_WORDS    (DPU_FMAP_BYTES / 8)
#define DPU_FMEM_BYTES    (3 * DPU_FMAP_BYTES)
#define DPU_DESC_WORDS    8
#define DPU_P_LAYER_WORDS (3 * DPU_MAX_CH)

#ifdef __cplusplus
extern "C" {
#endif

// 레이어 l의 출력 크기
void dpu_net_out_shape(int l, int *c, int *h, int *w);

// MODEL_LAYERS로 디스크립터 테이블 작성 (uint32 × DPU_DESC_WORDS × MODEL_N_LAYERS)
void dpu_net_build_desc(uint32_t *desc);

// 레이어 l의 출력이 들어가는 fmem 칸 번호
int dpu_net_out_slot(int l);

// 28×28 INT8 이미지를 fmem 입력 칸(채널 0)에 기록
void dpu_net_pack_input(const int8_t *img, int8_t *fmem);

// fmem 칸의 [c][h][w] 영역을 조밀 배열로 복사
void dpu_net_unpack(const int8_t *fmem, int slot, int c, int h, int w, int8_t *out);

// ARM 단독 정수 추론 (DPU와 비트 단위로 같은 결과). logits[MODEL_N_CLASS]
// layer_out이 NULL이 아니면 레이어별 조밀 출력을 layer_out[l]에 복사
void cpu_net_forward(const int8_t *img, int8_t *logits, int8_t *const *layer_out);

#ifdef __cplusplus
}
#endif

#endif
