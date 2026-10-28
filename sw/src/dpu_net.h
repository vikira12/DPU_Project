#ifndef DPU_NET_H
#define DPU_NET_H

#include <stdint.h>

// DPU 메모리 배치 (hls/dpu_types.h와 동일)
//   fmem: 특징맵 칸 3개 (0 입력, 1·2 핑퐁)
#define DPU_MAX_CH        64
#define DPU_MAX_HW        64
#define DPU_FMAP_BYTES    (DPU_MAX_CH * DPU_MAX_HW * DPU_MAX_HW)
#define DPU_FMAP_WORDS    (DPU_FMAP_BYTES / 8)
#define DPU_FMEM_BYTES    (3 * DPU_FMAP_BYTES)
#define DPU_DESC_WORDS    8
#define DPU_P_LAYER_WORDS (3 * DPU_MAX_CH)
#define DPU_MAX_LAYERS    16
#define DPU_MAX_W_WORDS   (DPU_MAX_LAYERS * DPU_MAX_CH * DPU_MAX_CH * 9 / 8)

#define LAYER_CONV3X3 0
#define LAYER_DW3X3   1
#define LAYER_PW1X1   2

// 모델 정보
typedef struct {
    const char *name;
    int n_layers, in_c, in_h, in_w, n_class;
    const uint16_t (*layers)[9];       // in_h, in_w, in_ch, out_ch, type, stride, relu, pool, 가중치 시작 워드
    const int8_t  *weights;
    int            w_words;
    const int32_t *params;
    int            p_words;
    // 테스트 벡터
    int            tv_n;
    const int8_t  *tv_images;          // [N][C][H][W]
    const uint8_t *tv_labels;
    const int8_t  *tv_logits;          // [N][n_class]
    const int8_t *const *tv_layer_out; // 첫 이미지 레이어별 출력
} dpu_model_t;

#ifdef __cplusplus
extern "C" {
#endif

extern const dpu_model_t *const DPU_MODELS[];
extern const int DPU_N_MODELS;

// 레이어 출력 크기
void dpu_net_out_shape(const dpu_model_t *m, int l, int *c, int *h, int *w);

// 레이어 MAC 수
uint32_t dpu_net_layer_macs(const dpu_model_t *m, int l);

// 디스크립터 테이블 생성
void dpu_net_build_desc(const dpu_model_t *m, uint32_t *desc);

// 레이어 출력 칸 번호
int dpu_net_out_slot(int l);

// 입력 이미지를 fmem에 기록
void dpu_net_pack_input(const dpu_model_t *m, const int8_t *img, int8_t *fmem);

// fmem 칸에서 [c][h][w] 읽기
void dpu_net_unpack(const int8_t *fmem, int slot, int c, int h, int w, int8_t *out);

// ARM 참조 구현 (레이어 하나, 전체 네트워크)
void cpu_net_layer(const dpu_model_t *m, int l, const int8_t *in, int8_t *out);
void cpu_net_forward(const dpu_model_t *m, const int8_t *img, int8_t *logits, int8_t *const *layer_out);

#ifdef __cplusplus
}
#endif

#endif
