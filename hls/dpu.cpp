#include "dpu.h"

// ----------------------------------------------------------------------
// 구조 개요
//
//  DDR의 레이어 디스크립터 테이블을 순서대로 읽어 레이어를 차례로 실행함 (start 한 번 = 네트워크 전체).
//  레이어마다 필요한 데이터를 먼저 온칩 버퍼(BRAM/LUTRAM)에 올려 두고 계산함.
//
//   0) 디스크립터에서 크기·종류·DDR 오프셋을 읽음
//   1) 레이어 시작 시 bias·재양자화 파라미터·weight를 온칩에 적재
//   2) 출력 행 단위(band)로 필요한 입력 행 × 전체 입력 채널을 온칩에 적재
//   3) 출력 채널 TOC개 묶음마다 TIC × TOC = 64 MAC/cycle, II=1로 계산
//   4) 재양자화·ReLU 후 풀링(또는 그대로 복사)해서 64비트 워드로 묶어 DDR에 기록
//
//  레이어 종류는 같은 64-MAC 엔진을 공유함
//   CONV3X3 : 입력 채널 묶음 g 전체 × 탭 9개
//   DW3X3   : 출력 채널 묶음과 같은 입력 묶음 하나 × 탭 9개, 가중치는 대각 위치에만 있음
//   PW1X1   : 입력 채널 묶음 g 전체 × 가운데 탭 1개
// ----------------------------------------------------------------------

#define W_BUF_DEPTH   (OC_TILES * IC_GROUPS * KK)            // 576
#define IN_BUF_DEPTH  (IC_GROUPS * BAND_ROWS * ROW_WORDS)     // 640
#define OUT_BUF_DEPTH ((TH / 2) * ROW_WORDS)                  // 32

static pixel_t get_byte(word_t v, int b) {
    #pragma HLS INLINE
    return (pixel_t)v.range(8 * b + 7, 8 * b);
}

// ----------------------------------------------------------------------
// 출력 채널별 파라미터 적재: buf[oc % TOC][oc / TOC]  (bias, multiplier, shift)
// ----------------------------------------------------------------------
static void load_oc_params(const acc_t *pmem, int p_off,
                           acc_t bias_buf[TOC][OC_TILES], acc_t mult_buf[TOC][OC_TILES],
                           ap_uint<6> shift_buf[TOC][OC_TILES], ap_uint<8> out_ch) {
    // 세 배열을 따로 버스트로 읽음 (한 루프에서 섞어 읽으면 버스트가 깨짐)
LOAD_BIAS:
    for (int oc = 0; oc < out_ch; oc++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_OUT_CH avg=32
        bias_buf[oc % TOC][oc / TOC] = pmem[p_off + oc];
    }
LOAD_MULT:
    for (int oc = 0; oc < out_ch; oc++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_OUT_CH avg=32
        mult_buf[oc % TOC][oc / TOC] = pmem[p_off + MAX_OUT_CH + oc];
    }
LOAD_SHIFT:
    for (int oc = 0; oc < out_ch; oc++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_OUT_CH avg=32
        shift_buf[oc % TOC][oc / TOC] = (ap_uint<6>)pmem[p_off + 2 * MAX_OUT_CH + oc];
    }
}

// ----------------------------------------------------------------------
// weight 적재
//   DDR: 출력 채널마다 576바이트 칸, 칸 안은 레이어 종류별 배치 (speakable_types.h)
//   온칩: w_buf[oc % TOC][ic % TIC][((oc / TOC) * IC_GROUPS + ic / TIC) * KK + kh * 3 + kw]
//   → 계산 루프에서 64개 뱅크를 같은 주소로 한 번에 읽을 수 있는 배치
//
//   DW3X3은 입력 채널 = 출력 채널이므로 대각 위치(ic % TIC = oc % TOC)에 넣고
//   같은 묶음의 다른 레인 7개는 0으로 채움 → 일반 엔진이 그대로 depthwise를 계산
//   PW1X1은 가운데 탭(k = 4)에만 넣음 (계산 루프도 가운데 탭만 돔)
//
//   출력 채널마다 필요한 워드만 버스트로 읽어 두고, 사이클당 1바이트씩 배치함
// ----------------------------------------------------------------------
static void load_weights(const word_t *wmem, int w_off, weight_t w_buf[TOC][TIC][W_BUF_DEPTH],
                         ap_uint<8> in_ch, ap_uint<8> out_ch, ap_uint<2> layer_type) {
    const bool is_dw = (layer_type == LAYER_DW3X3);
    const bool is_pw = (layer_type == LAYER_PW1X1);
    const int n_bytes = is_dw ? KK : (is_pw ? (int)in_ch : (int)(in_ch * KK));
    const int n_words = (n_bytes + WORD_BYTES - 1) / WORD_BYTES;

    word_t oc_words[OC_WEIGHT_WORDS];

LOAD_W_OC:
    for (int oc = 0; oc < out_ch; oc++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_OUT_CH avg=32
    LOAD_W_WORD:
        for (int wd = 0; wd < n_words; wd++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=OC_WEIGHT_WORDS avg=36
            oc_words[wd] = wmem[w_off + oc * OC_WEIGHT_WORDS + wd];
        }

        // 바이트 j → (ic, k) 를 나눗셈 없이 카운터로 추적 (CONV3X3)
        const int base = (oc / TOC) * IC_GROUPS * KK;
        int ic = 0, k = 0;
    LOAD_W_BYTE:
        for (int j = 0; j < n_bytes; j++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=9 max=MAX_IN_CH*KK avg=288
            const weight_t b  = get_byte(oc_words[j >> 3], j & 7);
            const int      ci = is_dw ? oc : (is_pw ? j : ic);
            const int      kk = is_dw ? j  : (is_pw ? 4 : k);
            const int      idx = base + (ci / TIC) * KK + kk;
            for (int t = 0; t < TIC; t++) {
                #pragma HLS UNROLL
                if (t == ci % TIC)  w_buf[oc % TOC][t][idx] = b;
                else if (is_dw)     w_buf[oc % TOC][t][idx] = 0;
            }
            if (k == KK - 1) { k = 0; ic++; }
            else             { k++; }
        }
    }
}

// ----------------------------------------------------------------------
// 입력 band 적재: 입력 행 ih0 ~ ih0 + BAND_ROWS − 1 (ih0 = 출력 시작 행 × stride − 1)
//   in_buf[c % TIC][col % 8][((c / TIC) * BAND_ROWS + r) * ROW_WORDS + col / 8]
//   이미지 바깥 행(위/아래 패딩)은 0으로 채움
// ----------------------------------------------------------------------
static void load_band(const word_t *fmem, int in_off, pixel_t in_buf[TIC][WORD_BYTES][IN_BUF_DEPTH],
                      int ih0, ap_uint<8> in_h, ap_uint<8> in_w, ap_uint<8> in_ch) {
    const int n_words = (in_w + WORD_BYTES - 1) / WORD_BYTES;

LOAD_IN_CH:
    for (int c = 0; c < in_ch; c++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_IN_CH avg=32
    LOAD_IN_ROW:
        for (int r = 0; r < BAND_ROWS; r++) {
            int ih   = ih0 + r;
            int base = ((c / TIC) * BAND_ROWS + r) * ROW_WORDS;
            if (ih >= 0 && ih < in_h) {
            LOAD_IN_WORD:
                for (int wd = 0; wd < n_words; wd++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=ROW_WORDS avg=4
                    word_t v = fmem[in_off + (c * MAX_IMG_H + ih) * ROW_WORDS + wd];
                    for (int b = 0; b < WORD_BYTES; b++) {
                        #pragma HLS UNROLL
                        in_buf[c % TIC][b][base + wd] = get_byte(v, b);
                    }
                }
            } else {
            ZERO_IN_WORD:
                for (int wd = 0; wd < n_words; wd++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=ROW_WORDS avg=4
                    for (int b = 0; b < WORD_BYTES; b++) {
                        #pragma HLS UNROLL
                        in_buf[c % TIC][b][base + wd] = 0;
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------
// 재양자화: sat8( round_away( acc × mult / 2^shift ) ) → [ReLU]
// ----------------------------------------------------------------------
static pixel_t requant(acc_t acc, acc_t mult, ap_uint<6> shift, ap_uint<1> do_relu) {
    #pragma HLS INLINE
    const ap_int<64> prod = (ap_int<64>)acc * (ap_int<64>)mult;
    const ap_int<64> mag  = (prod < 0) ? (ap_int<64>)(-prod) : prod;
    const ap_int<64> half = (shift == 0) ? (ap_int<64>)0 : (ap_int<64>)((ap_int<64>)1 << (shift - 1));
    const ap_int<64> q    = (mag + half) >> shift;
    ap_int<64> v = (prod < 0) ? (ap_int<64>)(-q) : q;
    if (v > 127)  v = 127;
    if (v < -128) v = -128;
    if (do_relu && v < 0) v = 0;
    return (pixel_t)v;
}

// ----------------------------------------------------------------------
// 출력 채널 타일 ot(TOC개)에 대해 band 안의 rows × out_w 픽셀 계산
//   (r, ow, g, kh, kw) 5중 루프를 하나로 펼쳐서 II=1 파이프라인 한 개로 돌림
//   → 픽셀마다 파이프라인을 다시 채우는 오버헤드가 없음
//   매 사이클: 입력 TIC개 읽기 × 가중치 TOC×TIC개 → 64 MAC, TOC개 누산기 갱신
// ----------------------------------------------------------------------
static void compute_tile(pixel_t in_buf[TIC][WORD_BYTES][IN_BUF_DEPTH],
                         weight_t w_buf[TOC][TIC][W_BUF_DEPTH],
                         acc_t bias_buf[TOC][OC_TILES], acc_t mult_buf[TOC][OC_TILES],
                         ap_uint<6> shift_buf[TOC][OC_TILES],
                         pixel_t out_buf[TOC][2][WORD_BYTES][OUT_BUF_DEPTH],
                         int ot, int rows, int out_w, ap_uint<8> in_w, ap_uint<8> in_ch,
                         ap_uint<2> layer_type, ap_uint<2> stride,
                         ap_uint<1> do_relu, ap_uint<32> &iter_count) {
    // 자동 식 재배열을 끄고 아래의 명시적 덧셈 트리를 그대로 쓰게 함.
    // (켜 두면 HLS가 acc를 트리 맨 앞에 붙여 누산 루프 경로가 9.3ns로 길어짐)
    #pragma HLS EXPRESSION_BALANCE off
    static_assert(TIC == 8, "아래 덧셈 트리는 TIC == 8 기준으로 작성됨");

    const bool is_dw = (layer_type == LAYER_DW3X3);
    const bool is_pw = (layer_type == LAYER_PW1X1);
    const bool s2    = (stride == 2);
    const int  k0 = is_pw ? 1 : 0;                              // 탭 범위 (PW는 가운데 탭만)
    const int  k1 = is_pw ? 1 : KSIZE - 1;
    const int  g0 = is_dw ? ot : 0;                             // 입력 채널 묶음 범위
    const int  g1 = is_dw ? ot : (int)((in_ch + TIC - 1) / TIC) - 1;
    const int  n_tap = is_pw ? 1 : KK;
    const int  total = rows * out_w * (g1 - g0 + 1) * n_tap;

    acc_t acc[TOC];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    int r = 0, ow = 0, g = g0, kh = k0, kw = k0;

COMPUTE:
    for (int it = 0; it < total; it++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=9 max=36864 avg=9216
        const bool first = (g == g0 && kh == k0 && kw == k0);
        const bool last  = (g == g1 && kh == k1 && kw == k1);

        // 좌우 패딩: 이미지 바깥 열은 0 (위아래는 load_band에서 0으로 채워 둠)
        const int  br     = (s2 ? (r << 1) : r) + kh;          // band 안의 입력 행
        const int  col    = (s2 ? (ow << 1) : ow) + kw - 1;
        const bool col_ok = (col >= 0) && (col < in_w);
        const int  cc     = col_ok ? col : 0;
        const int  addr   = (g * BAND_ROWS + br) * ROW_WORDS + (cc >> 3);

        pixel_t x[TIC];
        #pragma HLS ARRAY_PARTITION variable=x complete
        for (int tic = 0; tic < TIC; tic++) {
            #pragma HLS UNROLL
            // in_ch가 TIC의 배수가 아니면 마지막 묶음의 남는 채널은 0으로 (이전 레이어 잔여값 차단)
            const bool ch_ok = (g * TIC + tic) < in_ch;
            x[tic] = (col_ok && ch_ok) ? in_buf[tic][cc & 7][addr] : (pixel_t)0;
        }

        const int widx = (ot * IC_GROUPS + g) * KK + kh * KSIZE + kw;
        for (int toc = 0; toc < TOC; toc++) {
            #pragma HLS UNROLL
            // 8개 곱(16비트)을 3단 덧셈 트리로 합산 (최대 |8 × 128 × 128| < 2^18 → 20비트면 충분)
            // 트리는 누산 루프 밖에서 파이프라인되고, 루프를 도는 경로는 마지막 덧셈 하나뿐
            ap_int<20> p[TIC];
            #pragma HLS ARRAY_PARTITION variable=p complete
            for (int tic = 0; tic < TIC; tic++) {
                #pragma HLS UNROLL
                p[tic] = x[tic] * w_buf[toc][tic][widx];
            }
            const ap_int<20> s01 = p[0] + p[1], s23 = p[2] + p[3];
            const ap_int<20> s45 = p[4] + p[5], s67 = p[6] + p[7];
            const ap_int<20> s03 = s01 + s23,  s47 = s45 + s67;
            const ap_int<20> sum = s03 + s47;

            const acc_t next = (first ? bias_buf[toc][ot] : acc[toc]) + sum;
            acc[toc] = next;

            if (last)
                out_buf[toc][r & 1][ow & 7][(r >> 1) * ROW_WORDS + (ow >> 3)] =
                    requant(next, mult_buf[toc][ot], shift_buf[toc][ot], do_relu);
        }

        // 펼친 루프 인덱스 갱신 (kw → kh → g → ow → r 순)
        if (kw == k1) {
            kw = k0;
            if (kh == k1) {
                kh = k0;
                if (g == g1) {
                    g = g0;
                    if (ow == out_w - 1) { ow = 0; r++; }
                    else                 { ow++; }
                } else {
                    g++;
                }
            } else {
                kh++;
            }
        } else {
            kw++;
        }
    }

    iter_count += total;
}

// ----------------------------------------------------------------------
// 2x2 MaxPooling(stride 2) 또는 그대로 복사해서 DDR 출력 특징맵에 기록
//   한 워드(8픽셀)씩 묶어서 쓰고, 유효 폭을 넘는 바이트는 0으로 채움
//   (행 패딩 영역 안에서만 쓰므로 다른 행/채널을 덮어쓰지 않음)
// ----------------------------------------------------------------------
static void store_tile(pixel_t out_buf[TOC][2][WORD_BYTES][OUT_BUF_DEPTH], word_t *fmem, int out_off,
                       int ot, int oh0, int rows, int out_w, ap_uint<8> out_ch,
                       ap_uint<1> do_pool) {
    const int n_oc = (out_ch - ot * TOC < TOC) ? (int)(out_ch - ot * TOC) : TOC;

    if (do_pool) {
        // (toc, pr, px) 3중 루프를 펼쳐서 사이클당 출력 1픽셀 처리.
        // 2x2 창의 4개 값은 행 홀짝 × 열 하위 비트로 서로 다른 뱅크에 있어 한 사이클에 읽힘.
        // 8픽셀이 모이면 64비트 워드 하나로 기록
        const int pw      = out_w / 2;
        const int p_rows  = rows / 2;
        const int px_n    = ((pw + WORD_BYTES - 1) / WORD_BYTES) * WORD_BYTES;
        const int total   = n_oc * p_rows * px_n;

        int toc = 0, pr = 0, px = 0;
        word_t v = 0;
    POOL:
        for (int it = 0; it < total; it++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=0 max=TOC*(TH/2)*(MAX_IMG_W/2) avg=512
            const int c0   = 2 * px;                         // 짝수 열 → c0, c0+1은 같은 워드
            const int addr = pr * ROW_WORDS + (c0 >> 3);
            pixel_t p00 = out_buf[toc][0][c0 & 7][addr];
            pixel_t p01 = out_buf[toc][0][(c0 + 1) & 7][addr];
            pixel_t p10 = out_buf[toc][1][c0 & 7][addr];
            pixel_t p11 = out_buf[toc][1][(c0 + 1) & 7][addr];
            pixel_t m0 = (p00 > p01) ? p00 : p01;
            pixel_t m1 = (p10 > p11) ? p10 : p11;
            pixel_t m  = (m0 > m1) ? m0 : m1;

            // 리틀엔디언: 먼저 들어온 픽셀이 하위 바이트로 가도록 오른쪽으로 밀어 넣음
            v = v >> 8;
            v.range(63, 56) = (px < pw) ? m : (pixel_t)0;
            if ((px & 7) == 7)
                fmem[out_off + ((ot * TOC + toc) * MAX_IMG_H + oh0 / 2 + pr) * ROW_WORDS + (px >> 3)] = v;

            if (px == px_n - 1) {
                px = 0;
                if (pr == p_rows - 1) { pr = 0; toc++; }
                else                  { pr++; }
            } else {
                px++;
            }
        }
    } else {
        const int n_words = (out_w + WORD_BYTES - 1) / WORD_BYTES;
    COPY_OC:
        for (int toc = 0; toc < n_oc; toc++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=TOC avg=8
        COPY_H:
            for (int r = 0; r < rows; r++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=TH avg=8
            COPY_W:
                for (int wd = 0; wd < n_words; wd++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=ROW_WORDS avg=4
                    word_t v = 0;
                    for (int b = 0; b < WORD_BYTES; b++) {
                        #pragma HLS UNROLL
                        const int px = wd * WORD_BYTES + b;
                        pixel_t p = out_buf[toc][r & 1][b][(r >> 1) * ROW_WORDS + wd];
                        v.range(8 * b + 7, 8 * b) = (px < out_w) ? p : (pixel_t)0;
                    }
                    fmem[out_off + ((ot * TOC + toc) * MAX_IMG_H + oh0 + r) * ROW_WORDS + wd] = v;
                }
            }
        }
    }
}


// ----------------------------------------------------------------------
// 레이어 하나 실행 (디스크립터 하나)
// ----------------------------------------------------------------------
static void run_layer(word_t *fmem, const word_t *wmem, const acc_t *pmem,
                      int in_off, int out_off, int w_off, int p_off,
                      ap_uint<8> in_h, ap_uint<8> in_w, ap_uint<8> in_ch, ap_uint<8> out_ch,
                      ap_uint<2> layer_type, ap_uint<2> stride, ap_uint<1> do_relu, ap_uint<1> do_pool,
                      weight_t w_buf[TOC][TIC][W_BUF_DEPTH],
                      pixel_t in_buf[TIC][WORD_BYTES][IN_BUF_DEPTH],
                      pixel_t out_buf[TOC][2][WORD_BYTES][OUT_BUF_DEPTH],
                      acc_t bias_buf[TOC][OC_TILES], acc_t mult_buf[TOC][OC_TILES],
                      ap_uint<6> shift_buf[TOC][OC_TILES],
                      ap_uint<32> &iter_count, ap_uint<32> &mac_count) {
    load_oc_params(pmem, p_off, bias_buf, mult_buf, shift_buf, out_ch);
    load_weights(wmem, w_off, w_buf, in_ch, out_ch, layer_type);

    const bool s2    = (stride == 2);
    const int  out_h = s2 ? (int)((in_h - 1) >> 1) + 1 : (int)in_h;
    const int  out_w = s2 ? (int)((in_w - 1) >> 1) + 1 : (int)in_w;
    const int  th    = s2 ? TH / 2 : TH;               // band 출력 행 수 (풀링을 위해 짝수)
    const int  n_ot  = (out_ch + TOC - 1) / TOC;

BAND_LOOP:
    for (int oh0 = 0; oh0 < out_h; oh0 += th) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_IMG_H/TH avg=4
        const int rows = (out_h - oh0 < th) ? out_h - oh0 : th;

        load_band(fmem, in_off, in_buf, (s2 ? (oh0 << 1) : oh0) - 1, in_h, in_w, in_ch);

    OC_TILE_LOOP:
        for (int ot = 0; ot < n_ot; ot++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=OC_TILES avg=4
            compute_tile(in_buf, w_buf, bias_buf, mult_buf, shift_buf, out_buf, ot, rows, out_w,
                         in_w, in_ch, layer_type, stride, do_relu, iter_count);
            store_tile(out_buf, fmem, out_off, ot, oh0, rows, out_w, out_ch, do_pool);
        }
    }

    const ap_uint<32> taps    = (layer_type == LAYER_PW1X1) ? 1 : KK;
    const ap_uint<32> per_tap = (layer_type == LAYER_DW3X3) ? (ap_uint<32>)1 : (ap_uint<32>)in_ch;
    mac_count += (ap_uint<32>)out_ch * out_h * out_w * taps * per_tap;
}

// ----------------------------------------------------------------------
// 최상위 함수 (Vitis HLS가 IP로 합성하는 대상)
// ----------------------------------------------------------------------
void dpu_top(
    const acc_t  *desc,
    word_t       *fmem,
    const word_t *wmem,
    const acc_t  *pmem,
    ap_uint<8>   n_layers,
    perf_counters_t &perf
) {
    // ---- AXI 인터페이스 지정 ----
    // 특징맵은 gmem0, 디스크립터·가중치·파라미터는 gmem1. 둘 다 HP 포트에 붙일 64비트 AXI4 Master
    #pragma HLS INTERFACE m_axi port=fmem offset=slave bundle=gmem0 depth=3*FMAP_WORDS max_read_burst_length=16 max_write_burst_length=16
    #pragma HLS INTERFACE m_axi port=wmem offset=slave bundle=gmem1 depth=WMEM_WORDS max_read_burst_length=16
    #pragma HLS INTERFACE m_axi port=pmem offset=slave bundle=gmem1 depth=MAX_LAYERS*P_LAYER_WORDS max_read_burst_length=16
    #pragma HLS INTERFACE m_axi port=desc offset=slave bundle=gmem1 depth=MAX_LAYERS*DESC_WORDS max_read_burst_length=16

    // 버퍼 주소, 제어 레지스터, 스칼라 파라미터를 AXI4-Lite 하나(CTRL)로 통합
    #pragma HLS INTERFACE s_axilite port=desc     bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=fmem     bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=wmem     bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=pmem     bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=n_layers bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=perf     bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return   bundle=CTRL   // start/done/idle 핸드셰이크

    // ---- 온칩 버퍼 ----
    // 가중치: 64뱅크(TOC×TIC) × 576바이트 → 한 사이클에 64개 동시 읽기
    static weight_t w_buf[TOC][TIC][W_BUF_DEPTH];
    #pragma HLS ARRAY_PARTITION variable=w_buf dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=w_buf dim=2 complete

    // 입력 band: 채널(TIC) × 열 하위 3비트(8) = 64뱅크 × 640바이트
    // 열 방향으로도 나눠서 DDR 워드 하나(8바이트)를 한 사이클에 적재할 수 있게 함
    static pixel_t in_buf[TIC][WORD_BYTES][IN_BUF_DEPTH];
    #pragma HLS ARRAY_PARTITION variable=in_buf dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=in_buf dim=2 complete

    // 레이어 결과 (출력 채널 TOC개 × TH행): 풀링/복사 전 임시 저장
    // [toc][행 홀짝][열 % 8][(행 / 2) * 8 + 열 / 8] → 128뱅크 × 32바이트, LUTRAM으로 고정
    // (레지스터로 풀리면 가변 인덱스 멀티플렉서 때문에 LUT/FF가 폭증함)
    static pixel_t out_buf[TOC][2][WORD_BYTES][OUT_BUF_DEPTH];
    #pragma HLS ARRAY_PARTITION variable=out_buf dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=out_buf dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=out_buf dim=3 complete
    #pragma HLS BIND_STORAGE variable=out_buf type=ram_2p impl=lutram

    static acc_t bias_buf[TOC][OC_TILES];
    static acc_t mult_buf[TOC][OC_TILES];
    static ap_uint<6> shift_buf[TOC][OC_TILES];
    #pragma HLS ARRAY_PARTITION variable=bias_buf dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=mult_buf dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=shift_buf dim=1 complete

    ap_uint<32> iter_count = 0;
    ap_uint<32> mac_count  = 0;

LAYER_LOOP:
    for (int l = 0; l < n_layers; l++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_LAYERS avg=6
        acc_t d[DESC_WORDS];
        #pragma HLS ARRAY_PARTITION variable=d complete
    LOAD_DESC:
        for (int i = 0; i < DESC_WORDS; i++) {
            #pragma HLS PIPELINE II=1
            d[i] = desc[l * DESC_WORDS + i];
        }
        const ap_uint<32> dim = d[4];
        const ap_uint<32> cfg = d[5];
        run_layer(fmem, wmem, pmem, d[0], d[1], d[2], d[3],
                  dim.range(7, 0), dim.range(15, 8), dim.range(23, 16), dim.range(31, 24),
                  cfg.range(1, 0), cfg.range(3, 2), cfg[4], cfg[5],
                  w_buf, in_buf, out_buf, bias_buf, mult_buf, shift_buf, iter_count, mac_count);
    }

    perf.cycle_count = iter_count;
    perf.mac_count   = mac_count;
    perf.layer_count = n_layers;
}
