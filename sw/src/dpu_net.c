#include "dpu_net.h"

#include <string.h>

void dpu_net_out_shape(const dpu_model_t *m, int l, int *c, int *h, int *w) {
    const uint16_t *L = m->layers[l];
    const int s = L[5];
    int oh = (L[0] - 1) / s + 1;
    int ow = (L[1] - 1) / s + 1;
    if (L[7]) { oh /= 2; ow /= 2; }
    *c = L[3];
    *h = oh;
    *w = ow;
}

uint32_t dpu_net_layer_macs(const dpu_model_t *m, int l) {
    const uint16_t *L = m->layers[l];
    const uint32_t s = L[5];
    const uint32_t oh = (L[0] - 1) / s + 1, ow = (L[1] - 1) / s + 1;
    const uint32_t taps    = (L[4] == LAYER_PW1X1) ? 1 : 9;
    const uint32_t per_tap = (L[4] == LAYER_DW3X3) ? 1 : L[2];
    return (uint32_t)L[3] * oh * ow * taps * per_tap;
}

int dpu_net_out_slot(int l) {
    return (l % 2 == 0) ? 1 : 2;
}

void dpu_net_build_desc(const dpu_model_t *m, uint32_t *desc) {
    for (int l = 0; l < m->n_layers; l++) {
        const uint16_t *L = m->layers[l];
        uint32_t *d = desc + l * DPU_DESC_WORDS;
        const int in_slot = (l == 0) ? 0 : dpu_net_out_slot(l - 1);
        d[0] = (uint32_t)in_slot * DPU_FMAP_WORDS;
        d[1] = (uint32_t)dpu_net_out_slot(l) * DPU_FMAP_WORDS;
        d[2] = L[8];
        d[3] = (uint32_t)l * DPU_P_LAYER_WORDS;
        d[4] = (uint32_t)L[0] | ((uint32_t)L[1] << 8) | ((uint32_t)L[2] << 16) | ((uint32_t)L[3] << 24);
        d[5] = (uint32_t)L[4] | ((uint32_t)L[5] << 2) | ((uint32_t)L[6] << 4) | ((uint32_t)L[7] << 5);
        d[6] = 0;
        d[7] = 0;
    }
}

void dpu_net_pack_input(const dpu_model_t *m, const int8_t *img, int8_t *fmem) {
    for (int c = 0; c < m->in_c; c++)
        for (int y = 0; y < m->in_h; y++)
            memcpy(fmem + (c * DPU_MAX_HW + y) * DPU_MAX_HW, img + (c * m->in_h + y) * m->in_w, m->in_w);
}

void dpu_net_unpack(const int8_t *fmem, int slot, int c, int h, int w, int8_t *out) {
    const int8_t *base = fmem + (size_t)slot * DPU_FMAP_BYTES;
    for (int ch = 0; ch < c; ch++)
        for (int y = 0; y < h; y++)
            memcpy(out + (ch * h + y) * w, base + (ch * DPU_MAX_HW + y) * DPU_MAX_HW, w);
}

// ARM 참조 구현
static int8_t requant(int32_t acc, int32_t mult, int shift, int relu) {
    const int64_t prod = (int64_t)acc * mult;
    const int64_t mag  = prod < 0 ? -prod : prod;
    const int64_t half = shift ? ((int64_t)1 << (shift - 1)) : 0;
    const int64_t q    = (mag + half) >> shift;
    int64_t v = prod < 0 ? -q : q;
    if (v > 127)  v = 127;
    if (v < -128) v = -128;
    if (relu && v < 0) v = 0;
    return (int8_t)v;
}

static int8_t cpu_tmp[DPU_FMAP_BYTES];
static int8_t cpu_buf[2][DPU_FMAP_BYTES];

void cpu_net_layer(const dpu_model_t *m, int l, const int8_t *in, int8_t *out) {
    const uint16_t *L  = m->layers[l];
    const int in_h = L[0], in_w = L[1], in_ch = L[2], out_ch = L[3];
    const int type = L[4], s = L[5], relu = L[6], pool = L[7];
    const int out_h = (in_h - 1) / s + 1, out_w = (in_w - 1) / s + 1;
    const int8_t  *wl   = m->weights + (size_t)L[8] * 8;
    const int32_t *bias = m->params + l * DPU_P_LAYER_WORDS;
    const int32_t *mult = bias + DPU_MAX_CH;
    const int32_t *shf  = bias + 2 * DPU_MAX_CH;
    int8_t *dst = pool ? cpu_tmp : out;

    for (int oc = 0; oc < out_ch; oc++) {
        const int8_t *w = wl + oc * DPU_MAX_CH * 9;
        for (int oy = 0; oy < out_h; oy++) {
            for (int ox = 0; ox < out_w; ox++) {
                int32_t acc = bias[oc];
                if (type == LAYER_PW1X1) {
                    for (int ic = 0; ic < in_ch; ic++)
                        acc += in[(ic * in_h + oy * s) * in_w + ox * s] * w[ic];
                } else {
                    const int ic0 = (type == LAYER_DW3X3) ? oc : 0;
                    const int ic1 = (type == LAYER_DW3X3) ? oc + 1 : in_ch;
                    for (int ic = ic0; ic < ic1; ic++) {
                        const int8_t *wk = (type == LAYER_DW3X3) ? w : w + ic * 9;
                        for (int kh = 0; kh < 3; kh++) {
                            const int iy = oy * s + kh - 1;
                            if (iy < 0 || iy >= in_h) continue;
                            for (int kw = 0; kw < 3; kw++) {
                                const int ix = ox * s + kw - 1;
                                if (ix < 0 || ix >= in_w) continue;
                                acc += in[(ic * in_h + iy) * in_w + ix] * wk[kh * 3 + kw];
                            }
                        }
                    }
                }
                dst[(oc * out_h + oy) * out_w + ox] = requant(acc, mult[oc], shf[oc], relu);
            }
        }
    }

    if (pool) {
        const int ph = out_h / 2, pw = out_w / 2;
        for (int c = 0; c < out_ch; c++)
            for (int y = 0; y < ph; y++)
                for (int x = 0; x < pw; x++) {
                    const int8_t *p = cpu_tmp + (c * out_h + 2 * y) * out_w + 2 * x;
                    int8_t v = p[0];
                    if (p[1] > v)         v = p[1];
                    if (p[out_w] > v)     v = p[out_w];
                    if (p[out_w + 1] > v) v = p[out_w + 1];
                    out[(c * ph + y) * pw + x] = v;
                }
    }
}

void cpu_net_forward(const dpu_model_t *m, const int8_t *img, int8_t *logits, int8_t *const *layer_out) {
    const int8_t *in = img;
    for (int l = 0; l < m->n_layers; l++) {
        int8_t *out = cpu_buf[l % 2];
        cpu_net_layer(m, l, in, out);
        if (layer_out) {
            int c, h, w;
            dpu_net_out_shape(m, l, &c, &h, &w);
            memcpy(layer_out[l], out, (size_t)c * h * w);
        }
        in = out;
    }
    memcpy(logits, in, m->n_class);
}
