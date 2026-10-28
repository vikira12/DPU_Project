#include <cstdio>
#include <cstring>
#include <vector>

#include "dpu.h"
#include "../sw/src/dpu_net.h"

// DPU 테스트벤치
//  1) 레이어별 출력 비교
//  2) 전체 네트워크 로짓 비교
//  3) ARM 참조 구현 비교

static_assert(DPU_FMAP_WORDS == FMAP_WORDS, "fmem 배치 불일치");
static_assert(DPU_P_LAYER_WORDS == P_LAYER_WORDS, "파라미터 배치 불일치");
static_assert(DPU_DESC_WORDS == DESC_WORDS, "디스크립터 크기 불일치");
static_assert(DPU_MAX_LAYERS <= MAX_LAYERS, "레이어 수 한도 불일치");

static std::vector<word_t> fmem(3 * FMAP_WORDS);
static std::vector<word_t> wmem(DPU_MAX_W_WORDS);
static std::vector<acc_t>  pmem(DPU_MAX_LAYERS * P_LAYER_WORDS);
static std::vector<acc_t>  desc(DPU_MAX_LAYERS * DESC_WORDS);

static void fmem_to_bytes(std::vector<int8_t> &b) {
    b.resize(3 * FMAP_WORDS * 8);
    for (size_t i = 0; i < fmem.size(); i++)
        for (int k = 0; k < 8; k++) b[i * 8 + k] = (int8_t)fmem[i].range(8 * k + 7, 8 * k).to_int();
}

static void load_input(const dpu_model_t *m, const int8_t *img) {
    std::vector<int8_t> b(FMAP_WORDS * 8, 0);
    dpu_net_pack_input(m, img, b.data());
    for (int i = 0; i < FMAP_WORDS; i++) {
        word_t v = 0;
        for (int k = 0; k < 8; k++) v.range(8 * k + 7, 8 * k) = (ap_uint<8>)(uint8_t)b[i * 8 + k];
        fmem[i] = v;
    }
}

static int compare(const char *tag, const int8_t *got, const int8_t *ref, int n) {
    int bad = 0;
    for (int i = 0; i < n; i++)
        if (got[i] != ref[i]) {
            if (bad < 5) printf("  %s[%d]: got %d, ref %d\n", tag, i, got[i], ref[i]);
            bad++;
        }
    return bad;
}

static int test_model(const dpu_model_t *m) {
    printf("=== %s (%d layers, input %dx%dx%d) ===\n", m->name, m->n_layers, m->in_c, m->in_h, m->in_w);
    if (m->n_layers > DPU_MAX_LAYERS || m->w_words > DPU_MAX_W_WORDS) {
        printf("model too large for this runtime\n");
        return 1;
    }

    // 가중치, 파라미터, 디스크립터
    for (int i = 0; i < m->w_words; i++) {
        word_t v = 0;
        for (int k = 0; k < 8; k++) v.range(8 * k + 7, 8 * k) = (ap_uint<8>)(uint8_t)m->weights[i * 8 + k];
        wmem[i] = v;
    }
    for (int i = 0; i < m->p_words; i++) pmem[i] = m->params[i];
    uint32_t d32[DPU_MAX_LAYERS * DESC_WORDS];
    dpu_net_build_desc(m, d32);
    for (int i = 0; i < m->n_layers * DESC_WORDS; i++) desc[i] = (int32_t)d32[i];

    int fails = 0;
    perf_counters_t perf;
    std::vector<int8_t> bytes;
    std::vector<int8_t> dense(DPU_FMAP_BYTES);
    const int img_bytes = m->in_c * m->in_h * m->in_w;

    // 1) 레이어별
    load_input(m, m->tv_images);
    for (int l = 0; l < m->n_layers; l++) {
        dpu_top(desc.data() + l * DESC_WORDS, fmem.data(), wmem.data(), pmem.data(), 1, perf);
        int c, h, w;
        dpu_net_out_shape(m, l, &c, &h, &w);
        fmem_to_bytes(bytes);
        dpu_net_unpack(bytes.data(), dpu_net_out_slot(l), c, h, w, dense.data());
        char tag[16];
        snprintf(tag, sizeof tag, "L%d", l);
        const int bad = compare(tag, dense.data(), m->tv_layer_out[l], c * h * w);
        static const char *const kind[] = {"conv3x3", "dw3x3", "pw1x1"};
        printf("layer %2d %-7s s%d (%2dx%2dx%2d): %s, cycles %6u, MACs %8u\n", l, kind[m->layers[l][4]],
               m->layers[l][5], c, h, w, bad ? "FAIL" : "ok", perf.cycle_count.to_uint(), perf.mac_count.to_uint());
        fails += bad != 0;
    }

    // 2) 전체 네트워크
    int bad_img = 0, correct = 0;
    for (int n = 0; n < m->tv_n; n++) {
        load_input(m, m->tv_images + n * img_bytes);
        dpu_top(desc.data(), fmem.data(), wmem.data(), pmem.data(), m->n_layers, perf);
        fmem_to_bytes(bytes);
        int8_t logit[16];
        dpu_net_unpack(bytes.data(), dpu_net_out_slot(m->n_layers - 1), m->n_class, 1, 1, logit);
        bad_img += compare("logit", logit, m->tv_logits + n * m->n_class, m->n_class) != 0;
        int best = 0;
        for (int k = 1; k < m->n_class; k++) if (logit[k] > logit[best]) best = k;
        correct += best == m->tv_labels[n];
    }
    printf("network: %d/%d images bit-exact, accuracy %d/%d, cycles/inference %u, MACs %u, layers %u\n",
           m->tv_n - bad_img, m->tv_n, correct, m->tv_n, perf.cycle_count.to_uint(), perf.mac_count.to_uint(),
           perf.layer_count.to_uint());
    fails += bad_img != 0;

    // 3) ARM 참조
    int bad_cpu = 0;
    for (int n = 0; n < m->tv_n; n++) {
        int8_t logit[16];
        cpu_net_forward(m, m->tv_images + n * img_bytes, logit, nullptr);
        bad_cpu += compare("cpu", logit, m->tv_logits + n * m->n_class, m->n_class) != 0;
    }
    printf("cpu reference: %d/%d images bit-exact\n", m->tv_n - bad_cpu, m->tv_n);
    fails += bad_cpu != 0;
    return fails;
}

int main() {
    int fails = 0;
    for (int i = 0; i < DPU_N_MODELS; i++) fails += test_model(DPU_MODELS[i]);
    printf(fails ? "*** TEST FAILED ***\n" : "*** TEST PASSED ***\n");
    return fails ? 1 : 0;
}
