#include <cstdio>
#include <cstring>
#include <vector>

#include "dpu.h"
#include "../sw/src/dpu_net.h"
#include "../generated/mnist_model.h"
#include "../generated/mnist_testvec.h"

// ----------------------------------------------------------------------
// 테스트 내용
//  1) 레이어 하나씩 실행 (n_layers = 1, 디스크립터 포인터를 옮겨 가며) → 첫 이미지 레이어별 출력 비교
//  2) 네트워크 전체를 한 번에 실행 (n_layers = MODEL_N_LAYERS) → TV_N개 이미지 로짓 비교
//  3) ARM 참조 구현(dpu_net.c)도 같은 테스트 벡터와 비교
// 기준값은 model/dpu_model.py의 정수 추론 결과
// ----------------------------------------------------------------------

static_assert(DPU_FMAP_WORDS == FMAP_WORDS, "fmem 배치 불일치");
static_assert(DPU_P_LAYER_WORDS == P_LAYER_WORDS, "파라미터 배치 불일치");
static_assert(DPU_DESC_WORDS == DESC_WORDS, "디스크립터 크기 불일치");

static std::vector<word_t> fmem(3 * FMAP_WORDS);
static std::vector<word_t> wmem(WMEM_WORDS);
static std::vector<acc_t>  pmem(MAX_LAYERS * P_LAYER_WORDS);
static std::vector<acc_t>  desc(MAX_LAYERS * DESC_WORDS);

static void fmem_to_bytes(std::vector<int8_t> &b) {
    b.resize(3 * FMAP_WORDS * 8);
    for (size_t i = 0; i < fmem.size(); i++)
        for (int k = 0; k < 8; k++) b[i * 8 + k] = (int8_t)fmem[i].range(8 * k + 7, 8 * k).to_int();
}

static void load_input(const int8_t *img) {
    std::vector<int8_t> b(FMAP_WORDS * 8, 0);
    dpu_net_pack_input(img, b.data());
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

int main() {
    // 가중치·파라미터·디스크립터 적재
    for (int i = 0; i < MODEL_W_WORDS; i++) {
        word_t v = 0;
        for (int k = 0; k < 8; k++) v.range(8 * k + 7, 8 * k) = (ap_uint<8>)(uint8_t)MODEL_WEIGHTS[i * 8 + k];
        wmem[i] = v;
    }
    for (int i = 0; i < MODEL_P_WORDS; i++) pmem[i] = MODEL_PARAMS[i];
    uint32_t d32[MODEL_N_LAYERS * DESC_WORDS];
    dpu_net_build_desc(d32);
    for (int i = 0; i < MODEL_N_LAYERS * DESC_WORDS; i++) desc[i] = (int32_t)d32[i];

    int fails = 0;
    perf_counters_t perf;
    std::vector<int8_t> bytes;
    std::vector<int8_t> dense(DPU_FMAP_BYTES);

    // 1) 레이어별
    load_input(TV_IMAGES);
    for (int l = 0; l < MODEL_N_LAYERS; l++) {
        dpu_top(desc.data() + l * DESC_WORDS, fmem.data(), wmem.data(), pmem.data(), 1, perf);
        int c, h, w;
        dpu_net_out_shape(l, &c, &h, &w);
        fmem_to_bytes(bytes);
        dpu_net_unpack(bytes.data(), dpu_net_out_slot(l), c, h, w, dense.data());
        char tag[16];
        snprintf(tag, sizeof tag, "L%d", l);
        const int bad = compare(tag, dense.data(), TV_LAYER_OUT[l], c * h * w);
        printf("layer %d (%dx%dx%d): %s, cycles %u, MACs %u\n", l, c, h, w, bad ? "FAIL" : "ok",
               perf.cycle_count.to_uint(), perf.mac_count.to_uint());
        fails += bad != 0;
    }

    // 2) 전체 네트워크, TV_N개 이미지
    int bad_img = 0, correct = 0;
    for (int n = 0; n < TV_N; n++) {
        load_input(TV_IMAGES + n * MODEL_IN_H * MODEL_IN_W);
        dpu_top(desc.data(), fmem.data(), wmem.data(), pmem.data(), MODEL_N_LAYERS, perf);
        fmem_to_bytes(bytes);
        int8_t logit[MODEL_N_CLASS];
        dpu_net_unpack(bytes.data(), dpu_net_out_slot(MODEL_N_LAYERS - 1), MODEL_N_CLASS, 1, 1, logit);
        bad_img += compare("logit", logit, TV_LOGITS + n * MODEL_N_CLASS, MODEL_N_CLASS) != 0;
        int best = 0;
        for (int k = 1; k < MODEL_N_CLASS; k++) if (logit[k] > logit[best]) best = k;
        correct += best == TV_LABELS[n];
    }
    printf("network: %d/%d images bit-exact, accuracy %d/%d, cycles/inference %u, MACs %u, layers %u\n",
           TV_N - bad_img, TV_N, correct, TV_N, perf.cycle_count.to_uint(), perf.mac_count.to_uint(),
           perf.layer_count.to_uint());
    fails += bad_img != 0;

    // 3) ARM 참조 구현
    int bad_cpu = 0;
    for (int n = 0; n < TV_N; n++) {
        int8_t logit[MODEL_N_CLASS];
        cpu_net_forward(TV_IMAGES + n * MODEL_IN_H * MODEL_IN_W, logit, nullptr);
        bad_cpu += compare("cpu", logit, TV_LOGITS + n * MODEL_N_CLASS, MODEL_N_CLASS) != 0;
    }
    printf("cpu reference: %d/%d images bit-exact\n", TV_N - bad_cpu, TV_N);
    fails += bad_cpu != 0;

    printf(fails ? "*** TEST FAILED ***\n" : "*** TEST PASSED ***\n");
    return fails ? 1 : 0;
}
