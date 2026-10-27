// 보드 검증 프로그램 (베어메탈)
//  1) 레이어 하나씩 DPU 실행 → 첫 이미지 레이어별 출력 비트 비교
//  2) 네트워크 전체를 start 한 번으로 실행 → TV_N개 이미지 로짓 비트 비교, 정확도, 시간
//  3) ARM 단독 정수 추론 → 같은 비교, 시간 → 가속 배율
#include <stdio.h>
#include <string.h>

#include "xdpu_top.h"
#include "xil_cache.h"
#include "xil_io.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xiltimer.h"

#include "dpu_net.h"
#include "../../generated/mnist_model.h"
#include "../../generated/mnist_testvec.h"

#ifndef XPAR_DPU_0_BASEADDR
#define XPAR_DPU_0_BASEADDR 0x40000000
#endif

#define IMG_BYTES (MODEL_IN_H * MODEL_IN_W)

static XDpu_top dpu;

// DPU가 캐시를 거치지 않고 DDR을 직접 읽고 쓰므로 CPU가 쓴 뒤 flush, 결과를 읽기 전 invalidate
static int8_t   fmem[DPU_FMEM_BYTES]                         __attribute__((aligned(64)));
static int8_t   wmem[MODEL_W_WORDS * 8]                      __attribute__((aligned(64)));
static int32_t  pmem[MODEL_P_WORDS]                          __attribute__((aligned(64)));
static uint32_t desc[MODEL_N_LAYERS * DPU_DESC_WORDS]        __attribute__((aligned(64)));
static int8_t   dense[DPU_FMAP_BYTES];
static int8_t   cpu_layer[MODEL_N_LAYERS][DPU_FMAP_BYTES / 4];

static double elapsed_us(XTime a, XTime b) { return (double)(b - a) * 1e6 / (double)COUNTS_PER_SECOND; }

static void dpu_run(const uint32_t *d, int n_layers) {
    XDpu_top_Set_desc(&dpu, (u64)(UINTPTR)d);
    XDpu_top_Set_fmem(&dpu, (u64)(UINTPTR)fmem);
    XDpu_top_Set_wmem(&dpu, (u64)(UINTPTR)wmem);
    XDpu_top_Set_pmem(&dpu, (u64)(UINTPTR)pmem);
    XDpu_top_Set_n_layers(&dpu, n_layers);
    XDpu_top_Start(&dpu);
    while (!XDpu_top_IsDone(&dpu)) ;
}

// 입력 이미지를 fmem 입력 칸에 쓰고 flush (28행 × 64바이트)
static void put_input(const int8_t *img) {
    dpu_net_pack_input(img, fmem);
    Xil_DCacheFlushRange((UINTPTR)fmem, MODEL_IN_H * DPU_MAX_HW);
}

// 레이어 출력 [c][h][w]가 있는 fmem 영역 invalidate
static void get_output(int slot, int c, int h, int w, int8_t *out) {
    const int8_t *base = fmem + (size_t)slot * DPU_FMAP_BYTES;
    for (int ch = 0; ch < c; ch++)
        Xil_DCacheInvalidateRange((UINTPTR)(base + ch * DPU_MAX_HW * DPU_MAX_HW), h * DPU_MAX_HW);
    dpu_net_unpack(fmem, slot, c, h, w, out);
}

static int count_diff(const int8_t *a, const int8_t *b, int n) {
    int bad = 0;
    for (int i = 0; i < n; i++) bad += a[i] != b[i];
    return bad;
}

static int argmax(const int8_t *v) {
    int best = 0;
    for (int k = 1; k < MODEL_N_CLASS; k++) if (v[k] > v[best]) best = k;
    return best;
}

int main(void) {
    Xil_Out32(XPAR_GLOBAL_TMR_BASEADDR + 0x08U, 0x1U);   // JTAG 실행 시 꺼져 있는 글로벌 타이머 켜기
    xil_printf("\r\n=== DPU MNIST test ===\r\n");

    if (XDpu_top_Initialize(&dpu, XPAR_DPU_0_BASEADDR) != XST_SUCCESS) {
        xil_printf("DPU init failed\r\n");
        return -1;
    }

    memcpy(wmem, MODEL_WEIGHTS, sizeof wmem);
    memcpy(pmem, MODEL_PARAMS, sizeof pmem);
    dpu_net_build_desc(desc);
    memset(fmem, 0, sizeof fmem);
    Xil_DCacheFlushRange((UINTPTR)wmem, sizeof wmem);
    Xil_DCacheFlushRange((UINTPTR)pmem, sizeof pmem);
    Xil_DCacheFlushRange((UINTPTR)desc, sizeof desc);
    Xil_DCacheFlushRange((UINTPTR)fmem, sizeof fmem);

    int fails = 0;

    // 1) 레이어별
    int8_t *layer_out[MODEL_N_LAYERS];
    for (int l = 0; l < MODEL_N_LAYERS; l++) layer_out[l] = cpu_layer[l];
    int8_t logit_cpu[MODEL_N_CLASS];
    cpu_net_forward(TV_IMAGES, logit_cpu, layer_out);
    put_input(TV_IMAGES);
    for (int l = 0; l < MODEL_N_LAYERS; l++) {
        dpu_run(desc + l * DPU_DESC_WORDS, 1);
        const XDpu_top_Perf p = XDpu_top_Get_perf(&dpu);
        int c, h, w;
        dpu_net_out_shape(l, &c, &h, &w);
        get_output(dpu_net_out_slot(l), c, h, w, dense);
        const int bad_ref = count_diff(dense, TV_LAYER_OUT[l], c * h * w);
        const int bad_cpu = count_diff(dense, layer_out[l], c * h * w);
        xil_printf("layer %d (%dx%dx%d): ref %s, arm %s, cycles %d\r\n", l, c, h, w,
                   bad_ref ? "FAIL" : "ok", bad_cpu ? "FAIL" : "ok", (int)p.word_0);
        fails += (bad_ref || bad_cpu);
    }

    // 2) DPU 전체 네트워크
    int bad_dpu = 0, ok_dpu = 0;
    double t_dpu = 0;
    XDpu_top_Perf perf = {0, 0, 0};
    for (int n = 0; n < TV_N; n++) {
        const int8_t *img = TV_IMAGES + n * IMG_BYTES;
        int8_t logit[MODEL_N_CLASS];
        XTime t0, t1;
        XTime_GetTime(&t0);
        put_input(img);
        dpu_run(desc, MODEL_N_LAYERS);
        get_output(dpu_net_out_slot(MODEL_N_LAYERS - 1), MODEL_N_CLASS, 1, 1, logit);
        XTime_GetTime(&t1);
        t_dpu += elapsed_us(t0, t1);
        perf = XDpu_top_Get_perf(&dpu);
        bad_dpu += count_diff(logit, TV_LOGITS + n * MODEL_N_CLASS, MODEL_N_CLASS) != 0;
        ok_dpu  += argmax(logit) == TV_LABELS[n];
    }

    // 3) ARM 단독
    int bad_cpu = 0;
    double t_cpu = 0;
    for (int n = 0; n < TV_N; n++) {
        int8_t logit[MODEL_N_CLASS];
        XTime t0, t1;
        XTime_GetTime(&t0);
        cpu_net_forward(TV_IMAGES + n * IMG_BYTES, logit, NULL);
        XTime_GetTime(&t1);
        t_cpu += elapsed_us(t0, t1);
        bad_cpu += count_diff(logit, TV_LOGITS + n * MODEL_N_CLASS, MODEL_N_CLASS) != 0;
    }
    fails += (bad_dpu != 0) + (bad_cpu != 0);

    // xil_printf는 실수 출력이 안 되므로 printf 사용
    printf("DPU : %d/%d bit-exact, accuracy %d/%d, %.1f us/image (compute cycles %lu, MACs %lu, layers %lu)\r\n",
           TV_N - bad_dpu, TV_N, ok_dpu, TV_N, t_dpu / TV_N,
           (unsigned long)perf.word_0, (unsigned long)perf.word_1, (unsigned long)perf.word_2);
    printf("ARM : %d/%d bit-exact, %.1f us/image\r\n", TV_N - bad_cpu, TV_N, t_cpu / TV_N);
    printf("speedup x%.1f\r\n", t_cpu / t_dpu);
    xil_printf(fails ? "*** TEST FAILED ***\r\n" : "*** TEST PASSED ***\r\n");
    return 0;
}
