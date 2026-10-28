// 보드 테스트 (베어메탈)
//  1) 레이어별 DPU/ARM 비교, 시간
//  2) 전체 네트워크 DPU 비교, 정확도, 시간
//  3) ARM 단독 시간, 가속 배율
#include <stdio.h>
#include <string.h>

#include "xdpu_top.h"
#include "xil_cache.h"
#include "xil_io.h"
#include "xparameters.h"
#include "xiltimer.h"
#include "xil_printf.h"
#include "xuartps_hw.h"

#include "dpu_net.h"

#ifndef XPAR_DPU_0_BASEADDR
#define XPAR_DPU_0_BASEADDR 0x40000000
#endif

static XDpu_top dpu;

// DPU 공유 버퍼 (쓰기 후 flush, 읽기 전 invalidate)
static int8_t   fmem[DPU_FMEM_BYTES]                       __attribute__((aligned(64)));
static int8_t   wmem[DPU_MAX_W_WORDS * 8]                  __attribute__((aligned(64)));
static int32_t  pmem[DPU_MAX_LAYERS * DPU_P_LAYER_WORDS]   __attribute__((aligned(64)));
static uint32_t desc[DPU_MAX_LAYERS * DPU_DESC_WORDS]      __attribute__((aligned(64)));
static int8_t   dense[DPU_FMAP_BYTES];
static int8_t   cpu_out[DPU_MAX_LAYERS][DPU_FMAP_BYTES / 4];
static int8_t   cpu_scratch[DPU_FMAP_BYTES];

// COUNTS_PER_SECOND는 괄호 없는 매크로라 괄호 필요
static double elapsed_us(XTime a, XTime b) { return (double)(b - a) * 1e6 / (double)(COUNTS_PER_SECOND); }

// 타이머 점검: UART 115200 baud로 5760자(500ms) 전송 시간 측정
static void timer_check(void) {
    const UINTPTR uart = XPAR_XUARTPS_0_BASEADDR;
    fflush(stdout);
    while (!(Xil_In32(uart + XUARTPS_SR_OFFSET) & XUARTPS_SR_TXEMPTY)) ;
    XTime t0, t1;
    XTime_GetTime(&t0);
    for (int line = 0; line < 60; line++) {          // 줄당 96자
        for (int i = 0; i < 94; i++) outbyte('.');
        outbyte('\r');
        outbyte('\n');
    }
    while (!(Xil_In32(uart + XUARTPS_SR_OFFSET) & XUARTPS_SR_TXEMPTY)) ;
    XTime_GetTime(&t1);
    const double ms = elapsed_us(t0, t1) / 1e3;
    printf("timer check: 5760 chars @115200 baud = 500.0 ms expected, %.1f ms measured (x%.3f)\r\n",
           ms, 500.0 / ms);
}

static void dpu_run(const uint32_t *d, int n_layers) {
    XDpu_top_Set_desc(&dpu, (u64)(UINTPTR)d);
    XDpu_top_Set_fmem(&dpu, (u64)(UINTPTR)fmem);
    XDpu_top_Set_wmem(&dpu, (u64)(UINTPTR)wmem);
    XDpu_top_Set_pmem(&dpu, (u64)(UINTPTR)pmem);
    XDpu_top_Set_n_layers(&dpu, n_layers);
    XDpu_top_Start(&dpu);
    while (!XDpu_top_IsDone(&dpu)) ;
}

// 입력 기록
static void put_input(const dpu_model_t *m, const int8_t *img) {
    dpu_net_pack_input(m, img, fmem);
    for (int c = 0; c < m->in_c; c++)
        Xil_DCacheFlushRange((UINTPTR)(fmem + c * DPU_MAX_HW * DPU_MAX_HW), m->in_h * DPU_MAX_HW);
}

// 출력 읽기
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

static int argmax(const int8_t *v, int n) {
    int best = 0;
    for (int k = 1; k < n; k++) if (v[k] > v[best]) best = k;
    return best;
}

static int test_model(const dpu_model_t *m) {
    static const char *const kind[] = {"conv3x3", "dw3x3", "pw1x1"};
    const int img_bytes = m->in_c * m->in_h * m->in_w;
    printf("\r\n=== %s: %d layers, input %dx%dx%d ===\r\n", m->name, m->n_layers, m->in_c, m->in_h, m->in_w);
    if (m->n_layers > DPU_MAX_LAYERS || m->w_words > DPU_MAX_W_WORDS) {
        printf("model too large for this runtime\r\n");
        return 1;
    }

    memcpy(wmem, m->weights, (size_t)m->w_words * 8);
    memcpy(pmem, m->params, (size_t)m->p_words * 4);
    dpu_net_build_desc(m, desc);
    memset(fmem, 0, sizeof fmem);
    Xil_DCacheFlushRange((UINTPTR)wmem, (size_t)m->w_words * 8);
    Xil_DCacheFlushRange((UINTPTR)pmem, (size_t)m->p_words * 4);
    Xil_DCacheFlushRange((UINTPTR)desc, sizeof desc);
    Xil_DCacheFlushRange((UINTPTR)fmem, sizeof fmem);

    int fails = 0;

    // 1) 레이어별
    int8_t *layer_out[DPU_MAX_LAYERS];
    for (int l = 0; l < m->n_layers; l++) layer_out[l] = cpu_out[l];
    int8_t logit[16];
    cpu_net_forward(m, m->tv_images, logit, layer_out);

    printf("layer type     s  output      MACs      cycles  DPU us   ARM us  speedup  check\r\n");
    put_input(m, m->tv_images);
    double sum_dpu = 0, sum_cpu = 0;
    uint32_t sum_cyc = 0, sum_mac = 0;
    for (int l = 0; l < m->n_layers; l++) {
        XTime t0, t1, t2;
        XTime_GetTime(&t0);
        dpu_run(desc + l * DPU_DESC_WORDS, 1);
        XTime_GetTime(&t1);
        cpu_net_layer(m, l, l == 0 ? m->tv_images : layer_out[l - 1], cpu_scratch);
        XTime_GetTime(&t2);
        const XDpu_top_Perf p = XDpu_top_Get_perf(&dpu);
        int c, h, w;
        dpu_net_out_shape(m, l, &c, &h, &w);
        get_output(dpu_net_out_slot(l), c, h, w, dense);
        const int bad = count_diff(dense, m->tv_layer_out[l], c * h * w) + count_diff(dense, layer_out[l], c * h * w);
        const double td = elapsed_us(t0, t1), tc = elapsed_us(t1, t2);
        const uint32_t macs = dpu_net_layer_macs(m, l);
        printf("%5d %-7s  %d  %2dx%2dx%2d %9lu %9lu %8.1f %8.1f %7.1fx  %s\r\n", l, kind[m->layers[l][4]],
               m->layers[l][5], c, h, w, (unsigned long)macs, (unsigned long)p.word_0, td, tc, tc / td,
               bad ? "FAIL" : "ok");
        sum_dpu += td;
        sum_cpu += tc;
        sum_cyc += p.word_0;
        sum_mac += macs;
        fails += bad != 0;
    }
    printf("total                         %9lu %9lu %8.1f %8.1f %7.1fx\r\n",
           (unsigned long)sum_mac, (unsigned long)sum_cyc, sum_dpu, sum_cpu, sum_cpu / sum_dpu);

    // 2) 전체 네트워크
    int bad_dpu = 0, ok_dpu = 0;
    double t_dpu = 0;
    XDpu_top_Perf perf = {0, 0, 0};
    for (int n = 0; n < m->tv_n; n++) {
        const int8_t *img = m->tv_images + n * img_bytes;
        XTime t0, t1;
        XTime_GetTime(&t0);
        put_input(m, img);
        dpu_run(desc, m->n_layers);
        get_output(dpu_net_out_slot(m->n_layers - 1), m->n_class, 1, 1, logit);
        XTime_GetTime(&t1);
        t_dpu += elapsed_us(t0, t1);
        perf = XDpu_top_Get_perf(&dpu);
        bad_dpu += count_diff(logit, m->tv_logits + n * m->n_class, m->n_class) != 0;
        ok_dpu  += argmax(logit, m->n_class) == m->tv_labels[n];
    }

    // 3) ARM 단독
    int bad_cpu = 0;
    double t_cpu = 0;
    for (int n = 0; n < m->tv_n; n++) {
        XTime t0, t1;
        XTime_GetTime(&t0);
        cpu_net_forward(m, m->tv_images + n * img_bytes, logit, NULL);
        XTime_GetTime(&t1);
        t_cpu += elapsed_us(t0, t1);
        bad_cpu += count_diff(logit, m->tv_logits + n * m->n_class, m->n_class) != 0;
    }
    fails += (bad_dpu != 0) + (bad_cpu != 0);

    const double us_dpu = t_dpu / m->tv_n, us_cpu = t_cpu / m->tv_n;
    printf("DPU : %d/%d bit-exact, accuracy %d/%d, %.1f us/image (%.0f fps), compute cycles %lu, MACs %lu\r\n",
           m->tv_n - bad_dpu, m->tv_n, ok_dpu, m->tv_n, us_dpu, 1e6 / us_dpu,
           (unsigned long)perf.word_0, (unsigned long)perf.word_1);
    printf("ARM : %d/%d bit-exact, %.1f us/image (%.0f fps)\r\n", m->tv_n - bad_cpu, m->tv_n, us_cpu, 1e6 / us_cpu);
    printf("speedup x%.1f, DPU %.2f GOPS (2 ops/MAC)\r\n", us_cpu / us_dpu, 2.0 * perf.word_1 / us_dpu / 1e3);
    return fails;
}

int main(void) {
    Xil_Out32(XPAR_GLOBAL_TMR_BASEADDR + 0x08U, 0x1U);   // 글로벌 타이머 시작
    printf("\r\n##### DPU multi-model test #####\r\n");
    timer_check();

    if (XDpu_top_Initialize(&dpu, XPAR_DPU_0_BASEADDR) != XST_SUCCESS) {
        printf("DPU init failed\r\n");
        return -1;
    }

    int fails = 0;
    for (int i = 0; i < DPU_N_MODELS; i++) fails += test_model(DPU_MODELS[i]);
    printf(fails ? "\r\n*** TEST FAILED ***\r\n" : "\r\n*** TEST PASSED ***\r\n");
    return 0;
}
