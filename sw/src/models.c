// 모델 목록
#include "dpu_net.h"

#include "../../generated/mnist_model.h"
#include "../../generated/mnist_testvec.h"
#include "../../generated/cifar10_model.h"
#include "../../generated/cifar10_testvec.h"

#define MODEL_ENTRY(X, NAME) {                                              \
    NAME, X##_N_LAYERS, X##_IN_C, X##_IN_H, X##_IN_W, X##_N_CLASS,          \
    X##_LAYERS, X##_WEIGHTS, X##_W_WORDS, X##_PARAMS, X##_P_WORDS,          \
    X##_TV_N, X##_TV_IMAGES, X##_TV_LABELS, X##_TV_LOGITS, X##_TV_LAYER_OUT \
}

static const dpu_model_t mnist = MODEL_ENTRY(MNIST, "MNIST");
static const dpu_model_t cifar = MODEL_ENTRY(CIFAR, "CIFAR-10");

const dpu_model_t *const DPU_MODELS[] = {&mnist, &cifar};
const int DPU_N_MODELS = sizeof(DPU_MODELS) / sizeof(DPU_MODELS[0]);
