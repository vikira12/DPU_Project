"""MNIST CNN 학습 → INT8 양자화 → 정수 참조 추론 → C 헤더 생성

    python model/dpu_model.py            # 학습 + 양자화 + 헤더 생성
    python model/dpu_model.py --skip-train  # 저장된 float 가중치로 양자화·헤더만

정수 규약 (HLS dpu_top과 비트 단위로 같아야 함)
  - 활성화: INT8 대칭 (zero point 0), 레이어 출력마다 스케일 하나
  - 가중치: INT8 대칭, 출력 채널별 스케일
  - bias  : INT32, 스케일 = s_in × s_w[oc]
  - 재양자화: sat8( round_away( acc × mult / 2^shift ) ) → [ReLU] → [2×2 maxpool]
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "out")
GEN_DIR = os.path.join(os.path.dirname(ROOT), "generated")

LAYER_CONV3X3, LAYER_DW3X3, LAYER_PW1X1 = 0, 1, 2

# (종류, in_ch, out_ch, stride, relu, pool)
LAYERS = [
    (LAYER_CONV3X3, 1, 16, 1, 1, 1),   # 28×28 → 14×14
    (LAYER_CONV3X3, 16, 32, 1, 1, 1),  # 14×14 → 7×7
    (LAYER_CONV3X3, 32, 64, 2, 1, 0),  # 7×7 → 4×4
    (LAYER_CONV3X3, 64, 64, 1, 1, 1),  # 4×4 → 2×2
    (LAYER_CONV3X3, 64, 64, 1, 1, 1),  # 2×2 → 1×1
    (LAYER_PW1X1, 64, 10, 1, 0, 0),    # FC (1×1 공간에서 pointwise)
]

IN_SCALE = 1.0 / 127.0     # 입력 [0, 1] → INT8 [0, 127]
N_TEST_VEC = 100           # 헤더에 넣을 테스트 이미지 수

# DPU 메모리 배치 (dpu_types.h와 같아야 함)
MAX_CH = 64
WORD_BYTES = 8
OC_SLOT_BYTES = MAX_CH * 9  # 출력 채널 하나의 가중치 칸 576바이트


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList()
        for t, ic, oc, s, _, _ in LAYERS:
            k = 1 if t == LAYER_PW1X1 else 3
            self.convs.append(nn.Conv2d(ic, oc, k, stride=s, padding=k // 2))

    def forward(self, x):
        for conv, (_, _, _, _, relu, pool) in zip(self.convs, LAYERS):
            x = conv(x)
            if relu:
                x = F.relu(x)
            if pool:
                x = F.max_pool2d(x, 2)
        return x.flatten(1)


def load_mnist(train):
    tf = transforms.ToTensor()
    return datasets.MNIST(DATA_DIR, train=train, download=True, transform=tf)


def quant_input(x):
    """[0, 1] float 이미지 → INT8"""
    return torch.clamp(torch.round(x / IN_SCALE), -128, 127)


def train(epochs=4):
    torch.manual_seed(0)
    tr = torch.utils.data.DataLoader(load_mnist(True), batch_size=128, shuffle=True)
    te = torch.utils.data.DataLoader(load_mnist(False), batch_size=1000)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(tr))
    for ep in range(epochs):
        net.train()
        for x, y in tr:
            # 학습 입력도 INT8 격자에 맞춰 둠 (양자화 오차 감소)
            x = quant_input(x) * IN_SCALE
            loss = F.cross_entropy(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        print(f"epoch {ep + 1}: float acc {evaluate_float(net, te):.4f}")
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(OUT_DIR, "mnist_float.pt"))
    return net


@torch.no_grad()
def evaluate_float(net, loader):
    net.eval()
    ok = n = 0
    for x, y in loader:
        ok += (net(quant_input(x) * IN_SCALE).argmax(1) == y).sum().item()
        n += len(y)
    return ok / n


def requant_params(m):
    """실수 배율 m (> 0) → (mult, shift), mult < 2^31"""
    shift = 0
    while shift < 62 and m * (1 << (shift + 1)) < (1 << 31):
        shift += 1
    mult = int(round(m * (1 << shift)))
    return min(mult, (1 << 31) - 1), shift


def requant(acc, mult, shift, relu):
    """HLS requant()과 같은 계산. acc: int64 [N, C, H, W], mult/shift: [C]"""
    prod = acc * mult.reshape(1, -1, 1, 1)
    half = np.where(shift == 0, 0, np.left_shift(np.int64(1), np.maximum(shift - 1, 0)))
    q = (np.abs(prod) + half.reshape(1, -1, 1, 1)) >> shift.reshape(1, -1, 1, 1)
    v = np.clip(np.sign(prod) * q, -128, 127)
    if relu:
        v = np.maximum(v, 0)
    return v.astype(np.int64)


def int_layer(x, q, L):
    """정수 레이어 하나. x: int64 [N, C, H, W] → int64"""
    t, _, _, s, relu, pool = L
    k = 1 if t == LAYER_PW1X1 else 3
    # 누산값 최대 |64 × 9 × 128 × 128| + bias < 2^53 → float64 conv가 정확한 정수 결과
    acc = F.conv2d(torch.from_numpy(x).double(), torch.from_numpy(q["w"]).double(),
                   torch.from_numpy(q["b"]).double(), stride=s, padding=k // 2)
    v = requant(acc.round().long().numpy(), q["mult"], q["shift"], relu)
    if pool:
        v = F.max_pool2d(torch.from_numpy(v).double(), 2).long().numpy()
    return v


def int_forward(x, qlayers, keep=False):
    outs = []
    for q, L in zip(qlayers, LAYERS):
        x = int_layer(x, q, L)
        outs.append(x)
    return outs if keep else x


@torch.no_grad()
def quantize(net):
    """보정: 학습셋 2000장으로 레이어 출력 최대값 → 활성화 스케일"""
    net.eval()
    xs = torch.stack([load_mnist(True)[i][0] for i in range(2000)])
    x = quant_input(xs) * IN_SCALE
    amax = []
    for conv, (_, _, _, _, relu, pool) in zip(net.convs, LAYERS):
        x = conv(x)
        if relu:
            x = F.relu(x)
        if pool:
            x = F.max_pool2d(x, 2)
        amax.append(torch.quantile(x.abs().flatten()[::7], 0.9999).item())

    qlayers = []
    s_in = IN_SCALE
    for i, (conv, L) in enumerate(zip(net.convs, LAYERS)):
        w = conv.weight.detach().numpy().astype(np.float64)
        b = conv.bias.detach().numpy().astype(np.float64)
        s_w = np.maximum(np.abs(w).reshape(w.shape[0], -1).max(1), 1e-12) / 127.0
        s_out = amax[i] / 127.0
        wq = np.clip(np.round(w / s_w.reshape(-1, 1, 1, 1)), -127, 127).astype(np.int64)
        bq = np.round(b / (s_in * s_w)).astype(np.int64)
        mp = [requant_params(s_in * sw / s_out) for sw in s_w]
        qlayers.append(dict(w=wq, b=bq,
                            mult=np.array([m for m, _ in mp], dtype=np.int64),
                            shift=np.array([sh for _, sh in mp], dtype=np.int64),
                            s_out=s_out))
        s_in = s_out
    return qlayers


def evaluate_int(qlayers):
    ds = load_mnist(False)
    xs = quant_input(torch.stack([ds[i][0] for i in range(len(ds))])).long().numpy()
    ys = np.array([ds[i][1] for i in range(len(ds))])
    ok = 0
    for i in range(0, len(xs), 1000):
        logit = int_forward(xs[i:i + 1000], qlayers).reshape(-1, 10)
        ok += (logit.argmax(1) == ys[i:i + 1000]).sum()
    return ok / len(xs), xs, ys


# ----------------------------------------------------------------------
# C 헤더 생성
# ----------------------------------------------------------------------
def c_array(ctype, name, vals, per_line=16):
    vals = [int(v) for v in vals]
    lines = []
    for i in range(0, len(vals), per_line):
        lines.append("    " + ", ".join(str(v) for v in vals[i:i + per_line]) + ",")
    return f"static const {ctype} {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"


def pack_weights(q, L):
    """출력 채널마다 576바이트 칸. CONV3X3 [ic][3][3], PW1X1 [ic]"""
    w = q["w"]
    buf = np.zeros((w.shape[0], OC_SLOT_BYTES), dtype=np.int8)
    for oc in range(w.shape[0]):
        flat = w[oc].reshape(-1) if L[0] == LAYER_CONV3X3 else w[oc, :, 0, 0]
        buf[oc, :len(flat)] = flat
    return buf.reshape(-1)


def export(qlayers, xs, ys):
    os.makedirs(GEN_DIR, exist_ok=True)
    n = len(LAYERS)
    w_blobs = [pack_weights(q, L) for q, L in zip(qlayers, LAYERS)]
    w_off = np.cumsum([0] + [len(b) // WORD_BYTES for b in w_blobs])[:-1]
    params = []
    for q in qlayers:
        p = np.zeros(3 * MAX_CH, dtype=np.int64)
        oc = len(q["b"])
        p[0:oc] = q["b"]
        p[MAX_CH:MAX_CH + oc] = q["mult"]
        p[2 * MAX_CH:2 * MAX_CH + oc] = q["shift"]
        params.append(p)

    h = in_w = 28
    shapes = []
    for t, ic, oc, s, relu, pool in LAYERS:
        oh = (h - 1) // s + 1
        shapes.append((h, in_w, ic, oc))
        h = in_w = oh // 2 if pool else oh

    with open(os.path.join(GEN_DIR, "mnist_model.h"), "w", encoding="utf-8", newline="\n") as f:
        f.write("#ifndef MNIST_MODEL_H\n#define MNIST_MODEL_H\n\n#include <stdint.h>\n\n")
        f.write(f"#define MODEL_N_LAYERS   {n}\n")
        f.write(f"#define MODEL_IN_H       28\n#define MODEL_IN_W       28\n")
        f.write(f"#define MODEL_N_CLASS    10\n")
        f.write(f"#define MODEL_W_WORDS    {sum(len(b) for b in w_blobs) // WORD_BYTES}\n")
        f.write(f"#define MODEL_P_WORDS    {n * 3 * MAX_CH}\n\n")
        f.write("// 레이어별: in_h, in_w, in_ch, out_ch, type, stride, relu, pool, 가중치 시작 워드\n")
        f.write("static const uint16_t MODEL_LAYERS[MODEL_N_LAYERS][9] = {\n")
        for (ih, iw, ic, oc), L, wo in zip(shapes, LAYERS, w_off):
            t, _, _, s, relu, pool = L
            f.write(f"    {{{ih}, {iw}, {ic}, {oc}, {t}, {s}, {relu}, {pool}, {wo}}},\n")
        f.write("};\n\n")
        f.write("// 가중치: 레이어 순서로 이어 붙임, 출력 채널마다 576바이트 칸\n")
        f.write("__attribute__((aligned(64)))\n")
        f.write(c_array("int8_t", "MODEL_WEIGHTS", np.concatenate(w_blobs), 32))
        f.write("\n// 파라미터: 레이어마다 bias[64], mult[64], shift[64]\n")
        f.write("__attribute__((aligned(64)))\n")
        f.write(c_array("int32_t", "MODEL_PARAMS", np.concatenate(params), 16))
        f.write("\n#endif\n")

    # 테스트 벡터: 이미지, 정답, 정수 추론 로짓, 첫 이미지의 레이어별 출력
    xt = xs[:N_TEST_VEC]
    logits = int_forward(xt, qlayers).reshape(-1, 10)
    layer_outs = int_forward(xt[:1], qlayers, keep=True)
    with open(os.path.join(GEN_DIR, "mnist_testvec.h"), "w", encoding="utf-8", newline="\n") as f:
        f.write("#ifndef MNIST_TESTVEC_H\n#define MNIST_TESTVEC_H\n\n#include <stdint.h>\n\n")
        f.write(f"#define TV_N {N_TEST_VEC}\n\n")
        f.write(c_array("int8_t", "TV_IMAGES", xt.reshape(-1), 28))
        f.write(c_array("uint8_t", "TV_LABELS", ys[:N_TEST_VEC], 25))
        f.write(c_array("int8_t", "TV_LOGITS", logits.reshape(-1), 10))
        f.write("\n// 첫 이미지의 레이어별 출력 [C][H][W] (조밀 배치)\n")
        for i, o in enumerate(layer_outs):
            f.write(c_array("int8_t", f"TV_L{i}_OUT", o[0].reshape(-1), 16))
        f.write("static const int8_t *const TV_LAYER_OUT[%d] = {%s};\n"
                % (n, ", ".join(f"TV_L{i}_OUT" for i in range(n))))
        f.write("\n#endif\n")
    print(f"headers → {GEN_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--epochs", type=int, default=4)
    a = ap.parse_args()

    if a.skip_train:
        net = Net()
        net.load_state_dict(torch.load(os.path.join(OUT_DIR, "mnist_float.pt")))
    else:
        net = train(a.epochs)
    te = torch.utils.data.DataLoader(load_mnist(False), batch_size=1000)
    print(f"float acc {evaluate_float(net, te):.4f}")
    qlayers = quantize(net)
    acc, xs, ys = evaluate_int(qlayers)
    print(f"int8  acc {acc:.4f}")
    for i, q in enumerate(qlayers):
        print(f"L{i}: shift {q['shift'].min()}~{q['shift'].max()}, "
              f"mult {q['mult'].min()}~{q['mult'].max()}, s_out {q['s_out']:.5f}")
    export(qlayers, xs, ys)


if __name__ == "__main__":
    main()
