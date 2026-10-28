"""CNN 학습, INT8 양자화, C 헤더 생성 (MNIST, CIFAR-10)

    python model/dpu_model.py --model mnist
    python model/dpu_model.py --model cifar10 --epochs 20
    python model/dpu_model.py --model mnist --skip-train  # 저장된 가중치로 헤더만

양자화
  - 활성화: INT8 대칭, 레이어마다 스케일 1개
  - 가중치: INT8 대칭, 출력 채널별 스케일 (BN은 conv에 합침)
  - bias  : INT32, 스케일 s_in * s_w
  - 재양자화: sat8(round_away(acc * mult / 2^shift))
"""
import argparse
import os
import time

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
C, D, P = LAYER_CONV3X3, LAYER_DW3X3, LAYER_PW1X1

IN_SCALE = 1.0 / 127.0     # 입력 스케일

# DPU 메모리 배치 (dpu_types.h와 동일)
MAX_CH = 64
WORD_BYTES = 8
OC_SLOT_BYTES = MAX_CH * 9  # 출력 채널당 576바이트

# 레이어: (종류, in_ch, out_ch, stride, relu, pool)
MODELS = {
    "mnist": dict(
        prefix="MNIST", shape=(1, 28, 28), bn=False, epochs=4, lr=2e-3, n_tv=100,
        signed_input=False,
        layers=[
            (C, 1, 16, 1, 1, 1),    # 28×28 → 14×14
            (C, 16, 32, 1, 1, 1),   # 14×14 → 7×7
            (C, 32, 64, 2, 1, 0),   # 7×7 → 4×4
            (C, 64, 64, 1, 1, 1),   # 4×4 → 2×2
            (C, 64, 64, 1, 1, 1),   # 2×2 → 1×1
            (P, 64, 10, 1, 0, 0),   # FC
        ]),
    "cifar10": dict(
        prefix="CIFAR", shape=(3, 32, 32), bn=True, epochs=20, lr=4e-3, n_tv=200,
        signed_input=True,
        layers=[
            (C, 3, 32, 1, 1, 0),    # 32×32
            (C, 32, 32, 1, 1, 1),   # 32×32 → 16×16
            (D, 32, 32, 1, 1, 0),   # depthwise
            (P, 32, 64, 1, 1, 0),   # pointwise
            (C, 64, 64, 1, 1, 1),   # 16×16 → 8×8
            (D, 64, 64, 1, 1, 0),
            (P, 64, 64, 1, 1, 0),
            (C, 64, 64, 1, 1, 1),   # 8×8 → 4×4
            (C, 64, 64, 2, 1, 0),   # 4×4 → 2×2
            (C, 64, 64, 1, 1, 1),   # 2×2 → 1×1
            (P, 64, 10, 1, 0, 0),   # FC
        ]),
}


class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = cfg["layers"]
        self.convs = nn.ModuleList()
        for t, ic, oc, s, _, _ in self.layers:
            k = 1 if t == P else 3
            self.convs.append(nn.Conv2d(ic, oc, k, stride=s, padding=k // 2,
                                        groups=ic if t == D else 1))
        if cfg["bn"]:
            # FC 제외 BN
            self.bns = nn.ModuleList(nn.BatchNorm2d(L[2]) for L in self.layers[:-1])
        else:
            self.bns = None

    def forward(self, x, collect=None):
        for i, (conv, (_, _, _, _, relu, pool)) in enumerate(zip(self.convs, self.layers)):
            x = conv(x)
            if self.bns is not None and i < len(self.bns):
                x = self.bns[i](x)
            if relu:
                x = F.relu(x)
            if pool:
                x = F.max_pool2d(x, 2)
            if collect is not None:
                collect.append(x)
        return x.flatten(1)


def load_dataset(name, train, augment=False):
    tf = [transforms.ToTensor()]
    if augment and name == "cifar10":
        tf = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] + tf
    tf = transforms.Compose(tf)
    if name == "mnist":
        return datasets.MNIST(DATA_DIR, train=train, download=True, transform=tf)
    return datasets.CIFAR10(DATA_DIR, train=train, download=True, transform=tf)


def to_real(x, cfg):
    """[0, 1] 입력 변환 (CIFAR는 [-1, 1])"""
    return x * 2 - 1 if cfg["signed_input"] else x


def quant_input(x, cfg):
    """입력 INT8 양자화"""
    return torch.clamp(torch.round(to_real(x, cfg) / IN_SCALE), -128, 127)


def model_in(x, cfg):
    """INT8로 양자화한 입력을 실수로 복원"""
    return quant_input(x, cfg) * IN_SCALE


def train(name, cfg, epochs):
    torch.manual_seed(0)
    tr = torch.utils.data.DataLoader(load_dataset(name, True, augment=True), batch_size=128,
                                     shuffle=True, num_workers=0)
    te = torch.utils.data.DataLoader(load_dataset(name, False), batch_size=1000)
    net = Net(cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=5e-4 if cfg["bn"] else 0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=epochs * len(tr)) \
        if cfg["bn"] else torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(tr))
    for ep in range(epochs):
        net.train()
        t0 = time.time()
        for x, y in tr:
            loss = F.cross_entropy(net(model_in(x, cfg)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        print(f"epoch {ep + 1}: float acc {evaluate_float(net, te, cfg):.4f} ({time.time() - t0:.0f}s)",
              flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(OUT_DIR, f"{name}_float.pt"))
    return net


@torch.no_grad()
def evaluate_float(net, loader, cfg):
    net.eval()
    ok = n = 0
    for x, y in loader:
        ok += (net(model_in(x, cfg)).argmax(1) == y).sum().item()
        n += len(y)
    return ok / n


def requant_params(m):
    """실수 배율을 (mult, shift)로 변환"""
    shift = 0
    while shift < 62 and m * (1 << (shift + 1)) < (1 << 31):
        shift += 1
    mult = int(round(m * (1 << shift)))
    return min(mult, (1 << 31) - 1), shift


def requant(acc, mult, shift, relu):
    """재양자화 (HLS requant와 동일)"""
    prod = acc * mult.reshape(1, -1, 1, 1)
    half = np.where(shift == 0, 0, np.left_shift(np.int64(1), np.maximum(shift - 1, 0)))
    q = (np.abs(prod) + half.reshape(1, -1, 1, 1)) >> shift.reshape(1, -1, 1, 1)
    v = np.clip(np.sign(prod) * q, -128, 127)
    if relu:
        v = np.maximum(v, 0)
    return v.astype(np.int64)


def int_layer(x, q, L):
    """정수 레이어 연산"""
    t, ic, _, s, relu, pool = L
    k = 1 if t == P else 3
    # float64 conv로 정수 결과 계산 (누산값 < 2^53)
    acc = F.conv2d(torch.from_numpy(x).double(), torch.from_numpy(q["w"]).double(),
                   torch.from_numpy(q["b"]).double(), stride=s, padding=k // 2,
                   groups=ic if t == D else 1)
    v = requant(acc.round().long().numpy(), q["mult"], q["shift"], relu)
    if pool:
        v = F.max_pool2d(torch.from_numpy(v).double(), 2).long().numpy()
    return v


def int_forward(x, qlayers, layers, keep=False):
    outs = []
    for q, L in zip(qlayers, layers):
        x = int_layer(x, q, L)
        outs.append(x)
    return outs if keep else x


def folded(net, i):
    """BN을 합친 weight, bias"""
    conv = net.convs[i]
    w = conv.weight.detach().double()
    b = conv.bias.detach().double()
    if net.bns is not None and i < len(net.bns):
        bn = net.bns[i]
        g = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
        w = w * g.reshape(-1, 1, 1, 1)
        b = (b - bn.running_mean.double()) * g + bn.bias.detach().double()
    return w.numpy(), b.numpy()


@torch.no_grad()
def quantize(name, cfg, net):
    """양자화 (학습 이미지 2000장, 99.99% 분위수로 활성화 스케일 보정)"""
    net.eval()
    ds = load_dataset(name, True)
    xs = torch.stack([ds[i][0] for i in range(2000)])
    outs = []
    net(model_in(xs, cfg), collect=outs)
    amax = [torch.quantile(o.abs().flatten()[::7][:16_000_000], 0.9999).item() for o in outs]

    qlayers = []
    s_in = IN_SCALE
    for i, L in enumerate(cfg["layers"]):
        w, b = folded(net, i)
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


def evaluate_int(name, cfg, qlayers):
    ds = load_dataset(name, False)
    xs = quant_input(torch.stack([ds[i][0] for i in range(len(ds))]), cfg).long().numpy()
    ys = np.array([ds[i][1] for i in range(len(ds))])
    ok = 0
    for i in range(0, len(xs), 500):
        logit = int_forward(xs[i:i + 500], qlayers, cfg["layers"]).reshape(-1, 10)
        ok += (logit.argmax(1) == ys[i:i + 500]).sum()
    return ok / len(xs), xs, ys


# C 헤더 생성
def c_array(ctype, name, vals, per_line=16):
    vals = [int(v) for v in vals]
    lines = []
    for i in range(0, len(vals), per_line):
        lines.append("    " + ", ".join(str(v) for v in vals[i:i + per_line]) + ",")
    return f"static const {ctype} {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"


def pack_weights(q, L):
    """가중치 배치 (출력 채널당 576바이트)"""
    w = q["w"]
    buf = np.zeros((w.shape[0], OC_SLOT_BYTES), dtype=np.int8)
    for oc in range(w.shape[0]):
        flat = w[oc, :, 0, 0] if L[0] == P else w[oc].reshape(-1)
        buf[oc, :len(flat)] = flat
    return buf.reshape(-1)


def layer_shapes(cfg):
    c, h, w = cfg["shape"]
    shapes = []
    for t, ic, oc, s, relu, pool in cfg["layers"]:
        assert ic == c and (t != D or ic == oc) and ic <= MAX_CH and oc <= MAX_CH
        oh, ow = (h - 1) // s + 1, (w - 1) // s + 1
        shapes.append((h, w, ic, oc))
        c, h, w = oc, (oh // 2 if pool else oh), (ow // 2 if pool else ow)
    assert (c, h, w) == (10, 1, 1)
    return shapes


def export(name, cfg, qlayers, xs, ys):
    os.makedirs(GEN_DIR, exist_ok=True)
    X = cfg["prefix"]
    layers = cfg["layers"]
    n = len(layers)
    w_blobs = [pack_weights(q, L) for q, L in zip(qlayers, layers)]
    w_off = np.cumsum([0] + [len(b) // WORD_BYTES for b in w_blobs])[:-1]
    params = []
    for q in qlayers:
        p = np.zeros(3 * MAX_CH, dtype=np.int64)
        oc = len(q["b"])
        p[0:oc] = q["b"]
        p[MAX_CH:MAX_CH + oc] = q["mult"]
        p[2 * MAX_CH:2 * MAX_CH + oc] = q["shift"]
        params.append(p)
    shapes = layer_shapes(cfg)
    in_c, in_h, in_w = cfg["shape"]

    with open(os.path.join(GEN_DIR, f"{name}_model.h"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"#ifndef {X}_MODEL_H\n#define {X}_MODEL_H\n\n#include <stdint.h>\n\n")
        f.write(f"#define {X}_N_LAYERS   {n}\n")
        f.write(f"#define {X}_IN_C       {in_c}\n#define {X}_IN_H       {in_h}\n#define {X}_IN_W       {in_w}\n")
        f.write(f"#define {X}_N_CLASS    10\n")
        f.write(f"#define {X}_W_WORDS    {sum(len(b) for b in w_blobs) // WORD_BYTES}\n")
        f.write(f"#define {X}_P_WORDS    {n * 3 * MAX_CH}\n\n")
        f.write("// in_h, in_w, in_ch, out_ch, type, stride, relu, pool, 가중치 오프셋\n")
        f.write(f"static const uint16_t {X}_LAYERS[{X}_N_LAYERS][9] = {{\n")
        for (ih, iw, ic, oc), L, wo in zip(shapes, layers, w_off):
            t, _, _, s, relu, pool = L
            f.write(f"    {{{ih}, {iw}, {ic}, {oc}, {t}, {s}, {relu}, {pool}, {wo}}},\n")
        f.write("};\n\n")
        f.write("// 가중치 (출력 채널당 576바이트)\n")
        f.write("__attribute__((aligned(64)))\n")
        f.write(c_array("int8_t", f"{X}_WEIGHTS", np.concatenate(w_blobs), 32))
        f.write("\n// 파라미터 (레이어당 bias[64], mult[64], shift[64])\n")
        f.write("__attribute__((aligned(64)))\n")
        f.write(c_array("int32_t", f"{X}_PARAMS", np.concatenate(params), 16))
        f.write("\n#endif\n")

    # 테스트 벡터
    n_tv = cfg["n_tv"]
    xt = xs[:n_tv]
    logits = int_forward(xt, qlayers, layers).reshape(-1, 10)
    layer_outs = int_forward(xt[:1], qlayers, layers, keep=True)
    with open(os.path.join(GEN_DIR, f"{name}_testvec.h"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"#ifndef {X}_TESTVEC_H\n#define {X}_TESTVEC_H\n\n#include <stdint.h>\n\n")
        f.write(f"#define {X}_TV_N {n_tv}\n\n")
        f.write("// 이미지 [N][C][H][W]\n")
        f.write(c_array("int8_t", f"{X}_TV_IMAGES", xt.reshape(-1), 32))
        f.write(c_array("uint8_t", f"{X}_TV_LABELS", ys[:n_tv], 25))
        f.write(c_array("int8_t", f"{X}_TV_LOGITS", logits.reshape(-1), 10))
        f.write("\n// 첫 이미지 레이어별 출력\n")
        for i, o in enumerate(layer_outs):
            f.write(c_array("int8_t", f"{X}_TV_L{i}_OUT", o[0].reshape(-1), 16))
        f.write(f"static const int8_t *const {X}_TV_LAYER_OUT[{n}] = {{%s}};\n"
                % ", ".join(f"{X}_TV_L{i}_OUT" for i in range(n)))
        f.write("\n#endif\n")
    print(f"headers: {GEN_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="mnist")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--epochs", type=int)
    a = ap.parse_args()
    name, cfg = a.model, MODELS[a.model]

    if a.skip_train:
        net = Net(cfg)
        net.load_state_dict(torch.load(os.path.join(OUT_DIR, f"{name}_float.pt")))
    else:
        net = train(name, cfg, a.epochs or cfg["epochs"])
    te = torch.utils.data.DataLoader(load_dataset(name, False), batch_size=1000)
    print(f"float acc {evaluate_float(net, te, cfg):.4f}")
    qlayers = quantize(name, cfg, net)
    acc, xs, ys = evaluate_int(name, cfg, qlayers)
    print(f"int8  acc {acc:.4f}")
    for i, q in enumerate(qlayers):
        print(f"L{i}: shift {q['shift'].min()}~{q['shift'].max()}, "
              f"mult {q['mult'].min()}~{q['mult'].max()}, s_out {q['s_out']:.5f}")
    export(name, cfg, qlayers, xs, ys)


if __name__ == "__main__":
    main()
