#!/usr/bin/env python3
"""
Accuracy study: what do INT8 / INT4x2 / 2:4 sparsity do to a real CNN?

Project : Mixed-Precision Sparsity-Aware Systolic Array (dissertation)
Stage   : Month 4 / Week 14 - accuracy characterisation (objective O5)

Every conv / linear layer of a torchvision model is replaced by a layer that
does exactly what the array would: weights pruned 2:4 (optional) and quantised,
activations quantised, integer-valued products accumulated. The arithmetic is
"fake-quant" (integers carried in float32 and rescaled); quant_core.py proves
on random data that this equals the integer datapath and golden_model.py.

Configurations (all compared with the FP32 model):
    int8      INT8 weights + activations              (pe_mixed, MODE_INT8)
    int4      INT4 weights + activations              (pe_mixed, MODE_INT4X2)
    fp32_24   2:4 pruning only, FP32 arithmetic       (isolates pruning loss)
    int8_24   2:4 pruning + INT8
    int4_24   2:4 pruning + INT4

No fine-tuning is done. 2:4 pruning and INT4 without retraining can lose a lot
of accuracy - that is a result to report, not a bug.

Two kinds of result are written:
    layerwise.csv   per-layer relative MAE, each layer fed the FP32 input
                    (error of the layer in isolation)
    end_to_end.csv  whole network: top-1 / top-5 (if labels), top-1 agreement
                    with FP32, logit relative MAE

Usage:
    # 0. no-torch check of the arithmetic against golden_model.py
    python3 golden/quant_core.py

    # 1. plumbing smoke test: no downloads, no dataset (random weights/inputs;
    #    the NUMBERS are meaningless, only checks the script runs end to end)
    python3 golden/accuracy_study.py --weights random --synthetic \\
        --num-images 16 --calib-images 16 --layer-images 8 --out /tmp/acc_smoke

    # 2. real run (pretrained weights; ImageNet val folder in class sub-folders
    #    gives top-1; a flat folder of images gives agreement/logit error only)
    python3 golden/accuracy_study.py --model resnet18 --data-dir /path/to/imagenet/val \\
        --num-images 1000 --out results/accuracy_resnet18
    python3 golden/accuracy_study.py --model mobilenet_v2 --data-dir ... \\
        --out results/accuracy_mobilenet_v2

    # no internet on the compute machine? point at a saved state_dict:
    #   --checkpoint resnet18.pth
"""

import argparse
import copy
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_core as qc  # noqa: E402

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit("ERROR: torch/torchvision are required. (quant_core.py's selftest needs only numpy.)")

# name -> (bits or None for FP32, 2:4 pruning)
CONFIGS = {
    "int8": (8, False),
    "int4": (4, False),
    "fp32_24": (None, True),
    "int8_24": (8, True),
    "int4_24": (4, True),
}


# ---------------------------------------------------------------------------
# The layer that behaves like the array
# ---------------------------------------------------------------------------

class QLayer(nn.Module):
    """Conv2d / Linear with pruned + quantised weights and quantised input."""

    def __init__(self, layer, bits, sparse, act_clip):
        super().__init__()
        w = layer.weight.detach().cpu().double().numpy()
        if sparse:
            # K = flattened (Cin/groups, kh, kw): the contraction axis the array streams
            w = qc.prune_2of4_rows(w.reshape(w.shape[0], -1)).reshape(w.shape)
        if bits is not None:
            q, s = qc.quantize_weights(w, bits)
            w = q * s
        self.register_buffer("weight", torch.from_numpy(w).float())
        if layer.bias is not None:
            self.register_buffer("bias", layer.bias.detach().clone().float())
        else:
            self.bias = None
        self.bits = bits
        self.act_clip = max(float(act_clip), 1e-12)
        self.is_conv = isinstance(layer, nn.Conv2d)
        if self.is_conv:
            self.conv_args = (layer.stride, layer.padding, layer.dilation, layer.groups)

    def forward(self, x):
        if self.bits is not None:
            qm = qc.qmax(self.bits)
            scale = self.act_clip / qm
            x = torch.clamp(torch.round(x / scale), -qm - 1, qm) * scale
        if self.is_conv:
            return F.conv2d(x, self.weight, self.bias, *self.conv_args)
        return F.linear(x, self.weight, self.bias)


# ---------------------------------------------------------------------------
# Model / data
# ---------------------------------------------------------------------------

def build_model(args):
    import torchvision.models as tvm
    builder = getattr(tvm, args.model)
    pretrained = args.weights == "pretrained" and not args.checkpoint
    try:
        model = builder(weights="IMAGENET1K_V1" if pretrained else None)
    except TypeError:                       # older torchvision
        model = builder(pretrained=pretrained)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    if args.fold_bn:
        try:
            from torch.fx.experimental.optimization import fuse
            model = fuse(model)             # fold BatchNorm into the conv, as deployment does
        except Exception as exc:            # keep going unfused, but say so
            print("WARNING: BatchNorm folding failed (%s); quantising unfused model" % exc)
    return model.eval()


def quant_layers(model):
    """(name, module) for every layer the array would execute, in forward order."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def build_batches(args):
    """Return (calib_batches, eval_batches); each batch is (images, labels), label -1 = unknown."""
    g = torch.Generator().manual_seed(args.seed)
    n_cal, n_eval = args.calib_images, args.num_images
    if args.synthetic:
        x = torch.randn(n_cal + n_eval, 3, 224, 224, generator=g)
        y = torch.full((n_cal + n_eval,), -1, dtype=torch.long)
        ds = torch.utils.data.TensorDataset(x, y)
    else:
        import torchvision.transforms as T
        import torchvision.datasets as D
        tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        try:
            ds = D.ImageFolder(args.data_dir, transform=tf)     # class sub-folders -> labels
        except FileNotFoundError:
            from PIL import Image
            exts = (".jpg", ".jpeg", ".png", ".JPEG")
            files = sorted(os.path.join(r, f) for r, _, fs in os.walk(args.data_dir)
                           for f in fs if f.endswith(exts))
            if not files:
                sys.exit("ERROR: no images found under %s" % args.data_dir)

            class Flat(torch.utils.data.Dataset):
                def __len__(self):
                    return len(files)

                def __getitem__(self, i):
                    return tf(Image.open(files[i]).convert("RGB")), -1
            ds = Flat()
    perm = torch.randperm(len(ds), generator=g).tolist()
    need = n_cal + n_eval
    if len(perm) < need:
        print("WARNING: only %d images available, wanted %d" % (len(perm), need))
        n_cal = min(n_cal, len(perm) // 2)
    cal_idx, eval_idx = perm[:n_cal], perm[n_cal:n_cal + n_eval]
    mk = lambda idx: torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx),
                                                 batch_size=args.batch, shuffle=False)
    return mk(cal_idx), mk(eval_idx)


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate(model, loader, layers, pct, device):
    """Per-layer activation clip = percentile(|input|), maximised over batches."""
    clips = {n: 0.0 for n, _ in layers}
    hooks = []

    def make_hook(name):
        def hook(_m, inp):
            a = inp[0].detach().abs().flatten().float()
            if a.numel() > 2_000_000:
                a = a[torch.randint(0, a.numel(), (2_000_000,), device=a.device)]
            v = a.max() if pct >= 100 else torch.quantile(a, pct / 100.0)
            clips[name] = max(clips[name], float(v))
        return hook

    for n, m in layers:
        hooks.append(m.register_forward_pre_hook(make_hook(n)))
    for x, _ in loader:
        model(x.to(device))
    for h in hooks:
        h.remove()
    return clips


def bits_for(cfg, name, layers, args):
    bits, _ = CONFIGS[cfg]
    if bits == 4 and args.first_last_int8 and name in (layers[0][0], layers[-1][0]):
        return 8                            # common practice: keep first/last layer at INT8
    return bits


def make_qlayer(cfg, name, mod, clips, layers, args, device):
    return QLayer(mod, bits_for(cfg, name, layers, args), CONFIGS[cfg][1], clips[name]).to(device)


@torch.no_grad()
def layerwise(model, loader, layers, clips, args, device):
    """Error of each layer in isolation: quantised layer vs FP32 layer, same FP32 input."""
    q = {c: {n: make_qlayer(c, n, m, clips, layers, args, device) for n, m in layers}
         for c in CONFIGS}
    acc = {n: {c: [0.0, 0.0] for c in CONFIGS} for n, _ in layers}   # [sum|err|, sum|ref|]
    hooks = []

    def make_hook(name):
        def hook(_m, inp, out):
            for c in CONFIGS:
                yq = q[c][name](inp[0])
                acc[name][c][0] += float((yq - out).abs().sum())
                acc[name][c][1] += float(out.abs().sum())
        return hook

    for n, m in layers:
        hooks.append(m.register_forward_hook(make_hook(n)))
    for x, _ in loader:
        model(x.to(device))
    for h in hooks:
        h.remove()
    rows = []
    for n, m in layers:
        w = m.weight
        row = {"layer": n, "type": type(m).__name__, "K": int(w[0].numel()),
               "Cout": int(w.shape[0]),
               "groups": int(getattr(m, "groups", 1))}
        for c in CONFIGS:
            row[c] = acc[n][c][0] / max(acc[n][c][1], 1e-30)
        rows.append(row)
    return rows


@torch.no_grad()
def run_logits(model, loader, device):
    outs, labels = [], []
    for x, y in loader:
        outs.append(model(x.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(outs), torch.cat(labels)


def end_to_end(model, loader, layers, clips, args, device):
    ref, labels = run_logits(model, loader, device)
    have_labels = bool((labels >= 0).all())
    rows = []

    def summarise(name, logits):
        top1 = logits.argmax(1)
        row = {"config": name, "n_images": int(len(labels)),
               "top1_agree_fp32": float((top1 == ref.argmax(1)).float().mean()),
               "logit_rel_mae": float((logits - ref).abs().sum() / ref.abs().sum().clamp_min(1e-30))}
        if have_labels:
            row["top1"] = float((top1 == labels).float().mean())
            row["top5"] = float((logits.topk(5, 1).indices == labels[:, None]).any(1).float().mean())
        rows.append(row)

    summarise("fp32", ref)
    for c in CONFIGS:
        qm = copy.deepcopy(model)
        for n, m in layers:
            parent = qm
            parts = n.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], make_qlayer(c, n, m, clips, layers, args, device))
        summarise(c, run_logits(qm.eval(), loader, device)[0])
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(path, rows):
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="resnet18", choices=["resnet18", "mobilenet_v2"])
    ap.add_argument("--weights", default="pretrained", choices=["pretrained", "random"])
    ap.add_argument("--checkpoint", help="state_dict .pth (overrides --weights; for offline machines)")
    ap.add_argument("--data-dir", help="ImageNet-style folder (class sub-folders => labels) or flat image folder")
    ap.add_argument("--synthetic", action="store_true", help="random inputs; plumbing check only")
    ap.add_argument("--num-images", type=int, default=1000)
    ap.add_argument("--calib-images", type=int, default=64)
    ap.add_argument("--layer-images", type=int, default=32, help="images for the per-layer table")
    ap.add_argument("--act-percentile", type=float, default=99.99, help="activation clip percentile (100 = max)")
    ap.add_argument("--first-last-int8", action="store_true", help="keep first and last layer INT8 in INT4 configs")
    ap.add_argument("--no-fold-bn", dest="fold_bn", action="store_false")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/accuracy")
    args = ap.parse_args()

    if not args.synthetic and not args.data_dir:
        ap.error("give --data-dir, or --synthetic for a plumbing check")
    if args.weights == "random" and not args.synthetic:
        print("WARNING: random weights with real images - numbers are meaningless")
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    model = build_model(args).to(device)
    layers = quant_layers(model)
    print("model=%s  layers=%d  device=%s  fold_bn=%s" % (args.model, len(layers), device, args.fold_bn))

    cal_loader, eval_loader = build_batches(args)
    print("calibrating activation ranges ...")
    clips = calibrate(model, cal_loader, layers, args.act_percentile, device)

    print("per-layer error (isolated) ...")
    layer_loader = [b for _, b in zip(range((args.layer_images + args.batch - 1) // args.batch), eval_loader)]
    lrows = layerwise(model, layer_loader, layers, clips, args, device)
    write_csv(os.path.join(args.out, "layerwise.csv"), lrows)

    print("end-to-end ...")
    erows = end_to_end(model, eval_loader, layers, clips, args, device)
    write_csv(os.path.join(args.out, "end_to_end.csv"), erows)

    print("\nEnd-to-end (%s, %d images):" % (args.model, erows[0]["n_images"]))
    print("%-9s %8s %8s %14s %14s" % ("config", "top1", "top5", "agree_vs_fp32", "logit_relMAE"))
    for r in erows:
        print("%-9s %8s %8s %14.4f %14.5f" % (
            r["config"],
            "%.4f" % r["top1"] if "top1" in r else "n/a",
            "%.4f" % r["top5"] if "top5" in r else "n/a",
            r["top1_agree_fp32"], r["logit_rel_mae"]))
    worst = sorted(lrows, key=lambda r: -r["int4"])[:5]
    print("\nLayers most sensitive to INT4 (relative MAE, isolated):")
    for r in worst:
        print("  %-28s K=%-5d groups=%-4d int8=%.4f int4=%.4f int8_24=%.4f" % (
            r["layer"], r["K"], r["groups"], r["int8"], r["int4"], r["int8_24"]))
    print("\nWrote %s/{layerwise,end_to_end}.csv" % args.out)


if __name__ == "__main__":
    main()
