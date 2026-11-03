#!/usr/bin/env python3
"""Compare the DeepSeek-V4 torch reference against llama.cpp oracle dumps."""
import argparse, sys
import numpy as np, torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "python"))
from reference.deepseek4 import DeepSeekV4Reference  # noqa: E402


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True)
ap.add_argument("--dump", required=True)
ap.add_argument("--dtype", default="float32")
a = ap.parse_args()
lines = open(f"{a.dump}/tokens.txt").read().split("\n")
prompt = [int(t) for t in lines[0].split()]
gen_ll = [int(t) for t in lines[1].split()]
ref = DeepSeekV4Reference(a.gguf, dtype=getattr(torch, a.dtype))
caps = {}
def cap(step, name, t):
    if step == 0:
        caps[name] = t.numpy()
seq = list(prompt)
for step, ll_tok in enumerate(gen_ll):
    logits = ref.forward(seq, (lambda n, t: cap(step, n, t)))
    last = logits[-1].float().numpy()
    ll = np.load(f"{a.dump}/step{step}__logits_last.npy")
    print(f"step {step}: ref argmax {int(last.argmax())} llama {ll_tok} | logits cos {cos(last, ll):.6f} maxabs {np.abs(last-ll).max():.4g}")
    seq.append(ll_tok)  # teacher-force llama.cpp's token so later steps stay comparable
# prefill per-layer checks
pairs = [("hc_attn_pre-{}", "layer{}_hc_attn_pre"), ("attn_out-{}", "layer{}_attn_out"), ("ffn_out-{}", "layer{}_ffn_out")]
n_layer = ref.cfg.n_layer
for il in range(n_layer):
    row = []
    for ll_name, ref_name in pairs:
        try:
            x = np.load(f"{a.dump}/step0__{ll_name.format(il)}.npy")
        except FileNotFoundError:
            continue
        r = caps[ref_name.format(il)]
        row.append(f"{ref_name.format(il).split('_',1)[1]} {cos(x, r):.6f}")
    print(f"layer {il}: " + " | ".join(row))
