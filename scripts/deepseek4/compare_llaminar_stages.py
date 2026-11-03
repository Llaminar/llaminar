#!/usr/bin/env python3
"""Compare Llaminar DeepSeek-V4 stage dumps against python/reference/deepseek4.py.

usage: compare_llaminar_stages.py --gguf M.gguf --dump DIR --prompt-tokens "1 2 3" --generated "4 5 6"

DIR is a LLAMINAR_STAGE_DUMP_DIR of a greedy oneshot run. Occurrences of a stage
are ordered by their dump index; the first has the prompt's row count (prefill),
later one-row occurrences are decode steps. Decode step k is compared against the
last row of a full-recompute reference over prompt + generated[:k].
"""
import argparse, glob, os, re, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from reference.deepseek4 import DeepSeekV4Reference  # noqa: E402

# llaminar stage suffix -> (output file stem, reference capture suffix)
STAGES = [
    ("attn_norm", "output", "attn_norm"),
    ("q_norm_rope", "output", "q"),
    ("kv_norm_rope", "output", "kv"),
    ("attention", "output", "attn_core"),
    ("wo_b", "C", "attn_out"),
    ("hc_attn_write", "output", "hc_attn_post"),
    ("ffn_norm", "output", "ffn_norm"),
    ("moe_combine", "output", "ffn_out"),
    ("hc_ffn_write", "output", "layer_out"),
]


def load(stage_dir, stem):
    meta = {}
    path = Path(stage_dir) / "outputs" / f"{stem}_meta.txt"
    if not path.exists():
        cands = sorted((Path(stage_dir) / "outputs").glob("*_meta.txt"))
        if not cands:
            return None
        path = cands[0]
        stem = path.name[: -len("_meta.txt")]
    for line in path.read_text().splitlines():
        k, _, v = line.partition("=")
        meta[k] = v
    data = np.fromfile(Path(stage_dir) / "outputs" / f"{stem}.bin", dtype=np.float32)
    return data.reshape(int(meta["rows"]), int(meta["cols"]))


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--prompt-tokens", required=True)
    ap.add_argument("--generated", default="")
    a = ap.parse_args()
    prompt = [int(t) for t in a.prompt_tokens.split()]
    generated = [int(t) for t in a.generated.split()]
    ref = DeepSeekV4Reference(a.gguf)
    occurrences = {}
    for d in glob.glob(os.path.join(a.dump, "stage_*")):
        m = re.match(r"stage_(\d+)_[A-Z0-9_]+?_(layer\d+)_(.+)_rank\d+$", os.path.basename(d))
        if not m:
            continue
        occurrences.setdefault((m.group(2), m.group(3)), []).append((int(m.group(1)), d))
    steps = 1 + len(generated)
    for step in range(steps):
        seq = prompt + generated[:step]
        caps = {}
        ref.forward(seq, lambda n, t: caps.__setitem__(n, t.float().numpy()))
        print(f"== step {step} ({'prefill' if step == 0 else 'decode'}) tokens={len(seq)}")
        for layer in range(ref.cfg.n_layer):
            row = []
            for suffix, stem, cap in STAGES:
                occ = sorted(occurrences.get((f"layer{layer}", suffix), []))
                if step >= len(occ):
                    continue
                got = load(occ[step][1], stem)
                want = caps.get(f"layer{layer}_{cap}")
                if got is None or want is None:
                    continue
                want = want.reshape(len(seq), -1)
                want = want if step == 0 else want[-1:]
                got = got[: want.shape[0], : want.shape[1]]
                row.append(f"{suffix}={cos(got, want):.5f}")
            print(f"  layer {layer}: " + " ".join(row))


if __name__ == "__main__":
    main()
