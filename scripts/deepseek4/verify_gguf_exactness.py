#!/usr/bin/env python3
"""Check a remote DeepSeek-V4 GGUF against the official safetensors, value by value.

usage: verify_gguf_exactness.py [--json OUT] GGUF_SHARD_URL [GGUF_SHARD_URL ...]

Only HTTP range reads are used: the safetensors headers of
deepseek-ai/DeepSeek-V4-Flash-Vision-Exp (pinned revision) and the GGUF tensor
directories are parsed, then sampled rows are fetched and dequantized from both
sides:

* routed experts: official FP4 e2m1 + e8m0/32 scale vs GGUF MXFP4,
* FP8 e4m3 projections (128x128 e8m0 block scales) vs the GGUF type
  (BF16/F32 exact; Q8_0 reported with its relative L2 error),
* BF16 embedding/head/router vs the GGUF type.

"exact" means every sampled value is bit-identical after dequantization. This is
a SAMPLED check (one row per listed tensor), not exhaustive checkpoint equality.
--json records every compared byte range (URL, offset, length) on both sides.
Requires numpy and torch (for float8_e4m3fn).
"""
import json, struct, sys, urllib.request

import numpy as np
import torch

OFFICIAL = ("https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/resolve/"
            "6821d6ad3681a4b137b066b76094fa82ebd0a380/")
TREE = "https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/tree/main"


RANGES = []  # (url, start, length) of every payload read since the last reset


def rng(url, start, n, record=True):
    if record:
        RANGES.append((url, start, n))
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + n - 1}"})
    return urllib.request.urlopen(req).read()


def official_directory():
    tensors = {}
    files = sorted(f["path"] for f in json.load(urllib.request.urlopen(TREE)) if f["path"].endswith(".safetensors"))
    for name in files:
        n = struct.unpack("<Q", rng(OFFICIAL + name, 0, 8, record=False))[0]
        header = json.loads(rng(OFFICIAL + name, 8, n, record=False))
        header.pop("__metadata__", None)
        for key, value in header.items():
            tensors[key] = (OFFICIAL + name, 8 + n + value["data_offsets"][0], value["dtype"])
    return tensors


SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def gguf_directory(url):
    buf, pos = bytearray(), 0

    def read(n):
        nonlocal pos
        while pos + n > len(buf):
            buf.extend(rng(url, len(buf), 8 << 20, record=False))
        out = bytes(buf[pos:pos + n])
        pos += n
        return out

    u32 = lambda: struct.unpack("<I", read(4))[0]
    u64 = lambda: struct.unpack("<Q", read(8))[0]
    text = lambda: read(u64()).decode()

    def skip(t):
        if t in SCALAR:
            read(SCALAR[t])
        elif t == 8:
            text()
        else:
            et, count = u32(), u64()
            if et in SCALAR:
                read(SCALAR[et] * count)
            else:
                for _ in range(count):
                    skip(et)

    assert read(4) == b"GGUF"
    u32()
    n_tensors, n_kv = u64(), u64()
    alignment = 32
    for _ in range(n_kv):
        key, t = text(), u32()
        if key == "general.alignment":
            alignment = u32()
        else:
            skip(t)
    infos = []
    for _ in range(n_tensors):
        name = text()
        dims = [u64() for _ in range(u32())]
        infos.append((name, dims, u32(), u64()))
    data = (pos + alignment - 1) // alignment * alignment
    return {name: (url, data + off, typ, dims) for name, dims, typ, off in infos}


E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
KV_MXFP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float32)


def e8m0(raw):
    return np.ldexp(1.0, np.frombuffer(raw, np.uint8).astype(np.int32) - 127).astype(np.float32)


def bf16(raw):
    return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)


def gguf_row(entry, row, k):
    url, base, typ, _ = entry
    if typ == 39:  # MXFP4: e8m0 + 16 nibble bytes; low nibbles first half, high second
        nb = k // 32
        raw = np.frombuffer(rng(url, base + row * nb * 17, nb * 17), np.uint8).reshape(nb, 17)
        scale = np.ldexp(1.0, raw[:, 0].astype(np.int32) - 127) * 0.5
        q = raw[:, 1:]
        return (np.concatenate([KV_MXFP4[q & 15], KV_MXFP4[q >> 4]], 1) * scale[:, None]).ravel()
    if typ == 8:  # Q8_0
        nb = k // 32
        raw = np.frombuffer(rng(url, base + row * nb * 34, nb * 34), np.uint8).reshape(nb, 34)
        d = raw[:, :2].copy().view(np.float16).astype(np.float32)
        return (raw[:, 2:].copy().view(np.int8).astype(np.float32) * d).ravel()
    if typ == 30:
        return bf16(rng(url, base + row * k * 2, k * 2))
    if typ == 0:
        return np.frombuffer(rng(url, base + row * k * 4, k * 4), np.float32)
    raise ValueError(f"unhandled GGUF type {typ}")


def official_row(official, name, row, k):
    url, base, dtype = official[name + ".weight"] if name + ".weight" in official else official[name]
    if dtype == "I8":  # FP4 e2m1 x2, low nibble first; e8m0 scale per 32
        raw = np.frombuffer(rng(url, base + row * k // 2, k // 2), np.uint8)
        values = np.stack([E2M1[raw & 15], E2M1[raw >> 4]], 1).ravel()
        surl, sbase, _ = official[name + ".scale"]
        return values * np.repeat(e8m0(rng(surl, sbase + row * (k // 32), k // 32)), 32)
    if dtype == "F8_E4M3":  # 128x128 e8m0 block scales
        values = torch.frombuffer(bytearray(rng(url, base + row * k, k)), dtype=torch.float8_e4m3fn).float().numpy()
        blocks = (k + 127) // 128
        surl, sbase, _ = official[name + ".scale"]
        return values * np.repeat(e8m0(rng(surl, sbase + (row // 128) * blocks, blocks)), 128)[:k]
    if dtype == "BF16":
        return bf16(rng(url, base + row * k * 2, k * 2))
    raise ValueError(f"unhandled official dtype {dtype}")


# (official name, GGUF name, row-in-GGUF, row-in-official, K)
SAMPLES = [
    ("layers.0.ffn.experts.0.w1", "blk.0.ffn_gate_exps.weight", 0, 0, 4096),
    ("layers.21.ffn.experts.200.w3", "blk.21.ffn_up_exps.weight", 200 * 2048 + 1999, 1999, 4096),
    ("layers.42.ffn.experts.255.w2", "blk.42.ffn_down_exps.weight", 255 * 4096 + 4095, 4095, 2048),
    ("layers.0.attn.wq_b", "blk.0.attn_q_b.weight", 123, 123, 1024),
    ("layers.10.attn.wkv", "blk.10.attn_kv.weight", 77, 77, 4096),
    ("layers.5.attn.wo_b", "blk.5.attn_output_b.weight", 4000, 4000, 8192),
    ("layers.2.attn.wq_a", "blk.2.attn_q_a.weight", 500, 500, 4096),
    ("layers.2.ffn.shared_experts.w2", "blk.2.ffn_down_shexp.weight", 9, 9, 2048),
    ("layers.20.attn.indexer.wq_b", "blk.20.indexer.attn_q_b.weight", 3, 3, 1024),
    ("layers.0.ffn.gate", "blk.0.ffn_gate_inp.weight", 3, 3, 4096),
    ("embed", "token_embd.weight", 1000, 1000, 4096),
    ("head", "output.weight", 50000, 50000, 4096),
]
TYPE_NAMES = {0: "F32", 8: "Q8_0", 30: "BF16", 39: "MXFP4"}


def main():
    args = sys.argv[1:]
    json_out = None
    if args[:1] == ["--json"]:
        json_out, args = args[1], args[2:]
    official = official_directory()
    gguf = {}
    for url in args:
        gguf.update(gguf_directory(url))
    records = []
    for oname, gname, grow, orow, k in SAMPLES:
        if gname not in gguf:
            print(f"{gname:34s} missing")
            continue
        RANGES.clear()
        a = official_row(official, oname, orow, k)
        official_ranges = list(RANGES)
        RANGES.clear()
        b = gguf_row(gguf[gname], grow, k)
        gguf_ranges = list(RANGES)
        exact = np.array_equal(a, b)
        rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))
        typ = TYPE_NAMES.get(gguf[gname][2], gguf[gname][2])
        print(f"{gname:34s} {typ:6s} {'exact' if exact else f'rel_l2={rel:.2e}'}")
        records.append({
            "official_tensor": oname, "official_row": orow, "gguf_tensor": gname, "gguf_row": grow,
            "values": k, "gguf_type": typ, "bit_identical": bool(exact), "relative_l2": rel,
            "official_ranges": [{"url": u, "offset": o, "length": n} for u, o, n in official_ranges],
            "gguf_ranges": [{"url": u, "offset": o, "length": n} for u, o, n in gguf_ranges],
        })
    if json_out:
        with open(json_out, "w") as f:
            json.dump({"official": OFFICIAL, "gguf_shards": args, "scope": "sampled rows only; not exhaustive",
                       "samples": records}, f, indent=1)


if __name__ == "__main__":
    main()
