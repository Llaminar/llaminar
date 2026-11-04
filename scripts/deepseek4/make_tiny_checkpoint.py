#!/usr/bin/env python3
"""Build a tiny random DeepSeek-V4-Flash-Vision-Exp checkpoint in the original
DeepSeek layout (FP8 e4m3 128x128 block weights + UE8M0 scales, MXFP4 routed
experts, BF16/F32 small tensors, vision tower + aligner), with the real
tokenizer, so llama.cpp's convert_hf_to_gguf.py converts it exactly like the
production checkpoint (text GGUF + mmproj).

Weights are random but scaled so activations stay O(1) through the network.
"""
import argparse, json, math, os, shutil
import torch
from safetensors.torch import save_file

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e8m0(scale_pow2: torch.Tensor) -> torch.Tensor:
    """Encode exact powers of two as UE8M0 (biased exponent byte)."""
    exp = torch.log2(scale_pow2).round().to(torch.int32) + 127
    return exp.clamp(0, 254).to(torch.uint8).view(torch.float8_e8m0fnu)


def fp8_block(w: torch.Tensor, blk: int = 128):
    out, inp = w.shape
    so, si = math.ceil(out / blk), math.ceil(inp / blk)
    pad = torch.zeros(so * blk, si * blk)
    pad[:out, :inp] = w
    b = pad.view(so, blk, si, blk).abs().amax(dim=(1, 3)).clamp_min(1e-12)
    s = torch.exp2(torch.ceil(torch.log2(b / 448.0)))
    s_full = s.repeat_interleave(blk, 0).repeat_interleave(blk, 1)[:out, :inp]
    q = (w / s_full).to(torch.float8_e4m3fn)
    return q, e8m0(s)


def mxfp4(w: torch.Tensor):
    out, inp = w.shape
    assert inp % 32 == 0
    g = w.view(out, inp // 32, 32)
    amax = g.abs().amax(-1).clamp_min(1e-12)
    s = torch.exp2(torch.ceil(torch.log2(amax / 6.0)))
    x = g / s.unsqueeze(-1)
    mag = x.abs().unsqueeze(-1)
    idx = (mag - E2M1).abs().argmin(-1)
    code = idx | ((x < 0) & (idx > 0)).to(torch.int64) << 3
    code = code.view(out, inp).to(torch.uint8)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed.view(torch.int8), e8m0(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="dir with real config.json + tokenizer files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--moe-inter", type=int, default=128,
                    help="expert width; use 256 so K-quant/IQ2 recipes (256-element super-blocks) apply to ffn_down")
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    cfg = json.load(open(os.path.join(a.ref, "config.json")))
    D, L = 256, 6
    cfg.update({
        "hidden_size": D, "num_hidden_layers": L, "num_hash_layers": 3,
        "num_attention_heads": 4, "head_dim": 256, "qk_rope_head_dim": 64,
        "q_lora_rank": 128, "o_groups": 2, "o_lora_rank": 64,
        "moe_intermediate_size": a.moe_inter, "n_routed_experts": 16, "num_experts_per_tok": 6,
        "n_shared_experts": 1, "index_n_heads": 32, "index_head_dim": 128, "index_topk": 8,
        "sliding_window": 16, "num_nextn_predict_layers": 0,
        "compress_ratios": [0, 0, 4, 128, 4, 128],
        "vision_n_layers": 3, "vision_dim": 64, "vision_n_heads": 4, "vision_inter_dim": 96,
        "max_position_embeddings": 65536,
    })
    for k in ("dspark_block_size", "dspark_noise_token_id", "dspark_target_layer_ids", "dspark_markov_rank"):
        cfg.pop(k, None)
    V = cfg["vocab_size"]
    H, HD, QR = cfg["num_attention_heads"], cfg["head_dim"], cfg["q_lora_rank"]
    OG, OR, FF, NE = cfg["o_groups"], cfg["o_lora_rank"], cfg["moe_intermediate_size"], cfg["n_routed_experts"]
    IH, IHD, HC = cfg["index_n_heads"], cfg["index_head_dim"], cfg["hc_mult"]
    VD, VI, P, R = cfg["vision_dim"], cfg["vision_inter_dim"], cfg["vision_patch_size"], cfg["vision_downsample_ratio"]

    t = {}
    rn = lambda *s, std=1.0: torch.randn(*s) * std
    lin = lambda o, i: rn(o, i, std=1.0 / math.sqrt(i))
    bf = lambda x: x.to(torch.bfloat16)
    nrm = lambda n: bf(1.0 + 0.1 * torch.randn(n))

    def fp8(name, o, i, gain=1.0):
        q, s = fp8_block(lin(o, i) * gain)
        t[name + ".weight"], t[name + ".scale"] = q, s

    t["embed.weight"] = bf(rn(V, D))
    t["head.weight"] = bf(lin(V, D))
    t["norm.weight"] = nrm(D)
    t["hc_head_fn"] = rn(HC, HC * D, std=1.0 / math.sqrt(HC * D))
    t["hc_head_base"] = rn(HC, std=0.5)
    t["hc_head_scale"] = torch.full((1,), 0.5)
    for il in range(L):
        p = f"layers.{il}."
        ratio = cfg["compress_ratios"][il]
        t[p + "attn_norm.weight"] = nrm(D)
        t[p + "ffn_norm.weight"] = nrm(D)
        for hcn in ("attn", "ffn"):
            mix = (2 + HC) * HC
            t[p + f"hc_{hcn}_fn"] = rn(mix, HC * D, std=1.0 / math.sqrt(HC * D))
            t[p + f"hc_{hcn}_base"] = rn(mix, std=0.5)
            t[p + f"hc_{hcn}_scale"] = torch.tensor([0.5, 0.5, 1.0])
        t[p + "attn.attn_sink"] = rn(H, std=1.0)
        fp8(p + "attn.wq_a", QR, D)
        t[p + "attn.q_norm.weight"] = nrm(QR)
        fp8(p + "attn.wq_b", H * HD, QR)
        fp8(p + "attn.wkv", HD, D)
        t[p + "attn.kv_norm.weight"] = nrm(HD)
        fp8(p + "attn.wo_a", OG * OR, H * HD // OG)
        fp8(p + "attn.wo_b", D, OG * OR)
        if ratio:
            coff = 2 if ratio == 4 else 1
            t[p + "attn.compressor.ape"] = rn(ratio, coff * HD, std=0.5)
            t[p + "attn.compressor.norm.weight"] = nrm(HD)
            t[p + "attn.compressor.wgate.weight"] = bf(lin(coff * HD, D))
            t[p + "attn.compressor.wkv.weight"] = bf(lin(coff * HD, D))
            if ratio == 4:
                t[p + "attn.indexer.compressor.ape"] = rn(ratio, 2 * IHD, std=0.5)
                t[p + "attn.indexer.compressor.norm.weight"] = nrm(IHD)
                t[p + "attn.indexer.compressor.wgate.weight"] = bf(lin(2 * IHD, D))
                t[p + "attn.indexer.compressor.wkv.weight"] = bf(lin(2 * IHD, D))
                t[p + "attn.indexer.weights_proj.weight"] = bf(lin(IH, D))
                fp8(p + "attn.indexer.wq_b", IH * IHD, QR)
        t[p + "ffn.gate.weight"] = bf(lin(NE, D))
        t[p + "ffn.gate.bias"] = rn(NE, std=0.1)
        t[p + "ffn.gate.bias_vl"] = rn(NE, std=0.1)
        if il < cfg["num_hash_layers"]:
            t[p + "ffn.gate.tid2eid"] = torch.stack([torch.randperm(NE)[:cfg["num_experts_per_tok"]] for _ in range(V)]).to(torch.int64)
        for e in range(NE):
            for w, (o, i) in (("w1", (FF, D)), ("w2", (D, FF)), ("w3", (FF, D))):
                q, s = mxfp4(lin(o, i))
                t[p + f"ffn.experts.{e}.{w}.weight"], t[p + f"ffn.experts.{e}.{w}.scale"] = q, s
        fp8(p + "ffn.shared_experts.w1", FF, D)
        fp8(p + "ffn.shared_experts.w2", D, FF)
        fp8(p + "ffn.shared_experts.w3", FF, D)

    # vision tower + aligner (BF16 like the release)
    t["vision.patch_embed.proj.weight"] = bf(lin(VD, 3 * P * P))
    t["vision.patch_embed.proj.bias"] = bf(rn(VD, std=0.02))
    for b in range(cfg["vision_n_layers"]):
        p = f"vision.blocks.{b}."
        t[p + "norm1.weight"] = nrm(VD)
        t[p + "norm2.weight"] = nrm(VD)
        t[p + "attn.wqkv.weight"] = bf(lin(3 * VD, VD))
        t[p + "attn.wqkv.bias"] = bf(rn(3 * VD, std=0.02))
        t[p + "attn.wo.weight"] = bf(lin(VD, VD))
        t[p + "attn.wo.bias"] = bf(rn(VD, std=0.02))
        t[p + "mlp.w1.weight"] = bf(lin(2 * VI, VD))
        t[p + "mlp.w2.weight"] = bf(lin(VD, VI))
    t["vision.norm.weight"] = nrm(VD)
    t["aligner.w1.weight"] = bf(lin(D, VD * R * R))
    t["aligner.w1.bias"] = bf(rn(D, std=0.02))
    t["aligner.w2.weight"] = bf(lin(D, D))
    t["aligner.w2.bias"] = bf(rn(D, std=0.02))
    for n in ("image_start", "image_end", "image_newline", "image_pad"):
        t[n] = bf(rn(D))

    os.makedirs(a.out, exist_ok=True)
    t = {k: v.contiguous() for k, v in t.items()}
    save_file(t, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    json.dump({"metadata": {}, "weight_map": {k: "model.safetensors" for k in t}},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        shutil.copy(os.path.join(a.ref, f), a.out)
    n = sum(v.numel() for v in t.values())
    print(f"wrote {len(t)} tensors, {n/1e6:.1f}M elements -> {a.out}")


if __name__ == "__main__":
    main()
