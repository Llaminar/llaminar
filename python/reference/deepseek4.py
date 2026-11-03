"""
DeepSeek-V4 (Flash / Flash-Vision-Exp) PyTorch Reference Implementation

A dependency-light, full-precision reference of the DeepSeek-V4 text decoder
that reads weights directly from a llama.cpp ``deepseek4`` GGUF. It is the
mathematical oracle for Llaminar stage parity: every stage that Llaminar
publishes as a snapshot has a named capture here.

Mathematical source of truth is DeepSeek's reference ``inference/model.py``
(deepseek-ai/DeepSeek-V4-Flash-Vision-Exp @ 6821d6ad) with the QAT activation
quantization simulations removed (FP8 kv nope dims, FP4 indexer q/k), exactly
as llama.cpp evaluates the model. The randomized Hadamard rotation that the
reference applies to indexer q/k is orthogonal and therefore omitted: it does
not change any dot product in exact arithmetic.

Architecture summary (per main layer):
  - Hyper-connections: the residual stream is ``hc`` (=4) copies [S, hc, D].
    ``hc_pre`` mixes copies into one input using sigmoid "pre" weights computed
    from RMS-normalized flattened copies; ``hc_post`` writes the block output
    back with sigmoid "post" weights plus a Sinkhorn-normalized ``hc x hc``
    combination of the previous copies.
  - Attention: MQA with one 512-dim latent (K == V), low-rank Q
    (wq_a -> RMSNorm -> wq_b -> per-head RMS), RoPE on the last 64 dims,
    a raw sliding window, optional compressed KV blocks (ratio 4 "CSA" with
    overlapping gated pooling + lightning-indexer top-k, ratio 128 "HCA" with
    plain gated pooling, all complete blocks visible), per-head attention sinks,
    inverse RoPE on the output and a grouped low-rank output projection.
  - MoE: sqrt(softplus) scores, bias-shifted top-k (or token-id hash routing
    for the first ``hash_layer_count`` layers), normalized weights * scale,
    clamped SwiGLU experts plus one always-on shared expert.

The decode reference deliberately recomputes the full causal sequence: every
DeepSeek-V4 quantity at position p is a function of tokens [0, p] only, so the
last row of a full recompute is the exact value an incremental implementation
must produce. This makes the reference an independent check of incremental
compressor/indexer state machines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .loaders.dequantize import dequantize
from .loaders.gguf_parser import GGUFParser


@dataclass
class DeepSeekV4Config:
    """Hyperparameters read from ``deepseek4.*`` GGUF metadata."""

    n_layer: int
    n_embd: int
    n_head: int
    head_dim: int
    rope_dim: int
    q_lora_rank: int
    o_groups: int
    o_lora_rank: int
    window: int
    compress_ratios: List[int]
    n_expert: int
    n_expert_used: int
    n_hash_layers: int
    route_scale: float
    swiglu_limit: List[float]
    swiglu_limit_shexp: List[float]
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    norm_eps: float
    rope_theta: float
    compress_rope_theta: float
    rope_factor: float
    rope_orig_ctx: int
    beta_fast: float
    beta_slow: float
    vocab_size: int

    @staticmethod
    def from_metadata(md: Dict[str, object]) -> "DeepSeekV4Config":
        a = "deepseek4."
        n_layer = int(md[a + "block_count"])
        return DeepSeekV4Config(
            n_layer=n_layer,
            n_embd=int(md[a + "embedding_length"]),
            n_head=int(md[a + "attention.head_count"]),
            head_dim=int(md[a + "attention.key_length"]),
            rope_dim=int(md[a + "rope.dimension_count"]),
            q_lora_rank=int(md[a + "attention.q_lora_rank"]),
            o_groups=int(md[a + "attention.output_group_count"]),
            o_lora_rank=int(md[a + "attention.output_lora_rank"]),
            window=int(md[a + "attention.sliding_window"]),
            compress_ratios=[int(r) for r in md[a + "attention.compress_ratios"]][:n_layer],
            n_expert=int(md[a + "expert_count"]),
            n_expert_used=int(md[a + "expert_used_count"]),
            n_hash_layers=int(md[a + "hash_layer_count"]),
            route_scale=float(md[a + "expert_weights_scale"]),
            swiglu_limit=[float(x) for x in md[a + "swiglu_clamp_exp"]],
            swiglu_limit_shexp=[float(x) for x in md.get(a + "swiglu_clamp_shexp", md[a + "swiglu_clamp_exp"])],
            index_n_heads=int(md[a + "attention.indexer.head_count"]),
            index_head_dim=int(md[a + "attention.indexer.key_length"]),
            index_topk=int(md[a + "attention.indexer.top_k"]),
            hc_mult=int(md[a + "hyper_connection.count"]),
            hc_sinkhorn_iters=int(md[a + "hyper_connection.sinkhorn_iterations"]),
            hc_eps=float(md[a + "hyper_connection.epsilon"]),
            norm_eps=float(md[a + "attention.layer_norm_rms_epsilon"]),
            rope_theta=float(md[a + "rope.freq_base"]),
            compress_rope_theta=float(md[a + "attention.compress_rope_freq_base"]),
            rope_factor=float(md.get(a + "rope.scaling.factor", 1.0)),
            rope_orig_ctx=int(md.get(a + "rope.scaling.original_context_length", 0)),
            beta_fast=float(md.get(a + "rope.scaling.yarn_beta_fast", 32.0)),
            beta_slow=float(md.get(a + "rope.scaling.yarn_beta_slow", 1.0)),
            vocab_size=0,
        )


def load_gguf_tensors(path: str | Path) -> tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    """Read metadata and dequantize every tensor to FP32 numpy-backed torch tensors."""
    parser = GGUFParser(path)
    parser.open()
    parser.parse()
    tensors: Dict[str, torch.Tensor] = {}
    for info in parser.tensors:
        view = parser.read_tensor_data(info)
        raw = bytes(view)
        view.release()  # drop the mmap export so the parser can close
        arr = dequantize(raw, info.type, info.shape)
        tensors[info.name] = torch.from_numpy(np.array(arr, dtype=np.float32))
    md = dict(parser.metadata)
    parser.close()
    return md, tensors


def yarn_inv_freq(dim: int, base: float, orig_ctx: int, factor: float, beta_fast: float, beta_slow: float) -> torch.Tensor:
    """DeepSeek ``precompute_freqs_cis`` inverse frequencies (YaRN ramp when orig_ctx > 0).

    DeepSeek-V4 applies no YaRN magnitude (mscale) correction; llama.cpp cancels
    ggml's built-in correction with ``dsv4_rope_attn_factor``.
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    if orig_ctx > 0:
        def corr_dim(n_rot: float) -> float:
            return dim * math.log(orig_ctx / (n_rot * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corr_dim(beta_fast)), 0)
        high = min(math.ceil(corr_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = ((torch.arange(dim // 2, dtype=torch.float64) - low) / (high - low)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def apply_rope_tail(x: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor, rope_dim: int, inverse: bool = False) -> torch.Tensor:
    """Rotate adjacent (even, odd) pairs of the last ``rope_dim`` dims of ``x``.

    x: [..., S, ..., d] where the position axis is given by broadcasting ``pos``
    against ``x[..., 0]``; callers pass ``pos`` already shaped for broadcasting.
    """
    ang = pos.to(torch.float64)[..., None] * inv_freq  # [..., rope_dim/2]
    cos, sin = torch.cos(ang).to(x.dtype), torch.sin(ang).to(x.dtype)
    if inverse:
        sin = -sin
    head, tail = x[..., :-rope_dim], x[..., -rope_dim:]
    t = tail.unflatten(-1, (-1, 2))
    x0, x1 = t[..., 0], t[..., 1]
    r = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).flatten(-2)
    return torch.cat([head, r], dim=-1)


def rms(x: torch.Tensor, eps: float, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return y * weight if weight is not None else y


@dataclass
class DeepSeekV4Reference:
    """Full-recompute DeepSeek-V4 text reference over a GGUF checkpoint."""

    gguf_path: str
    dtype: torch.dtype = torch.float32
    cfg: DeepSeekV4Config = field(init=False)
    w: Dict[str, torch.Tensor] = field(init=False)

    def __post_init__(self) -> None:
        md, tensors = load_gguf_tensors(self.gguf_path)
        if md.get("general.architecture") != "deepseek4":
            raise ValueError(f"not a deepseek4 GGUF: {md.get('general.architecture')}")
        self.cfg = DeepSeekV4Config.from_metadata(md)
        self.w = {k: v.to(self.dtype) for k, v in tensors.items()}
        self.cfg.vocab_size = self.w["token_embd.weight"].shape[0]

    # ------------------------------------------------------------------ helpers
    def _t(self, name: str) -> torch.Tensor:
        return self.w[name]

    def _layer_rope(self, il: int) -> torch.Tensor:
        c = self.cfg
        if c.compress_ratios[il]:
            return yarn_inv_freq(c.rope_dim, c.compress_rope_theta, c.rope_orig_ctx, c.rope_factor, c.beta_fast, c.beta_slow)
        return yarn_inv_freq(c.rope_dim, c.rope_theta, 0, 1.0, c.beta_fast, c.beta_slow)

    def hc_mix(self, h: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor):
        """Return (pre [S,hc], post [S,hc], comb [S,hc(src),hc(dst)]) for one HC site."""
        c = self.cfg
        S, hc = h.shape[0], c.hc_mult
        flat = h.reshape(S, -1)
        mixes = rms(flat, c.norm_eps) @ fn.T
        pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + c.hc_eps
        post = 2 * torch.sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
        comb = (mixes[:, 2 * hc:] * scale[2] + base[2 * hc:]).view(S, hc, hc)
        comb = comb.softmax(-1) + c.hc_eps
        comb = comb / (comb.sum(-2, keepdim=True) + c.hc_eps)
        for _ in range(c.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + c.hc_eps)
            comb = comb / (comb.sum(-2, keepdim=True) + c.hc_eps)
        return pre, post, comb

    @staticmethod
    def hc_post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        # out[k] = post[k] * x + sum_j comb[j, k] * residual[j]
        return post[:, :, None] * x[:, None, :] + torch.einsum("sjk,sjd->skd", comb, residual)

    def compress(self, x: torch.Tensor, il: int, prefix: str, head_dim: int, ratio: int, inv_freq: torch.Tensor) -> torch.Tensor:
        """Gated-pooling compressed KV for every complete block: [S // ratio, head_dim]."""
        c = self.cfg
        S = x.shape[0]
        nb = S // ratio
        if nb == 0:
            return x.new_zeros(0, head_dim)
        kv = x @ self._t(f"blk.{il}.{prefix}_kv.weight").T
        sc = x @ self._t(f"blk.{il}.{prefix}_gate.weight").T
        ape = self._t(f"blk.{il}.{prefix}_ape.weight")  # [ratio, coff*hd]
        sc = sc + ape[torch.arange(S) % ratio]
        kv, sc = kv[: nb * ratio].view(nb, ratio, -1), sc[: nb * ratio].view(nb, ratio, -1)
        if ratio == 4:  # overlapping: previous block's first half + this block's second half
            d = head_dim
            prev_kv = torch.cat([kv.new_zeros(1, ratio, d), kv[:-1, :, :d]], 0)
            prev_sc = torch.cat([sc.new_full((1, ratio, d), float("-inf")), sc[:-1, :, :d]], 0)
            kv = torch.cat([prev_kv, kv[:, :, d:]], 1)
            sc = torch.cat([prev_sc, sc[:, :, d:]], 1)
        comp = (kv * sc.softmax(dim=1)).sum(1)
        comp = rms(comp, c.norm_eps, self._t(f"blk.{il}.{prefix}_norm.weight"))
        pos = torch.arange(nb) * ratio
        return apply_rope_tail(comp, pos, inv_freq, c.rope_dim)

    def attention(self, x: torch.Tensor, tokens: torch.Tensor, il: int, cap: Callable[[str, torch.Tensor], None]) -> torch.Tensor:
        c = self.cfg
        S = x.shape[0]
        H, hd, rd = c.n_head, c.head_dim, c.rope_dim
        p = f"blk.{il}."
        inv_freq = self._layer_rope(il)
        pos = torch.arange(S)
        qr = rms(x @ self._t(p + "attn_q_a.weight").T, c.norm_eps, self._t(p + "attn_q_a_norm.weight"))
        q = (qr @ self._t(p + "attn_q_b.weight").T).view(S, H, hd)
        q = rms(q, c.norm_eps)
        q = apply_rope_tail(q, pos[:, None], inv_freq, rd)
        kv = rms(x @ self._t(p + "attn_kv.weight").T, c.norm_eps, self._t(p + "attn_kv_a_norm.weight"))
        kv = apply_rope_tail(kv, pos, inv_freq, rd)
        cap("q", q.reshape(S, -1))
        cap("kv", kv)

        # key-visibility mask over [raw S | compressed nb]
        j = torch.arange(S)
        raw_vis = (j[None, :] <= pos[:, None]) & (pos[:, None] - j[None, :] < c.window)
        keys, vis = kv, raw_vis
        ratio = c.compress_ratios[il]
        if ratio:
            comp = self.compress(x, il, "attn_compressor", hd, ratio, inv_freq)
            nb = comp.shape[0]
            cap("compressed_kv", comp)
            b = torch.arange(nb)
            comp_vis = b[None, :] < ((pos[:, None] + 1) // ratio)
            if ratio == 4 and nb > 0:
                IH, ihd = c.index_n_heads, c.index_head_dim
                iq = (qr @ self._t(p + "indexer.attn_q_b.weight").T).view(S, IH, ihd)
                iq = apply_rope_tail(iq, pos[:, None], inv_freq, rd)
                ik = self.compress(x, il, "indexer_compressor", ihd, ratio, inv_freq)
                cap("indexer_q", iq.reshape(S, -1))
                cap("indexer_k", ik)
                iw = (x @ self._t(p + "indexer.proj.weight").T) * (ihd ** -0.5 * IH ** -0.5)
                score = torch.einsum("shd,bd->shb", iq, ik).relu()
                score = (score * iw[:, :, None]).sum(1)
                score = score.masked_fill(~comp_vis, float("-inf"))
                cap("indexer_score", score)
                k = min(c.index_topk, nb)
                top = score.topk(k, dim=-1).indices
                sel = torch.zeros_like(comp_vis)
                sel.scatter_(1, top, True)
                comp_vis = comp_vis & sel
                cap("indexer_selected", comp_vis.to(x.dtype))
            keys = torch.cat([kv, comp], 0)
            vis = torch.cat([raw_vis, comp_vis], 1)

        logits = torch.einsum("shd,td->sht", q, keys) * hd ** -0.5
        logits = logits.masked_fill(~vis[:, None, :], float("-inf"))
        sink = self._t(p + "attn_sinks.weight")  # [H]
        m = torch.maximum(logits.amax(-1), sink[None, :])
        e = torch.exp(logits - m[..., None])
        den = e.sum(-1) + torch.exp(sink[None, :] - m)
        o = torch.einsum("sht,td->shd", e / den[..., None], keys)
        o = apply_rope_tail(o, pos[:, None], inv_freq, rd, inverse=True)
        cap("attn_core", o.reshape(S, -1))

        G, R = c.o_groups, c.o_lora_rank
        wo_a = self._t(p + "attn_output_a.weight").view(G, R, -1)
        oa = torch.einsum("sgd,grd->sgr", o.reshape(S, G, -1), wo_a)
        out = oa.reshape(S, -1) @ self._t(p + "attn_output_b.weight").T
        return out

    def moe(self, x: torch.Tensor, tokens: torch.Tensor, il: int, cap: Callable[[str, torch.Tensor], None]) -> torch.Tensor:
        c = self.cfg
        p = f"blk.{il}."
        scores = F.softplus(x @ self._t(p + "ffn_gate_inp.weight").T).sqrt()
        cap("router_scores", scores)
        if il < c.n_hash_layers:
            idx = self._t(p + "ffn_gate_tid2eid.weight")[tokens].long()
        else:
            idx = (scores + self._t(p + "exp_probs_b.bias")).topk(c.n_expert_used, dim=-1).indices
        wts = scores.gather(1, idx)
        wts = wts / wts.sum(-1, keepdim=True) * c.route_scale
        cap("router_ids", idx.to(x.dtype))
        cap("router_weights", wts)
        lim = c.swiglu_limit[il]
        g_all, u_all, d_all = (self._t(p + n) for n in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"))
        y = torch.zeros_like(x)
        for s in range(x.shape[0]):
            for slot in range(c.n_expert_used):
                e = int(idx[s, slot])
                g = (g_all[e] @ x[s]).clamp(max=lim)
                u = (u_all[e] @ x[s]).clamp(-lim, lim)
                y[s] += wts[s, slot] * (d_all[e] @ (F.silu(g) * u))
        cap("moe_routed", y)
        lim_s = c.swiglu_limit_shexp[il]
        g = (x @ self._t(p + "ffn_gate_shexp.weight").T).clamp(max=lim_s)
        u = (x @ self._t(p + "ffn_up_shexp.weight").T).clamp(-lim_s, lim_s)
        sh = (F.silu(g) * u) @ self._t(p + "ffn_down_shexp.weight").T
        cap("moe_shared", sh)
        return y + sh

    def forward(self, tokens: Sequence[int], capture: Optional[Callable[[str, torch.Tensor], None]] = None) -> torch.Tensor:
        """Run the full causal sequence; returns logits for every position [S, vocab]."""
        c = self.cfg
        tokens = torch.as_tensor(list(tokens), dtype=torch.long)
        S = tokens.numel()

        def cap_fn(il: int):
            def f(name: str, t: torch.Tensor) -> None:
                if capture is not None:
                    capture(f"layer{il}_{name}" if il >= 0 else name, t.detach())
            return f

        x = self._t("token_embd.weight")[tokens]
        h = x[:, None, :].repeat(1, c.hc_mult, 1)
        cap_fn(-1)("embedding", x)
        for il in range(c.n_layer):
            cap = cap_fn(il)
            p = f"blk.{il}."
            res = h
            pre, post, comb = self.hc_mix(h, self._t(p + "hc_attn_fn.weight"), self._t(p + "hc_attn_scale.weight"), self._t(p + "hc_attn_base.weight"))
            a = torch.einsum("sj,sjd->sd", pre, h)
            cap("hc_attn_pre", a)
            a = rms(a, c.norm_eps, self._t(p + "attn_norm.weight"))
            a = self.attention(a, tokens, il, cap)
            cap("attn_out", a)
            h = self.hc_post(a, res, post, comb)
            cap("hc_attn_post", h.reshape(S, -1))
            res = h
            pre, post, comb = self.hc_mix(h, self._t(p + "hc_ffn_fn.weight"), self._t(p + "hc_ffn_scale.weight"), self._t(p + "hc_ffn_base.weight"))
            f = torch.einsum("sj,sjd->sd", pre, h)
            cap("hc_ffn_pre", f)
            f = rms(f, c.norm_eps, self._t(p + "ffn_norm.weight"))
            f = self.moe(f, tokens, il, cap)
            cap("ffn_out", f)
            h = self.hc_post(f, res, post, comb)
            cap("layer_out", h.reshape(S, -1))
        flat = h.reshape(S, -1)
        mixes = rms(flat, c.norm_eps) @ self._t("output_hc_fn.weight").T
        pre = torch.sigmoid(mixes * self._t("output_hc_scale.weight") + self._t("output_hc_base.weight")) + c.hc_eps
        y = torch.einsum("sj,sjd->sd", pre, h)
        cap_fn(-1)("hc_head", y)
        y = rms(y, c.norm_eps, self._t("output_norm.weight"))
        cap_fn(-1)("final_norm", y)
        logits = y @ self._t("output.weight").T
        cap_fn(-1)("logits", logits)
        return logits

    def greedy(self, prompt: Sequence[int], n_decode: int, capture_step: Optional[Callable[[int, str, torch.Tensor], None]] = None) -> List[int]:
        """Prefill then ``n_decode`` greedy steps (full recompute per step); returns generated ids."""
        seq = list(prompt)
        out: List[int] = []
        for step in range(n_decode + 1):
            cb = (lambda n, t, s=step: capture_step(s, n, t)) if capture_step else None
            logits = self.forward(seq, cb)
            nxt = int(logits[-1].argmax())
            out.append(nxt)
            seq.append(nxt)
        return out
