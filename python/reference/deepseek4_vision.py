"""
DeepSeek-V4-Flash-Vision-Exp image path: PyTorch reference over a llama.cpp mmproj GGUF.

Mathematical sources (deepseek-ai/DeepSeek-V4-Flash-Vision-Exp @ 6821d6ad):
``inference/image_processor.py`` (resize, patches, sentinel block),
``inference/vision.py`` (ViT, aligner) and ``inference/model.py``
(``merge_image_embeddings``, ``get_image_visible``, ``Gate.bias_vl``).
Prompt token ids follow the production vLLM processor
(``vllm/models/deepseek_v4/common/mm_preprocess.py``): every placeholder
``<｜deepseek_image｜>`` is replaced by an in-vocab sentinel block with ids
``IMAGE_SENTINEL_BASE_ID + type`` and a ``3 - start % 4`` IMAGE_PAD prefix.
That is the official out-of-vocab scheme (``vocab_size + type``) under another
spelling: every sentinel row is overwritten before the decoder sees it.

Pipeline for one image:
  1. RGB, aspect/min-pixel adjustment, PIL resize or ImageOps.pad (gray 127)
     to a 14-pixel grid, (x / 255 - 0.5) / 0.5 rounded to BF16.
  2. ViT: patch embedding (linear over (3, 14, 14)), N blocks of
     RMSNorm -> bidirectional MHA with 2D RoPE (rotate-half; first half of the
     rotary frequencies follow the row, second half the column) ->
     RMSNorm -> SwiGLU, final RMSNorm. Norm eps 1e-6.
  3. Aligner: zero-pad the patch grid to a multiple of 3, 3x3 unfold
     (channel-major, like ``F.unfold``), GELU(erf) MLP into the text width.
  4. Block: [IMAGE_PAD]*lead START, aligner rows interleaved in pairs
     ("N-layout", each row ends with NEWLINE, an odd row count gets a PAD
     row), [IMAGE_PAD]*pad_last, END. IMAGE rows take aligner outputs in
     ``perm`` order, the other sentinels learned vectors.

Text-model effects (``DeepSeekV4Reference.forward(..., vision=...)``):
  - routing of every sentinel position uses ``exp_probs_b_vl`` (also on hash
    layers, where it replaces the token table);
  - inside a [START, END] span, raw-window attention extends to the whole
    span in both directions (``get_window_topk_idxs_visible``).
Activations are FP32 (Llaminar's policy); the official path runs BF16.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
COMPRESS_PAD_TO = 4
IMAGE_SENTINEL_BASE_ID = 129257  # <|place_holder_mm_span_0431|> .. _0435|>
IMAGE_PLACEHOLDER_ID = 129264    # <｜deepseek_image｜>
MAX_IMAGE_TOKENS = 384
MAX_WH_RATIO = 8
VISION_ROPE_THETA = 10000.0
VISION_NORM_EPS = 1e-6


# --------------------------------------------------------------- preprocessing
def grid_tokens(best_height: int, best_width: int, p: int, r: int) -> Tuple[int, int, int]:
    n_llm_h = math.ceil((best_height // p) / r)
    n_llm_w = math.ceil((best_width // p) / r)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(height, width, p, r, max_n_token):
    ratio = height / width
    max_w_float = math.sqrt((max_n_token - 2) / ratio + 0.25) - 0.5
    max_h_float = max_w_float * ratio
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width, best_height = max_w * p * r, max_h * p * r
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        assert max_w > 1
        best_width, best_height = max_w * p * r, max_h * p * r
    else:
        max_w, max_h = math.floor(max_w_float), math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(max_w * p * r / width, max_h * p * r / height)
        best_width = math.floor(width * beta / p) * p
        best_height = math.floor(height * beta / p) * p
    n_llm_h, n_llm_w, num_tokens = grid_tokens(best_height, best_width, p, r)
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(height, width, best_height, best_width, p, r, max_n_token):
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(best_height, best_width, p, r)
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(height, width, p, r, budget)
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


@dataclass
class PreparedImage:
    patches: torch.Tensor  # [n_vit_h * n_vit_w, 3 * p * p] BF16 values in FP32
    n_vit_h: int
    n_vit_w: int
    n_llm_h: int
    n_llm_w: int
    pixels: np.ndarray     # [best_height, best_width, 3] uint8 after resize/pad


def preprocess(image, p: int = 14, r: int = 3, min_pixels: int = 147456) -> PreparedImage:
    """``image_processor.load_image`` for an already-opened PIL image."""
    from PIL import ImageOps

    image = image.convert("RGB")
    width, height = image.size
    if width > height * MAX_WH_RATIO:
        width = height * MAX_WH_RATIO
    if 0 < width * height < min_pixels:
        ratio = (min_pixels / (width * height)) ** 0.5
        width, height = int(width * ratio), int(height * ratio)
    best_width, best_height = math.ceil(width / p) * p, math.ceil(height / p) * p
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(height, width, best_height, best_width, p, r, MAX_IMAGE_TOKENS)
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if image.width >= MAX_WH_RATIO * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    pixels = np.asarray(image, dtype=np.uint8).copy()
    x = torch.from_numpy(pixels.astype(np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16).float()
    patches = x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3 * p * p)
    return PreparedImage(patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w, pixels)


def build_image_block(n_llm_h: int, n_llm_w: int) -> Tuple[List[int], List[int]]:
    """Pad-free sentinel types (final order) and the aligner-row order of IMAGE slots."""
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = torch.tensor(([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_PAD] * (row_len * pad_h))
    order = torch.arange(rows * row_len).view(rows // 2, 2, row_len).transpose(1, 2).reshape(-1)
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(n_llm_h * n_llm_w).view(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    block = [IMAGE_START] + types[order].tolist() + [IMAGE_PAD] * pad_last + [IMAGE_END]
    return block, perm.tolist()


def expand_prompt(tokens: Sequence[int], blocks: Sequence[Sequence[int]]) -> List[int]:
    """Replace each placeholder token with its sentinel block and compressor-alignment pad."""
    out: List[int] = []
    it = iter(blocks)
    for t in tokens:
        if t != IMAGE_PLACEHOLDER_ID:
            out.append(t)
            continue
        block = next(it)
        pad = COMPRESS_PAD_TO - 1 - len(out) % COMPRESS_PAD_TO
        out += [IMAGE_SENTINEL_BASE_ID + IMAGE_PAD] * pad + [IMAGE_SENTINEL_BASE_ID + t2 for t2 in block]
    if next(it, None) is not None:
        raise ValueError("more images than placeholders")
    return out


def is_sentinel(tokens: torch.Tensor) -> torch.Tensor:
    return (tokens >= IMAGE_SENTINEL_BASE_ID) & (tokens < IMAGE_SENTINEL_BASE_ID + 5)


def image_visible(tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``get_image_visible``: per-token span extents to the left/right (0 outside spans)."""
    S = tokens.numel()
    idx = torch.arange(S)
    is_start = tokens == IMAGE_SENTINEL_BASE_ID + IMAGE_START
    is_end = tokens == IMAGE_SENTINEL_BASE_ID + IMAGE_END
    valid = (is_start.cumsum(0) > is_end.cumsum(0)) | is_end
    starts = torch.where(is_start, idx, 0).cummax(0)[0]
    left = (idx - starts) * valid
    ends = torch.where(is_end, idx, S).flip(0).cummin(0)[0].flip(0)
    right = (ends - idx) * valid
    return left.clamp(max=MAX_IMAGE_TOKENS - 1), right.clamp(max=MAX_IMAGE_TOKENS)


def raw_window_visibility(tokens: torch.Tensor, window: int) -> torch.Tensor:
    """[S, S] raw-key visibility of ``get_window_topk_idxs`` / ``..._visible``."""
    S = tokens.numel()
    pos = torch.arange(S)
    j = torch.arange(S)
    if not bool(is_sentinel(tokens).any()):
        return (j[None, :] <= pos[:, None]) & (pos[:, None] - j[None, :] < window)
    left, right = image_visible(tokens)
    width = min(S, window + MAX_IMAGE_TOKENS)
    left_add = (left - (window - 1)).clamp(min=0)
    starts = (pos - (window - 1) - left_add).clamp(min=0)
    return (j[None, :] >= starts[:, None]) & (j[None, :] < (starts + width)[:, None]) & (j[None, :] <= (pos + right)[:, None])


# ------------------------------------------------------------------ the tower
def vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def rotate_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def rms(x: torch.Tensor, w: torch.Tensor, eps: float = VISION_NORM_EPS) -> torch.Tensor:
    return w * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps))


@dataclass
class DeepSeekV4Vision:
    """ViT + aligner + sentinel vectors from a ``deepseek4v`` mmproj GGUF."""

    mmproj_path: str
    w: Dict[str, torch.Tensor] = field(init=False)
    md: Dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        from .deepseek4 import load_gguf_tensors

        self.md, self.w = load_gguf_tensors(self.mmproj_path)
        if self.md.get("clip.projector_type") != "deepseek4v":
            raise ValueError(f"not a deepseek4v mmproj: {self.md.get('clip.projector_type')}")
        self.n_layers = int(self.md["clip.vision.block_count"])
        self.dim = int(self.md["clip.vision.embedding_length"])
        self.heads = int(self.md["clip.vision.attention.head_count"])
        self.patch = int(self.md["clip.vision.patch_size"])
        self.merge = int(self.md["clip.vision.projector.scale_factor"])
        self.min_pixels = int(self.md["clip.vision.image_min_pixels"])

    def vit(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        w, D, H = self.w, self.dim, self.heads
        hd = D // H
        x = patches @ w["v.patch_embd.weight"].reshape(D, -1).T + w["v.patch_embd.bias"]
        cos, sin = vision_cos_sin(n_h, n_w, hd // 2, VISION_ROPE_THETA)
        N = x.shape[0]
        for i in range(self.n_layers):
            p = f"v.blk.{i}."
            a = rms(x, w[p + "ln1.weight"])
            qkv = a @ w[p + "attn_qkv.weight"].T + w[p + "attn_qkv.bias"]
            q, k, v = (t.view(N, H, hd) for t in qkv.chunk(3, dim=-1))
            q, k = rotate_half(q, cos, sin), rotate_half(k, cos, sin)
            o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
            x = x + o.transpose(0, 1).reshape(N, D) @ w[p + "attn_out.weight"].T + w[p + "attn_out.bias"]
            m = rms(x, w[p + "ln2.weight"])
            g, u = m @ w[p + "ffn_gate.weight"].T, m @ w[p + "ffn_up.weight"].T
            x = x + (F.silu(g) * u) @ w[p + "ffn_down.weight"].T
        return rms(x, w["v.post_ln.weight"])

    def aligner(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.merge
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        h = F.gelu(x @ self.w["mm.1.weight"].T + self.w["mm.1.bias"])
        return h @ self.w["mm.2.weight"].T + self.w["mm.2.bias"]

    def encode(self, img: PreparedImage) -> Tuple[torch.Tensor, List[int]]:
        """Aligner outputs in block (``perm``) order and the pad-free block types."""
        block, perm = build_image_block(img.n_llm_h, img.n_llm_w)
        emb = self.aligner(self.vit(img.patches, img.n_vit_h, img.n_vit_w), img.n_vit_h, img.n_vit_w)
        return emb[torch.tensor(perm, dtype=torch.long)], block

    def sentinel_table(self) -> torch.Tensor:
        w = self.w
        return torch.stack([w["v.token_embd.img_start"], w["v.token_embd.img_pad"], w["v.token_embd.img_pad"],
                            w["v.image_newline"], w["v.token_embd.img_end"]])


def merge_embeddings(x: torch.Tensor, tokens: torch.Tensor, image_rows: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Overwrite sentinel rows: IMAGE rows take ``image_rows`` in order, others their learned vector."""
    x = x.clone()
    mask = is_sentinel(tokens)
    types = (tokens - IMAGE_SENTINEL_BASE_ID).clamp(0, 4)
    x[mask] = table[types[mask]]
    img = tokens == IMAGE_SENTINEL_BASE_ID + IMAGE
    if int(img.sum()) != image_rows.shape[0]:
        raise ValueError(f"{int(img.sum())} IMAGE sentinels but {image_rows.shape[0]} image rows")
    x[img] = image_rows
    return x
