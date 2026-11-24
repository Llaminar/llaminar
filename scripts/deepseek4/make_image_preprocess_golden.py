#!/usr/bin/env python3
"""Golden values for Llaminar's DeepSeek-V4 image admission (PIL is the authority).

For deterministic synthetic images this records what the official
``image_processor.py`` (via python/reference/deepseek4_vision.py, i.e. Pillow)
produces: grids, sentinel block, IMAGE-slot order, the resized/padded RGB8
pixels (SHA-256) and BF16 patch checksums. It also embeds small encoded PNG
images of every Pillow mode the decoder must convert like ``convert("RGB")``,
and the prompt expansion of a two-image prompt.

usage: make_image_preprocess_golden.py OUT.json
"""
import base64
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from reference.deepseek4_vision import preprocess, build_image_block, expand_prompt  # noqa: E402


def synthetic(w: int, h: int) -> np.ndarray:
    """Same formula as Test__DeepSeekV4ImagePreprocess.cpp: high-frequency content for resampling."""
    y, x = np.mgrid[0:h, 0:w].astype(np.int64)
    out = np.empty((h, w, 3), dtype=np.uint8)
    for c in range(3):
        out[..., c] = ((x * 7 + y * 13 + c * 101 + (x * y) % 37) & 255).astype(np.uint8)
    return out


SIZES = [(588, 588), (616, 266), (1000, 300), (300, 1000), (1700, 200), (100, 80), (3000, 2000),
         (300, 2500), (333, 777), (5000, 40), (14, 14), (1, 1), (2, 900)]


def main():
    cases = []
    for w, h in SIZES:
        img = Image.fromarray(synthetic(w, h))
        pi = preprocess(img)
        block, perm = build_image_block(pi.n_llm_h, pi.n_llm_w)
        bf = pi.patches.numpy().astype(np.float32).tobytes()
        cases.append({"width": w, "height": h, "vit_h": pi.n_vit_h, "vit_w": pi.n_vit_w,
                      "llm_h": pi.n_llm_h, "llm_w": pi.n_llm_w,
                      "out_width": int(pi.pixels.shape[1]), "out_height": int(pi.pixels.shape[0]),
                      "pixels_sha256": hashlib.sha256(pi.pixels.tobytes()).hexdigest(),
                      "patches_sha256": hashlib.sha256(bf).hexdigest(),
                      "block": block, "perm_sha256": hashlib.sha256(np.asarray(perm, np.int32).tobytes()).hexdigest()})

    decode = []
    base = synthetic(37, 23)
    for mode in ["RGB", "RGBA", "L", "LA", "P", "I;16"]:
        if mode == "RGBA":
            a = np.concatenate([base, (base[..., :1] ^ 0x5A)], axis=-1)
            img = Image.fromarray(a, "RGBA")
        elif mode == "LA":
            img = Image.fromarray(np.stack([base[..., 0], base[..., 1]], -1), "LA")
        elif mode == "L":
            img = Image.fromarray(base[..., 0], "L")
        elif mode == "P":
            img = Image.fromarray(base).convert("P", palette=Image.Palette.ADAPTIVE, colors=64)
        elif mode == "I;16":
            continue  # 16-bit sources are not admitted test material (Pillow keeps 16-bit modes)
        else:
            img = Image.fromarray(base)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        rgb = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), dtype=np.uint8)
        decode.append({"mode": mode, "png_base64": base64.b64encode(buf.getvalue()).decode(),
                       "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
                       "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})

    # Two images; the first placeholder starts at position 5 (pad 2), the second later.
    b1, _ = build_image_block(2, 3)
    b2, _ = build_image_block(3, 2)
    tokens = [0, 128803, 72, 101, 108, 129264, 33, 34, 129264, 35]
    expansion = {"tokens": tokens, "blocks": [b1, b2], "expanded": expand_prompt(tokens, [b1, b2])}

    json.dump({"source": "python/reference/deepseek4_vision.py (Pillow)", "cases": cases,
               "decode": decode, "expansion": expansion}, open(sys.argv[1], "w"), indent=1)
    print(f"wrote {len(cases)} geometry cases, {len(decode)} decode cases", file=sys.stderr)


if __name__ == "__main__":
    main()
