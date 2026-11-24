# DeepSeek-V4 bring-up tooling

Model-free helpers for DeepSeek-V4 (Flash / Flash-Vision-Exp) parity work.
None of these run in CI; they produce the tiny checkpoint and oracle dumps the
parity tests consume.

| File | Purpose |
|---|---|
| `make_tiny_checkpoint.py` | Writes a tiny random checkpoint in DeepSeek's original layout (FP8 e4m3 128x128 blocks + UE8M0 scales, MXFP4 routed experts, BF16/F32 small tensors, vision tower + aligner) with the real tokenizer, from the production `config.json`. |
| `llamacpp_dump.cpp` | llama.cpp oracle: prefill + greedy decode, writing selected graph tensors (regex on ggml node names) and last-row logits as `.npy`. |
| `compare_reference_llamacpp.py` | Compares `python/reference/deepseek4.py` against a llama.cpp dump (per-step logits, per-layer prefill stages). |
| `gguf_remote_types.py` | Lists tensor types of remote GGUF shards with HTTP range reads only (no download). |

## Tiny model recipe

```bash
# Inputs: config.json, tokenizer.json, tokenizer_config.json and
# generation_config.json from deepseek-ai/DeepSeek-V4-Flash-Vision-Exp.
python scripts/deepseek4/make_tiny_checkpoint.py --ref <hf-dir> --out <tiny-hf>
python <llama.cpp>/convert_hf_to_gguf.py <tiny-hf> --outfile dsv4-tiny-mxfp4.gguf
python <llama.cpp>/convert_hf_to_gguf.py <tiny-hf> --mmproj --outtype f32 \
    --outfile mmproj-dsv4-tiny-f32.gguf
```

The tiny geometry keeps every structural feature of the release model:
6 layers with compress ratios `[0, 0, 4, 128, 4, 128]`, 3 hash-routed layers,
`hc_mult=4`, 16 experts (top-6), a 16-token window and `index_topk=8` so a few
hundred prompt tokens exercise both the CSA indexer selection and HCA blocks.
`head_dim` (256) must differ from `index_head_dim` (128): llama.cpp b12079
divides by zero in its KV-cache rotation setup when they are equal.
`vision_n_layers` must be at least 3 because the converter maps `aligner.w{bid}`
only for `bid < vision_n_layers`.

## Oracle dump

```bash
g++ -O2 -std=c++17 scripts/deepseek4/llamacpp_dump.cpp \
    -I<llama.cpp>/include -I<llama.cpp>/ggml/include \
    -L<llama.cpp>/build/bin -lllama -lggml -lggml-base \
    -Wl,-rpath,<llama.cpp>/build/bin -o dsv4_dump
./dsv4_dump -m dsv4-tiny-mxfp4.gguf -o dump_dir -f prompt.txt -n 8 \
    -r '(result_output|hc_attn_pre-[0-9]+|attn_out-[0-9]+|ffn_out-[0-9]+|lid_top_k-[0-9]+|.*compress.*)'
python scripts/deepseek4/compare_reference_llamacpp.py --gguf dsv4-tiny-mxfp4.gguf --dump dump_dir
```

The lightning-indexer top-k is a discrete selection. llama.cpp quantizes
activations for Q8_0 matmuls and stores indexer keys in F16, so near-tied
candidates can be selected differently from an FP32 evaluation. Compare
selections with a near-tie tolerance, not exact set equality.
