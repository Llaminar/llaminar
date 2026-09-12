# Qwen3.5-4B generation controls — 2026-09-11

All twelve canonical MTP-off controls for `Qwen3.5-4B-Q8_0.gguf` now have
individual green Release-server observations. This is diagnostic control
acquisition, not an approved token corpus, a fresh deep-HF matrix run, or an
image certificate. Definitions and runtime arguments still come from the
typed model/topology inventory; the table below records evidence, not another
matrix source.

Every cell retains FP32 activations, FP16 KV, seed 4242, and four continuous
384-token requests: fresh harbor, full harbor restore, partial mountain
restore, and full mountain restore. Each passing cell completes all eight
harness checks, including matching restored token streams, hybrid recurrent
state restoration, production-path evidence, memory checks and clean shutdown.
The `Decode20` fixture's separate numerical horizon does not shorten its
generation-control horizon.

## Results

| Canonical topology/profile | Whole cell (s) | Four HTTP requests (s) |
|---|---:|---:|
| Single CUDA | 26.977 | 18.201 |
| Single ROCm | 47.973 | 37.689 |
| LocalPP, CPU0→CPU1 | 115.522 | 108.966 |
| NodeTP, two CPU MPI ranks | 81.552 | 74.333 |
| Single CPU | 115.161 | 108.561 |
| Single CPU, Decode20 | 115.166 | 109.002 |
| LocalPP, two CUDA | 44.408 | 18.422 |
| LocalTP, two CUDA | 32.373 | 17.418 |
| LocalPP, CUDA→ROCm | 64.732 | 36.168 |
| LocalTP, CUDA+ROCm | 91.016 | 62.975 |
| LocalPP, two ROCm | 69.151 | 41.606 |
| LocalTP, two ROCm | 70.457 | 37.930 |

Immutable result roots under ignored `parity-results/`:

- `generation-qwen35-4b-single-gpu-unseen-01/`: first two controls, 75.462 s.
- `generation-qwen35-4b-remaining-unseen-01/`: remaining ten previously unseen
  controls, sequential and fail-fast, 800.249 s. All ten pass on their first
  generation attempt; no production fix or retry is needed during this batch.

Each root contains `report.json`, per-cell exact configuration, harness/server
logs, terminal PerfStats and `generation/observations.json` with all responses.
Both batches use the same unchanged Release build and the same authenticated
`hybrid-q8-prefix-prerequisites-01/prerequisites.json` receipt: 647 Unit and 137
ProductionParityPreflight registrations. Reused prerequisite time is zero in
both reports. The model remains in the sealed persistent tmpfs; both runs
report cache hits and zero model-copy bytes. No build, reconfiguration, prompt
change, precision change, or inference-policy override occurs between them.

## Economy interpretation

The two recorded batches total 875.711 seconds, excluding the previously paid
shared gate. This is not a 75-minute whole-campaign economy certificate. The
CPU cells spend most of their wall time inside the four required HTTP requests;
repeating Unit/preflight or copying model files is not the cause. GPU cell
totals also include readiness, graph setup, validation and shutdown, so neither
column should be reported as a standalone decode-throughput benchmark.

These results establish longer-path repeatability and prefix behavior. They do
not establish that token generation is universally cheaper than cached HF
checkpoint diagnostics, nor that matching tokens bounds KL error. Independent
numerical provenance and exact MTP-versus-serial comparisons remain separate
requirements before the observations can become routine CI certificates.
