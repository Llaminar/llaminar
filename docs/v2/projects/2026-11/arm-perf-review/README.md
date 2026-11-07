# ARM performance / memory review — idea index

Created: 2026-11-07 (review carried out by read-only session `01a1b0ec`).
Reviewed against: `codex/qfn-ngram-continuation-arm64` @ `be6658c38`
(rebase onto `feat/qfn-ngram-continuation` @ `2f01249c9` complete; 28 ARM
commits replayed).

Scope: performance and memory improvement ideas gathered during the review of
PR #32 (`feat(cpu): add native ARM64 inference kernels`) and PR #35
(`feat(cpu): native ARM64 NEON execution for MXFP4`), plus the DGX Spark
(GB10) context in the Linux ARM handoff. One markdown file per idea.

This is a review artifact, not an implementation ledger. The ARM implementation
session remains the sole writer of production source; the tester session runs
all builds/measurements. Nothing here is a measured economy claim. File:line
references are as of the head above and may drift.

## Status legend

- **Open** — not implemented at the reviewed head.
- **Partial** — a related change landed; the specific idea, or a remaining part
  of it, is still open.
- **Addressed** — implemented at the reviewed head; commit(s) cited. The entry
  is kept for context and any residual verification.

## Index

| # | Idea | File | Status |
|---|---|---|---|
| 1 | Local kernel-benchmark A/B harness | `01-local-benchmark-harness.md` | Partial (Perf__ suite exists for projection; gaps below) |
| 2 | Shared CPU/GPU (UMA) memory admission | `02-uma-shared-memory-admission.md` | Partial (joint pool implemented `80985c08e`; live GB10 CUDA+CPU validation open) |
| 3 | Move projection metadata validation out of the K loop | `03-projection-metadata-validation-hoist.md` | Open |
| 4 | Wider decode column panel + activation reuse | `04-decode-column-panel-activation-reuse.md` | Open |
| 5 | Heterogeneous (X925/A725) core scheduling | `05-heterogeneous-core-scheduling.md` | Partial (dynamic tile claiming landed; no typed weighted policy) |
| 6 | I8MM (SMMLA) row-pair dot-product | `06-i8mm-row-pair-dot.md` | Open (highest-value next kernel item; see idea 14) |
| 7 | Hot-path allocation / cache cleanup (RoPE, RMSNorm, gamma TLS) | `07-hot-path-allocation-cleanup.md` | Open |
| 8 | ARM prepared-weight layout (drop dead compensation bytes, compact expansions) | `08-arm-prepared-weight-layout.md` | Open (ABI change; last) |
| 9 | Non-volatile rounding barrier evidence | `09-rounding-barrier-evidence.md` | Open (evidence required first) |
| 10 | FP32 RMSNorm scalar sum order + sanitizer warnings (issue #36) | `10-rmsnorm-fp32-sumsq-order.md` | Addressed (`3086894b5`, `ac3289bd9`); residuals listed |
| 11 | Attention validation hoist + integer-dot range proof | `11-attention-validation-and-integer-dot-range.md` | Open |
| 12 | Cache-topology-derived tiling / prefetch on GB10 | `12-cache-topology-tiling.md` | Partial (GB10 overlays installed; class-specific tuning remains) |
| 13 | Fewer OpenMP regions per decode step | `13-openmp-region-count-and-wakeup.md` | Partial (bounded spin `27585692d`, pooled bundle `408ff90e5`) |
| 14 | Dense-model MTP verifier economics (27B) | `14-dense-mtp-verifier-economics.md` | Open (MTP dynamic 4.23 vs off 4.61 tok/s) |
| 15 | NUMA-less and cache-unpublished Linux ARM hosts | `15-numa-less-hosts.md` | Partial (memory attribution `a572c3c9c`) |
| 16 | MoE routed experts, GDN projection and LM head in CPU decode | `16-moe-expert-and-lm-head-decode.md` | Open (bundle part addressed by `408ff90e5`) |

Measured numbers for all ideas: `00-measured-baseline.md` (GB10 serving,
stage attribution, spin sweep, thread scaling, overlay statistics).

## Cross-cutting notes

- **Exactness gate:** every native-path change must preserve byte-identical
  serial/MTP output. Kernel-level byte authentication with independent oracles
  is the standard, not tolerance-based comparison.
- **MXFP4 coordination:** any shared change to the NEON projection tile or
  `NibbleMap` decode must be coordinated with the PR #35 lane so there is one
  change certifying all `NibbleMap`/scale representations.
- **MTP/admission test lanes:** per the project owner (2026-11-07), the MTP/
  admission test lanes are mostly for exhaustive debugging — they may be run
  freely for this work (e.g. verifier-depth and acceptance measurement, idea
  14), but they are **not** intended to become mandatory CI/CD gates.
- **Mac vs Spark:** Mac (M1 Max, 8P+2E, no libnuma, no production engine
  configure) is the fast correctness loop; DGX Spark (GB10, 10x X925 + 10x
  A725, one NUMA node, CPU/GPU UMA) is where scheduling, memory and bandwidth
  claims must be measured. Worker sweeps must cover 1..20 physical cores with
  X925-only / A725-only / mixed placements measured separately.
- **Later this cycle:** one natural checkpoint for these ideas is after the
  complete engine Unit/preflight gate on ARM; kernel-only wins then need
  full-inference confirmation.
