# Measured ARM baseline (GB10 + M1 Max) — 2026-11-07

Status: **Reference**. What has actually been measured on the ARM branch, so
follow-up ideas start from numbers rather than estimates. All runs are
unprofiled unless marked; seals live under
`/Users/floriankinder/llaminar-spark-results/` on the main session's Mac.

## Hosts

| Host | CPU | Notes |
|---|---|---|
| DGX Spark .79 / .80 (GB10) | 10x Cortex-X925 (CPU 5-9, 15-19) + 10x Cortex-A725 (CPU 0-4, 10-14), no SMT, one NUMA node, 121 GB LPDDR5x shared with the GPU | X925 is about 2.8x an A725 per thread on NativeVNNI kernels. `core_id` repeats across the two clusters: count physical cores by thread-sibling sets (`host_cores.py`). |
| M1 Max | 8P + 2E, no NUMA API, no i8mm | Whole-engine runs only inside Docker Desktop's Linux VM (10 vCPU / 32 GB), which publishes no NUMA topology and no cache sizes (see `13-numa-less-hosts.md`). Native macOS runs the kernel portability suites only. |

## Whole-model CPU serving on GB10 (Release, 20 cores, `-c 8192`, greedy, seed 42)

| Model (GGUF) | Mode | Prefill tok/s | Decode tok/s | Notes |
|---|---|---|---|---|
| Qwen3-0.6B Q8_0 | — | 330 | 93.5 | before spin policy |
| Qwen2.5-0.5B q4_0 / q4_k_m | — | 510 / 384 | 144 / 103 | before spin policy |
| Qwen3.6-27B Q4_K_S (unsloth MTP repack) | MTP off | 10.3 | 4.61 | after workspace + spin fixes, 10/10 runs |
| Qwen3.6-27B Q4_K_S (unsloth MTP repack) | MTP dynamic | 10.6 | 4.23 | 52-65% acceptance; slower than off (idea 14) |
| Qwen3.6-35B-A3B UD-IQ3_S (MTP repack) | MTP off, n=256 | 25.4 | 14.34 | |
| Qwen3.6-35B-A3B UD-IQ3_S (MTP repack) | MTP dynamic, n=256 | 24.8 | 18.5-18.7 | 99% acceptance, depth 7 |

- Greedy text is byte-identical between MTP off and MTP on for both Qwen3.6
  models.
- HTTP chat with a RAM-tier prefix hit works (35B: 18 of 18 prompt tokens
  matched on the repeat request).
- Peak worker RSS (35B, `VmHWM`): 35.9 GiB.

## Where 35B CPU decode time goes (before `408ff90e5`)

Per-stage CPU timing, 35B-A3B, MTP off, 765 decode steps, 74.9 ms per step:

| Stage | Per call | Total | Share |
|---|---|---|---|
| MoE routed experts (`MOE_EXPERT_FFN`) | 803 us | 24.6 s | 43% |
| Fused router + routed gate/up bundle (`router_q8_reused_projection_gemv`, 16 x 512x2048) | 668 us | 20.4 s | 35% |
| GDN projection | 440 us | 10.1 s | 18% |
| MoE shared expert | 161 us | 4.9 s | |
| LM head (248K vocab) | 5.7 ms per token | 4.3 s | |
| GDN recurrence | 120 us | 2.8 s | |
| Full-attention fused QKV | 341 us | 2.6 s | |
| Attention | 37 us | | |

`408ff90e5` pools the fused bundle's members into one dynamic workshare (Mac
VM: 16x512x2048 at M=1 runs about 190-230 us against 770-1900 us
member-by-member). Its end-to-end GB10 effect was still being measured when
this file was written.

## MTP verifier economics on CPU

- 35B dynamic MTP before the spin fix: verifier forward 326 ms per step for
  about 4.7 rows (about 69 ms per row against 75 ms for one decode step). That
  was almost no amortization, because OpenMP worker wake-up dominated (idea
  13).
- After `GOMP_SPINCOUNT=3000000`: 35B MTP dynamic is 18.7 tok/s against 14.3
  MTP off (+30%).
- 27B dense, Mac VM, Q4_K 17408x5120 at 10 threads: M=1 takes 1.10 ms and
  M=4 takes 2.56 ms (2.3x decode), against 6.23 ms for four serial rows. At
  52% acceptance this cannot beat MTP off (idea 14).

## OpenMP idle-spin sweep (35B-MTP dynamic, n=256)

| `GOMP_SPINCOUNT` | Decode tok/s | Idle server cores busy |
|---|---|---|
| 300,000 (libgomp default) | 13.23 | 0 |
| 3,000,000 (installed) | 18.73 | 0 |
| 30,000,000 | 18.80 | — |
| infinite | 18.67 | 19 of 20 |

## Kernel thread scaling

- GB10 Q4_K N=2048 K=4096 (Integration build): decode at 20 cores with
  dynamic tiles is 44.7 us (27x over one thread); static split was 78 us;
  X925-only is 52.7 us. Prefill M=64 is about 1450 us at 20 cores.
- M1 Max VM, all 12 formats, 1..10 threads, every point byte-authenticated
  (480 points): prefill M=64 scales 8.2-8.75x at 10 threads; grouped M=4/15
  scales 5.7-8.1x; decode M=1 saturates around 8 threads (Q8_K regresses to
  2.96x at 10).

## Exact overlays (`144ece037`)

- 2,519 of 10,920 GB10 keys at 20 workers for Q4_K, Q5_K, Q6_K, Q8_0, IQ2_S,
  IQ3_S and IQ4_XS.
- Each format has two independent replicas, and every candidate is
  byte-authenticated.
- The median between-replica difference is about 3% per format. The worst
  outliers (13-23x) are all ultra-fast tiny shapes where timer noise
  dominates.
- On selected keys the median ratio is about 0.74-0.81 of the total policy.
- The end-to-end serving A/B (baseline, plus overlays, plus pooled bundle) was
  pending when this file was written.

## Updated serving results (final A/B/C, verified, n=256 greedy, 20 cores)

Supersedes the 35B rows in the table above and the earlier per-commit notes for
the commits named here. Decode tok/s, mean±sd; prefill shown as range (short
bench prompt):

| Run | Commit | 35B MTP dyn | 35B no-MTP | 35B prefill |
|---|---|---|---|---|
| A | `878a33fea` baseline | 18.18±0.40 | 14.14 | 24.0–24.7 |
| B | `144ece037` +2,519-key overlay table | 17.45±0.21 | 13.27 | 22.6–22.8 |
| C | `268cd81de` + pooled bundle (still carries B's overlays) | 27.19±0.18 | 20.48 | 37.1–37.8 |

- 27B dense: at C MTP dyn 4.24–4.25 vs no-MTP 4.49–4.62 (−7…−8%) — see idea 14
  for the depth-controller finding. At A: MTP 4.20 vs no-MTP 4.51.
- C true no-MTP stage timings: `MOE_EXPERT_FFN` **274 µs/call**, fused router
  bundle **176 µs/call** (16 projections, K=2048).
- B vs A: overlays regress end to end (~−4% MTP, −6% no-MTP, −7% prefill) —
  see idea 12. C vs B: pooled bundle gains +56% MTP, +54% no-MTP, +65%
  prefill. MTP gain over no-MTP at C: **+32.8%**.

## Overlay removal A/B (head, 2026-11-08; evidence K14-OVERLAY-AB-HEAD)

X `15f87f755` (with the 2,519-key table) vs Y `d397154da` (table emptied),
alternating 2×2, mode-asserted: removing the table

- 35B MTP decode: 27.35±0.06 → **28.51±0.29** (+4.2%);
- 35B no-MTP: 20.54 → **20.95** (+2.0%);
- 35B prefill: **+9–11%**;
- 27B: unchanged; text byte-identical.

The exact-overlay table is **retired** in `264970e14` (empty table; resolver,
collector and generator kept). **New head reference: 35B MTP dynamic ≈ 28.5
tok/s, no-MTP ≈ 21.0, 27B no-MTP 4.64.**

- See `17-gb10-decode-bandwidth-gap.md` for the bandwidth-gap analysis.

## GB10 GPU reference (Qwen3.8-Flash-Next, all-CUDA:0, sm_121, Release)

Engine `arm-validation/0cdcecc3c-spills` = `bef53edb6`; Qwen3.8-Flash-Next-
UD-Q4_K_XL (4-shard GGUF, ~111 GB, 177B MoE, qwen4exp, no MTP layers);
greedy, single stream, graph-captured bucket pipeline, 102 GB resident:

| Prompt | Prefill tok/s | Decode tok/s |
|---|---|---|
| 120 | 187.5 | — |
| 2160 | 318.6 | 22.82 (~115 GB/s effective weight reads) |
| 7200 (@ -c 8192) | 312.3 (23.06 s) | 23.11 |

- **Prefill plateaus ~315 tok/s** (120 → 2160 → 7200), so it is steady-state
  compute-bound, not a length/startup artifact; decode is stable ~23 tok/s.
- Observations: GEMM dominates (prefill bucket ~198 ms/512);
  `MOE_EXPERT_FFN` 48 samples/layer; `GDN_PROJECTION` significant.
- GPU-side reference only — used as the GB10 ceiling context for idea 17's
  CPU bandwidth comparison and the llama.cpp cross-engine check (CPU baseline
  still open).

## Progress ledger (periodic perf checkpoints — operator directive 2026-11-11)

Cadence: after each Main SETTLED/collection merge AND on a ~3-day recurring
check, re-run the canonical benchmark set on GB10 (35B-A3B UD-IQ3_S MTP
off + dynamic, 20T, greedy seed 42; 27B Q4_K_S MTP off/dynamic) and append
a dated entry here: measured tok/s, delta vs last entry, and the remaining
opportunity ranking. Hosts booked through arm-sparks; M1 = low priority.

### Entry 0 — checkpoint baseline (2026-11-11)

Landed/measured so far:
- **35B MTP dynamic 14.34 → 18.5-18.7 tok/s (+30%)** — spin policy
  (GOMP_SPINCOUNT=3000000, idea 13) + workshare pooling `408ff90e5`
  (fused bundle M=1: 190-230 us vs 770-1900 us member-by-member, Mac VM).
- **I8MM engine +1.3-1.7%** (kernel 1.01-1.19x, idea 6, tip cd55f6b27).
- **Single-rank binding A725-first → fastest-first** (idea 26; residual
  measurement rides idea-16 Phase A).
- **CUDA sm_121 spill elimination in-tip** (idea 21 arc: 176-commit SETTLED
  e8866ec04; 10 native cells 11/11, CB kernels zero spills) — unblocks GB10
  GPU + UMA validation work.
- **Bundle-1 (ideas 7+25: hot-path alloc cleanup + prefill bucket
  overhead)** proven on 28ddbe253 (BOUND), awaiting combined FF.
- Idea 12 (cache tiling) RETIRED — in-model A/B regression (methodology:
  idea 20). Idea 18 (GDN grouped verifier) CONFIRMED harmful (<1.0) — keep
  serial.

Remaining opportunity ranking (by expected impact):
1. **Idea 16** — MoE routed experts 43% + router bundle 35% + GDN 18% +
   LM head of decode time; Phase A (measurement) queued next window.
2. **Idea 17/24** — GB10 decode bandwidth gap (llama.cpp CUDA 28.8 vs
   Llaminar CUDA 22.8 tok/s; CPU 14.34) + GPU prefill 3.3x gap (~80%
   bucket-pipeline overhead; blocked on nsys tooling).
3. **Idea 8** — prepared-weight layout (dead-byte elimination 10.0%/5.56%,
   compact encodings IQ3_S 9.5→3.44 bpw); Phase A GO.
4. **Idea 14 B wave** — dense MTP verifier economics on the new base
   (MTP 5.45-5.55 vs 5.28 no-MTP: wins now; depth tuning open).
5. **Half-perf cell (FMLAL)** — admission-held, focused review queued.
6. Smaller: idea 3 (metadata hoist), 4 (column panel reuse), 11
   (validation hoist), 22 (FA2 tournament tail p95 8.9%), 23 (HC auto-plan),
   28 (IQ4_XS M=1 regression).
