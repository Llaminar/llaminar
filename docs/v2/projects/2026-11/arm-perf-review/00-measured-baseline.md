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
