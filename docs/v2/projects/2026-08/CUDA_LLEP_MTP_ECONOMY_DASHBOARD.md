# CUDA LLEP + MTP Economy Dashboard

Last updated: 2026-08-04

## Promotion Goal

For Qwen 3.6 35B MoE, promote the production stochastic CUDA2 LLEP lane only
when it satisfies all of the following in a Release build:

- live HTTP, long-context, prefix-cache, dynamic-depth, and PerfStats gates pass;
- decode is at least the llama.cpp CUDA1 MTP depth-3 reference of 171.81 tok/s;
- prefill is at least twice the matching llama.cpp CUDA1 MTP depth-3 reference;
- homogeneous CUDA execution remains one complete captured graph with no
  intermediate host outcome bridge, segmented replay, or non-terminal D2H.

## Current Scorecard

| Gate | Current evidence | Status |
|---|---|---|
| Production stochastic correctness | CUDA2 LLEP, prefix cache, dynamic MTP depth 4..15: 23/23 canonical E2E checks passed | Green |
| Full device generation parent | One captured launch and one terminal D2H; retired per-transaction host bridge counter remains zero | Green |
| Unit regression gate | 585/585 `V2_Unit_*` tests passed in 149.50 seconds | Green |
| Decode economy | Last pre-slice CUDA2 fixed-depth result: 166.73 tok/s, or 97.0% of the 171.81 tok/s reference | Refresh required |
| Prefill economy | Last pre-slice CUDA2 result: 182.02 tok/s | Matching llama.cpp reference refresh required |
| Dynamic-depth economy | Correct production path is installed; controller hysteresis and depth-selection economy are not yet promoted | Tuning required |

Canonical correctness evidence is retained locally at:

```text
build_v2_release/e2e_server_logs/20260804_094920_Qwen3.6-35B-A3B-UD-IQ3_S_qwen36-moe-llep-prefix-mtp-stochastic-d4to15-cuda2tp-focused-switch-resident-proof-v7_tp_port19084.perfstats.json
```

## Completed Architecture Slice

- Added a typed greedy/stochastic generation mode through the inference-runner
  interfaces and graph-cache identity.
- Installed one native device generation parent for greedy and stochastic MTP.
- Kept maintenance publication, graph materialization, sampling, generation
  control, terminal ledger validation, and accounting inside captured device
  execution.
- Made the terminal ledger authoritative so stale host controller state cannot
  drive outcome diagnostics.
- Added PerfStats policy checks for the stochastic native parent and the absence
  of retired host outcome bridges.
- Rebuilt all Integration consumers after the central `DebugEnv` ABI changed;
  this removed the apparent `vector::reserve` failures caused by stale test
  executables.
- Parallelized the NativeVNNI refresh-script unit harness, reducing its wall
  time from about 275 seconds to about 36 seconds.

## Next Measurement Sequence

1. Checkpoint this green correctness slice.
2. Refresh the llama.cpp CUDA1 fixed-prompt MTP depth-3 decode and prefill
   reference with current master and the exact production benchmark inputs.
3. Measure Llaminar CUDA2 LLEP at fixed depth 3 to isolate graph and collective
   economy from dynamic-controller behavior.
4. Profile the complete fixed-depth run with Nsight Systems and selected kernels
   with Nsight Compute; attribute graph nodes, collectives, launch gaps,
   occupancy, registers, spills, and memory throughput.
5. Tune communication and kernels until fixed depth clears both promotion
   thresholds, then tune dynamic-depth hysteresis against fixed-depth winners.
6. Rerun the canonical stochastic E2E/PerfStats gate after every promoted
   economy change.

## Progress Estimate

- Active CUDA2 LLEP + MTP promotion goal: **75%**. Correctness and target
  architecture are green; measured decode remains 3.0% short on the last
  comparable run, and the prefill reference still needs a clean refresh.
- Fully device-resident ExpertParallel MTP: **90%**. The CUDA2 LLEP production
  lane is proven device-owned; economy promotion and a symmetric refreshed ROCm
  proof remain.
- Fully device-resident LocalTP MTP: **85%**. Shared device-generation ownership
  is installed, while the current sprint still owes a refreshed dense LocalTP
  production E2E/performance proof after the common changes.
