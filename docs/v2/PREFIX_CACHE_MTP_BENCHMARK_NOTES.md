# Prefix Cache And MTP Benchmark Notes

Current Phase 14 scoreboard for CUDA/ROCm Qwen3.6 work. Raw run history stays in
`benchmark_results/` and `/tmp/llaminar-mtp-bench`.

## Headline Matrix

| Scope | Device | Model | Mode | Prefill tok/s | Decode tok/s | Status |
|---|---|---|---|---:|---:|---|
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | no MTP | 707.92 | 41.73 | current baseline |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | fixed d1 MTP, sequential verifier | 598.97 | 38.33 | safe, speed-negative |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | fixed d3 MTP, sequential verifier | 609.73 | 38.19 | 84.95% acceptance |
| Dense default, 595p/128d | CUDA | Qwen3.6 27B Q4_K_S | dynamic MTP | 610.88 | 38.52 | holds depth 1; below baseline |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | no MTP | 2707.70 | 119.91 | current baseline |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | fixed d1 MTP | 1946.82 | 148.50 | speed-positive |
| MoE default, 595p/128d | CUDA | Qwen3.6 35B A3B | dynamic MTP | 1943.69 | 145.36 | speed-positive |
| Dense long `qbf`, `-c64 -n48` | ROCm | Qwen3.6 27B Q4_K_S | fixed d3 MTP | n/a | 54.78 | 1.77x over 30.93 baseline |
| Dense default, 595p/128d | ROCm | Qwen3.6 27B Q4_K_S | best MTP | n/a | 46.74 | depth-sensitive |
| MoE default, 595p/64d | ROCm | Qwen3.6 35B A3B | fixed d1 MTP | n/a | 42.04 | 2.13x ratchet |
| TP / PP / EP overlay | Mixed | Dense and MoE | MTP | Pending | Pending | after single-device lanes |

CUDA dense artifacts:
`/tmp/llaminar-mtp-bench/dense-cuda-phase138-{nomtp,shared-d3}-default.json`.
CUDA MoE artifact:
`benchmark_results/cuda_moe_mtp/20260605T070628Z-iq4nl-word-decode`.

## llama.cpp CUDA Anchors

`ggml-org/llama.cpp@6ddc943`. MTP-off uses `llama-bench`; MTP uses generated
`mtp-*` sidecars with `llama-cli --single-turn`.

| Lane | Model | Prefill tok/s | Decode tok/s | Acceptance |
|---|---|---:|---:|---:|
| `llama-bench` no-MTP | Dense 27B | 1161.03 | 41.83 | n/a |
| `llama-cli` MTP d1 | Dense 27B | 608.9 | 54.9 | 93.75% |
| `llama-cli` MTP d3 | Dense 27B | 609.0 | 52.5 | 72.41% |
| `llama-bench` no-MTP | MoE 35B A3B | 2413.54 | 118.26 | n/a |
| `llama-cli` MTP d1 | MoE 35B A3B | 1032.9 | 142.0 | 96.88% |
| `llama-cli` MTP d3 | MoE 35B A3B | 1064.5 | 132.8 | 75.44% |

## Current Findings

- Dense CUDA MTP remains speed-negative because accepted speculative tokens still
  pay full decode-equivalent verifier forwards. Stats show
  `decode_equivalent_catchup_forward_one` at 9272.9 ms / 383 forwards, versus
  98.6 ms for shifted commits and 11.8 ms for sampling.
- ROCm and CUDA now share `MTPDecodeCatchup` as the safe oracle. It is correct,
  but it is not the final speed path.
- Phase 13.8 Candidate A, sidecar-chain verifier state, is retired by focused
  ROCm/generic and CUDA Qwen3.6 dense negative parity tests.
- Naive Candidate B, selecting state from an all-position verifier, is retired.
  CUDA M=2 diagnostics drift before the GDN copy boundary, and ROCm/generic M=4
  state parity agrees. New commit-replay negative tests pin this.
- Phase 13.8 is now a vLLM-style spec-decode transaction port rather than a
  free-form shortcut search. The target is shared valid-count / accepted-count /
  rejected-suffix metadata, accepted-count-aware GDN/short-conv state commits,
  shifted MTP-row commit from accepted target hidden rows, and atomic request
  publication. The current Llaminar greedy candidate starts after the live
  terminal condition, so draft depth `D` forwards `D` target rows; the last row
  is a bonus ready-token row only when all drafts are accepted.
- First shared metadata slice is green: `MTPSpecDecodeTransaction` is in core,
  `OrchestrationRunner` validates decode-equivalent catch-up results against it
  before commit, and `mtp.spec_decode_transaction_metadata` counters describe
  accepted/rejected verifier rows. `MTPSpecDecodeMetadata` now declares and
  binds the graph-facing int32 workspace buffers through `DeviceWorkspaceManager`,
  `DeviceGraphOrchestrator` requests them as runner-owned extra workspace when
  MTP is enabled on GPU, and GPU metadata upload hard-fails without an explicit
  non-null stream. The metadata now carries committed-state row/index and
  bonus-ready-token row/index fields too, pinning that an all-accepted bonus
  row may feed the next drafter condition but must not become live
  GDN/short-conv state. Focused unit coverage passed for
  `V2_Unit_MTPSpecDecodeMetadata`, `V2_Unit_MTPSpecDecodeTransaction`,
  `V2_Unit_MTPDecodeCatchup`, and `V2_Unit_PrefillDecodeTransition`.
- The named `vllm_style_spec_decode` hook is now selectable for Phase 13.8
  development. It still hard-fails outside
  `LLAMINAR_MTP_PHASE138_EQUIVALENCE_CHECK=1`, but under the equivalence harness
  `DeviceGraphOrchestrator` now runs the live target-verifier candidate:
  first shifted-row commit, uniform all-position verifier forward, device row
  sampling, shared metadata build, accepted shifted-row commit, metadata-driven
  GDN/short-conv restore, and correction-suffix replay. It remains
  non-promoted until real CUDA/ROCm equivalence/parity/benchmark evidence lands.
  First dense evidence landed on 2026-06-07: CUDA and ROCm
  `MTPGreedyMatchesPyTorchDecodeTokens` pass under the candidate/equivalence
  env, their perf JSON contains `phase138_vllm_style_spec_decode_runs` and
  `phase138_spec_decode_equivalence_matches`, and CUDA/ROCm
  `PrefixCacheMTPRestore` also pass under the same env. Benchmark numbers should
  not be quoted yet because the equivalence harness still runs the stepwise
  oracle after the candidate.
- A reject-after-prefix ROCm depth-3 candidate failure is fixed: runtime state
  matched the oracle, but the candidate sampled the correction-suffix ready token
  from stale normal logits. The candidate now re-enables all-position logits for
  suffix replay and samples row 0 from the explicit suffix output. Focused
  validation passed under the candidate/equivalence env for ROCm depth-3 greedy,
  CUDA depth-3 benchmark-prompt greedy, CUDA regular greedy, CUDA/ROCm
  prefix+MTP restore, and the Phase 13.8 unit guard set. This is now pinned by
  dedicated Phase138 CTest regressions for the ROCm depth-3 reject lane and the
  CUDA depth-3 benchmark-prompt lane.
- Device-metadata state publication is now green for the first backend slice:
  CUDA and ROCm short-conv/GDN kernels can restore live state from verifier
  snapshot rows selected by graph-facing `committed_state_rows[request_index]`.
  CUDA regression coverage captures metadata restore plus continuation into a
  CUDA graph; ROCm regression coverage proves the same explicit-stream path.
  The promoted hook is still hard-failed until full accepted-count kernel
  commit/replay parity and runner integration are done.
- `DeviceGraphOrchestrator` now has a runner-facing restore primitive that
  uploads an `MTPSpecDecodeMetadataBatch` to the bound workspace and publishes
  all local GDN state from `committed_state_rows` before truncating KV/bookkeeping.
  This is still scaffolding: `vllm_style_spec_decode` remains hard-failed until
  the verifier graph builds the batch live and commit-replay parity is green.
- The oracle-to-transaction bridge is now shared:
  `buildMTPSpecDecodeMetadataBatchFromGreedyCatchup()` converts
  `shared_stepwise` results into the graph-facing metadata batch and rejects
  accepted-prefix drift. `OrchestrationRunner` uses this bridge for its current
  transaction validation, so the live verifier path has a single contract to
  match.
- Target-verifier row semantics are now explicit. Accept-all verifier rows can
  publish the terminal accepted-input state and carry a separate bonus ready
  token. Reject paths are different: the sampled correction is an emitted output
  token, but it has not been forwarded by the verifier graph. Metadata now
  carries `target_verifier_state_commit_count` separately from committed output
  count so the optimized path must publish only the accepted input prefix and
  replay the correction suffix before claiming decode-equivalent state.
- The first equivalence harness primitive is in place:
  `compareMTPDecodeCatchupGreedyResults()` allows fewer candidate main forwards
  but requires identical committed tokens, ready/reject semantics, and shifted
  MTP commit count before any optimized path can be trusted.
- State equivalence is also explicit now:
  `compareMTPRuntimeStateSnapshots()` checks decode position, KV/MTP token
  counts, terminal state availability, and GDN state hashes for oracle versus
  candidate transaction runs.
- `LLAMINAR_MTP_PHASE138_EQUIVALENCE_CHECK=1` now enables a non-promoted
  runner harness: run the selected optimized candidate, restore the verifier
  base, run `shared_stepwise`, compare result plus runtime state, and emit
  `mtp.phase138_spec_decode_equivalence_matches` only on a clean match.

## Retained Actions

- CUDA/ROCm dense: consume the graph-facing spec-decode metadata buffers from
  the live target-verifier transaction, then promote a named
  `vllm_style_spec_decode` hook only after commit-replay equivalence, PyTorch
  parity, and benchmarks.
  Do not re-enable raw all-position verifier-row shortcuts.
- CUDA MoE: keep the 148.5 tok/s ratchet and extend long-prompt/controller
  evidence without weakening parity.
- ROCm: keep the 54.78 tok/s dense ratchet while aligning the state contract
  with CUDA.
- Shared: keep all-codebook M=2/3/4 GEMV equivalence and parity tests in the
  normal suites.
