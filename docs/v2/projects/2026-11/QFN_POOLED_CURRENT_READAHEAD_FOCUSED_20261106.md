# QFN current-demand read-ahead: focused MTP and ordinary generation

This checkpoint extends the existing shared-pool path. It does **not** qualify
the combined participant-local cache, future speculative read-ahead, numerical
HF parity, or Release throughput. The separate hierarchy implementation is
tracked in [the local hot-cache plan](QFN_LOCAL_HOT_CACHE_HIERARCHY_PLAN.md).

## Production changes

The early current-demand lookup now uses the original grouped verifier's
move-only prior-history capability and exact primitive/GEMM row-count identity.
It takes the snapshot once, hashes against that snapshot, and retains the same
prior at the later PLE consumer. Scalar MTP condition invocations retain their
ordinary accepted-history policy. A foreign history/count/geometry is rejected
before graph mutation.

Ordinary hosted generation now authenticates its actual retained forward and
owned sampler when binding replicated PLE. This is a semantic hosted fragment,
so looking only for a composable native forward child incorrectly rejected the
valid path. Native-composable ownership remains unchanged; equal shape or a
foreign executable cannot substitute for the original sampler, signature,
workspace and parent.

CTest's explicit candidate environment is preserved literally through local
and MPI launch discovery. The owning focused unit/integration regressions are
registered in production preflight. This iteration ran only the focused gates,
as requested.

## Validation

Build: coherent Integration build rooted at
`/tmp/qfn-fixed1-projection-build-r710/build`, using the existing selected
ROCm SDK, NCCL/RCCL dependencies and normal spill policy. No GPU arithmetic
kernel changed in this checkpoint.

| Check | Result | Scope |
|---|---|---|
| Eight focused host registrations | Pass | Verifier/condition row and prior identity, actual owner authority, ordinary retained identity, test inventory/environment projection |
| ROCm TP2 early-current fixed1 source admission | Pass, 152.4 s | Constructed and retired original/replacement graph families; 50 observations in each family |
| ROCm TP2 early-current fixed1 native exact | Pass, 122.6 s | 1,208 serial/grouped byte comparisons, four prefix-history witnesses |
| ROCm TP2 shared-owner ordinary generation | Pass, 112.1 s | Three-token first request proves initial decode and replay; two-token second request proves reset/reuse of original owners |

The early-current cases use two MI50s, 128 GPU experts and a CPU capacity tier,
SSD-backed ngrams, and 256 MiB requested shared-cache caps on each GPU. The
ordinary case uses two MI50s, eight GPU experts and a CPU capacity tier,
RAM-backed ngrams, coordinator ROCm:0 as a pure reader and ROCm:1 as the declared
zero-slot shared owner. All runs use the existing Qwen3.8-Flash-Next UD-Q4_K_XL
four-shard GGUF and Q8_0 shared MTP predictor identity. Ordinary execution has
MTP off; fixed1 retains capacity three.

The native runs have no skips or driver errors. Both actual GPU context
generations retired to quiescent state with canonical allocation bytes zero.
Owned prefix-cache artifacts were removed after retirement; CSVs and audit
records remain in the result directory.

## Test defects found during qualification

- The new source observer incorrectly exported a source-family definition from
  a query-only reader. It now observes that reader's actual retained decode
  graph, without claiming source ownership or reader prefill evidence.
- The initial ordinary test treated `GenerationResult::is_complete` as budget
  exhaustion. It records model stop instead; the test requires no error and
  the full requested token count, while still rejecting early EOS.
- A two-token first request proves only one decode launch after the prefill
  sample. The graph gate correctly refused to call this replay. The first
  request now asks for three tokens; the graph-certification predicate remains
  unchanged.

## Local evidence

Root: `/tmp/qfn-goal-continuation-r729`.

- `current-read-ahead-focused-host/summary.json` plus
  `current-authority-focused-host/summary.json`: seven first-pass registrations
  and the eighth after building its initially missing executable.
- `source-rocm-tp2-early-current-fixed1-observer-fixed/summary.json` and
  `source-observer-audit.json`.
- `exact-rocm-tp2-early-current-fixed1/artifact-audit.json`, `case-evidence/`,
  `resources.jsonl`, driver diagnostics and `cleanup.json`.
- `ordinary-rocm-tp2-owner1-ram-replay-budget3/artifact-audit.json`,
  `case-evidence/`, `resources.jsonl`, driver diagnostics and `cleanup.json`.

The initial failed test results remain separately named; none was relabeled as
passing evidence. Shared-only remains the current control. Combined-cache
implementation and matched performance qualification are still pending.
