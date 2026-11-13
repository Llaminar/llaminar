# QFN CUDA2 + MI50×4 OpenCode qualification

The complete requested run passes on the Release binary: **10/10 app phases,
98 requests, 107 tool calls, zero tool errors, and 56 generated-app tests**.
Independent HTTP, Unicode/emoji, SQLite persistence, validation, pagination and
concurrency checks pass. The app session took **1,255.27 seconds (20.92 minutes)**;
the complete owned lifetime, including startup and shutdown, took 1,491.40 seconds.

## Actual automatic selection

The original two-rank Auto request selected CUDA0/1 on rank0/NUMA0 as the TP2
continuation and ROCm3/4/6/9 on rank1/NUMA1 as four MI50 gfx906 expert followers.
Both domains use **Apportioned whole-expert ownership**, with dynamic expert
residency rebalancing. There is no CPU expert tier. The selected plan was applied
unchanged; no visibility masks or hand-selected replacement topology were used.

| Setting | Qualified value |
|---|---|
| Model | Qwen3.8-Flash-Next UD-Q4_K_XL, shared Q8_0 MTP predictor |
| Context / output limit | 262,144 / 32,768 tokens |
| MTP | Dynamic 1–15, retained capacity 15, speculative sampling in the app |
| Activation / KV precision | FP32 / FP16 |
| Ngrams | RAM backing; ConsumerLocal GPU row caches, 256 MiB admission ceiling per consumer |
| Prefix cache | 16 GiB RAM per continuation participant; 32 GiB shared disk |
| MPI / CPU affinity | Two original NUMA-owned ranks, 28 physical cores each |
| Client | OpenCode 1.18.34-llaminar.1; original ten-phase webapp prompts and tools |

Auto planning took **30.41 seconds**, still above the five-second target. The
selected MI50 group spans two switches. Whole-expert EP uses compact dispatch
and return traffic rather than follower dense-TP allreduces. Current ranking
prices directed mapped host-link service; same-switch preference breaks exact
cost ties. This run proves the selected configuration works, not that its group
contention or migration costs are optimal. Further planner tuning was deferred.

## Admission defect repaired

Original smoke `q690` failed while materializing the restored-prefix MTP bridge.
Its storage and `RestoredPrefixBridge` inventory entry were already admitted, but
role dispatch recognized that transaction only as `MainInference`. Production
serving construction uses `MTPCondition`.

`MoEOverlayActivationBatchInventory::require()` now recognizes that exact role
and transaction against the existing bridge source. Row, layer, sparse-query,
family and sidecar guards remain in force. No inventory entry, memory allowance,
numerical policy or kernel was added or changed.

The regression uses production inventory declarations with retained depths 1,
3 and 15, fixed/dynamic policies, enabled and disabled-with-retained-capacity
states, both supported roles, execution/materialization and both query programs.
Wrong role, phase, rows, layer, program, and removal of the bridge source fail.
The existing inventory sweep also covers absence when MTP capacity is not retained.

`q693` reproduces the original failure before the production change. Release
`q695` and Integration `q696` builds pass. `q697` passes all **571 MemoryPlanner
tests** and four focused integration aliases, including the explicit new
`V2_Integration_MappedBatchRestoredPrefixConditionAdmission` preflight entry.
Fresh model-maximum admission `q699` and actual inference smoke `q700` pass.
The full production preflight was not repeated for this focused iteration.

## Performance and application evidence

| Measurement | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| Matched short Release benchmark, median of three | 766.34 | 120.00 |
| Full stochastic coding session, summed tokens/time | 345.12 uncached | 76.64 |

The short benchmark uses 512 prompt tokens and 256 output tokens (255 decoded
after prefill), one warmup and three measured requests, with prefix cache off.
Dynamic MTP is active; this benchmark uses greedy verification and does not
independently establish stochastic depth transitions. It is a local throughput
measurement, not an external-engine comparison or numerical-parity certificate.

The app generated **50,592 tokens** and exercised stochastic MTP with **18,370
verifier runs and 366 depth updates**. Accepted/drafted tokens were 70.78%;
stochastic accept-test success was 84.97%. Prefix-token reuse was **98.035%**
(3,767,477 of 3,842,975 prompt tokens), average TTFT **2.814 seconds**. There were
no failed/disconnected requests and no compactions. Native output auditing
joins all 98 responses and 107 calls without changes. Protocol, tool coverage,
strict task success and the less-than-5% stress policy all pass.

Runtime checks prove the declared heterogeneous graph boundary, captured
participants, dynamic expert maintenance and stochastic MTP. No forbidden
transfer operations were observed. The resource gate passes across 245 samples:
anonymous-plus-swap peaks were 41,434,890,240 bytes on rank0 and 2,449,563,648
bytes on rank1; no swap was used. CUDA peaks were 21.80/21.15 GB and the four
ROCm peaks 24.62/24.62/24.45/24.45 GB. Both prefix tiers stayed within their
payload budgets while exercising promotions and eviction. All 1,253 stats
polls passed, with a maximum observed HTTP duration of 4.62 ms.

Shutdown persisted **2,080/2,080 RAM entries**: 1,940 payload writes totaling
33,282,340,224 bytes and 140 existing-backing reuses. Native retirement was
normal, with no forced termination or surviving children. Driver diagnostics
reported zero findings. The metadata-only archive audit passed with zero
orphan payloads; cleanup removed 2,332 regular files / 34,216,750,001 logical
bytes, leaving no archive entries. Audit and cleanup read no payload bytes.

## Retained evidence

Local artifacts are under `/tmp/qfn-cuda2-mi50x4-ram-opencode-20261113/`:

- Original Auto request, selected plan, planning log and retirement receipts.
- `artifact-join-bridge-fixed.json` and `bridge-admission-fix.patch` bind the
  rebuilt source to fresh context, smoke and app observations.
- `benchmark-q701/result.json` retains samples and measurement limitations.
- `app-q702/result.json`, `app-validation.json`, `runtime-contract.json`,
  `native-audit/result.json`, `resource-growth.json`, `driver-report.json`,
  `archive-audit.json`, `archive-reclamation.json` and retirement receipts pass.

Command/build/test receipts are under `/tmp/qfn-full-preflight-after-cpu-next/`.
Original red evidence remains unchanged. This qualification covers this local
hardware, configuration and app session; it does not publish an image or replace
the broader numerical, release or topology matrix gates.
