# A-2c scope: PMA-backed serving capacity + knob disambiguation + HTTP pool unification (2026-11-20)

Worktree: `~/repo/llaminar-cb-a2c` (branch `feat/cb-a2c-pma`, base 19ecebbfb; the A-3 staged diff under
review in `~/repo/llaminar-cb-develop-draft` is untouched).

## What A-2c delivers (preconditions for lifting `serving-capacity-lease-lifetime`)

The deny rule (src/v2/config/ConfigValidator.cpp, id `serving-capacity-lease-lifetime`) refuses authored
§9.4 capacity until BOTH the typed detached-response-lease lifetime is installed (A-3: typed
ResponseLease/ExecutionLease/PublicationLease + SerializedInferenceExecutor — staged, pending
acceptance) AND planning/PMA backing + boundary/preflight regressions exist. A-2c = the backing:

1. **Unified PMA-backed capacity**: `resolveServingCapacity` (ServingCapacityQualificationOwner /
   ServingCapacityRequest::authored) currently computes R/Q and the installed envelope without a
   PhysicalMemoryAuthority charge. Bind the resolved serving pool (resident/response-lease/queue
   admission buffers, SSE/delivery staging) into the typed PhysicalMemoryBOM as a new
   storage-oriented `PhysicalMemoryOwner` (e.g. `ServingResponsePool`) with estimator inputs, so
   capacity is admitted by the same authority as every other allocation (maxim: no parallel ledger).
2. **§10.1 ownership receipts / transport-completion wiring**: detached delivery publication uses
   the typed PublicationLease; the transport completion edge must consume the same receipts the
   executor publishes (server queue behavior), with focus regressions.
3. **Knob disambiguation**: `--batch-size` (decode batch), `--mtp-max-request-batch` (MTP row
   batch), and benchmark fixed-cohort flags are distinct from serving capacity; ensure the help /
   validation / BenchmarkRunner wiring cannot be read as serving-capacity controls.
4. **HTTP pool unification**: `SerializedInferenceExecutor::httpWorkerCount()` (ServerMode.cpp:440)
   currently `capacity + 2`; unify the pool width rules with the typed capacity contract and bound
   it through the same qualified envelope (no second pool accounting).
5. **Regressions**: real-boundary + preflight entries for the deny-rule removal and the PMA-BOM
   serving charge, mirroring the A-3 preflight registration pattern
   (V2_Integration_Serving*Regressions with LABELS ProductionTestPreflight/DeviceFree).

## Sequencing

- A-2c is developed here staged-only; it is NOT committed until A-3 is accepted and committed
  (the deny-rule lift text references the A-3 typed lifetime).
- After A-3 acceptance: commit A-3 on feat/continuous-batching (PR #67), then rebase/cherry-pick
  A-2c onto it, run the same focused + full gates in the persistent harness, package for the
  independent reviewer, then lift the deny rule only after acceptance.
- Real-host/GPU gates and G1-G9 remain open blockers (Docker-on-Mac scope, disclosed).

## Evidence chain (persist harness)

Same loop as A-3: P/repo synthetic step commits, step.sh focused/full gates, immutable packet
(staged diff + SCOPE + SHA256SUMS) under /private/tmp/llaminar-cb-phaseA2c.*, Intercom to
batch-inference-reviewer.
