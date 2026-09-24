# Prefix admission race found by container certification

2026-09-08. The full AVX512 container gate passed 643 Unit tests and 118
production-preflight tests, then failed the first Qwen35B Dynamic/Random
CUDA1+ROCm1+CPU2 numerical cell during convergence training. The failing
request pair reported cold admission/completion epoch 2 and restore
admission/completion epoch 2, but the replay missed the prefix archive.

## Root invariant

`DeviceGraphOrchestrator::lookupPrefix()` built a fingerprint using one
movement observation, then sampled movement again for the lookup's epoch.
`makePrefixParticipantLookup()` also accepted an epoch override; request,
rank and global orchestration supplied yet another live observation. A
background publication between these reads could label an epoch-1 key as an
epoch-2 admission. Harvest correctly rejected the stale fingerprint, while
the request summary incorrectly claimed no movement crossed the request.

```mermaid
sequenceDiagram
    participant C as Cache / device orchestrator
    participant M as Movement authority
    participant R as Request coordinator
    C->>M: Observe epoch 1
    C->>C: Construct key for epoch 1
    M->>M: Publish epoch 2 asynchronously
    Note over C,R: Old flow resampled epoch 2 and attached it to key 1
    C->>R: Fixed flow: key 1 + original epoch 1
    R->>R: Coordinate immutable lookup; no epoch override
    R->>C: Execute and harvest using admitted key 1
    C->>M: Observe current epoch 2
    C->>C: Reject stale archive write; preserve successful inference
    C->>R: Completion epoch 2 exposes the crossed interval
    R->>C: Next request admits key 2; cold prefill is explained
```

The fix binds the fingerprint and epoch in `PrefixCacheFingerprintResult`,
constructed from one authority observation. The cache retains that descriptor
instead of a key plus a later sample. Coordination has no epoch-override
argument. This is immutable request provenance, not another placement ledger.
No transfer synchronization, blocked maintenance, graph recapture, numerical
tolerance change, or prefix-test exemption is required.

## Evidence and regression scope

- Container first-failure evidence: `parity-results/ci-container-proof-14/avx512/`.
- `MovementAfterLookupCannotRelabelItsAdmission` reproduced the incorrect
  epoch deterministically before the fix in the production request coordinator.
- `PrefixLookupRetainsEpochUsedToBuildFingerprint` injects publication inside
  the actual device orchestrator's fingerprint construction, without models
  or device work, then checks stale-harvest rejection.
- Fingerprint tests bind key invalidation to the returned epoch, preserve
  placement-compatible policy semantics, and reject a competing epoch field.
- These tests are in the complete Unit gate. Numerical matrix execution still
  proves captured inference and prefix behavior on real CPU/CUDA/ROCm devices;
  deterministic Unit interleavings do not substitute for that gate.

The initial tests did cover movement between admission and completion, but
their mock kept the live epoch equal to the lookup epoch until harvest. That
missed the earlier publication window. The new regressions explicitly cover
both independent resampling sites rather than relying on timing to hit them.

## Fixed-source verification

The focused four-suite Unit gate passed 20 repetitions (80/80 suite runs).
The full rebuilt host gate then passed 643/643 Unit and 118/118 production
preflight tests. The existing rank-coordination mock was also corrected to
give its admitted lookup epochs independently of later live epochs, so it
now rejects the same resampling error instead of expecting it.

The exact failing real-model cell subsequently passed 20/20 independent
process runs, each validating all eight required CSV artifacts (160 total).
Evidence is retained under `parity-results/ci-prefix-stability-v2/`, with one
report and fresh artifact directory per repetition. Only unchanged-build
Unit/preflight evidence was reused after the first repetition; neither model
results nor CSV artifacts were reused. The unfiltered dual-ISA container
pipeline remains the certification authority, not this targeted stress gate.
