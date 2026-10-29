# QFN CPU materialization accounting repair

2026-10-29. Size limit: 15,000 UTF-8 bytes. Scope: the live allocation seams exposed while qualifying source
admission. This is a repair plan, not a certificate or a replacement memory ledger.

The bounded strategic repair is integrated. The full Unit suite passes **789/789**.
The last complete gate recorded 3,028 passes, 29 hardware skips and two GPU
fixture failures. Both repaired fixtures now pass on all four registered backend
lanes. Expanded native admission and complete-gate qualification are recorded
below. No focused result is a substitute for the complete gate.

## Current strategic contract

One immutable CPU storage declaration supplies both physical BOM geometry and
materialization. `CPUStorageRoot` delegates exact rounding to `AlignedVector`,
preclaims through the existing PMA, and attaches the lease to the actual backing.
Native tensor adoption preserves that backing; copies require their own claims.
NUMA binding precedes first touch. Heap placement does not silently become a
page mapping. No live ledger or capacity reserve is added.

The finite integration covers prepared weights, main arena roots, compact CPU
expert families, grouped workspace, and the existing mapped CPU publication
bank. Main arena registration and CPU planning consume `GraphArenaStorageLayout`
from the production schema. `CPUTensorStorageDeclaration` supplies the same
six-format geometry for pricing and allocation. CPU serial layouts declare the
actual token-row families, both distinct canonical banks and grouped roots;
the selected admission retains them through existing runtime configuration.
Independent endpoint/controller owners are counted independently. A reduced
runtime schedule must be a valid projection of its frozen declaration.

Source review also found that a retained replicated MTP predictor can have wider
Q/K/V and FFN roots than the main TP shard. The existing shifted-predictor
ownership policy now travels with the activation geometry, including setup-wire
identity. Model-free regressions compare actual graph builders with metadata
declarations across TP1/2/4/8, MTP policies and QFN PP placement.

This repair preserves graph scheduling, kernel arithmetic, numerical precision,
topology selection and GPU allocation policy. Ordinary full-CPU MoE execution
now declares its grouped/router scratch through the actual workspace owner.
Router-only continuations declare only their router root; per-layer compact
packets and separately owned serial endpoints retain separate physical claims.
Execution replay and full qualification remain required: setup-only source
admission must not be described as certifying later execution.

## Qualification and regression boundaries

- Full Unit: **789/789 pass**, `/tmp/qfn-cpu-strategic-prereqs-r29/unit.xml`.
- Focused preflight: 66/70 passed before the last fixture corrections; all four
  failures and six affected Unit suites subsequently passed in
  `/tmp/qfn-cpu-repair-focused-r27.xml`. The earlier red receipt is preserved.
- Production startup calibration now declares allocator-rounded retained bytes
  separately from logical packed payload and actual temporary packing demand.
  Every prepared backing owns its claim; calibration no longer holds a duplicate
  aggregate persistent claim. The all-format physical storage/retirement test is
  explicitly registered in production preflight.
- Fixture repairs author real arena, KV, recurrent checkpoint and metadata
  contributions. Vocabulary gathering alone does not imply KV sharding;
  all-attention MTP still retains hidden-state metadata. Standalone graph fixtures
  explicitly receive the admitted prepared store. Production model contexts
  reject test-only admission overrides, including direct rank construction.
- Expanded native admission: **17/17 pass in 973.0 seconds**, with all 17
  evidence audits passing, clean driver/kernel-memory reports and unchanged
  binaries. Receipt: `parity-results/qwen-flash-next/source-admission-cpu-closure-r28/`.
  This adds CPU mmap/adaptive, GPU/CPU adaptive, Dynamic/random ownership and
  an all-GPU control with host-only ngram service. Retirement and prefix cleanup pass.
- Source-policy guards complement typed/runtime ownership tests. They inspect
  individual preparation calls and placement arguments, reject known unadmitted
  entrypoints and named unbound/array workspace construction, and mutation-test
  exact infrastructure exceptions. Empty or missing source trees fail. The scan
  and **33 negative-control tests pass**, including their explicit Unit and
  ProductionTestPreflight registrations. These bounded lexical guards do not
  prove arbitrary C++ allocation, alias or template semantics.
- The complete r29 receipt retains **3,028 pass / 29 skip / two failures**. Its
  GPU prepared-service fixtures omitted the required PMA issuer. Their repaired
  CPU, AVX2, CUDA and ROCm tests pass in `/tmp/qfn-cpu-prepared-service-r31.xml`.
  Descriptor capacity may contain unused alignment padding: the regression
  requires live claim = actual backing, with actual backing <= admitted capacity.
- r31 completed Unit/host/CUDA/ROCm lanes without failures. Its exclusive lane
  timed out in `MoEImmutableOwnerBinding_ROCm2`, then the host rebooted during
  the following ROCm4 case. No complete receipt exists. Preserve the interrupted
  logs under `/tmp/qfn-cpu-strategic-prereqs-r31/`; post-reboot repetitions use
  `/tmp/qfn-r31-owner-binding-reproduce-r1/`. A fresh complete gate is required
  before committing the integrated accounting/MTP/CI slice.

The native CPU materialization canary checks activation, workspace and routed
weight owners only when planned, plus a live prepared bank on the same allocator.
It never requires maximum admitted capacity to be allocated. Exact byte geometry,
aliasing, failure rollback and final-owner retirement remain per-allocation tests.

## Original defect

The original CPU TP2 source-admission run retained actual expert engines,
primary/additional prepared weights and CPU arena roots, but its PMA attestation
reported zero materialized bytes for these owner categories. For example, the
routed owner has 103,756,595,200 planned bytes and zero committed/materialized
bytes. Planned bytes are not a measurement of physical allocation. The source
admission fixture currently authenticates exact ngram leases and records the
whole PMA report; it does not establish complete CPU weight/arena accounting.

Evidence: `parity-results/qwen-flash-next/source-admission-cpu2-identity-install-fixed-r1/`.
The `admission_memory.txt` and source code establish this independently of RSS.

The concrete seams are:

- `WeightManager::prepareMoEExpertOverlayWeights` initially packs CPU expert
  engines and publishes shared participant/domain registry aliases without an
  allocation lease. Later migration slots use `CpuExpertSlotPool`, which already
  has claims and is a different owner.
- `PreparedWeightStore::prepareGemm` accepts a persistent-weight owner but does
  not bind the CPU prepared allocation to it.
- `BufferArena::allocate` creates retained CPU tensor roots without a physical
  claim. Views and externally supplied roots must remain borrowed.

Unused reserved prefix capacity legitimately has zero materialization. A
workspace can legitimately use less than its admitted maximum. Neither should
be forced to claim capacity as live storage to make a report look complete.

## Bounded repair

1. Put exact final allocation geometry beside `CPUWeightStoragePlacement` and
   the actual packed representation. Share its alignment/rounding with planning
   and allocation. `packedWeightBytes()` is a logical-size query, not a physical
   allocation census. Verify retained vectors using `allocationBytes()` metadata.
   Direct local allocations and page-aligned migration slots differ for small
   shapes; do not apply the pool's 4096-byte minimum to every allocation.
2. Pass the existing PMA and typed persistent owner to CPU preparation. Preclaim
   exact storage before allocating; reject an insufficient owner before touching
   the allocator. After packing, validate the actual extent and publish only the
   completed owner. Preparation failure rolls back both allocation and claim.
3. Keep the lease with the physical packed-storage owner, including detach,
   attach and explicit `releaseWeights()` semantics. Participant, domain, graph,
   registry and RCU references must share that owner and claim once. A source-
   backed floating engine must borrow its already-owned source, not count a copy
   that was never allocated. Do not add a second registry or a host shadow ledger.
4. Bind arena-owned CPU tensor roots to the existing ActivationArena authority
   before allocation. External tensors, views and reusable family aliases do not
   claim again. Retire storage before releasing its lease, including exceptional
   graph construction and final runner/model retirement.
5. Preserve successful CPU clone/export/receive operations under a full persistent
   weight budget. Explicit temporary copies must use their staging authority;
   serialization-only reads should use the established borrowed export contract.
   Do not make every copy charge the already-full persistent owner.
6. Keep existing GPU loading pools, CPU migration slots and workspace claims
   intact. Do not fill missing totals with an aggregate reserve, change capacity
   arithmetic, increase admission budgets or infer live bytes from RSS.

## Regressions and qualification

Add focused device-free tests to the existing preparation/arena fixtures and
register explicit ProductionTestPreflight aliases. Prove exact rounded physical
bytes across all supported CPU formats, including sub-page shapes; refusal before
allocation; failure rollback; source-backed versus owned prepared storage;
shared graph-family and participant/domain aliases; delayed registry/RCU
retirement; detach/attach/release; external roots/views; repeated allocation and
exceptional teardown. Replay real packed kernels to preserve existing arithmetic.

Then strengthen real source admission to compare observed physical owners with
PMA materialization for these categories. Refresh CPU1, CPU TP2/PP2 and GPU+CPU
ExpertOverlay cells, and the complete Unit/preflight suite. Keep numerical,
serving movement and performance gates distinct from admission. Do not declare
all CPU storage accounted merely because the current source-admission sample is
green.
