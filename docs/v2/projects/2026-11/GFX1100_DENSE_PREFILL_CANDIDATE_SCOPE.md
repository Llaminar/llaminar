# Dense gfx1100 installed-versus-WMMA collection

The existing production dense collector now accepts an explicit
`--candidate-scope installed-vs-native-wmma` protocol. It compares the installed
Auto producer with the four compiled WMMA candidates, preserving each original
source format, physical shape, row bucket, byte proof and resource witness.
Bound plans can now publish only profitable WMMA changes while retaining every
other installed lookup. Collection alone does not install dispatch decisions.

The narrowed scope requires `--backend rocm`, explicit NativeQuantised
preparation and `--gpu-architecture gfx1100`. Its exact candidate order is Auto,
WMMA/N32/MT32, WMMA/N64/MT32, WMMA/N64/MT64, WMMA/N128/MT64. The native trainer
already owns filtering and executes installed production Auto. The independent
numerical oracle remains separate from that timed baseline.

## Identity and qualification

Exhaustive remains the default. Existing v4 cell identities and plan payloads
are unchanged. Narrowed cells add their scope to cell and batch identity; v5
plans bind the scope and complete per-source candidate roster. Missing or extra
candidate rows, mixed batches, cross-scope resumes and rehashed schema/scope
contradictions are rejected.

Optional v6 plans additionally freeze the original generated include against
the actual linked core table before any timing. The device-free
`ROCmDensePrefillPolicy.CompiledInventory` query serializes every compiled field,
including prepared family, producer and architecture. Python independently
serializes the accepted generated rows and requires the count/hash to agree.
The plan retains original include/query bytes, hashes and exact Release
executable/resolved-core identity. An existing unbound plan cannot acquire this
witness after collection. The native query opens no accelerator.

The cold query exposed an actual ELF ownership defect: a trainer-local inline
table could interpose on the core's GNU-unique table symbol. The production table
now has hidden visibility, and both trainer and standalone tests borrow its
read-only span. The loader regression exports a deliberately conflicting table:
its control DSO reproduces interposition, while the actual production selector
and table retain the same core owner. The final table is local, with no dynamic
export or interposing relocation. This is a cold evidence/ownership repair;
runtime lookup and kernel arithmetic are unchanged.

Profiler requests retain original source aliases and actual unforced Auto,
including a physical Auto launch outside the forced roster. A narrowed request
cannot claim a forced Dot4 candidate. Exact original executable/core checks,
complete regenerated profile obligations and alias pooling remain mandatory.

The complete collector regression file passes 80 tests. Its nine new scope
contracts are explicitly included in
`V2_Integration_NativeQuantisedDensePrefillEvidenceIdentity`; that focused gate
also passes. Eight source-default/native, CUDA/ROCm and gfx906/gfx1100 exhaustive
plan payloads were compared directly with the pre-change v4 implementation and
are identical. The Release native trainer builds for gfx906 and gfx1100 with
the spill guard enabled. Local evidence is in
`/tmp/llaminar-gfx1100-wmma-prefill-evidence/dense-candidate-scope-*`.

## Finite collection scope

Declare the model/operation, source formats, canonical shape names and M buckets
before collection. `modern_sweep_scope.modern_sweep_shape_names` projects the
shared inventory without renaming aliases. `gpu_prefill_measurements` remains
the physical geometry authority; do not introduce a separate backend shape list.
The modern inventory currently contains 95 N/K shapes, 22 row buckets and
21 source formats. Narrowing the roster would produce 219,450 candidate rows
for that full cross-product, versus 1,262,360 exhaustive rows. Those counts are
an inventory audit, not a claim that the full matrix has been measured.

For example, a declared ordinary QKV projection slice can be planned without
starting a GPU process:

```bash
PYTHONPATH=tests/v2/performance/kernels python -m \
  native_quantised_dispatch.production_dense_prefill_sweep plan \
  --backend rocm --prepared-family native-quantised \
  --gpu-architecture gfx1100 \
  --candidate-scope installed-vs-native-wmma \
  --shape Qwen36_Attn_QKVProjection --shape Qwen4Exp_6144x2560 \
  --m 512,1024 --warmup-runs 5 --bench-runs 31 \
  --binary /path/to/Release/v2_perf_native_quantised_sweep \
  --installed-base-include src/v2/kernels/rocm/gemm/ROCmDenseProductionPrefillOverlayGenerated.inc \
  --output-dir /path/to/local-corpus
```

Use the same declared arguments for `run`, `status` and `combine`; `run` also
requires `--devices`. Omitting `--format` retains every source format. Existing
M batching, atomic completion and `--maximum-new-cells` resume controls still
apply. Preserve other installed architecture/shape keys during later emission.

## Selective publication

`dense_prefill_policy_decisions` reconstructs one `InstallNativeWmma` or
`RetainInstalled` decision for every planned runtime key. It validates all
candidate correctness/resource/sample records and complete original source
aliases, including keys that retain policy. A WMMA candidate is eligible only
when at least one alias changes physical launch and each changed alias's raw
median strictly improves on actual Auto. Complete six-field launches determine
physical equivalence; timing noise on an identical launch is not an improvement.
Eligible candidates are ranked by geometric-mean gain, worst-alias gain and
candidate identity. Ties or a regressing alias retain the original policy.
The explicit rule is `changed-alias-strict-median-improvement-v1`.

Retention preserves literal exact rows, compatible architecture rows, generic
lookup absence and the original generated envelope. In particular, a physical
IQ4_XS key that inherits a historical lookup must not gain a concrete row merely
because Auto was timed. Every decision digest includes the plan, compiled-base
binding, producer and aggregate/timing/manifest hashes for all source cells.
Retained policy is not recertified as newly measured evidence.

Bound profiler requests project only installed keys, but still require the
winner and actual unforced Auto for every original alias of those keys. The
complete decision digest joins every request. Publication regenerates those
obligations and requires complete isolated evidence. If every key retains,
publication instead returns the exact frozen base and a typed no-change receipt;
there is no empty or fabricated profiler manifest. Existing unbound v4/v5
concrete-overlay behavior remains intact.

The include, summary and receipt destinations are admitted together before any
write. They cannot overwrite original plan, cell or profiler evidence, each
other or reserved staging names. Unique temporary directories prevent stale
staging symlinks from redirecting writes. The receipt is published last. Native
profiler-request CLI publication applies the same source protection.

The complete legacy collector suite passes 80 tests (9.389s); the new selective
replay passes 16 tests (4.125s), including real CLI collision and symlink controls.
The immutable canonical declaration catalog is shared between repeated reads;
mutable plans, timings, manifests and producer evidence are always reread.
Focused registrations explicitly belong to production preflight:

- `V2_Integration_NativeQuantisedDensePolicyBindingReplay`
- `V2_Integration_NativeQuantisedDensePolicyDecisionsReplay`
- `V2_Integration_ROCmDensePolicyOwnershipReplay`
- `V2_Integration_ROCmDensePrefillCompiledPolicyReplay`

## Declared broader collection

The original v6 plan at
`/tmp/llaminar-gfx1100-wmma-prefill-evidence/dense-broader-qkv/production_dense_prefill.plan.json`
contains 84 cells: all 21 formats, M512/M1024, QKV N12288/K5120 and the shared
N6144/K2560 projection. Each uses five warmups and 31 samples of 16 captured
complete operations. The plan digest is
`sha256:0a82bc187738b0184c142d9a4302bff7c88416a57e0531f1dbde5ee3ca7566bf`.
The exact Release executable and core were built for gfx906/gfx1100 with the
mandatory spill guard. Both Release and Integration builds pass.

Collection uses actual gfx1100 HIP5/PCI0000:b5:00.0, two M cells per retained
weight batch. Per-batch receipts bind original producer/artifact hashes and
retain the observer's activity reports. Rejected attempts remain archived;
isolation receipts are no longer an acceptance gate under the user's scope.
Collection, kernel profiling, generated installation and rebuilt Auto qualification
are distinct gates; all four are now complete for this declared84-cell scope.

The original run completed all 84 cells in 42 accepted batches; three
observed-overlap batches were rejected and recollected. The user accepts this
original performance evidence without requiring isolation receipts or another
timing collection. All byte/resource gates pass. The inventory collapses
to 68 runtime keys (17 per shape/bucket), and every key has an eligible changed
WMMA launch. Per-key pooled gains range from 1.2493 to 2.0426. The geometric
means across each 17-key group are:

| Shape | M512 | M1024 |
|---|---:|---:|
| QKV N12288/K5120 | 1.4160× | 1.4121× |
| Projection N6144/K2560 | 1.4326× | 1.3853× |

`decisions.json` and `candidate-summary.csv` retain the complete inventory;
its digest is
`sha256:bdfdb4906f49c60e1d1e0f7eb333d26db4c4117a9ba1431366465e393a226ead`.
All168 original-alias winner/baseline profile obligations are complete and pass
the canonical publication validator. Activation quantization and the selected
native producer are authenticated separately from canonical timings; power-state
restoration remains part of each counter record.

The selective emitter adds68 keys and retains616 original entries, for684 total.
The generated include SHA256 is
`3cb0480cc4cbe3f0e4f617c9c8d6c1da6e9bafe11fb5c5d8e3ee8659f740863b`;
`dense-broader-install-receipt.json` binds the unchanged decision digest and base.
Both normal HIP architecture builds pass.15 device-free policy/ownership cases
and all12 installed-source replay groups pass against the rebuilt table. Rebuilt
actual-Auto timing is complete for all84 source cells. Every cell is faster than
its original Auto median: gains range from1.22653 to2.05860×, with geometric mean
1.39951× across the84 original source cells. Each retains31 samples of16 captured
complete operations, observed selected launch and zero local-memory resource
witnesses. The collector admits only the expected compiled684-row inventory and
exact native producer; it does not substitute a forced candidate for Auto.

The accepted result is
`/tmp/llaminar-gfx1100-wmma-prefill-evidence/dense-installed-auto-accepted/campaign.json`,
schema `dense-installed-auto-accepted-campaign-v1`, producer SHA256
`ae674bfc6e31db7e5b7f3d2ba81f0f0b028e39011339cb6091ff032df8bc6bdf`.
It records complete=true, all_faster_than_original_auto=true and restored AUTO
power. This closes rebuilt-Auto qualification for the finite scope above; it
makes no claim about other geometries, a full model benchmark or full preflight.

The six focused registered integration checks all pass in 12.94s, including the
compiled policy query, actual loader ownership replay, overlay admission,
original evidence contracts, binding replay and selective-decision replay.
This is the authorized focused suite, not full preflight execution.


## Superseded isolation-only recollection

The user removed isolation receipts and isolation-only recollection from this
task's acceptance conditions. The original timing/profile artifacts remain
usable and unchanged. The replacement generation is canceled; historical audit
and partial replacement artifacts are retained locally. See the
[superseded audit](GFX1100_PREFILL_ISOLATION_REQUALIFICATION.md) for the original
investigation. No remaining work depends on that audit or its monitors.
