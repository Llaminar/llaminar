# Dense gfx1100 installed-versus-WMMA collection

The existing production dense collector now accepts an explicit
`--candidate-scope installed-vs-native-wmma` protocol. It compares the installed
Auto producer with the four compiled WMMA candidates, preserving each original
source format, physical shape, row bucket, byte proof and resource witness.
This is collection infrastructure; it does not install dispatch decisions.

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
  --shape Qwen36_Attn_QKVProjection --m 512,1024 \
  --binary /path/to/Release/v2_perf_native_quantised_sweep \
  --output-dir /path/to/local-corpus
```

Use the same declared arguments for `run`, `status` and `combine`; `run` also
requires `--devices`. Omitting `--format` retains every source format. Existing
M batching, atomic completion and `--maximum-new-cells` resume controls still
apply. Preserve other installed architecture/shape keys during later emission.

Selective publication remains separate work. The existing emitter still makes
concrete rows for Auto winners, and profiling still requires winner/baseline
observations for every source cell. A future install-WMMA/retain-installed
protocol must bind the base selector, cover every declared pooled key and
represent an all-Auto no-change result explicitly. This scope change does not
remove those obligations or claim automatic installation.
