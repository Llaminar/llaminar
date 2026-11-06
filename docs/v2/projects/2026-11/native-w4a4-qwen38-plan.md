# Native Qwen3.8 W4A4/H implementation plan

This is an implementation/evidence plan, not a production qualification report.
The complete objective is native support for
`raspbfox/Qwen3.8-27B-W4A4-GGUF`, starting from MXFP4, with a pinned exact
HF/PyTorch oracle, CPU/CUDA/ROCm execution, graph integration, full Unit/preflight,
real-model parity and capture/reset, and matched Release economics against
ruquant. MTP needs an independently validated basis contract. DFlash2 is excluded.
No dequantization, alternate-format, eager or serial-replay fallback satisfies it.

**Delivery checkpoint:** this initial documentation-only commit records the
contract and evidence from the isolated implementation worktree. Its native
implementation changes are not yet committed: the complete required gates
remain unqualified. The draft PR retains the full objective and depends on the
MXFP4 source stack; it does not advertise production W4A4 support.

## Source and artifact identity

The implementation starts at MXFP4 commit
`fbee08c6850ea56d1d0454fd87140cf9a0f69e64` on `feat/qwen38-w4a4`; the E8M0 stack
is not a prerequisite. The numerical reference and immutable dependency pins are
recorded in the working implementation's
`python/reference/quantization/README.md`. That reference and its implementation
will be published together with the qualified source slice.

The target is the 17,190,149,440-byte
`Qwen3.8-27B-W4A4-OST-GPTQ-wg64-rb4.gguf` at Hub revision
`174067dc0522d3c69912487e799b4e9541cb296c`, SHA-256
`b1015b212068a2838f23a68985065352e337d9e4e0f40abab3832e610a6a5ef8`.
Its actual header is `qwen35`, not the repository-level drafter architecture.
It contains 64 main layers and one nextn layer, with 272 W4A4 and 80 W4A4H tensors.
The other weights retain their original F32/Q8_0/Q4_0/BF16 representations.
The existing IQ4_XS Qwen3.8 matrix cell does not qualify this artifact.

On 2026-11-05 the complete target was authenticated in an owned remote directory.
The existing shared custom GGUF differed by 128 header bytes; its tensor
identities, offsets and payload lengths were identical. Combining the pinned
Hub header with the unchanged payload produced exactly the pinned file size and
SHA-256. A separate `sha256sum` of the published file confirmed the same digest.
The shared source was opened read-only and remains intact. This is byte-identical
artifact acquisition, not a format conversion or an arithmetic approximation.
The source custom file's digest was
`c87445b47fb988a3b267ea15555f70e6507dd689a2302ea7e2ef55f63aedf380`.

Local acquisition and initial negative-admission records remain outside Git in
`/home/fank/llaminar-work/qualification/qwen38-w4a4/` and
`/home/fank/llaminar-work/w4a4/`. The new remote root is
`/home/gpu/w4a4-native.TUOvibRt/`. These local records are not a release certificate.
Large reference/campaign payloads must be published through `Llaminar/corpora`,
not added to this source checkout.

## Numerical authority

W4A4/H is distinct from MXFP4 and from ordinary weight-only int4. Required
invariants are:

1. Preserve the FP32 row unit, integer weight-group multipliers, signed codes,
   clipping bits and declared rotation length. Source descriptors and physical
   storage sizes remain separate from arithmetic dispatch identities.
2. Quantize activations in groups of 64 using one whole-token range/unit.
   Preserve zero-inclusive extrema, rounding, clipping and the zero-row rule.
3. Perform the signed zero-point correction and integer multiplier products;
   shift each group term arithmetically by four before the int32 sum.
4. Round the FP32 epilogue in the pinned order. For example the native grid case
   yields `0x44af0001` (`1400.000122...`), not mathematical `1400.0`.
5. W4A4H performs the prescribed ordered Paley/Sylvester transform before A4.
   The target's 1536/4352 rotation lengths are examples, not hard-coded dispatch
   assumptions. The offline R1 is already folded into the exported model.
6. A K shard cannot choose an independent activation unit or arbitrary
   Hadamard fragment. TP admission must preserve the complete mathematical
   operand using explicit, economical collective/ownership contracts.
7. A K partition must combine the corrected, shifted **integer** partials before
   one FP32 epilogue. Reducing already-rounded floating partial outputs changes
   the contract. A zero local range must not select the zero-row unit before the
   global range is known.
8. Reject unsupported precision, geometry, arithmetic version and malformed
   metadata. No source format may be relabeled as an existing weight codebook.

## Work packages and gates

### 1. Independent reference

- Compile the pinned upstream CPU translation units to record complete inputs,
  transformed values, native A4 operands and outputs; retain source/compiler/DSO
  identity. Do not regenerate expected results during tests.
- Use a CPU FP32 PyTorch implementation with integer group corrections/shifts,
  bounded exact-integer BLAS and explicitly ordered transforms.
- Bind native operands into the pinned HF graph by replacing linear modules and
  retiring their unused floating parameters. Dequantized floating weights do
  not implement W4A4's online activation contract.
- Pin the installed HF source and PyTorch version; retain model identity in
  generated real-model evidence.
- Keep small recorded arithmetic traces device-free and explicit in preflight.
  Tiny HF prefill/decode/reset tests prove binding and lifecycle, not real-model
  numerical accuracy or production graph capture.

**Implemented evidence:** 11 recorded native cases match transform, A4 operand
and output bytes. The focused arithmetic/HF tests pass, including a complete
small recurrent/full-attention GGUF, module ownership, cached decode/reset,
malformed source-view retirement and unqualified-MTP rejection. After these
changes, the existing complete Python-reference Unit selection reports
**462 passed, 44 intentionally excluded integration cases**, with no model/GPU
execution. This is not the complete C++ Unit or production-preflight gate.
The explicit CTest entries are `V2_Integration_W4A4ReferenceReplay` and
`V2_Integration_W4A4HFReferenceContracts`; their full configured-gate execution
remains to be proven.

### 2. Native storage and CPU execution

- Add typed source storage and admission without advertising execution early.
- Carry policy through slicing, preparation, weight lifetime, transfer,
  placement and PMA-owned capacity accounting. Preserve exact physical bytes
  and include metadata/workspaces in the BOM.
- Implement CPU A4 preparation, transforms and exact int32 projections through
  kernel interfaces, with ordinary and grouped lanes. Prepared storage and
  activation workspace must have one owner.
- Exercise the actual implementation against recorded inputs and malformed
  contracts. Cover K/N boundaries, all admitted source encodings, M from one
  through supported grouped work, all positive worker counts and both ISAs.
- Measure physical-core scaling and prove grouped economics; a serial replay
  loop is not a production implementation.

**Current component evidence (2026-11-05):** `W4A4MatrixContract` and
`W4A4SourceView` admit exact native blocks without allocating or binding an
unrelated source pointer. `CPUW4A4Arithmetic` prepares ordered transforms and
packed A4 with invocation-owned banks, then shares weight decoding across up to
four tokens. Its overflow, extent, overlap, policy and rounding checks remain
active in Release. No weight dequantization or production serial-row replay is
used.

The actual CPU body has been compiled independently with AVX2 and AVX512 flags.
For **each** compiled candidate, all 11 source-recorded cases match transformed,
packed-A4 and output bytes with 1–8 workers and both ordinary/existing-team
entry (**176 exact executions per candidate**). Separate synthetic contracts
sweep M=0..65, compare grouped outputs with a serial diagnostic oracle, preserve
strided/inactive bytes, and reject incomplete/overlapping or incompatible
invocations. The six source-format contracts also pass. Logs are
`w4a4/cpu-native-replay-avx2.log`, `w4a4/cpu-arithmetic-contracts.log` and
`w4a4/cpu-native-avx512.log` beneath the local evidence root.

`W4A4WeightStorage` now retains a single native physical owner. Owned bytes
must authenticate their existing PMA lease and allocator before zero-copy
adoption; mapped pointers derive from a retained `MmapRegion` and checked file
coordinates. Complete-row slices preserve both ownership and the exact policy
without rescanning unchanged metadata. Five lifecycle/admission tests pass
against the real allocation/mapping authorities. `CPUW4A4Workspace` contributes
its exact typed counts to the existing arena/PMA BOM and borrows invocation-only
prefixes. Two further tests execute repeated full/partial/empty arithmetic with
real `DeviceWorkspaceManager` storage, proving stable addresses and unchanged
live claims. Evidence is `w4a4/storage-contract.log` and
`w4a4/storage-workspace-contracts.log`; these seven GTests were compiled with
the production storage bodies and linked to the built CPU core.

**Expanded component evidence (2026-11-06):** `CPUW4A4GemmKernel` now owns
ordinary, counted, fused and SwiGLU/verifier tensor entrypoints. Prepared A4
validity is sealed in original physical coordinates; later controller changes
cannot admit stale operands. `W4A4Tensor` and its KernelFactory retain one
immutable physical owner through source retirement, refusing writable,
weight-only-conversion and generic-transfer interfaces. The final CPU writer
fills one admitted allocation directly from bounded source windows and publishes
only after complete native validation. No second decoded source is retained.

The real `ModelLoader` now recognizes GGUF 44/45 and exact complete-row extents.
Mapped row views validate the original matrix policy before selecting rows;
unmapped production reads enter the existing Store's admitted final writer.
Transport windows may split K, but tensor/engine publication may not. CPU weight
BOMs price the same original packed allocation, and metadata serialization
preserves native type/extent identity. Source and published-profile MTP admission
reject the unqualified basis explicitly, including automatic MTP; this rejection
is not independent basis qualification. GPU admission still fails precisely
until its native device representation/execution is implemented.

The current combined diagnostic compiled the changed source bodies and linked
remaining infrastructure from the existing Integration core. It passes
**21 GTests** plus **11 recorded cases / 176 candidate invocations / 704 exact
retained GEMM executions**, including native writer/factory ingestion and
full/partial/empty/repeated work. Evidence:
`w4a4/cpu-model-source-contracts-4.log`. The successful rerun peaked at
405,590,016 cgroup bytes with zero OOM events; compilation peaked at
1,394,864,128 bytes. Earlier wrapper-only failures remain recorded: direct
Python-file invocation broke package imports, and login-shell cleanup ran
`clear_console -q` after `exit 0`. The owned noninteractive wrapper now uses
module invocation and `bash -c`, without modifying global tools or profiles.
The earlier arithmetic-only AVX2/AVX512 rerun also passed both CTests and
176 exact recorded executions **per ISA**; its complete evidence is
`w4a4/cpu-counted-rows-isa-replay.log`.

A subsequent current-body diagnostic adds the canonical CPU projection
workspace contribution and shared fused metadata geometry. The CPU, CUDA and
ROCm estimators now use one source-group discovery helper rather than another
W4A4-specific model-role list. Complete-K, metadata-versus-runtime bank identity
and nonconcurrent layer maximum merging are covered. **71 GTests** pass:
49 existing `WorkspaceMemoryEstimator` cases and 22 W4A4 contracts, followed
by the same **704 exact retained GEMM executions**. Evidence:
`w4a4/cpu-workspace-planning-contracts-2.log`, peak 1,107,533,824 cgroup bytes,
zero OOM events, successful process retirement.

This exercise exposed a native adapter ABI defect: ordinary graph consumers
pass `transpose_B=false`, whereas the initial W4A4 adapter demanded true.
The failing current-body control is retained in
`w4a4/cpu-workspace-orientation-negative.log` (70/71 tests passed).
Both legacy flag values now consume the same source-owned [N,K] matrix,
consistent with native prepared GEMM semantics; no weight transpose or changed
arithmetic occurs. Tensor replay now uses the ordinary false flag, and
`W4A4NativeProjectionOrientation` is explicitly registered in preflight.

These remain production-body **component** checks, not a full-current-core or
engine admission certificate. Complete WeightManager/graph execution,
automatic-planning sample ingestion, runtime ISA dispatch, total geometry
coverage, profiling and economy remain unfinished. Explicit Unit/preflight registrations
include source, tensor, storage, workspace, GEMM, recorded replay and
`W4A4MTPBasisAdmission` contracts; CMake owns their exact names. Complete
configured Unit/preflight execution remains unproven.

### 3. CUDA and ROCm execution

- Implement capture-safe activation preparation, transforms and int4/int32
  projections with exact streams and admitted persistent scratch.
- Keep whole-token scale selection and live row counts device-owned. No host
  count download, hot allocation, dynamic rebinding or hidden synchronization.
- Prove serial-row byte equivalence, empty/partial work, repeated replay,
  large-to-small replay and request reset on actual native captured graphs.
- Register focused replays in explicit backend suites and preflight. Compile
  every shipped architecture under the spill guard; native hardware runs do
  not prove another architecture's resource allocation.
- Profile exact candidates with Nsight/ROCm evidence, isolating register/spill,
  occupancy, memory and timing records. Retain both correctness and economy.

### 4. Graph and MTP admission

- Reuse the declarative Qwen35 architecture with typed operand policy; do not
  create a second model-specific imperative execution path.
- Bind native projection workspace and complete graph-cache identity before
  capture. Retain existing ordinary-format projections as declared by source.
- Validate rank-local/TP/PP placement and economical live collective extents;
  no communication of unused capacity or local replacement of a global unit.
- The pinned exporter copies the nextn layer while rotating the main graph.
  The artifact has no `nextn.h_unrotate` or independent nextn embedding/head.
  This establishes an admission risk, not a measured numerical defect.
  Refuse MTP with a precise unproven-basis diagnostic until the interface is
  independently authenticated; never silently turn automatic MTP off.
- Explicit main-model `--no-mtp` tests are legitimate, but cannot certify MTP.

### 5. Real-model qualification

- Discover CMake's current inventory and build/run the complete Unit and
  `ProductionTestPreflight` prerequisites. Diagnose inherited failures rather
  than skipping or relabeling them.
- Register the pinned artifact in the existing typed parity matrix, then use
  the canonical HF/CSV workflow to compare every relevant stage on CPU, CUDA
  and ROCm. Preserve real optimized dispatch and the actual sampler policy.
- Exercise production capture, cache reuse, prefix paths, repeated requests,
  partial buckets and reset. Assert PerfStats path/transfer counters.
- Report missing hardware as unqualified. No loader acceptance, tiny fixture,
  isolated arithmetic or alternate model certificate substitutes for this gate.

### 6. Release economy and delivery

- Build production Release binaries and pin the same target bytes, prompt,
  decode length, sampler, MTP policy, KV settings and topology for Llaminar and
  the pinned ruquant fork. Do not include DFlash2 in target-only comparisons.
- Measure default production placement first, then use explicit saved placement
  for controlled experiments. Include phase/kernel attribution and actual
  communication/transfer evidence, not only isolated kernel timing.
- Keep profiler runs separate from canonical timing. Tune CPU scaling and both
  GPU backends without changing the arithmetic contract or weakening gates.
- Checkpoint coherent green slices with ordinary hooks, preserve the MXFP4
  stack dependency in the feature PR, and deliver only after the full objective
  is audited against authoritative artifacts and results. No release-to-master
  action is authorized by this plan.

## Resource and process boundaries

All writes use the isolated implementation checkout/build roots. Shared MI50
models, the MXFP4 `lr` bundle, ruquant sources, SDKs and other lanes' build trees
remain untouched. The old Q4_0/Q6_K baseline files and `lr27` were already removed
by another authorized task; do not assume they still exist. Storage for one
additional pinned target and a small runtime bundle has been coordinated; no
exclusive GPU reservation is implied.

Avoid `V2_FetchModelsFixture`; CTest invocations use `-FA '.*'` where appropriate.
Use unrestricted Ninja/ccache builds while sizing container memory from actual
available RAM. The two large generated CUDA families must be compiled one
architecture at a time for their spill sweep, rather than reproducing the known
multi-architecture OOM. Reuse vendor closures only read-only with actual DSO
identity checks. Never transplant another lane's unverified repair or claim its
partial gate as this branch's evidence.

The WSL source filesystem is nearly full. This lane's Linux-backed Docker
volumes are `llaminar-w4a4-cpu-e408aff7` (build artifacts, mounted at
`/work/w4a4-cpu`) and `llaminar-w4a4-onednn-e408aff7` (read-only dependency
closure). The latter was initialized from the existing toolchain image; its
AVX512 `libdnnl.so.3.11` hashes to
`0ee075f1e8075dee48e7d9fc02820894778e9452fdf21bf7fbb8c5aa3d30cb42`.
The discarded local bootstrap tree is not an active build.

Complete prerequisite builds hit recorded cgroup OOM at 12 GiB, 16 GiB, and
16 GiB with compact DWARF. An eight-physical-core quota made the pinned Ninja
select ten native jobs, but overlapping parity-definition compilations still
exhausted 16 GiB. This was isolated with the unchanged
`Test__FrozenParityHistoryReplay.cpp` compiler command: it passed alone in
51.71 seconds, with 2,733,384 KiB maximum RSS and 2,844,180,480 bytes cgroup peak,
and no OOM. Rounding measured compile demand to 3 GiB admits five native jobs
within 16 GiB. Ninja selects that default under a three-CPU container quota;
`cmake --build --parallel` remains bare and the complete inventory is retained.
No test or numerical policy was weakened.

The last complete-inventory build retained `-O3`, `-g1`, assertions and
snapshots. It stopped with zero OOM events and 12,222,582,784 cgroup peak bytes:
an unconditional CUDA-GDN preflight obligation incorrectly pulled a CUDA-only
translation unit into the CPU-only build. Its log is
`w4a4/cpu-build-measured-admission.log`; watcher `7217a5e18` and its container
are retired. Only the focused backend-admission guard and real-finalizer
fixture/original-code control were ported from the peer repair. Twenty focused
repetitions passed unchanged, but one earlier Ninja manifest-regeneration
failure remains unexplained and the complete inventory has not been rebuilt.
The owned script/volume layout is `w4a4/build-layout.json`. Earlier OOMs and
the isolated compiler measurement remain in separate logs. Do not raise memory
blindly or start competing large builds; full Unit/preflight is still unqualified.

This local kernel lacks readable `/proc/self/io`; required accounting tests
need a suitable native environment, not a weakened validator. Separately,
Clover demonstrated that CTest 3.28 omits required timeout-signal metadata from
its JSON inventory, while an authenticated private CMake/CTest 4.2.0 pair
exports it. The wheel SHA-256 is
`5461ceca47ad352bdb3db2fdd5affdbc5707aaee415c5ff12773b8cc0d5f5949`;
this lane independently checked that wheel and both ELF tool hashes, then
extracted its own copy without package hooks into
`/work/w4a4-cpu/.tools/cmake420/`. The active build binds both tools through
subprocess PATH while retaining `/usr/local/bin/ninja`. The source already
scopes policy compatibility to libfort; no new global policy override was added.
Clover's broader prerequisite repairs remain unqualified and have not been
imported.
