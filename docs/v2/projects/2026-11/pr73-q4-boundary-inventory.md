# PR73 Q4 boundary investigation (bounded, unqualified)

## CPU-only object comparison

GNU 14.2 in local `g4-cross-test:avx` compiled the actual
`tests/v2/unit/kernels/cpu/Test__NEONRoPE.cpp` twice per ARM envelope.
Only `CPUFeatures.h` changed: b55fc7c34 versus 247e990e7. All kernel headers
were the same current checkout. A copied source root at `/tmp/q4-source`
was reused at the same logical path, avoiding two physical header instances.
No CUDA/HIP definitions, debug flags or generated sources were introduced.

Command (per envelope and phase):

```sh
/usr/bin/g++-14 -std=c++23 -O3 -fopenmp -mcpu=generic \
  -march=ENVELOPE -I/tmp/q4-source -I/source/tests/v2 \
  -c /source/tests/v2/unit/kernels/cpu/Test__NEONRoPE.cpp -o PHASE/ISA.o
```

| Envelope | Before and after SHA256 (identical with `cmp`) |
|---|---|
| armv8-a+simd | 097e8c1b9760025bd7480af6a23d480a8e23a2490fdd9b039d7e780d89ef8f91 |
| armv8-a+simd+dotprod | b4ef7624d51a96b2f21bbb9005fa18241b7280c225466aeb7bba0a50ca2a708d |
| armv8-a+simd+dotprod+i8mm+fp16fml | b78bab69b3127a3588311156620d506feba7b175a8d7f049358adc7daa015214 |

Local evidence: `/var/folders/3n/25k_99y927dcd8v657w8zzqh0000gn/T/tmp.wZhYWXCDuC`.
This is real compiler object equivalence for this CPU consumer, not native
execution or whole-image byte equivalence. The first overlay attempt failed
with duplicate physical header inclusion; it is not comparison evidence.

## Retained production dependency observations

Read-only `ninja -t deps` in Spark's existing
`ev8-candidate-cf29bdbc0-integration-neon`, against
`/src/build_v2_integration`, observed 83 CUDA object dependency records.
The compile database declares 106 CUDA rows / 101 distinct source files.
Thus this retained dependency set is **incomplete**, and predates Q4.
Local captured output: `/tmp/q4-production-deps.txt`.

| Observed consumer / header | Branch or function | Assessment |
|---|---|---|
| NCCLCoordinator.cu; CUDATurboQuantKernels.cu; CUDARingKVCacheTQ.cu; CUDAGatedDeltaNetKernels.cu; CUDARingKVCache.cu -> CPUFeatures.h | host capability functions and constexpr envelope/configuration branches | Shared envelope values must remain identical; host coherence block alone uses the pass predicate. These functions lack device annotations. No device-layout exemption inferred. |
| Same five -> ARMCPUFeatures.h | `compiledEnvelopeFeatures()` raw DOTPROD/FP16_FML/MATMUL_INT8 branches | Returns a host capability bitmask, not a struct layout; no device annotation. Host-only usage still requires full dependency/call inventory. |
| CUDAExpertTierWeightKernels.cu -> CPUNativeQuantisedEncoding.h / CPUNativeQuantisedPreparedFootprint.h | explicit `LLAMINAR_CPU_BLOCK_GEOMETRY_HD` host/device geometry helpers | No CPU ISA branch observed in these two headers. Helpers require normal full-CU qualification; no raw SIMD allowance granted. |
| Direct production .cu/.cuh/.hip/.inc sources | raw CPU feature / COMPILED_WITH tokens | Direct source search found none; this does **not** cover transitive headers or prove absence in device bodies. |
| Raw arm_neon.h / immintrin.h dependencies | retained 83 CUDA object records | None observed in this incomplete ARM snapshot; missing objects, x86 and HIP remain unknown. |

No arbitrary allowlist or automated negative certification is established.
Transitive headers beyond the rows above, device-conditional inline bodies,
and HIP dependencies remain unresolved until complete compiler dependency
closures exist. The isolated real nvcc header/kernel boundary fixture at
`/tmp/q4-boundary.bIU9iN` is narrower than this production inventory.

## Proposed full ARM CUDA scratch qualification (not started)

Coordinate resources with the CPU lane before starting. Snapshot exact source
commit plus tree hash and external closure; never update an existing build.
Use the admitted canonical ARM CUDA13/GNU14 toolchain, CUDA ON / ROCm OFF,
Integration build type and explicit NEON, NEON_DOTPROD, NEON_I8MM in three
separate output roots. ARM shipping CUDA architecture is `121` (owned by
`develop-pr-arm64.yml`); do not substitute the Dockerfile's x86 default
`80;86;89;90`. Build the complete production `cuda_backend` target with
unrestricted parallelism, retaining all generated specialization translation
units, spill policy and compiler database. This is a production-CU compile
qualification, not a replacement for existing Unit/preflight/proof gates.
Compare configured target source inventory with generated command rows and
submitted object inventory; require every production CUDA source, not the
83 previously observed objects or NCCLCoordinator alone. Capture dependency
closure and inspect raw SIMD boundaries and device CPU-macro branches before
claiming this part qualified. Native execution, x86 shipping architectures and
HIP device-byte qualification are separate and still pending.
