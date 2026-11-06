# Native ARM/macOS port: first slice and remaining blockers

Date: 2026-11-04. This is an implementation handoff, not an ARM support or
performance certificate.

This is the **historical first-slice snapshot**. Later implementations and
current bounded evidence are recorded in
[the comprehensive plan](ARM_CPU_INFERENCE_KERNEL_PLAN.md) and
[implementation evidence](ARM_CPU_KERNEL_IMPLEMENTATION_EVIDENCE.md).
Its six-test counts, failed header probe and initial ISA flags below are not
current-head claims.

## Source and host

- Base: `origin/feat/qfn-ngram-continuation`, commit
  `2cba3ca1d` (`fix(test): list CUDA host-registration preflight tests under HAVE_CUDA`).
- Working branch: `codex/qfn-ngram-continuation-arm64`.
- Host: macOS 27.0.1 / Darwin 27, AArch64 Apple Silicon, 64 GiB RAM,
  10 physical cores (eight performance, two efficiency).
- Native compilers tested: AppleClang 21.0.0 and Homebrew LLVM 22.1.8.
- Installed missing development dependencies: Ninja, ccache and Homebrew OpenMPI.
  Homebrew's current OpenMPI is 5.0.11, **not** the runtime's required internal
  OpenMPI 4.1.6 ABI. Installing it does not resolve that production dependency.
- No model, corpus submodule, or LFS payload was downloaded.

## Implemented

`cmake/CPUInstructionSet.cmake` owns the compiler envelope. AArch64 defaults
to native NEON (`-march=armv8-a+simd`); x86-64 retains its existing AVX512
default and AVX2 option. Cross-architecture ISA selections fail before probing
dependencies. Darwin's explicit `CMAKE_OSX_ARCHITECTURES` selection takes
precedence over the host processor; universal SIMD envelopes are rejected.

The OpenMP binding helper reads a Mach-O install name with `otool -D` on macOS
and retains ELF SONAME authentication elsewhere. Missing/ambiguous identities,
failed readers and malformed output fail closed. This does not port the
ELF-specific installed-loader preflight or other runtime loading.

The actual production `VectorPrimitives.cpp` now implements these operations
with AArch64 NEON, without x86 intrinsic emulation:

- dot product;
- four-row grouped dot product with shared weight loads;
- AXPY;
- in-place scale;
- vector add, including exact input/output aliasing.

Serial and four-row dot products share a compile-time kernel body, but grouped
execution does not call/replay serial rows. Each row retains its two independent
four-lane FMA chains, horizontal fold and explicit FMA tail. The native ARM
assembly contains `fmla.4s` and `faddp`; the four-row loop keeps eight SIMD
accumulators, loads each weight vector once for the group, and uses no stack
frame or spill/reload traffic in the inspected AppleClang Release object.
This is code-generation inspection, **not** IPC/cache or end-to-end economy
certification.

`tests/v2/portability` builds those production primitives independently of the
currently Linux-only core. It consumes the same test-registration module as
`tests/v2/CMakeLists.txt`, including explicit focused preflight registrations.
It is not a new inference entrypoint and not the complete production gate.
The focused tests cover all lengths 0–1025 plus larger/tail geometries,
unaligned starts, strided rows/outputs, output padding, cancellation, repeated
reuse, independent FP64 numerical checks, exact aliasing and special values.

## Validation evidence

| Check | Result |
|---|---|
| Native AppleClang Release, all six focused tests | PASS |
| Native AppleClang Integration + ASan/UBSan, all six | PASS |
| Native AppleClang Debug, `-Wall -Wextra -Wpedantic -Werror`, all six | PASS |
| Native LLVM Release, all six | PASS |
| Focused grouped-byte-equivalence regression, 20 consecutive runs | PASS |
| x86-64 AVX2 and AVX512 focused projects, cross-compiled on this host | Compile/link PASS; not executed |
| ARM + AVX2 and Darwin universal configurations | Correctly rejected |
| Full native `src/v2` Integration configuration | FAIL at Linux libnuma requirement |
| Native syntax probe of `CPUNativeVNNIGemv.h` | FAIL: x86-only `immintrin.h` |
| Complete production Unit/preflight, model inference, performance | Not runnable; not certified |

Rosetta is not installed: the x86 process probe reports
`arch: posix_spawnp: /usr/bin/true: Bad CPU type in executable`.
No emulator was installed and no cross-compiled x86 runtime pass is claimed.

Reproduce the focused native checks using the README's
[AArch64 portability commands](../../../../README.md#aarch64-portability-checks-not-inference).
For the sanitizer configuration, use a separate tree and add
`-DCMAKE_BUILD_TYPE=Integration`,
`-DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer'` and
`-DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'`.

The full application was attempted with:

```bash
cmake -S src/v2 -B build_v2_arm64 -G Ninja \
  -DCMAKE_BUILD_TYPE=Integration \
  -DCMAKE_MAKE_PROGRAM:FILEPATH="$(command -v ninja)" \
  -DLLAMINAR_CPU_ISA=NEON -DHAVE_CUDA=OFF -DHAVE_ROCM=OFF \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++
```

MPI discovery and the real Homebrew libomp Mach-O identity check succeed;
configuration then stops with:

```text
Native macOS inference is not implemented: the CPU memory/placement backend
requires Linux libnuma. Do not substitute a NUMA stub.
```

## Required next work

1. **Native macOS host-memory and topology authority.** `CPUBackend.cpp`,
   `HostMemoryCapacity.cpp`, `NUMAAllocator.cpp`, `HardwareInventory.cpp`,
   `NUMATopology.cpp` and execution placement currently depend on Linux
   libnuma/sysfs/affinity. Implement native observation/allocation/placement
   contracts, including Apple unified memory and physical-core geometry, while
   retaining `PhysicalMemoryAuthority` as the sole admission/allocation ledger.
   A fake libnuma or guessed capacity reserve is not a port.
2. **OS process and file ownership.** `MPISessionOwner.cpp` uses Linux pidfds,
   signalfd, subreaper/prctl and Linux process/descriptor proofs. Supply macOS
   lifecycle/retirement proofs with equivalent ownership guarantees. Model mmap,
   backing-filesystem reclaim and shared-memory contracts also need native APIs;
   `MmapRegion.h` unconditionally includes NUMA headers today.
3. **MPI ABI.** `NativeMPISessionSchizo.cpp` requires the exact OpenMPI 4.1.6
   ORTE module layout, and CMake requires `open-rte`/`open-pal`. Homebrew OpenMPI
   5 is insufficient. Preserve authenticated rank-environment publication:
   provide a supported native closure or implement and prove the new ABI,
   rather than skipping bootstrap or the guardian.
4. **Complete native ARM kernels and dispatch.** NativeVNNI GEMV/GEMM,
   preparation/packing, attention/GDN, and other primitives/tensor decode code
   still contain x86-specific headers and instructions. Port the actual grouped
   operations and fixed arithmetic trees across every supported format/codebook
   and positive worker count, with economy evidence. The shared ISA dispatch in
   `CPUFeatures.h` is currently x86-only; an ARM build must not inherit its
   generic scalar/cache guesses as production policy. Verify the native
   oneDNN closure rather than assuming its x86 build configuration transfers.
5. **Canonical gates and real inference.** Complete the OS-specific CMake/test
   infrastructure, then run the full Unit/preflight inventory and the applicable
   real-weight CPU model/MTP/prefix-cache gates. The focused six-test project
   cannot substitute for those gates. Qualify performance before advertising
   ARM inference, GPU support, or publishing an ARM image.

No Linux test allowlist, GPU feature, model correctness gate, or image
certification policy was weakened to obtain the focused passes above.
