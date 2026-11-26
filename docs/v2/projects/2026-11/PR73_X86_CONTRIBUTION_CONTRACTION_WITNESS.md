# PR73 bounded x86 contraction witness

The witness receipt is regenerated at build time in every qualified x86 build
tree (`v2_cpu_contraction_proof`). It is not a committed fixture; the evidence
below is a dated local run, not the source of truth.

## What is witnessed

One small TU (`tests/v2/unit/kernels/cpu/CPUContractionWitness.cpp`) wraps the
real `accumulateSingleScaleBlock{AVX2,AVX512}` helpers in `extern "C"`
`noinline` symbols, one policy per symbol: the expert policy must emit zero
fused instructions, the intentional backend-native policy one, and the
deliberately unguarded negative control one under the same invocation.

## Production command derivation (F-1)

- The command is the production core's actual ordered compile command for
  `src/v2/kernels/cpu/moe/CPUMoEKernel.cpp`, from CMake's compilation database.
  The standard and extensions (`-std=gnu++23`), PIC, OpenMP, configuration
  optimisation, target options, includes and definitions are therefore the
  core's own; nothing is reconstructed.
- The canonical resolver's configured ISA flags must appear exactly once as a
  contiguous block. Any other ISA option (`-m*` except `-m64`/`-mcmodel=`, the
  `cpu_host_compile_contract.py` classification) fails the proof; manual ISA
  flags are never stripped silently.
- Per x86 envelope (`AVX2`, `AVX512`), that block and the
  `LLAMINAR_COMPILED_WITH_*` definitions are replaced by
  `llaminar_cpu_instruction_set` output for the envelope.
- Two settings are recorded per envelope. `canonical` is exactly the derived
  production command, with the GNU-dialect default contraction. `stricter`
  appends `-ffp-contract=fast`. This is a recorded test setting; it does not
  change the production policy.

## Compiler identity (F-2)

- Configure records the compiler path, resolved path, binary SHA256 and
  `--version` banner in `cpu_contraction/plan.json`.
- The builder refuses a compiler whose measured identity differs from the plan.
- Installed verification executes no tool. It compares the receipt with the
  plan identity and the source closure, and checks the commands re-derived from
  the compilation database. A same-path compiler upgrade therefore leaves the
  receipt stale.

## Qualification

- **Single-config Release and Integration:** register
  `V2_Integration_CPUContributionContraction` in `ProductionTestPreflight`.
  These are the only configurations whose flags optimise, so only they can
  make the negative control fuse.
- **Debug, an empty build type and every multi-config generator:** configure
  reports `UNQUALIFIED` and registers no proof. No `-O3` is fabricated, and
  configure never fails for this reason. A multi-config tree is unqualified
  because one receipt binds one configuration's command.
- **Generated-command fixtures (every configuration):** the
  `v2_cpu_contraction_generated_commands` build target is a `v2_unit_gate`
  dependency.
  - It configures fixture projects with the real module, the ISA authority and
    the configured compiler.
  - The cases are AVX2/AVX512 × Release/Integration, Debug, empty, Ninja
    Multi-Config, and a manual ISA flag.
  - It writes a receipt bound to the sources and to the configure-recorded
    compiler identity (`cpu_contraction/compiler.json`).
  - `V2_Unit_CPUContractionProofGeneratedCommands` re-checks every observation
    from that receipt and executes no tool.
- **Tool-free Unit controls (every configuration):**
  `V2_Unit_CPUContractionProofAdmission` covers the proof receipt, identity,
  derivation and the matcher self-test. It also covers the generated-command
  receipt's missing, stale, compiler-upgrade and mutated-observation negatives.

## Scope

This does NOT close B5 globally: corrected-scale helpers, coded leaves,
planning copies and all per-TU/post-link production copies remain unwitnessed.
D0 workspaceFloats COMDAT evidence is unrelated to this contraction proof.

## Local fixture evidence (2026-11-27)

Builder and installed verification were both exercised on standalone fixture
trees. The installed run hid every compiler, objdump, CMake and Ninja binary
(executing them failed). In that run all Release/Integration, Debug and
multi-config tests passed, and missing or stale generated-command receipts
failed.

The fixture project linked the real `OpenMP::OpenMP_CXX`. The only run was in
the `g4-cross-test:avx` ARM container, with
`x86_64-linux-gnu-g++-14 (Ubuntu 14.2.0-4ubuntu2~24.04.1) 14.2.0` and
`x86_64-linux-gnu-objdump`. That is not a canonical image run.

It covered configured AVX2 and AVX512 × Release and Integration (4 trees). Each
tree compiled 4 commands (2 envelopes × 2 settings), and every tree reported
`expert=0`, `native=1`, `negative=1` for every symbol.

Derived `AVX2/canonical` command, Integration build, AVX512-configured tree:

```text
x86_64-linux-gnu-g++-14 -DLLAMINAR_COMPILED_WITH_AVX2=1 -DLLAMINAR_COMPILED_WITH_AVX512=0
  -DLLAMINAR_COMPILED_WITH_NEON=0 -DLLAMINAR_COMPILED_WITH_NEON_DOTPROD=0
  -DLLAMINAR_COMPILED_WITH_NEON_I8MM=0 -DLLAMINAR_COMPILED_WITH_FP16FML=0 -Dcore_EXPORTS
  -I/s/src/v2 -I/s/src -msse4.1 -mavx -mavx2 -mfma -mf16c -mbmi -mbmi2 -mpopcnt
  -O3 -g -DNDEBUG -DLLAMINAR_ENABLE_ASSERTIONS -std=gnu++23 -fPIC -fopenmp
```
