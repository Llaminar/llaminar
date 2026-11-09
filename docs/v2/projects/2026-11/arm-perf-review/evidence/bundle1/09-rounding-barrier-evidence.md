# Idea 9: non-volatile rounding-barrier evidence

**Verdict: REJECT the non-volatile variant (no economic benefit).** The current
`asm volatile("" : "+w"(value))` in `persistRoundedNEON` stays. No defect was found, so no
implementation ticket is needed.

- Tree: ARM tip `a27c7ed4f`.
- Variant: a scratch patch, never committed (`nonvolatile-barrier.patch`, sha256 `67116f52…`). It
  changes the single line `asm volatile` → `asm` in `CPUNativeNEONContributionContract.h:34`.

## 1. Method

1. **Toolchain.** Production compiler g++-14 (Ubuntu 14.2.0-4ubuntu2~24.04.1), aarch64,
   `-march=armv8-a+simd+dotprod -mcpu=generic` (NEON_DOTPROD envelope).
   - Flags are Release-equivalent: the build's own Integration `compile_commands.json` entries with
     `-g`, `-DLLAMINAR_ENABLE_ASSERTIONS` and `-DENABLE_PIPELINE_SNAPSHOTS` removed, which leaves
     `-O3 -DNDEBUG`. The exact flag set is in `toolchain.txt`.
   - Both variants use identical commands. The variant reads a patched copy of `src/` in which
     exactly one file differs (`diff -rq` evidence in the session log).
2. **Scope is total, not sampled.** A preprocessor (`-M`) scan of all 631 `llaminar2_core` TUs
   found 18 TUs whose include closure contains a `persistRoundedNEON` header (`tu-closure.txt`).
   All 18 were compiled in both variants (`compile_variants.py`), with 0 compile failures.
3. **ISA comparison** (`func_compare.py`). Per function (full mangled identity) I compared:
   - normalized disassembly;
   - the FP arithmetic opcode multiset (`fmul/fadd/fsub/fmla/fmls/f*madd/scvtf/fcvt*/frint*`);
   - the instruction count;
   - stack-pointer-relative load/store count, as the spill proxy.
4. **Cycle model.** `llvm-mca` 22.1.8, `-mcpu` neoverse-v2 (Grace), cortex-x925 (GB10 big core) and
   apple-m1.
   - Inputs are the FP-carrying innermost loops (backward-branch spans) of every function whose FP
     multiset or spill count changed (`loops.py`).
   - Interior control flow is removed, so each body is modelled as one straight-line block. This
     approximation is the reason the result is a model, not a timing.

## 2. Results

**Object level**

- 13 of 18 objects have byte-identical `.text`.
- 5 differ: CpuExpertSlotPool, CPUPreparedWeightWriter, KernelFactory, WeightManager,
  PlanningProjectionMeasurement.

**Function level** (unique mangled names, deduplicated across TUs, since template instantiations
are COMDAT-folded at link)

| Metric | Value |
|---|---|
| Functions whose code differs | 118 |
| …of which the FP opcode multiset differs | 4 |
| Total instruction delta (variant − base) | −2 |
| Total stack ld/st delta | +3 (2 functions up, 1 down) |

**What the differences are**

- 114 of the 118 are pure scheduling and register-allocation changes: same FP opcode multiset.
- The 4 FP-multiset changes are all `CPUNativeNEONProjectionTile<2|3|4>::run<NEONMultiScaleOps<…>,
  policy 0, …>` instantiations.
  - In every case `scvtf`, `fmul` and `fmla` change together by the same count, e.g. (3,3,3)→(4,4,4)
    or (6,6,6)→(4,4,4).
  - That is one more or one fewer copy of the per-block convert, scale-multiply, fused-accumulate
    body, i.e. a different unroll or tail-peel.
  - It is not a contraction change: fmul+fadd turning into fmla, or the reverse, would move `fadd`
    against `fmla`, and no `fadd` count changed.
- Two `Tile<4>` instantiations gain 2 stack load/stores each (83→85).

**llvm-mca block reciprocal throughput** (cycles per iteration; matched loops, base → variant;
`mca-summary.txt`)

| Loop | Neoverse-V2 | Cortex-X925 | Apple M1 |
|---|---|---|---|
| f0 loop2 (110→112 insns) | 20.7 → 21.0 | 12.5 → 13.0 | 24.5 → 24.8 |
| f2 loop1 (135→147) | 33.7 → 35.7 (+5.9%) | 26.0 → 26.0 | 46.0 → 47.0 |
| f3 loop2 (132→138) | 32.5 → 33.5 (+3.1%) | 24.0 → 24.0 | 44.0 → 44.5 |
| f4 (all loops) | equal | equal | equal |

f1 was restructured (different loop nesting), so it is not directly comparable. No modelled loop
improves under the variant.

## 3. Interpretation

- **Correctness risk.** The non-volatile form still passes the value through an opaque `"+w"` edge,
  and gcc-14 did not contract across it: no fmul+fadd→fmla changes appeared. The extra freedom it
  grants (hoisting, CSE, re-ordering of the barrier) was used only for unroll and scheduling choices.
  - **Byte identity was not established**, because no cross-variant output comparison was run. A
    rejection does not need it.
  - Adoption would require it, per the idea's own rule: any byte change rejects the variant.
- **Economy.**
  - The theory was that the volatile barrier limits ILP. In practice the compiler used the freedom
    to grow unrolled bodies and spill more.
  - Modelled throughput is equal or up to 5.9% worse on Grace, and equal or slightly worse on X925
    and M1.
  - There is no evidence of the hoisting or ILP benefit the idea hypothesised.
- **Unchanged reasons to keep `volatile`.** Its boundary semantics are robust to future compiler
  versions and to inlining, as the idea's own context notes (libm merge and indexer FMA defects).

## 4. Limits and optional confirmation

- One compiler: gcc-14, the production compiler. Clang was not evaluated.
- One envelope: NEON_DOTPROD. NEON and NEON_I8MM were not compiled.
- No runtime timing and no byte A/B.
- If a confirmation is wanted, the procedure costs about 1 h on the Mac container and needs no GB10
  window:
  1. Relink a variant `libllaminar2_core.so` from the 5 differing core objects.
  2. Build `Perf__CPUNativeQuantised_ThreadScaling` against each variant; header-only instantiations
     in the test TU must also be rebuilt.
  3. Run all formats, with a scratch output-hash print for cross-variant byte identity, alternating
     A/B timing.
- I do not recommend spending it: the static evidence already shows no upside.

## 5. Files

All files are in `/tmp/armqual-b1/09/`.

- `nonvolatile-barrier.patch`: the scratch variant.
- `tu-closure.txt`: the 18 TUs.
- `compile_variants.py`: the compile driver.
- `toolchain.txt`: compiler and flags.
- `text-compare.txt`: per-object `.text` hash.
- `func_compare.py`, `func-compare.json`: the per-function comparison.
- `selected.json`, `loops.py`, `loops/*.s`, `loops/sl/*.s`: loop extraction.
- `mca-summary.txt`: llvm-mca results.
