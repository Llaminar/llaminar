# Idea 9 — Non-volatile rounding barrier evidence

Status: **Open** (experiment candidate; evidence required — no MCA/PMU
comparison exists at the reviewed head).

## Context

`persistRoundedNEON` (`CPUNativeNEONContributionContract.h`) uses
`asm volatile("" : "+w"(value))` as a rounding barrier to prevent unwanted FP
contraction around the declared per-block multiply/round program. Notes from
the review and the implementer:

- The empty volatile asm emits no instruction; it constrains compiler
  scheduling, which can either limit ILP or prevent hoisting/spills — direction
  unknown without evidence.
- Global/TU `-ffp-contract=off` is **outside** the current numerical policy:
  boundaries must survive inlining while intentional FMA stays explicit, and
  existing local directives in one file do not govern the whole call graph.
  (The libm cos/sin merge and indexer FMA defects are evidence that source
  statements alone are insufficient.)
- A non-volatile identity barrier is a legitimate candidate but unproven; needs
  actual compiler/byte/resource evidence before adoption.

## Proposed experiment (measurement first)

1. Add a non-volatile `asm("" : "+w"(v))` variant behind a compile-time
   selection; build both in the same Release toolchain.
2. Compare: emitted ISA (fma vs mul+add boundaries in the exact programs),
   register pressure/spills, `llvm-mca`/`perf` cycles for one isolated kernel.
3. Keep scalar/vector byte auth vs the current program; any byte change is a
   rejection of the variant regardless of speed.

## Exactness / risk

- Contract-sensitive: the barrier exists to preserve declared per-block
  rounding; any scheduling freedom that changes emitted rounding changes bytes.
- Apply only after the evidence and with byte auth in the same commit.

## Scope

ARM NEON only (`persistRoundedNEON`); no x86 change implied.
