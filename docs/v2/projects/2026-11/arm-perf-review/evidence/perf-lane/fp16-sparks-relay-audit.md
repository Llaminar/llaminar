# SPARKS FP16 relay audit — gate1 archival COMPLETE (2026-11-10)

Read-only codegen audit (not a candidate experiment or runtime/perf result).
Source basis: 777f7a52a (retained in llaminar-perf-base777 worktree; the
station's main checkout owned by another worker). Reviewer #150 released the
half-cell v2.1 checkpoint (test-only implementation authorized second after
idea31). Native FMLAL/ISA/scratch = later separately approved slice.

## Verdict
Corrected rel-777f7a52a receipt + real dotRows bodies supersede the
wrapper-only limitation. Gate1 archival COMPLETE: actual SPARKS arithmetic
bodies, relocations, full argv/literal-includes/compiler/source/object
association verified. Scalar exact-256-lane implementation corroborated on
the target; gap = native NEON half-product vectorization (not a missing
arithmetic tree). The 981-byte includes supplement closes the abbreviated
gap. Gate1 does NOT authorize a native experiment or prove a perf win;
final-binary/runtime identity remains a separate later gate.

## Corrected 777f relay (SHAs, producer, flags)
- fp16-gate1-bound-777f.txt 1287 B faa39913...
- fp16-dotrows-E2-dr-777f.txt 1306124 B 1c32fbdb...
- fp16-gate1-argv-777f.txt 1252 B afa812ee...
- fp16-gate1-includes-777f.txt 981 B c1fbadb2...
- Producer: .80 GB10/X925 rel-777f7a52a-neon_i8mm GCC14.2.0 Release;
  object SHA e01c7c1a...; KernelFactory.cpp SHA a9dc843f... (equals
  immutable local 777f source; raw object not locally rehashed — binding =
  producer receipt + relocation-aware disasm, not pathname).
- Flags: -mcpu=generic -march=armv8-a+simd+dotprod+i8mm -O3 -DNDEBUG
  -std=gnu++23 -fPIC -fopenmp; DEFINES complete (HAVE_ONEDNN; AVX2/512=0;
  NEON/DOTPROD/I8MM=1; LLAMINAR_DISABLE_STACKTRACE; llaminar2_core_EXPORTS);
  includes = full build.ninja:6312 line (8 -I + 3 -isystem).

## Actual arithmetic-body observations (corrected E2 dump)
3 nonempty dotRows<4> bodies + real AArch64 relocations. Each HalfRNE body:
1 static scalar FMADD + 1 reduction FADD; 0 vector FMLA/FMLAL/FMLAL2/FCVT/
FCVTL; conversion integer ops; indexing explicitly retains k&0xff
(canonical 256-lane traversal exists).
| Variant | parsed | FMADD | FADD | BL | relocs |
| HalfRNE T/F | 198 | 1 | 1 | 7 | 21 |
| HalfRNE T/T | 283 | 1 | 1 | 7 | 21 |
| native F/F (unrelated) | 201 | 4 | 1 | 7 | 21 |
Sender totals 202/287/205 = parser-boundary, not timing. Aggregate 6 FMADD
+ 3 FADD includes the unrelated native body; NEVER 'six FMADD per m4 vs one
FMLAL'. Static opcode sites, not per-K costs or measured speedup.

## Superseded wrapper bundle (corrections mandatory, still pass
## sha256sum -c)
d1 bd644e4a..., d2 7a3c17ff..., d3 71e59fc0..., fp16 ef4f1387... (old
f9f190059): one wrapper/dispatch body each, empty dotRows headings, no
relocations; d1/d2 8 FMUL+4 FADD+8 FMADD = completion sites; d3 = 122 BL
(dispatch), no FP arithmetic. 1f0/1fc/208 = alpha*reduced-dot/bias/beta-FMA/
publication; 4 completed output regs != 4 increasing-K accumulators; bl0 =
unresolved placeholder (not recursion); empty headings != bodies. WITHDRAWN
claims stay withdrawn ('0 FMA', '8 FMUL+4 FADD per hot m4', 'no 256-lane
structure', '81 self-calls/no dotRows in relocs').

## Scope / next action
No more FP16 evidence collection needed (gate1 complete). No native/scratch
experiments now. Finish idea31's .80 economy + explicit disposition, then
the approved independent half performance cell/smoke on a separate 777f
branch; obtain its .80 baseline/economy before separately authorizing any
native leaf. Preserve exact-256-lane arithmetic, test-owned oracle/completion,
successful public-return checks, C0 protocol, immutable/guard checks,
requested-vs-observed identity. Gate1 codegen evidence proves neither runtime
selection nor economical FMLAL implementation.
