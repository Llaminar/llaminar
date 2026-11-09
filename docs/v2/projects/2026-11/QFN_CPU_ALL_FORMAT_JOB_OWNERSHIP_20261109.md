# CPU native quantised output-job ownership

Status: explicit diagnostic; production scheduling defaults remain unchanged.
This is a detailed qualification note, not a tuning dashboard.

## Scope and arithmetic

The common output-job scheduler must benefit models using different tensor
formats. `FusedOutputJobOwnership::Contiguous` gives each actual OpenMP worker
a contiguous interval of the existing independent jobs. It preserves native
weight bytes, activation quantisation, row kernels, fixed K partitions,
reduction order, output ownership and the existing partial-storage plan.
It changes no graph, transport or admission policy.

The new control is independent of `VerifierRowsPolicy`. The three-row arithmetic
candidate has a narrower five-format nibble domain; it is not an implementation
for all codebooks. The common ownership control has no source-format whitelist.
The existing canonical registry supplies all 21 formats to its lifecycle tests
and performance panel, including mixed adjacent source formats. Q8_1 coverage
includes prepared-kernel storage and does not advertise a new GGUF input format.

Correctness covers serial-row byte equality, dirty buffers, repeated execution,
large-to-small live prefixes, N tails, explicit/existing OpenMP teams and invalid
invocations rejected before writes. The explicit persistent FFN control also
checks both requested phases through the original planner before its first
write. That validation adds diagnostic work to complete-FFN clocks; individual
phase clocks give the cleaner work-assignment comparison.

## Fixture fidelity

The earlier FFN diagnostics used `std::vector` phase roots without recording
their alignment. Their correctness results stand, but their economy is not a
production-layout certificate. An unaligned root can introduce false sharing;
the old records do not prove that any particular root was misaligned.

Both new controls use the existing production `AlignedVector` for FP32 and Q8
phase roots and independent serial-oracle storage. Live element counts and row
strides remain unchanged. Q8 rows may legitimately begin inside a cache line;
there is no padding to make their start residues zero. Layout records publish
base alignment, exact live bytes, row stride and each expert's start residue,
after timing or profiler disarm. No raw addresses or payload hashes are needed.
The explicit `aligned_phase_roots_v1` revision distinguishes new clocks.

The original FFN owner has a focused `OutputLayoutContract` in its existing
production-preflight registration. The coherent Release build and 17 selected
native/fixture registrations pass with no skips. AVX2 checks here use runtime
selection in an AVX512 build, not a separate pure-AVX2 compilation certificate.

## Matched all-format panel

The original full-FFN owner uses authored H2560/F512, 16 experts and 608 routed
rows (M38 each), with native format matching on gate/up and down. These are
legal common geometries across the registry, not a recorded joint model route.
Both controls explicitly select Pairwise, preserving arithmetic independently
of any future Auto table. The candidate changes ownership for both phases.

Each process uses one real socket's 28 disjoint physical workers, a retained
team, 32 warmup batches and 30 samples of one invocation. Four fresh lifetimes
run Current, Contiguous, Contiguous, Current. Every lifecycle checks all phase
bytes, actual selected policy, layout metadata, unchanged binary/core identity
and complete retirement of the original MPI session helper. Timing suppresses
PerfStats; fresh proof calls occur outside the clocks. No profiler runs overlap.

All 21 formats pass byte equality, route, aligned-layout and clean-retirement
checks in all four lifetimes. Gate/up and down latency improve in both pairs
for every format. Nonpersistent complete FFN also improves in both pairs.
The table reports each pair separately; improvements are latency reductions,
not model-throughput gains.

| Source format | Gate/up reduction, % | Down reduction, % | Complete FFN reduction, % |
|---|---:|---:|---:|
| Q4_0 | 14.39 / 11.77 | 14.54 / 14.30 | 19.29 / 19.48 |
| IQ4_NL | 12.45 / 12.09 | 16.90 / 12.67 | 17.20 / 17.04 |
| IQ4_XS | 9.98 / 9.44 | 8.35 / 9.68 | 19.24 / 21.11 |
| Q4_1 | 16.28 / 15.17 | 15.19 / 14.30 | 21.02 / 21.87 |
| Q4_K | 14.38 / 14.45 | 14.04 / 13.51 | 20.71 / 21.47 |
| Q5_0 | 25.83 / 26.26 | 24.60 / 21.44 | 35.32 / 35.48 |
| Q5_1 | 26.22 / 27.06 | 25.04 / 25.15 | 36.34 / 35.22 |
| Q5_K | 26.24 / 27.31 | 25.67 / 24.48 | 34.85 / 36.95 |
| Q6_K | 13.25 / 13.35 | 8.02 / 8.88 | 16.56 / 13.56 |
| Q3_K | 6.49 / 7.21 | 5.32 / 5.76 | 8.68 / 9.32 |
| Q2_K | 4.95 / 5.50 | 4.66 / 4.75 | 5.70 / 6.50 |
| IQ3_S | 27.76 / 26.85 | 23.57 / 23.60 | 38.73 / 42.02 |
| IQ3_XXS | 28.01 / 27.45 | 22.19 / 22.73 | 34.99 / 34.56 |
| IQ2_S | 5.95 / 6.49 | 6.44 / 5.73 | 8.08 / 8.92 |
| IQ2_XS | 4.25 / 4.67 | 4.73 / 3.77 | 6.47 / 6.74 |
| IQ2_XXS | 26.54 / 26.39 | 21.87 / 24.08 | 35.24 / 36.11 |
| IQ1_S | 23.06 / 26.89 | 24.59 / 20.78 | 34.11 / 34.27 |
| IQ1_M | 3.52 / 3.95 | 3.24 / 3.96 | 4.85 / 5.06 |
| Q8_0 | 26.71 / 27.98 | 24.32 / 23.86 | 35.88 / 34.61 |
| Q8_1 | 25.55 / 23.46 | 24.62 / 22.72 | 34.54 / 34.82 |
| Q8_K | 25.01 / 24.58 | 23.59 / 23.40 | 34.68 / 35.16 |

Persistent FFN improved for 19 formats in both original pairs. Q4_0 and Q4_1
had multi-sample slow bursts in the first Contiguous lifetime, producing median
regressions of 31.74% and 21.90%; both returned to normal within their sample
series. The second Contiguous lifetime improved them by 19.39% and 21.76%.
A separate four-lifetime two-format repeat improved Q4_0 persistent latency by
17.47% / 17.77% and Q4_1 by 17.72% / 19.38%. All observations are retained;
the cause of the transient bursts is not established by these timings. They
must not be discarded or described as a proven format-specific regression.

The alignment-unspecified diagnostics remain under
`/tmp/qfn-cpu-prefill-choice-and-ownership-r1131/all-format-ownership/` and are
not relabeled. Fresh coherent build, 17 focused registrations, full aligned
panel, raw six-phase samples, and bounded persistent repeat belong to
`/tmp/qfn-cpu-all-format-aligned-r1133/`.

## Isolated execution evidence

Twelve original LinuxPerfControl lifetimes pass: three repetitions of each
ownership/phase combination at Q4_K gate/up, Q5_1 down, H2560/F320, E84/R3208
(M38/39), 28 physical workers. Each profiles one warmed production call after
separate byte checks and actual route proof. Profiling is separate from timing;
the wall-clock values below do not replace unprofiled samples.

Median aggregate per-worker counters:

| Phase / ownership | Instructions | Cycles | IPC | L1 load misses | LLC load misses |
|---|---:|---:|---:|---:|---:|
| Gate/up Current | 572.72M | 292.68M | 1.96 | 32.34M | 426,227 |
| Gate/up Contiguous | 574.00M | 244.99M | 2.34 | 32.27M | 394,854 |
| Down Current | 169.89M | 183.50M | 0.93 | 8.67M | 745,819 |
| Down Contiguous | 170.57M | 108.81M | 1.57 | 3.95M | 632,534 |

One Contiguous gate/up profile is slower (347.69M cycles, IPC 1.69); it is
retained with the other two, not discarded. The down profiles consistently
reduce cache misses and cycles. These aggregate counters support better
execution/cache locality; they do not isolate every stall or prove a cause
for the occasional slow burst.

Read-only ISA review in the preceding coherent build confirms integer chunk
assignment and no dynamic/atomic scheduler. Representative Q5, compact Q3/IQ2
hot bodies have no stack-relative SIMD instructions. Source-factor/Q6 paths
retain existing stack decoder constants/scalars; no blanket zero-stack claim
is made. The ownership change does not replace any arithmetic microkernel.
Evidence: `profiles/` under r1133 and
`/tmp/qfn-cpu-prefill-choice-and-ownership-r1131/isa/NOTE.md`.

## Exact-choice qualification and remaining work

Fresh original exact-batch collection at Q4_K N320/K2560, M64, 28 workers passes
all three choices and mixed-singleton holdouts. The small training bundle keeps
Pairwise: 153.986 microseconds versus 159.267 for OrderedThreeRows and 208.547
for WideRows. The candidate's large-bundle wins do not establish a universal
row-policy winner. The selected baseline has separate native Linux perf
evidence. No generated table was installed from this diagnostic.

Evidence: `/tmp/qfn-cpu-ordered-three-exact-r1132/`. Completed compatible sweep
observations stay retained; no broad sweep was restarted.

Before any default promotion, qualify mixed row counts and policy combinations,
repeat aligned candidate/baseline profiling and measure the real hybrid model.
The last model result remains 285.983 prefill / 40.013 benchmark decode tok/s
(39.857 after-prefill). These component tests do not meet the 500/40 target.
