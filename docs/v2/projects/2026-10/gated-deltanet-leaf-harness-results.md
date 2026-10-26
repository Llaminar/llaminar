# Gated DeltaNet leaf elimination: retained harness and exact compact variant

Date: 2026-10-24.

Follow-up: the authorized CPU arithmetic change, model qualification, and the
individual/combined backward-query experiments are recorded in
[the shared-readout report](gated-deltanet-shared-readout-and-backward-query-results.md).
The implementation boundary and measurements below describe the earlier slice.

The adjusted-query proposal passes exact-arithmetic reference checks but fails
Llaminar's existing serial-row byte contract. A different compact-state variant
passes native CPU byte checks: keep the original correction calculation and
rounding, suppress dense leaf-state publication, retain the exact correction
through acceptance, and fuse the accepted leaf's materialization into the next
verification. This preserves a memory-saving opportunity on branching trees.
It does **not** preserve the original reduction in checkpoint projections.

Implementation and reproduction commands:
[research harness](../../../../tests/v2/research/gdn_leaf_elimination/README.md).
No production inference implementation was changed or enabled.

## Why the original projection saving fails

Consider one value and one key component with FP32 operands
`k=q=decay=beta=1` and `v=2^-25`.

| Parent state | Original serial output | Adjusted-query output |
|---:|---:|---:|
| 0 | `2^-25` | `2^-25` |
| 1 | `0` | `2^-25` |

For parent 1, `fl(v - 1)` rounds to `-1`, so the updated state is zero.
For parent 0, the small value survives. The adjusted query is zero in both
cases, so both checkpoint actions are identically zero. A function consuming
only that adjusted action and the same raw operands cannot distinguish these
two serial results. This is a specific information-loss argument against the
proposed single-action realization, not a lower bound on every possible GDN
optimization.

The native CPU kernel reproduces the counterexample. Both explicit fused FP32
and FP64 adjusted-query evaluation fail to recover its serial bytes. Ordinary
non-endpoint gate inputs also differ. The all-node triangular formulation
likewise reassociates native state rounding; keeping that solve while changing
only the leaf formula is insufficient.

The retained reference sweep contains 980 cases: 140 exact rational cases and
420 each in FP64 and FP32. It covers empty/singleton, chain, star, ternary,
irregular and early-leaf trees; ordinary, zero, endpoint, tiny and nearly
parallel-key regimes; two seeds; both query variants; and four rounds.
Floating geometries are `(7,5)`, `(64,64)` and `(128,128)`; exact cases use
`(3,2)` and smaller trees. Every accepted endpoint is reconstructed and
continued, including no extension and explicit flushes. The accepted resolved
history has capacity three; additional focused tests exercise capacities
zero and one and changing tree sizes.

All 140 exact cases pass, including 3,040 candidate vectors and 3,600 endpoint
checks. For the primary, leaf-only FP32 variant:

| Compared surface | Different FP32 scalars / compared | Maximum absolute difference |
|---|---:|---:|
| Candidate outputs | 680,521 / 803,760 | 8.94e-8 |
| Accepted states | 11,230,717 / 89,445,400 | 2.98e-7 |
| Continued outputs | 719,111 / 858,920 | 1.04e-7 |

These errors are diagnostics, not acceptance tolerances. The strict command
exits 1 and records `byte_equivalence_observed: false`. The experiment is a new
retained workload, not a reconstruction of the original untracked JavaScript
conversation harness.

## Byte-exact memory-saving realization

For every value column, the new native trial executes the original operations:

1. Round each decayed parent-state element to FP32.
2. Accumulate the key projection in the production increasing-key FMA order.
3. Compute the original rounded correction `(v - projection) * beta`.
4. Form each updated state element with the original FMA, then accumulate its
   query readout in the original order.

An internal node writes its dense state for descendants. A leaf passes each
updated element directly to the readout without writing its dense state.
Sibling rows share parent loads in a four-row/eight-value AVX2 tile, with an
independent ordered accumulator for each row. The correction vectors remain
available until acceptance; they are not eliminated.

An accepted leaf retains one immutable dense parent and
`(decay, key, exact correction)`. Materializing that record reproduces the
original updated state element by element; it performs no new key projection.
The next round fuses this materialization into the first new key pass and
publishes the resolved parent for subsequent sibling groups. An empty pass
does not materialize it. A final dense consumer explicitly pays the update.
The reference commit copies the selected parent and record so later candidate
buffer reuse cannot invalidate their ownership.

This design deliberately retains dense internal states. Replacing them with
the proposal's algebraically factored actions would reintroduce the rounding
failure. It is a CPU research implementation, not a GPU lifecycle or captured
generation policy.

For the 39-node ternary example, a verification with one old pending update
writes 12 internal dense states plus one resolved parent, versus 39 candidate
dense states. Correction records and accepted-parent retention are additional
storage; this is not a total bandwidth or allocator-memory claim. The two key
and query projections per candidate remain. For a chain, one old pending
materialization cancels the single-leaf state-publication saving in steady
state. No general gain is claimed for an ordinary linear MTP workload.

## Native byte evidence

The same AVX2 trial was checked against the real production CPU M=1 oracle
under both AVX2 and AVX-512 runtime selections. Each native oracle call also
authenticates its complete state and output against an independent FMA control;
a failed control aborts the experiment. The trials use supplied core operands
with query scaling and gates computed identically to the native kernel.
Branch-local convolution and model-produced operands are outside this scope.

Each runtime selection passed 84 cases: three geometries, seven tree shapes,
four gate regimes, four consecutive rounds, every accepted endpoint, three
continued children per endpoint and terminal state publication. Per selection:

| Surface | FP32 scalars compared | Byte mismatches |
|---|---:|---:|
| Candidate outputs | 339,456 | 0 |
| Accepted states | 35,912,448 | 0 |
| Continued outputs | 1,088,256 | 0 |
| Continued/terminal dense states | 110,043,648 | 0 |
| Live-row/retained-update replay | 84,912 | 0 |

The replay inventory includes every M from 1 through 16, then 31 and 63, plus
63 → 2 → 0 → 16 → 1 → 63 transitions. Guards and inactive capacity stay
untouched. Pending records survive dirty output/correction-bank reuse. The
minimal information-loss witness above also passes for the compact candidate.

All three final standalone CTest checks passed; the reference suite contains
10 focused tests and took about 1.4 seconds. This is focused research evidence,
not a complete repository Unit/preflight or model certificate.

## Complete two-round component timing

The timing fixture uses a Release bridge and Release production core, one
thread pinned to CPU 0 (socket/NUMA node 0), FP32 `(dk,dv)=(128,128)`, and
cache-hot synthetic operands. It verifies two star-shaped candidate sets,
accepts a fixed row after each, and publishes a final dense state. Compact
timing includes fused pending resolution and the terminal flush. The dense
control uses the same four-row arithmetic and a cache-local row-first snapshot
schedule. No allocation or input reset occurs inside native timing.

Each process warmed each candidate for at least 100 ms and collected 21
alternating A/B and B/A sample pairs, with 64 complete transactions per sample.
Three fresh process invocations used the identical compiled bridge.
A final invocation used a rebuilt bridge bound to an immutable oracle snapshot;
it repeated the byte checks and complete timing workload.

| Leaves per round | Initial three-process range, dense/compact | Frozen-oracle final run |
|---:|---:|---:|
| 1 | 0.91–1.19x | 0.99x |
| 4 | 1.26–1.53x | 1.46x |
| 19 | 2.04–2.75x | 2.96x |
| 39 | 2.07–2.91x | 2.90x |

Every timed candidate's outputs and final state remained byte identical to the
native serial oracle. Absolute timings varied by more than 2x between process
invocations, so the results support further investigation rather than a stable
performance certificate. The single-leaf control has no consistent win.
These are comparisons with a matched dense-state prototype, not evidence of a
speedup over optimized TreeWY, Bole or ReplaySSM output-only implementations,
nor a whole-model Llaminar speedup.

Assembly inspection found no vector stack references in the exported compact,
dense-control and pending-resolution routines. The ordinary compact four-row
loops use YMM0 through YMM14. Hardware `perf` counters were unavailable because
the host reports `perf_event_paranoid=4`; no host setting was changed. IPC,
cache-miss attribution, physical-core scaling and NUMA scaling remain unproven.

## Identity and next qualification boundary

The isolated source worktree is `feat/gdn-speculative-leaf-harness`, based on
develop commit `ef1f54b09`. Native diagnostics used the active workspace's
existing production binaries and matching headers, including its Release
`build_v2_qfn_release` core. The CPU recurrence implementation matches the
develop source; its SHA256 is
`77a7de7b3e5dc7278c21c4710a94786cafca23235d7572ae2bd1b70b49b7b382`.
Reports retain the selected core, bridge and driver digests. Concurrent source
and binary changes in the active workspace prompted a final frozen-oracle run:
the core DSO and its 37-file source/header dependency closure were copied and
hashed, and `ldd` verified that the bridge resolved the captured DSO. Both ISA
selections, all three standalone tests, timing and assembly inspection were
repeated against this snapshot. Its core SHA256 is
`1611d4b5f9986efe819de3a17c17f78f1c2cb110e17d2da6406d4211f998eac0`;
the final bridge SHA256 is
`fd583d4a373fc57cfcbead4fd829f7b0fbe9b65287b1c33ad1e90f6c4cbb9ffb`.
The earlier timing reports identify their earlier core digest separately.
This procedure does not authenticate a newly rebuilt complete source checkout.

Local raw artifacts are under ignored `parity-results/gdn-leaf-harness/`:
`reference.json`, `exact-frozen-avx2.json`, `exact-frozen-avx512.json`,
`oracle-snapshot.json`, `isa-frozen/`, `lifecycle-frozen.json`, and
`lifecycle-release-final-{1,2,3}.json`. Other earlier timing trials are not the
final control comparison. Reproduction uses the retained harness and the
commands in its README; generated artifact directories do not belong in source
commits.

The original adjusted-query projection reduction is not admissible under the
current byte contract. The compact-state variant is a viable CPU arithmetic
candidate for branching verification. Production adoption still needs an
explicit tree/controller design, authority-owned stable state storage,
backend-specific reduction-order proofs, captured publication/reset and
prefix-cache tests, all supported CPU execution regimes, CUDA/ROCm kernels,
production preflight, model accuracy and acceptance panels, and measured
Release serving economics. No numerical threshold or serving reference was
relaxed to obtain these results.
