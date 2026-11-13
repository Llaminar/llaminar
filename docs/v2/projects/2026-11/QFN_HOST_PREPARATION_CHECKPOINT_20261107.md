# QFN host preparation: focused checkpoint

This is a qualification record, not a tuning dashboard. It checkpoints three
bounded host changes; complete source-admission, numerical and Release
qualification remain separate.

## Implementation

- Prepared GEMM adoption indexes existing entries by source name/device and
  tensor/device instead of scanning the entire model store after an ID miss.
  The indices retain keys only. Full source, slice, precision, kind and lifetime
  checks still consult the original entry under the existing mutex. Replacement
  and teardown update both indices. The regression also catches the former
  acceptance of an explicitly incompatible prepared kind.
- Expert registry hashing combines its ordered identity fields directly and
  avoids constructing formatted device names. Key equality remains unchanged.
  These are metadata map keys, not hashes of weights, sources or cache blobs.
- GPU dispatch tickets use the byte-zero fresh slices already supplied by the
  mapped arena. Binding no longer clears the complete payload again. CPU
  clearing, native backing initialization/first touch, header initialization and
  publication ordering remain unchanged.

No precision, kernel arithmetic, transport, graph inventory or runtime policy
changes are included. Standard-container control metadata retains its existing
allocation boundary; this change does not claim exact PMA pricing for that
metadata or introduce another physical-memory ledger.

## Focused validation

The coherent Integration checkout compiled the owning targets with the configured
CUDA SM80/86/89/90 and HIP gfx906/gfx1100 target set and spill guards enabled.
Twenty selected host CTest entries and six CUDA/ROCm native entries passed with
no skips and clean driver windows. This is **not** a complete preflight result.

The three new production-preflight registrations are
`V2_Integration_PreparedWeightAdoptionIndexedLifecycle`,
`V2_Integration_ExpertGemmRegistryHashDistribution` and
`V2_Integration_MappedDispatchTicketFreshZeroAllocation`.
Coverage includes replacement, reset, alias lifetime, wrong-kind/device rejection,
hash distribution over varied participant counts, dirty neighboring arena slices,
growth, publication reset and avoiding a second payload write. Original-code
controls fail the new wrong-kind, distribution and redundant-write assertions.
Existing native ticket ownership/publication tests provide actual device coverage.

All changed production implementation files and owning test bodies in this
checkpoint are byte-identical to those tested in that Integration checkout.
`TransferEngine.h` contributes documentation only; the Integration checkout also
contains unrelated metadata-copy declarations. CTest registration is relocated
where necessary in this feature branch. The whole Integration checkout contains
additional work, so these results do not certify this exact feature tree as a
complete model or release build.

Local evidence: `/tmp/qfn-host-preparation-integration-r863/lookup-memory/`,
`/tmp/qfn-prepared-adoption-lookup-r862/`,
`/tmp/qfn-expert-registry-hash-r861/` and
`/tmp/qfn-mapped-ticket-fresh-zero-r861/freeze-final/`.

## Performance evidence and limits

Private host microbenchmarks call the actual production implementations. Three
interleaved fresh processes per candidate, pinned to CPU 0 with one OpenMP thread,
measured 256 missing-ID adoptions against 76,976 prepared entries: median
666.057 ms before and 0.126 ms after. Compatible aliases improved from 10.852 ms
to 0.323 ms. Registration is excluded; these figures demonstrate lookup scaling,
not end-to-end model speed.

For the measured four-GPU registry geometry (2,352 keys and 4,704 queries),
successful lookup improved from 294.100 us to 194.762 us and nonresident lookup
from 451.688 us to 367.965 us. Not every tested shape improved. A slower byte-wise
hash candidate was rejected.

The original four-MI50 RAM source-admission/recreation case retained all 49
materializations, 2,351 compilation units and 65,433 captured stages per device.
Observed first-family preparation was 68.473 seconds and recreation 58.935
seconds, versus the preceding 83.317/68.303-second observation. Native SDK timing
also varied substantially, so the complete difference is not attributed to
these code changes. Model source assertions passed, but the **overall run is
red** because the driver recorded an IH2 ring overflow during recreation.
This is neither a clean performance certificate nor an inference accuracy test.
Evidence: `/tmp/qfn-host-preparation-model-r864/`.

A subsequent diagnostic CPU profile shows existing OpenMP weight packing now
dominates the separate model-loading phase; startup uses 28 physical-core workers.
Graph preparation still spends substantial CPU time in mapped allocation/clearing
and expert registry lookup. Anonymous-memory experiments suggest large clears
can benefit from parallelism, but do not qualify native mapped memory or a new
worker-team policy. No speculative OpenMP loop was installed.

## Remaining work

Characterize the repeated IH2 warning with native allocation/registration and
driver evidence, reduce its trigger, and install a focused structural regression.
The warning also occurred before this checkpoint and on different active MI50s;
that does not establish a hardware or software cause. Preserve failing receipts.

Keep further graph preparation changes bounded. Sharing dispatch storage between
alternate captured graphs requires the original family and CPU-return lifetime
authority; a geometry cache is insufficient. Return to matched Release ngram
cache/read-ahead overlap tuning after preparation is satisfactory. No new default
cache policy, complete topology qualification or performance victory is claimed
by this checkpoint.
