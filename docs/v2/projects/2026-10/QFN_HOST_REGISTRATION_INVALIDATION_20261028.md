# QFN registered host-page invalidation investigation

The accepted ROCm TP4 dynamic controller is checkpointed at `078f14c88`.
Its earlier OpenCode session remains an application pass with a red driver
gate. This fix now passes a fresh complete session; it does not change controller
economics or reinterpret the original failure as a pass.

## Reproduction and attribution

The diagnostic repeats the original first app request, including its exact
serialized bytes, against the same Release serving configuration: ROCm TP4,
RAM ngrams, dynamic MTP auto-initialized at 5 with capacity 15, context 86,336,
16 GiB RAM prefix capacity per participant and 32 GiB shared disk capacity.
It is an instrumented, single-request diagnostic, not a throughput comparison
or a substitute for the full ten-phase app.

The sealed run is
`parity-results/qwen-flash-next/first-app-userptr-diagnostic-r1`.
Paired kernel and HIP observers cover native startup through normal retirement.
They report no lost records, helper errors or incomplete retirement. Kernel
offsets are authenticated against the loaded Ubuntu `6.14.0-37-generic` AMDGPU
module, build ID `f4cd17aab5cb859073d4cf559890d3f3124cedac`.
Neither model/cache payloads nor native binary payloads are hashed.

The independent analysis is
`/tmp/qfn-userptr-event-analysis-r1-private/first-request-final/result.json`:

| Observation | Count | Interpretation |
|---|---:|---|
| Registered/unregistered USERPTR lifetimes | 2,249 each | Native lifetime evidence; not all are direct HIP registration calls. |
| Invalidations and matching evictions | 7,699 each | Live process mappings were repeatedly invalidated. |
| `kcompactd1` migrations | 7,688 | All invalidate one 4 KiB page; 6,511 during startup, 1,177 during the request. |
| `khugepaged` collapses | 11 | Seven during startup, three during the request and one during shutdown. |
| Paired restore-worker invocations | 2,859 | Owner/process evidence; the restore probe does not identify an exact restored range. |

The first collapse spans a 2 MiB interior of a live 2.5 MiB registration.
Its stack contains `collapse_huge_page`, `hpage_collapse_scan_pmd` and
`khugepaged`. Migration stacks instead contain `try_to_migrate_one`,
`rmap_walk_anon`, `migrate_folio_unmap`, `migrate_pages`, `compact_zone` and
`kcompactd`. All migration events join successful HIP registration lifetimes;
7,664 occur in 2.5 MiB registrations. The backing allocator is still unknown
where no observed allocation call overlaps the registration. A late `/proc/maps`
snapshot is not proof of an earlier allocation's identity.

Twenty-four migration events touch a first or last base page already protected
by the former edge-only policy. Consequently, `MADV_NOHUGEPAGE` cannot be claimed
to make these caller-owned pages immovable. No host compaction setting or driver
configuration has been changed.

The original workqueue warning did not recur in this short diagnostic. That is
not an immunity test: printk warnings are rate-limited. The actual invalidation
and restore events above establish the mechanism without relying on a warning.

## Confirmed fix and regressions

`HostRegistrationPageRange` replaces the former edge-only policy. Before either
ROCm caller-owned registration path enters HIP, one checked advice covers the
complete outward-rounded base-page span. Logical and rounded-end overflow fail
before entering the OS. Neighboring wholly unregistered pages retain their
huge-page eligibility. Advice remains installed after unregister because a page
can still belong to another registration. Native HIP-owned allocation paths are
unchanged.

Device-free tests check every boundary offset, full VMA coverage, overflow,
payload preservation and idempotence. A negative control executes the former
two-edge advice and proves that it leaves the interior eligible for collapse.
CUDA and ROCm native tests retain a captured graph, change source bytes between
three replays, reclaim an unrelated neighbor and compare all 31 MiB. ROCm also
checks the interior VMA policy and its persistence after unregister.

Explicit production-preflight registrations are
`V2_Integration_HostRegistrationPageRange` and
`V2_Integration_RegisteredHostFullSpan_{CUDA,ROCm}`. The complete existing
`GPUHostRegistrationLifetime` fixtures also exercise the extended case.

Current evidence:

- Focused six CTests pass, with a complete clean driver window:
  `/tmp/qfn-host-registration-full-span-focused-r1/result.json`.
- Complete Unit: 786 pass, zero failures or skips:
  `/tmp/qfn-host-registration-full-span-complete-unit-r1.xml`.
- Release and both complete Integration gate target builds pass. The fresh
  source-before/source-after Release freshness receipt is
  `/tmp/qfn-host-registration-full-span-release-receipt-r1/result.json`.

The fresh full-span diagnostic completed normally at
`parity-results/qwen-flash-next/first-app-userptr-named-fullspan-r3`.
Its independent complete-trace analysis is
`/tmp/qfn-userptr-named-fullspan-r3-analysis/result.json`. All 2,249 registered
lifetimes have matching retirements, with **zero invalidations, evictions or
restore-worker invocations**. The runtime, native-output and archive audits,
strict driver window and observer retirement all pass. Named-weight observers
were first validated against actual CPU and GPU positive controls. This is
evidence for the completed workload, not a proof that every registered base
page is immovable or a controlled throughput comparison.

The same trace joins 196 registrations of 2,621,440 bytes to the actual returned
prepared-weight tensor pointers: 49 unique `blk.*.ffn_gate_inp.weight` names,
each materialized for four devices with `MoERouter` role 14. These are observed
pointer/lifetime joins, not ownership guesses from a late address-map snapshot.
Other registration backing allocators remain unknown where no allocation probe
observed their creation.

Complete current-source prerequisites passed: **2,981 pass, 29 explicit hardware
skips, zero failures**, including all 786 Unit tests. The immutable canonical
receipt is `/tmp/qfn-host-registration-full-span-prerequisites-r1/prerequisites.json`.

The source metadata exporter subsequently required one unrelated executable
relink. Canonical whole-build reuse correctly remains rejected. A separate
reviewed **local-only exporter-delta continuation** proves the complete tested
Ninja dependency/runtime closure, exact 3,010-test inventory, source and build
inputs unchanged, with exactly that one excluded exporter link. It preserves
the original receipt and completion time; it is not a skipped suite, refreshed
gate, or CI/image/release certificate. Bounded summary:
`/tmp/qfn-preflight-exporter-delta-review-r1-private/SUMMARY.json`.

Fresh Off and Serving-Dynamic HTTP lifetimes pass at
`parity-results/qwen-flash-next/a1-fullspan-local-http-r2`. Each executes the
canonical 384-token fresh/full/partial/full prefix sequence. Exact serial token
comparison, actual retained branch/ticket/depth evidence, runtime/resources,
strict passive driver interval and normal owned retirement pass. Historical
fixed-depth HTTP receipts remain historical; this pair does not refresh the
complete nine-arm qualification.

Fresh 512/256 repetition observations give median decode 101.366691 tok/s for
accepted Dynamic, 43.511637 for fresh Off and 100.098175 for fresh Castagna fixed6:
+1.267272% Dynamic/fork. All nine measured streams are exactly 256 × token 1228;
both rates use 255 after-prefill tokens. **The native benchmark ignores stop-token
termination, while Castagna permits EOS.** Its flag changes termination rather
than masking EOS logits. Frozen packet natural-EOS wording overclaimed the
native contract. These results establish the matched observed finite trajectory,
not identical EOS semantics, general economics or statistical superiority.
The dated Castagna baseline is unchanged. Full bounded report:
`/tmp/qfn-fresh-head-to-head-fullspan-r1-private/report.json`.

The fresh full ten-phase app passes at
`parity-results/qwen-flash-next/a1-full-opencode-fullspan-local-r2`: **110 tool
attempts, zero tool errors, 85 requests and one client context compaction** in
1,950.81 seconds. All ten phases, protocol/tool coverage, independent webapp
acceptance and the independently retained 45-test application suite pass.
The native byte audit joins all 85 responses and all 110 calls with zero changes.
Current format-5 archive, actual retained graph/ticket/MTP/runtime/resource and
loader audits pass. The strict passive driver window has zero findings; native
exit is normal and unforced with no survivors. The existing owner reclaimed its
3,002 archive files after metadata evidence, reading zero payload bytes; the
retained application workspace remains available for review.

Before/after local admission sidecars pass with the original gate/source/Release
proof intact and canonical reuse explicitly rejected. Independent bounded review:
`/tmp/qfn-fullspan-final-review-r1-private/completed-app-review.json`.
The session uses normal serving EOS semantics and the PMA-admitted maximum
86,336-token context; 86,400 is capacity-rejected. It has 32,768 output capacity,
28 workers, 16 GiB RAM prefix per participant and 32 GiB shared disk prefix.
No forensic observer or profiler is in the throughput window. This proves the
completed workload, not universal page immovability or performance superiority
across unmeasured workloads/topologies. Earlier aggregate reds remain unchanged.

The separate proposal to retire one-time upload registrations earlier remains
on hold. Prepared host weights must remain available for rematerialization,
and captured DMA consumers must retain their registrations.

## Diagnostic scope correction

The first diagnostic inherited a validator intended for a multi-request HTTP
arm. Its aggregate result therefore remains red: a fresh request cannot prove
later prefix restores, and aggregate capture counters combine distinct cold
physical families rather than identifying repeat capture of one executable.
The separately retained single-request assessment validates the actual hosted
parent, ticket, attention, transfer and physical-family evidence. It does not
certify prefix restoration or the full app and does not overwrite the original
receipt. A fresh successor packet owns that narrower declared scope.
