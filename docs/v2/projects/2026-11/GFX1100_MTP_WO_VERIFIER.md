# gfx1100 attention-output verifier projection

The actual `GEMMStage` Wo fixture passes all 126 source/geometry cases in Release
and all 126 registered Integration replays. Its 1,890-cell complete-operation
panel identifies 761 conservative WMMA candidates across all 21 source formats.
Their geometric gain over current Automatic is 1.216389×, range
1.050248–1.819402×. Installed Automatic qualification is the next step.

## Operation and geometry

The production `QwenGraphBase::addWoProjection` records ordinary SingleQ8,
alpha1/beta0, transpose_B=false, the original prepared reference and verifier
rows. The fixture executes that actual stage, including activation quantization,
projection, canonical reduction and output publication. The preceding attention
output gate and the following collective are outside this explicit boundary.
TwoTermQ8 and gated-norm projections are separate arithmetic operations.

Model weight metadata supplies these dimensions. Dense gated Q has width12288;
Wo consumes attention width6144. Reusing gated-Q width for Wo would measure the
wrong operation.

| Geometry | N | K | Selected seed keys | Selected geometric gain |
|---|---:|---:|---:|---:|
| Dense full |5120|6144|118|1.277850×|
| Dense TP2-sized local |5120|3072|123|1.214919×|
| Dense TP4-sized local |5120|1536|109|1.153243×|
| MoE full |2048|4096|180|1.330048×|
| MoE TP2-sized local |2048|2048|141|1.156238×|
| MoE TP4-sized local |2048|1024|90|1.103260×|

The TP-sized cases use synthetic local matrices with those exact extents. They
qualify local projection arithmetic and economy; they do not prove source-slice
loading, original global column offsets or allreduce. Production sharding owns
an authoritative `WeightSliceSpec` for K-column slicing. A legacy TensorSlice
row-parallel tag alone is not evidence of that axis or original source binding.

Every canonical format keeps its original identity. IQ4_XS source4/superblock
prepares physical24; Q8_1/source20 and Q8_K/source21 both prepare physical19 but
remain distinct qualification keys.

## Capture and numerical contract

Three controls—Dot4, current Automatic and forced WMMA—and a separate M1 oracle
own independent prepared stores, kernels, payloads, workspaces and outputs. All
controls share the original host weight values. Each actual stage binds its
persistent workspace and exact stream before capture; the dependency ledger
retains original input/count ownership and output publication.

Every physical M2–16 replays full, one, empty, regrown, repeated and every live
prefix against independent actual M1 rows. Scratch and output start dirty;
inactive output capacity retains poison. Reset preserves captured topology.
Separate counted-Q8 and native-producer omissions must break nonempty output;
restoration must recover exact bytes immediately.

Captured argument checks authenticate the original prepared payload/scale/min
banks, input/output/Q8 addresses, source codebook, rows, N/K and grid. Native
partial producers and reducers are independently enumerated and joined by exact
partial address and partition count. All six geometries retain canonical P16;
K32 spans are respectively12/6/3/8/4/2. Numeric P/span/publication values are
exported for every native packet and independently checked by the collector.
The native producer has 80 VGPRs and7,040 LDS bytes, with zero reported private
storage. Dedicated counter qualification follows actual-Automatic installation.

## Seed economy and dispatch boundary

The Release panel contains 126 cases, 1,890 source/geometry/M cells and 58,590
rotated forward/reverse three-arm samples. Each internal-event interval retains
eight complete projections. Preparation, poison, observation and M1 reference
execution stay outside the timed interval. Unprofiled collection requires AUTO
power before and after the run.

Blanket WMMA wins 1,079 medians but only878 complete31-pair comparisons; its
all-cell geometric ratio is1.009067×. The generator independently recomputes all
medians from finite-positive raw triples and admits only keys winning every pair
with at least5% median gain. It retains761 keys and nonmonotone holes. For
example, Q4_0/full dense qualifies only M16, while Q5_0/MoE full qualifies M5–16.
Some source/geometry pairs have no qualifying width and retain ordered work.

These observations qualify a candidate table. Actual-Automatic selection must
retain original source, Projection operation, exact-hosted extent and canonical
serial policy. SwiGLU, fused subgroups, selected LM-head rows and unknown shapes
cannot borrow a standalone Wo result. Fresh actual-Automatic timing, replay and
separate exact-producer profiles are required before production admission.

The 126 per-source/geometry Integration registrations each have a60-second
deadline and share the normal serial GPU test lock. Every name is explicitly
included in ProductionTestPreflight. Only the focused feature suite ran; this is
not a full-preflight or model-level MTP certificate.

## Evidence

Root: `/tmp/llaminar-gfx1100-wmma-prefill-evidence`.

- Full Release replay/timing:
  `mtp-wo-verifier-seed-release-timing-1794798463220853244/receipt.json`, SHA256
  `1c9e1bbbb442aa318809f0a167883963683ce744a2161ac789d39c136754ad0c`.
- Independent timing audit: `wo-seed-timing-audit.json`, SHA256
  `7ceada3babb5184dd33e0386b07063049abc6e04ef9518fc319178a1fdea6c8a`.
- All126 registered Integration gates:
  `mtp-wo-verifier-seed-feature-1794798711544328514/receipt.json`.
- Candidate masks: `wo-verifier-policy-generation.json`, generated by
  `generate_wo_verifier_policy.py` from exact raw sample indices0–30.
- Frozen seed producer:
  `mtp-wo-verifier-seed-qualified-snapshot-1794799236147801820/snapshot.json`, SHA256
  `f54dd1fe63044612b416c505205c8ec51ea9d35cc302662bf6d01c3dcbbcf480`.
  It retains two final records and44 content-addressed file versions, including
  raw evidence, helpers, original sources, executables, core libraries, caches
  and build logs. The complete panel supersedes the initial six-case smoke.
- Builds: `wo-seed-release-build-r2.log` and
  `wo-seed-integration-build-r2.log`, retaining gfx906/gfx1100 and CUDA sm86.
- Collectors: `run_wo_verifier_seed.py` and `run_wo_seed_feature.py`.

The first configure request used the repository root instead of the existing
`src/v2` cache source; the corrected configuration/build passed without a
compiler or kernel workaround. Static review clarified the TP-sized local scope
and strengthened the collector's numeric P16/span/publication assertions before
the complete timing panel.
