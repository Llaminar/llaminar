# gfx1100 attention-output verifier projection

Automatic now selects 746 measured source/geometry/M keys across all 21 source
formats for the actual `GEMMStage` Wo operation. Two fresh 1,890-cell panels retain
every selected median win. Confirmation geometric gain is 1.237135×,
range 1.047725–1.813037×. All 129 focused Integration
registrations/137 GTests and 16 separately collected native counter passes are
green. TP-sized cases qualify local extents, not original source-slice loading.

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

Three controls—Dot4, actual Automatic and forced WMMA—and a separate M1 oracle
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
The native producer has 80 VGPRs and 7,040 LDS bytes, with zero reported private
storage. The separate actual-Automatic counter qualification is recorded below.

## Seed economy and dispatch boundary

The Release panel contains 126 cases, 1,890 source/geometry/M cells and 58,590
rotated forward/reverse three-arm samples. Each internal-event interval retains
eight complete projections. Preparation, poison, observation and M1 reference
execution stay outside the timed interval. Unprofiled collection requires AUTO
power before and after the run.

Blanket WMMA wins 1,079 medians but only 878 complete31-pair comparisons; its
all-cell geometric ratio is 1.009067×. The generator independently recomputes all
medians from finite-positive raw triples and admits only keys winning every pair
with at least 5% median gain. It retains761 keys and nonmonotone holes. For
example, Q4_0/full dense qualifies only M16, while Q5_0/MoE full qualifies M5–16.
Some source/geometry pairs have no qualifying width and retain ordered work.

The first actual-Automatic panel found two selected median losses: Q8_0 at
MoETP2/M9 andM10 (ratios 0.986403/0.995316). Applying the same 5% qualification
margin to that panel removes 15 seed keys, leaving 746 across all 21 formats. The
rejected 761-key producer and raw observations are frozen separately. No new key
or pooled source alias is introduced by this pruning. Five percent is a selection
margin, not a guaranteed future minimum.

Installed selection retains original source, Projection operation, exact-hosted
extent and canonical serial policy. SwiGLU, fused subgroups, selected LM-head
rows and unknown shapes cannot borrow a standalone Wo result. Existing scalar
and GateUp policies remain unchanged. A golden expectation table independent of
the production include checks the captured engine at every enabled case, so
correct Dot4 output cannot conceal a missing native selection.

The 126 per-source/geometry Integration registrations each have a 60-second
deadline and share the normal serial GPU test lock. Every name is explicitly
included in ProductionTestPreflight. Only the focused feature suite ran; this is
not a full-preflight or model-level MTP certificate.

## Installed Automatic qualification

Both final panels measure Dot4/Automatic/forced-WMMA triples using the unchanged
eight-operation complete-stage event interval and ordinary AUTO power.

| Panel | Cells | Native median wins | Native all 31-pair wins | Native geometric gain | Native range |
|---|---:|---:|---:|---:|---:|
| First final producer |1890|746/746|715|1.237702×|1.051454–1.822258×|
| Confirmation |1890|746/746|705|1.237135×|1.047725–1.813037×|

The 1,144 unselected cells retain Dot4. Their geometric ratios are 1.000463×
and 1.000621×; observed minima are 0.894558×/0.923305×. Independent
inventories match between the two ordered controls, but the timings vary.
No universal per-cell no-regression claim or causal attribution is made.

All 126 actual-stage registrations pass, plus six guarded-capacity cases, the
all-format/six-geometry/M1–33 selector replay and four existing mapped/output
epilogue cases:129 CTests/137 GTests. The selector also checks mismatched count,
source, weight, geometry, policy, device and operation identities. Guarded
capacities retain Dot4; their explicit forced-WMMA diagnostic still proves bytes.
The new replay registrations are explicit ProductionTestPreflight entries.

Sixteen separate counter launches cover IQ4_XS/M16 at Dense N5120/K6144 and
MoETP4-sized N2048/K1024, eight metrics per anchor. Each selected region contains
one actual Automatic projection in its retained graph. They authenticate the
exact stream, graph, GPU UUID/PCI, P16, block128 and grids 80×16/32×16. Both use
wave64, 80 logical VGPRs,7040 LDS bytes and zero scratch/private/register spills.
The loaded ELF matches the earlier all-source verifier producer:
`3cb215279203a6e5781dac965660e7fca1e5824f1ffaa926201cb4a4d84c896a`.
These are two new geometry anchors, not 126 independent counter profiles.
STABLE_STD is scoped to counters and AUTO restoration is verified before timing.

| Anchor | Occupancy | Memory busy | L2 hit | LDS bank conflict |
|---|---:|---:|---:|---:|
|Dense|30.096%|56.949%|20.776%|2.312%|
|MoETP4|10.177%|18.693%|40.261%|2.312%|

Only focused component/feature tests ran. This is not whole-preflight, TP source-loading
or model-level MTP certification. No rocBLAS replacement is involved.

## Evidence

Root: `/tmp/llaminar-gfx1100-wmma-prefill-evidence`.

- Full Release replay/timing:
  `mtp-wo-verifier-seed-release-timing-1794798463220853244/receipt.json`, SHA256
  `1c9e1bbbb442aa318809f0a167883963683ce744a2161ac789d39c136754ad0c`.
- Independent timing audit: `wo-seed-timing-audit.json`, SHA256
  `7ceada3babb5184dd33e0386b07063049abc6e04ef9518fc319178a1fdea6c8a`.
- All 126 registered Integration gates:
  `mtp-wo-verifier-seed-feature-1794798711544328514/receipt.json`.
- Candidate masks: `wo-verifier-policy-generation.json`, generated by
  `generate_wo_verifier_policy.py` from exact raw sample indices0–30.
- Frozen seed producer:
  `mtp-wo-verifier-seed-qualified-snapshot-1794799236147801820/snapshot.json`, SHA256
  `f54dd1fe63044612b416c505205c8ec51ea9d35cc302662bf6d01c3dcbbcf480`.
  It retains two final records and 44 content-addressed file versions, including
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

Final installed-Automatic records:

- First final timing: `mtp-wo-verifier-automatic-release-timing-1794800996338271499/receipt.json`, SHA256
  `763e5bf114763cb993dad5429e56b6425511507b597df8e660b9b3b0cf68d291`.
- Confirmation timing: `mtp-wo-verifier-automatic-release-timing-1794801468021812623/receipt.json`, SHA256
  `7798f9c8c577263a0186e0b53b5ddab7aa0bdaa8821a26c40d25328d53dc3f31`.
- Focused Integration feature: `mtp-wo-verifier-automatic-feature-1794801696033790963/receipt.json`, SHA256
  `ef77c348f13ed00eb357800de5b29878d3bb32596d43b2307ef735f54d6f5214`.
- Independent audit: `wo-automatic-746-first-timing-audit.json`, SHA256
  `a1d94421240ffbbe64bfa2833e94f84e221c1a4096538001c0ab894123921279`.
- Independent audit: `wo-automatic-746-second-timing-audit.json`, SHA256
  `c6f710d54aaae1fbe9d4b087708e034249ccc47d93ae93b6ed0d55f697294bb1`.
- Final versioned archive: `mtp-wo-verifier-automatic-qualified-snapshot-1794801991402072673/snapshot.json`, SHA256
  `d7780153d5be2182af0a34971e7e4d6aad0c2d03ff60c3d87cb705bd276fdaa9`. It retains final records, the seed and the rejected 761-key
  diagnostic as separately identified original versions.
- Rejected diagnostic: `mtp-wo-verifier-761-rejected-diagnostic-snapshot-1794800840643439180/snapshot.json`,
  SHA256 `dadee12c80e02e7744ae566529aaba22ae60c31e2caa9e2baabffd553193e2ca`.
- Installed masks: `wo-verifier-automatic-policy-generation.json`; final builds:
  `wo-automatic-release-build-r3.log` and `wo-automatic-integration-build-r3.log`.
  The existing full gfx906/gfx1100 and CUDA sm86 configuration is retained;
  this slice changes host selection and tests, not native kernel machine code.
- Collectors: `run_wo_verifier_auto.py`, `profile_wo_verifier_automatic.py` and
  `run_wo_auto_feature.py`. All collection is bound to original producer bytes.
