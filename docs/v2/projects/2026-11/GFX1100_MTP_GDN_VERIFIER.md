# gfx1100 complete GDN verifier qualification

The complete four-source fixture passes all43 Release cases and all three
registered Integration gates/43 cases. Full-source Release timing passes630
cells and19,530 alternating pairs. These are forced candidate measurements;
GDN Automatic admission is the next implementation step.

## Original operation and numerical contract

`Test__ROCmGDNVerifierTransaction.cpp` captures actual `GDNProjectionStage`
execution with K2560 and QKV/Z/alpha/beta widths10240/6144/48/48, matching the
existing four-source GDN component. Every quantized source in the canonical
21-format registry appears independently as QKV and Z. A separate Q4_0/Q8_0
mixed-source control retains both arithmetic modes. Original FP32 alpha/beta
are materialized under the established FP16RNE preparation policy.

Dot4, forced WMMA and serial M1 have separate prepared stores, mutable kernels,
workspaces and outputs. Within each owner, all four distinct original sources
belong to one prepared store. The M1 oracle executes the actual complete stage
with the same activation precision; it does not borrow grouped quantization,
weights from another candidate's prepared owner or grouped output snapshots.

SingleQ8 applies native candidates to QKV and Z. TwoTermQ8 retains the existing
corrected QKV producer and applies the candidate only to Z. Both modes include
all four projections, publication and preparation-independent execution in the
complete-stage timing interval. This work does not replace floating kernels or
rocBLAS, and does not claim serving-topology or model-level MTP speedups.

Every physical M2–16 covers full/one/empty/regrown/repeated work and every live
prefix. Outputs and transient scratch start dirty; active rows must equal
independent serial bytes and inactive capacity must remain poisoned. Reset
retains the captured topology. Exact native arguments authenticate original
input/Q8/scale/sum banks, rows, source descriptors, outputs, publication mode,
partition span, and bounded nonoverlapping partial planes within each packet.
Sequential subgroup packets may reuse an admitted bank.

Correction publication and projection authenticate their full prepared QKV
source, quantizer, output, row authority and four distinct primary/residual
banks. Omission controls independently remove the full activation-publisher
family, native projections or corrected projection, checking both the expected
failure and surviving sibling outputs. TwoTerm already publishes primary Q8
before Z; removing only Z's later quantizer would not reliably remove the
activation producer. Native inventory rejects private scratch.

The fixture explicitly initializes immutable hardware metadata and finds an
actual visible gfx1100 participant. It does not require that card to be the
host's first physical GPU.

## Forced candidate economy

Each sample contains eight complete stages between retained internal GPU events.
Setup, poison, observation and independent M1 execution are outside the timing
interval. There are31 alternating Dot4/WMMA pairs per full physical extent;
all prefix correctness checks precede timing. Power remains ordinary `AUTO`.

| Mode | Cells | Native median wins | All31-pair wins | All31 wins plus ≥5% median gain | Geometric gain of that stronger set |
|---|---:|---:|---:|---:|---:|
| SingleQ8 |315|161|122|117, across all21 sources|1.17242×|
| TwoTermQ8 |315|158|84|44, across12 sources|1.06417×|

Unconditional WMMA is uneconomical: geometric ratios across every measured M
are0.94493× for SingleQ8 and0.98488× for TwoTermQ8. SingleQ8's first stronger
winner varies from M9 to M15 by source. At M16 its gains span1.07099–1.33814×.
TwoTerm's smaller gains reflect the unchanged corrected QKV cost; IQ1_S/M16
still loses0.23%. Selection must retain the original arithmetic mode and source
identities, with Dot4 at losing or unmeasured work sizes. These observations do
not yet qualify actual Automatic execution or prescribe a monotonic threshold.

## Frozen evidence and next step

Root: `/tmp/llaminar-gfx1100-wmma-prefill-evidence`.

- Full Release replay:
  `mtp-gdn-verifier-release-replay-1794789637128590058/receipt.json`.
- Full Release timing and per-mode analysis:
  `mtp-gdn-verifier-release-timing-1794789732667064905/{receipt,analysis}.json`.
- Registered Integration gates:
  `mtp-gdn-verifier-feature-1794789844938488017/receipt.json`,92.20s CTest wall time.
- Both builds: `gdn-verifier-{release,integration}-final3-build.log`.
- Frozen producer: `mtp-gdn-verifier-seed-snapshot-1794790099422775362/snapshot.json`,
  four records/42 files, SHA256
  `923640b37a7562e7c4e74833761c266986d060d4dd373ea99256f8e173b9da34`.

Automatic needs an explicit complete four-source GDN binding, including the
QKV correction mode and prepared control precision. GDN copies compatible
subgroups, so attention's original-span identity cannot simply be reused or
weakened. A construction-only subgroup view should authenticate exact copied
storage and ordered roles against the complete original operation. SingleQ8
permits QKV/Z native roles; TwoTerm permits only Z. Unknown or unqualified GDN
bundles must not borrow a separately qualified scalar or attention key.

After installing conservative measured masks, rerun actual-Automatic timing and
focused exclusions, then collect exact-producer profiles for admitted grids.
The existing kernel resource evidence alone does not qualify these new complete
operation economics.

Reviewed admission design: use a separate `NativeQuantisedGDNVerifierOperation`
with named QKV/Z native members, a typed alpha/beta floating control pair,
original device/input/rows/K, correction mode/row identity, and shared Q8 banks.
The complete operation always has four roles; its eligible native role mask is
independent. Retain QKV provenance in a TwoTerm key even though only Z uses the
candidate. Share lower packet-membership mechanics with QKV while preserving
QKV's original count semantics and strict storage matching. Each backend API
must install both native scopes, leaving the unrelated one empty.

A neutral original GDN binding must remain visible when qualification fails so
unknown subgroups cannot borrow scalar keys. Physical FP16 weights plus FP16RNE
activation policy, exact N48/K2560 and distinct alpha/beta outputs authenticate
the control execution key. Stage `w_a/w_b` pointers alone do not prove original
FP32 derivation; this fixture passes the materialized half tensors there.
Focused exclusions must cover correction-mode/eligible-role changes, mixed
sources, foreign copied subgroup storage, control dtype/precision/geometry,
foreign correction counts and nested scope restoration.
