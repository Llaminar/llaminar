# gfx1100 connected dense FFN verifier

Automatic now selects native GateUp for 136 source/width keys across all 21
formats at the ordinary dense hidden5120/FFN17408 shape. Two complete FFN-chain
Release panels each pass 315 cells and retain median wins at every selected key:
geometric gains 1.113881× and 1.112072×. Six focused host
gates/39 cases, 26 affected GPU gates/86 cases and all 16 exact producer profiles
pass. This qualifies the bounded component; routed MTP, Wo and selected LM-head
qualification remain separate work.

## Original operation and independent controls

The ordinary dense builder in `QwenGraphBase.cpp` owns hidden5120/FFN17408:
two gate/up projections N17408/K5120 and down N5120/K17408. Each control retains
three distinct original prepared sources in one store. Four controls and the
independent M1 oracle have separate prepared engines, workspaces and outputs.
The ordinary SingleQ8 contract, alpha1/beta0, transpose_B=false, original
verifier row binding and connected gate/up outputs match the actual stage
configuration. Norm, residual addition and TP collectives are outside this
explicitly bounded FFN core transaction.

The seed controls are complete Dot4, prior Automatic, forced WMMA gate/up
with Automatic down, and complete forced WMMA. The incremental GU comparison
is prior Automatic versus forced GU with unchanged Automatic down. Down
already has independent scalar SwiGLU admission; a complete forced-WMMA win
against Dot4 alone cannot qualify adding GU to production Automatic.

Every source in the canonical21-format registry runs at M2–16. A separate
mixed-source control uses original Q4_0 gate with Q8_0 up/down. Runtime selection
is inspected independently for both stages; the collector authenticates existing
down source/M masks independently of the production selector.

## Capture, replay and timing

Both stages bind the same merged persistent workspace and exact stream before
capture. GU and down sequentially reuse Q8/scale/sum banks; no poison or rebinding
occurs between them. The graph dependency ledger declares gate/up as internal
down inputs with the original producer index. Their publication remains part
of the complete captured operation.

The M1 oracle executes both actual serial stages, including the existing M1
SwiGLU temporary and quantization behavior. Gate, up and final branch output
must match its bytes independently. Every physical M covers full, one, empty,
regrown, repeated and every intermediate live prefix. Outputs and transient
scratch start dirty; inactive output capacity stays poisoned. Session reset
retains captured topology.

Capture inspection authenticates original sources, counted Q8 publishers,
input/output banks, N/K, rows, launch geometry, native packet membership,
partial-plane bounds and original reducer bindings. Native partial producers
and reducers are enumerated independently and joined by role, exact address and
partition count. Separate omissions of GU quantization, SwiGLU quantization,
native GU and native down must break nonempty output. Down-only omissions must
leave both GU outputs byte-correct. Restore must immediately recover output.

Timing uses31 rotated forward/reverse four-arm samples for each full physical
extent, with eight complete chains per retained internal-event interval.
Preparation, poison, oracle and observation stay outside that interval.
Ordinary AUTO power is required before and after unprofiled collection.
Counters remain a separate qualification step after any native admission.

The 21 per-source Integration gates each retain a 60-second deadline; mixed
sources and guarded capacity have their own gates. All 23 names and the three
new device-free binding/policy/stage-boundary regressions are explicitly in
ProductionTestPreflight. The requested focused feature suite ran; the complete
preflight suite and model-level MTP certification are outside this evidence.

## Incremental gate/up candidate economy

The first complete Release panel passes22 cases,330 cells and10,230 four-arm
samples (40,920 arm observations). These are315 homogeneous source/M keys and
15 mixed-source controls. Unconditional GU WMMA against current Automatic has
geometric ratio0.995859× across the315 homogeneous keys: blanket selection is
slightly slower.

A conservative seed filter retains136 homogeneous keys across all21 formats.
Each wins all31 comparisons and has at least5% median gain. Their geometric
ratio is1.113514×, range1.050536–1.218267×. Q8 sources qualify M5–16; Q4_0 starts
at M9, while IQ1_S qualifies only M16. The mask is intentionally nonmonotone:
IQ4_XS qualifies M9–11 and M13–16 but excludes M12. Mixed-source wins remain
separate diagnostic controls and do not authorize a homogeneous source key.

The generator recomputes eligibility from exact sample indices0–30 and all
four finite-positive costs, checking the stored summary against those raw
observations. It never uses the full-forced-versus-Dot4 ratio as the GU
qualification label. The final installed-selection confirmation below uses
fresh actual-Automatic observations against the same previous stage choices.

## Installed Automatic qualification

`VerifierProjectionBundleKind::GateUp` authenticates the original const two-source
span in the actual stage before backend subgrouping. Its scope ends before down.
Original input, count/extent, device, Q8 banks, prepared sources and output bindings
must agree; copied/partial public spans, mixed sources, unknown shapes and guarded
capacity do not borrow this qualification. Legal backend subgroups retain their
original complete-operation authority. Device-free tests cover all source
cross-products, M1–33, the IQ4_XS/M12 exclusion and scope restoration on false,
exception and repeated actual GateUp-to-down stage execution.

Final comparisons retain four independently prepared complete chains: Dot4,
previous selection (Dot4 GU plus unchanged Automatic down), installed Automatic,
and full forced WMMA. The scalar down mask, native arithmetic and launchers are
unchanged. The old producer is frozen, and the collector independently checks
every down decision against that mask. Canonical partition/span and native
producer/reducer equality are asserted during replay; the seed log does not
retain a numeric per-packet P/span export.

Each actual-Automatic panel contains 315 source/M cells and 9,765 four-arm
samples. All 136 selected medians win in both panels. First-run gain is
1.044631–1.223892×; confirmation is
1.046938–1.220591×, geometric mean
1.112072×. The original seed's 5% filter is not a promised
minimum on subsequent measurements. Unselected cases retain the same captured
kernel/resource inventory; separate prepared allocations still show timing
variation. Unselected geometric ratios are 1.000448× and 0.998993×; the second
panel retains IQ2_XS/M2 and M3 ratios 0.903788× and 0.919963×, and IQ3_S/M4
is slower in both panels. The audits preserve these observations without
attributing their cause. This evidence does not claim every unselected cell is
faster.

The separate IQ4_XS/M16 profiler owner replays one actual chain with setup and
observation outside the marker. GU and down share a native symbol, so attribution
joins the exact counter Dispatch_Id and independently checks role, population
and grid: GU 544×16×1/two members and down 80×16×1/one member. All 16 passes
use the timing producer's loaded ELF `3cb215279203a6e5781dac965660e7fca1e5824f1ffaa926201cb4a4d84c896a`:
wave64, 128 threads, 80 VGPRs, 7,040 LDS bytes and zero scratch/private storage or
memory spills. GU/down measured occupancy is 45.76%/34.38%, memory busy
64.13%/60.17%, L2 hit 20.03%/14.85%, with 2.31% LDS bank conflict. Both report
100% GPU busy. These counter runs use STABLE_STD and restore AUTO; unprofiled
qualification uses AUTO throughout. Complete source builds retain gfx906/gfx1100
and CUDA sm86, with enforced spill guards.

## Evidence

Root: `/tmp/llaminar-gfx1100-wmma-prefill-evidence`.

- Final actual-Automatic snapshot: `mtp-ffn-verifier-automatic-qualified-snapshot-1794798130863860885/snapshot.json`, SHA256
  `d7cc52b5eaed9ec2f5bb3913528c1f4baa8e252381431c93121947aaac58dbb8`. It freezes both final timing panels, host/GPU gates,
  profile owner, all 16 counter records, raw observations, exact producer versions,
  compiler logs and independent audits. The seed snapshot below remains the
  immutable selection provenance.
- First/second actual-Automatic timing: `mtp-ffn-verifier-automatic-release-timing-1794796054085743285/receipt.json` and
  `mtp-ffn-verifier-automatic-release-timing-1794797063552840481/receipt.json`.
- Independent timing audits: `ffn-automatic-first-timing-audit.json` and
  `ffn-automatic-second-timing-audit.json`.
- Final source builds: `ffn-automatic-release-build-r1.log`,
  `ffn-automatic-integration-build-r1.log`, `ffn-automatic-host-build-r2.log`;
  affected target rebuild: `ffn-auto-affected-feature-build.log`.

- Reviewed smoke: `mtp-ffn-verifier-seed-release-replay-1794794089533027378/receipt.json`,
  three cases,33.792s GoogleTest wall time.
- Full Release replay/timing:
  `mtp-ffn-verifier-seed-release-timing-1794794144741780570/receipt.json`, SHA256
  `862098fd6eb536efecd9a331e6946d2effb5ad95dd21b84c8b08f2a2fd3fad54`.
- Independent seed audit: `ffn-seed-timing-audit.json`, SHA256
  `3904553a4879b024b69e7fe17adb4bc72059d007d2293fe9a4fe97b767bf3a22`.
- Candidate masks: `ffn-gateup-policy-generation.json`, SHA256
  `54bc099f2e67957ef474b780a1284e6b3b149eef70758b2502962178cd3a1939`.
- All22 registered Integration gates/22 GTests:
  `mtp-ffn-verifier-seed-feature-1794794778760379875/receipt.json`.
- Frozen final seed producer: `mtp-ffn-verifier-seed-qualified-snapshot-1794795098125462016/snapshot.json`,
  two green records/49 file versions, SHA256
  `95fd2555e919b9717f6ac114e933ef0d1cabcd85d29d37f3f3f7ebe58807b34d`.
  The full timing replay supersedes the earlier smoke for frozen qualification.
- Source builds: `ffn-verifier-release-build-r4.log` and
  `ffn-verifier-integration-build-r1.log`.
- All-source timing collector: `run_ffn_verifier_seed.py`.
- Registered feature collector: `run_ffn_seed_feature.py`.

The first build request omitted the new target from the Release test-target
allowlist. Reconfiguration added that explicit target; no compiler or kernel
workaround was needed. Read-only review corrected the fixture's transpose flag,
retained host count staging through its stream drain, and required independent
down selection and reducer identities before the full panel.
