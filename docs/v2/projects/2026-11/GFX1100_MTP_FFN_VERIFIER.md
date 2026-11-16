# gfx1100 connected dense FFN verifier

The complete FFN core fixture now captures normalized FP32 input through actual
`FusedGateUpGEMMStage` and fused-SwiGLU `GEMMStage`, ending at the FP32 branch
output. All-source Release replay/timing passes22 cases and330 complete-chain cells.
The136 stronger homogeneous winners span all21 source formats. All22 registered Integration gates also pass. No gate/up Automatic admission
is installed by this fixture.

## Original operation and independent controls

The ordinary dense builder in `QwenGraphBase.cpp` owns hidden5120/FFN17408:
two gate/up projections N17408/K5120 and down N5120/K17408. Each control retains
three distinct original prepared sources in one store. Four controls and the
independent M1 oracle have separate prepared engines, workspaces and outputs.
The ordinary SingleQ8 contract, alpha1/beta0, transpose_B=false, original
verifier row binding and connected gate/up outputs match the actual stage
configuration. Norm, residual addition and TP collectives are outside this
explicitly bounded FFN core transaction.

The four controls are complete Dot4, current Automatic, forced WMMA gate/up
with Automatic down, and complete forced WMMA. The incremental GU comparison
is current Automatic versus forced GU with unchanged Automatic down. Down
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

The21 per-source Integration gates each retain a60-second deadline; the mixed
control has its own gate. All22 names are explicitly included in the final
ProductionTestPreflight inventory. This is the requested focused feature scope,
not a full preflight or model-level MTP claim.

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
qualification label. Actual-Automatic replay/timing and separate exact-producer
profiles remain required before installing a production selection.

## Evidence

Root: `/tmp/llaminar-gfx1100-wmma-prefill-evidence`.

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
