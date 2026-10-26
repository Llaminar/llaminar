# GDN CPU import and exact GPU retention

Updated 2026-10-26. This is a local implementation/evidence summary, not release certification. **Canonical prerequisite R2 PASS:** 2841 passed, 29 hardware skips, zero failures / 2870 tests, 4841.8 seconds. The exact receipt is `parity-results/gdn-backend-port/dense-retained-complete-prerequisites-r2/prerequisites.json`. Current Unit 782/782 and host 1093 pass; Release and Integration retirement resource audits r4 pass. Canonical GPU generation R6 passes all 12 local control/compare invocations across CUDA/ROCm single-device and TP2: 48 requests and 18,432 committed tokens. The packaging-only follow-up is validated separately below.

## Selected implementation

CPU imports the fixed-epoch/shared-readout harness and its versioned CPU state/storage contract. GPU serving retains the original Dense recurrence arithmetic and Dense persistent state, adding exact compact verifier retention. Backward-epoch GPU prototypes were rejected and retired:31 new trial files,21 preflight aliases and7 performance aliases removed. CPU import/research, ordinary/prepared production paths and selected exact-retention diagnostics remain.

For grouped GPU verification, the shared cold selector supplies the SAME native geometry/output contract to stage preparation and the workspace BOM. Physical requests R and rows/request M come from the generation owner, never generic batch size or allocation capacity. The regression includes batch7/R3/M16→48 rows. Ordinary GPU, CPU and prepared scalar ROCm paths are unchanged siblings.

The actual GDNRecurrenceStage prepares an immutable backend invocation before recording. It retains canonical Tensor physical-storage leases and named workspace leases; the existing kernel/cache owns live state. Owner identity is the stable workspace manager id, not an invented device generation. Native stream/device, source pointers, geometry, policy and output contract participate in capture identity. Reset changes data and preserves topology. No host phase mirror, recapture, conversion fallback or new accounting ledger is introduced.

ExactCorrections retains immutable cp0 (including live0), relative eight-row checkpoints, normalized K, rounded decay and actual rounded delta. Accepted reconstruction uses at most seven original elementwise updates; no dot/query/model-row replay. Required Dense terminal output is explicit. RetainedPrefixesOnly omits only the unaccepted terminal bank, preserving cp0 and revised acceptance.

Typed selections distinguish accepted counts (0 restores cp0), own-request flattened rows (negative NoOp) and terminal lengths (0 NoOp). MTP publication borrows actual workspace extents through readonly spans; no fabricated pointer capacity. The retained stage rejects the still-raw legacy terminal-pointer hook rather than inventing its extent. Native typed terminal selection remains separately supported. Request-zero mirroring occurs only for Restore where required; NoOp preserves both destination and stale nonauthoritative mirror.

Default matrix axes are CUDA KeyMajor and ROCm ValueMajor. CUDA cold scheduling uses canonical cached SM count: qualified DK128 KeyMajor grids above SM count use checkpoint-word/min2; smaller grids keep partition stores. ROCm retains vectorized ValueMajor checkpoint endpoints. Publisher matrix tiles are4096 elements; every CTA validates the complete immutable metadata plan before any destination write. No atomics, cross-CTA status or host count readback.

Implementation owners: GDNRecurrenceStage, GDNSpeculativeWorkspaceContract, WorkspaceMemoryEstimator, MTPGraphOwnerPlan/DGO, GDNDenseCompactSnapshots, GDNPreparedRetainedInvocation, backend GatedDeltaNet adapters and the original dense kernel bodies. Dense/prepared ordinary paths remain controls, not fallbacks.

## GPU correctness and economy

Focused native evidence covers DK64/128, both axes, square128 and tails, R1/R3, nonfinite/signed-zero/subnormal inputs, physical M16 with ragged/empty live rows, all accepted prefixes/revisions and following M1 bytes. Parallel whole-plan validation includes297 request/head owners. Actual prepared adapter and stage tests use real state bindings, Tensor/workspace leases, a separate event-ordered publication stream, dirty output preservation and reset/replay. These component proofs do not replace canonical full-model prefix/TP qualification.

Complete component timing compares legacy native deinterleave→full snapshots→accepted copy→raw M1 against direct merged capture→publication→the same M1. Convolution is common/excluded. Each live48 run has12 pairs,64 transactions/sample and1152 actual timed endpoint checks.

| Backend / heads | Live2 latency reduction | Live4 | Live8 | Live15 |
|---|---:|---:|---:|---:|
| CUDA /12 |1.5–6.4%|-4.1–6.7%|3.5–13.9%|11.4–22.9%|
| CUDA /24 |-0.4–3.9%|4.4–19.9%|17.3–21.2%|14.3–23.8%|
| ROCm /12 |6.3–14.3%|18.0–31.3%|33.6–40.9%|35.7–44.4%|
| ROCm /24 |22.9–29.2%|35.8–46.1%|47.5–55.4%|53.6–60.0%|

ROCm wins all tested live2+ cells; live0 and12-head live1 remain slower. Threshold4/8 did not improve the complete boundary and are rejected defaults. Explicit Full/Threshold fixtures remain diagnostic programs; production selects ExactCorrections. The CPU comparison below does not establish a reliable split threshold; Shared remains selected.

Matched Release control/candidate/control uses the same real MoE workload, C4096/M512, fixed MTP3 or15 (cap15), seed42,595 prompt/384 output tokens, prefix off, one warmup/three measurements. Tokens and MTP work match within each backend. All30 GDN layers export successful prepared-program inventory; cold inventory itself is not replay proof.

| Backend/depth | Pooled-control→candidate decode tok/s | Gain | Prefill tok/s |
|---|---:|---:|---:|
| CUDA1/3 |226.219→225.999|-0.10%, flat|2838.587→2835.725|
| ROCm1/3 |89.858→98.946|+10.11%|1274.652→1274.896|
| CUDA1/15 |96.008→96.652|+0.67%, within drift|2816.843→2821.006|
| ROCm1/15 |30.212→34.728|+14.95%|1274.539→1273.645|

CUDA depth15 control drift0.82% exceeds its apparent gain. ROCm depth15 uses clean r2; the original overlap remains failed timing evidence. Quiet/driver/retirement checks pass. Workspace arenas save 683 MiB CUDA and 699.5 MiB ROCm in the measured model/geometry. CUDA demonstrates this storage reduction without an established throughput gain; its apparent gains remain within drift. These are matched local measurements, not universal shape or serving certification.

After prototype retirement, Release/Integration resource audits r4 retain184 CUDA architecture rows and61 gfx906 bodies raw-exact, removing148 CUDA rows and23 HIP bodies. No selected arithmetic change or memory-spill waiver. This is passive GDN translation-unit evidence, not achieved occupancy or fresh timing. Local receipts: `parity-results/gdn-backend-port/gpu-trial-retirement-{release,integration}-resources-r4/result.json`.

## CPU evidence

The latest CPU projection-tile frozen-history panel **PASS_REQUIRED_GROUPS_GREEN** uses current prerequisites: `parity-results/gdn-backend-port/cpu-current-projection-tile-panel-r1/report.json`. Its `previous-panel-comparison.json` confirms all 178 metric rows and 26 payloads byte-identical, with exact reset results. The native observer completed in 206.37 seconds with normal retirement and source/core/driver proofs passing.

The approved panel has 25 histories, 178 metric rows, 26 logit/reset files and seven reset observations. Required short/decode mean KL=.0346761083, maximum row=1.0996653, top1=.9791667, top5=.9041667. Only its explicitly approved short-decode maximum-row ceiling is 1.2; other thresholds/defaults are unchanged. QSA-threshold rows remain diagnostic, not whole-corpus acceptance.

A matched CPU Release control/candidate/control run improves decode12.42→13.16 tok/s (+5.91%), with identical tokens/MTP work and all16328 counter-family sums/counts. Prefill171.00 lies within controls169.73–172.94. Workload: Qwen3.6 35B-A3B IQ3_S,28 physical workers, fixed MTP3,595 prompt/384 output tokens. The original import-versus-parent comparison has different tokens/accepted work and a regression; preserve it, do not call it a matched kernel gain.

Latest AVX512 projection blocking improves the595-row prefill component5.558→5.379 ms (3.22%), with33 exact timed outputs/states and no vector spills. True AVX2 arithmetic/counter tests pass. CPU FullEncoded/Shared retention diagnostics cover 20736 endpoint checks in their tested component geometry and authored acceptance schedule. They prove those exact endpoint contracts, not a general performance crossover or production split threshold. Imported Shared remains selected: short-row crossover results are mixed, so no CPU split policy is promoted.

The CPU numerical change is readout association amplified at Q8 boundaries/expert selection. Same-input recurrence states can be exact while later logits differ. GDN-output TwoTermQ8 worsens panel mean/max and remains rejected; GDN QKV/output stay SingleQ8. Do not conflate this CPU versioned arithmetic with exact GPU retention.

## Qualification dashboard

| Topology | Serial control/repeat | Dynamic MTP vs serial | Full/partial prefix restoration |
|---|---|---|---|
| CUDA single | PASS | PASS | PASS |
| ROCm single | PASS | PASS | PASS |
| CUDA local TP2 | PASS | PASS | PASS |
| ROCm local TP2 | PASS | PASS | PASS |

Canonical generation R6 completed 12 separately owned invocations, each with four 384-token requests. All eight comparison runs match their topology's serial control token IDs exactly. Dynamic runs show actual stochastic verification and depth updates; full and partial requests restore hybrid and MTP state. Both TP2 participants expose the expected sharded prepared programs. Normal retirement, actual loaded core, clean driver windows and source/asset identity checks pass. Every owned temporary model copy and prefix archive was removed after retirement.

These collect/compare modes explicitly set certification_eligible=false: they are local diagnostics against unapproved controls, not approved-corpus or image certification. The summary is `parity-results/gdn-backend-port/generation-scratch-r6/summary.json`. Earlier failed observer receipts remain preserved. Prepared-program inventory is not an executed-kernel counter; these report fields do not establish per-layer native replay causality. Component capture/replay and publication proofs remain separate evidence.

The HIP SDK packaging follow-up exports the already authenticated runtime component and deploys its DSO, aliases, receipt and repairs into the canonical SDK, replacing the stock package copy. It removes the duplicate runtime owner under /usr/local and registers only the SDK's public library directory. No inference source, kernel arithmetic, compiler target or local runtime DSO changed. The complete Unit suite passes 782/782 after application; all five focused deployment/loader/packaging tests pass. The new SDK deployment regression is registered in ProductionTestPreflight. Evidence: `parity-results/gdn-backend-port/hip-sdk-packaging-followup-r3/result.json`.

The complete native prerequisite R2 and model runs qualify the frozen inference tree before this packaging-only follow-up. No second complete native gate or local Docker image build was run for that follow-up. Future image qualification belongs to the normal CI image workflow; local loader and synthetic deployment checks do not certify an unbuilt image.

Checkpoint scope keeps the selected source/tests/headers coherent. Seven skill edits/two references, the unrelated corpora gitlink and four proposal documents remain separate. GPU retention selects ExactCorrections; CPU keeps Shared because measured short-row crossovers do not justify a new split policy.

The HIP loader defect used a stale runtime despite correct compilation. Default-loader positive/shadow tests now authenticate the installed selected runtime and its packaging; the prior failed complete gate remains unchanged. Other focused repairs include empty-request publication, CUDA physical-row GEMV cursor, prefix generation v18 and preservation of malformed committed archives without truncation. Legitimate short-query omission is authenticated by actual program observations plus independent history.

Detailed imported CPU investigations remain in the dated `gated-deltanet-*-results.md` notes. Local raw receipts live under `parity-results/gdn-backend-port/`; they are not source payloads or release certificates.
