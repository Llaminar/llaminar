# Frozen-history accuracy panels

Use this regime for numerical acceptance in all PyTorch model parity tests:
dense and MoE, CPU, CUDA, ROCm and composed topologies. Before executing an
unmigrated test, convert it to this contract in the existing parity framework.
Retain its regression obligations and full layer/stage diagnostics. A wrapper
around one legacy prompt or a renamed result does not constitute migration.
This requirement concerns mathematical model parity; it does not replace
device-free Unit, focused kernel, exact MTP, generation or HTTP tests.

## Freeze comparable work

Build a versioned panel of distinct representative prompts, with held-out
histories and relevant context/dispatch boundaries. Preserve known failing
histories. Select histories, continuation lengths, required groups and limits
before evaluating a candidate; do not choose favorable tokens after seeing KL.
The corpus owns counts and depths, rather than a universal two-step limit or
a row count copied from a tuning run.

Bind the original production GGUF, tokenizer, reference implementation,
precision policy and exact topology. The independent CPU/FP32 PyTorch
reference reads that same GGUF, including its existing quantization. Preserve
source metadata and numerical checkpoints; do not add whole-model hash scans.

Freeze prompt IDs and teacher continuation IDs once. Reference, candidate and
approved control consume identical histories using incremental cached decode;
each engine retains its own arithmetic and routing. A candidate's new greedy
continuation must not replace the teacher input. For `D` decode observations,
the current pending-token contract stores `D+1` continuation IDs: the first
`D` are consumed and the final one remains pending. Record logical positions
and native physical buckets separately, so a replay preserves the workload
that selected the kernel under test.

## Measure and gate

Use the shared scorer in `scripts/ci/full_vocabulary_metrics.py`. For each
history, compare the final prefill LM-head row and every declared decode row:

- Compute `KL(reference || candidate)` over the complete vocabulary at
  temperature one, in nats. Original FP32 logit observations are normalized and
  scored in FP64. Do not trim vocabulary tails, mask inconvenient rows or
  repair nonfinite values.
- Report mean KL, top-1 agreement and mean top-5 set intersection divided by
  five, including means at each decode offset. Top-5 overlap is distinct from
  reference-top-token-in-top-K recall. Keep raw row identities and scores.
- Aggregate by backend, topology, context group and phase. Give each prompt
  equal weight within its group, irrespective of continuation length. Keep
  prefill and decode separate. Report paired KL changes against an approved
  control on the exact same rows; do not pool models or configurations.

Mean final-logit KL is the primary accuracy summary. Retain explicit per-offset,
single-row, tail, total-variation and agreement guards so an acceptable mean
cannot conceal a severe local regression. Use reviewed model/configuration
policies, not QFN thresholds copied to another model. A worst-row or worst-stage
score is a diagnostic, not a replacement for the panel means. Decode KL need
not rise monotonically: each step is a different conditional distribution;
compare candidates on matched histories and inspect outliers through replay.

Complete collection alone is `UNCALIBRATED`. Initial absolute pilots declare
required versus diagnostic groups and explicit limits; `REQUIRED_GROUPS_GREEN`
accepts only those required groups. Diagnostic groups remain unaccepted.
Subsequent calibrated comparisons bind an actual approved baseline and its
policy. Never use the candidate as its own baseline, silently relax limits or
describe a selected-topology pilot as complete model/image certification.

Missing, duplicated, reordered or nonfinite observations fail. Retain production
capture/path evidence, source/configuration identity, exact reset replay,
publication completeness, driver checks and native retirement. Reset probes
prove state reuse independently; they must not inflate the scored prompt count.
Panel KL does not relax serial/grouped MTP byte equivalence, integer identities,
prefix restoration or required stage invariants.

## Replay numerical failures through every layer and stage

Project an outlier's history and exact configuration back into its canonical
production parity cell. Generate a full-stage reference for those same frozen
inputs; final-logit-only reference captures cannot locate the first divergent
layer. Verify the new reference's final logits agree with the retained panel
reference before native replay. Keep both artifacts if they differ and
investigate the reference rather than replacing it.

Retain all six canonical numerical CSVs and path/prefix/MTP evidence required
by the cell. Use [CSV evidence](csv-evidence.md) to locate the first divergent
phase, decode offset, layer and stage. Preserve original stage metrics and
verdicts even when final-logit acceptance passes; investigate routing drift
with matched inputs before attributing downstream errors to a kernel.

## Implementation and migration

The RC gate implements panels for dense Qwen3.8 and QFN through the shared
full-stage parity framework:

- `scripts/ci/release_accuracy_panels.py` selects the six blessed cells'
  canonical serial controls from the exported native inventory. Its
  `blessed_serving_serial_controls` scope covers both shipping ISAs and shares
  the same immutable original-reference packs between them.
- `scripts/ci/release_accuracy_prompts.json` owns the authored histories and
  context boundaries. `AccuracyPanelHistory.h` supplies their frozen inputs
  to the existing native fixtures without removing stage, capture, prefix,
  reset, movement or source-admission obligations.
- `scripts/ci/release_accuracy_evidence.py` authenticates the panel receipts,
  per-row and prompt-balanced summary CSVs, full-stage evidence and exact image
  bindings before later RC phases and release publication.

The initial RC acceptance uses the existing model-owned absolute limits and
records `calibrated: false`. A passed RC panel does not claim a calibrated
paired baseline. The complete release additionally requires preflight,
OpenCode and benchmark evidence; see `docs/production-ci.md`.

Separate QFN pilot/calibration and failure-replay tools remain available:

- `python/reference/qwen4exp_accuracy_panel.py` collects original-reference
  final-logit histories through the canonical reference forward path.
- `scripts/ci/model_kld_corpus_v2.py` owns row coordinates, prompt-balanced
  aggregation, paired comparisons and pilot/calibration policies.
- `scripts/ci/run_model_kld_canaries.py` and
  `scripts/ci/model_kld_corpus_execution.py` own native panel execution.
- `tests/v2/integration/parity/README.md`, **Frozen QFN histories and full-stage
  diagnosis**, documents `--absolute-pilot`, canonical `--paired-cell` binding
  and the full-stage `--frozen-history` / `--history-replay` routes.

Those QFN-specific adapters still validate QFN source geometry; they must not
be used for dense models. Before running a non-panel test for another model,
extend the shared framework with that model's typed source, history and replay
bindings. Preserve its existing quantization, configuration, stage coverage and
accuracy obligations. Do not impersonate QFN or create an alternate forward,
scorer, topology inventory or private acceptance runner. Add focused regressions
for row alignment, scoring, reset, source identity and full-stage replay as part
of that migration, then run the migrated panel.

The older dense/MoE canaries in `scripts/ci/model_kld_canary.py` remain separate
from the RC panel adapter. Convert any unmigrated test before running it under
this policy; do not label legacy results as panel coverage. Broad diagnostic
coverage does not add blessed release cells. A skill edit does not change
executable registrations or grant a CI/image certificate. Keep
large corpus payloads in the optional `Llaminar/corpora` repository and use the
existing typed matrix for backend/topology coverage.
