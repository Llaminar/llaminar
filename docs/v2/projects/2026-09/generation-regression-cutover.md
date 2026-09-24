# Routine generation regression cutover — 2026-09-16

Implementation slice, not a Docker/image certificate.

## Installed workflow

The default image pipeline now runs Unit → production preflight → approved
HTTP token regression → HTTP long-context/needle E2E → remote MPI E2E →
benchmarks → certification, separately for AVX512 and AVX2. Only Off controls
and dynamic-depth MTP run routinely. The current canonical projection contains
242 cells: 175 Off and 67 dynamic. Fixed-depth mathematical cases remain in
the canonical inventory and are available for diagnosis.

`--diagnostic-mathematical-parity` explicitly adds the full HF/CSV matrix.
`--through parity` without that flag fails. The diagnostic matrix can reuse
the canonical unchanged prerequisite receipt; no per-cell Unit/preflight runs
were introduced. The routine generation report is bound to the complete
inventory, exact source approval, prerequisite receipt and immutable ISA image.
Certification and resume reauthenticate that binding.

The testing skill, AGENTS, README and CI/parity guides describe this policy.
Token drift calls for the matching HF mathematical cell and its first divergent
checkpoint, never automatic rebaselining.

## Versioned acquisition

Published `Llaminar/corpora` commit `d962ad0` archives the original native
AVX512 acquisition under `generation_regression/avx512/native-20260916/`.
Its LFS payload contains all 510 explicit cell/control mappings, prompt bytes,
seeds, 175 serial controls and 700 exact 384-token outputs. The original audit
revalidated 175/175 controls and 335/335 MTP comparisons. The exporter only
reads existing observations; it cannot run inference or approve its own output.

This is **unapproved historical acquisition**, not fresh image proof. The
source approval catalog intentionally has no approved ISA entries yet. Current
metadata differs from the archive: the prefill segment setting changed from
600 to 512 and remote eligibility metadata was added. The archive is not
rewritten to conceal those differences. Before routine model certification,
review independent numerical/ISA provenance and explicit configuration
compatibility, then publish the approved generation and source catalog pin.
Do not invent AVX2 evidence from AVX512 acquisition.

The corpus commit and LFS object were pushed before changing the source
gitlink; the source working tree remains uncommitted. GitHub Actions remains
disabled. No model campaign or Docker certification was launched in this slice.

## Verification

- Focused pipeline suite: 182 tests passed, including exact last-token drift,
  changed seeds/configuration, missing mappings, immutable export, opt-in math
  scheduling and source-bound certificates.
- Broader Python script suite: 1,544 tests passed in 312.05 seconds.
- Rebuilt `v2_unit_gate`; full registered Unit gate: 657/657 passed in 83.40 seconds.
- Updated skill validation and HTTP shell syntax checks passed.
- Immediately preceding ordinary sampling-admission changes: complete
  `PipelineMTPStateOwnership` CPU/CUDA/ROCm groups passed, 3/3 in 3.51 seconds.
  They cover history banks without MTP, multi-request bank resets and stop/
  penalty publication across twenty admissions. They are already preflight
  members. A complete refreshed preflight after those C++ changes remains to run;
  the earlier full 225/225 receipt must not be presented as covering them.

Logs: `/tmp/generation-cutover-{unit,all-scripts,unit-build,full-unit,list}.log`
and `/tmp/planning-sampling-admission-focused-20260916.log`.

The separate auto-planner goal still requires captured pipeline execution and
both Docker certifications; changing routine regression policy does not close
that implementation boundary.
