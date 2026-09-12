# Container model-manifest alias admission

2026-09-08. Container run 16 passed both ISA build gates, all 644 Unit and
118 AVX512 production preflight tests, all eight 35B overlay numerical cells,
and the CPU/CUDA Qwen2 pipeline-parallel cell. Its first 122B cell stopped
before inference because its configured GGUF was reported as undeclared.

The CTest manifest names `/src/models/<shard>` while the typed 122B model
definition names `/opt/llaminar-models/<shard>`. Both are intentional bind
mounts of the same source directory. `stat` inside the image proved identical
device/inode identities, but `filesystem::canonical` retains different path
strings for bind mounts. The old C++ child admission only compared those
strings; the Python staging authority already recognized source aliases.

`productionParityResolvedModelPath` now checks filesystem equivalence and
selects the staged filename from the matched declared entry. It still rejects
different files with identical names/sizes/content, missing staged copies,
and missing staging authority. Already-staged resolution remains idempotent.
Only metadata is read; no GGUF hashing or source-file loading is introduced.

`ModelManifestAcceptsSameFileThroughAnotherFilesystemAlias` reproduced the
old failure with hard links, which provide distinct canonical paths sharing
one device/inode without mount privileges. A companion regression rejects a
separate same-named file and a missing staged copy. Both live in the existing
device-free `V2_Unit_ModelParityDefinition` gate.

The exact first 122B cell (CUDA1 + CPU2, Static/Ordinal, MTP off) subsequently
passed individually in 54.8 seconds with eight CSV artifacts validated. Its
report is `parity-results/ci-model-alias-individual/report.json`; deterministic
red/green evidence is in `parity-results/ci-model-alias-{red-test,green-unit}.log`.
These focused checks are non-certifying and precede another container run.

The final local shared gate then passed 644/644 Unit and 118/118 production
preflight tests in 538.3 seconds. Evidence is retained in
`parity-results/ci-model-alias-final-gate/` and
`parity-results/ci-model-alias-final-preflight.log`.
