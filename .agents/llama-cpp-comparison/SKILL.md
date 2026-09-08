---
name: llama-cpp-comparison
description: Benchmark and profile Llaminar against llama.cpp on a matched real-model workload, authenticate prompt/token and runtime settings, and attribute prefill/decode gaps to inference phases and kernels. Use for cross-engine performance comparisons, not ordinary parity-campaign execution.
---

# Compare Llaminar with llama.cpp

Produce a reproducible, numerical-contract-aware comparison of **Release**
inference. Separate unprofiled end-to-end throughput from profiler attribution;
neither a microbenchmark win nor Llaminar-only hotspot ranking establishes why
Llaminar is slower than another engine.

## Establish the comparison

Read [matched benchmarks](references/benchmarks.md) before running either engine.
Freeze model bytes, device/topology, exact prompt bytes, context capacity, cache
precision, sampling policy, MTP policy, warmup count and generated-token budget.
Respect the user's constraints on formats, activation precision and VRAM.
Pin and record the llama.cpp commit/build; “current mainline” requires checking
the remote revision, not assuming an old checkout is still current.

For the short-prefill comparison, use a **512-token prompt aligned to the
512-row Llaminar prefill bucket**. The supplied Qwen prompt fixture is a starting
point, not a tokenizer authority. Revalidate actual token counts, prompt-cache
misses, captured bucket selection and physical batches in both engines. Do not
silently replace a requested long-context workload with this short workload.

## Attribute the gap

Read [phase attribution](references/attribution.md). Capture both engines on the
same workload and classify physical work as projections/reductions, attention,
GDN/convolution, normalization, activation preparation, KV/state/layout,
embedding, vocabulary head and sampling/control. Keep fused operations whole.

Use the applicable project backend workflow for attachment and kernel counters:

- [CUDA](../cuda-tuning/SKILL.md): Nsight Systems/Compute, captured-node evidence.
- [ROCm](../rocm-tuning/SKILL.md): HIP/RCCL traces and instruction/resource proof.
- [CPU](../cpu-tuning/SKILL.md): perf, ISA, affinity and physical-core scaling.

The optional [SQLite analyzer](scripts/attribute_nsight.py) accepts explicit
phase windows and ordered classification rules. It reports category sums,
unions, exclusive intervals, overlap and no-kernel intervals separately, plus
per-kernel geometry. It never guesses token boundaries or vocabulary-head
identity. Keep those model/run-specific inputs alongside the raw trace.

## Choose and validate a change

Rank **paired exposed-time deficits**, not only percent-of-total hot spots.
Compare arithmetic, reduction order, operand formats, actual memory traffic,
occupancy, spills, launch geometry and stream overlap for the selected kernels.
For GEMM dispatch work also use
[NativeVNNI tuning](../nativevnni-gemm-tuning/SKILL.md); for MTP work use
[MTP tuning](../mtp-tuning/SKILL.md). Never adopt a baseline's precision or
non-invariant reduction merely to match its timing.

Profile one exact candidate/shape per launch, outside canonical timing. Then
repeat the fixed Release benchmark, preferably A/B/A or interleaved repetitions,
and compare token identities and memory BOM as well as throughput. Retain a
change only when its model-level benefit exceeds the measured noise and the
applicable correctness gates pass. Use
[model parity](../model-parity-testing/SKILL.md) for the full Unit gate,
ProductionParityPreflight and affected canonical real-weight cells. Report
missing or still-running evidence explicitly; do not call a timing candidate
certified.

Save dated results, exact commands, revisions, resource records and rejected
experiments under the project investigation, with large raw artifacts outside
the worktree. This skill owns the procedure, not a changing leaderboard or
another model/topology matrix.
