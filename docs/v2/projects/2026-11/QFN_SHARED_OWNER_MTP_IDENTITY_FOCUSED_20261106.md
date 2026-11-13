# QFN shared-owner MTP: retained verifier identity

This focused slice follows `d4783d697`. Complete Unit and
ProductionTestPreflight remain skipped at the user's request. The new
device-free regression is explicitly registered in preflight.

## Defect and repair

ROCm TP2 with a pure-reader coordinator, a separate zero-capacity ngram cache
owner, RAM backing and fixed-depth-three MTP passed source admission, including
cold recreation. Its first executable generation parent then failed the
replicated-PLE producer check. A diagnostic-only rebuild reproduced the precise
failure: no authenticated retained verifier forward.

The hosted parent already selected the real verifier through its complete
`ForwardGraphSignature`. The validator recognized only capture pointers and
semantic graph pointers, neither of which represents that hosted verifier arm.
It consequently rejected a correctly materialized producer.

The validator now recognizes the exact signature of that typed hosted arm after
the original forward engine authenticates its retained executable. Device,
workspace, cohort, admission, arithmetic, sampler and terminal-publication
checks remain in force. Participant and canonical checks use the same membership
operation. Invariant failures now identify the offending device and contract.
The original hosted fragment value was mechanically extracted for direct unit
coverage; its stored fields and execution-identity comparison are unchanged.
No inference arithmetic, transport, allocation or launch policy changed.

## Focused evidence

The normal Integration build completed in 251.88 seconds. Six selected host
registrations passed in 2.83 seconds, without skips. These cover the complete
ForwardGraphTypes unit target, the new hosted-verifier regression and four
replicated-PLE cohort/admission/layout/lifecycle registrations. The new predicate
test covers CUDA/ROCm metadata, widths 1–16, changed keys, missing input banks,
foreign pointers and incorrect fragment kinds.

Both real-model cases use the unchanged four-shard Qwen3.8-Flash-Next
UD-Q4_K_XL and original Q8_0 predictor, FP32 activations, FP16 KV, static ordinal
expert placement and retained MTP capacity three. They retain the original two
MI50 continuations and CPU0 expert-capacity tier, with eight GPU-resident
experts. These are existing canonical owner-subset cells, without new topology
or precision substitutions.

| Existing cell | Source admission | Native exact inference | Exact comparisons |
|---|---:|---:|---:|
| RAM, owner ROCm1 C0, pure-reader coordinator ROCm0, fixed3 | 167.21 s | 161.49 s | 2,416 |
| SSD, owner/coordinator ROCm0 C64, reader ROCm1, fixed1 | 129.10 s | 125.27 s | 1,208 |

Each inference case passes fresh and three restored-prefix histories, actual
public two-token generation, stable original source/reader ownership and
certified HIP scheduler-ticket evidence. Both original GPU contexts retire with
zero canonical allocation bytes; all process and driver windows are clean.
The SSD/C64 forward evidence includes four GPU-cache promotions and four
evictions. Its terminal generation records 17 cache-hit rows and 15 source rows.
The C0 case correctly has no GPU-cache hits or promotions.

Sampled peak RSS is 133.28 GiB for RAM backing and 105.89 GiB for SSD backing.
These include the CPU expert tier and intrusive functional snapshots. They are
not Release throughput measurements. Native serial/grouped byte equivalence
does not replace the separate frozen-history PyTorch numerical panels.

## Evidence, cleanup and remaining scope

Local evidence is under `/tmp/qfn-goal-continuation-r729/`: the original and
diagnostic failures, coherent build, focused host report, both source audits,
model CSV audits, resource samples and driver receipts. Each successful model
run removed ten owned prefix-cache files, about 1.42 GB, after original-process
and device retirement. Both failed diagnostic lifetimes also removed their
owned caches. Source-admission fixtures completed their own cleanup.

This qualifies the two listed active-MTP ownership slices. Ordinary hosted
generation has a separately identified signature/sampler representation gap;
its repair and native regression are the next focused slice. Grouped ngram
read-ahead is privately staged but not part of this checkpoint. Other matrix
cells, numerical refresh, Release economy, full preflight and image publication
remain separate obligations.
