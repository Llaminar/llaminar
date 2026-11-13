# Six-GPU QFN recurrent MTP: focused qualification

This slice extends the fixed-depth-one checkpoint `ccd6b69ef` to repeated
original-slot draft transactions and retained verifier execution on CUDA TP2
continuations with four MI50 expert followers. Full Unit and
ProductionTestPreflight are deliberately omitted for this iteration at the
user's request. Focused regressions remain explicitly registered in preflight.

## Production ownership

The original admitted transaction slots retain immutable invocation metadata
and both endpoints' final payload receipts. Their bounded geometry comes from
the same capacity authority used by allocation. The source device rearms its
existing packet lane only after both prior endpoints and bank leases retire.
Repeated drafts reuse the same captured executable; there is no per-draft host
completion fence, host mirror of GPU execution state, or replacement admission
ring. Each actual follower cache retains the corresponding original-slot events.
The complete source frontier and all follower terminals precede metadata reuse.

Final packet receipts follow actual payload materialization and publication.
Original failure stays absorbing. Capture identity includes the retained policy,
slot roster and geometry, and tests reject inconsistent identities rather than
pretending that an invalid manifest describes a reusable capture.

## Cancellation defects

The new two-device cancellation trace queues two followers, launches only the
first source, and cancels the second original invocation. Both CUDA and ROCm
originally retired the source and first follower but stalled in the second
follower. Passive mapped metadata showed admission Abort/Failed with generation
zero while an earlier packet had overwritten the bank signal with its ordinary
terminal publication. The peer epoch resolver now rejects the admission Abort
it already acquires, before interpreting generation or bank signals. Activating
only that change turned both exact native failures green in 7.52 seconds with
clean drivers; the successful path adds no acquire load.

A separate controlled interleaving found that native rearm could overwrite
Abort with ready zero/one after authenticating a different atomic state word.
Close/open now compare-exchange the original ready word; Failed also terminates
an unsuccessful grant-acquisition attempt. Deterministic host replay uses the
same stateless production policy and real atomic operations. Neither repair
changes deadlines, model arithmetic, payload extents, or success criteria.

## Real-model evidence

The unchanged four-shard Qwen3.8-Flash-Next UD-Q4_K_XL and Q8_0 predictor run
with FP32 activations, FP16 KV, eight CUDA-resident experts, 504 apportioned
ROCm experts, RAM ngrams, DynamicOrdinal movement, and retained MTP capacity
three. RAM ngrams do not create a CPU expert participant.

Before the final abort-only repairs, three fresh exact model cases passed:

| Mode | Unique byte-exact stage/row comparisons | Prefix histories | Completed movement transactions | Elapsed including shutdown |
|---|---:|---:|---:|---:|
| Fixed1 | 1,208 | 4 | 4 | 236.90 s |
| Fixed3 | 2,416 | 4 | 4 | 239.54 s |
| Dynamic | 2,416 | 4 | 8 | 255.45 s |

All six original device contexts retired with zero canonical allocation bytes.
Dynamic MTP advanced three controller windows, attempted nine draft tokens and
executed four verifier transactions with serial-token-exact output. Peak RSS
was 48.62 / 48.77 / 49.67 GiB respectively. These are instrumented functional
runs, not Release throughput measurements or new PyTorch/KL qualification.

## Final focused qualification

The final coherent Integration build passes with CUDA SM80/86/89/90 and ROCm
gfx906/gfx1100. Canonical spill guards remain enabled. Owning-object resource
inspection covers both packet and epoch kernels on all six architecture targets;
the final normal build links the complete selected owning targets.

| Final check | Result |
|---|---|
| Focused host registrations | **57 pass**, 14.03 s; no skips |
| Focused GPU registrations | **16 pass**, 37.18 s; no skips |
| Original-slot cancellation across fresh process lifetimes | **20 CUDA + 20 ROCm pass**, 113.35 s; no retries after failure |
| Captured peer-epoch abort frontiers | Nine traces repeated 20 times per backend, including overwritten bank signals and a valid but cancelled successor descriptor |
| Final six-GPU dynamic-MTP inference | **Pass**, 266.07 s including shutdown; 2,416 byte-exact stage/row comparisons, four prefix histories and eight completed migrations |

The final model repeats the exact adaptive witness: three controller windows,
nine attempted draft tokens, four verifier transactions, and output identical to
the serial oracle. All six original contexts retire with zero canonical
allocation bytes, and all four MI50s return to their 10,866,688-byte baseline.
All focused native and model driver windows are clean.

Three permanent device-free replay tests exercise the same close/open/failure
policy as the native implementation. The prior Abort-overwrite policy fails the
cancellation test while ordinary-success and load-count siblings pass. The
captured packet regressions cover full, partial, empty and cancelled work with
actual live-byte assertions. Both the host and native defect regressions have
explicit ProductionTestPreflight registrations; the complete suite was not run.

The final process high-water RSS was 50.20 GiB; sampled RSS peaked at 49.86 GiB.
A local observer initially looked for the MTP matrix executable, but this exact
canonical cell belongs to the production matrix executable. The observer was
attached to that original process without restarting inference. Sampling began
after setup; the process-owned VmHWM retains the full lifetime high-water value.
This observer correction is not a production change or an inference failure.

This closes the focused recurrent inference/cancellation slice. It does not
qualify every ngram placement or topology, full numerical panels, Release economy,
public HTTP generation, a complete preflight run, or a published image.


## Evidence and cleanup

Local build and test evidence is under `/tmp/qfn-recurrent-integration-r723/`.
The failing cancellation logs, epoch-only causal control, focused host/native
reports, per-model CSV audits, driver windows, and process/sysfs resource samples
are retained. Model-owned prefix cache directories are removed only after the
original processes and all native contexts retire. Model/reference data and
unrelated workspace files are untouched.

The final run removed ten owned prefix-cache files (1,424,419,122 bytes) after
retirement. Initial fixed1/fixed3/dynamic runs also removed their owned caches;
all CSVs, diagnostics and resource observations remain available.
