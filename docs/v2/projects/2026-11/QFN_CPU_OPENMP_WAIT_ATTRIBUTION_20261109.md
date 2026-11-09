# QFN CPU OpenMP wait attribution

Closed offline diagnosis and matched wait-policy controls, 2026-11-09. The final focused ROCm 10.1 host, HIP, NUMA, inventory and repaired RCCL gates remain green. The original profile's recorder-retirement receipt stays red. Both reduced-spin candidates regress the matched model; retain the unchanged GNU wait policy. No throughput improvement is qualified here.

## Original process and runtime identity

The original two-rank NUMA-correct model/session owner supplies the process tree. The attachment records PID 1145389/discovery-rank 0 and PID 1145390/discovery-rank 1, with their original executable identity, namespace PID and start ticks. It excludes the public bootstrap launcher. Execution selection `[1,0]` maps those discovery identities to execution-ranks 1 and 0 respectively; this mapping comes from the exact configuration, not PID order. The attachment itself deliberately leaves execution_rank=null; the offline phase table applies that declared mapping.

Both actual compute processes map the same physical GNU provider, `/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0`, with identical device/inode/size identity (inode 64508206, 352304 bytes). Complete saved mapping lines and the configured OpenMP runtime binding evidence establish this is GNU libgomp, not an SDK compatibility alias containing LLVM libomp. No runtime substitution, preload or CPU-budget change was used. These observations belong to this profiling lifetime; they do not retroactively fill the older passive benchmark’s missing rank→PID publication.

## Capture and complete phase windows

The actual recorder uses `cycles/period=16000000/u` and period 1 for all four original Prefill/Decode entry/return events, with `CLOCK_MONOTONIC` for samples and boundaries. It attaches to both original compute PIDs. Phase attribution accepts only complete original entry/return pairs: the last two measured requests, iterations 2 and 3, produce two complete Prefill and two complete Decode windows (eight boundary events). The first measured request is not silently reconstructed.

| Measured iteration | Original phase | Complete boundary duration ms | Native reported phase ms |
|---|---|---:|---:|
| 2 | prefill | 1875.367386 | 1875.206091 |
| 2 | decode | 6919.475899 | 6919.459029 |
| 3 | prefill | 1743.268584 | 1743.246225 |
| 3 | decode | 6911.444475 | 6911.432767 |

These are instrumented phase boundaries, not a new passive throughput baseline. The extraction retains 179,741 cycle-sample observations; 141,077 fall inside the accepted windows and 38,664 are discarded outside them. No lost records or unparsed records are reported. Cycle periods, not symbol line counts or children-inclusive percentages, determine shares below.

## Exclusive user-cycle attribution

Each row pools the two complete windows for one phase and one actual process. The denominator is all included user-cycle sample periods in that process, across the observed TIDs; it is not a sum of CPU endpoint wall timers or a fraction of model elapsed time. OpenMP workers dominate the identified runtime samples, but the process denominator also includes observed main/helper threads. Decode windows include the original MTP work, not a fabricated M1-only cohort.

| Execution rank / PID | Phase | Samples | Observed TIDs | Total user-cycle periods | GNU libgomp share | Identified pause-loop share of total | Pause loops within libgomp |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 / 1145390 | decode | 60,069 | 36 | 961,104,000,000 | 75.748% | 75.397% | 99.536% |
| 0 / 1145390 | prefill | 12,458 | 35 | 199,328,000,000 | 59.215% | 59.079% | 99.770% |
| 1 / 1145389 | decode | 57,737 | 29 | 923,792,000,000 | 81.504% | 81.275% | 99.719% |
| 1 / 1145389 | prefill | 10,813 | 29 | 173,008,000,000 | 67.937% | 67.780% | 99.769% |

GNU libgomp therefore accounts for about **59–68% of Prefill** and **76–82% of Decode** sampled process user cycles. Almost all of that runtime share—99.54–99.77%—is concentrated in two saved machine-code loops. In the exact mapped provider, offsets 0x256c0–0x256cf and 0x258a0–0x258b1 contain `pause`, a bounded counter, a polled memory value and a backward branch. Adjacent paths can enter a syscall-based wait. The saved disassembly proves the busy-wait bodies; stripped internal symbols do not prove a specific caller or synchronization reason. The nearest exported-symbol annotation must not be read as a call to `omp_get_num_procs`.

## What this does and does not establish

The result is substantial CPU resource consumption in runtime waits. It does **not** establish “80% OpenMP wall overhead”, an 80% removable latency budget or an Amdahl speedup. Many workers can spin concurrently while another worker, MPI peer or GPU makes necessary progress. A sleeping/futex wait contributes few user-cycle samples. The same pause bodies can serve required barriers and idle team waits; exclusive IP sampling without callchains does not distinguish those call sites.

The complete routed FFN already shares one team across gate/up, SwiGLU/Q8 and down. This evidence does not justify claiming three removable team launches, deleting producer/consumer barriers, changing arithmetic, or reopening rejected scheduling work. Existing endpoint host clocks have their own cohorts: root measured requests and follower lifetime aggregates cannot be added or normalized into a fabricated common request set.

## Recorder retirement and original red receipt

`observed2/model/native/openmp-profile/retirement.json` records perf_exit_code=-2, perf_retired=true and error="perf exited with status -2". The original helper treated its own requested SIGINT as failure. **That original receipt remains red and unchanged.** Offline parsing found complete bounded event windows with no reported lost/unparsed records; it does not convert the original run to a green collection certificate.

A separate controlled invocation of the actual same recorder, with a sleep workload and requested SIGINT, records shutdown_sent_sigint=true, returncode=-2, group_empty=true and accepted_exit=true. Its log states that perf captured and wrote the data file, and the retained output is 102,288 bytes. This demonstrates expected requested-SIGINT retirement/flush handling; it is not an inference/performance measurement and does not excuse unexpected signals, recorder errors, or an unretired process group.

## Closed matched wait-policy controls

**Decision: keep the existing GNU wait policy unchanged.** Both lower-spin controls reduce observed model throughput substantially; high sampled pause-loop cycles did not imply removable wall latency.

The original admitted NUMA-correct model owner ran the same 512-prompt/256-output workload, one warmup and three measured iterations, on the same recorded executable/decoder generation. Rates below pool actual phase times. Decode uses the 255 after-prefill token steps per iteration; the separate native 256-token decode rate is not substituted.

| Actual waiting input | Prefill tok/s | After-prefill Decode tok/s | Prefill versus unset | Decode versus unset |
|---|---:|---:|---:|---:|
| Both variables unset | 299.559883 | 38.993654 | — | — |
| GOMP_SPINCOUNT=0; OMP_WAIT_POLICY unset | 199.393089 | 29.381258 | -33.44% | -24.65% |
| GOMP_SPINCOUNT=1000; OMP_WAIT_POLICY unset | 203.444590 | 32.726287 | -32.09% | -16.07% |

Both comparisons pass exact equality for all 768 generated IDs and complete MTP work, configuration except owned output paths, and initial placement (147 execution-rank 0 /49 execution-rank 1 records). Native/driver closure, process retirement and observation checks pass. Actual waiting inputs were observed in **both native MPI discovery-rank processes** and match the requested values: baseline has GOMP_SPINCOUNT and OMP_WAIT_POLICY absent; each candidate has only its specified GOMP_SPINCOUNT value, with OMP_WAIT_POLICY absent. Unset is an observed state in this baseline, not a value inferred from missing older evidence. Executable/decoder generation bindings are retained.

[GCC's GOMP_SPINCOUNT documentation](https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fSPINCOUNT.html) defines the spin budget before passive waiting. Zero immediately sleeps; 1000 permits a short spin budget before sleeping. The unchanged unset policy is not equivalent to zero. These controls test actual wait behavior rather than a different GEMM, team size, route arithmetic or dispatch policy.

All outputs and initial placement match, but completed dynamic movement differs across requests; the comparison retains those transaction/command/byte observations. This is a closed matched-work observation, not proof of identical joint expert routes or an isolated causal estimate of wakeup cost from one lifecycle per arm. It nevertheless provides no basis to promote either candidate: spin0 loses 33.44% Prefill/24.65% Decode; spin1000 loses 32.09%/16.07% in these controls. No “80% wall overhead” or model speedup follows from the earlier user-cycle shares. No default wait-policy change is made, and these two diagnostics are closed.

All three original run-owned payload archives were inventoried and reclaimed only after full original retirement. Each contains 16 payload entries plus metadata (18 files), with no payload reads:

| Arm | Logical bytes reclaimed | Metadata bytes read | Payload bytes read | Remaining entries |
|---|---:|---:|---:|---:|
| unset | 727,912,500 | 28,676 | 0 | 0 |
| spin0 | 727,912,485 | 28,661 | 0 | 0 |
| spin1000 | 727,912,501 | 28,677 | 0 | 0 |

These roughly 0.728 GB logical archives are distinct owned run data, not SDK/model/shared caches. Reclamation receipts remain beside each closed model result. Eight-row arithmetic and broader team/lifecycle changes remain separate work; neither is qualified by this wait-policy diagnostic.

## Closed evidence

The original profiling artifacts are under `/tmp/qfn-openmp-profile-root/`:

- `analysis/phase-attribution.json`: accepted windows, sample periods, per-process shares, discarded samples and original retirement limitation.
- `analysis/libgomp-hot-assembly.txt`: already acquired exact-provider disassembly; no new ELF acquisition for this note.
- `observed2/model/native/openmp-profile/attachment.json` and `retirement.json`: actual process/runtime/event policy and original red helper result.
- `requested-sigint-control.json`, `.log`, `.data`: independent actual-recorder requested-shutdown control.

The closed wait-policy observations are under `/tmp/qfn-openmp-wait-root/`: `spin0-vs-unset.json`, `spin1000-vs-unset.json`, and each arm's `model/native/openmp-wait/result.json`, model/native closure and `model/prefix-reclamation.json`. The fresh unset control 299.559883 Prefill/38.993654 after-prefill Decode is the comparison baseline; the earlier 300.092/38.357 passive run remains its own separate provisional observation. Instrumented phase timings are not pooled into either passive result.
