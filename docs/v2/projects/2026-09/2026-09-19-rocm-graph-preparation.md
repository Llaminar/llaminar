# ROCm graph preparation: RCCL capture dependency cost

## Reproduction and attribution

The Release HTTP server loading `Ornith-1.5-35B-Q8_0.gguf` on four MI50s,
with RCCL LocalTP, rebalanced ExpertOverlay, and dynamic-depth MTP, spent
roughly 75 seconds materializing its retained serving graph family after model
initialization. Four participants prepared in parallel, but the graph variants
were sequential on each device. A representative `main_decode` graph took
7,822 ms to record 650 stages and 1,528 native nodes; 121 collective stages
accounted for nearly all of that recording time. Graph finalization itself
was about 18 ms.

Sampling the process during capture placed the rooted broadcast workers inside
RCCL's `ncclStreamAdvanceToEvent`. In the pinned ROCm 7.2.4 RCCL source, that
routine created a temporary HIP stream, waited on the completion event,
queried capture dependencies, rewrote the target stream's dependencies, and
destroyed the stream for each captured collective. This is an RCCL graph
recording cost, not a model-weight I/O cost or an inference-time GPU kernel.

## Change and evidence

The source-controlled
`scripts/docker/patches/rccl-hip-capture-event-wait.patch` gives HIP capture a
direct `hipStreamWaitEvent` dependency. CUDA keeps its upstream fast-forward
path. `apply-rccl-capture-patch.sh` applies the hunk idempotently to the pinned
source and fails if the upstream text no longer matches. Both local CMake
source builds and release Docker builds apply it; their RCCL build identity
includes the patch content.

On the same machine and model, the four-device serving-family preparation fell
to about 14 seconds. The representative `main_decode` native recording fell
to 46–53 ms while its 1,528-node graph shape and roughly 20-ms finalization
remained unchanged. The full family still includes all three prefill buckets,
serial decode, restored-prefix MTP bridge, and retained MTP variants; no
graph, collective, or precision mode was disabled.

The remaining roughly 14 seconds per participant is spread across the actual
retained family: about 5.8 seconds for MTP forward executables, 5.6 seconds
for the three prefill buckets, and 3.0 seconds for serial decode plus the
restored-prefix bridge. These phases run concurrently across the four MI50s;
the per-participant timings must not be summed into server wall time.

The focused ROCm graph-capture preflight cases passed, including TP4 captured
epoch transactions and rooted canonical publication across the verifier-row
range. The rooted test now explicitly joins its upload event to the consuming
stream before capture, as required by `TransferEngine`. Its 64-broadcast stress
case bounds executable-node growth and checks byte-exact replay. HIP reports
about 2,080 transitive edges for this 65-node single-stream graph, so edge
count is not a useful linear-size assertion here. A real Release HTTP request
on the same four-MI50 dynamic-MTP topology also completed successfully.

The first full needle run passed every HTTP and long-context accuracy check,
but its final PerfStats post-check used an obsolete literal `capture` phase
check. Production now emits a ready `materialized_without_launch` phase during
setup and a matching `replay` on use. The shared graph-capture policy already
verifies that exact lifecycle; the redundant shell-level check was removed so
the harness has one authority.

After this correction, the complete Unit gate passed 658/658, the complete
model-free production preflight passed 244/244 (including the new rooted RCCL
capture regression), and the fresh Release four-MI50 dynamic-MTP HTTP suite
passed 43/43 checks. Its full long-context tier included three 5.2k-token
needle positions, strict multi-needle JSON recall, 2,048-token structured
generation, prefix-cache/reset checks, and clean GPU release. These are
correctness results; they do not certify inference throughput.
