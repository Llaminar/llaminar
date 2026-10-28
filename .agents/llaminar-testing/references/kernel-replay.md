# Production kernel replay

Use this as the canonical method for isolated kernel correctness and performance
experiments. Replay the production operation with fixed, representative inputs,
retained storage and its real dispatch policy. On CUDA and ROCm, retain a native
captured graph; on CPU, repeat the production call with persistent buffers and
the measured ISA, team and NUMA placement. Extend the owning integration/perf
harness or NativeVNNI collector instead of creating a second benchmark framework.

This is a diagnostic and test procedure. It does not authorize serial row replay,
host state mirrors or alternative execution modes in production.

## Freeze the work, not just the shape

- Record backend/device, format/codebook, dimensions, strides, live extents,
  capacity, precision, partition/reduction policy and workspace generation.
  Include the inputs that determine work: expert routes and weights, masks,
  sequence positions, recurrent state and active counts as applicable.
- For a model hotspot, retain representative inputs from the existing diagnostic
  framework at the actual operation boundary. Preserve those inputs across
  candidates. Synthetic boundary cases complement this workload; they do not
  establish model accuracy or representative routing by themselves.
- For communication-free TP scaling, replay each participant's real local shape
  and recorded producer outputs, including results normally supplied by
  collectives. Keep routes and useful work fixed. Simply disabling communication
  in the live model changes its inputs and cannot measure comparable compute.
  Label these curves as isolated compute scaling; verify the complete topology
  separately, with communication enabled.
- Time a complete operation, including required producers, reducers, suffixes and
  publications. An isolated leaf is useful for diagnosis, but omitting required
  work cannot establish a pipeline win. Retain observable outputs so the compiler
  cannot remove the work.

## Prove retained replay and ownership

Allocate buffers and workspace once through existing fixtures and transfer APIs.
Keep exact non-null streams and producer/event joins. Capture the real production
entrypoint with immutable argument bindings, then reuse that executable while
changing data through ordered test setup operations. Recapture only when graph
identity actually changes; do not recapture or reallocate to hide dirty reuse.

Inspect the retained native graph outside active capture: selected function and
policy, pointer/scalar arguments, grid/block dimensions, shared storage, operation
count and dependency edges. Use setup metadata and supported inspection APIs;
an otherwise harmless runtime query may invalidate an active capture. A matching
kernel name alone does not prove the intended production operation ran.

Replay adversarial transitions through the same storage: full, partial, empty,
skewed, large-to-small and full again, where the operation admits them. Poison
scratch and output storage that the producer must overwrite; preserve legitimate
read-modify-write and recurrent inputs. Check every required output, metadata
plane, sentinel, input that should remain unchanged, and guard/unused region
whose contract requires it to remain untouched. Do not require unspecified
intermediate capacity to be zero.

Use an independent mathematical/serial oracle or an authenticated unchanged
production control. Grouped MTP rows, exact metadata and arithmetic-preserving
changes require byte equality, including copied weight bits. A deliberate
precision change also needs the applicable numerical and model-panel gates;
component tolerance cannot replace serial/grouped byte invariance.

Derive coverage from admitted formats, backends and geometry. Include selector
boundaries and neighboring values, partial tails, empty participants and every
supported grouped-row regime, rather than only the fastest model shape. Keep
the shared oracle independent of backend code and dispatch decisions.

Fixture ownership must survive failures as well as successful replays. Close an
aborted capture before terminal synchronization, and retire pending work before
graphs, buffers, staging storage or streams are destroyed. Readbacks, poisoning
and blocking result collection stay in the test boundary.

## Measure unprofiled replay

Use Release code throughout the timing executable and linked core; do not mix
Integration snapshot ABI with a Release library. Validate bytes before timing,
warm the retained executable, collect repeated native-event samples, then check
bytes and guards again after timing. Put setup, transfers, allocation, inspection
and oracle comparisons outside the timed interval.

For very short operations, capture several complete operations in a serial
dependency chain and divide graph elapsed time by that count. Choose the count
to amortize submission/event overhead, record it, and report the result as
amortized captured-operation time. It still includes native scheduling overhead;
do not subtract an invented constant or call it hardware kernel service time.
CPU measurements use the existing unprofiled timer with the same work contract.

Repeated execution must preserve comparable work. For stateful kernels, restore
the same initial state before each sample or replay an explicitly defined state
trajectory shared by both candidates. Do not repeatedly mutate state and assume
each iteration remains equivalent. Record whether buffers are cache-hot; fixed
input replay does not reproduce every cache effect of full-model execution.

Use interleaved A/B and B/A or A/B/A comparisons on the same device and workload.
Retain raw samples and the observed noise band, including losing candidates.
Do not overlap canonical timing with unrelated builds or device workloads.
Correctness sweeps and one-shape timing are separate modes of the same fixture;
timing thresholds do not belong in routine functional regressions.

## Profile separately and qualify the result

Collect hardware counters in a separate process for one exact candidate, shape
and selected physical dispatch. Bind the dispatch to the replay schedule: dirty
checks, warmups, timed repeats and terminal checks can all launch the same kernel.
Derive its occurrence from that schedule and verify actual counts. Captured node
inventory is not an execution count. Use the backend skill's counter-group,
attachment and driver-failure rules; profiler durations never enter canonical
timing samples.

Join native function/geometry/resource observations to the exact compiled symbol,
architecture, source/build and loaded runtime. Inspect registers, shared/private
storage, spills and relevant instructions across the shipping architectures.
Report unavailable counters explicitly: static resources do not establish
achieved occupancy or memory efficiency. If a private trial is promoted, rerun
the final production entrypoint and verify its dispatch/resources; prototype
results alone do not qualify the installed path.

Keep source/build and runtime identities, input/geometry description, byte checks,
raw timings, profiler/ISA joins, driver observations and clean process retirement
with the experiment. Reuse existing evidence collectors, and keep large artifacts
outside tracked source. Record measurements and decisions in project notes rather
than this procedure.

Register focused CUDA/ROCm regressions in their integration suites and explicitly
in `ProductionTestPreflight` for production defects; keep Unit device-free.
Extend the affected all-format/backend sweep. Before promotion, run the applicable
canonical accuracy panel and matched unprofiled Release model benchmark. Prove
the optimized path executed and compare equivalent token/routing/MTP work. An
isolated replay win is a candidate until that end-to-end evidence is green.

Backend mechanics remain in [CUDA tuning](../../cuda-tuning/SKILL.md),
[ROCm tuning](../../rocm-tuning/SKILL.md),
[CPU tuning](../../cpu-tuning/SKILL.md) and the
[NativeVNNI pipeline](../../nativevnni-gemm-tuning/SKILL.md).
