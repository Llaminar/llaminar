# QFN retained decode on MI50

## What the 70 µs gate measures

One captured MI50 kernel performs the complete local post-attention hyperconnection,
router and expert/shared FFN operation for a single decode row. Timing ends before
the collective. The model’s separate FP32 HC injection node lies outside this
final-read measurement. The input weights are layer 1 of Qwen3.8-Flash-Next-UD-Q4_K_XL.
The comparator selects Castagna TP4 participant 3, with routed columns [384, 640) and shared columns [448, 640)
of the two 640-wide source matrices. Routed gate/up weights use Q4_K, routed
down uses Q5_1 and the three shared projections use Q8_0. H is 2560, E is
512 and top-K is 10.
Both arms start from the same seeded boundary activations and selected expert sets.
Llaminar retains its own arithmetic, movable expert storage and route histogram.
The fork uses its original arithmetic and omits the histogram.

The comparison uses the unchanged Castagna device section from commit
`3483d462715d61f21ed5877836be35a724446359`. Its exporter checks the revision,
source bytes and license. A separate 24-node staged graph provides Llaminar's byte
oracle. Each session checks 20 histories, reset, repeated and empty work, and
active work after empty. The measured arms each capture one kernel. Thirty-two
balanced ABBA/BAAB quartets each time 32 local operations per event interval.
The reported latency is the median of all 64 observations per arm. The paired
difference is the median of the 32 within-quartet differences.

Llaminar's automatic four-MI50 allocation for this model assigns 160 shared
columns per participant and distributes the routed columns as 128/128/128/256,
rotating the larger shard across layers. The measured S192 entry therefore covers
none of those 192 automatic layer/participant assignments. Of the 48 layers,
43 have the measured routed weight types. Layers 4, 30, 46 and 47 use Q8_0 down,
and layer 2 uses both Q5_K gate/up and Q8_0 down. Acceleration of normal serving
requires the S160 shapes and these additional formats, followed by a matched
whole-model throughput and accuracy run.

The pinned fork accepts shared widths in multiples of 64. Its down projection
consumes pairs of 32-value blocks. A direct launch at S160 would leave the last
32 values unprocessed. For the S160 implementation, check each local result
against Llaminar’s staged graph. Compare speed across engines on the complete
TP4 operation, using each engine’s own four slices of the same source weights
and including the collective and FP32 HC injection.

For TP1, TP2, TP4 and TP8, obtain both source intervals from the prepared
participant bindings and retain their offsets, widths and formats in each
measurement. Include the last full or partial block of every slice in the byte
oracle. Compare narrower and wider shards with the same selected experts and
separate routed FFN time from the HC, router and shared work common to both.
The slowest member and its collective determine TP latency. A mean across the
four cards can hide a persistent bottleneck on the wider shard.

## Checkpoint results

Three fresh Release processes on MI50 `0000:97:00.0` used the checkout's
pinned ROCm 10.1 SDK. Each session median had to be at most 70 µs and its
paired difference had to favor Llaminar. The device retired after every process
and the host driver observer recorded no errors during these sessions.

| Session | Llaminar (µs) | Castagna (µs) | Paired difference (µs) |
| --- | ---: | ---: | ---: |
| 1 | 69.70620 | 75.08620 | −5.274325 |
| 2 | 69.71870 | 75.10370 | −5.337500 |
| 3 | 69.62370 | 75.08790 | −5.503750 |

The winning body packs the expert activations by their live column width and
puts the shared activations directly after them. A final consumer can address
these values without integer division or visiting unused capacity. The four
components of the Q8 maximum each follow the reference lane tree before the
component maxima are combined. This preserves exceptional-value behavior and
reduces cross-lane work. The same source was compiled for gfx906 and gfx1100.
The eight device entries in the branch build match the measured build on their
respective architectures. The performance gate ran on gfx906.

An earlier body reached 69.79–69.98 µs on PCI97 with a private HIP/ROCr build.
Its ordinary-SDK result was 71.09–71.63 µs in three isolated sessions. That
difference motivated the live-width and lane-tree changes above. The final
three measurements use the ordinary SDK throughout.

The implementation keeps the ordered per-expert accumulation and FP32 add
operand order. Two lanes cooperate on an output with an eight-lane exchange.
An idle wave reloads the published route ledger. The router orders packed keys
with a native binary64 compare while preserving original FP32 weight bits.
The existing bounded positive Q8 scale can use the native half conversion on
this entry. The maximum reduction tree matters even when the scales of ordinary
finite inputs agree. Regrouping that tree changed 96 exceptional rows in both
the HC and FFN probes, and the shipped body preserves its order.

The strict compile covers gfx906 and gfx1100. The measured gfx906 entry uses 62
VGPRs, 104 SGPRs, 46,656 bytes of LDS and zero private memory. Its compiled size is 37,696 bytes. The selector
passed 93,702 native prefix comparisons. Scale conversion passed 190,464
inputs per replay, including all positive finite half words, representable
midpoints and their immediate neighbors. The native HC and FFN regressions also
exercise 80,327 exceptional rows, including signed zero and signaling NaNs. A six-MI50 replay with two socket
ranks and three two-member RCCL cohorts verified that all six participants
consumed new expert bytes and published changed outputs in the second epoch.
Grouped MTP performance, whole-model throughput and the other weight-format
specialisations require their own measurements. This entry is opt-in through
`LLAMINAR_QFN_RETAINED_M1_HC_FFN_CANDIDATE=1` for an eligible graph.

## Comparing the remaining weight formats

Each of the 15 decoder-group combinations needs an independent captured timing,
byte replay and per-device Roofline. Record the actual gate, up, down and shared
formats, M/H/F/S, selected expert count and compulsory prepared bytes. Measure
attainable bandwidth and issue rate for the relevant unpack/dot/lookup mix on
the same card. Take the maximum of the memory and compute terms within a phase,
then sum phases that must execute sequentially. Compare that floor with unprofiled
event time. Measured bus traffic, issued instructions and barriers explain excess
cost. Validate the model on held-out geometries before setting efficiency gates.

The existing M1/H2560/F256/S256 same-binary baseline illustrates the difference.
For all-Q4_0, routed and shared FFN weights occupied 12,165,120 GGUF source bytes
for one selected set and the complete operation issued 3.04 million vector wave
instructions, taking 152.03 µs. All-IQ2_S used 6,927,360 equivalent source bytes
yet issued 4.01 million vector wave instructions and took 190.03 µs. This makes
the IQ decode and lookup paths an early profiling target. These measurements
belong to the earlier S256 implementation. The retained S192 kernel and each
remaining family need their own measured bounds.

## Reproduce on a local MI50

Install the checkout's pinned ROCm and test build dependencies. Set
`QFN_GGUF` to the absolute path of shard 00001-of-00004 and put all four GGUF
shards in the same directory. The first shard used above has SHA-256
`4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082`.
Set `CASTAGNA_CHECKOUT` to a clean copy of the pinned fork. Set `BUILD_DIR`
and `RESULT_DIR` outside the source tree. Use a fresh results directory.
Choose an idle rank-local MI50 with `QFN_PCI`.

```bash
set -euo pipefail
unset HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL CUDA_VISIBLE_DEVICES LD_LIBRARY_PATH
: "${ROCM_PATH:?}" "${QFN_GGUF:?}" "${QFN_PCI:?}" "${CASTAGNA_CHECKOUT:?}" "${BUILD_DIR:?}" "${RESULT_DIR:?}"
git -C "$CASTAGNA_CHECKOUT" rev-parse HEAD
cmake -S src/v2 -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$(command -v ninja)" -DCMAKE_BUILD_TYPE=Release \
  -DHAVE_ROCM=ON -DLLAMINAR_BUILD_TESTS=ON -DLLAMINAR_SHARED_CORE=ON \
  -DROCM_PATH="$ROCM_PATH" -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM_PATH" '-DCMAKE_HIP_ARCHITECTURES=gfx906;gfx1100' \
  -DLLAMINAR_CASTAGNA_SOURCE_DIR="$CASTAGNA_CHECKOUT"
cmake --build "$BUILD_DIR" --parallel --target v2_integration_moe_intermediate_ffn_finalization
mkdir "$RESULT_DIR"
export LLAMINAR_TEST_CASTAGNA_GGUF="$QFN_GGUF" LLAMINAR_TEST_RETAINED_HC_FFN_PCI="$QFN_PCI"
for run in 1 2 3; do
  ctest --test-dir "$BUILD_DIR" --no-tests=error -V \
    -R '^V2_Perf_RetainedClosedHCFFN_RealFinal_ROCm$' | tee "$RESULT_DIR/run-$run.log"
done
python3 tests/v2/integration/kernels/rocm/castagna/summarize_retained_closed.py \
  --mi50-checkpoint "$RESULT_DIR"/run-{1,2,3}.log | tee "$RESULT_DIR/summary.json"
```
