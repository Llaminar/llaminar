# QFN GDN pre-attention on MI50

The retained projection handles one ordinary decode row on the measured
tensor-sharded participant. The real Qwen3.8-Flash-Next UD-Q4_K_XL model
has 36 GDN layers in its 48 main layers. At TP4 each GDN participant
produces 2,560 QKV columns, 1,536 Z columns and 12 columns each for
alpha and beta. It consumes the original prepared Q8_0 matrices for
QKV/Z and contiguous FP16 weights for alpha/beta. The latter use
FP16 round-to-nearest-even operands and the staged FP32 accumulation order.

The original layer frontier supplies either a residual write that the kernel
can absorb or a trunk already published by embedding or injection. In
the second case the producer remains in the graph. The participant retains
the original injection consumer, its norm input, raw GDN projection tensors,
and all later GDN state publications. Each retained entry uses admitted
serial-family scratch on the original stream. The full graph determines
which diagnostics and original output tensors remain live.

## Reproduction

Use a Release build with ROCm and the native test targets enabled. The native
case needs one gfx906 wave64 MI50 with 60 compute units. The host case
constructs the participant graph from typed declarations and covers both
original trunk sources and independent injection.

```bash
cmake --build "$BUILD_DIR" --parallel --target v2_test_retained_gdn_pre_attention_lowering v2_test_qwen4exp_device_primitives
ctest --test-dir "$BUILD_DIR" -C Release --no-tests=error --output-on-failure -R '^V2_Integration_(RetainedGDNPreAttentionLowering|ROCmRetainedPreAttentionGDN)$'
```

The native fixture compares the actual entries against the existing staged
engines on the same stream. The six combinations cover a previous write,
an in-place write and a published input with each FP16 policy. Each
capture ran 40 live and 14 interleaved empty launches with reset. The
original M1 HC/FFN object retained byte-identical instructions and compiler
resource metadata on gfx906 and gfx1100. Native inference selection for
this retained GDN kernel is limited to its qualified geometry.

## Matched full-model workload

Run the existing QFN TensorSharded ROCm4 Release benchmark on a single
rank-local MI50 P2P clique. Use all GPU experts, RAM ngrams, FP32
activations, FP16 KV, two-term HC up, single-Q8 GDN QKV, FP16 MoE
alpha/beta and the FP16 router. Keep MTP and prefix caching off in this
ordinary comparison. Set `LLAMINAR_QFN_RETAINED_M1_HC_FFN_CANDIDATE=1`
and `LLAMINAR_RCCL_RANK_ORDERED_PUSH=1` for both arms. Vary only
`LLAMINAR_QFN_RETAINED_GDN_PREATTENTION` from 0 to 1 on the same Release.

Use the same frozen review (512 IDs) and code (2,048 IDs) inputs for both
engines. Each prompt has one warmup and three measured generations of
256 tokens. Prefill uses all input tokens and decode uses 255 tokens
after the first. Use profiler-free samples for the throughput medians.

| Prompt | GDN off prefill / decode tok/s | GDN on prefill / decode tok/s | Castagna prefill / decode tok/s |
| --- | ---: | ---: | ---: |
| Review | 1040.08 / 78.21 | 1040.51 / 87.26 | 878.32 / 86.17 |
| Code | 1008.43 / 74.26 | 1008.43 / 82.02 | 1003.65 / 75.93 |

The six GDN-on outputs matched both their GDN-off histories and the earlier
peer-transport control. Native capture on all four participants showed 466
fewer nodes and 178 fewer stage objects in each ordinary decode graph.
The prefill captures had identical node counts. The independent symbol
census identified the actual GDN kernel at all 36 sites in each of four
decode captures per GPU. Its first analysis included the new GDN symbol in
the FFN-only count. Reclassifying the original capture by exact operator
also accounted for all 192 retained FFN nodes and their bindings per GPU.

The pinned Castagna Veloce revision was
`3483d462715d61f21ed5877836be35a724446359`. Each engine repeated its own
outputs; their first differing token was at position 0 on review and 44
on code. Mean KL, top-1 and top-5 use the separate frozen-logit panel.

## Final Release qualification

The final source build used GCC with GNU OpenMP, ROCm 10.1 and the shipped
gfx906/gfx1100 targets. Its same-binary GDN off/on runs measured 78.83/86.43
decode tok/s on review and 74.05/81.37 on code. Prefill measured
1038.97/1038.32 and 1008.64/1008.04 respectively. All six 256-token outputs
matched, both driver observations passed and all four devices returned to
their initial VRAM use.

The frozen PyTorch panel used 16 original histories, six teacher-forced
decode positions per history, and seven repeated rows after reset. All 119
final FP32 logit rows matched byte for byte between the GDN arms. Against the
reference, mean KL was 0.014315 on prefill and 0.022115 on decode. Top-1
agreement was 93.75% and 94.79%. Mean top-five overlap was 96.25% and
91.04%. Both arms passed the same thresholds. Each candidate participant
recorded 180 inline GDN capture sites across the graph variants, and each
control participant recorded zero. The frozen panel, native replay, and
full-model speed runs used the same final Release core.

## Grouped verifier projection

The grouped HC-up projection has a separate opt-in on MI50. On the original
two-term Q8_0 HC-up shape, each worker loads one 32-weight block and uses it
for up to four verified rows. Output owners fold each row's ten blocks in the
serial order, then add its two completed terms. The existing activation
publisher and device-owned live count supply the inputs. The kernel uses
10 KiB of workgroup memory and the existing projection destination.

The native replay captures the real dispatch and checks its symbol. It
compares the output, both quantized activation banks, scales, and untouched
guard bytes with independent serial launches. Physical capacities 2 through
8 cover every live count including zero, reset and repeated execution.

```bash
cmake --build "$BUILD_DIR" --parallel --target v2_test_qwen4exp_device_primitives
ctest --test-dir "$BUILD_DIR" -C Release --no-tests=error --output-on-failure -R '^V2_Integration_ROCmGroupedHCUpShortK$'
```

The matched four-GPU benchmark used the ordinary workload above with the
same Release binary in both arms. Set
`LLAMINAR_QFN_RETAINED_GDN_PREATTENTION=1` and vary
`LLAMINAR_QFN_ROCM_GROUPED_HC_UP_SHORT_K` from 0 to 1. The fixed controller
used three drafts. The dynamic controller admitted 1–15 drafts. Both used a
separate MTP-Off run of the same binary for the serial token comparison.

| Verifier policy | Review control / candidate decode tok/s | Code control / candidate decode tok/s | Candidate prefill tok/s, review / code |
| --- | ---: | ---: | ---: |
| Fixed depth 3 | 99.43 / 106.17 | 82.56 / 88.04 | 1001.45 / 970.32 |
| Dynamic depth | 105.65 / 111.05 | 86.25 / 90.73 | 998.66 / 969.34 |
| Castagna fixed depth 3 | 114.89 | 106.31 | 774.28 / 888.35 |

All 12 measured candidate generations matched their 256-token serial
histories and their respective control's verifier work. Review ran 74
verifications at fixed depth and 58 at dynamic depth. Code ran 86 and 69.

A separate rocprof fixed-depth review reproduced the unprofiled tokens and
work. All four cards executed the new kernel 97 times per verification.
Summed HC-up time fell from 3.48–3.54 to 1.34–1.35 ms per verification per
card. The earlier trace used RCCL gather/fold, while the newer one used the
qualified direct peer reduction. Together these changes reduced the
verifier graph from 2,392 to 2,295 nodes. Summed dispatch time fell from
34.30–34.51 to 28.68–28.79 ms per card per verification. The same-binary
unprofiled comparisons in the table isolate the grouped projection.

## Recreate the exact fixed-depth workload

Run from the source checkout. The source revision fixes both prompt texts.
The two digests allow the input to be checked before loading the model.
Use a fresh output directory for each arm.

```bash
python3 - "$RUN_DIR" <<'PY'
from pathlib import Path
import hashlib
import json
import subprocess
import sys

output = Path(sys.argv[1]).resolve()
output.mkdir()
def original(path):
    return subprocess.check_output(["git", "show", "e6dc6fe37:" + path], text=True)
review = original("benchmarks/prompts/qwen36_mtp_fixed.txt").strip() + " test" * 88
source = original("scripts/ci/run_model_parity_benchmarks.py")[:8843]
code = 'Review this Python inference test tooling. Identify one concrete correctness defect and propose a minimal repair with a regression. Explain your reasoning.\n\nFile: scripts/ci/run_model_parity_benchmarks.py\n```python\n' + source
prompts = {"review512": review, "code2048": code}
expected = ['5cbdeeafc357d7f0036f47ce25ba826809ca3f95f0fc6175f687a13a4ddf77b1', 'acece79e90a3f5e41dd25200a6c6a26eef9a0847f06e8fd6d2235003a8e7ad18']
assert [hashlib.sha256(v.encode()).hexdigest() for v in prompts.values()] == expected
campaign = {
    "schema": "llaminar.benchmark-campaign.v1", "output_directory": str(output / "campaign"),
    "sampling": {"temperature": 0, "top_k": 1, "top_p": 1.0, "seed": 42},
    "prompts": [{"id": k, "partition": "holdout", "text": v} for k, v in prompts.items()],
    "cells": [{"id": k, "prompt_id": k, "mtp": {"mode": "fixed", "depth": 3}} for k in prompts],
}
(output / "campaign.json").write_text(json.dumps(campaign))
PY
```

Use a Release build, its matching session helper, and a saved automatic
TensorSharded TP4 configuration with a 32,768-token context. On the
comparison host the two MPI ranks bind to their respective 28 physical
cores. The GPU-owning rank has all four participants behind the same switch.
Set `GROUPED` to 0 or 1 and use the same config and rankfile for both arms.

```bash
export LLAMINAR_BENCHMARK_ITERATIONS=3 LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=1
export LLAMINAR_QFN_FA_OUTPUT_ACTIVATION=single_q8 LLAMINAR_QFN_GDN_OUTPUT_ACTIVATION=single_q8
export LLAMINAR_GDN_PROJECTION_ARITHMETIC=fp16_moe_alpha_beta LLAMINAR_QFN_GDN_QKV_ACTIVATION=single_q8
export LLAMINAR_QFN_HC_UP_ACTIVATION=two_term_q8 LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16
export LLAMINAR_QFN_RETAINED_M1_HC_FFN_CANDIDATE=1 LLAMINAR_RCCL_RANK_ORDERED_PUSH=1
export LLAMINAR_QFN_RETAINED_GDN_PREATTENTION=1 LLAMINAR_QFN_ROCM_GROUPED_HC_UP_SHORT_K="$GROUPED"
mkdir -p "$RUN_DIR/mpi-session"
"$BUILD_DIR/llaminar_mpi_session" --session-base "$RUN_DIR/mpi-session" -- \
  mpirun -np 2 --rankfile "$RANKFILE" --bind-to core --report-bindings \
  --mca mpi_leave_pinned 1 --mca orte_allowed_exit_without_sync 1 -- \
  "$BUILD_DIR/llaminar2" benchmark --config "$QFN_TP4_CONFIG" -n 256 \
  --temperature 0 --top-k 1 --top-p 1 --seed 42 --benchmark-campaign "$RUN_DIR/campaign.json" \
  --mtp --mtp-depth-policy fixed --mtp-draft-tokens 3 --mtp-graph-capacity-draft-tokens 3 \
  --mtp-min-draft-tokens 1 --mtp-max-draft-tokens 3 --mtp-initial-draft-tokens 0
```
