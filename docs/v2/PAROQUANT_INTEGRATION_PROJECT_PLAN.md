# ParoQuant Weight Quantization Integration - Project Plan

**Date**: May 7, 2026  
**Status**: Planning  
**Target**: CUDA-first W4A16 inference, CPU reference path, ROCm follow-up  
**Paper**: [ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference](https://arxiv.org/abs/2511.10645)  
**Reference implementation**: [z-lab/paroquant](https://github.com/z-lab/paroquant)

---

## 1. Executive Summary

ParoQuant is a post-training weight quantization method for LLM inference. It combines AWQ-style affine INT4 weights with a learned channel scaling and pairwise rotation transform applied to activations at runtime. The paper reports materially better reasoning accuracy than AWQ at similar throughput and storage cost, with typical runtime overhead below 10%.

For Llaminar, ParoQuant should be implemented as an **affine INT4 weight format plus per-linear activation transform metadata**, not as a standalone tensor decode method. The transformed weights can reuse much of Llaminar's current quantized GEMM infrastructure if we repack Paro/AWQ `qweight`, `qzeros`, and `scales` into a Llaminar-native asymmetric 4-bit block layout. The novel runtime work is the activation transform:

```text
y = GEMM( rotate_and_scale(x, pairs, theta, channel_scales), quantized_rotated_weight )
```

The recommended path is:

1. Build an external converter from ParoQuant HF safetensors to a Llaminar ParoQuant bundle.
2. Repack ParoQuant's AWQ-style affine INT4 weights into Llaminar-native block storage.
3. Store `theta`, `pairs`, and `channel_scales` as sidecar metadata per quantized linear.
4. Add CPU and CUDA pairwise-rotation kernels.
5. Wire the transform through `PreparedWeightStore` and GEMM stages.
6. Initially support single-device CUDA and unfused correctness paths, then restore fused QKV/gate-up performance.

This plan intentionally treats AWQ compatibility as a shared substrate. Implementing a clean `AffineInt4` packing path for ParoQuant should make plain AWQ checkpoint support much cheaper later.

---

## 2. Paper and Repository Summary

### 2.1 Algorithm

ParoQuant is W4A16 PTQ:

1. Start from full-precision or dequantized model weights.
2. Learn channel-wise scales and a sequence of pairwise Givens rotations per input group.
3. Transform weights offline.
4. Quantize transformed weights using 4-bit affine quantization with group size 128.
5. At inference time, apply the inverse transform to activations before GEMM.

The implementation uses:

| Parameter | Default |
|---|---:|
| Weight bits | 4 |
| Activation precision | FP16/BF16/FP32 runtime input, not quantized for storage |
| Quantization group size | 128 |
| Rotation rounds (`krot`) | 8 |
| Independent pairs per group per round | up to 64 |

Conceptually, the model stores a transformed weight `W_rot = T W` and computes:

```text
X W = (X T^-1) (T W)
```

The runtime sidecar transform is therefore part of the linear layer contract. A ParoQuant weight is incomplete without its activation transform metadata.

### 2.2 Reported Benefits

From the paper:

| Method | Key tradeoff |
|---|---|
| AWQ | Fast, widely supported, weaker reasoning accuracy |
| QTIP | Strong accuracy, slower inference |
| ParoQuant | Better reasoning accuracy than AWQ, close to AWQ speed, faster than QTIP |

The paper reports roughly `+2.4%` average reasoning accuracy over AWQ, with less than 10% runtime overhead. Example reported decode throughput on A6000:

| Model | FP16 | AWQ | QTIP | ParoQuant |
|---|---:|---:|---:|---:|
| LLaMA-3-8B | 45 tok/s | 120 tok/s | 95 tok/s | 112 tok/s |
| Qwen3-4B | 78 tok/s | 176 tok/s | 117 tok/s | 160 tok/s |

### 2.3 Reference Repo Format

The `z-lab/paroquant` repo exports Hugging Face safetensors checkpoints with a config entry like:

```json
{
  "quantization_config": {
    "quant_method": "paroquant",
    "bits": 4,
    "group_size": 128,
    "krot": 8
  }
}
```

Each dense quantized linear stores:

| Tensor | Type | Shape pattern | Meaning |
|---|---|---|---|
| `qweight` | int32 | `[in_features, out_features / 8]` | AWQ-packed 4-bit weights |
| `qzeros` | int32 | `[groups, out_features / 8]` | AWQ-packed zero points |
| `scales` | fp16 | `[groups, out_features]` | affine group scales |
| `theta` | fp16 | `[krot, in_features / 2]` | rotation angles |
| `pairs` | int16 | `[krot, in_features]` | pair indices for each rotation round |
| `channel_scales` | fp16 | `[1, in_features]` | runtime inverse channel scales |

The reference runtime does:

```python
x = torch.ops.rotation.rotate(x, pairs, theta, channel_scales)
out = WQLinearMMFunction.apply(x, qweight, qzeros, scales, bits, group_size, bias, out_features)
```

The CUDA kernel at `paroquant/kernels/cuda/rotation.cu` applies pairwise rotations over 128-channel groups. Current public implementation supports CUDA and MLX; it does not provide a ROCm kernel.

---

## 3. Current Llaminar Touchpoints

### 3.1 Weight Loading

Current concrete loading is GGUF-centered:

| Component | Path | Relevance |
|---|---|---|
| GGUF loader | `src/v2/loaders/ModelLoader.cpp` | Maps GGUF tensor types to Llaminar tensor classes |
| Tensor type enum | `src/v2/tensors/TensorType.h` | Existing quantized type registry |
| Quantized blocks | `src/v2/tensors/BlockStructures.h` | Native storage for Q4/Q5/QK/IQ formats |
| Tensor classes | `src/v2/tensors/TensorClasses.h` | Tensor subclasses and typed storage |
| Weight plan/materialization | `src/v2/loaders/WeightPlan.h`, `src/v2/loaders/WeightManager.cpp` | Canonical weight requirements and materialization |
| Prepared model weights | `src/v2/loaders/PreparedWeightStore.h`, `src/v2/loaders/PreparedWeightStore.cpp` | Correct ownership point for prepared GEMM handles |

ParoQuant released checkpoints are not GGUF files. They are HF safetensors plus JSON metadata. An implementation therefore needs either:

1. A Llaminar-side safetensors/HF loader, or
2. A converter from ParoQuant safetensors to a Llaminar-native bundle, or
3. A GGUF extension with sidecar tensors and metadata.

The recommended first implementation is option 2, because it keeps runtime C++ scope contained and avoids blocking on a full safetensors model loader.

### 3.2 Quantized GEMM

Current quantized GEMM flows through prepared handles:

| Component | Path | Relevance |
|---|---|---|
| GEMM factory | `src/v2/kernels/KernelFactory.h`, `src/v2/kernels/KernelFactory.cpp` | Low-level handle creation, not model lifetime owner |
| Prepared store | `src/v2/loaders/PreparedWeightStore.cpp` | Owns model prepared handles and caches |
| CUDA packer | `src/v2/kernels/cuda/gemm/CUDAWeightPacker.cpp` | Packs native quantized formats for CUDA kernels |
| CUDA quant GEMM | `src/v2/kernels/cuda/gemm/CUDAQuantisedGemmKernel.cpp` | Decode/prefill GEMM implementation |
| CPU VNNI packer | `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIWeightPacker.h` | CPU reference/performance path |
| Stage execution | `src/v2/execution/compute_stages/stages/GEMMStage.cpp` | Calls prepared GEMM kernels |

Important repository convention: model-owned prepared GEMM state should remain in `PreparedWeightStore`, not in global `KernelFactory` registries.

### 3.3 Fusion Constraints

ParoQuant changes the contract for fused projections. If two projections have different rotation metadata, they cannot share the same pre-GEMM transformed activation.

| Existing area | Constraint |
|---|---|
| Fused QKV projection | Q, K, and V may each have different `pairs/theta/channel_scales` |
| Fused gate/up GEMM | Gate and up may have different transforms |
| Row-parallel down/Wo | Rotation pairs may cross K-shard boundaries |

Initial correctness should prefer de-fusing ParoQuant projections when transforms differ. Re-fusion can come after correctness and parity are stable.

---

## 4. Target Architecture

### 4.1 Core Concepts

Add three concepts:

| Concept | Responsibility |
|---|---|
| `AffineInt4Weight` | Represents AWQ/Paro-style group-wise affine INT4 weights after conversion into Llaminar layout |
| `ParoQuantTransform` | Holds `pairs`, `theta`, `channel_scales`, `group_size`, `krot`, and shape metadata for one linear input space |
| `ParoQuantPreparedGemm` | Prepared GEMM handle plus optional transform metadata and workspace requirements |

Do not model ParoQuant as only `ITensorGemmTileDataProvider::decode_block_at()`. The weight decode path alone cannot reproduce ParoQuant accuracy, because the runtime activation transform is semantically required.

### 4.2 Data Flow

```text
Paro HF checkpoint
  config.json
  model*.safetensors
        |
        | tools/paroquant_convert.py
        v
Llaminar ParoQuant bundle
  manifest.json
  weights/<canonical-name>.affine_int4.bin
  transforms/<canonical-name>.theta.fp16.bin
  transforms/<canonical-name>.pairs.i16.bin
  transforms/<canonical-name>.channel_scales.fp16.bin
        |
        | ParoQuantBundleLoader / WeightManager materialization
        v
FrozenModelWeightSet + PreparedWeightStore
        |
        | GEMMStage / projection stages
        v
rotate_and_scale(A) -> quantized GEMM -> output
```

### 4.3 Runtime Linear Contract

For each ParoQuant linear:

```text
Input A:          [M, K]
Stored weight B:  [N, K] in Llaminar orientation, already transformed offline
Transform:        pairs/theta/channel_scales over K
Output C:         [M, N]

A_rot = ParoRotate(A, transform)
C = A_rot * B^T + bias
```

Shape invariants:

| Invariant | Requirement |
|---|---|
| `K % group_size == 0` | Initial support assumes group size 128 |
| `pairs.shape == [krot, K]` | Pair list covers the full input dimension |
| `theta.shape == [krot, K / 2]` | One angle per pair |
| `channel_scales.shape == [K]` or `[1, K]` | Runtime inverse channel scaling |
| Converted weight orientation | Llaminar stores `[N, K]`; Paro/AWQ source is effectively `[K, N]` packed |

### 4.4 Artifact Format

Use a directory bundle for the first implementation:

```text
model.paroq/
  manifest.json
  tokenizer.json or tokenizer.model
  config.json
  weights/
    blk.0.attn_q.weight.affine_int4.bin
    blk.0.attn_k.weight.affine_int4.bin
    ...
  transforms/
    blk.0.attn_q.weight.theta.fp16.bin
    blk.0.attn_q.weight.pairs.i16.bin
    blk.0.attn_q.weight.channel_scales.fp16.bin
    ...
```

Minimum `manifest.json` fields:

```json
{
  "format": "llaminar-paroquant-bundle",
  "format_version": 1,
  "source": {
    "repo": "z-lab/paroquant",
    "commit": "<recorded by converter>",
    "model_id": "z-lab/Qwen3.5-4B-PARO"
  },
  "quantization": {
    "method": "paroquant",
    "bits": 4,
    "group_size": 128,
    "krot": 8,
    "weight_storage": "affine_int4_lm_block32"
  },
  "tensors": {
    "blk.0.attn_q.weight": {
      "rows": 4096,
      "cols": 4096,
      "weight": "weights/blk.0.attn_q.weight.affine_int4.bin",
      "transform": {
        "theta": "transforms/blk.0.attn_q.weight.theta.fp16.bin",
        "pairs": "transforms/blk.0.attn_q.weight.pairs.i16.bin",
        "channel_scales": "transforms/blk.0.attn_q.weight.channel_scales.fp16.bin"
      }
    }
  }
}
```

Future GGUF integration can encode the same logical payloads as extra tensors and metadata keys once the runtime path is proven.

---

## 5. Implementation Phases

### Phase 0: Format Audit and Golden Fixtures

**Goal**: Lock down ParoQuant/AWQ packing semantics before touching runtime code.

#### Tasks

- [ ] Clone `https://github.com/z-lab/paroquant` to `/tmp/paroquant` and record the commit used for validation.
- [ ] Inspect the current branch and, if needed, the `legacy` branch mentioned by the README for paper reproduction.
- [ ] Download one small public ParoQuant checkpoint metadata file and record its `quantization_config`.
- [ ] Write a Python fixture generator under `python/paroquant/` or `tools/` that:
  - Loads a tiny synthetic linear layer with ParoQuant-style tensors.
  - Uses upstream ParoQuant or AutoAWQ helpers as the dequantization oracle.
  - Emits small binary fixtures for `qweight`, `qzeros`, `scales`, `theta`, `pairs`, and `channel_scales`.
  - Emits expected FP32 weight and expected rotated activation output.
- [ ] Document AWQ nibble order, zero-point packing, and any reorder table in the fixture README.

#### Deliverables

- Golden fixtures committed under `tests/v2/fixtures/paroquant/` or generated deterministically during tests.
- A short fixture README that explains source layout and expected orientation.
- No Llaminar runtime changes yet.

#### Acceptance Criteria

- A Python command can prove that fixture dequantization matches upstream ParoQuant/AutoAWQ within exact or near-exact tolerance.
- The conversion code knows whether source zero points represent `q - zero` or `q + zero` for every packed nibble.
- The source-to-Llaminar transpose convention is explicit.

---

### Phase 1: Affine INT4 Repack Support

**Goal**: Convert Paro/AWQ affine INT4 weights into a Llaminar-native block layout that existing packers can consume.

#### Design

ParoQuant dequantization is affine:

```text
w = scale * (q - zero_point)
```

Llaminar's asymmetric 4-bit block formats can represent the same value as:

```text
w = scale * q + min
min = -scale * zero_point
```

Because Paro uses group size 128 and Llaminar native 4-bit blocks are commonly 32 elements, the converter can split one Paro group into four 32-element blocks with repeated `scale` and `min` metadata.

#### Tasks

- [ ] Add a converter-side `unpack_awq_int4.py` helper:
  - Input: source `qweight`, `qzeros`, `scales`, `bits=4`, `group_size=128`.
  - Output: logical FP32 or affine block view in Llaminar `[N, K]` orientation.
- [ ] Add a converter-side `pack_lm_affine_int4.py` helper:
  - Writes block32 payload nibbles.
  - Writes FP16 or FP32 scale/min side channels according to selected Llaminar tensor format.
- [ ] Decide whether to reuse an existing tensor type such as Q4_1-like storage or add an explicit `AFFINE_INT4` tensor type.
- [ ] If adding a tensor type, update:
  - `src/v2/tensors/TensorType.h`
  - `src/v2/tensors/BlockStructures.h`
  - `src/v2/tensors/TensorClasses.h`
  - `src/v2/tensors/TensorKernels.h`
  - `src/v2/tensors/NativeVnniFormatInfo.h`
- [ ] Add C++ unpack/repack tests against the Phase 0 fixtures.
- [ ] Confirm CUDA and CPU native-VNNI packers accept the chosen storage type.

#### Deliverables

- A Llaminar-native affine INT4 weight blob for each ParoQuant linear.
- Unit tests proving exact or bounded reconstruction against upstream ParoQuant dequantization.

#### Acceptance Criteria

- For random fixture weights, `max_abs_diff` between upstream dequantization and Llaminar dequantization is explainable by FP16 scale/min storage only.
- For a small linear, `A @ W_paro_dequant` and `A @ W_llaminar_dequant` match within the chosen tolerance.
- No model path uses `KernelFactory` global registries for model-owned prepared affine INT4 state.

---

### Phase 2: ParoQuant Bundle Converter

**Goal**: Produce a complete Llaminar-loadable artifact from a ParoQuant HF checkpoint.

#### Tasks

- [ ] Add `tools/paroquant_convert.py` or `python/paroquant/convert.py`.
- [ ] Parse HF `config.json`, including:
  - `architectures`
  - model dimensions
  - tokenizer references
  - `quantization_config`
  - skipped modules from `quantization_config.dynamic`, if present
- [ ] Read safetensors index and tensor shards.
- [ ] Map HF tensor names to Llaminar canonical names.
- [ ] For each quantized dense linear:
  - Read `qweight`, `qzeros`, `scales`, optional `bias`.
  - Read `theta`, `pairs`, `channel_scales`.
  - Repack affine INT4 weights into Llaminar orientation and storage.
  - Write transform sidecars.
- [ ] For unquantized/skipped tensors:
  - Copy or convert to existing Llaminar-supported precision.
  - Mark them as untransformed in the manifest.
- [ ] Special-case Qwen3.5 MoE exports:
  - Handle shared `gate_up_weight_*` transform tensors.
  - Split or map per-expert gate/up/down qweight tensors to Llaminar canonical expert names.
  - Preserve skipped GDN or gating modules that upstream marks as unquantized.
- [ ] Emit `manifest.json` with full tensor metadata and source provenance.
- [ ] Add a `--validate` mode that runs a few converted linears against the Python oracle.

#### Deliverables

- Converter command:

```bash
python tools/paroquant_convert.py \
  --input /path/to/hf-paroquant-checkpoint \
  --output /path/to/model.paroq \
  --model qwen35 \
  --validate
```

- Documentation in `docs/v2/` or `tools/README.md` explaining usage.

#### Acceptance Criteria

- Converter can process at least one public ParoQuant checkpoint into a deterministic output bundle.
- Re-running conversion produces byte-identical manifest and weight sidecars, except for intentionally variable provenance fields.
- Validation reports per-linear max error, cosine, and shape mapping.

---

### Phase 3: C++ Bundle Loader and Metadata Plumbing

**Goal**: Load the converted bundle into Llaminar's model/weight infrastructure.

#### Tasks

- [ ] Add C++ metadata types:
  - `src/v2/loaders/ParoQuantMetadata.h`
  - `ParoQuantTransformSpec`
  - `ParoQuantTensorSpec`
  - `ParoQuantBundleManifest`
- [ ] Add a bundle loader:
  - `src/v2/loaders/ParoQuantBundleLoader.h`
  - `src/v2/loaders/ParoQuantBundleLoader.cpp`
- [ ] Extend model path detection so `-m /path/to/model.paroq` selects the bundle loader.
- [ ] Materialize affine INT4 tensors and normal FP tensors into `FrozenModelWeightSet`.
- [ ] Attach transform metadata to the relevant weight requirements or prepared refs.
- [ ] Extend `WeightBinding`/`PreparedWeightRef` metadata as needed so stages can query whether a weight has a Paro transform.
- [ ] Keep prepared-handle ownership in `PreparedWeightStore`.
- [ ] Ensure tied LM heads and Qwen3.5-specific name mappings remain compatible with existing weight-plan behavior.

#### Deliverables

- `llaminar2 --dry-run -m model.paroq` can parse the bundle, validate tensor metadata, and print a useful error if unsupported modules are present.
- Single-device graph construction can see ParoQuant transform metadata for quantized linears.

#### Acceptance Criteria

- Unit tests load a tiny fixture bundle and verify tensor count, shapes, transform attachment, and skipped-module handling.
- Existing GGUF loading tests still pass unchanged.
- The loader rejects malformed manifests, missing sidecars, and shape-inconsistent transforms with clear errors.

---

### Phase 4: CPU Reference Rotation and GEMM Path

**Goal**: Implement a slow but correct ParoQuant execution path for tests and parity debugging.

#### Tasks

- [ ] Add CPU rotation implementation:
  - `src/v2/kernels/cpu/paroquant/ParoRotation.h`
  - `src/v2/kernels/cpu/paroquant/ParoRotation.cpp`
- [ ] Implement function:

```cpp
bool paroRotateCPU(
    const float* input,
    float* output,
    int rows,
    int cols,
    const int16_t* pairs,
    const uint16_t* theta_fp16,
    const uint16_t* channel_scales_fp16,
    int krot,
    int group_size);
```

- [ ] Match upstream semantics exactly:
  - Apply channel scaling in the same order as ParoQuant.
  - Apply pairwise rotations in the same round and pair order.
  - Use the same sign convention for `sin(theta)`.
- [ ] Add a `ParoQuantTransform` runtime class with CPU-accessible buffers.
- [ ] In `GEMMStage`, add a correctness-first branch:
  - If prepared weight has transform metadata and backend is CPU, rotate input into workspace.
  - Call the existing GEMM kernel with rotated input.
- [ ] Add a runtime guard so unsupported fused stages fail closed or de-fuse before execution.

#### Deliverables

- CPU unit tests for rotation and transformed linear.
- Ability to run a tiny ParoQuant fixture through `GEMMStage` with expected outputs.

#### Acceptance Criteria

- CPU rotation matches upstream fixture output within FP32 tolerance.
- CPU transformed linear matches upstream ParoQuant output for `M=1` and `M>1`.
- No hot-path allocation is introduced in normal graph execution; workspace requirements are explicit or cached.

---

### Phase 5: CUDA Rotation Kernel

**Goal**: Add a CUDA runtime transform that can feed existing CUDA quantized GEMM.

#### Tasks

- [ ] Port or reimplement the upstream CUDA rotation kernel under:
  - `src/v2/kernels/cuda/paroquant/ParoRotationCUDA.h`
  - `src/v2/kernels/cuda/paroquant/ParoRotationCUDA.cu`
- [ ] Use Llaminar device selection and stream conventions.
- [ ] Support at minimum:
  - `group_size=128`
  - `krot=8`
  - FP32 input/output for reference
  - FP16 or BF16 input/output if needed by current activation precision paths
- [ ] Upload and own transform sidecars through `PreparedWeightStore` or a store-owned helper.
- [ ] Add CUDA wrapper:

```cpp
bool paroRotateCUDA(
    const void* d_input,
    void* d_output,
    int rows,
    int cols,
    const int16_t* d_pairs,
    const uint16_t* d_theta_fp16,
    const uint16_t* d_channel_scales_fp16,
    int krot,
    int group_size,
    DeviceId device,
    cudaStream_t stream);
```

- [ ] Add device-side tests comparing CUDA output to CPU reference.

#### Deliverables

- CUDA rotation kernel integrated into the build when `HAVE_CUDA=ON`.
- Unit tests for shape variants used by Qwen/Qwen3/Qwen3.5.

#### Acceptance Criteria

- CUDA rotation matches CPU reference within FP16/FP32 expected tolerance.
- Kernel works for `M=1` decode and `M>1` prefill.
- Running with `cuda-memcheck` or existing sanitizer-compatible tests shows no out-of-bounds access on fixture sizes.

---

### Phase 6: Single-Device CUDA Runtime Integration

**Goal**: Run a converted ParoQuant model or fixture through normal single-device graph execution.

#### Tasks

- [ ] Extend `PreparedWeightStore::prepareGemm()` to prepare Paro transform sidecars with the GEMM handle.
- [ ] Add `PreparedWeightStore` lookup API for transform metadata associated with a `PreparedWeightRef`.
- [ ] Update `GEMMStage`:
  - Detect transform metadata.
  - Request or reuse a rotated-activation workspace.
  - Launch CUDA rotation before GEMM.
  - Pass rotated activation to existing quantized GEMM.
- [ ] Update graph/workspace planning so the rotated activation buffer is sized once and reused.
- [ ] Add clear error messages for unsupported backends, precisions, or TP modes.
- [ ] Add `--dry-run` / placement explanation output showing that ParoQuant linears are transformed.

#### Deliverables

- Single-device CUDA inference path for ParoQuant bundle models.
- Correctness tests over synthetic and, if available, small public ParoQuant checkpoints.

#### Acceptance Criteria

- Existing CUDA GGUF quantized inference still works.
- Paro fixture model produces matching logits against upstream Transformers/ParoQuant for at least a short prompt.
- No full-logit or full-weight D2H transfer is introduced in decode.

---

### Phase 7: Fused Projection Handling

**Goal**: Restore Llaminar graph performance without violating ParoQuant semantics.

#### Phase 7A: Correctness by De-Fusion

- [ ] Update graph builders or graph resolver so ParoQuant-transformed Q/K/V projections can be emitted as separate GEMM stages when transforms differ.
- [ ] Update gate/up FFN path so transformed gate and up projections run separately unless they share identical transform metadata.
- [ ] Add tests verifying graph shape for ParoQuant models.

#### Phase 7B: Selective Re-Fusion

- [ ] Detect identical transform metadata between projections and allow the existing fused path.
- [ ] Add a Paro-aware fused QKV stage that applies each transform separately but batches launches where practical.
- [ ] Add a Paro-aware fused gate/up stage with either:
  - two rotations into two workspaces followed by fused GEMM, or
  - a specialized fused kernel that rotates and multiplies both projections.
- [ ] Benchmark whether re-fusion is worth the added complexity for decode and prefill separately.

#### Acceptance Criteria

- Correctness path does not rely on transform metadata being identical.
- Fused paths are used only when mathematically valid.
- Parity tests catch intentionally swapped transform sidecars.

---

### Phase 8: Tensor Parallelism, Pipeline Parallelism, and MoE

**Goal**: Define safe support boundaries before broad deployment.

#### Initial Support Matrix

| Mode | Initial support | Notes |
|---|---|---|
| Single-device CPU | Reference only | Slow, for tests |
| Single-device CUDA | Yes | First production target |
| Single-device ROCm | No | Needs HIP rotation kernel |
| LOCAL TP column-parallel | Later | Q/K/V, gate/up, LM head are easier |
| LOCAL TP row/input-parallel | Deferred | Rotation pairs may cross K shards |
| GLOBAL TP | Deferred | Needs cross-rank transform strategy |
| LOCAL PP | Likely safe | Stage-local weights and transforms move with PP stage |
| Qwen3.5 MoE | Later | Needs expert mapping and shared transform handling |

#### Tasks

- [ ] Add validator rules that reject ParoQuant with unsupported TP modes.
- [ ] For column-parallel TP:
  - Ensure each shard has full input activation `K` before rotation.
  - Replicate transform sidecars across participating devices.
  - Shard only output columns of transformed weights.
- [ ] For row/input-parallel TP:
  - Investigate whether rotations can be constrained to shard-local channel groups.
  - If not, rotate full activation before sharding or keep mode unsupported.
- [ ] For PP:
  - Ensure bundle loader assigns transforms only to the owning PP layer range.
  - Avoid duplicate sidecar upload outside the owning stage.
- [ ] For MoE:
  - Implement Qwen3.5 expert tensor mapping from ParoQuant export names.
  - Preserve shared `gate_up` and `down` rotation sidecars.
  - Add expert-local transformed GEMM tests.

#### Acceptance Criteria

- Unsupported distributed combinations fail during config validation, not mid-inference.
- LOCAL PP with single-device stages can load and run transformed weights for assigned layers.
- MoE conversion validates all expert tensor counts and shapes before writing output.

---

### Phase 9: Performance Optimization

**Goal**: Make ParoQuant competitive with AWQ-like throughput inside Llaminar.

#### Tasks

- [ ] Profile unfused CUDA rotation plus existing quantized GEMM with `nsys`.
- [ ] Measure overhead separately for:
  - decode `M=1`
  - short prefill
  - long prefill
- [ ] Fuse rotation with activation quantization in `CUDAQuantisedGemmKernel` where possible.
- [ ] Avoid materializing full `A_rot` when GEMM consumes quantized activations immediately.
- [ ] Add persistent transform sidecar upload and caching in `PreparedWeightStore`.
- [ ] Add workspace reuse through `BufferArena` or the stage workspace planning mechanism.
- [ ] Investigate graph capture compatibility once Paro runtime path is stable.

#### Performance Targets

| Scenario | Target |
|---|---|
| Decode overhead vs same affine INT4 without rotation | less than 15% initially, less than 10% optimized |
| Extra memory traffic | no full D2H transfers, no per-token allocations |
| Rotation sidecar upload | once per model load or device materialization |
| Prefill | correctness first, then fused rotation/quantization if it appears in profiles |

---

### Phase 10: ROCm Follow-Up

**Goal**: Add HIP support after CUDA semantics and tests are stable.

#### Tasks

- [ ] Port CUDA rotation kernel to HIP under `src/v2/kernels/rocm/paroquant/`.
- [ ] Use existing ROCm backend memory allocation and stream conventions.
- [ ] Add MI50/gfx906-compatible kernel configuration.
- [ ] Integrate with ROCm native-VNNI quantized GEMM path.
- [ ] Add ROCm unit tests and parity tests guarded by `HAVE_ROCM=ON`.

#### Acceptance Criteria

- HIP rotation matches CPU reference.
- ROCm transformed GEMM fixture matches CUDA/CPU within tolerance.
- Existing ROCm quantized GEMM tests remain green.

---

## 6. Testing Plan

### 6.1 Unit Tests

Suggested tests:

| Test | Purpose |
|---|---|
| `Test__ParoQuantAwqUnpack` | Validate qweight/qzeros/scales unpacking against Python oracle |
| `Test__ParoQuantAffineInt4Repack` | Validate Llaminar block layout reconstruction |
| `Test__ParoQuantManifest` | Validate manifest parsing and error cases |
| `Test__ParoQuantRotationCPU` | Validate CPU rotation semantics |
| `Test__ParoQuantRotationCUDA` | Validate CUDA rotation against CPU |
| `Test__PreparedWeightStoreParoQuant` | Validate transform sidecar ownership and lookup |
| `Test__ParoQuantGEMMStage` | Validate transformed linear stage output |

### 6.2 Integration Tests

Suggested CTest names:

| Test | Purpose |
|---|---|
| `V2_Integration_ParoQuant_BundleLoad` | Load fixture bundle and build graph |
| `V2_Integration_ParoQuant_CUDA_Linear` | Run transformed linears through CUDA GEMM |
| `V2_Integration_Parity_Qwen_ParoQuant_CUDA` | Compare short-prompt logits against upstream reference |
| `V2_Integration_ParoQuant_RejectUnsupportedTP` | Validate unsupported mode errors |

### 6.3 Parity Metrics

Use existing parity test conventions where possible:

| Metric | Target |
|---|---|
| Per-linear cosine | greater than 0.999 for synthetic fixtures |
| Logit cosine vs upstream reference | model-dependent, initially greater than 0.995 for short prompts |
| Top-1 token match | exact for deterministic short fixtures |
| Transform CPU vs CUDA max abs diff | FP16/FP32 tolerance based on input precision |

### 6.4 Build and Test Commands

Use the Integration build for unit/integration/parity tests:

```bash
cmake -B build_v2_integration -S src/v2 -G Ninja \
  -DCMAKE_BUILD_TYPE=Integration \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DHAVE_CUDA=ON \
  -DHAVE_ROCM=ON

cmake --build build_v2_integration --parallel

ctest --test-dir build_v2_integration \
  -R "ParoQuant" \
  --output-on-failure \
  --parallel
```

Do not artificially limit build or test parallelism.

---

## 7. Implementation Guardrails

### 7.1 Ownership

- Model-owned prepared weight state belongs in `PreparedWeightStore`.
- `KernelFactory` should remain a low-level creator/helper, not a global owner of model ParoQuant sidecars.
- Transform sidecars should follow the same lifetime as their prepared weight handles.

### 7.2 Coherence and Workspace

- Stage code should not call raw coherence APIs directly when running under `DeviceGraphExecutor`.
- Rotated activation buffers should be part of planned stage workspace, not allocated per token.
- GPU tests that call kernels directly should use existing GPU coherence RAII helpers.

### 7.3 Validation

- Reject unsupported shapes early.
- Reject missing transform metadata for ParoQuant weights.
- Reject transform metadata attached to non-linear weights unless explicitly supported.
- In Debug/Integration builds, keep tensor verification active.

### 7.4 Fusion Safety

- Never reuse a rotated activation between projections unless transform metadata is identical.
- Prefer de-fusion over silent wrong math.
- Add tests that intentionally swap `theta` or `pairs` between projections and verify parity fails.

### 7.5 Distributed Safety

- For initial implementation, reject row-parallel ParoQuant GEMMs.
- Do not assume rotation pairs are shard-local.
- Record all skipped/unsupported modules in `--dry-run` output.

---

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| AWQ packing semantics are misread | Silent wrong weights | Phase 0 fixtures and upstream oracle before runtime work |
| Transform sign/order mismatch | Bad parity despite correct weights | CPU rotation reference plus CUDA fixture tests |
| Fused QKV/gate-up reuses wrong activation | Severe accuracy loss | De-fuse first, re-fuse only with metadata checks |
| Row-parallel TP crosses rotation pairs | Wrong distributed output | Reject unsupported TP modes initially |
| Full safetensors loader scope grows too large | Delays runtime path | Start with external converted bundle |
| CUDA-only reference implementation | ROCm gap | CPU reference now, HIP port after CUDA validation |
| Per-token workspace allocation | Decode slowdown | Plan workspace through graph/stage infrastructure |
| Upstream ParoQuant repo changes format | Converter drift | Record source commit and support manifest versioning |

---

## 9. Open Decisions

1. **Storage type name**: reuse an existing Q4_1-like tensor type or add explicit `AFFINE_INT4`.
2. **Artifact format**: keep first-class `.paroq` directory bundles, or convert into extended GGUF with sidecars after Phase 6.
3. **Converter location**: `tools/paroquant_convert.py` vs `python/paroquant/convert.py`.
4. **Skipped modules**: preserve upstream ParoQuant skipped modules exactly, or allow Llaminar policy overrides.
5. **Activation precision**: support FP32 first only, or immediately include FP16/BF16 rotation paths.
6. **Fused QKV strategy**: de-fuse long term for simplicity, or add Paro-aware fused projection kernels.
7. **TP policy**: reject all TP at first, or support column-parallel TP once single-device CUDA works.

Recommended initial choices:

| Decision | Recommendation |
|---|---|
| Storage type | Add explicit `AFFINE_INT4` if existing Q4_1 semantics are too GGUF-specific; otherwise reuse Q4_1-compatible block storage internally |
| Artifact | `.paroq` directory bundle first |
| Converter location | `tools/paroquant_convert.py` for command-line use |
| Activation precision | FP32 CPU reference, FP32 and FP16 CUDA if current GEMM path needs FP16 |
| Fusion | De-fuse first |
| TP | Single-device first, then column-parallel TP |

---

## 10. Suggested Agent Execution Order

An implementation agent should proceed in this order:

1. Read this plan and inspect the current versions of the files listed in Section 3.
2. Re-clone or update `/tmp/paroquant`, record the commit, and generate Phase 0 fixtures.
3. Implement AWQ/Paro affine INT4 unpack and repack in Python first.
4. Add C++ tests that consume the fixtures and validate the chosen Llaminar storage.
5. Add the `.paroq` bundle manifest and loader for tiny fixture bundles.
6. Implement CPU rotation reference.
7. Add `PreparedWeightStore` transform metadata plumbing.
8. Wire `GEMMStage` CPU fixture execution.
9. Port CUDA rotation and validate against CPU.
10. Wire single-device CUDA `GEMMStage` execution.
11. Add de-fusion rules for transformed QKV and gate/up.
12. Add integration/parity tests.
13. Profile and optimize rotation plus activation quantization.
14. Expand to TP, PP, MoE, and ROCm only after single-device parity is stable.

At each phase, keep changes narrow and verify existing GGUF quantization tests still pass.

---

## 11. Definition of Done

The project is complete when:

- A public ParoQuant HF checkpoint can be converted into a Llaminar artifact.
- Llaminar can load the artifact with `llaminar2` without a Python runtime dependency.
- Single-device CUDA inference runs through normal graph execution.
- CPU and CUDA rotation kernels match fixture references.
- Converted affine INT4 weights match upstream ParoQuant dequantization within expected tolerance.
- Short-prompt logits match upstream ParoQuant/Transformers reference within established parity thresholds.
- Unsupported TP/PP/MoE/ROCm modes fail early with actionable errors.
- Existing GGUF quantized model support remains unaffected.
- Documentation explains conversion, limitations, and test commands.

---

## 12. Appendix: ParoQuant Reference Repo Files

Useful upstream files from `z-lab/paroquant`:

| File | Purpose |
|---|---|
| `paroquant/optim/qlinear.py` | Training-time quantized linear module |
| `paroquant/optim/rotation.py` | Pairwise rotation utilities |
| `paroquant/optim/quantizer.py` | Affine quantizer logic |
| `paroquant/cli/convert.py` | Export path from optimized weights to runtime checkpoint |
| `paroquant/inference/backends/transformers/modules.py` | Runtime `RotateQuantizedLinear` implementation |
| `paroquant/inference/backends/vllm/plugin.py` | AWQ-Marlin based vLLM integration |
| `paroquant/kernels/cuda/rotation.cu` | CUDA pairwise rotation kernel |
| `paroquant/kernels/cuda/rotation.cuh` | Rotation kernel declarations |

When importing ideas from the reference implementation, preserve only behavior and compatible algorithms. Do not copy large source blocks into Llaminar; reimplement in Llaminar style and keep license/provenance notes in documentation where needed.
