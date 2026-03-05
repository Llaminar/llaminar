# Adding New Model Architectures to Llaminar V2

A step-by-step guide for adding support for a new LLM architecture.
Written from the experience of adding Qwen3 on top of the existing Qwen2 infrastructure.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Checklist Summary](#checklist-summary)
- [Step 1: Analyze the Architecture Differences](#step-1-analyze-the-architecture-differences)
- [Step 2: Define the Graph Schema](#step-2-define-the-graph-schema)
- [Step 3: Add New Compute Stages (if needed)](#step-3-add-new-compute-stages-if-needed)
- [Step 4: Register the Architecture](#step-4-register-the-architecture)
- [Step 5: Wire Weights in the Inference Runner](#step-5-wire-weights-in-the-inference-runner)
- [Step 6: Wire Stages in the Imperative Graph Builder](#step-6-wire-stages-in-the-imperative-graph-builder)
- [Step 7: Handle GGUF Weight Loading](#step-7-handle-gguf-weight-loading)
- [Step 8: Update Snapshot Key Mapping](#step-8-update-snapshot-key-mapping)
- [Step 9: Python Reference Pipeline](#step-9-python-reference-pipeline)
- [Step 10: Parity Tests](#step-10-parity-tests)
- [Step 11: CMake Integration](#step-11-cmake-integration)
- [Common Pitfalls](#common-pitfalls)
- [File Reference](#file-reference)

---

## Overview

Llaminar V2 uses a **declarative graph schema** to describe model architectures. Each architecture defines:

1. **Schema** (`*Schema.h`) — Declarative description of stages, buffers, dependencies, and TP annotations
2. **Weight sharding config** — How weights are distributed across devices for tensor parallelism
3. **Registry entry** — So the runtime dispatches to the correct schema factory

The execution flow is:

```
GGUF "general.architecture"
  → SchemaFactoryRegistry selects *SchemaFactory
    → Schema defines stage graph
      → Qwen2Graph.cpp creates concrete ComputeStages
        → DeviceGraphExecutor runs the graph
```

**Key design principle**: If your new architecture shares most of its structure with an existing one (e.g., Qwen3 vs Qwen2), you can **reuse** the existing graph builder, buffer specs, and weight structures. Only define a new schema and any new stages.

---

## Prerequisites

Before starting, you need:

- A **GGUF model file** for the target architecture (quantized with llama.cpp)
- The **HuggingFace model** (for PyTorch reference snapshots) — either as safetensors or loadable from the GGUF
- The **architecture paper or HuggingFace implementation** to understand differences from existing models
- A working build environment (see the main `copilot-instructions.md`)

---

## Checklist Summary

Every file that needs changes for a new architecture. Items marked with ★ are only needed if the architecture introduces a new operation (e.g., QKNorm in Qwen3).

### C++ Side

| # | File | Change |
|---|------|--------|
| 1 | `src/v2/models/<arch>/<Arch>Schema.h` | **CREATE**: Schema factory with stage graph + sharding config |
| 2 | `src/v2/execution/local_execution/graph/SchemaFactoryRegistry.cpp` | **EDIT**: Add `if` branch + include |
| 3 | `src/v2/execution/local_execution/graph/SchemaFactoryRegistry.h` | No change usually (generic interface) |
| 4★ | `src/v2/execution/compute_stages/IComputeStage.h` | **EDIT**: Add to `ComputeStageType` enum |
| 5★ | `src/v2/execution/local_execution/graph/GraphSchema.h` | **EDIT**: Add to `StageType` enum |
| 6★ | `src/v2/execution/compute_stages/stages/<NewStage>.h/cpp` | **CREATE**: New compute stage |
| 7★ | `src/v2/execution/compute_stages/ComputeStageFactory.h/cpp` | **EDIT**: Add factory method |
| 8 | `src/v2/execution/factory/InferenceRunnerFactory.cpp` | **EDIT**: Wire new optional weights |
| 9 | `src/v2/models/qwen/Qwen2Graph.h` | **EDIT**: Add fields to `Qwen2LayerWeights` |
| 10 | `src/v2/models/qwen/Qwen2Graph.cpp` | **EDIT**: Create new stages in graph |
| 11 | `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h` | **EDIT**: Add snapshot key mappings |
| 12 | `tests/v2/integration/parity/ParityTestBase.h` | **EDIT**: Add new stages to `per_layer_stages` |

### Python Side

| # | File | Change |
|---|------|--------|
| 13 | `python/reference/loaders/tensor_name_mapper.py` | **EDIT**: Add GGUF→HuggingFace weight name mappings |
| 14 | `python/reference/generate_qwen2_pipeline_snapshots.py` | **EDIT**: Add new stage capture logic |

### Tests + Build

| # | File | Change |
|---|------|--------|
| 15 | `tests/v2/integration/parity/<arch>/<Arch>ParityTestBase.h` | **CREATE**: Test base class |
| 16 | `tests/v2/integration/parity/<arch>/Test__<Arch>_SingleDevice_Parity.cpp` | **CREATE**: Parity test |
| 17 | `tests/v2/CMakeLists.txt` | **EDIT**: Add test targets |

---

## Step 1: Analyze the Architecture Differences

Compare your new architecture against the closest existing one. Document:

- **New operations**: e.g., Qwen3 adds per-head QK RMSNorm
- **Removed operations**: e.g., Qwen3 removes QKV biases
- **Changed parameters**: e.g., different head_dim, rope_theta, etc.
- **Weight differences**: New weight tensors, different shapes, optional tensors

**Example (Qwen3 vs Qwen2)**:

| Aspect | Qwen2 | Qwen3 |
|--------|-------|-------|
| QKV biases | Yes | No |
| QK normalization | None | Per-head RMSNorm |
| New weights | — | `attn_q_norm.weight`, `attn_k_norm.weight` |
| head_dim | `d_model / n_heads` | Explicit from GGUF `attention.key_length` |

---

## Step 2: Define the Graph Schema

Create `src/v2/models/<arch>/<Arch>Schema.h`. The schema factory implements `ISchemaFactory` and defines:

1. `architectureName()` — returns the GGUF architecture string
2. `getWeightShardingConfig()` — how weights are split for TP
3. `createSchema()` — the full compute graph as a `GraphSchema`
4. `isWeightOptional()` — which GGUF weight names are optional

**Key structures**:

```cpp
// Each stage in the graph
StageSpec{
    .name = "q_norm",                           // Stage name (used in graph deps)
    .type = StageType::QKNorm,                  // Stage type enum
    .inputs = {{"Q", BufferSemantic::InOut},     // Input buffers
               {"weights.q_norm", BufferSemantic::Input}},
    .outputs = {{"Q", BufferSemantic::InOut}},   // Output buffers
    .dependencies = {"qkv_proj"},                // Must run after these stages
    .tp_mode = TPMode::None,                     // TP annotation
    .is_optional = true,
    .exec_policy_key = "exec_rmsnorm"
};
```

**If your architecture is very similar to an existing one**, reuse its buffer names and buffer specs. For example, Qwen3Schema.h imports `Qwen2Schema.h` to reuse `BufferNames::*`:

```cpp
#include "../qwen/Qwen2Schema.h" // Reuse Qwen2 buffer names
```

**Weight sharding**: Each weight pattern specifies how it's distributed across TP ranks:

| Mode | Description | Example |
|------|-------------|---------|
| `ColumnParallel` | Split output dimension | Q/K/V projections, Gate/Up |
| `InputParallel` | Split input dimension, allreduce after | Wo, Down |
| `Replicate` | Full copy on each device | Norms, embeddings, small weights |

---

## Step 3: Add New Compute Stages (if needed)

If the architecture introduces a new operation (not just a different wiring of existing stages), you need a new `ComputeStage`.

### 3a. Add to enums

Two separate enums must be updated:

**`src/v2/execution/compute_stages/IComputeStage.h`** — runtime enum:
```cpp
enum class ComputeStageType {
    // ... existing entries ...
    QK_NORM,  // Per-head normalization (Qwen3)
};
```

**`src/v2/execution/local_execution/graph/GraphSchema.h`** — schema enum:
```cpp
enum class StageType {
    // ... existing entries ...
    QKNorm,  // Per-head normalization (Qwen3)
};
```

### 3b. Create the stage

Create `src/v2/execution/compute_stages/stages/<NewStage>.h` and `.cpp`.

The stage must implement `IComputeStage` and provide:
- `Params` struct with all configuration
- `execute()` — runs the computation
- `stageType()` — returns the `ComputeStageType`
- `getDumpInfo()` / `buildDumpInfoImpl()` — for stage dump framework

**Pattern**: See `QKNormStage.h/cpp` for a minimal example of a new stage that wraps an existing kernel (RMSNorm) with a different dimensional remapping.

### 3c. Register in ComputeStageFactory

**`src/v2/execution/compute_stages/ComputeStageFactory.h`** — add declaration:
```cpp
static std::unique_ptr<IComputeStage> createQKNorm(const QKNormStage::Params& params);
```

**`src/v2/execution/compute_stages/ComputeStageFactory.cpp`** — add implementation:
```cpp
std::unique_ptr<IComputeStage> ComputeStageFactory::createQKNorm(
    const QKNormStage::Params& params)
{
    return std::make_unique<QKNormStage>(params);
}
```

---

## Step 4: Register the Architecture

**`src/v2/execution/local_execution/graph/SchemaFactoryRegistry.cpp`**:

```cpp
#include "../../../models/<arch>/<Arch>Schema.h"

// In getFactory():
if (arch_lower == "myarch") {
    return std::make_unique<MyArchSchemaFactory>();
}

// In isSupported():
return arch_lower == "qwen2" || arch_lower == "qwen3" || arch_lower == "myarch";

// In supportedArchitectures():
return {"qwen2", "qwen3", "myarch"};
```

The architecture string must match the GGUF `general.architecture` metadata value exactly (case-insensitive).

---

## Step 5: Wire Weights in the Inference Runner

**`src/v2/execution/factory/InferenceRunnerFactory.cpp`** — in the layer weight loading section:

```cpp
// Load optional architecture-specific weights
auto q_norm = weight_mgr->getWeightForDevice(prefix + "attn_q_norm.weight");
auto k_norm = weight_mgr->getWeightForDevice(prefix + "attn_k_norm.weight");
layer.q_norm = q_norm ? q_norm.get() : nullptr;
layer.k_norm = k_norm ? k_norm.get() : nullptr;
```

**Pattern**: Optional weights use `getWeightForDevice()` which returns `nullptr` if the weight doesn't exist. Store as raw pointer in the layer struct (the weight manager owns the memory).

**Also update** `src/v2/models/qwen/Qwen2Graph.h` — add the new fields to `Qwen2LayerWeights`:

```cpp
struct Qwen2LayerWeights {
    // ... existing fields ...
    TensorBase* q_norm = nullptr;  ///< Q norm gamma [head_dim]
    TensorBase* k_norm = nullptr;  ///< K norm gamma [head_dim]
};
```

There are **four places** in InferenceRunnerFactory.cpp that load layer weights (single-device, TP, PP, hybrid). All four must be updated. Search for the existing patterns (e.g., `q_bias`) to find all locations.

---

## Step 6: Wire Stages in the Imperative Graph Builder

**`src/v2/models/qwen/Qwen2Graph.cpp`** — add conditional stage creation:

```cpp
// Stage 2.5: Per-head QK RMSNorm (Qwen3 only)
if (layer.q_norm && layer.k_norm)
{
    QKNormStage::Params q_norm_params;
    q_norm_params.input = buffers.Q;
    q_norm_params.output = buffers.Q;  // In-place
    q_norm_params.gamma = layer.q_norm;
    q_norm_params.n_heads = local_n_heads;
    q_norm_params.head_dim = config_.head_dim;
    q_norm_params.eps = config_.rms_norm_eps;
    q_norm_params.seq_len = total_tokens;
    q_norm_params.device_id = device;

    graph.addNode(prefix + "q_norm",
                  ComputeStageFactory::createQKNorm(q_norm_params), device);
    graph.addDependency(prefix + "q_norm", prefix + "qkv_proj");

    // Similar for k_norm...
}
```

**Key principle**: Gate new stages on the presence of their weights (`if (layer.q_norm != nullptr)`). This way the same graph builder handles both Qwen2 (where q_norm is null) and Qwen3 (where q_norm has trained values).

**Also update dependency chains**: If you insert new stages between existing ones, update the downstream stage's dependency. For example, RoPE must depend on the QK norm stages instead of directly on QKV projection:

```cpp
// Before: rope depends on qkv_proj
// After:  rope depends on q_norm and k_norm (which depend on qkv_proj)
if (layer.q_norm && layer.k_norm) {
    graph.addDependency(prefix + "rope", prefix + "q_norm");
    graph.addDependency(prefix + "rope", prefix + "k_norm");
} else {
    graph.addDependency(prefix + "rope", prefix + "qkv_proj");
}
```

---

## Step 7: Handle GGUF Weight Loading

GGUF weight loading is largely automatic — `ModelLoader.cpp` reads all tensors from the file using architecture-prefixed metadata keys. The architecture string from `general.architecture` is used to build keys like `qwen3.embedding_length`.

### What's already handled automatically

- Architecture detection: `general.architecture` → `model_.architecture`
- All standard hyperparameters: context_length, embedding_length, block_count, head_count, etc.
- Optional parameters: `attention.key_length`, `attention.value_length` (for explicit head_dim)
- RoPE theta: `rope.freq_base`
- RMSNorm epsilon: `attention.layer_norm_rms_epsilon`

### What you might need to change

If the GGUF metadata has new keys specific to your architecture, add them to the parsing in `ModelLoader.cpp`. The pattern is:

```cpp
std::string arch_prefix = model_.architecture + ".";
model_.some_new_param = get_uint(arch_prefix + "some_new_key");
```

### Tied embeddings

If your architecture uses tied embeddings (`output.weight` not present in GGUF), the runner factory handles this automatically with a fallback to `token_embd.weight`.

---

## Step 8: Update Snapshot Key Mapping

For parity testing, C++ stage names must map to Python snapshot file names.

**`src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h`** — add entries to `convertStageNameToSnapshotKey()`:

```cpp
static const std::unordered_map<std::string, std::string> suffix_map = {
    // ... existing entries ...
    {"_q_norm", "_Q_NORM"},   // New stage
    {"_k_norm", "_K_NORM"},   // New stage
};
```

The mapping works by suffix matching: stage name `layer0_q_norm` → find suffix `_q_norm` → replace with `_Q_NORM` → produce `layer0_Q_NORM`.

**Also update** `tests/v2/integration/parity/ParityTestBase.h` — add new stages to the `per_layer_stages` comparison list:

```cpp
std::vector<std::string> per_layer_stages = {
    "ATTENTION_NORM", "Q_PROJECTION", "K_PROJECTION", "V_PROJECTION",
    "Q_NORM", "K_NORM",  // New stages (skipped if snapshot doesn't exist)
    "Q_ROPE", "K_ROPE",
    // ...
};
```

Stages that don't have a matching snapshot file are silently skipped, so it's safe to add them even when testing architectures that don't have those stages.

---

## Step 9: Python Reference Pipeline

### 9a. GGUF → HuggingFace weight name mapping

**`python/reference/loaders/tensor_name_mapper.py`** — add mappings for new weight tensors:

```python
QWEN2_TENSOR_MAP = {
    # ... existing entries ...
    # QK RMSNorm (Qwen3: per-head normalization before RoPE)
    'blk.{}.attn_q_norm.weight': 'model.layers.{}.self_attn.q_norm.weight',
    'blk.{}.attn_k_norm.weight': 'model.layers.{}.self_attn.k_norm.weight',
}
```

> **CRITICAL**: If a weight mapping is missing, `load_state_dict(strict=False)` will silently leave the weight at its default value (often `torch.ones()`). The pipeline runs without errors but produces wrong reference snapshots. This was the root cause of the Qwen3 parity bug — QK norm gamma weights defaulted to 1.0 instead of the trained GGUF values (like 4.53, -0.73, etc.), causing cosine similarity to drop from 0.999 to 0.334.

### 9b. Snapshot capture logic

**`python/reference/generate_qwen2_pipeline_snapshots.py`** — add capture code for new stages:

```python
# 3.5. QK RMSNorm (Qwen3 only: per-head normalization before RoPE)
if hasattr(layer.self_attn, 'q_norm') and layer.self_attn.q_norm is not None:
    q = layer.self_attn.q_norm(q)
    k = layer.self_attn.k_norm(k)
    if capture:
        q_norm_flat = q.transpose(1, 2).reshape(bsz, seq_len, n_heads * d_head)
        k_norm_flat = k.transpose(1, 2).reshape(bsz, seq_len, n_kv_heads * d_head)
        self._save_snapshot('Q_NORM', q_norm_flat, layer_idx)
        self._save_snapshot('K_NORM', k_norm_flat, layer_idx)
```

**Pattern**: Use `hasattr()` checks so the same pipeline handles both Qwen2 (no q_norm) and Qwen3. Save snapshots in the flattened `[batch, seq_len, dim]` layout that the C++ pipeline expects.

### 9c. Model type detection

If your architecture uses a different GGUF `general.architecture` string, update `detect_model_type_from_metadata()` in `tensor_name_mapper.py`:

```python
if 'qwen' in arch:
    return 'qwen2'  # Both qwen2 and qwen3 use the same tensor map
```

If your new architecture needs a completely different tensor name mapping table, create a new `*_TENSOR_MAP` dict and add a branch in `TensorNameMapper.__init__()`.

### 9d. Generate snapshots

```bash
python3 python/reference/generate_qwen2_pipeline_snapshots.py \
    --model models/MyArch-Q8_0.gguf \
    --output pytorch_myarch_snapshots \
    --decode-steps 3 -v
```

---

## Step 10: Parity Tests

### 10a. Test base class

Create `tests/v2/integration/parity/<arch>/<Arch>ParityTestBase.h`:

```cpp
#pragma once
#include "../qwen2/Qwen2ParityTestBase.h"
#include "models/<arch>/<Arch>Schema.h"

namespace llaminar2::test::parity::<arch> {
    using namespace llaminar2::test::parity::qwen2;

    template <typename Derived>
    class MyArchConfigDrivenParityTest : public ConfigDrivenParityTest<Derived>
    {
    protected:
        using Base = ConfigDrivenParityTest<Derived>;
        void configureModel(std::shared_ptr<ModelContext> model_ctx) override
        {
            if (Base::cfg().is_local_tp() || Base::cfg().is_global_tp()) {
                MyArchSchemaFactory schema_factory;
                model_ctx->weightManager()->setWeightShardingConfig(
                    schema_factory.getWeightShardingConfig());
            }
        }
    };
}
```

### 10b. Single-device parity test

Create `tests/v2/integration/parity/<arch>/Test__<Arch>_SingleDevice_Parity.cpp` with test configs for CPU, CUDA, and ROCm backends:

```cpp
static const std::vector<TestConfig> kMyArchSingleDeviceConfigs = {
    {
        .name = "MyArch_CPU_KV_FP16",
        .devices = {ParityDeviceType::CPU},
        .parallelism = Parallelism::None,
        .collective = Collective::None,
        .thresholds = {
            .cosine_threshold = 0.999f,
            .decode_cosine_threshold = 0.99f,
            .early_layers_count = 4,
            .min_early_layers_passed = 3,
            .kl_threshold = 0.15f,
        },
        .model_path = "models/MyArch-Q8_0.gguf",
        .snapshot_dir = "pytorch_myarch_snapshots",
        .activation_precision = ActivationPrecision::FP32,
        .kv_cache_precision = KVCachePrecision::FP16,
    },
    // Add CUDA and ROCm configs...
};
```

---

## Step 11: CMake Integration

**`tests/v2/CMakeLists.txt`** — add the test target:

```cmake
# ---- MyArch Single-Device Parity ----
add_executable(v2_integration_parity_myarch_single_device
    integration/parity/<arch>/Test__MyArch_SingleDevice_Parity.cpp
    ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
)
target_link_libraries(v2_integration_parity_myarch_single_device
    llaminar2_core v2_test_utils GTest::gtest ZLIB::ZLIB
)
target_include_directories(v2_integration_parity_myarch_single_device PRIVATE
    ${WORKSPACE_ROOT_PARITY}/external/cnpy
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_v2_test(V2_Integration_Parity_MyArch_SingleDevice
    COMMAND $<TARGET_FILE:v2_integration_parity_myarch_single_device>
    LABELS "V2;Integration;Parity;MyArch;CPU;CUDA;ROCm;SingleDevice;PyTorch;GoldenReference;FullModel"
    MPI_PROCS 1
)
```

---

## Common Pitfalls

### 1. Missing Python weight mapping (★ most common)

If the GGUF tensor name mapper doesn't include a mapping for a new weight, `load_state_dict(strict=False)` silently leaves it at its default initialization (typically `ones()` or `zeros()`). The pipeline runs, but reference snapshots are wrong.

**Symptom**: High cosine similarity for stages before the missing weight, sharp drop at the stage using it.

**Fix**: Add the GGUF → HuggingFace mapping in `QWEN2_TENSOR_MAP`:
```python
'blk.{}.attn_q_norm.weight': 'model.layers.{}.self_attn.q_norm.weight',
```

### 2. Missing weight wiring in InferenceRunnerFactory.cpp

There are **four separate weight-loading sections** in InferenceRunnerFactory.cpp (single-device, local TP, global TP, PP). All four must have the new weight loading code.

**Symptom**: Works on single-device but crashes or produces garbage on TP/PP.

**Fix**: Search for an existing optional weight (e.g., `q_bias`) and ensure your new weight is loaded in every location where `q_bias` appears.

### 3. Snapshot key mismatch

C++ stage names (e.g., `layer0_q_norm`) must map to Python snapshot file names (e.g., `layer0_Q_NORM.npy`). If the mapping in `convertStageNameToSnapshotKey()` is missing, the parity test silently skips the comparison.

**Symptom**: Parity test passes but new stages aren't actually being compared.

**Fix**: Add to the `suffix_map` in `DeviceGraphOrchestrator.h` and add the stage name to `per_layer_stages` in `ParityTestBase.h`.

### 4. Dependency chain not updated

When inserting a new stage between existing ones, the downstream stage's dependency must be updated to point to the new stage.

**Symptom**: Stages execute in wrong order, garbage output, or stale buffer data.

**Fix**: Update `graph.addDependency()` calls — the downstream stage depends on the new stage, and the new stage depends on the upstream stage.

### 5. head_dim from GGUF vs computed

Some architectures (like Qwen3) specify `head_dim` explicitly via `attention.key_length` in GGUF metadata, which may differ from `d_model / n_heads`. Always prefer the GGUF value when available.

---

## File Reference

### Schema and Registration

| File | Purpose |
|------|---------|
| `src/v2/models/<arch>/<Arch>Schema.h` | Declarative graph schema + TP sharding config |
| `src/v2/models/IGraphConfigBuilder.h` | Interface for config builders |
| `src/v2/execution/local_execution/graph/GraphSchema.h` | `StageType` enum, `StageSpec` struct |
| `src/v2/execution/local_execution/graph/SchemaFactoryRegistry.cpp` | Architecture → factory dispatch |

### Compute Stages

| File | Purpose |
|------|---------|
| `src/v2/execution/compute_stages/IComputeStage.h` | `ComputeStageType` enum, stage interface |
| `src/v2/execution/compute_stages/stages/` | All stage implementations |
| `src/v2/execution/compute_stages/ComputeStageFactory.h/cpp` | Stage creation factory |

### Graph Building and Weight Wiring

| File | Purpose |
|------|---------|
| `src/v2/models/qwen/Qwen2Graph.h` | `Qwen2LayerWeights` struct, graph config |
| `src/v2/models/qwen/Qwen2Graph.cpp` | Imperative graph construction |
| `src/v2/execution/factory/InferenceRunnerFactory.cpp` | Weight loading → layer struct wiring |

### Snapshot and Parity Infrastructure

| File | Purpose |
|------|---------|
| `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h` | Stage name → snapshot key mapping |
| `tests/v2/integration/parity/ParityTestBase.h` | Parity comparison framework |
| `tests/v2/integration/parity/<arch>/` | Architecture-specific test files |

### Python Reference

| File | Purpose |
|------|---------|
| `python/reference/loaders/tensor_name_mapper.py` | GGUF → HuggingFace weight name mapping |
| `python/reference/loaders/gguf_loader.py` | GGUF file loading orchestrator |
| `python/reference/generate_qwen2_pipeline_snapshots.py` | PyTorch reference snapshot generator |

### GGUF Loading

| File | Purpose |
|------|---------|
| `src/v2/loaders/ModelLoader.cpp` | GGUF metadata parsing, architecture detection |

---

## Example: Qwen3 Addition (Condensed)

Qwen3 added per-head QK RMSNorm before RoPE. Here's what changed:

1. **New schema**: `src/v2/models/qwen3/Qwen3Schema.h` — reuses Qwen2 buffer names, adds `q_norm`/`k_norm` StageSpecs between QKV and RoPE, marks QKV biases as optional (Qwen3 has none)
2. **New stage**: `QKNormStage.h/cpp` — thin wrapper over RMSNorm that remaps `[seq_len, n_heads*head_dim]` to `[seq_len*n_heads, head_dim]`
3. **New enums**: `QK_NORM` in `ComputeStageType`, `QKNorm` in `StageType`
4. **Registry**: Added `"qwen3"` → `Qwen3SchemaFactory` in SchemaFactoryRegistry.cpp
5. **Weight wiring**: Added `q_norm`/`k_norm` loading in 4 places in InferenceRunnerFactory.cpp
6. **Graph building**: Conditional `if (layer.q_norm && layer.k_norm)` block in Qwen2Graph.cpp
7. **Snapshot mapping**: Added `_q_norm` → `_Q_NORM`, `_k_norm` → `_K_NORM` to suffix_map
8. **Python mapping**: Added `attn_q_norm.weight` / `attn_k_norm.weight` to `QWEN2_TENSOR_MAP`
9. **Python capture**: Added `hasattr(layer.self_attn, 'q_norm')` guard + snapshot save
10. **Parity test**: `Qwen3ParityTestBase.h` + `Test__Qwen3_SingleDevice_Parity.cpp`
11. **CMake**: Added test target in `tests/v2/CMakeLists.txt`

Total: ~500 lines of new C++ code, ~20 lines of Python changes, ~50 lines of CMake.
