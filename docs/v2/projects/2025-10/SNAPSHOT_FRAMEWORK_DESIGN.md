# V2 Snapshot Framework Design

**Author**: David Sanftenberg  
**Date**: October 30, 2025  
**Status**: Design Document

## Executive Summary

This document specifies a parity testing snapshot framework for Llaminar V2 that enables precise validation of pipeline correctness by comparing intermediate activation tensors against reference implementations (PyTorch, llama.cpp).

**Key Requirements**:
- Zero overhead in release builds (compile-time disabled via macros)
- TensorBase integration (all tensor types snapshotable)
- Device-aware (capture from GPU tensors)
- MPI-aware (rank 0 only or configurable)
- Batch-aware (capture per sequence in batch)
- Architecture-agnostic stage naming (Qwen, LLaMA, etc.)

---

## Background: V1 Parity Framework

### V1 Architecture
V1 had a comprehensive parity test framework with:
- **PipelineStage enum**: EMBEDDING, ATTENTION_NORM, Q_PROJECTION, ROPE_APPLICATION, etc.
- **SnapshotRegistry**: Global singleton for test-time snapshot storage
- **PrefillProvider integration**: Built-in `captureSnapshot()` in provider base class
- **PyTorch reference**: `generate_test_snapshots.py` for ground truth generation
- **Dynamic variance thresholds**: Automatic tolerance computation from PyTorch variance
- **Weight verification**: Comprehensive embedding + layer weight validation

### V1 Limitations for V2
1. **Operator-centric**: Snapshots captured in operator layer (V2 has no operators)
2. **Global registry**: Thread-unsafe singleton pattern
3. **Fixed device model**: No heterogeneous device handling
4. **Batch-agnostic**: No per-sequence capture in batches
5. **Compile-time overhead**: Always compiled in (no `#ifdef NDEBUG` guards)

---

## V2 Design Goals

### Primary Goals
1. **Zero release overhead**: Macros compile to no-ops in `CMAKE_BUILD_TYPE=Release`
2. **TensorBase integration**: All tensors (FP32, BF16, IQ4_NL, etc.) can snapshot
3. **Per-pipeline storage**: Thread-safe, no global state
4. **Device-transparent**: Automatic CPU←→GPU sync before capture
5. **MPI-aware**: Configurable rank filtering (default: rank 0 only)
6. **Batch-aware**: Capture individual sequences in batched forward passes

### Secondary Goals
1. **Hierarchical stage naming**: GLOBAL/LAYER/ATTENTION/FFN categories
2. **Python integration**: Reuse V1's PyTorch reference scripts with V2 adapter
3. **Incremental decode support**: Per-token snapshot capture
4. **Weight verification**: Validate model weights before activation testing

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                       Pipeline Code                         │
│  LLAMINAR_SNAPSHOT(ATTENTION_NORM, layer, tensor)          │
│  LLAMINAR_SNAPSHOT_BATCH(Q_PROJECTION, layer, batch, seq)  │
└────────────────────┬────────────────────────────────────────┘
                     │ (compiles to no-op in release)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    SnapshotCapture.h                        │
│  - LLAMINAR_SNAPSHOT* macros                                │
│  - PipelineStage enum                                       │
│  - Stage name formatting helpers                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    TensorBase::snapshot()                   │
│  - Device→CPU sync (if needed)                              │
│  - Data copy to SnapshotData struct                         │
│  - MPI rank filtering                                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              PipelineSnapshotManager                        │
│  - Per-pipeline snapshot storage (thread-safe)              │
│  - Save/load snapshots (NPY format)                         │
│  - Query interface for tests                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              SnapshotComparator                             │
│  - Load PyTorch reference snapshots                         │
│  - Compute comparison metrics (L2, max_abs, etc.)           │
│  - Apply dynamic variance thresholds                        │
│  - Report pass/fail with detailed diagnostics               │
└─────────────────────────────────────────────────────────────┘
```

### File Structure

```
src/v2/utils/
├── SnapshotCapture.h          # Macros, stage enum, helpers
├── PipelineSnapshotManager.h  # Per-pipeline storage
├── PipelineSnapshotManager.cpp
├── SnapshotComparator.h       # Test-time comparison
└── SnapshotComparator.cpp

src/v2/tensors/
└── Tensors.h                  # Add snapshot() method to TensorBase

tests/v2/parity/
├── Test__Qwen2_Parity.cpp     # Qwen2 parity test
├── Test__LLaMA_Parity.cpp     # LLaMA parity test (future)
└── ParityTestUtils.h          # Shared test utilities

python/reference/
├── v2_snapshot_adapter.py     # V2→V1 adapter for PyTorch scripts
└── generate_v2_snapshots.py   # V2-specific snapshot generation
```

---

## Detailed Design

### 1. Pipeline Stages (Hierarchical Enum)

**File**: `src/v2/utils/SnapshotCapture.h`

```cpp
namespace llaminar2 {
namespace snapshot {

/**
 * @brief Hierarchical pipeline stages for snapshot capture
 * 
 * Stages are organized by category for easy filtering:
 * - GLOBAL_*: Applies to entire model (embedding, final norm, LM head)
 * - LAYER_*: Per-layer stages (attention, FFN)
 * - ATTN_*: Attention-specific substages
 * - FFN_*: FFN-specific substages
 */
enum class PipelineStage {
    // ===== Global Stages (layer = -1) =====
    GLOBAL_EMBEDDING,        // Token embedding table lookup
    GLOBAL_FINAL_NORM,       // Final RMSNorm before LM head
    GLOBAL_LM_HEAD,          // Language model head output (logits)
    
    // ===== Per-Layer Stages =====
    LAYER_INPUT,             // Layer input (from prev layer or embedding)
    
    // Attention block
    LAYER_ATTN_NORM,         // Pre-attention RMSNorm
    LAYER_ATTN_OUTPUT,       // After attention output projection
    LAYER_ATTN_RESIDUAL,     // After attention residual add
    
    // FFN block
    LAYER_FFN_NORM,          // Pre-FFN RMSNorm
    LAYER_FFN_OUTPUT,        // After FFN down projection
    LAYER_FFN_RESIDUAL,      // After FFN residual add (final layer output)
    
    // ===== Attention Substages (for debugging) =====
    ATTN_Q_PROJECTION,       // Query projection output
    ATTN_K_PROJECTION,       // Key projection output
    ATTN_V_PROJECTION,       // Value projection output
    ATTN_ROPE_APPLIED,       // After RoPE (concatenated [Q | K])
    ATTN_SCORES,             // Attention scores (Q @ K^T)
    ATTN_WEIGHTS,            // After softmax
    ATTN_CONTEXT,            // Attention output (weights @ V)
    
    // ===== FFN Substages (for debugging) =====
    FFN_GATE,                // Gate projection output
    FFN_UP,                  // Up projection output
    FFN_SWIGLU,              // SwiGLU activation output (silu(gate) * up)
    FFN_DOWN,                // Down projection output (same as LAYER_FFN_OUTPUT)
};

/**
 * @brief Format stage name for file storage
 * 
 * Examples:
 * - formatStageName(GLOBAL_EMBEDDING, -1, 0, 0) → "token_0/EMBEDDING.npy"
 * - formatStageName(LAYER_ATTN_NORM, 5, 2, 1) → "token_2/ATTENTION_NORM_layer5_seq1.npy"
 * - formatStageName(ATTN_Q_PROJECTION, 0, 0, -1) → "token_0/Q_PROJECTION_layer0.npy"
 */
std::string formatStageName(
    PipelineStage stage, 
    int layer_idx,       // -1 for global stages
    int token_idx,       // Token position in sequence (incremental decode)
    int seq_idx = -1     // Sequence index in batch (-1 for single sequence)
);

/**
 * @brief Check if stage is enabled for capture
 * 
 * Controlled by LLAMINAR_SNAPSHOT_STAGES environment variable:
 * - "all" → capture all stages
 * - "attention" → capture only ATTN_* stages
 * - "ffn" → capture only FFN_* stages
 * - "layer" → capture only LAYER_* stages
 * - "global" → capture only GLOBAL_* stages
 * - Comma-separated: "attention,ffn,global"
 */
bool isStageEnabled(PipelineStage stage);

} // namespace snapshot
} // namespace llaminar2
```

### 2. Snapshot Data Structure

**File**: `src/v2/utils/SnapshotCapture.h`

```cpp
namespace llaminar2 {
namespace snapshot {

/**
 * @brief Captured snapshot data
 */
struct SnapshotData {
    std::vector<size_t> shape;      // Tensor shape
    std::vector<float> data;        // Copied tensor data (FP32)
    TensorType native_type;         // Original tensor type
    int device_index;               // Device where tensor resided
    int mpi_rank;                   // MPI rank that captured this
    std::string stage_name;         // Formatted stage name
    
    // Metadata
    int layer_index = -1;           // Layer index (-1 for global)
    int token_index = 0;            // Token position (incremental decode)
    int sequence_index = -1;        // Sequence in batch (-1 for single)
    
    // Save to NPY file
    bool save(const std::string& filepath) const;
    
    // Load from NPY file
    static SnapshotData load(const std::string& filepath);
};

} // namespace snapshot
} // namespace llaminar2
```

### 3. TensorBase Integration

**File**: `src/v2/tensors/Tensors.h`

```cpp
class TensorBase : public std::enable_shared_from_this<TensorBase> {
public:
    // ... existing methods ...
    
#ifndef NDEBUG
    /**
     * @brief Capture snapshot of tensor data (debug builds only)
     * 
     * Implementation:
     * 1. Sync device→CPU if needed (is_on_device())
     * 2. Copy data() to SnapshotData
     * 3. Return SnapshotData struct
     * 
     * @note This method is REMOVED in release builds (zero overhead)
     */
    virtual snapshot::SnapshotData captureSnapshot() const {
        snapshot::SnapshotData snap;
        snap.shape = this->shape();
        snap.native_type = this->native_type();
        snap.device_index = this->device_index();
        
        // Sync from device if needed (calls data() which handles sync)
        const float* host_data = this->data();
        size_t total_elements = 1;
        for (auto dim : snap.shape) total_elements *= dim;
        
        snap.data.resize(total_elements);
        std::memcpy(snap.data.data(), host_data, total_elements * sizeof(float));
        
        return snap;
    }
#endif
};
```

### 4. Snapshot Capture Macros

**File**: `src/v2/utils/SnapshotCapture.h`

```cpp
namespace llaminar2 {
namespace snapshot {

// Forward declaration
class PipelineSnapshotManager;

} // namespace snapshot
} // namespace llaminar2

// ===== Snapshot Macros =====

#ifndef NDEBUG

/**
 * @brief Capture single-sequence snapshot
 * 
 * Usage:
 *   LLAMINAR_SNAPSHOT(manager, PipelineStage::LAYER_ATTN_NORM, layer_idx, token_idx, tensor_ptr)
 */
#define LLAMINAR_SNAPSHOT(manager_ptr, stage, layer_idx, token_idx, tensor_ptr) \
    do { \
        if ((manager_ptr) && llaminar2::snapshot::isStageEnabled(stage)) { \
            auto snap_data = (tensor_ptr)->captureSnapshot(); \
            snap_data.stage_name = llaminar2::snapshot::formatStageName((stage), (layer_idx), (token_idx), -1); \
            snap_data.layer_index = (layer_idx); \
            snap_data.token_index = (token_idx); \
            snap_data.sequence_index = -1; \
            (manager_ptr)->addSnapshot(snap_data); \
        } \
    } while(0)

/**
 * @brief Capture batch-aware snapshot (per sequence)
 * 
 * Usage:
 *   LLAMINAR_SNAPSHOT_BATCH(manager, PipelineStage::ATTN_Q_PROJECTION, layer, token, seq, tensor)
 */
#define LLAMINAR_SNAPSHOT_BATCH(manager_ptr, stage, layer_idx, token_idx, seq_idx, tensor_ptr) \
    do { \
        if ((manager_ptr) && llaminar2::snapshot::isStageEnabled(stage)) { \
            auto snap_data = (tensor_ptr)->captureSnapshot(); \
            snap_data.stage_name = llaminar2::snapshot::formatStageName((stage), (layer_idx), (token_idx), (seq_idx)); \
            snap_data.layer_index = (layer_idx); \
            snap_data.token_index = (token_idx); \
            snap_data.sequence_index = (seq_idx); \
            (manager_ptr)->addSnapshot(snap_data); \
        } \
    } while(0)

/**
 * @brief MPI-aware snapshot (rank 0 only by default)
 * 
 * Usage:
 *   LLAMINAR_SNAPSHOT_MPI(manager, stage, layer, token, tensor, mpi_ctx)
 */
#define LLAMINAR_SNAPSHOT_MPI(manager_ptr, stage, layer_idx, token_idx, tensor_ptr, mpi_ctx) \
    do { \
        if ((mpi_ctx) && (mpi_ctx)->rank() == 0) { \
            LLAMINAR_SNAPSHOT((manager_ptr), (stage), (layer_idx), (token_idx), (tensor_ptr)); \
        } \
    } while(0)

#else
// Release builds: macros compile to no-ops
#define LLAMINAR_SNAPSHOT(manager_ptr, stage, layer_idx, token_idx, tensor_ptr) ((void)0)
#define LLAMINAR_SNAPSHOT_BATCH(manager_ptr, stage, layer_idx, token_idx, seq_idx, tensor_ptr) ((void)0)
#define LLAMINAR_SNAPSHOT_MPI(manager_ptr, stage, layer_idx, token_idx, tensor_ptr, mpi_ctx) ((void)0)
#endif
```

### 5. Pipeline Snapshot Manager

**File**: `src/v2/utils/PipelineSnapshotManager.h`

```cpp
namespace llaminar2 {
namespace snapshot {

/**
 * @brief Per-pipeline snapshot storage (thread-safe)
 * 
 * Design:
 * - Each pipeline instance owns a SnapshotManager
 * - No global state (unlike V1's singleton registry)
 * - Thread-safe via mutex (for parallel test execution)
 */
class PipelineSnapshotManager {
public:
    PipelineSnapshotManager() = default;
    ~PipelineSnapshotManager() = default;
    
    // Disable copy/move (owned by pipeline)
    PipelineSnapshotManager(const PipelineSnapshotManager&) = delete;
    PipelineSnapshotManager& operator=(const PipelineSnapshotManager&) = delete;
    
    /**
     * @brief Add snapshot to storage
     */
    void addSnapshot(const SnapshotData& snapshot);
    
    /**
     * @brief Query snapshot by stage name
     * 
     * @param stage_name Formatted stage name (e.g., "token_0/ATTENTION_NORM_layer5.npy")
     * @return Pointer to snapshot data, or nullptr if not found
     */
    const SnapshotData* getSnapshot(const std::string& stage_name) const;
    
    /**
     * @brief Get all snapshots (for batch saving)
     */
    std::vector<SnapshotData> getAllSnapshots() const;
    
    /**
     * @brief Save all snapshots to directory (NPY format)
     * 
     * @param output_dir Directory to save snapshots (created if needed)
     * @return Number of snapshots saved
     */
    int saveAll(const std::string& output_dir) const;
    
    /**
     * @brief Load snapshots from directory
     * 
     * @param input_dir Directory containing NPY files
     * @return Number of snapshots loaded
     */
    int loadAll(const std::string& input_dir);
    
    /**
     * @brief Clear all snapshots
     */
    void clear();
    
    /**
     * @brief Get snapshot count
     */
    size_t size() const;
    
private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, SnapshotData> snapshots_;
};

} // namespace snapshot
} // namespace llaminar2
```

### 6. Snapshot Comparator (Test Utility)

**File**: `src/v2/utils/SnapshotComparator.h`

```cpp
namespace llaminar2 {
namespace snapshot {

/**
 * @brief Comparison metrics
 */
struct ComparisonMetrics {
    float max_abs_diff = 0.0f;    // Maximum absolute difference
    float mean_abs_diff = 0.0f;   // Mean absolute difference
    float rel_l2_norm = 0.0f;     // Relative L2 norm
    size_t num_mismatches = 0;    // Number of elements exceeding tolerance
    
    bool passed = false;          // Whether comparison passed
    std::string message;          // Diagnostic message
};

/**
 * @brief Dynamic variance threshold
 */
struct VarianceThreshold {
    float variance = 0.0f;        // Measured PyTorch variance
    float magnitude = 0.0f;       // Tensor magnitude (mean abs value)
    float threshold = 0.0f;       // Computed threshold
    
    static constexpr float MIN_THRESHOLD = 1e-5f;
    static constexpr float MAGNITUDE_FACTOR = 0.015f;  // 1.5%
    static constexpr float VARIANCE_MULTIPLIER = 5.0f;
};

/**
 * @brief Snapshot comparison utility
 */
class SnapshotComparator {
public:
    /**
     * @brief Compare two snapshots with dynamic thresholds
     * 
     * @param llaminar Llaminar snapshot
     * @param pytorch PyTorch reference snapshot
     * @param variance_threshold Precomputed variance threshold (optional)
     * @return Comparison metrics
     */
    static ComparisonMetrics compare(
        const SnapshotData& llaminar,
        const SnapshotData& pytorch,
        const VarianceThreshold* variance_threshold = nullptr
    );
    
    /**
     * @brief Compute variance threshold from PyTorch runs
     * 
     * @param pytorch_snapshots Vector of PyTorch snapshots (3+ runs recommended)
     * @return Variance threshold
     */
    static VarianceThreshold computeVarianceThreshold(
        const std::vector<SnapshotData>& pytorch_snapshots
    );
    
    /**
     * @brief Load PyTorch reference snapshots from directory
     * 
     * @param snapshot_dir Directory containing NPY files
     * @return Map of stage_name → SnapshotData
     */
    static std::unordered_map<std::string, SnapshotData> loadPyTorchSnapshots(
        const std::string& snapshot_dir
    );
    
    /**
     * @brief Generate variance thresholds for all stages
     * 
     * Runs PyTorch 3 times, measures variance, computes thresholds
     * 
     * @param model_path Path to GGUF model
     * @param snapshot_dir Output directory for thresholds
     * @param n_runs Number of PyTorch runs (default: 3)
     * @return Map of stage_name → VarianceThreshold
     */
    static std::unordered_map<std::string, VarianceThreshold> generateVarianceThresholds(
        const std::string& model_path,
        const std::string& snapshot_dir,
        int n_runs = 3
    );
};

} // namespace snapshot
} // namespace llaminar2
```

---

## Usage Examples

### Example 1: Basic Pipeline Integration

**File**: `src/v2/pipelines/qwen/Qwen2Pipeline.cpp`

```cpp
#include "../../utils/SnapshotCapture.h"

bool Qwen2Pipeline::forward_batch(const std::vector<std::vector<int>>& token_batches) {
    // ... existing code ...
    
    // Embedding
    if (!embedding_batch(token_batches, current_hidden_.get())) {
        return false;
    }
    LLAMINAR_SNAPSHOT(snapshot_mgr_.get(), 
                     snapshot::PipelineStage::GLOBAL_EMBEDDING,
                     -1, current_positions_[0], current_hidden_.get());
    
    // Transformer layers
    for (int layer = 0; layer < n_layers_; ++layer) {
        LLAMINAR_SNAPSHOT(snapshot_mgr_.get(),
                         snapshot::PipelineStage::LAYER_INPUT,
                         layer, current_positions_[0], current_hidden_.get());
        
        if (!transformer_layer(layer, effective_seq_len)) {
            return false;
        }
    }
    
    // Final norm
    auto final_norm = getFinalNorm();
    norm_kernel->apply(/* ... */);
    LLAMINAR_SNAPSHOT(snapshot_mgr_.get(),
                     snapshot::PipelineStage::GLOBAL_FINAL_NORM,
                     -1, current_positions_[0], current_hidden_.get());
    
    // LM head
    if (!lm_head_batch(current_hidden_.get(), effective_seq_len)) {
        return false;
    }
    LLAMINAR_SNAPSHOT(snapshot_mgr_.get(),
                     snapshot::PipelineStage::GLOBAL_LM_HEAD,
                     -1, current_positions_[0], logits_.get());
    
    return true;
}

bool Qwen2Pipeline::attention_block(const LayerWeights& layer, int layer_idx, int effective_seq_len) {
    // ... RMSNorm ...
    LLAMINAR_SNAPSHOT(snapshot_mgr_.get(),
                     snapshot::PipelineStage::LAYER_ATTN_NORM,
                     layer_idx, current_positions_[0], normalized_hidden.get());
    
    // Q/K/V projections
    q_gemm->execute(/* ... */);
    LLAMINAR_SNAPSHOT(snapshot_mgr_.get(),
                     snapshot::PipelineStage::ATTN_Q_PROJECTION,
                     layer_idx, current_positions_[0], buffers.Q.get());
    
    k_gemm->execute(/* ... */);
    LLAMINAR_SNAPSHOT(snapshot_mgr_.get(),
                     snapshot::PipelineStage::ATTN_K_PROJECTION,
                     layer_idx, current_positions_[0], buffers.K.get());
    
    // ... RoPE, attention computation ...
    
    LLAMINAR_SNAPSHOT(snapshot_mgr_.get(),
                     snapshot::PipelineStage::LAYER_ATTN_OUTPUT,
                     layer_idx, current_positions_[0], buffers.attn_proj.get());
    
    // Residual
    // ... add residual ...
    LLAMINAR_SNAPSHOT(snapshot_mgr_.get(),
                     snapshot::PipelineStage::LAYER_ATTN_RESIDUAL,
                     layer_idx, current_positions_[0], current_hidden_.get());
    
    return true;
}
```

### Example 2: Parity Test

**File**: `tests/v2/parity/Test__Qwen2_Parity.cpp`

```cpp
#include <gtest/gtest.h>
#include "../../../src/v2/pipelines/qwen/Qwen2Pipeline.h"
#include "../../../src/v2/utils/SnapshotComparator.h"

using namespace llaminar2;
using namespace llaminar2::snapshot;

TEST(Qwen2Parity, PrefillParity) {
    // 1. Generate PyTorch reference snapshots
    std::string model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
    std::string snapshot_dir = "pytorch_snapshots_v2";
    
    // Generate variance thresholds (runs PyTorch 3 times)
    auto thresholds = SnapshotComparator::generateVarianceThresholds(
        model_path, snapshot_dir, 3);
    
    // 2. Run Llaminar inference with snapshot capture
    auto mpi_ctx = std::make_shared<MPIContext>(MPI_COMM_WORLD);
    auto model_ctx = std::make_shared<ModelContext>(model_path);
    
    PipelineConfig config;
    config.max_seq_len = 2048;
    config.n_threads = 28;
    config.batch_size = 1;
    
    auto pipeline = std::make_unique<Qwen2Pipeline>(
        model_ctx, mpi_ctx, 0, nullptr, config, 1);
    
    // Enable snapshot capture
    pipeline->enableSnapshots();
    
    // Run inference
    std::vector<int> tokens = {1234, 5678, 91011}; // Example tokens
    ASSERT_TRUE(pipeline->forward(tokens.data(), tokens.size()));
    
    // Save Llaminar snapshots
    pipeline->getSnapshotManager()->saveAll("llaminar_snapshots_v2");
    
    // 3. Compare snapshots
    auto pytorch_snapshots = SnapshotComparator::loadPyTorchSnapshots(snapshot_dir);
    auto llaminar_snapshots = pipeline->getSnapshotManager()->getAllSnapshots();
    
    int passed = 0, failed = 0;
    for (const auto& llaminar_snap : llaminar_snapshots) {
        auto pytorch_it = pytorch_snapshots.find(llaminar_snap.stage_name);
        if (pytorch_it == pytorch_snapshots.end()) {
            LOG_WARN("Missing PyTorch snapshot: " << llaminar_snap.stage_name);
            continue;
        }
        
        // Get variance threshold for this stage
        const VarianceThreshold* threshold = nullptr;
        auto thresh_it = thresholds.find(llaminar_snap.stage_name);
        if (thresh_it != thresholds.end()) {
            threshold = &thresh_it->second;
        }
        
        // Compare
        auto metrics = SnapshotComparator::compare(
            llaminar_snap, pytorch_it->second, threshold);
        
        if (metrics.passed) {
            ++passed;
            LOG_INFO("✓ " << llaminar_snap.stage_name << " PASSED");
        } else {
            ++failed;
            LOG_ERROR("✗ " << llaminar_snap.stage_name << " FAILED");
            LOG_ERROR("  " << metrics.message);
        }
    }
    
    LOG_INFO("Results: " << passed << " passed, " << failed << " failed");
    EXPECT_EQ(failed, 0) << "Parity test failed for " << failed << " stages";
}
```

### Example 3: Batch-Aware Snapshots

**File**: `src/v2/pipelines/qwen/Qwen2Pipeline.cpp`

```cpp
bool Qwen2Pipeline::forward_batch(const std::vector<std::vector<int>>& token_batches) {
    // ... existing code ...
    
    // Capture per-sequence snapshots in batch
    for (int seq_idx = 0; seq_idx < batch_size_; ++seq_idx) {
        // Extract single sequence data from batched tensor
        auto seq_tensor = extractSequence(current_hidden_.get(), seq_idx, padded_seq_len_);
        
        LLAMINAR_SNAPSHOT_BATCH(snapshot_mgr_.get(),
                               snapshot::PipelineStage::GLOBAL_EMBEDDING,
                               -1, current_positions_[seq_idx], seq_idx,
                               seq_tensor.get());
    }
    
    // ... rest of forward pass ...
}
```

---

## Python Integration

### V2 Snapshot Adapter

**File**: `python/reference/v2_snapshot_adapter.py`

```python
"""
V2 Snapshot Adapter: Bridges V2 pipeline snapshots to V1's PyTorch reference scripts
"""

import numpy as np
from pathlib import Path
from typing import Dict, List
import torch

from .qwen import QwenReference
from .pipeline_stages import PipelineStage

class V2SnapshotAdapter:
    """Adapts V2 snapshot format to V1 PyTorch reference"""
    
    # V2 → V1 stage mapping
    STAGE_MAP = {
        "GLOBAL_EMBEDDING": PipelineStage.EMBEDDING,
        "LAYER_ATTN_NORM": PipelineStage.ATTENTION_NORM,
        "ATTN_Q_PROJECTION": PipelineStage.Q_PROJECTION,
        "ATTN_K_PROJECTION": PipelineStage.K_PROJECTION,
        "ATTN_V_PROJECTION": PipelineStage.V_PROJECTION,
        "ATTN_ROPE_APPLIED": PipelineStage.ROPE_APPLICATION,
        "ATTN_SCORES": PipelineStage.ATTENTION_SCORES,
        "ATTN_WEIGHTS": PipelineStage.ATTENTION_SOFTMAX,
        "ATTN_CONTEXT": PipelineStage.ATTENTION_CONTEXT,
        "LAYER_ATTN_OUTPUT": PipelineStage.ATTENTION_OUTPUT,
        "LAYER_ATTN_RESIDUAL": PipelineStage.ATTENTION_RESIDUAL,
        "LAYER_FFN_NORM": PipelineStage.FFN_NORM,
        "FFN_GATE": PipelineStage.FFN_GATE,
        "FFN_UP": PipelineStage.FFN_UP,
        "FFN_SWIGLU": PipelineStage.FFN_SWIGLU,
        "LAYER_FFN_OUTPUT": PipelineStage.FFN_DOWN,
        "LAYER_FFN_RESIDUAL": PipelineStage.FFN_RESIDUAL,
        "GLOBAL_FINAL_NORM": PipelineStage.FINAL_NORM,
        "GLOBAL_LM_HEAD": PipelineStage.LM_HEAD,
    }
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.reference = QwenReference(str(model_path))
    
    def generate_snapshots(
        self, 
        tokens: List[int], 
        output_dir: str,
        n_runs: int = 3
    ) -> Dict[str, np.ndarray]:
        """
        Generate PyTorch reference snapshots using V2 stage names
        
        Args:
            tokens: Input token IDs
            output_dir: Output directory for snapshots
            n_runs: Number of runs for variance computation
        
        Returns:
            Map of V2 stage names to snapshot data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        snapshots = {}
        
        # Run PyTorch inference with snapshot capture
        for run_idx in range(n_runs):
            run_dir = output_path / f"run_{run_idx}"
            run_dir.mkdir(exist_ok=True)
            
            # Generate V1-format snapshots
            v1_snapshots = self.reference.generate_snapshots(tokens, str(run_dir))
            
            # Convert V1 → V2 stage names
            for v1_stage_name, data in v1_snapshots.items():
                v2_stage_name = self._convert_stage_name(v1_stage_name)
                snapshots[v2_stage_name] = data
                
                # Save with V2 naming
                np.save(run_dir / f"{v2_stage_name}.npy", data)
        
        return snapshots
    
    def _convert_stage_name(self, v1_name: str) -> str:
        """Convert V1 stage name to V2 format"""
        # V1: "token_0/ATTENTION_NORM_layer5.npy"
        # V2: "token_0/LAYER_ATTN_NORM_layer5.npy"
        
        parts = v1_name.split("/")
        if len(parts) != 2:
            return v1_name
        
        token_part, stage_part = parts
        
        # Extract stage and layer
        for v2_stage, v1_stage in self.STAGE_MAP.items():
            if v1_stage.name in stage_part:
                return f"{token_part}/{v2_stage}{stage_part[len(v1_stage.name):]}"
        
        return v1_name
```

### Variance Threshold Generation Script

**File**: `python/reference/generate_v2_variance_thresholds.py`

```python
#!/usr/bin/env python3
"""
Generate dynamic variance thresholds for V2 parity tests
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict
from v2_snapshot_adapter import V2SnapshotAdapter

def compute_variance_thresholds(
    snapshots_by_run: Dict[int, Dict[str, np.ndarray]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute variance thresholds from multiple PyTorch runs
    
    Args:
        snapshots_by_run: Map of run_idx → (stage_name → snapshot_data)
    
    Returns:
        Map of stage_name → {variance, magnitude, threshold}
    """
    thresholds = {}
    
    # Get all stage names
    stage_names = set()
    for run_snapshots in snapshots_by_run.values():
        stage_names.update(run_snapshots.keys())
    
    for stage_name in sorted(stage_names):
        # Collect data from all runs
        run_data = []
        for run_idx in sorted(snapshots_by_run.keys()):
            if stage_name in snapshots_by_run[run_idx]:
                run_data.append(snapshots_by_run[run_idx][stage_name])
        
        if len(run_data) < 2:
            print(f"WARNING: {stage_name} has < 2 runs, skipping")
            continue
        
        # Stack runs: [n_runs, ...]
        stacked = np.stack(run_data, axis=0)
        
        # Compute variance across runs
        variance = np.var(stacked, axis=0).mean()
        
        # Compute magnitude (mean abs value of first run)
        magnitude = np.abs(run_data[0]).mean()
        
        # Compute threshold: max(variance, magnitude × 1.5%) × 5.0
        MIN_THRESHOLD = 1e-5
        MAGNITUDE_FACTOR = 0.015  # 1.5%
        VARIANCE_MULTIPLIER = 5.0
        
        threshold = max(variance, magnitude * MAGNITUDE_FACTOR) * VARIANCE_MULTIPLIER
        threshold = max(threshold, MIN_THRESHOLD)
        
        thresholds[stage_name] = {
            "variance": float(variance),
            "magnitude": float(magnitude),
            "threshold": float(threshold)
        }
        
        print(f"{stage_name:50s} variance={variance:9.2e} mag={magnitude:9.2e} thresh={threshold:9.2e}")
    
    return thresholds

def main():
    parser = argparse.ArgumentParser(description="Generate V2 variance thresholds")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--output-dir", default="pytorch_snapshots_v2", help="Output directory")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of PyTorch runs")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1234, 5678, 91011], 
                       help="Input token IDs")
    args = parser.parse_args()
    
    print(f"Generating variance thresholds for {args.model}")
    print(f"  n_runs: {args.n_runs}")
    print(f"  tokens: {args.tokens}")
    print()
    
    # Generate snapshots
    adapter = V2SnapshotAdapter(args.model)
    
    snapshots_by_run = {}
    for run_idx in range(args.n_runs):
        print(f"=== Run {run_idx + 1}/{args.n_runs} ===")
        run_dir = Path(args.output_dir) / f"run_{run_idx}"
        snapshots = adapter.generate_snapshots(args.tokens, str(run_dir), n_runs=1)
        snapshots_by_run[run_idx] = snapshots
    
    # Compute thresholds
    print("\n=== Computing Variance Thresholds ===")
    thresholds = compute_variance_thresholds(snapshots_by_run)
    
    # Save to JSON
    import json
    threshold_file = Path(args.output_dir) / "variance_thresholds.json"
    with open(threshold_file, "w") as f:
        json.dump(thresholds, f, indent=2)
    
    print(f"\n✓ Saved variance thresholds to {threshold_file}")
    print(f"  Total stages: {len(thresholds)}")

if __name__ == "__main__":
    main()
```

---

## Environment Configuration

### Snapshot Control

```bash
# Enable snapshot capture (debug builds only)
export LLAMINAR_SNAPSHOT_ENABLED=1

# Filter stages to capture (default: all)
export LLAMINAR_SNAPSHOT_STAGES="attention,ffn"
# Options: all, global, layer, attention, ffn, or comma-separated list

# MPI rank filter (default: rank 0 only)
export LLAMINAR_SNAPSHOT_RANKS="0"
# Options: "0", "all", "0,1,2"

# Output directory for auto-save (optional)
export LLAMINAR_SNAPSHOT_DIR="llaminar_snapshots_v2"
```

### Usage in Tests

```cpp
// Enable snapshot capture in test
TEST(Qwen2Parity, PrefillParity) {
    // Set environment (overrides any shell settings)
    setenv("LLAMINAR_SNAPSHOT_ENABLED", "1", 1);
    setenv("LLAMINAR_SNAPSHOT_STAGES", "attention,global", 1);
    
    // ... run test ...
}
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Create `src/v2/utils/SnapshotCapture.h` with:
  - [ ] PipelineStage enum (hierarchical)
  - [ ] SnapshotData struct
  - [ ] formatStageName() helper
  - [ ] isStageEnabled() helper
  - [ ] LLAMINAR_SNAPSHOT macros

- [ ] Create `src/v2/utils/PipelineSnapshotManager.h/cpp`:
  - [ ] Thread-safe snapshot storage
  - [ ] saveAll() / loadAll() NPY I/O
  - [ ] Query interface

- [ ] Add `TensorBase::captureSnapshot()` method:
  - [ ] Device→CPU sync logic
  - [ ] `#ifndef NDEBUG` guard

### Phase 2: Integration
- [ ] Update `Qwen2Pipeline`:
  - [ ] Add `snapshot_mgr_` member
  - [ ] Add snapshots in `forward_batch()`
  - [ ] Add snapshots in `attention_block()`
  - [ ] Add snapshots in `ffn_block()`

- [ ] Update `PipelineBase`:
  - [ ] Add `enableSnapshots()` method
  - [ ] Add `getSnapshotManager()` accessor

### Phase 3: Comparison Utilities
- [ ] Create `src/v2/utils/SnapshotComparator.h/cpp`:
  - [ ] ComparisonMetrics struct
  - [ ] VarianceThreshold struct
  - [ ] compare() method
  - [ ] computeVarianceThreshold() method
  - [ ] loadPyTorchSnapshots() method

### Phase 4: Python Integration
- [ ] Create `python/reference/v2_snapshot_adapter.py`:
  - [ ] V2SnapshotAdapter class
  - [ ] Stage name mapping (V1↔V2)
  - [ ] generate_snapshots() method

- [ ] Create `python/reference/generate_v2_variance_thresholds.py`:
  - [ ] Multi-run PyTorch execution
  - [ ] Variance computation
  - [ ] Threshold generation
  - [ ] JSON export

### Phase 5: Testing
- [ ] Create `tests/v2/parity/Test__Qwen2_Parity.cpp`:
  - [ ] Prefill parity test
  - [ ] Incremental decode parity test
  - [ ] Weight verification test

- [ ] Create `tests/v2/parity/ParityTestUtils.h`:
  - [ ] Shared test utilities
  - [ ] Snapshot loading helpers
  - [ ] Comparison wrappers

### Phase 6: Documentation
- [ ] Update `docs/parity-test-framework.instructions.md`:
  - [ ] V2-specific sections
  - [ ] Migration guide (V1→V2)
  - [ ] Usage examples

- [ ] Create `docs/v2/SNAPSHOT_USAGE_GUIDE.md`:
  - [ ] Step-by-step tutorial
  - [ ] Debugging workflows
  - [ ] Best practices

---

## Migration from V1

### Key Differences

| Aspect | V1 | V2 |
|--------|----|----|
| **Capture point** | Operator layer | Pipeline (direct kernel calls) |
| **Storage** | Global SnapshotRegistry singleton | Per-pipeline SnapshotManager |
| **Stage naming** | Flat enum | Hierarchical enum (GLOBAL_*/LAYER_*/ATTN_*/FFN_*) |
| **Batch support** | Manual per-sequence extraction | Built-in LLAMINAR_SNAPSHOT_BATCH macro |
| **Device handling** | CPU-only | Automatic device→CPU sync |
| **Compile overhead** | Always compiled | `#ifndef NDEBUG` (zero release overhead) |

### Migration Steps

1. **Update stage names**: V1's `ATTENTION_NORM` → V2's `LAYER_ATTN_NORM`
2. **Replace registry calls**: `SnapshotRegistry::instance().addSnapshot()` → `snapshot_mgr_->addSnapshot()`
3. **Update macros**: Old `CAPTURE_SNAPSHOT()` → New `LLAMINAR_SNAPSHOT()`
4. **Adapt Python scripts**: Use `v2_snapshot_adapter.py` for backward compatibility

---

## Performance Impact

### Debug Builds (Snapshots Enabled)
- **Memory overhead**: ~10-50 MB per snapshot (depends on tensor size)
- **Time overhead**: ~1-5 ms per snapshot (device sync + memcpy)
- **Total overhead**: ~10-20% slowdown for full pipeline capture (100+ snapshots)

### Release Builds (Snapshots Disabled)
- **Memory overhead**: 0 bytes (macros compile to no-op)
- **Time overhead**: 0 ns (compiler removes dead code)
- **Total overhead**: 0% (verified with `nm` symbol inspection)

---

## Future Enhancements

### Phase 7: Multi-Device Snapshots
- Capture tensors on different devices (GPU, NPU, etc.)
- Automatic device→CPU transfer with async pipelines
- Device-specific comparison (e.g., CPU vs GPU outputs)

### Phase 8: Streaming Snapshots
- Stream snapshots to disk during inference (reduce memory)
- Binary format for faster I/O (MessagePack, FlatBuffers)
- Incremental comparison (don't wait for full run)

### Phase 9: Visual Debugging
- Snapshot diff visualization (heatmaps, histograms)
- Interactive comparison tool (web UI)
- Automatic divergence localization

---

## Tensor Dump Feature (Implemented)

**Status**: Implemented (December 2025)

The tensor dump feature allows saving raw tensor data to disk for debugging, integrated with the snapshot framework but available in any build (including Release).

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMINAR_SNAPSHOT_TENSOR_DUMP` | Enable tensor dump to disk | `0` (disabled) |
| `LLAMINAR_SNAPSHOT_DUMP_DIR` | Output directory | `/tmp/llaminar_tensor_dumps` |
| `LLAMINAR_SNAPSHOT_DUMP_LAYERS` | Layers to dump (comma-separated, or `all`) | `all` |
| `LLAMINAR_SNAPSHOT_DUMP_STAGES` | Stages to dump (comma-separated, or `all`) | `all` |
| `LLAMINAR_SNAPSHOT_DUMP_RANK` | MPI rank to dump (-1 for all) | `0` |

### Output Files

For each captured stage, three files are written:
- `<key>_rank<N>_fp32.bin` - FP32 dequantized data (raw float32 binary)
- `<key>_rank<N>_q8_1_blocks.bin` - Raw Q8_1 blocks (for Q8_1 tensors only)
- `<key>_rank<N>_metadata.txt` - Shape, type, and sample scale information

### Stage Names

**Global stages** (layer_idx = -1):
- `EMBEDDING` - Token embedding lookup
- `FINAL_NORM` - Final RMSNorm before LM head
- `LM_HEAD` - Final vocabulary projection (logits)

**Per-layer stages** (layer_idx = 0, 1, 2, ...):
- `ATTENTION_NORM` - Pre-attention RMSNorm
- `Q_PROJECTION`, `K_PROJECTION`, `V_PROJECTION` - QKV projections
- `Q_ROPE`, `K_ROPE` - After RoPE application
- `ATTENTION_CONTEXT` - Attention output (before output projection)
- `ATTENTION_OUTPUT` - After attention output projection
- `ATTENTION_RESIDUAL` - After attention residual connection
- `FFN_NORM` - Pre-FFN RMSNorm
- `FFN_GATE`, `FFN_UP` - FFN gate and up projections
- `FFN_SWIGLU` - After SwiGLU activation
- `FFN_DOWN` - FFN down projection
- `FFN_INPUT_RESIDUAL` - Saved residual before FFN processing
- `FFN_RESIDUAL` - After FFN residual connection

### Usage Example

```bash
# Dump only layer 21 FFN stages for debugging Q8_1 divergence
LLAMINAR_SNAPSHOT_TENSOR_DUMP=1 \
LLAMINAR_SNAPSHOT_DUMP_LAYERS=21 \
LLAMINAR_SNAPSHOT_DUMP_STAGES=FFN_INPUT_RESIDUAL,FFN_DOWN,FFN_RESIDUAL \
LLAMINAR_SNAPSHOT_DUMP_DIR=/tmp/layer21_debug \
./run_llaminar.sh -m models/qwen2.5-0.5b-instruct-q8_0.gguf -p "test prompt"

# View metadata
cat /tmp/layer21_debug/layer21_FFN_RESIDUAL_rank0_metadata.txt

# Load FP32 data in Python
import numpy as np
data = np.fromfile('/tmp/layer21_debug/layer21_FFN_RESIDUAL_rank0_fp32.bin', dtype=np.float32)
rows, cols = 9, 896  # From metadata
data = data.reshape(rows, cols)
```

### Implementation Details

- **Source**: `src/v2/utils/DebugEnv.h` (SnapshotConfig struct), `src/v2/pipelines/PipelineBase.cpp` (maybeDumpTensor method)
- **Integration**: Automatically called by CAPTURE_SNAPSHOT* macros
- **Thread Safety**: Uses debugEnv() singleton (parsed once at startup)
- **Zero Overhead**: Quick exits if dump not enabled (no getenv calls on hot path)

---

## References

- V1 Parity Framework: `docs/parity-test-framework.instructions.md`
- V2 Architecture: `.github/instructions/llaminar-v2-architecture.instructions.md`
- Dynamic Variance Thresholds: `docs/DYNAMIC_VARIANCE_THRESHOLDS.md`
- Python Reference: `python/reference/README.md`

---

## Conclusion

This design provides a robust, zero-overhead snapshot framework for V2 parity testing that:
- **Integrates seamlessly** with V2's operator-free architecture
- **Scales to heterogeneous devices** (CPU, GPU, NPU)
- **Supports batch processing** natively
- **Reuses V1's proven comparison infrastructure** (variance thresholds)
- **Compiles to zero overhead** in release builds
- **Provides tensor dump capability** for detailed debugging in any build

Next steps: Implement Phase 1 (core infrastructure) and validate with Qwen2 parity test.
