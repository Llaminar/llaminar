# Llaminar V2 Testability Refactoring Plan

**Document Version:** 1.0  
**Created:** January 11, 2026  
**Status:** Planning

## Executive Summary

This document outlines a phased approach to improve testability of Llaminar V2's distributed execution infrastructure. The goal is to enable comprehensive unit and integration testing of:

- Multi-rank execution without actual MPI
- Heterogeneous CPU/GPU device placement
- Weight streaming and placement strategies
- Phase-aware execution (prefill vs decode)

### Current State Analysis

| Category | Classes with Interfaces | Classes Needing Interfaces |
|----------|------------------------|---------------------------|
| Kernels | 14 (ITensorGemm, ITensorAttention, etc.) | 0 |
| MPI Layer | 0 | 2 (MPIContext, MPITopology) |
| Model Loading | 0 | 2 (ModelLoader, WeightManager) |
| Execution | 0 | 5 (ModelContext, CollectiveContext, etc.) |

### Priority Matrix

| Priority | Class | Interface | Blocking Factor |
|----------|-------|-----------|-----------------|
| CRITICAL | `MPIContext` | `IMPIContext` | Blocks ALL distributed tests |
| CRITICAL | `MPITopology` | `IMPITopology` | Blocks topology/placement tests |
| HIGH | `ModelLoader` | `IModelLoader` | Blocks weight loading tests |
| HIGH | `WeightManager` | `IWeightManager` | Blocks weight distribution tests |
| HIGH | `ModelContext` | `IModelContext` | Facade for model access |
| MEDIUM | `CollectiveContext` | `ICollectiveContext` | Blocks collective operation tests |
| MEDIUM | `WeightPlacementMap` | `IWeightPlacementMap` | Blocks placement strategy tests |
| MEDIUM | `WorkDistributor` | `IWorkDistributor` | Blocks work distribution tests |
| MEDIUM | `DeviceGraphBufferManager` | `IGraphBufferManager` | Blocks buffer allocation tests |

---

## Phase 1: MPI Layer Interfaces

**Goal:** Enable testing of distributed execution logic without real MPI.

### 1.1 IMPIContext Interface Design

**Location:** `src/v2/interfaces/IMPIContext.h`

```cpp
#pragma once
#include <cstddef>
#include <memory>
#include <vector>

namespace llaminar2 {

/// Abstract interface for MPI context operations
/// Enables testing distributed logic without real MPI runtime
class IMPIContext {
public:
    virtual ~IMPIContext() = default;

    // === Identity ===
    virtual int rank() const = 0;
    virtual int world_size() const = 0;
    virtual bool is_root() const = 0;

    // === Collective Operations ===
    virtual void barrier() const = 0;
    virtual void allreduce_sum_inplace(float* data, size_t count) const = 0;
    virtual void allreduce_sum_inplace(double* data, size_t count) const = 0;
    virtual void allgather(const void* sendbuf, void* recvbuf, 
                           size_t count, size_t element_size) const = 0;
    virtual void broadcast(void* data, size_t count, 
                           size_t element_size, int root) const = 0;

    // === Point-to-Point ===
    virtual void send(const void* data, size_t count, 
                      size_t element_size, int dest, int tag) const = 0;
    virtual void recv(void* data, size_t count, 
                      size_t element_size, int source, int tag) const = 0;

    // === Query ===
    virtual bool is_initialized() const = 0;
    virtual bool is_finalized() const = 0;

    // === Factory ===
    static std::shared_ptr<IMPIContext> create_real();
    static std::shared_ptr<IMPIContext> create_mock(int rank, int world_size);
};

} // namespace llaminar2
```

### 1.2 IMPITopology Interface Design

**Location:** `src/v2/interfaces/IMPITopology.h`

```cpp
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "interfaces/IMPIContext.h"
#include "devices/DeviceId.h"

namespace llaminar2 {

/// Device capability information for a single rank
struct RankDeviceInfo {
    int rank;
    std::vector<DeviceId> devices;           // Available devices on this rank
    size_t total_memory_bytes;               // Total device memory
    size_t available_memory_bytes;           // Available device memory
    bool has_gpu;                            // Has any GPU
    bool has_cuda;                           // Has CUDA GPU
    bool has_rocm;                           // Has ROCm GPU
    int numa_node;                           // NUMA node affinity
};

/// Abstract interface for MPI topology discovery
/// Enables testing heterogeneous configurations without real hardware
class IMPITopology {
public:
    virtual ~IMPITopology() = default;

    // === Topology Discovery ===
    virtual int world_size() const = 0;
    virtual int local_rank() const = 0;
    virtual int local_size() const = 0;
    virtual int node_count() const = 0;

    // === Device Queries ===
    virtual const RankDeviceInfo& rank_info(int rank) const = 0;
    virtual const RankDeviceInfo& local_info() const = 0;
    virtual std::vector<DeviceId> all_devices() const = 0;
    virtual std::vector<DeviceId> local_devices() const = 0;

    // === Capability Queries ===
    virtual bool is_homogeneous() const = 0;          // All ranks have same devices
    virtual bool has_any_gpu() const = 0;             // Any rank has GPU
    virtual bool all_have_gpu() const = 0;            // All ranks have GPU
    virtual size_t min_device_memory() const = 0;     // Minimum across all ranks
    virtual size_t total_device_memory() const = 0;   // Sum across all ranks

    // === Sharding Decisions ===
    virtual int ranks_per_node() const = 0;
    virtual int gpus_per_rank() const = 0;
    virtual bool should_enable_tensor_parallelism() const = 0;

    // === Factory ===
    static std::unique_ptr<IMPITopology> discover(const IMPIContext& ctx);
    static std::unique_ptr<IMPITopology> create_mock(
        std::vector<RankDeviceInfo> rank_infos);
};

} // namespace llaminar2
```

### 1.3 MockMPIContext Implementation

**Location:** `tests/v2/mocks/MockMPIContext.h`

```cpp
#pragma once
#include "interfaces/IMPIContext.h"
#include <functional>
#include <map>
#include <mutex>
#include <queue>
#include <condition_variable>

namespace llaminar2::test {

/// Simulated MPI context for unit testing
/// Supports configurable behavior and collective simulation
class MockMPIContext : public IMPIContext {
public:
    // === Configuration ===
    struct Config {
        int rank = 0;
        int world_size = 1;
        bool simulate_collectives = true;  // Actually accumulate in allreduce
        bool track_calls = true;           // Record call history
    };

    explicit MockMPIContext(Config config = {});
    ~MockMPIContext() override = default;

    // === IMPIContext Implementation ===
    int rank() const override { return config_.rank; }
    int world_size() const override { return config_.world_size; }
    bool is_root() const override { return config_.rank == 0; }

    void barrier() const override;
    void allreduce_sum_inplace(float* data, size_t count) const override;
    void allreduce_sum_inplace(double* data, size_t count) const override;
    void allgather(const void* sendbuf, void* recvbuf,
                   size_t count, size_t element_size) const override;
    void broadcast(void* data, size_t count,
                   size_t element_size, int root) const override;
    void send(const void* data, size_t count,
              size_t element_size, int dest, int tag) const override;
    void recv(void* data, size_t count,
              size_t element_size, int source, int tag) const override;

    bool is_initialized() const override { return true; }
    bool is_finalized() const override { return false; }

    // === Test Utilities ===
    size_t barrier_call_count() const;
    size_t allreduce_call_count() const;
    void reset_call_counts();

    // === Multi-Rank Simulation ===
    /// Connect multiple MockMPIContext instances for collective simulation
    static void connect_ranks(std::vector<std::shared_ptr<MockMPIContext>>& contexts);

    // === Failure Injection ===
    void inject_barrier_failure(bool enable);
    void inject_allreduce_corruption(float factor);  // Multiply results

private:
    Config config_;
    mutable size_t barrier_calls_ = 0;
    mutable size_t allreduce_calls_ = 0;
    mutable std::mutex mutex_;

    // For multi-rank simulation
    std::vector<std::weak_ptr<MockMPIContext>> connected_ranks_;
    mutable std::map<int, std::queue<std::vector<char>>> message_queues_;
    mutable std::condition_variable message_cv_;

    // Failure injection
    bool barrier_fails_ = false;
    float allreduce_corruption_factor_ = 1.0f;
};

} // namespace llaminar2::test
```

### 1.4 MockMPITopology Implementation

**Location:** `tests/v2/mocks/MockMPITopology.h`

```cpp
#pragma once
#include "interfaces/IMPITopology.h"
#include <stdexcept>

namespace llaminar2::test {

/// Mock topology for testing heterogeneous configurations
class MockMPITopology : public IMPITopology {
public:
    /// Create homogeneous topology (all ranks identical)
    static std::unique_ptr<MockMPITopology> create_homogeneous(
        int world_size,
        std::vector<DeviceId> devices_per_rank,
        size_t memory_per_device = 8ULL * 1024 * 1024 * 1024);

    /// Create heterogeneous topology (different devices per rank)
    static std::unique_ptr<MockMPITopology> create_heterogeneous(
        std::vector<RankDeviceInfo> rank_infos);

    /// Preset: 2-rank CPU-only configuration
    static std::unique_ptr<MockMPITopology> preset_2rank_cpu_only();

    /// Preset: 2-rank single GPU each
    static std::unique_ptr<MockMPITopology> preset_2rank_single_gpu();

    /// Preset: 4-rank with mixed CPU/GPU (rank 0,1 have GPU, rank 2,3 CPU only)
    static std::unique_ptr<MockMPITopology> preset_4rank_mixed_cpu_gpu();

    /// Preset: 2-rank multi-GPU (2 GPUs per rank)
    static std::unique_ptr<MockMPITopology> preset_2rank_multi_gpu();

    // === IMPITopology Implementation ===
    int world_size() const override { return static_cast<int>(rank_infos_.size()); }
    int local_rank() const override { return local_rank_; }
    int local_size() const override { return local_size_; }
    int node_count() const override { return node_count_; }

    const RankDeviceInfo& rank_info(int rank) const override;
    const RankDeviceInfo& local_info() const override;
    std::vector<DeviceId> all_devices() const override;
    std::vector<DeviceId> local_devices() const override;

    bool is_homogeneous() const override;
    bool has_any_gpu() const override;
    bool all_have_gpu() const override;
    size_t min_device_memory() const override;
    size_t total_device_memory() const override;

    int ranks_per_node() const override { return local_size_; }
    int gpus_per_rank() const override;
    bool should_enable_tensor_parallelism() const override;

    // === Test Configuration ===
    void set_local_rank(int rank) { local_rank_ = rank; }
    void set_local_size(int size) { local_size_ = size; }
    void set_node_count(int count) { node_count_ = count; }

private:
    explicit MockMPITopology(std::vector<RankDeviceInfo> rank_infos);

    std::vector<RankDeviceInfo> rank_infos_;
    int local_rank_ = 0;
    int local_size_ = 1;
    int node_count_ = 1;
};

} // namespace llaminar2::test
```

### 1.5 Files to Create/Modify

| Action | File | Description |
|--------|------|-------------|
| CREATE | `src/v2/interfaces/IMPIContext.h` | Interface definition |
| CREATE | `src/v2/interfaces/IMPITopology.h` | Interface definition |
| CREATE | `tests/v2/mocks/MockMPIContext.h` | Mock header |
| CREATE | `tests/v2/mocks/MockMPIContext.cpp` | Mock implementation |
| CREATE | `tests/v2/mocks/MockMPITopology.h` | Mock header |
| CREATE | `tests/v2/mocks/MockMPITopology.cpp` | Mock implementation |
| MODIFY | `src/v2/utils/MPIContext.h` | Inherit from IMPIContext |
| MODIFY | `src/v2/utils/MPITopology.h` | Inherit from IMPITopology |
| MODIFY | `src/v2/CMakeLists.txt` | Add interfaces to build |
| MODIFY | `tests/v2/CMakeLists.txt` | Add mocks to test build |

---

## Phase 2: Model Loading Interfaces

**Goal:** Enable testing of weight loading and distribution without real GGUF files.

### 2.1 IModelLoader Interface Design

**Location:** `src/v2/interfaces/IModelLoader.h`

```cpp
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "tensors/TensorBase.h"
#include "loaders/GGUFMetadata.h"

namespace llaminar2 {

/// Model hyperparameters extracted from GGUF
struct ModelHyperparams {
    int vocab_size;
    int hidden_size;
    int intermediate_size;
    int num_layers;
    int num_attention_heads;
    int num_kv_heads;
    int head_dim;
    float rope_theta;
    float rms_norm_eps;
    std::string architecture;
    std::string quantization_type;
};

/// Weight tensor descriptor (before loading)
struct WeightDescriptor {
    std::string name;
    std::vector<size_t> shape;
    DType dtype;
    size_t offset_in_file;
    size_t size_bytes;
    int layer_index;  // -1 for non-layer weights
};

/// Abstract interface for model loading
class IModelLoader {
public:
    virtual ~IModelLoader() = default;

    // === Metadata Access ===
    virtual const ModelHyperparams& hyperparams() const = 0;
    virtual const std::vector<WeightDescriptor>& weight_descriptors() const = 0;
    virtual bool has_weight(const std::string& name) const = 0;

    // === Weight Loading ===
    virtual std::unique_ptr<TensorBase> load_weight(
        const std::string& name) const = 0;
    virtual std::unique_ptr<TensorBase> load_weight_shard(
        const std::string& name,
        int shard_index,
        int num_shards,
        ShardDimension dim) const = 0;

    // === Tokenizer ===
    virtual std::vector<int> tokenize(const std::string& text) const = 0;
    virtual std::string detokenize(const std::vector<int>& tokens) const = 0;
    virtual int bos_token() const = 0;
    virtual int eos_token() const = 0;

    // === Factory ===
    static std::unique_ptr<IModelLoader> from_gguf(const std::string& path);
};

} // namespace llaminar2
```

### 2.2 IWeightManager Interface Design

**Location:** `src/v2/interfaces/IWeightManager.h`

```cpp
#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include "interfaces/IModelLoader.h"
#include "interfaces/IMPIContext.h"
#include "interfaces/IMPITopology.h"
#include "tensors/TensorBase.h"
#include "devices/DeviceId.h"

namespace llaminar2 {

/// Weight placement decision for a single weight
struct WeightPlacement {
    std::string name;
    DeviceId device;
    bool is_sharded;
    int shard_index;
    int num_shards;
    ShardDimension shard_dim;
    bool is_streamed;  // For weight streaming mode
};

/// Abstract interface for weight management
class IWeightManager {
public:
    virtual ~IWeightManager() = default;

    // === Weight Access ===
    virtual TensorBase* get_weight(const std::string& name) const = 0;
    virtual bool has_weight(const std::string& name) const = 0;
    virtual std::vector<std::string> weight_names() const = 0;

    // === Placement Queries ===
    virtual const WeightPlacement& placement(const std::string& name) const = 0;
    virtual DeviceId device_for_weight(const std::string& name) const = 0;
    virtual bool is_weight_local(const std::string& name) const = 0;

    // === Memory Info ===
    virtual size_t total_weight_memory() const = 0;
    virtual size_t local_weight_memory() const = 0;
    virtual std::unordered_map<DeviceId, size_t> memory_per_device() const = 0;

    // === Weight Streaming ===
    virtual bool is_streaming_enabled() const = 0;
    virtual void prefetch_layer_weights(int layer_index) = 0;
    virtual void evict_layer_weights(int layer_index) = 0;

    // === Factory ===
    static std::unique_ptr<IWeightManager> create(
        std::shared_ptr<IModelLoader> loader,
        std::shared_ptr<IMPIContext> mpi_ctx,
        std::shared_ptr<IMPITopology> topology,
        const WeightPlacementConfig& config);
};

/// Configuration for weight placement
struct WeightPlacementConfig {
    bool enable_sharding = true;
    bool enable_streaming = false;
    size_t streaming_cache_mb = 0;  // 0 = auto
    EvictionPolicy eviction_policy = EvictionPolicy::LRU;
    int prefetch_depth = 1;
};

} // namespace llaminar2
```

### 2.3 MockModelLoader Implementation

**Location:** `tests/v2/mocks/MockModelLoader.h`

```cpp
#pragma once
#include "interfaces/IModelLoader.h"
#include <functional>
#include <unordered_map>

namespace llaminar2::test {

/// Mock model loader for testing without real GGUF files
class MockModelLoader : public IModelLoader {
public:
    /// Builder pattern for configuration
    class Builder {
    public:
        Builder& with_architecture(const std::string& arch);
        Builder& with_hidden_size(int size);
        Builder& with_num_layers(int layers);
        Builder& with_num_heads(int heads);
        Builder& with_num_kv_heads(int kv_heads);
        Builder& with_vocab_size(int vocab);
        Builder& with_intermediate_size(int size);
        Builder& with_quantization(const std::string& quant);

        /// Add a weight with specific shape and type
        Builder& add_weight(const std::string& name,
                            std::vector<size_t> shape,
                            DType dtype = DType::FP32);

        /// Add standard Qwen2 layer weights
        Builder& add_qwen2_layer_weights(int layer_index, DType dtype);

        /// Add all weights for a complete Qwen2 model
        Builder& add_qwen2_all_weights(DType layer_dtype = DType::Q4_0);

        std::unique_ptr<MockModelLoader> build();

    private:
        ModelHyperparams params_;
        std::vector<WeightDescriptor> descriptors_;
    };

    // === Presets ===
    static std::unique_ptr<MockModelLoader> qwen2_0_5b_fp32();
    static std::unique_ptr<MockModelLoader> qwen2_0_5b_q4_0();
    static std::unique_ptr<MockModelLoader> qwen2_7b_q4_0();
    static std::unique_ptr<MockModelLoader> tiny_test_model();

    // === IModelLoader Implementation ===
    const ModelHyperparams& hyperparams() const override;
    const std::vector<WeightDescriptor>& weight_descriptors() const override;
    bool has_weight(const std::string& name) const override;

    std::unique_ptr<TensorBase> load_weight(
        const std::string& name) const override;
    std::unique_ptr<TensorBase> load_weight_shard(
        const std::string& name,
        int shard_index,
        int num_shards,
        ShardDimension dim) const override;

    std::vector<int> tokenize(const std::string& text) const override;
    std::string detokenize(const std::vector<int>& tokens) const override;
    int bos_token() const override { return 1; }
    int eos_token() const override { return 2; }

    // === Test Configuration ===
    /// Override weight generation (default: random values)
    using WeightGenerator = std::function<void(float*, size_t)>;
    void set_weight_generator(WeightGenerator gen);

    /// Inject specific weight data
    void inject_weight_data(const std::string& name,
                            std::vector<float> data);

    /// Control loading behavior
    void set_load_delay_ms(int delay);  // Simulate slow loading
    void fail_on_load(const std::string& name);  // Inject failure

private:
    friend class Builder;
    MockModelLoader() = default;

    ModelHyperparams params_;
    std::vector<WeightDescriptor> descriptors_;
    std::unordered_map<std::string, std::vector<float>> injected_weights_;
    WeightGenerator weight_generator_;
    int load_delay_ms_ = 0;
    std::unordered_set<std::string> fail_weights_;
};

} // namespace llaminar2::test
```

### 2.4 MockWeightManager Implementation

**Location:** `tests/v2/mocks/MockWeightManager.h`

```cpp
#pragma once
#include "interfaces/IWeightManager.h"
#include <unordered_map>

namespace llaminar2::test {

/// Mock weight manager for testing weight distribution logic
class MockWeightManager : public IWeightManager {
public:
    explicit MockWeightManager(std::shared_ptr<IModelLoader> loader);

    // === IWeightManager Implementation ===
    TensorBase* get_weight(const std::string& name) const override;
    bool has_weight(const std::string& name) const override;
    std::vector<std::string> weight_names() const override;

    const WeightPlacement& placement(const std::string& name) const override;
    DeviceId device_for_weight(const std::string& name) const override;
    bool is_weight_local(const std::string& name) const override;

    size_t total_weight_memory() const override;
    size_t local_weight_memory() const override;
    std::unordered_map<DeviceId, size_t> memory_per_device() const override;

    bool is_streaming_enabled() const override;
    void prefetch_layer_weights(int layer_index) override;
    void evict_layer_weights(int layer_index) override;

    // === Test Configuration ===
    /// Manually set placement for a weight
    void set_placement(const std::string& name, WeightPlacement placement);

    /// Mark weights as local/remote
    void set_local_weights(const std::vector<std::string>& names);
    void set_remote_weights(const std::vector<std::string>& names);

    /// Enable/configure streaming simulation
    void enable_streaming(size_t cache_size_mb);
    void set_prefetch_callback(std::function<void(int)> callback);
    void set_evict_callback(std::function<void(int)> callback);

    // === Inspection ===
    std::vector<int> prefetched_layers() const;
    std::vector<int> evicted_layers() const;

private:
    std::shared_ptr<IModelLoader> loader_;
    std::unordered_map<std::string, std::unique_ptr<TensorBase>> weights_;
    std::unordered_map<std::string, WeightPlacement> placements_;
    std::unordered_set<std::string> local_weights_;
    bool streaming_enabled_ = false;
    size_t streaming_cache_mb_ = 0;
    std::vector<int> prefetched_;
    std::vector<int> evicted_;
    std::function<void(int)> prefetch_callback_;
    std::function<void(int)> evict_callback_;
};

} // namespace llaminar2::test
```

### 2.5 Files to Create/Modify

| Action | File | Description |
|--------|------|-------------|
| CREATE | `src/v2/interfaces/IModelLoader.h` | Interface definition |
| CREATE | `src/v2/interfaces/IWeightManager.h` | Interface definition |
| CREATE | `tests/v2/mocks/MockModelLoader.h` | Mock header |
| CREATE | `tests/v2/mocks/MockModelLoader.cpp` | Mock implementation |
| CREATE | `tests/v2/mocks/MockWeightManager.h` | Mock header |
| CREATE | `tests/v2/mocks/MockWeightManager.cpp` | Mock implementation |
| MODIFY | `src/v2/loaders/ModelLoader.h` | Inherit from IModelLoader |
| MODIFY | `src/v2/loaders/WeightManager.h` | Inherit from IWeightManager |

---

## Phase 3: Execution Infrastructure Interfaces

**Goal:** Enable testing of work distribution, buffer management, and collective operations.

### 3.1 ICollectiveContext Interface Design

**Location:** `src/v2/interfaces/ICollectiveContext.h`

```cpp
#pragma once
#include <memory>
#include <vector>
#include "interfaces/IMPIContext.h"
#include "tensors/TensorBase.h"

namespace llaminar2 {

/// High-level collective operations on tensors
class ICollectiveContext {
public:
    virtual ~ICollectiveContext() = default;

    // === Tensor Collectives ===
    virtual void allreduce_sum(TensorBase* tensor) = 0;
    virtual void allreduce_sum(TensorBase* input, TensorBase* output) = 0;
    virtual void allgather(TensorBase* local, TensorBase* global) = 0;
    virtual void reduce_scatter(TensorBase* input, TensorBase* output) = 0;

    // === Synchronization ===
    virtual void barrier() = 0;
    virtual void sync_stream(int device_id) = 0;  // For GPU streams

    // === Query ===
    virtual int rank() const = 0;
    virtual int world_size() const = 0;
    virtual bool is_distributed() const = 0;

    // === Factory ===
    static std::unique_ptr<ICollectiveContext> create(
        std::shared_ptr<IMPIContext> mpi_ctx);
};

} // namespace llaminar2
```

### 3.2 IModelContext Interface Design

**Location:** `src/v2/interfaces/IModelContext.h`

```cpp
#pragma once
#include <memory>
#include "interfaces/IModelLoader.h"
#include "interfaces/IWeightManager.h"
#include "interfaces/IMPIContext.h"
#include "interfaces/IMPITopology.h"

namespace llaminar2 {

/// Facade interface for model access
/// Combines loader, weights, and runtime context
class IModelContext {
public:
    virtual ~IModelContext() = default;

    // === Component Access ===
    virtual const IModelLoader& loader() const = 0;
    virtual IWeightManager& weight_manager() = 0;
    virtual const IWeightManager& weight_manager() const = 0;
    virtual const IMPIContext& mpi_context() const = 0;
    virtual const IMPITopology& topology() const = 0;

    // === Convenience Accessors ===
    virtual const ModelHyperparams& hyperparams() const = 0;
    virtual int num_layers() const = 0;
    virtual int hidden_size() const = 0;
    virtual int num_heads() const = 0;
    virtual int head_dim() const = 0;

    // === Runtime State ===
    virtual bool is_distributed() const = 0;
    virtual int local_rank() const = 0;
    virtual int world_size() const = 0;

    // === Factory ===
    static std::unique_ptr<IModelContext> create(
        std::shared_ptr<IModelLoader> loader,
        std::shared_ptr<IMPIContext> mpi_ctx,
        std::shared_ptr<IMPITopology> topology,
        const WeightPlacementConfig& config);
};

} // namespace llaminar2
```

### 3.3 IWorkDistributor Interface Design

**Location:** `src/v2/interfaces/IWorkDistributor.h`

```cpp
#pragma once
#include <vector>
#include "interfaces/IMPITopology.h"

namespace llaminar2 {

/// Work assignment for a single computation
struct WorkAssignment {
    int rank;
    int start_row;
    int end_row;
    int start_col;
    int end_col;
    DeviceId device;
    bool participates;  // false if rank should skip this work
};

/// Abstract interface for work distribution strategies
class IWorkDistributor {
public:
    virtual ~IWorkDistributor() = default;

    // === Distribution Queries ===
    virtual WorkAssignment get_assignment(
        int rank,
        int total_rows,
        int total_cols) const = 0;

    virtual std::vector<WorkAssignment> get_all_assignments(
        int total_rows,
        int total_cols) const = 0;

    // === Strategy Info ===
    virtual std::string strategy_name() const = 0;
    virtual bool is_row_parallel() const = 0;
    virtual bool is_col_parallel() const = 0;

    // === Factory ===
    static std::unique_ptr<IWorkDistributor> create_row_parallel(
        const IMPITopology& topology);
    static std::unique_ptr<IWorkDistributor> create_col_parallel(
        const IMPITopology& topology);
    static std::unique_ptr<IWorkDistributor> create_2d(
        const IMPITopology& topology,
        int row_ranks,
        int col_ranks);
};

} // namespace llaminar2
```

### 3.4 IGraphBufferManager Interface Design

**Location:** `src/v2/interfaces/IDeviceGraphBufferManager.h`

```cpp
#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include "tensors/TensorBase.h"
#include "devices/DeviceId.h"

namespace llaminar2 {

/// Buffer allocation configuration
struct BufferConfig {
    std::string name;
    std::vector<size_t> shape;
    DType dtype;
    DeviceId device;
    bool persistent;      // Survives across iterations
    bool requires_grad;   // For future gradient support
};

/// Abstract interface for graph buffer management
class IGraphBufferManager {
public:
    virtual ~IGraphBufferManager() = default;

    // === Buffer Allocation ===
    virtual TensorBase* allocate(const BufferConfig& config) = 0;
    virtual TensorBase* get(const std::string& name) = 0;
    virtual bool has(const std::string& name) const = 0;
    virtual void release(const std::string& name) = 0;
    virtual void release_all_non_persistent() = 0;

    // === Memory Info ===
    virtual size_t total_allocated() const = 0;
    virtual size_t peak_allocated() const = 0;
    virtual std::unordered_map<DeviceId, size_t> allocated_per_device() const = 0;

    // === Lifecycle ===
    virtual void begin_iteration() = 0;
    virtual void end_iteration() = 0;

    // === Factory ===
    static std::unique_ptr<IGraphBufferManager> create(
        const IMPITopology& topology);
};

} // namespace llaminar2
```

### 3.5 Files to Create/Modify

| Action | File | Description |
|--------|------|-------------|
| CREATE | `src/v2/interfaces/ICollectiveContext.h` | Interface definition |
| CREATE | `src/v2/interfaces/IModelContext.h` | Interface definition |
| CREATE | `src/v2/interfaces/IWorkDistributor.h` | Interface definition |
| CREATE | `src/v2/interfaces/IDeviceGraphBufferManager.h` | Interface definition |
| CREATE | `tests/v2/mocks/MockCollectiveContext.h` | Mock header |
| CREATE | `tests/v2/mocks/MockCollectiveContext.cpp` | Mock implementation |
| CREATE | `tests/v2/mocks/MockModelContext.h` | Mock header |
| CREATE | `tests/v2/mocks/MockModelContext.cpp` | Mock implementation |
| CREATE | `tests/v2/mocks/MockWorkDistributor.h` | Mock header |
| CREATE | `tests/v2/mocks/MockWorkDistributor.cpp` | Mock implementation |
| CREATE | `tests/v2/mocks/MockDeviceGraphBufferManager.h` | Mock header |
| CREATE | `tests/v2/mocks/MockDeviceGraphBufferManager.cpp` | Mock implementation |

---

## Phase 4: Refactoring Core Components

**Status:** ✅ COMPLETED (January 11, 2026)

### 4.1 GraphOrchestrator Refactoring

**Current State:**
```cpp
class GraphOrchestrator {
public:
    GraphOrchestrator(ModelContext& model_ctx);  // Concrete dependency
private:
    ModelContext& model_ctx_;
};
```

**Target State:** ✅ IMPLEMENTED
```cpp
class GraphOrchestrator {
public:
    // NEW: Interface-based constructor for testing
    struct Dependencies {
        std::shared_ptr<IModelContext> model_ctx;
        std::shared_ptr<IMPITopology> topology = nullptr;
        std::shared_ptr<ICollectiveContext> collective_ctx = nullptr;
    };
    GraphOrchestrator(Dependencies deps, const Qwen2GraphConfig& config,
                      const GraphCacheConfig& cache_config = {});
    
    // EXISTING: Backward-compatible constructors (unchanged)
    GraphOrchestrator(std::shared_ptr<Qwen2Graph> graph_builder,
                      std::shared_ptr<MPIContext> mpi_ctx = nullptr,
                      const GraphCacheConfig& cache_config = {});
    GraphOrchestrator(const Qwen2GraphConfig& graph_config,
                      std::shared_ptr<MPIContext> mpi_ctx = nullptr,
                      const GraphCacheConfig& cache_config = {});
    
private:
    // New injected dependency members
    std::shared_ptr<IModelContext> injected_model_ctx_;
    std::shared_ptr<IMPITopology> injected_topology_;
    std::shared_ptr<ICollectiveContext> injected_collective_ctx_;
};
```

**Files Modified:**
- `src/v2/execution/GraphOrchestrator.h` - Added Dependencies struct, new constructor, private members
- `src/v2/execution/GraphOrchestrator.cpp` - Implemented Dependencies-based constructor

### 4.2 InferenceRunnerFactory Refactoring

**Current State:**
```cpp
class InferenceRunnerFactory {
public:
    static std::unique_ptr<IInferenceRunner> create(
        const std::string& model_path,
        const InferenceConfig& config);
};
```

**Target State:** ✅ IMPLEMENTED
```cpp
// Factory functions in InferenceRunnerFactory.h:

// Production factory (unchanged)
std::unique_ptr<IInferenceRunner> createInferenceRunner(
    std::shared_ptr<ModelContext> model_ctx,
    std::shared_ptr<MPIContext> mpi_ctx,
    DeviceId device,
    const InferenceRunnerConfig& config = {});

// NEW: Testing factory with interface-based dependencies
std::unique_ptr<IInferenceRunner> createTestableInferenceRunner(
    std::shared_ptr<IModelContext> model_ctx,
    DeviceId device,
    const InferenceRunnerConfig& config = {});
```

**Files Modified:**
- `src/v2/execution/InferenceRunnerFactory.h` - Added `createTestableInferenceRunner` declaration
- `src/v2/execution/InferenceRunnerFactory.cpp` - Implemented `createTestableInferenceRunner`

### 4.3 Constructor Injection Pattern

**Recommended Pattern:**

```cpp
// Stage with injected dependencies
class FusedAttentionStage : public ComputeStage {
public:
    struct Params {
        // Required buffers
        TensorBase* Q;
        TensorBase* K;
        TensorBase* V;
        TensorBase* output;
        
        // Injected dependencies (optional, defaults to production)
        std::shared_ptr<ICollectiveContext> collective_ctx = nullptr;
        std::shared_ptr<IKernelFactory> kernel_factory = nullptr;
    };
    
    explicit FusedAttentionStage(Params params);
};

// Usage in production
auto stage = std::make_unique<FusedAttentionStage>(FusedAttentionStage::Params{
    .Q = Q_buffer,
    .K = K_buffer,
    .V = V_buffer,
    .output = output_buffer
    // collective_ctx and kernel_factory use defaults
});

// Usage in tests
auto stage = std::make_unique<FusedAttentionStage>(FusedAttentionStage::Params{
    .Q = mock_Q,
    .K = mock_K,
    .V = mock_V,
    .output = mock_output,
    .collective_ctx = mock_collective_ctx,
    .kernel_factory = mock_kernel_factory
});
```

---

## Phase 5: Integration Test Scenarios

### 5.1 Heterogeneous CPU/GPU Across Ranks

**Test File:** `tests/v2/integration/Test__HeterogeneousExecution.cpp`

```cpp
TEST(Test__HeterogeneousExecution, MixedCpuGpuRanks) {
    // Setup: 4 ranks - 2 with GPU, 2 CPU-only
    auto topology = MockMPITopology::preset_4rank_mixed_cpu_gpu();
    auto loader = MockModelLoader::qwen2_0_5b_q4_0();
    
    // Each rank gets a mock MPI context
    std::vector<std::shared_ptr<MockMPIContext>> contexts(4);
    for (int i = 0; i < 4; ++i) {
        contexts[i] = std::make_shared<MockMPIContext>(
            MockMPIContext::Config{.rank = i, .world_size = 4});
    }
    MockMPIContext::connect_ranks(contexts);
    
    // Test: Weight placement should respect device capabilities
    for (int rank = 0; rank < 4; ++rank) {
        topology->set_local_rank(rank);
        auto weight_mgr = IWeightManager::create(loader, contexts[rank], 
                                                  topology, default_config);
        
        // Verify large weights on GPU ranks, smaller on CPU
        if (topology->local_info().has_gpu) {
            EXPECT_TRUE(weight_mgr->device_for_weight("model.layers.0.attn.q_proj.weight")
                        .is_gpu());
        } else {
            EXPECT_TRUE(weight_mgr->device_for_weight("model.layers.0.attn.q_proj.weight")
                        .is_cpu());
        }
    }
}
```

### 5.2 Work Distribution Across Ranks

**Test File:** `tests/v2/integration/Test__WorkDistribution.cpp`

```cpp
TEST(Test__WorkDistribution, TensorParallelGemm) {
    auto topology = MockMPITopology::preset_2rank_single_gpu();
    auto distributor = IWorkDistributor::create_col_parallel(*topology);
    
    int M = 128;   // Batch size
    int K = 896;   // Hidden size
    int N = 3584;  // Intermediate size
    
    auto assignments = distributor->get_all_assignments(M, N);
    
    ASSERT_EQ(assignments.size(), 2);
    
    // Rank 0 handles columns 0-1791
    EXPECT_EQ(assignments[0].start_col, 0);
    EXPECT_EQ(assignments[0].end_col, N / 2);
    
    // Rank 1 handles columns 1792-3583
    EXPECT_EQ(assignments[1].start_col, N / 2);
    EXPECT_EQ(assignments[1].end_col, N);
    
    // Both handle all rows
    for (const auto& a : assignments) {
        EXPECT_EQ(a.start_row, 0);
        EXPECT_EQ(a.end_row, M);
        EXPECT_TRUE(a.participates);
    }
}
```

### 5.3 Weight Placement with Streaming

**Test File:** `tests/v2/integration/Test__WeightStreaming.cpp`

```cpp
TEST(Test__WeightStreaming, LayerPrefetchAndEviction) {
    auto topology = MockMPITopology::preset_2rank_single_gpu();
    auto loader = MockModelLoader::qwen2_7b_q4_0();
    auto mpi_ctx = std::make_shared<MockMPIContext>(
        MockMPIContext::Config{.rank = 0, .world_size = 2});
    
    // Enable streaming with limited cache
    WeightPlacementConfig config{
        .enable_streaming = true,
        .streaming_cache_mb = 1024,  // 1GB cache for 7B model
        .eviction_policy = EvictionPolicy::LRU,
        .prefetch_depth = 2
    };
    
    auto weight_mgr = IWeightManager::create(loader, mpi_ctx, topology, config);
    
    EXPECT_TRUE(weight_mgr->is_streaming_enabled());
    
    // Simulate layer execution
    std::vector<int> prefetch_order;
    std::vector<int> evict_order;
    
    // Cast to mock for inspection
    auto* mock = dynamic_cast<MockWeightManager*>(weight_mgr.get());
    mock->set_prefetch_callback([&](int layer) { prefetch_order.push_back(layer); });
    mock->set_evict_callback([&](int layer) { evict_order.push_back(layer); });
    
    // Execute layers 0-5
    for (int layer = 0; layer < 6; ++layer) {
        weight_mgr->prefetch_layer_weights(layer + config.prefetch_depth);
        // ... execute layer ...
        if (layer > 1) {
            weight_mgr->evict_layer_weights(layer - 2);
        }
    }
    
    // Verify prefetch was ahead
    EXPECT_EQ(prefetch_order, std::vector<int>({2, 3, 4, 5, 6, 7}));
    EXPECT_EQ(evict_order, std::vector<int>({0, 1, 2, 3}));
}
```

### 5.4 Phase-Aware Execution (Prefill vs Decode)

**Test File:** `tests/v2/integration/Test__PhaseAwareExecution.cpp`

```cpp
TEST(Test__PhaseAwareExecution, PrefillUsesGpuDecodeUsesCpu) {
    // Configure phase-aware execution
    // CPU participates in decode (low-batch), GPU for prefill (high-batch)
    
    auto topology = MockMPITopology::create_heterogeneous({
        {.rank = 0, .devices = {DeviceId::cpu(0), DeviceId::cuda(0)},
         .total_memory_bytes = 16ULL * 1024 * 1024 * 1024,
         .has_gpu = true, .has_cuda = true}
    });
    
    auto loader = MockModelLoader::qwen2_0_5b_q4_0();
    auto mpi_ctx = std::make_shared<MockMPIContext>();
    auto model_ctx = IModelContext::create(loader, mpi_ctx, topology, {});
    
    // Create graph with phase awareness
    auto graph = Qwen2Graph::build(*model_ctx);
    auto executor = DeviceGraphExecutor::create(graph, *model_ctx);
    
    // Prefill with 100 tokens - should use GPU
    auto prefill_config = executor->get_device_config(ExecutionPhase::PREFILL, 100);
    EXPECT_TRUE(prefill_config.primary_device.is_gpu());
    
    // Decode with 1 token - can use CPU
    auto decode_config = executor->get_device_config(ExecutionPhase::DECODE, 1);
    // CPU fallback is acceptable for decode
    EXPECT_TRUE(decode_config.allow_cpu_fallback);
}

TEST(Test__PhaseAwareExecution, BatchSizeThresholds) {
    auto topology = MockMPITopology::preset_2rank_single_gpu();
    
    // Test batch size thresholds for device selection
    struct TestCase {
        ExecutionPhase phase;
        int batch_size;
        bool expect_gpu;
    };
    
    std::vector<TestCase> cases = {
        {ExecutionPhase::PREFILL, 1, true},     // GPU even for batch=1 prefill
        {ExecutionPhase::PREFILL, 100, true},   // GPU for large prefill
        {ExecutionPhase::DECODE, 1, false},     // CPU OK for batch=1 decode
        {ExecutionPhase::DECODE, 8, true},      // GPU for batched decode
    };
    
    for (const auto& tc : cases) {
        auto config = get_device_config(topology, tc.phase, tc.batch_size);
        if (tc.expect_gpu) {
            EXPECT_TRUE(config.primary_device.is_gpu())
                << "Phase=" << static_cast<int>(tc.phase) 
                << " batch=" << tc.batch_size;
        }
    }
}
```

---

## Phase 6: File Organization

### 6.1 Directory Structure

```
src/v2/
├── interfaces/                    # NEW: All interface definitions
│   ├── IMPIContext.h
│   ├── IMPITopology.h
│   ├── IModelLoader.h
│   ├── IWeightManager.h
│   ├── IModelContext.h
│   ├── ICollectiveContext.h
│   ├── IWorkDistributor.h
│   ├── IDeviceGraphBufferManager.h
│   └── IWeightPlacementMap.h
├── utils/
│   ├── MPIContext.h               # MODIFY: Inherit from IMPIContext
│   └── MPITopology.h              # MODIFY: Inherit from IMPITopology
├── loaders/
│   ├── ModelLoader.h              # MODIFY: Inherit from IModelLoader
│   └── WeightManager.h            # MODIFY: Inherit from IWeightManager
└── execution/
    ├── CollectiveContext.h        # MODIFY: Inherit from ICollectiveContext
    └── DeviceGraphBufferManager.h       # MODIFY: Inherit from IGraphBufferManager

tests/v2/
├── mocks/                         # NEW: All mock implementations
│   ├── MockMPIContext.h
│   ├── MockMPIContext.cpp
│   ├── MockMPITopology.h
│   ├── MockMPITopology.cpp
│   ├── MockModelLoader.h
│   ├── MockModelLoader.cpp
│   ├── MockWeightManager.h
│   ├── MockWeightManager.cpp
│   ├── MockModelContext.h
│   ├── MockModelContext.cpp
│   ├── MockCollectiveContext.h
│   ├── MockCollectiveContext.cpp
│   ├── MockWorkDistributor.h
│   ├── MockWorkDistributor.cpp
│   ├── MockDeviceGraphBufferManager.h
│   └── MockDeviceGraphBufferManager.cpp
├── integration/
│   ├── Test__HeterogeneousExecution.cpp
│   ├── Test__WorkDistribution.cpp
│   ├── Test__WeightStreaming.cpp
│   └── Test__PhaseAwareExecution.cpp
└── utils/
    └── TestTensorFactory.h        # EXISTING
```

### 6.2 Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Interface | `I<ClassName>` | `IMPIContext`, `IModelLoader` |
| Mock | `Mock<ClassName>` | `MockMPIContext`, `MockModelLoader` |
| Test File | `Test__<TestedClass>.cpp` | `Test__MockMPIContext.cpp` |
| Test Suite | `TEST(Test__<Class>, <TestName>)` | `TEST(Test__MockMPIContext, TracksBarrierCalls)` |

### 6.3 CMake Integration

**src/v2/CMakeLists.txt additions:**
```cmake
# Interfaces (header-only)
set(INTERFACE_HEADERS
    interfaces/IMPIContext.h
    interfaces/IMPITopology.h
    interfaces/IModelLoader.h
    interfaces/IWeightManager.h
    interfaces/IModelContext.h
    interfaces/ICollectiveContext.h
    interfaces/IWorkDistributor.h
    interfaces/IDeviceGraphBufferManager.h
)

# Add to library target
target_sources(llaminar2_core PUBLIC ${INTERFACE_HEADERS})
```

**tests/v2/CMakeLists.txt additions:**
```cmake
# Mock implementations
set(MOCK_SOURCES
    mocks/MockMPIContext.cpp
    mocks/MockMPITopology.cpp
    mocks/MockModelLoader.cpp
    mocks/MockWeightManager.cpp
    mocks/MockModelContext.cpp
    mocks/MockCollectiveContext.cpp
    mocks/MockWorkDistributor.cpp
    mocks/MockDeviceGraphBufferManager.cpp
)

# Create mock library
add_library(llaminar2_test_mocks STATIC ${MOCK_SOURCES})
target_link_libraries(llaminar2_test_mocks PUBLIC llaminar2_core)
target_include_directories(llaminar2_test_mocks PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Integration tests
add_executable(v2_integration_heterogeneous integration/Test__HeterogeneousExecution.cpp)
target_link_libraries(v2_integration_heterogeneous llaminar2_test_mocks GTest::gtest_main)
add_test(NAME V2_Integration_HeterogeneousExecution COMMAND v2_integration_heterogeneous)
```

---

## Phase 7: Migration Path

### 7.1 Migration Phases

```
Week 1-2: Phase 1 (MPI Interfaces)
├── Create IMPIContext, IMPITopology
├── Create MockMPIContext, MockMPITopology  
├── Update MPIContext to implement IMPIContext
├── Update MPITopology to implement IMPITopology
└── Write unit tests for mocks

Week 3-4: Phase 2 (Model Loading Interfaces)
├── Create IModelLoader, IWeightManager
├── Create MockModelLoader, MockWeightManager
├── Update ModelLoader to implement IModelLoader
├── Update WeightManager to implement IWeightManager
└── Write unit tests for mocks

Week 5-6: Phase 3 (Execution Interfaces)
├── Create remaining interfaces
├── Create remaining mocks
├── Update concrete classes
└── Write unit tests

Week 7-8: Phase 4 (Core Refactoring)
├── Add interface-accepting constructors to GraphOrchestrator
├── Add testing factory to InferenceRunnerFactory
├── Migrate internal code to use interfaces
└── Verify no regressions

Week 9-10: Phase 5 (Integration Tests)
├── Write heterogeneous execution tests
├── Write work distribution tests
├── Write weight streaming tests
├── Write phase-aware execution tests
└── Achieve target coverage
```

### 7.2 Backward Compatibility Strategy

**Principle:** All changes must be non-breaking until the final deprecation phase.

**Pattern: Dual Constructors**
```cpp
class GraphOrchestrator {
public:
    // NEW: Interface-based (preferred)
    explicit GraphOrchestrator(std::shared_ptr<IModelContext> model_ctx);
    
    // LEGACY: Concrete-based (deprecated but still works)
    [[deprecated("Use interface-based constructor")]]
    explicit GraphOrchestrator(ModelContext& model_ctx);
};
```

**Pattern: Factory Overloads**
```cpp
class InferenceRunnerFactory {
public:
    // Production use (unchanged)
    static std::unique_ptr<IInferenceRunner> create(
        const std::string& model_path,
        const InferenceConfig& config);
    
    // Testing use (new)
    static std::unique_ptr<IInferenceRunner> create_testable(
        std::shared_ptr<IModelContext> model_ctx,
        const InferenceConfig& config);
};
```

### 7.3 Verification Gates

Each phase must pass before proceeding:

| Phase | Gate Criteria |
|-------|--------------|
| 1 | All existing MPI tests pass, mock tests achieve 90%+ coverage |
| 2 | All model loading tests pass, can load/mock Qwen2 0.5B |
| 3 | All execution tests pass, mocks cover all interface methods |
| 4 | Full regression suite passes, no performance degradation |
| 5 | Integration tests cover all 4 scenarios, achieve 85%+ coverage |

---

## Appendix A: Interface Method Summary

### A.1 IMPIContext Methods (9 methods)

| Method | Purpose |
|--------|---------|
| `rank()` | Get local rank |
| `world_size()` | Get total ranks |
| `is_root()` | Check if rank 0 |
| `barrier()` | Global synchronization |
| `allreduce_sum_inplace()` | In-place sum reduction |
| `allgather()` | Gather from all ranks |
| `broadcast()` | Broadcast from root |
| `send()` | Point-to-point send |
| `recv()` | Point-to-point receive |

### A.2 IMPITopology Methods (14 methods)

| Method | Purpose |
|--------|---------|
| `world_size()` | Total MPI ranks |
| `local_rank()` | Rank within node |
| `local_size()` | Ranks on this node |
| `node_count()` | Number of nodes |
| `rank_info()` | Device info for rank |
| `local_info()` | Device info for local rank |
| `all_devices()` | All devices in cluster |
| `local_devices()` | Devices on this node |
| `is_homogeneous()` | All ranks identical? |
| `has_any_gpu()` | Any GPU in cluster? |
| `all_have_gpu()` | All ranks have GPU? |
| `min_device_memory()` | Minimum memory |
| `total_device_memory()` | Sum of all memory |
| `should_enable_tensor_parallelism()` | TP recommendation |

### A.3 IModelLoader Methods (10 methods)

| Method | Purpose |
|--------|---------|
| `hyperparams()` | Model hyperparameters |
| `weight_descriptors()` | List of weight metadata |
| `has_weight()` | Check weight exists |
| `load_weight()` | Load full weight |
| `load_weight_shard()` | Load sharded weight |
| `tokenize()` | Text to tokens |
| `detokenize()` | Tokens to text |
| `bos_token()` | Begin-of-sequence token |
| `eos_token()` | End-of-sequence token |

### A.4 IWeightManager Methods (12 methods)

| Method | Purpose |
|--------|---------|
| `get_weight()` | Get loaded weight |
| `has_weight()` | Check weight loaded |
| `weight_names()` | List all weights |
| `placement()` | Get weight placement |
| `device_for_weight()` | Device holding weight |
| `is_weight_local()` | Is weight on this rank? |
| `total_weight_memory()` | Total weight bytes |
| `local_weight_memory()` | Local weight bytes |
| `memory_per_device()` | Memory breakdown |
| `is_streaming_enabled()` | Streaming mode? |
| `prefetch_layer_weights()` | Prefetch layer |
| `evict_layer_weights()` | Evict layer |

---

## Appendix B: Mock Preset Reference

### B.1 MockMPITopology Presets

| Preset | Configuration | Use Case |
|--------|--------------|----------|
| `preset_2rank_cpu_only()` | 2 ranks, CPU only | Basic distributed testing |
| `preset_2rank_single_gpu()` | 2 ranks, 1 GPU each | Standard tensor parallelism |
| `preset_4rank_mixed_cpu_gpu()` | 4 ranks, 2 GPU + 2 CPU | Heterogeneous testing |
| `preset_2rank_multi_gpu()` | 2 ranks, 2 GPUs each | Multi-GPU per rank |

### B.2 MockModelLoader Presets

| Preset | Configuration | Use Case |
|--------|--------------|----------|
| `qwen2_0_5b_fp32()` | 0.5B params, FP32 | Accuracy testing |
| `qwen2_0_5b_q4_0()` | 0.5B params, Q4_0 | Standard inference |
| `qwen2_7b_q4_0()` | 7B params, Q4_0 | Large model testing |
| `tiny_test_model()` | Minimal model | Fast unit tests |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-11 | Copilot | Initial draft |
