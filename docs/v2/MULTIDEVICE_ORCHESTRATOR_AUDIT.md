# RankOrchestrator Class Audit

**Purpose**: Document the EXACT current interface of `RankOrchestrator` to plan minimal, surgical changes for Pipeline Parallelism support.

**Date**: 2026-02-03  
**Files Audited**:
- [RankOrchestrator.h](../../src/v2/execution/local_execution/orchestrators/RankOrchestrator.h) (526 lines)
- [RankOrchestrator.cpp](../../src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp) (1371 lines)
- [IRankOrchestrator.h](../../src/v2/execution/local_execution/orchestrators/IRankOrchestrator.h) (162 lines)
- [IInferenceRunner.h](../../src/v2/execution/local_execution/orchestrators/IInferenceRunner.h) (268 lines)

---

## 1. Inheritance Hierarchy

```
IInferenceRunner (base interface)
    ↳ IRankOrchestrator (adds multi-device methods)
        ↳ RankOrchestrator (concrete implementation)
```

---

## 2. IInferenceRunner Interface (Full)

**File**: [IInferenceRunner.h#L38-L268](../../src/v2/execution/local_execution/orchestrators/IInferenceRunner.h#L38-L268)

### Core Inference API (Pure Virtual)
```cpp
virtual bool forward(const int *tokens, int seq_len) = 0;
virtual const float *logits() const = 0;
virtual int vocab_size() const = 0;
virtual void clear_cache() = 0;
virtual int get_position() const = 0;
virtual ExecutionPath executionPath() const = 0;
virtual const char *architecture() const = 0;
```

### Batch API (Default Implementation)
```cpp
virtual bool forward_batch(const std::vector<std::vector<int>> &token_batches) { return false; }
virtual const float *getLogits(int seq_idx = 0) const { return logits(); }
virtual int batch_size() const { return 1; }
virtual int padded_seq_len() const { return 0; }
virtual const std::vector<int> &sequence_lengths() const { return {}; }
```

### Snapshot API (Default No-op)
```cpp
virtual void enableSnapshotCapture(const std::string &output_dir = "") {}
virtual void disableSnapshotCapture() {}
virtual void clearSnapshots() {}
virtual const float *getSnapshot(const std::string &key, size_t &out_size) const { return nullptr; }
virtual std::vector<std::string> getSnapshotKeys() const { return {}; }
```

### Hidden State API (For Pipeline Parallelism) - **ALREADY EXISTS!**
```cpp
virtual TensorBase* getHiddenState() { return nullptr; }
virtual const TensorBase* getHiddenState() const { return nullptr; }
virtual void setHiddenState(TensorBase* hidden_state) { (void)hidden_state; }
virtual bool hasHiddenStateInput() const { return false; }
virtual void clearHiddenStateInput() {}
```

### Profiling API
```cpp
virtual const GraphExecutorStats *executorStats() const { return nullptr; }
virtual void resetExecutorStats() {}
```

### Orchestration API
```cpp
virtual bool hasPlacementPlan() const { return false; }
virtual const PlacementPlan &getPlacementPlan() const { throw ...; }
```

---

## 3. IRankOrchestrator Interface (Full)

**File**: [IRankOrchestrator.h#L63-L162](../../src/v2/execution/local_execution/orchestrators/IRankOrchestrator.h#L63-L162)

```cpp
class IRankOrchestrator : public IInferenceRunner
{
public:
    // Multi-Device Query API
    virtual int device_count() const = 0;
    virtual IInferenceRunner *deviceRunner(int device_idx) = 0;
    virtual const IInferenceRunner *deviceRunner(int device_idx) const = 0;

    // LOCAL TP Context API
    virtual ILocalTPContext *localTPContext() = 0;
    virtual const ILocalTPContext *localTPContext() const = 0;

    // Device Status API
    virtual bool allDevicesReady() const = 0;
    virtual void synchronizeDevices() = 0;
};
```

---

## 4. RankOrchestrator::Config Structure

**File**: [RankOrchestrator.h#L84-L133](../../src/v2/execution/local_execution/orchestrators/RankOrchestrator.h#L84-L133)

```cpp
struct Config
{
    /// Devices participating in LOCAL TP
    std::vector<GlobalDeviceAddress> devices;

    /// Proportional weights for work distribution (sum to 1.0)
    std::vector<float> weights;

    /// Backend for collective operations (AUTO for auto-detection)
    CollectiveBackendType backend = CollectiveBackendType::AUTO;

    /// Maximum sequence length
    size_t max_seq_len = 4096;

    /// Batch size for inference
    int batch_size = 1;

    /// Activation precision for intermediate buffers
    ActivationPrecision activation_precision = ActivationPrecision::FP32;

    /// KV cache scale factor
    float kv_cache_scale = 1.0f;

    /// Use mapped memory for GPU tensors (zero-copy host access)
    bool use_mapped_memory = false;

    // Methods
    bool validate() const;
    std::vector<float> getNormalizedWeights() const;
};
```

---

## 5. RankOrchestrator Public Methods (Full)

**File**: [RankOrchestrator.h](../../src/v2/execution/local_execution/orchestrators/RankOrchestrator.h)

### Factory Methods
```cpp
static std::unique_ptr<RankOrchestrator> createForTest(
    std::shared_ptr<IModelContext> model_ctx,
    std::vector<std::unique_ptr<DeviceGraphOrchestrator>> device_runners,
    std::unique_ptr<ILocalTPContext> tp_ctx,
    const Config &config);
```

### Constructors
```cpp
// Primary constructor - creates tp_ctx_ and device_runners_ internally
RankOrchestrator(
    std::shared_ptr<IModelContext> model_ctx,
    const Config &config);

// Pre-existing TP context constructor
RankOrchestrator(
    std::shared_ptr<IModelContext> model_ctx,
    std::unique_ptr<ILocalTPContext> tp_ctx,
    const Config &config);

~RankOrchestrator() override;

// Non-copyable, movable
RankOrchestrator(const RankOrchestrator &) = delete;
RankOrchestrator &operator=(const RankOrchestrator &) = delete;
RankOrchestrator(RankOrchestrator &&) noexcept;
RankOrchestrator &operator=(RankOrchestrator &&) noexcept;
```

### IInferenceRunner Implementation
```cpp
bool forward(const int *tokens, int seq_len) override;
const float *logits() const override;
bool forward_batch(const std::vector<std::vector<int>> &token_batches) override;
const float *getLogits(int seq_idx = 0) const override;
int batch_size() const override;
int padded_seq_len() const override;
const std::vector<int> &sequence_lengths() const override;
int vocab_size() const override;
void clear_cache() override;
int get_position() const override;
ExecutionPath executionPath() const override;
const char *architecture() const override;
```

### Snapshot API
```cpp
void enableSnapshotCapture(const std::string &output_dir = "") override;
void disableSnapshotCapture() override;
void clearSnapshots() override;
const float *getSnapshot(const std::string &key, size_t &out_size) const override;
std::vector<std::string> getSnapshotKeys() const override;

// TP-aware snapshot (new capability)
TPSnapshot getTPSnapshot(const std::string &key) const;
std::vector<std::pair<std::string, SnapshotShardingMode>> getSnapshotKeysWithSharding() const;
```

### Profiling API
```cpp
const GraphExecutorStats *executorStats() const override;
void resetExecutorStats() override;
```

### Orchestration API
```cpp
bool hasPlacementPlan() const override;
const PlacementPlan &getPlacementPlan() const override;
```

### IRankOrchestrator Implementation
```cpp
int device_count() const override;
IInferenceRunner *deviceRunner(int device_idx) override;
const IInferenceRunner *deviceRunner(int device_idx) const override;
ILocalTPContext *localTPContext() override;
const ILocalTPContext *localTPContext() const override;
bool allDevicesReady() const override;
void synchronizeDevices() override;
```

---

## 6. RankOrchestrator Member Variables

**File**: [RankOrchestrator.h#L479-L526](../../src/v2/execution/local_execution/orchestrators/RankOrchestrator.h#L479-L526)

```cpp
private:
    /// Model context (shared across device runners)
    std::shared_ptr<IModelContext> model_ctx_;

    /// LOCAL TP context for collective operations
    std::unique_ptr<ILocalTPContext> tp_ctx_;

    /// Per-device inference runners
    std::vector<std::unique_ptr<DeviceGraphOrchestrator>> device_runners_;

    /// Configuration
    Config config_;

    /// Combined logits buffer after AllGather [vocab_size]
    std::unique_ptr<TensorBase> combined_logits_;

    /// Actual size of gathered logits from last gatherLogits() call
    size_t last_gathered_logits_size_ = 0;

    /// Aggregated executor stats (mutable for lazy computation)
    mutable std::unique_ptr<GraphExecutorStats> aggregated_stats_;

    /// Current position in KV cache
    int current_position_ = 0;

    /// Current batch size
    int current_batch_size_ = 1;

    /// Padded sequence length for current batch
    int current_padded_seq_len_ = 0;

    /// Sequence lengths for current batch
    std::vector<int> current_sequence_lengths_;

    /// Flag indicating if stats need re-aggregation
    mutable bool stats_dirty_ = true;
```

---

## 7. Private Methods

```cpp
private:
    // Private constructor for createForTest factory
    RankOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        std::vector<std::unique_ptr<DeviceGraphOrchestrator>> device_runners,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const Config &config);

    void initializeDeviceRunners();
    bool gatherLogits(size_t seq_len);
    void aggregateStats() const;
```

---

## 8. forward() Implementation Analysis

**File**: [RankOrchestrator.cpp#L620-L753](../../src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp#L620-L753)

### Execution Flow

```cpp
bool RankOrchestrator::forward(const int *tokens, int seq_len)
{
    // 1. Launch PARALLEL forward passes on ALL devices
    std::vector<std::future<bool>> futures;
    for (auto &runner : device_runners_) {
        IInferenceRunner *runner_iface = runner.get();
        futures.push_back(std::async(std::launch::async,
            [runner_iface, tokens, seq_len]() -> bool {
                return runner_iface->forward(tokens, seq_len);
            }));
    }

    // 2. Wait for ALL to complete, capture FIRST exception
    bool all_success = true;
    std::exception_ptr first_exception = nullptr;
    for (auto &future : futures) {
        try {
            bool success = future.get();
            if (!success) all_success = false;
        } catch (...) {
            // Smart exception handling for CUDA context cleanup
            if (!first_exception) first_exception = std::current_exception();
        }
    }

    // 3. Re-throw primary exception if any
    if (first_exception) std::rethrow_exception(first_exception);

    // 4. Gather logits from all devices (column-parallel LM head)
    if (all_success) {
        if (!gatherLogits(static_cast<size_t>(seq_len))) {
            all_success = false;
        }
        current_position_ += seq_len;
    }

    return all_success;
}
```

### Key Observations for PP Extension

1. **Parallel Execution**: Uses `std::async` to run all device runners in parallel - this is the TP model
2. **Same Tokens to All**: All devices receive the SAME `tokens` and `seq_len`
3. **Logits Gathering**: After parallel forward, gathers column-parallel logits
4. **No Hidden State**: Currently does NOT propagate hidden states between devices

---

## 9. gatherLogits() Implementation Analysis

**File**: [RankOrchestrator.cpp#L403-L565](../../src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp#L403-L565)

### Logic Flow

```cpp
bool RankOrchestrator::gatherLogits(size_t seq_len)
{
    // Single device: just copy from primary
    if (!tp_ctx_ || device_runners_.size() == 1) {
        memcpy(combined_logits_, device_runners_[0]->logits(), ...);
        return true;
    }

    // Check if column-parallel LM head is enabled
    bool has_column_parallel = false;
    for (auto &runner : device_runners_) {
        if (runner->inferenceState().logits_local) {
            has_column_parallel = true;
            break;
        }
    }

    if (!has_column_parallel) {
        // LM head is replicated, use primary device's full logits
        memcpy(combined_logits_, device_runners_[0]->logits(), ...);
        return true;
    }

    // Column-parallel: gather along vocab dimension (axis=1)
    // Each device has logits_local [seq_len, vocab_local]
    // Output is [seq_len, vocab_total]
    for (size_t row = 0; row < seq_len; ++row) {
        size_t col_offset = 0;
        for (size_t dev = 0; dev < device_data.size(); ++dev) {
            memcpy(output + row*total_vocab + col_offset, 
                   device_data[dev] + row*vocab_local, ...);
            col_offset += vocab_locals[dev];
        }
    }
    return true;
}
```

### Key Observations

1. **Vocab-Dimension Gather**: Concatenates `[seq, vocab_local]` → `[seq, vocab_total]`
2. **Uses `inferenceState().logits_local`**: Device runners expose local logits shard
3. **Row-by-Row Copy**: Manual gather, not using collective backend

---

## 10. initializeDeviceRunners() Implementation Analysis

**File**: [RankOrchestrator.cpp#L206-L400](../../src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp#L206-L400)

### Steps

1. **Build TensorParallelConfig for LOCAL TP weight sharding**
   ```cpp
   auto tp_config = TensorParallelConfig::fromLocalTPContext(*tp_ctx_, n_heads, ...);
   weight_mgr->setTensorParallelConfig(tp_config);
   ```

2. **Pre-reserve collective temp buffer**
   ```cpp
   size_t buffer_bytes = activationPrecisionBufferBytes(max_seq_len * hidden_size, ...);
   tp_ctx_->reserveTempBufferBytes(buffer_bytes * 1.1);
   ```

3. **Pre-load weights for all devices** (avoids race conditions)
   ```cpp
   weight_mgr->preloadForDevices(device_ids);
   ```

4. **Create per-device runners**
   ```cpp
   for (auto &device_addr : devices) {
       InferenceRunnerConfig runner_config;
       runner_config.local_tp_ctx = tp_ctx_.get();
       runner_config.local_tp_device_index = device_idx;
       
       auto runner = createTestableInferenceRunner(model_ctx_, device_id, runner_config);
       device_runners_.push_back(dynamic_cast<DeviceGraphOrchestrator*>(runner));
   }
   ```

5. **Allocate combined logits buffer**
   ```cpp
   combined_logits_ = std::make_unique<FP32Tensor>({max_tokens, vocab_size});
   ```

---

## 11. Constructor Logic

**File**: [RankOrchestrator.cpp#L116-L195](../../src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp#L116-L195)

### Primary Constructor

```cpp
RankOrchestrator(model_ctx, config)
{
    // 1. Validate config
    config_.validate();

    // 2. Create LOCAL TP context
    tp_ctx_ = createLocalTPContext(
        config_.devices,
        config_.getNormalizedWeights(),
        config_.backend);

    // 3. Initialize device runners
    initializeDeviceRunners();
}
```

### With Pre-existing TP Context

```cpp
RankOrchestrator(model_ctx, tp_ctx, config)
{
    // Skip creating tp_ctx_, use provided one
    tp_ctx_ = std::move(tp_ctx);
    initializeDeviceRunners();
}
```

---

## 12. Key Findings for PP Extension

### Already Exists in IInferenceRunner (Lines 192-223)
```cpp
// Hidden State API (for Pipeline Parallelism)
virtual TensorBase* getHiddenState() { return nullptr; }
virtual const TensorBase* getHiddenState() const { return nullptr; }
virtual void setHiddenState(TensorBase* hidden_state) { (void)hidden_state; }
virtual bool hasHiddenStateInput() const { return false; }
virtual void clearHiddenStateInput() {}
```

**These methods exist but have DEFAULT NO-OP implementations in IInferenceRunner!**

### What RankOrchestrator DOES NOT Override

- `getHiddenState()` - uses default `return nullptr`
- `setHiddenState()` - uses default no-op
- `hasHiddenStateInput()` - uses default `return false`
- `clearHiddenStateInput()` - uses default no-op

### Config Structure Gaps for PP

Current `Config` has:
- `devices` - for TP devices only
- `weights` - for proportional TP only
- No layer range specification
- No PP stage identification
- No inter-stage communication config

---

## 13. Summary: Minimal Changes for PP Support

### Option A: Extend RankOrchestrator

Add to `Config`:
```cpp
// PP configuration
int pp_stage_id = 0;                    // Pipeline stage index
int pp_total_stages = 1;                // Total PP stages
std::pair<int, int> layer_range = {0, -1}; // -1 = all layers
bool has_embedding = true;              // First stage
bool has_lm_head = true;                // Last stage
```

Override hidden state methods:
```cpp
TensorBase* getHiddenState() override;
void setHiddenState(TensorBase* hidden_state) override;
bool hasHiddenStateInput() const override;
void clearHiddenStateInput() override;
```

### Option B: Create PipelineParallelOrchestrator

New class that:
- Owns multiple `RankOrchestrator` instances (one per PP stage)
- Coordinates hidden state transfer between stages
- Manages inter-stage communication (P2P or memcpy)

### Option C: Unified PPTPOrchestrator

Single class supporting both:
- TP within stage (existing parallel execution)
- PP across stages (sequential with hidden state transfer)

---

## 14. File Locations Quick Reference

| Component | File | Line Range |
|-----------|------|------------|
| IInferenceRunner interface | `orchestrators/IInferenceRunner.h` | 38-268 |
| Hidden State API | `orchestrators/IInferenceRunner.h` | 192-223 |
| IRankOrchestrator | `orchestrators/IRankOrchestrator.h` | 63-162 |
| RankOrchestrator::Config | `orchestrators/RankOrchestrator.h` | 84-133 |
| RankOrchestrator class | `orchestrators/RankOrchestrator.h` | 75-526 |
| forward() implementation | `orchestrators/RankOrchestrator.cpp` | 620-753 |
| gatherLogits() | `orchestrators/RankOrchestrator.cpp` | 403-565 |
| initializeDeviceRunners() | `orchestrators/RankOrchestrator.cpp` | 206-400 |
| Constructor | `orchestrators/RankOrchestrator.cpp` | 116-195 |
