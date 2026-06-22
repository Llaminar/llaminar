# V2 Snapshot Framework - Implementation Roadmap

**Status**: Design Complete → Ready for Implementation  
**Estimated Effort**: ~3-4 days (5 phases)  
**Priority**: High (enables V2 parity testing)

---

## Overview

This roadmap breaks down the V2 snapshot framework implementation into 5 incremental phases, each deliverable and testable independently.

**Dependencies**:
- ✅ V2 tensor system (`src/v2/tensors/Tensors.h`)
- ✅ V2 pipeline infrastructure (`src/v2/pipelines/PipelineBase`)
- ✅ V1 parity test framework (reference implementation)
- ✅ Python reference scripts (`python/reference/`)

---

## Phase 1: Core Infrastructure (Day 1)

**Goal**: Create snapshot capture macros and data structures.

### Files to Create

1. **`src/v2/utils/SnapshotCapture.h`** (~300 lines)
   ```cpp
   namespace llaminar2::snapshot {
       enum class PipelineStage { ... };
       struct SnapshotData { ... };
       std::string formatStageName(...);
       bool isStageEnabled(...);
   }
   
   #ifndef NDEBUG
   #define LLAMINAR_SNAPSHOT(...)
   #define LLAMINAR_SNAPSHOT_BATCH(...)
   #define LLAMINAR_SNAPSHOT_MPI(...)
   #else
   #define LLAMINAR_SNAPSHOT(...) ((void)0)
   // ...
   #endif
   ```

2. **`src/v2/utils/SnapshotCapture.cpp`** (~150 lines)
   - Implement `formatStageName()`
   - Implement `isStageEnabled()` with env var parsing
   - Implement `SnapshotData::save()` (NPY format)
   - Implement `SnapshotData::load()` (NPY format)

3. **Update `src/v2/tensors/Tensors.h`** (add method to TensorBase)
   ```cpp
   class TensorBase {
   #ifndef NDEBUG
       virtual snapshot::SnapshotData captureSnapshot() const;
   #endif
   };
   ```

4. **Update `src/v2/CMakeLists.txt`**
   ```cmake
   add_library(llaminar2_core STATIC
       # ... existing files ...
       utils/SnapshotCapture.cpp
   )
   ```

### Testing

Create **`tests/v2/unit/Test__SnapshotCapture.cpp`**:
```cpp
TEST(SnapshotCapture, FormatStageName) {
    auto name = formatStageName(PipelineStage::LAYER_ATTN_NORM, 5, 0, -1);
    EXPECT_EQ(name, "token_0/LAYER_ATTN_NORM_layer5.npy");
}

TEST(SnapshotCapture, IsStageEnabled) {
    setenv("LLAMINAR_SNAPSHOT_STAGES", "attention", 1);
    EXPECT_TRUE(isStageEnabled(PipelineStage::ATTN_Q_PROJECTION));
    EXPECT_FALSE(isStageEnabled(PipelineStage::FFN_GATE));
}

TEST(SnapshotCapture, TensorSnapshot) {
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 3}, 0);
    auto snap = tensor->captureSnapshot();
    EXPECT_EQ(snap.shape, (std::vector<size_t>{2, 3}));
    EXPECT_EQ(snap.native_type, TensorType::FP32);
}

TEST(SnapshotCapture, SaveLoadNPY) {
    SnapshotData snap;
    snap.shape = {2, 3};
    snap.data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    snap.save("/tmp/test.npy");
    
    auto loaded = SnapshotData::load("/tmp/test.npy");
    EXPECT_EQ(loaded.shape, snap.shape);
    EXPECT_EQ(loaded.data, snap.data);
}
```

**Run**:
```bash
cmake --build build_v2 --target v2_test_snapshot_capture
./build_v2/tests/v2/v2_test_snapshot_capture
```

**Deliverable**: Basic snapshot capture compiles, macros work, NPY I/O works.

---

## Phase 2: Snapshot Manager (Day 2)

**Goal**: Per-pipeline snapshot storage with thread-safety.

### Files to Create

1. **`src/v2/utils/PipelineSnapshotManager.h`** (~150 lines)
   ```cpp
   class PipelineSnapshotManager {
   public:
       void addSnapshot(const SnapshotData& snap);
       const SnapshotData* getSnapshot(const std::string& name) const;
       std::vector<SnapshotData> getAllSnapshots() const;
       int saveAll(const std::string& dir) const;
       int loadAll(const std::string& dir);
       void clear();
       size_t size() const;
   private:
       mutable std::mutex mutex_;
       std::unordered_map<std::string, SnapshotData> snapshots_;
   };
   ```

2. **`src/v2/utils/PipelineSnapshotManager.cpp`** (~250 lines)
   - Implement all methods with proper locking
   - Use `std::filesystem` for directory creation
   - Batch NPY save/load

3. **Update `src/v2/pipelines/PipelineBase.h`**
   ```cpp
   class PipelineBase {
   protected:
       std::shared_ptr<snapshot::PipelineSnapshotManager> snapshot_mgr_;
   public:
       void enableSnapshots();
       std::shared_ptr<snapshot::PipelineSnapshotManager> getSnapshotManager() const;
   };
   ```

4. **Update `src/v2/pipelines/PipelineBase.cpp`**
   ```cpp
   void PipelineBase::enableSnapshots() {
   #ifndef NDEBUG
       if (!snapshot_mgr_) {
           snapshot_mgr_ = std::make_shared<snapshot::PipelineSnapshotManager>();
           LOG_INFO("Snapshot capture enabled");
       }
   #endif
   }
   ```

### Testing

Create **`tests/v2/unit/Test__PipelineSnapshotManager.cpp`**:
```cpp
TEST(PipelineSnapshotManager, AddRetrieve) {
    PipelineSnapshotManager mgr;
    
    SnapshotData snap;
    snap.stage_name = "token_0/EMBEDDING.npy";
    snap.shape = {100, 896};
    snap.data.resize(100 * 896, 1.0f);
    
    mgr.addSnapshot(snap);
    
    auto retrieved = mgr.getSnapshot("token_0/EMBEDDING.npy");
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->shape, snap.shape);
}

TEST(PipelineSnapshotManager, SaveLoadAll) {
    PipelineSnapshotManager mgr1;
    // Add 10 snapshots
    for (int i = 0; i < 10; ++i) {
        SnapshotData snap;
        snap.stage_name = "token_0/layer" + std::to_string(i) + ".npy";
        snap.shape = {10, 10};
        snap.data.resize(100, static_cast<float>(i));
        mgr1.addSnapshot(snap);
    }
    
    int saved = mgr1.saveAll("/tmp/snapshots_test");
    EXPECT_EQ(saved, 10);
    
    PipelineSnapshotManager mgr2;
    int loaded = mgr2.loadAll("/tmp/snapshots_test");
    EXPECT_EQ(loaded, 10);
    
    auto snap5 = mgr2.getSnapshot("token_0/layer5.npy");
    EXPECT_EQ(snap5->data[0], 5.0f);
}

TEST(PipelineSnapshotManager, ThreadSafety) {
    PipelineSnapshotManager mgr;
    
    std::vector<std::thread> threads;
    for (int t = 0; t < 10; ++t) {
        threads.emplace_back([&mgr, t]() {
            for (int i = 0; i < 100; ++i) {
                SnapshotData snap;
                snap.stage_name = "thread" + std::to_string(t) + "_snap" + std::to_string(i);
                snap.data = {static_cast<float>(t * 100 + i)};
                mgr.addSnapshot(snap);
            }
        });
    }
    
    for (auto& th : threads) th.join();
    
    EXPECT_EQ(mgr.size(), 1000);
}
```

**Deliverable**: Thread-safe snapshot storage with file I/O.

---

## Phase 3: Pipeline Integration (Day 2-3)

**Goal**: Add snapshot calls to Qwen2Pipeline.

### Files to Modify

1. **`src/v2/pipelines/qwen/Qwen2Pipeline.h`**
   ```cpp
   class Qwen2Pipeline : public PipelineBase {
   private:
       // ... existing members ...
       // snapshot_mgr_ inherited from PipelineBase
   };
   ```

2. **`src/v2/pipelines/qwen/Qwen2Pipeline.cpp`**
   
   Add includes:
   ```cpp
   #include "../../utils/SnapshotCapture.h"
   ```
   
   Update `forward_batch()`:
   ```cpp
   bool Qwen2Pipeline::forward_batch(...) {
       // Embedding
       embedding_batch(...);
       LLAMINAR_SNAPSHOT_MPI(snapshot_mgr_.get(), 
                            snapshot::PipelineStage::GLOBAL_EMBEDDING,
                            -1, current_positions_[0], current_hidden_.get(),
                            mpi_ctx_.get());
       
       // Layers
       for (int layer = 0; layer < n_layers_; ++layer) {
           LLAMINAR_SNAPSHOT_MPI(snapshot_mgr_.get(),
                                snapshot::PipelineStage::LAYER_INPUT,
                                layer, current_positions_[0], current_hidden_.get(),
                                mpi_ctx_.get());
           transformer_layer(layer, ...);
       }
       
       // Final norm + LM head
       // ...
       LLAMINAR_SNAPSHOT_MPI(snapshot_mgr_.get(),
                            snapshot::PipelineStage::GLOBAL_FINAL_NORM,
                            -1, current_positions_[0], current_hidden_.get(),
                            mpi_ctx_.get());
       // ...
   }
   ```
   
   Update `attention_block()`:
   ```cpp
   bool Qwen2Pipeline::attention_block(...) {
       // RMSNorm
       norm_kernel->apply(...);
       LLAMINAR_SNAPSHOT_MPI(snapshot_mgr_.get(),
                            snapshot::PipelineStage::LAYER_ATTN_NORM,
                            layer_idx, current_positions_[0], normalized_hidden.get(),
                            mpi_ctx_.get());
       
       // Q/K/V projections (with debug substages)
       q_gemm->execute(...);
       LLAMINAR_SNAPSHOT_MPI(snapshot_mgr_.get(),
                            snapshot::PipelineStage::ATTN_Q_PROJECTION,
                            layer_idx, current_positions_[0], buffers.Q.get(),
                            mpi_ctx_.get());
       
       // ... similar for K, V, RoPE, scores, weights, context ...
       
       // Output projection
       o_gemm->execute(...);
       LLAMINAR_SNAPSHOT_MPI(snapshot_mgr_.get(),
                            snapshot::PipelineStage::LAYER_ATTN_OUTPUT,
                            layer_idx, current_positions_[0], buffers.attn_proj.get(),
                            mpi_ctx_.get());
       
       // Residual
       // ...
       LLAMINAR_SNAPSHOT_MPI(snapshot_mgr_.get(),
                            snapshot::PipelineStage::LAYER_ATTN_RESIDUAL,
                            layer_idx, current_positions_[0], current_hidden_.get(),
                            mpi_ctx_.get());
   }
   ```
   
   Update `ffn_block()`:
   ```cpp
   bool Qwen2Pipeline::ffn_block(...) {
       // RMSNorm
       LLAMINAR_SNAPSHOT_MPI(..., PipelineStage::LAYER_FFN_NORM, ...);
       
       // Gate/Up/SwiGLU/Down with debug substages
       LLAMINAR_SNAPSHOT_MPI(..., PipelineStage::FFN_GATE, ...);
       LLAMINAR_SNAPSHOT_MPI(..., PipelineStage::FFN_UP, ...);
       LLAMINAR_SNAPSHOT_MPI(..., PipelineStage::FFN_SWIGLU, ...);
       LLAMINAR_SNAPSHOT_MPI(..., PipelineStage::LAYER_FFN_OUTPUT, ...);
       
       // Residual
       LLAMINAR_SNAPSHOT_MPI(..., PipelineStage::LAYER_FFN_RESIDUAL, ...);
   }
   ```

### Testing

Create **`tests/v2/integration/Test__Qwen2_SnapshotIntegration.cpp`**:
```cpp
TEST(Qwen2SnapshotIntegration, CaptureBasicStages) {
    auto mpi_ctx = std::make_shared<MPIContext>(MPI_COMM_WORLD);
    auto model_ctx = std::make_shared<ModelContext>("models/qwen2.5-0.5b-instruct-q4_0.gguf");
    
    PipelineConfig config;
    config.max_seq_len = 128;
    config.batch_size = 1;
    
    auto pipeline = std::make_unique<Qwen2Pipeline>(
        model_ctx, mpi_ctx, 0, nullptr, config, 1);
    
    // Enable snapshots
    pipeline->enableSnapshots();
    
    // Run inference
    std::vector<int> tokens = {1234, 5678, 91011};
    ASSERT_TRUE(pipeline->forward(tokens.data(), tokens.size()));
    
    // Check snapshots captured
    auto mgr = pipeline->getSnapshotManager();
    EXPECT_GT(mgr->size(), 0);
    
    // Verify key stages present
    EXPECT_NE(mgr->getSnapshot("token_0/GLOBAL_EMBEDDING.npy"), nullptr);
    EXPECT_NE(mgr->getSnapshot("token_0/LAYER_ATTN_NORM_layer0.npy"), nullptr);
    EXPECT_NE(mgr->getSnapshot("token_0/GLOBAL_FINAL_NORM.npy"), nullptr);
    EXPECT_NE(mgr->getSnapshot("token_0/GLOBAL_LM_HEAD.npy"), nullptr);
    
    // Save to disk
    int saved = mgr->saveAll("llaminar_snapshots_v2_test");
    EXPECT_GT(saved, 0);
}

TEST(Qwen2SnapshotIntegration, StageFiltering) {
    setenv("LLAMINAR_SNAPSHOT_STAGES", "attention", 1);
    
    auto pipeline = createQwen2Pipeline();
    pipeline->enableSnapshots();
    
    std::vector<int> tokens = {1234};
    pipeline->forward(tokens.data(), 1);
    
    auto mgr = pipeline->getSnapshotManager();
    
    // Should have attention stages
    EXPECT_NE(mgr->getSnapshot("token_0/ATTN_Q_PROJECTION_layer0.npy"), nullptr);
    
    // Should NOT have FFN stages
    EXPECT_EQ(mgr->getSnapshot("token_0/FFN_GATE_layer0.npy"), nullptr);
}
```

**Deliverable**: Qwen2Pipeline captures snapshots during inference.

---

## Phase 4: Comparison Utilities (Day 3)

**Goal**: Create tools for comparing Llaminar vs PyTorch snapshots.

### Files to Create

1. **`src/v2/utils/SnapshotComparator.h`** (~200 lines)
   ```cpp
   struct ComparisonMetrics { ... };
   struct VarianceThreshold { ... };
   
   class SnapshotComparator {
   public:
       static ComparisonMetrics compare(...);
       static VarianceThreshold computeVarianceThreshold(...);
       static std::unordered_map<std::string, SnapshotData> loadPyTorchSnapshots(...);
   };
   ```

2. **`src/v2/utils/SnapshotComparator.cpp`** (~400 lines)
   - Implement `compare()` with L2, max_abs, etc.
   - Implement variance threshold computation
   - Implement PyTorch snapshot loading
   - Implement batch comparison helper

### Testing

Create **`tests/v2/unit/Test__SnapshotComparator.cpp`**:
```cpp
TEST(SnapshotComparator, BasicComparison) {
    SnapshotData snap1, snap2;
    snap1.shape = {10, 10};
    snap1.data.resize(100, 1.0f);
    
    snap2.shape = {10, 10};
    snap2.data.resize(100, 1.001f);
    
    auto metrics = SnapshotComparator::compare(snap1, snap2, nullptr);
    
    EXPECT_NEAR(metrics.max_abs_diff, 0.001f, 1e-6f);
    EXPECT_LT(metrics.rel_l2_norm, 0.01f);
}

TEST(SnapshotComparator, VarianceThreshold) {
    std::vector<SnapshotData> runs;
    for (int i = 0; i < 3; ++i) {
        SnapshotData snap;
        snap.shape = {100};
        snap.data.resize(100);
        // Add small variance
        for (size_t j = 0; j < 100; ++j) {
            snap.data[j] = 1.0f + (i * 0.0001f);
        }
        runs.push_back(snap);
    }
    
    auto threshold = SnapshotComparator::computeVarianceThreshold(runs);
    
    EXPECT_GT(threshold.variance, 0.0f);
    EXPECT_GT(threshold.threshold, threshold.variance);
}
```

**Deliverable**: Comparison utilities working with proper metrics.

---

## Phase 5: Parity Test (Day 4)

**Goal**: Complete end-to-end parity test with PyTorch.

### Python Scripts

1. **`python/reference/v2_snapshot_adapter.py`** (~200 lines)
   - Reuse V1's `QwenReference` class
   - Map V2 stage names to V1 equivalents
   - Generate snapshots in V2 format

2. **`python/reference/generate_v2_variance_thresholds.py`** (~150 lines)
   - Run PyTorch 3 times
   - Compute variance per stage
   - Save thresholds JSON

### C++ Test

Create **`tests/v2/parity/Test__Qwen2_Parity.cpp`** (~400 lines):
```cpp
TEST(Qwen2Parity, PrefillParity) {
    // 1. Generate PyTorch snapshots + thresholds
    std::string model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
    std::string snapshot_dir = "pytorch_snapshots_v2";
    
    LOG_INFO("Generating PyTorch reference snapshots...");
    int ret = std::system(
        "python3 python/reference/generate_v2_variance_thresholds.py "
        "--model models/qwen2.5-0.5b-instruct-q4_0.gguf "
        "--output-dir pytorch_snapshots_v2 "
        "--n-runs 3 "
        "--tokens 1234 5678 91011");
    ASSERT_EQ(ret, 0) << "Failed to generate PyTorch snapshots";
    
    // 2. Load variance thresholds
    auto thresholds = loadVarianceThresholds(snapshot_dir + "/variance_thresholds.json");
    
    // 3. Run Llaminar
    auto mpi_ctx = std::make_shared<MPIContext>(MPI_COMM_WORLD);
    auto model_ctx = std::make_shared<ModelContext>(model_path);
    
    PipelineConfig config;
    config.max_seq_len = 2048;
    config.batch_size = 1;
    
    auto pipeline = std::make_unique<Qwen2Pipeline>(
        model_ctx, mpi_ctx, 0, nullptr, config, 1);
    
    pipeline->enableSnapshots();
    
    std::vector<int> tokens = {1234, 5678, 91011};
    ASSERT_TRUE(pipeline->forward(tokens.data(), tokens.size()));
    
    // 4. Save Llaminar snapshots
    pipeline->getSnapshotManager()->saveAll("llaminar_snapshots_v2");
    
    // 5. Load PyTorch snapshots
    auto pytorch_snapshots = SnapshotComparator::loadPyTorchSnapshots(snapshot_dir);
    
    // 6. Compare all stages
    auto llaminar_snapshots = pipeline->getSnapshotManager()->getAllSnapshots();
    
    int passed = 0, failed = 0;
    std::vector<std::string> failed_stages;
    
    for (const auto& llaminar_snap : llaminar_snapshots) {
        auto it = pytorch_snapshots.find(llaminar_snap.stage_name);
        if (it == pytorch_snapshots.end()) {
            LOG_WARN("Missing PyTorch snapshot: " << llaminar_snap.stage_name);
            continue;
        }
        
        // Get threshold
        const VarianceThreshold* threshold = nullptr;
        auto thresh_it = thresholds.find(llaminar_snap.stage_name);
        if (thresh_it != thresholds.end()) {
            threshold = &thresh_it->second;
        }
        
        // Compare
        auto metrics = SnapshotComparator::compare(
            llaminar_snap, it->second, threshold);
        
        if (metrics.passed) {
            ++passed;
            LOG_INFO("✓ " << llaminar_snap.stage_name << " PASSED");
            LOG_DEBUG("  max_abs: " << metrics.max_abs_diff 
                     << ", rel_l2: " << metrics.rel_l2_norm);
        } else {
            ++failed;
            failed_stages.push_back(llaminar_snap.stage_name);
            LOG_ERROR("✗ " << llaminar_snap.stage_name << " FAILED");
            LOG_ERROR("  " << metrics.message);
        }
    }
    
    LOG_INFO("=== Parity Test Results ===");
    LOG_INFO("Passed: " << passed);
    LOG_INFO("Failed: " << failed);
    
    if (failed > 0) {
        LOG_ERROR("Failed stages:");
        for (const auto& stage : failed_stages) {
            LOG_ERROR("  - " << stage);
        }
    }
    
    EXPECT_EQ(failed, 0) << "Parity test failed for " << failed << " stages";
}
```

### Running

```bash
# Build debug version
cmake -B build_v2 -DCMAKE_BUILD_TYPE=Debug
cmake --build build_v2 --target v2_test_qwen2_parity

# Run parity test
cd build_v2
mpirun -np 2 ./tests/v2/v2_test_qwen2_parity --gtest_filter="Qwen2Parity.PrefillParity"
```

**Expected Output**:
```
[INFO] Generating PyTorch reference snapshots...
[INFO] Run 1/3: 387 stages captured
[INFO] Run 2/3: 387 stages captured
[INFO] Run 3/3: 387 stages captured
[INFO] Computed variance thresholds for 387 stages
[INFO] Running Llaminar inference...
[INFO] Captured 387 snapshots
[INFO] Comparing snapshots...
✓ token_0/GLOBAL_EMBEDDING.npy PASSED
✓ token_0/LAYER_ATTN_NORM_layer0.npy PASSED
✓ token_0/ATTN_Q_PROJECTION_layer0.npy PASSED
...
✓ token_0/GLOBAL_LM_HEAD.npy PASSED

=== Parity Test Results ===
Passed: 387
Failed: 0
```

**Deliverable**: Complete V2 parity test passing!

---

## Summary Timeline

| Phase | Duration | Deliverable | Status |
|-------|----------|-------------|--------|
| **1. Core Infrastructure** | 1 day | Macros, data structures, NPY I/O | ⬜ Not started |
| **2. Snapshot Manager** | 1 day | Thread-safe storage, file I/O | ⬜ Not started |
| **3. Pipeline Integration** | 1 day | Qwen2Pipeline with snapshots | ⬜ Not started |
| **4. Comparison Utilities** | 0.5 day | Metrics, variance thresholds | ⬜ Not started |
| **5. Parity Test** | 0.5 day | End-to-end PyTorch comparison | ⬜ Not started |
| **Total** | **4 days** | **V2 parity testing complete** | **⬜ Not started** |

---

## Success Criteria

- [x] ✅ Design complete (this document)
- [ ] ⬜ Phase 1: Macros compile, NPY I/O works
- [ ] ⬜ Phase 2: Snapshot manager thread-safe, file I/O works
- [ ] ⬜ Phase 3: Qwen2Pipeline captures 100+ snapshots during inference
- [ ] ⬜ Phase 4: Comparison utilities produce accurate metrics
- [ ] ⬜ Phase 5: Parity test passes with 0 failures (387/387 stages)
- [ ] ⬜ Release build verification: `nm` shows no snapshot symbols

---

## Next Steps

1. **Review design documents**:
   - `docs/v2/projects/2025-10/SNAPSHOT_FRAMEWORK_DESIGN.md` (full spec)
   - `docs/v2/projects/2025-10/SNAPSHOT_QUICK_START.md` (usage guide)
   - `docs/v2/projects/2025-10/SNAPSHOT_ARCHITECTURE.md` (diagrams)
   - `docs/v2/projects/2025-10/SNAPSHOT_IMPLEMENTATION_ROADMAP.md` (this file)

2. **Start Phase 1**:
   - Create `src/v2/utils/SnapshotCapture.h`
   - Create `src/v2/utils/SnapshotCapture.cpp`
   - Update `src/v2/tensors/Tensors.h`
   - Write unit tests

3. **Incremental validation**:
   - Test after each phase
   - Commit working code
   - Document any design changes

---

**Ready to implement!** 🚀
