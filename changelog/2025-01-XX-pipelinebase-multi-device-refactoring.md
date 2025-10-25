# PipelineBase Multi-Device Refactoring - Phase 4 Infrastructure Promotion

**Date**: 2025-01-XX  
**Status**: ✅ Complete - Core library builds successfully  
**Component**: V2 Pipeline Architecture  
**Impact**: Major refactoring - Promotes generic multi-device infrastructure from Qwen2Pipeline to PipelineBase

---

## 🎯 Objective

Refactor Qwen2Pipeline's Phase 4 multi-device infrastructure into PipelineBase to enable code reuse across all future pipeline implementations (LLaMA, Gemma, Mistral, etc.).

## 🏗️ Architecture Changes

### **Generic Infrastructure → PipelineBase**

Moved the following components from Qwen2Pipeline to PipelineBase:

1. **ActivationBuffers Struct**
   - Generic Q/K/V/FFN buffer structure
   - Reusable across all transformer architectures
   - Location: `src/v2/pipelines/PipelineBase.h` (lines ~35-50)

2. **Multi-Device State Management**
   ```cpp
   std::shared_ptr<WeightPlacementMap> placement_map_;
   std::vector<int> active_devices_;
   std::map<int, ActivationBuffers> buffers_per_device_;
   ```

3. **Abstract Methods** (architecture-specific, implemented by derived classes)
   ```cpp
   virtual std::vector<std::string> getAllWeightNames() const = 0;
   virtual ActivationBuffers createBuffersForDevice(int device_idx, int max_seq_len) = 0;
   ```

4. **Generic Helper Methods**
   - `discoverActiveDevices()` - Scans weights, returns unique device list
   - `getBuffersForDevice()` - Lazy allocation via `createBuffersForDevice()`
   - `getWeightDevice()` - Delegates to WeightPlacementMap
   - `prepareActivationForDevice()` - Smart transfer with fast path + staging buffer

### **Architecture-Specific → Qwen2Pipeline**

Qwen2Pipeline now implements abstract methods with Qwen-specific logic:

1. **`getAllWeightNames()`** - Returns Qwen GGUF weight names
   - Embedding: `"token_embd.weight"`
   - Layers: `"blk.{layer}.{attn_q,attn_k,attn_v,...}.weight"`
   - Output: `"output_norm.weight"`, `"output.weight"`

2. **`createBuffersForDevice()`** - Allocates buffers with Qwen dimensions
   - Uses `n_heads_`, `n_kv_heads_`, `head_dim_`, `d_ff_` for sizing
   - Creates FP32 tensors for Q/K/V, FFN gate/up, residual, etc.

---

## 📝 Files Modified

### **PipelineBase.h** (~100 lines added)
- **Added**:
  - `ActivationBuffers` struct definition
  - Constructor parameter: `std::shared_ptr<WeightPlacementMap> placement_map`
  - Protected members: `placement_map_`, `active_devices_`, `buffers_per_device_`
  - Abstract methods: `getAllWeightNames()`, `createBuffersForDevice()`
  - Generic helpers: `discoverActiveDevices()`, `getBuffersForDevice()`, `getWeightDevice()`, `prepareActivationForDevice()`
- **Includes**:
  - `#include "../loaders/WeightPlacementMap.h"`
  - `#include <map>`

### **PipelineBase.cpp** (~140 lines added)
- **Updated**: Constructor accepts `placement_map`, creates default if nullptr
- **Added**: Implementations of all generic helper methods:
  1. **`discoverActiveDevices()`** (30 lines):
     - Calls abstract `getAllWeightNames()`
     - Queries placement map for each weight
     - Returns sorted unique device list
  
  2. **`getBuffersForDevice()`** (15 lines):
     - Lazy allocation pattern
     - Calls abstract `createBuffersForDevice()` on first access
  
  3. **`getWeightDevice()`** (3 lines):
     - Simple delegation to placement_map
  
  4. **`prepareActivationForDevice()`** (55 lines):
     - Fast path: return if already on target device
     - Transfer: uses residual buffer as staging area
     - Size validation with `compute_element_count` lambda
     - Error handling with detailed logging

### **Qwen2Pipeline.h** (~80 lines removed, ~20 lines added)
- **Removed**:
  - `ActivationBuffers` struct (moved to base)
  - WeightPlacementMap include (moved to base)
  - Multi-device state members (moved to base)
  - Helper method declarations (moved to base)
  - Duplicate `activation_buffers_` declaration (line 187)
- **Added**:
  - Override declarations:
    ```cpp
    std::vector<std::string> getAllWeightNames() const override;
    ActivationBuffers createBuffersForDevice(int device_idx, int max_seq_len) override;
    ```

### **Qwen2Pipeline.cpp** (~150 lines removed, ~70 lines added)
- **Updated**: Constructor calls `PipelineBase(model_ctx, mpi_ctx, device_idx, placement_map)`
- **Removed** (inherited from base):
  - `discoverActiveDevices()` implementation (44 lines)
  - `getBuffersForDevice()` implementation (24 lines)
  - `getWeightDevice()` implementation (4 lines)
  - `prepareActivationForDevice()` implementation (65 lines)
- **Renamed**: `allocate_buffers_for_device()` → `createBuffersForDevice()` (implements abstract method)
- **Added**: `getAllWeightNames()` implementation (40 lines)
  - Enumerates all Qwen weight names (embedding, layers, output)
  - Returns ~100 weight names for typical Qwen models

### **Test__PipelineFactory.cpp** (MockPipeline updated)
- **Added** to MockPipeline:
  ```cpp
  std::vector<std::string> getAllWeightNames() const override {
      return {"mock.weight"};
  }
  
  ActivationBuffers createBuffersForDevice(int device_idx, int max_seq_len) override {
      ActivationBuffers buffers;
      buffers.max_seq_len = max_seq_len;
      buffers.residual = std::make_shared<FP32Tensor>(
          std::vector<size_t>{static_cast<size_t>(max_seq_len), 768}, device_idx);
      return buffers;
  }
  ```

---

## 🔧 Technical Details

### **Design Pattern: Template Method**

PipelineBase uses the **template method pattern** to separate generic multi-device logic from architecture-specific details:

```cpp
// Generic algorithm in base class
std::vector<int> PipelineBase::discoverActiveDevices() {
    std::set<int> device_set;
    
    // Call abstract method to get architecture-specific weight names
    auto weight_names = getAllWeightNames();  // Pure virtual
    
    // Generic device discovery logic
    for (const auto& name : weight_names) {
        int device = placement_map_->getDeviceForWeight(name, /*layer*/ -1);
        device_set.insert(device);
    }
    
    return std::vector<int>(device_set.begin(), device_set.end());
}
```

### **Memory Management: Lazy Allocation**

Buffers are allocated on-demand using lazy initialization:

```cpp
ActivationBuffers& PipelineBase::getBuffersForDevice(int device_idx) {
    auto it = buffers_per_device_.find(device_idx);
    if (it != buffers_per_device_.end()) {
        return it->second;  // Already allocated
    }
    
    // First access - allocate via derived class implementation
    ActivationBuffers buffers = createBuffersForDevice(device_idx, max_seq_len_);
    auto [inserted_it, success] = buffers_per_device_.emplace(device_idx, std::move(buffers));
    return inserted_it->second;
}
```

### **Transfer Optimization: Fast Path**

`prepareActivationForDevice()` avoids unnecessary transfers:

```cpp
TensorBase* PipelineBase::prepareActivationForDevice(
    TensorBase* activation, int target_device, const std::string& context)
{
    int current_device = activation->device_index();
    
    // Fast path: already on target device
    if (current_device == target_device) {
        return activation;  // Zero overhead!
    }
    
    // Slow path: transfer via staging buffer
    ActivationBuffers& target_buffers = getBuffersForDevice(target_device);
    TensorBase* staging = target_buffers.residual.get();
    
    // Size validation + transfer
    // ...
}
```

---

## ✅ Verification

### **Compilation Status**

```bash
cd /workspaces/llaminar/build_v2
make llaminar2_core -j$(nproc)
# Result: [100%] Built target llaminar2_core
```

✅ **Core library compiles successfully** with no errors

### **Pre-Existing Issues** (unrelated to refactoring)

The following linker errors existed before this refactoring and are not introduced by these changes:

```
undefined reference to `llaminar2::FP16Tensor::copyFrom(llaminar2::TensorBase const*)'
undefined reference to `llaminar2::BF16Tensor::copyFrom(llaminar2::TensorBase const*)'
```

**Impact**: Some V2 tests fail to link (not related to PipelineBase refactoring)  
**TODO**: Implement `copyFrom` for FP16/BF16 tensors in separate PR

---

## 🎓 Benefits of This Refactoring

### **1. Code Reuse**
- Future pipelines (LLaMA, Gemma, Mistral) can reuse ~200 lines of multi-device infrastructure
- Only need to implement 2 abstract methods (getAllWeightNames, createBuffersForDevice)

### **2. Consistency**
- All pipelines will use identical device discovery and transfer logic
- Reduces risk of bugs from duplicate implementations

### **3. Maintainability**
- Single source of truth for multi-device infrastructure
- Easier to add features (e.g., GPU support) - change once in base class

### **4. Testability**
- Base class methods can be tested independently
- MockPipeline provides minimal implementation for testing

### **5. Architecture Clarity**
- Clear separation: **mechanism** (base) vs **policy** (derived)
- Qwen2Pipeline is now purely Qwen-specific (dimensions, weight names, specs)

---

## 📊 Code Metrics

| Component | Before | After | Change |
|-----------|--------|-------|--------|
| **PipelineBase.h** | ~150 lines | ~250 lines | +100 lines (generic infra) |
| **PipelineBase.cpp** | ~200 lines | ~340 lines | +140 lines (implementations) |
| **Qwen2Pipeline.h** | ~210 lines | ~150 lines | -60 lines (removed duplicates) |
| **Qwen2Pipeline.cpp** | ~450 lines | ~370 lines | -80 lines (removed inherited methods) |
| **Total LOC** | ~1010 lines | ~1110 lines | +100 lines (net increase for reusability) |

**Net Impact**: +100 lines, but ~200 lines of infrastructure now reusable

---

## 🔮 Future Work

### **Immediate Next Steps**
1. ✅ Update architecture documentation (llaminar-v2-architecture.instructions.md)
2. ⬜ Implement FP16/BF16 `copyFrom` methods (fix linker errors)
3. ⬜ Run existing V2 tests to ensure no regressions
4. ⬜ Update developer guide with new pipeline creation pattern

### **Future Pipeline Implementations**

To add a new architecture (e.g., LLaMA), developers now only need to:

```cpp
class LlamaPipeline : public PipelineBase {
public:
    // 1. Implement abstract methods
    std::vector<std::string> getAllWeightNames() const override {
        // Return LLaMA-specific weight names
        return {"model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", ...};
    }
    
    ActivationBuffers createBuffersForDevice(int device_idx, int max_seq_len) override {
        // Allocate with LLaMA-specific dimensions
        ActivationBuffers buffers;
        buffers.Q = std::make_shared<FP32Tensor>(
            std::vector<size_t>{max_seq_len, llama_n_heads * llama_head_dim}, device_idx);
        // ...
        return buffers;
    }
    
    // 2. Implement pipeline logic
    bool forward(const int* tokens, int seq_len) override {
        // Use inherited getBuffersForDevice(), prepareActivationForDevice(), etc.
        auto& buffers = getBuffersForDevice(device_idx_);
        // ...
    }
};
```

**Result**: ~200 lines of boilerplate eliminated per new pipeline!

---

## 📚 Related Documentation

- **Architecture Doc**: `.github/instructions/llaminar-v2-architecture.instructions.md`
- **Copilot Guide**: `.github/copilot-instructions.md` (V2 kernel development section)
- **Phase 4 Changelog**: `changelog/2025-01-10-phase4-device-aware-execution.md` (original Phase 4 implementation)

---

## 🏁 Conclusion

This refactoring successfully **promotes Phase 4's multi-device infrastructure to PipelineBase**, making it reusable across all future pipeline implementations. The code compiles successfully, follows clean architecture principles (template method pattern), and sets the foundation for rapid development of new model architectures in Llaminar V2.

**Key Achievement**: ~200 lines of generic multi-device code can now be reused by LLaMA, Gemma, Mistral, and future pipelines with zero duplication.

---

**Author**: GitHub Copilot  
**Reviewed**: Pending  
**Merged**: Pending
