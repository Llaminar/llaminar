# CRTP Tensor Refactor Prototype

## Overview

This document outlines a refactor of the tensor class hierarchy to use CRTP (Curiously Recurring Template Pattern) for type-safe `data()` / `mutable_data()` accessors while preserving runtime polymorphism.

## Goals

1. **Type-safe accessors**: `tensor.data()` returns the correct native type (float*, uint16_t*, Q8_1Block*, etc.)
2. **Zero overhead**: No virtual dispatch when type is known at compile time
3. **Runtime polymorphism**: Pipelines can still hold `unique_ptr<ITensor>` and work generically
4. **Backward compatibility**: Gradual migration path, old code continues to work
5. **Eliminate switch statements**: Kernels can be templated on tensor type

## Current Architecture Problems

```cpp
// Current TensorBase forces FP32 interface
class TensorBase {
    virtual const float* data() const = 0;      // Forces conversion!
    virtual float* mutable_data() = 0;          // Forces conversion!
};

// BF16Tensor has to provide FP32 interface AND native interface
class BF16Tensor : public TensorBase {
    const float* data() const override {
        throw std::runtime_error("BF16Tensor::data: immutable"); // BAD!
    }
    const uint16_t* bf16_data() const;  // Separate accessor
};
```

## Proposed Architecture

### File: `src/v2/tensors/ITensor.h` (NEW)

```cpp
#pragma once

#include <vector>
#include <cstddef>
#include <memory>

namespace llaminar2 {

/**
 * @brief Tensor type enumeration
 */
enum class TensorType {
    FP32 = 0,
    BF16 = 1,
    FP16 = 2,
    Q4_0 = 3,
    Q4_1 = 4,
    Q5_0 = 5,
    Q5_1 = 6,
    Q8_0 = 7,
    Q8_1 = 8,
    IQ4_NL = 9,
    // ... etc
};

/**
 * @brief Runtime tensor interface for polymorphic access
 * 
 * Use this when you need to hold tensors of unknown type at compile time.
 * For type-safe access, use typed_as<T>() to get a typed reference.
 */
class ITensor {
public:
    virtual ~ITensor() = default;
    
    // =========================================================================
    // Type Information
    // =========================================================================
    
    /// Native storage type of this tensor
    virtual TensorType native_type() const = 0;
    
    /// Shape of the tensor
    virtual const std::vector<size_t>& shape() const = 0;
    
    /// Total number of elements
    virtual size_t numel() const = 0;
    
    /// Size in bytes of the raw data buffer
    virtual size_t size_bytes() const = 0;
    
    /// Device index (-1 for CPU)
    virtual int device_index() const = 0;
    
    // =========================================================================
    // Raw Data Access (type-unsafe, use with caution)
    // =========================================================================
    
    /// Raw pointer to data buffer (caller must know the type)
    virtual const void* raw_data() const = 0;
    
    /// Mutable raw pointer to data buffer
    virtual void* raw_mutable_data() = 0;
    
    // =========================================================================
    // Type-Safe Downcasting
    // =========================================================================
    
    /// Check if this tensor can be cast to type T
    template<typename T>
    bool is() const {
        return native_type() == T::static_type();
    }
    
    /// Cast to typed tensor (asserts on type mismatch)
    template<typename T>
    T& typed_as() {
        assert(is<T>() && "Tensor type mismatch");
        return static_cast<T&>(*this);
    }
    
    template<typename T>
    const T& typed_as() const {
        assert(is<T>() && "Tensor type mismatch");
        return static_cast<const T&>(*this);
    }
    
    /// Try to cast to typed tensor (returns nullptr on mismatch)
    template<typename T>
    T* try_as() {
        return is<T>() ? static_cast<T*>(this) : nullptr;
    }
    
    template<typename T>
    const T* try_as() const {
        return is<T>() ? static_cast<const T*>(this) : nullptr;
    }
    
    // =========================================================================
    // Conversion Methods (optional, for compatibility)
    // =========================================================================
    
    /// Convert tensor data to FP32 (allocates new buffer)
    virtual void to_fp32(float* dst) const = 0;
    
    /// Create an FP32 copy of this tensor
    virtual std::unique_ptr<ITensor> to_fp32_tensor() const = 0;
};

} // namespace llaminar2
```

### File: `src/v2/tensors/TypedTensorBase.h` (NEW)

```cpp
#pragma once

#include "ITensor.h"
#include <vector>
#include <cstring>
#include <stdexcept>

namespace llaminar2 {

/**
 * @brief CRTP base class for typed tensors
 * 
 * Provides type-safe data() and mutable_data() that return the correct type.
 * 
 * @tparam Derived The concrete tensor class (CRTP pattern)
 * @tparam DataType The native storage type (float, uint16_t, Q8_1Block, etc.)
 */
template<typename Derived, typename DataType>
class TypedTensorBase : public ITensor {
public:
    using value_type = DataType;
    using pointer = DataType*;
    using const_pointer = const DataType*;
    
    // =========================================================================
    // Type-Safe Data Access (CRTP - no virtual overhead)
    // =========================================================================
    
    /**
     * @brief Get const pointer to native data
     * @return Pointer to data in native format
     */
    const_pointer data() const {
        return static_cast<const Derived*>(this)->data_impl();
    }
    
    /**
     * @brief Get mutable pointer to native data
     * @return Mutable pointer to data in native format
     */
    pointer mutable_data() {
        return static_cast<Derived*>(this)->mutable_data_impl();
    }
    
    // =========================================================================
    // ITensor Interface Implementation
    // =========================================================================
    
    const void* raw_data() const override {
        return static_cast<const void*>(data());
    }
    
    void* raw_mutable_data() override {
        return static_cast<void*>(mutable_data());
    }
    
    size_t size_bytes() const override {
        return numel() * sizeof(DataType);
    }
    
    // =========================================================================
    // Static Type Information
    // =========================================================================
    
    /// Get the TensorType for this class (must be specialized per Derived)
    static constexpr TensorType static_type();
    
protected:
    TypedTensorBase() = default;
    ~TypedTensorBase() override = default;
    
    // Non-copyable, movable
    TypedTensorBase(const TypedTensorBase&) = delete;
    TypedTensorBase& operator=(const TypedTensorBase&) = delete;
    TypedTensorBase(TypedTensorBase&&) = default;
    TypedTensorBase& operator=(TypedTensorBase&&) = default;
};

} // namespace llaminar2
```

### File: `src/v2/tensors/FP32Tensor.h` (REFACTORED)

```cpp
#pragma once

#include "TypedTensorBase.h"
#include <vector>
#include <cstring>

namespace llaminar2 {

/**
 * @brief FP32 tensor with contiguous float storage
 */
class FP32Tensor : public TypedTensorBase<FP32Tensor, float> {
public:
    // Static type for CRTP
    static constexpr TensorType static_type() { return TensorType::FP32; }
    
    // =========================================================================
    // Constructors
    // =========================================================================
    
    explicit FP32Tensor(const std::vector<size_t>& shape, int device_idx = -1)
        : shape_(shape), device_idx_(device_idx) {
        size_t total = 1;
        for (auto d : shape) total *= d;
        data_.resize(total);
    }
    
    FP32Tensor(const std::vector<size_t>& shape, const float* src_data, int device_idx = -1)
        : FP32Tensor(shape, device_idx) {
        std::memcpy(data_.data(), src_data, data_.size() * sizeof(float));
    }
    
    // =========================================================================
    // CRTP Implementation
    // =========================================================================
    
    const float* data_impl() const { return data_.data(); }
    float* mutable_data_impl() { return data_.data(); }
    
    // =========================================================================
    // ITensor Interface
    // =========================================================================
    
    TensorType native_type() const override { return TensorType::FP32; }
    const std::vector<size_t>& shape() const override { return shape_; }
    size_t numel() const override { return data_.size(); }
    int device_index() const override { return device_idx_; }
    
    void to_fp32(float* dst) const override {
        std::memcpy(dst, data_.data(), data_.size() * sizeof(float));
    }
    
    std::unique_ptr<ITensor> to_fp32_tensor() const override {
        return std::make_unique<FP32Tensor>(shape_, data_.data(), device_idx_);
    }
    
private:
    std::vector<float> data_;
    std::vector<size_t> shape_;
    int device_idx_ = -1;
};

} // namespace llaminar2
```

### File: `src/v2/tensors/BF16Tensor.h` (REFACTORED)

```cpp
#pragma once

#include "TypedTensorBase.h"
#include "SIMDHelpers.h"  // For bf16_to_fp32, fp32_to_bf16
#include <vector>

namespace llaminar2 {

/**
 * @brief BF16 tensor with contiguous uint16_t storage
 */
class BF16Tensor : public TypedTensorBase<BF16Tensor, uint16_t> {
public:
    static constexpr TensorType static_type() { return TensorType::BF16; }
    
    // =========================================================================
    // Constructors
    // =========================================================================
    
    explicit BF16Tensor(const std::vector<size_t>& shape, int device_idx = -1)
        : shape_(shape), device_idx_(device_idx) {
        size_t total = 1;
        for (auto d : shape) total *= d;
        data_.resize(total);
    }
    
    /// Construct from FP32 data (converts to BF16)
    BF16Tensor(const std::vector<size_t>& shape, const float* fp32_data, int device_idx = -1)
        : BF16Tensor(shape, device_idx) {
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] = simd::fp32_to_bf16(fp32_data[i]);
        }
    }
    
    // =========================================================================
    // CRTP Implementation
    // =========================================================================
    
    const uint16_t* data_impl() const { return data_.data(); }
    uint16_t* mutable_data_impl() { return data_.data(); }
    
    // =========================================================================
    // ITensor Interface
    // =========================================================================
    
    TensorType native_type() const override { return TensorType::BF16; }
    const std::vector<size_t>& shape() const override { return shape_; }
    size_t numel() const override { return data_.size(); }
    int device_index() const override { return device_idx_; }
    
    void to_fp32(float* dst) const override {
        for (size_t i = 0; i < data_.size(); ++i) {
            dst[i] = simd::bf16_to_fp32(data_[i]);
        }
    }
    
    std::unique_ptr<ITensor> to_fp32_tensor() const override;
    
    // =========================================================================
    // BF16-Specific Methods
    // =========================================================================
    
    /// Populate from FP32 data
    void from_fp32(const float* src, size_t count) {
        assert(count <= data_.size());
        for (size_t i = 0; i < count; ++i) {
            data_[i] = simd::fp32_to_bf16(src[i]);
        }
    }
    
private:
    std::vector<uint16_t> data_;
    std::vector<size_t> shape_;
    int device_idx_ = -1;
};

} // namespace llaminar2
```

### File: `src/v2/tensors/Q8_1Tensor.h` (REFACTORED)

```cpp
#pragma once

#include "TypedTensorBase.h"
#include "BlockStructures.h"  // For Q8_1Block
#include <vector>

namespace llaminar2 {

/**
 * @brief Q8_1 quantized tensor with block storage
 * 
 * Each Q8_1Block contains 32 int8 values plus a scale and min.
 */
class Q8_1Tensor : public TypedTensorBase<Q8_1Tensor, Q8_1Block> {
public:
    static constexpr TensorType static_type() { return TensorType::Q8_1; }
    static constexpr size_t BLOCK_SIZE = Q8_1Block::BLOCK_SIZE;  // 32
    
    // =========================================================================
    // Constructors
    // =========================================================================
    
    explicit Q8_1Tensor(const std::vector<size_t>& shape, int device_idx = -1)
        : shape_(shape), device_idx_(device_idx) {
        size_t total = 1;
        for (auto d : shape) total *= d;
        numel_ = total;
        num_blocks_ = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        blocks_.resize(num_blocks_);
    }
    
    // =========================================================================
    // CRTP Implementation
    // =========================================================================
    
    const Q8_1Block* data_impl() const { return blocks_.data(); }
    Q8_1Block* mutable_data_impl() { return blocks_.data(); }
    
    // =========================================================================
    // ITensor Interface
    // =========================================================================
    
    TensorType native_type() const override { return TensorType::Q8_1; }
    const std::vector<size_t>& shape() const override { return shape_; }
    size_t numel() const override { return numel_; }
    int device_index() const override { return device_idx_; }
    
    size_t size_bytes() const override {
        return num_blocks_ * sizeof(Q8_1Block);
    }
    
    void to_fp32(float* dst) const override {
        // Dequantize blocks to FP32
        for (size_t b = 0; b < num_blocks_; ++b) {
            const Q8_1Block& block = blocks_[b];
            size_t base = b * BLOCK_SIZE;
            for (size_t i = 0; i < BLOCK_SIZE && (base + i) < numel_; ++i) {
                dst[base + i] = block.d * static_cast<float>(block.qs[i]) + block.m;
            }
        }
    }
    
    std::unique_ptr<ITensor> to_fp32_tensor() const override;
    
    // =========================================================================
    // Q8_1-Specific Accessors
    // =========================================================================
    
    size_t num_blocks() const { return num_blocks_; }
    
    /// Alias for data() - for backward compatibility
    const Q8_1Block* q8_1_blocks() const { return data(); }
    Q8_1Block* mutable_q8_1_blocks() { return mutable_data(); }
    
private:
    std::vector<Q8_1Block> blocks_;
    std::vector<size_t> shape_;
    size_t numel_ = 0;
    size_t num_blocks_ = 0;
    int device_idx_ = -1;
};

} // namespace llaminar2
```

## Usage Examples

### Example 1: Templated Kernel (Compile-Time Type Safety)

```cpp
// Kernel that works with any typed tensor
template<typename InputTensor, typename WeightTensor, typename OutputTensor>
void gemm_kernel(const InputTensor& input, const WeightTensor& weight, OutputTensor& output,
                 int m, int n, int k) {
    // data() returns the correct type for each tensor!
    const auto* A = input.data();    // e.g., float* for FP32, uint16_t* for BF16
    const auto* B = weight.data();   // e.g., Q8_1Block* for Q8_1
    auto* C = output.mutable_data(); // e.g., float* for FP32
    
    // ... kernel implementation using native types
}

// Usage
FP32Tensor input({32, 128});
Q8_1Tensor weight({128, 256});
FP32Tensor output({32, 256});

gemm_kernel(input, weight, output, 32, 256, 128);  // Types deduced automatically
```

### Example 2: Runtime Dispatch with Type Erasure

```cpp
void process_tensor(ITensor& tensor) {
    switch (tensor.native_type()) {
        case TensorType::FP32: {
            auto& t = tensor.typed_as<FP32Tensor>();
            float* data = t.mutable_data();  // Type-safe!
            // process FP32...
            break;
        }
        case TensorType::BF16: {
            auto& t = tensor.typed_as<BF16Tensor>();
            uint16_t* data = t.mutable_data();  // Type-safe!
            // process BF16...
            break;
        }
        case TensorType::Q8_1: {
            auto& t = tensor.typed_as<Q8_1Tensor>();
            Q8_1Block* blocks = t.mutable_data();  // Type-safe!
            // process Q8_1...
            break;
        }
    }
}
```

### Example 3: Using try_as for Optional Type Handling

```cpp
void maybe_optimize_bf16(ITensor& tensor) {
    if (auto* bf16 = tensor.try_as<BF16Tensor>()) {
        // BF16-specific optimization
        uint16_t* data = bf16->mutable_data();
        apply_bf16_optimization(data, bf16->numel());
    }
    // else: not BF16, skip optimization
}
```

### Example 4: Generic Algorithm with Visitor Pattern

```cpp
// Type-safe visitor for tensor operations
template<typename Func>
void visit_tensor(ITensor& tensor, Func&& func) {
    switch (tensor.native_type()) {
        case TensorType::FP32:
            func(tensor.typed_as<FP32Tensor>());
            break;
        case TensorType::BF16:
            func(tensor.typed_as<BF16Tensor>());
            break;
        case TensorType::FP16:
            func(tensor.typed_as<FP16Tensor>());
            break;
        case TensorType::Q8_1:
            func(tensor.typed_as<Q8_1Tensor>());
            break;
        // ... other types
    }
}

// Usage with generic lambda
void zero_tensor(ITensor& tensor) {
    visit_tensor(tensor, [](auto& typed_tensor) {
        auto* data = typed_tensor.mutable_data();
        std::memset(data, 0, typed_tensor.size_bytes());
    });
}
```

## Migration Strategy

### Phase 1: Add New Interfaces (Non-Breaking)

1. Create `ITensor` interface alongside existing `TensorBase`
2. Have `TensorBase` inherit from `ITensor`
3. Add `raw_data()` / `raw_mutable_data()` to existing tensors
4. Add `typed_as<T>()` and `try_as<T>()` helpers

### Phase 2: Add CRTP Layer

1. Create `TypedTensorBase<Derived, DataType>` 
2. Refactor each tensor class to inherit from `TypedTensorBase`
3. Implement `data_impl()` / `mutable_data_impl()` in each class
4. Keep old accessors (`bf16_data()`, etc.) as deprecated aliases

### Phase 3: Update Kernels

1. Template kernels that can benefit from compile-time dispatch
2. Replace switch statements with `typed_as<T>()` where appropriate
3. Remove old `float* data()` overrides that threw exceptions

### Phase 4: Cleanup

1. Remove deprecated aliases (`bf16_data()` → `data()`)
2. Remove old `TensorBase` virtual methods
3. Update documentation

## Files to Modify

### New Files
- `src/v2/tensors/ITensor.h`
- `src/v2/tensors/TypedTensorBase.h`

### Modified Files
- `src/v2/tensors/Tensors.h` - Refactor all tensor classes
- `src/v2/tensors/TensorBase.h` - Inherit from ITensor
- `src/v2/kernels/cpu/gemm_v4/QuantisedGemmKernel.h` - Use typed accessors
- `src/v2/kernels/cpu/FloatingPointGemmKernel.h` - Use typed accessors
- `src/v2/pipelines/PipelineBase.cpp` - Use new patterns

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| Type safety | Runtime checks, exceptions | Compile-time for templated code |
| Performance | Virtual dispatch everywhere | Zero overhead with CRTP |
| Code clarity | `tensor.bf16_data()`, `tensor.q8_1_blocks()` | `tensor.data()` (correct type) |
| Kernel code | Switch on type + cast | Templated or `typed_as<T>()` |
| Memory access | Often through FP32 conversion | Direct native access |

## Open Questions

1. Should `ITensor` have `to_fp32()` or should conversion be a separate utility?
2. How to handle mixed-precision operations (BF16 input, FP32 output)?
3. Should block-based tensors (Q8_1) expose both block and element access?
4. How to handle tensor views that share underlying storage?
