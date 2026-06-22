# Project Plan: Tensor Base Class Refactor (Using TypedTensorBase)

## Overview

**Goal**: Refactor the tensor class hierarchy to provide compile-time type safety for `.data()` accessors by leveraging the existing `TypedTensorBase<Derived, DataType>` CRTP template.

**Status**: Planning → Ready to Implement  
**Created**: December 28, 2025  
**Author**: David Sanftenberg

---

## Problem Statement

### Current State

`TensorBase` defines virtual `data() → float*` and `mutable_data() → float*` accessors, forcing ALL tensors to pretend they contain floats:

| Tensor Type | Storage Type | `data()` Behavior |
|-------------|--------------|-------------------|
| FP32Tensor | `float` | ✓ Returns `float*` (correct) |
| BF16Tensor | `bf16_t` | ⚠ Converts to `float*` (lossy) |
| Q8_1Tensor | `Q8_1Block` | ✗ **THROWS** - has blocks, not floats |
| IQ4_NLTensor | `IQ4_NLBlock` | ✗ **THROWS** - has blocks, not floats |

### The Problem

```cpp
// This compiles but throws at runtime for Q8_1Tensor!
void process(TensorBase* tensor) {
    const float* data = tensor->data();  // 💥 Throws for block tensors
}

// Current workaround: inconsistent accessor names
q8_1_tensor->q8_1_blocks();   // Q8_1Tensor
q16_1_tensor->q16_1_blocks(); // Q16_1Tensor  
iq4_tensor->raw_blocks();     // IQ4_NLTensor
q8_0_tensor->???              // No accessor, uses get_raw_block_at()
```

---

## Solution: Leverage Existing TypedTensorBase

### Key Discovery

**We already have the solution!** `TypedTensorBase<Derived, DataType>` in `TypedTensorBase.h` provides:

```cpp
template <typename Derived, typename DataType>
class TypedTensorBase : public ITensor {
public:
    using value_type = DataType;  // float, bf16_t, Q8_1Block, etc.
    
    // CRTP: data() returns the CORRECT type for this tensor
    const DataType* data() const { 
        return static_cast<const Derived*>(this)->data_impl(); 
    }
    DataType* mutable_data() { 
        return static_cast<Derived*>(this)->mutable_data_impl(); 
    }
};
```

### Unified Interface

With `TypedTensorBase`, **all tensors use `data()`** - the return type changes based on the tensor:

| Tensor | Inherits From | `data()` Returns |
|--------|---------------|------------------|
| FP32Tensor | `TypedTensorBase<FP32Tensor, float>` | `const float*` |
| BF16Tensor | `TypedTensorBase<BF16Tensor, bf16_t>` | `const bf16_t*` |
| Q8_1Tensor | `TypedTensorBase<Q8_1Tensor, Q8_1Block>` | `const Q8_1Block*` |
| Q16_1Tensor | `TypedTensorBase<Q16_1Tensor, Q16_1Block>` | `const Q16_1Block*` |
| IQ4_NLTensor | `TypedTensorBase<IQ4_NLTensor, IQ4_NLBlock>` | `const IQ4_NLBlock*` |

**No separate ElementTensor/BlockTensor needed!** The `DataType` template parameter IS the distinction.

---

## Benefits

1. **Compile-time type safety**: `tensor.data()` returns the correct pointer type
2. **Unified interface**: All tensors use `data()`, not `q8_1_blocks()` vs `raw_blocks()`
3. **Zero overhead**: CRTP means no virtual dispatch when concrete type is known
4. **Generic algorithms**: `tensor_value_type_t<T>` enables template metaprogramming
5. **Existing infrastructure**: `TypedTensorBase` already exists and is tested
6. **Backward compatible aliases**: `blocks()` can be an alias for `data()` on block tensors

---

## Implementation Plan

### Phase 1: Integrate TypedTensorBase into Tensor Hierarchy

**Goal**: Make all tensors inherit from `TypedTensorBase<Derived, DataType>` alongside `TensorBase`.

#### 1.1 Determine Diamond Inheritance Strategy

Current:
```
ITensor
   └── TensorBase
          └── FP32Tensor, Q8_1Tensor, etc.
```

Option A - Multiple Inheritance (Recommended):
```
ITensor ←───────────────────────────────┐
   └── TypedTensorBase<D,T> : ITensor   │
   └── TensorBase : ITensor ────────────┘
          └── FP32Tensor : TypedTensorBase<FP32Tensor, float>, TensorBase
```

Option B - Linear Chain:
```
ITensor
   └── TypedTensorBase<D,T> : ITensor
          └── TensorBase : (needs to be templated or removed)
                 └── FP32Tensor
```

**Decision**: Option A (multiple inheritance with virtual ITensor base) is cleanest.
Both `TypedTensorBase` and `TensorBase` already inherit from `ITensor` virtually.

#### 1.2 Update FP32Tensor (Proof of Concept)

```cpp
// BEFORE
class FP32Tensor : public TensorBase, 
                   public IActivationTensor, 
                   public ITensorGemmTileDataProvider, ... {
    const float* data() const override;      // Virtual, from TensorBase
    float* mutable_data() override;
private:
    std::vector<float, AlignedAllocator<float>> host_data_;
};

// AFTER
class FP32Tensor : public TypedTensorBase<FP32Tensor, float>,
                   public TensorBase,
                   public IActivationTensor, 
                   public ITensorGemmTileDataProvider, ... {
public:
    // CRTP implementation (called by TypedTensorBase::data())
    const float* data_impl() const { return host_data_.data(); }
    float* mutable_data_impl() { return host_data_.data(); }
    
    static constexpr int static_type_id() { return TensorTypeId::FP32; }
    
private:
    std::vector<float, AlignedAllocator<float>> host_data_;
};
```

#### 1.3 Update Q8_1Tensor (Block Tensor Example)

```cpp
// BEFORE
class Q8_1Tensor : public TensorBase, public IActivationTensor, ... {
    const float* data() const override;       // THROWS!
    const Q8_1Block* q8_1_blocks() const;     // Custom accessor
    Q8_1Block* mutable_q8_1_blocks();
};

// AFTER
class Q8_1Tensor : public TypedTensorBase<Q8_1Tensor, Q8_1Block>,
                   public TensorBase,
                   public IActivationTensor, ... {
public:
    // CRTP implementation
    const Q8_1Block* data_impl() const { return blocks_ptr(); }
    Q8_1Block* mutable_data_impl() { return mutable_blocks_ptr(); }
    
    static constexpr int static_type_id() { return TensorTypeId::Q8_1; }
    
    // Backward-compatible aliases
    const Q8_1Block* blocks() const { return data(); }
    Q8_1Block* mutable_blocks() { return mutable_data(); }
    
    // Legacy alias (deprecated)
    [[deprecated("Use blocks() or data() instead")]]
    const Q8_1Block* q8_1_blocks() const { return data(); }
};
```

---

### Phase 2: Migrate All Tensor Types

#### Element-Type Tensors

| Tensor | `DataType` | `static_type_id()` |
|--------|------------|-------------------|
| FP32Tensor | `float` | `TensorTypeId::FP32` |
| BF16Tensor | `bf16_t` | `TensorTypeId::BF16` |
| FP16Tensor | `fp16_t` | `TensorTypeId::FP16` |
| INT8Tensor | `int8_t` | `TensorTypeId::INT8` |
| INT32Tensor | `int32_t` | `TensorTypeId::INT32` |

#### Block-Type Tensors

| Tensor | `DataType` | `static_type_id()` |
|--------|------------|-------------------|
| Q8_0Tensor | `Q8_0Block` | `TensorTypeId::Q8_0` |
| Q8_1Tensor | `Q8_1Block` | `TensorTypeId::Q8_1` |
| Q16_1Tensor | `Q16_1Block` | `TensorTypeId::Q16_1` |
| Q4_0Tensor | `Q4_0Block` | `TensorTypeId::Q4_0` |
| Q4_1Tensor | `Q4_1Block` | `TensorTypeId::Q4_1` |
| Q5_0Tensor | `Q5_0Block` | `TensorTypeId::Q5_0` |
| Q5_1Tensor | `Q5_1Block` | `TensorTypeId::Q5_1` |
| Q6_KTensor | `Q6_KBlock` | `TensorTypeId::Q6_K` |
| Q2_KTensor | `Q2_KBlock` | `TensorTypeId::Q2_K` |
| Q3_KTensor | `Q3_KBlock` | `TensorTypeId::Q3_K` |
| Q4_KTensor | `Q4_KBlock` | `TensorTypeId::Q4_K` |
| Q5_KTensor | `Q5_KBlock` | `TensorTypeId::Q5_K` |
| Q8_KTensor | `Q8_KBlock` | `TensorTypeId::Q8_K` |
| IQ4_NLTensor | `IQ4_NLBlock` | `TensorTypeId::IQ4_NL` |
| IQ4_XSTensor | `IQ4_XSBlock` | `TensorTypeId::IQ4_XS` |
| IQ2_XXSTensor | `IQ2_XXSBlock` | `TensorTypeId::IQ2_XXS` |
| IQ2_XSTensor | `IQ2_XSBlock` | `TensorTypeId::IQ2_XS` |
| IQ3_XXSTensor | `IQ3_XXSBlock` | `TensorTypeId::IQ3_XXS` |
| IQ2_STensor | `IQ2_SBlock` | `TensorTypeId::IQ2_S` |
| IQ3_STensor | `IQ3_SBlock` | `TensorTypeId::IQ3_S` |
| IQ1_STensor | `IQ1_SBlock` | `TensorTypeId::IQ1_S` |
| IQ1_MTensor | `IQ1_MBlock` | `TensorTypeId::IQ1_M` |

---

### Phase 3: Update TensorBase

#### 3.1 Deprecate Virtual data() Methods

```cpp
class TensorBase : public ITensor, public std::enable_shared_from_this<TensorBase> {
public:
    // DEPRECATED - use TypedTensorBase::data() instead
    [[deprecated("Use typed tensor's data() method instead")]]
    virtual const float* data() const { 
        throw std::runtime_error("Use typed_as<T>().data() for type-safe access");
    }
    
    [[deprecated("Use typed tensor's mutable_data() method instead")]]
    virtual float* mutable_data() {
        throw std::runtime_error("Use typed_as<T>().mutable_data() for type-safe access");
    }
    
    // Keep fp32_data() for explicit FP32 conversion
    virtual const float* fp32_data() const = 0;  // Dequantizes if needed
};
```

#### 3.2 Add Type Query Helpers

```cpp
class TensorBase {
public:
    // Check storage category at runtime
    bool is_element_tensor() const {
        switch (native_type()) {
            case TensorType::FP32:
            case TensorType::BF16:
            case TensorType::FP16:
            case TensorType::INT8:
            case TensorType::INT32:
                return true;
            default:
                return false;
        }
    }
    
    bool is_block_tensor() const { return !is_element_tensor(); }
};
```

---

### Phase 4: Update Call Sites

Most call sites already use typed pointers, so minimal changes needed:

#### Pattern 1: Already Typed (No Change)
```cpp
FP32Tensor* fp32 = ...;
const float* data = fp32->data();  // Already type-safe
```

#### Pattern 2: Through TensorBase* (Use typed_as)
```cpp
// BEFORE
const float* data = tensor->data();  // Might throw!

// AFTER  
const float* data = tensor->typed_as<FP32Tensor>().data();  // Compile-time safe
// OR
const float* data = tensor->fp32_data();  // Explicit dequantization
```

#### Pattern 3: Generic Algorithms (Use Concepts)
```cpp
// C++20 concepts for generic code
template<typename T>
concept HasFloatData = requires(T t) {
    { t.data() } -> std::convertible_to<const float*>;
};

template<typename T>
concept HasBlockData = requires(T t) {
    { t.data() } -> std::convertible_to<const typename T::value_type*>;
    requires !std::same_as<typename T::value_type, float>;
};
```

---

## Research Findings (Completed)

### R1: Current Interface Hierarchy

```
ITensor (ITensor.h)
├── native_type_id(), shape(), numel(), size_bytes(), device_index()
├── raw_data(), raw_mutable_data()
├── typed_as<T>(), try_as<T>(), is<T>() - Type-safe downcasting
└── to_fp32(float* dst) - Universal conversion

TensorBase : ITensor (Tensors.h)
├── shape(), native_type(), rows(), cols()
├── device_index(), set_device()
├── data(), mutable_data() → float*  ← THE PROBLEM
├── fp32_data(), to_fp32(), etc.
└── createGemm(), copyFrom()

TypedTensorBase<Derived, DataType> : ITensor (TypedTensorBase.h)
├── data() → const DataType*  ← THE SOLUTION
├── mutable_data() → DataType*
├── value_type, pointer, const_pointer aliases
└── is_typed_tensor_v<T>, tensor_value_type_t<T> traits
```

### R2: Block Accessor Inconsistency (Will Be Fixed)

| Tensor | Current Accessor | After Refactor |
|--------|------------------|----------------|
| Q8_1Tensor | `q8_1_blocks()` | `data()` (+ `blocks()` alias) |
| Q16_1Tensor | `q16_1_blocks()` | `data()` (+ `blocks()` alias) |
| IQ4_NLTensor | `raw_blocks()` | `data()` (+ `blocks()` alias) |
| Q8_0Tensor | `get_raw_block_at()` | `data()` (+ `blocks()` alias) |

### R3: Call Site Analysis (~35 sites)

Most `->data()` calls are on variables **already known to be FP32**:
- `input_fp32->data()` - typed pointer
- `layer.q_bias->data()` - bias tensors are always FP32
- `fp32_tensor->data()` - variable name indicates type

**Conclusion**: Minimal migration effort required for call sites.

---

## Files to Modify

### Files to Update

| File | Changes |
|------|---------|
| `src/v2/tensors/TypedTensorBase.h` | Minor: ensure virtual inheritance from ITensor |
| `src/v2/tensors/Tensors.h` | Add TypedTensorBase inheritance to all tensor classes |
| `src/v2/tensors/FP32Tensor.cpp` | Add `data_impl()`, `mutable_data_impl()` |
| `src/v2/tensors/BF16Tensor.cpp` | Add `data_impl()`, `mutable_data_impl()` |
| `src/v2/tensors/FP16Tensor.cpp` | Add `data_impl()`, `mutable_data_impl()` |
| `src/v2/tensors/INT8Tensor.cpp` | Add `data_impl()`, `mutable_data_impl()` |
| `src/v2/tensors/INT32Tensor.cpp` | Add `data_impl()`, `mutable_data_impl()` |
| `src/v2/tensors/Q8_0Tensor.cpp` | Add `data_impl()`, add `blocks()` alias |
| `src/v2/tensors/Q8_1Tensor.cpp` | Add `data_impl()`, rename `q8_1_blocks()` → `blocks()` |
| `src/v2/tensors/Q16_1Tensor.cpp` | Add `data_impl()`, rename `q16_1_blocks()` → `blocks()` |
| `src/v2/tensors/Q4_0Tensor.cpp` | Add `data_impl()`, add `blocks()` alias |
| `src/v2/tensors/IQ4_NLTensor.cpp` | Add `data_impl()`, add `blocks()` alias |
| ... (all other quantized tensor types) |

### New Files

| File | Purpose |
|------|---------|
| `tests/v2/unit/tensors/Test__TypedTensorBase.cpp` | Test CRTP data() for all tensor types |

### Call Sites to Audit

| Location | Current | Assessment |
|----------|---------|------------|
| `ComputeStage.cpp` | `tensor->data()` | Already typed (FP32Tensor*) |
| `Qwen2Graph.cpp` | `layer.q_bias->data()` | Bias is always FP32 |
| `WeightManager.cpp` | `fp32_tensor->data()` | Already typed |
| `GraphResolver.cpp` | `stage.inputs[4]->data()` | May need `fp32_data()` |
| `KernelFactory.cpp` | `gate->data()` | May need type check |

---

## Testing Strategy

### Unit Tests

```cpp
// Test__TypedTensorBase.cpp

TEST(Test__TypedTensorBase, FP32DataReturnsFloat) {
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{32, 64}, -1);
    static_assert(std::is_same_v<decltype(tensor->data()), const float*>);
    EXPECT_NE(tensor->data(), nullptr);
}

TEST(Test__TypedTensorBase, Q8_1DataReturnsBlock) {
    auto tensor = std::make_shared<Q8_1Tensor>(std::vector<size_t>{32, 64}, -1);
    static_assert(std::is_same_v<decltype(tensor->data()), const Q8_1Block*>);
    EXPECT_NE(tensor->data(), nullptr);
}

TEST(Test__TypedTensorBase, BlocksAliasMatchesData) {
    auto tensor = std::make_shared<Q8_1Tensor>(std::vector<size_t>{32, 64}, -1);
    EXPECT_EQ(tensor->blocks(), tensor->data());
}

TEST(Test__TypedTensorBase, ValueTypeTraits) {
    static_assert(std::is_same_v<tensor_value_type_t<FP32Tensor>, float>);
    static_assert(std::is_same_v<tensor_value_type_t<Q8_1Tensor>, Q8_1Block>);
    static_assert(is_typed_tensor_v<FP32Tensor>);
    static_assert(is_typed_tensor_v<Q8_1Tensor>);
}
```

### Migration Testing

1. **Phase 1**: Migrate FP32Tensor → run all unit tests
2. **Phase 2**: Migrate Q8_1Tensor → run all unit tests  
3. **Phase 3**: Migrate remaining tensors → run full test suite
4. **Phase 4**: Deprecate TensorBase::data() → check for warnings
5. **Phase 5**: Performance regression tests

---

## Timeline Estimate

| Phase | Effort | Status |
|-------|--------|--------|
| Research & Discovery | 2-3 hours | ✅ COMPLETE |
| Phase 1: FP32Tensor PoC | 1-2 hours | ⏳ Ready |
| Phase 2: Q8_1Tensor PoC | 1-2 hours | ⏳ Ready |
| Phase 3: All Tensors | 3-4 hours | ⏳ Pending |
| Phase 4: TensorBase cleanup | 1-2 hours | ⏳ Pending |
| Phase 5: Call site audit | 2-3 hours | ⏳ Pending |
| Testing & Validation | 2-3 hours | ⏳ Pending |

**Total: ~15-20 hours** (reduced from 20-30 hours with simpler approach)

---

## Success Criteria

1. **Compile-time type safety**: `tensor.data()` returns correct type based on tensor class
2. **Unified interface**: All tensors use `data()` + optional `blocks()` alias
3. **Zero runtime overhead**: CRTP eliminates virtual dispatch for typed access
4. **Backward compatibility**: Existing code using typed pointers works unchanged
5. **Deprecation path**: `TensorBase::data()` marked deprecated with clear migration
6. **All 188+ unit tests pass**

---

## Next Steps

1. [ ] Update `TypedTensorBase.h` to use virtual inheritance from `ITensor`
2. [ ] Migrate `FP32Tensor` as proof of concept
3. [ ] Migrate `Q8_1Tensor` with `blocks()` alias
4. [ ] Run unit tests to validate
5. [ ] Migrate remaining tensor types
6. [ ] Deprecate `TensorBase::data()` with `[[deprecated]]`
7. [ ] Audit and update call sites that use `TensorBase*->data()`
