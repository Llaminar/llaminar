# GGUF Weight Storage & Llaminar Matrix Conventions

**Author**: David Sanftenberg  
**Date**: October 7, 2025  
**Status**: Definitive Reference

## Executive Summary

This document establishes canonical conventions for weight matrix storage and matrix multiplication in Llaminar based on deep analysis of GGUF format, llama.cpp implementation, and PyTorch conventions.

### Key Finding

**GGUF stores weights differently than we assumed**, and **llama.cpp uses implicit transposes**. This explains our parity issues.

## 1. GGUF Storage Format (Ground Truth)

### Actual GGUF Tensor Shapes (Qwen 0.5B)

From empirical analysis of `qwen2.5-0.5b-instruct-q4_0.gguf`:

| Tensor Name | GGUF Raw Shape | Notes |
|-------------|----------------|-------|
| `token_embd.weight` | `[151936, 896]` | [vocab_size, d_model] → **TRANSPOSED** |
| `blk.0.attn_q.weight` | `[896, 896]` | [d_model, d_model] → Symmetric |
| `blk.0.attn_k.weight` | `[128, 896]` | **[out_features, in_features]** ← KEY! |
| `blk.0.attn_v.weight` | `[128, 896]` | **[out_features, in_features]** ← KEY! |
| `blk.0.attn_output.weight` | `[896, 896]` | [d_model, d_model] → Symmetric |
| `blk.0.ffn_gate.weight` | `[4864, 896]` | **[out_features, in_features]** |
| `blk.0.ffn_up.weight` | `[4864, 896]` | **[out_features, in_features]** |
| `blk.0.ffn_down.weight` | `[896, 4864]` | **[out_features, in_features]** |
| `output.weight` | `[151936, 896]` | [vocab_size, d_model] → **TRANSPOSED** |

### Critical Insight

**GGUF stores non-symmetric weight matrices as `[out_features, in_features]`!**

This is the **OPPOSITE** of what Llaminar currently expects (`[in_features, out_features]`).

## 2. llama.cpp Conventions

### Weight Matrix Creation

From `llama.cpp/src/llama-model.cpp` (Qwen2/Granite architecture):

```cpp
// Expected tensor shapes passed to create_tensor()
layer.wq = create_tensor({n_embd, n_embd_head_k * n_head}, ...)
           // {896, 896} = [in_features, out_features] in llama.cpp's view

layer.wk = create_tensor({n_embd, n_embd_k_gqa}, ...)
           // {896, 128} = [in_features, out_features] in llama.cpp's view
           
layer.wv = create_tensor({n_embd, n_embd_v_gqa}, ...)
           // {896, 128} = [in_features, out_features] in llama.cpp's view
           
layer.wo = create_tensor({n_embd_head_k * n_head, n_embd}, ...)
           // {896, 896} = [in_features, out_features] in llama.cpp's view
```

**BUT** - llama.cpp doesn't actually load weights with these shapes from GGUF!

The `create_tensor()` call is just specifying the **expected** shape. The actual GGUF file has them stored **transposed**.

### Matrix Multiplication Convention

From `llama.cpp/ggml/src/ggml.c`:

```c
struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,  // Weight matrix
        struct ggml_tensor  * b) {// Input activations
    ...
    // Output shape calculation
    const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    //                      ^^^^^^^^  ^^^^^^^^
    //                      Take a's second dim (not first!)
    ...
}
```

This means:
- `a` (weight): shape `[k, n]` where `ne[0]=k`, `ne[1]=n`
- `b` (input): shape `[k, m, ...]` where `ne[0]=k`, `ne[1]=m`
- Output: shape `[n, m, ...]`

**This computes `C = A^T @ B` - implicit transpose of weight matrix!**

### llama.cpp Strategy

```
1. GGUF stores K weight as [128, 896] (out, in)
2. llama.cpp expects logical shape [896, 128] (in, out)
3. ggml_mul_mat(W, x) implicitly transposes W
4. So [128, 896] stored → used as [896, 128]^T = [128, 896] ✓
5. Result: Works correctly despite "mismatch"
```

## 3. PyTorch Conventions

### nn.Linear Weight Storage

```python
layer = nn.Linear(in_features=896, out_features=128)
# weight shape: [128, 896] = [out_features, in_features]

# Forward pass:
output = F.linear(input, weight, bias)
# Computes: input @ weight.T
# input: [..., 896] @ weight.T: [896, 128] = output: [..., 128]
```

PyTorch stores as `[out_features, in_features]` and applies `x @ W.T`.

### Embedding Storage

```python
embedding = nn.Embedding(vocab_size=151936, embedding_dim=896)
# weight shape: [151936, 896] = [vocab_size, embedding_dim]

# Forward pass:
output = embedding(tokens)
# Indexes: weight[tokens, :]
```

## 4. Convention Comparison Table

| Component | GGUF Storage | llama.cpp Logical | PyTorch Storage | Llaminar Current | Llaminar Proposed |
|-----------|--------------|-------------------|-----------------|------------------|-------------------|
| K weight | `[128, 896]` | `[896, 128]` + implicit transpose | `[128, 896]` | `[896, 128]` ❌ | `[128, 896]` ✅ |
| V weight | `[128, 896]` | `[896, 128]` + implicit transpose | `[128, 896]` | `[896, 128]` ❌ | `[128, 896]` ✅ |
| O weight | `[896, 896]` | `[896, 896]` + implicit transpose | `[896, 896]` | `[896, 896]` ✅ | `[896, 896]` ✅ |
| FFN gate | `[4864, 896]` | `[896, 4864]` + implicit transpose | `[4864, 896]` | `[896, 4864]` ❌ | `[4864, 896]` ✅ |
| FFN up | `[4864, 896]` | `[896, 4864]` + implicit transpose | `[4864, 896]` | `[896, 4864]` ❌ | `[4864, 896]` ✅ |
| FFN down | `[896, 4864]` | `[4864, 896]` + implicit transpose | `[896, 4864]` | `[4864, 896]` ❌ | `[896, 4864]` ✅ |
| Embedding | `[151936, 896]` | Index `[tokens, :]` | `[151936, 896]` | `[151936, 896]` ✅ | `[151936, 896]` ✅ |

### Critical Observation

**Llaminar is currently trying to follow llama.cpp's "logical" convention** (which includes implicit transposes), but **GGUF stores weights in PyTorch convention** (`[out, in]`).

This is why:
- We have dimension mismatches
- Both "reverse all" and "reverse embeddings only" fail
- The issue is with K/V/FFN weights, not embeddings

## 5. Proposed Llaminar Conventions

### A. Weight Storage Convention

**ALL weights shall be stored as `[out_features, in_features]`** to match:
1. GGUF on-disk format (no unnecessary conversions)
2. PyTorch convention (for parity testing)
3. Industry standard (most ML frameworks use this)

This means:
```cpp
// Q projection: 896 → 896
std::shared_ptr<TensorBase> wq;  // shape: [896, 896]

// K projection: 896 → 128
std::shared_ptr<TensorBase> wk;  // shape: [128, 896]

// V projection: 896 → 128
std::shared_ptr<TensorBase> wv;  // shape: [128, 896]

// O projection: 896 → 896
std::shared_ptr<TensorBase> wo;  // shape: [896, 896]

// FFN gate: 896 → 4864
std::shared_ptr<TensorBase> w_gate;  // shape: [4864, 896]

// FFN up: 896 → 4864
std::shared_ptr<TensorBase> w_up;  // shape: [4864, 896]

// FFN down: 4864 → 896
std::shared_ptr<TensorBase> w_down;  // shape: [896, 4864]
```

### B. Matrix Multiplication Convention

**ALL matmuls shall use explicit transposes** - no implicit magic!

```cpp
// Linear layer: y = x @ W^T
// Where:
//   x: [..., in_features]
//   W: [out_features, in_features]  ← stored orientation
//   W^T: [in_features, out_features] ← transpose for matmul
//   y: [..., out_features]

// Example: K projection
// x: [seq_len, 896]
// wk: [128, 896]
// wk^T: [896, 128]
// result: [seq_len, 128] ✓

// CORRECT:
result = matmul(x, wk, false, true);  // x @ wk^T
//                     ^^^^^  ^^^^
//                     don't   DO transpose wk
//                     transpose x

// WRONG:
result = matmul(wk, x);  // wk @ x would give wrong shape!
```

### C. GGUF Loading Convention

```cpp
// In ModelLoader::parseTensorInfo()

bool is_embedding = (
    tensor.name == "token_embd.weight" || 
    tensor.name == "output.weight"
);

if (is_embedding && tensor.dimensions.size() == 2) {
    // Embeddings: GGUF stores [vocab, d_model] ← correct for us!
    // NO REVERSAL NEEDED (GGUF already matches PyTorch/Llaminar)
    // Keep as [vocab_size, d_model] for nn.Embedding indexing
}
else if (tensor.dimensions.size() == 2) {
    // Weight matrices: GGUF stores [out, in] ← correct for us!
    // NO REVERSAL NEEDED (GGUF already matches PyTorch/Llaminar)
    // Keep as [out_features, in_features]
}

// BOTTOM LINE: NO DIMENSION REVERSALS AT ALL!
```

### D. Kernel Documentation Contract

Every kernel/operation involving matrix multiplication **MUST** document:

```cpp
/**
 * @brief Applies linear projection: y = x @ W^T + b
 * 
 * @param x Input tensor with shape [..., in_features]
 * @param W Weight matrix with shape [out_features, in_features]
 * @param b Bias vector with shape [out_features] (optional)
 * @param y Output tensor with shape [..., out_features]
 * 
 * Mathematical operation:
 *   y[..., j] = sum_i(x[..., i] * W[j, i]) + b[j]
 * 
 * Matrix multiplication convention:
 *   result = matmul(x, W, transpose_a=false, transpose_b=true)
 *   This computes: x @ W^T
 * 
 * @note W is stored as [out_features, in_features] per Llaminar convention
 * @note W is transposed during matmul, NOT pre-transposed in storage
 */
bool linear_projection(...);
```

## 6. Migration Plan

### Phase 1: Fix ModelLoader (IMMEDIATE)

```cpp
// src/model_loader.cpp, line ~1368

// REMOVE ALL dimension reversals!
// GGUF already stores weights in the correct orientation

// OLD (WRONG):
if (is_embedding && tensor.dimensions.size() == 2) {
    std::reverse(tensor.dimensions.begin(), tensor.dimensions.end());
}

// NEW (CORRECT):
// NO REVERSALS AT ALL - GGUF format matches our convention!
// Just load dimensions as-is
```

### Phase 2: Fix All Matmul Calls (SYSTEMATIC)

Audit every matmul call and ensure:

```cpp
// BEFORE (ad-hoc, inconsistent):
auto result = matmul(W, x);  // What orientation? Transpose?
auto result = matmul(x, W);  // Same question!

// AFTER (explicit, consistent):
auto result = matmul(x, W, false, true);  // x @ W^T - clear!
//                         ^^^^^  ^^^^
//                         Don't   DO transpose W
//                         transpose x
```

Files to audit:
- `src/kernels/MPIAttentionKernel.cpp`
- `src/kernels/MPILinearKernel.cpp`
- `src/backends/prefill_backend.cpp`
- `src/backends/inference_backend.cpp`
- Any other matmul call sites

### Phase 3: Add Documentation (ONGOING)

1. Update all kernel headers with explicit contracts
2. Add matrix shape assertions in constructors
3. Document weight loading in ModelLoader header
4. Create validation tests for weight shapes

### Phase 4: Validation (CRITICAL)

```cpp
// Add to ModelWeights class:
void validateShapes() {
    // K projection: 896 → 128
    assert(k_proj[0]->shape() == std::vector<int64_t>{128, 896});
    
    // V projection: 896 → 128
    assert(v_proj[0]->shape() == std::vector<int64_t>{128, 896});
    
    // O projection: 896 → 896
    assert(o_proj[0]->shape() == std::vector<int64_t>{896, 896});
    
    // FFN gate: 896 → 4864
    assert(ffn_gate[0]->shape() == std::vector<int64_t>{4864, 896});
    
    // FFN up: 896 → 4864
    assert(ffn_up[0]->shape() == std::vector<int64_t>{4864, 896});
    
    // FFN down: 4864 → 896
    assert(ffn_down[0]->shape() == std::vector<int64_t>{896, 4864});
}
```

## 7. Why This Fixes Our Issues

### Current Problem

```
Llaminar expects: wk = [896, 128]  (trying to match llama.cpp "logical")
GGUF provides: wk = [128, 896]     (actual storage)
ModelLoader reversal: WRONG approach - makes it worse!
Result: Validation fails, parity breaks
```

### With New Convention

```
Llaminar expects: wk = [128, 896]  (matches GGUF and PyTorch)
GGUF provides: wk = [128, 896]     (actual storage)
ModelLoader: NO reversal needed!
Matmul: x @ wk^T computes [seq,896] @ [896,128] = [seq,128] ✓
Result: Validation passes, parity matches!
```

## 8. Reference Implementation

### Correct Linear Layer

```cpp
class LinearLayer {
public:
    /**
     * @brief Linear layer: y = x @ W^T + b
     * 
     * @param in_features Input dimension
     * @param out_features Output dimension
     * 
     * Weight storage: [out_features, in_features]
     * Matmul operation: x @ W^T
     */
    LinearLayer(int in_features, int out_features)
        : in_features_(in_features),
          out_features_(out_features) {
        // Weight shape validation
        weight_ = TensorFactory::create({out_features, in_features});
        bias_ = TensorFactory::create({out_features});
    }
    
    bool forward(const TensorBase& x, TensorBase& y) {
        // x: [..., in_features]
        // weight_: [out_features, in_features]
        // y: [..., out_features]
        
        // Compute: y = x @ weight_^T + bias_
        bool success = matmul(x, *weight_, y, 
                             false,  // Don't transpose x
                             true);  // DO transpose weight_
        
        if (success && bias_) {
            add_bias(y, *bias_);
        }
        
        return success;
    }
    
private:
    int in_features_;
    int out_features_;
    std::shared_ptr<TensorBase> weight_;  // [out_features, in_features]
    std::shared_ptr<TensorBase> bias_;    // [out_features]
};
```

## 9. Conclusion

### Root Cause

Llaminar was trying to match llama.cpp's "logical" weight shapes (which include implicit transposes), instead of matching the actual GGUF storage format.

### Solution

1. **Store weights as `[out, in]`** - matches GGUF and PyTorch
2. **Apply as `x @ W^T`** - explicit transpose in matmul
3. **NO dimension reversals in ModelLoader** - GGUF is already correct
4. **Document everything** - no more ad-hoc conventions

### Benefits

- ✅ Matches GGUF on-disk format (no unnecessary conversions)
- ✅ Matches PyTorch convention (perfect parity testing)
- ✅ Explicit transposes (no hidden magic)
- ✅ Industry standard (most frameworks use `[out, in]`)
- ✅ Fixes all validation errors
- ✅ Enables correct parity testing

### Next Steps

1. Remove all dimension reversals from ModelLoader
2. Audit and fix all matmul calls
3. Add shape validation
4. Test parity - should now match perfectly!
