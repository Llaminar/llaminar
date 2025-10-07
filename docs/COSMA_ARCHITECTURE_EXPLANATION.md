# COSMA Architecture Explanation

## Overview

You're absolutely right to be confused! The naming is somewhat misleading. Let me clarify the architecture:

## The Two COSMA Classes

### 1. **CosmaPrefillManager** (`cosma_prefill_manager.{h,cpp}`)

**Role:** Low-level COSMA infrastructure / utility library (singleton)

**Purpose:** Provides reusable COSMA-specific primitives and infrastructure

**Key Responsibilities:**
- **Weight Streaming:** Converts weights from row-major to COSMA distributed layout
- **Activation Conversion:** Converts activations to COSMA format
- **Strategy Caching:** Manages COSMA strategy cache for different matrix sizes
- **Memory Management:** Tracks memory budgets, handles allocations
- **Validation:** Optional tile validation, replicated comparison for debugging
- **Instrumentation:** Performance counters, debug snapshots, statistics
- **In-Layout Kernels:** RMSNorm, SwiGLU, Softmax operating on COSMA layout directly

**Key APIs:**
```cpp
CosmaView convert_activation_in(const float* row_major, int m, int k);
CosmaWeightHandle load_weight(const WeightDescriptor& desc);
CosmaView matmul(const CosmaView& A, const CosmaWeightHandle& W, ...);
bool rmsnorm_in_layout(const CosmaView& src, CosmaView& dst, ...);
```

**Think of it as:** The "COSMA backend library" - a toolbox of COSMA utilities that can be used by any provider.

### 2. **COSMAPrefillProvider** (`cosma_prefill_provider.{h,cpp}`)

**Role:** High-level prefill orchestrator (implements `PrefillProvider` interface)

**Purpose:** Orchestrates the full prefill execution flow

**Execution Strategy:**
- **Attention:** Uses `CosmaPrefillManager` for fused norm+QKV, then CPU primitives
- **FFN:** Uses `adaptiveMatMul()` for gate/up/down projections (NOT pure COSMA!)
- **Final:** Uses `adaptiveMatMul()` for LM head

**Key Point:** Despite the name, this provider is **HYBRID** - it uses both COSMA (via manager) AND OpenBLAS (via adaptiveMatMul).

## The Two Prefill Providers

### **OpenBLASPrefillProvider** (`openblas_prefill_provider.{h,cpp}`)

**Architecture:** Pure kernel-based approach

**Execution Flow:**
```
Embedding → [For each layer: Norm → Attention → Residual → Norm → FFN → Residual] → Norm → LM Head
```

**Uses Registered Kernels:**
- `MPIEmbeddingKernel` - Token embedding
- `MPIRMSNormKernel` - Normalization
- `MPIAttentionKernel` - **ALL attention ops** (QKV projections + attention + output projection)
- `MPILinearKernel` - FFN projections (gate, up, down)
- `MPISwiGLUKernel` - SwiGLU activation
- `MPIResidualKernel` - Residual connections

**Backend Selection:** Happens INSIDE the kernels (they may use OpenBLAS, COSMA, or adaptive selection)

### **COSMAPrefillProvider** (`cosma_prefill_provider.{h,cpp}`)

**Architecture:** Mixed approach - some COSMA, some adaptive

**Execution Flow:**
```
Embedding → [For each layer: 
  COSMA Attention (fused norm+QKV) → RoPE/Scores/Softmax (CPU) → Adaptive output proj
  Norm → Adaptive gate → Adaptive up → SwiGLU → Adaptive down → Residual
] → Norm → Adaptive LM Head
```

**Uses:**
- `CosmaPrefillManager::fused_rmsnorm_qkv()` - **COSMA** for attention norm+QKV
- CPU attention primitives (RoPE, scores, softmax) - **CPU**
- `adaptiveMatMul()` - **Adaptive** (may choose OpenBLAS or COSMA) for:
  - Attention output projection
  - FFN gate/up/down projections  
  - LM head projection
- Registered kernels: SwiGLU, Residual, RMSNorm (fallback)

## The Key Difference

### What is `adaptiveMatMul()`?

`adaptiveMatMul()` from `adaptive_matmul.{h,cpp}` **automatically selects** between:
- **OpenBLAS:** For small operations, single token, decode
- **COSMA:** For large prefill operations (>4K tokens typically)

**Selection criteria:**
```cpp
// Simplified logic:
if (ADAPTIVE_DISABLE_COSMA env var set) → OpenBLAS
else if (is_prefill && seq_len >= 4096) → COSMA
else → OpenBLAS
```

### So What's the Real Difference Between Providers?

| Aspect | OpenBLASPrefillProvider | COSMAPrefillProvider |
|--------|------------------------|---------------------|
| **Attention** | Uses `MPIAttentionKernel` (all-in-one) | Uses `CosmaPrefillManager::fused_rmsnorm_qkv()` + CPU primitives + `adaptiveMatMul()` |
| **FFN** | Uses `MPILinearKernel` for projections | Uses `adaptiveMatMul()` directly |
| **LM Head** | Uses `MPILinearKernel` | Uses `adaptiveMatMul()` |
| **Backend Decision** | Inside kernels | Explicit via `adaptiveMatMul()` |
| **COSMA Path** | May use COSMA internally in kernels | Explicitly uses COSMA for attention norm+QKV, adaptive for rest |
| **Code Path** | Kernel abstraction layer | Direct function calls |

## Why the Confusion?

The naming suggests:
- ❌ "OpenBLAS provider only uses OpenBLAS"
- ❌ "COSMA provider only uses COSMA"

But the reality is:
- ✅ **OpenBLASPrefillProvider:** Uses kernel abstraction; kernels internally may use OpenBLAS, COSMA, or adaptive selection
- ✅ **COSMAPrefillProvider:** Uses COSMA for attention norm+QKV; uses `adaptiveMatMul()` (which can choose OpenBLAS or COSMA) for everything else

## Better Mental Model

Think of it this way:

### OpenBLASPrefillProvider
- **Architecture:** Kernel-based (everything goes through registered kernels)
- **Backend Selection:** Hidden inside kernels (implementation detail)
- **Legacy naming:** Called "OpenBLAS" but kernels may use any backend

### COSMAPrefillProvider  
- **Architecture:** Direct execution (bypasses some kernels)
- **Backend Selection:** Explicit via `CosmaPrefillManager` + `adaptiveMatMul()`
- **Misleading naming:** Called "COSMA" but uses adaptive selection (may use OpenBLAS!)

## What Should We Call Them?

More accurate names would be:

| Current Name | Better Name | Rationale |
|--------------|-------------|-----------|
| `OpenBLASPrefillProvider` | `KernelBasedPrefillProvider` | Uses kernel abstraction layer |
| `COSMAPrefillProvider` | `DirectExecutionPrefillProvider` or `HybridPrefillProvider` | Bypasses kernels, uses mix of COSMA + adaptive |

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│             Prefill Provider Interface              │
└─────────────────────────────────────────────────────┘
                         ▲
                         │
        ┌────────────────┴────────────────┐
        │                                 │
┌───────▼────────┐              ┌────────▼─────────┐
│  OpenBLAS      │              │   COSMA          │
│  PrefillProvider│              │   PrefillProvider│
└────────────────┘              └──────────────────┘
        │                                 │
        │ Uses Kernels:                   │ Uses:
        │ - MPIAttentionKernel            │ - CosmaPrefillManager
        │ - MPILinearKernel               │ - adaptiveMatMul()
        │ - MPIRMSNormKernel              │ - Some kernels
        │ - MPISwiGLUKernel               │
        │ - MPIResidualKernel             │
        │                                 │
        └────────────┬────────────────────┘
                     ▼
        ┌────────────────────────────┐
        │     Backend Selection      │
        │  (OpenBLAS vs COSMA)      │
        │                           │
        │  - Inside kernels (left)  │
        │  - adaptiveMatMul (right) │
        └────────────────────────────┘
```

## Practical Implications

1. **Both providers may use COSMA** - it's not exclusive to COSMAPrefillProvider
2. **Both providers may use OpenBLAS** - it's not exclusive to OpenBLASPrefillProvider  
3. **Real difference is architecture:**
   - OpenBLAS provider: Kernel-based abstraction
   - COSMA provider: Direct execution with explicit backend choices
4. **COSMA provider is actually more complex** - mixes COSMA manager, adaptive selection, and fallback kernels

## Recommendations

### Short Term
1. **Add clarifying comments** to both provider headers explaining the actual backend selection
2. **Document** that adaptiveMatMul may choose OpenBLAS in COSMAPrefillProvider
3. **Explain** that kernels in OpenBLASPrefillProvider may use COSMA

### Long Term
1. **Consider renaming:**
   - `OpenBLASPrefillProvider` → `KernelBasedPrefillProvider`
   - `COSMAPrefillProvider` → `HybridPrefillProvider` or `DirectPrefillProvider`
2. **Unify backend selection** - why have two different paths for backend selection?
3. **Consider:** Should all providers use the same backend selection strategy?

## Summary

- **CosmaPrefillManager:** Low-level COSMA utility library (singleton)
- **COSMAPrefillProvider:** High-level orchestrator that uses COSMA for some ops, adaptive selection for others
- **OpenBLASPrefillProvider:** Kernel-based orchestrator where kernels may use any backend
- **adaptiveMatMul():** Backend selector that may choose OpenBLAS OR COSMA based on operation size
- **Naming is misleading:** Neither provider is exclusively tied to one backend
