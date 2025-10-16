# Q_PROJECTION Divergence Root Cause Analysis
**Date**: October 16, 2025  
**Author**: GitHub Copilot  
**Status**: 🔴 Critical Bug Identified - Fix Required

## Summary

Identified the root cause of Q_PROJECTION divergence between batch and sequential pipelines. The issue is in `AdaptiveMatmul.h::multiply_openblas()` - **incorrect leading dimension calculation for transposed matrices**.

## Problem

### Symptoms
- EMBEDDING matches perfectly (max_diff=0) ✓
- ATTENTION_NORM matches perfectly (max_diff=0) ✓  
- Q_PROJECTION diverges catastrophically:
  - Max absolute difference: **72.2547**
  - Relative L2 error: **0.981276** (values nearly completely different)
  - Sequential: `[0.0895339, 1.51824, 0.242193, 3.99491, -3.28598, ...]`
  - Batch: `[0.0903566, 0.282258, 0.128611, -0.728124, -1.19175, ...]`

### Investigation Results

**Inputs and weights are IDENTICAL:**
- Input first 5 elements: `[0.0696318, 0, 0.0718692, -0.0335947, -0.0210757]` ✓ SAME
- Weight first 5 elements: `[-0, -0.0030365, 0.006073, 0.0091095, 0.0030365]` ✓ SAME
- **But outputs are completely different!** ❌

This proves the bug is in the **matrix multiplication logic**, not in weight loading or data preparation.

### Root Cause

**File**: `src/AdaptiveMatmul.h`  
**Function**: `AdaptiveMatMulManager::multiply_openblas()`  
**Lines**: ~630-635

**Buggy Code**:
```cpp
int lda = transpose_A ? m : k; // RowMajor rule: lda = (transA? M : K)
int ldb = transpose_B ? k : n; // RowMajor rule: ldb = (transB? K : N) ❌ WRONG!
int ldc = n;

cblas_sgemm(CblasRowMajor, trans_A, trans_B,
            m, n, k,
            alpha, A, lda,
            B, ldb,
            beta, C, ldc);
```

**Why It's Wrong**:

For row-major matrices, the leading dimension is the **memory stride** (number of elements per row), regardless of transpose flags.

If B is stored as `[n, k]` row-major:
- Memory layout: n rows, k columns per row
- Leading dimension: `ldb = k` (always k, even when transposing)
- The `CblasTrans` flag tells BLAS to **interpret** it as `[k, n]` during the operation
- But the stride in memory is still k

**Current logic** `ldb = transpose_B ? k : n`:
- When `transpose_B=true`: ldb = k ✓ CORRECT by accident
- When `transpose_B=false`: ldb = n ❌ WRONG (should still be k if B is [n,k])

**The real issue**: The formula assumes B's storage layout changes based on the transpose flag, but it doesn't. B is always stored the same way.

### Comparison with Working Code

**Sequential (MPIAttentionOperator::matmul_with_bias)** - WORKS:
```cpp
// Weight is [N, K] stored row-major
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            M, N, K,
            1.0f, input, K,     // lda = K (input is [M,K])
            weight, K,          // ldb = K (weight is [N,K]) ✓ CORRECT!
            0.0f, output, N);
```

**Batch (calls adaptiveMatMul with transpose_B=true)** - BROKEN:
```cpp
// Calls adaptiveMatMul(..., transpose_B=true)
// Which calculates ldb = transpose_B ? k : n = k ✓
// This happens to work for this case, but the logic is still wrong
```

## Correct Fix

The leading dimension should be based on the **actual storage layout** of the matrix, not the transpose operation:

```cpp
// For row-major matrices:
// A is [m, k] -> lda = k
// B is [n, k] when transpose_B=true (will be transposed to [k,n] during op)
//   or [k, n] when transpose_B=false
// C is [m, n] -> ldc = n

int lda = k;  // A is always [m, k] row-major
int ldb = transpose_B ? k : n;  // B is [n,k] if transposing, [k,n] if not
int ldc = n;  // C is always [m, n] row-major
```

**OR**, if the function parameters represent logical dimensions:
```cpp
// If parameters represent: C = op(A) @ op(B) where op() is optional transpose
// A_storage: [m_A, k_A] where m_A=m or k, k_A=k or m depending on transpose_A
// B_storage: [k_B, n_B] where k_B=k or n, n_B=n or k depending on transpose_B

int lda = transpose_A ? m : k;  // Stride of A in storage
int ldb = transpose_B ? n : k;  // Stride of B in storage ❌ Still wrong!
```

Actually, wait. Let me reconsider...

## CBLAS Row-Major Convention

For `cblas_sgemm(CblasRowMajor, trans_A, trans_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)`:

Computes: `C = alpha * op(A) * op(B) + beta * C`

Where:
- `op(A)` = A if trans_A=CblasNoTrans, A^T if trans_A=CblasTrans
- `op(A)` must be m×k
- `op(B)` must be k×n  
- C is m×n

**Leading dimensions for ROW-MAJOR storage**:
- `lda` = number of **columns** in the storage layout of A (not op(A))
  - If A is stored as [m, k]: lda = k
  - If A is stored as [k, m] (transposed): lda = m
- `ldb` = number of **columns** in the storage layout of B
  - If B is stored as [k, n]: ldb = n
  - If B is stored as [n, k] (transposed): ldb = k
- `ldc` = number of **columns** in C = n (C is always [m, n])

So the **current formula is correct for the CALLING convention**, assuming:
- When `transpose_A=true`: A is stored as [k, m] with lda=m
- When `transpose_B=true`: B is stored as [n, k] with ldb=k

**But our actual storage**:
- Batch operator: weight is `[n, k]` (local_rows × D), calls with `transpose_B=true`
  - So ldb should be k ✓ Current formula gives k ✓ CORRECT
- Sequential: weight is `[n, k]`, uses `ldb=k` directly ✓ CORRECT

Wait, so why are the results different? Let me check if there's a different issue...

Actually, let me add more debugging to see what's being passed to cblas_sgemm in both cases:

