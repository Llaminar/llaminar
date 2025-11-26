# Q8_1 GEMM JIT Architecture

**Location:** `src/v2/kernels/cpu/gemm_v4/`
**Status:** Production (Optimized for Qwen 0.5B - 32B)

## Overview

The `Q8_1` GEMM system is a high-performance, JIT-compiled Int8 matrix multiplication engine designed for CPU inference. It performs the core computation of **Quantized Weights (Int8) × Quantized Activations (Int8)**, utilizing AVX512-VNNI instructions to achieve theoretical peak throughput.

Unlike static kernels, this system uses **Xbyak** to generate machine code at runtime, allowing for register-level optimizations specific to the batch size (M) and available instruction set.

It now supports **Generic Weight Packing** via the `IINT8Unpackable` interface, allowing it to operate on any quantized tensor format (Q4_0, Q8_0, Q8_1, etc.) that implements this interface, without needing format-specific kernels.

## Component Architecture

The system is composed of three layers:

### 1. The JIT Kernels (`Q8_1GemmJit_M*.h`)
These files define the assembly generation logic using the **Xbyak** JIT assembler.
- **`Q8_1GemmJit_M1.h`**: Generates a kernel specialized for **M=1** (Single token decode). Optimized for latency.
- **`Q8_1GemmJit_M2.h`**: Generates a kernel specialized for **M=2** (Small batch/speculative decoding).
- **`Q8_1GemmJit_M4.h`** (and others): Specialized for larger fixed M sizes.
- **Mechanism**:
    - Allocates executable memory.
    - Emits AVX512 instructions (`vmovups`, `vpdpbusd`, `vfmadd231ps`).
    - Hardcodes loop unrolling and register allocation for the specific M.

### 2. The Orchestrator (`Q8_1GemmKernel.h`)
This is the high-level C++ driver that manages execution strategy. It does **not** perform the math itself but orchestrates:
- **Threading**: OpenMP parallelization over N and M dimensions.
- **Cache Management**: K-Tiling and Block sizing.
- **Dispatch**: Selecting the correct JIT kernel (M1, M2, or generic) based on runtime dimensions.
- **Generic Weight Packing**: Uses the `IINT8Unpackable` interface to unpack weights from any source format (Q4_0, Q8_0, etc.) into the kernel's internal VNNI-optimized layout.
- **Quantization**: Handling the on-the-fly quantization of activation matrices (A) into `Q8_1Block` format.

### 3. Data Structures (`src/v2/tensors/BlockStructures.h`)
- **Activations (`Q8_1Block`)**:
    - Block size: 32 elements.
    - Content: 32 `int8_t` values (`qs`), 1 `float16` scale (`d`), and 1 `int16_t` sum (`sum_qs`).
    - **Purpose**: The `sum_qs` allows for efficient zero-point compensation during the dot product.
- **Weights (`Q8_1PackedWeights`)**:
    - Re-packed into a layout optimal for VNNI loading (e.g., [K/4][N][4]).
    - Contains pre-computed scales and compensation terms.
    - Populated via `IINT8Unpackable::unpack_block_to_int8()`.

## Algorithm & Math

The core operation is an Int8 dot product with scaling:

$$ C_{ij} = \sum_{k} (A_{ik} \times B_{kj}) \times \text{scale}_A \times \text{scale}_B $$

### 1. Quantization (A-Matrix)
Input FP32 activations ($A$) are quantized row-wise into blocks of 32:
- Find $max(|x|)$ in block.
- $d = max / 127.0$.
- $qs_i = x_i / d$.
- $sum\_qs = \sum qs_i$.

### 2. The VNNI Core
The JIT kernels use the `vpdpbusd` (Vector Packet Dot Product Bytes to Unsigned Doubleword) instruction:
- Loads 4 `int8` values from A and 4 `int8` values from B.
- Computes dot product.
- Accumulates into `int32` register.
- **Throughput**: 4x faster than FP32 FMA.

### 3. Compensation & Scaling
Since `vpdpbusd` operates on integers, the result must be dequantized:
$$ \text{Result}_{fp32} = \text{Accum}_{int32} \times d_A \times d_B + \text{Bias} $$
The kernel handles this efficiently by accumulating integer sums and applying floating-point scaling only at the very end of the K-loop (or block).

## Dynamic Features

The `Q8_1GemmKernel.h` orchestrator implements several dynamic features to maintain performance across model sizes:

### 1. Adaptive Blocking (Small Models)
For small matrices (e.g., Qwen 0.5B, N=896), standard blocking creates too few tasks to saturate high-core-count CPUs (e.g., 28 cores).
- **Logic**: Dynamically calculates block size to ensure `TaskCount > 4 * ThreadCount`.
- **Benefit**: +60% performance on 0.5B models.

### 2. K-Tiling (Large Models)
For large matrices (e.g., Qwen 32B, K=27,392), a single weight row is ~27KB. A standard block (64 rows) exceeds L2 cache (1MB).
- **Logic**: Splits the K-loop into tiles (e.g., 256KB).
- **Hardware Awareness**: Uses `CPUFeatures.h` to detect actual L2 and L3 cache sizes at runtime.
    - **L2 Constraint**: Ensures the working block fits in the per-core L2 cache (typically 1MB-2MB).
    - **L3 Constraint**: Ensures the total working set across all threads fits in the shared L3 cache.
- **Benefit**: Keeps weight tiles resident in L2 cache ("B-stationary"), preventing thrashing.
- **Result**: 4x performance boost on 32B models.

### 3. Parallel Quantization
For small batch sizes (M < Threads), the quantization of A (FP32 -> Int8) becomes a bottleneck if done serially.
- **Logic**: Parallelizes quantization over the K-dimension.

### 4. Optimized Generic Packing
The kernel uses a highly optimized generic packing routine (`pack_weights_generic`) that works with any tensor implementing `IINT8Unpackable`.
- **Logic**:
    - **Parallelization**: OpenMP parallelization over the N (row) dimension.
    - **Vectorization**: SIMD-accelerated summation for zero-point compensation.
    - **Memory**: Strided `memcpy` operations for efficient VNNI layout generation.
- **Benefit**: Enables support for Q4_0, Q8_0, Q8_1, and future formats with **zero performance penalty** compared to hand-written specialized packers.

## Advanced Fused Operations (New in V4)

To further reduce memory bandwidth and latency, the V4 kernel supports fusing common post-GEMM operations directly into the compute kernel.

### 1. Bias Addition
- **Feature**: Adds a bias vector to the result: $C_{ij} = \text{GEMM}_{ij} + \text{Bias}_j$.
- **Implementation**: Loaded and added in AVX512 registers immediately after dequantization, before storing to memory.
- **Benefit**: Eliminates a separate memory read/write pass for bias addition.

### 2. Attention Masking (ALiBi / Causal)
- **Feature**: Applies an additive mask to the result: $C_{ij} = C_{ij} + \text{Mask}_{ij}$.
- **Implementation**: The mask pointer is passed to the kernel. If non-null, the mask value corresponding to the $(i, j)$ position is loaded and added.
- **Benefit**: Essential for Attention layers, fusing the mask application (e.g., ALiBi slopes or causal mask) into the QKV projection or Attention Score computation.

### 3. Online Softmax Fusion
- **Feature**: Computes Softmax statistics (max and sum-exp) and optionally skips writing the raw output matrix $C$.
- **Logic**:
    1.  **Fused Max**: As each $C_{ij}$ is computed, update the row-wise maximum: $m_i = \max(m_i, C_{ij})$.
    2.  **Fused Sum**: Compute exponentials relative to the current max and accumulate: $s_i = \sum \exp(C_{ij} - m_i)$.
    3.  **Output Skip**: If `do_softmax=true`, the raw $C_{ij}$ values are **not** written to main memory. Only the statistics ($m_i, s_i$) are stored.
- **Benefit**:
    - **Latency**: 1.33x speedup for M=1 (Single Token).
    - **Bandwidth**: Saves $M \times N$ writes and reads.
    - **Context**: Critical for Attention layers where $C$ (Attention Scores) is transient and only the Softmax probability distribution is needed.

## Use Case

This kernel is the default engine for **CPU Inference** in Llaminar V2 when using quantized models. It is specifically designed for:
- **Input**: Any Quantized Weights implementing `IINT8Unpackable` (Q4_0, Q8_0, Q8_1, etc.) + FP32 Activations (quantized on-the-fly).
- **Hardware**: Intel CPUs with AVX512-VNNI (Skylake-X, Cascade Lake, Ice Lake, Sapphire Rapids).
- **Goal**: Maximize token generation speed (M=1) and prompt processing throughput (M=512).
