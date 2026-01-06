/**
 * @file QuantisedGemmJit_M1.h
 * @brief JIT-compiled AVX-512 VNNI GEMM kernel optimized for single-row (M=1) matrix multiplication.
 * @author David Sanftenberg
 *
 * @details
 * This file implements a high-performance JIT-compiled GEMM kernel for quantized matrix
 * multiplication using AVX-512 VNNI instructions. The kernel is specifically optimized for
 * the M=1 case (single query row), which is the critical path for autoregressive LLM decoding.
 *
 * ## Algorithm Overview
 *
 * The kernel computes: C[M×N] = A[M×K] × B[K×N] where:
 * - A is quantized to Q8_1 format (symmetric INT8 with per-block scale)
 * - B is pre-packed as Q4_0/IQ4_NL quantized weights with asymmetric correction
 *
 * ## Memory Layout
 *
 * **A (Activations)**: Q8_1Block format - 32 INT8 values + FP16 scale + INT16 sum_qs = 36 bytes
 * **B (Weights)**: Packed as QuantisedPackedWeights with separate:
 *   - `comp`: Compressed INT8 weight data [K/4][N][4]
 *   - `scales`: FP16 scales per K-block [K/32][N]
 *   - `mins`: FP16 asymmetric correction values [N] (optional, for IQ4_NL)
 *
 * ## Kernel Structure
 *
 * The kernel processes 64 output columns per iteration (4 ZMM registers × 16 FP32 values):
 *
 * ```
 * for each N in [0, N_dim) step 64:
 *     zmm_c0..zmm_c3 = 0  (4 × 16 FP32 accumulators)
 *     for each K in [0, K_dim) step 32:
 *         // Load A block (32 INT8 values)
 *         // Load B blocks (4 × 64 INT8 values packed)
 *         // VNNI: vpdpbusd accumulates INT8×INT8 → INT32
 *         // Apply asymmetric correction: result -= sum_qs × mins
 *         // Scale: result *= A_scale × B_scale
 *     // Fused post-ops: bias, mask, softmax, SwiGLU
 *     store zmm_c0..zmm_c3 to C[row, N:N+64]
 * ```
 *
 * ## Register Allocation
 *
 * | Register | Purpose |
 * |----------|---------|
 * | zmm0-3   | C accumulators (FP32, 64 output columns) |
 * | zmm4-7   | Temporary INT32 accumulators for VNNI |
 * | zmm8-11  | More INT32 accumulators |
 * | zmm12-15 | Workspace for post-ops |
 * | zmm16-17 | B scale values |
 * | zmm18    | Bias workspace |
 * | zmm19    | Mask workspace |
 * | zmm20-31 | Softmax/SwiGLU constants and temps |
 *
 * ## Fused Operations
 *
 * The kernel optionally fuses the following operations:
 * 1. **Bias Addition**: C += bias[N]
 * 2. **Attention Mask**: C += mask[M×N] (for causal masking)
 * 3. **Softmax**: Computes row-wise softmax (stores local max/sum for online softmax)
 * 4. **SwiGLU**: Computes gate * sigmoid(gate) * up (for FFN activation)
 *
 * ## Performance Considerations
 *
 * - Uses AVX-512 VNNI `vpdpbusd` for 4× INT8 dot products per cycle
 * - Processes 64 columns per iteration to maximize register utilization
 * - Asymmetric correction is computed incrementally to avoid extra passes
 * - Softmax uses polynomial exp() approximation (5th degree, max error < 0.001%)
 *
 * @see QuantisedGemmJit_M2 for the 2-row variant with better throughput for M≥2
 * @see QuantisedGemmKernel for the high-level kernel interface
 */

#pragma once

#include "../../../../../external/onednn/third_party/xbyak/xbyak.h"
#include "../../../tensors/Tensors.h"
#include <vector>
#include <cstdint>
#include <iostream>

namespace llaminar2
{
    namespace gemm_v4
    {

        /**
         * @struct QuantisedPackedWeights
         * @brief Pre-packed quantized weight matrix optimized for VNNI GEMM.
         *
         * @details
         * This structure holds quantized weights in a memory layout optimized for
         * AVX-512 VNNI instructions. The data is reorganized during packing to enable
         * efficient 4-wide INT8 dot products.
         *
         * ## Memory Layout
         *
         * The original Q4_0/IQ4_NL weights [K×N] are transformed to:
         *
         * ```
         * packed_data: [K/4][N][4] flattened to [K/4 * N * 4]
         *   - Groups of 4 consecutive K values for VNNI vpdpbusd
         *   - Each vpdpbusd processes 4 INT8×INT8 products simultaneously
         *
         * compensation: [K/32][N]
         *   - Pre-computed dot product of A's sum_qs with B for asymmetric correction
         *   - Eliminates the need to compute this at runtime
         *
         * scales: [K/32][N]
         *   - Per-block scale factors from original quantization
         *   - Applied after INT32 accumulation, before FP32 conversion
         *
         * mins: [K/32][N]
         *   - Asymmetric quantization offset (for IQ4_NL format)
         *   - Used in correction: result -= sum_qs × mins
         * ```
         *
         * ## Packing Example
         *
         * For K=128, N=64:
         * - Original B: [128][64] INT8
         * - Packed: [32][64][4] = 8192 bytes
         * - Scales: [4][64] = 256 floats
         * - Mins: [4][64] = 256 floats
         */
        struct QuantisedPackedWeights
        {
            /**
             * @brief Packed INT8 weight data in VNNI-friendly layout.
             *
             * Layout: [K/4][N][4] flattened. For vpdpbusd, we need 4 consecutive
             * INT8 values from the K dimension to compute 4 dot products in parallel.
             */
            std::vector<int8_t> packed_data;

            /**
             * @brief Pre-computed compensation values for asymmetric quantization.
             *
             * Layout: [K/32][N]. Each value is the sum of the corresponding weight
             * block, multiplied by the activation's sum_qs for asymmetric correction.
             */
            std::vector<int32_t> compensation;

            /**
             * @brief Per-block scale factors from weight quantization.
             *
             * Layout: [K/32][N]. After INT32 accumulation, the result is multiplied
             * by A_scale × B_scale to convert back to FP32.
             */
            std::vector<float> scales;

            /**
             * @brief Asymmetric quantization offsets (mins) for IQ4_NL format.
             *
             * Layout: [K/32][N]. For symmetric quantization (Q4_0, Q8_0), this is zero.
             * For asymmetric (IQ4_NL), this stores the minimum value used during
             * quantization, enabling: dequant = scale × quant + min.
             */
            std::vector<float> mins;

            /** @brief Original K dimension (number of input features). */
            int K;

            /** @brief Original N dimension (number of output features). */
            int N;

            /**
             * @brief Flag indicating if mins contains non-zero values.
             *
             * If false, mins vector may be empty or contain zeros, and the JIT kernel
             * can skip the asymmetric correction step for performance.
             */
            bool has_mins = false;
        };

        /**
         * @enum GemmOutputFormat
         * @brief Output format for quantized GEMM kernels
         *
         * Controls how the GEMM result is written to memory. FP32 is the default
         * for compatibility. Q8_1 enables fused requantization for bandwidth savings
         * in the Q8_1 activation pipeline. Q16_1 provides 256× more precision than
         * Q8_1 for high-dynamic-range outputs like K projections.
         */
        enum class GemmOutputFormat : uint8_t
        {
            FP32 = 0, ///< Output as FP32 (default, 4 bytes per value)
            Q8_1 = 1, ///< Output as Q8_1 blocks (fused requantization, ~1.125 bytes per value)
            BF16 = 2, ///< Output as BF16 (future, 2 bytes per value)
            FP16 = 3, ///< Output as FP16 (future, 2 bytes per value)
            Q16_1 = 4 ///< Output as Q16_1 blocks (high-precision quantization, ~2.25 bytes per value)
        };

        /**
         * @struct QuantisedGemmParams
         * @brief Runtime parameters passed to the JIT-compiled GEMM kernel.
         *
         * @details
         * This structure is passed by pointer to the generated JIT code. The kernel
         * reads these parameters at the start of execution and uses them throughout
         * the computation. The struct layout must match the offsets used in the
         * generated assembly code.
         *
         * ## Memory Offsets (used in generated assembly)
         *
         * | Field           | Offset | Size  | Description |
         * |-----------------|--------|-------|-------------|
         * | A               |   0    | 8     | Pointer to Q8_1 quantized activations |
         * | B_packed        |   8    | 8     | Pointer to packed INT8 weights |
         * | comp            |  16    | 8     | Pointer to compensation data (unused in M1) |
         * | scales          |  24    | 8     | Pointer to weight scales [K/32][N] |
         * | mins            |  32    | 8     | Pointer to asymmetric mins [N] or nullptr |
         * | C               |  40    | 8     | Pointer to output matrix (FP32) |
         * | K_blocks        |  48    | 4     | Number of K blocks (K/32) |
         * | N               |  52    | 4     | Output dimension N |
         * | ldc             |  56    | 4     | Leading dimension of C (stride between rows) |
         * | padding         |  60    | 4     | Alignment padding |
         * | bias            |  64    | 8     | Pointer to bias vector [N] or nullptr |
         * | mask            |  72    | 8     | Pointer to attention mask [M×N] or nullptr |
         * | local_max       |  80    | 8     | Pointer to store softmax local max |
         * | local_sum       |  88    | 8     | Pointer to store softmax local sum |
         * | do_softmax      |  96    | 1     | Flag to enable softmax computation |
         * | A_stride        | 100    | 4     | Stride between A rows (for M>1) |
         * | gate_input      | 104    | 8     | Pointer to gate tensor for SwiGLU |
         * | do_swiglu       | 112    | 1     | Flag to enable SwiGLU activation |
         * | output_format   | 113    | 1     | Output format (FP32=0, Q8_1=1, Q16_1=4) |
         * | C_q8_1          | 120    | 8     | Pointer to Q8_1 output buffer |
         * | C_q8_1_stride   | 128    | 4     | Stride between Q8_1 output rows (bytes) |
         * | C_q16_1         | 136    | 8     | Pointer to Q16_1 output buffer |
         * | C_q16_1_stride  | 144    | 4     | Stride between Q16_1 output rows (bytes) |
         * | q16_block_size  | 148    | 4     | Q16_1 block size (32, 64, or 128) |
         */
        struct QuantisedGemmParams
        {
            /** @brief Pointer to Q8_1 quantized activation matrix A[M×K]. */
            const void *A = nullptr;

            /** @brief Pointer to packed INT8 weight matrix B[K×N]. */
            const void *B_packed = nullptr;

            /** @brief Pointer to compensation values (pre-computed for asymmetric correction). */
            const void *comp = nullptr;

            /** @brief Pointer to weight scale factors [K/32][N]. */
            const void *scales = nullptr;

            /** @brief Pointer to asymmetric mins [N] or nullptr for symmetric quantization. */
            const void *mins = nullptr;

            /** @brief Pointer to output matrix C[M×N] (FP32). */
            float *C = nullptr;

            /** @brief Number of K blocks (K/32, where 32 is Q8_1 block size). */
            int K_blocks = 0;

            /** @brief Output dimension N (number of columns). */
            int N = 0;

            /** @brief Leading dimension of C (stride between rows in elements). */
            int ldc = 0;

            /** @brief Pointer to bias vector [N] or nullptr if no bias. */
            const float *bias = nullptr;

            /** @brief Pointer to attention mask [M×N] or nullptr if no mask. */
            const float *mask = nullptr;

            /**
             * @brief Pointer to store local max values for online softmax.
             *
             * For each 64-column block, stores the maximum value found. Used by
             * online softmax algorithm to compute global max across all blocks.
             * Size: ceil(N/64) floats.
             */
            float *local_max = nullptr;

            /**
             * @brief Pointer to store local sum values for online softmax.
             *
             * For each 64-column block, stores the sum of exp(x - local_max).
             * Used by online softmax algorithm to compute normalized probabilities.
             * Size: ceil(N/64) floats.
             */
            float *local_sum = nullptr;

            /** @brief Flag to enable softmax computation after GEMM. */
            bool do_softmax = false;

            /**
             * @brief Stride between rows of A in bytes.
             *
             * For M=1 kernel this is unused. For M=2 kernel, this specifies the
             * byte offset from row 0 to row 1 of the activation matrix.
             */
            int A_stride = 0;

            /**
             * @brief Pointer to gate tensor for SwiGLU activation.
             *
             * For FFN layers using SwiGLU, this points to the gate projection output.
             * The kernel computes: output = up * gate * sigmoid(gate).
             * Size: [M×N] floats.
             */
            const float *gate_input = nullptr;

            /** @brief Flag to enable SwiGLU activation after GEMM. */
            bool do_swiglu = false;

            // ================================================================
            // Output Format Control (for fused requantization)
            // ================================================================

            /**
             * @brief Output format for GEMM result.
             *
             * When set to Q8_1, the kernel performs fused requantization and writes
             * Q8_1 blocks to C_q8_1 instead of FP32 to C. This eliminates a separate
             * quantization pass for the Q8_1 activation pipeline.
             *
             * When set to Q16_1, the kernel outputs Q16_1 blocks for high-precision
             * quantization (256× more precision than Q8_1). This is critical for
             * K projections in HybridQ16 mode where high dynamic range would cause
             * Q8_1 to lose small values.
             */
            GemmOutputFormat output_format = GemmOutputFormat::FP32;

            /**
             * @brief Pointer to Q8_1 output buffer [M, N/32 blocks].
             *
             * Used when output_format == Q8_1. Each row contains N/32 Q8_1Block
             * structures (36 bytes each: FP16 scale + INT16 sum + 32×INT8 values).
             * When output_format == FP32, this field is ignored.
             */
            void *C_q8_1 = nullptr;

            /**
             * @brief Stride between output rows in bytes (for Q8_1 output).
             *
             * Specifies the byte offset from one output row to the next in C_q8_1.
             * Typically: (N / 32) * sizeof(Q8_1Block) = (N / 32) * 36.
             */
            int C_q8_1_stride = 0;

            // ================================================================
            // Q16_1 Output Parameters (for high-precision quantization)
            // ================================================================

            /**
             * @brief Pointer to Q16_1 output buffer [M, N/block_size blocks].
             *
             * Used when output_format == Q16_1. Block size is configurable via
             * q16_block_size (32, 64, or 128 elements). This enables 1-block-per-head
             * attention computation when block_size matches head_dim.
             *
             * Block structure (Q16_1Block_64 example, 136 bytes):
             *   - float d: FP32 scale factor (4 bytes)
             *   - int32_t sum_qs: Pre-computed sum of qs values (4 bytes)
             *   - int16_t qs[64]: 64 quantized INT16 values (128 bytes)
             */
            void *C_q16_1 = nullptr;

            /**
             * @brief Stride between output rows in bytes (for Q16_1 output).
             *
             * Specifies the byte offset from one output row to the next in C_q16_1.
             * Depends on block size:
             *   - 32-elem blocks: (N / 32) * 72 bytes
             *   - 64-elem blocks: (N / 64) * 136 bytes
             *   - 128-elem blocks: (N / 128) * 264 bytes
             */
            int C_q16_1_stride = 0;

            /**
             * @brief Q16_1 block size (elements per block: 32, 64, or 128).
             *
             * Should match model head_dim for optimal attention performance:
             *   - 64: Qwen2.5-0.5B, GPT-2 (1 block per head)
             *   - 128: Qwen3, Llama-3, Mistral (1 block per head)
             *   - 32: Legacy, GEMM compatibility
             *
             * The JIT kernel writes output as head_dim-sized blocks, enabling
             * 1-block-per-head integer attention without per-block scale tracking.
             */
            int q16_block_size = 64;
        };

        /**
         * @class QuantisedGemmJit_M1
         * @brief JIT-compiled GEMM kernel for single-row (M=1) quantized matrix multiplication.
         *
         * @details
         * This class uses the Xbyak JIT assembler to generate optimized AVX-512 VNNI
         * assembly code at runtime. The generated code is specialized for M=1 (single query
         * row), which is the dominant case during autoregressive LLM decoding.
         *
         * ## Code Generation
         *
         * The constructor calls `generate()` which emits x86-64 assembly code into a
         * buffer. The generated code is then callable as a function pointer via `getCode()`.
         *
         * ## Buffer Size
         *
         * The `4096 * 32` (128KB) buffer size is chosen to accommodate:
         * - Prologue/epilogue code (~200 bytes)
         * - N-loop with K-loop (~2KB)
         * - Softmax code with exp() polynomial (~1KB)
         * - SwiGLU code with sigmoid() polynomial (~1KB)
         * - Total with safety margin: ~128KB
         *
         * ## Usage Example
         *
         * ```cpp
         * // Create kernel (generates JIT code)
         * QuantisedGemmJit_M1 kernel;
         *
         * // Set up parameters
         * QuantisedGemmParams params = {...};
         *
         * // Get function pointer and call
         * using KernelFunc = void (*)(const QuantisedGemmParams*);
         * auto fn = kernel.getCode<KernelFunc>();
         * fn(&params);
         * ```
         *
         * @see QuantisedGemmParams for parameter structure
         * @see QuantisedPackedWeights for weight packing format
         */
        class QuantisedGemmJit_M1 : public Xbyak::CodeGenerator
        {
        public:
            /**
             * @brief Construct and generate the JIT kernel.
             *
             * Allocates a 128KB code buffer and generates optimized AVX-512 VNNI
             * assembly code for M=1 quantized GEMM with optional fused operations.
             */
            QuantisedGemmJit_M1() : Xbyak::CodeGenerator(4096 * 32)
            {
                generate();
            }

            /**
             * @brief Function pointer type for the generated kernel.
             *
             * The kernel takes a pointer to QuantisedGemmParams and computes
             * C = A × B with optional fused operations.
             */
            using kernel_func_t = void (*)(const QuantisedGemmParams *params);

            /**
             * @brief Get the function pointer to the generated JIT code.
             * @return Function pointer that can be called to execute the kernel.
             */
            kernel_func_t get_kernel()
            {
                return getCode<kernel_func_t>();
            }

            /**
             * @brief Pack weights into VNNI-friendly layout.
             * @param B Source INT8 weight matrix [K×N]
             * @param K Number of input features (rows of B)
             * @param N Number of output features (columns of B)
             * @param packed Output packed weight structure
             *
             * @note This is a stub implementation. The real packing logic is in
             *       QuantisedGemmKernel::pack_weights() which handles the actual
             *       memory layout transformation for VNNI instructions.
             */
            static void pack_weights(const int8_t *B, int K, int N, QuantisedPackedWeights &packed)
            {
                packed.K = K;
                packed.N = N;
                packed.packed_data.resize(K * N);
                packed.compensation.resize((K / 32) * N);
                packed.scales.resize((K / 32) * N);
            }

        private:
            /**
             * @brief Generate the JIT assembly code for M=1 quantized GEMM.
             *
             * @details
             * This method emits x86-64 assembly code using Xbyak. The generated code
             * implements an optimized GEMM kernel with the following structure:
             *
             * ## High-Level Algorithm
             *
             * ```
             * void kernel(const QuantisedGemmParams* params) {
             *     for (n = 0; n < N; n += 64) {           // N-loop: 64 columns per iteration
             *         C_acc[0:64] = 0;                     // 4 ZMM accumulators
             *         for (k = 0; k < K; k += 32) {        // K-loop: 32 elements per Q8_1 block
             *             // 1. Load A block (32 INT8 values)
             *             // 2. Load B blocks (4 × 16 × 4 INT8 values packed)
             *             // 3. VNNI accumulation: vpdpbusd (4×INT8 dot product → INT32)
             *             // 4. Asymmetric correction: acc -= sum_qs × mins
             *             // 5. Scale and convert to FP32
             *         }
             *         // Fused post-ops
             *         if (bias) C_acc += bias[n:n+64];
             *         if (mask) C_acc += mask[n:n+64];
             *         if (softmax) compute_local_max_sum(C_acc);
             *         if (swiglu) C_acc = swiglu(C_acc, gate[n:n+64]);
             *         store C_acc to C[0, n:n+64]
             *     }
             * }
             * ```
             *
             * ## Register Allocation
             *
             * ### General Purpose Registers
             * | Register | Purpose |
             * |----------|---------|
             * | rdi      | params->A (activation pointer) |
             * | rsi      | params->B_packed (weight pointer) |
             * | rdx      | params->comp (compensation pointer) |
             * | rcx      | params->scales (scale pointer) |
             * | r8       | params->C (output pointer) |
             * | r9       | params->K_blocks |
             * | rax      | Temporary / params reload |
             * | rbx      | Stride (N * 4 bytes) |
             * | r10      | N loop counter |
             * | r11      | K loop counter |
             * | r12      | B cursor |
             * | r13      | Compensation cursor |
             * | r14      | C cursor |
             * | r15      | A cursor |
             * | rbp      | Scales cursor |
             *
             * ### ZMM Registers (512-bit)
             * | Register | Purpose |
             * |----------|---------|
             * | zmm0-3   | FP32 accumulators (64 output columns) |
             * | zmm4-7   | INT32 temp accumulators for VNNI |
             * | zmm8     | Scale broadcast |
             * | zmm9     | Constant: 128 (for unsigned→signed conversion) |
             * | zmm10    | Constant: -128 |
             * | zmm11    | A values (broadcast from Q8_1 block) |
             * | zmm12-15 | B values (4 × 16 INT8) |
             * | zmm16    | Temporary for half-precision conversion |
             * | zmm17    | B scale values |
             * | zmm18    | Bias workspace |
             * | zmm19    | Mask workspace |
             * | zmm20-31 | Softmax/SwiGLU constants and temps |
             *
             * ## VNNI Instruction Usage
             *
             * The kernel uses `vpdpbusd` (Vector Packed Dot Product of Unsigned and
             * Signed Bytes with Dword Accumulation):
             *
             * ```asm
             * vpdpbusd zmm_dst, zmm_a, zmm_b
             * // For each group of 4 bytes at position i:
             * // dst[i] += a[i*4+0]*b[i*4+0] + a[i*4+1]*b[i*4+1]
             * //         + a[i*4+2]*b[i*4+2] + a[i*4+3]*b[i*4+3]
             * ```
             *
             * Since Q8_1 uses signed INT8 (-128 to 127), but vpdpbusd expects unsigned
             * A operand, we add 128 to A values and apply asymmetric correction:
             * - Original: A × B
             * - With unsigned conversion: (A + 128) × B - 128 × sum(B)
             */
            void generate()
            {
                using namespace Xbyak;

                // ==================== Register Definitions ====================
                // System V AMD64 ABI: First 6 integer args in rdi, rsi, rdx, rcx, r8, r9

                // Parameter registers (from params struct)
                const Reg64 &reg_A = rdi;       ///< params->A: Q8_1 activation pointer
                const Reg64 &reg_B = rsi;       ///< params->B_packed: packed weight pointer
                const Reg64 &reg_Comp = rdx;    ///< params->comp: compensation pointer
                const Reg64 &reg_Scales = rcx;  ///< params->scales: scale pointer
                const Reg64 &reg_C = r8;        ///< params->C: output pointer
                const Reg64 &reg_K_blocks = r9; ///< params->K_blocks: number of K blocks
                // Stack layout after prologue:
                //   [rsp]      = N value (pushed)
                //   [rsp + 8]  = params_ptr (pushed)
                //   [rsp + 16..271] = 256-byte zero buffer (for null mins fallback)
                //   [rsp + 264] = mins_stride (stored, overlaps end of zero buffer)

                // Working registers (callee-saved, pushed in prologue)
                const Reg64 &reg_tmp = rax;           ///< General temporary register
                const Reg64 &reg_stride = rbx;        ///< Stride = N * sizeof(float)
                const Reg64 &reg_loop_N = r10;        ///< N-loop counter
                const Reg64 &reg_loop_K = r11;        ///< K-loop counter
                const Reg64 &reg_B_cursor = r12;      ///< Current B position in K-loop
                const Reg64 &reg_Comp_cursor = r13;   ///< Current compensation position
                const Reg64 &reg_C_cursor = r14;      ///< Current C position in N-loop
                const Reg64 &reg_A_cursor = r15;      ///< Current A position in K-loop
                const Reg64 &reg_Scales_cursor = rbp; ///< Current scales position

                const Reg32 &reg_tmp_32 = eax; ///< 32-bit temp for immediate loading

                // ==================== ZMM Register Allocation ====================

                // C accumulators: 4 ZMM = 64 FP32 output columns per N-loop iteration
                const Zmm &zmm_c0 = zmm0; ///< C[0, n+0:n+16]
                const Zmm &zmm_c1 = zmm1; ///< C[0, n+16:n+32]
                const Zmm &zmm_c2 = zmm2; ///< C[0, n+32:n+48]
                const Zmm &zmm_c3 = zmm3; ///< C[0, n+48:n+64]

                // INT32 temp accumulators for VNNI (before scale application)
                const Zmm &zmm_acc0 = zmm4; ///< INT32 accumulator for columns 0-15
                const Zmm &zmm_acc1 = zmm5; ///< INT32 accumulator for columns 16-31
                const Zmm &zmm_acc2 = zmm6; ///< INT32 accumulator for columns 32-47
                const Zmm &zmm_acc3 = zmm7; ///< INT32 accumulator for columns 48-63

                // Constants and workspace
                const Zmm &zmm_scale = zmm8;    ///< Broadcast A scale (FP32 from FP16)
                const Ymm &ymm_scale = ymm8;    ///< Half-width alias for scale conversion
                const Zmm &zmm_128 = zmm9;      ///< Constant: 128 for unsigned conversion
                const Zmm &zmm_neg_128 = zmm10; ///< Constant: -128 for correction
                const Zmm &zmm_a = zmm11;       ///< A values broadcast (32 INT8)
                const Zmm &zmm_b0 = zmm12;      ///< B block 0 (16 × 4 INT8)
                const Zmm &zmm_b1 = zmm13;      ///< B block 1
                const Zmm &zmm_b2 = zmm14;      ///< B block 2
                const Zmm &zmm_b3 = zmm15;      ///< B block 3
                const Ymm &ymm_tmp = ymm16;     ///< Temp for FP16→FP32 conversion
                const Zmm &zmm_scale_b = zmm17; ///< B scale values broadcast
                const Zmm &zmm_bias = zmm18;    ///< Bias vector workspace
                const Zmm &zmm_mask = zmm19;    ///< Attention mask workspace
                // zmm20-31 reserved for softmax/SwiGLU constants and temps

                // ==================== PROLOGUE ====================
                // Save callee-saved registers per System V AMD64 ABI
                push(rbx);
                push(rbp);
                push(r12);
                push(r13);
                push(r14);
                push(r15);

                // ==================== ZERO BUFFER ALLOCATION ====================
                // Allocate 256 bytes (64 floats) of zeros on stack
                // Used when mins pointer is null (symmetric quantization)
                // This avoids conditional branches in the hot K-loop
                sub(rsp, 256);
                vpxord(zmm0, zmm0, zmm0);         // Zero register
                vmovups(ptr[rsp + 0 * 64], zmm0); // Store 64 bytes of zeros
                vmovups(ptr[rsp + 1 * 64], zmm0); // ...
                vmovups(ptr[rsp + 2 * 64], zmm0); // ...
                vmovups(ptr[rsp + 3 * 64], zmm0); // Total: 256 bytes

                // Save params pointer to stack (we need it for bias/mask)
                // NOTE: Stack offsets after this point are shifted by 256 bytes
                const Reg64 &reg_params = rdi;
                push(reg_params);

                // Load arguments from params struct
                // rdi is params
                mov(reg_B, ptr[reg_params + 8]);       // B_packed
                mov(reg_Comp, ptr[reg_params + 16]);   // comp
                mov(reg_Scales, ptr[reg_params + 24]); // scales
                // mins is at +32
                mov(reg_C, ptr[reg_params + 40]);                // C
                mov(reg_K_blocks.cvt32(), ptr[reg_params + 48]); // K_blocks

                // ==================== STACK FRAME SETUP ====================
                // Stack layout (from current rsp after params push):
                //   [rsp] = params_ptr (just pushed)
                //   [rsp + 8..263] = 256-byte zero buffer
                //   [rsp + 264] = mins_stride (stored via mov)

                // Push N to stack for later comparison in N-loop
                mov(reg_loop_N.cvt32(), ptr[reg_params + 52]); // N
                push(reg_loop_N);                              // Push N to stack
                // Now: [rsp] = N, [rsp + 8] = params_ptr, [rsp + 16..271] = zero buffer

                // Calculate stride in bytes for B and scales arrays
                mov(reg_stride.cvt32(), ptr[reg_params + 56]); // ldc
                shl(reg_stride, 2);                            // ldc * 4 (stride in bytes)

                // ==================== MINS POINTER HANDLING ====================
                // If mins is null (symmetric quantization), use stride=0 to keep reading zeros
                mov(rax, ptr[reg_params + 32]); // Load mins pointer
                test(rax, rax);                 // Check if null
                mov(rax, reg_stride);           // Preserve stride in rax
                Label mins_stride_ok;
                jnz(mins_stride_ok, T_NEAR); // Jump if mins is not null
                xor_(rax, rax);              // Set mins stride to 0
                L(mins_stride_ok);
                // Store mins stride at [rsp + 264] (inside zero buffer area, ok since we only need 256 bytes for loads)
                mov(ptr[rsp + 264], rax);

                // Load A last (overwrites params pointer in rdi)
                mov(reg_A, ptr[reg_params + 0]); // A

                // ==================== CONSTANT INITIALIZATION ====================
                // For vpdpbusd, first operand (A) must be unsigned (0-255)
                // Q8_1 uses signed INT8 (-128 to 127), so we add 128 to make unsigned
                // This requires asymmetric correction: result -= 128 × sum(B_block)
                mov(reg_tmp_32, 0x80808080);       // 128 in each byte
                vpbroadcastd(zmm_128, reg_tmp_32); // Broadcast to all 64 bytes

                mov(reg_tmp_32, -128);
                vpbroadcastd(zmm_neg_128, reg_tmp_32);

                // ==================== N-LOOP (64 columns per iteration) ====================
                xor_(reg_loop_N, reg_loop_N); // Initialize N counter to 0

                Label loop_N_label;
                L(loop_N_label);
                {
                    // Check if we have processed all N columns
                    cmp(reg_loop_N, ptr[rsp]); // Compare with N stored on stack (after zero buffer)
                    jge("end_kernel", T_NEAR); // Exit if loop_N >= N

                    // Reset A cursor to start of A for this N-block
                    mov(reg_A_cursor, reg_A);

                    // ==================== CURSOR SETUP FOR N-BLOCK ====================
                    // Reload params pointer (rdi was overwritten by A)
                    mov(rax, ptr[rsp + 8]);
                    mov(reg_Comp, ptr[rax + 16]);   // Restore comp pointer
                    mov(reg_Scales, ptr[rax + 24]); // Restore scales pointer

                    // Calculate byte offset for current N position
                    mov(reg_tmp, reg_loop_N);
                    shl(reg_tmp, 2); // loop_N * sizeof(float) = loop_N * 4

                    // Set up cursors to point to current N-block
                    mov(reg_Comp_cursor, reg_Comp);
                    add(reg_Comp_cursor, reg_tmp); // comp[loop_N]

                    mov(reg_Scales_cursor, reg_Scales);
                    add(reg_Scales_cursor, reg_tmp); // scales[loop_N]

                    mov(reg_C_cursor, reg_C);
                    add(reg_C_cursor, reg_tmp); // C[0, loop_N]

                    // ==================== LOAD INITIAL C VALUES ====================
                    // For beta=1 semantics: C = alpha*A*B + beta*C
                    // Load existing C values as initial accumulator values
                    //
                    // IMPORTANT: When output_format is Q8_1 or Q16_1, C pointer is nullptr.
                    // In that case, we zero-initialize the accumulators instead.
                    mov(rax, ptr[rsp + 8]); // Reload params (reg_tmp=rax was clobbered)
                    cmp(byte[rax + 113], static_cast<uint8_t>(GemmOutputFormat::Q8_1));
                    Label zero_accumulators;
                    je(zero_accumulators, T_NEAR);
                    cmp(byte[rax + 113], static_cast<uint8_t>(GemmOutputFormat::Q16_1));
                    Label load_c_values;
                    jne(load_c_values, T_NEAR); // If not Q8_1 or Q16_1, load from C

                    L(zero_accumulators);
                    {
                        // Quantized output: Zero-initialize accumulators
                        vxorps(zmm_c0, zmm_c0, zmm_c0);
                        vxorps(zmm_c1, zmm_c1, zmm_c1);
                        vxorps(zmm_c2, zmm_c2, zmm_c2);
                        vxorps(zmm_c3, zmm_c3, zmm_c3);
                        jmp("after_c_load", T_NEAR);
                    }
                    L(load_c_values);
                    vmovups(zmm_c0, ptr[reg_C_cursor + 0 * 64]); // C[0, n+0:n+16]
                    vmovups(zmm_c1, ptr[reg_C_cursor + 1 * 64]); // C[0, n+16:n+32]
                    vmovups(zmm_c2, ptr[reg_C_cursor + 2 * 64]); // C[0, n+32:n+48]
                    vmovups(zmm_c3, ptr[reg_C_cursor + 3 * 64]); // C[0, n+48:n+64]
                    L("after_c_load");

                    // ==================== MINS CURSOR FOR ASYMMETRIC CORRECTION ====================
                    // Repurpose reg_C_cursor (r14) temporarily for mins pointer
                    mov(rax, ptr[rsp + 8]);  // Reload params
                    mov(r14, ptr[rax + 32]); // mins pointer (offset 32)

                    // Null check: if mins is nullptr, use stack zero buffer
                    test(r14, r14);
                    Label mins_null, mins_ready;
                    jz(mins_null, T_NEAR); // Jump if mins is null

                    // Mins is valid: add offset
                    mov(reg_tmp, reg_loop_N);
                    shl(reg_tmp, 2);   // loop_N * 4
                    add(r14, reg_tmp); // r14 = &mins[loop_N]
                    jmp(mins_ready, T_NEAR);

                    L(mins_null);
                    lea(r14, ptr[rsp + 16]); // Point to stack zero buffer (at rsp + 16, after 2 pushes)

                    L(mins_ready);
                    // Now r14 is reg_Mins_cursor (either valid or pointing to zeros)

                    // B cursor calculation: (loop_N / 64) * (K_blocks * 2048)
                    mov(reg_tmp, reg_loop_N);
                    shr(reg_tmp, 6);             // loop_N / 64
                    imul(reg_tmp, reg_K_blocks); // * K_blocks
                    shl(reg_tmp, 11);            // * 2048

                    mov(reg_B_cursor, reg_B);
                    add(reg_B_cursor, reg_tmp);

                    // Loop over K blocks
                    mov(reg_loop_K, reg_K_blocks);

                    // ==================== K-LOOP (32 elements per Q8_1 block) ====================
                    // The K-loop processes one Q8_1 block (32 INT8 values) per iteration.
                    // Each Q8_1 block has: [d:FP16, sum_qs:INT16, qs[32]:INT8] = 36 bytes

                    Label loop_K_label;
                    L(loop_K_label);
                    {
                        // ==================== LOAD A SCALE (FP16 → FP32) ====================
                        // Q8_1Block layout: d (FP16 scale) at offset 0
                        vpbroadcastw(ymm_tmp, ptr[reg_A_cursor]); // Load FP16 scale
                        vcvtph2ps(zmm_scale, ymm_tmp);            // Convert to FP32 broadcast

                        // ==================== ASYMMETRIC CORRECTION (IQ4_NL Support) ====================
                        // For IQ4_NL format, weights use asymmetric quantization:
                        //   dequant = scale × quant + min
                        // This requires correction: result += sum_qs × A_scale × mins
                        //
                        // Q8_1Block layout: sum_qs (INT16) at offset 2
                        // sum_qs = sum of all 32 INT8 values in the block (used for correction)

                        vpbroadcastw(ymm_tmp, ptr[reg_A_cursor + 2]); // Load sum_qs as INT16
                        vpmovsxwd(zmm20, ymm_tmp);                    // Sign-extend INT16 → INT32
                        vcvtdq2ps(zmm20, zmm20);                      // Convert INT32 → FP32

                        // Multiply by A scale: zmm20 = sum_qs × A_scale
                        vmulps(zmm20, zmm20, zmm_scale);

                        // Load mins for current N-block (4 × 16 = 64 floats)
                        // mins points to asymmetric offset values from IQ4_NL dequantization
                        vmovups(zmm21, ptr[r14 + 0 * 64]); // mins[n+0:n+16]
                        vmovups(zmm22, ptr[r14 + 1 * 64]); // mins[n+16:n+32]
                        vmovups(zmm23, ptr[r14 + 2 * 64]); // mins[n+32:n+48]
                        vmovups(zmm24, ptr[r14 + 3 * 64]); // mins[n+48:n+64]

                        // Compute correction: (sum_qs × A_scale) × mins
                        vmulps(zmm21, zmm21, zmm20);
                        vmulps(zmm22, zmm22, zmm20);
                        vmulps(zmm23, zmm23, zmm20);
                        vmulps(zmm24, zmm24, zmm20);

                        // Add correction to C accumulators
                        vaddps(zmm_c0, zmm_c0, zmm21);
                        vaddps(zmm_c1, zmm_c1, zmm22);
                        vaddps(zmm_c2, zmm_c2, zmm23);
                        vaddps(zmm_c3, zmm_c3, zmm24);

                        // Advance mins cursor to next K-block
                        // Use stored stride (0 for symmetric weights, reg_stride for asymmetric)
                        add(r14, ptr[rsp + 264]);

                        // ==================== COMPENSATION FOR UNSIGNED CONVERSION ====================
                        // vpdpbusd expects unsigned A operand, but Q8_1 uses signed INT8.
                        // We add 128 to A values to make them unsigned (0-255 range).
                        // This requires correction: result -= 128 × sum(B_block)
                        // The compensation array stores pre-computed sum(B_block) for each N column.

                        vmovups(zmm_acc0, ptr[reg_Comp_cursor + 0 * 64]); // compensation[n+0:n+16]
                        vmovups(zmm_acc1, ptr[reg_Comp_cursor + 1 * 64]); // compensation[n+16:n+32]
                        vmovups(zmm_acc2, ptr[reg_Comp_cursor + 2 * 64]); // compensation[n+32:n+48]
                        vmovups(zmm_acc3, ptr[reg_Comp_cursor + 3 * 64]); // compensation[n+48:n+64]

                        // Multiply by -128 to compute: -128 × sum(B_block)
                        // Using INT32 multiply here because compensation values are INT32
                        vpmulld(zmm_acc0, zmm_acc0, zmm_neg_128);
                        vpmulld(zmm_acc1, zmm_acc1, zmm_neg_128);
                        vpmulld(zmm_acc2, zmm_acc2, zmm_neg_128);
                        vpmulld(zmm_acc3, zmm_acc3, zmm_neg_128);

                        // ==================== VNNI INNER LOOP (8 × 4 = 32 elements) ====================
                        // Process 32 INT8 values from A in 8 iterations of 4 values each.
                        // Q8_1Block layout: qs[32] (INT8 values) at offset 4
                        //
                        // vpdpbusd instruction computes 4 dot products in parallel:
                        //   acc[i] += a[i*4+0]*b[i*4+0] + a[i*4+1]*b[i*4+1]
                        //           + a[i*4+2]*b[i*4+2] + a[i*4+3]*b[i*4+3]

                        for (int i = 0; i < 8; ++i)
                        {
                            // Load 4 consecutive INT8 values from A and broadcast
                            vpbroadcastd(zmm_a, ptr[reg_A_cursor + 4 + i * 4]);

                            // Convert signed INT8 (-128..127) to unsigned (0..255) for vpdpbusd
                            // XOR with 0x80808080 adds 128 to each byte
                            vpxord(zmm_a, zmm_a, zmm_128);

                            // Load B blocks (4 × 16 × 4 = 256 bytes, already in VNNI layout)
                            // B is pre-packed so each ZMM load gets 16 columns × 4 K values
                            vmovups(zmm_b0, ptr[reg_B_cursor + 0 * 64]);
                            vpdpbusd(zmm_acc0, zmm_a, zmm_b0); // acc0 += dot(a, b0)

                            vmovups(zmm_b1, ptr[reg_B_cursor + 1 * 64]);
                            vpdpbusd(zmm_acc1, zmm_a, zmm_b1); // acc1 += dot(a, b1)

                            vmovups(zmm_b2, ptr[reg_B_cursor + 2 * 64]);
                            vpdpbusd(zmm_acc2, zmm_a, zmm_b2); // acc2 += dot(a, b2)

                            vmovups(zmm_b3, ptr[reg_B_cursor + 3 * 64]);
                            vpdpbusd(zmm_acc3, zmm_a, zmm_b3); // acc3 += dot(a, b3)

                            // Advance B cursor: 4 blocks × 64 bytes = 256 bytes
                            add(reg_B_cursor, 256);
                        }

                        // ==================== SCALE APPLICATION ====================
                        // Convert INT32 accumulator to FP32, then multiply by scales
                        vcvtdq2ps(zmm_acc0, zmm_acc0);
                        vcvtdq2ps(zmm_acc1, zmm_acc1);
                        vcvtdq2ps(zmm_acc2, zmm_acc2);
                        vcvtdq2ps(zmm_acc3, zmm_acc3);

                        // Load B scales and multiply: acc × B_scale
                        // B scales are per K-block, pre-loaded for current N-block
                        vmovups(zmm_scale_b, ptr[reg_Scales_cursor + 0 * 64]);
                        vmulps(zmm_acc0, zmm_acc0, zmm_scale_b);

                        vmovups(zmm_scale_b, ptr[reg_Scales_cursor + 1 * 64]);
                        vmulps(zmm_acc1, zmm_acc1, zmm_scale_b);

                        vmovups(zmm_scale_b, ptr[reg_Scales_cursor + 2 * 64]);
                        vmulps(zmm_acc2, zmm_acc2, zmm_scale_b);

                        vmovups(zmm_scale_b, ptr[reg_Scales_cursor + 3 * 64]);
                        vmulps(zmm_acc3, zmm_acc3, zmm_scale_b);

                        // Multiply by A scale: (acc × B_scale) × A_scale
                        // This completes the dequantization: result = A_scale × B_scale × int_acc
                        vmulps(zmm_acc0, zmm_acc0, zmm_scale);
                        vmulps(zmm_acc1, zmm_acc1, zmm_scale);
                        vmulps(zmm_acc2, zmm_acc2, zmm_scale);
                        vmulps(zmm_acc3, zmm_acc3, zmm_scale);

                        // Accumulate to C (C += scaled_acc for this K-block)
                        vaddps(zmm_c0, zmm_c0, zmm_acc0);
                        vaddps(zmm_c1, zmm_c1, zmm_acc1);
                        vaddps(zmm_c2, zmm_c2, zmm_acc2);
                        vaddps(zmm_c3, zmm_c3, zmm_acc3);

                        // ==================== ADVANCE POINTERS ====================
                        add(reg_A_cursor, 36); // Advance A by sizeof(Q8_1Block) = 36 bytes

                        // Advance Comp and Scales cursors by stride (N × sizeof(float))
                        // These arrays are laid out as [K/32][N], so stride = N × 4
                        add(reg_Comp_cursor, reg_stride);
                        add(reg_Scales_cursor, reg_stride);

                        // Decrement K counter and loop
                        dec(reg_loop_K);
                        jnz(loop_K_label, T_NEAR);
                    }

                    // ==================== RESTORE C CURSOR ====================
                    // We repurposed r14 for mins cursor during K-loop, restore it now
                    mov(rax, ptr[rsp + 8]);           // Reload params
                    mov(reg_C_cursor, ptr[rax + 40]); // C pointer
                    mov(reg_tmp, reg_loop_N);
                    shl(reg_tmp, 2);            // loop_N × sizeof(float)
                    add(reg_C_cursor, reg_tmp); // C cursor = C + loop_N × 4

                    // ==================== FUSED BIAS ADDITION ====================
                    // Optional: Add bias vector to output
                    // C[i] += bias[i] for i in [n, n+64)
                    mov(rax, ptr[rsp + 8]);  // Reload params
                    mov(rdx, ptr[rax + 64]); // bias pointer (offset 64)
                    test(rdx, rdx);          // Check if bias is nullptr
                    Label skip_bias;
                    jz(skip_bias, T_NEAR); // Skip if no bias
                    {
                        // Calculate address: bias + loop_N × sizeof(float)
                        lea(rdx, ptr[rdx + reg_loop_N * 4]);

                        // Load and add bias for 64 columns
                        vmovups(zmm_bias, ptr[rdx + 0 * 64]);
                        vaddps(zmm_c0, zmm_c0, zmm_bias);

                        vmovups(zmm_bias, ptr[rdx + 1 * 64]);
                        vaddps(zmm_c1, zmm_c1, zmm_bias);

                        vmovups(zmm_bias, ptr[rdx + 2 * 64]);
                        vaddps(zmm_c2, zmm_c2, zmm_bias);

                        vmovups(zmm_bias, ptr[rdx + 3 * 64]);
                        vaddps(zmm_c3, zmm_c3, zmm_bias);
                    }
                    L(skip_bias);

                    // ==================== FUSED ATTENTION MASK ====================
                    // Optional: Add attention mask to output (for causal masking)
                    // C[i] += mask[i] for i in [n, n+64)
                    // For M=1, mask is [1×N], indexed same as bias
                    mov(rdx, ptr[rax + 72]); // mask pointer (offset 72)
                    test(rdx, rdx);          // Check if mask is nullptr
                    Label skip_mask;
                    jz(skip_mask, T_NEAR); // Skip if no mask
                    {
                        // Calculate address: mask + loop_N × sizeof(float)
                        lea(rdx, ptr[rdx + reg_loop_N * 4]);

                        // Load and add mask for 64 columns
                        vmovups(zmm_mask, ptr[rdx + 0 * 64]);
                        vaddps(zmm_c0, zmm_c0, zmm_mask);

                        vmovups(zmm_mask, ptr[rdx + 1 * 64]);
                        vaddps(zmm_c1, zmm_c1, zmm_mask);

                        vmovups(zmm_mask, ptr[rdx + 2 * 64]);
                        vaddps(zmm_c2, zmm_c2, zmm_mask);

                        vmovups(zmm_mask, ptr[rdx + 3 * 64]);
                        vaddps(zmm_c3, zmm_c3, zmm_mask);
                    }
                    L(skip_mask);

                    // ==================== FUSED ONLINE SOFTMAX ====================
                    // Optional: Compute local max and sum(exp(x - max)) for online softmax
                    // This is used for attention score normalization. The kernel computes
                    // local max and local sum for each 64-column block. A separate pass
                    // combines these to compute the global softmax.
                    //
                    // Algorithm for this block:
                    //   1. max_local = max(C[n:n+64])
                    //   2. sum_local = sum(exp(C[i] - max_local)) for i in [n, n+64)
                    //   3. Store max_local and sum_local for later normalization
                    //
                    // The exp() function uses a 5th-degree polynomial approximation:
                    //   exp(x) ≈ (c5*t + c4)*t + c3)*t + c2)*t + c1) * 2^floor(x/ln2)
                    // where t = x/ln2 - floor(x/ln2) (fractional part)
                    mov(al, ptr[rax + 96]); // do_softmax flag (offset 96)
                    test(al, al);
                    Label skip_softmax;
                    jz(skip_softmax, T_NEAR);
                    {
                        // ==================== SOFTMAX REGISTER ALLOCATION ====================
                        const Zmm &zmm_max = zmm20;      ///< Running maximum value
                        const Zmm &zmm_sum = zmm21;      ///< Sum of exp(x - max)
                        const Zmm &zmm_tmp = zmm22;      ///< Temporary workspace
                        const Zmm &zmm_c1_const = zmm23; ///< Polynomial coeff c1 = 0.99999994
                        const Zmm &zmm_c2_const = zmm24; ///< Polynomial coeff c2 = ln(2) = 0.69314718
                        const Zmm &zmm_c3_coeff = zmm25; ///< Polynomial coeff c3 = 0.24022651
                        const Zmm &zmm_c4_const = zmm26; ///< Polynomial coeff c4 = 0.05550411
                        const Zmm &zmm_c5_const = zmm27; ///< Polynomial coeff c5 = 0.00961813
                        const Zmm &zmm_inv_ln2 = zmm28;  ///< 1/ln(2) = 1.44269504
                        const Zmm &zmm_max_clip = zmm29; ///< Upper clamp = 10.0f
                        const Zmm &zmm_min_clip = zmm30; ///< Lower clamp = -20.0f
                        const Zmm &zmm_127 = zmm31;      ///< Constant 127 for IEEE754 exponent

                        // ==================== LOAD POLYNOMIAL CONSTANTS ====================
                        // exp(x) polynomial: p(t) = c5*t^4 + c4*t^3 + c3*t^2 + c2*t + c1
                        // where t = frac(x / ln2)
                        mov(reg_tmp_32, 0x3fb8aa3b); // 1.44269504 (1/ln2)
                        vpbroadcastd(zmm_inv_ln2, reg_tmp_32);

                        mov(reg_tmp_32, 0x3f7ffffe); // 0.99999994 (c1)
                        vpbroadcastd(zmm_c1_const, reg_tmp_32);

                        mov(reg_tmp_32, 0x3f317218); // 0.69314718 (c2 = ln2)
                        vpbroadcastd(zmm_c2_const, reg_tmp_32);

                        mov(reg_tmp_32, 0x3e75fdf1); // 0.24022651 (c3)
                        vpbroadcastd(zmm_c3_coeff, reg_tmp_32);

                        mov(reg_tmp_32, 0x3d6356eb); // 0.05550411 (c4)
                        vpbroadcastd(zmm_c4_const, reg_tmp_32);

                        mov(reg_tmp_32, 0x3c1d9422); // 0.00961813 (c5)
                        vpbroadcastd(zmm_c5_const, reg_tmp_32);

                        // Clipping bounds to prevent overflow/underflow
                        mov(reg_tmp_32, 0x41200000); // 10.0f (max_clip for x - max)
                        vpbroadcastd(zmm_max_clip, reg_tmp_32);

                        mov(reg_tmp_32, 0xc1a00000); // -20.0f (min_clip to avoid exp underflow)
                        vpbroadcastd(zmm_min_clip, reg_tmp_32);

                        // IEEE754: float = sign(1) × 2^(exp-127) × 1.mantissa
                        // To compute 2^n, we set exp = n + 127
                        mov(reg_tmp_32, 127);
                        vpbroadcastd(zmm_127, reg_tmp_32);

                        // Reload params because eax (part of rax) was clobbered
                        mov(rax, ptr[rsp + 8]);

                        // ==================== STEP 1: COMPUTE BLOCK MAXIMUM ====================
                        // Find max across all 64 columns in the current block
                        vmaxps(zmm_max, zmm_c0, zmm_c1);  // max of first 32 columns
                        vmaxps(zmm_max, zmm_max, zmm_c2); // include columns 32-47
                        vmaxps(zmm_max, zmm_max, zmm_c3); // include columns 48-63

                        // ==================== HORIZONTAL REDUCTION (64 → 1) ====================
                        // Reduce 64 FP32 values in zmm_max to a single scalar maximum
                        // Step 1: Extract upper 256 bits (16 floats) and max with lower 256 bits
                        vextractf64x4(ymm4, zmm_max, 1);           // ymm4 = zmm_max[256:512]
                        vmaxps(ymm4, ymm4, Ymm(zmm_max.getIdx())); // ymm4 = max(ymm4, zmm_max[0:256])
                        // Step 2: Extract upper 128 bits and max with lower 128 bits
                        vextractf128(xmm5, ymm4, 1); // xmm5 = ymm4[128:256]
                        vmaxps(xmm4, xmm4, xmm5);    // xmm4 = max(8 floats)
                        // Step 3: Shuffle and max (8 → 4 → 2 → 1)
                        vpermilps(xmm5, xmm4, 0b01001110); // Swap pairs: [2,3,0,1]
                        vmaxps(xmm4, xmm4, xmm5);          // xmm4 = max(4 floats)
                        vpermilps(xmm5, xmm4, 0b10110001); // Swap within pairs: [1,0,3,2]
                        vmaxps(xmm4, xmm4, xmm5);          // xmm4[0] = scalar max

                        // Broadcast scalar max back to all 16 lanes of zmm_max
                        vpbroadcastd(zmm_max, xmm4);

                        // ==================== STORE LOCAL MAX ====================
                        // local_max array stores one scalar per 64-column block
                        // Offset = (loop_N / 64) × sizeof(float) = loop_N >> 4 (since 64/4=16)
                        mov(rdx, ptr[rax + 80]);      // local_max pointer (offset 80)
                        mov(r15, reg_loop_N);         // Use r15 as temp (callee-saved, already pushed)
                        shr(r15, 4);                  // Divide by 16: (loop_N * 4) / 64 = loop_N / 16
                        vmovss(ptr[rdx + r15], xmm4); // Store scalar max

                        // ==================== STEP 2: COMPUTE SUM OF EXP ====================
                        vpxord(zmm_sum, zmm_sum, zmm_sum); // Initialize sum accumulator to 0

                        /**
                         * @brief Lambda to compute exp(val - max) and accumulate to zmm_sum.
                         *
                         * @details
                         * Uses polynomial approximation for exp():
                         *   exp(x) = 2^(x/ln2) = 2^floor(x/ln2) × 2^frac(x/ln2)
                         *
                         * The fractional part is computed via 5th-degree polynomial:
                         *   2^t ≈ c1 + c2*t + c3*t^2 + c4*t^3 + c5*t^4
                         * where t = frac(x / ln2) ∈ [0, 1)
                         *
                         * The integer part is computed by bit manipulation:
                         *   2^n = (n + 127) << 23 (IEEE754 float exponent)
                         *
                         * Algorithm:
                         *   1. tmp = val - max (shift to prevent overflow)
                         *   2. Clip tmp to [-20, 10] to avoid exp overflow/underflow
                         *   3. xf = tmp × (1/ln2)
                         *   4. fx = floor(xf), t = xf - fx
                         *   5. p = polynomial(t) ≈ 2^t
                         *   6. scale = 2^fx (via IEEE754 manipulation)
                         *   7. result = p × scale
                         *   8. sum += result
                         */
                        auto compute_exp_accumulate = [&](const Zmm &zmm_val)
                        {
                            // Step 1: Subtract max for numerical stability (prevents exp overflow)
                            vsubps(zmm_tmp, zmm_val, zmm_max);

                            // Step 2: Clip to safe range [-20, 10]
                            // exp(-20) ≈ 2e-9 (underflow safe), exp(10) ≈ 22026 (overflow safe)
                            vminps(zmm_tmp, zmm_tmp, zmm_max_clip); // min(tmp, 10)
                            vmaxps(zmm_tmp, zmm_tmp, zmm_min_clip); // max(tmp, -20)

                            // Step 3: xf = x × (1/ln2) = log2(exp(x))
                            vmulps(zmm_tmp, zmm_tmp, zmm_inv_ln2);

                            // Step 4: fx = floor(xf), t = xf - fx (fractional part)
                            const Zmm &zmm_fx = zmm18;          // Reuse zmm_bias (free after bias section)
                            vrndscaleps(zmm_fx, zmm_tmp, 0x01); // Round toward -∞
                            vsubps(zmm_tmp, zmm_tmp, zmm_fx);   // t = xf - fx ∈ [0, 1)

                            // Step 5: Polynomial approximation for 2^t
                            // p = c5, then p = p*t + c4, then p = p*t + c3, etc.
                            const Zmm &zmm_p = zmm19;                  // Reuse zmm_mask (free after mask section)
                            vmovaps(zmm_p, zmm_c5_const);              // p = c5
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c4_const); // p = p*t + c4
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c3_coeff); // p = p*t + c3
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c2_const); // p = p*t + c2
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c1_const); // p = p*t + c1

                            // Step 6: Compute 2^fx via IEEE754 exponent manipulation
                            // float = 2^(exp-127), so 2^fx = float with exp = fx + 127
                            vcvtps2dq(zmm_fx, zmm_fx);       // Convert fx to int32
                            vpaddd(zmm_fx, zmm_fx, zmm_127); // exp = fx + 127
                            vpslld(zmm_fx, zmm_fx, 23);      // Shift to exponent position

                            // Step 7: result = p × 2^fx (reinterpret zmm_fx as float)
                            vmulps(zmm_p, zmm_p, zmm_fx); // zmm_p = exp(val - max)

                            // Step 8: Accumulate to sum
                            vaddps(zmm_sum, zmm_sum, zmm_p);
                        };

                        // Apply to all 4 accumulator registers (64 columns total)
                        compute_exp_accumulate(zmm_c0);
                        compute_exp_accumulate(zmm_c1);
                        compute_exp_accumulate(zmm_c2);
                        compute_exp_accumulate(zmm_c3);

                        // ==================== HORIZONTAL REDUCTION OF SUM ====================
                        // Reduce 64 FP32 values in zmm_sum to a single scalar sum
                        vextractf64x4(ymm4, zmm_sum, 1);           // ymm4 = upper 256 bits
                        vaddps(ymm4, ymm4, Ymm(zmm_sum.getIdx())); // Add lower 256 bits
                        vextractf128(xmm5, ymm4, 1);               // xmm5 = upper 128 bits
                        vaddps(xmm4, xmm4, xmm5);                  // Add lower 128 bits
                        vpermilps(xmm5, xmm4, 0b01001110);         // Shuffle for reduction
                        vaddps(xmm4, xmm4, xmm5);                  // xmm4 = sum(4 floats)
                        vpermilps(xmm5, xmm4, 0b10110001);         // Final shuffle
                        vaddps(xmm4, xmm4, xmm5);                  // xmm4[0] = scalar sum

                        // Store local sum to local_sum array
                        // Offset already calculated in r15 from max storage
                        mov(rdx, ptr[rax + 88]);      // local_sum pointer (offset 88)
                        vmovss(ptr[rdx + r15], xmm4); // Store scalar sum
                    }
                    L(skip_softmax);

                    // ==================== FUSED SWIGLU ACTIVATION ====================
                    // Optional: Compute SwiGLU activation for FFN layers
                    // SwiGLU(x, gate) = x × gate × sigmoid(gate)
                    //                = x × gate / (1 + exp(-gate))
                    //
                    // This is used in Llama/Qwen FFN where:
                    //   - x = "up" projection output (current C values)
                    //   - gate = gate projection output (from gate_input pointer)
                    mov(rax, ptr[rsp + 8]);  // Reload params pointer
                    cmp(byte[rax + 112], 1); // do_swiglu flag (offset 112)
                    Label skip_swiglu;
                    jne(skip_swiglu, T_NEAR); // Skip if not enabled
                    {
                        // ==================== SWIGLU REGISTER ALLOCATION ====================
                        // Reuse softmax registers since they're no longer needed
                        const Zmm &zmm_gate = zmm20;     ///< Gate values loaded from gate_input
                        const Zmm &zmm_tmp = zmm21;      ///< Temporary for sigmoid computation
                        const Zmm &zmm_c1_const = zmm22; ///< Polynomial coeff c1
                        const Zmm &zmm_c2_const = zmm23; ///< Polynomial coeff c2
                        const Zmm &zmm_c3_coeff = zmm24; ///< Polynomial coeff c3
                        const Zmm &zmm_c4_const = zmm25; ///< Polynomial coeff c4
                        const Zmm &zmm_c5_const = zmm26; ///< Polynomial coeff c5
                        const Zmm &zmm_inv_ln2 = zmm27;  ///< 1/ln(2)
                        const Zmm &zmm_max_clip = zmm28; ///< Upper clamp for sigmoid = 88.0
                        const Zmm &zmm_min_clip = zmm29; ///< Lower clamp for sigmoid = -88.0
                        const Zmm &zmm_127 = zmm30;      ///< IEEE754 exponent bias
                        const Zmm &zmm_one = zmm31;      ///< Constant 1.0f for sigmoid

                        // ==================== LOAD SIGMOID POLYNOMIAL CONSTANTS ====================
                        // Same polynomial as softmax exp(), different clipping bounds
                        mov(reg_tmp_32, 0x3fb8aa3b); // 1.44269504 (1/ln2)
                        vpbroadcastd(zmm_inv_ln2, reg_tmp_32);

                        mov(reg_tmp_32, 0x3f7ffffe); // 0.99999994 (c1)
                        vpbroadcastd(zmm_c1_const, reg_tmp_32);

                        mov(reg_tmp_32, 0x3f317218); // 0.69314718 (c2 = ln2)
                        vpbroadcastd(zmm_c2_const, reg_tmp_32);

                        mov(reg_tmp_32, 0x3e75fdf1); // 0.24022651 (c3)
                        vpbroadcastd(zmm_c3_coeff, reg_tmp_32);

                        mov(reg_tmp_32, 0x3d6356eb); // 0.05550411 (c4)
                        vpbroadcastd(zmm_c4_const, reg_tmp_32);

                        mov(reg_tmp_32, 0x3c1d9422); // 0.00961813 (c5)
                        vpbroadcastd(zmm_c5_const, reg_tmp_32);

                        mov(reg_tmp_32, 0x42b00000); // 88.0f (max_clip)
                        vpbroadcastd(zmm_max_clip, reg_tmp_32);

                        mov(reg_tmp_32, 0xc2b00000); // -88.0f (min_clip)
                        vpbroadcastd(zmm_min_clip, reg_tmp_32);

                        mov(reg_tmp_32, 0x3f800000); // 1.0f for sigmoid denominator
                        vpbroadcastd(zmm_one, reg_tmp_32);

                        mov(reg_tmp_32, 127);
                        vpbroadcastd(zmm_127, reg_tmp_32);

                        // Reload params because eax (part of rax) was clobbered
                        mov(rax, ptr[rsp + 8]);

                        // Load gate_input pointer (FFN gate projection output)
                        mov(rdx, ptr[rax + 104]); // gate_input pointer (offset 104)

                        /**
                         * @brief Lambda to compute Swish activation: val × gate × sigmoid(gate)
                         *
                         * @details
                         * Swish(x) = x × sigmoid(x) = x / (1 + exp(-x))
                         *
                         * For SwiGLU, we have:
                         *   - val = up projection output (current C values)
                         *   - gate = gate projection output (separate input)
                         *   - result = val × gate × sigmoid(gate) = val × swish(gate)
                         *
                         * sigmoid(gate) = 1 / (1 + exp(-gate))
                         *
                         * Algorithm:
                         *   1. Load gate values
                         *   2. Compute exp(-gate) using polynomial
                         *   3. Compute sigmoid = gate / (1 + exp(-gate))
                         *   4. val = val × sigmoid
                         *
                         * @param zmm_val Current accumulator value (up projection)
                         * @param offset Block offset (0-3) for 16-float chunks
                         */
                        auto compute_swish = [&](const Zmm &zmm_val, int offset)
                        {
                            // Step 1: Load gate values for this block
                            vmovups(zmm_gate, ptr[rdx + offset * 64]);

                            // Step 2: Compute exp(-gate) for sigmoid
                            // First, negate gate: y = -gate
                            vxorps(zmm_tmp, zmm_tmp, zmm_tmp);  // zmm_tmp = 0
                            vsubps(zmm_tmp, zmm_tmp, zmm_gate); // zmm_tmp = -gate

                            // Clip -gate to [-88, 88] to prevent exp overflow/underflow
                            vminps(zmm_tmp, zmm_tmp, zmm_max_clip);
                            vmaxps(zmm_tmp, zmm_tmp, zmm_min_clip);

                            // Compute exp(-gate) using same polynomial as softmax
                            // xf = -gate × (1/ln2)
                            vmulps(zmm_tmp, zmm_tmp, zmm_inv_ln2);

                            // fx = floor(xf), t = xf - fx
                            const Zmm &zmm_fx = zmm18;          // Reuse zmm_bias (free)
                            vrndscaleps(zmm_fx, zmm_tmp, 0x01); // Round toward -∞
                            vsubps(zmm_tmp, zmm_tmp, zmm_fx);   // t = frac(xf)

                            // Polynomial: p = c5, then Horner's method
                            const Zmm &zmm_p = zmm19; // Reuse zmm_mask (free)
                            vmovaps(zmm_p, zmm_c5_const);
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c4_const);
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c3_coeff);
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c2_const);
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c1_const);

                            // Compute 2^fx via IEEE754 exponent manipulation
                            vcvtps2dq(zmm_fx, zmm_fx);
                            vpaddd(zmm_fx, zmm_fx, zmm_127);
                            vpslld(zmm_fx, zmm_fx, 23);

                            // zmm_p = exp(-gate) = polynomial × 2^fx
                            vmulps(zmm_p, zmm_p, zmm_fx);

                            // Step 3: Compute sigmoid = gate / (1 + exp(-gate))
                            vaddps(zmm_p, zmm_p, zmm_one);  // zmm_p = 1 + exp(-gate)
                            vdivps(zmm_p, zmm_gate, zmm_p); // zmm_p = gate / (1 + exp(-gate)) = swish(gate)

                            // Step 4: Final result: val = val × swish(gate)
                            vmulps(zmm_val, zmm_val, zmm_p);
                        };

                        // Apply SwiGLU to all 4 accumulator blocks (64 columns total)
                        compute_swish(zmm_c0, 0); // Columns 0-15
                        compute_swish(zmm_c1, 1); // Columns 16-31
                        compute_swish(zmm_c2, 2); // Columns 32-47
                        compute_swish(zmm_c3, 3); // Columns 48-63
                    }
                    L(skip_swiglu);

                    // ==================== STORE OUTPUT ====================
                    // Check output format: FP32 (default), Q8_1, or Q16_1
                    mov(rax, ptr[rsp + 8]); // Reload params pointer
                    // output_format is at offset 113 (after do_swiglu at 112)
                    cmp(byte[rax + 113], static_cast<uint8_t>(GemmOutputFormat::Q8_1));
                    Label store_fp32, check_q16_1;
                    jne(check_q16_1, T_NEAR); // If not Q8_1, check Q16_1
                    {
                        // ==================== Q8_1 REQUANTIZATION STORE ====================
                        // Convert 64 FP32 values (zmm_c0..zmm_c3) to 2 Q8_1 blocks
                        // Each Q8_1Block: FP16 scale + INT16 sum + 32 INT8 values = 36 bytes
                        //
                        // IMPORTANT: Use xmm4-xmm15 for VEX-only operations (vrcpss, etc.)
                        // Extended registers (xmm16+) require EVEX encoding which some
                        // instructions don't support.

                        // Load Q8_1 output pointer
                        mov(rdx, ptr[rax + 120]); // C_q8_1 pointer

                        // Compute output address: C_q8_1 + (loop_N / 32) * 36
                        mov(r8, reg_loop_N);
                        shr(r8, 5);       // loop_N / 32 = block index
                        imul(r8, r8, 36); // block_index * 36 bytes per block
                        add(rdx, r8);     // rdx = output pointer for first block

                        // ============ BLOCK 0: zmm_c0 + zmm_c1 (32 values) ============
                        // Find max absolute value across 32 floats
                        vrangeps(zmm16, zmm_c0, zmm_c0, 0x0B); // abs(c0)
                        vrangeps(zmm17, zmm_c1, zmm_c1, 0x0B); // abs(c1)
                        vmaxps(zmm16, zmm16, zmm17);

                        // Horizontal max reduction
                        vextractf32x8(ymm17, zmm16, 1);
                        vmaxps(ymm16, ymm16, ymm17);
                        vextractf32x4(xmm17, ymm16, 1);
                        vmaxps(xmm16, xmm16, xmm17);
                        vshufps(xmm17, xmm16, xmm16, 0x4E);
                        vmaxps(xmm16, xmm16, xmm17);
                        vshufps(xmm17, xmm16, xmm16, 0xB1);
                        vmaxps(xmm16, xmm16, xmm17);
                        // xmm16[0] = max_abs

                        // Copy to low register for VEX operations
                        vmovaps(xmm4, xmm16); // xmm4 = max_abs

                        // Compute scale: d = max_abs / 127.0
                        mov(eax, 0x42FE0000); // 127.0f
                        vmovd(xmm5, eax);
                        vdivss(xmm6, xmm4, xmm5); // xmm6 = d = max_abs / 127

                        // Convert scale to FP16 and store
                        vcvtps2ph(xmm7, xmm6, 0);
                        vpextrw(ptr[rdx + 0], xmm7, 0); // Store d (FP16, 2 bytes)

                        // Compute inverse scale for quantization (VEX-only vrcpss)
                        vrcpss(xmm8, xmm6, xmm6);  // xmm8 = inv_d ≈ 1/d
                        vbroadcastss(zmm24, xmm8); // Broadcast to zmm24

                        // Quantize: qs[i] = round(ctx[i] * inv_d)
                        vmulps(zmm20, zmm_c0, zmm24);
                        vmulps(zmm21, zmm_c1, zmm24);

                        // Round to nearest and convert to int32
                        vcvtps2dq(zmm20, zmm20);
                        vcvtps2dq(zmm21, zmm21);

                        // Pack to int8 with saturation using vpmovdb
                        vpmovdb(xmm20, zmm20);                // 16 int32 -> 16 int8
                        vpmovdb(xmm21, zmm21);                // 16 int32 -> 16 int8
                        vinserti32x4(ymm20, ymm20, xmm21, 1); // Combine into 32 int8

                        // Compute sum_qs (sum of int8 values)
                        vpmovsxbd(zmm22, xmm20);
                        vextracti32x4(xmm23, ymm20, 1);
                        vpmovsxbd(zmm23, xmm23);
                        vpaddd(zmm22, zmm22, zmm23);

                        // Horizontal sum
                        vextracti32x8(ymm23, zmm22, 1);
                        vpaddd(ymm22, ymm22, ymm23);
                        vextracti32x4(xmm23, ymm22, 1);
                        vpaddd(xmm22, xmm22, xmm23);
                        vpshufd(xmm23, xmm22, 0x4E);
                        vpaddd(xmm22, xmm22, xmm23);
                        vpshufd(xmm23, xmm22, 0xB1);
                        vpaddd(xmm22, xmm22, xmm23);
                        vmovd(eax, xmm22);
                        mov(ptr[rdx + 2], ax); // Store sum_qs (INT16, 2 bytes)

                        // Store quantized values (32 int8)
                        vmovdqu32(ptr[rdx + 4], ymm20);

                        // ============ BLOCK 1: zmm_c2 + zmm_c3 (32 values) ============
                        vrangeps(zmm16, zmm_c2, zmm_c2, 0x0B);
                        vrangeps(zmm17, zmm_c3, zmm_c3, 0x0B);
                        vmaxps(zmm16, zmm16, zmm17);

                        vextractf32x8(ymm17, zmm16, 1);
                        vmaxps(ymm16, ymm16, ymm17);
                        vextractf32x4(xmm17, ymm16, 1);
                        vmaxps(xmm16, xmm16, xmm17);
                        vshufps(xmm17, xmm16, xmm16, 0x4E);
                        vmaxps(xmm16, xmm16, xmm17);
                        vshufps(xmm17, xmm16, xmm16, 0xB1);
                        vmaxps(xmm16, xmm16, xmm17);

                        // Copy to low register for VEX operations
                        vmovaps(xmm4, xmm16);

                        mov(eax, 0x42FE0000);
                        vmovd(xmm5, eax);
                        vdivss(xmm6, xmm4, xmm5);

                        vcvtps2ph(xmm7, xmm6, 0);
                        vpextrw(ptr[rdx + 36 + 0], xmm7, 0);

                        vrcpss(xmm8, xmm6, xmm6);
                        vbroadcastss(zmm24, xmm8);

                        vmulps(zmm20, zmm_c2, zmm24);
                        vmulps(zmm21, zmm_c3, zmm24);

                        vcvtps2dq(zmm20, zmm20);
                        vcvtps2dq(zmm21, zmm21);

                        vpmovdb(xmm20, zmm20);
                        vpmovdb(xmm21, zmm21);
                        vinserti32x4(ymm20, ymm20, xmm21, 1);

                        vpmovsxbd(zmm22, xmm20);
                        vextracti32x4(xmm23, ymm20, 1);
                        vpmovsxbd(zmm23, xmm23);
                        vpaddd(zmm22, zmm22, zmm23);

                        vextracti32x8(ymm23, zmm22, 1);
                        vpaddd(ymm22, ymm22, ymm23);
                        vextracti32x4(xmm23, ymm22, 1);
                        vpaddd(xmm22, xmm22, xmm23);
                        vpshufd(xmm23, xmm22, 0x4E);
                        vpaddd(xmm22, xmm22, xmm23);
                        vpshufd(xmm23, xmm22, 0xB1);
                        vpaddd(xmm22, xmm22, xmm23);
                        vmovd(eax, xmm22);
                        mov(ptr[rdx + 36 + 2], ax);

                        vmovdqu32(ptr[rdx + 36 + 4], ymm20);

                        jmp("after_store", T_NEAR);
                    }

                    // ==================== Q16_1 OUTPUT CHECK ====================
                    L(check_q16_1);
                    mov(rax, ptr[rsp + 8]); // Reload params pointer
                    cmp(byte[rax + 113], static_cast<uint8_t>(GemmOutputFormat::Q16_1));
                    jne(store_fp32, T_NEAR); // If not Q16_1, use FP32 store
                    {
                        // ==================== Q16_1 REQUANTIZATION STORE ====================
                        // Convert 64 FP32 values (zmm_c0..zmm_c3) to Q16_1 block(s)
                        // Q16_1 Block structure:
                        //   - float d: FP32 scale (4 bytes)
                        //   - int32_t sum_qs: Sum of quantized values (4 bytes)
                        //   - int16_t qs[block_size]: Quantized values (2 * block_size bytes)
                        //
                        // Block sizes: 32 (72 bytes), 64 (136 bytes), or 128 (264 bytes)
                        // The kernel always processes 64 columns, so:
                        //   - block_size=32: Write 2 Q16_1Block (32 values each)
                        //   - block_size=64: Write 1 Q16_1Block_64 (64 values)
                        //   - block_size=128: Accumulate first 64 of 128-value block (partial)
                        //
                        // For HybridQ16 K projection, block_size=64 (head_dim) is typical.

                        // Load Q16_1 output pointer and block size
                        // Note: We need to be careful about register usage here
                        // rdx/edx is used by div, so we save the pointer elsewhere first
                        mov(r11, ptr[rax + 136]); // C_q16_1 pointer - save in r11
                        mov(ecx, ptr[rax + 148]); // q16_block_size (offset 148)

                        // Compute output address: C_q16_1 + (loop_N / block_size) * block_bytes
                        // For block_size=64, block_bytes = 136
                        // We need to handle different block sizes at runtime

                        // Calculate block byte size based on q16_block_size
                        // block_bytes = 8 + 2 * block_size (d=4, sum_qs=4, qs[n]*2)
                        mov(r8d, ecx); // r8 = block_size
                        shl(r8d, 1);   // r8 = block_size * 2 (bytes for qs)
                        add(r8d, 8);   // r8 = block_bytes = 8 + 2*block_size
                        mov(r9d, r8d); // Save block_bytes in r9

                        // Calculate block index = loop_N / block_size
                        mov(eax, reg_loop_N.cvt32()); // eax = loop_N (32-bit) for division
                        xor_(edx, edx);               // Clear edx for division (edx:eax / ecx)
                        div(ecx);                     // eax = loop_N / block_size (quotient)
                        // Note: eax now contains block_index

                        // Calculate offset = block_index * block_bytes
                        imul(eax, r9d); // eax = block_index * block_bytes
                        mov(rdx, r11);  // rdx = C_q16_1 pointer (restored from r11)
                        add(rdx, rax);  // rdx = output pointer for this block (zero-extend eax to rax)

                        // For block_size=64: Write all 64 values as one Q16_1Block_64
                        // For block_size=32: Write as two Q16_1Block (handled in multiply method)
                        // For now, implement block_size=64 path (most common for HybridQ16)

                        // ============ QUANTIZE 64 FP32 VALUES TO Q16_1 ============
                        // Step 1: Find max absolute value across all 64 floats
                        vrangeps(zmm16, zmm_c0, zmm_c0, 0x0B); // abs(c0)
                        vrangeps(zmm17, zmm_c1, zmm_c1, 0x0B); // abs(c1)
                        vrangeps(zmm18, zmm_c2, zmm_c2, 0x0B); // abs(c2)
                        vrangeps(zmm19, zmm_c3, zmm_c3, 0x0B); // abs(c3)
                        vmaxps(zmm16, zmm16, zmm17);
                        vmaxps(zmm18, zmm18, zmm19);
                        vmaxps(zmm16, zmm16, zmm18);

                        // Horizontal max reduction to get single max_abs value
                        vextractf32x8(ymm17, zmm16, 1);
                        vmaxps(ymm16, ymm16, ymm17);
                        vextractf32x4(xmm17, ymm16, 1);
                        vmaxps(xmm16, xmm16, xmm17);
                        vshufps(xmm17, xmm16, xmm16, 0x4E);
                        vmaxps(xmm16, xmm16, xmm17);
                        vshufps(xmm17, xmm16, xmm16, 0xB1);
                        vmaxps(xmm16, xmm16, xmm17);
                        // xmm16[0] = max_abs

                        // Copy to low register for VEX operations
                        vmovaps(xmm4, xmm16); // xmm4 = max_abs

                        // Step 2: Compute scale: d = max_abs / 16383.0 (INT16 range)
                        // Using 16383 instead of 32767 for headroom
                        mov(eax, 0x467FFE00); // 16383.0f in IEEE754
                        vmovd(xmm5, eax);
                        vdivss(xmm6, xmm4, xmm5); // xmm6 = d = max_abs / 16383

                        // Store scale as FP32 (4 bytes at offset 0)
                        vmovss(ptr[rdx + 0], xmm6);

                        // Step 3: Compute inverse scale for quantization
                        vrcpss(xmm8, xmm6, xmm6);  // xmm8 = inv_d ≈ 1/d
                        vbroadcastss(zmm24, xmm8); // Broadcast to zmm24

                        // Step 4: Quantize: qs[i] = round(val[i] * inv_d)
                        vmulps(zmm20, zmm_c0, zmm24);
                        vmulps(zmm21, zmm_c1, zmm24);
                        vmulps(zmm22, zmm_c2, zmm24);
                        vmulps(zmm23, zmm_c3, zmm24);

                        // Round to nearest and convert to int32
                        vcvtps2dq(zmm20, zmm20);
                        vcvtps2dq(zmm21, zmm21);
                        vcvtps2dq(zmm22, zmm22);
                        vcvtps2dq(zmm23, zmm23);

                        // Step 5: Pack int32 to int16 with saturation (vpmovsdw)
                        // Each zmm has 16 int32 values -> 16 int16 values
                        vpmovsdw(ymm20, zmm20); // 16 int32 -> 16 int16
                        vpmovsdw(ymm21, zmm21);
                        vpmovsdw(ymm22, zmm22);
                        vpmovsdw(ymm23, zmm23);

                        // Combine into zmm registers for 64 int16 total
                        vinserti64x4(zmm20, zmm20, ymm21, 1); // zmm20 = 32 int16 (c0+c1)
                        vinserti64x4(zmm22, zmm22, ymm23, 1); // zmm22 = 32 int16 (c2+c3)

                        // Step 6: Compute sum_qs (sum of all 64 int16 values)
                        // Sign-extend to int32 and sum
                        vpmovsxwd(zmm16, ymm20); // First 16 int16 -> 16 int32
                        vextracti64x4(ymm17, zmm20, 1);
                        vpmovsxwd(zmm17, ymm17); // Next 16 int16 -> 16 int32
                        vpaddd(zmm16, zmm16, zmm17);

                        vpmovsxwd(zmm18, ymm22); // First 16 int16 of c2+c3
                        vextracti64x4(ymm19, zmm22, 1);
                        vpmovsxwd(zmm19, ymm19); // Next 16 int16
                        vpaddd(zmm18, zmm18, zmm19);

                        vpaddd(zmm16, zmm16, zmm18); // Sum all 64 values

                        // Horizontal sum for sum_qs
                        vextracti32x8(ymm17, zmm16, 1);
                        vpaddd(ymm16, ymm16, ymm17);
                        vextracti32x4(xmm17, ymm16, 1);
                        vpaddd(xmm16, xmm16, xmm17);
                        vpshufd(xmm17, xmm16, 0x4E);
                        vpaddd(xmm16, xmm16, xmm17);
                        vpshufd(xmm17, xmm16, 0xB1);
                        vpaddd(xmm16, xmm16, xmm17);
                        // xmm16[0] = sum_qs (int32)

                        // Store sum_qs as INT32 (4 bytes at offset 4)
                        vmovd(ptr[rdx + 4], xmm16);

                        // Step 7: Store quantized int16 values (64 * 2 = 128 bytes at offset 8)
                        vmovdqu32(ptr[rdx + 8], zmm20);      // 32 int16 values (64 bytes)
                        vmovdqu32(ptr[rdx + 8 + 64], zmm22); // Next 32 int16 values

                        jmp("after_store", T_NEAR);
                    }

                    L(store_fp32);
                    // ==================== FP32 STORE (default) ====================
                    // Write final C values back to memory as FP32
                    vmovups(ptr[reg_C_cursor + 0 * 64], zmm_c0); // C[0, n+0:n+16]
                    vmovups(ptr[reg_C_cursor + 1 * 64], zmm_c1); // C[0, n+16:n+32]
                    vmovups(ptr[reg_C_cursor + 2 * 64], zmm_c2); // C[0, n+32:n+48]
                    vmovups(ptr[reg_C_cursor + 3 * 64], zmm_c3); // C[0, n+48:n+64]
                    L("after_store");

                    // ==================== ADVANCE N-LOOP ====================
                    add(reg_loop_N, 64);       // Move to next 64-column block
                    jmp(loop_N_label, T_NEAR); // Continue N-loop
                }

                // ==================== EPILOGUE ====================
                L("end_kernel");
                add(rsp, 16);  // Deallocate stack: N (8 bytes) + params ptr (8 bytes)
                add(rsp, 256); // Deallocate zero buffer

                // Restore callee-saved registers (reverse order of prologue)
                pop(r15);
                pop(r14);
                pop(r13);
                pop(r12);
                pop(rbp);
                pop(rbx);
                ret(); // Return to caller
            }
        };

    } // namespace gemm_v4
} // namespace llaminar2
