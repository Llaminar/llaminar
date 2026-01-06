/**
 * @file QuantisedGemmJit_M2.h
 * @brief JIT-compiled AVX-512 VNNI GEMM kernel optimized for two-row (M=2) matrix multiplication.
 * @author David Sanftenberg
 *
 * @details
 * This file implements a high-performance JIT-compiled GEMM kernel for quantized matrix
 * multiplication using AVX-512 VNNI instructions. The kernel is optimized for M=2 (two query
 * rows), which provides better throughput than the M=1 kernel when processing multiple tokens.
 *
 * ## Design Rationale
 *
 * The M=2 kernel processes two rows simultaneously using 8 ZMM accumulators (4 per row).
 * This provides better instruction-level parallelism and memory bandwidth utilization compared
 * to running the M=1 kernel twice, because:
 *
 * 1. **Shared B loads**: Each B block is loaded once and used for both rows
 * 2. **Better cache utilization**: B data stays in L1/L2 cache across row computations
 * 3. **Reduced loop overhead**: Single N-loop and K-loop service both rows
 *
 * ## Register Allocation
 *
 * | Register | Purpose |
 * |----------|---------|
 * | zmm0-3   | Row 0 FP32 accumulators (64 output columns) |
 * | zmm4-7   | Row 1 FP32 accumulators (64 output columns) |
 * | zmm8-11  | Row 0 INT32 temp accumulators for VNNI |
 * | zmm12-15 | Row 1 INT32 temp accumulators for VNNI |
 * | zmm16-19 | B scale values and temps |
 * | zmm20-31 | Softmax/SwiGLU constants and temps |
 *
 * ## Memory Layout
 *
 * Same as M=1, but with A_stride parameter to access second row:
 * - Row 0 A: params->A
 * - Row 1 A: params->A + params->A_stride (where A_stride = K/32 × sizeof(Q8_1Block))
 *
 * ## Stack Frame
 *
 * The M=2 kernel allocates additional stack space for a zero buffer (256 bytes) used
 * when mins pointer is null (symmetric quantization). This avoids conditional branches
 * in the hot path.
 *
 * ## Fused Operations
 *
 * Same as M=1 kernel:
 * 1. **Bias Addition**: C += bias[N] (broadcast to both rows)
 * 2. **Attention Mask**: C += mask[M×N] (different mask per row)
 * 3. **Softmax**: Computes per-row local max/sum (independent for row 0 and row 1)
 * 4. **SwiGLU**: gate × sigmoid(gate) × up (different gate per row)
 *
 * @see QuantisedGemmJit_M1 for the single-row variant
 * @see QuantisedGemmKernel for the high-level kernel interface
 */

#pragma once

#include "../../../../../external/onednn/third_party/xbyak/xbyak.h"
#include "QuantisedGemmJit_M1.h" // For QuantisedGemmParams
#include <vector>
#include <cstdint>

namespace llaminar2
{
    namespace gemm_v4
    {

        /**
         * @class QuantisedGemmJit_M2
         * @brief JIT-compiled GEMM kernel for two-row (M=2) quantized matrix multiplication.
         *
         * @details
         * This class uses the Xbyak JIT assembler to generate optimized AVX-512 VNNI
         * assembly code at runtime. The generated code is specialized for M=2 (two query
         * rows), which provides better throughput than running M=1 twice.
         *
         * ## Code Generation
         *
         * The constructor calls `generate()` which emits x86-64 assembly code into a
         * buffer. The generated code is then callable as a function pointer via `getCode()`.
         *
         * ## Buffer Size
         *
         * The `4096 * 64` (256KB) buffer size is larger than M=1 to accommodate:
         * - Prologue/epilogue code (~300 bytes)
         * - N-loop with K-loop for both rows (~4KB)
         * - Row 0 and Row 1 softmax code (~2KB)
         * - Row 0 and Row 1 SwiGLU code (~2KB)
         * - Total with safety margin: ~256KB
         *
         * ## Usage Example
         *
         * ```cpp
         * // Create kernel (generates JIT code)
         * QuantisedGemmJit_M2 kernel;
         *
         * // Set up parameters (same as M1, but A has 2 rows)
         * QuantisedGemmParams params = {...};
         * params.A_stride = K_blocks * sizeof(Q8_1Block);  // Stride to row 1
         *
         * // Get function pointer and call
         * auto fn = kernel.getCode<kernel_func_t>();
         * fn(&params);
         * ```
         */
        class QuantisedGemmJit_M2 : public Xbyak::CodeGenerator
        {
        public:
            /**
             * @brief Construct and generate the JIT kernel.
             *
             * Allocates a 256KB code buffer and generates optimized AVX-512 VNNI
             * assembly code for M=2 quantized GEMM with optional fused operations.
             */
            QuantisedGemmJit_M2() : Xbyak::CodeGenerator(4096 * 64)
            {
                generate();
            }

            /**
             * @brief Function pointer type for the generated kernel.
             *
             * The kernel takes a pointer to QuantisedGemmParams and computes
             * C[2×N] = A[2×K] × B[K×N] with optional fused operations.
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

        private:
            /**
             * @brief Generate the JIT assembly code for M=2 quantized GEMM.
             *
             * @details
             * This method emits x86-64 assembly code using Xbyak. The generated code
             * implements an optimized GEMM kernel processing two rows simultaneously.
             *
             * ## High-Level Algorithm
             *
             * ```
             * void kernel(const QuantisedGemmParams* params) {
             *     // Allocate stack for zero buffer (when mins is null)
             *     float zero_buffer[64] = {0};
             *
             *     for (n = 0; n < N; n += 64) {           // N-loop: 64 columns per iteration
             *         Row0_acc[0:64] = 0;                  // zmm0-3: Row 0 accumulators
             *         Row1_acc[0:64] = 0;                  // zmm4-7: Row 1 accumulators
             *
             *         for (k = 0; k < K; k += 32) {        // K-loop: 32 elements per Q8_1 block
             *             // Load A row 0 and row 1 blocks
             *             // Load B blocks (shared between rows)
             *             // VNNI for row 0: zmm8-11
             *             // VNNI for row 1: zmm12-15
             *             // Asymmetric correction for both rows
             *             // Scale and accumulate
             *         }
             *
             *         // Fused post-ops for row 0
             *         if (bias) Row0_acc += bias[n:n+64];
             *         if (mask) Row0_acc += mask[0, n:n+64];
             *         if (softmax) compute_local_max_sum(Row0_acc);
             *         if (swiglu) Row0_acc = swiglu(Row0_acc, gate[0, n:n+64]);
             *
             *         // Fused post-ops for row 1
             *         if (bias) Row1_acc += bias[n:n+64];
             *         if (mask) Row1_acc += mask[1, n:n+64];
             *         if (softmax) compute_local_max_sum(Row1_acc);
             *         if (swiglu) Row1_acc = swiglu(Row1_acc, gate[1, n:n+64]);
             *
             *         store Row0_acc to C[0, n:n+64]
             *         store Row1_acc to C[1, n:n+64]
             *     }
             * }
             * ```
             *
             * ## Register Allocation
             *
             * ### General Purpose Registers
             * Same as M=1, with A_cursor used for row 0 and A_stride offset for row 1
             *
             * ### ZMM Registers (512-bit)
             * | Register | Purpose |
             * |----------|---------|
             * | zmm0-3   | Row 0 FP32 accumulators |
             * | zmm4-7   | Row 1 FP32 accumulators |
             * | zmm8-11  | Row 0 INT32 temp accumulators |
             * | zmm12-15 | Row 1 INT32 temp accumulators |
             * | zmm16-19 | Scales, constants, temps |
             * | zmm20-31 | Softmax/SwiGLU (varies by phase) |
             */
            void generate()
            {
                using namespace Xbyak;

                // ==================== REGISTER DEFINITIONS ====================
                // System V AMD64 ABI: First 6 integer args in rdi, rsi, rdx, rcx, r8, r9

                // Parameter registers (from params struct)
                const Reg64 &reg_A = rdi;       ///< params->A: Q8_1 activation pointer (row 0)
                const Reg64 &reg_B = rsi;       ///< params->B_packed: packed weight pointer
                const Reg64 &reg_Comp = rdx;    ///< params->comp: compensation pointer
                const Reg64 &reg_Scales = rcx;  ///< params->scales: scale pointer
                const Reg64 &reg_C = r8;        ///< params->C: output pointer
                const Reg64 &reg_K_blocks = r9; ///< params->K_blocks: number of K blocks
                // N is stored on stack at [rsp + 0]
                // ldc is loaded from params struct

                // Working registers (callee-saved, pushed in prologue)
                const Reg64 &reg_tmp = rax;           ///< General temporary register
                const Reg64 &reg_stride = rbx;        ///< Stride = N * sizeof(float)
                const Reg64 &reg_loop_N = r10;        ///< N-loop counter
                const Reg64 &reg_loop_K = r11;        ///< K-loop counter
                const Reg64 &reg_B_cursor = r12;      ///< Current B position in K-loop
                const Reg64 &reg_Comp_cursor = r13;   ///< Current compensation position
                const Reg64 &reg_C_cursor = r14;      ///< Current C position in N-loop
                const Reg64 &reg_A_cursor = r15;      ///< Current A position for row 0
                const Reg64 &reg_Scales_cursor = rbp; ///< Current scales position
                const Reg64 &reg_Mins_cursor = r8;    ///< Repurposed for mins pointer during K-loop

                const Reg32 &reg_tmp_32 = eax; ///< 32-bit temp for immediate loading

                // ==================== ZMM REGISTER ALLOCATION ====================

                // Row 0 C accumulators: 4 ZMM = 64 FP32 output columns
                // zmm0 = C[0, n+0:n+16], zmm1 = C[0, n+16:n+32]
                // zmm2 = C[0, n+32:n+48], zmm3 = C[0, n+48:n+64]

                // Row 1 C accumulators: 4 ZMM = 64 FP32 output columns
                // zmm4 = C[1, n+0:n+16], zmm5 = C[1, n+16:n+32]
                // zmm6 = C[1, n+32:n+48], zmm7 = C[1, n+48:n+64]

                // Row 0 INT32 temp accumulators for VNNI (before scale)
                // zmm8 = acc0[0:16], zmm9 = acc0[16:32]
                // zmm10 = acc0[32:48], zmm11 = acc0[48:64]

                // Row 1 INT32 temp accumulators for VNNI
                // zmm12 = acc1[0:16], zmm13 = acc1[16:32]
                // zmm14 = acc1[32:48], zmm15 = acc1[48:64]

                // B registers (shared between rows) - 4 blocks of 16×4 INT8
                const Zmm &zmm_b0 = zmm16; ///< B block for columns 0-15
                const Zmm &zmm_b1 = zmm17; ///< B block for columns 16-31
                const Zmm &zmm_b2 = zmm18; ///< B block for columns 32-47
                const Zmm &zmm_b3 = zmm19; ///< B block for columns 48-63

                // A registers (one per row)
                const Zmm &zmm_a0 = zmm20; ///< A[row 0] values broadcast
                const Zmm &zmm_a1 = zmm21; ///< A[row 1] values broadcast

                // Constants and workspace
                const Zmm &zmm_scale = zmm22;    ///< Scale broadcast workspace
                const Zmm &zmm_128 = zmm23;      ///< Constant 128 for unsigned conversion
                const Zmm &zmm_neg_128 = zmm24;  ///< Constant -128 for correction
                const Ymm &ymm_tmp = ymm25;      ///< Temp for FP16→FP32 conversion
                const Zmm &zmm_scale_b0 = zmm26; ///< B scale for columns 0-15
                const Zmm &zmm_scale_b1 = zmm27; ///< B scale for columns 16-31
                const Zmm &zmm_scale_b2 = zmm28; ///< B scale for columns 32-47
                const Zmm &zmm_scale_b3 = zmm29; ///< B scale for columns 48-63
                const Zmm &zmm_bias = zmm30;     ///< Bias vector workspace
                const Zmm &zmm_mask = zmm31;     ///< Attention mask workspace

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

                // ==================== LOAD PARAMETERS ====================
                // Save params pointer for later use
                const Reg64 &reg_params = rdi;
                push(reg_params);

                // Load struct members from params
                mov(reg_B, ptr[reg_params + 8]);       // B_packed pointer
                mov(reg_Comp, ptr[reg_params + 16]);   // compensation pointer
                mov(reg_Scales, ptr[reg_params + 24]); // scales pointer
                // mins is at offset +32, loaded later
                mov(reg_C, ptr[reg_params + 40]);                // C (output) pointer
                mov(reg_K_blocks.cvt32(), ptr[reg_params + 48]); // K_blocks count

                // Push N to stack for N-loop comparison
                mov(reg_loop_N.cvt32(), ptr[reg_params + 52]); // N dimension
                push(reg_loop_N);

                // Load ldc (leading dimension of C) and convert to bytes
                mov(reg_stride.cvt32(), ptr[reg_params + 56]); // ldc
                shl(reg_stride, 2);                            // ldc * sizeof(float)

                // ==================== MINS POINTER HANDLING ====================
                // If mins is null (symmetric quantization), use stack zero buffer
                mov(rax, ptr[reg_params + 32]); // Load mins pointer
                test(rax, rax);                 // Check if null
                mov(rax, reg_stride);           // Preserve stride in rax
                Label mins_ok;
                jnz(mins_ok, T_NEAR); // Jump if mins is not null
                xor_(rax, rax);       // Set mins stride to 0 (read same zeros every K-block)
                L(mins_ok);
                // Store mins stride at reserved location in stack frame
                // Location: rsp + 264 (after zero buffer and pushes)
                mov(ptr[rsp + 264], rax);

                // Load A pointer (final overwrite of rdi)
                mov(reg_A_cursor, reg_A);        // Temp use
                mov(reg_A, ptr[reg_params + 0]); // A pointer

                // ==================== CONSTANT INITIALIZATION ====================
                // vpdpbusd requires unsigned A operand (0-255 range)
                // Q8_1 uses signed INT8 (-128 to 127), so we XOR with 0x80 to convert
                mov(reg_tmp_32, 0x80808080); // 128 in each byte
                vpbroadcastd(zmm_128, reg_tmp_32);

                mov(reg_tmp_32, -128);
                vpbroadcastd(zmm_neg_128, reg_tmp_32);

                // ==================== N-LOOP (64 columns per iteration) ====================
                xor_(reg_loop_N, reg_loop_N); // Initialize N counter to 0

                Label loop_N_label;
                L(loop_N_label);
                {
                    // Check if we have processed all N columns
                    cmp(reg_loop_N, ptr[rsp]); // Compare with N on stack
                    jge("end_kernel", T_NEAR); // Exit if loop_N >= N

                    // Reset A cursor to start for this N-block
                    mov(reg_A_cursor, reg_A);

                    // ==================== CURSOR SETUP FOR N-BLOCK ====================
                    // Reload params (rdi was overwritten)
                    mov(rax, ptr[rsp + 8]);
                    mov(reg_Comp, ptr[rax + 16]);   // Restore comp pointer
                    mov(reg_Scales, ptr[rax + 24]); // Restore scales pointer

                    // Calculate byte offset: loop_N × sizeof(float)
                    mov(reg_tmp, reg_loop_N);
                    shl(reg_tmp, 2); // loop_N * 4

                    // Set up comp cursor
                    mov(reg_Comp_cursor, reg_Comp);
                    add(reg_Comp_cursor, reg_tmp);

                    // Set up scales cursor
                    mov(reg_Scales_cursor, reg_Scales);
                    add(reg_Scales_cursor, reg_tmp);

                    // ==================== MINS CURSOR (with null fallback) ====================
                    mov(rax, ptr[rsp + 8]);              // Reload params
                    mov(reg_Mins_cursor, ptr[rax + 32]); // Load mins pointer

                    test(reg_Mins_cursor, reg_Mins_cursor);
                    Label mins_null_in_loop, mins_ready_in_loop;
                    jz(mins_null_in_loop, T_NEAR); // Jump if mins is null

                    // Mins is valid: add offset
                    mov(reg_tmp, reg_loop_N);
                    shl(reg_tmp, 2);
                    add(reg_Mins_cursor, reg_tmp); // mins + loop_N * 4
                    jmp(mins_ready_in_loop, T_NEAR);

                    L(mins_null_in_loop);
                    lea(reg_Mins_cursor, ptr[rsp + 16]); // Point to stack zero buffer

                    L(mins_ready_in_loop);

                    // ==================== C CURSOR FOR BOTH ROWS ====================
                    // Reload C from params (r8 was repurposed for mins)
                    mov(rax, ptr[rsp + 8]);           // Reload params
                    mov(reg_C_cursor, ptr[rax + 40]); // C pointer to r14

                    // Add N offset
                    mov(reg_tmp, reg_loop_N);
                    shl(reg_tmp, 2);            // loop_N * 4
                    add(reg_C_cursor, reg_tmp); // C cursor = C + loop_N * 4

                    // ==================== B CURSOR CALCULATION ====================
                    // B is packed in blocks of 64 columns × K elements
                    // B cursor offset = (loop_N / 64) × K_blocks × 2048 bytes
                    //   where 2048 = 64 columns × 32 K-elements × 1 byte
                    mov(reg_tmp, reg_loop_N);
                    shr(reg_tmp, 6);             // loop_N / 64
                    imul(reg_tmp, reg_K_blocks); // × K_blocks
                    shl(reg_tmp, 11);            // × 2048

                    mov(reg_B_cursor, reg_B);
                    add(reg_B_cursor, reg_tmp);

                    // Initialize C Accumulators (zmm0..7) from memory
                    // Row 0
                    // =====================================================================
                    // C ACCUMULATOR INITIALIZATION
                    // =====================================================================
                    // For beta=1 accumulation, we load existing C values as initial accumulators.
                    // This allows C += A * B without separate FMA.
                    //
                    // Memory layout for C (row-major FP32):
                    //   Row 0: C[0, n:n+64] at reg_C_cursor + {0,64,128,192}
                    //   Row 1: C[1, n:n+64] at reg_C_cursor + ldc*4 + {0,64,128,192}
                    //
                    // IMPORTANT: When output_format is Q8_1 or Q16_1, C pointer is nullptr.
                    // In that case, we zero-initialize the accumulators instead.
                    mov(rax, ptr[rsp + 8]); // params ptr
                    cmp(byte[rax + 113], static_cast<uint8_t>(GemmOutputFormat::Q8_1));
                    Label zero_accumulators;
                    je(zero_accumulators, T_NEAR);
                    cmp(byte[rax + 113], static_cast<uint8_t>(GemmOutputFormat::Q16_1));
                    Label load_c_values;
                    jne(load_c_values, T_NEAR); // If not Q8_1 or Q16_1, load from C

                    L(zero_accumulators);
                    {
                        // Quantized output: Zero-initialize all 8 accumulators (2 rows × 64 columns)
                        vxorps(zmm0, zmm0, zmm0);
                        vxorps(zmm1, zmm1, zmm1);
                        vxorps(zmm2, zmm2, zmm2);
                        vxorps(zmm3, zmm3, zmm3);
                        vxorps(zmm4, zmm4, zmm4);
                        vxorps(zmm5, zmm5, zmm5);
                        vxorps(zmm6, zmm6, zmm6);
                        vxorps(zmm7, zmm7, zmm7);
                        jmp("after_c_load", T_NEAR);
                    }
                    L(load_c_values);
                    // Load Row 0 C values (64 FP32 elements = 256 bytes)
                    vmovups(zmm0, ptr[reg_C_cursor + 0 * 64]); // C[0, n:n+16]
                    vmovups(zmm1, ptr[reg_C_cursor + 1 * 64]); // C[0, n+16:n+32]
                    vmovups(zmm2, ptr[reg_C_cursor + 2 * 64]); // C[0, n+32:n+48]
                    vmovups(zmm3, ptr[reg_C_cursor + 3 * 64]); // C[0, n+48:n+64]

                    // Load Row 1 C values
                    // Row 1 offset = ldc * sizeof(float) = ldc * 4
                    mov(rax, ptr[rsp + 8]);              // params ptr
                    mov(reg_tmp.cvt32(), ptr[rax + 56]); // ldc (leading dimension of C)
                    shl(reg_tmp, 2);                     // ldc * 4 bytes

                    vmovups(zmm4, ptr[reg_C_cursor + reg_tmp + 0 * 64]); // C[1, n:n+16]
                    vmovups(zmm5, ptr[reg_C_cursor + reg_tmp + 1 * 64]); // C[1, n+16:n+32]
                    vmovups(zmm6, ptr[reg_C_cursor + reg_tmp + 2 * 64]); // C[1, n+32:n+48]
                    vmovups(zmm7, ptr[reg_C_cursor + reg_tmp + 3 * 64]); // C[1, n+48:n+64]
                    L("after_c_load");

                    // =====================================================================
                    // K-LOOP: Accumulate dot products over K dimension
                    // =====================================================================
                    // For each K block, we compute:
                    //   C[row, n:n+64] += sum_{k in block} A[row, k] * B[k, n:n+64]
                    //
                    // The loop processes one Q8_1 block (32 elements) per iteration.
                    // Total K iterations = K / 32 (stored in reg_K_blocks).
                    //
                    mov(reg_loop_K, reg_K_blocks);

                    Label loop_K_label;
                    L(loop_K_label);
                    {
                        // =============================================================
                        // ASYMMETRIC QUANTIZATION CORRECTION (IQ4_NL weights)
                        // =============================================================
                        // IQ4_NL uses asymmetric quantization where weights can have
                        // non-zero minimum values. The correction formula is:
                        //
                        //   correction[row] = sum_qs[row] * scale_A[row] * mins[n]
                        //
                        // Where:
                        //   sum_qs[row] = sum of quantized activation values in block
                        //   scale_A[row] = activation scale factor (FP16 in Q8_1 block)
                        //   mins[n] = per-column minimum values from IQ4_NL weights
                        //
                        // This correction is added to C accumulators BEFORE the main VNNI
                        // computation to account for the asymmetric quantization offset.
                        // =============================================================
                        // ---------------------------------------------------------
                        // Step 1-3: Row 0 asymmetric correction preparation
                        // ---------------------------------------------------------
                        // Load sum_qs from Row 0's Q8_1 block (INT16 at offset 2)
                        // Broadcast to all 16 lanes, sign-extend to INT32, convert to FP32
                        vpbroadcastw(ymm_tmp, ptr[reg_A_cursor + 2]); // sum_qs[0] -> all lanes
                        vpmovsxwd(zmm30, ymm_tmp);                    // INT16 -> INT32
                        vcvtdq2ps(zmm30, zmm30);                      // INT32 -> FP32

                        // Load scale from Row 0's Q8_1 block (FP16 at offset 0)
                        // Convert FP16 -> FP32 for computation
                        vpbroadcastw(ymm_tmp, ptr[reg_A_cursor]); // d[0] -> all lanes
                        vcvtph2ps(zmm22, ymm_tmp);                // FP16 -> FP32

                        // Compute sum_qs_0_scaled = sum_qs_0 * scale_A_0
                        vmulps(zmm30, zmm30, zmm22);

                        // ---------------------------------------------------------
                        // Step 4-6: Row 1 asymmetric correction preparation
                        // ---------------------------------------------------------
                        // Row 1's Q8_1 block is at offset A_stride from Row 0
                        mov(reg_tmp, ptr[rsp + 8]);               // params ptr
                        mov(reg_tmp.cvt32(), ptr[reg_tmp + 100]); // A_stride (36 bytes typically)

                        // Load sum_qs from Row 1's Q8_1 block
                        vpbroadcastw(ymm_tmp, ptr[reg_A_cursor + reg_tmp + 2]);
                        vpmovsxwd(zmm31, ymm_tmp);
                        vcvtdq2ps(zmm31, zmm31);

                        // Load scale from Row 1's Q8_1 block
                        vpbroadcastw(ymm_tmp, ptr[reg_A_cursor + reg_tmp]);
                        vcvtph2ps(zmm25, ymm_tmp);

                        // Compute sum_qs_1_scaled = sum_qs_1 * scale_A_1
                        vmulps(zmm31, zmm31, zmm25);

                        // ---------------------------------------------------------
                        // Step 7: Apply correction to C accumulators
                        // ---------------------------------------------------------
                        // Load mins values for current K block (64 FP32 values)
                        // mins pointer may be stack zero buffer if weights are symmetric
                        vmovups(zmm26, ptr[reg_Mins_cursor + 0 * 64]); // mins[n:n+16]
                        vmovups(zmm27, ptr[reg_Mins_cursor + 1 * 64]); // mins[n+16:n+32]
                        vmovups(zmm28, ptr[reg_Mins_cursor + 2 * 64]); // mins[n+32:n+48]
                        vmovups(zmm29, ptr[reg_Mins_cursor + 3 * 64]); // mins[n+48:n+64]

                        // Row 0 correction: C[0,:] += mins[:] * sum_qs_0_scaled
                        // Use zmm16-19 as temporaries
                        vmulps(zmm16, zmm26, zmm30);
                        vmulps(zmm17, zmm27, zmm30);
                        vmulps(zmm18, zmm28, zmm30);
                        vmulps(zmm19, zmm29, zmm30);

                        vaddps(zmm0, zmm0, zmm16); // C[0, n:n+16] += correction
                        vaddps(zmm1, zmm1, zmm17); // C[0, n+16:n+32] += correction
                        vaddps(zmm2, zmm2, zmm18); // C[0, n+32:n+48] += correction
                        vaddps(zmm3, zmm3, zmm19); // C[0, n+48:n+64] += correction

                        // Row 1 correction: C[1,:] += mins[:] * sum_qs_1_scaled
                        vmulps(zmm16, zmm26, zmm31);
                        vmulps(zmm17, zmm27, zmm31);
                        vmulps(zmm18, zmm28, zmm31);
                        vmulps(zmm19, zmm29, zmm31);

                        vaddps(zmm4, zmm4, zmm16); // C[1, n:n+16] += correction
                        vaddps(zmm5, zmm5, zmm17); // C[1, n+16:n+32] += correction
                        vaddps(zmm6, zmm6, zmm18); // C[1, n+32:n+48] += correction
                        vaddps(zmm7, zmm7, zmm19); // C[1, n+48:n+64] += correction

                        // Advance Mins cursor for next K block
                        // Stride is 0 for symmetric weights, reg_stride for asymmetric
                        add(reg_Mins_cursor, ptr[rsp + 264]);

                        // =============================================================
                        // VNNI INTEGER DOT PRODUCT COMPUTATION
                        // =============================================================
                        // Initialize integer accumulators with compensation term:
                        //   comp_init = comp[n] * (-128)
                        //
                        // This compensates for the INT8->UINT8 conversion done to A values.
                        // The VNNI vpdpbusd instruction expects unsigned×signed, so we add
                        // 128 to A values (making them unsigned) and subtract 128*comp here.
                        //
                        // Load Comp values (4 vectors × 16 INT32 = 64 columns)
                        vmovups(zmm_b0, ptr[reg_Comp_cursor + 0 * 64]);
                        vmovups(zmm_b1, ptr[reg_Comp_cursor + 1 * 64]);
                        vmovups(zmm_b2, ptr[reg_Comp_cursor + 2 * 64]);
                        vmovups(zmm_b3, ptr[reg_Comp_cursor + 3 * 64]);

                        // Multiply by -128 to get initialization bias
                        vpmulld(zmm_b0, zmm_b0, zmm_neg_128);
                        vpmulld(zmm_b1, zmm_b1, zmm_neg_128);
                        vpmulld(zmm_b2, zmm_b2, zmm_neg_128);
                        vpmulld(zmm_b3, zmm_b3, zmm_neg_128);

                        // Initialize Row 0 integer accumulators (zmm8-11)
                        vmovdqa64(zmm8, zmm_b0);
                        vmovdqa64(zmm9, zmm_b1);
                        vmovdqa64(zmm10, zmm_b2);
                        vmovdqa64(zmm11, zmm_b3);

                        // Initialize Row 1 integer accumulators (zmm12-15)
                        // Same initialization since both rows use same B weights
                        vmovdqa64(zmm12, zmm_b0);
                        vmovdqa64(zmm13, zmm_b1);
                        vmovdqa64(zmm14, zmm_b2);
                        vmovdqa64(zmm15, zmm_b3);

                        // =============================================================
                        // INNER LOOP: 8 VNNI iterations over 32-element K block
                        // =============================================================
                        // Each iteration processes 4 INT8 elements using vpdpbusd.
                        // 8 iterations × 4 elements = 32 elements per K block.
                        //
                        // vpdpbusd computes: acc += (uint8)a × (int8)b (4-way dot product)
                        // This gives 4 partial sums per lane, accumulated into INT32.
                        //
                        for (int i = 0; i < 8; ++i)
                        {
                            // Load 4 consecutive A values from Row 0's Q8_1 block
                            // Q8_1 layout: [d:FP16][sum_qs:INT16][qs[32]:INT8]
                            // qs starts at offset 4, so element i*4 is at offset 4+i*4
                            vpbroadcastd(zmm_a0, ptr[reg_A_cursor + 4 + i * 4]);
                            vpxord(zmm_a0, zmm_a0, zmm_128); // Convert INT8 → UINT8 by XOR with 0x80808080

                            // Load 4 consecutive A values from Row 1's Q8_1 block
                            // Row 1 is at offset A_stride from Row 0
                            mov(reg_tmp, ptr[rsp + 8]);               // params ptr
                            mov(reg_tmp.cvt32(), ptr[reg_tmp + 100]); // A_stride
                            vpbroadcastd(zmm_a1, ptr[reg_A_cursor + reg_tmp + 4 + i * 4]);
                            vpxord(zmm_a1, zmm_a1, zmm_128); // Convert INT8 → UINT8

                            // Load B weights for 64 columns (4 ZMM registers)
                            // B is packed column-major: B[k, n] for current k position
                            vmovups(zmm_b0, ptr[reg_B_cursor + 0 * 64]); // B[k, n:n+16]
                            vmovups(zmm_b1, ptr[reg_B_cursor + 1 * 64]); // B[k, n+16:n+32]
                            vmovups(zmm_b2, ptr[reg_B_cursor + 2 * 64]); // B[k, n+32:n+48]
                            vmovups(zmm_b3, ptr[reg_B_cursor + 3 * 64]); // B[k, n+48:n+64]

                            // VNNI dot products for Row 0:
                            // acc[col] += (uint8)A[0,k:k+4] · (int8)B[k:k+4, col]
                            vpdpbusd(zmm8, zmm_a0, zmm_b0);  // cols [n:n+16]
                            vpdpbusd(zmm9, zmm_a0, zmm_b1);  // cols [n+16:n+32]
                            vpdpbusd(zmm10, zmm_a0, zmm_b2); // cols [n+32:n+48]
                            vpdpbusd(zmm11, zmm_a0, zmm_b3); // cols [n+48:n+64]

                            // VNNI dot products for Row 1:
                            vpdpbusd(zmm12, zmm_a1, zmm_b0); // cols [n:n+16]
                            vpdpbusd(zmm13, zmm_a1, zmm_b1); // cols [n+16:n+32]
                            vpdpbusd(zmm14, zmm_a1, zmm_b2); // cols [n+32:n+48]
                            vpdpbusd(zmm15, zmm_a1, zmm_b3); // cols [n+48:n+64]

                            // Prefetch upcoming B data (4 iterations ahead = 1024 bytes)
                            prefetcht0(ptr[reg_B_cursor + 1024]);
                            // Prefetch upcoming A data (next K block = 144 bytes ahead)
                            prefetcht0(ptr[reg_A_cursor + 144]);

                            // Advance B cursor by 256 bytes (4 ZMM × 64 bytes)
                            add(reg_B_cursor, 256);
                        }

                        // =============================================================
                        // SCALE APPLICATION: Convert INT32 → FP32 and apply scales
                        // =============================================================
                        // The integer accumulators contain:
                        //   sum_{k} (A_quant[row,k] * B_quant[k,col])
                        //
                        // To convert to floating-point result:
                        //   C[row,col] += acc * scale_A[row] * scale_B[col]
                        //
                        // Convert Row 0 accumulators from INT32 to FP32
                        vcvtdq2ps(zmm8, zmm8);
                        vcvtdq2ps(zmm9, zmm9);
                        vcvtdq2ps(zmm10, zmm10);
                        vcvtdq2ps(zmm11, zmm11);

                        // Convert Row 1 accumulators from INT32 to FP32
                        vcvtdq2ps(zmm12, zmm12);
                        vcvtdq2ps(zmm13, zmm13);
                        vcvtdq2ps(zmm14, zmm14);
                        vcvtdq2ps(zmm15, zmm15);

                        // Load B scales for 64 columns (FP32, 4 vectors)
                        vmovups(zmm_scale_b0, ptr[reg_Scales_cursor + 0 * 64]);
                        vmovups(zmm_scale_b1, ptr[reg_Scales_cursor + 1 * 64]);
                        vmovups(zmm_scale_b2, ptr[reg_Scales_cursor + 2 * 64]);
                        vmovups(zmm_scale_b3, ptr[reg_Scales_cursor + 3 * 64]);

                        // ---------------------------------------------------------
                        // Row 0 scale application: result *= scale_B * scale_A[0]
                        // ---------------------------------------------------------
                        // Load Row 0's scale (FP16 at offset 0 of Q8_1 block)
                        vpbroadcastw(ymm_tmp, ptr[reg_A_cursor]);
                        vcvtph2ps(zmm_scale, ymm_tmp); // FP16 → FP32

                        // Multiply by B scales first, then A scale
                        vmulps(zmm8, zmm8, zmm_scale_b0);
                        vmulps(zmm9, zmm9, zmm_scale_b1);
                        vmulps(zmm10, zmm10, zmm_scale_b2);
                        vmulps(zmm11, zmm11, zmm_scale_b3);

                        vmulps(zmm8, zmm8, zmm_scale);
                        vmulps(zmm9, zmm9, zmm_scale);
                        vmulps(zmm10, zmm10, zmm_scale);
                        vmulps(zmm11, zmm11, zmm_scale);

                        // Accumulate into Row 0 C accumulators
                        vaddps(zmm0, zmm0, zmm8);
                        vaddps(zmm1, zmm1, zmm9);
                        vaddps(zmm2, zmm2, zmm10);
                        vaddps(zmm3, zmm3, zmm11);

                        // ---------------------------------------------------------
                        // Row 1 scale application: result *= scale_B * scale_A[1]
                        // ---------------------------------------------------------
                        // Load Row 1's scale from Q8_1 block at A_stride offset
                        mov(reg_tmp, ptr[rsp + 8]);               // params ptr
                        mov(reg_tmp.cvt32(), ptr[reg_tmp + 100]); // A_stride
                        vpbroadcastw(ymm_tmp, ptr[reg_A_cursor + reg_tmp]);
                        vcvtph2ps(zmm_scale, ymm_tmp); // FP16 → FP32

                        // Multiply by B scales first, then A scale
                        vmulps(zmm12, zmm12, zmm_scale_b0);
                        vmulps(zmm13, zmm13, zmm_scale_b1);
                        vmulps(zmm14, zmm14, zmm_scale_b2);
                        vmulps(zmm15, zmm15, zmm_scale_b3);

                        vmulps(zmm12, zmm12, zmm_scale);
                        vmulps(zmm13, zmm13, zmm_scale);
                        vmulps(zmm14, zmm14, zmm_scale);
                        vmulps(zmm15, zmm15, zmm_scale);

                        // Accumulate into Row 1 C accumulators
                        vaddps(zmm4, zmm4, zmm12);
                        vaddps(zmm5, zmm5, zmm13);
                        vaddps(zmm6, zmm6, zmm14);
                        vaddps(zmm7, zmm7, zmm15);

                        // =============================================================
                        // ADVANCE POINTERS FOR NEXT K BLOCK
                        // =============================================================
                        add(reg_A_cursor, 36);              // sizeof(Q8_1Block) = 36 bytes
                        add(reg_Comp_cursor, reg_stride);   // Move to next K block's comp data
                        add(reg_Scales_cursor, reg_stride); // Move to next K block's scales

                        // Decrement K loop counter and continue if not done
                        dec(reg_loop_K);
                        jnz(loop_K_label, T_NEAR);
                    }

                    // =================================================================
                    // POST-PROCESSING: BIAS, MASK, SOFTMAX, SWIGLU
                    // =================================================================
                    // After the K-loop, zmm0-3 contain Row 0 results and zmm4-7 contain
                    // Row 1 results. The following sections apply optional fused operations.
                    //
                    mov(rax, ptr[rsp + 8]); // Reload params pointer

                    // -----------------------------------------------------------------
                    // BIAS ADDITION
                    // -----------------------------------------------------------------
                    // If bias pointer is non-null, add per-column bias to both rows:
                    //   C[row, col] += bias[col]
                    //
                    // The same bias vector is added to both rows since it's per-column.
                    //
                    mov(rdx, ptr[rax + 64]); // bias ptr from params
                    test(rdx, rdx);          // Check if null
                    Label skip_bias;
                    jz(skip_bias, T_NEAR); // Skip if null
                    {
                        // Calculate bias offset: bias[n] where n = reg_loop_N
                        lea(rdx, ptr[rdx + reg_loop_N * 4]); // bias + n * sizeof(float)

                        // Load bias for columns [n:n+64] and add to both rows
                        vmovups(zmm_bias, ptr[rdx + 0 * 64]); // bias[n:n+16]
                        vaddps(zmm0, zmm0, zmm_bias);         // Row 0
                        vaddps(zmm4, zmm4, zmm_bias);         // Row 1

                        vmovups(zmm_bias, ptr[rdx + 1 * 64]); // bias[n+16:n+32]
                        vaddps(zmm1, zmm1, zmm_bias);
                        vaddps(zmm5, zmm5, zmm_bias);

                        vmovups(zmm_bias, ptr[rdx + 2 * 64]); // bias[n+32:n+48]
                        vaddps(zmm2, zmm2, zmm_bias);
                        vaddps(zmm6, zmm6, zmm_bias);

                        vmovups(zmm_bias, ptr[rdx + 3 * 64]); // bias[n+48:n+64]
                        vaddps(zmm3, zmm3, zmm_bias);
                        vaddps(zmm7, zmm7, zmm_bias);
                    }
                    L(skip_bias);

                    // -----------------------------------------------------------------
                    // ATTENTION MASK
                    // -----------------------------------------------------------------
                    // If mask pointer is non-null, add attention mask to results:
                    //   C[row, col] += mask[row, col]
                    //
                    // The mask is typically filled with -infinity (0xFF800000) for
                    // positions that should not attend, causing them to become 0
                    // after softmax. Each row has a separate mask.
                    //
                    mov(rdx, ptr[rax + 72]); // mask ptr from params
                    test(rdx, rdx);          // Check if null
                    Label skip_mask;
                    jz(skip_mask, T_NEAR); // Skip if null
                    {
                        // Row 0 mask at offset [row * ldc + col]
                        lea(rcx, ptr[rdx + reg_loop_N * 4]); // mask + n * sizeof(float)

                        vmovups(zmm_mask, ptr[rcx + 0 * 64]); // mask[0, n:n+16]
                        vaddps(zmm0, zmm0, zmm_mask);

                        vmovups(zmm_mask, ptr[rcx + 1 * 64]); // mask[0, n+16:n+32]
                        vaddps(zmm1, zmm1, zmm_mask);

                        vmovups(zmm_mask, ptr[rcx + 2 * 64]); // mask[0, n+32:n+48]
                        vaddps(zmm2, zmm2, zmm_mask);

                        vmovups(zmm_mask, ptr[rcx + 3 * 64]); // mask[0, n+48:n+64]
                        vaddps(zmm3, zmm3, zmm_mask);

                        // Row 1 mask at offset [1 * ldc + col]
                        mov(rsi, ptr[rax + 56]); // ldc
                        shl(rsi, 2);             // ldc * sizeof(float)
                        add(rcx, rsi);           // Advance to Row 1 mask

                        vmovups(zmm_mask, ptr[rcx + 0 * 64]); // mask[1, n:n+16]
                        vaddps(zmm4, zmm4, zmm_mask);

                        vmovups(zmm_mask, ptr[rcx + 1 * 64]); // mask[1, n+16:n+32]
                        vaddps(zmm5, zmm5, zmm_mask);

                        vmovups(zmm_mask, ptr[rcx + 2 * 64]); // mask[1, n+32:n+48]
                        vaddps(zmm6, zmm6, zmm_mask);

                        vmovups(zmm_mask, ptr[rcx + 3 * 64]); // mask[1, n+48:n+64]
                        vaddps(zmm7, zmm7, zmm_mask);
                    }
                    L(skip_mask);

                    // -----------------------------------------------------------------
                    // SOFTMAX (First Pass: Local Max and Exp-Sum)
                    // -----------------------------------------------------------------
                    // Implements the first pass of online softmax for attention scores.
                    // This computes local max and sum of exp(x - max) per row.
                    //
                    // Algorithm for each row:
                    //   1. Find max value across 64 columns
                    //   2. Compute exp(x - max) for each element
                    //   3. Sum all exp values
                    //   4. Store local_max and local_sum for later normalization
                    //
                    // The exp() function uses the same polynomial approximation as M1:
                    //   exp(x) ≈ 2^n * P(f) where n = floor(x/ln2), f = frac(x/ln2)
                    //   P(f) = c1 + f*(c2 + f*(c3 + f*(c4 + f*c5)))
                    //
                    mov(al, ptr[rax + 96]); // do_softmax flag from params
                    test(al, al);           // Check if enabled
                    Label skip_softmax;
                    jz(skip_softmax, T_NEAR);
                    {
                        // ---------------------------------------------------------
                        // Register Allocation for Softmax
                        // ---------------------------------------------------------
                        // zmm22: max value (broadcast scalar)
                        // zmm23: running sum of exp values
                        // zmm24: temporary for exp computation
                        // zmm25-29: polynomial coefficients c1-c5
                        // zmm30: 1/ln(2) = 1.44269504
                        // zmm31: max clipping threshold (+10)
                        // zmm8: min clipping threshold (-20)
                        // zmm9: constant 127 for IEEE754 exponent
                        // zmm10-12: temporaries for horizontal reduction
                        //
                        const Zmm &zmm_max = zmm22;
                        const Zmm &zmm_sum = zmm23;
                        const Zmm &zmm_tmp = zmm24;
                        const Zmm &zmm_c1_const = zmm25;
                        const Zmm &zmm_c2_const = zmm26;
                        const Zmm &zmm_c3_coeff = zmm27;
                        const Zmm &zmm_c4_const = zmm28;
                        const Zmm &zmm_c5_const = zmm29;
                        const Zmm &zmm_inv_ln2 = zmm30;
                        const Zmm &zmm_max_clip = zmm31;
                        const Zmm &zmm_min_clip = zmm8;
                        const Zmm &zmm_127 = zmm9;
                        const Zmm &zmm_tmp2 = zmm10;
                        const Zmm &zmm_p = zmm11;
                        const Zmm &zmm_fx = zmm12;

                        // ---------------------------------------------------------
                        // Load Polynomial Coefficients
                        // ---------------------------------------------------------
                        // These constants implement:
                        //   exp(x) = 2^n * P(f)
                        // where n = floor(x * 1.44269504), f = frac(x * 1.44269504)
                        //
                        mov(reg_tmp_32, 0x3fb8aa3b); // 1.44269504 = 1/ln(2)
                        vpbroadcastd(zmm_inv_ln2, reg_tmp_32);

                        mov(reg_tmp_32, 0x3f7ffffe); // 0.99999994 ≈ 1 (c1)
                        vpbroadcastd(zmm_c1_const, reg_tmp_32);

                        mov(reg_tmp_32, 0x3f317218); // 0.69314718 = ln(2) (c2)
                        vpbroadcastd(zmm_c2_const, reg_tmp_32);

                        mov(reg_tmp_32, 0x3e75fdf1); // 0.24022651 = ln(2)²/2 (c3)
                        vpbroadcastd(zmm_c3_coeff, reg_tmp_32);

                        mov(reg_tmp_32, 0x3d6356eb); // 0.05550411 = ln(2)³/6 (c4)
                        vpbroadcastd(zmm_c4_const, reg_tmp_32);

                        mov(reg_tmp_32, 0x3c1d9422); // 0.00961813 = ln(2)⁴/24 (c5)
                        vpbroadcastd(zmm_c5_const, reg_tmp_32);

                        // Clipping bounds to prevent overflow/underflow
                        mov(reg_tmp_32, 0x41200000); // +10.0f (max_clip)
                        vpbroadcastd(zmm_max_clip, reg_tmp_32);

                        mov(reg_tmp_32, 0xc1a00000); // -20.0f (min_clip)
                        vpbroadcastd(zmm_min_clip, reg_tmp_32);

                        // IEEE754 exponent bias for 2^n computation
                        mov(reg_tmp_32, 127);
                        vpbroadcastd(zmm_127, reg_tmp_32);

                        // Reload params (rax was clobbered by constant loads)
                        mov(rax, ptr[rsp + 8]);

                        // ---------------------------------------------------------
                        // Lambda: Compute exp(x - max) and accumulate to sum
                        // ---------------------------------------------------------
                        // This computes: sum += exp(val - max)
                        // Using: exp(x) = 2^n * P(f)
                        //   where n = floor(x/ln2), f = frac(x/ln2)
                        //
                        auto compute_exp_accumulate = [&](const Zmm &zmm_val)
                        {
                            // x = val - max (shift to make max = 0)
                            vsubps(zmm_tmp, zmm_val, zmm_max);

                            // Clip to prevent exp overflow/underflow
                            vminps(zmm_tmp, zmm_tmp, zmm_max_clip);
                            vmaxps(zmm_tmp, zmm_tmp, zmm_min_clip);

                            // x * (1/ln2) = x * 1.44269504
                            vmulps(zmm_tmp, zmm_tmp, zmm_inv_ln2);

                            // n = floor(x/ln2) - the integer part
                            vrndscaleps(zmm_fx, zmm_tmp, 0x01); // Round toward -∞

                            // f = x/ln2 - n - the fractional part
                            vsubps(zmm_tmp, zmm_tmp, zmm_fx);

                            // Horner's method: P(f) = c5*f^4 + c4*f^3 + c3*f^2 + c2*f + c1
                            vmovaps(zmm_p, zmm_c5_const);
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c4_const); // c5*f + c4
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c3_coeff); // (c5*f + c4)*f + c3
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c2_const); // ...
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c1_const); // final polynomial

                            // 2^n via IEEE754 exponent manipulation:
                            // float(n) → int → add 127 → shift to exponent position
                            vcvtps2dq(zmm_fx, zmm_fx);       // Convert n to int
                            vpaddd(zmm_fx, zmm_fx, zmm_127); // Add bias (127)
                            vpslld(zmm_fx, zmm_fx, 23);      // Shift to exponent bits

                            // exp(x) = P(f) * 2^n
                            vmulps(zmm_p, zmm_p, zmm_fx);

                            // Accumulate to sum
                            vaddps(zmm_sum, zmm_sum, zmm_p);
                        };

                        // ---------------------------------------------------------
                        // Row 0 Softmax: Find max, compute exp-sum
                        // ---------------------------------------------------------
                        // Step 1: Horizontal max across 64 elements (zmm0-3)
                        vmaxps(zmm_max, zmm0, zmm1);
                        vmaxps(zmm_max, zmm_max, zmm2);
                        vmaxps(zmm_max, zmm_max, zmm3);

                        // Reduce 512-bit max to scalar:
                        // Extract high 256 bits and max with low
                        vextractf64x4(ymm10, zmm_max, 1);
                        vmaxps(ymm10, ymm10, Ymm(zmm_max.getIdx()));
                        // Extract high 128 bits and max with low
                        vextractf128(xmm11, ymm10, 1);
                        vmaxps(xmm10, xmm10, xmm11);
                        // Shuffle and reduce to scalar
                        vpermilps(xmm11, xmm10, 0b01001110); // Swap 64-bit halves
                        vmaxps(xmm10, xmm10, xmm11);
                        vpermilps(xmm11, xmm10, 0b10110001); // Swap 32-bit pairs
                        vmaxps(xmm10, xmm10, xmm11);

                        // Broadcast scalar max back to zmm
                        vpbroadcastd(zmm_max, xmm10);

                        // Store local_max for Row 0 at index n/16
                        mov(rdx, ptr[rax + 80]); // local_max ptr
                        mov(r15, reg_loop_N);
                        shr(r15, 4); // Column block index: n / 16
                        vmovss(ptr[rdx + r15], xmm10);

                        // Step 2: Compute sum of exp(x - max)
                        vpxord(zmm_sum, zmm_sum, zmm_sum); // Initialize sum = 0
                        compute_exp_accumulate(zmm0);
                        compute_exp_accumulate(zmm1);
                        compute_exp_accumulate(zmm2);
                        compute_exp_accumulate(zmm3);

                        // Reduce sum to scalar (same pattern as max)
                        vextractf64x4(ymm10, zmm_sum, 1);
                        vaddps(ymm10, ymm10, Ymm(zmm_sum.getIdx()));
                        vextractf128(xmm11, ymm10, 1);
                        vaddps(xmm10, xmm10, xmm11);
                        vpermilps(xmm11, xmm10, 0b01001110);
                        vaddps(xmm10, xmm10, xmm11);
                        vpermilps(xmm11, xmm10, 0b10110001);
                        vaddps(xmm10, xmm10, xmm11);

                        // Store local_sum for Row 0
                        mov(rdx, ptr[rax + 88]); // local_sum ptr
                        vmovss(ptr[rdx + r15], xmm10);

                        // ---------------------------------------------------------
                        // Row 1 Softmax: Find max, compute exp-sum
                        // ---------------------------------------------------------
                        // Same algorithm applied to zmm4-7
                        vmaxps(zmm_max, zmm4, zmm5);
                        vmaxps(zmm_max, zmm_max, zmm6);
                        vmaxps(zmm_max, zmm_max, zmm7);

                        // Reduce to scalar
                        vextractf64x4(ymm10, zmm_max, 1);
                        vmaxps(ymm10, ymm10, Ymm(zmm_max.getIdx()));
                        vextractf128(xmm11, ymm10, 1);
                        vmaxps(xmm10, xmm10, xmm11);
                        vpermilps(xmm11, xmm10, 0b01001110);
                        vmaxps(xmm10, xmm10, xmm11);
                        vpermilps(xmm11, xmm10, 0b10110001);
                        vmaxps(xmm10, xmm10, xmm11);

                        vpbroadcastd(zmm_max, xmm10);

                        // Store local_max for Row 1 at index (N/16) + (n/16)
                        // Row 1 offset in local arrays = N / 16
                        mov(rdx.cvt32(), ptr[rax + 52]); // N dimension
                        shr(rdx, 4);                     // N / 16
                        add(rdx, r15);                   // Row 1 index

                        mov(rcx, ptr[rax + 80]);       // local_max ptr
                        vmovss(ptr[rcx + rdx], xmm10); // Store Row 1 local_max

                        // Compute Row 1 exp-sum
                        vpxord(zmm_sum, zmm_sum, zmm_sum);
                        compute_exp_accumulate(zmm4);
                        compute_exp_accumulate(zmm5);
                        compute_exp_accumulate(zmm6);
                        compute_exp_accumulate(zmm7);

                        // Reduce sum to scalar
                        vextractf64x4(ymm10, zmm_sum, 1);
                        vaddps(ymm10, ymm10, Ymm(zmm_sum.getIdx()));
                        vextractf128(xmm11, ymm10, 1);
                        vaddps(xmm10, xmm10, xmm11);
                        vpermilps(xmm11, xmm10, 0b01001110);
                        vaddps(xmm10, xmm10, xmm11);
                        vpermilps(xmm11, xmm10, 0b10110001);
                        vaddps(xmm10, xmm10, xmm11);

                        // Store Row 1 local_sum
                        mov(rcx, ptr[rax + 88]); // local_sum ptr
                        vmovss(ptr[rcx + rdx], xmm10);
                    }
                    L(skip_softmax);

                    // -----------------------------------------------------------------
                    // SWIGLU ACTIVATION FUNCTION
                    // -----------------------------------------------------------------
                    // Implements SwiGLU (Swish-Gated Linear Unit) for FFN blocks:
                    //   output = up_proj * sigmoid(gate_proj) * gate_proj
                    //          = up_proj * swish(gate_proj)
                    //
                    // Where:
                    //   - up_proj is the current value in zmm0-7 (result of up projection)
                    //   - gate_proj is loaded from gate_input pointer (result of gate projection)
                    //   - swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
                    //
                    // This fuses the element-wise multiplication with the gate projection.
                    //
                    mov(rax, ptr[rsp + 8]);  // params
                    cmp(byte[rax + 112], 1); // do_swiglu flag
                    Label skip_swiglu;
                    jne(skip_swiglu, T_NEAR); // Skip if not enabled
                    {
                        // ---------------------------------------------------------
                        // Register Allocation for SwiGLU
                        // ---------------------------------------------------------
                        // zmm8: gate value loaded from memory
                        // zmm9: temporary for sigmoid computation
                        // zmm10-14: polynomial coefficients c1-c5
                        // zmm15: 1/ln(2)
                        // zmm16-17: clipping bounds
                        // zmm18: constant 127
                        // zmm19: constant 1.0f
                        // zmm20-21: temporaries for exp and polynomial
                        //
                        const Zmm &zmm_gate = zmm8;
                        const Zmm &zmm_tmp = zmm9;
                        const Zmm &zmm_c1_const = zmm10;
                        const Zmm &zmm_c2_const = zmm11;
                        const Zmm &zmm_c3_coeff = zmm12;
                        const Zmm &zmm_c4_const = zmm13;
                        const Zmm &zmm_c5_const = zmm14;
                        const Zmm &zmm_inv_ln2 = zmm15;
                        const Zmm &zmm_max_clip = zmm16;
                        const Zmm &zmm_min_clip = zmm17;
                        const Zmm &zmm_127 = zmm18;
                        const Zmm &zmm_one = zmm19;
                        const Zmm &zmm_fx = zmm20;
                        const Zmm &zmm_p = zmm21;

                        // Load polynomial constants (same as softmax exp)
                        mov(reg_tmp_32, 0x3fb8aa3b); // 1/ln(2)
                        vpbroadcastd(zmm_inv_ln2, reg_tmp_32);
                        mov(reg_tmp_32, 0x3f7ffffe); // c1
                        vpbroadcastd(zmm_c1_const, reg_tmp_32);
                        mov(reg_tmp_32, 0x3f317218); // c2
                        vpbroadcastd(zmm_c2_const, reg_tmp_32);
                        mov(reg_tmp_32, 0x3e75fdf1); // c3
                        vpbroadcastd(zmm_c3_coeff, reg_tmp_32);
                        mov(reg_tmp_32, 0x3d6356eb); // c4
                        vpbroadcastd(zmm_c4_const, reg_tmp_32);
                        mov(reg_tmp_32, 0x3c1d9422); // c5
                        vpbroadcastd(zmm_c5_const, reg_tmp_32);

                        // Wider clipping bounds for SwiGLU (±88.0f)
                        mov(reg_tmp_32, 0x42b00000); // +88.0f
                        vpbroadcastd(zmm_max_clip, reg_tmp_32);
                        mov(reg_tmp_32, 0xc2b00000); // -88.0f
                        vpbroadcastd(zmm_min_clip, reg_tmp_32);

                        mov(reg_tmp_32, 0x3f800000); // 1.0f
                        vpbroadcastd(zmm_one, reg_tmp_32);
                        mov(reg_tmp_32, 127);
                        vpbroadcastd(zmm_127, reg_tmp_32);

                        // Reload params (rax was clobbered by constant loads)
                        mov(rax, ptr[rsp + 8]);

                        // Load gate_input pointer
                        mov(rdx, ptr[rax + 104]); // gate_input from params

                        // Calculate Row 1 offset (ldc * sizeof(float))
                        mov(reg_tmp.cvt32(), ptr[rax + 56]); // ldc
                        shl(reg_tmp, 2);                     // ldc * 4

                        // ---------------------------------------------------------
                        // Lambda: Compute swish(gate) and multiply with value
                        // ---------------------------------------------------------
                        // swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
                        //
                        // Algorithm:
                        //   1. Load gate value
                        //   2. Compute exp(-gate) using polynomial
                        //   3. sigmoid = 1 / (1 + exp(-gate))
                        //   4. swish = gate * sigmoid
                        //   5. result = value * swish
                        //
                        auto compute_swish = [&](const Zmm &zmm_val, const Reg64 &base, int offset)
                        {
                            // Load gate projection value
                            vmovups(zmm_gate, ptr[base + offset * 64]);

                            // Compute sigmoid(gate) = 1 / (1 + exp(-gate))
                            // First compute exp(-gate):
                            vxorps(zmm_tmp, zmm_tmp, zmm_tmp);  // 0.0f
                            vsubps(zmm_tmp, zmm_tmp, zmm_gate); // -gate

                            // Clip to prevent overflow
                            vminps(zmm_tmp, zmm_tmp, zmm_max_clip);
                            vmaxps(zmm_tmp, zmm_tmp, zmm_min_clip);

                            // exp(-gate) using same polynomial as softmax
                            vmulps(zmm_tmp, zmm_tmp, zmm_inv_ln2);
                            vrndscaleps(zmm_fx, zmm_tmp, 0x01); // floor
                            vsubps(zmm_tmp, zmm_tmp, zmm_fx);   // fractional part

                            // Horner's polynomial evaluation
                            vmovaps(zmm_p, zmm_c5_const);
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c4_const);
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c3_coeff);
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c2_const);
                            vfmadd213ps(zmm_p, zmm_tmp, zmm_c1_const);

                            // 2^n via IEEE754 manipulation
                            vcvtps2dq(zmm_fx, zmm_fx);
                            vpaddd(zmm_fx, zmm_fx, zmm_127);
                            vpslld(zmm_fx, zmm_fx, 23);
                            vmulps(zmm_p, zmm_p, zmm_fx); // exp(-gate)

                            // sigmoid = 1 / (1 + exp(-gate))
                            // We compute: swish = gate / (1 + exp(-gate))
                            vaddps(zmm_p, zmm_p, zmm_one);  // 1 + exp(-gate)
                            vdivps(zmm_p, zmm_gate, zmm_p); // gate / (1 + exp(-gate)) = swish

                            // output = value * swish(gate)
                            vmulps(zmm_val, zmm_val, zmm_p);
                        };

                        // Apply SwiGLU to Row 0 (zmm0-3)
                        compute_swish(zmm0, rdx, 0); // columns [n:n+16]
                        compute_swish(zmm1, rdx, 1); // columns [n+16:n+32]
                        compute_swish(zmm2, rdx, 2); // columns [n+32:n+48]
                        compute_swish(zmm3, rdx, 3); // columns [n+48:n+64]

                        // Advance to Row 1 gate values
                        add(rdx, reg_tmp); // gate_cursor += ldc * 4

                        // Apply SwiGLU to Row 1 (zmm4-7)
                        compute_swish(zmm4, rdx, 0);
                        compute_swish(zmm5, rdx, 1);
                        compute_swish(zmm6, rdx, 2);
                        compute_swish(zmm7, rdx, 3);
                    }
                    L(skip_swiglu);

                    // =================================================================
                    // EPILOGUE: Store results to C matrix
                    // =================================================================
                    // =================================================================
                    // OUTPUT STORE: FP32, Q8_1, or Q16_1 format based on output_format
                    // =================================================================
                    // Check output_format from params (offset 113)
                    mov(rax, ptr[rsp + 8]);      // params
                    movzx(ecx, byte[rax + 113]); // output_format (uint8_t)
                    cmp(ecx, static_cast<int>(GemmOutputFormat::Q8_1));
                    je("q8_1_store_m2", T_NEAR);
                    cmp(ecx, static_cast<int>(GemmOutputFormat::Q16_1));
                    je("q16_1_store_m2", T_NEAR);

                    // -----------------------------------------------------------------
                    // FP32 OUTPUT PATH (output_format == 0)
                    // -----------------------------------------------------------------
                    // Row 0 is at reg_C_cursor, Row 1 is at reg_C_cursor + ldc * 4.
                    mov(reg_tmp.cvt32(), ptr[rax + 56]); // ldc
                    shl(reg_tmp, 2);                     // ldc * sizeof(float)

                    // Store Row 0 results (64 FP32 values)
                    vmovups(ptr[reg_C_cursor + 0 * 64], zmm0); // C[0, n:n+16]
                    vmovups(ptr[reg_C_cursor + 1 * 64], zmm1); // C[0, n+16:n+32]
                    vmovups(ptr[reg_C_cursor + 2 * 64], zmm2); // C[0, n+32:n+48]
                    vmovups(ptr[reg_C_cursor + 3 * 64], zmm3); // C[0, n+48:n+64]

                    // Store Row 1 results at offset ldc * sizeof(float)
                    add(reg_C_cursor, reg_tmp);                // Advance to Row 1
                    vmovups(ptr[reg_C_cursor + 0 * 64], zmm4); // C[1, n:n+16]
                    vmovups(ptr[reg_C_cursor + 1 * 64], zmm5); // C[1, n+16:n+32]
                    vmovups(ptr[reg_C_cursor + 2 * 64], zmm6); // C[1, n+32:n+48]
                    vmovups(ptr[reg_C_cursor + 3 * 64], zmm7); // C[1, n+48:n+64]
                    sub(reg_C_cursor, reg_tmp);                // Restore to Row 0 position

                    // Advance C cursor to next N block (64 columns = 256 bytes)
                    add(reg_C_cursor, 256);
                    jmp("after_store_m2", T_NEAR);

                    // -----------------------------------------------------------------
                    // Q8_1 OUTPUT PATH (output_format == 1)
                    // -----------------------------------------------------------------
                    // Convert 64 FP32 values per row → 2 Q8_1 blocks per row
                    // Q8_1Block = { float16 d (scale), int16 sum_qs, int8 qs[32] } = 36 bytes
                    // Row 0: zmm0-zmm3 → blocks at C_q8_1 + (n/32) * 36
                    // Row 1: zmm4-zmm7 → blocks at C_q8_1 + C_q8_1_stride + (n/32) * 36
                    //
                    // IMPORTANT: Use xmm8-xmm15 for VEX-only operations (vrcpss, etc.)
                    // Extended registers (xmm16+) require EVEX encoding which some
                    // instructions don't support.
                    L("q8_1_store_m2");
                    {
                        // Load Q8_1 output pointer and stride
                        mov(rdx, ptr[rax + 120]); // C_q8_1 pointer (row 0)
                        mov(ebx, ptr[rax + 128]); // C_q8_1_stride (int, 4 bytes!)

                        // Calculate block offset: (n_iter * 64 / 32) * 36 = n_iter * 2 * 36 = n_iter * 72
                        // reg_loop_N contains current n value, divide by 64 to get iteration
                        mov(rax, reg_loop_N); // Current n
                        shr(rax, 5);          // n / 32 = block index for first of the 2 blocks
                        imul(rax, rax, 36);   // block_index * 36 bytes
                        add(rdx, rax);        // rdx = C_q8_1 + block_offset (row 0)

                        // Save row 1 base address in r9
                        lea(r9, ptr[rdx + rbx]); // r9 = C_q8_1 + stride + block_offset (row 1)

                        // =============================================================
                        // ROW 0, BLOCK 0: zmm0 (16 floats) + zmm1 (16 floats) → 32 INT8 + scale
                        // =============================================================
                        // Find max absolute value across zmm0 and zmm1
                        vrangeps(zmm16, zmm0, zmm0, 0x0B); // |zmm0|
                        vrangeps(zmm17, zmm1, zmm1, 0x0B); // |zmm1|
                        vmaxps(zmm16, zmm16, zmm17);       // max(|zmm0|, |zmm1|)

                        // Horizontal reduce to find max across zmm16
                        vextractf32x8(ymm17, zmm16, 1); // Upper 256 bits
                        vmaxps(ymm16, ymm16, ymm17);
                        vextractf32x4(xmm17, ymm16, 1); // EVEX: vextractf32x4 for extended regs
                        vmaxps(xmm16, xmm16, xmm17);
                        vshufps(xmm17, xmm16, xmm16, 0x4E); // Swap pairs
                        vmaxps(xmm16, xmm16, xmm17);
                        vshufps(xmm17, xmm16, xmm16, 0xB1); // Swap elements
                        vmaxps(xmm16, xmm16, xmm17);        // xmm16[0] = max_abs

                        // Copy to low register for VEX operations
                        vmovaps(xmm8, xmm16); // xmm8 = max_abs

                        // Compute scale: d = max_abs / 127.0f
                        mov(eax, 0x42FE0000); // 127.0f in hex
                        vmovd(xmm9, eax);
                        vdivss(xmm10, xmm8, xmm9); // xmm10 = d = max_abs / 127

                        // Handle zero case: if max_abs == 0, set d = 1.0 to avoid div by zero
                        vxorps(xmm11, xmm11, xmm11);
                        vucomiss(xmm8, xmm11);
                        mov(eax, 0x3F800000); // 1.0f
                        vmovd(xmm11, eax);
                        jnz("r0_b0_scale_ok", T_NEAR);
                        vmovss(xmm10, xmm11); // d = 1.0f if max_abs == 0
                        L("r0_b0_scale_ok");

                        // Compute id = 1.0f / d for quantization (VEX-only vrcpss)
                        vrcpss(xmm12, xmm10, xmm10); // xmm12 = id ≈ 1.0f / d
                        vbroadcastss(zmm20, xmm12);  // Broadcast id to zmm20

                        // Quantize: qs[i] = round(x[i] * id)
                        vmulps(zmm21, zmm0, zmm20); // zmm0 * id
                        vmulps(zmm22, zmm1, zmm20); // zmm1 * id

                        // Convert to INT32 with rounding
                        vcvtps2dq(zmm21, zmm21);
                        vcvtps2dq(zmm22, zmm22);

                        // Pack INT32 → INT8 using vpmovdb (saturating)
                        vpmovdb(xmm24, zmm21); // 16 int32 → 16 int8
                        vpmovdb(xmm25, zmm22); // 16 int32 → 16 int8

                        // Combine two xmm into ymm
                        vinserti32x4(ymm24, ymm24, xmm25, 1); // ymm24 = [xmm24, xmm25]

                        // Compute sum_qs using sign-extended INT32 values
                        vpmovsxbd(zmm26, xmm24);        // Sign-extend low 16 bytes
                        vextracti32x4(xmm27, ymm24, 1); // EVEX: Get upper 16 bytes
                        vpmovsxbd(zmm28, xmm27);        // Sign-extend to 16 INT32
                        vpaddd(zmm26, zmm26, zmm28);    // Sum both halves

                        // Horizontal sum reduction
                        vextracti32x8(ymm27, zmm26, 1);
                        vpaddd(ymm26, ymm26, ymm27);
                        vextracti32x4(xmm27, ymm26, 1); // EVEX: vextracti32x4 for extended regs
                        vpaddd(xmm26, xmm26, xmm27);
                        vpshufd(xmm27, xmm26, 0x4E);
                        vpaddd(xmm26, xmm26, xmm27);
                        vpshufd(xmm27, xmm26, 0xB1);
                        vpaddd(xmm26, xmm26, xmm27); // xmm26[0] = sum_qs

                        // Store Q8_1Block for Row 0, Block 0 at rdx
                        vcvtps2ph(xmm23, xmm10, 0x00);   // FP32 → FP16 (use xmm10 which has d)
                        vpextrw(ptr[rdx + 0], xmm23, 0); // d (2 bytes)

                        vmovd(eax, xmm26);
                        mov(word[rdx + 2], ax); // sum_qs (2 bytes)

                        vmovdqu32(ptr[rdx + 4], ymm24); // 32 bytes of qs (EVEX for ymm24)

                        // =============================================================
                        // ROW 0, BLOCK 1: zmm2 (16 floats) + zmm3 (16 floats) → 32 INT8 + scale
                        // =============================================================
                        vrangeps(zmm16, zmm2, zmm2, 0x0B);
                        vrangeps(zmm17, zmm3, zmm3, 0x0B);
                        vmaxps(zmm16, zmm16, zmm17);

                        vextractf32x8(ymm17, zmm16, 1);
                        vmaxps(ymm16, ymm16, ymm17);
                        vextractf32x4(xmm17, ymm16, 1); // EVEX
                        vmaxps(xmm16, xmm16, xmm17);
                        vshufps(xmm17, xmm16, xmm16, 0x4E);
                        vmaxps(xmm16, xmm16, xmm17);
                        vshufps(xmm17, xmm16, xmm16, 0xB1);
                        vmaxps(xmm16, xmm16, xmm17);

                        vmovaps(xmm8, xmm16);
                        mov(eax, 0x42FE0000);
                        vmovd(xmm9, eax);
                        vdivss(xmm10, xmm8, xmm9);

                        vxorps(xmm11, xmm11, xmm11);
                        vucomiss(xmm8, xmm11);
                        mov(eax, 0x3F800000);
                        vmovd(xmm11, eax);
                        jnz("r0_b1_scale_ok", T_NEAR);
                        vmovss(xmm10, xmm11);
                        L("r0_b1_scale_ok");

                        vrcpss(xmm12, xmm10, xmm10);
                        vbroadcastss(zmm20, xmm12);

                        vmulps(zmm21, zmm2, zmm20);
                        vmulps(zmm22, zmm3, zmm20);
                        vcvtps2dq(zmm21, zmm21);
                        vcvtps2dq(zmm22, zmm22);

                        vpmovdb(xmm24, zmm21);
                        vpmovdb(xmm25, zmm22);
                        vinserti32x4(ymm24, ymm24, xmm25, 1);

                        vpmovsxbd(zmm26, xmm24);
                        vextracti32x4(xmm27, ymm24, 1);
                        vpmovsxbd(zmm28, xmm27);
                        vpaddd(zmm26, zmm26, zmm28);
                        vextracti32x8(ymm27, zmm26, 1);
                        vpaddd(ymm26, ymm26, ymm27);
                        vextracti32x4(xmm27, ymm26, 1);
                        vpaddd(xmm26, xmm26, xmm27);
                        vpshufd(xmm27, xmm26, 0x4E);
                        vpaddd(xmm26, xmm26, xmm27);
                        vpshufd(xmm27, xmm26, 0xB1);
                        vpaddd(xmm26, xmm26, xmm27);

                        vcvtps2ph(xmm23, xmm10, 0x00);
                        vpextrw(ptr[rdx + 36 + 0], xmm23, 0);
                        vmovd(eax, xmm26);
                        mov(word[rdx + 36 + 2], ax);
                        vmovdqu32(ptr[rdx + 36 + 4], ymm24); // EVEX for ymm24

                        // =============================================================
                        // ROW 1, BLOCK 0: zmm4 (16 floats) + zmm5 (16 floats) → 32 INT8 + scale
                        // =============================================================
                        vrangeps(zmm16, zmm4, zmm4, 0x0B);
                        vrangeps(zmm17, zmm5, zmm5, 0x0B);
                        vmaxps(zmm16, zmm16, zmm17);

                        vextractf32x8(ymm17, zmm16, 1);
                        vmaxps(ymm16, ymm16, ymm17);
                        vextractf32x4(xmm17, ymm16, 1);
                        vmaxps(xmm16, xmm16, xmm17);
                        vshufps(xmm17, xmm16, xmm16, 0x4E);
                        vmaxps(xmm16, xmm16, xmm17);
                        vshufps(xmm17, xmm16, xmm16, 0xB1);
                        vmaxps(xmm16, xmm16, xmm17);

                        vmovaps(xmm8, xmm16);
                        mov(eax, 0x42FE0000);
                        vmovd(xmm9, eax);
                        vdivss(xmm10, xmm8, xmm9);

                        vxorps(xmm11, xmm11, xmm11);
                        vucomiss(xmm8, xmm11);
                        mov(eax, 0x3F800000);
                        vmovd(xmm11, eax);
                        jnz("r1_b0_scale_ok", T_NEAR);
                        vmovss(xmm10, xmm11);
                        L("r1_b0_scale_ok");

                        vrcpss(xmm12, xmm10, xmm10);
                        vbroadcastss(zmm20, xmm12);

                        vmulps(zmm21, zmm4, zmm20);
                        vmulps(zmm22, zmm5, zmm20);
                        vcvtps2dq(zmm21, zmm21);
                        vcvtps2dq(zmm22, zmm22);

                        vpmovdb(xmm24, zmm21);
                        vpmovdb(xmm25, zmm22);
                        vinserti32x4(ymm24, ymm24, xmm25, 1);

                        vpmovsxbd(zmm26, xmm24);
                        vextracti32x4(xmm27, ymm24, 1);
                        vpmovsxbd(zmm28, xmm27);
                        vpaddd(zmm26, zmm26, zmm28);
                        vextracti32x8(ymm27, zmm26, 1);
                        vpaddd(ymm26, ymm26, ymm27);
                        vextracti32x4(xmm27, ymm26, 1);
                        vpaddd(xmm26, xmm26, xmm27);
                        vpshufd(xmm27, xmm26, 0x4E);
                        vpaddd(xmm26, xmm26, xmm27);
                        vpshufd(xmm27, xmm26, 0xB1);
                        vpaddd(xmm26, xmm26, xmm27);

                        vcvtps2ph(xmm23, xmm10, 0x00);
                        vpextrw(ptr[r9 + 0], xmm23, 0);
                        vmovd(eax, xmm26);
                        mov(word[r9 + 2], ax);
                        vmovdqu32(ptr[r9 + 4], ymm24); // EVEX for ymm24

                        // =============================================================
                        // ROW 1, BLOCK 1: zmm6 (16 floats) + zmm7 (16 floats) → 32 INT8 + scale
                        // =============================================================
                        vrangeps(zmm16, zmm6, zmm6, 0x0B);
                        vrangeps(zmm17, zmm7, zmm7, 0x0B);
                        vmaxps(zmm16, zmm16, zmm17);

                        vextractf32x8(ymm17, zmm16, 1);
                        vmaxps(ymm16, ymm16, ymm17);
                        vextractf32x4(xmm17, ymm16, 1);
                        vmaxps(xmm16, xmm16, xmm17);
                        vshufps(xmm17, xmm16, xmm16, 0x4E);
                        vmaxps(xmm16, xmm16, xmm17);
                        vshufps(xmm17, xmm16, xmm16, 0xB1);
                        vmaxps(xmm16, xmm16, xmm17);

                        vmovaps(xmm8, xmm16);
                        mov(eax, 0x42FE0000);
                        vmovd(xmm9, eax);
                        vdivss(xmm10, xmm8, xmm9);

                        vxorps(xmm11, xmm11, xmm11);
                        vucomiss(xmm8, xmm11);
                        mov(eax, 0x3F800000);
                        vmovd(xmm11, eax);
                        jnz("r1_b1_scale_ok", T_NEAR);
                        vmovss(xmm10, xmm11);
                        L("r1_b1_scale_ok");

                        vrcpss(xmm12, xmm10, xmm10);
                        vbroadcastss(zmm20, xmm12);

                        vmulps(zmm21, zmm6, zmm20);
                        vmulps(zmm22, zmm7, zmm20);
                        vcvtps2dq(zmm21, zmm21);
                        vcvtps2dq(zmm22, zmm22);

                        vpmovdb(xmm24, zmm21);
                        vpmovdb(xmm25, zmm22);
                        vinserti32x4(ymm24, ymm24, xmm25, 1);

                        vpmovsxbd(zmm26, xmm24);
                        vextracti32x4(xmm27, ymm24, 1);
                        vpmovsxbd(zmm28, xmm27);
                        vpaddd(zmm26, zmm26, zmm28);
                        vextracti32x8(ymm27, zmm26, 1);
                        vpaddd(ymm26, ymm26, ymm27);
                        vextracti32x4(xmm27, ymm26, 1);
                        vpaddd(xmm26, xmm26, xmm27);
                        vpshufd(xmm27, xmm26, 0x4E);
                        vpaddd(xmm26, xmm26, xmm27);
                        vpshufd(xmm27, xmm26, 0xB1);
                        vpaddd(xmm26, xmm26, xmm27);

                        vcvtps2ph(xmm23, xmm10, 0x00);
                        vpextrw(ptr[r9 + 36 + 0], xmm23, 0);
                        vmovd(eax, xmm26);
                        mov(word[r9 + 36 + 2], ax);
                        vmovdqu32(ptr[r9 + 36 + 4], ymm24); // EVEX for ymm24

                        // Restore reg_stride (rbx) for next N-loop iteration
                        // Q8_1 store clobbered rbx with C_q8_1_stride
                        mov(rax, ptr[rsp + 8]);                 // Reload params
                        mov(reg_stride.cvt32(), ptr[rax + 56]); // ldc
                        shl(reg_stride, 2);                     // ldc * sizeof(float)
                    }
                    // No cursor advance needed for Q8_1 - block offset computed from loop_N
                    jmp("after_store_m2", T_NEAR);

                    // -----------------------------------------------------------------
                    // Q16_1 OUTPUT PATH (output_format == 4)
                    // -----------------------------------------------------------------
                    // Convert 64 FP32 values per row → 1 Q16_1Block_64 per row
                    // Q16_1Block_64 = { float d (scale), int32 sum_qs, int16 qs[64] } = 136 bytes
                    // Row 0: zmm0-zmm3 → block at C_q16_1 + (loop_N/64) * 136
                    // Row 1: zmm4-zmm7 → block at C_q16_1 + C_q16_1_stride + (loop_N/64) * 136
                    L("q16_1_store_m2");
                    {
                        // Load Q16_1 output pointer and stride
                        // Note: Save C_q16_1 pointer in r11 since rdx/edx is used by div
                        mov(rax, ptr[rsp + 8]);   // params
                        mov(r11, ptr[rax + 136]); // C_q16_1 pointer - save in r11
                        mov(ebx, ptr[rax + 144]); // C_q16_1_stride (offset 144)
                        mov(ecx, ptr[rax + 148]); // q16_block_size (offset 148)

                        // Calculate block byte size = 8 + 2 * block_size
                        mov(r8d, ecx);
                        shl(r8d, 1);
                        add(r8d, 8); // r8 = block_bytes

                        // Calculate block index = loop_N / block_size
                        mov(eax, reg_loop_N.cvt32()); // eax = loop_N (32-bit)
                        xor_(edx, edx);               // Clear for division (edx:eax / ecx)
                        div(ecx);                     // eax = loop_N / block_size
                        imul(eax, r8d);               // eax = block_index * block_bytes

                        // Set up output pointers
                        mov(rdx, r11);            // rdx = C_q16_1 base (restored from r11)
                        add(rdx, rax);            // rdx = Row 0 output pointer
                        mov(rcx, ptr[rsp + 8]);   // Reload params
                        mov(ebx, ptr[rcx + 144]); // Reload stride
                        lea(r9, ptr[rdx + rbx]);  // r9 = Row 1 output pointer

                        // =============================================================
                        // ROW 0: Quantize zmm0-zmm3 (64 floats) to Q16_1Block_64
                        // =============================================================
                        // Find max absolute value across all 64 floats
                        vrangeps(zmm16, zmm0, zmm0, 0x0B); // abs(zmm0)
                        vrangeps(zmm17, zmm1, zmm1, 0x0B);
                        vrangeps(zmm18, zmm2, zmm2, 0x0B);
                        vrangeps(zmm19, zmm3, zmm3, 0x0B);
                        vmaxps(zmm16, zmm16, zmm17);
                        vmaxps(zmm18, zmm18, zmm19);
                        vmaxps(zmm16, zmm16, zmm18);

                        // Horizontal max reduction
                        vextractf32x8(ymm17, zmm16, 1);
                        vmaxps(ymm16, ymm16, ymm17);
                        vextractf32x4(xmm17, ymm16, 1);
                        vmaxps(xmm16, xmm16, xmm17);
                        vshufps(xmm17, xmm16, xmm16, 0x4E);
                        vmaxps(xmm16, xmm16, xmm17);
                        vshufps(xmm17, xmm16, xmm16, 0xB1);
                        vmaxps(xmm16, xmm16, xmm17);
                        vmovaps(xmm8, xmm16); // xmm8 = max_abs

                        // Compute scale: d = max_abs / 16383.0f
                        mov(eax, 0x467FFE00); // 16383.0f
                        vmovd(xmm9, eax);
                        vdivss(xmm10, xmm8, xmm9); // xmm10 = d

                        // Store scale as FP32 (4 bytes at offset 0)
                        vmovss(ptr[rdx + 0], xmm10);

                        // Compute inverse scale
                        vrcpss(xmm12, xmm10, xmm10);
                        vbroadcastss(zmm20, xmm12);

                        // Quantize to int32
                        vmulps(zmm21, zmm0, zmm20);
                        vmulps(zmm22, zmm1, zmm20);
                        vmulps(zmm23, zmm2, zmm20);
                        vmulps(zmm24, zmm3, zmm20);
                        vcvtps2dq(zmm21, zmm21);
                        vcvtps2dq(zmm22, zmm22);
                        vcvtps2dq(zmm23, zmm23);
                        vcvtps2dq(zmm24, zmm24);

                        // Pack int32 to int16 with saturation
                        vpmovsdw(ymm21, zmm21);
                        vpmovsdw(ymm22, zmm22);
                        vpmovsdw(ymm23, zmm23);
                        vpmovsdw(ymm24, zmm24);
                        vinserti64x4(zmm21, zmm21, ymm22, 1); // 32 int16
                        vinserti64x4(zmm23, zmm23, ymm24, 1); // 32 int16

                        // Compute sum_qs
                        vpmovsxwd(zmm25, ymm21);
                        vextracti64x4(ymm26, zmm21, 1);
                        vpmovsxwd(zmm26, ymm26);
                        vpaddd(zmm25, zmm25, zmm26);
                        vpmovsxwd(zmm27, ymm23);
                        vextracti64x4(ymm28, zmm23, 1);
                        vpmovsxwd(zmm28, ymm28);
                        vpaddd(zmm27, zmm27, zmm28);
                        vpaddd(zmm25, zmm25, zmm27);

                        // Horizontal sum
                        vextracti32x8(ymm26, zmm25, 1);
                        vpaddd(ymm25, ymm25, ymm26);
                        vextracti32x4(xmm26, ymm25, 1);
                        vpaddd(xmm25, xmm25, xmm26);
                        vpshufd(xmm26, xmm25, 0x4E);
                        vpaddd(xmm25, xmm25, xmm26);
                        vpshufd(xmm26, xmm25, 0xB1);
                        vpaddd(xmm25, xmm25, xmm26);

                        // Store sum_qs as INT32 (4 bytes at offset 4)
                        vmovd(ptr[rdx + 4], xmm25);

                        // Store quantized int16 values (128 bytes at offset 8)
                        vmovdqu32(ptr[rdx + 8], zmm21);
                        vmovdqu32(ptr[rdx + 8 + 64], zmm23);

                        // =============================================================
                        // ROW 1: Quantize zmm4-zmm7 (64 floats) to Q16_1Block_64
                        // =============================================================
                        vrangeps(zmm16, zmm4, zmm4, 0x0B);
                        vrangeps(zmm17, zmm5, zmm5, 0x0B);
                        vrangeps(zmm18, zmm6, zmm6, 0x0B);
                        vrangeps(zmm19, zmm7, zmm7, 0x0B);
                        vmaxps(zmm16, zmm16, zmm17);
                        vmaxps(zmm18, zmm18, zmm19);
                        vmaxps(zmm16, zmm16, zmm18);

                        vextractf32x8(ymm17, zmm16, 1);
                        vmaxps(ymm16, ymm16, ymm17);
                        vextractf32x4(xmm17, ymm16, 1);
                        vmaxps(xmm16, xmm16, xmm17);
                        vshufps(xmm17, xmm16, xmm16, 0x4E);
                        vmaxps(xmm16, xmm16, xmm17);
                        vshufps(xmm17, xmm16, xmm16, 0xB1);
                        vmaxps(xmm16, xmm16, xmm17);
                        vmovaps(xmm8, xmm16);

                        mov(eax, 0x467FFE00);
                        vmovd(xmm9, eax);
                        vdivss(xmm10, xmm8, xmm9);
                        vmovss(ptr[r9 + 0], xmm10);

                        vrcpss(xmm12, xmm10, xmm10);
                        vbroadcastss(zmm20, xmm12);

                        vmulps(zmm21, zmm4, zmm20);
                        vmulps(zmm22, zmm5, zmm20);
                        vmulps(zmm23, zmm6, zmm20);
                        vmulps(zmm24, zmm7, zmm20);
                        vcvtps2dq(zmm21, zmm21);
                        vcvtps2dq(zmm22, zmm22);
                        vcvtps2dq(zmm23, zmm23);
                        vcvtps2dq(zmm24, zmm24);

                        vpmovsdw(ymm21, zmm21);
                        vpmovsdw(ymm22, zmm22);
                        vpmovsdw(ymm23, zmm23);
                        vpmovsdw(ymm24, zmm24);
                        vinserti64x4(zmm21, zmm21, ymm22, 1);
                        vinserti64x4(zmm23, zmm23, ymm24, 1);

                        vpmovsxwd(zmm25, ymm21);
                        vextracti64x4(ymm26, zmm21, 1);
                        vpmovsxwd(zmm26, ymm26);
                        vpaddd(zmm25, zmm25, zmm26);
                        vpmovsxwd(zmm27, ymm23);
                        vextracti64x4(ymm28, zmm23, 1);
                        vpmovsxwd(zmm28, ymm28);
                        vpaddd(zmm27, zmm27, zmm28);
                        vpaddd(zmm25, zmm25, zmm27);

                        vextracti32x8(ymm26, zmm25, 1);
                        vpaddd(ymm25, ymm25, ymm26);
                        vextracti32x4(xmm26, ymm25, 1);
                        vpaddd(xmm25, xmm25, xmm26);
                        vpshufd(xmm26, xmm25, 0x4E);
                        vpaddd(xmm25, xmm25, xmm26);
                        vpshufd(xmm26, xmm25, 0xB1);
                        vpaddd(xmm25, xmm25, xmm26);

                        vmovd(ptr[r9 + 4], xmm25);
                        vmovdqu32(ptr[r9 + 8], zmm21);
                        vmovdqu32(ptr[r9 + 8 + 64], zmm23);

                        // Restore reg_stride for next N-loop iteration
                        mov(rax, ptr[rsp + 8]);
                        mov(reg_stride.cvt32(), ptr[rax + 56]);
                        shl(reg_stride, 2);
                    }
                    jmp("after_store_m2", T_NEAR);

                    L("after_store_m2");

                    // =================================================================
                    // N-LOOP ITERATION: Continue to next column block
                    // =================================================================
                    add(reg_loop_N, 64);       // Advance column index by 64
                    jmp(loop_N_label, T_NEAR); // Continue N-loop
                }

                // =================================================================
                // FUNCTION EPILOGUE: Restore stack and return
                // =================================================================
                L("end_kernel");
                add(rsp, 16);  // Pop saved N and params from stack
                add(rsp, 256); // Free the zero buffer (256 bytes)
                pop(r15);      // Restore callee-saved registers
                pop(r14);
                pop(r13);
                pop(r12);
                pop(rbp);
                pop(rbx);
                ret(); // Return to caller
            }
        };

    } // namespace gemm_v4
} // namespace llaminar
