#pragma once

#include "SimdTraits.h"
#include <cstddef>

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        { // Changed from cpu to gemm to match GemmKernelTemplate

// ============================================================================
// MACRO-BASED EXPLICIT REGISTER MICRO-KERNEL GENERATOR
// ============================================================================
//
// Generates compile-time specialized micro-kernels with explicit register
// allocation to prevent compiler spilling. Uses preprocessor to generate
// unrolled FMA sequences for each (TILE_M × TILE_N) configuration.
//
// Benefits over generic loop-based approach:
// 1. Explicit register variables (c_0_0, c_0_1, etc.) - no array spilling
// 2. Fully unrolled inner loops - no branch overhead
// 3. Compiler can see entire computation pattern - better scheduling
// 4. Zero runtime overhead - all decisions made at compile time
// ============================================================================

// ----------------------------------------------------------------------------
// Helper macros for register declaration and FMA operations
// ----------------------------------------------------------------------------

// Declare a single accumulator register
#define DECLARE_ACC(row, col) \
    Vec c_##row##_##col = Traits::zero()

// Perform FMA: c_row_col += a_row * b_col
#define FMA_OP(row, col) \
    c_##row##_##col = Traits::fmadd(a_##row, b_##col, c_##row##_##col)

// Store accumulator to output
#define STORE_ACC(row, col) \
    Traits::store(C + (row) * ldc + (col) * kWidth, c_##row##_##col)

// Reduce accumulator to scalar and store (simple reduction)
#define REDUCE_ACC(row, col, output) \
    (output)[(row) * TILE_N + (col)] = Traits::reduce_add(c_##row##_##col)

// Reduce accumulator with alpha/beta scaling
#define REDUCE_ACCUMULATE_ACC(row, col, output, alpha, beta)        \
    do                                                              \
    {                                                               \
        const int idx = (row) * TILE_N + (col);                     \
        float reduced = Traits::reduce_add(c_##row##_##col);        \
        (output)[idx] = (alpha) * reduced + (beta) * (output)[idx]; \
    } while (0)

// ----------------------------------------------------------------------------
// Macro to declare all accumulators for a given tile size
// ----------------------------------------------------------------------------

// 1×1 tile
#define DECLARE_ACCS_1_1() DECLARE_ACC(0, 0)

// 1×2 tile
#define DECLARE_ACCS_1_2() \
    DECLARE_ACC(0, 0);     \
    DECLARE_ACC(0, 1)

// 1×4 tile
#define DECLARE_ACCS_1_4() \
    DECLARE_ACC(0, 0);     \
    DECLARE_ACC(0, 1);     \
    DECLARE_ACC(0, 2);     \
    DECLARE_ACC(0, 3)

// 2×1 tile
#define DECLARE_ACCS_2_1() \
    DECLARE_ACC(0, 0);     \
    DECLARE_ACC(1, 0)

// 2×2 tile
#define DECLARE_ACCS_2_2() \
    DECLARE_ACC(0, 0);     \
    DECLARE_ACC(0, 1);     \
    DECLARE_ACC(1, 0);     \
    DECLARE_ACC(1, 1)

// 2×4 tile
#define DECLARE_ACCS_2_4() \
    DECLARE_ACC(0, 0);     \
    DECLARE_ACC(0, 1);     \
    DECLARE_ACC(0, 2);     \
    DECLARE_ACC(0, 3);     \
    DECLARE_ACC(1, 0);     \
    DECLARE_ACC(1, 1);     \
    DECLARE_ACC(1, 2);     \
    DECLARE_ACC(1, 3)

// 4×1 tile
#define DECLARE_ACCS_4_1() \
    DECLARE_ACC(0, 0);     \
    DECLARE_ACC(1, 0);     \
    DECLARE_ACC(2, 0);     \
    DECLARE_ACC(3, 0)

// 4×2 tile
#define DECLARE_ACCS_4_2() \
    DECLARE_ACC(0, 0);     \
    DECLARE_ACC(0, 1);     \
    DECLARE_ACC(1, 0);     \
    DECLARE_ACC(1, 1);     \
    DECLARE_ACC(2, 0);     \
    DECLARE_ACC(2, 1);     \
    DECLARE_ACC(3, 0);     \
    DECLARE_ACC(3, 1)

// 4×4 tile
#define DECLARE_ACCS_4_4() \
    DECLARE_ACC(0, 0);     \
    DECLARE_ACC(0, 1);     \
    DECLARE_ACC(0, 2);     \
    DECLARE_ACC(0, 3);     \
    DECLARE_ACC(1, 0);     \
    DECLARE_ACC(1, 1);     \
    DECLARE_ACC(1, 2);     \
    DECLARE_ACC(1, 3);     \
    DECLARE_ACC(2, 0);     \
    DECLARE_ACC(2, 1);     \
    DECLARE_ACC(2, 2);     \
    DECLARE_ACC(2, 3);     \
    DECLARE_ACC(3, 0);     \
    DECLARE_ACC(3, 1);     \
    DECLARE_ACC(3, 2);     \
    DECLARE_ACC(3, 3)

// 8×2 tile
#define DECLARE_ACCS_8_2() \
    DECLARE_ACC(0, 0);     \
    DECLARE_ACC(0, 1);     \
    DECLARE_ACC(1, 0);     \
    DECLARE_ACC(1, 1);     \
    DECLARE_ACC(2, 0);     \
    DECLARE_ACC(2, 1);     \
    DECLARE_ACC(3, 0);     \
    DECLARE_ACC(3, 1);     \
    DECLARE_ACC(4, 0);     \
    DECLARE_ACC(4, 1);     \
    DECLARE_ACC(5, 0);     \
    DECLARE_ACC(5, 1);     \
    DECLARE_ACC(6, 0);     \
    DECLARE_ACC(6, 1);     \
    DECLARE_ACC(7, 0);     \
    DECLARE_ACC(7, 1)

// 8×4 tile
#define DECLARE_ACCS_8_4() \
    DECLARE_ACC(0, 0);     \
    DECLARE_ACC(0, 1);     \
    DECLARE_ACC(0, 2);     \
    DECLARE_ACC(0, 3);     \
    DECLARE_ACC(1, 0);     \
    DECLARE_ACC(1, 1);     \
    DECLARE_ACC(1, 2);     \
    DECLARE_ACC(1, 3);     \
    DECLARE_ACC(2, 0);     \
    DECLARE_ACC(2, 1);     \
    DECLARE_ACC(2, 2);     \
    DECLARE_ACC(2, 3);     \
    DECLARE_ACC(3, 0);     \
    DECLARE_ACC(3, 1);     \
    DECLARE_ACC(3, 2);     \
    DECLARE_ACC(3, 3);     \
    DECLARE_ACC(4, 0);     \
    DECLARE_ACC(4, 1);     \
    DECLARE_ACC(4, 2);     \
    DECLARE_ACC(4, 3);     \
    DECLARE_ACC(5, 0);     \
    DECLARE_ACC(5, 1);     \
    DECLARE_ACC(5, 2);     \
    DECLARE_ACC(5, 3);     \
    DECLARE_ACC(6, 0);     \
    DECLARE_ACC(6, 1);     \
    DECLARE_ACC(6, 2);     \
    DECLARE_ACC(6, 3);     \
    DECLARE_ACC(7, 0);     \
    DECLARE_ACC(7, 1);     \
    DECLARE_ACC(7, 2);     \
    DECLARE_ACC(7, 3)

// 8×6 tile (L1Opt configuration)
#define DECLARE_ACCS_8_6() \
    DECLARE_ACC(0, 0);     \
    DECLARE_ACC(0, 1);     \
    DECLARE_ACC(0, 2);     \
    DECLARE_ACC(0, 3);     \
    DECLARE_ACC(0, 4);     \
    DECLARE_ACC(0, 5);     \
    DECLARE_ACC(1, 0);     \
    DECLARE_ACC(1, 1);     \
    DECLARE_ACC(1, 2);     \
    DECLARE_ACC(1, 3);     \
    DECLARE_ACC(1, 4);     \
    DECLARE_ACC(1, 5);     \
    DECLARE_ACC(2, 0);     \
    DECLARE_ACC(2, 1);     \
    DECLARE_ACC(2, 2);     \
    DECLARE_ACC(2, 3);     \
    DECLARE_ACC(2, 4);     \
    DECLARE_ACC(2, 5);     \
    DECLARE_ACC(3, 0);     \
    DECLARE_ACC(3, 1);     \
    DECLARE_ACC(3, 2);     \
    DECLARE_ACC(3, 3);     \
    DECLARE_ACC(3, 4);     \
    DECLARE_ACC(3, 5);     \
    DECLARE_ACC(4, 0);     \
    DECLARE_ACC(4, 1);     \
    DECLARE_ACC(4, 2);     \
    DECLARE_ACC(4, 3);     \
    DECLARE_ACC(4, 4);     \
    DECLARE_ACC(4, 5);     \
    DECLARE_ACC(5, 0);     \
    DECLARE_ACC(5, 1);     \
    DECLARE_ACC(5, 2);     \
    DECLARE_ACC(5, 3);     \
    DECLARE_ACC(5, 4);     \
    DECLARE_ACC(5, 5);     \
    DECLARE_ACC(6, 0);     \
    DECLARE_ACC(6, 1);     \
    DECLARE_ACC(6, 2);     \
    DECLARE_ACC(6, 3);     \
    DECLARE_ACC(6, 4);     \
    DECLARE_ACC(6, 5);     \
    DECLARE_ACC(7, 0);     \
    DECLARE_ACC(7, 1);     \
    DECLARE_ACC(7, 2);     \
    DECLARE_ACC(7, 3);     \
    DECLARE_ACC(7, 4);     \
    DECLARE_ACC(7, 5)

// 16×2 tile
#define DECLARE_ACCS_16_2() \
    DECLARE_ACC(0, 0);      \
    DECLARE_ACC(0, 1);      \
    DECLARE_ACC(1, 0);      \
    DECLARE_ACC(1, 1);      \
    DECLARE_ACC(2, 0);      \
    DECLARE_ACC(2, 1);      \
    DECLARE_ACC(3, 0);      \
    DECLARE_ACC(3, 1);      \
    DECLARE_ACC(4, 0);      \
    DECLARE_ACC(4, 1);      \
    DECLARE_ACC(5, 0);      \
    DECLARE_ACC(5, 1);      \
    DECLARE_ACC(6, 0);      \
    DECLARE_ACC(6, 1);      \
    DECLARE_ACC(7, 0);      \
    DECLARE_ACC(7, 1);      \
    DECLARE_ACC(8, 0);      \
    DECLARE_ACC(8, 1);      \
    DECLARE_ACC(9, 0);      \
    DECLARE_ACC(9, 1);      \
    DECLARE_ACC(10, 0);     \
    DECLARE_ACC(10, 1);     \
    DECLARE_ACC(11, 0);     \
    DECLARE_ACC(11, 1);     \
    DECLARE_ACC(12, 0);     \
    DECLARE_ACC(12, 1);     \
    DECLARE_ACC(13, 0);     \
    DECLARE_ACC(13, 1);     \
    DECLARE_ACC(14, 0);     \
    DECLARE_ACC(14, 1);     \
    DECLARE_ACC(15, 0);     \
    DECLARE_ACC(15, 1)

// 16×4 tile
#define DECLARE_ACCS_16_4() \
    DECLARE_ACC(0, 0);      \
    DECLARE_ACC(0, 1);      \
    DECLARE_ACC(0, 2);      \
    DECLARE_ACC(0, 3);      \
    DECLARE_ACC(1, 0);      \
    DECLARE_ACC(1, 1);      \
    DECLARE_ACC(1, 2);      \
    DECLARE_ACC(1, 3);      \
    DECLARE_ACC(2, 0);      \
    DECLARE_ACC(2, 1);      \
    DECLARE_ACC(2, 2);      \
    DECLARE_ACC(2, 3);      \
    DECLARE_ACC(3, 0);      \
    DECLARE_ACC(3, 1);      \
    DECLARE_ACC(3, 2);      \
    DECLARE_ACC(3, 3);      \
    DECLARE_ACC(4, 0);      \
    DECLARE_ACC(4, 1);      \
    DECLARE_ACC(4, 2);      \
    DECLARE_ACC(4, 3);      \
    DECLARE_ACC(5, 0);      \
    DECLARE_ACC(5, 1);      \
    DECLARE_ACC(5, 2);      \
    DECLARE_ACC(5, 3);      \
    DECLARE_ACC(6, 0);      \
    DECLARE_ACC(6, 1);      \
    DECLARE_ACC(6, 2);      \
    DECLARE_ACC(6, 3);      \
    DECLARE_ACC(7, 0);      \
    DECLARE_ACC(7, 1);      \
    DECLARE_ACC(7, 2);      \
    DECLARE_ACC(7, 3);      \
    DECLARE_ACC(8, 0);      \
    DECLARE_ACC(8, 1);      \
    DECLARE_ACC(8, 2);      \
    DECLARE_ACC(8, 3);      \
    DECLARE_ACC(9, 0);      \
    DECLARE_ACC(9, 1);      \
    DECLARE_ACC(9, 2);      \
    DECLARE_ACC(9, 3);      \
    DECLARE_ACC(10, 0);     \
    DECLARE_ACC(10, 1);     \
    DECLARE_ACC(10, 2);     \
    DECLARE_ACC(10, 3);     \
    DECLARE_ACC(11, 0);     \
    DECLARE_ACC(11, 1);     \
    DECLARE_ACC(11, 2);     \
    DECLARE_ACC(11, 3);     \
    DECLARE_ACC(12, 0);     \
    DECLARE_ACC(12, 1);     \
    DECLARE_ACC(12, 2);     \
    DECLARE_ACC(12, 3);     \
    DECLARE_ACC(13, 0);     \
    DECLARE_ACC(13, 1);     \
    DECLARE_ACC(13, 2);     \
    DECLARE_ACC(13, 3);     \
    DECLARE_ACC(14, 0);     \
    DECLARE_ACC(14, 1);     \
    DECLARE_ACC(14, 2);     \
    DECLARE_ACC(14, 3);     \
    DECLARE_ACC(15, 0);     \
    DECLARE_ACC(15, 1);     \
    DECLARE_ACC(15, 2);     \
    DECLARE_ACC(15, 3)

// ----------------------------------------------------------------------------
// Macro to generate FMA operations for entire tile (one K iteration)
// ----------------------------------------------------------------------------

// 1×1 tile
#define FMA_TILE_1_1() \
    FMA_OP(0, 0)

// 1×2 tile
#define FMA_TILE_1_2() \
    FMA_OP(0, 0);      \
    FMA_OP(0, 1)

// 1×4 tile
#define FMA_TILE_1_4() \
    FMA_OP(0, 0);      \
    FMA_OP(0, 1);      \
    FMA_OP(0, 2);      \
    FMA_OP(0, 3)

// 2×1 tile
#define FMA_TILE_2_1() \
    FMA_OP(0, 0);      \
    FMA_OP(1, 0)

// 2×2 tile
#define FMA_TILE_2_2() \
    FMA_OP(0, 0);      \
    FMA_OP(0, 1);      \
    FMA_OP(1, 0);      \
    FMA_OP(1, 1)

// 2×4 tile
#define FMA_TILE_2_4() \
    FMA_OP(0, 0);      \
    FMA_OP(0, 1);      \
    FMA_OP(0, 2);      \
    FMA_OP(0, 3);      \
    FMA_OP(1, 0);      \
    FMA_OP(1, 1);      \
    FMA_OP(1, 2);      \
    FMA_OP(1, 3)

// 4×1 tile
#define FMA_TILE_4_1() \
    FMA_OP(0, 0);      \
    FMA_OP(1, 0);      \
    FMA_OP(2, 0);      \
    FMA_OP(3, 0)

// 4×2 tile
#define FMA_TILE_4_2() \
    FMA_OP(0, 0);      \
    FMA_OP(0, 1);      \
    FMA_OP(1, 0);      \
    FMA_OP(1, 1);      \
    FMA_OP(2, 0);      \
    FMA_OP(2, 1);      \
    FMA_OP(3, 0);      \
    FMA_OP(3, 1)

// 4×4 tile
#define FMA_TILE_4_4() \
    FMA_OP(0, 0);      \
    FMA_OP(0, 1);      \
    FMA_OP(0, 2);      \
    FMA_OP(0, 3);      \
    FMA_OP(1, 0);      \
    FMA_OP(1, 1);      \
    FMA_OP(1, 2);      \
    FMA_OP(1, 3);      \
    FMA_OP(2, 0);      \
    FMA_OP(2, 1);      \
    FMA_OP(2, 2);      \
    FMA_OP(2, 3);      \
    FMA_OP(3, 0);      \
    FMA_OP(3, 1);      \
    FMA_OP(3, 2);      \
    FMA_OP(3, 3)

// 8×2 tile
#define FMA_TILE_8_2() \
    FMA_OP(0, 0);      \
    FMA_OP(0, 1);      \
    FMA_OP(1, 0);      \
    FMA_OP(1, 1);      \
    FMA_OP(2, 0);      \
    FMA_OP(2, 1);      \
    FMA_OP(3, 0);      \
    FMA_OP(3, 1);      \
    FMA_OP(4, 0);      \
    FMA_OP(4, 1);      \
    FMA_OP(5, 0);      \
    FMA_OP(5, 1);      \
    FMA_OP(6, 0);      \
    FMA_OP(6, 1);      \
    FMA_OP(7, 0);      \
    FMA_OP(7, 1)

// 8×4 tile
#define FMA_TILE_8_4() \
    FMA_OP(0, 0);      \
    FMA_OP(0, 1);      \
    FMA_OP(0, 2);      \
    FMA_OP(0, 3);      \
    FMA_OP(1, 0);      \
    FMA_OP(1, 1);      \
    FMA_OP(1, 2);      \
    FMA_OP(1, 3);      \
    FMA_OP(2, 0);      \
    FMA_OP(2, 1);      \
    FMA_OP(2, 2);      \
    FMA_OP(2, 3);      \
    FMA_OP(3, 0);      \
    FMA_OP(3, 1);      \
    FMA_OP(3, 2);      \
    FMA_OP(3, 3);      \
    FMA_OP(4, 0);      \
    FMA_OP(4, 1);      \
    FMA_OP(4, 2);      \
    FMA_OP(4, 3);      \
    FMA_OP(5, 0);      \
    FMA_OP(5, 1);      \
    FMA_OP(5, 2);      \
    FMA_OP(5, 3);      \
    FMA_OP(6, 0);      \
    FMA_OP(6, 1);      \
    FMA_OP(6, 2);      \
    FMA_OP(6, 3);      \
    FMA_OP(7, 0);      \
    FMA_OP(7, 1);      \
    FMA_OP(7, 2);      \
    FMA_OP(7, 3)

// 8×6 tile
#define FMA_TILE_8_6() \
    FMA_OP(0, 0);      \
    FMA_OP(0, 1);      \
    FMA_OP(0, 2);      \
    FMA_OP(0, 3);      \
    FMA_OP(0, 4);      \
    FMA_OP(0, 5);      \
    FMA_OP(1, 0);      \
    FMA_OP(1, 1);      \
    FMA_OP(1, 2);      \
    FMA_OP(1, 3);      \
    FMA_OP(1, 4);      \
    FMA_OP(1, 5);      \
    FMA_OP(2, 0);      \
    FMA_OP(2, 1);      \
    FMA_OP(2, 2);      \
    FMA_OP(2, 3);      \
    FMA_OP(2, 4);      \
    FMA_OP(2, 5);      \
    FMA_OP(3, 0);      \
    FMA_OP(3, 1);      \
    FMA_OP(3, 2);      \
    FMA_OP(3, 3);      \
    FMA_OP(3, 4);      \
    FMA_OP(3, 5);      \
    FMA_OP(4, 0);      \
    FMA_OP(4, 1);      \
    FMA_OP(4, 2);      \
    FMA_OP(4, 3);      \
    FMA_OP(4, 4);      \
    FMA_OP(4, 5);      \
    FMA_OP(5, 0);      \
    FMA_OP(5, 1);      \
    FMA_OP(5, 2);      \
    FMA_OP(5, 3);      \
    FMA_OP(5, 4);      \
    FMA_OP(5, 5);      \
    FMA_OP(6, 0);      \
    FMA_OP(6, 1);      \
    FMA_OP(6, 2);      \
    FMA_OP(6, 3);      \
    FMA_OP(6, 4);      \
    FMA_OP(6, 5);      \
    FMA_OP(7, 0);      \
    FMA_OP(7, 1);      \
    FMA_OP(7, 2);      \
    FMA_OP(7, 3);      \
    FMA_OP(7, 4);      \
    FMA_OP(7, 5)

// 16×2 tile
#define FMA_TILE_16_2() \
    FMA_OP(0, 0);       \
    FMA_OP(0, 1);       \
    FMA_OP(1, 0);       \
    FMA_OP(1, 1);       \
    FMA_OP(2, 0);       \
    FMA_OP(2, 1);       \
    FMA_OP(3, 0);       \
    FMA_OP(3, 1);       \
    FMA_OP(4, 0);       \
    FMA_OP(4, 1);       \
    FMA_OP(5, 0);       \
    FMA_OP(5, 1);       \
    FMA_OP(6, 0);       \
    FMA_OP(6, 1);       \
    FMA_OP(7, 0);       \
    FMA_OP(7, 1);       \
    FMA_OP(8, 0);       \
    FMA_OP(8, 1);       \
    FMA_OP(9, 0);       \
    FMA_OP(9, 1);       \
    FMA_OP(10, 0);      \
    FMA_OP(10, 1);      \
    FMA_OP(11, 0);      \
    FMA_OP(11, 1);      \
    FMA_OP(12, 0);      \
    FMA_OP(12, 1);      \
    FMA_OP(13, 0);      \
    FMA_OP(13, 1);      \
    FMA_OP(14, 0);      \
    FMA_OP(14, 1);      \
    FMA_OP(15, 0);      \
    FMA_OP(15, 1)

// 16×4 tile
#define FMA_TILE_16_4() \
    FMA_OP(0, 0);       \
    FMA_OP(0, 1);       \
    FMA_OP(0, 2);       \
    FMA_OP(0, 3);       \
    FMA_OP(1, 0);       \
    FMA_OP(1, 1);       \
    FMA_OP(1, 2);       \
    FMA_OP(1, 3);       \
    FMA_OP(2, 0);       \
    FMA_OP(2, 1);       \
    FMA_OP(2, 2);       \
    FMA_OP(2, 3);       \
    FMA_OP(3, 0);       \
    FMA_OP(3, 1);       \
    FMA_OP(3, 2);       \
    FMA_OP(3, 3);       \
    FMA_OP(4, 0);       \
    FMA_OP(4, 1);       \
    FMA_OP(4, 2);       \
    FMA_OP(4, 3);       \
    FMA_OP(5, 0);       \
    FMA_OP(5, 1);       \
    FMA_OP(5, 2);       \
    FMA_OP(5, 3);       \
    FMA_OP(6, 0);       \
    FMA_OP(6, 1);       \
    FMA_OP(6, 2);       \
    FMA_OP(6, 3);       \
    FMA_OP(7, 0);       \
    FMA_OP(7, 1);       \
    FMA_OP(7, 2);       \
    FMA_OP(7, 3);       \
    FMA_OP(8, 0);       \
    FMA_OP(8, 1);       \
    FMA_OP(8, 2);       \
    FMA_OP(8, 3);       \
    FMA_OP(9, 0);       \
    FMA_OP(9, 1);       \
    FMA_OP(9, 2);       \
    FMA_OP(9, 3);       \
    FMA_OP(10, 0);      \
    FMA_OP(10, 1);      \
    FMA_OP(10, 2);      \
    FMA_OP(10, 3);      \
    FMA_OP(11, 0);      \
    FMA_OP(11, 1);      \
    FMA_OP(11, 2);      \
    FMA_OP(11, 3);      \
    FMA_OP(12, 0);      \
    FMA_OP(12, 1);      \
    FMA_OP(12, 2);      \
    FMA_OP(12, 3);      \
    FMA_OP(13, 0);      \
    FMA_OP(13, 1);      \
    FMA_OP(13, 2);      \
    FMA_OP(13, 3);      \
    FMA_OP(14, 0);      \
    FMA_OP(14, 1);      \
    FMA_OP(14, 2);      \
    FMA_OP(14, 3);      \
    FMA_OP(15, 0);      \
    FMA_OP(15, 1);      \
    FMA_OP(15, 2);      \
    FMA_OP(15, 3)

// ----------------------------------------------------------------------------
// Macro to store all accumulators to output
// ----------------------------------------------------------------------------

// 1×1 tile
#define STORE_TILE_1_1() \
    STORE_ACC(0, 0)

// 1×2 tile
#define STORE_TILE_1_2() \
    STORE_ACC(0, 0);     \
    STORE_ACC(0, 1)

// 1×4 tile
#define STORE_TILE_1_4() \
    STORE_ACC(0, 0);     \
    STORE_ACC(0, 1);     \
    STORE_ACC(0, 2);     \
    STORE_ACC(0, 3)

// 2×1 tile
#define STORE_TILE_2_1() \
    STORE_ACC(0, 0);     \
    STORE_ACC(1, 0)

// 2×2 tile
#define STORE_TILE_2_2() \
    STORE_ACC(0, 0);     \
    STORE_ACC(0, 1);     \
    STORE_ACC(1, 0);     \
    STORE_ACC(1, 1)

// 2×4 tile
#define STORE_TILE_2_4() \
    STORE_ACC(0, 0);     \
    STORE_ACC(0, 1);     \
    STORE_ACC(0, 2);     \
    STORE_ACC(0, 3);     \
    STORE_ACC(1, 0);     \
    STORE_ACC(1, 1);     \
    STORE_ACC(1, 2);     \
    STORE_ACC(1, 3)

// 4×1 tile
#define STORE_TILE_4_1() \
    STORE_ACC(0, 0);     \
    STORE_ACC(1, 0);     \
    STORE_ACC(2, 0);     \
    STORE_ACC(3, 0)

// 4×2 tile
#define STORE_TILE_4_2() \
    STORE_ACC(0, 0);     \
    STORE_ACC(0, 1);     \
    STORE_ACC(1, 0);     \
    STORE_ACC(1, 1);     \
    STORE_ACC(2, 0);     \
    STORE_ACC(2, 1);     \
    STORE_ACC(3, 0);     \
    STORE_ACC(3, 1)

// 4×4 tile
#define STORE_TILE_4_4() \
    STORE_ACC(0, 0);     \
    STORE_ACC(0, 1);     \
    STORE_ACC(0, 2);     \
    STORE_ACC(0, 3);     \
    STORE_ACC(1, 0);     \
    STORE_ACC(1, 1);     \
    STORE_ACC(1, 2);     \
    STORE_ACC(1, 3);     \
    STORE_ACC(2, 0);     \
    STORE_ACC(2, 1);     \
    STORE_ACC(2, 2);     \
    STORE_ACC(2, 3);     \
    STORE_ACC(3, 0);     \
    STORE_ACC(3, 1);     \
    STORE_ACC(3, 2);     \
    STORE_ACC(3, 3)

// 8×2 tile
#define STORE_TILE_8_2() \
    STORE_ACC(0, 0);     \
    STORE_ACC(0, 1);     \
    STORE_ACC(1, 0);     \
    STORE_ACC(1, 1);     \
    STORE_ACC(2, 0);     \
    STORE_ACC(2, 1);     \
    STORE_ACC(3, 0);     \
    STORE_ACC(3, 1);     \
    STORE_ACC(4, 0);     \
    STORE_ACC(4, 1);     \
    STORE_ACC(5, 0);     \
    STORE_ACC(5, 1);     \
    STORE_ACC(6, 0);     \
    STORE_ACC(6, 1);     \
    STORE_ACC(7, 0);     \
    STORE_ACC(7, 1)

// 8×4 tile
#define STORE_TILE_8_4() \
    STORE_ACC(0, 0);     \
    STORE_ACC(0, 1);     \
    STORE_ACC(0, 2);     \
    STORE_ACC(0, 3);     \
    STORE_ACC(1, 0);     \
    STORE_ACC(1, 1);     \
    STORE_ACC(1, 2);     \
    STORE_ACC(1, 3);     \
    STORE_ACC(2, 0);     \
    STORE_ACC(2, 1);     \
    STORE_ACC(2, 2);     \
    STORE_ACC(2, 3);     \
    STORE_ACC(3, 0);     \
    STORE_ACC(3, 1);     \
    STORE_ACC(3, 2);     \
    STORE_ACC(3, 3);     \
    STORE_ACC(4, 0);     \
    STORE_ACC(4, 1);     \
    STORE_ACC(4, 2);     \
    STORE_ACC(4, 3);     \
    STORE_ACC(5, 0);     \
    STORE_ACC(5, 1);     \
    STORE_ACC(5, 2);     \
    STORE_ACC(5, 3);     \
    STORE_ACC(6, 0);     \
    STORE_ACC(6, 1);     \
    STORE_ACC(6, 2);     \
    STORE_ACC(6, 3);     \
    STORE_ACC(7, 0);     \
    STORE_ACC(7, 1);     \
    STORE_ACC(7, 2);     \
    STORE_ACC(7, 3)

// 8×6 tile
#define STORE_TILE_8_6() \
    STORE_ACC(0, 0);     \
    STORE_ACC(0, 1);     \
    STORE_ACC(0, 2);     \
    STORE_ACC(0, 3);     \
    STORE_ACC(0, 4);     \
    STORE_ACC(0, 5);     \
    STORE_ACC(1, 0);     \
    STORE_ACC(1, 1);     \
    STORE_ACC(1, 2);     \
    STORE_ACC(1, 3);     \
    STORE_ACC(1, 4);     \
    STORE_ACC(1, 5);     \
    STORE_ACC(2, 0);     \
    STORE_ACC(2, 1);     \
    STORE_ACC(2, 2);     \
    STORE_ACC(2, 3);     \
    STORE_ACC(2, 4);     \
    STORE_ACC(2, 5);     \
    STORE_ACC(3, 0);     \
    STORE_ACC(3, 1);     \
    STORE_ACC(3, 2);     \
    STORE_ACC(3, 3);     \
    STORE_ACC(3, 4);     \
    STORE_ACC(3, 5);     \
    STORE_ACC(4, 0);     \
    STORE_ACC(4, 1);     \
    STORE_ACC(4, 2);     \
    STORE_ACC(4, 3);     \
    STORE_ACC(4, 4);     \
    STORE_ACC(4, 5);     \
    STORE_ACC(5, 0);     \
    STORE_ACC(5, 1);     \
    STORE_ACC(5, 2);     \
    STORE_ACC(5, 3);     \
    STORE_ACC(5, 4);     \
    STORE_ACC(5, 5);     \
    STORE_ACC(6, 0);     \
    STORE_ACC(6, 1);     \
    STORE_ACC(6, 2);     \
    STORE_ACC(6, 3);     \
    STORE_ACC(6, 4);     \
    STORE_ACC(6, 5);     \
    STORE_ACC(7, 0);     \
    STORE_ACC(7, 1);     \
    STORE_ACC(7, 2);     \
    STORE_ACC(7, 3);     \
    STORE_ACC(7, 4);     \
    STORE_ACC(7, 5)

// 16×2 tile
#define STORE_TILE_16_2() \
    STORE_ACC(0, 0);      \
    STORE_ACC(0, 1);      \
    STORE_ACC(1, 0);      \
    STORE_ACC(1, 1);      \
    STORE_ACC(2, 0);      \
    STORE_ACC(2, 1);      \
    STORE_ACC(3, 0);      \
    STORE_ACC(3, 1);      \
    STORE_ACC(4, 0);      \
    STORE_ACC(4, 1);      \
    STORE_ACC(5, 0);      \
    STORE_ACC(5, 1);      \
    STORE_ACC(6, 0);      \
    STORE_ACC(6, 1);      \
    STORE_ACC(7, 0);      \
    STORE_ACC(7, 1);      \
    STORE_ACC(8, 0);      \
    STORE_ACC(8, 1);      \
    STORE_ACC(9, 0);      \
    STORE_ACC(9, 1);      \
    STORE_ACC(10, 0);     \
    STORE_ACC(10, 1);     \
    STORE_ACC(11, 0);     \
    STORE_ACC(11, 1);     \
    STORE_ACC(12, 0);     \
    STORE_ACC(12, 1);     \
    STORE_ACC(13, 0);     \
    STORE_ACC(13, 1);     \
    STORE_ACC(14, 0);     \
    STORE_ACC(14, 1);     \
    STORE_ACC(15, 0);     \
    STORE_ACC(15, 1)

// 16×4 tile
#define STORE_TILE_16_4() \
    STORE_ACC(0, 0);      \
    STORE_ACC(0, 1);      \
    STORE_ACC(0, 2);      \
    STORE_ACC(0, 3);      \
    STORE_ACC(1, 0);      \
    STORE_ACC(1, 1);      \
    STORE_ACC(1, 2);      \
    STORE_ACC(1, 3);      \
    STORE_ACC(2, 0);      \
    STORE_ACC(2, 1);      \
    STORE_ACC(2, 2);      \
    STORE_ACC(2, 3);      \
    STORE_ACC(3, 0);      \
    STORE_ACC(3, 1);      \
    STORE_ACC(3, 2);      \
    STORE_ACC(3, 3);      \
    STORE_ACC(4, 0);      \
    STORE_ACC(4, 1);      \
    STORE_ACC(4, 2);      \
    STORE_ACC(4, 3);      \
    STORE_ACC(5, 0);      \
    STORE_ACC(5, 1);      \
    STORE_ACC(5, 2);      \
    STORE_ACC(5, 3);      \
    STORE_ACC(6, 0);      \
    STORE_ACC(6, 1);      \
    STORE_ACC(6, 2);      \
    STORE_ACC(6, 3);      \
    STORE_ACC(7, 0);      \
    STORE_ACC(7, 1);      \
    STORE_ACC(7, 2);      \
    STORE_ACC(7, 3);      \
    STORE_ACC(8, 0);      \
    STORE_ACC(8, 1);      \
    STORE_ACC(8, 2);      \
    STORE_ACC(8, 3);      \
    STORE_ACC(9, 0);      \
    STORE_ACC(9, 1);      \
    STORE_ACC(9, 2);      \
    STORE_ACC(9, 3);      \
    STORE_ACC(10, 0);     \
    STORE_ACC(10, 1);     \
    STORE_ACC(10, 2);     \
    STORE_ACC(10, 3);     \
    STORE_ACC(11, 0);     \
    STORE_ACC(11, 1);     \
    STORE_ACC(11, 2);     \
    STORE_ACC(11, 3);     \
    STORE_ACC(12, 0);     \
    STORE_ACC(12, 1);     \
    STORE_ACC(12, 2);     \
    STORE_ACC(12, 3);     \
    STORE_ACC(13, 0);     \
    STORE_ACC(13, 1);     \
    STORE_ACC(13, 2);     \
    STORE_ACC(13, 3);     \
    STORE_ACC(14, 0);     \
    STORE_ACC(14, 1);     \
    STORE_ACC(14, 2);     \
    STORE_ACC(14, 3);     \
    STORE_ACC(15, 0);     \
    STORE_ACC(15, 1);     \
    STORE_ACC(15, 2);     \
    STORE_ACC(15, 3)

// ----------------------------------------------------------------------------
// Macros for reduction (horizontal sum to scalars)
// ----------------------------------------------------------------------------

// 1×1 tile
#define REDUCE_TILE_1_1(out) \
    REDUCE_ACC(0, 0, out)

#define REDUCE_ACCUMULATE_TILE_1_1(out, alpha, beta) \
    REDUCE_ACCUMULATE_ACC(0, 0, out, alpha, beta)

// 1×2 tile
#define REDUCE_TILE_1_2(out) \
    REDUCE_ACC(0, 0, out);   \
    REDUCE_ACC(0, 1, out)

#define REDUCE_ACCUMULATE_TILE_1_2(out, alpha, beta) \
    REDUCE_ACCUMULATE_ACC(0, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(0, 1, out, alpha, beta)

// 1×4 tile
#define REDUCE_TILE_1_4(out) \
    REDUCE_ACC(0, 0, out);   \
    REDUCE_ACC(0, 1, out);   \
    REDUCE_ACC(0, 2, out);   \
    REDUCE_ACC(0, 3, out)

#define REDUCE_ACCUMULATE_TILE_1_4(out, alpha, beta) \
    REDUCE_ACCUMULATE_ACC(0, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(0, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(0, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(0, 3, out, alpha, beta)

// 2×1 tile
#define REDUCE_TILE_2_1(out) \
    REDUCE_ACC(0, 0, out);   \
    REDUCE_ACC(1, 0, out)

#define REDUCE_ACCUMULATE_TILE_2_1(out, alpha, beta) \
    REDUCE_ACCUMULATE_ACC(0, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 0, out, alpha, beta)

// 2×2 tile
#define REDUCE_TILE_2_2(out) \
    REDUCE_ACC(0, 0, out);   \
    REDUCE_ACC(0, 1, out);   \
    REDUCE_ACC(1, 0, out);   \
    REDUCE_ACC(1, 1, out)

#define REDUCE_ACCUMULATE_TILE_2_2(out, alpha, beta) \
    REDUCE_ACCUMULATE_ACC(0, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(0, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 1, out, alpha, beta)

// 2×4 tile
#define REDUCE_TILE_2_4(out) \
    REDUCE_ACC(0, 0, out);   \
    REDUCE_ACC(0, 1, out);   \
    REDUCE_ACC(0, 2, out);   \
    REDUCE_ACC(0, 3, out);   \
    REDUCE_ACC(1, 0, out);   \
    REDUCE_ACC(1, 1, out);   \
    REDUCE_ACC(1, 2, out);   \
    REDUCE_ACC(1, 3, out)

#define REDUCE_ACCUMULATE_TILE_2_4(out, alpha, beta) \
    REDUCE_ACCUMULATE_ACC(0, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(0, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(0, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(0, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 3, out, alpha, beta)

// 4×1 tile
#define REDUCE_TILE_4_1(out) \
    REDUCE_ACC(0, 0, out);   \
    REDUCE_ACC(1, 0, out);   \
    REDUCE_ACC(2, 0, out);   \
    REDUCE_ACC(3, 0, out)

#define REDUCE_ACCUMULATE_TILE_4_1(out, alpha, beta) \
    REDUCE_ACCUMULATE_ACC(0, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(2, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(3, 0, out, alpha, beta)

// 4×2 tile
#define REDUCE_TILE_4_2(out) \
    REDUCE_ACC(0, 0, out);   \
    REDUCE_ACC(0, 1, out);   \
    REDUCE_ACC(1, 0, out);   \
    REDUCE_ACC(1, 1, out);   \
    REDUCE_ACC(2, 0, out);   \
    REDUCE_ACC(2, 1, out);   \
    REDUCE_ACC(3, 0, out);   \
    REDUCE_ACC(3, 1, out)

#define REDUCE_ACCUMULATE_TILE_4_2(out, alpha, beta) \
    REDUCE_ACCUMULATE_ACC(0, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(0, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(2, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(2, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(3, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(3, 1, out, alpha, beta)

// 4×4 tile
#define REDUCE_TILE_4_4(out) \
    REDUCE_ACC(0, 0, out);   \
    REDUCE_ACC(0, 1, out);   \
    REDUCE_ACC(0, 2, out);   \
    REDUCE_ACC(0, 3, out);   \
    REDUCE_ACC(1, 0, out);   \
    REDUCE_ACC(1, 1, out);   \
    REDUCE_ACC(1, 2, out);   \
    REDUCE_ACC(1, 3, out);   \
    REDUCE_ACC(2, 0, out);   \
    REDUCE_ACC(2, 1, out);   \
    REDUCE_ACC(2, 2, out);   \
    REDUCE_ACC(2, 3, out);   \
    REDUCE_ACC(3, 0, out);   \
    REDUCE_ACC(3, 1, out);   \
    REDUCE_ACC(3, 2, out);   \
    REDUCE_ACC(3, 3, out)

#define REDUCE_ACCUMULATE_TILE_4_4(out, alpha, beta) \
    REDUCE_ACCUMULATE_ACC(0, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(0, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(0, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(0, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(2, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(2, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(2, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(2, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(3, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(3, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(3, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(3, 3, out, alpha, beta)

// 8×2 tile
#define REDUCE_TILE_8_2(out) \
    REDUCE_ACC(0, 0, out);   \
    REDUCE_ACC(0, 1, out);   \
    REDUCE_ACC(1, 0, out);   \
    REDUCE_ACC(1, 1, out);   \
    REDUCE_ACC(2, 0, out);   \
    REDUCE_ACC(2, 1, out);   \
    REDUCE_ACC(3, 0, out);   \
    REDUCE_ACC(3, 1, out);   \
    REDUCE_ACC(4, 0, out);   \
    REDUCE_ACC(4, 1, out);   \
    REDUCE_ACC(5, 0, out);   \
    REDUCE_ACC(5, 1, out);   \
    REDUCE_ACC(6, 0, out);   \
    REDUCE_ACC(6, 1, out);   \
    REDUCE_ACC(7, 0, out);   \
    REDUCE_ACC(7, 1, out)

#define REDUCE_ACCUMULATE_TILE_8_2(out, alpha, beta) \
    REDUCE_ACCUMULATE_ACC(0, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(0, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(2, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(2, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(3, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(3, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(4, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(4, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(5, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(5, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(6, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(6, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(7, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(7, 1, out, alpha, beta)

// 8×4 tile
#define REDUCE_TILE_8_4(out) \
    REDUCE_ACC(0, 0, out);   \
    REDUCE_ACC(0, 1, out);   \
    REDUCE_ACC(0, 2, out);   \
    REDUCE_ACC(0, 3, out);   \
    REDUCE_ACC(1, 0, out);   \
    REDUCE_ACC(1, 1, out);   \
    REDUCE_ACC(1, 2, out);   \
    REDUCE_ACC(1, 3, out);   \
    REDUCE_ACC(2, 0, out);   \
    REDUCE_ACC(2, 1, out);   \
    REDUCE_ACC(2, 2, out);   \
    REDUCE_ACC(2, 3, out);   \
    REDUCE_ACC(3, 0, out);   \
    REDUCE_ACC(3, 1, out);   \
    REDUCE_ACC(3, 2, out);   \
    REDUCE_ACC(3, 3, out);   \
    REDUCE_ACC(4, 0, out);   \
    REDUCE_ACC(4, 1, out);   \
    REDUCE_ACC(4, 2, out);   \
    REDUCE_ACC(4, 3, out);   \
    REDUCE_ACC(5, 0, out);   \
    REDUCE_ACC(5, 1, out);   \
    REDUCE_ACC(5, 2, out);   \
    REDUCE_ACC(5, 3, out);   \
    REDUCE_ACC(6, 0, out);   \
    REDUCE_ACC(6, 1, out);   \
    REDUCE_ACC(6, 2, out);   \
    REDUCE_ACC(6, 3, out);   \
    REDUCE_ACC(7, 0, out);   \
    REDUCE_ACC(7, 1, out);   \
    REDUCE_ACC(7, 2, out);   \
    REDUCE_ACC(7, 3, out)

#define REDUCE_ACCUMULATE_TILE_8_4(out, alpha, beta) \
    REDUCE_ACCUMULATE_ACC(0, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(0, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(0, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(0, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(2, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(2, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(2, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(2, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(3, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(3, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(3, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(3, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(4, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(4, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(4, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(4, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(5, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(5, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(5, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(5, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(6, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(6, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(6, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(6, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(7, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(7, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(7, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(7, 3, out, alpha, beta)

// 8×6 tile (L1Opt configuration)
#define REDUCE_TILE_8_6(out) \
    REDUCE_ACC(0, 0, out);   \
    REDUCE_ACC(0, 1, out);   \
    REDUCE_ACC(0, 2, out);   \
    REDUCE_ACC(0, 3, out);   \
    REDUCE_ACC(0, 4, out);   \
    REDUCE_ACC(0, 5, out);   \
    REDUCE_ACC(1, 0, out);   \
    REDUCE_ACC(1, 1, out);   \
    REDUCE_ACC(1, 2, out);   \
    REDUCE_ACC(1, 3, out);   \
    REDUCE_ACC(1, 4, out);   \
    REDUCE_ACC(1, 5, out);   \
    REDUCE_ACC(2, 0, out);   \
    REDUCE_ACC(2, 1, out);   \
    REDUCE_ACC(2, 2, out);   \
    REDUCE_ACC(2, 3, out);   \
    REDUCE_ACC(2, 4, out);   \
    REDUCE_ACC(2, 5, out);   \
    REDUCE_ACC(3, 0, out);   \
    REDUCE_ACC(3, 1, out);   \
    REDUCE_ACC(3, 2, out);   \
    REDUCE_ACC(3, 3, out);   \
    REDUCE_ACC(3, 4, out);   \
    REDUCE_ACC(3, 5, out);   \
    REDUCE_ACC(4, 0, out);   \
    REDUCE_ACC(4, 1, out);   \
    REDUCE_ACC(4, 2, out);   \
    REDUCE_ACC(4, 3, out);   \
    REDUCE_ACC(4, 4, out);   \
    REDUCE_ACC(4, 5, out);   \
    REDUCE_ACC(5, 0, out);   \
    REDUCE_ACC(5, 1, out);   \
    REDUCE_ACC(5, 2, out);   \
    REDUCE_ACC(5, 3, out);   \
    REDUCE_ACC(5, 4, out);   \
    REDUCE_ACC(5, 5, out);   \
    REDUCE_ACC(6, 0, out);   \
    REDUCE_ACC(6, 1, out);   \
    REDUCE_ACC(6, 2, out);   \
    REDUCE_ACC(6, 3, out);   \
    REDUCE_ACC(6, 4, out);   \
    REDUCE_ACC(6, 5, out);   \
    REDUCE_ACC(7, 0, out);   \
    REDUCE_ACC(7, 1, out);   \
    REDUCE_ACC(7, 2, out);   \
    REDUCE_ACC(7, 3, out);   \
    REDUCE_ACC(7, 4, out);   \
    REDUCE_ACC(7, 5, out)

#define REDUCE_ACCUMULATE_TILE_8_6(out, alpha, beta) \
    REDUCE_ACCUMULATE_ACC(0, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(0, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(0, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(0, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(0, 4, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(0, 5, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 4, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(1, 5, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(2, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(2, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(2, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(2, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(2, 4, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(2, 5, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(3, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(3, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(3, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(3, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(3, 4, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(3, 5, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(4, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(4, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(4, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(4, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(4, 4, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(4, 5, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(5, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(5, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(5, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(5, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(5, 4, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(5, 5, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(6, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(6, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(6, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(6, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(6, 4, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(6, 5, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(7, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(7, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(7, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(7, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(7, 4, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(7, 5, out, alpha, beta)

// 16×2 tile
#define REDUCE_TILE_16_2(out) \
    REDUCE_ACC(0, 0, out);    \
    REDUCE_ACC(0, 1, out);    \
    REDUCE_ACC(1, 0, out);    \
    REDUCE_ACC(1, 1, out);    \
    REDUCE_ACC(2, 0, out);    \
    REDUCE_ACC(2, 1, out);    \
    REDUCE_ACC(3, 0, out);    \
    REDUCE_ACC(3, 1, out);    \
    REDUCE_ACC(4, 0, out);    \
    REDUCE_ACC(4, 1, out);    \
    REDUCE_ACC(5, 0, out);    \
    REDUCE_ACC(5, 1, out);    \
    REDUCE_ACC(6, 0, out);    \
    REDUCE_ACC(6, 1, out);    \
    REDUCE_ACC(7, 0, out);    \
    REDUCE_ACC(7, 1, out);    \
    REDUCE_ACC(8, 0, out);    \
    REDUCE_ACC(8, 1, out);    \
    REDUCE_ACC(9, 0, out);    \
    REDUCE_ACC(9, 1, out);    \
    REDUCE_ACC(10, 0, out);   \
    REDUCE_ACC(10, 1, out);   \
    REDUCE_ACC(11, 0, out);   \
    REDUCE_ACC(11, 1, out);   \
    REDUCE_ACC(12, 0, out);   \
    REDUCE_ACC(12, 1, out);   \
    REDUCE_ACC(13, 0, out);   \
    REDUCE_ACC(13, 1, out);   \
    REDUCE_ACC(14, 0, out);   \
    REDUCE_ACC(14, 1, out);   \
    REDUCE_ACC(15, 0, out);   \
    REDUCE_ACC(15, 1, out)

#define REDUCE_ACCUMULATE_TILE_16_2(out, alpha, beta) \
    REDUCE_ACCUMULATE_ACC(0, 0, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(0, 1, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(1, 0, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(1, 1, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(2, 0, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(2, 1, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(3, 0, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(3, 1, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(4, 0, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(4, 1, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(5, 0, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(5, 1, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(6, 0, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(6, 1, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(7, 0, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(7, 1, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(8, 0, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(8, 1, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(9, 0, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(9, 1, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(10, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(10, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(11, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(11, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(12, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(12, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(13, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(13, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(14, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(14, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(15, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(15, 1, out, alpha, beta)

// 16×4 tile
#define REDUCE_TILE_16_4(out) \
    REDUCE_ACC(0, 0, out);    \
    REDUCE_ACC(0, 1, out);    \
    REDUCE_ACC(0, 2, out);    \
    REDUCE_ACC(0, 3, out);    \
    REDUCE_ACC(1, 0, out);    \
    REDUCE_ACC(1, 1, out);    \
    REDUCE_ACC(1, 2, out);    \
    REDUCE_ACC(1, 3, out);    \
    REDUCE_ACC(2, 0, out);    \
    REDUCE_ACC(2, 1, out);    \
    REDUCE_ACC(2, 2, out);    \
    REDUCE_ACC(2, 3, out);    \
    REDUCE_ACC(3, 0, out);    \
    REDUCE_ACC(3, 1, out);    \
    REDUCE_ACC(3, 2, out);    \
    REDUCE_ACC(3, 3, out);    \
    REDUCE_ACC(4, 0, out);    \
    REDUCE_ACC(4, 1, out);    \
    REDUCE_ACC(4, 2, out);    \
    REDUCE_ACC(4, 3, out);    \
    REDUCE_ACC(5, 0, out);    \
    REDUCE_ACC(5, 1, out);    \
    REDUCE_ACC(5, 2, out);    \
    REDUCE_ACC(5, 3, out);    \
    REDUCE_ACC(6, 0, out);    \
    REDUCE_ACC(6, 1, out);    \
    REDUCE_ACC(6, 2, out);    \
    REDUCE_ACC(6, 3, out);    \
    REDUCE_ACC(7, 0, out);    \
    REDUCE_ACC(7, 1, out);    \
    REDUCE_ACC(7, 2, out);    \
    REDUCE_ACC(7, 3, out);    \
    REDUCE_ACC(8, 0, out);    \
    REDUCE_ACC(8, 1, out);    \
    REDUCE_ACC(8, 2, out);    \
    REDUCE_ACC(8, 3, out);    \
    REDUCE_ACC(9, 0, out);    \
    REDUCE_ACC(9, 1, out);    \
    REDUCE_ACC(9, 2, out);    \
    REDUCE_ACC(9, 3, out);    \
    REDUCE_ACC(10, 0, out);   \
    REDUCE_ACC(10, 1, out);   \
    REDUCE_ACC(10, 2, out);   \
    REDUCE_ACC(10, 3, out);   \
    REDUCE_ACC(11, 0, out);   \
    REDUCE_ACC(11, 1, out);   \
    REDUCE_ACC(11, 2, out);   \
    REDUCE_ACC(11, 3, out);   \
    REDUCE_ACC(12, 0, out);   \
    REDUCE_ACC(12, 1, out);   \
    REDUCE_ACC(12, 2, out);   \
    REDUCE_ACC(12, 3, out);   \
    REDUCE_ACC(13, 0, out);   \
    REDUCE_ACC(13, 1, out);   \
    REDUCE_ACC(13, 2, out);   \
    REDUCE_ACC(13, 3, out);   \
    REDUCE_ACC(14, 0, out);   \
    REDUCE_ACC(14, 1, out);   \
    REDUCE_ACC(14, 2, out);   \
    REDUCE_ACC(14, 3, out);   \
    REDUCE_ACC(15, 0, out);   \
    REDUCE_ACC(15, 1, out);   \
    REDUCE_ACC(15, 2, out);   \
    REDUCE_ACC(15, 3, out)

#define REDUCE_ACCUMULATE_TILE_16_4(out, alpha, beta) \
    REDUCE_ACCUMULATE_ACC(0, 0, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(0, 1, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(0, 2, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(0, 3, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(1, 0, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(1, 1, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(1, 2, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(1, 3, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(2, 0, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(2, 1, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(2, 2, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(2, 3, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(3, 0, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(3, 1, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(3, 2, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(3, 3, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(4, 0, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(4, 1, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(4, 2, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(4, 3, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(5, 0, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(5, 1, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(5, 2, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(5, 3, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(6, 0, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(6, 1, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(6, 2, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(6, 3, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(7, 0, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(7, 1, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(7, 2, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(7, 3, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(8, 0, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(8, 1, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(8, 2, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(8, 3, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(9, 0, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(9, 1, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(9, 2, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(9, 3, out, alpha, beta);    \
    REDUCE_ACCUMULATE_ACC(10, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(10, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(10, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(10, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(11, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(11, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(11, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(11, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(12, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(12, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(12, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(12, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(13, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(13, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(13, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(13, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(14, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(14, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(14, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(14, 3, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(15, 0, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(15, 1, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(15, 2, out, alpha, beta);   \
    REDUCE_ACCUMULATE_ACC(15, 3, out, alpha, beta)

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
