/**
 * @file ROCmQuantisedGemmKernel_CK.hip
 * @brief ComposableKernel INT8 GEMM implementation for AMD ROCm GPUs
 *
 * This file implements INT8×INT8→INT32 GEMM operations for AMD GPUs using the
 * ComposableKernel (CK) library. It provides the CK-specific kernel dispatch
 * for quantized inference on AMD Instinct GPUs (gfx906).
 *
 * ============================================================================
 * QUICK START FOR NEW DEVELOPERS
 * ============================================================================
 *
 * What this file does:
 *   1. Configures CK tile-based INT8×INT8→INT32 GEMM kernels
 *   2. Provides 4-way dispatch (32×32, 64×64, 128×128, 128×128 MNPadding)
 *   3. Manages CK kernel context (warmup, creation, destruction)
 *
 * Key functions:
 *   - rocmQuantGemm_executeNoScale()          : CK INT8×INT8→INT32 GEMM (core dispatch)
 *   - rocmQuantGemm_createKernelContext()      : Pre-initialize CK kernels
 *   - rocmQuantGemm_warmupKernels()            : Force CK kernel compilation
 *   - rocmQuantGemm_isSupportedDimensions()    : Check CK dimension constraints
 *
 * Common utilities (quantization, scaling, memory) are in:
 *   - ROCmQuantisedGemmKernel.hip (shared by all backends)
 *
 * ============================================================================
 * ARCHITECTURE SUPPORT
 * ============================================================================
 *
 * | GPU       | Architecture | CK Template              | Status           |
 * |-----------|--------------|--------------------------|------------------|
 * | MI50/MI60 | gfx906       | DeviceGemmMultipleD_Dl   | ✅ Supported     |
 * | MI100     | gfx908       | DeviceGemmXdl (future)   | 🔄 Not yet       |
 * | MI200     | gfx90a       | DeviceGemmXdl (future)   | 🔄 Not yet       |
 * | MI300     | gfx940a      | DeviceGemmXdl (future)   | 🔄 Not yet       |
 *
 * The "DL" in DeviceGemmMultipleD_Dl stands for "Deep Learning" and uses
 * 4-way INT8 dot product instructions (v_dot4_i32_i8). The "XDL" variant
 * uses MFMA (Matrix Fused Multiply-Add) instructions on newer GPUs.
 *
 * ============================================================================
 * MINIMUM DIMENSION REQUIREMENTS (CRITICAL!)
 * ============================================================================
 *
 * CK's DeviceGemmMultipleD_Dl has tile-based requirements. We use 4-WAY DISPATCH
 * to optimize for different M values:
 *
 *   | M Range           | Kernel           | Tile Size | Why?                    |
 *   |-------------------|------------------|-----------|-------------------------|
 *   | M ≤ 32            | 32×32            | 32×32×32  | Minimal waste for decode|
 *   | 32 < M < 128      | 64×64            | 64×64×32  | Good balance for batches|
 *   | M ≥ 128, M%128==0 | 128×128          | 128×128×64| Peak throughput         |
 *   | M ≥ 128, M%128!=0 | 128×128 MNPadding| 128×128×64| Irregular prefill (e.g. M=596) |
 *
 * The 32×32 and 64×64 kernels use GemmSpecialization::MNPadding for irregular M.
 * The 128×128 kernel uses GemmSpecialization::Default for aligned M (peak perf).
 * The 128×128 MNPadding kernel handles M >= 128 with irregular dimensions.
 * N and K requirements: N ≥ 64, K ≥ 64 (satisfied by all LLM dimensions).
 *
 * For decode (M=1), the 32×32 kernel handles it directly - no explicit padding!
 *
 * ============================================================================
 * MEMORY LAYOUT CONVENTION (CRITICAL!)
 * ============================================================================
 *
 * This implementation uses **Row-Major for A and C, Column-Major for B**.
 * This follows CK's `mk_nk_mn` naming convention (A=Row, B=Col, C=Row):
 *
 * Matrix A (activations): [M × K] Row-Major
 *   - Element A[m,k] is at memory offset: m * K + k
 *   - Stride = K (number of columns)
 *   - Example: A[2,3] with K=5 is at offset 2*5 + 3 = 13
 *
 * Matrix B (weights): [K × N] Column-Major (stored as [N×K] Row-Major)
 *   - Original model weights [N×K] are stored Row-Major: W[n,k] = data[n*K + k]
 *   - CK interprets this as Column-Major [K×N]: B[k,n] = data[n*K + k]
 *   - No transpose needed - reinterpret layout for CK's mk_nk_mn convention
 *
 * Matrix C (output): [M × N] Row-Major
 *   - Element C[m,n] is at memory offset: m * N + n
 *   - Stride = N (number of columns)
 *
 * ============================================================================
 * CK INSTANCE FILE REFERENCE
 * ============================================================================
 *
 * The tile parameters MUST match the layout configuration. CK's instance
 * files are named by layout:
 *
 *   | Suffix    | Layouts          | Description                    |
 *   |-----------|------------------|--------------------------------|
 *   | mk_kn_mn  | Row, Row, Row    | Small tiles (64x64)            |
 *   | km_kn_mn  | Col, Row, Row    | Would require A transposed     |
 *   | mk_nk_mn  | Row, Col, Row    | ← WE USE THIS (128x128 tiles!) |
 *
 * CRITICAL: Using tile parameters from the wrong layout configuration will
 * produce incorrect numerical results WITHOUT any error messages or exceptions.
 *
 * Our tile parameters are copied from:
 *   https://github.com/ROCm/composable_kernel/blob/develop/library/src/
 *       tensor_operation_instance/gpu/gemm/device_gemm_dl_i8_i8_i8_mk_nk_mn_instance.cpp
 *
 * WHY mk_nk_mn IS FASTER:
 *   - Larger tiles: 128x128 vs 64x64 (4x more work per block)
 *   - More threads: BlockSize=256 vs 64 (better GPU utilization)
 *   - Better B cache locality: Column-Major B has contiguous K access
 *
 * ============================================================================
 * DEBUGGING HISTORY
 * ============================================================================
 *
 * Issue 1: Wrong tile parameters (January 2026)
 *   - Symptom: ~137% relative error with opposite sign
 *   - Cause: Used tile params from km_kn_mn (Col,Row,Row) with Row,Row,Row layout
 *   - Fix: Copied exact tile parameters from mk_kn_mn_instance.cpp
 *
 * Issue 2: D-tensor type mismatch bug (January 2026)
 *   - Symptom: ~10% zeros at deterministic positions, cos=0.984
 *   - Cause: CK's gridwise_gemm_dl_multiple_d.hpp wrote float to int32 buffer
 *   - Fix: Patched CK to use separate e_thread_buf with FloatC type
 *   - Location: external/composable_kernel/include/ck/tensor_operation/gpu/grid/
 *               gridwise_gemm_dl_multiple_d.hpp (lines 421, 639, 670)
 *
 * @see https://github.com/ROCm/composable_kernel - AMD ComposableKernel library
 * @see https://github.com/ROCm/composable_kernel/blob/develop/example/14_gemm_quantization/
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <chrono>
#include <iostream>
#include <streambuf>
#include <array>
#include <memory>

#include "../../../backends/rocm/HipDeviceGuard.h"

// ComposableKernel headers for INT8 GEMM

namespace {
struct NullStreamBuf final : public std::streambuf
{
    int overflow(int c) override { return c; }
};

struct ScopedCoutSilence final
{
    std::streambuf* old_buf_;

    ScopedCoutSilence()
    {
        static NullStreamBuf null_buf;
        old_buf_ = std::cout.rdbuf(&null_buf);
    }

    ~ScopedCoutSilence() { std::cout.rdbuf(old_buf_); }
};
} // namespace
// Using DeviceGemmMultipleD_Dl which is proven to work for INT8 quantization
// Based on: composable_kernel/example/14_gemm_quantization/gemm_dl_quantization_int8.cpp
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_dl.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/stream_config.hpp"

// ============================================================================
// ComposableKernel INT8×INT8→INT32 GEMM Type Definition (gfx906 - DeviceGemmMultipleD_Dl)
// ============================================================================
//
// ## Why DeviceGemmMultipleD_Dl?
//
// DeviceGemmMultipleD_Dl is preferred over the simpler DeviceGemmDl because:
// 1. It's the template used in CK's official INT8 quantization examples
// 2. It supports fused element-wise operations (for potential future scaling fusion)
// 3. It has proven, tested configurations for gfx906 (MI50/MI60)
//
// ## Memory Layout (Row-Major Convention)
//
// This configuration uses Row-Major layout for ALL matrices:
//
//   Matrix A (activations): [M × K] RowMajor
//     - Element A[m,k] located at: data[m * StrideA + k]
//     - StrideA = K (contiguous along K dimension)
//
//   Matrix B (weights): [K × N] RowMajor
//     - Element B[k,n] located at: data[k * StrideB + n]
//     - StrideB = N (contiguous along N dimension)
//     - NOTE: This is the TRANSPOSE of original model weights [N×K]
//
//   Matrix E (output): [M × N] RowMajor
//     - Element E[m,n] located at: data[m * StrideE + n]
//     - StrideE = N (contiguous along N dimension)
//
// ## Configuration Source
//
// Tile parameters copied from CK's official instance file for Row,Col,Row layout:
//   https://github.com/ROCm/composable_kernel/blob/develop/library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_i8_i8_i8_mk_nk_mn_instance.cpp
//
// The "mk_nk_mn" suffix indicates: A[M,K]=Row, B[K,N]=Col (same as [N,K] Row), C[M,N]=Row
//
// ============================================================================

namespace {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using I8  = int8_t;
using I32 = int32_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

// ============================================================================
// CK MINIMUM DIMENSION REQUIREMENTS
// ============================================================================
//
// DeviceGemmMultipleD_Dl uses tile-based computation with fixed block sizes.
// Dimensions below these minimums will fail IsSupportedArgument().
//
// We use THREE kernel configurations (3-way dispatch):
//
// SMALL KERNEL (32×32 tiles) - for decode (M ≤ 32):
//   - MPerBlock = 32, NPerBlock = 32, K0×K1 = 8×4 = 32
//   - Uses GemmMNPadding to handle M=1 natively (no explicit padding needed!)
//   - Optimal for decode where M is typically 1
//
// MEDIUM KERNEL (64×64 tiles) - for small batches (32 < M < 128):
//   - MPerBlock = 64, NPerBlock = 64, K0×K1 = 8×4 = 32
//   - Uses GemmMNPadding for irregular M dimensions
//   - Good balance between occupancy and efficiency
//
// LARGE KERNEL (128×128 tiles) - for prefill (M ≥ 128):
//   - MPerBlock = 128, NPerBlock = 128, K0×K1 = 16×4 = 64
//   - Uses GemmDefault when M % 128 == 0 (peak throughput)
//   - Uses GemmMNPadding when M % 128 != 0 (handles irregular M like M=596)
//
// DISPATCH STRATEGY (4-WAY):
//   M ≤ 32:           Use 32×32 kernel (GemmMNPadding handles irregular M)
//   32 < M < 128:     Use 64×64 kernel (GemmMNPadding handles irregular M)  
//   M ≥ 128, M%128==0: Use 128×128 kernel (GemmDefault, peak throughput)
//   M ≥ 128, M%128!=0: Use 128×128 kernel (GemmMNPadding, handles irregular M)
//
// Dimension requirements for each kernel (used by ck_dimensions_supported())
// Note: The 32×32 and 64×64 kernels use GemmMNPadding, so they can handle
// M values below their tile size. These are the MINIMUM dimensions after
// the C++ caller's explicit padding (M < 8 gets padded to 8).

constexpr int CK_LARGE_MIN_M = 128;  // 128×128 kernel requires M ≥ 128
constexpr int CK_LARGE_MIN_N = 128;  // 128×128 kernel requires N ≥ 128
constexpr int CK_LARGE_MIN_K = 64;   // 128×128 kernel requires K ≥ 64

// Minimum dimensions supported after C++ caller padding (M < 8 → 8)
// The 32×32 kernel with GemmMNPadding handles M=8 to M=32 natively
constexpr int CK_SMALL_MIN_M = 8;    // After C++ padding, minimum M is 8
constexpr int CK_SMALL_MIN_N = 64;   // Minimum N for 32×32/64×64 kernels  
constexpr int CK_SMALL_MIN_K = 64;   // Minimum K for all kernels

// Threshold for 3-way dispatch (handled in rocmQuantGemm_executeNoScale)
constexpr int CK_SMALL_KERNEL_THRESHOLD = 128;  // M < 128 uses 32×32 or 64×64 kernel

// Helper function to check if dimensions meet large kernel requirements
inline bool ck_large_dimensions_supported(int M, int N, int K)
{
    return M >= CK_LARGE_MIN_M && N >= CK_LARGE_MIN_N && K >= CK_LARGE_MIN_K;
}

// Helper function to check if dimensions meet small kernel requirements  
inline bool ck_small_dimensions_supported(int M, int N, int K)
{
    return M >= CK_SMALL_MIN_M && N >= CK_SMALL_MIN_N && K >= CK_SMALL_MIN_K;
}

// Legacy: check if ANY CK kernel can handle these dimensions (after padding)
inline bool ck_dimensions_supported(int M, int N, int K)
{
    // After C++ caller padding (M < 8 → 8), minimum supported dimensions are:
    // M >= 8, N >= 64, K >= 64
    // The 32×32 kernel with GemmMNPadding handles these natively.
    return M >= CK_SMALL_MIN_M && N >= CK_SMALL_MIN_N && K >= CK_SMALL_MIN_K;
}

// ============================================================================
// TYPE DEFINITIONS FOR CK INT8 GEMM
// ============================================================================
//
// This section defines all the CK template types used for INT8 GEMM.
// Understanding these types is essential for debugging and modifying the kernel.
//
// QUICK REFERENCE:
//   ADataType = int8_t    : Input activations (quantized)
//   BDataType = int8_t    : Weights (quantized)
//   AccDataType = int32_t : Accumulator during GEMM (prevents overflow)
//   EDataType = float     : Final output after scaling
//
// ============================================================================

using ADataType        = I8;
using BDataType        = I8;
using AccDataType      = I32;

// ============================================================================
// LAYOUT DEFINITIONS
// ============================================================================
//
// CK uses template types to specify memory layouts:
//   - RowMajor: Elements in a row are contiguous in memory
//   - ColumnMajor: Elements in a column are contiguous in memory
//
// For Row-Major layout:
//   - Stride = number of columns
//   - Element [i,j] at offset: i * stride + j
//
// For Column-Major layout:
//   - Stride = number of rows
//   - Element [i,j] at offset: i + j * stride
//
// Our configuration uses mk_nk_mn layout for optimal performance:
//   A[M×K] row-major: A[m,k] at m*K + k, Stride=K
//   B[N×K] col-major (as [K×N]): B[k,n] at k + n*K, Stride=K
//   E[M×N] row-major: E[m,n] at m*N + n, Stride=N
//
// NOTE: B is stored as model weights [N×K] Row-Major, which is Column-Major
// when viewed as [K×N]. No transpose needed - just reinterpret!
//
using ALayout  = Row;   // A[M×K] RowMajor - activations
using BLayout  = Col;   // B[K×N] ColumnMajor - weights (same memory as [N×K] RowMajor!)
using ELayout  = Row;   // E[M×N] RowMajor - output

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmMNPadding = ck::tensor_operation::device::GemmSpecialization::MNPadding;

// ============================================================================
// NO-SCALE KERNEL TYPE (USED BY TWO-KERNEL DEFAULT PATH)
// ============================================================================
//
// This kernel performs pure INT8×INT8→INT32 GEMM without any scaling.
// It is the FIRST STEP in the default two-kernel approach.
//
// USE CASES:
//   1. Two-Kernel Path (DEFAULT): First kernel, followed by applyScales_kernel
//   2. Debugging: Isolate whether issues are in the base GEMM or scaling
//   3. Reference: Verify GEMM correctness against CPU implementation
//
// Formula:
//   E[m,n] = sum_k( A_int8[m,k] × B_int8[k,n] )  [result is INT32]
//
// For production inference:
//   - Default path: This kernel + applyScales_kernel (via executeTwoKernel)
//   - Legacy path: DenseScale kernel (lower accuracy, retained for compatibility)
//
// ============================================================================
using DsDataType_None = ck::Tuple<>;  // No D tensors (empty tuple)
using EDataType_None  = I32;          // Output is INT32 (raw accumulator)
using DsLayout_None = ck::Tuple<>;    // No D layouts (empty tuple)

// PassThrough element op: e = static_cast<int32>(c)
// Just copies the accumulator value to output, no transformation
using CDEElementOp_None = PassThrough;

// clang-format off
using DeviceGemmInt8_NoScale = ck::tensor_operation::device::DeviceGemmMultipleD_Dl<
    ALayout,              // A layout: RowMajor [M×K]
    BLayout,              // B layout: ColumnMajor [K×N] (mk_nk_mn convention)
    DsLayout_None,        // No D tensors (empty tuple)
    ELayout,              // E layout: RowMajor [M×N]
    ADataType,            // A element type: int8_t
    BDataType,            // B element type: int8_t
    AccDataType,          // Accumulator type: int32_t
    DsDataType_None,      // No D data types (empty tuple)
    EDataType_None,       // E element type: int32_t (raw accumulator)
    AElementOp,           // A element-wise op: PassThrough
    BElementOp,           // B element-wise op: PassThrough
    CDEElementOp_None,    // Output element-wise op: PassThrough
    GemmDefault,          // GemmSpecialization::Default
    // =========================================================================
    // BLOCK AND TILE CONFIGURATION (mk_nk_mn - 128x128 tiles!)
    // Same configuration as DenseScale for consistent behavior
    // =========================================================================
    256,                  // BlockSize: 256 threads per block (16×16 thread tile)
    128,                  // MPerBlock: 128 output rows per block
    128,                  // NPerBlock: 128 output cols per block
    16,                   // K0PerBlock: K outer loop (16 iterations)
    4,                    // K1: K inner loop (16×4=64 K per iteration)
    4,                    // M1PerThread: 4 output rows per thread
    4,                    // N1PerThread: 4 output cols per thread  
    1,                    // KPerThread: 1 K element per thread per iteration
    S<2, 8>,              // M1N1ThreadClusterM1Xs: thread cluster M tiling
    S<2, 8>,              // M1N1ThreadClusterN1Xs: thread cluster N tiling
    S<8, 1, 1, 4>,        // ABlockTransferThreadSliceLengths_K0_M0_M1_K1
    S<2, 1, 128, 1>,      // ABlockTransferThreadClusterLengths_K0_M0_M1_K1
    S<1, 2, 0, 3>,        // ABlockTransferThreadClusterArrangeOrder
    S<1, 2, 0, 3>,        // ABlockTransferSrcAccessOrder
    S<4, 1, 1, 4>,        // ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1
    S<1, 2, 0, 3>,        // ABlockTransferSrcVectorTensorContiguousDimOrder
    S<1, 1, 1, 4>,        // ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1
    S<8, 1, 1, 4>,        // BBlockTransferThreadSliceLengths_K0_N0_N1_K1
    S<2, 1, 128, 1>,      // BBlockTransferThreadClusterLengths_K0_N0_N1_K1
    S<1, 2, 0, 3>,        // BBlockTransferThreadClusterArrangeOrder
    S<1, 2, 0, 3>,        // BBlockTransferSrcAccessOrder
    S<4, 1, 1, 4>,        // BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1
    S<1, 2, 0, 3>,        // BBlockTransferSrcVectorTensorContiguousDimOrder
    S<1, 1, 1, 4>,        // BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1
    S<0, 1, 2, 3, 4, 5>,  // CThreadTransferSrcDstAccessOrder
    5,                    // CThreadTransferSrcDstVectorDim (vectorize dim 5)
    4>;                   // CThreadTransferDstScalarPerVector (4-wide writes)
// clang-format on

// ============================================================================
// 32×32 TILE KERNEL (For M ≤ 32) - OPTIMAL FOR DECODE
// ============================================================================
//
// Benchmark results (K=3584, N=18944, MI50 gfx906):
//   M=1:  0.46ms, ~9.4 TFLOPS (vs 1.2ms with 128-pad = 2.6x faster!)
//   M=8:  0.47ms, ~9.3 TFLOPS (vs 1.2ms with 128-pad = 2.6x faster!)
//   M=32: 0.46ms, ~9.4 TFLOPS (vs 0.66ms with 128-pad = 1.4x faster!)
//
// This kernel uses GemmMNPadding to handle irregular M dimensions natively.
// Tile parameters from device_gemm_dl_i8_i8_i8_mk_nk_mn_irregular_instance.cpp
//   https://github.com/ROCm/composable_kernel/blob/develop/library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_i8_i8_i8_mk_nk_mn_irregular_instance.cpp
//
// clang-format off
using DeviceGemmInt8_NoScale_32x32 = ck::tensor_operation::device::DeviceGemmMultipleD_Dl<
    ALayout,              // A layout: RowMajor [M×K]
    BLayout,              // B layout: ColumnMajor [K×N]
    DsLayout_None,        // No D tensors
    ELayout,              // E layout: RowMajor [M×N]
    ADataType,            // int8_t
    BDataType,            // int8_t
    AccDataType,          // int32_t
    DsDataType_None,      // No D data
    EDataType_None,       // int32_t output
    AElementOp,           // PassThrough
    BElementOp,           // PassThrough
    CDEElementOp_None,    // PassThrough
    GemmMNPadding,        // Enable M/N padding for irregular dimensions!
    // =========================================================================
    // BLOCK AND TILE CONFIGURATION (32×32 tiles)
    // =========================================================================
    32,                   // BlockSize: 32 threads per block
    32,                   // MPerBlock: 32 output rows per block
    32,                   // NPerBlock: 32 output cols per block
    8,                    // K0PerBlock: K outer loop
    4,                    // K1: K inner loop (8×4=32 K per iteration)
    2,                    // M1PerThread
    4,                    // N1PerThread
    1,                    // KPerThread
    S<4, 2>,              // M1N1ThreadClusterM1Xs
    S<2, 2>,              // M1N1ThreadClusterN1Xs
    S<4, 1, 2, 4>,        // ABlockTransferThreadSliceLengths_K0_M0_M1_K1
    S<2, 1, 16, 1>,       // ABlockTransferThreadClusterLengths_K0_M0_M1_K1
    S<1, 2, 0, 3>,        // ABlockTransferThreadClusterArrangeOrder
    S<1, 2, 0, 3>,        // ABlockTransferSrcAccessOrder
    S<4, 1, 1, 4>,        // ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1
    S<1, 2, 0, 3>,        // ABlockTransferSrcVectorTensorContiguousDimOrder
    S<1, 1, 1, 4>,        // ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1
    S<4, 1, 2, 4>,        // BBlockTransferThreadSliceLengths_K0_N0_N1_K1
    S<2, 1, 16, 1>,       // BBlockTransferThreadClusterLengths_K0_N0_N1_K1
    S<1, 2, 0, 3>,        // BBlockTransferThreadClusterArrangeOrder
    S<1, 2, 0, 3>,        // BBlockTransferSrcAccessOrder
    S<4, 1, 1, 4>,        // BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1
    S<1, 2, 0, 3>,        // BBlockTransferSrcVectorTensorContiguousDimOrder
    S<1, 1, 1, 4>,        // BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1
    S<0, 1, 2, 3, 4, 5>,  // CThreadTransferSrcDstAccessOrder
    5,                    // CThreadTransferSrcDstVectorDim
    1>;                   // CThreadTransferDstScalarPerVector
// clang-format on

// ============================================================================
// 64×64 TILE KERNEL (For 32 < M < 128)
// ============================================================================
//
// Benchmark results (K=3584, N=18944, MI50 gfx906):
//   M=64: 0.51ms, ~17 TFLOPS (vs 0.66ms with 128-pad = 1.3x faster!)
//
// This kernel uses GemmMNPadding to handle irregular M dimensions natively.
// Tile parameters from device_gemm_dl_i8_i8_i8_mk_nk_mn_irregular_instance.cpp
//   https://github.com/ROCm/composable_kernel/blob/develop/library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_i8_i8_i8_mk_nk_mn_irregular_instance.cpp
//
// clang-format off
using DeviceGemmInt8_NoScale_64x64 = ck::tensor_operation::device::DeviceGemmMultipleD_Dl<
    ALayout,              // A layout: RowMajor [M×K]
    BLayout,              // B layout: ColumnMajor [K×N]
    DsLayout_None,        // No D tensors
    ELayout,              // E layout: RowMajor [M×N]
    ADataType,            // int8_t
    BDataType,            // int8_t
    AccDataType,          // int32_t
    DsDataType_None,      // No D data
    EDataType_None,       // int32_t output
    AElementOp,           // PassThrough
    BElementOp,           // PassThrough
    CDEElementOp_None,    // PassThrough
    GemmMNPadding,        // Enable M/N padding for irregular dimensions!
    // =========================================================================
    // BLOCK AND TILE CONFIGURATION (64×64 tiles)
    // =========================================================================
    64,                   // BlockSize: 64 threads per block
    64,                   // MPerBlock: 64 output rows per block
    64,                   // NPerBlock: 64 output cols per block
    8,                    // K0PerBlock: K outer loop
    4,                    // K1: K inner loop (8×4=32 K per iteration)
    4,                    // M1PerThread
    4,                    // N1PerThread
    1,                    // KPerThread
    S<4, 2>,              // M1N1ThreadClusterM1Xs
    S<4, 2>,              // M1N1ThreadClusterN1Xs
    S<4, 1, 2, 4>,        // ABlockTransferThreadSliceLengths_K0_M0_M1_K1
    S<2, 1, 32, 1>,       // ABlockTransferThreadClusterLengths_K0_M0_M1_K1
    S<1, 2, 0, 3>,        // ABlockTransferThreadClusterArrangeOrder
    S<1, 2, 0, 3>,        // ABlockTransferSrcAccessOrder
    S<4, 1, 1, 4>,        // ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1
    S<1, 2, 0, 3>,        // ABlockTransferSrcVectorTensorContiguousDimOrder
    S<1, 1, 1, 4>,        // ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1
    S<4, 1, 2, 4>,        // BBlockTransferThreadSliceLengths_K0_N0_N1_K1
    S<2, 1, 32, 1>,       // BBlockTransferThreadClusterLengths_K0_N0_N1_K1
    S<1, 2, 0, 3>,        // BBlockTransferThreadClusterArrangeOrder
    S<1, 2, 0, 3>,        // BBlockTransferSrcAccessOrder
    S<4, 1, 1, 4>,        // BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1
    S<1, 2, 0, 3>,        // BBlockTransferSrcVectorTensorContiguousDimOrder
    S<1, 1, 1, 4>,        // BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1
    S<0, 1, 2, 3, 4, 5>,  // CThreadTransferSrcDstAccessOrder
    5,                    // CThreadTransferSrcDstVectorDim
    1>;                   // CThreadTransferDstScalarPerVector
// clang-format on

// ============================================================================
// 128×128 TILE KERNEL WITH MN-PADDING (For M >= 128, irregular M)
// ============================================================================
//
// This kernel is identical to DeviceGemmInt8_NoScale but uses GemmMNPadding
// instead of GemmDefault, allowing it to handle M values that are not
// multiples of 128 (e.g., M=596 for prefill with 596 tokens).
//
// The GemmMNPadding specialization adds boundary checking at the tile level,
// which has minimal overhead (~5%) compared to GemmDefault.
//
// USE CASE:
//   - Prefill with arbitrary sequence lengths (M = seq_len)
//   - M >= 128 but M % 128 != 0 (e.g., M=596, M=1024+13, etc.)
//
// clang-format off
using DeviceGemmInt8_NoScale_128x128_MNPadding = ck::tensor_operation::device::DeviceGemmMultipleD_Dl<
    ALayout,              // A layout: RowMajor [M×K]
    BLayout,              // B layout: ColumnMajor [K×N] (mk_nk_mn convention)
    DsLayout_None,        // No D tensors (empty tuple)
    ELayout,              // E layout: RowMajor [M×N]
    ADataType,            // A element type: int8_t
    BDataType,            // B element type: int8_t
    AccDataType,          // Accumulator type: int32_t
    DsDataType_None,      // No D data types (empty tuple)
    EDataType_None,       // E element type: int32_t (raw accumulator)
    AElementOp,           // A element-wise op: PassThrough
    BElementOp,           // B element-wise op: PassThrough
    CDEElementOp_None,    // Output element-wise op: PassThrough
    GemmMNPadding,        // GemmSpecialization::MNPadding for irregular M!
    // =========================================================================
    // BLOCK AND TILE CONFIGURATION (mk_nk_mn - 128x128 tiles!)
    // Same configuration as DeviceGemmInt8_NoScale
    // =========================================================================
    256,                  // BlockSize: 256 threads per block (16×16 thread tile)
    128,                  // MPerBlock: 128 output rows per block
    128,                  // NPerBlock: 128 output cols per block
    16,                   // K0PerBlock: K outer loop (16 iterations)
    4,                    // K1: K inner loop (16×4=64 K per iteration)
    4,                    // M1PerThread: 4 output rows per thread
    4,                    // N1PerThread: 4 output cols per thread  
    1,                    // KPerThread: 1 K element per thread per iteration
    S<2, 8>,              // M1N1ThreadClusterM1Xs: thread cluster M tiling
    S<2, 8>,              // M1N1ThreadClusterN1Xs: thread cluster N tiling
    S<8, 1, 1, 4>,        // ABlockTransferThreadSliceLengths_K0_M0_M1_K1
    S<2, 1, 128, 1>,      // ABlockTransferThreadClusterLengths_K0_M0_M1_K1
    S<1, 2, 0, 3>,        // ABlockTransferThreadClusterArrangeOrder
    S<1, 2, 0, 3>,        // ABlockTransferSrcAccessOrder
    S<4, 1, 1, 4>,        // ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1
    S<1, 2, 0, 3>,        // ABlockTransferSrcVectorTensorContiguousDimOrder
    S<1, 1, 1, 4>,        // ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1
    S<8, 1, 1, 4>,        // BBlockTransferThreadSliceLengths_K0_N0_N1_K1
    S<2, 1, 128, 1>,      // BBlockTransferThreadClusterLengths_K0_N0_N1_K1
    S<1, 2, 0, 3>,        // BBlockTransferThreadClusterArrangeOrder
    S<1, 2, 0, 3>,        // BBlockTransferSrcAccessOrder
    S<4, 1, 1, 4>,        // BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1
    S<1, 2, 0, 3>,        // BBlockTransferSrcVectorTensorContiguousDimOrder
    S<1, 1, 1, 4>,        // BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1
    S<0, 1, 2, 3, 4, 5>,  // CThreadTransferSrcDstAccessOrder
    5,                    // CThreadTransferSrcDstVectorDim (vectorize dim 5)
    4>;                   // CThreadTransferDstScalarPerVector (4-wide writes)
// clang-format on

// ============================================================================
// CACHED KERNEL INSTANCES (Lazy Initialization Pattern)
// ============================================================================
//
// WHY CACHING MATTERS:
//   CK device objects are expensive to construct (~10-100ms for first call):
//     - Template instantiation and LLVM IR generation
//     - GPU ISA compatibility checks
//     - Internal buffer allocation
//
//   By using per-instance kernel context, we:
//     - Pay construction cost only on first use per kernel instance
//     - Keep CK object lifetime bound to ROCmQuantisedGemmKernel lifetime
//     - Avoid process-global mutable state across concurrent device execution
//     - Keep repeated dispatch on the same instance allocation-free
//
// CRITICAL: CK kernel objects may have internal mutable state (cached streams,
// workspace buffers) that is NOT thread-safe for concurrent use from different
// HIP devices. We maintain per-device instances to avoid deadlocks/races.
//
// 3-WAY DISPATCH STRATEGY (January 2026):
//
//   ┌────────────────────────┬───────────────────────────────────────────────────┐
//   │ M Range                │ Kernel Used                                       │
//   ├────────────────────────┼───────────────────────────────────────────────────┤
//   │ M ≤ 32                 │ 32×32 kernel (GemmMNPadding handles irregular M)  │
//   │ 32 < M < 128           │ 64×64 kernel (GemmMNPadding handles irregular M)  │
//   │ M ≥ 128, M % 128 == 0  │ 128×128 kernel (GemmDefault, peak throughput)     │
//   │ M ≥ 128, M % 128 != 0  │ 128×128 kernel (GemmMNPadding, handles irregular) │
//   └────────────────────────┴───────────────────────────────────────────────────┘
//
// Performance (FFN Up: K=3584, N=18944, MI50 gfx906):
//   M=1:   0.46ms, ~2.4 TFLOPS  (32×32 kernel)
//   M=8:   0.45ms, ~2.4 TFLOPS  (32×32 kernel)
//   M=32:  0.53ms, ~8.2 TFLOPS  (32×32 kernel)
//   M=64:  0.51ms, ~17 TFLOPS   (64×64 kernel)
//   M=128: 0.82ms, ~21 TFLOPS   (128×128 kernel)
//
// ============================================================================

struct ROCmCKKernelContext
{
    int rocm_device_id = 0;
    std::unique_ptr<DeviceGemmInt8_NoScale> kernel_128x128;
    std::unique_ptr<DeviceGemmInt8_NoScale_128x128_MNPadding> kernel_128x128_mn_padding;
    std::unique_ptr<DeviceGemmInt8_NoScale_64x64> kernel_64x64;
    std::unique_ptr<DeviceGemmInt8_NoScale_32x32> kernel_32x32;
};

static DeviceGemmInt8_NoScale& getCachedNoScaleKernel(ROCmCKKernelContext& context)
{
    if (!context.kernel_128x128)
    {
        context.kernel_128x128 = std::make_unique<DeviceGemmInt8_NoScale>();
    }
    return *context.kernel_128x128;
}

static DeviceGemmInt8_NoScale_128x128_MNPadding& getCachedNoScaleKernel128x128MNPadding(ROCmCKKernelContext& context)
{
    if (!context.kernel_128x128_mn_padding)
    {
        context.kernel_128x128_mn_padding = std::make_unique<DeviceGemmInt8_NoScale_128x128_MNPadding>();
    }
    return *context.kernel_128x128_mn_padding;
}

static DeviceGemmInt8_NoScale_64x64& getCachedNoScaleKernel64x64(ROCmCKKernelContext& context)
{
    if (!context.kernel_64x64)
    {
        context.kernel_64x64 = std::make_unique<DeviceGemmInt8_NoScale_64x64>();
    }
    return *context.kernel_64x64;
}

static DeviceGemmInt8_NoScale_32x32& getCachedNoScaleKernel32x32(ROCmCKKernelContext& context)
{
    if (!context.kernel_32x32)
    {
        context.kernel_32x32 = std::make_unique<DeviceGemmInt8_NoScale_32x32>();
    }
    return *context.kernel_32x32;
}

enum class NoScaleKernelClass
{
    K32x32,
    K64x64,
    K128x128Default,
    K128x128MNPadding,
};

static NoScaleKernelClass chooseNoScaleKernelClass(int M, int N, int K)
{
    if ((M % 128) == 0)
    {
        return NoScaleKernelClass::K128x128Default;
    }

    const double safe_k = static_cast<double>(std::max(K, 1));
    const double aspect_n_over_k = static_cast<double>(N) / safe_k;

    if (M <= 32 && aspect_n_over_k < 8.0)
    {
        return NoScaleKernelClass::K32x32;
    }

    if (M < 128 && aspect_n_over_k < 4.0)
    {
        return NoScaleKernelClass::K64x64;
    }

    return NoScaleKernelClass::K128x128MNPadding;
}

} // anonymous namespace

extern "C"
{
    bool rocmQuantGemm_setDevice(int rocm_device_id);

    // -------------------------------------------------------------------------
    // Dimension requirements (CK tile-based constraints)
    // -------------------------------------------------------------------------
    //
    // CK's DeviceGemmMultipleD_Dl uses fixed tile sizes which impose minimum
    // dimension requirements. Use these functions to check before invoking.
    // 
    // With the 3-way dispatch and GemmMNPadding on 32×32/64×64 kernels:
    //   M >= 8 (after C++ caller padding), N >= 64, K >= 64
    // The 32×32 kernel handles M=8 to M=32 natively without explicit padding.

    int rocmQuantGemm_getMinM() { return CK_SMALL_MIN_M; }
    int rocmQuantGemm_getMinN() { return CK_SMALL_MIN_N; }
    int rocmQuantGemm_getMinK() { return CK_SMALL_MIN_K; }

    bool rocmQuantGemm_areDimensionsSupported(int M, int N, int K)
    {
        return ck_dimensions_supported(M, N, K);
    }

    // -------------------------------------------------------------------------
    // CK Kernel Warmup (Pre-initialization)
    // -------------------------------------------------------------------------
    //
    // CK device objects are expensive to construct (5-10 seconds each on gfx906)
    // because of:
    //   - Template instantiation and LLVM IR generation
    //   - GPU ISA compatibility checks
    //   - Internal buffer allocation
    //
    // This function pre-initializes all 3 CK kernels (32x32, 64x64, 128x128)
    // at startup to avoid blocking on first inference call.
    //
    // Call this from BackendManager::initROCmBackend() after device setup.
    
    void *rocmQuantGemm_createKernelContext(int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return nullptr;

        auto *context = new (std::nothrow) ROCmCKKernelContext();
        if (!context)
            return nullptr;

        context->rocm_device_id = rocm_device_id;
        return context;
    }

    void rocmQuantGemm_destroyKernelContext(void *kernel_ctx, int rocm_device_id)
    {
        if (!kernel_ctx)
            return;
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return;
        auto *context = static_cast<ROCmCKKernelContext *>(kernel_ctx);
        delete context;
    }

    void rocmQuantGemm_warmupKernels()
    {
        // Force construction of all kernel variants for this warmup context
        ROCmCKKernelContext context{};
        (void)getCachedNoScaleKernel32x32(context);  // 32x32 for decode (M <= 32)

        (void)getCachedNoScaleKernel64x64(context);  // 64x64 for small batch (32 < M < 128)

        (void)getCachedNoScaleKernel(context);       // 128x128 for prefill (M >= 128, M%128==0)

        (void)getCachedNoScaleKernel128x128MNPadding(context);  // 128x128 MNPadding for M%128!=0
    }

    // =========================================================================
    // DIMENSION VALIDATION
    // =========================================================================
    //
    // rocmQuantGemm_isSupportedDimensions - Check if CK supports given M/N/K
    //
    // With the 3-way dispatch, CK now supports ALL M values ≥ 1:
    //   - M ≤ 32:     32×32 kernel with GemmMNPadding (handles M=1 natively)
    //   - 32 < M < 128: 64×64 kernel with GemmMNPadding
    //   - M ≥ 128:    128×128 kernel (peak throughput)
    //
    // Minimum requirements: N ≥ 64, K ≥ 64 (always satisfied by LLM dims)
    //
    // NOTE: In practice, this always returns true for valid LLM dimensions.
    //       Qwen2.5-0.5B smallest: M=1 (decode), K=896, N=896
    //       The 32×32 kernel handles M=1 directly without padding!
    //
    // =========================================================================
    bool rocmQuantGemm_isSupportedDimensions(int M, int N, int K)
    {
        // All positive dimensions are supported via 3-way dispatch
        return M > 0 && N > 0 && K > 0 && ck_dimensions_supported(M, N, K);
    }

    // =========================================================================
    // DEBUG: NoScale INT8 GEMM (Raw INT32 Output)
    // =========================================================================
    //
    // rocmQuantGemm_executeNoScale - INT8×INT8→INT32 without scaling
    //
    // FORMULA:
    //   E_int32[m,n] = sum_k(A_int8[m,k] × B_int8[k,n])
    //
    // 3-WAY DISPATCH STRATEGY (January 2026):
    //   - M <= 32:     Use 32×32 kernel (~9.4 TFLOPS, 2.6x faster than 128-pad)
    //   - 32 < M < 128: Use 64×64 kernel (~17 TFLOPS, 1.3x faster than 128-pad)
    //   - M >= 128:    Use 128×128 kernel (~26 TFLOPS, optimal for large M)
    //
    // The 32×32 and 64×64 kernels use GemmSpecialization::MNPadding to handle
    // irregular M dimensions. The 128×128 kernel uses GemmSpecialization::Default
    // and requires M ≥ 128.
    //
    // Benchmark Results (K=3584, N=18944, MI50 gfx906):
    //   M=1:   32×32: 0.46ms / 128-pad: 1.2ms = 2.6x faster
    //   M=8:   32×32: 0.47ms / 128-pad: 1.2ms = 2.6x faster
    //   M=32:  32×32: 0.46ms / 128-pad: 0.66ms = 1.4x faster
    //   M=64:  64×64: 0.51ms / 128-pad: 0.66ms = 1.3x faster
    //   M=128: 128×128: 0.66ms (direct, no padding)
    //
    // =========================================================================
    
    bool rocmQuantGemm_executeNoScale(
        const int8_t *d_A_int8,
        const int8_t *d_weights_int8,
        int32_t *d_C_int32,
        int M, int N, int K,
        int rocm_device_id, void* stream,
        void* kernel_ctx)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        if (M <= 0 || N <= 0 || K <= 0)
            return true;

        // No D tensors - empty arrays
        std::array<const void*, 0> p_ds = {};
        std::array<ck::index_t, 0> strides_ds = {};
        
        // DEBUG: Time the actual CK GEMM execution (including GPU work completion)
        // This adds a sync so we can measure ACTUAL kernel execution time
        static constexpr bool debug_timing = false;
        auto t0 = std::chrono::high_resolution_clock::now();

        ROCmCKKernelContext local_context;
        local_context.rocm_device_id = rocm_device_id;
        auto* context = kernel_ctx ? static_cast<ROCmCKKernelContext*>(kernel_ctx) : &local_context;

        if (context->rocm_device_id != rocm_device_id)
        {
            fprintf(stderr,
                "[rocmQuantGemm_executeNoScale] Context device mismatch: ctx=%d call=%d\n",
                context->rocm_device_id, rocm_device_id);
            return false;
        }

        // =====================================================================
        // 4-WAY DISPATCH (ASPECT-RATIO AWARE)
        // =====================================================================
        // Prefer selecting kernels by output/input aspect (N/K) first, then M.
        // This avoids overfitting to fixed row counts and generalizes across
        // smaller/larger prefill batches while retaining 128x128 fastpath on
        // exact multiples of 128.
        // =====================================================================

        // NOTE: ScopedCoutSilence removed - was silencing cout during CK execution
        // which could cause issues if kernel hangs (destructor never runs, cout stays silenced)
        // CK debug output like "arg.e_grid_desc_m_n_" will now be visible.
        
        const NoScaleKernelClass kernel_class = chooseNoScaleKernelClass(M, N, K);

        if (kernel_class == NoScaleKernelClass::K32x32)
        {
            // Small M: Use 32×32 kernel (optimal for decode, M=1-32)
            auto& device_gemm = getCachedNoScaleKernel32x32(*context);
            auto argument = device_gemm.MakeArgument(
                d_A_int8, d_weights_int8, p_ds, d_C_int32,
                M, N, K,
                K, K, strides_ds, N,
                AElementOp{}, BElementOp{}, CDEElementOp_None{}
            );

            if (!device_gemm.IsSupportedArgument(argument))
            {
                fprintf(stderr, "[rocmQuantGemm_executeNoScale] 32×32 kernel doesn't support M=%d N=%d K=%d\n",
                        M, N, K);
                return false;
            }

            auto invoker = device_gemm.MakeInvoker();
            invoker.Run(argument, StreamConfig{static_cast<hipStream_t>(stream), false});
        }
        else if (kernel_class == NoScaleKernelClass::K64x64)
        {
            // Medium M: Use 64×64 kernel (optimal for M=33-127)
            auto& device_gemm = getCachedNoScaleKernel64x64(*context);
            auto argument = device_gemm.MakeArgument(
                d_A_int8, d_weights_int8, p_ds, d_C_int32,
                M, N, K,
                K, K, strides_ds, N,
                AElementOp{}, BElementOp{}, CDEElementOp_None{}
            );

            if (!device_gemm.IsSupportedArgument(argument))
            {
                fprintf(stderr, "[rocmQuantGemm_executeNoScale] 64×64 kernel doesn't support M=%d N=%d K=%d\n",
                        M, N, K);
                return false;
            }

            auto invoker = device_gemm.MakeInvoker();
            invoker.Run(argument, StreamConfig{static_cast<hipStream_t>(stream), false});
        }
        else if (kernel_class == NoScaleKernelClass::K128x128Default)
        {
            // Large M, multiple of 128: Use 128×128 kernel with GemmDefault (peak throughput)
            auto& device_gemm = getCachedNoScaleKernel(*context);
            auto argument = device_gemm.MakeArgument(
                d_A_int8, d_weights_int8, p_ds, d_C_int32,
                M, N, K,
                K, K, strides_ds, N,
                AElementOp{}, BElementOp{}, CDEElementOp_None{}
            );

            if (!device_gemm.IsSupportedArgument(argument))
            {
                fprintf(stderr, "[rocmQuantGemm_executeNoScale] 128×128 kernel doesn't support M=%d N=%d K=%d\n",
                        M, N, K);
                return false;
            }

            auto invoker = device_gemm.MakeInvoker();
            invoker.Run(argument, StreamConfig{static_cast<hipStream_t>(stream), false});
        }
        else
        {
            auto& device_gemm = getCachedNoScaleKernel128x128MNPadding(*context);
            auto argument = device_gemm.MakeArgument(
                d_A_int8, d_weights_int8, p_ds, d_C_int32,
                M, N, K,
                K, K, strides_ds, N,
                AElementOp{}, BElementOp{}, CDEElementOp_None{}
            );

            if (!device_gemm.IsSupportedArgument(argument))
            {
                fprintf(stderr, "[rocmQuantGemm_executeNoScale] 128×128 MNPadding kernel doesn't support M=%d N=%d K=%d\n",
                        M, N, K);
                return false;
            }

            auto invoker = device_gemm.MakeInvoker();
            invoker.Run(argument, StreamConfig{static_cast<hipStream_t>(stream), false});
        }
        
        // DEBUG: If timing enabled, sync and report actual GEMM execution time
        if (debug_timing) {
            hipError_t sync_err = hipDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            fprintf(stderr, "[CK_GEMM_EXEC] M=%d N=%d K=%d  time=%.3fms  sync=%s\n", 
                    M, N, K, ms, (sync_err == hipSuccess ? "ok" : hipGetErrorString(sync_err)));
        }
        
        return true;
    }

} // extern "C"
