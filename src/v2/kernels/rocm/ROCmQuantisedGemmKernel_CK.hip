/**
 * @file ROCmQuantisedGemmKernel_CK.hip
 * @brief ComposableKernel INT8 GEMM implementation for AMD ROCm GPUs
 *
 * This file implements INT8×INT8→FP32 GEMM operations for AMD GPUs using the
 * ComposableKernel (CK) library. It provides efficient quantized inference for
 * LLM workloads on AMD Instinct GPUs.
 *
 * ============================================================================
 * QUICK START FOR NEW DEVELOPERS
 * ============================================================================
 *
 * What this file does:
 *   1. Takes INT8 quantized weights and activations
 *   2. Performs matrix multiplication: C = A × B
 *   3. Applies per-row and per-column scaling to get FP32 output
 *
 * Key functions (in order of preference):
 *   - rocmQuantGemm_executeTwoKernel_cached() : HOT PATH - CK GEMM + separate scaling (no alloc)
 *   - rocmQuantGemm_executeTwoKernel_padded() : For small M (pads to nearest tile)
 *   - rocmQuantGemm_executeTwoKernel()        : Allocating version (for testing)
 *   - rocmQuantGemm_quantizeActivations()     : FP32→INT8 activation quantization
 *   - rocmQuantGemm_uploadWeights()           : Upload weights to GPU
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

#include "../../backends/rocm/HipDeviceGuard.h"

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

enum class QuantShapeClass
{
    ExtremeWide,
    Wide,
    Balanced,
    Tall,
};

struct QuantAspectThresholds
{
    double extreme_wide;
    double wide;
    double balanced_min;
};

struct QuantLdsPolicy
{
    int max_k;
    int min_m;
};

static QuantAspectThresholds getQuantAspectThresholds()
{
    static const QuantAspectThresholds thresholds{64.0, 8.0, 0.125};

    return thresholds;
}

static QuantLdsPolicy getQuantLdsPolicy()
{
    static const QuantLdsPolicy policy{4096, 4};

    return policy;
}

static QuantShapeClass classifyQuantShape(int M, int K)
{
    const double safe_m = static_cast<double>(std::max(M, 1));
    const double aspect_k_over_m = static_cast<double>(K) / safe_m;
    const QuantAspectThresholds thresholds = getQuantAspectThresholds();

    if (aspect_k_over_m >= thresholds.extreme_wide)
    {
        return QuantShapeClass::ExtremeWide;
    }
    if (aspect_k_over_m >= thresholds.wide)
    {
        return QuantShapeClass::Wide;
    }
    if (aspect_k_over_m >= thresholds.balanced_min)
    {
        return QuantShapeClass::Balanced;
    }
    return QuantShapeClass::Tall;
}

static int chooseQuantThreadsFromShape(int M, int K)
{
    switch (classifyQuantShape(M, K))
    {
        case QuantShapeClass::ExtremeWide:
            return 512;
        case QuantShapeClass::Wide:
            return 256;
        case QuantShapeClass::Balanced:
            return 256;
        case QuantShapeClass::Tall:
            return 128;
    }

    return 256;
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

// ============================================================================
// HIP Kernel: FP32 → INT8 Activation Quantization (Phase 2)
// ============================================================================

/**
 * @brief Per-row symmetric quantization: FP32 → INT8 with per-row scale
 *
 * Algorithm:
 *   1. Find max_abs = max(|x_i|) for each row
 *   2. scale = max_abs / 127.0 (or 1.0 if max_abs == 0)
 *   3. int8_i = round(x_i / scale) clamped to [-127, 127]
 *
 * Grid: (M, 1, 1) blocks
 * Block: shape-selected threads (128/256/512), chosen from matrix aspect ratio K/M
 *
 * @param d_A_fp32 Input FP32 matrix [M × K] RowMajor
 * @param d_A_int8 Output INT8 matrix [M × K] RowMajor
 * @param d_scales Output per-row scales [M]
 * @param M Number of rows
 * @param K Number of columns per row
 */
template <bool UseLdsRowStaging, bool NoTail>
__global__ void quantizeActivationsQ8_kernel_t(
    const float *__restrict__ d_A_fp32,
    int8_t *__restrict__ d_A_int8,
    float *__restrict__ d_scales,
    int M, int K)
{
    const int row = blockIdx.x;
    if (row >= M)
        return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Shared memory for wavefront-level max merge.
    // Largest launch uses 512 threads => 8 wave64 groups.
    __shared__ float s_wave_max[8];
    __shared__ float s_absmax;

    extern __shared__ float s_row[];

    // Step 1: Find max absolute value in this row (parallel reduction)
    float local_max = 0.0f;
    const float *row_ptr = d_A_fp32 + static_cast<size_t>(row) * K;
    const bool fp32_aligned16 = ((reinterpret_cast<uintptr_t>(row_ptr) & 0x0F) == 0);

    if constexpr (UseLdsRowStaging)
    {
        for (int k = tid; k < K; k += block_size)
        {
            s_row[k] = row_ptr[k];
        }
        __syncthreads();
    }

    const float* src_ptr = UseLdsRowStaging ? s_row : row_ptr;
    const int vec_len = NoTail ? K : ((K / 4) * 4);
    const bool src_aligned16 = ((reinterpret_cast<uintptr_t>(src_ptr) & 0x0F) == 0);

    if (src_aligned16)
    {
        for (int k4 = tid * 4; k4 < vec_len; k4 += block_size * 4)
        {
            const float4 v = *reinterpret_cast<const float4 *>(src_ptr + k4);
            local_max = fmaxf(local_max, fabsf(v.x));
            local_max = fmaxf(local_max, fabsf(v.y));
            local_max = fmaxf(local_max, fabsf(v.z));
            local_max = fmaxf(local_max, fabsf(v.w));
        }
        if constexpr (!NoTail)
        {
            for (int k = vec_len + tid; k < K; k += block_size)
            {
                local_max = fmaxf(local_max, fabsf(src_ptr[k]));
            }
        }
    }
    else
    {
        for (int k = tid; k < K; k += block_size)
        {
            float val = src_ptr[k];
            float abs_val = fabsf(val);
            local_max = fmaxf(local_max, abs_val);
        }
    }

    // Wavefront reduction (wave64 on gfx906)
    const int lane = tid & 63;
    const int wave = tid >> 6;
    const int wave_count = block_size >> 6;
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1)
    {
        local_max = fmaxf(local_max, __shfl_down(local_max, offset));
    }

    if (lane == 0)
    {
        s_wave_max[wave] = local_max;
    }
    __syncthreads();

    // First wave merges all per-wave maxima
    if (wave == 0)
    {
        float v = (tid < wave_count) ? s_wave_max[tid] : 0.0f;
        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1)
        {
            v = fmaxf(v, __shfl_down(v, offset));
        }
        if (tid == 0)
        {
            s_absmax = v;
        }
    }
    __syncthreads();

    // Step 2: Compute scale (thread 0 writes to global memory)
    float max_abs = s_absmax;
    float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    if (tid == 0)
    {
        d_scales[row] = scale;
    }

    // Step 3: Quantize all elements in this row
    int8_t *out_row = d_A_int8 + static_cast<size_t>(row) * K;
    const bool int8_aligned4 = ((reinterpret_cast<uintptr_t>(out_row) & 0x03) == 0);

    if (src_aligned16 && int8_aligned4)
    {
        for (int k4 = tid * 4; k4 < vec_len; k4 += block_size * 4)
        {
            const float4 v = *reinterpret_cast<const float4 *>(src_ptr + k4);

            int q0 = static_cast<int>(rintf(v.x * inv_scale));
            int q1 = static_cast<int>(rintf(v.y * inv_scale));
            int q2 = static_cast<int>(rintf(v.z * inv_scale));
            int q3 = static_cast<int>(rintf(v.w * inv_scale));

            q0 = (q0 < -127) ? -127 : ((q0 > 127) ? 127 : q0);
            q1 = (q1 < -127) ? -127 : ((q1 > 127) ? 127 : q1);
            q2 = (q2 < -127) ? -127 : ((q2 > 127) ? 127 : q2);
            q3 = (q3 < -127) ? -127 : ((q3 > 127) ? 127 : q3);

            const uint32_t packed =
                static_cast<uint32_t>(static_cast<uint8_t>(q0)) |
                (static_cast<uint32_t>(static_cast<uint8_t>(q1)) << 8) |
                (static_cast<uint32_t>(static_cast<uint8_t>(q2)) << 16) |
                (static_cast<uint32_t>(static_cast<uint8_t>(q3)) << 24);

            *reinterpret_cast<uint32_t *>(out_row + k4) = packed;
        }

        if constexpr (!NoTail)
        {
            for (int k = vec_len + tid; k < K; k += block_size)
            {
                int q = static_cast<int>(rintf(src_ptr[k] * inv_scale));
                q = (q < -127) ? -127 : ((q > 127) ? 127 : q);
                out_row[k] = static_cast<int8_t>(q);
            }
        }
    }
    else
    {
        for (int k = tid; k < K; k += block_size)
        {
            float val = src_ptr[k];
            float scaled = val * inv_scale;

            // rintf() maps to v_rndne_f32 (1 instruction) instead of
            // roundf() which decomposes to trunc+sub+cmp+cndmask+bfi+add (6 instructions).
            // For int8 quantization, round-to-nearest-even vs round-half-away-from-zero
            // has negligible quality impact (differs only at exact x.5 tie points).
            int q = static_cast<int>(rintf(scaled));
            q = (q < -127) ? -127 : ((q > 127) ? 127 : q);

            out_row[k] = static_cast<int8_t>(q);
        }
    }
}

// ============================================================================
// Apply Scales Kernel (for two-kernel approach)
// ============================================================================
//
// Applies per-row and per-column scales to INT32 GEMM output:
//   E_fp32[m,n] = C_int32[m,n] * scale_A[m] * scale_B[n]
//
// This is the second kernel in the two-kernel approach, which is more reliable
// than CK's fused D-tensor handling on gfx906.
//
// Grid: ((M*N + 255) / 256, 1, 1) blocks
// Block: (256, 1, 1) threads
//
__global__ void applyScales_kernel(
    const int32_t* __restrict__ d_C_int32,  // [M×N] INT32 GEMM output
    float* __restrict__ d_E_fp32,           // [M×N] FP32 scaled output
    const float* __restrict__ d_scale_A,    // [M] per-row scales
    const float* __restrict__ d_scale_B,    // [N] per-column scales
    int M, int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    
    if (idx >= total) return;
    
    const int m = idx / N;
    const int n = idx % N;
    
    d_E_fp32[idx] = static_cast<float>(d_C_int32[idx]) * d_scale_A[m] * d_scale_B[n];
}

// ============================================================================
// Apply Scaling Kernel with Full Epilogue (alpha, beta, bias)
// ============================================================================
//
// Full-featured scaling epilogue matching CUDA's apply_scaling_kernel:
//   result = alpha * C_int32 * scale_A[m] * scale_B[n]
//   if (beta != 0) result += beta * C_existing[idx]
//   if (bias)      result += bias[n]
//
// Grid: ((N + 15) / 16, (M + 15) / 16, 1) blocks
// Block: (16, 16, 1) threads
//
__global__ void applyScaling_full_kernel(
    const int32_t* __restrict__ d_C_int32,  // [M×N] INT32 GEMM output
    float* __restrict__ d_C_fp32,           // [M×N] FP32 output
    const float* __restrict__ d_scale_A,    // [M] per-row scales
    const float* __restrict__ d_scale_B,    // [N] per-column scales
    int M, int N,
    float alpha, float beta,
    const float* __restrict__ d_C_existing, // [M×N] existing output (for beta != 0)
    const float* __restrict__ d_bias)       // [N] optional bias, broadcasted across rows
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N)
    {
        const int idx = row * N + col;
        const int32_t val_int32 = d_C_int32[idx];
        const float scale_combined = d_scale_A[row] * d_scale_B[col];
        float result = alpha * static_cast<float>(val_int32) * scale_combined;

        if (beta != 0.0f && d_C_existing != nullptr)
        {
            result += beta * d_C_existing[idx];
        }

        if (d_bias != nullptr)
        {
            result += d_bias[col];
        }

        d_C_fp32[idx] = result;
    }
}

// Decode-specialized scaling path (M=1, beta=0, no bias):
//   C_fp32[n] = alpha * C_int32[n] * scale_A[0] * scale_B[n]
// Uses 1D grid and vectorized x4 lanes to avoid the 16x16 underutilization
// of applyScaling_full_kernel when only a single row is active.
__global__ void applyScaling_m1_vec4_kernel(
    const int32_t* __restrict__ d_C_int32,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scale_A,
    const float* __restrict__ d_scale_B,
    int N,
    float alpha)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int n4 = tid * 4;

    const float row_scale = alpha * d_scale_A[0];

    if (n4 + 3 < N)
    {
        const int4 v = *reinterpret_cast<const int4*>(d_C_int32 + n4);
        const float4 sb = *reinterpret_cast<const float4*>(d_scale_B + n4);

        float4 out;
        out.x = static_cast<float>(v.x) * row_scale * sb.x;
        out.y = static_cast<float>(v.y) * row_scale * sb.y;
        out.z = static_cast<float>(v.z) * row_scale * sb.z;
        out.w = static_cast<float>(v.w) * row_scale * sb.w;

        *reinterpret_cast<float4*>(d_C_fp32 + n4) = out;
        return;
    }

    for (int n = n4; n < N && n < n4 + 4; ++n)
    {
        d_C_fp32[n] = static_cast<float>(d_C_int32[n]) * row_scale * d_scale_B[n];
    }
}

__global__ void fillConstantFloat_kernel(float* __restrict__ dst, int count, float value)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        dst[idx] = value;
    }
}

enum PrefillInt8TileVariant
{
    PREFILL_TILE_16x16 = 0,
    PREFILL_TILE_32x8 = 1,
    PREFILL_TILE_8x32 = 2,
    PREFILL_TILE_8x8 = 3,
};

static int select_prefill_int8_tile_variant_auto(int M, int N, int K, bool grid_kpar)
{
    (void)K;
    if (M <= 12 && N <= 512)
    {
        return PREFILL_TILE_8x8;
    }

    if (N >= 4 * M && N >= 1024)
    {
        return PREFILL_TILE_32x8;
    }

    if (M >= 4 * N && M >= 128)
    {
        return PREFILL_TILE_8x32;
    }

    if (grid_kpar && N >= 2 * M && N >= 2048)
    {
        return PREFILL_TILE_32x8;
    }

    return PREFILL_TILE_16x16;
}

static int resolve_prefill_int8_tile_variant(int forced_variant, int M, int N, int K, bool grid_kpar)
{
    if (forced_variant >= PREFILL_TILE_16x16 && forced_variant <= PREFILL_TILE_8x8)
    {
        return forced_variant;
    }
    return select_prefill_int8_tile_variant_auto(M, N, K, grid_kpar);
}

template <int BLOCK_X, int BLOCK_Y, int CPT>
__global__ void qgemm_int8_int8_vnni_prefill_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K)
{
    const int n_base = blockIdx.x * (BLOCK_X * CPT) + threadIdx.x * CPT;
    const int m = blockIdx.y * BLOCK_Y + threadIdx.y;

    if (m >= M || n_base >= N)
    {
        return;
    }

    const int k_groups = K / 4;
    const bool full_tile = (n_base + CPT - 1) < N;
    const int64_t b_row_stride_bytes = static_cast<int64_t>(N) * 4;
    const int8_t* a_ptr0 = d_A_int8 + static_cast<int64_t>(m) * K;
    const int8_t* b_ptr0 = d_B_int8_vnni + static_cast<int64_t>(n_base) * 4;

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        acc[c] = 0;
    }

    if constexpr (CPT == 4)
    {
        if (full_tile)
        {
            const int8_t* a_ptr = a_ptr0;
            const int8_t* b_ptr = b_ptr0;
            for (int kg = 0; kg < k_groups; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                const uint4 b_pack4 = *reinterpret_cast<const uint4*>(b_ptr);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.x), acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.y), acc[1], false);
                acc[2] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.z), acc[2], false);
                acc[3] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.w), acc[3], false);
                a_ptr += 4;
                b_ptr += b_row_stride_bytes;
            }
        }
        else
        {
            const int valid = max(0, N - n_base);
            const int8_t* a_ptr = a_ptr0;
            const int8_t* b0 = b_ptr0;
            const int8_t* b1 = b_ptr0 + 4;
            const int8_t* b2 = b_ptr0 + 8;
            const int8_t* b3 = b_ptr0 + 12;
            for (int kg = 0; kg < k_groups; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                if (valid > 0)
                {
                    acc[0] = __builtin_amdgcn_sdot4(a_packed, *reinterpret_cast<const int32_t*>(b0), acc[0], false);
                }
                if (valid > 1)
                {
                    acc[1] = __builtin_amdgcn_sdot4(a_packed, *reinterpret_cast<const int32_t*>(b1), acc[1], false);
                }
                if (valid > 2)
                {
                    acc[2] = __builtin_amdgcn_sdot4(a_packed, *reinterpret_cast<const int32_t*>(b2), acc[2], false);
                }
                if (valid > 3)
                {
                    acc[3] = __builtin_amdgcn_sdot4(a_packed, *reinterpret_cast<const int32_t*>(b3), acc[3], false);
                }
                a_ptr += 4;
                b0 += b_row_stride_bytes;
                b1 += b_row_stride_bytes;
                b2 += b_row_stride_bytes;
                b3 += b_row_stride_bytes;
            }
        }
    }
    else if constexpr (CPT == 2)
    {
        if (full_tile)
        {
            const int8_t* a_ptr = a_ptr0;
            const int8_t* b_ptr = b_ptr0;
            for (int kg = 0; kg < k_groups; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                const uint2 b_pack2 = *reinterpret_cast<const uint2*>(b_ptr);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack2.x), acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack2.y), acc[1], false);
                a_ptr += 4;
                b_ptr += b_row_stride_bytes;
            }
        }
        else
        {
            const int valid = max(0, N - n_base);
            const int8_t* a_ptr = a_ptr0;
            const int8_t* b0 = b_ptr0;
            const int8_t* b1 = b_ptr0 + 4;
            for (int kg = 0; kg < k_groups; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                if (valid > 0)
                {
                    acc[0] = __builtin_amdgcn_sdot4(a_packed, *reinterpret_cast<const int32_t*>(b0), acc[0], false);
                }
                if (valid > 1)
                {
                    acc[1] = __builtin_amdgcn_sdot4(a_packed, *reinterpret_cast<const int32_t*>(b1), acc[1], false);
                }
                a_ptr += 4;
                b0 += b_row_stride_bytes;
                b1 += b_row_stride_bytes;
            }
        }
    }
    else
    {
        const int8_t* a_ptr = a_ptr0;
        const int8_t* b_ptr = b_ptr0;
        for (int kg = 0; kg < k_groups; ++kg)
        {
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
            const int32_t b_packed = *reinterpret_cast<const int32_t*>(b_ptr);
            acc[0] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[0], false);
            a_ptr += 4;
            b_ptr += b_row_stride_bytes;
        }
    }

    if constexpr (CPT == 4)
    {
        if ((n_base + 3) < N)
        {
            *reinterpret_cast<uint4*>(d_C_int32 + static_cast<int64_t>(m) * N + n_base) =
                make_uint4(static_cast<uint32_t>(acc[0]), static_cast<uint32_t>(acc[1]), static_cast<uint32_t>(acc[2]), static_cast<uint32_t>(acc[3]));
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int n = n_base + c;
                if (n < N)
                {
                    d_C_int32[static_cast<int64_t>(m) * N + n] = acc[c];
                }
            }
        }
    }
    else if constexpr (CPT == 2)
    {
        if ((n_base + 1) < N)
        {
            *reinterpret_cast<uint2*>(d_C_int32 + static_cast<int64_t>(m) * N + n_base) =
                make_uint2(static_cast<uint32_t>(acc[0]), static_cast<uint32_t>(acc[1]));
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int n = n_base + c;
                if (n < N)
                {
                    d_C_int32[static_cast<int64_t>(m) * N + n] = acc[c];
                }
            }
        }
    }
    else
    {
        d_C_int32[static_cast<int64_t>(m) * N + n_base] = acc[0];
    }
}

template <int BLOCK_X, int BLOCK_Y, int CPT, bool STAGE_B = false, bool ATOMIC_ACCUM = true, bool SOFTWARE_PIPELINED = false, bool LDS_B_TILE_CACHE = false>
__global__ void qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K,
    int k_groups_per_slice,
    int slice_index_base,
    int grid_swizzle_variant)
{
    const int n_tile_idx = (grid_swizzle_variant != 0) ? static_cast<int>(blockIdx.y) : static_cast<int>(blockIdx.x);
    const int m_tile_idx = (grid_swizzle_variant != 0) ? static_cast<int>(blockIdx.x) : static_cast<int>(blockIdx.y);
    const int n_base = n_tile_idx * (BLOCK_X * CPT) + threadIdx.x * CPT;
    const int m = m_tile_idx * BLOCK_Y + threadIdx.y;
    const int split_idx = blockIdx.z;

    if (m >= M || n_base >= N)
    {
        return;
    }

    const int total_k_groups = K / 4;
    const int k_begin = (split_idx + slice_index_base) * k_groups_per_slice;
    if (k_begin >= total_k_groups)
    {
        return;
    }

    const int k_end = min(total_k_groups, k_begin + k_groups_per_slice);
    const bool full_tile = (n_base + CPT - 1) < N;
    const int64_t b_row_stride_bytes = static_cast<int64_t>(N) * 4;
    const int8_t* a_ptr0 = d_A_int8 + static_cast<int64_t>(m) * K + static_cast<int64_t>(k_begin) * 4;
    const int8_t* b_ptr0 = d_B_int8_vnni + (static_cast<int64_t>(k_begin) * N + n_base) * 4;

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        acc[c] = 0;
    }

    if constexpr (CPT == 4 && LDS_B_TILE_CACHE)
    {
        constexpr int KT = 8;
        __shared__ int32_t b_lds[KT * 128];

        const int tid = threadIdx.y * BLOCK_X + threadIdx.x;
        const int n_block_base = n_tile_idx * (BLOCK_X * CPT);
        const int n_local = threadIdx.x * CPT;

        const int32_t* a_row = reinterpret_cast<const int32_t*>(
            d_A_int8 + static_cast<int64_t>(m) * K);
        const int32_t* b_global_base = reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + static_cast<int64_t>(n_block_base) * 4);

        for (int kt = k_begin; kt < k_end; kt += KT)
        {
            const int kg_limit = min(KT, k_end - kt);

            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int flat = i * 256 + tid;
                const int kg_l = flat >> 7;
                const int n_l = flat & 127;
                int32_t b_val = 0;
                if (kg_l < kg_limit)
                {
                    b_val = b_global_base[static_cast<int64_t>(kt + kg_l) * N + n_l];
                }
                b_lds[kg_l * 128 + n_l] = b_val;
            }
            __syncthreads();

            if (kg_limit > 0)
            {
                int32_t a_packed = a_row[kt];
                #pragma unroll
                for (int kg_l = 0; kg_l < KT - 1; ++kg_l)
                {
                    if (kg_l >= kg_limit - 1)
                    {
                        break;
                    }

                    const int32_t a_next = a_row[kt + kg_l + 1];
                    const int off = kg_l * 128 + n_local;
                    acc[0] = __builtin_amdgcn_sdot4(a_packed, b_lds[off + 0], acc[0], false);
                    acc[1] = __builtin_amdgcn_sdot4(a_packed, b_lds[off + 1], acc[1], false);
                    acc[2] = __builtin_amdgcn_sdot4(a_packed, b_lds[off + 2], acc[2], false);
                    acc[3] = __builtin_amdgcn_sdot4(a_packed, b_lds[off + 3], acc[3], false);
                    a_packed = a_next;
                }

                const int off_last = (kg_limit - 1) * 128 + n_local;
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b_lds[off_last + 0], acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b_lds[off_last + 1], acc[1], false);
                acc[2] = __builtin_amdgcn_sdot4(a_packed, b_lds[off_last + 2], acc[2], false);
                acc[3] = __builtin_amdgcn_sdot4(a_packed, b_lds[off_last + 3], acc[3], false);
            }

            __syncthreads();
        }
    }
    else if constexpr (STAGE_B)
    {
        __shared__ int32_t s_b_local[CPT][BLOCK_X];
        const int valid = max(0, N - n_base);
        const int lane_x = threadIdx.x;
        const int8_t* a_ptr = a_ptr0;

        if constexpr (CPT == 4)
        {
            const int8_t* b0 = b_ptr0;
            const int8_t* b1 = b_ptr0 + 4;
            const int8_t* b2 = b_ptr0 + 8;
            const int8_t* b3 = b_ptr0 + 12;

            for (int kg = k_begin; kg < k_end; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                if (threadIdx.y == 0)
                {
                    s_b_local[0][lane_x] = (valid > 0) ? *reinterpret_cast<const int32_t*>(b0) : 0;
                    s_b_local[1][lane_x] = (valid > 1) ? *reinterpret_cast<const int32_t*>(b1) : 0;
                    s_b_local[2][lane_x] = (valid > 2) ? *reinterpret_cast<const int32_t*>(b2) : 0;
                    s_b_local[3][lane_x] = (valid > 3) ? *reinterpret_cast<const int32_t*>(b3) : 0;
                    b0 += b_row_stride_bytes;
                    b1 += b_row_stride_bytes;
                    b2 += b_row_stride_bytes;
                    b3 += b_row_stride_bytes;
                }
                __syncthreads();

                acc[0] = __builtin_amdgcn_sdot4(a_packed, s_b_local[0][lane_x], acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, s_b_local[1][lane_x], acc[1], false);
                acc[2] = __builtin_amdgcn_sdot4(a_packed, s_b_local[2][lane_x], acc[2], false);
                acc[3] = __builtin_amdgcn_sdot4(a_packed, s_b_local[3][lane_x], acc[3], false);

                __syncthreads();
                a_ptr += 4;
            }
        }
        else if constexpr (CPT == 2)
        {
            const int8_t* b0 = b_ptr0;
            const int8_t* b1 = b_ptr0 + 4;

            for (int kg = k_begin; kg < k_end; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                if (threadIdx.y == 0)
                {
                    s_b_local[0][lane_x] = (valid > 0) ? *reinterpret_cast<const int32_t*>(b0) : 0;
                    s_b_local[1][lane_x] = (valid > 1) ? *reinterpret_cast<const int32_t*>(b1) : 0;
                    b0 += b_row_stride_bytes;
                    b1 += b_row_stride_bytes;
                }
                __syncthreads();

                acc[0] = __builtin_amdgcn_sdot4(a_packed, s_b_local[0][lane_x], acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, s_b_local[1][lane_x], acc[1], false);

                __syncthreads();
                a_ptr += 4;
            }
        }
        else
        {
            const int8_t* b0 = b_ptr0;
            for (int kg = k_begin; kg < k_end; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                if (threadIdx.y == 0)
                {
                    s_b_local[0][lane_x] = (valid > 0) ? *reinterpret_cast<const int32_t*>(b0) : 0;
                    b0 += b_row_stride_bytes;
                }
                __syncthreads();

                acc[0] = __builtin_amdgcn_sdot4(a_packed, s_b_local[0][lane_x], acc[0], false);

                __syncthreads();
                a_ptr += 4;
            }
        }
    }
    else if constexpr (CPT == 4)
    {
        if constexpr (SOFTWARE_PIPELINED)
        {
            if (full_tile)
            {
                const int8_t* a_ptr = a_ptr0;
                const int32_t* b_ptr = reinterpret_cast<const int32_t*>(b_ptr0);
                const int b_row_stride_i32 = N;
                const int k_len = k_end - k_begin;

                if (k_len > 0)
                {
                    int32_t a_cur = *reinterpret_cast<const int32_t*>(a_ptr);
                    int32_t b0 = b_ptr[0];
                    int32_t b1 = b_ptr[1];
                    int32_t b2 = b_ptr[2];
                    int32_t b3 = b_ptr[3];
                    a_ptr += 4;
                    b_ptr += b_row_stride_i32;

                    for (int i = 1; i < k_len; ++i)
                    {
                        const int32_t a_nxt = *reinterpret_cast<const int32_t*>(a_ptr);
                        const int32_t bn0 = b_ptr[0];
                        const int32_t bn1 = b_ptr[1];
                        const int32_t bn2 = b_ptr[2];
                        const int32_t bn3 = b_ptr[3];

                        acc[0] = __builtin_amdgcn_sdot4(a_cur, b0, acc[0], false);
                        acc[1] = __builtin_amdgcn_sdot4(a_cur, b1, acc[1], false);
                        acc[2] = __builtin_amdgcn_sdot4(a_cur, b2, acc[2], false);
                        acc[3] = __builtin_amdgcn_sdot4(a_cur, b3, acc[3], false);

                        a_cur = a_nxt;
                        b0 = bn0;
                        b1 = bn1;
                        b2 = bn2;
                        b3 = bn3;
                        a_ptr += 4;
                        b_ptr += b_row_stride_i32;
                    }

                    acc[0] = __builtin_amdgcn_sdot4(a_cur, b0, acc[0], false);
                    acc[1] = __builtin_amdgcn_sdot4(a_cur, b1, acc[1], false);
                    acc[2] = __builtin_amdgcn_sdot4(a_cur, b2, acc[2], false);
                    acc[3] = __builtin_amdgcn_sdot4(a_cur, b3, acc[3], false);
                }
            }
            else
            {
                const int valid = max(0, N - n_base);
                const int8_t* a_ptr = a_ptr0;
                const int32_t* b0 = reinterpret_cast<const int32_t*>(b_ptr0);
                const int32_t* b1 = b0 + 1;
                const int32_t* b2 = b0 + 2;
                const int32_t* b3 = b0 + 3;
                const int b_row_stride_i32 = N;
                for (int kg = k_begin; kg < k_end; ++kg)
                {
                    const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                    if (valid > 0)
                    {
                        acc[0] = __builtin_amdgcn_sdot4(a_packed, *b0, acc[0], false);
                    }
                    if (valid > 1)
                    {
                        acc[1] = __builtin_amdgcn_sdot4(a_packed, *b1, acc[1], false);
                    }
                    if (valid > 2)
                    {
                        acc[2] = __builtin_amdgcn_sdot4(a_packed, *b2, acc[2], false);
                    }
                    if (valid > 3)
                    {
                        acc[3] = __builtin_amdgcn_sdot4(a_packed, *b3, acc[3], false);
                    }
                    a_ptr += 4;
                    b0 += b_row_stride_i32;
                    b1 += b_row_stride_i32;
                    b2 += b_row_stride_i32;
                    b3 += b_row_stride_i32;
                }
            }
        }
        else
        {
            if (full_tile)
            {
                const int8_t* a_ptr = a_ptr0;
                const int32_t* b_ptr = reinterpret_cast<const int32_t*>(b_ptr0);
                const int b_row_stride_i32 = N;
                for (int kg = k_begin; kg < k_end; ++kg)
                {
                    const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                    acc[0] = __builtin_amdgcn_sdot4(a_packed, b_ptr[0], acc[0], false);
                    acc[1] = __builtin_amdgcn_sdot4(a_packed, b_ptr[1], acc[1], false);
                    acc[2] = __builtin_amdgcn_sdot4(a_packed, b_ptr[2], acc[2], false);
                    acc[3] = __builtin_amdgcn_sdot4(a_packed, b_ptr[3], acc[3], false);
                    a_ptr += 4;
                    b_ptr += b_row_stride_i32;
                }
            }
            else
            {
                const int valid = max(0, N - n_base);
                const int8_t* a_ptr = a_ptr0;
                const int32_t* b0 = reinterpret_cast<const int32_t*>(b_ptr0);
                const int32_t* b1 = b0 + 1;
                const int32_t* b2 = b0 + 2;
                const int32_t* b3 = b0 + 3;
                const int b_row_stride_i32 = N;
                for (int kg = k_begin; kg < k_end; ++kg)
                {
                    const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                    if (valid > 0)
                    {
                        acc[0] = __builtin_amdgcn_sdot4(a_packed, *b0, acc[0], false);
                    }
                    if (valid > 1)
                    {
                        acc[1] = __builtin_amdgcn_sdot4(a_packed, *b1, acc[1], false);
                    }
                    if (valid > 2)
                    {
                        acc[2] = __builtin_amdgcn_sdot4(a_packed, *b2, acc[2], false);
                    }
                    if (valid > 3)
                    {
                        acc[3] = __builtin_amdgcn_sdot4(a_packed, *b3, acc[3], false);
                    }
                    a_ptr += 4;
                    b0 += b_row_stride_i32;
                    b1 += b_row_stride_i32;
                    b2 += b_row_stride_i32;
                    b3 += b_row_stride_i32;
                }
            }
        }
    }
    else if constexpr (CPT == 2)
    {
        if (full_tile)
        {
            const int8_t* a_ptr = a_ptr0;
            const int32_t* b_ptr = reinterpret_cast<const int32_t*>(b_ptr0);
            const int b_row_stride_i32 = N;
            for (int kg = k_begin; kg < k_end; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b_ptr[0], acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b_ptr[1], acc[1], false);
                a_ptr += 4;
                b_ptr += b_row_stride_i32;
            }
        }
        else
        {
            const int valid = max(0, N - n_base);
            const int8_t* a_ptr = a_ptr0;
            const int32_t* b0 = reinterpret_cast<const int32_t*>(b_ptr0);
            const int32_t* b1 = b0 + 1;
            const int b_row_stride_i32 = N;
            for (int kg = k_begin; kg < k_end; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                if (valid > 0)
                {
                    acc[0] = __builtin_amdgcn_sdot4(a_packed, *b0, acc[0], false);
                }
                if (valid > 1)
                {
                    acc[1] = __builtin_amdgcn_sdot4(a_packed, *b1, acc[1], false);
                }
                a_ptr += 4;
                b0 += b_row_stride_i32;
                b1 += b_row_stride_i32;
            }
        }
    }
    else
    {
        const int8_t* a_ptr = a_ptr0;
        const int8_t* b_ptr = b_ptr0;
        for (int kg = k_begin; kg < k_end; ++kg)
        {
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
            const int32_t b_packed = *reinterpret_cast<const int32_t*>(b_ptr);
            acc[0] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[0], false);
            a_ptr += 4;
            b_ptr += b_row_stride_bytes;
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N)
        {
            if constexpr (ATOMIC_ACCUM)
            {
                atomicAdd(&d_C_int32[static_cast<int64_t>(m) * N + n], acc[c]);
            }
            else
            {
                d_C_int32[static_cast<int64_t>(m) * N + n] = acc[c];
            }
        }
    }
}

// ============================================================================
// WIDE-TILE KERNEL: Full-M-coverage tiling for extreme-wide shapes (LM_Head)
// ============================================================================
//
// Design rationale:
//   For extreme-wide shapes (N >> K, M ≤ 128), the standard 32×8 tile covers
//   only 8 M-rows per block, requiring 16 M-tile passes.  Each pass re-reads
//   the same B columns from HBM because 311 MB far exceeds the 4 MB L2.
//   This kernel covers ALL M rows (up to 128) in a single block, loading B
//   into LDS once and reusing it across every M-row.
//
// Tile geometry:
//   M_TILE = 128 (full M coverage), N_TILE = 64, KT = 8 k-groups per LDS phase
//   256 threads = 32 (M-dim) × 8 (N-dim)
//   Each thread: 4 M-rows × 8 N-cols = 32 accumulators
//   Grid: (ceil(N/64), 1) — single M pass, B streamed through LDS exactly once
//
// Software-pipelined single-buffer LDS:
//   Global loads for the NEXT K-tile are issued into VGPRs at the top of the
//   loop body, overlapping with compute on the CURRENT tile from LDS.  After
//   compute completes (barrier), the prefetched data is written to LDS for
//   the next iteration.  This gives load/compute overlap WITHOUT needing
//   double-buffered LDS — the prefetch lives in ≤8 VGPRs during compute.
//
// LDS budget (single-buffer):
//   A tile: 128 × KT = 1024 × 4 = 4 KB
//   B tile: KT × N_TILE = 512 × 4 = 2 KB
//   Total: 6 KB — leaves room for 3+ waves
//
// Register budget:
//   32 accumulators + 4 A-regs + 8 B-regs + ~20 addr/temp
//   ≈ 66 VGPRs → floor(256/66) = 3 waves/SIMD
//
// ============================================================================

template <int N_TILE, int KT, int MIN_BLOCKS = 0>
__global__
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void __launch_bounds__(256, MIN_BLOCKS)
qgemm_int8_vnni_wide_tile_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K)
{
    // Thread layout: 256 threads = 32 (M-dim) × 8 (N-dim)
    constexpr int BLOCK_SIZE = 256;
    constexpr int THREADS_M = 32;
    constexpr int THREADS_N = BLOCK_SIZE / THREADS_M; // 8
    constexpr int M_PER_THREAD = 4;                   // 32 × 4 = 128 M-rows
    constexpr int N_PER_THREAD = N_TILE / THREADS_N;  // 64 / 8 = 8 N-cols
    constexpr int M_TILE = THREADS_M * M_PER_THREAD;  // 128

    static_assert(THREADS_M * THREADS_N == BLOCK_SIZE);
    static_assert(N_TILE % THREADS_N == 0);

    const int n_block = blockIdx.x * N_TILE;
    const int t_m = threadIdx.x % THREADS_M; // 0..31
    const int t_n = threadIdx.x / THREADS_M; // 0..7

    const int m_base = t_m * M_PER_THREAD;
    const int n_base = n_block + t_n * N_PER_THREAD;

    // Single-buffered LDS
    __shared__ int32_t a_lds[KT * M_TILE]; // 4 KB
    __shared__ int32_t b_lds[KT * N_TILE]; // 2 KB

    // 32 accumulators per thread
    int32_t acc[M_PER_THREAD][N_PER_THREAD];
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
            acc[mm][nn] = 0;

    const int total_k_groups = K / 4;

    // Reinterpret pointers for int32 (packed 4×int8) access
    const int32_t* a_global = reinterpret_cast<const int32_t*>(d_A_int8);
    const int32_t* b_global = reinterpret_cast<const int32_t*>(d_B_int8_vnni);

    const int a_row_stride = K / 4; // k-groups per A row

    // === Cooperative load constants (vectorized 128-bit loads) ===
    static_assert(KT % 4 == 0, "KT must be multiple of 4 for vectorized A loads");
    static_assert(N_TILE % 4 == 0, "N_TILE must be multiple of 4 for vectorized B loads");
    constexpr int A_K_CHUNKS = KT / 4;                         // 2
    constexpr int B_N_CHUNKS = N_TILE / 4;                      // 16
    constexpr int B_ACTIVE_THREADS = KT * B_N_CHUNKS;           // 128

    // =========================================================================
    // SINGLE-BUFFER PIPELINE with BATCHED GLOBAL LOADS
    //
    // Both A and B global loads are issued before any LDS writes, allowing
    // the memory pipeline to process them in parallel (different HBM banks).
    // This saves one load latency (~300 ns) per iteration vs serial loads.
    // Cost: +4 VGPRs (B staging lives alongside A staging).
    // =========================================================================

    // Pre-compute per-thread load coordinates (loop-invariant)
    const int a_mi = threadIdx.x / A_K_CHUNKS;
    const int a_kg = (threadIdx.x % A_K_CHUNKS) * 4;
    const int b_kk_local = threadIdx.x / B_N_CHUNKS;
    const int b_nl = (threadIdx.x % B_N_CHUNKS) * 4;
    const int b_gn = n_block + b_nl;
    const int b_lds_off = b_kk_local * N_TILE + b_nl;

    for (int kt_base = 0; kt_base < total_k_groups; kt_base += KT)
    {
        // === Stage 1: Issue BOTH global loads into staging registers ===
        // A load: all 256 threads participate
        uint4 a4;
        const int a_gkg = kt_base + a_kg;
        if (__builtin_expect(a_mi < M && a_gkg + 3 < total_k_groups, 1))
        {
            a4 = *reinterpret_cast<const uint4*>(
                a_global + static_cast<int64_t>(a_mi) * a_row_stride + a_gkg);
        }
        else
        {
            a4.x = (a_mi < M && a_gkg + 0 < total_k_groups)
                ? a_global[static_cast<int64_t>(a_mi) * a_row_stride + a_gkg + 0] : 0;
            a4.y = (a_mi < M && a_gkg + 1 < total_k_groups)
                ? a_global[static_cast<int64_t>(a_mi) * a_row_stride + a_gkg + 1] : 0;
            a4.z = (a_mi < M && a_gkg + 2 < total_k_groups)
                ? a_global[static_cast<int64_t>(a_mi) * a_row_stride + a_gkg + 2] : 0;
            a4.w = (a_mi < M && a_gkg + 3 < total_k_groups)
                ? a_global[static_cast<int64_t>(a_mi) * a_row_stride + a_gkg + 3] : 0;
        }

        // B load: first B_ACTIVE_THREADS threads participate
        uint4 b4 = make_uint4(0, 0, 0, 0);
        if (threadIdx.x < B_ACTIVE_THREADS)
        {
            const int b_gkk = kt_base + b_kk_local;
            if (__builtin_expect(b_gn + 3 < N && b_gkk < total_k_groups, 1))
            {
                b4 = *reinterpret_cast<const uint4*>(
                    b_global + static_cast<int64_t>(b_gkk) * N + b_gn);
            }
            else
            {
                b4.x = (b_gn + 0 < N && b_gkk < total_k_groups)
                    ? b_global[static_cast<int64_t>(b_gkk) * N + b_gn + 0] : 0;
                b4.y = (b_gn + 1 < N && b_gkk < total_k_groups)
                    ? b_global[static_cast<int64_t>(b_gkk) * N + b_gn + 1] : 0;
                b4.z = (b_gn + 2 < N && b_gkk < total_k_groups)
                    ? b_global[static_cast<int64_t>(b_gkk) * N + b_gn + 2] : 0;
                b4.w = (b_gn + 3 < N && b_gkk < total_k_groups)
                    ? b_global[static_cast<int64_t>(b_gkk) * N + b_gn + 3] : 0;
            }
        }

        // === Stage 2: Write staged data → LDS (s_waitcnt vmcnt(0) here) ===
        a_lds[(a_kg + 0) * M_TILE + a_mi] = a4.x;
        a_lds[(a_kg + 1) * M_TILE + a_mi] = a4.y;
        a_lds[(a_kg + 2) * M_TILE + a_mi] = a4.z;
        a_lds[(a_kg + 3) * M_TILE + a_mi] = a4.w;

        if (threadIdx.x < B_ACTIVE_THREADS)
        {
            *reinterpret_cast<uint4*>(&b_lds[b_lds_off]) = b4;
        }

        __syncthreads();

        // === Compute: fully unrolled KT × (4×8) v_dot4 ===
        #pragma unroll
        for (int kk = 0; kk < KT; ++kk)
        {
            int32_t a_reg[M_PER_THREAD];
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                a_reg[mm] = a_lds[kk * M_TILE + m_base + mm];

            int32_t b_reg[N_PER_THREAD];
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
                b_reg[nn] = b_lds[kk * N_TILE + t_n * N_PER_THREAD + nn];

            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    acc[mm][nn] = __builtin_amdgcn_sdot4(a_reg[mm], b_reg[nn],
                                                          acc[mm][nn], false);
        }

        __syncthreads();
    }

    // === Store results ===
    // Each thread writes 4 M-rows × 8 N-cols = 32 elements
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
    {
        const int m_idx = m_base + mm;
        if (m_idx < M)
        {
            // Try vectorized uint4 stores (4 consecutive int32s per store)
            if (n_base + N_PER_THREAD - 1 < N)
            {
                // All N_PER_THREAD columns are valid — write vectorized
                const int64_t row_off = static_cast<int64_t>(m_idx) * N + n_base;
                #pragma unroll
                for (int chunk = 0; chunk < N_PER_THREAD; chunk += 4)
                {
                    *reinterpret_cast<uint4*>(d_C_int32 + row_off + chunk) =
                        make_uint4(static_cast<uint32_t>(acc[mm][chunk + 0]),
                                   static_cast<uint32_t>(acc[mm][chunk + 1]),
                                   static_cast<uint32_t>(acc[mm][chunk + 2]),
                                   static_cast<uint32_t>(acc[mm][chunk + 3]));
                }
            }
            else
            {
                // Tail N-block: scalar stores with bounds checking
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                {
                    const int n_idx = n_base + nn;
                    if (n_idx < N)
                    {
                        d_C_int32[static_cast<int64_t>(m_idx) * N + n_idx] = acc[mm][nn];
                    }
                }
            }
        }
    }
}

// ============================================================================
// WIDE-TILE V2: A from L2, B in LDS — eliminates A LDS bank conflicts
// ============================================================================
//
// Key difference from V1: A is read directly from L2 cache instead of LDS.
// A is only 128×K = 256 KB for K=2048, fitting entirely in L2 after first access.
// Benefits:
//   - Removes 4 KB A LDS allocation → more LDS for other waves
//   - Eliminates A bank conflicts (4-way conflict in V1 due to stride-4 access)
//   - Removes A cooperative load phase (faster setup)
//   - L1 handles broadcast for threads sharing same m_base (8 t_n values)
//
// ============================================================================

template <int N_TILE, int KT>
__global__ void qgemm_int8_vnni_wide_tile_v2_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K)
{
    constexpr int BLOCK_SIZE = 256;
    constexpr int THREADS_M = 32;
    constexpr int THREADS_N = BLOCK_SIZE / THREADS_M; // 8
    constexpr int M_PER_THREAD = 4;
    constexpr int N_PER_THREAD = N_TILE / THREADS_N;
    constexpr int M_TILE = THREADS_M * M_PER_THREAD; // 128

    static_assert(THREADS_M * THREADS_N == BLOCK_SIZE);
    static_assert(N_TILE % THREADS_N == 0);

    const int n_block = blockIdx.x * N_TILE;
    const int t_m = threadIdx.x % THREADS_M;
    const int t_n = threadIdx.x / THREADS_M;
    const int m_base = t_m * M_PER_THREAD;
    const int n_base = n_block + t_n * N_PER_THREAD;

    // LDS: only B tile
    __shared__ int32_t b_lds[KT * N_TILE]; // 8 × 64 = 512 × 4 = 2 KB

    int32_t acc[M_PER_THREAD][N_PER_THREAD];
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
            acc[mm][nn] = 0;

    const int total_k_groups = K / 4;
    const int32_t* a_global = reinterpret_cast<const int32_t*>(d_A_int8);
    const int32_t* b_global = reinterpret_cast<const int32_t*>(d_B_int8_vnni);
    const int a_row_stride = K / 4;

    // Pre-compute A row pointers for each owned M-row
    const int32_t* a_rows[M_PER_THREAD];
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
    {
        const int m_idx = m_base + mm;
        a_rows[mm] = (m_idx < M) ? (a_global + static_cast<int64_t>(m_idx) * a_row_stride) : nullptr;
    }

    for (int kt_base = 0; kt_base < total_k_groups; kt_base += KT)
    {
        const int kt_remain = min(KT, total_k_groups - kt_base);

        // === Cooperative load B only: KT × N_TILE elements ===
        constexpr int B_TOTAL = KT * N_TILE;
        constexpr int B_LOADS_PER_THREAD = (B_TOTAL + BLOCK_SIZE - 1) / BLOCK_SIZE;
        #pragma unroll
        for (int li = 0; li < B_LOADS_PER_THREAD; ++li)
        {
            const int flat = li * BLOCK_SIZE + threadIdx.x;
            if (flat < B_TOTAL)
            {
                const int kk = flat / N_TILE;
                const int n_local = flat % N_TILE;
                const int global_n = n_block + n_local;
                int32_t val = 0;
                if (kk < kt_remain && global_n < N)
                {
                    val = b_global[static_cast<int64_t>(kt_base + kk) * N + global_n];
                }
                b_lds[kk * N_TILE + n_local] = val;
            }
        }

        __syncthreads();

        // === Compute: A from L2 (via L1 broadcast), B from LDS ===
        #pragma unroll
        for (int kk = 0; kk < KT; ++kk)
        {
            if (kk >= kt_remain) break;

            // Load A values directly from global memory (L2-cached)
            // All 8 t_n threads with same t_m read same addresses → L1 broadcast
            int32_t a_reg[M_PER_THREAD];
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
            {
                a_reg[mm] = a_rows[mm] ? a_rows[mm][kt_base + kk] : 0;
            }

            // Load B values from LDS
            int32_t b_reg[N_PER_THREAD];
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
            {
                b_reg[nn] = b_lds[kk * N_TILE + t_n * N_PER_THREAD + nn];
            }

            // 4×8 = 32 sdot4 operations
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
            {
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                {
                    acc[mm][nn] = __builtin_amdgcn_sdot4(
                        a_reg[mm], b_reg[nn], acc[mm][nn], false);
                }
            }
        }

        __syncthreads();
    }

    // === Store results ===
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
    {
        const int m_idx = m_base + mm;
        if (m_idx < M)
        {
            if (n_base + N_PER_THREAD - 1 < N)
            {
                const int64_t row_off = static_cast<int64_t>(m_idx) * N + n_base;
                #pragma unroll
                for (int chunk = 0; chunk < N_PER_THREAD; chunk += 4)
                {
                    *reinterpret_cast<uint4*>(d_C_int32 + row_off + chunk) =
                        make_uint4(static_cast<uint32_t>(acc[mm][chunk + 0]),
                                   static_cast<uint32_t>(acc[mm][chunk + 1]),
                                   static_cast<uint32_t>(acc[mm][chunk + 2]),
                                   static_cast<uint32_t>(acc[mm][chunk + 3]));
                }
            }
            else
            {
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                {
                    const int n_idx = n_base + nn;
                    if (n_idx < N)
                    {
                        d_C_int32[static_cast<int64_t>(m_idx) * N + n_idx] = acc[mm][nn];
                    }
                }
            }
        }
    }
}

// ============================================================================
// WIDE-TILE V3: LDS Double-Buffered Pipeline
// ============================================================================
//
// Key improvements over V1 (based on CK ISA analysis — see
// changelog/2025-07-24-ck-vs-v1-isa-analysis.md):
//
//   1. LDS double-buffering: overlap global→LDS loads of tile N+1 with
//      compute on tile N.  CK ISA shows this is the primary technique
//      behind its 14% advantage.  Expected gain: ~5-8%.
//
//   2. KT=16 support with multi-pass A cooperative load: V1's single-pass
//      A load (256 threads / (KT/4) chunks) only covers 128 M-rows when
//      KT=8.  With KT=16, A_K_CHUNKS=4 → 256/4=64 rows/pass → need 2
//      passes.  Halves loop iterations vs KT=8.  Expected gain: ~2-3%.
//
//   3. Software-pipelined main loop:
//        preload tile 0 → buf[0]
//        for tile 1..N-1:
//            issue global loads for tile t      (async, into staging regs)
//            compute tile t-1 from buf[(t-1)%2] (overlaps with loads!)
//            __syncthreads()
//            write staging → buf[t%2]            (vmcnt(0) ensures loads done)
//            __syncthreads()
//        compute last tile
//
// LDS Budget (KT=16, N_TILE=64):
//   a_lds[2][16 × 128] × 4 bytes = 16 KB
//   b_lds[2][16 × 64]  × 4 bytes =  8 KB
//   Total: 24 KB → 2 workgroups/CU (48 KB < 64 KB LDS)
//
// LDS Budget (KT=8, N_TILE=64):
//   a_lds[2][8 × 128] × 4 bytes = 8 KB
//   b_lds[2][8 × 64]  × 4 bytes = 4 KB
//   Total: 12 KB → 5 workgroups/CU
//
// Register Budget (~66 VGPRs for KT=16):
//   32 accumulators + 8 a_staging (2 × uint4) + 4 b_staging (1 × uint4)
//   + 12 operands (transient) + ~10 misc = ~66 VGPRs → 3 waves/SIMD
//
// ============================================================================

template <int N_TILE, int KT>
__global__ __launch_bounds__(256, 0)
void qgemm_int8_vnni_wide_tile_v3_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K)
{
    // Thread layout: 256 threads = 32 (M-dim) × 8 (N-dim)
    constexpr int BLOCK_SIZE = 256;
    constexpr int THREADS_M = 32;
    constexpr int THREADS_N = BLOCK_SIZE / THREADS_M; // 8
    constexpr int M_PER_THREAD = 4;                   // 32 × 4 = 128 M-rows
    constexpr int N_PER_THREAD = N_TILE / THREADS_N;  // 64 / 8 = 8 N-cols
    constexpr int M_TILE = THREADS_M * M_PER_THREAD;  // 128

    static_assert(THREADS_M * THREADS_N == BLOCK_SIZE);
    static_assert(N_TILE % THREADS_N == 0);
    static_assert(KT % 4 == 0, "KT must be multiple of 4 for vectorized A loads");
    static_assert(N_TILE % 4 == 0, "N_TILE must be multiple of 4 for vectorized B loads");

    const int n_block = blockIdx.x * N_TILE;
    const int t_m = threadIdx.x % THREADS_M; // 0..31
    const int t_n = threadIdx.x / THREADS_M; // 0..7

    const int m_base = t_m * M_PER_THREAD;
    const int n_base = n_block + t_n * N_PER_THREAD;

    // =========================================================================
    // LDS: Double-buffered (ping-pong between buf[0] and buf[1])
    // =========================================================================
    __shared__ int32_t a_lds[2][KT * M_TILE];
    __shared__ int32_t b_lds[2][KT * N_TILE];

    // Accumulators: 4 × 8 = 32 per thread
    int32_t acc[M_PER_THREAD][N_PER_THREAD];
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
            acc[mm][nn] = 0;

    const int total_k_groups = K / 4;

    // Reinterpret pointers for int32 (packed 4×int8) access
    const int32_t* a_global = reinterpret_cast<const int32_t*>(d_A_int8);
    const int32_t* b_global = reinterpret_cast<const int32_t*>(d_B_int8_vnni);
    const int a_row_stride = K / 4; // k-groups per A row

    // =========================================================================
    // A cooperative load geometry — multi-pass for KT ≥ 16
    //
    // With 256 threads and A_K_CHUNKS = KT/4 vec4-groups per row:
    //   KT=8  → A_K_CHUNKS=2, 256/2=128 rows/pass → 1 pass  (covers all 128)
    //   KT=16 → A_K_CHUNKS=4, 256/4=64  rows/pass → 2 passes (64+64=128)
    // =========================================================================
    constexpr int A_K_CHUNKS = KT / 4;
    constexpr int A_ROWS_PER_PASS = BLOCK_SIZE / A_K_CHUNKS;
    constexpr int A_NUM_PASSES = (M_TILE + A_ROWS_PER_PASS - 1) / A_ROWS_PER_PASS;

    // Per-thread A load coordinates (loop-invariant)
    const int a_kg = (threadIdx.x % A_K_CHUNKS) * 4;
    const int a_thread_mi_base = threadIdx.x / A_K_CHUNKS;

    // Pre-compute M-row index for each A pass
    int a_mi[A_NUM_PASSES];
    #pragma unroll
    for (int p = 0; p < A_NUM_PASSES; ++p)
        a_mi[p] = a_thread_mi_base + p * A_ROWS_PER_PASS;

    // =========================================================================
    // B cooperative load geometry
    //   KT=8,  N_TILE=64 → B_ACTIVE_THREADS=128 (first half of block)
    //   KT=16, N_TILE=64 → B_ACTIVE_THREADS=256 (all threads participate)
    // =========================================================================
    constexpr int B_N_CHUNKS = N_TILE / 4;
    constexpr int B_ACTIVE_THREADS = KT * B_N_CHUNKS;
    static_assert(B_ACTIVE_THREADS <= BLOCK_SIZE,
                  "B tile too large for single-pass cooperative load");

    const int b_kk_local = threadIdx.x / B_N_CHUNKS;
    const int b_nl = (threadIdx.x % B_N_CHUNKS) * 4;
    const int b_gn = n_block + b_nl;
    const int b_lds_off = b_kk_local * N_TILE + b_nl;

    const int num_tiles = (total_k_groups + KT - 1) / KT;
    if (num_tiles == 0) return;

    // Staging registers for double-buffered pipeline
    uint4 a_staging[A_NUM_PASSES];
    uint4 b_staging;

    // =========================================================================
    // PRELOAD: Load first tile (tile 0) into buf[0]
    // =========================================================================

    // A loads — multi-pass
    #pragma unroll
    for (int p = 0; p < A_NUM_PASSES; ++p)
    {
        const int mi = a_mi[p];
        const int gkg = a_kg; // kt_base = 0
        if (__builtin_expect(mi < M && gkg + 3 < total_k_groups, 1))
        {
            a_staging[p] = *reinterpret_cast<const uint4*>(
                a_global + static_cast<int64_t>(mi) * a_row_stride + gkg);
        }
        else
        {
            a_staging[p].x = (mi < M && gkg + 0 < total_k_groups)
                ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 0] : 0;
            a_staging[p].y = (mi < M && gkg + 1 < total_k_groups)
                ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 1] : 0;
            a_staging[p].z = (mi < M && gkg + 2 < total_k_groups)
                ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 2] : 0;
            a_staging[p].w = (mi < M && gkg + 3 < total_k_groups)
                ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 3] : 0;
        }
    }

    // B load
    b_staging = make_uint4(0, 0, 0, 0);
    if (threadIdx.x < B_ACTIVE_THREADS)
    {
        const int gkk = b_kk_local; // kt_base = 0
        if (__builtin_expect(b_gn + 3 < N && gkk < total_k_groups, 1))
        {
            b_staging = *reinterpret_cast<const uint4*>(
                b_global + static_cast<int64_t>(gkk) * N + b_gn);
        }
        else
        {
            b_staging.x = (b_gn + 0 < N && gkk < total_k_groups)
                ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 0] : 0;
            b_staging.y = (b_gn + 1 < N && gkk < total_k_groups)
                ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 1] : 0;
            b_staging.z = (b_gn + 2 < N && gkk < total_k_groups)
                ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 2] : 0;
            b_staging.w = (b_gn + 3 < N && gkk < total_k_groups)
                ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 3] : 0;
        }
    }

    // Write staged data to buf[0]
    #pragma unroll
    for (int p = 0; p < A_NUM_PASSES; ++p)
    {
        const int mi = a_mi[p];
        a_lds[0][(a_kg + 0) * M_TILE + mi] = a_staging[p].x;
        a_lds[0][(a_kg + 1) * M_TILE + mi] = a_staging[p].y;
        a_lds[0][(a_kg + 2) * M_TILE + mi] = a_staging[p].z;
        a_lds[0][(a_kg + 3) * M_TILE + mi] = a_staging[p].w;
    }
    if (threadIdx.x < B_ACTIVE_THREADS)
    {
        *reinterpret_cast<uint4*>(&b_lds[0][b_lds_off]) = b_staging;
    }

    __syncthreads(); // buf[0] ready for compute

    // =========================================================================
    // MAIN LOOP: Double-buffered pipeline
    //
    // For each tile t (1 .. num_tiles-1):
    //   1. Issue global loads for tile t      → staging regs  (async)
    //   2. Compute tile (t-1) from buf[(t-1)%2]               (overlaps!)
    //   3. __syncthreads()                    (all reads from cur_buf done)
    //   4. Write staging → buf[t%2]           (vmcnt(0) ensures loads done)
    //   5. __syncthreads()                    (LDS writes visible)
    // =========================================================================
    for (int tile = 1; tile < num_tiles; ++tile)
    {
        const int cur_buf = (tile - 1) & 1; // buffer to compute from
        const int nxt_buf = tile & 1;       // buffer to load into
        const int kt_base = tile * KT;

        // --- 1. Issue ALL global loads for next tile (async) ---

        // A loads — multi-pass
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            const int gkg = kt_base + a_kg;
            if (__builtin_expect(mi < M && gkg + 3 < total_k_groups, 1))
            {
                a_staging[p] = *reinterpret_cast<const uint4*>(
                    a_global + static_cast<int64_t>(mi) * a_row_stride + gkg);
            }
            else
            {
                a_staging[p].x = (mi < M && gkg + 0 < total_k_groups)
                    ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 0] : 0;
                a_staging[p].y = (mi < M && gkg + 1 < total_k_groups)
                    ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 1] : 0;
                a_staging[p].z = (mi < M && gkg + 2 < total_k_groups)
                    ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 2] : 0;
                a_staging[p].w = (mi < M && gkg + 3 < total_k_groups)
                    ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 3] : 0;
            }
        }

        // B load
        if (threadIdx.x < B_ACTIVE_THREADS)
        {
            const int gkk = kt_base + b_kk_local;
            if (__builtin_expect(b_gn + 3 < N && gkk < total_k_groups, 1))
            {
                b_staging = *reinterpret_cast<const uint4*>(
                    b_global + static_cast<int64_t>(gkk) * N + b_gn);
            }
            else
            {
                b_staging = make_uint4(0, 0, 0, 0);
                b_staging.x = (b_gn + 0 < N && gkk < total_k_groups)
                    ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 0] : 0;
                b_staging.y = (b_gn + 1 < N && gkk < total_k_groups)
                    ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 1] : 0;
                b_staging.z = (b_gn + 2 < N && gkk < total_k_groups)
                    ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 2] : 0;
                b_staging.w = (b_gn + 3 < N && gkk < total_k_groups)
                    ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 3] : 0;
            }
        }

        // --- 2. Compute on CURRENT buffer (overlaps with global loads!) ---
        #pragma unroll
        for (int kk = 0; kk < KT; ++kk)
        {
            int32_t a_reg[M_PER_THREAD];
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                a_reg[mm] = a_lds[cur_buf][kk * M_TILE + m_base + mm];

            int32_t b_reg[N_PER_THREAD];
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
                b_reg[nn] = b_lds[cur_buf][kk * N_TILE + t_n * N_PER_THREAD + nn];

            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    acc[mm][nn] = __builtin_amdgcn_sdot4(
                        a_reg[mm], b_reg[nn], acc[mm][nn], false);
        }

        // --- 3. Barrier: all threads done reading cur_buf ---
        __syncthreads();

        // --- 4. Write staged data to NEXT buffer ---
        // (compiler inserts s_waitcnt vmcnt(0) here for the global loads)
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            a_lds[nxt_buf][(a_kg + 0) * M_TILE + mi] = a_staging[p].x;
            a_lds[nxt_buf][(a_kg + 1) * M_TILE + mi] = a_staging[p].y;
            a_lds[nxt_buf][(a_kg + 2) * M_TILE + mi] = a_staging[p].z;
            a_lds[nxt_buf][(a_kg + 3) * M_TILE + mi] = a_staging[p].w;
        }
        if (threadIdx.x < B_ACTIVE_THREADS)
        {
            *reinterpret_cast<uint4*>(&b_lds[nxt_buf][b_lds_off]) = b_staging;
        }

        // --- 5. Barrier: LDS writes visible before next iteration ---
        __syncthreads();
    }

    // =========================================================================
    // FINAL TILE: Compute the last tile (already loaded into LDS)
    // =========================================================================
    {
        const int last_buf = (num_tiles - 1) & 1;

        #pragma unroll
        for (int kk = 0; kk < KT; ++kk)
        {
            int32_t a_reg[M_PER_THREAD];
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                a_reg[mm] = a_lds[last_buf][kk * M_TILE + m_base + mm];

            int32_t b_reg[N_PER_THREAD];
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
                b_reg[nn] = b_lds[last_buf][kk * N_TILE + t_n * N_PER_THREAD + nn];

            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    acc[mm][nn] = __builtin_amdgcn_sdot4(
                        a_reg[mm], b_reg[nn], acc[mm][nn], false);
        }
    }

    // =========================================================================
    // STORE RESULTS (same vectorized store as V1)
    // =========================================================================
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
    {
        const int m_idx = m_base + mm;
        if (m_idx < M)
        {
            if (n_base + N_PER_THREAD - 1 < N)
            {
                // All N_PER_THREAD columns are valid — vectorized uint4 stores
                const int64_t row_off = static_cast<int64_t>(m_idx) * N + n_base;
                #pragma unroll
                for (int chunk = 0; chunk < N_PER_THREAD; chunk += 4)
                {
                    *reinterpret_cast<uint4*>(d_C_int32 + row_off + chunk) =
                        make_uint4(static_cast<uint32_t>(acc[mm][chunk + 0]),
                                   static_cast<uint32_t>(acc[mm][chunk + 1]),
                                   static_cast<uint32_t>(acc[mm][chunk + 2]),
                                   static_cast<uint32_t>(acc[mm][chunk + 3]));
                }
            }
            else
            {
                // Tail N-block: scalar stores with bounds checking
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                {
                    const int n_idx = n_base + nn;
                    if (n_idx < N)
                    {
                        d_C_int32[static_cast<int64_t>(m_idx) * N + n_idx] = acc[mm][nn];
                    }
                }
            }
        }
    }
}

// ============================================================================
// WIDE-TILE V4: N128 Double-Buffered Pipeline (CK-parity target)
// ============================================================================
//
// V4 targets the FFN_Up/Gate 0.84× CK gap by matching CK's tile geometry:
//   - M128 × N128 × K16  (same as CK's 128x128 kernel)
//   - LDS double-buffering (from V3)
//   - Multi-pass B cooperative loading (new — handles KT×N_CHUNKS > 256)
//
// Why N128 matters for FFN_Up/Gate:
//   V1/N64 achieves only 84% compute density on N=18944 (partial tail block).
//   N128 raises this to 89%.  More importantly, N128 halves the grid block
//   count (148 vs 296), doubling the K-work per block and cutting overhead.
//
// Why V1/N128 failed but V4/N128 should work:
//   V1/N128 had no latency hiding — the wider tile increased VGPR pressure
//   (→ 2 waves/SIMD) without compensating.  V4 adds double-buffering to
//   overlap loads with compute, which the ISA analysis showed is CK's
//   primary technique for sustaining throughput at 2 waves/SIMD.
//
// LDS Budget (KT=16, N_TILE=128):
//   a_lds[2][16 × 128] × 4 bytes = 16 KB
//   b_lds[2][16 × 128] × 4 bytes = 16 KB
//   Total: 32 KB → 2 workgroups/CU (max 64 KB)
//
// LDS Budget (KT=8, N_TILE=128):
//   a_lds[2][8 × 128] × 4 bytes = 8 KB
//   b_lds[2][8 × 128] × 4 bytes = 8 KB
//   Total: 16 KB → 4 workgroups/CU
//
// Register Budget (N128):
//   64 accumulators (4M × 16N) + staging + operands ≈ 92 VGPRs → 2 waves/SIMD
//
// B Multi-Pass Loading:
//   KT=16, N=128 → B_TOTAL_ELEMENTS = 16 × 32 = 512 vec4-groups.
//   256 threads → 2 passes (256 elements each).
//   KT=8,  N=128 → B_TOTAL_ELEMENTS = 8 × 32 = 256 → 1 pass.
//
// ============================================================================

template <int N_TILE, int KT>
__global__ __launch_bounds__(256, 0)
void qgemm_int8_vnni_wide_tile_v4_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K)
{
    // Thread layout: 256 threads = 32 (M-dim) × 8 (N-dim)
    // With N128: each thread covers 4M × 16N = 64 accumulators
    constexpr int BLOCK_SIZE = 256;
    constexpr int THREADS_M = 32;
    constexpr int THREADS_N = BLOCK_SIZE / THREADS_M; // 8
    constexpr int M_PER_THREAD = 4;                   // 32 × 4 = 128 M-rows
    constexpr int N_PER_THREAD = N_TILE / THREADS_N;  // 128 / 8 = 16 N-cols
    constexpr int M_TILE = THREADS_M * M_PER_THREAD;  // 128

    static_assert(THREADS_M * THREADS_N == BLOCK_SIZE);
    static_assert(N_TILE % THREADS_N == 0);
    static_assert(KT % 4 == 0, "KT must be multiple of 4 for vectorized A loads");
    static_assert(N_TILE % 4 == 0, "N_TILE must be multiple of 4 for vectorized B loads");

    const int n_block = blockIdx.x * N_TILE;
    const int t_m = threadIdx.x % THREADS_M; // 0..31
    const int t_n = threadIdx.x / THREADS_M; // 0..7

    const int m_base = t_m * M_PER_THREAD;
    const int n_base = n_block + t_n * N_PER_THREAD;

    // =========================================================================
    // LDS: Double-buffered (ping-pong between buf[0] and buf[1])
    // =========================================================================
    __shared__ int32_t a_lds[2][KT * M_TILE];
    __shared__ int32_t b_lds[2][KT * N_TILE];

    // Accumulators: 4 × 16 = 64 per thread (N128) or 4 × 8 = 32 (N64)
    int32_t acc[M_PER_THREAD][N_PER_THREAD];
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
            acc[mm][nn] = 0;

    const int total_k_groups = K / 4;

    const int32_t* a_global = reinterpret_cast<const int32_t*>(d_A_int8);
    const int32_t* b_global = reinterpret_cast<const int32_t*>(d_B_int8_vnni);
    const int a_row_stride = K / 4;

    // =========================================================================
    // A cooperative load geometry — multi-pass (same as V3)
    // =========================================================================
    constexpr int A_K_CHUNKS = KT / 4;
    constexpr int A_ROWS_PER_PASS = BLOCK_SIZE / A_K_CHUNKS;
    constexpr int A_NUM_PASSES = (M_TILE + A_ROWS_PER_PASS - 1) / A_ROWS_PER_PASS;

    const int a_kg = (threadIdx.x % A_K_CHUNKS) * 4;
    const int a_thread_mi_base = threadIdx.x / A_K_CHUNKS;

    int a_mi[A_NUM_PASSES];
    #pragma unroll
    for (int p = 0; p < A_NUM_PASSES; ++p)
        a_mi[p] = a_thread_mi_base + p * A_ROWS_PER_PASS;

    // =========================================================================
    // B cooperative load geometry — multi-pass for large tiles
    //
    // B tile = KT × N_TILE int32 elements.  Each thread loads a uint4 (4 ints).
    // Total vec4-groups = KT × (N_TILE/4).
    //   KT=8,  N=64  → 128  groups → 1 pass (128 threads active)
    //   KT=8,  N=128 → 256  groups → 1 pass (256 threads active)
    //   KT=16, N=64  → 256  groups → 1 pass (256 threads active)
    //   KT=16, N=128 → 512  groups → 2 passes (256 threads × 2)
    //
    // Each thread is assigned a (kk, n_local) coordinate per pass.
    // =========================================================================
    constexpr int B_N_CHUNKS = N_TILE / 4;
    constexpr int B_TOTAL_GROUPS = KT * B_N_CHUNKS;
    constexpr int B_NUM_PASSES = (B_TOTAL_GROUPS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Pre-compute B load coordinates for each pass (loop-invariant)
    struct BLoadCoord {
        int kk_local;
        int nl;
        int gn;
        int lds_off;
        bool active;
    };
    BLoadCoord b_coords[B_NUM_PASSES];
    #pragma unroll
    for (int p = 0; p < B_NUM_PASSES; ++p)
    {
        const int flat = p * BLOCK_SIZE + threadIdx.x;
        if (flat < B_TOTAL_GROUPS)
        {
            b_coords[p].kk_local = flat / B_N_CHUNKS;
            b_coords[p].nl = (flat % B_N_CHUNKS) * 4;
            b_coords[p].gn = n_block + b_coords[p].nl;
            b_coords[p].lds_off = b_coords[p].kk_local * N_TILE + b_coords[p].nl;
            b_coords[p].active = true;
        }
        else
        {
            b_coords[p] = {0, 0, 0, 0, false};
        }
    }

    const int num_tiles = (total_k_groups + KT - 1) / KT;
    if (num_tiles == 0) return;

    // Staging registers for double-buffered pipeline
    uint4 a_staging[A_NUM_PASSES];
    uint4 b_staging[B_NUM_PASSES];

    // =========================================================================
    // Helper lambdas (device-code inlined by compiler)
    // =========================================================================

    // Issue all A global loads into staging registers
    auto issue_a_loads = [&](int kt_base) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            const int gkg = kt_base + a_kg;
            if (__builtin_expect(mi < M && gkg + 3 < total_k_groups, 1))
            {
                a_staging[p] = *reinterpret_cast<const uint4*>(
                    a_global + static_cast<int64_t>(mi) * a_row_stride + gkg);
            }
            else
            {
                a_staging[p].x = (mi < M && gkg + 0 < total_k_groups)
                    ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 0] : 0;
                a_staging[p].y = (mi < M && gkg + 1 < total_k_groups)
                    ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 1] : 0;
                a_staging[p].z = (mi < M && gkg + 2 < total_k_groups)
                    ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 2] : 0;
                a_staging[p].w = (mi < M && gkg + 3 < total_k_groups)
                    ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 3] : 0;
            }
        }
    };

    // Issue all B global loads into staging registers
    auto issue_b_loads = [&](int kt_base) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < B_NUM_PASSES; ++p)
        {
            if (b_coords[p].active)
            {
                const int gkk = kt_base + b_coords[p].kk_local;
                const int gn = b_coords[p].gn;
                if (__builtin_expect(gn + 3 < N && gkk < total_k_groups, 1))
                {
                    b_staging[p] = *reinterpret_cast<const uint4*>(
                        b_global + static_cast<int64_t>(gkk) * N + gn);
                }
                else
                {
                    b_staging[p] = make_uint4(0, 0, 0, 0);
                    b_staging[p].x = (gn + 0 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 0] : 0;
                    b_staging[p].y = (gn + 1 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 1] : 0;
                    b_staging[p].z = (gn + 2 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 2] : 0;
                    b_staging[p].w = (gn + 3 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 3] : 0;
                }
            }
        }
    };

    // Write staged A data to LDS buffer
    auto write_a_to_lds = [&](int buf) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            a_lds[buf][(a_kg + 0) * M_TILE + mi] = a_staging[p].x;
            a_lds[buf][(a_kg + 1) * M_TILE + mi] = a_staging[p].y;
            a_lds[buf][(a_kg + 2) * M_TILE + mi] = a_staging[p].z;
            a_lds[buf][(a_kg + 3) * M_TILE + mi] = a_staging[p].w;
        }
    };

    // Write staged B data to LDS buffer
    auto write_b_to_lds = [&](int buf) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < B_NUM_PASSES; ++p)
        {
            if (b_coords[p].active)
            {
                *reinterpret_cast<uint4*>(&b_lds[buf][b_coords[p].lds_off]) = b_staging[p];
            }
        }
    };

    // Compute GEMM tile from LDS buffer into accumulators
    auto compute_tile = [&](int buf) __attribute__((always_inline))
    {
        #pragma unroll
        for (int kk = 0; kk < KT; ++kk)
        {
            int32_t a_reg[M_PER_THREAD];
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                a_reg[mm] = a_lds[buf][kk * M_TILE + m_base + mm];

            int32_t b_reg[N_PER_THREAD];
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
                b_reg[nn] = b_lds[buf][kk * N_TILE + t_n * N_PER_THREAD + nn];

            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    acc[mm][nn] = __builtin_amdgcn_sdot4(
                        a_reg[mm], b_reg[nn], acc[mm][nn], false);
        }
    };

    // =========================================================================
    // PRELOAD: Load first tile (tile 0) into buf[0]
    // =========================================================================
    issue_a_loads(0);
    issue_b_loads(0);
    write_a_to_lds(0);
    write_b_to_lds(0);
    __syncthreads();

    // =========================================================================
    // MAIN LOOP: Double-buffered pipeline
    // =========================================================================
    for (int tile = 1; tile < num_tiles; ++tile)
    {
        const int cur_buf = (tile - 1) & 1;
        const int nxt_buf = tile & 1;
        const int kt_base = tile * KT;

        // 1. Issue global loads for next tile (async)
        issue_a_loads(kt_base);
        issue_b_loads(kt_base);

        // 2. Compute on current buffer (overlaps with global loads)
        compute_tile(cur_buf);

        // 3. Barrier: all threads done reading cur_buf
        __syncthreads();

        // 4. Write staged data to next buffer
        write_a_to_lds(nxt_buf);
        write_b_to_lds(nxt_buf);

        // 5. Barrier: LDS writes visible
        __syncthreads();
    }

    // =========================================================================
    // FINAL TILE
    // =========================================================================
    compute_tile((num_tiles - 1) & 1);

    // =========================================================================
    // STORE RESULTS (vectorized uint4 stores with tail N handling)
    // =========================================================================
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
    {
        const int m_idx = m_base + mm;
        if (m_idx < M)
        {
            if (n_base + N_PER_THREAD - 1 < N)
            {
                const int64_t row_off = static_cast<int64_t>(m_idx) * N + n_base;
                #pragma unroll
                for (int chunk = 0; chunk < N_PER_THREAD; chunk += 4)
                {
                    *reinterpret_cast<uint4*>(d_C_int32 + row_off + chunk) =
                        make_uint4(static_cast<uint32_t>(acc[mm][chunk + 0]),
                                   static_cast<uint32_t>(acc[mm][chunk + 1]),
                                   static_cast<uint32_t>(acc[mm][chunk + 2]),
                                   static_cast<uint32_t>(acc[mm][chunk + 3]));
                }
            }
            else
            {
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                {
                    const int n_idx = n_base + nn;
                    if (n_idx < N)
                    {
                        d_C_int32[static_cast<int64_t>(m_idx) * N + n_idx] = acc[mm][nn];
                    }
                }
            }
        }
    }
}

// ============================================================================
// WIDE-TILE V5: N128 Double-Buffered + ABBA Software-Pipelined Inner Loop
// ============================================================================
//
// V5 addresses V4's fatal flaw: 256 VGPR / 118 spills → 1 wave/SIMD.
// The root cause was `#pragma unroll` on the KT loop with 64 accumulators,
// causing the compiler to keep all KT × 20 operand VGPRs live simultaneously.
//
// V5 fixes this with three changes:
//
// 1. `__launch_bounds__(256, 2)` — forces 128 VGPR budget (2 waves/SIMD)
//
// 2. ABBA dual-register software pipeline for the KT loop:
//    - Two operand register sets: set[0] and set[1], 20 VGPRs each
//    - Load kk=0 into set[0], kk=1 into set[1], compute set[0],
//      load kk=2 into set[0] (reuse), compute set[1], etc.
//    - Peak live VGPRs: 64 (acc) + 2×20 (operands) + ~10 (misc) = ~114
//
// 3. Inline asm `s_waitcnt lgkmcnt(N)` for explicit partial waits:
//    - After loading set[next], wait with lgkmcnt = 5 (one set in-flight)
//    - Allows compute on set[cur] while set[next]'s reads complete
//
// Resource Budget (N128, KT=8):
//   Target: ~108 VGPRs → 2 waves/SIMD (vs V4's 256 + 118 spills = 1 wave)
//   Single-buffered inner loop: 20 operand VGPRs (a_reg[4] + b_reg[16])
//   kk loop NOT unrolled — prevents VGPR explosion, 2-wave hides LDS latency
//   LDS: same as V4 — a[2][KT×128] + b[2][KT×128] = 16/32 KB
//   Matches CK's occupancy profile: 128 VGPRs, 2 waves/SIMD, 0 spills
//
// ============================================================================

template <int N_TILE, int KT>
__global__ __launch_bounds__(256, 2)
void qgemm_int8_vnni_wide_tile_v5_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K)
{
    constexpr int BLOCK_SIZE = 256;
    constexpr int THREADS_M = 32;
    constexpr int THREADS_N = BLOCK_SIZE / THREADS_M; // 8
    constexpr int M_PER_THREAD = 4;
    constexpr int N_PER_THREAD = N_TILE / THREADS_N;  // 128/8 = 16
    constexpr int M_TILE = THREADS_M * M_PER_THREAD;  // 128

    static_assert(THREADS_M * THREADS_N == BLOCK_SIZE);
    static_assert(N_TILE % THREADS_N == 0);
    static_assert(KT % 4 == 0);
    static_assert(N_TILE % 4 == 0);
    static_assert(M_PER_THREAD == 4, "Vectorized A load requires M_PER_THREAD == 4");

    const int n_block = blockIdx.x * N_TILE;
    const int t_m = threadIdx.x % THREADS_M;
    const int t_n = threadIdx.x / THREADS_M;
    const int m_base = t_m * M_PER_THREAD;
    const int n_base = n_block + t_n * N_PER_THREAD;

    // =========================================================================
    // LDS: Double-buffered (same as V4)
    // =========================================================================
    __shared__ int32_t a_lds[2][KT * M_TILE];
    __shared__ int32_t b_lds[2][KT * N_TILE];

    // 64 accumulators — always live across the KT loop
    int32_t acc[M_PER_THREAD][N_PER_THREAD];
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
            acc[mm][nn] = 0;

    const int total_k_groups = K / 4;
    const int32_t* a_global = reinterpret_cast<const int32_t*>(d_A_int8);
    const int32_t* b_global = reinterpret_cast<const int32_t*>(d_B_int8_vnni);
    const int a_row_stride = K / 4;

    // =========================================================================
    // A cooperative load geometry (same as V4)
    // =========================================================================
    constexpr int A_K_CHUNKS = KT / 4;
    constexpr int A_ROWS_PER_PASS = BLOCK_SIZE / A_K_CHUNKS;
    constexpr int A_NUM_PASSES = (M_TILE + A_ROWS_PER_PASS - 1) / A_ROWS_PER_PASS;

    const int a_kg = (threadIdx.x % A_K_CHUNKS) * 4;
    const int a_thread_mi_base = threadIdx.x / A_K_CHUNKS;

    int a_mi[A_NUM_PASSES];
    #pragma unroll
    for (int p = 0; p < A_NUM_PASSES; ++p)
        a_mi[p] = a_thread_mi_base + p * A_ROWS_PER_PASS;

    // =========================================================================
    // B cooperative load geometry (same as V4)
    // =========================================================================
    constexpr int B_N_CHUNKS = N_TILE / 4;
    constexpr int B_TOTAL_GROUPS = KT * B_N_CHUNKS;
    constexpr int B_NUM_PASSES = (B_TOTAL_GROUPS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    struct BLoadCoord {
        int kk_local;
        int nl;
        int gn;
        int lds_off;
        bool active;
    };
    BLoadCoord b_coords[B_NUM_PASSES];
    #pragma unroll
    for (int p = 0; p < B_NUM_PASSES; ++p)
    {
        const int flat = p * BLOCK_SIZE + threadIdx.x;
        if (flat < B_TOTAL_GROUPS)
        {
            b_coords[p].kk_local = flat / B_N_CHUNKS;
            b_coords[p].nl = (flat % B_N_CHUNKS) * 4;
            b_coords[p].gn = n_block + b_coords[p].nl;
            b_coords[p].lds_off = b_coords[p].kk_local * N_TILE + b_coords[p].nl;
            b_coords[p].active = true;
        }
        else
        {
            b_coords[p] = {0, 0, 0, 0, false};
        }
    }

    const int num_tiles = (total_k_groups + KT - 1) / KT;
    if (num_tiles == 0) return;

    uint4 a_staging[A_NUM_PASSES];
    uint4 b_staging[B_NUM_PASSES];

    // =========================================================================
    // Global→LDS transfer lambdas (same as V4)
    // =========================================================================
    auto issue_a_loads = [&](int kt_base) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            const int gkg = kt_base + a_kg;
            if (__builtin_expect(mi < M && gkg + 3 < total_k_groups, 1))
            {
                a_staging[p] = *reinterpret_cast<const uint4*>(
                    a_global + static_cast<int64_t>(mi) * a_row_stride + gkg);
            }
            else
            {
                a_staging[p].x = (mi < M && gkg + 0 < total_k_groups)
                    ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 0] : 0;
                a_staging[p].y = (mi < M && gkg + 1 < total_k_groups)
                    ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 1] : 0;
                a_staging[p].z = (mi < M && gkg + 2 < total_k_groups)
                    ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 2] : 0;
                a_staging[p].w = (mi < M && gkg + 3 < total_k_groups)
                    ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 3] : 0;
            }
        }
    };

    auto issue_b_loads = [&](int kt_base) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < B_NUM_PASSES; ++p)
        {
            if (b_coords[p].active)
            {
                const int gkk = kt_base + b_coords[p].kk_local;
                const int gn = b_coords[p].gn;
                if (__builtin_expect(gn + 3 < N && gkk < total_k_groups, 1))
                {
                    b_staging[p] = *reinterpret_cast<const uint4*>(
                        b_global + static_cast<int64_t>(gkk) * N + gn);
                }
                else
                {
                    b_staging[p] = make_uint4(0, 0, 0, 0);
                    b_staging[p].x = (gn + 0 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 0] : 0;
                    b_staging[p].y = (gn + 1 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 1] : 0;
                    b_staging[p].z = (gn + 2 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 2] : 0;
                    b_staging[p].w = (gn + 3 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 3] : 0;
                }
            }
        }
    };

    auto write_a_to_lds = [&](int buf) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            a_lds[buf][(a_kg + 0) * M_TILE + mi] = a_staging[p].x;
            a_lds[buf][(a_kg + 1) * M_TILE + mi] = a_staging[p].y;
            a_lds[buf][(a_kg + 2) * M_TILE + mi] = a_staging[p].z;
            a_lds[buf][(a_kg + 3) * M_TILE + mi] = a_staging[p].w;
        }
    };

    auto write_b_to_lds = [&](int buf) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < B_NUM_PASSES; ++p)
        {
            if (b_coords[p].active)
            {
                *reinterpret_cast<uint4*>(&b_lds[buf][b_coords[p].lds_off]) = b_staging[p];
            }
        }
    };

    // =========================================================================
    // CHANGED: ABBA software-pipelined GEMM tile computation
    //
    // Instead of V4's `#pragma unroll for (kk=0..KT)` which explodes VGPRs:
    // - Two operand register sets: set[0] and set[1], 20 VGPRs each
    // - Runtime loop alternates: load→compute→load→compute (ABBA)
    // - Compute on set[cur] while set[nxt]'s reads are in-flight
    //
    // Register budget:
    //   64 (acc) + 2×4 (a_set) + 2×16 (b_set) + ~10 (addr/misc) = ~114
    //   With __launch_bounds__(256, 2) → compiler targets ≤128 VGPRs
    //
    // V5 rev2: Single-buffered inner loop (no ABBA).
    //   - 20 operand VGPRs (one set: a_reg[4] + b_reg[16]) instead of 40 (dual)
    //   - #pragma clang loop unroll(disable) prevents kk loop unroll explosion
    //   - 2-wave occupancy hides LDS read latency via wavefront switching
    //   - Target: 64 acc + 20 operand + ~24 outer = ~108 VGPRs → 0 spills
    // =========================================================================
    auto compute_tile = [&](int buf) __attribute__((always_inline))
    {
        const int32_t* __restrict__ a_ptr = &a_lds[buf][m_base];
        const int32_t* __restrict__ b_ptr = &b_lds[buf][t_n * N_PER_THREAD];

        // Single-buffered: load + compute each kk step.
        // No prefetching — 2 waves/SIMD handles latency hiding.
        #pragma clang loop unroll(disable)
        for (int kk = 0; kk < KT; ++kk)
        {
            // Operand registers: a_reg[4] + b_reg[16] = 20 VGPRs
            // Declared inside loop body → reused each iteration, never stacked
            int32_t a_reg[M_PER_THREAD];
            int32_t b_reg[N_PER_THREAD];

            // Vectorized LDS reads: 1 ds_read_b128 for A + 4 for B = 5 total
            *reinterpret_cast<uint4*>(&a_reg[0]) =
                *reinterpret_cast<const uint4*>(&a_ptr[kk * M_TILE]);

            #pragma unroll
            for (int c = 0; c < N_PER_THREAD; c += 4)
                *reinterpret_cast<uint4*>(&b_reg[c]) =
                    *reinterpret_cast<const uint4*>(&b_ptr[kk * N_TILE + c]);

            // 64 v_dot4 — fully unrolled for ALU throughput
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    acc[mm][nn] = __builtin_amdgcn_sdot4(
                        a_reg[mm], b_reg[nn], acc[mm][nn], false);
        }
    };

    // =========================================================================
    // PRELOAD: Load first tile (tile 0) into buf[0]
    // =========================================================================
    issue_a_loads(0);
    issue_b_loads(0);
    write_a_to_lds(0);
    write_b_to_lds(0);
    __syncthreads();

    // =========================================================================
    // MAIN LOOP: Double-buffered pipeline (same as V4)
    // =========================================================================
    for (int tile = 1; tile < num_tiles; ++tile)
    {
        const int cur_buf = (tile - 1) & 1;
        const int nxt_buf = tile & 1;
        const int kt_base = tile * KT;

        issue_a_loads(kt_base);
        issue_b_loads(kt_base);

        compute_tile(cur_buf);

        __syncthreads();

        write_a_to_lds(nxt_buf);
        write_b_to_lds(nxt_buf);

        __syncthreads();
    }

    // =========================================================================
    // FINAL TILE
    // =========================================================================
    compute_tile((num_tiles - 1) & 1);

    // =========================================================================
    // STORE RESULTS (vectorized uint4 stores)
    // =========================================================================
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
    {
        const int m_idx = m_base + mm;
        if (m_idx < M)
        {
            if (n_base + N_PER_THREAD - 1 < N)
            {
                const int64_t row_off = static_cast<int64_t>(m_idx) * N + n_base;
                #pragma unroll
                for (int chunk = 0; chunk < N_PER_THREAD; chunk += 4)
                {
                    *reinterpret_cast<uint4*>(d_C_int32 + row_off + chunk) =
                        make_uint4(static_cast<uint32_t>(acc[mm][chunk + 0]),
                                   static_cast<uint32_t>(acc[mm][chunk + 1]),
                                   static_cast<uint32_t>(acc[mm][chunk + 2]),
                                   static_cast<uint32_t>(acc[mm][chunk + 3]));
                }
            }
            else
            {
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                {
                    const int n_idx = n_base + nn;
                    if (n_idx < N)
                    {
                        d_C_int32[static_cast<int64_t>(m_idx) * N + n_idx] = acc[mm][nn];
                    }
                }
            }
        }
    }
}

// ============================================================================
// WIDE-TILE V6: Cross-Iteration Read-Compute Overlap via Partial Unroll
// ============================================================================
//
// V6 addresses V5's remaining scheduling inefficiency.
//
// ISA Analysis (V5 vs CK inner loop):
//
//   V5 (per kk step): 5 ds_read_b128 → lgkmcnt(3→2→1→0) → 64 v_dot4
//     - First 16 v_dot4 overlap with 3 outstanding reads (good)
//     - Remaining 48 v_dot4 have ZERO reads in-flight (wasted overlap)
//     - `#pragma clang loop unroll(disable)` prevents cross-step visibility
//
//   CK (fully unrolled): 128 ds_read_b128 interleaved with 2048 v_dot4
//     - 32 v_dot4 per compute block, 1-2 ds_read_b128 issued between blocks
//     - Always 5-6 LDS reads in-flight → constant latency hiding
//     - s_waitcnt lgkmcnt(5) → wait for oldest 1, keep 5 in pipeline
//
// V6 strategy: `#pragma clang loop unroll_count(2)` on the kk loop.
//
//   - Compiler sees 2 consecutive kk iterations (10 reads + 128 v_dot4)
//   - Can schedule kk+1's reads DURING kk's compute (CK's pattern)
//   - `__launch_bounds__(256, 2)` keeps 128 VGPR target
//   - Unlike V5r1 (ABBA), the compiler CHOOSES overlap degree — no forced
//     dual register sets. If overlap fits in 128 VGPRs, great; if not,
//     it serializes gracefully (degrading to V5 performance, not worse).
//
// Register Budget (N128, KT=8, unroll_count=2):
//   Peak live (with overlap): 64 acc + 20 (kk=0 operands) + 20 (kk=1 reads)
//                            + ~24 overhead = ~128 VGPRs
//   Peak live (no overlap):   64 acc + 20 operand + ~24 = ~108 VGPRs
//   Either way: ≤128 VGPRs target, 0 spills, 2 waves/SIMD
//
// ============================================================================

template <int N_TILE, int KT>
__global__ __launch_bounds__(256, 2)
void qgemm_int8_vnni_wide_tile_v6_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K)
{
    constexpr int BLOCK_SIZE = 256;
    constexpr int THREADS_M = 32;
    constexpr int THREADS_N = BLOCK_SIZE / THREADS_M; // 8
    constexpr int M_PER_THREAD = 4;
    constexpr int N_PER_THREAD = N_TILE / THREADS_N;  // 128/8 = 16
    constexpr int M_TILE = THREADS_M * M_PER_THREAD;  // 128

    static_assert(THREADS_M * THREADS_N == BLOCK_SIZE);
    static_assert(N_TILE % THREADS_N == 0);
    static_assert(KT % 4 == 0);
    static_assert(N_TILE % 4 == 0);
    static_assert(M_PER_THREAD == 4, "Vectorized A load requires M_PER_THREAD == 4");

    const int n_block = blockIdx.x * N_TILE;
    const int t_m = threadIdx.x % THREADS_M;
    const int t_n = threadIdx.x / THREADS_M;
    const int m_base = t_m * M_PER_THREAD;
    const int n_base = n_block + t_n * N_PER_THREAD;

    // =========================================================================
    // LDS: Double-buffered (same as V5)
    // =========================================================================
    __shared__ int32_t a_lds[2][KT * M_TILE];
    __shared__ int32_t b_lds[2][KT * N_TILE];

    // 64 accumulators — always live across the KT loop
    int32_t acc[M_PER_THREAD][N_PER_THREAD];
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
            acc[mm][nn] = 0;

    const int total_k_groups = K / 4;
    const int32_t* a_global = reinterpret_cast<const int32_t*>(d_A_int8);
    const int32_t* b_global = reinterpret_cast<const int32_t*>(d_B_int8_vnni);
    const int a_row_stride = K / 4;

    // =========================================================================
    // A cooperative load geometry (same as V5)
    // =========================================================================
    constexpr int A_K_CHUNKS = KT / 4;
    constexpr int A_ROWS_PER_PASS = BLOCK_SIZE / A_K_CHUNKS;
    constexpr int A_NUM_PASSES = (M_TILE + A_ROWS_PER_PASS - 1) / A_ROWS_PER_PASS;

    const int a_kg = (threadIdx.x % A_K_CHUNKS) * 4;
    const int a_thread_mi_base = threadIdx.x / A_K_CHUNKS;

    int a_mi[A_NUM_PASSES];
    #pragma unroll
    for (int p = 0; p < A_NUM_PASSES; ++p)
        a_mi[p] = a_thread_mi_base + p * A_ROWS_PER_PASS;

    // =========================================================================
    // B cooperative load geometry (same as V5)
    // =========================================================================
    constexpr int B_N_CHUNKS = N_TILE / 4;
    constexpr int B_TOTAL_GROUPS = KT * B_N_CHUNKS;
    constexpr int B_NUM_PASSES = (B_TOTAL_GROUPS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    struct BLoadCoord {
        int kk_local;
        int nl;
        int gn;
        int lds_off;
        bool active;
    };
    BLoadCoord b_coords[B_NUM_PASSES];
    #pragma unroll
    for (int p = 0; p < B_NUM_PASSES; ++p)
    {
        const int flat = p * BLOCK_SIZE + threadIdx.x;
        if (flat < B_TOTAL_GROUPS)
        {
            b_coords[p].kk_local = flat / B_N_CHUNKS;
            b_coords[p].nl = (flat % B_N_CHUNKS) * 4;
            b_coords[p].gn = n_block + b_coords[p].nl;
            b_coords[p].lds_off = b_coords[p].kk_local * N_TILE + b_coords[p].nl;
            b_coords[p].active = true;
        }
        else
        {
            b_coords[p] = {0, 0, 0, 0, false};
        }
    }

    const int num_tiles = (total_k_groups + KT - 1) / KT;
    if (num_tiles == 0) return;

    uint4 a_staging[A_NUM_PASSES];
    uint4 b_staging[B_NUM_PASSES];

    // =========================================================================
    // Global→LDS transfer lambdas (same as V5)
    // =========================================================================
    auto issue_a_loads = [&](int kt_base) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            const int gkg = kt_base + a_kg;
            if (__builtin_expect(mi < M && gkg + 3 < total_k_groups, 1))
            {
                a_staging[p] = *reinterpret_cast<const uint4*>(
                    a_global + static_cast<int64_t>(mi) * a_row_stride + gkg);
            }
            else
            {
                a_staging[p].x = (mi < M && gkg + 0 < total_k_groups)
                    ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 0] : 0;
                a_staging[p].y = (mi < M && gkg + 1 < total_k_groups)
                    ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 1] : 0;
                a_staging[p].z = (mi < M && gkg + 2 < total_k_groups)
                    ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 2] : 0;
                a_staging[p].w = (mi < M && gkg + 3 < total_k_groups)
                    ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 3] : 0;
            }
        }
    };

    auto issue_b_loads = [&](int kt_base) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < B_NUM_PASSES; ++p)
        {
            if (b_coords[p].active)
            {
                const int gkk = kt_base + b_coords[p].kk_local;
                const int gn = b_coords[p].gn;
                if (__builtin_expect(gn + 3 < N && gkk < total_k_groups, 1))
                {
                    b_staging[p] = *reinterpret_cast<const uint4*>(
                        b_global + static_cast<int64_t>(gkk) * N + gn);
                }
                else
                {
                    b_staging[p] = make_uint4(0, 0, 0, 0);
                    b_staging[p].x = (gn + 0 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 0] : 0;
                    b_staging[p].y = (gn + 1 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 1] : 0;
                    b_staging[p].z = (gn + 2 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 2] : 0;
                    b_staging[p].w = (gn + 3 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 3] : 0;
                }
            }
        }
    };

    auto write_a_to_lds = [&](int buf) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            a_lds[buf][(a_kg + 0) * M_TILE + mi] = a_staging[p].x;
            a_lds[buf][(a_kg + 1) * M_TILE + mi] = a_staging[p].y;
            a_lds[buf][(a_kg + 2) * M_TILE + mi] = a_staging[p].z;
            a_lds[buf][(a_kg + 3) * M_TILE + mi] = a_staging[p].w;
        }
    };

    auto write_b_to_lds = [&](int buf) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < B_NUM_PASSES; ++p)
        {
            if (b_coords[p].active)
            {
                *reinterpret_cast<uint4*>(&b_lds[buf][b_coords[p].lds_off]) = b_staging[p];
            }
        }
    };

    // =========================================================================
    // V6: SOFTWARE-PIPELINED INNER LOOP (non-unrolled pairs)
    //
    // Process kk steps in pairs with double-buffered operands. The pair
    // loop is NOT unrolled to stay within 128 VGPR budget:
    //   - Fully unrolled: 4 copies × 40 operands → 379 VGPR spills
    //   - Non-unrolled:   1 body  × 40 operands → fits ~118 VGPRs
    //
    // For each (kk, kk+1) pair:
    //   1) Load set 0 (5 ds_read) ─── sched_barrier ───
    //   2) Load set 1 (5 ds_read) ─── sched_barrier ───
    //   3) Compute kk   (64 v_dot4, compiler inserts lgkmcnt(5))
    //            ── sched_barrier ───
    //   4) Compute kk+1 (64 v_dot4, set 1 data ready)
    //            ── sched_barrier ───
    //
    // The ~8% loop overhead (address compute + branch) is more than
    // recovered by hiding ~30 cycles of LDS latency per pair.
    //
    // Register budget: 64 acc + 40 operands + 8 staging + ~6 misc = ~118
    // =========================================================================
    auto compute_tile = [&](int buf) __attribute__((always_inline))
    {
        const int32_t* __restrict__ a_ptr = &a_lds[buf][m_base];
        const int32_t* __restrict__ b_ptr = &b_lds[buf][t_n * N_PER_THREAD];

        static_assert(KT % 2 == 0, "KT must be even for paired processing");

        #pragma clang loop unroll(disable)
        for (int kk = 0; kk < KT; kk += 2)
        {
            int32_t a0[M_PER_THREAD], a1[M_PER_THREAD];
            int32_t b0[N_PER_THREAD], b1[N_PER_THREAD];

            // ═══ LOAD SET 0: kk operands (5 ds_read_b128) ═══════
            *reinterpret_cast<uint4*>(&a0[0]) =
                *reinterpret_cast<const uint4*>(&a_ptr[kk * M_TILE]);
            #pragma unroll
            for (int c = 0; c < N_PER_THREAD; c += 4)
                *reinterpret_cast<uint4*>(&b0[c]) =
                    *reinterpret_cast<const uint4*>(&b_ptr[kk * N_TILE + c]);

            // fence: set 0 reads issued before set 1 reads
            __builtin_amdgcn_sched_barrier(0);

            // ═══ LOAD SET 1: kk+1 operands (5 ds_read_b128) ═════
            *reinterpret_cast<uint4*>(&a1[0]) =
                *reinterpret_cast<const uint4*>(&a_ptr[(kk + 1) * M_TILE]);
            #pragma unroll
            for (int c = 0; c < N_PER_THREAD; c += 4)
                *reinterpret_cast<uint4*>(&b1[c]) =
                    *reinterpret_cast<const uint4*>(&b_ptr[(kk + 1) * N_TILE + c]);

            // fence: all 10 reads must be issued before compute
            __builtin_amdgcn_sched_barrier(0);

            // ═══ COMPUTE kk: 64 v_dot4 from set 0 ═══════════════
            // Compiler inserts lgkmcnt(5): set 0 ready, set 1 in-flight
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    acc[mm][nn] = __builtin_amdgcn_sdot4(
                        a0[mm], b0[nn], acc[mm][nn], false);

            // fence: kk compute complete before kk+1 compute
            __builtin_amdgcn_sched_barrier(0);

            // ═══ COMPUTE kk+1: 64 v_dot4 from set 1 ═════════════
            // Set 1 data ready (64 ALU cycles >> 30 LDS latency)
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    acc[mm][nn] = __builtin_amdgcn_sdot4(
                        a1[mm], b1[nn], acc[mm][nn], false);

            // fence: pair complete before next pair's reads
            __builtin_amdgcn_sched_barrier(0);
        }
    };

    // =========================================================================
    // PRELOAD: Load first tile (tile 0) into buf[0]
    // =========================================================================
    issue_a_loads(0);
    issue_b_loads(0);
    write_a_to_lds(0);
    write_b_to_lds(0);
    __syncthreads();

    // =========================================================================
    // MAIN LOOP: Double-buffered pipeline (same as V5)
    // =========================================================================
    for (int tile = 1; tile < num_tiles; ++tile)
    {
        const int cur_buf = (tile - 1) & 1;
        const int nxt_buf = tile & 1;
        const int kt_base = tile * KT;

        issue_a_loads(kt_base);
        issue_b_loads(kt_base);

        compute_tile(cur_buf);

        __syncthreads();

        write_a_to_lds(nxt_buf);
        write_b_to_lds(nxt_buf);

        __syncthreads();
    }

    // =========================================================================
    // FINAL TILE
    // =========================================================================
    compute_tile((num_tiles - 1) & 1);

    // =========================================================================
    // STORE RESULTS (vectorized uint4 stores)
    // =========================================================================
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
    {
        const int m_idx = m_base + mm;
        if (m_idx < M)
        {
            if (n_base + N_PER_THREAD - 1 < N)
            {
                const int64_t row_off = static_cast<int64_t>(m_idx) * N + n_base;
                #pragma unroll
                for (int chunk = 0; chunk < N_PER_THREAD; chunk += 4)
                {
                    *reinterpret_cast<uint4*>(d_C_int32 + row_off + chunk) =
                        make_uint4(static_cast<uint32_t>(acc[mm][chunk + 0]),
                                   static_cast<uint32_t>(acc[mm][chunk + 1]),
                                   static_cast<uint32_t>(acc[mm][chunk + 2]),
                                   static_cast<uint32_t>(acc[mm][chunk + 3]));
                }
            }
            else
            {
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                {
                    const int n_idx = n_base + nn;
                    if (n_idx < N)
                    {
                        d_C_int32[static_cast<int64_t>(m_idx) * N + n_idx] = acc[mm][nn];
                    }
                }
            }
        }
    }
}

// ============================================================================
// WIDE-TILE V7: Safe-Tile Split with Branchless Interior Loop
// ============================================================================
//
// V7 addresses the outer-pipeline overhead identified by rocprof profiling.
//
// rocprof per-dispatch analysis (7B FFN_Up, M=128 N=18944 K=3584):
//   V5 GPU kernel: 0.771ms   CK GPU kernel: 0.723ms   Gap: 6.6%
//
// V5 ISA analysis shows the compiler emits both fast and slow (boundary-
// checked) code paths within a single loop body: 15 branches in the fast
// path + 84 branches in the boundary path = 99 loop branches total.
// Even though branches are well-predicted, the duplicated boundary code:
//   1. Wastes icache: ~800 extra instructions in the loop function
//   2. Pollutes the L1I$ (16 KB/CU on gfx906), potentially evicting
//      hot compute code on larger loop bodies
//   3. Prevents optimal instruction scheduling — the compiler must
//      generate branch-safe code even in the fast path
//
// V7 strategy: Split the outer loop into TWO separate loops:
//
//   SAFE LOOP:     tiles 1..num_safe_tiles-1
//     - ALL loads are unconditional vectorized (global_load_dwordx4)
//     - ZERO boundary checks, ZERO branches in load path
//     - Compiler can generate maximally tight ISA
//
//   BOUNDARY LOOP: tiles num_safe_tiles..num_tiles-1
//     - Full boundary checks (same as V5)
//     - Only runs for the last 0-2 tiles (or never for aligned shapes)
//
// Safe tile computation:
//   A loads safe when: a_mi[p] < M && kt_base + a_kg + 3 < total_k_groups
//     → M >= M_TILE && (tile+1)*KT <= total_k_groups
//   B loads safe when: gn + 3 < N && gkk < total_k_groups
//     → n_block + N_TILE <= N && (tile+1)*KT <= total_k_groups
//   Combined: safe for tiles 0..floor(total_k_groups/KT)-1 when M==M_TILE
//             and this is an interior N-block.
//
// For 7B FFN_Up (M=128, N=18944, K=3584):
//   total_k_groups=896, KT=8 → 896/8=112 = num_tiles → ALL tiles safe
//   18944/128=148, 18944%128=0 → ALL N-blocks are interior
//   → Boundary loop NEVER executes. Safe loop covers everything.
//
// Register budget: identical to V5 (64 acc + 20 staging + 20 operand + ~13)
//   → 117 VGPRs, 0 spills, 2 waves/SIMD
//
// ============================================================================

template <int N_TILE, int KT>
__global__ __launch_bounds__(256, 2)
void qgemm_int8_vnni_wide_tile_v7_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K)
{
    constexpr int BLOCK_SIZE = 256;
    constexpr int THREADS_M = 32;
    constexpr int THREADS_N = BLOCK_SIZE / THREADS_M; // 8
    constexpr int M_PER_THREAD = 4;
    constexpr int N_PER_THREAD = N_TILE / THREADS_N;  // 128/8 = 16
    constexpr int M_TILE = THREADS_M * M_PER_THREAD;  // 128

    static_assert(THREADS_M * THREADS_N == BLOCK_SIZE);
    static_assert(N_TILE % THREADS_N == 0);
    static_assert(KT % 4 == 0);
    static_assert(N_TILE % 4 == 0);
    static_assert(M_PER_THREAD == 4, "Vectorized A load requires M_PER_THREAD == 4");

    const int n_block = blockIdx.x * N_TILE;
    const int t_m = threadIdx.x % THREADS_M;
    const int t_n = threadIdx.x / THREADS_M;
    const int m_base = t_m * M_PER_THREAD;
    const int n_base = n_block + t_n * N_PER_THREAD;

    // =========================================================================
    // LDS: Double-buffered
    // =========================================================================
    __shared__ int32_t a_lds[2][KT * M_TILE];
    __shared__ int32_t b_lds[2][KT * N_TILE];

    // 64 accumulators — always live across the KT loop
    int32_t acc[M_PER_THREAD][N_PER_THREAD];
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
            acc[mm][nn] = 0;

    const int total_k_groups = K / 4;
    const int32_t* a_global = reinterpret_cast<const int32_t*>(d_A_int8);
    const int32_t* b_global = reinterpret_cast<const int32_t*>(d_B_int8_vnni);
    const int a_row_stride = K / 4;

    // =========================================================================
    // A cooperative load geometry
    // =========================================================================
    constexpr int A_K_CHUNKS = KT / 4;
    constexpr int A_ROWS_PER_PASS = BLOCK_SIZE / A_K_CHUNKS;
    constexpr int A_NUM_PASSES = (M_TILE + A_ROWS_PER_PASS - 1) / A_ROWS_PER_PASS;

    const int a_kg = (threadIdx.x % A_K_CHUNKS) * 4;
    const int a_thread_mi_base = threadIdx.x / A_K_CHUNKS;

    int a_mi[A_NUM_PASSES];
    #pragma unroll
    for (int p = 0; p < A_NUM_PASSES; ++p)
        a_mi[p] = a_thread_mi_base + p * A_ROWS_PER_PASS;

    // =========================================================================
    // B cooperative load geometry
    // =========================================================================
    constexpr int B_N_CHUNKS = N_TILE / 4;
    constexpr int B_TOTAL_GROUPS = KT * B_N_CHUNKS;
    constexpr int B_NUM_PASSES = (B_TOTAL_GROUPS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    struct BLoadCoord {
        int kk_local;
        int nl;
        int gn;
        int lds_off;
        bool active;
    };
    BLoadCoord b_coords[B_NUM_PASSES];
    #pragma unroll
    for (int p = 0; p < B_NUM_PASSES; ++p)
    {
        const int flat = p * BLOCK_SIZE + threadIdx.x;
        if (flat < B_TOTAL_GROUPS)
        {
            b_coords[p].kk_local = flat / B_N_CHUNKS;
            b_coords[p].nl = (flat % B_N_CHUNKS) * 4;
            b_coords[p].gn = n_block + b_coords[p].nl;
            b_coords[p].lds_off = b_coords[p].kk_local * N_TILE + b_coords[p].nl;
            b_coords[p].active = true;
        }
        else
        {
            b_coords[p] = {0, 0, 0, 0, false};
        }
    }

    const int num_tiles = (total_k_groups + KT - 1) / KT;
    if (num_tiles == 0) return;

    uint4 a_staging[A_NUM_PASSES];
    uint4 b_staging[B_NUM_PASSES];

    // =========================================================================
    // V7 CHANGE: Pre-compute safe tile count
    //
    // Safe tiles are those where ALL loads (A and B) are guaranteed in-bounds
    // for ALL threads. This allows the safe loop to use unconditional
    // vectorized loads with zero boundary checks.
    //
    // Safe conditions (all must hold):
    //   1. M >= M_TILE (all a_mi[p] < M)
    //   2. n_block + N_TILE <= N (all B column accesses within bounds)
    //   3. (tile+1)*KT <= total_k_groups (all K-group accesses within bounds)
    //
    // num_safe_tiles = number of tiles satisfying condition 3 (0-indexed).
    // If conditions 1 or 2 fail, num_safe_tiles = 0 (all tiles boundary-checked).
    // =========================================================================
    const bool a_m_safe  = (M >= M_TILE);
    const bool b_n_safe  = (n_block + N_TILE <= N);
    const int  safe_k    = total_k_groups / KT;  // floor: tiles with full K coverage
    const int  num_safe  = (a_m_safe && b_n_safe) ? (safe_k < num_tiles ? safe_k : num_tiles) : 0;

    // =========================================================================
    // SAFE load lambdas: unconditional vectorized loads (no boundary checks)
    // =========================================================================
    auto safe_issue_a_loads = [&](int kt_base) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            a_staging[p] = *reinterpret_cast<const uint4*>(
                a_global + static_cast<int64_t>(a_mi[p]) * a_row_stride + kt_base + a_kg);
        }
    };

    auto safe_issue_b_loads = [&](int kt_base) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < B_NUM_PASSES; ++p)
        {
            if (b_coords[p].active)
            {
                b_staging[p] = *reinterpret_cast<const uint4*>(
                    b_global + static_cast<int64_t>(kt_base + b_coords[p].kk_local) * N + b_coords[p].gn);
            }
        }
    };

    // =========================================================================
    // BOUNDARY load lambdas: with boundary checks (same as V5)
    // =========================================================================
    auto boundary_issue_a_loads = [&](int kt_base) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            const int gkg = kt_base + a_kg;
            if (__builtin_expect(mi < M && gkg + 3 < total_k_groups, 1))
            {
                a_staging[p] = *reinterpret_cast<const uint4*>(
                    a_global + static_cast<int64_t>(mi) * a_row_stride + gkg);
            }
            else
            {
                a_staging[p].x = (mi < M && gkg + 0 < total_k_groups)
                    ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 0] : 0;
                a_staging[p].y = (mi < M && gkg + 1 < total_k_groups)
                    ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 1] : 0;
                a_staging[p].z = (mi < M && gkg + 2 < total_k_groups)
                    ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 2] : 0;
                a_staging[p].w = (mi < M && gkg + 3 < total_k_groups)
                    ? a_global[static_cast<int64_t>(mi) * a_row_stride + gkg + 3] : 0;
            }
        }
    };

    auto boundary_issue_b_loads = [&](int kt_base) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < B_NUM_PASSES; ++p)
        {
            if (b_coords[p].active)
            {
                const int gkk = kt_base + b_coords[p].kk_local;
                const int gn = b_coords[p].gn;
                if (__builtin_expect(gn + 3 < N && gkk < total_k_groups, 1))
                {
                    b_staging[p] = *reinterpret_cast<const uint4*>(
                        b_global + static_cast<int64_t>(gkk) * N + gn);
                }
                else
                {
                    b_staging[p] = make_uint4(0, 0, 0, 0);
                    b_staging[p].x = (gn + 0 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 0] : 0;
                    b_staging[p].y = (gn + 1 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 1] : 0;
                    b_staging[p].z = (gn + 2 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 2] : 0;
                    b_staging[p].w = (gn + 3 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 3] : 0;
                }
            }
        }
    };

    // =========================================================================
    // LDS write lambdas (same as V5)
    // =========================================================================
    auto write_a_to_lds = [&](int buf) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            a_lds[buf][(a_kg + 0) * M_TILE + mi] = a_staging[p].x;
            a_lds[buf][(a_kg + 1) * M_TILE + mi] = a_staging[p].y;
            a_lds[buf][(a_kg + 2) * M_TILE + mi] = a_staging[p].z;
            a_lds[buf][(a_kg + 3) * M_TILE + mi] = a_staging[p].w;
        }
    };

    auto write_b_to_lds = [&](int buf) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < B_NUM_PASSES; ++p)
        {
            if (b_coords[p].active)
            {
                *reinterpret_cast<uint4*>(&b_lds[buf][b_coords[p].lds_off]) = b_staging[p];
            }
        }
    };

    // =========================================================================
    // Compute tile (same as V5: single-buffered inner loop)
    // =========================================================================
    auto compute_tile = [&](int buf) __attribute__((always_inline))
    {
        const int32_t* __restrict__ a_ptr = &a_lds[buf][m_base];
        const int32_t* __restrict__ b_ptr = &b_lds[buf][t_n * N_PER_THREAD];

        #pragma clang loop unroll(disable)
        for (int kk = 0; kk < KT; ++kk)
        {
            int32_t a_reg[M_PER_THREAD];
            int32_t b_reg[N_PER_THREAD];

            *reinterpret_cast<uint4*>(&a_reg[0]) =
                *reinterpret_cast<const uint4*>(&a_ptr[kk * M_TILE]);

            #pragma unroll
            for (int c = 0; c < N_PER_THREAD; c += 4)
                *reinterpret_cast<uint4*>(&b_reg[c]) =
                    *reinterpret_cast<const uint4*>(&b_ptr[kk * N_TILE + c]);

            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    acc[mm][nn] = __builtin_amdgcn_sdot4(
                        a_reg[mm], b_reg[nn], acc[mm][nn], false);
        }
    };

    // =========================================================================
    // PRELOAD: tile 0 (safe or boundary)
    // =========================================================================
    if (num_safe > 0)
    {
        safe_issue_a_loads(0);
        safe_issue_b_loads(0);
    }
    else
    {
        boundary_issue_a_loads(0);
        boundary_issue_b_loads(0);
    }
    write_a_to_lds(0);
    write_b_to_lds(0);
    __syncthreads();

    // =========================================================================
    // SAFE LOOP: tiles 1..num_safe-1 (zero boundary checks)
    //
    // This is the hot path for aligned shapes. The compiler generates a
    // tight loop body without any boundary-checking branches or fallback
    // scalar load code, resulting in smaller ISA and better icache utilization.
    // =========================================================================
    for (int tile = 1; tile < num_safe; ++tile)
    {
        const int cur_buf = (tile - 1) & 1;
        const int nxt_buf = tile & 1;
        const int kt_base = tile * KT;

        safe_issue_a_loads(kt_base);
        safe_issue_b_loads(kt_base);

        compute_tile(cur_buf);

        __syncthreads();

        write_a_to_lds(nxt_buf);
        write_b_to_lds(nxt_buf);

        __syncthreads();
    }

    // =========================================================================
    // BOUNDARY LOOP: tiles max(1, num_safe)..num_tiles-1 (with boundary checks)
    //
    // Handles the last 0-2 tiles where K or N boundary checks are needed.
    // For aligned shapes (M=128, N%128==0, K%32==0), this loop never executes.
    // =========================================================================
    {
        const int boundary_start = (num_safe > 0) ? num_safe : 1;
        for (int tile = boundary_start; tile < num_tiles; ++tile)
        {
            const int cur_buf = (tile - 1) & 1;
            const int nxt_buf = tile & 1;
            const int kt_base = tile * KT;

            boundary_issue_a_loads(kt_base);
            boundary_issue_b_loads(kt_base);

            compute_tile(cur_buf);

            __syncthreads();

            write_a_to_lds(nxt_buf);
            write_b_to_lds(nxt_buf);

            __syncthreads();
        }
    }

    // =========================================================================
    // FINAL TILE
    // =========================================================================
    compute_tile((num_tiles - 1) & 1);

    // =========================================================================
    // STORE RESULTS (vectorized uint4 stores)
    // =========================================================================
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
    {
        const int m_idx = m_base + mm;
        if (m_idx < M)
        {
            if (n_base + N_PER_THREAD - 1 < N)
            {
                const int64_t row_off = static_cast<int64_t>(m_idx) * N + n_base;
                #pragma unroll
                for (int chunk = 0; chunk < N_PER_THREAD; chunk += 4)
                {
                    *reinterpret_cast<uint4*>(d_C_int32 + row_off + chunk) =
                        make_uint4(static_cast<uint32_t>(acc[mm][chunk + 0]),
                                   static_cast<uint32_t>(acc[mm][chunk + 1]),
                                   static_cast<uint32_t>(acc[mm][chunk + 2]),
                                   static_cast<uint32_t>(acc[mm][chunk + 3]));
                }
            }
            else
            {
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                {
                    const int n_idx = n_base + nn;
                    if (n_idx < N)
                    {
                        d_C_int32[static_cast<int64_t>(m_idx) * N + n_idx] = acc[mm][nn];
                    }
                }
            }
        }
    }
}

// ============================================================================
// STUBS: Phase 4 functions return success but log "not yet implemented"
// ============================================================================

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

    // -------------------------------------------------------------------------
    // Device management
    // -------------------------------------------------------------------------

    bool rocmQuantGemm_setDevice(int rocm_device_id)
    {
        hipError_t err = static_cast<hipError_t>(llaminar2::HipDeviceGuard::setDevice(rocm_device_id));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_setDevice] Failed to set device %d: %s\n",
                    rocm_device_id, hipGetErrorString(err));
            return false;
        }
        return true;
    }

    // -------------------------------------------------------------------------
    // Weight upload (Phase 1)
    // -------------------------------------------------------------------------

    bool rocmQuantGemm_uploadWeights(
        const int8_t *h_weights_int8,
        const float *h_scales_B,
        int8_t **d_weights_int8,
        float **d_scales_B,
        int K, int N,
        int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        // Allocate device memory for weights
        size_t weight_bytes = static_cast<size_t>(K) * N * sizeof(int8_t);
        hipError_t err = hipMalloc(d_weights_int8, weight_bytes);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_uploadWeights] hipMalloc weights failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        // CRITICAL: Verify hipMalloc actually allocated memory
        if (*d_weights_int8 == nullptr)
        {
            fprintf(stderr, "[rocmQuantGemm_uploadWeights] CRITICAL: hipMalloc returned success but pointer is NULL! "
                    "(bytes=%zu K=%d N=%d device=%d)\n",
                    weight_bytes, K, N, rocm_device_id);
            return false;
        }

        // Allocate device memory for scales
        size_t scale_bytes = static_cast<size_t>(N) * sizeof(float);
        err = hipMalloc(d_scales_B, scale_bytes);
        if (err != hipSuccess)
        {
            hipFree(*d_weights_int8);
            *d_weights_int8 = nullptr;
            fprintf(stderr, "[rocmQuantGemm_uploadWeights] hipMalloc scales failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }

        // Copy weights H2D
        err = hipMemcpy(*d_weights_int8, h_weights_int8, weight_bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess)
        {
            hipFree(*d_weights_int8);
            hipFree(*d_scales_B);
            *d_weights_int8 = nullptr;
            *d_scales_B = nullptr;
            fprintf(stderr, "[rocmQuantGemm_uploadWeights] hipMemcpy weights failed: %s "
                    "(src=%p dst=%p bytes=%zu K=%d N=%d device=%d)\n",
                    hipGetErrorString(err), (void*)h_weights_int8, (void*)*d_weights_int8,
                    weight_bytes, K, N, rocm_device_id);
            return false;
        }

        // Copy scales H2D
        err = hipMemcpy(*d_scales_B, h_scales_B, scale_bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess)
        {
            hipFree(*d_weights_int8);
            hipFree(*d_scales_B);
            *d_weights_int8 = nullptr;
            *d_scales_B = nullptr;
            fprintf(stderr, "[rocmQuantGemm_uploadWeights] hipMemcpy scales failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }

        return true;
    }

    // -------------------------------------------------------------------------
    // Work buffer allocation (Phase 2)
    // -------------------------------------------------------------------------

    bool rocmQuantGemm_ensureWorkBuffers(
        int8_t **d_A_int8,
        float **d_scales_A,
        int32_t **d_C_int32,
        int *work_buffer_M,
        int M, int K, int N,
        int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        // Only reallocate if M increased
        if (M <= *work_buffer_M)
        {
            return true;
        }

        // Free existing buffers
        if (*d_A_int8) {
            hipFree(*d_A_int8);
        }
        if (*d_scales_A) {
            hipFree(*d_scales_A);
        }
        if (*d_C_int32) {
            hipFree(*d_C_int32);
        }

        *d_A_int8 = nullptr;
        *d_scales_A = nullptr;
        *d_C_int32 = nullptr;

        // Allocate new buffers
        hipError_t err = hipMalloc(d_A_int8, static_cast<size_t>(M) * K * sizeof(int8_t));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_ensureWorkBuffers] hipMalloc A_int8 failed: %s\n", hipGetErrorString(err));
            return false;
        }

        err = hipMalloc(d_scales_A, static_cast<size_t>(M) * sizeof(float));
        if (err != hipSuccess)
        {
            hipFree(*d_A_int8);
            *d_A_int8 = nullptr;
            fprintf(stderr, "[rocmQuantGemm_ensureWorkBuffers] hipMalloc scales_A failed: %s\n", hipGetErrorString(err));
            return false;
        }

        err = hipMalloc(d_C_int32, static_cast<size_t>(M) * N * sizeof(int32_t));
        if (err != hipSuccess)
        {
            hipFree(*d_A_int8);
            hipFree(*d_scales_A);
            *d_A_int8 = nullptr;
            *d_scales_A = nullptr;
            fprintf(stderr, "[rocmQuantGemm_ensureWorkBuffers] hipMalloc C_int32 failed: %s\n", hipGetErrorString(err));
            return false;
        }

        *work_buffer_M = M;
        return true;
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

    // =========================================================================
    // TWO-KERNEL CK APPROACH (DEFAULT - BEST ACCURACY + PERFORMANCE)
    // =========================================================================
    //
    // rocmQuantGemm_executeTwoKernel - CK INT8→INT32 GEMM + applyScales_kernel
    //
    // ✅ THIS IS THE DEFAULT AND RECOMMENDED PATH FOR M >= 64
    //
    // This approach runs two kernels:
    //   1. CK's DeviceGemmMultipleD_Dl for INT8×INT8→INT32 (no D-tensor scaling)
    //   2. applyScales_kernel to convert INT32→FP32 with per-row/col scales
    //
    // FORMULA:
    //   C_int32[m,n] = sum_k(A_int8[m,k] × B_int8[k,n])    [CK GEMM]
    //   E_fp32[m,n] = C_int32[m,n] × scale_A[m] × scale_B[n]  [applyScales_kernel]
    //
    // WHY THIS IS DEFAULT (January 2026 benchmarks on MI50/MI60, Qwen2.5-7B):
    //
    //   | Approach      | M=512 FFN TFLOPS | Cosine Similarity |
    //   |---------------|------------------|-------------------|
    //   | Two-Kernel    | 3.403            | 0.9999 ✅         |
    //   | Fused         | 3.349 (-1.6%)    | 0.9917 ❌         |
    //   | hipBLAS       | 2.461 (-28%)     | 0.9999 ✅         |
    //
    //   Two-Kernel provides:
    //     ✅ Same accuracy as hipBLAS (gold standard)
    //     ✅ Slightly faster than fused CK (+1.6%)
    //     ✅ 38% faster than hipBLAS
    //
    // MEMORY:
    //   - Allocates temporary [M×N×4] bytes for INT32 accumulator
    //   - Freed immediately after scaling kernel completes
    //   - For hot-path, use rocmQuantGemm_executeTwoKernel_cached() instead
    //
    // =========================================================================
    bool rocmQuantGemm_executeTwoKernel(
        const int8_t *d_A_int8,       // [M × K] INT8 activations (row-major)
        const int8_t *d_B_int8,       // [K × N] INT8 weights (row-major)
        float *d_E_fp32,              // [M × N] FP32 output (row-major)
        const float *d_scale_A,       // [M] per-row scales
        const float *d_scale_B,       // [N] per-column scales
        int M, int N, int K,
        int rocm_device_id, void* stream,
        void* kernel_ctx)
    {
        if (M <= 0 || N <= 0 || K <= 0)
            return true;

        // 1. Allocate INT32 accumulator buffer
        int32_t *d_C_int32 = nullptr;
        hipError_t err = hipMalloc(&d_C_int32, static_cast<size_t>(M) * N * sizeof(int32_t));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel] Failed to allocate INT32 buffer: %s\n",
                    hipGetErrorString(err));
            return false;
        }

        // 2. Run CK INT8 GEMM: C_int32 = A_int8 × B_int8 (no scaling)
        if (!rocmQuantGemm_executeNoScale(d_A_int8, d_B_int8, d_C_int32, M, N, K, rocm_device_id, stream, kernel_ctx))
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel] CK NoScale GEMM failed\n");
            hipFree(d_C_int32);
            return false;
        }

        // 3. Apply scales: E_fp32[m,n] = C_int32[m,n] * scale_A[m] * scale_B[n]
        const int total = M * N;
        const int block_size = 256;
        const int grid_size = (total + block_size - 1) / block_size;

        hipLaunchKernelGGL(
            applyScales_kernel,
            dim3(grid_size), dim3(block_size), 0, static_cast<hipStream_t>(stream),
            d_C_int32, d_E_fp32, d_scale_A, d_scale_B, M, N);

        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel] applyScales_kernel failed: %s\n",
                    hipGetErrorString(err));
            hipFree(d_C_int32);
            return false;
        }

        // 4. Cleanup
        hipFree(d_C_int32);
        return true;
    }

    // =========================================================================
    // ALLOCATION-FREE TWO-KERNEL CK APPROACH (HOT-PATH OPTIMIZED)
    // =========================================================================
    //
    // rocmQuantGemm_executeTwoKernel_cached - CK INT8→INT32 GEMM + applyScales_kernel
    //
    // ✅ THIS IS THE PREFERRED PATH FOR HOT-PATH EXECUTION (no allocations)
    //
    // Same algorithm as rocmQuantGemm_executeTwoKernel but takes a pre-allocated
    // INT32 accumulator buffer to avoid hipMalloc/hipFree overhead on every call.
    //
    // PARAMETERS:
    //   d_C_int32 - Pre-allocated [M × N] INT32 buffer for accumulator
    //               Must be large enough for M * N * sizeof(int32_t) bytes
    //
    // =========================================================================
    bool rocmQuantGemm_executeTwoKernel_cached(
        const int8_t *d_A_int8,       // [M × K] INT8 activations (row-major)
        const int8_t *d_B_int8,       // [K × N] INT8 weights (row-major)
        float *d_E_fp32,              // [M × N] FP32 output (row-major)
        const float *d_scale_A,       // [M] per-row scales
        const float *d_scale_B,       // [N] per-column scales
        int32_t *d_C_int32,           // Pre-allocated [M × N] INT32 accumulator
        int M, int N, int K,
        int rocm_device_id, void* stream,
        void* kernel_ctx)
    {
        auto t_total_start = std::chrono::high_resolution_clock::now();
        auto t_step_start = t_total_start;
        auto t_step_end = t_total_start;
        
        auto log_timing = [&](const char* step) {
            t_step_end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t_step_end - t_step_start).count();
            if (ms > 0.5) {
                fprintf(stderr, "[CK_GEMM_TIMING] %s: %.3f ms (M=%d N=%d K=%d)\n", step, ms, M, N, K);
            }
            t_step_start = t_step_end;
        };
        
        if (M <= 0 || N <= 0 || K <= 0)
            return true;

        if (!d_C_int32)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_cached] Null INT32 buffer\n");
            return false;
        }

        // 1. Run CK INT8 GEMM: C_int32 = A_int8 × B_int8 (no scaling)
        if (!rocmQuantGemm_executeNoScale(d_A_int8, d_B_int8, d_C_int32, M, N, K, rocm_device_id, stream, kernel_ctx))
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_cached] CK NoScale GEMM failed\n");
            return false;
        }
        log_timing("CK_executeNoScale");

        // 2. Apply scales: E_fp32[m,n] = C_int32[m,n] * scale_A[m] * scale_B[n]
        const int total = M * N;
        const int block_size = 256;
        const int grid_size = (total + block_size - 1) / block_size;

        hipLaunchKernelGGL(
            applyScales_kernel,
            dim3(grid_size), dim3(block_size), 0, static_cast<hipStream_t>(stream),
            d_C_int32, d_E_fp32, d_scale_A, d_scale_B, M, N);

        (void)hipGetLastError();  // Clear stale errors
        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_cached] applyScales_kernel failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        log_timing("applyScales_kernel_launch");
        
        // Log total time
        auto t_total_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();
        if (total_ms > 1.0) {
            fprintf(stderr, "[CK_GEMM_TIMING] TOTAL: %.3f ms (M=%d N=%d K=%d)\n", total_ms, M, N, K);
        }

        return true;
    }

    // =========================================================================
    // M-PADDED TWO-KERNEL CK APPROACH (FOR DECODE: M < 8)
    // =========================================================================
    //
    // rocmQuantGemm_executeTwoKernel_padded - CK INT8 GEMM with M-padding
    //
    // This function handles the case where M < CK_SMALL_MIN_M (8) by:
    //   1. Allocating padded activation buffer [padded_m × K] (zero-initialized)
    //   2. Copying actual activations [M × K] into first M rows
    //   3. Creating padded scale buffer [padded_m] (actual scales + 1.0 padding)
    //   4. Running CK GEMM on padded dimensions [padded_m × N × K]
    //   5. Extracting first M rows of output to d_E_fp32 [M × N]
    //
    // WHY THIS APPROACH:
    //   - hipBLAS INT8 on gfx906 has severe limitation: N must be <= K
    //   - This breaks FFN Gate/Up projections where N > K (e.g., N=4864, K=896)
    //   - M-padding is more efficient: only pads activations (small for decode)
    //   - Small kernel (8×64 tiles) handles M=8 directly
    //
    // MEMORY OVERHEAD (much smaller with CK_SMALL_MIN_M=8!):
    //   - padded_m = 8 for decode (M=1 to M=7)
    //   - Extra allocation: (8-M) × K × sizeof(int8_t) + (8-M) × sizeof(float)
    //   - For M=1, K=896: 7 × 896 = ~6KB temporary (vs 111KB with padding to 128)
    //
    // =========================================================================
    bool rocmQuantGemm_executeTwoKernel_padded(
        const int8_t *d_A_int8,       // [M × K] INT8 activations (row-major)
        const int8_t *d_B_int8,       // [N × K] INT8 weights (row-major, viewed as [K × N] col-major)
        float *d_E_fp32,              // [M × N] FP32 output (only first M rows written)
        const float *d_scale_A,       // [M] per-row activation scales
        const float *d_scale_B,       // [N] per-column weight scales
        int32_t *d_C_int32,           // [padded_m × N] Pre-allocated INT32 accumulator
        int M,                        // Actual M (output rows needed)
        int padded_m,                 // Padded M for CK (>= CK_SMALL_MIN_M = 8)
        int N, int K,
        int rocm_device_id, void* stream,
        void* kernel_ctx)
    {
        if (M <= 0 || N <= 0 || K <= 0)
            return true;

        if (!d_C_int32)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] Null INT32 buffer\n");
            return false;
        }

        if (padded_m < M)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] padded_m (%d) < M (%d)\n", padded_m, M);
            return false;
        }

        hipError_t err;

        // 1. Allocate padded activation buffer [padded_m × K]
        int8_t *d_A_padded = nullptr;
        err = hipMalloc(&d_A_padded, static_cast<size_t>(padded_m) * K * sizeof(int8_t));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] Failed to allocate padded A: %s\n",
                    hipGetErrorString(err));
            return false;
        }

        // Zero-initialize the entire padded buffer (padding rows will be zeros)
        err = hipMemset(d_A_padded, 0, static_cast<size_t>(padded_m) * K * sizeof(int8_t));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] Failed to zero padded A: %s\n",
                    hipGetErrorString(err));
            hipFree(d_A_padded);
            return false;
        }

        // Copy actual activations into first M rows
        err = hipMemcpy(d_A_padded, d_A_int8, static_cast<size_t>(M) * K * sizeof(int8_t), hipMemcpyDeviceToDevice);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] Failed to copy A to padded: %s\n",
                    hipGetErrorString(err));
            hipFree(d_A_padded);
            return false;
        }

        // 2. Allocate padded scale buffer [padded_m]
        float *d_scale_A_padded = nullptr;
        err = hipMalloc(&d_scale_A_padded, static_cast<size_t>(padded_m) * sizeof(float));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] Failed to allocate padded scales: %s\n",
                    hipGetErrorString(err));
            hipFree(d_A_padded);
            return false;
        }

        // Copy actual scales and fill padding with 1.0 (neutral scale)
        err = hipMemcpy(d_scale_A_padded, d_scale_A, static_cast<size_t>(M) * sizeof(float), hipMemcpyDeviceToDevice);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] Failed to copy scales: %s\n",
                    hipGetErrorString(err));
            hipFree(d_A_padded);
            hipFree(d_scale_A_padded);
            return false;
        }

        // Fill padding scales with 1.0 (so padded rows have neutral scaling)
        if (padded_m > M)
        {
            const int padding_count = padded_m - M;
            const int fill_block = 256;
            const int fill_grid = (padding_count + fill_block - 1) / fill_block;
            hipLaunchKernelGGL(
                fillConstantFloat_kernel,
                dim3(fill_grid), dim3(fill_block), 0, static_cast<hipStream_t>(stream),
                d_scale_A_padded + M, padding_count, 1.0f);
            err = hipGetLastError();
            if (err != hipSuccess)
            {
                fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] Failed to fill padding scales: %s\n",
                        hipGetErrorString(err));
                hipFree(d_A_padded);
                hipFree(d_scale_A_padded);
                return false;
            }
        }

        // 3. Allocate padded output buffer [padded_m × N]
        float *d_E_padded = nullptr;
        err = hipMalloc(&d_E_padded, static_cast<size_t>(padded_m) * N * sizeof(float));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] Failed to allocate padded output: %s\n",
                    hipGetErrorString(err));
            hipFree(d_A_padded);
            hipFree(d_scale_A_padded);
            return false;
        }

        // 4. Run CK INT8 GEMM on padded dimensions: C_int32 = A_padded × B (no scaling)
        if (!rocmQuantGemm_executeNoScale(d_A_padded, d_B_int8, d_C_int32, padded_m, N, K, rocm_device_id, stream, kernel_ctx))
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] CK NoScale GEMM failed\n");
            hipFree(d_A_padded);
            hipFree(d_scale_A_padded);
            hipFree(d_E_padded);
            return false;
        }

        // 5. Apply scales to full padded output: E_padded[m,n] = C_int32[m,n] * scale_A_padded[m] * scale_B[n]
        const int total_padded = padded_m * N;
        const int block_size = 256;
        const int grid_size = (total_padded + block_size - 1) / block_size;

        hipLaunchKernelGGL(
            applyScales_kernel,
            dim3(grid_size), dim3(block_size), 0, static_cast<hipStream_t>(stream),
            d_C_int32, d_E_padded, d_scale_A_padded, d_scale_B, padded_m, N);

        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] applyScales_kernel failed: %s\n",
                    hipGetErrorString(err));
            hipFree(d_A_padded);
            hipFree(d_scale_A_padded);
            hipFree(d_E_padded);
            return false;
        }

        // 6. Extract first M rows of padded output to actual output
        //    E_fp32[M × N] = E_padded[0:M, :]
        err = hipMemcpy(d_E_fp32, d_E_padded, static_cast<size_t>(M) * N * sizeof(float), hipMemcpyDeviceToDevice);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded] Failed to extract output: %s\n",
                    hipGetErrorString(err));
            hipFree(d_A_padded);
            hipFree(d_scale_A_padded);
            hipFree(d_E_padded);
            return false;
        }

        // 7. Cleanup temporary buffers
        hipFree(d_A_padded);
        hipFree(d_scale_A_padded);
        hipFree(d_E_padded);

        return true;
    }

    // =========================================================================
    // M-PADDED TWO-KERNEL CK APPROACH WITH CACHED BUFFERS (ALLOCATION-FREE)
    // =========================================================================
    //
    // rocmQuantGemm_executeTwoKernel_padded_cached - Same as padded but with pre-allocated buffers
    //
    // This is the HOT-PATH version that uses pre-allocated padding buffers
    // passed by the caller, avoiding hipMalloc/hipFree which can take seconds on ROCm.
    //
    // The caller is responsible for ensuring the buffers are large enough:
    //   - d_A_padded: [padded_m × K] bytes
    //   - d_scale_A_padded: [padded_m] floats
    //   - d_E_padded: [padded_m × N] floats
    //
    // =========================================================================
    bool rocmQuantGemm_executeTwoKernel_padded_cached(
        const int8_t *d_A_int8,       // [M × K] INT8 activations (row-major)
        const int8_t *d_B_int8,       // [N × K] INT8 weights (row-major, viewed as [K × N] col-major)
        float *d_E_fp32,              // [M × N] FP32 output (only first M rows written)
        const float *d_scale_A,       // [M] per-row activation scales
        const float *d_scale_B,       // [N] per-column weight scales
        int32_t *d_C_int32,           // [padded_m × N] Pre-allocated INT32 accumulator
        int8_t *d_A_padded,           // [padded_m × K] Pre-allocated padded activations
        float *d_scale_A_padded,      // [padded_m] Pre-allocated padded scales
        float *d_E_padded,            // [padded_m × N] Pre-allocated padded output
        int M,                        // Actual M (output rows needed)
        int padded_m,                 // Padded M for CK (>= CK_SMALL_MIN_M = 8)
        int N, int K,
        int rocm_device_id, void* stream,
        void* kernel_ctx)
    {
        auto t_total_start = std::chrono::high_resolution_clock::now();
        auto t_step_start = t_total_start;
        auto t_step_end = t_total_start;
        
        auto log_timing = [&](const char* step) {
            t_step_end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t_step_end - t_step_start).count();
            if (ms > 0.5) {
                fprintf(stderr, "[CK_GEMM_TIMING] %s: %.3f ms (M=%d N=%d K=%d)\n", step, ms, M, N, K);
            }
            t_step_start = t_step_end;
        };
        
        if (M <= 0 || N <= 0 || K <= 0)
            return true;

        if (!d_C_int32 || !d_A_padded || !d_scale_A_padded || !d_E_padded)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded_cached] Null buffer\n");
            return false;
        }

        if (padded_m < M)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded_cached] padded_m (%d) < M (%d)\n", padded_m, M);
            return false;
        }

        hipError_t err;
        log_timing("param_validation");

        // 1. Zero-initialize the padded activation buffer (padding rows will be zeros)
        err = hipMemsetAsync(d_A_padded, 0, static_cast<size_t>(padded_m) * K * sizeof(int8_t), static_cast<hipStream_t>(stream));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded_cached] Failed to zero padded A: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        log_timing("hipMemsetAsync_A_padded");

        // Copy actual activations into first M rows (async)
        err = hipMemcpyAsync(d_A_padded, d_A_int8, static_cast<size_t>(M) * K * sizeof(int8_t), 
                             hipMemcpyDeviceToDevice, static_cast<hipStream_t>(stream));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded_cached] Failed to copy A to padded: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        log_timing("hipMemcpyAsync_A_to_padded");

        // 2. Copy actual scales and fill padding with 1.0 (neutral scale)
        err = hipMemcpyAsync(d_scale_A_padded, d_scale_A, static_cast<size_t>(M) * sizeof(float), 
                             hipMemcpyDeviceToDevice, static_cast<hipStream_t>(stream));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded_cached] Failed to copy scales: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        log_timing("hipMemcpyAsync_scales");

        // Fill padding scales with 1.0 (allocation-free, stream-ordered)
        if (padded_m > M)
        {
            const int padding_count = padded_m - M;
            const int fill_block = 256;
            const int fill_grid = (padding_count + fill_block - 1) / fill_block;
            hipLaunchKernelGGL(
                fillConstantFloat_kernel,
                dim3(fill_grid), dim3(fill_block), 0, static_cast<hipStream_t>(stream),
                d_scale_A_padded + M, padding_count, 1.0f);
            err = hipGetLastError();
            if (err != hipSuccess)
            {
                fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded_cached] Failed to fill padding scales: %s\n",
                        hipGetErrorString(err));
                return false;
            }
        }
        log_timing("fill_padding_scales");

        // 3. Run CK INT8 GEMM on padded dimensions: C_int32 = A_padded × B (no scaling)
        if (!rocmQuantGemm_executeNoScale(d_A_padded, d_B_int8, d_C_int32, padded_m, N, K, rocm_device_id, stream, kernel_ctx))
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded_cached] CK NoScale GEMM failed\n");
            return false;
        }
        log_timing("CK_executeNoScale");

        // 4. Apply scales to full padded output: E_padded[m,n] = C_int32[m,n] * scale_A_padded[m] * scale_B[n]
        const int total_padded = padded_m * N;
        const int block_size = 256;
        const int grid_size = (total_padded + block_size - 1) / block_size;

        hipLaunchKernelGGL(
            applyScales_kernel,
            dim3(grid_size), dim3(block_size), 0, static_cast<hipStream_t>(stream),
            d_C_int32, d_E_padded, d_scale_A_padded, d_scale_B, padded_m, N);

        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded_cached] applyScales_kernel failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        log_timing("applyScales_kernel_launch");

        // 5. Extract first M rows of padded output to actual output (async)
        //    E_fp32[M × N] = E_padded[0:M, :]
        err = hipMemcpyAsync(d_E_fp32, d_E_padded, static_cast<size_t>(M) * N * sizeof(float), 
                             hipMemcpyDeviceToDevice, static_cast<hipStream_t>(stream));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_padded_cached] Failed to extract output: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        log_timing("hipMemcpyAsync_extract_output");
        
        // Log total time
        auto t_total_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();
        if (total_ms > 1.0) {
            fprintf(stderr, "[CK_GEMM_TIMING] TOTAL: %.3f ms (M=%d N=%d K=%d)\n", total_ms, M, N, K);
        }

        // No cleanup needed - buffers are caller-owned!
        return true;
    }

    // =========================================================================
    // INT8 ALLOCATION HELPER
    // =========================================================================
    
    bool rocmQuantGemm_allocInt8(int8_t **d_ptr, size_t count, int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        hipError_t err = hipMalloc(d_ptr, count * sizeof(int8_t));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_allocInt8] hipMalloc failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    // =========================================================================
    // ACTIVATION QUANTIZATION (FP32 → INT8)
    // =========================================================================
    //
    // rocmQuantGemm_quantizeActivations - Per-row symmetric quantization
    //
    // ALGORITHM (per row m):
    //   1. Find max_abs = max(|A[m,k]|) for k in [0, K)
    //   2. scale_A[m] = max_abs / 127.0 (or 1.0 if max_abs == 0)
    //   3. A_int8[m,k] = round(A_fp32[m,k] / scale_A[m]) clamped to [-127, 127]
    //
    // WHY SYMMETRIC:
    //   - Zero-point = 0 simplifies GEMM (no offset correction needed)
    //   - Uses full [-127, 127] range (not [-128, 127] to stay symmetric)
    //   - Compatible with CK's INT8×INT8 GEMM
    //
    // EXECUTION:
    //   - Grid: (M, 1, 1) - one block per row
    //   - Block: shape-driven (128/256/512) from aspect ratio K/M
    //   - Fast-path dispatch:
    //       * NoTail when K % 4 == 0 (elides scalar tail loops)
    //       * LDS row staging when shape policy says rows are reusable
    //         (K <= policy.max_k and M >= policy.min_m)
    //   - Parallel reduction for max_abs finding
    //
    // =========================================================================

    bool rocmQuantGemm_quantizeActivations(
        const float *d_A_fp32,
        int8_t *d_A_int8,
        float *d_scales_A,
        int M, int K,
        int rocm_device_id, void* stream)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        if (M <= 0 || K <= 0)
            return true; // Nothing to do

        // Launch kernel: one block per row, thread count selected by shape class.
        // Policy is based on aspect ratio K/M (row width vs row count), not fixed dimensions.
        const int block_size = chooseQuantThreadsFromShape(M, K);
        const bool no_tail = ((K & 0x3) == 0);
        const QuantLdsPolicy lds_policy = getQuantLdsPolicy();
        const bool use_lds_row_staging = (K <= lds_policy.max_k) && (M >= lds_policy.min_m);
        dim3 grid(M, 1, 1);
        dim3 block(block_size, 1, 1);

        size_t shared_mem_bytes = use_lds_row_staging ? static_cast<size_t>(K) * sizeof(float) : 0;

        if (use_lds_row_staging)
        {
            if (no_tail)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(quantizeActivationsQ8_kernel_t<true, true>),
                    grid, block, shared_mem_bytes, static_cast<hipStream_t>(stream),
                    d_A_fp32, d_A_int8, d_scales_A, M, K);
            }
            else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(quantizeActivationsQ8_kernel_t<true, false>),
                    grid, block, shared_mem_bytes, static_cast<hipStream_t>(stream),
                    d_A_fp32, d_A_int8, d_scales_A, M, K);
            }
        }
        else
        {
            if (no_tail)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(quantizeActivationsQ8_kernel_t<false, true>),
                    grid, block, 0, static_cast<hipStream_t>(stream),
                    d_A_fp32, d_A_int8, d_scales_A, M, K);
            }
            else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(quantizeActivationsQ8_kernel_t<false, false>),
                    grid, block, 0, static_cast<hipStream_t>(stream),
                    d_A_fp32, d_A_int8, d_scales_A, M, K);
            }
        }

        // Check for launch errors
        (void)hipGetLastError();  // Clear stale errors
        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_quantizeActivations] Kernel launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }

        // No device-wide sync here; rely on stream ordering + a final sync/copy.

        return true;
    }

    // =========================================================================
    // MEMORY MANAGEMENT UTILITIES
    // =========================================================================
    //
    // These functions wrap HIP memory operations with error checking.
    // All device pointers must be allocated and freed on the same device.
    //
    // LIFECYCLE:
    //   1. rocmQuantGemm_allocFloat() - Allocate device memory
    //   2. rocmQuantGemm_copyHostToDevice() - Upload data
    //   3. [Run kernels...]
    //   4. rocmQuantGemm_copyDeviceToHost() - Download results
    //   5. rocmQuantGemm_freeDevice() - Free device memory
    //
    // =========================================================================

    void rocmQuantGemm_freeDevice(void *d_ptr, int rocm_device_id)
    {
        if (d_ptr)
        {
            // Set device before freeing - required by HIP/CUDA
            hipError_t set_err = static_cast<hipError_t>(llaminar2::HipDeviceGuard::setDevice(rocm_device_id));
            if (set_err == hipErrorDeinitialized || set_err == hipErrorNoDevice)
            {
                // HIP runtime is shutting down, skip cleanup
                // This is normal during static destruction at program exit
                return;
            }
            hipError_t err = hipFree(d_ptr);
            if (err != hipSuccess && err != hipErrorDeinitialized && err != hipErrorNoDevice)
            {
                // Only log actual errors, not shutdown-related ones
                fprintf(stderr, "WARNING: hipFree failed: %s\n", hipGetErrorString(err));
            }
        }
    }

    bool rocmQuantGemm_allocFloat(float **d_ptr, size_t count, int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        hipError_t err = hipMalloc(d_ptr, count * sizeof(float));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_allocFloat] hipMalloc failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool rocmQuantGemm_allocInt32(int32_t **d_ptr, size_t count, int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        hipError_t err = hipMalloc(d_ptr, count * sizeof(int32_t));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_allocInt32] hipMalloc failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool rocmQuantGemm_copyHostToDevice(float *d_dst, const float *h_src, size_t count, int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        hipError_t err = hipMemcpy(d_dst, h_src, count * sizeof(float), hipMemcpyHostToDevice);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_copyHostToDevice] hipMemcpy failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool rocmQuantGemm_copyDeviceToHost(float *h_dst, const float *d_src, size_t count, int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        hipError_t err = hipMemcpy(h_dst, d_src, count * sizeof(float), hipMemcpyDeviceToHost);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_copyDeviceToHost] hipMemcpy failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool rocmQuantGemm_copyInt32DeviceToHost(int32_t *h_dst, const int32_t *d_src, size_t count, int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        hipError_t err = hipMemcpy(h_dst, d_src, count * sizeof(int32_t), hipMemcpyDeviceToHost);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_copyInt32DeviceToHost] hipMemcpy failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    // =========================================================================
    // KERNEL-ONLY TIMED EXECUTION (FOR BENCHMARKING)
    // =========================================================================
    //
    // rocmQuantGemm_executeTwoKernel_timed - Same as _cached but returns kernel time
    //
    // Uses HIP events to measure ONLY the GPU kernel execution time, excluding:
    //   - Host-to-device data transfers
    //   - Device-to-host data transfers
    //   - CPU-side overhead
    //
    // RETURNS: Kernel execution time in milliseconds via kernel_time_ms parameter
    //
    // =========================================================================
    bool rocmQuantGemm_executeTwoKernel_timed(
        const int8_t *d_A_int8,       // [M × K] INT8 activations (already on device)
        const int8_t *d_B_int8,       // [K × N] INT8 weights (already on device)
        float *d_E_fp32,              // [M × N] FP32 output (on device)
        const float *d_scale_A,       // [M] per-row scales (on device)
        const float *d_scale_B,       // [N] per-column scales (on device)
        int32_t *d_C_int32,           // Pre-allocated [M × N] INT32 accumulator
        int M, int N, int K,
        int rocm_device_id,
        float *kernel_time_ms, void* stream,
        void* kernel_ctx)        // OUTPUT: kernel execution time in ms
    {
        if (M <= 0 || N <= 0 || K <= 0)
        {
            if (kernel_time_ms) *kernel_time_ms = 0.0f;
            return true;
        }

        if (!d_C_int32)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_timed] Null INT32 buffer\n");
            return false;
        }

        // Create HIP events for timing
        hipEvent_t start, stop;
        hipError_t err;
        
        err = hipEventCreate(&start);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_timed] Failed to create start event: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        
        err = hipEventCreate(&stop);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_timed] Failed to create stop event: %s\n",
                    hipGetErrorString(err));
            hipEventDestroy(start);
            return false;
        }

        // Record start event
        err = hipEventRecord(start, static_cast<hipStream_t>(stream));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_timed] Failed to record start event: %s\n",
                    hipGetErrorString(err));
            hipEventDestroy(start);
            hipEventDestroy(stop);
            return false;
        }

        // 1. Run CK INT8 GEMM: C_int32 = A_int8 × B_int8 (no scaling)
        if (!rocmQuantGemm_executeNoScale(d_A_int8, d_B_int8, d_C_int32, M, N, K, rocm_device_id, stream, kernel_ctx))
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_timed] CK NoScale GEMM failed\n");
            hipEventDestroy(start);
            hipEventDestroy(stop);
            return false;
        }

        // 2. Apply scales: E_fp32[m,n] = C_int32[m,n] * scale_A[m] * scale_B[n]
        const int total = M * N;
        const int block_size = 256;
        const int grid_size = (total + block_size - 1) / block_size;

        hipLaunchKernelGGL(
            applyScales_kernel,
            dim3(grid_size), dim3(block_size), 0, static_cast<hipStream_t>(stream),
            d_C_int32, d_E_fp32, d_scale_A, d_scale_B, M, N);

        err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_timed] applyScales_kernel failed: %s\n",
                    hipGetErrorString(err));
            hipEventDestroy(start);
            hipEventDestroy(stop);
            return false;
        }

        // Record stop event
        err = hipEventRecord(stop, static_cast<hipStream_t>(stream));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_timed] Failed to record stop event: %s\n",
                    hipGetErrorString(err));
            hipEventDestroy(start);
            hipEventDestroy(stop);
            return false;
        }

        // Wait for completion and get elapsed time
        err = hipEventSynchronize(stop);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_timed] Failed to sync stop event: %s\n",
                    hipGetErrorString(err));
            hipEventDestroy(start);
            hipEventDestroy(stop);
            return false;
        }

        float elapsed_ms = 0.0f;
        err = hipEventElapsedTime(&elapsed_ms, start, stop);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_timed] Failed to get elapsed time: %s\n",
                    hipGetErrorString(err));
            hipEventDestroy(start);
            hipEventDestroy(stop);
            return false;
        }

        if (kernel_time_ms)
            *kernel_time_ms = elapsed_ms;

        // Cleanup events
        hipEventDestroy(start);
        hipEventDestroy(stop);

        return true;
    }

    bool rocmQuantGemm_int8_int8_int32_vnni_prefill(
        const int8_t* d_A_int8,
        const int8_t* d_B_int8_vnni,
        int32_t* d_C_int32,
        int M, int N, int K,
        int tile_variant,
        int cpt,
        int rocm_device_id, void* stream)
    {
        if (!d_A_int8 || !d_B_int8_vnni || !d_C_int32 || M <= 0 || N <= 0 || K <= 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_int8_int32_vnni_prefill] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if ((K % 4) != 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_int8_int32_vnni_prefill] K must be multiple of 4 (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
        {
            return false;
        }

        const int variant = resolve_prefill_int8_tile_variant(tile_variant, M, N, K, false);
        const int cpt_v = (cpt == 2 || cpt == 4) ? cpt : 1;

        if (variant == PREFILL_TILE_32x8)
        {
            constexpr int BX = 32;
            constexpr int BY = 8;
            const dim3 block(BX, BY, 1);
            if (cpt_v == 4)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                                   dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
            else if (cpt_v == 2)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                                   dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
            else
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                                   dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
        }
        else if (variant == PREFILL_TILE_8x32)
        {
            constexpr int BX = 8;
            constexpr int BY = 32;
            const dim3 block(BX, BY, 1);
            if (cpt_v == 4)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                                   dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
            else if (cpt_v == 2)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                                   dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
            else
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                                   dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
        }
        else if (variant == PREFILL_TILE_8x8)
        {
            constexpr int BX = 8;
            constexpr int BY = 8;
            const dim3 block(BX, BY, 1);
            if (cpt_v == 4)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                                   dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
            else if (cpt_v == 2)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                                   dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
            else
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                                   dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
        }
        else
        {
            constexpr int BX = 16;
            constexpr int BY = 16;
            const dim3 block(BX, BY, 1);
            if (cpt_v == 4)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                                   dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
            else if (cpt_v == 2)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                                   dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
            else
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                                   dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
        }

        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_int8_int32_vnni_prefill] Kernel launch failed: %s (M=%d N=%d K=%d)\n",
                    hipGetErrorString(err), M, N, K);
            return false;
        }

        return true;
    }

    bool rocmQuantGemm_int8_int8_int32_vnni_prefill_grid_kpar(
        const int8_t* d_A_int8,
        const int8_t* d_B_int8_vnni,
        int32_t* d_C_int32,
        int M, int N, int K,
        int split_k_slices,
        int tile_variant,
        int cpt,
        int kernel_body_variant,
        int grid_swizzle_variant,
        int rocm_device_id, void* stream)
    {
        if (!d_A_int8 || !d_B_int8_vnni || !d_C_int32 || M <= 0 || N <= 0 || K <= 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_int8_int32_vnni_prefill_grid_kpar] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if ((K % 4) != 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_int8_int32_vnni_prefill_grid_kpar] K must be multiple of 4 (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        const int total_k_groups = K / 4;
        const int slices = max(1, min(split_k_slices, total_k_groups));
        const int k_groups_per_slice = (total_k_groups + slices - 1) / slices;

        if (!rocmQuantGemm_setDevice(rocm_device_id))
        {
            return false;
        }

        const int variant = resolve_prefill_int8_tile_variant(tile_variant, M, N, K, true);
        const int cpt_v = (cpt == 2 || cpt == 4) ? cpt : 1;
        const bool use_software_pipelined = (kernel_body_variant == 1 || kernel_body_variant == 2);
        const bool request_lds_b_tile = (kernel_body_variant == 2);
        const bool use_grid_swizzled = (grid_swizzle_variant == 1);

        {
            const size_t c_bytes = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(int32_t);
            hipError_t memset_err = hipMemsetAsync(d_C_int32, 0, c_bytes, static_cast<hipStream_t>(stream));
            if (memset_err != hipSuccess)
            {
                fprintf(stderr, "[rocmQuantGemm_int8_int8_int32_vnni_prefill_grid_kpar] hipMemsetAsync failed: %s\n", hipGetErrorString(memset_err));
                return false;
            }
        }

        if (variant == PREFILL_TILE_32x8)
        {
            constexpr int BX = 32;
            constexpr int BY = 8;
            const dim3 block(BX, BY, 1);
            if (cpt_v == 4)
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4));
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                const bool can_use_lds_b_tile = request_lds_b_tile && ((M % BY) == 0) && ((N % (BX * 4)) == 0);
                if (can_use_lds_b_tile)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, false, true>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
                else if (use_software_pipelined)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, true, false>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, false, false>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
            }
            else if (cpt_v == 2)
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2));
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2, false, true, false, false>),
                                   grid,
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
            }
            else
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX - 1) / BX);
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1, false, true, false, false>),
                                   grid,
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
            }
        }
        else if (variant == PREFILL_TILE_8x32)
        {
            constexpr int BX = 8;
            constexpr int BY = 32;
            const dim3 block(BX, BY, 1);
            if (cpt_v == 4)
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4));
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                const bool can_use_lds_b_tile = request_lds_b_tile && ((M % BY) == 0) && ((N % (BX * 4)) == 0);
                if (can_use_lds_b_tile)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, false, true>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
                else if (use_software_pipelined)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, true, false>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, false, false>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
            }
            else if (cpt_v == 2)
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2));
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2, false, true, false, false>),
                                   grid,
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
            }
            else
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX - 1) / BX);
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1, false, true, false, false>),
                                   grid,
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
            }
        }
        else if (variant == PREFILL_TILE_8x8)
        {
            constexpr int BX = 8;
            constexpr int BY = 8;
            const dim3 block(BX, BY, 1);
            if (cpt_v == 4)
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4));
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                const bool can_use_lds_b_tile = request_lds_b_tile && ((M % BY) == 0) && ((N % (BX * 4)) == 0);
                if (can_use_lds_b_tile)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, false, true>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
                else if (use_software_pipelined)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, true, false>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, false, false>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
            }
            else if (cpt_v == 2)
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2));
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2, false, true, false, false>),
                                   grid,
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
            }
            else
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX - 1) / BX);
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1, false, true, false, false>),
                                   grid,
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
            }
        }
        else
        {
            constexpr int BX = 16;
            constexpr int BY = 16;
            const dim3 block(BX, BY, 1);
            if (cpt_v == 4)
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4));
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                const bool can_use_lds_b_tile = request_lds_b_tile && ((M % BY) == 0) && ((N % (BX * 4)) == 0);
                if (can_use_lds_b_tile)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, false, true>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
                else if (use_software_pipelined)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, true, false>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, false, false>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
            }
            else if (cpt_v == 2)
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2));
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2, false, true, false, false>),
                                   grid,
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
            }
            else
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX - 1) / BX);
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1, false, true, false, false>),
                                   grid,
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
            }
        }

        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_int8_int32_vnni_prefill_grid_kpar] Kernel launch failed: %s (M=%d N=%d K=%d slices=%d)\n",
                    hipGetErrorString(err), M, N, K, slices);
            return false;
        }

        return true;
    }

    // =========================================================================
    // WIDE-TILE PREFILL DISPATCH
    // =========================================================================
    //
    // Forward declaration for parameterized N-tile dispatch.
    bool rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_ntile(
        const int8_t* d_A_int8,
        const int8_t* d_B_int8_vnni,
        int32_t* d_C_int32,
        int M, int N, int K,
        int n_tile_select,
        int rocm_device_id, void* stream);

    // For extreme-wide shapes (N >> K, M ≤ 128: LM_Head vocab projection).
    // Launches qgemm_int8_vnni_wide_tile_kernel<64, 8> which covers all M-rows
    // in a single block, eliminating B re-reads from HBM.
    //
    // =========================================================================
    bool rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile(
        const int8_t* d_A_int8,
        const int8_t* d_B_int8_vnni,
        int32_t* d_C_int32,
        int M, int N, int K,
        int rocm_device_id, void* stream)
    {
        return rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_ntile(
            d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, 64, rocm_device_id, stream);
    }

    // Parameterized N-tile dispatch for wide-tile V1 kernel.
    // n_tile_select: 64 (default) or 128 (wider tile, higher compute density, fewer blocks).
    bool rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_ntile(
        const int8_t* d_A_int8,
        const int8_t* d_B_int8_vnni,
        int32_t* d_C_int32,
        int M, int N, int K,
        int n_tile_select,
        int rocm_device_id, void* stream)
    {
        if (!d_A_int8 || !d_B_int8_vnni || !d_C_int32 || M <= 0 || N <= 0 || K <= 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if ((K % 4) != 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile] K must be multiple of 4 (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if (M > 128)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile] M must be ≤ 128 (got M=%d)\n", M);
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
        {
            return false;
        }

        constexpr int KT = 8;
        constexpr int BLOCK_SIZE = 256;

        if (n_tile_select == 128)
        {
            constexpr int N_TILE = 128;
            const unsigned n_tiles = static_cast<unsigned>((N + N_TILE - 1) / N_TILE);
            const dim3 grid(n_tiles, 1, 1);
            const dim3 block(BLOCK_SIZE, 1, 1);
            // N_TILE=128: 64 accumulators/thread, ~92 VGPRs → 2 waves/SIMD.
            // Benefit: half the grid blocks, 89% compute density (vs 84% for N=64).
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_vnni_wide_tile_kernel<N_TILE, KT, 0>),
                               grid, block, 0, static_cast<hipStream_t>(stream),
                               d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            constexpr int N_TILE = 64;
            const unsigned n_tiles = static_cast<unsigned>((N + N_TILE - 1) / N_TILE);
            const dim3 grid(n_tiles, 1, 1);
            const dim3 block(BLOCK_SIZE, 1, 1);
            // N_TILE=64: 32 accumulators/thread, 66 VGPRs → 3 waves/SIMD.
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_vnni_wide_tile_kernel<N_TILE, KT, 0>),
                               grid, block, 0, static_cast<hipStream_t>(stream),
                               d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }

        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile] Kernel launch failed: %s (M=%d N=%d K=%d ntile=%d)\n",
                    hipGetErrorString(err), M, N, K, n_tile_select);
            return false;
        }

        return true;
    }

    // =========================================================================
    // WIDE-TILE V2 DISPATCH (A from L2, B in LDS)
    // =========================================================================
    //
    // V2 variant reads A directly from L2 cache instead of cooperatively loading
    // to LDS.  Benefits:
    //   - Eliminates A LDS bank conflicts (4-way conflict in V1 stride-4 access)
    //   - Removes A cooperative load phase → faster per-iteration setup
    //   - Only 2 KB LDS (B tile only) vs 6 KB (V1 with A+B)
    //   - L1 handles broadcast for threads sharing same m_base
    //   - Naturally supports any KT (no multi-pass A loading needed)
    //
    // Best for shapes where A fits in L2 (A ≤ 4 MB → M*K ≤ 4M bytes → always
    // true for M=128, K up to 32K).
    //
    // kt_select: 8 → KT=8 (default), 16 → KT=16 (fewer sync barriers, more
    //            compute per tile, better for large K).
    //
    // =========================================================================
    bool rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_v2(
        const int8_t* d_A_int8,
        const int8_t* d_B_int8_vnni,
        int32_t* d_C_int32,
        int M, int N, int K,
        int kt_select,
        int rocm_device_id, void* stream)
    {
        if (!d_A_int8 || !d_B_int8_vnni || !d_C_int32 || M <= 0 || N <= 0 || K <= 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v2] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if ((K % 4) != 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v2] K must be multiple of 4 (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if (M > 128)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v2] M must be ≤ 128 (got M=%d)\n", M);
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
        {
            return false;
        }

        constexpr int N_TILE = 64;
        constexpr int BLOCK_SIZE = 256;

        const unsigned n_tiles = static_cast<unsigned>((N + N_TILE - 1) / N_TILE);
        const dim3 grid(n_tiles, 1, 1);
        const dim3 block(BLOCK_SIZE, 1, 1);

        if (kt_select == 16)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_vnni_wide_tile_v2_kernel<N_TILE, 16>),
                               grid, block, 0, static_cast<hipStream_t>(stream),
                               d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_vnni_wide_tile_v2_kernel<N_TILE, 8>),
                               grid, block, 0, static_cast<hipStream_t>(stream),
                               d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }

        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v2] Kernel launch failed: %s (M=%d N=%d K=%d kt=%d)\n",
                    hipGetErrorString(err), M, N, K, kt_select);
            return false;
        }

        return true;
    }

    // =========================================================================
    // WIDE-TILE V3 DISPATCH (LDS Double-Buffered Pipeline)
    // =========================================================================
    //
    // V3 variant uses LDS double-buffering to overlap global→LDS loads for
    // tile N+1 with compute on tile N.  Based on CK ISA analysis showing
    // this is the primary technique behind CK's 14% advantage over V1.
    //
    // Also fixes multi-pass A cooperative load for KT=16 (V1 only covers
    // 64 of 128 M-rows when KT=16).
    //
    // kt_select: 8 → KT=8 (3 waves/SIMD, 12 KB LDS),
    //           16 → KT=16 (2 workgroups/CU, 24 KB LDS, fewer iterations).
    //
    // =========================================================================
    bool rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_v3(
        const int8_t* d_A_int8,
        const int8_t* d_B_int8_vnni,
        int32_t* d_C_int32,
        int M, int N, int K,
        int kt_select,
        int rocm_device_id, void* stream)
    {
        if (!d_A_int8 || !d_B_int8_vnni || !d_C_int32 || M <= 0 || N <= 0 || K <= 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v3] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if ((K % 4) != 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v3] K must be multiple of 4 (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if (M > 128)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v3] M must be ≤ 128 (got M=%d)\n", M);
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
        {
            return false;
        }

        constexpr int N_TILE = 64;
        constexpr int BLOCK_SIZE = 256;

        const unsigned n_tiles = static_cast<unsigned>((N + N_TILE - 1) / N_TILE);
        const dim3 grid(n_tiles, 1, 1);
        const dim3 block(BLOCK_SIZE, 1, 1);

        if (kt_select == 16)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_vnni_wide_tile_v3_kernel<N_TILE, 16>),
                               grid, block, 0, static_cast<hipStream_t>(stream),
                               d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_vnni_wide_tile_v3_kernel<N_TILE, 8>),
                               grid, block, 0, static_cast<hipStream_t>(stream),
                               d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }

        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v3] Kernel launch failed: %s (M=%d N=%d K=%d kt=%d)\n",
                    hipGetErrorString(err), M, N, K, kt_select);
            return false;
        }

        return true;
    }

    // =========================================================================
    // WIDE-TILE V4 DISPATCH (N128 Double-Buffered Pipeline)
    // =========================================================================
    //
    // V4 targets FFN_Up/Gate shapes by matching CK's N128 tile geometry:
    //   - N_TILE = 128 (vs V1/V3's N_TILE = 64)
    //   - LDS double-buffering (same as V3)
    //   - Multi-pass B cooperative loading (new for N128)
    //
    // kt_select: 8  → KT=8  (16 KB LDS, 4 workgroups/CU, 2 waves/SIMD)
    //           16  → KT=16 (32 KB LDS, 2 workgroups/CU, 2 waves/SIMD)
    //
    // =========================================================================
    bool rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_v4(
        const int8_t* d_A_int8,
        const int8_t* d_B_int8_vnni,
        int32_t* d_C_int32,
        int M, int N, int K,
        int kt_select,
        int rocm_device_id, void* stream)
    {
        if (!d_A_int8 || !d_B_int8_vnni || !d_C_int32 || M <= 0 || N <= 0 || K <= 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v4] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if ((K % 4) != 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v4] K must be multiple of 4 (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if (M > 128)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v4] M must be ≤ 128 (got M=%d)\n", M);
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
        {
            return false;
        }

        constexpr int N_TILE = 128;
        constexpr int BLOCK_SIZE = 256;

        const unsigned n_tiles = static_cast<unsigned>((N + N_TILE - 1) / N_TILE);
        const dim3 grid(n_tiles, 1, 1);
        const dim3 block(BLOCK_SIZE, 1, 1);

        if (kt_select == 16)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_vnni_wide_tile_v4_kernel<N_TILE, 16>),
                               grid, block, 0, static_cast<hipStream_t>(stream),
                               d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_vnni_wide_tile_v4_kernel<N_TILE, 8>),
                               grid, block, 0, static_cast<hipStream_t>(stream),
                               d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }

        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v4] Kernel launch failed: %s (M=%d N=%d K=%d kt=%d)\n",
                    hipGetErrorString(err), M, N, K, kt_select);
            return false;
        }

        return true;
    }

    // =========================================================================
    // WIDE-TILE V5 DISPATCH (N128 ABBA Software-Pipelined)
    // =========================================================================
    //
    // V5 = V4 outer structure + ABBA inner loop + __launch_bounds__(256,2).
    // Targets 2 waves/SIMD (≤128 VGPRs) vs V4's 1 wave (256 VGPRs + spills).
    //
    // kt_select: 8  → KT=8  (16 KB LDS, 4 workgroups/CU, 2 waves/SIMD)
    //           16  → KT=16 (32 KB LDS, 2 workgroups/CU, 2 waves/SIMD)
    //
    // =========================================================================
    bool rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_v5(
        const int8_t* d_A_int8,
        const int8_t* d_B_int8_vnni,
        int32_t* d_C_int32,
        int M, int N, int K,
        int kt_select,
        int rocm_device_id, void* stream)
    {
        if (!d_A_int8 || !d_B_int8_vnni || !d_C_int32 || M <= 0 || N <= 0 || K <= 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v5] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if ((K % 4) != 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v5] K must be multiple of 4 (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if (M > 128)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v5] M must be ≤ 128 (got M=%d)\n", M);
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
        {
            return false;
        }

        constexpr int N_TILE = 128;
        constexpr int BLOCK_SIZE = 256;

        const unsigned n_tiles = static_cast<unsigned>((N + N_TILE - 1) / N_TILE);
        const dim3 grid(n_tiles, 1, 1);
        const dim3 block(BLOCK_SIZE, 1, 1);

        if (kt_select == 16)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_vnni_wide_tile_v5_kernel<N_TILE, 16>),
                               grid, block, 0, static_cast<hipStream_t>(stream),
                               d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_vnni_wide_tile_v5_kernel<N_TILE, 8>),
                               grid, block, 0, static_cast<hipStream_t>(stream),
                               d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }

        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v5] Kernel launch failed: %s (M=%d N=%d K=%d kt=%d)\n",
                    hipGetErrorString(err), M, N, K, kt_select);
            return false;
        }

        return true;
    }

    // =========================================================================
    // WIDE-TILE V6 DISPATCH (N128 Cross-Iteration Read-Compute Overlap)
    // =========================================================================
    //
    // V6 = V5 + #pragma clang loop unroll_count(2) on inner kk loop.
    // Targets CK-style ds_read/v_dot4 interleaving across kk steps.
    //
    // kt_select: 8  → KT=8  (16 KB LDS, 4 workgroups/CU, 2 waves/SIMD)
    //           16  → KT=16 (32 KB LDS, 2 workgroups/CU, 2 waves/SIMD)
    //
    // =========================================================================
    bool rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_v6(
        const int8_t* d_A_int8,
        const int8_t* d_B_int8_vnni,
        int32_t* d_C_int32,
        int M, int N, int K,
        int kt_select,
        int rocm_device_id, void* stream)
    {
        if (!d_A_int8 || !d_B_int8_vnni || !d_C_int32 || M <= 0 || N <= 0 || K <= 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v6] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if ((K % 4) != 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v6] K must be multiple of 4 (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if (M > 128)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v6] M must be ≤ 128 (got M=%d)\n", M);
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
        {
            return false;
        }

        constexpr int N_TILE = 128;
        constexpr int BLOCK_SIZE = 256;

        const unsigned n_tiles = static_cast<unsigned>((N + N_TILE - 1) / N_TILE);
        const dim3 grid(n_tiles, 1, 1);
        const dim3 block(BLOCK_SIZE, 1, 1);

        if (kt_select == 16)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_vnni_wide_tile_v6_kernel<N_TILE, 16>),
                               grid, block, 0, static_cast<hipStream_t>(stream),
                               d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_vnni_wide_tile_v6_kernel<N_TILE, 8>),
                               grid, block, 0, static_cast<hipStream_t>(stream),
                               d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }

        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v6] Kernel launch failed: %s (M=%d N=%d K=%d kt=%d)\n",
                    hipGetErrorString(err), M, N, K, kt_select);
            return false;
        }

        return true;
    }

    // =========================================================================
    // WIDE-TILE V7 DISPATCH (N128 Safe-Tile Split with Branchless Interior)
    // =========================================================================
    //
    // V7 = V5 outer structure + safe/boundary tile split.
    // Safe loop uses unconditional vectorized loads (zero boundary branches).
    // Boundary loop handles last 0-2 edge tiles with full checking.
    //
    // kt_select: 8  → KT=8  (16 KB LDS, 4 workgroups/CU, 2 waves/SIMD)
    //           16  → KT=16 (32 KB LDS, 2 workgroups/CU, 2 waves/SIMD)
    //
    // =========================================================================
    bool rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_v7(
        const int8_t* d_A_int8,
        const int8_t* d_B_int8_vnni,
        int32_t* d_C_int32,
        int M, int N, int K,
        int kt_select,
        int rocm_device_id, void* stream)
    {
        if (!d_A_int8 || !d_B_int8_vnni || !d_C_int32 || M <= 0 || N <= 0 || K <= 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v7] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if ((K % 4) != 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v7] K must be multiple of 4 (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if (M > 128)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v7] M must be ≤ 128 (got M=%d)\n", M);
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
        {
            return false;
        }

        constexpr int N_TILE = 128;
        constexpr int BLOCK_SIZE = 256;

        const unsigned n_tiles = static_cast<unsigned>((N + N_TILE - 1) / N_TILE);
        const dim3 grid(n_tiles, 1, 1);
        const dim3 block(BLOCK_SIZE, 1, 1);

        if (kt_select == 16)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_vnni_wide_tile_v7_kernel<N_TILE, 16>),
                               grid, block, 0, static_cast<hipStream_t>(stream),
                               d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_vnni_wide_tile_v7_kernel<N_TILE, 8>),
                               grid, block, 0, static_cast<hipStream_t>(stream),
                               d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }

        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v7] Kernel launch failed: %s (M=%d N=%d K=%d kt=%d)\n",
                    hipGetErrorString(err), M, N, K, kt_select);
            return false;
        }

        return true;
    }

    // =========================================================================
    // APPLY SCALING WITH FULL EPILOGUE (alpha, beta, bias)
    // =========================================================================
    //
    // rocmQuantGemm_applyScaling - Matches CUDA's cudaQuantGemm_applyScaling API
    //
    // Full-featured scaling epilogue for INT32→FP32 conversion:
    //   result = alpha * C_int32[m,n] * scale_A[m] * scale_B[n]
    //   if (beta != 0) result += beta * C_existing[m,n]
    //   if (bias)      result += bias[n]
    //
    // =========================================================================
    bool rocmQuantGemm_applyScaling(
        const int32_t *d_C_int32,    // [M×N] INT32 GEMM output
        float *d_C_fp32,             // [M×N] FP32 output
        const float *d_scales_A,     // [M] per-row scales
        const float *d_scales_B,     // [N] per-column scales
        int M, int N,
        float alpha, float beta,
        const float *d_C_existing,   // For beta != 0 (nullable)
        const float *d_bias,         // [N] optional bias (nullable)
        int rocm_device_id, void* stream)
    {
        // Validate required pointers
        if (!d_C_int32 || !d_C_fp32 || !d_scales_A || !d_scales_B)
        {
            fprintf(stderr, "[rocmQuantGemm_applyScaling] Null pointer: "
                    "d_C_int32=%p d_C_fp32=%p d_scales_A=%p d_scales_B=%p\n",
                    (void *)d_C_int32, (void *)d_C_fp32,
                    (void *)d_scales_A, (void *)d_scales_B);
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        // Decode-specialized fast path:
        // M=1 dominates INT8 VNNI decode and the generic 16x16 kernel wastes
        // 15/16 threads in this shape. Use a 1D vectorized kernel instead.
        if (M == 1 && beta == 0.0f && d_C_existing == nullptr && d_bias == nullptr)
        {
            const int block_size = 256;
            const int elems_per_thread = 4;
            const int grid_size = (N + block_size * elems_per_thread - 1) / (block_size * elems_per_thread);

            hipLaunchKernelGGL(
                applyScaling_m1_vec4_kernel,
                dim3(grid_size), dim3(block_size), 0, static_cast<hipStream_t>(stream),
                d_C_int32, d_C_fp32, d_scales_A, d_scales_B, N, alpha);

            (void)hipGetLastError();
            hipError_t err = hipGetLastError();
            if (err != hipSuccess)
            {
                fprintf(stderr, "[rocmQuantGemm_applyScaling] applyScaling_m1_vec4_kernel launch failed: %s "
                        "(M=%d, N=%d)\n", hipGetErrorString(err), M, N);
                return false;
            }
            return true;
        }

        // Use 16×16 thread blocks for 2D grid (matches CUDA)
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);

        hipLaunchKernelGGL(
            applyScaling_full_kernel,
            grid, block, 0, static_cast<hipStream_t>(stream),
            d_C_int32, d_C_fp32, d_scales_A, d_scales_B,
            M, N, alpha, beta, d_C_existing, d_bias);

        (void)hipGetLastError();  // Clear stale errors
        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_applyScaling] applyScaling_full_kernel launch failed: %s "
                    "(M=%d, N=%d)\n", hipGetErrorString(err), M, N);
            return false;
        }

        // No sync needed - coherence system handles sync when data is read
        return true;
    }

    // =========================================================================
    // IN-PLACE BIAS ADDITION KERNEL
    // =========================================================================
    //
    // Simple bias addition: output[m,n] += bias[n]
    // Broadcasts bias across rows.
    //
    // =========================================================================
    __global__ void biasAdd_kernel(float *output, const float *bias, int M, int N)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= M * N)
            return;
        
        const int col = idx % N;
        output[idx] += bias[col];
    }

    bool rocmQuantGemm_biasAdd(
        float *d_output,       // [M × N] FP32 output (modified in-place)
        const float *d_bias,   // [N] bias vector
        int M, int N,
        int rocm_device_id, void* stream)
    {
        if (!d_output || !d_bias)
        {
            fprintf(stderr, "[rocmQuantGemm_biasAdd] Null pointer: d_output=%p d_bias=%p\n",
                    (void *)d_output, (void *)d_bias);
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        const int total = M * N;
        const int block_size = 256;
        const int grid_size = (total + block_size - 1) / block_size;

        hipLaunchKernelGGL(
            biasAdd_kernel,
            dim3(grid_size), dim3(block_size), 0, static_cast<hipStream_t>(stream),
            d_output, d_bias, M, N);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_biasAdd] biasAdd_kernel launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }

        // No sync needed - coherence system handles sync when data is read
        return true;
    }

} // extern "C"
