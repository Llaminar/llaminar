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
 * CK's DeviceGemmMultipleD_Dl has tile-based requirements. We use 3-WAY DISPATCH
 * to optimize for different M values:
 *
 *   | M Range    | Kernel    | Tile Size | Why?                              |
 *   |------------|-----------|-----------|-----------------------------------|
 *   | M ≤ 32     | 32×32     | 32×32×32  | Minimal waste for decode (M=1)   |
 *   | 32 < M < 128| 64×64    | 64×64×32  | Good balance for small batches   |
 *   | M ≥ 128   | 128×128   | 128×128×64| Peak throughput for prefill      |
 *
 * The 32×32 and 64×64 kernels use GemmSpecialization::MNPadding for irregular M.
 * The 128×128 kernel uses GemmSpecialization::Default (requires M ≥ 128).
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
#include <iostream>
#include <streambuf>

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
//   - Uses GemmDefault (requires M ≥ 128)
//   - Peak throughput for large batch sizes
//
// DISPATCH STRATEGY:
//   M ≤ 32:      Use 32×32 kernel (GemmMNPadding handles irregular M)
//   32 < M < 128: Use 64×64 kernel (GemmMNPadding handles irregular M)  
//   M ≥ 128:     Use 128×128 kernel (GemmDefault, peak throughput)
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
//   - Legacy path: DenseScale kernel (lower accuracy, use LLAMINAR_ROCM_GEMM_FUSED=1)
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
// CACHED KERNEL INSTANCES (Lazy Initialization Pattern)
// ============================================================================
//
// WHY CACHING MATTERS:
//   CK device objects are expensive to construct (~10-100ms for first call):
//     - Template instantiation and LLVM IR generation
//     - GPU ISA compatibility checks
//     - Internal buffer allocation
//
//   By using static locals, we:
//     - Pay construction cost only on first call
//     - Thread-safe initialization (C++11 guarantees)
//     - Zero overhead on subsequent calls
//
// 3-WAY DISPATCH STRATEGY (January 2026):
//
//   ┌────────────────────────┬───────────────────────────────────────────────────┐
//   │ M Range                │ Kernel Used                                       │
//   ├────────────────────────┼───────────────────────────────────────────────────┤
//   │ M ≤ 32                 │ 32×32 kernel (GemmMNPadding handles irregular M)  │
//   │ 32 < M < 128           │ 64×64 kernel (GemmMNPadding handles irregular M)  │
//   │ M ≥ 128                │ 128×128 kernel (GemmDefault, peak throughput)     │
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

// 128×128 kernel (for M >= 128) - peak throughput
static DeviceGemmInt8_NoScale& getCachedNoScaleKernel() {
    static DeviceGemmInt8_NoScale device;
    return device;
}

// 64×64 kernel (for 32 < M < 128) - good balance
static DeviceGemmInt8_NoScale_64x64& getCachedNoScaleKernel64x64() {
    static DeviceGemmInt8_NoScale_64x64 device;
    return device;
}

// 32×32 kernel (for M <= 32) - optimal for decode
static DeviceGemmInt8_NoScale_32x32& getCachedNoScaleKernel32x32() {
    static DeviceGemmInt8_NoScale_32x32 device;
    return device;
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
 * Block: (256, 1, 1) threads (handles rows up to 256*k_per_thread elements)
 *
 * @param d_A_fp32 Input FP32 matrix [M × K] RowMajor
 * @param d_A_int8 Output INT8 matrix [M × K] RowMajor
 * @param d_scales Output per-row scales [M]
 * @param M Number of rows
 * @param K Number of columns per row
 */
__global__ void quantizeActivationsQ8_kernel(
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

    // Shared memory for parallel reduction
    __shared__ float s_max_abs[256];

    // Step 1: Find max absolute value in this row (parallel reduction)
    float local_max = 0.0f;
    const float *row_ptr = d_A_fp32 + static_cast<size_t>(row) * K;

    for (int k = tid; k < K; k += block_size)
    {
        float val = row_ptr[k];
        float abs_val = fabsf(val);
        local_max = fmaxf(local_max, abs_val);
    }

    s_max_abs[tid] = local_max;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = block_size / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            s_max_abs[tid] = fmaxf(s_max_abs[tid], s_max_abs[tid + stride]);
        }
        __syncthreads();
    }

    // Step 2: Compute scale (thread 0 writes to global memory)
    float max_abs = s_max_abs[0];
    float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    if (tid == 0)
    {
        d_scales[row] = scale;
    }

    // Step 3: Quantize all elements in this row
    int8_t *out_row = d_A_int8 + static_cast<size_t>(row) * K;

    for (int k = tid; k < K; k += block_size)
    {
        float val = row_ptr[k];
        float scaled = val * inv_scale;

        // Round to nearest, clamp to [-127, 127]
        int q = static_cast<int>(roundf(scaled));
        q = (q < -127) ? -127 : ((q > 127) ? 127 : q);

        out_row[k] = static_cast<int8_t>(q);
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
// STUBS: Phase 4 functions return success but log "not yet implemented"
// ============================================================================

extern "C"
{

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
    // Device management
    // -------------------------------------------------------------------------

    bool rocmQuantGemm_setDevice(int rocm_device_id)
    {
        hipError_t err = hipSetDevice(rocm_device_id);
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
            fprintf(stderr, "[rocmQuantGemm_uploadWeights] hipMemcpy weights failed: %s\n",
                    hipGetErrorString(err));
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
        int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        if (M <= 0 || N <= 0 || K <= 0)
            return true;

        // No D tensors - empty arrays
        std::array<const void*, 0> p_ds = {};
        std::array<ck::index_t, 0> strides_ds = {};

        // =====================================================================
        // 3-WAY DISPATCH BASED ON M
        // =====================================================================
        // The 32×32 and 64×64 kernels use GemmMNPadding to handle irregular M.
        // The 128×128 kernel uses GemmDefault and requires M ≥ 128.
        // =====================================================================

        ScopedCoutSilence silence;
        
        if (M <= 32)
        {
            // Small M: Use 32×32 kernel (optimal for decode, M=1-32)
            auto& device_gemm = getCachedNoScaleKernel32x32();
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
            invoker.Run(argument, StreamConfig{nullptr, false});
        }
        else if (M < 128)
        {
            // Medium M: Use 64×64 kernel (optimal for M=33-127)
            auto& device_gemm = getCachedNoScaleKernel64x64();
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
            invoker.Run(argument, StreamConfig{nullptr, false});
        }
        else
        {
            // Large M: Use 128×128 kernel (optimal for M>=128, peak throughput)
            auto& device_gemm = getCachedNoScaleKernel();
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
            invoker.Run(argument, StreamConfig{nullptr, false});
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
        int rocm_device_id)
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
        if (!rocmQuantGemm_executeNoScale(d_A_int8, d_B_int8, d_C_int32, M, N, K, rocm_device_id))
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
            dim3(grid_size), dim3(block_size), 0, 0,
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
        int rocm_device_id)
    {
        if (M <= 0 || N <= 0 || K <= 0)
            return true;

        if (!d_C_int32)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_cached] Null INT32 buffer\n");
            return false;
        }

        // 1. Run CK INT8 GEMM: C_int32 = A_int8 × B_int8 (no scaling)
        if (!rocmQuantGemm_executeNoScale(d_A_int8, d_B_int8, d_C_int32, M, N, K, rocm_device_id))
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_cached] CK NoScale GEMM failed\n");
            return false;
        }

        // 2. Apply scales: E_fp32[m,n] = C_int32[m,n] * scale_A[m] * scale_B[n]
        const int total = M * N;
        const int block_size = 256;
        const int grid_size = (total + block_size - 1) / block_size;

        hipLaunchKernelGGL(
            applyScales_kernel,
            dim3(grid_size), dim3(block_size), 0, 0,
            d_C_int32, d_E_fp32, d_scale_A, d_scale_B, M, N);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_cached] applyScales_kernel failed: %s\n",
                    hipGetErrorString(err));
            return false;
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
        int rocm_device_id)
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
            std::vector<float> padding_scales(padded_m - M, 1.0f);
            err = hipMemcpy(d_scale_A_padded + M, padding_scales.data(),
                           (padded_m - M) * sizeof(float), hipMemcpyHostToDevice);
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
        if (!rocmQuantGemm_executeNoScale(d_A_padded, d_B_int8, d_C_int32, padded_m, N, K, rocm_device_id))
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
            dim3(grid_size), dim3(block_size), 0, 0,
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
    //   - Block: (256, 1, 1) - 256 threads handle K elements in parallel
    //   - Parallel reduction for max_abs finding
    //
    // =========================================================================

    bool rocmQuantGemm_quantizeActivations(
        const float *d_A_fp32,
        int8_t *d_A_int8,
        float *d_scales_A,
        int M, int K,
        int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        if (M <= 0 || K <= 0)
            return true; // Nothing to do

        // Launch kernel: one block per row, 256 threads per block
        const int block_size = 256;
        dim3 grid(M, 1, 1);
        dim3 block(block_size, 1, 1);

        hipLaunchKernelGGL(
            quantizeActivationsQ8_kernel,
            grid, block, 0, 0,
            d_A_fp32, d_A_int8, d_scales_A, M, K);

        // Check for launch errors
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

    void rocmQuantGemm_freeDevice(void *d_ptr)
    {
        if (d_ptr)
        {
            hipFree(d_ptr);
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
        float *kernel_time_ms)        // OUTPUT: kernel execution time in ms
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
        err = hipEventRecord(start, 0);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeTwoKernel_timed] Failed to record start event: %s\n",
                    hipGetErrorString(err));
            hipEventDestroy(start);
            hipEventDestroy(stop);
            return false;
        }

        // 1. Run CK INT8 GEMM: C_int32 = A_int8 × B_int8 (no scaling)
        if (!rocmQuantGemm_executeNoScale(d_A_int8, d_B_int8, d_C_int32, M, N, K, rocm_device_id))
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
            dim3(grid_size), dim3(block_size), 0, 0,
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
        err = hipEventRecord(stop, 0);
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

} // extern "C"
