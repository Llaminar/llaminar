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
 *   - rocmQuantGemm_executeTwoKernel()   : DEFAULT - CK GEMM + separate scaling
 *   - rocmQuantGemm_executeDenseScale()  : Legacy - CK with fused scaling
 *   - rocmQuantGemm_executeHipBLAS()     : Fallback for M < 64
 *   - rocmQuantGemm_quantizeActivations(): FP32→INT8 activation quantization
 *   - rocmQuantGemm_uploadWeights()      : Upload weights to GPU
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
 * CK's DeviceGemmMultipleD_Dl has STRICT minimum size requirements:
 *
 *   | Dimension | Minimum | Why?                                        |
 *   |-----------|---------|---------------------------------------------|
 *   | M (rows)  | 64      | MPerBlock = 64, need at least 1 block       |
 *   | N (cols)  | 64      | NPerBlock = 64, need at least 1 block       |
 *   | K (inner) | 32      | K0PerBlock × K1 = 8 × 4 = 32               |
 *
 * If you pass smaller dimensions, IsSupportedArgument() returns false.
 * The kernel does NOT automatically pad dimensions.
 *
 * For decode (M=1), you must either:
 *   - Pad M to 64 (wastes GPU but works)
 *   - Use a different kernel (CPU, or future small-tile CK config)
 *
 * ============================================================================
 * MEMORY LAYOUT CONVENTION (CRITICAL!)
 * ============================================================================
 *
 * This implementation uses **Row-Major layout for ALL matrices**. This follows
 * CK's `mk_kn_mn` naming convention:
 *
 * Matrix A (activations): [M × K] Row-Major
 *   - Element A[m,k] is at memory offset: m * K + k
 *   - Stride = K (number of columns)
 *   - Example: A[2,3] with K=5 is at offset 2*5 + 3 = 13
 *
 * Matrix B (weights): [K × N] Row-Major
 *   - Element B[k,n] is at memory offset: k * N + n
 *   - Stride = N (number of columns)
 *   - NOTE: This is the TRANSPOSE of original model weights [N×K]
 *   - We transpose during weight packing in ROCmQuantisedGemmKernel.cpp
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
 *   | mk_kn_mn  | Row, Row, Row    | ← WE USE THIS                  |
 *   | km_kn_mn  | Col, Row, Row    | Would require A transposed     |
 *   | mk_nk_mn  | Row, Col, Row    | Would require B transposed     |
 *
 * CRITICAL: Using tile parameters from the wrong layout configuration will
 * produce incorrect numerical results WITHOUT any error messages or exceptions.
 *
 * Our tile parameters are copied from:
 *   external/composable_kernel/library/src/tensor_operation_instance/gpu/
 *       gemm_universal/device_gemm_dl_i8_i8_i8_mk_kn_mn_instance.cpp
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
#include <hipblas/hipblas.h>
#include <cstdint>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <streambuf>

// ============================================================================
// HIPBLAS HANDLE MANAGEMENT
// ============================================================================
// Thread-local hipBLAS handle for INT8 GEMM on small M (M < 64)
// Used as fallback when CK's minimum dimension requirements aren't met.

namespace {

thread_local hipblasHandle_t g_hipblas_handle = nullptr;
thread_local int g_hipblas_device_id = -1;

hipblasHandle_t getHipblasHandle(int device_id)
{
    if (g_hipblas_handle && g_hipblas_device_id == device_id)
        return g_hipblas_handle;
    
    // Destroy old handle if device changed
    if (g_hipblas_handle)
    {
        hipblasDestroy(g_hipblas_handle);
        g_hipblas_handle = nullptr;
    }
    
    // Set device and create handle
    if (hipSetDevice(device_id) != hipSuccess)
        return nullptr;
    
    if (hipblasCreate(&g_hipblas_handle) != HIPBLAS_STATUS_SUCCESS)
    {
        g_hipblas_handle = nullptr;
        return nullptr;
    }
    
    g_hipblas_device_id = device_id;
    return g_hipblas_handle;
}

} // anonymous namespace

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
// Tile parameters copied from CK's official instance file for Row,Row,Row layout:
//   https://github.com/ROCm/composable_kernel/blob/develop/library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_dl_i8_i8_i8_mk_kn_mn_instance.cpp
//
// The "mk_kn_mn" suffix indicates: A[M,K]=Row, B[K,N]=Row, C[M,N]=Row
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
// These values come from the tile configuration:
//   - MPerBlock = 64 (BlockSize for M dimension)
//   - NPerBlock = 64 (BlockSize for N dimension)
//   - K0PerBlock × K1 = 8 × 4 = 32 (Total K per block)
//
// For decode (M=1), this kernel CANNOT be used. Options:
//   1. Pad M to 64 (wasteful but works)
//   2. Use CPU fallback for decode
//   3. Implement small-tile config (future work)
//
constexpr int CK_MIN_M = 64;   // MPerBlock requirement
constexpr int CK_MIN_N = 64;   // NPerBlock requirement  
constexpr int CK_MIN_K = 32;   // K0PerBlock × K1 = 8 × 4

// Helper function to check if dimensions meet CK requirements
inline bool ck_dimensions_supported(int M, int N, int K)
{
    return M >= CK_MIN_M && N >= CK_MIN_N && K >= CK_MIN_K;
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
// Our configuration uses Row-Major for everything (standard C/C++ convention):
//   A[M×K] row-major: A[m,k] at m*K + k, Stride=K
//   B[K×N] row-major: B[k,n] at k*N + n, Stride=N (NOTE: transposed from model [N×K])
//   E[M×N] row-major: E[m,n] at m*N + n, Stride=N
//
using ALayout  = Row;   // A[M×K] RowMajor - activations
using BLayout  = Row;   // B[K×N] RowMajor - transposed weights
using ELayout  = Row;   // E[M×N] RowMajor - output

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// ============================================================================
// DENSE SCALE KERNEL TYPE (LEGACY - NOT DEFAULT)
// ============================================================================
//
// ⚠️ WARNING: This kernel has ~0.8% lower accuracy than Two-Kernel approach!
//    Cosine similarity: ~0.9917 vs ~0.9999 for Two-Kernel/hipBLAS
//    Use LLAMINAR_ROCM_GEMM_FUSED=1 to enable (not recommended)
//
// This kernel attempts to fuse scaling into the GEMM via CK's D-tensor.
//
// How it works:
//   1. Pre-compute combined_scale[m,n] = scale_A[m] * scale_B[n]
//   2. Run GEMM with fused scaling: E[m,n] = C_int32[m,n] * combined_scale[m,n]
//
// Why fused scaling has lower accuracy:
//   CK's CDEElementOp applies scaling during output writeback. The INT32→FP32
//   conversion and float multiplication happen in a single pass. This can
//   accumulate rounding errors differently than the two-kernel approach where
//   INT32 accumulator values are written exactly, then scaled.
//
// Why this design exists (historical context):
//   - Originally designed to avoid a second kernel launch overhead
//   - CK's broadcast D-tensors (stride=0) had bugs, so we use dense [M×N] scale
//   - Pre-compute overhead is negligible (~0.1ms for 128×128)
//
// Formula:
//   E[m,n] = (A_int8[m,:] × B_int8[:,n]) * combined_scale[m,n]
//          = (A_int8[m,:] × B_int8[:,n]) * scale_A[m] * scale_B[n]
//
// PREFER: rocmQuantGemm_executeTwoKernel() for production (same speed, better accuracy)
//
// ============================================================================
using DsDataType_Dense = ck::Tuple<F32>;    // Single [M×N] scale matrix
using EDataType_Dense  = F32;               // Output is FP32
using DsLayout_Dense   = ck::Tuple<Row>;    // Row-major scale matrix [M×N]

// Element-wise operation for single scale tensor
// Called by CK for each output element: e = static_cast<float>(c) * d
struct ScaleSingle
{
    ScaleSingle() = default;

    // CK calls this for each output element e[m,n]:
    //   c = INT32 accumulator (result of A×B)
    //   d = combined_scale[m,n] = scale_A[m] * scale_B[n]
    //   e = FP32 output
    template <typename E, typename C, typename D>
    __host__ __device__ constexpr void operator()(E& e, const C& c, const D& d) const
    {
        e = static_cast<E>(c) * d;
    }
};

using CDEElementOp_Dense = ScaleSingle;

// ============================================================================
// DENSE SCALE KERNEL TEMPLATE INSTANTIATION
// ============================================================================
//
// TEMPLATE PARAMETER REFERENCE (for debugging/modification):
//
// Layout parameters:
//   - ALayout: Row       = A matrix is row-major [M×K]
//   - BLayout: Row       = B matrix is row-major [K×N]  
//   - DsLayout: Tuple<Row> = D tensor is row-major [M×N]
//   - ELayout: Row       = Output is row-major [M×N]
//
// Data type parameters:
//   - ADataType: int8_t  = 8-bit signed activations
//   - BDataType: int8_t  = 8-bit signed weights
//   - AccDataType: int32_t = 32-bit accumulator (prevents overflow)
//   - DsDataType: Tuple<float> = 32-bit scale values
//   - EDataType: float   = 32-bit output
//
// Tile configuration (from CK's mk_kn_mn instance):
//   - BlockSize: 64      = 64 threads per thread block
//   - MPerBlock: 64      = Each block computes 64 rows of output
//   - NPerBlock: 64      = Each block computes 64 columns of output
//   - K0PerBlock: 8      = Outer K loop iterations per block
//   - K1: 4              = Inner K loop size (8×4 = 32 K per iteration)
//   - M1PerThread: 4     = Each thread computes 4 output rows
//   - N1PerThread: 4     = Each thread computes 4 output columns
//   - KPerThread: 1      = K elements per thread per iteration
//
// Block transfer parameters (advanced - rarely need modification):
//   - ABlockTransfer*: How threads cooperatively load A tile to LDS
//   - BBlockTransfer*: How threads cooperatively load B tile to LDS
//   - CThreadTransfer*: How threads write output to global memory
//
// clang-format off
using DeviceGemmInt8_DenseScale = ck::tensor_operation::device::DeviceGemmMultipleD_Dl<
    ALayout,              // A layout: RowMajor [M×K]
    BLayout,              // B layout: RowMajor [K×N]
    DsLayout_Dense,       // D tensor layout: Tuple<Row> for dense [M×N] scale
    ELayout,              // E layout: RowMajor [M×N]
    ADataType,            // A element type: int8_t
    BDataType,            // B element type: int8_t
    AccDataType,          // Accumulator type: int32_t
    DsDataType_Dense,     // D element types: Tuple<float>
    EDataType_Dense,      // E element type: float
    AElementOp,           // A element-wise op: PassThrough
    BElementOp,           // B element-wise op: PassThrough
    CDEElementOp_Dense,   // Output element-wise op: ScaleSingle
    GemmDefault,          // GemmSpecialization::Default
    // =========================================================================
    // BLOCK AND TILE CONFIGURATION
    // These values define the work distribution across GPU threads.
    // DO NOT modify unless you understand CK's tile mapping constraints!
    // =========================================================================
    64,                   // BlockSize: 64 threads per block (8×8 thread tile)
    64,                   // MPerBlock: 64 output rows per block (MINIMUM M!)
    64,                   // NPerBlock: 64 output cols per block (MINIMUM N!)
    8,                    // K0PerBlock: K outer loop (8 iterations)
    4,                    // K1: K inner loop (8×4=32, MINIMUM K!)
    4,                    // M1PerThread: 4 output rows per thread
    4,                    // N1PerThread: 4 output cols per thread
    1,                    // KPerThread: 1 K element per thread per iter
    S<4, 2>,              // M1N1ThreadClusterM1Xs: thread cluster M tiling
    S<4, 2>,              // M1N1ThreadClusterN1Xs: thread cluster N tiling
    // =========================================================================
    // A MATRIX BLOCK TRANSFER (loading A tile from global to LDS)
    // Format: S<K0, M0, M1, K1> - how threads slice and load A
    // =========================================================================
    S<4, 1, 2, 4>,        // ABlockTransferThreadSliceLengths_K0_M0_M1_K1
    S<2, 1, 32, 1>,       // ABlockTransferThreadClusterLengths_K0_M0_M1_K1
    S<1, 2, 0, 3>,        // ABlockTransferThreadClusterArrangeOrder
    S<1, 2, 0, 3>,        // ABlockTransferSrcAccessOrder
    S<4, 1, 1, 4>,        // ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1
    S<1, 2, 0, 3>,        // ABlockTransferSrcVectorTensorContiguousDimOrder
    S<1, 1, 1, 4>,        // ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1
    // =========================================================================
    // B MATRIX BLOCK TRANSFER (loading B tile from global to LDS)
    // Format: S<K0, N0, N1, K1> - how threads slice and load B
    // NOTE: These are layout-specific! Must match Row,Row,Row (mk_kn_mn).
    // =========================================================================
    S<2, 1, 4, 4>,        // BBlockTransferThreadSliceLengths_K0_N0_N1_K1
    S<4, 1, 16, 1>,       // BBlockTransferThreadClusterLengths_K0_N0_N1_K1
    S<0, 3, 1, 2>,        // BBlockTransferThreadClusterArrangeOrder
    S<0, 3, 1, 2>,        // BBlockTransferSrcAccessOrder
    S<1, 1, 4, 1>,        // BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1
    S<0, 3, 1, 2>,        // BBlockTransferSrcVectorTensorContiguousDimOrder
    S<1, 1, 4, 4>,        // BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1
    // =========================================================================
    // OUTPUT (C/E) THREAD TRANSFER (writing result to global memory)
    // =========================================================================
    S<0, 1, 2, 3, 4, 5>,  // CThreadTransferSrcDstAccessOrder
    5,                    // CThreadTransferSrcDstVectorDim (vectorize dim 5)
    4>;                   // CThreadTransferDstScalarPerVector (4-wide writes)
// clang-format on

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
    BLayout,              // B layout: RowMajor [K×N]
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
    // Same tile configuration as other kernels for consistent behavior
    64,                   // BlockSize: 64 threads per block
    64,                   // MPerBlock: 64 output rows per block (MINIMUM M!)
    64,                   // NPerBlock: 64 output cols per block (MINIMUM N!)
    8,                    // K0PerBlock: K outer loop (8 iterations)
    4,                    // K1: K inner loop (8×4=32, MINIMUM K!)
    4,                    // M1PerThread: 4 output rows per thread
    4,                    // N1PerThread: 4 output cols per thread  
    1,                    // KPerThread: 1 K element per thread per iteration
    S<4, 2>,              // M1N1ThreadClusterM1Xs: thread cluster M tiling
    S<4, 2>,              // M1N1ThreadClusterN1Xs: thread cluster N tiling
    S<4, 1, 2, 4>,        // ABlockTransferThreadSliceLengths_K0_M0_M1_K1
    S<2, 1, 32, 1>,       // ABlockTransferThreadClusterLengths_K0_M0_M1_K1
    S<1, 2, 0, 3>,        // ABlockTransferThreadClusterArrangeOrder
    S<1, 2, 0, 3>,        // ABlockTransferSrcAccessOrder
    S<4, 1, 1, 4>,        // ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1
    S<1, 2, 0, 3>,        // ABlockTransferSrcVectorTensorContiguousDimOrder
    S<1, 1, 1, 4>,        // ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1
    S<2, 1, 4, 4>,        // BBlockTransferThreadSliceLengths_K0_N0_N1_K1
    S<4, 1, 16, 1>,       // BBlockTransferThreadClusterLengths_K0_N0_N1_K1
    S<0, 3, 1, 2>,        // BBlockTransferThreadClusterArrangeOrder
    S<0, 3, 1, 2>,        // BBlockTransferSrcAccessOrder
    S<1, 1, 4, 1>,        // BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1
    S<0, 3, 1, 2>,        // BBlockTransferSrcVectorTensorContiguousDimOrder
    S<1, 1, 4, 4>,        // BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1
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
//   By using static locals, we:
//     - Pay construction cost only on first call
//     - Thread-safe initialization (C++11 guarantees)
//     - Zero overhead on subsequent calls
//
// AVAILABLE KERNELS:
//   ┌──────────────────┬─────────────────────────────────────────────────────┐
//   │ Kernel           │ Use Case                                            │
//   ├──────────────────┼─────────────────────────────────────────────────────┤
//   │ NoScale          │ Two-Kernel default: INT8→INT32 GEMM only           │
//   │ DenseScale       │ Legacy fused: INT8→FP32 with pre-computed scales   │
//   └──────────────────┴─────────────────────────────────────────────────────┘
//
// ============================================================================

static DeviceGemmInt8_DenseScale& getCachedDenseScaleKernel() {
    static DeviceGemmInt8_DenseScale device;
    return device;
}

static DeviceGemmInt8_NoScale& getCachedNoScaleKernel() {
    static DeviceGemmInt8_NoScale device;
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
// HIP Kernel: Compute Combined Scale Matrix
// ============================================================================
//
// Computes combined_scale[m,n] = scale_A[m] * scale_B[n] for dense D-tensor approach.
// This is used as a workaround for CK's DL kernel broadcast stride=0 bug.
//
// Grid: ((M*N + 255) / 256, 1, 1) blocks
// Block: (256, 1, 1) threads
//
__global__ void computeCombinedScales_kernel(
    const float* __restrict__ d_scale_A,   // [M] per-row scales
    const float* __restrict__ d_scale_B,   // [N] per-column scales  
    float* __restrict__ d_combined_scale,  // [M×N] output scale matrix
    int M, int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    
    if (idx >= total) return;
    
    const int m = idx / N;
    const int n = idx % N;
    
    d_combined_scale[idx] = d_scale_A[m] * d_scale_B[n];
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

    int rocmQuantGemm_getMinM() { return CK_MIN_M; }
    int rocmQuantGemm_getMinN() { return CK_MIN_N; }
    int rocmQuantGemm_getMinK() { return CK_MIN_K; }

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
    // LEGACY ENTRY POINT: Dense Scale GEMM (NOT DEFAULT)
    // =========================================================================
    //
    // ⚠️ NOTE: This is NOT the default path. Use rocmQuantGemm_executeTwoKernel()
    //    for best accuracy. Enable this with LLAMINAR_ROCM_GEMM_FUSED=1.
    //
    // rocmQuantGemm_executeDenseScale - INT8×INT8→FP32 with fused per-row/col scaling
    //
    // ALGORITHM:
    //   Step 1: Compute combined_scale[m,n] = scale_A[m] * scale_B[n]  (GPU kernel)
    //   Step 2: Run CK GEMM: E[m,n] = sum_k(A[m,k] × B[k,n]) * combined_scale[m,n]
    //
    // ACCURACY WARNING:
    //   This approach has ~0.8% lower cosine similarity vs FP32 reference.
    //   The fused D-tensor scaling accumulates rounding errors differently.
    //   Prefer rocmQuantGemm_executeTwoKernel() for production inference.
    //
    // PARAMETERS:
    //   d_A_int8        - [M×K] INT8 activation matrix (row-major, device pointer)
    //   d_weights_int8  - [K×N] INT8 weight matrix (row-major, device pointer)
    //   d_C_fp32        - [M×N] FP32 output matrix (row-major, device pointer)
    //   d_scales_A      - [M] per-row activation scales (device pointer)
    //   d_scales_B      - [N] per-column weight scales (device pointer)
    //   M, N, K         - Matrix dimensions (MINIMUM: M≥64, N≥64, K≥32)
    //   rocm_device_id  - GPU device ID to execute on
    //   d_work_buffer   - Optional pre-allocated [M×N] float buffer (or nullptr)
    //   work_buffer_size- Size of work buffer in bytes (0 if nullptr)
    //
    // RETURNS:
    //   true on success, false on error (check stderr for details)
    //
    // MEMORY:
    //   - If d_work_buffer is provided and large enough, uses it for combined_scale
    //   - Otherwise allocates temporary [M×N×4] bytes and frees after execution
    //
    // EXAMPLE:
    //   // Prefill: M=128, K=896, N=1024
    //   bool ok = rocmQuantGemm_executeDenseScale(
    //       d_A, d_B, d_C, d_scale_A, d_scale_B,
    //       128, 1024, 896, 0, nullptr, 0);
    //
    // =========================================================================
    bool rocmQuantGemm_executeDenseScale(
        const int8_t *d_A_int8,
        const int8_t *d_weights_int8,
        float *d_C_fp32,
        const float *d_scales_A,
        const float *d_scales_B,
        int M, int N, int K,
        int rocm_device_id,
        float *d_work_buffer,       // Optional: pre-allocated [M×N] buffer
        size_t work_buffer_size)    // Size of work buffer in bytes
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        if (M <= 0 || N <= 0 || K <= 0)
            return true;

        // Early-exit: CK requires minimum dimensions (tile-based computation)
        if (!ck_dimensions_supported(M, N, K))
        {
            fprintf(stderr, "[rocmQuantGemm_executeDenseScale] Dimensions too small for CK: "
                    "M=%d (min=%d), N=%d (min=%d), K=%d (min=%d)\n",
                    M, CK_MIN_M, N, CK_MIN_N, K, CK_MIN_K);
            return false;
        }

        // Check if we need to allocate the combined scale buffer
        float* d_combined_scale = nullptr;
        bool own_buffer = false;
        size_t required_size = static_cast<size_t>(M) * N * sizeof(float);
        
        if (d_work_buffer && work_buffer_size >= required_size)
        {
            d_combined_scale = d_work_buffer;
        }
        else
        {
            hipError_t err = hipMalloc(&d_combined_scale, required_size);
            if (err != hipSuccess)
            {
                fprintf(stderr, "[rocmQuantGemm_executeDenseScale] Failed to allocate combined scale buffer: %s\n",
                        hipGetErrorString(err));
                return false;
            }
            own_buffer = true;
        }

        // Launch kernel to compute combined_scale[m,n] = scale_A[m] * scale_B[n]
        const int total_elements = M * N;
        const int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;
        
        computeCombinedScales_kernel<<<grid_size, block_size>>>(
            d_scales_A, d_scales_B, d_combined_scale, M, N);
        
        // Check for kernel launch errors
        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeDenseScale] Combined scale kernel failed: %s\n",
                    hipGetErrorString(err));
            if (own_buffer) hipFree(d_combined_scale);
            return false;
        }

        // Now run GEMM with dense scale matrix
        auto& device_gemm = getCachedDenseScaleKernel();

        std::array<const void*, 1> p_ds = {d_combined_scale};
        std::array<ck::index_t, 1> strides_ds = {N};  // Row-major [M×N] stride

        auto argument = device_gemm.MakeArgument(
            d_A_int8,                   // p_a: [M×K] RowMajor
            d_weights_int8,             // p_b: [K×N] RowMajor
            p_ds,                       // p_ds: {combined_scale}
            d_C_fp32,                   // p_e: [M×N] RowMajor FP32 output
            M, N, K,                    // dimensions
            K,                          // StrideA = K
            N,                          // StrideB = N
            strides_ds,                 // StrideDs: {N}
            N,                          // StrideE = N
            AElementOp{},
            BElementOp{},
            CDEElementOp_Dense{}
        );

        if (!device_gemm.IsSupportedArgument(argument))
        {
            fprintf(stderr, "[rocmQuantGemm_executeDenseScale] CK does not support M=%d N=%d K=%d\n",
                    M, N, K);
            if (own_buffer) hipFree(d_combined_scale);
            return false;
        }

        // Silence CK's unconditional std::cout spam
        ScopedCoutSilence silence;
        auto invoker = device_gemm.MakeInvoker();
        invoker.Run(argument, StreamConfig{nullptr, false});
        
        if (own_buffer) hipFree(d_combined_scale);
        return true;
    }

    // =========================================================================
    // DIMENSION VALIDATION
    // =========================================================================
    //
    // rocmQuantGemm_isSupportedDimensions - Check if CK supports given M/N/K
    //
    // CK has MINIMUM dimension requirements based on tile size:
    //   - M ≥ 64 (MPerBlock = 64)
    //   - N ≥ 64 (NPerBlock = 64)  
    //   - K ≥ 32 (K0PerBlock × K1 = 8 × 4 = 32)
    //
    // USAGE:
    //   if (!rocmQuantGemm_isSupportedDimensions(M, N, K)) {
    //       // Options: pad inputs, use different kernel, or fail
    //   }
    //
    // NOTE: In practice, LLM inference dimensions always exceed these minimums.
    //       Qwen2.5-0.5B smallest: M=1 (decode), K=896, N=896 → only M can be small.
    //       For decode (M=1), batch multiple tokens or pad M to 64.
    //
    // =========================================================================
    bool rocmQuantGemm_isSupportedDimensions(int M, int N, int K)
    {
        // Simple bounds check - same logic used in executeDenseScale()
        // No try/catch overhead; CK's IsSupportedArgument is expensive and
        // redundant since we already know the tile constraints.
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
    // USE CASES:
    //   1. Verify base GEMM correctness before adding D-tensor scaling
    //   2. Compare against CPU reference implementation
    //   3. Debugging: isolate whether issues are in GEMM or scaling
    //
    // NOTE: Output is INT32. For inference, you need to apply scales separately
    //       or use one of the scaled variants (DenseScale, TwoKernel).
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

        // Early-exit: CK requires minimum dimensions (tile-based computation)
        if (!ck_dimensions_supported(M, N, K))
        {
            fprintf(stderr, "[rocmQuantGemm_executeNoScale] Dimensions too small for CK: "
                    "M=%d (min=%d), N=%d (min=%d), K=%d (min=%d)\n",
                    M, CK_MIN_M, N, CK_MIN_N, K, CK_MIN_K);
            return false;
        }

        auto& device_gemm = getCachedNoScaleKernel();

        // No D tensors - empty arrays
        std::array<const void*, 0> p_ds = {};
        std::array<ck::index_t, 0> strides_ds = {};

        auto argument = device_gemm.MakeArgument(
            d_A_int8,                   // p_a: [M×K] RowMajor
            d_weights_int8,             // p_b: [K×N] RowMajor
            p_ds,                       // p_ds: empty
            d_C_int32,                  // p_e: [M×N] RowMajor INT32 output
            M, N, K,                    // dimensions
            K,                          // StrideA = K
            N,                          // StrideB = N
            strides_ds,                 // StrideDs: empty
            N,                          // StrideE = N
            AElementOp{},
            BElementOp{},
            CDEElementOp_None{}
        );

        if (!device_gemm.IsSupportedArgument(argument))
        {
            fprintf(stderr, "[rocmQuantGemm_executeNoScale] CK does not support M=%d N=%d K=%d\n",
                    M, N, K);
            return false;
        }

        // Silence CK's unconditional std::cout spam
        ScopedCoutSilence silence;
        auto invoker = device_gemm.MakeInvoker();
        invoker.Run(argument, StreamConfig{nullptr, false});
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
    // HIPBLAS INT8 GEMM (SMALL M FALLBACK)
    // =========================================================================
    //
    // rocmQuantGemm_executeHipBLAS - INT8×INT8→FP32 GEMM using hipBLAS
    //
    // This function handles dimensions that CK cannot process (M < 64).
    // It uses hipBLAS's GemmEx with INT8 compute, then applies scaling.
    //
    // FORMULA:
    //   C_int32[m,n] = sum_k(A_int8[m,k] × B_int8[k,n])    [hipBLAS]
    //   E_fp32[m,n] = C_int32[m,n] × scale_A[m] × scale_B[n]  [applyScales_kernel]
    //
    // MEMORY LAYOUT:
    //   - A_int8: [M × K] Row-Major (RowMajor = lda=K)
    //   - B_int8: [K × N] Row-Major (ColMajor for BLAS = ldb=N, transposed)
    //   - C_int32/E_fp32: [M × N] Row-Major (RowMajor = ldc=N)
    //
    // HIPBLAS DETAILS:
    //   hipBLAS uses column-major by default, but we have row-major data.
    //   We compute C = A × B as: C^T = B^T × A^T
    //   This means we call: hipblasGemmEx(NOTRANS, NOTRANS, N, M, K, B, A, C)
    //   Output C is [M × N] row-major = [N × M] column-major
    //
    // WHY HIPBLAS FOR SMALL M:
    //   - CK requires M ≥ 64 (tile-based computation)
    //   - Decode generates M=1 tokens at a time
    //   - hipBLAS has no minimum dimension requirements
    //   - Performance is acceptable for small M (memory-bound anyway)
    //
    // =========================================================================
    bool rocmQuantGemm_executeHipBLAS(
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

        // 1. Get/create hipBLAS handle
        hipblasHandle_t handle = getHipblasHandle(rocm_device_id);
        if (!handle)
        {
            fprintf(stderr, "[rocmQuantGemm_executeHipBLAS] Failed to get hipBLAS handle\n");
            return false;
        }

        // 2. Allocate INT32 accumulator buffer
        int32_t *d_C_int32 = nullptr;
        hipError_t err = hipMalloc(&d_C_int32, static_cast<size_t>(M) * N * sizeof(int32_t));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_executeHipBLAS] Failed to allocate INT32 buffer: %s\n",
                    hipGetErrorString(err));
            return false;
        }

        // 3. Run hipBLAS INT8 GEMM: C_int32 = A_int8 × B_int8
        //
        // Row-major trick: compute C = A × B by doing C^T = B^T × A^T
        // Since our B is [K × N] row-major = [N × K] column-major,
        // and our A is [M × K] row-major = [K × M] column-major,
        // we call: GemmEx(opB=N, opA=N, N, M, K, B, N, A, K, C, N)
        //
        // This gives us [N × M] column-major output = [M × N] row-major!
        const int32_t alpha = 1;
        const int32_t beta = 0;

        hipblasStatus_t status = hipblasGemmEx(
            handle,
            HIPBLAS_OP_N,               // opB: no transpose (B is K×N row = N×K col)
            HIPBLAS_OP_N,               // opA: no transpose (A is M×K row = K×M col)
            N, M, K,                    // [N × M] col-major output = [M × N] row-major
            &alpha,                     // alpha = 1
            d_B_int8,                   // B: [K × N] row-major
            HIP_R_8I,                   // B type: INT8
            N,                          // ldb = N (row-major stride)
            d_A_int8,                   // A: [M × K] row-major
            HIP_R_8I,                   // A type: INT8
            K,                          // lda = K (row-major stride)
            &beta,                      // beta = 0
            d_C_int32,                  // C: [M × N] row-major output
            HIP_R_32I,                  // C type: INT32
            N,                          // ldc = N (row-major stride)
            HIPBLAS_COMPUTE_32I,        // compute type: 32-bit integer
            HIPBLAS_GEMM_DEFAULT);      // algorithm

        if (status != HIPBLAS_STATUS_SUCCESS)
        {
            fprintf(stderr, "[rocmQuantGemm_executeHipBLAS] hipblasGemmEx failed: %d\n",
                    static_cast<int>(status));
            hipFree(d_C_int32);
            return false;
        }

        // 4. Apply scales: E_fp32[m,n] = C_int32[m,n] * scale_A[m] * scale_B[n]
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
            fprintf(stderr, "[rocmQuantGemm_executeHipBLAS] applyScales_kernel failed: %s\n",
                    hipGetErrorString(err));
            hipFree(d_C_int32);
            return false;
        }

        // 5. Cleanup
        hipFree(d_C_int32);
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

} // extern "C"
