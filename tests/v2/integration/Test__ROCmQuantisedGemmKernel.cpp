/**
 * @file Test__ROCmQuantisedGemmKernel.cpp
 * @brief Integration tests for ROCmQuantisedGemmKernel - GPU-requiring tests
 *
 * Tests the full pipeline of ROCm INT8 quantized GEMM:
 * - Activation quantization (FP32 → INT8) on device
 * - Work buffer allocation and growth
 * - CK 3-way dispatch: 32×32, 64×64, 128×128 tile kernels
 * - Correctness across M sizes (1, 8, 32, 64, 128, 256)
 * - Full pipeline integration with realistic model dimensions
 *
 * CPU-only tests (weight packing, dimension validation) are in:
 *   tests/v2/unit/kernels/rocm/Test__ROCmQuantisedGemmKernel.cpp
 *
 * @note Requires ROCm device to run. Tests are skipped if no GPU available.
 * @note Run with build_v2_integration for proper snapshot support.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>

#include "kernels/rocm/ROCmQuantisedGemmKernel.h"
#include "execution/DeviceWorkspaceManager.h"
#include "tensors/Tensors.h"
#include "../utils/TestTensorFactory.h"
#include "utils/Logger.h"

// OneDNN for reference GEMM (much better than naive loop!)
#ifdef HAVE_ONEDNN
#include "kernels/cpu/gemm_v4/FloatingPointGemmKernel.h"
#include <dnnl.hpp>
#endif

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>

// Forward declarations for HIP functions used in integration tests
extern "C"
{
    // Activation quantization kernel
    bool rocmQuantGemm_quantizeActivations(
        const float *d_A_fp32,
        int8_t *d_A_int8,
        float *d_scales_A,
        int M, int K,
        int rocm_device_id);

    // Work buffer management
    bool rocmQuantGemm_ensureWorkBuffers(
        int8_t **d_A_int8,
        float **d_scales_A,
        int32_t **d_C_int32,
        int *work_buffer_M,
        int M, int K, int N,
        int rocm_device_id);

    // Base INT8×INT8→INT32 GEMM (no scaling) - 3-way dispatch entry point
    bool rocmQuantGemm_executeNoScale(
        const int8_t *d_A_int8,
        const int8_t *d_weights_int8,
        int32_t *d_C_int32,
        int M, int N, int K,
        int rocm_device_id);

    // Two-kernel path: INT8×INT8→INT32 GEMM + separate scale application
    bool rocmQuantGemm_executeTwoKernel(
        const int8_t *d_A_int8,
        const int8_t *d_weights_int8,
        float *d_C_fp32,
        const float *d_scales_A,
        const float *d_scales_B,
        int M, int N, int K,
        int rocm_device_id);

    // Memory management
    void rocmQuantGemm_freeDevice(void *d_ptr);
    bool rocmQuantGemm_copyHostToDevice(float *d_dst, const float *h_src, size_t count, int rocm_device_id);
    bool rocmQuantGemm_copyDeviceToHost(float *h_dst, const float *d_src, size_t count, int rocm_device_id);
    bool rocmQuantGemm_allocFloat(float **d_ptr, size_t count, int rocm_device_id);
    bool rocmQuantGemm_allocInt32(int32_t **d_ptr, size_t count, int rocm_device_id);

    // Apply scaling with full epilogue (alpha, beta, bias)
    bool rocmQuantGemm_applyScaling(
        const int32_t *d_C_int32, // [M×N] INT32 GEMM output
        float *d_C_fp32,          // [M×N] FP32 output
        const float *d_scales_A,  // [M] per-row scales
        const float *d_scales_B,  // [N] per-column scales
        int M, int N,
        float alpha, float beta,
        const float *d_C_existing, // For beta != 0 (nullable)
        const float *d_bias,       // [N] optional bias (nullable)
        int rocm_device_id);
}
#endif

namespace llaminar2
{
    namespace rocm
    {
        namespace integration_test
        {
            using namespace llaminar2::test;

            // =====================================================================
            // Test fixture
            // =====================================================================

            class ROCmQuantisedGemmIntegrationTest : public ::testing::Test
            {
            protected:
                void SetUp() override
                {
#ifdef HAVE_ROCM
                    int device_count = 0;
                    hipError_t err = hipGetDeviceCount(&device_count);
                    has_rocm_device_ = (err == hipSuccess && device_count > 0);

                    if (has_rocm_device_)
                    {
                        hipDeviceProp_t props;
                        (void)hipGetDeviceProperties(&props, 0);
                        LOG_INFO("[Integration] ROCm device: " << props.name
                                                               << " (" << props.gcnArchName << ")");
                    }
#else
                    has_rocm_device_ = false;
#endif
                }

                /**
                 * @brief Create FP32 activations with predictable values for testing
                 * @param rows Number of batch elements (M)
                 * @param cols Number of features (K)
                 * @return Vector of FP32 values
                 */
                std::vector<float> createActivations(size_t rows, size_t cols)
                {
                    std::vector<float> data(rows * cols);
                    for (size_t i = 0; i < rows * cols; ++i)
                    {
                        // Values in [-1, 1] range
                        data[i] = static_cast<float>(i % 256) / 128.0f - 1.0f;
                    }
                    return data;
                }

                bool has_rocm_device_ = false;

#ifdef HAVE_ROCM
                // Workspace manager for GEMM kernel tests
                std::unique_ptr<DeviceWorkspaceManager> workspace_;

                /**
                 * @brief Set up workspace for GEMM kernel
                 *
                 * The ROCmQuantisedGemmKernel needs workspace buffers for:
                 * - Quantized activations (INT8)
                 * - Per-row scales
                 * - INT32 intermediate output
                 */
                bool setupWorkspace(ROCmQuantisedGemmKernel &kernel, int M, int N, int K)
                {
                    auto reqs = kernel.getWorkspaceRequirements(M, N, K);
                    workspace_ = std::make_unique<DeviceWorkspaceManager>(DeviceId::rocm(0), 64 * 1024 * 1024); // 64MB
                    if (!workspace_->allocate(reqs))
                    {
                        LOG_ERROR("Failed to allocate workspace for GEMM kernel");
                        return false;
                    }
                    kernel.bindWorkspace(workspace_.get());
                    return true;
                }

                void cleanupWorkspace(ROCmQuantisedGemmKernel &kernel)
                {
                    if (workspace_)
                    {
                        kernel.unbindWorkspace();
                        workspace_.reset();
                    }
                }
#endif
            };

#ifdef HAVE_ROCM
            namespace
            {
                float cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b)
                {
                    if (a.size() != b.size() || a.empty())
                    {
                        return 0.0f;
                    }
                    double dot = 0.0;
                    double na = 0.0;
                    double nb = 0.0;
                    for (size_t i = 0; i < a.size(); ++i)
                    {
                        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
                        na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
                        nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
                    }
                    if (na == 0.0 || nb == 0.0)
                    {
                        return 0.0f;
                    }
                    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
                }

                void fillDeterministicInt8Pattern(std::vector<int8_t> &a, int M, int K)
                {
                    // Row-constant pattern: A[m,k] depends only on m.
                    // Keep values small to avoid overflow in int32 accumulation for large K.
                    a.resize(static_cast<size_t>(M) * K);
                    for (int m = 0; m < M; ++m)
                    {
                        const int v = (m % 13) - 6; // [-6, 6]
                        for (int k = 0; k < K; ++k)
                        {
                            a[static_cast<size_t>(m) * K + k] = static_cast<int8_t>(v);
                        }
                    }
                }

                void fillDeterministicInt8PatternTransposed(std::vector<int8_t> &b, int K, int N)
                {
                    // Column-constant pattern: B[k,n] depends only on n.
                    // Stored in ROW-MAJOR: B[k,n] at index k * N + n
                    b.resize(static_cast<size_t>(K) * N);
                    for (int k = 0; k < K; ++k)
                    {
                        for (int n = 0; n < N; ++n)
                        {
                            const int v = (n % 17) - 8; // [-8, 8]
                            b[static_cast<size_t>(k) * N + n] = static_cast<int8_t>(v);
                        }
                    }
                }

                /**
                 * @brief Fill B matrix in COLUMN-MAJOR format for CK kernel
                 *
                 * CK kernels expect B in column-major format: B[k,n] at index k + n * K
                 * This is equivalent to storing the transpose [N×K] in row-major.
                 */
                void fillDeterministicInt8PatternColumnMajor(std::vector<int8_t> &b, int K, int N)
                {
                    // Column-constant pattern: B[k,n] depends only on n.
                    // Stored in COLUMN-MAJOR: B[k,n] at index k + n * K
                    b.resize(static_cast<size_t>(K) * N);
                    for (int k = 0; k < K; ++k)
                    {
                        for (int n = 0; n < N; ++n)
                        {
                            const int v = (n % 17) - 8; // [-8, 8]
                            b[static_cast<size_t>(k) + static_cast<size_t>(n) * K] = static_cast<int8_t>(v);
                        }
                    }
                }

                /**
                 * @brief Compute INT8×INT8→INT32 reference GEMM using OneDNN
                 *
                 * C[m,n] = sum_k(A[m,k] * B[k,n])
                 *
                 * Uses OneDNN's optimized s8×s8→s32 matmul when available,
                 * falls back to OpenMP parallelized naive loop otherwise.
                 *
                 * @param A Input matrix A [M × K] row-major
                 * @param B Input matrix B [K × N] row-major
                 * @param C Output matrix C [M × N] row-major
                 * @param M Number of rows in A
                 * @param N Number of columns in B
                 * @param K Shared dimension
                 */
                void computeInt8GemmReference(
                    const std::vector<int8_t> &A,
                    const std::vector<int8_t> &B,
                    std::vector<int32_t> &C,
                    int M, int N, int K)
                {
                    C.resize(static_cast<size_t>(M) * N);

#ifdef HAVE_ONEDNN
                    using namespace dnnl;
                    using dt = memory::data_type;
                    using tag = memory::format_tag;

                    // Thread-local engine and stream
                    static thread_local engine eng(engine::kind::cpu, 0);
                    static thread_local stream strm(eng);

                    // Memory descriptors
                    memory::dims src_dims = {M, K};
                    memory::dims weights_dims = {K, N};
                    memory::dims dst_dims = {M, N};

                    auto src_md = memory::desc(src_dims, dt::s8, tag::ab);
                    auto weights_md = memory::desc(weights_dims, dt::s8, tag::ab);
                    auto dst_md = memory::desc(dst_dims, dt::s32, tag::ab);

                    auto matmul_pd = matmul::primitive_desc(eng, src_md, weights_md, dst_md);
                    auto src_mem = memory(src_md, eng, const_cast<int8_t *>(A.data()));
                    auto weights_mem = memory(weights_md, eng, const_cast<int8_t *>(B.data()));
                    auto dst_mem = memory(dst_md, eng, C.data());

                    auto matmul_prim = matmul(matmul_pd);
                    matmul_prim.execute(strm, {{DNNL_ARG_SRC, src_mem},
                                               {DNNL_ARG_WEIGHTS, weights_mem},
                                               {DNNL_ARG_DST, dst_mem}});
                    strm.wait();
#else
                    // Fallback: OpenMP parallelized naive implementation
#pragma omp parallel for collapse(2) schedule(static)
                    for (int m = 0; m < M; ++m)
                    {
                        for (int n = 0; n < N; ++n)
                        {
                            int32_t acc = 0;
                            for (int k = 0; k < K; ++k)
                            {
                                acc += static_cast<int32_t>(A[static_cast<size_t>(m) * K + k]) *
                                       static_cast<int32_t>(B[static_cast<size_t>(k) * N + n]);
                            }
                            C[static_cast<size_t>(m) * N + n] = acc;
                        }
                    }
#endif
                }

                /**
                 * @brief Compute INT8×INT8→INT32 reference GEMM with column-major B
                 *
                 * C[m,n] = sum_k(A[m,k] * B[k,n]) where B is stored column-major
                 *
                 * This matches CK's expected layout: A row-major, B column-major.
                 *
                 * @param A Input matrix A [M × K] row-major: A[m,k] at m*K+k
                 * @param B Input matrix B [K × N] column-major: B[k,n] at k+n*K
                 * @param C Output matrix C [M × N] row-major
                 */
                void computeInt8GemmReferenceColumnMajorB(
                    const std::vector<int8_t> &A,
                    const std::vector<int8_t> &B,
                    std::vector<int32_t> &C,
                    int M, int N, int K)
                {
                    C.resize(static_cast<size_t>(M) * N);

                    // OpenMP parallelized implementation for column-major B
#pragma omp parallel for collapse(2) schedule(static)
                    for (int m = 0; m < M; ++m)
                    {
                        for (int n = 0; n < N; ++n)
                        {
                            int32_t acc = 0;
                            for (int k = 0; k < K; ++k)
                            {
                                // A is row-major: A[m,k] at m*K+k
                                // B is column-major: B[k,n] at k+n*K
                                acc += static_cast<int32_t>(A[static_cast<size_t>(m) * K + k]) *
                                       static_cast<int32_t>(B[static_cast<size_t>(k) + static_cast<size_t>(n) * K]);
                            }
                            C[static_cast<size_t>(m) * N + n] = acc;
                        }
                    }
                }

                /**
                 * @brief Compute INT8×INT8→FP32 reference GEMM with per-row/per-col scales using OneDNN
                 *
                 * C[m,n] = sum_k(A[m,k] * B[k,n]) * scaleA[m] * scaleB[n]
                 *
                 * Uses OneDNN's optimized INT8 matmul for the core GEMM, then applies
                 * scales manually in a separate pass.
                 *
                 * @param A Input matrix A [M × K] row-major
                 * @param B Input matrix B [K × N] row-major
                 * @param scaleA Per-row scales for A [M]
                 * @param scaleB Per-column scales for B [N]
                 * @param C Output matrix C [M × N] row-major
                 * @param M Number of rows in A
                 * @param N Number of columns in B
                 * @param K Shared dimension
                 */
                void computeScaledInt8GemmReference(
                    const std::vector<int8_t> &A,
                    const std::vector<int8_t> &B,
                    const std::vector<float> &scaleA,
                    const std::vector<float> &scaleB,
                    std::vector<float> &C,
                    int M, int N, int K)
                {
                    C.resize(static_cast<size_t>(M) * N);

#ifdef HAVE_ONEDNN
                    using namespace dnnl;
                    using dt = memory::data_type;
                    using tag = memory::format_tag;

                    // Thread-local engine and stream
                    static thread_local engine eng(engine::kind::cpu, 0);
                    static thread_local stream strm(eng);

                    // Memory descriptors
                    memory::dims src_dims = {M, K};
                    memory::dims weights_dims = {K, N};
                    memory::dims dst_dims = {M, N};

                    auto src_md = memory::desc(src_dims, dt::s8, tag::ab);
                    auto weights_md = memory::desc(weights_dims, dt::s8, tag::ab);
                    auto dst_md = memory::desc(dst_dims, dt::f32, tag::ab);

                    auto matmul_pd = matmul::primitive_desc(eng, src_md, weights_md, dst_md);
                    auto src_mem = memory(src_md, eng, const_cast<int8_t *>(A.data()));
                    auto weights_mem = memory(weights_md, eng, const_cast<int8_t *>(B.data()));
                    auto dst_mem = memory(dst_md, eng, C.data());

                    auto matmul_prim = matmul(matmul_pd);
                    matmul_prim.execute(strm, {{DNNL_ARG_SRC, src_mem},
                                               {DNNL_ARG_WEIGHTS, weights_mem},
                                               {DNNL_ARG_DST, dst_mem}});
                    strm.wait();

                    // Apply scales manually: C[m,n] *= scaleA[m] * scaleB[n]
#pragma omp parallel for collapse(2) schedule(static)
                    for (int m = 0; m < M; ++m)
                    {
                        for (int n = 0; n < N; ++n)
                        {
                            C[static_cast<size_t>(m) * N + n] *= scaleA[m] * scaleB[n];
                        }
                    }
#else
                    // Fallback: OpenMP parallelized naive implementation
#pragma omp parallel for collapse(2) schedule(static)
                    for (int m = 0; m < M; ++m)
                    {
                        for (int n = 0; n < N; ++n)
                        {
                            int32_t acc = 0;
                            for (int k = 0; k < K; ++k)
                            {
                                acc += static_cast<int32_t>(A[static_cast<size_t>(m) * K + k]) *
                                       static_cast<int32_t>(B[static_cast<size_t>(k) * N + n]);
                            }
                            C[static_cast<size_t>(m) * N + n] =
                                static_cast<float>(acc) * scaleA[m] * scaleB[n];
                        }
                    }
#endif
                }

                /**
                 * @brief Compute INT8×INT8→FP32 reference GEMM with column-major B and scales
                 *
                 * C[m,n] = sum_k(A[m,k] * B[k,n]) * scaleA[m] * scaleB[n]
                 * where B is stored column-major
                 *
                 * @param A Input matrix A [M × K] row-major: A[m,k] at m*K+k
                 * @param B Input matrix B [K × N] column-major: B[k,n] at k+n*K
                 * @param scaleA Per-row scales for A [M]
                 * @param scaleB Per-column scales for B [N]
                 * @param C Output matrix C [M × N] row-major
                 */
                void computeScaledInt8GemmReferenceColumnMajorB(
                    const std::vector<int8_t> &A,
                    const std::vector<int8_t> &B,
                    const std::vector<float> &scaleA,
                    const std::vector<float> &scaleB,
                    std::vector<float> &C,
                    int M, int N, int K)
                {
                    C.resize(static_cast<size_t>(M) * N);

                    // OpenMP parallelized implementation for column-major B
#pragma omp parallel for collapse(2) schedule(static)
                    for (int m = 0; m < M; ++m)
                    {
                        for (int n = 0; n < N; ++n)
                        {
                            int32_t acc = 0;
                            for (int k = 0; k < K; ++k)
                            {
                                // A is row-major: A[m,k] at m*K+k
                                // B is column-major: B[k,n] at k+n*K
                                acc += static_cast<int32_t>(A[static_cast<size_t>(m) * K + k]) *
                                       static_cast<int32_t>(B[static_cast<size_t>(k) + static_cast<size_t>(n) * K]);
                            }
                            C[static_cast<size_t>(m) * N + n] =
                                static_cast<float>(acc) * scaleA[m] * scaleB[n];
                        }
                    }
                }
            } // namespace
#endif

            // =====================================================================
            // Phase 2: Activation Quantization Tests (GPU-requiring)
            // =====================================================================

            /**
             * @test Quantize small matrix of activations on device
             *
             * Tests the core activation quantization kernel:
             * - FP32 activations uploaded to device
             * - Quantized to INT8 with per-row scaling
             * - Results validated for symmetric quantization
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, QuantizeActivations_SmallMatrix)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                const int M = 4;
                const int K = 64;
                const int N = 32; // Needed for ensureWorkBuffers

                auto h_activations = createActivations(M, K);

                // Allocate device memory for FP32 activations
                float *d_activations = nullptr;
                ASSERT_TRUE(rocmQuantGemm_allocFloat(&d_activations, M * K, 0));

                // Allocate work buffers (INT8, scales, INT32 output)
                int8_t *d_A_int8 = nullptr;
                float *d_scales_A = nullptr;
                int32_t *d_C_int32 = nullptr;
                int work_buffer_M = 0;
                ASSERT_TRUE(rocmQuantGemm_ensureWorkBuffers(
                    &d_A_int8, &d_scales_A, &d_C_int32, &work_buffer_M,
                    M, K, N, 0));

                // Upload and quantize
                ASSERT_TRUE(rocmQuantGemm_copyHostToDevice(d_activations, h_activations.data(), M * K, 0));
                ASSERT_TRUE(rocmQuantGemm_quantizeActivations(d_activations, d_A_int8, d_scales_A, M, K, 0));

                // Download results
                std::vector<int8_t> h_A_int8(M * K);
                std::vector<float> h_scales_A(M);

                ASSERT_EQ(hipMemcpy(h_A_int8.data(), d_A_int8, M * K * sizeof(int8_t), hipMemcpyDeviceToHost), hipSuccess);
                ASSERT_EQ(hipMemcpy(h_scales_A.data(), d_scales_A, M * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

                // Verify scales are positive
                for (int m = 0; m < M; ++m)
                {
                    EXPECT_GT(h_scales_A[m], 0.0f) << "Scale for row " << m << " should be positive";
                }

                // Verify INT8 values are in valid range
                for (int i = 0; i < M * K; ++i)
                {
                    EXPECT_GE(h_A_int8[i], -127);
                    EXPECT_LE(h_A_int8[i], 127);
                }

                // Cleanup
                rocmQuantGemm_freeDevice(d_activations);
                rocmQuantGemm_freeDevice(d_A_int8);
                rocmQuantGemm_freeDevice(d_scales_A);
                rocmQuantGemm_freeDevice(d_C_int32);
            }

            // =====================================================================
            // Phase 3: Base INT8 GEMM without D-tensors (debugging isolation)
            // =====================================================================

            /**
             * @test Base INT8×INT8→INT32 GEMM without any D-tensor scaling
             *
             * Tests the raw CK GEMM kernel: rocmQuantGemm_executeNoScale()
             *
             * This test isolates the core GEMM operation from D-tensor handling.
             * If this passes but fused tests fail, the issue is in D-tensor configuration.
             * If this fails, the issue is in the base GEMM tile parameters.
             *
             * @note This tests the low-level C function directly, not the ROCmQuantisedGemmKernel class.
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, BaseGemm_NoScale_Deterministic128)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                // Same dimensions as the failing fused test
                const int M = 128;
                const int N = 128;
                const int K = 128;

                ASSERT_EQ(hipSetDevice(0), hipSuccess);

                std::vector<int8_t> h_A;
                std::vector<int8_t> h_B;
                fillDeterministicInt8Pattern(h_A, M, K);
                // CK kernel expects B in COLUMN-MAJOR format: B[k,n] at index k + n*K
                fillDeterministicInt8PatternColumnMajor(h_B, K, N);

                // Device buffers
                int8_t *d_A = nullptr;
                int8_t *d_B = nullptr;
                int32_t *d_C = nullptr;

                ASSERT_EQ(hipMalloc(&d_A, static_cast<size_t>(M) * K * sizeof(int8_t)), hipSuccess);
                ASSERT_EQ(hipMalloc(&d_B, static_cast<size_t>(K) * N * sizeof(int8_t)), hipSuccess);
                ASSERT_EQ(hipMalloc(&d_C, static_cast<size_t>(M) * N * sizeof(int32_t)), hipSuccess);

                ASSERT_EQ(hipMemcpy(d_A, h_A.data(), static_cast<size_t>(M) * K * sizeof(int8_t), hipMemcpyHostToDevice), hipSuccess);
                ASSERT_EQ(hipMemcpy(d_B, h_B.data(), static_cast<size_t>(K) * N * sizeof(int8_t), hipMemcpyHostToDevice), hipSuccess);

                ASSERT_TRUE(rocmQuantGemm_executeNoScale(d_A, d_B, d_C, M, N, K, /*rocm_device_id=*/0));

                std::vector<int32_t> h_C(static_cast<size_t>(M) * N);
                ASSERT_EQ(hipMemcpy(h_C.data(), d_C, static_cast<size_t>(M) * N * sizeof(int32_t), hipMemcpyDeviceToHost), hipSuccess);

                // CPU reference (column-major B to match CK kernel expectation)
                std::vector<int32_t> h_ref;
                computeInt8GemmReferenceColumnMajorB(h_A, h_B, h_ref, M, N, K);

                // Check for exact match (INT32 should be exact)
                int mismatch_count = 0;
                int32_t max_diff = 0;
                int worst_m = 0, worst_n = 0;
                for (int m = 0; m < M; ++m)
                {
                    for (int n = 0; n < N; ++n)
                    {
                        const size_t idx = static_cast<size_t>(m) * N + n;
                        const int32_t diff = std::abs(h_C[idx] - h_ref[idx]);
                        if (diff > 0)
                        {
                            ++mismatch_count;
                            if (diff > max_diff)
                            {
                                max_diff = diff;
                                worst_m = m;
                                worst_n = n;
                            }
                        }
                    }
                }

                LOG_INFO("[BaseGemm_NoScale] mismatches=" << mismatch_count
                                                          << " max_diff=" << max_diff
                                                          << " worst=(" << worst_m << "," << worst_n << ")");

                if (mismatch_count > 0)
                {
                    LOG_INFO("[BaseGemm_NoScale] Sample mismatches:");
                    int printed = 0;
                    for (int m = 0; m < M && printed < 10; ++m)
                    {
                        for (int n = 0; n < N && printed < 10; ++n)
                        {
                            const size_t idx = static_cast<size_t>(m) * N + n;
                            if (h_C[idx] != h_ref[idx])
                            {
                                LOG_INFO("  C[" << m << "," << n << "] = " << h_C[idx]
                                                << " (expected " << h_ref[idx] << ")");
                                ++printed;
                            }
                        }
                    }
                }

                EXPECT_EQ(mismatch_count, 0) << "Base GEMM should produce exact INT32 results";

                rocmQuantGemm_freeDevice(d_A);
                rocmQuantGemm_freeDevice(d_B);
                rocmQuantGemm_freeDevice(d_C);
            }

            // =====================================================================
            // Two-Kernel Path Tests (INT8×INT8→INT32 + scale application)
            // =====================================================================

            /**
             * @test Two-Kernel path correctness with 128×128×128 dimensions
             *
             * Tests rocmQuantGemm_executeTwoKernel() which separates GEMM and scaling:
             * 1. INT8×INT8→INT32 via CK GEMM
             * 2. Separate scale application kernel
             *
             * This is the production path for M ≤ 128.
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, TwoKernel_Deterministic128)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                const int M = 128;
                const int N = 128;
                const int K = 128;

                ASSERT_EQ(hipSetDevice(0), hipSuccess);

                std::vector<int8_t> h_A;
                std::vector<int8_t> h_B;
                fillDeterministicInt8Pattern(h_A, M, K);
                // CK kernel expects B in COLUMN-MAJOR format: B[k,n] at index k + n*K
                fillDeterministicInt8PatternColumnMajor(h_B, K, N);

                std::vector<float> h_scaleA(M);
                std::vector<float> h_scaleB(N);
                for (int m = 0; m < M; ++m)
                {
                    h_scaleA[m] = 0.001f * static_cast<float>(m + 1);
                }
                for (int n = 0; n < N; ++n)
                {
                    h_scaleB[n] = 0.002f * static_cast<float>(n + 1);
                }

                int8_t *d_A = nullptr;
                int8_t *d_B = nullptr;
                float *d_scaleA = nullptr;
                float *d_scaleB = nullptr;
                float *d_E = nullptr;

                ASSERT_EQ(hipMalloc(&d_A, static_cast<size_t>(M) * K * sizeof(int8_t)), hipSuccess);
                ASSERT_EQ(hipMalloc(&d_B, static_cast<size_t>(K) * N * sizeof(int8_t)), hipSuccess);
                ASSERT_EQ(hipMalloc(&d_scaleA, static_cast<size_t>(M) * sizeof(float)), hipSuccess);
                ASSERT_EQ(hipMalloc(&d_scaleB, static_cast<size_t>(N) * sizeof(float)), hipSuccess);
                ASSERT_EQ(hipMalloc(&d_E, static_cast<size_t>(M) * N * sizeof(float)), hipSuccess);

                ASSERT_EQ(hipMemcpy(d_A, h_A.data(), static_cast<size_t>(M) * K * sizeof(int8_t), hipMemcpyHostToDevice), hipSuccess);
                ASSERT_EQ(hipMemcpy(d_B, h_B.data(), static_cast<size_t>(K) * N * sizeof(int8_t), hipMemcpyHostToDevice), hipSuccess);
                ASSERT_EQ(hipMemcpy(d_scaleA, h_scaleA.data(), static_cast<size_t>(M) * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
                ASSERT_EQ(hipMemcpy(d_scaleB, h_scaleB.data(), static_cast<size_t>(N) * sizeof(float), hipMemcpyHostToDevice), hipSuccess);

                // Use Two-Kernel path (production path for M ≤ 128)
                ASSERT_TRUE(rocmQuantGemm_executeTwoKernel(
                    d_A, d_B, d_E,
                    d_scaleA, d_scaleB,
                    M, N, K,
                    /*rocm_device_id=*/0));

                std::vector<float> h_E(static_cast<size_t>(M) * N);
                ASSERT_EQ(hipMemcpy(h_E.data(), d_E, static_cast<size_t>(M) * N * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

                // CPU reference (column-major B to match CK kernel expectation)
                std::vector<float> h_ref;
                computeScaledInt8GemmReferenceColumnMajorB(h_A, h_B, h_scaleA, h_scaleB, h_ref, M, N, K);

                const float cos = cosineSimilarity(h_E, h_ref);
                LOG_INFO("[TwoKernel_Deterministic128] cos=" << cos);

                // Two-Kernel path should have very high accuracy (>0.9999)
                EXPECT_GT(cos, 0.9999f) << "Two-Kernel path cosine similarity too low";

                rocmQuantGemm_freeDevice(d_A);
                rocmQuantGemm_freeDevice(d_B);
                rocmQuantGemm_freeDevice(d_scaleA);
                rocmQuantGemm_freeDevice(d_scaleB);
                rocmQuantGemm_freeDevice(d_E);
            }

            /**
             * @test Quantize large matrix of activations on device
             *
             * Tests scalability of activation quantization with realistic dimensions.
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, QuantizeActivations_LargeMatrix)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                const int M = 128;
                const int K = 896; // Qwen2.5 hidden size
                const int N = 256; // Needed for ensureWorkBuffers

                auto h_activations = createActivations(M, K);

                // Allocate device memory for FP32 activations
                float *d_activations = nullptr;
                ASSERT_TRUE(rocmQuantGemm_allocFloat(&d_activations, M * K, 0));

                // Allocate work buffers
                int8_t *d_A_int8 = nullptr;
                float *d_scales_A = nullptr;
                int32_t *d_C_int32 = nullptr;
                int work_buffer_M = 0;
                ASSERT_TRUE(rocmQuantGemm_ensureWorkBuffers(
                    &d_A_int8, &d_scales_A, &d_C_int32, &work_buffer_M,
                    M, K, N, 0));

                // Upload and quantize
                ASSERT_TRUE(rocmQuantGemm_copyHostToDevice(d_activations, h_activations.data(), M * K, 0));
                ASSERT_TRUE(rocmQuantGemm_quantizeActivations(d_activations, d_A_int8, d_scales_A, M, K, 0));

                // Download results
                std::vector<int8_t> h_A_int8(M * K);
                std::vector<float> h_scales_A(M);

                ASSERT_EQ(hipMemcpy(h_A_int8.data(), d_A_int8, M * K * sizeof(int8_t), hipMemcpyDeviceToHost), hipSuccess);
                ASSERT_EQ(hipMemcpy(h_scales_A.data(), d_scales_A, M * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

                // Verify scales are positive
                for (int m = 0; m < M; ++m)
                {
                    EXPECT_GT(h_scales_A[m], 0.0f);
                }

                // Cleanup
                rocmQuantGemm_freeDevice(d_activations);
                rocmQuantGemm_freeDevice(d_A_int8);
                rocmQuantGemm_freeDevice(d_scales_A);
                rocmQuantGemm_freeDevice(d_C_int32);
            }

            /**
             * @test Quantize zero row - edge case for scale computation
             *
             * When a row is all zeros, the scale should still be positive (small epsilon).
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, QuantizeActivations_ZeroRow)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                const int M = 4;
                const int K = 64;
                const int N = 32; // Needed for ensureWorkBuffers

                // Create activations with one zero row
                std::vector<float> h_activations(M * K, 0.0f);
                for (int m = 0; m < M; ++m)
                {
                    if (m != 1)
                    { // Row 1 is all zeros
                        for (int k = 0; k < K; ++k)
                        {
                            h_activations[m * K + k] = static_cast<float>(k % 256) / 128.0f - 1.0f;
                        }
                    }
                }

                // Allocate device memory for FP32 activations
                float *d_activations = nullptr;
                ASSERT_TRUE(rocmQuantGemm_allocFloat(&d_activations, M * K, 0));

                // Allocate work buffers
                int8_t *d_A_int8 = nullptr;
                float *d_scales_A = nullptr;
                int32_t *d_C_int32 = nullptr;
                int work_buffer_M = 0;
                ASSERT_TRUE(rocmQuantGemm_ensureWorkBuffers(
                    &d_A_int8, &d_scales_A, &d_C_int32, &work_buffer_M,
                    M, K, N, 0));

                ASSERT_TRUE(rocmQuantGemm_copyHostToDevice(d_activations, h_activations.data(), M * K, 0));
                ASSERT_TRUE(rocmQuantGemm_quantizeActivations(d_activations, d_A_int8, d_scales_A, M, K, 0));

                // Download scales
                std::vector<float> h_scales_A(M);
                ASSERT_EQ(hipMemcpy(h_scales_A.data(), d_scales_A, M * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

                // All scales should be positive (zero row gets small epsilon scale)
                for (int m = 0; m < M; ++m)
                {
                    EXPECT_GT(h_scales_A[m], 0.0f) << "Scale for row " << m << " should be positive";
                }

                // Cleanup
                rocmQuantGemm_freeDevice(d_activations);
                rocmQuantGemm_freeDevice(d_A_int8);
                rocmQuantGemm_freeDevice(d_scales_A);
                rocmQuantGemm_freeDevice(d_C_int32);
            }

            /**
             * @test Verify reconstruction accuracy after quantization
             *
             * Quantize activations, then dequantize and compare to original.
             * Error should be bounded by quantization resolution.
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, QuantizeActivations_ReconstructionAccuracy)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                const int M = 16;
                const int K = 128;
                const int N = 64; // Needed for ensureWorkBuffers

                auto h_activations = createActivations(M, K);

                // Allocate device memory for FP32 activations
                float *d_activations = nullptr;
                ASSERT_TRUE(rocmQuantGemm_allocFloat(&d_activations, M * K, 0));

                // Allocate work buffers
                int8_t *d_A_int8 = nullptr;
                float *d_scales_A = nullptr;
                int32_t *d_C_int32 = nullptr;
                int work_buffer_M = 0;
                ASSERT_TRUE(rocmQuantGemm_ensureWorkBuffers(
                    &d_A_int8, &d_scales_A, &d_C_int32, &work_buffer_M,
                    M, K, N, 0));

                ASSERT_TRUE(rocmQuantGemm_copyHostToDevice(d_activations, h_activations.data(), M * K, 0));
                ASSERT_TRUE(rocmQuantGemm_quantizeActivations(d_activations, d_A_int8, d_scales_A, M, K, 0));

                // Download results
                std::vector<int8_t> h_A_int8(M * K);
                std::vector<float> h_scales_A(M);

                ASSERT_EQ(hipMemcpy(h_A_int8.data(), d_A_int8, M * K * sizeof(int8_t), hipMemcpyDeviceToHost), hipSuccess);
                ASSERT_EQ(hipMemcpy(h_scales_A.data(), d_scales_A, M * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

                // Compute reconstruction error
                float max_error = 0.0f;
                float total_error = 0.0f;
                for (int m = 0; m < M; ++m)
                {
                    float scale = h_scales_A[m];
                    for (int k = 0; k < K; ++k)
                    {
                        float original = h_activations[m * K + k];
                        float reconstructed = static_cast<float>(h_A_int8[m * K + k]) * scale;
                        float error = std::abs(original - reconstructed);
                        max_error = std::max(max_error, error);
                        total_error += error;
                    }
                }

                float avg_error = total_error / (M * K);

                // INT8 quantization: max error should be < 1 quantization step
                // For [-1, 1] range mapped to [-127, 127], step = 2/254 ≈ 0.008
                // Allow some tolerance for edge cases
                EXPECT_LT(max_error, 0.02f) << "Max reconstruction error too high";
                EXPECT_LT(avg_error, 0.01f) << "Average reconstruction error too high";

                LOG_INFO("[Integration] Reconstruction: max_error=" << max_error << ", avg_error=" << avg_error);

                // Cleanup
                rocmQuantGemm_freeDevice(d_activations);
                rocmQuantGemm_freeDevice(d_A_int8);
                rocmQuantGemm_freeDevice(d_scales_A);
                rocmQuantGemm_freeDevice(d_C_int32);
            }

            // =====================================================================
            // Work Buffer Management Tests (GPU-requiring)
            // =====================================================================

            /**
             * @test Work buffer allocation for various dimensions
             *
             * Verifies that ensureWorkBuffers allocates sufficient memory.
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, WorkBuffers_Allocation)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                int8_t *d_A_int8 = nullptr;
                float *d_scales_A = nullptr;
                int32_t *d_C_int32 = nullptr;
                int work_buffer_M = 0;

                const int M = 64;
                const int K = 128;
                const int N = 256;

                ASSERT_TRUE(rocmQuantGemm_ensureWorkBuffers(
                    &d_A_int8, &d_scales_A, &d_C_int32, &work_buffer_M,
                    M, K, N, 0));

                EXPECT_NE(d_A_int8, nullptr);
                EXPECT_NE(d_scales_A, nullptr);
                EXPECT_NE(d_C_int32, nullptr);
                EXPECT_GE(work_buffer_M, M);

                // Cleanup
                rocmQuantGemm_freeDevice(d_A_int8);
                rocmQuantGemm_freeDevice(d_scales_A);
                rocmQuantGemm_freeDevice(d_C_int32);
            }

            /**
             * @test Work buffer growth when M increases
             *
             * Verifies that buffers grow to accommodate larger batch sizes.
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, WorkBuffers_Growth)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                int8_t *d_A_int8 = nullptr;
                float *d_scales_A = nullptr;
                int32_t *d_C_int32 = nullptr;
                int work_buffer_M = 0;

                const int K = 128;
                const int N = 256;

                // Start with small M
                ASSERT_TRUE(rocmQuantGemm_ensureWorkBuffers(
                    &d_A_int8, &d_scales_A, &d_C_int32, &work_buffer_M,
                    16, K, N, 0));

                EXPECT_GE(work_buffer_M, 16);

                // Grow to larger M
                ASSERT_TRUE(rocmQuantGemm_ensureWorkBuffers(
                    &d_A_int8, &d_scales_A, &d_C_int32, &work_buffer_M,
                    64, K, N, 0));

                EXPECT_GE(work_buffer_M, 64);

                // Grow even larger
                ASSERT_TRUE(rocmQuantGemm_ensureWorkBuffers(
                    &d_A_int8, &d_scales_A, &d_C_int32, &work_buffer_M,
                    256, K, N, 0));

                EXPECT_GE(work_buffer_M, 256);

                // Cleanup
                rocmQuantGemm_freeDevice(d_A_int8);
                rocmQuantGemm_freeDevice(d_scales_A);
                rocmQuantGemm_freeDevice(d_C_int32);
            }

            // =====================================================================
            // Phase 3: CK GEMM Tests (GPU-requiring)
            // =====================================================================

            /**
             * @test CK GEMM with small aligned dimensions
             *
             * Tests the full INT8 GEMM pipeline with CK:
             * - Weight packing
             * - Activation quantization
             * - CK GEMM (INT8×INT8→INT32)
             * - Output scaling to FP32
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, CKGemm_SmallAligned)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                // Small aligned dimensions for debugging
                const int M = 32;
                const int K = 64;
                const int N = 64;

                // Create weights and pack
                auto weights = TestTensorFactory::createQ8_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});
                ROCmPackedWeights packed;
                ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

                // Create kernel
                ROCmQuantisedGemmKernel kernel(&packed, 0);

                // Setup workspace for GEMM kernel (required for work buffers)
                ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

                // Create input and output tensors
                auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
                auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});

                // Run GEMM using multiply_tensor (the implemented method)
                ASSERT_TRUE(kernel.multiply_tensor(input.get(), output.get(), M, N, K));

                // Compute reference using OneDNN (production-quality FP32 GEMM)
                // C = A × W^T where A is [M×K] and W is [N×K] (so W^T is [K×N])
                const float *h_activations = input->data();
                const float *h_weights_fp32 = weights->fp32_data();

                std::vector<float> reference(M * N, 0.0f);
#ifdef HAVE_ONEDNN
                // OneDNN matmul: A[M,K] × B[K,N] = C[M,N]
                // With transpose_B=true: A[M,K] × B^T[K,N] where B is stored as [N,K]
                // This matches: C[m,n] = sum_k(A[m,k] * W[n,k])
                ASSERT_TRUE(gemm_v4::run_onednn_fp32_matmul(
                    h_activations,    // A [M×K]
                    h_weights_fp32,   // B [N×K] (will be transposed)
                    reference.data(), // C [M×N]
                    M, N, K,
                    true, // transpose_B = true (W stored as [N×K])
                    1.0f, 0.0f));
#else
                // Fallback to naive loop if OneDNN not available
                for (int m = 0; m < M; ++m)
                {
                    for (int n = 0; n < N; ++n)
                    {
                        float sum = 0.0f;
                        for (int k = 0; k < K; ++k)
                        {
                            sum += h_activations[m * K + k] * h_weights_fp32[n * K + k];
                        }
                        reference[m * N + n] = sum;
                    }
                }
#endif

                // Compare using cosine similarity (robust to scale, doesn't blow up for near-zero)
                double dot_product = 0.0, norm_actual = 0.0, norm_ref = 0.0;
                for (int i = 0; i < M * N; ++i)
                {
                    float actual = output->data()[i];
                    float ref = reference[i];
                    dot_product += static_cast<double>(actual) * ref;
                    norm_actual += static_cast<double>(actual) * actual;
                    norm_ref += static_cast<double>(ref) * ref;
                }
                double cosine_sim = dot_product / (std::sqrt(norm_actual) * std::sqrt(norm_ref) + 1e-12);

                LOG_INFO("[Integration] CK GEMM " << M << "x" << K << "x" << N
                                                  << ": cosine_similarity=" << cosine_sim);

                // INT8 GEMM should have very high cosine similarity (>0.99)
                EXPECT_GT(cosine_sim, 0.99) << "CK GEMM cosine similarity too low";
            }

            /**
             * @test CK GEMM with FFN dimensions from Qwen2.5-0.5B
             *
             * Tests realistic dimensions: 896 → 4864 (FFN up/gate projection).
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, CKGemm_FFNDimensions)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                // Qwen2.5-0.5B FFN dimensions
                const int M = 64;   // Batch size
                const int K = 896;  // hidden_size
                const int N = 4864; // intermediate_size

                // Create weights and pack
                auto weights = TestTensorFactory::createQ8_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});
                ROCmPackedWeights packed;
                ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

                // Create kernel
                ROCmQuantisedGemmKernel kernel(&packed, 0);

                // Setup workspace for GEMM kernel (required for work buffers)
                ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

                // Create input and output tensors
                auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
                auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});

                // Run GEMM using multiply_tensor (the implemented method)
                ASSERT_TRUE(kernel.multiply_tensor(input.get(), output.get(), M, N, K));

                // Compute reference using OneDNN (production-quality FP32 GEMM)
                const float *h_activations = input->data();
                const float *h_weights_fp32 = weights->fp32_data();

                std::vector<float> reference(M * N, 0.0f);
#ifdef HAVE_ONEDNN
                ASSERT_TRUE(gemm_v4::run_onednn_fp32_matmul(
                    h_activations,    // A [M×K]
                    h_weights_fp32,   // B [N×K] (will be transposed)
                    reference.data(), // C [M×N]
                    M, N, K,
                    true, // transpose_B = true
                    1.0f, 0.0f));
#else
                for (int m = 0; m < M; ++m)
                {
                    for (int n = 0; n < N; ++n)
                    {
                        float sum = 0.0f;
                        for (int k = 0; k < K; ++k)
                        {
                            sum += h_activations[m * K + k] * h_weights_fp32[n * K + k];
                        }
                        reference[m * N + n] = sum;
                    }
                }
#endif

                // Compare using cosine similarity (robust to scale, doesn't blow up for near-zero)
                double dot_product = 0.0, norm_actual = 0.0, norm_ref = 0.0;
                for (int i = 0; i < M * N; ++i)
                {
                    float actual = output->data()[i];
                    float ref = reference[i];
                    dot_product += static_cast<double>(actual) * ref;
                    norm_actual += static_cast<double>(actual) * actual;
                    norm_ref += static_cast<double>(ref) * ref;
                }
                double cosine_sim = dot_product / (std::sqrt(norm_actual) * std::sqrt(norm_ref) + 1e-12);

                LOG_INFO("[Integration] CK GEMM FFN " << M << "x" << K << "x" << N
                                                      << ": cosine_similarity=" << cosine_sim);

                // INT8 GEMM should have very high cosine similarity (>0.99)
                EXPECT_GT(cosine_sim, 0.99) << "CK GEMM FFN cosine similarity too low";
            }

            // =====================================================================
            // Parameterized Matrix Size Tests
            // =====================================================================

            /**
             * @brief Helper to run GEMM test with given dimensions and return cosine similarity
             */
            double runGemmAndComputeCosineSimilarity(int M, int N, int K)
            {
                // Create weights and pack
                auto weights = TestTensorFactory::createQ8_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});

                ROCmPackedWeights packed;
                if (!packWeightsToROCm(weights.get(), packed))
                {
                    return -1.0;
                }

                // Create kernel
                ROCmQuantisedGemmKernel kernel(&packed, 0);

                // Setup workspace for GEMM kernel (required for work buffers)
                auto reqs = kernel.getWorkspaceRequirements(M, N, K);
                DeviceWorkspaceManager workspace(DeviceId::rocm(0), 64 * 1024 * 1024);
                if (!workspace.allocate(reqs))
                {
                    return -1.0;
                }
                kernel.bindWorkspace(&workspace);

                // Create input and output tensors
                auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
                auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});

                // Run GEMM
                if (!kernel.multiply_tensor(input.get(), output.get(), M, N, K))
                {
                    return -1.0;
                }

                // Compute reference using OneDNN
                const float *h_activations = input->data();
                const float *h_weights_fp32 = weights->fp32_data();
                std::vector<float> reference(M * N, 0.0f);

#ifdef HAVE_ONEDNN
                if (!gemm_v4::run_onednn_fp32_matmul(
                        h_activations, h_weights_fp32, reference.data(),
                        M, N, K, true, 1.0f, 0.0f))
                {
                    return -1.0;
                }
#else
                for (int m = 0; m < M; ++m)
                {
                    for (int n = 0; n < N; ++n)
                    {
                        float sum = 0.0f;
                        for (int k = 0; k < K; ++k)
                        {
                            sum += h_activations[m * K + k] * h_weights_fp32[n * K + k];
                        }
                        reference[m * N + n] = sum;
                    }
                }
#endif

                // Compute cosine similarity
                double dot_product = 0.0, norm_actual = 0.0, norm_ref = 0.0;
                for (int i = 0; i < M * N; ++i)
                {
                    float actual = output->data()[i];
                    float ref = reference[i];
                    dot_product += static_cast<double>(actual) * ref;
                    norm_actual += static_cast<double>(actual) * actual;
                    norm_ref += static_cast<double>(ref) * ref;
                }

                // Unbind workspace before returning (workspace will be destroyed at scope exit)
                kernel.unbindWorkspace();

                return dot_product / (std::sqrt(norm_actual) * std::sqrt(norm_ref) + 1e-12);
            }

            // =====================================================================
            // Qwen Model Dimension Tests (0.5B and 1.5B only)
            // =====================================================================
            //
            // Tests real model dimensions for Qwen 0.5B and 1.5B with decode (M=1)
            // and prefill (M=64, 128, 256) batch sizes.

            /**
             * @test Qwen2.5-0.5B decode (M=1)
             *
             * hidden_size=896, intermediate_size=4864
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, CKGemm_Qwen05B_Decode)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                struct TestCase
                {
                    const char *name;
                    int M, N, K;
                };
                std::vector<TestCase> cases = {
                    {"Wo_proj", 1, 896, 896},
                    {"FFN_gate", 1, 4864, 896},
                    {"FFN_down", 1, 896, 4864},
                };

                for (const auto &tc : cases)
                {
                    double cos = runGemmAndComputeCosineSimilarity(tc.M, tc.N, tc.K);
                    LOG_INFO("[Integration] Qwen0.5B decode " << tc.name << ": cos=" << cos);
                    EXPECT_GT(cos, 0.99) << "Qwen0.5B decode " << tc.name << " cosine too low";
                }
            }

            /**
             * @test Qwen2.5-1.5B decode (M=1)
             *
             * hidden_size=1536, intermediate_size=8960
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, CKGemm_Qwen15B_Decode)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                struct TestCase
                {
                    const char *name;
                    int M, N, K;
                };
                std::vector<TestCase> cases = {
                    {"Wo_proj", 1, 1536, 1536},
                    {"FFN_gate", 1, 8960, 1536},
                    {"FFN_down", 1, 1536, 8960},
                };

                for (const auto &tc : cases)
                {
                    double cos = runGemmAndComputeCosineSimilarity(tc.M, tc.N, tc.K);
                    LOG_INFO("[Integration] Qwen1.5B decode " << tc.name << ": cos=" << cos);
                    EXPECT_GT(cos, 0.99) << "Qwen1.5B decode " << tc.name << " cosine too low";
                }
            }

            /**
             * @test Qwen2.5-0.5B prefill (M=64, 128, 256)
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, CKGemm_Qwen05B_Prefill)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                const int K = 896, N_attn = 896, N_ffn = 4864;

                for (int M : {64, 128, 256})
                {
                    double cos_attn = runGemmAndComputeCosineSimilarity(M, N_attn, K);
                    LOG_INFO("[Integration] Qwen0.5B prefill Wo M=" << M << ": cos=" << cos_attn);
                    EXPECT_GT(cos_attn, 0.99) << "Qwen0.5B prefill Wo M=" << M << " cosine too low";

                    double cos_ffn = runGemmAndComputeCosineSimilarity(M, N_ffn, K);
                    LOG_INFO("[Integration] Qwen0.5B prefill FFN M=" << M << ": cos=" << cos_ffn);
                    EXPECT_GT(cos_ffn, 0.99) << "Qwen0.5B prefill FFN M=" << M << " cosine too low";
                }
            }

            /**
             * @test Qwen2.5-1.5B prefill (M=64, 128, 256)
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, CKGemm_Qwen15B_Prefill)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                const int K = 1536, N_attn = 1536, N_ffn = 8960;

                for (int M : {64, 128, 256})
                {
                    double cos_attn = runGemmAndComputeCosineSimilarity(M, N_attn, K);
                    LOG_INFO("[Integration] Qwen1.5B prefill Wo M=" << M << ": cos=" << cos_attn);
                    EXPECT_GT(cos_attn, 0.99) << "Qwen1.5B prefill Wo M=" << M << " cosine too low";

                    double cos_ffn = runGemmAndComputeCosineSimilarity(M, N_ffn, K);
                    LOG_INFO("[Integration] Qwen1.5B prefill FFN M=" << M << ": cos=" << cos_ffn);
                    EXPECT_GT(cos_ffn, 0.99) << "Qwen1.5B prefill FFN M=" << M << " cosine too low";
                }
            }

            // =====================================================================
            // Full Pipeline Integration Tests
            // =====================================================================

            /**
             * @test Full weight pack + activation quantize pipeline
             *
             * Tests:
             * 1. Create Q8_1 weights, pack to INT8
             * 2. Create FP32 activations, quantize to INT8 on device
             * 3. Verify both sets of INT8 data are valid
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, WeightPackAndActivationQuantize)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                const int M = 64;  // batch size
                const int K = 512; // input features
                const int N = 256; // output features

                // Step 1: Pack weights (Q8_1 → INT8)
                auto weights = TestTensorFactory::createQ8_1Random({static_cast<size_t>(N), static_cast<size_t>(K)});
                ROCmPackedWeights packed;
                ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

                LOG_INFO("[Integration] Packed weights: " << N << "×" << K << " → INT8");

                // Step 2: Create FP32 activations
                auto activations = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
                const float *h_activations = activations->data();

                // Step 3: Allocate and upload activations to device
                float *d_activations = nullptr;
                ASSERT_TRUE(rocmQuantGemm_allocFloat(&d_activations, M * K, 0));
                ASSERT_TRUE(rocmQuantGemm_copyHostToDevice(d_activations, h_activations, M * K, 0));

                // Step 4: Allocate work buffers
                int8_t *d_A_int8 = nullptr;
                float *d_scales_A = nullptr;
                int32_t *d_C_int32 = nullptr;
                int work_buffer_M = 0;

                ASSERT_TRUE(rocmQuantGemm_ensureWorkBuffers(
                    &d_A_int8, &d_scales_A, &d_C_int32, &work_buffer_M,
                    M, K, N, 0));

                // Step 5: Quantize activations on device
                ASSERT_TRUE(rocmQuantGemm_quantizeActivations(
                    d_activations, d_A_int8, d_scales_A, M, K, 0));

                // Step 6: Verify quantized activations
                std::vector<int8_t> h_A_int8(M * K);
                std::vector<float> h_scales_A(M);

                ASSERT_EQ(hipMemcpy(h_A_int8.data(), d_A_int8, M * K * sizeof(int8_t),
                                    hipMemcpyDeviceToHost),
                          hipSuccess);
                ASSERT_EQ(hipMemcpy(h_scales_A.data(), d_scales_A, M * sizeof(float),
                                    hipMemcpyDeviceToHost),
                          hipSuccess);

                // Check all scales are positive
                for (int m = 0; m < M; ++m)
                {
                    EXPECT_GT(h_scales_A[m], 0.0f) << "Row " << m << " has non-positive scale";
                }

                // Check INT8 values in valid range
                for (int i = 0; i < M * K; ++i)
                {
                    EXPECT_GE(h_A_int8[i], -127);
                    EXPECT_LE(h_A_int8[i], 127);
                }

                // Verify reconstruction accuracy
                float max_error = 0.0f;
                for (int m = 0; m < M; ++m)
                {
                    float scale = h_scales_A[m];
                    for (int k = 0; k < K; ++k)
                    {
                        float original = h_activations[m * K + k];
                        float reconstructed = h_A_int8[m * K + k] * scale;
                        float error = std::abs(original - reconstructed);
                        max_error = std::max(max_error, error);
                    }
                }

                float max_scale = *std::max_element(h_scales_A.begin(), h_scales_A.end());
                EXPECT_LT(max_error, max_scale * 1.01f)
                    << "Activation reconstruction error too large";

                LOG_INFO("[Integration] Quantized activations: " << M << "×" << K
                                                                 << " → INT8, max_error=" << max_error);

                // Cleanup
                rocmQuantGemm_freeDevice(d_activations);
                rocmQuantGemm_freeDevice(d_A_int8);
                rocmQuantGemm_freeDevice(d_scales_A);
                rocmQuantGemm_freeDevice(d_C_int32);
            }

            /**
             * @test Realistic model dimensions (Qwen2.5-0.5B layer sizes)
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, RealisticModelDimensions)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                // Qwen2.5-0.5B dimensions:
                // - hidden_size = 896
                // - intermediate_size = 4864
                // - num_attention_heads = 14

                struct TestCase
                {
                    const char *name;
                    int M, K, N;
                };

                std::vector<TestCase> cases = {
                    {"QKV projection", 128, 896, 896 * 3}, // seq_len=128, hidden→QKV
                    {"Attention output", 128, 896, 896},   // context→output
                    {"FFN gate", 128, 896, 4864},          // hidden→intermediate
                    {"FFN up", 128, 896, 4864},            // hidden→intermediate
                    {"FFN down", 128, 4864, 896},          // intermediate→hidden
                    {"LM head decode", 1, 896, 32000},     // single token→vocab (decode)
                };

                for (const auto &tc : cases)
                {
                    LOG_INFO("[Integration] Testing " << tc.name << ": "
                                                      << tc.M << "×" << tc.K << "×" << tc.N);

                    // Pack weights
                    auto weights = TestTensorFactory::createQ8_1Random({static_cast<size_t>(tc.N), static_cast<size_t>(tc.K)});
                    ROCmPackedWeights packed;
                    ASSERT_TRUE(packWeightsToROCm(weights.get(), packed))
                        << "Failed to pack weights for " << tc.name;

                    // Create and quantize activations
                    auto activations = TestTensorFactory::createFP32Random({static_cast<size_t>(tc.M), static_cast<size_t>(tc.K)});

                    float *d_activations = nullptr;
                    int8_t *d_A_int8 = nullptr;
                    float *d_scales_A = nullptr;
                    int32_t *d_C_int32 = nullptr;
                    int work_buffer_M = 0;

                    ASSERT_TRUE(rocmQuantGemm_allocFloat(&d_activations, tc.M * tc.K, 0));
                    ASSERT_TRUE(rocmQuantGemm_copyHostToDevice(
                        d_activations, activations->data(), tc.M * tc.K, 0));
                    ASSERT_TRUE(rocmQuantGemm_ensureWorkBuffers(
                        &d_A_int8, &d_scales_A, &d_C_int32, &work_buffer_M,
                        tc.M, tc.K, tc.N, 0));
                    ASSERT_TRUE(rocmQuantGemm_quantizeActivations(
                        d_activations, d_A_int8, d_scales_A, tc.M, tc.K, 0))
                        << "Failed to quantize activations for " << tc.name;

                    // Cleanup
                    rocmQuantGemm_freeDevice(d_activations);
                    rocmQuantGemm_freeDevice(d_A_int8);
                    rocmQuantGemm_freeDevice(d_scales_A);
                    rocmQuantGemm_freeDevice(d_C_int32);
                }
            }

            /**
             * @test Multiple batch sizes (prefill and decode patterns)
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, PrefillAndDecodeBatchSizes)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                const int K = 896;
                const int N = 896;

                // Test various batch sizes from decode (M=1) to prefill (M=2048)
                std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

                auto weights = TestTensorFactory::createQ8_1Random({static_cast<size_t>(N), static_cast<size_t>(K)});
                ROCmPackedWeights packed;
                ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

                int8_t *d_A_int8 = nullptr;
                float *d_scales_A = nullptr;
                int32_t *d_C_int32 = nullptr;
                int work_buffer_M = 0;

                for (int M : batch_sizes)
                {
                    auto activations = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});

                    float *d_activations = nullptr;
                    ASSERT_TRUE(rocmQuantGemm_allocFloat(&d_activations, M * K, 0));
                    ASSERT_TRUE(rocmQuantGemm_copyHostToDevice(
                        d_activations, activations->data(), M * K, 0));

                    // Work buffers should grow as needed
                    ASSERT_TRUE(rocmQuantGemm_ensureWorkBuffers(
                        &d_A_int8, &d_scales_A, &d_C_int32, &work_buffer_M,
                        M, K, N, 0))
                        << "Failed to allocate work buffers for M=" << M;

                    EXPECT_GE(work_buffer_M, M) << "Work buffer too small for M=" << M;

                    ASSERT_TRUE(rocmQuantGemm_quantizeActivations(
                        d_activations, d_A_int8, d_scales_A, M, K, 0))
                        << "Failed to quantize for M=" << M;

                    rocmQuantGemm_freeDevice(d_activations);
                }

                // Cleanup shared buffers
                rocmQuantGemm_freeDevice(d_A_int8);
                rocmQuantGemm_freeDevice(d_scales_A);
                rocmQuantGemm_freeDevice(d_C_int32);

                LOG_INFO("[Integration] Tested batch sizes: 1 to 2048");
            }

            // =============================================================================
            // Full Pipeline Tests - Scaling with Bias
            // =============================================================================

            /**
             * @test rocmQuantGemm_applyScaling - Full epilogue with alpha, beta, bias
             *
             * Tests the scaling epilogue kernel that converts INT32 GEMM output to FP32
             * with optional alpha/beta and bias.
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, ApplyScaling_WithBias)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                const int M = 32;
                const int N = 64;

                // Allocate device buffers
                int32_t *d_C_int32 = nullptr;
                float *d_C_fp32_no_bias = nullptr;
                float *d_C_fp32_with_bias = nullptr;
                float *d_scales_A = nullptr;
                float *d_scales_B = nullptr;
                float *d_bias = nullptr;

                ASSERT_TRUE(rocmQuantGemm_allocFloat(reinterpret_cast<float **>(&d_C_int32), (M * N * sizeof(int32_t)) / sizeof(float), 0));
                ASSERT_TRUE(rocmQuantGemm_allocFloat(&d_C_fp32_no_bias, M * N, 0));
                ASSERT_TRUE(rocmQuantGemm_allocFloat(&d_C_fp32_with_bias, M * N, 0));
                ASSERT_TRUE(rocmQuantGemm_allocFloat(&d_scales_A, M, 0));
                ASSERT_TRUE(rocmQuantGemm_allocFloat(&d_scales_B, N, 0));
                ASSERT_TRUE(rocmQuantGemm_allocFloat(&d_bias, N, 0));

                // Create host data
                std::vector<int32_t> h_C_int32(M * N);
                std::vector<float> h_scales_A(M, 1.0f); // Unit scales for simplicity
                std::vector<float> h_scales_B(N, 1.0f);
                std::vector<float> h_bias(N, 0.5f); // Constant bias for verification

                // Fill INT32 with deterministic values
                for (int i = 0; i < M * N; ++i)
                {
                    h_C_int32[i] = (i % 100) - 50; // [-50, 49]
                }

                // Upload to device
                hipMemcpy(d_C_int32, h_C_int32.data(), M * N * sizeof(int32_t), hipMemcpyHostToDevice);
                hipMemcpy(d_scales_A, h_scales_A.data(), M * sizeof(float), hipMemcpyHostToDevice);
                hipMemcpy(d_scales_B, h_scales_B.data(), N * sizeof(float), hipMemcpyHostToDevice);
                hipMemcpy(d_bias, h_bias.data(), N * sizeof(float), hipMemcpyHostToDevice);

                // Apply scaling without bias
                ASSERT_TRUE(rocmQuantGemm_applyScaling(
                    d_C_int32, d_C_fp32_no_bias, d_scales_A, d_scales_B,
                    M, N, 1.0f, 0.0f, nullptr, nullptr, 0));

                // Apply scaling with bias
                ASSERT_TRUE(rocmQuantGemm_applyScaling(
                    d_C_int32, d_C_fp32_with_bias, d_scales_A, d_scales_B,
                    M, N, 1.0f, 0.0f, nullptr, d_bias, 0));

                // Download results
                std::vector<float> h_C_fp32_no_bias(M * N);
                std::vector<float> h_C_fp32_with_bias(M * N);
                hipMemcpy(h_C_fp32_no_bias.data(), d_C_fp32_no_bias, M * N * sizeof(float), hipMemcpyDeviceToHost);
                hipMemcpy(h_C_fp32_with_bias.data(), d_C_fp32_with_bias, M * N * sizeof(float), hipMemcpyDeviceToHost);

                // Verify: with_bias should be no_bias + bias[col]
                int matches = 0;
                for (int m = 0; m < M; ++m)
                {
                    for (int n = 0; n < N; ++n)
                    {
                        int idx = m * N + n;
                        float expected = h_C_fp32_no_bias[idx] + h_bias[n];
                        float actual = h_C_fp32_with_bias[idx];
                        if (std::abs(expected - actual) < 1e-4f)
                            matches++;
                    }
                }

                float match_rate = static_cast<float>(matches) / static_cast<float>(M * N);
                EXPECT_GT(match_rate, 0.99f) << "Bias addition mismatch";

                LOG_INFO("[Integration] Bias scaling match rate: " << (match_rate * 100.0f) << "%");

                // Cleanup
                rocmQuantGemm_freeDevice(d_C_int32);
                rocmQuantGemm_freeDevice(d_C_fp32_no_bias);
                rocmQuantGemm_freeDevice(d_C_fp32_with_bias);
                rocmQuantGemm_freeDevice(d_scales_A);
                rocmQuantGemm_freeDevice(d_scales_B);
                rocmQuantGemm_freeDevice(d_bias);
            }

            /**
             * @test rocmQuantGemm_applyScaling with alpha != 1.0
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, ApplyScaling_WithAlpha)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                const int M = 16;
                const int N = 32;
                const float alpha = 2.0f;

                int32_t *d_C_int32 = nullptr;
                float *d_C_fp32 = nullptr;
                float *d_scales_A = nullptr;
                float *d_scales_B = nullptr;

                ASSERT_TRUE(rocmQuantGemm_allocFloat(reinterpret_cast<float **>(&d_C_int32), (M * N * sizeof(int32_t)) / sizeof(float), 0));
                ASSERT_TRUE(rocmQuantGemm_allocFloat(&d_C_fp32, M * N, 0));
                ASSERT_TRUE(rocmQuantGemm_allocFloat(&d_scales_A, M, 0));
                ASSERT_TRUE(rocmQuantGemm_allocFloat(&d_scales_B, N, 0));

                // Simple test: C_int32 = 1, scales = 1, alpha = 2 -> result should be 2.0
                std::vector<int32_t> h_C_int32(M * N, 1);
                std::vector<float> h_scales_A(M, 1.0f);
                std::vector<float> h_scales_B(N, 1.0f);

                hipMemcpy(d_C_int32, h_C_int32.data(), M * N * sizeof(int32_t), hipMemcpyHostToDevice);
                hipMemcpy(d_scales_A, h_scales_A.data(), M * sizeof(float), hipMemcpyHostToDevice);
                hipMemcpy(d_scales_B, h_scales_B.data(), N * sizeof(float), hipMemcpyHostToDevice);

                ASSERT_TRUE(rocmQuantGemm_applyScaling(
                    d_C_int32, d_C_fp32, d_scales_A, d_scales_B,
                    M, N, alpha, 0.0f, nullptr, nullptr, 0));

                std::vector<float> h_C_fp32(M * N);
                hipMemcpy(h_C_fp32.data(), d_C_fp32, M * N * sizeof(float), hipMemcpyDeviceToHost);

                // All values should be 2.0 (1 * 1.0 * 1.0 * 2.0)
                int matches = 0;
                for (float v : h_C_fp32)
                {
                    if (std::abs(v - 2.0f) < 1e-5f)
                        matches++;
                }

                EXPECT_EQ(matches, M * N) << "Alpha scaling incorrect";

                rocmQuantGemm_freeDevice(d_C_int32);
                rocmQuantGemm_freeDevice(d_C_fp32);
                rocmQuantGemm_freeDevice(d_scales_A);
                rocmQuantGemm_freeDevice(d_scales_B);
            }

            // MScaling tests removed - covered by decode/prefill tests above

        } // namespace integration_test
    } // namespace rocm
} // namespace llaminar2
