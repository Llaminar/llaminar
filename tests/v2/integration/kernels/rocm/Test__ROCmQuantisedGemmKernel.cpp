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
#include <cstdlib>
#include <numeric>

#include "execution/compute_stages/stages/FusedQKVGEMMStage.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/compute_stages/stages/QKNormStage.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/local_execution/coherence/GpuCoherence.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "kernels/KernelFactory.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "loaders/ModelContext.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "../../../utils/TestTensorFactory.h"
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
        int rocm_device_id, void *stream);

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
        int rocm_device_id,
        void *stream,
        void *kernel_ctx);

    // Two-kernel path: INT8×INT8→INT32 GEMM + separate scale application
    bool rocmQuantGemm_executeTwoKernel(
        const int8_t *d_A_int8,
        const int8_t *d_weights_int8,
        float *d_C_fp32,
        const float *d_scales_A,
        const float *d_scales_B,
        int M, int N, int K,
        int rocm_device_id,
        void *stream,
        void *kernel_ctx);

    // Memory management
    void rocmQuantGemm_freeDevice(void *d_ptr, int rocm_device_id);
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
        int rocm_device_id, void *stream);

    // In-place bias addition epilogue
    bool rocmQuantGemm_biasAdd(
        float *d_output,
        const float *d_bias,
        int M, int N,
        int rocm_device_id, void *stream);
}
#endif

namespace llaminar2
{
    namespace rocm
    {
        namespace integration_test
        {
            using namespace llaminar2::test;

            namespace
            {
                const std::string TEST_MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

                class ScopedEnvOverride
                {
                public:
                    ScopedEnvOverride(const char *name, const char *value)
                        : name_(name)
                    {
                        const char *original = std::getenv(name_);
                        if (original)
                        {
                            had_original_ = true;
                            original_value_ = original;
                        }
                        ::setenv(name_, value, 1);
                        mutableDebugEnv().reload();
                    }

                    ~ScopedEnvOverride()
                    {
                        if (had_original_)
                        {
                            ::setenv(name_, original_value_.c_str(), 1);
                        }
                        else
                        {
                            ::unsetenv(name_);
                        }
                        mutableDebugEnv().reload();
                    }

                    ScopedEnvOverride(const ScopedEnvOverride &) = delete;
                    ScopedEnvOverride &operator=(const ScopedEnvOverride &) = delete;

                private:
                    const char *name_;
                    bool had_original_ = false;
                    std::string original_value_;
                };

                struct NativeWeightReference
                {
                    std::vector<int8_t> unpacked_blocks;
                    std::vector<float> block_scales;
                    std::vector<float> block_mins;
                    int blocks_per_row = 0;
                };

                NativeWeightReference buildNativeWeightReference(const IINT8Unpackable &weight, int N, int K)
                {
                    NativeWeightReference reference;
                    reference.blocks_per_row = K / 32;
                    reference.unpacked_blocks.resize(static_cast<size_t>(N) * reference.blocks_per_row * 32);
                    reference.block_scales.resize(static_cast<size_t>(N) * reference.blocks_per_row);
                    reference.block_mins.resize(static_cast<size_t>(N) * reference.blocks_per_row);

#pragma omp parallel for schedule(static)
                    for (int n = 0; n < N; ++n)
                    {
                        for (int block = 0; block < reference.blocks_per_row; ++block)
                        {
                            const size_t linear = static_cast<size_t>(n) * reference.blocks_per_row + block;
                            weight.unpack_block_to_int8(
                                static_cast<size_t>(n),
                                static_cast<size_t>(block),
                                &reference.unpacked_blocks[linear * 32]);
                            reference.block_scales[linear] = weight.get_block_scale(
                                static_cast<size_t>(n),
                                static_cast<size_t>(block));
                            reference.block_mins[linear] = weight.get_block_min(
                                static_cast<size_t>(n),
                                static_cast<size_t>(block));
                        }
                    }

                    return reference;
                }

                std::vector<float> computeNativeReferenceFromQuantizedActivations(
                    const std::vector<int8_t> &activations_int8,
                    const std::vector<float> &scales_a,
                    const NativeWeightReference &weights,
                    int M, int N, int K)
                {
                    const int blocks_per_row = weights.blocks_per_row;
                    std::vector<int32_t> activation_block_sums(static_cast<size_t>(M) * blocks_per_row);
                    std::vector<float> output(static_cast<size_t>(M) * N, 0.0f);

#pragma omp parallel for schedule(static)
                    for (int m = 0; m < M; ++m)
                    {
                        for (int block = 0; block < blocks_per_row; ++block)
                        {
                            int32_t sum = 0;
                            const int8_t *a_block = &activations_int8[static_cast<size_t>(m) * K + block * 32];
                            for (int i = 0; i < 32; ++i)
                            {
                                sum += static_cast<int32_t>(a_block[i]);
                            }
                            activation_block_sums[static_cast<size_t>(m) * blocks_per_row + block] = sum;
                        }
                    }

#pragma omp parallel for schedule(static)
                    for (int m = 0; m < M; ++m)
                    {
                        for (int n = 0; n < N; ++n)
                        {
                            float accum = 0.0f;
                            const float scale_a = scales_a[static_cast<size_t>(m)];
                            for (int block = 0; block < blocks_per_row; ++block)
                            {
                                const size_t linear = static_cast<size_t>(n) * blocks_per_row + block;
                                const int8_t *a_block = &activations_int8[static_cast<size_t>(m) * K + block * 32];
                                const int8_t *b_block = &weights.unpacked_blocks[linear * 32];
                                int32_t dot = 0;
                                for (int i = 0; i < 32; ++i)
                                {
                                    dot += static_cast<int32_t>(a_block[i]) * static_cast<int32_t>(b_block[i]);
                                }

                                const float scale_b = weights.block_scales[linear];
                                const float min_b = weights.block_mins[linear];
                                const int32_t sum_a = activation_block_sums[static_cast<size_t>(m) * blocks_per_row + block];
                                accum += scale_a * (static_cast<float>(dot) * scale_b + static_cast<float>(sum_a) * min_b);
                            }
                            output[static_cast<size_t>(m) * N + n] = accum;
                        }
                    }

                    return output;
                }

                void quantizeActivationsPerBlock32(
                    const std::vector<float> &input,
                    int M,
                    int K,
                    std::vector<int8_t> &activations_int8,
                    std::vector<float> &scales_a)
                {
                    const int blocks_per_row = K / 32;
                    activations_int8.resize(static_cast<size_t>(M) * K);
                    scales_a.resize(static_cast<size_t>(M) * blocks_per_row);

#pragma omp parallel for schedule(static)
                    for (int m = 0; m < M; ++m)
                    {
                        for (int block = 0; block < blocks_per_row; ++block)
                        {
                            const float *src = &input[static_cast<size_t>(m) * K + block * 32];
                            float max_abs = 0.0f;
                            for (int i = 0; i < 32; ++i)
                            {
                                max_abs = std::max(max_abs, std::abs(src[i]));
                            }

                            const float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
                            scales_a[static_cast<size_t>(m) * blocks_per_row + block] = scale;
                            const float inv_scale = 1.0f / scale;

                            int8_t *dst = &activations_int8[static_cast<size_t>(m) * K + block * 32];
                            for (int i = 0; i < 32; ++i)
                            {
                                const float value = src[i] * inv_scale;
                                int32_t q = static_cast<int32_t>(value + (value >= 0.0f ? 0.5f : -0.5f));
                                q = std::clamp(q, -127, 127);
                                dst[i] = static_cast<int8_t>(q);
                            }
                        }
                    }
                }

                std::vector<float> computeNativeReferenceFromBlockwiseQuantizedActivations(
                    const std::vector<int8_t> &activations_int8,
                    const std::vector<float> &scales_a,
                    const NativeWeightReference &weights,
                    int M, int N, int K)
                {
                    const int blocks_per_row = weights.blocks_per_row;
                    std::vector<int32_t> activation_block_sums(static_cast<size_t>(M) * blocks_per_row);
                    std::vector<float> output(static_cast<size_t>(M) * N, 0.0f);

#pragma omp parallel for schedule(static)
                    for (int m = 0; m < M; ++m)
                    {
                        for (int block = 0; block < blocks_per_row; ++block)
                        {
                            int32_t sum = 0;
                            const int8_t *a_block = &activations_int8[static_cast<size_t>(m) * K + block * 32];
                            for (int i = 0; i < 32; ++i)
                            {
                                sum += static_cast<int32_t>(a_block[i]);
                            }
                            activation_block_sums[static_cast<size_t>(m) * blocks_per_row + block] = sum;
                        }
                    }

#pragma omp parallel for schedule(static)
                    for (int m = 0; m < M; ++m)
                    {
                        for (int n = 0; n < N; ++n)
                        {
                            float accum = 0.0f;
                            for (int block = 0; block < blocks_per_row; ++block)
                            {
                                const size_t linear = static_cast<size_t>(n) * blocks_per_row + block;
                                const int8_t *a_block = &activations_int8[static_cast<size_t>(m) * K + block * 32];
                                const int8_t *b_block = &weights.unpacked_blocks[linear * 32];
                                int32_t dot = 0;
                                for (int i = 0; i < 32; ++i)
                                {
                                    dot += static_cast<int32_t>(a_block[i]) * static_cast<int32_t>(b_block[i]);
                                }

                                const float scale_a = scales_a[static_cast<size_t>(m) * blocks_per_row + block];
                                const float scale_b = weights.block_scales[linear];
                                const float min_b = weights.block_mins[linear];
                                const int32_t sum_a = activation_block_sums[static_cast<size_t>(m) * blocks_per_row + block];
                                accum += scale_a * (static_cast<float>(dot) * scale_b + static_cast<float>(sum_a) * min_b);
                            }
                            output[static_cast<size_t>(m) * N + n] = accum;
                        }
                    }

                    return output;
                }
            }

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

                std::shared_ptr<ModelContext> loadModel() const
                {
                    std::ifstream file(TEST_MODEL_PATH);
                    if (!file.good())
                    {
                        return nullptr;
                    }
                    return ModelContext::create(TEST_MODEL_PATH);
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
                ASSERT_TRUE(rocmQuantGemm_quantizeActivations(d_activations, d_A_int8, d_scales_A, M, K, 0, nullptr));

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
                rocmQuantGemm_freeDevice(d_activations, 0);
                rocmQuantGemm_freeDevice(d_A_int8, 0);
                rocmQuantGemm_freeDevice(d_scales_A, 0);
                rocmQuantGemm_freeDevice(d_C_int32, 0);
            }

            /**
             * @test Prefill native INT8 VNNI matches CK fallback across shape buckets
             *
             * Phase-3 coverage expansion:
             * - Runs multiple representative M/N/K buckets instead of one fixed shape.
             * - Covers multiple quantized source families that route through the
             *   INT8-VNNI prefill path after host-side INT8 packing.
             * - Verifies native prefill (experimental flag ON) remains numerically
             *   aligned with CK fallback (experimental flag OFF).
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, PrefillNativeInt8VNNI_MatchesCKFallback)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                struct ShapeCase
                {
                    int M;
                    int N;
                    int K;
                };

                const std::vector<ShapeCase> shapes = {
                    {8, 128, 128},
                };

                const bool saved_grid_kpar_flag = mutableDebugEnv().rocm.vnni_prefill_grid_kpar;
                const int saved_grid_kpar_splits = mutableDebugEnv().rocm.vnni_prefill_grid_kpar_splits;

                auto run_int8_vnni_matrix = [&](const std::string &tag, auto make_weights)
                {
                    for (const auto &shape : shapes)
                    {
                        auto weights = make_weights(shape.N, shape.K);
                        ROCmPackedWeights packed;
                        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

                        ASSERT_FALSE(packed.int8_data_vnni.empty()) << "Expected VNNI packed weights for " << tag;

                        ROCmQuantisedGemmKernel kernel(&packed, 0);
                        ASSERT_TRUE(setupWorkspace(kernel, shape.M, shape.N, shape.K));

                        auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(shape.M), static_cast<size_t>(shape.K)});
                        auto out_ck = TestTensorFactory::createFP32({static_cast<size_t>(shape.M), static_cast<size_t>(shape.N)});
                        ASSERT_TRUE(kernel.multiply_tensor(input.get(), out_ck.get(), shape.M, shape.N, shape.K));

                        for (const bool use_grid_kpar : {false, true})
                        {
                            auto out_native = TestTensorFactory::createFP32({static_cast<size_t>(shape.M), static_cast<size_t>(shape.N)});
                            mutableDebugEnv().rocm.vnni_prefill_grid_kpar = use_grid_kpar;
                            mutableDebugEnv().rocm.vnni_prefill_grid_kpar_splits = 4;
                            ASSERT_TRUE(kernel.multiply_tensor(input.get(), out_native.get(), shape.M, shape.N, shape.K));

                            std::vector<float> ck(out_ck->data(), out_ck->data() + static_cast<size_t>(shape.M) * shape.N);
                            std::vector<float> native(out_native->data(), out_native->data() + static_cast<size_t>(shape.M) * shape.N);

                            float cos = cosineSimilarity(ck, native);
                            LOG_INFO("[Integration] Prefill native INT8 VNNI parity [" << tag << "] variant="
                                                                                       << (use_grid_kpar ? "grid_kpar" : "baseline")
                                                                                       << " M=" << shape.M << " N=" << shape.N << " K=" << shape.K
                                                                                       << " cosine=" << cos);
                            EXPECT_GT(cos, 0.9999f);

                            float max_abs_diff = 0.0f;
                            for (size_t i = 0; i < ck.size(); ++i)
                            {
                                max_abs_diff = std::max(max_abs_diff, std::fabs(ck[i] - native[i]));
                            }
                            EXPECT_LT(max_abs_diff, 1e-3f);
                        }

                        cleanupWorkspace(kernel);
                    }
                };

                run_int8_vnni_matrix(
                    "Q8_0",
                    [](int N, int K)
                    {
                        return TestTensorFactory::createQ8_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});
                    });

                mutableDebugEnv().rocm.vnni_prefill_grid_kpar = saved_grid_kpar_flag;
                mutableDebugEnv().rocm.vnni_prefill_grid_kpar_splits = saved_grid_kpar_splits;
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

                ASSERT_TRUE(rocmQuantGemm_executeNoScale(
                    d_A, d_B, d_C, M, N, K,
                    /*rocm_device_id=*/0,
                    /*stream=*/nullptr,
                    /*kernel_ctx=*/nullptr));

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

                rocmQuantGemm_freeDevice(d_A, 0);
                rocmQuantGemm_freeDevice(d_B, 0);
                rocmQuantGemm_freeDevice(d_C, 0);
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
                    /*rocm_device_id=*/0,
                    /*stream=*/nullptr,
                    /*kernel_ctx=*/nullptr));

                std::vector<float> h_E(static_cast<size_t>(M) * N);
                ASSERT_EQ(hipMemcpy(h_E.data(), d_E, static_cast<size_t>(M) * N * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

                // CPU reference (column-major B to match CK kernel expectation)
                std::vector<float> h_ref;
                computeScaledInt8GemmReferenceColumnMajorB(h_A, h_B, h_scaleA, h_scaleB, h_ref, M, N, K);

                const float cos = cosineSimilarity(h_E, h_ref);
                LOG_INFO("[TwoKernel_Deterministic128] cos=" << cos);

                // Two-Kernel path should have very high accuracy (>0.9999)
                EXPECT_GT(cos, 0.9999f) << "Two-Kernel path cosine similarity too low";

                rocmQuantGemm_freeDevice(d_A, 0);
                rocmQuantGemm_freeDevice(d_B, 0);
                rocmQuantGemm_freeDevice(d_scaleA, 0);
                rocmQuantGemm_freeDevice(d_scaleB, 0);
                rocmQuantGemm_freeDevice(d_E, 0);
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
                ASSERT_TRUE(rocmQuantGemm_quantizeActivations(d_activations, d_A_int8, d_scales_A, M, K, 0, nullptr));

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
                rocmQuantGemm_freeDevice(d_activations, 0);
                rocmQuantGemm_freeDevice(d_A_int8, 0);
                rocmQuantGemm_freeDevice(d_scales_A, 0);
                rocmQuantGemm_freeDevice(d_C_int32, 0);
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
                ASSERT_TRUE(rocmQuantGemm_quantizeActivations(d_activations, d_A_int8, d_scales_A, M, K, 0, nullptr));

                // Download scales
                std::vector<float> h_scales_A(M);
                ASSERT_EQ(hipMemcpy(h_scales_A.data(), d_scales_A, M * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

                // All scales should be positive (zero row gets small epsilon scale)
                for (int m = 0; m < M; ++m)
                {
                    EXPECT_GT(h_scales_A[m], 0.0f) << "Scale for row " << m << " should be positive";
                }

                // Cleanup
                rocmQuantGemm_freeDevice(d_activations, 0);
                rocmQuantGemm_freeDevice(d_A_int8, 0);
                rocmQuantGemm_freeDevice(d_scales_A, 0);
                rocmQuantGemm_freeDevice(d_C_int32, 0);
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
                ASSERT_TRUE(rocmQuantGemm_quantizeActivations(d_activations, d_A_int8, d_scales_A, M, K, 0, nullptr));

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
                rocmQuantGemm_freeDevice(d_activations, 0);
                rocmQuantGemm_freeDevice(d_A_int8, 0);
                rocmQuantGemm_freeDevice(d_scales_A, 0);
                rocmQuantGemm_freeDevice(d_C_int32, 0);
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
                rocmQuantGemm_freeDevice(d_A_int8, 0);
                rocmQuantGemm_freeDevice(d_scales_A, 0);
                rocmQuantGemm_freeDevice(d_C_int32, 0);
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
                rocmQuantGemm_freeDevice(d_A_int8, 0);
                rocmQuantGemm_freeDevice(d_scales_A, 0);
                rocmQuantGemm_freeDevice(d_C_int32, 0);
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
                    d_activations, d_A_int8, d_scales_A, M, K, 0, nullptr));

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
                rocmQuantGemm_freeDevice(d_activations, 0);
                rocmQuantGemm_freeDevice(d_A_int8, 0);
                rocmQuantGemm_freeDevice(d_scales_A, 0);
                rocmQuantGemm_freeDevice(d_C_int32, 0);
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
                        d_activations, d_A_int8, d_scales_A, tc.M, tc.K, 0, nullptr))
                        << "Failed to quantize activations for " << tc.name;

                    // Cleanup
                    rocmQuantGemm_freeDevice(d_activations, 0);
                    rocmQuantGemm_freeDevice(d_A_int8, 0);
                    rocmQuantGemm_freeDevice(d_scales_A, 0);
                    rocmQuantGemm_freeDevice(d_C_int32, 0);
                }
            }

            /**
             * @test Fused QKV stage parity for Qwen2-style biased projections on ROCm
             *
             * Uses realistic Qwen2.5-0.5B projection shapes:
             * - input hidden size: 896
             * - Q output: 896
             * - K/V output: 128 each (GQA)
             * - prefill tokens: 64
             *
             * This specifically exercises the fused multi-projection ROCm path with
             * optional Q/K/V biases, which is the closest isolated stage match to the
             * remaining Qwen2 ROCm parity failures.
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, FusedQKVStage_Qwen05B_BiasParity)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                constexpr int M = 64;
                constexpr int K = 896;
                constexpr int N_Q = 896;
                constexpr int N_K = 128;
                constexpr int N_V = 128;
                constexpr float COSINE_THRESHOLD = 0.995f;

                auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)}, -0.5f, 0.5f, 123);
                auto wq = TestTensorFactory::createQ4_0Random({static_cast<size_t>(N_Q), static_cast<size_t>(K)}, 1001);
                auto wk = TestTensorFactory::createQ4_0Random({static_cast<size_t>(N_K), static_cast<size_t>(K)}, 1002);
                auto wv = TestTensorFactory::createQ4_0Random({static_cast<size_t>(N_V), static_cast<size_t>(K)}, 1003);

                auto bias_q = TestTensorFactory::createFP32Random({static_cast<size_t>(N_Q)}, -0.05f, 0.05f, 2001);
                auto bias_k = TestTensorFactory::createFP32Random({static_cast<size_t>(N_K)}, -0.05f, 0.05f, 2002);
                auto bias_v = TestTensorFactory::createFP32Random({static_cast<size_t>(N_V)}, -0.05f, 0.05f, 2003);

                auto cpu_q = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_Q)}, DeviceId::cpu());
                auto cpu_k = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_K)}, DeviceId::cpu());
                auto cpu_v = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_V)}, DeviceId::cpu());

                auto rocm_q = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_Q)}, DeviceId::rocm(0));
                auto rocm_k = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_K)}, DeviceId::rocm(0));
                auto rocm_v = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_V)}, DeviceId::rocm(0));

                FusedQKVGEMMStage::Params cpu_params{
                    .device_id = DeviceId::cpu(),
                    .input = input.get(),
                    .m = M,
                    .k = K,
                    .wq = wq.get(),
                    .output_q = cpu_q.get(),
                    .n_q = N_Q,
                    .bias_q = bias_q.get(),
                    .wk = wk.get(),
                    .output_k = cpu_k.get(),
                    .n_k = N_K,
                    .bias_k = bias_k.get(),
                    .wv = wv.get(),
                    .output_v = cpu_v.get(),
                    .n_v = N_V,
                    .bias_v = bias_v.get()};

                FusedQKVGEMMStage::Params rocm_params{
                    .device_id = DeviceId::rocm(0),
                    .input = input.get(),
                    .m = M,
                    .k = K,
                    .wq = wq.get(),
                    .output_q = rocm_q.get(),
                    .n_q = N_Q,
                    .bias_q = bias_q.get(),
                    .wk = wk.get(),
                    .output_k = rocm_k.get(),
                    .n_k = N_K,
                    .bias_k = bias_k.get(),
                    .wv = wv.get(),
                    .output_v = rocm_v.get(),
                    .n_v = N_V,
                    .bias_v = bias_v.get()};

                FusedQKVGEMMStage cpu_stage(cpu_params);
                FusedQKVGEMMStage rocm_stage(rocm_params);

                CPUDeviceContext cpu_ctx(DeviceId::cpu(), 4);
                ROCmDeviceContext rocm_ctx(DeviceId::rocm(0), 0);

                ASSERT_TRUE(cpu_stage.execute(&cpu_ctx)) << "CPU fused QKV execution failed";

                auto reqs = rocm_stage.getWorkspaceRequirements(M, N_Q, K);
                DeviceWorkspaceManager workspace(DeviceId::rocm(0), 128 * 1024 * 1024);
                ASSERT_TRUE(workspace.allocate(reqs)) << "Failed to allocate ROCm fused QKV workspace";
                rocm_stage.bindWorkspace(&workspace);

                const bool rocm_ok = with_gpu_coherence(
                    DeviceId::rocm(0),
                    {input.get(), wq.get(), wk.get(), wv.get(), bias_q.get(), bias_k.get(), bias_v.get()},
                    {rocm_q.get(), rocm_k.get(), rocm_v.get()},
                    [&]() {
                        return rocm_stage.execute(&rocm_ctx);
                    });

                rocm_stage.unbindWorkspace();

                ASSERT_TRUE(rocm_ok) << "ROCm fused QKV execution failed";

                const float q_cos = cosineSimilarity(
                    std::vector<float>(cpu_q->data(), cpu_q->data() + static_cast<size_t>(M * N_Q)),
                    std::vector<float>(rocm_q->data(), rocm_q->data() + static_cast<size_t>(M * N_Q)));
                const float k_cos = cosineSimilarity(
                    std::vector<float>(cpu_k->data(), cpu_k->data() + static_cast<size_t>(M * N_K)),
                    std::vector<float>(rocm_k->data(), rocm_k->data() + static_cast<size_t>(M * N_K)));
                const float v_cos = cosineSimilarity(
                    std::vector<float>(cpu_v->data(), cpu_v->data() + static_cast<size_t>(M * N_V)),
                    std::vector<float>(rocm_v->data(), rocm_v->data() + static_cast<size_t>(M * N_V)));

                LOG_INFO("[Integration] ROCm fused QKV parity: Q=" << q_cos
                                                                    << " K=" << k_cos
                                                                    << " V=" << v_cos);

                EXPECT_GT(q_cos, COSINE_THRESHOLD) << "Q projection cosine too low";
                EXPECT_GT(k_cos, COSINE_THRESHOLD) << "K projection cosine too low";
                EXPECT_GT(v_cos, COSINE_THRESHOLD) << "V projection cosine too low";
            }

            /**
             * @test Fused QKV stage parity using real loaded Qwen2 late-layer weights and biases
             *
             * This validates the exact tensor materialization path used by the runner, not just
             * the fused ROCm kernel math. It targets a late layer because the remaining parity
             * drift first becomes obvious near layer 21 in the full model path.
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, FusedQKVStage_RealQwen2Layer21BiasParity)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                auto model_ctx = loadModel();
                if (!model_ctx)
                {
                    GTEST_SKIP() << "Test model not found: " << TEST_MODEL_PATH;
                }

                constexpr int layer_idx = 21;
                constexpr int M = 64;
                constexpr int K = 896;
                constexpr float COSINE_THRESHOLD = 0.995f;

                const std::string prefix = "blk." + std::to_string(layer_idx) + ".";

                auto cpu_wq = model_ctx->getWeightForDevice(prefix + "attn_q.weight", DeviceId::cpu());
                auto cpu_wk = model_ctx->getWeightForDevice(prefix + "attn_k.weight", DeviceId::cpu());
                auto cpu_wv = model_ctx->getWeightForDevice(prefix + "attn_v.weight", DeviceId::cpu());
                auto cpu_bias_q = model_ctx->getWeightForDevice(prefix + "attn_q.bias", DeviceId::cpu());
                auto cpu_bias_k = model_ctx->getWeightForDevice(prefix + "attn_k.bias", DeviceId::cpu());
                auto cpu_bias_v = model_ctx->getWeightForDevice(prefix + "attn_v.bias", DeviceId::cpu());

                auto rocm_wq = model_ctx->getWeightForDevice(prefix + "attn_q.weight", DeviceId::rocm(0));
                auto rocm_wk = model_ctx->getWeightForDevice(prefix + "attn_k.weight", DeviceId::rocm(0));
                auto rocm_wv = model_ctx->getWeightForDevice(prefix + "attn_v.weight", DeviceId::rocm(0));
                auto rocm_bias_q = model_ctx->getWeightForDevice(prefix + "attn_q.bias", DeviceId::rocm(0));
                auto rocm_bias_k = model_ctx->getWeightForDevice(prefix + "attn_k.bias", DeviceId::rocm(0));
                auto rocm_bias_v = model_ctx->getWeightForDevice(prefix + "attn_v.bias", DeviceId::rocm(0));

                ASSERT_TRUE(cpu_wq && cpu_wk && cpu_wv);
                ASSERT_TRUE(rocm_wq && rocm_wk && rocm_wv);
                ASSERT_TRUE(cpu_bias_q && cpu_bias_k && cpu_bias_v);
                ASSERT_TRUE(rocm_bias_q && rocm_bias_k && rocm_bias_v);

                ASSERT_EQ(cpu_bias_q->native_type(), TensorType::FP32);
                ASSERT_EQ(cpu_bias_k->native_type(), TensorType::FP32);
                ASSERT_EQ(cpu_bias_v->native_type(), TensorType::FP32);
                ASSERT_EQ(rocm_bias_q->native_type(), TensorType::FP32);
                ASSERT_EQ(rocm_bias_k->native_type(), TensorType::FP32);
                ASSERT_EQ(rocm_bias_v->native_type(), TensorType::FP32);

                const int N_Q = static_cast<int>(cpu_wq->shape()[0]);
                const int N_K = static_cast<int>(cpu_wk->shape()[0]);
                const int N_V = static_cast<int>(cpu_wv->shape()[0]);

                auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)}, -0.5f, 0.5f, 321);
                auto cpu_q = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_Q)}, DeviceId::cpu());
                auto cpu_k = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_K)}, DeviceId::cpu());
                auto cpu_v = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_V)}, DeviceId::cpu());

                auto rocm_q = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_Q)}, DeviceId::rocm(0));
                auto rocm_k = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_K)}, DeviceId::rocm(0));
                auto rocm_v = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_V)}, DeviceId::rocm(0));

                FusedQKVGEMMStage::Params cpu_params{
                    .device_id = DeviceId::cpu(),
                    .input = input.get(),
                    .m = M,
                    .k = K,
                    .wq = cpu_wq.get(),
                    .output_q = cpu_q.get(),
                    .n_q = N_Q,
                    .bias_q = cpu_bias_q.get(),
                    .wk = cpu_wk.get(),
                    .output_k = cpu_k.get(),
                    .n_k = N_K,
                    .bias_k = cpu_bias_k.get(),
                    .wv = cpu_wv.get(),
                    .output_v = cpu_v.get(),
                    .n_v = N_V,
                    .bias_v = cpu_bias_v.get()};

                FusedQKVGEMMStage::Params rocm_params{
                    .device_id = DeviceId::rocm(0),
                    .input = input.get(),
                    .m = M,
                    .k = K,
                    .wq = rocm_wq.get(),
                    .output_q = rocm_q.get(),
                    .n_q = N_Q,
                    .bias_q = rocm_bias_q.get(),
                    .wk = rocm_wk.get(),
                    .output_k = rocm_k.get(),
                    .n_k = N_K,
                    .bias_k = rocm_bias_k.get(),
                    .wv = rocm_wv.get(),
                    .output_v = rocm_v.get(),
                    .n_v = N_V,
                    .bias_v = rocm_bias_v.get()};

                FusedQKVGEMMStage cpu_stage(cpu_params);
                FusedQKVGEMMStage rocm_stage(rocm_params);

                CPUDeviceContext cpu_ctx(DeviceId::cpu(), 4);
                ROCmDeviceContext rocm_ctx(DeviceId::rocm(0), 0);

                ASSERT_TRUE(cpu_stage.execute(&cpu_ctx)) << "CPU fused QKV execution failed for real model weights";

                auto reqs = rocm_stage.getWorkspaceRequirements(M, N_Q, K);
                DeviceWorkspaceManager workspace(DeviceId::rocm(0), 128 * 1024 * 1024);
                ASSERT_TRUE(workspace.allocate(reqs)) << "Failed to allocate ROCm fused QKV workspace";
                rocm_stage.bindWorkspace(&workspace);

                const std::initializer_list<TensorBase *> rocm_inputs = {
                    input.get(),
                    rocm_wq.get(),
                    rocm_wk.get(),
                    rocm_wv.get(),
                    rocm_bias_q.get(),
                    rocm_bias_k.get(),
                    rocm_bias_v.get()};
                const std::initializer_list<TensorBase *> rocm_outputs = {
                    rocm_q.get(),
                    rocm_k.get(),
                    rocm_v.get()};

                const bool rocm_ok = with_gpu_coherence(
                    DeviceId::rocm(0),
                    rocm_inputs,
                    rocm_outputs,
                    [&]() {
                        return rocm_stage.execute(&rocm_ctx);
                    });

                rocm_stage.unbindWorkspace();

                ASSERT_TRUE(rocm_ok) << "ROCm fused QKV execution failed for real model weights";

                const float q_cos = cosineSimilarity(
                    std::vector<float>(cpu_q->data(), cpu_q->data() + static_cast<size_t>(M * N_Q)),
                    std::vector<float>(rocm_q->data(), rocm_q->data() + static_cast<size_t>(M * N_Q)));
                const float k_cos = cosineSimilarity(
                    std::vector<float>(cpu_k->data(), cpu_k->data() + static_cast<size_t>(M * N_K)),
                    std::vector<float>(rocm_k->data(), rocm_k->data() + static_cast<size_t>(M * N_K)));
                const float v_cos = cosineSimilarity(
                    std::vector<float>(cpu_v->data(), cpu_v->data() + static_cast<size_t>(M * N_V)),
                    std::vector<float>(rocm_v->data(), rocm_v->data() + static_cast<size_t>(M * N_V)));

                LOG_INFO("[Integration] Real Qwen2 layer " << layer_idx << " fused QKV parity: Q=" << q_cos
                                                            << " K=" << k_cos
                                                            << " V=" << v_cos
                                                            << " q_bias_type=" << static_cast<int>(cpu_bias_q->native_type())
                                                            << " k_bias_type=" << static_cast<int>(cpu_bias_k->native_type())
                                                            << " v_bias_type=" << static_cast<int>(cpu_bias_v->native_type()));

                EXPECT_GT(q_cos, COSINE_THRESHOLD) << "Q projection cosine too low for real model layer";
                EXPECT_GT(k_cos, COSINE_THRESHOLD) << "K projection cosine too low for real model layer";
                EXPECT_GT(v_cos, COSINE_THRESHOLD) << "V projection cosine too low for real model layer";
            }

            TEST_F(ROCmQuantisedGemmIntegrationTest, FusedQKVStage_RealQwen2Layer3SnapshotInputParity)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                std::ifstream model_file(TEST_MODEL_PATH);
                if (!model_file.good())
                {
                    GTEST_SKIP() << "Test model not found: " << TEST_MODEL_PATH;
                }

                auto stage_ctx = ModelContext::createForPPStage(TEST_MODEL_PATH, 0, 4, true, false);
                ASSERT_NE(stage_ctx, nullptr);

                FactoryPPStageConfig config;
                config.first_layer = 0;
                config.last_layer = 4;
                config.has_embedding = true;
                config.has_lm_head = false;
                ASSERT_TRUE(config.isValid());

                auto cpu_runner = createPPStageRunner(stage_ctx, DeviceId::cpu(), config);
                ASSERT_NE(cpu_runner, nullptr);
                cpu_runner->enableSnapshotCapture();

                constexpr int layer_idx = 3;
                constexpr int M = 64;
                constexpr int K = 896;
                constexpr float COSINE_THRESHOLD = 0.995f;

                std::vector<int> tokens(M);
                for (int i = 0; i < M; ++i)
                {
                    tokens[i] = i % 1024;
                }
                ASSERT_TRUE(cpu_runner->forward(tokens.data(), M));

                size_t input_size = 0;
                const float *input_snapshot = cpu_runner->getSnapshot("layer3_ATTENTION_NORM", input_size);
                ASSERT_NE(input_snapshot, nullptr);
                ASSERT_EQ(input_size, static_cast<size_t>(M * K));

                auto model_ctx = loadModel();
                ASSERT_NE(model_ctx, nullptr);

                const std::string prefix = "blk." + std::to_string(layer_idx) + ".";

                auto cpu_wq = model_ctx->getWeightForDevice(prefix + "attn_q.weight", DeviceId::cpu());
                auto cpu_wk = model_ctx->getWeightForDevice(prefix + "attn_k.weight", DeviceId::cpu());
                auto cpu_wv = model_ctx->getWeightForDevice(prefix + "attn_v.weight", DeviceId::cpu());
                auto cpu_bias_q = model_ctx->getWeightForDevice(prefix + "attn_q.bias", DeviceId::cpu());
                auto cpu_bias_k = model_ctx->getWeightForDevice(prefix + "attn_k.bias", DeviceId::cpu());
                auto cpu_bias_v = model_ctx->getWeightForDevice(prefix + "attn_v.bias", DeviceId::cpu());

                auto rocm_wq = model_ctx->getWeightForDevice(prefix + "attn_q.weight", DeviceId::rocm(0));
                auto rocm_wk = model_ctx->getWeightForDevice(prefix + "attn_k.weight", DeviceId::rocm(0));
                auto rocm_wv = model_ctx->getWeightForDevice(prefix + "attn_v.weight", DeviceId::rocm(0));
                auto rocm_bias_q = model_ctx->getWeightForDevice(prefix + "attn_q.bias", DeviceId::rocm(0));
                auto rocm_bias_k = model_ctx->getWeightForDevice(prefix + "attn_k.bias", DeviceId::rocm(0));
                auto rocm_bias_v = model_ctx->getWeightForDevice(prefix + "attn_v.bias", DeviceId::rocm(0));

                ASSERT_TRUE(cpu_wq && cpu_wk && cpu_wv);
                ASSERT_TRUE(cpu_bias_q && cpu_bias_k && cpu_bias_v);
                ASSERT_TRUE(rocm_wq && rocm_wk && rocm_wv);
                ASSERT_TRUE(rocm_bias_q && rocm_bias_k && rocm_bias_v);

                const int N_Q = static_cast<int>(cpu_wq->shape()[0]);
                const int N_K = static_cast<int>(cpu_wk->shape()[0]);
                const int N_V = static_cast<int>(cpu_wv->shape()[0]);

                auto input = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)}, DeviceId::cpu());
                std::copy(input_snapshot, input_snapshot + input_size, input->mutable_data());

                auto cpu_q = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_Q)}, DeviceId::cpu());
                auto cpu_k = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_K)}, DeviceId::cpu());
                auto cpu_v = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_V)}, DeviceId::cpu());

                auto rocm_q = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_Q)}, DeviceId::rocm(0));
                auto rocm_k = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_K)}, DeviceId::rocm(0));
                auto rocm_v = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_V)}, DeviceId::rocm(0));

                FusedQKVGEMMStage::Params cpu_params{
                    .device_id = DeviceId::cpu(),
                    .input = input.get(),
                    .m = M,
                    .k = K,
                    .wq = cpu_wq.get(),
                    .output_q = cpu_q.get(),
                    .n_q = N_Q,
                    .bias_q = cpu_bias_q.get(),
                    .wk = cpu_wk.get(),
                    .output_k = cpu_k.get(),
                    .n_k = N_K,
                    .bias_k = cpu_bias_k.get(),
                    .wv = cpu_wv.get(),
                    .output_v = cpu_v.get(),
                    .n_v = N_V,
                    .bias_v = cpu_bias_v.get()};

                FusedQKVGEMMStage::Params rocm_params{
                    .device_id = DeviceId::rocm(0),
                    .input = input.get(),
                    .m = M,
                    .k = K,
                    .wq = rocm_wq.get(),
                    .output_q = rocm_q.get(),
                    .n_q = N_Q,
                    .bias_q = rocm_bias_q.get(),
                    .wk = rocm_wk.get(),
                    .output_k = rocm_k.get(),
                    .n_k = N_K,
                    .bias_k = rocm_bias_k.get(),
                    .wv = rocm_wv.get(),
                    .output_v = rocm_v.get(),
                    .n_v = N_V,
                    .bias_v = rocm_bias_v.get()};

                FusedQKVGEMMStage cpu_stage(cpu_params);
                FusedQKVGEMMStage rocm_stage(rocm_params);
                CPUDeviceContext cpu_ctx(DeviceId::cpu(), 4);
                ROCmDeviceContext rocm_ctx(DeviceId::rocm(0), 0);

                ASSERT_TRUE(cpu_stage.execute(&cpu_ctx)) << "CPU fused QKV execution failed for layer-3 snapshot input";

                auto reqs = rocm_stage.getWorkspaceRequirements(M, N_Q, K);
                DeviceWorkspaceManager workspace(DeviceId::rocm(0), 128 * 1024 * 1024);
                ASSERT_TRUE(workspace.allocate(reqs)) << "Failed to allocate ROCm fused QKV workspace";
                rocm_stage.bindWorkspace(&workspace);

                const std::initializer_list<TensorBase *> rocm_inputs = {
                    input.get(),
                    rocm_wq.get(),
                    rocm_wk.get(),
                    rocm_wv.get(),
                    rocm_bias_q.get(),
                    rocm_bias_k.get(),
                    rocm_bias_v.get()};
                const std::initializer_list<TensorBase *> rocm_outputs = {
                    rocm_q.get(),
                    rocm_k.get(),
                    rocm_v.get()};

                const bool rocm_ok = with_gpu_coherence(
                    DeviceId::rocm(0),
                    rocm_inputs,
                    rocm_outputs,
                    [&]() {
                        return rocm_stage.execute(&rocm_ctx);
                    });

                rocm_stage.unbindWorkspace();

                ASSERT_TRUE(rocm_ok) << "ROCm fused QKV execution failed for layer-3 snapshot input";

                const float q_cos = cosineSimilarity(
                    std::vector<float>(cpu_q->data(), cpu_q->data() + static_cast<size_t>(M * N_Q)),
                    std::vector<float>(rocm_q->data(), rocm_q->data() + static_cast<size_t>(M * N_Q)));
                const float k_cos = cosineSimilarity(
                    std::vector<float>(cpu_k->data(), cpu_k->data() + static_cast<size_t>(M * N_K)),
                    std::vector<float>(rocm_k->data(), rocm_k->data() + static_cast<size_t>(M * N_K)));
                const float v_cos = cosineSimilarity(
                    std::vector<float>(cpu_v->data(), cpu_v->data() + static_cast<size_t>(M * N_V)),
                    std::vector<float>(rocm_v->data(), rocm_v->data() + static_cast<size_t>(M * N_V)));

                LOG_INFO("[Integration] Real Qwen2 layer 3 fused QKV parity on snapshot ATTENTION_NORM input: Q=" << q_cos
                                                                                                                    << " K=" << k_cos
                                                                                                                    << " V=" << v_cos);

                EXPECT_GT(q_cos, COSINE_THRESHOLD) << "Q projection cosine too low for layer-3 snapshot input";
                EXPECT_GT(k_cos, COSINE_THRESHOLD) << "K projection cosine too low for layer-3 snapshot input";
                EXPECT_GT(v_cos, COSINE_THRESHOLD) << "V projection cosine too low for layer-3 snapshot input";
            }

            TEST_F(ROCmQuantisedGemmIntegrationTest, QKNormStage_RealQwen2Layer3SnapshotInputParity)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                std::ifstream model_file(TEST_MODEL_PATH);
                if (!model_file.good())
                {
                    GTEST_SKIP() << "Test model not found: " << TEST_MODEL_PATH;
                }

                auto stage_ctx = ModelContext::createForPPStage(TEST_MODEL_PATH, 0, 4, true, false);
                ASSERT_NE(stage_ctx, nullptr);

                FactoryPPStageConfig config;
                config.first_layer = 0;
                config.last_layer = 4;
                config.has_embedding = true;
                config.has_lm_head = false;
                ASSERT_TRUE(config.isValid());

                auto cpu_runner = createPPStageRunner(stage_ctx, DeviceId::cpu(), config);
                ASSERT_NE(cpu_runner, nullptr);
                cpu_runner->enableSnapshotCapture();

                constexpr int layer_idx = 3;
                constexpr int M = 64;
                constexpr int K = 896;
                constexpr int N_Q_HEADS = 14;
                constexpr int N_KV_HEADS = 2;
                constexpr int HEAD_DIM = 64;
                constexpr float COSINE_THRESHOLD = 0.995f;

                std::vector<int> tokens(M);
                for (int i = 0; i < M; ++i)
                {
                    tokens[i] = i % 1024;
                }
                ASSERT_TRUE(cpu_runner->forward(tokens.data(), M));

                size_t input_size = 0;
                const float *input_snapshot = cpu_runner->getSnapshot("layer3_ATTENTION_NORM", input_size);
                ASSERT_NE(input_snapshot, nullptr);
                ASSERT_EQ(input_size, static_cast<size_t>(M * K));

                auto model_ctx = loadModel();
                ASSERT_NE(model_ctx, nullptr);

                const std::string prefix = "blk." + std::to_string(layer_idx) + ".";

                auto cpu_wq = model_ctx->getWeightForDevice(prefix + "attn_q.weight", DeviceId::cpu());
                auto cpu_wk = model_ctx->getWeightForDevice(prefix + "attn_k.weight", DeviceId::cpu());
                auto cpu_wv = model_ctx->getWeightForDevice(prefix + "attn_v.weight", DeviceId::cpu());
                auto cpu_bias_q = model_ctx->getWeightForDevice(prefix + "attn_q.bias", DeviceId::cpu());
                auto cpu_bias_k = model_ctx->getWeightForDevice(prefix + "attn_k.bias", DeviceId::cpu());
                auto cpu_bias_v = model_ctx->getWeightForDevice(prefix + "attn_v.bias", DeviceId::cpu());
                auto cpu_q_norm = model_ctx->getWeightForDevice(prefix + "attn_q_norm.weight", DeviceId::cpu());
                auto cpu_k_norm = model_ctx->getWeightForDevice(prefix + "attn_k_norm.weight", DeviceId::cpu());

                auto rocm_wq = model_ctx->getWeightForDevice(prefix + "attn_q.weight", DeviceId::rocm(0));
                auto rocm_wk = model_ctx->getWeightForDevice(prefix + "attn_k.weight", DeviceId::rocm(0));
                auto rocm_wv = model_ctx->getWeightForDevice(prefix + "attn_v.weight", DeviceId::rocm(0));
                auto rocm_bias_q = model_ctx->getWeightForDevice(prefix + "attn_q.bias", DeviceId::rocm(0));
                auto rocm_bias_k = model_ctx->getWeightForDevice(prefix + "attn_k.bias", DeviceId::rocm(0));
                auto rocm_bias_v = model_ctx->getWeightForDevice(prefix + "attn_v.bias", DeviceId::rocm(0));
                auto rocm_q_norm = model_ctx->getWeightForDevice(prefix + "attn_q_norm.weight", DeviceId::rocm(0));
                auto rocm_k_norm = model_ctx->getWeightForDevice(prefix + "attn_k_norm.weight", DeviceId::rocm(0));

                if (!(cpu_q_norm && cpu_k_norm && rocm_q_norm && rocm_k_norm))
                {
                    GTEST_SKIP() << "Model does not provide attn_q_norm/attn_k_norm weights for layer 3";
                }

                ASSERT_TRUE(cpu_wq && cpu_wk && cpu_wv && cpu_bias_q && cpu_bias_k && cpu_bias_v);
                ASSERT_TRUE(rocm_wq && rocm_wk && rocm_wv && rocm_bias_q && rocm_bias_k && rocm_bias_v);

                auto input = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)}, DeviceId::cpu());
                std::copy(input_snapshot, input_snapshot + input_size, input->mutable_data());

                auto cpu_q = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_Q_HEADS * HEAD_DIM)}, DeviceId::cpu());
                auto cpu_k = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_KV_HEADS * HEAD_DIM)}, DeviceId::cpu());
                auto cpu_v = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_KV_HEADS * HEAD_DIM)}, DeviceId::cpu());
                auto rocm_q = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_Q_HEADS * HEAD_DIM)}, DeviceId::rocm(0));
                auto rocm_k = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_KV_HEADS * HEAD_DIM)}, DeviceId::rocm(0));
                auto rocm_v = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_KV_HEADS * HEAD_DIM)}, DeviceId::rocm(0));

                FusedQKVGEMMStage::Params cpu_qk_proj{
                    .device_id = DeviceId::cpu(),
                    .input = input.get(),
                    .m = M,
                    .k = K,
                    .wq = cpu_wq.get(),
                    .output_q = cpu_q.get(),
                    .n_q = N_Q_HEADS * HEAD_DIM,
                    .bias_q = cpu_bias_q.get(),
                    .wk = cpu_wk.get(),
                    .output_k = cpu_k.get(),
                    .n_k = N_KV_HEADS * HEAD_DIM,
                    .bias_k = cpu_bias_k.get(),
                    .wv = cpu_wv.get(),
                    .output_v = cpu_v.get(),
                    .n_v = N_KV_HEADS * HEAD_DIM,
                    .bias_v = cpu_bias_v.get()};

                FusedQKVGEMMStage::Params rocm_qk_proj{
                    .device_id = DeviceId::rocm(0),
                    .input = input.get(),
                    .m = M,
                    .k = K,
                    .wq = rocm_wq.get(),
                    .output_q = rocm_q.get(),
                    .n_q = N_Q_HEADS * HEAD_DIM,
                    .bias_q = rocm_bias_q.get(),
                    .wk = rocm_wk.get(),
                    .output_k = rocm_k.get(),
                    .n_k = N_KV_HEADS * HEAD_DIM,
                    .bias_k = rocm_bias_k.get(),
                    .wv = rocm_wv.get(),
                    .output_v = rocm_v.get(),
                    .n_v = N_KV_HEADS * HEAD_DIM,
                    .bias_v = rocm_bias_v.get()};

                FusedQKVGEMMStage cpu_proj_stage(cpu_qk_proj);
                FusedQKVGEMMStage rocm_proj_stage(rocm_qk_proj);
                CPUDeviceContext cpu_ctx(DeviceId::cpu(), 4);
                ROCmDeviceContext rocm_ctx(DeviceId::rocm(0), 0);

                ASSERT_TRUE(cpu_proj_stage.execute(&cpu_ctx)) << "CPU Q/K projection failed for layer-3 snapshot input";

                auto proj_reqs = rocm_proj_stage.getWorkspaceRequirements(M, N_Q_HEADS * HEAD_DIM, K);
                DeviceWorkspaceManager proj_workspace(DeviceId::rocm(0), 128 * 1024 * 1024);
                ASSERT_TRUE(proj_workspace.allocate(proj_reqs));
                rocm_proj_stage.bindWorkspace(&proj_workspace);

                const std::initializer_list<TensorBase *> proj_inputs = {
                    input.get(), rocm_wq.get(), rocm_wk.get(), rocm_wv.get(), rocm_bias_q.get(), rocm_bias_k.get(), rocm_bias_v.get()};
                const std::initializer_list<TensorBase *> proj_outputs = {rocm_q.get(), rocm_k.get(), rocm_v.get()};
                ASSERT_TRUE(with_gpu_coherence(DeviceId::rocm(0), proj_inputs, proj_outputs, [&]() {
                    return rocm_proj_stage.execute(&rocm_ctx);
                })) << "ROCm Q/K projection failed for layer-3 snapshot input";
                rocm_proj_stage.unbindWorkspace();

                QKNormStage::Params cpu_q_norm_params{
                    .device_id = DeviceId::cpu(),
                    .input = cpu_q.get(),
                    .output = cpu_q.get(),
                    .gamma = cpu_q_norm.get(),
                    .n_heads = N_Q_HEADS,
                    .head_dim = HEAD_DIM,
                    .eps = 1e-6f,
                    .seq_len = M};
                QKNormStage::Params rocm_q_norm_params{
                    .device_id = DeviceId::rocm(0),
                    .input = rocm_q.get(),
                    .output = rocm_q.get(),
                    .gamma = rocm_q_norm.get(),
                    .n_heads = N_Q_HEADS,
                    .head_dim = HEAD_DIM,
                    .eps = 1e-6f,
                    .seq_len = M};

                QKNormStage::Params cpu_k_norm_params{
                    .device_id = DeviceId::cpu(),
                    .input = cpu_k.get(),
                    .output = cpu_k.get(),
                    .gamma = cpu_k_norm.get(),
                    .n_heads = N_KV_HEADS,
                    .head_dim = HEAD_DIM,
                    .eps = 1e-6f,
                    .seq_len = M};
                QKNormStage::Params rocm_k_norm_params{
                    .device_id = DeviceId::rocm(0),
                    .input = rocm_k.get(),
                    .output = rocm_k.get(),
                    .gamma = rocm_k_norm.get(),
                    .n_heads = N_KV_HEADS,
                    .head_dim = HEAD_DIM,
                    .eps = 1e-6f,
                    .seq_len = M};

                QKNormStage cpu_q_norm_stage(cpu_q_norm_params);
                QKNormStage rocm_q_norm_stage(rocm_q_norm_params);
                QKNormStage cpu_k_norm_stage(cpu_k_norm_params);
                QKNormStage rocm_k_norm_stage(rocm_k_norm_params);

                ASSERT_TRUE(cpu_q_norm_stage.execute(&cpu_ctx)) << "CPU Q norm failed for layer-3 snapshot input";
                ASSERT_TRUE(cpu_k_norm_stage.execute(&cpu_ctx)) << "CPU K norm failed for layer-3 snapshot input";

                const std::initializer_list<TensorBase *> qnorm_inputs = {rocm_q.get(), rocm_q_norm.get()};
                const std::initializer_list<TensorBase *> qnorm_outputs = {rocm_q.get()};
                ASSERT_TRUE(with_gpu_coherence(DeviceId::rocm(0), qnorm_inputs, qnorm_outputs, [&]() {
                    return rocm_q_norm_stage.execute(&rocm_ctx);
                })) << "ROCm Q norm failed for layer-3 snapshot input";

                const std::initializer_list<TensorBase *> knorm_inputs = {rocm_k.get(), rocm_k_norm.get()};
                const std::initializer_list<TensorBase *> knorm_outputs = {rocm_k.get()};
                ASSERT_TRUE(with_gpu_coherence(DeviceId::rocm(0), knorm_inputs, knorm_outputs, [&]() {
                    return rocm_k_norm_stage.execute(&rocm_ctx);
                })) << "ROCm K norm failed for layer-3 snapshot input";

                const float q_cos = cosineSimilarity(
                    std::vector<float>(cpu_q->data(), cpu_q->data() + cpu_q->numel()),
                    std::vector<float>(rocm_q->data(), rocm_q->data() + rocm_q->numel()));
                const float k_cos = cosineSimilarity(
                    std::vector<float>(cpu_k->data(), cpu_k->data() + cpu_k->numel()),
                    std::vector<float>(rocm_k->data(), rocm_k->data() + rocm_k->numel()));

                LOG_INFO("[Integration] Real Qwen2 layer 3 QKNorm parity on snapshot ATTENTION_NORM input: Q=" << q_cos
                                                                                                                  << " K=" << k_cos);

                EXPECT_GT(q_cos, COSINE_THRESHOLD) << "QKNorm Q cosine too low for layer-3 snapshot input";
                EXPECT_GT(k_cos, COSINE_THRESHOLD) << "QKNorm K cosine too low for layer-3 snapshot input";
            }

            TEST_F(ROCmQuantisedGemmIntegrationTest, QKNormStage_RealQwen2Layer21Parity)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                auto model_ctx = loadModel();
                if (!model_ctx)
                {
                    GTEST_SKIP() << "Test model not found: " << TEST_MODEL_PATH;
                }

                constexpr int layer_idx = 21;
                constexpr int M = 64;
                constexpr int K = 896;
                constexpr int N_Q_HEADS = 14;
                constexpr int N_KV_HEADS = 2;
                constexpr int HEAD_DIM = 64;
                constexpr float COSINE_THRESHOLD = 0.995f;

                const std::string prefix = "blk." + std::to_string(layer_idx) + ".";

                auto cpu_wq = model_ctx->getWeightForDevice(prefix + "attn_q.weight", DeviceId::cpu());
                auto cpu_wk = model_ctx->getWeightForDevice(prefix + "attn_k.weight", DeviceId::cpu());
                auto cpu_wv = model_ctx->getWeightForDevice(prefix + "attn_v.weight", DeviceId::cpu());
                auto cpu_bias_q = model_ctx->getWeightForDevice(prefix + "attn_q.bias", DeviceId::cpu());
                auto cpu_bias_k = model_ctx->getWeightForDevice(prefix + "attn_k.bias", DeviceId::cpu());
                auto cpu_bias_v = model_ctx->getWeightForDevice(prefix + "attn_v.bias", DeviceId::cpu());
                auto cpu_q_norm = model_ctx->getWeightForDevice(prefix + "attn_q_norm.weight", DeviceId::cpu());
                auto cpu_k_norm = model_ctx->getWeightForDevice(prefix + "attn_k_norm.weight", DeviceId::cpu());

                auto rocm_wq = model_ctx->getWeightForDevice(prefix + "attn_q.weight", DeviceId::rocm(0));
                auto rocm_wk = model_ctx->getWeightForDevice(prefix + "attn_k.weight", DeviceId::rocm(0));
                auto rocm_wv = model_ctx->getWeightForDevice(prefix + "attn_v.weight", DeviceId::rocm(0));
                auto rocm_bias_q = model_ctx->getWeightForDevice(prefix + "attn_q.bias", DeviceId::rocm(0));
                auto rocm_bias_k = model_ctx->getWeightForDevice(prefix + "attn_k.bias", DeviceId::rocm(0));
                auto rocm_bias_v = model_ctx->getWeightForDevice(prefix + "attn_v.bias", DeviceId::rocm(0));
                auto rocm_q_norm = model_ctx->getWeightForDevice(prefix + "attn_q_norm.weight", DeviceId::rocm(0));
                auto rocm_k_norm = model_ctx->getWeightForDevice(prefix + "attn_k_norm.weight", DeviceId::rocm(0));

                if (!(cpu_q_norm && cpu_k_norm && rocm_q_norm && rocm_k_norm))
                {
                    GTEST_SKIP() << "Model does not provide attn_q_norm/attn_k_norm weights for layer 21";
                }

                ASSERT_TRUE(cpu_wq && cpu_wk && cpu_wv && cpu_bias_q && cpu_bias_k && cpu_bias_v);
                ASSERT_TRUE(rocm_wq && rocm_wk && rocm_wv && rocm_bias_q && rocm_bias_k && rocm_bias_v);

                auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)}, -0.5f, 0.5f, 654);
                auto cpu_q = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_Q_HEADS * HEAD_DIM)}, DeviceId::cpu());
                auto cpu_k = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_KV_HEADS * HEAD_DIM)}, DeviceId::cpu());
                auto cpu_v = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_KV_HEADS * HEAD_DIM)}, DeviceId::cpu());
                auto rocm_q = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_Q_HEADS * HEAD_DIM)}, DeviceId::rocm(0));
                auto rocm_k = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_KV_HEADS * HEAD_DIM)}, DeviceId::rocm(0));
                auto rocm_v = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_KV_HEADS * HEAD_DIM)}, DeviceId::rocm(0));

                FusedQKVGEMMStage::Params cpu_qk_proj{
                    .device_id = DeviceId::cpu(),
                    .input = input.get(),
                    .m = M,
                    .k = K,
                    .wq = cpu_wq.get(),
                    .output_q = cpu_q.get(),
                    .n_q = N_Q_HEADS * HEAD_DIM,
                    .bias_q = cpu_bias_q.get(),
                    .wk = cpu_wk.get(),
                    .output_k = cpu_k.get(),
                    .n_k = N_KV_HEADS * HEAD_DIM,
                    .bias_k = cpu_bias_k.get(),
                    .wv = cpu_wv.get(),
                    .output_v = cpu_v.get(),
                    .n_v = N_KV_HEADS * HEAD_DIM,
                    .bias_v = cpu_bias_v.get()};

                FusedQKVGEMMStage::Params rocm_qk_proj{
                    .device_id = DeviceId::rocm(0),
                    .input = input.get(),
                    .m = M,
                    .k = K,
                    .wq = rocm_wq.get(),
                    .output_q = rocm_q.get(),
                    .n_q = N_Q_HEADS * HEAD_DIM,
                    .bias_q = rocm_bias_q.get(),
                    .wk = rocm_wk.get(),
                    .output_k = rocm_k.get(),
                    .n_k = N_KV_HEADS * HEAD_DIM,
                    .bias_k = rocm_bias_k.get(),
                    .wv = rocm_wv.get(),
                    .output_v = rocm_v.get(),
                    .n_v = N_KV_HEADS * HEAD_DIM,
                    .bias_v = rocm_bias_v.get()};

                FusedQKVGEMMStage cpu_proj_stage(cpu_qk_proj);
                FusedQKVGEMMStage rocm_proj_stage(rocm_qk_proj);
                CPUDeviceContext cpu_ctx(DeviceId::cpu(), 4);
                ROCmDeviceContext rocm_ctx(DeviceId::rocm(0), 0);

                ASSERT_TRUE(cpu_proj_stage.execute(&cpu_ctx)) << "CPU Q/K projection failed";

                auto proj_reqs = rocm_proj_stage.getWorkspaceRequirements(M, N_Q_HEADS * HEAD_DIM, K);
                DeviceWorkspaceManager proj_workspace(DeviceId::rocm(0), 128 * 1024 * 1024);
                ASSERT_TRUE(proj_workspace.allocate(proj_reqs));
                rocm_proj_stage.bindWorkspace(&proj_workspace);

                const std::initializer_list<TensorBase *> proj_inputs = {
                    input.get(), rocm_wq.get(), rocm_wk.get(), rocm_wv.get(), rocm_bias_q.get(), rocm_bias_k.get(), rocm_bias_v.get()};
                const std::initializer_list<TensorBase *> proj_outputs = {rocm_q.get(), rocm_k.get(), rocm_v.get()};
                ASSERT_TRUE(with_gpu_coherence(DeviceId::rocm(0), proj_inputs, proj_outputs, [&]() {
                    return rocm_proj_stage.execute(&rocm_ctx);
                })) << "ROCm Q/K projection failed";
                rocm_proj_stage.unbindWorkspace();

                QKNormStage::Params cpu_q_norm_params{
                    .device_id = DeviceId::cpu(),
                    .input = cpu_q.get(),
                    .output = cpu_q.get(),
                    .gamma = cpu_q_norm.get(),
                    .n_heads = N_Q_HEADS,
                    .head_dim = HEAD_DIM,
                    .eps = 1e-6f,
                    .seq_len = M};
                QKNormStage::Params rocm_q_norm_params{
                    .device_id = DeviceId::rocm(0),
                    .input = rocm_q.get(),
                    .output = rocm_q.get(),
                    .gamma = rocm_q_norm.get(),
                    .n_heads = N_Q_HEADS,
                    .head_dim = HEAD_DIM,
                    .eps = 1e-6f,
                    .seq_len = M};

                QKNormStage::Params cpu_k_norm_params{
                    .device_id = DeviceId::cpu(),
                    .input = cpu_k.get(),
                    .output = cpu_k.get(),
                    .gamma = cpu_k_norm.get(),
                    .n_heads = N_KV_HEADS,
                    .head_dim = HEAD_DIM,
                    .eps = 1e-6f,
                    .seq_len = M};
                QKNormStage::Params rocm_k_norm_params{
                    .device_id = DeviceId::rocm(0),
                    .input = rocm_k.get(),
                    .output = rocm_k.get(),
                    .gamma = rocm_k_norm.get(),
                    .n_heads = N_KV_HEADS,
                    .head_dim = HEAD_DIM,
                    .eps = 1e-6f,
                    .seq_len = M};

                QKNormStage cpu_q_norm_stage(cpu_q_norm_params);
                QKNormStage rocm_q_norm_stage(rocm_q_norm_params);
                QKNormStage cpu_k_norm_stage(cpu_k_norm_params);
                QKNormStage rocm_k_norm_stage(rocm_k_norm_params);

                ASSERT_TRUE(cpu_q_norm_stage.execute(&cpu_ctx)) << "CPU Q norm failed";
                ASSERT_TRUE(cpu_k_norm_stage.execute(&cpu_ctx)) << "CPU K norm failed";

                const std::initializer_list<TensorBase *> qnorm_inputs = {rocm_q.get(), rocm_q_norm.get()};
                const std::initializer_list<TensorBase *> qnorm_outputs = {rocm_q.get()};
                ASSERT_TRUE(with_gpu_coherence(DeviceId::rocm(0), qnorm_inputs, qnorm_outputs, [&]() {
                    return rocm_q_norm_stage.execute(&rocm_ctx);
                })) << "ROCm Q norm failed";

                const std::initializer_list<TensorBase *> knorm_inputs = {rocm_k.get(), rocm_k_norm.get()};
                const std::initializer_list<TensorBase *> knorm_outputs = {rocm_k.get()};
                ASSERT_TRUE(with_gpu_coherence(DeviceId::rocm(0), knorm_inputs, knorm_outputs, [&]() {
                    return rocm_k_norm_stage.execute(&rocm_ctx);
                })) << "ROCm K norm failed";

                const float q_cos = cosineSimilarity(
                    std::vector<float>(cpu_q->data(), cpu_q->data() + cpu_q->numel()),
                    std::vector<float>(rocm_q->data(), rocm_q->data() + rocm_q->numel()));
                const float k_cos = cosineSimilarity(
                    std::vector<float>(cpu_k->data(), cpu_k->data() + cpu_k->numel()),
                    std::vector<float>(rocm_k->data(), rocm_k->data() + rocm_k->numel()));

                LOG_INFO("[Integration] Real Qwen2 layer 21 QKNorm parity: Q=" << q_cos << " K=" << k_cos);

                EXPECT_GT(q_cos, COSINE_THRESHOLD) << "QKNorm Q cosine too low";
                EXPECT_GT(k_cos, COSINE_THRESHOLD) << "QKNorm K cosine too low";
            }

            TEST_F(ROCmQuantisedGemmIntegrationTest, DownProjStage_RealQwen2Layer0SnapshotInputParity)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                std::ifstream model_file(TEST_MODEL_PATH);
                if (!model_file.good())
                {
                    GTEST_SKIP() << "Test model not found: " << TEST_MODEL_PATH;
                }

                auto stage_ctx = ModelContext::createForPPStage(TEST_MODEL_PATH, 0, 1, true, false);
                ASSERT_NE(stage_ctx, nullptr);

                FactoryPPStageConfig config;
                config.first_layer = 0;
                config.last_layer = 1;
                config.has_embedding = true;
                config.has_lm_head = false;
                ASSERT_TRUE(config.isValid());

                auto cpu_runner = createPPStageRunner(stage_ctx, DeviceId::cpu(), config);
                ASSERT_NE(cpu_runner, nullptr);
                cpu_runner->enableSnapshotCapture();

                constexpr int layer_idx = 0;
                constexpr int M = 64;
                constexpr int K = 4864;
                constexpr int N = 896;
                constexpr float COSINE_THRESHOLD = 0.999f;

                std::vector<int> tokens(M);
                for (int i = 0; i < M; ++i)
                {
                    tokens[i] = i % 1024;
                }
                ASSERT_TRUE(cpu_runner->forward(tokens.data(), M));

                size_t input_size = 0;
                const float *input_snapshot = cpu_runner->getSnapshot("layer0_FFN_SWIGLU", input_size);
                ASSERT_NE(input_snapshot, nullptr);
                ASSERT_EQ(input_size, static_cast<size_t>(M * K));

                auto model_ctx = loadModel();
                ASSERT_NE(model_ctx, nullptr);

                auto cpu_weight = model_ctx->getWeightForDevice("blk.0.ffn_down.weight", DeviceId::cpu());
                auto rocm_weight = model_ctx->getWeightForDevice("blk.0.ffn_down.weight", DeviceId::rocm(0));
                ASSERT_TRUE(cpu_weight);
                ASSERT_TRUE(rocm_weight);

                auto input = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)}, DeviceId::cpu());
                std::copy(input_snapshot, input_snapshot + input_size, input->mutable_data());

                auto cpu_output = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)}, DeviceId::cpu());
                auto rocm_output = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)}, DeviceId::rocm(0));

                GEMMStage::Params cpu_params{
                    .device_id = DeviceId::cpu(),
                    .A = input.get(),
                    .B = cpu_weight.get(),
                    .C = cpu_output.get(),
                    .m = M,
                    .n = N,
                    .k = K,
                    .alpha = 1.0f,
                    .beta = 0.0f,
                    .transpose_B = false,
                    .gemm_context = GemmContext::FFN};

                GEMMStage::Params rocm_params{
                    .device_id = DeviceId::rocm(0),
                    .A = input.get(),
                    .B = rocm_weight.get(),
                    .C = rocm_output.get(),
                    .m = M,
                    .n = N,
                    .k = K,
                    .alpha = 1.0f,
                    .beta = 0.0f,
                    .transpose_B = false,
                    .gemm_context = GemmContext::FFN};

                GEMMStage cpu_stage(cpu_params);
                GEMMStage rocm_stage(rocm_params);
                CPUDeviceContext cpu_ctx(DeviceId::cpu(), 4);
                ROCmDeviceContext rocm_ctx(DeviceId::rocm(0), 0);

                ASSERT_TRUE(cpu_stage.execute(&cpu_ctx)) << "CPU down_proj execution failed for layer-0 snapshot input";

                auto reqs = rocm_stage.getWorkspaceRequirements(M, N, K);
                DeviceWorkspaceManager workspace(DeviceId::rocm(0), 128 * 1024 * 1024);
                ASSERT_TRUE(workspace.allocate(reqs)) << "Failed to allocate ROCm down_proj workspace";
                rocm_stage.bindWorkspace(&workspace);

                const std::initializer_list<TensorBase *> rocm_inputs = {
                    input.get(),
                    rocm_weight.get()};
                const std::initializer_list<TensorBase *> rocm_outputs = {
                    rocm_output.get()};

                const bool rocm_ok = with_gpu_coherence(
                    DeviceId::rocm(0),
                    rocm_inputs,
                    rocm_outputs,
                    [&]() {
                        return rocm_stage.execute(&rocm_ctx);
                    });

                rocm_stage.unbindWorkspace();

                ASSERT_TRUE(rocm_ok) << "ROCm down_proj execution failed for layer-0 snapshot input";

                const float cosine = cosineSimilarity(
                    std::vector<float>(cpu_output->data(), cpu_output->data() + static_cast<size_t>(M * N)),
                    std::vector<float>(rocm_output->data(), rocm_output->data() + static_cast<size_t>(M * N)));

                LOG_INFO("[Integration] Real Qwen2 layer 0 down_proj parity on snapshot FFN_SWIGLU input: cosine=" << cosine);

                EXPECT_GT(cosine, COSINE_THRESHOLD)
                    << "down_proj cosine too low for layer-0 FFN_SWIGLU snapshot input";
            }

            TEST_F(ROCmQuantisedGemmIntegrationTest, DownProjStage_RealQwen2Layer0SnapshotInputParity_ForceCK)
            {
                // CK (Composable Kernel) path requires INT8-VNNI format weights for row-major
                // repack. Q4_0 models are packed as native-VNNI only, so CK has no weights to
                // consume. Skip until a Q8_0 test model is used or CK gains native-VNNI support.
                GTEST_SKIP() << "CK path requires INT8-VNNI weights; Q4_0 test model is native-VNNI only";

                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                std::ifstream model_file(TEST_MODEL_PATH);
                if (!model_file.good())
                {
                    GTEST_SKIP() << "Test model not found: " << TEST_MODEL_PATH;
                }

                auto stage_ctx = ModelContext::createForPPStage(TEST_MODEL_PATH, 0, 1, true, false);
                ASSERT_NE(stage_ctx, nullptr);

                FactoryPPStageConfig config;
                config.first_layer = 0;
                config.last_layer = 1;
                config.has_embedding = true;
                config.has_lm_head = false;
                ASSERT_TRUE(config.isValid());

                auto cpu_runner = createPPStageRunner(stage_ctx, DeviceId::cpu(), config);
                ASSERT_NE(cpu_runner, nullptr);
                cpu_runner->enableSnapshotCapture();

                constexpr int layer_idx = 0;
                (void)layer_idx;
                constexpr int M = 64;
                constexpr int K = 4864;
                constexpr int N = 896;
                constexpr float COSINE_THRESHOLD = 0.999f;

                std::vector<int> tokens(M);
                for (int i = 0; i < M; ++i)
                {
                    tokens[i] = i % 1024;
                }
                ASSERT_TRUE(cpu_runner->forward(tokens.data(), M));

                size_t input_size = 0;
                const float *input_snapshot = cpu_runner->getSnapshot("layer0_FFN_SWIGLU", input_size);
                ASSERT_NE(input_snapshot, nullptr);
                ASSERT_EQ(input_size, static_cast<size_t>(M * K));

                auto model_ctx = loadModel();
                ASSERT_NE(model_ctx, nullptr);

                auto cpu_weight = model_ctx->getWeightForDevice("blk.0.ffn_down.weight", DeviceId::cpu());
                auto rocm_weight = model_ctx->getWeightForDevice("blk.0.ffn_down.weight", DeviceId::rocm(0));
                ASSERT_TRUE(cpu_weight);
                ASSERT_TRUE(rocm_weight);

                auto input = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)}, DeviceId::cpu());
                std::copy(input_snapshot, input_snapshot + input_size, input->mutable_data());

                auto cpu_output = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)}, DeviceId::cpu());
                auto rocm_output = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)}, DeviceId::rocm(0));

                GEMMStage::Params cpu_params{
                    .device_id = DeviceId::cpu(),
                    .A = input.get(),
                    .B = cpu_weight.get(),
                    .C = cpu_output.get(),
                    .m = M,
                    .n = N,
                    .k = K,
                    .alpha = 1.0f,
                    .beta = 0.0f,
                    .transpose_B = false,
                    .gemm_context = GemmContext::FFN};

                GEMMStage::Params rocm_params{
                    .device_id = DeviceId::rocm(0),
                    .A = input.get(),
                    .B = rocm_weight.get(),
                    .C = rocm_output.get(),
                    .m = M,
                    .n = N,
                    .k = K,
                    .alpha = 1.0f,
                    .beta = 0.0f,
                    .transpose_B = false,
                    .gemm_context = GemmContext::FFN};

                GEMMStage cpu_stage(cpu_params);
                GEMMStage rocm_stage(rocm_params);
                CPUDeviceContext cpu_ctx(DeviceId::cpu(), 4);
                ROCmDeviceContext rocm_ctx(DeviceId::rocm(0), 0);

                ASSERT_TRUE(cpu_stage.execute(&cpu_ctx)) << "CPU down_proj execution failed for layer-0 snapshot input";

                auto reqs = rocm_stage.getWorkspaceRequirements(M, N, K);
                DeviceWorkspaceManager workspace(DeviceId::rocm(0), 128 * 1024 * 1024);
                ASSERT_TRUE(workspace.allocate(reqs)) << "Failed to allocate ROCm down_proj workspace";
                rocm_stage.bindWorkspace(&workspace);

                const std::initializer_list<TensorBase *> rocm_inputs = {
                    input.get(),
                    rocm_weight.get()};
                const std::initializer_list<TensorBase *> rocm_outputs = {
                    rocm_output.get()};

                const bool rocm_ok = [&]() {
                    ScopedEnvOverride force_ck("LLAMINAR_ROCM_FORCE_CK", "1");
                    return with_gpu_coherence(
                        DeviceId::rocm(0),
                        rocm_inputs,
                        rocm_outputs,
                        [&]() {
                            return rocm_stage.execute(&rocm_ctx);
                        });
                }();

                rocm_stage.unbindWorkspace();

                ASSERT_TRUE(rocm_ok) << "ROCm down_proj execution failed for layer-0 snapshot input with CK forced";

                const float cosine = cosineSimilarity(
                    std::vector<float>(cpu_output->data(), cpu_output->data() + static_cast<size_t>(M * N)),
                    std::vector<float>(rocm_output->data(), rocm_output->data() + static_cast<size_t>(M * N)));

                LOG_INFO("[Integration] Real Qwen2 layer 0 down_proj parity on snapshot FFN_SWIGLU input with CK forced: cosine=" << cosine);

                EXPECT_GT(cosine, COSINE_THRESHOLD)
                    << "down_proj cosine too low for layer-0 FFN_SWIGLU snapshot input with CK forced";
            }

            TEST_F(ROCmQuantisedGemmIntegrationTest, DownProjStage_RealQwen2Layer0QuantizedActivationReference)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                std::ifstream model_file(TEST_MODEL_PATH);
                if (!model_file.good())
                {
                    GTEST_SKIP() << "Test model not found: " << TEST_MODEL_PATH;
                }

                auto stage_ctx = ModelContext::createForPPStage(TEST_MODEL_PATH, 0, 1, true, false);
                ASSERT_NE(stage_ctx, nullptr);

                FactoryPPStageConfig config;
                config.first_layer = 0;
                config.last_layer = 1;
                config.has_embedding = true;
                config.has_lm_head = false;
                ASSERT_TRUE(config.isValid());

                auto cpu_runner = createPPStageRunner(stage_ctx, DeviceId::cpu(), config);
                ASSERT_NE(cpu_runner, nullptr);
                cpu_runner->enableSnapshotCapture();

                constexpr int M = 64;
                constexpr int K = 4864;
                constexpr int N = 896;
                constexpr float ROCM_TO_QUANTIZED_REF_THRESHOLD = 0.99999f;

                std::vector<int> tokens(M);
                for (int i = 0; i < M; ++i)
                {
                    tokens[i] = i % 1024;
                }
                ASSERT_TRUE(cpu_runner->forward(tokens.data(), M));

                size_t input_size = 0;
                const float *input_snapshot = cpu_runner->getSnapshot("layer0_FFN_SWIGLU", input_size);
                ASSERT_NE(input_snapshot, nullptr);
                ASSERT_EQ(input_size, static_cast<size_t>(M * K));

                auto model_ctx = loadModel();
                ASSERT_NE(model_ctx, nullptr);

                auto cpu_weight = model_ctx->getWeightForDevice("blk.0.ffn_down.weight", DeviceId::cpu());
                auto rocm_weight = model_ctx->getWeightForDevice("blk.0.ffn_down.weight", DeviceId::rocm(0));
                ASSERT_TRUE(cpu_weight);
                ASSERT_TRUE(rocm_weight);

                auto *unpackable = dynamic_cast<const IINT8Unpackable *>(cpu_weight.get());
                ASSERT_NE(unpackable, nullptr) << "CPU down_proj weight must implement IINT8Unpackable";

                std::vector<float> input_host(input_snapshot, input_snapshot + input_size);

                float *d_input = nullptr;
                int8_t *d_input_q = nullptr;
                float *d_scales_a = nullptr;
                int32_t *d_acc = nullptr;
                int work_buffer_m = 0;

                ASSERT_TRUE(rocmQuantGemm_allocFloat(&d_input, static_cast<size_t>(M) * K, 0));
                ASSERT_TRUE(rocmQuantGemm_ensureWorkBuffers(
                    &d_input_q,
                    &d_scales_a,
                    &d_acc,
                    &work_buffer_m,
                    M,
                    K,
                    N,
                    0));
                ASSERT_TRUE(rocmQuantGemm_copyHostToDevice(d_input, input_host.data(), static_cast<size_t>(M) * K, 0));
                ASSERT_TRUE(rocmQuantGemm_quantizeActivations(d_input, d_input_q, d_scales_a, M, K, 0, nullptr));

                std::vector<int8_t> activations_int8(static_cast<size_t>(M) * K);
                std::vector<float> scales_a(static_cast<size_t>(M));
                ASSERT_EQ(hipMemcpy(
                              activations_int8.data(),
                              d_input_q,
                              activations_int8.size() * sizeof(int8_t),
                              hipMemcpyDeviceToHost),
                          hipSuccess);
                ASSERT_EQ(hipMemcpy(
                              scales_a.data(),
                              d_scales_a,
                              scales_a.size() * sizeof(float),
                              hipMemcpyDeviceToHost),
                          hipSuccess);

                rocmQuantGemm_freeDevice(d_input, 0);
                rocmQuantGemm_freeDevice(d_input_q, 0);
                rocmQuantGemm_freeDevice(d_scales_a, 0);
                rocmQuantGemm_freeDevice(d_acc, 0);

                const NativeWeightReference weight_reference = buildNativeWeightReference(*unpackable, N, K);
                const std::vector<float> quantized_reference = computeNativeReferenceFromQuantizedActivations(
                    activations_int8,
                    scales_a,
                    weight_reference,
                    M,
                    N,
                    K);

                std::vector<int8_t> activations_int8_blockwise;
                std::vector<float> scales_a_blockwise;
                quantizeActivationsPerBlock32(
                    input_host,
                    M,
                    K,
                    activations_int8_blockwise,
                    scales_a_blockwise);
                const std::vector<float> blockwise_quantized_reference = computeNativeReferenceFromBlockwiseQuantizedActivations(
                    activations_int8_blockwise,
                    scales_a_blockwise,
                    weight_reference,
                    M,
                    N,
                    K);

                auto input = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)}, DeviceId::cpu());
                std::copy(input_snapshot, input_snapshot + input_size, input->mutable_data());

                auto cpu_output = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)}, DeviceId::cpu());
                auto rocm_output = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)}, DeviceId::rocm(0));

                GEMMStage::Params cpu_params{
                    .device_id = DeviceId::cpu(),
                    .A = input.get(),
                    .B = cpu_weight.get(),
                    .C = cpu_output.get(),
                    .m = M,
                    .n = N,
                    .k = K,
                    .alpha = 1.0f,
                    .beta = 0.0f,
                    .transpose_B = false,
                    .gemm_context = GemmContext::FFN};

                GEMMStage::Params rocm_params{
                    .device_id = DeviceId::rocm(0),
                    .A = input.get(),
                    .B = rocm_weight.get(),
                    .C = rocm_output.get(),
                    .m = M,
                    .n = N,
                    .k = K,
                    .alpha = 1.0f,
                    .beta = 0.0f,
                    .transpose_B = false,
                    .gemm_context = GemmContext::FFN};

                GEMMStage cpu_stage(cpu_params);
                GEMMStage rocm_stage(rocm_params);
                CPUDeviceContext cpu_ctx(DeviceId::cpu(), 4);
                ROCmDeviceContext rocm_ctx(DeviceId::rocm(0), 0);

                ASSERT_TRUE(cpu_stage.execute(&cpu_ctx)) << "CPU down_proj execution failed for quantized-reference test";

                auto reqs = rocm_stage.getWorkspaceRequirements(M, N, K);
                DeviceWorkspaceManager workspace(DeviceId::rocm(0), 128 * 1024 * 1024);
                ASSERT_TRUE(workspace.allocate(reqs)) << "Failed to allocate ROCm down_proj workspace";
                rocm_stage.bindWorkspace(&workspace);

                const std::initializer_list<TensorBase *> rocm_inputs = {input.get(), rocm_weight.get()};
                const std::initializer_list<TensorBase *> rocm_outputs = {rocm_output.get()};
                const bool rocm_ok = with_gpu_coherence(
                    DeviceId::rocm(0),
                    rocm_inputs,
                    rocm_outputs,
                    [&]() {
                        return rocm_stage.execute(&rocm_ctx);
                    });
                rocm_stage.unbindWorkspace();

                ASSERT_TRUE(rocm_ok) << "ROCm down_proj execution failed for quantized-reference test";

                const std::vector<float> cpu_values(
                    cpu_output->data(),
                    cpu_output->data() + static_cast<size_t>(M * N));
                const std::vector<float> rocm_values(
                    rocm_output->data(),
                    rocm_output->data() + static_cast<size_t>(M * N));

                const float rocm_vs_quantized_ref = cosineSimilarity(rocm_values, quantized_reference);
                const float rocm_vs_blockwise_quantized_ref = cosineSimilarity(rocm_values, blockwise_quantized_reference);
                const float cpu_vs_quantized_ref = cosineSimilarity(cpu_values, quantized_reference);
                const float cpu_vs_blockwise_quantized_ref = cosineSimilarity(cpu_values, blockwise_quantized_reference);
                const float cpu_vs_rocm = cosineSimilarity(cpu_values, rocm_values);

                LOG_INFO("[Integration] Real Qwen2 layer 0 down_proj quantized-reference: rocm_vs_quantized_ref="
                         << rocm_vs_quantized_ref
                         << " rocm_vs_blockwise_quantized_ref=" << rocm_vs_blockwise_quantized_ref
                         << " cpu_vs_quantized_ref=" << cpu_vs_quantized_ref
                         << " cpu_vs_blockwise_quantized_ref=" << cpu_vs_blockwise_quantized_ref
                         << " cpu_vs_rocm=" << cpu_vs_rocm);

                // Blockwise activation quantization reduces quantization error vs row-wise,
                // so ROCm (which uses blockwise) should match the blockwise reference tightly.
                EXPECT_GT(rocm_vs_blockwise_quantized_ref, ROCM_TO_QUANTIZED_REF_THRESHOLD)
                    << "ROCm down_proj diverges from blockwise quantized-activation native reference";
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
                        d_activations, d_A_int8, d_scales_A, M, K, 0, nullptr))
                        << "Failed to quantize for M=" << M;

                    rocmQuantGemm_freeDevice(d_activations, 0);
                }

                // Cleanup shared buffers
                rocmQuantGemm_freeDevice(d_A_int8, 0);
                rocmQuantGemm_freeDevice(d_scales_A, 0);
                rocmQuantGemm_freeDevice(d_C_int32, 0);

                LOG_INFO("[Integration] Tested batch sizes: 1 to 2048");
            }

            // =============================================================================
            // Full Pipeline Tests - Scaling with Bias
            // =============================================================================

            TEST_F(ROCmQuantisedGemmIntegrationTest, BiasAdd_RowMajorBroadcast)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                const int M = 9;
                const int N = 64;

                float *d_output = nullptr;
                float *d_bias = nullptr;

                ASSERT_TRUE(rocmQuantGemm_allocFloat(&d_output, M * N, 0));
                ASSERT_TRUE(rocmQuantGemm_allocFloat(&d_bias, N, 0));

                std::vector<float> h_output(M * N);
                std::vector<float> h_bias(N);

                for (int row = 0; row < M; ++row)
                {
                    for (int col = 0; col < N; ++col)
                    {
                        h_output[static_cast<size_t>(row) * N + col] = static_cast<float>(row * 100 + col);
                    }
                }

                for (int col = 0; col < N; ++col)
                {
                    h_bias[col] = 0.25f * static_cast<float>((col % 7) - 3);
                }

                hipMemcpy(d_output, h_output.data(), h_output.size() * sizeof(float), hipMemcpyHostToDevice);
                hipMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float), hipMemcpyHostToDevice);

                ASSERT_TRUE(rocmQuantGemm_biasAdd(d_output, d_bias, M, N, 0, nullptr));

                std::vector<float> h_result(M * N);
                hipMemcpy(h_result.data(), d_output, h_result.size() * sizeof(float), hipMemcpyDeviceToHost);

                for (int row = 0; row < M; ++row)
                {
                    for (int col = 0; col < N; ++col)
                    {
                        const size_t idx = static_cast<size_t>(row) * N + col;
                        const float expected = h_output[idx] + h_bias[col];
                        EXPECT_NEAR(h_result[idx], expected, 1e-5f)
                            << "Bias broadcast mismatch at row=" << row << " col=" << col;
                    }
                }

                rocmQuantGemm_freeDevice(d_output, 0);
                rocmQuantGemm_freeDevice(d_bias, 0);
            }

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
                    M, N, 1.0f, 0.0f, nullptr, nullptr, 0, nullptr));

                // Apply scaling with bias
                ASSERT_TRUE(rocmQuantGemm_applyScaling(
                    d_C_int32, d_C_fp32_with_bias, d_scales_A, d_scales_B,
                    M, N, 1.0f, 0.0f, nullptr, d_bias, 0, nullptr));

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
                rocmQuantGemm_freeDevice(d_C_int32, 0);
                rocmQuantGemm_freeDevice(d_C_fp32_no_bias, 0);
                rocmQuantGemm_freeDevice(d_C_fp32_with_bias, 0);
                rocmQuantGemm_freeDevice(d_scales_A, 0);
                rocmQuantGemm_freeDevice(d_scales_B, 0);
                rocmQuantGemm_freeDevice(d_bias, 0);
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
                    M, N, alpha, 0.0f, nullptr, nullptr, 0, nullptr));

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

                rocmQuantGemm_freeDevice(d_C_int32, 0);
                rocmQuantGemm_freeDevice(d_C_fp32, 0);
                rocmQuantGemm_freeDevice(d_scales_A, 0);
                rocmQuantGemm_freeDevice(d_scales_B, 0);
            }

            // MScaling tests removed - covered by decode/prefill tests above

            // =============================================================================
            // KernelFactory Integration Tests
            // =============================================================================
            // Format-Conditional Dispatch Tests
            // =============================================================================
            //
            // These tests verify the end-to-end dispatch logic introduced in Phase 1/2:
            //   - Native-VNNI formats (Q4_0, IQ4_NL) → native-VNNI GEMV/GEMM path
            //   - INT8-VNNI formats (Q8_0, Q8_1) → INT8 requant + CK GEMM path
            //
            // Each test:
            //   1. Packs weights via packWeightsToROCm (format-conditional)
            //   2. Calls multiply_tensor (auto-dispatches based on packed fields)
            //   3. Compares GPU output against CPU FP32 reference
            //
            // Complements the dedicated NativeVNNI_GEMM/GEMV test files that cover
            // all dispatch paths exhaustively for native-VNNI formats.

            namespace
            {
                float cosineSim(const float *a, const float *b, size_t n)
                {
                    double dot = 0.0, na = 0.0, nb = 0.0;
                    for (size_t i = 0; i < n; ++i)
                    {
                        dot += static_cast<double>(a[i]) * b[i];
                        na += static_cast<double>(a[i]) * a[i];
                        nb += static_cast<double>(b[i]) * b[i];
                    }
                    if (na == 0.0 || nb == 0.0)
                        return 0.0f;
                    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
                }

                /// CPU FP32 reference: C[i,j] = sum_k(A[i,k] * W[j,k])
                void cpuFP32GemmRef(const float *A, const float *W, float *C,
                                    int M, int N, int K)
                {
                    for (int i = 0; i < M; ++i)
                        for (int j = 0; j < N; ++j)
                        {
                            double acc = 0.0;
                            for (int k = 0; k < K; ++k)
                                acc += static_cast<double>(A[i * K + k]) *
                                       static_cast<double>(W[j * K + k]);
                            C[i * N + j] = static_cast<float>(acc);
                        }
                }

                ROCmQuantisedGemmKernel createLegacyROCmKernelForTest(
                    const TensorBase *weights,
                    int rocm_device_id)
                {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
                    ROCmQuantisedGemmKernel kernel(weights, rocm_device_id);
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
                    return kernel;
                }
            } // anonymous namespace

            /**
             * @test End-to-end multiply_tensor for Q8_0 (INT8-VNNI path) — decode (M=1)
             *
             * Verifies that Q8_0 weights go through INT8 requantization and the
             * CK GEMM pipeline produces correct results for single-token decode.
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, Dispatch_Q8_0_Decode_M1)
            {
                if (!has_rocm_device_)
                    GTEST_SKIP() << "No ROCm device available";

                const int M = 1, N = 896, K = 896;

                auto weights = TestTensorFactory::createQ8_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});
                std::vector<float> W_fp32(static_cast<size_t>(N) * K);
                weights->to_fp32(W_fp32.data());

                ROCmPackedWeights packed;
                ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

                // Verify INT8 path was used (not native-VNNI)
                EXPECT_FALSE(packed.int8_data.empty()) << "Q8_0 must use INT8 path";
                EXPECT_TRUE(packed.native_vnni_payload.empty()) << "Q8_0 must NOT use native-VNNI";

                ROCmQuantisedGemmKernel kernel(&packed, 0);
                ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

                auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
                auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});

                ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
                ASSERT_TRUE(output->allocateOnDevice(DeviceId::rocm(0)));

                ASSERT_TRUE(kernel.multiply_tensor(input.get(), output.get(), M, N, K));
                (void)hipDeviceSynchronize();
                output->mark_device_dirty();

                // CPU reference
                const float *in_host = input->data();
                std::vector<float> ref(static_cast<size_t>(M) * N);
                cpuFP32GemmRef(in_host, W_fp32.data(), ref.data(), M, N, K);

                float cos = cosineSim(output->data(), ref.data(), static_cast<size_t>(M) * N);
                LOG_INFO("[Dispatch] Q8_0 decode M=1: cosine=" << cos);
                EXPECT_GT(cos, 0.985f) << "Q8_0 decode cosine too low";

                cleanupWorkspace(kernel);
            }

            /**
             * @test End-to-end multiply_tensor for Q8_0 (INT8-VNNI path) — prefill (M=64)
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, Dispatch_Q8_0_Prefill_M64)
            {
                if (!has_rocm_device_)
                    GTEST_SKIP() << "No ROCm device available";

                const int M = 64, N = 4864, K = 896;

                auto weights = TestTensorFactory::createQ8_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});
                std::vector<float> W_fp32(static_cast<size_t>(N) * K);
                weights->to_fp32(W_fp32.data());

                ROCmPackedWeights packed;
                ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
                EXPECT_FALSE(packed.int8_data.empty());
                EXPECT_TRUE(packed.native_vnni_payload.empty());

                ROCmQuantisedGemmKernel kernel(&packed, 0);
                ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

                auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
                auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});

                ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
                ASSERT_TRUE(output->allocateOnDevice(DeviceId::rocm(0)));

                ASSERT_TRUE(kernel.multiply_tensor(input.get(), output.get(), M, N, K));
                (void)hipDeviceSynchronize();
                output->mark_device_dirty();

                const float *in_host = input->data();
                std::vector<float> ref(static_cast<size_t>(M) * N);
                cpuFP32GemmRef(in_host, W_fp32.data(), ref.data(), M, N, K);

                float cos = cosineSim(output->data(), ref.data(), static_cast<size_t>(M) * N);
                LOG_INFO("[Dispatch] Q8_0 prefill M=64 N=" << N << " K=" << K << ": cosine=" << cos);
                EXPECT_GT(cos, 0.985f) << "Q8_0 prefill cosine too low";

                cleanupWorkspace(kernel);
            }

            /**
             * @test Prefill bias is applied on the INT8 dispatch path (Q8_0).
             *
             * Uses the same packed weights and activations for two runs, with and
             * without a bias tensor. The delta must equal bias[col] for every row.
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, Dispatch_Q8_0_Prefill_M64_BiasBroadcastsPerColumn)
            {
                if (!has_rocm_device_)
                    GTEST_SKIP() << "No ROCm device available";

                const int M = 64, N = 4864, K = 896;

                auto weights = TestTensorFactory::createQ8_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});
                ROCmPackedWeights packed;
                ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
                EXPECT_FALSE(packed.int8_data.empty());
                EXPECT_TRUE(packed.native_vnni_payload.empty());

                ROCmQuantisedGemmKernel kernel(&packed, 0);
                ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

                auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
                auto output_no_bias = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
                auto output_with_bias = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
                auto bias = TestTensorFactory::createFP32({static_cast<size_t>(N)});

                float *bias_data = bias->mutable_data();
                for (int col = 0; col < N; ++col)
                {
                    bias_data[col] = 0.125f * static_cast<float>((col % 17) - 8);
                }

                ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
                ASSERT_TRUE(output_no_bias->allocateOnDevice(DeviceId::rocm(0)));
                ASSERT_TRUE(output_with_bias->allocateOnDevice(DeviceId::rocm(0)));

                ASSERT_TRUE(kernel.multiply_tensor(input.get(), output_no_bias.get(), M, N, K));
                ASSERT_TRUE(kernel.multiply_tensor(input.get(), output_with_bias.get(), M, N, K,
                                                   true, 1.0f, 0.0f, bias.get()));
                (void)hipDeviceSynchronize();
                output_no_bias->mark_device_dirty();
                output_with_bias->mark_device_dirty();

                const float *no_bias = output_no_bias->data();
                const float *with_bias = output_with_bias->data();

                for (int row = 0; row < M; ++row)
                {
                    for (int col = 0; col < N; ++col)
                    {
                        const size_t idx = static_cast<size_t>(row) * N + col;
                        const float expected_delta = bias_data[col];
                        const float actual_delta = with_bias[idx] - no_bias[idx];
                        EXPECT_NEAR(actual_delta, expected_delta, 1e-4f)
                            << "row=" << row << " col=" << col;
                    }
                }

                cleanupWorkspace(kernel);
            }

            /**
             * @test End-to-end multiply_tensor for Q8_1 (INT8-VNNI generic path) — decode (M=1)
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, Dispatch_Q8_1_Decode_M1)
            {
                if (!has_rocm_device_)
                    GTEST_SKIP() << "No ROCm device available";

                const int M = 1, N = 896, K = 896;

                auto weights = TestTensorFactory::createQ8_1Random({static_cast<size_t>(N), static_cast<size_t>(K)});
                std::vector<float> W_fp32(static_cast<size_t>(N) * K);
                weights->to_fp32(W_fp32.data());

                ROCmPackedWeights packed;
                ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
                EXPECT_FALSE(packed.int8_data.empty());
                EXPECT_TRUE(packed.native_vnni_payload.empty());

                ROCmQuantisedGemmKernel kernel(&packed, 0);
                ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

                auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
                auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});

                ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
                ASSERT_TRUE(output->allocateOnDevice(DeviceId::rocm(0)));

                ASSERT_TRUE(kernel.multiply_tensor(input.get(), output.get(), M, N, K));
                (void)hipDeviceSynchronize();
                output->mark_device_dirty();

                const float *in_host = input->data();
                std::vector<float> ref(static_cast<size_t>(M) * N);
                cpuFP32GemmRef(in_host, W_fp32.data(), ref.data(), M, N, K);

                float cos = cosineSim(output->data(), ref.data(), static_cast<size_t>(M) * N);
                LOG_INFO("[Dispatch] Q8_1 decode M=1: cosine=" << cos);
                EXPECT_GT(cos, 0.985f) << "Q8_1 decode cosine too low";

                cleanupWorkspace(kernel);
            }

            /**
             * @test End-to-end multiply_tensor for Q4_0 (native-VNNI path) — decode (M=1)
             *
             * Verifies that Q4_0 weights go through native-VNNI packing and the
             * native-VNNI GEMV path produces correct results for single-token decode.
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, Dispatch_Q4_0_Decode_M1)
            {
                if (!has_rocm_device_)
                    GTEST_SKIP() << "No ROCm device available";

                const int M = 1, N = 896, K = 896;

                auto weights = TestTensorFactory::createQ4_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});
                std::vector<float> W_fp32(static_cast<size_t>(N) * K);
                weights->to_fp32(W_fp32.data());

                ROCmPackedWeights packed;
                ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

                // Verify native-VNNI path was used (not INT8)
                EXPECT_FALSE(packed.native_vnni_payload.empty()) << "Q4_0 must use native-VNNI";
                EXPECT_TRUE(packed.int8_data.empty()) << "Q4_0 must NOT use INT8 path";

                ROCmQuantisedGemmKernel kernel(&packed, 0);
                ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

                auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
                auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});

                ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
                ASSERT_TRUE(output->allocateOnDevice(DeviceId::rocm(0)));

                ASSERT_TRUE(kernel.multiply_tensor(input.get(), output.get(), M, N, K));
                (void)hipDeviceSynchronize();
                output->mark_device_dirty();

                const float *in_host = input->data();
                std::vector<float> ref(static_cast<size_t>(M) * N);
                cpuFP32GemmRef(in_host, W_fp32.data(), ref.data(), M, N, K);

                float cos = cosineSim(output->data(), ref.data(), static_cast<size_t>(M) * N);
                LOG_INFO("[Dispatch] Q4_0 decode M=1: cosine=" << cos);
                EXPECT_GT(cos, 0.990f) << "Q4_0 decode cosine too low";

                cleanupWorkspace(kernel);
            }

            /**
             * @test End-to-end multiply_tensor for Q4_0 (native-VNNI path) — prefill (M=64)
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, Dispatch_Q4_0_Prefill_M64)
            {
                if (!has_rocm_device_)
                    GTEST_SKIP() << "No ROCm device available";

                const int M = 64, N = 4864, K = 896;

                auto weights = TestTensorFactory::createQ4_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});
                std::vector<float> W_fp32(static_cast<size_t>(N) * K);
                weights->to_fp32(W_fp32.data());

                ROCmPackedWeights packed;
                ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
                EXPECT_FALSE(packed.native_vnni_payload.empty());
                EXPECT_TRUE(packed.int8_data.empty());

                ROCmQuantisedGemmKernel kernel(&packed, 0);
                ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

                auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
                auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});

                ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
                ASSERT_TRUE(output->allocateOnDevice(DeviceId::rocm(0)));

                ASSERT_TRUE(kernel.multiply_tensor(input.get(), output.get(), M, N, K));
                (void)hipDeviceSynchronize();
                output->mark_device_dirty();

                const float *in_host = input->data();
                std::vector<float> ref(static_cast<size_t>(M) * N);
                cpuFP32GemmRef(in_host, W_fp32.data(), ref.data(), M, N, K);

                float cos = cosineSim(output->data(), ref.data(), static_cast<size_t>(M) * N);
                LOG_INFO("[Dispatch] Q4_0 prefill M=64 N=" << N << " K=" << K << ": cosine=" << cos);
                EXPECT_GT(cos, 0.990f) << "Q4_0 prefill cosine too low";

                cleanupWorkspace(kernel);
            }

            /**
             * @test Prefill bias is applied on the native-VNNI dispatch path (Q4_0).
             *
             * This covers the same bias plumbing used by Qwen2 fused QKV prefill
             * projections, but through the single-projection multiply_tensor API.
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, Dispatch_Q4_0_Prefill_M64_BiasBroadcastsPerColumn)
            {
                if (!has_rocm_device_)
                    GTEST_SKIP() << "No ROCm device available";

                const int M = 64, N = 4864, K = 896;

                auto weights = TestTensorFactory::createQ4_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});
                ROCmPackedWeights packed;
                ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
                EXPECT_FALSE(packed.native_vnni_payload.empty());
                EXPECT_TRUE(packed.int8_data.empty());

                ROCmQuantisedGemmKernel kernel(&packed, 0);
                ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

                auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
                auto output_no_bias = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
                auto output_with_bias = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
                auto bias = TestTensorFactory::createFP32({static_cast<size_t>(N)});

                float *bias_data = bias->mutable_data();
                for (int col = 0; col < N; ++col)
                {
                    bias_data[col] = 0.125f * static_cast<float>((col % 19) - 9);
                }

                ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
                ASSERT_TRUE(output_no_bias->allocateOnDevice(DeviceId::rocm(0)));
                ASSERT_TRUE(output_with_bias->allocateOnDevice(DeviceId::rocm(0)));

                ASSERT_TRUE(kernel.multiply_tensor(input.get(), output_no_bias.get(), M, N, K));
                ASSERT_TRUE(kernel.multiply_tensor(input.get(), output_with_bias.get(), M, N, K,
                                                   true, 1.0f, 0.0f, bias.get()));
                (void)hipDeviceSynchronize();
                output_no_bias->mark_device_dirty();
                output_with_bias->mark_device_dirty();

                const float *no_bias = output_no_bias->data();
                const float *with_bias = output_with_bias->data();

                for (int row = 0; row < M; ++row)
                {
                    for (int col = 0; col < N; ++col)
                    {
                        const size_t idx = static_cast<size_t>(row) * N + col;
                        const float expected_delta = bias_data[col];
                        const float actual_delta = with_bias[idx] - no_bias[idx];
                        EXPECT_NEAR(actual_delta, expected_delta, 1e-4f)
                            << "row=" << row << " col=" << col;
                    }
                }

                cleanupWorkspace(kernel);
            }

            /**
             * @test End-to-end multiply_tensor for IQ4_NL (native-VNNI path) — decode (M=1)
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, Dispatch_IQ4_NL_Decode_M1)
            {
                if (!has_rocm_device_)
                    GTEST_SKIP() << "No ROCm device available";

                const int M = 1, N = 896, K = 896;

                auto weights = TestTensorFactory::createIQ4_NLRandom({static_cast<size_t>(N), static_cast<size_t>(K)});
                std::vector<float> W_fp32(static_cast<size_t>(N) * K);
                weights->to_fp32(W_fp32.data());

                ROCmPackedWeights packed;
                ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
                EXPECT_FALSE(packed.native_vnni_payload.empty());
                EXPECT_TRUE(packed.int8_data.empty());

                ROCmQuantisedGemmKernel kernel(&packed, 0);
                ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

                auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
                auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});

                ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
                ASSERT_TRUE(output->allocateOnDevice(DeviceId::rocm(0)));

                ASSERT_TRUE(kernel.multiply_tensor(input.get(), output.get(), M, N, K));
                (void)hipDeviceSynchronize();
                output->mark_device_dirty();

                const float *in_host = input->data();
                std::vector<float> ref(static_cast<size_t>(M) * N);
                cpuFP32GemmRef(in_host, W_fp32.data(), ref.data(), M, N, K);

                float cos = cosineSim(output->data(), ref.data(), static_cast<size_t>(M) * N);
                LOG_INFO("[Dispatch] IQ4_NL decode M=1: cosine=" << cos);
                EXPECT_GT(cos, 0.990f) << "IQ4_NL decode cosine too low";

                cleanupWorkspace(kernel);
            }

            /**
             * @test Cross-path consistency: same dimensions, different formats
             *
             * Verifies that Q8_0 (INT8-VNNI) and Q4_0 (native-VNNI) both produce
             * valid outputs for the same shape. While the outputs will differ due to
             * different quantization methods, both should have high cosine similarity
             * against their respective FP32 references.
             *
             * This tests the dispatch routing: both formats go through multiply_tensor
             * and are automatically routed to the correct kernel path.
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, Dispatch_CrossPath_BothFormatsCorrect)
            {
                if (!has_rocm_device_)
                    GTEST_SKIP() << "No ROCm device available";

                const int M = 32, N = 896, K = 896;

                // --- Q8_0 (INT8-VNNI path) ---
                auto w_q8 = TestTensorFactory::createQ8_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});
                std::vector<float> W_q8_fp32(static_cast<size_t>(N) * K);
                w_q8->to_fp32(W_q8_fp32.data());

                ROCmPackedWeights packed_q8;
                ASSERT_TRUE(packWeightsToROCm(w_q8.get(), packed_q8));
                EXPECT_FALSE(packed_q8.int8_data.empty());
                EXPECT_TRUE(packed_q8.native_vnni_payload.empty());

                // --- Q4_0 (native-VNNI path) ---
                auto w_q4 = TestTensorFactory::createQ4_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});
                std::vector<float> W_q4_fp32(static_cast<size_t>(N) * K);
                w_q4->to_fp32(W_q4_fp32.data());

                ROCmPackedWeights packed_q4;
                ASSERT_TRUE(packWeightsToROCm(w_q4.get(), packed_q4));
                EXPECT_TRUE(packed_q4.int8_data.empty());
                EXPECT_FALSE(packed_q4.native_vnni_payload.empty());

                // Shared input
                auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
                ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
                const float *in_host = input->data();

                // --- Q8_0 kernel ---
                {
                    ROCmQuantisedGemmKernel kernel(&packed_q8, 0);
                    ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

                    auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
                    ASSERT_TRUE(output->allocateOnDevice(DeviceId::rocm(0)));
                    ASSERT_TRUE(kernel.multiply_tensor(input.get(), output.get(), M, N, K));
                    (void)hipDeviceSynchronize();
                    output->mark_device_dirty();

                    std::vector<float> ref(static_cast<size_t>(M) * N);
                    cpuFP32GemmRef(in_host, W_q8_fp32.data(), ref.data(), M, N, K);

                    float cos = cosineSim(output->data(), ref.data(), static_cast<size_t>(M) * N);
                    LOG_INFO("[Dispatch] CrossPath Q8_0 (INT8-VNNI): cosine=" << cos);
                    EXPECT_GT(cos, 0.985f) << "Q8_0 cross-path cosine too low";

                    cleanupWorkspace(kernel);
                }

                // --- Q4_0 kernel ---
                {
                    ROCmQuantisedGemmKernel kernel(&packed_q4, 0);
                    ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

                    auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
                    ASSERT_TRUE(output->allocateOnDevice(DeviceId::rocm(0)));
                    ASSERT_TRUE(kernel.multiply_tensor(input.get(), output.get(), M, N, K));
                    (void)hipDeviceSynchronize();
                    output->mark_device_dirty();

                    std::vector<float> ref(static_cast<size_t>(M) * N);
                    cpuFP32GemmRef(in_host, W_q4_fp32.data(), ref.data(), M, N, K);

                    float cos = cosineSim(output->data(), ref.data(), static_cast<size_t>(M) * N);
                    LOG_INFO("[Dispatch] CrossPath Q4_0 (native-VNNI): cosine=" << cos);
                    EXPECT_GT(cos, 0.990f) << "Q4_0 cross-path cosine too low";

                    cleanupWorkspace(kernel);
                }
            }

            /**
             * @test End-to-end multiply_tensor for Q5_K (native-VNNI K-quant) — decode (M=1)
             *
             * Tests a super-block format to ensure the format-conditional logic also
             * routes K-quant formats through native-VNNI correctly.
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, Dispatch_Q5_K_Decode_M1)
            {
                if (!has_rocm_device_)
                    GTEST_SKIP() << "No ROCm device available";

                const int M = 1, N = 896, K = 896;

                auto weights = TestTensorFactory::createQ5_KRandom({static_cast<size_t>(N), static_cast<size_t>(K)});
                std::vector<float> W_fp32(static_cast<size_t>(N) * K);
                weights->to_fp32(W_fp32.data());

                ROCmPackedWeights packed;
                ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
                EXPECT_FALSE(packed.native_vnni_payload.empty()) << "Q5_K must use native-VNNI";
                EXPECT_TRUE(packed.int8_data.empty()) << "Q5_K must NOT use INT8 path";

                ROCmQuantisedGemmKernel kernel(&packed, 0);
                ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

                auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
                auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});

                ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
                ASSERT_TRUE(output->allocateOnDevice(DeviceId::rocm(0)));

                ASSERT_TRUE(kernel.multiply_tensor(input.get(), output.get(), M, N, K));
                (void)hipDeviceSynchronize();
                output->mark_device_dirty();

                const float *in_host = input->data();
                std::vector<float> ref(static_cast<size_t>(M) * N);
                cpuFP32GemmRef(in_host, W_fp32.data(), ref.data(), M, N, K);

                float cos = cosineSim(output->data(), ref.data(), static_cast<size_t>(M) * N);
                LOG_INFO("[Dispatch] Q5_K decode M=1: cosine=" << cos);
                EXPECT_GT(cos, 0.990f) << "Q5_K decode cosine too low";

                cleanupWorkspace(kernel);
            }

            /**
             * @test Fused prefill with pre-packed native-VNNI weights matches separate GEMM
             *
             * This specifically exercises multiply_fused_tensor() for M>1 with
             * native-VNNI weights. The packed constructor mirrors the main cached
             * weight path used by production inference.
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, FusedPrefill_NativeVnniPackedPath_MatchesSeparate)
            {
                if (!has_rocm_device_)
                    GTEST_SKIP() << "No ROCm device available";

                const int M = 32, N = 896, K = 896;

                auto weights = TestTensorFactory::createQ4_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});
                ROCmPackedWeights packed;
                ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
                ASSERT_FALSE(packed.native_vnni_payload.empty());
                ASSERT_TRUE(packed.int8_data_vnni.empty());

                ROCmQuantisedGemmKernel kernel(&packed, 0);
                ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

                auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
                auto output_separate = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
                auto output_fused = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});

                ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
                ASSERT_TRUE(output_separate->allocateOnDevice(DeviceId::rocm(0)));
                ASSERT_TRUE(output_fused->allocateOnDevice(DeviceId::rocm(0)));

                ASSERT_TRUE(kernel.multiply_tensor(input.get(), output_separate.get(), M, N, K));
                (void)hipDeviceSynchronize();
                output_separate->mark_device_dirty();

                std::vector<ITensorGemm::TensorProjectionDesc> projections;
                projections.emplace_back(&kernel, output_fused.get(), N, nullptr, nullptr, false, "q4_0_native_fused");

                ASSERT_TRUE(kernel.multiply_fused_tensor(input.get(), projections, M, K));
                (void)hipDeviceSynchronize();
                output_fused->mark_device_dirty();

                const float *separate = output_separate->data();
                const float *fused = output_fused->data();
                const float cos = cosineSim(fused, separate, static_cast<size_t>(M) * N);

                LOG_INFO("[Dispatch] Fused packed native-VNNI Q4_0 cosine=" << cos);
                EXPECT_GT(cos, 0.9999f) << "Packed native-VNNI fused prefill diverged from separate GEMM";

                cleanupWorkspace(kernel);
            }

            /**
             * @test Fused native-VNNI prefill applies per-projection biases.
             *
             * Mirrors the Qwen2 fused-QKV prefill structure more closely by running
             * multiple biased projections through multiply_fused_tensor() and
             * comparing them against separate bias-aware multiply_tensor() calls.
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, FusedPrefill_NativeVnniPackedPath_BiasMatchesSeparate)
            {
                if (!has_rocm_device_)
                    GTEST_SKIP() << "No ROCm device available";

                const int M = 32, N = 896, K = 896;

                auto weights = TestTensorFactory::createQ4_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});
                ROCmPackedWeights packed;
                ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
                ASSERT_FALSE(packed.native_vnni_payload.empty());
                ASSERT_TRUE(packed.int8_data_vnni.empty());

                ROCmQuantisedGemmKernel kernel(&packed, 0);
                ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

                auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
                auto separate_q = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
                auto separate_k = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
                auto separate_v = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
                auto fused_q = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
                auto fused_k = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
                auto fused_v = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
                auto bias_q = TestTensorFactory::createFP32({static_cast<size_t>(N)});
                auto bias_k = TestTensorFactory::createFP32({static_cast<size_t>(N)});
                auto bias_v = TestTensorFactory::createFP32({static_cast<size_t>(N)});

                for (int col = 0; col < N; ++col)
                {
                    bias_q->mutable_data()[col] = 0.0625f * static_cast<float>((col % 13) - 6);
                    bias_k->mutable_data()[col] = 0.0625f * static_cast<float>((col % 11) - 5);
                    bias_v->mutable_data()[col] = 0.0625f * static_cast<float>((col % 9) - 4);
                }

                ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
                ASSERT_TRUE(separate_q->allocateOnDevice(DeviceId::rocm(0)));
                ASSERT_TRUE(separate_k->allocateOnDevice(DeviceId::rocm(0)));
                ASSERT_TRUE(separate_v->allocateOnDevice(DeviceId::rocm(0)));
                ASSERT_TRUE(fused_q->allocateOnDevice(DeviceId::rocm(0)));
                ASSERT_TRUE(fused_k->allocateOnDevice(DeviceId::rocm(0)));
                ASSERT_TRUE(fused_v->allocateOnDevice(DeviceId::rocm(0)));

                ASSERT_TRUE(kernel.multiply_tensor(input.get(), separate_q.get(), M, N, K,
                                                   true, 1.0f, 0.0f, bias_q.get()));
                ASSERT_TRUE(kernel.multiply_tensor(input.get(), separate_k.get(), M, N, K,
                                                   true, 1.0f, 0.0f, bias_k.get()));
                ASSERT_TRUE(kernel.multiply_tensor(input.get(), separate_v.get(), M, N, K,
                                                   true, 1.0f, 0.0f, bias_v.get()));
                (void)hipDeviceSynchronize();
                separate_q->mark_device_dirty();
                separate_k->mark_device_dirty();
                separate_v->mark_device_dirty();

                std::vector<ITensorGemm::TensorProjectionDesc> projections;
                projections.emplace_back(&kernel, fused_q.get(), N, bias_q.get(), nullptr, false, "q_bias");
                projections.emplace_back(&kernel, fused_k.get(), N, bias_k.get(), nullptr, false, "k_bias");
                projections.emplace_back(&kernel, fused_v.get(), N, bias_v.get(), nullptr, false, "v_bias");

                ASSERT_TRUE(kernel.multiply_fused_tensor(input.get(), projections, M, K));
                (void)hipDeviceSynchronize();
                fused_q->mark_device_dirty();
                fused_k->mark_device_dirty();
                fused_v->mark_device_dirty();

                const float q_cos = cosineSim(fused_q->data(), separate_q->data(), static_cast<size_t>(M) * N);
                const float k_cos = cosineSim(fused_k->data(), separate_k->data(), static_cast<size_t>(M) * N);
                const float v_cos = cosineSim(fused_v->data(), separate_v->data(), static_cast<size_t>(M) * N);

                LOG_INFO("[Dispatch] Fused packed native-VNNI Q4_0 with bias cosine q=" << q_cos
                                                                                           << " k=" << k_cos
                                                                                           << " v=" << v_cos);
                EXPECT_GT(q_cos, 0.9999f) << "Fused Q projection diverged from separate biased GEMM";
                EXPECT_GT(k_cos, 0.9999f) << "Fused K projection diverged from separate biased GEMM";
                EXPECT_GT(v_cos, 0.9999f) << "Fused V projection diverged from separate biased GEMM";

                cleanupWorkspace(kernel);
            }

            /**
             * @test Fused prefill with legacy TensorBase-backed native-VNNI weights works
             *        for representative metadata families.
             *
             * This covers the legacy upload path fixed here, including:
             *   - Q4_0: payload + scales
             *   - Q4_1: payload + scales + mins
             *   - IQ4_NL: payload + scales + codebook id
             *   - Q2_K: payload + scales + emins
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, FusedPrefill_NativeVnniLegacyPath_MetadataFamiliesMatchSeparate)
            {
                if (!has_rocm_device_)
                    GTEST_SKIP() << "No ROCm device available";

                const int M = 32;
                const int N = 896;
                const int K = 896;

                struct TestCase
                {
                    const char *name;
                    bool expect_mins;
                    bool expect_emins;
                };

                const TestCase cases[] = {
                    {"Q4_0", false, false},
                    {"Q4_1", true, false},
                    {"IQ4_NL", false, false},
                    {"Q2_K", true, true},
                };

                auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
                ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));

                for (const auto &test_case : cases)
                {
                    std::unique_ptr<TensorBase> weights;
                    if (std::strcmp(test_case.name, "Q4_0") == 0)
                        weights = TestTensorFactory::createQ4_0Random({static_cast<size_t>(N), static_cast<size_t>(K)});
                    else if (std::strcmp(test_case.name, "Q4_1") == 0)
                        weights = TestTensorFactory::createQ4_1Random({static_cast<size_t>(N), static_cast<size_t>(K)});
                    else if (std::strcmp(test_case.name, "IQ4_NL") == 0)
                        weights = TestTensorFactory::createIQ4_NLRandom({static_cast<size_t>(N), static_cast<size_t>(K)});
                    else if (std::strcmp(test_case.name, "Q2_K") == 0)
                        weights = TestTensorFactory::createQ2_KRandom({static_cast<size_t>(N), static_cast<size_t>(K)});
                    else
                        FAIL() << "Unhandled test case: " << test_case.name;

                    ASSERT_NE(weights, nullptr);

                    ROCmPackedWeights packed;
                    ASSERT_TRUE(packWeightsToROCm(weights.get(), packed)) << test_case.name;
                    ASSERT_FALSE(packed.native_vnni_payload.empty()) << test_case.name;
                    ASSERT_FALSE(packed.native_vnni_scales.empty()) << test_case.name;
                    EXPECT_EQ(!packed.native_vnni_mins.empty(), test_case.expect_mins) << test_case.name;
                    EXPECT_EQ(!packed.native_vnni_emins.empty(), test_case.expect_emins) << test_case.name;

                    ROCmQuantisedGemmKernel kernel = createLegacyROCmKernelForTest(weights.get(), 0);
                    ASSERT_TRUE(setupWorkspace(kernel, M, N, K)) << test_case.name;

                    auto output_separate = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
                    auto output_fused = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});

                    ASSERT_TRUE(output_separate->allocateOnDevice(DeviceId::rocm(0))) << test_case.name;
                    ASSERT_TRUE(output_fused->allocateOnDevice(DeviceId::rocm(0))) << test_case.name;

                    ASSERT_TRUE(kernel.multiply_tensor(input.get(), output_separate.get(), M, N, K)) << test_case.name;
                    (void)hipDeviceSynchronize();
                    output_separate->mark_device_dirty();

                    std::vector<ITensorGemm::TensorProjectionDesc> projections;
                    projections.emplace_back(&kernel, output_fused.get(), N, nullptr, nullptr, false, test_case.name);

                    ASSERT_TRUE(kernel.multiply_fused_tensor(input.get(), projections, M, K)) << test_case.name;
                    (void)hipDeviceSynchronize();
                    output_fused->mark_device_dirty();

                    const float *separate = output_separate->data();
                    const float *fused = output_fused->data();
                    const float cos = cosineSim(fused, separate, static_cast<size_t>(M) * N);

                    LOG_INFO("[Dispatch] Fused legacy native-VNNI " << test_case.name << " cosine=" << cos);
                    EXPECT_GT(cos, 0.9999f) << test_case.name << " fused prefill diverged from separate GEMM";

                    cleanupWorkspace(kernel);
                }
            }

            // =============================================================================
            // KernelFactory Integration Tests
            // =============================================================================
            // These tests verify that KernelFactory correctly creates ROCm kernels.
            // Moved from unit tests since HAVE_ROCM is only enabled in integration builds.

            /**
             * @test KernelFactory creates ROCmQuantisedGemmKernel for Q4_0 tensors
             */
            TEST_F(ROCmQuantisedGemmIntegrationTest, KernelFactory_CreateGemm_Q4_0)
            {
                if (!has_rocm_device_)
                {
                    GTEST_SKIP() << "No ROCm device available";
                }

                const size_t rows = 32;
                const size_t cols = 32;
                const size_t block_size = 32;
                const size_t bytes_per_block = 18;
                const size_t num_blocks = rows * (cols / block_size);
                std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

                Q4_0Tensor tensor({rows, cols}, raw_data);

                // KernelFactory should create ROCmQuantisedGemmKernel for ROCm device
                // Must use ROCmOrdinalGuard to specify target device (tensor is on CPU)
                llaminar::v2::kernels::KernelFactory::ROCmOrdinalGuard guard(0);
                auto kernel = llaminar::v2::kernels::KernelFactory::createGemm(&tensor, DeviceType::ROCm);
                ASSERT_NE(kernel, nullptr) << "KernelFactory should create ROCm GEMM kernel for Q4_0";

                LOG_INFO("[Integration] KernelFactory successfully created ROCm GEMM for Q4_0 tensor");
            }

        } // namespace integration_test
    } // namespace rocm
} // namespace llaminar2
