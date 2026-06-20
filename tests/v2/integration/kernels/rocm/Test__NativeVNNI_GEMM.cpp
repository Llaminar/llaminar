/**
 * @file Test__NativeVNNI_GEMM.cpp
 * @brief Integration tests for ALL native-VNNI GEMM kernel dispatch paths (M > 1)
 *
 * The native-VNNI GEMM system has multiple kernel variants selected by a
 * dispatch heuristic based on N, K, and M.  This test file exercises every
 * dispatch path for correctness by comparing GPU output against a hipBLAS
 * FP32 SGEMM reference (dequantize weights → dense FP32 matmul via hipBLAS).
 *
 * Dispatch paths tested:
 *
 *   1. STREAMING (N >= 16384 && K <= 4096)
 *      - Barrier-free, LDS-free register streaming.
 *      - Shapes: LM_Head (N=20480 proxy), 7B FFN_Up (N=18944), boundary (N=16384)
 *
 *   2. N128 / M16 / 3-wave  (N > K, total_wgs <= 256)
 *      - Cooperative-LDS kernel, 3-wave occupancy, small grids.
 *      - Shape: 0.5B FFN_Up (N=4864, K=896) at small M
 *
 *   3. N128 / M32 / 2-wave  (N > K, total_wgs > 256)
 *      - Cooperative-LDS kernel, 2-wave occupancy, better tile efficiency.
 *      - Shape: 3B FFN_Up (N=11008, K=2048) at large M
 *
 *   4. N64 / M16 / 3-wave  (N <= K, total_wgs_m16 <= 128)
 *      - Cooperative-LDS kernel, N64 tile, 3-wave small-grid path.
 *      - Shape: 0.5B AttnOut (N=896, K=896) at small M
 *
 *   5. N64 / M16..M64 / 2-wave  (N <= K, larger grids)
 *      - Cooperative-LDS kernel, N64 tile, M_TILE selected by M.
 *      - Shapes: 0.5B FFN_Dn (N=896, K=4864), 7B FFN_Dn (N=3584, K=18944)
 *
 * Formats tested: All 15 native-VNNI GEMM formats (Q4_0, IQ4_NL, Q4_1, Q5_0,
 * Q5_1, Q6_K, Q3_K, Q2_K, IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS, IQ1_S, IQ1_M).
 *
 * M values: 1, 7, 16, 17, 32, 33, 64 — covering exact tile boundaries,
 * non-tile-aligned M (boundary tiles), and M=1 (GEMV fallback).
 *
 * @note Requires ROCm device and build_v2_integration.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"
#include "../../../utils/TestTensorFactory.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#endif

using namespace llaminar2;
using namespace llaminar2::rocm;
using namespace llaminar2::test;

namespace
{

    // =============================================================================
    // Helpers
    // =============================================================================

    float cosineSimilarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            norm_a += static_cast<double>(a[i]) * a[i];
            norm_b += static_cast<double>(b[i]) * b[i];
        }
        if (norm_a == 0.0 || norm_b == 0.0)
            return 0.0f;
        return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
    }

    float maxAbsError(const float *a, const float *b, size_t n)
    {
        float max_err = 0.0f;
        for (size_t i = 0; i < n; ++i)
            max_err = std::max(max_err, std::fabs(a[i] - b[i]));
        return max_err;
    }

    size_t firstBitwiseMismatchIndex(const std::vector<float> &lhs,
                                     const std::vector<float> &rhs)
    {
        const size_t count = std::min(lhs.size(), rhs.size());
        for (size_t i = 0; i < count; ++i)
        {
            if (std::memcmp(&lhs[i], &rhs[i], sizeof(float)) != 0)
            {
                return i;
            }
        }

        return (lhs.size() == rhs.size()) ? lhs.size() : count;
    }

#ifdef HAVE_ROCM
    /// hipBLAS FP32 reference GEMM: C[i,j] = sum_k(A[i,k] * W[j,k])
    /// A is [M x K] row-major, W is [N x K] row-major (weight layout)
    /// Row-major C = A * W^T is computed via the column-major identity:
    ///   C' = W'^T * A'  (where X' denotes the col-major view of row-major X)
    bool hipblasFP32Gemm(const float *A_host, const float *W_host, float *C_host,
                         int M, int N, int K)
    {
        hipblasHandle_t handle;
        if (hipblasCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
            return false;

        const size_t size_A = static_cast<size_t>(M) * K * sizeof(float);
        const size_t size_W = static_cast<size_t>(N) * K * sizeof(float);
        const size_t size_C = static_cast<size_t>(M) * N * sizeof(float);

        float *d_A = nullptr, *d_W = nullptr, *d_C = nullptr;
        hipMalloc(&d_A, size_A);
        hipMalloc(&d_W, size_W);
        hipMalloc(&d_C, size_C);

        hipMemcpy(d_A, A_host, size_A, hipMemcpyHostToDevice);
        hipMemcpy(d_W, W_host, size_W, hipMemcpyHostToDevice);

        const float alpha = 1.0f, beta = 0.0f;
        hipblasStatus_t status = hipblasSgemm(
            handle,
            HIPBLAS_OP_T,   // transpose W' (col-major view of row-major W[N,K])
            HIPBLAS_OP_N,   // no transpose A' (col-major view of row-major A[M,K])
            N, M, K,        // m=N, n=M, k=K
            &alpha,
            d_W, K,         // lda = K (W' is [K,N] col-major)
            d_A, K,         // ldb = K (A' is [K,M] col-major)
            &beta,
            d_C, N          // ldc = N (C' is [N,M] col-major)
        );

        hipDeviceSynchronize();
        hipMemcpy(C_host, d_C, size_C, hipMemcpyDeviceToHost);

        (void)hipFree(d_A);
        (void)hipFree(d_W);
        (void)hipFree(d_C);
        hipblasDestroy(handle);
        return (status == HIPBLAS_STATUS_SUCCESS);
    }
#endif

    // =============================================================================
    // Test parameter
    // =============================================================================

    struct GEMMTestParam
    {
        std::string name;
        std::string format; // e.g. "Q4_0", "IQ4_NL", "Q4_1", etc.
        int N, K, M;
        std::string dispatch; // Expected dispatch path (for documentation only)
        float min_cosine;
    };

    std::string paramName(const ::testing::TestParamInfo<GEMMTestParam> &info)
    {
        return info.param.name;
    }

    // =============================================================================
    // Fixture
    // =============================================================================

    class NativeVNNIGEMMTest
        : public ::testing::Test,
          public ::testing::WithParamInterface<GEMMTestParam>
    {
    protected:
        void SetUp() override
        {
#ifdef HAVE_ROCM
            int count = 0;
            hipError_t err = hipGetDeviceCount(&count);
            has_gpu_ = (err == hipSuccess && count > 0);
            if (has_gpu_)
            {
                hipDeviceProp_t props;
                (void)hipGetDeviceProperties(&props, 0);
                gpu_name_ = std::string(props.name) + " (" + props.gcnArchName + ")";
            }
#else
            has_gpu_ = false;
#endif
        }

        bool has_gpu_ = false;
        std::string gpu_name_;

#ifdef HAVE_ROCM
        std::unique_ptr<DeviceWorkspaceManager> workspace_;

        bool setupWorkspace(ROCmQuantisedGemmKernel &kernel, int M, int N, int K)
        {
            auto reqs = kernel.getWorkspaceRequirements(M, N, K);
            size_t ws_bytes = std::max(static_cast<size_t>(256 * 1024 * 1024),
                                       static_cast<size_t>(reqs.total_bytes() * 2));
            workspace_ = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(0), ws_bytes);
            if (!workspace_->allocate(reqs))
            {
                LOG_ERROR("Workspace allocation failed (" << ws_bytes << " bytes)");
                return false;
            }
            kernel.bindWorkspace(workspace_.get());
            return true;
        }

        void cleanupWorkspace(ROCmQuantisedGemmKernel &kernel)
        {
            if (workspace_)
                kernel.unbindWorkspace();
        }
#endif
    };

    // =============================================================================
    // Core test body: GPU vs CPU FP32 reference
    // =============================================================================

    TEST_P(NativeVNNIGEMMTest, MatchesFP32Reference)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_gpu_)
            GTEST_SKIP() << "No ROCm device";

        const auto &p = GetParam();

        LOG_INFO("[NativeVNNI_GEMM] " << p.name
                                      << "  format=" << p.format
                                      << "  M=" << p.M << " N=" << p.N << " K=" << p.K
                                      << "  dispatch=" << p.dispatch
                                      << "  GPU=" << gpu_name_);

        // 1. Create quantized weights [N x K]
        std::unique_ptr<TensorBase> weights;
        const auto sz = std::make_pair(static_cast<size_t>(p.N), static_cast<size_t>(p.K));
        if (p.format == "Q4_0")
            weights = TestTensorFactory::createQ4_0Random({sz.first, sz.second});
        else if (p.format == "IQ4_NL")
            weights = TestTensorFactory::createIQ4_NLRandom({sz.first, sz.second});
        else if (p.format == "IQ4_XS")
            weights = TestTensorFactory::createIQ4_XSRandom({sz.first, sz.second});
        else if (p.format == "Q4_1")
            weights = TestTensorFactory::createQ4_1Random({sz.first, sz.second});
        else if (p.format == "Q5_0")
            weights = TestTensorFactory::createQ5_0Random({sz.first, sz.second});
        else if (p.format == "Q5_1")
            weights = TestTensorFactory::createQ5_1Random({sz.first, sz.second});
        else if (p.format == "Q6_K")
            weights = TestTensorFactory::createQ6_KRandom({sz.first, sz.second});
        else if (p.format == "Q3_K")
            weights = TestTensorFactory::createQ3_KRandom({sz.first, sz.second});
        else if (p.format == "Q2_K")
            weights = TestTensorFactory::createQ2_KRandom({sz.first, sz.second});
        else if (p.format == "IQ3_S")
            weights = TestTensorFactory::createIQ3_SRandom({sz.first, sz.second});
        else if (p.format == "IQ3_XXS")
            weights = TestTensorFactory::createIQ3_XXSRandom({sz.first, sz.second});
        else if (p.format == "IQ2_S")
            weights = TestTensorFactory::createIQ2_SRandom({sz.first, sz.second});
        else if (p.format == "IQ2_XS")
            weights = TestTensorFactory::createIQ2_XSRandom({sz.first, sz.second});
        else if (p.format == "IQ2_XXS")
            weights = TestTensorFactory::createIQ2_XXSRandom({sz.first, sz.second});
        else if (p.format == "IQ1_S")
            weights = TestTensorFactory::createIQ1_SRandom({sz.first, sz.second});
        else if (p.format == "IQ1_M")
            weights = TestTensorFactory::createIQ1_MRandom({sz.first, sz.second});
        else
            FAIL() << "Unknown format: " << p.format;

        ASSERT_NE(weights, nullptr);

        // 2. Dequantize to FP32 reference
        std::vector<float> W_fp32(static_cast<size_t>(p.N) * p.K);
        weights->to_fp32(W_fp32.data());

        // 3. Pack for ROCm
        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
        ASSERT_FALSE(packed.native_vnni_payload.empty());

        // 4. Kernel + workspace
        ROCmQuantisedGemmKernel kernel(&packed, 0);
        ASSERT_TRUE(setupWorkspace(kernel, p.M, p.N, p.K));

        // 5. Random input [M x K], zero-init output [M x N]
        auto input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(p.M), static_cast<size_t>(p.K)});
        auto output_gpu = TestTensorFactory::createFP32(
            {static_cast<size_t>(p.M), static_cast<size_t>(p.N)});

        // 5b. Upload input to GPU and allocate output on device
        ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)))
            << p.name << ": failed to upload input to device";
        ASSERT_TRUE(output_gpu->allocateOnDevice(DeviceId::rocm(0)))
            << p.name << ": failed to allocate output on device";

        // 6. GPU GEMM
        ASSERT_TRUE(kernel.multiply_tensor(input.get(), output_gpu.get(),
                                           p.M, p.N, p.K))
            << p.name << ": multiply_tensor failed";
        (void)hipDeviceSynchronize();

        // 6b. Mark output device-dirty so data() triggers D2H sync
        output_gpu->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

        // 7. hipBLAS FP32 reference (fast GPU-based ground truth)
        const float *input_host = input->data(); // triggers D2H if needed
        std::vector<float> ref(static_cast<size_t>(p.M) * p.N);
        ASSERT_TRUE(hipblasFP32Gemm(input_host, W_fp32.data(), ref.data(), p.M, p.N, p.K))
            << p.name << ": hipBLAS reference GEMM failed";

        // 8. Compare — per-row cosine + overall
        const float *gpu = output_gpu->data();
        float worst_cos = 1.0f;
        int worst_row = -1;

        for (int r = 0; r < p.M; ++r)
        {
            float cos = cosineSimilarity(
                gpu + r * p.N, ref.data() + r * p.N, static_cast<size_t>(p.N));
            if (cos < worst_cos)
            {
                worst_cos = cos;
                worst_row = r;
            }
            // Spot-check NaN/Inf
            for (int c = 0; c < std::min(p.N, 512); ++c)
            {
                ASSERT_FALSE(std::isnan(gpu[r * p.N + c]))
                    << p.name << " row=" << r << " col=" << c << " is NaN";
                ASSERT_FALSE(std::isinf(gpu[r * p.N + c]))
                    << p.name << " row=" << r << " col=" << c << " is Inf";
            }
        }

        float overall_cos = cosineSimilarity(
            gpu, ref.data(), static_cast<size_t>(p.M) * p.N);
        float mae = maxAbsError(
            gpu, ref.data(), static_cast<size_t>(p.M) * p.N);

        LOG_INFO("[NativeVNNI_GEMM] " << p.name
                                      << "  overall_cos=" << overall_cos
                                      << "  worst_row_cos=" << worst_cos << " (row " << worst_row << ")"
                                      << "  max_abs_err=" << mae);

        EXPECT_GT(overall_cos, p.min_cosine)
            << p.name << ": overall cosine " << overall_cos
            << " < " << p.min_cosine;
        EXPECT_GT(worst_cos, p.min_cosine - 0.005f)
            << p.name << ": worst row cosine " << worst_cos
            << " (row " << worst_row << ")";

        cleanupWorkspace(kernel);
#endif
    }

    /**
     * @test Representative native-VNNI GEMM dispatches are bitwise stable.
     *
     * Guards against the CUDA-style failure modes we just fixed there:
     * repeated-run drift from split-K atomics or shared scratch reuse.
     * Native ROCm GEMM should be deterministic because each output element is
     * written exactly once, but this test keeps that property under CI.
     */
    TEST_F(NativeVNNIGEMMTest, Q4_0_RepeatedRuns_AreBitwiseStable_AcrossDispatchFamilies)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_gpu_)
        {
            GTEST_SKIP() << "No ROCm device";
        }

        struct DeterminismShape
        {
            const char *name;
            int M;
            int N;
            int K;
        };

        const std::vector<DeterminismShape> shapes = {
            {"streaming_lm_head_proxy", 16, 20480, 896},
            {"n128_cooperative_ffn_up", 16, 4864, 896},
            {"n64_cooperative_ffn_down", 128, 896, 4864},
        };
        constexpr int kRepeatRuns = 4;

        for (const auto &shape : shapes)
        {
            LOG_INFO("[NativeVNNI_GEMM] Determinism check " << shape.name
                                                            << " M=" << shape.M
                                                            << " N=" << shape.N
                                                            << " K=" << shape.K);

            auto weights = TestTensorFactory::createQ4_0Random(
                {static_cast<size_t>(shape.N), static_cast<size_t>(shape.K)});
            ASSERT_NE(weights, nullptr) << shape.name;

            ROCmPackedWeights packed;
            ASSERT_TRUE(packWeightsToROCm(weights.get(), packed)) << shape.name;
            ASSERT_FALSE(packed.native_vnni_payload.empty()) << shape.name;

            ROCmQuantisedGemmKernel kernel(&packed, 0);
            ASSERT_TRUE(setupWorkspace(kernel, shape.M, shape.N, shape.K)) << shape.name;

            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(shape.M), static_cast<size_t>(shape.K)});
            auto output_gpu = TestTensorFactory::createFP32(
                {static_cast<size_t>(shape.M), static_cast<size_t>(shape.N)});

            ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0))) << shape.name;
            ASSERT_TRUE(output_gpu->allocateOnDevice(DeviceId::rocm(0))) << shape.name;

            std::vector<float> reference;
            for (int run = 0; run < kRepeatRuns; ++run)
            {
                ASSERT_TRUE(kernel.multiply_tensor(input.get(), output_gpu.get(),
                                                   shape.M, shape.N, shape.K))
                    << shape.name << " run=" << run;
                (void)hipDeviceSynchronize();
                output_gpu->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

                const float *gpu = output_gpu->data();
                std::vector<float> snapshot(
                    gpu, gpu + static_cast<size_t>(shape.M) * shape.N);

                if (run == 0)
                {
                    reference = std::move(snapshot);
                    continue;
                }

                const size_t mismatch = firstBitwiseMismatchIndex(reference, snapshot);
                EXPECT_EQ(mismatch, reference.size())
                    << shape.name
                    << " run=" << run
                    << " first_mismatch=" << mismatch
                    << " reference=" << reference[mismatch]
                    << " candidate=" << snapshot[mismatch];

                if (mismatch != reference.size())
                {
                    break;
                }
            }

            cleanupWorkspace(kernel);
        }
#endif
    }

    // =============================================================================
    // Test cases — organised by dispatch path
    //
    // Naming: <Format>_<DispatchPath>_<Shape>_M<value>
    //
    // Shape dimensions are kept small enough for CI speed but large enough to
    // exercise the real dispatch conditions.  Where necessary a proxy N is used:
    //   - LM_Head proxy: N=20480 (> 16384 threshold, ~7× smaller than real 151936)
    //   - FFN shapes use real Qwen dimensions
    // =============================================================================

    // Proxy N for LM_Head-class shapes: large enough to trigger streaming (>= 16384)
    // but small enough that the CPU FP32 reference completes in reasonable CI time.
    static constexpr int LM_N = 20480;

    static const std::vector<GEMMTestParam> ALL_GEMM_PARAMS = {

        // =====================================================================
        // PATH 1: STREAMING  (N >= 16384 && K <= 4096)
        //   streaming_lm_head_kernel, M_TILE=16, THREADS_N=256, zero barriers
        // =====================================================================

        // --- Q4_0 ---
        // LM_Head proxy, K=896 (0.5B hidden dim)
        {"Q4_0_Stream_LM05B_M1", "Q4_0", LM_N, 896, 1, "Streaming", 0.990f},
        {"Q4_0_Stream_LM05B_M16", "Q4_0", LM_N, 896, 16, "Streaming", 0.990f},
        {"Q4_0_Stream_LM05B_M32", "Q4_0", LM_N, 896, 32, "Streaming", 0.990f},
        {"Q4_0_Stream_LM05B_M64", "Q4_0", LM_N, 896, 64, "Streaming", 0.990f},
        // LM_Head proxy, K=2048 (3B hidden dim)
        {"Q4_0_Stream_LM3B_M16", "Q4_0", LM_N, 2048, 16, "Streaming", 0.990f},
        {"Q4_0_Stream_LM3B_M32", "Q4_0", LM_N, 2048, 32, "Streaming", 0.990f},
        // LM_Head proxy, K=3584 (7B hidden dim)
        {"Q4_0_Stream_LM7B_M16", "Q4_0", LM_N, 3584, 16, "Streaming", 0.990f},
        {"Q4_0_Stream_LM7B_M32", "Q4_0", LM_N, 3584, 32, "Streaming", 0.990f},
        // 7B FFN_Up (N=18944, K=3584) — real shape, just above 16384
        {"Q4_0_Stream_FFNUp7B_M16", "Q4_0", 18944, 3584, 16, "Streaming", 0.990f},
        {"Q4_0_Stream_FFNUp7B_M32", "Q4_0", 18944, 3584, 32, "Streaming", 0.990f},
        // Exact boundary: N=16384
        {"Q4_0_Stream_Bound_M16", "Q4_0", 16384, 896, 16, "Streaming", 0.990f},
        {"Q4_0_Stream_Bound_M32", "Q4_0", 16384, 896, 32, "Streaming", 0.990f},
        // Non-aligned M (boundary tile in streaming kernel)
        {"Q4_0_Stream_LM05B_M7", "Q4_0", LM_N, 896, 7, "Streaming", 0.990f},
        {"Q4_0_Stream_LM05B_M17", "Q4_0", LM_N, 896, 17, "Streaming", 0.990f},
        {"Q4_0_Stream_LM05B_M33", "Q4_0", LM_N, 896, 33, "Streaming", 0.990f},

        // --- IQ4_NL ---
        {"IQ4NL_Stream_LM05B_M1", "IQ4_NL", LM_N, 896, 1, "Streaming", 0.990f},
        {"IQ4NL_Stream_LM05B_M16", "IQ4_NL", LM_N, 896, 16, "Streaming", 0.990f},
        {"IQ4NL_Stream_LM05B_M32", "IQ4_NL", LM_N, 896, 32, "Streaming", 0.990f},
        {"IQ4NL_Stream_LM05B_M64", "IQ4_NL", LM_N, 896, 64, "Streaming", 0.990f},
        {"IQ4NL_Stream_LM3B_M16", "IQ4_NL", LM_N, 2048, 16, "Streaming", 0.990f},
        {"IQ4NL_Stream_LM7B_M16", "IQ4_NL", LM_N, 3584, 16, "Streaming", 0.990f},
        {"IQ4NL_Stream_FFNUp7B_M32", "IQ4_NL", 18944, 3584, 32, "Streaming", 0.990f},
        {"IQ4NL_Stream_LM05B_M17", "IQ4_NL", LM_N, 896, 17, "Streaming", 0.990f},

        // =====================================================================
        // PATH 2: N128 / M16 / 3-wave  (N > K, WGs_m16 <= 256)
        //   native_vnni_gemm_kernel, N_TILE=128, M_TILE=16, MIN_BLOCKS=3
        //   Shape needed: N > K, and (N/128) * (M/16) <= 256
        //   0.5B FFN_Up: N=4864/128=38 blocks → M=16: 38*1=38 WGs (<<256) ✓
        //                                        M=32: 38*2=76 WGs (<=256) ✓
        //                                        M=64: 38*4=152 WGs (<=256) ✓
        // =====================================================================

        // --- Q4_0 ---
        {"Q4_0_N128M16x3_FFNUp05_M16", "Q4_0", 4864, 896, 16, "N128_M16_3wave", 0.990f},
        {"Q4_0_N128M16x3_FFNUp05_M32", "Q4_0", 4864, 896, 32, "N128_M16_3wave", 0.990f},
        {"Q4_0_N128M16x3_FFNUp05_M7", "Q4_0", 4864, 896, 7, "N128_M16_3wave", 0.990f},
        {"Q4_0_N128M16x3_FFNUp05_M17", "Q4_0", 4864, 896, 17, "N128_M16_3wave", 0.990f},

        // --- IQ4_NL ---
        {"IQ4NL_N128M16x3_FFNUp05_M16", "IQ4_NL", 4864, 896, 16, "N128_M16_3wave", 0.990f},
        {"IQ4NL_N128M16x3_FFNUp05_M32", "IQ4_NL", 4864, 896, 32, "N128_M16_3wave", 0.990f},

        // =====================================================================
        // PATH 3: N128 / M32 / 2-wave  (N > K, WGs_m16 > 256)
        //   native_vnni_gemm_kernel, N_TILE=128, M_TILE=32, MIN_BLOCKS=2
        //   Shape needed: N > K, and (N/128) * (M/16) > 256, and m_tile >= 32
        //   3B FFN_Up: N=11008/128=86 blocks
        //     M=64: 86*4=344 WGs (>256), m_tile = min(64,32)=32 → M32 ✓
        //     M=128: 86*8=688 WGs (>256), m_tile=32 → M32 ✓
        // =====================================================================

        // --- Q4_0 ---
        {"Q4_0_N128M32x2_FFNUp3B_M64", "Q4_0", 11008, 2048, 64, "N128_M32_2wave", 0.990f},
        {"Q4_0_N128M32x2_FFNUp3B_M128", "Q4_0", 11008, 2048, 128, "N128_M32_2wave", 0.990f},
        {"Q4_0_N128M32x2_FFNUp3B_M33", "Q4_0", 11008, 2048, 33, "N128_M32_2wave", 0.990f},

        // --- IQ4_NL ---
        {"IQ4NL_N128M32x2_FFNUp3B_M64", "IQ4_NL", 11008, 2048, 64, "N128_M32_2wave", 0.990f},
        {"IQ4NL_N128M32x2_FFNUp3B_M128", "IQ4_NL", 11008, 2048, 128, "N128_M32_2wave", 0.990f},

        // =====================================================================
        // PATH 4: N64 / M16 / 3-wave  (N <= K, WGs_m16 <= 128)
        //   native_vnni_gemm_kernel, N_TILE=64, M_TILE=16, MIN_BLOCKS=3
        //   Shape needed: N <= K, M <= 64, and (N/64) * (M/16) <= 128
        //   0.5B AttnOut: N=896/64=14 blocks
        //     M=16: 14*1=14 WGs (<=128) ✓
        //     M=32: 14*2=28 WGs (<=128) ✓
        //     M=64: 14*4=56 WGs (<=128) ✓
        //   0.5B FFN_Dn: N=896/64=14 blocks
        //     M=16: 14*1=14 WGs (<=128) ✓
        // =====================================================================

        // --- Q4_0 ---
        {"Q4_0_N64M16x3_Attn05_M16", "Q4_0", 896, 896, 16, "N64_M16_3wave", 0.990f},
        {"Q4_0_N64M16x3_Attn05_M32", "Q4_0", 896, 896, 32, "N64_M16_3wave", 0.990f},
        {"Q4_0_N64M16x3_Attn05_M64", "Q4_0", 896, 896, 64, "N64_M16_3wave", 0.990f},
        {"Q4_0_N64M16x3_FFNDn05_M16", "Q4_0", 896, 4864, 16, "N64_M16_3wave", 0.990f},
        {"Q4_0_N64M16x3_Attn05_M7", "Q4_0", 896, 896, 7, "N64_M16_3wave", 0.990f},

        // --- IQ4_NL ---
        {"IQ4NL_N64M16x3_Attn05_M16", "IQ4_NL", 896, 896, 16, "N64_M16_3wave", 0.990f},
        {"IQ4NL_N64M16x3_Attn05_M32", "IQ4_NL", 896, 896, 32, "N64_M16_3wave", 0.990f},

        // =====================================================================
        // PATH 5: N64 / M16 / 2-wave  (N <= K, M<=64, WGs > 128, m_tile=16)
        //   native_vnni_gemm_kernel, N_TILE=64, M_TILE=16, MIN_BLOCKS=2
        //   Shape needed: N <= K, M=32..64, WGs_m16 > 128
        //   7B AttnOut: N=3584/64=56 blocks
        //     M=64: 56*4=224 WGs (>128), m_tile=16 → launch(N64,M16) ✓
        // =====================================================================

        // --- Q4_0 ---
        {"Q4_0_N64M16x2_Attn7B_M64", "Q4_0", 3584, 3584, 64, "N64_M16_2wave", 0.990f},

        // --- IQ4_NL ---
        {"IQ4NL_N64M16x2_Attn7B_M64", "IQ4_NL", 3584, 3584, 64, "N64_M16_2wave", 0.990f},

        // =====================================================================
        // PATH 6: N64 / M32 / 2-wave  (N <= K, M > 64 → m_tile=32)
        //   native_vnni_gemm_kernel, N_TILE=64, M_TILE=32, MIN_BLOCKS=2
        //   0.5B AttnOut at M=128: m_tile = min(128, 32)=32
        //   M=128 is large enough to get m_tile=32 on most shapes
        // =====================================================================

        // --- Q4_0 ---
        {"Q4_0_N64M32x2_Attn05_M128", "Q4_0", 896, 896, 128, "N64_M32_2wave", 0.990f},
        {"Q4_0_N64M32x2_FFNDn05_M128", "Q4_0", 896, 4864, 128, "N64_M32_2wave", 0.990f},
        {"Q4_0_N64M32x2_FFNDn7B_M128", "Q4_0", 3584, 18944, 128, "N64_M32_2wave", 0.990f},

        // --- IQ4_NL ---
        {"IQ4NL_N64M32x2_Attn05_M128", "IQ4_NL", 896, 896, 128, "N64_M32_2wave", 0.990f},
        {"IQ4NL_N64M32x2_FFNDn7B_M128", "IQ4_NL", 3584, 18944, 128, "N64_M32_2wave", 0.990f},

        // --- IQ4_XS (Qwen3.6 MoE routed down-proj format) ---
        {"IQ4XS_N64M32x2_Attn05_M128", "IQ4_XS", 896, 896, 128, "N64_M32_2wave", 0.985f},
        {"IQ4XS_N64M32x2_FFNDn05_M128", "IQ4_XS", 896, 4864, 128, "N64_M32_2wave", 0.985f},

        // =====================================================================
        // PATH 7: N64 / M64 / 2-wave  (N <= K, M >= ~256 → m_tile=64)
        //   native_vnni_gemm_kernel, N_TILE=64, M_TILE=64, MIN_BLOCKS=2
        //   m_tile selection: min(M gg, 64) — need to check the actual heuristic
        //   7B FFN_Dn at M=256: N=3584 <= K=18944, m_tile = 64 from switch default
        // =====================================================================

        // --- Q4_0 ---
        {"Q4_0_N64M64x2_FFNDn7B_M256", "Q4_0", 3584, 18944, 256, "N64_M64_2wave", 0.990f},
        {"Q4_0_N64M64x2_FFNDn05_M256", "Q4_0", 896, 4864, 256, "N64_M64_2wave", 0.990f},

        // --- IQ4_NL ---
        {"IQ4NL_N64M64x2_FFNDn7B_M256", "IQ4_NL", 3584, 18944, 256, "N64_M64_2wave", 0.990f},

        // --- IQ4_XS (Qwen3.6 MoE routed down-proj format) ---
        {"IQ4XS_N64M64x2_FFNDn05_M256", "IQ4_XS", 896, 4864, 256, "N64_M64_2wave", 0.985f},

        // =====================================================================
        // NEW FORMATS: representative coverage (streaming + cooperative paths)
        //
        // Each new format gets:
        //   - A streaming test (LM_Head proxy, N=20480, K=896)
        //   - A cooperative N128 test (FFN_Up, N=4864, K=896)
        //   - A cooperative N64 test (AttnOut, N=896, K=896)
        // Cosine gates are relaxed for lower-bit formats (more quantization error).
        // =====================================================================

        // --- Q4_1 (asymmetric, Pattern A) ---
        {"Q4_1_Stream_LM05B_M16", "Q4_1", LM_N, 896, 16, "Streaming", 0.990f},
        {"Q4_1_N128M16x3_FFNUp05_M16", "Q4_1", 4864, 896, 16, "N128_M16_3wave", 0.990f},
        {"Q4_1_N64M16x3_Attn05_M32", "Q4_1", 896, 896, 32, "N64_M16_3wave", 0.990f},

        // --- IQ4_XS (Qwen3.6 MoE routed down-proj format) ---
        {"IQ4XS_Qwen36MoEDown_M2", "IQ4_XS", 512, 256, 2, "Qwen36MoE_DownSmallM", 0.985f},
        {"IQ4XS_Stream_LM05B_M16", "IQ4_XS", LM_N, 896, 16, "Streaming", 0.985f},
        {"IQ4XS_N128M16x3_FFNUp05_M16", "IQ4_XS", 4864, 896, 16, "N128_M16_3wave", 0.985f},
        {"IQ4XS_N64M16x3_Attn05_M32", "IQ4_XS", 896, 896, 32, "N64_M16_3wave", 0.985f},

        // --- Q5_0 (5-bit symmetric, Pattern S) ---
        {"Q5_0_Stream_LM05B_M16", "Q5_0", LM_N, 896, 16, "Streaming", 0.990f},
        {"Q5_0_N128M16x3_FFNUp05_M16", "Q5_0", 4864, 896, 16, "N128_M16_3wave", 0.990f},
        {"Q5_0_N64M16x3_Attn05_M32", "Q5_0", 896, 896, 32, "N64_M16_3wave", 0.990f},

        // --- Q5_1 (5-bit asymmetric, Pattern A) ---
        {"Q5_1_Stream_LM05B_M16", "Q5_1", LM_N, 896, 16, "Streaming", 0.990f},
        {"Q5_1_N128M16x3_FFNUp05_M16", "Q5_1", 4864, 896, 16, "N128_M16_3wave", 0.990f},
        {"Q5_1_N64M16x3_Attn05_M32", "Q5_1", 896, 896, 32, "N64_M16_3wave", 0.990f},

        // --- Q6_K (6-bit dual-scale, Pattern D) ---
        {"Q6_K_Stream_LM05B_M16", "Q6_K", LM_N, 896, 16, "Streaming", 0.990f},
        {"Q6_K_N128M16x3_FFNUp05_M16", "Q6_K", 4864, 896, 16, "N128_M16_3wave", 0.990f},
        {"Q6_K_N64M16x3_Attn05_M32", "Q6_K", 896, 896, 32, "N64_M16_3wave", 0.990f},

        // --- Q3_K (3-bit dual-scale, Pattern D) ---
        {"Q3_K_Stream_LM05B_M16", "Q3_K", LM_N, 896, 16, "Streaming", 0.985f},
        {"Q3_K_N128M16x3_FFNUp05_M16", "Q3_K", 4864, 896, 16, "N128_M16_3wave", 0.985f},
        {"Q3_K_N64M16x3_Attn05_M32", "Q3_K", 896, 896, 32, "N64_M16_3wave", 0.985f},

        // --- Q2_K (2-bit dual-scale+asymmetric, Pattern DA) ---
        {"Q2_K_Stream_LM05B_M16", "Q2_K", LM_N, 896, 16, "Streaming", 0.970f},
        {"Q2_K_N128M16x3_FFNUp05_M16", "Q2_K", 4864, 896, 16, "N128_M16_3wave", 0.970f},
        {"Q2_K_N64M16x3_Attn05_M32", "Q2_K", 896, 896, 32, "N64_M16_3wave", 0.970f},

        // --- IQ3_S (3-bit grid+signs, Pattern S) ---
        {"IQ3_S_Stream_LM05B_M16", "IQ3_S", LM_N, 896, 16, "Streaming", 0.985f},
        {"IQ3_S_N128M16x3_FFNUp05_M16", "IQ3_S", 4864, 896, 16, "N128_M16_3wave", 0.985f},
        {"IQ3_S_N64M16x3_Attn05_M32", "IQ3_S", 896, 896, 32, "N64_M16_3wave", 0.985f},

        // --- IQ3_XXS (3-bit grid+signs, Pattern S) ---
        {"IQ3XXS_Stream_LM05B_M16", "IQ3_XXS", LM_N, 896, 16, "Streaming", 0.980f},
        {"IQ3XXS_N128M16x3_FFNUp05_M16", "IQ3_XXS", 4864, 896, 16, "N128_M16_3wave", 0.980f},
        {"IQ3XXS_N64M16x3_Attn05_M32", "IQ3_XXS", 896, 896, 32, "N64_M16_3wave", 0.980f},

        // --- IQ2_S (2-bit grid+signs, Pattern D) ---
        {"IQ2_S_Qwen36MoEGateUp_M2", "IQ2_S", 256, 512, 2, "Qwen36MoE_GateUpSmallM", 0.960f},
        {"IQ2_S_Stream_LM05B_M16", "IQ2_S", LM_N, 896, 16, "Streaming", 0.960f},
        {"IQ2_S_N128M16x3_FFNUp05_M16", "IQ2_S", 4864, 896, 16, "N128_M16_3wave", 0.960f},
        {"IQ2_S_N64M16x3_Attn05_M32", "IQ2_S", 896, 896, 32, "N64_M16_3wave", 0.960f},

        // --- IQ2_XS (2-bit grid+signs, Pattern D) ---
        {"IQ2XS_Stream_LM05B_M16", "IQ2_XS", LM_N, 896, 16, "Streaming", 0.950f},
        {"IQ2XS_N128M16x3_FFNUp05_M16", "IQ2_XS", 4864, 896, 16, "N128_M16_3wave", 0.950f},
        {"IQ2XS_N64M16x3_Attn05_M32", "IQ2_XS", 896, 896, 32, "N64_M16_3wave", 0.950f},

        // --- IQ2_XXS (2-bit grid+signs, Pattern S) ---
        {"IQ2XXS_Stream_LM05B_M16", "IQ2_XXS", LM_N, 896, 16, "Streaming", 0.940f},
        {"IQ2XXS_N128M16x3_FFNUp05_M16", "IQ2_XXS", 4864, 896, 16, "N128_M16_3wave", 0.940f},
        {"IQ2XXS_N64M16x3_Attn05_M32", "IQ2_XXS", 896, 896, 32, "N64_M16_3wave", 0.940f},

        // --- IQ1_S (1-bit grid, Pattern A, embedded scales) ---
        {"IQ1_S_Stream_LM05B_M16", "IQ1_S", LM_N, 896, 16, "Streaming", 0.900f},
        {"IQ1_S_N128M16x3_FFNUp05_M16", "IQ1_S", 4864, 896, 16, "N128_M16_3wave", 0.900f},
        {"IQ1_S_N64M16x3_Attn05_M32", "IQ1_S", 896, 896, 32, "N64_M16_3wave", 0.900f},

        // --- IQ1_M (1-bit grid, Pattern DA, embedded scales) ---
        {"IQ1_M_Stream_LM05B_M16", "IQ1_M", LM_N, 896, 16, "Streaming", 0.900f},
        {"IQ1_M_N128M16x3_FFNUp05_M16", "IQ1_M", 4864, 896, 16, "N128_M16_3wave", 0.900f},
        {"IQ1_M_N64M16x3_Attn05_M32", "IQ1_M", 896, 896, 32, "N64_M16_3wave", 0.900f},
    };

    INSTANTIATE_TEST_SUITE_P(
        AllDispatchPaths,
        NativeVNNIGEMMTest,
        ::testing::ValuesIn(ALL_GEMM_PARAMS),
        paramName);

} // anonymous namespace
