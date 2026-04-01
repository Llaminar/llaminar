/**
 * @file Test__NativeVNNI_GEMV.cpp
 * @brief Integration tests for native-VNNI GEMV kernel — GPU accuracy comparison
 *
 * Tests the complete native-VNNI GEMV pipeline on GPU for all 16 supported formats:
 *   1. Create random quantized weights
 *   2. Pack with packWeightsToROCm() → produces native-VNNI payload/scales
 *   3. Create ROCmQuantisedGemmKernel with workspace
 *   4. Run M=1 GEMV (multiply_tensor) on GPU
 *   5. Compute FP32 reference on CPU: dequantize weights → dense FP32 matmul
 *   6. Compare GPU vs CPU via cosine similarity and max absolute error
 *
 * Accuracy expectations:
 * - Native-VNNI is lossless within each block's quantization (FP16 per-block scales
 *   are preserved). The only numerical error comes from the INT8 activation
 *   quantization step (FP32 input → INT8 → FP32 scale_A).
 * - Expected cosine similarity > 0.99 for all formats.
 * - Lower BPW formats (Q2_K, IQ2) inherently have more quantization noise, so
 *   the comparison is against their own dequantized reference (not FP32 weights).
 *
 * @note Requires ROCm device. Tests are skipped if no GPU is available.
 * @note Run with build_v2_integration.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"
#include "../../../utils/TestTensorFactory.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
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
        {
            max_err = std::max(max_err, std::fabs(a[i] - b[i]));
        }
        return max_err;
    }

    /// CPU FP32 reference GEMV: output[j] = sum_k(input[k] * W_dequant[j][k])
    /// W is stored as [N, K] i.e. row j has K elements.
    void cpuFP32Gemv(const float *input_fp32, // [K]
                     const float *W_dequant,  // [N × K] row-major
                     float *output,           // [N]
                     int N, int K)
    {
        for (int j = 0; j < N; ++j)
        {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
            {
                acc += static_cast<double>(input_fp32[k]) *
                       static_cast<double>(W_dequant[j * K + k]);
            }
            output[j] = static_cast<float>(acc);
        }
    }

    // =============================================================================
    // Format descriptors
    // =============================================================================

    struct GEMVFormatSpec
    {
        std::string name;
        bool is_superblock; ///< K must be multiple of 256
        float min_cosine;   ///< Minimum expected cosine similarity

        std::function<std::unique_ptr<TensorBase>(size_t N, size_t K)> create;
    };

    // Cosine thresholds:
    // - Higher BPW formats have less quantization noise → higher cosine vs. dequantized reference
    // - The comparison is native-VNNI GPU vs. FP32-dequantized CPU GEMV with the SAME quantized weights
    // - Main error source: INT8 activation quantization (FP32→INT8→scale_A)
    // - IQ2/IQ3 grid formats have additional decode approximation but should still be > 0.98

    static const std::vector<GEMVFormatSpec> ALL_GEMV_FORMATS = {
        // Tier 1: Simple 32-element blocks
        {"Q4_0", false, 0.990f,
         [](size_t N, size_t K)
         { return TestTensorFactory::createQ4_0Random({N, K}); }},
        {"IQ4_NL", false, 0.990f,
         [](size_t N, size_t K)
         { return TestTensorFactory::createIQ4_NLRandom({N, K}); }},
        {"Q4_1", false, 0.990f,
         [](size_t N, size_t K)
         { return TestTensorFactory::createQ4_1Random({N, K}); }},
        {"Q5_0", false, 0.990f,
         [](size_t N, size_t K)
         { return TestTensorFactory::createQ5_0Random({N, K}); }},
        {"Q5_1", false, 0.990f,
         [](size_t N, size_t K)
         { return TestTensorFactory::createQ5_1Random({N, K}); }},

        // Tier 1 super-block
        {"IQ4_XS", true, 0.985f,
         [](size_t N, size_t K)
         { return TestTensorFactory::createIQ4_XSRandom({N, K}); }},

        // Tier 2: K-quant super-blocks
        {"Q4_K", true, 0.985f,
         [](size_t N, size_t K)
         { return TestTensorFactory::createQ4_KRandom({N, K}); }},
        {"Q5_K", true, 0.985f,
         [](size_t N, size_t K)
         { return TestTensorFactory::createQ5_KRandom({N, K}); }},
        {"Q6_K", true, 0.990f,
         [](size_t N, size_t K)
         { return TestTensorFactory::createQ6_KRandom({N, K}); }},
        {"Q3_K", true, 0.980f,
         [](size_t N, size_t K)
         { return TestTensorFactory::createQ3_KRandom({N, K}); }},
        {"Q2_K", true, 0.970f,
         [](size_t N, size_t K)
         { return TestTensorFactory::createQ2_KRandom({N, K}); }},

        // Tier 3: IQ grid-index super-blocks
        {"IQ3_S", true, 0.975f,
         [](size_t N, size_t K)
         { return TestTensorFactory::createIQ3_SRandom({N, K}); }},
        {"IQ3_XXS", true, 0.970f,
         [](size_t N, size_t K)
         { return TestTensorFactory::createIQ3_XXSRandom({N, K}); }},
        {"IQ2_S", true, 0.960f,
         [](size_t N, size_t K)
         { return TestTensorFactory::createIQ2_SRandom({N, K}); }},
        {"IQ2_XS", true, 0.960f,
         [](size_t N, size_t K)
         { return TestTensorFactory::createIQ2_XSRandom({N, K}); }},
        {"IQ2_XXS", true, 0.950f,
         [](size_t N, size_t K)
         { return TestTensorFactory::createIQ2_XXSRandom({N, K}); }},

        // Tier 4: IQ1 ultra-low-bit grid-index super-blocks
        {"IQ1_S", true, 0.930f,
         [](size_t N, size_t K)
         { return TestTensorFactory::createIQ1_SRandom({N, K}); }},
        {"IQ1_M", true, 0.930f,
         [](size_t N, size_t K)
         { return TestTensorFactory::createIQ1_MRandom({N, K}); }},
    };

    // =============================================================================
    // Test fixture
    // =============================================================================

    class NativeVNNIGEMVTest : public ::testing::Test
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
                LOG_INFO("[NativeVNNI_GEMV] ROCm device: " << props.name
                                                           << " (" << props.gcnArchName << ")");
            }
#else
            has_rocm_device_ = false;
#endif
        }

        bool has_rocm_device_ = false;

#ifdef HAVE_ROCM
        std::unique_ptr<DeviceWorkspaceManager> workspace_;

        bool setupWorkspace(ROCmQuantisedGemmKernel &kernel, int M, int N, int K)
        {
            auto reqs = kernel.getWorkspaceRequirements(M, N, K);
            workspace_ = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(0), 64 * 1024 * 1024); // 64MB
            if (!workspace_->allocate(reqs))
            {
                LOG_ERROR("Failed to allocate workspace");
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

        /// Upload input/output to GPU, run multiply_tensor, mark output device-dirty.
        /// This ensures the M=1 GEMV fast path is used (requires use_gpu_path=true).
        bool runGemvOnGpu(ROCmQuantisedGemmKernel &kernel,
                          TensorBase *input, TensorBase *output,
                          int M, int N, int K)
        {
            const auto device = DeviceId::rocm(0);
            if (!input->ensureOnDevice(device) || !output->ensureOnDevice(device))
                return false;
            bool ok = kernel.multiply_tensor(input, output, M, N, K);
            if (ok)
                output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
            return ok;
        }
#endif
    };

    // =============================================================================
    // Parameterized test: GEMV accuracy for all formats
    // =============================================================================

    class NativeVNNIGEMVFormatTest
        : public NativeVNNIGEMVTest,
          public ::testing::WithParamInterface<GEMVFormatSpec>
    {
    };

    TEST_P(NativeVNNIGEMVFormatTest, M1_GEMV_MatchesFP32Reference)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_rocm_device_)
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        const auto &fmt = GetParam();
        const int M = 1;
        // Moderate dimensions — large enough to exercise the kernel,
        // small enough for fast CI. K must be multiple of 256 for super-block.
        const int N = 128;
        const int K = fmt.is_superblock ? 256 : 128;

        // 1. Create quantized weights
        auto weights = fmt.create(static_cast<size_t>(N), static_cast<size_t>(K));
        ASSERT_NE(weights, nullptr) << "Failed to create " << fmt.name << " tensor";

        // 2. Dequantize weights to FP32 for CPU reference
        std::vector<float> W_fp32(static_cast<size_t>(N) * K);
        weights->to_fp32(W_fp32.data());

        // 3. Pack for ROCm
        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed))
            << "packWeightsToROCm failed for " << fmt.name;
        ASSERT_FALSE(packed.native_vnni_payload.empty())
            << fmt.name << ": native-VNNI payload was not populated";

        // 4. Create kernel + workspace
        ROCmQuantisedGemmKernel kernel(&packed, 0);
        ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

        // 5. Create random FP32 input [1, K]
        auto input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)});
        auto output_gpu = TestTensorFactory::createFP32(
            {static_cast<size_t>(M), static_cast<size_t>(N)});

        // 6. Run GPU GEMV (uses native-VNNI for M=1 with alpha=1, beta=0)
        ASSERT_TRUE(runGemvOnGpu(kernel, input.get(), output_gpu.get(), M, N, K))
            << fmt.name << ": multiply_tensor failed";

        // 7. CPU FP32 reference
        std::vector<float> ref(static_cast<size_t>(N));
        cpuFP32Gemv(input->data(), W_fp32.data(), ref.data(), N, K);

        // 8. Compare
        const float *gpu_ptr = output_gpu->data();
        float cos = cosineSimilarity(gpu_ptr, ref.data(), static_cast<size_t>(N));
        float mae = maxAbsError(gpu_ptr, ref.data(), static_cast<size_t>(N));

        LOG_INFO("[NativeVNNI_GEMV] " << fmt.name
                                      << " N=" << N << " K=" << K
                                      << " cosine=" << cos
                                      << " max_abs_error=" << mae);

        EXPECT_GT(cos, fmt.min_cosine)
            << fmt.name << ": cosine similarity " << cos
            << " below threshold " << fmt.min_cosine;

        // NaN/Inf check
        for (int j = 0; j < N; ++j)
        {
            EXPECT_FALSE(std::isnan(gpu_ptr[j]))
                << fmt.name << ": output[" << j << "] is NaN";
            EXPECT_FALSE(std::isinf(gpu_ptr[j]))
                << fmt.name << ": output[" << j << "] is Inf";
        }

        cleanupWorkspace(kernel);
#endif
    }

    INSTANTIATE_TEST_SUITE_P(
        AllFormats,
        NativeVNNIGEMVFormatTest,
        ::testing::ValuesIn(ALL_GEMV_FORMATS),
        [](const ::testing::TestParamInfo<GEMVFormatSpec> &info)
        {
            return info.param.name;
        });

    // =============================================================================
    // Larger matrix test (representative model dimension)
    // =============================================================================

#ifdef HAVE_ROCM

    /**
     * @test Verify native-VNNI GEMV accuracy at Qwen2.5-0.5B model dimensions.
     *
     * Tests Q4_0 (representative simple format) with:
     *   N=896 (d_model), K=896 — self-attention projection shape
     */
    TEST_F(NativeVNNIGEMVTest, Q4_0_ModelDimension_896x896)
    {
        if (!has_rocm_device_)
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        const int M = 1, N = 896, K = 896;

        auto weights = TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(N), static_cast<size_t>(K)});

        std::vector<float> W_fp32(static_cast<size_t>(N) * K);
        weights->to_fp32(W_fp32.data());

        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

        ROCmQuantisedGemmKernel kernel(&packed, 0);
        ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

        auto input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)});
        auto output_gpu = TestTensorFactory::createFP32(
            {static_cast<size_t>(M), static_cast<size_t>(N)});

        ASSERT_TRUE(runGemvOnGpu(kernel, input.get(), output_gpu.get(), M, N, K));

        std::vector<float> ref(static_cast<size_t>(N));
        cpuFP32Gemv(input->data(), W_fp32.data(), ref.data(), N, K);

        const float *gpu_ptr = output_gpu->data();
        float cos = cosineSimilarity(gpu_ptr, ref.data(), static_cast<size_t>(N));
        LOG_INFO("[NativeVNNI_GEMV] Q4_0 model-dim 896×896 cosine=" << cos);
        EXPECT_GT(cos, 0.99f);

        cleanupWorkspace(kernel);
    }

    /**
     * @test Verify native-VNNI GEMV for Q6_K (dual-scale format) at model dimensions.
     *
     * Q6_K is the most complex Tier 2 format with dual-scale accumulation.
     * Uses K=3584 (Qwen2.5-7B d_model) to test large super-block processing.
     */
    TEST_F(NativeVNNIGEMVTest, Q6_K_ModelDimension_3584x3584)
    {
        if (!has_rocm_device_)
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        const int M = 1, N = 3584, K = 3584;

        auto weights = TestTensorFactory::createQ6_KRandom(
            {static_cast<size_t>(N), static_cast<size_t>(K)});

        std::vector<float> W_fp32(static_cast<size_t>(N) * K);
        weights->to_fp32(W_fp32.data());

        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

        ROCmQuantisedGemmKernel kernel(&packed, 0);
        ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

        auto input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)});
        auto output_gpu = TestTensorFactory::createFP32(
            {static_cast<size_t>(M), static_cast<size_t>(N)});

        ASSERT_TRUE(runGemvOnGpu(kernel, input.get(), output_gpu.get(), M, N, K));

        std::vector<float> ref(static_cast<size_t>(N));
        cpuFP32Gemv(input->data(), W_fp32.data(), ref.data(), N, K);

        const float *gpu_ptr = output_gpu->data();
        float cos = cosineSimilarity(gpu_ptr, ref.data(), static_cast<size_t>(N));
        LOG_INFO("[NativeVNNI_GEMV] Q6_K model-dim 3584×3584 cosine=" << cos);
        EXPECT_GT(cos, 0.99f);

        cleanupWorkspace(kernel);
    }

    /**
     * @test Verify IQ3_S (Tier 3 grid format) at model-like dimensions.
     *
     * IQ3_S uses 512-entry uint32_t grid LUT and 9-bit grid indices.
     * Tests K=3584 to exercise multiple super-blocks.
     */
    TEST_F(NativeVNNIGEMVTest, IQ3_S_ModelDimension_3584x3584)
    {
        if (!has_rocm_device_)
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        const int M = 1, N = 3584, K = 3584;

        auto weights = TestTensorFactory::createIQ3_SRandom(
            {static_cast<size_t>(N), static_cast<size_t>(K)});

        std::vector<float> W_fp32(static_cast<size_t>(N) * K);
        weights->to_fp32(W_fp32.data());

        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

        ROCmQuantisedGemmKernel kernel(&packed, 0);
        ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

        auto input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)});
        auto output_gpu = TestTensorFactory::createFP32(
            {static_cast<size_t>(M), static_cast<size_t>(N)});

        ASSERT_TRUE(runGemvOnGpu(kernel, input.get(), output_gpu.get(), M, N, K));

        std::vector<float> ref(static_cast<size_t>(N));
        cpuFP32Gemv(input->data(), W_fp32.data(), ref.data(), N, K);

        const float *gpu_ptr = output_gpu->data();
        float cos = cosineSimilarity(gpu_ptr, ref.data(), static_cast<size_t>(N));
        LOG_INFO("[NativeVNNI_GEMV] IQ3_S model-dim 3584×3584 cosine=" << cos);
        EXPECT_GT(cos, 0.975f);

        cleanupWorkspace(kernel);
    }

    /**
     * @test Verify IQ2_XXS (extreme low-bit Tier 3 format) at model-like dimensions.
     *
     * IQ2_XXS is 2.1 bpw — the lowest BPW format currently supported.
     * This is the format most likely to show bugs due to its complex decode.
     */
    TEST_F(NativeVNNIGEMVTest, IQ2_XXS_ModelDimension_3584x3584)
    {
        if (!has_rocm_device_)
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        const int M = 1, N = 3584, K = 3584;

        auto weights = TestTensorFactory::createIQ2_XXSRandom(
            {static_cast<size_t>(N), static_cast<size_t>(K)});

        std::vector<float> W_fp32(static_cast<size_t>(N) * K);
        weights->to_fp32(W_fp32.data());

        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));

        ROCmQuantisedGemmKernel kernel(&packed, 0);
        ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

        auto input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)});
        auto output_gpu = TestTensorFactory::createFP32(
            {static_cast<size_t>(M), static_cast<size_t>(N)});

        ASSERT_TRUE(runGemvOnGpu(kernel, input.get(), output_gpu.get(), M, N, K));

        std::vector<float> ref(static_cast<size_t>(N));
        cpuFP32Gemv(input->data(), W_fp32.data(), ref.data(), N, K);

        const float *gpu_ptr = output_gpu->data();
        float cos = cosineSimilarity(gpu_ptr, ref.data(), static_cast<size_t>(N));
        LOG_INFO("[NativeVNNI_GEMV] IQ2_XXS model-dim 3584×3584 cosine=" << cos);
        EXPECT_GT(cos, 0.95f);

        cleanupWorkspace(kernel);
    }

    // =============================================================================
    // Summary test: run all 16 formats at moderate dimensions and print table
    // =============================================================================

    /**
     * @test Comprehensive accuracy sweep: all 16 native-VNNI formats in one test
     *
     * Runs each format through M=1 GEMV with N=128, K varies by format.
     * Prints a summary table at the end with cosine similarity and max abs error.
     */
    TEST_F(NativeVNNIGEMVTest, AccuracySweep_All16Formats)
    {
        if (!has_rocm_device_)
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        struct Result
        {
            std::string name;
            float cosine;
            float max_abs_err;
            bool passed;
        };

        std::vector<Result> results;
        const int M = 1;
        const int N = 128;

        for (const auto &fmt : ALL_GEMV_FORMATS)
        {
            const int K = fmt.is_superblock ? 256 : 128;

            auto weights = fmt.create(static_cast<size_t>(N), static_cast<size_t>(K));
            if (!weights)
            {
                results.push_back({fmt.name, 0.0f, 999.0f, false});
                continue;
            }

            std::vector<float> W_fp32(static_cast<size_t>(N) * K);
            weights->to_fp32(W_fp32.data());

            ROCmPackedWeights packed;
            if (!packWeightsToROCm(weights.get(), packed) ||
                packed.native_vnni_payload.empty())
            {
                results.push_back({fmt.name, 0.0f, 999.0f, false});
                continue;
            }

            ROCmQuantisedGemmKernel kernel(&packed, 0);
            if (!setupWorkspace(kernel, M, N, K))
            {
                results.push_back({fmt.name, 0.0f, 999.0f, false});
                continue;
            }

            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(K)});
            auto output_gpu = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(N)});

            bool ok = runGemvOnGpu(kernel, input.get(), output_gpu.get(), M, N, K);
            if (!ok)
            {
                cleanupWorkspace(kernel);
                results.push_back({fmt.name, 0.0f, 999.0f, false});
                continue;
            }

            std::vector<float> ref(static_cast<size_t>(N));
            cpuFP32Gemv(input->data(), W_fp32.data(), ref.data(), N, K);

            const float *gpu_ptr = output_gpu->data();
            float cos = cosineSimilarity(gpu_ptr, ref.data(), static_cast<size_t>(N));
            float mae = maxAbsError(gpu_ptr, ref.data(), static_cast<size_t>(N));

            cleanupWorkspace(kernel);
            results.push_back({fmt.name, cos, mae, cos > fmt.min_cosine});
        }

        // Print summary table
        LOG_INFO("╔════════════════════════════════════════════════════════════╗");
        LOG_INFO("║     NATIVE-VNNI GEMV ACCURACY SWEEP (M=1, N=128)        ║");
        LOG_INFO("╠═══════════╦══════════════╦══════════════╦════════════════╣");
        LOG_INFO("║  Format   ║    Cosine    ║  Max AbsErr  ║    Status     ║");
        LOG_INFO("╠═══════════╬══════════════╬══════════════╬════════════════╣");

        int pass_count = 0;
        for (const auto &r : results)
        {
            char line[256];
            snprintf(line, sizeof(line),
                     "║ %-9s ║ %10.6f   ║ %10.6f   ║     %s      ║",
                     r.name.c_str(), r.cosine, r.max_abs_err,
                     r.passed ? " ✓ " : " ✗ ");
            LOG_INFO(line);
            if (r.passed)
                pass_count++;
        }

        LOG_INFO("╚═══════════╩══════════════╩══════════════╩════════════════╝");
        LOG_INFO("[NativeVNNI_GEMV] Passed " << pass_count << "/" << results.size());

        // Soft assertion: log failures but let individual parameterized tests be the hard gate
        for (const auto &r : results)
        {
            EXPECT_TRUE(r.passed)
                << r.name << ": cosine=" << r.cosine << " max_abs_err=" << r.max_abs_err;
        }
    }

#endif // HAVE_ROCM

} // anonymous namespace
