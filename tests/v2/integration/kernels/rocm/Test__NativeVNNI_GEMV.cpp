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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "kernels/rocm/ROCmWeightPacker.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include "utils/Logger.h"
#include "../../../utils/TestTensorFactory.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>

extern "C" void rocmGemv_native_vnni_set_tuning_overrides(int kb, int target_waves_per_cu);
extern "C" void rocmGemv_native_vnni_reset_tuning_overrides();
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

    float relativeL2Error(const float *a, const float *b, size_t n)
    {
        double sum_sq_diff = 0.0;
        double sum_sq_ref = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            sum_sq_diff += diff * diff;
            sum_sq_ref += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        return static_cast<float>(std::sqrt(sum_sq_diff / std::max(sum_sq_ref, 1e-30)));
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

    std::unique_ptr<Q4_KTensor> createQ4KWithNonzeroMins(size_t N, size_t K)
    {
        constexpr size_t BLOCK_SIZE = Q4_KBlock::BLOCK_SIZE;
        const size_t blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
        std::vector<uint8_t> raw_data(N * blocks_per_row * sizeof(Q4_KBlock));
        auto *blocks = reinterpret_cast<Q4_KBlock *>(raw_data.data());

        std::mt19937 rng(0x4B36);
        std::uniform_int_distribution<int> nibble_dist(0, 15);

        for (size_t row = 0; row < N; ++row)
        {
            for (size_t sb = 0; sb < blocks_per_row; ++sb)
            {
                Q4_KBlock &block = blocks[row * blocks_per_row + sb];
                const float scale = 0.0025f + 0.00005f * static_cast<float>((row + sb) % 17);
                const float min_scale = 0.00045f + 0.00003f * static_cast<float>((row * 3 + sb) % 11);
                block.d = fp32_to_fp16(scale);
                block.dmin = fp32_to_fp16(min_scale);

                // 0x44 decodes to nonzero scale and min selectors in both halves of
                // the Q4_K superblock, giving this regression real asymmetric
                // min-correction work rather than the friendlier zero-min fixture.
                std::memset(block.scales, 0x44, sizeof(block.scales));

                for (uint8_t &q : block.qs)
                {
                    const int lo = nibble_dist(rng);
                    const int hi = nibble_dist(rng);
                    q = static_cast<uint8_t>((hi << 4) | lo);
                }
            }
        }

        return std::make_unique<Q4_KTensor>(std::vector<size_t>{N, K}, raw_data);
    }

    void cpuQ4LikeNativeVNNIGemvFromPacked(const int8_t *a_int8,
                                          const float *a_scales_blockwise,
                                          const std::vector<uint8_t> &payload,
                                          const std::vector<uint16_t> &scales,
                                          const std::vector<uint16_t> &mins,
                                          float *output,
                                          int N,
                                          int K)
    {
        constexpr int BLOCK_SIZE = 32;
        constexpr int PAYLOAD_BYTES = 16;
        const int blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int n = 0; n < N; ++n)
        {
            float acc = 0.0f;
            for (int b = 0; b < blocks_per_row; ++b)
            {
                const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) +
                                      static_cast<size_t>(n);
                const uint8_t *block_payload = payload.data() + linear * PAYLOAD_BYTES;
                const float block_scale = fp16_to_fp32(scales[linear]);
                const float block_min = fp16_to_fp32(mins[linear]);
                const float activation_scale = a_scales_blockwise[b];
                const int8_t *a_block = a_int8 + static_cast<size_t>(b) * BLOCK_SIZE;

                int32_t dot = 0;
                int32_t sum_a = 0;
                for (int i = 0; i < 16; ++i)
                {
                    const int32_t a = static_cast<int32_t>(a_block[i]);
                    const int32_t q = static_cast<int32_t>(block_payload[i] & 0x0F);
                    dot += a * q;
                    sum_a += a;
                }
                for (int i = 0; i < 16; ++i)
                {
                    const int32_t a = static_cast<int32_t>(a_block[i + 16]);
                    const int32_t q = static_cast<int32_t>(block_payload[i] >> 4);
                    dot += a * q;
                    sum_a += a;
                }

                acc += (static_cast<float>(dot) * block_scale +
                        static_cast<float>(sum_a) * block_min) *
                       activation_scale;
            }
            output[n] = acc;
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
        bool direct_native_pack = false; ///< Use packNativeVNNI even if the default packer prefers INT8-VNNI.
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
        {"Q8_0", false, 0.990f,
         [](size_t N, size_t K)
         { return TestTensorFactory::createQ8_0Random({N, K}); }, true},
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
        bool packForNativeVNNIGemv(const GEMVFormatSpec &fmt,
                                   const TensorBase *weights,
                                   ROCmPackedWeights &packed)
        {
            if (fmt.direct_native_pack)
                return packNativeVNNI(weights, packed);
            return packWeightsToROCm(weights, packed);
        }

        class NativeVNNITuningOverrideGuard
        {
        public:
            explicit NativeVNNITuningOverrideGuard(int kb)
            {
                rocmGemv_native_vnni_set_tuning_overrides(kb, -1);
            }

            ~NativeVNNITuningOverrideGuard()
            {
                rocmGemv_native_vnni_reset_tuning_overrides();
            }
        };

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
        ASSERT_TRUE(packForNativeVNNIGemv(fmt, weights.get(), packed))
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

    TEST_F(NativeVNNIGEMVTest, Q4_0_Decode_AlphaBetaAccumulatesExistingOutput)
    {
        if (!has_rocm_device_)
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        const int M = 1, N = 896, K = 896;
        const float alpha = 0.375f;
        const float beta = 1.0f;

        auto weights = TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(N), static_cast<size_t>(K)});

        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
        ASSERT_FALSE(packed.native_vnni_payload.empty());

        ROCmQuantisedGemmKernel kernel(&packed, 0);
        ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

        auto input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)});
        auto output_base = TestTensorFactory::createFP32(
            {static_cast<size_t>(M), static_cast<size_t>(N)});
        auto output_accum = TestTensorFactory::createFP32(
            {static_cast<size_t>(M), static_cast<size_t>(N)});

        std::vector<float> initial(static_cast<size_t>(N));
        for (int col = 0; col < N; ++col)
        {
            initial[col] = 0.03125f * static_cast<float>((col % 23) - 11);
            output_accum->mutable_data()[col] = initial[col];
        }

        const auto device = DeviceId::rocm(0);
        ASSERT_TRUE(input->ensureOnDevice(device));
        ASSERT_TRUE(output_base->allocateOnDevice(device));
        ASSERT_TRUE(output_accum->ensureOnDevice(device));

        ASSERT_TRUE(kernel.multiply_tensor(input.get(), output_base.get(), M, N, K));
        ASSERT_TRUE(kernel.multiply_tensor(input.get(), output_accum.get(), M, N, K,
                                           true, alpha, beta));
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        output_base->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        output_accum->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

        const float *base = output_base->data();
        const float *accum = output_accum->data();
        std::vector<float> expected(static_cast<size_t>(N));
        float max_abs_diff = 0.0f;
        for (int col = 0; col < N; ++col)
        {
            expected[col] = alpha * base[col] + beta * initial[col];
            max_abs_diff = std::max(max_abs_diff, std::fabs(accum[col] - expected[col]));
        }

        const float cos = cosineSimilarity(accum, expected.data(), static_cast<size_t>(N));
        LOG_INFO("[NativeVNNI_GEMV] Q4_0 decode alpha/beta accumulation cosine=" << cos
                                                                                 << " max_abs_diff=" << max_abs_diff);
        EXPECT_GT(cos, 0.99999f);
        EXPECT_LT(max_abs_diff, 2e-3f);

        cleanupWorkspace(kernel);
    }

    TEST_F(NativeVNNIGEMVTest, Q4_0_FusedSwiGLUDown_Decode_AlphaBetaAccumulatesExistingOutput)
    {
        if (!has_rocm_device_)
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        const int M = 1, N = 256, K = 896;
        const float alpha = 0.625f;
        const float beta = 1.0f;

        auto weights = TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(N), static_cast<size_t>(K)});

        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
        ASSERT_FALSE(packed.native_vnni_payload.empty());

        ROCmQuantisedGemmKernel kernel(&packed, 0);
        ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

        auto gate = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)}, -0.75f, 0.75f, 301);
        auto up = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)}, -0.75f, 0.75f, 302);
        auto output_base = TestTensorFactory::createFP32(
            {static_cast<size_t>(M), static_cast<size_t>(N)});
        auto output_accum = TestTensorFactory::createFP32(
            {static_cast<size_t>(M), static_cast<size_t>(N)});

        std::vector<float> initial(static_cast<size_t>(N));
        for (int col = 0; col < N; ++col)
        {
            initial[col] = 0.0625f * static_cast<float>((col % 17) - 8);
            output_accum->mutable_data()[col] = initial[col];
        }

        const auto device = DeviceId::rocm(0);
        ASSERT_TRUE(gate->ensureOnDevice(device));
        ASSERT_TRUE(up->ensureOnDevice(device));
        ASSERT_TRUE(output_base->allocateOnDevice(device));
        ASSERT_TRUE(output_accum->ensureOnDevice(device));

        ASSERT_TRUE(kernel.multiply_tensor_with_fused_swiglu(
            gate.get(), up.get(), output_base.get(), M, N, K));
        ASSERT_TRUE(kernel.multiply_tensor_with_fused_swiglu(
            gate.get(), up.get(), output_accum.get(), M, N, K, alpha, beta));
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        output_base->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        output_accum->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

        const float *base = output_base->data();
        const float *accum = output_accum->data();
        std::vector<float> expected(static_cast<size_t>(N));
        float max_abs_diff = 0.0f;
        for (int col = 0; col < N; ++col)
        {
            expected[col] = alpha * base[col] + beta * initial[col];
            max_abs_diff = std::max(max_abs_diff, std::fabs(accum[col] - expected[col]));
        }

        const float cos = cosineSimilarity(accum, expected.data(), static_cast<size_t>(N));
        LOG_INFO("[NativeVNNI_GEMV] Q4_0 fused SwiGLU/down decode alpha/beta accumulation cosine=" << cos
                                                                                                   << " max_abs_diff=" << max_abs_diff);
        EXPECT_GT(cos, 0.99999f);
        EXPECT_LT(max_abs_diff, 2e-3f);

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

    TEST_F(NativeVNNIGEMVTest, Q4_K_Qwen36GDNTimeProjection_MatchesPackedNativeContract)
    {
        if (!has_rocm_device_)
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        const int M = 1;
        const int N = 1024;
        const int K = 5120;
        const int blocks_per_row = K / 32;

        auto weights = createQ4KWithNonzeroMins(
            static_cast<size_t>(N), static_cast<size_t>(K));
        ASSERT_NE(weights, nullptr);

        std::vector<float> W_fp32(static_cast<size_t>(N) * K);
        weights->to_fp32(W_fp32.data());

        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
        ASSERT_FALSE(packed.native_vnni_payload.empty());
        ASSERT_FALSE(packed.native_vnni_scales.empty());
        ASSERT_FALSE(packed.native_vnni_mins.empty());
        ASSERT_EQ(packed.native_vnni_codebook_id, 5)
            << "Q4_K must reuse the Q4_1 native-VNNI codebook.";
        const std::vector<uint8_t> host_payload = packed.native_vnni_payload;
        const std::vector<uint16_t> host_scales = packed.native_vnni_scales;
        const std::vector<uint16_t> host_mins = packed.native_vnni_mins;

        ROCmQuantisedGemmKernel kernel(&packed, 0);
        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
        ASSERT_NE(stream, nullptr);
        kernel.setGPUStream(stream);
        ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

        auto input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)}, -0.75f, 0.75f, 3605);
        auto output_gpu = TestTensorFactory::createFP32(
            {static_cast<size_t>(M), static_cast<size_t>(N)});

        ASSERT_TRUE(runGemvOnGpu(kernel, input.get(), output_gpu.get(), M, N, K));
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        const void *d_quant_a = workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A);
        const void *d_scales_a = workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A_BLOCKWISE);
        ASSERT_NE(d_quant_a, nullptr);
        ASSERT_NE(d_scales_a, nullptr);

        std::vector<int8_t> quant_a(static_cast<size_t>(K));
        std::vector<float> scales_a(static_cast<size_t>(blocks_per_row));
        ASSERT_EQ(hipMemcpyAsync(quant_a.data(), d_quant_a,
                                 quant_a.size() * sizeof(int8_t),
                                 hipMemcpyDeviceToHost, stream),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(scales_a.data(), d_scales_a,
                                 scales_a.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream),
                  hipSuccess);

        auto *output_fp32 = dynamic_cast<FP32Tensor *>(output_gpu.get());
        ASSERT_NE(output_fp32, nullptr);
        const void *d_output = output_fp32->gpu_data_ptr();
        ASSERT_NE(d_output, nullptr);
        std::vector<float> gpu_output(static_cast<size_t>(N));
        ASSERT_EQ(hipMemcpyAsync(gpu_output.data(), d_output,
                                 gpu_output.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream),
                  hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        std::vector<float> native_contract_ref(static_cast<size_t>(N));
        cpuQ4LikeNativeVNNIGemvFromPacked(
            quant_a.data(), scales_a.data(), host_payload, host_scales, host_mins,
            native_contract_ref.data(), N, K);

        const float native_cos = cosineSimilarity(
            gpu_output.data(), native_contract_ref.data(), static_cast<size_t>(N));
        const float native_rel_l2 = relativeL2Error(
            gpu_output.data(), native_contract_ref.data(), static_cast<size_t>(N));
        const float native_max_abs = maxAbsError(
            gpu_output.data(), native_contract_ref.data(), static_cast<size_t>(N));

        std::vector<float> fp32_ref(static_cast<size_t>(N));
        cpuFP32Gemv(input->data(), W_fp32.data(), fp32_ref.data(), N, K);
        const float fp32_cos = cosineSimilarity(
            gpu_output.data(), fp32_ref.data(), static_cast<size_t>(N));

        LOG_INFO("[NativeVNNI_GEMV] Q4_K Qwen3.6 GDN time exact-contract"
                 << " native_cos=" << native_cos
                 << " native_rel_l2=" << native_rel_l2
                 << " native_max_abs=" << native_max_abs
                 << " fp32_cos=" << fp32_cos);

        EXPECT_GT(native_cos, 0.999999f);
        EXPECT_LT(native_rel_l2, 2e-5f);
        EXPECT_LT(native_max_abs, 2e-3f);
        EXPECT_GT(fp32_cos, 0.9998f)
            << "The full-FP32 reference may be lower than the packed native "
               "contract because this path intentionally quantizes activations.";

        cleanupWorkspace(kernel);
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
    }

    /**
     * @test Native-VNNI GEMV is bitwise stable across repeated runs.
     *
     * Guards against the CUDA-style failure mode where split-K atomics or
     * shared scratch make repeated runs diverge. This shape is large enough
     * to exercise the GEMV scatter+reduce path (KB > 1) on ROCm.
     */
    TEST_F(NativeVNNIGEMVTest, Q4_0_RepeatedRuns_AreBitwiseStable_OnScatterReduceShape)
    {
        if (!has_rocm_device_)
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        const int M = 1;
        const int N = 3584;
        const int K = 3584;
        constexpr int kRepeatRuns = 5;

        auto weights = TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(N), static_cast<size_t>(K)});
        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
        ASSERT_FALSE(packed.native_vnni_payload.empty());

        ROCmQuantisedGemmKernel kernel(&packed, 0);
        ASSERT_TRUE(setupWorkspace(kernel, M, N, K));

        auto input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)});
        auto output_gpu = TestTensorFactory::createFP32(
            {static_cast<size_t>(M), static_cast<size_t>(N)});

        std::vector<float> reference;
        for (int run = 0; run < kRepeatRuns; ++run)
        {
            ASSERT_TRUE(runGemvOnGpu(kernel, input.get(), output_gpu.get(), M, N, K))
                << "run=" << run;

            const float *gpu_ptr = output_gpu->data();
            std::vector<float> snapshot(gpu_ptr, gpu_ptr + static_cast<size_t>(N));

            if (run == 0)
            {
                reference = std::move(snapshot);
                continue;
            }

            const size_t mismatch = firstBitwiseMismatchIndex(reference, snapshot);
            EXPECT_EQ(mismatch, reference.size())
                << "run=" << run
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
            if (!packForNativeVNNIGemv(fmt, weights.get(), packed) ||
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

    TEST_F(NativeVNNIGEMVTest, SplitK_KB8MatchesKB1_AllNativeCodebooks)
    {
        if (!has_rocm_device_)
        {
            GTEST_SKIP() << "No ROCm device available";
        }

        const int M = 1;
        const int N = 256;
        const int K = 1024;

        for (const auto &fmt : ALL_GEMV_FORMATS)
        {
            auto weights = fmt.create(static_cast<size_t>(N), static_cast<size_t>(K));
            ASSERT_NE(weights, nullptr) << fmt.name << ": failed to create weights";

            ROCmPackedWeights packed;
            ASSERT_TRUE(packForNativeVNNIGemv(fmt, weights.get(), packed))
                << fmt.name << ": native-VNNI packing failed";
            ASSERT_FALSE(packed.native_vnni_payload.empty())
                << fmt.name << ": native-VNNI payload was not populated";

            ROCmQuantisedGemmKernel kernel(&packed, 0);
            ASSERT_TRUE(setupWorkspace(kernel, M, N, K))
                << fmt.name << ": workspace allocation failed";

            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(K)}, -0.75f, 0.75f, 1701);
            auto output_kb1 = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(N)});
            auto output_kb8 = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(N)});

            {
                NativeVNNITuningOverrideGuard guard(1);
                ASSERT_TRUE(runGemvOnGpu(kernel, input.get(), output_kb1.get(), M, N, K))
                    << fmt.name << ": KB=1 native-VNNI GEMV failed";
            }
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

            {
                NativeVNNITuningOverrideGuard guard(8);
                ASSERT_TRUE(runGemvOnGpu(kernel, input.get(), output_kb8.get(), M, N, K))
                    << fmt.name << ": KB=8 native-VNNI GEMV failed";
            }
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

            const float *direct = output_kb1->data();
            const float *split = output_kb8->data();
            const float cosine = cosineSimilarity(split, direct, static_cast<size_t>(N));
            const float rel_l2 = relativeL2Error(split, direct, static_cast<size_t>(N));
            const float max_abs = maxAbsError(split, direct, static_cast<size_t>(N));

            LOG_INFO("[NativeVNNI_GEMV] " << fmt.name
                                           << " KB=8 vs KB=1 cosine=" << cosine
                                           << " rel_l2=" << rel_l2
                                           << " max_abs=" << max_abs);

            EXPECT_GT(cosine, 0.99999f) << fmt.name << ": split-K changed decode direction";
            EXPECT_LT(rel_l2, 5e-4f) << fmt.name << ": split-K relative error too large";
            EXPECT_LT(max_abs, 5e-2f) << fmt.name << ": split-K max absolute error too large";

            cleanupWorkspace(kernel);
        }
    }

#endif // HAVE_ROCM

} // anonymous namespace
