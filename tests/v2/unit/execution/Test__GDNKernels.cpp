/**
 * @file Test__GDNKernels.cpp
 * @brief Unit tests for Phase C: GDN Kernels
 *
 * Tests GDNProjectionStage, ShortConv1dStage, and GDNRecurrenceStage
 * for correctness, state management, and OpenMP parallelization.
 *
 * Three test categories:
 * 1. Stage tests with real CPU kernels (end-to-end correctness)
 * 2. Stage tests with mock kernels (verify delegation)
 * 3. Direct kernel tests (algorithm correctness in isolation)
 *
 * Parity is verified against the reference formulas from HuggingFace
 * transformers 5.4.0 (torch_recurrent_gated_delta_rule,
 * torch_causal_conv1d_update).
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "execution/compute_stages/stages/GDNProjectionStage.h"
#include "execution/compute_stages/stages/ShortConv1dStage.h"
#include "execution/compute_stages/stages/GDNRecurrenceStage.h"
#include "execution/compute_stages/IComputeStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include "execution/cache/HybridCacheManager.h"
#include "kernels/cpu/gdn/CPUShortConvolution.h"
#include "kernels/cpu/gdn/CPUGatedDeltaNet.h"
#include "../../mocks/MockComputeStage.h"
#ifdef HAVE_ROCM
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "kernels/rocm/ROCmWeightPacker.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "kernels/rocm/gdn/ROCmGatedDeltaNet.h"
#include "kernels/rocm/gdn/ROCmShortConvolution.h"
#include <hip/hip_runtime.h>
#endif
#include "tensors/Tensors.h"
#include "tensors/TensorKernels.h"
#include "utils/DebugEnv.h"
#include "../../utils/TestTensorFactory.h"
#include "../../utils/PreparedWeightTestHarness.h"

using namespace llaminar2;
using ::testing::_;
using ::testing::Return;

namespace
{
    std::unique_ptr<IDeviceContext> makeCPUContext()
    {
        return std::make_unique<CPUDeviceContext>(DeviceId::cpu(), 1);
    }

    // Helper: create FP32 tensor with given data
    std::shared_ptr<FP32Tensor> makeFP32(const std::vector<size_t> &shape, const float *data = nullptr)
    {
        auto t = std::make_shared<FP32Tensor>(shape, DeviceId::cpu());
        if (data)
        {
            std::memcpy(t->mutable_data(), data, t->numel() * sizeof(float));
        }
        else
        {
            std::memset(t->mutable_data(), 0, t->numel() * sizeof(float));
        }
        return t;
    }

    // Helper: create FP32 tensor filled with a constant
    std::shared_ptr<FP32Tensor> makeFP32Const(const std::vector<size_t> &shape, float val)
    {
        auto t = std::make_shared<FP32Tensor>(shape, DeviceId::cpu());
        float *d = t->mutable_data();
        for (size_t i = 0; i < t->numel(); ++i)
            d[i] = val;
        return t;
    }

    // Helper: create FP32 tensor with sequential values
    std::shared_ptr<FP32Tensor> makeFP32Seq(const std::vector<size_t> &shape, float start = 0.1f, float step = 0.1f)
    {
        auto t = std::make_shared<FP32Tensor>(shape, DeviceId::cpu());
        float *d = t->mutable_data();
        for (size_t i = 0; i < t->numel(); ++i)
            d[i] = start + step * static_cast<float>(i);
        return t;
    }

    // Helper: create FP32 tensor with random normal values
    std::shared_ptr<FP32Tensor> makeFP32Random(const std::vector<size_t> &shape, float mean = 0.0f, float stddev = 0.1f, unsigned seed = 42)
    {
        auto t = std::make_shared<FP32Tensor>(shape, DeviceId::cpu());
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(mean, stddev);
        float *d = t->mutable_data();
        for (size_t i = 0; i < t->numel(); ++i)
            d[i] = dist(gen);
        return t;
    }

    // SiLU reference
    float silu(float x) { return x / (1.0f + std::exp(-x)); }

    // Softplus reference
    float softplus(float x) { return (x > 20.0f) ? x : std::log1p(std::exp(x)); }

    // CPU kernel instances for end-to-end stage tests
    CPUShortConvolution g_cpu_conv;
    CPUGatedDeltaNet g_cpu_gdn;

    class ScopedEnvVar
    {
    public:
        explicit ScopedEnvVar(const char *name)
            : name_(name)
        {
            const char *value = std::getenv(name_);
            if (value)
            {
                had_value_ = true;
                old_value_ = value;
            }
            unsetenv(name_);
        }

        ~ScopedEnvVar()
        {
            if (had_value_)
                setenv(name_, old_value_.c_str(), 1);
            else
                unsetenv(name_);
            mutableDebugEnv().rocm.reload();
        }

        void set(const char *value)
        {
            setenv(name_, value, 1);
            mutableDebugEnv().rocm.reload();
        }

        void clear()
        {
            unsetenv(name_);
            mutableDebugEnv().rocm.reload();
        }

    private:
        const char *name_;
        bool had_value_ = false;
        std::string old_value_;
    };

#ifdef HAVE_ROCM
    struct GDNProjectionCodebookSpec
    {
        std::string name;
        float min_cosine;
        std::function<std::unique_ptr<TensorBase>(size_t rows, size_t cols)> create;
    };

    static const std::vector<GDNProjectionCodebookSpec> ALL_GDN_PROJECTION_CODEBOOKS = {
        {"Q4_0", 0.990f, [](size_t rows, size_t cols)
         { return test::TestTensorFactory::createQ4_0Random({rows, cols}); }},
        {"Q8_0", 0.990f, [](size_t rows, size_t cols)
         { return test::TestTensorFactory::createQ8_0Random({rows, cols}); }},
        {"IQ4_NL", 0.990f, [](size_t rows, size_t cols)
         { return test::TestTensorFactory::createIQ4_NLRandom({rows, cols}); }},
        {"Q4_1", 0.990f, [](size_t rows, size_t cols)
         { return test::TestTensorFactory::createQ4_1Random({rows, cols}); }},
        {"Q5_0", 0.990f, [](size_t rows, size_t cols)
         { return test::TestTensorFactory::createQ5_0Random({rows, cols}); }},
        {"Q5_1", 0.990f, [](size_t rows, size_t cols)
         { return test::TestTensorFactory::createQ5_1Random({rows, cols}); }},
        {"IQ4_XS", 0.985f, [](size_t rows, size_t cols)
         { return test::TestTensorFactory::createIQ4_XSRandom({rows, cols}); }},
        {"Q4_K", 0.985f, [](size_t rows, size_t cols)
         { return test::TestTensorFactory::createQ4_KRandom({rows, cols}); }},
        {"Q5_K", 0.985f, [](size_t rows, size_t cols)
         { return test::TestTensorFactory::createQ5_KRandom({rows, cols}); }},
        {"Q6_K", 0.990f, [](size_t rows, size_t cols)
         { return test::TestTensorFactory::createQ6_KRandom({rows, cols}); }},
        {"Q3_K", 0.980f, [](size_t rows, size_t cols)
         { return test::TestTensorFactory::createQ3_KRandom({rows, cols}); }},
        {"Q2_K", 0.970f, [](size_t rows, size_t cols)
         { return test::TestTensorFactory::createQ2_KRandom({rows, cols}); }},
        {"IQ3_S", 0.975f, [](size_t rows, size_t cols)
         { return test::TestTensorFactory::createIQ3_SRandom({rows, cols}); }},
        {"IQ3_XXS", 0.970f, [](size_t rows, size_t cols)
         { return test::TestTensorFactory::createIQ3_XXSRandom({rows, cols}); }},
        {"IQ2_S", 0.960f, [](size_t rows, size_t cols)
         { return test::TestTensorFactory::createIQ2_SRandom({rows, cols}); }},
        {"IQ2_XS", 0.960f, [](size_t rows, size_t cols)
         { return test::TestTensorFactory::createIQ2_XSRandom({rows, cols}); }},
        {"IQ2_XXS", 0.950f, [](size_t rows, size_t cols)
         { return test::TestTensorFactory::createIQ2_XXSRandom({rows, cols}); }},
        {"IQ1_S", 0.930f, [](size_t rows, size_t cols)
         { return test::TestTensorFactory::createIQ1_SRandom({rows, cols}); }},
        {"IQ1_M", 0.930f, [](size_t rows, size_t cols)
         { return test::TestTensorFactory::createIQ1_MRandom({rows, cols}); }},
    };

    float vectorCosine(const std::vector<float> &actual, const std::vector<float> &expected)
    {
        double dot = 0.0;
        double actual_norm = 0.0;
        double expected_norm = 0.0;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            dot += static_cast<double>(actual[i]) * expected[i];
            actual_norm += static_cast<double>(actual[i]) * actual[i];
            expected_norm += static_cast<double>(expected[i]) * expected[i];
        }
        if (actual_norm == 0.0 || expected_norm == 0.0)
            return 0.0f;
        return static_cast<float>(dot / (std::sqrt(actual_norm) * std::sqrt(expected_norm)));
    }

    float vectorRelativeL2(const std::vector<float> &actual, const std::vector<float> &expected)
    {
        double sum_sq_diff = 0.0;
        double sum_sq_ref = 0.0;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            const double diff = static_cast<double>(actual[i]) - expected[i];
            sum_sq_diff += diff * diff;
            sum_sq_ref += static_cast<double>(expected[i]) * expected[i];
        }
        return static_cast<float>(std::sqrt(sum_sq_diff / std::max(sum_sq_ref, 1e-30)));
    }

    float vectorMaxAbsDiff(const std::vector<float> &actual, const std::vector<float> &expected)
    {
        float max_abs = 0.0f;
        for (size_t i = 0; i < actual.size(); ++i)
            max_abs = std::max(max_abs, std::abs(actual[i] - expected[i]));
        return max_abs;
    }

    float vectorMaxAbsValue(const std::vector<float> &values)
    {
        float max_abs = 0.0f;
        for (float value : values)
            max_abs = std::max(max_abs, std::abs(value));
        return max_abs;
    }

    void appendCpuGDNProjectionReference(
        const float *input,
        const std::vector<float> &weights,
        int rows,
        int cols,
        std::vector<float> &out)
    {
        for (int row = 0; row < rows; ++row)
        {
            double acc = 0.0;
            for (int col = 0; col < cols; ++col)
                acc += static_cast<double>(input[col]) * weights[static_cast<size_t>(row) * cols + col];
            out.push_back(static_cast<float>(acc));
        }
    }

    void appendTensorData(FP32Tensor *tensor, int count, std::vector<float> &out)
    {
        const float *data = tensor->data();
        out.insert(out.end(), data, data + count);
    }

    struct GDNProjectionBundle
    {
        std::unique_ptr<TensorBase> weights;
        std::vector<float> weights_fp32;
        rocm::ROCmPackedWeights packed;
        std::unique_ptr<rocm::ROCmQuantisedGemmKernel> kernel;
    };
#endif

    class RecordingWorkspaceGemm final : public ITensorGemm, public IWorkspaceConsumer
    {
    public:
        explicit RecordingWorkspaceGemm(std::string name)
            : name_(std::move(name))
        {
        }

        bool supports_device(int) const override { return true; }

        bool multiply_tensor(
            const TensorBase *,
            TensorBase *,
            int,
            int,
            int,
            bool,
            float,
            float,
            const TensorBase *,
            const IMPIContext *,
            int,
            DeviceWorkspaceManager *,
            int) override
        {
            return false;
        }

        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override
        {
            observed_m.push_back(m);
            observed_n.push_back(n);
            observed_k.push_back(k);

            WorkspaceRequirements reqs;
            reqs.buffers.push_back({
                name_ + "_n" + std::to_string(n),
                static_cast<size_t>(std::max(1, n)),
                1,
                true});
            return reqs;
        }

        void bindWorkspace(DeviceWorkspaceManager *workspace) override { workspace_ = workspace; }
        bool hasWorkspace() const override { return workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }

        mutable std::vector<int> observed_m;
        mutable std::vector<int> observed_n;
        mutable std::vector<int> observed_k;

    private:
        std::string name_;
        DeviceWorkspaceManager *workspace_ = nullptr;
    };

} // namespace

TEST(GDNROCmConfig, ConcurrentDecodeFlagDefaultsOffAndParsesEnv)
{
    ScopedEnvVar flag("LLAMINAR_ROCM_GDN_CONCURRENT_DECODE");

    flag.clear();
    EXPECT_FALSE(debugEnv().rocm.gdn_concurrent_decode);

    flag.set("1");
    EXPECT_TRUE(debugEnv().rocm.gdn_concurrent_decode);

    flag.set("0");
    EXPECT_FALSE(debugEnv().rocm.gdn_concurrent_decode);
}

TEST(GDNROCmConfig, SharedExpertGroupedDecodeFlagDefaultsOffAndParsesEnv)
{
    ScopedEnvVar flag("LLAMINAR_ROCM_SHARED_EXPERT_GROUPED_DECODE");

    flag.clear();
    EXPECT_FALSE(debugEnv().rocm.shared_expert_grouped_decode);

    flag.set("1");
    EXPECT_TRUE(debugEnv().rocm.shared_expert_grouped_decode);

    flag.set("0");
    EXPECT_FALSE(debugEnv().rocm.shared_expert_grouped_decode);
}

TEST(Test__GDNKernels, ShortConv_WorkspaceRequirementUsesActiveSeqLen)
{
    ShortConv1dStage::Params p;
    p.device_id = DeviceId::rocm(0);
    p.seq_len = 1;
    p.channels = 8192;

    ShortConv1dStage stage(p);
    const auto reqs = stage.getWorkspaceRequirements(/*m=*/1536);

    const auto *scratch = reqs.find(ShortConv1dStage::WS_INPLACE_PREFILL_SCRATCH);
    ASSERT_NE(scratch, nullptr);
    EXPECT_EQ(scratch->size_bytes,
              static_cast<size_t>(1536) * static_cast<size_t>(8192) * sizeof(float));
    EXPECT_TRUE(scratch->required);

    const WorkspaceDescriptor *effective_len = nullptr;
    for (const auto &buffer : reqs.buffers)
    {
        if (buffer.name.find(ShortConv1dStage::WS_EFFECTIVE_SEQ_LEN_SCALAR) != std::string::npos)
        {
            effective_len = &buffer;
            break;
        }
    }
    ASSERT_NE(effective_len, nullptr);
    EXPECT_GE(effective_len->size_bytes, sizeof(int));
    EXPECT_TRUE(effective_len->required);
}

TEST(Test__GDNKernels, Recurrence_WorkspaceRequirementSharesMergedQKVScratch)
{
    GDNRecurrenceStage::Params p;
    p.device_id = DeviceId::rocm(0);
    p.seq_len = 1;
    p.n_heads = 32;
    p.n_k_heads = 4;
    p.d_k = 128;
    p.d_v = 128;
    p.global_v_head_offset = 8;

    GDNRecurrenceStage stage(p);
    const auto reqs = stage.getWorkspaceRequirements(/*m=*/1536);

    const auto *scratch = reqs.find(GDNRecurrenceStage::WS_DEINTERLEAVE_SCRATCH);
    ASSERT_NE(scratch, nullptr);
    const size_t row_floats = static_cast<size_t>(32) * static_cast<size_t>((2 * 128) + 128);
    EXPECT_EQ(scratch->size_bytes,
              static_cast<size_t>(1536) * row_floats * sizeof(float));
    EXPECT_TRUE(scratch->required);
}

TEST(Test__GDNKernels, Projection_WorkspaceRequirementsUsePerProjectionN)
{
    RecordingWorkspaceGemm qkv("qkv");
    RecordingWorkspaceGemm z("z");
    RecordingWorkspaceGemm alpha("alpha");
    RecordingWorkspaceGemm beta("beta");

    GDNProjectionStage::Params p;
    p.device_id = DeviceId::rocm(0);
    p.m = 4;
    p.k = 128;
    p.n_qkv = 96;
    p.n_z = 64;
    p.n_a = 8;
    p.n_b = 12;
    p.gemm_qkv = &qkv;
    p.gemm_z = &z;
    p.gemm_a = &alpha;
    p.gemm_b = &beta;

    GDNProjectionStage stage(p);
    const auto reqs = stage.getWorkspaceRequirements(/*m=*/4, /*n=*/999, /*k=*/128);
    (void)reqs;

    ASSERT_EQ(qkv.observed_n, std::vector<int>({96}));
    ASSERT_EQ(z.observed_n, std::vector<int>({64}));
    ASSERT_EQ(alpha.observed_n, std::vector<int>({8}));
    ASSERT_EQ(beta.observed_n, std::vector<int>({12}));

    EXPECT_EQ(qkv.observed_m, std::vector<int>({4}));
    EXPECT_EQ(z.observed_k, std::vector<int>({128}));
}

// ============================================================================
// Mock Kernels for verifying stage delegation
// ============================================================================

class MockShortConvolution : public ITensorShortConvolution
{
public:
    MOCK_METHOD(bool, forward,
                (const float *input, const float *weight, const float *bias,
                 float *output, float *conv_state,
                 int seq_len, int channels, int kernel_size,
                 bool apply_silu),
                (override));
};

class MockGatedDeltaNet : public ITensorGatedDeltaNet
{
public:
    MOCK_METHOD(bool, chunk_forward,
                (const float *Q, const float *K, const float *V,
                 const float *alpha, const float *beta_raw,
                 const float *A_log, const float *dt_bias,
                 float *output, float *state,
                 int seq_len, int n_heads, int d_k, int d_v,
                 int chunk_size, bool use_qk_l2norm),
                (override));

    MOCK_METHOD(bool, recurrent_step,
                (const float *q, const float *k, const float *v,
                 const float *alpha, const float *beta_raw,
                 const float *A_log, const float *dt_bias,
                 float *output, float *state,
                 int n_heads, int d_k, int d_v,
                 bool use_qk_l2norm),
                (override));

    MOCK_METHOD(bool, deinterleave_qkv_device,
                (const float *d_merged_qkv,
                 float *&d_q, float *&d_k, float *&d_v,
                 int seq_len, int n_k_heads, int n_v_heads,
                 int head_dim_k, int head_dim_v, int global_v_head_offset),
                (override));
};

TEST(Test__GDNKernels, Recurrence_GPUDeinterleaveRequiresBoundWorkspaceBeforeKernelDispatch)
{
#ifdef HAVE_ROCM
    static constexpr int seq_len = 2;
    static constexpr int n_heads = 4;
    static constexpr int n_k_heads = 2;
    static constexpr int d_k = 8;
    static constexpr int d_v = 8;

    const int qkv_cols = n_k_heads * d_k * 2 + n_heads * d_v;
    FP32Tensor merged_qkv({seq_len, qkv_cols}, DeviceId::rocm(0));
    FP32Tensor alpha({seq_len, n_heads}, DeviceId::rocm(0));
    FP32Tensor beta({seq_len, n_heads}, DeviceId::rocm(0));
    FP32Tensor a_log({1, n_heads}, DeviceId::rocm(0));
    FP32Tensor dt_bias({1, n_heads}, DeviceId::rocm(0));
    FP32Tensor output({seq_len, n_heads * d_v}, DeviceId::rocm(0));
    std::vector<float> recurrence_state(static_cast<size_t>(n_heads * d_k * d_v), 0.0f);

    MockGatedDeltaNet kernel;
    EXPECT_CALL(kernel, deinterleave_qkv_device(_, _, _, _, _, _, _, _, _, _)).Times(0);
    EXPECT_CALL(kernel, chunk_forward(_, _, _, _, _, _, _, _, _, _, _, _, _, _, _)).Times(0);

    GDNRecurrenceStage::Params p;
    p.device_id = DeviceId::rocm(0);
    p.seq_len = seq_len;
    p.n_heads = n_heads;
    p.n_k_heads = n_k_heads;
    p.d_k = d_k;
    p.d_v = d_v;
    p.global_v_head_offset = 1;
    p.Q = &merged_qkv;
    p.K = &merged_qkv;
    p.V = &merged_qkv;
    p.alpha = &alpha;
    p.beta = &beta;
    p.A_log = &a_log;
    p.dt_bias = &dt_bias;
    p.output = &output;
    p.recurrence_state = recurrence_state.data();
    p.kernel = &kernel;

    GDNRecurrenceStage stage(p);
    llaminar2::testing::MockDeviceContext ctx(DeviceId::rocm(0), ComputeBackendType::GPU_ROCM);

    EXPECT_FALSE(stage.execute(&ctx))
        << "Graph-stage GPU deinterleave must require the declared workspace buffer "
           "instead of falling through to backend-owned scratch allocation";
#else
    GTEST_SKIP() << "ROCm backend not compiled";
#endif
}

template <int Tag>
class CountingProjectionGemm : public ITensorGemm
{
public:
    explicit CountingProjectionGemm(float fill_value,
                                    bool supports_fused = false,
                                    bool fused_success = false)
        : fill_value_(fill_value),
          supports_fused_(supports_fused),
          fused_success_(fused_success)
    {
    }

    bool supports_device(int) const override { return true; }
    bool supports_fused_projection() const override { return supports_fused_; }

    bool multiply_tensor(
        const TensorBase *,
        TensorBase *C,
        int m,
        int n,
        int k,
        bool,
        float,
        float,
        const TensorBase *,
        const IMPIContext *,
        int,
        DeviceWorkspaceManager *,
        int) override
    {
        (void)k;
        ++multiply_calls;
        auto *output = dynamic_cast<FP32Tensor *>(C);
        if (!output)
            return false;
        float *values = output->mutable_data();
        for (int i = 0; i < m * n; ++i)
            values[i] = fill_value_;
        return true;
    }

    bool multiply_fused_tensor(
        const TensorBase *,
        const std::vector<TensorProjectionDesc> &projections,
        int m,
        int,
        const IMPIContext * = nullptr,
        DeviceWorkspaceManager * = nullptr) override
    {
        ++fused_calls;
        fused_projection_count += static_cast<int>(projections.size());
        if (!fused_success_)
            return false;

        for (const auto &projection : projections)
        {
            auto *typed_kernel = dynamic_cast<CountingProjectionGemm<Tag> *>(projection.kernel);
            auto *output = dynamic_cast<FP32Tensor *>(projection.output);
            if (!typed_kernel || !output)
                return false;

            float *values = output->mutable_data();
            for (int i = 0; i < m * projection.n; ++i)
                values[i] = typed_kernel->fill_value_;
        }
        return true;
    }

    int multiply_calls = 0;
    int fused_calls = 0;
    int fused_projection_count = 0;

private:
    float fill_value_;
    bool supports_fused_ = false;
    bool fused_success_ = false;
};

// ============================================================================
// C1: GDNProjectionStage Tests
// ============================================================================

TEST(Test__GDNKernels, Projection_TypeAndBackend)
{
    GDNProjectionStage::Params p;
    GDNProjectionStage stage(p);
    EXPECT_EQ(stage.type(), ComputeStageType::GDN_PROJECTION);
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
#ifdef HAVE_CUDA
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::GPU_CUDA));
#else
    EXPECT_FALSE(stage.supportsBackend(ComputeBackendType::GPU_CUDA));
#endif
}

TEST(Test__GDNKernels, Projection_NullPointers_Fails)
{
    auto ctx = makeCPUContext();
    GDNProjectionStage::Params p;
    GDNProjectionStage stage(p);
    EXPECT_FALSE(stage.execute(ctx.get()));
}

TEST(Test__GDNKernels, Projection_EstimatedFlops)
{
    GDNProjectionStage::Params p;
    p.m = 10;
    p.k = 512;
    p.n_qkv = 768;
    p.n_z = 256;
    p.n_a = 16;
    p.n_b = 16;
    GDNProjectionStage stage(p);

    // Flops = 2 * M * K * (n_qkv + n_z + n_a + n_b)
    const size_t expected = 2ull * 10 * 512 * (768 + 256 + 16 + 16);
    EXPECT_EQ(stage.estimatedFlops(), expected);
}

TEST(Test__GDNKernels, Projection_DumpInfo)
{
    auto input = makeFP32({4, 64});
    auto w_qkv = makeFP32({64, 96});
    auto out_qkv = makeFP32({4, 96});
    auto w_z = makeFP32({64, 32});
    auto out_z = makeFP32({4, 32});
    auto w_a = makeFP32({64, 4});
    auto out_a = makeFP32({4, 4});
    auto w_b = makeFP32({64, 4});
    auto out_b = makeFP32({4, 4});

    GDNProjectionStage::Params p;
    p.input = input.get();
    p.w_qkv = w_qkv.get();
    p.output_qkv = out_qkv.get();
    p.w_z = w_z.get();
    p.output_z = out_z.get();
    p.w_a = w_a.get();
    p.output_a = out_a.get();
    p.w_b = w_b.get();
    p.output_b = out_b.get();

    GDNProjectionStage stage(p);
    auto info = stage.buildDumpInfoImpl();

    EXPECT_EQ(info.inputs.size(), 5u);  // input + 4 weights
    EXPECT_EQ(info.outputs.size(), 4u); // 4 outputs
}

TEST(Test__GDNKernels, Projection_BufferContract)
{
    GDNProjectionStage::Params p;
    p.input_buffer_id = BufferId::HIDDEN_STATE;
    p.output_qkv_buffer_id = BufferId::Q_PROJ;
    p.output_z_buffer_id = BufferId::K_PROJ;

    GDNProjectionStage stage(p);
    auto contract = stage.bufferContract();
    // Should have at least the configured IDs
    SUCCEED(); // Contract creation doesn't crash
}

TEST(Test__GDNKernels, Projection_MixedKernelTypesUsePerProjectionFallback)
{
    auto ctx = makeCPUContext();

    auto input = makeFP32Seq({2, 3});
    auto w_qkv = makeFP32({3, 4});
    auto out_qkv = makeFP32({2, 4});
    auto w_z = makeFP32({3, 2});
    auto out_z = makeFP32({2, 2});
    auto w_a = makeFP32({3, 1});
    auto out_a = makeFP32({2, 1});
    auto w_b = makeFP32({3, 1});
    auto out_b = makeFP32({2, 1});

    CountingProjectionGemm<0> qkv_gemm(1.0f);
    CountingProjectionGemm<0> z_gemm(2.0f);
    CountingProjectionGemm<1> a_gemm(3.0f);
    CountingProjectionGemm<1> b_gemm(4.0f);

    GDNProjectionStage::Params p;
    p.input = input.get();
    p.m = 2;
    p.k = 3;
    p.w_qkv = w_qkv.get();
    p.output_qkv = out_qkv.get();
    p.n_qkv = 4;
    p.w_z = w_z.get();
    p.output_z = out_z.get();
    p.n_z = 2;
    p.w_a = w_a.get();
    p.output_a = out_a.get();
    p.n_a = 1;
    p.w_b = w_b.get();
    p.output_b = out_b.get();
    p.n_b = 1;
    p.gemm_qkv = &qkv_gemm;
    p.gemm_z = &z_gemm;
    p.gemm_a = &a_gemm;
    p.gemm_b = &b_gemm;

    GDNProjectionStage stage(p);
    EXPECT_TRUE(stage.execute(ctx.get()));

    EXPECT_EQ(qkv_gemm.fused_calls, 0);
    EXPECT_EQ(z_gemm.fused_calls, 0);
    EXPECT_EQ(a_gemm.fused_calls, 0);
    EXPECT_EQ(b_gemm.fused_calls, 0);
    EXPECT_EQ(qkv_gemm.multiply_calls, 1);
    EXPECT_EQ(z_gemm.multiply_calls, 1);
    EXPECT_EQ(a_gemm.multiply_calls, 1);
    EXPECT_EQ(b_gemm.multiply_calls, 1);

    EXPECT_FLOAT_EQ(out_qkv->data()[0], 1.0f);
    EXPECT_FLOAT_EQ(out_z->data()[0], 2.0f);
    EXPECT_FLOAT_EQ(out_a->data()[0], 3.0f);
    EXPECT_FLOAT_EQ(out_b->data()[0], 4.0f);
}

TEST(Test__GDNKernels, Projection_MixedKernelTypesFuseSupportedSubgroups)
{
    auto ctx = makeCPUContext();

    auto input = makeFP32Seq({2, 3});
    auto w_qkv = makeFP32({3, 4});
    auto out_qkv = makeFP32({2, 4});
    auto w_z = makeFP32({3, 2});
    auto out_z = makeFP32({2, 2});
    auto w_a = makeFP32({3, 1});
    auto out_a = makeFP32({2, 1});
    auto w_b = makeFP32({3, 1});
    auto out_b = makeFP32({2, 1});

    CountingProjectionGemm<0> qkv_gemm(1.0f, true, true);
    CountingProjectionGemm<0> z_gemm(2.0f, true, true);
    CountingProjectionGemm<1> a_gemm(3.0f);
    CountingProjectionGemm<1> b_gemm(4.0f);

    GDNProjectionStage::Params p;
    p.input = input.get();
    p.m = 2;
    p.k = 3;
    p.w_qkv = w_qkv.get();
    p.output_qkv = out_qkv.get();
    p.n_qkv = 4;
    p.w_z = w_z.get();
    p.output_z = out_z.get();
    p.n_z = 2;
    p.w_a = w_a.get();
    p.output_a = out_a.get();
    p.n_a = 1;
    p.w_b = w_b.get();
    p.output_b = out_b.get();
    p.n_b = 1;
    p.gemm_qkv = &qkv_gemm;
    p.gemm_z = &z_gemm;
    p.gemm_a = &a_gemm;
    p.gemm_b = &b_gemm;

    GDNProjectionStage stage(p);
    EXPECT_TRUE(stage.execute(ctx.get()));

    EXPECT_EQ(qkv_gemm.fused_calls, 1);
    EXPECT_EQ(qkv_gemm.fused_projection_count, 2);
    EXPECT_EQ(z_gemm.fused_calls, 0);
    EXPECT_EQ(qkv_gemm.multiply_calls, 0);
    EXPECT_EQ(z_gemm.multiply_calls, 0);
    EXPECT_EQ(a_gemm.multiply_calls, 1);
    EXPECT_EQ(b_gemm.multiply_calls, 1);

    EXPECT_FLOAT_EQ(out_qkv->data()[0], 1.0f);
    EXPECT_FLOAT_EQ(out_z->data()[0], 2.0f);
    EXPECT_FLOAT_EQ(out_a->data()[0], 3.0f);
    EXPECT_FLOAT_EQ(out_b->data()[0], 4.0f);
}

TEST(Test__GDNKernels, Projection_FusedSubgroupFailureHardFails)
{
    auto ctx = makeCPUContext();

    auto input = makeFP32Seq({2, 3});
    auto w_qkv = makeFP32({3, 4});
    auto out_qkv = makeFP32({2, 4});
    auto w_z = makeFP32({3, 2});
    auto out_z = makeFP32({2, 2});
    auto w_a = makeFP32({3, 1});
    auto out_a = makeFP32({2, 1});
    auto w_b = makeFP32({3, 1});
    auto out_b = makeFP32({2, 1});

    CountingProjectionGemm<0> qkv_gemm(1.0f, true, false);
    CountingProjectionGemm<0> z_gemm(2.0f, true, false);
    CountingProjectionGemm<1> a_gemm(3.0f);
    CountingProjectionGemm<1> b_gemm(4.0f);

    GDNProjectionStage::Params p;
    p.input = input.get();
    p.m = 2;
    p.k = 3;
    p.w_qkv = w_qkv.get();
    p.output_qkv = out_qkv.get();
    p.n_qkv = 4;
    p.w_z = w_z.get();
    p.output_z = out_z.get();
    p.n_z = 2;
    p.w_a = w_a.get();
    p.output_a = out_a.get();
    p.n_a = 1;
    p.w_b = w_b.get();
    p.output_b = out_b.get();
    p.n_b = 1;
    p.gemm_qkv = &qkv_gemm;
    p.gemm_z = &z_gemm;
    p.gemm_a = &a_gemm;
    p.gemm_b = &b_gemm;

    GDNProjectionStage stage(p);
    EXPECT_FALSE(stage.execute(ctx.get()));

    EXPECT_EQ(qkv_gemm.fused_calls, 1);
    EXPECT_EQ(qkv_gemm.fused_projection_count, 2);
    EXPECT_EQ(qkv_gemm.multiply_calls, 0);
    EXPECT_EQ(z_gemm.multiply_calls, 0);
    EXPECT_EQ(a_gemm.multiply_calls, 0);
    EXPECT_EQ(b_gemm.multiply_calls, 0);
}

TEST(Test__GDNKernels, Projection_Qwen36NodeLocalTPPrefillShapeResolvesPreparedMixedFallback)
{
    auto ctx = makeCPUContext();

    const int M = 9;
    const int K = 5120;
    const int N_QKV = 7168;
    const int N_Z = 3072;
    const int N_ALPHA = 24;
    const int N_BETA = 24;

    auto input = test::TestTensorFactory::createFP32Random(
        {static_cast<size_t>(M), static_cast<size_t>(K)}, -1.0f, 1.0f, 42);
    auto w_qkv = test::TestTensorFactory::createQ5_KRandom(
        {static_cast<size_t>(N_QKV), static_cast<size_t>(K)});
    auto w_z = test::TestTensorFactory::createQ4_KRandom(
        {static_cast<size_t>(N_Z), static_cast<size_t>(K)});
    auto w_a = test::TestTensorFactory::createFP32Random(
        {static_cast<size_t>(N_ALPHA), static_cast<size_t>(K)}, -0.1f, 0.1f, 43);
    auto w_b = test::TestTensorFactory::createFP32Random(
        {static_cast<size_t>(N_BETA), static_cast<size_t>(K)}, -0.1f, 0.1f, 44);

    auto out_qkv = test::TestTensorFactory::createFP32Zeros(
        {static_cast<size_t>(M), static_cast<size_t>(N_QKV)});
    auto out_z = test::TestTensorFactory::createFP32Zeros(
        {static_cast<size_t>(M), static_cast<size_t>(N_Z)});
    auto out_a = test::TestTensorFactory::createFP32Zeros(
        {static_cast<size_t>(M), static_cast<size_t>(N_ALPHA)});
    auto out_b = test::TestTensorFactory::createFP32Zeros(
        {static_cast<size_t>(M), static_cast<size_t>(N_BETA)});

    constexpr ModelContextId model_id{3600};
    PreparedWeightStore store(model_id);
    auto qkv_binding = test::makePreparedWeightTestBinding(
        w_qkv.get(), DeviceId::cpu(), "blk.0.gdn_qkv_proj.weight", model_id);
    auto z_binding = test::makePreparedWeightTestBinding(
        w_z.get(), DeviceId::cpu(), "blk.0.gdn_z_proj.weight", model_id);
    auto a_binding = test::makePreparedWeightTestBinding(
        w_a.get(), DeviceId::cpu(), "blk.0.gdn_alpha_proj.weight", model_id);
    auto b_binding = test::makePreparedWeightTestBinding(
        w_b.get(), DeviceId::cpu(), "blk.0.gdn_beta_proj.weight", model_id);

    auto qkv_ref = store.prepareGemm(qkv_binding);
    auto z_ref = store.prepareGemm(z_binding);
    auto a_ref = store.prepareGemm(a_binding);
    auto b_ref = store.prepareGemm(b_binding);
    ASSERT_NE(store.gemmKernel(qkv_ref), nullptr);
    ASSERT_NE(store.gemmKernel(z_ref), nullptr);
    ASSERT_NE(store.gemmKernel(a_ref), nullptr);
    ASSERT_NE(store.gemmKernel(b_ref), nullptr);

    w_qkv->release_raw_data();
    w_z->release_raw_data();

    GDNProjectionStage::Params p;
    p.input = input.get();
    p.m = M;
    p.k = K;
    p.w_qkv = w_qkv.get();
    p.output_qkv = out_qkv.get();
    p.n_qkv = N_QKV;
    p.w_z = w_z.get();
    p.output_z = out_z.get();
    p.n_z = N_Z;
    p.w_a = w_a.get();
    p.output_a = out_a.get();
    p.n_a = N_ALPHA;
    p.w_b = w_b.get();
    p.output_b = out_b.get();
    p.n_b = N_BETA;
    p.prepared_ref_qkv = qkv_ref;
    p.prepared_ref_z = z_ref;
    p.prepared_ref_a = a_ref;
    p.prepared_ref_b = b_ref;
    p.prepared_store = &store;

    GDNProjectionStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    auto assertFiniteNonzero = [](const TensorBase *tensor, int rows, int cols, const char *name)
    {
        const float *data = tensor->data();
        bool any_nonzero = false;
        for (size_t i = 0; i < static_cast<size_t>(rows) * cols; ++i)
        {
            ASSERT_TRUE(std::isfinite(data[i])) << name << " non-finite at " << i;
            any_nonzero = any_nonzero || data[i] != 0.0f;
        }
        EXPECT_TRUE(any_nonzero) << name << " is all zero";
    };

    assertFiniteNonzero(out_qkv.get(), M, N_QKV, "qkv");
    assertFiniteNonzero(out_z.get(), M, N_Z, "z");
    assertFiniteNonzero(out_a.get(), M, N_ALPHA, "alpha");
    assertFiniteNonzero(out_b.get(), M, N_BETA, "beta");
}

// ============================================================================
// C2: ShortConv1dStage Tests
// ============================================================================

TEST(Test__GDNKernels, Conv1d_TypeAndBackend)
{
    ShortConv1dStage::Params p;
    p.kernel = &g_cpu_conv;
    ShortConv1dStage stage(p);
    EXPECT_EQ(stage.type(), ComputeStageType::SHORT_CONV1D);
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
}

TEST(Test__GDNKernels, Conv1d_NullPointers_Fails)
{
    auto ctx = makeCPUContext();
    ShortConv1dStage::Params p;
    p.kernel = &g_cpu_conv;
    ShortConv1dStage stage(p);
    EXPECT_FALSE(stage.execute(ctx.get()));
}

TEST(Test__GDNKernels, Conv1d_Decode_IdentityWeight)
{
    // With kernel_size=2 and weight=[0, 1], the conv1d just passes through
    // the current input (modulo SiLU activation)
    const int channels = 4;
    const int kernel_size = 2;
    const int state_len = kernel_size - 1; // 1

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto input = makeFP32({1, static_cast<size_t>(channels)}, input_data.data());
    auto output = makeFP32({1, static_cast<size_t>(channels)});

    // Weight: [0, 1] for each channel — passes through the current input
    std::vector<float> weight_data(channels * kernel_size, 0.0f);
    for (int c = 0; c < channels; ++c)
        weight_data[c * kernel_size + (kernel_size - 1)] = 1.0f;
    auto weight = makeFP32({static_cast<size_t>(channels), static_cast<size_t>(kernel_size)}, weight_data.data());

    std::vector<float> conv_state(channels * state_len, 0.0f);

    auto ctx = makeCPUContext();

    ShortConv1dStage::Params p;
    p.kernel = &g_cpu_conv;
    p.input = input.get();
    p.output = output.get();
    p.weight = weight.get();
    p.seq_len = 1;
    p.channels = channels;
    p.kernel_size = kernel_size;
    p.conv_state = conv_state.data();

    ShortConv1dStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    const float *out = output->data();
    for (int c = 0; c < channels; ++c)
    {
        EXPECT_NEAR(out[c], silu(input_data[c]), 1e-5f)
            << "Channel " << c;
    }
}

TEST(Test__GDNKernels, Conv1d_Decode_WithState)
{
    // kernel_size=4, channels=2, single decode step
    // state = [channels, kernel_size-1] = [2, 3]
    // Computes: conv_output = sum(state[c,k] * w[c,k]) + input[c] * w[c,3]
    // Then SiLU

    const int channels = 2;
    const int kernel_size = 4;
    const int state_len = kernel_size - 1; // 3

    // Weight for channel 0: [1, 0, 0, 1], channel 1: [0, 0, 1, 2]
    std::vector<float> weight_data = {1, 0, 0, 1, 0, 0, 1, 2};
    auto weight = makeFP32({static_cast<size_t>(channels), static_cast<size_t>(kernel_size)}, weight_data.data());

    // State for channel 0: [0.5, 0.3, 0.1], channel 1: [0.2, 0.4, 0.6]
    std::vector<float> conv_state = {0.5f, 0.3f, 0.1f, 0.2f, 0.4f, 0.6f};

    std::vector<float> input_data = {2.0f, 1.0f};
    auto input = makeFP32({1, static_cast<size_t>(channels)}, input_data.data());
    auto output = makeFP32({1, static_cast<size_t>(channels)});

    auto ctx = makeCPUContext();

    ShortConv1dStage::Params p;
    p.kernel = &g_cpu_conv;
    p.input = input.get();
    p.output = output.get();
    p.weight = weight.get();
    p.seq_len = 1;
    p.channels = channels;
    p.kernel_size = kernel_size;
    p.conv_state = conv_state.data();

    ShortConv1dStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // Channel 0: w=[1,0,0,1], state=[0.5,0.3,0.1], new=2.0
    //   raw = 1*0.5 + 0*0.3 + 0*0.1 + 1*2.0 = 2.5
    //   expected = silu(2.5)
    // Channel 1: w=[0,0,1,2], state=[0.2,0.4,0.6], new=1.0
    //   raw = 0*0.2 + 0*0.4 + 1*0.6 + 2*1.0 = 2.6
    //   expected = silu(2.6)

    const float *out = output->data();
    EXPECT_NEAR(out[0], silu(2.5f), 1e-5f);
    EXPECT_NEAR(out[1], silu(2.6f), 1e-5f);

    // After decode, state should be shifted: channel 0: [0.3, 0.1, 2.0], channel 1: [0.4, 0.6, 1.0]
    EXPECT_NEAR(conv_state[0], 0.3f, 1e-6f);
    EXPECT_NEAR(conv_state[1], 0.1f, 1e-6f);
    EXPECT_NEAR(conv_state[2], 2.0f, 1e-6f);
    EXPECT_NEAR(conv_state[3], 0.4f, 1e-6f);
    EXPECT_NEAR(conv_state[4], 0.6f, 1e-6f);
    EXPECT_NEAR(conv_state[5], 1.0f, 1e-6f);
}

TEST(Test__GDNKernels, Conv1d_Prefill_Causal)
{
    // Verify that prefill conv1d is causal: output[t] depends only on input[0..t]
    // kernel_size=3, channels=1, seq_len=4
    const int channels = 1;
    const int kernel_size = 3;
    const int seq_len = 4;

    // Weight: [1, 2, 3] (kernel for channel 0)
    std::vector<float> weight_data = {1.0f, 2.0f, 3.0f};
    auto weight = makeFP32({1, 3}, weight_data.data());

    // Input: [1, 2, 3, 4] — single channel, layout [seq_len, channels]
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto input = makeFP32({static_cast<size_t>(seq_len), 1}, input_data.data());
    auto output = makeFP32({static_cast<size_t>(seq_len), 1});

    auto ctx = makeCPUContext();

    ShortConv1dStage::Params p;
    p.kernel = &g_cpu_conv;
    p.input = input.get();
    p.output = output.get();
    p.weight = weight.get();
    p.seq_len = seq_len;
    p.channels = channels;
    p.kernel_size = kernel_size;
    // No conv_state for prefill-only test

    ShortConv1dStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    const float *out = output->data();

    // Causal conv with zero-padding on the left:
    // t=0: pad=[0,0], current=1 → sum = w[0]*0 + w[1]*0 + w[2]*1 = 3*1 = 3
    // t=1: pad=[0], current=[1,2] → sum = w[0]*0 + w[1]*1 + w[2]*2 = 2+6 = 8
    // t=2: current=[1,2,3] → sum = w[0]*1 + w[1]*2 + w[2]*3 = 1+4+9 = 14
    // t=3: current=[2,3,4] → sum = w[0]*2 + w[1]*3 + w[2]*4 = 2+6+12 = 20
    EXPECT_NEAR(out[0], silu(3.0f), 1e-5f);
    EXPECT_NEAR(out[1], silu(8.0f), 1e-5f);
    EXPECT_NEAR(out[2], silu(14.0f), 1e-5f);
    EXPECT_NEAR(out[3], silu(20.0f), 1e-5f);
}

TEST(Test__GDNKernels, Conv1d_Prefill_StoresState)
{
    // After prefill, conv_state should contain the last (kernel_size-1) inputs
    const int channels = 2;
    const int kernel_size = 3;
    const int seq_len = 5;
    const int state_len = kernel_size - 1;

    auto weight = makeFP32Const({static_cast<size_t>(channels), static_cast<size_t>(kernel_size)}, 1.0f);

    // Input: [seq_len, channels] with sequential values
    std::vector<float> input_data(seq_len * channels);
    for (int t = 0; t < seq_len; ++t)
        for (int c = 0; c < channels; ++c)
            input_data[t * channels + c] = static_cast<float>(t * channels + c + 1);

    auto input = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(channels)}, input_data.data());
    auto output = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(channels)});

    std::vector<float> conv_state(channels * state_len, 0.0f);

    auto ctx = makeCPUContext();

    ShortConv1dStage::Params p;
    p.kernel = &g_cpu_conv;
    p.input = input.get();
    p.output = output.get();
    p.weight = weight.get();
    p.seq_len = seq_len;
    p.channels = channels;
    p.kernel_size = kernel_size;
    p.conv_state = conv_state.data();

    ShortConv1dStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // State should have the last 2 timesteps (t=3 and t=4)
    // t=3: [7, 8], t=4: [9, 10]
    // state layout: [channels, state_len] = [2, 2]
    // channel 0: state[0,0]=input[3,0]=7, state[0,1]=input[4,0]=9
    // channel 1: state[1,0]=input[3,1]=8, state[1,1]=input[4,1]=10
    EXPECT_NEAR(conv_state[0], 7.0f, 1e-6f);  // ch0, t=3
    EXPECT_NEAR(conv_state[1], 9.0f, 1e-6f);  // ch0, t=4
    EXPECT_NEAR(conv_state[2], 8.0f, 1e-6f);  // ch1, t=3
    EXPECT_NEAR(conv_state[3], 10.0f, 1e-6f); // ch1, t=4
}

TEST(Test__GDNKernels, Conv1d_PrefillThenDecode_Consistency)
{
    // Running prefill then decode should give the same result as
    // running a longer prefill that includes the decode token.
    const int channels = 3;
    const int kernel_size = 4;
    const int prefill_len = 6;
    const int state_len = kernel_size - 1;

    auto weight = makeFP32Random({static_cast<size_t>(channels), static_cast<size_t>(kernel_size)}, 0.0f, 0.5f, 123);

    // Full sequence: prefill tokens + 1 decode token
    std::vector<float> full_input((prefill_len + 1) * channels);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto &v : full_input)
        v = dist(gen);

    // Path 1: Full prefill of length prefill_len+1
    auto input_full = makeFP32({static_cast<size_t>(prefill_len + 1), static_cast<size_t>(channels)}, full_input.data());
    auto output_full = makeFP32({static_cast<size_t>(prefill_len + 1), static_cast<size_t>(channels)});

    auto ctx = makeCPUContext();
    {
        ShortConv1dStage::Params p;
        p.kernel = &g_cpu_conv;
        p.input = input_full.get();
        p.output = output_full.get();
        p.weight = weight.get();
        p.seq_len = prefill_len + 1;
        p.channels = channels;
        p.kernel_size = kernel_size;
        ShortConv1dStage stage(p);
        ASSERT_TRUE(stage.execute(ctx.get()));
    }

    // Path 2: Prefill of length prefill_len, then decode of 1 token
    auto input_prefill = makeFP32({static_cast<size_t>(prefill_len), static_cast<size_t>(channels)}, full_input.data());
    auto output_prefill = makeFP32({static_cast<size_t>(prefill_len), static_cast<size_t>(channels)});
    std::vector<float> conv_state(channels * state_len, 0.0f);

    {
        ShortConv1dStage::Params p;
        p.kernel = &g_cpu_conv;
        p.input = input_prefill.get();
        p.output = output_prefill.get();
        p.weight = weight.get();
        p.seq_len = prefill_len;
        p.channels = channels;
        p.kernel_size = kernel_size;
        p.conv_state = conv_state.data();
        ShortConv1dStage stage(p);
        ASSERT_TRUE(stage.execute(ctx.get()));
    }

    // Decode the last token
    auto input_decode = makeFP32({1, static_cast<size_t>(channels)}, full_input.data() + prefill_len * channels);
    auto output_decode = makeFP32({1, static_cast<size_t>(channels)});

    {
        ShortConv1dStage::Params p;
        p.kernel = &g_cpu_conv;
        p.input = input_decode.get();
        p.output = output_decode.get();
        p.weight = weight.get();
        p.seq_len = 1;
        p.channels = channels;
        p.kernel_size = kernel_size;
        p.conv_state = conv_state.data();
        ShortConv1dStage stage(p);
        ASSERT_TRUE(stage.execute(ctx.get()));
    }

    // The decode output should match the last timestep of the full prefill
    const float *full_out = output_full->data();
    const float *decode_out = output_decode->data();
    for (int c = 0; c < channels; ++c)
    {
        EXPECT_NEAR(decode_out[c], full_out[prefill_len * channels + c], 1e-5f)
            << "Channel " << c << " mismatch between prefill+decode and full prefill";
    }
}

TEST(Test__GDNKernels, Conv1d_Decode_NoState_Fails)
{
    auto input = makeFP32({1, 4});
    auto output = makeFP32({1, 4});
    auto weight = makeFP32({4, 4});
    auto ctx = makeCPUContext();

    ShortConv1dStage::Params p;
    p.kernel = &g_cpu_conv;
    p.input = input.get();
    p.output = output.get();
    p.weight = weight.get();
    p.seq_len = 1;
    p.channels = 4;
    p.kernel_size = 4;
    p.conv_state = nullptr; // No state!

    ShortConv1dStage stage(p);
    EXPECT_FALSE(stage.execute(ctx.get()));
}

// ============================================================================
// C3-C5: GDNRecurrenceStage Tests
// ============================================================================

TEST(Test__GDNKernels, Recurrence_TypeAndBackend)
{
    GDNRecurrenceStage::Params p;
    p.kernel = &g_cpu_gdn;
    GDNRecurrenceStage stage(p);
    EXPECT_EQ(stage.type(), ComputeStageType::GDN_RECURRENCE);
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
}

TEST(Test__GDNKernels, Recurrence_NullPointers_Fails)
{
    auto ctx = makeCPUContext();
    GDNRecurrenceStage::Params p;
    p.kernel = &g_cpu_gdn;
    GDNRecurrenceStage stage(p);
    EXPECT_FALSE(stage.execute(ctx.get()));
}

TEST(Test__GDNKernels, Recurrence_Decode_ZeroState)
{
    // With zero initial state and specific inputs, verify output
    const int n_heads = 2;
    const int d_k = 4;
    const int d_v = 4;

    // Q, K, V: [1, n_heads * dim]
    auto Q = makeFP32Random({1, static_cast<size_t>(n_heads * d_k)}, 0.0f, 0.5f, 1);
    auto K = makeFP32Random({1, static_cast<size_t>(n_heads * d_k)}, 0.0f, 0.5f, 2);
    auto V = makeFP32Random({1, static_cast<size_t>(n_heads * d_v)}, 0.0f, 0.5f, 3);
    auto alpha = makeFP32Const({1, static_cast<size_t>(n_heads)}, 0.0f); // softplus(0 + dtbias)
    auto beta = makeFP32Const({1, static_cast<size_t>(n_heads)}, 0.0f);  // sigmoid(0) = 0.5
    auto A_log = makeFP32Const({static_cast<size_t>(n_heads)}, 0.0f);    // -exp(0) = -1
    auto dt_bias = makeFP32Const({static_cast<size_t>(n_heads)}, 0.0f);
    auto output = makeFP32({1, static_cast<size_t>(n_heads * d_v)});

    // Zero state
    std::vector<float> state(n_heads * d_k * d_v, 0.0f);

    auto ctx = makeCPUContext();

    GDNRecurrenceStage::Params p;
    p.kernel = &g_cpu_gdn;
    p.Q = Q.get();
    p.K = K.get();
    p.V = V.get();
    p.alpha = alpha.get();
    p.beta = beta.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias.get();
    p.output = output.get();
    p.recurrence_state = state.data();
    p.seq_len = 1;
    p.n_heads = n_heads;
    p.d_k = d_k;
    p.d_v = d_v;
    p.use_qk_l2norm = false; // Disable for deterministic test

    GDNRecurrenceStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // With zero state:
    // g = -exp(0) * softplus(0 + 0) = -1 * softplus(0) = -1 * ln(2) ≈ -0.6931
    // decay = exp(g) ≈ 0.5
    // S is zero, so after decay still zero
    // kv_mem = S * k = 0
    // delta = (v - 0) * sigmoid(0) = v * 0.5
    // S += outer(k, v * 0.5)
    // o = S * q = outer(k, v*0.5) * q = k*(q·k_after_l2norm)*v*0.5... but with use_qk_l2norm=false
    // Actually: o_h[v] = sum_j S[j,v] * q_h[j] where S[j,v] = k[j] * delta[v]
    //         = sum_j k[j] * v[v] * 0.5 * q[j] * scale
    //         = (sum_j k[j] * q[j] * scale) * v[v] * 0.5
    // This is a scalar * v[v] * 0.5

    // Just verify output is not all zeros (state was updated)
    const float *out = output->data();
    float sum = 0.0f;
    for (int i = 0; i < n_heads * d_v; ++i)
        sum += std::abs(out[i]);
    EXPECT_GT(sum, 0.0f) << "Output should not be all zeros after recurrence with non-zero inputs";

    // Verify state was updated (should no longer be zero)
    float state_sum = 0.0f;
    for (size_t i = 0; i < state.size(); ++i)
        state_sum += std::abs(state[i]);
    EXPECT_GT(state_sum, 0.0f) << "Recurrence state should be updated";
}

TEST(Test__GDNKernels, Recurrence_Decode_StateDecay)
{
    // Verify that with beta=0 (no update), the state decays exponentially
    const int n_heads = 1;
    const int d_k = 2;
    const int d_v = 2;

    auto Q = makeFP32Const({1, static_cast<size_t>(n_heads * d_k)}, 1.0f);
    auto K = makeFP32Const({1, static_cast<size_t>(n_heads * d_k)}, 1.0f);
    auto V = makeFP32Const({1, static_cast<size_t>(n_heads * d_v)}, 1.0f);
    auto alpha = makeFP32Const({1, 1}, 0.0f);
    auto beta = makeFP32Const({1, 1}, -100.0f); // sigmoid(-100) ≈ 0, no update
    // GGUF stores -exp(A_log_raw), so for raw_param=0: stored = -exp(0) = -1.0
    auto A_log = makeFP32Const({1}, -1.0f); // GGUF pre-exponentiated: -exp(0) = -1
    auto dt_bias = makeFP32Const({1}, 0.0f);
    auto output = makeFP32({1, static_cast<size_t>(n_heads * d_v)});

    // Initial state: all 1.0
    std::vector<float> state(d_k * d_v, 1.0f);

    auto ctx = makeCPUContext();

    GDNRecurrenceStage::Params p;
    p.kernel = &g_cpu_gdn;
    p.Q = Q.get();
    p.K = K.get();
    p.V = V.get();
    p.alpha = alpha.get();
    p.beta = beta.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias.get();
    p.output = output.get();
    p.recurrence_state = state.data();
    p.seq_len = 1;
    p.n_heads = n_heads;
    p.d_k = d_k;
    p.d_v = d_v;
    p.use_qk_l2norm = false;

    GDNRecurrenceStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // g = A_log * softplus(0) = -1.0 * ln(2) ≈ -0.6931
    // decay = exp(g) ≈ 0.5
    // beta_sig ≈ 0, so delta ≈ 0 and state is only decayed
    // State should be approximately 0.5 (decayed from 1.0)
    const float expected_decay = std::exp(-std::log(2.0f)); // 0.5
    for (int i = 0; i < d_k * d_v; ++i)
    {
        EXPECT_NEAR(state[i], expected_decay, 0.01f)
            << "State[" << i << "] should have decayed by exp(g)";
    }
}

TEST(Test__GDNKernels, Recurrence_Decode_SingleHeadReference)
{
    // Reference test: manually compute the recurrence for a single head
    // and compare with the stage output
    const int n_heads = 1;
    const int d_k = 3;
    const int d_v = 3;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    // Hand-chosen values
    std::vector<float> q_data = {1.0f, 0.0f, 0.0f}; // unit vector along dim 0
    std::vector<float> k_data = {0.0f, 1.0f, 0.0f}; // unit vector along dim 1
    std::vector<float> v_data = {0.5f, 0.5f, 0.5f}; // uniform

    auto Q = makeFP32({1, 3}, q_data.data());
    auto K = makeFP32({1, 3}, k_data.data());
    auto V = makeFP32({1, 3}, v_data.data());

    // Set gates to produce known values
    // alpha=100 → softplus(100+0) ≈ 100, g = A_log*softplus = -1.0*100 = -100 → decay ≈ 0
    // This effectively zeroes the old state before update
    auto alpha = makeFP32Const({1, 1}, 100.0f);
    // beta=100 → sigmoid(100) ≈ 1.0
    auto beta = makeFP32Const({1, 1}, 100.0f);
    // GGUF stores -exp(A_log_raw), so for raw_param=0: stored = -exp(0) = -1.0
    auto A_log = makeFP32Const({1}, -1.0f);
    auto dt_bias = makeFP32Const({1}, 0.0f);
    auto output = makeFP32({1, 3});

    std::vector<float> state(d_k * d_v, 0.5f); // Initial state

    auto ctx = makeCPUContext();

    GDNRecurrenceStage::Params p;
    p.kernel = &g_cpu_gdn;
    p.Q = Q.get();
    p.K = K.get();
    p.V = V.get();
    p.alpha = alpha.get();
    p.beta = beta.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias.get();
    p.output = output.get();
    p.recurrence_state = state.data();
    p.seq_len = 1;
    p.n_heads = 1;
    p.d_k = d_k;
    p.d_v = d_v;
    p.use_qk_l2norm = false;

    GDNRecurrenceStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // Manual computation:
    // g = A_log * softplus(alpha + dt_bias) = -1.0 * softplus(100) ≈ -100
    // decay = exp(-100) ≈ 0
    // S decayed ≈ 0 (state effectively zeroed)
    // kv_mem = S * k = 0 (S is zeroed, k=[0,1,0])
    // delta = (v - kv_mem) * beta = (0.5,0.5,0.5) * 1.0 = (0.5,0.5,0.5)
    // S += outer(k, delta) = k=[0,1,0]^T * delta=[0.5,0.5,0.5]
    //   S[0,:] = 0, S[1,:] = [0.5,0.5,0.5], S[2,:] = 0
    // q_scaled = q * scale = [scale, 0, 0]
    // o = S * q_scaled = sum_j S[j,:] * q_scaled[j] = S[0,:]*scale + 0 + 0 = 0
    const float *out = output->data();
    for (int v = 0; v < d_v; ++v)
    {
        EXPECT_NEAR(out[v], 0.0f, 0.01f)
            << "Output[" << v << "] — q selects dim 0 of S which is 0 (k was along dim 1)";
    }
}

TEST(Test__GDNKernels, Recurrence_Prefill_MatchesSequentialDecode)
{
    // Prefill of seq_len tokens should give identical output to running
    // seq_len sequential decode steps
    const int n_heads = 2;
    const int d_k = 4;
    const int d_v = 4;
    const int seq_len = 8;

    auto Q = makeFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * d_k)}, 0.0f, 0.3f, 100);
    auto K = makeFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * d_k)}, 0.0f, 0.3f, 200);
    auto V = makeFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * d_v)}, 0.0f, 0.3f, 300);
    auto alpha = makeFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads)}, 0.0f, 0.5f, 400);
    auto beta_raw = makeFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads)}, 0.0f, 0.5f, 500);
    auto A_log = makeFP32Const({static_cast<size_t>(n_heads)}, 1.0f);
    auto dt_bias = makeFP32Const({static_cast<size_t>(n_heads)}, 0.5f);

    // Path 1: Prefill
    auto output_prefill = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * d_v)});
    std::vector<float> state_prefill(n_heads * d_k * d_v, 0.0f);

    auto ctx = makeCPUContext();
    {
        GDNRecurrenceStage::Params p;
        p.kernel = &g_cpu_gdn;
        p.Q = Q.get();
        p.K = K.get();
        p.V = V.get();
        p.alpha = alpha.get();
        p.beta = beta_raw.get();
        p.A_log = A_log.get();
        p.dt_bias = dt_bias.get();
        p.output = output_prefill.get();
        p.recurrence_state = state_prefill.data();
        p.seq_len = seq_len;
        p.n_heads = n_heads;
        p.d_k = d_k;
        p.d_v = d_v;
        p.use_qk_l2norm = false;

        GDNRecurrenceStage stage(p);
        ASSERT_TRUE(stage.execute(ctx.get()));
    }

    // Path 2: Sequential decode steps
    auto output_decode = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * d_v)});
    std::vector<float> state_decode(n_heads * d_k * d_v, 0.0f);

    const float *q_data = Q->data();
    const float *k_data = K->data();
    const float *v_data = V->data();
    const float *a_data = alpha->data();
    const float *b_data = beta_raw->data();

    for (int t = 0; t < seq_len; ++t)
    {
        // Create single-step tensors pointing to time t
        auto q_t = makeFP32({1, static_cast<size_t>(n_heads * d_k)}, q_data + t * n_heads * d_k);
        auto k_t = makeFP32({1, static_cast<size_t>(n_heads * d_k)}, k_data + t * n_heads * d_k);
        auto v_t = makeFP32({1, static_cast<size_t>(n_heads * d_v)}, v_data + t * n_heads * d_v);
        auto a_t = makeFP32({1, static_cast<size_t>(n_heads)}, a_data + t * n_heads);
        auto b_t = makeFP32({1, static_cast<size_t>(n_heads)}, b_data + t * n_heads);

        auto out_t = makeFP32({1, static_cast<size_t>(n_heads * d_v)});

        GDNRecurrenceStage::Params p;
        p.kernel = &g_cpu_gdn;
        p.Q = q_t.get();
        p.K = k_t.get();
        p.V = v_t.get();
        p.alpha = a_t.get();
        p.beta = b_t.get();
        p.A_log = A_log.get();
        p.dt_bias = dt_bias.get();
        p.output = out_t.get();
        p.recurrence_state = state_decode.data();
        p.seq_len = 1;
        p.n_heads = n_heads;
        p.d_k = d_k;
        p.d_v = d_v;
        p.use_qk_l2norm = false;

        GDNRecurrenceStage stage(p);
        ASSERT_TRUE(stage.execute(ctx.get()));

        // Copy output for timestep t
        std::memcpy(output_decode->mutable_data() + t * n_heads * d_v,
                    out_t->data(), n_heads * d_v * sizeof(float));
    }

    // Compare: prefill output should match sequential decode output
    const float *pf = output_prefill->data();
    const float *dc = output_decode->data();
    for (int i = 0; i < seq_len * n_heads * d_v; ++i)
    {
        EXPECT_NEAR(pf[i], dc[i], 1e-4f)
            << "Mismatch at index " << i << " (t=" << i / (n_heads * d_v) << ")";
    }

    // Final states should also match
    for (size_t i = 0; i < state_prefill.size(); ++i)
    {
        EXPECT_NEAR(state_prefill[i], state_decode[i], 1e-4f)
            << "State mismatch at index " << i;
    }
}

TEST(Test__GDNKernels, Recurrence_ChunkForwardFromNonZeroStateMatchesSequentialDecode)
{
    const int n_heads = 2;
    const int d_k = 4;
    const int d_v = 4;
    const int prompt_len = 5;
    const int suffix_len = 2;
    const int total_len = prompt_len + suffix_len;
    const int qk_stride = n_heads * d_k;
    const int v_stride = n_heads * d_v;

    auto Q = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(qk_stride)}, 0.0f, 0.3f, 110);
    auto K = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(qk_stride)}, 0.0f, 0.3f, 210);
    auto V = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(v_stride)}, 0.0f, 0.3f, 310);
    auto alpha = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(n_heads)}, 0.0f, 0.5f, 410);
    auto beta_raw = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(n_heads)}, 0.0f, 0.5f, 510);
    auto A_log = makeFP32Const({static_cast<size_t>(n_heads)}, 1.0f);
    auto dt_bias = makeFP32Const({static_cast<size_t>(n_heads)}, 0.5f);

    auto prompt_output = makeFP32({static_cast<size_t>(prompt_len), static_cast<size_t>(v_stride)});
    auto suffix_output = makeFP32({static_cast<size_t>(suffix_len), static_cast<size_t>(v_stride)});
    std::vector<float> chunk_state(static_cast<size_t>(n_heads * d_k * d_v), 0.0f);

    ASSERT_TRUE(g_cpu_gdn.chunk_forward(
        Q->data(), K->data(), V->data(),
        alpha->data(), beta_raw->data(),
        A_log->data(), dt_bias->data(),
        prompt_output->mutable_data(), chunk_state.data(),
        prompt_len, n_heads, d_k, d_v,
        /*chunk_size=*/64, /*use_qk_l2norm=*/false));

    ASSERT_TRUE(g_cpu_gdn.chunk_forward(
        Q->data() + static_cast<size_t>(prompt_len) * qk_stride,
        K->data() + static_cast<size_t>(prompt_len) * qk_stride,
        V->data() + static_cast<size_t>(prompt_len) * v_stride,
        alpha->data() + static_cast<size_t>(prompt_len) * n_heads,
        beta_raw->data() + static_cast<size_t>(prompt_len) * n_heads,
        A_log->data(), dt_bias->data(),
        suffix_output->mutable_data(), chunk_state.data(),
        suffix_len, n_heads, d_k, d_v,
        /*chunk_size=*/64, /*use_qk_l2norm=*/false));

    auto ref_output = makeFP32({static_cast<size_t>(total_len), static_cast<size_t>(v_stride)});
    std::vector<float> ref_state(static_cast<size_t>(n_heads * d_k * d_v), 0.0f);
    for (int t = 0; t < total_len; ++t)
    {
        ASSERT_TRUE(g_cpu_gdn.recurrent_step(
            Q->data() + static_cast<size_t>(t) * qk_stride,
            K->data() + static_cast<size_t>(t) * qk_stride,
            V->data() + static_cast<size_t>(t) * v_stride,
            alpha->data() + static_cast<size_t>(t) * n_heads,
            beta_raw->data() + static_cast<size_t>(t) * n_heads,
            A_log->data(), dt_bias->data(),
            ref_output->mutable_data() + static_cast<size_t>(t) * v_stride,
            ref_state.data(),
            n_heads, d_k, d_v,
            /*use_qk_l2norm=*/false));
    }

    const float *suffix = suffix_output->data();
    const float *ref = ref_output->data() + static_cast<size_t>(prompt_len) * v_stride;
    for (int i = 0; i < suffix_len * v_stride; ++i)
    {
        EXPECT_NEAR(suffix[i], ref[i], 1e-4f)
            << "suffix output mismatch at index " << i;
    }
    for (size_t i = 0; i < ref_state.size(); ++i)
    {
        EXPECT_NEAR(chunk_state[i], ref_state[i], 1e-4f)
            << "state mismatch at index " << i;
    }
}

TEST(Test__GDNKernels, ShortConv_PaddedPrefillStateMatchesUnpaddedDecode)
{
    const int real_len = 5;
    const int bucket_len = 8;
    const int channels = 3;
    const int kernel_size = 4;
    const int state_len = kernel_size - 1;

    std::vector<float> bucket_input(static_cast<size_t>(bucket_len * channels));
    for (int t = 0; t < bucket_len; ++t)
    {
        for (int c = 0; c < channels; ++c)
        {
            bucket_input[static_cast<size_t>(t * channels + c)] =
                t < real_len ? 0.1f * static_cast<float>(1 + t * channels + c)
                             : 100.0f + static_cast<float>(t * channels + c);
        }
    }
    std::vector<float> real_input(bucket_input.begin(), bucket_input.begin() + real_len * channels);
    std::vector<float> decode_input = {0.33f, -0.21f, 0.17f};
    std::vector<float> weight(static_cast<size_t>(channels * kernel_size));
    std::vector<float> bias(static_cast<size_t>(channels));
    for (int i = 0; i < channels * kernel_size; ++i)
        weight[static_cast<size_t>(i)] = 0.05f * static_cast<float>((i % 5) + 1);
    for (int c = 0; c < channels; ++c)
        bias[static_cast<size_t>(c)] = -0.02f * static_cast<float>(c + 1);

    auto padded_in = makeFP32({static_cast<size_t>(bucket_len), static_cast<size_t>(channels)}, bucket_input.data());
    auto padded_out = makeFP32({static_cast<size_t>(bucket_len), static_cast<size_t>(channels)});
    auto real_in = makeFP32({static_cast<size_t>(real_len), static_cast<size_t>(channels)}, real_input.data());
    auto real_out = makeFP32({static_cast<size_t>(real_len), static_cast<size_t>(channels)});
    auto decode_in = makeFP32({1, static_cast<size_t>(channels)}, decode_input.data());
    auto padded_decode_out = makeFP32({1, static_cast<size_t>(channels)});
    auto ref_decode_out = makeFP32({1, static_cast<size_t>(channels)});
    auto weight_t = makeFP32({static_cast<size_t>(channels), static_cast<size_t>(kernel_size)}, weight.data());
    auto bias_t = makeFP32({static_cast<size_t>(channels)}, bias.data());
    std::vector<float> padded_state(static_cast<size_t>(channels * state_len), 0.0f);
    std::vector<float> ref_state(static_cast<size_t>(channels * state_len), 0.0f);
    auto ctx = makeCPUContext();

    ShortConv1dStage::Params padded_params;
    padded_params.device_id = DeviceId::cpu();
    padded_params.input = padded_in.get();
    padded_params.output = padded_out.get();
    padded_params.weight = weight_t.get();
    padded_params.bias = bias_t.get();
    padded_params.conv_state = padded_state.data();
    padded_params.seq_len = bucket_len;
    padded_params.channels = channels;
    padded_params.kernel_size = kernel_size;
    padded_params.kernel = &g_cpu_conv;
    ShortConv1dStage padded_stage(padded_params);
    IComputeStage::PrefillReplayParams replay;
    replay.real_seq_len = real_len;
    replay.bucket_seq_len = bucket_len;
    padded_stage.updatePrefillReplayParams(replay);
    ASSERT_TRUE(padded_stage.execute(ctx.get()));

    ShortConv1dStage::Params ref_params = padded_params;
    ref_params.input = real_in.get();
    ref_params.output = real_out.get();
    ref_params.conv_state = ref_state.data();
    ref_params.seq_len = real_len;
    ShortConv1dStage ref_stage(ref_params);
    ASSERT_TRUE(ref_stage.execute(ctx.get()));

    ShortConv1dStage::Params padded_decode = padded_params;
    padded_decode.input = decode_in.get();
    padded_decode.output = padded_decode_out.get();
    padded_decode.seq_len = 1;
    ShortConv1dStage padded_decode_stage(padded_decode);
    ASSERT_TRUE(padded_decode_stage.execute(ctx.get()));

    ShortConv1dStage::Params ref_decode = ref_params;
    ref_decode.input = decode_in.get();
    ref_decode.output = ref_decode_out.get();
    ref_decode.seq_len = 1;
    ShortConv1dStage ref_decode_stage(ref_decode);
    ASSERT_TRUE(ref_decode_stage.execute(ctx.get()));

    const float *padded_decode_data = padded_decode_out->data();
    const float *ref_decode_data = ref_decode_out->data();
    for (int c = 0; c < channels; ++c)
    {
        EXPECT_NEAR(padded_decode_data[c], ref_decode_data[c], 1e-6f)
            << "decode output mismatch at channel " << c;
    }
    for (size_t i = 0; i < ref_state.size(); ++i)
    {
        EXPECT_NEAR(padded_state[i], ref_state[i], 1e-6f)
            << "conv state mismatch at index " << i;
    }

    const float *padded_prefill = padded_out->data();
    for (int i = real_len * channels; i < bucket_len * channels; ++i)
        EXPECT_EQ(padded_prefill[i], 0.0f) << "padded output row must be inert";
}

TEST(Test__GDNKernels, ShortConv_ChunkForwardFromNonZeroStateMatchesSequentialDecode)
{
    const int channels = 19;
    const int kernel_size = 4;
    const int state_len = kernel_size - 1;
    const int prompt_len = 5;
    const int suffix_len = 2;
    const int next_len = 1;
    const int total_len = prompt_len + suffix_len + next_len;

    auto input = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(channels)}, 0.0f, 0.2f, 1401);
    auto weight = makeFP32Random({static_cast<size_t>(channels), static_cast<size_t>(kernel_size)}, 0.0f, 0.06f, 1402);
    auto bias = makeFP32Random({static_cast<size_t>(channels)}, 0.0f, 0.02f, 1403);
    auto prompt_out = makeFP32({static_cast<size_t>(prompt_len), static_cast<size_t>(channels)});
    auto suffix_out = makeFP32({static_cast<size_t>(suffix_len), static_cast<size_t>(channels)});
    auto next_out = makeFP32({static_cast<size_t>(next_len), static_cast<size_t>(channels)});
    auto ref_out = makeFP32({static_cast<size_t>(total_len), static_cast<size_t>(channels)});
    std::vector<float> chunk_state(static_cast<size_t>(channels * state_len), 0.0f);
    std::vector<float> ref_state(static_cast<size_t>(channels * state_len), 0.0f);

    ASSERT_TRUE(g_cpu_conv.forward(
        input->data(), weight->data(), bias->data(),
        prompt_out->mutable_data(), chunk_state.data(),
        prompt_len, channels, kernel_size,
        /*apply_silu=*/true));
    ASSERT_TRUE(g_cpu_conv.forward(
        input->data() + static_cast<size_t>(prompt_len) * channels,
        weight->data(), bias->data(),
        suffix_out->mutable_data(), chunk_state.data(),
        suffix_len, channels, kernel_size,
        /*apply_silu=*/true));
    const int next_row = prompt_len + suffix_len;
    ASSERT_TRUE(g_cpu_conv.forward(
        input->data() + static_cast<size_t>(next_row) * channels,
        weight->data(), bias->data(),
        next_out->mutable_data(), chunk_state.data(),
        next_len, channels, kernel_size,
        /*apply_silu=*/true));

    for (int t = 0; t < total_len; ++t)
    {
        ASSERT_TRUE(g_cpu_conv.forward(
            input->data() + static_cast<size_t>(t) * channels,
            weight->data(), bias->data(),
            ref_out->mutable_data() + static_cast<size_t>(t) * channels,
            ref_state.data(),
            1, channels, kernel_size,
            /*apply_silu=*/true));
    }

    const float *suffix = suffix_out->data();
    const float *next = next_out->data();
    const float *ref = ref_out->data();
    for (int i = 0; i < suffix_len * channels; ++i)
    {
        EXPECT_NEAR(suffix[i], ref[static_cast<size_t>(prompt_len) * channels + i], 1e-6f)
            << "suffix output mismatch at index " << i;
    }
    for (int c = 0; c < channels; ++c)
    {
        EXPECT_NEAR(next[c], ref[static_cast<size_t>(next_row) * channels + c], 1e-6f)
            << "next output mismatch at channel " << c;
    }
    for (size_t i = 0; i < ref_state.size(); ++i)
    {
        EXPECT_NEAR(chunk_state[i], ref_state[i], 1e-6f)
            << "conv state mismatch at index " << i;
    }
}

TEST(Test__GDNKernels, Recurrence_PaddedPrefillStateMatchesUnpaddedDecode)
{
    const int real_len = 5;
    const int bucket_len = 8;
    const int n_heads = 2;
    const int d_k = 4;
    const int d_v = 3;
    const int qk_dim = n_heads * d_k;
    const int v_dim = n_heads * d_v;

    auto fill_sequence = [](std::vector<float> &values, float scale, int real_len, int row_width)
    {
        for (int row = 0; row < static_cast<int>(values.size()) / row_width; ++row)
        {
            for (int col = 0; col < row_width; ++col)
            {
                values[static_cast<size_t>(row * row_width + col)] =
                    row < real_len ? scale * static_cast<float>((row + 1) * (col + 2))
                                   : 50.0f + scale * static_cast<float>(row * row_width + col);
            }
        }
    };

    std::vector<float> Q_bucket(static_cast<size_t>(bucket_len * qk_dim));
    std::vector<float> K_bucket(static_cast<size_t>(bucket_len * qk_dim));
    std::vector<float> V_bucket(static_cast<size_t>(bucket_len * v_dim));
    std::vector<float> alpha_bucket(static_cast<size_t>(bucket_len * n_heads));
    std::vector<float> beta_bucket(static_cast<size_t>(bucket_len * n_heads));
    fill_sequence(Q_bucket, 0.011f, real_len, qk_dim);
    fill_sequence(K_bucket, -0.009f, real_len, qk_dim);
    fill_sequence(V_bucket, 0.013f, real_len, v_dim);
    fill_sequence(alpha_bucket, 0.017f, real_len, n_heads);
    fill_sequence(beta_bucket, -0.015f, real_len, n_heads);

    std::vector<float> Q_real(Q_bucket.begin(), Q_bucket.begin() + real_len * qk_dim);
    std::vector<float> K_real(K_bucket.begin(), K_bucket.begin() + real_len * qk_dim);
    std::vector<float> V_real(V_bucket.begin(), V_bucket.begin() + real_len * v_dim);
    std::vector<float> alpha_real(alpha_bucket.begin(), alpha_bucket.begin() + real_len * n_heads);
    std::vector<float> beta_real(beta_bucket.begin(), beta_bucket.begin() + real_len * n_heads);
    std::vector<float> Q_decode(qk_dim), K_decode(qk_dim), V_decode(v_dim), alpha_decode(n_heads), beta_decode(n_heads);
    for (int i = 0; i < qk_dim; ++i)
    {
        Q_decode[static_cast<size_t>(i)] = 0.02f * static_cast<float>(i + 1);
        K_decode[static_cast<size_t>(i)] = -0.018f * static_cast<float>(i + 2);
    }
    for (int i = 0; i < v_dim; ++i)
        V_decode[static_cast<size_t>(i)] = 0.014f * static_cast<float>(i + 3);
    for (int h = 0; h < n_heads; ++h)
    {
        alpha_decode[static_cast<size_t>(h)] = 0.07f * static_cast<float>(h + 1);
        beta_decode[static_cast<size_t>(h)] = -0.04f * static_cast<float>(h + 1);
    }

    auto A_log = makeFP32Const({static_cast<size_t>(n_heads)}, -0.5f);
    auto dt_bias = makeFP32Const({static_cast<size_t>(n_heads)}, 0.1f);
    auto padded_q = makeFP32({static_cast<size_t>(bucket_len), static_cast<size_t>(qk_dim)}, Q_bucket.data());
    auto padded_k = makeFP32({static_cast<size_t>(bucket_len), static_cast<size_t>(qk_dim)}, K_bucket.data());
    auto padded_v = makeFP32({static_cast<size_t>(bucket_len), static_cast<size_t>(v_dim)}, V_bucket.data());
    auto padded_alpha = makeFP32({static_cast<size_t>(bucket_len), static_cast<size_t>(n_heads)}, alpha_bucket.data());
    auto padded_beta = makeFP32({static_cast<size_t>(bucket_len), static_cast<size_t>(n_heads)}, beta_bucket.data());
    auto padded_out = makeFP32({static_cast<size_t>(bucket_len), static_cast<size_t>(v_dim)});
    auto real_q = makeFP32({static_cast<size_t>(real_len), static_cast<size_t>(qk_dim)}, Q_real.data());
    auto real_k = makeFP32({static_cast<size_t>(real_len), static_cast<size_t>(qk_dim)}, K_real.data());
    auto real_v = makeFP32({static_cast<size_t>(real_len), static_cast<size_t>(v_dim)}, V_real.data());
    auto real_alpha = makeFP32({static_cast<size_t>(real_len), static_cast<size_t>(n_heads)}, alpha_real.data());
    auto real_beta = makeFP32({static_cast<size_t>(real_len), static_cast<size_t>(n_heads)}, beta_real.data());
    auto real_out = makeFP32({static_cast<size_t>(real_len), static_cast<size_t>(v_dim)});
    auto decode_q = makeFP32({1, static_cast<size_t>(qk_dim)}, Q_decode.data());
    auto decode_k = makeFP32({1, static_cast<size_t>(qk_dim)}, K_decode.data());
    auto decode_v = makeFP32({1, static_cast<size_t>(v_dim)}, V_decode.data());
    auto decode_alpha = makeFP32({1, static_cast<size_t>(n_heads)}, alpha_decode.data());
    auto decode_beta = makeFP32({1, static_cast<size_t>(n_heads)}, beta_decode.data());
    auto padded_decode_out = makeFP32({1, static_cast<size_t>(v_dim)});
    auto ref_decode_out = makeFP32({1, static_cast<size_t>(v_dim)});
    std::vector<float> padded_state(static_cast<size_t>(n_heads * d_k * d_v), 0.0f);
    std::vector<float> ref_state(static_cast<size_t>(n_heads * d_k * d_v), 0.0f);
    auto ctx = makeCPUContext();

    GDNRecurrenceStage::Params padded_params;
    padded_params.device_id = DeviceId::cpu();
    padded_params.Q = padded_q.get();
    padded_params.K = padded_k.get();
    padded_params.V = padded_v.get();
    padded_params.alpha = padded_alpha.get();
    padded_params.beta = padded_beta.get();
    padded_params.A_log = A_log.get();
    padded_params.dt_bias = dt_bias.get();
    padded_params.output = padded_out.get();
    padded_params.recurrence_state = padded_state.data();
    padded_params.seq_len = bucket_len;
    padded_params.n_heads = n_heads;
    padded_params.d_k = d_k;
    padded_params.d_v = d_v;
    padded_params.chunk_size = 64;
    padded_params.kernel = &g_cpu_gdn;
    GDNRecurrenceStage padded_stage(padded_params);
    IComputeStage::PrefillReplayParams replay;
    replay.real_seq_len = real_len;
    replay.bucket_seq_len = bucket_len;
    padded_stage.updatePrefillReplayParams(replay);
    ASSERT_TRUE(padded_stage.execute(ctx.get()));

    GDNRecurrenceStage::Params ref_params = padded_params;
    ref_params.Q = real_q.get();
    ref_params.K = real_k.get();
    ref_params.V = real_v.get();
    ref_params.alpha = real_alpha.get();
    ref_params.beta = real_beta.get();
    ref_params.output = real_out.get();
    ref_params.recurrence_state = ref_state.data();
    ref_params.seq_len = real_len;
    GDNRecurrenceStage ref_stage(ref_params);
    ASSERT_TRUE(ref_stage.execute(ctx.get()));

    GDNRecurrenceStage::Params padded_decode = padded_params;
    padded_decode.Q = decode_q.get();
    padded_decode.K = decode_k.get();
    padded_decode.V = decode_v.get();
    padded_decode.alpha = decode_alpha.get();
    padded_decode.beta = decode_beta.get();
    padded_decode.output = padded_decode_out.get();
    padded_decode.seq_len = 1;
    GDNRecurrenceStage padded_decode_stage(padded_decode);
    ASSERT_TRUE(padded_decode_stage.execute(ctx.get()));

    GDNRecurrenceStage::Params ref_decode = padded_decode;
    ref_decode.output = ref_decode_out.get();
    ref_decode.recurrence_state = ref_state.data();
    GDNRecurrenceStage ref_decode_stage(ref_decode);
    ASSERT_TRUE(ref_decode_stage.execute(ctx.get()));

    const float *padded_decode_data = padded_decode_out->data();
    const float *ref_decode_data = ref_decode_out->data();
    for (int i = 0; i < v_dim; ++i)
    {
        EXPECT_NEAR(padded_decode_data[i], ref_decode_data[i], 1e-5f)
            << "decode output mismatch at column " << i;
    }
    for (size_t i = 0; i < ref_state.size(); ++i)
    {
        EXPECT_NEAR(padded_state[i], ref_state[i], 1e-5f)
            << "recurrence state mismatch at index " << i;
    }

    const float *padded_prefill = padded_out->data();
    for (int i = real_len * v_dim; i < bucket_len * v_dim; ++i)
        EXPECT_EQ(padded_prefill[i], 0.0f) << "padded recurrence output row must be inert";
}

#ifdef HAVE_ROCM
TEST(Test__GDNKernels, ROCmPrefillMatchesSequentialDecodeQwen35Shape)
{
    int device_count = 0;
    ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
    if (device_count <= 0)
        GTEST_SKIP() << "No ROCm device available";

    const DeviceId device = DeviceId::rocm(0);
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    const int n_heads = 16;
    const int d_k = 128;
    const int d_v = 128;
    const int prefill_len = 37;
    const int total_len = prefill_len + 1;
    const size_t qk_elems = static_cast<size_t>(total_len) * n_heads * d_k;
    const size_t v_elems = static_cast<size_t>(total_len) * n_heads * d_v;
    const size_t gate_elems = static_cast<size_t>(total_len) * n_heads;
    const size_t output_elems = static_cast<size_t>(total_len) * n_heads * d_v;

    // Use production head dimensions with deterministic small-magnitude inputs
    // so the ROCm prefill recurrence can be compared against decode exactly.
    auto Q = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(n_heads * d_k)}, 0.0f, 0.03f, 1101);
    auto K = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(n_heads * d_k)}, 0.0f, 0.03f, 1102);
    auto V = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(n_heads * d_v)}, 0.0f, 0.03f, 1103);
    auto alpha = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(n_heads)}, -0.15f, 0.2f, 1104);
    auto beta = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(n_heads)}, 0.0f, 0.2f, 1105);
    auto A_log = makeFP32Const({static_cast<size_t>(n_heads)}, -0.5f);
    auto dt_bias = makeFP32Const({static_cast<size_t>(n_heads)}, 0.1f);
    auto prefill_out = makeFP32({static_cast<size_t>(total_len), static_cast<size_t>(n_heads * d_v)});
    auto decode_out = makeFP32({static_cast<size_t>(total_len), static_cast<size_t>(n_heads * d_v)});

    ASSERT_EQ(Q->numel(), qk_elems);
    ASSERT_EQ(K->numel(), qk_elems);
    ASSERT_EQ(V->numel(), v_elems);
    ASSERT_EQ(alpha->numel(), gate_elems);
    ASSERT_EQ(beta->numel(), gate_elems);
    ASSERT_EQ(prefill_out->numel(), output_elems);
    ASSERT_EQ(decode_out->numel(), output_elems);

    // Directly exercise the ROCm kernel implementation, bypassing stage
    // orchestration so this test only covers recurrence state compatibility.
    ASSERT_TRUE(Q->ensureOnDevice(device));
    ASSERT_TRUE(K->ensureOnDevice(device));
    ASSERT_TRUE(V->ensureOnDevice(device));
    ASSERT_TRUE(alpha->ensureOnDevice(device));
    ASSERT_TRUE(beta->ensureOnDevice(device));
    ASSERT_TRUE(A_log->ensureOnDevice(device));
    ASSERT_TRUE(dt_bias->ensureOnDevice(device));
    ASSERT_TRUE(prefill_out->allocateOnDevice(device));
    ASSERT_TRUE(decode_out->allocateOnDevice(device));

    auto *d_Q = static_cast<const float *>(Q->gpu_data_ptr());
    auto *d_K = static_cast<const float *>(K->gpu_data_ptr());
    auto *d_V = static_cast<const float *>(V->gpu_data_ptr());
    auto *d_alpha = static_cast<const float *>(alpha->gpu_data_ptr());
    auto *d_beta = static_cast<const float *>(beta->gpu_data_ptr());
    auto *d_A_log = static_cast<const float *>(A_log->gpu_data_ptr());
    auto *d_dt_bias = static_cast<const float *>(dt_bias->gpu_data_ptr());
    auto *d_prefill = static_cast<float *>(prefill_out->gpu_data_ptr());
    auto *d_decode = static_cast<float *>(decode_out->gpu_data_ptr());

    ASSERT_NE(d_Q, nullptr);
    ASSERT_NE(d_K, nullptr);
    ASSERT_NE(d_V, nullptr);
    ASSERT_NE(d_alpha, nullptr);
    ASSERT_NE(d_beta, nullptr);
    ASSERT_NE(d_A_log, nullptr);
    ASSERT_NE(d_dt_bias, nullptr);
    ASSERT_NE(d_prefill, nullptr);
    ASSERT_NE(d_decode, nullptr);

    // Path 1: prefill the prompt history, then decode one more token using the
    // same ROCm kernel instance. This locks down the hidden GPU recurrence
    // state that production decode consumes after prefill.
    ROCmGatedDeltaNet prefill_kernel(0);
    ASSERT_TRUE(prefill_kernel.chunk_forward(
        d_Q, d_K, d_V, d_alpha, d_beta, d_A_log, d_dt_bias,
        d_prefill, nullptr, prefill_len, n_heads, d_k, d_v,
        /*chunk_size=*/64, /*use_qk_l2norm=*/true));
    const int qk_stride = n_heads * d_k;
    const int v_stride = n_heads * d_v;
    ASSERT_TRUE(prefill_kernel.recurrent_step(
        d_Q + static_cast<size_t>(prefill_len) * qk_stride,
        d_K + static_cast<size_t>(prefill_len) * qk_stride,
        d_V + static_cast<size_t>(prefill_len) * v_stride,
        d_alpha + static_cast<size_t>(prefill_len) * n_heads,
        d_beta + static_cast<size_t>(prefill_len) * n_heads,
        d_A_log,
        d_dt_bias,
        d_prefill + static_cast<size_t>(prefill_len) * v_stride,
        nullptr,
        n_heads, d_k, d_v,
        /*use_qk_l2norm=*/true));

    // Path 2: independent decode kernel receives every token one at a time.
    // Its outputs should match both the prefill history and the first decode
    // token if prefill state writeback has the same semantics as recurrent_step().
    ROCmGatedDeltaNet decode_kernel(0);
    for (int t = 0; t < total_len; ++t)
    {
        ASSERT_TRUE(decode_kernel.recurrent_step(
            d_Q + static_cast<size_t>(t) * qk_stride,
            d_K + static_cast<size_t>(t) * qk_stride,
            d_V + static_cast<size_t>(t) * v_stride,
            d_alpha + static_cast<size_t>(t) * n_heads,
            d_beta + static_cast<size_t>(t) * n_heads,
            d_A_log,
            d_dt_bias,
            d_decode + static_cast<size_t>(t) * v_stride,
            nullptr,
            n_heads, d_k, d_v,
            /*use_qk_l2norm=*/true));
    }

    prefill_out->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    decode_out->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    ASSERT_TRUE(prefill_out->ensureOnHost());
    ASSERT_TRUE(decode_out->ensureOnHost());

    // Compare both an absolute bound and a relative norm so a localized state
    // mismatch and a broad scale drift both fail loudly.
    const float *prefill = prefill_out->data();
    const float *decode = decode_out->data();
    float max_abs_diff = 0.0f;
    double sum_sq_diff = 0.0;
    double sum_sq_ref = 0.0;
    for (size_t i = 0; i < output_elems; ++i)
    {
        const float diff = std::abs(prefill[i] - decode[i]);
        max_abs_diff = std::max(max_abs_diff, diff);
        sum_sq_diff += static_cast<double>(diff) * diff;
        sum_sq_ref += static_cast<double>(decode[i]) * decode[i];
    }

    const double rel_l2 = std::sqrt(sum_sq_diff / std::max(sum_sq_ref, 1e-30));
    EXPECT_LT(max_abs_diff, 2e-4f) << "ROCm GDN prefill must match recurrent decode at Qwen35 shape";
    EXPECT_LT(rel_l2, 1e-4) << "ROCm GDN prefill must produce decode-compatible recurrent state";
}

TEST(Test__GDNKernels, ROCmChunkForwardFromNonZeroStateMatchesSequentialDecodeQwen35Shape)
{
    int device_count = 0;
    ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
    if (device_count <= 0)
        GTEST_SKIP() << "No ROCm device available";

    const DeviceId device = DeviceId::rocm(0);
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    const int n_heads = 16;
    const int d_k = 128;
    const int d_v = 128;
    const int prompt_len = 11;
    const int suffix_len = 2;
    const int next_len = 1;
    const int total_len = prompt_len + suffix_len + next_len;
    const int qk_stride = n_heads * d_k;
    const int v_stride = n_heads * d_v;
    const size_t qk_elems = static_cast<size_t>(total_len) * qk_stride;
    const size_t v_elems = static_cast<size_t>(total_len) * v_stride;
    const size_t gate_elems = static_cast<size_t>(total_len) * n_heads;

    auto Q = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(qk_stride)}, 0.0f, 0.03f, 1301);
    auto K = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(qk_stride)}, 0.0f, 0.03f, 1302);
    auto V = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(v_stride)}, 0.0f, 0.03f, 1303);
    auto alpha = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(n_heads)}, -0.15f, 0.2f, 1304);
    auto beta = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(n_heads)}, 0.0f, 0.2f, 1305);
    auto A_log = makeFP32Const({static_cast<size_t>(n_heads)}, -0.5f);
    auto dt_bias = makeFP32Const({static_cast<size_t>(n_heads)}, 0.1f);
    auto prompt_out = makeFP32({static_cast<size_t>(prompt_len), static_cast<size_t>(v_stride)});
    auto suffix_out = makeFP32({static_cast<size_t>(suffix_len), static_cast<size_t>(v_stride)});
    auto next_out = makeFP32({static_cast<size_t>(next_len), static_cast<size_t>(v_stride)});
    auto ref_out = makeFP32({static_cast<size_t>(total_len), static_cast<size_t>(v_stride)});

    ASSERT_EQ(Q->numel(), qk_elems);
    ASSERT_EQ(K->numel(), qk_elems);
    ASSERT_EQ(V->numel(), v_elems);
    ASSERT_EQ(alpha->numel(), gate_elems);
    ASSERT_EQ(beta->numel(), gate_elems);

    ASSERT_TRUE(Q->ensureOnDevice(device));
    ASSERT_TRUE(K->ensureOnDevice(device));
    ASSERT_TRUE(V->ensureOnDevice(device));
    ASSERT_TRUE(alpha->ensureOnDevice(device));
    ASSERT_TRUE(beta->ensureOnDevice(device));
    ASSERT_TRUE(A_log->ensureOnDevice(device));
    ASSERT_TRUE(dt_bias->ensureOnDevice(device));
    ASSERT_TRUE(prompt_out->allocateOnDevice(device));
    ASSERT_TRUE(suffix_out->allocateOnDevice(device));
    ASSERT_TRUE(next_out->allocateOnDevice(device));
    ASSERT_TRUE(ref_out->allocateOnDevice(device));

    auto *d_Q = static_cast<const float *>(Q->gpu_data_ptr());
    auto *d_K = static_cast<const float *>(K->gpu_data_ptr());
    auto *d_V = static_cast<const float *>(V->gpu_data_ptr());
    auto *d_alpha = static_cast<const float *>(alpha->gpu_data_ptr());
    auto *d_beta = static_cast<const float *>(beta->gpu_data_ptr());
    auto *d_A_log = static_cast<const float *>(A_log->gpu_data_ptr());
    auto *d_dt_bias = static_cast<const float *>(dt_bias->gpu_data_ptr());
    auto *d_prompt = static_cast<float *>(prompt_out->gpu_data_ptr());
    auto *d_suffix = static_cast<float *>(suffix_out->gpu_data_ptr());
    auto *d_next = static_cast<float *>(next_out->gpu_data_ptr());
    auto *d_ref = static_cast<float *>(ref_out->gpu_data_ptr());

    ASSERT_NE(d_Q, nullptr);
    ASSERT_NE(d_K, nullptr);
    ASSERT_NE(d_V, nullptr);
    ASSERT_NE(d_alpha, nullptr);
    ASSERT_NE(d_beta, nullptr);
    ASSERT_NE(d_A_log, nullptr);
    ASSERT_NE(d_dt_bias, nullptr);
    ASSERT_NE(d_prompt, nullptr);
    ASSERT_NE(d_suffix, nullptr);
    ASSERT_NE(d_next, nullptr);
    ASSERT_NE(d_ref, nullptr);

    ROCmGatedDeltaNet chunk_kernel(0);
    ASSERT_TRUE(chunk_kernel.chunk_forward(
        d_Q, d_K, d_V, d_alpha, d_beta, d_A_log, d_dt_bias,
        d_prompt, nullptr, prompt_len, n_heads, d_k, d_v,
        /*chunk_size=*/64, /*use_qk_l2norm=*/true));
    ASSERT_TRUE(chunk_kernel.chunk_forward(
        d_Q + static_cast<size_t>(prompt_len) * qk_stride,
        d_K + static_cast<size_t>(prompt_len) * qk_stride,
        d_V + static_cast<size_t>(prompt_len) * v_stride,
        d_alpha + static_cast<size_t>(prompt_len) * n_heads,
        d_beta + static_cast<size_t>(prompt_len) * n_heads,
        d_A_log, d_dt_bias,
        d_suffix, nullptr, suffix_len, n_heads, d_k, d_v,
        /*chunk_size=*/64, /*use_qk_l2norm=*/true));
    const int next_row = prompt_len + suffix_len;
    ASSERT_TRUE(chunk_kernel.recurrent_step(
        d_Q + static_cast<size_t>(next_row) * qk_stride,
        d_K + static_cast<size_t>(next_row) * qk_stride,
        d_V + static_cast<size_t>(next_row) * v_stride,
        d_alpha + static_cast<size_t>(next_row) * n_heads,
        d_beta + static_cast<size_t>(next_row) * n_heads,
        d_A_log, d_dt_bias,
        d_next, nullptr,
        n_heads, d_k, d_v,
        /*use_qk_l2norm=*/true));

    ROCmGatedDeltaNet ref_kernel(0);
    for (int t = 0; t < total_len; ++t)
    {
        ASSERT_TRUE(ref_kernel.recurrent_step(
            d_Q + static_cast<size_t>(t) * qk_stride,
            d_K + static_cast<size_t>(t) * qk_stride,
            d_V + static_cast<size_t>(t) * v_stride,
            d_alpha + static_cast<size_t>(t) * n_heads,
            d_beta + static_cast<size_t>(t) * n_heads,
            d_A_log, d_dt_bias,
            d_ref + static_cast<size_t>(t) * v_stride,
            nullptr,
            n_heads, d_k, d_v,
            /*use_qk_l2norm=*/true));
    }

    suffix_out->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    next_out->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    ref_out->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    ASSERT_TRUE(suffix_out->ensureOnHost());
    ASSERT_TRUE(next_out->ensureOnHost());
    ASSERT_TRUE(ref_out->ensureOnHost());

    const float *suffix = suffix_out->data();
    const float *next = next_out->data();
    const float *ref = ref_out->data();

    float max_abs_diff = 0.0f;
    double sum_sq_diff = 0.0;
    double sum_sq_ref = 0.0;
    for (int i = 0; i < suffix_len * v_stride; ++i)
    {
        const float expected = ref[static_cast<size_t>(prompt_len) * v_stride + i];
        const float diff = std::abs(suffix[i] - expected);
        max_abs_diff = std::max(max_abs_diff, diff);
        sum_sq_diff += static_cast<double>(diff) * diff;
        sum_sq_ref += static_cast<double>(expected) * expected;
    }
    for (int i = 0; i < v_stride; ++i)
    {
        const float expected = ref[static_cast<size_t>(next_row) * v_stride + i];
        const float diff = std::abs(next[i] - expected);
        max_abs_diff = std::max(max_abs_diff, diff);
        sum_sq_diff += static_cast<double>(diff) * diff;
        sum_sq_ref += static_cast<double>(expected) * expected;
    }

    const double rel_l2 = std::sqrt(sum_sq_diff / std::max(sum_sq_ref, 1e-30));
    EXPECT_LT(max_abs_diff, 2e-4f) << "ROCm GDN suffix chunk from nonzero state must match sequential decode";
    EXPECT_LT(rel_l2, 1e-4) << "ROCm GDN suffix chunk from nonzero state must preserve decode-compatible state";
}

TEST(Test__GDNKernels, ROCmPaddedGDNRealLengthStateMatchesUnpaddedDecodeAcrossReplayLengths)
{
    int device_count = 0;
    ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
    if (device_count <= 0)
        GTEST_SKIP() << "No ROCm device available";

    const DeviceId device = DeviceId::rocm(0);
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    const int n_heads = 16;
    const int d_k = 128;
    const int d_v = 128;
    const int bucket_len = 41;
    const int decode_row = bucket_len;
    const int total_len = bucket_len + 1;
    const int qk_stride = n_heads * d_k;
    const int v_stride = n_heads * d_v;
    const size_t qk_elems = static_cast<size_t>(total_len) * qk_stride;
    const size_t v_elems = static_cast<size_t>(total_len) * v_stride;
    const size_t gate_elems = static_cast<size_t>(total_len) * n_heads;
    const size_t output_elems = static_cast<size_t>(total_len) * v_stride;

    auto Q = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(qk_stride)}, 0.0f, 0.025f, 2101);
    auto K = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(qk_stride)}, 0.0f, 0.025f, 2102);
    auto V = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(v_stride)}, 0.0f, 0.025f, 2103);
    auto alpha = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(n_heads)}, -0.1f, 0.15f, 2104);
    auto beta = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(n_heads)}, 0.0f, 0.15f, 2105);
    auto A_log = makeFP32Const({static_cast<size_t>(n_heads)}, -0.5f);
    auto dt_bias = makeFP32Const({static_cast<size_t>(n_heads)}, 0.1f);
    auto padded_out = makeFP32({static_cast<size_t>(total_len), static_cast<size_t>(v_stride)});
    auto ref_out = makeFP32({static_cast<size_t>(total_len), static_cast<size_t>(v_stride)});

    ASSERT_EQ(Q->numel(), qk_elems);
    ASSERT_EQ(K->numel(), qk_elems);
    ASSERT_EQ(V->numel(), v_elems);
    ASSERT_EQ(alpha->numel(), gate_elems);
    ASSERT_EQ(beta->numel(), gate_elems);
    ASSERT_EQ(padded_out->numel(), output_elems);
    ASSERT_EQ(ref_out->numel(), output_elems);

    ASSERT_TRUE(Q->ensureOnDevice(device));
    ASSERT_TRUE(K->ensureOnDevice(device));
    ASSERT_TRUE(V->ensureOnDevice(device));
    ASSERT_TRUE(alpha->ensureOnDevice(device));
    ASSERT_TRUE(beta->ensureOnDevice(device));
    ASSERT_TRUE(A_log->ensureOnDevice(device));
    ASSERT_TRUE(dt_bias->ensureOnDevice(device));
    ASSERT_TRUE(padded_out->allocateOnDevice(device));
    ASSERT_TRUE(ref_out->allocateOnDevice(device));

    auto *d_Q = static_cast<const float *>(Q->gpu_data_ptr());
    auto *d_K = static_cast<const float *>(K->gpu_data_ptr());
    auto *d_V = static_cast<const float *>(V->gpu_data_ptr());
    auto *d_alpha = static_cast<const float *>(alpha->gpu_data_ptr());
    auto *d_beta = static_cast<const float *>(beta->gpu_data_ptr());
    auto *d_A_log = static_cast<const float *>(A_log->gpu_data_ptr());
    auto *d_dt_bias = static_cast<const float *>(dt_bias->gpu_data_ptr());
    auto *d_padded = static_cast<float *>(padded_out->gpu_data_ptr());
    auto *d_ref = static_cast<float *>(ref_out->gpu_data_ptr());

    int *d_effective_len = nullptr;
    ASSERT_EQ(hipMalloc(&d_effective_len, sizeof(int)), hipSuccess);

    for (int real_len : {37, 31})
    {
        ASSERT_EQ(hipMemcpy(d_effective_len, &real_len, sizeof(int), hipMemcpyHostToDevice), hipSuccess);

        ROCmGatedDeltaNet padded_kernel(0);
        ASSERT_TRUE(padded_kernel.chunkForwardWithEffectiveSeqLen(
            d_Q, d_K, d_V, d_alpha, d_beta, d_A_log, d_dt_bias,
            d_padded, nullptr, bucket_len, n_heads, d_k, d_v,
            /*chunk_size=*/64, /*use_qk_l2norm=*/true,
            d_effective_len));
        ASSERT_TRUE(padded_kernel.recurrent_step(
            d_Q + static_cast<size_t>(decode_row) * qk_stride,
            d_K + static_cast<size_t>(decode_row) * qk_stride,
            d_V + static_cast<size_t>(decode_row) * v_stride,
            d_alpha + static_cast<size_t>(decode_row) * n_heads,
            d_beta + static_cast<size_t>(decode_row) * n_heads,
            d_A_log,
            d_dt_bias,
            d_padded + static_cast<size_t>(decode_row) * v_stride,
            nullptr,
            n_heads, d_k, d_v,
            /*use_qk_l2norm=*/true));

        ROCmGatedDeltaNet ref_kernel(0);
        ASSERT_TRUE(ref_kernel.chunk_forward(
            d_Q, d_K, d_V, d_alpha, d_beta, d_A_log, d_dt_bias,
            d_ref, nullptr, real_len, n_heads, d_k, d_v,
            /*chunk_size=*/64, /*use_qk_l2norm=*/true));
        ASSERT_TRUE(ref_kernel.recurrent_step(
            d_Q + static_cast<size_t>(decode_row) * qk_stride,
            d_K + static_cast<size_t>(decode_row) * qk_stride,
            d_V + static_cast<size_t>(decode_row) * v_stride,
            d_alpha + static_cast<size_t>(decode_row) * n_heads,
            d_beta + static_cast<size_t>(decode_row) * n_heads,
            d_A_log,
            d_dt_bias,
            d_ref + static_cast<size_t>(decode_row) * v_stride,
            nullptr,
            n_heads, d_k, d_v,
            /*use_qk_l2norm=*/true));

        padded_out->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        ref_out->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        ASSERT_TRUE(padded_out->ensureOnHost());
        ASSERT_TRUE(ref_out->ensureOnHost());

        const float *padded = padded_out->data() + static_cast<size_t>(decode_row) * v_stride;
        const float *ref = ref_out->data() + static_cast<size_t>(decode_row) * v_stride;
        float max_abs_diff = 0.0f;
        double sum_sq_diff = 0.0;
        double sum_sq_ref = 0.0;
        for (int i = 0; i < v_stride; ++i)
        {
            const float diff = std::abs(padded[i] - ref[i]);
            max_abs_diff = std::max(max_abs_diff, diff);
            sum_sq_diff += static_cast<double>(diff) * diff;
            sum_sq_ref += static_cast<double>(ref[i]) * ref[i];
        }
        const double rel_l2 = std::sqrt(sum_sq_diff / std::max(sum_sq_ref, 1e-30));
        EXPECT_LT(max_abs_diff, 2e-4f) << "real_len=" << real_len;
        EXPECT_LT(rel_l2, 1e-4) << "real_len=" << real_len;
    }

    hipFree(d_effective_len);
}

TEST(Test__GDNKernels, ROCmPaddedShortConvRealLengthStateMatchesUnpaddedDecodeAcrossReplayLengths)
{
    int device_count = 0;
    ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
    if (device_count <= 0)
        GTEST_SKIP() << "No ROCm device available";

    const DeviceId device = DeviceId::rocm(0);
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    const int channels = 32;
    const int kernel_size = 4;
    const int bucket_len = 17;
    const int decode_row = bucket_len;
    const int total_len = bucket_len + 1;
    auto input = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(channels)}, 0.0f, 0.04f, 2201);
    auto weight = makeFP32Random({static_cast<size_t>(channels), static_cast<size_t>(kernel_size)}, 0.0f, 0.03f, 2202);
    auto bias = makeFP32Random({static_cast<size_t>(channels)}, 0.0f, 0.01f, 2203);
    auto padded_out = makeFP32({static_cast<size_t>(total_len), static_cast<size_t>(channels)});
    auto ref_out = makeFP32({static_cast<size_t>(total_len), static_cast<size_t>(channels)});

    ASSERT_TRUE(input->ensureOnDevice(device));
    ASSERT_TRUE(weight->ensureOnDevice(device));
    ASSERT_TRUE(bias->ensureOnDevice(device));
    ASSERT_TRUE(padded_out->allocateOnDevice(device));
    ASSERT_TRUE(ref_out->allocateOnDevice(device));

    auto *d_input = static_cast<const float *>(input->gpu_data_ptr());
    auto *d_weight = static_cast<const float *>(weight->gpu_data_ptr());
    auto *d_bias = static_cast<const float *>(bias->gpu_data_ptr());
    auto *d_padded = static_cast<float *>(padded_out->gpu_data_ptr());
    auto *d_ref = static_cast<float *>(ref_out->gpu_data_ptr());

    int *d_effective_len = nullptr;
    ASSERT_EQ(hipMalloc(&d_effective_len, sizeof(int)), hipSuccess);

    for (int real_len : {13, 9})
    {
        ASSERT_EQ(hipMemcpy(d_effective_len, &real_len, sizeof(int), hipMemcpyHostToDevice), hipSuccess);

        ROCmShortConvolution padded_kernel(0);
        ASSERT_TRUE(padded_kernel.forwardWithEffectiveSeqLen(
            d_input, d_weight, d_bias,
            d_padded, nullptr,
            bucket_len, channels, kernel_size,
            d_effective_len,
            /*apply_silu=*/true));
        ASSERT_TRUE(padded_kernel.forward(
            d_input + static_cast<size_t>(decode_row) * channels,
            d_weight,
            d_bias,
            d_padded + static_cast<size_t>(decode_row) * channels,
            nullptr,
            1, channels, kernel_size,
            /*apply_silu=*/true));

        ROCmShortConvolution ref_kernel(0);
        ASSERT_TRUE(ref_kernel.forward(
            d_input, d_weight, d_bias,
            d_ref, nullptr,
            real_len, channels, kernel_size,
            /*apply_silu=*/true));
        ASSERT_TRUE(ref_kernel.forward(
            d_input + static_cast<size_t>(decode_row) * channels,
            d_weight,
            d_bias,
            d_ref + static_cast<size_t>(decode_row) * channels,
            nullptr,
            1, channels, kernel_size,
            /*apply_silu=*/true));

        padded_out->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        ref_out->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        ASSERT_TRUE(padded_out->ensureOnHost());
        ASSERT_TRUE(ref_out->ensureOnHost());

        const float *padded = padded_out->data() + static_cast<size_t>(decode_row) * channels;
        const float *ref = ref_out->data() + static_cast<size_t>(decode_row) * channels;
        for (int channel = 0; channel < channels; ++channel)
        {
            EXPECT_NEAR(padded[channel], ref[channel], 1e-5f)
                << "real_len=" << real_len << " channel=" << channel;
        }
    }

    hipFree(d_effective_len);
}

TEST(Test__GDNKernels, ROCmShortConvChunkForwardFromNonZeroStateMatchesSequentialDecode)
{
    int device_count = 0;
    ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
    if (device_count <= 0)
        GTEST_SKIP() << "No ROCm device available";

    const DeviceId device = DeviceId::rocm(0);
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    const int channels = 64;
    const int kernel_size = 4;
    const int prompt_len = 11;
    const int suffix_len = 2;
    const int next_len = 1;
    const int total_len = prompt_len + suffix_len + next_len;

    auto input = makeFP32Random({static_cast<size_t>(total_len), static_cast<size_t>(channels)}, 0.0f, 0.08f, 2401);
    auto weight = makeFP32Random({static_cast<size_t>(channels), static_cast<size_t>(kernel_size)}, 0.0f, 0.05f, 2402);
    auto bias = makeFP32Random({static_cast<size_t>(channels)}, 0.0f, 0.01f, 2403);
    auto prompt_out = makeFP32({static_cast<size_t>(prompt_len), static_cast<size_t>(channels)});
    auto suffix_out = makeFP32({static_cast<size_t>(suffix_len), static_cast<size_t>(channels)});
    auto next_out = makeFP32({static_cast<size_t>(next_len), static_cast<size_t>(channels)});
    auto ref_out = makeFP32({static_cast<size_t>(total_len), static_cast<size_t>(channels)});

    ASSERT_TRUE(input->ensureOnDevice(device));
    ASSERT_TRUE(weight->ensureOnDevice(device));
    ASSERT_TRUE(bias->ensureOnDevice(device));
    ASSERT_TRUE(prompt_out->allocateOnDevice(device));
    ASSERT_TRUE(suffix_out->allocateOnDevice(device));
    ASSERT_TRUE(next_out->allocateOnDevice(device));
    ASSERT_TRUE(ref_out->allocateOnDevice(device));

    auto *d_input = static_cast<const float *>(input->gpu_data_ptr());
    auto *d_weight = static_cast<const float *>(weight->gpu_data_ptr());
    auto *d_bias = static_cast<const float *>(bias->gpu_data_ptr());
    auto *d_prompt = static_cast<float *>(prompt_out->gpu_data_ptr());
    auto *d_suffix = static_cast<float *>(suffix_out->gpu_data_ptr());
    auto *d_next = static_cast<float *>(next_out->gpu_data_ptr());
    auto *d_ref = static_cast<float *>(ref_out->gpu_data_ptr());
    ASSERT_NE(d_input, nullptr);
    ASSERT_NE(d_weight, nullptr);
    ASSERT_NE(d_bias, nullptr);
    ASSERT_NE(d_prompt, nullptr);
    ASSERT_NE(d_suffix, nullptr);
    ASSERT_NE(d_next, nullptr);
    ASSERT_NE(d_ref, nullptr);

    ROCmShortConvolution chunk_kernel(0);
    ASSERT_TRUE(chunk_kernel.forward(
        d_input, d_weight, d_bias,
        d_prompt, nullptr,
        prompt_len, channels, kernel_size,
        /*apply_silu=*/true));
    ASSERT_TRUE(chunk_kernel.forward(
        d_input + static_cast<size_t>(prompt_len) * channels,
        d_weight, d_bias,
        d_suffix, nullptr,
        suffix_len, channels, kernel_size,
        /*apply_silu=*/true));
    const int next_row = prompt_len + suffix_len;
    ASSERT_TRUE(chunk_kernel.forward(
        d_input + static_cast<size_t>(next_row) * channels,
        d_weight, d_bias,
        d_next, nullptr,
        next_len, channels, kernel_size,
        /*apply_silu=*/true));

    ROCmShortConvolution ref_kernel(0);
    for (int t = 0; t < total_len; ++t)
    {
        ASSERT_TRUE(ref_kernel.forward(
            d_input + static_cast<size_t>(t) * channels,
            d_weight, d_bias,
            d_ref + static_cast<size_t>(t) * channels,
            nullptr,
            1, channels, kernel_size,
            /*apply_silu=*/true));
    }

    suffix_out->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    next_out->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    ref_out->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
    ASSERT_TRUE(suffix_out->ensureOnHost());
    ASSERT_TRUE(next_out->ensureOnHost());
    ASSERT_TRUE(ref_out->ensureOnHost());

    const float *suffix = suffix_out->data();
    const float *next = next_out->data();
    const float *ref = ref_out->data();
    for (int i = 0; i < suffix_len * channels; ++i)
    {
        EXPECT_NEAR(suffix[i], ref[static_cast<size_t>(prompt_len) * channels + i], 1e-5f)
            << "suffix output mismatch at index " << i;
    }
    for (int c = 0; c < channels; ++c)
    {
        EXPECT_NEAR(next[c], ref[static_cast<size_t>(next_row) * channels + c], 1e-5f)
            << "next output mismatch at channel " << c;
    }
}

TEST(Test__GDNKernels, ROCmProjectionDecodeAllNativeCodebooksMatchesReference)
{
    int device_count = 0;
    ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
    if (device_count <= 0)
        GTEST_SKIP() << "No ROCm device available";

    const DeviceId device = DeviceId::rocm(0);
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ScopedEnvVar concurrent_decode("LLAMINAR_ROCM_CONCURRENT_DECODE");
    ScopedEnvVar gdn_concurrent_decode("LLAMINAR_ROCM_GDN_CONCURRENT_DECODE");
    ScopedEnvVar forced_kb("LLAMINAR_ROCM_NVNNI_GEMV_KB");
    concurrent_decode.set("0");
    gdn_concurrent_decode.set("0");
    forced_kb.set("8");

    const int M = 1;
    const int K = 256;
    const int n_heads = 16;
    const int d_k = 32;
    const int d_v = 32;
    const int n_qkv = 2 * n_heads * d_k + n_heads * d_v;
    const int n_z = n_heads * d_v;
    const int n_a = n_heads;
    const int n_b = n_heads;
    const std::array<int, 4> projection_sizes = {n_qkv, n_z, n_a, n_b};

    for (const auto &fmt : ALL_GDN_PROJECTION_CODEBOOKS)
    {
        std::array<std::unique_ptr<GDNProjectionBundle>, 4> bundles;
        WorkspaceRequirements combined_requirements;

        for (size_t projection = 0; projection < bundles.size(); ++projection)
        {
            auto bundle = std::make_unique<GDNProjectionBundle>();
            const int rows = projection_sizes[projection];
            bundle->weights = fmt.create(static_cast<size_t>(rows), static_cast<size_t>(K));
            ASSERT_NE(bundle->weights, nullptr)
                << fmt.name << ": failed to create projection " << projection << " weights";

            bundle->weights_fp32.resize(static_cast<size_t>(rows) * K);
            bundle->weights->to_fp32(bundle->weights_fp32.data());

            ASSERT_TRUE(rocm::packNativeVNNI(bundle->weights.get(), bundle->packed))
                << fmt.name << ": native-VNNI pack failed for projection " << projection;
            ASSERT_FALSE(bundle->packed.native_vnni_payload.empty())
                << fmt.name << ": native-VNNI payload missing for projection " << projection;

            bundle->kernel = std::make_unique<rocm::ROCmQuantisedGemmKernel>(&bundle->packed, 0);
            combined_requirements.merge(bundle->kernel->getWorkspaceRequirements(M, rows, K));
            bundles[projection] = std::move(bundle);
        }

        DeviceWorkspaceManager workspace(device, 128 * 1024 * 1024);
        ASSERT_TRUE(workspace.allocate(combined_requirements))
            << fmt.name << ": failed to allocate GDN projection workspace";
        for (auto &bundle : bundles)
            bundle->kernel->bindWorkspace(&workspace);

        auto input = test::TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)}, -0.75f, 0.75f, 2601);
        auto out_qkv = test::TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(n_qkv)});
        auto out_z = test::TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(n_z)});
        auto out_a = test::TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(n_a)});
        auto out_b = test::TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(n_b)});

        ASSERT_TRUE(input->ensureOnDevice(device));
        ASSERT_TRUE(out_qkv->allocateOnDevice(device));
        ASSERT_TRUE(out_z->allocateOnDevice(device));
        ASSERT_TRUE(out_a->allocateOnDevice(device));
        ASSERT_TRUE(out_b->allocateOnDevice(device));

        std::vector<ITensorGemm::TensorProjectionDesc> projections = {
            {bundles[0]->kernel.get(), out_qkv.get(), n_qkv, nullptr, "qkv"},
            {bundles[1]->kernel.get(), out_z.get(), n_z, nullptr, "z"},
            {bundles[2]->kernel.get(), out_a.get(), n_a, nullptr, "alpha"},
            {bundles[3]->kernel.get(), out_b.get(), n_b, nullptr, "beta"},
        };

        ASSERT_TRUE(bundles[0]->kernel->multiply_fused_tensor(
            input.get(), projections, M, K, nullptr, &workspace))
            << fmt.name << ": fused GDN projection failed";
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        out_qkv->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        out_z->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        out_a->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);
        out_b->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, device);

        const float *input_host = input->data();
        std::vector<float> expected;
        expected.reserve(static_cast<size_t>(n_qkv + n_z + n_a + n_b));
        for (size_t projection = 0; projection < bundles.size(); ++projection)
        {
            appendCpuGDNProjectionReference(
                input_host,
                bundles[projection]->weights_fp32,
                projection_sizes[projection],
                K,
                expected);
        }

        std::vector<float> actual;
        actual.reserve(expected.size());
        appendTensorData(out_qkv.get(), n_qkv, actual);
        appendTensorData(out_z.get(), n_z, actual);
        appendTensorData(out_a.get(), n_a, actual);
        appendTensorData(out_b.get(), n_b, actual);

        ASSERT_EQ(actual.size(), expected.size());
        const float cosine = vectorCosine(actual, expected);
        const float rel_l2 = vectorRelativeL2(actual, expected);
        const float max_abs = vectorMaxAbsDiff(actual, expected);
        const float max_ref_abs = vectorMaxAbsValue(expected);
        const float max_abs_ratio = max_abs / std::max(max_ref_abs, 1e-6f);

        LOG_INFO("[GDNProjectionNativeVNNI] " << fmt.name
                                              << " cosine=" << cosine
                                              << " rel_l2=" << rel_l2
                                              << " max_abs=" << max_abs
                                              << " max_abs_ratio=" << max_abs_ratio);

        EXPECT_GT(cosine, fmt.min_cosine)
            << fmt.name << ": GDN fused projection accuracy below threshold";
        EXPECT_LT(rel_l2, 0.05f)
            << fmt.name << ": GDN fused projection relative error too large";
        EXPECT_LT(max_abs_ratio, 0.02f)
            << fmt.name << ": GDN fused projection localized error too large";

        for (auto &bundle : bundles)
            bundle->kernel->unbindWorkspace();
    }
}
#endif

TEST(Test__GDNKernels, Recurrence_L2Norm)
{
    // Verify L2 normalization works correctly
    const int n_heads = 1;
    const int d_k = 4;
    const int d_v = 4;

    // Q with norm != 1: [3, 4, 0, 0] → norm = 5 → normalized = [0.6, 0.8, 0, 0]
    std::vector<float> q_data = {3.0f, 4.0f, 0.0f, 0.0f};
    std::vector<float> k_data = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> v_data = {1.0f, 1.0f, 1.0f, 1.0f};

    auto Q = makeFP32({1, 4}, q_data.data());
    auto K = makeFP32({1, 4}, k_data.data());
    auto V = makeFP32({1, 4}, v_data.data());

    // Large negative gate → decay ≈ 0 (fresh state each step)
    auto alpha = makeFP32Const({1, 1}, 100.0f);
    auto beta = makeFP32Const({1, 1}, 100.0f); // sigmoid ≈ 1
    auto A_log = makeFP32Const({1}, 0.0f);
    auto dt_bias = makeFP32Const({1}, 0.0f);

    // Run with L2 norm enabled
    auto output_l2 = makeFP32({1, 4});
    std::vector<float> state_l2(d_k * d_v, 0.0f);

    auto ctx = makeCPUContext();
    {
        GDNRecurrenceStage::Params p;
        p.kernel = &g_cpu_gdn;
        p.Q = Q.get();
        p.K = K.get();
        p.V = V.get();
        p.alpha = alpha.get();
        p.beta = beta.get();
        p.A_log = A_log.get();
        p.dt_bias = dt_bias.get();
        p.output = output_l2.get();
        p.recurrence_state = state_l2.data();
        p.seq_len = 1;
        p.n_heads = 1;
        p.d_k = d_k;
        p.d_v = d_v;
        p.use_qk_l2norm = true;

        GDNRecurrenceStage stage(p);
        ASSERT_TRUE(stage.execute(ctx.get()));
    }

    // Run without L2 norm
    auto output_no_l2 = makeFP32({1, 4});
    std::vector<float> state_no_l2(d_k * d_v, 0.0f);

    // Re-create tensors (data was modified by l2norm)
    auto Q2 = makeFP32({1, 4}, q_data.data());
    auto K2 = makeFP32({1, 4}, k_data.data());
    {
        GDNRecurrenceStage::Params p;
        p.kernel = &g_cpu_gdn;
        p.Q = Q2.get();
        p.K = K2.get();
        p.V = V.get();
        p.alpha = alpha.get();
        p.beta = beta.get();
        p.A_log = A_log.get();
        p.dt_bias = dt_bias.get();
        p.output = output_no_l2.get();
        p.recurrence_state = state_no_l2.data();
        p.seq_len = 1;
        p.n_heads = 1;
        p.d_k = d_k;
        p.d_v = d_v;
        p.use_qk_l2norm = false;

        GDNRecurrenceStage stage(p);
        ASSERT_TRUE(stage.execute(ctx.get()));
    }

    // Outputs should differ because L2 norm changes the effective Q/K magnitudes
    const float *l2_out = output_l2->data();
    const float *no_l2_out = output_no_l2->data();
    bool differ = false;
    for (int i = 0; i < d_v; ++i)
    {
        if (std::abs(l2_out[i] - no_l2_out[i]) > 1e-6f)
            differ = true;
    }
    EXPECT_TRUE(differ) << "L2 normalization should change the output";
}

TEST(Test__GDNKernels, Recurrence_EstimatedFlops)
{
    GDNRecurrenceStage::Params p;
    p.kernel = &g_cpu_gdn;
    p.seq_len = 10;
    p.n_heads = 4;
    p.d_k = 128;
    p.d_v = 128;
    GDNRecurrenceStage stage(p);

    // 10 * 4 * (4*128*128 + 128) = 10 * 4 * 65664 = 2626560
    EXPECT_EQ(stage.estimatedFlops(), 10ull * 4 * (4 * 128 * 128 + 128));
}

TEST(Test__GDNKernels, Recurrence_NoState_Fails)
{
    auto Q = makeFP32({1, 8});
    auto K = makeFP32({1, 8});
    auto V = makeFP32({1, 8});
    auto alpha = makeFP32({1, 2});
    auto beta = makeFP32({1, 2});
    auto A_log = makeFP32({2});
    auto dt_bias = makeFP32({2});
    auto output = makeFP32({1, 8});

    auto ctx = makeCPUContext();

    GDNRecurrenceStage::Params p;
    p.kernel = &g_cpu_gdn;
    p.Q = Q.get();
    p.K = K.get();
    p.V = V.get();
    p.alpha = alpha.get();
    p.beta = beta.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias.get();
    p.output = output.get();
    p.recurrence_state = nullptr;
    p.seq_len = 1;
    p.n_heads = 2;
    p.d_k = 4;
    p.d_v = 4;

    GDNRecurrenceStage stage(p);
    EXPECT_FALSE(stage.execute(ctx.get()));
}

// ============================================================================
// C6: GDN State Management Integration
// ============================================================================

TEST(Test__GDNKernels, StateIntegration_HybridCacheManager)
{
    // Verify that HybridCacheManager provides GDNLayerState with correct dimensions
    // and that we can pass state pointers to the stages
    HybridCacheManager::Config config;
    config.n_layers = 4;
    config.layer_types = {"gdn", "gdn", "gdn", "full_attention"};
    config.n_heads = 2;
    config.head_dim = 4;
    config.conv_kernel_size = 4;

    HybridCacheManager manager(config, nullptr);

    // GDN layers should have state
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(manager.getLayerStateType(i), LayerStateType::GDN_STATE);
        auto *state = manager.getGDNState(i);
        ASSERT_NE(state, nullptr);
        EXPECT_EQ(state->n_heads, 2);
        EXPECT_EQ(state->head_dim, 4);
        EXPECT_EQ(state->conv_kernel_size, 4);

        // Recurrence state: [n_heads, head_dim, head_dim] = [2, 4, 4] = 32
        EXPECT_EQ(state->recurrence_state.size(), 32u);

        // Conv state: [n_heads, conv_kernel-1, head_dim] = [2, 3, 4] = 24
        EXPECT_EQ(state->conv_state.size(), 24u);
    }

    // Full attention layer should not have GDN state
    EXPECT_EQ(manager.getLayerStateType(3), LayerStateType::KV_CACHE);
    EXPECT_EQ(manager.getGDNState(3), nullptr);
}

TEST(Test__GDNKernels, StateIntegration_PassToRecurrenceStage)
{
    // Create a HybridCacheManager and pass its state to GDNRecurrenceStage
    HybridCacheManager::Config config;
    config.n_layers = 1;
    config.layer_types = {"gdn"};
    config.n_heads = 2;
    config.head_dim = 4;
    config.conv_kernel_size = 4;

    HybridCacheManager manager(config, nullptr);
    auto *gdn_state = manager.getGDNState(0);
    ASSERT_NE(gdn_state, nullptr);

    // Create minimal inputs
    auto Q = makeFP32Random({1, 8}, 0.0f, 0.3f, 42);
    auto K = makeFP32Random({1, 8}, 0.0f, 0.3f, 43);
    auto V = makeFP32Random({1, 8}, 0.0f, 0.3f, 44);
    auto alpha = makeFP32Const({1, 2}, 0.0f);
    auto beta_raw = makeFP32Const({1, 2}, 0.0f);
    auto A_log = makeFP32Const({2}, 0.0f);
    auto dt_bias = makeFP32Const({2}, 0.0f);
    auto output = makeFP32({1, 8});

    auto ctx = makeCPUContext();

    GDNRecurrenceStage::Params p;
    p.kernel = &g_cpu_gdn;
    p.Q = Q.get();
    p.K = K.get();
    p.V = V.get();
    p.alpha = alpha.get();
    p.beta = beta_raw.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias.get();
    p.output = output.get();
    p.recurrence_state = gdn_state->recurrence_state.data();
    p.seq_len = 1;
    p.n_heads = 2;
    p.d_k = 4;
    p.d_v = 4;
    p.use_qk_l2norm = false;

    GDNRecurrenceStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // State should be modified
    float state_sum = 0.0f;
    for (float v : gdn_state->recurrence_state)
        state_sum += std::abs(v);
    EXPECT_GT(state_sum, 0.0f) << "GDNLayerState recurrence_state should be updated";
}

TEST(Test__GDNKernels, StateIntegration_PassToConv1dStage)
{
    // Create a HybridCacheManager and pass its conv state to ShortConv1dStage
    HybridCacheManager::Config config;
    config.n_layers = 1;
    config.layer_types = {"gdn"};
    config.n_heads = 2;
    config.head_dim = 4;
    config.conv_kernel_size = 4;

    HybridCacheManager manager(config, nullptr);
    auto *gdn_state = manager.getGDNState(0);
    ASSERT_NE(gdn_state, nullptr);

    // Conv state layout: [channels, kernel_size-1]
    // For GDNLayerState: [n_heads, conv_kernel-1, head_dim] but the ShortConv1d
    // operates on channels = QKV_dim. For this test, we just verify the state
    // pointer can be used.
    const int channels = 4; // simplified for test
    const int kernel_size = 4;

    auto input = makeFP32Random({1, static_cast<size_t>(channels)}, 0.0f, 0.5f, 42);
    auto output = makeFP32({1, static_cast<size_t>(channels)});
    auto weight = makeFP32Random({static_cast<size_t>(channels), static_cast<size_t>(kernel_size)}, 0.0f, 0.3f, 43);

    // Use a separate state buffer sized for the test
    std::vector<float> test_conv_state(channels * (kernel_size - 1), 0.0f);

    auto ctx = makeCPUContext();

    ShortConv1dStage::Params p;
    p.kernel = &g_cpu_conv;
    p.input = input.get();
    p.output = output.get();
    p.weight = weight.get();
    p.seq_len = 1;
    p.channels = channels;
    p.kernel_size = kernel_size;
    p.conv_state = test_conv_state.data();

    ShortConv1dStage stage(p);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // After decode, the new input should be stored in the state
    bool state_updated = false;
    for (float v : test_conv_state)
    {
        if (std::abs(v) > 1e-8f)
        {
            state_updated = true;
            break;
        }
    }
    EXPECT_TRUE(state_updated) << "Conv state should be updated after decode step";
}

TEST(Test__GDNKernels, StateIntegration_ResetClearsState)
{
    HybridCacheManager::Config config;
    config.n_layers = 2;
    config.layer_types = {"gdn", "gdn"};
    config.n_heads = 2;
    config.head_dim = 4;
    config.conv_kernel_size = 4;

    HybridCacheManager manager(config, nullptr);

    // Set some non-zero values
    auto *state0 = manager.getGDNState(0);
    auto *state1 = manager.getGDNState(1);
    ASSERT_NE(state0, nullptr);
    ASSERT_NE(state1, nullptr);

    for (auto &v : state0->recurrence_state)
        v = 1.0f;
    for (auto &v : state0->conv_state)
        v = 1.0f;
    for (auto &v : state1->recurrence_state)
        v = 2.0f;
    for (auto &v : state1->conv_state)
        v = 2.0f;

    // Reset should zero everything
    manager.reset();

    for (float v : state0->recurrence_state)
        EXPECT_EQ(v, 0.0f);
    for (float v : state0->conv_state)
        EXPECT_EQ(v, 0.0f);
    for (float v : state1->recurrence_state)
        EXPECT_EQ(v, 0.0f);
    for (float v : state1->conv_state)
        EXPECT_EQ(v, 0.0f);
}

// ============================================================================
// Integration: Conv1d output feeds into recurrence
// ============================================================================

TEST(Test__GDNKernels, Pipeline_Conv1dThenRecurrence)
{
    // End-to-end: conv1d → split QKV → recurrence
    // Verifies that output dimensions chain correctly
    const int seq_len = 4;
    const int n_heads = 2;
    const int d_k = 4;
    const int d_v = 4;
    const int qkv_dim = 2 * n_heads * d_k + n_heads * d_v; // 24
    const int kernel_size = 3;

    // Create conv1d input and weight
    auto conv_input = makeFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(qkv_dim)}, 0.0f, 0.5f, 1);
    auto conv_output = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(qkv_dim)});
    auto conv_weight = makeFP32Random({static_cast<size_t>(qkv_dim), static_cast<size_t>(kernel_size)}, 0.0f, 0.3f, 2);

    auto ctx = makeCPUContext();

    // Step 1: Conv1d
    {
        ShortConv1dStage::Params p;
        p.kernel = &g_cpu_conv;
        p.input = conv_input.get();
        p.output = conv_output.get();
        p.weight = conv_weight.get();
        p.seq_len = seq_len;
        p.channels = qkv_dim;
        p.kernel_size = kernel_size;

        ShortConv1dStage stage(p);
        ASSERT_TRUE(stage.execute(ctx.get()));
    }

    // Step 2: Split conv output into Q, K, V
    // Q: first n_heads*d_k dims, K: next n_heads*d_k, V: last n_heads*d_v
    const int q_dim = n_heads * d_k; // 8
    const int k_dim = n_heads * d_k; // 8
    const int v_dim = n_heads * d_v; // 8

    auto Q = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(q_dim)});
    auto K = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(k_dim)});
    auto V = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(v_dim)});

    const float *conv_out = conv_output->data();
    for (int t = 0; t < seq_len; ++t)
    {
        std::memcpy(Q->mutable_data() + t * q_dim, conv_out + t * qkv_dim, q_dim * sizeof(float));
        std::memcpy(K->mutable_data() + t * q_dim, conv_out + t * qkv_dim + q_dim, k_dim * sizeof(float));
        std::memcpy(V->mutable_data() + t * v_dim, conv_out + t * qkv_dim + q_dim + k_dim, v_dim * sizeof(float));
    }

    // Step 3: Recurrence
    auto alpha = makeFP32Const({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads)}, 0.0f);
    auto beta_raw = makeFP32Const({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads)}, 0.0f);
    auto A_log = makeFP32Const({static_cast<size_t>(n_heads)}, 1.0f);
    auto dt_bias = makeFP32Const({static_cast<size_t>(n_heads)}, 0.0f);
    auto rec_output = makeFP32({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * d_v)});
    std::vector<float> state(n_heads * d_k * d_v, 0.0f);

    {
        GDNRecurrenceStage::Params p;
        p.kernel = &g_cpu_gdn;
        p.Q = Q.get();
        p.K = K.get();
        p.V = V.get();
        p.alpha = alpha.get();
        p.beta = beta_raw.get();
        p.A_log = A_log.get();
        p.dt_bias = dt_bias.get();
        p.output = rec_output.get();
        p.recurrence_state = state.data();
        p.seq_len = seq_len;
        p.n_heads = n_heads;
        p.d_k = d_k;
        p.d_v = d_v;
        p.use_qk_l2norm = false;

        GDNRecurrenceStage stage(p);
        ASSERT_TRUE(stage.execute(ctx.get()));
    }

    // Verify output has reasonable values (not NaN/Inf)
    const float *out = rec_output->data();
    for (int i = 0; i < seq_len * n_heads * d_v; ++i)
    {
        EXPECT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(out[i])) << "Inf at index " << i;
    }
}

// ============================================================================
// Mock-based delegation tests
// ============================================================================

TEST(Test__GDNKernels, Conv1d_NullKernel_Fails)
{
    auto input = makeFP32({1, 4});
    auto output = makeFP32({1, 4});
    auto weight = makeFP32({4, 4});
    std::vector<float> state(4 * 3, 0.0f);
    auto ctx = makeCPUContext();

    ShortConv1dStage::Params p;
    p.input = input.get();
    p.output = output.get();
    p.weight = weight.get();
    p.seq_len = 1;
    p.channels = 4;
    p.kernel_size = 4;
    p.conv_state = state.data();
    p.kernel = nullptr; // Explicitly null

    ShortConv1dStage stage(p);
    EXPECT_FALSE(stage.execute(ctx.get()));
}

TEST(Test__GDNKernels, Conv1d_DelegatesToKernel)
{
    // Verify the stage delegates to ITensorShortConvolution::forward()
    auto input = makeFP32({1, 4}, std::vector<float>{1, 2, 3, 4}.data());
    auto output = makeFP32({1, 4});
    auto weight = makeFP32({4, 2}, std::vector<float>{1, 0, 1, 0, 1, 0, 1, 0}.data());
    std::vector<float> state(4 * 1, 0.0f);
    auto ctx = makeCPUContext();

    MockShortConvolution mock;
    EXPECT_CALL(mock, forward(_, _, _, _, _, 1, 4, 2, true))
        .WillOnce(Return(true));

    ShortConv1dStage::Params p;
    p.kernel = &mock;
    p.input = input.get();
    p.output = output.get();
    p.weight = weight.get();
    p.seq_len = 1;
    p.channels = 4;
    p.kernel_size = 2;
    p.conv_state = state.data();

    ShortConv1dStage stage(p);
    EXPECT_TRUE(stage.execute(ctx.get()));
}

TEST(Test__GDNKernels, Conv1d_KernelFailurePropagates)
{
    auto input = makeFP32({1, 4});
    auto output = makeFP32({1, 4});
    auto weight = makeFP32({4, 2});
    std::vector<float> state(4 * 1, 0.0f);
    auto ctx = makeCPUContext();

    MockShortConvolution mock;
    EXPECT_CALL(mock, forward(_, _, _, _, _, _, _, _, _))
        .WillOnce(Return(false));

    ShortConv1dStage::Params p;
    p.kernel = &mock;
    p.input = input.get();
    p.output = output.get();
    p.weight = weight.get();
    p.seq_len = 1;
    p.channels = 4;
    p.kernel_size = 2;
    p.conv_state = state.data();

    ShortConv1dStage stage(p);
    EXPECT_FALSE(stage.execute(ctx.get()));
}

TEST(Test__GDNKernels, Recurrence_NullKernel_Fails)
{
    auto Q = makeFP32({1, 8});
    auto K = makeFP32({1, 8});
    auto V = makeFP32({1, 8});
    auto alpha = makeFP32({1, 2});
    auto beta = makeFP32({1, 2});
    auto A_log = makeFP32({2});
    auto dt_bias = makeFP32({2});
    auto output = makeFP32({1, 8});
    std::vector<float> state(2 * 4 * 4, 0.0f);
    auto ctx = makeCPUContext();

    GDNRecurrenceStage::Params p;
    p.kernel = nullptr; // Explicitly null
    p.Q = Q.get();
    p.K = K.get();
    p.V = V.get();
    p.alpha = alpha.get();
    p.beta = beta.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias.get();
    p.output = output.get();
    p.recurrence_state = state.data();
    p.seq_len = 1;
    p.n_heads = 2;
    p.d_k = 4;
    p.d_v = 4;

    GDNRecurrenceStage stage(p);
    EXPECT_FALSE(stage.execute(ctx.get()));
}

TEST(Test__GDNKernels, Recurrence_Decode_DelegatesToRecurrentStep)
{
    // seq_len=1 should call recurrent_step, not chunk_forward
    auto Q = makeFP32Random({1, 8}, 0.0f, 0.3f, 42);
    auto K = makeFP32Random({1, 8}, 0.0f, 0.3f, 43);
    auto V = makeFP32Random({1, 8}, 0.0f, 0.3f, 44);
    auto alpha = makeFP32Const({1, 2}, 0.0f);
    auto beta = makeFP32Const({1, 2}, 0.0f);
    auto A_log = makeFP32Const({2}, 0.0f);
    auto dt_bias = makeFP32Const({2}, 0.0f);
    auto output = makeFP32({1, 8});
    std::vector<float> state(2 * 4 * 4, 0.0f);
    auto ctx = makeCPUContext();

    MockGatedDeltaNet mock;
    EXPECT_CALL(mock, recurrent_step(_, _, _, _, _, _, _, _, _, 2, 4, 4, _))
        .WillOnce(Return(true));
    // chunk_forward should NOT be called
    EXPECT_CALL(mock, chunk_forward(_, _, _, _, _, _, _, _, _, _, _, _, _, _, _))
        .Times(0);

    GDNRecurrenceStage::Params p;
    p.kernel = &mock;
    p.Q = Q.get();
    p.K = K.get();
    p.V = V.get();
    p.alpha = alpha.get();
    p.beta = beta.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias.get();
    p.output = output.get();
    p.recurrence_state = state.data();
    p.seq_len = 1;
    p.n_heads = 2;
    p.d_k = 4;
    p.d_v = 4;

    GDNRecurrenceStage stage(p);
    EXPECT_TRUE(stage.execute(ctx.get()));
}

TEST(Test__GDNKernels, Recurrence_Prefill_DelegatesToChunkForward)
{
    // seq_len>1 should call chunk_forward, not recurrent_step
    auto Q = makeFP32Random({4, 8}, 0.0f, 0.3f, 42);
    auto K = makeFP32Random({4, 8}, 0.0f, 0.3f, 43);
    auto V = makeFP32Random({4, 8}, 0.0f, 0.3f, 44);
    auto alpha = makeFP32Const({4, 2}, 0.0f);
    auto beta = makeFP32Const({4, 2}, 0.0f);
    auto A_log = makeFP32Const({2}, 0.0f);
    auto dt_bias = makeFP32Const({2}, 0.0f);
    auto output = makeFP32({4, 8});
    std::vector<float> state(2 * 4 * 4, 0.0f);
    auto ctx = makeCPUContext();

    MockGatedDeltaNet mock;
    EXPECT_CALL(mock, chunk_forward(_, _, _, _, _, _, _, _, _, 4, 2, 4, 4, _, _))
        .WillOnce(Return(true));
    // recurrent_step should NOT be called
    EXPECT_CALL(mock, recurrent_step(_, _, _, _, _, _, _, _, _, _, _, _, _))
        .Times(0);

    GDNRecurrenceStage::Params p;
    p.kernel = &mock;
    p.Q = Q.get();
    p.K = K.get();
    p.V = V.get();
    p.alpha = alpha.get();
    p.beta = beta.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias.get();
    p.output = output.get();
    p.recurrence_state = state.data();
    p.seq_len = 4;
    p.n_heads = 2;
    p.d_k = 4;
    p.d_v = 4;

    GDNRecurrenceStage stage(p);
    EXPECT_TRUE(stage.execute(ctx.get()));
}

TEST(Test__GDNKernels, Recurrence_KernelFailurePropagates)
{
    auto Q = makeFP32Random({1, 8}, 0.0f, 0.3f, 42);
    auto K = makeFP32Random({1, 8}, 0.0f, 0.3f, 43);
    auto V = makeFP32Random({1, 8}, 0.0f, 0.3f, 44);
    auto alpha = makeFP32Const({1, 2}, 0.0f);
    auto beta = makeFP32Const({1, 2}, 0.0f);
    auto A_log = makeFP32Const({2}, 0.0f);
    auto dt_bias = makeFP32Const({2}, 0.0f);
    auto output = makeFP32({1, 8});
    std::vector<float> state(2 * 4 * 4, 0.0f);
    auto ctx = makeCPUContext();

    MockGatedDeltaNet mock;
    EXPECT_CALL(mock, recurrent_step(_, _, _, _, _, _, _, _, _, _, _, _, _))
        .WillOnce(Return(false));

    GDNRecurrenceStage::Params p;
    p.kernel = &mock;
    p.Q = Q.get();
    p.K = K.get();
    p.V = V.get();
    p.alpha = alpha.get();
    p.beta = beta.get();
    p.A_log = A_log.get();
    p.dt_bias = dt_bias.get();
    p.output = output.get();
    p.recurrence_state = state.data();
    p.seq_len = 1;
    p.n_heads = 2;
    p.d_k = 4;
    p.d_v = 4;

    GDNRecurrenceStage stage(p);
    EXPECT_FALSE(stage.execute(ctx.get()));
}
