/**
 * @file Test__MoEExpertComputeStage.cpp
 * @brief Unit tests for MoE FFN stages: MoEExpertComputeStage, SharedExpertFFNStage, SharedExpertGateStage
 *
 * Tests the three MoE stages used by Qwen 3.5 MoE:
 * 1. MoEExpertComputeStage: Router (softmax top-k) → expert SwiGLU → weighted combine
 * 2. SharedExpertFFNStage: Dense SwiGLU on shared expert weights
 * 3. SharedExpertGateStage: sigmoid(gate · input) × shared_output
 *
 * Uses small synthetic tensors (no model loading required).
 */

#include <gtest/gtest.h>
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/compute_stages/stages/MoERoutingStage.h"
#include "execution/moe/CanonicalMoERouteRecord.h"
#include "execution/moe/RoutedExpertOwnerAssignment.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "execution/local_execution/graph/GraphSchema.h"
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"
#include "kernels/KernelFactory.h"
#include "kernels/IMoEKernel.h"
#include "kernels/cpu/moe/CPUCanonicalRouteReducer.h"
#include "kernels/cpu/primitives/VectorPrimitives.h"
#include "loaders/PreparedWeightStore.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "mocks/MockComputeStage.h"
#include "utils/PerfStatsCollector.h"
#include "utils/OpenMPUtils.h"
#include "utils/QuantizedVerifierFormats.h"
#include "utils/PreparedWeightTestHarness.h"
#include "utils/VerifierRowTestInventory.h"

#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace llaminar2;
using namespace llaminar2::test;
using namespace llaminar2::testing;

namespace
{
    /**
     * @brief Temporarily set an environment variable for perfstats assertions.
     *
     * PerfStatsCollector intentionally reads its enablement from process
     * environment so production code has no test-only switches.  These unit
     * regressions need counters to prove the grouped verifier route was
     * exercised, so the tests scope that environment mutation tightly.
     */
    class ScopedEnv
    {
    public:
        ScopedEnv(const char *name, const char *value)
            : name_(name)
        {
            const char *old_value = std::getenv(name_.c_str());
            if (old_value)
                old_value_ = std::string(old_value);
            if (value)
                setenv(name_.c_str(), value, 1);
            else
                unsetenv(name_.c_str());
        }

        ~ScopedEnv()
        {
            if (old_value_)
                setenv(name_.c_str(), old_value_->c_str(), 1);
            else
                unsetenv(name_.c_str());
        }

        ScopedEnv(const ScopedEnv &) = delete;
        ScopedEnv &operator=(const ScopedEnv &) = delete;

    private:
        std::string name_;
        std::optional<std::string> old_value_;
    };

    /**
     * @brief Temporarily bound OpenMP work to a small deterministic team.
     *
     * The all-format arithmetic regressions below launch many deliberately
     * tiny expert problems.  Using the machine-wide production team would
     * measure repeated team construction rather than the stage contract and
     * can turn a device-free unit test into a multi-minute workload on a
     * many-core host.  Thread-count scaling remains the responsibility of the
     * NativeVNNI performance suites; this guard keeps the correctness sweep
     * fast while executing the same kernels and arithmetic implementation.
     */
    class ScopedOpenMPThreadCount
    {
    public:
        /**
         * @brief Set the OpenMP team size until this scope is destroyed.
         * @param threads Positive number of workers used by nested kernels.
         */
        explicit ScopedOpenMPThreadCount(int threads)
        {
#ifdef _OPENMP
            previous_ = omp_get_max_threads();
            omp_set_num_threads(threads);
#else
            (void)threads;
#endif
        }

        /** @brief Restore the caller's OpenMP team size. */
        ~ScopedOpenMPThreadCount()
        {
#ifdef _OPENMP
            omp_set_num_threads(previous_);
#endif
        }

        ScopedOpenMPThreadCount(const ScopedOpenMPThreadCount &) = delete;
        ScopedOpenMPThreadCount &operator=(
            const ScopedOpenMPThreadCount &) = delete;

    private:
#ifdef _OPENMP
        int previous_ = 1;
#endif
    };

    bool hasPerfCounterWithRoute(
        const std::vector<PerfStatRecord> &records,
        const char *domain,
        const char *name,
        const char *route)
    {
        return std::any_of(
            records.begin(),
            records.end(),
            [&](const PerfStatRecord &record)
            {
                if (record.domain != domain || record.name != name)
                    return false;
                const auto it = record.tags.find("route");
                return it != record.tags.end() && it->second == route;
            });
    }

    class CapturingMoEKernel : public IMoEKernel
    {
    public:
        int observed_gate_type_id = -1;

        bool routeWithTensors(
            ITensor *hidden,
            ITensor *gate_weights,
            int seq_len, int d_model,
            int num_experts, int top_k,
            bool normalize_weights,
            ITensor *output_indices,
            ITensor *output_weights,
            MoERoutingResult &result) override
        {
            (void)hidden;
            (void)gate_weights;
            (void)seq_len;
            (void)d_model;
            (void)num_experts;
            (void)top_k;
            (void)normalize_weights;
            (void)output_indices;
            (void)output_weights;
            (void)result;
            return false;
        }

        void gatherTokenBatch(
            const float *hidden,
            float *batch_buffer,
            const int *token_indices,
            int num_tokens, int d_model) override
        {
            for (int i = 0; i < num_tokens; ++i)
                std::copy_n(hidden + static_cast<size_t>(token_indices[i]) * d_model,
                            d_model,
                            batch_buffer + static_cast<size_t>(i) * d_model);
        }

        void scatterAddWeighted(
            float *output,
            const float *expert_output,
            const int *token_indices,
            const float *weights,
            int num_tokens, int d_model) override
        {
            for (int i = 0; i < num_tokens; ++i)
            {
                float *dst = output + static_cast<size_t>(token_indices[i]) * d_model;
                const float *src = expert_output + static_cast<size_t>(i) * d_model;
                for (int j = 0; j < d_model; ++j)
                    dst[j] += weights[i] * src[j];
            }
        }

        void sharedExpertGate(
            const float *input,
            const float *gate_inp,
            float *shared_output,
            int seq_len, int d_model) override
        {
            for (int t = 0; t < seq_len; ++t)
            {
                const float *row = input + static_cast<size_t>(t) * d_model;
                float dot = 0.0f;
                for (int j = 0; j < d_model; ++j)
                    dot += gate_inp[j] * row[j];

                const float gate = 1.0f / (1.0f + std::exp(-dot));
                float *out = shared_output + static_cast<size_t>(t) * d_model;
                for (int j = 0; j < d_model; ++j)
                    out[j] *= gate;
            }
        }

        void sharedExpertGateFromTensors(
            ITensor *input, ITensor *gate_inp, ITensor *shared_output,
            int seq_len, int d_model) override
        {
            observed_gate_type_id = gate_inp ? gate_inp->native_type_id() : -1;
            IMoEKernel::sharedExpertGateFromTensors(
                input, gate_inp, shared_output, seq_len, d_model);
        }

        void swiGLU(float *gate, const float *up, int count) override
        {
            for (int i = 0; i < count; ++i)
            {
                const float x = gate[i];
                gate[i] = x / (1.0f + std::exp(-x)) * up[i];
            }
        }

        bool supports_device(int device_idx) const override
        {
            return device_idx < 0;
        }
    };

    class StreamCapturingGemm final : public ITensorGemm
    {
    public:
        bool supports_device(int) const override { return true; }

        void bindGPUStream(ExplicitGPUStream stream) override
        {
            observed_stream = stream.get();
            ++set_stream_calls;
        }

        void clearGPUStreamBinding() override
        {
            observed_stream = nullptr;
        }

        bool multiply_tensor(const TensorBase *, TensorBase *,
                             int, int, int,
                             bool, float, float,
                             const TensorBase *,
                             const IMPIContext *,
                             int,
                             DeviceWorkspaceManager *,
                             int) override
        {
            return false;
        }

        void *observed_stream = nullptr;
        int set_stream_calls = 0;
    };

    class WorkspaceOnlyGemm final : public ITensorGemm, public IWorkspaceConsumer
    {
    public:
        WorkspaceOnlyGemm(int default_n, int default_k)
            : default_n_(default_n),
              default_k_(default_k)
        {
        }

        bool supports_device(int device_idx) const override
        {
            return device_idx >= 0;
        }

        bool multiply_tensor(
            const TensorBase *,
            TensorBase *,
            int, int, int,
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
            const int rows = std::max(1, std::min(m, 4));
            const int out_cols = n > 0 ? n : default_n_;
            const int in_cols = k > 0 ? k : default_k_;
            const int k_groups = (in_cols + 31) / 32;

            WorkspaceRequirements reqs;
            reqs.buffers.push_back({
                GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS,
                static_cast<size_t>(k_groups) * static_cast<size_t>(rows) *
                    static_cast<size_t>(out_cols) * sizeof(float),
                256,
                true,
                WorkspaceExecutionRegime::CompactDecodeOnly});
            return reqs;
        }

        void bindWorkspace(DeviceWorkspaceManager *workspace) override
        {
            workspace_ = workspace;
        }

        bool hasWorkspace() const override
        {
            return workspace_ != nullptr;
        }

        DeviceWorkspaceManager *getWorkspace() const override
        {
            return workspace_;
        }

    private:
        int default_n_ = 0;
        int default_k_ = 0;
        DeviceWorkspaceManager *workspace_ = nullptr;
    };

    PreparedWeightRef registerWorkspaceOnlyGemm(
        PreparedWeightStore &store,
        TensorBase *tensor,
        DeviceId device,
        const std::string &canonical_name,
        std::shared_ptr<ITensorGemm> kernel)
    {
        namespace kf = llaminar::v2::kernels;

        auto binding = makePreparedWeightTestBinding(
            tensor,
            device,
            canonical_name,
            store.modelId());

        auto prepared_weights = std::make_shared<kf::KernelFactory::PreparedGemmWeights>();
        prepared_weights->owned_kernel = kernel;
        prepared_weights->kernel = kernel.get();

        auto handle = std::make_shared<kf::KernelFactory::PreparedGemmHandle>();
        handle->tensor = tensor;
        handle->device_id = device;
        handle->kind = kf::KernelFactory::GemmPreparationKind::CUDA_INT8_PACKED;
        handle->prepared_weights = std::move(prepared_weights);

        return store.registerPreparedGemmHandle(
            binding,
            PreparedWeightKind::CudaInt8PackedGemm,
            device,
            std::move(handle));
    }
}

// =========================================================================
// Test Fixture
// =========================================================================

class MoEExpertComputeStageTest : public ::testing::Test
{
protected:
    std::unique_ptr<MockDeviceContext> cpu_ctx_;

    // Small dimensions for unit testing
    static constexpr int D_MODEL = 64;
    static constexpr int INTERMEDIATE = 32;
    static constexpr int NUM_EXPERTS = 4;
    static constexpr int TOP_K = 2;
    static constexpr int SEQ_LEN = 2;

    void SetUp() override
    {
        cpu_ctx_ = std::make_unique<MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
    }

    /// Create a 3D Q4_K expert tensor in GGUF layout [cols, rows, num_experts]
    /// where cols (ne[0]) must be a multiple of 256 (Q4_K block size)
    /// Memory: ne[0]=cols is fastest-varying, ne[2]=num_experts is slowest
    std::shared_ptr<Q4_KTensor> createExpertQ4K(int num_experts, int rows, int cols, uint32_t seed = 42)
    {
        // GGUF 3D convention: shape = [ne[0]=cols, ne[1]=rows, ne[2]=num_experts]
        std::vector<size_t> shape = {static_cast<size_t>(cols),
                                     static_cast<size_t>(rows),
                                     static_cast<size_t>(num_experts)};
        size_t blocks_per_row = cols / Q4_KBlock::BLOCK_SIZE;
        size_t total_blocks = num_experts * rows * blocks_per_row;
        std::vector<uint8_t> raw(total_blocks * sizeof(Q4_KBlock));

        // Fill with deterministic random data
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        auto *blocks = reinterpret_cast<Q4_KBlock *>(raw.data());
        for (size_t i = 0; i < total_blocks; ++i)
        {
            // Set d and dmin scales
            float d_val = dist(rng) * 0.01f;
            float dmin_val = std::abs(dist(rng)) * 0.001f;
            blocks[i].d = fp32_to_fp16(d_val);
            blocks[i].dmin = fp32_to_fp16(dmin_val);

            // Fill qs with random bytes
            for (size_t j = 0; j < sizeof(blocks[i].qs); ++j)
                blocks[i].qs[j] = static_cast<uint8_t>(rng());

            // Fill scales
            for (size_t j = 0; j < sizeof(blocks[i].scales); ++j)
                blocks[i].scales[j] = static_cast<uint8_t>(rng());
        }

        return std::make_shared<Q4_KTensor>(shape, raw);
    }

    /// Create a 3D Q5_K expert tensor in GGUF layout [cols, rows, num_experts]
    /// where cols (ne[0]) must be a multiple of 256 (Q5_K block size)
    /// Memory: ne[0]=cols is fastest-varying, ne[2]=num_experts is slowest
    std::shared_ptr<Q5_KTensor> createExpertQ5K(int num_experts, int rows, int cols, uint32_t seed = 42)
    {
        // GGUF 3D convention: shape = [ne[0]=cols, ne[1]=rows, ne[2]=num_experts]
        std::vector<size_t> shape = {static_cast<size_t>(cols),
                                     static_cast<size_t>(rows),
                                     static_cast<size_t>(num_experts)};
        size_t blocks_per_row = cols / Q5_KBlock::BLOCK_SIZE;
        size_t total_blocks = num_experts * rows * blocks_per_row;
        std::vector<uint8_t> raw(total_blocks * sizeof(Q5_KBlock));

        std::mt19937 rng(seed);
        auto *blocks = reinterpret_cast<Q5_KBlock *>(raw.data());
        for (size_t i = 0; i < total_blocks; ++i)
        {
            blocks[i].d = fp32_to_fp16(0.01f);
            blocks[i].dmin = fp32_to_fp16(0.001f);
            for (size_t j = 0; j < sizeof(blocks[i].qs); ++j)
                blocks[i].qs[j] = static_cast<uint8_t>(rng());
            for (size_t j = 0; j < sizeof(blocks[i].qh); ++j)
                blocks[i].qh[j] = static_cast<uint8_t>(rng());
            for (size_t j = 0; j < sizeof(blocks[i].scales); ++j)
                blocks[i].scales[j] = static_cast<uint8_t>(rng());
        }

        return std::make_shared<Q5_KTensor>(shape, raw);
    }

    /// Create a 3D IQ3_S expert tensor in GGUF layout [cols, rows, num_experts].
    std::shared_ptr<IQ3_STensor> createExpertIQ3S(int num_experts, int rows, int cols, uint32_t seed = 42)
    {
        std::vector<size_t> shape = {static_cast<size_t>(cols),
                                     static_cast<size_t>(rows),
                                     static_cast<size_t>(num_experts)};
        size_t blocks_per_row = (static_cast<size_t>(cols) + IQ3_SBlock::BLOCK_SIZE - 1) /
                                IQ3_SBlock::BLOCK_SIZE;
        size_t total_blocks = static_cast<size_t>(num_experts) *
                              static_cast<size_t>(rows) *
                              blocks_per_row;
        std::vector<uint8_t> raw(total_blocks * sizeof(IQ3_SBlock));

        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 0.1f);

        auto *blocks = reinterpret_cast<IQ3_SBlock *>(raw.data());
        for (size_t i = 0; i < total_blocks; ++i)
        {
            float max_abs = 0.0f;
            for (size_t j = 0; j < IQ3_SBlock::BLOCK_SIZE; ++j)
                max_abs = std::max(max_abs, std::abs(dist(rng)));

            blocks[i].d = fp32_to_fp16(std::max(max_abs / 3.5f, 1e-6f));
            std::memset(blocks[i].qh, 0, sizeof(blocks[i].qh));
            std::memset(blocks[i].signs, 0, sizeof(blocks[i].signs));
            std::memset(blocks[i].scales, 0x44, sizeof(blocks[i].scales));
            for (uint8_t &q : blocks[i].qs)
                q = static_cast<uint8_t>(rng() & 0xFF);
        }

        return std::make_shared<IQ3_STensor>(shape, raw);
    }

    /// Compute reference SwiGLU: silu(gate) * up, where silu(x) = x * sigmoid(x)
    static float silu(float x)
    {
        return x / (1.0f + std::exp(-x));
    }

    /// Reference: sigmoid
    static float sigmoid(float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }

    /// Reference: dot product
    static float dot(const float *a, const float *b, int n)
    {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i)
            sum += a[i] * b[i];
        return sum;
    }

    static float maxAbsDiff(const float *a, const float *b, size_t n)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < n; ++i)
            max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
        return max_diff;
    }

    static float relativeL2(const float *a, const float *b, size_t n)
    {
        double diff = 0.0;
        double ref = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            diff += d * d;
            ref += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        return ref > 0.0 ? static_cast<float>(std::sqrt(diff / ref)) : 0.0f;
    }

    static void expectRowsByteEqual(
        const float *actual,
        const float *reference,
        int rows,
        int cols,
        const std::string &label)
    {
        const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        if (std::memcmp(actual, reference, total * sizeof(float)) == 0)
            return;

        size_t first = 0;
        while (first < total && actual[first] == reference[first])
            ++first;

        uint32_t actual_bits = 0;
        uint32_t reference_bits = 0;
        if (first < total)
        {
            std::memcpy(&actual_bits, actual + first, sizeof(actual_bits));
            std::memcpy(&reference_bits, reference + first, sizeof(reference_bits));
        }

        const size_t safe_cols = cols > 0 ? static_cast<size_t>(cols) : total;
        ADD_FAILURE()
            << label
            << " first_mismatch=" << first
            << " row=" << (first / safe_cols)
            << " col=" << (first % safe_cols)
            << " actual=" << (first < total ? actual[first] : 0.0f)
            << " reference=" << (first < total ? reference[first] : 0.0f)
            << " actual_bits=0x" << std::hex << actual_bits
            << " reference_bits=0x" << reference_bits << std::dec
            << " relative_l2=" << relativeL2(actual, reference, total)
            << " max_abs=" << maxAbsDiff(actual, reference, total);
    }

    using CPUVerifierWeightCreator = std::function<std::unique_ptr<TensorBase>(
        const std::vector<size_t> &shape,
        int seed)>;

    struct CPUVerifierFormatCase
    {
        const char *label;
        CPUVerifierWeightCreator create;
    };

    std::vector<CPUVerifierFormatCase> cpuVerifierNativeFormats()
    {
        std::vector<CPUVerifierFormatCase> formats;
        formats.reserve(quantizedVerifierFormats().size());
        for (const auto &format : quantizedVerifierFormats())
        {
            formats.push_back({
                format.label,
                [creator = format.create](const std::vector<size_t> &shape, int seed)
                {
                    return creator(shape, static_cast<uint32_t>(seed));
                }});
        }
        return formats;
    }

    /// Compute routing results and return as FP32 tensors for MoEExpertComputeStage input.
    /// Runs IMoEKernel::route() directly, converts indices to float.
    struct RoutingResult
    {
        std::shared_ptr<FP32Tensor> indices; // float-cast expert IDs [seq_len * top_k]
        std::shared_ptr<FP32Tensor> weights; // normalized weights [seq_len * top_k]
        std::shared_ptr<MoERoutedPipelineKernelOwner> kernel_owner;
    };

    RoutingResult computeRouting(TensorBase *input, TensorBase *gate_weights,
                                 int seq_len, int d_model, int num_experts, int top_k,
        bool norm_topk_prob = true)
    {
        using KernelFactory = llaminar::v2::kernels::KernelFactory;
        auto kernel_owner = std::make_shared<MoERoutedPipelineKernelOwner>();
        kernel_owner->kernel = KernelFactory::createMoEKernel(DeviceId::cpu());
        IMoEKernel *kernel = kernel_owner->kernel.get();
        const size_t n = static_cast<size_t>(seq_len) * top_k;
        auto indices = std::make_shared<FP32Tensor>(std::vector<size_t>{n, 1});
        auto weights = std::make_shared<FP32Tensor>(std::vector<size_t>{n, 1});
        MoERoutingResult routing;
        if (!kernel->routeWithTensors(
                input,
                gate_weights,
                seq_len,
                d_model,
                num_experts,
                top_k,
                norm_topk_prob,
                indices.get(),
                weights.get(),
                routing))
        {
            throw std::runtime_error("CPU routing fixture failed to publish routing tensors");
        }
        return {indices, weights, std::move(kernel_owner)};
    }
};

/**
 * @brief Prove MoE launch state is never shared implicitly by the factory.
 *
 * This is intentionally a CPU-only unit test. The ownership rule is backend
 * independent, while CUDA and ROCm execution remains in integration tests.
 */
TEST(Test__MoEKernelOwnership, FactoryCreatesIndependentLaunchState)
{
    using KernelFactory = llaminar::v2::kernels::KernelFactory;

    auto routing_kernel = KernelFactory::createMoEKernel(DeviceId::cpu());
    auto expert_kernel = KernelFactory::createMoEKernel(DeviceId::cpu());

    ASSERT_NE(routing_kernel, nullptr);
    ASSERT_NE(expert_kernel, nullptr);
    EXPECT_NE(routing_kernel.get(), expert_kernel.get())
        << "Separately captured stages must not alias mutable stream, workspace, "
           "descriptor-table, or scratch state.";
}

/**
 * @brief Prove payload transaction purpose owns histogram lifecycle policy.
 *
 * Prefix rehydration and current-batch movement have distinct graph
 * authorities, but both consume the shared immutable rebalance config ABI.
 * They must not share reset semantics: rehydration preserves routing evidence
 * imported with the prefix, while current-batch placement consumes it and
 * starts the next window.
 */
TEST(Test__MoEExpertComputeStage,
     PrefillLLEPTransferPurposeMakesHistogramOwnershipExplicit)
{
    DeviceMoERebalanceConfig base;
    base.flags =
        static_cast<uint32_t>(DeviceMoERebalanceFlags::HotReplicaCache) |
        static_cast<uint32_t>(DeviceMoERebalanceFlags::DeferRuntimeApply) |
        static_cast<uint32_t>(
            DeviceMoERebalanceFlags::ResetHistogramsAfterApply);

    const auto rehydration = prefillLLEPTransferConfig(
        base,
        PrefillLLEPTransferPurpose::PrefixRuntimeRehydration);
    EXPECT_FALSE(hasDeviceMoERebalanceFlag(
        rehydration.flags,
        DeviceMoERebalanceFlags::ResetHistogramsAfterApply));
    EXPECT_TRUE(hasDeviceMoERebalanceFlag(
        rehydration.flags,
        DeviceMoERebalanceFlags::HotReplicaCache));
    EXPECT_TRUE(hasDeviceMoERebalanceFlag(
        rehydration.flags,
        DeviceMoERebalanceFlags::DeferRuntimeApply));

    DeviceMoERebalanceConfig no_reset = base;
    no_reset.flags &=
        ~static_cast<uint32_t>(
            DeviceMoERebalanceFlags::ResetHistogramsAfterApply);
    const auto current_batch = prefillLLEPTransferConfig(
        no_reset,
        PrefillLLEPTransferPurpose::CurrentBatchMovement);
    EXPECT_TRUE(hasDeviceMoERebalanceFlag(
        current_batch.flags,
        DeviceMoERebalanceFlags::ResetHistogramsAfterApply));

    EXPECT_THROW(
        (void)prefillLLEPTransferConfig(
            base,
            static_cast<PrefillLLEPTransferPurpose>(255u)),
        std::invalid_argument);
}

// =========================================================================
// SharedExpertFFNStage Tests
// =========================================================================

TEST_F(MoEExpertComputeStageTest, SharedExpert_OutputNonZero)
{
    // Create small FP32 weight tensors for shared expert
    auto input = TestTensorFactory::createFP32Random({SEQ_LEN, D_MODEL}, -0.5f, 0.5f, 100);
    auto gate_w = TestTensorFactory::createFP32Random({INTERMEDIATE, D_MODEL}, -0.1f, 0.1f, 101);
    auto up_w = TestTensorFactory::createFP32Random({INTERMEDIATE, D_MODEL}, -0.1f, 0.1f, 102);
    auto down_w = TestTensorFactory::createFP32Random({D_MODEL, INTERMEDIATE}, -0.1f, 0.1f, 103);
    auto output = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});
    auto prepared = makePreparedFFNFixture(
        gate_w.get(), up_w.get(), down_w.get(), DeviceId::cpu(), 0, "ffn_shexp");

    // Zero the output
    std::memset(output->mutable_data(), 0, output->numel() * sizeof(float));

    SharedExpertFFNStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.gate_w = gate_w.get();
    params.up_w = up_w.get();
    params.down_w = down_w.get();
    params.output = output.get();
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.intermediate = INTERMEDIATE;
    params.prepared_ref_gate = prepared.gate_ref;
    params.prepared_ref_up = prepared.up_ref;
    params.prepared_ref_down = prepared.down_ref;
    params.prepared_store = prepared.store.get();

    SharedExpertFFNStage stage(params);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    // Output should be non-zero
    const float *out = output->data();
    bool any_nonzero = false;
    for (int i = 0; i < SEQ_LEN * D_MODEL; ++i)
    {
        if (out[i] != 0.0f)
        {
            any_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero) << "SharedExpertFFN output is all zeros";

    // No NaN or Inf
    for (int i = 0; i < SEQ_LEN * D_MODEL; ++i)
    {
        EXPECT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(out[i])) << "Inf at index " << i;
    }
}

TEST_F(MoEExpertComputeStageTest, SharedExpert_MatchesReference)
{
    const int seq = 1;
    const int d = 8;
    const int inter = 4;

    // Create small hand-verifiable tensors
    auto input = TestTensorFactory::createFP32({seq, d});
    auto gate_w = TestTensorFactory::createFP32({inter, d});
    auto up_w = TestTensorFactory::createFP32({inter, d});
    auto down_w = TestTensorFactory::createFP32({d, inter});
    auto output = TestTensorFactory::createFP32({seq, d});
    auto prepared = makePreparedFFNFixture(
        gate_w.get(), up_w.get(), down_w.get(), DeviceId::cpu(), 0, "ffn_shexp");

    // Fill with simple values
    float *inp = input->mutable_data();
    for (int i = 0; i < d; ++i)
        inp[i] = 0.1f * (i + 1);

    float *gw = gate_w->mutable_data();
    float *uw = up_w->mutable_data();
    float *dw = down_w->mutable_data();

    // Identity-ish weights
    for (int i = 0; i < inter * d; ++i)
    {
        gw[i] = (i % d == i / inter) ? 1.0f : 0.0f;
        uw[i] = (i % d == i / inter) ? 1.0f : 0.0f;
    }
    for (int i = 0; i < d * inter; ++i)
        dw[i] = (i % inter == i / d) ? 1.0f : 0.0f;

    std::memset(output->mutable_data(), 0, output->numel() * sizeof(float));

    SharedExpertFFNStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.gate_w = gate_w.get();
    params.up_w = up_w.get();
    params.down_w = down_w.get();
    params.output = output.get();
    params.seq_len = seq;
    params.d_model = d;
    params.intermediate = inter;
    params.prepared_ref_gate = prepared.gate_ref;
    params.prepared_ref_up = prepared.up_ref;
    params.prepared_ref_down = prepared.down_ref;
    params.prepared_store = prepared.store.get();

    SharedExpertFFNStage stage(params);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    // Compute reference
    // gate_proj: for each intermediate neuron i, gate_out[i] = sum(gate_w[i,:] * input[:])
    // With our identity-ish weights: gate_out[i] = input[i] for i < inter
    // up_proj[i] = input[i] for i < inter
    // SwiGLU: silu(gate_out[i]) * up_out[i]
    // Down: output[d] = sum(down_w[d,:] * activated[:])
    std::vector<float> gate_out(inter), up_out(inter), activated(inter);
    for (int i = 0; i < inter; ++i)
    {
        gate_out[i] = dot(gw + i * d, inp, d);
        up_out[i] = dot(uw + i * d, inp, d);
        activated[i] = silu(gate_out[i]) * up_out[i];
    }
    std::vector<float> ref_out(d);
    for (int dd = 0; dd < d; ++dd)
        ref_out[dd] = dot(dw + dd * inter, activated.data(), inter);

    const float *out = output->data();
    for (int dd = 0; dd < d; ++dd)
    {
        EXPECT_NEAR(out[dd], ref_out[dd], 1e-5f)
            << "Mismatch at dim " << dd;
    }
}

TEST_F(MoEExpertComputeStageTest, SharedExpert_NullInputReturnsError)
{
    auto output = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});

    SharedExpertFFNStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = nullptr; // null
    params.output = output.get();
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.intermediate = INTERMEDIATE;

    SharedExpertFFNStage stage(params);
    EXPECT_FALSE(stage.execute(cpu_ctx_.get()));
}

// =========================================================================
// SharedExpertGateStage Tests
// =========================================================================

TEST_F(MoEExpertComputeStageTest, SharedGate_AppliesSigmoidGating)
{
    const int seq = 2;
    const int d = 8;

    auto input = TestTensorFactory::createFP32({seq, d});
    auto gate_inp = TestTensorFactory::createFP32({1, d});
    auto shared_output = TestTensorFactory::createFP32({seq, d});

    // Fill input with 1s
    for (int i = 0; i < seq * d; ++i)
        input->mutable_data()[i] = 1.0f;

    // Fill gate_inp to produce known dot products
    for (int i = 0; i < d; ++i)
        gate_inp->mutable_data()[i] = 0.0f; // sigmoid(0) = 0.5

    // Fill shared output with 2.0
    for (int i = 0; i < seq * d; ++i)
        shared_output->mutable_data()[i] = 2.0f;

    SharedExpertGateStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.gate_inp = gate_inp.get();
    params.shared_output = shared_output.get();
    params.seq_len = seq;
    params.d_model = d;

    SharedExpertGateStage stage(params);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    // With gate_inp = 0 and input = 1, dot = 0, sigmoid(0) = 0.5
    // shared_output should be 2.0 * 0.5 = 1.0
    const float *out = shared_output->data();
    for (int i = 0; i < seq * d; ++i)
    {
        EXPECT_NEAR(out[i], 1.0f, 1e-5f) << "Gate mismatch at index " << i;
    }
}

TEST_F(MoEExpertComputeStageTest, SharedGate_RejectsStageLocalGateNormalizationOnEveryBackend)
{
    BF16Tensor gate_inp({1, 4});
    ASSERT_EQ(gate_inp.native_type(), TensorType::BF16);

    for (const DeviceId device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        SharedExpertGateStage::Params params;
        params.device_id = device;
        params.gate_inp = &gate_inp;

        EXPECT_THROW((void)SharedExpertGateStage(params), std::invalid_argument)
            << device.to_string()
            << " must receive the immutable FP32 model-prepared input gate";
    }
}

TEST_F(MoEExpertComputeStageTest,
       SharedGate_GPURejectsForwardTimeGateNormalization)
{
    BF16Tensor gate_inp({1, 4});
    SharedExpertGateStage::Params params;
    params.device_id = DeviceId::cuda(0);
    params.gate_inp = &gate_inp;

    EXPECT_THROW((void)SharedExpertGateStage(params), std::invalid_argument)
        << "GPU graph construction must receive a model-prepared FP32 gate vector; "
           "the stage may not create a host mirror for later upload.";
}

TEST_F(MoEExpertComputeStageTest,
       CombinedSharedVerifierRejectsHotPathGateNormalization)
{
    const int d = 4;

    BF16Tensor gate_inp({1, static_cast<size_t>(d)});
    const float gate_values[d] = {0.03125f, -0.125f, 0.0625f, 0.25f};
    gate_inp.from_fp32(gate_values, d);
    ASSERT_EQ(gate_inp.native_type(), TensorType::BF16);

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.shared_gate_inp = &gate_inp;
    params.combine_shared_expert_in_verifier = true;

    EXPECT_THROW((void)MoEExpertComputeStage(params), std::invalid_argument)
        << "GPU forward must not allocate a host FP32 mirror for a model weight; "
           "weight preparation owns normalization before graph construction.";
}

TEST_F(MoEExpertComputeStageTest,
       ExactExpertMaskMatchIdentifiesSparseCPUOwnershipPublication)
{
    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cpu();
    params.num_experts = 4;
    params.expert_mask = {true, false, true, false};

    MoEExpertComputeStage stage(std::move(params));

    EXPECT_TRUE(stage.expertMaskMatches({true, false, true, false}));
    EXPECT_FALSE(stage.expertMaskMatches({false, true, true, false}));
    EXPECT_FALSE(stage.expertMaskMatches({true, false, true}))
        << "Mask geometry is part of exact publication identity";
}

/**
 * @brief A one-row retained MTP sidecar must not be accounted as decode.
 *
 * Sparse participant-local graph families bind an explicit immutable service
 * phase. Row count cannot recover that identity because both an ordinary
 * decode graph and an MTP predictor graph have M=1. This regression exercises
 * the final stage-to-kernel hint selected after binding, which is the boundary
 * that previously discarded GroupedVerifier and let device histogram deltas
 * misclassify the predictor as decode.
 */
TEST_F(MoEExpertComputeStageTest,
       SparseOneRowMTPPhaseOverridesDecodeShapedGeometry)
{
    constexpr int experts = 1;
    constexpr int top_k = 1;
    constexpr int d_model = 4;

    auto input = TestTensorFactory::createFP32({1, d_model});
    auto routing_indices = TestTensorFactory::createFP32({1, top_k});
    auto routing_weights = TestTensorFactory::createFP32({1, top_k});
    auto output = TestTensorFactory::createFP32({1, d_model});

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.routing_indices = routing_indices.get();
    params.routing_weights = routing_weights.get();
    params.output = output.get();
    params.seq_len = 1;
    params.d_model = d_model;
    params.num_experts = experts;
    params.top_k = top_k;
    params.expert_mask = {false};

    MoEExpertComputeStage stage(std::move(params));
    ASSERT_EQ(
        stage.serviceTelemetryPhaseHintForTesting(),
        MoEOverlayServicePhaseHint::Decode)
        << "Unbound one-row geometry is ordinary decode";

    std::vector<bool> expert_mask{false};
    std::vector<ITensorGemm *> engines(static_cast<size_t>(experts), nullptr);
    ASSERT_TRUE(stage.bindSparseOverlayInvocation(
        MoEExpertComputeStage::SparseOverlayInvocation{
            .input = input.get(),
            .routing_indices = routing_indices.get(),
            .routing_weights = routing_weights.get(),
            .output = output.get(),
            .live_rows = 1,
            .binding_kind =
                MoEExpertComputeStage::SparseOverlayBindingKind::SetupPriming,
            .service_phase = MoEOverlayServicePhaseHint::GroupedVerifier,
            .expert_mask = &expert_mask,
            .gate_engines = engines,
            .up_engines = engines,
            .down_engines = engines,
        }));
    EXPECT_EQ(
        stage.serviceTelemetryPhaseHintForTesting(),
        MoEOverlayServicePhaseHint::GroupedVerifier)
        << "The typed graph role, not M=1, owns service identity";
}

/**
 * @brief Inline sparse execution retains the same MTP phase as replay families.
 *
 * A continuation-domain participant may execute its local sparse packet inline
 * while remote participants use retained replay families. Both paths are one
 * semantic graph family and must publish to the same service plane.
 */
TEST_F(MoEExpertComputeStageTest,
       InlineOneRowMTPPhaseOverridesDecodeShapedGeometry)
{
    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cpu();
    params.seq_len = 1;
    params.service_phase = MoEOverlayServicePhaseHint::GroupedVerifier;

    MoEExpertComputeStage stage(std::move(params));
    EXPECT_EQ(
        stage.serviceTelemetryPhaseHintForTesting(),
        MoEOverlayServicePhaseHint::GroupedVerifier)
        << "Inline MTP predictor service must not be inferred from M=1";
}

TEST_F(MoEExpertComputeStageTest, SharedGate_FusedCombinePublishesGatedSharedAndCombinedOutput)
{
    const int seq = 2;
    const int d = 4;

    auto input = TestTensorFactory::createFP32({seq, d});
    auto gate_inp = TestTensorFactory::createFP32({1, d});
    auto shared_output = TestTensorFactory::createFP32({seq, d});
    auto routed_residual = TestTensorFactory::createFP32({seq, d});
    auto combined_output = TestTensorFactory::createFP32({seq, d});

    for (int i = 0; i < seq * d; ++i)
    {
        input->mutable_data()[i] = static_cast<float>((i % d) + 1);
        shared_output->mutable_data()[i] = 2.0f + static_cast<float>(i);
        routed_residual->mutable_data()[i] = -1.0f + 0.25f * static_cast<float>(i);
        combined_output->mutable_data()[i] = -99.0f;
    }
    for (int i = 0; i < d; ++i)
        gate_inp->mutable_data()[i] = 0.0f; // sigmoid(0) = 0.5 for every row.

    const std::vector<float> original_shared(
        shared_output->data(), shared_output->data() + shared_output->numel());

    SharedExpertGateStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.gate_inp = gate_inp.get();
    params.shared_output = shared_output.get();
    params.routed_residual = routed_residual.get();
    params.combined_output = combined_output.get();
    params.seq_len = seq;
    params.d_model = d;

    SharedExpertGateStage stage(params);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    for (int i = 0; i < seq * d; ++i)
    {
        EXPECT_NEAR(shared_output->data()[i], 0.5f * original_shared[i], 1e-5f)
            << "Fused combine must still materialize the gated shared output";
        EXPECT_NEAR(combined_output->data()[i],
                    routed_residual->data()[i] + 0.5f * original_shared[i],
                    1e-5f)
            << "Fused combine mismatch at element " << i;
    }

    const auto contract = stage.bufferContract();
    EXPECT_EQ(contract.inouts.size(), 1u)
        << "The fused stage reads and rewrites MOE_SHARED_EXPERT_OUTPUT.";
    EXPECT_EQ(contract.outputs.size(), 1u)
        << "The fused stage also writes the final combined output.";
}

TEST_F(MoEExpertComputeStageTest, SharedGate_LargePositiveDotSaturates)
{
    const int seq = 1;
    const int d = 4;

    auto input = TestTensorFactory::createFP32({seq, d});
    auto gate_inp = TestTensorFactory::createFP32({1, d});
    auto shared_output = TestTensorFactory::createFP32({seq, d});

    // Large positive gate → sigmoid ≈ 1.0
    for (int i = 0; i < d; ++i)
    {
        input->mutable_data()[i] = 10.0f;
        gate_inp->mutable_data()[i] = 10.0f;
    }
    for (int i = 0; i < d; ++i)
        shared_output->mutable_data()[i] = 3.0f;

    SharedExpertGateStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.gate_inp = gate_inp.get();
    params.shared_output = shared_output.get();
    params.seq_len = seq;
    params.d_model = d;

    SharedExpertGateStage stage(params);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    // dot = 10*10*4 = 400, sigmoid(400) ≈ 1.0
    // shared_output ≈ 3.0
    const float *out = shared_output->data();
    for (int i = 0; i < d; ++i)
    {
        EXPECT_NEAR(out[i], 3.0f, 1e-4f);
    }
}

TEST_F(MoEExpertComputeStageTest, SharedGate_LargeNegativeDotCanSaturateToZero)
{
    const int seq = 1;
    const int d = 4;

    auto input = TestTensorFactory::createFP32({seq, d});
    auto gate_inp = TestTensorFactory::createFP32({1, d});
    auto shared_output = TestTensorFactory::createFP32({seq, d});

    // dot = 10 * -10 * 4 = -400. The sigmoid underflows to 0.0f when
    // stored as float, so the in-place shared expert output becomes zero.
    for (int i = 0; i < d; ++i)
    {
        input->mutable_data()[i] = 10.0f;
        gate_inp->mutable_data()[i] = -10.0f;
    }
    shared_output->mutable_data()[0] = 3.0f;
    shared_output->mutable_data()[1] = -2.0f;
    shared_output->mutable_data()[2] = 5.0f;
    shared_output->mutable_data()[3] = -7.0f;

    SharedExpertGateStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.gate_inp = gate_inp.get();
    params.shared_output = shared_output.get();
    params.seq_len = seq;
    params.d_model = d;

    SharedExpertGateStage stage(params);
    EXPECT_TRUE(stage.allowsZeroOutput())
        << "Stage verifier must not treat a saturated shared-expert gate as uninitialized";
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    const float *out = shared_output->data();
    for (int i = 0; i < d; ++i)
        EXPECT_EQ(out[i], 0.0f) << "Gate should saturate to zero at dim " << i;
}

TEST_F(MoEExpertComputeStageTest, SharedGate_NullInputReturnsError)
{
    auto shared_output = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});

    SharedExpertGateStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = nullptr;
    params.shared_output = shared_output.get();
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;

    SharedExpertGateStage stage(params);
    EXPECT_FALSE(stage.execute(cpu_ctx_.get()));
}

// =========================================================================
// MoEExpertComputeStage Tests (Router + Expert FFN + Combine)
// =========================================================================

TEST_F(MoEExpertComputeStageTest, MoEFFN_OutputNonZero_Q4K)
{
    // D_MODEL=64 is not divisible by 256 (Q4_K block size).
    // We need cols that are multiples of 256 for quantized tensors.
    // Use d_model=256 for this test.
    const int d = 256;
    const int inter = 256; // Must be multiple of 256
    const int seq = 1;
    const int experts = 4;
    const int topk = 2;

    auto input = TestTensorFactory::createFP32Random({seq, d}, -0.5f, 0.5f, 200);
    auto gate_weights = TestTensorFactory::createFP32Random({experts, d}, -0.1f, 0.1f, 201);
    auto output = TestTensorFactory::createFP32({seq, d});
    std::memset(output->mutable_data(), 0, output->numel() * sizeof(float));

    // Expert tensors: [experts, inter, d] for gate/up, [experts, d, inter] for down
    auto gate_exps = createExpertQ4K(experts, inter, d, 210);
    auto up_exps = createExpertQ4K(experts, inter, d, 211);
    auto down_exps = createExpertQ4K(experts, d, inter, 212);

    // Compute routing externally
    auto routing = computeRouting(input.get(), gate_weights.get(), seq, d, experts, topk);

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.routing_indices = routing.indices.get();
    params.routing_weights = routing.weights.get();
    params.routed_pipeline_kernel_owner = routing.kernel_owner;
    params.gate_exps = gate_exps.get();
    params.up_exps = up_exps.get();
    params.down_exps = down_exps.get();
    params.output = output.get();
    params.seq_len = seq;
    params.d_model = d;
    params.num_experts = experts;
    params.top_k = topk;
    params.expert_intermediate = inter;

    // Extract 2D expert views from 3D packed tensors (required by GEMM path)
    ASSERT_TRUE(MoEExpertComputeStage::extractExpertViews(params));
    ASSERT_TRUE(MoEExpertComputeStage::prepareExpertGemmEngines(params));

    MoEExpertComputeStage stage(params);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    // Output should be non-zero
    const float *out = output->data();
    bool any_nonzero = false;
    for (int i = 0; i < seq * d; ++i)
    {
        if (out[i] != 0.0f)
        {
            any_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero) << "MoEFFN output is all zeros";

    // No NaN/Inf
    for (int i = 0; i < seq * d; ++i)
    {
        EXPECT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(out[i])) << "Inf at index " << i;
    }
}

/**
 * @brief A sparse CPU ExpertOverlay row must publish the same NativeVNNI input.
 *
 * The continuation router is intentionally absent from the second invocation:
 * its compact row models an authoritative CPU-cold packet received through the
 * graph-native sparse collective.  A transport-owned Q8_1 publication must
 * make that serial M=1 execution byte-identical to the normal router-owned
 * decode path.  This prevents the one-row tier case from being masked by the
 * separate grouped-row transport implementation.
 */
TEST_F(MoEExpertComputeStageTest,
       CpuTransportedSerialRowPublishesNativeVNNIRouterQ8BeforeExpertCompute)
{
    ScopedEnv perf_env("LLAMINAR_PERF_STATS_SUMMARY", "1");
    ScopedOpenMPThreadCount omp_threads(/*threads=*/1);
    PerfStatsCollector::reset();

    constexpr int d = 256;
    constexpr int intermediate = 256;
    constexpr int experts = 4;
    constexpr int top_k = 2;

    auto input = TestTensorFactory::createFP32Random(
        {1, d}, -0.5f, 0.5f, 220);
    auto router_weights = TestTensorFactory::createFP32Random(
        {experts, d}, -0.1f, 0.1f, 221);
    auto gate_experts = createExpertQ4K(experts, intermediate, d, 222);
    auto up_experts = createExpertQ4K(experts, intermediate, d, 223);
    auto down_experts = createExpertQ4K(experts, d, intermediate, 224);
    const auto routing = computeRouting(
        input.get(), router_weights.get(), 1, d, experts, top_k);

    const auto make_params = [&]()
    {
        MoEExpertComputeStage::Params params;
        params.device_id = DeviceId::cpu();
        params.input = input.get();
        params.routing_indices = routing.indices.get();
        params.routing_weights = routing.weights.get();
        params.gate_exps = gate_experts.get();
        params.up_exps = up_experts.get();
        params.down_exps = down_experts.get();
        params.seq_len = 1;
        params.d_model = d;
        params.num_experts = experts;
        params.top_k = top_k;
        params.expert_intermediate = intermediate;
        params.layer_idx = 37;
        return params;
    };

    auto router_output = TestTensorFactory::createFP32({1, d});
    auto router_params = make_params();
    router_params.output = router_output.get();
    router_params.routed_pipeline_kernel_owner = routing.kernel_owner;
    ASSERT_TRUE(MoEExpertComputeStage::extractExpertViews(router_params));
    ASSERT_TRUE(MoEExpertComputeStage::prepareExpertGemmEngines(router_params));
    MoEExpertComputeStage router_stage(std::move(router_params));
    ASSERT_TRUE(router_stage.execute(cpu_ctx_.get()));

    auto transported_output = TestTensorFactory::createFP32({1, d});
    auto transported_params = make_params();
    transported_params.output = transported_output.get();
    transported_params.cpu_router_q8_input_publication =
        CPURouterQ8InputPublicationPolicy::PublishTransportedRows;
    ASSERT_TRUE(MoEExpertComputeStage::extractExpertViews(transported_params));
    ASSERT_TRUE(MoEExpertComputeStage::prepareExpertGemmEngines(transported_params));
    MoEExpertComputeStage transported_stage(std::move(transported_params));
    ASSERT_TRUE(transported_stage.execute(cpu_ctx_.get()));

    expectRowsByteEqual(
        transported_output->data(),
        router_output->data(),
        /*rows=*/1,
        d,
        "CPU transported serial NativeVNNI expert row");

    const auto records = PerfStatsCollector::snapshot({"moe_overlay"});
    EXPECT_TRUE(std::any_of(
        records.begin(),
        records.end(),
        [](const PerfStatRecord &record)
        {
            const auto source = record.tags.find("source");
            return record.domain == "moe_overlay" &&
                   record.name == "cpu_transport_router_q8_publications" &&
                   record.phase == "decode" &&
                   source != record.tags.end() &&
                   source->second == "compact_transport_serial_row" &&
                   record.count > 0;
        }))
        << PerfStatsCollector::summaryString({"moe_overlay"});
    PerfStatsCollector::reset();
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_OutputNonZero_Q5K)
{
    const int d = 256;
    const int inter = 256;
    const int seq = 1;
    const int experts = 4;
    const int topk = 2;

    auto input = TestTensorFactory::createFP32Random({seq, d}, -0.5f, 0.5f, 300);
    auto gate_weights = TestTensorFactory::createFP32Random({experts, d}, -0.1f, 0.1f, 301);
    auto output = TestTensorFactory::createFP32({seq, d});
    std::memset(output->mutable_data(), 0, output->numel() * sizeof(float));

    // Use Q5_K for down_exps (like the real model)
    auto gate_exps = createExpertQ4K(experts, inter, d, 310);
    auto up_exps = createExpertQ4K(experts, inter, d, 311);
    auto down_exps = createExpertQ5K(experts, d, inter, 312);

    // Compute routing externally
    auto routing = computeRouting(input.get(), gate_weights.get(), seq, d, experts, topk);

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.routing_indices = routing.indices.get();
    params.routing_weights = routing.weights.get();
    params.routed_pipeline_kernel_owner = routing.kernel_owner;
    params.gate_exps = gate_exps.get();
    params.up_exps = up_exps.get();
    params.down_exps = down_exps.get();
    params.output = output.get();
    params.seq_len = seq;
    params.d_model = d;
    params.num_experts = experts;
    params.top_k = topk;
    params.expert_intermediate = inter;

    ASSERT_TRUE(MoEExpertComputeStage::extractExpertViews(params));
    ASSERT_TRUE(MoEExpertComputeStage::prepareExpertGemmEngines(params));

    MoEExpertComputeStage stage(params);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    const float *out = output->data();
    bool any_nonzero = false;
    for (int i = 0; i < seq * d; ++i)
    {
        if (out[i] != 0.0f)
        {
            any_nonzero = true;
            break;
        }
        EXPECT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(out[i])) << "Inf at index " << i;
    }
    EXPECT_TRUE(any_nonzero) << "MoEFFN Q5K output is all zeros";
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_MultipleTokens)
{
    const int d = 256;
    const int inter = 256;
    const int seq = 4;
    const int experts = 4;
    const int topk = 2;

    auto input = TestTensorFactory::createFP32Random({seq, d}, -0.5f, 0.5f, 400);
    auto gate_weights = TestTensorFactory::createFP32Random({experts, d}, -0.1f, 0.1f, 401);
    auto output = TestTensorFactory::createFP32({seq, d});
    std::memset(output->mutable_data(), 0, output->numel() * sizeof(float));

    auto gate_exps = createExpertQ4K(experts, inter, d, 410);
    auto up_exps = createExpertQ4K(experts, inter, d, 411);
    auto down_exps = createExpertQ4K(experts, d, inter, 412);

    // Compute routing externally
    auto routing = computeRouting(input.get(), gate_weights.get(), seq, d, experts, topk);

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.routing_indices = routing.indices.get();
    params.routing_weights = routing.weights.get();
    params.routed_pipeline_kernel_owner = routing.kernel_owner;
    params.gate_exps = gate_exps.get();
    params.up_exps = up_exps.get();
    params.down_exps = down_exps.get();
    params.output = output.get();
    params.seq_len = seq;
    params.d_model = d;
    params.num_experts = experts;
    params.top_k = topk;
    params.expert_intermediate = inter;

    ASSERT_TRUE(MoEExpertComputeStage::extractExpertViews(params));
    ASSERT_TRUE(MoEExpertComputeStage::prepareExpertGemmEngines(params));

    MoEExpertComputeStage stage(params);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    // Each token should get a non-zero output
    const float *out = output->data();
    for (int t = 0; t < seq; ++t)
    {
        bool token_nonzero = false;
        for (int dd = 0; dd < d; ++dd)
        {
            float v = out[t * d + dd];
            EXPECT_FALSE(std::isnan(v)) << "NaN at token " << t << " dim " << dd;
            if (v != 0.0f)
                token_nonzero = true;
        }
        EXPECT_TRUE(token_nonzero) << "Token " << t << " output is all zeros";
    }
}

TEST_F(MoEExpertComputeStageTest, SharedExpert_RuntimeMVerifierAllNativeFormatsMatchSerialDecodeByteExact)
{
    ScopedEnv perf_env("LLAMINAR_PERF_STATS_SUMMARY", "1");
    PerfStatsCollector::reset();

    const int d = 256;
    const int inter = 256;

    const auto formats = cpuVerifierNativeFormats();
    for (size_t format_index = 0; format_index < formats.size(); ++format_index)
    {
        const auto &format = formats[format_index];
        SCOPED_TRACE(format.label);
        PerfStatsCollector::reset();

        auto gate_w = format.create(
            {static_cast<size_t>(inter), static_cast<size_t>(d)},
            static_cast<int>(701 + 10 * format_index));
        auto up_w = format.create(
            {static_cast<size_t>(inter), static_cast<size_t>(d)},
            static_cast<int>(702 + 10 * format_index));
        auto down_w = format.create(
            {static_cast<size_t>(d), static_cast<size_t>(inter)},
            static_cast<int>(703 + 10 * format_index));

        auto run_shared = [&](TensorBase *run_input, TensorBase *run_output, int run_seq, int layer)
        {
            auto prepared = makePreparedFFNFixture(
                gate_w.get(), up_w.get(), down_w.get(), DeviceId::cpu(), layer, "ffn_shexp");

            SharedExpertFFNStage::Params params;
            params.device_id = DeviceId::cpu();
            params.input = run_input;
            params.gate_w = gate_w.get();
            params.up_w = up_w.get();
            params.down_w = down_w.get();
            params.output = run_output;
            params.seq_len = run_seq;
            params.d_model = d;
            params.intermediate = inter;
            params.prepared_ref_gate = prepared.gate_ref;
            params.prepared_ref_up = prepared.up_ref;
            params.prepared_ref_down = prepared.down_ref;
            params.prepared_store = prepared.store.get();
            params.force_decode_equivalent_verifier_prefill = run_seq > 1;

            SharedExpertFFNStage stage(params);
            if (run_seq > 1)
                EXPECT_TRUE(stage.usesCPUDecodeEquivalentVerifierPrefillForTesting());
            return stage.execute(cpu_ctx_.get());
        };

        for (const int seq : kGroupedVerifierRuntimeRows)
        {
            SCOPED_TRACE("seq=" + std::to_string(seq));
            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(seq), static_cast<size_t>(d)},
                -0.5f,
                0.5f,
                700 + seq);
            auto multi_output = TestTensorFactory::createFP32({static_cast<size_t>(seq), static_cast<size_t>(d)});
            auto serial_output = TestTensorFactory::createFP32({static_cast<size_t>(seq), static_cast<size_t>(d)});

            ASSERT_TRUE(run_shared(
                input.get(),
                multi_output.get(),
                seq,
                static_cast<int>(1000 + 100 * format_index + seq)));

            for (int row = 0; row < seq; ++row)
            {
                FP32Tensor row_input({1, static_cast<size_t>(d)});
                FP32Tensor row_output({1, static_cast<size_t>(d)});
                std::copy_n(input->data() + static_cast<size_t>(row) * d,
                            d,
                            row_input.mutable_data());
                ASSERT_TRUE(run_shared(
                    &row_input,
                    &row_output,
                    1,
                    static_cast<int>(2000 + 100 * format_index + seq * 10 + row)));
                std::copy_n(row_output.data(),
                            d,
                            serial_output->mutable_data() + static_cast<size_t>(row) * d);
            }

            expectRowsByteEqual(
                multi_output->data(),
                serial_output->data(),
                seq,
                d,
                std::string("shared expert verifier format=") + format.label);
        }

        const auto records = PerfStatsCollector::snapshot({"mtp", "kernel"});
        EXPECT_TRUE(hasPerfCounterWithRoute(
            records,
            "mtp",
            "moe_shared_grouped_decode_equivalent_verifier_prefill_rows",
            "cpu_grouped_verifier_hooks"))
            << "CPU shared expert verifier must use grouped verifier hooks, not row replay for "
            << format.label << ".\n"
            << PerfStatsCollector::summaryString({"mtp", "kernel"});
        EXPECT_TRUE(std::any_of(
            records.begin(),
            records.end(),
            [](const PerfStatRecord &record)
            {
                return record.domain == "kernel" &&
                       record.name == "cpu_native_vnni_grouped_verifier_swiglu_down_calls";
            }))
            << "CPU shared expert verifier must exercise the grouped SwiGLU/down kernel for "
            << format.label << ".\n"
            << PerfStatsCollector::summaryString({"mtp", "kernel"});
    }
    PerfStatsCollector::reset();
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_RuntimeMVerifierMatchesSerialDecode_AllNativeFormatsRouterQ8Reuse)
{
    ScopedEnv perf_env("LLAMINAR_PERF_STATS_SUMMARY", "1");
    ScopedOpenMPThreadCount omp_threads(/*threads=*/1);
    constexpr int d = 256;
    constexpr int inter = 256;
    constexpr int experts = 4;
    constexpr int topk = 2;

    const auto formats = cpuVerifierNativeFormats();
    for (size_t format_index = 0; format_index < formats.size(); ++format_index)
    {
        const auto &format = formats[format_index];
        SCOPED_TRACE(format.label);
        PerfStatsCollector::reset();

        auto gate_weights = TestTensorFactory::createFP32Random(
            {experts, d},
            -0.1f,
            0.1f,
            static_cast<uint32_t>(721 + format_index));
        std::vector<std::shared_ptr<TensorBase>> gate_views;
        std::vector<std::shared_ptr<TensorBase>> up_views;
        std::vector<std::shared_ptr<TensorBase>> down_views;
        gate_views.reserve(experts);
        up_views.reserve(experts);
        down_views.reserve(experts);
        for (int expert = 0; expert < experts; ++expert)
        {
            gate_views.emplace_back(format.create(
                {static_cast<size_t>(inter), static_cast<size_t>(d)},
                static_cast<int>(730 + 100 * format_index + expert)));
            up_views.emplace_back(format.create(
                {static_cast<size_t>(inter), static_cast<size_t>(d)},
                static_cast<int>(740 + 100 * format_index + expert)));
            down_views.emplace_back(format.create(
                {static_cast<size_t>(d), static_cast<size_t>(inter)},
                static_cast<int>(750 + 100 * format_index + expert)));
        }

        auto run_moe = [&](TensorBase *run_input,
                           TensorBase *run_indices,
                           TensorBase *run_weights,
                           const std::shared_ptr<MoERoutedPipelineKernelOwner> &kernel_owner,
                           TensorBase *run_output,
                           int run_seq,
                           bool verifier_rows,
                           const std::vector<bool> *run_expert_mask,
                           TensorBase *run_canonical_routes = nullptr)
        {
            MoEExpertComputeStage::Params params;
            params.device_id = DeviceId::cpu();
            params.input = run_input;
            params.routing_indices = run_indices;
            params.routing_weights = run_weights;
            params.routed_pipeline_kernel_owner = kernel_owner;
            params.expert_gate_views = gate_views;
            params.expert_up_views = up_views;
            params.expert_down_views = down_views;
            params.output = run_output;
            params.canonical_route_contributions = run_canonical_routes;
            params.canonical_route_arithmetic = run_canonical_routes
                ? MoECanonicalRouteArithmeticPolicy::
                      UnweightedExpertRowThenOrderedFMA
                : MoECanonicalRouteArithmeticPolicy::Unspecified;
            params.canonical_route_layout = run_canonical_routes
                ? MoECanonicalRoutePublicationLayout::
                      PackedIndexedRouteRows
                : MoECanonicalRoutePublicationLayout::Unspecified;
            params.seq_len = run_seq;
            params.d_model = d;
            params.num_experts = experts;
            params.top_k = topk;
            params.expert_intermediate = inter;
            params.force_decode_equivalent_verifier_prefill = verifier_rows;
            if (run_expert_mask)
            {
                /*
                 * Keep the contiguous range deliberately broad. The explicit
                 * ownership mask is authoritative, so a latent range-based
                 * scheduler will compute an extra expert and fail byte parity.
                 */
                params.local_expert_start = 0;
                params.local_expert_count = experts;
                params.expert_mask = *run_expert_mask;
            }

            if (!MoEExpertComputeStage::prepareExpertGemmEngines(params))
            {
                return false;
            }
            MoEExpertComputeStage stage(params);
            return stage.execute(cpu_ctx_.get());
        };

        constexpr int owner_participants = 2;
        constexpr int owner_layer = 0;
        const std::array<std::vector<bool>, owner_participants> random_owner_masks = {
            routed_expert_ownership::expertMaskForParticipant(
                experts,
                owner_participants,
                0,
                owner_layer,
                RoutedExpertOwnerOrder::Random),
            routed_expert_ownership::expertMaskForParticipant(
                experts,
                owner_participants,
                1,
                owner_layer,
                RoutedExpertOwnerOrder::Random)};
        ASSERT_EQ(
            random_owner_masks[0],
            (std::vector<bool>{true, false, true, false}));
        ASSERT_EQ(
            random_owner_masks[1],
            (std::vector<bool>{false, true, false, true}));

        const std::array<std::array<std::vector<bool>, owner_participants>, 2>
            ownership_maps = {{
                {
                    routed_expert_ownership::expertMaskForParticipant(
                        experts,
                        owner_participants,
                        0,
                        owner_layer,
                        RoutedExpertOwnerOrder::Ordinal),
                    routed_expert_ownership::expertMaskForParticipant(
                        experts,
                        owner_participants,
                        1,
                        owner_layer,
                        RoutedExpertOwnerOrder::Ordinal),
                },
                random_owner_masks,
            }};
        const std::array<const char *, 2> ownership_labels = {
            "ordinal",
            "random",
        };

        auto prove_distributed_canonical_rows =
            [&](TensorBase *run_input,
                TensorBase *run_indices,
                TensorBase *run_weights,
                const std::shared_ptr<MoERoutedPipelineKernelOwner> &kernel_owner,
                TensorBase *serial_rows,
                int run_seq,
                bool verifier_rows)
        {
            std::vector<float> first_ownership_result;
            for (size_t ownership_index = 0;
                 ownership_index < ownership_maps.size();
                 ++ownership_index)
            {
                SCOPED_TRACE(
                    std::string("distributed_owner_map=") +
                    ownership_labels[ownership_index] +
                    " seq=" + std::to_string(run_seq));

                const size_t route_slot_count =
                    static_cast<size_t>(run_seq) *
                    static_cast<size_t>(topk);
                const size_t record_width =
                    canonical_moe_route_record::recordWidth(d);
                const size_t packed_elements =
                    route_slot_count * record_width +
                    canonical_moe_route_record::kTrailerElements;
                std::array<std::shared_ptr<FP32Tensor>, owner_participants>
                    participant_routes;
                for (int participant = 0;
                     participant < owner_participants;
                     ++participant)
                {
                    participant_routes[static_cast<size_t>(participant)] =
                        TestTensorFactory::createFP32(
                            {packed_elements});
                    auto unused_compact_output =
                        TestTensorFactory::createFP32(
                            {static_cast<size_t>(run_seq),
                             static_cast<size_t>(d)});
                    ASSERT_TRUE(run_moe(
                        run_input,
                        run_indices,
                        run_weights,
                        kernel_owner,
                        unused_compact_output.get(),
                        run_seq,
                        verifier_rows,
                        &ownership_maps[ownership_index]
                             [static_cast<size_t>(participant)],
                        participant_routes[static_cast<size_t>(participant)]
                            .get()));
                }

                /*
                 * Model the production variable gather without MPI. Copy rank
                 * blocks in reverse participant order so gathered record order
                 * cannot accidentally match the router's row/top-k order. The
                 * root reducer must use each record's exact slot metadata.
                 */
                auto gathered_routes =
                    TestTensorFactory::createFP32({packed_elements});
                size_t gathered_count = 0u;
                for (int participant = owner_participants - 1;
                     participant >= 0;
                     --participant)
                {
                    const auto &participant_records =
                        participant_routes[static_cast<size_t>(participant)];
                    const size_t participant_count =
                        canonical_moe_route_record::readRecordCount(
                            participant_records->data(),
                            participant_records->numel());
                    ASSERT_LE(
                        gathered_count + participant_count,
                        route_slot_count);
                    std::copy_n(
                        participant_records->data(),
                        participant_count * record_width,
                        gathered_routes->mutable_data() +
                            gathered_count * record_width);
                    gathered_count += participant_count;
                }
                ASSERT_TRUE(canonical_moe_route_record::writeRecordCount(
                    gathered_routes->mutable_data(),
                    gathered_routes->numel(),
                    gathered_count));

                auto canonical_output = TestTensorFactory::createFP32(
                    {static_cast<size_t>(run_seq),
                     static_cast<size_t>(d)});
                MoECanonicalRouteReduceStage::Params reduce_params;
                reduce_params.device_id = DeviceId::cpu();
                reduce_params.canonical_route_contributions =
                    gathered_routes.get();
                reduce_params.routing_weights = run_weights;
                reduce_params.output = canonical_output.get();
                reduce_params.seq_len = run_seq;
                reduce_params.top_k = topk;
                reduce_params.d_model = d;
                reduce_params.canonical_route_arithmetic =
                    MoECanonicalRouteArithmeticPolicy::
                        UnweightedExpertRowThenOrderedFMA;
                reduce_params.canonical_route_layout =
                    MoECanonicalRoutePublicationLayout::
                        PackedIndexedRouteRows;
                reduce_params.reduction_role =
                    MoECanonicalRouteReductionRole::RootOwner;
                MoECanonicalRouteReduceStage reduce_stage(
                    std::move(reduce_params));
                ASSERT_TRUE(reduce_stage.execute(cpu_ctx_.get()));

                const std::string context =
                    std::string("CPU distributed canonical ") +
                    (verifier_rows ? "grouped verifier" : "decode") +
                    " ownership=" + ownership_labels[ownership_index] +
                    " format=" + format.label +
                    " M=" + std::to_string(run_seq);
                expectRowsByteEqual(
                    canonical_output->data(),
                    serial_rows->data(),
                    run_seq,
                    d,
                    context);

                if (first_ownership_result.empty())
                {
                    first_ownership_result.assign(
                        canonical_output->data(),
                        canonical_output->data() + canonical_output->numel());
                }
                else
                {
                    expectRowsByteEqual(
                        canonical_output->data(),
                        first_ownership_result.data(),
                        run_seq,
                        d,
                        context + " versus prior ownership map");
                }
            }
        };

        {
            constexpr int decode_rows = 1;
            auto decode_input = TestTensorFactory::createFP32Random(
                {decode_rows, d},
                -0.5f,
                0.5f,
                static_cast<uint32_t>(790 + 100 * format_index));
            auto decode_routing = computeRouting(
                decode_input.get(),
                gate_weights.get(),
                decode_rows,
                d,
                experts,
                topk);
            auto serial_decode_output = TestTensorFactory::createFP32(
                {decode_rows, d});
            ASSERT_TRUE(run_moe(
                decode_input.get(),
                decode_routing.indices.get(),
                decode_routing.weights.get(),
                decode_routing.kernel_owner,
                serial_decode_output.get(),
                decode_rows,
                /*verifier_rows=*/false,
                /*run_expert_mask=*/nullptr));
            prove_distributed_canonical_rows(
                decode_input.get(),
                decode_routing.indices.get(),
                decode_routing.weights.get(),
                decode_routing.kernel_owner,
                serial_decode_output.get(),
                decode_rows,
                /*verifier_rows=*/false);
        }

        bool observed_grouped_multirow_launch = false;
        for (const int seq : kGroupedVerifierRuntimeRows)
        {
            SCOPED_TRACE("seq=" + std::to_string(seq));
            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(seq), static_cast<size_t>(d)},
                -0.5f,
                0.5f,
                static_cast<uint32_t>(720 + seq + 100 * format_index));
            auto grouped_output = TestTensorFactory::createFP32(
                {static_cast<size_t>(seq), static_cast<size_t>(d)});
            auto ordinary_prefill_output = TestTensorFactory::createFP32(
                {static_cast<size_t>(seq), static_cast<size_t>(d)});
            auto serial_output = TestTensorFactory::createFP32(
                {static_cast<size_t>(seq), static_cast<size_t>(d)});

            auto grouped_routing = computeRouting(
                input.get(), gate_weights.get(), seq, d, experts, topk);
            /*
             * Sparse transport uses {-1, 0.0f} as an inert route slot. Keep
             * that production sentinel in the all-format byte-equivalence
             * sweep so grouped ordinary-prefill and verifier scheduling prove
             * that padding neither fails validation nor changes arithmetic.
             */
            grouped_routing.indices->mutable_data()[topk - 1] = -1.0f;
            grouped_routing.weights->mutable_data()[topk - 1] = 0.0f;
            ASSERT_TRUE(run_moe(
                input.get(),
                grouped_routing.indices.get(),
                grouped_routing.weights.get(),
                grouped_routing.kernel_owner,
                grouped_output.get(),
                seq,
                /*verifier_rows=*/true,
                /*run_expert_mask=*/nullptr));
            ASSERT_TRUE(run_moe(
                input.get(),
                grouped_routing.indices.get(),
                grouped_routing.weights.get(),
                grouped_routing.kernel_owner,
                ordinary_prefill_output.get(),
                seq,
                /*verifier_rows=*/false,
                /*run_expert_mask=*/nullptr));

            for (int row = 0; row < seq; ++row)
            {
                FP32Tensor row_input({1u, static_cast<size_t>(d)});
                FP32Tensor row_output({1u, static_cast<size_t>(d)});
                std::copy_n(
                    input->data() + static_cast<size_t>(row) * d,
                    d,
                    row_input.mutable_data());
                auto row_routing = computeRouting(
                    &row_input, gate_weights.get(), 1, d, experts, topk);
                if (row == 0)
                {
                    row_routing.indices->mutable_data()[topk - 1] = -1.0f;
                    row_routing.weights->mutable_data()[topk - 1] = 0.0f;
                }
                ASSERT_TRUE(run_moe(
                    &row_input,
                    row_routing.indices.get(),
                    row_routing.weights.get(),
                    row_routing.kernel_owner,
                    &row_output,
                    1,
                    /*verifier_rows=*/false,
                    /*run_expert_mask=*/nullptr));
                std::copy_n(
                    row_output.data(),
                    d,
                    serial_output->mutable_data() + static_cast<size_t>(row) * d);
            }

            expectRowsByteEqual(
                grouped_output->data(),
                serial_output->data(),
                seq,
                d,
                std::string("CPU routed expert verifier format=") + format.label);
            expectRowsByteEqual(
                ordinary_prefill_output->data(),
                serial_output->data(),
                seq,
                d,
                std::string("CPU routed expert ordinary prefill format=") +
                    format.label);

            if (std::find(
                    kGroupedVerifierBoundaryRows.begin(),
                    kGroupedVerifierBoundaryRows.end(),
                    seq) != kGroupedVerifierBoundaryRows.end())
            {
                prove_distributed_canonical_rows(
                    input.get(),
                    grouped_routing.indices.get(),
                    grouped_routing.weights.get(),
                    grouped_routing.kernel_owner,
                    serial_output.get(),
                    seq,
                    /*verifier_rows=*/true);
            }

            for (int participant = 0;
                 participant < owner_participants;
                 ++participant)
            {
                SCOPED_TRACE(
                    "random_owner_participant=" + std::to_string(participant));
                const std::vector<bool> &owner_mask =
                    random_owner_masks[static_cast<size_t>(participant)];
                auto masked_grouped_output = TestTensorFactory::createFP32(
                    {static_cast<size_t>(seq), static_cast<size_t>(d)});
                auto masked_prefill_output = TestTensorFactory::createFP32(
                    {static_cast<size_t>(seq), static_cast<size_t>(d)});
                auto masked_serial_output = TestTensorFactory::createFP32(
                    {static_cast<size_t>(seq), static_cast<size_t>(d)});

                ASSERT_TRUE(run_moe(
                    input.get(),
                    grouped_routing.indices.get(),
                    grouped_routing.weights.get(),
                    grouped_routing.kernel_owner,
                    masked_grouped_output.get(),
                    seq,
                    /*verifier_rows=*/true,
                    &owner_mask));
                ASSERT_TRUE(run_moe(
                    input.get(),
                    grouped_routing.indices.get(),
                    grouped_routing.weights.get(),
                    grouped_routing.kernel_owner,
                    masked_prefill_output.get(),
                    seq,
                    /*verifier_rows=*/false,
                    &owner_mask));

                for (int row = 0; row < seq; ++row)
                {
                    FP32Tensor row_input({1u, static_cast<size_t>(d)});
                    FP32Tensor row_output({1u, static_cast<size_t>(d)});
                    std::copy_n(
                        input->data() + static_cast<size_t>(row) * d,
                        d,
                        row_input.mutable_data());
                    auto row_routing = computeRouting(
                        &row_input, gate_weights.get(), 1, d, experts, topk);
                    if (row == 0)
                    {
                        row_routing.indices->mutable_data()[topk - 1] = -1.0f;
                        row_routing.weights->mutable_data()[topk - 1] = 0.0f;
                    }
                    ASSERT_TRUE(run_moe(
                        &row_input,
                        row_routing.indices.get(),
                        row_routing.weights.get(),
                        row_routing.kernel_owner,
                        &row_output,
                        1,
                        /*verifier_rows=*/false,
                        &owner_mask));
                    std::copy_n(
                        row_output.data(),
                        d,
                        masked_serial_output->mutable_data() +
                            static_cast<size_t>(row) * d);
                }

                const std::string ownership_context =
                    " random-owner participant=" +
                    std::to_string(participant) + " format=" + format.label;
                expectRowsByteEqual(
                    masked_grouped_output->data(),
                    masked_serial_output->data(),
                    seq,
                    d,
                    "CPU routed expert verifier" + ownership_context);
                expectRowsByteEqual(
                    masked_prefill_output->data(),
                    masked_serial_output->data(),
                    seq,
                    d,
                    "CPU routed expert ordinary prefill" + ownership_context);
            }

            const auto records = PerfStatsCollector::snapshot(
                {"mtp", "prefill", "kernel"});
            const std::string expected_seq_len = std::to_string(seq);
            const std::string expected_route_rows =
                std::to_string(seq * topk - 1);
            EXPECT_TRUE(std::any_of(
                records.begin(),
                records.end(),
                [&](const PerfStatRecord &record)
                {
                    const auto seq_it = record.tags.find("seq_len");
                    return record.domain == "kernel" &&
                           record.name ==
                               "cpu_moe_grouped_verifier_router_q8_reuse_calls" &&
                           seq_it != record.tags.end() &&
                           seq_it->second == expected_seq_len &&
                           record.count > 0;
                }))
                << "CPU " << format.label << " M=" << seq
                << " must consume router-published Q8_1 rows.\n"
                << PerfStatsCollector::summaryString({"mtp", "kernel"});
            EXPECT_TRUE(std::any_of(
                records.begin(),
                records.end(),
                [&](const PerfStatRecord &record)
                {
                    const auto seq_it = record.tags.find("seq_len");
                    return record.domain == "kernel" &&
                           record.name ==
                               "cpu_moe_prefill_router_q8_reuse_calls" &&
                           seq_it != record.tags.end() &&
                           seq_it->second == expected_seq_len &&
                           record.count > 0;
                }))
                << "CPU " << format.label << " M=" << seq
                << " ordinary prefill must consume router-published Q8_1 rows.\n"
                << PerfStatsCollector::summaryString({"prefill", "kernel"});
            EXPECT_TRUE(std::any_of(
                records.begin(),
                records.end(),
                [](const PerfStatRecord &record)
                {
                    return record.domain == "kernel" &&
                           record.name ==
                               "cpu_native_vnni_router_q8_grouped_decode_equivalent_projection_calls" &&
                           record.count > 0;
                }))
                << "CPU " << format.label
                << " verifier must execute the prequantized NativeVNNI grouped gate/up kernel.\n"
                << PerfStatsCollector::summaryString({"mtp", "kernel"});
            EXPECT_TRUE(std::any_of(
                records.begin(),
                records.end(),
                [&](const PerfStatRecord &record)
                {
                    const auto rows_it = record.tags.find("local_route_rows");
                    return record.domain == "kernel" &&
                           record.name ==
                               "cpu_moe_native_vnni_layer_batched_expert_calls" &&
                           record.phase == "prefill" &&
                           rows_it != record.tags.end() &&
                           rows_it->second == expected_route_rows &&
                           record.count > 0;
                }))
                << "CPU " << format.label << " M=" << seq
                << " ordinary prefill must execute the layer-batched "
                   "NativeVNNI production path.\n"
                << PerfStatsCollector::summaryString({"prefill", "kernel"});
            EXPECT_TRUE(std::any_of(
                records.begin(),
                records.end(),
                [&](const PerfStatRecord &record)
                {
                    const auto rows_it = record.tags.find("local_route_rows");
                    return record.domain == "kernel" &&
                           record.name ==
                               "cpu_moe_native_vnni_layer_batched_expert_calls" &&
                           record.phase == "verifier" &&
                           rows_it != record.tags.end() &&
                           rows_it->second == expected_route_rows &&
                           record.count > 0;
                }))
                << "CPU " << format.label << " M=" << seq
                << " verifier must execute the layer-batched NativeVNNI "
                   "production path.\n"
                << PerfStatsCollector::summaryString({"mtp", "kernel"});
            observed_grouped_multirow_launch =
                observed_grouped_multirow_launch ||
                std::any_of(
                    records.begin(),
                    records.end(),
                    [](const PerfStatRecord &record)
                    {
                        const auto m_it = record.tags.find("m");
                        const auto route_it = record.tags.find("route");
                        return record.domain == "kernel" &&
                               record.name ==
                                   "cpu_native_vnni_fused_verifier_rows_projection_launch" &&
                               m_it != record.tags.end() &&
                               std::stoi(m_it->second) > 1 &&
                               route_it != record.tags.end() &&
                               route_it->second.rfind("grouped_", 0) == 0 &&
                               record.count > 0;
                    });
            EXPECT_TRUE(std::any_of(
                records.begin(),
                records.end(),
                [](const PerfStatRecord &record)
                {
                    return record.domain == "kernel" &&
                           record.name == "cpu_moe_decode_router_q8_reuse_calls" &&
                           record.count > 0;
                }))
                << "CPU " << format.label
                << " serial oracle must use production M=1 router-Q8 reuse.\n"
                << PerfStatsCollector::summaryString({"mtp", "kernel"});
        }

        EXPECT_TRUE(observed_grouped_multirow_launch)
            << "CPU " << format.label
            << " layer-batched production execution must use a real multi-row "
               "grouped launch when an expert receives multiple rows.\n"
            << PerfStatsCollector::summaryString({"prefill", "kernel"});

        const auto records = PerfStatsCollector::snapshot({"mtp", "kernel"});
        EXPECT_TRUE(hasPerfCounterWithRoute(
            records,
            "mtp",
            "moe_routed_grouped_decode_equivalent_verifier_prefill_rows",
            "cpu_expert_slot_grouped"))
            << "CPU routed expert verifier must use grouped expert-slot execution for "
            << format.label << ".\n"
            << PerfStatsCollector::summaryString({"mtp", "kernel"});
        EXPECT_TRUE(hasPerfCounterWithRoute(
            PerfStatsCollector::snapshot({"prefill", "kernel"}),
            "prefill",
            "moe_routed_grouped_decode_equivalent_prefill_rows",
            "cpu_expert_slot_grouped"))
            << "CPU routed expert ordinary prefill must use grouped expert-slot execution for "
            << format.label << ".\n"
            << PerfStatsCollector::summaryString({"prefill", "kernel"});
    }
    PerfStatsCollector::reset();
}

/**
 * @brief Require packed canonical publication to reject malformed transactions.
 *
 * Sparse participant gathers are intentionally unordered, but they must contain
 * exactly one record for every live original router slot and no record for an
 * inert slot. A duplicate, omission, or inert payload indicates a broken
 * ownership/publication transaction and must fail rather than produce a
 * numerically plausible partial row.
 */
TEST_F(
    MoEExpertComputeStageTest,
    PackedCanonicalRouteReducerRejectsDuplicateMissingAndInertSlots)
{
    constexpr int seq_len = 2;
    constexpr int top_k = 2;
    constexpr int d_model = 4;
    constexpr size_t route_slot_count =
        static_cast<size_t>(seq_len * top_k);
    constexpr size_t record_width =
        canonical_moe_route_record::recordWidth(d_model);
    constexpr size_t packed_elements =
        route_slot_count * record_width +
        canonical_moe_route_record::kTrailerElements;

    FP32Tensor routing_weights(
        {static_cast<size_t>(seq_len), static_cast<size_t>(top_k)});
    const std::array<float, route_slot_count> weights = {
        0.5f,
        0.25f,
        0.0f,
        0.75f,
    };
    std::copy(weights.begin(), weights.end(), routing_weights.mutable_data());

    const auto execute_slots =
        [&](const std::vector<uint32_t> &flat_slots) -> bool
    {
        FP32Tensor packed_records({packed_elements});
        FP32Tensor output(
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        for (size_t record_index = 0;
             record_index < flat_slots.size();
             ++record_index)
        {
            float *const route_record =
                canonical_moe_route_record::record(
                    packed_records.mutable_data(),
                    record_index,
                    d_model);
            for (int column = 0; column < d_model; ++column)
            {
                route_record[static_cast<size_t>(column)] =
                    static_cast<float>(flat_slots[record_index] * 10u +
                                       static_cast<uint32_t>(column + 1));
            }
            if (!canonical_moe_route_record::writeFlatRouteSlot(
                    route_record,
                    d_model,
                    static_cast<size_t>(flat_slots[record_index])))
            {
                return false;
            }
        }
        if (!canonical_moe_route_record::writeRecordCount(
                packed_records.mutable_data(),
                packed_records.numel(),
                flat_slots.size()))
        {
            return false;
        }

        MoECanonicalRouteReduceStage::Params params;
        params.device_id = DeviceId::cpu();
        params.canonical_route_contributions = &packed_records;
        params.routing_weights = &routing_weights;
        params.output = &output;
        params.seq_len = seq_len;
        params.top_k = top_k;
        params.d_model = d_model;
        params.canonical_route_arithmetic =
            MoECanonicalRouteArithmeticPolicy::
                UnweightedExpertRowThenOrderedFMA;
        params.canonical_route_layout =
            MoECanonicalRoutePublicationLayout::PackedIndexedRouteRows;
        params.reduction_role =
            MoECanonicalRouteReductionRole::RootOwner;
        MoECanonicalRouteReduceStage stage(std::move(params));
        return stage.execute(cpu_ctx_.get());
    };

    EXPECT_TRUE(execute_slots({3u, 0u, 1u}))
        << "record order is intentionally arbitrary after a rooted gather";
    EXPECT_FALSE(execute_slots({0u, 0u, 3u}))
        << "a duplicate live slot must be fatal";
    EXPECT_FALSE(execute_slots({0u, 1u}))
        << "a missing live slot must be fatal";
    EXPECT_FALSE(execute_slots({0u, 1u, 2u, 3u}))
        << "an inert zero-weight slot must never have a published row";
}

/**
 * @brief Prove the CPU canonical fold is M-total and never requests a small team.
 *
 * The original root-only reduction used `num_threads(M)`. At grouped M=4 this
 * made libgomp retire 24 workers on a 28-core socket, and the next NativeVNNI
 * projection recreated them for every model layer. The typed planner permits
 * only caller-thread execution or the complete configured/existing team.
 */
TEST(
    CPUCanonicalRouteReducerTest,
    PositiveGeometryUsesCallerOrStableFullTeamAndMatchesOrderedRows)
{
    llaminar::v2::ThreadCountGuard thread_count(/*num_threads=*/8);
    constexpr int top_k = 8;
    constexpr std::array<int, 4> column_counts = {31, 32, 33, 257};

    for (int rows = 1; rows <= 31; ++rows)
    {
        for (int columns : column_counts)
        {
            SCOPED_TRACE(
                "rows=" + std::to_string(rows) +
                " columns=" + std::to_string(columns));
            const auto plan = cpu::moe::planCanonicalRouteFold(
                rows,
                top_k,
                columns);
            EXPECT_TRUE(
                plan.executor ==
                    cpu::moe::CanonicalRouteFoldExecutor::CallerThread ||
                plan.executor ==
                    cpu::moe::CanonicalRouteFoldExecutor::FullOpenMPTeam);
            EXPECT_EQ(plan.team_threads, 8);
            EXPECT_GT(plan.column_tile, 0);
            EXPECT_GT(plan.column_tiles_per_row, 0);
            EXPECT_EQ(
                plan.parallel_tasks,
                static_cast<size_t>(rows) *
                    static_cast<size_t>(plan.column_tiles_per_row));

            const size_t route_slots =
                static_cast<size_t>(rows) * top_k;
            std::vector<float> route_rows(
                route_slots * static_cast<size_t>(columns));
            std::vector<float> route_weights(route_slots);
            for (size_t slot = 0; slot < route_slots; ++slot)
            {
                route_weights[slot] =
                    slot % 11u == 0u
                        ? 0.0f
                        : static_cast<float>((slot % 9u) + 1u) / 19.0f;
                for (int column = 0; column < columns; ++column)
                {
                    route_rows[slot * static_cast<size_t>(columns) +
                               static_cast<size_t>(column)] =
                        static_cast<float>(
                            static_cast<int>((slot * 37u +
                                              static_cast<size_t>(column) * 13u) %
                                             211u) -
                            105) /
                        127.0f;
                }
            }

            std::vector<float> expected(
                static_cast<size_t>(rows) * columns,
                0.0f);
            for (int row = 0; row < rows; ++row)
            {
                float *const expected_row =
                    expected.data() + static_cast<size_t>(row) * columns;
                for (int route = 0; route < top_k; ++route)
                {
                    const size_t slot =
                        static_cast<size_t>(row) * top_k +
                        static_cast<size_t>(route);
                    if (route_weights[slot] == 0.0f)
                        continue;
                    primitives::vec_axpy(
                        expected_row,
                        route_rows.data() + slot * columns,
                        route_weights[slot],
                        columns);
                }
            }

            std::vector<float> actual(expected.size(), 1.0f);
            cpu::moe::CanonicalRouteFoldPlan used_plan;
            ASSERT_TRUE(cpu::moe::reduceCanonicalRouteRowsOrderedFMA(
                {
                    .route_rows = route_rows.data(),
                    .route_weights = route_weights.data(),
                    .record_for_route_slot = nullptr,
                    .output = actual.data(),
                    .rows = rows,
                    .top_k = top_k,
                    .columns = columns,
                },
                &used_plan));
            EXPECT_EQ(used_plan.executor, plan.executor);
            EXPECT_EQ(
                std::memcmp(
                    actual.data(),
                    expected.data(),
                    expected.size() * sizeof(float)),
                0)
                << "column tiling must preserve every ordered FMA byte";
        }
    }
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_RuntimeMVerifierMatchesSerialDecode_IQ3S_TopK8)
{
    ScopedEnv perf_env("LLAMINAR_PERF_STATS_SUMMARY", "1");
    PerfStatsCollector::reset();

    const int d = 256;
    const int inter = 256;
    const int experts = 16;
    const int topk = 8;

    auto gate_weights = TestTensorFactory::createFP32Random({experts, d}, -0.1f, 0.1f, 741);

    auto gate_exps = createExpertIQ3S(experts, inter, d, 750);
    auto up_exps = createExpertIQ3S(experts, inter, d, 751);
    auto down_exps = createExpertIQ3S(experts, d, inter, 752);

    auto run_moe = [&](TensorBase *run_input,
                       TensorBase *run_indices,
                       TensorBase *run_weights,
                       const std::shared_ptr<MoERoutedPipelineKernelOwner> &kernel_owner,
                       TensorBase *run_output,
                       int run_seq)
    {
        MoEExpertComputeStage::Params params;
        params.device_id = DeviceId::cpu();
        params.input = run_input;
        params.routing_indices = run_indices;
        params.routing_weights = run_weights;
        params.routed_pipeline_kernel_owner = kernel_owner;
        params.gate_exps = gate_exps.get();
        params.up_exps = up_exps.get();
        params.down_exps = down_exps.get();
        params.output = run_output;
        params.seq_len = run_seq;
        params.d_model = d;
        params.num_experts = experts;
        params.top_k = topk;
        params.expert_intermediate = inter;
        params.force_decode_equivalent_verifier_prefill = run_seq > 1;

        if (!MoEExpertComputeStage::extractExpertViews(params))
            return false;
        if (!MoEExpertComputeStage::prepareExpertGemmEngines(params))
            return false;
        MoEExpertComputeStage stage(params);
        return stage.execute(cpu_ctx_.get());
    };

    for (const int seq : kGroupedVerifierRuntimeRows)
    {
        SCOPED_TRACE("seq=" + std::to_string(seq));
        auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(seq), static_cast<size_t>(d)}, -0.5f, 0.5f, 740 + seq);
        auto multi_output = TestTensorFactory::createFP32({static_cast<size_t>(seq), static_cast<size_t>(d)});
        auto serial_output = TestTensorFactory::createFP32({static_cast<size_t>(seq), static_cast<size_t>(d)});
        auto routing = computeRouting(input.get(), gate_weights.get(), seq, d, experts, topk);

        ASSERT_TRUE(run_moe(input.get(),
                            routing.indices.get(),
                            routing.weights.get(),
                            routing.kernel_owner,
                            multi_output.get(),
                            seq));

        for (int row = 0; row < seq; ++row)
        {
            FP32Tensor row_input({1, static_cast<size_t>(d)});
            FP32Tensor row_output({1, static_cast<size_t>(d)});

            std::copy_n(input->data() + static_cast<size_t>(row) * d,
                        d,
                        row_input.mutable_data());
            auto row_routing = computeRouting(
                &row_input, gate_weights.get(), 1, d, experts, topk);

            ASSERT_TRUE(run_moe(
                &row_input,
                row_routing.indices.get(),
                row_routing.weights.get(),
                row_routing.kernel_owner,
                &row_output,
                1));
            std::copy_n(row_output.data(),
                        d,
                        serial_output->mutable_data() + static_cast<size_t>(row) * d);
        }

        expectRowsByteEqual(
            multi_output->data(),
            serial_output->data(),
            seq,
            d,
            "IQ3S top-k8 routed expert verifier");
    }

    const auto records = PerfStatsCollector::snapshot({"mtp", "kernel"});
    EXPECT_TRUE(hasPerfCounterWithRoute(
        records,
        "mtp",
        "moe_routed_grouped_decode_equivalent_verifier_prefill_rows",
        "cpu_expert_slot_grouped"))
        << "CPU top-k8 routed expert verifier must use grouped expert-slot execution.\n"
        << PerfStatsCollector::summaryString({"mtp", "kernel"});
    PerfStatsCollector::reset();
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_RuntimeMVerifierMatchesSerialDecode_QwenSizedQ4KQ5K_TopK8)
{
    ScopedEnv perf_env("LLAMINAR_PERF_STATS_SUMMARY", "1");
    PerfStatsCollector::reset();

    const int d = 2048;
    const int inter = 512;
    const int experts = 32;
    const int topk = 8;

    auto gate_weights = TestTensorFactory::createFP32Random(
        {experts, d}, -0.1f, 0.1f, 769);
    auto gate_exps = createExpertQ4K(experts, inter, d, 770);
    auto up_exps = createExpertQ4K(experts, inter, d, 771);
    auto down_exps = createExpertQ5K(experts, d, inter, 772);

    auto run_moe = [&](TensorBase *run_input,
                       TensorBase *run_indices,
                       TensorBase *run_weights,
                       const std::shared_ptr<MoERoutedPipelineKernelOwner> &kernel_owner,
                       TensorBase *run_output,
                       int run_seq)
    {
        MoEExpertComputeStage::Params params;
        params.device_id = DeviceId::cpu();
        params.input = run_input;
        params.routing_indices = run_indices;
        params.routing_weights = run_weights;
        params.routed_pipeline_kernel_owner = kernel_owner;
        params.gate_exps = gate_exps.get();
        params.up_exps = up_exps.get();
        params.down_exps = down_exps.get();
        params.output = run_output;
        params.seq_len = run_seq;
        params.d_model = d;
        params.num_experts = experts;
        params.top_k = topk;
        params.expert_intermediate = inter;
        params.force_decode_equivalent_verifier_prefill = run_seq > 1;

        if (!MoEExpertComputeStage::extractExpertViews(params))
            return false;
        if (!MoEExpertComputeStage::prepareExpertGemmEngines(params))
            return false;
        MoEExpertComputeStage stage(params);
        return stage.execute(cpu_ctx_.get());
    };

    for (const int seq : kGroupedVerifierBoundaryRows)
    {
        SCOPED_TRACE("seq=" + std::to_string(seq));
        auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(seq), static_cast<size_t>(d)}, -0.5f, 0.5f, 760 + seq);
        auto multi_output = TestTensorFactory::createFP32({static_cast<size_t>(seq), static_cast<size_t>(d)});
        auto serial_output = TestTensorFactory::createFP32({static_cast<size_t>(seq), static_cast<size_t>(d)});
        auto routing = computeRouting(
            input.get(), gate_weights.get(), seq, d, experts, topk);

        ASSERT_TRUE(run_moe(&*input,
                            routing.indices.get(),
                            routing.weights.get(),
                            routing.kernel_owner,
                            &*multi_output,
                            seq));

        for (int row = 0; row < seq; ++row)
        {
            FP32Tensor row_input({1, static_cast<size_t>(d)});
            FP32Tensor row_output({1, static_cast<size_t>(d)});

            std::copy_n(input->data() + static_cast<size_t>(row) * d,
                        d,
                        row_input.mutable_data());
            auto row_routing = computeRouting(
                &row_input, gate_weights.get(), 1, d, experts, topk);

            ASSERT_TRUE(run_moe(
                &row_input,
                row_routing.indices.get(),
                row_routing.weights.get(),
                row_routing.kernel_owner,
                &row_output,
                1));
            std::copy_n(row_output.data(),
                        d,
                        serial_output->mutable_data() + static_cast<size_t>(row) * d);
        }

        expectRowsByteEqual(
            multi_output->data(),
            serial_output->data(),
            seq,
            d,
            "Qwen-sized Q4K/Q5K top-k8 routed expert verifier");
    }

    const auto records = PerfStatsCollector::snapshot({"mtp", "kernel"});
    EXPECT_TRUE(hasPerfCounterWithRoute(
        records,
        "mtp",
        "moe_routed_grouped_decode_equivalent_verifier_prefill_rows",
        "cpu_expert_slot_grouped"))
        << "CPU Qwen-sized routed expert verifier must use grouped expert-slot execution.\n"
        << PerfStatsCollector::summaryString({"mtp", "kernel"});
    PerfStatsCollector::reset();
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_NullWeightsReturnsError)
{
    auto input = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});
    auto output = TestTensorFactory::createFP32({SEQ_LEN, D_MODEL});
    auto routing_idx = TestTensorFactory::createFP32({SEQ_LEN * TOP_K, 1});
    auto routing_wt = TestTensorFactory::createFP32({SEQ_LEN * TOP_K, 1});

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.routing_indices = routing_idx.get();
    params.routing_weights = routing_wt.get();
    params.gate_exps = nullptr; // null expert weights
    params.up_exps = nullptr;
    params.down_exps = nullptr;
    params.output = output.get();
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.expert_intermediate = INTERMEDIATE;

    MoEExpertComputeStage stage(params);
    EXPECT_FALSE(stage.execute(cpu_ctx_.get()));
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_CPUActiveExpertWithoutPreparedEngineReturnsError)
{
    const int d = 256;
    const int inter = 256;
    const int seq = 2;
    const int experts = 4;
    const int topk = 1;

    auto input = TestTensorFactory::createFP32Random({seq, d}, -0.5f, 0.5f, 260);
    auto output = TestTensorFactory::createFP32({seq, d});
    auto gate_exps = createExpertQ4K(experts, inter, d, 261);
    auto up_exps = createExpertQ4K(experts, inter, d, 262);
    auto down_exps = createExpertQ4K(experts, d, inter, 263);

    auto routing_indices = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(seq * topk), 1});
    auto routing_weights = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(seq * topk), 1});
    for (int i = 0; i < seq * topk; ++i)
    {
        routing_indices->mutable_data()[i] = 0.0f;
        routing_weights->mutable_data()[i] = 1.0f;
    }

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.routing_indices = routing_indices.get();
    params.routing_weights = routing_weights.get();
    params.gate_exps = gate_exps.get();
    params.up_exps = up_exps.get();
    params.down_exps = down_exps.get();
    params.output = output.get();
    params.seq_len = seq;
    params.d_model = d;
    params.num_experts = experts;
    params.top_k = topk;
    params.expert_intermediate = inter;

    ASSERT_TRUE(MoEExpertComputeStage::extractExpertViews(params));

    MoEExpertComputeStage stage(params);
    EXPECT_FALSE(stage.execute(cpu_ctx_.get()));
}

TEST_F(MoEExpertComputeStageTest, MTPMoESidecarReusesPreparedExpertSlabsAfterRawRelease)
{
    const int d = 256;
    const int inter = 256;
    const int experts = 4;
    const int local_start = 2;
    const int local_count = 2;
    const int sidecar_layer_idx = 28;

    auto gate_exps = createExpertQ4K(experts, inter, d, 281);
    auto up_exps = createExpertQ4K(experts, inter, d, 282);
    auto down_exps = createExpertQ4K(experts, d, inter, 283);

    auto make_params = [&]()
    {
        MoEExpertComputeStage::Params params;
        params.device_id = DeviceId::cpu();
        params.gate_exps = gate_exps.get();
        params.up_exps = up_exps.get();
        params.down_exps = down_exps.get();
        params.num_experts = experts;
        params.top_k = 1;
        params.d_model = d;
        params.expert_intermediate = inter;
        params.layer_idx = sidecar_layer_idx;
        params.local_expert_start = local_start;
        params.local_expert_count = local_count;
        params.expert_mask.assign(experts, false);
        for (int expert = local_start; expert < local_start + local_count; ++expert)
            params.expert_mask[static_cast<size_t>(expert)] = true;
        return params;
    };

    PreparedWeightStore store(ModelContextId{1234});
    auto initial = make_params();
    initial.prepared_store = &store;
    ASSERT_TRUE(MoEExpertComputeStage::extractExpertViews(initial));
    ASSERT_TRUE(MoEExpertComputeStage::prepareExpertGemmEngines(initial));
    ASSERT_TRUE(initial.gate_slab_ref.has_value());
    EXPECT_EQ(store.totalPopulatedExperts(), static_cast<size_t>(local_count * 3));

    gate_exps->release_raw_data();
    up_exps->release_raw_data();
    down_exps->release_raw_data();
    ASSERT_TRUE(gate_exps->is_raw_data_released());
    ASSERT_TRUE(up_exps->is_raw_data_released());
    ASSERT_TRUE(down_exps->is_raw_data_released());

    auto sidecar = make_params();
    sidecar.prepared_store = &store;
    ASSERT_TRUE(MoEExpertComputeStage::extractExpertViews(sidecar));
    ASSERT_TRUE(MoEExpertComputeStage::prepareExpertGemmEngines(sidecar))
        << "MTP sidecar graph build must reuse PreparedWeightStore slabs after host raw data release";

    EXPECT_TRUE(sidecar.gate_slab_ref.has_value());
    EXPECT_TRUE(sidecar.up_slab_ref.has_value());
    EXPECT_TRUE(sidecar.down_slab_ref.has_value());
    EXPECT_EQ(sidecar.prepared_gate_gemm[0], nullptr);
    EXPECT_EQ(sidecar.prepared_up_gemm[0], nullptr);
    EXPECT_EQ(sidecar.prepared_down_gemm[0], nullptr);
    for (int expert = local_start; expert < local_start + local_count; ++expert)
    {
        EXPECT_NE(sidecar.prepared_gate_gemm[static_cast<size_t>(expert)], nullptr);
        EXPECT_NE(sidecar.prepared_up_gemm[static_cast<size_t>(expert)], nullptr);
        EXPECT_NE(sidecar.prepared_down_gemm[static_cast<size_t>(expert)], nullptr);
    }
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_DifferentTokensGetDifferentOutputs)
{
    const int d = 256;
    const int inter = 256;
    const int seq = 2;
    const int experts = 4;
    const int topk = 2;

    // Create two very different tokens
    auto input = TestTensorFactory::createFP32({seq, d});
    float *inp = input->mutable_data();
    for (int i = 0; i < d; ++i)
    {
        inp[i] = 1.0f;      // Token 0: all 1s
        inp[d + i] = -1.0f; // Token 1: all -1s
    }

    auto gate_weights = TestTensorFactory::createFP32Random({experts, d}, -0.1f, 0.1f, 501);
    auto output = TestTensorFactory::createFP32({seq, d});
    std::memset(output->mutable_data(), 0, output->numel() * sizeof(float));

    auto gate_exps = createExpertQ4K(experts, inter, d, 510);
    auto up_exps = createExpertQ4K(experts, inter, d, 511);
    auto down_exps = createExpertQ4K(experts, d, inter, 512);

    // Compute routing externally
    auto routing = computeRouting(input.get(), gate_weights.get(), seq, d, experts, topk);

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.routing_indices = routing.indices.get();
    params.routing_weights = routing.weights.get();
    params.routed_pipeline_kernel_owner = routing.kernel_owner;
    params.gate_exps = gate_exps.get();
    params.up_exps = up_exps.get();
    params.down_exps = down_exps.get();
    params.output = output.get();
    params.seq_len = seq;
    params.d_model = d;
    params.num_experts = experts;
    params.top_k = topk;
    params.expert_intermediate = inter;

    ASSERT_TRUE(MoEExpertComputeStage::extractExpertViews(params));
    ASSERT_TRUE(MoEExpertComputeStage::prepareExpertGemmEngines(params));

    MoEExpertComputeStage stage(params);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    // Token 0 and Token 1 should produce different outputs
    const float *out = output->data();
    float diff = 0.0f;
    for (int i = 0; i < d; ++i)
        diff += std::abs(out[i] - out[d + i]);

    EXPECT_GT(diff, 0.0f) << "Different input tokens produced identical outputs";
}

TEST_F(MoEExpertComputeStageTest, MoEFFN_NormTopKProbSumsToOne)
{
    // Verify that with norm_topk_prob=true, expert weights sum to ~1
    // We test this indirectly by checking the output has reasonable magnitude
    const int d = 256;
    const int inter = 256;
    const int seq = 1;
    const int experts = 8;
    const int topk = 2;

    auto input = TestTensorFactory::createFP32Random({seq, d}, -0.5f, 0.5f, 600);
    auto gate_weights = TestTensorFactory::createFP32Random({experts, d}, -0.1f, 0.1f, 601);
    auto output_norm = TestTensorFactory::createFP32({seq, d});
    auto output_no_norm = TestTensorFactory::createFP32({seq, d});
    std::memset(output_norm->mutable_data(), 0, output_norm->numel() * sizeof(float));
    std::memset(output_no_norm->mutable_data(), 0, output_no_norm->numel() * sizeof(float));

    auto gate_exps = createExpertQ4K(experts, inter, d, 610);
    auto up_exps = createExpertQ4K(experts, inter, d, 611);
    auto down_exps = createExpertQ4K(experts, d, inter, 612);

    // Run with norm_topk_prob=true
    {
        auto routing = computeRouting(input.get(), gate_weights.get(), seq, d, experts, topk, true);

        MoEExpertComputeStage::Params params;
        params.device_id = DeviceId::cpu();
        params.input = input.get();
        params.routing_indices = routing.indices.get();
        params.routing_weights = routing.weights.get();
        params.routed_pipeline_kernel_owner = routing.kernel_owner;
        params.gate_exps = gate_exps.get();
        params.up_exps = up_exps.get();
        params.down_exps = down_exps.get();
        params.output = output_norm.get();
        params.seq_len = seq;
        params.d_model = d;
        params.num_experts = experts;
        params.top_k = topk;
        params.expert_intermediate = inter;

        ASSERT_TRUE(MoEExpertComputeStage::extractExpertViews(params));
        ASSERT_TRUE(MoEExpertComputeStage::prepareExpertGemmEngines(params));

        MoEExpertComputeStage stage(params);
        ASSERT_TRUE(stage.execute(cpu_ctx_.get()));
    }

    // Run with norm_topk_prob=false
    {
        auto routing = computeRouting(input.get(), gate_weights.get(), seq, d, experts, topk, false);

        MoEExpertComputeStage::Params params;
        params.device_id = DeviceId::cpu();
        params.input = input.get();
        params.routing_indices = routing.indices.get();
        params.routing_weights = routing.weights.get();
        params.routed_pipeline_kernel_owner = routing.kernel_owner;
        params.gate_exps = gate_exps.get();
        params.up_exps = up_exps.get();
        params.down_exps = down_exps.get();
        params.output = output_no_norm.get();
        params.seq_len = seq;
        params.d_model = d;
        params.num_experts = experts;
        params.top_k = topk;
        params.expert_intermediate = inter;

        ASSERT_TRUE(MoEExpertComputeStage::extractExpertViews(params));
        ASSERT_TRUE(MoEExpertComputeStage::prepareExpertGemmEngines(params));

        MoEExpertComputeStage stage(params);
        ASSERT_TRUE(stage.execute(cpu_ctx_.get()));
    }

    // Both outputs should be non-zero and possibly different
    const float *norm_out = output_norm->data();
    const float *no_norm_out = output_no_norm->data();

    bool norm_nonzero = false;
    bool no_norm_nonzero = false;
    for (int i = 0; i < d; ++i)
    {
        if (norm_out[i] != 0.0f)
            norm_nonzero = true;
        if (no_norm_out[i] != 0.0f)
            no_norm_nonzero = true;
    }
    EXPECT_TRUE(norm_nonzero);
    EXPECT_TRUE(no_norm_nonzero);
}

// =========================================================================
// Stage Metadata Tests
// =========================================================================

TEST_F(MoEExpertComputeStageTest, MoEFFN_TypeAndName)
{
    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cpu();
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.expert_intermediate = INTERMEDIATE;
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;

    MoEExpertComputeStage stage(params);
    EXPECT_EQ(stage.type(), ComputeStageType::MOE_EXPERT_FFN);
    EXPECT_EQ(stage.name(), "moe_ffn");
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
    EXPECT_FALSE(stage.supportsBackend(ComputeBackendType::GPU_CUDA));
    EXPECT_GT(stage.estimatedFlops(), 0u);
}

/**
 * @brief Only a materialized single-participant route bank is a final output.
 *
 * The mapped heterogeneous continuation receives remote rows directly into
 * its canonical bank before the ordered reducer. Snapshot capture must observe
 * that completed bank at the reducer boundary. A deferred dense merge has not
 * completed those rows yet and therefore must not advertise the same output.
 */
TEST_F(
    MoEExpertComputeStageTest,
    CanonicalReducerPublishesOnlyCompleteMappedRouteBank)
{
    FP32Tensor route_contributions({2, 4});
    FP32Tensor routed_output({1, 4});

    MoECanonicalRouteReduceStage::Params params;
    params.device_id = DeviceId::cuda(0);
    params.canonical_route_contributions = &route_contributions;
    params.output = &routed_output;
    params.seq_len = 1;
    params.top_k = 2;
    params.d_model = 4;
    params.canonical_route_arithmetic =
        MoECanonicalRouteArithmeticPolicy::
            PreweightedContributionThenOrderedAdd;
    params.canonical_route_layout =
        MoECanonicalRoutePublicationLayout::DenseOriginalRouteSlots;
    params.reduction_role = MoECanonicalRouteReductionRole::RootOwner;
    params.external_route_source =
        MoEExternalCanonicalRouteSource::RootCanonicalRouteBank;

    const auto has_named_output = [](
                                      const StageDumpInfo &dump,
                                      const std::string_view name)
    {
        return std::any_of(
            dump.outputs.begin(),
            dump.outputs.end(),
            [&](const StageDumpInfo::OutputBuffer &output)
            {
                return output.name && name == output.name;
            });
    };

    MoECanonicalRouteReduceStage complete_stage(params);
    const StageDumpInfo complete_dump =
        complete_stage.getDumpInfoSnapshot();
    EXPECT_TRUE(has_named_output(
        complete_dump, "canonical_route_contributions"));
    EXPECT_TRUE(has_named_output(complete_dump, "output"));

    params.external_route_source =
        MoEExternalCanonicalRouteSource::DeferredDenseMerge;
    MoECanonicalRouteReduceStage deferred_stage(std::move(params));
    const StageDumpInfo deferred_dump =
        deferred_stage.getDumpInfoSnapshot();
    EXPECT_FALSE(has_named_output(
        deferred_dump, "canonical_route_contributions"));
    EXPECT_TRUE(has_named_output(deferred_dump, "output"));
}

/**
 * @brief A graph-native overlay can be GPU-capable without raw tensor views.
 *
 * CPU and GPU tier participants may bind different packed slices under the
 * same canonical GGUF tensor name. The production overlay therefore lowers an
 * exact registry triplet and removes raw parents before constructing the
 * stage. This regression proves backend validation accepts that sole-authority
 * form instead of forcing the graph to inspect an unrelated packed slice.
 */
TEST_F(
    MoEExpertComputeStageTest,
    RegistryOnlyPreparedExpertTripletSupportsGPUWithoutRawViews)
{
    constexpr int kExperts = 4;
    StreamCapturingGemm gate;
    StreamCapturingGemm up;
    StreamCapturingGemm down;

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cuda(0);
    params.num_experts = kExperts;
    params.top_k = 1;
    params.expert_intermediate = 16;
    params.seq_len = 1;
    params.d_model = 16;
    params.expert_mask = {false, true, false, false};
    params.prepared_gate_gemm.assign(kExperts, nullptr);
    params.prepared_up_gemm.assign(kExperts, nullptr);
    params.prepared_down_gemm.assign(kExperts, nullptr);
    params.prepared_gate_gemm[1] = &gate;
    params.prepared_up_gemm[1] = &up;
    params.prepared_down_gemm[1] = &down;
    params.expert_weight_resolution_policy =
        MoEExpertWeightResolutionPolicy::PreparedRegistryOnly;

    MoEExpertComputeStage stage(std::move(params));
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::GPU_CUDA));
#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::GPU_ROCM));
#endif
}

/**
 * @brief Prove routed-expert scratch follows later and family-wide row counts.
 *
 * A live server sequence first admitted 39 rows and then 68 rows.  After the
 * router envelope was repaired, the expert stage exposed the same stale-first-
 * request defect in `moe_staging_indices`.  This device-free regression checks
 * both GPU backends and the full 4096-row family envelope so no individual MoE
 * buffer can make request order part of the allocation contract again.
 */
TEST_F(
    MoEExpertComputeStageTest,
    GPUWorkspaceRowsCoverConcreteLaterRequestAndFamilyEnvelope)
{
    constexpr int kConcreteRows = 39;
    constexpr int kLaterRequestRows = 68;
    constexpr int kFamilyRows = 4096;
    constexpr int kModelWidth = 2048;
    constexpr int kIntermediate = 768;
    constexpr int kExperts = 256;
    constexpr int kTopK = 8;

    const auto expectedStagingBytes = [](int rows)
    {
        return static_cast<size_t>(rows) * sizeof(int);
    };

    const auto verifyBackend = [&](DeviceId device)
    {
        MoEExpertComputeStage::Params params;
        params.device_id = device;
        params.seq_len = kConcreteRows;
        params.d_model = kModelWidth;
        params.expert_intermediate = kIntermediate;
        params.num_experts = kExperts;
        params.top_k = kTopK;

        MoEExpertComputeStage stage(params);

        const WorkspaceRequirements concrete =
            stage.getWorkspaceRequirements(/*m=*/1);
        const WorkspaceDescriptor *concrete_staging =
            concrete.find(MoEWorkspaceBuffers::STAGING_INDICES);
        ASSERT_NE(concrete_staging, nullptr);
        EXPECT_EQ(
            concrete_staging->size_bytes,
            expectedStagingBytes(kConcreteRows));

        const WorkspaceRequirements later =
            stage.getWorkspaceRequirements(kLaterRequestRows);
        const WorkspaceDescriptor *later_staging =
            later.find(MoEWorkspaceBuffers::STAGING_INDICES);
        ASSERT_NE(later_staging, nullptr);
        EXPECT_EQ(
            later_staging->size_bytes,
            expectedStagingBytes(kLaterRequestRows));

        const WorkspaceRequirements family =
            stage.getWorkspaceRequirements(kFamilyRows);
        const WorkspaceDescriptor *family_staging =
            family.find(MoEWorkspaceBuffers::STAGING_INDICES);
        ASSERT_NE(family_staging, nullptr);
        EXPECT_EQ(
            family_staging->size_bytes,
            expectedStagingBytes(kFamilyRows));
        EXPECT_GT(family_staging->size_bytes, later_staging->size_bytes);
    };

    verifyBackend(DeviceId::cuda(0));
    verifyBackend(DeviceId::rocm(0));
}

TEST_F(MoEExpertComputeStageTest, DirectArrivalBindingUsesStageStream)
{
    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cuda(0);
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.expert_intermediate = INTERMEDIATE;
    params.seq_len = 1;
    params.d_model = D_MODEL;
    params.prepared_gate_gemm.assign(NUM_EXPERTS, nullptr);
    params.prepared_up_gemm.assign(NUM_EXPERTS, nullptr);
    params.prepared_down_gemm.assign(NUM_EXPERTS, nullptr);

    auto gate = std::make_shared<StreamCapturingGemm>();
    auto up = std::make_shared<StreamCapturingGemm>();
    auto down = std::make_shared<StreamCapturingGemm>();
    auto untouched = std::make_shared<StreamCapturingGemm>();
    params.prepared_gate_gemm[1] = gate.get();
    params.prepared_up_gemm[1] = up.get();
    params.prepared_down_gemm[1] = down.get();
    params.prepared_gate_gemm[2] = untouched.get();

    MoEExpertComputeStage stage(params);
    void *stage_stream = reinterpret_cast<void *>(0x1234);
    stage.setGPUStream(stage_stream);
    stage.bindPreparedExpertEnginesForTesting({1});

    EXPECT_EQ(gate->observed_stream, stage_stream);
    EXPECT_EQ(up->observed_stream, stage_stream);
    EXPECT_EQ(down->observed_stream, stage_stream);
    EXPECT_EQ(gate->set_stream_calls, 1);
    EXPECT_EQ(up->set_stream_calls, 1);
    EXPECT_EQ(down->set_stream_calls, 1);
    EXPECT_EQ(untouched->observed_stream, nullptr);
    EXPECT_EQ(untouched->set_stream_calls, 0);
}

TEST_F(MoEExpertComputeStageTest, PendingGpuDirectArrivalRequiresExplicitComputeStream)
{
    FP32Tensor input({1, static_cast<size_t>(D_MODEL)});
    FP32Tensor output({1, static_cast<size_t>(D_MODEL)});

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cuda(0);
    params.input = &input;
    params.output = &output;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.expert_intermediate = INTERMEDIATE;
    params.seq_len = 1;
    params.d_model = D_MODEL;
    params.layer_idx = 3;

    MoEExpertComputeStage stage(params);

    GpuDirectTransferCompletion completion;
    completion.device_id = params.device_id;
    completion.device_ordinal = 0;
    completion.ready_event = std::shared_ptr<void>(
        reinterpret_cast<void *>(0x1234),
        [](void *) {});
    stage.addPendingGpuDirectTransferForTesting(std::move(completion));

    EXPECT_THROW((void)stage.execute(cpu_ctx_.get()), std::logic_error)
        << "A pending GPU-direct arrival must not fall back to the default stream";
    EXPECT_EQ(stage.pendingGpuDirectTransferCountForTesting(), 1u)
        << "Failed event consumption must leave the pending arrival visible";
}

TEST_F(MoEExpertComputeStageTest, RebuiltStageAdoptsPendingGpuDirectArrivalFromStore)
{
    PreparedWeightStore store(ModelContextId{17});
    ExpertSlabDescriptor desc;
    desc.layer_idx = 3;
    desc.role = WeightRole::MoEExpertGate;
    desc.device = DeviceId::cuda(0);
    desc.num_experts = NUM_EXPERTS;
    desc.local_expert_start = 0;
    desc.local_expert_count = NUM_EXPERTS;
    desc.rows_per_expert = INTERMEDIATE;
    desc.cols_per_expert = D_MODEL;
    auto gate_ref = store.registerExpertSlab(desc);

    ExpertArrival arrival;
    arrival.expert_id = 1;
    arrival.engine = reinterpret_cast<ITensorGemm *>(0x7000);
    arrival.derivation = WeightDerivationKind::RebalancedExpertReplica;
    GpuDirectTransferCompletion completion;
    completion.device_id = DeviceId::cuda(0);
    completion.device_ordinal = 0;
    completion.ready_event = std::shared_ptr<void>(
        reinterpret_cast<void *>(0x1234),
        [](void *) {});
    arrival.gpu_direct_completion = completion;
    store.registerArrivedExperts(gate_ref, {arrival});

    FP32Tensor input({1, static_cast<size_t>(D_MODEL)});
    FP32Tensor output({1, static_cast<size_t>(D_MODEL)});

    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cuda(0);
    params.input = &input;
    params.output = &output;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.expert_intermediate = INTERMEDIATE;
    params.seq_len = 1;
    params.d_model = D_MODEL;
    params.layer_idx = 3;
    params.prepared_store = &store;
    params.gate_slab_ref = gate_ref;

    MoEExpertComputeStage stage(params);
    stage.addPendingGpuDirectTransfersFromStoreForTesting({1});

    ASSERT_EQ(stage.pendingGpuDirectTransferCountForTesting(), 1u);
    EXPECT_THROW((void)stage.execute(cpu_ctx_.get()), std::logic_error)
        << "A rebuilt stage must inherit GPU-direct readiness and require an explicit stream";
}

TEST_F(MoEExpertComputeStageTest, FixedTopologyPrefillUsesOwnerOnlyMaskWhenReplicasActive)
{
    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cuda(0);
    params.seq_len = 8;
    params.d_model = 16;
    params.num_experts = 4;
    params.top_k = 2;
    params.expert_intermediate = 32;
    params.my_socket_id = 0;
    params.layer_idx = 0;

    params.expert_mask = {true, true, true, false};
    params.replica_set.domain_id = "cuda_ep";
    params.replica_set.base_ownership =
        MoELayeredExpertOwnership::uniform(1, 2, {0, 1, 0, 1});
    params.replica_set.replica_participants_by_layer.assign(
        1,
        std::vector<std::vector<bool>>(4, std::vector<bool>(2, false)));
    params.replica_set.setReplicaOnParticipant(0, 1, 0);
    params.replica_set.rebuildAggregateReplicaFlags();
    params.replica_set.buildPrefillMask(
        params.my_socket_id, params.expert_mask, params.layer_idx);

    MoEExpertComputeStage stage(params);
    const auto ids = stage.fixedTopologyPrefillExpertIdsForTesting();
    const auto mask = stage.fixedTopologyPrefillExpertMaskBytesForTesting();

#if defined(HAVE_CUDA)
    EXPECT_TRUE(stage.usesFixedTopologyGroupedPrefillForTesting());
#else
    EXPECT_FALSE(stage.usesFixedTopologyGroupedPrefillForTesting())
        << "A CPU-only unit build must not advertise a compiled CUDA route";
#endif
    EXPECT_EQ(ids, std::vector<int>({0, 2}));
    ASSERT_EQ(mask.size(), 4u);
    EXPECT_EQ(mask[0], 1u);
    EXPECT_EQ(mask[1], 0u) << "replica expert is resident for decode but owner-only for prefill";
    EXPECT_EQ(mask[2], 1u);
    EXPECT_EQ(mask[3], 0u);
}

TEST_F(MoEExpertComputeStageTest, FixedTopologyVerifierReplayStillRejectsReplicas)
{
    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cuda(0);
    params.seq_len = 1;
    params.d_model = 16;
    params.num_experts = 4;
    params.top_k = 2;
    params.expert_intermediate = 32;
    params.my_socket_id = 0;
    params.layer_idx = 0;
    params.force_grouped_verifier_prefill_for_decode = true;

    params.expert_mask = {true, true, true, false};
    params.replica_set.domain_id = "cuda_ep";
    params.replica_set.base_ownership =
        MoELayeredExpertOwnership::uniform(1, 2, {0, 1, 0, 1});
    params.replica_set.replica_participants_by_layer.assign(
        1,
        std::vector<std::vector<bool>>(4, std::vector<bool>(2, false)));
    params.replica_set.setReplicaOnParticipant(0, 1, 0);
    params.replica_set.rebuildAggregateReplicaFlags();
    params.replica_set.buildPrefillMask(
        params.my_socket_id, params.expert_mask, params.layer_idx);

    MoEExpertComputeStage stage(params);
    EXPECT_FALSE(stage.usesFixedTopologyGroupedVerifierReplayForTesting());
}

TEST_F(MoEExpertComputeStageTest, GpuGroupedVerifierRequiresAcceptedStateHistogramCommit)
{
    /*
     * Constructing a stage with a CUDA identity does not launch GPU work. This
     * verifies that the production grouped-verifier flag declares the later
     * accepted-state commit, while ordinary prefill remains outside that
     * transaction.
     */
    MoEExpertComputeStage::Params params;
    params.device_id = DeviceId::cuda(0);
    params.seq_len = 3;
    params.num_experts = 4;
    params.top_k = 2;
    params.force_grouped_verifier_prefill_for_decode = true;
    params.force_decode_equivalent_verifier_prefill = false;
    params.grouped_verifier_histogram_role =
        MoEGroupedVerifierHistogramRole::DeferredAcceptedRows;

    MoEExpertComputeStage stage(params);
    EXPECT_EQ(
        stage.groupedVerifierHistogramRole(),
        MoEGroupedVerifierHistogramRole::DeferredAcceptedRows)
        << "The real CUDA/ROCm grouped-verifier graph role must require "
           "accepted-state routing-history publication.";

    params.grouped_verifier_histogram_role =
        MoEGroupedVerifierHistogramRole::StaticNoPublication;
    MoEExpertComputeStage static_verifier(params);
    EXPECT_EQ(
        static_verifier.groupedVerifierHistogramRole(),
        MoEGroupedVerifierHistogramRole::StaticNoPublication)
        << "Static grouped verification must name its history boundary without "
           "advertising a deferred publication transaction.";

    params.grouped_verifier_histogram_role =
        MoEGroupedVerifierHistogramRole::NotOwner;
    MoEExpertComputeStage grouped_sidecar(params);
    EXPECT_EQ(
        grouped_sidecar.groupedVerifierHistogramRole(),
        MoEGroupedVerifierHistogramRole::NotOwner)
        << "Grouped sidecars must never publish main-model decode history.";

    params.force_grouped_verifier_prefill_for_decode = false;
    MoEExpertComputeStage ordinary_prefill(params);
    EXPECT_EQ(
        ordinary_prefill.groupedVerifierHistogramRole(),
        MoEGroupedVerifierHistogramRole::NotOwner)
        << "Ordinary prefill is not an accepted decode-demand boundary.";
}

TEST_F(MoEExpertComputeStageTest, SharedExpert_TypeAndName)
{
    SharedExpertFFNStage::Params params;
    params.device_id = DeviceId::cpu();
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.intermediate = INTERMEDIATE;

    SharedExpertFFNStage stage(params);
    EXPECT_EQ(stage.type(), ComputeStageType::MOE_SHARED_EXPERT_FFN);
    EXPECT_EQ(stage.name(), "shared_expert_ffn");
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
    EXPECT_GT(stage.estimatedFlops(), 0u);
}

TEST_F(
    MoEExpertComputeStageTest,
    SharedExpertGpuContractPreparesArenaOwnedGateAndUpScratch)
{
    constexpr int rows = 4;
    constexpr int d_model = 8;
    constexpr int intermediate = 16;

    auto input = TestTensorFactory::createFP32({rows, d_model});
    auto output = TestTensorFactory::createFP32({rows, d_model});
    auto gate_scratch =
        TestTensorFactory::createFP32({rows, intermediate});
    auto up_scratch =
        TestTensorFactory::createFP32({rows, intermediate});

    SharedExpertFFNStage::Params params;
    params.device_id = DeviceId::cuda(0);
    params.input = input.get();
    params.output = output.get();
    params.gate_scratch = gate_scratch.get();
    params.up_scratch = up_scratch.get();
    params.seq_len = rows;
    params.d_model = d_model;
    params.intermediate = intermediate;
    params.input_buffer_id = BufferId::NORMALIZED;
    params.output_buffer_id = BufferId::MOE_SHARED_EXPERT_OUTPUT;
    params.gate_scratch_buffer_id = BufferId::MOE_GATE_SCRATCH;
    params.up_scratch_buffer_id = BufferId::MOE_UP_SCRATCH;

    SharedExpertFFNStage stage(params);
    const StageBufferContract contract = stage.bufferContract();

    ASSERT_EQ(contract.outputs.size(), 3u);
    EXPECT_EQ(contract.outputs[0].id, BufferId::MOE_SHARED_EXPERT_OUTPUT);
    EXPECT_EQ(contract.outputs[1].id, BufferId::MOE_GATE_SCRATCH);
    EXPECT_EQ(contract.outputs[2].id, BufferId::MOE_UP_SCRATCH);
    EXPECT_TRUE(contract.outputs[1].prepare_write_storage);
    EXPECT_TRUE(contract.outputs[2].prepare_write_storage);
}

TEST_F(MoEExpertComputeStageTest, SharedExpert_CudaSmallMDeclaresGateUpSideStreamWorkspace)
{
    constexpr int rows = 4;
    constexpr int d_model = 64;
    constexpr int intermediate = 128;
    const DeviceId device = DeviceId::cuda(0);

    auto input = TestTensorFactory::createFP32({rows, d_model});
    auto gate_w = TestTensorFactory::createFP32({intermediate, d_model});
    auto up_w = TestTensorFactory::createFP32({intermediate, d_model});
    auto down_w = TestTensorFactory::createFP32({d_model, intermediate});
    auto output = TestTensorFactory::createFP32({rows, d_model});

    PreparedWeightStore store(ModelContextId{9101});
    auto gate_ref = registerWorkspaceOnlyGemm(
        store,
        gate_w.get(),
        device,
        "blk.0.ffn_shexp_gate.weight",
        std::make_shared<WorkspaceOnlyGemm>(intermediate, d_model));
    auto up_ref = registerWorkspaceOnlyGemm(
        store,
        up_w.get(),
        device,
        "blk.0.ffn_shexp_up.weight",
        std::make_shared<WorkspaceOnlyGemm>(intermediate, d_model));
    auto down_ref = registerWorkspaceOnlyGemm(
        store,
        down_w.get(),
        device,
        "blk.0.ffn_shexp_down.weight",
        std::make_shared<WorkspaceOnlyGemm>(d_model, intermediate));

    SharedExpertFFNStage::Params params;
    params.device_id = device;
    params.input = input.get();
    params.gate_w = gate_w.get();
    params.up_w = up_w.get();
    params.down_w = down_w.get();
    params.output = output.get();
    params.seq_len = rows;
    params.d_model = d_model;
    params.intermediate = intermediate;
    params.prepared_ref_gate = gate_ref;
    params.prepared_ref_up = up_ref;
    params.prepared_ref_down = down_ref;
    params.prepared_store = &store;
    params.force_grouped_verifier_prefill_for_decode = true;

    SharedExpertFFNStage stage(params);
    const WorkspaceRequirements reqs =
        stage.getWorkspaceRequirements(rows, d_model, intermediate);

    const auto *grouped =
        reqs.find(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS);
    const auto *side_stream =
        reqs.find(GemmWorkspaceBuffers::CUDA_CONCURRENT_DECODE_GEMV_KPAR_PARTIALS);

    ASSERT_NE(grouped, nullptr)
        << "The stage must merge the underlying grouped-verifier KPAR buffer.";
    EXPECT_EQ(side_stream, nullptr)
        << "Grouped verifier projections execute in-order on their graph stream "
           "and must not reserve the M=1 concurrent-decode arena.";
}

TEST_F(MoEExpertComputeStageTest, SharedGate_TypeAndName)
{
    SharedExpertGateStage::Params params;
    params.device_id = DeviceId::cpu();
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;

    SharedExpertGateStage stage(params);
    EXPECT_EQ(stage.type(), ComputeStageType::MOE_SHARED_EXPERT_GATE);
    EXPECT_EQ(stage.name(), "shared_expert_gate");
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
}

// =========================================================================
// isNonGemmWeight Tests (regression: 3D tensors must not go through GEMM prep)
// =========================================================================

TEST_F(MoEExpertComputeStageTest, IsNonGemmWeight_ExpertTensorsExcluded)
{
    // Verify isNonGemmWeight correctly excludes MoE tensors
    WeightShardingConfig config;

    EXPECT_TRUE(config.isNonGemmWeight("blk.0.ffn_gate_exps.weight"));
    EXPECT_TRUE(config.isNonGemmWeight("blk.5.ffn_up_exps.weight"));
    EXPECT_TRUE(config.isNonGemmWeight("blk.10.ffn_down_exps.weight"));
    EXPECT_TRUE(config.isNonGemmWeight("blk.0.ffn_gate_inp.weight"));
    EXPECT_TRUE(config.isNonGemmWeight("blk.0.ffn_gate_inp_shexp.weight"));

    // Regular weights should NOT be excluded
    EXPECT_FALSE(config.isNonGemmWeight("blk.0.ffn_gate.weight"));
    EXPECT_FALSE(config.isNonGemmWeight("blk.0.ffn_up.weight"));
    EXPECT_FALSE(config.isNonGemmWeight("blk.0.ffn_down.weight"));
    EXPECT_FALSE(config.isNonGemmWeight("blk.0.attn_q.weight"));
}
