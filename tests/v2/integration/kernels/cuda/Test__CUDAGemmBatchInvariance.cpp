/**
 * @file Test__CUDAGemmBatchInvariance.cpp
 * @brief Adversarial CUDA GEMM batch-invariance and workspace-ordering tests.
 *
 * Exercises `CUDAQuantisedGemmKernel` projection and fused-projection entry
 * points with identical inputs, production launch policy, explicit streams,
 * persistent workspace, and both serial and concurrent projection schedules.
 * Every retained schedule must reproduce the same output bytes across repeats.
 *
 * These cases localize failures among kernel arithmetic, side-stream event
 * ordering, and workspace partition ownership. They deliberately use the real
 * optimized path; there is no alternate deterministic prefill implementation.
 */

#include <gtest/gtest.h>

#include "tensors/Tensors.h"
#include "tensors/TensorKernels.h"
#include "kernels/KernelFactory.h"
#include "kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#include "backends/ComputeBackend.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/coherence/GpuCoherence.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/compute_stages/ComputeStageUtils.h"
#include "transfer/TransferEngine.h"
#include "../../../utils/TestModelHelper.h"
#include "loaders/ModelLoader.h"
#include "tensors/TensorFactory.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"
#include "utils/MPIContext.h"
#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#include <cuda_runtime.h>
#endif

#include "../../../utils/CUDATestUtils.h"
#include "../../../utils/GpuPreparedGemmHarness.h"
#include "../../../utils/TestTensorFactory.h"
#include "../../../utils/ScopedGPUStream.h"

#include <vector>
#include <array>
#include <cmath>
#include <cstring>
#include <random>
#include <numeric>
#include <filesystem>
#include <iomanip>

using namespace llaminar2;
using namespace llaminar2::test::cuda;
using namespace llaminar2::test;

using KernelDeviceType = llaminar::v2::kernels::DeviceType;
using TensorProjectionDesc = llaminar2::ITensorGemm::TensorProjectionDesc;

#ifdef HAVE_CUDA
extern "C"
{
    void cudaNativeVNNIPrefill_setBK256Mode(int mode);
    int cudaNativeVNNIPrefill_getBK256Mode();
    void cudaNativeVNNIPrefill_setCanonicalKPartitionMode(bool enabled);
    bool cudaNativeVNNIPrefill_getCanonicalKPartitionMode();
    void cudaNativeVNNIPrefill_setExactOverlayEnabled(bool enabled);
    bool cudaNativeVNNIPrefill_getExactOverlayEnabled();
    bool cudaNativeVNNIPrefill_getWorkspacePlan(
        uint8_t codebook_id,
        int M,
        int N,
        int K,
        int cuda_device_id,
        size_t *canonical_kpart_partials_bytes,
        int *planned_k_partitions);
    bool cudaNativeVNNIPrefill_getWorkspaceEnvelope(
        uint8_t codebook_id,
        int max_M,
        int N,
        int K,
        int cuda_device_id,
        size_t *canonical_kpart_partials_bytes,
        int *planned_k_partitions,
        int *planned_rows);
    void cudaNativeVNNIPrefill_setForceTile(int tile_id);
    void cudaNativeVNNIPrefill_getForceTile(int *tile_id);
    void cudaNativeVNNIPrefill_getLastLaunchSelection(
        int *tile_id,
        int *k_partitions,
        int *used_bk256,
        int *used_canonical_kpart);
    int cudaNativeVNNIPrefill_getLastLaunchUsedExactOverlay();
    bool cudaNativeVNNIGemvTuned_queryCanonicalM1Schedule(
        uint8_t codebook_id,
        int n,
        int k,
        int sm_count,
        int *uses_ordered_reducer,
        int *k_partitions);
}
#endif

namespace
{
    constexpr const char *MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
    constexpr int NUM_REPETITIONS = 10;

#ifdef HAVE_CUDA
    class ScopedCudaPrefillModes
    {
    public:
        ScopedCudaPrefillModes()
            : bk256_(cudaNativeVNNIPrefill_getBK256Mode()),
              canonical_kpart_(
                  cudaNativeVNNIPrefill_getCanonicalKPartitionMode()),
              exact_overlay_enabled_(
                  cudaNativeVNNIPrefill_getExactOverlayEnabled())
        {
            cudaNativeVNNIPrefill_getForceTile(&force_tile_);
        }

        ~ScopedCudaPrefillModes()
        {
            cudaNativeVNNIPrefill_setForceTile(force_tile_);
            cudaNativeVNNIPrefill_setBK256Mode(bk256_);
            cudaNativeVNNIPrefill_setCanonicalKPartitionMode(
                canonical_kpart_);
            cudaNativeVNNIPrefill_setExactOverlayEnabled(
                exact_overlay_enabled_);
        }

        ScopedCudaPrefillModes(const ScopedCudaPrefillModes &) = delete;
        ScopedCudaPrefillModes &operator=(const ScopedCudaPrefillModes &) = delete;

    private:
        int force_tile_ = -1;
        int bk256_ = 0;
        bool canonical_kpart_ = false;
        bool exact_overlay_enabled_ = true;
    };

    class ScopedDebugEnvOverride
    {
    public:
        ScopedDebugEnvOverride(const char *name, const char *value)
            : name_(name)
        {
            if (const char *old = std::getenv(name))
            {
                had_old_ = true;
                old_value_ = old;
            }
            setenv(name, value, 1);
            mutableDebugEnv().reload();
        }

        ~ScopedDebugEnvOverride()
        {
            if (had_old_)
                setenv(name_.c_str(), old_value_.c_str(), 1);
            else
                unsetenv(name_.c_str());
            mutableDebugEnv().reload();
        }

        ScopedDebugEnvOverride(const ScopedDebugEnvOverride &) = delete;
        ScopedDebugEnvOverride &operator=(const ScopedDebugEnvOverride &) = delete;

    private:
        std::string name_;
        bool had_old_ = false;
        std::string old_value_;
    };
#endif

    ITensorGemm *getPreparedKernel(const TensorBase *tensor, DeviceId device_id)
    {
        auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor);
        const bool is_gpu_quantized =
            (device_id.is_cuda() || device_id.is_rocm()) &&
            unpackable != nullptr &&
            const_cast<IINT8Unpackable *>(unpackable)->vnniFormatInfo() != nullptr;

        if (is_gpu_quantized)
        {
            static std::vector<GpuPreparedGemm> gpu_prepared;
            const uint64_t prepared_id = static_cast<uint64_t>(gpu_prepared.size());
            gpu_prepared.push_back(makeGpuPreparedGemm(
                const_cast<TensorBase *>(tensor),
                device_id,
                "test.cuda_gemm_nondet.weight." + std::to_string(prepared_id),
                ModelContextId{11100 + prepared_id}));
            return gpu_prepared.back().kernel;
        }

        static std::vector<std::shared_ptr<llaminar::v2::kernels::KernelFactory::PreparedGemmHandle>> handles;
        auto prepared = llaminar::v2::kernels::KernelFactory::prepareGemmHandleLocal(tensor, device_id);
        if (!prepared)
            return nullptr;
        handles.push_back(std::move(prepared));
        return llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(handles.back().get());
    }

    double cosineSimilarity(const float *a, const float *b, size_t count)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            norm_a += static_cast<double>(a[i]) * a[i];
            norm_b += static_cast<double>(b[i]) * b[i];
        }
        double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        return denom < 1e-12 ? 0.0 : dot / denom;
    }

    size_t countDiffs(const float *a, const float *b, size_t count)
    {
        size_t diffs = 0;
        for (size_t i = 0; i < count; ++i)
        {
            if (a[i] != b[i])
                ++diffs;
        }
        return diffs;
    }

    float maxAbsDiff(const float *a, const float *b, size_t count)
    {
        float m = 0.0f;
        for (size_t i = 0; i < count; ++i)
            m = std::max(m, std::abs(a[i] - b[i]));
        return m;
    }

    void expectBitwiseEqual(const std::vector<float> &actual,
                            const std::vector<float> &expected,
                            const char *label)
    {
        ASSERT_EQ(actual.size(), expected.size()) << label;
        EXPECT_EQ(countDiffs(actual.data(), expected.data(), actual.size()), 0u)
            << label << " changed bitwise across repeated production runs";
        EXPECT_FLOAT_EQ(maxAbsDiff(actual.data(), expected.data(), actual.size()), 0.0f)
            << label << " drifted across repeated production runs";
    }

    int32_t orderedFloatBits(float value)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        if (bits & 0x80000000u)
            return static_cast<int32_t>(0x80000000u - bits);
        return static_cast<int32_t>(bits);
    }

    uint32_t ulpDiff(float a, float b)
    {
        const int64_t ia = static_cast<int64_t>(orderedFloatBits(a));
        const int64_t ib = static_cast<int64_t>(orderedFloatBits(b));
        return static_cast<uint32_t>(std::llabs(ia - ib));
    }

    struct DiffSummary
    {
        size_t diff_count = 0;
        float max_abs = 0.0f;
        uint32_t max_ulp = 0;
        std::array<size_t, 8> bins = {};
    };

    DiffSummary summarizeDiffs(const float *a, const float *b, size_t count)
    {
        DiffSummary summary;
        for (size_t i = 0; i < count; ++i)
        {
            const float abs_diff = std::abs(a[i] - b[i]);
            summary.max_abs = std::max(summary.max_abs, abs_diff);

            const uint32_t ulp = ulpDiff(a[i], b[i]);
            summary.max_ulp = std::max(summary.max_ulp, ulp);
            if (ulp != 0)
                ++summary.diff_count;

            if (ulp == 0)
                ++summary.bins[0];
            else if (ulp == 1)
                ++summary.bins[1];
            else if (ulp <= 2)
                ++summary.bins[2];
            else if (ulp <= 4)
                ++summary.bins[3];
            else if (ulp <= 8)
                ++summary.bins[4];
            else if (ulp <= 16)
                ++summary.bins[5];
            else if (ulp <= 64)
                ++summary.bins[6];
            else
                ++summary.bins[7];
        }
        return summary;
    }

    void printDiffSummary(const char *label, const DiffSummary &summary)
    {
        std::cout << label
                  << ": diffs=" << summary.diff_count
                  << " max_abs=" << std::scientific << summary.max_abs
                  << " max_ulp=" << std::dec << summary.max_ulp
                  << " bins=[0:" << summary.bins[0]
                  << " 1:" << summary.bins[1]
                  << " 2:" << summary.bins[2]
                  << " <=4:" << summary.bins[3]
                  << " <=8:" << summary.bins[4]
                  << " <=16:" << summary.bins[5]
                  << " <=64:" << summary.bins[6]
                  << " >64:" << summary.bins[7] << "]\n";
    }

} // anonymous namespace

// ============================================================================
// Test Fixture
// ============================================================================

class Test__CUDAGemmBatchInvariance : public CUDATestBase
{
protected:
    std::mt19937 rng_{42};
    std::uniform_real_distribution<float> dist_{-1.0f, 1.0f};
    std::unique_ptr<DeviceWorkspaceManager> workspace_;
    std::unique_ptr<llaminar2::test::ScopedGPUStream> producer_stream_;

    /**
     * @brief Lazily create the fixture's explicit non-default CUDA stream.
     *
     * The same handle is bound to every kernel in a coherence transaction and
     * passed to publication, so tests cannot accidentally launch on one stream
     * and publish completion on another.
     */
    void *explicitProducerStream()
    {
        if (!producer_stream_)
            producer_stream_ =
                std::make_unique<llaminar2::test::ScopedGPUStream>(gpu_device_);
        return producer_stream_->get();
    }

    bool setupSharedWorkspace(
        const std::vector<ITensorGemm *> &kernels,
        int M,
        const std::vector<int> &Ns,
        int K)
    {
        WorkspaceRequirements shared_reqs;
        for (size_t i = 0; i < kernels.size(); ++i)
        {
            auto *ws = dynamic_cast<IWorkspaceConsumer *>(kernels[i]);
            if (!ws)
                continue;
            auto reqs = ws->getWorkspaceRequirements(M, Ns[i], K);
            for (const auto &buf : reqs.buffers)
            {
                auto it = std::find_if(
                    shared_reqs.buffers.begin(), shared_reqs.buffers.end(),
                    [&](const WorkspaceDescriptor &e)
                    { return e.name == buf.name; });
                if (it == shared_reqs.buffers.end())
                {
                    shared_reqs.buffers.push_back(buf);
                }
                else
                {
                    it->size_bytes = std::max(it->size_bytes, buf.size_bytes);
                    it->alignment = std::max(it->alignment, buf.alignment);
                    it->required = it->required || buf.required;
                }
            }
        }

        workspace_ = std::make_unique<DeviceWorkspaceManager>(gpu_device_, 64 * 1024 * 1024);
        if (!workspace_->allocate(shared_reqs))
            return false;

        for (auto *k : kernels)
        {
            if (k)
                k->setGPUStream(explicitProducerStream());
            auto *ws = dynamic_cast<IWorkspaceConsumer *>(k);
            if (ws)
                ws->bindWorkspace(workspace_.get());
        }
        return true;
    }

    void cleanupSharedWorkspace(const std::vector<ITensorGemm *> &kernels)
    {
        for (auto *k : kernels)
        {
            auto *ws = dynamic_cast<IWorkspaceConsumer *>(k);
            if (ws && ws->hasWorkspace())
                ws->unbindWorkspace();
        }
        workspace_.reset();
    }

};

// ============================================================================
// Test: Fused Gate/Up GEMM self-consistency (N repeated calls)
//
// This directly mirrors the FFN gate_up_proj stage in Qwen2:
//   M=9, K=896, N_gate=N_up=4864
// with shared workspace, calling multiply_fused_tensor repeatedly.
// ============================================================================

TEST_F(Test__CUDAGemmBatchInvariance, FusedGateUp_SelfConsistency)
{
    if (!std::filesystem::exists(MODEL_PATH))
        GTEST_SKIP() << "Model not found: " << MODEL_PATH;

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);
    ASSERT_TRUE(loader.loadModel(MODEL_PATH)) << "Failed to load model";

    // Load layer 0 gate and up weights (same as parity test)
    auto w_gate_base = loader.loadTensor("blk.0.ffn_gate.weight", DeviceId::cpu());
    auto w_up_base = loader.loadTensor("blk.0.ffn_up.weight", DeviceId::cpu());
    ASSERT_NE(w_gate_base, nullptr);
    ASSERT_NE(w_up_base, nullptr);

    auto *w_gate = dynamic_cast<Q4_0Tensor *>(w_gate_base.get());
    auto *w_up = dynamic_cast<Q4_0Tensor *>(w_up_base.get());
    ASSERT_NE(w_gate, nullptr);
    ASSERT_NE(w_up, nullptr);

    const int M = 9;                                         // Qwen2 parity prompt: 9 tokens
    const int N_gate = static_cast<int>(w_gate->shape()[0]); // 4864
    const int N_up = static_cast<int>(w_up->shape()[0]);     // 4864
    const int K = static_cast<int>(w_gate->shape()[1]);      // 896

    std::cout << "FusedGateUp non-determinism test: M=" << M
              << " K=" << K << " N_gate=" << N_gate << " N_up=" << N_up
              << " repetitions=" << NUM_REPETITIONS << "\n";

    // Upload weights to GPU
    ASSERT_TRUE(w_gate->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(w_up->ensureOnDevice(gpu_device_));

    // Create CUDA kernels
    auto *kernel_gate = getPreparedKernel(
        w_gate, gpu_device_);
    auto *kernel_up = getPreparedKernel(
        w_up, gpu_device_);
    ASSERT_NE(kernel_gate, nullptr);
    ASSERT_NE(kernel_up, nullptr);

    // Set up SHARED workspace (mirrors FusedGateUpGEMMStage)
    ASSERT_TRUE(setupSharedWorkspace(
        {kernel_gate, kernel_up},
        M, {N_gate, N_up}, K));

    // Create fixed input (deterministic seed)
    auto input = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    float *in_data = input->mutable_data();
    for (int i = 0; i < M * K; ++i)
        in_data[i] = dist_(rng_);

    // Store reference output from first call
    std::vector<float> ref_gate(M * N_gate);
    std::vector<float> ref_up(M * N_up);

    int gate_diffs_total = 0, up_diffs_total = 0;
    double min_gate_cos = 1.0, min_up_cos = 1.0;
    float max_gate_diff = 0.0f, max_up_diff = 0.0f;

    for (int rep = 0; rep < NUM_REPETITIONS; ++rep)
    {
        // Fresh output tensors each iteration
        auto out_gate = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_gate)});
        auto out_up = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_up)});

        // Build projection descriptors
        std::vector<TensorProjectionDesc> projections;
        projections.emplace_back(kernel_gate, out_gate.get(), N_gate,
                                 nullptr, "gate");
        projections.emplace_back(kernel_up, out_up.get(), N_up,
                                 nullptr, "up");

        // Call fused method with coherence wrapper
        ASSERT_TRUE(with_gpu_coherence(
            gpu_device_,
            {input.get()},
            {out_gate.get(), out_up.get()},
            explicitProducerStream(),
            [&]
            {
                return kernel_gate->multiply_fused_tensor(
                    input.get(), projections, M, K, nullptr);
            }));

        const float *gate_data = out_gate->data();
        const float *up_data = out_up->data();

        if (rep == 0)
        {
            // Store reference
            std::memcpy(ref_gate.data(), gate_data, ref_gate.size() * sizeof(float));
            std::memcpy(ref_up.data(), up_data, ref_up.size() * sizeof(float));
            std::cout << "  Rep 0: reference captured\n";
        }
        else
        {
            // Compare against reference
            size_t g_diffs = countDiffs(gate_data, ref_gate.data(), ref_gate.size());
            size_t u_diffs = countDiffs(up_data, ref_up.data(), ref_up.size());
            double g_cos = cosineSimilarity(gate_data, ref_gate.data(), ref_gate.size());
            double u_cos = cosineSimilarity(up_data, ref_up.data(), ref_up.size());
            float g_max = maxAbsDiff(gate_data, ref_gate.data(), ref_gate.size());
            float u_max = maxAbsDiff(up_data, ref_up.data(), ref_up.size());

            gate_diffs_total += (g_diffs > 0 ? 1 : 0);
            up_diffs_total += (u_diffs > 0 ? 1 : 0);
            min_gate_cos = std::min(min_gate_cos, g_cos);
            min_up_cos = std::min(min_up_cos, u_cos);
            max_gate_diff = std::max(max_gate_diff, g_max);
            max_up_diff = std::max(max_up_diff, u_max);

            std::cout << "  Rep " << rep << ": gate diffs=" << g_diffs
                      << "/" << ref_gate.size()
                      << " cos=" << std::fixed << std::setprecision(6) << g_cos
                      << " max_abs=" << std::scientific << g_max
                      << " | up diffs=" << u_diffs
                      << "/" << ref_up.size()
                      << " cos=" << std::fixed << std::setprecision(6) << u_cos
                      << " max_abs=" << std::scientific << u_max << "\n";
        }
    }

    std::cout << "\n=== SUMMARY ===\n"
              << "  Gate: " << gate_diffs_total << "/" << (NUM_REPETITIONS - 1)
              << " runs had diffs, min_cos=" << std::fixed << std::setprecision(6) << min_gate_cos
              << " max_abs=" << std::scientific << max_gate_diff << "\n"
              << "  Up:   " << up_diffs_total << "/" << (NUM_REPETITIONS - 1)
              << " runs had diffs, min_cos=" << std::fixed << std::setprecision(6) << min_up_cos
              << " max_abs=" << std::scientific << max_up_diff << "\n";

    // The test passes if cosine is very high (>= 0.9999) across all repetitions.
    // If this FAILS, it means the kernel is non-deterministic.
    EXPECT_GE(min_gate_cos, 0.9999)
        << "Gate projection is non-deterministic across " << NUM_REPETITIONS << " calls";
    EXPECT_GE(min_up_cos, 0.9999)
        << "Up projection is non-deterministic across " << NUM_REPETITIONS << " calls";

    cleanupSharedWorkspace({kernel_gate, kernel_up});
}

// ============================================================================
// Test: Fused QKV GEMM self-consistency (N repeated calls)
//
// Mirrors the attention QKV stage: M=9, K=896, N_q=896, N_k=128, N_v=128
// ============================================================================

TEST_F(Test__CUDAGemmBatchInvariance, FusedQKV_SelfConsistency)
{
    if (!std::filesystem::exists(MODEL_PATH))
        GTEST_SKIP() << "Model not found: " << MODEL_PATH;

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);
    ASSERT_TRUE(loader.loadModel(MODEL_PATH)) << "Failed to load model";

    auto w_q_base = loader.loadTensor("blk.0.attn_q.weight", DeviceId::cpu());
    auto w_k_base = loader.loadTensor("blk.0.attn_k.weight", DeviceId::cpu());
    auto w_v_base = loader.loadTensor("blk.0.attn_v.weight", DeviceId::cpu());
    ASSERT_NE(w_q_base, nullptr);
    ASSERT_NE(w_k_base, nullptr);
    ASSERT_NE(w_v_base, nullptr);

    auto *w_q = dynamic_cast<Q4_0Tensor *>(w_q_base.get());
    auto *w_k = dynamic_cast<Q4_0Tensor *>(w_k_base.get());
    auto *w_v = dynamic_cast<Q4_0Tensor *>(w_v_base.get());
    ASSERT_NE(w_q, nullptr);
    ASSERT_NE(w_k, nullptr);
    ASSERT_NE(w_v, nullptr);

    const int M = 9;
    const int N_q = static_cast<int>(w_q->shape()[0]);
    const int N_k = static_cast<int>(w_k->shape()[0]);
    const int N_v = static_cast<int>(w_v->shape()[0]);
    const int K = static_cast<int>(w_q->shape()[1]);

    std::cout << "FusedQKV non-determinism test: M=" << M
              << " K=" << K << " N_q=" << N_q << " N_k=" << N_k << " N_v=" << N_v
              << " repetitions=" << NUM_REPETITIONS << "\n";

    ASSERT_TRUE(w_q->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(w_k->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(w_v->ensureOnDevice(gpu_device_));

    auto *k_q = getPreparedKernel(w_q, gpu_device_);
    auto *k_k = getPreparedKernel(w_k, gpu_device_);
    auto *k_v = getPreparedKernel(w_v, gpu_device_);
    ASSERT_NE(k_q, nullptr);
    ASSERT_NE(k_k, nullptr);
    ASSERT_NE(k_v, nullptr);

    ASSERT_TRUE(setupSharedWorkspace(
        {k_q, k_k, k_v},
        M, {N_q, N_k, N_v}, K));

    auto input = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    for (int i = 0; i < M * K; ++i)
        input->mutable_data()[i] = dist_(rng_);

    std::vector<float> ref_q(M * N_q), ref_k(M * N_k), ref_v(M * N_v);
    double min_q_cos = 1.0, min_k_cos = 1.0, min_v_cos = 1.0;
    size_t max_q_diffs = 0, max_k_diffs = 0, max_v_diffs = 0;
    float max_q_abs = 0.0f, max_k_abs = 0.0f, max_v_abs = 0.0f;

    for (int rep = 0; rep < NUM_REPETITIONS; ++rep)
    {
        auto out_q = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_q)});
        auto out_k = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_k)});
        auto out_v = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N_v)});

        std::vector<TensorProjectionDesc> projections;
        projections.emplace_back(k_q, out_q.get(), N_q, nullptr, "Q");
        projections.emplace_back(k_k, out_k.get(), N_k, nullptr, "K");
        projections.emplace_back(k_v, out_v.get(), N_v, nullptr, "V");

        ASSERT_TRUE(with_gpu_coherence(
            gpu_device_,
            {input.get()},
            {out_q.get(), out_k.get(), out_v.get()},
            explicitProducerStream(),
            [&]
            {
                return k_q->multiply_fused_tensor(
                    input.get(), projections, M, K, nullptr);
            }));

        const float *q_data = out_q->data();
        const float *k_data = out_k->data();
        const float *v_data = out_v->data();

        if (rep == 0)
        {
            std::memcpy(ref_q.data(), q_data, ref_q.size() * sizeof(float));
            std::memcpy(ref_k.data(), k_data, ref_k.size() * sizeof(float));
            std::memcpy(ref_v.data(), v_data, ref_v.size() * sizeof(float));
            std::cout << "  Rep 0: reference captured\n";
        }
        else
        {
            double q_cos = cosineSimilarity(q_data, ref_q.data(), ref_q.size());
            double k_cos = cosineSimilarity(k_data, ref_k.data(), ref_k.size());
            double v_cos = cosineSimilarity(v_data, ref_v.data(), ref_v.size());
            size_t q_diffs = countDiffs(q_data, ref_q.data(), ref_q.size());
            size_t k_diffs = countDiffs(k_data, ref_k.data(), ref_k.size());
            size_t v_diffs = countDiffs(v_data, ref_v.data(), ref_v.size());
            float q_max = maxAbsDiff(q_data, ref_q.data(), ref_q.size());
            float k_max = maxAbsDiff(k_data, ref_k.data(), ref_k.size());
            float v_max = maxAbsDiff(v_data, ref_v.data(), ref_v.size());

            min_q_cos = std::min(min_q_cos, q_cos);
            min_k_cos = std::min(min_k_cos, k_cos);
            min_v_cos = std::min(min_v_cos, v_cos);
            max_q_diffs = std::max(max_q_diffs, q_diffs);
            max_k_diffs = std::max(max_k_diffs, k_diffs);
            max_v_diffs = std::max(max_v_diffs, v_diffs);
            max_q_abs = std::max(max_q_abs, q_max);
            max_k_abs = std::max(max_k_abs, k_max);
            max_v_abs = std::max(max_v_abs, v_max);

            std::cout << "  Rep " << rep
                      << ": Q diffs=" << q_diffs << " cos=" << std::fixed << std::setprecision(6) << q_cos
                      << " max_abs=" << std::scientific << q_max
                      << " | K diffs=" << k_diffs << " cos=" << std::fixed << k_cos
                      << " max_abs=" << std::scientific << k_max
                      << " | V diffs=" << v_diffs << " cos=" << std::fixed << v_cos
                      << " max_abs=" << std::scientific << v_max << "\n";
        }
    }

    std::cout << "\n=== SUMMARY ===\n"
              << "  Q: min_cos=" << std::fixed << std::setprecision(6) << min_q_cos
              << " max_diffs=" << max_q_diffs
              << " max_abs=" << std::scientific << max_q_abs << "\n"
              << "  K: min_cos=" << std::fixed << min_k_cos
              << " max_diffs=" << max_k_diffs
              << " max_abs=" << std::scientific << max_k_abs << "\n"
              << "  V: min_cos=" << std::fixed << min_v_cos
              << " max_diffs=" << max_v_diffs
              << " max_abs=" << std::scientific << max_v_abs << "\n";

    EXPECT_GE(min_q_cos, 0.9999) << "Q projection non-deterministic";
    EXPECT_GE(min_k_cos, 0.9999) << "K projection non-deterministic";
    EXPECT_GE(min_v_cos, 0.9999) << "V projection non-deterministic";
    EXPECT_EQ(max_q_diffs, 0u) << "Q projection changed bitwise across repeated calls";
    EXPECT_EQ(max_k_diffs, 0u) << "K projection changed bitwise across repeated calls";
    EXPECT_EQ(max_v_diffs, 0u) << "V projection changed bitwise across repeated calls";
    EXPECT_FLOAT_EQ(max_q_abs, 0.0f) << "Q projection max_abs drift should be zero";
    EXPECT_FLOAT_EQ(max_k_abs, 0.0f) << "K projection max_abs drift should be zero";
    EXPECT_FLOAT_EQ(max_v_abs, 0.0f) << "V projection max_abs drift should be zero";

    cleanupSharedWorkspace({k_q, k_k, k_v});
}

// ============================================================================
// Test: Single multiply_tensor self-consistency (not fused)
//
// Calls multiply_tensor() separately for a single kernel N times.
// A failure here localizes the defect to the projection kernel; a fused-only
// failure localizes it to stream/event ordering or workspace partitioning.
// ============================================================================

TEST_F(Test__CUDAGemmBatchInvariance, SingleKernel_SelfConsistency)
{
    if (!std::filesystem::exists(MODEL_PATH))
        GTEST_SKIP() << "Model not found: " << MODEL_PATH;

    auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    TensorFactory factory(*mpi_ctx);
    ModelLoader loader(&factory);
    ASSERT_TRUE(loader.loadModel(MODEL_PATH)) << "Failed to load model";

    // Use FFN up weight — same tensor that shows variance
    auto w_up_base = loader.loadTensor("blk.0.ffn_up.weight", DeviceId::cpu());
    ASSERT_NE(w_up_base, nullptr);

    auto *w_up = dynamic_cast<Q4_0Tensor *>(w_up_base.get());
    ASSERT_NE(w_up, nullptr);

    const int M = 9;
    const int N = static_cast<int>(w_up->shape()[0]); // 4864
    const int K = static_cast<int>(w_up->shape()[1]); // 896

    std::cout << "Single kernel non-determinism test: M=" << M
              << " N=" << N << " K=" << K
              << " repetitions=" << NUM_REPETITIONS << "\n";

    ASSERT_TRUE(w_up->ensureOnDevice(gpu_device_));

    auto *kernel = getPreparedKernel(
        w_up, gpu_device_);
    ASSERT_NE(kernel, nullptr);
    kernel->setGPUStream(explicitProducerStream());

    // Set up workspace for single kernel
    auto *ws = dynamic_cast<IWorkspaceConsumer *>(kernel);
    if (ws)
    {
        auto reqs = ws->getWorkspaceRequirements(M, N, K);
        workspace_ = std::make_unique<DeviceWorkspaceManager>(gpu_device_, 64 * 1024 * 1024);
        ASSERT_TRUE(workspace_->allocate(reqs));
        ws->bindWorkspace(workspace_.get());
    }

    auto input = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    for (int i = 0; i < M * K; ++i)
        input->mutable_data()[i] = dist_(rng_);

    std::vector<float> ref(M * N);
    double min_cos = 1.0;
    float worst_max_diff = 0.0f;

    for (int rep = 0; rep < NUM_REPETITIONS; ++rep)
    {
        auto output = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

        ASSERT_TRUE(with_gpu_coherence(
            gpu_device_,
            {input.get()},
            {output.get()},
            explicitProducerStream(),
            [&]
            {
                return kernel->multiply_tensor(
                    input.get(), output.get(), M, N, K, true, 1.0f, 0.0f, nullptr, nullptr, -1);
            }));

        const float *data = output->data();

        if (rep == 0)
        {
            std::memcpy(ref.data(), data, ref.size() * sizeof(float));
            std::cout << "  Rep 0: reference captured\n";
        }
        else
        {
            size_t diffs = countDiffs(data, ref.data(), ref.size());
            double cos = cosineSimilarity(data, ref.data(), ref.size());
            float md = maxAbsDiff(data, ref.data(), ref.size());
            min_cos = std::min(min_cos, cos);
            worst_max_diff = std::max(worst_max_diff, md);

            std::cout << "  Rep " << rep << ": diffs=" << diffs << "/" << ref.size()
                      << " cos=" << std::fixed << std::setprecision(6) << cos
                      << " max_abs=" << std::scientific << md << "\n";
        }
    }

    std::cout << "\n=== SUMMARY ===\n"
              << "  min_cos=" << std::fixed << std::setprecision(6) << min_cos
              << " worst_max_abs=" << std::scientific << worst_max_diff << "\n";

    // Every retained production schedule must repeat bitwise. Any difference is
    // now a contract violation rather than an expected reduction-order effect.
    EXPECT_GE(min_cos, 0.9999)
        << "Single multiply_tensor is non-deterministic across " << NUM_REPETITIONS << " calls";

    if (ws)
    {
        ws->unbindWorkspace();
        workspace_.reset();
    }
}

TEST_F(Test__CUDAGemmBatchInvariance, NativeVNNI_SelfConsistency)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA build required";
#else
    // NativeVNNI is now the sole CUDA GEMM path. Verify it produces
    // bitwise-identical results across repeated invocations.
    ScopedCudaPrefillModes mode_guard;

    cudaNativeVNNIPrefill_setForceTile(-1);
    cudaNativeVNNIPrefill_setBK256Mode(0);

    const int M = 9;
    const int N = 896;
    const int K = 4864;

    auto weight = TestTensorFactory::createQ4_0Random(
        {static_cast<size_t>(N), static_cast<size_t>(K)}, 17);
    auto *w_down = dynamic_cast<Q4_0Tensor *>(weight.get());
    ASSERT_NE(w_down, nullptr);
    ASSERT_TRUE(w_down->ensureOnDevice(gpu_device_));

    auto *kernel = getPreparedKernel(w_down, gpu_device_);
    ASSERT_NE(kernel, nullptr);
    kernel->setGPUStream(explicitProducerStream());

    auto *ws = dynamic_cast<IWorkspaceConsumer *>(kernel);
    ASSERT_NE(ws, nullptr);
    auto reqs = ws->getWorkspaceRequirements(M, N, K);
    workspace_ = std::make_unique<DeviceWorkspaceManager>(gpu_device_, 128 * 1024 * 1024);
    ASSERT_TRUE(workspace_->allocate(reqs));
    ws->bindWorkspace(workspace_.get());

    auto input = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    for (int i = 0; i < M * K; ++i)
        input->mutable_data()[i] = dist_(rng_);

    std::vector<float> ref(M * N);
    size_t max_diffs = 0;
    float max_abs = 0.0f;
    double min_cos = 1.0;

    for (int rep = 0; rep < 4; ++rep)
    {
        auto output = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

        ASSERT_TRUE(with_gpu_coherence(
            gpu_device_,
            {input.get()},
            {output.get()},
            explicitProducerStream(),
            [&]
            {
                return kernel->multiply_tensor(
                    input.get(), output.get(), M, N, K, true, 1.0f, 0.0f, nullptr, nullptr, -1);
            }));

        const float *data = output->data();
        if (rep == 0)
        {
            std::memcpy(ref.data(), data, ref.size() * sizeof(float));
        }
        else
        {
            max_diffs = std::max(max_diffs, countDiffs(data, ref.data(), ref.size()));
            max_abs = std::max(max_abs, maxAbsDiff(data, ref.data(), ref.size()));
            min_cos = std::min(min_cos, cosineSimilarity(data, ref.data(), ref.size()));
        }
    }

    std::cout << "NativeVNNI self-consistency: min_cos=" << std::fixed << std::setprecision(6) << min_cos
              << " max_diffs=" << max_diffs
              << " max_abs=" << std::scientific << max_abs << "\n";

    EXPECT_EQ(max_diffs, 0u) << "NativeVNNI path changed bitwise across repeats";
    EXPECT_FLOAT_EQ(max_abs, 0.0f) << "NativeVNNI path drift should be zero";
    EXPECT_GE(min_cos, 0.999999) << "NativeVNNI path should be deterministic";

    int selected_tile = -1;
    int selected_k_partitions = 0;
    int used_bk256 = 0;
    int used_canonical_kpart = 0;
    cudaNativeVNNIPrefill_getLastLaunchSelection(
        &selected_tile,
        &selected_k_partitions,
        &used_bk256,
        &used_canonical_kpart);
    std::cout << "NativeVNNI final auto selection: tile=" << selected_tile
              << " k_partitions=" << selected_k_partitions
              << " bk256=" << used_bk256
              << " canonical_kpart=" << used_canonical_kpart << "\n";
    ws->unbindWorkspace();
    workspace_.reset();
#endif
}

TEST_F(Test__CUDAGemmBatchInvariance, NativeVNNI_GroupedSmallMProductionScheduleIsStable)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA build required";
#else
    ScopedCudaPrefillModes mode_guard;

    cudaNativeVNNIPrefill_setForceTile(-1);
    cudaNativeVNNIPrefill_setBK256Mode(0);

    const int M = 72;
    const int N = 4864;
    const int K = 896;

    auto weight = TestTensorFactory::createQ4_0Random(
        {static_cast<size_t>(N), static_cast<size_t>(K)}, 23);
    auto *w_down = dynamic_cast<Q4_0Tensor *>(weight.get());
    ASSERT_NE(w_down, nullptr);
    ASSERT_TRUE(w_down->ensureOnDevice(gpu_device_));

    auto *kernel = getPreparedKernel(w_down, gpu_device_);
    ASSERT_NE(kernel, nullptr);
    kernel->setGPUStream(explicitProducerStream());

    auto *ws = dynamic_cast<IWorkspaceConsumer *>(kernel);
    ASSERT_NE(ws, nullptr);
    auto reqs = ws->getWorkspaceRequirements(M, N, K);
    workspace_ = std::make_unique<DeviceWorkspaceManager>(gpu_device_, 128 * 1024 * 1024);
    ASSERT_TRUE(workspace_->allocate(reqs));
    ws->bindWorkspace(workspace_.get());

    auto input = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    for (int i = 0; i < M * K; ++i)
        input->mutable_data()[i] = dist_(rng_);

    std::vector<float> reference(static_cast<size_t>(M) * N);
    for (int repetition = 0; repetition < 2; ++repetition)
    {
        auto output = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
        ASSERT_TRUE(with_gpu_coherence(
            gpu_device_,
            {input.get()},
            {output.get()},
            explicitProducerStream(),
            [&]
            {
                return kernel->multiply_tensor(
                    input.get(), output.get(), M, N, K, true, 1.0f, 0.0f,
                    nullptr, nullptr, -1);
            }));
        const float *data = output->data();
        if (repetition == 0)
            std::memcpy(reference.data(), data, reference.size() * sizeof(float));
        else
            EXPECT_EQ(countDiffs(reference.data(), data, reference.size()), 0u)
                << "Production grouped schedule changed bytes across repeats";
    }

    int selected_tile = -1;
    int selected_k_partitions = 0;
    int used_bk256 = 0;
    int used_canonical_kpart = 0;
    cudaNativeVNNIPrefill_getLastLaunchSelection(
        &selected_tile,
        &selected_k_partitions,
        &used_bk256,
        &used_canonical_kpart);
    std::cout << "NativeVNNI grouped-small-M production selection: tile=" << selected_tile
              << " k_partitions=" << selected_k_partitions
              << " bk256=" << used_bk256
              << " canonical_kpart=" << used_canonical_kpart << "\n";
    EXPECT_EQ(selected_k_partitions, 1);
    EXPECT_EQ(used_canonical_kpart, 0);

    ws->unbindWorkspace();
    workspace_.reset();
#endif
}

/**
 * @test Prove ordinary CUDA NativeVNNI prefill remains total below overlays.
 *
 * Exact cells are additive and must never substitute for a total generic
 * format/geometry policy. This test explicitly bypasses installed cells, then
 * exercises representative Qwen3.6 dense Q4_K and Q5_1 projections, requires
 * a legal physical tile and reduction schedule, and verifies that PerfStats
 * publishes the generic selection. An unseen shape must therefore neither fail
 * dispatch nor silently suppress route telemetry.
 */
TEST_F(Test__CUDAGemmBatchInvariance, NativeVNNI_Qwen36DensePromptPrefillUsesTotalHeuristic)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA build required";
#else
    ScopedCudaPrefillModes mode_guard;
    ScopedEnvVar perf_stats_env("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();
    ASSERT_TRUE(PerfStatsCollector::isEnabled());

    cudaNativeVNNIPrefill_setForceTile(-1);
    cudaNativeVNNIPrefill_setBK256Mode(0);
    cudaNativeVNNIPrefill_setExactOverlayEnabled(false);

    struct Shape
    {
        const char *name;
        int codebook;
        int m;
        int n;
        int k;
    };

    const Shape shapes[] = {
        {"Qwen36_FFN_GateUp_Q4_K", 5, 595, 17408, 5120},
        {"Qwen36_FFN_DownProjection_Q4_K", 5, 600, 5120, 17408},
        {"Qwen36_Attention_KVProjection_Q4_K", 5, 600, 1024, 5120},
        {"Qwen36_GDN_OutputProjection_Q4_K", 5, 595, 5120, 6144},
        {"Qwen36_GDN_InnerProjection_Q5_1", 7, 595, 10240, 5120},
        {"Qwen36_FFN_DownProjection_Q5_1", 7, 600, 5120, 17408},
    };

    ASSERT_EQ(cudaSetDevice(gpu_device_.ordinal), cudaSuccess);
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    for (const auto &shape : shapes)
    {
        SCOPED_TRACE(shape.name);

        std::unique_ptr<TensorBase> weight;
        if (shape.codebook == 5)
        {
            weight = TestTensorFactory::createQ4_KRandom(
                {static_cast<size_t>(shape.n), static_cast<size_t>(shape.k)},
                static_cast<uint32_t>(9000 + shape.n + shape.k));
        }
        else
        {
            ASSERT_EQ(shape.codebook, 7);
            weight = TestTensorFactory::createQ5_1Random(
                {static_cast<size_t>(shape.n), static_cast<size_t>(shape.k)},
                static_cast<uint32_t>(9000 + shape.n + shape.k));
        }
        ASSERT_TRUE(weight->ensureOnDevice(gpu_device_, stream));

        auto *kernel = getPreparedKernel(weight.get(), gpu_device_);
        ASSERT_NE(kernel, nullptr);

        auto *ws = dynamic_cast<IWorkspaceConsumer *>(kernel);
        ASSERT_NE(ws, nullptr);
        auto reqs = ws->getWorkspaceRequirements(shape.m, shape.n, shape.k);
        workspace_ = std::make_unique<DeviceWorkspaceManager>(gpu_device_, 512 * 1024 * 1024);
        ASSERT_TRUE(workspace_->allocate(reqs));
        ws->bindWorkspace(workspace_.get());

        FP32Tensor input({static_cast<size_t>(shape.m), static_cast<size_t>(shape.k)});
        FP32Tensor output({static_cast<size_t>(shape.m), static_cast<size_t>(shape.n)});
        for (int i = 0; i < shape.m * shape.k; ++i)
            input.mutable_data()[i] = dist_(rng_);

        ASSERT_TRUE(input.ensureOnDevice(gpu_device_, stream));
        ASSERT_TRUE(output.ensureOnDevice(gpu_device_, stream));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        kernel->setGPUStream(stream);
        ASSERT_TRUE(kernel->multiply_tensor(
            &input,
            &output,
            shape.m,
            shape.n,
            shape.k,
            true,
            1.0f,
            0.0f,
            nullptr,
            nullptr,
            gpu_device_.ordinal,
            workspace_.get()));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        int selected_tile = -1;
        int selected_k_partitions = 0;
        int used_bk256 = 0;
        int used_canonical_kpart = 0;
        cudaNativeVNNIPrefill_getLastLaunchSelection(
            &selected_tile,
            &selected_k_partitions,
            &used_bk256,
            &used_canonical_kpart);

        EXPECT_GE(selected_tile, 0);
        EXPECT_LE(selected_tile, 5);
        EXPECT_EQ(selected_k_partitions, 1);
        EXPECT_EQ(used_bk256, 0);
        EXPECT_EQ(used_canonical_kpart, 0);

        const auto records = PerfStatsCollector::snapshot({"kernel"});
        bool found = false;
        for (const auto &record : records)
        {
            if (record.name != "cuda_native_vnni_prefill_calls")
                continue;

            auto tag = [&](const char *name) -> std::string
            {
                auto it = record.tags.find(name);
                return it == record.tags.end() ? std::string{} : it->second;
            };

            if (tag("codebook") == std::to_string(shape.codebook) &&
                tag("m") == std::to_string(shape.m) &&
                tag("n") == std::to_string(shape.n) &&
                tag("k") == std::to_string(shape.k))
            {
                found = true;
                EXPECT_EQ(tag("codebook"), std::to_string(shape.codebook));
                EXPECT_EQ(tag("tile_id"), std::to_string(selected_tile));
                EXPECT_EQ(
                    tag("k_partitions"),
                    std::to_string(selected_k_partitions));
                EXPECT_EQ(tag("bk256"), "0");
                EXPECT_EQ(tag("canonical_kpart"), "0");
                EXPECT_EQ(tag("sums_a"), "1")
                    << "Qwen3.6 dense asymmetric NativeVNNI prefill must consume the "
                       "workspace-owned activation block sums instead of recomputing them "
                       "inside every output tile.";
                EXPECT_EQ(record.device, "cuda:" + std::to_string(gpu_device_.ordinal));
                EXPECT_EQ(record.phase, "gemm");
                break;
            }
        }
        EXPECT_TRUE(found) << "NativeVNNI heuristic prefill dispatch must emit structured route counters for "
                           << shape.name << "\n"
                           << PerfStatsCollector::summaryString({"kernel"}, 0);

        ws->unbindWorkspace();
        workspace_.reset();
    }

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    PerfStatsCollector::reset();
#endif
}

TEST_F(Test__CUDAGemmBatchInvariance, NativeVNNICanonicalKpartDeclaresPersistentWorkspaceScratch)
{
#ifdef HAVE_CUDA
    ScopedCudaPrefillModes modes;
    cudaNativeVNNIPrefill_setForceTile(/*T64x128_w2x2=*/1);
    cudaNativeVNNIPrefill_setBK256Mode(-1);
    cudaNativeVNNIPrefill_setCanonicalKPartitionMode(true);

    constexpr int M = 32;
    constexpr int N = 512;
    constexpr int K = 2048;
    constexpr size_t padded_m = 128;

    ASSERT_EQ(cudaSetDevice(gpu_device_.ordinal), cudaSuccess);
    int sm_count = 0;
    ASSERT_EQ(
        cudaDeviceGetAttribute(
            &sm_count,
            cudaDevAttrMultiProcessorCount,
            gpu_device_.cuda_ordinal()),
        cudaSuccess);
    int uses_ordered_reducer = 0;
    int canonical_k_partitions = 0;
    ASSERT_TRUE(cudaNativeVNNIGemvTuned_queryCanonicalM1Schedule(
        /*Q4_0=*/0,
        N,
        K,
        sm_count,
        &uses_ordered_reducer,
        &canonical_k_partitions));
    ASSERT_EQ(uses_ordered_reducer, 1);
    ASSERT_GT(canonical_k_partitions, 1);
    const size_t expected_canonical_bytes =
        static_cast<size_t>(canonical_k_partitions) *
        padded_m * N * sizeof(float);

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    auto weight = TestTensorFactory::createQ4_0Random(
        {static_cast<size_t>(N), static_cast<size_t>(K)}, 424242u);
    ASSERT_TRUE(weight->ensureOnDevice(gpu_device_, stream));

    auto *kernel = getPreparedKernel(weight.get(), gpu_device_);
    ASSERT_NE(kernel, nullptr);

    auto *ws = dynamic_cast<IWorkspaceConsumer *>(kernel);
    ASSERT_NE(ws, nullptr);
    auto reqs = ws->getWorkspaceRequirements(M, N, K);

    bool found_canonical_kpart = false;
    for (const auto &buf : reqs.buffers)
    {
        if (buf.name ==
            GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_CANONICAL_KPART_PARTIALS)
        {
            found_canonical_kpart = true;
            EXPECT_GE(buf.size_bytes, expected_canonical_bytes);
            EXPECT_TRUE(buf.required);
        }
    }
    ASSERT_TRUE(found_canonical_kpart)
        << "Canonical public-M=1 K partitions must declare persistent partials";

    workspace_ = std::make_unique<DeviceWorkspaceManager>(gpu_device_, 128 * 1024 * 1024);
    ASSERT_TRUE(workspace_->allocate(reqs));
    ws->bindWorkspace(workspace_.get());

    FP32Tensor input({static_cast<size_t>(M), static_cast<size_t>(K)});
    FP32Tensor output({static_cast<size_t>(M), static_cast<size_t>(N)});
    for (int i = 0; i < M * K; ++i)
        input.mutable_data()[i] = dist_(rng_);

    ASSERT_TRUE(input.ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(output.ensureOnDevice(gpu_device_, stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    kernel->setGPUStream(stream);
    ASSERT_TRUE(kernel->multiply_tensor(
        &input,
        &output,
        M,
        N,
        K,
        true,
        1.0f,
        0.0f,
        nullptr,
        nullptr,
        gpu_device_.ordinal,
        workspace_.get()));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    int selected_tile = -1;
    int selected_k_partitions = 0;
    int used_bk256 = 0;
    int used_canonical_kpart = 0;
    cudaNativeVNNIPrefill_getLastLaunchSelection(
        &selected_tile,
        &selected_k_partitions,
        &used_bk256,
        &used_canonical_kpart);
    EXPECT_EQ(selected_tile, 1);
    EXPECT_EQ(selected_k_partitions, canonical_k_partitions);
    EXPECT_EQ(used_bk256, 0);
    EXPECT_EQ(used_canonical_kpart, 1);

    ws->unbindWorkspace();
    workspace_.reset();
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
#endif
}

/**
 * @test Prove installed Q6_K production overlays and workspace planning agree.
 *
 * Exact prefill overlays are consulted twice during graph construction: once
 * while the kernel declares persistent workspace and again when capture emits
 * the selected launch. Those decisions must be identical. In particular, a
 * canonical K-partition winner needs its reduction partials before capture,
 * whereas a direct full-K winner must not reserve that large buffer.
 *
 * This test exercises one authenticated canonical winner and the largest
 * high-impact direct winner from the Qwen3.6 35B MoE Q6_K sweep. It uses the
 * ordinary production policy controls throughout; forcing a tile would bypass
 * the installed overlay and would therefore fail to test the real path.
 */
TEST_F(Test__CUDAGemmBatchInvariance, NativeVNNI_Qwen36Q6ExactOverlayMatchesWorkspacePlan)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA build required";
#else
    ScopedCudaPrefillModes mode_guard;
    cudaNativeVNNIPrefill_setForceTile(-1);
    cudaNativeVNNIPrefill_setBK256Mode(0);
    cudaNativeVNNIPrefill_setCanonicalKPartitionMode(false);
    cudaNativeVNNIPrefill_setExactOverlayEnabled(true);

    struct OverlayCase
    {
        const char *name;
        int m;
        int n;
        int k;
        int expected_tile;
        bool expected_canonical_kpart;
    };

    constexpr std::array<OverlayCase, 2> cases = {{
        {
            .name = "expert_gate_up_m64",
            .m = 64,
            .n = 512,
            .k = 2048,
            .expected_tile = 0,
            .expected_canonical_kpart = true,
        },
        {
            .name = "gdn_qkv_m512",
            .m = 512,
            .n = 8192,
            .k = 2048,
            .expected_tile = 3,
            .expected_canonical_kpart = false,
        },
    }};

    ASSERT_EQ(cudaSetDevice(gpu_device_.ordinal), cudaSuccess);
    cudaStream_t stream = nullptr;
    ASSERT_EQ(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        cudaSuccess);

    for (size_t case_index = 0; case_index < cases.size(); ++case_index)
    {
        const OverlayCase &test_case = cases[case_index];
        SCOPED_TRACE(test_case.name);

        auto weight = TestTensorFactory::createQ6_KRandom(
            {static_cast<size_t>(test_case.n),
             static_cast<size_t>(test_case.k)},
            46000u + static_cast<uint32_t>(case_index));
        ASSERT_TRUE(weight->ensureOnDevice(gpu_device_, stream));

        auto *kernel = getPreparedKernel(weight.get(), gpu_device_);
        ASSERT_NE(kernel, nullptr);
        auto *workspace_consumer =
            dynamic_cast<IWorkspaceConsumer *>(kernel);
        ASSERT_NE(workspace_consumer, nullptr);

        const WorkspaceRequirements requirements =
            workspace_consumer->getWorkspaceRequirements(
                test_case.m, test_case.n, test_case.k);
        const auto *canonical_partials = requirements.find(
            GemmWorkspaceBuffers::
                CUDA_NATIVE_VNNI_PREFILL_CANONICAL_KPART_PARTIALS);
        if (test_case.expected_canonical_kpart)
        {
            ASSERT_NE(canonical_partials, nullptr)
                << "The exact canonical overlay must reserve its reducer "
                   "partials before graph capture";
            EXPECT_TRUE(canonical_partials->required);
            EXPECT_GT(canonical_partials->size_bytes, 0u);
        }
        else
        {
            EXPECT_EQ(canonical_partials, nullptr)
                << "A direct exact overlay must not reserve canonical "
                   "K-partition scratch";
        }

        workspace_ = std::make_unique<DeviceWorkspaceManager>(
            gpu_device_, requirements.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(workspace_->allocate(requirements));
        workspace_consumer->bindWorkspace(workspace_.get());

        FP32Tensor input(
            {static_cast<size_t>(test_case.m),
             static_cast<size_t>(test_case.k)});
        FP32Tensor output(
            {static_cast<size_t>(test_case.m),
             static_cast<size_t>(test_case.n)});
        for (int i = 0; i < test_case.m * test_case.k; ++i)
            input.mutable_data()[i] = dist_(rng_);

        ASSERT_TRUE(input.ensureOnDevice(gpu_device_, stream));
        ASSERT_TRUE(output.ensureOnDevice(gpu_device_, stream));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        kernel->setGPUStream(stream);
        ASSERT_TRUE(kernel->multiply_tensor(
            &input,
            &output,
            test_case.m,
            test_case.n,
            test_case.k,
            true,
            1.0f,
            0.0f,
            nullptr,
            nullptr,
            gpu_device_.ordinal,
            workspace_.get()));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        int selected_tile = -1;
        int selected_k_partitions = 0;
        int used_bk256 = 0;
        int used_canonical_kpart = 0;
        cudaNativeVNNIPrefill_getLastLaunchSelection(
            &selected_tile,
            &selected_k_partitions,
            &used_bk256,
            &used_canonical_kpart);

        EXPECT_EQ(cudaNativeVNNIPrefill_getLastLaunchUsedExactOverlay(), 1);
        EXPECT_EQ(selected_tile, test_case.expected_tile);
        EXPECT_EQ(used_bk256, 0);
        EXPECT_EQ(
            used_canonical_kpart,
            test_case.expected_canonical_kpart ? 1 : 0);
        EXPECT_EQ(
            selected_k_partitions > 1,
            test_case.expected_canonical_kpart);

        /*
         * Additive tournaments must be able to remeasure this exact key
         * without using the incumbent overlay as their AUTO baseline. Keep
         * all other production controls unchanged so provenance alone proves
         * that the explicit profiler scope owns selection.
         */
        cudaNativeVNNIPrefill_setExactOverlayEnabled(false);
        ASSERT_TRUE(kernel->multiply_tensor(
            &input,
            &output,
            test_case.m,
            test_case.n,
            test_case.k,
            true,
            1.0f,
            0.0f,
            nullptr,
            nullptr,
            gpu_device_.ordinal,
            workspace_.get()));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(cudaNativeVNNIPrefill_getLastLaunchUsedExactOverlay(), 0)
            << "Profiler bypass must expose the generic AUTO policy";
        cudaNativeVNNIPrefill_setExactOverlayEnabled(true);

        workspace_consumer->unbindWorkspace();
        workspace_.reset();
    }

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
#endif
}

/**
 * @test Prove graph-family workspace covers every installed prefill bucket.
 *
 * The Qwen3.6 35B MoE Q8 gate/up geometry is deliberately non-monotonic:
 * M=64,128,256 use the byte-exact public-M1 K-partition tree, while larger
 * buckets use direct full-K tiles. Planning only M=4096 previously declared
 * zero reducer scratch and made the first M=256 captured replay fail fatally.
 *
 * This regression asks every supported codebook for every production bucket's
 * exact requirement, then proves the family envelope covers their maximum. It
 * additionally checks the real Q8 workspace consumer reserves all four
 * concurrent projection slots and that toggling exact overlays changes a
 * cached request rather than returning stale policy state.
 */
TEST_F(Test__CUDAGemmBatchInvariance,
       NativeVNNIPrefillWorkspaceEnvelopeCoversAllCodebooksAndBuckets)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA build required";
#else
    ScopedCudaPrefillModes mode_guard;
    ScopedDebugEnvOverride concurrent_env(
        "LLAMINAR_CUDA_CONCURRENT_PREFILL", "1");
    cudaNativeVNNIPrefill_setForceTile(-1);
    cudaNativeVNNIPrefill_setBK256Mode(0);
    cudaNativeVNNIPrefill_setCanonicalKPartitionMode(false);
    cudaNativeVNNIPrefill_setExactOverlayEnabled(true);

    constexpr std::array<uint8_t, 16> codebooks = {
        0, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 19};
    constexpr int kMaxM = 4096;
    constexpr int kN = 512;
    constexpr int kK = 2048;

    ASSERT_EQ(cudaSetDevice(gpu_device_.ordinal), cudaSuccess);
    for (const uint8_t codebook : codebooks)
    {
        SCOPED_TRACE(
            "codebook=" + std::to_string(static_cast<int>(codebook)));
        size_t observed_max_bytes = 0;
        for (const int bucket : kDefaultPrefillGraphBucketSizes)
        {
            if (bucket > kMaxM)
                break;
            size_t bucket_bytes = 0;
            int bucket_partitions = 1;
            ASSERT_TRUE(cudaNativeVNNIPrefill_getWorkspacePlan(
                codebook,
                bucket,
                kN,
                kK,
                gpu_device_.ordinal,
                &bucket_bytes,
                &bucket_partitions));
            observed_max_bytes = std::max(observed_max_bytes, bucket_bytes);
        }

        size_t envelope_bytes = 0;
        int envelope_partitions = 1;
        int envelope_rows = 0;
        ASSERT_TRUE(cudaNativeVNNIPrefill_getWorkspaceEnvelope(
            codebook,
            kMaxM,
            kN,
            kK,
            gpu_device_.ordinal,
            &envelope_bytes,
            &envelope_partitions,
            &envelope_rows));
        EXPECT_EQ(envelope_bytes, observed_max_bytes);
        EXPECT_EQ(envelope_bytes == 0, envelope_rows == 0);
        EXPECT_EQ(envelope_bytes == 0, envelope_partitions == 1);
    }

    size_t q8_envelope_bytes = 0;
    int q8_partitions = 1;
    int q8_rows = 0;
    ASSERT_TRUE(cudaNativeVNNIPrefill_getWorkspaceEnvelope(
        /*Q8_0=*/19,
        kMaxM,
        kN,
        kK,
        gpu_device_.ordinal,
        &q8_envelope_bytes,
        &q8_partitions,
        &q8_rows));
    EXPECT_EQ(q8_rows, 256)
        << "The installed Q8 overlay's last canonical bucket is M=256";
    ASSERT_GT(q8_partitions, 1);
    EXPECT_EQ(
        q8_envelope_bytes,
        static_cast<size_t>(q8_partitions) * q8_rows * kN * sizeof(float));

    auto weight = TestTensorFactory::createQ8_0Random(
        {static_cast<size_t>(kN), static_cast<size_t>(kK)},
        /*seed=*/46256);
    auto *kernel = getPreparedKernel(weight.get(), gpu_device_);
    ASSERT_NE(kernel, nullptr);
    auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
    ASSERT_NE(workspace_consumer, nullptr);

    const WorkspaceRequirements exact_requirements =
        workspace_consumer->getWorkspaceRequirements(kMaxM, kN, kK);
    const auto *exact_partials = exact_requirements.find(
        GemmWorkspaceBuffers::
            CUDA_NATIVE_VNNI_PREFILL_CANONICAL_KPART_PARTIALS);
    ASSERT_NE(exact_partials, nullptr);
    EXPECT_EQ(exact_partials->size_bytes, q8_envelope_bytes * 4u)
        << "Concurrent QKV-style replay owns four disjoint reducer slots";

    cudaNativeVNNIPrefill_setExactOverlayEnabled(false);
    const WorkspaceRequirements generic_requirements =
        workspace_consumer->getWorkspaceRequirements(kMaxM, kN, kK);
    EXPECT_EQ(
        generic_requirements.find(
            GemmWorkspaceBuffers::
                CUDA_NATIVE_VNNI_PREFILL_CANONICAL_KPART_PARTIALS),
        nullptr)
        << "The workspace cache key must include exact-overlay policy state";
#endif
}

TEST_F(Test__CUDAGemmBatchInvariance, NativeVNNI_Qwen36QKVConcurrentPrefillUsesCanonicalKpartScratchSlots)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA build required";
#else
    ScopedCudaPrefillModes mode_guard;
    ScopedDebugEnvOverride concurrent_env(
        "LLAMINAR_CUDA_CONCURRENT_PREFILL", "1");
    cudaNativeVNNIPrefill_setForceTile(-1);
    cudaNativeVNNIPrefill_setBK256Mode(0);
    cudaNativeVNNIPrefill_setCanonicalKPartitionMode(true);

    constexpr int M = 600;
    constexpr int N = 1024;
    constexpr int K = 5120;

    ASSERT_EQ(cudaSetDevice(gpu_device_.ordinal), cudaSuccess);
    int sm_count = 0;
    ASSERT_EQ(
        cudaDeviceGetAttribute(
            &sm_count,
            cudaDevAttrMultiProcessorCount,
            gpu_device_.cuda_ordinal()),
        cudaSuccess);
    int uses_ordered_reducer = 0;
    int canonical_k_partitions = 0;
    ASSERT_TRUE(cudaNativeVNNIGemvTuned_queryCanonicalM1Schedule(
        /*Q4_K=*/5,
        N,
        K,
        sm_count,
        &uses_ordered_reducer,
        &canonical_k_partitions));
    ASSERT_EQ(uses_ordered_reducer, 1);
    ASSERT_GT(canonical_k_partitions, 1);

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    auto wq = TestTensorFactory::createQ4_KRandom({N, K}, 8101u);
    auto wk = TestTensorFactory::createQ4_KRandom({N, K}, 8102u);
    auto wv = TestTensorFactory::createQ4_KRandom({N, K}, 8103u);
    ASSERT_TRUE(wq->ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(wk->ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(wv->ensureOnDevice(gpu_device_, stream));

    auto *q_kernel = getPreparedKernel(wq.get(), gpu_device_);
    auto *k_kernel = getPreparedKernel(wk.get(), gpu_device_);
    auto *v_kernel = getPreparedKernel(wv.get(), gpu_device_);
    ASSERT_NE(q_kernel, nullptr);
    ASSERT_NE(k_kernel, nullptr);
    ASSERT_NE(v_kernel, nullptr);

    auto *q_ws = dynamic_cast<IWorkspaceConsumer *>(q_kernel);
    auto *k_ws = dynamic_cast<IWorkspaceConsumer *>(k_kernel);
    auto *v_ws = dynamic_cast<IWorkspaceConsumer *>(v_kernel);
    ASSERT_NE(q_ws, nullptr);
    ASSERT_NE(k_ws, nullptr);
    ASSERT_NE(v_ws, nullptr);

    WorkspaceRequirements reqs;
    reqs.merge(q_ws->getWorkspaceRequirements(M, N, K));
    reqs.merge(k_ws->getWorkspaceRequirements(M, N, K));
    reqs.merge(v_ws->getWorkspaceRequirements(M, N, K));

    const auto *canonical_kpart = reqs.find(
        GemmWorkspaceBuffers::CUDA_NATIVE_VNNI_PREFILL_CANONICAL_KPART_PARTIALS);
    ASSERT_NE(canonical_kpart, nullptr);
    EXPECT_GE(
        canonical_kpart->size_bytes,
        3ULL * static_cast<size_t>(canonical_k_partitions) * 640ULL *
            static_cast<size_t>(N) * sizeof(float))
        << "Fused concurrent Q/K/V prefill needs one canonical K-partition "
           "partial slot per producer stream.";

    workspace_ = std::make_unique<DeviceWorkspaceManager>(
        gpu_device_, reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(workspace_->allocate(reqs));
    q_ws->bindWorkspace(workspace_.get());
    k_ws->bindWorkspace(workspace_.get());
    v_ws->bindWorkspace(workspace_.get());

    FP32Tensor input({M, K});
    FP32Tensor q_output({M, N});
    FP32Tensor k_output({M, N});
    FP32Tensor v_output({M, N});
    for (int i = 0; i < M * K; ++i)
        input.mutable_data()[i] = dist_(rng_);

    ASSERT_TRUE(input.ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(q_output.ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(k_output.ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(v_output.ensureOnDevice(gpu_device_, stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    q_kernel->setGPUStream(stream);
    k_kernel->setGPUStream(stream);
    v_kernel->setGPUStream(stream);

    std::vector<TensorProjectionDesc> projections = {
        {q_kernel, &q_output, N, nullptr, "Q"},
        {k_kernel, &k_output, N, nullptr, "K"},
        {v_kernel, &v_output, N, nullptr, "V"}};

    ASSERT_TRUE(q_kernel->multiply_fused_tensor(
        &input,
        projections,
        M,
        K,
        nullptr,
        workspace_.get()));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    int selected_tile = -1;
    int selected_k_partitions = 0;
    int used_bk256 = 0;
    int used_canonical_kpart = 0;
    cudaNativeVNNIPrefill_getLastLaunchSelection(
        &selected_tile,
        &selected_k_partitions,
        &used_bk256,
        &used_canonical_kpart);
    EXPECT_EQ(selected_k_partitions, canonical_k_partitions);
    EXPECT_EQ(used_bk256, 0);
    EXPECT_EQ(used_canonical_kpart, 1);

    q_ws->unbindWorkspace();
    k_ws->unbindWorkspace();
    v_ws->unbindWorkspace();
    workspace_.reset();
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
#endif
}

TEST_F(Test__CUDAGemmBatchInvariance, NativeVNNI_Qwen36IQ3SQKVConcurrentPrefillLaunches)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA build required";
#else
    ScopedCudaPrefillModes mode_guard;
    cudaNativeVNNIPrefill_setForceTile(-1);
    cudaNativeVNNIPrefill_setBK256Mode(0);

    constexpr int M = 600;
    constexpr int N = 1024;
    constexpr int K = 5120;

    ASSERT_EQ(cudaSetDevice(gpu_device_.ordinal), cudaSuccess);
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    auto wq = TestTensorFactory::createIQ3_SRandom({N, K}, 8111u);
    auto wk = TestTensorFactory::createIQ3_SRandom({N, K}, 8112u);
    auto wv = TestTensorFactory::createIQ3_SRandom({N, K}, 8113u);
    ASSERT_TRUE(wq->ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(wk->ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(wv->ensureOnDevice(gpu_device_, stream));

    auto *q_kernel = getPreparedKernel(wq.get(), gpu_device_);
    auto *k_kernel = getPreparedKernel(wk.get(), gpu_device_);
    auto *v_kernel = getPreparedKernel(wv.get(), gpu_device_);
    ASSERT_NE(q_kernel, nullptr);
    ASSERT_NE(k_kernel, nullptr);
    ASSERT_NE(v_kernel, nullptr);

    auto *q_ws = dynamic_cast<IWorkspaceConsumer *>(q_kernel);
    auto *k_ws = dynamic_cast<IWorkspaceConsumer *>(k_kernel);
    auto *v_ws = dynamic_cast<IWorkspaceConsumer *>(v_kernel);
    ASSERT_NE(q_ws, nullptr);
    ASSERT_NE(k_ws, nullptr);
    ASSERT_NE(v_ws, nullptr);

    WorkspaceRequirements reqs;
    reqs.merge(q_ws->getWorkspaceRequirements(M, N, K));
    reqs.merge(k_ws->getWorkspaceRequirements(M, N, K));
    reqs.merge(v_ws->getWorkspaceRequirements(M, N, K));

    workspace_ = std::make_unique<DeviceWorkspaceManager>(gpu_device_, 256 * 1024 * 1024);
    ASSERT_TRUE(workspace_->allocate(reqs));
    q_ws->bindWorkspace(workspace_.get());
    k_ws->bindWorkspace(workspace_.get());
    v_ws->bindWorkspace(workspace_.get());

    FP32Tensor input({M, K});
    FP32Tensor q_output({M, N});
    FP32Tensor k_output({M, N});
    FP32Tensor v_output({M, N});
    for (int i = 0; i < M * K; ++i)
        input.mutable_data()[i] = dist_(rng_);

    ASSERT_TRUE(input.ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(q_output.ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(k_output.ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(v_output.ensureOnDevice(gpu_device_, stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    q_kernel->setGPUStream(stream);
    k_kernel->setGPUStream(stream);
    v_kernel->setGPUStream(stream);

    std::vector<TensorProjectionDesc> projections = {
        {q_kernel, &q_output, N, nullptr, "Q"},
        {k_kernel, &k_output, N, nullptr, "K"},
        {v_kernel, &v_output, N, nullptr, "V"}};

    ASSERT_TRUE(q_kernel->multiply_fused_tensor(
        &input,
        projections,
        M,
        K,
        nullptr,
        workspace_.get()));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    q_ws->unbindWorkspace();
    k_ws->unbindWorkspace();
    v_ws->unbindWorkspace();
    workspace_.reset();
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
#endif
}

TEST_F(Test__CUDAGemmBatchInvariance, NativeVNNI_Qwen36MoEMixedQKVConcurrentDecodeBindsGemvWorkspace)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA build required";
#else
    ScopedCudaPrefillModes mode_guard;
    cudaNativeVNNIPrefill_setForceTile(-1);
    cudaNativeVNNIPrefill_setBK256Mode(0);

    constexpr int M = 1;
    constexpr int N_Q = 8192;
    constexpr int N_KV = 512;
    constexpr int K = 2048;

    ASSERT_EQ(cudaSetDevice(gpu_device_.ordinal), cudaSuccess);
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    auto wq = TestTensorFactory::createIQ3_SRandom({N_Q, K}, 8121u);
    auto wk = TestTensorFactory::createQ8_0Random({N_KV, K}, 8122u);
    auto wv = TestTensorFactory::createQ8_0Random({N_KV, K}, 8123u);
    ASSERT_TRUE(wq->ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(wk->ensureOnDevice(gpu_device_, stream));
    ASSERT_TRUE(wv->ensureOnDevice(gpu_device_, stream));

    auto *q_kernel = getPreparedKernel(wq.get(), gpu_device_);
    auto *k_kernel = getPreparedKernel(wk.get(), gpu_device_);
    auto *v_kernel = getPreparedKernel(wv.get(), gpu_device_);
    ASSERT_NE(q_kernel, nullptr);
    ASSERT_NE(k_kernel, nullptr);
    ASSERT_NE(v_kernel, nullptr);

    auto *q_ws = dynamic_cast<IWorkspaceConsumer *>(q_kernel);
    auto *k_ws = dynamic_cast<IWorkspaceConsumer *>(k_kernel);
    auto *v_ws = dynamic_cast<IWorkspaceConsumer *>(v_kernel);
    ASSERT_NE(q_ws, nullptr);
    ASSERT_NE(k_ws, nullptr);
    ASSERT_NE(v_ws, nullptr);

    WorkspaceRequirements reqs;
    reqs.merge(q_ws->getWorkspaceRequirements(M, N_Q, K));
    reqs.merge(k_ws->getWorkspaceRequirements(M, N_KV, K));
    reqs.merge(v_ws->getWorkspaceRequirements(M, N_KV, K));
    addCudaConcurrentDecodeGemvSideStreamWorkspace(
        reqs,
        gpu_device_,
        M,
        /*projection_count=*/3);

    const auto *kpar =
        reqs.find(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS);
    ASSERT_NE(kpar, nullptr)
        << "Q8_0 K/V decode may use two-phase KPAR and must have declared partials.";
    const auto *concurrent_kpar =
        reqs.find(GemmWorkspaceBuffers::CUDA_CONCURRENT_DECODE_GEMV_KPAR_PARTIALS);
    ASSERT_NE(concurrent_kpar, nullptr)
        << "Concurrent decode must reserve side-stream GEMV partial slots; otherwise "
           "Q/K/V projections can race through the same KPAR reduction buffer.";
    EXPECT_GE(concurrent_kpar->size_bytes, 2ULL * kpar->size_bytes);

    workspace_ = std::make_unique<DeviceWorkspaceManager>(
        gpu_device_, reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(workspace_->allocate(reqs));
    q_ws->bindWorkspace(workspace_.get());
    k_ws->bindWorkspace(workspace_.get());
    v_ws->bindWorkspace(workspace_.get());

    auto input = std::make_unique<FP32Tensor>(std::vector<size_t>{M, K});
    for (int i = 0; i < M * K; ++i)
        input->mutable_data()[i] = dist_(rng_);

    struct DecodeSnapshot
    {
        std::vector<float> q;
        std::vector<float> k;
        std::vector<float> v;
    };

    auto run_decode = [&](const char *concurrent_decode) -> DecodeSnapshot
    {
        ScopedDebugEnvOverride concurrent_env("LLAMINAR_CUDA_CONCURRENT_DECODE", concurrent_decode);
        auto q_output = std::make_unique<FP32Tensor>(std::vector<size_t>{M, N_Q});
        auto k_output = std::make_unique<FP32Tensor>(std::vector<size_t>{M, N_KV});
        auto v_output = std::make_unique<FP32Tensor>(std::vector<size_t>{M, N_KV});
        std::vector<TensorProjectionDesc> projections = {
            {q_kernel, q_output.get(), N_Q, nullptr, "Q"},
            {k_kernel, k_output.get(), N_KV, nullptr, "K"},
            {v_kernel, v_output.get(), N_KV, nullptr, "V"}};

        EXPECT_TRUE(with_gpu_coherence(
            gpu_device_,
            {input.get()},
            {q_output.get(), k_output.get(), v_output.get()},
            static_cast<void *>(stream),
            [&]
            {
                return q_kernel->multiply_fused_tensor(
                    input.get(), projections, M, K, nullptr, workspace_.get());
            }))
            << "fused decode with LLAMINAR_CUDA_CONCURRENT_DECODE=" << concurrent_decode;

        const float *q = q_output->data();
        const float *k = k_output->data();
        const float *v = v_output->data();
        return {
            std::vector<float>(q, q + N_Q),
            std::vector<float>(k, k + N_KV),
            std::vector<float>(v, v + N_KV),
        };
    };

    q_kernel->setGPUStream(stream);
    k_kernel->setGPUStream(stream);
    v_kernel->setGPUStream(stream);

    const DecodeSnapshot serial = run_decode("0");
    const DecodeSnapshot concurrent = run_decode("1");

    const DiffSummary q_diff = summarizeDiffs(concurrent.q.data(), serial.q.data(), concurrent.q.size());
    const DiffSummary k_diff = summarizeDiffs(concurrent.k.data(), serial.k.data(), concurrent.k.size());
    const DiffSummary v_diff = summarizeDiffs(concurrent.v.data(), serial.v.data(), concurrent.v.size());
    printDiffSummary("concurrent-vs-serial Q", q_diff);
    printDiffSummary("concurrent-vs-serial K", k_diff);
    printDiffSummary("concurrent-vs-serial V", v_diff);

    EXPECT_EQ(q_diff.diff_count, 0u)
        << "Concurrent Q projection must retain serial decode bytes";
    EXPECT_EQ(k_diff.diff_count, 0u)
        << "Concurrent K projection must retain serial decode bytes";
    EXPECT_EQ(v_diff.diff_count, 0u)
        << "Concurrent V projection must retain serial decode bytes";

    q_ws->unbindWorkspace();
    k_ws->unbindWorkspace();
    v_ws->unbindWorkspace();
    workspace_.reset();
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
#endif
}

TEST_F(Test__CUDAGemmBatchInvariance, NativeVNNIConcurrentPrefillRepeatsBitwise)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA build required";
#else
    ScopedCudaPrefillModes mode_guard;
    ScopedDebugEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
    ScopedDebugEnvOverride concurrent_env("LLAMINAR_CUDA_CONCURRENT_PREFILL", "1");
    ScopedDebugEnvOverride force_decode_env("LLAMINAR_CUDA_CONCURRENT_DECODE", "1");
    ASSERT_FALSE(debugEnv().gemm.deterministic);
    EXPECT_TRUE(debugEnv().gemm.cuda_concurrent_prefill);

    cudaNativeVNNIPrefill_setForceTile(/*T64x128_w4x2=*/4);
    cudaNativeVNNIPrefill_setBK256Mode(-1);

    constexpr int M = 72;
    constexpr int N = 1024;
    constexpr int K = 896;

    auto w_gate = TestTensorFactory::createQ4_0Random({N, K}, 9101u);
    auto w_up = TestTensorFactory::createQ4_0Random({N, K}, 9102u);
    ASSERT_TRUE(w_gate->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(w_up->ensureOnDevice(gpu_device_));

    auto *gate_kernel = getPreparedKernel(w_gate.get(), gpu_device_);
    auto *up_kernel = getPreparedKernel(w_up.get(), gpu_device_);
    ASSERT_NE(gate_kernel, nullptr);
    ASSERT_NE(up_kernel, nullptr);

    ASSERT_TRUE(setupSharedWorkspace({gate_kernel, up_kernel}, M, {N, N}, K));

    auto input = std::make_unique<FP32Tensor>(std::vector<size_t>{M, K});
    for (int i = 0; i < M * K; ++i)
        input->mutable_data()[i] = dist_(rng_);

    std::vector<float> ref_gate;
    std::vector<float> ref_up;
    for (int rep = 0; rep < 3; ++rep)
    {
        auto out_gate = std::make_unique<FP32Tensor>(std::vector<size_t>{M, N});
        auto out_up = std::make_unique<FP32Tensor>(std::vector<size_t>{M, N});
        std::vector<TensorProjectionDesc> projections = {
            {gate_kernel, out_gate.get(), N, nullptr, "gate"},
            {up_kernel, out_up.get(), N, nullptr, "up"}};

        ASSERT_TRUE(with_gpu_coherence(
            gpu_device_,
            {input.get()},
            {out_gate.get(), out_up.get()},
            explicitProducerStream(),
            [&]
            {
                return gate_kernel->multiply_fused_tensor(
                    input.get(), projections, M, K, nullptr, workspace_.get());
            }))
            << "concurrent fused prefill repetition " << rep;

        int selected_tile = -1;
        int selected_k_partitions = 0;
        int used_bk256 = 0;
        int used_canonical_kpart = 0;
        cudaNativeVNNIPrefill_getLastLaunchSelection(
            &selected_tile,
            &selected_k_partitions,
            &used_bk256,
            &used_canonical_kpart);
        EXPECT_EQ(selected_k_partitions, 1);
        EXPECT_EQ(used_canonical_kpart, 0);

        const float *gate = out_gate->data();
        const float *up = out_up->data();
        std::vector<float> gate_snapshot(gate, gate + static_cast<size_t>(M) * N);
        std::vector<float> up_snapshot(up, up + static_cast<size_t>(M) * N);
        if (rep == 0)
        {
            ref_gate = std::move(gate_snapshot);
            ref_up = std::move(up_snapshot);
            continue;
        }
        expectBitwiseEqual(gate_snapshot, ref_gate, "concurrent fused prefill gate");
        expectBitwiseEqual(up_snapshot, ref_up, "concurrent fused prefill up");
    }

    cleanupSharedWorkspace({gate_kernel, up_kernel});
#endif
}

/**
 * @test Captured concurrent QKV prefill joins every bias on the root stream.
 *
 * Production prefill records Q/K/V projections on side streams forked from one
 * graph root.  External bias events must be joined to that root before capture;
 * asking the dependency ledger to import them on a side stream is an invalid
 * transaction.  This synthetic real-kernel regression initializes the exact
 * optimized fan-out eagerly, then records and replays it under the production
 * dependency ledger and requires byte-identical output.
 */
TEST_F(Test__CUDAGemmBatchInvariance,
       CapturedConcurrentQKVPrefillPrejoinsBiasesOnRootStream)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA build required";
#else
    ScopedCudaPrefillModes mode_guard;
    ScopedDebugEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
    ScopedDebugEnvOverride concurrent_env(
        "LLAMINAR_CUDA_CONCURRENT_PREFILL", "1");
    ASSERT_FALSE(debugEnv().gemm.deterministic);
    ASSERT_TRUE(debugEnv().gemm.cuda_concurrent_prefill);

    constexpr int M = 32;
    constexpr int N_Q = 256;
    constexpr int N_KV = 128;
    constexpr int K = 256;

    auto wq = TestTensorFactory::createQ4_0Random({N_Q, K}, 9311u);
    auto wk = TestTensorFactory::createQ4_0Random({N_KV, K}, 9312u);
    auto wv = TestTensorFactory::createQ4_0Random({N_KV, K}, 9313u);
    void *const root_stream = explicitProducerStream();
    ASSERT_TRUE(wq->ensureOnDevice(gpu_device_, root_stream));
    ASSERT_TRUE(wk->ensureOnDevice(gpu_device_, root_stream));
    ASSERT_TRUE(wv->ensureOnDevice(gpu_device_, root_stream));

    auto *q_kernel = getPreparedKernel(wq.get(), gpu_device_);
    auto *k_kernel = getPreparedKernel(wk.get(), gpu_device_);
    auto *v_kernel = getPreparedKernel(wv.get(), gpu_device_);
    ASSERT_NE(q_kernel, nullptr);
    ASSERT_NE(k_kernel, nullptr);
    ASSERT_NE(v_kernel, nullptr);
    ASSERT_TRUE(setupSharedWorkspace(
        {q_kernel, k_kernel, v_kernel},
        M,
        {N_Q, N_KV, N_KV},
        K));

    auto input = std::make_unique<FP32Tensor>(
        std::vector<size_t>{M, K});
    auto bias_q = std::make_unique<FP32Tensor>(
        std::vector<size_t>{N_Q});
    auto bias_k = std::make_unique<FP32Tensor>(
        std::vector<size_t>{N_KV});
    auto bias_v = std::make_unique<FP32Tensor>(
        std::vector<size_t>{N_KV});
    for (size_t i = 0; i < input->numel(); ++i)
        input->mutable_data()[i] = dist_(rng_);
    for (size_t i = 0; i < bias_q->numel(); ++i)
        bias_q->mutable_data()[i] = 0.001f * static_cast<float>((i % 17) - 8);
    for (size_t i = 0; i < bias_k->numel(); ++i)
        bias_k->mutable_data()[i] = 0.002f * static_cast<float>((i % 13) - 6);
    for (size_t i = 0; i < bias_v->numel(); ++i)
        bias_v->mutable_data()[i] = 0.003f * static_cast<float>((i % 11) - 5);

    auto output_q = std::make_unique<FP32Tensor>(
        std::vector<size_t>{M, N_Q});
    auto output_k = std::make_unique<FP32Tensor>(
        std::vector<size_t>{M, N_KV});
    auto output_v = std::make_unique<FP32Tensor>(
        std::vector<size_t>{M, N_KV});
    for (TensorBase *tensor : {
             static_cast<TensorBase *>(input.get()),
             static_cast<TensorBase *>(bias_q.get()),
             static_cast<TensorBase *>(bias_k.get()),
             static_cast<TensorBase *>(bias_v.get()),
             static_cast<TensorBase *>(output_q.get()),
             static_cast<TensorBase *>(output_k.get()),
             static_cast<TensorBase *>(output_v.get())})
    {
        ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_, root_stream));
    }

    std::vector<TensorProjectionDesc> projections = {
        {q_kernel, output_q.get(), N_Q, bias_q.get(), "Q"},
        {k_kernel, output_k.get(), N_KV, bias_k.get(), "K"},
        {v_kernel, output_v.get(), N_KV, bias_v.get(), "V"},
    };

    /*
     * The warmup is part of the contract: it creates the persistent side-stream
     * pool and resolves all prepared-weight state before native capture begins.
     */
    ASSERT_TRUE(with_gpu_coherence(
        gpu_device_,
        {input.get(), bias_q.get(), bias_k.get(), bias_v.get()},
        {output_q.get(), output_k.get(), output_v.get()},
        root_stream,
        [&]
        {
            return q_kernel->multiply_fused_tensor(
                input.get(), projections, M, K, nullptr, workspace_.get());
        }));
    auto stream = static_cast<cudaStream_t>(root_stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float> expected_q(static_cast<size_t>(M) * N_Q);
    std::vector<float> expected_k(static_cast<size_t>(M) * N_KV);
    std::vector<float> expected_v(static_cast<size_t>(M) * N_KV);
    ASSERT_EQ(cudaMemcpyAsync(
                  expected_q.data(), output_q->gpu_data_ptr(),
                  expected_q.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(
                  expected_k.data(), output_k->gpu_data_ptr(),
                  expected_k.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(
                  expected_v.data(), output_v->gpu_data_ptr(),
                  expected_v.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    for (TensorBase *external : {
             static_cast<TensorBase *>(input.get()),
             static_cast<TensorBase *>(bias_q.get()),
             static_cast<TensorBase *>(bias_k.get()),
             static_cast<TensorBase *>(bias_v.get())})
    {
        TransferEngine::requireDeviceInput(
            external, gpu_device_, root_stream);
    }

    int stage_identity = 0;
    GraphCaptureDependencyLedger::StagePlan stage_plan{
        .stage_identity = &stage_identity,
        .stage_name = "captured_concurrent_qkv_bias",
        .external_inputs = {
            input.get(), bias_q.get(), bias_k.get(), bias_v.get()},
        .internal_inputs = {},
        .outputs = {output_q.get(), output_k.get(), output_v.get()},
    };
    GraphCaptureDependencyLedger ledger(
        gpu_device_,
        root_stream,
        {std::move(stage_plan)},
        "CUDA concurrent QKV bias regression");

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    ASSERT_EQ(
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
        cudaSuccess);
    bool capture_body_ok = false;
    std::string capture_error;
    try
    {
        GraphCaptureGuard guard(&ledger);
        ScopedGraphCaptureStage stage_scope(&stage_identity);
        capture_body_ok = q_kernel->multiply_fused_tensor(
            input.get(), projections, M, K, nullptr, workspace_.get());
        if (capture_body_ok)
            stage_scope.complete();
    }
    catch (const std::exception &error)
    {
        capture_error = error.what();
    }
    const cudaError_t capture_end = cudaStreamEndCapture(stream, &graph);
    ASSERT_TRUE(capture_error.empty()) << capture_error;
    ASSERT_TRUE(capture_body_ok);
    ASSERT_EQ(capture_end, cudaSuccess);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(
        cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
        cudaSuccess);
    ASSERT_NE(graph_exec, nullptr);
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    std::vector<float> actual_q(expected_q.size());
    std::vector<float> actual_k(expected_k.size());
    std::vector<float> actual_v(expected_v.size());
    ASSERT_EQ(cudaMemcpyAsync(
                  actual_q.data(), output_q->gpu_data_ptr(),
                  actual_q.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(
                  actual_k.data(), output_k->gpu_data_ptr(),
                  actual_k.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(
                  actual_v.data(), output_v->gpu_data_ptr(),
                  actual_v.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    expectBitwiseEqual(actual_q, expected_q, "captured concurrent Q bias");
    expectBitwiseEqual(actual_k, expected_k, "captured concurrent K bias");
    expectBitwiseEqual(actual_v, expected_v, "captured concurrent V bias");

    ASSERT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    cleanupSharedWorkspace({q_kernel, k_kernel, v_kernel});
    llaminar::v2::kernels::KernelFactory::clearCacheFor(wq.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(wk.get());
    llaminar::v2::kernels::KernelFactory::clearCacheFor(wv.get());
#endif
}

TEST_F(Test__CUDAGemmBatchInvariance, NativeVNNIConcurrentDecodeRepeatsBitwise)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA build required";
#else
    ScopedCudaPrefillModes mode_guard;
    ScopedDebugEnvOverride deterministic_env("LLAMINAR_DETERMINISTIC", "0");
    ScopedDebugEnvOverride concurrent_prefill_env("LLAMINAR_CUDA_CONCURRENT_PREFILL", "1");
    ScopedDebugEnvOverride concurrent_decode_env("LLAMINAR_CUDA_CONCURRENT_DECODE", "1");
    ASSERT_FALSE(debugEnv().gemm.deterministic);
    EXPECT_TRUE(debugEnv().gemm.cuda_concurrent_decode);

    cudaNativeVNNIPrefill_setForceTile(-1);
    cudaNativeVNNIPrefill_setBK256Mode(0);

    constexpr int M = 1;
    constexpr int N_Q = 512;
    constexpr int N_KV = 256;
    constexpr int K = 2048;

    auto wq = TestTensorFactory::createQ8_0Random({N_Q, K}, 9201u);
    auto wk = TestTensorFactory::createQ8_0Random({N_KV, K}, 9202u);
    auto wv = TestTensorFactory::createQ8_0Random({N_KV, K}, 9203u);
    ASSERT_TRUE(wq->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(wk->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(wv->ensureOnDevice(gpu_device_));

    auto *q_kernel = getPreparedKernel(wq.get(), gpu_device_);
    auto *k_kernel = getPreparedKernel(wk.get(), gpu_device_);
    auto *v_kernel = getPreparedKernel(wv.get(), gpu_device_);
    ASSERT_NE(q_kernel, nullptr);
    ASSERT_NE(k_kernel, nullptr);
    ASSERT_NE(v_kernel, nullptr);

    WorkspaceRequirements reqs;
    auto *q_ws = dynamic_cast<IWorkspaceConsumer *>(q_kernel);
    auto *k_ws = dynamic_cast<IWorkspaceConsumer *>(k_kernel);
    auto *v_ws = dynamic_cast<IWorkspaceConsumer *>(v_kernel);
    ASSERT_NE(q_ws, nullptr);
    ASSERT_NE(k_ws, nullptr);
    ASSERT_NE(v_ws, nullptr);
    reqs.merge(q_ws->getWorkspaceRequirements(M, N_Q, K));
    reqs.merge(k_ws->getWorkspaceRequirements(M, N_KV, K));
    reqs.merge(v_ws->getWorkspaceRequirements(M, N_KV, K));
    addCudaConcurrentDecodeGemvSideStreamWorkspace(
        reqs,
        gpu_device_,
        M,
        /*projection_count=*/3);

    ASSERT_EQ(cudaSetDevice(gpu_device_.ordinal), cudaSuccess);
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    workspace_ = std::make_unique<DeviceWorkspaceManager>(
        gpu_device_, reqs.total_bytes_with_alignment() + 4096);
    ASSERT_TRUE(workspace_->allocate(reqs));
    q_ws->bindWorkspace(workspace_.get());
    k_ws->bindWorkspace(workspace_.get());
    v_ws->bindWorkspace(workspace_.get());
    q_kernel->setGPUStream(stream);
    k_kernel->setGPUStream(stream);
    v_kernel->setGPUStream(stream);

    auto input = std::make_unique<FP32Tensor>(std::vector<size_t>{M, K});
    for (int i = 0; i < M * K; ++i)
        input->mutable_data()[i] = dist_(rng_);

    std::vector<float> ref_q;
    std::vector<float> ref_k;
    std::vector<float> ref_v;
    for (int rep = 0; rep < 4; ++rep)
    {
        auto q_output = std::make_unique<FP32Tensor>(std::vector<size_t>{M, N_Q});
        auto k_output = std::make_unique<FP32Tensor>(std::vector<size_t>{M, N_KV});
        auto v_output = std::make_unique<FP32Tensor>(std::vector<size_t>{M, N_KV});
        std::vector<TensorProjectionDesc> projections = {
            {q_kernel, q_output.get(), N_Q, nullptr, "Q"},
            {k_kernel, k_output.get(), N_KV, nullptr, "K"},
            {v_kernel, v_output.get(), N_KV, nullptr, "V"}};

        ASSERT_TRUE(with_gpu_coherence(
            gpu_device_,
            {input.get()},
            {q_output.get(), k_output.get(), v_output.get()},
            static_cast<void *>(stream),
            [&]
            {
                return q_kernel->multiply_fused_tensor(
                    input.get(), projections, M, K, nullptr, workspace_.get());
            }))
            << "concurrent fused decode repetition " << rep;

        const float *q = q_output->data();
        const float *k = k_output->data();
        const float *v = v_output->data();
        std::vector<float> q_snapshot(q, q + N_Q);
        std::vector<float> k_snapshot(k, k + N_KV);
        std::vector<float> v_snapshot(v, v + N_KV);
        if (rep == 0)
        {
            ref_q = std::move(q_snapshot);
            ref_k = std::move(k_snapshot);
            ref_v = std::move(v_snapshot);
            continue;
        }
        expectBitwiseEqual(q_snapshot, ref_q, "concurrent fused decode Q");
        expectBitwiseEqual(k_snapshot, ref_k, "concurrent fused decode K");
        expectBitwiseEqual(v_snapshot, ref_v, "concurrent fused decode V");
    }

    q_ws->unbindWorkspace();
    k_ws->unbindWorkspace();
    v_ws->unbindWorkspace();
    workspace_.reset();
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
#endif
}

// ============================================================================
// Custom main with MPI initialization
// ============================================================================

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
