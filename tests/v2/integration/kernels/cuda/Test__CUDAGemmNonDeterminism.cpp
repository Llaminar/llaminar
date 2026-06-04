/**
 * @file Test__CUDAGemmNonDeterminism.cpp
 * @brief Standalone test to reproduce CUDA GEMM non-determinism
 *
 * Exercises CUDAQuantisedGemmKernel multiply_fused_tensor() in isolation,
 * calling the same kernel with the same input N times and comparing outputs.
 *
 * This reproduces the issue seen in LocalPP_HOST_CUDA_CPU parity tests
 * where FFN_UP shows massive cosine variance (0.70-0.99) across runs.
 *
 * Tests:
 *  - Self-consistency: same kernel, same input → same output across N calls
 *  - Concurrent vs sequential dispatch
 *  - Shared workspace vs separate workspace
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
#include "../../../utils/TestModelHelper.h"
#include "loaders/ModelLoader.h"
#include "tensors/TensorFactory.h"
#include "utils/MPIContext.h"
#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#include <cuda_runtime.h>
#endif

#include "../../../utils/CUDATestUtils.h"
#include "../../../utils/GpuPreparedGemmHarness.h"
#include "../../../utils/TestTensorFactory.h"

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
    void cudaNativeVNNIPrefill_setStreamKMode(int mode);
    int cudaNativeVNNIPrefill_getStreamKMode();
    void cudaNativeVNNIPrefill_setBK256Mode(int mode);
    int cudaNativeVNNIPrefill_getBK256Mode();
    void cudaNativeVNNIPrefill_setDeterministicMode(bool enabled);
    bool cudaNativeVNNIPrefill_getDeterministicMode();
    void cudaNativeVNNIPrefill_setForceTile(int tile_id, int split_k);
    void cudaNativeVNNIPrefill_getForceTile(int *tile_id, int *split_k);
    void cudaNativeVNNIPrefill_getLastLaunchSelection(int *tile_id, int *split_k, int *used_bk256, int *used_streamk);
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
            : streamk_(cudaNativeVNNIPrefill_getStreamKMode()),
              bk256_(cudaNativeVNNIPrefill_getBK256Mode()),
              deterministic_(cudaNativeVNNIPrefill_getDeterministicMode())
        {
            cudaNativeVNNIPrefill_getForceTile(&force_tile_, &force_split_k_);
        }

        ~ScopedCudaPrefillModes()
        {
            cudaNativeVNNIPrefill_setForceTile(force_tile_, force_split_k_);
            cudaNativeVNNIPrefill_setStreamKMode(streamk_);
            cudaNativeVNNIPrefill_setBK256Mode(bk256_);
            cudaNativeVNNIPrefill_setDeterministicMode(deterministic_);
        }

        ScopedCudaPrefillModes(const ScopedCudaPrefillModes &) = delete;
        ScopedCudaPrefillModes &operator=(const ScopedCudaPrefillModes &) = delete;

    private:
        int force_tile_ = -1;
        int force_split_k_ = 0;
        int streamk_ = 0;
        int bk256_ = 0;
        bool deterministic_ = false;
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

class Test__CUDAGemmNonDeterminism : public CUDATestBase
{
protected:
    std::mt19937 rng_{42};
    std::uniform_real_distribution<float> dist_{-1.0f, 1.0f};
    std::unique_ptr<DeviceWorkspaceManager> workspace_;

    struct SplitKComparisonResult
    {
        std::string weight_name;
        int m = 0;
        int n = 0;
        int k = 0;
        int auto_tile = -1;
        int auto_split_k = 0;
        int auto_bk256 = 0;
        int auto_streamk = 0;
        DiffSummary sk1_repeat;
        DiffSummary sk2_repeat;
        DiffSummary skauto_repeat;
        DiffSummary sk1_vs_sk2;
        DiffSummary sk1_vs_skauto;
        double cosine_sk1_vs_sk2 = 0.0;
        double cosine_sk1_vs_skauto = 0.0;
    };

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

#ifdef HAVE_CUDA
    bool compareSplitKForWeight(const std::string &weight_name, SplitKComparisonResult &result)
    {
        auto mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
        TensorFactory factory(*mpi_ctx);
        ModelLoader loader(&factory);
        if (!tryLoadModel(loader, MODEL_PATH))
            return false;

        auto weight_base = loader.loadTensor(weight_name, DeviceId::cpu());
        auto *weight = dynamic_cast<Q4_0Tensor *>(weight_base.get());
        if (!weight)
            return false;

        const int M = 9;
        const int N = static_cast<int>(weight->shape()[0]);
        const int K = static_cast<int>(weight->shape()[1]);

        if (!weight->ensureOnDevice(gpu_device_))
            return false;

        auto *kernel = getPreparedKernel(
            weight, gpu_device_);
        if (!kernel)
            return false;

        auto *ws = dynamic_cast<IWorkspaceConsumer *>(kernel);
        if (!ws)
            return false;

        auto reqs = ws->getWorkspaceRequirements(M, N, K);
        workspace_ = std::make_unique<DeviceWorkspaceManager>(gpu_device_, 64 * 1024 * 1024);
        if (!workspace_->allocate(reqs))
            return false;
        ws->bindWorkspace(workspace_.get());

        auto input = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
        for (int i = 0; i < M * K; ++i)
            input->mutable_data()[i] = dist_(rng_);

        const int saved_streamk = cudaNativeVNNIPrefill_getStreamKMode();
        const int saved_bk256 = cudaNativeVNNIPrefill_getBK256Mode();
        const bool saved_det = cudaNativeVNNIPrefill_getDeterministicMode();
        int saved_force_tile = -1;
        int saved_force_sk = 0;
        cudaNativeVNNIPrefill_getForceTile(&saved_force_tile, &saved_force_sk);

        auto restore_modes = [&]()
        {
            cudaNativeVNNIPrefill_setForceTile(saved_force_tile, saved_force_sk);
            cudaNativeVNNIPrefill_setStreamKMode(saved_streamk);
            cudaNativeVNNIPrefill_setBK256Mode(saved_bk256);
            cudaNativeVNNIPrefill_setDeterministicMode(saved_det);
        };

        auto cleanup = [&]()
        {
            ws->unbindWorkspace();
            workspace_.reset();
        };

        auto run_projection = [&](int bk256_mode, int tile_id, int split_k, bool deterministic, std::vector<float> &out) -> bool
        {
            cudaNativeVNNIPrefill_setBK256Mode(bk256_mode);
            cudaNativeVNNIPrefill_setStreamKMode(-1);
            cudaNativeVNNIPrefill_setDeterministicMode(deterministic);
            cudaNativeVNNIPrefill_setForceTile(tile_id, split_k);

            auto output = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

            if (!with_gpu_coherence(
                    gpu_device_,
                    {input.get()},
                    {output.get()},
                    [&]
                    {
                        return kernel->multiply_tensor(
                            input.get(), output.get(), M, N, K,
                            true, 1.0f, 0.0f, nullptr, nullptr, -1);
                    }))
            {
                return false;
            }

            const float *data = output->data();
            out.assign(data, data + static_cast<size_t>(M) * N);
            return true;
        };

        std::vector<float> auto_out;
        if (!run_projection(/*bk256_mode=*/0, /*tile_id=*/-1, /*split_k=*/0, /*deterministic=*/false, auto_out))
        {
            restore_modes();
            cleanup();
            return false;
        }

        int auto_tile = -1;
        int auto_sk = 0;
        int auto_bk256 = 0;
        int auto_streamk = 0;
        cudaNativeVNNIPrefill_getLastLaunchSelection(&auto_tile, &auto_sk, &auto_bk256, &auto_streamk);

        const int forced_bk256_mode = auto_bk256 ? 1 : -1;
        const int forced_tile_id = auto_bk256 ? -1 : auto_tile;
        const int auto_forced_sk = (auto_sk > 1) ? auto_sk : 2;

        std::vector<float> sk1_a, sk1_b, sk2_a, sk2_b, skauto_a, skauto_b;
        const bool ok = run_projection(forced_bk256_mode, forced_tile_id, 1, false, sk1_a) &&
                        run_projection(forced_bk256_mode, forced_tile_id, 1, false, sk1_b) &&
                        run_projection(forced_bk256_mode, forced_tile_id, 2, false, sk2_a) &&
                        run_projection(forced_bk256_mode, forced_tile_id, 2, false, sk2_b) &&
                        run_projection(forced_bk256_mode, forced_tile_id, auto_forced_sk, false, skauto_a) &&
                        run_projection(forced_bk256_mode, forced_tile_id, auto_forced_sk, false, skauto_b);

        restore_modes();

        if (!ok)
        {
            cleanup();
            return false;
        }

        result.weight_name = weight_name;
        result.m = M;
        result.n = N;
        result.k = K;
        result.auto_tile = auto_tile;
        result.auto_split_k = auto_sk;
        result.auto_bk256 = auto_bk256;
        result.auto_streamk = auto_streamk;
        result.sk1_repeat = summarizeDiffs(sk1_a.data(), sk1_b.data(), sk1_a.size());
        result.sk2_repeat = summarizeDiffs(sk2_a.data(), sk2_b.data(), sk2_a.size());
        result.skauto_repeat = summarizeDiffs(skauto_a.data(), skauto_b.data(), skauto_a.size());
        result.sk1_vs_sk2 = summarizeDiffs(sk1_a.data(), sk2_a.data(), sk1_a.size());
        result.sk1_vs_skauto = summarizeDiffs(sk1_a.data(), skauto_a.data(), sk1_a.size());
        result.cosine_sk1_vs_sk2 = cosineSimilarity(sk1_a.data(), sk2_a.data(), sk1_a.size());
        result.cosine_sk1_vs_skauto = cosineSimilarity(sk1_a.data(), skauto_a.data(), sk1_a.size());

        cleanup();
        return true;
    }

    void printSplitKComparison(const SplitKComparisonResult &result)
    {
        std::cout << "Split-K comparison for " << result.weight_name
                  << " shape=(M=" << result.m << ",N=" << result.n << ",K=" << result.k << ")"
                  << " auto_tile=" << result.auto_tile
                  << " auto_split_k=" << result.auto_split_k
                  << " auto_bk256=" << result.auto_bk256
                  << " auto_streamk=" << result.auto_streamk << "\n";
        printDiffSummary("split_k=1 repeat", result.sk1_repeat);
        printDiffSummary("split_k=2 repeat", result.sk2_repeat);
        printDiffSummary("split_k=auto repeat", result.skauto_repeat);
        printDiffSummary("split_k=1 vs split_k=2", result.sk1_vs_sk2);
        printDiffSummary("split_k=1 vs split_k=auto", result.sk1_vs_skauto);
        std::cout << "split_k=1 vs split_k=2 cosine="
                  << std::fixed << std::setprecision(9) << result.cosine_sk1_vs_sk2 << "\n";
        std::cout << "split_k=1 vs split_k=auto cosine="
                  << std::fixed << std::setprecision(9) << result.cosine_sk1_vs_skauto << "\n";
    }
#endif
};

// ============================================================================
// Test: Fused Gate/Up GEMM self-consistency (N repeated calls)
//
// This directly mirrors the FFN gate_up_proj stage in Qwen2:
//   M=9, K=896, N_gate=N_up=4864
// with shared workspace, calling multiply_fused_tensor repeatedly.
// ============================================================================

TEST_F(Test__CUDAGemmNonDeterminism, FusedGateUp_SelfConsistency)
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

TEST_F(Test__CUDAGemmNonDeterminism, FusedQKV_SelfConsistency)
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
// If THIS is non-deterministic, it's the inner kernel (split-K atomicAdd).
// If this is deterministic but fused is not, it's the workspace sharing.
// ============================================================================

TEST_F(Test__CUDAGemmNonDeterminism, SingleKernel_SelfConsistency)
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

    // DIAGNOSTIC: This test reveals whether individual kernel calls are deterministic.
    // We expect split-K with atomicAdd to show ULP-level diffs, not massive corruption.
    EXPECT_GE(min_cos, 0.9999)
        << "Single multiply_tensor is non-deterministic across " << NUM_REPETITIONS << " calls";

    if (ws)
    {
        ws->unbindWorkspace();
        workspace_.reset();
    }
}

// ============================================================================
// Diagnostic: compare Q-projection split_k=1 vs split_k=2 on the same tile.
// Disabled by default because this is an investigation aid, not a stable CI test.
// Run manually with --gtest_also_run_disabled_tests.
// ============================================================================

TEST_F(Test__CUDAGemmNonDeterminism, DISABLED_QProjection_SplitKComparison)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA build required";
#else
    if (!std::filesystem::exists(MODEL_PATH))
        GTEST_SKIP() << "Model not found: " << MODEL_PATH;

    SplitKComparisonResult result;
    ASSERT_TRUE(compareSplitKForWeight("blk.0.attn_q.weight", result));
    printSplitKComparison(result);
#endif
}

TEST_F(Test__CUDAGemmNonDeterminism, DISABLED_FFNDown_SplitKComparison)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA build required";
#else
    if (!std::filesystem::exists(MODEL_PATH))
        GTEST_SKIP() << "Model not found: " << MODEL_PATH;

    SplitKComparisonResult result;
    ASSERT_TRUE(compareSplitKForWeight("blk.0.ffn_down.weight", result));
    printSplitKComparison(result);
#endif
}

TEST_F(Test__CUDAGemmNonDeterminism, NativeVNNI_SelfConsistency)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA build required";
#else
    // NativeVNNI is now the sole CUDA GEMM path. Verify it produces
    // bitwise-identical results across repeated invocations.
    ScopedCudaPrefillModes mode_guard;
    cudaNativeVNNIPrefill_setForceTile(-1, 0);
    cudaNativeVNNIPrefill_setStreamKMode(0);
    cudaNativeVNNIPrefill_setBK256Mode(0);
    cudaNativeVNNIPrefill_setDeterministicMode(false);

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
    int selected_split_k = 0;
    int used_bk256 = 0;
    int used_streamk = 0;
    cudaNativeVNNIPrefill_getLastLaunchSelection(
        &selected_tile, &selected_split_k, &used_bk256, &used_streamk);
    std::cout << "NativeVNNI final auto selection: tile=" << selected_tile
              << " split_k=" << selected_split_k
              << " bk256=" << used_bk256
              << " streamk=" << used_streamk << "\n";
    ws->unbindWorkspace();
    workspace_.reset();
#endif
}

TEST_F(Test__CUDAGemmNonDeterminism, NativeVNNI_GroupedSmallMDeterministicModeIsStable)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA build required";
#else
    ScopedCudaPrefillModes mode_guard;
    cudaNativeVNNIPrefill_setForceTile(-1, 0);
    cudaNativeVNNIPrefill_setStreamKMode(0);
    cudaNativeVNNIPrefill_setBK256Mode(0);
    cudaNativeVNNIPrefill_setDeterministicMode(true);

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

    auto output = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});
    ASSERT_TRUE(with_gpu_coherence(
        gpu_device_,
        {input.get()},
        {output.get()},
        [&]
        {
            return kernel->multiply_tensor(
                input.get(), output.get(), M, N, K, true, 1.0f, 0.0f, nullptr, nullptr, -1);
        }));

    int selected_tile = -1;
    int selected_split_k = 0;
    int used_bk256 = 0;
    int used_streamk = 0;
    cudaNativeVNNIPrefill_getLastLaunchSelection(
        &selected_tile, &selected_split_k, &used_bk256, &used_streamk);
    std::cout << "NativeVNNI grouped-small-M deterministic selection: tile=" << selected_tile
              << " split_k=" << selected_split_k
              << " bk256=" << used_bk256
              << " streamk=" << used_streamk << "\n";
    EXPECT_EQ(selected_split_k, 1)
        << "deterministic grouped-small-M prefill must avoid split-K accumulation-order drift";
    EXPECT_EQ(used_streamk, 0)
        << "deterministic grouped-small-M prefill must avoid stream-K atomic accumulation";

    ws->unbindWorkspace();
    workspace_.reset();
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
