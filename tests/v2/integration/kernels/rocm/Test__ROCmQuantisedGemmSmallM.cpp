/**
 * @file Test__ROCmQuantisedGemmSmallM.cpp
 * @brief Focused ROCm small-M GEMM regressions for MTP verifier decode.
 */

#include <gtest/gtest.h>

#include "execution/compute_stages/stages/GDNProjectionStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"
#include "../../../utils/GpuPreparedGemmHarness.h"
#include "../../../utils/TestTensorFactory.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

using namespace llaminar2;
using namespace llaminar2::rocm;
using namespace llaminar2::test;

namespace
{
    enum class PackedPath
    {
        INT8VNNI,
        NativeVNNI,
    };

    using WeightCreator = std::function<std::unique_ptr<TensorBase>(
        const std::vector<size_t> &shape,
        uint32_t seed)>;

    struct NativeFormatCase
    {
        const char *label;
        WeightCreator create;
        float min_cosine;
    };

    class ScopedEnv
    {
    public:
        ScopedEnv(const char *name, const char *value)
            : name_(name)
        {
            const char *old_value = std::getenv(name);
            if (old_value)
            {
                had_old_value_ = true;
                old_value_ = old_value;
            }
            setenv(name_.c_str(), value, 1);
            mutableDebugEnv().reload();
        }

        ~ScopedEnv()
        {
            if (had_old_value_)
                setenv(name_.c_str(), old_value_.c_str(), 1);
            else
                unsetenv(name_.c_str());
            mutableDebugEnv().reload();
        }

        ScopedEnv(const ScopedEnv &) = delete;
        ScopedEnv &operator=(const ScopedEnv &) = delete;

    private:
        std::string name_;
        bool had_old_value_ = false;
        std::string old_value_;
    };

    bool hasROCmDevice()
    {
#ifdef HAVE_ROCM
        int count = 0;
        const hipError_t err = hipGetDeviceCount(&count);
        return err == hipSuccess && count > 0;
#else
        return false;
#endif
    }

    void cpuFP32GemmRef(const float *A, const float *W, float *C, int M, int N, int K)
    {
        for (int row = 0; row < M; ++row)
        {
            for (int col = 0; col < N; ++col)
            {
                double acc = 0.0;
                for (int kk = 0; kk < K; ++kk)
                    acc += static_cast<double>(A[row * K + kk]) *
                           static_cast<double>(W[col * K + kk]);
                C[row * N + col] = static_cast<float>(acc);
            }
        }
    }

    float cosineSim(const float *a, const float *b, size_t n)
    {
        double dot = 0.0;
        double na = 0.0;
        double nb = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        if (na == 0.0 || nb == 0.0)
            return 0.0f;
        return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
    }

    void expectNearFP32(
        const std::vector<float> &actual,
        const std::vector<float> &expected,
        float abs_tolerance,
        const char *label)
    {
        ASSERT_EQ(actual.size(), expected.size()) << label;

        float max_abs = 0.0f;
        size_t max_index = 0;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            const float diff = std::fabs(actual[i] - expected[i]);
            if (diff > max_abs)
            {
                max_abs = diff;
                max_index = i;
            }
        }

        EXPECT_LE(max_abs, abs_tolerance)
            << label << " max_abs=" << max_abs
            << " index=" << max_index
            << " actual=" << actual[max_index]
            << " expected=" << expected[max_index];
    }

    std::unique_ptr<DeviceWorkspaceManager> bindWorkspace(
        ROCmQuantisedGemmKernel &kernel,
        int M,
        int N,
        int K)
    {
        auto workspace = std::make_unique<DeviceWorkspaceManager>(
            DeviceId::rocm(0),
            64 * 1024 * 1024);
        if (!workspace->allocate(kernel.getWorkspaceRequirements(M, N, K)))
            return nullptr;
        kernel.bindWorkspace(workspace.get());
        return workspace;
    }

    void expectPackedPath(const ROCmPackedWeights &packed, PackedPath path)
    {
        switch (path)
        {
        case PackedPath::INT8VNNI:
            EXPECT_FALSE(packed.int8_data_vnni.empty());
            EXPECT_TRUE(packed.native_vnni_payload.empty());
            break;
        case PackedPath::NativeVNNI:
            EXPECT_FALSE(packed.native_vnni_payload.empty());
            EXPECT_FALSE(packed.native_vnni_scales.empty());
            EXPECT_TRUE(packed.int8_data_vnni.empty());
            break;
        }
    }

    std::vector<NativeFormatCase> nativeFormatCases()
    {
        return {
            {"Q4_0", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ4_0Random(shape, seed); }, 0.985f},
            {"Q4_1", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ4_1Random(shape, seed); }, 0.985f},
            {"Q5_0", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ5_0Random(shape, seed); }, 0.985f},
            {"Q5_1", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ5_1Random(shape, seed); }, 0.985f},
            {"Q6_K", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ6_KRandom(shape, seed); }, 0.985f},
            {"Q3_K", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ3_KRandom(shape, seed); }, 0.985f},
            {"Q2_K", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ2_KRandom(shape, seed); }, 0.985f},
            {"Q4_K", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ4_KRandom(shape, seed); }, 0.985f},
            {"Q5_K", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createQ5_KRandom(shape, seed); }, 0.985f},
            {"IQ4_NL", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ4_NLRandom(shape, seed); }, 0.985f},
            {"IQ4_XS", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ4_XSRandom(shape, seed); }, 0.985f},
            {"IQ3_S", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ3_SRandom(shape, seed); }, 0.985f},
            {"IQ3_XXS", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ3_XXSRandom(shape, seed); }, 0.985f},
            {"IQ2_S", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ2_SRandom(shape, seed); }, 0.985f},
            {"IQ2_XS", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ2_XSRandom(shape, seed); }, 0.985f},
            {"IQ2_XXS", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ2_XXSRandom(shape, seed); }, 0.985f},
            {"IQ1_S", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ1_SRandom(shape, seed); }, 0.985f},
            {"IQ1_M", [](const std::vector<size_t> &shape, uint32_t seed) -> std::unique_ptr<TensorBase>
             { return TestTensorFactory::createIQ1_MRandom(shape, seed); }, 0.985f},
        };
    }

    template <typename CreateWeights>
    void runDispatchSmallMMatchesReference(
        const char *label,
        int M,
        int N,
        int K,
        PackedPath expected_path,
        CreateWeights createWeights,
        float min_cosine)
    {
        auto weights = createWeights(
            {static_cast<size_t>(N), static_cast<size_t>(K)},
            42);
        std::vector<float> W_fp32(static_cast<size_t>(N) * K);
        weights->to_fp32(W_fp32.data());

        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
        expectPackedPath(packed, expected_path);

        ROCmQuantisedGemmKernel kernel(&packed, 0);
        auto workspace = bindWorkspace(kernel, M, N, K);
        ASSERT_NE(workspace, nullptr);

        auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
        auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});

        ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(output->allocateOnDevice(DeviceId::rocm(0)));

        ASSERT_TRUE(kernel.multiply_tensor(input.get(), output.get(), M, N, K));
#ifdef HAVE_ROCM
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
#endif
        output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

        std::vector<float> ref(static_cast<size_t>(M) * N);
        cpuFP32GemmRef(input->data(), W_fp32.data(), ref.data(), M, N, K);

        const float cos = cosineSim(output->data(), ref.data(), static_cast<size_t>(M) * N);
        LOG_INFO("[SmallM] " << label << " M=" << M << " cosine=" << cos);
        EXPECT_GT(cos, min_cosine);

        kernel.unbindWorkspace();
    }

    inline float swigluRef(float gate, float up)
    {
        return gate / (1.0f + std::exp(-gate)) * up;
    }

    template <typename CreateWeights>
    void runFusedSwiGLUDownSmallMMatchesReference(
        const char *label,
        int M,
        int N,
        int K,
        PackedPath expected_path,
        CreateWeights createWeights,
        float min_cosine,
        bool graph_capture = false,
        int graph_replays = 1)
    {
        ASSERT_GE(graph_replays, 1);

        auto weights = createWeights(
            {static_cast<size_t>(N), static_cast<size_t>(K)},
            4242);
        std::vector<float> W_fp32(static_cast<size_t>(N) * K);
        weights->to_fp32(W_fp32.data());

        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
        expectPackedPath(packed, expected_path);

        ROCmQuantisedGemmKernel kernel(&packed, 0);
        kernel.prepareWeights();
        ASSERT_TRUE(kernel.weights_converted());

#ifdef HAVE_ROCM
        hipStream_t stream = nullptr;
        if (graph_capture)
        {
            ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
            kernel.setGPUStream(stream);
        }
#endif

        auto workspace = bindWorkspace(kernel, M, N, K);
        ASSERT_NE(workspace, nullptr);

        auto gate = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)},
            -0.5f,
            0.5f,
            5151);
        auto up = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)},
            -0.5f,
            0.5f,
            6161);
        auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});

        ASSERT_TRUE(gate->ensureOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(up->ensureOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(output->allocateOnDevice(DeviceId::rocm(0)));

#ifdef HAVE_ROCM
        hipGraph_t graph = nullptr;
        hipGraphExec_t exec = nullptr;
        if (graph_capture)
        {
            ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
            ASSERT_TRUE(kernel.multiply_tensor_with_fused_swiglu(
                gate.get(),
                up.get(),
                output.get(),
                M,
                N,
                K));
            ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
            ASSERT_NE(graph, nullptr);
            ASSERT_EQ(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), hipSuccess);
            ASSERT_NE(exec, nullptr);

            for (int replay = 0; replay < graph_replays; ++replay)
            {
                ASSERT_EQ(hipMemsetAsync(output->gpu_data_ptr(),
                                         0,
                                         static_cast<size_t>(M) * N * sizeof(float),
                                         stream),
                          hipSuccess);
                ASSERT_EQ(hipGraphLaunch(exec, stream), hipSuccess)
                    << "replay=" << replay;
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess)
                    << "replay=" << replay;
            }
        }
        else
#endif
        {
            ASSERT_TRUE(kernel.multiply_tensor_with_fused_swiglu(
                gate.get(),
                up.get(),
                output.get(),
                M,
                N,
                K));
#ifdef HAVE_ROCM
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
#endif
        }
        output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

        std::vector<float> swiglu(static_cast<size_t>(M) * K);
        for (size_t i = 0; i < swiglu.size(); ++i)
            swiglu[i] = swigluRef(gate->data()[i], up->data()[i]);

        std::vector<float> ref(static_cast<size_t>(M) * N);
        cpuFP32GemmRef(swiglu.data(), W_fp32.data(), ref.data(), M, N, K);

        const float cos = cosineSim(output->data(), ref.data(), static_cast<size_t>(M) * N);
        LOG_INFO("[SmallM] " << label << " fused SwiGLU down M=" << M << " cosine=" << cos);
        EXPECT_GT(cos, min_cosine);

#ifdef HAVE_ROCM
        if (exec)
            EXPECT_EQ(hipGraphExecDestroy(exec), hipSuccess);
        if (graph)
            EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
        if (stream)
        {
            kernel.setGPUStream(nullptr);
            EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
        }
#endif
        kernel.unbindWorkspace();
    }

    template <typename CreateWeights>
    void runFusedQKVSmallMMatchesSeparate(
        const char *label,
        int M,
        int K,
        PackedPath expected_path,
        CreateWeights createWeights,
        float min_cosine,
        int Nq = 896,
        int Nk = 128,
        int Nv = 128)
    {
        auto wq = createWeights({static_cast<size_t>(Nq), static_cast<size_t>(K)}, 101);
        auto wk = createWeights({static_cast<size_t>(Nk), static_cast<size_t>(K)}, 102);
        auto wv = createWeights({static_cast<size_t>(Nv), static_cast<size_t>(K)}, 103);

        ROCmPackedWeights packed_q;
        ROCmPackedWeights packed_k;
        ROCmPackedWeights packed_v;
        ASSERT_TRUE(packWeightsToROCm(wq.get(), packed_q));
        ASSERT_TRUE(packWeightsToROCm(wk.get(), packed_k));
        ASSERT_TRUE(packWeightsToROCm(wv.get(), packed_v));
        expectPackedPath(packed_q, expected_path);
        expectPackedPath(packed_k, expected_path);
        expectPackedPath(packed_v, expected_path);

        ROCmQuantisedGemmKernel q_kernel(&packed_q, 0);
        ROCmQuantisedGemmKernel k_kernel(&packed_k, 0);
        ROCmQuantisedGemmKernel v_kernel(&packed_v, 0);

        auto q_workspace = bindWorkspace(q_kernel, M, Nq, K);
        auto k_workspace = bindWorkspace(k_kernel, M, Nk, K);
        auto v_workspace = bindWorkspace(v_kernel, M, Nv, K);
        ASSERT_NE(q_workspace, nullptr);
        ASSERT_NE(k_workspace, nullptr);
        ASSERT_NE(v_workspace, nullptr);

        auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
        auto separate_q = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(Nq)});
        auto separate_k = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(Nk)});
        auto separate_v = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(Nv)});
        auto fused_q = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(Nq)});
        auto fused_k = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(Nk)});
        auto fused_v = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(Nv)});
        auto bias_q = TestTensorFactory::createFP32({static_cast<size_t>(Nq)});
        auto bias_k = TestTensorFactory::createFP32({static_cast<size_t>(Nk)});
        auto bias_v = TestTensorFactory::createFP32({static_cast<size_t>(Nv)});

        for (int col = 0; col < Nq; ++col)
            bias_q->mutable_data()[col] = 0.03125f * static_cast<float>((col % 9) - 4);
        for (int col = 0; col < Nk; ++col)
        {
            bias_k->mutable_data()[col] = 0.03125f * static_cast<float>((col % 7) - 3);
            bias_v->mutable_data()[col] = 0.03125f * static_cast<float>((col % 5) - 2);
        }

        ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(separate_q->allocateOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(separate_k->allocateOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(separate_v->allocateOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(fused_q->allocateOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(fused_k->allocateOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(fused_v->allocateOnDevice(DeviceId::rocm(0)));

        ASSERT_TRUE(q_kernel.multiply_tensor(input.get(), separate_q.get(), M, Nq, K,
                                             true, 1.0f, 0.0f, bias_q.get()));
        ASSERT_TRUE(k_kernel.multiply_tensor(input.get(), separate_k.get(), M, Nk, K,
                                             true, 1.0f, 0.0f, bias_k.get()));
        ASSERT_TRUE(v_kernel.multiply_tensor(input.get(), separate_v.get(), M, Nv, K,
                                             true, 1.0f, 0.0f, bias_v.get()));
#ifdef HAVE_ROCM
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
#endif
        separate_q->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        separate_k->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        separate_v->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

        std::vector<ITensorGemm::TensorProjectionDesc> projections;
        projections.emplace_back(&q_kernel, fused_q.get(), Nq, bias_q.get(), "q_small_m");
        projections.emplace_back(&k_kernel, fused_k.get(), Nk, bias_k.get(), "k_small_m");
        projections.emplace_back(&v_kernel, fused_v.get(), Nv, bias_v.get(), "v_small_m");

        ASSERT_TRUE(q_kernel.multiply_fused_tensor(input.get(), projections, M, K));
#ifdef HAVE_ROCM
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
#endif
        fused_q->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        fused_k->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        fused_v->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

        const float q_cos = cosineSim(fused_q->data(), separate_q->data(), static_cast<size_t>(M) * Nq);
        const float k_cos = cosineSim(fused_k->data(), separate_k->data(), static_cast<size_t>(M) * Nk);
        const float v_cos = cosineSim(fused_v->data(), separate_v->data(), static_cast<size_t>(M) * Nv);

        LOG_INFO("[SmallM] " << label << " fused QKV M=" << M << " cosine q="
                             << q_cos << " k=" << k_cos << " v=" << v_cos);
        EXPECT_GT(q_cos, min_cosine);
        EXPECT_GT(k_cos, min_cosine);
        EXPECT_GT(v_cos, min_cosine);

        q_kernel.unbindWorkspace();
        k_kernel.unbindWorkspace();
        v_kernel.unbindWorkspace();
    }

    template <typename CreateWeights>
    void runFusedProjectionGroupSmallMMatchesSeparate(
        const char *label,
        int M,
        int K,
        PackedPath expected_path,
        CreateWeights createWeights,
        float min_cosine,
        const std::vector<int> &Ns,
        bool graph_capture,
        bool bind_fused_to_shared_workspace = false,
        int graph_replays = 1)
    {
        ASSERT_FALSE(Ns.empty());
        ASSERT_GE(graph_replays, 1);

#ifdef HAVE_ROCM
        hipStream_t stream = nullptr;
        if (graph_capture)
            ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
#endif

        std::vector<std::unique_ptr<TensorBase>> weights;
        std::vector<ROCmPackedWeights> packed(Ns.size());
        std::vector<std::unique_ptr<ROCmQuantisedGemmKernel>> kernels;
        std::vector<std::unique_ptr<DeviceWorkspaceManager>> workspaces;
        weights.reserve(Ns.size());
        kernels.reserve(Ns.size());
        workspaces.reserve(Ns.size());

        for (size_t i = 0; i < Ns.size(); ++i)
        {
            weights.push_back(createWeights({static_cast<size_t>(Ns[i]), static_cast<size_t>(K)},
                                            static_cast<uint32_t>(200 + i)));
            ASSERT_TRUE(packWeightsToROCm(weights.back().get(), packed[i]));
            expectPackedPath(packed[i], expected_path);
            kernels.push_back(std::make_unique<ROCmQuantisedGemmKernel>(&packed[i], 0));
#ifdef HAVE_ROCM
            if (stream)
                kernels.back()->setGPUStream(stream);
#endif
            workspaces.push_back(bindWorkspace(*kernels.back(), M, Ns[i], K));
            ASSERT_NE(workspaces.back(), nullptr);
        }

        auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
        ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));

        std::vector<std::unique_ptr<FP32Tensor>> separate;
        std::vector<std::unique_ptr<FP32Tensor>> fused;
        separate.reserve(Ns.size());
        fused.reserve(Ns.size());
        for (int n : Ns)
        {
            separate.push_back(TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(n)}));
            fused.push_back(TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(n)}));
            ASSERT_TRUE(separate.back()->allocateOnDevice(DeviceId::rocm(0)));
            ASSERT_TRUE(fused.back()->allocateOnDevice(DeviceId::rocm(0)));
        }

        for (size_t i = 0; i < Ns.size(); ++i)
        {
            ASSERT_TRUE(kernels[i]->multiply_tensor(input.get(), separate[i].get(), M, Ns[i], K));
        }
#ifdef HAVE_ROCM
        ASSERT_EQ(stream ? hipStreamSynchronize(stream) : hipDeviceSynchronize(), hipSuccess);
#endif
        for (auto &output : separate)
            output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

        std::unique_ptr<DeviceWorkspaceManager> shared_workspace;
        if (bind_fused_to_shared_workspace)
        {
            WorkspaceRequirements combined;
            for (size_t i = 0; i < Ns.size(); ++i)
                combined.merge(kernels[i]->getWorkspaceRequirements(M, Ns[i], K));

            shared_workspace = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(0),
                combined.total_bytes_with_alignment() + 64 * 1024 * 1024);
            ASSERT_TRUE(shared_workspace->allocate(combined));

            for (auto &kernel : kernels)
                kernel->bindWorkspace(shared_workspace.get());
        }

        std::vector<ITensorGemm::TensorProjectionDesc> projections;
        projections.reserve(Ns.size());
        for (size_t i = 0; i < Ns.size(); ++i)
        {
            projections.emplace_back(kernels[i].get(),
                                     fused[i].get(),
                                     Ns[i],
                                     nullptr,
                                     "small_m_group");
        }

#ifdef HAVE_ROCM
        hipGraph_t graph = nullptr;
        hipGraphExec_t exec = nullptr;
        if (graph_capture)
        {
            ASSERT_TRUE(kernels.front()->multiply_fused_tensor(input.get(), projections, M, K));
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

            ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
            ASSERT_TRUE(kernels.front()->multiply_fused_tensor(input.get(), projections, M, K));
            ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
            ASSERT_NE(graph, nullptr);
            ASSERT_EQ(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), hipSuccess);
            ASSERT_NE(exec, nullptr);
            for (int replay = 0; replay < graph_replays; ++replay)
            {
                for (size_t i = 0; i < Ns.size(); ++i)
                {
                    ASSERT_EQ(hipMemsetAsync(fused[i]->gpu_data_ptr(),
                                             0,
                                             static_cast<size_t>(M) * Ns[i] * sizeof(float),
                                             stream),
                              hipSuccess);
                }
                ASSERT_EQ(hipGraphLaunch(exec, stream), hipSuccess)
                    << "replay=" << replay;
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess)
                    << "replay=" << replay;
            }
        }
        else
#endif
        {
            ASSERT_TRUE(kernels.front()->multiply_fused_tensor(input.get(), projections, M, K));
#ifdef HAVE_ROCM
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
#endif
        }

        for (size_t i = 0; i < Ns.size(); ++i)
        {
            fused[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
            const float cos = cosineSim(fused[i]->data(),
                                        separate[i]->data(),
                                        static_cast<size_t>(M) * Ns[i]);
            LOG_INFO("[SmallM] " << label << " projection=" << i
                                 << " M=" << M << " N=" << Ns[i]
                                 << " cosine=" << cos);
            EXPECT_GT(cos, min_cosine);
        }

#ifdef HAVE_ROCM
        if (exec)
            EXPECT_EQ(hipGraphExecDestroy(exec), hipSuccess);
        if (graph)
            EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
        if (stream)
            EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
        for (auto &kernel : kernels)
            kernel->unbindWorkspace();
    }

    template <typename CreateWeights>
    void runFusedProjectionGroupSmallMMatchesReference(
        const char *label,
        int M,
        int K,
        PackedPath expected_path,
        CreateWeights createWeights,
        float min_cosine,
        const std::vector<int> &Ns,
        const std::vector<const char *> &projection_names,
        bool graph_capture,
        bool bind_fused_to_shared_workspace = false,
        int graph_replays = 1)
    {
        ASSERT_FALSE(Ns.empty());
        ASSERT_EQ(Ns.size(), projection_names.size());
        ASSERT_GE(graph_replays, 1);

#ifdef HAVE_ROCM
        hipStream_t stream = nullptr;
        if (graph_capture)
            ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
#endif

        std::vector<std::unique_ptr<TensorBase>> weights;
        std::vector<std::vector<float>> weights_fp32;
        std::vector<ROCmPackedWeights> packed(Ns.size());
        std::vector<std::unique_ptr<ROCmQuantisedGemmKernel>> kernels;
        std::vector<std::unique_ptr<DeviceWorkspaceManager>> workspaces;
        weights.reserve(Ns.size());
        weights_fp32.reserve(Ns.size());
        kernels.reserve(Ns.size());
        workspaces.reserve(Ns.size());

        for (size_t i = 0; i < Ns.size(); ++i)
        {
            weights.push_back(createWeights({static_cast<size_t>(Ns[i]), static_cast<size_t>(K)},
                                            static_cast<uint32_t>(900 + i)));
            weights_fp32.emplace_back(static_cast<size_t>(Ns[i]) * static_cast<size_t>(K));
            weights.back()->to_fp32(weights_fp32.back().data());

            ASSERT_TRUE(packWeightsToROCm(weights.back().get(), packed[i]));
            expectPackedPath(packed[i], expected_path);
            kernels.push_back(std::make_unique<ROCmQuantisedGemmKernel>(&packed[i], 0));
#ifdef HAVE_ROCM
            if (stream)
                kernels.back()->setGPUStream(stream);
#endif
            workspaces.push_back(bindWorkspace(*kernels.back(), M, Ns[i], K));
            ASSERT_NE(workspaces.back(), nullptr);
        }

        auto input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(M), static_cast<size_t>(K)},
            -0.75f,
            0.75f,
            8080);
        ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));

        std::vector<std::unique_ptr<FP32Tensor>> fused;
        fused.reserve(Ns.size());
        for (int n : Ns)
        {
            fused.push_back(TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(n)}));
            ASSERT_TRUE(fused.back()->allocateOnDevice(DeviceId::rocm(0)));
        }

        std::unique_ptr<DeviceWorkspaceManager> shared_workspace;
        if (bind_fused_to_shared_workspace)
        {
            WorkspaceRequirements combined;
            for (size_t i = 0; i < Ns.size(); ++i)
                combined.merge(kernels[i]->getWorkspaceRequirements(M, Ns[i], K));

            shared_workspace = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(0),
                combined.total_bytes_with_alignment() + 64 * 1024 * 1024);
            ASSERT_TRUE(shared_workspace->allocate(combined));

            for (auto &kernel : kernels)
                kernel->bindWorkspace(shared_workspace.get());
        }

        std::vector<ITensorGemm::TensorProjectionDesc> projections;
        projections.reserve(Ns.size());
        for (size_t i = 0; i < Ns.size(); ++i)
        {
            projections.emplace_back(kernels[i].get(),
                                     fused[i].get(),
                                     Ns[i],
                                     nullptr,
                                     projection_names[i]);
        }

#ifdef HAVE_ROCM
        hipGraph_t graph = nullptr;
        hipGraphExec_t exec = nullptr;
        if (graph_capture)
        {
            ASSERT_TRUE(kernels.front()->multiply_fused_tensor(input.get(), projections, M, K));
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

            ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
            ASSERT_TRUE(kernels.front()->multiply_fused_tensor(input.get(), projections, M, K));
            ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
            ASSERT_NE(graph, nullptr);
            ASSERT_EQ(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), hipSuccess);
            ASSERT_NE(exec, nullptr);

            for (int replay = 0; replay < graph_replays; ++replay)
            {
                for (size_t i = 0; i < Ns.size(); ++i)
                {
                    ASSERT_EQ(hipMemsetAsync(fused[i]->gpu_data_ptr(),
                                             0,
                                             static_cast<size_t>(M) * static_cast<size_t>(Ns[i]) * sizeof(float),
                                             stream),
                              hipSuccess);
                }
                ASSERT_EQ(hipGraphLaunch(exec, stream), hipSuccess)
                    << "replay=" << replay;
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess)
                    << "replay=" << replay;
            }
        }
        else
#endif
        {
            ASSERT_TRUE(kernels.front()->multiply_fused_tensor(input.get(), projections, M, K));
#ifdef HAVE_ROCM
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
#endif
        }

        for (size_t i = 0; i < Ns.size(); ++i)
        {
            fused[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

            std::vector<float> ref(static_cast<size_t>(M) * static_cast<size_t>(Ns[i]));
            cpuFP32GemmRef(input->data(), weights_fp32[i].data(), ref.data(), M, Ns[i], K);

            const float cos = cosineSim(fused[i]->data(),
                                        ref.data(),
                                        static_cast<size_t>(M) * static_cast<size_t>(Ns[i]));
            LOG_INFO("[SmallM] " << label
                                 << " projection=" << projection_names[i]
                                 << " M=" << M << " N=" << Ns[i]
                                 << " cosine=" << cos);
            EXPECT_GT(cos, min_cosine)
                << "projection=" << projection_names[i]
                << " N=" << Ns[i];
        }

#ifdef HAVE_ROCM
        if (exec)
            EXPECT_EQ(hipGraphExecDestroy(exec), hipSuccess);
        if (graph)
            EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
        if (stream)
            EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
        for (auto &kernel : kernels)
            kernel->unbindWorkspace();
    }

    void runMixedProjectionGroupSmallMMatchesSeparate(
        const char *label,
        int M,
        int K,
        const std::vector<WeightCreator> &createWeights,
        float min_cosine,
        const std::vector<int> &Ns,
        bool graph_capture,
        int graph_replays = 1)
    {
        ASSERT_FALSE(Ns.empty());
        ASSERT_EQ(createWeights.size(), Ns.size());
        ASSERT_GE(graph_replays, 1);

#ifdef HAVE_ROCM
        hipStream_t stream = nullptr;
        if (graph_capture)
            ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
#endif

        std::vector<std::unique_ptr<TensorBase>> weights;
        std::vector<ROCmPackedWeights> packed(Ns.size());
        std::vector<std::unique_ptr<ROCmQuantisedGemmKernel>> kernels;
        std::vector<std::unique_ptr<DeviceWorkspaceManager>> workspaces;
        weights.reserve(Ns.size());
        kernels.reserve(Ns.size());
        workspaces.reserve(Ns.size());

        for (size_t i = 0; i < Ns.size(); ++i)
        {
            weights.push_back(createWeights[i]({static_cast<size_t>(Ns[i]), static_cast<size_t>(K)},
                                               static_cast<uint32_t>(700 + i)));
            ASSERT_TRUE(packWeightsToROCm(weights.back().get(), packed[i]));
            expectPackedPath(packed[i], PackedPath::NativeVNNI);
            kernels.push_back(std::make_unique<ROCmQuantisedGemmKernel>(&packed[i], 0));
#ifdef HAVE_ROCM
            if (stream)
                kernels.back()->setGPUStream(stream);
#endif
            workspaces.push_back(bindWorkspace(*kernels.back(), M, Ns[i], K));
            ASSERT_NE(workspaces.back(), nullptr);
        }

        auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
        ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));

        std::vector<std::unique_ptr<FP32Tensor>> separate;
        std::vector<std::unique_ptr<FP32Tensor>> fused;
        separate.reserve(Ns.size());
        fused.reserve(Ns.size());
        for (int n : Ns)
        {
            separate.push_back(TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(n)}));
            fused.push_back(TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(n)}));
            ASSERT_TRUE(separate.back()->allocateOnDevice(DeviceId::rocm(0)));
            ASSERT_TRUE(fused.back()->allocateOnDevice(DeviceId::rocm(0)));
        }

        for (size_t i = 0; i < Ns.size(); ++i)
            ASSERT_TRUE(kernels[i]->multiply_tensor(input.get(), separate[i].get(), M, Ns[i], K));
#ifdef HAVE_ROCM
        ASSERT_EQ(stream ? hipStreamSynchronize(stream) : hipDeviceSynchronize(), hipSuccess);
#endif
        for (auto &output : separate)
            output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

        WorkspaceRequirements combined;
        for (size_t i = 0; i < Ns.size(); ++i)
            combined.merge(kernels[i]->getWorkspaceRequirements(M, Ns[i], K));

        auto shared_workspace = std::make_unique<DeviceWorkspaceManager>(
            DeviceId::rocm(0),
            combined.total_bytes_with_alignment() + 64 * 1024 * 1024);
        ASSERT_TRUE(shared_workspace->allocate(combined));
        for (auto &kernel : kernels)
            kernel->bindWorkspace(shared_workspace.get());

        std::vector<ITensorGemm::TensorProjectionDesc> projections;
        projections.reserve(Ns.size());
        for (size_t i = 0; i < Ns.size(); ++i)
        {
            projections.emplace_back(kernels[i].get(),
                                     fused[i].get(),
                                     Ns[i],
                                     nullptr,
                                     "mixed_small_m_group");
        }

#ifdef HAVE_ROCM
        hipGraph_t graph = nullptr;
        hipGraphExec_t exec = nullptr;
        if (graph_capture)
        {
            ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
            ASSERT_TRUE(kernels.front()->multiply_fused_tensor(input.get(), projections, M, K));
            ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
            ASSERT_NE(graph, nullptr);
            ASSERT_EQ(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), hipSuccess);
            ASSERT_NE(exec, nullptr);
            for (int replay = 0; replay < graph_replays; ++replay)
            {
                for (size_t i = 0; i < Ns.size(); ++i)
                {
                    ASSERT_EQ(hipMemsetAsync(fused[i]->gpu_data_ptr(),
                                             0,
                                             static_cast<size_t>(M) * Ns[i] * sizeof(float),
                                             stream),
                              hipSuccess);
                }
                ASSERT_EQ(hipGraphLaunch(exec, stream), hipSuccess)
                    << "replay=" << replay;
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess)
                    << "replay=" << replay;
            }
        }
        else
#endif
        {
            ASSERT_TRUE(kernels.front()->multiply_fused_tensor(input.get(), projections, M, K));
#ifdef HAVE_ROCM
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
#endif
        }

        for (size_t i = 0; i < Ns.size(); ++i)
        {
            fused[i]->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
            const float cos = cosineSim(fused[i]->data(),
                                        separate[i]->data(),
                                        static_cast<size_t>(M) * Ns[i]);
            LOG_INFO("[SmallM] " << label << " mixed projection=" << i
                                 << " M=" << M << " N=" << Ns[i]
                                 << " cosine=" << cos);
            EXPECT_GT(cos, min_cosine);
        }

#ifdef HAVE_ROCM
        if (exec)
            EXPECT_EQ(hipGraphExecDestroy(exec), hipSuccess);
        if (graph)
            EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
        if (stream)
            EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
        for (auto &kernel : kernels)
            kernel->unbindWorkspace();
    }

#ifdef HAVE_ROCM
    template <typename CreateWeights>
    void runGraphCapturedDispatchSmallMMatchesReference(
        const char *label,
        int M,
        int N,
        int K,
        CreateWeights createWeights,
        float min_cosine)
    {
        auto weights = createWeights(
            {static_cast<size_t>(N), static_cast<size_t>(K)},
            777);
        std::vector<float> W_fp32(static_cast<size_t>(N) * K);
        weights->to_fp32(W_fp32.data());

        ROCmPackedWeights packed;
        ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
        expectPackedPath(packed, PackedPath::NativeVNNI);

        hipStream_t stream = nullptr;
        ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);

        ROCmQuantisedGemmKernel kernel(&packed, 0);
        kernel.setGPUStream(stream);
        auto workspace = bindWorkspace(kernel, M, N, K);
        ASSERT_NE(workspace, nullptr);

        auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(M), static_cast<size_t>(K)});
        auto output = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});

        ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(output->allocateOnDevice(DeviceId::rocm(0)));

        ASSERT_TRUE(kernel.multiply_tensor(input.get(), output.get(), M, N, K));
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

        hipGraph_t graph = nullptr;
        hipGraphExec_t exec = nullptr;
        ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
        ASSERT_TRUE(kernel.multiply_tensor(input.get(), output.get(), M, N, K));
        ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
        ASSERT_NE(graph, nullptr);
        ASSERT_EQ(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), hipSuccess);
        ASSERT_NE(exec, nullptr);

        ASSERT_EQ(hipMemsetAsync(output->gpu_data_ptr(), 0, static_cast<size_t>(M) * N * sizeof(float), stream),
                  hipSuccess);
        ASSERT_EQ(hipGraphLaunch(exec, stream), hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        output->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

        std::vector<float> ref(static_cast<size_t>(M) * N);
        cpuFP32GemmRef(input->data(), W_fp32.data(), ref.data(), M, N, K);

        const float cos = cosineSim(output->data(), ref.data(), static_cast<size_t>(M) * N);
        LOG_INFO("[SmallM] " << label << " graph-captured M=" << M << " cosine=" << cos);
        EXPECT_GT(cos, min_cosine);

        if (exec)
            EXPECT_EQ(hipGraphExecDestroy(exec), hipSuccess);
        if (graph)
            EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
        kernel.setGPUStream(nullptr);
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
        kernel.unbindWorkspace();
    }
#endif
}

TEST(Test__ROCmQuantisedGemmSmallM, DispatchQ80M2MatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const int N = 896;
    const int K = 896;
    runDispatchSmallMMatchesReference(
        "Q8_0 INT8-VNNI",
        2,
        N,
        K,
        PackedPath::INT8VNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ8_0Random(shape, seed); },
        0.985f);
}

TEST(Test__ROCmQuantisedGemmSmallM, DispatchQ4KM2MatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const int N = 896;
    const int K = 1024;
    runDispatchSmallMMatchesReference(
        "Q4_K native-VNNI",
        2,
        N,
        K,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.985f);
}

TEST(Test__ROCmQuantisedGemmSmallM, DispatchQ80SmallMMatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const int N = 512;
    const int K = 1024;
    for (int M : {2, 3, 4})
    {
        runDispatchSmallMMatchesReference(
            "Q8_0 INT8-VNNI small-M sweep",
            M,
            N,
            K,
            PackedPath::INT8VNNI,
            [](const std::vector<size_t> &shape, uint32_t seed)
            { return TestTensorFactory::createQ8_0Random(shape, seed); },
            0.985f);
    }
}

TEST(Test__ROCmQuantisedGemmSmallM, DispatchNativeSmallMAllCodebooksMatchReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const int N = 512;
    const int K = 1024;
    for (const auto &format : nativeFormatCases())
    {
        for (int M : {2, 3, 4})
        {
            runDispatchSmallMMatchesReference(
                format.label,
                M,
                N,
                K,
                PackedPath::NativeVNNI,
                format.create,
                format.min_cosine);
        }
    }
}

TEST(Test__ROCmQuantisedGemmSmallM, DispatchPlainAsymmetricNativeSmallMUsesFreshBlockSums)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const int N = 384;
    const int K = 768;
    for (int M : {2, 4})
    {
        runDispatchSmallMMatchesReference(
            "Q4_1 native-VNNI asymmetric min correction",
            M,
            N,
            K,
            PackedPath::NativeVNNI,
            [](const std::vector<size_t> &shape, uint32_t seed)
            { return TestTensorFactory::createQ4_1Random(shape, seed); },
            0.985f);

        runDispatchSmallMMatchesReference(
            "Q5_1 native-VNNI asymmetric min correction",
            M,
            N,
            K,
            PackedPath::NativeVNNI,
            [](const std::vector<size_t> &shape, uint32_t seed)
            { return TestTensorFactory::createQ5_1Random(shape, seed); },
            0.985f);
    }
}

TEST(Test__ROCmQuantisedGemmSmallM, DispatchQ4KM2RecordsNativeRouteCounter)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    const int N = 896;
    const int K = 1024;
    runDispatchSmallMMatchesReference(
        "Q4_K native-VNNI counter",
        2,
        N,
        K,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.985f);

    const auto records = PerfStatsCollector::snapshot({"kernel.rocm_native_vnni_m2_calls"});
    auto route_record = std::find_if(
        records.begin(),
        records.end(),
        [](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "rocm_native_vnni_m2_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter;
        });

    ASSERT_NE(route_record, records.end())
        << "Q4_K M=2 verifier GEMM must use the graph-native ROCm two-row native route";
    EXPECT_GE(route_record->value, 1.0);
    EXPECT_EQ(route_record->device, "rocm:0");
    EXPECT_EQ(route_record->tags.at("codebook"), "5");
    EXPECT_EQ(route_record->tags.at("n"), std::to_string(N));
    EXPECT_EQ(route_record->tags.at("k"), std::to_string(K));

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, DispatchQ5KSmallMRecordsNativeRouteCounter)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    const int M = 4;
    const int N = 512;
    const int K = 1024;
    runDispatchSmallMMatchesReference(
        "Q5_K native-VNNI small-M counter",
        M,
        N,
        K,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ5_KRandom(shape, seed); },
        0.985f);

    const auto records = PerfStatsCollector::snapshot({"kernel.rocm_native_vnni_small_m_calls"});
    auto route_record = std::find_if(
        records.begin(),
        records.end(),
        [M, N, K](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "rocm_native_vnni_small_m_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.count("m") != 0 &&
                   record.tags.at("m") == std::to_string(M) &&
                   record.tags.count("n") != 0 &&
                   record.tags.at("n") == std::to_string(N) &&
                   record.tags.count("k") != 0 &&
                   record.tags.at("k") == std::to_string(K);
        });

    ASSERT_NE(route_record, records.end())
        << "Q5_K M=4 verifier GEMM must use the graph-native ROCm small-M native route";
    EXPECT_GE(route_record->value, 1.0);
    EXPECT_EQ(route_record->device, "rocm:0");
    EXPECT_EQ(route_record->tags.at("codebook"), "7");

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedSwiGLUDownQ4KM2RecordsNativeRouteCounter)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    const int M = 2;
    const int N = 512;
    const int K = 1024;
    runFusedSwiGLUDownSmallMMatchesReference(
        "Q4_K native-VNNI fused SwiGLU down counter",
        M,
        N,
        K,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.985f);

    const auto records = PerfStatsCollector::snapshot({"kernel.rocm_native_vnni_small_m_calls"});
    auto route_record = std::find_if(
        records.begin(),
        records.end(),
        [M, N, K](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "rocm_native_vnni_small_m_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.count("source") != 0 &&
                   record.tags.at("source") == "fused_swiglu" &&
                   record.tags.count("m") != 0 &&
                   record.tags.at("m") == std::to_string(M) &&
                   record.tags.count("n") != 0 &&
                   record.tags.at("n") == std::to_string(N) &&
                   record.tags.count("k") != 0 &&
                   record.tags.at("k") == std::to_string(K);
        });

    ASSERT_NE(route_record, records.end())
        << "Q4_K M=2 fused SwiGLU down must use the graph-native ROCm small-M route";
    EXPECT_GE(route_record->value, 1.0);
    EXPECT_EQ(route_record->device, "rocm:0");
    EXPECT_EQ(route_record->tags.at("codebook"), "5");

    const auto m2_records = PerfStatsCollector::snapshot({"kernel.rocm_native_vnni_m2_calls"});
    auto m2_record = std::find_if(
        m2_records.begin(),
        m2_records.end(),
        [N, K](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "rocm_native_vnni_m2_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.count("source") != 0 &&
                   record.tags.at("source") == "fused_swiglu" &&
                   record.tags.count("n") != 0 &&
                   record.tags.at("n") == std::to_string(N) &&
                   record.tags.count("k") != 0 &&
                   record.tags.at("k") == std::to_string(K);
        });
    ASSERT_NE(m2_record, m2_records.end())
        << "Q4_K M=2 fused SwiGLU down should record the two-row native route";
    EXPECT_GE(m2_record->value, 1.0);

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedSwiGLUDownQ5KM4RecordsNativeRouteCounter)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    const int M = 4;
    const int N = 512;
    const int K = 1024;
    runFusedSwiGLUDownSmallMMatchesReference(
        "Q5_K native-VNNI fused SwiGLU down counter",
        M,
        N,
        K,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ5_KRandom(shape, seed); },
        0.985f);

    const auto records = PerfStatsCollector::snapshot({"kernel.rocm_native_vnni_small_m_calls"});
    auto route_record = std::find_if(
        records.begin(),
        records.end(),
        [M, N, K](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "rocm_native_vnni_small_m_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.count("source") != 0 &&
                   record.tags.at("source") == "fused_swiglu" &&
                   record.tags.count("m") != 0 &&
                   record.tags.at("m") == std::to_string(M) &&
                   record.tags.count("n") != 0 &&
                   record.tags.at("n") == std::to_string(N) &&
                   record.tags.count("k") != 0 &&
                   record.tags.at("k") == std::to_string(K);
        });

    ASSERT_NE(route_record, records.end())
        << "Q5_K M=4 fused SwiGLU down must use the graph-native ROCm small-M route";
    EXPECT_GE(route_record->value, 1.0);
    EXPECT_EQ(route_record->device, "rocm:0");
    EXPECT_EQ(route_record->tags.at("codebook"), "7");

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedFusedSwiGLUDownQ4KQwen36FFNDownM2MatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    constexpr int M = 2;
    constexpr int N = 5120;
    constexpr int K = 17408;
    runFusedSwiGLUDownSmallMMatchesReference(
        "Q4_K native-VNNI Qwen3.6 FFN down verifier shape",
        M,
        N,
        K,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.9999f,
        true,
        4);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    auto route_record = std::find_if(
        records.begin(),
        records.end(),
        [](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "rocm_native_vnni_small_m_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.count("source") != 0 &&
                   record.tags.at("source") == "fused_swiglu" &&
                   record.tags.count("m") != 0 &&
                   record.tags.at("m") == "2" &&
                   record.tags.count("n") != 0 &&
                   record.tags.at("n") == "5120" &&
                   record.tags.count("k") != 0 &&
                   record.tags.at("k") == "17408";
        });

    ASSERT_NE(route_record, records.end())
        << "Qwen3.6 MTP verifier FFN down must use graph-native ROCm fused SwiGLU/down small-M route";
    EXPECT_GE(route_record->value, 1.0);
    EXPECT_EQ(route_record->device, "rocm:0");
    EXPECT_EQ(route_record->tags.at("codebook"), "5");

    double graph_atomic_launches = 0.0;
    double graph_split_reduce_launches = 0.0;
    for (const auto &record : records)
    {
        if (record.domain != "kernel" ||
            record.name != "rocm_native_vnni_small_m_launch" ||
            record.kind != PerfStatRecord::Kind::Counter ||
            record.tags.count("m") == 0 ||
            record.tags.at("m") != "2" ||
            record.tags.count("n") == 0 ||
            record.tags.at("n") != "5120" ||
            record.tags.count("k") == 0 ||
            record.tags.at("k") != "17408")
        {
            continue;
        }

        if (record.tags.count("path") != 0 &&
            record.tags.at("path") == "atomic_reduce" &&
            record.tags.count("kb") != 0 &&
            std::stoi(record.tags.at("kb")) > 1)
        {
            graph_atomic_launches += record.value;
        }
        if (record.tags.count("path") != 0 &&
            record.tags.at("path") == "split_reduce")
        {
            graph_split_reduce_launches += record.value;
        }
    }

    EXPECT_EQ(graph_atomic_launches, 0.0)
        << "GPU-graph Qwen3.6 small-M FFN down should not silently force atomic K-partitioning";
    EXPECT_GE(graph_split_reduce_launches, 1.0)
        << "GPU-graph Qwen3.6 small-M FFN down should use declared workspace split/reduce by default";

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedFusedSwiGLUDownQ4KQwen36FFNDownM4MatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    constexpr int M = 4;
    constexpr int N = 5120;
    constexpr int K = 17408;
    runFusedSwiGLUDownSmallMMatchesReference(
        "Q4_K native-VNNI Qwen3.6 FFN down shifted-prefill shape",
        M,
        N,
        K,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.9999f,
        true,
        4);

    const auto records = PerfStatsCollector::snapshot({"kernel.rocm_native_vnni_small_m_calls"});
    auto route_record = std::find_if(
        records.begin(),
        records.end(),
        [](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "rocm_native_vnni_small_m_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.count("source") != 0 &&
                   record.tags.at("source") == "fused_swiglu" &&
                   record.tags.count("m") != 0 &&
                   record.tags.at("m") == "4" &&
                   record.tags.count("n") != 0 &&
                   record.tags.at("n") == "5120" &&
                   record.tags.count("k") != 0 &&
                   record.tags.at("k") == "17408";
        });

    ASSERT_NE(route_record, records.end())
        << "Qwen3.6 MTP shifted prefill FFN down must use graph-native ROCm fused SwiGLU/down small-M route";
    EXPECT_GE(route_record->value, 1.0);
    EXPECT_EQ(route_record->device, "rocm:0");
    EXPECT_EQ(route_record->tags.at("codebook"), "5");

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, ConcurrentDispatchQ4KM2MatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv concurrent_m2_rows("LLAMINAR_ROCM_CONCURRENT_M2_ROWS", "1");

    const int N = 896;
    const int K = 1024;
    runDispatchSmallMMatchesReference(
        "Q4_K native-VNNI concurrent rows",
        2,
        N,
        K,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.985f);
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedDispatchQ4KSmallMMatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

#ifdef HAVE_ROCM
    const int N = 896;
    const int K = 1024;
    for (int M : {2, 3, 4})
    {
        runGraphCapturedDispatchSmallMMatchesReference(
            "Q4_K native-VNNI",
            M,
            N,
            K,
            [](const std::vector<size_t> &shape, uint32_t seed)
            { return TestTensorFactory::createQ4_KRandom(shape, seed); },
            0.985f);
    }
#endif
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedDispatchIQ3SSmallMMatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

#ifdef HAVE_ROCM
    const int N = 512;
    const int K = 1024;
    for (int M : {3, 4})
    {
        runGraphCapturedDispatchSmallMMatchesReference(
            "IQ3_S native-VNNI",
            M,
            N,
            K,
            [](const std::vector<size_t> &shape, uint32_t seed)
            { return TestTensorFactory::createIQ3_SRandom(shape, seed); },
            0.985f);
    }
#endif
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedQ80QKVM2MatchesSeparate)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const int K = 896;
    runFusedQKVSmallMMatchesSeparate(
        "Q8_0 INT8-VNNI",
        2,
        K,
        PackedPath::INT8VNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ8_0Random(shape, seed); },
        0.9999f);
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedQ4KQKVM2MatchesSeparate)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const int K = 1024;
    runFusedQKVSmallMMatchesSeparate(
        "Q4_K native-VNNI",
        2,
        K,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.9999f);
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedQ5KQKVSmallMMatchesSeparate)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const int K = 1024;
    for (int M : {3, 4})
    {
        runFusedQKVSmallMMatchesSeparate(
            "Q5_K native-VNNI",
            M,
            K,
            PackedPath::NativeVNNI,
            [](const std::vector<size_t> &shape, uint32_t seed)
            { return TestTensorFactory::createQ5_KRandom(shape, seed); },
            0.9999f);
    }
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedQ5KQKVSmallMRecordsSharedQuantizedNativeRoute)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    const int M = 4;
    const int K = 1024;
    runFusedQKVSmallMMatchesSeparate(
        "Q5_K native-VNNI shared small-M quant counter",
        M,
        K,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ5_KRandom(shape, seed); },
        0.9999f);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    auto shared_quant_record = std::find_if(
        records.begin(),
        records.end(),
        [M, K](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "rocm_fused_small_m_shared_quant_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.count("m") != 0 &&
                   record.tags.at("m") == std::to_string(M) &&
                   record.tags.count("k") != 0 &&
                   record.tags.at("k") == std::to_string(K) &&
                   record.tags.count("projections") != 0 &&
                   record.tags.at("projections") == "3";
        });

    ASSERT_NE(shared_quant_record, records.end())
        << "Fused Q5_K M=4 QKV must quantize activations once before projection dispatch";
    EXPECT_GE(shared_quant_record->value, 1.0);
    EXPECT_EQ(shared_quant_record->device, "rocm:0");

    double batched_calls = 0.0;
    double batched_projection_calls = 0.0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == std::to_string(M) &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == std::to_string(K) &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "3")
        {
            batched_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_projection_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == std::to_string(M) &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == std::to_string(K))
        {
            batched_projection_calls += record.value;
        }
    }
    if (batched_calls < 1.0 || batched_projection_calls < 3.0)
    {
        for (const auto &record : records)
        {
            if (record.domain != "kernel" ||
                (record.name.find("rocm_native_vnni") == std::string::npos &&
                 record.name.find("rocm_fused") == std::string::npos))
            {
                continue;
            }
            std::string tags;
            for (const auto &tag : record.tags)
            {
                if (!tags.empty())
                    tags += ",";
                tags += tag.first + "=" + tag.second;
            }
            LOG_INFO("[SmallM] observed counter name=" << record.name
                     << " value=" << record.value
                     << " device=" << record.device
                     << " tags=" << tags);
        }
    }
    EXPECT_GE(batched_calls, 1.0)
        << "Fused QKV M=4 should use one graph-native batched native route";
    EXPECT_GE(batched_projection_calls, 3.0)
        << "Fused QKV M=4 batched route should cover Q, K, and V projections";

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedQ4KQKVM2RecordsSharedQuantizedNativeRoute)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    const int K = 1024;
    runFusedQKVSmallMMatchesSeparate(
        "Q4_K native-VNNI shared quant counter",
        2,
        K,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.9999f);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    auto shared_quant_record = std::find_if(
        records.begin(),
        records.end(),
        [K](const PerfStatRecord &record)
        {
            return record.domain == "kernel" &&
                   record.name == "rocm_fused_m2_shared_quant_calls" &&
                   record.kind == PerfStatRecord::Kind::Counter &&
                   record.tags.count("k") != 0 &&
                   record.tags.at("k") == std::to_string(K) &&
                   record.tags.count("projections") != 0 &&
                   record.tags.at("projections") == "3";
        });

    ASSERT_NE(shared_quant_record, records.end())
        << "Fused Q4_K M=2 QKV must quantize activations once before projection dispatch";
    EXPECT_GE(shared_quant_record->value, 1.0);
    EXPECT_EQ(shared_quant_record->device, "rocm:0");

    double batched_calls = 0.0;
    double batched_projection_calls = 0.0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == std::to_string(K) &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "3")
        {
            batched_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_projection_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == std::to_string(K))
        {
            batched_projection_calls += record.value;
        }
    }
    EXPECT_GE(batched_calls, 1.0)
        << "Fused QKV M=2 should use one graph-native batched native route";
    EXPECT_GE(batched_projection_calls, 3.0)
        << "Fused QKV M=2 batched route should cover Q, K, and V projections";

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedQ4KQwen36QKVM2MatchesSeparate)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    runFusedQKVSmallMMatchesSeparate(
        "Q4_K native-VNNI Qwen3.6 QKV shape",
        2,
        5120,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.9999f,
        5120,
        1024,
        1024);
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedFusedQ4KQwen36FFNGateUpM2MatchesSeparate)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    runFusedProjectionGroupSmallMMatchesSeparate(
        "Q4_K native-VNNI Qwen3.6 FFN gate/up MTP verifier group",
        2,
        5120,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.9999f,
        {17408, 17408},
        true,
        true,
        2);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    double batched_calls = 0.0;
    double batched_projection_calls = 0.0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "2")
        {
            batched_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_projection_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120")
        {
            batched_projection_calls += record.value;
        }
    }

    EXPECT_GE(batched_calls, 1.0)
        << "Graph-captured Qwen3.6 FFN gate/up should use one batched native route";
    EXPECT_GE(batched_projection_calls, 2.0)
        << "Batched FFN gate/up route should cover both projection payloads";

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedFusedQ5KQwen36FFNGateUpM2MatchesSeparate)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    runFusedProjectionGroupSmallMMatchesSeparate(
        "Q5_K native-VNNI Qwen3.6 FFN gate/up MTP verifier group",
        2,
        5120,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ5_KRandom(shape, seed); },
        0.999f,
        {17408, 17408},
        true,
        true,
        2);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    double batched_calls = 0.0;
    double batched_projection_calls = 0.0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "2")
        {
            batched_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_projection_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120")
        {
            batched_projection_calls += record.value;
        }
    }

    EXPECT_GE(batched_calls, 1.0)
        << "Graph-captured Qwen3.6 Q5_K FFN gate/up should use one batched native route";
    EXPECT_GE(batched_projection_calls, 2.0)
        << "Batched Q5_K FFN gate/up route should cover both projection payloads";

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedQ4KQwen36FFNGateUpM2UsesCanonicalBatchedSplitKWorkspace)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    constexpr int M = 2;
    constexpr int K = 5120;
    constexpr int N = 17408;

    std::vector<std::unique_ptr<TensorBase>> weights;
    std::vector<ROCmPackedWeights> packed(2);
    std::vector<std::unique_ptr<ROCmQuantisedGemmKernel>> kernels;
    WorkspaceRequirements combined;

    for (int i = 0; i < 2; ++i)
    {
        weights.push_back(TestTensorFactory::createQ4_KRandom(
            {static_cast<size_t>(N), static_cast<size_t>(K)},
            static_cast<uint32_t>(501 + i)));
        ASSERT_TRUE(packWeightsToROCm(weights.back().get(), packed[i]));
        expectPackedPath(packed[i], PackedPath::NativeVNNI);
        kernels.push_back(std::make_unique<ROCmQuantisedGemmKernel>(&packed[i], 0));
        combined.merge(kernels.back()->getWorkspaceRequirements(M, N, K));
    }

    int single_partial_buffers = 0;
    int batched_partial_buffers = 0;
    int old_slice_named_buffers = 0;
    const std::string old_partial_prefix =
        std::string(GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL) + "_";
    for (const auto &buf : combined.buffers)
    {
        if (buf.name == GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL)
            ++single_partial_buffers;
        if (buf.name == GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED)
            ++batched_partial_buffers;
        if (buf.name != GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED &&
            buf.name.rfind(old_partial_prefix, 0) == 0)
            ++old_slice_named_buffers;
    }
    EXPECT_EQ(single_partial_buffers, 1)
        << "ROCm split-K single-projection scratch should be canonical across graph instances";
    EXPECT_EQ(batched_partial_buffers, 1)
        << "ROCm split-K GEMV-many scratch should be a single canonical arena";
    EXPECT_EQ(old_slice_named_buffers, 0)
        << "Per-kernel scatter partial names cause decode graph workspace reallocations";

    DeviceWorkspaceManager workspace(
        DeviceId::rocm(0),
        combined.total_bytes_with_alignment() + 64 * 1024 * 1024);
    ASSERT_TRUE(workspace.allocate(combined));
    for (auto &kernel : kernels)
        kernel->bindWorkspace(&workspace);

    auto input = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(M), static_cast<size_t>(K)});
    auto gate = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
    auto up = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
    ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
    ASSERT_TRUE(gate->allocateOnDevice(DeviceId::rocm(0)));
    ASSERT_TRUE(up->allocateOnDevice(DeviceId::rocm(0)));

    std::vector<ITensorGemm::TensorProjectionDesc> projections;
    projections.emplace_back(kernels[0].get(), gate.get(), N, nullptr, "gate");
    projections.emplace_back(kernels[1].get(), up.get(), N, nullptr, "up");

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    EXPECT_TRUE(kernels.front()->multiply_fused_tensor(input.get(), projections, M, K))
        << "Canonical batched split-K arena must still provide distinct per-projection partial slices";
#ifdef HAVE_ROCM
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
#endif

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    double batched_calls = 0.0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "2")
        {
            batched_calls += record.value;
        }
    }
    EXPECT_GE(batched_calls, 1.0)
        << "Canonical batched split-K arena should keep the graph-native GEMV-many route active";
    PerfStatsCollector::reset();

    for (auto &kernel : kernels)
        kernel->unbindWorkspace();
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedQ4KQwen36FFNGateUpM2RejectsUndersizedDeclaredPartialWorkspace)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    constexpr int M = 2;
    constexpr int K = 5120;
    constexpr int N = 17408;

    std::vector<std::unique_ptr<TensorBase>> weights;
    std::vector<ROCmPackedWeights> packed(2);
    std::vector<std::unique_ptr<ROCmQuantisedGemmKernel>> kernels;
    WorkspaceRequirements combined;

    for (int i = 0; i < 2; ++i)
    {
        weights.push_back(TestTensorFactory::createQ4_KRandom(
            {static_cast<size_t>(N), static_cast<size_t>(K)},
            static_cast<uint32_t>(601 + i)));
        ASSERT_TRUE(packWeightsToROCm(weights.back().get(), packed[i]));
        expectPackedPath(packed[i], PackedPath::NativeVNNI);
        kernels.push_back(std::make_unique<ROCmQuantisedGemmKernel>(&packed[i], 0));
        combined.merge(kernels.back()->getWorkspaceRequirements(M, N, K));
    }

    bool shrunk_partial = false;
    for (auto &buf : combined.buffers)
    {
        if (!shrunk_partial && buf.name == GemmWorkspaceBuffers::ROCM_SCATTER_PARTIAL_BATCHED)
        {
            buf.size_bytes = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);
            shrunk_partial = true;
        }
    }
    ASSERT_TRUE(shrunk_partial);

    DeviceWorkspaceManager workspace(
        DeviceId::rocm(0),
        combined.total_bytes_with_alignment() + 64 * 1024 * 1024);
    ASSERT_TRUE(workspace.allocate(combined));
    for (auto &kernel : kernels)
        kernel->bindWorkspace(&workspace);

    auto input = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(M), static_cast<size_t>(K)});
    ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));

    auto gate = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
    auto up = TestTensorFactory::createFP32({static_cast<size_t>(M), static_cast<size_t>(N)});
    ASSERT_TRUE(gate->allocateOnDevice(DeviceId::rocm(0)));
    ASSERT_TRUE(up->allocateOnDevice(DeviceId::rocm(0)));

    std::vector<ITensorGemm::TensorProjectionDesc> projections;
    projections.emplace_back(kernels[0].get(), gate.get(), N, nullptr, "gate");
    projections.emplace_back(kernels[1].get(), up.get(), N, nullptr, "up");

    EXPECT_FALSE(kernels.front()->multiply_fused_tensor(input.get(), projections, M, K))
        << "Batched small-M split-K must reject undersized declared partial workspace before HIP launch";
#ifdef HAVE_ROCM
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
#endif

    for (auto &kernel : kernels)
        kernel->unbindWorkspace();
}

TEST(Test__ROCmQuantisedGemmSmallM, DispatchQ4KQwen36GDNAlphaM1MatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    runDispatchSmallMMatchesReference(
        "Q4_K native-VNNI Qwen3.6 GDN alpha M=1",
        1,
        48,
        5120,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.985f);
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedFusedQ4KQwen36GDNDecodeM1MatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    runFusedProjectionGroupSmallMMatchesReference(
        "Q4_K native-VNNI Qwen3.6 dense GDN decode projection group",
        1,
        5120,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.985f,
        {10240, 6144, 48, 48},
        {"qkv", "z", "alpha", "beta"},
        true,
        true,
        4);
}

TEST(Test__ROCmQuantisedGemmSmallM, Qwen36GDNProjectionStageMixedQuantizedAndRawFPMatchesReference)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

#ifndef HAVE_ROCM
    GTEST_SKIP() << "HAVE_ROCM not enabled";
#else
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    constexpr int M = 1;
    constexpr int K = 5120;
    constexpr int N_QKV = 10240;
    constexpr int N_Z = 6144;
    constexpr int N_ALPHA = 48;
    constexpr int N_BETA = 48;

    auto input = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(M), static_cast<size_t>(K)},
        -0.5f,
        0.5f,
        13801);
    auto w_qkv = TestTensorFactory::createQ5_KRandom(
        {static_cast<size_t>(N_QKV), static_cast<size_t>(K)}, 13802);
    auto w_z = TestTensorFactory::createQ4_KRandom(
        {static_cast<size_t>(N_Z), static_cast<size_t>(K)}, 13803);
    auto w_a = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(N_ALPHA), static_cast<size_t>(K)},
        -0.1f,
        0.1f,
        13804);
    auto w_b = TestTensorFactory::createFP32Random(
        {static_cast<size_t>(N_BETA), static_cast<size_t>(K)},
        -0.1f,
        0.1f,
        13805);

    auto out_qkv = TestTensorFactory::createFP32Zeros(
        {static_cast<size_t>(M), static_cast<size_t>(N_QKV)});
    auto out_z = TestTensorFactory::createFP32Zeros(
        {static_cast<size_t>(M), static_cast<size_t>(N_Z)});
    auto out_a = TestTensorFactory::createFP32Zeros(
        {static_cast<size_t>(M), static_cast<size_t>(N_ALPHA)});
    auto out_b = TestTensorFactory::createFP32Zeros(
        {static_cast<size_t>(M), static_cast<size_t>(N_BETA)});

    ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));
    ASSERT_TRUE(out_qkv->allocateOnDevice(DeviceId::rocm(0)));
    ASSERT_TRUE(out_z->allocateOnDevice(DeviceId::rocm(0)));
    ASSERT_TRUE(out_a->allocateOnDevice(DeviceId::rocm(0)));
    ASSERT_TRUE(out_b->allocateOnDevice(DeviceId::rocm(0)));

    auto qkv_prepared = makeGpuPreparedGemm(
        w_qkv.get(), DeviceId::rocm(0), "blk.0.attn_qkv.weight", ModelContextId{1388});
    auto z_prepared = makeGpuPreparedGemm(
        w_z.get(), DeviceId::rocm(0), "blk.0.attn_gate.weight", ModelContextId{1388});
    auto a_prepared = makeGpuPreparedFloatingPointGemm(
        w_a.get(), DeviceId::rocm(0), "blk.0.ssm_alpha.weight", ModelContextId{1388});
    auto b_prepared = makeGpuPreparedFloatingPointGemm(
        w_b.get(), DeviceId::rocm(0), "blk.0.ssm_beta.weight", ModelContextId{1388});

    GDNProjectionStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.input = input.get();
    params.m = M;
    params.k = K;
    params.w_qkv = w_qkv.get();
    params.output_qkv = out_qkv.get();
    params.n_qkv = N_QKV;
    params.w_z = w_z.get();
    params.output_z = out_z.get();
    params.n_z = N_Z;
    params.w_a = w_a.get();
    params.output_a = out_a.get();
    params.n_a = N_ALPHA;
    params.w_b = w_b.get();
    params.output_b = out_b.get();
    params.n_b = N_BETA;
    params.gemm_qkv = qkv_prepared.kernel;
    params.gemm_z = z_prepared.kernel;
    params.gemm_a = a_prepared.kernel;
    params.gemm_b = b_prepared.kernel;

    GDNProjectionStage stage(params);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
    stage.setGPUStream(stream);

    DeviceWorkspaceManager workspace(DeviceId::rocm(0), 256 * 1024 * 1024);
    ASSERT_TRUE(workspace.allocate(stage.getWorkspaceRequirements(M, 0, K)));
    stage.bindWorkspace(&workspace);

    ROCmDeviceContext ctx(DeviceId::rocm(0), 0);
    ASSERT_TRUE(stage.execute(&ctx));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    out_a->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    out_b->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    std::vector<float> ref_a(static_cast<size_t>(M) * N_ALPHA);
    std::vector<float> ref_b(static_cast<size_t>(M) * N_BETA);
    cpuFP32GemmRef(input->data(), w_a->data(), ref_a.data(), M, N_ALPHA, K);
    cpuFP32GemmRef(input->data(), w_b->data(), ref_b.data(), M, N_BETA, K);

    std::vector<float> actual_a(out_a->data(), out_a->data() + ref_a.size());
    std::vector<float> actual_b(out_b->data(), out_b->data() + ref_b.size());
    expectNearFP32(actual_a, ref_a, 1e-3f, "alpha");
    expectNearFP32(actual_b, ref_b, 1e-3f, "beta");

    stage.unbindWorkspace();
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedFusedQ4KGDNProjectionM2MatchesSeparate)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    runFusedProjectionGroupSmallMMatchesSeparate(
        "Q4_K native-VNNI Qwen3.6 GDN projection group",
        2,
        5120,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.9999f,
        {10240, 10240, 1024, 1024},
        true,
        true,
        4);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    double batched_calls = 0.0;
    double batched_projection_calls = 0.0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "4")
        {
            batched_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_projection_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120")
        {
            batched_projection_calls += record.value;
        }
    }

    EXPECT_GE(batched_calls, 1.0)
        << "Graph-captured Qwen3.6 GDN projection group should use one batched native route";
    EXPECT_GE(batched_projection_calls, 4.0)
        << "Batched GDN route should cover qkv/z/alpha/beta projection payloads";

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedFusedQ4KGDNProjectionM4MatchesSeparate)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    runFusedProjectionGroupSmallMMatchesSeparate(
        "Q4_K native-VNNI Qwen3.6 GDN projection shifted-prefill group",
        4,
        5120,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.9999f,
        {10240, 10240, 1024, 1024},
        true,
        true,
        4);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    double batched_calls = 0.0;
    double batched_projection_calls = 0.0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "4" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "4")
        {
            batched_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_projection_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "4" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120")
        {
            batched_projection_calls += record.value;
        }
    }

    EXPECT_GE(batched_calls, 1.0)
        << "Graph-captured Qwen3.6 M=4 GDN projection group should use one batched native route";
    EXPECT_GE(batched_projection_calls, 4.0)
        << "Batched M=4 GDN route should cover qkv/z/alpha/beta projection payloads";

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedFusedQ4KQwen36GDNQkvZPairM2UsesHeterogeneousNBatchedRoute)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    runFusedProjectionGroupSmallMMatchesSeparate(
        "Q4_K native-VNNI Qwen3.6 GDN qkv/z heterogeneous-N pair",
        2,
        5120,
        PackedPath::NativeVNNI,
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        0.9999f,
        {10240, 6144},
        true,
        true,
        4);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    double batched_calls = 0.0;
    double heterogeneous_n_bypasses = 0.0;
    double shared_quant_calls = 0.0;
    double batched_projection_calls = 0.0;
    double graph_atomic_launches = 0.0;
    double graph_split_reduce_launches = 0.0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "2")
        {
            batched_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_bypasses" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("reason") != 0 &&
            record.tags.at("reason") == "heterogeneous_n_pair" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "2")
        {
            heterogeneous_n_bypasses += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_fused_small_m_shared_quant_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "2")
        {
            shared_quant_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_projection_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120")
        {
            batched_projection_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_launch" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "2" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120" &&
            record.tags.count("batched") != 0 &&
            record.tags.at("batched") == "true" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "2")
        {
            if (record.tags.count("path") != 0 &&
                record.tags.at("path") == "atomic_reduce" &&
                record.tags.count("kb") != 0 &&
                std::stoi(record.tags.at("kb")) > 1)
            {
                graph_atomic_launches += record.value;
            }
            if (record.tags.count("path") != 0 &&
                record.tags.at("path") == "split_reduce")
            {
                graph_split_reduce_launches += record.value;
            }
        }
    }

    EXPECT_GE(batched_calls, 1.0)
        << "Real Qwen3.6 GDN qkv/z should use the graph-captured heterogeneous-N batched route";
    EXPECT_EQ(heterogeneous_n_bypasses, 0.0)
        << "The heterogeneous-N qkv/z shape should be handled by the generic batched kernel, not bypassed";
    EXPECT_GE(shared_quant_calls, 1.0)
        << "The batched qkv/z subgroup should quantize activations once";
    EXPECT_GE(batched_projection_calls, 2.0)
        << "The batched qkv/z subgroup should cover both heterogeneous-N projection payloads";
    EXPECT_EQ(graph_atomic_launches, 0.0)
        << "GPU-graph Qwen3.6 batched GDN qkv/z should not silently force atomic K-partitioning";
    EXPECT_GE(graph_split_reduce_launches, 1.0)
        << "GPU-graph Qwen3.6 batched GDN qkv/z should use declared workspace split/reduce by default";

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedFusedMixedCodebookGDNProjectionM4UsesMixedBatchedRoute)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    std::vector<WeightCreator> creators = {
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); },
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ5_KRandom(shape, seed); },
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ5_KRandom(shape, seed); }};

    runMixedProjectionGroupSmallMMatchesSeparate(
        "mixed Q4_K/Q5_K native-VNNI Qwen3.6 GDN projection group",
        4,
        5120,
        creators,
        0.9999f,
        {10240, 10240, 1024, 1024},
        true,
        4);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    double mixed_batched_calls = 0.0;
    double full_mixed_bypasses = 0.0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "4" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "4" &&
            record.tags.count("codebook") != 0 &&
            record.tags.at("codebook") == "mixed")
        {
            mixed_batched_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_mixed_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "4" &&
            record.tags.count("k") != 0 &&
            record.tags.at("k") == "5120" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "4")
        {
            mixed_batched_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_bypasses" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("reason") != 0 &&
            record.tags.at("reason") == "mixed_codebook" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "4")
        {
            full_mixed_bypasses += record.value;
        }
    }

    EXPECT_GE(mixed_batched_calls, 1.0)
        << "Mixed-codebook GDN groups should use the graph-capturable mixed native small-M batched route";
    EXPECT_EQ(full_mixed_bypasses, 0.0)
        << "Mixed-codebook GDN groups should no longer bypass to per-projection small-M GEMV";

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedMixedCodebookQwen36GDNQkvZPairM4MatchesSeparate)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    std::vector<WeightCreator> creators = {
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ5_KRandom(shape, seed); },
        [](const std::vector<size_t> &shape, uint32_t seed)
        { return TestTensorFactory::createQ4_KRandom(shape, seed); }};

    runMixedProjectionGroupSmallMMatchesSeparate(
        "mixed Q5_K/Q4_K native-VNNI Qwen3.6 GDN qkv/z heterogeneous-N pair",
        4,
        5120,
        creators,
        0.9999f,
        {10240, 6144},
        true,
        4);

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    double mixed_bypasses = 0.0;
    double batched_calls = 0.0;
    double mixed_batched_calls = 0.0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_bypasses" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("reason") != 0 &&
            record.tags.at("reason") == "mixed_codebook" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "2" &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "4")
        {
            mixed_bypasses += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "4" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "2" &&
            record.tags.count("codebook") != 0 &&
            record.tags.at("codebook") == "mixed")
        {
            batched_calls += record.value;
        }
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_batched_mixed_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == "4" &&
            record.tags.count("projections") != 0 &&
            record.tags.at("projections") == "2")
        {
            mixed_batched_calls += record.value;
        }
    }

    EXPECT_EQ(mixed_bypasses, 0.0)
        << "Mixed-codebook Qwen3.6 qkv/z M=4 should no longer bypass the batched route";
    EXPECT_GE(batched_calls, 1.0)
        << "Mixed-codebook Qwen3.6 qkv/z M=4 must enter the mixed batched route";
    EXPECT_GE(mixed_batched_calls, 1.0)
        << "Mixed-codebook Qwen3.6 qkv/z M=4 must record the mixed route counter";

    PerfStatsCollector::reset();
}

TEST(Test__ROCmQuantisedGemmSmallM, FusedQ4KGDNProjectionM2RejectsUnsafeKBOverride)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv force_unsafe_kb("LLAMINAR_ROCM_NVNNI_GEMV_KB", "32");

    constexpr int M = 2;
    constexpr int K = 5120;
    const std::vector<int> Ns = {10240, 10240, 1024, 1024};

    std::vector<std::unique_ptr<TensorBase>> weights;
    std::vector<ROCmPackedWeights> packed(Ns.size());
    std::vector<std::unique_ptr<ROCmQuantisedGemmKernel>> kernels;
    weights.reserve(Ns.size());
    kernels.reserve(Ns.size());

    WorkspaceRequirements combined;
    for (size_t i = 0; i < Ns.size(); ++i)
    {
        weights.push_back(TestTensorFactory::createQ4_KRandom(
            {static_cast<size_t>(Ns[i]), static_cast<size_t>(K)},
            static_cast<uint32_t>(300 + i)));
        ASSERT_TRUE(packWeightsToROCm(weights.back().get(), packed[i]));
        expectPackedPath(packed[i], PackedPath::NativeVNNI);
        kernels.push_back(std::make_unique<ROCmQuantisedGemmKernel>(&packed[i], 0));
        combined.merge(kernels.back()->getWorkspaceRequirements(M, Ns[i], K));
    }

    DeviceWorkspaceManager workspace(
        DeviceId::rocm(0),
        combined.total_bytes_with_alignment() + 64 * 1024 * 1024);
    ASSERT_TRUE(workspace.allocate(combined));
    for (auto &kernel : kernels)
        kernel->bindWorkspace(&workspace);

    auto input = TestTensorFactory::createFP32Random({M, K});
    ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));

    std::vector<std::unique_ptr<FP32Tensor>> outputs;
    std::vector<ITensorGemm::TensorProjectionDesc> projections;
    outputs.reserve(Ns.size());
    projections.reserve(Ns.size());
    for (size_t i = 0; i < Ns.size(); ++i)
    {
        outputs.push_back(TestTensorFactory::createFP32({M, static_cast<size_t>(Ns[i])}));
        ASSERT_TRUE(outputs.back()->allocateOnDevice(DeviceId::rocm(0)));
        projections.emplace_back(kernels[i].get(),
                                 outputs.back().get(),
                                 Ns[i],
                                 nullptr,
                                 "gdn_projection");
    }

    EXPECT_FALSE(kernels.front()->multiply_fused_tensor(input.get(), projections, M, K))
        << "Unsafe small-M split-K overrides must hard-fail before launching graph-captured kernels";

    for (auto &kernel : kernels)
        kernel->unbindWorkspace();
}

TEST(Test__ROCmQuantisedGemmSmallM, GraphCapturedSingleProjectionQ4KFFNDownM2HonorsAtomicKBOverride)
{
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    ScopedEnv force_kb("LLAMINAR_ROCM_NVNNI_GEMV_KB", "2");
    ScopedEnv force_graphs("LLAMINAR_GPU_GRAPHS", "1");
    ScopedEnv force_atomic("LLAMINAR_ROCM_NVNNI_ATOMIC_REDUCE", "1");
    ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();

    constexpr int M = 2;
    constexpr int K = 17408;
    constexpr int N = 5120;

    auto weights = TestTensorFactory::createQ4_KRandom({N, K}, 909);
    ROCmPackedWeights packed;
    ASSERT_TRUE(packWeightsToROCm(weights.get(), packed));
    expectPackedPath(packed, PackedPath::NativeVNNI);

    ROCmQuantisedGemmKernel kernel(&packed, 0);
    auto workspace = bindWorkspace(kernel, M, N, K);
    ASSERT_NE(workspace, nullptr);

    auto input = TestTensorFactory::createFP32Random({M, K});
    ASSERT_TRUE(input->ensureOnDevice(DeviceId::rocm(0)));

    auto output = TestTensorFactory::createFP32({M, N});
    ASSERT_TRUE(output->allocateOnDevice(DeviceId::rocm(0)));

    EXPECT_TRUE(kernel.multiply_tensor(input.get(), output.get(), M, N, K))
        << "Explicit ROCm atomic-reduce with forced KB=2 should use atomic K-partitioning";

#ifdef HAVE_ROCM
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
#endif

    const auto records = PerfStatsCollector::snapshot({"kernel"});
    double atomic_launches = 0.0;
    double split_reduce_launches = 0.0;
    for (const auto &record : records)
    {
        if (record.domain != "kernel" ||
            record.name != "rocm_native_vnni_small_m_launch" ||
            record.kind != PerfStatRecord::Kind::Counter ||
            record.tags.count("m") == 0 ||
            record.tags.at("m") != "2" ||
            record.tags.count("n") == 0 ||
            record.tags.at("n") != "5120" ||
            record.tags.count("k") == 0 ||
            record.tags.at("k") != "17408")
        {
            continue;
        }
        if (record.tags.count("path") != 0 &&
            record.tags.at("path") == "atomic_reduce" &&
            record.tags.count("kb") != 0 &&
            record.tags.at("kb") == "2")
        {
            atomic_launches += record.value;
        }
        if (record.tags.count("path") != 0 &&
            record.tags.at("path") == "split_reduce")
        {
            split_reduce_launches += record.value;
        }
    }

    EXPECT_GE(atomic_launches, 1.0)
        << "Explicit ROCm atomic-reduce KB override should be honored by the atomic route";
    EXPECT_EQ(split_reduce_launches, 0.0)
        << "Explicit ROCm atomic-reduce KB override must not re-enable split/reduce replay";

    PerfStatsCollector::reset();
    kernel.unbindWorkspace();
}
