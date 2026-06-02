/**
 * @file Test__ROCmQuantisedGemmSmallM.cpp
 * @brief Focused ROCm small-M GEMM regressions for MTP verifier decode.
 */

#include <gtest/gtest.h>

#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"
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

    template <typename CreateWeights>
    void runFusedQKVSmallMMatchesSeparate(
        const char *label,
        int M,
        int K,
        PackedPath expected_path,
        CreateWeights createWeights,
        float min_cosine)
    {
        const int Nq = 896;
        const int Nk = 128;
        const int Nv = 128;

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

    double shared_native_projection_calls = 0.0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_small_m_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("m") != 0 &&
            record.tags.at("m") == std::to_string(M) &&
            record.tags.count("shared_quant") != 0 &&
            record.tags.at("shared_quant") == "true")
        {
            shared_native_projection_calls += record.value;
        }
    }
    if (shared_native_projection_calls < 3.0)
    {
        for (const auto &record : records)
        {
            if (record.domain != "kernel" ||
                record.name.find("rocm_native_vnni") == std::string::npos)
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
    EXPECT_GE(shared_native_projection_calls, 3.0)
        << "Fused QKV M=4 should dispatch Q, K, and V through the shared-quant native route";

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

    double shared_native_projection_calls = 0.0;
    for (const auto &record : records)
    {
        if (record.domain == "kernel" &&
            record.name == "rocm_native_vnni_m2_calls" &&
            record.kind == PerfStatRecord::Kind::Counter &&
            record.tags.count("shared_quant") != 0 &&
            record.tags.at("shared_quant") == "true")
        {
            shared_native_projection_calls += record.value;
        }
    }
    EXPECT_GE(shared_native_projection_calls, 3.0)
        << "Fused QKV M=2 should dispatch Q, K, and V through the shared-quant native route";

    PerfStatsCollector::reset();
}
