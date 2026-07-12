/**
 * @file Perf__CUDANativeVNNIDecodeTrainer.cpp
 * @brief Strong CUDA NativeVNNI public-M1 and grouped-verifier policy trainer.
 *
 * This trainer drives the production GPU-prepared GEMM object rather than a raw
 * test-only kernel. Every candidate is forced explicitly and proven through
 * PerfStats route telemetry. Fast M1 candidates are compared with the frozen
 * public-M1 route for numerical diagnostics. Grouped verifier runtime-M execution
 * is compared byte-for-byte with ordinary public-M1 decode of each row and is
 * repeated to expose first-call or graph-lifetime instability.
 *
 * Timing uses an explicit non-default CUDA stream. Eager and graph-captured
 * launches are separate evidence surfaces, and every exact CUDA-event sample
 * is retained in a sidecar CSV. Serial row replay exists only inside this
 * performance oracle; production execution remains one grouped launch.
 */

#include <gtest/gtest.h>

#ifdef HAVE_CUDA

#include <cuda_runtime.h>

#include "backends/DeviceId.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/KernelFactory.h"
#include "utils/PerfStatsCollector.h"
#include "../../../../utils/GpuPreparedGemmHarness.h"
#include "../../../../utils/NativeVNNITrainerEvidence.h"
#include "../../../../utils/TestTensorFactory.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;

extern "C"
{
    void cudaNativeVNNIGemvSweep_setConfig(
        int kernel_family,
        int tile_n,
        int cpt,
        int target_waves,
        int mkg,
        int max_kb,
        int exact_kb,
        int force_two_phase);
    void cudaNativeVNNIGemvSweep_clearConfig();
}

namespace
{
    /** Default promotion-strength sample counts. */
    constexpr int kDefaultWarmups = 5;
    constexpr int kDefaultSamples = 30;

    /** One externally loadable source format and deterministic tensor factory. */
    struct FormatSpec
    {
        std::string name;
        std::function<std::unique_ptr<TensorBase>(size_t, size_t)> create;
    };

    /**
     * @brief Build structurally valid deterministic quantized source weights.
     *
     * The candidate comparison needs production packing and codebook behavior,
     * not expensive random host generation. A small block-dependent payload
     * keeps the tensor nonuniform while retaining O(raw bytes) setup cost for
     * large LM-head shapes.
     */
    template <typename TensorT, typename BlockT>
    std::unique_ptr<TensorBase> createTrainerTensor(size_t n, size_t k)
    {
        const size_t blocks_per_row =
            (k + BlockT::BLOCK_SIZE - 1) / BlockT::BLOCK_SIZE;
        std::vector<uint8_t> raw(n * blocks_per_row * sizeof(BlockT), 0);
        constexpr uint16_t kHalfOne = 0x3c00;
        for (size_t block = 0; block < n * blocks_per_row; ++block)
        {
            uint8_t *destination = raw.data() + block * sizeof(BlockT);
            std::memcpy(
                destination,
                &kHalfOne,
                std::min(sizeof(kHalfOne), sizeof(BlockT)));
            for (size_t byte = sizeof(kHalfOne); byte < sizeof(BlockT); ++byte)
            {
                destination[byte] = static_cast<uint8_t>(
                    ((block * 17u) + (byte * 13u)) & 0x3u);
            }
        }
        return std::make_unique<TensorT>(
            std::vector<size_t>{n, k}, raw);
    }

    const std::vector<FormatSpec> kFormats = {
        {"Q4_0", [](size_t n, size_t k)
         { return createTrainerTensor<Q4_0Tensor, Q4_0Block>(n, k); }},
        {"IQ4_NL", [](size_t n, size_t k)
         { return createTrainerTensor<IQ4_NLTensor, IQ4_NLBlock>(n, k); }},
        {"IQ4_XS", [](size_t n, size_t k)
         { return createTrainerTensor<IQ4_XSTensor, IQ4_XSBlock>(n, k); }},
        {"Q4_1", [](size_t n, size_t k)
         { return createTrainerTensor<Q4_1Tensor, Q4_1Block>(n, k); }},
        {"Q4_K", [](size_t n, size_t k)
         { return createTrainerTensor<Q4_KTensor, Q4_KBlock>(n, k); }},
        {"Q5_0", [](size_t n, size_t k)
         { return createTrainerTensor<Q5_0Tensor, Q5_0Block>(n, k); }},
        {"Q5_1", [](size_t n, size_t k)
         { return createTrainerTensor<Q5_1Tensor, Q5_1Block>(n, k); }},
        {"Q5_K", [](size_t n, size_t k)
         { return createTrainerTensor<Q5_KTensor, Q5_KBlock>(n, k); }},
        {"Q6_K", [](size_t n, size_t k)
         { return createTrainerTensor<Q6_KTensor, Q6_KBlock>(n, k); }},
        {"Q3_K", [](size_t n, size_t k)
         { return createTrainerTensor<Q3_KTensor, Q3_KBlock>(n, k); }},
        {"Q2_K", [](size_t n, size_t k)
         { return createTrainerTensor<Q2_KTensor, Q2_KBlock>(n, k); }},
        {"IQ3_S", [](size_t n, size_t k)
         { return createTrainerTensor<IQ3_STensor, IQ3_SBlock>(n, k); }},
        {"IQ3_XXS", [](size_t n, size_t k)
         { return createTrainerTensor<IQ3_XXSTensor, IQ3_XXSBlock>(n, k); }},
        {"IQ2_S", [](size_t n, size_t k)
         { return createTrainerTensor<IQ2_STensor, IQ2_SBlock>(n, k); }},
        {"IQ2_XS", [](size_t n, size_t k)
         { return createTrainerTensor<IQ2_XSTensor, IQ2_XSBlock>(n, k); }},
        {"IQ2_XXS", [](size_t n, size_t k)
         { return createTrainerTensor<IQ2_XXSTensor, IQ2_XXSBlock>(n, k); }},
        {"IQ1_S", [](size_t n, size_t k)
         { return createTrainerTensor<IQ1_STensor, IQ1_SBlock>(n, k); }},
        {"IQ1_M", [](size_t n, size_t k)
         { return createTrainerTensor<IQ1_MTensor, IQ1_MBlock>(n, k); }},
        {"Q8_0", [](size_t n, size_t k)
         { return createTrainerTensor<Q8_0Tensor, Q8_0Block>(n, k); }},
        {"Q8_1", [](size_t n, size_t k)
         { return createTrainerTensor<Q8_1Tensor, Q8_1Block>(n, k); }},
        {"Q8_K", [](size_t n, size_t k)
         { return createTrainerTensor<Q8_KTensor, Q8_KBlock>(n, k); }},
    };

    /** One production decode projection shape. */
    struct Shape
    {
        std::string name;
        int n = 0;
        int k = 0;
    };

    const std::vector<Shape> kProductionShapes = {
        {"Qwen36_Attn_QKVProjection", 12288, 5120},
        {"Qwen36_FFN_GateUp", 17408, 5120},
        {"Qwen36_FFN_DownProjection", 5120, 17408},
        {"Qwen36_GDN_InnerProjection", 10240, 5120},
        {"Qwen36_GDN_ZProjection", 6144, 5120},
        {"Qwen36_GDN_TimeProjection", 1024, 5120},
        {"Qwen36_GDN_OutputProjection", 5120, 6144},
        {"Qwen36_LM_Head", 248320, 5120},
        {"35BMoE_Expert_GateUp", 512, 2048},
        {"35BMoE_Expert_Down", 2048, 512},
        {"Qwen36MoE_GDN_QKVProjection", 8192, 2048},
        {"Qwen36MoE_GDN_ZProjection", 4096, 2048},
    };

    /** Production execution surface represented by one corpus alias. */
    enum class ExecutionMode
    {
        Eager,
        GraphCaptured,
    };

    const char *executionModeName(ExecutionMode mode)
    {
        return mode == ExecutionMode::Eager ? "eager" : "graph_captured";
    }

    /** NativeVNNI launch family understood by the production sweep override. */
    enum class CandidateFamily
    {
        Wide = 0,
        KPar = 1,
        Direct = 2,
        InheritSerialM1 = 4,
    };

    /** One normalized forceable candidate from the common registry. */
    struct Candidate
    {
        std::string id;
        CandidateFamily family = CandidateFamily::KPar;
        int tile_n = 0;
        int cpt = 0;
        int target_waves = 0;
        int mkg = 0;
        int max_kb = 0;
        int force_two_phase = 0;
        int exact_kb = 0;

        [[nodiscard]] bool inheritsSerialM1() const
        {
            return family == CandidateFamily::InheritSerialM1;
        }

        [[nodiscard]] std::string familyName() const
        {
            switch (family)
            {
            case CandidateFamily::Wide:
                return "wide";
            case CandidateFamily::KPar:
                return "kpar";
            case CandidateFamily::Direct:
                return "direct";
            case CandidateFamily::InheritSerialM1:
                return "inherit_serial_m1";
            }
            return "unknown";
        }
    };

    /** Parsed trainer breadth and output locations. */
    struct TrainerConfig
    {
        int warmups = kDefaultWarmups;
        int samples = kDefaultSamples;
        int max_cases = std::numeric_limits<int>::max();
        std::set<std::string> formats;
        std::set<std::string> shapes;
        std::set<std::string> candidate_families;
        std::set<std::string> candidate_ids;
        std::vector<int> m_values = {
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16};
        std::vector<ExecutionMode> execution_modes = {
            ExecutionMode::Eager,
            ExecutionMode::GraphCaptured,
        };
        std::string aggregate_path = "/tmp/llaminar_cuda_decode_strong.csv";
        std::string timing_path = "/tmp/llaminar_cuda_decode_strong.timing.csv";
    };

    std::string trim(std::string value)
    {
        const size_t begin = value.find_first_not_of(" \t\r\n");
        if (begin == std::string::npos)
            return {};
        const size_t end = value.find_last_not_of(" \t\r\n");
        return value.substr(begin, end - begin + 1);
    }

    std::string lower(std::string value)
    {
        std::transform(
            value.begin(), value.end(), value.begin(),
            [](unsigned char c)
            { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    std::string envString(const char *name)
    {
        const char *value = std::getenv(name);
        return value ? trim(value) : std::string{};
    }

    int envPositiveInt(const char *name, int fallback)
    {
        const std::string value = envString(name);
        if (value.empty())
            return fallback;
        const int parsed = std::atoi(value.c_str());
        if (parsed <= 0)
            throw std::runtime_error(std::string(name) + " must be positive");
        return parsed;
    }

    std::set<std::string> envCsvSet(const char *name)
    {
        std::set<std::string> result;
        std::stringstream stream(envString(name));
        std::string token;
        while (std::getline(stream, token, ','))
        {
            token = lower(trim(token));
            if (!token.empty())
                result.insert(token);
        }
        return result;
    }

    std::vector<int> envCsvInts(const char *name, std::vector<int> fallback)
    {
        const std::string raw = envString(name);
        if (raw.empty())
            return fallback;
        fallback.clear();
        std::stringstream stream(raw);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            const int value = std::atoi(trim(token).c_str());
            if (value < 1)
                throw std::runtime_error(std::string(name) + " requires M>=1");
            fallback.push_back(value);
        }
        return fallback;
    }

    TrainerConfig loadTrainerConfig()
    {
        TrainerConfig config;
        config.warmups = envPositiveInt(
            "LLAMINAR_CUDA_NVNNI_DECODE_WARMUPS", kDefaultWarmups);
        config.samples = envPositiveInt(
            "LLAMINAR_CUDA_NVNNI_DECODE_SAMPLES", kDefaultSamples);
        config.max_cases = envPositiveInt(
            "LLAMINAR_CUDA_NVNNI_DECODE_MAX_CASES",
            std::numeric_limits<int>::max());
        config.formats = envCsvSet("LLAMINAR_CUDA_NVNNI_DECODE_FORMATS");
        config.shapes = envCsvSet("LLAMINAR_CUDA_NVNNI_DECODE_SHAPES");
        config.candidate_families = envCsvSet(
            "LLAMINAR_CUDA_NVNNI_DECODE_FAMILIES");
        config.candidate_ids = envCsvSet(
            "LLAMINAR_CUDA_NVNNI_DECODE_CANDIDATES");
        config.m_values = envCsvInts(
            "LLAMINAR_CUDA_NVNNI_DECODE_M", config.m_values);

        const auto execution_modes = envCsvSet(
            "LLAMINAR_CUDA_NVNNI_DECODE_EXECUTION_MODES");
        if (!execution_modes.empty())
        {
            config.execution_modes.clear();
            for (const std::string &mode : execution_modes)
            {
                if (mode == "eager")
                    config.execution_modes.push_back(ExecutionMode::Eager);
                else if (mode == "graph" || mode == "graph_captured")
                    config.execution_modes.push_back(ExecutionMode::GraphCaptured);
                else
                    throw std::runtime_error("unknown CUDA execution mode " + mode);
            }
        }

        const std::string aggregate = envString(
            "LLAMINAR_CUDA_NVNNI_DECODE_CSV");
        const std::string timing = envString(
            "LLAMINAR_CUDA_NVNNI_DECODE_TIMING_CSV");
        if (!aggregate.empty())
            config.aggregate_path = aggregate;
        if (!timing.empty())
            config.timing_path = timing;
        return config;
    }

    bool selected(const std::set<std::string> &filter, const std::string &name)
    {
        return filter.empty() || filter.count(lower(name)) != 0;
    }

    std::vector<Candidate> fastM1Candidates(const TrainerConfig &config)
    {
        std::vector<Candidate> result;
        const auto add = [&](Candidate candidate)
        {
            if (!selected(config.candidate_families, candidate.familyName()) ||
                !selected(config.candidate_ids, candidate.id))
            {
                return;
            }
            result.push_back(std::move(candidate));
        };
        for (const auto [tile_n, cpt] : std::array{
                 std::pair{128, 1}, std::pair{128, 2}, std::pair{256, 2},
                 std::pair{256, 4}, std::pair{512, 4}})
        {
            add(Candidate{
                "cuda.nvnni.decode.fast_m1.wide.tn" +
                    std::to_string(tile_n) + ".cpt" + std::to_string(cpt),
                CandidateFamily::Wide, tile_n, cpt});
        }
        for (const auto [tile_n, cpt] : std::array{
                 std::pair{128, 1}, std::pair{64, 1}, std::pair{32, 1}})
        {
            add(Candidate{
                "cuda.nvnni.decode.fast_m1.direct.tn" +
                    std::to_string(tile_n) + ".cpt" + std::to_string(cpt),
                CandidateFamily::Direct, tile_n, cpt});
        }
        const auto add_kpar = [&](int tile_n, int cpt, int kb)
        {
            add(Candidate{
                "cuda.nvnni.decode.fast_m1.kpar.tn" +
                    std::to_string(tile_n) + ".cpt" + std::to_string(cpt) +
                    ".kb" + std::to_string(kb),
                CandidateFamily::KPar,
                tile_n,
                cpt,
                0,
                0,
                0,
                1,
                kb});
        };
        for (const auto [tile_n, cpt] : std::array{
                 std::pair{128, 1}, std::pair{128, 2}, std::pair{256, 2},
                 std::pair{256, 4}, std::pair{64, 1}, std::pair{64, 2},
                 std::pair{32, 1}})
        {
            for (int kb = 1; kb <= 64; ++kb)
                add_kpar(tile_n, cpt, kb);
            for (int kb : std::array{72, 80, 88, 96, 104, 112,
                                     120, 128, 144, 160, 192, 256})
            {
                add_kpar(tile_n, cpt, kb);
            }
        }
        return result;
    }

    std::vector<Candidate> candidatesForM(int m, const TrainerConfig &config)
    {
        if (m == 1)
            return fastM1Candidates(config);
        Candidate verifier{
            "cuda.nvnni.decode.verifier.inherit_serial_m1",
            CandidateFamily::InheritSerialM1};
        if (!selected(config.candidate_families, verifier.familyName()) ||
            !selected(config.candidate_ids, verifier.id))
        {
            return {};
        }
        return {verifier};
    }

    /** Set and restore PerfStats collection for trainer route proofs. */
    class ScopedPerfStatsEnvironment
    {
    public:
        ScopedPerfStatsEnvironment()
        {
            const char *previous = std::getenv("LLAMINAR_PERF_STATS_JSON");
            if (previous)
            {
                had_previous_ = true;
                previous_ = previous;
            }
            (void)setenv("LLAMINAR_PERF_STATS_JSON", "1", 1);
        }

        ~ScopedPerfStatsEnvironment()
        {
            if (had_previous_)
                (void)setenv("LLAMINAR_PERF_STATS_JSON", previous_.c_str(), 1);
            else
                (void)unsetenv("LLAMINAR_PERF_STATS_JSON");
        }

        ScopedPerfStatsEnvironment(const ScopedPerfStatsEnvironment &) = delete;
        ScopedPerfStatsEnvironment &operator=(
            const ScopedPerfStatsEnvironment &) = delete;

    private:
        bool had_previous_ = false;
        std::string previous_;
    };

    /** Scope one explicit M1 candidate around production dispatch. */
    class CandidateOverride
    {
    public:
        explicit CandidateOverride(const Candidate &candidate)
            : active_(!candidate.inheritsSerialM1())
        {
            if (active_)
            {
                cudaNativeVNNIGemvSweep_setConfig(
                    static_cast<int>(candidate.family),
                    candidate.tile_n,
                    candidate.cpt,
                    candidate.target_waves,
                    candidate.mkg,
                    candidate.max_kb,
                    candidate.exact_kb,
                    candidate.force_two_phase);
            }
            else
            {
                cudaNativeVNNIGemvSweep_clearConfig();
            }
        }

        ~CandidateOverride()
        {
            cudaNativeVNNIGemvSweep_clearConfig();
        }

        CandidateOverride(const CandidateOverride &) = delete;
        CandidateOverride &operator=(const CandidateOverride &) = delete;

    private:
        bool active_ = false;
    };

    /** Own a captured production launch and executable graph. */
    class CapturedLaunch
    {
    public:
        ~CapturedLaunch()
        {
            reset();
        }

        template <typename Launch>
        bool capture(cudaStream_t stream, Launch &&launch)
        {
            reset();
            if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) !=
                cudaSuccess)
            {
                return false;
            }
            const bool launch_ok = launch();
            const cudaError_t end_status = cudaStreamEndCapture(stream, &graph_);
            if (!launch_ok || end_status != cudaSuccess || graph_ == nullptr)
            {
                reset();
                return false;
            }
            if (cudaGraphInstantiate(&executable_, graph_, 0) != cudaSuccess ||
                executable_ == nullptr)
            {
                reset();
                return false;
            }
            return true;
        }

        bool launch(cudaStream_t stream) const
        {
            return executable_ != nullptr &&
                   cudaGraphLaunch(executable_, stream) == cudaSuccess;
        }

        void reset()
        {
            if (executable_)
                (void)cudaGraphExecDestroy(executable_);
            if (graph_)
                (void)cudaGraphDestroy(graph_);
            executable_ = nullptr;
            graph_ = nullptr;
        }

    private:
        cudaGraph_t graph_ = nullptr;
        cudaGraphExec_t executable_ = nullptr;
    };

    size_t workspaceBudgetFor(const WorkspaceRequirements &requirements)
    {
        constexpr size_t kMinimumBudget = 64ull * 1024ull * 1024ull;
        return std::max(
            kMinimumBudget,
            requirements.total_bytes_with_alignment() +
                static_cast<size_t>(4) * 1024 * 1024);
    }

    /** Route fields emitted by the production NativeVNNI dispatcher. */
    struct RouteEvidence
    {
        bool valid = false;
        uint64_t count = 0;
        std::string candidate_id = "missing";
        std::string path = "missing";
        int tile_n = 0;
        int cpt = 0;
        int effective_kb = 0;
    };

    RouteEvidence findRoute(
        int m,
        int n,
        int k,
        uint8_t codebook,
        const char *semantic_contract)
    {
        RouteEvidence result;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"kernel.cuda_native_vnni_gemv_dispatch"}))
        {
            if (record.domain != "kernel" ||
                record.name != "cuda_native_vnni_gemv_dispatch" ||
                record.kind != PerfStatRecord::Kind::Counter)
            {
                continue;
            }
            const auto tag = [&](const char *name) -> std::string
            {
                const auto iterator = record.tags.find(name);
                return iterator == record.tags.end()
                           ? std::string{}
                           : iterator->second;
            };
            if (tag("m") != std::to_string(m) ||
                tag("n") != std::to_string(n) ||
                tag("k") != std::to_string(k) ||
                tag("codebook") != std::to_string(static_cast<unsigned>(codebook)) ||
                tag("semantic_contract") != semantic_contract)
            {
                continue;
            }
            result.valid = true;
            result.count += record.count;
            result.candidate_id = tag("effective_candidate_id");
            result.path = tag("route");
            result.tile_n = std::stoi(tag("tile_n"));
            result.cpt = std::stoi(tag("cpt"));
            result.effective_kb = std::stoi(tag("effective_kb"));
        }
        return result;
    }

    bool copyDeviceOutput(
        TensorBase *output,
        size_t count,
        cudaStream_t stream,
        std::vector<float> &host)
    {
        if (!output || !output->gpu_data_ptr() || stream == nullptr)
            return false;
        host.resize(count);
        return cudaMemcpyAsync(
                   host.data(),
                   output->gpu_data_ptr(),
                   count * sizeof(float),
                   cudaMemcpyDeviceToHost,
                   stream) == cudaSuccess &&
               cudaStreamSynchronize(stream) == cudaSuccess;
    }

    /** Public-M1 row oracle and the generated route that produced it. */
    struct SerialEvidence
    {
        bool valid = false;
        std::string failure;
        std::vector<float> output;
        RouteEvidence route;
        std::string digest;
    };

    SerialEvidence runSerialRows(
        ITensorGemm *kernel,
        const TensorBase *grouped_input,
        int m,
        int n,
        int k,
        uint8_t codebook,
        DeviceId device)
    {
        SerialEvidence result;
        auto fail = [&](const char *reason)
        {
            result.failure = reason;
            return result;
        };
        if (!kernel || !grouped_input || m < 1)
            return fail("invalid_arguments");

        cudaNativeVNNIGemvSweep_clearConfig();
        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
            return fail("stream_create");
        kernel->setGPUStream(stream);
        auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
        std::unique_ptr<DeviceWorkspaceManager> workspace;
        if (workspace_consumer)
        {
            const auto requirements = workspace_consumer->getWorkspaceRequirements(
                std::max(1, m), n, k);
            workspace = std::make_unique<DeviceWorkspaceManager>(
                device, workspaceBudgetFor(requirements));
            if (!workspace->allocate(requirements))
            {
                kernel->setGPUStream(nullptr);
                (void)cudaStreamDestroy(stream);
                return fail("workspace_allocate");
            }
            workspace_consumer->bindWorkspace(workspace.get());
        }
        const auto cleanup = [&]
        {
            if (workspace_consumer)
                workspace_consumer->unbindWorkspace();
            kernel->setGPUStream(nullptr);
            (void)cudaStreamDestroy(stream);
        };

        result.output.resize(static_cast<size_t>(m) * static_cast<size_t>(n));
        PerfStatsCollector::reset();
        const float *source = static_cast<const float *>(grouped_input->data());
        for (int row = 0; row < m; ++row)
        {
            auto row_input = TestTensorFactory::createFP32(
                {1u, static_cast<size_t>(k)});
            auto row_output = TestTensorFactory::createFP32(
                {1u, static_cast<size_t>(n)});
            std::memcpy(
                row_input->mutable_data(),
                source + static_cast<size_t>(row) * static_cast<size_t>(k),
                static_cast<size_t>(k) * sizeof(float));
            if (!row_input->ensureOnDevice(device) ||
                !row_output->allocateOnDevice(device) ||
                !kernel->multiply_tensor(row_input.get(), row_output.get(), 1, n, k) ||
                cudaStreamSynchronize(stream) != cudaSuccess)
            {
                cleanup();
                return fail("serial_row_launch");
            }
            std::vector<float> row_host;
            if (!copyDeviceOutput(
                    row_output.get(), static_cast<size_t>(n), stream, row_host))
            {
                cleanup();
                return fail("serial_row_download");
            }
            std::copy(
                row_host.begin(),
                row_host.end(),
                result.output.begin() +
                    static_cast<size_t>(row) * static_cast<size_t>(n));
        }
        result.route = findRoute(1, n, k, codebook, "fast");
        result.valid = result.route.valid &&
                       result.route.count >= static_cast<uint64_t>(m);
        if (!result.valid)
            result.failure = "serial_route_proof";
        result.digest = trainer::nativeByteDigest(result.output);
        cleanup();
        return result;
    }

    /** Complete strong evidence for one candidate and execution surface. */
    struct CandidateEvidence
    {
        bool valid = false;
        std::string failure;
        bool graph_capture_ok = false;
        bool workspace_ok = false;
        bool explicit_stream_ok = false;
        RouteEvidence route;
        trainer::FP32Evidence comparison;
        size_t repeat_byte_mismatches = 0;
        bool numerical_correctness = false;
        trainer::TimingEvidence timing;
        std::vector<double> samples;
        double effective_bandwidth_gbs = 0.0;
    };

    CandidateEvidence runCandidate(
        ITensorGemm *kernel,
        TensorBase *input,
        const SerialEvidence &serial,
        const Candidate &candidate,
        ExecutionMode mode,
        int m,
        int n,
        int k,
        uint8_t codebook,
        size_t weight_bytes,
        int warmups,
        int sample_count,
        DeviceId device)
    {
        CandidateEvidence result;
        auto fail = [&](const char *reason)
        {
            result.failure = reason;
            return result;
        };
        if (!kernel || !input || !serial.valid || sample_count <= 0)
            return fail("invalid_arguments");

        CandidateOverride override(candidate);
        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
            return fail("stream_create");
        result.explicit_stream_ok = stream != nullptr;
        kernel->setGPUStream(stream);
        auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
        std::unique_ptr<DeviceWorkspaceManager> workspace;
        if (workspace_consumer)
        {
            const auto requirements = workspace_consumer->getWorkspaceRequirements(
                m, n, k);
            workspace = std::make_unique<DeviceWorkspaceManager>(
                device, workspaceBudgetFor(requirements));
            if (!workspace->allocate(requirements))
            {
                kernel->setGPUStream(nullptr);
                (void)cudaStreamDestroy(stream);
                return fail("workspace_allocate");
            }
            workspace_consumer->bindWorkspace(workspace.get());
        }
        result.workspace_ok = true;

        auto output = TestTensorFactory::createFP32(
            {static_cast<size_t>(m), static_cast<size_t>(n)});
        if (!input->ensureOnDevice(device) || !output->allocateOnDevice(device))
        {
            if (workspace_consumer)
                workspace_consumer->unbindWorkspace();
            kernel->setGPUStream(nullptr);
            (void)cudaStreamDestroy(stream);
            return fail("tensor_prepare");
        }

        const auto run_once = [&]() -> bool
        {
            if (m >= 2)
            {
                std::vector<ITensorGemm::TensorProjectionDesc> projections = {
                    {kernel, output.get(), n, nullptr, "trainer_verifier"}};
                return kernel->multiply_fused_verifier_rows_decode_equivalent(
                    input,
                    projections,
                    m,
                    k,
                    nullptr,
                    workspace.get());
            }
            return kernel->multiply_tensor(input, output.get(), m, n, k);
        };
        CapturedLaunch captured;
        if (mode == ExecutionMode::GraphCaptured)
        {
            // Prime lazy, non-captured host setup before starting capture.
            if (!run_once() || cudaStreamSynchronize(stream) != cudaSuccess)
            {
                if (workspace_consumer)
                    workspace_consumer->unbindWorkspace();
                kernel->setGPUStream(nullptr);
                (void)cudaStreamDestroy(stream);
                return fail("graph_primer");
            }
            PerfStatsCollector::reset();
            result.graph_capture_ok = captured.capture(stream, run_once);
            if (!result.graph_capture_ok)
            {
                if (workspace_consumer)
                    workspace_consumer->unbindWorkspace();
                kernel->setGPUStream(nullptr);
                (void)cudaStreamDestroy(stream);
                return fail("graph_capture");
            }
        }
        else
        {
            result.graph_capture_ok = true;
            PerfStatsCollector::reset();
        }
        const auto execute_once = [&]() -> bool
        {
            return mode == ExecutionMode::GraphCaptured
                       ? captured.launch(stream)
                       : run_once();
        };
        const auto cleanup = [&]
        {
            captured.reset();
            if (workspace_consumer)
                workspace_consumer->unbindWorkspace();
            kernel->setGPUStream(nullptr);
            (void)cudaStreamDestroy(stream);
        };

        if (!execute_once() || cudaStreamSynchronize(stream) != cudaSuccess)
        {
            cleanup();
            return fail("candidate_launch");
        }
        result.route = findRoute(
            m,
            n,
            k,
            codebook,
            m == 1 ? "fast" : "verifier_serial_m1_bitwise");
        std::vector<float> first_output;
        if (!copyDeviceOutput(
                output.get(),
                static_cast<size_t>(m) * static_cast<size_t>(n),
                stream,
                first_output) ||
            !execute_once() ||
            cudaStreamSynchronize(stream) != cudaSuccess)
        {
            cleanup();
            return fail("repeat_launch");
        }
        std::vector<float> repeated_output;
        if (!copyDeviceOutput(
                output.get(),
                static_cast<size_t>(m) * static_cast<size_t>(n),
                stream,
                repeated_output))
        {
            cleanup();
            return fail("repeat_download");
        }
        result.repeat_byte_mismatches = trainer::nativeByteMismatchCount(
            first_output, repeated_output);
        result.comparison = trainer::compareFP32(
            repeated_output, serial.output, static_cast<size_t>(n));
        result.numerical_correctness =
            result.comparison.nonfinite_count == 0 &&
            result.comparison.cosine >= 0.999;

        for (int warmup = 0; warmup < warmups; ++warmup)
        {
            if (!execute_once())
            {
                cleanup();
                return fail("warmup_launch");
            }
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess)
        {
            cleanup();
            return fail("warmup_sync");
        }

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        if (cudaEventCreate(&start) != cudaSuccess ||
            cudaEventCreate(&stop) != cudaSuccess)
        {
            if (start)
                (void)cudaEventDestroy(start);
            if (stop)
                (void)cudaEventDestroy(stop);
            cleanup();
            return fail("event_create");
        }
        result.samples.reserve(static_cast<size_t>(sample_count));
        for (int sample = 0; sample < sample_count; ++sample)
        {
            if (cudaEventRecord(start, stream) != cudaSuccess ||
                !execute_once() ||
                cudaEventRecord(stop, stream) != cudaSuccess ||
                cudaEventSynchronize(stop) != cudaSuccess)
            {
                (void)cudaEventDestroy(start);
                (void)cudaEventDestroy(stop);
                cleanup();
                return fail("timed_launch");
            }
            float elapsed_ms = 0.0f;
            if (cudaEventElapsedTime(&elapsed_ms, start, stop) != cudaSuccess)
            {
                (void)cudaEventDestroy(start);
                (void)cudaEventDestroy(stop);
                cleanup();
                return fail("timed_read");
            }
            result.samples.push_back(static_cast<double>(elapsed_ms) * 1000.0);
        }
        (void)cudaEventDestroy(start);
        (void)cudaEventDestroy(stop);
        std::sort(result.samples.begin(), result.samples.end());
        result.timing = trainer::summarizeSortedTimingSamples(result.samples);
        if (result.timing.median > 0.0)
        {
            result.effective_bandwidth_gbs =
                static_cast<double>(weight_bytes) /
                (result.timing.median * 1.0e-6) / 1.0e9;
        }
        result.valid = result.route.valid;
        if (!result.valid)
            result.failure = "route_proof";
        cleanup();
        return result;
    }

    /** Aggregate row retained until the winner bit is known. */
    struct MeasuredRow
    {
        const FormatSpec *format = nullptr;
        const Shape *shape = nullptr;
        Candidate candidate;
        ExecutionMode mode = ExecutionMode::Eager;
        int m = 0;
        uint8_t source_codebook = 0;
        uint8_t execution_codebook = 0;
        size_t weight_bytes = 0;
        SerialEvidence serial;
        CandidateEvidence candidate_evidence;
        bool correctness_pass = false;
        bool is_winner = false;
    };

    void writeTimingRows(
        std::FILE *file,
        const MeasuredRow &row)
    {
        for (size_t index = 0;
             index < row.candidate_evidence.samples.size();
             ++index)
        {
            const double latency = row.candidate_evidence.samples[index];
            std::fprintf(
                file,
                "cuda,decode,%s,%u,%u,%s,%s,%d,%d,%d,%s,%zu,1,%.9f,%a\n",
                row.format->name.c_str(),
                static_cast<unsigned>(row.source_codebook),
                static_cast<unsigned>(row.execution_codebook),
                row.shape->name.c_str(),
                executionModeName(row.mode),
                row.m,
                row.shape->n,
                row.shape->k,
                row.candidate.id.c_str(),
                index,
                latency,
                latency);
        }
    }

    void writeAggregateRow(
        std::FILE *file,
        const MeasuredRow &row,
        int warmups)
    {
        const auto &evidence = row.candidate_evidence;
        const auto &comparison = evidence.comparison;
        std::fprintf(
            file,
            "cuda,decode,%s,%u,%u,%s,%s,%d,%d,%d,%s,%s,%d,%d,%d,%d,%d,%d,%d,%zu,%d,%zu,"
            "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%zu,%zu,%zu,%.17g,%.17g,%.17g,%.17g,%s,%s,%s,"
            "1,%d,1,1,%d,%s,%s,%d,%d,%d,%s,1,%d,%d,%d\n",
            row.format->name.c_str(),
            static_cast<unsigned>(row.source_codebook),
            static_cast<unsigned>(row.execution_codebook),
            row.shape->name.c_str(),
            executionModeName(row.mode),
            row.m,
            row.shape->n,
            row.shape->k,
            row.candidate.id.c_str(),
            row.candidate.familyName().c_str(),
            row.candidate.tile_n,
            row.candidate.cpt,
            row.candidate.target_waves,
            row.candidate.mkg,
            row.candidate.max_kb,
            row.candidate.exact_kb,
            row.candidate.force_two_phase,
            row.weight_bytes,
            warmups,
            evidence.samples.size(),
            evidence.timing.min,
            evidence.timing.median,
            evidence.timing.p95,
            evidence.timing.mad,
            evidence.timing.cv,
            evidence.effective_bandwidth_gbs,
            comparison.mismatch_count,
            comparison.first_mismatch_index,
            evidence.repeat_byte_mismatches,
            comparison.max_abs,
            comparison.relative_l2,
            comparison.cosine,
            comparison.symmetric_kld,
            comparison.actual_digest.c_str(),
            comparison.expected_digest.c_str(),
            evidence.timing.digest.c_str(),
            evidence.graph_capture_ok ? 1 : 0,
            evidence.route.valid ? 1 : 0,
            evidence.route.candidate_id.c_str(),
            evidence.route.path.c_str(),
            evidence.route.tile_n,
            evidence.route.cpt,
            evidence.route.effective_kb,
            row.serial.route.candidate_id.c_str(),
            evidence.numerical_correctness ? 1 : 0,
            row.correctness_pass ? 1 : 0,
            row.is_winner ? 1 : 0);
    }

    /** CUDA-only performance fixture. */
    class CUDANativeVNNIDecodeTrainer : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0)
                GTEST_SKIP() << "No CUDA device available";
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
            device_ = DeviceId::cuda(0);
        }

        DeviceId device_ = DeviceId::cpu();
    };

    TEST_F(CUDANativeVNNIDecodeTrainer, TrainerCsv_StrongDecode_AllFormats)
    {
        ScopedPerfStatsEnvironment perf_stats;
        const TrainerConfig config = loadTrainerConfig();
        std::filesystem::create_directories(
            std::filesystem::path(config.aggregate_path).parent_path());
        std::filesystem::create_directories(
            std::filesystem::path(config.timing_path).parent_path());

        std::FILE *aggregate = std::fopen(config.aggregate_path.c_str(), "w");
        std::FILE *timing = std::fopen(config.timing_path.c_str(), "w");
        ASSERT_NE(aggregate, nullptr) << config.aggregate_path;
        ASSERT_NE(timing, nullptr) << config.timing_path;
        std::fprintf(
            aggregate,
            "backend,phase,source_format,source_codebook,execution_codebook,shape,execution_mode,m,n,k,"
            "candidate_id,family,tile_n,cpt,target_waves,mkg,max_kb,exact_kb,force_two_phase,weight_bytes,warmup_count,"
            "sample_count,min_us,median_us,p95_us,mad_us,cv,effective_bandwidth_gbs,bit_mismatches,"
            "first_bit_mismatch,repeat_byte_mismatches,max_abs,relative_l2,cosine,symmetric_kld,"
            "grouped_output_digest,serial_output_digest,timing_sample_digest,supported,graph_capture_ok,"
            "workspace_ok,explicit_stream_ok,route_counter_ok,observed_candidate_id,observed_path,"
            "observed_tile_n,observed_cpt,observed_effective_kb,serial_m1_candidate_id,serial_route_counter_ok,"
            "numerical_correctness,correctness_pass,is_winner\n");
        std::fprintf(
            timing,
            "backend,phase,source_format,source_codebook,execution_codebook,shape,execution_mode,m,n,k,"
            "candidate_id,sample_index,timed_replays,latency_us,latency_us_hex\n");

        int executed_cases = 0;
        for (const FormatSpec &format : kFormats)
        {
            if (!selected(config.formats, format.name))
                continue;
            for (const Shape &shape : kProductionShapes)
            {
                if (!selected(config.shapes, shape.name))
                    continue;
                if (executed_cases >= config.max_cases)
                    break;

                auto weights = format.create(
                    static_cast<size_t>(shape.n),
                    static_cast<size_t>(shape.k));
                ASSERT_NE(weights, nullptr) << format.name;
                const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(
                    weights.get());
                ASSERT_NE(unpackable, nullptr) << format.name;
                const NativeVnniFormatInfo *source_info = unpackable->vnniFormatInfo();
                ASSERT_NE(source_info, nullptr) << format.name;

                auto prepared = makeGpuPreparedGemm(
                    weights.get(),
                    device_,
                    "perf.cuda_native_vnni_decode_strong.weight");
                ASSERT_NE(prepared.kernel, nullptr) << format.name;
                DeviceNativeVNNIMatrixDesc matrix_desc;
                ASSERT_TRUE(prepared.kernel->exportNativeVNNIMatrixDesc(matrix_desc))
                    << format.name;

                for (const int m : config.m_values)
                {
                    const std::vector<Candidate> candidates = candidatesForM(
                        m, config);
                    ASSERT_FALSE(candidates.empty())
                        << "No CUDA candidates selected for M=" << m;
                    auto input = TestTensorFactory::createFP32Random(
                        {static_cast<size_t>(m), static_cast<size_t>(shape.k)},
                        -0.25f,
                        0.25f,
                        0xC0DAu + static_cast<uint32_t>(m));
                    const SerialEvidence serial = runSerialRows(
                        prepared.kernel,
                        input.get(),
                        m,
                        shape.n,
                        shape.k,
                        matrix_desc.codebook_id,
                        device_);
                    ASSERT_TRUE(serial.valid)
                        << format.name << " " << shape.name << " M=" << m
                        << " serial failure=" << serial.failure;

                    std::vector<MeasuredRow> rows;
                    for (const ExecutionMode mode : config.execution_modes)
                    {
                        for (const Candidate &candidate : candidates)
                        {
                            CandidateEvidence evidence = runCandidate(
                                prepared.kernel,
                                input.get(),
                                serial,
                                candidate,
                                mode,
                                m,
                                shape.n,
                                shape.k,
                                matrix_desc.codebook_id,
                                weights->size_bytes(),
                                config.warmups,
                                config.samples,
                                device_);
                            ASSERT_TRUE(evidence.valid)
                                << format.name << " " << shape.name << " M=" << m
                                << " candidate=" << candidate.id
                                << " mode=" << executionModeName(mode)
                                << " failure=" << evidence.failure;
                            const bool route_matches =
                                evidence.route.candidate_id == candidate.id;
                            const bool exact_required = m >= 2;
                            const bool correctness_pass =
                                route_matches &&
                                evidence.repeat_byte_mismatches == 0 &&
                                evidence.numerical_correctness &&
                                (!exact_required ||
                                 evidence.comparison.mismatch_count == 0);
                            rows.push_back(MeasuredRow{
                                &format,
                                &shape,
                                candidate,
                                mode,
                                m,
                                source_info->codebook_id,
                                matrix_desc.codebook_id,
                                weights->size_bytes(),
                                serial,
                                std::move(evidence),
                                correctness_pass,
                                false});
                        }
                    }

                    for (const ExecutionMode mode : config.execution_modes)
                    {
                        auto winner = rows.end();
                        for (auto iterator = rows.begin(); iterator != rows.end(); ++iterator)
                        {
                            if (iterator->mode != mode || !iterator->correctness_pass)
                                continue;
                            if (winner == rows.end() ||
                                iterator->candidate_evidence.timing.median <
                                    winner->candidate_evidence.timing.median)
                            {
                                winner = iterator;
                            }
                        }
                        ASSERT_NE(winner, rows.end())
                            << format.name << " " << shape.name << " M=" << m
                            << " mode=" << executionModeName(mode)
                            << " has no correct candidate";
                        winner->is_winner = true;
                    }

                    for (const MeasuredRow &row : rows)
                    {
                        writeTimingRows(timing, row);
                        writeAggregateRow(aggregate, row, config.warmups);
                        std::fprintf(
                            stderr,
                            "[CUDANativeVNNI][STRONG] format=%s source_cb=%u exec_cb=%u "
                            "shape=%s mode=%s M=%d candidate=%s median_us=%.3f "
                            "bit_mismatches=%zu repeat_byte_mismatches=%zu pass=%d winner=%d\n",
                            format.name.c_str(),
                            static_cast<unsigned>(row.source_codebook),
                            static_cast<unsigned>(row.execution_codebook),
                            shape.name.c_str(),
                            executionModeName(row.mode),
                            m,
                            row.candidate.id.c_str(),
                            row.candidate_evidence.timing.median,
                            row.candidate_evidence.comparison.mismatch_count,
                            row.candidate_evidence.repeat_byte_mismatches,
                            row.correctness_pass ? 1 : 0,
                            row.is_winner ? 1 : 0);
                    }
                    std::fflush(aggregate);
                    std::fflush(timing);
                    ++executed_cases;
                }
            }
            if (executed_cases >= config.max_cases)
                break;
        }
        std::fclose(aggregate);
        std::fclose(timing);
        ASSERT_GT(executed_cases, 0) << "CUDA strong trainer selected no cases";
    }
} // namespace

#else

TEST(CUDANativeVNNIDecodeTrainer, RequiresCUDA)
{
    GTEST_SKIP() << "CUDA backend not enabled";
}

#endif
