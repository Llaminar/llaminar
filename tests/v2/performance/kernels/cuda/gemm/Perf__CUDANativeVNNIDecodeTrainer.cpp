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
 * launches are separate evidence surfaces. Each CUDA event brackets several
 * identical launches so sub-20-microsecond kernels are not ranked by event
 * quantization; the sidecar records both the replay count and exact per-launch
 * sample. A separate paired-confirmation mode interleaves one frozen policy
 * candidate with its exact-reference candidate, preserving pair identity for
 * simultaneous confidence-bound analysis. Serial row replay exists only inside
 * this performance oracle; production execution remains one grouped launch.
 */

#include <gtest/gtest.h>

#ifdef HAVE_CUDA

#include <cuda_runtime.h>

#include "backends/DeviceId.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/cuda/gemm/CUDADeviceWorkspace.h"
#include "kernels/KernelFactory.h"
#include "utils/PerfStatsCollector.h"
#include "../../../../utils/GpuPreparedGemmHarness.h"
#include "../../../../utils/NativeVNNITrainerEvidence.h"
#include "../../../../utils/TestTensorFactory.h"
#include "../../native_vnni_dispatch/NativeVNNIPairedRequestManifest.h"
#include "../../native_vnni_dispatch/NativeVNNIProfilerControl.h"
#include "../../native_vnni_dispatch/NativeVNNIShapeManifest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
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

    /**
     * @brief Default maximum launches inside one CUDA event sample.
     *
     * CUDA events have finite timing resolution. A single event around a tiny
     * decode kernel can therefore report a quantized value whose apparent
     * variation is larger than the candidate differences being learned. The
     * trainer amortizes that resolution over an adaptively selected number of
     * launches up to this cap and stores the normalized per-launch latency.
     * This changes measurement precision only; every launch still runs the
     * exact candidate and production graph route.
     */
    constexpr int kDefaultTimedReplayCap = 16;

    /**
     * @brief Desired minimum duration of one CUDA event timing window.
     *
     * A probe launch chooses the smallest replay count that should make the
     * event span approximately this duration, bounded by the configured cap.
     * Large kernels therefore remain one replay while tiny decode kernels gain
     * enough event resolution to distinguish nearby candidate schedules.
     */
    constexpr double kTargetTimedWindowUs = 512.0;

    /** FNV-1a constants used for a stable cross-run measurement-order seed. */
    constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
    constexpr uint64_t kFnvPrime = 1099511628211ULL;
    constexpr const char *kBroadMeasurementProtocol =
        "sample_interleaved_v1";

    /**
     * @brief Invoke one CUDA driver profiler-control entry point by symbol.
     *
     * CUDA 13 still exports `cuProfilerStart`/`cuProfilerStop` from libcuda,
     * but the minimal toolkit image used by this project no longer installs
     * the historical `cuda_profiler_api.h` runtime header. Resolving the stable
     * driver ABI keeps app-controlled Nsight collection available without
     * declaring an undocumented runtime prototype or adding profiler behavior
     * to production code.
     */
    bool invokeCudaProfilerControl(const char *symbol)
    {
        using ControlFunction = int (*)();
        static void *driver = []
        {
            return ::dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        }();
        if (!driver)
            return false;
        ::dlerror();
        void *raw = ::dlsym(driver, symbol);
        if (!raw || ::dlerror() != nullptr)
            return false;
        return reinterpret_cast<ControlFunction>(raw)() == 0;
    }

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

    using Shape = native_vnni_dispatch::NativeVNNIShapeSpec;

    /**
     * @brief Return the shared model and generic-certification shape inventory.
     *
     * CUDA deliberately owns no private shape list. The JSON manifest is also
     * consumed by ROCm, CPU refresh orchestration, and the sealed common
     * compiler, so adding a policy dimension to one backend necessarily adds it
     * to all three.
     */
    const std::vector<Shape> &trainingShapes()
    {
        return native_vnni_dispatch::nativeVnniShapeManifest();
    }

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

    /**
     * @brief Derive the deterministic candidate permutation seed for one cell.
     *
     * Candidate order must vary across shape and execution-mode cells so fixed
     * registry position cannot correlate with clock, temperature, or first-use
     * drift. The seed is emitted beside every aggregate and raw timing row,
     * making the exact broad-sweep permutation auditable.
     */
    uint64_t measurementOrderSeed(
        const FormatSpec &format,
        const Shape &shape,
        int m,
        ExecutionMode mode)
    {
        uint64_t hash = kFnvOffsetBasis;
        const auto mix_string = [&](const std::string &value)
        {
            for (const unsigned char byte : value)
            {
                hash ^= static_cast<uint64_t>(byte);
                hash *= kFnvPrime;
            }
            hash ^= 0xffULL;
            hash *= kFnvPrime;
        };
        const auto mix_integer = [&](uint64_t value)
        {
            for (int byte = 0; byte < 8; ++byte)
            {
                hash ^= (value >> (byte * 8)) & 0xffULL;
                hash *= kFnvPrime;
            }
        };
        mix_string(format.name);
        mix_string(shape.name);
        mix_integer(static_cast<uint64_t>(m));
        mix_integer(static_cast<uint64_t>(mode));
        return hash;
    }

    /**
     * @brief Derive an auditable candidate-order seed for one paired sample.
     *
     * Every pair starts from the cell-level broad-sweep seed and mixes its
     * zero-based pair index. This makes selected-first and reference-first
     * execution vary within a cell while remaining exactly reproducible from
     * the retained CSV. A paired result therefore cannot accidentally inherit
     * a permanent first-candidate thermal or clock-order advantage.
     */
    uint64_t pairedOrderSeed(uint64_t cell_seed, size_t pair_index)
    {
        uint64_t hash = cell_seed;
        for (int byte = 0; byte < 8; ++byte)
        {
            hash ^= (static_cast<uint64_t>(pair_index) >> (byte * 8)) & 0xffULL;
            hash *= kFnvPrime;
        }
        return hash;
    }

    /**
     * @brief Extend a runtime-cell seed with one typed request identity.
     *
     * A tournament may contain several candidate edges for the same
     * format/shape/mode cell. Giving every edge an independently mixed seed
     * prevents all of those sessions from running the selected role first on
     * exactly the same sample indices while retaining complete replayability.
     */
    uint64_t pairedRequestSeed(
        uint64_t cell_seed,
        const std::string &request_id)
    {
        uint64_t hash = cell_seed;
        for (const unsigned char byte : request_id)
        {
            hash ^= static_cast<uint64_t>(byte);
            hash *= kFnvPrime;
        }
        return hash;
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
        int timed_replay_cap = kDefaultTimedReplayCap;
        int max_cases = std::numeric_limits<int>::max();
        std::set<std::string> formats;
        std::set<std::string> shapes;
        std::set<std::string> candidate_families;
        std::set<std::string> candidate_ids;
        std::vector<int> m_values = {
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16, 31};
        std::vector<ExecutionMode> execution_modes = {
            ExecutionMode::Eager,
            ExecutionMode::GraphCaptured,
        };
        std::string aggregate_path = "/tmp/llaminar_cuda_decode_strong.csv";
        std::string timing_path = "/tmp/llaminar_cuda_decode_strong.timing.csv";
        std::string paired_path;
        std::string paired_request_manifest_path;
        std::string profiler_request_id;
        std::vector<
            native_vnni_dispatch::NativeVNNIPairedTimingRequest>
            paired_requests;

        /** Return whether this invocation is a confirmation-only transaction. */
        [[nodiscard]] bool pairedConfirmation() const
        {
            return !paired_path.empty() && !paired_request_manifest_path.empty();
        }
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
        config.timed_replay_cap = envPositiveInt(
            "LLAMINAR_CUDA_NVNNI_DECODE_TIMED_REPLAYS",
            kDefaultTimedReplayCap);
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
        const std::string paired = envString(
            "LLAMINAR_CUDA_NVNNI_DECODE_PAIRED_CSV");
        const std::string paired_requests = envString(
            "LLAMINAR_CUDA_NVNNI_DECODE_PAIRED_REQUEST_MANIFEST");
        if (!aggregate.empty())
            config.aggregate_path = aggregate;
        if (!timing.empty())
            config.timing_path = timing;
        if (paired.empty() != paired_requests.empty())
        {
            throw std::runtime_error(
                "paired CUDA confirmation requires both "
                "LLAMINAR_CUDA_NVNNI_DECODE_PAIRED_REQUEST_MANIFEST and "
                "LLAMINAR_CUDA_NVNNI_DECODE_PAIRED_CSV");
        }
        if (!paired.empty())
        {
            if (!config.formats.empty() || !config.shapes.empty() ||
                !config.candidate_families.empty() ||
                !config.candidate_ids.empty() ||
                !envString("LLAMINAR_CUDA_NVNNI_DECODE_M").empty() ||
                !envString(
                    "LLAMINAR_CUDA_NVNNI_DECODE_EXECUTION_MODES").empty() ||
                !envString("LLAMINAR_CUDA_NVNNI_DECODE_MAX_CASES").empty())
            {
                throw std::runtime_error(
                    "paired request manifest is authoritative and cannot be "
                    "combined with format/shape/candidate/M/mode/case filters");
            }
            const auto manifest =
                native_vnni_dispatch::loadNativeVnniPairedRequestManifest(
                    paired_requests);
            if (manifest.requests.empty())
            {
                throw std::runtime_error(
                    "paired request manifest is already green and has no work");
            }
            config.paired_path = paired;
            config.paired_request_manifest_path = paired_requests;
            config.paired_requests = manifest.requests;
            config.m_values = {1};
        }
        config.profiler_request_id =
            native_vnni_dispatch::profilerRequestId();
        if (!config.profiler_request_id.empty())
        {
            if (config.pairedConfirmation())
            {
                throw std::runtime_error(
                    "isolated CUDA profiling cannot run inside paired timing");
            }
            if (config.formats.size() != 1 || config.shapes.size() != 1 ||
                config.candidate_ids.size() != 1 ||
                !config.candidate_families.empty() ||
                config.m_values.size() != 1 ||
                config.execution_modes.size() != 1 || config.max_cases != 1)
            {
                throw std::runtime_error(
                    "isolated CUDA profiling requires one explicit format, "
                    "shape, candidate, M, mode, and max case");
            }
        }
        return config;
    }

    bool selected(const std::set<std::string> &filter, const std::string &name)
    {
        return filter.empty() || filter.count(lower(name)) != 0;
    }

    /**
     * @brief Return the smallest exact KB spelling for every useful partition.
     *
     * @param k_groups Number of 32-value NativeVNNI groups along K.
     * @return Sorted exact-KB candidates with one representative for each
     *         distinct `ceil(k_groups / KB)` partition width.
     *
     * Two KB values that produce the same partition width execute identical
     * useful dot products. The larger value merely launches more empty partial
     * producers and asks the ordered reducer to consume more explicit zeroes.
     * Keeping the smallest spelling therefore preserves every distinct FP32
     * reduction tree while removing candidates that are strictly more costly.
     */
    std::vector<int> economicalExactKBlocks(int k_groups)
    {
        std::vector<int> result;
        std::set<int> represented_partition_widths;
        const int maximum = std::min(k_groups, 256);
        for (int kb = 1; kb <= maximum; ++kb)
        {
            const int blocks_per_partition = (k_groups + kb - 1) / kb;
            if (represented_partition_widths.insert(blocks_per_partition).second)
                result.push_back(kb);
        }
        return result;
    }

    /**
     * @brief Enumerate exact K-partition counts for the requested sweep mode.
     *
     * The unattended production sweep keeps only the first KB spelling for
     * each distinct blocks-per-partition width.  Later spellings execute the
     * same useful arithmetic while publishing one or more trailing zero
     * partials, so they cannot be economical production winners.
     *
     * An explicit candidate-id filter has different semantics: it is a request
     * to exercise that exact arithmetic and publication schedule.  Diagnostic
     * integration tests use this path to poison the partial workspace and
     * prove that trailing empty partitions overwrite stale bytes before the
     * ordered reducer runs.  Keeping all valid KB values forceable here avoids
     * weakening that regression while preserving the economical default
     * corpus.
     *
     * @param config Parsed trainer selection filters.
     * @param k_groups Number of 32-value K groups in the selected shape.
     * @return Exact KB values to instantiate for this shape.
     */
    std::vector<int> exactKBlocksForSweep(
        const TrainerConfig &config,
        int k_groups)
    {
        if (config.candidate_ids.empty())
            return economicalExactKBlocks(k_groups);

        std::vector<int> result;
        const int maximum = std::min(k_groups, 256);
        result.reserve(static_cast<size_t>(maximum));
        for (int kb = 1; kb <= maximum; ++kb)
            result.push_back(kb);
        return result;
    }

    std::vector<Candidate> fastM1Candidates(
        const TrainerConfig &config,
        int k_groups)
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
            /*
             * Exact KB fixes partition boundaries and is therefore part of the
             * arithmetic identity, not a continuous occupancy hint. Promotion
             * evidence must represent every distinct reduction tree through
             * KB256. KB values that produce the same blocks-per-partition as a
             * smaller KB perform identical useful arithmetic plus trailing zero
             * partitions, so the larger spelling is strictly dominated and is
             * excluded by the shared economy rule.
             */
            for (const int kb : exactKBlocksForSweep(config, k_groups))
                add_kpar(tile_n, cpt, kb);
        }
        return result;
    }

    std::vector<Candidate> candidatesForM(
        int m,
        int k_groups,
        const TrainerConfig &config)
    {
        if (m == 1)
            return fastM1Candidates(config, k_groups);
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

    /**
     * @brief Return the explicit trainer-only oracle for an unseen M1 shape.
     *
     * A generated production policy cannot resolve a geometry before that
     * geometry has been measured. The trainer therefore forces one universally
     * reachable ordered K-partition schedule to establish numerical eligibility
     * for Fast candidates. This is diagnostic evidence only: the override is
     * scoped inside the harness, never exposed through production dispatch, and
     * never used as the verifier oracle after the staged M1 policy exists.
     */
    const Candidate &diagnosticM1OracleCandidate()
    {
        static const Candidate candidate{
            "cuda.nvnni.decode.fast_m1.kpar.tn128.cpt1.kb1",
            CandidateFamily::KPar,
            128,
            1,
            0,
            0,
            0,
            1,
            1};
        return candidate;
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
        ExecutionMode mode,
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
                tag("execution_mode") != executionModeName(mode) ||
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

    /** Serial-row oracle and the explicit route that produced it. */
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
        ExecutionMode mode,
        int m,
        int n,
        int k,
        uint8_t codebook,
        DeviceId device,
        const Candidate *diagnostic_m1_oracle)
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
        std::optional<CandidateOverride> oracle_override;
        if (diagnostic_m1_oracle)
            oracle_override.emplace(*diagnostic_m1_oracle);
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

            /*
             * Poison the reusable reduction arena before every candidate. This
             * makes an omitted partial write deterministic and immediately
             * visible to the serial oracle instead of relying on allocator
             * history to reproduce stale graph-workspace bytes.
             */
            if (workspace->hasBuffer(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS))
            {
                void *partials = workspace->getBuffer(
                    GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS);
                const size_t partial_bytes = workspace->getBufferSize(
                    GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS);
                if (cudaMemsetAsync(partials, 0xA5, partial_bytes, stream) !=
                    cudaSuccess)
                {
                    workspace_consumer->unbindWorkspace();
                    kernel->setGPUStream(nullptr);
                    (void)cudaStreamDestroy(stream);
                    return fail("workspace_poison");
                }
            }
        }
        const auto cleanup = [&]
        {
            if (workspace_consumer)
                workspace_consumer->unbindWorkspace();
            kernel->setGPUStream(nullptr);
            (void)cudaStreamDestroy(stream);
        };

        result.output.resize(static_cast<size_t>(m) * static_cast<size_t>(n));
        const float *source = static_cast<const float *>(grouped_input->data());
        std::vector<std::unique_ptr<FP32Tensor>> row_inputs;
        std::vector<std::unique_ptr<FP32Tensor>> row_outputs;
        row_inputs.reserve(static_cast<size_t>(m));
        row_outputs.reserve(static_cast<size_t>(m));
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
                !row_output->allocateOnDevice(device))
            {
                cleanup();
                return fail("serial_row_prepare");
            }
            row_inputs.push_back(std::move(row_input));
            row_outputs.push_back(std::move(row_output));
        }

        const auto launch_serial_rows = [&]() -> bool
        {
            for (int row = 0; row < m; ++row)
            {
                if (!kernel->multiply_tensor(
                        row_inputs[static_cast<size_t>(row)].get(),
                        row_outputs[static_cast<size_t>(row)].get(),
                        1,
                        n,
                        k))
                {
                    return false;
                }
            }
            return true;
        };

        CapturedLaunch captured;
        if (mode == ExecutionMode::GraphCaptured)
        {
            /*
             * Capture a disposable primer first. This performs any CUDA graph
             * instantiation work without polluting route evidence, while also
             * forcing the generated resolver to execute under the captured
             * surface rather than borrowing the eager M1 route.
             */
            CapturedLaunch primer;
            if (!primer.capture(stream, launch_serial_rows) ||
                !primer.launch(stream) ||
                cudaStreamSynchronize(stream) != cudaSuccess)
            {
                cleanup();
                return fail("serial_graph_primer");
            }
            primer.reset();
            PerfStatsCollector::reset();
            if (!captured.capture(stream, launch_serial_rows) ||
                !captured.launch(stream) ||
                cudaStreamSynchronize(stream) != cudaSuccess)
            {
                cleanup();
                return fail("serial_graph_launch");
            }
        }
        else
        {
            PerfStatsCollector::reset();
            if (!launch_serial_rows() ||
                cudaStreamSynchronize(stream) != cudaSuccess)
            {
                cleanup();
                return fail("serial_eager_launch");
            }
        }

        for (int row = 0; row < m; ++row)
        {
            std::vector<float> row_host;
            if (!copyDeviceOutput(
                    row_outputs[static_cast<size_t>(row)].get(),
                    static_cast<size_t>(n),
                    stream,
                    row_host))
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
        result.route = findRoute(mode, 1, n, k, codebook, "fast");
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
        std::vector<size_t> sample_measurement_orders;
        std::vector<uint64_t> sample_order_seeds;
        int timed_replays = 0;
        double effective_bandwidth_gbs = 0.0;
    };

    /** Complete result for one sample-interleaved broad candidate matrix. */
    struct InterleavedCandidateBatch
    {
        bool valid = false;
        std::string failure;
        std::vector<CandidateEvidence> evidence;
        int isolated_profile_launches = 0;
    };

    /**
     * @brief Measure a complete broad candidate set in shuffled sample rounds.
     *
     * Candidate-block timing is vulnerable to slow device clock and thermal
     * drift: candidate A can finish all thirty samples before candidate B even
     * begins. Randomizing that block once removes registry-order bias across
     * cells but does not make the two medians contemporaneous. This routine
     * prepares correctness, workspace, and captured graphs once per candidate,
     * then runs exactly one timed window for every candidate in each shuffled
     * round. All candidates therefore see the same sequence of device states.
     *
     * @return Per-candidate evidence in the same order as @p candidates.
     */
    InterleavedCandidateBatch runCandidatesInterleaved(
        ITensorGemm *kernel,
        TensorBase *input,
        const SerialEvidence &serial,
        const std::vector<Candidate> &candidates,
        ExecutionMode mode,
        int m,
        int n,
        int k,
        uint8_t codebook,
        size_t weight_bytes,
        int warmups,
        int sample_count,
        int timed_replay_cap,
        uint64_t cell_order_seed,
        const std::string &profiler_request_id,
        DeviceId device)
    {
        InterleavedCandidateBatch result;
        if (!kernel || !input || !serial.valid || candidates.empty() ||
            warmups < 0 || sample_count <= 0 || timed_replay_cap <= 0)
        {
            result.failure = "invalid_arguments";
            return result;
        }

        struct PreparedCandidate
        {
            const Candidate *candidate = nullptr;
            CandidateEvidence evidence;
            std::unique_ptr<CapturedLaunch> captured;
        };

        cudaStream_t stream = nullptr;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
        std::unique_ptr<DeviceWorkspaceManager> workspace;
        std::vector<PreparedCandidate> prepared;

        const auto cleanup = [&]
        {
            if (start)
                (void)cudaEventDestroy(start);
            if (stop)
                (void)cudaEventDestroy(stop);
            for (PreparedCandidate &state : prepared)
            {
                if (state.captured)
                    state.captured->reset();
            }
            if (workspace_consumer)
                workspace_consumer->unbindWorkspace();
            kernel->setGPUStream(nullptr);
            if (stream)
                (void)cudaStreamDestroy(stream);
        };
        const auto fail = [&](const std::string &reason)
        {
            result.failure = reason;
            cleanup();
            return std::move(result);
        };

        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
            cudaSuccess)
        {
            return fail("stream_create");
        }
        kernel->setGPUStream(stream);
        if (workspace_consumer)
        {
            const auto requirements = workspace_consumer->getWorkspaceRequirements(
                m, n, k);
            workspace = std::make_unique<DeviceWorkspaceManager>(
                device, workspaceBudgetFor(requirements));
            if (!workspace->allocate(requirements))
                return fail("workspace_allocate");
            workspace_consumer->bindWorkspace(workspace.get());
        }

        auto output = TestTensorFactory::createFP32(
            {static_cast<size_t>(m), static_cast<size_t>(n)});
        if (!input->ensureOnDevice(device) || !output->allocateOnDevice(device))
            return fail("tensor_prepare");
        if (cudaEventCreate(&start) != cudaSuccess ||
            cudaEventCreate(&stop) != cudaSuccess)
        {
            return fail("event_create");
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
        const auto execute_once = [&](PreparedCandidate &state) -> bool
        {
            if (mode == ExecutionMode::GraphCaptured)
                return state.captured && state.captured->launch(stream);
            CandidateOverride override(*state.candidate);
            return run_once();
        };

        prepared.reserve(candidates.size());
        for (const Candidate &candidate : candidates)
        {
            PreparedCandidate state;
            state.candidate = &candidate;
            state.evidence.explicit_stream_ok = stream != nullptr;
            state.evidence.workspace_ok = true;
            state.evidence.graph_capture_ok =
                mode == ExecutionMode::Eager;

            if (mode == ExecutionMode::GraphCaptured)
            {
                /*
                 * The primer resolves lazy host setup. The scoped override is
                 * needed only while the graph nodes are recorded; replay uses
                 * the concrete launch parameters embedded in that graph.
                 */
                {
                    CandidateOverride override(candidate);
                    if (!run_once() ||
                        cudaStreamSynchronize(stream) != cudaSuccess)
                    {
                        return fail(candidate.id + ":graph_primer");
                    }
                }
                PerfStatsCollector::reset();
                state.captured = std::make_unique<CapturedLaunch>();
                {
                    CandidateOverride override(candidate);
                    state.evidence.graph_capture_ok = state.captured->capture(
                        stream, run_once);
                }
                if (!state.evidence.graph_capture_ok)
                    return fail(candidate.id + ":graph_capture");
            }
            else
            {
                PerfStatsCollector::reset();
            }

            prepared.push_back(std::move(state));
            PreparedCandidate &current = prepared.back();
            if (!execute_once(current) ||
                cudaStreamSynchronize(stream) != cudaSuccess)
            {
                return fail(candidate.id + ":candidate_launch");
            }
            current.evidence.route = findRoute(
                mode,
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
                !execute_once(current) ||
                cudaStreamSynchronize(stream) != cudaSuccess)
            {
                return fail(candidate.id + ":repeat_launch");
            }
            std::vector<float> repeated_output;
            if (!copyDeviceOutput(
                    output.get(),
                    static_cast<size_t>(m) * static_cast<size_t>(n),
                    stream,
                    repeated_output))
            {
                return fail(candidate.id + ":repeat_download");
            }
            current.evidence.repeat_byte_mismatches =
                trainer::nativeByteMismatchCount(
                    first_output, repeated_output);
            current.evidence.comparison = trainer::compareFP32(
                repeated_output,
                serial.output,
                static_cast<size_t>(m) * static_cast<size_t>(n));
            current.evidence.numerical_correctness =
                current.evidence.comparison.nonfinite_count == 0 &&
                current.evidence.comparison.cosine >= 0.999;

            int timed_replays = timed_replay_cap;
            if (timed_replay_cap > 1)
            {
                if (cudaEventRecord(start, stream) != cudaSuccess ||
                    !execute_once(current) ||
                    cudaEventRecord(stop, stream) != cudaSuccess ||
                    cudaEventSynchronize(stop) != cudaSuccess)
                {
                    return fail(candidate.id + ":timed_probe");
                }
                float probe_ms = 0.0f;
                if (cudaEventElapsedTime(&probe_ms, start, stop) != cudaSuccess)
                    return fail(candidate.id + ":timed_probe_read");
                const double probe_us =
                    static_cast<double>(probe_ms) * 1000.0;
                if (probe_us > 0.0)
                {
                    timed_replays = std::clamp(
                        static_cast<int>(std::ceil(
                            kTargetTimedWindowUs / probe_us)),
                        1,
                        timed_replay_cap);
                }
            }
            current.evidence.timed_replays = timed_replays;
            current.evidence.samples.reserve(
                static_cast<size_t>(sample_count));
            current.evidence.sample_measurement_orders.reserve(
                static_cast<size_t>(sample_count));
            current.evidence.sample_order_seeds.reserve(
                static_cast<size_t>(sample_count));
            current.evidence.valid = current.evidence.route.valid;
            if (!current.evidence.valid)
                return fail(candidate.id + ":route_proof");
        }

        std::vector<size_t> order(prepared.size());
        std::iota(order.begin(), order.end(), 0u);
        for (int warmup = 0; warmup < warmups; ++warmup)
        {
            std::iota(order.begin(), order.end(), 0u);
            const uint64_t seed = pairedOrderSeed(
                cell_order_seed,
                static_cast<size_t>(sample_count + warmup));
            std::mt19937_64 engine(seed);
            std::shuffle(order.begin(), order.end(), engine);
            for (const size_t index : order)
            {
                if (!execute_once(prepared[index]))
                    return fail("interleaved_warmup_launch");
            }
            if (cudaStreamSynchronize(stream) != cudaSuccess)
                return fail("interleaved_warmup_sync");
        }

        if (!profiler_request_id.empty())
        {
            if (prepared.size() != 1)
                return fail("profiler_requires_one_candidate");
            /*
             * Nsight Compute starts with collection disabled. Setup, graph
             * capture, correctness launches, route proof, and warmup above are
             * therefore absent from the profiler report. The only enabled
             * work is this one production candidate launch (or graph replay)
             * and its completion synchronization. This launch is not appended
             * to any canonical timing sample.
             */
            if (!invokeCudaProfilerControl("cuProfilerStart"))
                return fail("profiler_start");
            const bool launch_ok = execute_once(prepared.front());
            const cudaError_t synchronize_status =
                cudaStreamSynchronize(stream);
            const bool stop_ok =
                invokeCudaProfilerControl("cuProfilerStop");
            if (!launch_ok || synchronize_status != cudaSuccess ||
                !stop_ok)
            {
                return fail("profiler_target_launch");
            }
            result.isolated_profile_launches = 1;
            std::fprintf(
                stderr,
                "[NativeVNNIProfiler][CUDA] request=%s candidate=%s "
                "mode=%s M=%d N=%d K=%d launches=1\n",
                profiler_request_id.c_str(),
                prepared.front().candidate->id.c_str(),
                executionModeName(mode),
                m,
                n,
                k);
        }

        /*
         * A two-candidate confirmation must balance the first-launch role by
         * construction. Independent shuffles only balance in expectation and
         * can give one candidate a persistent clock, cache, or temperature
         * advantage in a finite run. The cell seed chooses which role starts
         * the alternating sequence, then every following sample reverses it.
         * Even sample counts are exactly balanced; odd counts differ by one.
         */
        const size_t first_candidate_offset =
            static_cast<size_t>(cell_order_seed & 1ULL);
        for (int sample = 0; sample < sample_count; ++sample)
        {
            std::iota(order.begin(), order.end(), 0u);
            const uint64_t sample_seed = pairedOrderSeed(
                cell_order_seed, static_cast<size_t>(sample));
            if (prepared.size() == 2)
            {
                if ((static_cast<size_t>(sample) + first_candidate_offset) % 2 != 0)
                    std::reverse(order.begin(), order.end());
            }
            else
            {
                std::mt19937_64 engine(sample_seed);
                std::shuffle(order.begin(), order.end(), engine);
            }
            for (size_t within_sample_order = 0;
                 within_sample_order < order.size();
                 ++within_sample_order)
            {
                PreparedCandidate &current = prepared[order[within_sample_order]];
                if (cudaEventRecord(start, stream) != cudaSuccess)
                    return fail("interleaved_timed_launch");
                for (int replay = 0;
                     replay < current.evidence.timed_replays;
                     ++replay)
                {
                    if (!execute_once(current))
                        return fail("interleaved_timed_launch");
                }
                if (cudaEventRecord(stop, stream) != cudaSuccess ||
                    cudaEventSynchronize(stop) != cudaSuccess)
                {
                    return fail("interleaved_timed_launch");
                }
                float elapsed_ms = 0.0f;
                if (cudaEventElapsedTime(&elapsed_ms, start, stop) != cudaSuccess)
                    return fail("interleaved_timed_read");
                current.evidence.samples.push_back(
                    static_cast<double>(elapsed_ms) * 1000.0 /
                    static_cast<double>(current.evidence.timed_replays));
                current.evidence.sample_measurement_orders.push_back(
                    within_sample_order);
                current.evidence.sample_order_seeds.push_back(sample_seed);
            }
        }

        result.evidence.reserve(prepared.size());
        for (PreparedCandidate &state : prepared)
        {
            std::vector<double> sorted_samples = state.evidence.samples;
            std::sort(sorted_samples.begin(), sorted_samples.end());
            state.evidence.timing =
                trainer::summarizeSortedTimingSamples(sorted_samples);
            if (state.evidence.timing.median > 0.0)
            {
                state.evidence.effective_bandwidth_gbs =
                    static_cast<double>(weight_bytes) /
                    (state.evidence.timing.median * 1.0e-6) / 1.0e9;
            }
            result.evidence.push_back(std::move(state.evidence));
        }
        cleanup();
        result.valid = true;
        return result;
    }

    /**
     * @brief Emit one candidate observation from an interleaved timing pair.
     *
     * The paired sidecar intentionally has its own schema. It is confirmation
     * evidence for a frozen decision, not broad fitting evidence, and must not
     * be accepted by the ordinary CUDA corpus adapter. Pair index, within-pair
     * order, and both deterministic seeds are retained so the Python
     * certificate can reject missing, duplicated, or non-interleaved rows.
     */
    void writePairedSample(
        std::FILE *file,
        const std::string &request_id,
        const FormatSpec &format,
        const Shape &shape,
        ExecutionMode mode,
        int m,
        uint8_t source_codebook,
        uint8_t execution_codebook,
        size_t pair_index,
        size_t within_pair_order,
        uint64_t cell_order_seed,
        uint64_t pair_order_seed,
        const char *candidate_role,
        const Candidate &candidate,
        const CandidateEvidence &evidence,
        double latency,
        const SerialEvidence &serial,
        const TrainerConfig &config,
        bool correctness_pass)
    {
        if (!file || !std::isfinite(latency) || latency <= 0.0)
            throw std::runtime_error("paired CUDA evidence has invalid latency");
        const auto &comparison = evidence.comparison;
        std::fprintf(
            file,
            "cuda-paired-interleaved-v2,%s,cuda,decode,%s,%u,%u,%s,%s,%d,%d,%d,"
            "%zu,%d,%zu,%llu,%llu,%s,%s,%d,%.9f,%a,%d,%zu,%zu,%zu,"
            "%.17g,%.17g,%.17g,%.17g,%s,%s,%d,%d,%d,%d,%s,%s,%d,%d,%d,"
            "%s,%d,%d\n",
            request_id.c_str(),
            format.name.c_str(),
            static_cast<unsigned>(source_codebook),
            static_cast<unsigned>(execution_codebook),
            shape.name.c_str(),
            executionModeName(mode),
            m,
            shape.N,
            shape.K,
            pair_index,
            config.samples,
            within_pair_order,
            static_cast<unsigned long long>(cell_order_seed),
            static_cast<unsigned long long>(pair_order_seed),
            candidate_role,
            candidate.id.c_str(),
            evidence.timed_replays,
            latency,
            latency,
            config.warmups,
            comparison.mismatch_count,
            comparison.first_mismatch_index,
            evidence.repeat_byte_mismatches,
            comparison.max_abs,
            comparison.relative_l2,
            comparison.cosine,
            comparison.symmetric_kld,
            comparison.actual_digest.c_str(),
            comparison.expected_digest.c_str(),
            evidence.graph_capture_ok ? 1 : 0,
            evidence.workspace_ok ? 1 : 0,
            evidence.explicit_stream_ok ? 1 : 0,
            evidence.route.valid ? 1 : 0,
            evidence.route.candidate_id.c_str(),
            evidence.route.path.c_str(),
            evidence.route.tile_n,
            evidence.route.cpt,
            evidence.route.effective_kb,
            serial.route.candidate_id.c_str(),
            evidence.numerical_correctness ? 1 : 0,
            correctness_pass ? 1 : 0);
    }

    /**
     * @brief Execute every concrete edge in one typed paired request manifest.
     *
     * The manifest is authoritative for format, shape, execution mode, and the
     * two forceable candidate IDs. For each request the persistent interleaved
     * runner prepares one explicit stream, production workspace, output tensor,
     * and captured graph per candidate. It then executes one timing window per
     * candidate in independently shuffled sample rounds. This matches the broad
     * trainer lifecycle while making selected/reference samples contemporary.
     *
     * Multiple tournament edges for the same runtime cell are written to one
     * v2 CSV. The request ID gives every edge an independent pair-index
     * namespace, allowing the strict Python reader to consume the transaction
     * without splitting files or guessing candidate relationships.
     */
    void runPairedConfirmation(
        const TrainerConfig &config,
        DeviceId device)
    {
        ASSERT_TRUE(config.pairedConfirmation());
        ASSERT_EQ(config.m_values, std::vector<int>({1}))
            << "paired CUDA confirmation currently certifies Fast M1 only";
        ASSERT_FALSE(config.paired_requests.empty());

        const std::filesystem::path output_path(config.paired_path);
        if (!output_path.parent_path().empty())
            std::filesystem::create_directories(output_path.parent_path());
        std::FILE *output = std::fopen(config.paired_path.c_str(), "w");
        ASSERT_NE(output, nullptr) << config.paired_path;
        std::fprintf(
            output,
            "protocol_version,request_id,backend,phase,source_format,source_codebook,"
            "execution_codebook,shape,execution_mode,m,n,k,pair_index,"
            "configured_pair_count,within_pair_order,cell_order_seed,"
            "pair_order_seed,candidate_role,candidate_id,timed_replays,"
            "latency_us,latency_us_hex,warmup_count,bit_mismatches,"
            "first_bit_mismatch,repeat_byte_mismatches,max_abs,relative_l2,"
            "cosine,symmetric_kld,grouped_output_digest,serial_output_digest,"
            "graph_capture_ok,workspace_ok,explicit_stream_ok,route_counter_ok,"
            "observed_candidate_id,observed_path,observed_tile_n,observed_cpt,"
            "observed_effective_kb,serial_m1_candidate_id,numerical_correctness,"
            "correctness_pass\n");

        size_t executed_cases = 0;
        for (const auto &request : config.paired_requests)
        {
            const auto format_iterator = std::find_if(
                kFormats.begin(),
                kFormats.end(),
                [&](const FormatSpec &format)
                { return lower(format.name) == lower(request.source_format); });
            ASSERT_NE(format_iterator, kFormats.end())
                << request.request_id << " unknown format "
                << request.source_format;
            const FormatSpec &format = *format_iterator;

            const auto shape_iterator = std::find_if(
                trainingShapes().begin(),
                trainingShapes().end(),
                [&](const Shape &shape)
                { return shape.name == request.shape; });
            ASSERT_NE(shape_iterator, trainingShapes().end())
                << request.request_id << " unknown shape " << request.shape;
            const Shape &shape = *shape_iterator;
            ASSERT_EQ(shape.N, request.n) << request.request_id;
            ASSERT_EQ(shape.K, request.k) << request.request_id;
            ASSERT_EQ(request.m, 1) << request.request_id;

            ExecutionMode mode = ExecutionMode::Eager;
            if (request.execution_mode == "eager")
                mode = ExecutionMode::Eager;
            else if (request.execution_mode == "graph_captured")
                mode = ExecutionMode::GraphCaptured;
            else
            {
                FAIL() << request.request_id << " unsupported execution mode";
            }

            auto weights = format.create(
                static_cast<size_t>(shape.N),
                static_cast<size_t>(shape.K));
            ASSERT_NE(weights, nullptr) << request.request_id;
            const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(
                weights.get());
            ASSERT_NE(unpackable, nullptr) << request.request_id;
            const NativeVnniFormatInfo *source_info =
                unpackable->vnniFormatInfo();
            ASSERT_NE(source_info, nullptr) << request.request_id;
            ASSERT_EQ(
                static_cast<int>(source_info->codebook_id),
                request.source_codebook) << request.request_id;

            auto prepared = makeGpuPreparedGemm(
                weights.get(),
                device,
                "perf.cuda_native_vnni_decode_paired.weight");
            ASSERT_NE(prepared.kernel, nullptr) << request.request_id;
            DeviceNativeVNNIMatrixDesc matrix_desc;
            ASSERT_TRUE(prepared.kernel->exportNativeVNNIMatrixDesc(matrix_desc))
                << request.request_id;
            ASSERT_EQ(
                static_cast<int>(matrix_desc.codebook_id),
                request.execution_codebook) << request.request_id;

            TrainerConfig pair_config = config;
            pair_config.candidate_ids = {
                lower(request.selected_candidate_id),
                lower(request.exact_candidate_id),
            };
            const std::vector<Candidate> candidates = candidatesForM(
                request.m, shape.K / 32, pair_config);
            ASSERT_EQ(candidates.size(), 2u)
                << request.request_id
                << " did not resolve both concrete paired candidates";
            const auto candidate_by_id = [&](const std::string &id)
                -> const Candidate *
            {
                const auto iterator = std::find_if(
                    candidates.begin(),
                    candidates.end(),
                    [&](const Candidate &candidate)
                    { return candidate.id == lower(id); });
                return iterator == candidates.end() ? nullptr : &*iterator;
            };
            const Candidate *selected_candidate = candidate_by_id(
                request.selected_candidate_id);
            const Candidate *exact_candidate = candidate_by_id(
                request.exact_candidate_id);
            ASSERT_NE(selected_candidate, nullptr) << request.request_id;
            ASSERT_NE(exact_candidate, nullptr) << request.request_id;

            auto input = TestTensorFactory::createFP32Random(
                {1u, static_cast<size_t>(shape.K)},
                -0.25f,
                0.25f,
                0xC0DBu);
            const SerialEvidence serial = runSerialRows(
                prepared.kernel,
                input.get(),
                mode,
                request.m,
                shape.N,
                shape.K,
                matrix_desc.codebook_id,
                device,
                &diagnosticM1OracleCandidate());
            ASSERT_TRUE(serial.valid)
                << request.request_id << " serial failure=" << serial.failure;

            const uint64_t cell_seed = pairedRequestSeed(
                measurementOrderSeed(format, shape, request.m, mode),
                request.request_id);
            const std::vector<Candidate> paired_candidates = {
                *selected_candidate,
                *exact_candidate,
            };
            InterleavedCandidateBatch batch = runCandidatesInterleaved(
                prepared.kernel,
                input.get(),
                serial,
                paired_candidates,
                mode,
                request.m,
                shape.N,
                shape.K,
                matrix_desc.codebook_id,
                weights->size_bytes(),
                config.warmups,
                config.samples,
                config.timed_replay_cap,
                cell_seed,
                config.profiler_request_id,
                device);
            ASSERT_TRUE(batch.valid)
                << request.request_id
                << " paired interleaved failure=" << batch.failure;
            ASSERT_EQ(batch.evidence.size(), paired_candidates.size());

            for (size_t candidate_index = 0;
                 candidate_index < paired_candidates.size();
                 ++candidate_index)
            {
                const Candidate &candidate = paired_candidates[candidate_index];
                const CandidateEvidence &evidence = batch.evidence[candidate_index];
                const bool correctness_pass =
                    evidence.route.candidate_id == candidate.id &&
                    evidence.repeat_byte_mismatches == 0 &&
                    evidence.numerical_correctness;
                ASSERT_TRUE(correctness_pass)
                    << request.request_id << " candidate=" << candidate.id;
                ASSERT_EQ(
                    evidence.samples.size(),
                    static_cast<size_t>(config.samples));
                ASSERT_EQ(
                    evidence.sample_measurement_orders.size(),
                    evidence.samples.size());
                ASSERT_EQ(
                    evidence.sample_order_seeds.size(),
                    evidence.samples.size());
                const char *role = candidate_index == 0 ? "selected" : "exact";
                for (int pair_index = 0; pair_index < config.samples; ++pair_index)
                {
                    writePairedSample(
                        output,
                        request.request_id,
                        format,
                        shape,
                        mode,
                        request.m,
                        source_info->codebook_id,
                        matrix_desc.codebook_id,
                        static_cast<size_t>(pair_index),
                        evidence.sample_measurement_orders[
                            static_cast<size_t>(pair_index)],
                        cell_seed,
                        evidence.sample_order_seeds[
                            static_cast<size_t>(pair_index)],
                        role,
                        candidate,
                        evidence,
                        evidence.samples[static_cast<size_t>(pair_index)],
                        serial,
                        config,
                        correctness_pass);
                }
            }
            std::fflush(output);
            ++executed_cases;
        }
        std::fclose(output);
        ASSERT_EQ(executed_cases, config.paired_requests.size())
            << "CUDA paired confirmation did not execute the complete manifest";
    }

    /** Aggregate row retained until the winner bit is known. */
    struct MeasuredRow
    {
        const FormatSpec *format = nullptr;
        const Shape *shape = nullptr;
        Candidate candidate;
        ExecutionMode mode = ExecutionMode::Eager;
        int m = 0;
        size_t measurement_order = 0;
        uint64_t measurement_order_seed = 0;
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
        if (row.candidate_evidence.sample_measurement_orders.size() !=
                row.candidate_evidence.samples.size() ||
            row.candidate_evidence.sample_order_seeds.size() !=
                row.candidate_evidence.samples.size())
        {
            throw std::runtime_error(
                "interleaved CUDA timing metadata does not match samples");
        }
        for (size_t index = 0;
             index < row.candidate_evidence.samples.size();
             ++index)
        {
            const double latency = row.candidate_evidence.samples[index];
            std::fprintf(
                file,
                "cuda,decode,%s,%u,%u,%s,%s,%d,%d,%d,%s,%zu,%llu,%s,%zu,%llu,%zu,%d,%.9f,%a\n",
                row.format->name.c_str(),
                static_cast<unsigned>(row.source_codebook),
                static_cast<unsigned>(row.execution_codebook),
                row.shape->name.c_str(),
                executionModeName(row.mode),
                row.m,
                row.shape->N,
                row.shape->K,
                row.candidate.id.c_str(),
                row.measurement_order,
                static_cast<unsigned long long>(row.measurement_order_seed),
                kBroadMeasurementProtocol,
                row.candidate_evidence.sample_measurement_orders[index],
                static_cast<unsigned long long>(
                    row.candidate_evidence.sample_order_seeds[index]),
                index,
                row.candidate_evidence.timed_replays,
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
            "cuda,decode,%s,%u,%u,%s,%s,%d,%d,%d,%s,%zu,%llu,%s,%s,%d,%d,%d,%d,%d,%d,%d,%zu,%d,%zu,"
            "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%zu,%zu,%zu,%.17g,%.17g,%.17g,%.17g,%s,%s,%s,"
            "1,%d,1,1,%d,%s,%s,%d,%d,%d,%s,1,%d,%d,%d\n",
            row.format->name.c_str(),
            static_cast<unsigned>(row.source_codebook),
            static_cast<unsigned>(row.execution_codebook),
            row.shape->name.c_str(),
            executionModeName(row.mode),
            row.m,
            row.shape->N,
            row.shape->K,
            row.candidate.id.c_str(),
            row.measurement_order,
            static_cast<unsigned long long>(row.measurement_order_seed),
            kBroadMeasurementProtocol,
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
        if (config.pairedConfirmation())
        {
            runPairedConfirmation(config, device_);
            return;
        }
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
            "candidate_id,measurement_order,measurement_order_seed,measurement_protocol,family,tile_n,cpt,target_waves,mkg,max_kb,exact_kb,force_two_phase,weight_bytes,warmup_count,"
            "sample_count,min_us,median_us,p95_us,mad_us,cv,effective_bandwidth_gbs,bit_mismatches,"
            "first_bit_mismatch,repeat_byte_mismatches,max_abs,relative_l2,cosine,symmetric_kld,"
            "grouped_output_digest,serial_output_digest,timing_sample_digest,supported,graph_capture_ok,"
            "workspace_ok,explicit_stream_ok,route_counter_ok,observed_candidate_id,observed_path,"
            "observed_tile_n,observed_cpt,observed_effective_kb,serial_m1_candidate_id,serial_route_counter_ok,"
            "numerical_correctness,correctness_pass,is_winner\n");
        std::fprintf(
            timing,
            "backend,phase,source_format,source_codebook,execution_codebook,shape,execution_mode,m,n,k,"
            "candidate_id,measurement_order,measurement_order_seed,measurement_protocol,sample_measurement_order,sample_order_seed,sample_index,timed_replays,latency_us,latency_us_hex\n");

        int executed_cases = 0;
        int isolated_profile_launches = 0;
        for (const FormatSpec &format : kFormats)
        {
            if (!selected(config.formats, format.name))
                continue;
            for (const Shape &shape : trainingShapes())
            {
                if (!selected(config.shapes, shape.name))
                    continue;
                if (executed_cases >= config.max_cases)
                    break;

                auto weights = format.create(
                    static_cast<size_t>(shape.N),
                    static_cast<size_t>(shape.K));
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
                        m, shape.K / 32, config);
                    ASSERT_FALSE(candidates.empty())
                        << "No CUDA candidates selected for M=" << m;
                    auto input = TestTensorFactory::createFP32Random(
                        {static_cast<size_t>(m), static_cast<size_t>(shape.K)},
                        -0.25f,
                        0.25f,
                        0xC0DAu + static_cast<uint32_t>(m));
                    std::vector<MeasuredRow> rows;
                    for (const ExecutionMode mode : config.execution_modes)
                    {
                        const SerialEvidence serial = runSerialRows(
                            prepared.kernel,
                            input.get(),
                            mode,
                            m,
                            shape.N,
                            shape.K,
                            matrix_desc.codebook_id,
                            device_,
                            m == 1 ? &diagnosticM1OracleCandidate() : nullptr);
                        ASSERT_TRUE(serial.valid)
                            << format.name << " " << shape.name << " M=" << m
                            << " mode=" << executionModeName(mode)
                            << " serial failure=" << serial.failure;

                        std::vector<Candidate> ordered_candidates = candidates;
                        const uint64_t order_seed = measurementOrderSeed(
                            format, shape, m, mode);
                        std::mt19937_64 order_engine(order_seed);
                        std::shuffle(
                            ordered_candidates.begin(),
                            ordered_candidates.end(),
                            order_engine);
                        InterleavedCandidateBatch batch =
                            runCandidatesInterleaved(
                                prepared.kernel,
                                input.get(),
                                serial,
                                ordered_candidates,
                                mode,
                                m,
                                shape.N,
                                shape.K,
                                matrix_desc.codebook_id,
                                weights->size_bytes(),
                                config.warmups,
                                config.samples,
                                config.timed_replay_cap,
                                order_seed,
                                config.profiler_request_id,
                                device_);
                        ASSERT_TRUE(batch.valid)
                            << format.name << " " << shape.name << " M=" << m
                            << " mode=" << executionModeName(mode)
                            << " interleaved failure=" << batch.failure;
                        ASSERT_EQ(
                            batch.evidence.size(),
                            ordered_candidates.size());
                        isolated_profile_launches +=
                            batch.isolated_profile_launches;
                        for (size_t measurement_order = 0;
                             measurement_order < ordered_candidates.size();
                             ++measurement_order)
                        {
                            const Candidate &candidate =
                                ordered_candidates[measurement_order];
                            CandidateEvidence evidence = std::move(
                                batch.evidence[measurement_order]);
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
                                .format = &format,
                                .shape = &shape,
                                .candidate = candidate,
                                .mode = mode,
                                .m = m,
                                .measurement_order = measurement_order,
                                .measurement_order_seed = order_seed,
                                .source_codebook = source_info->codebook_id,
                                .execution_codebook = matrix_desc.codebook_id,
                                .weight_bytes = weights->size_bytes(),
                                .serial = serial,
                                .candidate_evidence = std::move(evidence),
                                .correctness_pass = correctness_pass,
                                .is_winner = false});
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
        if (!config.profiler_request_id.empty())
        {
            ASSERT_EQ(isolated_profile_launches, 1)
                << "each CUDA profiler request must execute one target launch";
        }
    }
} // namespace

#else

TEST(CUDANativeVNNIDecodeTrainer, RequiresCUDA)
{
    GTEST_SKIP() << "CUDA backend not enabled";
}

#endif
