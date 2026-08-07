/**
 * @file Perf__NativeVNNI_Throughput.cpp
 * @brief Per-format bandwidth benchmark for native-VNNI GEMV kernels
 *
 * Measures decode throughput for serial M=1 GEMV and runtime-M grouped
 * verifier GEMV. The canonical verifier inventory is M=2..16 plus M=31:
 * fifteen draft tokens require sixteen target rows, while M=31 guards against
 * accidentally turning that current graph capacity into a kernel limit.
 * shapes across all native-VNNI formats.
 * at model-realistic dimensions (Qwen2.5-0.5B, 3B, and 7B layer shapes).
 *
 * Each sub-8-bit format is benchmarked against the INT8 VNNI reference
 * (Q8_0 packed to INT8 scatter GEMV) on the same shape, yielding:
 *
 * Metrics reported:
 *   - Kernel time (μs): min/mean across benchmark runs
 *   - Effective bandwidth (GB/s): weight_bytes_read / kernel_time
 *   - BW efficiency (%): effective_BW / HBM_peak_BW
 *   - Speedup vs INT8: int8_min_us / format_min_us
 *   - Theoretical speedup: 8.0 / bpw (from streaming fewer bytes)
 *   - Kernel efficiency: actual_speedup / theoretical_speedup × 100%
 *   - Cosine similarity: GPU vs HipBLAS FP32 reference or, for very large
 *     trainer-only sweeps, GPU vs reset-AUTO native output.
 *
 * Multi-GPU support: work items are distributed across all available GPUs
 * using cost-descending round-robin to balance load evenly.
 *
 * The benchmark uses multiply_tensor() which includes:
 *   1. FP32→INT8 activation quantization on GPU
 *   2. Native-VNNI kernel dispatch (or INT8 scatter GEMV for reference)
 *   3. Scale application (FP32 output)
 *
 * @note Requires ROCm device. Tests skip if no GPU is available.
 * @note Run with build_v2_release for representative timing.
 */

#include <gtest/gtest.h>
#include "transfer/TransferEngine.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <omp.h>

#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"
#include "../../../utils/GpuPreparedGemmHarness.h"
#include "../../../utils/NativeVNNITrainerEvidence.h"
#include "../../../utils/QuantizedVerifierFormats.h"
#include "../../../utils/TestTensorFactory.h"
#include "../native_vnni_dispatch/NativeVNNIProfilerControl.h"
#include "../native_vnni_dispatch/NativeVNNIShapeManifest.h"
#include "fort.hpp"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include "GpuVerification.h"
extern "C" void rocmGemv_native_vnni_set_tuning_overrides(int kb, int target_waves_per_cu);
extern "C" void rocmGemv_native_vnni_reset_tuning_overrides();
extern "C" bool rocmGemv_native_vnni_query_serial_m1_config(
    uint8_t codebook_id,
    int N,
    int K,
    int *kb,
    int *target_waves_per_cu);
extern "C" bool rocmGemv_native_vnni_query_generated_config(
    uint8_t codebook_id,
    int M,
    int N,
    int K,
    int *kb,
    int *target_waves_per_cu);
extern "C" double rocmGemv_native_vnni_measure_generated_config_ns(
    uint8_t codebook_id,
    int M,
    const int *Ns,
    const int *Ks,
    int query_count,
    int iterations,
    uint64_t *out_checksum);
#endif

using namespace llaminar2;
using namespace llaminar2::rocm;
using namespace llaminar2::test;

namespace
{
#ifdef HAVE_ROCM
    using gpu_verify::destroyAllHipBLAS;
    using gpu_verify::gpuCosineSimilarity;
    using gpu_verify::gpuReferenceFP32Gemm;
    using gpu_verify::GpuWeightsCache;
#endif

    // =============================================================================
    // Constants
    // =============================================================================

    // MI50/MI60 theoretical HBM2 bandwidth (GB/s)
    // MI60 = 1.0 TB/s, MI50 = 0.77 TB/s. Use MI60 as reference.
    constexpr double HBM2_PEAK_GBPS = 1000.0;

    constexpr int WARMUP_RUNS = 5;
    constexpr int BENCH_RUNS = 20;

    /// Correctness gate: cosine similarity between the full native-VNNI path
    /// and an FP32 reference. This benchmark intentionally includes GPU
    /// activation quantization and FP16 native packed metadata, so asymmetric
    /// formats can sit a little below the generic gate without implying a
    /// decode bug. The exact packed-contract regression lives in
    /// V2_Integration_ROCm_NativeVNNI_GEMV.
    constexpr float COSINE_SIM_GATE = 0.9999f;

    inline float cosine_gate_for([[maybe_unused]] const std::string &format_name)
    {
        // Q4_K uses the Q4_1 native-VNNI packed contract: unsigned nibbles plus
        // FP16 sub-block scale/min metadata and quantized FP32 activations. On
        // the Qwen3.6 GDN time-projection M=3 verifier fixture, that legitimate
        // packed contract can dip just below the asymmetric FP32 health gate
        // even though the exact packed-contract integration test is bit-stable.
        if (format_name == "Q4_K")
            return 0.9997f;
        // Q2_K is another packed K-quant path where the native quantized
        // contract can be visibly lower than the full-FP32 hipBLAS reference
        // without indicating a dispatch mismatch. Keep this as a health gate;
        // exact dispatch equivalence is covered by the NativeVNNI GEMV
        // integration suite and model parity before any generated table install.
        if (format_name == "Q2_K")
            return 0.9998f;
        if (format_name == "Q4_1" || format_name == "Q5_1" || format_name == "Q5_K")
        {
            return 0.9998f;
        }
        return COSINE_SIM_GATE;
    }

    /// Number of GPUs to use (auto-detected, capped at available)
    static int NUM_GPUS = 1;

    // =============================================================================
    // Format descriptors
    // =============================================================================

    struct PerfFormatSpec
    {
        std::string name;
        double bpw;         ///< Bits per weight element
        bool is_superblock; ///< K must be multiple of 256

        std::function<std::unique_ptr<TensorBase>(size_t N, size_t K)> create;
    };

    static const std::vector<PerfFormatSpec> ALL_PERF_FORMATS = {
        // Tier 1: Simple 32-element blocks
        {"Q4_0", 4.5, false, [](size_t N, size_t K)
         { return TestTensorFactory::createQ4_0Random({N, K}); }},
        {"IQ4_NL", 4.5, false, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ4_NLRandom({N, K}); }},
        {"Q4_1", 5.0, false, [](size_t N, size_t K)
         { return TestTensorFactory::createQ4_1Random({N, K}); }},
        {"Q5_0", 5.5, false, [](size_t N, size_t K)
         { return TestTensorFactory::createQ5_0Random({N, K}); }},
        {"Q5_1", 6.0, false, [](size_t N, size_t K)
         { return TestTensorFactory::createQ5_1Random({N, K}); }},
        // Tier 1 super-block
        {"IQ4_XS", 4.5, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ4_XSRandom({N, K}); }},

        // Tier 2: K-quant super-blocks
        {"Q4_K", 4.5, true, [](size_t N, size_t K)
         { return TestTensorFactory::createQ4_KRandom({N, K}); }},
        {"Q5_K", 5.5, true, [](size_t N, size_t K)
         { return TestTensorFactory::createQ5_KRandom({N, K}); }},
        {"Q6_K", 6.6, true, [](size_t N, size_t K)
         { return TestTensorFactory::createQ6_KRandom({N, K}); }},
        {"Q3_K", 3.4, true, [](size_t N, size_t K)
         { return TestTensorFactory::createQ3_KRandom({N, K}); }},
        {"Q2_K", 2.6, true, [](size_t N, size_t K)
         { return TestTensorFactory::createQ2_KRandom({N, K}); }},

        // Tier 3: IQ grid-index super-blocks
        {"IQ3_S", 3.4, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ3_SRandom({N, K}); }},
        {"IQ3_XXS", 3.1, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ3_XXSRandom({N, K}); }},
        {"IQ2_S", 2.5, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ2_SRandom({N, K}); }},
        {"IQ2_XS", 2.3, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ2_XSRandom({N, K}); }},
        {"IQ2_XXS", 2.1, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ2_XXSRandom({N, K}); }},

        // Tier 4: IQ1 ultra-low-bit grid-index super-blocks
        {"IQ1_S", 1.6, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ1_SRandom({N, K}); }},
        {"IQ1_M", 1.9, true, [](size_t N, size_t K)
         { return TestTensorFactory::createIQ1_MRandom({N, K}); }},
        // Direct 8-bit path. The default ROCm packer still owns the INT8
        // scatter route, so focused NativeVNNI training explicitly packs this
        // through IINT8Unpackable below before dispatching codebook 19.
        {"Q8_0", 8.5, false, [](size_t N, size_t K)
         { return TestTensorFactory::createQ8_0Random({N, K}); }},
        {"Q8_1", 8.5, false, [](size_t N, size_t K)
         { return TestTensorFactory::createQ8_1Random({N, K}); }},
        {"Q8_K", 8.5, true, [](size_t N, size_t K)
         { return TestTensorFactory::createQ8_KRandom({N, K}); }},
    };

    /**
     * @brief Reconstruct one quantized tensor from caller-owned GGUF block bytes.
     *
     * Profiler-only collection does not compare tensor values, but it must still
     * exercise the real source-format upload and native-VNNI repack pipeline.
     * This small factory therefore preserves the exact concrete tensor type and
     * raw byte layout instead of substituting a prepared or execution-codebook
     * tensor that would bypass production preparation.
     *
     * @param type Runtime tensor format copied from the independently generated
     *             seed row.
     * @param shape Logical two-dimensional weight shape `[N, K]`.
     * @param raw_data Complete row-major GGUF block payload for `shape`.
     * @return A concrete quantized tensor that owns a copy of `raw_data`.
     * @throws std::runtime_error when a format is added to the profiler matrix
     *         without adding its matching concrete constructor here.
     */
    static std::unique_ptr<TensorBase> makeProfilerQuantizedTensor(
        TensorType type,
        const std::vector<size_t> &shape,
        const std::vector<uint8_t> &raw_data)
    {
        switch (type)
        {
        case TensorType::Q4_0:
            return std::make_unique<Q4_0Tensor>(shape, raw_data);
        case TensorType::IQ4_NL:
            return std::make_unique<IQ4_NLTensor>(shape, raw_data);
        case TensorType::Q4_1:
            return std::make_unique<Q4_1Tensor>(shape, raw_data);
        case TensorType::Q5_0:
            return std::make_unique<Q5_0Tensor>(shape, raw_data);
        case TensorType::Q5_1:
            return std::make_unique<Q5_1Tensor>(shape, raw_data);
        case TensorType::IQ4_XS:
            return std::make_unique<IQ4_XSTensor>(shape, raw_data);
        case TensorType::Q4_K:
            return std::make_unique<Q4_KTensor>(shape, raw_data);
        case TensorType::Q5_K:
            return std::make_unique<Q5_KTensor>(shape, raw_data);
        case TensorType::Q6_K:
            return std::make_unique<Q6_KTensor>(shape, raw_data);
        case TensorType::Q3_K:
            return std::make_unique<Q3_KTensor>(shape, raw_data);
        case TensorType::Q2_K:
            return std::make_unique<Q2_KTensor>(shape, raw_data);
        case TensorType::IQ3_S:
            return std::make_unique<IQ3_STensor>(shape, raw_data);
        case TensorType::IQ3_XXS:
            return std::make_unique<IQ3_XXSTensor>(shape, raw_data);
        case TensorType::IQ2_S:
            return std::make_unique<IQ2_STensor>(shape, raw_data);
        case TensorType::IQ2_XS:
            return std::make_unique<IQ2_XSTensor>(shape, raw_data);
        case TensorType::IQ2_XXS:
            return std::make_unique<IQ2_XXSTensor>(shape, raw_data);
        case TensorType::IQ1_S:
            return std::make_unique<IQ1_STensor>(shape, raw_data);
        case TensorType::IQ1_M:
            return std::make_unique<IQ1_MTensor>(shape, raw_data);
        case TensorType::Q8_0:
            return std::make_unique<Q8_0Tensor>(shape, raw_data);
        case TensorType::Q8_1:
            return std::make_unique<Q8_1Tensor>(shape, raw_data);
        case TensorType::Q8_K:
            return std::make_unique<Q8_KTensor>(shape, raw_data);
        default:
            throw std::runtime_error(
                "ROCm profiler fixture has no concrete constructor for tensor type " +
                std::to_string(static_cast<int>(type)));
        }
    }

    /**
     * @brief Build a value-independent profiler weight without full-matrix RNG.
     *
     * Kernel resource use, occupancy, dispatch geometry, and memory addresses do
     * not depend on quantized weight values. Generating an independent Gaussian
     * value for every logical weight therefore adds no profiler signal. It was
     * nevertheless responsible for roughly half of direct trainer CPU cycles on
     * the representative 512-request batch.
     *
     * One ordinary random row is still created through the format's established
     * test factory so every packed field, scale, minimum, and codebook index is
     * valid. The row's raw GGUF blocks are then tiled across N with an exponential
     * copy. The resulting tensor has exactly the requested byte footprint and is
     * passed through the same production upload and GPU repack path as canonical
     * timing. Canonical timing never calls this helper and retains independently
     * randomized full matrices for correctness and latency evidence.
     *
     * @param format Source-format descriptor selected by the exact profiler plan.
     * @param N Number of output rows in the projection weight.
     * @param K Number of logical values in each row.
     * @return A concrete source-format tensor with `N * K` logical values.
     */
    static std::unique_ptr<TensorBase> makeProfilerWeightFixture(
        const PerfFormatSpec &format,
        size_t N,
        size_t K)
    {
        auto seed_row = format.create(1, K);
        if (!seed_row || !seed_row->raw_data())
            throw std::runtime_error("ROCm profiler seed row has no raw data");

        const size_t row_bytes =
            llaminar2::test::quantizedRawBytesForGpuPreparedTest(*seed_row);
        if (row_bytes == 0)
            throw std::runtime_error("ROCm profiler seed row has zero raw bytes");
        if (N > std::numeric_limits<size_t>::max() / row_bytes)
            throw std::overflow_error("ROCm profiler fixture byte count overflow");

        std::vector<uint8_t> raw_data(N * row_bytes);
        std::memcpy(raw_data.data(), seed_row->raw_data(), row_bytes);

        // Double the initialized prefix on each iteration. This preserves an
        // exact valid row while reducing setup from O(N*K) random draws to a
        // bandwidth-bound O(raw bytes) copy.
        size_t initialized_rows = 1;
        while (initialized_rows < N)
        {
            const size_t copied_rows =
                std::min(initialized_rows, N - initialized_rows);
            std::memcpy(
                raw_data.data() + initialized_rows * row_bytes,
                raw_data.data(),
                copied_rows * row_bytes);
            initialized_rows += copied_rows;
        }

        return makeProfilerQuantizedTensor(
            seed_row->native_type(), {N, K}, raw_data);
    }

    using GEMVShape = native_vnni_dispatch::NativeVNNIShapeSpec;

    /**
     * @brief Shared model and held-out shape inventory for ROCm measurements.
     *
     * The legacy ROCm-only list was broader than CUDA but still had no frozen
     * development/sealed ownership. Loading the common manifest ensures every
     * backend trains and certifies identical aspect, work, and small-K cells.
     */
    static const std::vector<GEMVShape> &SHAPES =
        native_vnni_dispatch::nativeVnniShapeManifest();

    static std::string toLower(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    static std::string trim(std::string value)
    {
        const auto begin = value.find_first_not_of(" \t\n\r");
        if (begin == std::string::npos)
            return {};
        const auto end = value.find_last_not_of(" \t\n\r");
        return value.substr(begin, end - begin + 1);
    }

    static std::string getEnvString(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return {};
        return trim(raw);
    }

    static std::optional<int> getEnvInt(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return std::nullopt;
        return std::atoi(raw);
    }

    static std::set<std::string> getEnvCsvSet(const char *name)
    {
        std::set<std::string> values;
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return values;

        std::stringstream stream(raw);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            token = toLower(trim(token));
            if (!token.empty())
                values.insert(token);
        }
        return values;
    }

    static bool shouldRunName(const std::set<std::string> &filters, const std::string &name)
    {
        return filters.empty() || filters.count(toLower(name)) > 0;
    }

    /** Numerical role of one forceable ROCm decode trainer candidate. */
    enum class DecodeCandidateKind
    {
        FastExplicitKB,
        VerifierInheritSerialM1,
    };

    /** Production execution surfaces measured independently by the trainer. */
    enum class DecodeExecutionMode
    {
        Eager,
        GraphCaptured,
    };

    static const char *decodeExecutionModeName(DecodeExecutionMode mode)
    {
        return mode == DecodeExecutionMode::Eager ? "eager" : "graph_captured";
    }

    /**
     * @brief Stable candidate identity used by the ROCm decode trainer.
     *
     * An explicit KB is a real M=1 schedule choice.  Target-wave overrides are
     * deliberately absent: once KB is explicit they do not alter the launch
     * grid, arithmetic, or workspace and therefore are not distinct effective
     * candidates.  The grouped verifier has a separate candidate that inherits
     * the frozen serial-M1 split policy instead of pretending it may retune KB.
     */
    struct DecodeVariant
    {
        std::string name;
        DecodeCandidateKind kind = DecodeCandidateKind::FastExplicitKB;
        int kb = -1;

        [[nodiscard]] bool appliesToM(int M) const
        {
            return (M == 1 && kind == DecodeCandidateKind::FastExplicitKB) ||
                   (M >= 2 &&
                    kind == DecodeCandidateKind::VerifierInheritSerialM1);
        }
    };

    /**
     * @brief Return the registry identity represented by a trainer variant.
     *
     * Environment filters use short human-readable spellings such as `KB11`,
     * while profiler requests and generated dispatch policies use stable fully
     * qualified registry IDs. Keeping this conversion beside `DecodeVariant`
     * makes exact-plan matching independent of aliases accepted by the command
     * line parser.
     */
    static std::string effectiveDecodeCandidateId(const DecodeVariant &variant)
    {
        if (variant.kind == DecodeCandidateKind::VerifierInheritSerialM1)
            return "rocm.nvnni.decode.verifier.inherit_serial_m1";
        return "rocm.nvnni.decode.fast.kb" + std::to_string(variant.kb);
    }

    /** One immutable exact launch in a process-amortized ROCm profile. */
    struct ROCmProfilerBatchRequest
    {
        std::string request_id;
        std::string operation_kind;
        std::string source_format;
        std::string execution_mode;
        std::string shape_name;
        int m = 0;
        std::string projection_n_vector;
        int aggregate_n = 0;
        int k = 0;
        std::string effective_candidate_id;
        std::string output_path;
        bool consumed = false;
    };

    /**
     * @brief Own the collector-authored exact launch plan for one ROCm process.
     *
     * The ordinary trainer is intentionally a Cartesian sweep. Profiling must
     * be stricter: a broad environment union is permitted only to amortize HIP
     * context, fixture, and rocprofiler setup. This class intersects every loop
     * cell with the exact TSV rows, marks each row immediately before its one
     * selected-region launch, and rejects both duplicate and omitted launches.
     */
    class ROCmProfilerBatch
    {
    public:
        static constexpr size_t kMaximumRequestsPerProcess = 256;
        static constexpr size_t kMaximumGraphRequestsPerProcess = 256;

        explicit ROCmProfilerBatch(const std::string &path)
        {
            if (path.empty())
                return;
            std::ifstream input(path);
            if (!input)
                throw std::runtime_error(
                    "failed to open ROCm profiler batch plan: " + path);
            std::string line;
            if (!std::getline(input, line) || line != kHeader)
                throw std::runtime_error(
                    "ROCm profiler batch plan has an invalid schema header");

            std::set<std::string> request_ids;
            std::set<std::string> output_paths;
            while (std::getline(input, line))
            {
                if (line.empty())
                    continue;
                std::vector<std::string> fields;
                std::stringstream stream(line);
                std::string field;
                while (std::getline(stream, field, '\t'))
                    fields.push_back(field);
                if (fields.size() != 11)
                    throw std::runtime_error(
                        "ROCm profiler batch row must contain eleven TSV fields");
                ROCmProfilerBatchRequest request{
                    .request_id = fields[0],
                    .operation_kind = fields[1],
                    .source_format = fields[2],
                    .execution_mode = fields[3],
                    .shape_name = fields[4],
                    .m = parsePositive(fields[5], "M"),
                    .projection_n_vector = fields[6],
                    .aggregate_n = parsePositive(fields[7], "N"),
                    .k = parsePositive(fields[8], "K"),
                    .effective_candidate_id = fields[9],
                    .output_path = fields[10],
                };
                if (request.request_id.empty() ||
                    request.operation_kind.empty() ||
                    request.source_format.empty() ||
                    request.execution_mode.empty() ||
                    request.shape_name.empty() ||
                    request.projection_n_vector.empty() ||
                    request.effective_candidate_id.empty() ||
                    request.output_path.empty())
                {
                    throw std::runtime_error(
                        "ROCm profiler batch row contains an empty identity field");
                }
                if (!request_ids.insert(request.request_id).second)
                    throw std::runtime_error(
                        "ROCm profiler batch request IDs must be unique");
                if (!output_paths.insert(request.output_path).second)
                    throw std::runtime_error(
                        "ROCm profiler batch output paths must be unique");
                requests_.push_back(std::move(request));
            }
            if (requests_.empty())
                throw std::runtime_error("ROCm profiler batch plan is empty");
            const size_t graph_requests = static_cast<size_t>(std::count_if(
                requests_.begin(),
                requests_.end(),
                [](const ROCmProfilerBatchRequest &request)
                { return request.execution_mode == "graph_captured"; }));
            if (requests_.size() > kMaximumRequestsPerProcess ||
                graph_requests > kMaximumGraphRequestsPerProcess)
            {
                throw std::runtime_error(
                    "ROCm profiler batch exceeds the validated ROCm 7.1 "
                    "selected-region lifetime: requests=" +
                    std::to_string(requests_.size()) +
                    " graph_requests=" + std::to_string(graph_requests) +
                    " limits=" + std::to_string(kMaximumRequestsPerProcess) +
                    "/" + std::to_string(kMaximumGraphRequestsPerProcess));
            }
        }

        /** Return whether this invocation owns a collector-authored plan. */
        [[nodiscard]] bool enabled() const noexcept
        {
            return !requests_.empty();
        }

        /** Return the exact number of required selected-region launches. */
        [[nodiscard]] size_t size() const noexcept
        {
            return requests_.size();
        }

        /** Return whether any planned cell uses one prepared format/shape. */
        [[nodiscard]] bool containsShape(
            std::string_view source_format,
            std::string_view shape_name) const
        {
            return std::any_of(
                requests_.begin(), requests_.end(),
                [&](const ROCmProfilerBatchRequest &request)
                {
                    return request.source_format == source_format &&
                           request.shape_name == shape_name;
                });
        }

        /** Return exact candidate IDs requested for one physical work cell. */
        [[nodiscard]] std::set<std::string> candidateIds(
            std::string_view source_format,
            std::string_view shape_name,
            std::string_view execution_mode,
            int m,
            int n,
            int k) const
        {
            std::set<std::string> result;
            for (const ROCmProfilerBatchRequest &request : requests_)
            {
                if (matchesCell(
                        request, source_format, shape_name, execution_mode,
                        m, n, k))
                {
                    result.insert(request.effective_candidate_id);
                }
            }
            return result;
        }

        /** Claim one request immediately before its production launch. */
        ROCmProfilerBatchRequest *claim(
            std::string_view source_format,
            std::string_view shape_name,
            std::string_view execution_mode,
            int m,
            int n,
            int k,
            std::string_view effective_candidate_id)
        {
            ROCmProfilerBatchRequest *result = nullptr;
            for (ROCmProfilerBatchRequest &request : requests_)
            {
                if (!matchesCell(
                        request, source_format, shape_name, execution_mode,
                        m, n, k) ||
                    request.effective_candidate_id != effective_candidate_id)
                {
                    continue;
                }
                if (result)
                    throw std::runtime_error(
                        "ROCm profiler batch has duplicate exact launch identities");
                result = &request;
            }
            if (!result)
                return nullptr;
            if (result->consumed)
                throw std::runtime_error(
                    "ROCm profiler batch request was claimed more than once: " +
                    result->request_id);
            result->consumed = true;
            return result;
        }

        /** Fail a successful process that omitted any planned launch. */
        void requireComplete() const
        {
            for (const ROCmProfilerBatchRequest &request : requests_)
            {
                if (!request.consumed)
                    throw std::runtime_error(
                        "ROCm profiler batch omitted request " +
                        request.request_id);
            }
        }

    private:
        static constexpr std::string_view kHeader =
            "request_id\toperation_kind\tsource_format\texecution_mode\t"
            "shape_name\tm\tprojection_n_vector\taggregate_n\tk\t"
            "effective_candidate_id\toutput_path";

        static int parsePositive(const std::string &value, const char *name)
        {
            size_t consumed = 0;
            const int parsed = std::stoi(value, &consumed);
            if (consumed != value.size() || parsed <= 0)
                throw std::runtime_error(
                    std::string("ROCm profiler batch has invalid ") + name);
            return parsed;
        }

        static bool matchesCell(
            const ROCmProfilerBatchRequest &request,
            std::string_view source_format,
            std::string_view shape_name,
            std::string_view execution_mode,
            int m,
            int n,
            int k)
        {
            return request.operation_kind == "NativeVNNIDecodeProjection" &&
                   request.source_format == source_format &&
                   request.execution_mode == execution_mode &&
                   request.m == m && request.aggregate_n == n &&
                   request.k == k &&
                   request.projection_n_vector == std::to_string(n) &&
                   request.shape_name == shape_name;
        }

        std::vector<ROCmProfilerBatchRequest> requests_;
    };

#ifdef HAVE_ROCM
    /** Scope one normalized candidate route around a measured production call. */
    class DecodeTuningGuard
    {
    public:
        explicit DecodeTuningGuard(const DecodeVariant &variant)
        {
            if (variant.kind == DecodeCandidateKind::VerifierInheritSerialM1)
                rocmGemv_native_vnni_reset_tuning_overrides();
            else
                rocmGemv_native_vnni_set_tuning_overrides(variant.kb, -1);
        }

        ~DecodeTuningGuard()
        {
            rocmGemv_native_vnni_reset_tuning_overrides();
        }

        DecodeTuningGuard(const DecodeTuningGuard &) = delete;
        DecodeTuningGuard &operator=(const DecodeTuningGuard &) = delete;
    };

    /**
     * @brief Own one production-contract ROCm capture and output publication.
     *
     * A raw HIP stream capture is insufficient for a public tensor kernel:
     * the kernel publishes each device write, but recording graph nodes has not
     * executed those writes yet. Production capture therefore records tensor
     * dependencies in a frozen ledger and publishes the output only after each
     * replay. The trainer uses that same lifecycle so graph measurements prove
     * an ordering contract that inference can execute unchanged.
     */
    class CapturedDecodeLaunch
    {
    public:
        CapturedDecodeLaunch() = default;

        ~CapturedDecodeLaunch()
        {
            reset();
        }

        /**
         * @brief Record one declared projection and freeze its executable graph.
         *
         * @param stream Exact non-default HIP stream used for capture/replay.
         * @param device ROCm device that owns all declared tensors.
         * @param inputs Stable graph-external inputs joined before capture.
         * @param outputs Stable outputs published after every graph replay.
         * @param launch Public production entrypoint that records device work.
         * @return True only when capture and instantiation both succeed.
         */
        template <typename Launch>
        bool capture(
            hipStream_t stream,
            DeviceId device,
            const std::vector<const TensorBase *> &inputs,
            const std::vector<TensorBase *> &outputs,
            Launch &&launch)
        {
            reset();
            if (!stream || !device.is_gpu() || outputs.empty())
                return false;

            std::vector<const TensorBase *> declared_outputs;
            declared_outputs.reserve(outputs.size());
            for (TensorBase *output : outputs)
            {
                if (!output)
                    return false;
                declared_outputs.push_back(output);
            }

            GraphCaptureDependencyLedger ledger(
                device,
                reinterpret_cast<void *>(stream),
                {GraphCaptureDependencyLedger::StagePlan{
                    .stage_identity = this,
                    .stage_name = "ROCm NativeVNNI trainer projection",
                    .external_inputs = inputs,
                    .internal_inputs = {},
                    .outputs = std::move(declared_outputs),
                }},
                "ROCm NativeVNNI trainer captured launch");

            if (hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal) != hipSuccess)
                return false;

            bool launch_ok = false;
            std::exception_ptr launch_error;
            try
            {
                GraphCaptureGuard guard(&ledger);
                ScopedGraphCaptureStage stage(this);
                launch_ok = launch();
                if (launch_ok)
                    stage.complete();
            }
            catch (...)
            {
                launch_error = std::current_exception();
            }

            const hipError_t end_status = hipStreamEndCapture(stream, &graph_);
            if (launch_error)
            {
                reset();
                std::rethrow_exception(launch_error);
            }
            if (!launch_ok || end_status != hipSuccess || graph_ == nullptr)
            {
                reset();
                return false;
            }
            if (hipGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0) != hipSuccess ||
                executable_ == nullptr)
            {
                reset();
                return false;
            }
            device_ = device;
            outputs_ = outputs;
            return true;
        }

        /**
         * @brief Replay and publish one completed graph generation in order.
         *
         * Publication records completion events after `hipGraphLaunch()` on
         * this exact stream. It performs no synchronization or host transfer.
         */
        bool launch(hipStream_t stream) const
        {
            if (!stream || executable_ == nullptr || !device_.has_value() ||
                hipGraphLaunch(executable_, stream) != hipSuccess)
            {
                return false;
            }
            for (TensorBase *output : outputs_)
            {
                TransferEngine::publishDeviceWrite(
                    output,
                    *device_,
                    reinterpret_cast<void *>(stream));
            }
            return true;
        }

        void reset()
        {
            if (executable_)
                (void)hipGraphExecDestroy(executable_);
            if (graph_)
                (void)hipGraphDestroy(graph_);
            executable_ = nullptr;
            graph_ = nullptr;
            outputs_.clear();
            device_.reset();
        }

        CapturedDecodeLaunch(const CapturedDecodeLaunch &) = delete;
        CapturedDecodeLaunch &operator=(const CapturedDecodeLaunch &) = delete;

    private:
        hipGraph_t graph_ = nullptr;
        hipGraphExec_t executable_ = nullptr;
        std::vector<TensorBase *> outputs_;
        std::optional<DeviceId> device_;
    };
#endif

    /** Parse a canonical candidate or a reviewed legacy KB/TW alias. */
    static DecodeVariant parseDecodeVariantToken(const std::string &token)
    {
        const std::string value = toLower(trim(token));
        if (value.empty() || value == "auto")
            throw std::runtime_error(
                "AUTO is resolver behavior, not a forceable ROCm decode candidate");

        std::string compact;
        compact.reserve(value.size());
        for (unsigned char c : value)
        {
            if (std::isalnum(c))
                compact.push_back(static_cast<char>(c));
        }

        if (compact == "inheritserialm1" || compact == "serialm1")
        {
            return DecodeVariant{
                "INHERIT_SERIAL_M1",
                DecodeCandidateKind::VerifierInheritSerialM1,
                -1};
        }

        const auto kb_pos = compact.find("kb");
        const auto tw_pos = compact.find("tw");
        if (kb_pos != 0)
            throw std::runtime_error("Invalid ROCm NativeVNNI decode variant: " + token);

        const std::string kb_text =
            tw_pos == std::string::npos
                ? compact.substr(kb_pos + 2)
                : compact.substr(kb_pos + 2, tw_pos - (kb_pos + 2));
        const int kb = std::atoi(kb_text.c_str());
        if (kb <= 0 || kb > 64)
            throw std::runtime_error("Invalid ROCm NativeVNNI decode variant values: " + token);
        if (tw_pos != std::string::npos &&
            std::atoi(compact.substr(tw_pos + 2).c_str()) <= 0)
        {
            throw std::runtime_error("Invalid nominal target-wave alias: " + token);
        }

        return DecodeVariant{
            "KB" + std::to_string(kb),
            DecodeCandidateKind::FastExplicitKB,
            kb};
    }

    /** Return the complete normalized M=1 and grouped-verifier inventory. */
    static std::vector<DecodeVariant> getDecodeVariants()
    {
        const char *raw = std::getenv("LLAMINAR_ROCM_NVNNI_DECODE_VARIANTS");
        std::vector<DecodeVariant> variants;
        if (raw && *raw)
        {
            std::stringstream stream(raw);
            std::string token;
            while (std::getline(stream, token, ','))
            {
                token = trim(token);
                if (!token.empty())
                    variants.push_back(parseDecodeVariantToken(token));
            }
        }
        else
        {
            /*
             * Every graph-safe split count is forceable.  Unsupported values
             * for a short K dimension remain measured route failures in the
             * corpus instead of silently disappearing from the matrix.
             */
            for (int kb = 1; kb <= 64; ++kb)
            {
                variants.push_back(DecodeVariant{
                    "KB" + std::to_string(kb),
                    DecodeCandidateKind::FastExplicitKB,
                    kb});
            }
            variants.push_back(DecodeVariant{
                "INHERIT_SERIAL_M1",
                DecodeCandidateKind::VerifierInheritSerialM1,
                -1});
        }
        if (variants.empty())
            throw std::runtime_error("ROCm decode trainer selected no explicit candidates");

        std::vector<DecodeVariant> normalized;
        normalized.reserve(variants.size());
        for (const auto &variant : variants)
        {
            const bool duplicate = std::any_of(
                normalized.begin(),
                normalized.end(),
                [&](const DecodeVariant &existing)
                {
                    return existing.kind == variant.kind &&
                           existing.kb == variant.kb;
                });
            if (!duplicate)
                normalized.push_back(variant);
        }
        return normalized;
    }

    /** Parse the eager/captured production surfaces selected for one sweep. */
    static std::vector<DecodeExecutionMode> getDecodeExecutionModes()
    {
        const char *raw = std::getenv("LLAMINAR_ROCM_NVNNI_DECODE_EXECUTION_MODES");
        const std::string spec = (raw && *raw) ? raw : "eager,graph_captured";
        std::vector<DecodeExecutionMode> modes;
        std::stringstream stream(spec);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            token = toLower(trim(token));
            DecodeExecutionMode mode;
            if (token == "eager")
                mode = DecodeExecutionMode::Eager;
            else if (token == "graph" || token == "captured" ||
                     token == "graph_captured")
                mode = DecodeExecutionMode::GraphCaptured;
            else
                throw std::runtime_error(
                    "Invalid ROCm NativeVNNI decode execution mode: " + token);
            if (std::find(modes.begin(), modes.end(), mode) == modes.end())
                modes.push_back(mode);
        }
        if (modes.empty())
            throw std::runtime_error("ROCm decode trainer selected no execution modes");
        return modes;
    }

    static std::vector<int> getDecodeMValues()
    {
        const char *raw = std::getenv("LLAMINAR_ROCM_NVNNI_DECODE_M");
        const std::string spec =
            (raw && *raw)
                ? raw
                : "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31";

        std::vector<int> values;
        std::stringstream stream(spec);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            token = trim(token);
            if (token.empty())
                continue;
            const int m = std::atoi(token.c_str());
            if (m <= 0)
                throw std::runtime_error("Invalid ROCm NativeVNNI decode M value: " + token);
            values.push_back(m);
        }
        if (values.empty())
            throw std::runtime_error("ROCm decode trainer selected no M values");
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        return values;
    }

    static double nativePackedWeightBytes(const ROCmPackedWeights &packed)
    {
        return static_cast<double>(packed.native_vnni_payload.size()) +
               static_cast<double>(packed.native_vnni_scales.size() * sizeof(uint16_t)) +
               static_cast<double>(packed.native_vnni_mins.size() * sizeof(uint16_t)) +
               static_cast<double>(packed.native_vnni_emins.size() * sizeof(uint32_t));
    }

    static const NativeVnniFormatInfo &requireNativeVnniInfo(const TensorBase *weights, const std::string &format_name)
    {
        const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(weights);
        const NativeVnniFormatInfo *info = unpackable ? unpackable->vnniFormatInfo() : nullptr;
        if (!info)
            throw std::runtime_error("ROCm NativeVNNI decode format " + format_name + " did not expose vnniFormatInfo()");
        return *info;
    }

    /**
     * @brief Populate NativeVNNI host fields for formats not emitted by the default packer.
     *
     * Q8_0 normally uses ROCm's INT8 scatter GEMV path, but the generated
     * verifier-row dispatch table has to cover codebook 19 as well.  This
     * helper uses the tensor's native packing interface to create the same
     * payload/scales/mins arrays that production NativeVNNI descriptors expose.
     */
    static bool ensureNativeVNNIPayloadForDecodeTrainer(
        TensorBase *weights,
        ROCmPackedWeights &packed,
        int N,
        int K,
        const std::string &format_name)
    {
        if (!packed.native_vnni_payload.empty())
            return true;
        if (!weights || N <= 0 || K <= 0 || (K % 32) != 0)
            return false;

        const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(weights);
        const NativeVnniFormatInfo *info = unpackable ? unpackable->vnniFormatInfo() : nullptr;
        if (!info || info->payload_bytes <= 0)
        {
            std::fprintf(stderr,
                         "[ROCmNativeVNNI][DECODE][TRAINER][ERROR] %s does not expose packable NativeVNNI metadata\n",
                         format_name.c_str());
            return false;
        }

        const int blocks_per_row = K / 32;
        const size_t total_blocks =
            static_cast<size_t>(N) * static_cast<size_t>(blocks_per_row);
        packed.native_vnni_payload.assign(
            total_blocks * static_cast<size_t>(info->payload_bytes), uint8_t{0});
        packed.native_vnni_scales.assign(total_blocks, uint16_t{0});
        packed.native_vnni_mins.clear();
        packed.native_vnni_emins.clear();
        if (info->is_asymmetric)
            packed.native_vnni_mins.assign(total_blocks, uint16_t{0});
        if (info->has_emins)
            packed.native_vnni_emins.assign(total_blocks, uint32_t{0});
        // GPU preparation deliberately erases the source-layout distinction
        // between Q8_0, Q8_1, and Q8_K after packVnniBlock() has converted each
        // source block into the common signed-INT8 payload plus FP16 scale.
        // Publish the same execution identity as WeightManager and
        // ROCmWeightPacker; retaining source IDs 20/21 here asks the production
        // launcher for codebooks that cannot exist after device preparation.
        packed.native_vnni_codebook_id =
            canonicalDeviceVnniCodebookId(info->codebook_id);
        packed.native_vnni_blocks_per_row = static_cast<uint32_t>(blocks_per_row);
        packed.N = N;
        packed.K = K;

        VnniPackContext ctx{};
        ctx.raw_bytes = reinterpret_cast<const uint8_t *>(weights->raw_data());
        ctx.N = N;
        ctx.K = K;
        ctx.blocks_per_row = blocks_per_row;
        ctx.payload_bytes = info->payload_bytes;
        ctx.payload_array = packed.native_vnni_payload.data();
        ctx.scales_array = packed.native_vnni_scales.data();
        ctx.mins_array = packed.native_vnni_mins.empty()
                             ? nullptr
                             : packed.native_vnni_mins.data();
        ctx.emins_array = packed.native_vnni_emins.empty()
                              ? nullptr
                              : packed.native_vnni_emins.data();

        for (int n = 0; n < N; ++n)
            for (int b = 0; b < blocks_per_row; ++b)
                unpackable->packVnniBlock(ctx, n, b);

        return true;
    }

    // =============================================================================
    // Benchmark result
    // =============================================================================

    struct BenchResult
    {
        std::string format_name;
        double bpw;
        std::string shape_name;
        int M;
        int N, K;

        double min_us;
        double mean_us;
        double stddev_us;

        double weight_bytes;  // native-VNNI payload + scales + mins bytes
        double eff_bw_gbps;   // effective bandwidth at min time
        double bw_efficiency; // % of HBM peak

        // INT8 reference comparison (populated when reference available)
        double int8_min_us = 0.0;         // INT8 VNNI reference min time for same shape
        double speedup_vs_int8 = 0.0;     // int8_min_us / min_us (>1 = faster than INT8)
        double theoretical_speedup = 0.0; // 8.0 / bpw (expected from bandwidth savings)
        double kernel_efficiency = 0.0;   // (speedup_vs_int8 / theoretical_speedup) * 100%

        // Correctness (GPU-based HipBLAS reference)
        float cosine_sim = 0.0f;
        bool correctness_pass = false;
    };

    // =============================================================================
    // Statistics helper
    // =============================================================================

    static void computeStats(const std::vector<double> &times_us,
                             double &mean, double &min_val,
                             double &max_val, double &stddev)
    {
        mean = std::accumulate(times_us.begin(), times_us.end(), 0.0) /
               static_cast<double>(times_us.size());
        min_val = *std::min_element(times_us.begin(), times_us.end());
        max_val = *std::max_element(times_us.begin(), times_us.end());

        double sq_sum = 0.0;
        for (double t : times_us)
            sq_sum += (t - mean) * (t - mean);
        stddev = std::sqrt(sq_sum / static_cast<double>(times_us.size()));
    }

#ifdef HAVE_ROCM
    /**
     * @test Keep ROCm generated decode dispatch within a few host nanoseconds.
     *
     * This host-only test invokes the exact cached policy query used by the
     * production NativeVNNI launcher. It never initializes HIP. The repeated
     * key models graph construction or eager launch; graph replay itself does
     * not call the host selector. Rotating production geometries model graph
     * bucket capture and mixed-projection eager execution. A loop-only control
     * carries identical indexing and compiler-fence overhead so the reported
     * median isolates policy lookup cost.
     *
     * Codebook 5 is intentionally used until the regenerated ROCm policy has
     * certified codebook-19 totality. A generated miss remains a hard miss;
     * this benchmark must never manufacture a Q8 policy to make itself pass.
     */
    TEST(ROCmNativeVNNIPerfOffline, GeneratedDecodeDispatchCacheLatency)
    {
        struct Query
        {
            int n;
            int k;
        };

        constexpr uint8_t codebook_id = 5;
        constexpr int iterations = 100000;
        constexpr int samples = 7;
        constexpr std::array<Query, 1> hot = {{{512, 2048}}};
        constexpr std::array<Query, 8> working_set = {{
            {512, 2048},
            {2048, 512},
            {2048, 2048},
            {4096, 2048},
            {8192, 2048},
            {1024, 5120},
            {6144, 5120},
            {5120, 17408},
        }};

        for (const Query &query : working_set)
        {
            int kb = 0;
            int target_waves = 0;
            ASSERT_TRUE(rocmGemv_native_vnni_query_generated_config(
                codebook_id,
                1,
                query.n,
                query.k,
                &kb,
                &target_waves))
                << "missing ROCm NativeVNNI policy for "
                << query.n << 'x' << query.k;
        }

        const auto measure = [&](const auto &queries)
        {
            std::array<double, samples> net_ns{};
            std::array<int, queries.size()> ns{};
            std::array<int, queries.size()> ks{};
            for (size_t index = 0; index < queries.size(); ++index)
            {
                ns[index] = queries[index].n;
                ks[index] = queries[index].k;
            }

            for (int sample = 0; sample < samples; ++sample)
            {
                uint64_t checksum = 0;
                net_ns[static_cast<size_t>(sample)] =
                    rocmGemv_native_vnni_measure_generated_config_ns(
                        codebook_id,
                        1,
                        ns.data(),
                        ks.data(),
                        static_cast<int>(queries.size()),
                        iterations,
                        &checksum);
                EXPECT_NE(checksum, 0u);
            }

            std::sort(net_ns.begin(), net_ns.end());
            return net_ns[net_ns.size() / 2];
        };

        const double hot_ns = measure(hot);
        const double working_set_ns = measure(working_set);

        EXPECT_LT(hot_ns, 10.0);
        EXPECT_LT(working_set_ns, 20.0);
        std::cout
            << "[ROCmNativeVNNI][DISPATCH_CACHE_LATENCY] hot_ns="
            << hot_ns
            << " working_set_ns=" << working_set_ns
            << std::endl;
    }

    /**
     * @test Prove generated M=1 policy totality for every execution codebook.
     *
     * Exact overlays are additive and therefore cannot establish generic
     * dispatch coverage. Probe unseen valid geometries in every aspect regime
     * through the same host-only production query. Q8_0, Q8_1, and Q8_K all
     * normalize to execution codebook 19, so that one runtime identity must be
     * covered just as completely as every compressed codebook.
     */
    TEST(ROCmNativeVNNIPerfOffline, GeneratedDecodeDispatchIsCodebookAndGeometryTotal)
    {
        constexpr std::array<uint8_t, 16> codebooks = {
            0, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 19,
        };
        struct Query
        {
            int n;
            int k;
        };
        constexpr std::array<Query, 4> unseen_geometries = {{
            {64, 1024},
            {1024, 1024},
            {4096, 1024},
            {32768, 1024},
        }};

        for (uint8_t codebook : codebooks)
        {
            for (const Query &query : unseen_geometries)
            {
                int kb = 0;
                int target_waves = 0;
                ASSERT_TRUE(rocmGemv_native_vnni_query_generated_config(
                    codebook,
                    1,
                    query.n,
                    query.k,
                    &kb,
                    &target_waves))
                    << "missing generated ROCm policy for CB="
                    << static_cast<int>(codebook)
                    << " N=" << query.n
                    << " K=" << query.k;
                EXPECT_GT(kb, 0);
                EXPECT_GT(target_waves, 0);
            }
        }

        int kb = 0;
        int target_waves = 0;
        EXPECT_FALSE(rocmGemv_native_vnni_query_generated_config(
            18, 1, 1024, 1024, &kb, &target_waves));
    }
#endif

    // =============================================================================
    // Test fixture
    // =============================================================================

    class NativeVNNIPerfTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
#ifdef HAVE_ROCM
            int device_count = 0;
            hipError_t err = hipGetDeviceCount(&device_count);
            has_device_ = (err == hipSuccess && device_count > 0);
            if (has_device_)
            {
                NUM_GPUS = std::min(device_count, 3);
                (void)hipSetDevice(0);
                hipDeviceProp_t props;
                if (hipGetDeviceProperties(&props, 0) == hipSuccess)
                    device_name_ = std::string(props.name) + " (" + props.gcnArchName + ")";
                else
                    device_name_ = "ROCm device 0";
            }
#else
            has_device_ = false;
#endif
        }

        void TearDown() override
        {
#ifdef HAVE_ROCM
            destroyAllHipBLAS();
#endif
        }

        bool has_device_ = false;
        std::string device_name_;

#ifdef HAVE_ROCM
        /// Time a GEMV kernel on a specific device. Returns sorted timing vector in μs.
        static std::vector<double> timeKernel(ROCmQuantisedGemmKernel &kernel,
                                              TensorBase *input, TensorBase *output,
                                              int M, int N, int K, int device_id,
                                              hipStream_t stream)
        {
            (void)hipSetDevice(device_id);

            // Warmup
            for (int i = 0; i < WARMUP_RUNS; ++i)
                kernel.multiply_tensor(input, output, M, N, K);
            (void)hipStreamSynchronize(stream);

            // Timed runs
            hipEvent_t start = nullptr, stop = nullptr;
            (void)hipEventCreate(&start);
            (void)hipEventCreate(&stop);

            std::vector<double> times_us;
            times_us.reserve(BENCH_RUNS);

            for (int i = 0; i < BENCH_RUNS; ++i)
            {
                (void)hipStreamSynchronize(stream);
                (void)hipEventRecord(start, stream);
                kernel.multiply_tensor(input, output, M, N, K);
                (void)hipEventRecord(stop, stream);
                (void)hipEventSynchronize(stop);

                float ms = 0.0f;
                (void)hipEventElapsedTime(&ms, start, stop);
                times_us.push_back(static_cast<double>(ms) * 1000.0);
            }

            (void)hipEventDestroy(start);
            (void)hipEventDestroy(stop);

            std::sort(times_us.begin(), times_us.end());
            return times_us;
        }

        /// Benchmark INT8 VNNI reference (Q8_0 → INT8 scatter GEMV) for a shape.
        /// Thread-safe: creates all resources locally.
        /// Returns min kernel time in μs.
        static double benchmarkINT8Reference(const GEMVShape &shape, int M, int device_id)
        {
            (void)hipSetDevice(device_id);

            // Create Q8_0 weights — packs to INT8 VNNI (no native-VNNI payload)
            auto weights = TestTensorFactory::createQ8_0Random(
                {static_cast<size_t>(shape.N), static_cast<size_t>(shape.K)});
            if (!weights)
                return 0.0;

            ROCmPackedWeights packed;
            if (!packWeightsToROCm(weights.get(), packed))
                return 0.0;

            ROCmQuantisedGemmKernel kernel(&packed, device_id);
            hipStream_t stream = nullptr;
            if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess || !stream)
                return 0.0;
            kernel.setGPUStream(stream);
            auto reqs = kernel.getWorkspaceRequirements(M, shape.N, shape.K);
            const size_t budget = reqs.total_bytes_with_alignment() + (4 * 1024 * 1024);
            auto workspace = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(device_id), budget);
            if (!workspace->allocate(reqs))
            {
                (void)hipStreamDestroy(stream);
                return 0.0;
            }
            kernel.bindWorkspace(workspace.get());

            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(M), static_cast<size_t>(shape.K)});
            auto output = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(shape.N)});
            if (!input->ensureOnDevice(DeviceId::rocm(device_id)))
            {
                kernel.unbindWorkspace();
                (void)hipStreamDestroy(stream);
                return 0.0;
            }
            if (!output->allocateOnDevice(DeviceId::rocm(device_id)))
            {
                kernel.unbindWorkspace();
                (void)hipStreamDestroy(stream);
                return 0.0;
            }

            auto times = timeKernel(kernel, input.get(), output.get(),
                                    M, shape.N, shape.K, device_id, stream);
            kernel.unbindWorkspace();
            (void)hipStreamDestroy(stream);
            return times.empty() ? 0.0 : times.front();
        }

        /// Run a single format+shape benchmark on a specific device.
        /// Thread-safe: does not use gtest assertions internally.
        static BenchResult benchmarkFormat(const PerfFormatSpec &fmt,
                                           const GEMVShape &shape,
                                           int M,
                                           double int8_ref_us,
                                           TensorBase *weights,
                                           ROCmPackedWeights *packed_weights,
                                           double packed_weight_bytes,
                                           const GpuWeightsCache *gpu_weights,
                                           TensorBase *shared_input,
                                           int device_id)
        {
            (void)hipSetDevice(device_id);

            BenchResult result{};
            result.format_name = fmt.name;
            result.bpw = fmt.bpw;
            result.shape_name = shape.name;
            result.M = M;
            result.N = shape.N;
            result.K = shape.K;

            if (!weights || !packed_weights)
                return result;

            const bool has_host_native_payload =
                !packed_weights->native_vnni_payload.empty() &&
                !packed_weights->native_vnni_scales.empty();
            const bool has_device_native_payload =
                packed_weights->d_native_vnni_payload != nullptr &&
                packed_weights->d_native_vnni_scales != nullptr;
            if (!has_host_native_payload && !has_device_native_payload)
                return result;

            // Calculate weight bytes (native-VNNI payload + scales + mins)
            result.weight_bytes = packed_weight_bytes > 0.0
                                      ? packed_weight_bytes
                                      : nativePackedWeightBytes(*packed_weights);

            // 2. Create kernel + workspace
            ROCmQuantisedGemmKernel kernel(packed_weights, device_id);
            hipStream_t stream = nullptr;
            if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess || !stream)
            {
                std::fprintf(stderr,
                             "[ROCmNativeVNNI][DECODE][TRAINER][ERROR] stream create failed for %s/%s M=%d\n",
                             fmt.name.c_str(), shape.name.c_str(), M);
                return result;
            }
            kernel.setGPUStream(stream);
            auto reqs = kernel.getWorkspaceRequirements(M, shape.N, shape.K);
            const size_t budget = reqs.total_bytes_with_alignment() + (4 * 1024 * 1024);
            auto workspace = std::make_unique<DeviceWorkspaceManager>(
                DeviceId::rocm(device_id), budget);
            if (!workspace->allocate(reqs))
            {
                std::fprintf(stderr,
                             "[ROCmNativeVNNI][DECODE][TRAINER][ERROR] workspace allocation failed for %s/%s M=%d bytes=%zu\n",
                             fmt.name.c_str(),
                             shape.name.c_str(),
                             M,
                             reqs.total_bytes_with_alignment());
                (void)hipStreamDestroy(stream);
                return result;
            }
            kernel.bindWorkspace(workspace.get());

            // 3. Reuse a per-case input when the caller needs variant-vs-variant
            // equivalence, otherwise create a local one for the FP32 reference path.
            std::unique_ptr<TensorBase> owned_input;
            TensorBase *input = shared_input;
            if (!input)
            {
                owned_input = TestTensorFactory::createFP32Random(
                    {static_cast<size_t>(M), static_cast<size_t>(shape.K)});
                input = owned_input.get();
            }
            auto output = TestTensorFactory::createFP32(
                {static_cast<size_t>(M), static_cast<size_t>(shape.N)});
            if (!input->ensureOnDevice(DeviceId::rocm(device_id)))
            {
                std::fprintf(stderr,
                             "[ROCmNativeVNNI][DECODE][TRAINER][ERROR] input upload failed for %s/%s M=%d\n",
                             fmt.name.c_str(), shape.name.c_str(), M);
                kernel.unbindWorkspace();
                (void)hipStreamDestroy(stream);
                return result;
            }
            if (!output->allocateOnDevice(DeviceId::rocm(device_id)))
            {
                std::fprintf(stderr,
                             "[ROCmNativeVNNI][DECODE][TRAINER][ERROR] output allocation failed for %s/%s M=%d\n",
                             fmt.name.c_str(), shape.name.c_str(), M);
                kernel.unbindWorkspace();
                (void)hipStreamDestroy(stream);
                return result;
            }

            // 4. Correctness: GPU-based FP32 reference via hipBLAS
            {
                if (!kernel.multiply_tensor(input, output.get(), M, shape.N, shape.K))
                {
                    std::fprintf(stderr,
                                 "[ROCmNativeVNNI][DECODE][TRAINER][ERROR] multiply failed for %s/%s M=%d\n",
                                 fmt.name.c_str(), shape.name.c_str(), M);
                    kernel.unbindWorkspace();
                    (void)hipStreamDestroy(stream);
                    return result;
                }
                (void)hipStreamSynchronize(stream);
                TransferEngine::publishCurrentDeviceWrite(output, stream);

                const size_t out_elems =
                    static_cast<size_t>(M) * static_cast<size_t>(shape.N);
                const float *d_gpu_output = reinterpret_cast<const float *>(
                    dynamic_cast<FP32Tensor *>(output.get())->gpu_data_ptr());

                if (gpu_weights && gpu_weights->d_weights)
                {
                    auto *in_fp32 = dynamic_cast<FP32Tensor *>(input);
                    const float *d_input = reinterpret_cast<const float *>(
                        in_fp32->gpu_data_ptr());
                    if (d_input)
                    {
                        float *d_ref_output = nullptr;
                        auto hip_err = hipMalloc(&d_ref_output, out_elems * sizeof(float));
                        if (hip_err == hipSuccess)
                        {
                            // hipBLAS GEMM with M=1 is effectively GEMV
                            bool gemm_ok = gpuReferenceFP32Gemm(
                                d_input, gpu_weights->d_weights,
                                d_ref_output, M, shape.N, shape.K, device_id);
                            (void)hipDeviceSynchronize();

                            if (gemm_ok)
                            {
                                result.cosine_sim = gpuCosineSimilarity(
                                    d_gpu_output, d_ref_output, out_elems, device_id);
                                result.correctness_pass =
                                    (result.cosine_sim >= cosine_gate_for(result.format_name));
                            }
                            (void)hipFree(d_ref_output);
                        }
                    }
                }

                // Re-upload output for timed runs
                output->ensureOnDevice(DeviceId::rocm(device_id));
            }

            // 5. Timed runs
            auto times = timeKernel(kernel, input, output.get(),
                                    M, shape.N, shape.K, device_id, stream);

            double max_us;
            computeStats(times, result.mean_us, result.min_us, max_us, result.stddev_us);

            // 6. Compute metrics
            result.eff_bw_gbps = (result.weight_bytes / (result.min_us * 1e-6)) / 1e9;
            result.bw_efficiency = (result.eff_bw_gbps / HBM2_PEAK_GBPS) * 100.0;
            result.theoretical_speedup = 8.0 / result.bpw;

            if (int8_ref_us > 0.0 && result.min_us > 0.0)
            {
                result.int8_min_us = int8_ref_us;
                result.speedup_vs_int8 = int8_ref_us / result.min_us;
                result.kernel_efficiency =
                    (result.speedup_vs_int8 / result.theoretical_speedup) * 100.0;
            }

            kernel.unbindWorkspace();
            (void)hipStreamDestroy(stream);
            return result;
        }

        static BenchResult benchmarkFormat(const PerfFormatSpec &fmt,
                                           const GEMVShape &shape,
                                           int M,
                                           double int8_ref_us,
                                           TensorBase *weights,
                                           const GpuWeightsCache *gpu_weights,
                                           int device_id)
        {
            ROCmPackedWeights packed;
            if (!weights || !packWeightsToROCm(weights, packed))
            {
                BenchResult result{};
                result.format_name = fmt.name;
                result.bpw = fmt.bpw;
                result.shape_name = shape.name;
                result.M = M;
                result.N = shape.N;
                result.K = shape.K;
                return result;
            }
            if (!ensureNativeVNNIPayloadForDecodeTrainer(
                    weights, packed, shape.N, shape.K, fmt.name))
            {
                BenchResult result{};
                result.format_name = fmt.name;
                result.bpw = fmt.bpw;
                result.shape_name = shape.name;
                result.M = M;
                result.N = shape.N;
                result.K = shape.K;
                return result;
            }
            return benchmarkFormat(fmt,
                                   shape,
                                   M,
                                   int8_ref_us,
                                   weights,
                                   &packed,
                                   nativePackedWeightBytes(packed),
                                   gpu_weights,
                                   nullptr,
                                   device_id);
        }

#endif
    };

#ifdef HAVE_ROCM
    /**
     * @brief Restore one environment variable after a trainer-only override.
     */
    class ScopedTrainerEnvironment
    {
    public:
        ScopedTrainerEnvironment(const char *name, const char *value)
            : name_(name)
        {
            if (const char *previous = std::getenv(name))
            {
                had_previous_ = true;
                previous_ = previous;
            }
            (void)setenv(name_.c_str(), value, 1);
        }

        ~ScopedTrainerEnvironment()
        {
            if (had_previous_)
                (void)setenv(name_.c_str(), previous_.c_str(), 1);
            else
                (void)unsetenv(name_.c_str());
        }

        ScopedTrainerEnvironment(const ScopedTrainerEnvironment &) = delete;
        ScopedTrainerEnvironment &operator=(const ScopedTrainerEnvironment &) = delete;

    private:
        std::string name_;
        bool had_previous_ = false;
        std::string previous_;
    };

    /** Device and route evidence for the production serial-M1 oracle. */
    struct SerialM1TrainerEvidence
    {
        bool valid = false;
        std::string failure_reason;
        std::vector<float> output;
        int kb = 0;
        int target_waves = 0;
        int observed_kb = 0;
        int observed_target_waves = 0;
        uint64_t observed_route_count = 0;
        std::string observed_path = "missing";
        bool route_counter_ok = false;
        std::string output_digest;
    };

    /** Complete measured evidence for one explicit ROCm decode candidate. */
    struct DecodeCandidateTrainerEvidence
    {
        bool valid = false;
        std::string failure_reason;
        llaminar2::test::trainer::TimingEvidence timing;
        std::vector<double> timing_samples_us;
        llaminar2::test::trainer::FP32Evidence comparison;
        size_t repeat_byte_mismatches = 0;
        bool numerical_correctness = false;
        bool route_counter_ok = false;
        int observed_kb = 0;
        int observed_target_waves = 0;
        std::string observed_candidate_id = "missing";
        std::string observed_path = "missing";
        double effective_bandwidth_gbs = 0.0;
        int isolated_profile_launches = 0;
    };

    /**
     * @brief Copy one device-resident FP32 tensor through the explicit stream.
     */
    static bool copyTrainerOutputToHost(
        TensorBase *output,
        size_t count,
        hipStream_t stream,
        std::vector<float> &host)
    {
        if (!output || !output->gpu_data_ptr() || stream == nullptr)
            return false;
        host.resize(count);
        return hipMemcpyAsync(
                   host.data(),
                   output->gpu_data_ptr(),
                   count * sizeof(float),
                   hipMemcpyDeviceToHost,
                   stream) == hipSuccess &&
               hipStreamSynchronize(stream) == hipSuccess;
    }

    /**
     * @brief Prove that PerfStats observed one requested NativeVNNI route.
     *
     * Route identity is part of correctness evidence. Without this check, a
     * byte-equal fallback or a clamped KB value could be mislabeled as the
     * candidate the generated table later attempts to dispatch.
     */
    static bool findObservedDecodeRoute(
        int M,
        int N,
        int K,
        uint8_t codebook,
        int &kb,
        int &target_waves,
        std::string &path,
        uint64_t *count = nullptr)
    {
        uint64_t observed_count = 0;
        bool matched_shape = false;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"kernel.rocm_native_vnni_small_m_launch"}))
        {
            if (record.domain != "kernel" ||
                record.name != "rocm_native_vnni_small_m_launch" ||
                record.kind != PerfStatRecord::Kind::Counter)
            {
                continue;
            }
            const auto tag = [&](const char *name) -> std::string
            {
                const auto iterator = record.tags.find(name);
                return iterator == record.tags.end() ? std::string{} : iterator->second;
            };
            if (tag("m") != std::to_string(M) ||
                tag("n") != std::to_string(N) ||
                tag("k") != std::to_string(K) ||
                tag("codebook") != std::to_string(static_cast<unsigned>(codebook)))
            {
                continue;
            }
            matched_shape = true;
            kb = std::stoi(tag("kb"));
            target_waves = std::stoi(tag("target_waves_per_cu"));
            path = tag("path");
            observed_count += record.count;
        }
        if (count)
            *count = observed_count;
        return matched_shape && observed_count > 0;
    }

    /**
     * @brief Execute the production prepared kernel one row at a time.
     *
     * Row replay is a test oracle only. It never participates in production
     * execution or candidate timing. Each row calls the ordinary production
     * M=1 kernel under the generated serial policy, and the observed route must
     * agree with the query API before grouped verifier evidence is accepted.
     */
    static SerialM1TrainerEvidence runProductionSerialM1Oracle(
        ROCmQuantisedGemmKernel &kernel,
        TensorBase *grouped_input,
        int M,
        int N,
        int K,
        uint8_t execution_codebook,
        int device_id)
    {
        SerialM1TrainerEvidence result;
        auto fail = [&](const char *reason)
        {
            result.failure_reason = reason;
            return result;
        };
        if (!grouped_input || M <= 0 || N <= 0 || K <= 0)
            return fail("invalid_arguments");
        if (!rocmGemv_native_vnni_query_serial_m1_config(
                execution_codebook,
                N,
                K,
                &result.kb,
                &result.target_waves))
        {
            return fail("serial_policy_query");
        }

        rocmGemv_native_vnni_reset_tuning_overrides();
        PerfStatsCollector::reset();
        hipStream_t stream = nullptr;
        if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess ||
            stream == nullptr)
        {
            return fail("stream_create");
        }
        kernel.setGPUStream(stream);
        const auto requirements = kernel.getWorkspaceRequirements(1, N, K);
        auto workspace = std::make_unique<DeviceWorkspaceManager>(
            DeviceId::rocm(device_id),
            requirements.total_bytes_with_alignment() + 4 * 1024 * 1024);
        if (!workspace->allocate(requirements))
        {
            kernel.clearGPUStreamBinding();
            (void)hipStreamDestroy(stream);
            return fail("workspace_allocate");
        }
        kernel.bindWorkspace(workspace.get());

        const float *grouped_host = grouped_input->data();
        result.output.resize(static_cast<size_t>(M) * static_cast<size_t>(N));
        for (int row = 0; row < M; ++row)
        {
            auto row_input = TestTensorFactory::createFP32(
                {1, static_cast<size_t>(K)});
            std::copy_n(
                grouped_host + static_cast<size_t>(row) * static_cast<size_t>(K),
                K,
                row_input->mutable_data());
            auto row_output = TestTensorFactory::createFP32(
                {1, static_cast<size_t>(N)});
            if (!row_input->ensureOnDevice(DeviceId::rocm(device_id), stream) ||
                !row_output->allocateOnDevice(DeviceId::rocm(device_id)) ||
                !kernel.multiply_tensor(row_input.get(), row_output.get(), 1, N, K) ||
                hipStreamSynchronize(stream) != hipSuccess)
            {
                kernel.unbindWorkspace();
                kernel.clearGPUStreamBinding();
                (void)hipStreamDestroy(stream);
                return fail("serial_row_launch");
            }
            if (hipMemcpyAsync(
                    result.output.data() + static_cast<size_t>(row) * static_cast<size_t>(N),
                    row_output->gpu_data_ptr(),
                    static_cast<size_t>(N) * sizeof(float),
                    hipMemcpyDeviceToHost,
                    stream) != hipSuccess ||
                hipStreamSynchronize(stream) != hipSuccess)
            {
                kernel.unbindWorkspace();
                kernel.clearGPUStreamBinding();
                (void)hipStreamDestroy(stream);
                return fail("serial_row_download");
            }
        }

        int observed_kb = 0;
        int observed_waves = 0;
        std::string observed_path;
        uint64_t observed_count = 0;
        result.route_counter_ok = findObservedDecodeRoute(
            1,
            N,
            K,
            execution_codebook,
            observed_kb,
            observed_waves,
            observed_path,
            &observed_count) &&
                                  observed_count >= static_cast<uint64_t>(M) &&
                                  observed_kb == result.kb &&
                                  observed_waves == result.target_waves;
        result.observed_kb = observed_kb;
        result.observed_target_waves = observed_waves;
        result.observed_route_count = observed_count;
        result.observed_path = observed_path.empty() ? "missing" : observed_path;
        result.output_digest =
            llaminar2::test::trainer::nativeByteDigest(result.output);
        result.valid = result.route_counter_ok;
        if (!result.valid)
            result.failure_reason = "serial_route_proof";

        kernel.unbindWorkspace();
        kernel.clearGPUStreamBinding();
        (void)hipStreamDestroy(stream);
        return result;
    }

    /**
     * @brief Measure one explicit production candidate against serial M=1.
     */
    static DecodeCandidateTrainerEvidence runProductionDecodeCandidate(
        ROCmQuantisedGemmKernel &kernel,
        TensorBase *input,
        const SerialM1TrainerEvidence &serial,
        const DecodeVariant &variant,
        DecodeExecutionMode execution_mode,
        const std::string &source_format,
        int M,
        int N,
        int K,
        uint8_t execution_codebook,
        double weight_bytes,
        int warmups,
        int timing_samples,
        const std::string &profiler_request_id,
        int device_id)
    {
        DecodeCandidateTrainerEvidence result;
        auto fail = [&](const char *reason)
        {
            result.failure_reason = reason;
            return result;
        };
        if (!serial.valid || !input || warmups < 0 || timing_samples <= 0)
            return fail("invalid_arguments");
        if (!variant.appliesToM(M))
            return fail("candidate_contract_mismatch");

        DecodeTuningGuard tuning(variant);
        hipStream_t stream = nullptr;
        if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess ||
            stream == nullptr)
        {
            return fail("stream_create");
        }
        kernel.setGPUStream(stream);
        const auto requirements = kernel.getWorkspaceRequirements(M, N, K);
        auto workspace = std::make_unique<DeviceWorkspaceManager>(
            DeviceId::rocm(device_id),
            requirements.total_bytes_with_alignment() + 4 * 1024 * 1024);
        if (!workspace->allocate(requirements))
        {
            kernel.clearGPUStreamBinding();
            (void)hipStreamDestroy(stream);
            return fail("workspace_allocate");
        }
        kernel.bindWorkspace(workspace.get());

        auto output = TestTensorFactory::createFP32(
            {static_cast<size_t>(M), static_cast<size_t>(N)});
        const DeviceId device = DeviceId::rocm(device_id);
        if (!input->ensureOnDevice(device, stream) ||
            !output->allocateOnDevice(device))
        {
            kernel.unbindWorkspace();
            kernel.clearGPUStreamBinding();
            (void)hipStreamDestroy(stream);
            return fail("tensor_prepare");
        }
        TransferEngine::requireDeviceInput(
            input,
            device,
            reinterpret_cast<void *>(stream));
        const std::vector<const TensorBase *> capture_inputs = {input};
        const std::vector<TensorBase *> capture_outputs = {output.get()};

        const auto run_once = [&]() -> bool
        {
            if (M >= 2)
            {
                auto verifier_scope = kernel.beginVerifierDecodeEquivalentScope();
                return kernel.multiply_tensor(input, output.get(), M, N, K);
            }
            return kernel.multiply_tensor(input, output.get(), M, N, K);
        };

        PerfStatsCollector::reset();
        CapturedDecodeLaunch captured_launch;
        if (execution_mode == DecodeExecutionMode::GraphCaptured)
        {
            /*
             * Resolve lazy preparation before capture, then clear its route
             * evidence. The production graph itself contains only the stable
             * projection and its frozen tensor-dependency contract.
             */
            if (!run_once() || hipStreamSynchronize(stream) != hipSuccess)
            {
                kernel.unbindWorkspace();
                kernel.clearGPUStreamBinding();
                (void)hipStreamDestroy(stream);
                return fail("graph_primer");
            }
            PerfStatsCollector::reset();
            if (!captured_launch.capture(
                    stream,
                    device,
                    capture_inputs,
                    capture_outputs,
                    run_once))
            {
                kernel.unbindWorkspace();
                kernel.clearGPUStreamBinding();
                (void)hipStreamDestroy(stream);
                return fail("graph_capture");
            }
        }
        const auto execute_once = [&]() -> bool
        {
            return execution_mode == DecodeExecutionMode::GraphCaptured
                       ? captured_launch.launch(stream)
                       : run_once();
        };
        const auto cleanup = [&]()
        {
            captured_launch.reset();
            kernel.unbindWorkspace();
            kernel.clearGPUStreamBinding();
            (void)hipStreamDestroy(stream);
        };
        if (!execute_once() || hipStreamSynchronize(stream) != hipSuccess)
        {
            cleanup();
            return fail("candidate_launch");
        }
        uint64_t observed_count = 0;
        const bool observed = findObservedDecodeRoute(
            M,
            N,
            K,
            execution_codebook,
            result.observed_kb,
            result.observed_target_waves,
            result.observed_path,
            &observed_count);
        result.observed_candidate_id =
            !observed
                ? "missing"
                : (variant.kind == DecodeCandidateKind::VerifierInheritSerialM1
                       ? variant.name
                       : "KB" + std::to_string(result.observed_kb));
        const int expected_kb =
            variant.kind == DecodeCandidateKind::VerifierInheritSerialM1
                ? serial.kb
                : variant.kb;
        const bool inherited_waves_match =
            variant.kind != DecodeCandidateKind::VerifierInheritSerialM1 ||
            result.observed_target_waves == serial.target_waves;
        result.route_counter_ok = observed && observed_count > 0 &&
                                  result.observed_kb == expected_kb &&
                                  inherited_waves_match &&
                                  (result.observed_path == "direct" ||
                                   result.observed_path == "split_reduce");

        if (!profiler_request_id.empty())
        {
            /*
             * The canonical timing corpus already proved byte correctness,
             * repeatability, and latency for this exact physical candidate.
             * A profiler process exists solely to gather counters from one
             * additional production launch. Repeating D2H validation and the
             * event-timed sample loop on every rocprof counter pass would add
             * no evidence and adds measurable avoidable collection overhead.
             *
             * Keep a small, fixed amount of preconditioning so the selected
             * launch does not observe first-use caches or lazy runtime setup.
             * These launches execute while rocprof collection is paused. The
             * explicit stream synchronization is local to this request and is
             * the only boundary needed before opening its ROCTx range.
             */
            constexpr int kProfilerPreconditioningLaunches = 2;
            if (!result.route_counter_ok)
            {
                cleanup();
                return fail("profiler_route");
            }
            for (int launch = 0;
                 launch < kProfilerPreconditioningLaunches;
                 ++launch)
            {
                if (!execute_once())
                {
                    cleanup();
                    return fail("profiler_precondition_launch");
                }
            }
            if (hipStreamSynchronize(stream) != hipSuccess)
            {
                cleanup();
                return fail("profiler_precondition_sync");
            }

            /*
             * rocprofiler-sdk selected-region collection remains paused during
             * all setup and preconditioning. Resume for exactly one launch,
             * then pause before leaving its request-specific range. The launch
             * is terminal for this graph executable because ROCm 7.1 profiler
             * interception mutates HIP graph packet bookkeeping.
             */
            const std::string profiler_range =
                "NativeVNNIProfile::" + profiler_request_id;
            if (roctxRangePushA(profiler_range.c_str()) < 0)
            {
                cleanup();
                return fail("profiler_range_push");
            }
            if (roctxProfilerResume(0) != 0)
            {
                (void)roctxRangePop();
                cleanup();
                return fail("profiler_resume");
            }
            const bool launch_ok = execute_once();
            const hipError_t synchronize_status = hipStreamSynchronize(stream);
            const int pause_status = roctxProfilerPause(0);
            const int range_level = roctxRangePop();
            if (!launch_ok || synchronize_status != hipSuccess || pause_status != 0)
            {
                cleanup();
                return fail("profiler_target_launch");
            }
            if (range_level < 0)
            {
                cleanup();
                return fail("profiler_range_pop");
            }
            result.isolated_profile_launches = 1;
            std::fprintf(
                stderr,
                "[NativeVNNIProfiler][ROCm] request=%s candidate=%s "
                "mode=%s M=%d N=%d K=%d launches=1\n",
                profiler_request_id.c_str(),
                variant.name.c_str(),
                decodeExecutionModeName(execution_mode),
                M,
                N,
                K);
            cleanup();
            result.valid = true;
            return result;
        }

        std::vector<float> first_output;
        if (!copyTrainerOutputToHost(
                output.get(),
                static_cast<size_t>(M) * static_cast<size_t>(N),
                stream,
                first_output) ||
            !execute_once() ||
            hipStreamSynchronize(stream) != hipSuccess)
        {
            cleanup();
            return fail("repeat_launch");
        }
        std::vector<float> repeated_output;
        if (!copyTrainerOutputToHost(
                output.get(),
                static_cast<size_t>(M) * static_cast<size_t>(N),
                stream,
                repeated_output))
        {
            cleanup();
            return fail("repeat_download");
        }

        result.repeat_byte_mismatches =
            llaminar2::test::trainer::nativeByteMismatchCount(
                first_output, repeated_output);
        /*
         * Certify every grouped row. Restricting this comparison to N values
         * would validate row zero only and let later-row indexing, workspace,
         * or publication failures enter the learned verifier policy.
         */
        result.comparison = llaminar2::test::trainer::compareFP32(
            repeated_output,
            serial.output,
            static_cast<size_t>(M) * static_cast<size_t>(N));
        result.numerical_correctness =
            result.comparison.nonfinite_count == 0 &&
            result.comparison.cosine >=
                static_cast<double>(cosine_gate_for(source_format));

        for (int warmup = 0; warmup < warmups; ++warmup)
        {
            if (!execute_once())
            {
                cleanup();
                return fail("warmup_launch");
            }
        }
        if (hipStreamSynchronize(stream) != hipSuccess)
        {
            cleanup();
            return fail("warmup_sync");
        }

        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;
        if (hipEventCreate(&start) != hipSuccess ||
            hipEventCreate(&stop) != hipSuccess)
        {
            if (start)
                (void)hipEventDestroy(start);
            if (stop)
                (void)hipEventDestroy(stop);
            cleanup();
            return fail("event_create");
        }
        result.timing_samples_us.reserve(static_cast<size_t>(timing_samples));
        for (int sample = 0; sample < timing_samples; ++sample)
        {
            if (hipEventRecord(start, stream) != hipSuccess ||
                !execute_once() ||
                hipEventRecord(stop, stream) != hipSuccess ||
                hipEventSynchronize(stop) != hipSuccess)
            {
                (void)hipEventDestroy(start);
                (void)hipEventDestroy(stop);
                cleanup();
                return fail("timed_launch");
            }
            float elapsed_ms = 0.0f;
            if (hipEventElapsedTime(&elapsed_ms, start, stop) != hipSuccess)
            {
                (void)hipEventDestroy(start);
                (void)hipEventDestroy(stop);
                cleanup();
                return fail("timed_read");
            }
            result.timing_samples_us.push_back(
                static_cast<double>(elapsed_ms) * 1000.0);
        }
        (void)hipEventDestroy(start);
        (void)hipEventDestroy(stop);

        std::sort(
            result.timing_samples_us.begin(),
            result.timing_samples_us.end());
        result.timing =
            llaminar2::test::trainer::summarizeSortedTimingSamples(
                result.timing_samples_us);
        if (result.timing.median > 0.0)
        {
            result.effective_bandwidth_gbs =
                weight_bytes / (result.timing.median * 1.0e-6) / 1.0e9;
        }
        result.valid = true;

        cleanup();
        return result;
    }
#endif

    // =============================================================================
    // Test: Single-shape sweep across all 18 formats (quick CI check)
    // =============================================================================

    TEST_F(NativeVNNIPerfTest, AllFormats_0_5B_FFN_Up)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        const GEMVShape shape{"0.5B_FFN_Up", 4864, 896};

        fprintf(stderr, "\n[NativeVNNI Perf] Device: %s\n", device_name_.c_str());
        fprintf(stderr, "[NativeVNNI Perf] Shape: %s (N=%d K=%d) | %d warmup + %d runs\n",
                shape.name.c_str(), shape.N, shape.K, WARMUP_RUNS, BENCH_RUNS);

        // INT8 reference
        constexpr int M = 1;
        double int8_us = benchmarkINT8Reference(shape, M, 0);
        fprintf(stderr, "[NativeVNNI Perf] INT8 reference: %.1f μs\n", int8_us);

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Format" << "BPW" << "Weight KB" << "Min μs" << "Mean μs"
              << "Speedup" << "Kern Eff" << "BW GB/s" << "BW Eff %" << "Cosine"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 9; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &fmt : ALL_PERF_FORMATS)
        {
            auto weights = fmt.create(
                static_cast<size_t>(shape.N), static_cast<size_t>(shape.K));
            GpuWeightsCache gpu_w;
            if (weights)
            {
                std::vector<float> w_fp32(static_cast<size_t>(shape.N) * shape.K);
                weights->to_fp32(w_fp32.data());
                gpu_w.upload(w_fp32.data(), shape.N, shape.K, 0);
            }

            auto r = benchmarkFormat(fmt, shape, M, int8_us, weights.get(), &gpu_w, 0);

            ASSERT_GT(r.min_us, 0.0)
                << fmt.name << '/' << shape.name
                << " did not execute the production NativeVNNI path";
            ASSERT_TRUE(r.correctness_pass)
                << fmt.name << '/' << shape.name
                << " failed its FP32-reference correctness gate";

            char buf_bpw[16], buf_kb[16], buf_min[16], buf_mean[16];
            char buf_speedup[16], buf_keff[16], buf_bw[16], buf_eff[16], buf_cos[16];
            snprintf(buf_bpw, sizeof(buf_bpw), "%.1f", r.bpw);
            snprintf(buf_kb, sizeof(buf_kb), "%.0f", r.weight_bytes / 1024.0);
            snprintf(buf_min, sizeof(buf_min), "%.1f", r.min_us);
            snprintf(buf_mean, sizeof(buf_mean), "%.1f", r.mean_us);
            snprintf(buf_speedup, sizeof(buf_speedup), "%.2fx", r.speedup_vs_int8);
            snprintf(buf_keff, sizeof(buf_keff), "%.0f%%", r.kernel_efficiency);
            snprintf(buf_bw, sizeof(buf_bw), "%.1f", r.eff_bw_gbps);
            snprintf(buf_eff, sizeof(buf_eff), "%.1f%%", r.bw_efficiency);
            snprintf(buf_cos, sizeof(buf_cos), "%.4f", r.cosine_sim);
            table << r.format_name << buf_bpw << buf_kb << buf_min << buf_mean
                  << buf_speedup << buf_keff << buf_bw << buf_eff << buf_cos
                  << fort::endr;
        }

        fprintf(stderr, "\n%s\n", table.to_string().c_str());
#endif
    }

    TEST_F(NativeVNNIPerfTest, TrainerCsv_CodebookTagged)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        std::set<std::string> format_filters = getEnvCsvSet("LLAMINAR_ROCM_NVNNI_DECODE_FORMATS");
        if (format_filters.empty())
            format_filters.insert("q4_0");
        std::set<std::string> shape_filters = getEnvCsvSet("LLAMINAR_ROCM_NVNNI_DECODE_SHAPES");
        if (shape_filters.empty())
            shape_filters.insert("0.5b_attnout");
        const int max_cases = std::max(1, getEnvInt("LLAMINAR_ROCM_NVNNI_DECODE_MAX_CASES").value_or(1));
        const std::string csv_path = getEnvString("LLAMINAR_ROCM_NVNNI_DECODE_CSV");
        const std::string timing_csv_path =
            getEnvString("LLAMINAR_ROCM_NVNNI_DECODE_TIMING_CSV");
        const int trainer_warmups = std::max(
            0,
            getEnvInt("LLAMINAR_ROCM_NVNNI_DECODE_WARMUPS").value_or(5));
        const int trainer_samples = std::max(
            1,
            getEnvInt("LLAMINAR_ROCM_NVNNI_DECODE_SAMPLES").value_or(30));
        const std::vector<DecodeVariant> variants = getDecodeVariants();
        const std::vector<DecodeExecutionMode> execution_modes =
            getDecodeExecutionModes();
        const std::vector<int> m_values = getDecodeMValues();
        const std::string profiler_request_id =
            native_vnni_dispatch::profilerRequestId();
        const std::string profiler_batch_path =
            native_vnni_dispatch::profilerEnvironment(
                native_vnni_dispatch::kProfilerBatchPathEnvironment);
        ASSERT_TRUE(
            profiler_request_id.empty() || profiler_batch_path.empty())
            << "ROCm profiler request and batch modes are mutually exclusive";
        ROCmProfilerBatch profiler_batch(profiler_batch_path);
        const bool profiler_only =
            !profiler_request_id.empty() || profiler_batch.enabled();
        if (!profiler_request_id.empty())
        {
            ASSERT_EQ(format_filters.size(), 1u)
                << "isolated ROCm profiling requires exactly one source format";
            ASSERT_EQ(shape_filters.size(), 1u)
                << "isolated ROCm profiling requires exactly one shape";
            ASSERT_EQ(variants.size(), 1u)
                << "isolated ROCm profiling requires exactly one candidate";
            ASSERT_EQ(execution_modes.size(), 1u)
                << "isolated ROCm profiling requires exactly one execution mode";
            ASSERT_EQ(m_values.size(), 1u)
                << "isolated ROCm profiling requires exactly one runtime M";
            ASSERT_EQ(max_cases, 1)
                << "isolated ROCm profiling requires exactly one shape case";
        }
        ScopedTrainerEnvironment stats_enabled("LLAMINAR_PERF_STATS_JSON", "1");

        std::FILE *csv = nullptr;
        if (!csv_path.empty())
        {
            csv = std::fopen(csv_path.c_str(), "w");
            ASSERT_NE(csv, nullptr) << "Failed to open ROCm NativeVNNI decode CSV: " << csv_path;
            std::fprintf(
                csv,
                "backend,phase,source_format,source_codebook,execution_codebook,"
                "shape,execution_mode,m,n,k,candidate_id,kb,target_waves,weight_bytes,"
                "warmup_count,sample_count,min_us,median_us,p95_us,mad_us,cv,"
                "effective_bandwidth_gbs,bit_mismatches,first_bit_mismatch,"
                "repeat_byte_mismatches,max_abs,relative_l2,cosine,symmetric_kld,"
                "grouped_output_digest,serial_output_digest,timing_sample_digest,"
                "route_counter_ok,observed_candidate_id,observed_path,serial_m1_kb,"
                "serial_m1_target_waves,serial_route_counter_ok,numerical_correctness,"
                "correctness_pass,is_winner\n");
        }
        std::FILE *timing_csv = nullptr;
        if (!timing_csv_path.empty())
        {
            timing_csv = std::fopen(timing_csv_path.c_str(), "w");
            ASSERT_NE(timing_csv, nullptr)
                << "Failed to open ROCm NativeVNNI raw timing CSV: "
                << timing_csv_path;
            std::fprintf(
                timing_csv,
                "backend,phase,source_format,source_codebook,execution_codebook,"
                "shape,execution_mode,m,n,k,candidate_id,kb,target_waves,"
                "sample_index,timed_replays,latency_us,latency_us_hex\n");
        }

        int executed_cases = 0;
        int executed_rows = 0;
        int isolated_profile_launches = 0;
        uint64_t preparation_id = 1;
        for (const auto &fmt : ALL_PERF_FORMATS)
        {
            if (!shouldRunName(format_filters, fmt.name))
                continue;

            for (const auto &shape : SHAPES)
            {
                if (!shouldRunName(shape_filters, shape.name))
                    continue;
                if (profiler_batch.enabled() &&
                    !profiler_batch.containsShape(fmt.name, shape.name))
                {
                    continue;
                }

                auto weights = profiler_only
                                   ? makeProfilerWeightFixture(
                                         fmt,
                                         static_cast<size_t>(shape.N),
                                         static_cast<size_t>(shape.K))
                                   : fmt.create(
                                         static_cast<size_t>(shape.N),
                                         static_cast<size_t>(shape.K));
                ASSERT_NE(weights, nullptr) << fmt.name << '/' << shape.name;
                const auto &format_metadata =
                    llaminar2::test::quantizedVerifierFormat(fmt.name);
                auto prepared = llaminar2::test::makeGpuPreparedGemm(
                    weights.get(),
                    DeviceId::rocm(0),
                    "perf.rocm.nvnni.decode." + fmt.name + "." + shape.name,
                    ModelContextId{preparation_id++});
                auto *kernel = dynamic_cast<ROCmQuantisedGemmKernel *>(prepared.kernel);
                ASSERT_NE(kernel, nullptr)
                    << fmt.name << '/' << shape.name
                    << " did not produce a production ROCm prepared kernel";
                const uint8_t execution_codebook =
                    format_metadata.device_execution_codebook_id;
                const double weight_bytes = static_cast<double>(
                    llaminar2::test::quantizedRawBytesForGpuPreparedTest(*weights));

                struct VariantRow
                {
                    DecodeVariant variant;
                    DecodeExecutionMode execution_mode = DecodeExecutionMode::Eager;
                    size_t execution_mode_index = 0;
                    DecodeCandidateTrainerEvidence result;
                    bool eligible = false;
                };

                for (int M : m_values)
                {
                    if (executed_cases >= max_cases)
                        break;

                    size_t applicable_candidates = 0;
                    if (profiler_batch.enabled())
                    {
                        for (const DecodeExecutionMode execution_mode :
                             execution_modes)
                        {
                            applicable_candidates += profiler_batch.candidateIds(
                                fmt.name,
                                shape.name,
                                decodeExecutionModeName(execution_mode),
                                M,
                                shape.N,
                                shape.K)
                                                         .size();
                        }
                        if (applicable_candidates == 0)
                            continue;
                    }
                    else
                    {
                        applicable_candidates = static_cast<size_t>(
                            std::count_if(
                                variants.begin(),
                                variants.end(),
                                [M](const DecodeVariant &variant)
                                { return variant.appliesToM(M); }));
                    }

                    auto input = TestTensorFactory::createFP32Random(
                        {static_cast<size_t>(M), static_cast<size_t>(shape.K)},
                        -0.35f,
                        0.35f,
                        static_cast<uint32_t>(
                            0xC001u + M * 131u + fmt.name.size() * 17u +
                            shape.name.size()));
                    const SerialM1TrainerEvidence serial =
                        runProductionSerialM1Oracle(
                            *kernel,
                            input.get(),
                            M,
                            shape.N,
                            shape.K,
                            execution_codebook,
                            0);
                    ASSERT_TRUE(serial.valid)
                        << fmt.name << '/' << shape.name << " M=" << M
                        << " serial M1 oracle failed: " << serial.failure_reason
                        << " queried={kb=" << serial.kb
                        << ",waves=" << serial.target_waves
                        << "} observed={kb=" << serial.observed_kb
                        << ",waves=" << serial.observed_target_waves
                        << ",count=" << serial.observed_route_count
                        << ",path=" << serial.observed_path << '}';

                    ASSERT_GT(applicable_candidates, 0u)
                        << "No candidate selected for the required "
                        << (M == 1 ? "Fast M=1" : "VerifierSerialM1Bitwise")
                        << " contract at M=" << M;

                    std::vector<VariantRow> rows;
                    rows.reserve(applicable_candidates * execution_modes.size());
                    std::vector<bool> active_modes(
                        execution_modes.size(), false);
                    for (size_t mode_index = 0;
                         mode_index < execution_modes.size();
                         ++mode_index)
                    {
                        const DecodeExecutionMode execution_mode =
                            execution_modes[mode_index];
                        const std::string execution_mode_name =
                            decodeExecutionModeName(execution_mode);
                        const std::set<std::string> requested_candidate_ids =
                            profiler_batch.enabled()
                                ? profiler_batch.candidateIds(
                                      fmt.name,
                                      shape.name,
                                      execution_mode_name,
                                      M,
                                      shape.N,
                                      shape.K)
                                : std::set<std::string>{};
                        if (profiler_batch.enabled() &&
                            requested_candidate_ids.empty())
                        {
                            continue;
                        }
                        for (const auto &variant : variants)
                        {
                            if (!variant.appliesToM(M))
                                continue;
                            const std::string effective_candidate_id =
                                effectiveDecodeCandidateId(variant);
                            if (profiler_batch.enabled() &&
                                !requested_candidate_ids.contains(
                                    effective_candidate_id))
                            {
                                continue;
                            }
                            std::string target_profiler_request_id =
                                profiler_request_id;
                            if (profiler_batch.enabled())
                            {
                                ROCmProfilerBatchRequest *request =
                                    profiler_batch.claim(
                                        fmt.name,
                                        shape.name,
                                        execution_mode_name,
                                        M,
                                        shape.N,
                                        shape.K,
                                        effective_candidate_id);
                                ASSERT_NE(request, nullptr)
                                    << "ROCm profiler batch omitted exact candidate "
                                    << effective_candidate_id;
                                target_profiler_request_id = request->request_id;
                            }
                            DecodeCandidateTrainerEvidence result =
                                runProductionDecodeCandidate(
                                    *kernel,
                                    input.get(),
                                    serial,
                                    variant,
                                    execution_mode,
                                    fmt.name,
                                    M,
                                    shape.N,
                                    shape.K,
                                    execution_codebook,
                                    weight_bytes,
                                    trainer_warmups,
                                    trainer_samples,
                                    target_profiler_request_id,
                                    0);
                            ASSERT_TRUE(result.valid)
                                << fmt.name << '/' << shape.name << " M=" << M
                                << ' ' << variant.name << ' '
                                << decodeExecutionModeName(execution_mode)
                                << " candidate failed: " << result.failure_reason;
                            const bool isolated_profiler_request =
                                !target_profiler_request_id.empty();
                            const bool verifier_exact =
                                isolated_profiler_request || M == 1 ||
                                result.comparison.bitwiseEqual();
                            const bool eligible =
                                isolated_profiler_request
                                    ? result.route_counter_ok &&
                                          result.isolated_profile_launches == 1
                                    : result.route_counter_ok &&
                                          result.repeat_byte_mismatches == 0 &&
                                          result.numerical_correctness &&
                                          verifier_exact;
                            rows.push_back(VariantRow{
                                variant,
                                execution_mode,
                                mode_index,
                                std::move(result),
                                eligible});
                            active_modes[mode_index] = true;
                            isolated_profile_launches +=
                                rows.back().result.isolated_profile_launches;
                        }
                    }

                    std::vector<int> best_indices(execution_modes.size(), -1);
                    for (size_t row_index = 0; row_index < rows.size(); ++row_index)
                    {
                        const auto &candidate = rows[row_index];
                        if (!candidate.eligible)
                            continue;
                        int &best_index =
                            best_indices[candidate.execution_mode_index];
                        if (profiler_batch.enabled() ||
                            !profiler_request_id.empty())
                        {
                            if (best_index < 0)
                                best_index = static_cast<int>(row_index);
                            continue;
                        }
                        if (candidate.result.timing.median <= 0.0)
                            continue;
                        if (best_index < 0 ||
                            candidate.result.timing.median <
                                rows[static_cast<size_t>(best_index)].result.timing.median)
                            best_index = static_cast<int>(row_index);
                    }

                    if (csv)
                    {
                        for (size_t row_index = 0; row_index < rows.size(); ++row_index)
                        {
                            const auto &row = rows[row_index];
                            const auto &r = row.result;
                            std::fprintf(
                                csv,
                                "rocm,decode,%s,%u,%u,%s,%s,%d,%d,%d,%s,%d,%d,"
                                "%.0f,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%zu,%zu,"
                                "%zu,%.9g,%.9g,%.9g,%.9g,%s,%s,%s,%d,%s,%s,%d,%d,"
                                "%d,%d,%d,%d\n",
                                fmt.name.c_str(),
                                static_cast<unsigned>(format_metadata.source_codebook_id),
                                static_cast<unsigned>(execution_codebook),
                                shape.name.c_str(),
                                decodeExecutionModeName(row.execution_mode),
                                M,
                                shape.N,
                                shape.K,
                                row.variant.name.c_str(),
                                r.observed_kb,
                                r.observed_target_waves,
                                weight_bytes,
                                trainer_warmups,
                                trainer_samples,
                                r.timing.min,
                                r.timing.median,
                                r.timing.p95,
                                r.timing.mad,
                                r.timing.cv,
                                r.effective_bandwidth_gbs,
                                r.comparison.mismatch_count,
                                r.comparison.first_mismatch_index,
                                r.repeat_byte_mismatches,
                                r.comparison.max_abs,
                                r.comparison.relative_l2,
                                r.comparison.cosine,
                                r.comparison.symmetric_kld,
                                r.comparison.actual_digest.c_str(),
                                serial.output_digest.c_str(),
                                r.timing.digest.c_str(),
                                r.route_counter_ok ? 1 : 0,
                                r.observed_candidate_id.c_str(),
                                r.observed_path.c_str(),
                                serial.kb,
                                serial.target_waves,
                                serial.route_counter_ok ? 1 : 0,
                                r.numerical_correctness ? 1 : 0,
                                row.eligible ? 1 : 0,
                                best_indices[row.execution_mode_index] >= 0 &&
                                        static_cast<int>(row_index) ==
                                            best_indices[row.execution_mode_index]
                                    ? 1
                                    : 0);
                            ++executed_rows;

                            if (timing_csv)
                            {
                                for (size_t sample_index = 0;
                                     sample_index < r.timing_samples_us.size();
                                     ++sample_index)
                                {
                                    std::fprintf(
                                        timing_csv,
                                        "rocm,decode,%s,%u,%u,%s,%s,%d,%d,%d,%s,"
                                        "%d,%d,%zu,1,%.9f,%a\n",
                                        fmt.name.c_str(),
                                        static_cast<unsigned>(
                                            format_metadata.source_codebook_id),
                                        static_cast<unsigned>(execution_codebook),
                                        shape.name.c_str(),
                                        decodeExecutionModeName(row.execution_mode),
                                        M,
                                        shape.N,
                                        shape.K,
                                        row.variant.name.c_str(),
                                        r.observed_kb,
                                        r.observed_target_waves,
                                        sample_index,
                                        r.timing_samples_us[sample_index],
                                        r.timing_samples_us[sample_index]);
                                }
                                std::fflush(timing_csv);
                            }
                        }
                        std::fflush(csv);
                    }

                    for (size_t mode_index = 0;
                         mode_index < execution_modes.size();
                         ++mode_index)
                    {
                        if (!active_modes[mode_index])
                            continue;
                        const int best_index = best_indices[mode_index];
                        if (best_index < 0)
                        {
                            for (const auto &row : rows)
                            {
                                if (row.execution_mode_index != mode_index)
                                    continue;
                                std::fprintf(
                                    stderr,
                                    "[ROCmNativeVNNI][DECODE][TRAINER][CANDIDATE] format=%s shape=%s mode=%s M=%d variant=%s time_us=%.3f cosine=%.6f correctness=%d\n",
                                    fmt.name.c_str(),
                                    shape.name.c_str(),
                                    decodeExecutionModeName(row.execution_mode),
                                    M,
                                    row.variant.name.c_str(),
                                    row.result.timing.median,
                                    row.result.comparison.cosine,
                                    row.eligible ? 1 : 0);
                            }
                        }
                        ASSERT_GE(best_index, 0)
                            << "No correct ROCm NativeVNNI decode variant for "
                            << fmt.name << "/" << shape.name << " M=" << M
                            << " mode="
                            << decodeExecutionModeName(execution_modes[mode_index]);

                        const auto &best = rows[static_cast<size_t>(best_index)];
                        std::fprintf(stderr,
                                     "[ROCmNativeVNNI][DECODE][TRAINER] format=%s "
                                     "source_codebook=%u execution_codebook=%u shape=%s "
                                     "mode=%s M=%d best=%s median_us=%.3f bit_mismatches=%zu "
                                     "repeat_byte_mismatches=%zu route=%s\n",
                                     fmt.name.c_str(),
                                     static_cast<unsigned>(format_metadata.source_codebook_id),
                                     static_cast<unsigned>(execution_codebook),
                                     shape.name.c_str(),
                                     decodeExecutionModeName(best.execution_mode),
                                     M,
                                     best.variant.name.c_str(),
                                     best.result.timing.median,
                                     best.result.comparison.mismatch_count,
                                     best.result.repeat_byte_mismatches,
                                     best.result.observed_candidate_id.c_str());
                        ASSERT_TRUE(best.eligible)
                            << fmt.name << "/" << shape.name << " M=" << M
                            << " mode=" << decodeExecutionModeName(best.execution_mode)
                            << " first_bit_mismatch="
                            << best.result.comparison.first_mismatch_index;
                    }
                    ++executed_cases;
                }
                if (executed_cases >= max_cases)
                    break;
            }
        }

        if (csv)
        {
            std::fclose(csv);
            ASSERT_GT(executed_rows, 0) << "ROCm NativeVNNI decode trainer CSV had no rows.";
        }
        if (timing_csv)
            ASSERT_EQ(std::fclose(timing_csv), 0);
        ASSERT_GT(executed_cases, 0) << "No ROCm NativeVNNI decode trainer cases selected.";
        if (!profiler_request_id.empty())
        {
            ASSERT_EQ(isolated_profile_launches, 1)
                << "each ROCm profiler request must execute one target launch";
        }
        else if (profiler_batch.enabled())
        {
            EXPECT_NO_THROW(profiler_batch.requireComplete());
            ASSERT_EQ(isolated_profile_launches, profiler_batch.size())
                << "every ROCm profiler batch member must execute once";
        }
#endif
    }

    // =============================================================================
    // Test: Full matrix — all formats × all shapes — multi-GPU
    // =============================================================================

    TEST_F(NativeVNNIPerfTest, AllFormats_AllShapes_Matrix)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        fprintf(stderr, "\n[NativeVNNI Perf] Device: %s\n", device_name_.c_str());
        fprintf(stderr, "[NativeVNNI Perf] %zu formats × %zu shapes | %d warmup + %d runs each\n",
                ALL_PERF_FORMATS.size(), SHAPES.size(), WARMUP_RUNS, BENCH_RUNS);
        fprintf(stderr, "[NativeVNNI Perf] Using %d GPU(s) for parallel benchmarking\n", NUM_GPUS);

        // =========================================================================
        // Phase 1: Benchmark INT8 VNNI reference for each shape — multi-GPU
        // =========================================================================
        fprintf(stderr, "\n[Phase 1] Benchmarking INT8 VNNI reference on %d GPU(s)...\n",
                NUM_GPUS);

        struct Int8Work
        {
            int shape_idx;
            double cost;
        };
        std::vector<Int8Work> int8_work;
        for (int si = 0; si < (int)SHAPES.size(); ++si)
            int8_work.push_back({si, (double)SHAPES[si].N * SHAPES[si].K});

        std::sort(int8_work.begin(), int8_work.end(),
                  [](const Int8Work &a, const Int8Work &b)
                  { return a.cost > b.cost; });

        std::vector<std::vector<Int8Work>> int8_per_gpu(NUM_GPUS);
        for (size_t i = 0; i < int8_work.size(); ++i)
            int8_per_gpu[i % NUM_GPUS].push_back(int8_work[i]);

        std::vector<double> int8_times(SHAPES.size(), 0.0);
        std::atomic<int> int8_done{0};
        constexpr int kThroughputM = 1;

        {
            std::vector<std::thread> threads;
            for (int g = 0; g < NUM_GPUS; ++g)
            {
                threads.emplace_back([&, g]()
                                     {
                    for (const auto &w : int8_per_gpu[g])
                    {
                        double ref_us = benchmarkINT8Reference(SHAPES[w.shape_idx], kThroughputM, g);
                        int8_times[w.shape_idx] = ref_us;
                        int done = ++int8_done;
                        fprintf(stderr, "  [GPU %d] INT8 %s: %.1f μs  (%d/%zu)\n",
                                g, SHAPES[w.shape_idx].name.c_str(), ref_us,
                                done, int8_work.size());
                    } });
            }
            for (auto &t : threads)
                t.join();
        }

        std::unordered_map<std::string, double> int8_ref_us;
        for (int si = 0; si < (int)SHAPES.size(); ++si)
            int8_ref_us[SHAPES[si].name] = int8_times[si];

        // =========================================================================
        // Phase 2: Benchmark all native-VNNI formats — multi-GPU
        // =========================================================================
        fprintf(stderr, "\n[Phase 2] Benchmarking %zu native-VNNI formats on %d GPU(s)...\n",
                ALL_PERF_FORMATS.size(), NUM_GPUS);

        struct WorkGroup
        {
            int format_idx;
            int shape_idx;
            double cost;
        };
        std::vector<WorkGroup> groups;
        for (int fi = 0; fi < (int)ALL_PERF_FORMATS.size(); ++fi)
            for (int si = 0; si < (int)SHAPES.size(); ++si)
                groups.push_back({fi, si, (double)SHAPES[si].N * SHAPES[si].K});

        std::sort(groups.begin(), groups.end(),
                  [](const WorkGroup &a, const WorkGroup &b)
                  { return a.cost > b.cost; });

        std::vector<std::vector<WorkGroup>> per_gpu(NUM_GPUS);
        for (size_t i = 0; i < groups.size(); ++i)
            per_gpu[i % NUM_GPUS].push_back(groups[i]);

        const size_t num_shapes = SHAPES.size();
        const size_t total = ALL_PERF_FORMATS.size() * num_shapes;
        std::vector<BenchResult> results(total);
        std::atomic<int> phase2_done{0};
        const size_t total_groups = groups.size();

        {
            std::vector<std::thread> threads;
            for (int g = 0; g < NUM_GPUS; ++g)
            {
                threads.emplace_back([&, g]()
                                     {
                    (void)hipSetDevice(g);
                    for (const auto &wg : per_gpu[g])
                    {
                        const auto &fmt = ALL_PERF_FORMATS[wg.format_idx];
                        const auto &shape = SHAPES[wg.shape_idx];

                        auto weights = fmt.create(
                            static_cast<size_t>(shape.N),
                            static_cast<size_t>(shape.K));
                        GpuWeightsCache gpu_w;
                        if (weights)
                        {
                            std::vector<float> w_fp32(
                                static_cast<size_t>(shape.N) * shape.K);
                            weights->to_fp32(w_fp32.data());
                            gpu_w.upload(w_fp32.data(), shape.N, shape.K, g);
                        }

                        double ref_us = int8_ref_us.count(shape.name)
                                            ? int8_ref_us[shape.name]
                                            : 0.0;

                        auto r = benchmarkFormat(fmt, shape, kThroughputM, ref_us,
                                                 weights.get(), &gpu_w, g);

                        size_t idx = wg.format_idx * num_shapes + wg.shape_idx;
                        results[idx] = std::move(r);

                        int done = ++phase2_done;
                        fprintf(stderr, "  [GPU %d] %s/%s %.1f μs cos=%.4f %s  (%d/%zu)\n",
                                g, fmt.name.c_str(), shape.name.c_str(),
                                results[idx].min_us, results[idx].cosine_sim,
                                results[idx].correctness_pass ? "✓" : "✗",
                                done, total_groups);
                    } });
            }
            for (auto &t : threads)
                t.join();
        }

        // =========================================================================
        // Phase 3: Print per-shape comparison tables
        // =========================================================================
        for (const auto &shape : SHAPES)
        {
            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            auto ref_it = int8_ref_us.find(shape.name);
            double ref_us = (ref_it != int8_ref_us.end()) ? ref_it->second : 0.0;

            char title[256];
            snprintf(title, sizeof(title),
                     "Shape: %s (N=%d K=%d) | INT8 ref: %.1f μs",
                     shape.name.c_str(), shape.N, shape.K, ref_us);

            table << fort::header
                  << "Format" << "BPW" << "Wt KB" << "Min μs"
                  << "Speedup" << "Theoret." << "Kern Eff"
                  << "BW GB/s" << "BW Eff %" << "Cosine" << fort::endr;

            table.column(0).set_cell_text_align(fort::text_align::left);
            for (int c = 1; c <= 9; ++c)
                table.column(c).set_cell_text_align(fort::text_align::right);

            for (const auto &r : results)
            {
                if (r.shape_name != shape.name)
                    continue;

                char b_bpw[16], b_kb[16], b_min[16];
                char b_speedup[16], b_theo[16], b_keff[16];
                char b_bw[16], b_bweff[16], b_cos[16];

                snprintf(b_bpw, sizeof(b_bpw), "%.1f", r.bpw);
                snprintf(b_kb, sizeof(b_kb), "%.0f", r.weight_bytes / 1024.0);
                snprintf(b_min, sizeof(b_min), "%.1f", r.min_us);
                snprintf(b_speedup, sizeof(b_speedup), "%.2fx", r.speedup_vs_int8);
                snprintf(b_theo, sizeof(b_theo), "%.2fx", r.theoretical_speedup);
                snprintf(b_keff, sizeof(b_keff), "%.0f%%", r.kernel_efficiency);
                snprintf(b_bw, sizeof(b_bw), "%.1f", r.eff_bw_gbps);
                snprintf(b_bweff, sizeof(b_bweff), "%.1f%%", r.bw_efficiency);
                snprintf(b_cos, sizeof(b_cos), "%.4f", r.cosine_sim);

                table << r.format_name << b_bpw << b_kb << b_min
                      << b_speedup << b_theo << b_keff
                      << b_bw << b_bweff << b_cos << fort::endr;
            }

            fprintf(stderr, "\n%s\n%s\n", title, table.to_string().c_str());
        }

        // =========================================================================
        // Phase 4: Grand Summary — average across all shapes, sorted by kern eff
        // =========================================================================
        fprintf(stderr, "\n");
        fort::utf8_table summary;
        summary.set_border_style(FT_DOUBLE2_STYLE);
        summary << fort::header
                << "Format" << "BPW" << "Avg Min μs" << "Avg Speedup"
                << "Theoretical" << "Avg Kern Eff" << "Avg BW GB/s"
                << "Avg Cosine" << "Status"
                << fort::endr;

        summary.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 8; ++c)
            summary.column(c).set_cell_text_align(fort::text_align::right);

        struct FormatSummary
        {
            std::string name;
            double bpw;
            double avg_min_us;
            double avg_speedup;
            double theoretical;
            double avg_kern_eff;
            double avg_bw;
            double avg_cosine;
            bool all_pass;
        };
        std::vector<FormatSummary> format_summaries;

        for (const auto &fmt : ALL_PERF_FORMATS)
        {
            double total_min = 0.0, total_speedup = 0.0, total_keff = 0.0;
            double total_bw = 0.0, total_cos = 0.0;
            bool all_pass = true;
            int count = 0;
            for (const auto &r : results)
            {
                if (r.format_name == fmt.name)
                {
                    total_min += r.min_us;
                    total_speedup += r.speedup_vs_int8;
                    total_keff += r.kernel_efficiency;
                    total_bw += r.eff_bw_gbps;
                    total_cos += r.cosine_sim;
                    if (!r.correctness_pass)
                        all_pass = false;
                    ++count;
                }
            }
            if (count == 0)
                continue;

            format_summaries.push_back({
                fmt.name,
                fmt.bpw,
                total_min / count,
                total_speedup / count,
                8.0 / fmt.bpw,
                total_keff / count,
                total_bw / count,
                total_cos / count,
                all_pass,
            });
        }

        // Sort by kernel efficiency ascending (worst first) for tuning focus
        std::sort(format_summaries.begin(), format_summaries.end(),
                  [](const FormatSummary &a, const FormatSummary &b)
                  { return a.avg_kern_eff < b.avg_kern_eff; });

        for (const auto &fs : format_summaries)
        {
            char b_bpw[16], b_min[16], b_speedup[16], b_theo[16];
            char b_keff[16], b_bw[16], b_cos[16];
            snprintf(b_bpw, sizeof(b_bpw), "%.1f", fs.bpw);
            snprintf(b_min, sizeof(b_min), "%.1f", fs.avg_min_us);
            snprintf(b_speedup, sizeof(b_speedup), "%.2fx", fs.avg_speedup);
            snprintf(b_theo, sizeof(b_theo), "%.2fx", fs.theoretical);
            snprintf(b_keff, sizeof(b_keff), "%.0f%%", fs.avg_kern_eff);
            snprintf(b_bw, sizeof(b_bw), "%.1f", fs.avg_bw);
            snprintf(b_cos, sizeof(b_cos), "%.4f", fs.avg_cosine);
            const char *status = fs.all_pass ? "✓" : "✗";

            summary << fs.name << b_bpw << b_min << b_speedup
                    << b_theo << b_keff << b_bw << b_cos << status
                    << fort::endr;
        }

        fprintf(stderr, "GRAND SUMMARY: Average across all shapes (sorted by Kern Eff ascending — worst first)\n");
        fprintf(stderr, "%s\n", summary.to_string().c_str());
        fprintf(stderr, "Speedup = INT8_time / format_time (>1x = faster than INT8)\n");
        fprintf(stderr, "Theoretical = 8.0/BPW (ideal speedup from bandwidth savings alone)\n");
        fprintf(stderr, "Kern Eff = Speedup/Theoretical × 100%% (how close to bandwidth-optimal)\n");
        fprintf(stderr, "Cosine = GPU output vs HipBLAS FP32 reference (gate: >= %.4f for all formats)\n",
                COSINE_SIM_GATE);

        // Validate correctness (per-format gate)
        for (const auto &r : results)
        {
            const float gate = cosine_gate_for(r.format_name);
            EXPECT_GT(r.min_us, 0.0)
                << r.format_name << "/" << r.shape_name
                << " did not execute the production NativeVNNI path";
            EXPECT_TRUE(r.correctness_pass)
                << r.format_name << "/" << r.shape_name
                << " failed its FP32-reference correctness gate";
            EXPECT_GE(r.cosine_sim, gate)
                << r.format_name << "/" << r.shape_name
                << " cosine=" << r.cosine_sim
                << " (gate=" << gate << ")";
        }
#endif
    }

    // =============================================================================
    // Test: BPW-vs-bandwidth scaling curve (focused)
    // =============================================================================

    TEST_F(NativeVNNIPerfTest, BPW_Scaling_7B_FFN_Down)
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "HAVE_ROCM not defined";
#else
        if (!has_device_)
            GTEST_SKIP() << "No ROCm device available";

        // Largest realistic shape — best for exposing bandwidth differences
        const GEMVShape shape{"7B_FFN_Dn", 3584, 18944};

        fprintf(stderr, "\n[NativeVNNI Perf] BPW Scaling Curve\n");
        fprintf(stderr, "[NativeVNNI Perf] Shape: %s (N=%d K=%d) — largest GEMV shape\n",
                shape.name.c_str(), shape.N, shape.K);

        // Select representative formats spanning the BPW range
        const std::vector<std::string> selected = {
            "IQ2_XXS",
            "IQ2_XS",
            "Q2_K",
            "IQ3_XXS",
            "Q3_K",
            "Q4_0",
            "Q4_K",
            "Q5_0",
            "Q5_K",
            "Q6_K",
        };

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Format" << "BPW" << "Weight MB" << "Min μs" << "BW GB/s"
              << "BW Eff %" << "Bytes/Elem" << "Cosine" << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 7; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &sel_name : selected)
        {
            auto it = std::find_if(ALL_PERF_FORMATS.begin(), ALL_PERF_FORMATS.end(),
                                   [&](const PerfFormatSpec &f)
                                   { return f.name == sel_name; });
            if (it == ALL_PERF_FORMATS.end())
                continue;

            auto weights = it->create(
                static_cast<size_t>(shape.N), static_cast<size_t>(shape.K));
            GpuWeightsCache gpu_w;
            if (weights)
            {
                std::vector<float> w_fp32(static_cast<size_t>(shape.N) * shape.K);
                weights->to_fp32(w_fp32.data());
                gpu_w.upload(w_fp32.data(), shape.N, shape.K, 0);
            }

            constexpr int kInspectorM = 1;
            auto r = benchmarkFormat(*it, shape, kInspectorM, 0.0, weights.get(), &gpu_w, 0);

            double bytes_per_elem = r.weight_bytes / (static_cast<double>(r.N) * r.K);

            char buf_bpw[16], buf_mb[16], buf_min[16], buf_bw[16];
            char buf_eff[16], buf_bpe[16], buf_cos[16];
            snprintf(buf_bpw, sizeof(buf_bpw), "%.1f", r.bpw);
            snprintf(buf_mb, sizeof(buf_mb), "%.2f", r.weight_bytes / (1024.0 * 1024.0));
            snprintf(buf_min, sizeof(buf_min), "%.1f", r.min_us);
            snprintf(buf_bw, sizeof(buf_bw), "%.1f", r.eff_bw_gbps);
            snprintf(buf_eff, sizeof(buf_eff), "%.1f%%", r.bw_efficiency);
            snprintf(buf_bpe, sizeof(buf_bpe), "%.3f", bytes_per_elem);
            snprintf(buf_cos, sizeof(buf_cos), "%.4f", r.cosine_sim);

            table << r.format_name << buf_bpw << buf_mb << buf_min << buf_bw
                  << buf_eff << buf_bpe << buf_cos << fort::endr;
        }

        fprintf(stderr, "\n%s\n", table.to_string().c_str());
        fprintf(stderr, "Expected: lower BPW = less data = lower μs (if decode ALU < BW savings)\n");
        fprintf(stderr, "HBM2 peak bandwidth reference: %.0f GB/s\n", HBM2_PEAK_GBPS);
#endif
    }

} // anonymous namespace
