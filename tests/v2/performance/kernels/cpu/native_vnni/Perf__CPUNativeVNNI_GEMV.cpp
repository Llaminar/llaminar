/**
 * @file Perf__CPUNativeVNNI_GEMV.cpp
 * @brief Comprehensive GEMV decode performance sweep across all Qwen model
 *        sizes (0.5B through 32B), TP degrees (1/2/4), and both quantized
 *        production eager-interleaved Q8_0 GEMV path.
 *
 * For each (model, shape, TP degree):
 *   1. Report tile config (nbc, k_tiles, category)
 *   2. Benchmark packed VNNI GEMV (FP32→Q8_1 quantize + INT8 packed access)
 *   3. Report latency, effective bandwidth, and roofline utilization
 *
 * Also decomposes per-GEMV overhead:
 *   - Activation quantization (FP32→Q8_1 per call)
 *   - Kernel compute time
 *   - Total including overhead
 *
 * System roofline: Calibrated at startup via STREAM-like DRAM read benchmark.
 * Cold cache: weight data flushed via clflushopt before each timed iteration.
 *
 * @note Run with Release build:
 *   ctest -R V2_Perf_CPUNativeVNNI_GEMV --verbose
 *   Individual tests: --gtest_filter="*Q8_0_AllModels*"
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unistd.h>
#include <vector>

#include "kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h"
#include "kernels/cpu/native_vnni/CPUNativeVNNITileConfig.h"
#include "kernels/cpu/native_vnni/CPUNativeVNNITraits.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"
#include "fort.hpp"

#include "utils/NativeVNNITrainerEvidence.h"
#include "utils/NativeVNNIPrefillProbePlan.h"
#include "utils/TestTensorFactory.h"
#include "utils/VerifierRowTestInventory.h"
#include "../../native_vnni_dispatch/NativeVNNIPairedRequestManifest.h"
#include "../../native_vnni_dispatch/NativeVNNIShapeManifest.h"
#include "../../native_vnni_dispatch/NativeVNNIProfilerControl.h"

using namespace llaminar2;
using namespace llaminar2::cpu::native_vnni;
using namespace llaminar2::test;

namespace
{

    // =========================================================================
    // MPI environment
    // =========================================================================
    void mpi_abort_handler(int sig)
    {
        const char *msg = "\n[FATAL] Signal in GEMV perf test — MPI_Abort\n";
        [[maybe_unused]] auto _ = write(STDERR_FILENO, msg, strlen(msg));
        MPI_Abort(MPI_COMM_WORLD, sig);
        _exit(128 + sig);
    }

    class MPIEnvironment : public ::testing::Environment
    {
    public:
        void SetUp() override
        {
            int initialized = 0;
            MPI_Initialized(&initialized);
            if (!initialized)
                MPI_Init(nullptr, nullptr);
            std::signal(SIGSEGV, mpi_abort_handler);
            std::signal(SIGABRT, mpi_abort_handler);
        }
        void TearDown() override
        {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized)
                MPI_Finalize();
        }
    };

    static auto *g_mpi_env [[maybe_unused]] =
        ::testing::AddGlobalTestEnvironment(new MPIEnvironment);

    /**
     * @brief Coordinates complete candidate rounds across CPU measurement ranks.
     *
     * The production prefill collector runs one process per physical socket.
     * Process-local complete rounds are not sufficient in that topology: one
     * rank can finish a cheap cell, advance to a larger M, and change the peer
     * socket's power or memory-pressure regime while that peer is still timing
     * its previous cell.  The resulting latency transition is real, but it is
     * not evidence about either candidate's steady-state economy.
     *
     * When enabled, this coordinator performs one collective after every local
     * complete round.  A rank whose candidates have all converged continues to
     * launch unmeasured pacing rounds until every rank is ready.  Consequently
     * no rank can advance to the next M while a peer is still collecting the
     * current M.  A one-rank MPI world follows exactly the same code path and
     * reduces to a local decision.
     */
    class MPIPrefillRoundCoordinator
    {
    public:
        /**
         * @param enabled Whether cross-rank complete-round synchronization is required.
         */
        explicit MPIPrefillRoundCoordinator(bool enabled)
            : enabled_(enabled)
        {
            int initialized = 0;
            MPI_Initialized(&initialized);
            if (!initialized)
                throw std::runtime_error(
                    "CPU prefill MPI round coordination requires MPI_Init");
            checkMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank_), "MPI_Comm_rank");
            checkMPI(MPI_Comm_size(MPI_COMM_WORLD, &world_size_), "MPI_Comm_size");
        }

        /** Return the stable provenance label written into trainer evidence. */
        const char *policyName() const
        {
            return enabled_
                ? "mpi-complete-round-v1"
                : "process-local-complete-round-v1";
        }

        /** Return this process's rank in the measurement communicator. */
        int rank() const { return rank_; }

        /** Return the number of socket-local processes in the measurement job. */
        int worldSize() const { return world_size_; }

        /**
         * @brief Prove every rank visits the same format and M phase inventory.
         *
         * Complete-round collectives would deadlock if paired jobs used
         * different format counts or M inventories. The refresh plan groups
         * identical phase vectors, and this runtime handshake independently
         * rejects a broken schedule before either rank enters the first
         * per-format/per-M collective.
         */
        void requireMatchingPhaseInventory(
            size_t format_phase_count,
            const std::vector<int> &m_values) const
        {
            if (!enabled_ || world_size_ == 1)
                return;

            uint64_t digest = 1469598103934665603ULL;
            digest ^= static_cast<uint64_t>(format_phase_count);
            digest *= 1099511628211ULL;
            for (int m : m_values)
            {
                digest ^= static_cast<uint64_t>(m);
                digest *= 1099511628211ULL;
            }
            digest ^= static_cast<uint64_t>(m_values.size());
            digest *= 1099511628211ULL;

            uint64_t minimum_digest = 0;
            uint64_t maximum_digest = 0;
            checkMPI(
                MPI_Allreduce(
                    &digest, &minimum_digest, 1, MPI_UINT64_T,
                    MPI_MIN, MPI_COMM_WORLD),
                "MPI_Allreduce(min M inventory)");
            checkMPI(
                MPI_Allreduce(
                    &digest, &maximum_digest, 1, MPI_UINT64_T,
                    MPI_MAX, MPI_COMM_WORLD),
                "MPI_Allreduce(max M inventory)");
            if (minimum_digest != maximum_digest)
                throw std::runtime_error(
                    "CPU prefill MPI ranks received different format/M phase inventories");
        }

        /** Wait until every rank has reached the same measurement boundary. */
        void barrier() const
        {
            if (enabled_ && world_size_ > 1)
                checkMPI(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
        }

        /**
         * @brief Return true while any rank still needs measured local rounds.
         *
         * All ranks call this method at the same round boundary.  Ranks that
         * report false locally still execute pacing rounds while the collective
         * result remains true.
         */
        bool anyRankActive(bool locally_active) const
        {
            if (!enabled_ || world_size_ == 1)
                return locally_active;
            const int local = locally_active ? 1 : 0;
            int global = 0;
            checkMPI(
                MPI_Allreduce(
                    &local, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD),
                "MPI_Allreduce(active rounds)");
            return global != 0;
        }

        /**
         * @brief Fail the complete measurement job when any rank reports failure.
         *
         * A normal GTest fatal assertion returns only from the local test body.
         * Its peer can then enter the next-M barrier while the failed rank enters
         * MPI_Finalize, producing mismatched collectives and a permanent hang.
         * This reduction gives every rank the same decision point and aborts the
         * communicator before either process can advance independently.
         */
        void failAllRanksIf(
            bool local_failure,
            const std::string &local_diagnostic) const
        {
            const bool global_failure = anyRankActive(local_failure);
            if (!global_failure)
                return;
            if (local_failure && !local_diagnostic.empty())
                std::fprintf(stderr, "%s\n", local_diagnostic.c_str());
            if (enabled_ && world_size_ > 1)
            {
                std::fflush(stderr);
                MPI_Abort(MPI_COMM_WORLD, 1);
                std::abort();
            }
            throw std::runtime_error(
                local_diagnostic.empty()
                    ? "a peer CPU prefill rank rejected its timing evidence"
                    : local_diagnostic);
        }

    private:
        static void checkMPI(int error, const char *operation)
        {
            if (error != MPI_SUCCESS)
                throw std::runtime_error(
                    std::string(operation) + " failed during CPU prefill training");
        }

        bool enabled_ = false;
        int rank_ = 0;
        int world_size_ = 1;
    };

    // =========================================================================
    // System roofline — measured via STREAM-like calibration
    // =========================================================================

    // Forward declaration (defined after GEMVResult)
    static inline void flush_cache_range(const void *ptr, size_t bytes);

    /**
     * @brief Measure actual DRAM read bandwidth using a STREAM-like benchmark.
     *
     * Allocates a large buffer (256 MB, well beyond L3), flushes it from cache,
     * then performs a multi-threaded sequential read pass. Repeated 5 times;
     * reports the best (peak sustainable) bandwidth in GB/s.
     *
     * This replaces the hardcoded 60 GB/s constant with a machine-calibrated
     * value, making efficiency percentages meaningful across different hardware.
     */
    static double calibrateDramBandwidth()
    {
        constexpr size_t BUF_SIZE = 256 * 1024 * 1024; // 256 MB
        constexpr int CALIB_RUNS = 5;

        // Allocate page-aligned buffer
        auto *buf = static_cast<char *>(std::aligned_alloc(4096, BUF_SIZE));
        if (!buf)
            return 60.0; // fallback

// First-touch: each thread touches its own pages for NUMA locality
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < BUF_SIZE; i += 4096)
            buf[i] = static_cast<char>(i & 0xFF);

        double best_gbs = 0;
        for (int run = 0; run < CALIB_RUNS; ++run)
        {
            // Flush entire buffer from cache
            flush_cache_range(buf, BUF_SIZE);

            // Timed parallel read — accumulate to prevent dead-code elimination
            volatile uint64_t sink = 0;
            auto t0 = std::chrono::high_resolution_clock::now();

            uint64_t local_sum = 0;
#pragma omp parallel reduction(+ : local_sum)
            {
                int tid = omp_get_thread_num();
                int nthreads = omp_get_num_threads();
                size_t chunk = BUF_SIZE / nthreads;
                size_t start = tid * chunk;
                size_t end = (tid == nthreads - 1) ? BUF_SIZE : start + chunk;
                const uint64_t *p = reinterpret_cast<const uint64_t *>(buf + start);
                const uint64_t *pe = reinterpret_cast<const uint64_t *>(buf + end);
                uint64_t acc = 0;
                while (p < pe)
                {
                    acc += *p;
                    p += 8; // stride 64 bytes (one cache line)
                }
                local_sum += acc;
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            sink = local_sum; // prevent optimization
            (void)sink;

            double secs = std::chrono::duration<double>(t1 - t0).count();
            double gbs = (double)BUF_SIZE / secs / 1e9;
            best_gbs = std::max(best_gbs, gbs);
        }

        std::free(buf);
        return best_gbs;
    }

    // Cached calibrated bandwidth (computed once, shared across all tests)
    static double s_calibrated_bw_gbs = 0;
    static volatile int64_t s_vnni_floor_sink = 0;

    static double systemBandwidthGB()
    {
        if (s_calibrated_bw_gbs <= 0)
        {
            s_calibrated_bw_gbs = calibrateDramBandwidth();
            std::cout << "\n=== DRAM Bandwidth Calibration ===\n"
                      << "Measured: " << std::fixed << std::setprecision(1)
                      << s_calibrated_bw_gbs << " GB/s"
                      << "  (threads=" << omp_get_max_threads() << ")\n"
                      << std::endl;
        }
        return s_calibrated_bw_gbs;
    }

    // =========================================================================
    // Model definitions — Qwen2.5 family (0.5B through 32B)
    // =========================================================================
    struct ModelConfig
    {
        std::string name;
        int d_model;
        int n_heads;
        int n_kv_heads;
        int head_dim;
        int d_ff;
        int vocab; // vocabulary size for LM_Head
    };

    static const std::vector<ModelConfig> ALL_MODELS = {
        {"0.5B", 896, 14, 2, 64, 4864, 151936},
        {"1.5B", 1536, 12, 2, 128, 8960, 151936},
        {"3B", 2048, 16, 2, 128, 11008, 151936},
        {"7B", 3584, 28, 4, 128, 18944, 152064},
        {"14B", 5120, 40, 8, 128, 13824, 152064},
        {"32B", 5120, 40, 8, 128, 27648, 152064},
    };

    struct FormatSpec
    {
        std::string name;
    };

    static const std::vector<FormatSpec> MTP_SMALL_M_FORMATS = {
        {"Q4_0"}, {"Q4_1"}, {"Q5_0"}, {"Q5_1"},
        {"Q8_0"}, {"Q8_1"}, {"Q8_K"}, {"Q2_K"}, {"Q3_K"},
        {"Q4_K"}, {"Q5_K"}, {"Q6_K"}, {"IQ4_NL"},
        {"IQ4_XS"}, {"IQ3_S"}, {"IQ3_XXS"}, {"IQ2_S"},
        {"IQ2_XS"}, {"IQ2_XXS"}, {"IQ1_S"}, {"IQ1_M"},
    };

    std::string trim(std::string value)
    {
        const auto begin = value.find_first_not_of(" \t\n\r");
        if (begin == std::string::npos)
            return {};
        const auto end = value.find_last_not_of(" \t\n\r");
        return value.substr(begin, end - begin + 1);
    }

    std::string toLower(std::string value)
    {
        std::transform(
            value.begin(),
            value.end(),
            value.begin(),
            [](unsigned char c)
            {
                return static_cast<char>(std::tolower(c));
            });
        return value;
    }

    /**
     * @brief Read an optional environment variable as a trimmed string.
     */
    std::string getEnvString(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return {};
        return trim(raw);
    }

    /**
     * @brief Read an optional environment variable as an integer.
     */
    std::optional<int> getEnvInt(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return std::nullopt;
        return std::atoi(raw);
    }

    /**
     * @brief Read an optional environment variable as a floating-point value.
     */
    std::optional<double> getEnvDouble(const char *name)
    {
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return std::nullopt;
        return std::atof(raw);
    }

    /**
     * @brief Temporarily force one production CPU NativeVNNI tile parameter.
     *
     * Tile configuration is cached by `DebugEnv`; changing only the process
     * environment would therefore leave the trainer measuring a stale route.
     * This guard updates both owners on entry and restores both on exit so a
     * candidate trial cannot contaminate the next candidate or test.
     */
    class ScopedCPUNativeVNNITileOverride
    {
    public:
        ScopedCPUNativeVNNITileOverride(const char *name, std::string value)
            : name_(name),
              had_previous_(std::getenv(name) != nullptr),
              previous_(had_previous_ ? std::getenv(name) : "")
        {
            setenv(name_.c_str(), value.c_str(), 1);
            mutableDebugEnv().cpu_vnni.reload();
        }

        ~ScopedCPUNativeVNNITileOverride()
        {
            if (had_previous_)
                setenv(name_.c_str(), previous_.c_str(), 1);
            else
                unsetenv(name_.c_str());
            mutableDebugEnv().cpu_vnni.reload();
        }

        ScopedCPUNativeVNNITileOverride(
            const ScopedCPUNativeVNNITileOverride &) = delete;
        ScopedCPUNativeVNNITileOverride &operator=(
            const ScopedCPUNativeVNNITileOverride &) = delete;

    private:
        std::string name_;
        bool had_previous_;
        std::string previous_;
    };

    /**
     * @brief Keep standalone verifier microbench runs on a representative CPU team.
     *
     * Production inference and CTest normally enter through the launcher/MPI
     * wrapper, which sets socket-aware OpenMP placement.  Directly launching
     * this perf binary on a large dual-socket host leaves libgomp free to use
     * every hardware thread, turning the tiny M=2..4 verifier probe into a
     * scheduler and cross-socket migration benchmark.  Explicit user settings
     * remain authoritative; this helper only supplies a sane local default for
     * the verifier-row microbench when no threading policy was provided.
     */
    void applyVerifierRowsThreadCapForStandalonePerf()
    {
        if (const auto forced = getEnvInt("LLAMINAR_CPU_NVNNI_VERIFIER_THREADS");
            forced.has_value() && *forced > 0)
        {
            omp_set_num_threads(*forced);
            return;
        }

        const char *omp_threads = std::getenv("OMP_NUM_THREADS");
        if (omp_threads && *omp_threads)
            return;

        constexpr int kStandaloneThreadCap = 32;
        const int max_threads = omp_get_max_threads();
        if (max_threads > kStandaloneThreadCap)
        {
            omp_set_num_threads(kStandaloneThreadCap);
            std::fprintf(
                stderr,
                "[CPUNativeVNNI][VERIFIER_ROWS] OMP_NUM_THREADS unset; "
                "capping standalone perf probe from %d to %d threads. "
                "Set OMP_NUM_THREADS or LLAMINAR_CPU_NVNNI_VERIFIER_THREADS "
                "to override.\n",
                max_threads,
                kStandaloneThreadCap);
        }
    }

    /**
     * @brief Capture the exact persistent OpenMP team used by one CPU launch.
     *
     * Linux performance events attached to the process leader cannot be reset
     * atomically across already-inherited worker events. The profiler instead
     * opens one event group against every concrete worker TID after warmup has
     * established libgomp's team. Indexing by OpenMP thread number keeps the
     * main thread at element zero, which lets `LinuxPerfControl` exclude its
     * own enable/disable syscalls from the measured main-thread interval.
     *
     * @return Worker TIDs ordered by stable OpenMP thread number.
     */
    std::vector<pid_t> captureOpenMPProfilerThreadIds()
    {
        std::vector<pid_t> thread_ids(
            static_cast<size_t>(omp_get_max_threads()),
            static_cast<pid_t>(-1));
#pragma omp parallel shared(thread_ids)
        {
            const int thread_number = omp_get_thread_num();
            thread_ids[static_cast<size_t>(thread_number)] =
                static_cast<pid_t>(::syscall(SYS_gettid));
        }
        if (std::find(thread_ids.begin(), thread_ids.end(), -1) !=
            thread_ids.end())
        {
            throw std::runtime_error(
                "OpenMP profiler failed to capture every worker TID");
        }
        return thread_ids;
    }

    /**
     * @brief One exact request in a process-amortized CPU profiler batch.
     *
     * The TSV repeats every launch discriminator that the CPU trainer can
     * observe. It intentionally carries the request-specific output path so a
     * reused OpenMP team can never merge two counter intervals into one file.
     */
    struct CPUProfilerBatchRequest
    {
        std::string request_id;
        std::string operation_kind;
        std::string source_format;
        std::string execution_mode;
        int m = 0;
        std::string projection_n_vector;
        int aggregate_n = 0;
        int k = 0;
        std::string effective_candidate_id;
        std::string output_path;
        bool consumed = false;
    };

    /**
     * @brief Load and account for exact CPU profiler requests in this process.
     *
     * Batching is only a setup optimization. ``claim()`` performs a complete
     * identity match, and ``requireComplete()`` prevents a successful process
     * from silently omitting one planned launch. The hardware control object
     * is rebound and reset separately for every claimed row.
     */
    class CPUProfilerBatch
    {
    public:
        CPUProfilerBatch()
        {
            const std::string path = native_vnni_dispatch::profilerEnvironment(
                native_vnni_dispatch::kProfilerBatchPathEnvironment);
            if (path.empty())
                return;

            std::ifstream input(path);
            if (!input)
                throw std::runtime_error(
                    "failed to open CPU profiler batch plan: " + path);
            std::string line;
            if (!std::getline(input, line) || line != kHeader)
                throw std::runtime_error(
                    "CPU profiler batch plan has an invalid schema header");
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
                if (fields.size() != 10)
                    throw std::runtime_error(
                        "CPU profiler batch row must contain ten TSV fields");
                CPUProfilerBatchRequest request{
                    .request_id = fields[0],
                    .operation_kind = fields[1],
                    .source_format = fields[2],
                    .execution_mode = fields[3],
                    .m = parsePositive(fields[4], "M"),
                    .projection_n_vector = fields[5],
                    .aggregate_n = parsePositive(fields[6], "N"),
                    .k = parsePositive(fields[7], "K"),
                    .effective_candidate_id = fields[8],
                    .output_path = fields[9],
                };
                if (request.request_id.empty() ||
                    request.operation_kind.empty() ||
                    request.source_format.empty() ||
                    request.execution_mode.empty() ||
                    request.projection_n_vector.empty() ||
                    request.effective_candidate_id.empty() ||
                    request.output_path.empty())
                {
                    throw std::runtime_error(
                        "CPU profiler batch row contains an empty identity field");
                }
                if (!request_ids.insert(request.request_id).second)
                    throw std::runtime_error(
                        "CPU profiler batch request IDs must be unique");
                if (!output_paths.insert(request.output_path).second)
                    throw std::runtime_error(
                        "CPU profiler batch output paths must be unique");
                requests_.push_back(std::move(request));
            }
            if (requests_.empty())
                throw std::runtime_error("CPU profiler batch plan is empty");
        }

        [[nodiscard]] bool enabled() const noexcept
        {
            return !requests_.empty();
        }

        [[nodiscard]] size_t size() const noexcept
        {
            return requests_.size();
        }

        /** Return whether the batch contains any candidate for one work cell. */
        [[nodiscard]] bool containsCell(
            std::string_view operation_kind,
            std::string_view source_format,
            std::string_view execution_mode,
            int m,
            int n,
            int k) const
        {
            return std::any_of(
                requests_.begin(), requests_.end(),
                [&](const CPUProfilerBatchRequest &request)
                {
                    return matchesCell(
                        request, operation_kind, source_format, execution_mode,
                        m, n, k);
                });
        }

        /** Claim exactly one request for the production candidate invocation. */
        CPUProfilerBatchRequest *claim(
            std::string_view operation_kind,
            std::string_view source_format,
            std::string_view execution_mode,
            int m,
            int n,
            int k,
            std::string_view effective_candidate_id)
        {
            CPUProfilerBatchRequest *result = nullptr;
            for (CPUProfilerBatchRequest &request : requests_)
            {
                if (!matchesCell(
                        request, operation_kind, source_format, execution_mode,
                        m, n, k) ||
                    request.effective_candidate_id != effective_candidate_id)
                {
                    continue;
                }
                if (result != nullptr)
                    throw std::runtime_error(
                        "CPU profiler batch has duplicate exact launch identities");
                result = &request;
            }
            if (result && result->consumed)
                throw std::runtime_error(
                    "CPU profiler batch request was claimed more than once: " +
                    result->request_id);
            if (result)
                result->consumed = true;
            return result;
        }

        /** Fail the process if any exact batch member did not launch once. */
        void requireComplete() const
        {
            for (const CPUProfilerBatchRequest &request : requests_)
            {
                if (!request.consumed)
                    throw std::runtime_error(
                        "CPU profiler batch omitted request " +
                        request.request_id);
            }
        }

        /** Return the first request for constructing the reusable event owner. */
        const CPUProfilerBatchRequest &first() const
        {
            if (requests_.empty())
                throw std::runtime_error("CPU profiler batch is disabled");
            return requests_.front();
        }

    private:
        static constexpr std::string_view kHeader =
            "request_id\toperation_kind\tsource_format\texecution_mode\tm\t"
            "projection_n_vector\taggregate_n\tk\t"
            "effective_candidate_id\toutput_path";

        static int parsePositive(const std::string &value, const char *name)
        {
            size_t consumed = 0;
            const int parsed = std::stoi(value, &consumed);
            if (consumed != value.size() || parsed <= 0)
                throw std::runtime_error(
                    std::string("CPU profiler batch has invalid ") + name);
            return parsed;
        }

        static bool matchesCell(
            const CPUProfilerBatchRequest &request,
            std::string_view operation_kind,
            std::string_view source_format,
            std::string_view execution_mode,
            int m,
            int n,
            int k)
        {
            return request.operation_kind == operation_kind &&
                   request.source_format == source_format &&
                   request.execution_mode == execution_mode &&
                   request.m == m && request.aggregate_n == n &&
                   request.k == k &&
                   request.projection_n_vector == std::to_string(n);
        }

        std::vector<CPUProfilerBatchRequest> requests_;
    };

    /**
     * @brief Profile one exact CPU request using reusable per-worker events.
     */
    template <typename Launch>
    void profileExactCPURequest(
        native_vnni_dispatch::LinuxPerfControl &control,
        const std::string &request_id,
        const std::string &output_path,
        Launch &&launch)
    {
        control.rebind(request_id, output_path);
        if (!control.prepared())
            control.prepare(captureOpenMPProfilerThreadIds());
        const auto parallel_perf_control = [](auto &&action)
        {
#pragma omp parallel
            {
                action(static_cast<size_t>(omp_get_thread_num()));
            }
        };
        control.begin(parallel_perf_control);
        launch();
        control.end(parallel_perf_control);
    }

    /**
     * @brief Read a comma-separated environment variable as lowercase tokens.
     */
    std::set<std::string> getEnvCsvSet(const char *name)
    {
        std::set<std::string> values;
        const char *raw = std::getenv(name);
        if (!raw || *raw == '\0')
            return values;

        std::stringstream ss(raw);
        std::string item;
        while (std::getline(ss, item, ','))
        {
            item = toLower(trim(item));
            if (!item.empty())
                values.insert(item);
        }
        return values;
    }

    /**
     * @brief Return true when a test case name is allowed by an optional filter.
     */
    bool shouldRunName(const std::set<std::string> &filters, const std::string &name)
    {
        return filters.empty() || filters.count(toLower(name)) > 0;
    }

    struct VectorMetrics
    {
        double cosine = 1.0;
        double relative_l2 = 0.0;
        double symmetric_kl = 0.0;
        double max_abs = 0.0;
    };

    /**
     * @brief Compare two verifier output tensors with distribution-aware metrics.
     *
     * Relative L2 alone can miss row permutations and near-tie shifts that are
     * obvious once the values are interpreted as logits.  The verifier-row
     * microbench therefore tracks cosine similarity and symmetric KL as well,
     * mirroring the stricter Phase 9.8 acceptance checks.
     */
    VectorMetrics computeVectorMetrics(const float *actual, const float *expected, size_t count)
    {
        VectorMetrics metrics{};
        if (count == 0)
            return metrics;

        double dot = 0.0;
        double actual_norm = 0.0;
        double expected_norm = 0.0;
        double diff_norm = 0.0;
        metrics.max_abs = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            const double a = actual[i];
            const double e = expected[i];
            const double d = a - e;
            dot += a * e;
            actual_norm += a * a;
            expected_norm += e * e;
            diff_norm += d * d;
            metrics.max_abs = std::max(metrics.max_abs, std::abs(d));
        }
        if (actual_norm <= 1.0e-30 && expected_norm <= 1.0e-30)
        {
            metrics.cosine = 1.0;
        }
        else
        {
            metrics.cosine =
                dot / (std::sqrt(actual_norm) * std::sqrt(expected_norm) + 1.0e-30);
        }
        metrics.relative_l2 =
            std::sqrt(diff_norm) / (std::sqrt(expected_norm) + 1.0e-30);

        const double actual_max =
            static_cast<double>(*std::max_element(actual, actual + count));
        const double expected_max =
            static_cast<double>(*std::max_element(expected, expected + count));
        double actual_sum = 0.0;
        double expected_sum = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            actual_sum += std::exp(static_cast<double>(actual[i]) - actual_max);
            expected_sum += std::exp(static_cast<double>(expected[i]) - expected_max);
        }

        constexpr double eps = 1.0e-300;
        double kl_actual_expected = 0.0;
        double kl_expected_actual = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            const double p =
                std::exp(static_cast<double>(actual[i]) - actual_max) /
                (actual_sum + eps);
            const double q =
                std::exp(static_cast<double>(expected[i]) - expected_max) /
                (expected_sum + eps);
            kl_actual_expected += p * std::log((p + eps) / (q + eps));
            kl_expected_actual += q * std::log((q + eps) / (p + eps));
        }
        metrics.symmetric_kl = 0.5 * (kl_actual_expected + kl_expected_actual);
        return metrics;
    }

    void assertVerifierMetricsStrict(
        const VectorMetrics &metrics,
        const std::string &label)
    {
        EXPECT_GE(metrics.cosine, 0.999999)
            << label << " cosine=" << metrics.cosine
            << " relative_l2=" << metrics.relative_l2
            << " symmetric_kl=" << metrics.symmetric_kl
            << " max_abs=" << metrics.max_abs;
        EXPECT_LE(metrics.relative_l2, 1.0e-6)
            << label << " cosine=" << metrics.cosine
            << " symmetric_kl=" << metrics.symmetric_kl
            << " max_abs=" << metrics.max_abs;
        EXPECT_LE(metrics.symmetric_kl, 1.0e-9)
            << label << " cosine=" << metrics.cosine
            << " relative_l2=" << metrics.relative_l2
            << " max_abs=" << metrics.max_abs;
    }

    struct TimingSummary
    {
        double min_us = 0.0;
        double mean_us = 0.0;
        double stddev_us = 0.0;
    };

    template <typename Fn>
    TimingSummary timeMicrobench(int warmup, int iterations, Fn &&fn)
    {
        for (int i = 0; i < warmup; ++i)
            fn();

        std::vector<double> samples;
        samples.reserve(static_cast<size_t>(iterations));
        for (int i = 0; i < iterations; ++i)
        {
            const auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            const auto t1 = std::chrono::high_resolution_clock::now();
            samples.push_back(
                std::chrono::duration<double, std::micro>(t1 - t0).count());
        }

        TimingSummary summary{};
        summary.min_us = *std::min_element(samples.begin(), samples.end());
        summary.mean_us =
            std::accumulate(samples.begin(), samples.end(), 0.0) /
            static_cast<double>(samples.size());
        double variance = 0.0;
        for (double value : samples)
        {
            const double delta = value - summary.mean_us;
            variance += delta * delta;
        }
        summary.stddev_us = std::sqrt(variance / static_cast<double>(samples.size()));
        return summary;
    }

    /** Exact sorted samples and common robust aggregates for policy training. */
    struct StrongTimingMeasurement
    {
        std::vector<double> acquisition_samples_us;
        std::vector<double> samples_us;
        llaminar2::test::trainer::TimingEvidence evidence;
        llaminar2::test::trainer::AdaptiveTimingDecision adaptive;
        std::string stop_reason = "fixed_samples";
    };

    /**
     * @brief Select deterministic serial-M1 oracle rows for a prefill timing cell.
     *
     * The dedicated all-format correctness sweep owns exhaustive byte equality.
     * Replaying all M serial GEMVs again inside every performance cell made the
     * Cartesian trainer spend days recomputing an already certified oracle.  A
     * timing cell instead checks row-pair boundaries, quartile boundaries, and
     * the final rows.  These positions expose launch-grid and row-index defects
     * while bounding oracle work independently of long-context M.
     *
     * @param M Number of rows published by the candidate launch.
     * @param require_exhaustive_oracle When true, return every row in natural
     *        order for grouped-candidate admission. When false, return the
     *        bounded long-prefill witness set used by corpus collection.
     * @return Ordered, unique row indices to replay through serial M1 decode.
     */
    std::vector<int> prefillSerialOracleRows(
        int M,
        bool require_exhaustive_oracle = false)
    {
        if (require_exhaustive_oracle)
        {
            std::vector<int> all_rows(static_cast<size_t>(M));
            std::iota(all_rows.begin(), all_rows.end(), 0);
            return all_rows;
        }

        std::set<int> selected;
        const auto include = [&](int row)
        {
            if (row >= 0 && row < M)
                selected.insert(row);
        };
        for (int row = 0; row < std::min(M, 4); ++row)
            include(row);
        for (int anchor : {M / 4, M / 2, (3 * M) / 4})
        {
            include(anchor - 1);
            include(anchor);
        }
        for (int row = std::max(0, M - 6); row < M; ++row)
            include(row);
        return {selected.begin(), selected.end()};
    }

    /**
     * @brief Owns deterministic Q8_1 activation fixtures shared by formats.
     *
     * The ordinary-prefill trainer historically seeded an activation from
     * ``(M, N, K, source-format-name length)``. Source formats with the same
     * name length therefore quantized exactly the same bytes, but separate
     * trainer processes rebuilt those bytes independently. A coalesced process
     * retains that historical seed contract and memoizes the quantized result.
     * Weight packing and every production kernel launch remain format-local.
     *
     * Fixtures are released after the final selected format in one name-length
     * class. This bounds retained memory to the format classes that are still
     * reachable while allowing all M values in that class to reuse quantization.
     */
    class PrefillActivationFixtureCache
    {
    public:
        /** Construct a cache for one fixed trainer geometry. */
        PrefillActivationFixtureCache(int n, int k)
            : n_(n), k_(k)
        {
            if (n_ <= 0 || k_ <= 0)
                throw std::invalid_argument(
                    "CPU prefill activation fixture geometry must be positive");
        }

        /**
         * @brief Return the byte-identical legacy fixture for one format/M.
         *
         * @param m Number of activation rows.
         * @param source_format Source GGUF tensor format label.
         * @param k_blocks Number of Q8_1 blocks expected by packed weights.
         */
        const std::vector<Q8_1Block> &fixtureFor(
            int m,
            std::string_view source_format,
            int k_blocks)
        {
            if (m <= 0 || k_blocks <= 0 ||
                static_cast<size_t>(k_blocks) * Q8_1Block::BLOCK_SIZE !=
                    static_cast<size_t>(k_))
            {
                throw std::invalid_argument(
                    "CPU prefill activation fixture block geometry is invalid");
            }
            const uint64_t key = fixtureKey(m, source_format.size());
            auto [iterator, inserted] = fixtures_.try_emplace(key);
            if (!inserted)
                return iterator->second;

            std::mt19937 rng(static_cast<uint32_t>(
                0xB47Cu + m * 131u + n_ + k_ + source_format.size()));
            std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
            std::vector<float> input(static_cast<size_t>(m) * k_);
            for (float &value : input)
                value = distribution(rng);

            std::vector<Q8_1Block> &quantized_rows = iterator->second;
            quantized_rows.resize(static_cast<size_t>(m) * k_blocks);
            quantize_activations_to_q8_1(
                input.data(),
                quantized_rows.data(),
                m,
                k_,
                k_blocks);
            return quantized_rows;
        }

        /** Release fixtures that no later selected format can consume. */
        void releaseFormatNameLength(size_t format_name_length)
        {
            for (auto iterator = fixtures_.begin(); iterator != fixtures_.end();)
            {
                if (keyFormatNameLength(iterator->first) == format_name_length)
                    iterator = fixtures_.erase(iterator);
                else
                    ++iterator;
            }
        }

        /** Return the number of resident fixtures for focused regressions. */
        size_t fixtureCount() const { return fixtures_.size(); }

    private:
        static uint64_t fixtureKey(int m, size_t format_name_length)
        {
            if (format_name_length > std::numeric_limits<uint32_t>::max())
                throw std::invalid_argument("CPU prefill format label is too long");
            return (static_cast<uint64_t>(static_cast<uint32_t>(m)) << 32) |
                static_cast<uint32_t>(format_name_length);
        }

        static size_t keyFormatNameLength(uint64_t key)
        {
            return static_cast<size_t>(static_cast<uint32_t>(key));
        }

        int n_;
        int k_;
        std::unordered_map<uint64_t, std::vector<Q8_1Block>> fixtures_;
    };

    /**
     * @brief Time one CPU candidate with the backend-neutral trainer protocol.
     *
     * The raw samples remain available for a retained sidecar.  Policy code
     * consumes the upper median and dispersion reconstructed from those exact
     * native doubles rather than trusting a rounded aggregate CSV field.
     */
    template <typename Fn>
    StrongTimingMeasurement timeStrongTrainerCandidate(
        int warmup,
        int iterations,
        Fn &&fn)
    {
        for (int index = 0; index < warmup; ++index)
            fn();

        StrongTimingMeasurement result;
        result.samples_us.reserve(static_cast<size_t>(iterations));
        for (int index = 0; index < iterations; ++index)
        {
            const auto begin = std::chrono::steady_clock::now();
            fn();
            const auto end = std::chrono::steady_clock::now();
            result.samples_us.push_back(
                std::chrono::duration<double, std::micro>(end - begin).count());
        }
        result.acquisition_samples_us = result.samples_us;
        std::sort(result.samples_us.begin(), result.samples_us.end());
        result.evidence =
            llaminar2::test::trainer::summarizeSortedTimingSamples(
                result.samples_us);
        result.adaptive.measured_duration_us = std::accumulate(
            result.samples_us.begin(), result.samples_us.end(), 0.0);
        return result;
    }

    /**
     * @brief Time a CPU candidate until elapsed and stability evidence is sufficient.
     *
     * A production prefill matrix contains both tiny attention projections and
     * very expensive long-context FFN GEMMs. Fixed repetition counts spend most
     * of the sweep re-confirming expensive kernels after their timing has
     * stabilized. This protocol retains a minimum sample population, then
     * admits an early stop only after the measured kernel duration and
     * acquisition-order half-window stability thresholds are both satisfied.
     * Inexpensive or drifting kernels continue to the hard sample ceiling.
     */
    template <typename Fn>
    StrongTimingMeasurement timeStrongTrainerCandidateAdaptive(
        int warmup,
        int minimum_iterations,
        int maximum_iterations,
        double minimum_duration_us,
        double maximum_median_relative_drift,
        Fn &&fn)
    {
        for (int index = 0; index < warmup; ++index)
            fn();

        const int bounded_maximum = std::max(1, maximum_iterations);
        const int bounded_minimum = std::clamp(
            minimum_iterations, 1, bounded_maximum);
        std::vector<double> acquisition_samples;
        acquisition_samples.reserve(static_cast<size_t>(bounded_maximum));
        llaminar2::test::trainer::AdaptiveTimingDecision decision;
        for (int index = 0; index < bounded_maximum; ++index)
        {
            const auto begin = std::chrono::steady_clock::now();
            fn();
            const auto end = std::chrono::steady_clock::now();
            acquisition_samples.push_back(
                std::chrono::duration<double, std::micro>(end - begin).count());
            decision = llaminar2::test::trainer::evaluateAdaptiveTiming(
                acquisition_samples,
                static_cast<size_t>(bounded_minimum),
                static_cast<size_t>(bounded_maximum),
                minimum_duration_us,
                maximum_median_relative_drift);
            if (decision.should_stop)
                break;
        }

        StrongTimingMeasurement result;
        result.acquisition_samples_us = acquisition_samples;
        result.adaptive = decision;
        const size_t stationary_begin = std::min(
            decision.stationary_window_begin, acquisition_samples.size());
        result.samples_us.assign(
            acquisition_samples.begin() +
                static_cast<std::ptrdiff_t>(stationary_begin),
            acquisition_samples.end());
        result.stop_reason = decision.promotion_evidence
                                 ? "stationary_window"
                                 : "max_samples";
        std::sort(result.samples_us.begin(), result.samples_us.end());
        result.evidence =
            llaminar2::test::trainer::summarizeSortedTimingSamples(
                result.samples_us);
        return result;
    }

    /** Route identity observed from the real grouped verifier entry point. */
    struct CPUVerifierRouteEvidence
    {
        bool found = false;
        std::string requested_policy;
        std::string effective_policy;
        std::string build_isa;
        std::string isa;
        int k_tiles = 0;
        int n_block_chunks = 0;
        int threads = 0;
        uint64_t count = 0;
    };

    /** Find the exact candidate route recorded for one CPU verifier shape. */
    CPUVerifierRouteEvidence findCPUVerifierRoute(
        int M,
        int N,
        int K,
        uint8_t codebook)
    {
        CPUVerifierRouteEvidence result;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"kernel.cpu_native_vnni_verifier_rows_launch"}))
        {
            if (record.domain != "kernel" ||
                record.name != "cpu_native_vnni_verifier_rows_launch" ||
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
            if (tag("m") != std::to_string(M) ||
                tag("n") != std::to_string(N) ||
                tag("k") != std::to_string(K) ||
                tag("codebook") != std::to_string(codebook))
            {
                continue;
            }
            result.found = true;
            result.requested_policy = tag("requested_policy");
            result.effective_policy = tag("effective_policy");
            result.build_isa = tag("build_isa");
            result.isa = tag("isa");
            result.k_tiles = std::stoi(tag("k_tiles"));
            result.n_block_chunks = std::stoi(tag("n_block_chunks"));
            result.threads = std::stoi(tag("threads"));
            result.count += record.count;
        }
        return result;
    }

    /** Route identity observed from the real production M=1 entry point. */
    struct CPUDecodeRouteEvidence
    {
        bool found = false;
        std::string effective_policy;
        std::string build_isa;
        std::string isa;
        int k_tiles = 0;
        int n_block_chunks = 0;
        bool serial_kpart = false;
        int threads = 0;
        uint64_t count = 0;
    };

    /**
     * @brief Find the exact M=1 candidate route recorded for one projection.
     *
     * Trainers request a nominal NBC policy, but small N geometries can make
     * several nominal widths the same physical launch.  The production route
     * counter publishes the effective width, allowing the adapter to retain
     * normalized requests as unsupported evidence without profiling aliases.
     */
    CPUDecodeRouteEvidence findCPUDecodeRoute(
        int N,
        int K,
        uint8_t codebook)
    {
        CPUDecodeRouteEvidence result;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"kernel.cpu_native_vnni_decode_launch"}))
        {
            if (record.domain != "kernel" ||
                record.name != "cpu_native_vnni_decode_launch" ||
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
            if (tag("n") != std::to_string(N) ||
                tag("k") != std::to_string(K) ||
                tag("codebook") != std::to_string(codebook))
            {
                continue;
            }
            result.found = true;
            result.effective_policy = tag("effective_policy");
            result.build_isa = tag("build_isa");
            result.isa = tag("isa");
            result.k_tiles = std::stoi(tag("k_tiles"));
            result.n_block_chunks = std::stoi(tag("n_block_chunks"));
            result.serial_kpart = tag("serial_kpart") == "1";
            result.threads = std::stoi(tag("threads"));
            result.count += record.count;
        }
        return result;
    }

    /** Route identity observed from ordinary production M>1 NativeVNNI GEMM. */
    struct CPUPrefillRouteEvidence
    {
        bool found = false;
        std::string route;
        std::string requested_policy;
        std::string effective_policy;
        std::string build_isa;
        std::string isa;
        int k_tiles = 0;
        int k_tile_blocks = 0;
        int n_block_chunks = 0;
        bool pair_grid_execution_found = false;
        int pair_grid_execution_n_block_chunks = 0;
        int pair_grid_parallel_tasks = 0;
        uint64_t pair_grid_execution_count = 0;
        int threads = 0;
        uint64_t count = 0;

        /** @brief Render the registry ID for the physical route that ran. */
        std::string candidateId() const
        {
            if (route == "row_chunk_grid")
                return "cpu.nvnni.prefill.row_chunk_grid.full_k";
            if (route == "two_row_n_major")
            {
                return "cpu.nvnni.prefill.two_row_tiles.nbc" +
                       std::to_string(n_block_chunks) + ".full_k";
            }
            if (route == "two_row_pair_grid")
            {
                return "cpu.nvnni.prefill.two_row_pair_grid.nbc" +
                       std::to_string(n_block_chunks) + ".full_k";
            }
            if (route == "decode_equivalent_kpart_rows")
            {
                return effective_policy == "WideRows"
                           ? "cpu.nvnni.prefill.decode_equivalent_kpart.wide_rows"
                           : "cpu.nvnni.prefill.decode_equivalent_kpart.pairwise";
            }
            return {};
        }
    };

    /**
     * @brief Read the exact physical full-K prefill route for one trial cell.
     *
     * Candidate requests can normalize when a coarse N block leaves too few
     * output tasks and production chooses the row-chunk grid. The trainer must
     * retain that row as unsupported evidence under the requested candidate,
     * while naming the observed route separately; otherwise two nominal knobs
     * could be promoted as independent schedules despite executing identically.
     */
    CPUPrefillRouteEvidence findCPUPrefillRoute(
        int M,
        int N,
        int K,
        uint8_t codebook)
    {
        CPUPrefillRouteEvidence result;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"kernel.cpu_native_vnni_prefill_gemm_launch"}))
        {
            if (record.domain != "kernel" ||
                record.name != "cpu_native_vnni_prefill_gemm_launch" ||
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
            if (tag("m") != std::to_string(M) ||
                tag("n") != std::to_string(N) ||
                tag("k") != std::to_string(K) ||
                tag("codebook") != std::to_string(codebook))
            {
                continue;
            }
            result.found = true;
            result.route = tag("route");
            result.requested_policy = tag("requested_policy");
            result.effective_policy = tag("effective_policy");
            result.build_isa = tag("build_isa");
            result.isa = tag("isa");
            result.k_tiles = std::stoi(tag("k_tiles"));
            result.k_tile_blocks = std::stoi(tag("k_tile_blocks"));
            result.n_block_chunks = std::stoi(tag("n_block_chunks"));
            result.threads = std::stoi(tag("threads"));
            result.count += record.count;
        }
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"kernel.cpu_native_vnni_pair_grid_execution"}))
        {
            if (record.domain != "kernel" ||
                record.name != "cpu_native_vnni_pair_grid_execution" ||
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
            if (tag("m") != std::to_string(M) ||
                tag("n") != std::to_string(N) ||
                tag("k") != std::to_string(K) ||
                tag("codebook") != std::to_string(codebook))
            {
                continue;
            }
            result.pair_grid_execution_found = true;
            result.pair_grid_execution_n_block_chunks =
                std::stoi(tag("n_block_chunks"));
            result.pair_grid_parallel_tasks =
                std::stoi(tag("parallel_tasks"));
            result.pair_grid_execution_count += record.count;
        }
        return result;
    }

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
    /**
     * @brief Convert a ZMM INT32 accumulator into a cheap scalar checksum.
     *
     * The instruction-floor microbench intentionally avoids full GEMV metadata
     * work.  It still needs a visible result so the compiler cannot remove the
     * VPDPBUSD loops.  Summing one stored ZMM per accumulator gives us a stable
     * checksum while keeping the diagnostic focused on dot-product throughput.
     */
    int64_t checksumAccumulator(__m512i acc)
    {
        alignas(64) int32_t values[16];
        _mm512_store_si512(reinterpret_cast<__m512i *>(values), acc);
        int64_t sum = 0;
        for (int value : values)
            sum += value;
        return sum;
    }

    /**
     * @brief Minimal NativeVNNI row-scaling diagnostic for MTP verifier rows.
     *
     * This is not a production kernel.  It strips the verifier GEMV down to the
     * invariant inner operation: signed packed-B vectors reused across M
     * independent verifier rows via VPDPBUSD.  The result acts as a ceiling test
     * for the current packed layout.  If this diagnostic cannot approach the
     * desired M=4 speedup, a production rewrite needs a different B layout or a
     * different matrix ISA rather than only local loop polish.
     */
    struct VNNIFloorResult
    {
        int rows = 0;
        double grouped_us = 0.0;
        double serial_us = 0.0;
        double speedup = 0.0;
        bool checksums_match = false;
    };

    VNNIFloorResult runVerifierRowsVNNIFloor(int rows, int work_blocks, int warmup, int iterations)
    {
        if (rows < 1 || rows > 4 || work_blocks <= 0)
        {
            ADD_FAILURE()
                << "Invalid VNNI floor diagnostic shape: rows=" << rows
                << " work_blocks=" << work_blocks;
            return {};
        }

        alignas(64) std::array<std::array<uint8_t, 4>, 4> a_bytes{};
        alignas(64) std::array<std::array<int8_t, 64>, 8> b_bytes{};
        for (int row = 0; row < 4; ++row)
        {
            for (int group = 0; group < 4; ++group)
                a_bytes[row][group] = static_cast<uint8_t>(1 + row + group);
        }
        for (int group = 0; group < 8; ++group)
        {
            for (int lane = 0; lane < 64; ++lane)
                b_bytes[group][lane] = static_cast<int8_t>(((group + lane) % 5) - 2);
        }

        alignas(64) __m512i a_broadcasts[4];
        for (int row = 0; row < 4; ++row)
        {
            uint32_t packed = 0;
            std::memcpy(&packed, a_bytes[row].data(), sizeof(packed));
            a_broadcasts[row] = _mm512_set1_epi32(static_cast<int32_t>(packed));
        }

        alignas(64) __m512i b_vectors[8];
        for (int group = 0; group < 8; ++group)
        {
            b_vectors[group] =
                _mm512_load_si512(reinterpret_cast<const __m512i *>(b_bytes[group].data()));
        }

        auto run_serial = [&]() -> int64_t
        {
            int64_t checksum = 0;
            for (int row = 0; row < rows; ++row)
            {
                __m512i acc0 = _mm512_setzero_si512();
                __m512i acc1 = _mm512_setzero_si512();
                __m512i acc2 = _mm512_setzero_si512();
                __m512i acc3 = _mm512_setzero_si512();
                for (int block = 0; block < work_blocks; ++block)
                {
                    for (int group = 0; group < 8; ++group)
                    {
                        const __m512i b = b_vectors[(group + block) & 7];
                        const __m512i a = a_broadcasts[row];
                        acc0 = _mm512_dpbusd_epi32(acc0, a, b);
                        acc1 = _mm512_dpbusd_epi32(acc1, a, b);
                        acc2 = _mm512_dpbusd_epi32(acc2, a, b);
                        acc3 = _mm512_dpbusd_epi32(acc3, a, b);
                    }
                }
                checksum += checksumAccumulator(acc0);
                checksum += checksumAccumulator(acc1);
                checksum += checksumAccumulator(acc2);
                checksum += checksumAccumulator(acc3);
            }
            s_vnni_floor_sink += checksum;
            return checksum;
        };

        auto run_grouped = [&]() -> int64_t
        {
            alignas(64) __m512i acc0[4];
            alignas(64) __m512i acc1[4];
            alignas(64) __m512i acc2[4];
            alignas(64) __m512i acc3[4];
            for (int row = 0; row < 4; ++row)
            {
                acc0[row] = _mm512_setzero_si512();
                acc1[row] = _mm512_setzero_si512();
                acc2[row] = _mm512_setzero_si512();
                acc3[row] = _mm512_setzero_si512();
            }
            for (int block = 0; block < work_blocks; ++block)
            {
                for (int group = 0; group < 8; ++group)
                {
                    const __m512i b = b_vectors[(group + block) & 7];
                    for (int row = 0; row < rows; ++row)
                    {
                        const __m512i a = a_broadcasts[row];
                        acc0[row] = _mm512_dpbusd_epi32(acc0[row], a, b);
                        acc1[row] = _mm512_dpbusd_epi32(acc1[row], a, b);
                        acc2[row] = _mm512_dpbusd_epi32(acc2[row], a, b);
                        acc3[row] = _mm512_dpbusd_epi32(acc3[row], a, b);
                    }
                }
            }

            int64_t checksum = 0;
            for (int row = 0; row < rows; ++row)
            {
                checksum += checksumAccumulator(acc0[row]);
                checksum += checksumAccumulator(acc1[row]);
                checksum += checksumAccumulator(acc2[row]);
                checksum += checksumAccumulator(acc3[row]);
            }
            s_vnni_floor_sink += checksum;
            return checksum;
        };

        const int64_t serial_checksum = run_serial();
        const int64_t grouped_checksum = run_grouped();

        const TimingSummary grouped_timing =
            timeMicrobench(warmup, iterations, [&]()
                           { (void)run_grouped(); });
        const TimingSummary serial_timing =
            timeMicrobench(warmup, iterations, [&]()
                           { (void)run_serial(); });

        VNNIFloorResult result{};
        result.rows = rows;
        result.grouped_us = grouped_timing.min_us;
        result.serial_us = serial_timing.min_us;
        result.speedup =
            grouped_timing.min_us > 0.0 ? serial_timing.min_us / grouped_timing.min_us : 0.0;
        result.checksums_match = serial_checksum == grouped_checksum;
        return result;
    }
#endif

    ISAPath parseISAPathForVerifierBench()
    {
        const std::string raw = toLower(getEnvString("LLAMINAR_CPU_NVNNI_VERIFIER_ISA"));
        if (raw == "avx512")
            return ISAPath::AVX512;
        if (raw == "avx2")
            return ISAPath::AVX2;
        if (raw == "scalar")
            return ISAPath::SCALAR;
        return ISAPath::AUTO;
    }

    const char *isaPathName(ISAPath path)
    {
        switch (path)
        {
        case ISAPath::AVX512:
            return "AVX512";
        case ISAPath::AVX2:
            return "AVX2";
        case ISAPath::SCALAR:
            return "SCALAR";
        case ISAPath::AUTO:
        default:
            return "AUTO";
        }
    }

    /** Return the ISA selected by the production runtime dispatcher. */
    const char *activeISANameForVerifierTrainer()
    {
        switch (activeISALevel())
        {
        case ISALevel::AVX512:
            return "AVX512";
        case ISALevel::AVX2:
            return "AVX2";
        case ISALevel::Scalar:
        default:
            return "SCALAR";
        }
    }

    /**
     * @brief Normalize the ambient runtime-ISA request for evidence provenance.
     *
     * activeISALevel() is cached on first use, so a trainer must launch a
     * fresh process for every requested regime. Recording the request beside
     * the effective route catches a mislabeled AVX512-build/AVX2-runtime shard
     * before it reaches the policy compiler.
     */
    std::string requestedISANameForVerifierTrainer()
    {
        const std::string raw = toLower(getEnvString("LLAMINAR_ISA_LEVEL"));
        if (raw.empty())
            return "AUTO";
        if (raw == "avx512")
            return "AVX512";
        if (raw == "avx2")
            return "AVX2";
        if (raw == "scalar")
            return "SCALAR";
        return "INVALID:" + raw;
    }

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
    bool verifierBenchUsesAVX512(ISAPath path)
    {
        const ISALevel active_isa = activeISALevel();
        return (path == ISAPath::AUTO && active_isa >= ISALevel::AVX512) ||
               path == ISAPath::AVX512;
    }

#endif

    std::unique_ptr<TensorBase> createWeightsForFormat(
        const std::string &fmt_name, size_t N, size_t K)
    {
        if (fmt_name == "Q4_0")
            return TestTensorFactory::createQ4_0Random({N, K});
        if (fmt_name == "Q4_1")
            return TestTensorFactory::createQ4_1Random({N, K});
        if (fmt_name == "Q5_0")
            return TestTensorFactory::createQ5_0Random({N, K});
        if (fmt_name == "Q5_1")
            return TestTensorFactory::createQ5_1Random({N, K});
        if (fmt_name == "Q8_0")
            return TestTensorFactory::createQ8_0Random({N, K});
        if (fmt_name == "Q8_1")
            return TestTensorFactory::createQ8_1Random({N, K});
        if (fmt_name == "Q8_K")
            return TestTensorFactory::createQ8_KRandom({N, K});
        if (fmt_name == "Q2_K")
            return TestTensorFactory::createQ2_KRandom({N, K});
        if (fmt_name == "Q3_K")
            return TestTensorFactory::createQ3_KRandom({N, K});
        if (fmt_name == "Q4_K")
            return TestTensorFactory::createQ4_KRandom({N, K});
        if (fmt_name == "Q5_K")
            return TestTensorFactory::createQ5_KRandom({N, K});
        if (fmt_name == "Q6_K")
            return TestTensorFactory::createQ6_KRandom({N, K});
        if (fmt_name == "IQ4_NL")
            return TestTensorFactory::createIQ4_NLRandom({N, K});
        if (fmt_name == "IQ4_XS")
            return TestTensorFactory::createIQ4_XSRandom({N, K});
        if (fmt_name == "IQ3_S")
            return TestTensorFactory::createIQ3_SRandom({N, K});
        if (fmt_name == "IQ3_XXS")
            return TestTensorFactory::createIQ3_XXSRandom({N, K});
        if (fmt_name == "IQ2_S")
            return TestTensorFactory::createIQ2_SRandom({N, K});
        if (fmt_name == "IQ2_XS")
            return TestTensorFactory::createIQ2_XSRandom({N, K});
        if (fmt_name == "IQ2_XXS")
            return TestTensorFactory::createIQ2_XXSRandom({N, K});
        if (fmt_name == "IQ1_S")
            return TestTensorFactory::createIQ1_SRandom({N, K});
        if (fmt_name == "IQ1_M")
            return TestTensorFactory::createIQ1_MRandom({N, K});
        return nullptr;
    }

    /**
     * @brief Create the exact eager prepared-weight layout needed by one
     *        isolated CPU profiler launch without regenerating source weights.
     *
     * Canonical trainer observations construct a real quantized tensor, run
     * the production packer, prove byte equality, and measure latency. The
     * profiler transaction augments that immutable observation with hardware
     * counters for exactly one already-identified prepared kernel candidate.
     * Recreating a multi-gigabyte source tensor and decoding it into the same
     * eager layout before every isolated counter launch adds no profiler
     * signal; it only makes fixture setup dominate corpus collection.
     *
     * This helper obtains intrinsic codebook metadata from a tiny real tensor,
     * then constructs the same aligned interleaved dimensions and metadata
     * offsets as `packWeightsCPUNativeVNNI()`. Every byte is first-touched by
     * the active OpenMP team so NUMA placement matches the profiled process.
     * Data bytes are deterministic and non-zero, while scales, compensation,
     * and asymmetric minima are finite. NativeVNNI kernels have no data-
     * dependent control flow, so the instruction stream and memory geometry
     * are identical to a normally packed invocation.
     *
     * @warning This is profiler-fixture construction only. Correctness,
     *          canonical timing, dispatch installation, and production
     *          inference must always use the real source-format packer.
     *
     * @param format Source tensor format whose intrinsic metadata is required.
     * @param N Output-row count of the exact profiled geometry.
     * @param K Reduction width of the exact profiled geometry.
     * @return Fully owned eager prepared weights accepted by the production
     *         CPU NativeVNNI GEMV/GEMM entry points.
     */
    CPUNativeVNNIPackedWeights createProfilerPackedWeightsForFormat(
        const FormatSpec &format,
        int N,
        int K)
    {
        if (N <= 0 || K <= 0)
            throw std::invalid_argument(
                "CPU profiler packed-weight geometry must be positive");

        // A 256-element probe satisfies both ordinary 32-element blocks and
        // every superblock format while keeping metadata discovery negligible.
        constexpr size_t PROBE_N = 1;
        constexpr size_t PROBE_K = 256;
        auto probe = createWeightsForFormat(format.name, PROBE_N, PROBE_K);
        if (!probe)
            throw std::runtime_error(
                "CPU profiler cannot create format probe for " + format.name);
        const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(
            probe.get());
        const NativeVnniFormatInfo *metadata =
            unpackable ? unpackable->vnniFormatInfo() : nullptr;
        if (!metadata)
            throw std::runtime_error(
                "CPU profiler format has no NativeVNNI metadata: " +
                format.name);

        CPUNativeVNNIPackedWeights packed;
        packed.N = N;
        packed.K = K;
        packed.N_padded = ((N + 63) / 64) * 64;
        packed.blocks_per_row = (K + 31) / 32;
        packed.payload_bytes = metadata->payload_bytes;
        packed.codebook_id = metadata->codebook_id;
        packed.is_asymmetric = metadata->is_asymmetric;
        packed.is_superblock = metadata->is_superblock;
        packed.is_nibble_lut = is_nibble_lut_format(metadata->codebook_id);
        packed.data_stride = packed.is_nibble_lut ? 1024 : 2048;
        packed.interleaved_block_stride =
            packed.data_stride + 256 +
            (packed.is_asymmetric ? 128 : 0);

        const size_t n_chunks =
            static_cast<size_t>(packed.N_padded / 64);
        const size_t interleaved_bytes =
            n_chunks * static_cast<size_t>(packed.blocks_per_row) *
            static_cast<size_t>(packed.interleaved_block_stride);
        packed.native_interleaved.resize_uninitialized(interleaved_bytes);

        const uint16_t one_fp16 = fp32_to_fp16(1.0f);
        const uint16_t min_fp16 = fp32_to_fp16(0.125f);
        uint8_t *const bytes = packed.native_interleaved.data();
#pragma omp parallel
        {
            // First-touch every page in parallel. The index hash supplies
            // repeatable non-zero data without a serial random-number stream.
#pragma omp for schedule(static)
            for (int64_t byte_index = 0;
                 byte_index < static_cast<int64_t>(interleaved_bytes);
                 ++byte_index)
            {
                const uint64_t mixed =
                    static_cast<uint64_t>(byte_index) * 0x9E3779B185EBCA87ULL +
                    static_cast<uint64_t>(packed.codebook_id) *
                        0xC2B2AE3D27D4EB4FULL;
                bytes[static_cast<size_t>(byte_index)] =
                    static_cast<uint8_t>((mixed >> 24) | 1U);
            }

            // Overwrite inline metadata with finite values at exactly the
            // offsets consumed by the production kernels. Compensation is a
            // diagnostic value here: canonical numerical evidence comes from
            // the real packed tensor, outside isolated profiling.
#pragma omp for schedule(static)
            for (int64_t block_index = 0;
                 block_index < static_cast<int64_t>(
                     n_chunks *
                     static_cast<size_t>(packed.blocks_per_row));
                 ++block_index)
            {
                uint8_t *const block =
                    bytes + static_cast<size_t>(block_index) *
                                packed.interleaved_block_stride;
                auto *const compensation = reinterpret_cast<int16_t *>(
                    block + packed.data_stride);
                auto *const scales = reinterpret_cast<uint16_t *>(
                    block + packed.data_stride + 128);
                auto *const minima = packed.is_asymmetric
                    ? reinterpret_cast<uint16_t *>(
                          block + packed.data_stride + 256)
                    : nullptr;
                for (int column = 0; column < 64; ++column)
                {
                    compensation[column] = 0;
                    scales[column] = one_fp16;
                    if (minima)
                        minima[column] = min_fp16;
                }
            }
        }

        return packed;
    }

    /**
     * @brief Select the prepared-weight fixture for an isolated CPU profiler run.
     *
     * Canonical profiler collection always uses the synthetic prepared fixture:
     * it preserves the production launch geometry while avoiding source-format
     * generation and packing outside the controlled counter interval.  The
     * real-packed mode exists only for a bounded A/B diagnostic that proves the
     * synthetic bytes do not change the target kernel's dynamic instruction or
     * cache behavior.  It must never be enabled by a corpus collection command.
     *
     * @return `true` for the canonical synthetic fixture, or `false` only when
     *         `LLAMINAR_CPU_NVNNI_PROFILER_REAL_PACKED_WEIGHTS=1` is explicitly
     *         requested by the focused profiler-fixture A/B.
     */
    bool useSyntheticProfilerPreparedWeights()
    {
        return getEnvInt(
                   "LLAMINAR_CPU_NVNNI_PROFILER_REAL_PACKED_WEIGHTS")
                   .value_or(0) == 0;
    }

    // =========================================================================
    // Shape definitions
    // =========================================================================
    struct GEMVShape
    {
        std::string name;
        std::string category; // "Attn", "FFN", "LM_Head"
        std::string shard;    // "ColPar", "RowPar", "Replicate"
        int N;
        int K;
    };

    static std::vector<GEMVShape> buildShapes(const ModelConfig &m, int tp = 1)
    {
        int n_q = m.n_heads * m.head_dim;
        int n_kv = m.n_kv_heads * m.head_dim;
        std::string suffix = (tp > 1) ? "_TP" + std::to_string(tp) : "";

        // Megatron-style TP sharding:
        //   Column-parallel (QKV, FFN Gate/Up): split N (output)
        //   Row-parallel (Wo, FFN Down):        split K (input)
        //   LM_Head: Column-parallel (split vocab)
        return {
            {m.name + "_Q_proj" + suffix, "Attn", "ColPar", n_q / tp, m.d_model},
            {m.name + "_K_proj" + suffix, "Attn", "ColPar", n_kv / tp, m.d_model},
            {m.name + "_V_proj" + suffix, "Attn", "ColPar", n_kv / tp, m.d_model},
            {m.name + "_Wo_proj" + suffix, "Attn", "RowPar", m.d_model, n_q / tp},
            {m.name + "_FFN_Gate" + suffix, "FFN", "ColPar", m.d_ff / tp, m.d_model},
            {m.name + "_FFN_Up" + suffix, "FFN", "ColPar", m.d_ff / tp, m.d_model},
            {m.name + "_FFN_Down" + suffix, "FFN", "RowPar", m.d_model, m.d_ff / tp},
            {m.name + "_LM_Head" + suffix, "LM_Head", "ColPar", m.vocab / tp, m.d_model},
        };
    }

    // =========================================================================
    // Benchmark infrastructure
    // =========================================================================
    static constexpr int WARMUP = 30;
    static constexpr int ITERS = 100;

    struct GEMVResult
    {
        std::string shape_name;
        std::string category;
        std::string shard;
        int N, K;
        // Tile config
        int nbc;              // n_block_chunks
        int k_tiles;          // k-parallel tiles (0=N-parallel only)
        std::string tile_cat; // ShapeCategory name
        // Packed VNNI GEMV (current production: quantize + packed access)
        double packed_us;
        double packed_bw_gbs;
        // Roofline comparison
        double roofline_bw;
        double packed_roof_pct;
        // Weight data sizes
        double packed_bytes; // VNNI packed buffer
    };

    // =========================================================================
    // Cache flush utility — evict a memory range from all cache levels
    // =========================================================================

    /**
     * @brief Flush a memory range from all CPU caches using clflushopt.
     *
     * Walks through the range in 64-byte (cache line) steps and issues
     * clflushopt for each line, then an mfence to ensure completion.
     * This forces the next access to go to DRAM, eliminating cache effects
     * from benchmarks.
     */
    static inline void flush_cache_range(const void *ptr, size_t bytes)
    {
        char *p = const_cast<char *>(static_cast<const char *>(ptr));
        for (size_t off = 0; off < bytes; off += 64)
#if defined(__CLFLUSHOPT__)
            _mm_clflushopt(p + off);
#else
            _mm_clflush(p + off);
#endif
        _mm_mfence();
    }

    /**
     * @brief Benchmark packed VNNI GEMV directly, return p10 latency (us).
     *
     * Calls gemv_native_vnni() directly to measure the packed VNNI path
     * (FP32→Q8_1 quantization + INT8 packed VNNI compute). This bypasses
     * multiply_tensor's Q8_0 native shortcut, giving a real VNNI-vs-native
     * comparison for Q8_0 weights.
     *
     * Flushes the packed weight data from cache before each timed iteration
     * to measure true DRAM bandwidth efficiency without cache effects.
     */
    double benchPackedVNNI(const CPUNativeVNNIPackedWeights &packed,
                           const float *A, float *C, int N)
    {
        // Identify the weight data buffer and size for cache flushing
        const void *weight_data = packed.native_interleaved.data();
        size_t weight_bytes = packed.native_interleaved.size();

        for (int i = 0; i < WARMUP; ++i)
            gemv_native_vnni(packed, A, C);

        std::vector<double> times(ITERS);
        for (int i = 0; i < ITERS; ++i)
        {
            flush_cache_range(weight_data, weight_bytes);
            auto t0 = std::chrono::high_resolution_clock::now();
            gemv_native_vnni(packed, A, C);
            auto t1 = std::chrono::high_resolution_clock::now();
            times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        std::sort(times.begin(), times.end());
        return times[std::max(0, (int)(ITERS * 0.1) - 1)];
    }

    static const char *categoryName(ShapeCategory cat)
    {
        switch (cat)
        {
        case ShapeCategory::ATTENTION:
            return "ATTN";
        case ShapeCategory::FFN:
            return "FFN";
        case ShapeCategory::LM_HEAD:
            return "LM";
        default:
            return "GEN";
        }
    }

    /**
     * @brief Benchmark one shape and return result.
     */
    GEMVResult benchShape(const GEMVShape &shape, double roofline_bw)
    {
        GEMVResult r{};
        r.shape_name = shape.name;
        r.category = shape.category;
        r.shard = shape.shard;
        r.N = shape.N;
        r.K = shape.K;
        r.roofline_bw = roofline_bw;

        // Create Q8_0 weights
        auto weights = TestTensorFactory::createQ8_0Random(
            {(size_t)shape.N, (size_t)shape.K});
        if (!weights)
            return r;

        auto *q8_tensor = dynamic_cast<Q8_0Tensor *>(weights.get());
        if (!q8_tensor)
            return r;

        // Random activations
        int bpr = (shape.K + 31) / 32;
        std::vector<float> A(shape.K);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : A)
            v = dist(rng);

        // --- Tile config ---
        int payload = 32; // Q8_0 payload bytes per block
        int threads = omp_get_max_threads();
        auto tile = computeTileConfig(shape.N, shape.K, 1, payload, threads);
        r.nbc = tile.n_block_chunks;
        r.k_tiles = tile.k_tiles;
        r.tile_cat = categoryName(tile.category);

        // --- Weight data sizes ---
        int N_chunks = (shape.N + 63) / 64;
        // Packed VNNI: interleaved blocks + scales + comps
        // INT8 pre-decoded: stride=2048 per chunk per K-block
        r.packed_bytes = (double)N_chunks * bpr * (2048 + 256 + 256); // interleaved + scales + comp
        // --- Packed VNNI GEMV (quantize FP32→Q8_1 + INT8 packed VNNI) ---
        CPUNativeVNNIGemmKernel packed_kernel(weights.get());
        if (packed_kernel.isValid())
        {
            std::vector<float> C_packed(shape.N, 0.0f);
            r.packed_us = benchPackedVNNI(
                packed_kernel.packedWeights(), A.data(), C_packed.data(), shape.N);
            r.packed_bw_gbs = r.packed_bytes / (r.packed_us * 1e-6) / 1e9;
            r.packed_roof_pct = r.packed_bw_gbs / roofline_bw * 100.0;
        }

        return r;
    }

    // =========================================================================
    // Table renderers
    // =========================================================================

    void renderGEMVTable(const std::string &title,
                         const std::vector<GEMVResult> &results)
    {
        double roofline = results.empty() ? 0 : results[0].roofline_bw;

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Shape" << "Cat" << "Shard" << "N" << "K"
              << "Tile" << "nbc" << "kt"
              << "Q8_0 \xc2\xb5s" << "Q8_0 BW" << "Q8_0 R%"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::left);
        table.column(2).set_cell_text_align(fort::text_align::left);
        table.column(5).set_cell_text_align(fort::text_align::left);
        for (int c : {3, 4, 6, 7, 8, 9, 10})
            table.column(c).set_cell_text_align(fort::text_align::right);

        double sum_packed = 0;
        double sum_packed_bytes = 0;
        std::string last_cat;

        for (const auto &r : results)
        {
            if (!last_cat.empty() && r.category != last_cat)
                table << fort::separator;
            last_cat = r.category;

            char pu[16], pbw[16], pr[16];
            std::snprintf(pu, sizeof(pu), "%.0f", r.packed_us);
            std::snprintf(pbw, sizeof(pbw), "%.1f", r.packed_bw_gbs);
            std::snprintf(pr, sizeof(pr), "%.0f%%", r.packed_roof_pct);

            table << r.shape_name << r.category << r.shard
                  << r.N << r.K
                  << r.tile_cat << r.nbc << r.k_tiles
                  << pu << pbw << pr
                  << fort::endr;

            sum_packed += r.packed_us;
            sum_packed_bytes += r.packed_bytes;
        }

        // Summary row
        table << fort::separator;
        double avg_packed_bw = sum_packed_bytes / (sum_packed * 1e-6) / 1e9;
        char tpu[16], tpbw[16], tpr[16];
        std::snprintf(tpu, sizeof(tpu), "%.0f", sum_packed);
        std::snprintf(tpbw, sizeof(tpbw), "%.1f", avg_packed_bw);
        std::snprintf(tpr, sizeof(tpr), "%.0f%%", avg_packed_bw / roofline * 100);

        table << "TOTAL" << "" << "" << "" << "" << "" << "" << ""
              << tpu << tpbw << tpr
              << fort::endr;

        std::cout << "\n"
                  << title << "\n"
                  << "Threads: " << omp_get_max_threads()
                  << "  |  Roofline: " << std::fixed << std::setprecision(1) << roofline << " GB/s (calibrated)"
                  << "  |  Cold cache (clflushopt before each iteration)"
                  << "  |  Q8_0 = eager interleaved NativeVNNI\n\n"
                  << table.to_string() << std::endl;
    }

    /**
     * @brief Render per-model decode latency summary (simulated full-layer token time).
     */
    void renderDecodeProjection(const std::string &title,
                                const std::vector<std::pair<std::string, std::vector<GEMVResult>>> &model_results)
    {
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Model" << "Layers"
              << "Q8_0 ms/tok" << "Q8_0 tok/s"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 3; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        // Approximate layer counts
        auto layerCount = [](const std::string &name) -> int
        {
            if (name == "0.5B")
                return 24;
            if (name == "1.5B")
                return 28;
            if (name == "3B")
                return 36;
            if (name == "7B")
                return 28;
            if (name == "14B")
                return 48;
            if (name == "32B")
                return 64;
            return 28;
        };

        for (auto &[model_name, results] : model_results)
        {
            int layers = layerCount(model_name);
            // Sum per-layer GEMV time (all shapes except LM_Head)
            double layer_packed_us = 0;
            double lm_packed_us = 0;

            for (const auto &r : results)
            {
                if (r.category == "LM_Head")
                {
                    lm_packed_us += r.packed_us;
                }
                else
                {
                    layer_packed_us += r.packed_us;
                }
            }

            double packed_ms = (layer_packed_us * layers + lm_packed_us) / 1000.0;

            char pm[16], pt[16];
            std::snprintf(pm, sizeof(pm), "%.1f", packed_ms);
            std::snprintf(pt, sizeof(pt), "%.1f", 1000.0 / packed_ms);

            table << model_name << layers
                  << pm << pt
                  << fort::endr;
        }

        std::cout << "\n"
                  << title << "\n"
                  << "Projection: (per-layer GEMV total × layers) + LM_Head\n"
                  << "GEMV-only time (excludes attention, norms, sampling)\n\n"
                  << table.to_string() << std::endl;
    }

    /**
     * @brief Render TP scaling comparison table.
     */
    void renderTPScaling(const std::string &title,
                         const std::vector<GEMVResult> &tp1,
                         const std::vector<GEMVResult> &tp2,
                         const std::vector<GEMVResult> &tp4)
    {
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Layer" << "Shard"
              << "TP1 N\xc3\x97K" << "Q8_0 \xc2\xb5s"
              << "TP2 N\xc3\x97K" << "Q8_0 \xc2\xb5s" << "TP Eff"
              << "TP4 N\xc3\x97K" << "Q8_0 \xc2\xb5s" << "TP Eff"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::left);
        for (int c : {2, 3, 4, 5, 6, 7, 8, 9})
            table.column(c).set_cell_text_align(fort::text_align::right);

        size_t n = std::min({tp1.size(), tp2.size(), tp4.size()});
        for (size_t i = 0; i < n; ++i)
        {
            double eff2 = (tp1[i].packed_us / 2.0) / tp2[i].packed_us;
            double eff4 = (tp1[i].packed_us / 4.0) / tp4[i].packed_us;

            char nk1[24], u1[16], nk2[24], u2[16], e2[16], nk4[24], u4[16], e4[16];
            std::snprintf(nk1, sizeof(nk1), "%d\xc3\x97%d", tp1[i].N, tp1[i].K);
            std::snprintf(u1, sizeof(u1), "%.0f", tp1[i].packed_us);
            std::snprintf(nk2, sizeof(nk2), "%d\xc3\x97%d", tp2[i].N, tp2[i].K);
            std::snprintf(u2, sizeof(u2), "%.0f", tp2[i].packed_us);
            std::snprintf(e2, sizeof(e2), "%.0f%%", eff2 * 100);
            std::snprintf(nk4, sizeof(nk4), "%d\xc3\x97%d", tp4[i].N, tp4[i].K);
            std::snprintf(u4, sizeof(u4), "%.0f", tp4[i].packed_us);
            std::snprintf(e4, sizeof(e4), "%.0f%%", eff4 * 100);

            // Extract short layer name from shape name (after model prefix)
            std::string layer = tp1[i].shape_name;
            auto pos = layer.find('_');
            if (pos != std::string::npos)
                layer = layer.substr(pos + 1);

            table << layer << tp1[i].shard
                  << nk1 << u1
                  << nk2 << u2 << e2
                  << nk4 << u4 << e4
                  << fort::endr;
        }

        std::cout << "\n"
                  << title << "\n"
                  << "TP Eff = (TP1_time / TP_degree) / actual_time\n"
                  << "Eager-interleaved Q8_0 NativeVNNI GEMV\n\n"
                  << table.to_string() << std::endl;
    }

    // =========================================================================
    // Test fixture
    // =========================================================================
    class CPUNativeVNNIGemvTest : public ::testing::Test
    {
    };

    /**
     * @test Prove grouped-candidate admission cannot sample away an output row.
     *
     * Ordinary long-prefill timing intentionally checks a bounded witness set,
     * but a schedule being considered for grouped MTP must compare every row
     * against serial M1.  This regression keeps those two modes visibly
     * distinct and verifies the exhaustive mode's complete natural ordering.
     */
    TEST_F(
        CPUNativeVNNIGemvTest,
        PrefillSerialOracleRows_ExhaustiveModeCoversEveryRow)
    {
        constexpr int kM = 31;
        const std::vector<int> sampled = prefillSerialOracleRows(kM);
        const std::vector<int> exhaustive =
            prefillSerialOracleRows(kM, /*require_exhaustive_oracle=*/true);

        ASSERT_LT(sampled.size(), static_cast<size_t>(kM));
        ASSERT_EQ(exhaustive.size(), static_cast<size_t>(kM));
        for (int row = 0; row < kM; ++row)
            EXPECT_EQ(exhaustive[static_cast<size_t>(row)], row);
    }

    /**
     * @test Prove profiler-only prepared fixtures reproduce every production
     *       source format's eager memory geometry.
     *
     * Isolated hardware-counter launches may skip expensive source generation
     * only if the resulting production kernel observes the same codebook,
     * dimensions, strides, metadata offsets, alignment, and total byte extent.
     * This all-format regression compares a small real pack against the
     * synthetic profiler fixture and also verifies that every metadata lane is
     * initialized to a finite value.
     */
    TEST_F(
        CPUNativeVNNIGemvTest,
        ProfilerPreparedWeights_MatchProductionLayout_AllFormats)
    {
        constexpr int N = 65;
        constexpr int K = 256;
        const uint16_t expected_scale = fp32_to_fp16(1.0f);
        const uint16_t expected_minimum = fp32_to_fp16(0.125f);

        for (const FormatSpec &format : MTP_SMALL_M_FORMATS)
        {
            SCOPED_TRACE(format.name);
            auto source = createWeightsForFormat(format.name, N, K);
            ASSERT_NE(source, nullptr);
            CPUNativeVNNIGemmKernel production_kernel(source.get());
            ASSERT_TRUE(production_kernel.isValid());
            const CPUNativeVNNIPackedWeights &production =
                production_kernel.packedWeights();

            CPUNativeVNNIPackedWeights profiler =
                createProfilerPackedWeightsForFormat(format, N, K);
            EXPECT_EQ(profiler.N, production.N);
            EXPECT_EQ(profiler.K, production.K);
            EXPECT_EQ(profiler.N_padded, production.N_padded);
            EXPECT_EQ(profiler.blocks_per_row, production.blocks_per_row);
            EXPECT_EQ(profiler.payload_bytes, production.payload_bytes);
            EXPECT_EQ(profiler.codebook_id, production.codebook_id);
            EXPECT_EQ(profiler.is_asymmetric, production.is_asymmetric);
            EXPECT_EQ(profiler.is_superblock, production.is_superblock);
            EXPECT_EQ(profiler.is_nibble_lut, production.is_nibble_lut);
            EXPECT_EQ(profiler.data_stride, production.data_stride);
            EXPECT_EQ(
                profiler.interleaved_block_stride,
                production.interleaved_block_stride);
            EXPECT_EQ(
                profiler.native_interleaved.size(),
                production.native_interleaved.size());
            EXPECT_TRUE(profiler.hasInterleavedData());
            EXPECT_EQ(
                reinterpret_cast<uintptr_t>(
                    profiler.native_interleaved.data()) % 64,
                0u);

            const int n_chunks = profiler.N_padded / 64;
            for (int chunk = 0; chunk < n_chunks; ++chunk)
            {
                for (int block = 0;
                     block < profiler.blocks_per_row;
                     ++block)
                {
                    const int16_t *compensation =
                        profiler.chunkComp(chunk, block);
                    const uint16_t *scales =
                        profiler.chunkScales(chunk, block);
                    const uint16_t *minima = profiler.is_asymmetric
                        ? profiler.chunkMins(chunk, block)
                        : nullptr;
                    for (int column = 0; column < 64; ++column)
                    {
                        EXPECT_EQ(compensation[column], 0);
                        EXPECT_EQ(scales[column], expected_scale);
                        if (minima)
                            EXPECT_EQ(minima[column], expected_minimum);
                    }
                }
            }

            CPUNativeVNNIGemmKernel profiler_kernel(std::move(profiler));
            EXPECT_TRUE(profiler_kernel.isValid());
        }
    }

    /**
     * @test Prove coalesced activation fixtures preserve legacy input bytes.
     *
     * This is intentionally a tiny CPU-only regression. The production
     * trainer still performs full serial-row parity for every measured format;
     * this test isolates the lifecycle optimization and proves that two source
     * labels from the same historical seed class reuse one quantized fixture.
     */
    TEST_F(
        CPUNativeVNNIGemvTest,
        PrefillActivationFixtureCache_PreservesLegacyFormatSeed)
    {
        constexpr int N = 64;
        constexpr int K = 64;
        constexpr int M = 3;
        constexpr int K_BLOCKS = K / Q8_1Block::BLOCK_SIZE;
        const auto legacy_fixture = [](std::string_view source_format)
        {
            std::mt19937 rng(static_cast<uint32_t>(
                0xB47Cu + M * 131u + N + K + source_format.size()));
            std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
            std::vector<float> input(static_cast<size_t>(M) * K);
            for (float &value : input)
                value = distribution(rng);
            std::vector<Q8_1Block> result(
                static_cast<size_t>(M) * K_BLOCKS);
            quantize_activations_to_q8_1(
                input.data(), result.data(), M, K, K_BLOCKS);
            return result;
        };

        PrefillActivationFixtureCache cache(N, K);
        const std::vector<Q8_1Block> expected = legacy_fixture("Q4_0");
        const auto &first = cache.fixtureFor(M, "Q4_0", K_BLOCKS);
        ASSERT_EQ(first.size(), expected.size());
        EXPECT_EQ(
            std::memcmp(
                first.data(),
                expected.data(),
                expected.size() * sizeof(Q8_1Block)),
            0);
        const auto &same_seed_class = cache.fixtureFor(M, "Q5_0", K_BLOCKS);
        EXPECT_EQ(&first, &same_seed_class);
        EXPECT_EQ(cache.fixtureCount(), 1u);

        (void)cache.fixtureFor(M, "IQ3_S", K_BLOCKS);
        EXPECT_EQ(cache.fixtureCount(), 2u);
        cache.releaseFormatNameLength(std::string_view("Q4_0").size());
        EXPECT_EQ(cache.fixtureCount(), 1u);
    }

    // =========================================================================
    // TEST 1: Q8_0 All Models — Full shape sweep (TP=1)
    //
    // Comprehensive decode GEMV benchmark across all Qwen model sizes.
    // Measures the single production eager-interleaved path for each shape.
    // =========================================================================
    TEST_F(CPUNativeVNNIGemvTest, Q8_0_AllModels)
    {
        double roofline = systemBandwidthGB();
        std::vector<std::pair<std::string, std::vector<GEMVResult>>> all_models;

        for (const auto &model : ALL_MODELS)
        {
            auto shapes = buildShapes(model);
            std::vector<GEMVResult> results;
            for (const auto &shape : shapes)
                results.push_back(benchShape(shape, roofline));

            renderGEMVTable(
                "=== Q8_0 GEMV Decode: Qwen " + model.name + " (TP=1) ===",
                results);
            all_models.push_back({model.name, results});
        }

        renderDecodeProjection(
            "=== Decode Latency Projection — GEMV Only (Q8_0) ===",
            all_models);
    }

    // =========================================================================
    // TEST 2: Q8_0 7B TP Scaling — TP=1/2/4
    //
    // Primary benchmark for the user's current model.
    // =========================================================================
    TEST_F(CPUNativeVNNIGemvTest, Q8_0_7B_TP_Scaling)
    {
        double roofline = systemBandwidthGB();
        const auto &model = ALL_MODELS[3]; // 7B

        auto shapes_tp1 = buildShapes(model, 1);
        auto shapes_tp2 = buildShapes(model, 2);
        auto shapes_tp4 = buildShapes(model, 4);

        std::vector<GEMVResult> res1, res2, res4;
        for (const auto &s : shapes_tp1)
            res1.push_back(benchShape(s, roofline));
        for (const auto &s : shapes_tp2)
            res2.push_back(benchShape(s, roofline));
        for (const auto &s : shapes_tp4)
            res4.push_back(benchShape(s, roofline));

        renderGEMVTable("=== Q8_0 GEMV Decode: Qwen 7B — TP=1 ===", res1);
        renderGEMVTable("=== Q8_0 GEMV Decode: Qwen 7B — TP=2 ===", res2);
        renderGEMVTable("=== Q8_0 GEMV Decode: Qwen 7B — TP=4 ===", res4);

        renderTPScaling(
            "=== Q8_0 GEMV TP Scaling: Qwen 7B (TP=1/2/4) ===",
            res1, res2, res4);
    }

    // =========================================================================
    // TEST 3: All Models TP Scaling — TP=1/2/4
    //
    // Comprehensive TP scaling across all model sizes.
    // =========================================================================
    TEST_F(CPUNativeVNNIGemvTest, Q8_0_AllModels_TP_Scaling)
    {
        double roofline = systemBandwidthGB();

        for (const auto &model : ALL_MODELS)
        {
            // Skip TP shapes that would produce N<32 or K<32
            // (0.5B K_proj has n_kv=2, head=64 → KV_N=128; TP=4 → 32)
            int min_kv = model.n_kv_heads * model.head_dim;
            bool skip_tp4 = (min_kv / 4 < 32);

            auto s1 = buildShapes(model, 1);
            auto s2 = buildShapes(model, 2);

            std::vector<GEMVResult> r1, r2;
            for (const auto &s : s1)
                r1.push_back(benchShape(s, roofline));
            for (const auto &s : s2)
                r2.push_back(benchShape(s, roofline));

            if (!skip_tp4)
            {
                auto s4 = buildShapes(model, 4);
                std::vector<GEMVResult> r4;
                for (const auto &s : s4)
                    r4.push_back(benchShape(s, roofline));

                renderTPScaling(
                    "=== Q8_0 GEMV TP Scaling: Qwen " + model.name + " ===",
                    r1, r2, r4);
            }
            else
            {
                // TP=2 only
                renderGEMVTable("=== Q8_0 GEMV: Qwen " + model.name + " TP=1 ===", r1);
                renderGEMVTable("=== Q8_0 GEMV: Qwen " + model.name + " TP=2 ===", r2);
            }
        }
    }

    // =========================================================================
    // TEST 4: Tile Config Inspection — All shapes, all models
    //
    // No benchmarking — just shows what tile config the heuristic picks
    // for each shape at each TP degree. Quick diagnostic test.
    // =========================================================================
    TEST_F(CPUNativeVNNIGemvTest, TileConfig_AllShapes)
    {
        int threads = omp_get_max_threads();
        int payload = 32; // Q8_0

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Model" << "Shape" << "TP" << "N" << "K"
              << "Category" << "nbc" << "k_tiles" << "N_chunks" << "Tasks"
              << "Wt MB"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::left);
        table.column(5).set_cell_text_align(fort::text_align::left);
        for (int c : {2, 3, 4, 6, 7, 8, 9, 10})
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &model : ALL_MODELS)
        {
            for (int tp : {1, 2, 4})
            {
                if (tp > 1)
                {
                    int min_kv = model.n_kv_heads * model.head_dim;
                    if (min_kv / tp < 32)
                        continue;
                }

                auto shapes = buildShapes(model, tp);
                for (const auto &s : shapes)
                {
                    auto cfg = computeTileConfig(s.N, s.K, 1, payload, threads);
                    int N_chunks = (s.N + 63) / 64;
                    int n_tasks = (N_chunks + cfg.n_block_chunks - 1) / cfg.n_block_chunks;
                    int k_tasks = (cfg.k_tiles > 0) ? cfg.k_tiles : 1;
                    int total_tasks = n_tasks * k_tasks;

                    // Weight size in MB (Q8_0 native: 34 bytes/block, bpr blocks/row)
                    int bpr = (s.K + 31) / 32;
                    double wt_mb = (double)s.N * bpr * 34.0 / 1e6;

                    char wt_buf[16];
                    std::snprintf(wt_buf, sizeof(wt_buf), "%.1f", wt_mb);

                    // Short shape name
                    std::string sname = s.name;
                    auto pos = sname.find('_');
                    if (pos != std::string::npos)
                        sname = sname.substr(pos + 1);

                    table << model.name << sname << tp
                          << s.N << s.K
                          << categoryName(cfg.category) << cfg.n_block_chunks
                          << cfg.k_tiles << N_chunks << total_tasks
                          << wt_buf
                          << fort::endr;
                }
                table << fort::separator;
            }
        }

        std::cout << "\n=== Tile Config Inspection (Q8_0, M=1) ===\n"
                  << "Threads: " << threads << "\n\n"
                  << table.to_string() << std::endl;
    }

    // =========================================================================
    // TEST 5: Quantization Overhead Isolation — 7B shapes
    //
    // Measures the FP32→Q8_1 activation quantization cost separately from
    // the GEMV compute. Helps quantify what % of packed GEMV time is
    // spent on quantization vs actual matrix-vector multiply.
    // =========================================================================
    TEST_F(CPUNativeVNNIGemvTest, QuantOverhead_7B)
    {
        const auto &model = ALL_MODELS[3]; // 7B
        auto shapes = buildShapes(model, 1);
        double roofline = systemBandwidthGB();

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Shape" << "K" << "K_blocks"
              << "Quant \xc2\xb5s" << "Q8_0 \xc2\xb5s" << "Quant %"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 5; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &shape : shapes)
        {
            int K = shape.K;
            int K_blocks = (K + 31) / 32;

            // Random activations
            std::vector<float> A(K);
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto &v : A)
                v = dist(rng);

            // Benchmark quantization alone
            std::vector<Q8_1Block> q8_buf(K_blocks);
            for (int i = 0; i < WARMUP; ++i)
            {
                for (int kb = 0; kb < K_blocks; ++kb)
                {
                    int off = kb * 32;
                    int len = std::min(32, K - off);
                    simd::quantize_single_block(A.data() + off, q8_buf[kb], len);
                }
            }

            std::vector<double> quant_times(ITERS);
            for (int i = 0; i < ITERS; ++i)
            {
                auto t0 = std::chrono::high_resolution_clock::now();
                int kb = 0;
#if defined(__AVX512F__)
                bool aligned = (K % 32 == 0);
                if (aligned)
                {
                    for (; kb + 1 < K_blocks; kb += 2)
                        simd::quantize_two_blocks_avx512(
                            A.data() + kb * 32, q8_buf[kb], q8_buf[kb + 1]);
                }
#endif
                for (; kb < K_blocks; ++kb)
                {
                    int off = kb * 32;
                    int len = std::min(32, K - off);
                    simd::quantize_single_block(A.data() + off, q8_buf[kb], len);
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                quant_times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
            }
            std::sort(quant_times.begin(), quant_times.end());
            double quant_us = quant_times[std::max(0, (int)(ITERS * 0.1) - 1)];

            // Full production eager-interleaved GEMV.
            auto r = benchShape(shape, roofline);

            double quant_pct = (r.packed_us > 0) ? (quant_us / r.packed_us) * 100.0 : 0;

            char qu[16], pu[16], qp[16];
            std::snprintf(qu, sizeof(qu), "%.1f", quant_us);
            std::snprintf(pu, sizeof(pu), "%.0f", r.packed_us);
            std::snprintf(qp, sizeof(qp), "%.1f%%", quant_pct);

            std::string sname = shape.name;
            auto pos = sname.find('_');
            if (pos != std::string::npos)
                sname = sname.substr(pos + 1);

            table << sname << K << K_blocks
                  << qu << pu << qp
                  << fort::endr;
        }

        std::cout << "\n=== Quantization Overhead Isolation: Qwen 7B ===\n"
                  << "Quant = FP32\xe2\x86\x92Q8_1 activation quantization only\n"
                  << "Q8_0 = eager interleaved path (FP32\xe2\x86\x92Q8_1 quant + NativeVNNI compute)\n\n"
                  << table.to_string() << std::endl;
    }

    TEST_F(CPUNativeVNNIGemvTest, MTP_SmallM_FusedProjection_AllFormats)
    {
        constexpr int K = 512;
        constexpr int N0 = 768;
        constexpr int N1 = 512;
        constexpr int LOCAL_WARMUP = 3;
        constexpr int LOCAL_ITERS = 10;
        const std::array<int, 3> rows = {2, 3, 4};

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header << "Format" << "M" << "K" << "N0" << "N1"
              << "Fused us" << fort::endr;
        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 5; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &fmt : MTP_SMALL_M_FORMATS)
        {
            auto weights0 = createWeightsForFormat(fmt.name, N0, K);
            auto weights1 = createWeightsForFormat(fmt.name, N1, K);
            ASSERT_NE(weights0, nullptr) << fmt.name;
            ASSERT_NE(weights1, nullptr) << fmt.name;

            CPUNativeVNNIGemmKernel kernel0(weights0.get());
            CPUNativeVNNIGemmKernel kernel1(weights1.get());
            ASSERT_TRUE(kernel0.isValid()) << fmt.name;
            ASSERT_TRUE(kernel1.isValid()) << fmt.name;

            for (int M : rows)
            {
                auto input = TestTensorFactory::createFP32Random(
                    {static_cast<size_t>(M), static_cast<size_t>(K)},
                    -1.0f,
                    1.0f,
                    static_cast<uint32_t>(3000 + M + fmt.name.size()));
                FP32Tensor out0({static_cast<size_t>(M), static_cast<size_t>(N0)});
                FP32Tensor out1({static_cast<size_t>(M), static_cast<size_t>(N1)});
                std::vector<ITensorGemm::TensorProjectionDesc> projections = {
                    {&kernel0, &out0, N0, nullptr, "mtp_projection_0"},
                    {&kernel1, &out1, N1, nullptr, "mtp_projection_1"}};

                for (int i = 0; i < LOCAL_WARMUP; ++i)
                    ASSERT_TRUE(kernel0.multiply_fused_tensor(input.get(), projections, M, K));

                std::vector<double> times;
                times.reserve(LOCAL_ITERS);
                for (int i = 0; i < LOCAL_ITERS; ++i)
                {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    ASSERT_TRUE(kernel0.multiply_fused_tensor(input.get(), projections, M, K));
                    auto t1 = std::chrono::high_resolution_clock::now();
                    times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                }
                std::sort(times.begin(), times.end());
                const double p10 = times[std::max(0, static_cast<int>(times.size() / 10) - 1)];

                char fused_us[32];
                std::snprintf(fused_us, sizeof(fused_us), "%.1f", p10);
                table << fmt.name << M << K << N0 << N1 << fused_us << fort::endr;
            }
        }

        std::cout << "\n=== MTP Small-M CPU NativeVNNI Fused Projection Perf ===\n"
                  << "Rows M=2/3/4, two projections, activation quantized once per fused call.\n\n"
                  << table.to_string() << std::endl;
    }

    TEST_F(CPUNativeVNNIGemvTest, MoE_SingleTokenGateUp_FusedProjectionSpeedup)
    {
        /*
         * MoE decode does many M=1 expert gate/up projections for the same hidden
         * row. The production path should quantize that row once and run all
         * active expert projections inside one OpenMP region. This microbench
         * catches AVX2-only builds regressing to per-projection GEMV scheduling.
         *
         * Useful knobs:
         * - LLAMINAR_CPU_NVNNI_MOE_GATEUP_FORMATS=Q4_K,Q8_0
         * - LLAMINAR_CPU_NVNNI_MOE_GATEUP_K=5120
         * - LLAMINAR_CPU_NVNNI_MOE_GATEUP_N=1536
         * - LLAMINAR_CPU_NVNNI_MOE_GATEUP_EXPERTS=8
         * - LLAMINAR_CPU_NVNNI_MOE_GATEUP_MIN_SPEEDUP=1.10
         */
        const std::set<std::string> format_filters =
            getEnvCsvSet("LLAMINAR_CPU_NVNNI_MOE_GATEUP_FORMATS");
        const int K =
            std::max(32, getEnvInt("LLAMINAR_CPU_NVNNI_MOE_GATEUP_K").value_or(5120));
        const int expert_N =
            std::max(64, getEnvInt("LLAMINAR_CPU_NVNNI_MOE_GATEUP_N").value_or(1536));
        const int active_experts =
            std::max(1, getEnvInt("LLAMINAR_CPU_NVNNI_MOE_GATEUP_EXPERTS").value_or(8));
        const int warmup =
            std::max(0, getEnvInt("LLAMINAR_CPU_NVNNI_MOE_GATEUP_WARMUP").value_or(2));
        const int iterations =
            std::max(1, getEnvInt("LLAMINAR_CPU_NVNNI_MOE_GATEUP_ITERS").value_or(5));
        const double min_required_speedup =
            getEnvDouble("LLAMINAR_CPU_NVNNI_MOE_GATEUP_MIN_SPEEDUP").value_or(1.10);

        ASSERT_EQ(K % 32, 0)
            << "MoE gate/up fused projection perf expects K to be block-aligned";

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header << "Format" << "Experts" << "Proj" << "N" << "K"
              << "Fused us" << "Serial us" << "Speedup" << "RelL2" << fort::endr;
        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 8; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        int executed_cases = 0;
        for (const auto &fmt : MTP_SMALL_M_FORMATS)
        {
            if (format_filters.empty())
            {
                if (fmt.name != "Q4_K")
                    continue;
            }
            else if (!shouldRunName(format_filters, fmt.name))
            {
                continue;
            }

            const int projection_count = active_experts * 2;
            std::vector<std::unique_ptr<TensorBase>> weights;
            std::vector<std::unique_ptr<CPUNativeVNNIGemmKernel>> kernels;
            std::vector<std::unique_ptr<FP32Tensor>> fused_outputs;
            std::vector<std::unique_ptr<FP32Tensor>> serial_outputs;
            std::vector<ITensorGemm::TensorProjectionDesc> projections;
            weights.reserve(static_cast<size_t>(projection_count));
            kernels.reserve(static_cast<size_t>(projection_count));
            fused_outputs.reserve(static_cast<size_t>(projection_count));
            serial_outputs.reserve(static_cast<size_t>(projection_count));
            projections.reserve(static_cast<size_t>(projection_count));

            for (int p = 0; p < projection_count; ++p)
            {
                auto weights_for_projection =
                    createWeightsForFormat(fmt.name, static_cast<size_t>(expert_N), static_cast<size_t>(K));
                ASSERT_NE(weights_for_projection, nullptr) << fmt.name;

                auto kernel =
                    std::make_unique<CPUNativeVNNIGemmKernel>(weights_for_projection.get());
                ASSERT_TRUE(kernel->isValid()) << fmt.name << " projection=" << p;

                auto fused_output =
                    std::make_unique<FP32Tensor>(
                        std::vector<size_t>{1, static_cast<size_t>(expert_N)});
                auto serial_output =
                    std::make_unique<FP32Tensor>(
                        std::vector<size_t>{1, static_cast<size_t>(expert_N)});

                const char *projection_name = (p % 2 == 0) ? "moe_gate" : "moe_up";
                projections.push_back(
                    {kernel.get(), fused_output.get(), expert_N, nullptr, projection_name});
                weights.push_back(std::move(weights_for_projection));
                kernels.push_back(std::move(kernel));
                fused_outputs.push_back(std::move(fused_output));
                serial_outputs.push_back(std::move(serial_output));
            }

            auto input = TestTensorFactory::createFP32Random(
                {1, static_cast<size_t>(K)},
                -1.0f,
                1.0f,
                static_cast<uint32_t>(23000 + K + expert_N + active_experts + fmt.name.size()));

            auto run_fused = [&]()
            {
                const bool ok =
                    kernels.front()->multiply_fused_tensor(input.get(), projections, 1, K);
                if (!ok)
                    ADD_FAILURE() << "Fused MoE gate/up projection failed for " << fmt.name;
            };
            auto run_serial = [&]()
            {
                for (int p = 0; p < projection_count; ++p)
                {
                    const bool ok =
                        kernels[static_cast<size_t>(p)]->multiply_tensor(
                            input.get(),
                            serial_outputs[static_cast<size_t>(p)].get(),
                            1,
                            expert_N,
                            K);
                    if (!ok)
                        ADD_FAILURE() << "Serial MoE gate/up projection failed for "
                                      << fmt.name << " projection=" << p;
                }
            };

            run_fused();
            run_serial();

            std::vector<float> fused_flat;
            std::vector<float> serial_flat;
            fused_flat.reserve(static_cast<size_t>(projection_count) * static_cast<size_t>(expert_N));
            serial_flat.reserve(static_cast<size_t>(projection_count) * static_cast<size_t>(expert_N));
            for (int p = 0; p < projection_count; ++p)
            {
                const float *fused_data =
                    fused_outputs[static_cast<size_t>(p)]->data();
                const float *serial_data =
                    serial_outputs[static_cast<size_t>(p)]->data();
                fused_flat.insert(fused_flat.end(), fused_data, fused_data + expert_N);
                serial_flat.insert(serial_flat.end(), serial_data, serial_data + expert_N);
            }

            const VectorMetrics metrics =
                computeVectorMetrics(fused_flat.data(), serial_flat.data(), fused_flat.size());
            const std::string label =
                fmt.name + " MoE gate/up fused projection";
            assertVerifierMetricsStrict(metrics, label);

            const TimingSummary fused_timing =
                timeMicrobench(warmup, iterations, run_fused);
            const TimingSummary serial_timing =
                timeMicrobench(warmup, iterations, run_serial);
            const double speedup =
                fused_timing.min_us > 0.0 ? serial_timing.min_us / fused_timing.min_us : 0.0;

            char fused_us[32], serial_us[32], speed[32], rel_l2[32];
            std::snprintf(fused_us, sizeof(fused_us), "%.1f", fused_timing.min_us);
            std::snprintf(serial_us, sizeof(serial_us), "%.1f", serial_timing.min_us);
            std::snprintf(speed, sizeof(speed), "%.2fx", speedup);
            std::snprintf(rel_l2, sizeof(rel_l2), "%.3e", metrics.relative_l2);
            table << fmt.name << active_experts << projection_count << expert_N << K
                  << fused_us << serial_us << speed << rel_l2 << fort::endr;

            std::fprintf(
                stderr,
                "[CPUNativeVNNI][MOE_GATEUP] format=%s experts=%d projections=%d N=%d K=%d "
                "fused_us=%.3f serial_us=%.3f speedup=%.3f rel_l2=%.9e\n",
                fmt.name.c_str(),
                active_experts,
                projection_count,
                expert_N,
                K,
                fused_timing.min_us,
                serial_timing.min_us,
                speedup,
                metrics.relative_l2);

            EXPECT_GT(speedup, min_required_speedup)
                << label << " did not preserve fused MoE decode economy.";
            ++executed_cases;
        }

        EXPECT_GT(executed_cases, 0)
            << "No CPU NativeVNNI MoE gate/up fused projection cases selected.";

        std::cout << "\n=== MoE Single-Token Gate/Up Fused Projection Perf ===\n"
                  << "M=1, two projections per active expert, fused batch vs serial per-projection GEMV.\n\n"
                  << table.to_string() << std::endl;
    }

    TEST_F(CPUNativeVNNIGemvTest, MTP_SmallM_TrainerCsv_AllFormats)
    {
        /**
         * This is the CPU analogue of the CUDA/ROCm NativeVNNI trainer smoke:
         * emit stable, machine-readable rows for the current default CPU small-M
         * route. CPU does not yet consume a generated dispatch include, so the
         * only variant is "DEFAULT"; later analyzer work can add override
         * candidates without changing the CSV schema.
         *
         * Useful knobs:
         * - LLAMINAR_CPU_NVNNI_SMALL_M_FORMATS=Q4_0,IQ4_XS
         * - LLAMINAR_CPU_NVNNI_SMALL_M_M=2,3,4
         * - LLAMINAR_CPU_NVNNI_SMALL_M_CSV=/tmp/cpu_small_m.csv
         */
        constexpr int K = 512;
        constexpr int N0 = 768;
        constexpr int N1 = 512;
        const std::array<int, 3> default_rows = {2, 3, 4};

        const std::set<std::string> format_filters =
            getEnvCsvSet("LLAMINAR_CPU_NVNNI_SMALL_M_FORMATS");
        const std::set<std::string> m_filters =
            getEnvCsvSet("LLAMINAR_CPU_NVNNI_SMALL_M_M");
        const int max_cases =
            std::max(1, getEnvInt("LLAMINAR_CPU_NVNNI_SMALL_M_MAX_CASES").value_or(1000000));
        const int warmup =
            std::max(0, getEnvInt("LLAMINAR_CPU_NVNNI_SMALL_M_WARMUP").value_or(3));
        const int iters =
            std::max(1, getEnvInt("LLAMINAR_CPU_NVNNI_SMALL_M_ITERS").value_or(10));
        const std::string csv_path =
            getEnvString("LLAMINAR_CPU_NVNNI_SMALL_M_CSV");

        std::FILE *csv = nullptr;
        if (!csv_path.empty())
        {
            csv = std::fopen(csv_path.c_str(), "w");
            ASSERT_NE(csv, nullptr)
                << "Failed to open CPU NativeVNNI small-M trainer CSV: " << csv_path;
            std::fprintf(
                csv,
                "backend,phase,format,codebook,shape,k,m,projections,total_n,projection_ns,variant,min_us,mean_us,stddev_us,correctness_pass,is_best\n");
        }

        int executed_cases = 0;
        int emitted_rows = 0;
        for (const auto &fmt : MTP_SMALL_M_FORMATS)
        {
            if (!shouldRunName(format_filters, fmt.name))
                continue;

            auto weights0 = createWeightsForFormat(fmt.name, N0, K);
            auto weights1 = createWeightsForFormat(fmt.name, N1, K);
            ASSERT_NE(weights0, nullptr) << fmt.name;
            ASSERT_NE(weights1, nullptr) << fmt.name;

            CPUNativeVNNIGemmKernel kernel0(weights0.get());
            CPUNativeVNNIGemmKernel kernel1(weights1.get());
            ASSERT_TRUE(kernel0.isValid()) << fmt.name;
            ASSERT_TRUE(kernel1.isValid()) << fmt.name;
            const uint8_t codebook = kernel0.packedWeights().codebook_id;

            for (int M : default_rows)
            {
                if (!m_filters.empty() && m_filters.count(std::to_string(M)) == 0)
                    continue;
                if (executed_cases >= max_cases)
                    break;

                auto input = TestTensorFactory::createFP32Random(
                    {static_cast<size_t>(M), static_cast<size_t>(K)},
                    -1.0f,
                    1.0f,
                    static_cast<uint32_t>(9000 + M + fmt.name.size()));
                FP32Tensor out0({static_cast<size_t>(M), static_cast<size_t>(N0)});
                FP32Tensor out1({static_cast<size_t>(M), static_cast<size_t>(N1)});
                std::vector<ITensorGemm::TensorProjectionDesc> projections = {
                    {&kernel0, &out0, N0, nullptr, "mtp_projection_0"},
                    {&kernel1, &out1, N1, nullptr, "mtp_projection_1"}};

                for (int i = 0; i < warmup; ++i)
                    ASSERT_TRUE(kernel0.multiply_fused_tensor(input.get(), projections, M, K));

                std::vector<double> times;
                times.reserve(static_cast<size_t>(iters));
                for (int i = 0; i < iters; ++i)
                {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    ASSERT_TRUE(kernel0.multiply_fused_tensor(input.get(), projections, M, K));
                    auto t1 = std::chrono::high_resolution_clock::now();
                    times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                }
                const double min_us = *std::min_element(times.begin(), times.end());
                const double mean_us =
                    std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
                double variance = 0.0;
                for (double value : times)
                {
                    const double delta = value - mean_us;
                    variance += delta * delta;
                }
                variance /= static_cast<double>(times.size());
                const double stddev_us = std::sqrt(variance);

                if (csv)
                {
                    std::fprintf(
                        csv,
                        "cpu,small_m_fused_projection,%s,%u,SmallM_FusedProjection,%d,%d,2,%d,%d+%d,DEFAULT,%.3f,%.3f,%.3f,1,1\n",
                        fmt.name.c_str(),
                        static_cast<unsigned>(codebook),
                        K,
                        M,
                        N0 + N1,
                        N0,
                        N1,
                        min_us,
                        mean_us,
                        stddev_us);
                    std::fflush(csv);
                    ++emitted_rows;
                }

                std::fprintf(
                    stderr,
                    "[CPUNativeVNNI][SMALL_M][TRAINER] format=%s codebook=%u M=%d variant=DEFAULT time_us=%.3f\n",
                    fmt.name.c_str(),
                    static_cast<unsigned>(codebook),
                    M,
                    min_us);
                ++executed_cases;
            }

            if (executed_cases >= max_cases)
                break;
        }

        if (csv)
        {
            std::fclose(csv);
            ASSERT_GT(emitted_rows, 0)
                << "CPU NativeVNNI small-M trainer CSV had no rows.";
        }
        ASSERT_GT(executed_cases, 0)
            << "No CPU NativeVNNI small-M trainer cases selected.";
    }

    TEST_F(CPUNativeVNNIGemvTest, MTP_VerifierRows_GroupedVsSerial_Synthetic)
    {
        /**
         * Tight feedback loop for Phase 9.8 CPU verifier-row kernels.
         *
         * The full Qwen3.6 verifier economy benchmark is still the acceptance
         * gate, but it is too expensive for microkernel iteration.  This test
         * isolates the exact primitive we are tuning: pre-quantized M=2..4
         * grouped verifier rows versus M individual decode GEMVs.  It keeps
         * correctness strict with cosine, relative L2, and symmetric KL while
         * exporting timing rows that can be fed into the same trainer/dashboard
         * workflow used by the GPU dispatch sweeps.
         *
         * Useful knobs:
         * - LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS=Q4_K,Q6_K or "all"
         * - LLAMINAR_CPU_NVNNI_VERIFIER_M=2,3,4
         * - LLAMINAR_CPU_NVNNI_VERIFIER_SHAPE_NAME=Qwen36_FFN_DownProjection
         * - LLAMINAR_CPU_NVNNI_VERIFIER_N=5120
         * - LLAMINAR_CPU_NVNNI_VERIFIER_K=5120
         * - LLAMINAR_CPU_NVNNI_VERIFIER_ITERS=20
         * - LLAMINAR_CPU_NVNNI_VERIFIER_CSV=/tmp/cpu_verifier_rows.csv
         * - LLAMINAR_CPU_NVNNI_VERIFIER_VARIANTS=1 for M=3/4 forced wide vs pairwise A/B
         * - LLAMINAR_CPU_NVNNI_VERIFIER_MIN_SPEEDUP=1.0 to require economy
         * - LLAMINAR_CPU_NVNNI_VERIFIER_THREADS=28 to override standalone thread cap
         */
        applyVerifierRowsThreadCapForStandalonePerf();

        struct VerifierShape
        {
            std::string name;
            int N = 0;
            int K = 0;
        };

        const int default_N = 5120;
        const int default_K = 5120;
        const int N =
            getEnvInt("LLAMINAR_CPU_NVNNI_VERIFIER_N").value_or(default_N);
        const int K =
            getEnvInt("LLAMINAR_CPU_NVNNI_VERIFIER_K").value_or(default_K);
        ASSERT_GT(N, 0);
        ASSERT_GT(K, 0);
        ASSERT_EQ(K % 32, 0)
            << "NativeVNNI verifier microbench expects K to be block-aligned";

        /*
         * The refresh/training wrapper drives this test once per production
         * projection shape.  Preserve that shape name in CSV output so the
         * generated policy table can be audited later; otherwise different
         * Qwen3.6 shapes with the same codebook/M bucket become indistinguishable
         * in summaries and generated comments.
         */
        std::string shape_name =
            getEnvString("LLAMINAR_CPU_NVNNI_VERIFIER_SHAPE_NAME");
        if (shape_name.empty())
            shape_name = "Qwen36Verifier";
        const std::vector<VerifierShape> shapes = {
            {shape_name, N, K},
        };

        const std::string format_filter_raw =
            getEnvString("LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS");
        std::set<std::string> format_filters =
            getEnvCsvSet("LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS");
        if (format_filter_raw.empty())
        {
            /*
             * Compact default for local iteration: one K-quant, one INT8
             * predecoded K-quant, and one IQ path.  Set the env var to "all"
             * before generating broad training data.
             */
            format_filters = {toLower("Q4_K"), toLower("Q6_K"), toLower("IQ4_XS")};
        }
        const bool run_all_formats = format_filters.count("all") > 0;
        if (run_all_formats)
            format_filters.clear();

        const std::set<std::string> m_filters =
            getEnvCsvSet("LLAMINAR_CPU_NVNNI_VERIFIER_M");
        const int warmup =
            std::max(0, getEnvInt("LLAMINAR_CPU_NVNNI_VERIFIER_WARMUP").value_or(2));
        const int iterations =
            std::max(1, getEnvInt("LLAMINAR_CPU_NVNNI_VERIFIER_ITERS").value_or(5));
        const int max_cases =
            std::max(1, getEnvInt("LLAMINAR_CPU_NVNNI_VERIFIER_MAX_CASES").value_or(1000000));
        const ISAPath isa_path = parseISAPathForVerifierBench();
        const std::array<int, 3> rows = {2, 3, 4};
        const std::string csv_path =
            getEnvString("LLAMINAR_CPU_NVNNI_VERIFIER_CSV");
        const bool run_m4_variants =
            getEnvInt("LLAMINAR_CPU_NVNNI_VERIFIER_VARIANTS").value_or(0) != 0;
        const double min_required_speedup =
            getEnvDouble("LLAMINAR_CPU_NVNNI_VERIFIER_MIN_SPEEDUP")
                .value_or(csv_path.empty() ? 1.0 : 0.0);

        std::FILE *csv = nullptr;
        if (!csv_path.empty())
        {
            csv = std::fopen(csv_path.c_str(), "w");
            ASSERT_NE(csv, nullptr)
                << "Failed to open CPU NativeVNNI verifier CSV: " << csv_path;
            std::fprintf(
                csv,
                "backend,phase,format,codebook,is_nibble_lut,payload_bytes,is_asymmetric,is_superblock,shape,n,k,m,isa,grouped_min_us,serial_min_us,speedup,grouped_mean_us,serial_mean_us,cosine,relative_l2,symmetric_kl,max_abs,correctness_pass\n");
        }

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header << "Format" << "M" << "N" << "K" << "ISA"
              << "Grouped us" << "Serial us" << "Speedup"
              << "Cos" << "RelL2" << "SymKL" << fort::endr;
        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 10; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        int executed_cases = 0;
        int emitted_rows = 0;
        for (const auto &shape : shapes)
        {
            for (const auto &fmt : MTP_SMALL_M_FORMATS)
            {
                if (!shouldRunName(format_filters, fmt.name))
                    continue;

                auto weights = createWeightsForFormat(
                    fmt.name,
                    static_cast<size_t>(shape.N),
                    static_cast<size_t>(shape.K));
                ASSERT_NE(weights, nullptr) << fmt.name;

                CPUNativeVNNIGemmKernel kernel(weights.get());
                ASSERT_TRUE(kernel.isValid()) << fmt.name;
                const auto &packed = kernel.packedWeights();
                const int K_blocks = packed.blocks_per_row;

                for (int M : rows)
                {
                    if (!m_filters.empty() && m_filters.count(std::to_string(M)) == 0)
                        continue;
                    if (executed_cases >= max_cases)
                        break;

                    std::mt19937 rng(
                        static_cast<uint32_t>(17000 + M + shape.N + shape.K + fmt.name.size()));
                    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                    std::vector<float> input(static_cast<size_t>(M) * shape.K);
                    for (float &value : input)
                        value = dist(rng);

                    std::vector<Q8_1Block> q8_rows(
                        static_cast<size_t>(M) * static_cast<size_t>(K_blocks));
                    quantize_activations_to_q8_1(
                        input.data(),
                        q8_rows.data(),
                        M,
                        shape.K,
                        K_blocks);

                    std::vector<float> grouped(
                        static_cast<size_t>(M) * static_cast<size_t>(shape.N),
                        0.0f);
                    std::vector<float> serial(
                        static_cast<size_t>(M) * static_cast<size_t>(shape.N),
                        0.0f);

                    auto run_grouped = [&]()
                    {
                        gemm_native_vnni_preq_decode_equivalent_rows(
                            packed,
                            q8_rows.data(),
                            grouped.data(),
                            M,
                            shape.N,
                            isa_path);
                    };
                    auto run_serial = [&]()
                    {
                        for (int row = 0; row < M; ++row)
                        {
                            gemv_native_vnni_preq(
                                packed,
                                q8_rows.data() + static_cast<size_t>(row) * K_blocks,
                                serial.data() + static_cast<size_t>(row) * shape.N,
                                isa_path);
                        }
                    };

                    run_grouped();
                    run_serial();
                    const VectorMetrics metrics =
                        computeVectorMetrics(
                            grouped.data(),
                            serial.data(),
                            grouped.size());
                    const std::string label =
                        fmt.name + " " + shape.name + " M=" + std::to_string(M);
                    assertVerifierMetricsStrict(metrics, label);

                    const TimingSummary grouped_timing =
                        timeMicrobench(warmup, iterations, run_grouped);
                    const TimingSummary serial_timing =
                        timeMicrobench(warmup, iterations, run_serial);
                    const double speedup =
                        grouped_timing.min_us > 0.0
                            ? serial_timing.min_us / grouped_timing.min_us
                            : 0.0;

                    char grouped_us[32], serial_us[32], speed[32];
                    char cos_buf[32], l2_buf[32], kl_buf[32];
                    std::snprintf(grouped_us, sizeof(grouped_us), "%.1f", grouped_timing.min_us);
                    std::snprintf(serial_us, sizeof(serial_us), "%.1f", serial_timing.min_us);
                    std::snprintf(speed, sizeof(speed), "%.2fx", speedup);
                    std::snprintf(cos_buf, sizeof(cos_buf), "%.9f", metrics.cosine);
                    std::snprintf(l2_buf, sizeof(l2_buf), "%.3e", metrics.relative_l2);
                    std::snprintf(kl_buf, sizeof(kl_buf), "%.3e", metrics.symmetric_kl);

                    table << fmt.name << M << shape.N << shape.K << isaPathName(isa_path)
                          << grouped_us << serial_us << speed
                          << cos_buf << l2_buf << kl_buf << fort::endr;

                    if (csv)
                    {
                        std::fprintf(
                            csv,
                            "cpu,verifier_rows,%s,%u,%d,%d,%d,%d,%s,%d,%d,%d,%s,%.3f,%.3f,%.6f,%.3f,%.3f,%.9f,%.9e,%.9e,%.9e,1\n",
                            fmt.name.c_str(),
                            static_cast<unsigned>(packed.codebook_id),
                            packed.is_nibble_lut ? 1 : 0,
                            packed.payload_bytes,
                            packed.is_asymmetric ? 1 : 0,
                            packed.is_superblock ? 1 : 0,
                            shape.name.c_str(),
                            shape.N,
                            shape.K,
                            M,
                            isaPathName(isa_path),
                            grouped_timing.min_us,
                            serial_timing.min_us,
                            speedup,
                            grouped_timing.mean_us,
                            serial_timing.mean_us,
                            metrics.cosine,
                            metrics.relative_l2,
                            metrics.symmetric_kl,
                            metrics.max_abs);
                        std::fflush(csv);
                        ++emitted_rows;
                    }

                    std::fprintf(
                        stderr,
                        "[CPUNativeVNNI][VERIFIER_ROWS] format=%s codebook=%u nibble=%d payload=%d asymmetric=%d superblock=%d shape=%s N=%d K=%d M=%d isa=%s grouped_us=%.3f serial_us=%.3f speedup=%.3f cosine=%.9f rel_l2=%.9e symmetric_kl=%.9e\n",
                        fmt.name.c_str(),
                        static_cast<unsigned>(packed.codebook_id),
                        packed.is_nibble_lut ? 1 : 0,
                        packed.payload_bytes,
                        packed.is_asymmetric ? 1 : 0,
                        packed.is_superblock ? 1 : 0,
                        shape.name.c_str(),
                        shape.N,
                        shape.K,
                        M,
                        isaPathName(isa_path),
                        grouped_timing.min_us,
                        serial_timing.min_us,
                        speedup,
                        metrics.cosine,
                        metrics.relative_l2,
                        metrics.symmetric_kl);

                    EXPECT_GT(grouped_timing.min_us, 0.0);
                    EXPECT_GT(serial_timing.min_us, 0.0);
                    if (min_required_speedup > 0.0)
                    {
                        EXPECT_GT(speedup, min_required_speedup)
                            << label << " verifier rows are decode-equivalent "
                            << "but not economical enough for MTP.";
                    }

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                    if (run_m4_variants && M >= 3)
                    {
                        if (!verifierBenchUsesAVX512(isa_path))
                        {
                            FAIL()
                                << "LLAMINAR_CPU_NVNNI_VERIFIER_VARIANTS requires "
                                << "an active AVX512-VNNI verifier path";
                        }

                        std::vector<float> wide_rows(
                            static_cast<size_t>(M) * static_cast<size_t>(shape.N),
                            0.0f);
                        std::vector<float> pairwise_rows(
                            static_cast<size_t>(M) * static_cast<size_t>(shape.N),
                            0.0f);

                        /*
                         * Force both policies through the production verifier
                         * entrypoint.  Unlike the older direct M=4 probe, this
                         * also exercises k-tiled long-K shapes, so the trainer
                         * can learn the policy that matters for real verifier
                         * projections.
                         */
                        auto run_wide_policy = [&]()
                        {
                            gemm_native_vnni_preq_decode_equivalent_rows(
                                packed,
                                q8_rows.data(),
                                wide_rows.data(),
                                M,
                                shape.N,
                                isa_path,
                                VerifierRowsPolicy::WideRows);
                        };
                        auto run_pairwise_policy = [&]()
                        {
                            gemm_native_vnni_preq_decode_equivalent_rows(
                                packed,
                                q8_rows.data(),
                                pairwise_rows.data(),
                                M,
                                shape.N,
                                isa_path,
                                VerifierRowsPolicy::Pairwise);
                        };

                        run_wide_policy();
                        run_pairwise_policy();
                        const VectorMetrics wide_metrics =
                            computeVectorMetrics(
                                wide_rows.data(),
                                serial.data(),
                                wide_rows.size());
                        const VectorMetrics pairwise_metrics =
                            computeVectorMetrics(
                                pairwise_rows.data(),
                                serial.data(),
                                pairwise_rows.size());
                        assertVerifierMetricsStrict(
                            wide_metrics,
                            label + " forced wide verifier policy");
                        assertVerifierMetricsStrict(
                            pairwise_metrics,
                            label + " forced pairwise verifier policy");

                        const TimingSummary wide_timing =
                            timeMicrobench(warmup, iterations, run_wide_policy);
                        const TimingSummary pairwise_timing =
                            timeMicrobench(warmup, iterations, run_pairwise_policy);
                        const double wide_speedup =
                            wide_timing.min_us > 0.0
                                ? serial_timing.min_us / wide_timing.min_us
                                : 0.0;
                        const double pairwise_speedup =
                            pairwise_timing.min_us > 0.0
                                ? serial_timing.min_us / pairwise_timing.min_us
                                : 0.0;

                        if (csv)
                        {
                            auto emit_variant = [&](const char *phase,
                                                    const TimingSummary &timing,
                                                    double variant_speedup,
                                                    const VectorMetrics &variant_metrics)
                            {
                                std::fprintf(
                                    csv,
                                    "cpu,%s,%s,%u,%d,%d,%d,%d,%s,%d,%d,%d,%s,%.3f,%.3f,%.6f,%.3f,%.3f,%.9f,%.9e,%.9e,%.9e,1\n",
                                    phase,
                                    fmt.name.c_str(),
                                    static_cast<unsigned>(packed.codebook_id),
                                    packed.is_nibble_lut ? 1 : 0,
                                    packed.payload_bytes,
                                    packed.is_asymmetric ? 1 : 0,
                                    packed.is_superblock ? 1 : 0,
                                    shape.name.c_str(),
                                    shape.N,
                                    shape.K,
                                    M,
                                    isaPathName(isa_path),
                                    timing.min_us,
                                    serial_timing.min_us,
                                    variant_speedup,
                                    timing.mean_us,
                                    serial_timing.mean_us,
                                    variant_metrics.cosine,
                                    variant_metrics.relative_l2,
                                    variant_metrics.symmetric_kl,
                                    variant_metrics.max_abs);
                                std::fflush(csv);
                                ++emitted_rows;
                            };
                            emit_variant(
                                "verifier_rows_wide_policy",
                                wide_timing,
                                wide_speedup,
                                wide_metrics);
                            emit_variant(
                                "verifier_rows_pairwise_policy",
                                pairwise_timing,
                                pairwise_speedup,
                                pairwise_metrics);
                        }

                        std::fprintf(
                            stderr,
                            "[CPUNativeVNNI][VERIFIER_ROWS_POLICY_VARIANT] "
                            "format=%s shape=%s M=%d wide_us=%.3f pairwise_us=%.3f "
                            "serial_us=%.3f wide_speedup=%.3f pairwise_speedup=%.3f "
                            "wide_vs_pairwise=%.3f cosine=%.9f rel_l2=%.9e symmetric_kl=%.9e\n",
                            fmt.name.c_str(),
                            shape.name.c_str(),
                            M,
                            wide_timing.min_us,
                            pairwise_timing.min_us,
                            serial_timing.min_us,
                            wide_speedup,
                            pairwise_speedup,
                            pairwise_timing.min_us > 0.0
                                ? pairwise_timing.min_us / wide_timing.min_us
                                : 0.0,
                            wide_metrics.cosine,
                            wide_metrics.relative_l2,
                            wide_metrics.symmetric_kl);
                    }
#else
                    if (run_m4_variants && M >= 3)
                    {
                        FAIL()
                            << "LLAMINAR_CPU_NVNNI_VERIFIER_VARIANTS requires an AVX512-VNNI build";
                    }
#endif
                    ++executed_cases;
                }

                if (executed_cases >= max_cases)
                    break;
            }
        }

        if (csv)
        {
            std::fclose(csv);
            ASSERT_GT(emitted_rows, 0)
                << "CPU NativeVNNI verifier-row CSV had no rows.";
        }

        ASSERT_GT(executed_cases, 0)
            << "No CPU NativeVNNI verifier-row microbench cases selected.";

        std::cout << "\n=== CPU NativeVNNI Verifier Rows: Grouped vs Serial Decode GEMVs ===\n"
                  << "Pre-quantized Q8_1 activations; strict cosine/L2/KL equivalence against serial M decode GEMVs.\n\n"
                  << table.to_string() << std::endl;
    }

    /**
     * @brief One forceable CPU M=1 production schedule.
     *
     * The canonical ID is the policy/compiler identity. The shorter route name
     * is emitted by production telemetry and independently proves that the
     * requested schedule survived physical-width normalization.
     */
    struct CPUDecodeScheduleCandidate
    {
        const char *route_name;
        const char *canonical_id;
        DecodeSchedulePolicy policy;
    };

    inline constexpr std::array<CPUDecodeScheduleCandidate, 5>
        CPU_DECODE_SCHEDULE_CANDIDATES = {{
            {"Nbc1", "cpu.nvnni.decode.n_chunk_grid.nbc1", DecodeSchedulePolicy::Nbc1},
            {"Nbc2", "cpu.nvnni.decode.n_chunk_grid.nbc2", DecodeSchedulePolicy::Nbc2},
            {"Nbc4", "cpu.nvnni.decode.n_chunk_grid.nbc4", DecodeSchedulePolicy::Nbc4},
            {"Nbc8", "cpu.nvnni.decode.n_chunk_grid.nbc8", DecodeSchedulePolicy::Nbc8},
            {"Nbc16", "cpu.nvnni.decode.n_chunk_grid.nbc16", DecodeSchedulePolicy::Nbc16},
        }};

    /** @brief One forceable grouped verifier schedule. */
    struct CPUGroupedDecodeScheduleCandidate
    {
        const char *route_name;
        const char *canonical_id;
        VerifierRowsPolicy policy;
    };

    /**
     * @brief Complete grouped candidate inventory used by production dispatch.
     *
     * These are not diagnostic alternatives. Every value enters
     * `gemm_native_vnni_preq_decode_equivalent_rows()`, publish their physical
     * route through perfstats, and preserve the serial-M1 accumulation order.
     */
    inline constexpr std::array<CPUGroupedDecodeScheduleCandidate, 9>
        CPU_GROUPED_DECODE_SCHEDULE_CANDIDATES = {{
            {
                "Pairwise",
                "cpu.nvnni.verifier.pairwise",
                VerifierRowsPolicy::Pairwise,
            },
            {
                "WideRows",
                "cpu.nvnni.verifier.wide_rows",
                VerifierRowsPolicy::WideRows,
            },
            {
                "FullKRowChunkGrid",
                "cpu.nvnni.verifier.full_k.row_chunk_grid",
                VerifierRowsPolicy::FullKRowChunkGrid,
            },
            {
                "FullKTwoRowNbc1",
                "cpu.nvnni.verifier.full_k.two_row_n_major.nbc1",
                VerifierRowsPolicy::FullKTwoRowNbc1,
            },
            {
                "FullKTwoRowNbc2",
                "cpu.nvnni.verifier.full_k.two_row_n_major.nbc2",
                VerifierRowsPolicy::FullKTwoRowNbc2,
            },
            {
                "FullKTwoRowPairGridNbc1",
                "cpu.nvnni.verifier.full_k.two_row_pair_grid.nbc1",
                VerifierRowsPolicy::FullKTwoRowPairGridNbc1,
            },
            {
                "FullKTwoRowPairGridNbc2",
                "cpu.nvnni.verifier.full_k.two_row_pair_grid.nbc2",
                VerifierRowsPolicy::FullKTwoRowPairGridNbc2,
            },
            {
                "FullKTwoRowPairGridNbc4",
                "cpu.nvnni.verifier.full_k.two_row_pair_grid.nbc4",
                VerifierRowsPolicy::FullKTwoRowPairGridNbc4,
            },
            {
                "FullKTwoRowPairGridNbc8",
                "cpu.nvnni.verifier.full_k.two_row_pair_grid.nbc8",
                VerifierRowsPolicy::FullKTwoRowPairGridNbc8,
            },
        }};

    /** Return one forceable schedule by its compiler-facing canonical ID. */
    const CPUDecodeScheduleCandidate *findCPUDecodeScheduleCandidate(
        std::string_view candidate_id)
    {
        const std::string normalized = toLower(std::string(candidate_id));
        const auto iterator = std::find_if(
            CPU_DECODE_SCHEDULE_CANDIDATES.begin(),
            CPU_DECODE_SCHEDULE_CANDIDATES.end(),
            [&](const CPUDecodeScheduleCandidate &candidate)
            {
                return normalized == toLower(candidate.canonical_id);
            });
        return iterator == CPU_DECODE_SCHEDULE_CANDIDATES.end()
                   ? nullptr
                   : &*iterator;
    }

    /** Return one grouped schedule by its compiler-facing canonical ID. */
    const CPUGroupedDecodeScheduleCandidate *
    findCPUGroupedDecodeScheduleCandidate(std::string_view candidate_id)
    {
        const std::string normalized = toLower(std::string(candidate_id));
        const auto iterator = std::find_if(
            CPU_GROUPED_DECODE_SCHEDULE_CANDIDATES.begin(),
            CPU_GROUPED_DECODE_SCHEDULE_CANDIDATES.end(),
            [&](const CPUGroupedDecodeScheduleCandidate &candidate)
            {
                return normalized == toLower(candidate.canonical_id);
            });
        return iterator == CPU_GROUPED_DECODE_SCHEDULE_CANDIDATES.end()
                   ? nullptr
                   : &*iterator;
    }

    /** Mix one byte string into a stable nonzero FNV-1a measurement seed. */
    uint64_t cpuPairedSeed(std::string_view identity)
    {
        uint64_t hash = 1469598103934665603ULL;
        for (const unsigned char byte : identity)
        {
            hash ^= static_cast<uint64_t>(byte);
            hash *= 1099511628211ULL;
        }
        return hash == 0 ? 1 : hash;
    }

    /** Extend a cell seed with one pair index for retained replay provenance. */
    uint64_t cpuPairedOrderSeed(uint64_t cell_seed, size_t pair_index)
    {
        uint64_t hash = cell_seed;
        for (int byte = 0; byte < 8; ++byte)
        {
            hash ^= (static_cast<uint64_t>(pair_index) >> (byte * 8)) & 0xffULL;
            hash *= 1099511628211ULL;
        }
        return hash == 0 ? 1 : hash;
    }

    /** Correctness and physical-route proof prepared before paired timing. */
    struct CPUPairedCandidateEvidence
    {
        CPUDecodeRouteEvidence route;
        llaminar2::test::trainer::FP32Evidence comparison;
        size_t repeat_byte_mismatches = 0;
        bool route_ok = false;
        bool numerical_correctness = false;
        bool correctness_pass = false;
    };

    /**
     * @brief Write one half of an architecture-qualified CPU timing pair.
     *
     * CSV formatting occurs after the timed launch. It is deliberately kept
     * outside the measured window with route snapshots, output comparisons,
     * fixture construction, and every dynamic allocation.
     */
    void writeCPUPairedSample(
        std::ostream &output,
        const native_vnni_dispatch::NativeVNNIPairedTimingRequest &request,
        const CPUDecodeScheduleCandidate &candidate,
        const CPUPairedCandidateEvidence &evidence,
        std::string_view candidate_role,
        size_t pair_index,
        size_t within_pair_order,
        size_t configured_pair_count,
        uint64_t cell_order_seed,
        uint64_t pair_order_seed,
        double latency_us,
        int warmup_count)
    {
        const char *observed_path = evidence.route.serial_kpart
                                        ? "serial-kpart"
                                        : "serial-full-k";
        output << "native-vnni-paired-interleaved-v3,"
               << request.request_id << ",cpu,decode_m1,"
               << request.source_format << ','
               << request.source_codebook << ','
               << request.execution_codebook << ','
               << request.architecture_class << ','
               << request.shape << ",eager,1,"
               << request.n << ',' << request.k << ','
               << pair_index << ',' << configured_pair_count << ','
               << within_pair_order << ',' << cell_order_seed << ','
               << pair_order_seed << ',' << candidate_role << ','
               << candidate.canonical_id << ",1,"
               << std::setprecision(17) << latency_us << ','
               << std::hexfloat << latency_us << std::defaultfloat << ','
               << warmup_count << ','
               << evidence.comparison.mismatch_count << ','
               << evidence.comparison.first_mismatch_index << ','
               << evidence.repeat_byte_mismatches << ','
               << std::setprecision(17)
               << evidence.comparison.max_abs << ','
               << evidence.comparison.relative_l2 << ','
               << evidence.comparison.cosine << ','
               << evidence.comparison.symmetric_kld << ','
               << evidence.comparison.actual_digest << ','
               << evidence.comparison.expected_digest
               << ",0,1,1," << (evidence.route_ok ? 1 : 0) << ','
               << candidate.canonical_id << ',' << observed_path << ','
               << evidence.route.n_block_chunks << ','
               << evidence.route.k_tiles << ",0,"
               << "cpu.nvnni.frozen-serial-m1-arithmetic-v1,"
               << (evidence.numerical_correctness ? 1 : 0) << ','
               << (evidence.correctness_pass ? 1 : 0) << '\n';
    }

    /**
     * @brief Execute an authoritative CPU paired-confirmation transaction.
     *
     * Every request constructs and packs its fixture before timing, primes the
     * OpenMP team and persistent K-partial high-water buffer, proves each forced
     * route with one untimed performance-counter launch, and then disables the
     * collector. The measured loop contains only steady-clock reads and one
     * production `gemv_native_vnni_preq()` call. No output vector, workspace,
     * route tag, CSV field, or sample container is allocated or freed there.
     */
    void runCPUM1PairedConfirmation(
        const std::string &request_manifest_path,
        const std::string &output_path)
    {
        const auto manifest =
            native_vnni_dispatch::loadNativeVnniPairedRequestManifest(
                request_manifest_path);
        ASSERT_EQ(manifest.backend, "cpu");
        ASSERT_FALSE(manifest.requests.empty());
        ASSERT_FALSE(PerfStatsCollector::isEnabled())
            << "CPU paired timing requires ambient perfstats/profiling to be disabled";
        ASSERT_TRUE(native_vnni_dispatch::profilerRequestId().empty())
            << "isolated Linux-perf profiling cannot run inside paired timing";

        const int warmups = std::max(
            5, getEnvInt("LLAMINAR_CPU_NVNNI_DECODE_WARMUP").value_or(5));
        const int pair_count = std::max(
            30, getEnvInt("LLAMINAR_CPU_NVNNI_DECODE_ITERS").value_or(30));
        const std::string build_isa = compiledNativeVNNIBuildISAName();
        const std::string runtime_isa = activeISANameForVerifierTrainer();
        const int threads = omp_get_max_threads();
        const std::string architecture_suffix =
            "|build=" + build_isa + "|runtime=" + runtime_isa +
            "|threads=" + std::to_string(threads);

        const std::filesystem::path output_file_path(output_path);
        if (output_file_path.has_parent_path())
            std::filesystem::create_directories(output_file_path.parent_path());
        std::ofstream output(output_file_path, std::ios::out | std::ios::trunc);
        ASSERT_TRUE(output.is_open()) << output_path;
        output
            << "protocol_version,request_id,backend,phase,source_format,"
               "source_codebook,execution_codebook,architecture_class,shape,"
               "execution_mode,m,n,k,pair_index,configured_pair_count,"
               "within_pair_order,cell_order_seed,pair_order_seed,candidate_role,"
               "candidate_id,timed_replays,latency_us,latency_us_hex,warmup_count,"
               "bit_mismatches,first_bit_mismatch,repeat_byte_mismatches,max_abs,"
               "relative_l2,cosine,symmetric_kld,grouped_output_digest,"
               "serial_output_digest,graph_capture_ok,workspace_ok,"
               "explicit_stream_ok,route_counter_ok,observed_candidate_id,"
               "observed_path,observed_tile_n,observed_cpt,observed_effective_kb,"
               "serial_m1_candidate_id,numerical_correctness,correctness_pass\n";

        size_t executed_requests = 0;
        for (const auto &request : manifest.requests)
        {
            ASSERT_EQ(request.execution_mode, "eager") << request.request_id;
            ASSERT_EQ(request.m, 1) << request.request_id;
            ASSERT_GE(request.architecture_class.size(), architecture_suffix.size())
                << request.request_id;
            ASSERT_EQ(
                request.architecture_class.substr(
                    request.architecture_class.size() - architecture_suffix.size()),
                architecture_suffix)
                << request.request_id << " belongs to another CPU ISA/thread regime";

            const auto &shape_manifest =
                native_vnni_dispatch::nativeVnniShapeManifest();
            const auto shape_iterator = std::find_if(
                shape_manifest.begin(), shape_manifest.end(),
                [&](const native_vnni_dispatch::NativeVNNIShapeSpec &shape)
                {
                    return shape.name == request.shape;
                });
            if (shape_iterator == shape_manifest.end())
            {
                ASSERT_EQ(request.shape.rfind("CPUDecodeAutoSeal_", 0), 0u)
                    << request.request_id << " unknown shape " << request.shape;
                ASSERT_GT(request.n, 0) << request.request_id;
                ASSERT_GT(request.k, 0) << request.request_id;
                ASSERT_EQ(request.k % 32, 0) << request.request_id;
            }
            else
            {
                ASSERT_EQ(shape_iterator->N, request.n) << request.request_id;
                ASSERT_EQ(shape_iterator->K, request.k) << request.request_id;
            }

            const auto format_iterator = std::find_if(
                MTP_SMALL_M_FORMATS.begin(), MTP_SMALL_M_FORMATS.end(),
                [&](const FormatSpec &format)
                {
                    return toLower(format.name) ==
                           toLower(request.source_format);
                });
            ASSERT_NE(format_iterator, MTP_SMALL_M_FORMATS.end())
                << request.request_id << " unknown format "
                << request.source_format;
            const CPUDecodeScheduleCandidate *selected =
                findCPUDecodeScheduleCandidate(request.selected_candidate_id);
            const CPUDecodeScheduleCandidate *exact =
                findCPUDecodeScheduleCandidate(request.exact_candidate_id);
            ASSERT_NE(selected, nullptr) << request.request_id;
            ASSERT_NE(exact, nullptr) << request.request_id;
            ASSERT_NE(selected, exact) << request.request_id;

            auto weights = createWeightsForFormat(
                format_iterator->name,
                static_cast<size_t>(request.n),
                static_cast<size_t>(request.k));
            ASSERT_NE(weights, nullptr) << request.request_id;
            CPUNativeVNNIGemmKernel kernel(weights.get());
            ASSERT_TRUE(kernel.isValid()) << request.request_id;
            const auto &packed = kernel.packedWeights();
            ASSERT_EQ(
                static_cast<int>(packed.codebook_id), request.source_codebook)
                << request.request_id;
            ASSERT_EQ(
                static_cast<int>(packed.codebook_id), request.execution_codebook)
                << request.request_id;

            std::mt19937 rng(static_cast<uint32_t>(
                0xDEC0DEu + request.n + request.k +
                format_iterator->name.size()));
            std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
            std::vector<float> input(static_cast<size_t>(request.k));
            for (float &value : input)
                value = distribution(rng);
            std::vector<Q8_1Block> quantized(
                static_cast<size_t>(packed.blocks_per_row));
            quantize_activations_to_q8_1(
                input.data(), quantized.data(), 1, request.k,
                packed.blocks_per_row);

            std::vector<float> oracle(static_cast<size_t>(request.n), 0.0f);
            std::array<std::vector<float>, 2> candidate_outputs = {
                std::vector<float>(static_cast<size_t>(request.n), 0.0f),
                std::vector<float>(static_cast<size_t>(request.n), 0.0f),
            };
            std::array<std::vector<float>, 2> first_outputs = {
                std::vector<float>(static_cast<size_t>(request.n), 0.0f),
                std::vector<float>(static_cast<size_t>(request.n), 0.0f),
            };
            std::array<std::vector<double>, 2> samples = {
                std::vector<double>(static_cast<size_t>(pair_count), 0.0),
                std::vector<double>(static_cast<size_t>(pair_count), 0.0),
            };
            constexpr ISAPath isa_path = ISAPath::AUTO;
            gemv_native_vnni_preq(
                packed, quantized.data(), oracle.data(), isa_path,
                DecodeSchedulePolicy::FrozenSerialOracle);

            const std::array<const CPUDecodeScheduleCandidate *, 2> candidates = {
                selected, exact};
            std::array<CPUPairedCandidateEvidence, 2> evidence;
            for (size_t candidate_index = 0;
                 candidate_index < candidates.size(); ++candidate_index)
            {
                const CPUDecodeScheduleCandidate &candidate =
                    *candidates[candidate_index];
                const auto launch = [&]
                {
                    gemv_native_vnni_preq(
                        packed,
                        quantized.data(),
                        candidate_outputs[candidate_index].data(),
                        isa_path,
                        candidate.policy);
                };

                PerfStatsCollector::reset();
                ASSERT_EQ(setenv("LLAMINAR_PERF_STATS_JSON", "1", 1), 0);
                launch();
                evidence[candidate_index].route = findCPUDecodeRoute(
                    request.n, request.k, packed.codebook_id);
                ASSERT_EQ(unsetenv("LLAMINAR_PERF_STATS_JSON"), 0);
                ASSERT_FALSE(PerfStatsCollector::isEnabled());
                std::copy(
                    candidate_outputs[candidate_index].begin(),
                    candidate_outputs[candidate_index].end(),
                    first_outputs[candidate_index].begin());
                launch();

                CPUPairedCandidateEvidence &candidate_evidence =
                    evidence[candidate_index];
                candidate_evidence.repeat_byte_mismatches =
                    llaminar2::test::trainer::nativeByteMismatchCount(
                        first_outputs[candidate_index],
                        candidate_outputs[candidate_index]);
                candidate_evidence.comparison =
                    llaminar2::test::trainer::compareFP32(
                        candidate_outputs[candidate_index], oracle,
                        static_cast<size_t>(request.n));
                candidate_evidence.route_ok =
                    candidate_evidence.route.found &&
                    candidate_evidence.route.count == 1 &&
                    candidate_evidence.route.build_isa == build_isa &&
                    candidate_evidence.route.isa == runtime_isa &&
                    candidate_evidence.route.threads == threads &&
                    candidate_evidence.route.effective_policy ==
                        candidate.route_name &&
                    candidate_evidence.route.n_block_chunks ==
                        decodeScheduleNBlockChunks(candidate.policy);
                candidate_evidence.numerical_correctness =
                    candidate_evidence.comparison.nonfinite_count == 0;
                candidate_evidence.correctness_pass =
                    candidate_evidence.route_ok &&
                    candidate_evidence.comparison.bitwiseEqual() &&
                    candidate_evidence.repeat_byte_mismatches == 0 &&
                    candidate_evidence.numerical_correctness;
                ASSERT_TRUE(candidate_evidence.correctness_pass)
                    << request.request_id << " candidate="
                    << candidate.canonical_id;
            }

            /*
             * Prime both schedules in alternating order. This starts the
             * OpenMP worker team, raises the K-partial buffer to its required
             * high-water mark, and removes first-use faults before timing.
             */
            for (int warmup = 0; warmup < warmups; ++warmup)
            {
                const size_t first = static_cast<size_t>(warmup & 1);
                for (size_t offset = 0; offset < candidates.size(); ++offset)
                {
                    const size_t candidate_index = (first + offset) % 2;
                    gemv_native_vnni_preq(
                        packed,
                        quantized.data(),
                        candidate_outputs[candidate_index].data(),
                        isa_path,
                        candidates[candidate_index]->policy);
                }
            }

            const uint64_t cell_seed = cpuPairedSeed(request.request_id);
            for (int pair_index = 0; pair_index < pair_count; ++pair_index)
            {
                const uint64_t pair_seed = cpuPairedOrderSeed(
                    cell_seed, static_cast<size_t>(pair_index));
                const size_t first = static_cast<size_t>(
                    (pair_index + (cell_seed & 1ULL)) & 1ULL);
                for (size_t offset = 0; offset < candidates.size(); ++offset)
                {
                    const size_t candidate_index = (first + offset) % 2;
                    const auto begin = std::chrono::steady_clock::now();
                    gemv_native_vnni_preq(
                        packed,
                        quantized.data(),
                        candidate_outputs[candidate_index].data(),
                        isa_path,
                        candidates[candidate_index]->policy);
                    const auto end = std::chrono::steady_clock::now();
                    samples[candidate_index][static_cast<size_t>(pair_index)] =
                        std::chrono::duration<double, std::micro>(end - begin)
                            .count();
                }

                for (size_t candidate_index = 0;
                     candidate_index < candidates.size(); ++candidate_index)
                {
                    writeCPUPairedSample(
                        output,
                        request,
                        *candidates[candidate_index],
                        evidence[candidate_index],
                        candidate_index == 0 ? "selected" : "exact",
                        static_cast<size_t>(pair_index),
                        candidate_index == first ? 0u : 1u,
                        static_cast<size_t>(pair_count),
                        cell_seed,
                        pair_seed,
                        samples[candidate_index][static_cast<size_t>(pair_index)],
                        warmups);
                }
            }
            output.flush();
            ASSERT_TRUE(output.good()) << output_path;
            ++executed_requests;
        }

        ASSERT_EQ(executed_requests, manifest.requests.size());
        PerfStatsCollector::reset();
    }

    /** Correctness and route proof for one grouped paired candidate. */
    struct CPUGroupedPairedCandidateEvidence
    {
        CPUVerifierRouteEvidence route;
        llaminar2::test::trainer::FP32Evidence comparison;
        size_t repeat_byte_mismatches = 0;
        bool route_ok = false;
        bool numerical_correctness = false;
        bool correctness_pass = false;
    };

    /**
     * @brief Serialize one grouped-verifier half-pair after its timed launch.
     *
     * The generic paired reader intentionally owns one common CSV protocol for
     * M=1 and grouped decode. The phase and M fields disambiguate the surfaces;
     * the canonical candidate ID is emitted only after the production route
     * counter proved the corresponding Pairwise or WideRows policy executed.
     */
    void writeCPUGroupedPairedSample(
        std::ostream &output,
        const native_vnni_dispatch::NativeVNNIPairedTimingRequest &request,
        const CPUGroupedDecodeScheduleCandidate &candidate,
        const CPUGroupedPairedCandidateEvidence &evidence,
        std::string_view candidate_role,
        size_t pair_index,
        size_t within_pair_order,
        size_t configured_pair_count,
        uint64_t cell_order_seed,
        uint64_t pair_order_seed,
        double latency_us,
        int warmup_count)
    {
        output << "native-vnni-paired-interleaved-v3,"
               << request.request_id << ",cpu,verifier_rows,"
               << request.source_format << ','
               << request.source_codebook << ','
               << request.execution_codebook << ','
               << request.architecture_class << ','
               << request.shape << ",eager," << request.m << ','
               << request.n << ',' << request.k << ','
               << pair_index << ',' << configured_pair_count << ','
               << within_pair_order << ',' << cell_order_seed << ','
               << pair_order_seed << ',' << candidate_role << ','
               << candidate.canonical_id << ",1,"
               << std::setprecision(17) << latency_us << ','
               << std::hexfloat << latency_us << std::defaultfloat << ','
               << warmup_count << ','
               << evidence.comparison.mismatch_count << ','
               << evidence.comparison.first_mismatch_index << ','
               << evidence.repeat_byte_mismatches << ','
               << std::setprecision(17)
               << evidence.comparison.max_abs << ','
               << evidence.comparison.relative_l2 << ','
               << evidence.comparison.cosine << ','
               << evidence.comparison.symmetric_kld << ','
               << evidence.comparison.actual_digest << ','
               << evidence.comparison.expected_digest
               << ",0,1,1," << (evidence.route_ok ? 1 : 0) << ','
               << candidate.canonical_id << ",grouped-verifier-rows,"
               << evidence.route.n_block_chunks << ','
               << evidence.route.k_tiles << ",0,"
               << "cpu.nvnni.production.serial-m1-v2,"
               << (evidence.numerical_correctness ? 1 : 0) << ','
               << (evidence.correctness_pass ? 1 : 0) << '\n';
    }

    /**
     * @brief Execute fresh-leaf paired timing for grouped CPU verification.
     *
     * Fixture construction, activation quantization, serial-row publication,
     * route telemetry, repeat snapshots, and sample storage are all prepared
     * before the interleaved timing loop. The measured interval contains one
     * production grouped launch and two steady-clock reads. Every candidate is
     * compared over all `M*N` FP32 words against independent production M=1
     * decode rows; a route normalization or one-byte mismatch aborts the cell.
     */
    void runCPUGroupedPairedConfirmation(
        const std::string &request_manifest_path,
        const std::string &output_path)
    {
        const auto manifest =
            native_vnni_dispatch::loadNativeVnniPairedRequestManifest(
                request_manifest_path);
        ASSERT_EQ(manifest.backend, "cpu");
        ASSERT_FALSE(manifest.requests.empty());
        ASSERT_FALSE(PerfStatsCollector::isEnabled())
            << "CPU grouped paired timing requires profiling to be disabled";
        ASSERT_TRUE(native_vnni_dispatch::profilerRequestId().empty())
            << "isolated Linux-perf profiling cannot run inside paired timing";

        const int warmups = std::max(
            5, getEnvInt("LLAMINAR_CPU_NVNNI_VERIFIER_WARMUP").value_or(5));
        const int pair_count = std::max(
            30, getEnvInt("LLAMINAR_CPU_NVNNI_VERIFIER_ITERS").value_or(30));
        const std::string build_isa = compiledNativeVNNIBuildISAName();
        const std::string runtime_isa = activeISANameForVerifierTrainer();
        const int threads = omp_get_max_threads();
        const std::string architecture_suffix =
            "|build=" + build_isa + "|runtime=" + runtime_isa +
            "|threads=" + std::to_string(threads);

        const std::filesystem::path output_file_path(output_path);
        if (output_file_path.has_parent_path())
            std::filesystem::create_directories(output_file_path.parent_path());
        std::ofstream output(output_file_path, std::ios::out | std::ios::trunc);
        ASSERT_TRUE(output.is_open()) << output_path;
        output
            << "protocol_version,request_id,backend,phase,source_format,"
               "source_codebook,execution_codebook,architecture_class,shape,"
               "execution_mode,m,n,k,pair_index,configured_pair_count,"
               "within_pair_order,cell_order_seed,pair_order_seed,candidate_role,"
               "candidate_id,timed_replays,latency_us,latency_us_hex,warmup_count,"
               "bit_mismatches,first_bit_mismatch,repeat_byte_mismatches,max_abs,"
               "relative_l2,cosine,symmetric_kld,grouped_output_digest,"
               "serial_output_digest,graph_capture_ok,workspace_ok,"
               "explicit_stream_ok,route_counter_ok,observed_candidate_id,"
               "observed_path,observed_tile_n,observed_cpt,observed_effective_kb,"
               "serial_m1_candidate_id,numerical_correctness,correctness_pass\n";

        size_t executed_requests = 0;
        for (const auto &request : manifest.requests)
        {
            ASSERT_EQ(request.execution_mode, "eager") << request.request_id;
            ASSERT_GT(request.m, 1) << request.request_id;
            ASSERT_GE(request.architecture_class.size(), architecture_suffix.size())
                << request.request_id;
            ASSERT_EQ(
                request.architecture_class.substr(
                    request.architecture_class.size() - architecture_suffix.size()),
                architecture_suffix)
                << request.request_id << " belongs to another CPU ISA/thread regime";
            ASSERT_EQ(request.shape.rfind("CPUGroupedDecodeAutoSeal_", 0), 0u)
                << request.request_id << " unknown grouped seal shape "
                << request.shape;
            ASSERT_GT(request.n, 0) << request.request_id;
            ASSERT_GT(request.k, 0) << request.request_id;
            ASSERT_EQ(request.k % 32, 0) << request.request_id;

            const auto format_iterator = std::find_if(
                MTP_SMALL_M_FORMATS.begin(), MTP_SMALL_M_FORMATS.end(),
                [&](const FormatSpec &format)
                {
                    return toLower(format.name) ==
                           toLower(request.source_format);
                });
            ASSERT_NE(format_iterator, MTP_SMALL_M_FORMATS.end())
                << request.request_id << " unknown format "
                << request.source_format;
            const CPUGroupedDecodeScheduleCandidate *selected =
                findCPUGroupedDecodeScheduleCandidate(
                    request.selected_candidate_id);
            const CPUGroupedDecodeScheduleCandidate *exact =
                findCPUGroupedDecodeScheduleCandidate(request.exact_candidate_id);
            ASSERT_NE(selected, nullptr) << request.request_id;
            ASSERT_NE(exact, nullptr) << request.request_id;
            const bool single_candidate_witness =
                request.reason ==
                "grouped_frozen_leaf_single_candidate_witness";
            if (single_candidate_witness)
            {
                ASSERT_EQ(selected, exact) << request.request_id;
            }
            else
            {
                ASSERT_NE(selected, exact) << request.request_id;
            }

            auto weights = createWeightsForFormat(
                format_iterator->name,
                static_cast<size_t>(request.n),
                static_cast<size_t>(request.k));
            ASSERT_NE(weights, nullptr) << request.request_id;
            CPUNativeVNNIGemmKernel kernel(weights.get());
            ASSERT_TRUE(kernel.isValid()) << request.request_id;
            const auto &packed = kernel.packedWeights();
            ASSERT_EQ(
                static_cast<int>(packed.codebook_id), request.source_codebook)
                << request.request_id;
            ASSERT_EQ(
                static_cast<int>(packed.codebook_id), request.execution_codebook)
                << request.request_id;

            std::mt19937 rng(static_cast<uint32_t>(
                0x6A0F00Du + request.m * 131u + request.n + request.k +
                format_iterator->name.size()));
            std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
            std::vector<float> input(
                static_cast<size_t>(request.m) * request.k);
            for (float &value : input)
                value = distribution(rng);
            std::vector<Q8_1Block> quantized(
                static_cast<size_t>(request.m) * packed.blocks_per_row);
            quantize_activations_to_q8_1(
                input.data(),
                quantized.data(),
                request.m,
                request.k,
                packed.blocks_per_row);

            const size_t output_elements =
                static_cast<size_t>(request.m) * request.n;
            std::vector<float> serial(output_elements, 0.0f);
            constexpr ISAPath isa_path = ISAPath::AUTO;
            for (int row = 0; row < request.m; ++row)
            {
                gemv_native_vnni_preq(
                    packed,
                    quantized.data() +
                        static_cast<size_t>(row) * packed.blocks_per_row,
                    serial.data() + static_cast<size_t>(row) * request.n,
                    isa_path);
            }

            const std::array<const CPUGroupedDecodeScheduleCandidate *, 2>
                candidates = {selected, exact};
            std::array<std::vector<float>, 2> candidate_outputs = {
                std::vector<float>(output_elements, 0.0f),
                std::vector<float>(output_elements, 0.0f),
            };
            std::array<std::vector<float>, 2> first_outputs = {
                std::vector<float>(output_elements, 0.0f),
                std::vector<float>(output_elements, 0.0f),
            };
            std::array<std::vector<double>, 2> samples = {
                std::vector<double>(static_cast<size_t>(pair_count), 0.0),
                std::vector<double>(static_cast<size_t>(pair_count), 0.0),
            };
            std::array<CPUGroupedPairedCandidateEvidence, 2> evidence;

            const auto launch = [&](size_t candidate_index)
            {
                gemm_native_vnni_preq_decode_equivalent_rows(
                    packed,
                    quantized.data(),
                    candidate_outputs[candidate_index].data(),
                    request.m,
                    request.n,
                    isa_path,
                    candidates[candidate_index]->policy);
            };
            for (size_t candidate_index = 0;
                 candidate_index < candidates.size(); ++candidate_index)
            {
                PerfStatsCollector::reset();
                ASSERT_EQ(setenv("LLAMINAR_PERF_STATS_JSON", "1", 1), 0);
                launch(candidate_index);
                evidence[candidate_index].route = findCPUVerifierRoute(
                    request.m, request.n, request.k, packed.codebook_id);
                ASSERT_EQ(unsetenv("LLAMINAR_PERF_STATS_JSON"), 0);
                ASSERT_FALSE(PerfStatsCollector::isEnabled());
                std::copy(
                    candidate_outputs[candidate_index].begin(),
                    candidate_outputs[candidate_index].end(),
                    first_outputs[candidate_index].begin());
                launch(candidate_index);

                CPUGroupedPairedCandidateEvidence &candidate_evidence =
                    evidence[candidate_index];
                candidate_evidence.repeat_byte_mismatches =
                    llaminar2::test::trainer::nativeByteMismatchCount(
                        first_outputs[candidate_index],
                        candidate_outputs[candidate_index]);
                candidate_evidence.comparison =
                    llaminar2::test::trainer::compareFP32(
                        candidate_outputs[candidate_index],
                        serial,
                        output_elements);
                candidate_evidence.route_ok =
                    candidate_evidence.route.found &&
                    candidate_evidence.route.count == 1 &&
                    candidate_evidence.route.build_isa == build_isa &&
                    candidate_evidence.route.isa == runtime_isa &&
                    candidate_evidence.route.threads == threads &&
                    candidate_evidence.route.effective_policy ==
                        candidates[candidate_index]->route_name;
                candidate_evidence.numerical_correctness =
                    candidate_evidence.comparison.nonfinite_count == 0;
                candidate_evidence.correctness_pass =
                    candidate_evidence.route_ok &&
                    candidate_evidence.comparison.bitwiseEqual() &&
                    candidate_evidence.repeat_byte_mismatches == 0 &&
                    candidate_evidence.numerical_correctness;
                ASSERT_TRUE(candidate_evidence.correctness_pass)
                    << request.request_id << " candidate="
                    << candidates[candidate_index]->canonical_id
                    << " route_found=" << candidate_evidence.route.found
                    << " route_count=" << candidate_evidence.route.count
                    << " route_build=" << candidate_evidence.route.build_isa
                    << " route_isa=" << candidate_evidence.route.isa
                    << " route_threads=" << candidate_evidence.route.threads
                    << " effective_policy="
                    << candidate_evidence.route.effective_policy
                    << " expected_policy="
                    << candidates[candidate_index]->route_name
                    << " route_ok=" << candidate_evidence.route_ok
                    << " bit_mismatches="
                    << candidate_evidence.comparison.mismatch_count
                    << " first_bit_mismatch="
                    << candidate_evidence.comparison.first_mismatch_index
                    << " repeat_byte_mismatches="
                    << candidate_evidence.repeat_byte_mismatches
                    << " nonfinite="
                    << candidate_evidence.comparison.nonfinite_count
                    << " max_abs=" << candidate_evidence.comparison.max_abs
                    << " relative_l2="
                    << candidate_evidence.comparison.relative_l2
                    << " cosine=" << candidate_evidence.comparison.cosine;
            }

            for (int warmup = 0; warmup < warmups; ++warmup)
            {
                const size_t first = static_cast<size_t>(warmup & 1);
                launch(first);
                launch(1 - first);
            }

            const uint64_t cell_seed = cpuPairedSeed(request.request_id);
            for (int pair_index = 0; pair_index < pair_count; ++pair_index)
            {
                const uint64_t pair_seed = cpuPairedOrderSeed(
                    cell_seed, static_cast<size_t>(pair_index));
                const size_t first = static_cast<size_t>(
                    (pair_index + (cell_seed & 1ULL)) & 1ULL);
                for (size_t offset = 0; offset < candidates.size(); ++offset)
                {
                    const size_t candidate_index = (first + offset) % 2;
                    const auto begin = std::chrono::steady_clock::now();
                    launch(candidate_index);
                    const auto end = std::chrono::steady_clock::now();
                    samples[candidate_index][static_cast<size_t>(pair_index)] =
                        std::chrono::duration<double, std::micro>(end - begin)
                            .count();
                }
                for (size_t candidate_index = 0;
                     candidate_index < candidates.size(); ++candidate_index)
                {
                    writeCPUGroupedPairedSample(
                        output,
                        request,
                        *candidates[candidate_index],
                        evidence[candidate_index],
                        candidate_index == 0 ? "selected" : "exact",
                        static_cast<size_t>(pair_index),
                        candidate_index == first ? 0u : 1u,
                        static_cast<size_t>(pair_count),
                        cell_seed,
                        pair_seed,
                        samples[candidate_index]
                               [static_cast<size_t>(pair_index)],
                        warmups);
                }
            }
            output.flush();
            ASSERT_TRUE(output.good()) << output_path;
            ++executed_requests;
        }

        ASSERT_EQ(executed_requests, manifest.requests.size());
        PerfStatsCollector::reset();
    }

    /**
     * @test Emit promotion-grade CPU M=1 decode schedule evidence.
     *
     * Every candidate enters the real pre-quantized production GEMV launcher.
     * Candidates may change only OpenMP ownership of adjacent 64-column chunks;
     * the frozen serial K partition and ascending reduction tree remain fixed.
     * Each forceable route must therefore repeat byte-identically and match the
     * explicit frozen-serial oracle for the complete FP32 output.
     *
     * Canonical timing and Linux-perf evidence are deliberately separate. A
     * profiler process prepares the fixture, enables counters around exactly
     * one target launch, and verifies that launch's production route. It does
     * not replay warmups, correctness checks, or timing samples.
     */
    TEST_F(CPUNativeVNNIGemvTest, TrainerCsv_StrongDecode_AllFormats)
    {
        applyVerifierRowsThreadCapForStandalonePerf();

        const std::string paired_request_manifest_path = getEnvString(
            "LLAMINAR_CPU_NVNNI_DECODE_PAIRED_REQUEST_MANIFEST");
        const std::string paired_output_path = getEnvString(
            "LLAMINAR_CPU_NVNNI_DECODE_PAIRED_CSV");
        ASSERT_EQ(
            paired_request_manifest_path.empty(), paired_output_path.empty())
            << "CPU paired confirmation requires both the request manifest and CSV";
        if (!paired_request_manifest_path.empty())
        {
            ASSERT_TRUE(getEnvString("LLAMINAR_CPU_NVNNI_DECODE_FORMATS").empty());
            ASSERT_TRUE(getEnvString("LLAMINAR_CPU_NVNNI_DECODE_CANDIDATES").empty());
            ASSERT_TRUE(getEnvString("LLAMINAR_CPU_NVNNI_DECODE_M").empty());
            ASSERT_TRUE(getEnvString("LLAMINAR_CPU_NVNNI_DECODE_SHAPE_NAME").empty());
            ASSERT_TRUE(getEnvString("LLAMINAR_CPU_NVNNI_DECODE_STRONG_CSV").empty());
            ASSERT_TRUE(getEnvString("LLAMINAR_CPU_NVNNI_DECODE_TIMING_CSV").empty());
            runCPUM1PairedConfirmation(
                paired_request_manifest_path, paired_output_path);
            return;
        }

        const int N = getEnvInt("LLAMINAR_CPU_NVNNI_DECODE_N").value_or(5120);
        const int K = getEnvInt("LLAMINAR_CPU_NVNNI_DECODE_K").value_or(5120);
        ASSERT_GT(N, 0);
        ASSERT_GT(K, 0);
        ASSERT_EQ(K % 32, 0);

        std::string shape_name =
            getEnvString("LLAMINAR_CPU_NVNNI_DECODE_SHAPE_NAME");
        if (shape_name.empty())
            shape_name = "Qwen36Decode";
        std::set<std::string> format_filters =
            getEnvCsvSet("LLAMINAR_CPU_NVNNI_DECODE_FORMATS");
        if (format_filters.empty())
            format_filters = {toLower("Q4_K")};
        if (format_filters.count("all") != 0)
            format_filters.clear();
        const std::set<std::string> candidate_filters =
            getEnvCsvSet("LLAMINAR_CPU_NVNNI_DECODE_CANDIDATES");
        const std::set<std::string> m_filters =
            getEnvCsvSet("LLAMINAR_CPU_NVNNI_DECODE_M");
        for (const std::string &raw_m : m_filters)
            ASSERT_EQ(raw_m, "1") << "CPU decode trainer supports only M=1";

        const std::string profiler_request_id =
            native_vnni_dispatch::profilerRequestId();
        CPUProfilerBatch profiler_batch;
        ASSERT_FALSE(
            !profiler_request_id.empty() && profiler_batch.enabled())
            << "single-request and batch CPU profiling are mutually exclusive";
        const bool profiling_requested =
            !profiler_request_id.empty() || profiler_batch.enabled();
        const std::string single_perf_output_path =
            native_vnni_dispatch::profilerEnvironment(
                native_vnni_dispatch::kPerfStatsPathEnvironment);
        std::optional<native_vnni_dispatch::LinuxPerfControl> perf_control;
        if (!profiler_request_id.empty())
        {
            perf_control.emplace(profiler_request_id);
        }
        else if (profiler_batch.enabled())
        {
            const CPUProfilerBatchRequest &first = profiler_batch.first();
            perf_control.emplace(first.request_id, first.output_path);
        }

        const int warmups = std::max(
            0,
            getEnvInt("LLAMINAR_CPU_NVNNI_DECODE_WARMUP").value_or(5));
        const int samples = std::max(
            1,
            getEnvInt("LLAMINAR_CPU_NVNNI_DECODE_ITERS").value_or(30));
        const int max_cases = std::max(
            1,
            getEnvInt("LLAMINAR_CPU_NVNNI_DECODE_MAX_CASES")
                .value_or(1000000));
        if (!profiler_request_id.empty())
        {
            ASSERT_EQ(format_filters.size(), 1u);
            ASSERT_EQ(candidate_filters.size(), 1u);
            ASSERT_EQ(max_cases, 1);
        }

        constexpr ISAPath isa_path = ISAPath::AUTO;
        const std::string build_isa = compiledNativeVNNIBuildISAName();
        const std::string requested_runtime_isa =
            requestedISANameForVerifierTrainer();
        const std::string effective_runtime_isa =
            activeISANameForVerifierTrainer();
        ASSERT_TRUE(
            effective_runtime_isa == "AVX2" ||
            effective_runtime_isa == "AVX512");
        if (requested_runtime_isa != "AUTO")
            ASSERT_EQ(requested_runtime_isa, effective_runtime_isa);

        const std::string csv_path =
            getEnvString("LLAMINAR_CPU_NVNNI_DECODE_STRONG_CSV");
        const std::string timing_csv_path =
            getEnvString("LLAMINAR_CPU_NVNNI_DECODE_TIMING_CSV");
        (void)setenv("LLAMINAR_PERF_STATS_JSON", "1", 1);
        PerfStatsCollector::reset();
        std::FILE *csv = nullptr;
        if (!csv_path.empty())
        {
            csv = std::fopen(csv_path.c_str(), "w");
            ASSERT_NE(csv, nullptr);
            std::fprintf(
                csv,
                "backend,phase,source_format,source_codebook,execution_codebook,"
                "shape,execution_mode,m,n,k,candidate_id,build_isa,"
                "runtime_isa_requested,runtime_isa_effective,threads,weight_bytes,"
                "warmup_count,sample_count,min_us,median_us,p95_us,mad_us,cv,"
                "bit_mismatches,first_bit_mismatch,repeat_byte_mismatches,max_abs,"
                "relative_l2,cosine,symmetric_kld,output_digest,oracle_output_digest,"
                "timing_sample_digest,route_counter_ok,observed_candidate_id,k_tiles,"
                "n_block_chunks,serial_kpart,numerical_correctness,correctness_pass,"
                "is_winner\n");
        }
        std::FILE *timing_csv = nullptr;
        if (!timing_csv_path.empty())
        {
            timing_csv = std::fopen(timing_csv_path.c_str(), "w");
            ASSERT_NE(timing_csv, nullptr);
            std::fprintf(
                timing_csv,
                "backend,phase,source_format,source_codebook,execution_codebook,"
                "shape,execution_mode,m,n,k,candidate_id,build_isa,"
                "runtime_isa_requested,runtime_isa_effective,sample_index,"
                "latency_us,latency_us_hex\n");
        }

        struct CandidateRow
        {
            CPUDecodeScheduleCandidate candidate{};
            StrongTimingMeasurement timing;
            llaminar2::test::trainer::FP32Evidence comparison;
            CPUDecodeRouteEvidence route;
            size_t repeat_byte_mismatches = 0;
            int timing_warmups = 0;
            bool route_ok = false;
            bool numerical_correctness = false;
            bool correctness_pass = false;
        };

        /**
         * @brief One format-local weight fixture prepared outside timing.
         *
         * The individual TestTensorFactory generators deliberately retain
         * their historical format-specific seed and serial PRNG stream.  They
         * are independent of one another, however, so preparing selected
         * formats concurrently preserves every generated byte while avoiding
         * a long single-core setup phase on large vocabulary projections.
         * Isolated profiler runs instead construct one layout-identical
         * prepared fixture at a time because their canonical timing transaction
         * already performed source conversion and correctness. Packed-kernel
         * construction and all measured launches remain sequential across
         * formats so their OpenMP teams never contend or retain several large
         * interleaved buffers concurrently.
         */
        struct PreparedDecodeWeights
        {
            const FormatSpec *format = nullptr;
            std::unique_ptr<TensorBase> weights;
        };

        std::vector<const FormatSpec *> selected_formats;
        selected_formats.reserve(MTP_SMALL_M_FORMATS.size());
        for (const auto &format : MTP_SMALL_M_FORMATS)
        {
            if (!shouldRunName(format_filters, format.name))
                continue;
            if (static_cast<int>(selected_formats.size()) >= max_cases)
                break;
            if (profiler_batch.enabled() && !profiler_batch.containsCell(
                    "NativeVNNIFastM1Projection",
                    format.name,
                    "eager",
                    1,
                    N,
                    K))
            {
                continue;
            }
            selected_formats.push_back(&format);
        }
        ASSERT_FALSE(selected_formats.empty())
            << "No CPU M=1 decode formats selected";

        std::vector<PreparedDecodeWeights> prepared_weights(
            selected_formats.size());
#pragma omp parallel for schedule(dynamic, 1)
        for (int index = 0;
             index < static_cast<int>(selected_formats.size());
             ++index)
        {
            const FormatSpec *format =
                selected_formats[static_cast<size_t>(index)];
            PreparedDecodeWeights &prepared =
                prepared_weights[static_cast<size_t>(index)];
            prepared.format = format;
            if (!profiling_requested)
            {
                prepared.weights = createWeightsForFormat(
                    format->name,
                    static_cast<size_t>(N),
                    static_cast<size_t>(K));
            }
        }

        int executed_cases = 0;
        int emitted_rows = 0;
        int isolated_profile_launches = 0;
        for (auto &prepared : prepared_weights)
        {
            ASSERT_NE(prepared.format, nullptr);
            const FormatSpec &format = *prepared.format;
            std::unique_ptr<CPUNativeVNNIGemmKernel> kernel;
            if (profiling_requested && useSyntheticProfilerPreparedWeights())
            {
                kernel = std::make_unique<CPUNativeVNNIGemmKernel>(
                    createProfilerPackedWeightsForFormat(format, N, K));
            }
            else
            {
                ASSERT_NE(prepared.weights, nullptr) << format.name;
                kernel = std::make_unique<CPUNativeVNNIGemmKernel>(
                    prepared.weights.get());
            }
            ASSERT_TRUE(kernel->isValid()) << format.name;
            const auto &packed = kernel->packedWeights();
            const size_t weight_bytes =
                packed.native_interleaved.size() + packed.payload.size();

            std::mt19937 rng(static_cast<uint32_t>(
                0xDEC0DEu + N + K + format.name.size()));
            std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
            std::vector<float> input(static_cast<size_t>(K));
            for (float &value : input)
                value = distribution(rng);
            std::vector<Q8_1Block> quantized(
                static_cast<size_t>(packed.blocks_per_row));
            quantize_activations_to_q8_1(
                input.data(),
                quantized.data(),
                1,
                K,
                packed.blocks_per_row);

            if (profiling_requested)
            {
                /*
                 * The immutable timing observation named by each request has
                 * already proved byte equality, repeatability, fused-bundle
                 * routing, and latency. Replaying those checks here used to
                 * execute the oracle, two ordinary candidates, four fused
                 * projections, warmups, and another timing series before one
                 * counter-owned launch. Apart from dominating collection time,
                 * that work changed the cache state immediately before perf.
                 *
                 * Isolated profiling owns one production M=1 invocation only.
                 * Its route counter is reset immediately before the interval
                 * and must report exactly the requested physical schedule and
                 * ISA once. This keeps profiler features attached to the exact
                 * candidate launch whose authenticated timing they augment.
                 */
                bool profiled_format = false;
                for (const CPUDecodeScheduleCandidate &candidate :
                     CPU_DECODE_SCHEDULE_CANDIDATES)
                {
                    if (!candidate_filters.empty() &&
                        !shouldRunName(
                            candidate_filters, candidate.route_name) &&
                        !shouldRunName(
                            candidate_filters, candidate.canonical_id))
                    {
                        continue;
                    }
                    CPUProfilerBatchRequest *batch_request = nullptr;
                    if (profiler_batch.enabled())
                    {
                        batch_request = profiler_batch.claim(
                            "NativeVNNIFastM1Projection",
                            format.name,
                            "eager",
                            1,
                            N,
                            K,
                            candidate.canonical_id);
                        if (!batch_request)
                            continue;
                    }

                    std::vector<float> output(static_cast<size_t>(N), 0.0f);
                    const auto run_candidate = [&]()
                    {
                        gemv_native_vnni_preq(
                            packed,
                            quantized.data(),
                            output.data(),
                            isa_path,
                            candidate.policy);
                    };
                    ASSERT_TRUE(perf_control.has_value());
                    const std::string &exact_request_id = batch_request
                        ? batch_request->request_id
                        : profiler_request_id;
                    const std::string &exact_output_path = batch_request
                        ? batch_request->output_path
                        : single_perf_output_path;

                    PerfStatsCollector::reset();
                    profileExactCPURequest(
                        *perf_control,
                        exact_request_id,
                        exact_output_path,
                        run_candidate);
                    const CPUDecodeRouteEvidence route =
                        findCPUDecodeRoute(N, K, packed.codebook_id);
                    ASSERT_TRUE(route.found);
                    ASSERT_EQ(route.count, 1u);
                    ASSERT_EQ(route.build_isa, build_isa);
                    ASSERT_EQ(route.isa, effective_runtime_isa);
                    ASSERT_EQ(route.effective_policy, candidate.route_name);

                    ++isolated_profile_launches;
                    profiled_format = true;
                    std::fprintf(
                        stderr,
                        "[NativeVNNIProfiler][CPU_DECODE] request=%s "
                        "candidate=%s format=%s M=1 N=%d K=%d launches=1\n",
                        exact_request_id.c_str(),
                        candidate.canonical_id,
                        format.name.c_str(),
                        N,
                        K);
                }
                if (profiled_format)
                    ++executed_cases;
                continue;
            }

            std::vector<float> oracle(static_cast<size_t>(N), 0.0f);
            gemv_native_vnni_preq(
                packed,
                quantized.data(),
                oracle.data(),
                isa_path,
                DecodeSchedulePolicy::FrozenSerialOracle);
            const std::string oracle_digest =
                llaminar2::test::trainer::nativeByteDigest(oracle);

            std::vector<CandidateRow> rows;
            rows.reserve(CPU_DECODE_SCHEDULE_CANDIDATES.size());
            for (const CPUDecodeScheduleCandidate &candidate :
                 CPU_DECODE_SCHEDULE_CANDIDATES)
            {
                if (!candidate_filters.empty() &&
                    !shouldRunName(candidate_filters, candidate.route_name) &&
                    !shouldRunName(candidate_filters, candidate.canonical_id))
                {
                    continue;
                }
                std::vector<float> output(static_cast<size_t>(N), 0.0f);
                const auto run_candidate = [&]()
                {
                    gemv_native_vnni_preq(
                        packed,
                        quantized.data(),
                        output.data(),
                        isa_path,
                        candidate.policy);
                };
                PerfStatsCollector::reset();
                run_candidate();
                const CPUDecodeRouteEvidence route =
                    findCPUDecodeRoute(N, K, packed.codebook_id);
                const std::vector<float> first_output = output;
                run_candidate();
                const size_t repeat_byte_mismatches =
                    llaminar2::test::trainer::nativeByteMismatchCount(
                        first_output, output);
                const auto comparison =
                    llaminar2::test::trainer::compareFP32(
                        output, oracle, static_cast<size_t>(N));
                /*
                 * The learned M=1 schedule is shared by three production
                 * bundles: an ordinary projection, gate/up-style projections
                 * sharing one activation, and expert-down projections with
                 * projection-local activations. Exercise all three for every
                 * format/candidate cell. Otherwise a correct ordinary GEMV
                 * could conceal a fused launcher that ignored runtime ISA,
                 * K-partition, or the selected N-block width.
                 */
                std::array<std::vector<float>, 2> shared_outputs = {
                    std::vector<float>(static_cast<size_t>(N), 0.0f),
                    std::vector<float>(static_cast<size_t>(N), 0.0f),
                };
                std::array<FusedGemvDesc, 2> shared_descriptors = {{
                    {&packed, shared_outputs[0].data(), nullptr, N},
                    {&packed, shared_outputs[1].data(), nullptr, N},
                }};
                gemv_native_vnni_fused_preq(
                    quantized.data(),
                    shared_descriptors.data(),
                    static_cast<int>(shared_descriptors.size()),
                    isa_path,
                    candidate.policy);

                std::array<std::vector<float>, 2> multi_input_outputs = {
                    std::vector<float>(static_cast<size_t>(N), 0.0f),
                    std::vector<float>(static_cast<size_t>(N), 0.0f),
                };
                std::array<FusedGemvMultiInputDesc, 2>
                    multi_input_descriptors = {{
                        {
                            quantized.data(),
                            &packed,
                            multi_input_outputs[0].data(),
                            N,
                        },
                        {
                            quantized.data(),
                            &packed,
                            multi_input_outputs[1].data(),
                            N,
                        },
                    }};
                gemv_fused_multi_input_preq(
                    multi_input_descriptors.data(),
                    static_cast<int>(multi_input_descriptors.size()),
                    isa_path,
                    candidate.policy);

                bool fused_outputs_equal = true;
                for (const auto &fused_output : shared_outputs)
                {
                    fused_outputs_equal =
                        fused_outputs_equal &&
                        llaminar2::test::trainer::nativeByteMismatchCount(
                            fused_output, oracle) == 0;
                }
                for (const auto &fused_output : multi_input_outputs)
                {
                    fused_outputs_equal =
                        fused_outputs_equal &&
                        llaminar2::test::trainer::nativeByteMismatchCount(
                            fused_output, oracle) == 0;
                }

                uint64_t fused_route_count = 0;
                bool fused_route_ok = true;
                for (const auto &record : PerfStatsCollector::snapshot(
                         {"kernel.cpu_native_vnni_fused_verifier_rows_projection_launch"}))
                {
                    if (record.name !=
                        "cpu_native_vnni_fused_verifier_rows_projection_launch")
                    {
                        continue;
                    }
                    fused_route_ok =
                        fused_route_ok && record.tags.at("m") == "1" &&
                        record.tags.at("n") == std::to_string(N) &&
                        record.tags.at("k") == std::to_string(K) &&
                        record.tags.at("codebook") ==
                            std::to_string(packed.codebook_id) &&
                        record.tags.at("isa") == effective_runtime_isa &&
                        record.tags.at("effective_decode_policy") ==
                            route.effective_policy &&
                        record.tags.at("n_block_chunks") ==
                            std::to_string(route.n_block_chunks) &&
                        record.tags.at("route") ==
                            (route.k_tiles > 1
                                 ? "decode_k_parallel_row"
                                 : "decode_full_k_row");
                    fused_route_count += record.count;
                }
                fused_route_ok =
                    fused_route_ok && fused_route_count == 4;
                const bool route_ok =
                    route.found && route.count > 0 &&
                    route.build_isa == build_isa &&
                    route.isa == effective_runtime_isa &&
                    route.effective_policy == candidate.route_name &&
                    fused_route_ok;
                const int candidate_warmups = route_ok ? warmups : 0;
                const int candidate_samples = route_ok ? samples : 1;
                const StrongTimingMeasurement timing =
                    timeStrongTrainerCandidate(
                        candidate_warmups, candidate_samples, run_candidate);
                const bool numerical_correctness =
                    comparison.nonfinite_count == 0;
                const bool correctness_pass =
                    route_ok && comparison.bitwiseEqual() &&
                    repeat_byte_mismatches == 0 && fused_outputs_equal &&
                    numerical_correctness;
                if (route_ok)
                {
                    EXPECT_TRUE(comparison.bitwiseEqual())
                        << format.name << " candidate=" << candidate.canonical_id;
                    EXPECT_EQ(repeat_byte_mismatches, 0u)
                        << format.name << " candidate=" << candidate.canonical_id;
                }
                EXPECT_TRUE(fused_route_ok)
                    << format.name << " candidate=" << candidate.canonical_id
                    << " did not execute the requested fused M=1 schedule";
                EXPECT_TRUE(fused_outputs_equal)
                    << format.name << " candidate=" << candidate.canonical_id
                    << " fused production bundle differs from serial decode";
                rows.push_back(CandidateRow{
                    candidate,
                    timing,
                    comparison,
                    route,
                    repeat_byte_mismatches,
                    candidate_warmups,
                    route_ok,
                    numerical_correctness,
                    correctness_pass});
            }

            int best_index = -1;
            for (size_t index = 0; index < rows.size(); ++index)
            {
                if (!rows[index].correctness_pass)
                    continue;
                if (best_index < 0 ||
                    rows[index].timing.evidence.median <
                        rows[static_cast<size_t>(best_index)]
                            .timing.evidence.median)
                {
                    best_index = static_cast<int>(index);
                }
            }
            ASSERT_GE(best_index, 0)
                << format.name << " has no forceable byte-exact M=1 candidate";

            for (size_t index = 0; index < rows.size(); ++index)
            {
                const CandidateRow &row = rows[index];
                if (csv)
                {
                    std::fprintf(
                        csv,
                        "cpu,decode_m1,%s,%u,%u,%s,eager,1,%d,%d,%s,%s,%s,%s,%d,%zu,"
                        "%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%zu,%zu,%zu,%.9g,%.9g,%.9g,"
                        "%.9g,%s,%s,%s,%d,%s,%d,%d,%d,%d,%d,%d\n",
                        format.name.c_str(),
                        static_cast<unsigned>(packed.codebook_id),
                        static_cast<unsigned>(packed.codebook_id),
                        shape_name.c_str(),
                        N,
                        K,
                        row.candidate.canonical_id,
                        row.route.build_isa.c_str(),
                        requested_runtime_isa.c_str(),
                        row.route.isa.c_str(),
                        row.route.threads,
                        weight_bytes,
                        row.timing_warmups,
                        static_cast<int>(row.timing.samples_us.size()),
                        row.timing.evidence.min,
                        row.timing.evidence.median,
                        row.timing.evidence.p95,
                        row.timing.evidence.mad,
                        row.timing.evidence.cv,
                        row.comparison.mismatch_count,
                        row.comparison.first_mismatch_index,
                        row.repeat_byte_mismatches,
                        row.comparison.max_abs,
                        row.comparison.relative_l2,
                        row.comparison.cosine,
                        row.comparison.symmetric_kld,
                        row.comparison.actual_digest.c_str(),
                        oracle_digest.c_str(),
                        row.timing.evidence.digest.c_str(),
                        row.route_ok ? 1 : 0,
                        row.route.effective_policy.c_str(),
                        row.route.k_tiles,
                        row.route.n_block_chunks,
                        row.route.serial_kpart ? 1 : 0,
                        row.numerical_correctness ? 1 : 0,
                        row.correctness_pass ? 1 : 0,
                        static_cast<int>(index) == best_index ? 1 : 0);
                    ++emitted_rows;
                }
                if (timing_csv)
                {
                    for (size_t sample_index = 0;
                         sample_index < row.timing.samples_us.size();
                         ++sample_index)
                    {
                        const double latency_us =
                            row.timing.samples_us[sample_index];
                        std::fprintf(
                            timing_csv,
                            "cpu,decode_m1,%s,%u,%u,%s,eager,1,%d,%d,%s,%s,%s,%s,%zu,%.9f,%a\n",
                            format.name.c_str(),
                            static_cast<unsigned>(packed.codebook_id),
                            static_cast<unsigned>(packed.codebook_id),
                            shape_name.c_str(),
                            N,
                            K,
                            row.candidate.canonical_id,
                            row.route.build_isa.c_str(),
                            requested_runtime_isa.c_str(),
                            row.route.isa.c_str(),
                            sample_index,
                            latency_us,
                            latency_us);
                    }
                }
            }
            if (csv)
                std::fflush(csv);
            if (timing_csv)
                std::fflush(timing_csv);
            ++executed_cases;
        }

        if (csv)
        {
            ASSERT_EQ(std::fclose(csv), 0);
            if (!profiling_requested)
                ASSERT_GT(emitted_rows, 0);
        }
        if (timing_csv)
            ASSERT_EQ(std::fclose(timing_csv), 0);
        ASSERT_GT(executed_cases, 0) << "No CPU M=1 decode cases selected";
        if (!profiler_request_id.empty())
            ASSERT_EQ(isolated_profile_launches, 1);
        else if (profiler_batch.enabled())
        {
            EXPECT_NO_THROW(profiler_batch.requireComplete());
            ASSERT_EQ(
                static_cast<size_t>(isolated_profile_launches),
                profiler_batch.size());
        }
        PerfStatsCollector::reset();
    }

    /**
     * @test Prove the installed grouped-verifier policy through production Auto.
     *
     * Candidate trainers deliberately force Pairwise and WideRows so both
     * physical implementations receive independent correctness and timing
     * evidence. That is necessary for fitting, but it does not prove that the
     * generated table was installed correctly or that the ordinary inference
     * call resolves it. This smoke test closes that final publication gap.
     *
     * Every NativeVNNI source format is prepared at the Qwen 3.6 MoE GDN
     * Z-projection geometry. The complete grouped MTP depth inventory then
     * enters `gemm_native_vnni_preq_decode_equivalent_rows()` with its default
     * `VerifierRowsPolicy::Auto`. Route telemetry must identify Auto as the
     * requested policy and the generated winner as the effective physical
     * policy. Every output word must match an independent production M=1 row,
     * and a second launch must reproduce the same bytes.
     *
     * Run this binary once for each certified build/runtime surface:
     * AVX2/AVX2, AVX512/AVX2, and AVX512/AVX512. The generated policy is keyed
     * to one physical socket, so this standalone test fixes the OpenMP team at
     * the 28 physical cores used by certification.
     */
    TEST_F(CPUNativeVNNIGemvTest,
           ProductionAutoVerifierRowsPolicyAllFormatsIsSerialRowExact)
    {
        constexpr int N = 4096;
        constexpr int K = 2048;
        constexpr int certified_threads = 28;
        constexpr std::array<int, 5> verifier_rows = {2, 4, 8, 16, 31};
        constexpr int maximum_m = verifier_rows.back();

        omp_set_num_threads(certified_threads);
        ASSERT_EQ(omp_get_max_threads(), certified_threads);
        ASSERT_EQ(setenv("LLAMINAR_PERF_STATS_JSON", "1", 1), 0);

        const std::string expected_build_isa =
            compiledNativeVNNIBuildISAName();
        const std::string expected_runtime_isa =
            activeISANameForVerifierTrainer();
        ASSERT_TRUE(
            expected_runtime_isa == "AVX2" ||
            expected_runtime_isa == "AVX512");

        for (size_t format_index = 0;
             format_index < MTP_SMALL_M_FORMATS.size();
             ++format_index)
        {
            const FormatSpec &format = MTP_SMALL_M_FORMATS[format_index];
            SCOPED_TRACE(format.name);

            auto weights = createWeightsForFormat(format.name, N, K);
            ASSERT_NE(weights, nullptr);
            CPUNativeVNNIGemmKernel kernel(weights.get());
            ASSERT_TRUE(kernel.isValid());
            const auto &packed = kernel.packedWeights();
            const NativeVNNITileConfig expected_serial_geometry =
                computeTileConfig(
                    N,
                    K,
                    1,
                    packed.payload_bytes,
                    certified_threads);

            std::mt19937 rng(static_cast<uint32_t>(
                0xA170000u + format_index * 977u));
            std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
            std::vector<float> input(
                static_cast<size_t>(maximum_m) * K);
            for (float &value : input)
                value = distribution(rng);

            std::vector<Q8_1Block> quantized_rows(
                static_cast<size_t>(maximum_m) * packed.blocks_per_row);
            quantize_activations_to_q8_1(
                input.data(),
                quantized_rows.data(),
                maximum_m,
                K,
                packed.blocks_per_row);

            std::vector<float> serial(
                static_cast<size_t>(maximum_m) * N,
                0.0f);
            for (int row = 0; row < maximum_m; ++row)
            {
                gemv_native_vnni_preq(
                    packed,
                    quantized_rows.data() +
                        static_cast<size_t>(row) * packed.blocks_per_row,
                    serial.data() + static_cast<size_t>(row) * N,
                    ISAPath::AUTO);
            }

            for (int M : verifier_rows)
            {
                SCOPED_TRACE(std::string("M=") + std::to_string(M));
                std::vector<float> grouped(
                    static_cast<size_t>(M) * N,
                    0.0f);

                const VerifierRowsPolicy selected_policy =
                    selectVerifierRowsPolicy(packed, M, N, K);
                const bool selected_wide_rows =
                    selected_policy == VerifierRowsPolicy::WideRows;
                const std::string expected_effective_policy =
                    selected_wide_rows &&
                            expected_runtime_isa == "AVX512" && M >= 3
                        ? "WideRows"
                        : "Pairwise";

                PerfStatsCollector::reset();
                gemm_native_vnni_preq_decode_equivalent_rows(
                    packed,
                    quantized_rows.data(),
                    grouped.data(),
                    M,
                    N,
                    ISAPath::AUTO,
                    VerifierRowsPolicy::Auto);
                const CPUVerifierRouteEvidence route =
                    findCPUVerifierRoute(M, N, K, packed.codebook_id);

                ASSERT_TRUE(route.found);
                EXPECT_EQ(route.count, 1u);
                EXPECT_EQ(route.requested_policy, "Auto");
                EXPECT_EQ(route.effective_policy, expected_effective_policy);
                EXPECT_EQ(route.build_isa, expected_build_isa);
                EXPECT_EQ(route.isa, expected_runtime_isa);
                EXPECT_EQ(route.threads, certified_threads);
                EXPECT_EQ(route.k_tiles, expected_serial_geometry.k_tiles);
                EXPECT_EQ(
                    route.n_block_chunks,
                    expected_serial_geometry.n_block_chunks);

                const size_t output_elements =
                    static_cast<size_t>(M) * N;
                EXPECT_EQ(
                    llaminar2::test::trainer::nativeByteMismatchCount(
                        grouped.data(),
                        output_elements,
                        serial.data(),
                        output_elements),
                    0u)
                    << "Production Auto differs from serial M=1 decode";

                const std::vector<float> first_grouped = grouped;
                gemm_native_vnni_preq_decode_equivalent_rows(
                    packed,
                    quantized_rows.data(),
                    grouped.data(),
                    M,
                    N,
                    ISAPath::AUTO,
                    VerifierRowsPolicy::Auto);
                EXPECT_EQ(
                    llaminar2::test::trainer::nativeByteMismatchCount(
                        grouped.data(),
                        output_elements,
                        first_grouped.data(),
                        output_elements),
                    0u)
                    << "Production Auto changed bytes across repeated launches";
            }
        }

        PerfStatsCollector::reset();
        ASSERT_EQ(unsetenv("LLAMINAR_PERF_STATS_JSON"), 0);
    }

    /**
     * @test Prove every prefill-style full-K grouped schedule for all formats.
     *
     * The ordinary prefill tournament first establishes whether reusing packed
     * weight decode across verifier rows is worthwhile. Admission into grouped
     * production requires a stronger test through the verifier entry point
     * itself. This case forces every winning physical family at each supported
     * speculative depth, verifies its route identity, compares the complete
     * M-by-N publication with independent production M=1 rows, and repeats the
     * grouped launch to detect work-sharing races.
     *
     * N=4096, K=2048 is the Qwen 3.6 MoE GDN Z projection that motivated this
     * optimization. Serial M=1 uses one K tile for every registry format at
     * this geometry, so every candidate below is a genuine full-K schedule.
     */
    TEST_F(CPUNativeVNNIGemvTest,
           FullKGroupedSchedulesAllFormatsAreSerialRowExact)
    {
        constexpr int N = 4096;
        constexpr int K = 2048;
        constexpr int certified_threads = 28;
        constexpr std::array<int, 5> verifier_rows = {2, 4, 8, 16, 31};
        constexpr int maximum_m = verifier_rows.back();
        constexpr std::array<VerifierRowsPolicy, 7> candidates = {{
            VerifierRowsPolicy::FullKRowChunkGrid,
            VerifierRowsPolicy::FullKTwoRowNbc1,
            VerifierRowsPolicy::FullKTwoRowNbc2,
            VerifierRowsPolicy::FullKTwoRowPairGridNbc1,
            VerifierRowsPolicy::FullKTwoRowPairGridNbc2,
            VerifierRowsPolicy::FullKTwoRowPairGridNbc4,
            VerifierRowsPolicy::FullKTwoRowPairGridNbc8,
        }};

        omp_set_num_threads(certified_threads);
        ASSERT_EQ(setenv("LLAMINAR_PERF_STATS_JSON", "1", 1), 0);
        const std::string expected_build_isa =
            compiledNativeVNNIBuildISAName();
        const std::string expected_runtime_isa =
            activeISANameForVerifierTrainer();

        for (size_t format_index = 0;
             format_index < MTP_SMALL_M_FORMATS.size();
             ++format_index)
        {
            const FormatSpec &format = MTP_SMALL_M_FORMATS[format_index];
            SCOPED_TRACE(format.name);
            auto weights = createWeightsForFormat(format.name, N, K);
            ASSERT_NE(weights, nullptr);
            CPUNativeVNNIGemmKernel kernel(weights.get());
            ASSERT_TRUE(kernel.isValid());
            const auto &packed = kernel.packedWeights();
            const NativeVNNITileConfig serial_geometry = computeTileConfig(
                N, K, 1, packed.payload_bytes, certified_threads);
            ASSERT_LE(serial_geometry.k_tiles, 1)
                << "Focused full-K grouped geometry unexpectedly selected K-partition";

            std::mt19937 rng(static_cast<uint32_t>(
                0xF011000u + format_index * 977u));
            std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
            std::vector<float> input(
                static_cast<size_t>(maximum_m) * K);
            for (float &value : input)
                value = distribution(rng);
            std::vector<Q8_1Block> quantized_rows(
                static_cast<size_t>(maximum_m) * packed.blocks_per_row);
            quantize_activations_to_q8_1(
                input.data(),
                quantized_rows.data(),
                maximum_m,
                K,
                packed.blocks_per_row);

            std::vector<float> serial(
                static_cast<size_t>(maximum_m) * N,
                0.0f);
            for (int row = 0; row < maximum_m; ++row)
            {
                gemv_native_vnni_preq(
                    packed,
                    quantized_rows.data() +
                        static_cast<size_t>(row) * packed.blocks_per_row,
                    serial.data() + static_cast<size_t>(row) * N,
                    ISAPath::AUTO);
            }

            for (int M : verifier_rows)
            {
                for (VerifierRowsPolicy candidate : candidates)
                {
                    SCOPED_TRACE(
                        std::string("M=") + std::to_string(M) +
                        " candidate=" + verifierRowsPolicyName(candidate));
                    std::vector<float> grouped(
                        static_cast<size_t>(M) * N,
                        0.0f);

                    PerfStatsCollector::reset();
                    gemm_native_vnni_preq_decode_equivalent_rows(
                        packed,
                        quantized_rows.data(),
                        grouped.data(),
                        M,
                        N,
                        ISAPath::AUTO,
                        candidate);
                    const CPUVerifierRouteEvidence route = findCPUVerifierRoute(
                        M, N, K, packed.codebook_id);
                    const VerifierRowsPolicy expected_policy =
                        normalizeVerifierRowsPolicy(
                            candidate,
                            (N + 63) / 64,
                            expected_runtime_isa == "AVX512",
                            M);
                    ASSERT_TRUE(route.found);
                    EXPECT_EQ(route.count, 1u);
                    EXPECT_EQ(route.requested_policy, verifierRowsPolicyName(candidate));
                    EXPECT_EQ(
                        route.effective_policy,
                        verifierRowsPolicyName(expected_policy));
                    EXPECT_EQ(route.build_isa, expected_build_isa);
                    EXPECT_EQ(route.isa, expected_runtime_isa);
                    EXPECT_EQ(route.threads, certified_threads);
                    EXPECT_EQ(route.k_tiles, serial_geometry.k_tiles);
                    EXPECT_EQ(
                        route.n_block_chunks,
                        verifierRowsPolicyNBlockChunks(
                            expected_policy, serial_geometry.n_block_chunks));

                    const size_t output_elements =
                        static_cast<size_t>(M) * N;
                    EXPECT_EQ(
                        llaminar2::test::trainer::nativeByteMismatchCount(
                            grouped.data(),
                            output_elements,
                            serial.data(),
                            output_elements),
                        0u);

                    std::array<std::vector<float>, 2> fused_outputs = {{
                        std::vector<float>(output_elements, 0.0f),
                        std::vector<float>(output_elements, 0.0f),
                    }};
                    std::array<FusedVerifierRowsDesc, 2> descriptors = {{
                        {
                            &packed,
                            fused_outputs[0].data(),
                            nullptr,
                            N,
                            N,
                            nullptr,
                            DecodeSchedulePolicy::Auto,
                            candidate,
                        },
                        {
                            &packed,
                            fused_outputs[1].data(),
                            nullptr,
                            N,
                            N,
                            nullptr,
                            DecodeSchedulePolicy::Auto,
                            candidate,
                        },
                    }};
                    const auto run_fused = [&]()
                    {
                        return gemm_native_vnni_fused_verifier_rows_preq(
                            quantized_rows.data(),
                            descriptors.data(),
                            static_cast<int>(descriptors.size()),
                            M,
                            packed.blocks_per_row,
                            ISAPath::AUTO);
                    };
                    ASSERT_TRUE(run_fused());
                    const auto first_fused = fused_outputs;
                    ASSERT_TRUE(run_fused());
                    for (size_t projection = 0;
                         projection < fused_outputs.size();
                         ++projection)
                    {
                        EXPECT_EQ(
                            llaminar2::test::trainer::nativeByteMismatchCount(
                                fused_outputs[projection].data(),
                                output_elements,
                                serial.data(),
                                output_elements),
                            0u)
                            << "fused projection differs from serial rows";
                        EXPECT_EQ(
                            llaminar2::test::trainer::nativeByteMismatchCount(
                                fused_outputs[projection],
                                first_fused[projection]),
                            0u)
                            << "fused projection changed bytes across replay";
                    }

                    const char *expected_fused_route =
                        expected_policy == VerifierRowsPolicy::FullKRowChunkGrid
                            ? "grouped_full_k_row_chunk_grid"
                        : verifierRowsPolicyUsesFullKNMajor(expected_policy)
                            ? "grouped_full_k_two_row_n_major"
                            : "grouped_full_k_pair_grid";
                    uint64_t fused_route_count = 0;
                    for (const auto &record : PerfStatsCollector::snapshot(
                             {"kernel.cpu_native_vnni_fused_verifier_rows_projection_launch"}))
                    {
                        if (record.name !=
                            "cpu_native_vnni_fused_verifier_rows_projection_launch")
                        {
                            continue;
                        }
                        EXPECT_EQ(
                            record.tags.at("effective_verifier_policy"),
                            verifierRowsPolicyName(expected_policy));
                        EXPECT_EQ(record.tags.at("route"), expected_fused_route);
                        EXPECT_EQ(
                            record.tags.at("physical_row_tile"),
                            expected_policy == VerifierRowsPolicy::FullKRowChunkGrid
                                ? "1"
                                : "2");
                        fused_route_count += record.count;
                    }
                    EXPECT_EQ(
                        fused_route_count,
                        2u * descriptors.size());

                    const std::vector<float> first = grouped;
                    gemm_native_vnni_preq_decode_equivalent_rows(
                        packed,
                        quantized_rows.data(),
                        grouped.data(),
                        M,
                        N,
                        ISAPath::AUTO,
                        candidate);
                    EXPECT_EQ(
                        llaminar2::test::trainer::nativeByteMismatchCount(
                            grouped, first),
                        0u);
                }
            }
        }

        PerfStatsCollector::reset();
        ASSERT_EQ(unsetenv("LLAMINAR_PERF_STATS_JSON"), 0);
    }

    /**
     * @test Measure production grouped-policy lookup latency in isolation.
     *
     * A grouped projection can be small enough that dispatch overhead is
     * visible. The generated selector has two materially different routes:
     * exact overlays use binary search over a compact key table, while unseen
     * geometries traverse the total generic predicate surface. Measure a hot
     * repeated key and a rotating working set for both routes. The loop-only
     * control includes the same query indexing, checksum update, and compiler
     * fence, and is subtracted from every reported sample.
     *
     * The ceilings are deliberately wider than this host's observed medians
     * while remaining low enough to catch removal of either cache level. Hot
     * graph replay should remain within a few nanoseconds; rotating exact and
     * generic keys may pay the direct-table hash but must not fall back to a
     * fresh generated search on every call.
     */
    TEST_F(CPUNativeVNNIGemvTest, VerifierRowsPolicyDispatchLatency)
    {
        constexpr int N = 4096;
        constexpr int K = 2048;
        constexpr int certified_threads = 28;
        constexpr int iterations = 100000;
        constexpr int samples = 7;
        constexpr std::array<int, 5> exact_m = {2, 4, 8, 16, 31};

        omp_set_num_threads(certified_threads);
        auto weights = createWeightsForFormat("Q4_K", N, K);
        ASSERT_NE(weights, nullptr);
        CPUNativeVNNIGemmKernel kernel(weights.get());
        ASSERT_TRUE(kernel.isValid());
        const auto &packed = kernel.packedWeights();

        struct Query
        {
            int m = 2;
            int n = 1;
            int k = 32;
            int k_tiles = 0;
        };

        const auto serial_k_tiles = [&](int n, int k)
        {
            return computeTileConfig(
                       n,
                       k,
                       1,
                       packed.payload_bytes,
                       certified_threads)
                .k_tiles;
        };
        const std::array<Query, 1> exact_hot = {
            {{4, N, K, serial_k_tiles(N, K)}}};
        std::array<Query, exact_m.size()> exact_working_set{};
        for (size_t index = 0; index < exact_working_set.size(); ++index)
        {
            exact_working_set[index] = {
                exact_m[index], N, K, serial_k_tiles(N, K)};
        }

        const std::array<Query, 1> generic_hot = {
            {{8, 3073, 2016, serial_k_tiles(3073, 2016)}}};
        std::array<Query, 64> generic_working_set{};
        for (size_t index = 0; index < generic_working_set.size(); ++index)
        {
            const int query_n = 3001 + static_cast<int>(index) * 17;
            const int query_k = 1888 + static_cast<int>(index % 11) * 32;
            generic_working_set[index] = {
                exact_m[index % exact_m.size()],
                query_n,
                query_k,
                serial_k_tiles(query_n, query_k),
            };
        }

        const ISALevel effective_isa = activeISALevel();
        const auto ambient_selector = [&](const Query &query)
        {
            return selectVerifierRowsPolicy(
                packed, query.m, query.n, query.k);
        };
        const auto production_selector = [&](const Query &query)
        {
            return selectVerifierRowsPolicy(
                packed,
                query.m,
                query.n,
                query.k,
                effective_isa,
                certified_threads,
                query.k_tiles);
        };

        auto measure = [&](const auto &queries, const auto &selector)
        {
            std::array<double, samples> net_ns{};
            uint64_t checksum = 0;

            for (int warmup = 0; warmup < 1000; ++warmup)
            {
                const Query &query =
                    queries[static_cast<size_t>(warmup) % queries.size()];
                checksum += static_cast<uint64_t>(selector(query));
                std::atomic_signal_fence(std::memory_order_seq_cst);
            }

            for (int sample = 0; sample < samples; ++sample)
            {
                const auto baseline_begin = std::chrono::steady_clock::now();
                for (int iteration = 0; iteration < iterations; ++iteration)
                {
                    const Query &query = queries[
                        static_cast<size_t>(iteration) % queries.size()];
                    checksum += static_cast<uint64_t>(query.m & 1);
                    std::atomic_signal_fence(std::memory_order_seq_cst);
                }
                const auto baseline_end = std::chrono::steady_clock::now();

                const auto selector_begin = std::chrono::steady_clock::now();
                for (int iteration = 0; iteration < iterations; ++iteration)
                {
                    const Query &query = queries[
                        static_cast<size_t>(iteration) % queries.size()];
                    checksum += static_cast<uint64_t>(selector(query));
                    std::atomic_signal_fence(std::memory_order_seq_cst);
                }
                const auto selector_end = std::chrono::steady_clock::now();

                const double baseline_ns =
                    std::chrono::duration<double, std::nano>(
                        baseline_end - baseline_begin)
                        .count();
                const double selector_ns =
                    std::chrono::duration<double, std::nano>(
                        selector_end - selector_begin)
                        .count();
                net_ns[static_cast<size_t>(sample)] = std::max(
                    0.0,
                    (selector_ns - baseline_ns) /
                        static_cast<double>(iterations));
            }

            std::sort(net_ns.begin(), net_ns.end());
            EXPECT_NE(checksum, 0u);
            return net_ns[net_ns.size() / 2];
        };

        const double ambient_exact_hot_ns =
            measure(exact_hot, ambient_selector);
        const double production_exact_hot_ns =
            measure(exact_hot, production_selector);
        const double production_exact_working_set_ns =
            measure(exact_working_set, production_selector);
        const double ambient_generic_hot_ns =
            measure(generic_hot, ambient_selector);
        const double production_generic_hot_ns =
            measure(generic_hot, production_selector);
        const double production_generic_working_set_ns =
            measure(generic_working_set, production_selector);

        EXPECT_LT(production_exact_hot_ns, 5.0);
        EXPECT_LT(production_generic_hot_ns, 5.0);
        EXPECT_LT(production_exact_working_set_ns, 15.0);
        EXPECT_LT(production_generic_working_set_ns, 25.0);

        std::cout
            << "[CPUNativeVNNI][VERIFIER_POLICY_LATENCY] build="
            << compiledNativeVNNIBuildISAName()
            << " runtime=" << activeISANameForVerifierTrainer()
            << " threads=" << certified_threads
            << " ambient_exact_hot_ns=" << ambient_exact_hot_ns
            << " production_exact_hot_ns=" << production_exact_hot_ns
            << " production_exact_working_set_ns="
            << production_exact_working_set_ns
            << " ambient_generic_hot_ns=" << ambient_generic_hot_ns
            << " production_generic_hot_ns=" << production_generic_hot_ns
            << " production_generic_working_set_ns="
            << production_generic_working_set_ns
            << std::endl;
    }

    /**
     * @test Emit promotion-grade CPU grouped-verifier candidate evidence.
     *
     * This is the only CPU verifier-row surface intended for learned-policy
     * refresh.  Every explicit policy runs through the production grouped
     * entry point, publishes its effective route, repeats byte-identically,
     * and compares all FP32 words against independent serial M=1 GEMVs during
     * canonical timing. An isolated profiler transaction is already bound to
     * one authenticated timing observation, so it executes exactly one target
     * launch and validates only that launch's production route identity. The
     * older table-oriented benchmark above remains a performance diagnostic;
     * its relaxed metrics are never consumed by the common policy compiler.
     */
    TEST_F(CPUNativeVNNIGemvTest, TrainerCsv_StrongVerifierRows_AllFormats)
    {
        applyVerifierRowsThreadCapForStandalonePerf();

        const std::string paired_request_manifest_path = getEnvString(
            "LLAMINAR_CPU_NVNNI_VERIFIER_PAIRED_REQUEST_MANIFEST");
        const std::string paired_output_path = getEnvString(
            "LLAMINAR_CPU_NVNNI_VERIFIER_PAIRED_CSV");
        ASSERT_EQ(
            paired_request_manifest_path.empty(), paired_output_path.empty())
            << "CPU grouped paired confirmation requires both request and CSV paths";
        if (!paired_request_manifest_path.empty())
        {
            ASSERT_TRUE(getEnvString("LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS").empty());
            ASSERT_TRUE(getEnvString("LLAMINAR_CPU_NVNNI_VERIFIER_CANDIDATES").empty());
            ASSERT_TRUE(getEnvString("LLAMINAR_CPU_NVNNI_VERIFIER_M").empty());
            ASSERT_TRUE(getEnvString("LLAMINAR_CPU_NVNNI_VERIFIER_SHAPE_NAME").empty());
            ASSERT_TRUE(getEnvString("LLAMINAR_CPU_NVNNI_VERIFIER_STRONG_CSV").empty());
            ASSERT_TRUE(getEnvString("LLAMINAR_CPU_NVNNI_VERIFIER_TIMING_CSV").empty());
            runCPUGroupedPairedConfirmation(
                paired_request_manifest_path, paired_output_path);
            return;
        }

        const int N = getEnvInt("LLAMINAR_CPU_NVNNI_VERIFIER_N").value_or(5120);
        const int K = getEnvInt("LLAMINAR_CPU_NVNNI_VERIFIER_K").value_or(5120);
        ASSERT_GT(N, 0);
        ASSERT_GT(K, 0);
        ASSERT_EQ(K % 32, 0);

        std::string shape_name =
            getEnvString("LLAMINAR_CPU_NVNNI_VERIFIER_SHAPE_NAME");
        if (shape_name.empty())
            shape_name = "Qwen36Verifier";
        std::set<std::string> format_filters =
            getEnvCsvSet("LLAMINAR_CPU_NVNNI_VERIFIER_FORMATS");
        if (format_filters.empty())
            format_filters = {toLower("Q4_K")};
        if (format_filters.count("all") != 0)
            format_filters.clear();
        const std::set<std::string> m_filters =
            getEnvCsvSet("LLAMINAR_CPU_NVNNI_VERIFIER_M");
        const std::set<std::string> candidate_filters =
            getEnvCsvSet("LLAMINAR_CPU_NVNNI_VERIFIER_CANDIDATES");
        const std::string profiler_request_id =
            native_vnni_dispatch::profilerRequestId();
        CPUProfilerBatch profiler_batch;
        ASSERT_FALSE(
            !profiler_request_id.empty() && profiler_batch.enabled())
            << "single-request and batch CPU profiling are mutually exclusive";
        const bool profiling_requested =
            !profiler_request_id.empty() || profiler_batch.enabled();
        const std::string single_perf_output_path =
            native_vnni_dispatch::profilerEnvironment(
                native_vnni_dispatch::kPerfStatsPathEnvironment);
        std::optional<native_vnni_dispatch::LinuxPerfControl> perf_control;
        if (!profiler_request_id.empty())
        {
            // Validate profiler provenance before allocating large fixtures.
            perf_control.emplace(profiler_request_id);
        }
        else if (profiler_batch.enabled())
        {
            const CPUProfilerBatchRequest &first = profiler_batch.first();
            perf_control.emplace(first.request_id, first.output_path);
        }
        std::vector<int> verifier_rows;
        if (m_filters.empty())
        {
            verifier_rows.assign(
                kGroupedVerifierRuntimeRows.begin(),
                kGroupedVerifierRuntimeRows.end());
        }
        else
        {
            for (const std::string &raw_m : m_filters)
            {
                size_t consumed = 0;
                const int parsed_m = std::stoi(raw_m, &consumed);
                ASSERT_EQ(consumed, raw_m.size())
                    << "Invalid CPU verifier M value: " << raw_m;
                if (parsed_m == 1)
                    continue;
                ASSERT_GT(parsed_m, 1)
                    << "Grouped CPU verifier M must be at least two";
                verifier_rows.push_back(parsed_m);
            }
            std::sort(verifier_rows.begin(), verifier_rows.end());
            verifier_rows.erase(
                std::unique(verifier_rows.begin(), verifier_rows.end()),
                verifier_rows.end());
        }
        ASSERT_FALSE(verifier_rows.empty())
            << "CPU verifier trainer selected no grouped runtime-M rows";
        const int warmups = std::max(
            0,
            getEnvInt("LLAMINAR_CPU_NVNNI_VERIFIER_WARMUP").value_or(5));
        const int samples = std::max(
            1,
            getEnvInt("LLAMINAR_CPU_NVNNI_VERIFIER_ITERS").value_or(30));
        const int max_cases = std::max(
            1,
            getEnvInt("LLAMINAR_CPU_NVNNI_VERIFIER_MAX_CASES").value_or(1000000));
        if (!profiler_request_id.empty())
        {
            ASSERT_EQ(format_filters.size(), 1u)
                << "isolated CPU profiling requires exactly one source format";
            ASSERT_EQ(verifier_rows.size(), 1u)
                << "isolated CPU profiling requires exactly one runtime M";
            ASSERT_EQ(candidate_filters.size(), 1u)
                << "isolated CPU profiling requires exactly one candidate";
            ASSERT_EQ(max_cases, 1)
                << "isolated CPU profiling requires exactly one shape case";
        }
        /*
         * Promotion evidence must exercise the same ambient runtime dispatch
         * as inference. The older diagnostic can still force ISAPath directly,
         * but this strong trainer intentionally uses AUTO and relies on a fresh
         * process with LLAMINAR_ISA_LEVEL set by the refresh driver.
         */
        constexpr ISAPath isa_path = ISAPath::AUTO;
        const std::string build_isa = compiledNativeVNNIBuildISAName();
        const std::string requested_runtime_isa =
            requestedISANameForVerifierTrainer();
        const std::string effective_runtime_isa =
            activeISANameForVerifierTrainer();
        ASSERT_TRUE(
            requested_runtime_isa == "AUTO" ||
            requested_runtime_isa == "AVX2" ||
            requested_runtime_isa == "AVX512")
            << "Strong CPU NativeVNNI training requires AVX2 or AVX512, got "
            << requested_runtime_isa;
        ASSERT_TRUE(
            effective_runtime_isa == "AVX2" ||
            effective_runtime_isa == "AVX512")
            << "Strong CPU NativeVNNI training cannot certify scalar dispatch";
        if (requested_runtime_isa != "AUTO")
        {
            ASSERT_EQ(requested_runtime_isa, effective_runtime_isa)
                << "Requested CPU ISA was not available in this build/runtime";
        }
        const std::string csv_path =
            getEnvString("LLAMINAR_CPU_NVNNI_VERIFIER_STRONG_CSV");
        const std::string timing_csv_path =
            getEnvString("LLAMINAR_CPU_NVNNI_VERIFIER_TIMING_CSV");

        (void)setenv("LLAMINAR_PERF_STATS_JSON", "1", 1);
        PerfStatsCollector::reset();

        std::FILE *csv = nullptr;
        if (!csv_path.empty())
        {
            csv = std::fopen(csv_path.c_str(), "w");
            ASSERT_NE(csv, nullptr)
                << "Failed to open strong CPU verifier CSV: " << csv_path;
            std::fprintf(
                csv,
                "backend,phase,source_format,source_codebook,execution_codebook,"
                "shape,execution_mode,m,n,k,candidate_id,build_isa,"
                "runtime_isa_requested,runtime_isa_effective,threads,weight_bytes,"
                "warmup_count,sample_count,min_us,median_us,p95_us,mad_us,cv,"
                "serial_median_us,speedup,bit_mismatches,first_bit_mismatch,"
                "repeat_byte_mismatches,max_abs,relative_l2,cosine,symmetric_kld,"
                "grouped_output_digest,serial_output_digest,timing_sample_digest,"
                "route_counter_ok,observed_candidate_id,k_tiles,n_block_chunks,"
                "numerical_correctness,correctness_pass,is_winner\n");
        }

        std::FILE *timing_csv = nullptr;
        if (!timing_csv_path.empty())
        {
            timing_csv = std::fopen(timing_csv_path.c_str(), "w");
            ASSERT_NE(timing_csv, nullptr)
                << "Failed to open strong CPU timing sidecar: "
                << timing_csv_path;
            std::fprintf(
                timing_csv,
                "backend,phase,source_format,source_codebook,execution_codebook,"
                "shape,execution_mode,m,n,k,candidate_id,build_isa,"
                "runtime_isa_requested,runtime_isa_effective,sample_index,"
                "latency_us,latency_us_hex\n");
        }

        using Candidate = CPUGroupedDecodeScheduleCandidate;
        constexpr auto &CANDIDATES = CPU_GROUPED_DECODE_SCHEDULE_CANDIDATES;

        struct CandidateRow
        {
            Candidate candidate{};
            StrongTimingMeasurement timing;
            llaminar2::test::trainer::FP32Evidence comparison;
            CPUVerifierRouteEvidence route;
            size_t repeat_byte_mismatches = 0;
            double speedup = 0.0;
            int timing_warmup_count = 0;
            bool route_ok = false;
            bool numerical_correctness = false;
            bool correctness_pass = false;
        };

        int executed_cases = 0;
        int emitted_rows = 0;
        int isolated_profile_launches = 0;
        for (const auto &format : MTP_SMALL_M_FORMATS)
        {
            if (!shouldRunName(format_filters, format.name))
                continue;

            std::unique_ptr<TensorBase> weights;
            std::unique_ptr<CPUNativeVNNIGemmKernel> kernel;
            if (profiling_requested && useSyntheticProfilerPreparedWeights())
            {
                kernel = std::make_unique<CPUNativeVNNIGemmKernel>(
                    createProfilerPackedWeightsForFormat(format, N, K));
            }
            else
            {
                weights = createWeightsForFormat(
                    format.name,
                    static_cast<size_t>(N),
                    static_cast<size_t>(K));
                ASSERT_NE(weights, nullptr) << format.name;
                kernel = std::make_unique<CPUNativeVNNIGemmKernel>(
                    weights.get());
            }
            ASSERT_TRUE(kernel->isValid()) << format.name;
            const auto &packed = kernel->packedWeights();
            const size_t weight_bytes =
                packed.native_interleaved.size() + packed.payload.size();

            for (int M : verifier_rows)
            {
                if (executed_cases >= max_cases)
                    break;
                if (profiler_batch.enabled() && !profiler_batch.containsCell(
                        "NativeVNNIDecodeProjection",
                        format.name,
                        "eager",
                        M,
                        N,
                        K))
                {
                    continue;
                }

                std::mt19937 rng(static_cast<uint32_t>(
                    0xC0DEu + M * 131u + N + K + format.name.size()));
                std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
                std::vector<float> input(static_cast<size_t>(M) * K);
                for (float &value : input)
                    value = distribution(rng);

                const int K_blocks = packed.blocks_per_row;
                std::vector<Q8_1Block> quantized_rows(
                    static_cast<size_t>(M) * static_cast<size_t>(K_blocks));
                quantize_activations_to_q8_1(
                    input.data(),
                    quantized_rows.data(),
                    M,
                    K,
                    K_blocks);

                if (profiling_requested)
                {
                    /*
                     * Correctness, repeatability, and timing belong to the
                     * immutable observation named by this profiler request.
                     * Replaying those checks here used to execute roughly
                     * 3*M serial GEMVs plus warmup/repeat/timing launches for
                     * every one counter-owned launch. Besides making a full
                     * exact-point corpus take many hours, those launches could
                     * perturb caches immediately before the measured region.
                     *
                     * The profiler process therefore performs only fixture
                     * preparation followed by one armed production candidate.
                     * PerfStatsCollector is reset immediately before that
                     * launch, and the resulting route telemetry proves that
                     * the exact forceable candidate reached the expected ISA.
                     */
                    bool profiled_cell = false;
                    for (const Candidate &candidate : CANDIDATES)
                    {
                        if (!candidate_filters.empty() &&
                            !shouldRunName(
                                candidate_filters, candidate.route_name) &&
                            !shouldRunName(
                                candidate_filters, candidate.canonical_id))
                        {
                            continue;
                        }
                        CPUProfilerBatchRequest *batch_request = nullptr;
                        if (profiler_batch.enabled())
                        {
                            batch_request = profiler_batch.claim(
                                "NativeVNNIDecodeProjection",
                                format.name,
                                "eager",
                                M,
                                N,
                                K,
                                candidate.canonical_id);
                            if (!batch_request)
                                continue;
                        }

                        std::vector<float> grouped(
                            static_cast<size_t>(M) * static_cast<size_t>(N),
                            0.0f);
                        const auto run_candidate = [&]()
                        {
                            gemm_native_vnni_preq_decode_equivalent_rows(
                                packed,
                                quantized_rows.data(),
                                grouped.data(),
                                M,
                                N,
                                isa_path,
                                candidate.policy);
                        };
                        ASSERT_TRUE(perf_control.has_value());
                        const std::string &exact_request_id = batch_request
                            ? batch_request->request_id
                            : profiler_request_id;
                        const std::string &exact_output_path = batch_request
                            ? batch_request->output_path
                            : single_perf_output_path;

                        PerfStatsCollector::reset();
                        profileExactCPURequest(
                            *perf_control,
                            exact_request_id,
                            exact_output_path,
                            run_candidate);
                        const CPUVerifierRouteEvidence route =
                            findCPUVerifierRoute(M, N, K, packed.codebook_id);
                        ASSERT_TRUE(route.found);
                        ASSERT_EQ(route.count, 1u);
                        ASSERT_EQ(route.build_isa, build_isa);
                        ASSERT_EQ(route.isa, effective_runtime_isa);
                        ASSERT_EQ(route.effective_policy, candidate.route_name);

                        ++isolated_profile_launches;
                        profiled_cell = true;
                        std::fprintf(
                            stderr,
                            "[NativeVNNIProfiler][CPU] request=%s candidate=%s "
                            "format=%s M=%d N=%d K=%d launches=1\n",
                            exact_request_id.c_str(),
                            candidate.canonical_id,
                            format.name.c_str(),
                            M,
                            N,
                            K);
                    }
                    if (profiled_cell)
                        ++executed_cases;
                    continue;
                }

                std::vector<float> serial(
                    static_cast<size_t>(M) * static_cast<size_t>(N),
                    0.0f);
                const auto run_serial = [&]()
                {
                    for (int row = 0; row < M; ++row)
                    {
                        gemv_native_vnni_preq(
                            packed,
                            quantized_rows.data() +
                                static_cast<size_t>(row) * K_blocks,
                            serial.data() + static_cast<size_t>(row) * N,
                            isa_path);
                    }
                };
                run_serial();
                const std::string serial_digest =
                    llaminar2::test::trainer::nativeByteDigest(serial);
                const StrongTimingMeasurement serial_timing =
                    timeStrongTrainerCandidate(warmups, samples, run_serial);

                /*
                 * A single-projection candidate can be byte exact while the
                 * production fused descriptor scheduler still decomposes a
                 * deeper runtime M into independent one-row tasks. Exercise a
                 * two-projection bundle in every trainer cell so AVX2-build,
                 * AVX512-build/AVX2-runtime, and AVX512-runtime evidence all
                 * prove the real grouped projection route as well.
                 */
                std::vector<float> fused_projection0(
                    static_cast<size_t>(M) * static_cast<size_t>(N),
                    0.0f);
                std::vector<float> fused_projection1(
                    static_cast<size_t>(M) * static_cast<size_t>(N),
                    0.0f);
                std::array<FusedVerifierRowsDesc, 2> fused_descriptors = {{
                    {&packed, fused_projection0.data(), nullptr, N, N},
                    {&packed, fused_projection1.data(), nullptr, N, N},
                }};
                const auto run_fused_projection_bundle = [&]()
                {
                    return gemm_native_vnni_fused_verifier_rows_preq(
                        quantized_rows.data(),
                        fused_descriptors.data(),
                        static_cast<int>(fused_descriptors.size()),
                        M,
                        K_blocks,
                        isa_path);
                };

                PerfStatsCollector::reset();
                ASSERT_TRUE(run_fused_projection_bundle());
                const std::vector<float> first_fused_projection0 =
                    fused_projection0;
                const std::vector<float> first_fused_projection1 =
                    fused_projection1;
                ASSERT_TRUE(run_fused_projection_bundle());
                EXPECT_EQ(
                    llaminar2::test::trainer::nativeByteMismatchCount(
                        fused_projection0, serial),
                    0u)
                    << format.name << " M=" << M
                    << " fused projection 0 differs from serial M=1 decode";
                EXPECT_EQ(
                    llaminar2::test::trainer::nativeByteMismatchCount(
                        fused_projection1, serial),
                    0u)
                    << format.name << " M=" << M
                    << " fused projection 1 differs from serial M=1 decode";
                EXPECT_EQ(
                    llaminar2::test::trainer::nativeByteMismatchCount(
                        fused_projection0, first_fused_projection0),
                    0u)
                    << format.name << " M=" << M
                    << " fused projection 0 changed across repeats";
                EXPECT_EQ(
                    llaminar2::test::trainer::nativeByteMismatchCount(
                        fused_projection1, first_fused_projection1),
                    0u)
                    << format.name << " M=" << M
                    << " fused projection 1 changed across repeats";

                const VerifierRowsPolicy fused_auto_policy =
                    normalizeVerifierRowsPolicy(
                        selectVerifierRowsPolicy(packed, M, N, K),
                        (N + 63) / 64,
                        effective_runtime_isa == "AVX512",
                        M);
                const bool fused_auto_wide =
                    fused_auto_policy == VerifierRowsPolicy::WideRows;
                uint64_t fused_route_count = 0;
                for (const auto &record : PerfStatsCollector::snapshot(
                         {"kernel.cpu_native_vnni_fused_verifier_rows_projection_launch"}))
                {
                    if (record.name !=
                        "cpu_native_vnni_fused_verifier_rows_projection_launch")
                    {
                        continue;
                    }
                    EXPECT_EQ(record.tags.at("m"), std::to_string(M));
                    EXPECT_EQ(record.tags.at("n"), std::to_string(N));
                    EXPECT_EQ(record.tags.at("k"), std::to_string(K));
                    EXPECT_EQ(
                        record.tags.at("codebook"),
                        std::to_string(packed.codebook_id));
                    EXPECT_EQ(record.tags.at("isa"), effective_runtime_isa);
                    const int observed_k_tiles =
                        std::stoi(record.tags.at("k_tiles"));
                    const bool grouped_k_parallel = observed_k_tiles > 1;
                    EXPECT_EQ(
                        record.tags.at("route"),
                        grouped_k_parallel
                            ? "grouped_k_parallel_row_tiles"
                        : fused_auto_wide
                            ? "grouped_full_k_wide_rows"
                            : "grouped_full_k_pair_grid");
                    EXPECT_EQ(
                        record.tags.at("effective_verifier_policy"),
                        verifierRowsPolicyName(fused_auto_policy));
                    EXPECT_EQ(
                        record.tags.at("physical_row_tile"),
                        fused_auto_wide ? "4" : "2");
                    fused_route_count += record.count;
                }
                EXPECT_EQ(
                    fused_route_count,
                    2u * fused_descriptors.size())
                    << format.name << " M=" << M
                    << " did not publish one fused grouped route per projection "
                       "and repeat";

                std::vector<CandidateRow> rows;
                rows.reserve(CANDIDATES.size());
                for (const Candidate &candidate : CANDIDATES)
                {
                    if (!candidate_filters.empty() &&
                        !shouldRunName(candidate_filters, candidate.route_name) &&
                        !shouldRunName(
                            candidate_filters, candidate.canonical_id))
                    {
                        continue;
                    }
                    std::vector<float> grouped(
                        static_cast<size_t>(M) * static_cast<size_t>(N),
                        0.0f);
                    const auto run_candidate = [&]()
                    {
                        gemm_native_vnni_preq_decode_equivalent_rows(
                            packed,
                            quantized_rows.data(),
                            grouped.data(),
                            M,
                            N,
                            isa_path,
                            candidate.policy);
                    };

                    PerfStatsCollector::reset();
                    run_candidate();
                    const CPUVerifierRouteEvidence route =
                        findCPUVerifierRoute(M, N, K, packed.codebook_id);
                    const std::vector<float> first_output = grouped;
                    run_candidate();
                    const size_t repeat_byte_mismatches =
                        llaminar2::test::trainer::nativeByteMismatchCount(
                            first_output,
                            grouped);

                    /*
                     * MTP inference schedules projection bundles through the
                     * fused descriptor entry point. A direct grouped launch
                     * is therefore insufficient admission evidence: the same
                     * candidate must survive descriptor planning, publish the
                     * same normalized physical identity, and preserve every
                     * serial-row byte through that production scheduler.
                     *
                     * Keep two descriptors here. A one-projection call would
                     * miss ownership bugs in the shared OpenMP task index and
                     * would not prove that projection-local output offsets are
                     * independent. Timing remains on the single-projection
                     * candidate above so the learned cost is attributable to
                     * one dispatch key; this fused replay is a correctness and
                     * route-identity gate outside the measured interval.
                     */
                    for (FusedVerifierRowsDesc &descriptor : fused_descriptors)
                        descriptor.verifier_schedule = candidate.policy;
                    PerfStatsCollector::reset();
                    ASSERT_TRUE(run_fused_projection_bundle())
                        << format.name << " M=" << M << " candidate="
                        << candidate.route_name;
                    const std::vector<float> first_candidate_fused_projection0 =
                        fused_projection0;
                    const std::vector<float> first_candidate_fused_projection1 =
                        fused_projection1;
                    ASSERT_TRUE(run_fused_projection_bundle())
                        << format.name << " M=" << M << " candidate="
                        << candidate.route_name;

                    const size_t output_elements =
                        static_cast<size_t>(M) * static_cast<size_t>(N);
                    ASSERT_EQ(
                        llaminar2::test::trainer::nativeByteMismatchCount(
                            fused_projection0.data(),
                            output_elements,
                            serial.data(),
                            output_elements),
                        0u)
                        << format.name << " M=" << M << " candidate="
                        << candidate.route_name
                        << " fused projection 0 differs from serial M=1 decode";
                    ASSERT_EQ(
                        llaminar2::test::trainer::nativeByteMismatchCount(
                            fused_projection1.data(),
                            output_elements,
                            serial.data(),
                            output_elements),
                        0u)
                        << format.name << " M=" << M << " candidate="
                        << candidate.route_name
                        << " fused projection 1 differs from serial M=1 decode";
                    ASSERT_EQ(
                        llaminar2::test::trainer::nativeByteMismatchCount(
                            fused_projection0,
                            first_candidate_fused_projection0),
                        0u)
                        << format.name << " M=" << M << " candidate="
                        << candidate.route_name
                        << " fused projection 0 changed across repeats";
                    ASSERT_EQ(
                        llaminar2::test::trainer::nativeByteMismatchCount(
                            fused_projection1,
                            first_candidate_fused_projection1),
                        0u)
                        << format.name << " M=" << M << " candidate="
                        << candidate.route_name
                        << " fused projection 1 changed across repeats";

                    const NativeVNNITileConfig serial_geometry =
                        computeTileConfig(
                            N,
                            K,
                            1,
                            packed.payload_bytes,
                            omp_get_max_threads());
                    VerifierRowsPolicy expected_fused_policy =
                        normalizeVerifierRowsPolicy(
                            candidate.policy,
                            (N + 63) / 64,
                            effective_runtime_isa == "AVX512",
                            M);
                    if (serial_geometry.k_tiles > 1 &&
                        verifierRowsPolicyRequiresFullK(expected_fused_policy))
                    {
                        expected_fused_policy = VerifierRowsPolicy::Pairwise;
                    }
                    const bool expected_fused_wide_rows =
                        expected_fused_policy == VerifierRowsPolicy::WideRows &&
                        effective_runtime_isa == "AVX512" && M >= 3;
                    const char *expected_fused_route =
                        serial_geometry.k_tiles > 1
                            ? "grouped_k_parallel_row_tiles"
                        : expected_fused_policy ==
                                  VerifierRowsPolicy::FullKRowChunkGrid
                            ? "grouped_full_k_row_chunk_grid"
                        : verifierRowsPolicyUsesFullKNMajor(
                                  expected_fused_policy)
                            ? "grouped_full_k_two_row_n_major"
                        : expected_fused_wide_rows
                            ? "grouped_full_k_wide_rows"
                            : "grouped_full_k_pair_grid";
                    const int expected_physical_row_tile =
                        expected_fused_policy ==
                                VerifierRowsPolicy::FullKRowChunkGrid
                            ? 1
                        : expected_fused_wide_rows
                            ? 4
                            : 2;
                    const int expected_fused_n_block_chunks =
                        verifierRowsPolicyNBlockChunks(
                            expected_fused_policy,
                            serial_geometry.n_block_chunks);
                    uint64_t candidate_fused_route_count = 0;
                    for (const auto &record : PerfStatsCollector::snapshot(
                             {"kernel.cpu_native_vnni_fused_verifier_rows_projection_launch"}))
                    {
                        if (record.name !=
                            "cpu_native_vnni_fused_verifier_rows_projection_launch")
                        {
                            continue;
                        }
                        ASSERT_EQ(record.tags.at("m"), std::to_string(M));
                        ASSERT_EQ(record.tags.at("n"), std::to_string(N));
                        ASSERT_EQ(record.tags.at("k"), std::to_string(K));
                        ASSERT_EQ(
                            record.tags.at("codebook"),
                            std::to_string(packed.codebook_id));
                        ASSERT_EQ(record.tags.at("isa"), effective_runtime_isa);
                        ASSERT_EQ(
                            record.tags.at("requested_verifier_policy"),
                            candidate.route_name);
                        ASSERT_EQ(
                            record.tags.at("effective_verifier_policy"),
                            verifierRowsPolicyName(expected_fused_policy));
                        ASSERT_EQ(record.tags.at("route"), expected_fused_route);
                        ASSERT_EQ(
                            record.tags.at("physical_row_tile"),
                            std::to_string(expected_physical_row_tile));
                        ASSERT_EQ(
                            std::stoi(record.tags.at("k_tiles")),
                            serial_geometry.k_tiles);
                        ASSERT_EQ(
                            std::stoi(record.tags.at("n_block_chunks")),
                            expected_fused_n_block_chunks);
                        candidate_fused_route_count += record.count;
                    }
                    ASSERT_EQ(
                        candidate_fused_route_count,
                        2u * fused_descriptors.size())
                        << format.name << " M=" << M << " candidate="
                        << candidate.route_name
                        << " did not publish one fused route per projection and repeat";

                    /*
                     * The candidate owns one contiguous M-by-N publication.
                     * Compare all rows so work-sharing bugs on any row fail the
                     * same byte gate used to authorize the generated policy.
                     */
                    const auto comparison =
                        llaminar2::test::trainer::compareFP32(
                            grouped,
                            serial,
                            static_cast<size_t>(M) * static_cast<size_t>(N));
                    const bool route_ok =
                        route.found && route.count > 0 &&
                        route.build_isa == build_isa &&
                        route.isa == effective_runtime_isa &&
                        route.effective_policy == candidate.route_name;
                    /*
                     * A normalized request is retained as negative support
                     * evidence, but it is not an independently forceable
                     * candidate and therefore cannot win dispatch. One sample
                     * proves the observed call remains executable without
                     * spending the promotion timing budget on the same
                     * effective Pairwise route a second time.
                     */
                    const int candidate_warmups = route_ok ? warmups : 0;
                    const int candidate_samples = route_ok ? samples : 1;
                    const StrongTimingMeasurement timing =
                        timeStrongTrainerCandidate(
                            candidate_warmups,
                            candidate_samples,
                            run_candidate);
                    const bool numerical_correctness =
                        comparison.nonfinite_count == 0;
                    const bool correctness_pass =
                        route_ok &&
                        comparison.bitwiseEqual() &&
                        repeat_byte_mismatches == 0 &&
                        numerical_correctness;
                    if (route_ok)
                    {
                        EXPECT_TRUE(comparison.bitwiseEqual())
                            << format.name << " M=" << M << " candidate="
                            << candidate.route_name
                            << " is a forceable production grouped route and "
                               "must be byte-identical to serial M=1 decode";
                        EXPECT_EQ(repeat_byte_mismatches, 0u)
                            << format.name << " M=" << M << " candidate="
                            << candidate.route_name
                            << " changed native output bytes across repeats";
                    }
                    const double speedup =
                        timing.evidence.median > 0.0
                            ? serial_timing.evidence.median /
                                  timing.evidence.median
                            : 0.0;
                    rows.push_back(CandidateRow{
                        candidate,
                        timing,
                        comparison,
                        route,
                        repeat_byte_mismatches,
                        speedup,
                        candidate_warmups,
                        route_ok,
                        numerical_correctness,
                        correctness_pass});
                }

                int best_index = -1;
                for (size_t index = 0; index < rows.size(); ++index)
                {
                    if (!rows[index].correctness_pass)
                        continue;
                    if (best_index < 0 ||
                        rows[index].timing.evidence.median <
                            rows[static_cast<size_t>(best_index)]
                                .timing.evidence.median)
                    {
                        best_index = static_cast<int>(index);
                    }
                }
                ASSERT_GE(best_index, 0)
                    << format.name << " M=" << M
                    << " has no byte-exact production grouped candidate";

                for (size_t index = 0; index < rows.size(); ++index)
                {
                    const CandidateRow &row = rows[index];
                    if (csv)
                    {
                        std::fprintf(
                            csv,
                            "cpu,verifier_rows,%s,%u,%u,%s,eager,%d,%d,%d,%s,%s,%s,%s,%d,%zu,"
                            "%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%zu,%zu,%zu,"
                            "%.9g,%.9g,%.9g,%.9g,%s,%s,%s,%d,%s,%d,%d,%d,%d,%d\n",
                            format.name.c_str(),
                            static_cast<unsigned>(packed.codebook_id),
                            static_cast<unsigned>(packed.codebook_id),
                            shape_name.c_str(),
                            M,
                            N,
                            K,
                            row.candidate.route_name,
                            row.route.build_isa.c_str(),
                            requested_runtime_isa.c_str(),
                            row.route.isa.c_str(),
                            row.route.threads,
                            weight_bytes,
                            row.timing_warmup_count,
                            static_cast<int>(row.timing.samples_us.size()),
                            row.timing.evidence.min,
                            row.timing.evidence.median,
                            row.timing.evidence.p95,
                            row.timing.evidence.mad,
                            row.timing.evidence.cv,
                            serial_timing.evidence.median,
                            row.speedup,
                            row.comparison.mismatch_count,
                            row.comparison.first_mismatch_index,
                            row.repeat_byte_mismatches,
                            row.comparison.max_abs,
                            row.comparison.relative_l2,
                            row.comparison.cosine,
                            row.comparison.symmetric_kld,
                            row.comparison.actual_digest.c_str(),
                            serial_digest.c_str(),
                            row.timing.evidence.digest.c_str(),
                            row.route_ok ? 1 : 0,
                            row.route.effective_policy.c_str(),
                            row.route.k_tiles,
                            row.route.n_block_chunks,
                            row.numerical_correctness ? 1 : 0,
                            row.correctness_pass ? 1 : 0,
                            static_cast<int>(index) == best_index ? 1 : 0);
                        ++emitted_rows;
                    }

                    if (timing_csv)
                    {
                        for (size_t sample_index = 0;
                             sample_index < row.timing.samples_us.size();
                             ++sample_index)
                        {
                            const double latency_us =
                                row.timing.samples_us[sample_index];
                            std::fprintf(
                                timing_csv,
                                "cpu,verifier_rows,%s,%u,%u,%s,eager,%d,%d,%d,%s,%s,%s,%s,%zu,%.9f,%a\n",
                                format.name.c_str(),
                                static_cast<unsigned>(packed.codebook_id),
                                static_cast<unsigned>(packed.codebook_id),
                                shape_name.c_str(),
                                M,
                                N,
                                K,
                                row.candidate.route_name,
                                row.route.build_isa.c_str(),
                                requested_runtime_isa.c_str(),
                                row.route.isa.c_str(),
                                sample_index,
                                latency_us,
                                latency_us);
                        }
                    }
                }
                if (csv)
                    std::fflush(csv);
                if (timing_csv)
                    std::fflush(timing_csv);

                const CandidateRow &best =
                    rows[static_cast<size_t>(best_index)];
                std::fprintf(
                    stderr,
                    "[CPUNativeVNNI][VERIFIER][STRONG] format=%s codebook=%u shape=%s M=%d candidate=%s build_isa=%s requested_runtime_isa=%s effective_runtime_isa=%s median_us=%.3f speedup=%.3f bit_mismatches=%zu repeat_byte_mismatches=%zu\n",
                    format.name.c_str(),
                    static_cast<unsigned>(packed.codebook_id),
                    shape_name.c_str(),
                    M,
                    best.candidate.route_name,
                    best.route.build_isa.c_str(),
                    requested_runtime_isa.c_str(),
                    best.route.isa.c_str(),
                    best.timing.evidence.median,
                    best.speedup,
                    best.comparison.mismatch_count,
                    best.repeat_byte_mismatches);
                ++executed_cases;
            }

            if (executed_cases >= max_cases)
                break;
        }

        if (csv)
        {
            ASSERT_EQ(std::fclose(csv), 0);
            if (!profiling_requested)
                ASSERT_GT(emitted_rows, 0);
        }
        if (timing_csv)
            ASSERT_EQ(std::fclose(timing_csv), 0);
        ASSERT_GT(executed_cases, 0)
            << "No strong CPU verifier-row cases selected";
        if (!profiler_request_id.empty())
        {
            ASSERT_EQ(isolated_profile_launches, 1)
                << "each CPU profiler request must execute one target launch";
        }
        else if (profiler_batch.enabled())
        {
            EXPECT_NO_THROW(profiler_batch.requireComplete());
            ASSERT_EQ(
                static_cast<size_t>(isolated_profile_launches),
                profiler_batch.size())
                << "every CPU profiler batch member must launch exactly once";
        }
        PerfStatsCollector::reset();
    }

    /**
     * @test Publish the production serial-M1 arithmetic route for planning.
     *
     * The ordinary-prefill covering array must know whether an unmeasured
     * production geometry inherits a full-K serial row or a serial K-part
     * reduction. Reimplementing the cache-aware tile heuristic in Python would
     * let the collection plan silently drift from inference. This zero-kernel
     * probe instead evaluates the production `computeTileConfig()` directly
     * for every supported CPU execution codebook and every canonical shape.
     *
     * The refresh transaction runs this test once per build/runtime ISA
     * regime before constructing its timing plan. The resulting manifest is
     * architecture-local evidence: cache topology, OpenMP thread count, and
     * the exact production helper all participate in the published route.
     */
    TEST_F(CPUNativeVNNIGemvTest, TrainerCsv_PrefillSerialRouteManifest)
    {
        applyVerifierRowsThreadCapForStandalonePerf();

        const std::string output_path = getEnvString(
            "LLAMINAR_CPU_NVNNI_PREFILL_ROUTE_CSV");
        const std::string isa_regime = getEnvString(
            "LLAMINAR_CPU_NVNNI_PREFILL_ROUTE_ISA_REGIME");
        ASSERT_FALSE(output_path.empty())
            << "route-manifest probe requires an explicit output path";
        ASSERT_FALSE(isa_regime.empty())
            << "route-manifest probe requires an explicit ISA regime";

        std::FILE *csv = std::fopen(output_path.c_str(), "w");
        ASSERT_NE(csv, nullptr) << "Unable to open " << output_path;
        std::fprintf(
            csv,
            "schema_version,shape,n,k,execution_codebook,payload_bytes,"
            "threads,isa_regime,k_tiles,bundle_signature\n");

        const int threads = omp_get_max_threads();
        size_t emitted_rows = 0;
        std::vector<native_vnni_dispatch::NativeVNNIShapeSpec> route_shapes =
            native_vnni_dispatch::nativeVnniShapeManifest();

        /**
         * A failed generic fit may need fresh geometries after every nearby
         * checked-in certification point has already been measured. The Python
         * planner authors a source-digest-bound candidate ring, and this test
         * evaluates those dimensions with the production tile helper before any
         * timed kernel is launched. The final refinement plan embeds only the
         * selected route rows, so this transient probe file never becomes a
         * production model-shape catalog.
         */
        const std::string refinement_probe_path = getEnvString(
            "LLAMINAR_CPU_NVNNI_PREFILL_ROUTE_REFINEMENT_PROBE_JSON");
        if (!refinement_probe_path.empty())
        {
            std::ifstream probe_input(refinement_probe_path);
            ASSERT_TRUE(probe_input.good())
                << "Unable to open refinement route probe "
                << refinement_probe_path;
            nlohmann::json probe;
            ASSERT_NO_THROW(probe_input >> probe);
            ASSERT_TRUE(probe.is_object());
            const std::string probe_schema =
                probe.value("schema_version", std::string());
            ASSERT_TRUE(
                probe_schema == "cpu-prefill-generic-refinement-probe-v1" ||
                probe_schema == "cpu-decode-sealed-route-probe-v3");
            if (probe_schema == "cpu-prefill-generic-refinement-probe-v1")
            {
                ASSERT_TRUE(probe.contains("source_policy_digest"));
                ASSERT_TRUE(probe.contains("source_promotion_diagnostics_digest"));
            }
            else
            {
                ASSERT_TRUE(probe.contains("frozen_generic_policy_digest"));
                ASSERT_TRUE(probe.contains("development_corpus_digest"));
                ASSERT_TRUE(probe.contains("sealed_build_id"));
                ASSERT_TRUE(probe.contains("reserve_commitment"));
            }
            ASSERT_TRUE(probe.contains("shapes"));
            ASSERT_TRUE(probe.at("shapes").is_array());
            ASSERT_FALSE(probe.at("shapes").empty());

            std::set<std::string> names;
            std::set<std::pair<int, int>> dimensions;
            for (const auto &shape : route_shapes)
            {
                names.insert(shape.name);
                dimensions.emplace(shape.N, shape.K);
            }
            for (const auto &record : probe.at("shapes"))
            {
                ASSERT_TRUE(record.is_object());
                ASSERT_EQ(record.size(), 3u);
                native_vnni_dispatch::NativeVNNIShapeSpec shape;
                ASSERT_NO_THROW(shape.name = record.at("name").get<std::string>());
                ASSERT_NO_THROW(shape.N = record.at("n").get<int>());
                ASSERT_NO_THROW(shape.K = record.at("k").get<int>());
                ASSERT_TRUE(
                    shape.name.rfind("CPUPrefillAutoRefine_", 0) == 0u ||
                    shape.name.rfind("CPUDecodeAutoSeal_", 0) == 0u);
                ASSERT_GT(shape.N, 0);
                ASSERT_GT(shape.K, 0);
                ASSERT_EQ(shape.K % 32, 0);
                ASSERT_TRUE(names.insert(shape.name).second)
                    << "Duplicate route-probe shape " << shape.name;
                ASSERT_TRUE(dimensions.emplace(shape.N, shape.K).second)
                    << "Duplicate route-probe geometry "
                    << shape.N << "x" << shape.K;
                route_shapes.push_back(std::move(shape));
            }
        }

        for (const auto &shape : route_shapes)
        {
            for (int raw_codebook = 0; raw_codebook <= 255; ++raw_codebook)
            {
                const auto format = getRuntimeFormatInfo(
                    static_cast<uint8_t>(raw_codebook));
                if (format.payload_bytes <= 0)
                    continue;

                const NativeVNNITileConfig serial = computeTileConfig(
                    shape.N,
                    shape.K,
                    1,
                    format.payload_bytes,
                    threads);
                const bool serial_kpart = serial.k_tiles > 1;
                std::fprintf(
                    csv,
                    "cpu-prefill-serial-route-v1,%s,%d,%d,%d,%d,%d,%s,%d,%s\n",
                    shape.name.c_str(),
                    shape.N,
                    shape.K,
                    raw_codebook,
                    format.payload_bytes,
                    threads,
                    isa_regime.c_str(),
                    serial.k_tiles,
                    serial_kpart
                        ? "single-native-vnni-prefill:serial-kpart:fp32-output:v2"
                        : "single-native-vnni-prefill:serial-full-k:fp32-output:v2");
                ++emitted_rows;
            }
        }

        ASSERT_EQ(std::fclose(csv), 0);
        ASSERT_GT(emitted_rows, 0u);
    }

    /**
     * @test Emit promotion-grade ordinary CPU prefill schedule evidence.
     *
     * Unlike grouped verifier rows, this surface represents ordinary M>1
     * GEMM and may extend through long-prefill depths. Every candidate fixes a
     * single full-K accumulation tile and varies only physical N work sharing.
     * The test enters the production pre-quantized GEMM launcher, proves the
     * complete M-by-N output twice against independent serial M1 rows, records
     * the effective route (including normalized requests), and retains exact
     * steady-clock samples for the common learned-policy pipeline.
     *
     * Isolated Linux-perf collection consumes those immutable observations.
     * Each request prepares its fixture, installs its forceable route outside
     * the counter interval, executes one production GEMM while counters are
     * armed, and verifies the resulting route identity. Serial oracles,
     * correctness repeats, warmups, and adaptive timing remain exclusive to
     * canonical evidence collection.
     */
    TEST_F(CPUNativeVNNIGemvTest, TrainerCsv_StrongPrefill_AllFormats)
    {
        applyVerifierRowsThreadCapForStandalonePerf();

        const int N = getEnvInt("LLAMINAR_CPU_NVNNI_PREFILL_N").value_or(1024);
        const int K = getEnvInt("LLAMINAR_CPU_NVNNI_PREFILL_K").value_or(256);
        ASSERT_GT(N, 0);
        ASSERT_GT(K, 0);
        ASSERT_EQ(K % 32, 0);

        std::string shape_name =
            getEnvString("LLAMINAR_CPU_NVNNI_PREFILL_SHAPE_NAME");
        if (shape_name.empty())
            shape_name = "CPUNativeVNNIPrefillSmoke";
        std::set<std::string> format_filters =
            getEnvCsvSet("LLAMINAR_CPU_NVNNI_PREFILL_FORMATS");
        if (format_filters.empty())
            format_filters = {toLower("Q4_K")};
        if (format_filters.count("all") != 0)
            format_filters.clear();

        std::vector<int> m_values;
        const std::set<std::string> m_filters =
            getEnvCsvSet("LLAMINAR_CPU_NVNNI_PREFILL_M");
        if (m_filters.empty())
        {
            m_values = {64, 256, 1024};
        }
        else
        {
            for (const std::string &raw_m : m_filters)
            {
                size_t consumed = 0;
                const int parsed_m = std::stoi(raw_m, &consumed);
                ASSERT_EQ(consumed, raw_m.size())
                    << "Invalid CPU prefill M value: " << raw_m;
                ASSERT_GT(parsed_m, 1)
                    << "Ordinary CPU prefill M must be at least two";
                m_values.push_back(parsed_m);
            }
            std::sort(m_values.begin(), m_values.end());
            m_values.erase(
                std::unique(m_values.begin(), m_values.end()),
                m_values.end());
        }
        ASSERT_FALSE(m_values.empty());

        const std::set<std::string> candidate_filters =
            getEnvCsvSet("LLAMINAR_CPU_NVNNI_PREFILL_CANDIDATES");
        /*
         * Candidate-family admission for grouped MTP is deliberately stronger
         * than ordinary long-prefill timing.  The usual trainer samples row
         * boundaries and quartiles because replaying thousands of serial M1
         * rows in every prefill cell would dominate corpus collection.  A
         * focused grouped-candidate A/B is small (currently M <= 31) and must
         * instead prove every published FP32 word before the schedule can be
         * added to the verifier registry.  Keep that stronger oracle opt-in so
         * the diagnostic cannot silently inflate the production prefill sweep.
         */
        const bool exhaustive_serial_oracle =
            getEnvInt(
                "LLAMINAR_CPU_NVNNI_PREFILL_EXHAUSTIVE_SERIAL_ORACLE")
                .value_or(0) != 0;
        const char *serial_oracle_policy = exhaustive_serial_oracle
                                               ? "exhaustive-all-rows-v1"
                                               : "boundary-quartile-sentinel-v1";
        const std::string profiler_request_id =
            native_vnni_dispatch::profilerRequestId();
        CPUProfilerBatch profiler_batch;
        ASSERT_FALSE(
            !profiler_request_id.empty() && profiler_batch.enabled())
            << "single-request and batch CPU profiling are mutually exclusive";
        const bool profiling_requested =
            !profiler_request_id.empty() || profiler_batch.enabled();
        const std::string single_perf_output_path =
            native_vnni_dispatch::profilerEnvironment(
                native_vnni_dispatch::kPerfStatsPathEnvironment);
        std::optional<native_vnni_dispatch::LinuxPerfControl> perf_control;
        if (!profiler_request_id.empty())
        {
            // Validate profiler provenance before allocating large fixtures.
            perf_control.emplace(profiler_request_id);
        }
        else if (profiler_batch.enabled())
        {
            const CPUProfilerBatchRequest &first = profiler_batch.first();
            perf_control.emplace(first.request_id, first.output_path);
        }
        const bool mpi_round_sync =
            !profiling_requested &&
            getEnvInt("LLAMINAR_CPU_NVNNI_PREFILL_MPI_ROUND_SYNC")
                    .value_or(0) != 0;
        const bool anchored_candidate_expansion =
            !profiling_requested &&
            getEnvInt("LLAMINAR_CPU_NVNNI_PREFILL_ANCHORED_EXPANSION")
                    .value_or(0) != 0;
        std::vector<const FormatSpec *> selected_formats;
        for (const auto &format : MTP_SMALL_M_FORMATS)
        {
            if (shouldRunName(format_filters, format.name))
                selected_formats.push_back(&format);
        }
        ASSERT_FALSE(selected_formats.empty())
            << "No strong CPU prefill source formats selected";
        std::stable_sort(
            selected_formats.begin(),
            selected_formats.end(),
            [](const FormatSpec *left, const FormatSpec *right)
            {
                return left->name.size() < right->name.size();
            });
        const MPIPrefillRoundCoordinator round_coordinator(mpi_round_sync);
        round_coordinator.requireMatchingPhaseInventory(
            selected_formats.size(), m_values);
        const int warmups = std::max(
            0, getEnvInt("LLAMINAR_CPU_NVNNI_PREFILL_WARMUP").value_or(5));
        const int samples = std::max(
            1, getEnvInt("LLAMINAR_CPU_NVNNI_PREFILL_ITERS").value_or(30));
        const int maximum_samples = std::max(
            samples,
            getEnvInt("LLAMINAR_CPU_NVNNI_PREFILL_MAX_ITERS").value_or(90));
        const int minimum_samples = std::clamp(
            getEnvInt("LLAMINAR_CPU_NVNNI_PREFILL_MIN_ITERS").value_or(5),
            1,
            samples);
        const int stationary_minimum_samples = std::clamp(
            getEnvInt(
                "LLAMINAR_CPU_NVNNI_PREFILL_STATIONARY_MIN_ITERS")
                .value_or(minimum_samples),
            minimum_samples,
            maximum_samples);
        const double warmup_round_timeout_us = std::max(
            1000000.0,
            getEnvDouble(
                "LLAMINAR_CPU_NVNNI_PREFILL_WARMUP_ROUND_TIMEOUT_US")
                .value_or(30000000.0));
        const double transition_warmup_latency_multiplier = std::max(
            0.0,
            getEnvDouble(
                "LLAMINAR_CPU_NVNNI_PREFILL_TRANSITION_WARMUP_LATENCY_MULTIPLIER")
                .value_or(60.0));
        const double transition_warmup_budget_ceiling_us = std::max(
            0.0,
            getEnvDouble(
                "LLAMINAR_CPU_NVNNI_PREFILL_TRANSITION_WARMUP_BUDGET_CEILING_US")
                .value_or(2000000.0));
        const double timing_budget_us = std::max(
            0.0,
            getEnvDouble("LLAMINAR_CPU_NVNNI_PREFILL_TIMING_BUDGET_US")
                .value_or(2000000.0));
        const double median_stability_limit = std::clamp(
            getEnvDouble("LLAMINAR_CPU_NVNNI_PREFILL_MEDIAN_STABILITY")
                .value_or(0.02),
            0.0,
            1.0);
        const int max_cases = std::max(
            1,
            getEnvInt("LLAMINAR_CPU_NVNNI_PREFILL_MAX_CASES")
                .value_or(1000000));
        if (!profiler_request_id.empty())
        {
            ASSERT_EQ(format_filters.size(), 1u)
                << "isolated CPU prefill profiling requires one format";
            ASSERT_EQ(m_values.size(), 1u)
                << "isolated CPU prefill profiling requires one M";
            ASSERT_EQ(candidate_filters.size(), 1u)
                << "isolated CPU prefill profiling requires one candidate";
            ASSERT_EQ(max_cases, 1)
                << "isolated CPU prefill profiling requires one shape";
        }
        const std::string build_isa = compiledNativeVNNIBuildISAName();
        const std::string requested_runtime_isa =
            requestedISANameForVerifierTrainer();
        const std::string effective_runtime_isa =
            activeISANameForVerifierTrainer();
        ASSERT_TRUE(
            effective_runtime_isa == "AVX2" ||
            effective_runtime_isa == "AVX512")
            << "Strong CPU prefill training cannot certify scalar dispatch";
        if (requested_runtime_isa != "AUTO")
        {
            ASSERT_EQ(requested_runtime_isa, effective_runtime_isa)
                << "Requested CPU prefill ISA was unavailable";
        }

        const std::string csv_path =
            getEnvString("LLAMINAR_CPU_NVNNI_PREFILL_STRONG_CSV");
        const std::string timing_csv_path =
            getEnvString("LLAMINAR_CPU_NVNNI_PREFILL_TIMING_CSV");
        const bool append_csv =
            getEnvInt("LLAMINAR_CPU_NVNNI_PREFILL_APPEND_CSV").value_or(0) != 0;
        std::FILE *csv = nullptr;
        if (!csv_path.empty())
        {
            if (append_csv)
            {
                std::ifstream existing(csv_path);
                std::string header;
                ASSERT_TRUE(existing.good() && std::getline(existing, header));
                ASSERT_NE(
                    header.find("complete_round_probe_duration_us"),
                    std::string::npos)
                    << "CPU prefill append rejected a legacy aggregate schema";
            }
            csv = std::fopen(csv_path.c_str(), append_csv ? "a" : "w");
            ASSERT_NE(csv, nullptr);
            if (!append_csv)
            {
                std::fprintf(
                    csv,
                    "backend,phase,source_format,source_codebook,execution_codebook,"
                    "shape,execution_mode,m,n,k,candidate_id,build_isa,"
                    "runtime_isa_requested,runtime_isa_effective,threads,"
                    "measurement_coordination,mpi_world_size,mpi_rank,weight_bytes,"
                    "serial_oracle_policy,serial_oracle_rows,"
                    "preconditioning_policy,preconditioning_m,"
                    "preconditioning_budget_us,preconditioning_duration_us,"
                    "warmup_count,warmup_round_count,warmup_budget_policy,"
                    "warmup_probe_latency_us,warmup_budget_floor_us,"
                    "warmup_latency_multiplier,warmup_budget_ceiling_us,"
                    "warmup_budget_us,warmup_duration_us,global_warmup_round_count,"
                    "warmup_wall_duration_us,warmup_max_round_duration_us,"
                    "complete_round_probe_duration_us,warmup_round_timeout_us,"
                    "timing_order_seed,global_timing_round_count,sample_count,"
                    "timing_protocol,timing_ceiling_policy,min_sample_count,stable_sample_count,"
                    "max_sample_count,timing_budget_us,timed_duration_us,"
                    "stationary_sample_begin,stationary_sample_count,"
                    "stationary_duration_us,"
                    "median_stability_limit,median_relative_drift,timing_converged,"
                    "timing_stop_reason,min_us,median_us,p95_us,mad_us,cv,"
                    "serial_median_us,speedup,bit_mismatches,first_bit_mismatch,"
                    "repeat_byte_mismatches,max_abs,relative_l2,cosine,symmetric_kld,"
                    "grouped_output_digest,serial_output_digest,timing_sample_digest,"
                    "route_counter_ok,observed_candidate_id,k_tiles,k_tile_blocks,"
                    "n_block_chunks,numerical_correctness,correctness_pass,is_winner\n");
            }
        }
        std::FILE *timing_csv = nullptr;
        if (!timing_csv_path.empty())
        {
            if (append_csv)
            {
                std::ifstream existing(timing_csv_path);
                std::string header;
                ASSERT_TRUE(existing.good() && std::getline(existing, header));
                ASSERT_NE(header.find("latency_us_hex"), std::string::npos)
                    << "CPU prefill append rejected an incompatible timing schema";
            }
            timing_csv =
                std::fopen(timing_csv_path.c_str(), append_csv ? "a" : "w");
            ASSERT_NE(timing_csv, nullptr);
            if (!append_csv)
            {
                std::fprintf(
                    timing_csv,
                    "backend,phase,source_format,source_codebook,execution_codebook,"
                    "shape,execution_mode,m,n,k,candidate_id,build_isa,"
                    "runtime_isa_requested,runtime_isa_effective,sample_index,"
                    "latency_us,latency_us_hex\n");
            }
        }

        struct Candidate
        {
            std::string id;
            int n_block_chunks = 1;
            VerifierRowsPolicy verifier_policy = VerifierRowsPolicy::Pairwise;
            bool serial_kpart = false;
            PrefillSchedulePolicy schedule =
                PrefillSchedulePolicy::TwoRowNMajor;
        };
        std::vector<Candidate> candidates;
        const int n_chunks = (N + 63) / 64;
        const int row_chunk_target_tasks = std::max(1, omp_get_max_threads() / 4);
        candidates.push_back({
            .id = "cpu.nvnni.prefill.row_chunk_grid.full_k",
            .n_block_chunks = std::max(
                1,
                (n_chunks + row_chunk_target_tasks - 1) /
                    row_chunk_target_tasks),
            .schedule = PrefillSchedulePolicy::RowChunkGrid,
        });
        for (int n_block_chunks : {1, 2, 4, 8, 16})
        {
            candidates.push_back({
                .id = "cpu.nvnni.prefill.two_row_tiles.nbc" +
                      std::to_string(n_block_chunks) + ".full_k",
                .n_block_chunks = n_block_chunks,
                .schedule = PrefillSchedulePolicy::TwoRowNMajor,
            });
            candidates.push_back({
                .id = "cpu.nvnni.prefill.two_row_pair_grid.nbc" +
                      std::to_string(n_block_chunks) + ".full_k",
                .n_block_chunks = n_block_chunks,
                .schedule = PrefillSchedulePolicy::TwoRowPairGrid,
            });
        }
        candidates.push_back({
            .id = "cpu.nvnni.prefill.decode_equivalent_kpart.pairwise",
            .n_block_chunks = 1,
            .verifier_policy = VerifierRowsPolicy::Pairwise,
            .serial_kpart = true,
            .schedule = PrefillSchedulePolicy::Auto,
        });
        candidates.push_back({
            .id = "cpu.nvnni.prefill.decode_equivalent_kpart.wide_rows",
            .n_block_chunks = 1,
            .verifier_policy = VerifierRowsPolicy::WideRows,
            .serial_kpart = true,
            .schedule = PrefillSchedulePolicy::Auto,
        });

        struct CandidateRow
        {
            Candidate candidate;
            StrongTimingMeasurement timing;
            llaminar2::test::trainer::FP32Evidence comparison;
            CPUPrefillRouteEvidence route;
            size_t repeat_byte_mismatches = 0;
            double normalized_route_latency_us = 0.0;
            double speedup = 0.0;
            int timing_warmup_count = 0;
            int warmup_round_count = 0;
            double warmup_duration_us = 0.0;
            bool route_ok = false;
            bool numerical_correctness = false;
            bool correctness_pass = false;
            std::string warmup_budget_policy;
            double warmup_probe_latency_us = 0.0;
            double warmup_budget_floor_us = 0.0;
            double warmup_latency_multiplier = 0.0;
            double warmup_budget_ceiling_us = 0.0;
            double warmup_budget_us = 0.0;
        };

        /**
         * @brief Reusable evidence from one real production route launch.
         *
         * Several logical trainer candidates can normalize to the same
         * physical route when their arithmetic family disagrees with serial
         * M=1. The first compatible candidate launches and validates that
         * route. Unsupported aliases copy this immutable probe instead of
         * launching the identical kernel again. They remain ineligible for
         * timing and promotion because their requested route ID still differs
         * from the observed route ID.
         */
        struct PhysicalProbeEvidence
        {
            llaminar2::test::trainer::FP32Evidence comparison;
            CPUPrefillRouteEvidence route;
            size_t repeat_byte_mismatches = 0;
            double latency_us = 0.0;
            bool numerical_correctness = false;
        };

        (void)setenv("LLAMINAR_PERF_STATS_JSON", "1", 1);
        PerfStatsCollector::reset();
        int executed_cases = 0;
        int emitted_rows = 0;
        int isolated_profile_launches = 0;
        constexpr ISAPath isa_path = ISAPath::AUTO;
        PrefillActivationFixtureCache activation_fixtures(N, K);
        for (size_t format_index = 0;
             format_index < selected_formats.size();
             ++format_index)
        {
            const FormatSpec &format = *selected_formats[format_index];
            std::unique_ptr<TensorBase> weights;
            std::unique_ptr<CPUNativeVNNIGemmKernel> kernel;
            if (profiling_requested && useSyntheticProfilerPreparedWeights())
            {
                kernel = std::make_unique<CPUNativeVNNIGemmKernel>(
                    createProfilerPackedWeightsForFormat(format, N, K));
            }
            else
            {
                weights = createWeightsForFormat(format.name, N, K);
                ASSERT_NE(weights, nullptr) << format.name;
                kernel = std::make_unique<CPUNativeVNNIGemmKernel>(
                    weights.get());
            }
            ASSERT_TRUE(kernel->isValid()) << format.name;
            const auto &packed = kernel->packedWeights();
            const size_t weight_bytes =
                packed.native_interleaved.size() + packed.payload.size();
            const int k_blocks = packed.blocks_per_row;

            // Long production jobs visit several M buckets for the same packed
            // source tensor.  Frequency and memory-domain preconditioning is a
            // process/source property, not a reason to burn one second per
            // candidate again at every M.  The first measured cell establishes
            // the complete-candidate load envelope.  Later cells use their first
            // correctness launch as a latency probe and warm for enough complete
            // rounds to cover the configured number of equivalent launches.
            const double source_preconditioning_budget_us =
                !profiling_requested
                ? std::max(
                      0.0,
                      getEnvDouble(
                          "LLAMINAR_CPU_NVNNI_PREFILL_WARMUP_BUDGET_US")
                          .value_or(0.0))
                : 0.0;
            const double transition_warmup_budget_us =
                !profiling_requested
                ? std::max(
                      0.0,
                      getEnvDouble(
                          "LLAMINAR_CPU_NVNNI_PREFILL_TRANSITION_WARMUP_BUDGET_US")
                          .value_or(0.0))
                : 0.0;
            const std::string preconditioning_policy =
                source_preconditioning_budget_us > 0.0
                ? "source-complete-round-v1"
                : "disabled";
            bool source_preconditioned =
                source_preconditioning_budget_us <= 0.0;
            int source_preconditioning_m = 0;
            double source_preconditioning_duration_us = 0.0;

            for (int M : m_values)
            {
                if (executed_cases >= max_cases)
                    break;
                if (profiler_batch.enabled() && !profiler_batch.containsCell(
                        "NativeVNNIPrefillProjection",
                        format.name,
                        "eager",
                        M,
                        N,
                        K))
                {
                    continue;
                }
                // Every rank enters the same M phase before fixture generation.
                // A second boundary below prevents a faster correctness setup
                // from entering warmup while a peer still runs its serial oracle.
                round_coordinator.barrier();
                const std::vector<Q8_1Block> &quantized_rows =
                    activation_fixtures.fixtureFor(
                        M, format.name, k_blocks);

                std::vector<float> grouped(static_cast<size_t>(M) * N, 0.0f);
                const NativeVNNITileConfig serial_route = computeTileConfig(
                    N,
                    K,
                    1,
                    packed.payload_bytes,
                    omp_get_max_threads());
                const bool serial_m1_uses_kpart = serial_route.k_tiles > 1;
                const auto with_candidate_route =
                    [&](const Candidate &candidate, auto &&operation)
                {
                    ScopedCPUNativeVNNITileOverride force_n_blocks(
                        "LLAMINAR_CPU_VNNI_N_BLOCK_CHUNKS",
                        std::to_string(candidate.n_block_chunks));
                    ScopedCPUNativeVNNITileOverride force_full_k(
                        "LLAMINAR_CPU_VNNI_K_TILE_BLOCKS",
                        std::to_string(k_blocks));
                    operation();
                };
                const auto launch_candidate = [&](const Candidate &candidate)
                {
                    gemm_native_vnni_preq(
                        packed,
                        quantized_rows.data(),
                        grouped.data(),
                        M,
                        N,
                        isa_path,
                        candidate.verifier_policy,
                        candidate.schedule);
                };
                const auto run_candidate = [&](const Candidate &candidate)
                {
                    with_candidate_route(
                        candidate, [&]() { launch_candidate(candidate); });
                };
                const auto measure_candidate = [&](const Candidate &candidate)
                {
                    double latency_us = 0.0;
                    with_candidate_route(candidate, [&]()
                    {
                        const auto begin = std::chrono::steady_clock::now();
                        launch_candidate(candidate);
                        const auto end = std::chrono::steady_clock::now();
                        latency_us = std::chrono::duration<double, std::micro>(
                            end - begin).count();
                    });
                    return latency_us;
                };
                const auto route_matches_candidate =
                    [&](const Candidate &candidate,
                        const CPUPrefillRouteEvidence &route)
                {
                    const bool pair_grid =
                        candidate.schedule ==
                            PrefillSchedulePolicy::TwoRowPairGrid;
                    return route.found && route.count > 0 &&
                           route.build_isa == build_isa &&
                           route.isa == effective_runtime_isa &&
                           (candidate.serial_kpart
                                ? route.k_tiles > 1 &&
                                      route.k_tile_blocks ==
                                          (k_blocks + route.k_tiles - 1) /
                                              route.k_tiles
                                : route.k_tiles == 1 &&
                                      route.k_tile_blocks == k_blocks) &&
                           (!pair_grid ||
                                (route.pair_grid_execution_found &&
                                 route.pair_grid_execution_count == route.count &&
                                 route.pair_grid_execution_n_block_chunks ==
                                     candidate.n_block_chunks &&
                                 route.pair_grid_parallel_tasks ==
                                     ((M + 1) / 2) *
                                         ((n_chunks +
                                           candidate.n_block_chunks - 1) /
                                          candidate.n_block_chunks))) &&
                           route.candidateId() == candidate.id;
                };

                if (profiling_requested)
                {
                    /*
                     * A profiler request is joined to an immutable canonical
                     * timing row. It needs a hardware-counter witness for that
                     * exact production candidate, not another serial oracle,
                     * route tournament, warmup, or adaptive timing epoch.
                     * Candidate overrides are installed before counters arm;
                     * the controlled region therefore contains exactly one
                     * GEMM invocation and no environment mutation.
                     */
                    ASSERT_TRUE(perf_control.has_value());
                    size_t cell_profile_launches = 0;
                    for (const Candidate &candidate : candidates)
                    {
                        if (!candidate_filters.empty() &&
                            !shouldRunName(candidate_filters, candidate.id))
                        {
                            continue;
                        }
                        CPUProfilerBatchRequest *batch_request = nullptr;
                        if (profiler_batch.enabled())
                        {
                            batch_request = profiler_batch.claim(
                                "NativeVNNIPrefillProjection",
                                format.name,
                                "eager",
                                M,
                                N,
                                K,
                                candidate.id);
                            if (!batch_request)
                                continue;
                        }
                        const std::string &exact_request_id = batch_request
                            ? batch_request->request_id
                            : profiler_request_id;
                        const std::string &exact_output_path = batch_request
                            ? batch_request->output_path
                            : single_perf_output_path;

                        with_candidate_route(candidate, [&]()
                        {
                            PerfStatsCollector::reset();
                            profileExactCPURequest(
                                *perf_control,
                                exact_request_id,
                                exact_output_path,
                                [&]() { launch_candidate(candidate); });
                        });
                        const CPUPrefillRouteEvidence route =
                            findCPUPrefillRoute(M, N, K, packed.codebook_id);
                        ASSERT_EQ(route.count, 1u);
                        ASSERT_TRUE(route_matches_candidate(candidate, route))
                            << "profiler selected a normalized CPU prefill "
                               "candidate: requested="
                            << candidate.id << " observed="
                            << route.candidateId();

                        ++isolated_profile_launches;
                        ++cell_profile_launches;
                        std::fprintf(
                            stderr,
                            "[NativeVNNIProfiler][CPU_PREFILL] request=%s "
                            "candidate=%s format=%s M=%d N=%d K=%d launches=1\n",
                            exact_request_id.c_str(),
                            candidate.id.c_str(),
                            format.name.c_str(),
                            M,
                            N,
                            K);
                    }
                    ASSERT_GT(cell_profile_launches, 0u)
                        << "profiler batch selected a cell without a "
                           "forceable requested candidate";
                    ++executed_cases;
                    continue;
                }

                const std::vector<int> serial_oracle_rows =
                    prefillSerialOracleRows(M, exhaustive_serial_oracle);
                ASSERT_FALSE(serial_oracle_rows.empty());
                std::vector<float> serial(
                    serial_oracle_rows.size() * static_cast<size_t>(N), 0.0f);
                const auto run_serial = [&]()
                {
                    for (size_t oracle_index = 0;
                         oracle_index < serial_oracle_rows.size();
                         ++oracle_index)
                    {
                        const int row = serial_oracle_rows[oracle_index];
                        gemv_native_vnni_preq(
                            packed,
                            quantized_rows.data() +
                                static_cast<size_t>(row) * k_blocks,
                            serial.data() + oracle_index * static_cast<size_t>(N),
                            isa_path);
                    }
                };
                run_serial();
                const std::string serial_digest =
                    llaminar2::test::trainer::nativeByteDigest(serial);
                const StrongTimingMeasurement serial_timing =
                    timeStrongTrainerCandidateAdaptive(
                        std::min(warmups, 2),
                        std::min(minimum_samples, 3),
                        std::min(samples, 5),
                        std::min(timing_budget_us, 1000000.0),
                        median_stability_limit,
                        run_serial);

                std::vector<CandidateRow> rows;
                rows.reserve(candidates.size());
                std::vector<float> grouped_oracle(serial.size(), 0.0f);

                // Probe candidates from the production-compatible arithmetic
                // family first. Their real launch populates the physical-route
                // cache used by the unsupported family later in this loop.
                // The final row order is restored to registry order below so
                // existing corpus serialization remains stable.
                std::vector<Candidate> probe_candidates = candidates;
                std::stable_sort(
                    probe_candidates.begin(),
                    probe_candidates.end(),
                    [&](const Candidate &left, const Candidate &right)
                    {
                        const bool left_matches =
                            left.serial_kpart == serial_m1_uses_kpart;
                        const bool right_matches =
                            right.serial_kpart == serial_m1_uses_kpart;
                        return left_matches && !right_matches;
                    });
                std::unordered_map<std::string, PhysicalProbeEvidence>
                    physical_route_probes;
                const auto snapshot_grouped_oracle = [&]()
                {
                    for (size_t oracle_index = 0;
                         oracle_index < serial_oracle_rows.size();
                         ++oracle_index)
                    {
                        const size_t grouped_offset =
                            static_cast<size_t>(serial_oracle_rows[oracle_index]) * N;
                        std::copy_n(
                            grouped.data() + grouped_offset,
                            static_cast<size_t>(N),
                            grouped_oracle.data() +
                                oracle_index * static_cast<size_t>(N));
                    }
                };

                // Correctness and route identity are established before any
                // latency can influence candidate eligibility. The same shared
                // output allocation is reused across candidates so a long-M
                // cell does not retain six multi-gigabyte output tensors. A
                // normalized alias consumes an earlier physical probe only;
                // it never contributes another launch or timing sample.
                for (const Candidate &candidate : probe_candidates)
                {
                    if (!candidate_filters.empty() &&
                        !shouldRunName(candidate_filters, candidate.id))
                    {
                        continue;
                    }

                    const std::string expected_physical_id(
                        llaminar2::test::trainer::
                            expectedCPUPrefillPhysicalCandidateId(
                                candidate.id,
                                serial_m1_uses_kpart,
                                candidate.serial_kpart,
                                candidate.verifier_policy ==
                                    VerifierRowsPolicy::WideRows,
                                effective_runtime_isa == "AVX512",
                                M,
                                N,
                                omp_get_max_threads()));
                    const auto cached_probe =
                        physical_route_probes.find(expected_physical_id);
                    if (cached_probe != physical_route_probes.end())
                    {
                        const PhysicalProbeEvidence &probe =
                            cached_probe->second;
                        const bool route_ok =
                            route_matches_candidate(candidate, probe.route);
                        rows.push_back(CandidateRow{
                            candidate,
                            {},
                            probe.comparison,
                            probe.route,
                            probe.repeat_byte_mismatches,
                            probe.latency_us,
                            0.0,
                            0,
                            0,
                            0.0,
                            route_ok,
                            probe.numerical_correctness,
                            route_ok && probe.comparison.bitwiseEqual() &&
                                probe.repeat_byte_mismatches == 0 &&
                                probe.numerical_correctness,
                        });
                        continue;
                    }

                    PerfStatsCollector::reset();
                    const double first_latency_us =
                        measure_candidate(candidate);
                    snapshot_grouped_oracle();
                    const CPUPrefillRouteEvidence route =
                        findCPUPrefillRoute(M, N, K, packed.codebook_id);
                    const auto first_comparison =
                        llaminar2::test::trainer::compareFP32(
                            grouped_oracle, serial, serial.size());
                    const std::string first_digest =
                        llaminar2::test::trainer::nativeByteDigest(grouped_oracle);
                    const bool route_ok =
                        route_matches_candidate(candidate, route);
                    auto comparison = first_comparison;
                    std::string second_digest = first_digest;
                    if (route_ok)
                    {
                        run_candidate(candidate);
                        snapshot_grouped_oracle();
                        comparison = llaminar2::test::trainer::compareFP32(
                            grouped_oracle, serial, serial.size());
                        second_digest =
                            llaminar2::test::trainer::nativeByteDigest(
                                grouped_oracle);
                    }
                    const size_t repeat_byte_mismatches =
                        first_digest == second_digest ? 0u : 1u;
                    const bool numerical_correctness =
                        first_comparison.nonfinite_count == 0 &&
                        comparison.nonfinite_count == 0;
                    const bool correctness_pass =
                        route_ok && first_comparison.bitwiseEqual() &&
                        comparison.bitwiseEqual() &&
                        repeat_byte_mismatches == 0 && numerical_correctness;
                    if (route_ok)
                    {
                        EXPECT_TRUE(first_comparison.bitwiseEqual())
                            << format.name << " M=" << M << " candidate="
                            << candidate.id << " first launch differs from serial M1"
                            << " mismatches=" << first_comparison.mismatch_count
                            << " first=" << first_comparison.first_mismatch_index
                            << " actual=" << first_comparison.actual_digest
                            << " serial=" << first_comparison.expected_digest;
                        EXPECT_TRUE(comparison.bitwiseEqual())
                            << format.name << " M=" << M << " candidate="
                            << candidate.id << " repeat differs from serial M1"
                            << " mismatches=" << comparison.mismatch_count
                            << " first=" << comparison.first_mismatch_index
                            << " actual=" << comparison.actual_digest
                            << " serial=" << comparison.expected_digest;
                    }

                    const std::string observed_physical_id =
                        route.candidateId();
                    if (!observed_physical_id.empty())
                    {
                        EXPECT_EQ(observed_physical_id, expected_physical_id)
                            << format.name << " M=" << M << " candidate="
                            << candidate.id
                            << " disagrees with the probe normalization plan";
                        physical_route_probes.try_emplace(
                            observed_physical_id,
                            PhysicalProbeEvidence{
                                comparison,
                                route,
                                repeat_byte_mismatches,
                                first_latency_us,
                                numerical_correctness,
                            });
                    }
                    rows.push_back(CandidateRow{
                        candidate,
                        {},
                        comparison,
                        route,
                        repeat_byte_mismatches,
                        first_latency_us,
                        0.0,
                        0,
                        0,
                        0.0,
                        route_ok,
                        numerical_correctness,
                        correctness_pass,
                    });
                }

                std::stable_sort(
                    rows.begin(),
                    rows.end(),
                    [&](const CandidateRow &left, const CandidateRow &right)
                    {
                        const auto inventory_index =
                            [&](const std::string &candidate_id)
                        {
                            const auto iterator = std::find_if(
                                candidates.begin(),
                                candidates.end(),
                                [&](const Candidate &candidate)
                                {
                                    return candidate.id == candidate_id;
                                });
                            return std::distance(candidates.begin(), iterator);
                        };
                        return inventory_index(left.candidate.id) <
                               inventory_index(right.candidate.id);
                    });

                std::vector<size_t> forceable_indices;
                for (size_t index = 0; index < rows.size(); ++index)
                {
                    if (rows[index].route_ok)
                        forceable_indices.push_back(index);
                }
                round_coordinator.failAllRanksIf(
                    forceable_indices.empty(),
                    "CPU prefill route discovery rejected: format=" +
                        format.name + " M=" + std::to_string(M) +
                        " has no forceable full-K prefill candidate");

                const uint32_t timing_order_seed = static_cast<uint32_t>(
                    0x51A7u + M * 257u + N * 17u + K * 31u +
                    packed.codebook_id * 101u +
                    (build_isa == "AVX512" ? 0xA512u : 0xA2u));
                std::mt19937 order_rng(timing_order_seed);
                const int minimum_warmup_rounds = std::max(0, warmups - 2);
                const bool source_warmup = !source_preconditioned;
                const double warmup_budget_floor_us = source_preconditioned
                    ? transition_warmup_budget_us
                    : source_preconditioning_budget_us;
                size_t active_warmup_candidates = 0;
                for (CandidateRow &row : rows)
                {
                    row.warmup_probe_latency_us =
                        row.normalized_route_latency_us;
                    row.warmup_budget_floor_us = warmup_budget_floor_us;
                    if (warmup_budget_floor_us <= 0.0)
                    {
                        row.warmup_budget_policy = "disabled";
                    }
                    else if (source_warmup)
                    {
                        row.warmup_budget_policy = "source-fixed-v1";
                        row.warmup_budget_us = warmup_budget_floor_us;
                    }
                    else if (transition_warmup_latency_multiplier == 0.0)
                    {
                        // An additive candidate-expansion run carries an
                        // existing schedule in every complete round as a
                        // contemporaneous clock anchor. Its transition phase
                        // therefore needs to establish only the configured
                        // fixed thermal/load floor; multiplying a very short
                        // probe into hundreds of redundant launches would add
                        // no independent evidence. The adapter admits this
                        // policy only for an authenticated anchored cohort.
                        row.warmup_budget_policy = "transition-fixed-v1";
                        row.warmup_budget_us = warmup_budget_floor_us;
                    }
                    else
                    {
                        row.warmup_budget_policy =
                            "transition-probe-scaled-v1";
                        row.warmup_latency_multiplier =
                            transition_warmup_latency_multiplier;
                        row.warmup_budget_ceiling_us = std::max(
                            warmup_budget_floor_us,
                            transition_warmup_budget_ceiling_us);
                        row.warmup_budget_us = std::max(
                            warmup_budget_floor_us,
                            std::min(
                                row.warmup_budget_ceiling_us,
                                row.warmup_probe_latency_us *
                                    row.warmup_latency_multiplier));
                    }
                    if (row.route_ok &&
                        (row.warmup_round_count < minimum_warmup_rounds ||
                        row.warmup_duration_us < row.warmup_budget_us)
                    )
                    {
                        ++active_warmup_candidates;
                    }
                }
                // A complete round deliberately launches every forceable
                // candidate. For very large exact-overlay GEMMs that useful
                // work can exceed the fixed hang-detection floor even though
                // every kernel is making progress. Scale the watchdog from
                // the already measured per-candidate probes, retaining a 4x
                // margin for shuffled order, peer skew, and thermal drift.
                const double probed_complete_round_us = std::accumulate(
                    forceable_indices.begin(),
                    forceable_indices.end(),
                    0.0,
                    [&](double total, size_t index)
                    {
                        return total + rows[index].warmup_probe_latency_us;
                    });
                const double effective_warmup_round_timeout_us = std::max(
                    warmup_round_timeout_us,
                    probed_complete_round_us * 4.0 + 1000000.0);
                // Warmup uses the same complete-round load envelope as timing.
                // A candidate that reaches its own elapsed budget continues as
                // a pacing launch until the final candidate on every socket is
                // ready.  The pre-loop barrier also keeps serial-oracle setup
                // outside the authenticated preconditioning cadence.
                round_coordinator.barrier();
                const auto warmup_wall_begin = std::chrono::steady_clock::now();
                double warmup_max_round_duration_us = 0.0;
                size_t global_warmup_round_count = 0;
                bool any_rank_warming = round_coordinator.anyRankActive(
                    active_warmup_candidates > 0);
                while (any_rank_warming)
                {
                    const auto warmup_round_begin =
                        std::chrono::steady_clock::now();
                    bool local_warmup_failure = false;
                    std::ostringstream local_warmup_diagnostic;
                    std::vector<size_t> round = forceable_indices;
                    std::shuffle(round.begin(), round.end(), order_rng);
                    for (size_t index : round)
                    {
                        CandidateRow &row = rows[index];
                        const double warmup_latency_us =
                            measure_candidate(row.candidate);
                        if ((!std::isfinite(warmup_latency_us) ||
                             warmup_latency_us <= 0.0) &&
                            !local_warmup_failure)
                        {
                            local_warmup_failure = true;
                            local_warmup_diagnostic
                                << "CPU prefill warmup rejected: format="
                                << format.name << " M=" << M << " candidate="
                                << row.candidate.id
                                << " timer made no forward progress; latency_us="
                                << warmup_latency_us;
                        }
                        if (local_warmup_failure)
                            continue;
                        row.warmup_duration_us += warmup_latency_us;
                        ++row.warmup_round_count;
                    }
                    active_warmup_candidates = 0;
                    for (size_t index : forceable_indices)
                    {
                        const CandidateRow &row = rows[index];
                        if (row.warmup_round_count < minimum_warmup_rounds ||
                            row.warmup_duration_us < row.warmup_budget_us)
                        {
                            ++active_warmup_candidates;
                        }
                    }
                    ++global_warmup_round_count;
                    const double round_wall_duration_us =
                        std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - warmup_round_begin)
                            .count();
                    warmup_max_round_duration_us = std::max(
                        warmup_max_round_duration_us,
                        round_wall_duration_us);
                    if (round_wall_duration_us >=
                            effective_warmup_round_timeout_us &&
                        !local_warmup_failure)
                    {
                        local_warmup_failure = true;
                        local_warmup_diagnostic
                            << "CPU prefill warmup rejected: format="
                            << format.name << " M=" << M
                            << " one complete round exceeded the watchdog"
                            << " duration_us=" << round_wall_duration_us
                            << " timeout_us="
                            << effective_warmup_round_timeout_us
                            << " active_candidates=" << active_warmup_candidates
                            << " rounds=" << global_warmup_round_count;
                    }
                    round_coordinator.failAllRanksIf(
                        local_warmup_failure,
                        local_warmup_diagnostic.str());
                    any_rank_warming = round_coordinator.anyRankActive(
                        active_warmup_candidates > 0);
                }
                const double warmup_wall_duration_us =
                    std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - warmup_wall_begin)
                        .count();
                for (size_t index : forceable_indices)
                {
                    rows[index].timing_warmup_count =
                        2 + rows[index].warmup_round_count;
                }
                if (!source_preconditioned)
                {
                    source_preconditioning_duration_us =
                        std::numeric_limits<double>::infinity();
                    for (size_t index : forceable_indices)
                    {
                        source_preconditioning_duration_us = std::min(
                            source_preconditioning_duration_us,
                            rows[index].warmup_duration_us);
                    }
                    round_coordinator.failAllRanksIf(
                        source_preconditioning_duration_us <
                            source_preconditioning_budget_us,
                        "CPU prefill source preconditioning ended before its "
                        "elapsed budget");
                    source_preconditioning_m = M;
                    source_preconditioned = true;
                }

                // Canonical timing is collected in shuffled complete rounds.
                // Ordinary prefill waits until every candidate is independently
                // stationary. Anchored candidate expansion instead retains one
                // common elapsed/sample-complete epoch: all candidates and the
                // existing route anchor see exactly the same rounds, and the
                // adapter admits their medians only after attaching the
                // contemporaneous-anchor normalization proof. Requiring seven
                // independent raw half-window tests here would select lucky
                // quiet tails and repeatedly discard otherwise valid common
                // epochs, defeating both the normalization design and the
                // bounded collection contract.
                size_t global_timing_round_count = 0;
                bool any_rank_timing = round_coordinator.anyRankActive(
                    !forceable_indices.empty());
                while (any_rank_timing)
                {
                    // Ordinary timing candidates are independent evidence
                    // streams. Freeze each stream at its first promotable
                    // stationary suffix so a later random tail cannot revoke
                    // evidence merely because another candidate or MPI peer
                    // needs more rounds. Anchored expansion is intentionally
                    // different: all schedules must retain matching complete
                    // rounds for contemporaneous-anchor normalization.
                    std::vector<size_t> round;
                    round.reserve(forceable_indices.size());
                    for (size_t index : forceable_indices)
                    {
                        const CandidateRow &row = rows[index];
                        const bool needs_another_sample =
                            llaminar2::test::trainer::
                                timingCandidateParticipatesInActiveRound(
                                    anchored_candidate_expansion,
                                    row.timing.adaptive,
                                    row.timing.acquisition_samples_us.size(),
                                    static_cast<size_t>(maximum_samples),
                                    timing_budget_us);
                        if (needs_another_sample)
                            round.push_back(index);
                    }
                    std::shuffle(round.begin(), round.end(), order_rng);
                    for (size_t index : round)
                    {
                        CandidateRow &row = rows[index];
                        row.timing.acquisition_samples_us.push_back(
                            measure_candidate(row.candidate));
                        row.timing.adaptive =
                            llaminar2::test::trainer::evaluateAdaptiveTiming(
                                row.timing.acquisition_samples_us,
                                static_cast<size_t>(
                                    stationary_minimum_samples),
                                static_cast<size_t>(maximum_samples),
                                timing_budget_us,
                                median_stability_limit);
                    }
                    ++global_timing_round_count;

                    bool local_needs_timing = false;
                    for (size_t index : forceable_indices)
                    {
                        const CandidateRow &row = rows[index];
                        const bool anchored_epoch_complete =
                            llaminar2::test::trainer::
                                anchoredCompleteRoundTimingEvidence(
                                    row.timing.acquisition_samples_us.size(),
                                    static_cast<size_t>(
                                        stationary_minimum_samples),
                                    row.timing.adaptive.measured_duration_us,
                                    timing_budget_us);
                        local_needs_timing = local_needs_timing ||
                            (anchored_candidate_expansion
                                ? !anchored_epoch_complete
                                : llaminar2::test::trainer::
                                      adaptiveTimingNeedsAnotherSample(
                                          row.timing.adaptive,
                                          row.timing.acquisition_samples_us.size(),
                                          static_cast<size_t>(maximum_samples),
                                          timing_budget_us));
                    }
                    any_rank_timing = round_coordinator.anyRankActive(
                        local_needs_timing);
                }
                if (anchored_candidate_expansion)
                {
                    // The entire acquisition sequence is the authenticated
                    // common epoch. Re-evaluate its diagnostics with an
                    // all-sample window so the aggregate and timing sidecar
                    // prove the exact same bytes; median drift remains visible
                    // but is not mistaken for seven independent promotion
                    // gates before anchor normalization exists.
                    for (size_t index : forceable_indices)
                    {
                        CandidateRow &row = rows[index];
                        row.timing.adaptive =
                            llaminar2::test::trainer::evaluateAdaptiveTiming(
                                row.timing.acquisition_samples_us,
                                row.timing.acquisition_samples_us.size(),
                                static_cast<size_t>(maximum_samples),
                                timing_budget_us,
                                median_stability_limit);
                    }
                }
                for (CandidateRow &row : rows)
                {
                    if (!row.route_ok)
                    {
                        row.timing.acquisition_samples_us = {
                            row.normalized_route_latency_us};
                        row.timing.adaptive =
                            llaminar2::test::trainer::evaluateAdaptiveTiming(
                                row.timing.acquisition_samples_us,
                                1u,
                                1u,
                                0.0,
                                median_stability_limit);
                    }
                    const size_t stationary_begin = std::min(
                        row.timing.adaptive.stationary_window_begin,
                        row.timing.acquisition_samples_us.size());
                    row.timing.samples_us.assign(
                        row.timing.acquisition_samples_us.begin() +
                            static_cast<std::ptrdiff_t>(stationary_begin),
                        row.timing.acquisition_samples_us.end());
                    std::sort(
                        row.timing.samples_us.begin(),
                        row.timing.samples_us.end());
                    row.timing.evidence =
                        llaminar2::test::trainer::summarizeSortedTimingSamples(
                            row.timing.samples_us);
                    row.timing.stop_reason =
                        !row.route_ok
                        ? "fixed_samples"
                        : (anchored_candidate_expansion
                            ? "anchored_complete_rounds"
                            : (row.timing.adaptive.promotion_evidence
                            ? "stationary_window"
                            : "hard_samples_and_four_elapsed"));
                    const double projected_serial_median_us =
                        serial_timing.evidence.median *
                        static_cast<double>(M) /
                        static_cast<double>(serial_oracle_rows.size());
                    row.speedup = row.timing.evidence.median > 0.0
                        ? projected_serial_median_us /
                              row.timing.evidence.median
                        : 0.0;
                }

                int best_index = -1;
                for (size_t index = 0; index < rows.size(); ++index)
                {
                    const CandidateRow &row = rows[index];
                    const bool timing_admitted = anchored_candidate_expansion
                        ? llaminar2::test::trainer::
                              anchoredCompleteRoundTimingEvidence(
                                  row.timing.acquisition_samples_us.size(),
                                  static_cast<size_t>(
                                      stationary_minimum_samples),
                                  row.timing.adaptive.measured_duration_us,
                                  timing_budget_us)
                        : row.timing.adaptive.promotion_evidence;
                    if (!row.correctness_pass || !timing_admitted)
                        continue;
                    if (best_index < 0 ||
                        row.timing.evidence.median <
                            rows[static_cast<size_t>(best_index)]
                                .timing.evidence.median)
                    {
                        best_index = static_cast<int>(index);
                    }
                }
                for (size_t index = 0; index < rows.size(); ++index)
                {
                    const CandidateRow &row = rows[index];
                    const std::string observed_id = row.route.candidateId();
                    if (csv)
                    {
                        std::fprintf(
                            csv,
                            "cpu,prefill_gemm,%s,%u,%u,%s,eager,%d,%d,%d,%s,%s,%s,%s,%d,%s,%d,%d,%zu,%s,%zu,"
                            "%s,%d,%.9f,%.9f,%d,%d,%s,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%zu,%.9f,%.9f,%.9f,%.9f,%u,%zu,%d,elapsed-stability-interleaved-v16,samples-and-four-elapsed-v3,%d,%d,%d,"
                            "%.9f,%.9f,%zu,%zu,%.9f,%.9f,%.9f,%d,%s,"
                            "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%zu,%zu,%zu,"
                            "%.9g,%.9g,%.9g,%.9g,%s,%s,%s,%d,%s,%d,%d,%d,%d,%d,%d\n",
                            format.name.c_str(),
                            static_cast<unsigned>(packed.codebook_id),
                            static_cast<unsigned>(packed.codebook_id),
                            shape_name.c_str(),
                            M, N, K,
                            row.candidate.id.c_str(),
                            row.route.build_isa.c_str(),
                            requested_runtime_isa.c_str(),
                            row.route.isa.c_str(),
                            row.route.threads,
                            round_coordinator.policyName(),
                            round_coordinator.worldSize(),
                            round_coordinator.rank(),
                            weight_bytes,
                            serial_oracle_policy,
                            serial_oracle_rows.size(),
                            preconditioning_policy.c_str(),
                            source_preconditioning_m,
                            source_preconditioning_budget_us,
                            source_preconditioning_duration_us,
                            row.timing_warmup_count,
                            row.warmup_round_count,
                            row.warmup_budget_policy.c_str(),
                            row.warmup_probe_latency_us,
                            row.warmup_budget_floor_us,
                            row.warmup_latency_multiplier,
                            row.warmup_budget_ceiling_us,
                            row.warmup_budget_us,
                            row.warmup_duration_us,
                            global_warmup_round_count,
                            warmup_wall_duration_us,
                            warmup_max_round_duration_us,
                            probed_complete_round_us,
                            effective_warmup_round_timeout_us,
                            timing_order_seed,
                            global_timing_round_count,
                            static_cast<int>(
                                row.timing.acquisition_samples_us.size()),
                            minimum_samples,
                            stationary_minimum_samples,
                            maximum_samples,
                            timing_budget_us,
                            row.timing.adaptive.measured_duration_us,
                            row.timing.adaptive.stationary_window_begin,
                            row.timing.adaptive.stationary_sample_count,
                            row.timing.adaptive.stationary_duration_us,
                            median_stability_limit,
                            row.timing.adaptive.median_relative_drift,
                            row.timing.adaptive.median_stable ? 1 : 0,
                            row.timing.stop_reason.c_str(),
                            row.timing.evidence.min,
                            row.timing.evidence.median,
                            row.timing.evidence.p95,
                            row.timing.evidence.mad,
                            row.timing.evidence.cv,
                            serial_timing.evidence.median *
                                static_cast<double>(M) /
                                static_cast<double>(serial_oracle_rows.size()),
                            row.speedup,
                            row.comparison.mismatch_count,
                            row.comparison.first_mismatch_index,
                            row.repeat_byte_mismatches,
                            row.comparison.max_abs,
                            row.comparison.relative_l2,
                            row.comparison.cosine,
                            row.comparison.symmetric_kld,
                            row.comparison.actual_digest.c_str(),
                            serial_digest.c_str(),
                            row.timing.evidence.digest.c_str(),
                            row.route_ok ? 1 : 0,
                            observed_id.c_str(),
                            row.route.k_tiles,
                            row.route.k_tile_blocks,
                            row.route.n_block_chunks,
                            row.numerical_correctness ? 1 : 0,
                            row.correctness_pass ? 1 : 0,
                            static_cast<int>(index) == best_index ? 1 : 0);
                        ++emitted_rows;
                    }
                    if (timing_csv)
                    {
                        for (size_t sample_index = 0;
                             sample_index < row.timing.acquisition_samples_us.size();
                             ++sample_index)
                        {
                            const double latency_us =
                                row.timing.acquisition_samples_us[sample_index];
                            std::fprintf(
                                timing_csv,
                                "cpu,prefill_gemm,%s,%u,%u,%s,eager,%d,%d,%d,%s,%s,%s,%s,%zu,%.9f,%a\n",
                                format.name.c_str(),
                                static_cast<unsigned>(packed.codebook_id),
                                static_cast<unsigned>(packed.codebook_id),
                                shape_name.c_str(),
                                M, N, K,
                                row.candidate.id.c_str(),
                                row.route.build_isa.c_str(),
                                requested_runtime_isa.c_str(),
                                row.route.isa.c_str(),
                                sample_index,
                                latency_us,
                                latency_us);
                        }
                    }
                }
                if (csv)
                    std::fflush(csv);
                if (timing_csv)
                    std::fflush(timing_csv);

                // Preserve the complete aggregate and acquisition-order trace
                // before rejecting an unstable cell.  The refresh transaction
                // leaves these files marked `.inprogress`, so they are usable as
                // diagnostics but can never be promoted into a dispatch table.
                bool local_failure = false;
                std::ostringstream local_diagnostic;
                for (const CandidateRow &row : rows)
                {
                    if (!row.route_ok)
                        continue;
                    const bool anchored_epoch_complete =
                        llaminar2::test::trainer::
                            anchoredCompleteRoundTimingEvidence(
                                row.timing.acquisition_samples_us.size(),
                                static_cast<size_t>(
                                    stationary_minimum_samples),
                                row.timing.adaptive.measured_duration_us,
                                timing_budget_us);
                    const bool timing_admitted = anchored_candidate_expansion
                        ? anchored_epoch_complete
                        : row.timing.adaptive.promotion_evidence;
                    if (!timing_admitted && !local_failure)
                    {
                        local_failure = true;
                        local_diagnostic
                            << "CPU prefill timing rejected: format="
                            << format.name << " M=" << M << " candidate="
                            << row.candidate.id
                            << " remained unstable at the hard timing ceiling"
                            << " acquisition_samples="
                            << row.timing.acquisition_samples_us.size()
                            << " stationary_samples="
                            << row.timing.adaptive.stationary_sample_count
                            << " acquisition_duration_us="
                            << row.timing.adaptive.measured_duration_us
                            << " stationary_duration_us="
                            << row.timing.adaptive.stationary_duration_us
                            << " drift="
                            << row.timing.adaptive.median_relative_drift
                            << " cv=" << row.timing.evidence.cv;
                    }
                }
                if (best_index < 0 && !local_failure)
                {
                    local_failure = true;
                    local_diagnostic
                        << "CPU prefill timing rejected: format=" << format.name
                        << " M=" << M
                        << " has no stable, forceable, byte-exact full-K candidate";
                }
                round_coordinator.failAllRanksIf(
                    local_failure, local_diagnostic.str());
                ++executed_cases;
            }
            const bool final_seed_class =
                format_index + 1 == selected_formats.size() ||
                selected_formats[format_index + 1]->name.size() !=
                    format.name.size();
            if (final_seed_class)
            {
                activation_fixtures.releaseFormatNameLength(
                    format.name.size());
            }
            if (executed_cases >= max_cases)
                break;
        }

        if (csv)
        {
            ASSERT_EQ(std::fclose(csv), 0);
            if (!profiling_requested)
                ASSERT_GT(emitted_rows, 0);
        }
        if (timing_csv)
            ASSERT_EQ(std::fclose(timing_csv), 0);
        ASSERT_GT(executed_cases, 0)
            << "No strong CPU prefill cases selected";
        if (!profiler_request_id.empty())
        {
            ASSERT_EQ(isolated_profile_launches, 1)
                << "each CPU prefill profiler request must launch once";
        }
        else if (profiler_batch.enabled())
        {
            EXPECT_NO_THROW(profiler_batch.requireComplete());
            ASSERT_EQ(
                static_cast<size_t>(isolated_profile_launches),
                profiler_batch.size())
                << "every CPU prefill profiler batch member must launch once";
        }
        PerfStatsCollector::reset();
    }

    TEST_F(CPUNativeVNNIGemvTest, MTP_VerifierRows_VNNIInstructionFloor_Synthetic)
    {
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        if (activeISALevel() < ISALevel::AVX512)
        {
            GTEST_SKIP() << "AVX512-VNNI instruction-floor diagnostic requires AVX512-VNNI.";
        }

        const int work_blocks =
            std::max(1, getEnvInt("LLAMINAR_CPU_NVNNI_FLOOR_BLOCKS").value_or(4096));
        const int warmup =
            std::max(0, getEnvInt("LLAMINAR_CPU_NVNNI_FLOOR_WARMUP").value_or(5));
        const int iterations =
            std::max(1, getEnvInt("LLAMINAR_CPU_NVNNI_FLOOR_ITERS").value_or(20));
        const std::string csv_path =
            getEnvString("LLAMINAR_CPU_NVNNI_FLOOR_CSV");

        std::FILE *csv = nullptr;
        if (!csv_path.empty())
        {
            csv = std::fopen(csv_path.c_str(), "w");
            ASSERT_NE(csv, nullptr)
                << "Failed to open CPU NativeVNNI floor CSV: " << csv_path;
            std::fprintf(
                csv,
                "backend,phase,m,work_blocks,grouped_min_us,serial_min_us,speedup,checksums_match\n");
        }

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header << "M" << "Blocks" << "Grouped us"
              << "Serial us" << "Speedup" << "Checksum" << fort::endr;
        for (int c = 0; c <= 5; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (int rows : {2, 3, 4})
        {
            const VNNIFloorResult result =
                runVerifierRowsVNNIFloor(rows, work_blocks, warmup, iterations);
            EXPECT_TRUE(result.checksums_match)
                << "Grouped VNNI floor checksum diverged at M=" << rows;
            EXPECT_GT(result.grouped_us, 0.0);
            EXPECT_GT(result.serial_us, 0.0);

            char grouped_us[32], serial_us[32], speedup[32];
            std::snprintf(grouped_us, sizeof(grouped_us), "%.3f", result.grouped_us);
            std::snprintf(serial_us, sizeof(serial_us), "%.3f", result.serial_us);
            std::snprintf(speedup, sizeof(speedup), "%.3fx", result.speedup);
            table << rows << work_blocks << grouped_us << serial_us
                  << speedup << (result.checksums_match ? "yes" : "no")
                  << fort::endr;

            std::fprintf(
                stderr,
                "[CPUNativeVNNI][VNNI_FLOOR] M=%d blocks=%d grouped_us=%.3f serial_us=%.3f speedup=%.3f checksums_match=%d\n",
                rows,
                work_blocks,
                result.grouped_us,
                result.serial_us,
                result.speedup,
                result.checksums_match ? 1 : 0);

            if (csv)
            {
                std::fprintf(
                    csv,
                    "cpu,vnni_instruction_floor,%d,%d,%.3f,%.3f,%.6f,%d\n",
                    rows,
                    work_blocks,
                    result.grouped_us,
                    result.serial_us,
                    result.speedup,
                    result.checksums_match ? 1 : 0);
                std::fflush(csv);
            }
        }

        std::cout << "\n=== CPU NativeVNNI Verifier Rows: VPDPBUSD Instruction Floor ===\n"
                  << "Diagnostic only: packed-B reuse without GEMV metadata, scaling, or output writes.\n\n"
                  << table.to_string() << std::endl;

        if (csv)
            std::fclose(csv);
#else
        GTEST_SKIP() << "AVX512-VNNI instruction-floor diagnostic is unavailable in this build.";
#endif
    }

} // anonymous namespace
