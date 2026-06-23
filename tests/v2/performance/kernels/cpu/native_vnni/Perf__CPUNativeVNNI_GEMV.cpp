/**
 * @file Perf__CPUNativeVNNI_GEMV.cpp
 * @brief Comprehensive GEMV decode performance sweep across all Qwen model
 *        sizes (0.5B through 32B), TP degrees (1/2/4), and both quantized
 *        GEMV paths: packed VNNI and native Q8_0.
 *
 * For each (model, shape, TP degree):
 *   1. Report tile config (nbc, k_tiles, category)
 *   2. Benchmark packed VNNI GEMV (FP32→Q8_1 quantize + INT8 packed access)
 *   3. Benchmark native Q8_0 GEMV (direct block access, FP32 dequant + FMA)
 *   4. Report latency, effective BW, roofline %, and speedup
 *
 * NOTE: The VNNI column calls gemv_native_vnni() directly (which includes
 * activation quantization + packed INT8 VNNI compute). This gives a real
 * comparison vs native Q8_0, rather than going through multiply_tensor()
 * which short-circuits to the native Q8_0 path for codebook_id==19.
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
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <cstring>
#include <iomanip>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h"
#include "kernels/cpu/native_vnni/CPUNativeVNNITileConfig.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"
#include "fort.hpp"

#include "utils/TestTensorFactory.h"

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
        {"Q8_0"}, {"Q8_1"}, {"Q2_K"}, {"Q3_K"},
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
        // Native Q8_0 GEMV (new: direct block access, FP32 compute)
        double native_us;
        double native_bw_gbs;
        // Roofline and comparison
        double roofline_bw;
        double packed_roof_pct;
        double native_roof_pct;
        double speedup; // packed_us / native_us
        // Weight data sizes
        double packed_bytes; // VNNI packed buffer
        double native_bytes; // Q8_0 native blocks
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

    /**
     * @brief Benchmark native Q8_0 GEMV via unified dispatch, return p10 latency (us).
     *
     * Calls gemv_native_vnni() with Q8_0 blocks pointer, which dispatches to
     * the native Q8_0 path (direct block access + VNNI vpdpbusd).
     *
     * Flushes the Q8_0 weight blocks from cache before each timed iteration
     * to measure true DRAM bandwidth efficiency without cache effects.
     */
    double benchNativeQ8_0(const CPUNativeVNNIPackedWeights &packed,
                           const Q8_0Block *blocks, const float *A,
                           float *C, int N, int bpr)
    {
        // Weight data: N rows × bpr blocks × 34 bytes/block
        size_t weight_bytes = static_cast<size_t>(N) * bpr * sizeof(Q8_0Block);

        for (int i = 0; i < WARMUP; ++i)
            gemv_native_vnni(packed, A, C);

        std::vector<double> times(ITERS);
        for (int i = 0; i < ITERS; ++i)
        {
            flush_cache_range(blocks, weight_bytes);
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
        // Native Q8_0: 34 bytes per block
        r.native_bytes = (double)shape.N * bpr * 34.0;

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

        // --- Native Q8_0 GEMV (direct block access, FP32 dequant) ---
        {
            std::vector<float> C(shape.N, 0.0f);
            r.native_us = benchNativeQ8_0(
                packed_kernel.packedWeights(), q8_tensor->typed_data(),
                A.data(), C.data(), shape.N, bpr);
            r.native_bw_gbs = r.native_bytes / (r.native_us * 1e-6) / 1e9;
            r.native_roof_pct = r.native_bw_gbs / roofline_bw * 100.0;
        }

        // Speedup of native over packed
        r.speedup = (r.native_us > 0) ? r.packed_us / r.native_us : 0;

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
              << "VNNI \xc2\xb5s" << "VNNI BW" << "VNNI R%"
              << "Q8_0 \xc2\xb5s" << "Q8_0 BW" << "Q8_0 R%"
              << "Speedup"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::left);
        table.column(2).set_cell_text_align(fort::text_align::left);
        table.column(5).set_cell_text_align(fort::text_align::left);
        for (int c : {3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14})
            table.column(c).set_cell_text_align(fort::text_align::right);

        double sum_packed = 0, sum_native = 0;
        double sum_packed_bytes = 0, sum_native_bytes = 0;
        std::string last_cat;

        for (const auto &r : results)
        {
            if (!last_cat.empty() && r.category != last_cat)
                table << fort::separator;
            last_cat = r.category;

            char pu[16], pbw[16], pr[16], nu[16], nbw[16], nr[16], sp[16];
            std::snprintf(pu, sizeof(pu), "%.0f", r.packed_us);
            std::snprintf(pbw, sizeof(pbw), "%.1f", r.packed_bw_gbs);
            std::snprintf(pr, sizeof(pr), "%.0f%%", r.packed_roof_pct);
            std::snprintf(nu, sizeof(nu), "%.0f", r.native_us);
            std::snprintf(nbw, sizeof(nbw), "%.1f", r.native_bw_gbs);
            std::snprintf(nr, sizeof(nr), "%.0f%%", r.native_roof_pct);
            std::snprintf(sp, sizeof(sp), "%.2fx", r.speedup);

            table << r.shape_name << r.category << r.shard
                  << r.N << r.K
                  << r.tile_cat << r.nbc << r.k_tiles
                  << pu << pbw << pr
                  << nu << nbw << nr
                  << sp
                  << fort::endr;

            sum_packed += r.packed_us;
            sum_native += r.native_us;
            sum_packed_bytes += r.packed_bytes;
            sum_native_bytes += r.native_bytes;
        }

        // Summary row
        table << fort::separator;
        double avg_packed_bw = sum_packed_bytes / (sum_packed * 1e-6) / 1e9;
        double avg_native_bw = sum_native_bytes / (sum_native * 1e-6) / 1e9;
        char tpu[16], tpbw[16], tpr[16], tnu[16], tnbw[16], tnr[16], tsp[16];
        std::snprintf(tpu, sizeof(tpu), "%.0f", sum_packed);
        std::snprintf(tpbw, sizeof(tpbw), "%.1f", avg_packed_bw);
        std::snprintf(tpr, sizeof(tpr), "%.0f%%", avg_packed_bw / roofline * 100);
        std::snprintf(tnu, sizeof(tnu), "%.0f", sum_native);
        std::snprintf(tnbw, sizeof(tnbw), "%.1f", avg_native_bw);
        std::snprintf(tnr, sizeof(tnr), "%.0f%%", avg_native_bw / roofline * 100);
        std::snprintf(tsp, sizeof(tsp), "%.2fx", sum_packed / sum_native);

        table << "TOTAL" << "" << "" << "" << "" << "" << "" << ""
              << tpu << tpbw << tpr
              << tnu << tnbw << tnr
              << tsp
              << fort::endr;

        std::cout << "\n"
                  << title << "\n"
                  << "Threads: " << omp_get_max_threads()
                  << "  |  Roofline: " << std::fixed << std::setprecision(1) << roofline << " GB/s (calibrated)"
                  << "  |  Cold cache (clflushopt before each iteration)"
                  << "  |  Q8_0 = native blocks (VNNI vpdpbusd)\n\n"
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
              << "VNNI ms/tok" << "VNNI tok/s"
              << "Q8_0 ms/tok" << "Q8_0 tok/s"
              << "Speedup"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 6; ++c)
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
            double layer_packed_us = 0, layer_native_us = 0;
            double lm_packed_us = 0, lm_native_us = 0;

            for (const auto &r : results)
            {
                if (r.category == "LM_Head")
                {
                    lm_packed_us += r.packed_us;
                    lm_native_us += r.native_us;
                }
                else
                {
                    layer_packed_us += r.packed_us;
                    layer_native_us += r.native_us;
                }
            }

            double packed_ms = (layer_packed_us * layers + lm_packed_us) / 1000.0;
            double native_ms = (layer_native_us * layers + lm_native_us) / 1000.0;

            char pm[16], pt[16], nm[16], nt[16], sp[16];
            std::snprintf(pm, sizeof(pm), "%.1f", packed_ms);
            std::snprintf(pt, sizeof(pt), "%.1f", 1000.0 / packed_ms);
            std::snprintf(nm, sizeof(nm), "%.1f", native_ms);
            std::snprintf(nt, sizeof(nt), "%.1f", 1000.0 / native_ms);
            std::snprintf(sp, sizeof(sp), "%.2fx", packed_ms / native_ms);

            table << model_name << layers
                  << pm << pt
                  << nm << nt
                  << sp
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
            double eff2 = (tp1[i].native_us / 2.0) / tp2[i].native_us;
            double eff4 = (tp1[i].native_us / 4.0) / tp4[i].native_us;

            char nk1[24], u1[16], nk2[24], u2[16], e2[16], nk4[24], u4[16], e4[16];
            std::snprintf(nk1, sizeof(nk1), "%d\xc3\x97%d", tp1[i].N, tp1[i].K);
            std::snprintf(u1, sizeof(u1), "%.0f", tp1[i].native_us);
            std::snprintf(nk2, sizeof(nk2), "%d\xc3\x97%d", tp2[i].N, tp2[i].K);
            std::snprintf(u2, sizeof(u2), "%.0f", tp2[i].native_us);
            std::snprintf(e2, sizeof(e2), "%.0f%%", eff2 * 100);
            std::snprintf(nk4, sizeof(nk4), "%d\xc3\x97%d", tp4[i].N, tp4[i].K);
            std::snprintf(u4, sizeof(u4), "%.0f", tp4[i].native_us);
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
                  << "Native Q8_0 GEMV only\n\n"
                  << table.to_string() << std::endl;
    }

    // =========================================================================
    // Test fixture
    // =========================================================================
    class CPUNativeVNNIGemvTest : public ::testing::Test
    {
    };

    // =========================================================================
    // TEST 1: Q8_0 All Models — Full shape sweep (TP=1)
    //
    // Comprehensive decode GEMV benchmark across all Qwen model sizes.
    // Compares packed VNNI (production) vs native Q8_0 for each shape.
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
              << "Quant \xc2\xb5s" << "VNNI \xc2\xb5s" << "Q8_0 \xc2\xb5s"
              << "Quant %" << "Speedup"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 7; ++c)
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

            // Full packed GEMV and native GEMV
            auto r = benchShape(shape, roofline);

            double quant_pct = (r.packed_us > 0) ? (quant_us / r.packed_us) * 100.0 : 0;

            char qu[16], pu[16], nu[16], qp[16], sp[16];
            std::snprintf(qu, sizeof(qu), "%.1f", quant_us);
            std::snprintf(pu, sizeof(pu), "%.0f", r.packed_us);
            std::snprintf(nu, sizeof(nu), "%.0f", r.native_us);
            std::snprintf(qp, sizeof(qp), "%.1f%%", quant_pct);
            std::snprintf(sp, sizeof(sp), "%.2fx", r.speedup);

            std::string sname = shape.name;
            auto pos = sname.find('_');
            if (pos != std::string::npos)
                sname = sname.substr(pos + 1);

            table << sname << K << K_blocks
                  << qu << pu << nu
                  << qp << sp
                  << fort::endr;
        }

        std::cout << "\n=== Quantization Overhead Isolation: Qwen 7B ===\n"
                  << "Quant = FP32\xe2\x86\x92Q8_1 activation quantization only\n"
                  << "VNNI = packed INT8 path (FP32\xe2\x86\x92Q8_1 quant + VNNI compute)\n"
                  << "Q8_0 = native blocks (FP32 dequant + FMA)\n\n"
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
