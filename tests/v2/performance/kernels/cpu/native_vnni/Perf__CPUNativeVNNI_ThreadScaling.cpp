/**
 * @file Perf__CPUNativeVNNI_ThreadScaling.cpp
 * @brief Measures production CPU NativeVNNI scaling from one thread through
 *        one physical socket for decode, grouped verification, and prefill.
 *
 * Each selected format keeps one prepared projection resident while varying
 * only the OpenMP team width. Each regime enters its production `Auto`
 * dispatcher:
 *
 * - M=1 uses the decode policy (x86 generated table; ARM native total policy).
 * - M=4 and M=15 use the decode-equivalent verifier policy.
 * - M=64 uses the ordinary prefill policy.
 *
 * Every timed configuration is authenticated outside the timing interval:
 * its complete output bank must equal the serial M=1 production rows computed
 * with one worker, byte for byte. A scaling point therefore also proves thread
 * width and batch invariance of the exact arithmetic that was timed.
 *
 * Every M>1 point also times M independent production serial M=1 calls at the
 * same team width, so grouped/prefill amortization is a measured ratio over
 * the exact serial arithmetic it reproduces.
 *
 * The default N=7168, K=5120 projection has a prepared weight footprint larger
 * than the last-level cache of the reference Xeon socket and of GB10. It
 * exposes decode bandwidth saturation while giving grouped decode and prefill
 * enough independent work to reveal compute scaling.
 *
 * Run on physical cores of one socket/package, for example:
 *
 * @code
 * OMP_NUM_THREADS=28 OMP_PLACES=cores OMP_PROC_BIND=close taskset -c 0-27 \
 *   ./build_v2_release_avx512/tests/v2/v2_perf_cpu_native_vnni_thread_scaling
 * OMP_NUM_THREADS=20 OMP_PLACES=cores OMP_PROC_BIND=close \
 *   LLAMINAR_CPU_THREAD_SCALING_FORMATS=all \
 *   ./build_v2_release_neon/tests/v2/v2_perf_cpu_native_vnni_thread_scaling
 * @endcode
 *
 * Environment overrides:
 *
 * - `LLAMINAR_CPU_THREAD_SCALING_FORMATS` (CSV labels or `all`; default Q4_K)
 * - `LLAMINAR_CPU_THREAD_SCALING_THREADS` (CSV widths; default every width
 *   from one through `omp_get_max_threads()`)
 * - `LLAMINAR_CPU_THREAD_SCALING_N`
 * - `LLAMINAR_CPU_THREAD_SCALING_K`
 * - `LLAMINAR_CPU_THREAD_SCALING_PREFILL_M`
 * - `LLAMINAR_CPU_THREAD_SCALING_WARMUP`
 * - `LLAMINAR_CPU_THREAD_SCALING_ITERS`
 */

#include "../../../../utils/NativeVNNITestPartialStorage.h"
#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "utils/QuantizedVerifierFormats.h"
#include "utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::cpu::native_vnni;
using namespace llaminar2::test;

namespace
{

/**
 * @brief Read a positive integer environment override.
 *
 * Invalid values fail the benchmark instead of silently selecting a different
 * workload. This keeps command lines and reported geometry unambiguous.
 */
int positiveEnvironmentValue(const char *name, int default_value)
{
    const char *raw = std::getenv(name);
    if (!raw || *raw == '\0')
        return default_value;

    char *end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || parsed <= 0 ||
        parsed > std::numeric_limits<int>::max())
    {
        throw std::invalid_argument(
            std::string(name) + " must be a positive integer");
    }
    return static_cast<int>(parsed);
}

/**
 * @brief Return every requested OpenMP team width.
 *
 * The default is every width from one through the configured maximum, so a
 * scaling curve cannot hide a regression between sampled points. An explicit
 * CSV restricts the sweep for diagnosis; it never adds widths above the
 * configured team limit.
 */
std::vector<int> threadSamples(int maximum_threads)
{
    std::vector<int> result;
    const char *raw = std::getenv("LLAMINAR_CPU_THREAD_SCALING_THREADS");
    if (!raw || *raw == '\0')
    {
        for (int threads = 1; threads <= maximum_threads; ++threads)
            result.push_back(threads);
        return result;
    }
    std::stringstream stream(raw);
    std::string token;
    while (std::getline(stream, token, ','))
    {
        char *end = nullptr;
        const long parsed = std::strtol(token.c_str(), &end, 10);
        if (token.empty() || *end != '\0' || parsed <= 0 || parsed > maximum_threads)
            throw std::invalid_argument(
                "LLAMINAR_CPU_THREAD_SCALING_THREADS entries must be in 1..max threads");
        result.push_back(static_cast<int>(parsed));
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

/**
 * @brief Resolve the requested canonical source formats.
 * @return Registry entries in canonical order; unknown labels are fatal.
 */
std::vector<const QuantizedVerifierFormatCase *> selectedFormats()
{
    const char *raw = std::getenv("LLAMINAR_CPU_THREAD_SCALING_FORMATS");
    const std::string request = raw && *raw ? raw : "Q4_K";
    std::vector<const QuantizedVerifierFormatCase *> result;
    if (request == "all")
    {
        for (const auto &format : quantizedVerifierFormats())
            result.push_back(&format);
        return result;
    }
    std::stringstream stream(request);
    std::string token;
    while (std::getline(stream, token, ','))
    {
        const auto &formats = quantizedVerifierFormats();
        const auto found = std::find_if(formats.begin(), formats.end(),
            [&](const auto &format) { return token == format.label; });
        if (found == formats.end())
            throw std::invalid_argument("Unknown thread-scaling format: " + token);
        result.push_back(&*found);
    }
    return result;
}

/** @brief Summary of one stable thread-count timing series. */
struct TimingSummary
{
    double median_us = 0.0;
    double minimum_us = 0.0;
    double p95_us = 0.0;
};

/**
 * @brief Time a steady-state production launch without setup work.
 *
 * Warmup establishes the requested OpenMP team and grows any persistent
 * thread-local workspace before the timer starts. The measured region contains
 * only a clock read, one kernel invocation, and a second clock read.
 */
template <typename Launch>
TimingSummary measureLaunch(int warmup_count, int iteration_count, Launch launch)
{
    for (int warmup = 0; warmup < warmup_count; ++warmup)
        launch();

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iteration_count));
    for (int iteration = 0; iteration < iteration_count; ++iteration)
    {
        const auto begin = std::chrono::steady_clock::now();
        launch();
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(
            std::chrono::duration<double, std::micro>(end - begin).count());
    }
    std::sort(samples.begin(), samples.end());

    const size_t median_index = samples.size() / 2;
    const size_t p95_index = std::min(
        samples.size() - 1,
        static_cast<size_t>(std::ceil(samples.size() * 0.95)) - 1);
    return {
        .median_us = samples[median_index],
        .minimum_us = samples.front(),
        .p95_us = samples[p95_index],
    };
}

/**
 * @brief Prove libgomp created the requested team before timing kernels.
 */
int observeOpenMPTeamSize()
{
    int observed = 0;
#pragma omp parallel shared(observed)
    {
#pragma omp single
        observed = omp_get_num_threads();
    }
    return observed;
}

/** @brief One operation measured at one OpenMP thread count. */
struct ScalingPoint
{
    std::string format;
    std::string regime;
    std::string policy;
    int m = 0;
    int threads = 0;
    int n_block_chunks = 0;
    int serial_k_tiles = 0;
    int regime_k_tiles = 0;
    TimingSummary timing;
    double throughput_gops = 0.0;
    double speedup = 0.0;
    double parallel_efficiency = 0.0;
    double fraction_of_max_throughput = 0.0;
    /// Median of M independent production serial M=1 calls at the same width
    /// (zero for the M=1 regime itself).
    double serial_rows_median_us = 0.0;
    /// serial_rows_median_us / timing.median_us: grouped/prefill amortization.
    double grouped_vs_serial_speedup = 0.0;
};

/**
 * @brief Fill deterministic FP32 activations without timing their quantization.
 */
std::vector<Q8_1Block> makeActivations(int rows, int k, int k_blocks)
{
    std::vector<float> fp32(static_cast<size_t>(rows) * k);
    for (size_t index = 0; index < fp32.size(); ++index)
    {
        const int centered = static_cast<int>((index * 17 + 11) % 251) - 125;
        fp32[index] = static_cast<float>(centered) / 127.0f;
    }

    std::vector<Q8_1Block> quantized(
        static_cast<size_t>(rows) * static_cast<size_t>(k_blocks));
    quantize_activations_to_q8_1(
        fp32.data(), quantized.data(), rows, k, k_blocks);
    return quantized;
}

/**
 * @brief Describe the production schedule the Auto dispatcher resolves.
 *
 * x86 reports its generated decode/verifier policy names. The ARM native
 * total policy is a fixed shared-row program: four-row tiles over full K with
 * one 64-column chunk per task for every Auto regime.
 */
std::string resolvedPolicyName(
    const CPUNativeVNNIPackedWeights &packed, int m, int n, int k, int threads, bool grouped)
{
#if defined(__aarch64__) && defined(__ARM_NEON)
    (void)packed; (void)n; (void)k; (void)threads;
    const auto tile = neon_execution::rowTile(VerifierRowsPolicy::Auto);
    return std::string(m == 1 ? "neon_decode" : (grouped ? "neon_grouped" : "neon_prefill")) +
        (tile == NEONProjectionRowTile::Four ? "_rows4" : "_rows2") + "_nbc" +
        std::to_string(neon_execution::nChunks(DecodeSchedulePolicy::Auto));
#else
    if (m == 1)
    {
        const NativeVNNITileConfig serial_geometry = computeTileConfig(
            n, k, 1, packed.preparedFootprint(), threads);
        const ISALevel runtime_isa = activeISALevel();
        return decodeSchedulePolicyName(selectDecodeSchedulePolicy(
            packed, n, k, runtime_isa == ISALevel::AVX512, runtime_isa == ISALevel::AVX2,
            serial_geometry.k_tiles > 1, serial_geometry.k_tiles));
    }
    if (grouped)
        return verifierRowsPolicyName(selectVerifierRowsPolicy(packed, m, n, k));
    return "Auto";
#endif
}

/**
 * @brief Measure all production arithmetic regimes across one socket.
 */
TEST(Perf__CPUNativeVNNIThreadScaling, ProductionAutoRoutes)
{
    omp_set_dynamic(0);
    const int maximum_threads = omp_get_max_threads();
    ASSERT_GT(maximum_threads, 0);

    const int n =
        positiveEnvironmentValue("LLAMINAR_CPU_THREAD_SCALING_N", 7168);
    const int k =
        positiveEnvironmentValue("LLAMINAR_CPU_THREAD_SCALING_K", 5120);
    const int prefill_m = positiveEnvironmentValue(
        "LLAMINAR_CPU_THREAD_SCALING_PREFILL_M", 64);
    const int warmup_count = positiveEnvironmentValue(
        "LLAMINAR_CPU_THREAD_SCALING_WARMUP", 3);
    const int iteration_count = positiveEnvironmentValue(
        "LLAMINAR_CPU_THREAD_SCALING_ITERS", 9);
    ASSERT_EQ(k % 256, 0)
        << "thread-scaling K must be superblock aligned for every format";
    const std::vector<int> thread_widths = threadSamples(maximum_threads);

    struct Regime
    {
        std::string_view name;
        int m;
    };
    const std::vector<Regime> regimes = {
        {"gemv_decode", 1},
        {"grouped_decode", 4},
        {"grouped_decode", 15},
        {"gemm_prefill", prefill_m},
    };
    const int maximum_m = std::max(prefill_m, 15);

    std::vector<ScalingPoint> points;
    for (const QuantizedVerifierFormatCase *format : selectedFormats())
    {
        SCOPED_TRACE(format->label);
        auto weights = format->create(
            {static_cast<size_t>(n), static_cast<size_t>(k)}, 0x51A1u);
        ASSERT_NE(weights, nullptr);
        CPUNativeVNNIGemmKernel kernel(weights.get());
        ASSERT_TRUE(kernel.isValid());
        const CPUNativeVNNIPackedWeights &packed = kernel.packedWeights();

        const std::vector<Q8_1Block> activations =
            makeActivations(maximum_m, k, packed.blocks_per_row);
        std::vector<float> output(
            static_cast<size_t>(maximum_m) * static_cast<size_t>(n), 0.0f);

        // Serial production rows with one worker are the byte oracle for every
        // regime and team width; computed once, outside every timing interval.
        std::vector<float> serial_rows(output.size(), 0.0f);
        {
            omp_set_num_threads(1);
            NativeVNNITestPartialStorage serial_partials(packed, 1);
            for (int row = 0; row < maximum_m; ++row)
                gemv_native_vnni_preq(packed,
                    activations.data() + static_cast<size_t>(row) * packed.blocks_per_row,
                    serial_rows.data() + static_cast<size_t>(row) * n,
                    serial_partials.span(), ISAPath::AUTO, DecodeSchedulePolicy::Auto);
        }

        for (const Regime &regime : regimes)
        {
            const size_t first_point = points.size();
            const bool grouped = regime.name == "grouped_decode";
            for (const int threads : thread_widths)
            {
                omp_set_num_threads(threads);
                ASSERT_EQ(observeOpenMPTeamSize(), threads)
                    << "OpenMP runtime did not honor the requested team width";

#if defined(__aarch64__) && defined(__ARM_NEON)
                const int n_block_chunks = neon_execution::nChunks(DecodeSchedulePolicy::Auto);
                const int serial_k_tiles = 1; // Native ARM programs reduce full K in registers.
                const int regime_k_tiles = 1;
#else
                const NativeVNNITileConfig serial_geometry = computeTileConfig(
                    n, k, 1, packed.preparedFootprint(), threads);
                const NativeVNNITileConfig regime_geometry = computeTileConfig(
                    n, k, regime.m, packed.preparedFootprint(), threads);
                const int n_block_chunks = regime_geometry.n_block_chunks;
                const int serial_k_tiles = serial_geometry.k_tiles;
                const int regime_k_tiles = regime_geometry.k_tiles;
#endif
                std::string policy = resolvedPolicyName(packed, regime.m, n, k, threads, grouped);

                NativeVNNITestPartialStorage partial_storage(packed, regime.m);
                const auto launch = [&]()
                {
                    if (regime.m == 1)
                    {
                        gemv_native_vnni_preq(
                            packed,
                            activations.data(),
                            output.data(), partial_storage.span(),
                            ISAPath::AUTO,
                            DecodeSchedulePolicy::Auto);
                        return;
                    }
                    if (grouped)
                    {
                        gemm_native_vnni_preq_decode_equivalent_rows(
                            packed,
                            activations.data(),
                            output.data(), partial_storage.span(),
                            regime.m,
                            n,
                            ISAPath::AUTO,
                            VerifierRowsPolicy::Auto);
                        return;
                    }
                    gemm_native_vnni_preq(
                        packed,
                        activations.data(),
                        output.data(), partial_storage.span(),
                        regime.m,
                        n,
                        ISAPath::AUTO,
                        VerifierRowsPolicy::Auto,
                        PrefillSchedulePolicy::Auto);
                };

                std::fill(output.begin(), output.end(), 0.0f);
                const TimingSummary timing =
                    measureLaunch(warmup_count, iteration_count, launch);
                ASSERT_GT(timing.median_us, 0.0);
                ASSERT_EQ(std::memcmp(output.data(), serial_rows.data(),
                              static_cast<size_t>(regime.m) * n * sizeof(float)), 0)
                    << format->label << ' ' << regime.name << " M=" << regime.m
                    << " threads=" << threads
                    << " must publish the serial one-worker M=1 bytes";
                // Same-width baseline: M independent production serial calls,
                // the arithmetic every grouped/prefill row must reproduce.
                double serial_rows_median_us = 0.0;
                if (regime.m > 1)
                {
                    NativeVNNITestPartialStorage serial_partials(packed, 1);
                    const auto serial_launch = [&]()
                    {
                        for (int row = 0; row < regime.m; ++row)
                            gemv_native_vnni_preq(packed,
                                activations.data() + static_cast<size_t>(row) * packed.blocks_per_row,
                                output.data() + static_cast<size_t>(row) * n,
                                serial_partials.span(), ISAPath::AUTO, DecodeSchedulePolicy::Auto);
                    };
                    serial_rows_median_us =
                        measureLaunch(warmup_count, iteration_count, serial_launch).median_us;
                    ASSERT_EQ(std::memcmp(output.data(), serial_rows.data(),
                                  static_cast<size_t>(regime.m) * n * sizeof(float)), 0)
                        << format->label << " serial baseline M=" << regime.m
                        << " threads=" << threads << " changed bytes across team widths";
                }
                const double operations =
                    2.0 * static_cast<double>(regime.m) *
                    static_cast<double>(n) * static_cast<double>(k);
                points.push_back({
                    .format = format->label,
                    .regime = std::string(regime.name),
                    .policy = std::move(policy),
                    .m = regime.m,
                    .threads = threads,
                    .n_block_chunks = n_block_chunks,
                    .serial_k_tiles = serial_k_tiles,
                    .regime_k_tiles = regime_k_tiles,
                    .timing = timing,
                    .throughput_gops = operations / (timing.median_us * 1.0e3),
                    .serial_rows_median_us = serial_rows_median_us,
                    .grouped_vs_serial_speedup = serial_rows_median_us > 0.0
                        ? serial_rows_median_us / timing.median_us : 0.0,
                });
            }

            const double single_thread_us = points[first_point].timing.median_us;
            double maximum_throughput = 0.0;
            for (size_t index = first_point; index < points.size(); ++index)
                maximum_throughput = std::max(maximum_throughput, points[index].throughput_gops);
            for (size_t index = first_point; index < points.size(); ++index)
            {
                ScalingPoint &point = points[index];
                point.speedup = single_thread_us / point.timing.median_us;
                point.parallel_efficiency =
                    point.speedup / static_cast<double>(point.threads);
                point.fraction_of_max_throughput =
                    point.throughput_gops / maximum_throughput;
            }
        }
    }

    std::cout
        << "\nCPU NativeVNNI production thread scaling"
        << " (N=" << n << " K=" << k
        << ", isa=" << isaLevelName(activeISALevel())
        << ", max_threads=" << maximum_threads << ")\n"
        << "format,regime,policy,M,threads,nbc,serial_k_tiles,regime_k_tiles,"
           "median_us,min_us,p95_us,gops,speedup,efficiency_pct,"
           "fraction_of_max_pct,serial_rows_median_us,grouped_vs_serial\n";
    for (const ScalingPoint &point : points)
    {
        std::cout
            << point.format << ','
            << point.regime << ',' << point.policy << ','
            << point.m << ',' << point.threads << ','
            << point.n_block_chunks << ','
            << point.serial_k_tiles << ','
            << point.regime_k_tiles << ','
            << std::fixed << std::setprecision(3)
            << point.timing.median_us << ','
            << point.timing.minimum_us << ','
            << point.timing.p95_us << ','
            << point.throughput_gops << ','
            << point.speedup << ','
            << point.parallel_efficiency * 100.0 << ','
            << point.fraction_of_max_throughput * 100.0 << ','
            << point.serial_rows_median_us << ','
            << point.grouped_vs_serial_speedup << '\n';
    }

    omp_set_num_threads(maximum_threads);
}

} // namespace
