/**
 * @file Perf__CPUNativeVNNI_ThreadScaling.cpp
 * @brief Measures production CPU NativeVNNI scaling from one thread through
 *        one physical socket for decode, grouped verification, and prefill.
 *
 * This benchmark intentionally keeps one prepared Q4_K projection resident
 * while varying only the OpenMP team width. Each regime enters its production
 * `Auto` dispatcher:
 *
 * - M=1 uses the generated decode policy.
 * - M=4 and M=15 use the generated decode-equivalent verifier policy.
 * - M=64 uses the ordinary prefill heuristic.
 *
 * The default N=7168, K=5120 projection has a prepared weight footprint larger
 * than the last-level cache of the reference Xeon socket. It therefore exposes
 * the bandwidth saturation expected from decode while still giving grouped
 * decode and prefill enough independent work to reveal compute scaling.
 *
 * Run the benchmark on physical cores from one socket:
 *
 * @code
 * OMP_NUM_THREADS=28 OMP_PLACES=cores OMP_PROC_BIND=close \
 *   taskset -c 0-27 \
 *   ./build_v2_release_avx512/tests/v2/v2_perf_cpu_native_vnni_thread_scaling
 * @endcode
 *
 * Environment overrides:
 *
 * - `LLAMINAR_CPU_THREAD_SCALING_N`
 * - `LLAMINAR_CPU_THREAD_SCALING_K`
 * - `LLAMINAR_CPU_THREAD_SCALING_PREFILL_M`
 * - `LLAMINAR_CPU_THREAD_SCALING_WARMUP`
 * - `LLAMINAR_CPU_THREAD_SCALING_ITERS`
 */

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
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
 * @brief Return logarithmic and socket-fraction thread samples.
 *
 * For the reference 28-core socket this produces exactly
 * `{1, 2, 4, 7, 14, 28}`. Duplicate values are removed for smaller machines,
 * which keeps this diagnostic useful on developer workstations as well.
 */
std::vector<int> threadSamples(int maximum_threads)
{
    std::vector<int> result = {
        1,
        std::min(2, maximum_threads),
        std::min(4, maximum_threads),
        std::max(1, maximum_threads / 4),
        std::max(1, maximum_threads / 2),
        maximum_threads,
    };
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
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
        << "Q4_K thread-scaling K must be superblock aligned";

    auto weights = TestTensorFactory::createQ4_KRandom(
        {static_cast<size_t>(n), static_cast<size_t>(k)}, 0x51A1u);
    ASSERT_NE(weights, nullptr);
    CPUNativeVNNIGemmKernel kernel(weights.get());
    ASSERT_TRUE(kernel.isValid());
    const CPUNativeVNNIPackedWeights &packed = kernel.packedWeights();

    const int maximum_m = std::max(prefill_m, 15);
    const std::vector<Q8_1Block> activations =
        makeActivations(maximum_m, k, packed.blocks_per_row);
    std::vector<float> output(
        static_cast<size_t>(maximum_m) * static_cast<size_t>(n), 0.0f);

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

    std::vector<ScalingPoint> points;
    for (const Regime &regime : regimes)
    {
        const size_t first_point = points.size();
        for (const int threads : threadSamples(maximum_threads))
        {
            omp_set_num_threads(threads);
            ASSERT_EQ(observeOpenMPTeamSize(), threads)
                << "OpenMP runtime did not honor the requested team width";

            const NativeVNNITileConfig serial_geometry = computeTileConfig(
                n, k, 1, packed.payload_bytes, threads);
            const NativeVNNITileConfig regime_geometry = computeTileConfig(
                n, k, regime.m, packed.payload_bytes, threads);
            std::string policy = "Auto";
            if (regime.m == 1)
            {
                const ISALevel runtime_isa = activeISALevel();
                const DecodeSchedulePolicy selected =
                    selectDecodeSchedulePolicy(
                        packed,
                        n,
                        k,
                        runtime_isa == ISALevel::AVX512,
                        runtime_isa == ISALevel::AVX2,
                        serial_geometry.k_tiles > 1,
                        serial_geometry.k_tiles);
                policy = decodeSchedulePolicyName(selected);
            }
            else if (regime.name == "grouped_decode")
            {
                policy = verifierRowsPolicyName(
                    selectVerifierRowsPolicy(packed, regime.m, n, k));
            }

            const auto launch = [&]()
            {
                if (regime.m == 1)
                {
                    gemv_native_vnni_preq(
                        packed,
                        activations.data(),
                        output.data(),
                        ISAPath::AUTO,
                        DecodeSchedulePolicy::Auto);
                    return;
                }
                if (regime.name == "grouped_decode")
                {
                    gemm_native_vnni_preq_decode_equivalent_rows(
                        packed,
                        activations.data(),
                        output.data(),
                        regime.m,
                        n,
                        ISAPath::AUTO,
                        VerifierRowsPolicy::Auto);
                    return;
                }
                gemm_native_vnni_preq(
                    packed,
                    activations.data(),
                    output.data(),
                    regime.m,
                    n,
                    ISAPath::AUTO,
                    VerifierRowsPolicy::Auto,
                    PrefillSchedulePolicy::Auto);
            };

            const TimingSummary timing =
                measureLaunch(warmup_count, iteration_count, launch);
            ASSERT_GT(timing.median_us, 0.0);
            const double operations =
                2.0 * static_cast<double>(regime.m) *
                static_cast<double>(n) * static_cast<double>(k);
            points.push_back({
                .regime = std::string(regime.name),
                .policy = std::move(policy),
                .m = regime.m,
                .threads = threads,
                .n_block_chunks = regime_geometry.n_block_chunks,
                .serial_k_tiles = serial_geometry.k_tiles,
                .regime_k_tiles = regime_geometry.k_tiles,
                .timing = timing,
                .throughput_gops = operations / (timing.median_us * 1.0e3),
            });
        }

        const double single_thread_us = points[first_point].timing.median_us;
        const double maximum_throughput = points.back().throughput_gops;
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

    std::cout
        << "\nCPU NativeVNNI production thread scaling"
        << " (Q4_K N=" << n << " K=" << k
        << ", prepared_weight_bytes="
        << packed.native_interleaved.size() + packed.payload.size()
        << ", max_threads=" << maximum_threads << ")\n"
        << "regime,policy,M,threads,nbc,serial_k_tiles,regime_k_tiles,"
           "median_us,min_us,p95_us,gops,speedup,efficiency_pct,"
           "fraction_of_max_pct\n";
    for (const ScalingPoint &point : points)
    {
        std::cout
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
            << point.fraction_of_max_throughput * 100.0 << '\n';
    }

    omp_set_num_threads(maximum_threads);
}

} // namespace
