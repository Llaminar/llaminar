/**
 * @file Perf__CPUNativeQuantised_DecodeBandwidth.cpp
 * @brief Portable M=1 decode bandwidth against the actual prepared working set.
 *
 * Decode GEMV is a streaming kernel: every invocation reads each prepared weight
 * byte once. Its useful economy figure is therefore the bandwidth it achieves
 * for the bytes it actually streams, separated by where those bytes reside.
 * This harness sweeps a geometric ladder of projection shapes whose exact
 * prepared footprint doubles at every step, from a few KiB to hundreds of MiB,
 * and reports the bandwidth of each step at each requested team width. The
 * L1-, L2-, last-level- and DRAM-resident regimes then appear as plateaus of
 * the measured curve. No cache capacity is assumed: AArch64 has no user-mode
 * cache-size query, and some kernels publish no sysfs sizes. Published sysfs
 * sizes only annotate a step; unpublished ones are reported as such.
 *
 * Every step enters the production `Auto` M=1 dispatcher
 * (`gemv_native_quantised_preq`), exactly as `Perf__CPUNativeQuantised_ThreadScaling`
 * does, and is authenticated byte for byte against serial one-worker production
 * rows outside the timing interval. Streamed bytes are computed from the
 * prepared owner itself: `(N_padded / 64) * blocks_per_row` execution units of
 * `interleaved_block_stride` bytes, plus the Q8_1 activation row and the FP32
 * output row. The harness contains no intrinsics and runs unchanged on x86
 * and AArch64.
 *
 * @code
 * OMP_PLACES=cores OMP_PROC_BIND=close LLAMINAR_CPU_DECODE_BW_FORMATS=Q4_K,Q8_0 \
 *   LLAMINAR_CPU_DECODE_BW_THREADS=1,20 LLAMINAR_CPU_DECODE_BW_JSON=bw.json \
 *   ./build_v2_release/tests/v2/v2_perf_cpu_native_quantised_decode_bandwidth \
 *   --gtest_filter='*Ladder'
 * python3 scripts/perf_baseline.py record --kind kernel-metrics --input bw.json \
 *   --build-dir build_v2_release --output bw-baseline.json
 * @endcode
 *
 * Environment overrides (every invalid value is fatal):
 *
 * - `LLAMINAR_CPU_DECODE_BW_FORMATS`: CSV source labels or `all` (default Q4_K).
 * - `LLAMINAR_CPU_DECODE_BW_THREADS`: CSV team widths (default `1` and
 *   `omp_get_max_threads()`).
 * - `LLAMINAR_CPU_DECODE_BW_MAX_KIB`: largest prepared footprint (default
 *   131072 KiB). Raise it beyond the last-level cache of large-L3 hosts.
 * - `LLAMINAR_CPU_DECODE_BW_WARMUP` / `LLAMINAR_CPU_DECODE_BW_ITERS`: default 3 / 15.
 * - `LLAMINAR_CPU_DECODE_BW_JSON`: write a `kernel-metrics` document consumed
 *   by `scripts/perf_baseline.py`.
 */

#include "../../../../utils/NativeQuantisedTestPartialStorage.h"
#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/cpu/gemm/CPUNativeQuantisedExecutionContract.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedGemmKernel.h"
#include "nlohmann/json.hpp"
#include "utils/CPUFeatures.h"
#include "utils/QuantizedVerifierFormats.h"
#include "utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::cpu::native_quantised;
using namespace llaminar2::test;

namespace
{

/**
 * @brief Read a positive integer environment override; invalid values are fatal.
 * @param name Environment variable.
 * @param default_value Value used when the variable is unset or empty.
 * @return The parsed positive value.
 */
long positiveEnvironment(const char *name, long default_value)
{
    const char *raw = std::getenv(name);
    if (!raw || !*raw)
        return default_value;
    char *end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (*end != '\0' || value <= 0)
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    return value;
}

/**
 * @brief Resolve the requested team widths.
 * @param maximum_threads `omp_get_max_threads()` of this process.
 * @return Strictly increasing unique widths, each within the OpenMP limit.
 */
std::vector<int> requestedThreads(int maximum_threads)
{
    std::vector<int> widths;
    const char *raw = std::getenv("LLAMINAR_CPU_DECODE_BW_THREADS");
    if (!raw || !*raw)
        widths = {1, maximum_threads};
    else
    {
        std::stringstream stream(raw);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            char *end = nullptr;
            const long width = std::strtol(token.c_str(), &end, 10);
            if (token.empty() || *end != '\0' || width <= 0 || width > maximum_threads)
                throw std::invalid_argument("LLAMINAR_CPU_DECODE_BW_THREADS entry out of range: " + token);
            widths.push_back(static_cast<int>(width));
        }
    }
    std::sort(widths.begin(), widths.end());
    widths.erase(std::unique(widths.begin(), widths.end()), widths.end());
    return widths;
}

/**
 * @brief Resolve canonical source formats by label.
 * @param request CSV labels or `all`.
 * @return Registry entries in request order; unknown labels are fatal.
 */
std::vector<const QuantizedVerifierFormatCase *> formatsFor(const std::string &request)
{
    std::vector<const QuantizedVerifierFormatCase *> result;
    const auto &formats = quantizedVerifierFormats();
    if (request == "all")
    {
        for (const auto &format : formats)
            result.push_back(&format);
        return result;
    }
    std::stringstream stream(request);
    std::string token;
    while (std::getline(stream, token, ','))
    {
        const auto found = std::find_if(formats.begin(), formats.end(),
            [&](const auto &format) { return token == format.label; });
        if (found == formats.end())
            throw std::invalid_argument("Unknown decode-bandwidth format: " + token);
        result.push_back(&*found);
    }
    return result;
}

/** @brief One published data/unified cache level of CPU 0, if the kernel exposes it. */
struct PublishedCache
{
    int level = 0;
    std::uint64_t bytes = 0;
};

/**
 * @brief Read cache sizes CPU 0 publishes in sysfs, without inventing missing ones.
 * @return Data/unified levels whose `size` file exists; empty when unpublished.
 */
std::vector<PublishedCache> publishedCaches()
{
    std::vector<PublishedCache> caches;
    for (int index = 0; index < 8; ++index)
    {
        const std::string base = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(index) + "/";
        std::ifstream type_file(base + "type"), level_file(base + "level"), size_file(base + "size");
        if (!type_file || !level_file || !size_file)
            continue;
        std::string type, size;
        int level = 0;
        type_file >> type;
        level_file >> level;
        size_file >> size;
        if (type == "Instruction" || size.empty())
            continue;
        std::uint64_t multiplier = 1;
        const char unit = size.back();
        if (unit == 'K')
            multiplier = 1024u;
        else if (unit == 'M')
            multiplier = 1024u * 1024u;
        const std::uint64_t value = std::strtoull(size.c_str(), nullptr, 10) * multiplier;
        if (value)
            caches.push_back({level, value});
    }
    return caches;
}

/**
 * @brief Place one footprint relative to the published cache levels.
 * @param bytes Streamed bytes per invocation.
 * @param caches Published data/unified levels.
 * @return "<=L1d", "<=L2", ... ">LLC", or "unpublished" when nothing is published.
 */
std::string residency(std::uint64_t bytes, const std::vector<PublishedCache> &caches)
{
    if (caches.empty())
        return "unpublished";
    std::vector<PublishedCache> sorted = caches;
    std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) { return a.level < b.level; });
    for (const auto &cache : sorted)
        if (bytes <= cache.bytes)
            return cache.level == 1 ? "<=L1d" : "<=L" + std::to_string(cache.level);
    return ">LLC";
}

/**
 * @brief Quantize deterministic FP32 activations once, outside every timing interval.
 * @param k Logical reduction width.
 * @param k_blocks Q8_1 blocks per row.
 * @return One Q8_1 activation row.
 */
std::vector<Q8_1Block> makeActivationRow(int k, int k_blocks)
{
    std::vector<float> fp32(static_cast<size_t>(k));
    for (size_t index = 0; index < fp32.size(); ++index)
        fp32[index] = static_cast<float>(static_cast<int>((index * 17 + 11) % 251) - 125) / 127.0f;
    std::vector<Q8_1Block> quantized(static_cast<size_t>(k_blocks));
    quantize_activations_to_q8_1(fp32.data(), quantized.data(), 1, k, k_blocks);
    return quantized;
}

/** @brief One timed ladder step at one team width. */
struct BandwidthPoint
{
    std::string format;
    int n = 0;
    int k = 0;
    int threads = 0;
    std::uint64_t weight_bytes = 0;
    std::uint64_t streamed_bytes = 0;
    std::string residency;
    double median_us = 0.0;
    std::vector<double> gbps_samples;
    double median_gbps = 0.0;
};

/** @brief Measurement parameters for one ladder sweep. */
struct LadderSpec
{
    std::vector<const QuantizedVerifierFormatCase *> formats;
    std::vector<int> threads;
    std::uint64_t max_weight_bytes = 0;
    int warmup = 0;
    int iterations = 0;
};

/**
 * @brief Generate the doubling shape ladder.
 *
 * Starting at N=64, K=256, the ladder alternately doubles N and K, so every
 * step doubles the element count and keeps shapes near-square. K stays a
 * multiple of 256 (every superblock format) and N a multiple of 64 (one
 * execution chunk). Generation stops before the prepared footprint, estimated
 * at the format's own bytes per element, would exceed the maximum.
 */
std::vector<std::pair<int, int>> shapeLadder(double bytes_per_element, std::uint64_t max_weight_bytes)
{
    std::vector<std::pair<int, int>> shapes;
    int n = 64, k = 256;
    for (int step = 0; step < 40; ++step)
    {
        if (static_cast<double>(n) * k * bytes_per_element > static_cast<double>(max_weight_bytes))
            break;
        shapes.emplace_back(n, k);
        if (step % 2 == 0)
            n *= 2;
        else
            k *= 2;
    }
    return shapes;
}

/**
 * @brief Measure every format, ladder step and team width.
 * @param spec Formats, widths, footprint ceiling and repetition counts.
 * @return One authenticated point per (format, step, width).
 */
std::vector<BandwidthPoint> measureLadder(const LadderSpec &spec)
{
    std::vector<BandwidthPoint> points;
    const auto caches = publishedCaches();
    for (const QuantizedVerifierFormatCase *format : spec.formats)
    {
        SCOPED_TRACE(format->label);
        // Calibrate bytes/element from one tiny prepared owner of this format.
        double bytes_per_element = 0.0;
        {
            auto probe = format->create({64, 256}, 0xB0u);
            CPUNativeQuantisedGemmKernel probe_kernel(probe.get());
            if (!probe_kernel.isValid())
                throw std::runtime_error(std::string("format did not prepare: ") + format->label);
            const auto &packed = probe_kernel.packedWeights();
            bytes_per_element = static_cast<double>(packed.N_padded / 64) * packed.blocks_per_row *
                                packed.interleaved_block_stride / (64.0 * 256.0);
        }
        for (const auto &[n, k] : shapeLadder(bytes_per_element, spec.max_weight_bytes))
        {
            auto weights = format->create({static_cast<size_t>(n), static_cast<size_t>(k)}, 0x5EEDu);
            CPUNativeQuantisedGemmKernel kernel(weights.get());
            if (!kernel.isValid())
                throw std::runtime_error("ladder shape did not prepare");
            const CPUNativeQuantisedPackedWeights &packed = kernel.packedWeights();
            const auto footprint = packed.preparedFootprint();
            const std::uint64_t n_chunks = static_cast<std::uint64_t>(packed.N_padded / 64);
            const std::uint64_t k_blocks = static_cast<std::uint64_t>(packed.blocks_per_row);
            const std::uint64_t weight_bytes = n_chunks * k_blocks * footprint.weight_bytes_per_n_chunk_k_block;
            const std::uint64_t streamed = weight_bytes + k_blocks * footprint.activation_bytes_per_row_k_block +
                                           n_chunks * footprint.output_bytes_per_row_n_chunk;
            const auto activations = makeActivationRow(k, packed.blocks_per_row);

            std::vector<float> serial(static_cast<size_t>(n), 0.0f);
            {
                omp_set_num_threads(1);
                NativeQuantisedTestPartialStorage partials(packed, 1);
                gemv_native_quantised_preq(packed, activations.data(), serial.data(), partials.span(),
                                           ISAPath::AUTO, DecodeSchedulePolicy::Auto);
            }
            for (const int threads : spec.threads)
            {
                omp_set_num_threads(threads);
                int observed = 0;
#pragma omp parallel shared(observed)
                {
#pragma omp single
                    observed = omp_get_num_threads();
                }
                if (observed != threads)
                    throw std::runtime_error("OpenMP did not honor the requested team width");
                std::vector<float> output(static_cast<size_t>(n), 0.0f);
                NativeQuantisedTestPartialStorage partials(packed, 1);
                const auto launch = [&]() {
                    gemv_native_quantised_preq(packed, activations.data(), output.data(), partials.span(),
                                               ISAPath::AUTO, DecodeSchedulePolicy::Auto);
                };
                for (int warm = 0; warm < spec.warmup; ++warm)
                    launch();
                BandwidthPoint point{.format = format->label, .n = n, .k = k, .threads = threads,
                                     .weight_bytes = weight_bytes, .streamed_bytes = streamed,
                                     .residency = residency(streamed, caches)};
                std::vector<double> microseconds;
                for (int iteration = 0; iteration < spec.iterations; ++iteration)
                {
                    const auto begin = std::chrono::steady_clock::now();
                    launch();
                    const auto end = std::chrono::steady_clock::now();
                    microseconds.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
                }
                // Authenticate the timed arithmetic after timing, outside every interval.
                if (std::memcmp(output.data(), serial.data(), output.size() * sizeof(float)) != 0)
                    throw std::runtime_error(std::string(format->label) + " N=" + std::to_string(n) +
                                             " K=" + std::to_string(k) + " threads=" + std::to_string(threads) +
                                             " did not publish the serial one-worker M=1 bytes");
                for (const double us : microseconds)
                    point.gbps_samples.push_back(static_cast<double>(streamed) / (us * 1e3));
                std::vector<double> sorted = microseconds;
                std::sort(sorted.begin(), sorted.end());
                point.median_us = sorted[sorted.size() / 2];
                point.median_gbps = static_cast<double>(streamed) / (point.median_us * 1e3);
                points.push_back(std::move(point));
            }
        }
    }
    return points;
}

/**
 * @brief Render the human table and optionally the `kernel-metrics` JSON document.
 * @param points Authenticated measurements.
 * @param spec Parameters recorded with the document.
 * @param json_path Destination, or empty for no file.
 */
void publish(const std::vector<BandwidthPoint> &points, const LadderSpec &spec, const std::string &json_path)
{
    std::cout << std::left << std::setw(10) << "format" << std::right << std::setw(7) << "N" << std::setw(7) << "K"
              << std::setw(12) << "stream KiB" << std::setw(12) << "residency" << std::setw(5) << "thr"
              << std::setw(11) << "median us" << std::setw(10) << "GB/s" << std::setw(12) << "GB/s/core" << '\n';
    for (const auto &point : points)
        std::cout << std::left << std::setw(10) << point.format << std::right << std::setw(7) << point.n
                  << std::setw(7) << point.k << std::setw(12) << point.streamed_bytes / 1024 << std::setw(12)
                  << point.residency << std::setw(5) << point.threads << std::fixed << std::setprecision(2)
                  << std::setw(11) << point.median_us << std::setw(10) << point.median_gbps << std::setw(12)
                  << point.median_gbps / point.threads << '\n';
    if (json_path.empty())
        return;
    nlohmann::json metrics = nlohmann::json::object();
    nlohmann::json shapes = nlohmann::json::array();
    for (const auto &point : points)
    {
        const std::string key = point.format + ".n" + std::to_string(point.n) + "k" + std::to_string(point.k) +
                                ".t" + std::to_string(point.threads) + ".gbps";
        metrics[key] = {{"value", point.median_gbps}, {"unit", "GB/s"}, {"better", "higher"},
                        {"samples", point.gbps_samples}};
        shapes.push_back({{"format", point.format}, {"n", point.n}, {"k", point.k}, {"threads", point.threads},
                          {"weight_bytes", point.weight_bytes}, {"streamed_bytes", point.streamed_bytes},
                          {"residency", point.residency}});
    }
    nlohmann::json formats = nlohmann::json::array();
    for (const auto *format : spec.formats)
        formats.push_back(format->label);
    nlohmann::json caches = nlohmann::json::array();
    for (const auto &cache : publishedCaches())
        caches.push_back({{"level", cache.level}, {"bytes", cache.bytes}});
    const nlohmann::json document = {
        {"harness", "Perf__CPUNativeQuantised_DecodeBandwidth"},
        {"parameters", {{"formats", formats}, {"threads", spec.threads},
                        {"max_weight_bytes", spec.max_weight_bytes}, {"warmup", spec.warmup},
                        {"iterations", spec.iterations}, {"published_caches", caches}, {"shapes", shapes},
                        {"build_isa", compiledNativeQuantisedBuildISAName()},
                        {"runtime_isa", isaLevelName(activeISALevel())}}},
        {"metrics", metrics}};
    std::ofstream out(json_path);
    out << document.dump(1) << '\n';
    if (!out)
        throw std::runtime_error("could not write " + json_path);
}

} // namespace

/** @brief Environment-driven ladder: the canonical bandwidth measurement. */
TEST(Perf__CPUNativeQuantisedDecodeBandwidth, Ladder)
{
    omp_set_dynamic(0);
    const char *formats = std::getenv("LLAMINAR_CPU_DECODE_BW_FORMATS");
    const char *json = std::getenv("LLAMINAR_CPU_DECODE_BW_JSON");
    const LadderSpec spec{
        .formats = formatsFor(formats && *formats ? formats : "Q4_K"),
        .threads = requestedThreads(omp_get_max_threads()),
        .max_weight_bytes = static_cast<std::uint64_t>(positiveEnvironment("LLAMINAR_CPU_DECODE_BW_MAX_KIB", 131072)) * 1024u,
        .warmup = static_cast<int>(positiveEnvironment("LLAMINAR_CPU_DECODE_BW_WARMUP", 3)),
        .iterations = static_cast<int>(positiveEnvironment("LLAMINAR_CPU_DECODE_BW_ITERS", 15)),
    };
    const auto points = measureLadder(spec);
    ASSERT_FALSE(points.empty());
    publish(points, spec, json ? json : "");
}

/**
 * @brief Model-free contract: every format authenticates, accounting is exact and
 * the JSON document carries one finite metric per point. No speed gate.
 */
TEST(Perf__CPUNativeQuantisedDecodeBandwidth, ContractTinyLadder)
{
    omp_set_dynamic(0);
    const int widest = std::min(2, omp_get_max_threads());
    const LadderSpec spec{.formats = formatsFor("all"), .threads = widest > 1 ? std::vector<int>{1, widest}
                                                                              : std::vector<int>{1},
                          .max_weight_bytes = 64u * 1024u, .warmup = 1, .iterations = 2};
    const auto points = measureLadder(spec);
    ASSERT_FALSE(points.empty());
    std::set<std::string> covered;
    for (const auto &point : points)
    {
        covered.insert(point.format);
        EXPECT_LE(point.weight_bytes, spec.max_weight_bytes) << point.format;
        EXPECT_GT(point.streamed_bytes, point.weight_bytes) << point.format;
        // Accounting is the prepared owner's own extent: N_padded/64 chunks × K/32 blocks × stride.
        EXPECT_EQ(point.weight_bytes % static_cast<std::uint64_t>(point.n / 64 * (point.k / 32)), 0u) << point.format;
        EXPECT_EQ(point.gbps_samples.size(), static_cast<size_t>(spec.iterations));
        EXPECT_GT(point.median_gbps, 0.0);
    }
    EXPECT_EQ(covered.size(), spec.formats.size()) << "every format must reach at least one ladder step";
    const std::string path = ::testing::TempDir() + "decode_bandwidth_contract.json";
    publish(points, spec, path);
    std::ifstream in(path);
    const auto document = nlohmann::json::parse(in);
    EXPECT_EQ(document.at("harness"), "Perf__CPUNativeQuantised_DecodeBandwidth");
    EXPECT_EQ(document.at("metrics").size(), points.size());
    for (const auto &[name, metric] : document.at("metrics").items())
    {
        EXPECT_EQ(metric.at("better"), "higher") << name;
        EXPECT_EQ(metric.at("samples").size(), static_cast<size_t>(spec.iterations)) << name;
    }
    std::remove(path.c_str());
}
