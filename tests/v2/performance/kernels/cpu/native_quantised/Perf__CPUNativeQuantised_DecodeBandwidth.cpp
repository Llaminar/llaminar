/**
 * @file Perf__CPUNativeQuantised_DecodeBandwidth.cpp
 * @brief Portable M=1 decode effective-byte rate over a doubling working-set ladder.
 *
 * Each ladder step prepares one NativeQuantised projection whose prepared
 * extent doubles from step to step (a few KiB up to `LLAMINAR_CPU_DECODE_BW_MAX_KIB`)
 * and times the production `Auto` M=1 dispatcher (`gemv_native_quantised_preq`)
 * at each *requested* OpenMP width.
 *
 * What is reported, and what is not:
 *
 * - **Effective-byte rate.** `effective_bytes / median_time`, where
 *   `effective_bytes` = prepared weight extent ((N_padded/64) x (K/32) x
 *   `interleaved_block_stride`) + one Q8_1 activation row + one FP32 output
 *   row. It is a logical-footprint proxy, NOT measured memory traffic:
 *   repeated activation reads, partial-sum traffic and cache-line effects are
 *   not counted, and nothing here proves each byte is read exactly once.
 * - **Requested width, not executing width.** `omp_set_num_threads(w)` sets
 *   the request; the production dispatcher may use fewer workers (for
 *   example, serial execution for small work), and OpenMP threads need not be
 *   physical cores. No per-core normalization is reported. `omp_get_num_procs()`
 *   and `omp_get_max_threads()` are recorded for context only.
 * - **No cache attribution.** Plateaus in the curve are observations; they do
 *   not identify cache levels. CPU 0's published sysfs cache sizes (if any)
 *   are recorded as host context only.
 * - **Timing statistic.** Each point's time is the *upper median*:
 *   `sorted(samples)[n / 2]` (the middle sample for odd n, the upper of the
 *   two middle samples for even n).
 * - **Authentication.** Every timed invocation starts from a NaN-sentinel
 *   output (written before the timer starts) and its complete output is
 *   compared byte for byte, after the timer stops, with serial one-worker rows
 *   of the same production entry point. This proves the timed launches equal
 *   the serial production arithmetic; it is not an independent reference
 *   implementation. Sentinel writes and comparisons are outside the timed
 *   interval but do warm the output row.
 *
 * Environment (every value is validated; malformed, out-of-range or duplicate
 * values are fatal):
 *
 * - `LLAMINAR_CPU_DECODE_BW_FORMATS`: CSV unique source labels or `all` (default Q4_K).
 * - `LLAMINAR_CPU_DECODE_BW_THREADS`: CSV unique requested widths in
 *   [1, omp_get_max_threads()] (default `1` and `omp_get_max_threads()`).
 * - `LLAMINAR_CPU_DECODE_BW_MAX_KIB`: largest prepared extent, [1, 2^30] KiB (default 131072).
 * - `LLAMINAR_CPU_DECODE_BW_WARMUP`: [1, 10000] (default 3).
 * - `LLAMINAR_CPU_DECODE_BW_ITERS`: [1, 100000] (default 15).
 * - `LLAMINAR_CPU_DECODE_BW_JSON`: write a `kernel-metrics` document for
 *   `scripts/perf_baseline.py`.
 *
 * Every requested format must contribute at least one ladder step; a ceiling
 * below a format's smallest step is fatal rather than silently omitted.
 */

#include "../../../../utils/NativeQuantisedTestPartialStorage.h"
#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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
 * @brief Parse one bounded decimal integer; every malformed or out-of-range input is fatal.
 * @param raw Text to parse, or null/empty for the default.
 * @param name Variable name for diagnostics.
 * @param default_value Value for null/empty input (must itself lie within the bounds).
 * @param minimum Inclusive lower bound.
 * @param maximum Inclusive upper bound.
 * @return The parsed value.
 */
long long parseBounded(const char *raw, const std::string &name, long long default_value, long long minimum,
                       long long maximum)
{
    if (!raw || !*raw)
        return default_value;
    if (!(std::isdigit(static_cast<unsigned char>(raw[0])) || raw[0] == '-'))
        throw std::invalid_argument(name + "='" + raw + "' must start with a digit or '-'");
    errno = 0;
    char *end = nullptr;
    const long long value = std::strtoll(raw, &end, 10);
    if (errno == ERANGE || end == raw || *end != '\0' || value < minimum || value > maximum)
        throw std::invalid_argument(name + "='" + raw + "' must be an integer in [" + std::to_string(minimum) +
                                    ", " + std::to_string(maximum) + "]");
    return value;
}

/**
 * @brief Split a CSV list, rejecting empty and duplicate entries.
 * @param raw Comma-separated text.
 * @param name Variable name for diagnostics.
 * @return Entries in input order.
 */
std::vector<std::string> uniqueList(const std::string &raw, const std::string &name)
{
    std::vector<std::string> items;
    std::stringstream stream(raw);
    std::string token;
    while (std::getline(stream, token, ','))
    {
        if (token.empty() || std::find(items.begin(), items.end(), token) != items.end())
            throw std::invalid_argument(name + " contains an empty or duplicate entry: '" + token + "'");
        items.push_back(token);
    }
    if (items.empty() || raw.back() == ',')
        throw std::invalid_argument(name + " must be a non-empty list without trailing comma");
    return items;
}

/**
 * @brief Resolve requested OpenMP widths.
 * @param raw CSV of widths, or null/empty for `1` and `maximum_threads`.
 * @param maximum_threads `omp_get_max_threads()`.
 * @return Ascending unique requested widths.
 */
std::vector<int> requestedWidths(const char *raw, int maximum_threads)
{
    std::vector<int> widths;
    if (!raw || !*raw)
        widths = maximum_threads > 1 ? std::vector<int>{1, maximum_threads} : std::vector<int>{1};
    else
        for (const auto &entry : uniqueList(raw, "LLAMINAR_CPU_DECODE_BW_THREADS"))
            widths.push_back(static_cast<int>(
                parseBounded(entry.c_str(), "LLAMINAR_CPU_DECODE_BW_THREADS entry", 1, 1, maximum_threads)));
    std::sort(widths.begin(), widths.end());
    return widths;
}

/**
 * @brief Resolve canonical source formats by unique label.
 * @param request CSV labels or `all`.
 * @return Registry entries in request order; unknown or duplicate labels are fatal.
 */
std::vector<const QuantizedVerifierFormatCase *> formatsFor(const std::string &request)
{
    const auto &formats = quantizedVerifierFormats();
    std::vector<const QuantizedVerifierFormatCase *> result;
    if (request == "all")
    {
        for (const auto &format : formats)
            result.push_back(&format);
        return result;
    }
    for (const auto &token : uniqueList(request, "LLAMINAR_CPU_DECODE_BW_FORMATS"))
    {
        const auto found = std::find_if(formats.begin(), formats.end(),
            [&](const auto &format) { return token == format.label; });
        if (found == formats.end())
            throw std::invalid_argument("Unknown decode-bandwidth format: " + token);
        result.push_back(&*found);
    }
    return result;
}

/** @brief CPU 0 data/unified cache sizes the kernel publishes; context only, never attribution. */
nlohmann::json publishedCpu0Caches()
{
    nlohmann::json caches = nlohmann::json::array();
    for (int index = 0; index < 8; ++index)
    {
        const std::string base = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(index) + "/";
        std::ifstream type_file(base + "type"), level_file(base + "level"), size_file(base + "size");
        std::string type, size;
        int level = 0;
        if (type_file >> type && level_file >> level && size_file >> size && type != "Instruction")
            caches.push_back({{"level", level}, {"size", size}});
    }
    return caches;
}

/** @brief Deterministic Q8_1 activation row, quantized once outside every timing interval. */
std::vector<Q8_1Block> makeActivationRow(int k, int k_blocks)
{
    std::vector<float> fp32(static_cast<size_t>(k));
    for (size_t index = 0; index < fp32.size(); ++index)
        fp32[index] = static_cast<float>(static_cast<int>((index * 17 + 11) % 251) - 125) / 127.0f;
    std::vector<Q8_1Block> quantized(static_cast<size_t>(k_blocks));
    quantize_activations_to_q8_1(fp32.data(), quantized.data(), 1, k, k_blocks);
    return quantized;
}

/**
 * @brief Require a complete output to equal the serial production rows byte for byte.
 * @param output Output produced by one timed invocation.
 * @param serial Serial one-worker rows of the same production entry point.
 * @param context Diagnostic description of the invocation.
 * @throws std::runtime_error On any byte difference, including an unwritten sentinel.
 */
void authenticate(const std::vector<float> &output, const std::vector<float> &serial, const std::string &context)
{
    if (output.size() != serial.size() ||
        std::memcmp(output.data(), serial.data(), output.size() * sizeof(float)) != 0)
        throw std::runtime_error(context + " did not publish the serial one-worker M=1 bytes");
}

/** @brief One timed ladder step at one requested width. */
struct EffectiveRatePoint
{
    std::string format;
    int n = 0;
    int k = 0;
    int requested_threads = 0;
    std::uint64_t n_chunks = 0;
    std::uint64_t k_blocks = 0;
    std::uint64_t block_stride = 0;
    std::uint64_t weight_bytes = 0;
    std::uint64_t effective_bytes = 0;
    double upper_median_us = 0.0;
    std::vector<double> rate_samples_gbps;
    double rate_gbps = 0.0;
};

/** @brief Measurement parameters for one ladder sweep. */
struct LadderSpec
{
    std::vector<const QuantizedVerifierFormatCase *> formats;
    std::vector<int> requested_threads;
    std::uint64_t max_weight_bytes = 0;
    int warmup = 0;
    int iterations = 0;
};

/**
 * @brief Generate the doubling shape ladder (N and K alternately doubled; K a multiple of 256).
 * @param bytes_per_element Prepared bytes per weight element of the format.
 * @param max_weight_bytes Inclusive ceiling on the estimated prepared extent.
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
 * @brief Time every format, ladder step and requested width.
 * @return Authenticated points; every requested format contributes at least one.
 * @throws std::runtime_error When a format prepares no step or any invocation is not byte-exact.
 */
std::vector<EffectiveRatePoint> measureLadder(const LadderSpec &spec)
{
    std::vector<EffectiveRatePoint> points;
    const float sentinel = std::numeric_limits<float>::quiet_NaN();
    for (const QuantizedVerifierFormatCase *format : spec.formats)
    {
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
        const auto shapes = shapeLadder(bytes_per_element, spec.max_weight_bytes);
        if (shapes.empty())
            throw std::runtime_error(std::string(format->label) + ": LLAMINAR_CPU_DECODE_BW_MAX_KIB is below its smallest ladder step");
        for (const auto &[n, k] : shapes)
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
            const auto activations = makeActivationRow(k, packed.blocks_per_row);
            std::vector<float> serial(static_cast<size_t>(n), sentinel);
            {
                omp_set_num_threads(1);
                NativeQuantisedTestPartialStorage partials(packed, 1);
                gemv_native_quantised_preq(packed, activations.data(), serial.data(), partials.span(),
                                           ISAPath::AUTO, DecodeSchedulePolicy::Auto);
            }
            for (const int requested : spec.requested_threads)
            {
                omp_set_num_threads(requested);
                std::vector<float> output(static_cast<size_t>(n), sentinel);
                NativeQuantisedTestPartialStorage partials(packed, 1);
                const auto launch = [&]() {
                    gemv_native_quantised_preq(packed, activations.data(), output.data(), partials.span(),
                                               ISAPath::AUTO, DecodeSchedulePolicy::Auto);
                };
                const std::string context = std::string(format->label) + " N=" + std::to_string(n) + " K=" +
                                            std::to_string(k) + " requested_threads=" + std::to_string(requested);
                for (int warm = 0; warm < spec.warmup; ++warm)
                    launch();
                EffectiveRatePoint point{
                    .format = format->label, .n = n, .k = k, .requested_threads = requested,
                    .n_chunks = n_chunks, .k_blocks = k_blocks,
                    .block_stride = footprint.weight_bytes_per_n_chunk_k_block, .weight_bytes = weight_bytes,
                    .effective_bytes = weight_bytes + k_blocks * footprint.activation_bytes_per_row_k_block +
                                       n_chunks * footprint.output_bytes_per_row_n_chunk};
                std::vector<double> microseconds;
                for (int iteration = 0; iteration < spec.iterations; ++iteration)
                {
                    std::fill(output.begin(), output.end(), sentinel);
                    const auto begin = std::chrono::steady_clock::now();
                    launch();
                    const auto end = std::chrono::steady_clock::now();
                    authenticate(output, serial, context + " iteration=" + std::to_string(iteration));
                    const double us = std::chrono::duration<double, std::micro>(end - begin).count();
                    if (!(us > 0.0) || !std::isfinite(us))
                        throw std::runtime_error(context + ": non-positive or non-finite duration");
                    microseconds.push_back(us);
                }
                for (const double us : microseconds)
                    point.rate_samples_gbps.push_back(static_cast<double>(point.effective_bytes) / (us * 1e3));
                std::vector<double> sorted = microseconds;
                std::sort(sorted.begin(), sorted.end());
                point.upper_median_us = sorted[sorted.size() / 2];
                point.rate_gbps = static_cast<double>(point.effective_bytes) / (point.upper_median_us * 1e3);
                points.push_back(std::move(point));
            }
        }
    }
    return points;
}

/** @brief Build the `kernel-metrics` document for `scripts/perf_baseline.py`. */
nlohmann::json kernelMetricsDocument(const std::vector<EffectiveRatePoint> &points, const LadderSpec &spec)
{
    nlohmann::json metrics = nlohmann::json::object(), steps = nlohmann::json::array(), formats = nlohmann::json::array();
    for (const auto &point : points)
    {
        const std::string key = point.format + ".n" + std::to_string(point.n) + "k" + std::to_string(point.k) +
                                ".req" + std::to_string(point.requested_threads) + ".effective_gbps";
        if (metrics.contains(key))
            throw std::logic_error("duplicate metric key " + key);
        metrics[key] = {{"value", point.rate_gbps}, {"unit", "GB/s"}, {"better", "higher"},
                        {"samples", point.rate_samples_gbps}};
        steps.push_back({{"format", point.format}, {"n", point.n}, {"k", point.k},
                         {"requested_threads", point.requested_threads}, {"weight_bytes", point.weight_bytes},
                         {"effective_bytes", point.effective_bytes}, {"upper_median_us", point.upper_median_us}});
    }
    for (const auto *format : spec.formats)
        formats.push_back(format->label);
    return {{"harness", "Perf__CPUNativeQuantised_DecodeBandwidth"},
            {"parameters", {{"rate", "effective bytes / upper-median time (logical footprint proxy, not measured traffic)"},
                            {"formats", formats}, {"requested_threads", spec.requested_threads},
                            {"omp_num_procs", omp_get_num_procs()}, {"omp_max_threads", omp_get_max_threads()},
                            {"max_weight_bytes", spec.max_weight_bytes}, {"warmup", spec.warmup},
                            {"iterations", spec.iterations}, {"cpu0_published_caches", publishedCpu0Caches()},
                            {"build_isa", compiledNativeQuantisedBuildISAName()},
                            {"runtime_isa_tag", isaLevelName(activeISALevel())}, {"steps", steps}}},
            {"metrics", metrics}};
}

/** @brief Print the human table. */
void printTable(const std::vector<EffectiveRatePoint> &points)
{
    std::cout << std::left << std::setw(10) << "format" << std::right << std::setw(7) << "N" << std::setw(7) << "K"
              << std::setw(15) << "effective KiB" << std::setw(6) << "req" << std::setw(16) << "upper-median us"
              << std::setw(16) << "effective GB/s" << '\n';
    for (const auto &point : points)
        std::cout << std::left << std::setw(10) << point.format << std::right << std::setw(7) << point.n
                  << std::setw(7) << point.k << std::setw(15) << point.effective_bytes / 1024 << std::setw(6)
                  << point.requested_threads << std::fixed << std::setprecision(2) << std::setw(16)
                  << point.upper_median_us << std::setw(16) << point.rate_gbps << '\n';
}

/** @brief Resolve the measurement from the environment with every value validated. */
LadderSpec specFromEnvironment()
{
    const char *formats = std::getenv("LLAMINAR_CPU_DECODE_BW_FORMATS");
    const auto max_kib = parseBounded(std::getenv("LLAMINAR_CPU_DECODE_BW_MAX_KIB"),
                                      "LLAMINAR_CPU_DECODE_BW_MAX_KIB", 131072, 1, 1LL << 30);
    return {.formats = formatsFor(formats && *formats ? formats : "Q4_K"),
            .requested_threads = requestedWidths(std::getenv("LLAMINAR_CPU_DECODE_BW_THREADS"), omp_get_max_threads()),
            .max_weight_bytes = static_cast<std::uint64_t>(max_kib) * 1024u,
            .warmup = static_cast<int>(parseBounded(std::getenv("LLAMINAR_CPU_DECODE_BW_WARMUP"),
                                                    "LLAMINAR_CPU_DECODE_BW_WARMUP", 3, 1, 10000)),
            .iterations = static_cast<int>(parseBounded(std::getenv("LLAMINAR_CPU_DECODE_BW_ITERS"),
                                                        "LLAMINAR_CPU_DECODE_BW_ITERS", 15, 1, 100000))};
}

} // namespace

/** @brief Environment-driven ladder: the canonical measurement. */
TEST(Perf__CPUNativeQuantisedDecodeBandwidth, Ladder)
{
    omp_set_dynamic(0);
    const LadderSpec spec = specFromEnvironment();
    const auto points = measureLadder(spec);
    printTable(points);
    if (const char *json = std::getenv("LLAMINAR_CPU_DECODE_BW_JSON"); json && *json)
    {
        std::ofstream out(json);
        out << kernelMetricsDocument(points, spec).dump(1) << '\n';
        if (!out)
            throw std::runtime_error(std::string("could not write ") + json);
    }
}

/**
 * @brief Model-free contract over every format: exact extents, every invocation
 * authenticated, finite positive timings and rates, and a numerically valid JSON
 * document with one metric per point. No speed gate.
 */
TEST(Perf__CPUNativeQuantisedDecodeBandwidth, ContractTinyLadder)
{
    omp_set_dynamic(0);
    const int widest = std::min(2, omp_get_max_threads());
    const LadderSpec spec{.formats = formatsFor("all"),
                          .requested_threads = widest > 1 ? std::vector<int>{1, widest} : std::vector<int>{1},
                          .max_weight_bytes = 64u * 1024u, .warmup = 1, .iterations = 2};
    const auto points = measureLadder(spec);
    std::set<std::string> covered;
    for (const auto &point : points)
    {
        SCOPED_TRACE(point.format + " N=" + std::to_string(point.n) + " K=" + std::to_string(point.k));
        covered.insert(point.format);
        EXPECT_EQ(point.n_chunks, static_cast<std::uint64_t>((point.n + 63) / 64));
        EXPECT_EQ(point.k_blocks, static_cast<std::uint64_t>(point.k / 32));
        EXPECT_GT(point.block_stride, 0u);
        EXPECT_EQ(point.weight_bytes, point.n_chunks * point.k_blocks * point.block_stride);
        EXPECT_EQ(point.effective_bytes, point.weight_bytes + point.k_blocks * sizeof(Q8_1Block) +
                                             point.n_chunks * 64u * sizeof(float));
        EXPECT_LE(point.weight_bytes, spec.max_weight_bytes);
        EXPECT_TRUE(std::isfinite(point.upper_median_us) && point.upper_median_us > 0.0);
        EXPECT_TRUE(std::isfinite(point.rate_gbps) && point.rate_gbps > 0.0);
        ASSERT_EQ(point.rate_samples_gbps.size(), static_cast<size_t>(spec.iterations));
        for (const double sample : point.rate_samples_gbps)
            EXPECT_TRUE(std::isfinite(sample) && sample > 0.0);
    }
    EXPECT_EQ(covered.size(), spec.formats.size()) << "every format must contribute a ladder step";
    const auto document = kernelMetricsDocument(points, spec);
    ASSERT_EQ(document.at("metrics").size(), points.size());
    size_t index = 0;
    for (const auto &[name, metric] : document.at("metrics").items())
    {
        SCOPED_TRACE(name);
        const double value = metric.at("value").get<double>();
        EXPECT_TRUE(std::isfinite(value) && value > 0.0);
        EXPECT_EQ(metric.at("better"), "higher");
        ASSERT_EQ(metric.at("samples").size(), static_cast<size_t>(spec.iterations));
        for (const auto &sample : metric.at("samples"))
            EXPECT_TRUE(std::isfinite(sample.get<double>()) && sample.get<double>() > 0.0);
        ++index;
    }
    EXPECT_EQ(index, points.size());
}

/**
 * @brief Negative controls for the authentication the harness relies on: a
 * one-ULP perturbation of a real production output and an unwritten (stale
 * sentinel) output must both be rejected; the unmodified output is accepted.
 */
TEST(Perf__CPUNativeQuantisedDecodeBandwidth, NegativeControlRejectsPerturbedAndStaleOutput)
{
    const auto &format = quantizedVerifierFormats().front();
    auto weights = format.create({128, 256}, 0x5EEDu);
    CPUNativeQuantisedGemmKernel kernel(weights.get());
    ASSERT_TRUE(kernel.isValid());
    const auto &packed = kernel.packedWeights();
    const auto activations = makeActivationRow(256, packed.blocks_per_row);
    const float sentinel = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> serial(128, sentinel), output(128, sentinel);
    NativeQuantisedTestPartialStorage partials(packed, 1);
    omp_set_num_threads(1);
    gemv_native_quantised_preq(packed, activations.data(), serial.data(), partials.span(), ISAPath::AUTO,
                               DecodeSchedulePolicy::Auto);
    gemv_native_quantised_preq(packed, activations.data(), output.data(), partials.span(), ISAPath::AUTO,
                               DecodeSchedulePolicy::Auto);
    EXPECT_NO_THROW(authenticate(output, serial, "unmodified"));
    auto perturbed = output;
    perturbed[64] = std::nextafter(perturbed[64], std::numeric_limits<float>::infinity());
    EXPECT_THROW(authenticate(perturbed, serial, "one-ULP"), std::runtime_error);
    const std::vector<float> stale(128, sentinel);
    EXPECT_THROW(authenticate(stale, serial, "stale sentinel"), std::runtime_error);
}

/** @brief Environment parsing rejects overflow, out-of-range, malformed and duplicate input. */
TEST(Perf__CPUNativeQuantisedDecodeBandwidth, EnvironmentParsingFailsClosed)
{
    EXPECT_EQ(parseBounded(nullptr, "X", 15, 1, 100000), 15);
    EXPECT_EQ(parseBounded("100000", "X", 15, 1, 100000), 100000);
    for (const char *bad : {"2147483648", "9223372036854775808", "-1", "0", "100001", "1e3", "12abc", " 5", "0x10"})
        EXPECT_THROW(parseBounded(bad, "X", 15, 1, 100000), std::invalid_argument) << bad;
    EXPECT_THROW(parseBounded("1073741825", "MAX_KIB", 1, 1, 1LL << 30), std::invalid_argument);
    EXPECT_THROW(formatsFor("Q4_K,Q4_K"), std::invalid_argument);
    EXPECT_THROW(formatsFor("Q4_K,"), std::invalid_argument);
    EXPECT_THROW(formatsFor("NOPE"), std::invalid_argument);
    EXPECT_THROW(requestedWidths("1,1", 4), std::invalid_argument);
    EXPECT_THROW(requestedWidths("5", 4), std::invalid_argument);
    EXPECT_EQ(requestedWidths("4,1", 4), (std::vector<int>{1, 4}));
    const LadderSpec too_small{.formats = formatsFor("Q8_0"), .requested_threads = {1},
                               .max_weight_bytes = 1024u, .warmup = 1, .iterations = 1};
    EXPECT_THROW(measureLadder(too_small), std::runtime_error);
}
