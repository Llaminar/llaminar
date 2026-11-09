/**
 * @file Perf__CPUNativeNEONExactOverlay.cpp
 * @brief Collect authenticated native ARM NativeQuantised exact-overlay candidates.
 *
 * Native NEON projections have exactly two numerically neutral schedule
 * dimensions: the shared-row tile (two or four rows) and the adjacent 64-column
 * chunks one dynamically claimed task owns. This collector measures every
 * candidate of both for the canonical exact-overlay production shapes
 * (`native_quantised_decode_shapes_v5.json`, `exact_overlay: true`), each requested
 * source format and each requested physical M, at the invocation worker count.
 *
 * Every candidate output is compared byte for byte with serial production M=1
 * rows before its samples count. Candidates are timed round-robin with a
 * rotating start so slow drift (thermal, frequency) spreads over all of them.
 * Rows are appended to `LLAMINAR_NEON_OVERLAY_CSV`; the generator
 * `native_quantised_dispatch/neon_exact_overlay.py` turns authenticated rows into
 * the installed `CPUNativeNEONExactOverlayGenerated.inc`. No generic model is
 * fitted and unmeasured keys keep the total native policy.
 *
 * Environment:
 * - `LLAMINAR_NEON_OVERLAY_CSV` (required) output path, appended.
 * - `LLAMINAR_NEON_OVERLAY_FORMATS` CSV source labels or `all` (default Q4_K).
 * - `LLAMINAR_NEON_OVERLAY_M` CSV physical M (default 1,2,3,4,5,6,7,8,12,15,16,31,64).
 * - `LLAMINAR_NEON_OVERLAY_SHAPES` CSV manifest names (default every exact shape).
 * - `LLAMINAR_NEON_OVERLAY_MAX_WEIGHT_ELEMENTS` skip larger shapes (default manifest CPU cap).
 * - `LLAMINAR_NEON_OVERLAY_ROUNDS` timed rounds per candidate (default 15).
 */

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
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

#include "../../native_quantised_dispatch/NativeQuantisedShapeManifest.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedGemmKernel.h"
#include "utils/QuantizedVerifierFormats.h"

using namespace llaminar2;
using namespace llaminar2::cpu::native_quantised;
using namespace llaminar2::test;
using llaminar2::test::native_quantised_dispatch::NativeQuantisedShapeSpec;
using llaminar2::test::native_quantised_dispatch::nativeQuantisedShapeManifest;

namespace
{
    /** @return Environment string or a default. */
    std::string envString(const char *name, const char *fallback)
    {
        const char *raw = std::getenv(name);
        return raw && *raw ? raw : fallback;
    }

    /** @return Comma-separated tokens of one environment value. */
    std::vector<std::string> csv(const std::string &value)
    {
        std::vector<std::string> result;
        std::stringstream stream(value);
        std::string token;
        while (std::getline(stream, token, ','))
            if (!token.empty())
                result.push_back(token);
        return result;
    }

    /** @return Positive integer environment value. */
    long positive(const char *name, long fallback)
    {
        const char *raw = std::getenv(name);
        if (!raw || !*raw)
            return fallback;
        char *end = nullptr;
        const long value = std::strtol(raw, &end, 10);
        if (*end != '\0' || value <= 0)
            throw std::invalid_argument(std::string(name) + " must be a positive integer");
        return value;
    }

    /** @brief One numerically neutral physical candidate. */
    struct Candidate
    {
        NEONProjectionRowTile tile;
        int chunks;
    };

    /** @return Name of a row tile in the CSV and generated table. */
    const char *tileName(NEONProjectionRowTile tile)
    {
        return tile == NEONProjectionRowTile::Pair ? "Pair" : "Four";
    }

    /** @return Explicit verifier candidate forcing one row tile. */
    VerifierRowsPolicy tilePolicy(NEONProjectionRowTile tile)
    {
        return tile == NEONProjectionRowTile::Pair ? VerifierRowsPolicy::Pairwise : VerifierRowsPolicy::WideRows;
    }

    /** @return Explicit M=1 ownership candidate. */
    DecodeSchedulePolicy chunkPolicy(int chunks)
    {
        switch (chunks)
        {
        case 1: return DecodeSchedulePolicy::Nbc1;
        case 2: return DecodeSchedulePolicy::Nbc2;
        case 4: return DecodeSchedulePolicy::Nbc4;
        case 8: return DecodeSchedulePolicy::Nbc8;
        }
        throw std::invalid_argument("unknown native chunk candidate");
    }

    /** @return Deterministic Q8 activations for M rows. */
    std::vector<Q8_1Block> activations(int rows, int k, int blocks)
    {
        std::vector<float> values(static_cast<size_t>(rows) * k);
        for (size_t index = 0; index < values.size(); ++index)
            values[index] = static_cast<float>(static_cast<int>((index * 37 + 5) % 251) - 125) / 127.0f;
        std::vector<Q8_1Block> quantized(static_cast<size_t>(rows) * blocks);
        quantize_activations_to_q8_1(values.data(), quantized.data(), rows, k, blocks);
        return quantized;
    }

    /** @return Percentile of sorted samples. */
    double percentile(const std::vector<double> &sorted, double fraction)
    {
        const size_t index = std::min(sorted.size() - 1,
            static_cast<size_t>(fraction * static_cast<double>(sorted.size() - 1) + 0.5));
        return sorted[index];
    }
} // namespace

TEST(Perf__CPUNativeNEONExactOverlay, CollectAuthenticatedCandidates)
{
    const std::string output_path = envString("LLAMINAR_NEON_OVERLAY_CSV", "");
    ASSERT_FALSE(output_path.empty()) << "LLAMINAR_NEON_OVERLAY_CSV is required";
    omp_set_dynamic(0);
    const int workers = omp_get_max_threads();
    const long rounds = positive("LLAMINAR_NEON_OVERLAY_ROUNDS", 15);
    const long max_elements = positive("LLAMINAR_NEON_OVERLAY_MAX_WEIGHT_ELEMENTS", 1L << 31);

    std::vector<int> m_values;
    for (const auto &token : csv(envString("LLAMINAR_NEON_OVERLAY_M", "1,2,3,4,5,6,7,8,12,15,16,31,64")))
        m_values.push_back(std::stoi(token));
    ASSERT_FALSE(m_values.empty());
    const int maximum_m = *std::max_element(m_values.begin(), m_values.end());

    const std::string format_request = envString("LLAMINAR_NEON_OVERLAY_FORMATS", "Q4_K");
    std::vector<const QuantizedVerifierFormatCase *> formats;
    for (const auto &format : quantizedVerifierFormats())
    {
        const auto requested = csv(format_request);
        if (format_request == "all" ||
            std::find(requested.begin(), requested.end(), format.label) != requested.end())
            formats.push_back(&format);
    }
    ASSERT_FALSE(formats.empty()) << "No known format in LLAMINAR_NEON_OVERLAY_FORMATS";

    const auto shape_filter = csv(envString("LLAMINAR_NEON_OVERLAY_SHAPES", ""));
    std::vector<NativeQuantisedShapeSpec> shapes;
    for (const auto &shape : nativeQuantisedShapeManifest())
        if (shape.exact_overlay &&
            (shape_filter.empty() ||
             std::find(shape_filter.begin(), shape_filter.end(), shape.name) != shape_filter.end()) &&
            static_cast<long>(shape.N) * shape.K <= max_elements)
            shapes.push_back(shape);
    ASSERT_FALSE(shapes.empty()) << "No exact-overlay shape selected";

    std::ofstream output(output_path, std::ios::app);
    ASSERT_TRUE(output) << output_path;
    output.seekp(0, std::ios::end);
    if (output.tellp() == 0)
        output << "schema,isa,codebook,superblock,source_format,shape,n,k,m,workers,tile,nbc,"
                  "samples,median_us,p10_us,p90_us,byte_exact\n";

    for (const auto *format : formats)
        for (const auto &shape : shapes)
        {
            SCOPED_TRACE(std::string(format->label) + " " + shape.name);
            auto weights = format->create({static_cast<size_t>(shape.N), static_cast<size_t>(shape.K)}, 0xA11Au);
            ASSERT_NE(weights, nullptr);
            CPUNativeQuantisedGemmKernel kernel(weights.get());
            ASSERT_TRUE(kernel.isValid());
            const auto &packed = kernel.packedWeights();
            const auto input = activations(maximum_m, shape.K, packed.blocks_per_row);
            std::vector<float> oracle(static_cast<size_t>(maximum_m) * shape.N);
            for (int row = 0; row < maximum_m; ++row)
                gemv_native_quantised_preq(packed, input.data() + static_cast<size_t>(row) * packed.blocks_per_row,
                    oracle.data() + static_cast<size_t>(row) * shape.N, {}, ISAPath::AUTO,
                    DecodeSchedulePolicy::Nbc1);
            std::vector<float> out(oracle.size());

            for (const int m : m_values)
            {
                std::vector<Candidate> candidates;
                for (const int chunks : {1, 2, 4, 8})
                {
                    if (m == 1)
                        candidates.push_back({NEONProjectionRowTile::Four, chunks});
                    else
                        for (const auto tile : {NEONProjectionRowTile::Pair, NEONProjectionRowTile::Four})
                            candidates.push_back({tile, chunks});
                }
                const auto launch = [&](const Candidate &candidate)
                {
                    if (m == 1)
                        gemv_native_quantised_preq(packed, input.data(), out.data(), {}, ISAPath::AUTO,
                            chunkPolicy(candidate.chunks));
                    else
                        gemm_native_quantised_preq_decode_equivalent_rows(packed, input.data(), out.data(), {}, m,
                            shape.N, ISAPath::AUTO, tilePolicy(candidate.tile), false, candidate.chunks);
                };
                std::vector<bool> exact(candidates.size());
                for (size_t index = 0; index < candidates.size(); ++index)
                {
                    std::fill(out.begin(), out.end(), -1.0f);
                    launch(candidates[index]);
                    exact[index] = std::memcmp(out.data(), oracle.data(),
                        static_cast<size_t>(m) * shape.N * sizeof(float)) == 0;
                    EXPECT_TRUE(exact[index]) << "M=" << m << " tile=" << tileName(candidates[index].tile)
                                              << " nbc=" << candidates[index].chunks;
                }
                std::vector<std::vector<double>> samples(candidates.size());
                for (long round = 0; round < rounds; ++round)
                    for (size_t offset = 0; offset < candidates.size(); ++offset)
                    {
                        const size_t index = (static_cast<size_t>(round) + offset) % candidates.size();
                        const auto begin = std::chrono::steady_clock::now();
                        launch(candidates[index]);
                        const auto end = std::chrono::steady_clock::now();
                        samples[index].push_back(std::chrono::duration<double, std::micro>(end - begin).count());
                    }
                for (size_t index = 0; index < candidates.size(); ++index)
                {
                    auto sorted = samples[index];
                    std::sort(sorted.begin(), sorted.end());
                    output << "neon-exact-overlay-v1," << isaLevelName(activeISALevel()) << ','
                           << static_cast<int>(packed.codebook_id) << ',' << (packed.is_superblock ? 1 : 0) << ','
                           << format->label << ',' << shape.name << ',' << shape.N << ',' << shape.K << ','
                           << m << ',' << workers << ',' << tileName(candidates[index].tile) << ','
                           << candidates[index].chunks << ',' << sorted.size() << ',' << std::fixed
                           << std::setprecision(3) << percentile(sorted, 0.5) << ',' << percentile(sorted, 0.1)
                           << ',' << percentile(sorted, 0.9) << ',' << (exact[index] ? 1 : 0) << '\n';
                }
                output.flush();
            }
        }
}
