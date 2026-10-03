/**
 * @file Test__ROCmMoEProjectionBoundary.cpp
 * @brief Captured, all-format proof of lossless routed projection boundaries.
 *
 * The oracle is the installed complete production pipeline. The candidate
 * retains the very same gate/up/SwiGLU bytes, then uses exact native weight
 * column slices for down projection. Every output byte must agree, including
 * inactive rows after full/short/empty replay. Existing serial-row regressions
 * independently certify the complete pipeline's arithmetic.
 */
#include <gtest/gtest.h>
#include "../../../utils/ROCmMoEProjectionBoundaryFixture.h"
#include "kernels/common/DeviceQ8ActivationNumericalContract.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>

namespace
{
    using llaminar2::MoEPrefillProjectionExecution;
    using llaminar2::MoEExpertProjectionOwnership;
    using llaminar2::WeightRole;
    using llaminar2::test::ROCmMoEProjectionBoundaryFixture;
    using llaminar2::test::ProjectionBoundaryGraph;
    using llaminar2::test::checkProjectionHip;

    /** @brief Compare representations, rejecting NaNs and signed-zero drift too. */
    void expectExact(const std::vector<float> &actual, const std::vector<float> &expected)
    {
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < actual.size(); ++i)
        {
            ASSERT_TRUE(std::isfinite(expected[i])) << "nonfinite oracle at " << i;
            ASSERT_EQ(std::bit_cast<uint32_t>(actual[i]), std::bit_cast<uint32_t>(expected[i]))
                << "first changed projection element " << i;
        }
    }

    /** @brief Validate every published Q8 word with the independent CPU contract. */
    void proveHiddenQuantization(ROCmMoEProjectionBoundaryFixture &fixture, int rows)
    {
        const size_t count = static_cast<size_t>(rows) * fixture.width;
        const auto input = fixture.download(fixture.hiddenRows(), count);
        const auto actual = fixture.download(fixture.quantizedHiddenRows(), count);
        const auto scales = fixture.download(fixture.quantizedHiddenScales(), count / 32);
        for (size_t block = 0; block < scales.size(); ++block)
        {
            float maximum = 0.f;
            for (int lane = 0; lane < 32; ++lane)
                maximum = std::max(maximum, std::fabs(input[block * 32 + lane]));
            const float scale = llaminar2::device_q8_activation_contract::scale(maximum);
            ASSERT_EQ(std::bit_cast<uint32_t>(scales[block]), std::bit_cast<uint32_t>(scale)) << "block " << block;
            // Reciprocal is pure per-block data. Computing it once here is
            // identical to the shared contract's per-value call and keeps the
            // CPU oracle cheap across the complete format/geometry sweep.
            const float inverse = llaminar2::device_fp32_contract::reciprocalPositive(scale);
            for (int lane = 0; lane < 32; ++lane)
            {
                const auto index = block * 32 + lane;
                const float value = llaminar2::device_fp32_contract::multiply(input[index], inverse);
                const int expected = std::clamp(static_cast<int>(std::rint(value)), -127, 127);
                ASSERT_EQ(int(actual[index]), expected) << "element " << index;
            }
        }
    }

    /** @brief Prove retained phase/slice graphs without selecting a test kernel. */
    void prove(const llaminar2::test::QuantizedVerifierFormatCase &format,
        int width, int intermediate, const std::vector<int> &row_counts)
    {
        SCOPED_TRACE(format.label);
        ROCmMoEProjectionBoundaryFixture fixture(format, width, intermediate,
            *std::max_element(row_counts.begin(), row_counts.end()));
        constexpr std::array degrees{1, 2, 4, 8};
        std::array<std::vector<llaminar2::DeviceNativeVNNIMatrixDesc *>, degrees.size()> slices;
        std::array<std::vector<float *>, degrees.size()> shard_outputs;
        for (size_t degree_index = 0; degree_index < degrees.size(); ++degree_index)
            for (int shard = 0; shard < degrees[degree_index]; ++shard)
            {
                const auto ownership = MoEExpertProjectionOwnership::gateUpOwnedDownColumns(
                    {fixture.experts, width, intermediate}, shard, degrees[degree_index]);
                const auto down = ownership.projection(WeightRole::MoEExpertDown);
                slices[degree_index].push_back(fixture.slice(ownership));
                shard_outputs[degree_index].push_back(fixture.allocate<float>(
                    static_cast<size_t>(fixture.capacity) * down.rows));
            }
        auto *assembled = fixture.allocate<float>(static_cast<size_t>(fixture.capacity) * width);
        checkProjectionHip(hipStreamSynchronize(fixture.stream), "join projection fixture setup");
        for (const int rows : row_counts)
        {
            SCOPED_TRACE("M=" + std::to_string(rows));
            fixture.reset(rows, rows, 1);
            ProjectionBoundaryGraph reference(fixture.stream, [&] {
                fixture.launch(rows, MoEPrefillProjectionExecution::complete(width));
            });
            for (size_t degree_index = 0; degree_index < degrees.size(); ++degree_index)
            {
                const int degree = degrees[degree_index];
                SCOPED_TRACE("degree=" + std::to_string(degree));
                ProjectionBoundaryGraph candidate(fixture.stream, [&] {
                    fixture.launch(rows, MoEPrefillProjectionExecution::gateUp(width));
                    for (int shard = 0; shard < degree; ++shard)
                    {
                        const auto ownership = MoEExpertProjectionOwnership::gateUpOwnedDownColumns(
                            {fixture.experts, width, intermediate}, shard, degree);
                        const auto down = ownership.projection(WeightRole::MoEExpertDown);
                        fixture.launch(rows,
                            MoEPrefillProjectionExecution::down(down.source_rows, down.first_row, down.rows),
                            slices[degree_index][shard], shard_outputs[degree_index][shard]);
                        // This diagnostic joins disjoint columns on one device.
                        // No arithmetic is performed and no topology is claimed.
                        checkProjectionHip(hipMemcpy2DAsync(assembled + down.first_row,
                            width * sizeof(float), shard_outputs[degree_index][shard],
                            down.rows * sizeof(float), down.rows * sizeof(float), rows,
                            hipMemcpyDeviceToDevice, fixture.stream), "assemble exact column slices");
                    }
                });
                int replay = 0;
                for (const int live : {rows, rows / 2, 0, rows})
                {
                    SCOPED_TRACE("live=" + std::to_string(live));
                    fixture.reset(rows, live, ++replay);
                    reference.replay();
                    if (degree_index == 0) proveHiddenQuantization(fixture, rows);
                    const auto expected = fixture.download(fixture.output, static_cast<size_t>(rows) * width);
                    const auto intermediate_bytes = fixture.download(fixture.swiglu,
                        static_cast<size_t>(live) * fixture.top_k * intermediate);
                    const auto intermediate_scales = fixture.download(fixture.swiglu_scales,
                        static_cast<size_t>(live) * fixture.top_k * (intermediate / 32));
                    candidate.replay();
                    expectExact(fixture.download(assembled, expected.size()), expected);
                    EXPECT_EQ(fixture.download(fixture.swiglu, intermediate_bytes.size()), intermediate_bytes);
                    expectExact(fixture.download(fixture.swiglu_scales, intermediate_scales.size()), intermediate_scales);
                    if (live > 0)
                        EXPECT_TRUE(std::any_of(expected.begin(), expected.end(), [](float x) { return x != 0.0f; }))
                            << "all-zero synthetic data is not a projection witness";
                }
            }
        }
    }
} // namespace

TEST(ROCmMoEProjectionBoundary, CapturedAllFormatsAndColumnDegreesAreByteExact)
{
    for (const auto &format : llaminar2::test::quantizedMoEVerifierFormats())
        prove(format, 512, 256, {1, 2, 3, 15, 16, 17, 33, 65});
}

TEST(ROCmMoEProjectionBoundary, CapturedProductionWidthRetainsFullMatrixArithmetic)
{
    for (const char *name : {"IQ3_S", "IQ4_XS", "Q6_K", "Q8_0"})
        prove(llaminar2::test::quantizedMoEVerifierFormat(name), 2048, 512, {1, 16, 64});
}

TEST(ROCmMoEProjectionBoundary, CapturedHiddenQuantizationCoversPackedWorkgroupBoundaries)
{
    // The hidden quantizer is weight-format independent. The preceding full
    // codebook sweep checks its actual consumers; this focused test isolates
    // row-dispatch boundaries, odd subgroup tails and changing retained inputs.
    const auto &format = llaminar2::test::quantizedMoEVerifierFormat("Q8_0");
    for (const int width : {96, 256, 2048})
    {
        ROCmMoEProjectionBoundaryFixture fixture(format, width, 256, 512);
        std::vector<int> rows{1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 448, 512};
        if (width == 256)
        {
            rows.clear();
            for (int row = 1; row <= 65; ++row) rows.push_back(row);
        }
        for (int row : rows)
        {
            SCOPED_TRACE("K=" + std::to_string(width) + "/M=" + std::to_string(row));
            fixture.reset(row, row, 1);
            ProjectionBoundaryGraph graph(fixture.stream, [&] {
                fixture.launch(row, MoEPrefillProjectionExecution::gateUp(width));
            });
            for (int replay = 0; replay < 3; ++replay)
            {
                fixture.reset(row, replay == 1 ? 0 : row, replay + 3);
                if (replay == 2)
                {
                    // Adjacent subgroups see unrelated maxima; accidental
                    // wave64-wide reduction or broadcast cannot hide behind
                    // equal scales. Include zero and signed-zero blocks.
                    constexpr std::array magnitudes{0.f, .0078125f, .03125f, .25f, 1.f, 16.f};
                    std::vector<float> input(static_cast<size_t>(row) * width);
                    for (size_t i = 0; i < input.size(); ++i)
                        input[i] = float(int((i * 37 + 11) % 257) - 128) * magnitudes[(i / 32) % magnitudes.size()];
                    fixture.publishHiddenRows(input);
                }
                graph.replay();
                proveHiddenQuantization(fixture, row);
            }
        }
    }
}
