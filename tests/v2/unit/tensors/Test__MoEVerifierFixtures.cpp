/**
 * @file Test__MoEVerifierFixtures.cpp
 * @brief Device-free proofs that composed MoE verifier witnesses are non-degenerate.
 *
 * Replica byte equality is useful only when the routed FFN produces actual data.
 * Independent dense row decoding checks payload diversity and representable Q8
 * SwiGLU scales for the hidden row and seeds that exposed the zero IQ4_XS witness.
 */
#include "utils/QuantizedVerifierFormats.h"
#include "tensors/FP16Utils.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;

TEST(MoEVerifierFixtures, IQ4XSRowsAndExpertSlicesHaveIndependentPayloads)
{
    constexpr size_t width = 2048, rows = 513;
    constexpr size_t row_bytes = width / IQ4_XSBlock::BLOCK_SIZE * sizeof(IQ4_XSBlock);
    for (const uint32_t seed : {81301u, 81302u, 910021u})
    {
        SCOPED_TRACE(seed);
        auto weights = createBoundedMoEIQ4XS({rows, width}, seed);
        auto repeated = createBoundedMoEIQ4XS({rows, width}, seed);
        const auto *raw = static_cast<const unsigned char *>(weights->raw_data());
        ASSERT_NE(raw, nullptr);
        EXPECT_EQ(std::memcmp(raw, repeated->raw_data(), rows * row_bytes), 0);
        for (const size_t row : {1u, 2u, 512u})
            EXPECT_NE(std::memcmp(raw, raw + row * row_bytes, row_bytes), 0)
                << "row/expert-slice periodicity at row " << row;
    }
}

TEST(MoEVerifierFixtures, IQ4XSSwiGLUHasFiniteNonzeroRepresentableQ8Scale)
{
    constexpr size_t width = 2048, rows = 16;
    std::array<float, width> hidden{};
    for (size_t col = 0; col < width; ++col)
        hidden[col] = .31f * std::sin(.0071f * static_cast<float>(col + 3)) +
                      .23f * std::cos(.0137f * static_cast<float>(col + 17)) +
                      .11f * std::sin(.0311f * static_cast<float>(col + 29));
    for (const uint32_t seed : {81301u, 910021u})
    {
        SCOPED_TRACE(seed);
        auto gate = createBoundedMoEIQ4XS({rows, width}, seed);
        auto up = createBoundedMoEIQ4XS({rows, width}, seed + 1);
        std::array<float, width> gate_row{}, up_row{};
        float maximum_intermediate = 0;
        size_t positive_gates = 0, negative_gates = 0;
        for (size_t row = 0; row < rows; ++row)
        {
            gate->to_fp32_row(row, gate_row.data());
            up->to_fp32_row(row, up_row.data());
            double g = 0, u = 0;
            for (size_t col = 0; col < width; ++col)
            {
                ASSERT_TRUE(std::isfinite(gate_row[col]));
                ASSERT_TRUE(std::isfinite(up_row[col]));
                g += static_cast<double>(gate_row[col]) * hidden[col];
                u += static_cast<double>(up_row[col]) * hidden[col];
            }
            positive_gates += g > 0;
            negative_gates += g < 0;
            const float intermediate = static_cast<float>(g / (1 + std::exp(-g)) * u);
            ASSERT_TRUE(std::isfinite(intermediate));
            maximum_intermediate = std::max(maximum_intermediate, std::abs(intermediate));
        }
        EXPECT_GT(positive_gates, 0u);
        EXPECT_GT(negative_gates, 0u);
        const float rounded_scale = fp16_to_fp32(fp32_to_fp16(maximum_intermediate / 127.f));
        EXPECT_GT(rounded_scale, 0.f);
        EXPECT_TRUE(std::isfinite(rounded_scale));
    }
}

TEST(MoEVerifierFixtures, AllNativeFormatsBoundComposedInputsToFiniteHalfMetadata)
{
    constexpr float hidden_bound = 48.f * (.011f * 23 + .003f * 8 + .0075f);
    for (const auto &format : quantizedMoEVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        auto gate = format.create({8, 512}, 718101);
        auto up = format.create({8, 512}, 718102);
        const float multiplier = composedMoEVerifierInputMultiplier(*gate, *up, hidden_bound);
        ASSERT_GT(multiplier, 0.f);
        ASSERT_LE(multiplier, 1.f);
        int exponent = 0;
        EXPECT_EQ(std::frexp(multiplier, &exponent), .5f);
        std::array<float, 512> row{};
        for (const TensorBase *weights : {gate.get(), up.get()})
            for (size_t r = 0; r < 8; ++r)
            {
                weights->to_fp32_row(r, row.data());
                double norm = 0;
                for (const float value : row) norm += std::abs(static_cast<double>(value));
                const double projection_bound = norm * multiplier * hidden_bound;
                EXPECT_LE(projection_bound, 512.);
                EXPECT_TRUE(std::isfinite(fp16_to_fp32(fp32_to_fp16(
                    static_cast<float>(projection_bound * projection_bound / 127.)))));
            }
    }
}
