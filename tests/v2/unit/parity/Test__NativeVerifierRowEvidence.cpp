/**
 * @file Test__NativeVerifierRowEvidence.cpp
 * @brief Adversarial native byte-evidence tests without models or devices.
 *
 * These tests distinguish strict batch invariance from a high cosine, reject
 * identical invalid payloads, and guard row selection against malformed sizes.
 */
#include "utils/NativeVerifierRowEvidence.h"
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace llaminar2::test
{
    TEST(NativeVerifierRowEvidence, ExactFiniteValuesIncludingIdenticalSignedZerosPass)
    {
        const std::array values{0.0f, -0.0f, 1.0f, -2.5f, std::numeric_limits<float>::denorm_min()};
        EXPECT_TRUE(compareNativeVerifierRow(values, values).passed());
    }

    TEST(NativeVerifierRowEvidence, OneUlpAndSignedZeroChangesFail)
    {
        for (const auto pair : {std::array{1.0f, std::nextafter(1.0f, 2.0f)},
                                std::array{0.0f, -0.0f}})
        {
            const std::array serial{2.0f, pair[0]}, grouped{2.0f, pair[1]};
            const auto result = compareNativeVerifierRow(grouped, serial);
            EXPECT_EQ(result.status, NativeVerifierRowStatus::ByteMismatch);
            EXPECT_EQ(result.first_mismatch, 1);
            EXPECT_NE(result.grouped_bits, result.serial_bits);
        }
    }

    TEST(NativeVerifierRowEvidence, IdenticalNanAndInfinitiesAreNotProof)
    {
        for (const float value : {std::numeric_limits<float>::quiet_NaN(),
                                  std::numeric_limits<float>::infinity(),
                                  -std::numeric_limits<float>::infinity()})
        {
            const std::array values{1.0f, value};
            EXPECT_EQ(compareNativeVerifierRow(values, values).status,
                      NativeVerifierRowStatus::Nonfinite);
        }
    }

    TEST(NativeVerifierRowEvidence, EmptyAndMismatchedRowsFail)
    {
        const std::array values{1.0f, 2.0f};
        EXPECT_FALSE(compareNativeVerifierRow({}, {}).passed());
        EXPECT_FALSE(compareNativeVerifierRow({}, values).passed());
        EXPECT_EQ(compareNativeVerifierRow(values, std::span(values).first(1)).status,
                  NativeVerifierRowStatus::GeometryMismatch);
    }

    TEST(NativeVerifierRowEvidence, EveryRowThroughAndBeyondDepthFifteenUsesExactStride)
    {
        for (std::size_t rows = 2; rows <= 32; ++rows)
        {
            std::vector<float> matrix(rows * 7);
            for (std::size_t i = 0; i < matrix.size(); ++i)
                matrix[i] = static_cast<float>(i);
            for (std::size_t row = 0; row < rows; ++row)
            {
                const auto serial = std::span(matrix).subspan(row * 7, 7);
                EXPECT_TRUE(compareNativeVerifierMatrixRow(matrix, rows, row, serial).passed());
                EXPECT_FALSE(compareNativeVerifierMatrixRow(matrix, rows, (row + 1) % rows, serial).passed());
            }
        }
    }

    TEST(NativeVerifierRowEvidence, InvalidGeometryCannotTruncateOrReadOutsideMatrix)
    {
        const std::array grouped{1.0f, 2.0f, 3.0f, 4.0f};
        const std::array serial{1.0f, 2.0f};
        EXPECT_FALSE(compareNativeVerifierMatrixRow(grouped, 0, 0, serial).passed());
        EXPECT_FALSE(compareNativeVerifierMatrixRow(grouped, 2, 2, serial).passed());
        EXPECT_FALSE(compareNativeVerifierMatrixRow(grouped, 3, 0, serial).passed());
        EXPECT_FALSE(compareNativeVerifierMatrixRow(grouped, std::numeric_limits<std::size_t>::max(), 0, serial).passed());
        EXPECT_FALSE(compareNativeVerifierMatrixRow(std::span(grouped).first(3), 2, 0, serial).passed());
    }

    TEST(NativeVerifierRowEvidence, StageIdentityExcludesSidecarAndSelectedRowAliases)
    {
        for (const auto key : {"EMBEDDING", "FINAL_NORM", "LM_HEAD", "layer0_FFN_NORM",
                               "layer123_MOE_ROUTING_INDICES", "layer2_ATTENTION_OUTPUT_ALLREDUCED",
                               "layer1_PLE_HASH_INDICES", "layer1_PLE_GATHER",
                               "layer1_PLE_GATE", "layer1_PLE_CONVOLUTION"})
            EXPECT_TRUE(isNativeVerifierRowCheckpoint(key)) << key;
        for (const auto key : {"MTP0_LM_HEAD", "LM_HEAD_ROWS_SELECT", "layer_FFN_NORM",
                               "layerX_FFN_NORM", "layer1_unrelated_FFN_NORM", "OTHER_LM_HEAD"})
            EXPECT_FALSE(isNativeVerifierRowCheckpoint(key)) << key;
    }
    /** @test Rejected proposals, not greedy response tokens, own later verifier row inputs. */
    TEST(NativeVerifierRowEvidence, ActualDraftInputsOwnEveryLiveRowAndExcludePadding)
    {
        const std::array<std::int32_t, 3> drafts{31, 7, 29};
        const auto rows = nativeVerifierInputRows(13, drafts, 3, 16);
        EXPECT_EQ(rows, (std::vector<std::int32_t>{13, 31, 7, 29}));
        EXPECT_EQ(rows.size(), 4u);
        const std::array<std::int32_t, 1> clipped{91};
        EXPECT_EQ(nativeVerifierInputRows(13, clipped, 1, 4),
                  (std::vector<std::int32_t>{13, 91}));
        const std::array grouped{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 99.f, 99.f};
        const std::array serial{5.f, 6.f};
        EXPECT_TRUE(compareNativeVerifierMatrixRow(grouped, 4, 2, serial).passed());
        EXPECT_FALSE(compareNativeVerifierMatrixRow(grouped, 4, 1, serial).passed());
    }

    /** @test Reset/stale, missing and padded traces never invent replay inputs. */
    TEST(NativeVerifierRowEvidence, IncompleteOrChangedDraftObservationRejects)
    {
        const std::array<std::int32_t, 2> drafts{31, 7};
        EXPECT_THROW(nativeVerifierInputRows(13, drafts, 3, 4), std::invalid_argument);
        EXPECT_THROW(nativeVerifierInputRows(13, drafts, 1, 4), std::invalid_argument);
        EXPECT_THROW(nativeVerifierInputRows(13, drafts, 2, 2), std::invalid_argument);
        EXPECT_THROW(nativeVerifierInputRows(-1, drafts, 2, 4), std::invalid_argument);
        const std::array<std::int32_t, 1> invalid{-1};
        EXPECT_THROW(nativeVerifierInputRows(13, invalid, 1, 2), std::invalid_argument);
        EXPECT_THROW(nativeVerifierInputRows(13, {}, 0, 2), std::invalid_argument);
    }

    /** @test PLE identities remain exact beyond every floating integer mantissa. */
    TEST(NativeVerifierRowEvidence, IntegerRowsPreserveNonrepresentableSignedIdentities)
    {
        constexpr std::int64_t large = 9'007'199'254'740'993LL;
        const std::array<std::int64_t, 4> grouped{
            17, 23, large, std::numeric_limits<std::int64_t>::max()};
        const std::array<std::int64_t, 2> serial{
            large, std::numeric_limits<std::int64_t>::max()};
        EXPECT_TRUE(compareNativeVerifierIntegerMatrixRow(grouped, 2, 1, serial).passed());
        auto changed = serial;
        --changed[0];
        const auto mismatch = compareNativeVerifierIntegerMatrixRow(grouped, 2, 1, changed);
        EXPECT_EQ(mismatch.status, NativeVerifierRowStatus::ByteMismatch);
        EXPECT_EQ(mismatch.grouped_value, large);
        EXPECT_EQ(mismatch.serial_value, large - 1);
        EXPECT_FALSE(compareNativeVerifierIntegerMatrixRow(grouped, 2, 0, serial).passed());
        EXPECT_FALSE(compareNativeVerifierIntegerMatrixRow(grouped, 3, 1, serial).passed());
        EXPECT_FALSE(compareNativeVerifierIntegerMatrixRow(grouped, 2, 2, serial).passed());
        EXPECT_FALSE(compareNativeVerifierIntegerMatrixRow({}, 2, 0, serial).passed());
    }

}
