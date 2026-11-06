/**
 * @file Test__W4A4Format.cpp
 * @brief Device-free contracts for exact W4A4 storage and immutable operand admission.
 *
 * These tests execute the production admission body with small native blocks.
 * They prove descriptor precedence, metadata identity, integer bounds and exact
 * source views without loading models, initializing devices or inventing a
 * floating representation. Numerical kernel replay has a separate recorded
 * input/output gate; these construction checks are deliberately named contracts.
 */
#include "tensors/W4A4MatrixContract.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <type_traits>

namespace llaminar2::w4a4
{
    namespace
    {
        /**
         * @brief Build small owned native blocks without any dynamic tensor allocation.
         * @tparam Count Compile-time source extent.
         * @param rotation Stored complete row/transform-block width.
         * @return Uniform valid group-64 source headers and zero codes.
         */
        template<std::size_t Count>
        std::array<SourceBlock, Count> sourceBlocks(int rotation)
        {
            std::array<SourceBlock, Count> result{};
            for (auto &block : result)
            {
                block.unit = 0.125f;
                block.descriptor = 0x60004321u;
                block.rotation_length = rotation;
            }
            return result;
        }
    }

    TEST(W4A4FormatContract, AdjacentSignedNibblesAndMultiplierTagsAreExact)
    {
        constexpr std::array<int, 16> expected{0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1};
        for (unsigned bits = 0; bits < 256; ++bits)
            EXPECT_EQ(signedNibble(static_cast<std::uint8_t>(bits)), expected[bits & 15]);
        for (unsigned group = 0; group < 4; ++group)
        {
            EXPECT_EQ(weightMultiplier(0x60004321u, group), group + 1);
            EXPECT_EQ(weightMultiplier(0x40000009u, group), 9);
            EXPECT_EQ(weightMultiplier(0xfffffe00u, group), 1);
        }
        EXPECT_EQ(sizeof(SourceBlock), 144);
        EXPECT_FALSE(std::is_default_constructible_v<MatrixContract>);
    }

    TEST(W4A4FormatContract, NegativeGroupTermsShiftBeforeSumming)
    {
        // Four group dots from the recorded all-one activation/isolated -8 case.
        // -30600 / 16 rounds toward negative infinity independently per group.
        const auto one = shiftedGroupTerm(-56, -8, 0, 255, 1);
        EXPECT_EQ(one, -1913);
        EXPECT_EQ(one * 4, -7652);
        EXPECT_NE(one * 4, (-30600 * 4) >> kTermShift);
        EXPECT_EQ(shiftedGroupTerm(4096, -512, 8, 255, 15), 979200);
        EXPECT_EQ(kMaximumGroupTerm, 1836000);
    }

    TEST(W4A4FormatContract, WholeRowsRetainExactUnitsClipAndViewExtents)
    {
        auto source = sourceBlocks<4>(512);
        source[2].unit = source[3].unit = 0.25f;
        const auto matrix = MatrixContract::admit(Transform::Identity, 2, 512, source);
        ASSERT_TRUE(matrix) << matrix.error().message();
        EXPECT_EQ(matrix->rows(), 2);
        EXPECT_EQ(matrix->columns(), 512);
        EXPECT_EQ(matrix->blocksPerRow(), 2);
        EXPECT_EQ(matrix->groupsPerRow(), 8);
        EXPECT_EQ(matrix->sourceBytes(), 576);
        EXPECT_EQ(matrix->encodedClipBits(), 0u);
        EXPECT_EQ(matrix->activationClip(), 1.0f);
        EXPECT_FALSE(matrix->hadamard());
        const auto row = MatrixContract::admit(Transform::Identity, 1, 512,
                                              std::span<const SourceBlock>(source).subspan(2));
        ASSERT_TRUE(row) << row.error().message();
        EXPECT_EQ(row->sourceBytes(), 288);
        // A physical K fragment cannot choose a new local activation unit.
        const auto fragment = MatrixContract::admit(Transform::Identity, 2, 256,
                                                   std::span<const SourceBlock>(source).first(2));
        ASSERT_FALSE(fragment);
        EXPECT_EQ(fragment.error().failure, AdmissionFailure::InvalidRotation);
    }

    TEST(W4A4FormatContract, NativeTransformFactorizationIsOrderedAndGeometryBound)
    {
        constexpr std::array<HadamardGeometry, 6> geometries{{
            {1, 1, 1}, {256, 1, 256}, {1280, 20, 64},
            {1536, 12, 128}, {4352, 68, 64}, {5120, 20, 256}}};
        for (const auto expected : geometries)
        {
            const auto actual = factorHadamard(expected.length);
            ASSERT_TRUE(actual);
            EXPECT_EQ(actual->length, expected.length);
            EXPECT_EQ(actual->paley, expected.paley);
            EXPECT_EQ(actual->sylvester, expected.sylvester);
        }
        for (int length : {0, -1, 7, 14, 17}) EXPECT_FALSE(factorHadamard(length));
        auto source = sourceBlocks<68>(4352);
        const auto matrix = MatrixContract::admit(Transform::PaleySylvester, 1, 17408, source);
        ASSERT_TRUE(matrix) << matrix.error().message();
        ASSERT_TRUE(matrix->hadamard());
        EXPECT_EQ(matrix->hadamard()->paley, 68);
        EXPECT_EQ(matrix->rotationLength(), 4352);
        EXPECT_EQ(matrix->groupsPerRow(), 272);
        source.front().rotation_length = 1536;
        const auto wrong = MatrixContract::admit(Transform::PaleySylvester, 1, 17408, source);
        ASSERT_FALSE(wrong);
        EXPECT_EQ(wrong.error().failure, AdmissionFailure::InvalidRotation);
    }

    TEST(W4A4FormatContract, NativeExtentChecksShapeAndEveryAccumulationBoundary)
    {
        for (std::size_t columns : {256u, 512u, 5120u, 6144u, 17408u, 32768u})
        {
            const auto extent = MatrixContract::sourceExtent(3, columns);
            ASSERT_TRUE(extent);
            EXPECT_EQ(*extent, 3 * columns / 256 * 144);
        }
        EXPECT_FALSE(MatrixContract::sourceExtent(0, 256));
        EXPECT_FALSE(MatrixContract::sourceExtent(1, 0));
        EXPECT_FALSE(MatrixContract::sourceExtent(1, 255));
        EXPECT_FALSE(MatrixContract::sourceExtent(static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1, 256));
        constexpr std::size_t last_width = (kMaximumGroups / kGroupsPerBlock) * kBlockElements;
        EXPECT_TRUE(MatrixContract::sourceExtent(1, last_width));
        const auto too_wide = MatrixContract::sourceExtent(1, last_width + kBlockElements);
        ASSERT_FALSE(too_wide);
        EXPECT_EQ(too_wide.error().failure, AdmissionFailure::IntegerRange);
        auto source = sourceBlocks<2>(512);
        EXPECT_FALSE(MatrixContract::admit(Transform::Identity, 1, 512,
                                          std::span<const SourceBlock>(source).first(1)));
        EXPECT_FALSE(MatrixContract::admit(static_cast<Transform>(99), 1, 512, source));
    }

    TEST(W4A4FormatContract, IncrementalAdmissionRetainsWholeSourcePolicyAndSinglePublication)
    {
        static_assert(!std::is_copy_constructible_v<MatrixAdmission>);
        static_assert(!std::is_default_constructible_v<MatrixAdmission>);
        for (auto transform : {Transform::Identity, Transform::PaleySylvester})
        {
            const auto source = sourceBlocks<6>(transform == Transform::Identity ? 512 : 128);
            const auto bytes = std::as_bytes(std::span(source));
            for (size_t width = 1; width <= source.size(); ++width)
            {
                auto original = MatrixAdmission::begin(transform, 3, 512);
                ASSERT_TRUE(original);
                ASSERT_TRUE(original->append(bytes.first(width * sizeof(SourceBlock))));
                auto current = std::move(*original);
                EXPECT_FALSE(original->append({}));
                EXPECT_FALSE(std::move(*original).finish());
                for (size_t first = width; first < source.size(); first += width)
                    ASSERT_TRUE(current.append(bytes.subspan(first * sizeof(SourceBlock),
                        std::min(width, source.size() - first) * sizeof(SourceBlock))));
                const auto result = std::move(current).finish();
                ASSERT_TRUE(result);
                EXPECT_EQ(result->rows(), 3);
                EXPECT_EQ(result->columns(), 512);
                EXPECT_EQ(result->sourceBytes(), bytes.size());
                EXPECT_EQ(result->rotationLength(), source.front().rotation_length);
                EXPECT_EQ(result->activationClip(), 1.0f);
                EXPECT_FALSE(std::move(current).finish());
                EXPECT_FALSE(current.append(bytes));
            }
            // Transport has byte alignment, not a promise of SourceBlock alignment.
            std::array<std::byte, sizeof(source) + 1> unaligned{};
            std::memcpy(unaligned.data() + 1, source.data(), sizeof(source));
            auto unaligned_admission = MatrixAdmission::begin(transform, 3, 512);
            ASSERT_TRUE(unaligned_admission);
            ASSERT_TRUE(unaligned_admission->append(std::span(unaligned).subspan(1)));
            EXPECT_TRUE(std::move(*unaligned_admission).finish());

            auto incomplete = MatrixAdmission::begin(transform, 3, 512);
            ASSERT_TRUE(incomplete);
            ASSERT_TRUE(incomplete->append(bytes.first(sizeof(SourceBlock))));
            const auto missing = std::move(*incomplete).finish();
            ASSERT_FALSE(missing);
            EXPECT_EQ(missing.error().failure, AdmissionFailure::SourceExtent);
            auto partial_block = MatrixAdmission::begin(transform, 3, 512);
            ASSERT_TRUE(partial_block);
            EXPECT_FALSE(partial_block->append(bytes.first(sizeof(SourceBlock) - 1)));
            EXPECT_FALSE(partial_block->append(bytes));
            EXPECT_FALSE(std::move(*partial_block).finish());
        }
    }

    TEST(W4A4FormatContract, IncrementalFailureRetainsOriginalCoordinatesAndCannotRecover)
    {
        for (auto transform : {Transform::Identity, Transform::PaleySylvester})
        for (auto failure : {AdmissionFailure::InvalidUnit, AdmissionFailure::InconsistentRowUnit,
                             AdmissionFailure::InconsistentRotation, AdmissionFailure::InconsistentClip,
                             AdmissionFailure::ZeroMultiplier})
        {
            auto source = sourceBlocks<6>(transform == Transform::Identity ? 512 : 128);
            auto &bad = source.back();
            switch (failure)
            {
            case AdmissionFailure::InvalidUnit: bad.unit = -1.0f; break;
            case AdmissionFailure::InconsistentRowUnit: bad.unit = 0.25f; break;
            case AdmissionFailure::InconsistentRotation: bad.rotation_length /= 2; break;
            case AdmissionFailure::InconsistentClip: bad.clip_bits = std::bit_cast<uint32_t>(0.5f); break;
            case AdmissionFailure::ZeroMultiplier: bad.descriptor = 0x60004320u; break;
            default: FAIL() << "Unsupported admission fixture";
            }
            const auto bytes = std::as_bytes(std::span(source));
            auto admission = MatrixAdmission::begin(transform, 3, 512);
            ASSERT_TRUE(admission);
            for (size_t block = 0; block + 1 < source.size(); ++block)
                ASSERT_TRUE(admission->append(bytes.subspan(block * sizeof(SourceBlock), sizeof(SourceBlock))));
            const auto rejected = admission->append(bytes.last(sizeof(SourceBlock)));
            ASSERT_FALSE(rejected);
            EXPECT_EQ(rejected.error().failure, failure);
            ASSERT_TRUE(rejected.error().location);
            EXPECT_EQ(rejected.error().location->row, 2u);
            EXPECT_EQ(rejected.error().location->block, 1u);
            const auto retry = admission->append({});
            ASSERT_FALSE(retry);
            EXPECT_EQ(retry.error().failure, failure);
            const auto result = std::move(*admission).finish();
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error().failure, failure);
            ASSERT_TRUE(result.error().location);
            EXPECT_EQ(result.error().location->row, 2u);
            EXPECT_EQ(result.error().location->block, 1u);
        }
    }

    TEST(W4A4FormatContract, MalformedMetadataReportsTheExactBlock)
    {
        for (const auto failure : {AdmissionFailure::InvalidUnit, AdmissionFailure::InconsistentRowUnit,
                                   AdmissionFailure::InconsistentRotation, AdmissionFailure::InconsistentClip,
                                   AdmissionFailure::ZeroMultiplier})
        {
            auto source = sourceBlocks<4>(512);
            auto &bad = source.back();
            switch (failure)
            {
            case AdmissionFailure::InvalidUnit: bad.unit = std::numeric_limits<float>::quiet_NaN(); break;
            case AdmissionFailure::InconsistentRowUnit: bad.unit = 0.25f; break;
            case AdmissionFailure::InconsistentRotation: bad.rotation_length = 256; break;
            case AdmissionFailure::InconsistentClip: bad.clip_bits = std::bit_cast<std::uint32_t>(0.75f); break;
            case AdmissionFailure::ZeroMultiplier: bad.descriptor = 0x60004320u; break;
            default: FAIL() << "Unexpected fixture mutation";
            }
            const auto result = MatrixContract::admit(Transform::Identity, 2, 512, source);
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error().failure, failure);
            ASSERT_TRUE(result.error().location);
            EXPECT_EQ(result.error().location->row, 1);
            EXPECT_EQ(result.error().location->block, 1);
            EXPECT_NE(result.error().message()[0], '\0');
        }
        for (float clip : {-1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
        {
            auto source = sourceBlocks<1>(256);
            source.front().clip_bits = std::bit_cast<std::uint32_t>(clip);
            const auto result = MatrixContract::admit(Transform::Identity, 1, 256, source);
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error().failure, AdmissionFailure::InvalidClip);
        }
    }
} // namespace llaminar2::w4a4
