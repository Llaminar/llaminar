/**
 * @file Test__W4A4GPUTransformPolicy.cpp
 * @brief Device-free source, rounding and identity contracts for GPU H metadata.
 *
 * These tests exercise the production cold policy builder without creating a
 * GPU, launching kernels or certifying arithmetic replay. Real device replay
 * must still compare transformed/A4/output bytes with immutable upstream traces.
 * Host arrays below prove that applying policy changes only borrowed metadata;
 * they are never presented as GPU allocations or executed by a device kernel.
 */
#include "kernels/common/W4A4GPUTransformPolicy.h"
#include "../../utils/W4A4SourceFixture.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <exception>
#include <stdexcept>

namespace llaminar2::w4a4
{
    namespace
    {
        /** @brief Restore the calling thread's rounding policy on every fixture exit. */
        class RoundingScope
        {
        public:
            /** @brief Select an actual host mode. @param mode Required fenv rounding constant.
             * @throws std::runtime_error If the native thread's mode cannot be observed or selected.
             */
            explicit RoundingScope(int mode) : original_(std::fegetround())
            {
                if (original_ == -1 || std::fesetround(mode) != 0)
                    throw std::runtime_error("W4A4 GPU policy contract requires real host rounding control");
            }
            /** @brief Restore the exact entering mode, never leave another test in downward rounding. */
            ~RoundingScope() { if (std::fesetround(original_) != 0) std::terminate(); }
            RoundingScope(const RoundingScope &) = delete;
            RoundingScope &operator=(const RoundingScope &) = delete;
        private:
            int original_; ///< Original thread-owned arithmetic mode.
        };

        /**
         * @brief Admit genuine complete source rows before deriving policy.
         * @param columns Complete original K.
         * @param rotation Source-owned transform width, not a K execution shard.
         * @return Sealed two-row native H source policy.
         */
        MatrixContract source(int columns, int rotation)
        {
            auto blocks = test::w4a4SourceBlocks(TensorType::W4A4H, 2, columns);
            for (auto &block : blocks) block.rotation_length = rotation;
            auto admitted = MatrixContract::admit(Transform::PaleySylvester, 2, columns, blocks);
            if (!admitted) throw std::runtime_error(admitted.error().message());
            return *admitted;
        }
    }

    TEST(W4A4GPUTransformPolicyContract, SourceGeometryAndNonzeroResiduesRemainExact)
    {
        RoundingScope rounding(FE_TONEAREST);
        struct Geometry { int columns, rotation, paley, sylvester; };
        const std::array<Geometry, 9> geometries{{
            {256, 1, 1, 1}, {256, 256, 1, 256}, {1280, 1280, 20, 64},
            {1536, 1536, 12, 128}, {4352, 4352, 68, 64}, {5120, 5120, 20, 256},
            {17408, 4352, 68, 64}, {65536, 65536, 1, 65536}, {27648, 27648, 108, 256}}};
        for (const auto &geometry : geometries)
        {
            SCOPED_TRACE(::testing::Message() << "K=" << geometry.columns << " rotation=" << geometry.rotation);
            const auto policy = GPUHadamardPolicy::fromSource(source(geometry.columns, geometry.rotation));
            const auto &identity = policy.captureIdentity();
            EXPECT_EQ(identity.arithmetic_version, kArithmeticVersion);
            EXPECT_EQ(identity.original_columns, geometry.columns);
            EXPECT_EQ(identity.rotation_length, geometry.rotation);
            EXPECT_EQ(identity.paley, geometry.paley);
            EXPECT_EQ(identity.sylvester, geometry.sylvester);
            EXPECT_EQ(identity.residues[0] & 1u, 0u);
            if (geometry.paley == 1)
                EXPECT_EQ(identity.residues, (std::array<std::uint64_t, 2>{}));
            else
            {
                const int prime = geometry.paley - 1;
                // Exact cardinality plus every nonzero square determines the
                // entire residue set, including the zero/unused-bit exclusions.
                EXPECT_EQ(std::popcount(identity.residues[0]) + std::popcount(identity.residues[1]), (prime - 1) / 2);
                for (int value = 1; value < prime; ++value)
                {
                    const int residue = (value * value) % prime;
                    EXPECT_EQ((identity.residues[residue / 64] >> (residue % 64)) & 1u, 1u);
                }
            }
            const auto normalization = std::bit_cast<float>(identity.normalization_bits);
            EXPECT_TRUE(std::isfinite(normalization));
            EXPECT_GT(normalization, 0.0f);
            // Exact power-of-two identities are metadata controls, not a
            // replacement for recorded non-power-of-two device arithmetic.
            if (geometry.rotation == 1) EXPECT_EQ(identity.normalization_bits, 0x3f800000u);
            if (geometry.rotation == 256) EXPECT_EQ(identity.normalization_bits, 0x3d800000u);
        }
        RecordProperty("admitted_transform_geometries", int(geometries.size()));
    }

    TEST(W4A4GPUTransformPolicyContract, ApplyingPolicyChangesOnlyBorrowedArithmeticMetadata)
    {
        RoundingScope rounding(FE_TONEAREST);
        constexpr int columns = 1280;
        std::array<float, 2 * columns> input{}, butterfly{}, transformed{};
        input.fill(1.0f); butterfly.fill(2.0f); transformed.fill(3.0f);
        std::array<std::uint8_t, 2> validity{1, 0};
        device::HadamardView view{.input = input.data(), .input_stride = columns,
            .active_rows = validity.data(), .butterfly = butterfly.data(), .transformed = transformed.data(),
            .residues_low = 0, .residues_high = 0, .rows = 2, .columns = columns,
            .rotation_length = 0, .paley = 0, .sylvester = 0, .normalization = 0.0f};
        const auto policy = GPUHadamardPolicy::fromSource(source(columns, columns));
        const auto before = view;
        policy.applyTo(view);
        EXPECT_EQ(view.input, before.input);
        EXPECT_EQ(view.input_stride, before.input_stride);
        EXPECT_EQ(view.active_rows, before.active_rows);
        EXPECT_EQ(view.butterfly, before.butterfly);
        EXPECT_EQ(view.transformed, before.transformed);
        EXPECT_EQ(view.rows, before.rows);
        EXPECT_EQ(view.columns, before.columns);
        EXPECT_EQ(view.rotation_length, columns);
        EXPECT_EQ(view.paley, 20);
        EXPECT_EQ(view.sylvester, 64);
        EXPECT_EQ(view.residues_low, policy.captureIdentity().residues[0]);
        EXPECT_EQ(view.residues_high, policy.captureIdentity().residues[1]);
        EXPECT_EQ(std::bit_cast<std::uint32_t>(view.normalization), policy.captureIdentity().normalization_bits);
        EXPECT_TRUE(std::all_of(input.begin(), input.end(), [](float value) { return value == 1.0f; }));
        EXPECT_TRUE(std::all_of(butterfly.begin(), butterfly.end(), [](float value) { return value == 2.0f; }));
        EXPECT_TRUE(std::all_of(transformed.begin(), transformed.end(), [](float value) { return value == 3.0f; }));
        EXPECT_EQ(validity, (std::array<std::uint8_t, 2>{1, 0}));
    }

    TEST(W4A4GPUTransformPolicyContract, ForeignKIdentitySourcesAndRoundingCannotPublishPolicy)
    {
        RoundingScope nearest(FE_TONEAREST);
        auto blocks = test::w4a4SourceBlocks(TensorType::W4A4, 2, 512);
        const auto identity = MatrixContract::admit(Transform::Identity, 2, 512, blocks);
        ASSERT_TRUE(identity);
        EXPECT_THROW((void)GPUHadamardPolicy::fromSource(*identity), std::invalid_argument);
        const auto original = source(512, 128);
        const auto policy = GPUHadamardPolicy::fromSource(original);
        device::HadamardView view{};
        view.columns = 256; view.paley = 99; view.residues_low = 123;
        EXPECT_THROW(policy.applyTo(view), std::invalid_argument);
        EXPECT_EQ(view.columns, 256);
        EXPECT_EQ(view.paley, 99);
        EXPECT_EQ(view.residues_low, 123u);
        {
            RoundingScope downward(FE_DOWNWARD);
            EXPECT_THROW((void)GPUHadamardPolicy::fromSource(original), std::invalid_argument);
        }
        EXPECT_EQ(std::fegetround(), FE_TONEAREST);
        EXPECT_NO_THROW((void)GPUHadamardPolicy::fromSource(original));
    }

    TEST(W4A4GPUTransformPolicyContract, CompleteRowSlicesPreserveButDifferentOriginalKChangesIdentity)
    {
        RoundingScope rounding(FE_TONEAREST);
        const auto original = source(5120, 5120);
        const auto slice = original.rowSlice(1, 1);
        ASSERT_TRUE(slice);
        const auto complete = GPUHadamardPolicy::fromSource(original);
        EXPECT_EQ(complete.captureIdentity(), GPUHadamardPolicy::fromSource(*slice).captureIdentity());
        const auto wider = GPUHadamardPolicy::fromSource(source(10240, 5120));
        EXPECT_NE(complete.captureIdentity(), wider.captureIdentity());
        EXPECT_EQ(complete.captureIdentity().residues, wider.captureIdentity().residues);
        EXPECT_EQ(complete.captureIdentity().normalization_bits, wider.captureIdentity().normalization_bits);
    }
} // namespace llaminar2::w4a4
