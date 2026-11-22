/**
 * @file Test__SamplerLaw.cpp
 * @brief §6.1 substrate: admitted per-row sampler laws and penalty-state obligation.
 *
 * Different request laws must be representable without recapture/singleton
 * serialization (the table is the typed substrate); the penalty-state BOM
 * obligation is the exact 4 x vocabulary bytes per slot charge. Device-free.
 */

#include <gtest/gtest.h>

#include "execution/serving/SamplerLaw.h"

namespace llaminar2::test
{
    TEST(SamplerLaw, TableAdmitsDistinctImmutableLaws)
    {
        SamplerLawTable table;
        SamplerLaw greedy;
        greedy.greedy = true;
        SamplerLaw stochastic;
        stochastic.temperature = 0.8f;
        stochastic.top_k = 40;
        stochastic.top_p = 0.95f;
        stochastic.seed_coordinates =
            DrawCoordinates{resolveRequestSeed(12345, 0), 0, DrawPurpose::Sampling};
        table.admit(greedy);
        table.admit(stochastic);
        ASSERT_EQ(table.size(), 2u);
        EXPECT_TRUE(table.at(0).greedy);
        EXPECT_FALSE(table.at(1).greedy);
        EXPECT_FLOAT_EQ(table.at(1).temperature, 0.8f);
        EXPECT_NE(table.at(1).seed_coordinates.seed, 0u); // resolved nonzero
    }

    TEST(SamplerLaw, PenaltyStateObligationIsFourBytesPerVocabularyTokenPerSlot)
    {
        EXPECT_EQ(penaltyStateDemand(1, 152000), 4u * 152000u);
        EXPECT_EQ(penaltyStateDemand(2, 1000), 8000u);
        EXPECT_THROW(penaltyStateDemand(0, 1000), std::invalid_argument);
        EXPECT_THROW(penaltyStateDemand(1, 0), std::invalid_argument);
        EXPECT_THROW(penaltyStateDemand(2, std::numeric_limits<std::uint64_t>::max()),
                     std::overflow_error);
    }

    TEST(SamplerLaw, PenaltyStateChargesExactOwnerLine)
    {
        PhysicalMemoryBOMBuilder bom({
            .world_rank = 0,
            .device = DeviceId::cpu(),
            .total_bytes = 1u << 30u,
            .admission_available_bytes = 1u << 30u,
        });
        addPenaltyStateDemand(bom, 1, 152000);
        const auto built = bom.build();
        EXPECT_EQ(built.charge(PhysicalMemoryOwner::SequenceMetadata).planned_bytes,
                  4u * 152000u);
    }

    TEST(SamplerLaw, PenaltyDemandRejectsHugeVocabularyWrap)
    {
        // vocab = 2^62 wraps a naive per_slot multiplication (zero divisor).
        EXPECT_THROW(penaltyStateDemand(1, 1ull << 62ull), std::overflow_error);
    }
} // namespace llaminar2::test
