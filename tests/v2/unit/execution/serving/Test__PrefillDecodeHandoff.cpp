/**
 * @file Test__PrefillDecodeHandoff.cpp
 * @brief §9.5 typed slot-owned prefill/decode handoff contract.
 *
 * Legal chain PrefillPending -> UnsampledReady -> SampledRecorded ->
 * FirstSidecarPending -> FirstSidecarComplete -> Retired; every other move is
 * refused; terminal FP32 logits obligation is 4 x vocabulary per slot.
 * Device-free.
 */

#include <gtest/gtest.h>

#include "execution/serving/PrefillDecodeHandoff.h"

namespace llaminar2::test
{
    TEST(PrefillDecodeHandoff, LegalChainAndRetirement)
    {
        PrefillDecodeHandoff handoff;
        handoff.slot = 3;
        handoff.publishUnsampled();
        EXPECT_EQ(handoff.phase, HandoffPhase::UnsampledReady);
        handoff.markSampledRecorded();
        handoff.markFirstSidecarPending();
        handoff.markFirstSidecarComplete();
        EXPECT_EQ(handoff.phase, HandoffPhase::FirstSidecarComplete);
        handoff.retire();
        EXPECT_EQ(handoff.phase, HandoffPhase::Retired);
        EXPECT_THROW(handoff.retire(), std::logic_error);
    }

    TEST(PrefillDecodeHandoff, InvalidTransitionsRefused)
    {
        PrefillDecodeHandoff h;
        EXPECT_THROW(h.markSampledRecorded(), std::logic_error);   // before publish
        h.publishUnsampled();
        EXPECT_THROW(h.publishUnsampled(), std::logic_error);      // already ready
        EXPECT_THROW(h.markFirstSidecarPending(), std::logic_error); // not sampled
        h.markSampledRecorded();
        EXPECT_THROW(h.markSampledRecorded(), std::logic_error);   // sampled exactly once
    }

    TEST(PrefillDecodeHandoff, TerminalLogitsObligationAndCharge)
    {
        EXPECT_EQ(terminalLogitsDemand(1, 152000), 4u * 152000u);
        EXPECT_EQ(terminalLogitsDemand(2, 1000), 8000u);
        EXPECT_THROW(terminalLogitsDemand(0, 100), std::invalid_argument);
        PhysicalMemoryBOMBuilder bom({
            .world_rank = 0, .device = DeviceId::cpu(),
            .total_bytes = 1u << 30u, .admission_available_bytes = 1u << 30u,
        });
        addTerminalLogitsDemand(bom, 1, 152000);
        const auto built = bom.build();
        EXPECT_EQ(built.charge(PhysicalMemoryOwner::SequenceMetadata).planned_bytes,
                  4u * 152000u);
    }

    TEST(PrefillDecodeHandoff, TerminalLogitsRejectsHugeVocabularyWrap)
    {
        EXPECT_THROW(terminalLogitsDemand(1, 1ull << 62ull), std::overflow_error);
    }
} // namespace llaminar2::test
