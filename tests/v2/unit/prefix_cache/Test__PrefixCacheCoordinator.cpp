#include "execution/prefix_cache/PrefixCacheCoordinator.h"

#include <gtest/gtest.h>
#include <mpi.h>

using namespace llaminar2;

namespace
{
    PrefixParticipantLookup participant(
        int id,
        int tokens,
        bool terminal_logits,
        bool terminal_hidden,
        bool supported = true,
        uint64_t fingerprint_key = 0x1000)
    {
        PrefixParticipantLookup lookup;
        lookup.domain_id = "test-domain";
        lookup.participant_id = id;
        lookup.device = DeviceId::cpu();
        lookup.placement_epoch = 7;
        lookup.fingerprint_key = fingerprint_key;
        lookup.supported = supported;
        lookup.cache_enabled = true;
        lookup.hit = tokens > 0;
        lookup.matched_tokens = tokens;
        lookup.matched_blocks = tokens / 2;
        lookup.has_terminal_logits = terminal_logits;
        lookup.has_terminal_hidden = terminal_hidden;
        if (!supported)
            lookup.bypass_reason = "unsupported participant";
        return lookup;
    }

    class FakeDomainCoordinator : public IPrefixCollectiveCoordinator
    {
    public:
        int min_int = 0;
        bool and_bool = true;
        bool or_bool = true;
        bool fail = false;

        bool allMinInt(int, int *global_value) override
        {
            if (fail)
                return false;
            *global_value = min_int;
            return true;
        }

        bool allMinUInt64(uint64_t, uint64_t *global_value) override
        {
            if (fail)
                return false;
            *global_value = 0x1000;
            return true;
        }

        bool allMaxUInt64(uint64_t, uint64_t *global_value) override
        {
            if (fail)
                return false;
            *global_value = 0x1000;
            return true;
        }

        bool allAndBool(bool, bool *global_value) override
        {
            if (fail)
                return false;
            *global_value = and_bool;
            return true;
        }

        bool allOrBool(bool, bool *global_value) override
        {
            if (fail)
                return false;
            *global_value = or_bool;
            return true;
        }
    };
} // namespace

TEST(Test__PrefixCacheCoordinator, ClampsToMinimumMatchedTokensAndBlocks)
{
    auto result = coordinatePrefixLookups({
        participant(/*id=*/0, /*tokens=*/8, /*terminal_logits=*/true, /*terminal_hidden=*/true),
        participant(/*id=*/1, /*tokens=*/4, /*terminal_logits=*/true, /*terminal_hidden=*/true),
    });

    EXPECT_TRUE(result.supported);
    EXPECT_EQ(result.domain_id, "test-domain");
    EXPECT_EQ(result.placement_epoch, 7u);
    EXPECT_EQ(result.fingerprint_key, 0x1000u);
    EXPECT_TRUE(result.cache_enabled);
    EXPECT_EQ(result.common_matched_tokens, 4);
    EXPECT_EQ(result.common_matched_blocks, 2);
    EXPECT_TRUE(result.common_terminal_logits);
    EXPECT_TRUE(result.common_terminal_hidden);
    EXPECT_NE(result.clamp_reason.find("common prefix"), std::string::npos);
}

TEST(Test__PrefixCacheCoordinator, TerminalStateRequiresEveryParticipant)
{
    auto result = coordinatePrefixLookups({
        participant(/*id=*/0, /*tokens=*/6, /*terminal_logits=*/true, /*terminal_hidden=*/true),
        participant(/*id=*/1, /*tokens=*/6, /*terminal_logits=*/false, /*terminal_hidden=*/true),
    });

    EXPECT_TRUE(result.supported);
    EXPECT_EQ(result.common_matched_tokens, 6);
    EXPECT_FALSE(result.common_terminal_logits);
    EXPECT_TRUE(result.common_terminal_hidden);
    EXPECT_NE(result.clamp_reason.find("terminal state"), std::string::npos);
}

TEST(Test__PrefixCacheCoordinator, UnsupportedParticipantBypassesAndZerosCommonPrefix)
{
    auto result = coordinatePrefixLookups({
        participant(/*id=*/0, /*tokens=*/8, /*terminal_logits=*/true, /*terminal_hidden=*/true),
        participant(/*id=*/1, /*tokens=*/0, /*terminal_logits=*/false, /*terminal_hidden=*/false,
                    /*supported=*/false),
    });

    EXPECT_FALSE(result.supported);
    EXPECT_TRUE(result.cache_enabled);
    EXPECT_EQ(result.common_matched_tokens, 0);
    EXPECT_EQ(result.common_matched_blocks, 0);
    EXPECT_FALSE(result.common_terminal_logits);
    EXPECT_FALSE(result.common_terminal_hidden);
    EXPECT_EQ(result.clamp_reason, "unsupported participant");
}

TEST(Test__PrefixCacheCoordinator, DomainCoordinatorAppliesScalarReductions)
{
    FakeDomainCoordinator coordinator;
    coordinator.min_int = 2;
    coordinator.and_bool = false;
    coordinator.or_bool = true;

    auto result = coordinatePrefixLookups({
        participant(/*id=*/0, /*tokens=*/8, /*terminal_logits=*/true, /*terminal_hidden=*/true),
    },
                                           &coordinator);

    EXPECT_FALSE(result.supported);
    EXPECT_TRUE(result.cache_enabled);
    EXPECT_EQ(result.common_matched_tokens, 0);
    EXPECT_EQ(result.clamp_reason, "at least one prefix participant is unsupported");
}

TEST(Test__PrefixCacheCoordinator, ConvertsLookupResultsIntoParticipants)
{
    PrefixLookupResult hit;
    hit.supported = true;
    hit.cache_enabled = true;
    hit.cached_tokens = 6;
    hit.block_size = 2;
    hit.fingerprint_key = 0xfeed;
    hit.has_terminal_logits = true;
    hit.has_terminal_hidden = false;

    PrefixParticipantLookup participant =
        makePrefixParticipantLookup(/*participant_id=*/3, DeviceId::rocm(1), hit);

    EXPECT_EQ(participant.participant_id, 3);
    EXPECT_EQ(participant.device, DeviceId::rocm(1));
    EXPECT_EQ(participant.fingerprint_key, 0xfeedu);
    EXPECT_TRUE(participant.supported);
    EXPECT_TRUE(participant.cache_enabled);
    EXPECT_TRUE(participant.hit);
    EXPECT_EQ(participant.matched_tokens, 6);
    EXPECT_EQ(participant.matched_blocks, 3);
    EXPECT_TRUE(participant.has_terminal_logits);
    EXPECT_FALSE(participant.has_terminal_hidden);
}

TEST(Test__PrefixCacheCoordinator, BuildsAggregateLookupResultForRunnerCode)
{
    auto coordination = coordinatePrefixLookups({
        participant(/*id=*/0, /*tokens=*/4, /*terminal_logits=*/true, /*terminal_hidden=*/true),
        participant(/*id=*/1, /*tokens=*/4, /*terminal_logits=*/true, /*terminal_hidden=*/true),
    });

    PrefixLookupResult aggregate = makePrefixLookupResult(coordination, /*block_size=*/2);

    EXPECT_TRUE(aggregate.supported);
    EXPECT_TRUE(aggregate.cache_enabled);
    EXPECT_EQ(aggregate.cached_tokens, 4);
    EXPECT_EQ(aggregate.fingerprint_key, 0x1000u);
    EXPECT_EQ(aggregate.block_size, 2);
    EXPECT_TRUE(aggregate.has_terminal_logits);
    EXPECT_TRUE(aggregate.has_terminal_hidden);
    EXPECT_TRUE(aggregate.bypass_reason.empty());
}

TEST(Test__PrefixCacheCoordinator, MPICoordinatorReducesSingleRankScalars)
{
    int initialized = 0;
    ASSERT_EQ(MPI_Initialized(&initialized), MPI_SUCCESS);

    bool initialized_here = false;
    if (!initialized)
    {
        int argc = 0;
        char **argv = nullptr;
        ASSERT_EQ(MPI_Init(&argc, &argv), MPI_SUCCESS);
        initialized_here = true;
    }

    MPIPrefixCollectiveCoordinator coordinator(MPI_COMM_WORLD);

    int min_value = 0;
    EXPECT_TRUE(coordinator.allMinInt(7, &min_value));
    EXPECT_EQ(min_value, 7);

    uint64_t min_u64 = 0;
    EXPECT_TRUE(coordinator.allMinUInt64(11, &min_u64));
    EXPECT_EQ(min_u64, 11u);

    uint64_t max_u64 = 0;
    EXPECT_TRUE(coordinator.allMaxUInt64(11, &max_u64));
    EXPECT_EQ(max_u64, 11u);

    bool bool_value = false;
    EXPECT_TRUE(coordinator.allAndBool(true, &bool_value));
    EXPECT_TRUE(bool_value);
    EXPECT_TRUE(coordinator.allOrBool(false, &bool_value));
    EXPECT_FALSE(bool_value);

    if (initialized_here)
    {
        int finalized = 0;
        ASSERT_EQ(MPI_Finalized(&finalized), MPI_SUCCESS);
        if (!finalized)
            ASSERT_EQ(MPI_Finalize(), MPI_SUCCESS);
    }
}

TEST(Test__PrefixCacheCoordinator, FingerprintMismatchDisablesCommonHit)
{
    auto result = coordinatePrefixLookups({
        participant(/*id=*/0, /*tokens=*/8, /*terminal_logits=*/true, /*terminal_hidden=*/true,
                    /*supported=*/true, /*fingerprint_key=*/0x1000),
        participant(/*id=*/1, /*tokens=*/8, /*terminal_logits=*/true, /*terminal_hidden=*/true,
                    /*supported=*/true, /*fingerprint_key=*/0x2000),
    });

    EXPECT_FALSE(result.supported);
    EXPECT_EQ(result.common_matched_tokens, 0);
    EXPECT_EQ(result.fingerprint_key, 0u);
    EXPECT_NE(result.clamp_reason.find("fingerprint mismatch"), std::string::npos);
}
