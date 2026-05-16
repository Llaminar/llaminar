#include "execution/moe/MoEGraphRoleRunner.h"
#include "mocks/MockMPIContext.h"

#include <gtest/gtest.h>

#include <string>
#include <type_traits>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        template <typename T, typename = void>
        struct has_submitDomainWork : std::false_type
        {
        };

        template <typename T>
        struct has_submitDomainWork<T, std::void_t<decltype(&T::submitDomainWork)>> : std::true_type
        {
        };

        template <typename T, typename = void>
        struct has_dispatchRows : std::false_type
        {
        };

        template <typename T>
        struct has_dispatchRows<T, std::void_t<decltype(&T::dispatchRows)>> : std::true_type
        {
        };

        template <typename T, typename = void>
        struct has_runDomain : std::false_type
        {
        };

        template <typename T>
        struct has_runDomain<T, std::void_t<decltype(&T::runDomain)>> : std::true_type
        {
        };

        static_assert(std::is_same_v<decltype(&MoEGraphRoleRunner::forward),
                                     bool (MoEGraphRoleRunner::*)(const int *, int)>);
        static_assert(!has_submitDomainWork<MoEGraphRoleRunner>::value);
        static_assert(!has_dispatchRows<MoEGraphRoleRunner>::value);
        static_assert(!has_runDomain<MoEGraphRoleRunner>::value);

    } // namespace

    TEST(Test__MoEGraphRoleRunner_NoTensorPayloadAPI, CpuParticipantIdentityIncludesWorldRankAndParticipantId)
    {
        MoEGraphParticipantSpec cpu_spec;
        cpu_spec.world_rank = 3;
        cpu_spec.participant_id = 7;
        cpu_spec.device = DeviceId::cpu();
        cpu_spec.role = MoEGraphParticipantRole::CpuExpert;

        const std::string key = MoEGraphRoleRunner::participantIdentityKey(cpu_spec);
        EXPECT_NE(key.find("world=3"), std::string::npos);
        EXPECT_NE(key.find("participant=7"), std::string::npos);

        MoEGraphParticipantSpec same_participant_different_rank = cpu_spec;
        same_participant_different_rank.world_rank = 4;
        EXPECT_NE(MoEGraphRoleRunner::participantIdentityKey(cpu_spec),
                  MoEGraphRoleRunner::participantIdentityKey(same_participant_different_rank));

        MoEGraphParticipantSpec same_rank_different_participant = cpu_spec;
        same_rank_different_participant.participant_id = 8;
        EXPECT_NE(MoEGraphRoleRunner::participantIdentityKey(cpu_spec),
                  MoEGraphRoleRunner::participantIdentityKey(same_rank_different_participant));
    }

    TEST(Test__MoEGraphRoleRunner_NoTensorPayloadAPI, NoLocalParticipantsStillCompletesScalarStep)
    {
        MoEGraphRoleRunner::Config config;
        config.participant_specs = {
            MoEGraphParticipantSpec{.world_rank = 0,
                                    .participant_id = 0,
                                    .device = DeviceId::cpu(),
                                    .role = MoEGraphParticipantRole::Relay,
                                    .continuation_root = true},
        };
        config.mpi_ctx = std::make_shared<MockMPIContext>(0, 1);
        config.continuation_root = true;
        config.architecture = "qwen35moe";

        MoEGraphRoleRunner runner(std::move(config));

        EXPECT_TRUE(runner.forward(nullptr, 0));
        const auto &header = runner.lastStepHeader();
        EXPECT_EQ(header.generation_id, 1u);
        EXPECT_EQ(header.step_id, 0u);
        EXPECT_EQ(header.seq_len, 0);

        const auto &statuses = runner.lastStatuses();
        ASSERT_EQ(statuses.size(), 1u);
        EXPECT_TRUE(statuses.front().ok);
        EXPECT_EQ(statuses.front().participant.world_rank, 0);
        EXPECT_EQ(statuses.front().participant.participant_id, 0);
    }

} // namespace llaminar2::test
