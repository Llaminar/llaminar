/**
 * @file MoEMigrationCommandAxisContract.inc
 * @brief Preserve recorded logical command axes through physical cycle assembly.
 *
 * The existing four-target closed-flow fixture supplies explicit whole-owner
 * coordinates and command metadata. Replaying its fixed ordered input through
 * the production cycle constructor must not replace each logical objective
 * with the aggregate cycle objective. Empty and partial-work resets exercise
 * the same implementation without native allocation, transport or GPU work.
 */

TEST(MoEMigrationCommandAxisContract, MixedCommandsRetainAxesAcrossEmptyResetAndRepeatedReplay)
{
    using namespace llaminar2;
    struct Command
    {
        int source_tier;
        int source_participant;
        DeviceId source_device;
        int destination_tier;
        int destination_participant;
        DeviceId destination_device;
        MoEOverlayTierMigrationDirection direction;
        MoEOptimizationMovementAxis axis;
    };
    // Version1 fixed command trace from the existing closed-slot-flow geometry.
    // Participant IDs2/3 are distinct ordinary CPU owners, not shard-cohort aliases.
    const std::array<Command, 4> trace{{
        {2, 2, DeviceId::cpu(), 0, 0, DeviceId::cuda(0),
         MoEOverlayTierMigrationDirection::Promotion, MoEOptimizationMovementAxis::TierResidency},
        {0, 0, DeviceId::cuda(0), 1, 1, DeviceId::rocm(0),
         MoEOverlayTierMigrationDirection::Demotion, MoEOptimizationMovementAxis::TierResidency},
        {1, 1, DeviceId::rocm(0), 2, 3, DeviceId::cpu(),
         MoEOverlayTierMigrationDirection::Demotion, MoEOptimizationMovementAxis::Combined},
        {2, 3, DeviceId::cpu(), 2, 2, DeviceId::cpu(),
         MoEOverlayTierMigrationDirection::SamePriority, MoEOptimizationMovementAxis::ParticipantPlacement},
    }};
    const auto owner = [](int expert, int tier, int participant, DeviceId device) {
        return MoEExpertOwner{.layer_idx = 3, .expert_id = expert, .tier_idx = tier,
            .owner_participant = participant, .device = device, .resident = true};
    };
    std::vector<MoEOverlayTierMigration> recorded;
    for (std::size_t index = 0; index < trace.size(); ++index)
    {
        const auto &entry = trace[index];
        recorded.push_back({.layer_idx = 3, .expert_id = int(index),
            .activation_count = 7 + index, .estimated_weight_bytes = 4096,
            .direction = entry.direction, .axis = entry.axis,
            .source = owner(int(index), entry.source_tier, entry.source_participant, entry.source_device),
            .destination = owner(int(index), entry.destination_tier, entry.destination_participant, entry.destination_device)});
    }
    std::vector<std::size_t> recorded_cover;
    for (int replay = 0; replay < 3; ++replay)
    {
        auto working = recorded;
        const auto cycles = buildMoEOverlayMigrationCycles(working);
        ASSERT_EQ(cycles.size(), 1u);
        EXPECT_TRUE(cycles.front().valid(working));
        if (replay == 0) recorded_cover = cycles.front().migration_indices;
        else EXPECT_EQ(cycles.front().migration_indices, recorded_cover);
        ASSERT_EQ(working.size(), recorded.size());
        for (std::size_t index = 0; index < working.size(); ++index)
        {
            EXPECT_EQ(working[index].layer_idx, recorded[index].layer_idx);
            EXPECT_EQ(working[index].expert_id, recorded[index].expert_id);
            EXPECT_EQ(working[index].activation_count, recorded[index].activation_count);
            EXPECT_EQ(working[index].estimated_weight_bytes, recorded[index].estimated_weight_bytes);
            EXPECT_EQ(working[index].direction, recorded[index].direction);
            EXPECT_EQ(working[index].axis, recorded[index].axis);
            EXPECT_EQ(working[index].source, recorded[index].source);
            EXPECT_EQ(working[index].destination, recorded[index].destination);
        }
        // Reset to empty, then a smaller closed peer-only workload. Prior mixed
        // work cannot contaminate subsequent command axes or physical cover.
        working.clear();
        EXPECT_TRUE(buildMoEOverlayMigrationCycles(working).empty());
        working = {recorded.front(), recorded.front()};
        working[0].source = owner(0, 0, 0, DeviceId::cuda(0));
        working[0].destination = owner(0, 0, 1, DeviceId::cuda(1));
        working[1].expert_id = 1;
        working[1].source = owner(1, 0, 1, DeviceId::cuda(1));
        working[1].destination = owner(1, 0, 0, DeviceId::cuda(0));
        for (auto &entry : working)
        {
            entry.axis = MoEOptimizationMovementAxis::ParticipantPlacement;
            entry.direction = MoEOverlayTierMigrationDirection::SamePriority;
        }
        ASSERT_EQ(buildMoEOverlayMigrationCycles(working).size(), 1u);
        for (const auto &entry : working)
            EXPECT_EQ(entry.axis, MoEOptimizationMovementAxis::ParticipantPlacement);
        working.pop_back();
        EXPECT_THROW((void)buildMoEOverlayMigrationCycles(working), std::logic_error);
    }
}
