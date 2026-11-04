/**
 * @file CPUJointAuthoritativeTargetContract.inc
 * @brief Dense original-source proposal reconstruction and exact typed-target wire semantics.
 *
 * These metadata contracts use the actual authority and tiny original GGUF directory.
 * They grant neither physical readiness nor dynamic maintenance. Cohort widths beyond
 * local sockets are declarations, never claims of additional native participants.
 */
TEST(CPUJointAuthoritativeTargetContract, CompleteSourceTargetsRoundTripWithoutScalarAliases)
{
    using namespace llaminar2;
    using namespace llaminar2::test;
    for (int width = 2; width <= 8; ++width)
    {
        SCOPED_TRACE(width);
        JointPlacementOriginal original(width);
        MoEOverlayResidencyAuthority authority(original.config());
        auto placements = original.plan.placements;
        std::swap(placements[0].routed_expert_tier[0], placements[0].routed_expert_tier[1]);
        auto tx = MoEOverlayJointPlacementCandidateTestAccess::prepareTransaction(authority, placements);
        tx.histogram_window = frozenWindow(17, {7, 11, 3, 5});
        tx.histogram_generation = tx.histogram_window->generation;
        for (auto &edge : tx.migrations)
            edge.activation_count = tx.histogram_window->activationCount(edge.layer_idx, edge.expert_id);
        ASSERT_TRUE(tx.valid());
        const auto plan = authority.exportAuthoritativeResidencyPlan(tx);
        ASSERT_TRUE(plan.valid());
        EXPECT_TRUE(plan.entries[0].candidate_target.isCPUCohort());
        EXPECT_TRUE(plan.entries[1].candidate_target.isWhole());
        EXPECT_TRUE(plan.entries[2].candidate_target.isCPUCohort()); // unchanged also authenticated
        const auto proposal = makeMoEOverlayDistributedResidencyProposal(plan, tx);
        ASSERT_TRUE(proposal.valid());
        std::vector<uint8_t> wire(moeOverlayDistributedResidencyProposalWireBytes(proposal));
        std::string error;
        ASSERT_TRUE(encodeMoEOverlayDistributedResidencyProposal(proposal, wire, &error)) << error;
        EXPECT_EQ(wire.size(), moeOverlayDistributedResidencyProposalWireBytes(1, 4));
        EXPECT_EQ(wire.size(), size_t(96) + moeOverlayDistributedHistogramWireBytes(1, 4) +
                                  size_t(4) * (4 + 4 + 4 + 8 + 8));
        MoEOverlayDistributedResidencyProposal decoded;
        ASSERT_TRUE(decodeMoEOverlayDistributedResidencyProposal(wire, 1, 4, &decoded, &error)) << error;
        EXPECT_EQ(decoded.plan.entries, plan.entries);
        const auto adopted = MoEOverlayJointPlacementCandidateTestAccess::reconstruct(authority, decoded.plan);
        ASSERT_TRUE(adopted.valid());
        EXPECT_EQ(fingerprintMoEOverlayResidencyExecutionPlan(adopted), proposal.execution_fingerprint);
        EXPECT_EQ(authority.snapshot(), tx.previous); // reconstruction never publishes
        EXPECT_THROW(authority.adoptAuthoritativeResidencyPlan(plan), std::logic_error);
        // ABI2 interpreted this same-sized word as an unconditional whole participant.
        auto old = wire;
        old[4] = 2; old[5] = old[6] = old[7] = 0;
        EXPECT_FALSE(decodeMoEOverlayDistributedResidencyProposal(old, 1, 4, &decoded, &error));
        for (size_t index : {size_t(0), size_t(2)})
            for (const auto bad_target : {MoEOverlayRouteTarget(), MoEOverlayRouteTarget::whole(1),
                                          MoEOverlayRouteTarget::cpuCohort(7)})
            {
                auto bad = plan;
                bad.entries[index].candidate_target = bad_target;
                EXPECT_THROW(MoEOverlayJointPlacementCandidateTestAccess::reconstruct(authority, bad),
                             std::exception);
            }
        auto partial = plan;
        partial.entries.pop_back();
        EXPECT_THROW(MoEOverlayJointPlacementCandidateTestAccess::reconstruct(authority, partial),
                     std::invalid_argument);
        // Equal-sized target words do not erase a receiver's original cut/member
        // definition: its independent execution fingerprint must disagree.
        JointPlacementOriginal different_cuts(width == 8 ? 2 : width + 1);
        MoEOverlayResidencyAuthority cut_receiver(different_cuts.config());
        const auto cut_tx = MoEOverlayJointPlacementCandidateTestAccess::reconstruct(cut_receiver, plan);
        EXPECT_NE(fingerprintMoEOverlayResidencyExecutionPlan(cut_tx), proposal.execution_fingerprint);
        JointPlacementOriginal different_members(width);
        for (auto &endpoint : different_members.plan.domains[1].participants)
            endpoint.numa_node += 16;
        const auto changed_source = std::make_shared<const MoERoutedExpertPlacementPlan>(different_members.plan);
        const auto changed_runtime = resolveMoEExpertOverlayRuntimePlan(changed_source,
            {.current_world_rank = 0, .validate_mvp_root_reachability = false});
        different_members.preparation = std::make_shared<const MoEExpertOverlayPreparationPlan>(
            MoEExpertOverlayPreparationPlan::build(*changed_runtime, different_members.loader));
        MoEOverlayResidencyAuthority member_receiver(different_members.config());
        const auto member_tx = MoEOverlayJointPlacementCandidateTestAccess::reconstruct(member_receiver, plan);
        EXPECT_NE(fingerprintMoEOverlayResidencyExecutionPlan(member_tx), proposal.execution_fingerprint);
        MoEOverlayResidencyAuthority foreign(original.config());
        EXPECT_THROW(foreign.exportAuthoritativeResidencyPlan(tx), std::logic_error);
    }
}
