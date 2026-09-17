/**
 * @file Test__PipelineMTPPublicationEdges.inc
 * @brief Device-free proofs of the pipeline accepted-state and wire authorities.
 *
 * Opaque addresses identify owners only; no GPU context or kernel is touched.
 * Every rejection must precede capture, including forbidden owner state
 * on followers and a checkpoint different from the forward's rollback owner.
 */

/** @brief Construct a complete immutable publication identity without device work. */
static MTPSpeculativeStatePublicationStage::Params pipelinePublicationIdentity(
    DeviceId device, int32_t *words, int width, bool tail)
{
    using Publication = MTPSpeculativeStatePublicationStage;
    const auto follower = pipelineFollowerIdentity(words);
    Publication::Params params;
    params.device_id = device;
    params.backend = follower.backend;
    params.authority = tail ? Publication::Authority::CompactOutcome : Publication::Authority::PipelineFollower;
    params.request_count = 1;
    params.verifier_rows_per_request = width;
    params.accepted_restore_rows_device = words + 1;
    params.target_cached_tokens_device = words + 2;
    params.accepted_state_counts_device = words + 3;
    params.publication_ok_flags_device = words + 4;
    params.main_kv_bindings = {{.cache = follower.checkpoint.cache,
        .first_sequence_index = 0, .base_checkpoint_device = words,
        .base_checkpoint_bytes = follower.checkpoint.checkpoint_bytes}};
    if (tail)
    {
        params.outcome_tokens_device = words + 5;
        params.outcome_meta_device = words + 6;
        params.outcome_token_stride = params.outcome_meta_stride = 32;
        params.base_cached_tokens_device = words + 7;
        params.next_condition_tokens_device = words + 8;
        params.all_drafts_accepted_flags_device = words + 9;
        params.stopped_flags_device = words + 10;
        params.next_verifier_condition_tokens_device = words + 11;
        params.penalty_policy_device = words + 12;
        params.generated_token_counts_device = words + 13;
        params.vocab_size = 32;
    }
    return params;
}

/** @test Local outcome policy and the immutable four-word wire have separate identities. */
TEST(Test__PipelineForwardEdges, PublicationOwnsOneAuthorityAcrossEveryParticipant)
{
    for (const bool cuda : {true, false})
    {
        llaminar2::test::MockLocalTPContext transport;
        std::vector<GlobalDeviceAddress> devices;
        for (int ordinal = 0; ordinal < 3; ++ordinal)
            devices.push_back(GlobalDeviceAddress::fromLocalDeviceId(
                cuda ? DeviceId::cuda(ordinal) : DeviceId::rocm(ordinal)));
        transport.setDevices(devices);
        transport.setBackend(cuda ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
        FP32Tensor hidden({16, 32});
        int32_t words[32]{};
        for (int participant = 0; participant < 3; ++participant)
        for (int width = 2; width <= 16; ++width)
        {
            const bool tail = participant == 2;
            PipelineForwardGraphEdges edges(transport, participant, hidden, 32,
                tail ? std::nullopt : std::optional(pipelineFollowerIdentity(words)));
            auto params = pipelinePublicationIdentity(devices[participant].toLocalDeviceId(), words, width, tail);
            EXPECT_NO_THROW(edges.validatePublication(params));
            MTPSpeculativeStatePublicationStage stage(params);
            EXPECT_TRUE(stage.hasSameCaptureIdentity(params));
            EXPECT_FALSE(stage.isCollectiveStage());
            const auto wire = PipelineForwardGraphEdges::PublicationBanks::from(params);
            EXPECT_TRUE(wire.valid());
            EXPECT_EQ(edges.publicationTransport(wire), nullptr) << "Unit admission must never capture GPU work";
            if (tail)
            {
                auto clipped = params;
                clipped.max_state_commit_rows = width - 1;
                EXPECT_FALSE(stage.hasSameCaptureIdentity(clipped));
                EXPECT_EQ(PipelineForwardGraphEdges::PublicationBanks::from(clipped), wire);
            }
            const auto reject = [&](auto change) {
                auto invalid = params;
                change(invalid);
                EXPECT_THROW(edges.validatePublication(invalid), std::invalid_argument);
                EXPECT_TRUE(stage.hasSameCaptureIdentity(params)) << "Rejection cannot mutate the installed identity";
            };
            reject([](auto &p) { p.device_id = DeviceId::cpu(); });
            reject([](auto &p) { p.request_count = 2; });
            reject([](auto &p) { p.verifier_rows_per_request = 1; });
            reject([](auto &p) { p.backend = nullptr; });
            reject([](auto &p) { p.main_kv_bindings.clear(); });
            reject([](auto &p) { p.publication_ok_flags_device = nullptr; });
            reject([](auto &p) { p.accepted_state_counts_device = p.target_cached_tokens_device; });
            reject([](auto &p) { p.authority = MTPSpeculativeStatePublicationStage::Authority::Unbound; });
            if (!tail)
            {
                reject([](auto &p) { p.authority = MTPSpeculativeStatePublicationStage::Authority::CompactOutcome; });
                reject([&](auto &p) { p.outcome_meta_device = words + 6; });
                reject([&](auto &p) { p.generation_control_device = words + 7; });
                reject([&](auto &p) { p.main_kv_bindings[0].base_checkpoint_device = words + 8; });
                reject([](auto &p) { ++p.main_kv_bindings[0].first_sequence_index; });
            }
            else
                reject([](auto &p) { p.authority = MTPSpeculativeStatePublicationStage::Authority::PipelineFollower; });
        }
    }
}
