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
    DeviceId device, std::span<int32_t> owner, int width, bool tail)
{
    using Publication = MTPSpeculativeStatePublicationStage;
    auto *words = owner.data();
    const auto follower = pipelineFollowerIdentity(words);
    Publication::Params params;
    params.device_id = device;
    params.backend = follower.backend;
    params.authority = tail ? Publication::Authority::CompactOutcome : Publication::Authority::PipelineFollower;
    params.request_count = 1;
    params.verifier_rows_per_request = width;
    params.accepted_restore_rows_device = owner.subspan(1);
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
            EXPECT_FALSE(edges.preparedPublication(wire)) << "Unit admission must never capture GPU work";
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

namespace llaminar2::test
{
/** @brief Replay the exact cold policy and native rendezvous adapter used by serving setup. */
struct PipelinePublicationCaptureReplayAccess
{
    using Wave = PipelineForwardGraphEdges::NativePublicationCaptureWave;
    /** @brief Reify recorded original identities without a GPU/source creator.
     * This exercises the executor's closed-proof boundary; actual transport
     * issuance still requires the production prepared worker and original banks.
     */
    static DeviceGraphExecutor::RetainedParentCompositionHook recordedPriorChild(
        std::shared_ptr<const uint64_t> owner, const IGPUGraphCapture &wire, ComputeGraph &graph,
        const void *stage, DeviceId device, void *stream, uint64_t workspace,
        DeviceGraphExecutor::RetainedParentCompositionHook::Function compose,
        std::function<void()> validate)
    {
        DeviceGraphExecutor::RetainedParentCompositionHook hook(std::move(compose));
        hook.authority_ = std::shared_ptr<const RetainedParentInputAuthority>(new RetainedParentInputAuthority(
            {}, owner, &wire, &graph, stage, device, stream, *owner, workspace, 0, std::move(validate)));
        return hook;
    }

    static PipelineForwardGraphEdges::PublicationOrder order(const PipelineForwardGraphEdges::CapturedDomain &domain)
    {
        return PipelineForwardGraphEdges::domainPublicationOrder(domain);
    }
};
}

namespace
{
/** @brief Record original native setup operands without a GPU context. */
class PipelinePublicationBoundaryRecorder final : public llaminar2::test::MockLocalTPContext
{
public:
    struct Arrival
    {
        std::string name;
        int member;
        void *stream;
        int timeout_ms;
    };
    std::vector<Arrival> arrivals;
    bool accept = true;

    bool graphCaptureBoundaryOnStream(const std::string &name, int member, void *stream, int timeout_ms) override
    {
        arrivals.push_back({name, member, stream, timeout_ms});
        return accept;
    }
};

/** @brief Establish original descriptor coordinates for the policy replay. */
void configurePublicationCohort(PipelinePublicationBoundaryRecorder &context, int count)
{
    std::vector<GlobalDeviceAddress> devices;
    for (int member = 0; member < count; ++member)
        devices.push_back(GlobalDeviceAddress::rocm(member));
    context.setDevices(std::move(devices));
    context.setBackend(CollectiveBackendType::RCCL);
}

constexpr std::string_view kPublicationBegin =
    "before_begin:phase=capture_begin:step=1:wave=0:identity=implicit:pipeline_mtp_committed_state..pipeline_mtp_committed_state:context=pipeline_mtp_committed_state";
constexpr std::string_view kPublicationEnd =
    "before_begin:phase=capture_end:step=1:wave=0:identity=implicit:pipeline_mtp_committed_state..pipeline_mtp_committed_state:context=pipeline_mtp_committed_state";
constexpr std::string_view kPublicationFailedEnd =
    "before_begin:phase=capture_end_failed:step=1:wave=0:identity=implicit:pipeline_mtp_committed_state..pipeline_mtp_committed_state:context=pipeline_mtp_committed_state";

TEST(Test__PipelineForwardEdges, NativePublicationWaveIncludesPassiveOriginalMembers)
{
    using Access = llaminar2::test::PipelinePublicationCaptureReplayAccess;
    using Order = PipelineForwardGraphEdges::PublicationOrder;
    for (const auto &widths : std::vector<std::vector<int>>{{4,4}, {1,2}, {2,1}, {1,2,4}, {3,5,11}})
    {
        int degree = 0;
        for (const auto width : widths) degree += width;
        PipelinePublicationBoundaryRecorder original;
        configurePublicationCohort(original, degree);
        int member = 0;
        for (std::size_t stage = 0; stage < widths.size(); ++stage)
            for (int local = 0; local < widths[stage]; ++local, ++member)
            {
                PipelineForwardGraphEdges::CapturedDomain domain{
                    .stage_index = stage, .stage_count = widths.size(), .participant = local,
                    .native_context = &original, .native_participant = member};
                const auto expected = stage + 1 < widths.size() ? Order::ExchangeThenCommit :
                    local == 0 ? Order::CommitThenExchange : Order::LocalCommit;
                ASSERT_EQ(Access::order(domain), expected);
                domain.native_context = nullptr;
                EXPECT_EQ(Access::order(domain), expected) << "Transport vendor cannot add a wire role";

                Access::Wave wave(original, member);
                void *const stream = reinterpret_cast<void *>(std::uintptr_t(64 + member));
                if (expected == Order::LocalCommit)
                    ASSERT_TRUE(wave.joinPassive(stream));
                else
                {
                    ASSERT_TRUE(wave.activeBoundary(kPublicationBegin, stream));
                    EXPECT_FALSE(wave.complete());
                    ASSERT_TRUE(wave.activeBoundary(kPublicationEnd, stream));
                }
                EXPECT_TRUE(wave.complete());
                const auto &begin = original.arrivals[2 * member];
                const auto &end = original.arrivals[2 * member + 1];
                EXPECT_EQ(begin.name, "pipeline_forward:mtp_publication_transport:native_capture_begin");
                EXPECT_EQ(end.name, "pipeline_forward:mtp_publication_transport:native_capture_end");
                EXPECT_EQ(begin.member, member);
                EXPECT_EQ(end.member, member);
                EXPECT_EQ(begin.stream, stream);
                EXPECT_EQ(end.stream, stream);
                EXPECT_GT(begin.timeout_ms, 0);
                EXPECT_EQ(end.timeout_ms, begin.timeout_ms);
            }
        EXPECT_EQ(original.arrivals.size(), static_cast<size_t>(2 * degree));
    }
}

TEST(Test__PipelineForwardEdges, NativePublicationWaveRejectsMalformedOrIncompleteCapturePhases)
{
    using Wave = llaminar2::test::PipelinePublicationCaptureReplayAccess::Wave;
    PipelinePublicationBoundaryRecorder original;
    configurePublicationCohort(original, 8);
    void *const stream = reinterpret_cast<void *>(std::uintptr_t(64));
    void *const foreign_stream = reinterpret_cast<void *>(std::uintptr_t(128));
    EXPECT_THROW((void)Wave(original, -1), std::invalid_argument);
    EXPECT_THROW((void)Wave(original, 8), std::invalid_argument);
    for (void *invalid : std::array<void *, 3>{nullptr, reinterpret_cast<void *>(1), reinterpret_cast<void *>(2)})
    {
        Wave passive(original, 7);
        EXPECT_FALSE(passive.joinPassive(invalid));
        EXPECT_FALSE(passive.complete());
    }
    Wave reordered(original, 0);
    EXPECT_FALSE(reordered.activeBoundary(kPublicationEnd, stream));
    EXPECT_FALSE(reordered.activeBoundary(kPublicationBegin, stream));
    Wave foreign(original, 0);
    EXPECT_FALSE(foreign.activeBoundary("before_begin:phase=capture_begin:context=foreign", stream));
    EXPECT_TRUE(original.arrivals.empty());

    Wave duplicate(original, 0);
    ASSERT_TRUE(duplicate.activeBoundary(kPublicationBegin, stream));
    EXPECT_FALSE(duplicate.complete()) << "A missing end cannot publish this setup";
    EXPECT_FALSE(duplicate.activeBoundary(kPublicationBegin, stream));
    EXPECT_FALSE(duplicate.activeBoundary(kPublicationEnd, stream));
    EXPECT_EQ(original.arrivals.size(), 1u);

    Wave changed_stream(original, 0);
    ASSERT_TRUE(changed_stream.activeBoundary(kPublicationBegin, stream));
    EXPECT_FALSE(changed_stream.activeBoundary(kPublicationEnd, foreign_stream));
    EXPECT_FALSE(changed_stream.complete());
    EXPECT_EQ(original.arrivals.size(), 2u);

    Wave failed_capture(original, 0);
    ASSERT_TRUE(failed_capture.activeBoundary(kPublicationBegin, stream));
    ASSERT_TRUE(failed_capture.activeBoundary(kPublicationFailedEnd, stream));
    EXPECT_FALSE(failed_capture.complete());
    ASSERT_EQ(original.arrivals.size(), 4u);
    EXPECT_EQ(original.arrivals.back().name, "pipeline_forward:mtp_publication_transport:native_capture_end");

    original.accept = false;
    Wave rejected(original, 7);
    EXPECT_FALSE(rejected.joinPassive(stream));
    EXPECT_FALSE(rejected.complete());
    EXPECT_EQ(original.arrivals.size(), 5u) << "A failed begin cannot enter end";
}
}

namespace
{
/** @brief Preserve a genuine second input while proving only META's earlier child. */
class PipelinePriorChildReadStage final : public IComputeStage
{
public:
    explicit PipelinePriorChildReadStage(DeviceId device) : IComputeStage(device) {}
    bool execute(IDeviceContext *) override { return true; }
    ComputeStageType type() const override { return ComputeStageType::COPY; }
    std::string name() const override { return "pipeline_prior_child_read"; }
    bool supportsBackend(ComputeBackendType) const override { return true; }
    bool isGraphCapturable() const override { return true; }
    CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
    StageBufferContract bufferContract() const override {
        StageBufferContract result;
        result.addInput(BufferId::STOCHASTIC_BATCH_OUTPUT_META, "INT32");
        result.addInput(BufferId::MTP_LOGICAL_SEQUENCE_STATE, "INT32");
        return result;
    }
protected:
    StageDumpInfo buildDumpInfoImpl() const override { return {}; }
};

/** @test The exact prior-child proof survives repeated use and rejects stale or substituted authority. */
TEST(Test__PipelineForwardEdges, PriorChildFrontierProofReplay)
{
    using Executor = DeviceGraphExecutor;
    using Access = llaminar2::test::PipelinePublicationCaptureReplayAccess;
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        void *const stream = reinterpret_cast<void *>(uintptr_t(64));
        void *const foreign_stream = reinterpret_cast<void *>(uintptr_t(128));
        ComputeGraph graph;
        auto consumer = std::make_unique<PipelinePriorChildReadStage>(device);
        const auto *consumer_identity = consumer.get();
        graph.addNode("local_commit", std::move(consumer), device);
        ComputeGraph foreign_graph;
        FakeReplayGraphCapture wire(stream, false, 1), local(stream, false, 1), parent(stream, false, 0);
        const auto recording = std::make_shared<const uint64_t>(7);
        uint64_t current_recording = *recording;
        size_t validations = 0, compositions = 0;
        const auto make = [&](uint64_t workspace = 11) {
            return Access::recordedPriorChild(recording, wire, graph, consumer_identity, device, stream, workspace,
                [&](IGPUGraphCapture &destination, const ComputeGraph &actual,
                    std::span<const Executor::GraphSegmentCache::RetainedCaptureUnitTemplateView> units) {
                    ++compositions;
                    EXPECT_EQ(&actual, &graph);
                    EXPECT_EQ(units.size(), 1u);
                    EXPECT_EQ(units.front().capture, &local);
                    static_cast<FakeReplayGraphCapture&>(destination).markGraphBuilt();
                    return true;
                }, [&] {
                    ++validations;
                    if (current_recording != *recording)
                        throw std::logic_error("recorded original wire generation changed");
                });
        };
        auto hook = make();
        ASSERT_TRUE(hook.inputAuthority());
        EXPECT_TRUE(hook.inputAuthority()->sameBinding(*make().inputAuthority()));
        EXPECT_FALSE(hook.inputAuthority()->sameBinding(*make(12).inputAuthority()));
        FakeReplayGraphCapture foreign_wire(stream, false, 1);
        const auto foreign_producer = Access::recordedPriorChild(recording, foreign_wire, graph,
            consumer_identity, device, stream, 11,
            [](auto&, const auto&, auto) { return true; }, [] {});
        EXPECT_FALSE(hook.inputAuthority()->sameBinding(*foreign_producer.inputAuthority()));
        const auto unit = Executor::GraphSegmentCache::RetainedCaptureUnitTemplateView{
            .capture = &local, .stream = stream};
        for (int replay = 0; replay < 20; ++replay)
            EXPECT_TRUE(hook(parent, graph, std::span(&unit, 1)));
        EXPECT_EQ(compositions, 20u);
        EXPECT_EQ(validations, 20u);
        EXPECT_THROW(hook.inputAuthority()->require(foreign_graph, device, stream), std::logic_error);
        EXPECT_THROW(hook.inputAuthority()->require(graph, device, foreign_stream), std::logic_error);
        EXPECT_THROW(hook.inputAuthority()->require(graph, DeviceId::cpu(), stream), std::logic_error);
        current_recording = 8;
        EXPECT_THROW(hook(parent, graph, std::span(&unit, 1)), std::logic_error);
        EXPECT_EQ(compositions, 20u) << "A stale source must fail before composition";
        current_recording = 7;

        FakeReplayGPUContext gpu_ctx(0, false, true);
        Executor::GraphSegmentCache cache;
        cache.capture_device = device;
        cache.capture_stream = stream;
        cache.prior_child_input_authority = hook.inputAuthority();
        DeviceGraphCaptureController::buildCapturePlan(graph, cache, nullptr, false, false,
            Executor::GraphReplayPlanPolicy::RequireRetainedParentComposition);
        ASSERT_EQ(cache.segments.size(), 1u);
        EXPECT_EQ(cache.segments.front().retained_parent_input_ids,
            (std::vector<BufferId>{BufferId::STOCHASTIC_BATCH_OUTPUT_META}));
        EXPECT_EQ(graph.getNode("local_commit")->stage->bufferContract().allArenaReads().size(), 2u)
            << "The logical-state input remains strict";

        // Replacing the closed ExchangeThenCommit composer with an arbitrary
        // callback loses its input authority. It cannot reuse the old proof.
        Executor executor;
        llaminar2::testing::MockDeviceContext ctx(device,
            device.type == DeviceType::CUDA ? ComputeBackendType::GPU_CUDA : ComputeBackendType::GPU_ROCM);
        Executor::RetainedParentCompositionHook arbitrary = [](auto&, const auto&, auto) { return true; };
        EXPECT_FALSE(arbitrary.inputAuthority());
        EXPECT_THROW((void)executor.executeWithCachedGraphReplay(graph, &ctx, cache, stream, &gpu_ctx,
            nullptr, false, false, true, {}, Executor::GraphReplayPlanPolicy::RequireRetainedParentComposition,
            {}, {}, arbitrary), std::logic_error);
        cache.prior_child_input_authority.reset();
        cache.initialized = true;
        cache.executable_submission_state = Executor::GraphSegmentCache::ExecutableSubmissionState::ReplayReady;
        EXPECT_THROW((void)executor.executeWithCachedGraphReplay(graph, &ctx, cache, stream, &gpu_ctx,
            nullptr, false, false, true, {}, Executor::GraphReplayPlanPolicy::RequireRetainedParentComposition,
            {}, {}, hook), std::logic_error);
        cache.initialized = false;
        cache.executable_submission_state = Executor::GraphSegmentCache::ExecutableSubmissionState::Empty;
        cache.capture_stream = nullptr; // No native stream was created by this recorded-boundary replay.
        cache.segments.clear();
    }
}
}

/** @test Production issuance rejects a local consumer with no original captured wire source. */
TEST(Test__PipelineForwardEdges, PriorChildFrontierRequiresActualTransport)
{
    for (const bool cuda : {true, false})
    {
        const auto device = cuda ? DeviceId::cuda(0) : DeviceId::rocm(0);
        llaminar2::test::MockLocalTPContext transport;
        transport.setDevices({GlobalDeviceAddress::fromLocalDeviceId(device),
            GlobalDeviceAddress::fromLocalDeviceId(cuda ? DeviceId::cuda(1) : DeviceId::rocm(1))});
        transport.setBackend(cuda ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
        FP32Tensor hidden({16, 32});
        int32_t words[32]{};
        PipelineForwardGraphEdges edges(transport, 0, hidden, 32, pipelineFollowerIdentity(words));
        ComputeGraph graph;
        auto stage = std::make_unique<MTPSpeculativeStatePublicationStage>(
            pipelinePublicationIdentity(device, words, 4, false));
        auto *const original_stage = stage.get();
        graph.addNode("mtp_speculative_state_publication", std::move(stage), device);
        FakeReplayGPUContext gpu(0, false, true);
        try {
            (void)edges.publicationComposer(graph, original_stage, {}, {}, 11, gpu,
                reinterpret_cast<void *>(uintptr_t(64)));
            FAIL() << "An unrecorded wire source cannot mint input authority";
        } catch (const std::logic_error &error) {
            EXPECT_NE(std::string(error.what()).find("original retained transport admission"), std::string::npos);
        }
    }
}
