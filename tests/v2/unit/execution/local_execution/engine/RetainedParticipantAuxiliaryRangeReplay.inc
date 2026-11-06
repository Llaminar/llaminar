/**
 * @file RetainedParticipantAuxiliaryRangeReplay.inc
 * @brief Actual graph/controller metadata replay for disjoint LocalHot auxiliary bodies.
 *
 * Existing capturable-stage spies exercise declarative topology only. They
 * cannot mint source, native capture, storage or worker permits. The genuine
 * source-role positive remains the original source/native component fixture.
 * Main Read stays on the model lane; auxiliary source/reader work belongs once
 * to the same original retained parent and never to ordinary main replay.
 */
#include "execution/local_execution/graph/RetainedParentTemplateStageCoverage.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "../../../../utils/CPUJointSourceTestPlan.h"

namespace
{
/** @brief Original metadata identity, never native source admission. */
int auxiliary_range_transport_identity;

/** @brief Build a finite reader declaration over existing graph-stage fixtures.
 * @param device Original participant-local GPU label; no device is opened.
 * @param identity Immutable body capture identity.
 * @param follower Preserve the actual heterogeneous follower envelope and marker.
 * @param original Same admitted execution plan, when the native envelope requires one.
 * @param prefix Participant-local stage spelling; it must not alter collective rendezvous identity.
 * @return Complete main/auxiliary graph metadata without a native executable.
 */
ComputeGraph auxiliaryReaderGraph(DeviceId device, std::uint64_t identity = 71, bool follower = false,
    std::shared_ptr<const MoEExpertOverlayExecutionPlan> original = {}, const std::string& prefix = {})
{
    ComputeGraph graph;
    const auto source = device.is_cuda() ? DeviceId::cuda(0) : DeviceId::rocm(0);
    graph.setRetainedParticipantDefinitionDomain({source, &auxiliary_range_transport_identity});
    for (const auto* name : {"probe_ready", "aux_begin", "aux_result", "trunk", "main_read", "aux_retire", "output"})
        graph.addNode(prefix + name, std::make_unique<FakeSegmentStage>(device, true), device);
    graph.addDependency(prefix + "aux_begin", prefix + "probe_ready");
    graph.addDependency(prefix + "aux_result", prefix + "aux_begin");
    graph.addDependency(prefix + "trunk", prefix + "probe_ready");
    graph.addDependency(prefix + "main_read", prefix + "trunk");
    graph.addDependency(prefix + "aux_retire", prefix + "main_read");
    graph.addDependency(prefix + "aux_retire", prefix + "aux_result");
    graph.addDependency(prefix + "output", prefix + "main_read");
    graph.setTerminalNode(prefix + "output");
    if (follower)
    {
        graph.setHeterogeneousTicketUnitContract(prefix + "probe_ready", GraphHeterogeneousTicketUnitContract{
            .identity = "local_hot_follower_before_original_ticket",
            .disposition = GraphHeterogeneousTicketUnitDisposition::BeforeManualBoundary});
        graph.setHeterogeneousTicketUnitContract(prefix + "output", GraphHeterogeneousTicketUnitContract{
            .identity = "local_hot_follower_terminal",
            .disposition = GraphHeterogeneousTicketUnitDisposition::TransactionTerminal});
        graph.setNativeCaptureEnvelope(GraphNativeCaptureEnvelope::HeterogeneousTicketFollowerTransaction);
    }
    graph.setRetainedParticipantAuxiliaryRange({
        .role = RetainedGraphParticipantAuxiliaryRole::Reader,
        .participant = device,
        .nodes = {prefix + "aux_begin", prefix + "aux_result", prefix + "aux_retire"},
        .probe_ready = prefix + "probe_ready", .ready_publication = prefix + "aux_result",
        .main_read = prefix + "main_read", .retirement = prefix + "aux_retire", .identity = identity,
        .original_execution = std::move(original)});
    return graph;
}

/** @brief Rebuild the production main capture plan without any backend/native operation.
 * @param graph Exact existing graph definition.
 * @param policy Explicit retained-parent policy, never an eager fallback.
 * @return Actual controller segment cache with auxiliary nodes omitted.
 */
DeviceGraphExecutor::GraphSegmentCache auxiliaryMainPlan(
    ComputeGraph& graph, DeviceGraphExecutor::GraphReplayPlanPolicy policy =
        DeviceGraphExecutor::GraphReplayPlanPolicy::RequireRetainedParentComposition)
{
    DeviceGraphExecutor::GraphSegmentCache cache;
    cache.graph_replay_plan_policy = policy;
    DeviceGraphCaptureController::buildCapturePlan(graph, cache, nullptr, false, false, policy);
    return cache;
}
}

/** @test Splitting an unnamed unit gives both halves shared semantic identities, independent of stage spelling. */
TEST(Test__GraphSegmentCache, RetainedParticipantAuxiliarySplitTailHasSharedBoundaryIdentityReplay)
{
    for (const auto device : {DeviceId::cuda(1), DeviceId::rocm(1)})
    {
        std::vector<std::string> expected;
        for (const std::string prefix : {"layer1_ple_pooled/", "layer1_ple_pooled_reader/"})
        {
            auto graph = auxiliaryReaderGraph(device, 71, false, {}, prefix);
            const auto generation = graph.topologyGeneration();
            for (int replay = 0; replay < 20; ++replay)
            {
                auto cache = auxiliaryMainPlan(graph);
                ASSERT_EQ(cache.segments.size(), 2u);
                std::vector<std::string> identities;
                for (const auto& unit : cache.segments)
                {
                    EXPECT_FALSE(unit.capture_wave_identity.empty())
                        << "Implicit boundary names leak participant-local stage spelling into TP rendezvous";
                    identities.push_back(unit.capture_wave_identity);
                }
                EXPECT_EQ(identities.front(), "participant_auxiliary_probe");
                EXPECT_EQ(identities.back(), "participant_auxiliary_continuation");
                if (expected.empty()) expected = identities;
                EXPECT_EQ(identities, expected);
                EXPECT_EQ(graph.topologyGeneration(), generation);
                cache.reset();
            }
        }
    }
}

/** @test Real controller main partition retains probe/trunk/read/output once and never duplicates auxiliary work. */
TEST(Test__GraphSegmentCache, RetainedParticipantAuxiliaryMainPartitionReplay)
{
    for (const auto device : {DeviceId::cuda(1), DeviceId::rocm(1)})
        for (const bool follower : {false, true})
            for (int replay = 0; replay < 20; ++replay)
            {
                SCOPED_TRACE(::testing::Message() << device.toString() << " follower=" << follower << " replay=" << replay);
                auto graph = auxiliaryReaderGraph(device, 71, follower);
                const auto declared = *graph.retainedParticipantAuxiliaryRange();
                const auto topology = graph.topologyGeneration();
                const auto variant = DeviceGraphCaptureController::computeCaptureVariantSignature(graph);
                ASSERT_NE(variant, 0u);
                for (int plan = 0; plan < 2; ++plan)
                {
                    auto cache = auxiliaryMainPlan(graph);
                    ASSERT_EQ(cache.segments.size(), 2u);
                    std::vector<std::string> names;
                    for (const auto& segment : cache.segments)
                    {
                        EXPECT_TRUE(segment.capturable);
                        names.insert(names.end(), segment.stage_names.begin(), segment.stage_names.end());
                    }
                    EXPECT_EQ(names, (std::vector<std::string>{"probe_ready", "trunk", "main_read", "output"}));
                    for (const auto& body : declared.nodes)
                        EXPECT_EQ(std::count(names.begin(), names.end(), body), 0);
                    EXPECT_EQ(graph.topologyGeneration(), topology);
                    EXPECT_EQ(*graph.retainedParticipantAuxiliaryRange(), declared);
                    EXPECT_EQ(DeviceGraphCaptureController::computeCaptureVariantSignature(graph), variant);
                }
                // An auxiliary declaration cannot silently change the original
                // execution contract to plain full-graph or eager replay.
                if (follower)
                    EXPECT_THROW((void)auxiliaryMainPlan(graph,
                        DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph), std::logic_error);
                else
                    EXPECT_THROW((void)auxiliaryMainPlan(graph,
                        DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph), std::invalid_argument);
            }
}

/** @test Declaration guards reject role, geometry and edge mutations, and failed binding leaves no partial range. */
TEST(Test__GraphSegmentCache, RetainedParticipantAuxiliaryRejectsForeignAndInvalidRangesReplay)
{
    for (const auto device : {DeviceId::cuda(1), DeviceId::rocm(1)})
        for (int mutation = 0; mutation < 16; ++mutation)
        {
            auto graph = auxiliaryReaderGraph(device);
            auto range = *graph.retainedParticipantAuxiliaryRange();
            // Keep the same actual stages/domain but begin a fresh declaration.
            auto original = std::move(graph);
            ComputeGraph fresh;
            fresh.setRetainedParticipantDefinitionDomain(*original.retainedParticipantDefinitionDomain());
            for (const auto& name : original.getExecutionOrder())
            {
                const auto* node = original.getNode(name);
                fresh.addNode(name, std::make_unique<FakeSegmentStage>(device, true), device);
                for (const auto& dependency : node->dependencies) fresh.addDependency(name, dependency);
            }
            fresh.setTerminalNode("output");
            switch (mutation)
            {
            case 0: range.role = RetainedGraphParticipantAuxiliaryRole::Source; break; // No original source plan.
            case 1: range.role = static_cast<RetainedGraphParticipantAuxiliaryRole>(99); break;
            case 2: range.participant = DeviceId::cpu(); break;
            case 3: range.participant = device.is_cuda() ? DeviceId::cuda(2) : DeviceId::rocm(2); break;
            case 4: range.identity = 0; break;
            case 5: range.nodes.push_back("aux_retire"); break;
            case 6: range.nodes[1] = "missing"; break;
            case 7: range.nodes = {"aux_result", "aux_begin", "aux_retire"}; break;
            case 8: range.nodes.insert(range.nodes.begin(), "main_read"); break;
            case 9: range.probe_ready = "main_read"; break;
            case 10: range.ready_publication = "trunk"; break;
            case 11: range.retirement = "aux_result"; break;
            case 12: fresh.addDependency("aux_begin", "output"); break;
            case 13: fresh.addDependency("aux_result", "aux_result"); break;
            case 14: range.ready_publication = "aux_retire"; break;
            case 15: fresh.addDependency("aux_begin", "main_read"); break;
            }
            const auto generation = fresh.topologyGeneration();
            EXPECT_THROW(fresh.setRetainedParticipantAuxiliaryRange(range), std::invalid_argument) << mutation;
            EXPECT_FALSE(fresh.retainedParticipantAuxiliaryRange()) << mutation;
            EXPECT_EQ(fresh.topologyGeneration(), generation) << mutation;
        }
    auto source_as_reader = auxiliaryReaderGraph(DeviceId::cuda(1));
    source_as_reader.getNode("aux_begin")->device = DeviceId::cuda(0);
    EXPECT_THROW(source_as_reader.requireRetainedParticipantAuxiliaryRange(), std::invalid_argument);
    auto changed_edge = auxiliaryReaderGraph(DeviceId::cuda(1));
    changed_edge.addDependency("aux_begin", "output");
    EXPECT_THROW((void)auxiliaryMainPlan(changed_edge), std::exception);
}

/** @test Duplicate declarations, merge and clear preserve a single exact identity, with no auxiliary leakage after reset. */
TEST(Test__GraphSegmentCache, RetainedParticipantAuxiliaryMergeClearAndIdentityReplay)
{
    auto graph = auxiliaryReaderGraph(DeviceId::cuda(1));
    const auto range = *graph.retainedParticipantAuxiliaryRange();
    const auto variant = DeviceGraphCaptureController::computeCaptureVariantSignature(graph);
    const auto generation = graph.topologyGeneration();
    EXPECT_THROW(graph.setRetainedParticipantAuxiliaryRange(range), std::invalid_argument);
    EXPECT_EQ(graph.topologyGeneration(), generation);
    EXPECT_EQ(*graph.retainedParticipantAuxiliaryRange(), range);
    auto different = auxiliaryReaderGraph(DeviceId::cuda(1), 72);
    EXPECT_NE(DeviceGraphCaptureController::computeCaptureVariantSignature(different), variant);

    // The actual caller prefixes the fragment with an existing model producer.
    // Its external edge enters probe_ready, never an arbitrary auxiliary node.
    ComputeGraph parent;
    parent.addNode("outer_producer", std::make_unique<FakeSegmentStage>(DeviceId::cuda(1), true), DeviceId::cuda(1));
    parent.merge(std::move(graph), "outer_producer");
    ASSERT_TRUE(parent.retainedParticipantAuxiliaryRange());
    EXPECT_EQ(*parent.retainedParticipantAuxiliaryRange(), range);
    EXPECT_FALSE(graph.retainedParticipantAuxiliaryRange());
    EXPECT_NO_THROW(parent.requireRetainedParticipantAuxiliaryRange());
    auto plan = auxiliaryMainPlan(parent);
    ASSERT_EQ(plan.segments.size(), 2u);
    EXPECT_EQ(plan.segments.front().stage_names,
        (std::vector<std::string>{"outer_producer", "probe_ready"}));
    EXPECT_EQ(plan.segments.back().stage_names,
        (std::vector<std::string>{"trunk", "main_read", "output"}));
    EXPECT_EQ(plan.segments.front().capture_wave_identity, "participant_auxiliary_probe");
    const auto count = parent.size();
    const auto topology = parent.topologyGeneration();
    EXPECT_THROW(parent.merge(std::move(different)), std::invalid_argument);
    EXPECT_EQ(parent.size(), count);
    EXPECT_EQ(parent.topologyGeneration(), topology);
    EXPECT_TRUE(different.retainedParticipantAuxiliaryRange());

    auto moved = std::move(parent);
    EXPECT_NO_THROW(moved.requireRetainedParticipantAuxiliaryRange());
    EXPECT_EQ(*moved.retainedParticipantAuxiliaryRange(), range);
    auto moved_plan = auxiliaryMainPlan(moved);
    ASSERT_EQ(moved_plan.segments.size(), plan.segments.size());
    for (std::size_t unit = 0; unit < plan.segments.size(); ++unit)
    {
        EXPECT_EQ(moved_plan.segments[unit].stage_names, plan.segments[unit].stage_names);
        EXPECT_EQ(moved_plan.segments[unit].capture_wave_identity, plan.segments[unit].capture_wave_identity);
    }
    moved.clear();
    EXPECT_FALSE(moved.retainedParticipantAuxiliaryRange());
    EXPECT_FALSE(moved.retainedParticipantDefinitionDomain());
    EXPECT_EQ(DeviceGraphCaptureController::computeCaptureVariantSignature(moved), 0u);
    moved.addNode("fresh", std::make_unique<FakeSegmentStage>(DeviceId::cuda(1), true), DeviceId::cuda(1));
    auto fresh = auxiliaryMainPlan(moved, DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph);
    ASSERT_EQ(fresh.segments.size(), 1u);
    EXPECT_EQ(fresh.segments.front().stage_names, (std::vector<std::string>{"fresh"}));
}

/** @test Device-owned one-executable semantics cannot be downgraded to admit an unwired auxiliary path. */
TEST(Test__GraphSegmentCache, RetainedParticipantAuxiliaryPreservesDeviceOwnedEnvelopeRejection)
{
    auto graph = auxiliaryReaderGraph(DeviceId::cuda(1));
    graph.setNativeCaptureEnvelope(GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction);
    EXPECT_THROW((void)auxiliaryMainPlan(graph), std::logic_error);
    EXPECT_THROW((void)auxiliaryMainPlan(graph,
        DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph), std::invalid_argument);
}

namespace
{
/** @brief Build a real heterogeneous compiler contract around one reader's finite body.
 * @param device Participant-local GPU label; no context or executable is issued.
 * @param authority Whether this participant owns an unrelated original CPU ticket service.
 * @param probeAtEnd Whether an existing logical unit already ends at Probe.
 * @param tag Participant-local stage spelling, deliberately independent of shared unit identity.
 * @return Actual graph declaration with original marker counts and disjoint auxiliary stages.
 * Heterogeneous ticket authority and ngram source roles are independent: this
 * declaration remains an ngram Reader and never fabricates a SourcePlan.
 */
ComputeGraph auxiliaryProbeGraph(DeviceId device, bool authority, bool probeAtEnd, const std::string& tag)
{
    ComputeGraph graph;
    graph.setRetainedParticipantDefinitionDomain({
        device.is_cuda() ? DeviceId::cuda(0) : DeviceId::rocm(0), &auxiliary_range_transport_identity});
    const auto gpu = [&](const std::string& name) {
        graph.addNode(tag + name, std::make_unique<FakeSegmentStage>(device, true), device);
    };
    const auto service = [&](const std::string& name) {
        addFakeSegmentStage(graph, tag + name, false, true, ComputeStageType::MOE_LOCAL_EXPERT,
            false, false, false, nullptr, DeviceId::cpu(), false,
            ManualGraphBoundaryScheduling::ConcurrentTicketService,
            ConcurrentManualFailureRole::DeviceIngressPublisher);
    };
    gpu("ticket0");
    if (authority) service("service0");
    for (const auto* name : {"header", "probe", "aux_begin", "aux_result"}) gpu(name);
    if (authority && probeAtEnd) service("service1");
    for (const auto* name : {"trunk", "ready_wait", "main_read", "aux_retire", "terminal"}) gpu(name);
    if (authority) graph.addDependency(tag + "service0", tag + "ticket0");
    graph.addDependency(tag + "header", tag + (authority ? "service0" : "ticket0"));
    graph.addDependency(tag + "probe", tag + "header");
    graph.addDependency(tag + "aux_begin", tag + "probe");
    graph.addDependency(tag + "aux_result", tag + "aux_begin");
    if (authority && probeAtEnd) graph.addDependency(tag + "service1", tag + "probe");
    graph.addDependency(tag + "trunk", tag + (authority && probeAtEnd ? "service1" : "probe"));
    graph.addDependency(tag + "ready_wait", tag + "trunk");
    graph.addDependency(tag + "main_read", tag + "ready_wait");
    graph.addDependency(tag + "aux_retire", tag + "aux_result");
    graph.addDependency(tag + "aux_retire", tag + "main_read");
    graph.addDependency(tag + "terminal", tag + "main_read");
    graph.setHeterogeneousTicketUnitContract(tag + "ticket0", GraphHeterogeneousTicketUnitContract{
        .identity = "shared_original_ticket_unit"});
    if (probeAtEnd)
        graph.setHeterogeneousTicketUnitContract(tag + "probe", GraphHeterogeneousTicketUnitContract{
            .identity = "shared_original_probe_unit"});
    graph.setHeterogeneousTicketUnitContract(tag + "terminal", GraphHeterogeneousTicketUnitContract{
        .identity = "shared_original_terminal_unit",
        .disposition = GraphHeterogeneousTicketUnitDisposition::TransactionTerminal});
    graph.setTerminalNode(tag + "terminal");
    graph.setNativeCaptureEnvelope(authority ? GraphNativeCaptureEnvelope::HeterogeneousTicketAuthorityTransaction
        : GraphNativeCaptureEnvelope::HeterogeneousTicketFollowerTransaction);
    graph.setRetainedParticipantAuxiliaryRange({.role = RetainedGraphParticipantAuxiliaryRole::Reader,
        .participant = device, .nodes = {tag + "aux_begin", tag + "aux_result", tag + "aux_retire"},
        .probe_ready = tag + "probe", .ready_publication = tag + "aux_result",
        .main_read = tag + "main_read", .retirement = tag + "aux_retire", .identity = 91});
    return graph;
}

/** @return Original retained-parent policy; no new ticket service is created by the probe split. */
DeviceGraphExecutor::GraphReplayPlanPolicy auxiliaryProbePolicy(bool authority)
{
    return authority ? DeviceGraphExecutor::GraphReplayPlanPolicy::RequireRetainedParentWithConcurrentTicketService
        : DeviceGraphExecutor::GraphReplayPlanPolicy::RequireRetainedParentComposition;
}
}

/** @test Actual compiler split exposes Probe, preserves original logical units and shares identity across different legal local stages. */
TEST(Test__GraphSegmentCache, RetainedParticipantAuxiliaryProbeFrontierPartitionReplay)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        for (const bool authority : {false, true})
        {
            std::string sharedPrefixIdentity;
            for (const auto* tag : {"left_", "right_"})
            {
                auto graph = auxiliaryProbeGraph(DeviceId(backend, 1), authority, false, tag);
                const auto topology = graph.topologyGeneration();
                const auto variant = DeviceGraphCaptureController::computeCaptureVariantSignature(graph);
                const auto declaration = *graph.retainedParticipantAuxiliaryRange();
                DeviceGraphExecutor::GraphSegmentCache cache;
                for (int replay = 0; replay < 20; ++replay)
                {
                    DeviceGraphCaptureController::buildCapturePlan(graph, cache, nullptr, false, false,
                        auxiliaryProbePolicy(authority));
                    std::vector<const DeviceGraphExecutor::GraphSegment*> captured;
                    std::size_t manual = 0;
                    for (const auto& unit : cache.segments)
                    {
                        if (unit.capturable) captured.push_back(&unit);
                        else { ++manual; EXPECT_EQ(unit.stage_names, (std::vector<std::string>{std::string(tag) + "service0"})); }
                    }
                    ASSERT_EQ(captured.size(), 3u);
                    EXPECT_EQ(manual, authority ? 1u : 0u);
                    EXPECT_EQ(captured[0]->stage_names, (std::vector<std::string>{std::string(tag) + "ticket0"}));
                    EXPECT_EQ(captured[0]->capture_wave_identity, "shared_original_ticket_unit");
                    EXPECT_EQ(captured[1]->stage_names,
                        (std::vector<std::string>{std::string(tag) + "header", std::string(tag) + "probe"}));
                    EXPECT_EQ(captured[1]->capture_wave_identity,
                        "shared_original_terminal_unit:participant_auxiliary_probe");
                    EXPECT_EQ(captured[2]->stage_names, (std::vector<std::string>{std::string(tag) + "trunk",
                        std::string(tag) + "ready_wait", std::string(tag) + "main_read", std::string(tag) + "terminal"}));
                    EXPECT_EQ(captured[2]->capture_wave_identity, "shared_original_terminal_unit");
                    if (sharedPrefixIdentity.empty()) sharedPrefixIdentity = captured[1]->capture_wave_identity;
                    EXPECT_EQ(captured[1]->capture_wave_identity, sharedPrefixIdentity);
                    EXPECT_EQ(graph.getNode(std::string(tag) + "terminal")->heterogeneous_ticket_unit_contract->disposition,
                        GraphHeterogeneousTicketUnitDisposition::TransactionTerminal);
                    EXPECT_FALSE(graph.getNode(std::string(tag) + "probe")->heterogeneous_ticket_unit_contract);
                    EXPECT_EQ(graph.topologyGeneration(), topology);
                    EXPECT_EQ(*graph.retainedParticipantAuxiliaryRange(), declaration);
                    EXPECT_EQ(DeviceGraphCaptureController::computeCaptureVariantSignature(graph), variant);
                    EXPECT_EQ(cache.graph_replay_plan_policy, auxiliaryProbePolicy(authority));
                    EXPECT_FALSE(cache.initialized);
                    EXPECT_FALSE(cache.retained_parent_capture); // Metadata is not native parent admission.
                    cache.reset();
                }
            }
        }
}

/** @test A probe already closing an original logical unit keeps that unit and its identity through reset/replay. */
TEST(Test__GraphSegmentCache, RetainedParticipantAuxiliaryProbeEndIsNotSplitAgain)
{
    for (const auto device : {DeviceId::cuda(1), DeviceId::rocm(1)})
        for (const bool authority : {false, true})
        {
            auto graph = auxiliaryProbeGraph(device, authority, true, "existing_");
            const auto topology = graph.topologyGeneration();
            for (int replay = 0; replay < 20; ++replay)
            {
                auto cache = auxiliaryMainPlan(graph, auxiliaryProbePolicy(authority));
                std::vector<const DeviceGraphExecutor::GraphSegment*> captured;
                for (const auto& unit : cache.segments) if (unit.capturable) captured.push_back(&unit);
                ASSERT_EQ(captured.size(), 3u);
                EXPECT_EQ(captured[1]->stage_names, (std::vector<std::string>{"existing_header", "existing_probe"}));
                EXPECT_EQ(captured[1]->capture_wave_identity, "shared_original_probe_unit");
                EXPECT_EQ(captured[2]->capture_wave_identity, "shared_original_terminal_unit");
                EXPECT_EQ(captured[2]->stage_names,
                    (std::vector<std::string>{"existing_trunk", "existing_ready_wait", "existing_main_read", "existing_terminal"}));
                EXPECT_EQ(graph.topologyGeneration(), topology);
            }
        }
}

/** @test Actual Controller rejects a public graph mutation placing model Read before the unique Probe frontier. */
TEST(Test__GraphSegmentCache, RetainedParticipantAuxiliaryProbeRequiresLaterMainRead)
{
    for (const auto device : {DeviceId::cuda(1), DeviceId::rocm(1)})
        for (const bool authority : {false, true})
        {
            auto graph = auxiliaryProbeGraph(device, authority, false, "bad_");
            // Public mutable node declarations can be malformed before sealing;
            // force an acyclic early Read and invalidate the order through the
            // normal dependency API, not controller-internal duplicate rows.
            graph.getNode("bad_main_read")->dependencies.clear();
            graph.addDependency("bad_probe", "bad_main_read");
            // Keep the original heterogeneous terminal last, so the actual
            // probe-frontier guard, rather than an unrelated marker failure,
            // diagnoses the early Read on both authority and follower.
            graph.addDependency("bad_terminal", "bad_ready_wait");
            EXPECT_THROW((void)auxiliaryMainPlan(graph, auxiliaryProbePolicy(authority)), std::logic_error);
        }
}

/** @test Main epoch decorators project only main leaves while the parent still owns auxiliary retirement. */
TEST(Test__GraphSegmentCache, RetainedParticipantAuxiliaryMainLeafProjectionReplay)
{
    for (const auto device : {DeviceId::cuda(1), DeviceId::rocm(1)})
        for (int replay = 0; replay < 20; ++replay)
        {
            auto graph = auxiliaryReaderGraph(device, 71, true);
            EXPECT_EQ(graph.getLeafNodes(), (std::vector<std::string>{"aux_retire", "output"}));
            EXPECT_EQ(graph.getLeafNodes(GraphLeafScope::MainExecution), (std::vector<std::string>{"output"}));
            const auto main_leaves = graph.getLeafNodes(GraphLeafScope::MainExecution);
            graph.addNode("epoch_release", std::make_unique<FakeSegmentStage>(device, true), device);
            for (const auto& leaf : main_leaves) graph.addDependency("epoch_release", leaf);
            EXPECT_NO_THROW(graph.moveHeterogeneousTicketTransactionTerminal("output", "epoch_release"));
            EXPECT_EQ(graph.terminalNode(), "epoch_release");
            EXPECT_EQ(graph.getLeafNodes(), (std::vector<std::string>{"aux_retire", "epoch_release"}));
            EXPECT_EQ(graph.getLeafNodes(GraphLeafScope::MainExecution), (std::vector<std::string>{"epoch_release"}));
            EXPECT_TRUE(std::find(graph.getNode("epoch_release")->dependencies.begin(), graph.getNode("epoch_release")->dependencies.end(), "aux_retire") == graph.getNode("epoch_release")->dependencies.end());
            EXPECT_NO_THROW(graph.requireRetainedParticipantAuxiliaryRange());
            EXPECT_THROW(graph.getLeafNodes(static_cast<GraphLeafScope>(255)), std::invalid_argument);
        }
}

namespace
{
/**
 * @brief Supply graph-only recording observations to the actual child-export validator.
 * @param cache The unchanged production compiler partition under test.
 * @param gpu Device-free stream owner; no native source/attachment is fabricated.
 * @note A successful export proves only main-child coverage. The real parent
 * still requires the original auxiliary-body seal before it can instantiate.
 */
void recordAuxiliaryMainChildren(DeviceGraphExecutor::GraphSegmentCache& cache,
                                FakeReplayGPUContext& gpu)
{
    if (!cache.ensureCaptureStream(&gpu))
        throw std::runtime_error("Recorded main-child fixture has no exact stream");
    // Match the executor's atomic graph-only materialization boundary. A
    // partition alone is neither an in-progress capture nor a replay-ready one.
    cache.initialized = false;
    cache.needs_capture = true;
    for (auto& unit : cache.segments)
        if (unit.capturable)
            unit.capture = std::make_unique<FakeReplayGraphCapture>(cache.capture_stream, false);
}
}

/** @test Main child export preserves its exact projection while the original parent owns the disjoint body. */
TEST(Test__GraphSegmentCache, RetainedParticipantAuxiliaryChildExportProjectionReplay)
{
    for (const auto device : {DeviceId::cuda(1), DeviceId::rocm(1)})
        for (const bool authority : {false, true})
            for (const bool probeAtEnd : {false, true})
                for (int replay = 0; replay < 20; ++replay)
                {
                    SCOPED_TRACE(::testing::Message() << device.toString() << " authority=" << authority
                        << " partial=" << probeAtEnd << " replay=" << replay);
                    auto graph = auxiliaryProbeGraph(device, authority, probeAtEnd, "export_");
                    FakeReplayGPUContext gpu;
                    auto cache = auxiliaryMainPlan(graph, auxiliaryProbePolicy(authority));
                    recordAuxiliaryMainChildren(cache, gpu);
                    std::string error;
                    const auto views = cache.retainedCaptureUnitTemplatesForParentComposition(graph, &error);
                    ASSERT_TRUE(views) << error;
                    std::vector<std::string> expected, observed;
                    for (const auto& name : graph.getExecutionOrder())
                        if (!graph.isRetainedParticipantAuxiliaryNode(name) &&
                            !graph.getNode(name)->stage->isManualGraphBoundary())
                            expected.push_back(name);
                    for (const auto& unit : *views)
                    {
                        EXPECT_EQ(unit.stream, cache.capture_stream);
                        EXPECT_FALSE(unit.capture->hasExecutable());
                        observed.insert(observed.end(), unit.stage_names.begin(), unit.stage_names.end());
                    }
                    EXPECT_EQ(observed, expected);
                    EXPECT_FALSE(cache.retained_parent_capture);
                    EXPECT_FALSE(cache.original_storage_attachment);

                    // These graph-only children cannot be exported as complete
                    // executable replays, even if a caller changes cache flags.
                    cache.initialized = true;
                    cache.needs_capture = false;
                    EXPECT_FALSE(cache.retainedCaptureUnitTemplates(graph, &error));
                    EXPECT_FALSE(error.empty());
                }
}

/** @test Projected coverage never excuses omitted, duplicated, reordered or auxiliary main-child work. */
TEST(Test__GraphSegmentCache, RetainedParticipantAuxiliaryChildExportRejectsBrokenCoverage)
{
    for (const auto device : {DeviceId::cuda(1), DeviceId::rocm(1)})
        for (const bool authority : {false, true})
            for (int defect = 0; defect < 7; ++defect)
            {
                SCOPED_TRACE(::testing::Message() << device.toString() << " authority=" << authority
                    << " defect=" << defect);
                auto graph = auxiliaryProbeGraph(device, authority, false, "bad_export_");
                FakeReplayGPUContext gpu;
                auto cache = auxiliaryMainPlan(graph, auxiliaryProbePolicy(authority));
                recordAuxiliaryMainChildren(cache, gpu);
                auto last = std::find_if(cache.segments.rbegin(), cache.segments.rend(),
                    [](const auto& unit) { return unit.capturable; });
                ASSERT_NE(last, cache.segments.rend());
                ASSERT_GT(last->stage_names.size(), 1u);
                switch (defect)
                {
                case 0: last->stage_names.pop_back(); break;
                case 1: last->stage_names.push_back(last->stage_names.back()); break;
                case 2: std::swap(last->stage_names[0], last->stage_names[1]); break;
                case 3: last->stage_names.push_back("bad_export_aux_result"); break;
                case 4: graph.getNode("bad_export_aux_result")->device = DeviceId::cpu(); break;
                case 5: graph.addDependency("bad_export_aux_begin", "bad_export_main_read"); break;
                case 6: last->capture->instantiate(); break;
                }
                std::string error;
                EXPECT_FALSE(cache.retainedCaptureUnitTemplatesForParentComposition(graph, &error));
                EXPECT_FALSE(error.empty());
            }
}

/**
 * @test The actual heterogeneous composer accepts the same validated main projection.
 *
 * GPU startup is explicitly excluded. Reaching TransferEngine's native-backend
 * requirement proves the real composer's coverage checks accepted the children;
 * this fixture deliberately cannot certify native graph lowering or attachment.
 * The source-admission/native-parent tests own those separate obligations.
 */
TEST(Test__GraphSegmentCache, RetainedParticipantAuxiliaryParentCompositionCoverageReplay)
{
    ScopedEnvVar exclude_native("LLAMINAR_FORCE_CPU_ONLY_STARTUP", "1");
    for (const auto device : {DeviceId::cuda(1), DeviceId::rocm(1)})
        for (const bool authority : {false, true})
            for (const bool probeAtEnd : {false, true})
                for (int replay = 0; replay < 20; ++replay)
                {
                    SCOPED_TRACE(::testing::Message() << device.toString() << " authority=" << authority
                        << " probe_at_end=" << probeAtEnd << " replay=" << replay);
                    auto graph = auxiliaryProbeGraph(device, authority, probeAtEnd, "compose_");
                    const auto plan = makeMoEOverlayRetainedParentPlan(graph);
                    ASSERT_TRUE(plan);
                    FakeReplayGPUContext gpu;
                    auto cache = auxiliaryMainPlan(graph, plan->replay_policy);
                    recordAuxiliaryMainChildren(cache, gpu);
                    std::string error;
                    const auto views = cache.retainedCaptureUnitTemplatesForParentComposition(graph, &error);
                    ASSERT_TRUE(views) << error;
                    FakeReplayGraphCapture parent(cache.capture_stream, false, 0);
                    try
                    {
                        (void)plan->composer(parent, graph, *views);
                        FAIL() << "Device-free composition must stop at the excluded native backend";
                    }
                    catch (const std::invalid_argument& ex)
                    {
                        EXPECT_EQ(std::string(ex.what()),
                            "TransferEngine mapped timeline transaction requires native 64-bit timeline support for " + device.toString());
                    }
                    EXPECT_EQ(parent.nodeCount(), 0u);
                    EXPECT_EQ(parent.instantiate_calls_, 0);
                    EXPECT_EQ(parent.launch_calls_, 0);
                }
}

/** @test Auxiliary exclusion never excuses a corrupt main sequence or mutated original body at composition. */
TEST(Test__GraphSegmentCache, RetainedParticipantAuxiliaryParentCompositionRejectsBrokenCoverage)
{
    ScopedEnvVar exclude_native("LLAMINAR_FORCE_CPU_ONLY_STARTUP", "1");
    for (const auto device : {DeviceId::cuda(1), DeviceId::rocm(1)})
        for (const bool authority : {false, true})
            for (int defect = 0; defect < 6; ++defect)
            {
                auto graph = auxiliaryProbeGraph(device, authority, false, "bad_compose_");
                const auto plan = makeMoEOverlayRetainedParentPlan(graph);
                ASSERT_TRUE(plan);
                FakeReplayGPUContext gpu;
                auto cache = auxiliaryMainPlan(graph, plan->replay_policy);
                recordAuxiliaryMainChildren(cache, gpu);
                auto last = std::find_if(cache.segments.rbegin(), cache.segments.rend(),
                    [](const auto& unit) { return unit.capturable; });
                ASSERT_NE(last, cache.segments.rend());
                switch (defect)
                {
                case 0: last->stage_names.pop_back(); break;
                case 1: last->stage_names.push_back(last->stage_names.back()); break;
                case 2: std::swap(last->stage_names[0], last->stage_names[1]); break;
                case 3: last->stage_names.push_back("bad_compose_aux_result"); break;
                case 4: graph.getNode("bad_compose_aux_result")->device = DeviceId::cpu(); break;
                case 5: graph.addDependency("bad_compose_aux_begin", "bad_compose_main_read"); break;
                }
                // Deliberately call the actual composer directly with corrupt
                // views so its own check cannot borrow the exporter rejection.
                std::vector<DeviceGraphExecutor::GraphSegmentCache::RetainedCaptureUnitTemplateView> views;
                for (const auto& unit : cache.segments)
                    if (unit.capturable)
                        views.push_back({.capture = unit.capture.get(), .stream = cache.capture_stream,
                            .stage_names = unit.stage_names, .captured_node_count = unit.capture->nodeCount()});
                FakeReplayGraphCapture parent(cache.capture_stream, false, 0);
                try
                {
                    (void)plan->composer(parent, graph, views);
                    FAIL() << "Corrupt composition unexpectedly accepted defect " << defect;
                }
                catch (const std::invalid_argument& ex)
                {
                    EXPECT_EQ(std::string(ex.what()).find("TransferEngine"), std::string::npos) << defect;
                }
                EXPECT_EQ(parent.nodeCount(), 0u);
                EXPECT_EQ(parent.instantiate_calls_, 0);
                EXPECT_EQ(parent.launch_calls_, 0);
            }
}

namespace
{
/**
 * @brief Record only the actual full declarative metadata used by the exporter.
 * @param graph Original graph whose pointer/variant roster is frozen.
 * @param cache Main controller partition receiving that full roster.
 * @note No source permit, native parent or body attachment is created here.
 */
void retainAuxiliaryTemplateStageIdentity(
    const ComputeGraph& graph, DeviceGraphExecutor::GraphSegmentCache& cache)
{
    auto& plan = cache.retained_composed_parent_replay;
    plan.graph = &graph;
    plan.topology_generation = graph.topologyGeneration();
    plan.child_unit_count = cache.segments.size();
    for (const auto& name : graph.getExecutionOrder())
    {
        const auto* stage = graph.getNode(name)->stage.get();
        plan.stages.push_back(const_cast<IComputeStage*>(stage));
        plan.stage_variant_signatures.push_back(stage->graphCaptureVariantSignature());
    }
}
}

/** @test Metadata accepts the main projection while retaining every auxiliary stage identity, without authorizing execution. */
TEST(Test__GraphSegmentCache, RetainedParticipantAuxiliaryExecutableTemplateMetadataReplay)
{
    for (const auto device : {DeviceId::cuda(1), DeviceId::rocm(1)})
        for (int replay = 0; replay < 20; ++replay)
        {
            auto graph = auxiliaryReaderGraph(device);
            FakeReplayGPUContext gpu;
            auto cache = auxiliaryMainPlan(graph);
            retainAuxiliaryTemplateStageIdentity(graph, cache);
            std::string error;
            EXPECT_TRUE(validateRetainedParentTemplateStageCoverage(
                graph, cache.segments, cache.retained_composed_parent_replay, &error)) << error;
            EXPECT_EQ(cache.retained_composed_parent_replay.stages.size(), graph.size());
            EXPECT_FALSE(cache.original_storage_attachment);
            EXPECT_FALSE(cache.participant_auxiliary_preparation);
            EXPECT_FALSE(cache.retained_parent_capture);

            // Exact main metadata cannot turn a body-less cache into an
            // executable. Native/model tests own the real attachment positive.
            recordAuxiliaryMainChildren(cache, gpu);
            cache.initialized = true;
            cache.needs_capture = false;
            EXPECT_FALSE(cache.deviceLoopGraphTemplate(graph, &error));
            EXPECT_EQ(error, "composed parent has incomplete or stale native capture identity");

            // An observed native-parent stub is still insufficient: no test
            // fabricates a source/preparation/attachment permit to pass export.
            cache.retained_parent_capture = std::make_unique<FakeReplayGraphCapture>(cache.capture_stream, true, 5);
            EXPECT_FALSE(cache.deviceLoopGraphTemplate(graph, &error));
            EXPECT_EQ(error, "composed parent has no original attached participant auxiliary body");
        }
}

/** @test Projection rejects missing, duplicate, reordered and foreign work; stale auxiliary/full-stage identities stay fatal. */
TEST(Test__GraphSegmentCache, RetainedParticipantAuxiliaryExecutableTemplateRejectsBrokenMetadata)
{
    for (const auto device : {DeviceId::cuda(1), DeviceId::rocm(1)})
        for (int defect = 0; defect < 10; ++defect)
        {
            SCOPED_TRACE(::testing::Message() << device.toString() << " defect=" << defect);
            auto graph = auxiliaryReaderGraph(device);
            auto cache = auxiliaryMainPlan(graph);
            retainAuxiliaryTemplateStageIdentity(graph, cache);
            auto& last = cache.segments.back();
            auto& plan = cache.retained_composed_parent_replay;
            const auto& order = graph.getExecutionOrder();
            const auto aux = static_cast<size_t>(std::find(order.begin(), order.end(), "aux_result") - order.begin());
            switch (defect)
            {
            case 0: last.stage_names.pop_back(); break;
            case 1: last.stage_names.push_back(last.stage_names.back()); break;
            case 2: std::swap(last.stage_names[0], last.stage_names[1]); break;
            case 3: last.stage_names.push_back("aux_result"); break;
            case 4: graph.getNode("aux_result")->device = DeviceId::cpu(); break;
            case 5: graph.addDependency("aux_begin", "main_read"); break;
            case 6: plan.stages[aux] = plan.stages.front(); break;
            case 7: ++plan.stage_variant_signatures[aux]; break;
            case 8: plan.stages.pop_back(); break;
            case 9: plan.stage_variant_signatures.pop_back(); break;
            }
            std::string error;
            EXPECT_FALSE(validateRetainedParentTemplateStageCoverage(graph, cache.segments, plan, &error));
            EXPECT_FALSE(error.empty());
        }
}

/** @test No-auxiliary graphs retain strict full coverage and stored stage identity. */
TEST(Test__GraphSegmentCache, RetainedParticipantAuxiliaryExecutableTemplatePreservesPlainCoverage)
{
    ComputeGraph graph;
    addFakeSegmentStage(graph, "first", true);
    addFakeSegmentStage(graph, "second", true);
    graph.addDependency("second", "first");
    auto cache = auxiliaryMainPlan(graph);
    retainAuxiliaryTemplateStageIdentity(graph, cache);
    std::string error;
    EXPECT_TRUE(validateRetainedParentTemplateStageCoverage(
        graph, cache.segments, cache.retained_composed_parent_replay, &error)) << error;
    cache.segments.back().stage_names.pop_back();
    EXPECT_FALSE(validateRetainedParentTemplateStageCoverage(
        graph, cache.segments, cache.retained_composed_parent_replay, &error));
    EXPECT_EQ(error, "composed parent does not cover its complete source graph");
}

namespace
{
/** @return Canonically resolved mixed-GPU world-one metadata, never native/source authority. */
std::shared_ptr<const MoEExpertOverlayExecutionPlan> nativeAuxiliaryOriginalPlan()
{
    auto declaration = std::make_shared<MoERoutedExpertPlacementPlan>(llaminar2::test::cpuJointSourceTestPlan());
    declaration->domains[0].participants = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    auto& follower = declaration->domains[1];
    follower.name = "rocm_followers";
    follower.scope = ExecutionDomainScope::RANK_LOCAL;
    follower.backend = CollectiveBackendType::RCCL;
    follower.owner_rank = 0;
    follower.participants = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
    follower.world_ranks.clear();
    declaration->routed_tiers[1].domain = follower.name;
    declaration->routed_tiers[1].fallback = false;
    const auto resolved = resolveMoEExpertOverlayRuntimePlan(declaration, {.current_world_rank = 0});
    return std::make_shared<const MoEExpertOverlayExecutionPlan>(buildMoEExpertOverlayExecutionPlan(*resolved, 1));
}
}

/** @test The actual native controller/composer accepts only the original mixed-GPU auxiliary shape.
 * Graph-only children are cold compiler artifacts; none is instantiated or replayed here.
 */
TEST(Test__GraphSegmentCache, NativeAuxiliaryParentOriginalPlanReplay)
{
    ScopedEnvVar exclude_native("LLAMINAR_FORCE_CPU_ONLY_STARTUP", "1");
    const auto original = nativeAuxiliaryOriginalPlan();
    ASSERT_NO_THROW(requireRankLocalMixedGPUAuxiliaryExecution(*original));
    for (int replay = 0; replay < 20; ++replay)
    for (const auto device : {DeviceId::cuda(1), DeviceId::rocm(1)})
    for (const bool passive : {false, true})
    {
        auto graph = auxiliaryReaderGraph(device, 71, false, original);
        if (passive)
        {
            graph.addNode("passive_reducer", std::make_unique<FakeSegmentStage>(device, true, false,
                ComputeStageType::COPY, false, false, false, nullptr, true), device);
            graph.addDependency("passive_reducer", "trunk");
            graph.addDependency("main_read", "passive_reducer");
        }
        graph.setNativeCaptureEnvelope(GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction);
        graph.setNativeTransactionCaptureContract({.identity = "original_complete_native_transaction", .terminal_node = "output"});
        EXPECT_NO_THROW(graph.requireRetainedParticipantNativeAuxiliaryParent());
        DeviceGraphExecutor::DecodeCapturePolicy policy;
        policy.allow_cached_graph_replay = true;
        policy.graph_replay_plan_policy = DeviceGraphExecutor::GraphReplayPlanPolicy::RequireRetainedParentComposition;
        policy.retained_parent_composer = [](auto&, const auto&, auto) { return false; };
        ASSERT_TRUE(DeviceGraphCaptureController::constrainReplayPolicyToNativeEnvelope(
            graph.nativeCaptureEnvelope(), policy));
        EXPECT_EQ(policy.graph_replay_plan_policy, DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph);
        EXPECT_FALSE(policy.retained_parent_composer);
        auto plan = makeMoEOverlayRetainedParentPlan(graph);
        ASSERT_TRUE(plan);
        policy.graph_replay_plan_policy = plan->replay_policy;
        policy.retained_parent_composer = std::move(plan->composer);
        policy.defer_final_sync = true;
        EXPECT_EQ(plan->replay_policy, DeviceGraphExecutor::GraphReplayPlanPolicy::RequireRetainedParentComposition);
        FakeReplayGPUContext gpu;
        auto cache = auxiliaryMainPlan(graph, policy.graph_replay_plan_policy);
        ASSERT_EQ(cache.segments.size(), 2u);
        for (const auto& unit : cache.segments)
        {
            EXPECT_TRUE(unit.capturable);
            EXPECT_EQ(unit.capture.get(), nullptr);
        }
        EXPECT_EQ(cache.segments.front().capture_wave_identity, "participant_auxiliary_probe");
        EXPECT_EQ(cache.segments.back().capture_wave_identity, "original_complete_native_transaction");
        recordAuxiliaryMainChildren(cache, gpu);
        std::string error;
        const auto units = cache.retainedCaptureUnitTemplatesForParentComposition(graph, &error);
        ASSERT_TRUE(units) << error;
        FakeReplayGraphCapture parent(cache.capture_stream, false, 0);
        // Actual coverage/lowering reaches the deliberately excluded native backend.
        try { (void)policy.retained_parent_composer(parent, graph, *units); FAIL(); }
        catch (const std::invalid_argument& ex)
        {
            EXPECT_EQ(std::string(ex.what()),
                "TransferEngine mapped timeline transaction requires native 64-bit timeline support for " + device.toString());
        }
        EXPECT_EQ(parent.instantiate_calls_, 0);
        EXPECT_EQ(parent.launch_calls_, 0);
        EXPECT_THROW((void)auxiliaryMainPlan(graph,
            DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph), std::invalid_argument);
        EXPECT_THROW((void)auxiliaryMainPlan(graph,
            DeviceGraphExecutor::GraphReplayPlanPolicy::RequireRetainedParentWithConcurrentTicketService), std::logic_error);
    }
}

/** @test Missing, foreign, CPU, distributed, homogeneous and manual shapes cannot gain native-parent eligibility. */
TEST(Test__GraphSegmentCache, NativeAuxiliaryParentRejectsForeignAuthorityReplay)
{
    const auto original = nativeAuxiliaryOriginalPlan();
    const std::vector<std::function<void(MoEExpertOverlayExecutionPlan&)>> mutations{
        [](auto& p) { p.world_size = 2; },
        [](auto& p) { p.continuation_root_rank = 1; },
        [](auto& p) { p.rank_plans.clear(); },
        [](auto& p) { p.current_rank.execution_kind = OverlayRankExecutionKind::ContinuationPeer; },
        [](auto& p) { p.domains[1].scope = ExecutionDomainScope::GLOBAL; },
        [](auto& p) { p.domains[1].participants[0].owned_by_current_rank = false; },
        [](auto& p) { p.domains[1].participants[0].locally_addressable = false; },
        [](auto& p) { p.domains[1].participants[0].world_rank = 1; },
        [](auto& p) { p.domains[1].participants[0].world_rank_known = false; },
        [](auto& p) { p.domains[1].participants[0].local_device = DeviceId::cpu(); },
        [](auto& p) { p.domains[1].participants[0].address.device_type = DeviceType::CUDA; },
        [](auto& p) { p.current_rank.cpu_fallback_expert_domains.push_back("cpu"); p.rank_plans[0] = p.current_rank; },
        [](auto& p) { p.current_rank.local_devices.push_back(DeviceId::rocm(7)); p.rank_plans[0] = p.current_rank; },
        [](auto& p) { for (auto& d : p.domains) for (auto& a : d.participants) a.address.device_type = DeviceType::CUDA; }
    };
    for (int replay = 0; replay < 20; ++replay)
    {
        for (const auto& mutate : mutations)
        {
            auto changed = std::make_shared<MoEExpertOverlayExecutionPlan>(*original);
            mutate(*changed);
            EXPECT_THROW(requireRankLocalMixedGPUAuxiliaryExecution(*changed), std::invalid_argument);
            auto graph = auxiliaryReaderGraph(DeviceId::rocm(1), 71, false, changed);
            graph.setNativeCaptureEnvelope(GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction);
            EXPECT_THROW((void)makeMoEOverlayRetainedParentPlan(graph), std::invalid_argument);
        }
        auto absent = auxiliaryReaderGraph(DeviceId::rocm(1));
        absent.setNativeCaptureEnvelope(GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction);
        EXPECT_THROW((void)makeMoEOverlayRetainedParentPlan(absent), std::invalid_argument);
        auto ordinary = auxiliaryReaderGraph(DeviceId::rocm(1), 71, false, original);
        EXPECT_THROW(ordinary.requireRetainedParticipantNativeAuxiliaryParent(), std::invalid_argument);
        auto foreign = auxiliaryReaderGraph(DeviceId::rocm(7), 71, false, original);
        foreign.setNativeCaptureEnvelope(GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction);
        EXPECT_THROW(foreign.requireRetainedParticipantNativeAuxiliaryParent(), std::invalid_argument);
        auto marked = auxiliaryReaderGraph(DeviceId::rocm(1), 71, false, original);
        marked.setNativeCaptureEnvelope(GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction);
        marked.setHeterogeneousTicketUnitContract("output", {.identity="foreign_ticket", .disposition=GraphHeterogeneousTicketUnitDisposition::TransactionTerminal});
        EXPECT_THROW(marked.requireRetainedParticipantNativeAuxiliaryParent(), std::invalid_argument);
    }
}
