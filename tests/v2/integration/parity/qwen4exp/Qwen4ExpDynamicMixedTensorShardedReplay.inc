/**
 * @file Qwen4ExpDynamicMixedTensorShardedReplay.inc
 * @brief Production Dynamic QFN swaps between complete two-device GPU and two-socket CPU shards.
 *
 * The model-owned source, native graph, economy controller and residency
 * authority are the serving implementations. The fixture supplies the
 * canonical bounded quota and changes neither route packets nor bank epochs.
 */
namespace
{
/**
 * @brief Select the best observed rank-local TP2 pair for the requested backend.
 * @param inventory Immutable two-rank hardware/PCI and directed peer observations.
 * @param backend Requested native CUDA backend or the qualified MI50 ROCm geometry.
 * @return Original owning rank and pair, preferring mutual P2P and nearest common switch.
 */
std::optional<ObservedNativeReplayOwner> bestOriginalMixedReplayPair(const ClusterInventory& inventory, DeviceType backend)
{
    std::optional<ObservedNativeReplayOwner> selected;
    ObservedPeerGroupQuality best;
    for (const auto& rank : inventory.ranks)
    {
        std::vector<DeviceId> eligible;
        for (const auto& gpu : rank.gpus)
            if (gpu.type == backend && rankOwnsGPUDevice(rank, gpu) &&
                (backend == DeviceType::CUDA ||
                 std::string_view(gpu.native_isa).substr(0, gpu.native_isa.find(':')) == "gfx906"))
                eligible.push_back(backend == DeviceType::CUDA
                    ? DeviceId::cuda(gpu.local_device_id) : DeviceId::rocm(gpu.local_device_id));

        if (eligible.size() < 2)
            continue;
        const ObservedPeerGrouping peers(rank);
        const std::array<std::size_t, 1> sizes{2};
        auto groups = peers.select(eligible, sizes);
        const auto quality = peers.quality(groups.front());
        if (!selected || quality > best)
        {
            selected = ObservedNativeReplayOwner{rank.rank, std::move(groups.front())};
            best = quality;
        }
    }
    return selected;
}

/**
 * @brief Bind the canonical two-socket QFN source to the observed native GPU pair.
 * @param inventory Original MPI hardware inventory in world-rank order.
 * @param gpu_owner Original rank and ordered devices selected from that inventory.
 * @return Public paired-source configuration with both complete shard tiers.
 */
OrchestrationConfig mixedTensorShardedFixed3Config(
    const ClusterInventory& inventory, const ObservedNativeReplayOwner& gpu_owner)
{
    auto seed = qwen4ExpCPUJointWorld2SourceAdmissionCases().front();
    auto placement = std::make_shared<MoERoutedExpertPlacementPlan>(*seed.topology.expert_overlay_plan);
    auto& gpu = placement->domains.front();
    auto& cpu = placement->domains.back();
    gpu.scope = ExecutionDomainScope::RANK_LOCAL;
    gpu.backend = gpu_owner.devices.front().is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
    gpu.owner_rank = gpu_owner.discovery_owner;
    gpu.world_ranks.clear();
    gpu.participants.clear();
    seed.topology.participants.clear();
    for (const auto device : gpu_owner.devices)
    {
        const int numa = inventory.ranks.at(gpu.owner_rank).cpu.numa_node;
        const auto address = device.is_cuda() ? GlobalDeviceAddress::cuda(device.ordinal, numa)
                                              : GlobalDeviceAddress::rocm(device.ordinal, numa);
        gpu.participants.push_back(address);
        seed.topology.participants.push_back({address, gpu.owner_rank});
    }
    gpu.routed_compute_policy = RoutedExpertComputePolicy::TensorSharded;
    for (std::size_t index = 0; index < cpu.world_ranks.size(); ++index)
    {
        const int rank = cpu.world_ranks[index];
        const auto address = GlobalDeviceAddress::cpu(inventory.ranks.at(rank).cpu.numa_node);
        cpu.participants[index] = address;
        seed.topology.participants.push_back({address, rank});
    }
    placement->continuation_domain_spec.setDensePolicy(DenseParallelPolicy::TensorParallel);
    placement->dense_domains = {gpu.toExecutionDomainDefinition()};
    seed.topology.expert_overlay_plan = std::move(placement);
    seed.expert_overlay->movement = ModelParityExpertMovement::Dynamic;
    Qwen4ExpMTPAdmissionCase cell{
        .main = std::move(seed), .policy = ModelParityMTP::Depth3, .retained_depth = 3,
        .terminal_head_policy = MTPTerminalHeadPolicy::VocabularySharded};
    auto config = cell.config(Qwen4ExpMTPAdmissionExecution::Active,
        Qwen4ExpMTPAdmissionPurpose::ProductionServing);
    // Reuse the published short-window economic-movement parity policy. It
    // changes cadence through ordinary config while retaining measured prices,
    // real routed traffic and the production's accept/reject decision.
    config.moe_rebalance = cell.main.dynamic_rebalance;
    config.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
    config.routed_expert_compute_policy = RoutedExpertComputePolicy::TensorSharded;
    config.prefix_cache.enabled = false;
    return config;
}

/**
 * @brief Verify every current expert is resident in one complete cohort with its real two members.
 * @param snapshot Immutable original residency publication.
 * @param gpu_owner Original rank-local GPU cohort selected from MPI discovery.
 */
void expectCompleteMixedTensorShards(const MoEOverlayResidencySnapshot& snapshot,
    const ObservedNativeReplayOwner& gpu_owner)
{
    ASSERT_TRUE(snapshot.valid());
    ASSERT_EQ(snapshot.ownershipKind(), MoEOverlayResidencyOwnershipKind::JointFragments);
    ASSERT_EQ(snapshot.expertCount(), 512);
    ASSERT_GE(snapshot.layerCount(), 48);
    const auto& owners = snapshot.placementOwnerMap();
    for (int l = snapshot.firstModelLayer(); l < snapshot.firstModelLayer() + snapshot.layerCount(); ++l)
    {
        SCOPED_TRACE(l);
        std::array<int, 2> population{};
        for (int expert = 0; expert < snapshot.expertCount(); ++expert)
        {
            const auto* owner = owners.placementFor(l, expert);
            ASSERT_NE(owner, nullptr);
            ASSERT_TRUE(owner->resident);
            ASSERT_TRUE(owner->shard_cohort);
            ASSERT_GE(owner->tier_idx, 0);
            ASSERT_LT(owner->tier_idx, 2);
            ++population[owner->tier_idx];
            const auto members = owner->shard_cohort->participants();
            ASSERT_EQ(members.size(), 2u);
            if (owner->tier_idx == 0)
            {
                for (std::size_t member = 0; member < members.size(); ++member)
                {
                    EXPECT_EQ(members[member].world_rank, gpu_owner.discovery_owner);
                    EXPECT_EQ(members[member].address.toLocalDeviceId(), gpu_owner.devices[member]);
                }
            }
            else
            {
                EXPECT_TRUE(members.front().address.isCPU());
                EXPECT_TRUE(members.back().address.isCPU());
                EXPECT_NE(members.front().world_rank, members.back().world_rank);
                EXPECT_NE(members.front().address.numa_node, members.back().address.numa_node);
            }
        }
        EXPECT_EQ(population, (std::array{64, 448}));
    }
}

/**
 * @brief Exercise requests until the real maintenance owner accepts both directions, then prove post-swap replay.
 * @param runner Original coordinated inference root.
 * @param gpu_owner Original admitted rank-local GPU cohort.
 */
void proveMixedTensorShardedMovement(OrchestrationRunner& runner, const ObservedNativeReplayOwner& gpu_owner)
{
    const auto initial = runner.expertOverlayResidencySnapshotForDiagnostics();
    ASSERT_TRUE(initial);
    ASSERT_NO_FATAL_FAILURE(expectCompleteMixedTensorShards(*initial, gpu_owner));
    const auto topology = runner.moeOptimizationMovementTopology();
    ASSERT_TRUE(topology.valid());
    EXPECT_EQ(topology.authority, MoEOptimizationAuthority::Host);
    EXPECT_EQ(topology.axes, MoEOptimizationMovementAxes::TierResidency);

    const SamplingParams sampling{.temperature = 0.6f, .top_k = 32, .top_p = 0.95f, .seed = 1729u};
    std::vector<int32_t> prompt(64);
    for (std::size_t i = 0; i < prompt.size(); ++i)
        prompt[i] = static_cast<int32_t>(100 + i);
    // Use the production benchmark prompt that previously exposed the first
    // resident-tier discrepancy. Authenticated tokenization stays with QFN.
    const auto replay_prompt = runner.tokenizer()->encode(
        resolveBenchmarkPrompt(OrchestrationConfig{}).text, false, false);
    ASSERT_GT(replay_prompt.size(), 512u);
    const SamplingParams greedy{.temperature = 0.0f, .top_k = 1, .top_p = 1.0f, .seed = 42u};
    const auto active_policy = runner.mtpRequestPolicy();
    ASSERT_TRUE(active_policy.enabled);
    auto ordinary_policy = active_policy;
    ordinary_policy.enabled = false;
    constexpr int replay_tokens = 64;
    std::vector<int32_t> before_movement;
    const auto initial_pin = runner.withMoEOverlayInferencePlacementScope([&] {
        runner.clearCache();
        ASSERT_TRUE(runner.configureMTPRequestPolicy(ordinary_policy)) << runner.lastError();
        const auto result = runner.generate(replay_prompt, replay_tokens, greedy);
        ASSERT_TRUE(result.success()) << result.error;
        ASSERT_EQ(result.tokenCount(), static_cast<std::size_t>(replay_tokens));
        before_movement = result.tokens;
        runner.clearCache();
        ASSERT_TRUE(runner.configureMTPRequestPolicy(active_policy)) << runner.lastError();
    });
    ASSERT_EQ(initial_pin.placement_epoch, initial->epoch);
    ASSERT_EQ(before_movement.size(), static_cast<std::size_t>(replay_tokens));
    if (gpu_owner.devices.front().is_cuda())
    {
        const auto publications = PerfStatsCollector::snapshot({"moe.router_selected_descriptors_enqueued"});
        ASSERT_TRUE(std::any_of(publications.begin(), publications.end(), [](const auto &record) {
            return record.name == "router_selected_descriptors_enqueued" && record.count > 0;
        })) << "CUDA ordinary decode must use the router's captured descriptor publication";
    }
    // Close real routed-demand windows without injecting a histogram or
    // commanding a swap. Vary the leading token between warm-up requests.
    for (int request = 0; request < 16; ++request)
    {
        if (runner.moeRuntimeMovementEpoch() > initial->epoch)
            break;
        runner.clearCache();
        prompt.front() = 100 + request;
        const auto output = runner.generate(prompt, 16, sampling);
        ASSERT_TRUE(output.success()) << output.error;
        ASSERT_GT(output.tokenCount(), 4u);
        ASSERT_FALSE(runner.moeOptimizationStatus().failed()) << runner.moeOptimizationStatus().diagnostic;
    }
    const auto published = runner.expertOverlayResidencySnapshotForDiagnostics();
    ASSERT_TRUE(published);
    ASSERT_GT(published->epoch, initial->epoch) << runner.moeOptimizationStatus().diagnostic;
    ASSERT_NO_FATAL_FAILURE(expectCompleteMixedTensorShards(*published, gpu_owner));

    const auto ledger = runner.moeOptimizationMovementLedger();
    ASSERT_TRUE(ledger.complete());
    ASSERT_FALSE(ledger.edges.empty());
    ASSERT_TRUE(std::any_of(ledger.host_admissions.begin(), ledger.host_admissions.end(),
        [&](const auto& receipt) { return receipt.valid() && receipt.candidate_epoch == published->epoch; }));
    bool promoted = false;
    bool demoted = false;
    std::set<std::pair<int, int>> terminal_moves;
    for (auto it = ledger.edges.rbegin(); it != ledger.edges.rend(); ++it)
    {
        const auto& edge = *it;
        if (edge.candidate_epoch > published->epoch)
            continue;
        ASSERT_TRUE(edge.valid());
        ASSERT_TRUE(edge.source.isCohort());
        ASSERT_TRUE(edge.destination.isCohort());
        EXPECT_EQ(edge.source.cohort().original->participants().size(), 2u);
        EXPECT_EQ(edge.destination.cohort().original->participants().size(), 2u);
        EXPECT_EQ(edge.axis, MoEOptimizationMovementAxis::TierResidency);
        EXPECT_NE(edge.source.cohort().tier, edge.destination.cohort().tier);
        promoted |= edge.direction == MoEOptimizationMovementDirection::Promotion;
        demoted |= edge.direction == MoEOptimizationMovementDirection::Demotion;
        if (terminal_moves.emplace(edge.layer, edge.expert).second)
        {
            const auto* owner = published->placementOwnerMap().placementFor(edge.layer, edge.expert);
            ASSERT_NE(owner, nullptr);
            EXPECT_EQ(MoEOptimizationMovementEndpoint::fromOwner(*owner), edge.destination);
        }
    }
    EXPECT_TRUE(promoted);
    EXPECT_TRUE(demoted);
    EXPECT_GT(runner.moeOptimizationStatus().completed_movement.physical_bytes, 0u);

    std::vector<int32_t> expected;
    const auto pinned = runner.withMoEOverlayInferencePlacementScope([&] {
        for (int replay = 0; replay < 2; ++replay)
        {
            runner.clearCache();
            const auto before = qfnGroupedVerifierRuns("grouped_decode_equivalent_stochastic_verifier_runs");
            const auto result = runner.generate(prompt, 16, sampling);
            ASSERT_TRUE(result.success()) << result.error;
            ASSERT_GT(result.tokenCount(), 4u);
            EXPECT_GT(qfnGroupedVerifierRuns("grouped_decode_equivalent_stochastic_verifier_runs"), before);
            if (replay == 0)
                expected = result.tokens;
            else
                EXPECT_EQ(result.tokens, expected);
        }
        ASSERT_NO_FATAL_FAILURE(expectQFNGroupedGreedyMatchesOrdinary(runner, prompt));
        for (const auto& policy : {ordinary_policy, active_policy})
        {
            runner.clearCache();
            ASSERT_TRUE(runner.configureMTPRequestPolicy(policy)) << runner.lastError();
            const auto result = runner.generate(replay_prompt, replay_tokens, greedy);
            ASSERT_TRUE(result.success()) << result.error;
            EXPECT_EQ(result.tokens, before_movement) << "grouped MTP=" << policy.enabled;
        }
    });
    EXPECT_GE(pinned.placement_epoch, published->epoch);
    EXPECT_GT(pinned.next_sequence_id, pinned.first_sequence_id);
}
}

namespace
{
/** @brief Compare generation across a published swap on the original native cohort. */
void runMixedTensorShardedMovementReplay(DeviceType backend)
{
    const auto mpi = MPIContextFactory::global();
    ASSERT_TRUE(mpi);
    const auto inventory = mpi->clusterInventory();
    ASSERT_TRUE(inventory);
    ASSERT_EQ(inventory->ranks.size(), 2u);
    ASSERT_EQ(inventory->ranks[0].node_id, inventory->ranks[1].node_id);
    std::array<int, 2> packages{};
    for (std::size_t rank = 0; rank < packages.size(); ++rank)
    {
        const auto& original = inventory->ranks[rank];
        const auto socket = std::find_if(original.cpu_socket_info.begin(), original.cpu_socket_info.end(),
            [&](const auto& endpoint) { return endpoint.numa_node == original.cpu.numa_node; });
        ASSERT_NE(socket, original.cpu_socket_info.end());
        ASSERT_GE(socket->socket_id, 0);
        packages[rank] = socket->socket_id;
    }
    ASSERT_NE(packages.front(), packages.back());
    const auto gpu_owner = bestOriginalMixedReplayPair(*inventory, backend);
    ASSERT_TRUE(gpu_owner);
    const QFNDynamicReplayScratch scratch;
    ASSERT_FALSE(scratch.path().empty());
    auto config = mixedTensorShardedFixed3Config(*inventory, *gpu_owner);
    config.prefix_cache.disk_dir = (scratch.path() / "prefix-cache").string();
    publishOrchestrationStartupPolicy(config);
    initCPUBackend(inventory->ranks.at(mpi->rank()).cpu.numa_node);
    Qwen4ExpSourceAdmissionLifetime lifetime(config, config.prefix_cache.disk_dir, mpi);
    auto *runner = &lifetime.runner();
    runner->disableSnapshotCapture();
    ASSERT_TRUE(runner->initialize()) << runner->lastError();
    auto* concrete = dynamic_cast<OrchestrationRunner*>(runner);
    ASSERT_NE(concrete, nullptr);
    runner->setMPICoordinatedMode(true);
    EXPECT_EQ(runner->coordinatedRootRank(), gpu_owner->discovery_owner);
    // Both ranks must declare model reuse before SHUTDOWN collectively
    // drains the live transport and restores the original prepared banks.
    const auto &model = lifetime.reuseContract();
    ASSERT_TRUE(model);
    ASSERT_TRUE(model->reuse_authority);
    if (mpi->rank() == runner->coordinatedRootRank())
    {
        // Close follower admission even when a witness raises an exception.
        const auto stop_followers = [](IOrchestrationRunner* source) { source->shutdownMPIWorkers(); };
        const std::unique_ptr<IOrchestrationRunner, decltype(stop_followers)> admission(runner, stop_followers);
        // Enter the same collective preparation as HTTP serving. Initialize()
        // builds graphs and starts calibration, but generation and movement
        // assertions require the final sealed service and migration profile.
        if (runner->prepareForInference())
            proveMixedTensorShardedMovement(*concrete, *gpu_owner);
        else
            ADD_FAILURE() << runner->lastError();
    }
    else
        runner->runMPIWorkerLoop();
    // The root's guard ends before model-owner export. Followers also exit
    // their worker loop before the model lifetime resets any native owner.
    runner->shutdown();
    EXPECT_EQ(model->reuse_authority->state(), ModelContextReuseAuthority::State::Reusable);
    ASSERT_NO_THROW(lifetime.finish());
    EXPECT_EQ(lifetime.retiredDeviceCount(),
        mpi->rank() == gpu_owner->discovery_owner ? gpu_owner->devices.size() : 0u);
}
} // namespace

#ifdef HAVE_ROCM
/** @test Real mixed-cohort movement preserves inference over original ROCm participants. */
TEST(Qwen4ExpMixedTensorShardedTP2, DynamicResidencyPublishesCompleteCohortSwapBeforeGroupedReplay)
{
    runMixedTensorShardedMovementReplay(DeviceType::ROCm);
}
#endif

#ifdef HAVE_CUDA
/** @test Preserve ordinary and grouped QFN histories through a CUDA/CPU expert swap. */
TEST(Qwen4ExpMixedTensorShardedTP2, CUDACPUCohortMovementPreservesGeneratedTokens)
{
    runMixedTensorShardedMovementReplay(DeviceType::CUDA);
}
#endif
