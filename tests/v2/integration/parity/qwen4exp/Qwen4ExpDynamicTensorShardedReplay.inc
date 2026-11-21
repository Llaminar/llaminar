/**
 * @file Qwen4ExpDynamicTensorShardedReplay.inc
 * @brief Production grouped QFN inference under Dynamic tensor-sharded residency.
 *
 * The real paired-source fixture uses the published startup policy, generation
 * loop and movement ledger. One fully resident shard cohort owns each expert
 * exactly once logically. Its four GPU fragments cannot exchange owners.
 */

/** @brief Give each test process a private prefix archive until its runner retires. */
class QFNDynamicReplayScratch
{
public:
    QFNDynamicReplayScratch()
    {
        auto pattern = (std::filesystem::temp_directory_path() / "qfn-dynamic-replay-XXXXXX").string();
        if (const auto *created = ::mkdtemp(pattern.data()))
            path_ = created;
    }
    QFNDynamicReplayScratch(const QFNDynamicReplayScratch &) = delete;
    QFNDynamicReplayScratch &operator=(const QFNDynamicReplayScratch &) = delete;

    ~QFNDynamicReplayScratch()
    {
        if (!path_.empty())
        {
            std::error_code error;
            std::filesystem::remove_all(path_, error);
        }
    }

    /** @return The owned scratch directory, or an empty path on creation failure. */
    const std::filesystem::path &path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

/** @return Number of real terminal grouped verifier submissions for one sampling law. */
double qfnGroupedVerifierRuns(std::string_view counter)
{
    double count = 0;
    for (const auto &record : PerfStatsCollector::snapshot({"mtp"}))
        if (record.name == counter)
            count += record.value;
    return count;
}

/**
 * @brief Compare grouped and ordinary histories through the same prepared runner.
 * @param runner Original initialized runner; the mixed caller pins placement for both requests.
 * @param prompt Identical input token IDs for both captured generation policies.
 */
void expectQFNGroupedGreedyMatchesOrdinary(IOrchestrationRunner& runner,
    const std::vector<int32_t>& prompt)
{
    const auto active = runner.mtpRequestPolicy();
    ASSERT_TRUE(active.enabled);
    auto off = active;
    off.enabled = false;
    const SamplingParams greedy{.temperature = 0.0f, .seed = 1729u};
    constexpr std::string_view counter = "grouped_decode_equivalent_greedy_verifier_runs";

    runner.clearCache();
    ASSERT_TRUE(runner.configureMTPRequestPolicy(off)) << runner.lastError();
    const auto before = qfnGroupedVerifierRuns(counter);
    const auto serial = runner.generate(prompt, 16, greedy);
    ASSERT_TRUE(serial.success()) << serial.error;
    ASSERT_GT(serial.tokenCount(), 4u);
    EXPECT_EQ(qfnGroupedVerifierRuns(counter), before);

    runner.clearCache();
    ASSERT_TRUE(runner.configureMTPRequestPolicy(active)) << runner.lastError();
    const auto grouped = runner.generate(prompt, 16, greedy);
    ASSERT_TRUE(grouped.success()) << grouped.error;
    EXPECT_EQ(grouped.tokens, serial.tokens);
    EXPECT_GT(qfnGroupedVerifierRuns(counter), before);
}

#ifdef HAVE_ROCM
/** @test A fully resident Dynamic TP4 cohort retains its epoch across grouped graph replay. */
TEST(Qwen4ExpTensorShardedROCmTP4, DynamicResidencyOneCohortRetainsEpochAcrossGroupedReplay)
{
    const QFNDynamicReplayScratch scratch;
    ASSERT_FALSE(scratch.path().empty());
    auto config = singleTierTensorShardedFixed3Config(ModelParityExpertMovement::Dynamic);
    config.prefix_cache.disk_dir = (scratch.path() / "prefix-cache").string();
    publishOrchestrationStartupPolicy(config);
    Qwen4ExpSourceAdmissionLifetime lifetime(config, config.prefix_cache.disk_dir);
    auto *runner = &lifetime.runner();
    runner->disableSnapshotCapture();
    ASSERT_TRUE(runner->initialize()) << runner->lastError();
    EXPECT_EQ(runner->config().moe_rebalance.mode, MoERebalanceRuntimeMode::Dynamic);

    {
        const auto &reuse = lifetime.reuseContract();
        ASSERT_TRUE(reuse);
        const auto& plan = reuse->prepared_routed_weight_plan;
        ASSERT_TRUE(plan);
        ASSERT_EQ(plan->routed_tiers.size(), 1u);
        ASSERT_EQ(plan->domains.size(), 1u);
        EXPECT_EQ(plan->domains.front().routed_compute_policy, RoutedExpertComputePolicy::TensorSharded);
        EXPECT_EQ(plan->domains.front().participants.size(), 4u);
        ASSERT_FALSE(plan->placements.empty());
        for (const auto& placement : plan->placements)
        {
            EXPECT_EQ(placement.routed_expert_tier.size(), 512u);
            EXPECT_TRUE(std::all_of(placement.routed_expert_tier.begin(),
                placement.routed_expert_tier.end(), [](int tier) { return tier == 0; }));
        }
    }

    bool stationary_binding = false;
    for (const auto& record : PerfStatsCollector::snapshot({"moe_overlay_residency"}))
    {
        const auto reason = record.tags.find("reason");
        stationary_binding |= record.name == "maintenance_not_started" &&
            reason != record.tags.end() && reason->second == "single_fully_resident_tensor_shard_cohort";
    }
    EXPECT_TRUE(stationary_binding);

    const auto topology = runner->moeOptimizationMovementTopology();
    ASSERT_TRUE(topology.valid());
    EXPECT_EQ(topology.axes, MoEOptimizationMovementAxes::None);
    const auto original_epoch = runner->moeRuntimeMovementEpoch();

    std::vector<int32_t> prompt(64);
    for (std::size_t i = 0; i < prompt.size(); ++i)
        prompt[i] = static_cast<int32_t>(100 + i);
    const SamplingParams sampling{.temperature = 0.6f, .top_k = 32, .top_p = 0.95f, .seed = 1729u};
    std::vector<int32_t> expected;
    for (int replay = 0; replay < 2; ++replay)
    {
        SCOPED_TRACE(replay);
        runner->clearCache();
        const auto before = qfnGroupedVerifierRuns("grouped_decode_equivalent_stochastic_verifier_runs");
        const auto result = runner->generate(prompt, 16, sampling);
        ASSERT_TRUE(result.success()) << result.error;
        ASSERT_GT(result.tokenCount(), 4u);
        EXPECT_GT(qfnGroupedVerifierRuns("grouped_decode_equivalent_stochastic_verifier_runs"), before);
        if (replay == 0)
            expected = result.tokens;
        else
            EXPECT_EQ(result.tokens, expected);
        EXPECT_EQ(runner->moeRuntimeMovementEpoch(), original_epoch);
        EXPECT_EQ(runner->moeOptimizationMovementTopology(), topology);
        const auto status = runner->moeOptimizationStatus();
        EXPECT_FALSE(status.failed());
        EXPECT_EQ(status.published_movement_waves, 0u);
        EXPECT_EQ(status.completed_movement.transactions, 0u);
        EXPECT_EQ(status.completed_movement.physical_bytes, 0u);
        const auto ledger = runner->moeOptimizationMovementLedger();
        EXPECT_TRUE(ledger.complete());
        EXPECT_TRUE(ledger.edges.empty());
        EXPECT_TRUE(ledger.device_publications.empty());
    }
    ASSERT_NO_FATAL_FAILURE(expectQFNGroupedGreedyMatchesOrdinary(*runner, prompt));
    EXPECT_EQ(runner->moeRuntimeMovementEpoch(), original_epoch);
    ASSERT_NO_THROW(lifetime.finish());
    EXPECT_EQ(lifetime.retiredDeviceCount(), 4u);
}
#endif
