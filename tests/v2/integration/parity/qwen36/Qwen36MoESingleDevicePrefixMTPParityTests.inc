#ifndef QWEN36_MOE_PREFIX_MTP_SUITE
#error "QWEN36_MOE_PREFIX_MTP_SUITE must be defined before including this file"
#endif

#ifndef QWEN36_MOE_PREFIX_MTP_CASE
#error "QWEN36_MOE_PREFIX_MTP_CASE must be defined before including this file"
#endif

#ifndef QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE
#error "QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE must be defined before including this file"
#endif

#ifndef QWEN36_MOE_PREFIX_MTP_DEPTH3_CASE
#error "QWEN36_MOE_PREFIX_MTP_DEPTH3_CASE must be defined before including this file"
#endif

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, PrefixRestoreFullHit)
{
    runMoEPrefixRestoreParity(
        QWEN36_MOE_PREFIX_MTP_CASE(),
        PrefixRestoreParityMode::FullHit);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, PrefixRestorePartialHit)
{
    runMoEPrefixRestoreParity(
        QWEN36_MOE_PREFIX_MTP_CASE(),
        PrefixRestoreParityMode::PartialHit);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, IncrementalDecodeMatchesFullContext)
{
    runMoEIncrementalDecodeMatchesFullContext(QWEN36_MOE_PREFIX_MTP_CASE());
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, GreedyFreshRunnerDeterminism)
{
    runMoEGreedyFreshRunnerDeterminism(QWEN36_MOE_PREFIX_MTP_CASE());
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, MTPGreedyMatchesPyTorchDecodeTokens)
{
    runMoEMTPParity(QWEN36_MOE_PREFIX_MTP_CASE(), false);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, PrefixCacheMTPRestore)
{
    runMoEMTPParity(QWEN36_MOE_PREFIX_MTP_CASE(), true);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, MTPBenchmarkPromptSidecarStageBreakdown)
{
    runMoEMTPSidecarStageBreakdownDiagnostic(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        4);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, MTPBenchmarkStyleSkipGatherGreedyMatchesReference)
{
    runMoEMTPBenchmarkStyleSkipGatherParity(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        4);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, MTPBenchmarkStyleDepth1EightTokensMatchesReference)
{
    ScopedEnvironmentValues replay_check({
        {"LLAMINAR_MTP_VERIFY_COMMIT_REPLAY_CHECK", "1"},
        {"LLAMINAR_MTP_VERIFY_COMMIT_REPLAY_DEPTH", "4"},
    });
    runMoEMTPBenchmarkStyleSkipGatherParity(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        8,
        1,
        {},
        true);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, MTPBudgetLimitedOneTokenStepsMatchReference)
{
    runMoEMTPBudgetOneStepMatchesReference(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        4);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, MTPBenchmarkStyleDynamicDepthRequestStateResets)
{
    runMoEMTPDynamicDepthRequestStateResetBenchmarkStyle(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        16);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, MTPGreedyDepth3MatchesBaselineTokens)
{
    runMoEMTPParity(QWEN36_MOE_PREFIX_MTP_DEPTH3_CASE(), false, 3);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, MTPBenchmarkStyleDepth3LongPromptGreedyMatchesReference)
{
    auto test_case = QWEN36_MOE_PREFIX_MTP_DEPTH3_CASE();
    test_case.max_seq_len = 4096;
    runMoEMTPBenchmarkStyleSkipGatherParity(
        test_case,
        16,
        3,
        {},
        true);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, NoMTPBenchmarkStyleSkipGatherGreedyMatchesGatheredArgmax)
{
    ScopedEnvironmentValues perf_stats_enabled({
        {"LLAMINAR_PERF_STATS_SUMMARY", "1"},
    });
    PerfStatsCollector::reset();
    runMoENoMTPBenchmarkStyleSkipGatherArgmaxParity(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        16);
    PerfStatsCollector::reset();
}

#undef QWEN36_MOE_PREFIX_MTP_DEPTH3_CASE
#undef QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE
#undef QWEN36_MOE_PREFIX_MTP_CASE
#undef QWEN36_MOE_PREFIX_MTP_SUITE
