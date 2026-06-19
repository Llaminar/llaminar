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

#ifndef QWEN36_MOE_PREFIX_MTP_EXPECTS_DIRECT_PUBLICATION
#define QWEN36_MOE_PREFIX_MTP_EXPECTS_DIRECT_PUBLICATION 0
#endif

#ifndef QWEN36_MOE_PREFIX_MTP_EXPECTS_PERSISTENT_SIDECAR_METADATA
#define QWEN36_MOE_PREFIX_MTP_EXPECTS_PERSISTENT_SIDECAR_METADATA 0
#endif

#ifndef QWEN36_MOE_PREFIX_MTP_TESTS_DEVICE_RESIDENT_PUBLICATION
#define QWEN36_MOE_PREFIX_MTP_TESTS_DEVICE_RESIDENT_PUBLICATION 0
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

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, MTPStochasticSamplingVerifierRuns)
{
    runMoEStochasticMTPVerifierParity(QWEN36_MOE_PREFIX_MTP_CASE());
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

#if QWEN36_MOE_PREFIX_MTP_EXPECTS_PERSISTENT_SIDECAR_METADATA
TEST(QWEN36_MOE_PREFIX_MTP_SUITE, MTPBenchmarkStyleUsesPersistentMoESidecarMetadata)
{
    runMoEMTPPersistentSidecarMetadataParity(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE());
}
#endif

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, VerifierRowsDecodeEquivalentM1)
{
    runMoEMainVerifierDecodeEquivalentRowsMatchSerialDecode(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        1);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, VerifierRowsDecodeEquivalentM2)
{
    runMoEMainVerifierDecodeEquivalentRowsMatchSerialDecode(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        2);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, VerifierRowsDecodeEquivalentM3)
{
    runMoEMainVerifierDecodeEquivalentRowsMatchSerialDecode(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        3);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, VerifierRowsDecodeEquivalentM4)
{
    runMoEMainVerifierDecodeEquivalentRowsMatchSerialDecode(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        4);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, VerifierRowsGroupedDecodeEquivalentM1)
{
    runMoEMainVerifierGroupedRowsMatchSerialDecode(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        1);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, VerifierRowsGroupedDecodeEquivalentM2)
{
    runMoEMainVerifierGroupedRowsMatchSerialDecode(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        2);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, VerifierRowsGroupedDecodeEquivalentM3)
{
    runMoEMainVerifierGroupedRowsMatchSerialDecode(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        3);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, VerifierRowsGroupedDecodeEquivalentM4)
{
    runMoEMainVerifierGroupedRowsMatchSerialDecode(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        4);
}

#if QWEN36_MOE_PREFIX_MTP_EXPECTS_DIRECT_PUBLICATION
TEST(QWEN36_MOE_PREFIX_MTP_SUITE, MTPSidecarPreservesMainRuntimeState)
{
    ScopedEnvironmentValues sidecar_preservation_check({
        {"LLAMINAR_MTP_VERIFY_SIDECAR_PRESERVES_MAIN_STATE", "1"},
        {"LLAMINAR_PREFIX_PROBE_HASH_KV_PAYLOADS", "1"},
        {"LLAMINAR_PREFIX_PROBE_HASH_GDN_DEVICE_STATE", "1"},
    });
    runMoEMTPBenchmarkStyleSkipGatherParity(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        4,
        1,
        {},
        true);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, MainVerifierAllPositionRowsMatchSerialDecode)
{
    runMoEMainVerifierAllPositionRowsMatchSerialDecode(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE());
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, MainVerifierRowIndexedAllPositionRowsMatchSerialDecode)
{
    runMoEMainVerifierAllPositionRowsMatchSerialDecode(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        true);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, MainVerifierRowIndexedSkipGatherRowsMatchSerialDecode)
{
    runMoEMainVerifierAllPositionRowsMatchSerialDecode(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        true,
        true);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, MainVerifierCompactDeviceOutcomeMatchesRowArgmax)
{
    runMoEMainVerifierAllPositionRowsMatchSerialDecode(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        true,
        true,
        true);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, MainVerifierDeferredCompactOutcomeMatchesRowArgmax)
{
    runMoEMainVerifierAllPositionRowsMatchSerialDecode(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        true,
        true,
        true,
        true);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, MainVerifierPublishedStateMatchesSerialContinuation)
{
    runMoEMainVerifierAllPositionRowsMatchSerialDecode(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        true,
        true,
        true,
        true,
        true);
}
#else
TEST(QWEN36_MOE_PREFIX_MTP_SUITE, MainVerifierUsesDecodeEquivalentReplayWhenPublicationUnsupported)
{
    runMoEMainVerifierUsesDecodeEquivalentReplayWhenPublicationUnsupported(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE());
}
#endif

#if QWEN36_MOE_PREFIX_MTP_TESTS_DEVICE_RESIDENT_PUBLICATION
TEST(QWEN36_MOE_PREFIX_MTP_SUITE, MainVerifierDeviceResidentPublicationM2MatchesSerialContinuation)
{
    runMoEMainVerifierAllPositionRowsMatchSerialDecode(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        true,
        true,
        true,
        true,
        true,
        2,
        true,
        true);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, MainVerifierDeviceResidentPublicationM3MatchesSerialContinuation)
{
    runMoEMainVerifierAllPositionRowsMatchSerialDecode(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        true,
        true,
        true,
        true,
        true,
        3,
        true,
        true);
}

TEST(QWEN36_MOE_PREFIX_MTP_SUITE, MainVerifierDeviceResidentPublicationM4MatchesSerialContinuation)
{
    runMoEMainVerifierAllPositionRowsMatchSerialDecode(
        QWEN36_MOE_PREFIX_MTP_BENCHMARK_CASE(),
        true,
        true,
        true,
        true,
        true,
        4,
        true,
        true);
}
#endif

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
#undef QWEN36_MOE_PREFIX_MTP_TESTS_DEVICE_RESIDENT_PUBLICATION
#undef QWEN36_MOE_PREFIX_MTP_EXPECTS_SIDECAR_REPLAY_PRESERVED
#undef QWEN36_MOE_PREFIX_MTP_EXPECTS_DIRECT_PUBLICATION
#undef QWEN36_MOE_PREFIX_MTP_SUITE
