// Intentionally no include guard: each backend-specific test translation unit
// defines QWEN36_DENSE_PREFIX_MTP_SUITE and QWEN36_DENSE_PREFIX_MTP_CASE, then
// includes this file to keep the deep SingleDevice parity surface symmetric.

#ifndef QWEN36_DENSE_PREFIX_MTP_SUITE
#error "QWEN36_DENSE_PREFIX_MTP_SUITE must be defined before including this file"
#endif

#ifndef QWEN36_DENSE_PREFIX_MTP_CASE
#error "QWEN36_DENSE_PREFIX_MTP_CASE must be defined before including this file"
#endif

#define QWEN36_DENSE_PREFIX_MTP_TEST_IMPL(suite, name) TEST(suite, name)
#define QWEN36_DENSE_PREFIX_MTP_TEST(suite, name) QWEN36_DENSE_PREFIX_MTP_TEST_IMPL(suite, name)

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, PrefixRestoreFullHit)
{
    runDensePrefixRestoreParity(QWEN36_DENSE_PREFIX_MTP_CASE(), PrefixRestoreParityMode::FullHit);
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, PrefixRestorePartialHit)
{
    runDensePrefixRestoreParity(QWEN36_DENSE_PREFIX_MTP_CASE(), PrefixRestoreParityMode::PartialHit);
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, SplitPrefillMatchesPyTorchDecodeTokens)
{
    runDenseSplitPrefillParity(QWEN36_DENSE_PREFIX_MTP_CASE(), 4);
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, MTPGreedyMatchesPyTorchDecodeTokens)
{
    runDenseMTPParity(QWEN36_DENSE_PREFIX_MTP_CASE(), false);
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, MTPGreedyDepth3MatchesPyTorchDecodeTokens)
{
    runDenseMTPParity(QWEN36_DENSE_PREFIX_MTP_CASE(), false, 3);
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, MTPGreedyDynamicDepthMatchesPyTorchDecodeTokens)
{
    runDenseDynamicMTPParity(QWEN36_DENSE_PREFIX_MTP_CASE(), false);
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, PrefixCacheMTPRestore)
{
    runDenseMTPParity(QWEN36_DENSE_PREFIX_MTP_CASE(), true);
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, PrefixCacheMTPDynamicDepthRestore)
{
    runDenseDynamicMTPParity(QWEN36_DENSE_PREFIX_MTP_CASE(), true);
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, MTPBenchmarkPromptDynamicDepthMatchesPyTorchDecodeTokens)
{
    runDenseBenchmarkStyleDynamicMTPParitySinglePass(QWEN36_DENSE_PREFIX_MTP_CASE());
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, MTPBenchmarkPromptFixedDepth1MatchesPyTorchDecodeTokens)
{
    runDenseBenchmarkStyleFixedMTPParity(QWEN36_DENSE_PREFIX_MTP_CASE(), 1);
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, MTPBenchmarkPromptFixedDepth3MatchesPyTorchDecodeTokens)
{
    runDenseBenchmarkStyleFixedMTPParity(QWEN36_DENSE_PREFIX_MTP_CASE(), 3);
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, NoMTPPhase138ContinuationMatchesPyTorch)
{
    runDenseNoMTPPhase138ContinuationMatchesPyTorch(QWEN36_DENSE_PREFIX_MTP_CASE(), 8);
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, NoMTPPhase138ThinkContinuationStageParity)
{
    runDenseNoMTPPhase138ThinkContinuationStageParity(QWEN36_DENSE_PREFIX_MTP_CASE());
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, MTPFirstTransactionLeavesSequentialState)
{
    runDenseMTPFirstTransactionLeavesSequentialState(QWEN36_DENSE_PREFIX_MTP_CASE());
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, NoMTPBenchmarkStyleFreshRunnerDeterminism)
{
    runDenseNoMTPBenchmarkStyleFreshRunnerDeterminism(
        QWEN36_DENSE_PREFIX_MTP_CASE(),
        /*decode_token_budget=*/128,
        /*reused_cycle_count=*/4);
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, NoMTPBenchmarkPromptKnownWindowPyTorchTokenParity)
{
    runDenseBenchmarkPromptKnownWindowPyTorchTokenParity(QWEN36_DENSE_PREFIX_MTP_CASE());
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, OneRowRestoreLongPrefixMatchesSequential)
{
    runDenseOneRowRestoreLongPrefixMatchesSequential(QWEN36_DENSE_PREFIX_MTP_CASE());
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, MTPEnabledForwardOnlyMatchesNoMTPDecode)
{
    runDenseMTPEnabledForwardOnlyMatchesNoMTP(QWEN36_DENSE_PREFIX_MTP_CASE());
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, MTPStochasticSamplingVerifierRuns)
{
    runDenseStochasticMTPVerifierParity(QWEN36_DENSE_PREFIX_MTP_CASE());
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, VerifierRowsDecodeEquivalentM1)
{
    runDenseMainVerifierDecodeEquivalentRowsMatchSerialDecode(
        QWEN36_DENSE_PREFIX_MTP_CASE(),
        1);
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, VerifierRowsDecodeEquivalentM2)
{
    runDenseMainVerifierDecodeEquivalentRowsMatchSerialDecode(
        QWEN36_DENSE_PREFIX_MTP_CASE(),
        2);
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, VerifierRowsDecodeEquivalentM3)
{
    runDenseMainVerifierDecodeEquivalentRowsMatchSerialDecode(
        QWEN36_DENSE_PREFIX_MTP_CASE(),
        3);
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, VerifierRowsDecodeEquivalentM4)
{
    runDenseMainVerifierDecodeEquivalentRowsMatchSerialDecode(
        QWEN36_DENSE_PREFIX_MTP_CASE(),
        4);
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, VerifierRowsGroupedDecodeEquivalentM1)
{
    runDenseMainVerifierGroupedRowsMatchSerialDecode(
        QWEN36_DENSE_PREFIX_MTP_CASE(),
        1);
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, VerifierRowsGroupedDecodeEquivalentM2)
{
    runDenseMainVerifierGroupedRowsMatchSerialDecode(
        QWEN36_DENSE_PREFIX_MTP_CASE(),
        2);
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, VerifierRowsGroupedDecodeEquivalentM3)
{
    runDenseMainVerifierGroupedRowsMatchSerialDecode(
        QWEN36_DENSE_PREFIX_MTP_CASE(),
        3);
}

QWEN36_DENSE_PREFIX_MTP_TEST(QWEN36_DENSE_PREFIX_MTP_SUITE, VerifierRowsGroupedDecodeEquivalentM4)
{
    runDenseMainVerifierGroupedRowsMatchSerialDecode(
        QWEN36_DENSE_PREFIX_MTP_CASE(),
        4);
}

#undef QWEN36_DENSE_PREFIX_MTP_TEST
#undef QWEN36_DENSE_PREFIX_MTP_TEST_IMPL
#undef QWEN36_DENSE_PREFIX_MTP_CASE
#undef QWEN36_DENSE_PREFIX_MTP_SUITE
