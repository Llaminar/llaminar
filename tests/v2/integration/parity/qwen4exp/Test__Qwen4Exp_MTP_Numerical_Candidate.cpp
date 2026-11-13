/**
 * @file Test__Qwen4Exp_MTP_Numerical_Candidate.cpp
 * @brief Unqualified paired-source entrypoint using the existing canonical numerical campaign.
 *
 * Optional separate registration supplies exact source closure without fetching
 * predictors for the unchanged main matrix. No custom runner, comparison,
 * snapshot-only bypass or weaker numerical acceptance is introduced.
 */
#include "Qwen4ExpParityTestBase.h"
#include "Qwen4ExpMTPNumericalCandidateCases.h"
#include <gtest/gtest.h>

using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen4exp;

/** @brief The ordinary QFN fixture retains the complete production runner lifecycle. */
class Qwen4ExpMTPNumericalTest
    : public Qwen4ExpConfigDrivenParityTest<Qwen4ExpMTPNumericalTest>, public ModelParityCaseParameter {};

/** @test Main, recursive predictor, accepted-prefix, native byte and path gates stay mandatory. */
TEST_P(Qwen4ExpMTPNumericalTest, ProductionParity)
{
    runQwen4ExpProductionParityCampaign();
}

INSTANTIATE_TEST_SUITE_P(Qwen4ExpMTP, Qwen4ExpMTPNumericalTest,
    ::testing::ValuesIn(qwen4ExpMTPNumericalCandidateCases()),
    [](const ::testing::TestParamInfo<ModelParityCase> &info) { return info.param.testName(); });
