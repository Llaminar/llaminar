/**
 * @file Test__Qwen4ExpMTPExpertReferenceContract.cpp
 * @brief Real shared MTP suffix consumer preserves QFN canonical HC/final/logit verdicts.
 *
 * An abstract static probe never runs SetUp or constructs a model/backend. The
 * tiny expert bank uses actual route/weight/equation proofs; these tests do not
 * claim native QFN arithmetic or a model-sized original GGUF load.
 */
#include "integration/parity/ParityTestBase.h"
#include "integration/parity/qwen4exp/Qwen4ExpMTPExpertReference.h"
#include <gtest/gtest.h>
#include <type_traits>

namespace llaminar2::test::parity
{
namespace
{
/** @brief Expose the exact mutation boundary invoked by production MTP parity, without execution owners. */
class SuffixProbe : public ParityTestBase
{
public:
    using ParityTestBase::applyIndependentMoESuffixEvidence;
};
/** @brief Bind real two-slot cutoff/normalized-weight proofs and an independently indexed expert bank. */
struct Equation
{
    std::vector<float> hf_router{.5f, .251f, .249f}, router{.5f, .249f, .251f};
    std::vector<float> hf_ids{0, 1}, ids{0, 2}, weights{.5f/.751f, .251f/.751f};
    std::vector<int> bank_ids{0, 1, 2};
    std::vector<float> bank{1, 0, 0, 1, 0, -1};
    std::vector<float> canonical, conditioned;
    MoERouteConditionedPrerequisites prerequisites;
    MoERouteConditionedProof routed;
    MoESuffixInputProof inputs{MoEHFInputVerdict::Equivalent,
                              MoEHFInputVerdict::Equivalent, MoEHFInputVerdict::Equivalent};
    std::vector<float> fc{.2f, .4f}, attention{.3f, .1f}, shared{.1f, .2f};
    MoEReferenceNormalization normalization{{1.f, .75f}, 1e-6f};
    /** @brief Derive all numerical authorities through the real common proof functions. */
    Equation()
    {
        prerequisites = {
            .boundary = compareMoERoutingBoundarySelections(hf_ids.data(), ids.data(), ids.size(),
                hf_router.data(), hf_router.size(), router.data(), router.size(), 2),
            .production_weights = proveMoENormalizedWeights(router, ids, weights, 2),
            .reference_weights = proveMoENormalizedWeights(hf_router, hf_ids, weights, 2),
            .router_kl = symmetricProbabilityKLDivergence(hf_router.data(), router.data(), 3, 3),
            .maximum_router_kl = .001, .hf_input_equivalent = true, .selected_set_changed = true};
        canonical = sumMoEReferenceBank(bank, bank_ids, hf_ids, weights, 2, 2);
        conditioned = sumMoEReferenceBank(bank, bank_ids, ids, weights, 2, 2);
        routed = proveMoERouteConditionedEquation(prerequisites, false, conditioned,
            conditioned, canonical, canonical, .98);
    }
    /** @return Independent non-routed HF operands retained for an ordinary-RMS control only. */
    MoEHFSuffixOperands operands() const { return {fc, attention, shared}; }
};
/**
 * @brief Construct original canonical raw stage records including one optionally corrupted output.
 * @param corrupt Index of failed stage, or4 for no failure. @return Complete canonical suffix/logit bank.
 */
std::vector<StageComparisonResult> stages(std::size_t corrupt)
{
    constexpr std::array names{"MTP0_MOE_COMBINED_OUTPUT", "MTP0_FFN_RESIDUAL", "MTP0_FINAL_NORM", "MTP0_LM_HEAD"};
    std::vector<StageComparisonResult> result;
    for (std::size_t index = 0; index < names.size(); ++index)
    {
        // The HC residual really has4H while combined/final haveH. The
        // canonical contract must not reinterpret it as ordinary H-wide RMS.
        std::vector<float> expected(index == 1 ? 8 : 2, 1.f), actual = expected;
        if (index == corrupt)
            for (auto &value : actual) value = -value;
        result.push_back(compareParityTensorData(actual.data(), expected.data(), actual.size(), names[index], .98f));
    }
    return result;
}
}

/** @test A valid independent expert equation never grants QFN suffix or logit acceptance. */
TEST(Qwen4ExpMTPExpertReferenceContract,CanonicalCombinedHCFinalAndLogitVerdictsRemainOriginal)
{
    Equation equation;
    ASSERT_TRUE(equation.prerequisites.valid());
    ASSERT_TRUE(equation.routed.equivalent);
    static_assert(std::is_empty_v<MoECanonicalComparisons>);
    for (std::size_t corrupt = 0; corrupt < 4; ++corrupt)
    {
        SCOPED_TRACE(corrupt);
        auto actualStages = stages(corrupt);
        const auto original = actualStages;
        const MoESuffixRows noInventedSuffix{};
        SuffixProbe::applyIndependentMoESuffixEvidence(MoECanonicalComparisons{},
            equation.prerequisites, equation.routed, equation.inputs, noInventedSuffix, noInventedSuffix,
            equation.canonical, equation.conditioned, equation.operands(), .98, actualStages);
        ASSERT_EQ(actualStages.size(), original.size());
        for (std::size_t i = 0; i < original.size(); ++i)
        {
            EXPECT_EQ(actualStages[i].passed, original[i].passed);
            EXPECT_FLOAT_EQ(actualStages[i].cosine_similarity, original[i].cosine_similarity);
            EXPECT_FLOAT_EQ(actualStages[i].rel_l2_norm, original[i].rel_l2_norm);
            EXPECT_FLOAT_EQ(actualStages[i].max_abs_diff, original[i].max_abs_diff);
            EXPECT_FALSE(actualStages[i].route_conditioned_proof.has_value());
        }
        EXPECT_FALSE(actualStages[corrupt].passed);
    }
}

/** @test Ordinary families retain the exact old reconstruction and suffix adjudication. */
TEST(Qwen4ExpMTPExpertReferenceContract,OrdinaryRMSRetainsExistingProofAndDefault)
{
    Equation equation;
    ASSERT_TRUE(equation.routed.equivalent);
    EXPECT_TRUE(std::holds_alternative<MoEOrdinaryRMSReconstruction>(MoEIndependentReference{}.suffix));
    const auto canonical = reconstructMoEHFSuffix(equation.canonical, equation.operands(), equation.normalization);
    const auto conditioned = reconstructMoEHFSuffix(equation.conditioned, equation.operands(), equation.normalization);
    ASSERT_TRUE(canonical);
    ASSERT_TRUE(conditioned);
    auto actualStages = stages(4);
    for (std::size_t i = 0; i < 3; ++i) actualStages[i].passed = false;
    const std::array<bool, 3> prior{false, false, false};
    const auto expected = proveMoERouteConditionedSuffix(equation.prerequisites, equation.routed,
        equation.inputs, *conditioned, *canonical, prior, canonical, conditioned, .98);
    SuffixProbe::applyIndependentMoESuffixEvidence(MoEOrdinaryRMSReconstruction{equation.normalization},
        equation.prerequisites, equation.routed, equation.inputs, *conditioned, *canonical,
        equation.canonical, equation.conditioned, equation.operands(), .98, actualStages);
    for (std::size_t i = 0; i < 3; ++i)
    {
        ASSERT_TRUE(actualStages[i].route_conditioned_proof);
        EXPECT_EQ(actualStages[i].passed, expected[i].equivalent);
        EXPECT_EQ(actualStages[i].route_conditioned_proof->evaluated, expected[i].evaluated);
        EXPECT_EQ(actualStages[i].route_conditioned_proof->canonical_passed, prior[i]);
        EXPECT_DOUBLE_EQ(actualStages[i].route_conditioned_proof->conditioned.relative_l2,
                         expected[i].conditioned.relative_l2);
    }
    EXPECT_TRUE(actualStages.back().passed);
    EXPECT_FALSE(actualStages.back().route_conditioned_proof.has_value());
}

/** @test Missing RMS parameters fail their own typed contract rather than choosing QFN's canonical arm. */
TEST(Qwen4ExpMTPExpertReferenceContract,EmptyOrdinaryNormalizationCannotSelectCanonicalComparisons)
{
    Equation equation;
    auto actualStages = stages(4);
    const MoESuffixRows rows{};
    SuffixProbe::applyIndependentMoESuffixEvidence(MoEOrdinaryRMSReconstruction{},
        equation.prerequisites, equation.routed, equation.inputs, rows, rows,
        equation.canonical, equation.conditioned, equation.operands(), .98, actualStages);
    for (std::size_t i = 0; i < 3; ++i)
    {
        EXPECT_FALSE(actualStages[i].passed);
        ASSERT_TRUE(actualStages[i].route_conditioned_proof);
        EXPECT_FALSE(actualStages[i].route_conditioned_proof->evaluated);
    }
    EXPECT_TRUE(actualStages.back().passed);
}

/** @test Unsupported geometry or forged expert identities fail before invoking Python or reading sources. */
TEST(Qwen4ExpMTPExpertReferenceContract,InvalidBankRequestFailsBeforeSourceExecution)
{
    const qwen4exp::Qwen4ExpMTPExpertReferenceIdentity identity{
        .main = "unopened-main.gguf", .predictor = "unopened-predictor.gguf",
        .snapshots = "unopened-reference", .artifacts = "unopened-artifacts",
        .checkpoint_prefix = "decode_step0_MTP0_"};
    EXPECT_THROW((void)qwen4exp::loadQwen4ExpMTPExpertReference(identity, {0, 1}, 2, 2560), std::runtime_error);
    EXPECT_THROW((void)qwen4exp::loadQwen4ExpMTPExpertReference(identity, {1, 1}, 1, 2560), std::runtime_error);
    EXPECT_THROW((void)qwen4exp::loadQwen4ExpMTPExpertReference(identity, {-1, 1}, 1, 2560), std::runtime_error);
    EXPECT_THROW((void)qwen4exp::loadQwen4ExpMTPExpertReference(identity, {512}, 1, 2560), std::runtime_error);
    std::vector<int> excessiveUnion;
    for (int expert = 0; expert < 21; ++expert) excessiveUnion.push_back(expert);
    EXPECT_THROW((void)qwen4exp::loadQwen4ExpMTPExpertReference(identity, excessiveUnion, 1, 2560), std::runtime_error);
}
} // namespace llaminar2::test::parity
