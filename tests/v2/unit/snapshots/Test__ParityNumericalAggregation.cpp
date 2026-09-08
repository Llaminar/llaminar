/**
 * @file Test__ParityNumericalAggregation.cpp
 * @brief Unit regressions for branch-aware MoE parity aggregation.
 */

#include <gtest/gtest.h>

#include "../../utils/ParityNumericalAggregation.h"

#include <array>

namespace llaminar2::test::parity
{
    TEST(Test__ParityNumericalAggregation, RoutingUsesDedicatedMetrics)
    {
        EXPECT_FALSE(parityStageContributesToLayerCosine(
            "MOE_ROUTING_INDICES", true));
        EXPECT_FALSE(parityStageContributesToLayerCosine(
            "MOE_ROUTING_WEIGHTS", true));
        EXPECT_FALSE(parityStageContributesToLayerCosine(
            "MOE_ROUTE_CONTRIBUTIONS", true));
    }

    TEST(Test__ParityNumericalAggregation, RawExpertSumRequiresSameDiscreteBranch)
    {
        EXPECT_TRUE(parityStageContributesToLayerCosine(
            "MOE_EXPERT_OUTPUT", true));
        EXPECT_FALSE(parityStageContributesToLayerCosine(
            "MOE_EXPERT_OUTPUT", false));
    }

    TEST(Test__ParityNumericalAggregation, SemanticDownstreamStagesRemainAuthoritative)
    {
        EXPECT_TRUE(parityStageContributesToLayerCosine(
            "MOE_COMBINED_OUTPUT", false));
        EXPECT_TRUE(parityStageContributesToLayerCosine(
            "FFN_RESIDUAL", false));
        EXPECT_TRUE(parityStageContributesToLayerCosine(
            "LM_HEAD", false));
    }

    TEST(
        Test__ParityNumericalAggregation,
        RecursiveMTPAggregateHasStrictProductionFloor)
    {
        EXPECT_FALSE(productionRecursiveMTPAggregatePasses(0.981706, 0.98))
            << "The observed depth-fifteen drift must fail closed";
        EXPECT_FALSE(productionRecursiveMTPAggregatePasses(0.989999, 0.98));
        EXPECT_TRUE(productionRecursiveMTPAggregatePasses(0.99, 0.98));
        EXPECT_FALSE(productionRecursiveMTPAggregatePasses(0.994, 0.995))
            << "A stricter model threshold must remain authoritative";
        EXPECT_TRUE(productionRecursiveMTPAggregatePasses(0.995, 0.995));
    }

    TEST(
        Test__ParityNumericalAggregation,
        ExplicitRecursiveMTPFloorDoesNotWeakenModelThreshold)
    {
        EXPECT_FALSE(productionRecursiveMTPAggregatePasses(
            0.979999, 0.98, 0.98));
        EXPECT_TRUE(productionRecursiveMTPAggregatePasses(
            0.981706, 0.98, 0.98));
        EXPECT_FALSE(productionRecursiveMTPAggregatePasses(
            0.989, 0.995, 0.98))
            << "A case-local recursive floor cannot weaken its model threshold";
        EXPECT_DOUBLE_EQ(
            productionRecursiveMTPAggregateRequiredCosine(0.995, 0.98),
            0.995);
    }

    TEST(
        Test__ParityNumericalAggregation,
        PromotedExpertComparisonIgnoresUnrelatedRouteSetDrift)
    {
        const std::array production_routes{3.0f, 4.0f, 3.0f, 5.0f};
        const std::array reference_routes{4.0f, 3.0f, 3.0f, 6.0f};
        const std::array domain_participants{0.0f, 1.0f, 0.0f, 1.0f};
        const std::array production_contributions{
            1.0f, 2.0f, 0.0f, 0.0f,
            3.0f, 4.0f, 0.0f, 0.0f};
        const std::array reference_contributions{
            0.0f, 0.0f, 2.0f, 4.0f,
            6.0f, 8.0f, 0.0f, 0.0f};

        const auto result = compareRoutedExpertContribution(
            production_routes,
            reference_routes,
            domain_participants,
            /*routed_expert=*/3,
            /*expected_domain_participant=*/0,
            /*top_k=*/2,
            production_contributions,
            reference_contributions);

        EXPECT_TRUE(result.validGeometry());
        EXPECT_TRUE(result.finite());
        EXPECT_EQ(result.routed_rows, 2u);
        EXPECT_EQ(result.comparable_rows, 2u);
        EXPECT_EQ(result.production_executed_rows, 2u);
        EXPECT_EQ(result.reference_executed_rows, 2u);
        EXPECT_EQ(result.exact_zero_rows, 0u);
        EXPECT_EQ(result.one_sided_zero_rows, 0u);
        EXPECT_EQ(result.compared_elements, 4u);
        EXPECT_FLOAT_EQ(result.cosine_similarity, 1.0f)
            << "Slot permutation and unrelated top-k drift must not obscure the moved expert";
    }

    TEST(
        Test__ParityNumericalAggregation,
        IncrementalComparisonAlignsTrailingReferenceRow)
    {
        const std::array production_routes{7.0f, 8.0f};
        const std::array reference_routes{1.0f, 2.0f, 8.0f, 7.0f};
        const std::array domain_participants{-1.0f, 0.0f};
        const std::array production_contributions{
            1.0f, 0.0f, 0.0f, 0.0f};
        const std::array reference_contributions{
            42.0f, 42.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f};

        const auto result = compareRoutedExpertContribution(
            production_routes,
            reference_routes,
            domain_participants,
            /*routed_expert=*/7,
            /*expected_domain_participant=*/-1,
            /*top_k=*/2,
            production_contributions,
            reference_contributions);

        EXPECT_TRUE(result.validGeometry());
        EXPECT_TRUE(result.finite());
        EXPECT_EQ(result.routed_rows, 1u);
        EXPECT_EQ(result.comparable_rows, 1u);
        EXPECT_EQ(result.compared_elements, 2u);
        EXPECT_FLOAT_EQ(result.cosine_similarity, 1.0f);
    }

    TEST(
        Test__ParityNumericalAggregation,
        ExactlyZeroRoutedExpertIsAnExplicitEqualityWitness)
    {
        const std::array production_routes{7.0f, 8.0f};
        const std::array reference_routes{8.0f, 7.0f};
        const std::array domain_participants{0.0f, 1.0f};
        const std::array production_contributions{
            0.0f, 0.0f, 4.0f, 5.0f};
        const std::array reference_contributions{
            6.0f, 7.0f, 0.0f, 0.0f};

        const auto result = compareRoutedExpertContribution(
            production_routes,
            reference_routes,
            domain_participants,
            /*routed_expert=*/7,
            /*expected_domain_participant=*/0,
            /*top_k=*/2,
            production_contributions,
            reference_contributions);

        EXPECT_TRUE(result.validGeometry());
        EXPECT_TRUE(result.finite());
        EXPECT_EQ(result.routed_rows, 1u);
        EXPECT_EQ(result.comparable_rows, 1u);
        EXPECT_EQ(result.production_executed_rows, 0u);
        EXPECT_EQ(result.reference_executed_rows, 0u);
        EXPECT_EQ(result.exact_zero_rows, 1u);
        EXPECT_EQ(result.one_sided_zero_rows, 0u);
        EXPECT_EQ(result.compared_elements, 2u);
        EXPECT_FLOAT_EQ(result.cosine_similarity, 1.0f)
            << "Exact zero in both independent paths is mathematical equality";
    }

    TEST(
        Test__ParityNumericalAggregation,
        OneSidedZeroRoutedExpertFailsClosed)
    {
        const std::array routes{7.0f, 8.0f};
        const std::array domain_participants{0.0f, 1.0f};
        const std::array production_contributions{
            0.0f, 0.0f, 4.0f, 5.0f};
        const std::array reference_contributions{
            1.0f, 2.0f, 6.0f, 7.0f};

        const auto result = compareRoutedExpertContribution(
            routes,
            routes,
            domain_participants,
            /*routed_expert=*/7,
            /*expected_domain_participant=*/0,
            /*top_k=*/2,
            production_contributions,
            reference_contributions);

        EXPECT_TRUE(result.validGeometry());
        EXPECT_TRUE(result.finite());
        EXPECT_EQ(result.comparable_rows, 1u);
        EXPECT_EQ(result.production_executed_rows, 0u);
        EXPECT_EQ(result.reference_executed_rows, 1u);
        EXPECT_EQ(result.exact_zero_rows, 0u);
        EXPECT_EQ(result.one_sided_zero_rows, 1u);
        EXPECT_FLOAT_EQ(result.cosine_similarity, 0.0f);
    }

    TEST(
        Test__ParityNumericalAggregation,
        ExpertAbsentFromReferenceIsObservedButNotNumericallyJudged)
    {
        const std::array production_routes{3.0f, 4.0f};
        const std::array reference_routes{5.0f, 6.0f};
        const std::array domain_participants{0.0f, 1.0f};
        const std::array production_contributions{
            1.0f, 2.0f, 0.0f, 0.0f};
        const std::array reference_contributions{
            -1.0f, -2.0f, 0.0f, 0.0f};

        const auto result = compareRoutedExpertContribution(
            production_routes,
            reference_routes,
            domain_participants,
            /*routed_expert=*/3,
            /*expected_domain_participant=*/0,
            /*top_k=*/2,
            production_contributions,
            reference_contributions);

        EXPECT_TRUE(result.validGeometry());
        EXPECT_TRUE(result.finite());
        EXPECT_EQ(result.routed_rows, 1u);
        EXPECT_EQ(result.comparable_rows, 0u);
        EXPECT_EQ(result.compared_elements, 0u);
        EXPECT_FLOAT_EQ(result.cosine_similarity, 0.0f);
    }

    TEST(
        Test__ParityNumericalAggregation,
        MalformedRouteGeometryFailsClosed)
    {
        const std::array production_routes{3.5f, 4.0f};
        const std::array reference_routes{3.0f, 4.0f};
        const std::array domain_participants{0.0f, 1.0f};
        const std::array contributions{1.0f, 2.0f, 0.0f, 0.0f};

        const auto result = compareRoutedExpertContribution(
            production_routes,
            reference_routes,
            domain_participants,
            /*routed_expert=*/3,
            /*expected_domain_participant=*/0,
            /*top_k=*/2,
            contributions,
            contributions);

        EXPECT_FALSE(result.validGeometry());
        EXPECT_FALSE(result.finite());
    }

    TEST(
        Test__ParityNumericalAggregation,
        NonzeroContributionCosineIsScaleInvariantBelowLegacyFloor)
    {
        const std::array routes{7.0f, 8.0f};
        const std::array domain_participants{-1.0f, 0.0f};
        const std::array production_contributions{
            1.0e-7f, -2.0e-7f, 4.0f, 5.0f};
        const std::array reference_contributions{
            2.0e-7f, -4.0e-7f, 6.0f, 7.0f};

        const auto result = compareRoutedExpertContribution(
            routes,
            routes,
            domain_participants,
            /*routed_expert=*/7,
            /*expected_domain_participant=*/-1,
            /*top_k=*/2,
            production_contributions,
            reference_contributions);

        EXPECT_EQ(
            result.state,
            RoutedExpertContributionState::
                ComparableNonzeroBelowCosineResolution);
        EXPECT_TRUE(result.validGeometry());
        EXPECT_TRUE(result.finite());
        EXPECT_TRUE(result.numericallyComparable());
        EXPECT_TRUE(result.passes(0.999f));
        EXPECT_FLOAT_EQ(result.cosine_similarity, 1.0f)
            << "Cosine is scale invariant; a low-energy route is not absent";
    }

    TEST(
        Test__ParityNumericalAggregation,
        TinyOrthogonalContributionsStillFailNumericalParity)
    {
        const std::array routes{7.0f, 8.0f};
        const std::array domain_participants{-1.0f, 0.0f};
        const std::array production_contributions{
            1.0e-7f, 0.0f, 4.0f, 5.0f};
        const std::array reference_contributions{
            0.0f, 1.0e-7f, 6.0f, 7.0f};

        const auto result = compareRoutedExpertContribution(
            routes,
            routes,
            domain_participants,
            /*routed_expert=*/7,
            /*expected_domain_participant=*/-1,
            /*top_k=*/2,
            production_contributions,
            reference_contributions);

        EXPECT_EQ(
            result.state,
            RoutedExpertContributionState::
                ComparableNonzeroBelowCosineResolution);
        EXPECT_TRUE(result.numericallyComparable());
        EXPECT_FALSE(result.passes(0.98f));
        EXPECT_FLOAT_EQ(result.cosine_similarity, 0.0f)
            << "Removing the magnitude floor must not bless wrong vectors";
    }

    TEST(
        Test__ParityNumericalAggregation,
        BelowResolutionRouteNeedsPassingCompletedAggregate)
    {
        const std::array routes{7.0f, 8.0f};
        const std::array domain_participants{-1.0f, 0.0f};
        const std::array production_contributions{
            1.0e-7f, 0.0f, 4.0f, 5.0f};
        const std::array reference_contributions{
            0.0f, 1.0e-7f, 6.0f, 7.0f};

        const auto result = compareRoutedExpertContribution(
            routes,
            routes,
            domain_participants,
            /*routed_expert=*/7,
            /*expected_domain_participant=*/-1,
            /*top_k=*/2,
            production_contributions,
            reference_contributions);

        EXPECT_EQ(
            classifyRoutedExpertContributionProof(
                result,
                /*cosine_threshold=*/0.98f,
                /*post_return_aggregate_cosine=*/0.97f,
                RoutedExpertReferenceLineage::Canonical),
            RoutedExpertContributionProof::NumericalMismatch);
        const auto aggregate_proof =
            classifyRoutedExpertContributionProof(
                result,
                /*cosine_threshold=*/0.98f,
                /*post_return_aggregate_cosine=*/0.99f,
                RoutedExpertReferenceLineage::Canonical);
        EXPECT_EQ(
            aggregate_proof,
            RoutedExpertContributionProof::
                PostReturnAggregateBelowCosineResolution);
        EXPECT_TRUE(routedExpertContributionProofPasses(aggregate_proof));
        EXPECT_GT(result.absolute_l2_error, 0.0f);
        EXPECT_GT(result.root_mean_square_error, 0.0f);
    }

    TEST(
        Test__ParityNumericalAggregation,
        ResolvableRouteMismatchCannotBeRescuedByAggregate)
    {
        const std::array routes{7.0f, 8.0f};
        const std::array domain_participants{-1.0f, 0.0f};
        const std::array production_contributions{
            1.0f, 0.0f, 4.0f, 5.0f};
        const std::array reference_contributions{
            0.0f, 1.0f, 6.0f, 7.0f};

        const auto result = compareRoutedExpertContribution(
            routes,
            routes,
            domain_participants,
            /*routed_expert=*/7,
            /*expected_domain_participant=*/-1,
            /*top_k=*/2,
            production_contributions,
            reference_contributions);

        ASSERT_EQ(
            result.state,
            RoutedExpertContributionState::ComparableNonzero);
        const auto proof = classifyRoutedExpertContributionProof(
            result,
            /*cosine_threshold=*/0.98f,
            /*post_return_aggregate_cosine=*/1.0f,
            RoutedExpertReferenceLineage::Canonical);
        EXPECT_EQ(
            proof,
            RoutedExpertContributionProof::NumericalMismatch);
        EXPECT_FALSE(routedExpertContributionProofPasses(proof));
        EXPECT_EQ(
            routedExpertContributionProofDisposition(proof),
            RoutedExpertContributionDisposition::Failed);
    }

    TEST(
        Test__ParityNumericalAggregation,
        PriorRoutingDivergenceMakesLaterNumericalMismatchInconclusive)
    {
        const std::array routes{7.0f, 8.0f};
        const std::array domain_participants{0.0f, 1.0f};
        const std::array production_contributions{
            1.0f, 0.0f, 4.0f, 5.0f};
        const std::array reference_contributions{
            0.0f, 1.0f, 6.0f, 7.0f};
        const auto result = compareRoutedExpertContribution(
            routes,
            routes,
            domain_participants,
            /*routed_expert=*/7,
            /*expected_domain_participant=*/0,
            /*top_k=*/2,
            production_contributions,
            reference_contributions);

        const auto proof = classifyRoutedExpertContributionProof(
            result,
            /*cosine_threshold=*/0.98f,
            /*post_return_aggregate_cosine=*/0.95f,
            RoutedExpertReferenceLineage::DivergedByPriorRouting);
        EXPECT_EQ(
            proof,
            RoutedExpertContributionProof::
                InconclusiveAfterRouteDivergence);
        EXPECT_FALSE(routedExpertContributionProofPasses(proof));
        EXPECT_EQ(
            routedExpertContributionProofDisposition(proof),
            RoutedExpertContributionDisposition::Inconclusive);
    }

    TEST(
        Test__ParityNumericalAggregation,
        CurrentRouteDivergencePreservesOnlyMatchedPerRouteEvidence)
    {
        const auto current = advanceRoutedExpertReferenceLineage(
            RoutedExpertReferenceLineage::Canonical,
            /*current_routes_equal=*/false);
        EXPECT_EQ(
            current.current_layer,
            RoutedExpertReferenceLineage::DivergedAtCurrentRouting);
        EXPECT_EQ(
            current.next_layer,
            RoutedExpertReferenceLineage::DivergedByPriorRouting);
        EXPECT_TRUE(routedExpertInputLineageIsCanonical(
            current.current_layer));
        EXPECT_FALSE(routedExpertAggregateLineageIsCanonical(
            current.current_layer));

        const auto later = advanceRoutedExpertReferenceLineage(
            current.next_layer,
            /*current_routes_equal=*/true);
        EXPECT_EQ(
            later.current_layer,
            RoutedExpertReferenceLineage::DivergedByPriorRouting);
        EXPECT_EQ(
            later.next_layer,
            RoutedExpertReferenceLineage::DivergedByPriorRouting);
        EXPECT_FALSE(routedExpertInputLineageIsCanonical(
            later.current_layer));
    }

    TEST(
        Test__ParityNumericalAggregation,
        CurrentRouteDivergenceCannotCertifyDeferredDenseAggregate)
    {
        const std::array routes{7.0f, 8.0f};
        const std::array domain_participants{-1.0f, 0.0f};
        const std::array production_contributions{
            0.0f, 0.0f, 4.0f, 5.0f};
        const std::array reference_contributions{
            1.0f, 2.0f, 6.0f, 7.0f};
        const auto comparison = compareRoutedExpertContribution(
            routes,
            routes,
            domain_participants,
            /*routed_expert=*/7,
            /*expected_domain_participant=*/-1,
            /*top_k=*/2,
            production_contributions,
            reference_contributions);

        ASSERT_EQ(
            comparison.state,
            RoutedExpertContributionState::OneSidedZero);
        const auto canonical = classifyPublishedRoutedExpertContribution(
            RoutedExpertContributionPublication::DeferredRemoteAggregate,
            comparison,
            /*cosine_threshold=*/0.99f,
            /*post_return_expert_output_cosine=*/0.999f,
            RoutedExpertReferenceLineage::Canonical);
        EXPECT_EQ(
            canonical,
            RoutedExpertContributionProof::
                PostReturnAggregateDeferredPublication);
        EXPECT_TRUE(routedExpertContributionProofPasses(canonical));

        const auto different_route_set =
            classifyPublishedRoutedExpertContribution(
                RoutedExpertContributionPublication::DeferredRemoteAggregate,
                comparison,
                /*cosine_threshold=*/0.99f,
                /*post_return_expert_output_cosine=*/0.999f,
                RoutedExpertReferenceLineage::DivergedAtCurrentRouting);
        EXPECT_EQ(
            different_route_set,
            RoutedExpertContributionProof::
                InconclusiveAfterRouteDivergence);
        EXPECT_EQ(
            routedExpertContributionProofDisposition(different_route_set),
            RoutedExpertContributionDisposition::Inconclusive);
    }

    TEST(
        Test__ParityNumericalAggregation,
        PriorRoutingDivergenceCannotExcuseMissingExecution)
    {
        const std::array routes{7.0f, 8.0f};
        const std::array domain_participants{0.0f, 1.0f};
        const std::array production_contributions{
            0.0f, 0.0f, 4.0f, 5.0f};
        const std::array reference_contributions{
            1.0f, 2.0f, 6.0f, 7.0f};
        const auto result = compareRoutedExpertContribution(
            routes,
            routes,
            domain_participants,
            /*routed_expert=*/7,
            /*expected_domain_participant=*/0,
            /*top_k=*/2,
            production_contributions,
            reference_contributions);

        const auto proof = classifyRoutedExpertContributionProof(
            result,
            /*cosine_threshold=*/0.98f,
            /*post_return_aggregate_cosine=*/1.0f,
            RoutedExpertReferenceLineage::DivergedByPriorRouting);
        EXPECT_EQ(proof, RoutedExpertContributionProof::OneSidedZero);
        EXPECT_EQ(
            routedExpertContributionProofDisposition(proof),
            RoutedExpertContributionDisposition::Failed);
    }
} // namespace llaminar2::test::parity
