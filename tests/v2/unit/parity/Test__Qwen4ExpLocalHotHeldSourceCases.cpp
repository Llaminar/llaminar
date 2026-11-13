/**
 * @file Test__Qwen4ExpLocalHotHeldSourceCases.cpp
 * @brief Fast original-source and ownership checks for actual held-source model requests.
 *
 * These declarations grant no native source/cohort, receipt or graph permission.
 * Native causal progress and whole model retirement are separate model cases.
 */
#include "integration/parity/qwen4exp/Qwen4ExpLocalHotHeldSourceCases.h"
#include "config/ConfigValidator.h"
#include <gtest/gtest.h>
#include <set>

namespace llaminar2::test::parity::qwen4exp {
/** @test Both original native backends preserve exact seed policy while inverting only local capacity. */
TEST(Qwen4ExpLocalHotHeldSourceCases, OriginalPoliciesRequirePoolC0AndOnePositiveOriginalConsumer)
{
    const auto cases = qwen4ExpLocalHotHeldSourceCases();
    ASSERT_EQ(cases.size(), 4u);
    std::set<std::string> identities;
    std::set<std::pair<DeviceType, std::size_t>> covered;
    for (const auto& cell : cases)
    {
        EXPECT_TRUE(identities.insert(cell.testName()).second);
        EXPECT_EQ(cell.mtp, ModelParityMTP::Off);
        const auto consumers = qwen4ExpLocalHotOriginalConsumers(cell);
        ASSERT_EQ(consumers.size(), 2u);
        const auto& gpu = cell.ngram_table_policy->gpu_adaptive;
        ASSERT_TRUE(gpu.local_hot_cache);
        gpu.local_hot_cache->requireConsumers(consumers);
        EXPECT_EQ(gpu.source_backing, AdaptiveImmutableRowBacking::RAM);
        EXPECT_EQ(gpu.placement.mode, AdaptiveGPUCachePlacementMode::DeclaredRankLocal);
        for (const auto& owner : gpu.placement.owners)
        {
            EXPECT_EQ(owner.maximum_cache_rows, 0u);
            EXPECT_EQ(owner.cache_physical_byte_limit, 0u);
        }
        std::size_t positive = 0;
        for (std::size_t index = 0; index < consumers.size(); ++index)
        {
            const auto budget = gpu.local_hot_cache->ceilingsFor(consumers[index]);
            if (budget.cache_physical_byte_limit)
            {
                ++positive;
                EXPECT_EQ(budget.maximum_cache_rows, std::uint64_t{1} << 20);
                EXPECT_EQ(budget.cache_physical_byte_limit, std::size_t{64} << 20);
                EXPECT_TRUE(covered.insert({consumers[index].backend, index}).second);
            }
            else EXPECT_EQ(budget.maximum_cache_rows, 0u);
        }
        EXPECT_EQ(positive, 1u);
        EXPECT_EQ(gpu.local_hot_cache->policy, gpu.gpu_policy);
        const auto target = consumers.front().backend == DeviceType::ROCm
            ? Qwen4ExpHardwareTarget::ROCm2 : Qwen4ExpHardwareTarget::CUDA2;
        const auto pressure = target == Qwen4ExpHardwareTarget::ROCm2
            ? Qwen4ExpOwnerSubsetCachePressure::Zero : Qwen4ExpOwnerSubsetCachePressure::Positive;
        const auto original = qwen4ExpCombinedLocalHotC0CasesFromOriginalSeed(
            {target, 2, AdaptiveImmutableRowBacking::RAM, pressure, ModelParityExpertMovement::Static},
            {ModelParityMTP::Off}).front();
        auto restored = cell;
        restored.topology.test_id = original.topology.test_id;
        restored.ngram_table_policy->gpu_adaptive.local_hot_cache = original.ngram_table_policy->gpu_adaptive.local_hot_cache;
        EXPECT_EQ(::testing::PrintToString(restored), ::testing::PrintToString(original));
        EXPECT_TRUE(ConfigValidator::createStandard().validateToStrings(
            cell.makeOrchestrationConfig(cell.model.model_path, 0)).empty());
    }
    EXPECT_EQ(covered.size(), 4u);
}
} // namespace llaminar2::test::parity::qwen4exp
