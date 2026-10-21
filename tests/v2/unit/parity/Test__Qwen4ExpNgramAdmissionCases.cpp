/**
 * @file Test__Qwen4ExpNgramAdmissionCases.cpp
 * @brief Device-free declarations for full-source ngram admission, not inferred model support.
 *
 * These checks authenticate unchanged canonical cells, exact upfront PP layer
 * ownership and the explicit CUDA capacity input without loading any model or
 * initializing a backend. Only the separate real-model admission tests prove
 * actual table materialization and the final physical-memory certificate.
 */
#include "integration/parity/qwen4exp/Qwen4ExpNgramAdmissionCases.h"
#include <gtest/gtest.h>
#include <array>
#include <set>

namespace llaminar2::test::parity::qwen4exp
{
/** @test Three storage admissions remain separate from the unchanged mathematical product. */
TEST(Qwen4ExpNgramAdmissionCases, PreservesTwentyCanonicalCellsAndFullMainSource)
{
    ASSERT_EQ(qwen4ExpProductionCases().size(), 20u);
    ASSERT_EQ(qwen4ExpNgramAdmissionCases().size(), 3u);
    std::set<std::string> names;
    for (const auto &cell : qwen4ExpNgramAdmissionCases())
    {
        EXPECT_TRUE(names.insert(cell.testName()).second);
        EXPECT_EQ(cell.model.model_path, kMainModelPath);
        EXPECT_EQ(cell.model.transformer_layers, 48);
        EXPECT_EQ(cell.model.max_seq_len, 4096);
        EXPECT_EQ(cell.model.decode_steps, 2);
        EXPECT_EQ(cell.activation_precision, ActivationPrecision::FP32);
        EXPECT_EQ(cell.kv_cache_precision, KVCachePrecision::FP16);
        EXPECT_EQ(cell.mtp, ModelParityMTP::Off);
        EXPECT_EQ(cell.retained_mtp_draft_capacity, 0);
        ASSERT_TRUE(cell.ngram_table_policy);
        EXPECT_NE(cell.ngram_table_policy->storage, ImmutableRowStoragePreference::MemoryMapped);
        EXPECT_NE(cell.ngram_table_policy->storage, ImmutableRowStoragePreference::RAM);
    }
}

/** @test Automatic CPU and24GiB CUDA use explicit unchanged production topology inputs. */
TEST(Qwen4ExpNgramAdmissionCases, AutomaticCasesHaveFixedTopologyAndCudaCeiling)
{
    std::size_t automatic = 0;
    for (const auto &cell : qwen4ExpNgramAdmissionCases())
    {
        if (cell.ngram_table_policy->storage != ImmutableRowStoragePreference::Automatic) continue;
        ++automatic;
        const auto config = qwen4ExpNgramAdmissionConfig(cell, kMainModelPath, 0);
        EXPECT_EQ(config.ngram_table_policy.storage, ImmutableRowStoragePreference::Automatic);
        EXPECT_EQ(qwen4ExpNgramAdmissionExpectedStorage(cell), ImmutableRowStorage::RAM);
        if (cell.topology.isExpertOverlay())
        {
            ASSERT_TRUE(cell.expert_overlay);
            EXPECT_EQ(*cell.expert_overlay, ModelParityExpertOverlayPolicy{});
            ASSERT_TRUE(config.moe_routed_expert_plan);
            ASSERT_EQ(config.moe_routed_expert_plan->domains.size(), 2u);
            EXPECT_EQ(config.moe_routed_expert_plan->domains.front().participants,
                      (std::vector{GlobalDeviceAddress::cuda(0)}));
            EXPECT_EQ(config.moe_routed_expert_plan->domains.back().participants,
                      (std::vector{GlobalDeviceAddress::cpu(0)}));
            EXPECT_EQ(config.max_gpu_memory_mb, 24u * 1024u);
            EXPECT_LT(*config.max_gpu_memory_mb * 1024ull * 1024ull, 28800138240ull);
        }
        else
        {
            EXPECT_EQ(config.device_for_this_rank, GlobalDeviceAddress::cpu(0));
            EXPECT_FALSE(config.max_gpu_memory_mb);
        }
    }
    EXPECT_EQ(automatic, 2u);
}

/** @test The PLE GPU owns exactly layer1, separately from the embedding GPU and CPU tail. */
TEST(Qwen4ExpNgramAdmissionCases, ResidentCaseOwnsExactlyPleLayerWithoutEmbeddingOrHead)
{
    const auto &cell = qwen4ExpNgramAdmissionCases().back();
    ASSERT_EQ(cell.ngram_table_policy->storage, ImmutableRowStoragePreference::VRAM);
    const auto config = qwen4ExpNgramAdmissionConfig(cell, kMainModelPath, 0);
    EXPECT_FALSE(config.moe_routed_expert_plan);
    EXPECT_FALSE(config.max_gpu_memory_mb);
    ASSERT_EQ(config.domain_definitions.size(), 3u);
    ASSERT_EQ(config.pp_stage_definitions.size(), 3u);
    const std::array devices{GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1), GlobalDeviceAddress::cpu(0)};
    constexpr std::array first{0, 1, 2};
    constexpr std::array last{0, 1, 47};
    for (std::size_t index = 0; index < devices.size(); ++index)
    {
        const auto &domain = config.domain_definitions[index];
        const auto &stage = config.pp_stage_definitions[index];
        EXPECT_EQ(domain.devices, (std::vector{devices[index]}));
        EXPECT_EQ(domain.scope, TPScope::RANK_LOCAL);
        EXPECT_EQ(domain.owner_rank, 0);
        EXPECT_EQ(stage.domain_name, domain.name);
        EXPECT_EQ(stage.first_layer, first[index]);
        EXPECT_EQ(stage.last_layer, last[index]);
    }
    EXPECT_EQ(devices[0].toLocalDeviceId(), DeviceId::rocm(0));
    EXPECT_EQ(devices[1].toLocalDeviceId(), DeviceId::rocm(1));
    EXPECT_EQ(devices[2].toLocalDeviceId(), DeviceId::cpu());
    EXPECT_EQ(qwen4ExpNgramAdmissionExpectedStorage(cell), ImmutableRowStorage::VRAM);
}
} // namespace llaminar2::test::parity::qwen4exp
