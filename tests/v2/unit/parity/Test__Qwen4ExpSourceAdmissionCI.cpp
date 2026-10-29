/**
 * @file Test__Qwen4ExpSourceAdmissionCI.cpp
 * @brief Device-free source-only metadata projection preserves canonical cases.
 *
 * This exercises the actual parameter printer without a hardware probe, model,
 * native allocation or numerical reference. Python controls separately join the
 * emitted identities to sparse CPU/GPU inventory and reject stale evidence.
 */
#include "integration/parity/qwen4exp/Qwen4ExpSourceAdmissionCI.h"
#include <gtest/gtest.h>
#include <sstream>
using namespace llaminar2::test::parity;
using namespace llaminar2::test::parity::qwen4exp;

/** @test The source wrapper preserves every shared policy and exact path leaf. */
TEST(Qwen4ExpSourceAdmissionCI, ProjectionUsesSharedExpansionAndPreservesNames)
{
    const auto original = qwen4ExpExpandSourceAdmissionMTP(qwen4ExpExtendedSourceAdmissionCases());
    const auto projected = qwen4ExpSourceAdmissionParameters(original);
    ASSERT_EQ(projected.size(), original.size());
    ASSERT_FALSE(original.empty());
    std::ostringstream numerical_before, numerical_after;
    llaminar2::test::parity::PrintTo(original.front(), &numerical_before);
    llaminar2::test::parity::PrintTo(static_cast<const ModelParityCase &>(projected.front()), &numerical_after);
    EXPECT_EQ(numerical_before.str(), numerical_after.str());
    EXPECT_NE(numerical_before.str().find("\"model_parity_schema\":1"), std::string::npos);
    EXPECT_EQ(numerical_before.str().find("\"qfn_source_admission_schema\""), std::string::npos);
    for (std::size_t i = 0; i < original.size(); ++i)
    {
        EXPECT_EQ(projected[i].testName(), original[i].testName());
        EXPECT_EQ(projected[i].mtp, original[i].mtp);
        EXPECT_EQ(projected[i].topology.mpi_ranks, original[i].topology.mpi_ranks);
        std::ostringstream source;
        PrintTo(projected[i], &source);
        EXPECT_NE(source.str().find("\"qfn_source_admission_schema\":2"), std::string::npos);
        EXPECT_NE(source.str().find("\"participants\":["), std::string::npos);
        EXPECT_EQ(source.str().find("\"model_parity_schema\""), std::string::npos);
        for (const auto &participant : original[i].topology.participants)
        {
            EXPECT_NE(source.str().find("\"ordinal\":" + std::to_string(participant.address.device_ordinal)), std::string::npos);
            EXPECT_NE(source.str().find("\"numa_node\":" + std::to_string(participant.address.numa_node)), std::string::npos);
        }
    }
}

/** @test Public cap and retained3 metadata stay explicit without changing ordinary source identity. */
TEST(Qwen4ExpSourceAdmissionCI, CapacityAndRetainedBoundsAreTypedMetadata)
{
    const auto probes = qwen4ExpCUDAFixedHeadCapacityRejectionCases();
    const auto wrapped = qwen4ExpSourceAdmissionParameters(probes);
    ASSERT_EQ(wrapped.size(), probes.size());
    for (std::size_t i = 0; i < probes.size(); ++i)
    {
        EXPECT_EQ(wrapped[i].testName(), probes[i].cell.testName());
        EXPECT_EQ(wrapped[i].gpu_memory_cap_mb, probes[i].gpu_memory_cap_mb);
        std::ostringstream source;
        PrintTo(wrapped[i], &source);
        EXPECT_NE(source.str().find("\"gpu_memory_cap_mb\":" + std::to_string(probes[i].gpu_memory_cap_mb)), std::string::npos);
        EXPECT_NE(source.str().find("\"retained_mtp_draft_capacity\":" + std::to_string(probes[i].cell.retained_mtp_draft_capacity)), std::string::npos);
    }
    const auto positives = qwen4ExpCUDAFixedHeadFourExpertRetained3SourceAdmissionCases();
    const auto positive_metadata = qwen4ExpSourceAdmissionParameters(positives);
    ASSERT_EQ(positive_metadata.size(), positives.size());
    for (std::size_t i = 0; i < positives.size(); ++i)
    {
        EXPECT_EQ(positive_metadata[i].testName(), positives[i].testName());
        EXPECT_FALSE(positive_metadata[i].gpu_memory_cap_mb);
        std::ostringstream source;
        PrintTo(positive_metadata[i], &source);
        EXPECT_EQ(source.str().find("\"gpu_memory_cap_mb\""), std::string::npos);
        EXPECT_NE(source.str().find("\"retained_mtp_draft_capacity\":" + std::to_string(positives[i].retained_mtp_draft_capacity)), std::string::npos);
        EXPECT_NE(source.str().find(std::string("\"mtp_enabled\":") + (positives[i].mtp != ModelParityMTP::Off ? "true" : "false")), std::string::npos);
    }
}

/** @test Only the explicitly provisioned original family receives a large-card prerequisite. */
TEST(Qwen4ExpSourceAdmissionCI, LargeCardClassAppliesOnlyToOriginalFull15CUDAFixedHeadPositives)
{
    const auto original = qwen4ExpExpandSourceAdmissionMTP(qwen4ExpFixedHeadProductionCases());
    const auto projected = qwen4ExpFixedHeadSourceAdmissionParameters(original);
    ASSERT_EQ(projected.size(), original.size());
    std::size_t qualified = 0;
    for (std::size_t i = 0; i < projected.size(); ++i)
    {
        const auto &cell = projected[i];
        EXPECT_EQ(cell.testName(), original[i].testName());
        EXPECT_EQ(cell.retained_mtp_draft_capacity, original[i].retained_mtp_draft_capacity);
        EXPECT_EQ(cell.topology.expert_overlay_plan, original[i].topology.expert_overlay_plan);
        if (cell.minimum_cuda_total_memory_bytes)
        {
            ++qualified;
            EXPECT_EQ(*cell.minimum_cuda_total_memory_bytes, std::size_t{32} * 1024 * 1024 * 1024);
            EXPECT_NE(cell.mtp, ModelParityMTP::Off);
            EXPECT_EQ(cell.retained_mtp_draft_capacity, 15);
            ASSERT_TRUE(cell.ngram_table_policy->isFixedHeadSharded());
            std::size_t cuda = 0;
            for (const auto &participant : cell.topology.participants)
                if (participant.address.device_type == llaminar2::DeviceType::CUDA) ++cuda;
            EXPECT_EQ(cuda, 2u);
            std::ostringstream metadata;
            PrintTo(cell, &metadata);
            EXPECT_NE(metadata.str().find("\"minimum_cuda_total_memory_bytes\":34359738368"), std::string::npos);
        }
        if (cell.mtp == ModelParityMTP::Off) EXPECT_FALSE(cell.minimum_cuda_total_memory_bytes);
    }
    // Two communicator orders × four movement/owner policies × three active policies.
    EXPECT_EQ(qualified, 24u);
    for (const auto &cell : qwen4ExpSourceAdmissionParameters(
            qwen4ExpCUDAFixedHeadFourExpertRetained3SourceAdmissionCases()))
        EXPECT_FALSE(cell.minimum_cuda_total_memory_bytes);
    for (const auto &cell : qwen4ExpSourceAdmissionParameters(qwen4ExpCUDAFixedHeadCapacityRejectionCases()))
        EXPECT_FALSE(cell.minimum_cuda_total_memory_bytes);
    for (const auto &cell : qwen4ExpSourceAdmissionParameters(
            qwen4ExpExpandSourceAdmissionMTP(qwen4ExpRejectedSourceAdmissionCases())))
        EXPECT_FALSE(cell.minimum_cuda_total_memory_bytes);
}
