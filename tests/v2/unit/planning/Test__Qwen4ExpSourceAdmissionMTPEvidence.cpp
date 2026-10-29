/**
 * @file Test__Qwen4ExpSourceAdmissionMTPEvidence.cpp
 * @brief Device-free adversarial replay of source admission's actual serving manifest.
 *
 * Evidence is bound to independently planned PP/TP leaves and exact retained
 * row/outcome families. No model, allocator, native graph or accelerator runs.
 */
#include "../../integration/parity/qwen4exp/Qwen4ExpSourceAdmissionEvidence.h"
#include <gtest/gtest.h>
using namespace llaminar2;
using namespace llaminar2::test::parity::qwen4exp;
namespace
{
/** @return One immutable counter observation with explicit role tags. */
PerfStatRecord record(std::string device, std::string name, std::map<std::string, std::string> tags = {})
{
    PerfStatRecord value;
    value.domain = "forward_graph";
    value.phase = "setup";
    value.device = std::move(device);
    value.name = std::move(name);
    value.tags = std::move(tags);
    value.value = 1;
    value.count = 1;
    return value;
}
/** @brief Aggregate counters exactly as the existing collector's immutable snapshot. */
void add(std::vector<PerfStatRecord> &rows, PerfStatRecord value)
{
    auto found =
        std::find_if(rows.begin(), rows.end(), [&](const auto &r)
                     { return r.name == value.name && r.device == value.device && r.tags == value.tags; });
    if (found == rows.end())
        rows.push_back(std::move(value));
    else
    {
        found->value += value.value;
        ++found->count;
    }
}
/** @return Complete captured setup, without any controller execution observation. */
std::vector<PerfStatRecord> evidence(std::span<const Qwen4ExpMTPSourceServingParticipant> leaves)
{
    std::vector<PerfStatRecord> result;
    for (const auto &leaf : leaves)
    {
        const auto device = leaf.device.toString();
        if (leaf.completion_required)
            add(result, record(device, "serving_graph_family_participant_completions"));
        if (!leaf.device.is_gpu())
            continue;
        for (std::size_t program = 0; program < leaf.sparse_program_count; ++program)
        {
            for (const auto purpose : kMTPConditionForwardPurposes)
                add(result, record(device, "serving_graph_family_materializations",
                                   {{"sparse_query_program", std::to_string(program)}, {"role", "mtp_condition"},
                                    {"physical_rows", "1"},
                                    {"commit_purpose", std::to_string(static_cast<int>(purpose))},
                                    {"submission", "materialized_unlaunched"}}));
            for (int rows : leaf.verifier_rows)
                for (const auto outcome :
                     {MTPVerifierOutcomeGraphMode::Greedy, MTPVerifierOutcomeGraphMode::Disabled})
                    add(result, record(device, "serving_graph_family_materializations",
                                       {{"sparse_query_program", std::to_string(program)}, {"role", "mtp_grouped_verifier"},
                                        {"physical_rows", std::to_string(rows)},
                                        {"outcome_mode", std::to_string(static_cast<int>(outcome))},
                                        {"submission", "materialized_unlaunched"}}));
        }
    }
    return result;
}
/** @return Two actual GPU serving leaves, not expert-only resources. */
std::vector<Qwen4ExpMTPSourceServingParticipant> leaves()
{
    return {{"pp0/tp0", DeviceId::cuda(0), {2, 4, 8, 16}},
            {"pp1/tp0", DeviceId::rocm(0), {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}}};
}
} // namespace
TEST(Qwen4ExpSourceAdmissionMTPEvidence, CompleteManifestRequiresEveryParticipantRoleRowAndOutcome)
{
    const auto expected = leaves();
    const auto complete = evidence(expected);
    EXPECT_EQ(requireQwen4ExpMTPSourceSetupFamilies({}, complete, expected), 2u);
    auto missing = complete;
    std::erase_if(missing, [&](const auto &r) { return r.device == expected.back().device.toString(); });
    EXPECT_THROW(requireQwen4ExpMTPSourceSetupFamilies({}, missing, expected), std::logic_error);
    missing = complete;
    std::erase_if(missing,
                  [](const auto &r)
                  {
                      auto role = r.tags.find("role");
                      return role != r.tags.end() && role->second == "mtp_condition";
                  });
    EXPECT_THROW(requireQwen4ExpMTPSourceSetupFamilies({}, missing, expected), std::logic_error);
    missing = complete;
    std::erase_if(missing,
                  [](const auto &r)
                  {
                      auto row = r.tags.find("physical_rows");
                      return row != r.tags.end() && row->second == "3";
                  });
    EXPECT_THROW(requireQwen4ExpMTPSourceSetupFamilies({}, missing, expected), std::logic_error);
    missing = complete;
    missing.pop_back();
    EXPECT_THROW(requireQwen4ExpMTPSourceSetupFamilies({}, missing, expected), std::logic_error);
}
TEST(Qwen4ExpSourceAdmissionMTPEvidence, StaleResetExtraAndIncompleteSubmissionEvidenceFail)
{
    const auto expected = leaves();
    const auto complete = evidence(expected);
    EXPECT_THROW(requireQwen4ExpMTPSourceSetupFamilies(complete, complete, expected), std::logic_error);
    auto prior = complete;
    prior.front().value = 2;
    EXPECT_THROW(requireQwen4ExpMTPSourceSetupFamilies(prior, complete, expected), std::logic_error);
    auto after = complete;
    after.pop_back();
    EXPECT_THROW(requireQwen4ExpMTPSourceSetupFamilies(complete, after, expected), std::logic_error);
    after = complete;
    after.push_back(record(DeviceId::cuda(7).toString(), "serving_graph_family_participant_completions"));
    EXPECT_THROW(requireQwen4ExpMTPSourceSetupFamilies({}, after, expected), std::logic_error);
    after = complete;
    after[1].tags["submission"] = "executed";
    EXPECT_THROW(requireQwen4ExpMTPSourceSetupFamilies({}, after, expected), std::logic_error);
    after = complete;
    for (auto &r : after)
        r.value += 1;
    EXPECT_EQ(requireQwen4ExpMTPSourceSetupFamilies(complete, after, expected), 2u);
}
TEST(Qwen4ExpSourceAdmissionMTPEvidence,
     SharedCPUCounterIdentityRetainsPlanMultiplicityAndDeferredHostIsExplicit)
{
    std::vector<Qwen4ExpMTPSourceServingParticipant> expected{{"pp0", DeviceId::cpu(), {}},
                                                              {"pp1", DeviceId::cpu(), {}}};
    auto rows = evidence(expected);
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows.front().value, 2);
    EXPECT_EQ(requireQwen4ExpMTPSourceSetupFamilies({}, rows, expected), 2u);
    rows.front().value = 1;
    EXPECT_THROW(requireQwen4ExpMTPSourceSetupFamilies({}, rows, expected), std::logic_error);
    expected = {{"ordinary-host", DeviceId::cpu(), {}, false}};
    EXPECT_EQ(requireQwen4ExpMTPSourceSetupFamilies({}, {}, expected), 0u);
    expected = {{"bad-gpu", DeviceId::cuda(0), {2}, false}};
    EXPECT_THROW(requireQwen4ExpMTPSourceSetupFamilies({}, {}, expected), std::invalid_argument);
}
TEST(Qwen4ExpSourceAdmissionMTPEvidence, ActualPlanExcludesExpertDomainsAndRetainsEveryNestedTPMember)
{
    RankExecutionPlan plan;
    plan.runtime.mtp.enabled = true;
    plan.runtime.mtp.graph_capacity_draft_tokens = 15;
    plan.runtime.mtp.draft_tokens = 3;
    plan.runtime.mtp.max_request_batch = 1;
    plan.primary_device = GlobalDeviceAddress::cuda(0);
    plan.local_tp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    TPDomainParticipation expert;
    expert.domain_name = "expert-only";
    expert.devices = {GlobalDeviceAddress::cuda(7)};
    plan.my_domains.push_back(expert);
    auto expected = qwen4ExpMTPSourceServingParticipants(plan, Qwen4ExpMTPSourceProgramFamily::RankingOnly, Qwen4ExpMTPHostSetup::OrdinaryDeferred);
    ASSERT_EQ(expected.size(), 2u);
    EXPECT_EQ(expected[0].device, DeviceId::cuda(0));
    EXPECT_EQ(expected[1].device, DeviceId::cuda(1));
    EXPECT_FALSE(expected[0].verifier_rows.empty());
    const auto sparse = qwen4ExpMTPSourceServingParticipants(
        plan, Qwen4ExpMTPSourceProgramFamily::CompleteSparse, Qwen4ExpMTPHostSetup::OrdinaryDeferred);
    EXPECT_EQ(sparse.front().sparse_program_count, kSparseQueryExecutionProgramCount);
    EXPECT_EQ(expected.front().sparse_program_count, 1u);
    plan.local_tp_devices.clear();
    plan.local_pp_devices = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::cpu(0)};
    plan.local_pp_stage_tp_info.resize(2);
    plan.local_pp_stage_tp_info[0].devices = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)};
    plan.local_pp_stage_tp_info[1].devices = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)};
    expected = qwen4ExpMTPSourceServingParticipants(plan, Qwen4ExpMTPSourceProgramFamily::RankingOnly, Qwen4ExpMTPHostSetup::OrdinaryDeferred);
    ASSERT_EQ(expected.size(), 4u);
    EXPECT_TRUE(expected[2].completion_required);
    EXPECT_TRUE(expected[3].completion_required);
    EXPECT_EQ(requireQwen4ExpMTPSourceSetupFamilies({}, evidence(expected), expected), 4u);
    plan.local_pp_stage_tp_info.pop_back();
    EXPECT_THROW(qwen4ExpMTPSourceServingParticipants(plan, Qwen4ExpMTPSourceProgramFamily::RankingOnly, Qwen4ExpMTPHostSetup::OrdinaryDeferred), std::invalid_argument);
}

/** @brief Every retained program owns one family; missing, duplicated and foreign identities fail. */
TEST(Qwen4ExpSourceAdmissionMTPEvidence, SparseProgramsHaveExactIndependentFamilies)
{
    auto expected = leaves();
    for (auto &leaf : expected) leaf.sparse_program_count = kSparseQueryExecutionProgramCount;
    const auto complete = evidence(expected);
    EXPECT_EQ(requireQwen4ExpMTPSourceSetupFamilies({}, complete, expected), expected.size());
    auto missing = complete;
    std::erase_if(missing, [](const auto &r) {
        auto p = r.tags.find("sparse_query_program");
        return p != r.tags.end() && p->second == std::to_string(
            sparseQueryExecutionProgramIndex(SparseQueryExecutionProgram::ProvenNoRanking));
    });
    EXPECT_THROW(requireQwen4ExpMTPSourceSetupFamilies({}, missing, expected), std::logic_error);
    auto duplicate = complete;
    add(duplicate, complete.at(1));
    EXPECT_THROW(requireQwen4ExpMTPSourceSetupFamilies({}, duplicate, expected), std::logic_error);
    auto foreign = complete;
    foreign.at(1).tags["sparse_query_program"] = std::to_string(kSparseQueryExecutionProgramCount);
    try { requireQwen4ExpMTPSourceSetupFamilies({}, foreign, expected); FAIL(); }
    catch (const std::logic_error &error) {
        EXPECT_NE(std::string(error.what()).find("expected=0 actual=1"), std::string::npos);
        EXPECT_NE(std::string(error.what()).find("expected=1 actual=0"), std::string::npos);
    }
}

/** @brief Resolved CPU overlay TP requires every planned participant despite shared CPU counter identity. */
TEST(Qwen4ExpSourceAdmissionMTPEvidence, CPUOverlayTPMaterializationIsNotOrdinaryDeferredSetup)
{
    RankExecutionPlan plan;
    plan.primary_device = GlobalDeviceAddress::cpu(0);
    plan.local_tp_devices = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)};
    const auto overlay = qwen4ExpMTPSourceServingParticipants(plan,
        Qwen4ExpMTPSourceProgramFamily::CompleteSparse, Qwen4ExpMTPHostSetup::OverlayMaterialized);
    ASSERT_EQ(overlay.size(), 2u);
    EXPECT_TRUE(overlay[0].completion_required);
    EXPECT_TRUE(overlay[1].completion_required);
    auto actual = evidence(overlay);
    EXPECT_EQ(requireQwen4ExpMTPSourceSetupFamilies({}, actual, overlay), 2u);
    actual.front().value = 1;
    EXPECT_THROW(requireQwen4ExpMTPSourceSetupFamilies({}, actual, overlay), std::logic_error);
    EXPECT_THROW(requireQwen4ExpMTPSourceSetupFamilies({}, {}, overlay), std::logic_error);
    const auto ordinary = qwen4ExpMTPSourceServingParticipants(plan,
        Qwen4ExpMTPSourceProgramFamily::CompleteSparse, Qwen4ExpMTPHostSetup::OrdinaryDeferred);
    EXPECT_FALSE(ordinary[0].completion_required);
    EXPECT_EQ(requireQwen4ExpMTPSourceSetupFamilies({}, {}, ordinary), 0u);
}
