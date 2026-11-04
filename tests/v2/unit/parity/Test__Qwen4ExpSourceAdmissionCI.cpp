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
#include <set>
#include <nlohmann/json.hpp>
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

/** @test A physical ROCm prerequisite preserves the original six runtime requests and every fitting control. */
TEST(Qwen4ExpSourceAdmissionCI, ROCmReplicatedFull15RequiresItsAuthoredPhysicalClass)
{
    const auto original = qwen4ExpExpandSourceAdmissionMTP(qwen4ExpROCm4OverlayPolicyProductionCases());
    const auto projected = qwen4ExpROCm4OverlayPolicySourceAdmissionParameters(original);
    ASSERT_EQ(projected.size(), 12u);
    std::size_t larger = 0;
    for (std::size_t i = 0; i < original.size(); ++i)
    {
        const auto &cell = projected[i];
        std::ostringstream before, after, metadata;
        llaminar2::test::parity::PrintTo(original[i], &before);
        llaminar2::test::parity::PrintTo(static_cast<const ModelParityCase &>(cell), &after);
        EXPECT_EQ(before.str(), after.str());
        EXPECT_EQ(cell.topology.expert_overlay_plan, original[i].topology.expert_overlay_plan);
        EXPECT_FALSE(cell.minimum_cuda_total_memory_bytes);
        EXPECT_FALSE(cell.gpu_memory_cap_mb);
        EXPECT_FALSE(cell.capacity_boundary);
        PrintTo(cell, &metadata);
        const bool replicated = cell.topology.expert_overlay_plan->continuation_domain_spec.effectiveDensePolicy() ==
            llaminar2::DenseParallelPolicy::PrefillTensorParallelDecodeReplicated;
        EXPECT_EQ(cell.minimum_rocm_total_memory_bytes.has_value(), replicated && cell.mtp != ModelParityMTP::Off);
        if (cell.minimum_rocm_total_memory_bytes)
        {
            ++larger;
            EXPECT_EQ(*cell.minimum_rocm_total_memory_bytes, std::size_t{48} * 1024 * 1024 * 1024);
            EXPECT_EQ(cell.retained_mtp_draft_capacity, 15);
            EXPECT_NE(metadata.str().find("\"minimum_rocm_total_memory_bytes\":51539607552"), std::string::npos);
        }
    }
    EXPECT_EQ(larger, 6u);
    for (const auto &cell : qwen4ExpSourceAdmissionParameters(qwen4ExpROCm4ReplicatedCapacityRejectionCases()))
    {
        EXPECT_FALSE(cell.minimum_cuda_total_memory_bytes);
        EXPECT_FALSE(cell.minimum_rocm_total_memory_bytes);
        EXPECT_EQ(cell.gpu_memory_cap_mb, 32768u);
        EXPECT_EQ(cell.capacity_boundary, Qwen4ExpSourceCapacityBoundary::RoutedExpertResidency);
        std::ostringstream metadata;
        PrintTo(cell, &metadata);
        EXPECT_NE(metadata.str().find("\"capacity_boundary\":\"routed-expert-residency\""), std::string::npos);
    }
    auto foreign = original;
    for (auto &cell : foreign)
        cell.topology.participants.front().address = llaminar2::GlobalDeviceAddress::cuda(0);
    EXPECT_THROW(qwen4ExpROCm4OverlayPolicySourceAdmissionParameters(foreign), std::invalid_argument);
}

/** @test Actual GTest source discovery retains64typed policies without another compute/storage cohort. */
TEST(Qwen4ExpSourceAdmissionCI, GPUAdaptiveDiscoveryRetainsAllTypedExpertAndMTPPolicies)
{
    const auto cells=qwen4ExpExpandSourceAdmissionMTP(qwen4ExpGPUAdaptiveSourceAdmissionSeeds());
    const auto parameters=qwen4ExpSourceAdmissionParameters(cells);
    const ::testing::internal::ParamGenerator<Qwen4ExpSourceAdmissionCase> discovered=::testing::ValuesIn(parameters);
    std::set<std::string> names;std::size_t count=0,movement=0;
    for(const auto& cell:discovered){
        ASSERT_LT(count,cells.size());
        EXPECT_EQ(cell.testName(),cells[count].testName());EXPECT_EQ(cell.expert_overlay,cells[count].expert_overlay);
        EXPECT_EQ(cell.mtp,cells[count].mtp);EXPECT_EQ(cell.ngram_table_policy,cells[count].ngram_table_policy);
        EXPECT_TRUE(names.insert(cell.testName()).second);EXPECT_EQ(cell.topology.participants.size(),2u);
        EXPECT_FALSE(cell.minimum_cuda_total_memory_bytes);EXPECT_FALSE(cell.minimum_rocm_total_memory_bytes);EXPECT_FALSE(cell.gpu_memory_cap_mb);
        std::ostringstream printed;PrintTo(cell,&printed);const auto json=nlohmann::json::parse(printed.str());
        EXPECT_EQ(json.at("qfn_source_admission_schema"),2);EXPECT_EQ(json.at("id"),cell.testName());
        EXPECT_EQ(json.at("mpi_ranks"),1);EXPECT_EQ(json.at("participants").size(),2u);
        EXPECT_EQ(json.at("retained_mtp_draft_capacity"),cell.mtp==ModelParityMTP::Off?0:15);
        EXPECT_EQ(json.at("mtp_enabled"),cell.mtp!=ModelParityMTP::Off);
        if(cell.requiresPhysicalExpertMovement())++movement;
        ++count;
    }
    EXPECT_EQ(count,64u);EXPECT_EQ(names.size(),64u);EXPECT_EQ(movement,32u);
    EXPECT_EQ(qwen4ExpGPUAdaptiveProductionCases().size(),4u);
}

/** @test Native discovery retains both rank0 roles and NUMA1 contributor without a fabricated PLE participant. */
TEST(Qwen4ExpSourceAdmissionCI, CPUJointWorld2ExportsAllThreeOriginalPhysicalRoles) {
    const auto cells=qwen4ExpSourceAdmissionParameters(qwen4ExpCPUJointWorld2SourceAdmissionCases());ASSERT_EQ(cells.size(),1u);
    std::ostringstream text;PrintTo(cells.front(),&text);const auto metadata=nlohmann::json::parse(text.str());
    EXPECT_EQ(metadata.at("mpi_ranks"),2);EXPECT_FALSE(metadata.at("mtp_enabled").get<bool>());
    EXPECT_EQ(metadata.at("retained_mtp_draft_capacity"),0);
    ASSERT_EQ(metadata.at("participants").size(),3u);
    EXPECT_EQ(metadata.at("participants")[0].at("rank"),0);EXPECT_EQ(metadata.at("participants")[0].at("backend"),"rocm");
    for(size_t member=0;member<2;++member) {
        const auto& cpu=metadata.at("participants")[member+1];
        EXPECT_EQ(cpu.at("rank"),member);EXPECT_EQ(cpu.at("backend"),"cpu");
        EXPECT_EQ(cpu.at("ordinal"),0);EXPECT_EQ(cpu.at("numa_node"),member);
    }
    EXPECT_EQ(cells.front().testName(),qwen4ExpCPUJointWorld2SourceAdmissionCases().front().testName());
    EXPECT_NE(cells.front().testName().find("2xMPI"),std::string::npos);
    EXPECT_FALSE(metadata.contains("storage_participants")); // No extra cache owner/follower is invented.
}

namespace {
/** @brief Recorded original observations only; no source, backend, communicator or native readiness exists. */
llaminar2::ClusterInventory originalSocketRecords(int population) {
    llaminar2::ClusterInventory inventory;
    for(int member=0;member<population;++member) {
        llaminar2::RankInventory rank;rank.rank=member;rank.node_id=7;rank.local_rank=member;
        rank.hostname="label_"+std::to_string(member);rank.cpu.type=llaminar2::DeviceType::CPU;
        rank.cpu.numa_node=member;
        for(int socket=0;socket<population;++socket) {
            llaminar2::CPUSocketInfo fact;fact.socket_id=11+socket*3;fact.numa_node=socket;
            rank.cpu_socket_info.push_back(fact);
        }
        inventory.ranks.push_back(std::move(rank));
    }
    return inventory;
}
/** @brief Extend only recorded plan membership; no extra hardware/source cases are registered. */
ModelParityCase recordedSocketCell(int population) {
    auto cell=qwen4ExpCPUJointWorld2SourceAdmissionCases().front();
    auto plan=std::make_shared<llaminar2::MoERoutedExpertPlacementPlan>(*cell.topology.expert_overlay_plan);
    auto& cpu=plan->domains.back();cpu.participants.clear();cpu.world_ranks.clear();
    cell.topology.participants.resize(1);cell.topology.mpi_ranks=population;
    for(int member=0;member<population;++member) {
        cpu.participants.push_back(llaminar2::GlobalDeviceAddress::cpu(member));cpu.world_ranks.push_back(member);
        cell.topology.participants.push_back({llaminar2::GlobalDeviceAddress::cpu(member),member});
    }
    cell.topology.expert_overlay_plan=std::move(plan);return cell;
}
}
/** @test Conditional source metadata cannot infer physical sockets from two available NUMA IDs. */
TEST(Qwen4ExpSourceAdmissionCI, CPUJointSocketRequirementIsOriginalAndConditional) {
    const auto& joint=qwen4ExpCPUJointWorld2SourceAdmissionCases().front();
    std::ostringstream text;PrintTo(Qwen4ExpSourceAdmissionCase{joint},&text);
    const auto metadata=nlohmann::json::parse(text.str());
    ASSERT_EQ(metadata.at("cpu_expert_socket_membership").size(),2u);
    for(int member=0;member<2;++member) {
        const auto& expected=metadata.at("cpu_expert_socket_membership")[member];
        EXPECT_EQ(expected.at("rank"),member);EXPECT_EQ(expected.at("numa_node"),member);
        EXPECT_EQ(expected.at("backend"),"cpu");EXPECT_EQ(expected.at("ordinal"),0);
        EXPECT_FALSE(expected.contains("socket_id")); // A plan cannot fabricate observed package IDs.
    }
    const auto& ordinary=qwen4ExpProductionCases().front();
    std::ostringstream original;PrintTo(Qwen4ExpSourceAdmissionCase{ordinary},&original);
    EXPECT_FALSE(nlohmann::json::parse(original.str()).contains("cpu_expert_socket_membership"));
    EXPECT_TRUE(qwen4ExpCPUJointSocketMembers(ordinary).empty());
    EXPECT_NO_THROW(requireOriginalCPUJointSocketRanks(ordinary,llaminar2::ClusterInventory{}));
}
/** @test The actual bound-rank validator consumes complete original records through eight, independent of labels/arrival. */
TEST(Qwen4ExpSourceAdmissionCI, CPUJointBoundRanksUseOriginalNodeAndPhysicalSocketIdentity) {
    for(int population=2;population<=8;++population) {
        auto cell=recordedSocketCell(population);auto original=originalSocketRecords(population);
        EXPECT_NO_THROW(requireOriginalCPUJointSocketRanks(cell,original));
        std::reverse(original.ranks.begin(),original.ranks.end());
        EXPECT_NO_THROW(requireOriginalCPUJointSocketRanks(cell,original));
    }
    const auto cell=recordedSocketCell(2);const auto original=originalSocketRecords(2);
    for(int mutation=0;mutation<10;++mutation) {
        auto bad=original;
        switch(mutation) {
        case 0:bad.ranks.pop_back();break;
        case 1:bad.ranks[1].rank=0;break;
        case 2:bad.ranks[1].cpu.numa_node=0;break;
        case 3:bad.ranks[1].node_id=8;bad.ranks[1].hostname=bad.ranks[0].hostname;break;
        case 4:bad.ranks[1].cpu_socket_info[1].socket_id=bad.ranks[1].cpu_socket_info[0].socket_id;break;
        case 5:bad.ranks[1].cpu_socket_info[1].numa_node=0;break;
        case 6:bad.ranks[1].cpu_socket_info.clear();break;
        case 7:bad.ranks[1].cpu_socket_info[1].socket_id=-1;break;
        case 8:bad.ranks[1].node_id=-1;break;
        case 9:bad.ranks[1].cpu.type=llaminar2::DeviceType::ROCm;break;
        }
        EXPECT_THROW(requireOriginalCPUJointSocketRanks(cell,bad),std::invalid_argument)<<mutation;
    }
    auto missing=cell;missing.topology.participants.pop_back();
    EXPECT_THROW(qwen4ExpCPUJointSocketMembers(missing),std::invalid_argument);
    auto duplicate=cell;duplicate.topology.participants.push_back(duplicate.topology.participants.back());
    EXPECT_THROW(qwen4ExpCPUJointSocketMembers(duplicate),std::invalid_argument);
}

/** @test The actual native hardware publisher projects only original unambiguous socket facts. */
TEST(Qwen4ExpSourceAdmissionCI, CPUJointNativeSocketProjectionPreservesOriginalFacts) {
    auto facts=originalSocketRecords(2).ranks.front().cpu_socket_info;
    EXPECT_EQ(qwen4ExpOriginalSocketForNUMA(facts,0),std::optional<int>{11});
    EXPECT_EQ(qwen4ExpOriginalSocketForNUMA(facts,1),std::optional<int>{14});
    EXPECT_FALSE(qwen4ExpOriginalSocketForNUMA(facts,9));
    EXPECT_FALSE(qwen4ExpOriginalSocketForNUMA(facts,-1));
    facts.push_back(facts.front());EXPECT_FALSE(qwen4ExpOriginalSocketForNUMA(facts,0));
    facts.pop_back();facts.front().socket_id=-1;EXPECT_FALSE(qwen4ExpOriginalSocketForNUMA(facts,0));
    facts.front().socket_id=0;EXPECT_EQ(qwen4ExpOriginalSocketForNUMA(facts,0),std::optional<int>{0});
}

/** @test Recorded original participant roles owe exact counts; no new source/native topology is claimed. */
TEST(Qwen4ExpSourceAdmissionCI, CPUJointSourceObservationCountUsesOriginalLocalMembership) {
    const auto cell=qwen4ExpCPUJointWorld2SourceAdmissionCases().front();
    auto plan=*cell.topology.expert_overlay_plan;
    // This generic observer input includes an ORIGINAL declared GPU-only rank,
    // not an out-of-range integer or a native launch on this two-socket host.
    plan.domains.front().owner_rank=2;plan.domains.front().world_ranks={2};
    const auto members=llaminar2::MoEExpertOwnerMap::declaredParticipants(plan,{.reject_tensor_sharded_domains=false});
    ASSERT_TRUE(std::any_of(members.begin(),members.end(),[](const auto& participant){
        return participant.world_rank==2 && participant.address.isGPU();}));
    for(int layer=0;layer<3;++layer) {
        llaminar2::RoutedExpertLayerPlacement row;row.layer=layer;
        row.routed_expert_tier.assign(512,1);
        std::fill_n(row.routed_expert_tier.begin(),plan.routed_tiers.front().max_experts_per_layer,0);
        plan.placements.push_back(std::move(row));
    }
    EXPECT_EQ(qwen4ExpCPUJointSourceExpectedLayerObservations(plan,members,0),3u);
    EXPECT_EQ(qwen4ExpCPUJointSourceExpectedLayerObservations(plan,members,1),3u);
    EXPECT_EQ(qwen4ExpCPUJointSourceExpectedLayerObservations(plan,members,2),0u);
    auto unresolved=members;
    for(auto& participant:unresolved)if(participant.address.isCPU())participant.world_rank_known=false;
    EXPECT_THROW(qwen4ExpCPUJointSourceExpectedLayerObservations(plan,unresolved,0),std::invalid_argument);
    EXPECT_THROW(qwen4ExpCPUJointSourceExpectedLayerObservations(plan,members,-1),std::invalid_argument);
    auto whole=plan;whole.domains.back().routed_compute_policy=llaminar2::RoutedExpertComputePolicy::Apportioned;
    EXPECT_EQ(qwen4ExpCPUJointSourceExpectedLayerObservations(whole,members,0),0u);
    EXPECT_EQ(qwen4ExpCPUJointSourceExpectedLayerObservations(whole,members,1),0u);
    plan.placements.clear();EXPECT_EQ(qwen4ExpCPUJointSourceExpectedLayerObservations(plan,members,0),0u);
}

/** @test Native exact resource evidence retains original ranks without a frozen-history descriptor. */
TEST(Qwen4ExpSourceAdmissionCI, CPUJointExactResourcesUseOriginalInventoryWithoutFrozenHistory)
{
    for (int population : {2, 4, 8})
    {
        const auto cell=recordedSocketCell(population);
        auto inventory=originalSocketRecords(population);
        for(auto& rank:inventory.ranks)rank.cpu_worker_threads=3+rank.rank;
        for(int rank=0;rank<population;++rank)
        {
            const auto facts=qwen4ExpCPUJointExactRankResources(cell,inventory,rank);
            EXPECT_EQ(facts.rank,rank);EXPECT_EQ(facts.world_size,population);
            EXPECT_EQ(facts.node,7);EXPECT_EQ(facts.numa,rank);
            EXPECT_EQ(facts.socket,11+rank*3);EXPECT_EQ(facts.workers,3+rank);
        }
        EXPECT_THROW(qwen4ExpCPUJointExactRankResources(cell,inventory,-1),std::invalid_argument);
        EXPECT_THROW(qwen4ExpCPUJointExactRankResources(cell,inventory,population),std::invalid_argument);
        auto missing=inventory;missing.ranks.pop_back();
        EXPECT_THROW(qwen4ExpCPUJointExactRankResources(cell,missing,0),std::invalid_argument);
        auto aliased=inventory;aliased.ranks[1].cpu.numa_node=0;
        EXPECT_THROW(qwen4ExpCPUJointExactRankResources(cell,aliased,0),std::invalid_argument);
        inventory.ranks[0].cpu_worker_threads=0;
        EXPECT_THROW(qwen4ExpCPUJointExactRankResources(cell,inventory,0),std::invalid_argument);
    }
}
