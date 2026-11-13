/**
 * @file Test__PlanningHostServiceEvidence.cpp
 * @brief Replay bounded service receipts and discovery-to-execution ownership.
 *
 * The production codec, sealed projection and canonical BOM execute without
 * threads or accelerators. Malformed traces fail before installed class lookup;
 * recorded class sizes make rank substitution observable in the physical BOM.
 */
#include "planning/PlanningHostServiceEvidence.h"
#include "planning/ExecutionRankMembership.h"
#include "planning/MemoryPlanner.h"
#include "../../utils/PlanningHostServiceFixture.h"
#include <gtest/gtest.h>

using namespace llaminar2;
namespace {
/** @return Exact portable field sizes; these hashes are codec inputs, not qualifications. */
std::vector<RetainedHostServiceClassProjection> classes() {
    return {{RetainedHostServiceRole::ImmutableLookupIO, std::string(64, 'a'), std::string(64, 'b'),{8u<<20,4096,4096,184,8,0,2u<<20}},
            {RetainedHostServiceRole::ExecutorParentTicket, std::string(64, 'c'), std::string(64, 'd'),{8u<<20,4096,4096,184,8,0,2u<<20}}};
}

/** @return Three authenticated metadata ranks with distinct physical-node labels. */
ClusterInventory inventory() {
    ClusterInventory result;
    result.world_size = 3;
    for (int rank = 0; rank < result.world_size; ++rank) {
        RankInventory local;
        local.rank = rank; local.node_id = rank; local.local_rank = 0;
        local.hostname = "receipt-rank-" + std::to_string(rank);
        local.cpu.memory_bytes = local.cpu.free_memory_bytes = 1ull << 30;
        result.ranks.push_back(std::move(local));
    }
    result.buildNodeAggregations();
    return result;
}
}

/** @test Empty and both-role frames preserve exact identities across repeated replay. */
TEST(PlanningHostServiceEvidence, BoundedReceiptRoundTripsWithoutNativeWork) {
    const auto attempts = RetainedHostServiceProgram::nativeEvidenceHashAttempts();
    for (const std::size_t count : {0u, 1u, 2u}) {
        auto input = classes(); input.resize(count);
        const auto frame = encodePlanningHostServiceClasses(input);
        EXPECT_EQ(frame.size(), 5u + 185u * count);
        for (int replay = 0; replay < 3; ++replay) {
            const auto decoded = decodePlanningHostServiceClasses(frame);
            ASSERT_EQ(decoded.size(), count);
            for (std::size_t index = 0; index < count; ++index) {
                EXPECT_EQ(decoded[index].role, input[index].role);
                EXPECT_EQ(decoded[index].resourceIdentity, input[index].resourceIdentity);
                EXPECT_EQ(decoded[index].liveIdentity, input[index].liveIdentity);
                EXPECT_EQ(decoded[index].declaration,input[index].declaration);
            }
        }
    }
    EXPECT_EQ(RetainedHostServiceProgram::nativeEvidenceHashAttempts(), attempts);
}

/** @test Truncation, incompatible versions, duplicate roles and bad hashes are rejected. */
TEST(PlanningHostServiceEvidence, RejectsIncompleteAndIncompatibleReceipts) {
    const auto valid = encodePlanningHostServiceClasses(classes());
    for (std::size_t size = 0; size < valid.size(); ++size)
        EXPECT_THROW(decodePlanningHostServiceClasses(std::span(valid).first(size)), std::invalid_argument);
    for (const int defect : {0, 1, 2, 3, 4, 5}) {
        auto malformed = valid;
        if (defect == 0) malformed.push_back(0);
        if (defect == 1) malformed[3] = 1;
        if (defect == 2) malformed[4] = 3;
        if (defect == 3) malformed[5] = 0;
        if (defect == 4) malformed[190] = malformed[5];
        if (defect == 5) malformed[6] = 'A';
        EXPECT_THROW(decodePlanningHostServiceClasses(malformed), std::invalid_argument) << defect;
    }
    auto duplicate = classes(); duplicate[1].role = duplicate[0].role;
    EXPECT_THROW(encodePlanningHostServiceClasses(duplicate), std::invalid_argument);
    auto invalid = classes(); invalid[0].liveIdentity.pop_back();
    EXPECT_THROW(encodePlanningHostServiceClasses(invalid), std::invalid_argument);
}

/** @test Compacted execution ranks consume their own discovery classes, never rank zero's. */
TEST(PlanningHostServiceEvidence, RankCompactionPreservesOriginalClassAndCPU0Owner) {
    const ExecutionRankMembership membership(inventory(), {2, 0});
    PlanningHostServiceBudgets budgets;
    for (int rank = 0; rank < 3; ++rank)
        budgets.emplace(rank, test::planningHostServiceProjection(4096u * (rank + 1u)));
    for (int rank = 0; rank < 2; ++rank) {
        const auto projection = projectPlanningHostServiceBudget(budgets, membership, rank);
        RetainedHostServiceDeclarations::Builder builder(rank);
        const RetainedHostServiceInstance instance{rank, DeviceId::rocm(7), RetainedHostServiceRole::ImmutableLookupIO};
        builder.addProjected(instance, projection);
        const auto declarations = std::move(builder).freeze();
        const PhysicalMemoryResource cpu{.world_rank = rank, .device = DeviceId::cpu(),
            .total_bytes = 1ull << 30, .admission_available_bytes = 1ull << 30};
        const auto plan = declarations->bom(instance, cpu);
        const auto* resource = plan.find({rank, DeviceId::cpu()});
        ASSERT_NE(resource, nullptr);
        EXPECT_EQ(resource->bytes(PhysicalMemoryOwner::RetainedHostServiceRuntime),
            4096u * (membership.discoveryRanks()[rank] + 1u));
        EXPECT_EQ(resource->bytes(PhysicalMemoryOwner::RetainedHostServiceStorage),
            projection.program(instance.role).allocationBytes());
        EXPECT_EQ(plan.find({rank, DeviceId::rocm(7)}), nullptr);
    }
    EXPECT_THROW(projectPlanningHostServiceBudget(budgets, membership, -1), std::out_of_range);
    EXPECT_THROW(projectPlanningHostServiceBudget(budgets, membership, 2), std::out_of_range);
}

/** @test Missing evidence never borrows another rank or turns an unknown class into a grant. */
TEST(PlanningHostServiceEvidence, MissingAndUnusedClassesRemainDistinct) {
    const ExecutionRankMembership membership(inventory(), {2});
    PlanningHostServiceBudgets budgets{{0, test::planningHostServiceProjection()}};
    const auto absent = projectPlanningHostServiceBudget(budgets, membership, 0);
    EXPECT_THROW(absent.program(RetainedHostServiceRole::ImmutableLookupIO), std::exception);
    const auto unused = RetainedHostServiceBudgetProjection::installed(classes());
    EXPECT_EQ(unused.classes().size(), 2u);
    // Unknown evidence may travel as metadata. Actual use still selects the
    // installed source-owned class and cannot materialize this invented value.
    EXPECT_THROW(unused.program(RetainedHostServiceRole::ExecutorParentTicket), std::invalid_argument);
}
