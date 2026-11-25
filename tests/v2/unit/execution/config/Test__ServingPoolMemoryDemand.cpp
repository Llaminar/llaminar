/**
 * @file Test__ServingPoolMemoryDemand.cpp
 * @brief Unit tests for the typed PMA charge of the accepted serving response pool.
 *
 * A-2c first contract: the resolved serving capacity (R/S/Q) is admitted by
 * the same physical-memory authority as every other allocation, through one
 * typed estimator input and exactly one owner charge
 * (PhysicalMemoryOwner::ServingResponsePool). These tests are device-free.
 */

#include <gtest/gtest.h>
#include <cstdint>

#include "execution/config/ServingCapacity.h"
#include "execution/config/ServingPoolMemoryDemand.h"
#include "planning/PhysicalMemoryBOM.h"

namespace llaminar2::test
{
    namespace
    {
        PhysicalMemoryBOM buildPoolBom(const ResolvedServingCapacity &capacity,
                                       const ServingPoolPerResponseBudget &budget)
        {
            PhysicalMemoryBOMBuilder bom({
                .world_rank = 0,
                .device = DeviceId::cpu(),
                .total_bytes = 1u << 30u,
                .admission_available_bytes = 1u << 30u,
            });
            addServingPoolDemand(bom, capacity, budget, ServingSlotDecisionGeometry{});
            return bom.build();
        }
    } // namespace

    TEST(ServingPoolMemoryDemand, AutomaticCapacityChargesExactProduct)
    {
        const auto capability = ServingCapacityQualificationOwner::installedEnvelope();
        const auto resolved =
            resolveServingCapacity(ServingCapacityRequest::automatic(), capability);
        const ServingPoolPerResponseBudget budget{4096};
        const auto demand = servingPoolMemoryDemand(resolved, budget, ServingSlotDecisionGeometry{});
        EXPECT_EQ(demand, static_cast<std::uint64_t>(resolved.responseLeases()) * 4096u);
        const auto bom = buildPoolBom(resolved, budget);
        const auto &line = bom.charge(PhysicalMemoryOwner::ServingResponsePool);
        EXPECT_EQ(line.planned_bytes, demand);
        EXPECT_EQ(line.already_resident_bytes, 0u);
    }

    TEST(ServingPoolMemoryDemand, AuthoredCapacityHasNoOtherOwnerDelta)
    {
        const auto capability = ServingCapacityQualificationOwner::installedEnvelope();
        const auto resolved =
            resolveServingCapacity(ServingCapacityRequest::authored(1, 8, 4), capability);
        const ServingPoolPerResponseBudget budget{2048};
        const auto demand = servingPoolMemoryDemand(resolved, budget, ServingSlotDecisionGeometry{});
        EXPECT_EQ(demand, 8u * 2048u);
        const auto bom = buildPoolBom(resolved, budget);
        // Only the serving pool line is charged: every other owner is zero.
        for (std::size_t i = 0; i < static_cast<std::size_t>(PhysicalMemoryOwner::Count); ++i)
        {
            const auto owner = PhysicalMemoryBOM::ownerAt(i);
            if (owner == PhysicalMemoryOwner::ServingResponsePool)
                continue;
            EXPECT_EQ(bom.charge(owner).planned_bytes, 0u) << toString(owner);
        }
        EXPECT_EQ(toString(PhysicalMemoryOwner::ServingResponsePool),
                  std::string_view{"serving_response_pool"});
    }

    TEST(ServingPoolMemoryDemand, RejectsZeroBudgetAndOverflow)
    {
        const auto capability = ServingCapacityQualificationOwner::installedEnvelope();
        const auto resolved =
            resolveServingCapacity(ServingCapacityRequest::automatic(), capability);
        EXPECT_THROW(servingPoolMemoryDemand(resolved, ServingPoolPerResponseBudget{0}, ServingSlotDecisionGeometry{}),
                     std::invalid_argument);
        EXPECT_THROW(servingPoolMemoryDemand(
                         resolved, ServingPoolPerResponseBudget{std::numeric_limits<std::uint64_t>::max()}, ServingSlotDecisionGeometry{}),
                     std::overflow_error);
    }

    TEST(ServingPoolMemoryDemand, ContinuousBatchingPricesOneDecisionRowPerResidentSlot)
    {
        const auto batched = resolveInstalledServingCapacity(ServingCapacityRequest::authored(4, 40, 36));
        const auto budget = ServingPoolPerResponseBudget::defaultPolicy();
        const auto slots = ServingSlotDecisionGeometry::forVocabulary(151936);
        EXPECT_EQ(servingPoolMemoryDemand(batched, budget, slots),
                  40u * budget.bytes_per_response + 4u * 151936u * sizeof(float));
        // Planning prices one row for S = 1 (serving rows may serve it); the
        // serialized executor's own demand (unset vocabulary) retains none.
        const auto serial = resolveInstalledServingCapacity(ServingCapacityRequest::automatic());
        EXPECT_EQ(servingPoolMemoryDemand(serial, budget, slots),
                  servingPoolMemoryDemand(serial, budget, ServingSlotDecisionGeometry{}) + 151936u * sizeof(float));
        // Batched demand without a vocabulary is a pricing error, never zero.
        EXPECT_THROW((void)servingPoolMemoryDemand(batched, budget, ServingSlotDecisionGeometry{}),
                     std::invalid_argument);
        EXPECT_THROW((void)servingPoolMemoryDemand(batched, budget, ServingSlotDecisionGeometry::forVocabulary(0)),
                     std::invalid_argument);
        EXPECT_THROW((void)ServingSlotDecisionGeometry::forVocabulary(-1), std::invalid_argument);
    }
} // namespace llaminar2::test
