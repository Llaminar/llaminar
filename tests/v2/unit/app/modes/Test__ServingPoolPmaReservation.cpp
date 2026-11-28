/**
 * @file Test__ServingPoolPmaReservation.cpp
 * @brief PMA-RESERVED serving pool capacity contract (device-free).
 *
 * The serving response pool hosting demand is charged through the SINGLE typed
 * estimator (servingPoolMemoryDemand) and the capacity is COMMITTED through
 * PhysicalMemoryAuthority::reserveNewAllocations for
 * PhysicalMemoryOwner::ServingResponsePool (aggregate fit is certified by the
 * admission; the reserve enforces the admitted owner allowance; the
 * materialization claim + typed layout belong to the real engine slice).
 * These regressions prove: exact capacity reservation, the canonical owner-
 * limit logic_error, the distinct missing-line diagnostic, coexistence within
 * a fitting plan, reservation retention through the shared State, and the
 * device-free plain constructor (no test-only null shortcut).
 */
#include <gtest/gtest.h>

#include "app/modes/SerializedInferenceExecutor.h"
#include "execution/config/ServingCapacity.h"
#include "execution/config/ServingPoolMemoryDemand.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "utils/ServingCapacityTestIssuer.h"

#include <cstdint>
#include <memory>

namespace
{
    using namespace llaminar2;

    /** @brief One CPU resource observation with readable byte counts. */
    PhysicalMemoryResource cpuResource(std::size_t total, std::size_t available)
    {
        return {.world_rank = 0,
                .device = DeviceId::cpu(),
                .total_bytes = total,
                .admission_available_bytes = available};
    }

    /** @brief Authority whose CPU plan admits exactly budget bytes for the
     *  serving pool owner (plus an optional competing owner charge). */
    std::shared_ptr<PhysicalMemoryAuthority> makeAuthority(
        std::size_t serving_pool_bytes, std::size_t competing_bytes = 0)
    {
        PhysicalMemoryBOMBuilder cpu(cpuResource(serving_pool_bytes + competing_bytes,
                                                 serving_pool_bytes + competing_bytes));
        cpu.add(PhysicalMemoryOwner::ServingResponsePool, serving_pool_bytes);
        if (competing_bytes)
            cpu.add(PhysicalMemoryOwner::PrefixHostTier, competing_bytes);
        PhysicalMemoryPlanBuilder builder;
        const auto plan = builder.add(cpu.build()).build();
        auto admission = std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(
            std::move(plan));
        return std::make_shared<PhysicalMemoryAuthority>(admission, 0);
    }

    ResolvedServingCapacity capacity2()
    {
        return ServingCapacityTestIssuer::fromComponents(1, 2, 0);
    }
} // namespace

/** @brief Exact owner charge: the executor claims exactly the typed demand. */
TEST(ServingPoolPmaReservation, ReservesExactTypedDemand)
{
    const auto budget = ServingPoolPerResponseBudget{1000};
    const auto resolved = capacity2(); // R=2 -> demand = 2000 bytes
    const std::uint64_t demand = servingPoolMemoryDemand(resolved, budget, ServingSlotDecisionGeometry{});
    ASSERT_EQ(demand, 2000u);

    auto authority = makeAuthority(demand);
    {
        SerializedInferenceExecutor executor(resolved, authority, budget);
        EXPECT_TRUE(executor.servingPoolPmaReserved());
        EXPECT_EQ(executor.servingPoolCommittedBytes(),
                  static_cast<std::size_t>(demand));
        // CAPACITY reserved with the exact demand (no materialized claim yet:
        // typed-layout materialization is the real-engine slice).
        EXPECT_EQ(authority->reservedBytes(DeviceId::cpu(),
                                           PhysicalMemoryOwner::ServingResponsePool),
                  static_cast<std::size_t>(demand));
        EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(),
                                          PhysicalMemoryOwner::ServingResponsePool,
                                          PhysicalMemoryMaterializationKind::NewAllocation),
                  0u);
    }
    // Destruction releases the reservation: no residual charge.
    EXPECT_EQ(authority->reservedBytes(DeviceId::cpu(),
                                       PhysicalMemoryOwner::ServingResponsePool),
              0u);
}

/** @brief COEXISTENCE within a FITTING plan: the aggregate physical fit is
 *  certified at admission (2000 serving + 2500 competing = 4500 against
 *  4500); the reservation commits the serving owner allowance even with a
 *  LIVE competing PMA claim, and the competing fixture itself allocates no
 *  physical backing (a ledger claim only). This proves coexistence, NOT
 *  deferred exhaustion — exhaustion is not deferred here. */
TEST(ServingPoolPmaReservation, FittingPlanCoexistsWithLiveCompetingAllocation)
{
    const auto budget = ServingPoolPerResponseBudget{1000};
    const auto resolved = capacity2(); // demand 2000
    // 4500 total CPU: serving line 2000 admitted; competing owner line 2500.
    PhysicalMemoryBOMBuilder cpu(cpuResource(4500, 4500));
    cpu.add(PhysicalMemoryOwner::ServingResponsePool, 2000u);
    cpu.add(PhysicalMemoryOwner::PrefixHostTier, 2500u);
    PhysicalMemoryPlanBuilder builder;
    const auto plan = builder.add(cpu.build()).build();
    auto admission = std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(
        std::move(plan));
    auto live_authority = std::make_shared<PhysicalMemoryAuthority>(admission, 0);
    auto live_competing = live_authority->claimNewAllocation(
        DeviceId::cpu(),
        PhysicalMemoryOwner::PrefixHostTier, 2500u);
    EXPECT_TRUE(live_competing.valid());
    {
        // Capacity commit succeeds against the admitted serving line even with
        // the live competing allocation (commit != materialization).
        SerializedInferenceExecutor executor(resolved, live_authority, budget);
        EXPECT_TRUE(executor.servingPoolPmaReserved());
        EXPECT_EQ(executor.servingPoolCommittedBytes(), 2000u);
    }
    // After release: no residual serving reservation.
    EXPECT_EQ(live_authority->reservedBytes(DeviceId::cpu(),
                                            PhysicalMemoryOwner::ServingResponsePool),
              0u);
}

/** @brief Exactly matching capacity is admitted; the full amount is claimed. */
TEST(ServingPoolPmaReservation, AdmittedAtExactCapacity)
{
    const auto budget = ServingPoolPerResponseBudget{1000};
    const auto resolved = capacity2();
    auto authority = makeAuthority(2000);
    SerializedInferenceExecutor executor(resolved, authority, budget);
    EXPECT_EQ(executor.servingPoolCommittedBytes(), 2000u);
}

/** @brief NO test-only null shortcut: a null authority on the PMA path is a
 *  precise rejection; device-free tests use the plain capacity constructor. */
TEST(ServingPoolPmaReservation, NullAuthorityOnPmaPathIsPreciseRejection)
{
    const auto budget = ServingPoolPerResponseBudget{1000};
    try
    {
        SerializedInferenceExecutor executor(capacity2(), nullptr, budget);
        FAIL() << "null authority must be rejected on the PMA path";
    }
    catch (const std::invalid_argument &e)
    {
        EXPECT_NE(std::string(e.what()).find("device-free tests"), std::string::npos);
    }
    // Device-free UNBACKED mode uses the plain capacity constructor ONLY
    // (explicit queue so R >= S + Q holds).
    SerializedInferenceExecutor plain(2, 1);
    EXPECT_FALSE(plain.servingPoolPmaReserved());
    EXPECT_EQ(plain.servingPoolCommittedBytes(), 0u);
}



/** @brief FAIL-CLOSED SERVING DEMAND DECISION (BOM gate): serving requires a
 *  SEALED request authority; the owner gets the exact demand, followers get
 *  none, an UNSEALED serving state is fatal (never a silent non-owner), a
 *  mismatched seal is a follower, and non-serving is none. */
TEST(ServingPoolPmaReservation, FailClosedServingDemandForRank)
{
    // Typed authored intent: 1 resident slot, 2 response leases (demand 2000).
    const auto request = ServingCapacityRequest::authored(1, 2, 0);
    const ServingPoolPerResponseBudget budget{1000};
    // Owner rank (2 == sealed 2) -> exact demand (store first, ASSERT the
    // optional is engaged, then compare — never unchecked disengagement).
    {
        const auto demand = servingPoolDemandForRank(true, 2, 2, request, ServingSlotDecisionGeometry{}, budget);
        ASSERT_TRUE(demand.has_value());
        EXPECT_EQ(*demand, 2000u);
    }
    // Follower rank -> no demand.
    EXPECT_FALSE(servingPoolDemandForRank(true, 2, 0, request, ServingSlotDecisionGeometry{}, budget).has_value());
    // Mismatched seal (sealed 2, rank 3) is a follower: no demand.
    EXPECT_FALSE(servingPoolDemandForRank(true, 2, 3, request, ServingSlotDecisionGeometry{}, budget).has_value());
    // Non-serving -> no demand (even unsealed).
    EXPECT_FALSE(servingPoolDemandForRank(false, std::nullopt, 0, request, ServingSlotDecisionGeometry{}, budget).has_value());
    // Serving + UNSEALED -> FATAL (not a silent non-owner).
    EXPECT_THROW((void)servingPoolDemandForRank(true, std::nullopt, 0, request, ServingSlotDecisionGeometry{}, budget),
                 std::logic_error);
}

/** @brief POSITIVE OWNER-LIMIT REFUSAL (restored, actual diagnostic): a line
 *  that EXISTS but is SMALLER than the typed demand is refused by the reserve's
 *  OWN logic_error (owner incremental allowance) — with NO residual
 *  reservation/claim afterwards. The zero/absent missing-line case is the
 *  DISTINCT precise invalid_argument test. */
TEST(ServingPoolPmaReservation, InsufficientOwnerLineRefusesWithActualLogicError)
{
    const auto budget = ServingPoolPerResponseBudget{1000};
    const auto resolved = capacity2(); // demand 2000
    // Owner line 1500 < 2000 demand (competing 500 admitted separately).
    auto authority = makeAuthority(1500, 500);
    try
    {
        SerializedInferenceExecutor executor(resolved, authority, budget);
        FAIL() << "insufficient owner line must be refused";
    }
    catch (const std::invalid_argument &e)
    {
        FAIL() << "insufficient positive line must use the canonical reserve "
                  "logic_error, not a pre-intercept invalid_argument: "
               << e.what();
    }
    catch (const std::logic_error &e)
    {
        // CANONICAL owner-limit diagnostic (exact content, not merely the
        // type — std::invalid_argument derives from std::logic_error, so the
        // type alone does not discriminate).
        const std::string msg = e.what();
        EXPECT_NE(msg.find("admitted_incremental=1500"), std::string::npos);
        EXPECT_NE(msg.find("requested=2000"), std::string::npos);
        EXPECT_EQ(msg.find("addServingPoolDemand"), std::string::npos);
    }
    // NO residual reservation or claim after the refusal.
    EXPECT_EQ(authority->reservedBytes(DeviceId::cpu(),
                                       PhysicalMemoryOwner::ServingResponsePool),
              0u);
    EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(),
                                      PhysicalMemoryOwner::ServingResponsePool,
                                      PhysicalMemoryMaterializationKind::NewAllocation),
              0u);
}

/** @brief MISSING OWNER LINE: an authority whose admitted plan has no
 *  ServingResponsePool line is a PRECISE invalid_argument (never a deep
 *  reserve logic_error) — the memory plan must include addServingPoolDemand. */
TEST(ServingPoolPmaReservation, MissingOwnerLineIsPreciseFailure)
{
    const auto budget = ServingPoolPerResponseBudget{1000};
    // CPU plan admits ONLY a competing owner: no serving pool line at all.
    PhysicalMemoryBOMBuilder cpu(cpuResource(5000, 5000));
    cpu.add(PhysicalMemoryOwner::PrefixHostTier, 500u);
    PhysicalMemoryPlanBuilder builder;
    const auto plan = builder.add(cpu.build()).build();
    auto admission = std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(
        std::move(plan));
    auto authority = std::make_shared<PhysicalMemoryAuthority>(admission, 0);
    try
    {
        SerializedInferenceExecutor executor(capacity2(), authority, budget);
        FAIL() << "construction must fail without the serving pool owner line";
    }
    catch (const std::invalid_argument &e)
    {
        EXPECT_NE(std::string(e.what()).find("addServingPoolDemand"), std::string::npos);
    }
}

/** @brief CAPACITY RESERVATION retained with the shared State: the
 *  reservation outlives the executor while any response reservation holds the
 *  State, and releases only when the LAST holder drops it (final physical
 *  ownership, not worker completion). */
TEST(ServingPoolPmaReservation, ServingReservationRetainedWithStateReleasedLast)
{
    const auto budget = ServingPoolPerResponseBudget{1000};
    const auto resolved = capacity2();
    auto authority = makeAuthority(2000);
    auto executor = std::make_unique<SerializedInferenceExecutor>(
        resolved, authority, budget);
    EXPECT_TRUE(executor->servingPoolPmaReserved());
    EXPECT_EQ(executor->servingPoolCommittedBytes(), 2000u);
    // Keep ONE reservation alive past the executor.
    auto reservation = executor->tryReserve();
    ASSERT_TRUE(reservation != nullptr);
    executor.reset(); // executor (and its worker) released
    // The State (and its capacity reservation) survives via the reservation:
    // final physical ownership, not worker completion.
    EXPECT_EQ(authority->reservedBytes(DeviceId::cpu(),
                                       PhysicalMemoryOwner::ServingResponsePool),
              2000u);
    reservation = nullptr; // last holder drops the State
    EXPECT_EQ(authority->reservedBytes(DeviceId::cpu(),
                                       PhysicalMemoryOwner::ServingResponsePool),
              0u);
}
