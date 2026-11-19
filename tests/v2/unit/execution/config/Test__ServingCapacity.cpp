/**
 * @file Test__ServingCapacity.cpp
 * @brief Unit tests for the typed serving-capacity contract (design §9.4).
 *
 * The contract is the single ownership-count input for continuous batching:
 * resident slots S, response leases R, and queue/preparing capacity Q. These
 * tests prove the exact-or-reject resolution semantics, the invariant envelope
 * (S >= 1, Q >= 0, R >= S + Q), explicit per-axis admission checks
 * (Q bounds preparing-or-queued, S bounds execution-attached, R bounds the
 * total including detached delivery), value immutability, and that an authored
 * positive capacity is never silently reduced. `capacityOnePreset()` is a
 * minimal fixture preset: the current HTTP accepted-response pool
 * (kDefaultCapacity = 32) is intentionally not represented in this slice.
 */

#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

#include "execution/config/ServingCapacity.h"

namespace llaminar2::test
{
    namespace
    {
        constexpr ServingCapacityCapability kCapacityOnePreset = ServingCapacityCapability::capacityOnePreset();
    } // namespace

    // Borrowed-immutability contract: value types may be copied/moved into a
    // new object, but never reassigned behind an existing borrower.
    static_assert(!std::is_copy_assignable_v<ServingCapacityRequest>);
    static_assert(!std::is_move_assignable_v<ServingCapacityRequest>);
    static_assert(std::is_copy_constructible_v<ServingCapacityRequest>);
    static_assert(std::is_move_constructible_v<ServingCapacityRequest>);
    static_assert(!std::is_copy_assignable_v<ServingCapacityCapability>);
    static_assert(!std::is_move_assignable_v<ServingCapacityCapability>);
    static_assert(std::is_copy_constructible_v<ServingCapacityCapability>);
    static_assert(std::is_move_constructible_v<ServingCapacityCapability>);
    static_assert(!std::is_copy_assignable_v<ResolvedServingCapacity>);
    static_assert(!std::is_move_assignable_v<ResolvedServingCapacity>);
    static_assert(std::is_copy_constructible_v<ResolvedServingCapacity>);
    static_assert(std::is_move_constructible_v<ResolvedServingCapacity>);

    TEST(ServingCapacity, AutomaticResolvesToPreset)
    {
        const auto resolved = resolveServingCapacity(ServingCapacityRequest::automatic(), kCapacityOnePreset);
        EXPECT_TRUE(resolved.isCapacityOnePreset());
        EXPECT_EQ(resolved.residentSlots(), 1);
        EXPECT_EQ(resolved.responseLeases(), 1);
        EXPECT_EQ(resolved.queueCapacity(), 0);
    }

    TEST(ServingCapacity, AuthoredExactValuesArePreserved)
    {
        const auto request = ServingCapacityRequest::authored(1, 1, 0);
        EXPECT_FALSE(request.isAutomatic());
        EXPECT_EQ(request.residentSlots(), 1);
        EXPECT_EQ(request.responseLeases(), 1);
        EXPECT_EQ(request.queueCapacity(), 0);

        const auto resolved = resolveServingCapacity(request, kCapacityOnePreset);
        EXPECT_EQ(resolved.residentSlots(), 1);
        EXPECT_EQ(resolved.responseLeases(), 1);
        EXPECT_EQ(resolved.queueCapacity(), 0);
        EXPECT_TRUE(resolved.isCapacityOnePreset());
    }

    TEST(ServingCapacity, HigherResidentSlotsAreRejectedNotClamped)
    {
        const auto request = ServingCapacityRequest::authored(2, 4, 1);
        EXPECT_THROW(resolveServingCapacity(request, kCapacityOnePreset), ServingCapacityUnqualified);
    }

    TEST(ServingCapacity, HigherQueueCapacityIsRejectedNotClamped)
    {
        const auto request = ServingCapacityRequest::authored(1, 4, 1);
        EXPECT_THROW(resolveServingCapacity(request, kCapacityOnePreset), ServingCapacityUnqualified);
    }

    TEST(ServingCapacity, HigherResponseLeasesAreRejectedNotClamped)
    {
        const auto request = ServingCapacityRequest::authored(1, 2, 0);
        EXPECT_THROW(resolveServingCapacity(request, kCapacityOnePreset), ServingCapacityUnqualified);
    }

    TEST(ServingCapacity, AuthoredRequiresPositiveResidentsAndResponses)
    {
        EXPECT_THROW(ServingCapacityRequest::authored(0, 1, 0), std::invalid_argument);
        EXPECT_THROW(ServingCapacityRequest::authored(-1, 1, 0), std::invalid_argument);
        EXPECT_THROW(ServingCapacityRequest::authored(1, 0, 0), std::invalid_argument);
        EXPECT_THROW(ServingCapacityRequest::authored(1, -1, 0), std::invalid_argument);
    }

    TEST(ServingCapacity, AuthoredRequiresNonnegativeQueue)
    {
        EXPECT_THROW(ServingCapacityRequest::authored(1, 1, -1), std::invalid_argument);
    }

    TEST(ServingCapacity, ResolutionRejectsResponseLeasesBelowResidentsPlusQueue)
    {
        // R == S + Q is the exact invariant boundary and is accepted.
        const auto exact = resolveServingCapacity(ServingCapacityRequest::authored(1, 1, 0), kCapacityOnePreset);
        EXPECT_EQ(exact.responseLeases(), 1);

        // R < S + Q is structurally invalid and rejected at resolution time even
        // though every per-axis value is individually nonnegative.
        const auto request = ServingCapacityRequest::authored(1, 1, 1);
        EXPECT_THROW(resolveServingCapacity(request, kCapacityOnePreset), std::invalid_argument);
    }

    TEST(ServingCapacity, AdmissionChecksQueueAxisFirst)
    {
        const auto resolved = resolveServingCapacity(ServingCapacityRequest::automatic(), kCapacityOnePreset);
        // Q = 0: no preparing-or-queued request is admitted.
        EXPECT_FALSE(resolved.fitsPreparingOrQueued(1));
        EXPECT_FALSE(resolved.admits(1, 0, 0));
        EXPECT_FALSE(resolved.admits(1, 1, 0));
    }

    TEST(ServingCapacity, AdmissionChecksResidentAxis)
    {
        const auto resolved = resolveServingCapacity(ServingCapacityRequest::automatic(), kCapacityOnePreset);
        // S = 1: at most one execution-attached request is admitted.
        EXPECT_TRUE(resolved.fitsExecutionAttached(1));
        EXPECT_FALSE(resolved.fitsExecutionAttached(2));
        EXPECT_FALSE(resolved.admits(0, 2, 0));
    }

    TEST(ServingCapacity, AdmissionChecksResponseTotalAxis)
    {
        const auto resolved = resolveServingCapacity(ServingCapacityRequest::automatic(), kCapacityOnePreset);
        // R = 1: one detached delivery may coexist with nothing else.
        EXPECT_TRUE(resolved.fitsResponseLeaseTotal(1));
        EXPECT_FALSE(resolved.fitsResponseLeaseTotal(2));
        EXPECT_TRUE(resolved.admits(0, 0, 1));
        EXPECT_TRUE(resolved.admits(0, 1, 0)); // one attached request inside R = 1
        EXPECT_FALSE(resolved.admits(0, 1, 1)); // total 2 exceeds R
        EXPECT_FALSE(resolved.admits(1, 0, 1)); // queue axis rejects first
    }

    TEST(ServingCapacity, AdmissionBoundRejectsNegativeCounters)
    {
        const auto resolved = resolveServingCapacity(ServingCapacityRequest::automatic(), kCapacityOnePreset);
        EXPECT_THROW(resolved.admits(-1, 0, 0), std::invalid_argument);
        EXPECT_THROW(resolved.admits(0, -1, 0), std::invalid_argument);
        EXPECT_THROW(resolved.admits(0, 0, -1), std::invalid_argument);
        EXPECT_THROW(resolved.fitsPreparingOrQueued(-1), std::invalid_argument);
        EXPECT_THROW(resolved.fitsExecutionAttached(-1), std::invalid_argument);
        EXPECT_THROW(resolved.fitsResponseLeaseTotal(-1), std::invalid_argument);
    }

    TEST(ServingCapacity, WideningArithmeticDoesNotOverflowAtIntMax)
    {
        const auto resolved = resolveServingCapacity(ServingCapacityRequest::automatic(), kCapacityOnePreset);
        constexpr int kIntMax = std::numeric_limits<int>::max();
        // Total is computed in int64 before comparison; no wrap occurs.
        EXPECT_FALSE(resolved.fitsResponseLeaseTotal(kIntMax));
        EXPECT_FALSE(resolved.admits(kIntMax, 0, 0));
        EXPECT_FALSE(resolved.admits(0, kIntMax, 0));

        // Cross-axis validation uses int64: R < S + Q must not wrap.
        EXPECT_THROW(resolveServingCapacity(ServingCapacityRequest::authored(1, 1, kIntMax), kCapacityOnePreset),
                     std::invalid_argument);
        EXPECT_THROW(resolveServingCapacity(ServingCapacityRequest::authored(kIntMax, kIntMax, 0), kCapacityOnePreset),
                     ServingCapacityUnqualified); // exceeds envelope, no wrap
    }

    TEST(ServingCapacity, CapacityOnePresetIsSelfConsistent)
    {
        EXPECT_TRUE(kCapacityOnePreset.isSelfConsistent());
        EXPECT_EQ(kCapacityOnePreset.maxResidentSlots(), 1);
        EXPECT_EQ(kCapacityOnePreset.maxResponseLeases(), 1);
        EXPECT_EQ(kCapacityOnePreset.maxQueueCapacity(), 0);
    }

    TEST(ServingCapacity, ValuesAreValueIdentity)
    {
        EXPECT_EQ(resolveServingCapacity(ServingCapacityRequest::automatic(), kCapacityOnePreset),
                  resolveServingCapacity(ServingCapacityRequest::authored(1, 1, 0), kCapacityOnePreset));
        EXPECT_EQ(ServingCapacityRequest::authored(1, 1, 0), ServingCapacityRequest::authored(1, 1, 0));
        EXPECT_NE(ServingCapacityRequest::authored(1, 1, 0), ServingCapacityRequest::authored(1, 2, 0));
    }
} // namespace llaminar2::test
