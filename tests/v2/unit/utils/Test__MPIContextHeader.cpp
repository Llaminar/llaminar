/**
 * @file Test__MPIContextHeader.cpp
 * @brief Standalone public MPI context ownership-header regression, without MPI initialization.
 *
 * Including the real production header first reproduces premature destruction
 * of a forward-declared unique-owned topology on libc++. Constructor exception
 * cleanup must be compiled only after the topology is complete. This test uses
 * the actual MPI SDK declarations, not a stub; it neither constructs a context
 * nor initializes MPI, inventories devices or certifies runtime retirement.
 */
#include "utils/MPIContext.h"
#include <gtest/gtest.h>
#include <type_traits>

static_assert(std::is_base_of_v<llaminar2::IMPIContext, llaminar2::MPIContext>);
static_assert(std::is_constructible_v<llaminar2::MPIContext, int, int>);
static_assert(std::is_constructible_v<llaminar2::MPIContext, int, int, MPI_Comm>);
static_assert(std::is_constructible_v<llaminar2::MPIContext, int, int, int, MPI_Comm>);
static_assert(std::is_destructible_v<llaminar2::MPIContext>);

/** @test Real public ownership declarations are complete without relying on caller include order. */
TEST(MPIContextHeader, BorrowedConstructionOwnsCompleteTypeCleanup)
{
    SUCCEED(); // The focused defect is a translation-unit failure, not an MPI protocol test.
}
