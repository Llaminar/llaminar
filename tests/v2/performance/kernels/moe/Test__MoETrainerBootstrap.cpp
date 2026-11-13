/**
 * @file Test__MoETrainerBootstrap.cpp
 * @brief Prove the MoE trainer's real executable initializes its MPI owner.
 *
 * Descriptor-derived expert synthesis borrows the canonical physical CPU
 * inventory. That inventory belongs to the original MPI context, so a plain
 * GTest entrypoint is insufficient even for one rank. This focused regression
 * checks initialization before calling that authority, avoiding an MPI abort
 * if the executable is accidentally linked to the wrong entrypoint again.
 * Preflight selects CPU-only startup; no model or accelerator is required.
 */
#include "backends/ComputeBackend.h"
#include "backends/HardwareInventory.h"
#include "utils/MPIContext.h"

#include <gtest/gtest.h>
#include <mpi.h>

/** @brief Require the production thread contract before borrowing rank inventory. */
TEST(MoETrainerBootstrap, CanonicalInventoryStartsAfterMPIInitialization)
{
    int initialized = 0;
    ASSERT_EQ(MPI_Initialized(&initialized), MPI_SUCCESS);
    ASSERT_EQ(initialized, 1) << "MoE trainer requires the canonical MPI GTest entrypoint";
    int provided = MPI_THREAD_SINGLE;
    ASSERT_EQ(MPI_Query_thread(&provided), MPI_SUCCESS);
    ASSERT_GE(provided, MPI_THREAD_MULTIPLE);

    const auto context = llaminar2::MPIContextFactory::global();
    ASSERT_NE(context, nullptr);
    const auto inventory = context->clusterInventory();
    ASSERT_NE(inventory, nullptr);
    EXPECT_EQ(context->clusterInventory().get(), inventory.get());
    const auto *hardware = llaminar2::DeviceManager::instance().hardware();
    ASSERT_NE(hardware, nullptr);
    EXPECT_GT(hardware->total_physical_cores(), 0);
}
