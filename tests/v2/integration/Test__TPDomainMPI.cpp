/**
 * @file Test__TPDomainMPI.cpp
 * @brief Multi-rank MPI integration tests for TPDomain
 *
 * Tests TPDomain and MultiDomainTPConfig with real MPI communication:
 *   - MPI_Comm_split for creating domain communicators
 *   - Cross-rank domain creation and validation
 *   - Communicator cleanup
 *
 * Run with: mpirun -np 2 ./v2_integration_tpdomain_mpi
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <set>

#include "config/TPDomain.h"
#include "utils/NodeTopology.h"
#include "utils/Logger.h"
#include "backends/DeviceId.h"

using namespace llaminar2;

/**
 * @brief Test fixture for TPDomain MPI integration tests
 */
class Test__TPDomainMPI : public ::testing::Test
{
protected:
    int rank_ = 0;
    int world_size_ = 1;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    }

    void TearDown() override
    {
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

// =============================================================================
// TPDomainBuilder Tests
// =============================================================================

TEST_F(Test__TPDomainMPI, CreateGPUIntraRankDomain)
{
    TPDomainBuilder builder(MPI_COMM_WORLD);

    // Create GPU domain with simulated devices
    std::vector<DeviceId> gpus = {DeviceId::cuda(0), DeviceId::rocm(0)};
    std::string name = "gpu_tp_rank" + std::to_string(rank_);

    auto domain = builder.createGPUIntraRankDomain(gpus, name);

    // Verify domain properties
    EXPECT_EQ(domain.type, TPDomainType::GPU_INTRA_RANK);
    EXPECT_EQ(domain.communicator, MPI_COMM_NULL); // No comm needed for intra-rank
    EXPECT_EQ(domain.domain_size, 2);
    EXPECT_EQ(domain.local_rank_in_domain, 0); // Always 0 for intra-rank
    EXPECT_EQ(domain.devices.size(), 2);
    EXPECT_EQ(domain.name, name);
    EXPECT_FALSE(domain.isCrossRank());
    EXPECT_FALSE(domain.isTrivial());

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(Test__TPDomainMPI, CreateCPUCrossRankDomain)
{
    TPDomainBuilder builder(MPI_COMM_WORLD);

    // All ranks participate (same color)
    int color = 0;
    int key = rank_; // Order by world rank

    auto domain = builder.createCPUCrossRankDomain(color, key, "cpu_tp_cross");

    // Verify domain properties
    EXPECT_EQ(domain.type, TPDomainType::CPU_CROSS_RANK);
    EXPECT_NE(domain.communicator, MPI_COMM_NULL); // Should have valid communicator
    EXPECT_EQ(domain.domain_size, world_size_);
    EXPECT_EQ(domain.local_rank_in_domain, rank_); // Ordered by world rank
    EXPECT_FALSE(domain.devices.empty());
    EXPECT_TRUE(domain.isCrossRank());

    // Domain is trivial only if world_size == 1
    if (world_size_ > 1)
    {
        EXPECT_FALSE(domain.isTrivial());
    }
    else
    {
        EXPECT_TRUE(domain.isTrivial());
    }

    // Cleanup communicator
    if (domain.communicator != MPI_COMM_NULL &&
        domain.communicator != MPI_COMM_WORLD)
    {
        MPI_Comm_free(&domain.communicator);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(Test__TPDomainMPI, SplitCommunicatorAllParticipate)
{
    TPDomainBuilder builder(MPI_COMM_WORLD);

    // All ranks participate
    MPI_Comm new_comm = builder.splitCommunicator(true, "all_participate");

    ASSERT_NE(new_comm, MPI_COMM_NULL);

    // Verify size matches
    int new_size;
    MPI_Comm_size(new_comm, &new_size);
    EXPECT_EQ(new_size, world_size_);

    // Verify rank ordering preserved
    int new_rank;
    MPI_Comm_rank(new_comm, &new_rank);
    EXPECT_EQ(new_rank, rank_);

    // Cleanup
    MPI_Comm_free(&new_comm);

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(Test__TPDomainMPI, SplitCommunicatorPartialParticipate)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 ranks for partial participation test";
    }

    TPDomainBuilder builder(MPI_COMM_WORLD);

    // Only even ranks participate
    bool participating = (rank_ % 2 == 0);
    MPI_Comm new_comm = builder.splitCommunicator(participating, "even_ranks");

    if (participating)
    {
        ASSERT_NE(new_comm, MPI_COMM_NULL);

        // Verify size is half (rounded up)
        int new_size;
        MPI_Comm_size(new_comm, &new_size);
        int expected_size = (world_size_ + 1) / 2; // Ceiling division
        EXPECT_EQ(new_size, expected_size);

        MPI_Comm_free(&new_comm);
    }
    else
    {
        // Non-participants get MPI_COMM_NULL
        EXPECT_EQ(new_comm, MPI_COMM_NULL);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// MultiDomainTPConfig::create Tests
// =============================================================================

TEST_F(Test__TPDomainMPI, MultiDomainConfigCreate)
{
    // Detect or mock topology
    auto topology = NodeTopology::createForTest(2, 1, 4); // 2 sockets, 1 NUMA/socket, 4 cores/NUMA

    // Simulate local devices (CPU + one GPU per rank)
    std::vector<DeviceId> local_devices;
    local_devices.push_back(DeviceId::cpu());
    if (rank_ == 0)
    {
        local_devices.push_back(DeviceId::cuda(0));
    }
    else
    {
        local_devices.push_back(DeviceId::rocm(0));
    }

    auto config = MultiDomainTPConfig::create(topology, MPI_COMM_WORLD, local_devices);

    // Should have at least one domain
    EXPECT_FALSE(config.domains().empty());

    // GPU domain should exist (intra-rank)
    if (local_devices.size() > 1)
    { // Has GPU
        EXPECT_NE(config.gpuDomain(), nullptr);
    }

    // With world_size > 1 or multi-socket, should have CPU domain
    if (world_size_ > 1)
    {
        EXPECT_NE(config.cpuDomain(), nullptr);
        EXPECT_TRUE(config.cpuDomain()->isCrossRank());
    }

    // String representation should not be empty
    EXPECT_FALSE(config.toString().empty());

    // Cleanup happens in destructor
    config.cleanup();

    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Communicator Cleanup Tests
// =============================================================================

TEST_F(Test__TPDomainMPI, CommunicatorCleanup)
{
    TPDomainBuilder builder(MPI_COMM_WORLD);

    // Create multiple communicators
    auto domain1 = builder.createCPUCrossRankDomain(0, rank_, "domain1");
    auto domain2 = builder.createCPUCrossRankDomain(1, rank_, "domain2");

    // Track that we have created comms
    EXPECT_EQ(builder.createdCommunicators().size(), 2);

    // Manual cleanup
    for (auto &comm : builder.createdCommunicators())
    {
        if (comm != MPI_COMM_NULL && comm != MPI_COMM_WORLD)
        {
            // Don't actually free here - just verify tracking works
        }
    }

    // Free the communicators we created in the domains
    if (domain1.communicator != MPI_COMM_NULL &&
        domain1.communicator != MPI_COMM_WORLD)
    {
        MPI_Comm_free(&domain1.communicator);
    }
    if (domain2.communicator != MPI_COMM_NULL &&
        domain2.communicator != MPI_COMM_WORLD)
    {
        MPI_Comm_free(&domain2.communicator);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Cross-Rank Consistency Tests
// =============================================================================

TEST_F(Test__TPDomainMPI, AllRanksSeeSameDomainSize)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 ranks for cross-rank consistency test";
    }

    TPDomainBuilder builder(MPI_COMM_WORLD);
    auto domain = builder.createCPUCrossRankDomain(0, rank_, "consistency_test");

    // All ranks should see same domain_size
    int local_size = domain.domain_size;
    std::vector<int> all_sizes(world_size_);

    MPI_Allgather(&local_size, 1, MPI_INT,
                  all_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    for (int i = 0; i < world_size_; ++i)
    {
        EXPECT_EQ(all_sizes[i], world_size_)
            << "Rank " << i << " has inconsistent domain_size";
    }

    // Cleanup
    if (domain.communicator != MPI_COMM_NULL &&
        domain.communicator != MPI_COMM_WORLD)
    {
        MPI_Comm_free(&domain.communicator);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(Test__TPDomainMPI, LocalRanksAreUnique)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 ranks for unique rank test";
    }

    TPDomainBuilder builder(MPI_COMM_WORLD);
    auto domain = builder.createCPUCrossRankDomain(0, rank_, "unique_rank_test");

    // All ranks should have unique local_rank_in_domain
    int local_rank = domain.local_rank_in_domain;
    std::vector<int> all_ranks(world_size_);

    MPI_Allgather(&local_rank, 1, MPI_INT,
                  all_ranks.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::set<int> unique_ranks(all_ranks.begin(), all_ranks.end());
    EXPECT_EQ(unique_ranks.size(), static_cast<size_t>(world_size_));

    // Verify range is [0, world_size)
    for (int r : all_ranks)
    {
        EXPECT_GE(r, 0);
        EXPECT_LT(r, world_size_);
    }

    // Cleanup
    if (domain.communicator != MPI_COMM_NULL &&
        domain.communicator != MPI_COMM_WORLD)
    {
        MPI_Comm_free(&domain.communicator);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(Test__TPDomainMPI, EmptyDeviceList)
{
    TPDomainBuilder builder(MPI_COMM_WORLD);

    // Empty GPU list
    std::vector<DeviceId> empty_gpus;
    auto domain = builder.createGPUIntraRankDomain(empty_gpus, "empty_gpu");

    EXPECT_EQ(domain.domain_size, 0);
    EXPECT_TRUE(domain.devices.empty());
    EXPECT_TRUE(domain.isTrivial());
    EXPECT_FALSE(domain.isValid());

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(Test__TPDomainMPI, NonParticipatingCPUDomain)
{
    TPDomainBuilder builder(MPI_COMM_WORLD);

    // Use MPI_UNDEFINED for non-participating
    int color = MPI_UNDEFINED;
    auto domain = builder.createCPUCrossRankDomain(color, 0, "non_participating");

    // Should get empty/invalid domain
    EXPECT_EQ(domain.communicator, MPI_COMM_NULL);
    EXPECT_EQ(domain.domain_size, 0);
    EXPECT_TRUE(domain.isTrivial());

    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Main with MPI initialization
// =============================================================================

int main(int argc, char **argv)
{
    // Initialize MPI before GTest
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    // Initialize GTest
    ::testing::InitGoogleTest(&argc, argv);

    // Run tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
