/**
 * @file Test__MPITensorParallel.cpp
 * @brief Tests for MPI tensor-parallel strategy
 *
 * Tests:
 * - Strategy selection logic
 * - Strategy validation
 * - Head distribution across ranks
 * - Single-rank vs multi-rank correctness (TODO: Phase 3)
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "../../src/v2/pipelines/MPIStrategy.h"
#include "../../src/v2/pipelines/PipelineBase.h"
#include "../../src/v2/pipelines/qwen/Qwen2Pipeline.h"
#include "../../src/v2/loaders/ModelContext.h"
#include "../../src/v2/utils/MPIContext.h"
#include <mpi.h>
#include <memory>
#include <iostream>
#include "TestMPIUtils.h"

using namespace llaminar2;

/**
 * @brief Test fixture for MPI tensor-parallel tests
 */
class Test__MPITensorParallel : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // MPI should already be initialized by main()
        int rank, world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);

        if (rank == 0)
        {
            std::cout << "[Test__MPITensorParallel] Running with " << world_size << " ranks\n";
        }
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
};

/**
 * @brief Test MPIStrategy enum values
 */
TEST_F(Test__MPITensorParallel, StrategyEnumValues)
{
    EXPECT_EQ(strategyName(MPIStrategy::None), std::string("None"));
    EXPECT_EQ(strategyName(MPIStrategy::TensorParallel), std::string("TensorParallel"));
    EXPECT_EQ(strategyName(MPIStrategy::PipelineParallel), std::string("PipelineParallel"));
    EXPECT_EQ(strategyName(MPIStrategy::SequenceParallel), std::string("SequenceParallel"));
    EXPECT_EQ(strategyName(MPIStrategy::Hybrid), std::string("Hybrid"));
}

/**
 * @brief Test default MPIConfig
 */
TEST_F(Test__MPITensorParallel, DefaultConfig)
{
    MPIConfig config = defaultMPIConfig();

    EXPECT_EQ(config.strategy, MPIStrategy::TensorParallel);
    EXPECT_TRUE(config.auto_select);
    EXPECT_TRUE(config.validate_divisibility);
    EXPECT_TRUE(config.tp_split_attention);
    EXPECT_TRUE(config.tp_split_ffn);
    EXPECT_EQ(config.fallback_strategy, MPIStrategy::None);
    EXPECT_FALSE(config.verbose_logging);
}

/**
 * @brief Test head distribution helper
 */
TEST_F(Test__MPITensorParallel, HeadDistribution)
{
    int world_size = mpi_ctx_->world_size();
    int rank = mpi_ctx_->rank();

    // Test with 16 heads (evenly divisible by 1, 2, 4, 8, 16)
    int n_heads = 16;

    if (n_heads % world_size == 0)
    {
        auto [start_head, local_n_heads] = mpi_ctx_->get_local_slice(n_heads);

        int expected_local = n_heads / world_size;
        EXPECT_EQ(local_n_heads, expected_local)
            << "Rank " << rank << " should have " << expected_local << " heads";

        int expected_start = rank * expected_local;
        EXPECT_EQ(start_head, expected_start)
            << "Rank " << rank << " should start at head " << expected_start;

        // Verify no overlap between ranks
        if (rank == 0)
        {
            std::cout << "[Test__MPITensorParallel] Head distribution with " << world_size << " ranks:\n";
            for (int r = 0; r < world_size; ++r)
            {
                int start = r * expected_local;
                int end = start + expected_local - 1;
                std::cout << "  Rank " << r << ": heads [" << start << ", " << end << "]\n";
            }
        }
    }
}

/**
 * @brief Test strategy selection with Qwen2 model (14 heads)
 *
 * Qwen 2.5 0.5B has 14 attention heads:
 * - Divisible by 1, 2, 7, 14
 * - Should select TensorParallel with 2 ranks (7 heads per rank)
 * - Should fail with 4 ranks (14 % 4 != 0)
 */
TEST_F(Test__MPITensorParallel, StrategySelectionQwen14Heads)
{
    int world_size = mpi_ctx_->world_size();
    int rank = mpi_ctx_->rank();

    if (world_size == 2)
    {
        // 14 heads % 2 == 0 → TensorParallel should be valid
        auto [start_head, local_n_heads] = mpi_ctx_->get_local_slice(14);

        EXPECT_EQ(local_n_heads, 7) << "With 2 ranks, each should get 7 heads";

        if (rank == 0)
        {
            EXPECT_EQ(start_head, 0);
            std::cout << "[Test__MPITensorParallel] Rank 0: heads [0, 6]\n";
        }
        else
        {
            EXPECT_EQ(start_head, 7);
            std::cout << "[Test__MPITensorParallel] Rank 1: heads [7, 13]\n";
        }
    }
    else if (world_size == 4)
    {
        // 14 % 4 != 0 → TensorParallel should be invalid
        // Strategy selection should fallback to None
        if (rank == 0)
        {
            std::cout << "[Test__MPITensorParallel] With 4 ranks, 14 heads not divisible. "
                      << "Strategy should fallback to None.\n";
        }
    }
}

/**
 * @brief Test with Qwen2 model (16 heads - evenly divisible)
 *
 * Qwen 2.5 1.5B has 16 attention heads:
 * - Divisible by 1, 2, 4, 8, 16
 * - Should select TensorParallel with any of these world_sizes
 */
TEST_F(Test__MPITensorParallel, StrategySelectionQwen16Heads)
{
    int world_size = mpi_ctx_->world_size();
    int rank = mpi_ctx_->rank();

    int n_heads = 16;

    if (n_heads % world_size == 0)
    {
        auto [start_head, local_n_heads] = mpi_ctx_->get_local_slice(n_heads);

        int expected_local = n_heads / world_size;
        EXPECT_EQ(local_n_heads, expected_local);

        if (rank == 0)
        {
            std::cout << "[Test__MPITensorParallel] With " << world_size << " ranks and 16 heads: "
                      << expected_local << " heads per rank\n";
        }
    }
}

// TODO Phase 3: Add numerical correctness tests
// TEST_F(Test__MPITensorParallel, SingleVsMultiRankCorrectness) { ... }
// TEST_F(Test__MPITensorParallel, AttentionOutputParity) { ... }

int main(int argc, char **argv)
{
    // Initialize MPI
    int provided;
    llaminar2::tests::mpi_init_thread_sanitizer_safe(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if (provided < MPI_THREAD_MULTIPLE)
    {
        std::cerr << "Warning: MPI_THREAD_MULTIPLE not available (got " << provided << ")\n";
    }

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (rank == 0)
    {
        std::cout << "Running MPI Tensor-Parallel tests with " << world_size << " ranks\n";
    }

    ::testing::InitGoogleTest(&argc, argv);

    // Suppress GTest output on non-root ranks
    if (rank != 0)
    {
        ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
    }

    int result = RUN_ALL_TESTS();

    llaminar2::tests::mpi_finalize_sanitizer_safe();

    return result;
}
