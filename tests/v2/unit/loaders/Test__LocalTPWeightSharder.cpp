/**
 * @file Test__LocalTPWeightSharder.cpp
 * @brief Unit tests for LocalTPWeightSharder
 * @author David Sanftenberg
 * @date January 2026
 *
 * Tests for LOCAL TP weight sharding:
 * - Column-parallel sharding (Q, K, V projections)
 * - Row-parallel sharding (output projections)
 * - Equal weight distribution
 * - Proportional weight distribution
 * - Single-device shard extraction
 */

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>

#include "loaders/ILocalTPWeightSharder.h"
#include "collective/LocalTPContext.h"
#include "tensors/TensorFactory.h"
#include "tensors/Tensors.h"
#include "backends/GlobalDeviceAddress.h"
#include "utils/MPIContext.h"

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__LocalTPWeightSharder : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create test devices
        cuda0_ = GlobalDeviceAddress::cuda(0, 0);
        cuda1_ = GlobalDeviceAddress::cuda(1, 0);
        rocm0_ = GlobalDeviceAddress::rocm(0, 0);

        // Create sharder
        sharder_ = createLocalTPWeightSharder();

        // Create MPI context for TensorFactory (single rank for unit tests)
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
    }

    /**
     * @brief Create a test FP32 weight tensor with known values
     * @param rows Number of rows
     * @param cols Number of columns
     * @return FP32Tensor with values row*cols + col (for verification)
     */
    std::unique_ptr<FP32Tensor> createTestWeight(size_t rows, size_t cols)
    {
        std::vector<size_t> shape = {rows, cols};
        auto tensor = std::make_unique<FP32Tensor>(shape);
        float *data = tensor->mutable_data();
        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t c = 0; c < cols; ++c)
            {
                data[r * cols + c] = static_cast<float>(r * cols + c);
            }
        }
        return tensor;
    }

    GlobalDeviceAddress cuda0_;
    GlobalDeviceAddress cuda1_;
    GlobalDeviceAddress rocm0_;
    std::unique_ptr<ILocalTPWeightSharder> sharder_;
    std::shared_ptr<MPIContext> mpi_ctx_;
};

// =============================================================================
// Construction Tests
// =============================================================================

/**
 * @test Factory creates valid sharder
 */
TEST_F(Test__LocalTPWeightSharder, FactoryCreatesValidSharder)
{
    auto sharder = createLocalTPWeightSharder();
    ASSERT_NE(sharder, nullptr);
}

// =============================================================================
// Column-Parallel Sharding Tests
// =============================================================================

/**
 * @test shardColumnParallel creates correct number of shards
 */
TEST_F(Test__LocalTPWeightSharder, ShardColumnParallelCreatesCorrectCount)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);
    auto weight = createTestWeight(64, 128);

    auto shards = sharder_->shardColumnParallel(weight.get(), *ctx);

    EXPECT_EQ(shards.size(), 2);
}

/**
 * @test shardColumnParallel with equal weights splits evenly
 */
TEST_F(Test__LocalTPWeightSharder, ShardColumnParallelEqualWeights)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);
    auto weight = createTestWeight(64, 128);

    auto shards = sharder_->shardColumnParallel(weight.get(), *ctx);

    // Each shard should have 64 rows, 64 columns (half)
    ASSERT_EQ(shards.size(), 2);
    EXPECT_EQ(shards[0]->rows(), 64);
    EXPECT_EQ(shards[0]->cols(), 64);
    EXPECT_EQ(shards[1]->rows(), 64);
    EXPECT_EQ(shards[1]->cols(), 64);
}

/**
 * @test shardColumnParallel with proportional weights
 */
TEST_F(Test__LocalTPWeightSharder, ShardColumnParallelProportional)
{
    // 75%/25% split
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {0.75f, 0.25f}, CollectiveBackendType::NCCL);
    auto weight = createTestWeight(64, 128);

    auto shards = sharder_->shardColumnParallel(weight.get(), *ctx);

    ASSERT_EQ(shards.size(), 2);

    // First shard should get ~96 columns (75% of 128)
    EXPECT_EQ(shards[0]->rows(), 64);
    EXPECT_GE(shards[0]->cols(), 90);
    EXPECT_LE(shards[0]->cols(), 100);

    // Second shard should get ~32 columns (25% of 128)
    EXPECT_EQ(shards[1]->rows(), 64);
    EXPECT_GE(shards[1]->cols(), 28);
    EXPECT_LE(shards[1]->cols(), 38);

    // Total columns should equal original
    EXPECT_EQ(shards[0]->cols() + shards[1]->cols(), 128);
}

/**
 * @test shardColumnParallel preserves data correctly
 */
TEST_F(Test__LocalTPWeightSharder, ShardColumnParallelDataCorrect)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);
    auto weight = createTestWeight(4, 8);

    auto shards = sharder_->shardColumnParallel(weight.get(), *ctx);

    // First shard: columns 0-3
    const float *s0_data = shards[0]->data();
    for (size_t r = 0; r < 4; ++r)
    {
        for (size_t c = 0; c < 4; ++c)
        {
            float expected = static_cast<float>(r * 8 + c);
            EXPECT_FLOAT_EQ(s0_data[r * 4 + c], expected)
                << "Shard 0 mismatch at row=" << r << ", col=" << c;
        }
    }

    // Second shard: columns 4-7
    const float *s1_data = shards[1]->data();
    for (size_t r = 0; r < 4; ++r)
    {
        for (size_t c = 0; c < 4; ++c)
        {
            float expected = static_cast<float>(r * 8 + (c + 4));
            EXPECT_FLOAT_EQ(s1_data[r * 4 + c], expected)
                << "Shard 1 mismatch at row=" << r << ", col=" << c;
        }
    }
}

/**
 * @test shardColumnParallel with three devices
 */
TEST_F(Test__LocalTPWeightSharder, ShardColumnParallelThreeDevices)
{
    auto rocm1 = GlobalDeviceAddress::rocm(1, 0);
    auto ctx = createLocalTPContext({cuda0_, rocm0_, rocm1}, {}, CollectiveBackendType::PCIE_BAR);
    auto weight = createTestWeight(64, 99); // Not evenly divisible by 3

    auto shards = sharder_->shardColumnParallel(weight.get(), *ctx);

    ASSERT_EQ(shards.size(), 3);

    // Total columns should equal original
    size_t total_cols = shards[0]->cols() + shards[1]->cols() + shards[2]->cols();
    EXPECT_EQ(total_cols, 99);

    // Each shard should have ~33 columns (99/3)
    for (const auto &shard : shards)
    {
        EXPECT_EQ(shard->rows(), 64);
        EXPECT_GE(shard->cols(), 31);
        EXPECT_LE(shard->cols(), 35);
    }
}

// =============================================================================
// Row-Parallel Sharding Tests
// =============================================================================

/**
 * @test shardRowParallel creates correct number of shards
 */
TEST_F(Test__LocalTPWeightSharder, ShardRowParallelCreatesCorrectCount)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);
    auto weight = createTestWeight(128, 64);

    auto shards = sharder_->shardRowParallel(weight.get(), *ctx);

    EXPECT_EQ(shards.size(), 2);
}

/**
 * @test shardRowParallel with equal weights splits evenly
 */
TEST_F(Test__LocalTPWeightSharder, ShardRowParallelEqualWeights)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);
    auto weight = createTestWeight(128, 64);

    auto shards = sharder_->shardRowParallel(weight.get(), *ctx);

    // Each shard should have 64 rows (half), 64 columns (full)
    ASSERT_EQ(shards.size(), 2);
    EXPECT_EQ(shards[0]->rows(), 64);
    EXPECT_EQ(shards[0]->cols(), 64);
    EXPECT_EQ(shards[1]->rows(), 64);
    EXPECT_EQ(shards[1]->cols(), 64);
}

/**
 * @test shardRowParallel with proportional weights
 */
TEST_F(Test__LocalTPWeightSharder, ShardRowParallelProportional)
{
    // 60%/40% split
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {0.6f, 0.4f}, CollectiveBackendType::NCCL);
    auto weight = createTestWeight(100, 64);

    auto shards = sharder_->shardRowParallel(weight.get(), *ctx);

    ASSERT_EQ(shards.size(), 2);

    // First shard should get ~60 rows (60% of 100)
    EXPECT_GE(shards[0]->rows(), 56);
    EXPECT_LE(shards[0]->rows(), 64);
    EXPECT_EQ(shards[0]->cols(), 64);

    // Second shard should get ~40 rows (40% of 100)
    EXPECT_GE(shards[1]->rows(), 36);
    EXPECT_LE(shards[1]->rows(), 44);
    EXPECT_EQ(shards[1]->cols(), 64);

    // Total rows should equal original
    EXPECT_EQ(shards[0]->rows() + shards[1]->rows(), 100);
}

/**
 * @test shardRowParallel preserves data correctly
 */
TEST_F(Test__LocalTPWeightSharder, ShardRowParallelDataCorrect)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);
    auto weight = createTestWeight(8, 4);

    auto shards = sharder_->shardRowParallel(weight.get(), *ctx);

    // First shard: rows 0-3
    const float *s0_data = shards[0]->data();
    for (size_t r = 0; r < 4; ++r)
    {
        for (size_t c = 0; c < 4; ++c)
        {
            float expected = static_cast<float>(r * 4 + c);
            EXPECT_FLOAT_EQ(s0_data[r * 4 + c], expected)
                << "Shard 0 mismatch at row=" << r << ", col=" << c;
        }
    }

    // Second shard: rows 4-7
    const float *s1_data = shards[1]->data();
    for (size_t r = 0; r < 4; ++r)
    {
        for (size_t c = 0; c < 4; ++c)
        {
            float expected = static_cast<float>((r + 4) * 4 + c);
            EXPECT_FLOAT_EQ(s1_data[r * 4 + c], expected)
                << "Shard 1 mismatch at row=" << r << ", col=" << c;
        }
    }
}

// =============================================================================
// Single-Device Shard Tests
// =============================================================================

/**
 * @test getColumnShard returns correct shard for specific device
 */
TEST_F(Test__LocalTPWeightSharder, GetColumnShardForDevice)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);
    auto weight = createTestWeight(64, 128);

    auto shard0 = sharder_->getColumnShard(weight.get(), *ctx, cuda0_);
    auto shard1 = sharder_->getColumnShard(weight.get(), *ctx, cuda1_);

    // Shard 0: first 64 columns
    EXPECT_EQ(shard0->cols(), 64);

    // Shard 1: last 64 columns
    EXPECT_EQ(shard1->cols(), 64);
}

/**
 * @test getRowShard returns correct shard for specific device
 */
TEST_F(Test__LocalTPWeightSharder, GetRowShardForDevice)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);
    auto weight = createTestWeight(128, 64);

    auto shard0 = sharder_->getRowShard(weight.get(), *ctx, cuda0_);
    auto shard1 = sharder_->getRowShard(weight.get(), *ctx, cuda1_);

    // Shard 0: first 64 rows
    EXPECT_EQ(shard0->rows(), 64);
    EXPECT_EQ(shard0->cols(), 64);

    // Shard 1: last 64 rows
    EXPECT_EQ(shard1->rows(), 64);
    EXPECT_EQ(shard1->cols(), 64);
}

/**
 * @test getColumnShard throws for unknown device
 */
TEST_F(Test__LocalTPWeightSharder, GetColumnShardUnknownDeviceThrows)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);
    auto weight = createTestWeight(64, 128);

    EXPECT_THROW(
        sharder_->getColumnShard(weight.get(), *ctx, rocm0_),
        std::invalid_argument);
}

/**
 * @test getRowShard throws for null weight
 */
TEST_F(Test__LocalTPWeightSharder, GetRowShardNullWeightThrows)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);

    EXPECT_THROW(
        sharder_->getRowShard(nullptr, *ctx, cuda0_),
        std::invalid_argument);
}

// =============================================================================
// Query Method Tests
// =============================================================================

/**
 * @test columnCountForDevice returns correct count
 */
TEST_F(Test__LocalTPWeightSharder, ColumnCountForDevice)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);

    EXPECT_EQ(sharder_->columnCountForDevice(128, *ctx, cuda0_), 64);
    EXPECT_EQ(sharder_->columnCountForDevice(128, *ctx, cuda1_), 64);
    EXPECT_EQ(sharder_->columnCountForDevice(100, *ctx, cuda0_), 50);
}

/**
 * @test rowCountForDevice returns correct count
 */
TEST_F(Test__LocalTPWeightSharder, RowCountForDevice)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {0.7f, 0.3f}, CollectiveBackendType::NCCL);

    int r0 = sharder_->rowCountForDevice(100, *ctx, cuda0_);
    int r1 = sharder_->rowCountForDevice(100, *ctx, cuda1_);

    EXPECT_EQ(r0 + r1, 100);
    EXPECT_GE(r0, 65); // ~70%
    EXPECT_LE(r0, 75);
    EXPECT_GE(r1, 25); // ~30%
    EXPECT_LE(r1, 35);
}
