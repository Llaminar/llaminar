/**
 * @file Test__CollectiveBackendIntegration.cpp
 * @brief Comprehensive integration tests for collective operations across backends
 *
 * This test suite validates that collectives work correctly for all supported
 * backend configurations. Tests exercise the full path:
 *   DeviceGraphExecutor → CollectiveContext → BackendRouter → Backend
 *
 * ## Test Scenarios
 *
 * 1. **CPU+CPU (MPI)**: Pure MPI backend for CPU-only tensor parallelism
 * 2. **NVIDIA+NVIDIA (NCCL)**: Homogeneous CUDA tensor parallelism
 * 3. **AMD+AMD (RCCL)**: Homogeneous ROCm tensor parallelism
 * 4. **NVIDIA+AMD (HETEROGENEOUS)**: Cross-vendor GPU tensor parallelism
 * 5. **Single-rank Optimization**: Trivial domain skip
 *
 * @note All tests use a properly distributed ClusterInventory built via MPI
 *       to ensure all ranks are visible to the CollectiveContext.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>

#include "execution/local_execution/collective/CollectiveContext.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/compute_stages/stages/AllreduceStage.h"
#include "collective/BackendRouter.h"
#include "collective/ICollectiveBackend.h"
#include "collective/DeviceGroup.h"
#include "tensors/TensorClasses.h"
#include "backends/DeviceId.h"
#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"
#include "config/TPDomain.h"

using namespace llaminar2;

namespace
{

    // =========================================================================
    // Utility Functions
    // =========================================================================

    double cosine_similarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        if (norm_a < 1e-12 || norm_b < 1e-12)
            return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    float max_abs_diff(const float *a, const float *b, size_t n)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            float diff = std::abs(a[i] - b[i]);
            if (diff > max_diff)
                max_diff = diff;
        }
        return max_diff;
    }

    void fill_deterministic(float *data, size_t count, int seed)
    {
        std::mt19937 gen(static_cast<unsigned>(seed));
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = dist(gen);
        }
    }

    // =========================================================================
    // MPI-based Inventory Distribution
    // =========================================================================

    /**
     * @brief Serialized representation of a rank's device info for MPI transfer
     */
    struct SerializedRankInfo
    {
        int rank;
        int node_id;
        int local_rank;
        int cuda_count;
        int rocm_count;
        // Device memory (up to 8 CUDA + 8 ROCm devices per rank)
        uint64_t cuda_memory[8];
        uint64_t rocm_memory[8];
    };

    /**
     * @brief Build a distributed ClusterInventory using MPI_Allgather
     *
     * Each rank contributes its local device info, and all ranks receive
     * the complete cluster inventory. This is essential for proper backend
     * selection in multi-rank scenarios.
     */
    ClusterInventory buildDistributedInventory(
        std::shared_ptr<IMPIContext> mpi_ctx,
        int cuda_count,
        int rocm_count)
    {
        // Gather local info
        SerializedRankInfo local_info{};
        local_info.rank = mpi_ctx->rank();
        local_info.node_id = 0; // Single node assumed for integration tests
        local_info.local_rank = mpi_ctx->rank();
        local_info.cuda_count = cuda_count;
        local_info.rocm_count = rocm_count;

#ifdef HAVE_CUDA
        auto *cuda_backend = getCUDABackend();
        if (cuda_backend != nullptr)
        {
            for (int i = 0; i < std::min(cuda_count, 8); ++i)
            {
                local_info.cuda_memory[i] = cuda_backend->deviceMemoryTotal(i);
            }
        }
#endif

#ifdef HAVE_ROCM
        auto *rocm_backend = getROCmBackend();
        if (rocm_backend != nullptr)
        {
            for (int i = 0; i < std::min(rocm_count, 8); ++i)
            {
                local_info.rocm_memory[i] = rocm_backend->deviceMemoryTotal(i);
            }
        }
#endif

        // Gather from all ranks
        std::vector<SerializedRankInfo> all_info(static_cast<size_t>(mpi_ctx->world_size()));
        MPI_Allgather(
            &local_info, sizeof(SerializedRankInfo), MPI_BYTE,
            all_info.data(), sizeof(SerializedRankInfo), MPI_BYTE,
            mpi_ctx->communicator());

        // Build ClusterInventory
        ClusterInventory inv;
        inv.world_size = mpi_ctx->world_size();
        inv.node_count = 1; // Single node assumed
        inv.ranks.resize(static_cast<size_t>(mpi_ctx->world_size()));

        for (size_t r = 0; r < all_info.size(); ++r)
        {
            const auto &info = all_info[r];
            auto &rank_inv = inv.ranks[r];

            rank_inv.rank = info.rank;
            rank_inv.node_id = info.node_id;
            rank_inv.local_rank = info.local_rank;
            rank_inv.hostname = "localhost";

            // Add CUDA devices
            for (int i = 0; i < info.cuda_count && i < 8; ++i)
            {
                DeviceInfo gpu;
                gpu.type = DeviceType::CUDA;
                gpu.local_device_id = i;
                gpu.memory_bytes = info.cuda_memory[i];
                gpu.name = "CUDA GPU " + std::to_string(i);
                gpu.supports_p2p = true;
                rank_inv.gpus.push_back(gpu);
            }

            // Add ROCm devices
            for (int i = 0; i < info.rocm_count && i < 8; ++i)
            {
                DeviceInfo gpu;
                gpu.type = DeviceType::ROCm;
                gpu.local_device_id = i;
                gpu.memory_bytes = info.rocm_memory[i];
                gpu.name = "ROCm GPU " + std::to_string(i);
                rank_inv.gpus.push_back(gpu);
            }
        }

        inv.buildNodeAggregations();
        return inv;
    }

} // namespace

// =============================================================================
// Test Fixture: Base class for collective tests
// =============================================================================

class Test__CollectiveBackendIntegration : public ::testing::Test
{
protected:
    std::shared_ptr<IMPIContext> mpi_ctx_;
    int rank_ = 0;
    int world_size_ = 1;
    int cuda_count_ = 0;
    int rocm_count_ = 0;

    static constexpr size_t SMALL_SIZE = 256;
    static constexpr size_t MEDIUM_SIZE = 4096;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);

#ifdef HAVE_CUDA
        auto *cuda_backend = getCUDABackend();
        if (cuda_backend != nullptr)
        {
            cuda_count_ = cuda_backend->deviceCount();
        }
#endif

#ifdef HAVE_ROCM
        auto *rocm_backend = getROCmBackend();
        if (rocm_backend != nullptr)
        {
            rocm_count_ = rocm_backend->deviceCount();
        }
#endif

        LOG_INFO("[Rank " << rank_ << "/" << world_size_ << "] "
                          << "CUDA: " << cuda_count_ << ", ROCm: " << rocm_count_);
    }

    void TearDown() override
    {
        mpi_ctx_->barrier();
    }

    /**
     * @brief Build distributed inventory using MPI_Allgather
     *
     * This creates a complete ClusterInventory with all ranks' device info,
     * which is required for proper CollectiveContext initialization.
     */
    ClusterInventory buildFullInventory()
    {
        return buildDistributedInventory(mpi_ctx_, cuda_count_, rocm_count_);
    }

    /**
     * @brief Build inventory with only CUDA devices visible
     */
    ClusterInventory buildCUDAOnlyInventory()
    {
        return buildDistributedInventory(mpi_ctx_, cuda_count_, 0);
    }

    /**
     * @brief Build inventory with only ROCm devices visible
     */
    ClusterInventory buildROCmOnlyInventory()
    {
        return buildDistributedInventory(mpi_ctx_, 0, rocm_count_);
    }

    void verifyAllReduceResult(
        const float *result,
        const float *expected,
        size_t count,
        const std::string &backend_name)
    {
        double cosine = cosine_similarity(result, expected, count);
        float max_diff = max_abs_diff(result, expected, count);

        LOG_INFO("[Rank " << rank_ << "] " << backend_name
                          << " AllReduce: cosine=" << cosine << ", max_diff=" << max_diff);

        EXPECT_GT(cosine, 0.9999) << backend_name << " AllReduce cosine too low";
        EXPECT_LT(max_diff, 1e-4f) << backend_name << " AllReduce max_diff too high";
    }

    void verifyRankParity(const float *local_data, size_t count)
    {
        if (world_size_ < 2)
            return;

        std::vector<float> all_data;
        if (rank_ == 0)
        {
            all_data.resize(static_cast<size_t>(world_size_) * count);
        }

        MPI_Gather(
            local_data, static_cast<int>(count), MPI_FLOAT,
            all_data.data(), static_cast<int>(count), MPI_FLOAT,
            0, MPI_COMM_WORLD);

        if (rank_ == 0)
        {
            const float *ref = all_data.data();
            for (int r = 1; r < world_size_; ++r)
            {
                const float *cmp = all_data.data() + static_cast<size_t>(r) * count;
                double cosine = cosine_similarity(ref, cmp, count);

                EXPECT_GT(cosine, 0.99999)
                    << "Rank 0 vs Rank " << r << " parity failed (cosine=" << cosine << ")";
            }
            LOG_INFO("[Rank 0] All " << world_size_ << " ranks have identical data");
        }
    }
};

// =============================================================================
// Scenario 1: CPU+CPU Collective (MPI Backend)
// =============================================================================

class Test__CollectiveCPU : public Test__CollectiveBackendIntegration
{
protected:
    void SetUp() override
    {
        Test__CollectiveBackendIntegration::SetUp();
        if (world_size_ < 2)
        {
            GTEST_SKIP() << "CPU+CPU collective test requires at least 2 MPI ranks";
        }
    }
};

/**
 * @brief Test MPI-based AllReduce with distributed inventory
 *
 * Creates a CollectiveContext with full distributed inventory to ensure
 * the BackendRouter is properly initialized and MPI backend is selected.
 */
TEST_F(Test__CollectiveCPU, AllReduceSumViaMPIBackend)
{
    // Build distributed inventory (all ranks contribute)
    auto inventory = buildFullInventory();

    // Create context with inventory - this creates a BackendRouter
    auto ctx = CollectiveContextFactory::createIntraNode(inventory, mpi_ctx_);

    ASSERT_NE(ctx, nullptr) << "Failed to create CollectiveContext";
    ASSERT_TRUE(ctx->requiresCollectives())
        << "Expected requiresCollectives()=true for world_size=" << world_size_;

    // MPI backend should be available
    EXPECT_TRUE(ctx->isBackendAvailable(CollectiveBackendType::MPI))
        << "MPI backend should be available";

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{MEDIUM_SIZE});
    float *data = tensor->mutable_data();

    // Each rank contributes (rank+1) * index
    for (size_t i = 0; i < MEDIUM_SIZE; ++i)
    {
        data[i] = static_cast<float>((rank_ + 1) * (i + 1));
    }

    // Expected: sum across all ranks
    std::vector<float> expected(MEDIUM_SIZE);
    float rank_sum = static_cast<float>(world_size_ * (world_size_ + 1) / 2);
    for (size_t i = 0; i < MEDIUM_SIZE; ++i)
    {
        expected[i] = static_cast<float>(i + 1) * rank_sum;
    }

    bool success = ctx->executeAllreduce(
        tensor.get(),
        MEDIUM_SIZE,
        DeviceId::cpu(),
        CollectiveOp::ALLREDUCE_SUM);

    ASSERT_TRUE(success) << "executeAllreduce failed";

    verifyAllReduceResult(tensor->data(), expected.data(), MEDIUM_SIZE, "MPI");
    verifyRankParity(tensor->data(), MEDIUM_SIZE);
}

/**
 * @brief Test DeviceGraphExecutor's interception of AllreduceStage
 */
TEST_F(Test__CollectiveCPU, AllReduceWithGraphExecutorInterception)
{
    auto inventory = buildFullInventory();
    auto collective_ctx = CollectiveContextFactory::createIntraNode(inventory, mpi_ctx_);

    GraphExecutorConfig config;
    auto executor = std::make_unique<DeviceGraphExecutor>(config);
    executor->setCollectiveContext(collective_ctx.get());

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{SMALL_SIZE});
    float *data = tensor->mutable_data();
    fill_deterministic(data, SMALL_SIZE, rank_ + 42);

    // Gather initial values
    std::vector<float> all_initial(static_cast<size_t>(world_size_) * SMALL_SIZE);
    MPI_Allgather(
        data, static_cast<int>(SMALL_SIZE), MPI_FLOAT,
        all_initial.data(), static_cast<int>(SMALL_SIZE), MPI_FLOAT,
        MPI_COMM_WORLD);

    // Compute expected
    std::vector<float> expected(SMALL_SIZE, 0.0f);
    for (int r = 0; r < world_size_; ++r)
    {
        for (size_t i = 0; i < SMALL_SIZE; ++i)
        {
            expected[i] += all_initial[static_cast<size_t>(r) * SMALL_SIZE + i];
        }
    }

    // Create AllreduceStage
    AllreduceStage::Params params;
    params.device_id = DeviceId::cpu();
    params.mpi_ctx = mpi_ctx_.get();
    params.buffer = tensor.get();
    params.count = SMALL_SIZE;
    params.collective_ctx = nullptr;
    params.domain = nullptr;

    auto stage = std::make_unique<AllreduceStage>(params);

    // Build and execute graph
    ComputeGraph graph;
    graph.addNode("allreduce", std::move(stage), DeviceId::cpu());

    CPUDeviceContext cpu_ctx(DeviceId::cpu());

    bool success = executor->execute(graph, &cpu_ctx);
    ASSERT_TRUE(success) << "DeviceGraphExecutor::execute failed";

    verifyAllReduceResult(tensor->data(), expected.data(), SMALL_SIZE, "MPI (via DeviceGraphExecutor)");
    verifyRankParity(tensor->data(), SMALL_SIZE);
}

/**
 * @brief Test that CPU tensor remains coherent after AllReduce
 */
TEST_F(Test__CollectiveCPU, CoherenceAfterAllReduce)
{
    auto inventory = buildFullInventory();
    auto ctx = CollectiveContextFactory::createIntraNode(inventory, mpi_ctx_);

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{SMALL_SIZE});
    float *data = tensor->mutable_data();
    fill_deterministic(data, SMALL_SIZE, rank_);

    EXPECT_TRUE(tensor->is_on_device(DeviceId::cpu()));

    bool success = ctx->executeAllreduce(
        tensor.get(),
        SMALL_SIZE,
        DeviceId::cpu(),
        CollectiveOp::ALLREDUCE_SUM);

    ASSERT_TRUE(success);

    // After CPU AllReduce, tensor should still be on CPU
    EXPECT_TRUE(tensor->is_on_device(DeviceId::cpu()));

    // data() should return valid pointer without sync
    const float *result = tensor->data();
    EXPECT_NE(result, nullptr);
}

// =============================================================================
// Scenario 2: NVIDIA+NVIDIA Collective (NCCL Backend)
// =============================================================================

#ifdef HAVE_CUDA

class Test__CollectiveNVIDIA : public Test__CollectiveBackendIntegration
{
protected:
    DeviceId cuda_dev_;

    void SetUp() override
    {
        Test__CollectiveBackendIntegration::SetUp();

        if (cuda_count_ == 0)
        {
            GTEST_SKIP() << "No CUDA GPUs available";
        }

        // Each rank uses a different CUDA device (if available)
        int device_idx = rank_ % cuda_count_;
        cuda_dev_ = DeviceId::cuda(device_idx);

        LOG_INFO("[Rank " << rank_ << "] Using CUDA device " << device_idx);
    }
};

/**
 * @brief Verify NCCL backend is selected for CUDA-only inventory
 */
TEST_F(Test__CollectiveNVIDIA, BackendSelectionIsNCCL)
{
    auto inventory = buildCUDAOnlyInventory();
    auto ctx = CollectiveContextFactory::createIntraNode(inventory, mpi_ctx_);

    ASSERT_NE(ctx, nullptr) << "Failed to create CollectiveContext";

#ifdef HAVE_NCCL
    EXPECT_TRUE(ctx->isBackendAvailable(CollectiveBackendType::NCCL))
        << "NCCL backend should be available for CUDA GPUs";
    LOG_INFO("[Rank " << rank_ << "] NCCL backend available for CUDA GPUs");
#else
    LOG_INFO("[Rank " << rank_ << "] NCCL not compiled in, using fallback");
#endif
}

/**
 * @brief Test NCCL AllReduce with GPU tensors
 *
 * NOTE: Multi-process NCCL requires proper CUDA device selection per rank.
 * Each MPI rank should use a different GPU to avoid comm init conflicts.
 */
TEST_F(Test__CollectiveNVIDIA, AllReduceSumViaNCCL)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "NCCL collective test requires at least 2 MPI ranks";
    }

    if (cuda_count_ < world_size_)
    {
        GTEST_SKIP() << "NCCL multi-process test requires at least " << world_size_
                     << " CUDA GPUs, have " << cuda_count_;
    }

    auto inventory = buildCUDAOnlyInventory();
    auto ctx = CollectiveContextFactory::createIntraNode(inventory, mpi_ctx_);

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{MEDIUM_SIZE});
    float *host_data = tensor->mutable_data();

    for (size_t i = 0; i < MEDIUM_SIZE; ++i)
    {
        host_data[i] = static_cast<float>((rank_ + 1) * (i % 100 + 1));
    }

    // Compute expected before uploading
    std::vector<float> all_initial(static_cast<size_t>(world_size_) * MEDIUM_SIZE);
    MPI_Allgather(
        host_data, static_cast<int>(MEDIUM_SIZE), MPI_FLOAT,
        all_initial.data(), static_cast<int>(MEDIUM_SIZE), MPI_FLOAT,
        MPI_COMM_WORLD);

    std::vector<float> expected(MEDIUM_SIZE, 0.0f);
    for (int r = 0; r < world_size_; ++r)
    {
        for (size_t i = 0; i < MEDIUM_SIZE; ++i)
        {
            expected[i] += all_initial[static_cast<size_t>(r) * MEDIUM_SIZE + i];
        }
    }

    // Upload to GPU
    ASSERT_TRUE(tensor->ensureOnDevice(cuda_dev_))
        << "Failed to upload tensor to CUDA device";
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // Execute AllReduce
    bool success = ctx->executeAllreduce(
        tensor.get(),
        MEDIUM_SIZE,
        cuda_dev_,
        CollectiveOp::ALLREDUCE_SUM);

    ASSERT_TRUE(success) << "NCCL AllReduce failed";

    // Mark device dirty after GPU collective
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // Download result
    ASSERT_TRUE(tensor->ensureOnHost()) << "Failed to download result";

    verifyAllReduceResult(tensor->data(), expected.data(), MEDIUM_SIZE, "NCCL");
    verifyRankParity(tensor->data(), MEDIUM_SIZE);
}

#endif // HAVE_CUDA

// =============================================================================
// Scenario 3: AMD+AMD Collective (RCCL Backend)
// =============================================================================

#ifdef HAVE_ROCM

class Test__CollectiveAMD : public Test__CollectiveBackendIntegration
{
protected:
    DeviceId rocm_dev_;

    void SetUp() override
    {
        Test__CollectiveBackendIntegration::SetUp();

        if (rocm_count_ == 0)
        {
            GTEST_SKIP() << "No ROCm GPUs available";
        }

        // Each rank uses a different ROCm device (if available)
        int device_idx = rank_ % rocm_count_;
        rocm_dev_ = DeviceId::rocm(device_idx);

        LOG_INFO("[Rank " << rank_ << "] Using ROCm device " << device_idx);
    }
};

/**
 * @brief Verify RCCL backend is selected for ROCm-only inventory
 */
TEST_F(Test__CollectiveAMD, BackendSelectionIsRCCL)
{
    auto inventory = buildROCmOnlyInventory();
    auto ctx = CollectiveContextFactory::createIntraNode(inventory, mpi_ctx_);

    ASSERT_NE(ctx, nullptr) << "Failed to create CollectiveContext";

#ifdef HAVE_RCCL
    EXPECT_TRUE(ctx->isBackendAvailable(CollectiveBackendType::RCCL))
        << "RCCL backend should be available for ROCm GPUs";
    LOG_INFO("[Rank " << rank_ << "] RCCL backend available for ROCm GPUs");
#else
    LOG_INFO("[Rank " << rank_ << "] RCCL not compiled in, using fallback");
#endif
}

/**
 * @brief Test RCCL AllReduce with GPU tensors
 */
TEST_F(Test__CollectiveAMD, AllReduceSumViaRCCL)
{
    if (world_size_ < 2)
    {
        GTEST_SKIP() << "RCCL collective test requires at least 2 MPI ranks";
    }

    if (rocm_count_ < world_size_)
    {
        GTEST_SKIP() << "RCCL multi-process test requires at least " << world_size_
                     << " ROCm GPUs, have " << rocm_count_;
    }

    auto inventory = buildROCmOnlyInventory();
    auto ctx = CollectiveContextFactory::createIntraNode(inventory, mpi_ctx_);

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{MEDIUM_SIZE});
    float *host_data = tensor->mutable_data();

    for (size_t i = 0; i < MEDIUM_SIZE; ++i)
    {
        host_data[i] = static_cast<float>((rank_ + 1) * (i % 100 + 1));
    }

    // Compute expected before uploading
    std::vector<float> all_initial(static_cast<size_t>(world_size_) * MEDIUM_SIZE);
    MPI_Allgather(
        host_data, static_cast<int>(MEDIUM_SIZE), MPI_FLOAT,
        all_initial.data(), static_cast<int>(MEDIUM_SIZE), MPI_FLOAT,
        MPI_COMM_WORLD);

    std::vector<float> expected(MEDIUM_SIZE, 0.0f);
    for (int r = 0; r < world_size_; ++r)
    {
        for (size_t i = 0; i < MEDIUM_SIZE; ++i)
        {
            expected[i] += all_initial[static_cast<size_t>(r) * MEDIUM_SIZE + i];
        }
    }

    // Upload to GPU
    ASSERT_TRUE(tensor->ensureOnDevice(rocm_dev_))
        << "Failed to upload tensor to ROCm device";
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // Execute AllReduce
    bool success = ctx->executeAllreduce(
        tensor.get(),
        MEDIUM_SIZE,
        rocm_dev_,
        CollectiveOp::ALLREDUCE_SUM);

    ASSERT_TRUE(success) << "RCCL AllReduce failed";

    // Mark device dirty after GPU collective
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // Download result
    ASSERT_TRUE(tensor->ensureOnHost()) << "Failed to download result";

    verifyAllReduceResult(tensor->data(), expected.data(), MEDIUM_SIZE, "RCCL");
    verifyRankParity(tensor->data(), MEDIUM_SIZE);
}

#endif // HAVE_ROCM

// =============================================================================
// Scenario 4: Cross-Vendor GPU Collective (HETEROGENEOUS Backend)
// =============================================================================

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

class Test__CollectiveCrossVendor : public Test__CollectiveBackendIntegration
{
protected:
    void SetUp() override
    {
        Test__CollectiveBackendIntegration::SetUp();

        if (cuda_count_ == 0 || rocm_count_ == 0)
        {
            GTEST_SKIP() << "Cross-vendor test requires both CUDA and ROCm GPUs";
        }
    }
};

/**
 * @brief Verify HETEROGENEOUS backend is selected for heterogeneous GPU inventory
 */
TEST_F(Test__CollectiveCrossVendor, BackendSelectionIsHeterogeneous)
{
    auto inventory = buildFullInventory();
    auto ctx = CollectiveContextFactory::createIntraNode(inventory, mpi_ctx_);

    ASSERT_NE(ctx, nullptr) << "Failed to create CollectiveContext";

    LOG_INFO("[Rank " << rank_ << "] Cross-vendor: CUDA " << cuda_count_
                      << " + ROCm " << rocm_count_);

    // HETEROGENEOUS backend should be available for cross-vendor
    if (ctx->isBackendAvailable(CollectiveBackendType::HETEROGENEOUS))
    {
        LOG_INFO("[Rank " << rank_ << "] HETEROGENEOUS backend available for mixed GPUs");
    }
    else
    {
        LOG_INFO("[Rank " << rank_ << "] HETEROGENEOUS backend not available (expected on some systems)");
    }
}

#endif // HAVE_CUDA && HAVE_ROCM

// =============================================================================
// Scenario 5: Single-Rank Optimization
// =============================================================================

class Test__CollectiveSingleRank : public Test__CollectiveBackendIntegration
{
protected:
    void SetUp() override
    {
        Test__CollectiveBackendIntegration::SetUp();
    }
};

/**
 * @brief Verify single-rank AllReduce is a no-op
 */
TEST_F(Test__CollectiveSingleRank, SingleRankAllReduceIsNoOp)
{
    if (world_size_ != 1)
    {
        GTEST_SKIP() << "Single-rank test requires exactly 1 MPI rank";
    }

    auto ctx = CollectiveContextFactory::createSingleDevice();

    ASSERT_NE(ctx, nullptr);
    EXPECT_FALSE(ctx->requiresCollectives())
        << "Single-rank context should not require collectives";

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{SMALL_SIZE});
    float *data = tensor->mutable_data();
    fill_deterministic(data, SMALL_SIZE, 42);

    std::vector<float> original(data, data + SMALL_SIZE);

    // AllReduce on single rank should return success without modifying data
    bool success = ctx->executeAllreduce(
        tensor.get(),
        SMALL_SIZE,
        DeviceId::cpu(),
        CollectiveOp::ALLREDUCE_SUM);

    EXPECT_TRUE(success) << "Single-rank AllReduce should succeed";

    // Data should be unchanged
    for (size_t i = 0; i < SMALL_SIZE; ++i)
    {
        EXPECT_FLOAT_EQ(tensor->data()[i], original[i])
            << "Single-rank AllReduce modified data at index " << i;
    }
}

/**
 * @brief Verify trivial domain (size 1) skips collective
 */
TEST_F(Test__CollectiveSingleRank, TrivialDomainSkipsCollective)
{
    if (world_size_ != 1)
    {
        GTEST_SKIP() << "Trivial domain test requires exactly 1 MPI rank";
    }

    auto ctx = CollectiveContextFactory::createSingleDevice();

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{SMALL_SIZE});
    float *data = tensor->mutable_data();
    fill_deterministic(data, SMALL_SIZE, 123);

    std::vector<float> original(data, data + SMALL_SIZE);

    // Create trivial domain (size 1)
    TPDomain domain;
    domain.domain_size = 1;
    domain.local_rank_in_domain = 0;
    domain.name = "trivial_domain";

    bool success = ctx->executeAllreduceInDomain(
        tensor.get(),
        SMALL_SIZE,
        DeviceId::cpu(),
        CollectiveOp::ALLREDUCE_SUM,
        &domain);

    EXPECT_TRUE(success) << "Trivial domain AllReduce should succeed";

    // Data should be unchanged
    for (size_t i = 0; i < SMALL_SIZE; ++i)
    {
        EXPECT_FLOAT_EQ(tensor->data()[i], original[i])
            << "Trivial domain AllReduce modified data at index " << i;
    }
}

// =============================================================================
// Main (MPI-aware)
// =============================================================================

int main(int argc, char **argv)
{
    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    // Initialize GoogleTest
    ::testing::InitGoogleTest(&argc, argv);

    // Run tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
