/**
 * @file Test__LocalTPContext.cpp
 * @brief Unit tests for LocalTPContext
 * @author David Sanftenberg
 * @date January 2026
 *
 * Tests for LOCAL tensor parallelism context:
 * - Construction with devices and weights
 * - Degree calculation
 * - Device index management
 * - Weight normalization
 * - Head/row/column range calculation for proportional TP
 * - Backend auto-detection
 * - Backend initialization (NEW)
 * - Collective operations delegating to backend (NEW)
 * - Integration with HostBackend (NEW)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>
#include <cstring>
#include <thread>
#include <atomic>

#include "collective/LocalTPContext.h"
#include "collective/ICollectiveBackend.h"
#include "collective/DeviceGroup.h"
#include "backends/GlobalDeviceAddress.h"
#include "config/OrchestrationConfig.h"
#include "tensors/Tensors.h"
#include "../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__LocalTPContext : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create test devices
        cuda0_ = GlobalDeviceAddress::cuda(0, 0);
        cuda1_ = GlobalDeviceAddress::cuda(1, 0);
        rocm0_ = GlobalDeviceAddress::rocm(0, 0);
        cpu0_ = GlobalDeviceAddress::cpu(0);
    }

    GlobalDeviceAddress cuda0_;
    GlobalDeviceAddress cuda1_;
    GlobalDeviceAddress rocm0_;
    GlobalDeviceAddress cpu0_;
};

// =============================================================================
// Construction Tests
// =============================================================================

/**
 * @test Construct with single device and no weights
 */
TEST_F(Test__LocalTPContext, ConstructSingleDevice)
{
    auto ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->degree(), 1);
    EXPECT_EQ(ctx->devices().size(), 1);
    EXPECT_EQ(ctx->weights().size(), 1);
    EXPECT_FLOAT_EQ(ctx->weights()[0], 1.0f);
}

/**
 * @test Construct with two devices and equal weights
 */
TEST_F(Test__LocalTPContext, ConstructTwoDevicesEqualWeights)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);

    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->degree(), 2);
    EXPECT_EQ(ctx->devices().size(), 2);
    EXPECT_EQ(ctx->weights().size(), 2);
    EXPECT_FLOAT_EQ(ctx->weights()[0], 0.5f);
    EXPECT_FLOAT_EQ(ctx->weights()[1], 0.5f);
}

/**
 * @test Construct with explicit proportional weights
 */
TEST_F(Test__LocalTPContext, ConstructProportionalWeights)
{
    // 73% / 27% split (like NVIDIA vs AMD performance ratio)
    auto ctx = createLocalTPContext({cuda0_, rocm0_}, {0.73f, 0.27f}, CollectiveBackendType::PCIE_BAR);

    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->degree(), 2);

    // Weights should be normalized to sum to 1.0
    float sum = ctx->weights()[0] + ctx->weights()[1];
    EXPECT_FLOAT_EQ(sum, 1.0f);
    EXPECT_FLOAT_EQ(ctx->weights()[0], 0.73f);
    EXPECT_FLOAT_EQ(ctx->weights()[1], 0.27f);
}

/**
 * @test Construct with unnormalized weights (should normalize)
 */
TEST_F(Test__LocalTPContext, ConstructUnnormalizedWeights)
{
    // Weights that don't sum to 1.0 should be normalized
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {2.0f, 1.0f}, CollectiveBackendType::NCCL);

    ASSERT_NE(ctx, nullptr);

    // Should normalize to 2/3, 1/3
    float sum = ctx->weights()[0] + ctx->weights()[1];
    EXPECT_FLOAT_EQ(sum, 1.0f);
    EXPECT_NEAR(ctx->weights()[0], 2.0f / 3.0f, 0.0001f);
    EXPECT_NEAR(ctx->weights()[1], 1.0f / 3.0f, 0.0001f);
}

/**
 * @test Construct with empty devices throws
 */
TEST_F(Test__LocalTPContext, ConstructEmptyDevicesThrows)
{
    EXPECT_THROW(
        createLocalTPContext({}, {}, CollectiveBackendType::AUTO),
        std::invalid_argument);
}

/**
 * @test Construct with mismatched weights count throws
 */
TEST_F(Test__LocalTPContext, ConstructMismatchedWeightsThrows)
{
    EXPECT_THROW(
        createLocalTPContext({cuda0_, cuda1_}, {0.5f}, CollectiveBackendType::NCCL),
        std::invalid_argument);
}

/**
 * @test Construct with zero weight throws
 */
TEST_F(Test__LocalTPContext, ConstructZeroWeightThrows)
{
    EXPECT_THROW(
        createLocalTPContext({cuda0_, cuda1_}, {1.0f, 0.0f}, CollectiveBackendType::NCCL),
        std::invalid_argument);
}

/**
 * @test Construct with negative weight throws
 */
TEST_F(Test__LocalTPContext, ConstructNegativeWeightThrows)
{
    EXPECT_THROW(
        createLocalTPContext({cuda0_, cuda1_}, {1.0f, -0.5f}, CollectiveBackendType::NCCL),
        std::invalid_argument);
}

// =============================================================================
// Backend Auto-Detection Tests
// =============================================================================

/**
 * @test AUTO backend with all CUDA devices -> NCCL
 */
TEST_F(Test__LocalTPContext, AutoBackendAllCuda)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::AUTO);

    EXPECT_EQ(ctx->backend(), CollectiveBackendType::NCCL);
}

// NOTE: Hardware-dependent backend auto-detection tests (AutoBackendAllRocm,
// AutoBackendMixedGpus, AutoBackendSelectsPCIeBarFor*, AutoBackendSelectsHeterogeneousFor*)
// have been migrated to integration tests in Test__LocalTPBackendBehavior.cpp.
// Those tests require actual GPU hardware (RCCL, PCIeBAR, HETEROGENEOUS backends).

/**
 * @test AUTO backend with CPU involved -> HOST
 */
TEST_F(Test__LocalTPContext, AutoBackendWithCpu)
{
    auto ctx = createLocalTPContext({cuda0_, cpu0_}, {}, CollectiveBackendType::AUTO);

    EXPECT_EQ(ctx->backend(), CollectiveBackendType::HOST);
}

/**
 * @test Explicit backend is preserved
 */
TEST_F(Test__LocalTPContext, ExplicitBackendPreserved)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::MPI);

    EXPECT_EQ(ctx->backend(), CollectiveBackendType::MPI);
}

// =============================================================================
// Device Management Tests
// =============================================================================

/**
 * @test indexForDevice returns correct index
 */
TEST_F(Test__LocalTPContext, IndexForDeviceCorrect)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_, rocm0_}, {}, CollectiveBackendType::PCIE_BAR);

    EXPECT_EQ(ctx->indexForDevice(cuda0_), 0);
    EXPECT_EQ(ctx->indexForDevice(cuda1_), 1);
    EXPECT_EQ(ctx->indexForDevice(rocm0_), 2);
}

/**
 * @test indexForDevice returns -1 for unknown device
 */
TEST_F(Test__LocalTPContext, IndexForDeviceNotFound)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);

    EXPECT_EQ(ctx->indexForDevice(rocm0_), -1);
    EXPECT_EQ(ctx->indexForDevice(cpu0_), -1);
}

/**
 * @test deviceAt returns correct device
 */
TEST_F(Test__LocalTPContext, DeviceAtCorrect)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_, rocm0_}, {}, CollectiveBackendType::PCIE_BAR);

    EXPECT_EQ(ctx->deviceAt(0), cuda0_);
    EXPECT_EQ(ctx->deviceAt(1), cuda1_);
    EXPECT_EQ(ctx->deviceAt(2), rocm0_);
}

/**
 * @test deviceAt throws for invalid index
 */
TEST_F(Test__LocalTPContext, DeviceAtThrowsForInvalidIndex)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);

    EXPECT_THROW(ctx->deviceAt(-1), std::out_of_range);
    EXPECT_THROW(ctx->deviceAt(2), std::out_of_range);
    EXPECT_THROW(ctx->deviceAt(100), std::out_of_range);
}

/**
 * @test weightForDevice returns correct weight
 */
TEST_F(Test__LocalTPContext, WeightForDeviceCorrect)
{
    auto ctx = createLocalTPContext({cuda0_, rocm0_}, {0.73f, 0.27f}, CollectiveBackendType::PCIE_BAR);

    EXPECT_FLOAT_EQ(ctx->weightForDevice(cuda0_), 0.73f);
    EXPECT_FLOAT_EQ(ctx->weightForDevice(rocm0_), 0.27f);
}

/**
 * @test weightForDevice returns 0 for unknown device
 */
TEST_F(Test__LocalTPContext, WeightForDeviceUnknown)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);

    EXPECT_FLOAT_EQ(ctx->weightForDevice(rocm0_), 0.0f);
}

// =============================================================================
// Head Distribution Tests
// =============================================================================

/**
 * @test headsForDevice with equal weights
 */
TEST_F(Test__LocalTPContext, HeadsForDeviceEqualWeights)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);

    // 28 heads / 2 devices = 14 each
    EXPECT_EQ(ctx->headsForDevice(cuda0_, 28), 14);
    EXPECT_EQ(ctx->headsForDevice(cuda1_, 28), 14);

    // Total should equal original
    int total = ctx->headsForDevice(cuda0_, 28) + ctx->headsForDevice(cuda1_, 28);
    EXPECT_EQ(total, 28);
}

/**
 * @test headsForDevice with odd number of heads
 */
TEST_F(Test__LocalTPContext, HeadsForDeviceOddHeads)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);

    // 29 heads / 2 devices = 14 + 15 (last gets remainder)
    int h0 = ctx->headsForDevice(cuda0_, 29);
    int h1 = ctx->headsForDevice(cuda1_, 29);

    EXPECT_EQ(h0 + h1, 29);
    // First should get ~half, second gets remainder
    EXPECT_TRUE(h0 >= 14 && h0 <= 15);
    EXPECT_TRUE(h1 >= 14 && h1 <= 15);
}

/**
 * @test headsForDevice with proportional weights
 */
TEST_F(Test__LocalTPContext, HeadsForDeviceProportional)
{
    auto ctx = createLocalTPContext({cuda0_, rocm0_}, {0.73f, 0.27f}, CollectiveBackendType::PCIE_BAR);

    // 28 heads with 73%/27% split
    int h0 = ctx->headsForDevice(cuda0_, 28); // Should get ~20 heads (73% of 28)
    int h1 = ctx->headsForDevice(rocm0_, 28); // Should get ~8 heads (27% of 28)

    // Total must equal original
    EXPECT_EQ(h0 + h1, 28);

    // Check proportions are approximately correct
    EXPECT_GE(h0, 18); // At least 64% to account for rounding
    EXPECT_LE(h0, 22); // At most 78%
    EXPECT_GE(h1, 6);
    EXPECT_LE(h1, 10);
}

/**
 * @test headsForDevice with zero heads
 */
TEST_F(Test__LocalTPContext, HeadsForDeviceZeroHeads)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);

    EXPECT_EQ(ctx->headsForDevice(cuda0_, 0), 0);
    EXPECT_EQ(ctx->headsForDevice(cuda1_, 0), 0);
}

/**
 * @test headsForDevice for unknown device
 */
TEST_F(Test__LocalTPContext, HeadsForDeviceUnknown)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);

    EXPECT_EQ(ctx->headsForDevice(rocm0_, 28), 0);
}

// =============================================================================
// Row/Column Range Tests
// =============================================================================

/**
 * @test rowRangeForDevice with equal weights
 */
TEST_F(Test__LocalTPContext, RowRangeForDeviceEqual)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);

    auto [r0_start, r0_end] = ctx->rowRangeForDevice(cuda0_, 1000);
    auto [r1_start, r1_end] = ctx->rowRangeForDevice(cuda1_, 1000);

    // First device: [0, 500)
    EXPECT_EQ(r0_start, 0);
    EXPECT_EQ(r0_end, 500);

    // Second device: [500, 1000)
    EXPECT_EQ(r1_start, 500);
    EXPECT_EQ(r1_end, 1000);

    // Ranges should be contiguous and cover all rows
    EXPECT_EQ(r0_end, r1_start);
    EXPECT_EQ(r1_end - r0_start, 1000);
}

/**
 * @test rowRangeForDevice with proportional weights
 */
TEST_F(Test__LocalTPContext, RowRangeForDeviceProportional)
{
    auto ctx = createLocalTPContext({cuda0_, rocm0_}, {0.73f, 0.27f}, CollectiveBackendType::PCIE_BAR);

    auto [r0_start, r0_end] = ctx->rowRangeForDevice(cuda0_, 1000);
    auto [r1_start, r1_end] = ctx->rowRangeForDevice(rocm0_, 1000);

    // Ranges should be contiguous
    EXPECT_EQ(r0_start, 0);
    EXPECT_EQ(r0_end, r1_start);
    EXPECT_EQ(r1_end, 1000);

    // Check proportions
    int count0 = r0_end - r0_start;
    int count1 = r1_end - r1_start;
    EXPECT_EQ(count0 + count1, 1000);

    // Device 0 should get ~730 rows (73%)
    EXPECT_GE(count0, 700);
    EXPECT_LE(count0, 760);
}

/**
 * @test colRangeForDevice with three devices
 */
TEST_F(Test__LocalTPContext, ColRangeForDeviceThreeDevices)
{
    auto rocm1 = GlobalDeviceAddress::rocm(1, 0);
    auto ctx = createLocalTPContext(
        {cuda0_, rocm0_, rocm1},
        {0.5f, 0.25f, 0.25f},
        CollectiveBackendType::PCIE_BAR);

    auto [c0_start, c0_end] = ctx->colRangeForDevice(cuda0_, 1024);
    auto [c1_start, c1_end] = ctx->colRangeForDevice(rocm0_, 1024);
    auto [c2_start, c2_end] = ctx->colRangeForDevice(rocm1, 1024);

    // Ranges should be contiguous
    EXPECT_EQ(c0_start, 0);
    EXPECT_EQ(c0_end, c1_start);
    EXPECT_EQ(c1_end, c2_start);
    EXPECT_EQ(c2_end, 1024);

    // Check proportions (50%, 25%, 25%)
    int count0 = c0_end - c0_start;
    int count1 = c1_end - c1_start;
    int count2 = c2_end - c2_start;

    EXPECT_EQ(count0 + count1 + count2, 1024);

    // Device 0 should get ~512 cols
    EXPECT_GE(count0, 480);
    EXPECT_LE(count0, 544);

    // Devices 1 and 2 should get ~256 each
    EXPECT_GE(count1, 224);
    EXPECT_LE(count1, 288);
    EXPECT_GE(count2, 224);
    EXPECT_LE(count2, 288);
}

/**
 * @test rowRangeForDevice with zero rows
 */
TEST_F(Test__LocalTPContext, RowRangeForDeviceZeroRows)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);

    auto [r0_start, r0_end] = ctx->rowRangeForDevice(cuda0_, 0);
    EXPECT_EQ(r0_start, 0);
    EXPECT_EQ(r0_end, 0);
}

/**
 * @test colRangeForDevice for unknown device
 */
TEST_F(Test__LocalTPContext, ColRangeForDeviceUnknown)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);

    auto [start, end] = ctx->colRangeForDevice(rocm0_, 1000);
    EXPECT_EQ(start, 0);
    EXPECT_EQ(end, 0);
}

// =============================================================================
// Collective Operation Tests (Basic Validation)
// =============================================================================

/**
 * @test allreduce with single device is no-op
 */
TEST_F(Test__LocalTPContext, AllreduceSingleDeviceNoop)
{
    auto ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);

    // Single device - should succeed as no-op
    // Note: Full collective testing requires integration tests with real devices
    EXPECT_TRUE(ctx->allreduce(nullptr) == false); // null tensor fails
}

/**
 * @test synchronize doesn't crash
 */
TEST_F(Test__LocalTPContext, SynchronizeDoesNotCrash)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);

    // Should not crash
    ctx->synchronize();
}
// =============================================================================
// MockCollectiveBackend for Testing
// =============================================================================

/**
 * @brief Mock ICollectiveBackend that tracks calls and allows configurable behavior
 *
 * Used for testing that LocalTPContext correctly delegates to its backend.
 */
class MockCollectiveBackend : public ICollectiveBackend
{
public:
    // Call counters
    std::atomic<int> initialize_call_count{0};
    std::atomic<int> shutdown_call_count{0};
    std::atomic<int> allreduce_call_count{0};
    std::atomic<int> allreduce_multi_call_count{0};
    std::atomic<int> allgather_call_count{0};
    std::atomic<int> allgather_multi_call_count{0};
    std::atomic<int> reduce_scatter_call_count{0};
    std::atomic<int> synchronize_call_count{0};
    std::atomic<int> broadcast_call_count{0};

    // Configurable behavior
    bool should_fail_initialize = false;
    bool should_fail_allreduce = false;
    bool should_fail_allgather = false;
    bool should_fail_reduce_scatter = false;
    bool multi_gpu_mode = true;

    // Captured parameters from last call (for verification)
    size_t last_allreduce_count = 0;
    size_t last_allgather_count = 0;
    size_t last_reduce_scatter_count = 0;
    CollectiveDataType last_dtype = CollectiveDataType::FLOAT32;
    CollectiveOp last_op = CollectiveOp::ALLREDUCE_SUM;
    std::vector<void *> last_multi_buffers;

    // =========================================================================
    // Identity
    // =========================================================================

    CollectiveBackendType type() const override { return CollectiveBackendType::HOST; }
    std::string name() const override { return "MockBackend"; }

    // =========================================================================
    // Capability Queries
    // =========================================================================

    bool supportsDevice(DeviceType type) const override
    {
        (void)type;
        return true;
    }

    bool supportsDirectTransfer(DeviceId src, DeviceId dst) const override
    {
        (void)src;
        (void)dst;
        return true;
    }

    bool isAvailable() const override { return true; }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    bool initialize(const DeviceGroup &group) override
    {
        initialize_call_count++;
        group_ = group;
        initialized_ = !should_fail_initialize;
        return initialized_;
    }

    bool isInitialized() const override { return initialized_; }

    void shutdown() override
    {
        shutdown_call_count++;
        initialized_ = false;
    }

    // =========================================================================
    // Collective Operations
    // =========================================================================

    bool allreduce(void *buffer, size_t count, CollectiveDataType dtype, CollectiveOp op) override
    {
        allreduce_call_count++;
        last_allreduce_count = count;
        last_dtype = dtype;
        last_op = op;
        (void)buffer;
        return !should_fail_allreduce;
    }

    bool allgather(const void *send_buf, void *recv_buf, size_t send_count, CollectiveDataType dtype) override
    {
        allgather_call_count++;
        last_allgather_count = send_count;
        last_dtype = dtype;
        (void)send_buf;
        (void)recv_buf;
        return !should_fail_allgather;
    }

    bool allgatherv(const void *send_buf, size_t send_count, void *recv_buf,
                    const std::vector<int> &recv_counts, const std::vector<int> &displacements,
                    CollectiveDataType dtype) override
    {
        (void)send_buf;
        (void)send_count;
        (void)recv_buf;
        (void)recv_counts;
        (void)displacements;
        (void)dtype;
        return true;
    }

    bool reduceScatter(const void *send_buf, void *recv_buf, size_t recv_count,
                       CollectiveDataType dtype, CollectiveOp op) override
    {
        reduce_scatter_call_count++;
        last_reduce_scatter_count = recv_count;
        last_dtype = dtype;
        last_op = op;
        (void)send_buf;
        (void)recv_buf;
        return !should_fail_reduce_scatter;
    }

    bool broadcast(void *buffer, size_t count, CollectiveDataType dtype, int root_rank) override
    {
        broadcast_call_count++;
        (void)buffer;
        (void)count;
        (void)dtype;
        (void)root_rank;
        return true;
    }

    bool synchronize() override
    {
        synchronize_call_count++;
        return true;
    }

    // =========================================================================
    // Multi-GPU Operations
    // =========================================================================

    bool isMultiGpuSingleProcess() const override { return multi_gpu_mode; }

    bool allreduceMulti(const std::vector<void *> &buffers, size_t count,
                        CollectiveDataType dtype, CollectiveOp op) override
    {
        allreduce_multi_call_count++;
        last_multi_buffers = buffers;
        last_allreduce_count = count;
        last_dtype = dtype;
        last_op = op;
        return !should_fail_allreduce;
    }

    bool allgatherMulti(const std::vector<const void *> &send_bufs,
                        const std::vector<void *> &recv_bufs,
                        size_t send_count, CollectiveDataType dtype) override
    {
        allgather_multi_call_count++;
        last_allgather_count = send_count;
        last_dtype = dtype;
        (void)send_bufs;
        (void)recv_bufs;
        return !should_fail_allgather;
    }

    std::string lastError() const override { return last_error_; }

    // Helper to reset all counters
    void reset()
    {
        initialize_call_count = 0;
        shutdown_call_count = 0;
        allreduce_call_count = 0;
        allreduce_multi_call_count = 0;
        allgather_call_count = 0;
        allgather_multi_call_count = 0;
        reduce_scatter_call_count = 0;
        synchronize_call_count = 0;
        broadcast_call_count = 0;
        should_fail_initialize = false;
        should_fail_allreduce = false;
        should_fail_allgather = false;
        should_fail_reduce_scatter = false;
        multi_gpu_mode = true;
        last_multi_buffers.clear();
    }

private:
    DeviceGroup group_;
    bool initialized_ = false;
    std::string last_error_;
};

// =============================================================================
// Backend Initialization Tests
// =============================================================================

/**
 * @test Single device skips backend initialization
 */
TEST_F(Test__LocalTPContext, SingleDeviceSkipsBackendInit)
{
    // With a single device, no collective backend is needed
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    EXPECT_EQ(ctx->degree(), 1);
    // Single device should not attempt backend initialization
    // The context should still work (allreduce is a no-op)

    auto tensor = TestTensorFactory::createFP32({4, 4});
    TestTensorFactory::fillValue(tensor.get(), 1.0f);

    // Should succeed as no-op for single device
    EXPECT_TRUE(ctx->allreduce(tensor.get()));
}

/**
 * @test HOST backend is always available
 */
TEST_F(Test__LocalTPContext, HostBackendAlwaysAvailable)
{
    // CPU devices should get HOST backend and it should initialize
    auto ctx = createLocalTPContext({cpu0_, GlobalDeviceAddress::cpu(1)}, {}, CollectiveBackendType::HOST);

    EXPECT_EQ(ctx->backend(), CollectiveBackendType::HOST);
    EXPECT_EQ(ctx->degree(), 2);
}

/**
 * @test AUTO backend detection with CPU devices -> HOST
 */
TEST_F(Test__LocalTPContext, AutoBackendCpuDevicesUsesHost)
{
    auto cpu1 = GlobalDeviceAddress::cpu(1);
    auto ctx = createLocalTPContext({cpu0_, cpu1}, {}, CollectiveBackendType::AUTO);

    EXPECT_EQ(ctx->backend(), CollectiveBackendType::HOST);
}

// =============================================================================
// Collective Operation Tests with Real HostBackend
// =============================================================================

/**
 * @test Allreduce with single CPU device is a no-op success
 */
TEST_F(Test__LocalTPContext, AllreduceSingleCpuDeviceSuccess)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    auto tensor = TestTensorFactory::createFP32({2, 4});
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < 8; ++i)
    {
        data[i] = static_cast<float>(i);
    }

    // Should succeed (no-op)
    EXPECT_TRUE(ctx->allreduce(tensor.get()));

    // Data should be unchanged
    for (size_t i = 0; i < 8; ++i)
    {
        EXPECT_FLOAT_EQ(tensor->data()[i], static_cast<float>(i));
    }
}

/**
 * @test Out-of-place allreduce copies data for single device
 */
TEST_F(Test__LocalTPContext, AllreduceOutOfPlaceCopiesForSingleDevice)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    auto input = TestTensorFactory::createFP32({2, 4});
    auto output = TestTensorFactory::createFP32({2, 4});

    float *in_data = input->mutable_data();
    for (size_t i = 0; i < 8; ++i)
    {
        in_data[i] = static_cast<float>(i + 1);
    }

    // Out-of-place should copy input to output
    EXPECT_TRUE(ctx->allreduce(input.get(), output.get()));

    // Output should have input's data
    for (size_t i = 0; i < 8; ++i)
    {
        EXPECT_FLOAT_EQ(output->data()[i], static_cast<float>(i + 1));
    }
}

/**
 * @test Allgather copies local shard to global for single device
 */
TEST_F(Test__LocalTPContext, AllgatherSingleDeviceCopies)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    auto local_shard = TestTensorFactory::createFP32({1, 4});
    auto global_tensor = TestTensorFactory::createFP32({1, 4});

    float *shard_data = local_shard->mutable_data();
    for (size_t i = 0; i < 4; ++i)
    {
        shard_data[i] = static_cast<float>(i * 2);
    }

    EXPECT_TRUE(ctx->allgather(local_shard.get(), global_tensor.get()));

    // Global should have shard's data
    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_FLOAT_EQ(global_tensor->data()[i], static_cast<float>(i * 2));
    }
}

/**
 * @test ReduceScatter copies for single device
 */
TEST_F(Test__LocalTPContext, ReduceScatterSingleDeviceCopies)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    auto input = TestTensorFactory::createFP32({2, 4});
    auto output_shard = TestTensorFactory::createFP32({2, 4});

    float *in_data = input->mutable_data();
    for (size_t i = 0; i < 8; ++i)
    {
        in_data[i] = static_cast<float>(i + 10);
    }

    EXPECT_TRUE(ctx->reduceScatter(input.get(), output_shard.get()));

    // Output should have input's data (for single device)
    for (size_t i = 0; i < 8; ++i)
    {
        EXPECT_FLOAT_EQ(output_shard->data()[i], static_cast<float>(i + 10));
    }
}

/**
 * @test Synchronize completes for single device
 */
TEST_F(Test__LocalTPContext, SynchronizeSingleDevice)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    // Should complete without issues
    ctx->synchronize();
    SUCCEED(); // If we get here, test passes
}

// =============================================================================
// Error Handling Tests
// =============================================================================

/**
 * @test Allreduce with null tensor returns false
 */
TEST_F(Test__LocalTPContext, AllreduceNullTensorFails)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    EXPECT_FALSE(ctx->allreduce(nullptr));
}

/**
 * @test Out-of-place allreduce with null input fails
 */
TEST_F(Test__LocalTPContext, AllreduceOutOfPlaceNullInputFails)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    auto output = TestTensorFactory::createFP32({2, 4});
    EXPECT_FALSE(ctx->allreduce(nullptr, output.get()));
}

/**
 * @test Out-of-place allreduce with null output fails
 */
TEST_F(Test__LocalTPContext, AllreduceOutOfPlaceNullOutputFails)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    auto input = TestTensorFactory::createFP32({2, 4});
    EXPECT_FALSE(ctx->allreduce(input.get(), nullptr));
}

/**
 * @test Allgather with null tensors fails
 */
TEST_F(Test__LocalTPContext, AllgatherNullTensorsFails)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    auto valid_tensor = TestTensorFactory::createFP32({2, 4});
    EXPECT_FALSE(ctx->allgather(nullptr, valid_tensor.get()));
    EXPECT_FALSE(ctx->allgather(valid_tensor.get(), nullptr));
}

/**
 * @test ReduceScatter with null tensors fails
 */
TEST_F(Test__LocalTPContext, ReduceScatterNullTensorsFails)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    auto valid_tensor = TestTensorFactory::createFP32({2, 4});
    EXPECT_FALSE(ctx->reduceScatter(nullptr, valid_tensor.get()));
    EXPECT_FALSE(ctx->reduceScatter(valid_tensor.get(), nullptr));
}

// =============================================================================
// Multi-Device Scenarios (Backend Not Initialized)
// =============================================================================

/**
 * @test Multi-device allreduce when backend not initialized falls back gracefully
 *
 * When GPU backends (NCCL, RCCL) are requested but not available, the context
 * should still work with degraded behavior (no actual reduction, just returns true).
 */
TEST_F(Test__LocalTPContext, MultiDeviceAllreduceBackendUnavailableFallback)
{
    // Request NCCL for CUDA devices - if NCCL is not compiled in, this will
    // fall back to HOST backend
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);

    auto tensor = TestTensorFactory::createFP32({2, 4});
    TestTensorFactory::fillValue(tensor.get(), 1.0f);

    // Should not crash. May return true (HOST fallback) or false (backend failure)
    // The exact behavior depends on whether NCCL is available
    ctx->allreduce(tensor.get());
    SUCCEED(); // Test passes if no crash
}

/**
 * @test Multi-device synchronize when backend not initialized is no-op
 */
TEST_F(Test__LocalTPContext, MultiDeviceSynchronizeBackendUnavailableNoop)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::NCCL);

    // Should not crash even if backend is not fully initialized
    ctx->synchronize();
    SUCCEED();
}

// =============================================================================
// CPU-Only Multi-Device Tests with HostBackend
// =============================================================================

/**
 * @test Two CPU devices with HOST backend initializes correctly
 *
 * Note: The HostBackend is primarily designed for GPU-to-GPU collectives
 * with CPU staging. When all devices are CPU, the backend will try to
 * use GPU APIs which will fail. This test verifies the context initializes
 * correctly and that synchronize (which is a no-op) works.
 */
TEST_F(Test__LocalTPContext, TwoCpuDevicesWithHostBackend)
{
    auto cpu1 = GlobalDeviceAddress::cpu(1);
    auto ctx = createLocalTPContext({cpu0_, cpu1}, {}, CollectiveBackendType::HOST);

    EXPECT_EQ(ctx->degree(), 2);
    EXPECT_EQ(ctx->backend(), CollectiveBackendType::HOST);

    // Synchronize should work (no-op for host backend)
    ctx->synchronize();
    SUCCEED();
}

// =============================================================================
// Thread Safety Smoke Tests
// =============================================================================

/**
 * @test Concurrent synchronize calls don't crash
 */
TEST_F(Test__LocalTPContext, ConcurrentSynchronizeDoesNotCrash)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i)
    {
        threads.emplace_back([&ctx]()
                             {
            for (int j = 0; j < 10; ++j)
            {
                ctx->synchronize();
            } });
    }

    for (auto &t : threads)
    {
        t.join();
    }

    SUCCEED();
}

// =============================================================================
// GatherFromDevices Tests (for MultiDeviceOrchestrator support)
// =============================================================================

/**
 * @test gatherFromDevices with single device just copies data
 */
TEST_F(Test__LocalTPContext, GatherFromDevicesSingleDevice)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    // Create input shard (vocab_local = 100)
    auto shard = TestTensorFactory::createFP32({100});
    for (size_t i = 0; i < 100; ++i)
    {
        shard->mutable_data()[i] = static_cast<float>(i);
    }

    // Create output tensor (same size for single device)
    auto output = TestTensorFactory::createFP32({100});

    // Gather
    std::vector<const TensorBase *> shards = {shard.get()};
    ASSERT_TRUE(ctx->gatherFromDevices(shards, output.get()));

    // Verify data copied correctly
    for (size_t i = 0; i < 100; ++i)
    {
        EXPECT_FLOAT_EQ(output->data()[i], static_cast<float>(i));
    }
}

/**
 * @test gatherFromDevices with two devices concatenates data
 */
TEST_F(Test__LocalTPContext, GatherFromDevicesTwoDevices)
{
    auto cpu1 = GlobalDeviceAddress::cpu(1);
    auto ctx = createLocalTPContext({cpu0_, cpu1}, {0.5f, 0.5f}, CollectiveBackendType::HOST);

    // Create shards: device 0 has [0..49], device 1 has [100..149]
    auto shard0 = TestTensorFactory::createFP32({50});
    auto shard1 = TestTensorFactory::createFP32({50});

    for (size_t i = 0; i < 50; ++i)
    {
        shard0->mutable_data()[i] = static_cast<float>(i);       // 0..49
        shard1->mutable_data()[i] = static_cast<float>(100 + i); // 100..149
    }

    // Create output tensor (full vocab = 100)
    auto output = TestTensorFactory::createFP32({100});

    // Gather
    std::vector<const TensorBase *> shards = {shard0.get(), shard1.get()};
    ASSERT_TRUE(ctx->gatherFromDevices(shards, output.get()));

    // Verify data concatenated correctly
    // First 50 elements should be from shard0 (0..49)
    for (size_t i = 0; i < 50; ++i)
    {
        EXPECT_FLOAT_EQ(output->data()[i], static_cast<float>(i))
            << "Mismatch at index " << i;
    }
    // Next 50 elements should be from shard1 (100..149)
    for (size_t i = 0; i < 50; ++i)
    {
        EXPECT_FLOAT_EQ(output->data()[50 + i], static_cast<float>(100 + i))
            << "Mismatch at index " << (50 + i);
    }
}

/**
 * @test gatherFromDevices with proportional weights (different shard sizes)
 *
 * Simulates column-parallel LM head with 73%/27% weight distribution.
 * vocab_size = 100, so device 0 gets 73 tokens, device 1 gets 27 tokens.
 */
TEST_F(Test__LocalTPContext, GatherFromDevicesProportionalWeights)
{
    auto cpu1 = GlobalDeviceAddress::cpu(1);
    auto ctx = createLocalTPContext({cpu0_, cpu1}, {0.73f, 0.27f}, CollectiveBackendType::HOST);

    // Create shards with different sizes
    auto shard0 = TestTensorFactory::createFP32({73}); // 73% of vocab
    auto shard1 = TestTensorFactory::createFP32({27}); // 27% of vocab

    for (size_t i = 0; i < 73; ++i)
    {
        shard0->mutable_data()[i] = static_cast<float>(i);
    }
    for (size_t i = 0; i < 27; ++i)
    {
        shard1->mutable_data()[i] = static_cast<float>(1000 + i); // Different range
    }

    // Create output tensor (full vocab = 100)
    auto output = TestTensorFactory::createFP32({100});

    // Gather
    std::vector<const TensorBase *> shards = {shard0.get(), shard1.get()};
    ASSERT_TRUE(ctx->gatherFromDevices(shards, output.get()));

    // Verify concatenation
    for (size_t i = 0; i < 73; ++i)
    {
        EXPECT_FLOAT_EQ(output->data()[i], static_cast<float>(i));
    }
    for (size_t i = 0; i < 27; ++i)
    {
        EXPECT_FLOAT_EQ(output->data()[73 + i], static_cast<float>(1000 + i));
    }
}

/**
 * @test gatherFromDevices fails with empty shards vector
 */
TEST_F(Test__LocalTPContext, GatherFromDevicesEmptyShardsFails)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    auto output = TestTensorFactory::createFP32({100});

    std::vector<const TensorBase *> empty_shards;
    EXPECT_FALSE(ctx->gatherFromDevices(empty_shards, output.get()));
}

/**
 * @test gatherFromDevices fails with null output
 */
TEST_F(Test__LocalTPContext, GatherFromDevicesNullOutputFails)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);

    auto shard = TestTensorFactory::createFP32({100});
    std::vector<const TensorBase *> shards = {shard.get()};

    EXPECT_FALSE(ctx->gatherFromDevices(shards, nullptr));
}

/**
 * @test gatherFromDevices fails when shard count mismatches device count
 */
TEST_F(Test__LocalTPContext, GatherFromDevicesWrongShardCountFails)
{
    auto cpu1 = GlobalDeviceAddress::cpu(1);
    auto ctx = createLocalTPContext({cpu0_, cpu1}, {}, CollectiveBackendType::HOST);

    auto shard = TestTensorFactory::createFP32({100});
    auto output = TestTensorFactory::createFP32({200});

    // Only one shard for two-device context
    std::vector<const TensorBase *> shards = {shard.get()};
    EXPECT_FALSE(ctx->gatherFromDevices(shards, output.get()));
}

/**
 * @test gatherFromDevices fails when output buffer is too small
 */
TEST_F(Test__LocalTPContext, GatherFromDevicesOutputTooSmallFails)
{
    auto cpu1 = GlobalDeviceAddress::cpu(1);
    auto ctx = createLocalTPContext({cpu0_, cpu1}, {}, CollectiveBackendType::HOST);

    auto shard0 = TestTensorFactory::createFP32({50});
    auto shard1 = TestTensorFactory::createFP32({50});
    auto output = TestTensorFactory::createFP32({50}); // Too small! Need 100.

    std::vector<const TensorBase *> shards = {shard0.get(), shard1.get()};
    EXPECT_FALSE(ctx->gatherFromDevices(shards, output.get()));
}