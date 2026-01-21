/**
 * @file Test__UPIBackend.cpp
 * @brief Unit tests for UPICollectiveBackend
 *
 * Tests the UPI-based collective backend for cross-socket CPU tensor parallelism.
 * These tests do NOT require actual MPI execution - they test construction,
 * capability queries, and type conversion functions.
 *
 * For integration tests with actual MPI collectives, see Test__UPIBackendMPI.cpp
 */

#include <gtest/gtest.h>
#include "v2/collective/backends/UPIBackend.h"
#include "v2/collective/DeviceGroup.h"
#include "v2/backends/DeviceId.h"

namespace llaminar2::test
{

    // =============================================================================
    // Test Fixture
    // =============================================================================

    class Test__UPIBackend : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Create backend with MPI_COMM_NULL for capability testing
            // This allows testing construction and queries without actual MPI
            backend_null_ = std::make_unique<UPICollectiveBackend>(MPI_COMM_NULL);
        }

        void TearDown() override
        {
            if (backend_null_ && backend_null_->isInitialized())
            {
                backend_null_->shutdown();
            }
        }

        // Helper to create a local-scope CPU group
        DeviceGroup createLocalCPUGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("local_cpu_upi")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cpu())
                .setLocalRank(0)
                .build();
        }

        std::unique_ptr<UPICollectiveBackend> backend_null_;
    };

    // =============================================================================
    // Identity Tests
    // =============================================================================

    TEST_F(Test__UPIBackend, NameIsUPI)
    {
        EXPECT_EQ(backend_null_->name(), "UPI");
    }

    TEST_F(Test__UPIBackend, TypeIsMPI)
    {
        // UPI uses MPI internally, so type() returns MPI
        EXPECT_EQ(backend_null_->type(), CollectiveBackendType::MPI);
    }

    // =============================================================================
    // Validity Tests
    // =============================================================================

    TEST_F(Test__UPIBackend, InvalidCommReturnsInvalid)
    {
        // Backend created with MPI_COMM_NULL should report invalid
        EXPECT_FALSE(backend_null_->isValid());
        EXPECT_EQ(backend_null_->domainRank(), -1);
        EXPECT_EQ(backend_null_->domainSize(), 0);
    }

    TEST_F(Test__UPIBackend, IsNotAvailableWithNullComm)
    {
        // Without a valid communicator, backend is not available
        EXPECT_FALSE(backend_null_->isAvailable());
    }

    // =============================================================================
    // Device Support Tests
    // =============================================================================

    TEST_F(Test__UPIBackend, SupportsOnlyCPUDeviceType)
    {
        // UPI operates on host memory - only CPU is supported directly
        EXPECT_TRUE(backend_null_->supportsDevice(DeviceType::CPU));
        EXPECT_FALSE(backend_null_->supportsDevice(DeviceType::CUDA));
        EXPECT_FALSE(backend_null_->supportsDevice(DeviceType::ROCm));
    }

    TEST_F(Test__UPIBackend, SupportsDirectTransferOnlyForCPU)
    {
        // UPI can only directly transfer between CPU buffers
        DeviceId cpu1 = DeviceId::cpu();
        DeviceId cpu2 = DeviceId::cpu();
        DeviceId cuda0 = DeviceId::cuda(0);
        DeviceId rocm0 = DeviceId::rocm(0);

        // CPU ↔ CPU: supported
        EXPECT_TRUE(backend_null_->supportsDirectTransfer(cpu1, cpu2));

        // CPU ↔ GPU: not supported (requires staging)
        EXPECT_FALSE(backend_null_->supportsDirectTransfer(cpu1, cuda0));
        EXPECT_FALSE(backend_null_->supportsDirectTransfer(cuda0, cpu1));
        EXPECT_FALSE(backend_null_->supportsDirectTransfer(cpu1, rocm0));
        EXPECT_FALSE(backend_null_->supportsDirectTransfer(rocm0, cpu1));

        // GPU ↔ GPU: not supported
        EXPECT_FALSE(backend_null_->supportsDirectTransfer(cuda0, rocm0));
    }

    // =============================================================================
    // Lifecycle Tests
    // =============================================================================

    TEST_F(Test__UPIBackend, InitializeFailsWithNullComm)
    {
        auto group = createLocalCPUGroup();
        EXPECT_FALSE(backend_null_->initialize(group));
        EXPECT_FALSE(backend_null_->isInitialized());
    }

    TEST_F(Test__UPIBackend, ShutdownSucceedsEvenWithoutInit)
    {
        // Shutdown should be safe even without initialization
        EXPECT_NO_THROW(backend_null_->shutdown());
        EXPECT_FALSE(backend_null_->isInitialized());
    }

    TEST_F(Test__UPIBackend, DoubleShutdownIsSafe)
    {
        // Multiple shutdown calls should be safe
        backend_null_->shutdown();
        EXPECT_NO_THROW(backend_null_->shutdown());
    }

    // =============================================================================
    // Collective Operation Failure Tests (with null communicator)
    // =============================================================================

    TEST_F(Test__UPIBackend, AllreduceFailsWithNullComm)
    {
        float buffer[4] = {1.0f, 2.0f, 3.0f, 4.0f};

        // Try to initialize (which fails due to null comm)
        auto group = createLocalCPUGroup();
        backend_null_->initialize(group); // Will fail

        EXPECT_FALSE(backend_null_->allreduce(
            buffer, 4, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));
    }

    TEST_F(Test__UPIBackend, AllgatherFailsWithNullComm)
    {
        float send[2] = {1.0f, 2.0f};
        float recv[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        auto group = createLocalCPUGroup();
        backend_null_->initialize(group); // Will fail

        EXPECT_FALSE(backend_null_->allgather(
            send, recv, 2, CollectiveDataType::FLOAT32));
    }

    TEST_F(Test__UPIBackend, BroadcastFailsWithNullComm)
    {
        float buffer[4] = {1.0f, 2.0f, 3.0f, 4.0f};

        auto group = createLocalCPUGroup();
        backend_null_->initialize(group); // Will fail

        EXPECT_FALSE(backend_null_->broadcast(
            buffer, 4, CollectiveDataType::FLOAT32, 0));
    }

    TEST_F(Test__UPIBackend, SynchronizeFailsWithNullComm)
    {
        auto group = createLocalCPUGroup();
        backend_null_->initialize(group); // Will fail

        EXPECT_FALSE(backend_null_->synchronize());
    }

    // =============================================================================
    // Type Conversion Tests
    // =============================================================================

    TEST_F(Test__UPIBackend, ToMPITypeFP32)
    {
        MPI_Datatype dtype = UPICollectiveBackend::toMPIDatatype(CollectiveDataType::FLOAT32);
        EXPECT_EQ(dtype, MPI_FLOAT);
    }

    TEST_F(Test__UPIBackend, ToMPITypeFP16)
    {
        MPI_Datatype dtype = UPICollectiveBackend::toMPIDatatype(CollectiveDataType::FLOAT16);
        EXPECT_EQ(dtype, MPI_UINT16_T);
    }

    TEST_F(Test__UPIBackend, ToMPITypeBF16)
    {
        MPI_Datatype dtype = UPICollectiveBackend::toMPIDatatype(CollectiveDataType::BFLOAT16);
        EXPECT_EQ(dtype, MPI_UINT16_T);
    }

    TEST_F(Test__UPIBackend, ToMPITypeINT32)
    {
        MPI_Datatype dtype = UPICollectiveBackend::toMPIDatatype(CollectiveDataType::INT32);
        EXPECT_EQ(dtype, MPI_INT);
    }

    TEST_F(Test__UPIBackend, ToMPITypeINT8)
    {
        MPI_Datatype dtype = UPICollectiveBackend::toMPIDatatype(CollectiveDataType::INT8);
        EXPECT_EQ(dtype, MPI_INT8_T);
    }

    TEST_F(Test__UPIBackend, ToMPIOpSum)
    {
        MPI_Op op = UPICollectiveBackend::toMPIOp(CollectiveOp::ALLREDUCE_SUM);
        EXPECT_EQ(op, MPI_SUM);
    }

    TEST_F(Test__UPIBackend, ToMPIOpMax)
    {
        MPI_Op op = UPICollectiveBackend::toMPIOp(CollectiveOp::ALLREDUCE_MAX);
        EXPECT_EQ(op, MPI_MAX);
    }

    TEST_F(Test__UPIBackend, ToMPIOpMin)
    {
        MPI_Op op = UPICollectiveBackend::toMPIOp(CollectiveOp::ALLREDUCE_MIN);
        EXPECT_EQ(op, MPI_MIN);
    }

    // =============================================================================
    // Bandwidth Estimation Tests
    // =============================================================================

    TEST_F(Test__UPIBackend, EstimatedBandwidthIsPositive)
    {
        // Even with null comm, bandwidth estimation should work
        // (uses system detection, falls back to default)
        float bandwidth = backend_null_->estimatedBandwidthGBps();
        EXPECT_GT(bandwidth, 0.0f);
    }

    TEST_F(Test__UPIBackend, EstimatedBandwidthIsReasonable)
    {
        // Bandwidth should be in reasonable range for UPI/Infinity Fabric
        // Intel UPI: ~50 GB/s, AMD IF: ~100 GB/s
        float bandwidth = backend_null_->estimatedBandwidthGBps();
        EXPECT_GE(bandwidth, 20.0f);  // At least 20 GB/s
        EXPECT_LE(bandwidth, 200.0f); // At most 200 GB/s (future systems)
    }

    // =============================================================================
    // DomainComm Accessor Test
    // =============================================================================

    TEST_F(Test__UPIBackend, DomainCommReturnsCorrectValue)
    {
        EXPECT_EQ(backend_null_->domainComm(), MPI_COMM_NULL);
    }

} // namespace llaminar2::test
