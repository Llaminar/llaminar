/**
 * @file Test__MPIBackend.cpp
 * @brief Unit tests for MPIBackend
 *
 * Tests the MPI-based collective backend for inter-node communication.
 * These tests use nullptr mpi_ctx to test capability queries without
 * requiring actual MPI operations.
 */

#include <gtest/gtest.h>
#include "v2/collective/backends/MPIBackend.h"
#include "v2/collective/DeviceGroup.h"
#include "v2/backends/DeviceId.h"

namespace llaminar2::test
{

    // ═══════════════════════════════════════════════════════════════════════════
    // Test Fixture
    // ═══════════════════════════════════════════════════════════════════════════

    class Test__MPIBackend : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Create backend without MPI context for capability testing
            backend_ = std::make_unique<MPIBackend>(nullptr);
        }

        void TearDown() override
        {
            if (backend_ && backend_->isInitialized())
            {
                backend_->shutdown();
            }
        }

        // Helper to create a global-scope CPU group
        DeviceGroup createGlobalCPUGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("global_cpu")
                .setScope(CollectiveScope::GLOBAL)
                .addDevice(DeviceId::cpu())
                .setLocalRank(0)
                .build();
        }

        // Helper to create a local-scope CPU group
        DeviceGroup createLocalCPUGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("local_cpu")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cpu())
                .setLocalRank(0)
                .build();
        }

        std::unique_ptr<MPIBackend> backend_;
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Identity Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__MPIBackend, TypeReturnsMPI)
    {
        EXPECT_EQ(backend_->type(), CollectiveBackendType::MPI);
    }

    TEST_F(Test__MPIBackend, NameReturnsMPI)
    {
        EXPECT_EQ(backend_->name(), "MPI");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Device Support Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__MPIBackend, SupportsOnlyCPUDeviceType)
    {
        // MPI operates on host memory - only CPU is supported directly
        EXPECT_TRUE(backend_->supportsDevice(DeviceType::CPU));
        EXPECT_FALSE(backend_->supportsDevice(DeviceType::CUDA));
        EXPECT_FALSE(backend_->supportsDevice(DeviceType::ROCm));
    }

    TEST_F(Test__MPIBackend, SupportsDirectTransferOnlyForCPU)
    {
        // MPI can only directly transfer between CPU buffers
        DeviceId cpu1 = DeviceId::cpu();
        DeviceId cpu2 = DeviceId::cpu();
        DeviceId cuda0 = DeviceId::cuda(0);
        DeviceId rocm0 = DeviceId::rocm(0);

        // CPU ↔ CPU: supported
        EXPECT_TRUE(backend_->supportsDirectTransfer(cpu1, cpu2));
        
        // CPU ↔ GPU: not supported (requires staging)
        EXPECT_FALSE(backend_->supportsDirectTransfer(cpu1, cuda0));
        EXPECT_FALSE(backend_->supportsDirectTransfer(cuda0, cpu1));
        
        // GPU ↔ GPU: not supported
        EXPECT_FALSE(backend_->supportsDirectTransfer(cuda0, rocm0));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Availability Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__MPIBackend, IsNotAvailableWithoutMPIContext)
    {
        // Without MPI context, backend is not available
        EXPECT_FALSE(backend_->isAvailable());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Lifecycle Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__MPIBackend, InitializeFailsWithoutMPIContext)
    {
        auto group = createGlobalCPUGroup();
        EXPECT_FALSE(backend_->initialize(group));
        EXPECT_FALSE(backend_->isInitialized());
    }

    TEST_F(Test__MPIBackend, ShutdownSucceedsEvenWithoutInit)
    {
        // Shutdown should be safe even without initialization
        EXPECT_NO_THROW(backend_->shutdown());
        EXPECT_FALSE(backend_->isInitialized());
    }

    TEST_F(Test__MPIBackend, DoubleShutdownIsSafe)
    {
        // Multiple shutdown calls should be safe
        backend_->shutdown();
        EXPECT_NO_THROW(backend_->shutdown());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Collective Operation Failure Tests (without MPI context)
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__MPIBackend, AllreduceFailsWithoutMPIContext)
    {
        float buffer[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        
        // Even if we try to initialize (which fails), allreduce should fail
        auto group = createGlobalCPUGroup();
        backend_->initialize(group);  // Will fail
        
        EXPECT_FALSE(backend_->allreduce(
            buffer, 4, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));
    }

    TEST_F(Test__MPIBackend, AllgatherFailsWithoutMPIContext)
    {
        float send[2] = {1.0f, 2.0f};
        float recv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        
        auto group = createGlobalCPUGroup();
        backend_->initialize(group);  // Will fail
        
        EXPECT_FALSE(backend_->allgather(
            send, recv, 2, CollectiveDataType::FLOAT32));
    }

    TEST_F(Test__MPIBackend, BroadcastFailsWithoutMPIContext)
    {
        float buffer[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        
        auto group = createGlobalCPUGroup();
        backend_->initialize(group);  // Will fail
        
        EXPECT_FALSE(backend_->broadcast(
            buffer, 4, CollectiveDataType::FLOAT32, 0));
    }

    TEST_F(Test__MPIBackend, ReduceScatterFailsWithoutMPIContext)
    {
        float send[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float recv[2] = {0.0f, 0.0f};
        
        auto group = createGlobalCPUGroup();
        backend_->initialize(group);  // Will fail
        
        EXPECT_FALSE(backend_->reduceScatter(
            send, recv, 2, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));
    }

    TEST_F(Test__MPIBackend, SynchronizeFailsWithoutMPIContext)
    {
        auto group = createGlobalCPUGroup();
        backend_->initialize(group);  // Will fail
        
        EXPECT_FALSE(backend_->synchronize());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Error Reporting Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__MPIBackend, LastErrorReportsInitializationFailure)
    {
        auto group = createGlobalCPUGroup();
        backend_->initialize(group);  // Will fail
        
        std::string error = backend_->lastError();
        EXPECT_FALSE(error.empty());
        EXPECT_NE(error.find("MPI"), std::string::npos);
    }

    TEST_F(Test__MPIBackend, LastErrorReportsOperationFailure)
    {
        float buffer[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        backend_->allreduce(buffer, 4, CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM);
        
        std::string error = backend_->lastError();
        EXPECT_FALSE(error.empty());
    }

} // namespace llaminar2::test
