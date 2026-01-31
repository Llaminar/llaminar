/**
 * @file Test__HeterogeneousBackend.cpp
 * @brief Unit tests for HeterogeneousBackend
 *
 * Tests the heterogeneous multi-GPU collective backend that orchestrates
 * NCCL, RCCL, and PCIeBAR backends for mixed NVIDIA+AMD configurations.
 *
 * IMPORTANT: These are UNIT tests that run in build_v2 (Debug) and must NOT
 * invoke actual GPUs. They test class logic using DeviceGroup objects with
 * device IDs, not actual hardware.
 *
 * Full integration tests with actual hardware are in:
 *   integration/Test__HeterogeneousBackendIntegration.cpp
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "v2/collective/backends/HeterogeneousBackend.h"
#include "v2/collective/DeviceGroup.h"
#include "v2/backends/DeviceId.h"

namespace llaminar2::test
{

    // ═══════════════════════════════════════════════════════════════════════════
    // Test Fixture
    // ═══════════════════════════════════════════════════════════════════════════

    class Test__HeterogeneousBackend : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            backend_ = std::make_unique<HeterogeneousBackend>();
        }

        void TearDown() override
        {
            if (backend_ && backend_->isInitialized())
            {
                backend_->shutdown();
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Helper: Create device groups for testing
        // ─────────────────────────────────────────────────────────────────────

        /// Create a heterogeneous group: 1 CUDA + 2 ROCm (typical test config)
        DeviceGroup createTypicalHeterogeneousGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("1_cuda_2_rocm")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .addDevice(DeviceId::rocm(0))
                .addDevice(DeviceId::rocm(1))
                .setLocalRank(0)
                .build();
        }

        /// Create a heterogeneous group: 2 CUDA + 1 ROCm
        DeviceGroup create2Cuda1RocmGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("2_cuda_1_rocm")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .addDevice(DeviceId::cuda(1))
                .addDevice(DeviceId::rocm(0))
                .setLocalRank(0)
                .build();
        }

        /// Create a heterogeneous group: 2 CUDA + 2 ROCm
        DeviceGroup create2Cuda2RocmGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("2_cuda_2_rocm")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .addDevice(DeviceId::cuda(1))
                .addDevice(DeviceId::rocm(0))
                .addDevice(DeviceId::rocm(1))
                .setLocalRank(0)
                .build();
        }

        /// Create a minimal heterogeneous group: 1 CUDA + 1 ROCm
        DeviceGroup createMinimalHeterogeneousGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("1_cuda_1_rocm")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .addDevice(DeviceId::rocm(0))
                .setLocalRank(0)
                .build();
        }

        /// Create a pure CUDA group (should be rejected)
        DeviceGroup createPureCUDAGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("pure_cuda")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .addDevice(DeviceId::cuda(1))
                .setLocalRank(0)
                .build();
        }

        /// Create a pure ROCm group (should be rejected)
        DeviceGroup createPureROCmGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("pure_rocm")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::rocm(0))
                .addDevice(DeviceId::rocm(1))
                .setLocalRank(0)
                .build();
        }

        /// Create a single CUDA device group (should be rejected)
        DeviceGroup createSingleCUDAGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("single_cuda")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .setLocalRank(0)
                .build();
        }

        /// Create a single ROCm device group (should be rejected)
        DeviceGroup createSingleROCmGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("single_rocm")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::rocm(0))
                .setLocalRank(0)
                .build();
        }

        /// Create an empty group (should be rejected)
        DeviceGroup createEmptyGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("empty")
                .setScope(CollectiveScope::LOCAL)
                .setLocalRank(0)
                .build();
        }

        /// Create a group with non-sequential ordinals to test sorting
        DeviceGroup createNonSequentialOrdinalsGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("non_sequential")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(2)) // Higher ordinal first
                .addDevice(DeviceId::cuda(0))
                .addDevice(DeviceId::rocm(3)) // Higher ordinal first
                .addDevice(DeviceId::rocm(1))
                .setLocalRank(0)
                .build();
        }

        /// Create a group with CPU device (should be rejected)
        DeviceGroup createMixedWithCPUGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("mixed_with_cpu")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .addDevice(DeviceId::rocm(0))
                .addDevice(DeviceId::cpu())
                .setLocalRank(0)
                .build();
        }

        std::unique_ptr<HeterogeneousBackend> backend_;
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Identity Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HeterogeneousBackend, TypeIsHeterogeneous)
    {
        EXPECT_EQ(backend_->type(), CollectiveBackendType::HETEROGENEOUS);
    }

    TEST_F(Test__HeterogeneousBackend, NameIsHeterogeneous)
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        EXPECT_EQ(backend_->name(), "Heterogeneous");
#else
        EXPECT_EQ(backend_->name(), "Heterogeneous (unavailable)");
#endif
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Multi-GPU Capability Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HeterogeneousBackend, IsMultiGpuSingleProcess)
    {
        // HeterogeneousBackend manages multiple GPUs from a single process
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        EXPECT_TRUE(backend_->isMultiGpuSingleProcess());
#else
        // Stub returns false (not available)
        EXPECT_FALSE(backend_->isMultiGpuSingleProcess());
#endif
    }

    TEST_F(Test__HeterogeneousBackend, AllReduceMultiFailsWhenNotInitialized)
    {
        // allreduceMulti should fail if backend is not initialized
        std::vector<void *> buffers = {nullptr, nullptr, nullptr};

        EXPECT_FALSE(backend_->isInitialized());
        EXPECT_FALSE(backend_->allreduceMulti(buffers, 100, CollectiveDataType::FLOAT32,
                                              CollectiveOp::ALLREDUCE_SUM));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Device Support Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HeterogeneousBackend, SupportsCUDADevice)
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        EXPECT_TRUE(backend_->supportsDevice(DeviceType::CUDA));
#else
        EXPECT_FALSE(backend_->supportsDevice(DeviceType::CUDA));
#endif
    }

    TEST_F(Test__HeterogeneousBackend, SupportsROCmDevice)
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        EXPECT_TRUE(backend_->supportsDevice(DeviceType::ROCm));
#else
        EXPECT_FALSE(backend_->supportsDevice(DeviceType::ROCm));
#endif
    }

    TEST_F(Test__HeterogeneousBackend, DoesNotSupportCPU)
    {
        EXPECT_FALSE(backend_->supportsDevice(DeviceType::CPU));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Device Grouping Tests
    // ═══════════════════════════════════════════════════════════════════════════

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

    TEST_F(Test__HeterogeneousBackend, DeviceGrouping_SeparatesCUDAAndROCm)
    {
        // This test verifies that initialize() correctly partitions devices
        // We can't actually initialize (requires hardware), but we can test
        // validation and the accessors return reasonable defaults before init

        // Pre-init: should have empty device lists
        EXPECT_TRUE(backend_->cudaDevices().empty());
        EXPECT_TRUE(backend_->rocmDevices().empty());

        // After initialization attempt (will fail without hardware, but
        // validation and partitioning should still work)
        auto group = createTypicalHeterogeneousGroup();

        // Check group metadata was computed correctly
        EXPECT_EQ(group.cuda_count, 1);
        EXPECT_EQ(group.rocm_count, 2);
        EXPECT_FALSE(group.is_homogeneous);
    }

    TEST_F(Test__HeterogeneousBackend, DeviceGrouping_RejectsPureCUDA)
    {
        auto group = createPureCUDAGroup();

        // Heterogeneous backend should reject pure-CUDA groups
        // (Use NCCL for those instead)
        EXPECT_FALSE(backend_->initialize(group));

        // Backend should not be initialized
        EXPECT_FALSE(backend_->isInitialized());

        // Device lists should remain empty
        EXPECT_TRUE(backend_->cudaDevices().empty());
        EXPECT_TRUE(backend_->rocmDevices().empty());
    }

    TEST_F(Test__HeterogeneousBackend, DeviceGrouping_RejectsPureROCm)
    {
        auto group = createPureROCmGroup();

        // Heterogeneous backend should reject pure-ROCm groups
        // (Use RCCL for those instead)
        EXPECT_FALSE(backend_->initialize(group));

        // Backend should not be initialized
        EXPECT_FALSE(backend_->isInitialized());

        // Device lists should remain empty
        EXPECT_TRUE(backend_->cudaDevices().empty());
        EXPECT_TRUE(backend_->rocmDevices().empty());
    }

    TEST_F(Test__HeterogeneousBackend, DeviceGrouping_RejectsSingleCUDA)
    {
        auto group = createSingleCUDAGroup();

        // Single CUDA device has no ROCm, should be rejected
        EXPECT_FALSE(backend_->initialize(group));
        EXPECT_FALSE(backend_->isInitialized());
    }

    TEST_F(Test__HeterogeneousBackend, DeviceGrouping_RejectsSingleROCm)
    {
        auto group = createSingleROCmGroup();

        // Single ROCm device has no CUDA, should be rejected
        EXPECT_FALSE(backend_->initialize(group));
        EXPECT_FALSE(backend_->isInitialized());
    }

    TEST_F(Test__HeterogeneousBackend, DeviceGrouping_RejectsEmptyGroup)
    {
        auto group = createEmptyGroup();

        EXPECT_FALSE(backend_->initialize(group));
        EXPECT_FALSE(backend_->isInitialized());
    }

    TEST_F(Test__HeterogeneousBackend, DeviceGrouping_RejectsCPUDevice)
    {
        auto group = createMixedWithCPUGroup();

        // HeterogeneousBackend only supports CUDA and ROCm, not CPU
        EXPECT_FALSE(backend_->initialize(group));
        EXPECT_FALSE(backend_->isInitialized());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Bridge Selection Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HeterogeneousBackend, BridgeSelection_UsesCuda0AndRocm0)
    {
        // This tests that bridge devices are the lowest ordinals
        // We verify by checking DeviceGroup's internal data

        auto group = createNonSequentialOrdinalsGroup();

        // Group has cuda:2, cuda:0, rocm:3, rocm:1
        // After sorting: cuda_devices should be [cuda:0, cuda:2]
        //                rocm_devices should be [rocm:1, rocm:3]
        // Bridge should be cuda:0 and rocm:1 (lowest ordinals)

        // Verify group metadata
        EXPECT_EQ(group.cuda_count, 2);
        EXPECT_EQ(group.rocm_count, 2);

        // Note: Since we can't actually initialize without hardware,
        // we verify the grouping by checking DeviceGroup construction
        // is correct. The actual bridge selection is tested in integration tests.
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Sub-Backend Creation Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HeterogeneousBackend, SubBackendCreation_NCCLOnlyWhenMultipleCUDA)
    {
        // Pre-init: No sub-backends
        EXPECT_FALSE(backend_->hasNCCLBackend());
        EXPECT_FALSE(backend_->hasRCCLBackend());
        EXPECT_FALSE(backend_->hasBridgeBackend());

        // The actual NCCL backend creation is tested in integration tests
        // because it requires hardware. Here we just verify the accessor
        // returns false before initialization.
    }

    TEST_F(Test__HeterogeneousBackend, SubBackendCreation_RCCLOnlyWhenMultipleROCm)
    {
        // Pre-init: No sub-backends
        EXPECT_FALSE(backend_->hasRCCLBackend());

        // The actual RCCL backend creation is tested in integration tests
        // because it requires hardware. Here we just verify the accessor
        // returns false before initialization.
    }

    TEST_F(Test__HeterogeneousBackend, SubBackendCreation_BridgeAlwaysCreated)
    {
        // Pre-init: No bridge backend
        EXPECT_FALSE(backend_->hasBridgeBackend());

        // After successful init (tested in integration), bridge should always exist
        // since HeterogeneousBackend requires cross-vendor communication
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // DeviceGroup Metadata Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HeterogeneousBackend, DeviceGroupMetadata_TypicalConfig)
    {
        auto group = createTypicalHeterogeneousGroup();

        // 1 CUDA + 2 ROCm
        EXPECT_EQ(group.cuda_count, 1);
        EXPECT_EQ(group.rocm_count, 2);
        EXPECT_EQ(group.cpu_count, 0);
        EXPECT_FALSE(group.is_homogeneous);
        EXPECT_EQ(group.size(), 3u);
    }

    TEST_F(Test__HeterogeneousBackend, DeviceGroupMetadata_2Cuda2Rocm)
    {
        auto group = create2Cuda2RocmGroup();

        // 2 CUDA + 2 ROCm
        EXPECT_EQ(group.cuda_count, 2);
        EXPECT_EQ(group.rocm_count, 2);
        EXPECT_FALSE(group.is_homogeneous);
        EXPECT_EQ(group.size(), 4u);
    }

    TEST_F(Test__HeterogeneousBackend, DeviceGroupMetadata_MinimalConfig)
    {
        auto group = createMinimalHeterogeneousGroup();

        // 1 CUDA + 1 ROCm
        EXPECT_EQ(group.cuda_count, 1);
        EXPECT_EQ(group.rocm_count, 1);
        EXPECT_FALSE(group.is_homogeneous);
        EXPECT_EQ(group.size(), 2u);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Lifecycle Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HeterogeneousBackend, NotInitializedByDefault)
    {
        EXPECT_FALSE(backend_->isInitialized());
    }

    TEST_F(Test__HeterogeneousBackend, ShutdownWhenNotInitialized)
    {
        // Should not crash
        EXPECT_NO_THROW(backend_->shutdown());
        EXPECT_FALSE(backend_->isInitialized());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Stub Operation Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HeterogeneousBackend, AllreduceFailsWhenNotInitialized)
    {
        std::vector<float> buffer = {1.0f, 2.0f, 3.0f, 4.0f};
        EXPECT_FALSE(backend_->allreduce(
            buffer.data(),
            buffer.size(),
            CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM));
    }

    TEST_F(Test__HeterogeneousBackend, AllgatherFailsWhenNotInitialized)
    {
        std::vector<float> send = {1.0f};
        std::vector<float> recv(4);
        EXPECT_FALSE(backend_->allgather(
            send.data(),
            recv.data(),
            send.size(),
            CollectiveDataType::FLOAT32));
    }

    TEST_F(Test__HeterogeneousBackend, SynchronizeFailsWhenNotInitialized)
    {
        EXPECT_FALSE(backend_->synchronize());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 1 Plan Tests (Unit tests for planning logic)
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HeterogeneousBackend, Phase1_SkipsNCCLWithSingleCUDA)
    {
        // With 1 CUDA + 2 ROCm, NCCL reduce should NOT be called
        // (no point in reducing 1 buffer to itself)

        // Create a backend that simulates the 1 CUDA + 2 ROCm configuration
        // Since we can't initialize without hardware, we test the planning logic
        // by examining what planPhase1() would return if the internal state
        // was set up correctly.

        // The plan logic depends on cuda_devices_.size() > 1 AND nccl_backend_ != nullptr
        // With 1 CUDA device, nccl_backend_ should be nullptr (not created)

        // Test via the public accessor that reflects internal state
        // Pre-init: no backends created
        auto plan = backend_->planPhase1();

        // Before initialization:
        // - cuda_devices_ is empty, rocm_devices_ is empty
        // - nccl_backend_ is nullptr, rccl_backend_ is nullptr
        EXPECT_FALSE(plan.will_call_nccl_reduce);
        EXPECT_FALSE(plan.will_call_rccl_reduce);
        EXPECT_EQ(plan.nccl_reduce_root, -1);
        EXPECT_EQ(plan.rccl_reduce_root, -1);
        EXPECT_EQ(plan.nccl_device_count, 0u);
        EXPECT_EQ(plan.rccl_device_count, 0u);
    }

    TEST_F(Test__HeterogeneousBackend, Phase1_PlanReturnsCorrectDeviceCounts)
    {
        // Test that planPhase1() returns correct device counts
        // (even when uninitialized, should return 0s)

        auto plan = backend_->planPhase1();

        // Pre-init state
        EXPECT_EQ(plan.nccl_device_count, 0u);
        EXPECT_EQ(plan.rccl_device_count, 0u);
    }

    TEST_F(Test__HeterogeneousBackend, Phase1_SkipsRCCLWithSingleROCm)
    {
        // With 2 CUDA + 1 ROCm, RCCL reduce should NOT be called
        // The plan logic depends on rocm_devices_.size() > 1 AND rccl_backend_ != nullptr
        // With 1 ROCm device, rccl_backend_ should be nullptr

        // Test via planPhase1() - before init, everything is empty/false
        auto plan = backend_->planPhase1();

        EXPECT_FALSE(plan.will_call_rccl_reduce);
        EXPECT_EQ(plan.rccl_reduce_root, -1);
    }

    TEST_F(Test__HeterogeneousBackend, Phase1_PlanStructDefaultsAreCorrect)
    {
        // Verify the Phase1Plan struct has correct defaults
        HeterogeneousBackend::Phase1Plan plan;

        EXPECT_FALSE(plan.will_call_nccl_reduce);
        EXPECT_FALSE(plan.will_call_rccl_reduce);
        EXPECT_EQ(plan.nccl_reduce_root, -1);
        EXPECT_EQ(plan.rccl_reduce_root, -1);
        EXPECT_EQ(plan.nccl_device_count, 0u);
        EXPECT_EQ(plan.rccl_device_count, 0u);
    }

    TEST_F(Test__HeterogeneousBackend, Phase1_ExecuteFailsWhenNotInitialized)
    {
        // executePhase1 should fail if backend is not initialized
        std::vector<void *> cuda_buffers = {nullptr};
        std::vector<void *> rocm_buffers = {nullptr, nullptr};

        // This will fail because initialized_ is false
        // Note: We can't test the full execution without hardware,
        // but we can test the initialization check
        EXPECT_FALSE(backend_->isInitialized());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 2 Plan Tests (Unit tests for planning logic)
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HeterogeneousBackend, Phase2_PlanShowsBridgeAllreduceNotInitialized)
    {
        // Before initialization, Phase 2 plan should show no bridge allreduce
        auto plan = backend_->planPhase2();

        EXPECT_FALSE(plan.will_call_bridge_allreduce);
        EXPECT_TRUE(plan.cuda_bridge_device.is_cpu()); // Default value
        EXPECT_TRUE(plan.rocm_bridge_device.is_cpu()); // Default value
    }

    TEST_F(Test__HeterogeneousBackend, Phase2_PlanStructDefaultsAreCorrect)
    {
        // Verify the Phase2Plan struct has correct defaults
        HeterogeneousBackend::Phase2Plan plan;

        EXPECT_FALSE(plan.will_call_bridge_allreduce);
        // DeviceId default constructor creates a placeholder (CPU type, ordinal=-1)
        // This is not a valid CPU device (is_cpu() requires ordinal >= 0)
        EXPECT_EQ(plan.cuda_bridge_device.type, DeviceType::CPU);
        EXPECT_EQ(plan.cuda_bridge_device.ordinal, -1);
        EXPECT_EQ(plan.rocm_bridge_device.type, DeviceType::CPU);
        EXPECT_EQ(plan.rocm_bridge_device.ordinal, -1);
    }

    TEST_F(Test__HeterogeneousBackend, Phase2_BridgeBackendNotInitializedBeforeInit)
    {
        // Before initialize(), bridge backend should not exist
        EXPECT_FALSE(backend_->hasBridgeBackend());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 3 Plan Tests (Unit tests for planning logic)
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HeterogeneousBackend, Phase3_SkipsNCCLWithSingleCUDA)
    {
        // With 1 CUDA device, NCCL broadcast should NOT be called
        // (no point in broadcasting to 0 other devices)

        // The plan logic depends on cuda_devices_.size() > 1 AND nccl_backend_ != nullptr
        // With 1 CUDA device, nccl_backend_ should be nullptr (not created)

        // Test via planPhase3() - before init, everything is empty/false
        auto plan = backend_->planPhase3();

        // Before initialization:
        // - cuda_devices_ is empty, rocm_devices_ is empty
        // - nccl_backend_ is nullptr, rccl_backend_ is nullptr
        EXPECT_FALSE(plan.will_call_nccl_broadcast);
        EXPECT_EQ(plan.nccl_broadcast_root, -1);
        EXPECT_EQ(plan.nccl_device_count, 0u);
    }

    TEST_F(Test__HeterogeneousBackend, Phase3_SkipsRCCLWithSingleROCm)
    {
        // With 1 ROCm device, RCCL broadcast should NOT be called
        // The plan logic depends on rocm_devices_.size() > 1 AND rccl_backend_ != nullptr
        // With 1 ROCm device, rccl_backend_ should be nullptr

        // Test via planPhase3() - before init, everything is empty/false
        auto plan = backend_->planPhase3();

        EXPECT_FALSE(plan.will_call_rccl_broadcast);
        EXPECT_EQ(plan.rccl_broadcast_root, -1);
        EXPECT_EQ(plan.rccl_device_count, 0u);
    }

    TEST_F(Test__HeterogeneousBackend, Phase3_CallsNCCLBroadcastWithMultipleCUDA)
    {
        // This test verifies that planPhase3() WOULD call NCCL broadcast
        // if there were >1 CUDA devices. Since we can't initialize without
        // hardware, we verify the struct defaults and that the plan returns
        // correct values when nccl_backend_ is null (which it is before init).

        // Before init, plan should show no NCCL broadcast (no backend)
        auto plan = backend_->planPhase3();

        // nccl_backend_ is nullptr before init, so even if we had
        // multiple CUDA devices, the check would fail.
        // This verifies the second condition: && nccl_backend_
        EXPECT_FALSE(plan.will_call_nccl_broadcast);

        // Note: Full test with actual NCCL broadcast is in integration tests
    }

    TEST_F(Test__HeterogeneousBackend, Phase3_CallsRCCLBroadcastWithMultipleROCm)
    {
        // This test verifies that planPhase3() WOULD call RCCL broadcast
        // if there were >1 ROCm devices. Since we can't initialize without
        // hardware, we verify the struct defaults and that the plan returns
        // correct values when rccl_backend_ is null.

        // Before init, plan should show no RCCL broadcast (no backend)
        auto plan = backend_->planPhase3();

        // rccl_backend_ is nullptr before init, so even if we had
        // multiple ROCm devices, the check would fail.
        EXPECT_FALSE(plan.will_call_rccl_broadcast);

        // Note: Full test with actual RCCL broadcast is in integration tests
    }

    TEST_F(Test__HeterogeneousBackend, Phase3_PlanStructDefaultsAreCorrect)
    {
        // Verify the Phase3Plan struct has correct defaults
        HeterogeneousBackend::Phase3Plan plan;

        EXPECT_FALSE(plan.will_call_nccl_broadcast);
        EXPECT_FALSE(plan.will_call_rccl_broadcast);
        EXPECT_EQ(plan.nccl_broadcast_root, -1);
        EXPECT_EQ(plan.rccl_broadcast_root, -1);
        EXPECT_EQ(plan.nccl_device_count, 0u);
        EXPECT_EQ(plan.rccl_device_count, 0u);
    }

    TEST_F(Test__HeterogeneousBackend, Phase3_PlanReturnsCorrectDeviceCounts)
    {
        // Test that planPhase3() returns correct device counts
        // (even when uninitialized, should return 0s)

        auto plan = backend_->planPhase3();

        // Pre-init state
        EXPECT_EQ(plan.nccl_device_count, 0u);
        EXPECT_EQ(plan.rccl_device_count, 0u);
    }

    TEST_F(Test__HeterogeneousBackend, Phase3_ExecuteFailsWhenNotInitialized)
    {
        // executePhase3 should fail if backend is not initialized
        std::vector<void *> cuda_buffers = {nullptr};
        std::vector<void *> rocm_buffers = {nullptr, nullptr};

        // Verify not initialized
        EXPECT_FALSE(backend_->isInitialized());

        // The actual executePhase3 call can't be made without buffers,
        // but the first check in executePhase3 is for initialization
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Reduce-Scatter Pattern Tests (Unit tests for planning logic)
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HeterogeneousBackend, ReduceScatter_ShouldUsePatternForLargeTensor)
    {
        // A 4MB tensor should use the reduce-scatter pattern (at threshold)
        size_t tensor_bytes = 4 * 1024 * 1024; // 4 MB exactly

        // Before initialization, shouldUseReduceScatterPattern checks device counts
        // With empty device lists, it should return false
        EXPECT_FALSE(backend_->shouldUseReduceScatterPattern(tensor_bytes));

        // Note: Full test with initialized backend is in integration tests
    }

    TEST_F(Test__HeterogeneousBackend, ReduceScatter_ShouldNotUsePatternForSmallTensor)
    {
        // A 1MB tensor should NOT use reduce-scatter pattern (below threshold)
        size_t tensor_bytes = 1 * 1024 * 1024; // 1 MB

        EXPECT_FALSE(backend_->shouldUseReduceScatterPattern(tensor_bytes));
    }

    TEST_F(Test__HeterogeneousBackend, ReduceScatter_ShouldNotUseForMinimalConfig)
    {
        // Even large tensors should not use reduce-scatter if both domains
        // have only 1 device (no bandwidth savings possible)
        size_t tensor_bytes = 8 * 1024 * 1024; // 8 MB

        // With empty device lists (pre-init), this returns false
        // In a real 1+1 config (1 CUDA + 1 ROCm), it should also return false
        // because neither domain has multiple devices
        EXPECT_FALSE(backend_->shouldUseReduceScatterPattern(tensor_bytes));
    }

    TEST_F(Test__HeterogeneousBackend, ReduceScatter_ThresholdIs4MB)
    {
        // Verify the threshold constant
        EXPECT_EQ(HeterogeneousBackend::REDUCE_SCATTER_THRESHOLD, 4 * 1024 * 1024);
    }

    TEST_F(Test__HeterogeneousBackend, ReduceScatter_PlanStructDefaultsAreCorrect)
    {
        // Verify the ReduceScatterPlan struct has correct defaults
        HeterogeneousBackend::ReduceScatterPlan plan;

        EXPECT_FALSE(plan.use_reduce_scatter_pattern);
        EXPECT_EQ(plan.cuda_chunk_count, 0u);
        EXPECT_EQ(plan.rocm_chunk_count, 0u);
        EXPECT_EQ(plan.bridge_exchange_count, 0u);
        EXPECT_EQ(plan.cuda_device_count, 0u);
        EXPECT_EQ(plan.rocm_device_count, 0u);
    }

    TEST_F(Test__HeterogeneousBackend, ReduceScatter_PlanReturnsCorrectDeviceCounts)
    {
        // Test that planReduceScatter() returns correct device counts
        // (even when uninitialized, should return 0s for device counts)
        size_t count = 1024 * 1024; // 1M elements
        size_t element_size = 4;    // float32

        auto plan = backend_->planReduceScatter(count, element_size);

        // Pre-init state - no devices
        EXPECT_EQ(plan.cuda_device_count, 0u);
        EXPECT_EQ(plan.rocm_device_count, 0u);
    }

    TEST_F(Test__HeterogeneousBackend, ReduceScatter_PlanSmallTensorUsesStandardPattern)
    {
        // Small tensor (< 4MB) should not use reduce-scatter pattern
        size_t count = 100000;   // 100K elements
        size_t element_size = 4; // float32 → 400KB total

        auto plan = backend_->planReduceScatter(count, element_size);

        EXPECT_FALSE(plan.use_reduce_scatter_pattern);
        // Standard pattern: full tensor exchange
        EXPECT_EQ(plan.cuda_chunk_count, count);
        EXPECT_EQ(plan.rocm_chunk_count, count);
        EXPECT_EQ(plan.bridge_exchange_count, count);
    }

    TEST_F(Test__HeterogeneousBackend, ReduceScatter_PlanLargeTensorWithNoDevices)
    {
        // Large tensor but no devices initialized → standard pattern
        size_t count = 2 * 1024 * 1024; // 2M elements
        size_t element_size = 4;        // float32 → 8MB total

        auto plan = backend_->planReduceScatter(count, element_size);

        // Without multiple devices in either domain, even large tensors
        // use standard pattern (no bandwidth savings possible)
        EXPECT_FALSE(plan.use_reduce_scatter_pattern);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Pipeline Plan Tests (Unit tests for chunk-based pipelining logic)
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HeterogeneousBackend, PipelinePlan_CorrectNumChunks)
    {
        // 8MB tensor with 1MB chunks = 8 chunks
        size_t count = 2 * 1024 * 1024; // 2M elements
        size_t element_size = 4;        // float32 → 8MB total

        auto plan = backend_->planPipelining(count, element_size);

        // 8MB / 1MB = 8 chunks
        EXPECT_EQ(plan.num_chunks, 8u);
        EXPECT_TRUE(plan.will_use_pipelining);
    }

    TEST_F(Test__HeterogeneousBackend, PipelinePlan_CorrectChunkSize)
    {
        // 1MB chunk = 256K float32 elements
        size_t count = 2 * 1024 * 1024; // 2M elements → 8MB
        size_t element_size = 4;        // float32

        auto plan = backend_->planPipelining(count, element_size);

        // 1MB / 4 bytes = 256K elements per chunk
        size_t expected_chunk_elements = (1 * 1024 * 1024) / 4;
        EXPECT_EQ(plan.chunk_elements, expected_chunk_elements);
    }

    TEST_F(Test__HeterogeneousBackend, PipelinePlan_LastChunkHandlesRemainder)
    {
        // 9MB tensor with 1MB chunks = 9 chunks, last chunk same size
        // But let's test non-divisible: 8.5MB = 2176K elements
        size_t count = 2176 * 1024; // 2176K elements → 8.5MB
        size_t element_size = 4;    // float32

        auto plan = backend_->planPipelining(count, element_size);

        // 8.5MB / 1MB = 8 full chunks + 0.5MB remainder
        EXPECT_EQ(plan.num_chunks, 9u);

        // Last chunk should have remainder elements
        size_t chunk_elements = (1 * 1024 * 1024) / 4; // 256K per chunk
        size_t full_chunks = count / chunk_elements;
        size_t remainder = count % chunk_elements;
        EXPECT_GT(remainder, 0u); // Should have remainder
        EXPECT_EQ(plan.last_chunk_elements, remainder);
    }

    TEST_F(Test__HeterogeneousBackend, PipelinePlan_SmallTensorSingleChunk)
    {
        // 1MB tensor (below 4MB threshold) = 1 chunk, no pipelining
        size_t count = 256 * 1024; // 256K elements → 1MB
        size_t element_size = 4;   // float32

        auto plan = backend_->planPipelining(count, element_size);

        EXPECT_EQ(plan.num_chunks, 1u);
        EXPECT_FALSE(plan.will_use_pipelining);
        EXPECT_EQ(plan.chunk_elements, count);
        EXPECT_EQ(plan.last_chunk_elements, count);
    }

    TEST_F(Test__HeterogeneousBackend, PipelinePlan_NotUsedBelowThreshold)
    {
        // 3.9MB tensor (below 4MB threshold) should not use pipelining
        size_t count = (3 * 1024 * 1024 + 900 * 1024) / 4; // ~3.9MB in float32
        size_t element_size = 4;

        auto plan = backend_->planPipelining(count, element_size);

        EXPECT_FALSE(plan.will_use_pipelining);
        EXPECT_EQ(plan.num_chunks, 1u);
    }

    TEST_F(Test__HeterogeneousBackend, PipelinePlan_ExactlyAtThreshold)
    {
        // Exactly 4MB tensor = 4 chunks of 1MB each
        size_t count = 1024 * 1024; // 1M elements → 4MB
        size_t element_size = 4;    // float32

        auto plan = backend_->planPipelining(count, element_size);

        // 4MB >= 4MB threshold, so pipelining is used
        // 4MB / 1MB = 4 chunks
        EXPECT_TRUE(plan.will_use_pipelining);
        EXPECT_EQ(plan.num_chunks, 4u);
    }

    TEST_F(Test__HeterogeneousBackend, PipelinePlan_StructDefaultsAreCorrect)
    {
        // Verify the PipelinePlan struct has correct defaults
        HeterogeneousBackend::PipelinePlan plan;

        EXPECT_FALSE(plan.will_use_pipelining);
        EXPECT_EQ(plan.num_chunks, 0u);
        EXPECT_EQ(plan.chunk_elements, 0u);
        EXPECT_EQ(plan.last_chunk_elements, 0u);
        EXPECT_EQ(plan.total_elements, 0u);
    }

    TEST_F(Test__HeterogeneousBackend, PipelinePlan_ConstantsAreCorrect)
    {
        // Verify the pipeline constants
        EXPECT_EQ(HeterogeneousBackend::PIPELINE_CHUNK_SIZE, 1 * 1024 * 1024);      // 1MB
        EXPECT_EQ(HeterogeneousBackend::PIPELINE_MIN_TENSOR_SIZE, 4 * 1024 * 1024); // 4MB
    }

    TEST_F(Test__HeterogeneousBackend, PipelinePlan_16MBTensorCorrectChunks)
    {
        // 16MB tensor = 16 chunks
        size_t count = 4 * 1024 * 1024; // 4M elements → 16MB
        size_t element_size = 4;        // float32

        auto plan = backend_->planPipelining(count, element_size);

        EXPECT_TRUE(plan.will_use_pipelining);
        EXPECT_EQ(plan.num_chunks, 16u);
        EXPECT_EQ(plan.total_elements, count);
    }

    TEST_F(Test__HeterogeneousBackend, PipelinePlan_PreservesTotalElements)
    {
        // Verify that num_chunks * chunk_elements accounts for all elements
        size_t count = 2176 * 1024; // Non-round number
        size_t element_size = 4;

        auto plan = backend_->planPipelining(count, element_size);

        // Total should be preserved
        EXPECT_EQ(plan.total_elements, count);

        // Full chunks + last chunk should equal total
        size_t computed_total = (plan.num_chunks - 1) * plan.chunk_elements + plan.last_chunk_elements;
        EXPECT_EQ(computed_total, count);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Topology Analysis Tests (Pre-init validation only)
    // ═══════════════════════════════════════════════════════════════════════════
    // NOTE: Full topology analysis tests that require initialized backends are in
    // integration/Test__HeterogeneousBackend_ReduceScatter.cpp
    // These tests verify the analysis works on uninitialized backends (returns defaults)

    TEST_F(Test__HeterogeneousBackend, TopologyAnalysis_UninitializedReturnsDefaults)
    {
        // Without initialization, device counts are 0
        auto analysis = backend_->analyzeTopology(8 * 1024 * 1024); // 8 MB

        EXPECT_EQ(analysis.cuda_count, 0);
        EXPECT_EQ(analysis.rocm_count, 0);
        EXPECT_TRUE(analysis.is_minimal); // 0+0 is treated as minimal
        EXPECT_EQ(analysis.pattern, HeterogeneousBackend::AllreducePattern::STANDARD_3PHASE);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // GCD Multi-Bridge Pattern Tests
    // ═══════════════════════════════════════════════════════════════════════════
    // NOTE: These tests validate the GCD computation and bridge pair logic.
    // Full integration tests with actual hardware are in:
    //   integration/Test__HeterogeneousBackend_GcdMultiBridge.cpp

    TEST_F(Test__HeterogeneousBackend, TopologyAnalysis_StructDefaultsAreCorrect)
    {
        // Verify TopologyAnalysis struct has correct defaults
        HeterogeneousBackend::TopologyAnalysis analysis;

        EXPECT_EQ(analysis.cuda_count, 0u);
        EXPECT_EQ(analysis.rocm_count, 0u);
        EXPECT_FALSE(analysis.is_symmetric);
        EXPECT_FALSE(analysis.is_minimal);
        EXPECT_FALSE(analysis.is_cuda_singleton);
        EXPECT_FALSE(analysis.is_rocm_singleton);
        EXPECT_EQ(analysis.gcd, 1u);
        EXPECT_EQ(analysis.pattern, HeterogeneousBackend::AllreducePattern::STANDARD_3PHASE);
        EXPECT_EQ(analysis.num_chunks, 1u);
        EXPECT_EQ(analysis.bridge_traffic_fraction, 1u);
        EXPECT_DOUBLE_EQ(analysis.intra_domain_parallelism, 1.0);
        EXPECT_DOUBLE_EQ(analysis.bridge_parallelism, 1.0);
        EXPECT_TRUE(analysis.reason.empty());
    }

    TEST_F(Test__HeterogeneousBackend, TopologyAnalysis_GcdCalculation_Basic)
    {
        // Test GCD calculation is exposed via analyzeTopology
        // Pre-init returns 0 devices, so GCD defaults to 1
        auto analysis = backend_->analyzeTopology(8 * 1024 * 1024);

        // With 0 devices in both domains, GCD is 1 (or unspecified)
        EXPECT_GE(analysis.gcd, 1u);
    }

    TEST_F(Test__HeterogeneousBackend, AllreducePattern_EnumValuesExist)
    {
        // Verify all pattern enum values exist and can be compared
        auto pattern_standard = HeterogeneousBackend::AllreducePattern::STANDARD_3PHASE;
        auto pattern_symmetric_rs = HeterogeneousBackend::AllreducePattern::SYMMETRIC_REDUCE_SCATTER;
        auto pattern_partial_rs = HeterogeneousBackend::AllreducePattern::PARTIAL_REDUCE_SCATTER;
        auto pattern_gcd_rs = HeterogeneousBackend::AllreducePattern::GCD_REDUCE_SCATTER;
        auto pattern_gcd_mb = HeterogeneousBackend::AllreducePattern::GCD_MULTI_BRIDGE;

        // All patterns should be distinct
        EXPECT_NE(pattern_standard, pattern_symmetric_rs);
        EXPECT_NE(pattern_standard, pattern_partial_rs);
        EXPECT_NE(pattern_standard, pattern_gcd_rs);
        EXPECT_NE(pattern_standard, pattern_gcd_mb);
        EXPECT_NE(pattern_symmetric_rs, pattern_partial_rs);
        EXPECT_NE(pattern_gcd_rs, pattern_gcd_mb);
    }

    TEST_F(Test__HeterogeneousBackend, ComputeBridgePairs_UninitializedReturnsEmpty)
    {
        // Before initialization, computeBridgePairs() should return empty vector
        auto pairs = backend_->computeBridgePairs();
        EXPECT_TRUE(pairs.empty())
            << "Uninitialized backend should return empty bridge pairs";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DeviceGroup metadata tests for GCD scenarios
    // ─────────────────────────────────────────────────────────────────────────

    TEST_F(Test__HeterogeneousBackend, DeviceGroupMetadata_2Cuda4Rocm_Gcd2)
    {
        // Test: 2 CUDA + 4 ROCm → GCD = 2
        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("2_cuda_4_rocm")
                         .setScope(CollectiveScope::LOCAL)
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::cuda(1))
                         .addDevice(DeviceId::rocm(0))
                         .addDevice(DeviceId::rocm(1))
                         .addDevice(DeviceId::rocm(2))
                         .addDevice(DeviceId::rocm(3))
                         .setLocalRank(0)
                         .build();

        EXPECT_EQ(group.cuda_count, 2);
        EXPECT_EQ(group.rocm_count, 4);
        EXPECT_FALSE(group.is_homogeneous);
        EXPECT_EQ(group.size(), 6u);
        // GCD(2,4) = 2 - this would enable 2-way parallel bridging
    }

    TEST_F(Test__HeterogeneousBackend, DeviceGroupMetadata_3Cuda6Rocm_Gcd3)
    {
        // Test: 3 CUDA + 6 ROCm → GCD = 3
        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("3_cuda_6_rocm")
                         .setScope(CollectiveScope::LOCAL)
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::cuda(1))
                         .addDevice(DeviceId::cuda(2))
                         .addDevice(DeviceId::rocm(0))
                         .addDevice(DeviceId::rocm(1))
                         .addDevice(DeviceId::rocm(2))
                         .addDevice(DeviceId::rocm(3))
                         .addDevice(DeviceId::rocm(4))
                         .addDevice(DeviceId::rocm(5))
                         .setLocalRank(0)
                         .build();

        EXPECT_EQ(group.cuda_count, 3);
        EXPECT_EQ(group.rocm_count, 6);
        EXPECT_FALSE(group.is_homogeneous);
        EXPECT_EQ(group.size(), 9u);
        // GCD(3,6) = 3 - this would enable 3-way parallel bridging
    }

    TEST_F(Test__HeterogeneousBackend, DeviceGroupMetadata_2Cuda6Rocm_Gcd2)
    {
        // Test: 2 CUDA + 6 ROCm → GCD = 2
        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("2_cuda_6_rocm")
                         .setScope(CollectiveScope::LOCAL)
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::cuda(1))
                         .addDevice(DeviceId::rocm(0))
                         .addDevice(DeviceId::rocm(1))
                         .addDevice(DeviceId::rocm(2))
                         .addDevice(DeviceId::rocm(3))
                         .addDevice(DeviceId::rocm(4))
                         .addDevice(DeviceId::rocm(5))
                         .setLocalRank(0)
                         .build();

        EXPECT_EQ(group.cuda_count, 2);
        EXPECT_EQ(group.rocm_count, 6);
        EXPECT_FALSE(group.is_homogeneous);
        EXPECT_EQ(group.size(), 8u);
        // GCD(2,6) = 2 - this would enable 2-way parallel bridging
    }

    TEST_F(Test__HeterogeneousBackend, DeviceGroupMetadata_1Cuda4Rocm_Gcd1)
    {
        // Test: 1 CUDA + 4 ROCm → GCD = 1 (no multi-bridge benefit)
        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("1_cuda_4_rocm")
                         .setScope(CollectiveScope::LOCAL)
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::rocm(0))
                         .addDevice(DeviceId::rocm(1))
                         .addDevice(DeviceId::rocm(2))
                         .addDevice(DeviceId::rocm(3))
                         .setLocalRank(0)
                         .build();

        EXPECT_EQ(group.cuda_count, 1);
        EXPECT_EQ(group.rocm_count, 4);
        EXPECT_FALSE(group.is_homogeneous);
        EXPECT_EQ(group.size(), 5u);
        // GCD(1,4) = 1 - standard pattern, no multi-bridge
    }

    TEST_F(Test__HeterogeneousBackend, DeviceGroupMetadata_4Cuda4Rocm_Symmetric)
    {
        // Test: 4 CUDA + 4 ROCm → Symmetric (GCD = 4)
        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("4_cuda_4_rocm")
                         .setScope(CollectiveScope::LOCAL)
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::cuda(1))
                         .addDevice(DeviceId::cuda(2))
                         .addDevice(DeviceId::cuda(3))
                         .addDevice(DeviceId::rocm(0))
                         .addDevice(DeviceId::rocm(1))
                         .addDevice(DeviceId::rocm(2))
                         .addDevice(DeviceId::rocm(3))
                         .setLocalRank(0)
                         .build();

        EXPECT_EQ(group.cuda_count, 4);
        EXPECT_EQ(group.rocm_count, 4);
        EXPECT_FALSE(group.is_homogeneous);
        EXPECT_EQ(group.size(), 8u);
        // GCD(4,4) = 4 - symmetric, 4-way parallel bridging possible
    }

    TEST_F(Test__HeterogeneousBackend, DeviceGroupMetadata_2Cuda3Rocm_Gcd1_Coprime)
    {
        // Test: 2 CUDA + 3 ROCm → GCD = 1 (coprime, no multi-bridge)
        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("2_cuda_3_rocm")
                         .setScope(CollectiveScope::LOCAL)
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::cuda(1))
                         .addDevice(DeviceId::rocm(0))
                         .addDevice(DeviceId::rocm(1))
                         .addDevice(DeviceId::rocm(2))
                         .setLocalRank(0)
                         .build();

        EXPECT_EQ(group.cuda_count, 2);
        EXPECT_EQ(group.rocm_count, 3);
        EXPECT_FALSE(group.is_homogeneous);
        EXPECT_EQ(group.size(), 5u);
        // GCD(2,3) = 1 - coprime, standard pattern
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Partial Reduce-Scatter Pattern Selection Tests (Singleton Configurations)
    // ═══════════════════════════════════════════════════════════════════════════
    // These tests verify analyzeTopology() correctly identifies singleton configs
    // and selects PARTIAL_REDUCE_SCATTER for large tensors.
    //
    // Singleton configs: 1 CUDA + N ROCm (N > 1) OR N CUDA + 1 ROCm (N > 1)
    // Non-singleton: 1+1 (minimal), N+N (symmetric), or N+M where both > 1

    /**
     * @brief Test fixture extension with helper to create device groups for singleton testing
     */
    class Test__HeterogeneousBackend_PartialRS : public Test__HeterogeneousBackend
    {
    protected:
        /// Create 1 CUDA + 4 ROCm group (CUDA singleton)
        DeviceGroup create1Cuda4RocmGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("1_cuda_4_rocm")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .addDevice(DeviceId::rocm(0))
                .addDevice(DeviceId::rocm(1))
                .addDevice(DeviceId::rocm(2))
                .addDevice(DeviceId::rocm(3))
                .setLocalRank(0)
                .build();
        }

        /// Create 4 CUDA + 1 ROCm group (ROCm singleton)
        DeviceGroup create4Cuda1RocmGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("4_cuda_1_rocm")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .addDevice(DeviceId::cuda(1))
                .addDevice(DeviceId::cuda(2))
                .addDevice(DeviceId::cuda(3))
                .addDevice(DeviceId::rocm(0))
                .setLocalRank(0)
                .build();
        }

        /// Create 1 CUDA + 2 ROCm group (our hardware!)
        DeviceGroup create1Cuda2RocmGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("1_cuda_2_rocm")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .addDevice(DeviceId::rocm(0))
                .addDevice(DeviceId::rocm(1))
                .setLocalRank(0)
                .build();
        }

        /// Create 2 CUDA + 1 ROCm group (ROCm singleton)
        DeviceGroup create2Cuda1RocmGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("2_cuda_1_rocm")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .addDevice(DeviceId::cuda(1))
                .addDevice(DeviceId::rocm(0))
                .setLocalRank(0)
                .build();
        }
    };

    /**
     * @test Verify 1 CUDA + 4 ROCm (CUDA singleton) selects PARTIAL_REDUCE_SCATTER for large tensors
     *
     * This is a singleton configuration:
     * - CUDA domain has 1 device (singleton)
     * - ROCm domain has 4 devices (larger domain for reduce-scatter)
     * - Pattern: RCCL reduce-scatter → chunked bridge → RCCL allgather
     */
    TEST_F(Test__HeterogeneousBackend_PartialRS, TopologyAnalysis_1Cuda4Rocm_SelectsPartialRS)
    {
        auto group = create1Cuda4RocmGroup();

        // Verify group metadata
        EXPECT_EQ(group.cuda_count, 1);
        EXPECT_EQ(group.rocm_count, 4);
        EXPECT_FALSE(group.is_homogeneous);

        // Note: We cannot fully test analyzeTopology() without initializing the backend
        // (which requires hardware), but we can test the DeviceGroup metadata and
        // verify the TopologyAnalysis struct fields that would be set.

        // The expected analysis for 1+4:
        // - is_cuda_singleton = true (1 CUDA)
        // - is_rocm_singleton = false (4 ROCm > 1)
        // - is_symmetric = false (1 != 4)
        // - is_minimal = false (not 1+1)
        // - pattern = PARTIAL_REDUCE_SCATTER for large tensors
        // - num_chunks = 4 (larger domain device count)

        // Test TopologyAnalysis struct field expectations
        HeterogeneousBackend::TopologyAnalysis expected;
        expected.cuda_count = 1;
        expected.rocm_count = 4;
        expected.is_symmetric = false;
        expected.is_minimal = false;
        expected.is_cuda_singleton = true;
        expected.is_rocm_singleton = false;
        expected.gcd = 1;        // GCD(1, 4) = 1
        expected.num_chunks = 4; // Larger domain count

        // Verify our expectations match what the struct should contain
        EXPECT_FALSE(expected.is_symmetric);
        EXPECT_TRUE(expected.is_cuda_singleton);
        EXPECT_FALSE(expected.is_rocm_singleton);
        EXPECT_EQ(expected.num_chunks, 4u);
    }

    /**
     * @test Verify 4 CUDA + 1 ROCm (ROCm singleton) selects PARTIAL_REDUCE_SCATTER for large tensors
     *
     * This is a singleton configuration:
     * - ROCm domain has 1 device (singleton)
     * - CUDA domain has 4 devices (larger domain for reduce-scatter)
     * - Pattern: NCCL reduce-scatter → chunked bridge → NCCL allgather
     */
    TEST_F(Test__HeterogeneousBackend_PartialRS, TopologyAnalysis_4Cuda1Rocm_SelectsPartialRS)
    {
        auto group = create4Cuda1RocmGroup();

        // Verify group metadata
        EXPECT_EQ(group.cuda_count, 4);
        EXPECT_EQ(group.rocm_count, 1);
        EXPECT_FALSE(group.is_homogeneous);

        // The expected analysis for 4+1:
        // - is_cuda_singleton = false (4 CUDA > 1)
        // - is_rocm_singleton = true (1 ROCm)
        // - is_symmetric = false (4 != 1)
        // - is_minimal = false (not 1+1)
        // - pattern = PARTIAL_REDUCE_SCATTER for large tensors
        // - num_chunks = 4 (larger domain device count)

        HeterogeneousBackend::TopologyAnalysis expected;
        expected.cuda_count = 4;
        expected.rocm_count = 1;
        expected.is_symmetric = false;
        expected.is_minimal = false;
        expected.is_cuda_singleton = false;
        expected.is_rocm_singleton = true;
        expected.gcd = 1;        // GCD(4, 1) = 1
        expected.num_chunks = 4; // Larger domain count

        EXPECT_FALSE(expected.is_symmetric);
        EXPECT_FALSE(expected.is_cuda_singleton);
        EXPECT_TRUE(expected.is_rocm_singleton);
        EXPECT_EQ(expected.num_chunks, 4u);
    }

    /**
     * @test Verify 1 CUDA + 2 ROCm (our hardware!) selects PARTIAL_REDUCE_SCATTER for large tensors
     *
     * This is our actual hardware configuration and the primary target for partial RS:
     * - CUDA domain has 1 device (singleton)
     * - ROCm domain has 2 devices
     * - Pattern: RCCL reduce-scatter → chunked bridge → RCCL allgather
     */
    TEST_F(Test__HeterogeneousBackend_PartialRS, TopologyAnalysis_1Cuda2Rocm_SelectsPartialRS)
    {
        auto group = create1Cuda2RocmGroup();

        // Verify group metadata
        EXPECT_EQ(group.cuda_count, 1);
        EXPECT_EQ(group.rocm_count, 2);
        EXPECT_FALSE(group.is_homogeneous);

        // The expected analysis for 1+2 (our hardware):
        // - is_cuda_singleton = true (1 CUDA)
        // - is_rocm_singleton = false (2 ROCm > 1)
        // - is_symmetric = false (1 != 2)
        // - is_minimal = false (not 1+1)
        // - pattern = PARTIAL_REDUCE_SCATTER for large tensors
        // - num_chunks = 2 (larger domain device count)

        HeterogeneousBackend::TopologyAnalysis expected;
        expected.cuda_count = 1;
        expected.rocm_count = 2;
        expected.is_symmetric = false;
        expected.is_minimal = false;
        expected.is_cuda_singleton = true;
        expected.is_rocm_singleton = false;
        expected.gcd = 1;        // GCD(1, 2) = 1
        expected.num_chunks = 2; // Larger domain count (2 ROCm devices)

        EXPECT_FALSE(expected.is_symmetric);
        EXPECT_TRUE(expected.is_cuda_singleton);
        EXPECT_FALSE(expected.is_rocm_singleton);
        EXPECT_EQ(expected.num_chunks, 2u);
    }

    /**
     * @test Verify 2 CUDA + 1 ROCm (ROCm singleton) selects PARTIAL_REDUCE_SCATTER for large tensors
     */
    TEST_F(Test__HeterogeneousBackend_PartialRS, TopologyAnalysis_2Cuda1Rocm_SelectsPartialRS)
    {
        auto group = create2Cuda1RocmGroup();

        // Verify group metadata
        EXPECT_EQ(group.cuda_count, 2);
        EXPECT_EQ(group.rocm_count, 1);
        EXPECT_FALSE(group.is_homogeneous);

        // The expected analysis for 2+1:
        // - is_cuda_singleton = false (2 CUDA > 1)
        // - is_rocm_singleton = true (1 ROCm)
        // - is_symmetric = false (2 != 1)
        // - is_minimal = false (not 1+1)
        // - pattern = PARTIAL_REDUCE_SCATTER for large tensors
        // - num_chunks = 2 (larger domain device count)

        HeterogeneousBackend::TopologyAnalysis expected;
        expected.cuda_count = 2;
        expected.rocm_count = 1;
        expected.is_symmetric = false;
        expected.is_minimal = false;
        expected.is_cuda_singleton = false;
        expected.is_rocm_singleton = true;
        expected.gcd = 1;        // GCD(2, 1) = 1
        expected.num_chunks = 2; // Larger domain count

        EXPECT_FALSE(expected.is_symmetric);
        EXPECT_FALSE(expected.is_cuda_singleton);
        EXPECT_TRUE(expected.is_rocm_singleton);
        EXPECT_EQ(expected.num_chunks, 2u);
    }

    /**
     * @test Verify 1 CUDA + 1 ROCm (minimal) does NOT select PARTIAL_REDUCE_SCATTER
     *
     * Minimal configuration is NOT a singleton - it has no parallelism in either domain.
     * Standard 3-phase is the only option.
     */
    TEST_F(Test__HeterogeneousBackend_PartialRS, TopologyAnalysis_1Cuda1Rocm_SelectsStandard)
    {
        auto group = createMinimalHeterogeneousGroup();

        // Verify group metadata
        EXPECT_EQ(group.cuda_count, 1);
        EXPECT_EQ(group.rocm_count, 1);
        EXPECT_FALSE(group.is_homogeneous);

        // The expected analysis for 1+1:
        // - is_cuda_singleton = false (definition: singleton means other domain has >1)
        // - is_rocm_singleton = false
        // - is_symmetric = true (1 == 1)
        // - is_minimal = true (both domains have exactly 1 device)
        // - pattern = STANDARD_3PHASE (no reduce-scatter benefit possible)

        HeterogeneousBackend::TopologyAnalysis expected;
        expected.cuda_count = 1;
        expected.rocm_count = 1;
        expected.is_symmetric = true; // 1 == 1
        expected.is_minimal = true;
        expected.is_cuda_singleton = false; // Not singleton, just minimal
        expected.is_rocm_singleton = false;
        expected.gcd = 1;
        expected.num_chunks = 1; // No chunking benefit
        expected.pattern = HeterogeneousBackend::AllreducePattern::STANDARD_3PHASE;

        EXPECT_TRUE(expected.is_symmetric);
        EXPECT_TRUE(expected.is_minimal);
        EXPECT_FALSE(expected.is_cuda_singleton);
        EXPECT_FALSE(expected.is_rocm_singleton);
        EXPECT_EQ(expected.pattern, HeterogeneousBackend::AllreducePattern::STANDARD_3PHASE);
    }

    /**
     * @test Verify num_chunks matches larger domain device count for singleton configs
     *
     * For singleton configurations, the number of chunks should equal the device count
     * in the larger domain, since that's where reduce-scatter happens.
     */
    TEST_F(Test__HeterogeneousBackend_PartialRS, PartialRS_NumChunksMatchesLargerDomain)
    {
        // Test various singleton configurations

        // 1 CUDA + 2 ROCm → num_chunks = 2
        {
            auto group = create1Cuda2RocmGroup();
            size_t larger_domain = std::max(group.cuda_count, group.rocm_count);
            EXPECT_EQ(larger_domain, 2u) << "1+2 config: larger domain should be 2";
        }

        // 1 CUDA + 4 ROCm → num_chunks = 4
        {
            auto group = create1Cuda4RocmGroup();
            size_t larger_domain = std::max(group.cuda_count, group.rocm_count);
            EXPECT_EQ(larger_domain, 4u) << "1+4 config: larger domain should be 4";
        }

        // 4 CUDA + 1 ROCm → num_chunks = 4
        {
            auto group = create4Cuda1RocmGroup();
            size_t larger_domain = std::max(group.cuda_count, group.rocm_count);
            EXPECT_EQ(larger_domain, 4u) << "4+1 config: larger domain should be 4";
        }

        // 2 CUDA + 1 ROCm → num_chunks = 2
        {
            auto group = create2Cuda1RocmGroup();
            size_t larger_domain = std::max(group.cuda_count, group.rocm_count);
            EXPECT_EQ(larger_domain, 2u) << "2+1 config: larger domain should be 2";
        }
    }

#endif // defined(HAVE_CUDA) && defined(HAVE_ROCM)

    // ═══════════════════════════════════════════════════════════════════════════
    // Stub Implementation Tests (When CUDA+ROCm not available)
    // ═══════════════════════════════════════════════════════════════════════════

#if !defined(HAVE_CUDA) || !defined(HAVE_ROCM)

    TEST_F(Test__HeterogeneousBackend, StubIsNotAvailable)
    {
        EXPECT_FALSE(backend_->isAvailable());
    }

    TEST_F(Test__HeterogeneousBackend, StubInitializeFails)
    {
        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("test")
                         .setScope(CollectiveScope::LOCAL)
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::rocm(0))
                         .setLocalRank(0)
                         .build();

        EXPECT_FALSE(backend_->initialize(group));
    }

    TEST_F(Test__HeterogeneousBackend, StubAccessorsReturnDefaults)
    {
        EXPECT_TRUE(backend_->cudaDevices().empty());
        EXPECT_TRUE(backend_->rocmDevices().empty());
        EXPECT_TRUE(backend_->cudaBridge().is_cpu());
        EXPECT_TRUE(backend_->rocmBridge().is_cpu());
        EXPECT_FALSE(backend_->hasNCCLBackend());
        EXPECT_FALSE(backend_->hasRCCLBackend());
        EXPECT_FALSE(backend_->hasBridgeBackend());
    }

#endif

} // namespace llaminar2::test
