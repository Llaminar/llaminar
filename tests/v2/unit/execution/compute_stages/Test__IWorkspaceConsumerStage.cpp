/**
 * @file Test__IWorkspaceConsumerStage.cpp
 * @brief Unit tests for IWorkspaceConsumerStage mixin interface
 *
 * Tests the workspace delegation pattern where stages forward workspace
 * management calls to their underlying kernels.
 */

#include <gtest/gtest.h>
#include "v2/execution/compute_stages/IWorkspaceConsumerStage.h"
#include "v2/execution/local_execution/device/DeviceWorkspaceManager.h"
#include "v2/execution/local_execution/device/WorkspaceDescriptor.h"
#include "v2/interfaces/IWorkspaceConsumer.h"
#include "v2/kernels/cpu/CPUInvocationWorkspace.h"
#include "v2/planning/PhysicalMemoryAuthority.h"
#include "v2/backends/BackendManager.h"
#include <algorithm>
#include <future>

namespace llaminar2::test
{

    /** @brief An immutable operation exposing the same invocation binding contract as CPU projections. */
    class InvocationWorkspaceKernel final : public CPUInvocationWorkspaceConsumer
    {
    public:
        /** @return One explicitly sized persistent output scratch region. */
        WorkspaceRequirements getWorkspaceRequirements(int m, int, int) const override
        {
            WorkspaceRequirements requirements;
            requirements.buffers.push_back({"invocation", size_t(m) * sizeof(float), 64});
            return requirements;
        }
        /** @brief Write only this invocation's arena; no kernel state or allocation changes. */
        void execute(DeviceWorkspaceManager *workspace, size_t count, float value) const
        {
            auto scratch = CPUInvocationWorkspace::require<float>(workspace, "invocation", count);
            std::fill(scratch.begin(), scratch.end(), value);
        }
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Mock Kernel for Testing Delegation
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Mock kernel implementing IWorkspaceConsumer for testing delegation
     */
    class MockWorkspaceConsumerKernel : public IWorkspaceConsumer
    {
    public:
        WorkspaceRequirements getWorkspaceRequirements(int m, int n, int k) const override
        {
            last_m_ = m;
            last_n_ = n;
            last_k_ = k;
            getWorkspaceRequirements_called_ = true;

            WorkspaceRequirements reqs;
            reqs.buffers.push_back({"test_buffer", static_cast<size_t>(m * 4)});
            if (n > 0)
            {
                reqs.buffers.push_back({"secondary_buffer", static_cast<size_t>(n * 2)});
            }
            return reqs;
        }

        void bindWorkspace(DeviceWorkspaceManager *ws) override
        {
            bound_ws_ = ws;
            bindWorkspace_called_ = true;
        }

        void unbindWorkspace() override
        {
            bound_ws_ = nullptr;
            unbindWorkspace_called_ = true;
        }

        bool hasWorkspace() const override
        {
            return bound_ws_ != nullptr;
        }

        DeviceWorkspaceManager *getWorkspace() const override
        {
            return bound_ws_;
        }

        // Test introspection methods
        bool wasGetWorkspaceRequirementsCalled() const { return getWorkspaceRequirements_called_; }
        bool wasBindWorkspaceCalled() const { return bindWorkspace_called_; }
        bool wasUnbindWorkspaceCalled() const { return unbindWorkspace_called_; }
        int getLastM() const { return last_m_; }
        int getLastN() const { return last_n_; }
        int getLastK() const { return last_k_; }
        DeviceWorkspaceManager *getBoundWorkspace() const { return bound_ws_; }

        void reset()
        {
            getWorkspaceRequirements_called_ = false;
            bindWorkspace_called_ = false;
            unbindWorkspace_called_ = false;
            last_m_ = 0;
            last_n_ = 0;
            last_k_ = 0;
            bound_ws_ = nullptr;
        }

    private:
        DeviceWorkspaceManager *bound_ws_ = nullptr;
        mutable bool getWorkspaceRequirements_called_ = false;
        bool bindWorkspace_called_ = false;
        bool unbindWorkspace_called_ = false;
        mutable int last_m_ = 0;
        mutable int last_n_ = 0;
        mutable int last_k_ = 0;
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Testable Stage Implementation
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Concrete test stage that allows setting the kernel for testing
     */
    class TestableWorkspaceConsumerStage : public IWorkspaceConsumerStage
    {
    public:
        void setKernel(IWorkspaceConsumer *k) { kernel_ = k; }

        IWorkspaceConsumer *getKernelAsWorkspaceConsumer() override { return kernel_; }

    private:
        IWorkspaceConsumer *kernel_ = nullptr;
    };

    /** @test Shared immutable kernels never borrow another participant's binding, including concurrent calls. */
    TEST(Test__IWorkspaceConsumerStage, InvocationOwnedWorkspacesAreIndependentAndRetireThroughTheirAuthority)
    {
        // Only register the CPU allocator; this unit must not discover GPUs.
        if (!hasCPUBackend())
            initCPUBackend(-1);
        InvocationWorkspaceKernel kernel;
        constexpr size_t count = 64;
        const auto requirements = kernel.getWorkspaceRequirements(count, 0, 0);
        const size_t bytes = requirements.total_bytes_with_alignment();
        PhysicalMemoryPlanBuilder builder;
        builder.add({.world_rank = 0, .device = DeviceId::cpu(),
                .total_bytes = 2 * bytes, .admission_available_bytes = 2 * bytes},
            PhysicalMemoryOwner::ExecutionWorkspace, 2 * bytes);
        const auto memory = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
        {
            DeviceWorkspaceManager first(DeviceId::cpu(), bytes, memory), second(DeviceId::cpu(), bytes, memory);
            ASSERT_TRUE(first.allocate(requirements));
            ASSERT_TRUE(second.allocate(requirements));
            TestableWorkspaceConsumerStage first_stage, second_stage;
            first_stage.setKernel(&kernel);
            second_stage.setKernel(&kernel);
            first_stage.bindWorkspace(&first);
            second_stage.bindWorkspace(&second);
            ASSERT_TRUE(first_stage.hasWorkspace());
            ASSERT_TRUE(second_stage.hasWorkspace());
            EXPECT_FALSE(kernel.hasWorkspace());
            EXPECT_EQ(kernel.getWorkspace(), nullptr);

            auto left = std::async(std::launch::async, [&] {
                for (int i = 0; i < 20; ++i) kernel.execute(first_stage.getWorkspace(), count, 3.0f);
            });
            auto right = std::async(std::launch::async, [&] {
                for (int i = 0; i < 20; ++i) kernel.execute(second_stage.getWorkspace(), count, 7.0f);
            });
            left.get();
            right.get();
            for (auto value : CPUInvocationWorkspace::require<float>(&first, "invocation", count)) EXPECT_EQ(value, 3.0f);
            for (auto value : CPUInvocationWorkspace::require<float>(&second, "invocation", count)) EXPECT_EQ(value, 7.0f);
            first_stage.unbindWorkspace();
            EXPECT_FALSE(first_stage.hasWorkspace());
            EXPECT_TRUE(second_stage.hasWorkspace());
            kernel.execute(second_stage.getWorkspace(), count, 9.0f);
            EXPECT_THROW(kernel.execute(nullptr, count, 1.0f), std::runtime_error);
            EXPECT_THROW(kernel.execute(&second, count + 1, 1.0f), std::runtime_error);
            EXPECT_THROW(CPUInvocationWorkspace::require<float>(&second, "missing", 1), std::runtime_error);
            EXPECT_THROW(CPUInvocationWorkspace::multiply(SIZE_MAX, 2), std::overflow_error);
            second_stage.unbindWorkspace();
        }
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Tests: Null Kernel Behavior
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(Test__IWorkspaceConsumerStage, NullKernel_GetWorkspaceRequirements_ReturnsEmpty)
    {
        TestableWorkspaceConsumerStage stage;
        // kernel_ is nullptr by default

        auto reqs = stage.getWorkspaceRequirements(100, 200, 300);

        EXPECT_TRUE(reqs.buffers.empty());
        EXPECT_EQ(reqs.total_bytes(), 0u);
    }

    TEST(Test__IWorkspaceConsumerStage, NullKernel_BindWorkspace_StoresWorkspaceLocally)
    {
        TestableWorkspaceConsumerStage stage;
        // kernel_ is nullptr

        // Create a dummy pointer to use as workspace (we won't actually use it)
        DeviceWorkspaceManager *dummy_ws = reinterpret_cast<DeviceWorkspaceManager *>(0x12345678);

        stage.bindWorkspace(dummy_ws);

        // getWorkspace should return the stored pointer even with null kernel
        EXPECT_EQ(stage.getWorkspace(), dummy_ws);
    }

    TEST(Test__IWorkspaceConsumerStage, NullKernel_HasWorkspace_ReturnsFalse)
    {
        TestableWorkspaceConsumerStage stage;
        // kernel_ is nullptr

        // Even if we bind workspace, hasWorkspace should return false because
        // the kernel (null) can't confirm it has workspace
        DeviceWorkspaceManager *dummy_ws = reinterpret_cast<DeviceWorkspaceManager *>(0x12345678);
        stage.bindWorkspace(dummy_ws);

        EXPECT_FALSE(stage.hasWorkspace());
    }

    TEST(Test__IWorkspaceConsumerStage, NullKernel_UnbindWorkspace_ClearsStoredPointer)
    {
        TestableWorkspaceConsumerStage stage;
        DeviceWorkspaceManager *dummy_ws = reinterpret_cast<DeviceWorkspaceManager *>(0x12345678);

        stage.bindWorkspace(dummy_ws);
        EXPECT_EQ(stage.getWorkspace(), dummy_ws);

        stage.unbindWorkspace();
        EXPECT_EQ(stage.getWorkspace(), nullptr);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Tests: Valid Kernel Delegation
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(Test__IWorkspaceConsumerStage, ValidKernel_GetWorkspaceRequirements_DelegatesToKernel)
    {
        TestableWorkspaceConsumerStage stage;
        MockWorkspaceConsumerKernel kernel;
        stage.setKernel(&kernel);

        auto reqs = stage.getWorkspaceRequirements(128, 64, 32);

        EXPECT_TRUE(kernel.wasGetWorkspaceRequirementsCalled());
        EXPECT_EQ(kernel.getLastM(), 128);
        EXPECT_EQ(kernel.getLastN(), 64);
        EXPECT_EQ(kernel.getLastK(), 32);

        // Verify the requirements came from the kernel
        EXPECT_EQ(reqs.buffers.size(), 2u); // test_buffer and secondary_buffer
        EXPECT_EQ(reqs.buffers[0].name, "test_buffer");
        EXPECT_EQ(reqs.buffers[0].size_bytes, 128u * 4); // m * 4
        EXPECT_EQ(reqs.buffers[1].name, "secondary_buffer");
        EXPECT_EQ(reqs.buffers[1].size_bytes, 64u * 2); // n * 2
    }

    TEST(Test__IWorkspaceConsumerStage, ValidKernel_GetWorkspaceRequirements_WithZeroN)
    {
        TestableWorkspaceConsumerStage stage;
        MockWorkspaceConsumerKernel kernel;
        stage.setKernel(&kernel);

        auto reqs = stage.getWorkspaceRequirements(256, 0, 0);

        EXPECT_TRUE(kernel.wasGetWorkspaceRequirementsCalled());
        EXPECT_EQ(kernel.getLastM(), 256);
        EXPECT_EQ(kernel.getLastN(), 0);
        EXPECT_EQ(kernel.getLastK(), 0);

        // With n=0, only test_buffer should be created
        EXPECT_EQ(reqs.buffers.size(), 1u);
        EXPECT_EQ(reqs.buffers[0].name, "test_buffer");
        EXPECT_EQ(reqs.buffers[0].size_bytes, 256u * 4);
    }

    TEST(Test__IWorkspaceConsumerStage, ValidKernel_BindWorkspace_DelegatesToKernel)
    {
        TestableWorkspaceConsumerStage stage;
        MockWorkspaceConsumerKernel kernel;
        stage.setKernel(&kernel);

        DeviceWorkspaceManager *dummy_ws = reinterpret_cast<DeviceWorkspaceManager *>(0xABCDEF00);

        stage.bindWorkspace(dummy_ws);

        EXPECT_TRUE(kernel.wasBindWorkspaceCalled());
        EXPECT_EQ(kernel.getBoundWorkspace(), dummy_ws);
        EXPECT_EQ(stage.getWorkspace(), dummy_ws);
    }

    TEST(Test__IWorkspaceConsumerStage, ValidKernel_HasWorkspace_ReturnsKernelStatus)
    {
        TestableWorkspaceConsumerStage stage;
        MockWorkspaceConsumerKernel kernel;
        stage.setKernel(&kernel);

        // Initially no workspace bound
        EXPECT_FALSE(stage.hasWorkspace());

        // Bind workspace
        DeviceWorkspaceManager *dummy_ws = reinterpret_cast<DeviceWorkspaceManager *>(0xDEADBEEF);
        stage.bindWorkspace(dummy_ws);

        // Now kernel has workspace, so stage should report true
        EXPECT_TRUE(stage.hasWorkspace());
    }

    TEST(Test__IWorkspaceConsumerStage, ValidKernel_UnbindWorkspace_DelegatesToKernel)
    {
        TestableWorkspaceConsumerStage stage;
        MockWorkspaceConsumerKernel kernel;
        stage.setKernel(&kernel);

        DeviceWorkspaceManager *dummy_ws = reinterpret_cast<DeviceWorkspaceManager *>(0xCAFEBABE);
        stage.bindWorkspace(dummy_ws);

        EXPECT_TRUE(stage.hasWorkspace());
        EXPECT_FALSE(kernel.wasUnbindWorkspaceCalled());

        stage.unbindWorkspace();

        EXPECT_TRUE(kernel.wasUnbindWorkspaceCalled());
        EXPECT_EQ(kernel.getBoundWorkspace(), nullptr);
        EXPECT_EQ(stage.getWorkspace(), nullptr);
        EXPECT_FALSE(stage.hasWorkspace());
    }

    TEST(Test__IWorkspaceConsumerStage, ValidKernel_GetWorkspace_ReturnsStoredPointer)
    {
        TestableWorkspaceConsumerStage stage;
        MockWorkspaceConsumerKernel kernel;
        stage.setKernel(&kernel);

        EXPECT_EQ(stage.getWorkspace(), nullptr);

        DeviceWorkspaceManager *dummy_ws = reinterpret_cast<DeviceWorkspaceManager *>(0x11111111);
        stage.bindWorkspace(dummy_ws);

        EXPECT_EQ(stage.getWorkspace(), dummy_ws);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Tests: Kernel Switching Behavior
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(Test__IWorkspaceConsumerStage, KernelSwitch_FromValidToNull_HasWorkspaceReturnsFalse)
    {
        TestableWorkspaceConsumerStage stage;
        MockWorkspaceConsumerKernel kernel;

        // Start with valid kernel
        stage.setKernel(&kernel);
        DeviceWorkspaceManager *dummy_ws = reinterpret_cast<DeviceWorkspaceManager *>(0x22222222);
        stage.bindWorkspace(dummy_ws);

        EXPECT_TRUE(stage.hasWorkspace());

        // Switch to null kernel
        stage.setKernel(nullptr);

        // Workspace pointer is still stored, but hasWorkspace should return false
        // because no kernel to confirm it
        EXPECT_EQ(stage.getWorkspace(), dummy_ws);
        EXPECT_FALSE(stage.hasWorkspace());
    }

    TEST(Test__IWorkspaceConsumerStage, KernelSwitch_FromNullToValid_AfterBind)
    {
        TestableWorkspaceConsumerStage stage;
        MockWorkspaceConsumerKernel kernel;

        // Bind workspace while kernel is null
        DeviceWorkspaceManager *dummy_ws = reinterpret_cast<DeviceWorkspaceManager *>(0x33333333);
        stage.bindWorkspace(dummy_ws);

        EXPECT_FALSE(stage.hasWorkspace());

        // Now set a valid kernel
        stage.setKernel(&kernel);

        // hasWorkspace still false because kernel wasn't bound
        EXPECT_FALSE(stage.hasWorkspace());

        // Bind again to kernel
        stage.bindWorkspace(dummy_ws);
        EXPECT_TRUE(stage.hasWorkspace());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Tests: Edge Cases
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(Test__IWorkspaceConsumerStage, BindWorkspace_WithNullWorkspace)
    {
        TestableWorkspaceConsumerStage stage;
        MockWorkspaceConsumerKernel kernel;
        stage.setKernel(&kernel);

        // First bind a valid workspace
        DeviceWorkspaceManager *dummy_ws = reinterpret_cast<DeviceWorkspaceManager *>(0x44444444);
        stage.bindWorkspace(dummy_ws);
        EXPECT_TRUE(stage.hasWorkspace());

        // Bind null workspace
        stage.bindWorkspace(nullptr);

        EXPECT_EQ(stage.getWorkspace(), nullptr);
        EXPECT_FALSE(stage.hasWorkspace());
    }

    TEST(Test__IWorkspaceConsumerStage, MultipleBindUnbindCycles)
    {
        TestableWorkspaceConsumerStage stage;
        MockWorkspaceConsumerKernel kernel;
        stage.setKernel(&kernel);

        DeviceWorkspaceManager *ws1 = reinterpret_cast<DeviceWorkspaceManager *>(0x55555555);
        DeviceWorkspaceManager *ws2 = reinterpret_cast<DeviceWorkspaceManager *>(0x66666666);

        // Cycle 1
        stage.bindWorkspace(ws1);
        EXPECT_EQ(stage.getWorkspace(), ws1);
        EXPECT_TRUE(stage.hasWorkspace());

        stage.unbindWorkspace();
        EXPECT_EQ(stage.getWorkspace(), nullptr);
        EXPECT_FALSE(stage.hasWorkspace());

        // Cycle 2
        stage.bindWorkspace(ws2);
        EXPECT_EQ(stage.getWorkspace(), ws2);
        EXPECT_TRUE(stage.hasWorkspace());

        stage.unbindWorkspace();
        EXPECT_EQ(stage.getWorkspace(), nullptr);
        EXPECT_FALSE(stage.hasWorkspace());
    }

    TEST(Test__IWorkspaceConsumerStage, GetWorkspaceRequirements_CalledMultipleTimes)
    {
        TestableWorkspaceConsumerStage stage;
        MockWorkspaceConsumerKernel kernel;
        stage.setKernel(&kernel);

        auto reqs1 = stage.getWorkspaceRequirements(100, 50, 25);
        EXPECT_EQ(kernel.getLastM(), 100);

        auto reqs2 = stage.getWorkspaceRequirements(200, 0, 0);
        EXPECT_EQ(kernel.getLastM(), 200);

        auto reqs3 = stage.getWorkspaceRequirements(300, 150, 75);
        EXPECT_EQ(kernel.getLastM(), 300);
        EXPECT_EQ(kernel.getLastN(), 150);
        EXPECT_EQ(kernel.getLastK(), 75);

        // All three calls should work independently
        EXPECT_EQ(reqs1.buffers.size(), 2u);
        EXPECT_EQ(reqs2.buffers.size(), 1u);
        EXPECT_EQ(reqs3.buffers.size(), 2u);
    }

} // namespace llaminar2::test
