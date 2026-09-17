/**
 * @file CPUProjectionTestWorkspace.h
 * @brief Explicit CPU transform and activation arena for direct-kernel tests.
 *
 * Production stages declare and receive this storage from their graph owner.
 * Direct kernel tests use the same declaration and memory authority here;
 * nothing is retained in a shared engine or a thread-local payload cache.
 * Construct outside repeated or concurrently executed invocations.
 */
#pragma once
#include "backends/BackendManager.h"
#include "kernels/cpu/CPUInvocationWorkspace.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "tensors/TensorKernels.h"
#include <initializer_list>
#include <memory>

namespace llaminar2::test
{
    /**
     * @brief Merge declarations of serially invoked engines, preserving each format.
     * @param rows Largest simultaneous row count in the fixture.
     * @param kernels Exact prepared engines sharing this participant's workspace.
     * @return Named buffers, each sized by its actual declaring engine.
     */
    inline WorkspaceRequirements cpuProjectionTestRequirements(
        int rows, std::initializer_list<const ITensorGemm *> kernels)
    {
        WorkspaceRequirements requirements;
        for (const auto *kernel : kernels)
        {
            const auto *consumer = dynamic_cast<const IWorkspaceConsumer *>(kernel);
            if (!consumer) throw std::invalid_argument("Projection fixture requires a prepared workspace consumer");
            requirements.merge(consumer->getWorkspaceRequirements(rows));
        }
        return requirements;
    }
    /** @brief One independently admitted, reusable participant workspace. */
    class CPUProjectionTestWorkspace final
    {
    public:
        /** @brief Admit and allocate the SwiGLU tile and Q8 bank for the maximum shape. */
        CPUProjectionTestWorkspace(int rows, int input_columns, WorkspaceRequirements extra = {})
            : CPUProjectionTestWorkspace([&] {
                auto requirements = cpuSwiGLUWorkspaceRequirements(rows, input_columns);
                requirements.merge(cpuProjectionQ8WorkspaceRequirements(rows, input_columns));
                requirements.merge(extra);
                return requirements;
            }())
        {
        }
        /**
         * @brief Admit exactly the consumer's declaration, without implicit scratch.
         * @param requirements Complete named buffers needed by the tested invocation.
         */
        explicit CPUProjectionTestWorkspace(const WorkspaceRequirements &requirements)
        {
            if (!hasCPUBackend()) initCPUBackend(-1);
            const size_t bytes = requirements.total_bytes_with_alignment();
            PhysicalMemoryPlanBuilder builder;
            builder.add({.world_rank = 0, .device = DeviceId::cpu(),
                .total_bytes = bytes, .admission_available_bytes = bytes},
                PhysicalMemoryOwner::ExecutionWorkspace, bytes);
            memory_ = std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
            workspace_ = std::make_unique<DeviceWorkspaceManager>(DeviceId::cpu(), bytes, memory_);
            if (!workspace_->allocate(requirements))
                throw std::runtime_error("CPU projection test workspace allocation failed");
        }
        /** @return The exact arena passed to a direct kernel invocation. */
        DeviceWorkspaceManager *get() const noexcept { return workspace_.get(); }

    private:
        std::shared_ptr<PhysicalMemoryAuthority> memory_; ///< Outlives the physical allocation.
        std::unique_ptr<DeviceWorkspaceManager> workspace_; ///< Retires bytes before the ledger.
    };

    /**
     * @brief Test-owned equivalent of graph workspace admission and stage binding.
     *
     * Declare after the stage so the borrow is revoked before the arena retires.
     * Sizing comes only from the stage's production declaration; a missing buffer
     * therefore fails the test instead of being hidden by extra fixture capacity.
     */
    class CPUStageTestWorkspace final
    {
    public:
        /**
         * @brief Admit and bind one stage's maximum invocation geometry.
         * @param stage Existing stage whose lifetime encloses this scope.
         * @param rows Maximum row count, or zero to use the stage's bound geometry.
         */
        explicit CPUStageTestWorkspace(IWorkspaceConsumer &stage, int rows = 0)
            : stage_(stage), workspace_(stage.getWorkspaceRequirements(rows))
        {
            try
            {
                stage_.bindWorkspace(workspace_.get());
            }
            catch (...)
            {
                // A stage may have bound several projections before rejecting
                // another; revoke every borrow before constructor unwinding.
                stage_.unbindWorkspace();
                throw;
            }
        }
        /** @brief Revoke stage/engine borrows before retiring the admitted bytes. */
        ~CPUStageTestWorkspace() { stage_.unbindWorkspace(); }
        CPUStageTestWorkspace(const CPUStageTestWorkspace &) = delete;
        CPUStageTestWorkspace &operator=(const CPUStageTestWorkspace &) = delete;

    private:
        IWorkspaceConsumer &stage_; ///< Borrow, valid until this scope ends.
        CPUProjectionTestWorkspace workspace_; ///< One exact arena and its ledger.
    };
}
