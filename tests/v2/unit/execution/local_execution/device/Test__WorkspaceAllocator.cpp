#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "execution/local_execution/device/WorkspaceAllocator.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "interfaces/IWorkspaceConsumer.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace llaminar2;

namespace
{
    constexpr size_t kOneGiB = 1024ULL * 1024ULL * 1024ULL;

    std::optional<DeviceId> selectAvailableGpuWithMemory()
    {
#ifdef HAVE_ROCM
        if (auto *rocm = getROCmBackend())
        {
            if (rocm->deviceCount() > 0 && rocm->deviceMemoryFree(0) > kOneGiB)
            {
                return DeviceId::rocm(0);
            }
        }
#endif

#ifdef HAVE_CUDA
        if (auto *cuda = getCUDABackend())
        {
            if (cuda->deviceCount() > 0 && cuda->deviceMemoryFree(0) > kOneGiB)
            {
                return DeviceId::cuda(0);
            }
        }
#endif

        return std::nullopt;
    }

    WorkspaceSizingHints tinyHints()
    {
        // The allocator currently enforces a 768MB model-aware floor, so keep
        // all non-floor dimensions tiny to make these tests as light as the
        // production allocator permits.
        WorkspaceSizingHints hints;
        hints.max_seq_len = 1;
        hints.n_heads = 1;
        hints.head_dim = 1;
        hints.d_model = 1;
        hints.batch_size = 1;
        hints.vocab_size = 1;
        return hints;
    }

    WorkspaceBudgetConfig unitBudgetConfig()
    {
        WorkspaceBudgetConfig config;
        config.gpu_fraction = 0.8f;
        config.min_budget = 64 * 1024 * 1024;
        config.max_budget = 2ULL * 1024ULL * 1024ULL * 1024ULL;
        config.headroom = 64 * 1024 * 1024;
        return config;
    }

    class MockWorkspaceConsumer : public IWorkspaceConsumer
    {
    public:
        explicit MockWorkspaceConsumer(std::vector<WorkspaceDescriptor> buffers)
            : buffers_(std::move(buffers))
        {
        }

        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override
        {
            ++requirements_calls_;
            last_m_ = m;
            last_n_ = n;
            last_k_ = k;

            WorkspaceRequirements reqs;
            reqs.buffers = buffers_;
            return reqs;
        }

        void bindWorkspace(DeviceWorkspaceManager *workspace) override
        {
            bound_workspace_ = workspace;
            ++bind_calls_;
        }

        void unbindWorkspace() override
        {
            bound_workspace_ = nullptr;
        }

        bool hasWorkspace() const override
        {
            return bound_workspace_ != nullptr;
        }

        DeviceWorkspaceManager *getWorkspace() const override
        {
            return bound_workspace_;
        }

        int bindCalls() const { return bind_calls_; }
        int requirementsCalls() const { return requirements_calls_; }
        DeviceWorkspaceManager *boundWorkspace() const { return bound_workspace_; }
        int lastM() const { return last_m_; }
        int lastN() const { return last_n_; }
        int lastK() const { return last_k_; }

    private:
        std::vector<WorkspaceDescriptor> buffers_;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
        int bind_calls_ = 0;
        mutable int requirements_calls_ = 0;
        mutable int last_m_ = -1;
        mutable int last_n_ = -1;
        mutable int last_k_ = -1;
    };

    WorkspaceConsumerRequest requestFor(MockWorkspaceConsumer &consumer, DeviceId device)
    {
        WorkspaceConsumerRequest request;
        request.consumer = &consumer;
        request.device = device;
        request.m = 7;
        request.n = 11;
        request.k = 13;
        return request;
    }
} // namespace

TEST(Test__WorkspaceAllocator, ReallocatesExistingWorkspaceWhenNamedBufferGrows)
{
    auto device = selectAvailableGpuWithMemory();
    if (!device)
    {
        GTEST_SKIP() << "No CUDA/ROCm GPU with enough free memory for WorkspaceAllocator unit test";
    }

    WorkspaceAllocator allocator;
    ComputeGraph graph;
    const auto hints = tinyHints();
    const auto config = unitBudgetConfig();

    MockWorkspaceConsumer initial_consumer({
        {"shared_scratch", 1024, 256, true},
        {"old_only_scratch", 2048, 256, true},
    });

    ASSERT_TRUE(allocator.allocateForGraph(
        graph,
        hints,
        {requestFor(initial_consumer, *device)},
        config));

    auto *initial_workspace = allocator.getDeviceWorkspace(*device);
    ASSERT_NE(initial_workspace, nullptr);
    ASSERT_TRUE(initial_workspace->hasBuffer("shared_scratch"));
    ASSERT_TRUE(initial_workspace->hasBuffer("old_only_scratch"));
    EXPECT_EQ(initial_workspace->getBufferSize("shared_scratch"), 1024u);
    EXPECT_EQ(initial_workspace->getBufferSize("old_only_scratch"), 2048u);

    MockWorkspaceConsumer larger_consumer({
        {"shared_scratch", 4096, 256, true},
        {"new_only_scratch", 512, 256, true},
    });

    ASSERT_TRUE(allocator.allocateForGraph(
        graph,
        hints,
        {requestFor(larger_consumer, *device)},
        config));

    auto *reallocated_workspace = allocator.getDeviceWorkspace(*device);
    ASSERT_NE(reallocated_workspace, nullptr);

    // Regression coverage for the old name-only reuse check: the buffer existed
    // before, but it was too small.  The allocator must rebuild the workspace
    // with the larger size and merge both old and new requirements.
    EXPECT_TRUE(reallocated_workspace->hasBuffer("shared_scratch"));
    EXPECT_TRUE(reallocated_workspace->hasBuffer("old_only_scratch"));
    EXPECT_TRUE(reallocated_workspace->hasBuffer("new_only_scratch"));
    EXPECT_EQ(reallocated_workspace->getBufferSize("shared_scratch"), 4096u);
    EXPECT_EQ(reallocated_workspace->getBufferSize("old_only_scratch"), 2048u);
    EXPECT_EQ(reallocated_workspace->getBufferSize("new_only_scratch"), 512u);
    EXPECT_EQ(larger_consumer.boundWorkspace(), reallocated_workspace);
    EXPECT_EQ(larger_consumer.bindCalls(), 1);
    EXPECT_GE(larger_consumer.requirementsCalls(), 1);
    EXPECT_EQ(larger_consumer.lastM(), 7);
    EXPECT_EQ(larger_consumer.lastN(), 11);
    EXPECT_EQ(larger_consumer.lastK(), 13);
}

TEST(Test__WorkspaceAllocator, ReusesExistingWorkspaceWhenAllRequestedBuffersFit)
{
    auto device = selectAvailableGpuWithMemory();
    if (!device)
    {
        GTEST_SKIP() << "No CUDA/ROCm GPU with enough free memory for WorkspaceAllocator unit test";
    }

    WorkspaceAllocator allocator;
    ComputeGraph graph;
    const auto hints = tinyHints();
    const auto config = unitBudgetConfig();

    MockWorkspaceConsumer initial_consumer({
        {"shared_scratch", 4096, 256, true},
        {"old_only_scratch", 2048, 256, true},
    });

    ASSERT_TRUE(allocator.allocateForGraph(
        graph,
        hints,
        {requestFor(initial_consumer, *device)},
        config));

    auto *initial_workspace = allocator.getDeviceWorkspace(*device);
    ASSERT_NE(initial_workspace, nullptr);

    MockWorkspaceConsumer smaller_consumer({
        {"shared_scratch", 1024, 256, true},
    });

    ASSERT_TRUE(allocator.allocateForGraph(
        graph,
        hints,
        {requestFor(smaller_consumer, *device)},
        config));

    auto *reused_workspace = allocator.getDeviceWorkspace(*device);
    ASSERT_NE(reused_workspace, nullptr);
    EXPECT_EQ(reused_workspace, initial_workspace);
    EXPECT_EQ(smaller_consumer.boundWorkspace(), initial_workspace);
    EXPECT_EQ(smaller_consumer.bindCalls(), 1);
    EXPECT_TRUE(reused_workspace->hasBuffer("shared_scratch"));
    EXPECT_TRUE(reused_workspace->hasBuffer("old_only_scratch"));
    EXPECT_EQ(reused_workspace->getBufferSize("shared_scratch"), 4096u);
    EXPECT_EQ(reused_workspace->getBufferSize("old_only_scratch"), 2048u);
}
