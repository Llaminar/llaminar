#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "execution/local_execution/device/WorkspaceAllocator.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "interfaces/IWorkspaceConsumer.h"

#include <optional>
#include <cstdint>
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
            saw_request_m7_n11_k13_ =
                saw_request_m7_n11_k13_ || (m == 7 && n == 11 && k == 13);
            saw_request_m600_n1_ =
                saw_request_m600_n1_ || (m == 600 && n == 1);
            saw_decode_m1_ = saw_decode_m1_ || (m == 1);

            WorkspaceRequirements reqs;
            reqs.buffers = buffers_;
            return reqs;
        }

        void bindWorkspace(DeviceWorkspaceManager *workspace) override
        {
            bound_workspace_ = workspace;
            ++bind_calls_;
            bind_sequence_.push_back(workspace);
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
        const std::vector<DeviceWorkspaceManager *> &bindSequence() const { return bind_sequence_; }
        int lastM() const { return last_m_; }
        int lastN() const { return last_n_; }
        int lastK() const { return last_k_; }
        bool sawRequestM7N11K13() const { return saw_request_m7_n11_k13_; }
        bool sawRequestM600N1() const { return saw_request_m600_n1_; }
        bool sawDecodeM1() const { return saw_decode_m1_; }

        void setBuffers(std::vector<WorkspaceDescriptor> buffers) { buffers_ = std::move(buffers); }

    private:
        std::vector<WorkspaceDescriptor> buffers_;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
        std::vector<DeviceWorkspaceManager *> bind_sequence_;
        int bind_calls_ = 0;
        mutable int requirements_calls_ = 0;
        mutable int last_m_ = -1;
        mutable int last_n_ = -1;
        mutable int last_k_ = -1;
        mutable bool saw_request_m7_n11_k13_ = false;
        mutable bool saw_request_m600_n1_ = false;
        mutable bool saw_decode_m1_ = false;
    };

    class DeclaredShapeWorkspaceStage : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        DeclaredShapeWorkspaceStage(
            DeviceId device,
            std::vector<size_t> input_shape,
            std::vector<size_t> output_shape,
            bool declare_decode_only_buffer = false)
            : IComputeStage(device),
              input_shape_(std::move(input_shape)),
              output_shape_(std::move(output_shape)),
              declare_decode_only_buffer_(declare_decode_only_buffer)
        {
        }

        bool execute(IDeviceContext *) override { return true; }
        ComputeStageType type() const override { return ComputeStageType::GEMM; }
        bool supportsBackend(ComputeBackendType) const override { return true; }
        StageDumpInfo buildDumpInfoImpl() const override { return {}; }

        StageBufferRequirements getBufferRequirements() const override
        {
            StageBufferRequirements reqs;
            reqs.addInput("input", input_shape_, BufferTensorType::FP32);
            reqs.addOutput("output", output_shape_, BufferTensorType::FP32);
            return reqs;
        }

        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override
        {
            ++requirements_calls_;
            last_m_ = m;
            last_n_ = n;
            last_k_ = k;
            saw_declared_m_ = saw_declared_m_ || (m == static_cast<int>(input_shape_[0]));
            saw_decode_m_ = saw_decode_m_ || (m == 1);

            WorkspaceRequirements reqs;
            reqs.buffers.push_back({"declared_shape_scratch", 4096, 256, true});
            if (declare_decode_only_buffer_ && m == 1)
            {
                reqs.buffers.push_back({"decode_only_scratch", 2048, 256, true});
            }
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

        bool hasWorkspace() const override { return bound_workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return bound_workspace_; }

        int requirementsCalls() const { return requirements_calls_; }
        int bindCalls() const { return bind_calls_; }
        int lastM() const { return last_m_; }
        int lastN() const { return last_n_; }
        int lastK() const { return last_k_; }
        bool sawDeclaredM() const { return saw_declared_m_; }
        bool sawDecodeM() const { return saw_decode_m_; }
        DeviceWorkspaceManager *boundWorkspace() const { return bound_workspace_; }

    private:
        std::vector<size_t> input_shape_;
        std::vector<size_t> output_shape_;
        bool declare_decode_only_buffer_ = false;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
        int bind_calls_ = 0;
        mutable int requirements_calls_ = 0;
        mutable int last_m_ = -1;
        mutable int last_n_ = -1;
        mutable int last_k_ = -1;
        mutable bool saw_declared_m_ = false;
        mutable bool saw_decode_m_ = false;
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
    const uint64_t initial_generation = allocator.deviceGeneration(*device);
    EXPECT_GT(initial_generation, 0u);
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
    EXPECT_GT(allocator.deviceGeneration(*device), initial_generation);

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
    // 2 bindWorkspace calls: first nullptr (ABA protection before old workspace
    // destruction), then the actual new workspace pointer.
    EXPECT_EQ(larger_consumer.bindCalls(), 2);
    EXPECT_GE(larger_consumer.requirementsCalls(), 2);
    EXPECT_TRUE(larger_consumer.sawRequestM7N11K13());
    EXPECT_TRUE(larger_consumer.sawDecodeM1());
}

TEST(Test__WorkspaceAllocator, BindsAllExtraConsumersToMergedWorkspace)
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

    MockWorkspaceConsumer main_kv_cache({
        {"kvcache_conv_scratch_k", 1024, 256, true},
        {"kvcache_conv_scratch_v", 1024, 256, true},
    });
    MockWorkspaceConsumer mtp_kv_cache({
        {"kvcache_conv_scratch_k", 4096, 256, true},
        {"kvcache_conv_scratch_v", 4096, 256, true},
    });

    auto main_request = requestFor(main_kv_cache, *device);
    main_request.m = 600;
    main_request.n = 1;
    auto mtp_request = requestFor(mtp_kv_cache, *device);
    mtp_request.m = 600;
    mtp_request.n = 1;

    ASSERT_TRUE(allocator.allocateForGraph(
        graph,
        hints,
        {main_request, mtp_request},
        config));

    auto *workspace = allocator.getDeviceWorkspace(*device);
    ASSERT_NE(workspace, nullptr);
    EXPECT_EQ(main_kv_cache.boundWorkspace(), workspace);
    EXPECT_EQ(mtp_kv_cache.boundWorkspace(), workspace);
    EXPECT_TRUE(workspace->hasBuffer("kvcache_conv_scratch_k"));
    EXPECT_TRUE(workspace->hasBuffer("kvcache_conv_scratch_v"));
    EXPECT_EQ(workspace->getBufferSize("kvcache_conv_scratch_k"), 4096u);
    EXPECT_EQ(workspace->getBufferSize("kvcache_conv_scratch_v"), 4096u);
    EXPECT_TRUE(main_kv_cache.sawRequestM600N1());
    EXPECT_TRUE(main_kv_cache.sawDecodeM1());
    EXPECT_TRUE(mtp_kv_cache.sawRequestM600N1());
    EXPECT_TRUE(mtp_kv_cache.sawDecodeM1());
}

TEST(Test__WorkspaceAllocator, GraphConsumerUsesDeclaredStageShapeForWorkspaceM)
{
    auto device = selectAvailableGpuWithMemory();
    if (!device)
    {
        GTEST_SKIP() << "No CUDA/ROCm GPU with enough free memory for WorkspaceAllocator unit test";
    }

    WorkspaceAllocator allocator;
    ComputeGraph graph;
    auto hints = tinyHints();
    hints.max_seq_len = 4096;

    auto stage = std::make_unique<DeclaredShapeWorkspaceStage>(
        *device,
        std::vector<size_t>{2, 8},
        std::vector<size_t>{2, 16});
    auto *raw_stage = stage.get();
    graph.addNode("mtp_gate_up_like", std::move(stage), *device);

    ASSERT_TRUE(allocator.allocateForGraph(
        graph,
        hints,
        {},
        unitBudgetConfig()));

    EXPECT_NE(raw_stage->boundWorkspace(), nullptr);
    EXPECT_GE(raw_stage->requirementsCalls(), 2);
    EXPECT_EQ(raw_stage->bindCalls(), 1);
    EXPECT_TRUE(raw_stage->sawDeclaredM())
        << "Workspace sizing must honor graph-declared M for MTP verifier replay";
    EXPECT_TRUE(raw_stage->sawDecodeM())
        << "GPU graph workspaces must also be sized for one-row decode replay";
    EXPECT_EQ(raw_stage->lastK(), 8);
    EXPECT_EQ(raw_stage->lastN(), 0)
        << "Prepared kernels keep their own output width when no explicit N is required";
}

TEST(Test__WorkspaceAllocator, GraphConsumerIncludesDecodeOnlyWorkspaceForPrefillSizedGraph)
{
    auto device = selectAvailableGpuWithMemory();
    if (!device)
    {
        GTEST_SKIP() << "No CUDA/ROCm GPU with enough free memory for WorkspaceAllocator unit test";
    }

    WorkspaceAllocator allocator;
    ComputeGraph graph;
    auto hints = tinyHints();
    hints.max_seq_len = 4096;

    auto stage = std::make_unique<DeclaredShapeWorkspaceStage>(
        *device,
        std::vector<size_t>{128, 8},
        std::vector<size_t>{128, 16},
        true);
    auto *raw_stage = stage.get();
    graph.addNode("cuda_fused_decode_like", std::move(stage), *device);

    ASSERT_TRUE(allocator.allocateForGraph(
        graph,
        hints,
        {},
        unitBudgetConfig()));

    ASSERT_NE(raw_stage->boundWorkspace(), nullptr);
    EXPECT_TRUE(raw_stage->sawDeclaredM());
    EXPECT_TRUE(raw_stage->sawDecodeM());
    EXPECT_TRUE(raw_stage->boundWorkspace()->hasBuffer("decode_only_scratch"))
        << "A prefill-sized graph allocation must include buffers declared only for M=1 decode";
}

TEST(Test__WorkspaceAllocator, GraphConsumerSkipsCPUWorkspaceForDeclaredStage)
{
    if (!hasCPUBackend())
    {
        initCPUBackend(-1);
    }

    WorkspaceAllocator allocator;
    ComputeGraph graph;
    auto hints = tinyHints();
    hints.max_seq_len = 64;

    WorkspaceBudgetConfig config;
    config.cpu_fraction = 0.1f;
    config.min_budget = 1 * 1024 * 1024;
    config.max_budget = 8 * 1024 * 1024;
    config.headroom = 0;

    auto stage = std::make_unique<DeclaredShapeWorkspaceStage>(
        DeviceId::cpu(),
        std::vector<size_t>{3, 8},
        std::vector<size_t>{3, 16});
    auto *raw_stage = stage.get();
    graph.addNode("cpu_gdn_verifier_state_like", std::move(stage), DeviceId::cpu());

    ASSERT_TRUE(allocator.allocateForGraph(
        graph,
        hints,
        {},
        config));

    auto *workspace = allocator.getDeviceWorkspace(DeviceId::cpu());
    EXPECT_EQ(workspace, nullptr)
        << "Graph-level DeviceWorkspaceManager binding is GPU-only; CPU scratch is owned by CPU kernels";
    EXPECT_EQ(raw_stage->boundWorkspace(), nullptr);
    EXPECT_EQ(raw_stage->requirementsCalls(), 0);
    EXPECT_EQ(raw_stage->bindCalls(), 0);
}

TEST(Test__WorkspaceAllocator, ExtraCPUConsumersAreIgnoredByGraphAllocator)
{
    if (!hasCPUBackend())
    {
        initCPUBackend(-1);
    }

    WorkspaceAllocator allocator;
    ComputeGraph graph;
    const auto hints = tinyHints();

    WorkspaceBudgetConfig config;
    config.cpu_fraction = 0.1f;
    config.min_budget = 1 * 1024 * 1024;
    config.max_budget = 8 * 1024 * 1024;
    config.headroom = 0;

    MockWorkspaceConsumer initial_consumer({
        {"shared_scratch", 1024, 256, true},
        {"old_only_scratch", 2048, 256, true},
    });

    ASSERT_TRUE(allocator.allocateForGraph(
        graph,
        hints,
        {requestFor(initial_consumer, DeviceId::cpu())},
        config));

    auto *initial_workspace = allocator.getDeviceWorkspace(DeviceId::cpu());
    EXPECT_EQ(initial_workspace, nullptr);
    EXPECT_EQ(initial_consumer.boundWorkspace(), nullptr);
    EXPECT_EQ(initial_consumer.requirementsCalls(), 0);
    EXPECT_EQ(initial_consumer.bindCalls(), 0);
    const uint64_t initial_generation = allocator.deviceGeneration(DeviceId::cpu());
    EXPECT_EQ(initial_generation, 0u);

    MockWorkspaceConsumer larger_consumer({
        {"shared_scratch", 4096, 256, true},
        {"new_only_scratch", 512, 256, true},
    });

    ASSERT_TRUE(allocator.allocateForGraph(
        graph,
        hints,
        {requestFor(larger_consumer, DeviceId::cpu())},
        config));

    auto *reallocated_workspace = allocator.getDeviceWorkspace(DeviceId::cpu());
    EXPECT_EQ(reallocated_workspace, nullptr);
    EXPECT_EQ(larger_consumer.boundWorkspace(), nullptr);
    EXPECT_EQ(larger_consumer.requirementsCalls(), 0);
    EXPECT_EQ(larger_consumer.bindCalls(), 0);
    EXPECT_EQ(allocator.deviceGeneration(DeviceId::cpu()), initial_generation);
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
    const uint64_t initial_generation = allocator.deviceGeneration(*device);
    EXPECT_GT(initial_generation, 0u);

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
    EXPECT_EQ(allocator.deviceGeneration(*device), initial_generation);
    EXPECT_EQ(reused_workspace, initial_workspace);
    EXPECT_EQ(smaller_consumer.boundWorkspace(), initial_workspace);
    EXPECT_EQ(smaller_consumer.bindCalls(), 1);
    EXPECT_TRUE(reused_workspace->hasBuffer("shared_scratch"));
    EXPECT_TRUE(reused_workspace->hasBuffer("old_only_scratch"));
    EXPECT_EQ(reused_workspace->getBufferSize("shared_scratch"), 4096u);
    EXPECT_EQ(reused_workspace->getBufferSize("old_only_scratch"), 2048u);
}

// Regression test for ABA pointer aliasing bug: when the old workspace is freed
// and the new one is allocated at the same heap address, kernels that gate state
// invalidation on `if (workspace_ != workspace)` would skip re-initialization
// (e.g., RoPE inv_freq upload). The allocator must call bindWorkspace(nullptr)
// before destroying the old workspace so the subsequent bind to the new workspace
// always triggers the state-invalidation path regardless of address reuse.
TEST(Test__WorkspaceAllocator, ReallocUnbindsBeforeDestroyToPreventABAPointerAliasing)
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

    // Use the same consumer for both allocations (simulates a persistent kernel
    // singleton that survives across graph rebuilds).
    MockWorkspaceConsumer consumer({
        {"scratch", 1024, 256, true},
    });

    // First allocation — consumer gets bound to workspace W1.
    ASSERT_TRUE(allocator.allocateForGraph(
        graph,
        hints,
        {requestFor(consumer, *device)},
        config));

    auto *w1 = consumer.boundWorkspace();
    ASSERT_NE(w1, nullptr);
    EXPECT_EQ(consumer.bindCalls(), 1);
    EXPECT_EQ(consumer.bindSequence().size(), 1u);
    EXPECT_EQ(consumer.bindSequence()[0], w1);

    // Grow the consumer's requirements so the second allocateForGraph triggers
    // the realloc path (existing "scratch" buffer is too small at 1024).
    consumer.setBuffers({
        {"scratch", 4096, 256, true},
        {"extra", 2048, 256, true},
    });

    ASSERT_TRUE(allocator.allocateForGraph(
        graph,
        hints,
        {requestFor(consumer, *device)},
        config));

    auto *w2 = consumer.boundWorkspace();
    ASSERT_NE(w2, nullptr);

    // Verify the bind sequence: initial bind (W1), nullptr unbind (ABA
    // protection), then new bind (W2).
    ASSERT_EQ(consumer.bindSequence().size(), 3u)
        << "Expected 3 bindWorkspace calls: initial, nullptr (ABA), new";
    EXPECT_EQ(consumer.bindSequence()[0], w1);
    EXPECT_EQ(consumer.bindSequence()[1], nullptr)
        << "Realloc must call bindWorkspace(nullptr) before destroying old workspace";
    EXPECT_EQ(consumer.bindSequence()[2], w2);
    EXPECT_NE(consumer.bindSequence()[2], nullptr);
}
