/**
 * @file Test__WorkspaceAllocator.cpp
 * @brief Device-free and backend-specific tests for graph-family workspace planning.
 *
 * The suite proves that setup-time workspace discovery is complete before a
 * graph family publishes raw addresses. CPU executable stages, CUDA graphs,
 * and ROCm graphs share the same typed allocation contract; tests that need an
 * accelerator say so explicitly, while CPU coverage remains in the unit gate.
 */

#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "execution/local_execution/device/WorkspaceAllocator.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "interfaces/IWorkspaceConsumer.h"

#include <algorithm>
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
            max_m_ = std::max(max_m_, m);

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
        int maxM() const { return max_m_; }

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
        mutable int max_m_ = -1;
    };

    class DeclaredShapeWorkspaceStage : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        DeclaredShapeWorkspaceStage(
            DeviceId device,
            std::vector<size_t> input_shape,
            std::vector<size_t> output_shape,
            bool declare_decode_only_buffer = false,
            bool scale_scratch_with_m = false,
            bool declare_compact_regime_buffer = false,
            bool compact_rows_need_larger_shared_buffer = false,
            bool declare_prefill_regime_buffer = false)
            : IComputeStage(device),
              input_shape_(std::move(input_shape)),
              output_shape_(std::move(output_shape)),
              declare_decode_only_buffer_(declare_decode_only_buffer),
              scale_scratch_with_m_(scale_scratch_with_m),
              declare_compact_regime_buffer_(
                  declare_compact_regime_buffer),
              compact_rows_need_larger_shared_buffer_(
                  compact_rows_need_larger_shared_buffer),
              declare_prefill_regime_buffer_(
                  declare_prefill_regime_buffer)
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
            max_m_ = std::max(max_m_, m);
            saw_declared_m_ = saw_declared_m_ || (m == static_cast<int>(input_shape_[0]));
            saw_decode_m_ = saw_decode_m_ || (m == 1);

            WorkspaceRequirements reqs;
            reqs.buffers.push_back({
                "declared_shape_scratch",
                compact_rows_need_larger_shared_buffer_
                    ? (m <= 16 ? size_t{48} * 1024
                               : size_t{16} * 1024)
                : scale_scratch_with_m_
                    ? static_cast<size_t>(std::max(1, m)) * 256
                    : size_t{4096},
                256,
                true});
            if (declare_decode_only_buffer_ && m == 1)
            {
                reqs.buffers.push_back({"decode_only_scratch", 2048, 256, true});
            }
            if (declare_compact_regime_buffer_)
            {
                reqs.buffers.push_back({
                    "compact_regime_scratch",
                    12 * 1024,
                    256,
                    true,
                    WorkspaceExecutionRegime::CompactDecodeOnly});
            }
            if (declare_prefill_regime_buffer_)
            {
                reqs.buffers.push_back({
                    "prefill_regime_scratch",
                    12 * 1024,
                    256,
                    true,
                    WorkspaceExecutionRegime::PrefillOnly});
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
        int maxM() const { return max_m_; }
        bool sawDeclaredM() const { return saw_declared_m_; }
        bool sawDecodeM() const { return saw_decode_m_; }
        DeviceWorkspaceManager *boundWorkspace() const { return bound_workspace_; }

    private:
        std::vector<size_t> input_shape_;
        std::vector<size_t> output_shape_;
        bool declare_decode_only_buffer_ = false;
        bool scale_scratch_with_m_ = false;
        bool declare_compact_regime_buffer_ = false;
        bool compact_rows_need_larger_shared_buffer_ = false;
        bool declare_prefill_regime_buffer_ = false;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
        int bind_calls_ = 0;
        mutable int requirements_calls_ = 0;
        mutable int last_m_ = -1;
        mutable int last_n_ = -1;
        mutable int last_k_ = -1;
        mutable int max_m_ = -1;
        mutable bool saw_declared_m_ = false;
        mutable bool saw_decode_m_ = false;
    };

    /**
     * @brief Minimal graph stage with caller-defined workspace descriptors.
     *
     * The graph-family regressions need two distinct topologies that share one
     * workspace name at different capacities while also owning graph-local
     * names.  Keeping this stage declarative makes the test exercise the real
     * ComputeGraph scan and IWorkspaceConsumer binding path instead of reaching
     * into allocator internals.
     */
    class GraphFamilyWorkspaceStage final
        : public IComputeStage,
          public IWorkspaceConsumer
    {
    public:
        GraphFamilyWorkspaceStage(
            DeviceId device,
            std::vector<size_t> shape,
            std::vector<WorkspaceDescriptor> descriptors)
            : IComputeStage(device),
              shape_(std::move(shape)),
              descriptors_(std::move(descriptors))
        {
        }

        bool execute(IDeviceContext *) override { return true; }
        ComputeStageType type() const override
        {
            return ComputeStageType::GEMM;
        }
        bool supportsBackend(ComputeBackendType) const override
        {
            return true;
        }
        StageDumpInfo buildDumpInfoImpl() const override { return {}; }

        StageBufferRequirements getBufferRequirements() const override
        {
            StageBufferRequirements requirements;
            requirements.addInput(
                "input",
                shape_,
                BufferTensorType::FP32);
            requirements.addOutput(
                "output",
                shape_,
                BufferTensorType::FP32);
            return requirements;
        }

        WorkspaceRequirements getWorkspaceRequirements(
            int m,
            int n = 0,
            int k = 0) const override
        {
            last_m_ = m;
            last_n_ = n;
            last_k_ = k;
            WorkspaceRequirements requirements;
            requirements.buffers = descriptors_;
            return requirements;
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

        DeviceWorkspaceManager *boundWorkspace() const
        {
            return bound_workspace_;
        }

        int bindCalls() const { return bind_calls_; }
        int lastM() const { return last_m_; }
        int lastN() const { return last_n_; }
        int lastK() const { return last_k_; }

    private:
        std::vector<size_t> shape_;
        std::vector<WorkspaceDescriptor> descriptors_;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
        int bind_calls_ = 0;
        mutable int last_m_ = 0;
        mutable int last_n_ = 0;
        mutable int last_k_ = 0;
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

/**
 * @brief Terminal projection sizing is total across sharded and replicated participants.
 *
 * The regression is deliberately device-free. It proves the policy used by
 * phase-split LocalTP workspace planning without allocating GPU memory in the
 * unit gate: a full-vocabulary family envelope dominates an earlier sharded
 * participant, while an independently wider participant is never truncated.
 */
TEST(Test__WorkspaceAllocator, TerminalProjectionEnvelopeCoversEverySerialParticipant)
{
    WorkspaceSizingHints hints;

    EXPECT_EQ(hints.resolveTerminalProjectionColumns(0), 0)
        << "No family envelope preserves the prepared kernel's own N";
    EXPECT_EQ(hints.resolveTerminalProjectionColumns(124160), 124160);

    hints.serial_family_max_terminal_projection_columns = 248320;
    EXPECT_EQ(hints.resolveTerminalProjectionColumns(0), 248320);
    EXPECT_EQ(hints.resolveTerminalProjectionColumns(124160), 248320)
        << "A sharded prefill head must reserve the later replicated width";
    EXPECT_EQ(hints.resolveTerminalProjectionColumns(248320), 248320);
    EXPECT_EQ(hints.resolveTerminalProjectionColumns(496640), 496640)
        << "The family envelope must never truncate a wider participant";
}

TEST(Test__WorkspaceAllocator, ExtendsExistingWorkspaceWithoutInvalidatingCapturedAddresses)
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
    void *captured_shared_address =
        initial_workspace->getBuffer("shared_scratch");
    void *captured_old_only_address =
        initial_workspace->getBuffer("old_only_scratch");
    ASSERT_NE(captured_shared_address, nullptr);
    ASSERT_NE(captured_old_only_address, nullptr);

    MockWorkspaceConsumer larger_consumer({
        {"shared_scratch", 4096, 256, true},
        {"new_only_scratch", 512, 256, true},
    });

    ASSERT_TRUE(allocator.allocateForGraph(
        graph,
        hints,
        {requestFor(larger_consumer, *device)},
        config));

    auto *extended_workspace = allocator.getDeviceWorkspace(*device);
    ASSERT_NE(extended_workspace, nullptr);
    EXPECT_EQ(extended_workspace, initial_workspace);
    EXPECT_EQ(allocator.deviceGeneration(*device), initial_generation)
        << "Append-only growth preserves every captured workspace address";

    // A larger current view may use a new address, but the original allocation
    // remains owned until runner teardown. Names whose size did not change keep
    // exactly the same address.
    EXPECT_TRUE(extended_workspace->hasBuffer("shared_scratch"));
    EXPECT_TRUE(extended_workspace->hasBuffer("old_only_scratch"));
    EXPECT_TRUE(extended_workspace->hasBuffer("new_only_scratch"));
    EXPECT_EQ(extended_workspace->getBufferSize("shared_scratch"), 4096u);
    EXPECT_EQ(extended_workspace->getBufferSize("old_only_scratch"), 2048u);
    EXPECT_EQ(extended_workspace->getBufferSize("new_only_scratch"), 512u);
    EXPECT_NE(
        extended_workspace->getBuffer("shared_scratch"),
        captured_shared_address);
    EXPECT_EQ(
        extended_workspace->getBuffer("old_only_scratch"),
        captured_old_only_address);
    EXPECT_EQ(larger_consumer.boundWorkspace(), extended_workspace);
    EXPECT_EQ(larger_consumer.bindCalls(), 1);
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

/**
 * @brief A serial graph family reserves its largest mutable scratch before capture.
 *
 * The first graph deliberately declares only two rows. The family contract
 * advertises 128 rows, so the allocator must query and reserve that capacity
 * immediately. A later 64-row graph then binds the original address without an
 * append-only allocation or workspace generation change.
 */
TEST(Test__WorkspaceAllocator, SerialGraphFamilyUsesLargestParticipantAddressFromFirstCapture)
{
    auto device = selectAvailableGpuWithMemory();
    if (!device)
    {
        GTEST_SKIP() << "No CUDA/ROCm GPU with enough free memory for WorkspaceAllocator unit test";
    }

    WorkspaceAllocator allocator;
    auto hints = tinyHints();
    hints.max_seq_len = 2;
    hints.serial_family_max_rows = 128;
    hints.graph_family_policy =
        WorkspaceGraphFamilyPolicy::SerialDeviceFamilyLargestParticipant;

    ComputeGraph small_graph;
    auto small_stage = std::make_unique<DeclaredShapeWorkspaceStage>(
        *device,
        std::vector<size_t>{2, 8},
        std::vector<size_t>{2, 16},
        /*declare_decode_only_buffer=*/false,
        /*scale_scratch_with_m=*/true);
    auto *small_stage_ptr = small_stage.get();
    small_graph.addNode(
        "serial_family_small",
        std::move(small_stage),
        *device);

    ASSERT_TRUE(allocator.allocateForGraph(
        small_graph,
        hints,
        {},
        unitBudgetConfig()));

    auto *workspace = allocator.getDeviceWorkspace(*device);
    ASSERT_NE(workspace, nullptr);
    ASSERT_TRUE(workspace->hasBuffer("declared_shape_scratch"));
    EXPECT_EQ(
        workspace->getBufferSize("declared_shape_scratch"),
        size_t{128} * 256);
    EXPECT_EQ(small_stage_ptr->maxM(), 128);
    void *const family_address =
        workspace->getBuffer("declared_shape_scratch");
    ASSERT_NE(family_address, nullptr);
    const uint64_t generation = allocator.deviceGeneration(*device);

    ComputeGraph medium_graph;
    auto medium_stage = std::make_unique<DeclaredShapeWorkspaceStage>(
        *device,
        std::vector<size_t>{64, 8},
        std::vector<size_t>{64, 16},
        /*declare_decode_only_buffer=*/false,
        /*scale_scratch_with_m=*/true);
    auto *medium_stage_ptr = medium_stage.get();
    medium_graph.addNode(
        "serial_family_medium",
        std::move(medium_stage),
        *device);
    hints.max_seq_len = 64;

    ASSERT_TRUE(allocator.allocateForGraph(
        medium_graph,
        hints,
        {},
        unitBudgetConfig()));

    EXPECT_EQ(allocator.getDeviceWorkspace(*device), workspace);
    EXPECT_EQ(allocator.deviceGeneration(*device), generation);
    EXPECT_EQ(
        workspace->getBuffer("declared_shape_scratch"),
        family_address);
    EXPECT_EQ(
        workspace->getBufferSize("declared_shape_scratch"),
        size_t{128} * 256);
    EXPECT_EQ(medium_stage_ptr->maxM(), 128);
}

/**
 * @brief Serial graph roles reuse one physical block without local overlap.
 *
 * Main-forward scratch and grouped-verifier recurrent captures are live at
 * different points in the device event timeline. The verifier's three buffers
 * must therefore receive distinct addresses from each other while remaining
 * free to alias the already-complete main graph's two buffers.
 */
TEST(Test__WorkspaceAllocator, SerialGraphRolesAliasPrimaryBlockWithoutPhysicalGrowth)
{
    auto device = selectAvailableGpuWithMemory();
    if (!device)
    {
        GTEST_SKIP() << "No CUDA/ROCm GPU with enough free memory for WorkspaceAllocator unit test";
    }

    WorkspaceAllocator allocator;
    ComputeGraph main_graph;
    auto hints = tinyHints();
    hints.graph_family_policy =
        WorkspaceGraphFamilyPolicy::SerialDeviceFamilyExactParticipant;
    const auto config = unitBudgetConfig();

    constexpr size_t kHalfMiB = 512 * 1024;
    constexpr size_t kQuarterMiB = 256 * 1024;
    auto main_stage = std::make_unique<GraphFamilyWorkspaceStage>(
        *device,
        std::vector<size_t>{1, 128},
        std::vector<WorkspaceDescriptor>{
            {"main_projection_scratch", kHalfMiB, 256, true},
            {"main_attention_scratch", kHalfMiB, 256, true},
        });
    main_graph.addNode(
        "main_decode",
        std::move(main_stage),
        *device);

    ComputeGraph verifier_graph;
    auto verifier_stage =
        std::make_unique<GraphFamilyWorkspaceStage>(
            *device,
            std::vector<size_t>{16, 128},
            std::vector<WorkspaceDescriptor>{
                {"verifier_layer0_state_slots",
                 kQuarterMiB,
                 256,
                 true},
                {"verifier_layer1_state_slots",
                 kQuarterMiB,
                 256,
                 true},
                {"verifier_layer2_state_slots",
                 kQuarterMiB,
                 256,
                 true},
            });
    auto *verifier_stage_ptr = verifier_stage.get();
    verifier_graph.addNode(
        "grouped_verifier",
        std::move(verifier_stage),
        *device);

    ASSERT_TRUE(allocator.allocateForGraphFamily(
        main_graph,
        WorkspaceGraphParticipantRole::Decode,
        {WorkspaceGraphParticipant{
            .graph = &verifier_graph,
            .role =
                WorkspaceGraphParticipantRole::GroupedVerifier,
        }},
        hints,
        {},
        config));

    DeviceWorkspaceManager *workspace =
        allocator.getDeviceWorkspace(*device);
    ASSERT_NE(workspace, nullptr);
    const size_t physical_bytes = workspace->used();
    const uint64_t generation = allocator.deviceGeneration(*device);
    ASSERT_GE(workspace->primaryBlockSize(), 2 * kHalfMiB);

    EXPECT_EQ(workspace->used(), physical_bytes);
    EXPECT_EQ(allocator.deviceGeneration(*device), generation);
    EXPECT_EQ(verifier_stage_ptr->boundWorkspace(), workspace);

    std::vector<std::uintptr_t> verifier_addresses;
    for (const char *name : {
             "verifier_layer0_state_slots",
             "verifier_layer1_state_slots",
             "verifier_layer2_state_slots"})
    {
        ASSERT_TRUE(workspace->hasBuffer(name));
        verifier_addresses.push_back(
            reinterpret_cast<std::uintptr_t>(workspace->getBuffer(name)));
    }
    std::sort(verifier_addresses.begin(), verifier_addresses.end());
    EXPECT_GE(
        verifier_addresses[1] - verifier_addresses[0],
        kQuarterMiB);
    EXPECT_GE(
        verifier_addresses[2] - verifier_addresses[1],
        kQuarterMiB);

    const auto main_projection = reinterpret_cast<std::uintptr_t>(
        workspace->getBuffer("main_projection_scratch"));
    const auto main_attention = reinterpret_cast<std::uintptr_t>(
        workspace->getBuffer("main_attention_scratch"));
    const auto main_begin = std::min(main_projection, main_attention);
    const auto main_end =
        std::max(main_projection, main_attention) + kHalfMiB;
    EXPECT_TRUE(std::any_of(
        verifier_addresses.begin(),
        verifier_addresses.end(),
        [&](std::uintptr_t address)
        {
            return address >= main_begin && address < main_end;
        }))
        << "A serial verifier role should reuse bytes owned by the completed main graph";
}

/**
 * @brief A shared serial-family name can never grow after its address is public.
 *
 * Captured graph executables retain the original pointer. Permitting a later
 * participant to append a larger allocation under the same name would leave
 * the earlier executable addressing stale storage. The family declaration must
 * therefore include the maximum compact/large-M capacity before first capture,
 * and an incomplete declaration must fail closed without changing allocation
 * size, generation, or the original pointer.
 */
TEST(Test__WorkspaceAllocator, SerialGraphFamilyRejectsLateGrowthOfPublishedName)
{
    auto device = selectAvailableGpuWithMemory();
    if (!device)
    {
        GTEST_SKIP() << "No CUDA/ROCm GPU with enough free memory for WorkspaceAllocator unit test";
    }

    WorkspaceAllocator allocator;
    ComputeGraph graph;
    auto hints = tinyHints();
    hints.graph_family_policy =
        WorkspaceGraphFamilyPolicy::SerialDeviceFamilyExactParticipant;
    const auto config = unitBudgetConfig();

    MockWorkspaceConsumer initial_graph({
        {"shared_projection_scratch", 256 * 1024, 256, true},
        {"initial_graph_only_scratch", 768 * 1024, 256, true},
    });
    ASSERT_TRUE(allocator.allocateForGraph(
        graph,
        hints,
        {requestFor(initial_graph, *device)},
        config));

    DeviceWorkspaceManager *workspace =
        allocator.getDeviceWorkspace(*device);
    ASSERT_NE(workspace, nullptr);
    void *const published_address =
        workspace->getBuffer("shared_projection_scratch");
    const size_t physical_bytes = workspace->used();
    const uint64_t generation = allocator.deviceGeneration(*device);

    MockWorkspaceConsumer underplanned_later_graph({
        {"shared_projection_scratch", 512 * 1024, 256, true},
    });
    EXPECT_FALSE(allocator.allocateForGraph(
        graph,
        hints,
        {requestFor(underplanned_later_graph, *device)},
        config));

    EXPECT_EQ(workspace->used(), physical_bytes);
    EXPECT_EQ(allocator.deviceGeneration(*device), generation);
    EXPECT_EQ(
        workspace->getBuffer("shared_projection_scratch"),
        published_address);
    EXPECT_EQ(
        workspace->getBufferSize("shared_projection_scratch"),
        256 * 1024);
    EXPECT_EQ(underplanned_later_graph.boundWorkspace(), nullptr);
}

/**
 * @brief A serial family rejects a graph topology omitted from generation one.
 *
 * Dynamic MoE maintenance used to introduce its local histogram name only when
 * the first scheduler window closed. The allocator silently fit that new name
 * into unused primary bytes, leaving preflight incomplete and making captured
 * pointer ownership depend on request history. Every serial participant must
 * now be declared together; discovering even a small new name later is fatal.
 */
TEST(Test__WorkspaceAllocator, SerialGraphFamilyRejectsLateUndeclaredName)
{
    const auto device = selectAvailableGpuWithMemory();
    if (!device.has_value())
    {
        GTEST_SKIP() << "No GPU with enough free memory for workspace allocation";
    }

    WorkspaceAllocator allocator;
    ComputeGraph graph;
    auto hints = tinyHints();
    hints.graph_family_policy =
        WorkspaceGraphFamilyPolicy::
            SerialDeviceFamilyExactParticipant;
    const auto config = unitBudgetConfig();

    MockWorkspaceConsumer declared_graph({
        {"declared_decode_scratch", 256 * 1024, 256, true},
    });
    ASSERT_TRUE(allocator.allocateForGraph(
        graph,
        hints,
        {requestFor(declared_graph, *device)},
        config));

    DeviceWorkspaceManager *workspace =
        allocator.getDeviceWorkspace(*device);
    ASSERT_NE(workspace, nullptr);
    const size_t physical_bytes = workspace->used();
    const uint64_t generation =
        allocator.deviceGeneration(*device);

    MockWorkspaceConsumer omitted_maintenance({
        {"moe_rebalance_local_histogram_test", 8 * 1024, 256, true},
    });
    EXPECT_FALSE(allocator.allocateForGraph(
        graph,
        hints,
        {requestFor(omitted_maintenance, *device)},
        config));

    EXPECT_EQ(workspace->used(), physical_bytes);
    EXPECT_EQ(allocator.deviceGeneration(*device), generation);
    EXPECT_FALSE(
        workspace->hasBuffer(
            "moe_rebalance_local_histogram_test"));
    EXPECT_EQ(omitted_maintenance.boundWorkspace(), nullptr);
}

/**
 * @brief Compact grouped-decode demand is preflighted independently of prefill M.
 *
 * Candidate workspaces can be non-monotonic: M=16 may need reduction partials
 * that M=4096 does not. This regression proves the configured compact bound is
 * queried even when the active participant has another row count.
 */
TEST(Test__WorkspaceAllocator, SerialFamilyPreflightsMaximumCompactRows)
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
    hints.serial_family_max_compact_rows = 16;
    hints.graph_family_policy =
        WorkspaceGraphFamilyPolicy::SerialDeviceFamilyExactParticipant;

    auto stage = std::make_unique<DeclaredShapeWorkspaceStage>(
        *device,
        std::vector<size_t>{4096, 8},
        std::vector<size_t>{4096, 16},
        /*declare_decode_only_buffer=*/false,
        /*scale_scratch_with_m=*/true);
    auto *stage_ptr = stage.get();
    graph.addNode(
        "non_monotonic_compact_workspace",
        std::move(stage),
        *device);

    ASSERT_TRUE(allocator.allocateForGraph(
        graph,
        hints,
        {},
        unitBudgetConfig()));
    EXPECT_GE(stage_ptr->requirementsCalls(), 3);
    EXPECT_EQ(stage_ptr->maxM(), 4096);
    EXPECT_EQ(
        stage_ptr->boundWorkspace()->getBufferSize(
            "declared_shape_scratch"),
        size_t{4096} * 256);

    /*
     * maxM alone cannot prove the compact query because the active graph is
     * larger. The final query is the largest-participant request only when that
     * policy is selected, so ExactParticipant leaves compact M=16 observable.
     */
    EXPECT_EQ(stage_ptr->lastM(), 16);
}

/**
 * @brief Lifecycle tags, rather than M alone, keep compact scratch out of prefill.
 *
 * Grouped verification is M-total and can therefore request an arbitrarily
 * large runtime row count while tiling through bounded scratch. Kernels cannot
 * infer "prefill" from M. They declare the compact buffer for every M>1 and
 * this role-aware planner removes it only from the active large-M participant,
 * then publishes it through the independently packed compact participant.
 */
TEST(Test__WorkspaceAllocator, CompactRegimeDescriptorAliasesInsteadOfInflatingPrefill)
{
    auto device = selectAvailableGpuWithMemory();
    if (!device)
    {
        GTEST_SKIP() << "No CUDA/ROCm GPU with enough free memory for WorkspaceAllocator unit test";
    }

    WorkspaceAllocator allocator;
    ComputeGraph graph;
    auto hints = tinyHints();
    hints.max_seq_len = 64;
    hints.serial_family_max_compact_rows = 16;
    hints.graph_family_policy =
        WorkspaceGraphFamilyPolicy::SerialDeviceFamilyExactParticipant;

    auto stage = std::make_unique<DeclaredShapeWorkspaceStage>(
        *device,
        std::vector<size_t>{64, 8},
        std::vector<size_t>{64, 16},
        /*declare_decode_only_buffer=*/false,
        /*scale_scratch_with_m=*/true,
        /*declare_compact_regime_buffer=*/true);
    graph.addNode(
        "role_aware_compact_workspace",
        std::move(stage),
        *device);

    ASSERT_TRUE(allocator.allocateForGraph(
        graph,
        hints,
        {},
        unitBudgetConfig()));

    DeviceWorkspaceManager *workspace =
        allocator.getDeviceWorkspace(*device);
    ASSERT_NE(workspace, nullptr);
    ASSERT_TRUE(workspace->hasBuffer("declared_shape_scratch"));
    ASSERT_TRUE(workspace->hasBuffer("compact_regime_scratch"));

    constexpr size_t kActiveParticipantBytes = 64 * 256;
    EXPECT_EQ(workspace->primaryBlockSize(), kActiveParticipantBytes)
        << "The compact descriptor must be absent from the large-M layout; "
           "summing it with prefill would reserve 28 KiB instead of 16 KiB.";
    EXPECT_EQ(workspace->used(), kActiveParticipantBytes);

    const auto active_begin = reinterpret_cast<std::uintptr_t>(
        workspace->getBuffer("declared_shape_scratch"));
    const auto active_end = active_begin + kActiveParticipantBytes;
    const auto compact_begin = reinterpret_cast<std::uintptr_t>(
        workspace->getBuffer("compact_regime_scratch"));
    const auto compact_end = compact_begin + 12 * 1024;
    EXPECT_GE(compact_begin, active_begin);
    EXPECT_LE(compact_end, active_end)
        << "Compact grouped-verifier storage must alias the serial prefill block.";
}

/**
 * @brief Prefill-only scratch is absent from decode and grouped participants.
 *
 * A kernel object may expose both NativeVNNI prefill accumulators and grouped
 * verifier KPAR storage. Device-event ordering makes those graph roles mutually
 * exclusive, so the compact arena must be able to occupy the prefill interval
 * rather than forcing the physical workspace to contain their union.
 */
TEST(Test__WorkspaceAllocator, PrefillRegimeDescriptorIsReusableByCompactParticipant)
{
    auto device = selectAvailableGpuWithMemory();
    if (!device)
    {
        GTEST_SKIP() << "No CUDA/ROCm GPU with enough free memory for WorkspaceAllocator unit test";
    }

    WorkspaceAllocator allocator;
    ComputeGraph graph;
    auto hints = tinyHints();
    hints.max_seq_len = 64;
    hints.serial_family_max_compact_rows = 16;
    hints.graph_family_policy =
        WorkspaceGraphFamilyPolicy::SerialDeviceFamilyExactParticipant;

    auto stage = std::make_unique<DeclaredShapeWorkspaceStage>(
        *device,
        std::vector<size_t>{64, 8},
        std::vector<size_t>{64, 16},
        /*declare_decode_only_buffer=*/false,
        /*scale_scratch_with_m=*/false,
        /*declare_compact_regime_buffer=*/true,
        /*compact_rows_need_larger_shared_buffer=*/false,
        /*declare_prefill_regime_buffer=*/true);
    graph.addNode(
        "role_aware_prefill_workspace",
        std::move(stage),
        *device);

    ASSERT_TRUE(allocator.allocateForGraph(
        graph,
        hints,
        {},
        unitBudgetConfig()));

    DeviceWorkspaceManager *workspace =
        allocator.getDeviceWorkspace(*device);
    ASSERT_NE(workspace, nullptr);
    ASSERT_TRUE(workspace->hasBuffer("prefill_regime_scratch"));
    ASSERT_TRUE(workspace->hasBuffer("compact_regime_scratch"));

    const auto prefill_begin = reinterpret_cast<std::uintptr_t>(
        workspace->getBuffer("prefill_regime_scratch"));
    const auto compact_begin = reinterpret_cast<std::uintptr_t>(
        workspace->getBuffer("compact_regime_scratch"));
    EXPECT_EQ(prefill_begin, compact_begin)
        << "Mutually exclusive prefill and compact descriptors should reuse "
           "the same largest-first interval.";
    EXPECT_EQ(workspace->primaryBlockSize(), size_t{16} * 1024)
        << "The physical block should contain common scratch plus the larger "
           "of the two role-exclusive 12 KiB arenas, not their union.";
}

/**
 * @brief Fixed-shape auxiliary consumers never inherit graph token rows.
 *
 * MTP metadata uses M as request count. This regression models the production
 * failure where a one-request metadata block was first captured at four bytes
 * and a later 4096-row prefill attempted to reinterpret it as 4096 requests.
 */
TEST(Test__WorkspaceAllocator, FixedShapeExtraConsumerPreservesRequestCountAcrossFamily)
{
    auto device = selectAvailableGpuWithMemory();
    if (!device)
    {
        GTEST_SKIP() << "No CUDA/ROCm GPU with enough free memory for WorkspaceAllocator unit test";
    }

    WorkspaceAllocator allocator;
    ComputeGraph graph;
    auto hints = tinyHints();
    hints.max_seq_len = 37;
    hints.serial_family_max_rows = 4096;
    hints.serial_family_max_compact_rows = 16;
    hints.graph_family_policy =
        WorkspaceGraphFamilyPolicy::SerialDeviceFamilyLargestParticipant;

    MockWorkspaceConsumer metadata_consumer({
        {"fixed_request_metadata", sizeof(int32_t), 256, true},
    });
    WorkspaceConsumerRequest request{
        .consumer = &metadata_consumer,
        .device = *device,
        .m = 1,
        .n = 3,
        .k = 0,
        .shape_policy =
            WorkspaceConsumerShapePolicy::FixedDeclaredShape,
    };

    ASSERT_TRUE(allocator.allocateForGraph(
        graph,
        hints,
        {request},
        unitBudgetConfig()));
    EXPECT_EQ(metadata_consumer.maxM(), 1)
        << "Prompt and grouped token rows must not be substituted for the "
           "metadata request-count dimension.";
    EXPECT_EQ(metadata_consumer.lastN(), 3);
    ASSERT_NE(metadata_consumer.boundWorkspace(), nullptr);
    EXPECT_EQ(
        metadata_consumer.boundWorkspace()->getBufferSize(
            "fixed_request_metadata"),
        sizeof(int32_t));
}

/**
 * @brief Shared names publish their family-wide maximum before first capture.
 *
 * Attention conversion scratch is non-monotonic in the real graph: compact
 * grouped verification can require a larger buffer than active prefill. The
 * name and pointer are shared, so the allocator must promote the active
 * descriptor to the compact participant's capacity before publishing it.
 */
TEST(Test__WorkspaceAllocator, SharedSerialNameUsesMaximumCapacityAcrossParticipants)
{
    auto device = selectAvailableGpuWithMemory();
    if (!device)
    {
        GTEST_SKIP() << "No CUDA/ROCm GPU with enough free memory for WorkspaceAllocator unit test";
    }

    WorkspaceAllocator allocator;
    ComputeGraph graph;
    auto hints = tinyHints();
    hints.max_seq_len = 64;
    hints.serial_family_max_compact_rows = 16;
    hints.graph_family_policy =
        WorkspaceGraphFamilyPolicy::SerialDeviceFamilyExactParticipant;

    auto stage = std::make_unique<DeclaredShapeWorkspaceStage>(
        *device,
        std::vector<size_t>{64, 8},
        std::vector<size_t>{64, 16},
        /*declare_decode_only_buffer=*/false,
        /*scale_scratch_with_m=*/false,
        /*declare_compact_regime_buffer=*/false,
        /*compact_rows_need_larger_shared_buffer=*/true);
    graph.addNode(
        "non_monotonic_shared_workspace",
        std::move(stage),
        *device);

    ASSERT_TRUE(allocator.allocateForGraph(
        graph,
        hints,
        {},
        unitBudgetConfig()));

    DeviceWorkspaceManager *workspace =
        allocator.getDeviceWorkspace(*device);
    ASSERT_NE(workspace, nullptr);
    EXPECT_EQ(
        workspace->getBufferSize("declared_shape_scratch"),
        size_t{48} * 1024)
        << "The first captured address must satisfy the largest participant "
           "that shares this workspace name.";
    EXPECT_EQ(workspace->primaryBlockSize(), size_t{48} * 1024);
}

/**
 * @brief Distinct graph topologies are planned together before first capture.
 *
 * This reproduces the production MTP failure at allocator scale.  The main
 * graph first advertised a smaller shared GEMM bank; a later sidecar needed a
 * larger capacity and introduced sidecar-only control workspace.  A complete
 * family declaration must publish both names at generation one, retain the
 * sidecar's exact M/K geometry, and alias mutually exclusive graph-local
 * storage instead of summing every graph.
 */
TEST(Test__WorkspaceAllocator, CompleteSerialGraphFamilyPublishesSidecarTopologyAtGenerationOne)
{
    auto device = selectAvailableGpuWithMemory();
    if (!device)
    {
        GTEST_SKIP() << "No CUDA/ROCm GPU with enough free memory for WorkspaceAllocator unit test";
    }

    constexpr size_t kPrimaryOnlyBytes = 128 * 1024;
    constexpr size_t kSidecarOnlyBytes = 192 * 1024;
    constexpr size_t kPrimarySharedBytes = 256 * 1024;
    constexpr size_t kSidecarSharedBytes = 512 * 1024;

    ComputeGraph primary_graph;
    auto primary_stage = std::make_unique<GraphFamilyWorkspaceStage>(
        *device,
        std::vector<size_t>{4096, 2048},
        std::vector<WorkspaceDescriptor>{
            {"shared_gemm_bank", kPrimarySharedBytes, 256, true},
            {"primary_only_bank", kPrimaryOnlyBytes, 256, true},
        });
    auto *primary_stage_ptr = primary_stage.get();
    primary_graph.addNode(
        "primary_projection",
        std::move(primary_stage),
        *device);

    ComputeGraph sidecar_graph;
    auto sidecar_stage = std::make_unique<GraphFamilyWorkspaceStage>(
        *device,
        std::vector<size_t>{5, 4096},
        std::vector<WorkspaceDescriptor>{
            {"shared_gemm_bank", kSidecarSharedBytes, 256, true},
            {"sidecar_only_bank", kSidecarOnlyBytes, 256, true},
        });
    auto *sidecar_stage_ptr = sidecar_stage.get();
    sidecar_graph.addNode(
        "mtp_sidecar_projection",
        std::move(sidecar_stage),
        *device);

    WorkspaceAllocator allocator;
    auto hints = tinyHints();
    hints.max_seq_len = 4096;
    hints.serial_family_max_rows = 4096;
    hints.serial_family_max_compact_rows = 16;
    hints.graph_family_policy =
        WorkspaceGraphFamilyPolicy::SerialDeviceFamilyLargestParticipant;

    ASSERT_TRUE(allocator.allocateForGraphFamily(
        primary_graph,
        WorkspaceGraphParticipantRole::Decode,
        {WorkspaceGraphParticipant{
            .graph = &sidecar_graph,
            .role =
                WorkspaceGraphParticipantRole::GroupedVerifier,
        }},
        hints,
        {},
        unitBudgetConfig()));

    DeviceWorkspaceManager *workspace =
        allocator.getDeviceWorkspace(*device);
    ASSERT_NE(workspace, nullptr);
    EXPECT_EQ(allocator.deviceGeneration(*device), 1U);
    EXPECT_EQ(primary_stage_ptr->boundWorkspace(), workspace);
    EXPECT_EQ(sidecar_stage_ptr->boundWorkspace(), workspace);
    EXPECT_EQ(sidecar_stage_ptr->lastM(), 5);
    EXPECT_EQ(sidecar_stage_ptr->lastK(), 4096);
    EXPECT_EQ(
        workspace->getBufferSize("shared_gemm_bank"),
        kSidecarSharedBytes);
    EXPECT_TRUE(workspace->hasBuffer("primary_only_bank"));
    EXPECT_TRUE(workspace->hasBuffer("sidecar_only_bank"));

    /*
     * The primary graph observes only the first 256 KiB of the shared name, so
     * its local bank may reuse the canonical shared buffer's otherwise-unused
     * tail.  The sidecar observes all 512 KiB and therefore places its local
     * bank after that range.  This directional live-extent rule is why the
     * complete family still occupies only 512 + 192 KiB.
     */
    EXPECT_EQ(
        workspace->primaryBlockSize(),
        kSidecarSharedBytes + kSidecarOnlyBytes);
    const auto shared_begin = reinterpret_cast<std::uintptr_t>(
        workspace->getBuffer("shared_gemm_bank"));
    const auto primary_only_begin = reinterpret_cast<std::uintptr_t>(
        workspace->getBuffer("primary_only_bank"));
    const auto sidecar_only_begin = reinterpret_cast<std::uintptr_t>(
        workspace->getBuffer("sidecar_only_bank"));
    EXPECT_GE(
        primary_only_begin,
        shared_begin + kPrimarySharedBytes);
    EXPECT_GE(
        sidecar_only_begin,
        shared_begin + kSidecarSharedBytes);
}

/**
 * @brief External workspace remains live beside every serial graph participant.
 *
 * KV-cache and resident-metadata consumers are not independent graph roles:
 * their scratch can be used from prefill, decode, and grouped verification.
 * This regression prevents the family planner from aliasing an exact graph's
 * local buffer over a common external buffer merely because the primary graph
 * happened to be the first participant declared.
 */
TEST(Test__WorkspaceAllocator, CommonConsumerDoesNotAliasAnyExactParticipant)
{
    const auto device = selectAvailableGpuWithMemory();
    if (!device.has_value())
    {
        GTEST_SKIP() << "No GPU with enough free memory for workspace allocation";
    }

    constexpr size_t kBufferBytes = 64 * 1024;
    ComputeGraph primary_graph;
    primary_graph.addNode(
        "primary_stage",
        std::make_unique<GraphFamilyWorkspaceStage>(
            *device,
            std::vector<size_t>{1, 64},
            std::vector<WorkspaceDescriptor>{
                {"primary_graph_live", kBufferBytes, 256, true},
            }),
        *device);

    ComputeGraph exact_prefill_graph;
    exact_prefill_graph.addNode(
        "exact_prefill_stage",
        std::make_unique<GraphFamilyWorkspaceStage>(
            *device,
            std::vector<size_t>{64, 64},
            std::vector<WorkspaceDescriptor>{
                {"exact_prefill_live", kBufferBytes, 256, true},
            }),
        *device);

    MockWorkspaceConsumer common_consumer({
        {"common_kvcache_live", kBufferBytes, 256, true},
    });
    WorkspaceConsumerRequest common_request =
        requestFor(common_consumer, *device);

    WorkspaceAllocator allocator;
    auto hints = tinyHints();
    hints.max_seq_len = 1;
    hints.serial_family_max_rows = 64;
    hints.serial_family_max_compact_rows = 16;
    hints.graph_family_policy =
        WorkspaceGraphFamilyPolicy::
            SerialDeviceFamilyLargestParticipant;

    ASSERT_TRUE(allocator.allocateForGraphFamily(
        primary_graph,
        WorkspaceGraphParticipantRole::Decode,
        {WorkspaceGraphParticipant{
            .graph = &exact_prefill_graph,
            .role = WorkspaceGraphParticipantRole::Prefill,
        }},
        hints,
        {common_request},
        unitBudgetConfig()));

    DeviceWorkspaceManager *workspace =
        allocator.getDeviceWorkspace(*device);
    ASSERT_NE(workspace, nullptr);
    ASSERT_EQ(common_consumer.boundWorkspace(), workspace);

    const auto common_begin = reinterpret_cast<std::uintptr_t>(
        workspace->getBuffer("common_kvcache_live"));
    const auto common_end = common_begin + kBufferBytes;
    for (const char *graph_buffer :
         {"primary_graph_live", "exact_prefill_live"})
    {
        const auto graph_begin = reinterpret_cast<std::uintptr_t>(
            workspace->getBuffer(graph_buffer));
        const auto graph_end = graph_begin + kBufferBytes;
        EXPECT_TRUE(
            graph_end <= common_begin ||
            common_end <= graph_begin)
            << graph_buffer
            << " must not overlap common external-consumer storage";
    }
}

/**
 * @brief Exact attention participants use semantic head geometry.
 *
 * Attention consumes a `[rows, d_model]` activation, while split-decode
 * workspace is indexed by request, head, split, and head dimension. This
 * regression prevents exact graph-family scans from interpreting the
 * activation's `d_model` width as `head_dim`, which previously reported a
 * spurious four-times-larger capacity only when the real decode graph arrived.
 */
TEST(Test__WorkspaceAllocator, ExactAttentionUsesHeadDimensionNotActivationWidth)
{
    const auto device = selectAvailableGpuWithMemory();
    if (!device.has_value())
    {
        GTEST_SKIP() << "No GPU with enough free memory for workspace allocation";
    }

    ComputeGraph primary_graph;
    primary_graph.addNode(
        "primary_stage",
        std::make_unique<GraphFamilyWorkspaceStage>(
            *device,
            std::vector<size_t>{1, 4096},
            std::vector<WorkspaceDescriptor>{
                {"primary_scratch", 4096, 256, true},
            }),
        *device);

    auto exact_attention_stage =
        std::make_unique<GraphFamilyWorkspaceStage>(
            *device,
            std::vector<size_t>{1, 4096},
            std::vector<WorkspaceDescriptor>{
                {"attention_scratch", 4096, 256, true},
            });
    auto *exact_attention = exact_attention_stage.get();

    ComputeGraph exact_decode_graph;
    exact_decode_graph.addNode(
        "layer3_attention",
        std::move(exact_attention_stage),
        *device);

    WorkspaceAllocator allocator;
    auto hints = tinyHints();
    hints.n_heads = 64;
    hints.head_dim = 128;
    hints.d_model = 4096;
    hints.graph_family_policy =
        WorkspaceGraphFamilyPolicy::
            SerialDeviceFamilyLargestParticipant;

    ASSERT_TRUE(allocator.allocateForGraphFamily(
        primary_graph,
        WorkspaceGraphParticipantRole::Prefill,
        {WorkspaceGraphParticipant{
            .graph = &exact_decode_graph,
            .role = WorkspaceGraphParticipantRole::Decode,
        }},
        hints,
        {},
        unitBudgetConfig()));

    EXPECT_EQ(exact_attention->lastM(), 1);
    EXPECT_EQ(exact_attention->lastN(), 64);
    EXPECT_EQ(exact_attention->lastK(), 128);
    EXPECT_NE(exact_attention->lastK(), 4096)
        << "Activation d_model is not an attention workspace axis";
}

/**
 * @brief Device-owned maintenance records survive later graph execution.
 *
 * The maintenance graph finishes before the next prefill/decode participant,
 * but its status and controller records remain live until the request epilogue
 * copies diagnostics to the host. Treating execution completion as storage
 * death allowed a later prefill to overwrite those records with arbitrary
 * activations. This regression proves a persistent participant remains disjoint
 * from every graph-local layout while those local layouts may still alias.
 */
TEST(Test__WorkspaceAllocator, PersistentParticipantDoesNotAliasSerialGraphLocalStorage)
{
    const auto device = selectAvailableGpuWithMemory();
    if (!device.has_value())
    {
        GTEST_SKIP() << "No GPU with enough free memory for workspace allocation";
    }

    constexpr size_t kLocalBytes = 64 * 1024;
    constexpr size_t kPersistentBytes = 32 * 1024;

    ComputeGraph decode_graph;
    decode_graph.addNode(
        "decode_stage",
        std::make_unique<GraphFamilyWorkspaceStage>(
            *device,
            std::vector<size_t>{1, 64},
            std::vector<WorkspaceDescriptor>{
                {"decode_graph_local", kLocalBytes, 256, true},
            }),
        *device);

    ComputeGraph prefill_graph;
    prefill_graph.addNode(
        "prefill_stage",
        std::make_unique<GraphFamilyWorkspaceStage>(
            *device,
            std::vector<size_t>{64, 64},
            std::vector<WorkspaceDescriptor>{
                {"prefill_graph_local", kLocalBytes, 256, true},
            }),
        *device);

    ComputeGraph maintenance_graph;
    auto maintenance_stage =
        std::make_unique<GraphFamilyWorkspaceStage>(
            *device,
            std::vector<size_t>{1, 1},
            std::vector<WorkspaceDescriptor>{
                {"maintenance_status",
                 kPersistentBytes,
                 256,
                 true},
                {"maintenance_controller_state",
                 kPersistentBytes,
                 256,
                 true},
            });
    auto *maintenance_stage_ptr =
        maintenance_stage.get();
    maintenance_graph.addNode(
        "moe_device_rebalance_maintenance",
        std::move(maintenance_stage),
        *device);

    WorkspaceAllocator allocator;
    auto hints = tinyHints();
    hints.max_seq_len = 1;
    hints.serial_family_max_rows = 64;
    hints.graph_family_policy =
        WorkspaceGraphFamilyPolicy::
            SerialDeviceFamilyLargestParticipant;

    ASSERT_TRUE(allocator.allocateForGraphFamily(
        decode_graph,
        WorkspaceGraphParticipantRole::Decode,
        {
            WorkspaceGraphParticipant{
                .graph = &prefill_graph,
                .role =
                    WorkspaceGraphParticipantRole::Prefill,
            },
            WorkspaceGraphParticipant{
                .graph = &maintenance_graph,
                .role =
                    WorkspaceGraphParticipantRole::
                        MoERebalanceMaintenance,
                .lifetime =
                    WorkspaceGraphParticipantLifetime::
                        PersistentAcrossParticipants,
            },
        },
        hints,
        {},
        unitBudgetConfig()));

    DeviceWorkspaceManager *workspace =
        allocator.getDeviceWorkspace(*device);
    ASSERT_NE(workspace, nullptr);
    EXPECT_EQ(
        maintenance_stage_ptr->boundWorkspace(),
        workspace);

    const auto interval =
        [&](const char *name, size_t bytes)
    {
        const auto begin =
            reinterpret_cast<std::uintptr_t>(
                workspace->getBuffer(name));
        return std::pair{
            begin,
            begin + bytes};
    };
    const auto overlaps =
        [](const auto &lhs, const auto &rhs)
    {
        return lhs.first < rhs.second &&
               rhs.first < lhs.second;
    };

    const auto decode =
        interval("decode_graph_local", kLocalBytes);
    const auto prefill =
        interval("prefill_graph_local", kLocalBytes);
    const auto status =
        interval("maintenance_status", kPersistentBytes);
    const auto controller =
        interval(
            "maintenance_controller_state",
            kPersistentBytes);

    EXPECT_FALSE(overlaps(status, decode));
    EXPECT_FALSE(overlaps(status, prefill));
    EXPECT_FALSE(overlaps(controller, decode));
    EXPECT_FALSE(overlaps(controller, prefill));
    EXPECT_FALSE(overlaps(status, controller));
}

/**
 * @brief Exact M=16 graphs retain their role instead of inferring it from M.
 *
 * An ordinary sixteen-token prompt and a depth-fifteen grouped verifier have
 * identical row geometry but disjoint workspace regimes. This regression
 * proves the typed participant role, rather than `M > 1`, controls filtering.
 */
TEST(Test__WorkspaceAllocator, ExactParticipantRoleDisambiguatesPrefillFromGroupedM)
{
    auto device = selectAvailableGpuWithMemory();
    if (!device)
    {
        GTEST_SKIP() << "No CUDA/ROCm GPU with enough free memory for WorkspaceAllocator unit test";
    }

    ComputeGraph primary_graph;
    primary_graph.addNode(
        "primary",
        std::make_unique<GraphFamilyWorkspaceStage>(
            *device,
            std::vector<size_t>{1, 128},
            std::vector<WorkspaceDescriptor>{
                {"shared", 4096, 256, true},
            }),
        *device);

    ComputeGraph sixteen_row_prefill;
    sixteen_row_prefill.addNode(
        "sixteen_row_prefill",
        std::make_unique<GraphFamilyWorkspaceStage>(
            *device,
            std::vector<size_t>{16, 128},
            std::vector<WorkspaceDescriptor>{
                {"prefill_only",
                 8192,
                 256,
                 true,
                 WorkspaceExecutionRegime::PrefillOnly},
                {"compact_only",
                 16384,
                 256,
                 true,
                 WorkspaceExecutionRegime::CompactDecodeOnly},
            }),
        *device);

    WorkspaceAllocator allocator;
    auto hints = tinyHints();
    hints.max_seq_len = 1;
    hints.serial_family_max_rows = 16;
    hints.serial_family_max_compact_rows = 16;
    hints.graph_family_policy =
        WorkspaceGraphFamilyPolicy::SerialDeviceFamilyExactParticipant;

    ASSERT_TRUE(allocator.allocateForGraphFamily(
        primary_graph,
        WorkspaceGraphParticipantRole::Decode,
        {WorkspaceGraphParticipant{
            .graph = &sixteen_row_prefill,
            .role = WorkspaceGraphParticipantRole::Prefill,
        }},
        hints,
        {},
        unitBudgetConfig()));

    DeviceWorkspaceManager *workspace =
        allocator.getDeviceWorkspace(*device);
    ASSERT_NE(workspace, nullptr);
    EXPECT_TRUE(workspace->hasBuffer("prefill_only"));
    EXPECT_FALSE(workspace->hasBuffer("compact_only"));
}

/**
 * @brief Exact prefill graphs size terminal projection by requests, not tokens.
 *
 * The LM-head input tensor spans all 4096 hidden rows even though the graph
 * selects and projects only one terminal row for each request. Feeding the
 * tensor's first dimension into GEMM workspace planning creates multi-gigabyte
 * accumulator banks that no production launch consumes.
 */
TEST(Test__WorkspaceAllocator, ExactPrefillTerminalProjectionUsesRequestCardinality)
{
    auto device = selectAvailableGpuWithMemory();
    if (!device)
    {
        GTEST_SKIP() << "No CUDA/ROCm GPU with enough free memory for WorkspaceAllocator unit test";
    }

    ComputeGraph primary_graph;
    primary_graph.addNode(
        "primary",
        std::make_unique<GraphFamilyWorkspaceStage>(
            *device,
            std::vector<size_t>{1, 128},
            std::vector<WorkspaceDescriptor>{
                {"primary_shared", 4096, 256, true},
            }),
        *device);

    ComputeGraph maximum_prefill_graph;
    auto lm_head = std::make_unique<DeclaredShapeWorkspaceStage>(
        *device,
        std::vector<size_t>{4096, 4096},
        std::vector<size_t>{1, 248320},
        /*declare_decode_only_buffer=*/false,
        /*scale_scratch_with_m=*/true);
    auto *lm_head_ptr = lm_head.get();
    maximum_prefill_graph.addNode(
        "lm_head",
        std::move(lm_head),
        *device);

    WorkspaceAllocator allocator;
    auto hints = tinyHints();
    hints.max_seq_len = 1;
    hints.batch_size = 1;
    hints.vocab_size = 248320;
    hints.serial_family_max_terminal_projection_columns =
        hints.vocab_size;
    hints.graph_family_policy =
        WorkspaceGraphFamilyPolicy::SerialDeviceFamilyExactParticipant;

    ASSERT_TRUE(allocator.allocateForGraphFamily(
        primary_graph,
        WorkspaceGraphParticipantRole::Decode,
        {WorkspaceGraphParticipant{
            .graph = &maximum_prefill_graph,
            .role = WorkspaceGraphParticipantRole::Prefill,
        }},
        hints,
        {},
        unitBudgetConfig()));

    EXPECT_EQ(lm_head_ptr->maxM(), 1);
    EXPECT_FALSE(lm_head_ptr->sawDeclaredM());
}

/**
 * @brief Exact grouped LM-head workspace retains every verifier row.
 *
 * Ordinary prefill projects only one terminal row per request, but grouped MTP
 * verification projects every compact row. Both can have the same numeric M;
 * this regression proves that role, rather than geometry, selects cardinality.
 */
TEST(Test__WorkspaceAllocator, ExactGroupedTerminalProjectionUsesVerifierCardinality)
{
    const auto device = selectAvailableGpuWithMemory();
    if (!device.has_value())
    {
        GTEST_SKIP() << "No GPU with enough free memory for workspace allocation";
    }

    ComputeGraph grouped_graph;
    auto lm_head = std::make_unique<DeclaredShapeWorkspaceStage>(
        *device,
        std::vector<size_t>{15, 4096},
        std::vector<size_t>{15, 248320},
        /*declare_decode_only_buffer=*/false,
        /*scale_scratch_with_m=*/true);
    auto *lm_head_ptr = lm_head.get();
    grouped_graph.addNode(
        "lm_head",
        std::move(lm_head),
        *device);

    WorkspaceAllocator allocator;
    auto hints = tinyHints();
    hints.max_seq_len = 15;
    hints.batch_size = 1;
    hints.vocab_size = 248320;
    hints.serial_family_max_terminal_projection_columns =
        hints.vocab_size;
    hints.graph_family_policy =
        WorkspaceGraphFamilyPolicy::
            SerialDeviceFamilyExactParticipant;

    ASSERT_TRUE(allocator.allocateForGraphFamily(
        grouped_graph,
        WorkspaceGraphParticipantRole::GroupedVerifier,
        {},
        hints,
        {},
        unitBudgetConfig()));

    EXPECT_EQ(lm_head_ptr->maxM(), 15);
    EXPECT_TRUE(lm_head_ptr->sawDeclaredM());
}

TEST(Test__WorkspaceAllocator, GraphConsumerAllocatesAndBindsCPUWorkspaceForDeclaredStage)
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
    ASSERT_NE(workspace, nullptr)
        << "CPU kernels with declared persistent scratch are graph-family workspace consumers";
    EXPECT_EQ(raw_stage->boundWorkspace(), workspace);
    EXPECT_GT(raw_stage->requirementsCalls(), 0);
    EXPECT_EQ(raw_stage->bindCalls(), 1);
    EXPECT_TRUE(workspace->hasBuffer("declared_shape_scratch"));
    EXPECT_GE(
        workspace->getBufferSize("declared_shape_scratch"),
        size_t{4096});
}

TEST(Test__WorkspaceAllocator, ExtraCPUConsumersUseTheSameAppendOnlyWorkspaceLifecycle)
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
    ASSERT_NE(initial_workspace, nullptr);
    EXPECT_EQ(initial_consumer.boundWorkspace(), initial_workspace);
    EXPECT_GT(initial_consumer.requirementsCalls(), 0);
    EXPECT_EQ(initial_consumer.bindCalls(), 1);
    EXPECT_TRUE(initial_workspace->hasBuffer("shared_scratch"));
    EXPECT_TRUE(initial_workspace->hasBuffer("old_only_scratch"));
    const uint64_t initial_generation = allocator.deviceGeneration(DeviceId::cpu());
    EXPECT_GT(initial_generation, 0u);

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
    ASSERT_EQ(reallocated_workspace, initial_workspace)
        << "CPU graph-family growth must preserve the manager and old addresses";
    EXPECT_EQ(larger_consumer.boundWorkspace(), reallocated_workspace);
    EXPECT_GT(larger_consumer.requirementsCalls(), 0);
    EXPECT_EQ(larger_consumer.bindCalls(), 1);
    EXPECT_EQ(reallocated_workspace->getBufferSize("shared_scratch"), 4096u);
    EXPECT_TRUE(reallocated_workspace->hasBuffer("old_only_scratch"));
    EXPECT_TRUE(reallocated_workspace->hasBuffer("new_only_scratch"));
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

/**
 * @brief Repeated graph-family growth never enters an ABA/null-binding state.
 *
 * The old replacement design needed a null bind before destroying the manager
 * to avoid reusing the same host address for a different allocation. Append-only
 * growth removes that state transition entirely: the manager identity and
 * generation stay fixed, and consumers only ever observe that live manager.
 */
TEST(Test__WorkspaceAllocator, AppendOnlyGrowthNeverPublishesNullOrNewManager)
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
    const uint64_t generation = allocator.deviceGeneration(*device);
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

    ASSERT_EQ(consumer.bindSequence().size(), 2u);
    EXPECT_EQ(consumer.bindSequence()[0], w1);
    EXPECT_EQ(consumer.bindSequence()[1], w1);
    EXPECT_EQ(w2, w1);
    EXPECT_EQ(allocator.deviceGeneration(*device), generation);
    EXPECT_TRUE(std::none_of(
        consumer.bindSequence().begin(),
        consumer.bindSequence().end(),
        [](DeviceWorkspaceManager *workspace)
        {
            return workspace == nullptr;
        }));
}
