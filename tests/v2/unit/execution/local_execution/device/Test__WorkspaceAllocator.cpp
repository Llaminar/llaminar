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
    ComputeGraph graph;
    auto hints = tinyHints();
    hints.graph_family_policy =
        WorkspaceGraphFamilyPolicy::SerialDeviceFamilyExactParticipant;
    const auto config = unitBudgetConfig();

    constexpr size_t kHalfMiB = 512 * 1024;
    constexpr size_t kQuarterMiB = 256 * 1024;
    MockWorkspaceConsumer main_graph({
        {"main_projection_scratch", kHalfMiB, 256, true},
        {"main_attention_scratch", kHalfMiB, 256, true},
    });
    ASSERT_TRUE(allocator.allocateForGraph(
        graph,
        hints,
        {requestFor(main_graph, *device)},
        config));

    DeviceWorkspaceManager *workspace =
        allocator.getDeviceWorkspace(*device);
    ASSERT_NE(workspace, nullptr);
    const size_t physical_bytes = workspace->used();
    const uint64_t generation = allocator.deviceGeneration(*device);
    ASSERT_GE(workspace->primaryBlockSize(), 2 * kHalfMiB);

    MockWorkspaceConsumer grouped_verifier({
        {"verifier_layer0_state_slots", kQuarterMiB, 256, true},
        {"verifier_layer1_state_slots", kQuarterMiB, 256, true},
        {"verifier_layer2_state_slots", kQuarterMiB, 256, true},
    });
    ASSERT_TRUE(allocator.allocateForGraph(
        graph,
        hints,
        {requestFor(grouped_verifier, *device)},
        config));

    EXPECT_EQ(workspace->used(), physical_bytes);
    EXPECT_EQ(allocator.deviceGeneration(*device), generation);
    EXPECT_EQ(grouped_verifier.boundWorkspace(), workspace);

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

    const auto main_begin = reinterpret_cast<std::uintptr_t>(
        workspace->getBuffer("main_projection_scratch"));
    const auto main_end =
        reinterpret_cast<std::uintptr_t>(
            workspace->getBuffer("main_attention_scratch")) +
        kHalfMiB;
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
