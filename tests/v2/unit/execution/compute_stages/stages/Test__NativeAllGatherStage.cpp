/**
 * @file Test__NativeAllGatherStage.cpp
 * @brief Device-free validation of the native byte collective's graph contract.
 *
 * Host tensors supply metadata only. No test calls a GPU allocator or presents
 * host storage as executable GPU data. Native capture/replay and publication
 * are proved separately by the CUDA/ROCm integration registrations.
 */
#include <gtest/gtest.h>
#include "execution/compute_stages/stages/NativeAllGatherStage.h"
#include "execution/compute_stages/stages/NativeVocabularyAllGatherStage.h"
#include "collective/VocabularyGatherWorkspaceContract.h"
#include "models/GraphTypes.h"
#include "models/qwen35/Qwen35Graph.h"
#include "execution/compute_stages/stages/MTPVerifierOutcomeStage.h"
#include "execution/compute_stages/stages/TPLocalReduceOverlap.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "memory/StageBufferContract.h"
#include "tensors/GpuTensorView.h"
#include "tensors/Tensors.h"
#include "../../../../mocks/MockLocalTPContext.h"
#include "../../../../mocks/MockComputeStage.h"

#include <limits>
#include <array>

using namespace llaminar2;

namespace
{
    /** @brief Metadata-only counted exchange that becomes ready after graph construction. */
    class ColdCountedExchange final : public CapturedAllGatherStage
    {
    public:
        /** @brief Retain setup readiness and banks without allocating on a GPU. */
        ColdCountedExchange(DeviceId device, CapturedAllGatherBuffers buffers, std::shared_ptr<bool> ready)
            : CapturedAllGatherStage(device), buffers_(buffers), ready_(std::move(ready)) {}
        /** @return Explicit counted transport identity, preserved by both overlap edges. */
        ComputeStageType type() const override { return ComputeStageType::DEVICE_COUNTED_ALLGATHER; }
        /** @return Stable graph identity used by the metadata fixture. */
        std::string name() const override { return "intermediate_packet"; }
        /** @return Device-free tests deliberately never enqueue hardware work. */
        bool execute(IDeviceContext *) override { return false; }
        /** @return Only the declared family is meaningful; no backend is initialized. */
        bool supportsBackend(ComputeBackendType backend) const override
        { return backend == ComputeBackendType::GPU_ROCM; }
        /** @return No tensors are read back by a metadata-only fixture. */
        StageDumpInfo buildDumpInfoImpl() const override { return {}; }
        /** @return No hardware enqueue is possible in this metadata-only fixture. */
        bool enqueueAcquiredInput(const AcquiredDeviceTransferInput &) const override { return false; }
        /** @brief Reject attempts to treat the metadata fixture as executable storage. */
        void validateEnqueue(const StageGPUExecution &) const override { throw std::logic_error("metadata only"); }
        /** @return Original arena identities, not copies created by overlap lowering. */
        CapturedAllGatherBuffers exchangeBuffers() const override { return buffers_; }
        /** @return Both lifetimes remain visible at the completion edge. */
        StageBufferContract bufferContract() const override
        { return StageBufferContract::build().addInput(buffers_.input_id).addOutput(buffers_.output_id); }
        /** @return The retained original stage's readiness, modified by setup. */
        bool isGraphCapturable() const override { return *ready_; }
        /** @return Cold topology eligibility is independent of bound storage. */
        bool supportsGraphCaptureAfterLaunchPreparation() const override { return true; }
        /** @return Exact buckets support setup without executing inference. */
        bool supportsLazyPrefillGraphCapturePreflight() const override { return true; }
        /** @return Counted messages also support masked padded buckets. */
        bool supportsPaddedPrefillGraphCapturePreflight() const override { return true; }
    private:
        CapturedAllGatherBuffers buffers_;
        std::shared_ptr<bool> ready_;
    };

    /** @brief Immutable graph metadata over deliberately device-free storage. */
    class NativeAllGatherContract : public ::testing::Test
    {
    protected:
        test::MockLocalTPContext tp;
        FP32Tensor input{{4, 16}, DeviceId::cpu()};
        FP32Tensor output{{8, 16}, DeviceId::cpu()};
        NativeAllGatherStage::Params params;
        ComputeGraph graph;

        /** @brief Declare a native gather and disjoint local work without devices. */
        void declareWindow()
        {
            for (const auto *name : {"producer", "compute"})
                graph.addNode(name, std::make_unique<llaminar2::testing::MockComputeStage>(
                    ComputeStageType::GEMM, name, params.device_id), params.device_id);
            graph.addNode(params.stage_name, std::make_unique<NativeAllGatherStage>(params), params.device_id);
            graph.addDependency(params.stage_name, "producer");
        }

        /** @brief Reversed physical ordinals prevent confusing rank with device ID. */
        void SetUp() override
        {
            tp.setDevices({GlobalDeviceAddress::rocm(3), GlobalDeviceAddress::rocm(1)});
            tp.setBackend(CollectiveBackendType::RCCL);
            tp.setRawAllgatherGraphCaptureSupported(true);
            params.device_id = DeviceId::rocm(3);
            params.tp_ctx = &tp;
            params.local_input = &input;
            params.rank_major_output = &output;
            params.bytes_per_participant = 128;
            params.participant = 0;
            params.stage_name = "intermediate_packet";
            params.input_buffer_id = BufferId::HIDDEN_STATE;
            params.output_buffer_id = BufferId::NORMALIZED;
        }
    };

    TEST_F(NativeAllGatherContract, ExactBytePrefixAndReadWriteRolesAreExplicit)
    {
        // Byte transport may be unaligned and need not consume full capacity.
        for (std::size_t bytes : {1u, 3u, 4u, 128u, 255u, 256u})
        {
            params.bytes_per_participant = bytes;
            NativeAllGatherStage stage(params);
            EXPECT_EQ(stage.estimatedMemoryBytes(), bytes * 3);
            EXPECT_EQ(stage.estimatedFlops(), 0u);
            EXPECT_EQ(stage.type(), ComputeStageType::NATIVE_ALLGATHER);
            EXPECT_TRUE(isCollectiveComputeStageType(stage.type()));
            const auto contract = stage.bufferContract();
            ASSERT_EQ(contract.inputs.size(), 1u);
            ASSERT_EQ(contract.outputs.size(), 1u);
            EXPECT_TRUE(contract.inouts.empty());
            EXPECT_EQ(contract.inputs[0].id, *params.input_buffer_id);
            EXPECT_EQ(contract.outputs[0].id, *params.output_buffer_id);
            EXPECT_FALSE(stage.supportsBackend(ComputeBackendType::CPU));
        }
    }

    TEST_F(NativeAllGatherContract, RejectsMissingAndAliasedArenaBindings)
    {
        auto rejected = params;
        rejected.local_input = nullptr;
        EXPECT_THROW(NativeAllGatherStage{rejected}, std::invalid_argument);
        rejected = params;
        rejected.rank_major_output = nullptr;
        EXPECT_THROW(NativeAllGatherStage{rejected}, std::invalid_argument);
        rejected = params;
        rejected.rank_major_output = &input;
        EXPECT_THROW(NativeAllGatherStage{rejected}, std::invalid_argument);
        rejected = params;
        rejected.input_buffer_id.reset();
        EXPECT_THROW(NativeAllGatherStage{rejected}, std::invalid_argument);
        rejected = params;
        rejected.output_buffer_id.reset();
        EXPECT_THROW(NativeAllGatherStage{rejected}, std::invalid_argument);
        rejected = params;
        rejected.output_buffer_id = rejected.input_buffer_id;
        EXPECT_THROW(NativeAllGatherStage{rejected}, std::invalid_argument);
        rejected = params;
        rejected.stage_name.clear();
        EXPECT_THROW(NativeAllGatherStage{rejected}, std::invalid_argument);
    }

    TEST_F(NativeAllGatherContract, RejectsZeroOverflowAndShortPhysicalBuffers)
    {
        params.bytes_per_participant = 0;
        EXPECT_THROW(NativeAllGatherStage{params}, std::invalid_argument);
        params.bytes_per_participant = std::numeric_limits<std::size_t>::max();
        EXPECT_THROW(NativeAllGatherStage{params}, std::overflow_error);
        params.bytes_per_participant = input.size_bytes() + 1;
        EXPECT_THROW(NativeAllGatherStage{params}, std::invalid_argument);
        FP32Tensor short_output({63}, DeviceId::cpu());
        params.bytes_per_participant = 128;
        params.rank_major_output = &short_output;
        EXPECT_THROW(NativeAllGatherStage{params}, std::invalid_argument);
    }

    TEST_F(NativeAllGatherContract, RejectsWrongParticipantAndNonNativeMembership)
    {
        auto rejected = params;
        rejected.participant = 1; // Physical GPU 3 is communicator member zero.
        EXPECT_THROW(NativeAllGatherStage{rejected}, std::invalid_argument);
        rejected.participant = -1;
        EXPECT_THROW(NativeAllGatherStage{rejected}, std::invalid_argument);
        rejected.participant = 2;
        EXPECT_THROW(NativeAllGatherStage{rejected}, std::invalid_argument);
        rejected = params;
        rejected.device_id = DeviceId::cpu();
        EXPECT_THROW(NativeAllGatherStage{rejected}, std::invalid_argument);
        rejected = params;
        rejected.tp_ctx = nullptr;
        EXPECT_THROW(NativeAllGatherStage{rejected}, std::invalid_argument);
        tp.setBackend(CollectiveBackendType::NCCL);
        EXPECT_THROW(NativeAllGatherStage{params}, std::invalid_argument);
        tp.setBackend(CollectiveBackendType::RCCL);
        tp.setDevices({GlobalDeviceAddress::rocm(3), GlobalDeviceAddress::cuda(1)});
        EXPECT_THROW(NativeAllGatherStage{params}, std::invalid_argument);
        tp.setDevices({GlobalDeviceAddress::rocm(3), GlobalDeviceAddress::rocm(3)});
        EXPECT_THROW(NativeAllGatherStage{params}, std::invalid_argument);
        tp.setDevices({GlobalDeviceAddress::rocm(3)});
        EXPECT_THROW(NativeAllGatherStage{params}, std::invalid_argument);
    }

    TEST_F(NativeAllGatherContract, FrozenMembershipCannotBeReboundAfterConstruction)
    {
        NativeAllGatherStage stage(params);
        tp.setDevices({GlobalDeviceAddress::rocm(1), GlobalDeviceAddress::rocm(3)});
        EXPECT_THROW((void)stage.isGraphCapturable(), std::logic_error);
        tp.setDevices({GlobalDeviceAddress::rocm(3), GlobalDeviceAddress::rocm(1)});
        tp.setRawAllgatherGraphCaptureSupported(false);
        EXPECT_THROW((void)stage.isGraphCapturable(), std::invalid_argument);
        EXPECT_THROW(NativeAllGatherStage{params}, std::invalid_argument);
    }

    TEST_F(NativeAllGatherContract, CudaUsesSameContractWithoutInitializingDevice)
    {
        tp.setDevices({GlobalDeviceAddress::cuda(3), GlobalDeviceAddress::cuda(1)});
        tp.setBackend(CollectiveBackendType::NCCL);
        params.device_id = DeviceId::cuda(3);
        NativeAllGatherStage stage(params);
        EXPECT_EQ(stage.estimatedMemoryBytes(), 384u);
        EXPECT_FALSE(stage.supportsBackend(ComputeBackendType::GPU_ROCM));
        EXPECT_THROW((void)stage.requireGPUStream(), std::logic_error);
    }
    TEST_F(NativeAllGatherContract, RejectsPhysicalAliasingDespiteDistinctTensorIdentities)
    {
        // Pointer arithmetic is metadata-only: these views must never execute.
        std::array<std::uint8_t, 1024> storage{};
        GpuTensorView send(storage.data() + 64, 1, 32, TensorType::INT32, params.device_id);
        GpuTensorView receive(storage.data() + 128, 1, 64, TensorType::INT32, params.device_id);
        params.local_input = &send;
        params.rank_major_output = &receive;
        EXPECT_THROW(NativeAllGatherStage{params}, std::invalid_argument);
        receive.update_view(storage.data() + 192, 1);
        EXPECT_NO_THROW(NativeAllGatherStage{params}); // Exactly adjacent, not overlapping.
        receive.update_view(reinterpret_cast<void *>(std::numeric_limits<std::uintptr_t>::max() - 63), 1);
        EXPECT_THROW(NativeAllGatherStage{params}, std::invalid_argument);
    }

    TEST_F(NativeAllGatherContract, OverlapPreservesJoinIdentityAndBothBankLifetimes)
    {
        declareWindow();
        overlapTPLocalAllGather(graph, params.stage_name, "compute");
        EXPECT_EQ(graph.getExecutionOrder(), (std::vector<std::string>{
            "producer", "intermediate_packet_submit", "compute", "intermediate_packet"}));
        const auto *submit = graph.getNode("intermediate_packet_submit")->stage.get();
        const auto *join = graph.getNode(params.stage_name)->stage.get();
        EXPECT_EQ(submit->type(), ComputeStageType::NATIVE_ALLGATHER);
        EXPECT_EQ(join->type(), ComputeStageType::NATIVE_ALLGATHER);
        EXPECT_EQ(submit->bufferContract().inputs.at(0).id, *params.input_buffer_id);
        EXPECT_TRUE(submit->bufferContract().outputs.empty());
        EXPECT_EQ(join->bufferContract().inputs.at(0).id, *params.input_buffer_id);
        EXPECT_EQ(join->bufferContract().outputs.at(0).id, *params.output_buffer_id);
        EXPECT_THROW(overlapTPLocalAllGather(graph, params.stage_name, "compute"), std::invalid_argument);
    }

    TEST_F(NativeAllGatherContract, CountedOverlapRetainsColdEligibilityAndOriginalReadinessAuthority)
    {
        declareWindow();
        auto ready = std::make_shared<bool>(false);
        graph.getNode(params.stage_name)->stage = std::make_unique<ColdCountedExchange>(params.device_id,
            CapturedAllGatherBuffers{&input, &output, *params.input_buffer_id, *params.output_buffer_id}, ready);
        EXPECT_NO_THROW(overlapTPLocalAllGather(graph, params.stage_name, "compute"));
        EXPECT_EQ(graph.getExecutionOrder(), (std::vector<std::string>{
            "producer", "intermediate_packet_submit", "compute", "intermediate_packet"}));
        for (const auto *name : {"intermediate_packet_submit", "intermediate_packet"})
        {
            const auto *stage = graph.getNode(name)->stage.get();
            EXPECT_EQ(stage->type(), ComputeStageType::DEVICE_COUNTED_ALLGATHER);
            EXPECT_FALSE(stage->isGraphCapturable());
            EXPECT_TRUE(stage->supportsGraphCaptureAfterLaunchPreparation());
            EXPECT_TRUE(stage->supportsLazyPrefillGraphCapturePreflight());
            EXPECT_TRUE(stage->supportsPaddedPrefillGraphCapturePreflight());
        }
        *ready = true;
        EXPECT_TRUE(graph.getNode("intermediate_packet_submit")->stage->isGraphCapturable());
        EXPECT_TRUE(graph.getNode("intermediate_packet")->stage->isGraphCapturable());
    }

    TEST_F(NativeAllGatherContract, OverlapRejectsAnyUseOfEitherTransportBankBeforeMutation)
    {
        declareWindow();
        auto *compute = static_cast<llaminar2::testing::MockComputeStage *>(graph.getNode("compute")->stage.get());
        for (const auto id : {*params.input_buffer_id, *params.output_buffer_id})
            for (const auto &contract : {StageBufferContract::build().addInput(id),
                    StageBufferContract::build().addOutput(id), StageBufferContract::build().addInOut(id)})
            {
                compute->setBufferContract(contract);
                EXPECT_THROW(overlapTPLocalAllGather(graph, params.stage_name, "compute"), std::invalid_argument);
                EXPECT_EQ(graph.getExecutionOrder().size(), 3u);
                EXPECT_NE(dynamic_cast<NativeAllGatherStage *>(graph.getNode(params.stage_name)->stage.get()), nullptr);
            }
    }

    TEST_F(NativeAllGatherContract, OverlapRejectsCyclesForeignDevicesAndCaptureBoundaries)
    {
        declareWindow();
        auto *compute = graph.getNode("compute");
        compute->device = DeviceId::rocm(1);
        EXPECT_THROW(overlapTPLocalAllGather(graph, params.stage_name, "compute"), std::invalid_argument);
        compute->device = params.device_id;
        graph.addDependency("compute", params.stage_name);
        EXPECT_THROW(overlapTPLocalAllGather(graph, params.stage_name, "compute"), std::invalid_argument);
        compute->dependencies.clear();
        graph.addDependency("producer", "compute");
        EXPECT_THROW(overlapTPLocalAllGather(graph, params.stage_name, "compute"), std::invalid_argument);
        graph.getNode("producer")->dependencies.clear();
        graph.getNode(params.stage_name)->graph_capture_wave = GraphCaptureWaveContract{.identity = "other"};
        EXPECT_THROW(overlapTPLocalAllGather(graph, params.stage_name, "compute"), std::invalid_argument);
        EXPECT_EQ(graph.getExecutionOrder().size(), 3u);
    }
} // namespace

/** @brief Native terminal geometry is validated without GPU discovery or allocation. */
TEST_F(NativeAllGatherContract, VocabularyPublicationRetainsShardedProjectionOwnership)
{
    NativeVocabularyAllGatherStage::Params declaration;
    declaration.device_id = params.device_id;
    declaration.tp_ctx = &tp;
    declaration.local_logits = &input;
    declaration.full_logits = &output;
    declaration.rows = DeviceRowRange::deviceCounted(4, reinterpret_cast<const int32_t *>(0x1000));
    declaration.local_vocabulary = 16;
    declaration.vocabulary = 32;
    declaration.participant = 0;
    declaration.stage_name = "lm_head_allgather";
    NativeVocabularyAllGatherStage stage(declaration);
    EXPECT_TRUE(stage.isGraphCapturable());
    EXPECT_EQ(stage.getWorkspaceRequirements(4).buffers.front().size_bytes, 4u * 32u * sizeof(float));
    EXPECT_EQ(stage.bufferContract().inputs.front().id, BufferId::LOGITS_LOCAL);
    EXPECT_EQ(stage.bufferContract().outputs.front().id, BufferId::LOGITS);
    EXPECT_FALSE(stage.supportsBackend(ComputeBackendType::CPU));

    GraphConfig config;
    config.default_device = params.device_id;
    config.tp_ctx = &tp;
    config.lm_head_column_parallel = true;
    config.mtp.terminal_head_policy = MTPTerminalHeadPolicy::VocabularySharded;
    EXPECT_FALSE(config.mtpPublishesNativeGatheredVocabulary());
    config.mtp.graph_capacity_draft_tokens = 4;
    EXPECT_TRUE(config.mtpParticipantLogitsAreVocabularySharded());
    EXPECT_FALSE(config.mtpUsesMirroredTerminalHeadBinding());
    EXPECT_TRUE(config.mtpPublishesNativeGatheredVocabulary());
    config.mtp.terminal_head_policy = MTPTerminalHeadPolicy::MirroredFullVocabulary;
    EXPECT_TRUE(config.mtpUsesMirroredTerminalHeadBinding());
    EXPECT_FALSE(config.mtpPublishesNativeGatheredVocabulary());

    auto invalid = declaration;
    invalid.vocabulary = 33;
    EXPECT_THROW(NativeVocabularyAllGatherStage{invalid}, std::invalid_argument);
    invalid = declaration;
    invalid.rows = declaration.rows.slice(1, 2);
    EXPECT_THROW(NativeVocabularyAllGatherStage{invalid}, std::invalid_argument);
    invalid = declaration;
    invalid.full_logits = &input;
    EXPECT_THROW(NativeVocabularyAllGatherStage{invalid}, std::invalid_argument);
    invalid = declaration;
    invalid.participant = 1;
    EXPECT_THROW(NativeVocabularyAllGatherStage{invalid}, std::invalid_argument);
    invalid = declaration;
    invalid.rows = DeviceRowRange::fullyActive(1);
    NativeVocabularyAllGatherStage singleton(invalid);
    EXPECT_TRUE(singleton.getWorkspaceRequirements(1).buffers.empty());
}

/** @brief Serial setup must price larger retained rows before publishing scratch. */
TEST_F(NativeAllGatherContract, NativeVocabularyWorkspaceFamilyPreviewsRows)
{
    NativeVocabularyAllGatherStage::Params declaration;
    declaration.device_id = params.device_id;
    declaration.tp_ctx = &tp;
    declaration.local_logits = &input;
    declaration.full_logits = &output;
    declaration.rows = DeviceRowRange::fullyActive(1);
    declaration.local_vocabulary = 16;
    declaration.vocabulary = 32;
    declaration.participant = 0;
    declaration.stage_name = "lm_head_allgather";
    NativeVocabularyAllGatherStage serial(declaration);
    const auto geometry = serial.getBufferRequirements();
    ASSERT_EQ(geometry.buffers.size(), 2u);
    EXPECT_EQ(geometry.buffers[0].shape, (std::vector<std::size_t>{1, 16}));
    EXPECT_EQ(geometry.buffers[1].shape, (std::vector<std::size_t>{1, 32}));
    EXPECT_TRUE(serial.getWorkspaceRequirements(1).buffers.empty());
    const auto family = serial.getWorkspaceRequirements(16);
    ASSERT_EQ(family.buffers.size(), 1u);
    EXPECT_EQ(family.buffers.front().name, VocabularyGatherWorkspaceContract::rankMajorBank);
    EXPECT_EQ(family.buffers.front().size_bytes, 16u * 32u * sizeof(float));
    EXPECT_EQ(serial.params().rows.capacity(), 1);
    EXPECT_THROW(serial.getWorkspaceRequirements(0), std::invalid_argument);
    EXPECT_THROW(serial.getWorkspaceRequirements(-1), std::invalid_argument);

    declaration.rows = DeviceRowRange::fullyActive(4);
    NativeVocabularyAllGatherStage grouped(declaration);
    EXPECT_EQ(grouped.getWorkspaceRequirements(1).buffers.front().size_bytes,
        4u * 32u * sizeof(float));
}

namespace
{
    /** @brief Expose only production terminal publication wiring, without device execution. */
    class NativeVocabularyVerifierGraph final : public Qwen35Graph
    {
    public:
        using Qwen35Graph::Qwen35Graph;
        using Qwen35Graph::addMTPVerifierOutcomeToGraph;
    };
}

/** @brief A sharded policy needs the actual full-tensor gather producer before local verification. */
TEST_F(NativeAllGatherContract, ShardedVerifierRequiresCompletedVocabularyProducer)
{
    GraphConfig config;
    config.default_device = params.device_id;
    config.tp_ctx = &tp;
    config.n_layers = 1;
    config.d_model = 16;
    config.d_ff = 32;
    config.n_heads = config.n_kv_heads = 2;
    config.head_dim = 8;
    config.vocab_size = 32;
    config.vocab_local = 16;
    config.lm_head_column_parallel = true;
    config.mtp.terminal_head_policy = MTPTerminalHeadPolicy::VocabularySharded;
    config.mtp.enabled = true;
    config.grouped_mtp_verifier = true;
    config.compute_all_position_logits = true;
    config.compute_row_indexed_logits = true;
    config.row_indexed_logits_row_count = 4;
    config.mtp_verifier_outcome_graph_mode = MTPVerifierOutcomeGraphMode::Greedy;
    config.mtp_verifier_outcome_ownership = MTPVerifierOutcomeOwnershipPolicy::ParticipantLocal;
    config.mtp_verifier_outcome_graph_binding = {
        .verifier_input_tokens_device = reinterpret_cast<const int32_t *>(0x1000),
        .active_verifier_row_count_device = reinterpret_cast<const int32_t *>(0x1100),
        .transaction_commit_budget_device = reinterpret_cast<const uint32_t *>(0x1200),
        .next_leading_committed_output_count_device = reinterpret_cast<const int32_t *>(0x1300),
        .stop_tokens_device = reinterpret_cast<const int32_t *>(0x2000),
        .penalty_policy_device = reinterpret_cast<const MTPGreedyPenaltyPolicy *>(0x2100),
        .generated_token_counts_device = reinterpret_cast<int32_t *>(0x2200),
        .generated_token_count_capacity = config.vocab_size,
        .verifier_tokens_device = reinterpret_cast<int32_t *>(0x3000),
        .argmax_values_device = reinterpret_cast<float *>(0x4000),
        .argmax_partial_values_device = reinterpret_cast<float *>(0x5000),
        .argmax_partial_indices_device = reinterpret_cast<int32_t *>(0x6000),
        .argmax_partial_capacity = config.vocab_size,
        .output_tokens_device = reinterpret_cast<int32_t *>(0x7000),
        .output_meta_device = reinterpret_cast<int32_t *>(0x8000),
        .output_token_capacity = 4,
        .output_meta_capacity = sampling_math::kSpeculativeBatchMetaCount,
    };
    NativeVocabularyVerifierGraph builder(config);
    graph.addNode("lm_head", std::make_unique<llaminar2::testing::MockComputeStage>(
        ComputeStageType::GEMM, "lm_head", params.device_id), params.device_id);
    EXPECT_THROW(builder.addMTPVerifierOutcomeToGraph(graph, "lm_head", &output, 4, params.device_id), std::runtime_error);
    NativeVocabularyAllGatherStage::Params declaration;
    declaration.device_id = params.device_id;
    declaration.tp_ctx = &tp;
    declaration.local_logits = &input;
    declaration.full_logits = &output;
    declaration.rows = DeviceRowRange::deviceCounted(4, config.mtp_verifier_outcome_graph_binding.active_verifier_row_count_device);
    declaration.local_vocabulary = 16;
    declaration.vocabulary = 32;
    declaration.participant = 0;
    declaration.stage_name = "lm_head_allgather";
    graph.addNode(declaration.stage_name, std::make_unique<NativeVocabularyAllGatherStage>(declaration), params.device_id);
    graph.addDependency(declaration.stage_name, "lm_head");
    FP32Tensor foreign_owner({4, 32}, DeviceId::cpu());
    EXPECT_THROW(builder.addMTPVerifierOutcomeToGraph(graph, declaration.stage_name, &foreign_owner, 4, params.device_id), std::runtime_error);
    EXPECT_EQ(builder.addMTPVerifierOutcomeToGraph(graph, declaration.stage_name, &output, 4, params.device_id), "mtp_verifier_outcome");
    const auto *outcome = dynamic_cast<const MTPVerifierOutcomeStage *>(graph.getNode("mtp_verifier_outcome")->stage.get());
    ASSERT_NE(outcome, nullptr);
    EXPECT_TRUE(outcome->getParams().participant_full_vocabulary);
    EXPECT_EQ(outcome->getParams().logits, &output);
}
