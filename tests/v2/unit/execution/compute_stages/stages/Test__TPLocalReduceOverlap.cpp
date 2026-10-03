/**
 * @file Test__TPLocalReduceOverlap.cpp
 * @brief Device-free proof of the native reduction overlap's paired graph contract.
 *
 * Invalid windows must be rejected before any graph mutation or GPU resource
 * acquisition. Positive cases inspect the complete fork/compute/join ordering
 * and the tensor lifetime visible to arena planning.
 */
#include <gtest/gtest.h>
#include "execution/compute_stages/stages/TPLocalReduceOverlap.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "tensors/Tensors.h"
#include "../../../../mocks/MockComputeStage.h"
#include "../../../../mocks/MockLocalTPContext.h"

using namespace llaminar2;

namespace
{
    /** @brief Models a real stage whose arena/kernel bindings follow graph lowering. */
    class UnpreparedCapturableStage final : public llaminar2::testing::MockComputeStage
    {
    public:
        /** @brief Declare immutable device topology without preparing a GPU. */
        explicit UnpreparedCapturableStage(DeviceId device)
            : MockComputeStage(ComputeStageType::GEMM, "unprepared", device) {}
        /** @return Preparation, deliberately not construction, establishes readiness. */
        bool isGraphCapturable() const override { return false; }
        /** @return Typed setup eligibility remains valid before arena binding. */
        bool supportsLazyPrefillGraphCapturePreflight() const override { return true; }
    };
    /** @brief Complete native contract with deliberately host-only test storage. */
    class ReduceOverlapTest : public ::testing::Test
    {
    protected:
        DeviceId device = DeviceId::rocm(0);
        test::MockLocalTPContext tp;
        FP32Tensor tensor{{8, 16}, DeviceId::cpu()};
        ComputeGraph graph;
        TPLocalRootedCollectiveStage::Params params;

        /** @brief Establish two independent nodes without initializing a GPU. */
        void SetUp() override
        {
            tp.setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});
            tp.setBackend(CollectiveBackendType::RCCL);
            params.device_id = device;
            params.tp_ctx = &tp;
            params.tensor = &tensor;
            params.count = 128;
            params.participant_device_index = 0;
            params.stage_name = "reduce";
            params.tensor_buffer_id = BufferId::MOE_SHARED_EXPERT_OUTPUT;
            graph.addNode("producer", std::make_unique<llaminar2::testing::MockComputeStage>(
                ComputeStageType::GEMM, "producer", device), device);
            graph.addNode("compute", std::make_unique<llaminar2::testing::MockComputeStage>(
                ComputeStageType::GEMM, "compute", device), device);
        }

        /** @brief Install the only public paired builder. */
        void install() { addTPLocalReduceOverlap(graph, params, {"producer", "compute"}); }
    };

    TEST_F(ReduceOverlapTest, OrdersPairedEdgesAndKeepsPartialLiveUntilJoin)
    {
        install();
        EXPECT_EQ(graph.getExecutionOrder(), (std::vector<std::string>{
            "producer", "reduce_submit", "compute", "reduce"}));
        const auto *submit = graph.getNode("reduce_submit")->stage.get();
        const auto *join = graph.getNode("reduce")->stage.get();
        ASSERT_EQ(submit->bufferContract().inputs.size(), 1u);
        ASSERT_EQ(join->bufferContract().inouts.size(), 1u);
        EXPECT_EQ(submit->bufferContract().inputs[0].id, *params.tensor_buffer_id);
        EXPECT_EQ(join->bufferContract().inouts[0].id, *params.tensor_buffer_id);
        EXPECT_TRUE(submit->bufferContract().outputs.empty());
        EXPECT_EQ(submit->graphLaunchPreparationPolicy(), GraphLaunchPreparationPolicy::CaptureOnly);
        EXPECT_EQ(join->graphLaunchPreparationPolicy(), GraphLaunchPreparationPolicy::CaptureOnly);
        EXPECT_THROW(install(), std::invalid_argument);
    }

    TEST_F(ReduceOverlapTest, ContributorJoinRetainsReadOnlyRole)
    {
        params.participant_device_index = 1;
        install();
        const auto contract = graph.getNode("reduce")->stage->bufferContract();
        EXPECT_EQ(contract.inputs.size(), 1u);
        EXPECT_TRUE(contract.inouts.empty());
        EXPECT_TRUE(contract.outputs.empty());
    }

    TEST_F(ReduceOverlapTest, ConstructionUsesSetupEligibilityNotUnboundKernelReadiness)
    {
        graph.getNode("producer")->stage = std::make_unique<UnpreparedCapturableStage>(device);
        graph.getNode("compute")->stage = std::make_unique<UnpreparedCapturableStage>(device);
        EXPECT_NO_THROW(install());
        EXPECT_EQ(graph.getExecutionOrder().size(), 4u);
    }

    TEST_F(ReduceOverlapTest, RejectsSidebandsAndUnsupportedOperationBeforeMutation)
    {
        params.sideband_workspace_bindings.emplace_back();
        EXPECT_THROW(install(), std::invalid_argument);
        params.sideband_workspace_bindings.clear();
        params.operation = TPLocalRootedCollectiveOperation::Broadcast;
        EXPECT_THROW(install(), std::invalid_argument);
        EXPECT_EQ(graph.getExecutionOrder().size(), 2u);
    }

    TEST_F(ReduceOverlapTest, RejectsNonNativeCollectiveBackend)
    {
        tp.setBackend(CollectiveBackendType::AUTO);
        EXPECT_THROW(install(), std::invalid_argument);
        tp.setBackend(CollectiveBackendType::NCCL);
        EXPECT_THROW(install(), std::invalid_argument);
        EXPECT_EQ(graph.getExecutionOrder().size(), 2u);
    }

    TEST_F(ReduceOverlapTest, RejectsAliasedInputOutputAndInout)
    {
        auto *compute = static_cast<llaminar2::testing::MockComputeStage *>(graph.getNode("compute")->stage.get());
        for (const auto &contract : {
                StageBufferContract::build().addInput(*params.tensor_buffer_id),
                StageBufferContract::build().addOutput(*params.tensor_buffer_id),
                StageBufferContract::build().addInOut(*params.tensor_buffer_id)})
        {
            compute->setBufferContract(contract);
            EXPECT_THROW(install(), std::invalid_argument);
            EXPECT_EQ(graph.getExecutionOrder().size(), 2u);
        }
    }

    TEST_F(ReduceOverlapTest, RejectsCyclicWindow)
    {
        graph.addDependency("producer", "compute");
        EXPECT_THROW(install(), std::invalid_argument);
        EXPECT_EQ(graph.getExecutionOrder().size(), 2u);
    }

    TEST_F(ReduceOverlapTest, RejectsCaptureBoundaryAndForeignDevice)
    {
        auto *compute = graph.getNode("compute");
        compute->device = DeviceId::rocm(1);
        EXPECT_THROW(install(), std::invalid_argument);
        compute->device = device;
        compute->graph_capture_wave = GraphCaptureWaveContract{.identity = "other_wave"};
        EXPECT_THROW(install(), std::invalid_argument);
        compute->graph_capture_wave.reset();
        graph.setNativeCaptureEnvelope(GraphNativeCaptureEnvelope::HeterogeneousTicketFollowerTransaction);
        EXPECT_THROW(install(), std::invalid_argument);
    }

    /** @brief Allreduce lowering retains the existing stage's complete declaration. */
    class AllreduceOverlapTest : public ReduceOverlapTest
    {
    protected:
        TPAllreduceStage::Params sum;

        /** @brief Add one ordinary allreduce; no GPU preparation occurs in Unit. */
        void SetUp() override
        {
            ReduceOverlapTest::SetUp();
            sum.device_id = device;
            sum.tp_ctx = &tp;
            sum.tensor = &tensor;
            sum.count = 128;
            sum.stage_name = "reduce";
            sum.precision = "fp16";
            sum.tensor_buffer_id = params.tensor_buffer_id;
            graph.addNode("reduce", std::make_unique<TPAllreduceStage>(sum), device);
            graph.addDependency("reduce", "producer");
        }

        /** @brief Update only an unlowered declaration, then install the paired edges. */
        void install()
        {
            if (auto *stage = dynamic_cast<TPAllreduceStage *>(graph.getNode("reduce")->stage.get()))
                stage->setParams(sum);
            overlapTPLocalAllreduce(graph, "reduce", "compute");
        }
    };

    TEST_F(AllreduceOverlapTest, PairsSubmissionAndInoutPublicationWithoutChangingArenaIdentity)
    {
        install();
        EXPECT_EQ(graph.getExecutionOrder(), (std::vector<std::string>{
            "producer", "reduce_submit", "compute", "reduce"}));
        const auto submit = graph.getNode("reduce_submit")->stage->bufferContract();
        const auto join = graph.getNode("reduce")->stage->bufferContract();
        ASSERT_EQ(submit.inputs.size(), 1u);
        ASSERT_EQ(join.inouts.size(), 1u);
        EXPECT_EQ(submit.inputs[0].id, *sum.tensor_buffer_id);
        EXPECT_EQ(join.inouts[0].id, *sum.tensor_buffer_id);
        EXPECT_TRUE(submit.inouts.empty());
        EXPECT_TRUE(submit.outputs.empty());
        EXPECT_THROW(install(), std::invalid_argument);
    }

    TEST_F(AllreduceOverlapTest, RejectsControlSidebandsRankFoldAndOutOfBoundsBeforeMutation)
    {
        sum.sidebands.emplace_back();
        EXPECT_THROW(install(), std::invalid_argument);
        sum.sidebands.clear();
        sum.sideband_workspace_bindings.emplace_back();
        EXPECT_THROW(install(), std::invalid_argument);
        sum.sideband_workspace_bindings.clear();
        sum.arithmetic_policy = TPAllreduceArithmeticPolicy::CanonicalRankOrder;
        EXPECT_THROW(install(), std::invalid_argument);
        sum.arithmetic_policy = TPAllreduceArithmeticPolicy::NativeCollective;
        sum.count = tensor.numel() + 1;
        EXPECT_THROW(install(), std::invalid_argument);
        EXPECT_EQ(graph.getExecutionOrder().size(), 3u);
    }

    TEST_F(AllreduceOverlapTest, RejectsAliasedPartialAndBothCycleDirections)
    {
        auto *compute = static_cast<llaminar2::testing::MockComputeStage *>(graph.getNode("compute")->stage.get());
        for (const auto &contract : {
                StageBufferContract::build().addInput(*sum.tensor_buffer_id),
                StageBufferContract::build().addOutput(*sum.tensor_buffer_id),
                StageBufferContract::build().addInOut(*sum.tensor_buffer_id)})
        {
            compute->setBufferContract(contract);
            EXPECT_THROW(install(), std::invalid_argument);
        }
        compute->setBufferContract({});
        graph.addDependency("compute", "reduce");
        EXPECT_THROW(install(), std::invalid_argument);
        graph.getNode("compute")->dependencies.clear();
        graph.addDependency("producer", "compute");
        EXPECT_THROW(install(), std::invalid_argument);
        EXPECT_EQ(graph.getExecutionOrder().size(), 3u);
    }

    TEST_F(AllreduceOverlapTest, RejectsForeignDeviceTransportAndCaptureBoundary)
    {
        tp.setBackend(CollectiveBackendType::NCCL);
        EXPECT_THROW(install(), std::invalid_argument);
        tp.setBackend(CollectiveBackendType::RCCL);
        auto *compute = graph.getNode("compute");
        compute->device = DeviceId::rocm(1);
        EXPECT_THROW(install(), std::invalid_argument);
        compute->device = device;
        compute->graph_capture_wave = GraphCaptureWaveContract{.identity = "other_wave"};
        EXPECT_THROW(install(), std::invalid_argument);
        compute->graph_capture_wave.reset();
        graph.setNativeCaptureEnvelope(GraphNativeCaptureEnvelope::HeterogeneousTicketFollowerTransaction);
        EXPECT_THROW(install(), std::invalid_argument);
        EXPECT_EQ(graph.getExecutionOrder().size(), 3u);
    }
}
