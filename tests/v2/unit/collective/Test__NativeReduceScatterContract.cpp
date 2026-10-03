/**
 * @file Test__NativeReduceScatterContract.cpp
 * @brief Device-free rejection proofs for native reduce-scatter buffer geometry.
 *
 * Addresses below are metadata only and are never dereferenced. The tests
 * distinguish receive counts from total input, reject aliasing/overflow before
 * enqueue, and cover every native collective element format.
 * Stage checks additionally prove immutable membership, two-bank lifetimes,
 * overlap independence and row-based precision selection without GPU startup.
 */
#include "collective/NativeReduceScatterContract.h"
#include "collective/AllreducePrecisionPolicy.h"
#include "execution/compute_stages/stages/TPColumnReduceScatterStage.h"
#include "execution/compute_stages/stages/TPLocalReduceOverlap.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "tensors/Tensors.h"
#include "tensors/GpuTensorView.h"
#include "../../mocks/MockLocalTPContext.h"
#include "../../mocks/MockComputeStage.h"
#include <gtest/gtest.h>
#include <array>

using namespace llaminar2;

namespace
{
    /** @brief Create an opaque address for range-only validation, not storage. */
    void *address(std::uintptr_t value) { return reinterpret_cast<void *>(value); }

    TEST(NativeReduceScatterContract, AcceptsExactAdjacentRangesForEveryTypeAndDegree)
    {
        for (auto type : {CollectiveDataType::FLOAT32, CollectiveDataType::FLOAT16,
                         CollectiveDataType::BFLOAT16, CollectiveDataType::INT32,
                         CollectiveDataType::INT8})
        {
            const std::size_t bytes = type == CollectiveDataType::INT8 ? 1 :
                (type == CollectiveDataType::FLOAT32 || type == CollectiveDataType::INT32) ? 4 : 2;
            for (int degree = 2; degree <= 8; ++degree)
                for (int participant = 0; participant < degree; ++participant)
                    for (std::size_t count : {1u, 7u, 257u})
                    {
                        const auto receive = 4096 + degree * count * bytes;
                        EXPECT_TRUE(nativeReduceScatterBuffersValid(address(4096), address(receive),
                            count, type, degree, participant, address(1)));
                        EXPECT_FALSE(nativeReduceScatterBuffersValid(address(4096), address(receive - bytes),
                            count, type, degree, participant, address(1)));
                    }
        }
    }

    TEST(NativeReduceScatterContract, RejectsIncompleteAndUnsupportedDeclarations)
    {
        const auto valid = [](const void *input, void *output, std::size_t count,
                              CollectiveDataType type, int degree, int participant, void *stream) {
            return nativeReduceScatterBuffersValid(input, output, count, type, degree, participant, stream);
        };
        constexpr auto fp32 = CollectiveDataType::FLOAT32;
        EXPECT_FALSE(valid(nullptr, address(8192), 16, fp32, 2, 0, address(1)));
        EXPECT_FALSE(valid(address(4096), nullptr, 16, fp32, 2, 0, address(1)));
        EXPECT_FALSE(valid(address(4096), address(8192), 16, fp32, 2, 0, nullptr));
        EXPECT_FALSE(valid(address(4096), address(8192), 0, fp32, 2, 0, address(1)));
        EXPECT_FALSE(valid(address(4096), address(8192), 16, fp32, 1, 0, address(1)));
        EXPECT_FALSE(valid(address(4096), address(8192), 16, fp32, 2, -1, address(1)));
        EXPECT_FALSE(valid(address(4096), address(8192), 16, fp32, 2, 2, address(1)));
        EXPECT_FALSE(valid(address(4096), address(8192), 16,
            static_cast<CollectiveDataType>(999), 2, 0, address(1)));
    }

    TEST(NativeReduceScatterContract, RejectsMisalignmentAliasingAndWrappingRanges)
    {
        constexpr auto maximum = std::numeric_limits<std::uintptr_t>::max();
        constexpr auto fp32 = CollectiveDataType::FLOAT32;
        EXPECT_FALSE(nativeReduceScatterBuffersValid(address(4097), address(8192), 16, fp32, 2, 0, address(1)));
        EXPECT_FALSE(nativeReduceScatterBuffersValid(address(4096), address(8193), 16, fp32, 2, 0, address(1)));
        EXPECT_FALSE(nativeReduceScatterBuffersValid(address(4096), address(4096), 16, fp32, 2, 0, address(1)));
        EXPECT_FALSE(nativeReduceScatterBuffersValid(address(4096), address(4096 - 4), 16, fp32, 2, 0, address(1)));
        EXPECT_FALSE(nativeReduceScatterBuffersValid(address(4096), address(8192), maximum, fp32, 8, 0, address(1)));
        EXPECT_FALSE(nativeReduceScatterBuffersValid(address(maximum - 3), address(8192), 1, fp32, 2, 0, address(1)));
        EXPECT_FALSE(nativeReduceScatterBuffersValid(address(4096), address(maximum - 3), 2, fp32, 2, 0, address(1)));
    }

    /** @brief Host-only graph declaration with reversed physical participant order. */
    class ColumnReduceScatterContract : public ::testing::Test
    {
    protected:
        test::MockLocalTPContext tp;
        FP32Tensor partial{{7, 16}, DeviceId::cpu()};
        FP32Tensor packing{{7, 16}, DeviceId::cpu()};
        TPColumnReduceScatterStage::Params params;

        /** @brief Declare exact native geometry without initializing an accelerator. */
        void SetUp() override
        {
            tp.setDevices({GlobalDeviceAddress::rocm(3), GlobalDeviceAddress::rocm(1)});
            tp.setBackend(CollectiveBackendType::RCCL);
            params.device_id = DeviceId::rocm(3);
            params.tp_ctx = &tp;
            params.tensor = &partial;
            params.packing = &packing;
            params.rows = 7;
            params.model_columns = 16;
            params.participant = 0;
            params.precision = "fp16";
            params.stage_name = "column_sum";
            params.tensor_buffer_id = BufferId::MOE_SHARED_EXPERT_OUTPUT;
            params.packing_buffer_id = BufferId::MOE_PROJECTION_GATHERED_COLUMNS;
        }
    };

    TEST_F(ColumnReduceScatterContract, TwoExistingBanksAndExactNativeMembership)
    {
        TPColumnReduceScatterStage stage(params);
        EXPECT_TRUE(isCollectiveComputeStageType(stage.type()));
        EXPECT_FALSE(stage.supportsBackend(ComputeBackendType::CPU));
        const auto contract = stage.bufferContract();
        ASSERT_EQ(contract.inouts.size(), 1u);
        ASSERT_EQ(contract.outputs.size(), 1u);
        EXPECT_EQ(contract.inouts[0].id, *params.tensor_buffer_id);
        EXPECT_EQ(contract.outputs[0].id, *params.packing_buffer_id);
        EXPECT_TRUE(contract.workspaces.empty());
        EXPECT_THROW((void)stage.requireGPUStream(), std::logic_error);
        tp.setDevices({GlobalDeviceAddress::rocm(1), GlobalDeviceAddress::rocm(3)});
        EXPECT_THROW((void)stage.isGraphCapturable(), std::logic_error);
    }

    TEST_F(ColumnReduceScatterContract, RejectsInvalidGeometryMembershipAndBankRoles)
    {
        for (int rows : {0, -1, 8})
        {
            auto invalid = params;
            invalid.rows = rows;
            EXPECT_THROW(TPColumnReduceScatterStage{invalid}, std::invalid_argument);
        }
        for (int columns : {0, 15, 17, 32})
        {
            auto invalid = params;
            invalid.model_columns = columns;
            EXPECT_THROW(TPColumnReduceScatterStage{invalid}, std::invalid_argument);
        }
        auto invalid = params;
        invalid.participant = 1;
        EXPECT_THROW(TPColumnReduceScatterStage{invalid}, std::invalid_argument);
        invalid = params;
        invalid.packing = &partial;
        EXPECT_THROW(TPColumnReduceScatterStage{invalid}, std::invalid_argument);
        invalid = params;
        invalid.packing_buffer_id = params.tensor_buffer_id;
        EXPECT_THROW(TPColumnReduceScatterStage{invalid}, std::invalid_argument);
        invalid = params;
        invalid.tensor_buffer_id.reset();
        EXPECT_THROW(TPColumnReduceScatterStage{invalid}, std::invalid_argument);
        invalid = params;
        invalid.precision = "implicit-lossy";
        EXPECT_THROW(TPColumnReduceScatterStage{invalid}, std::invalid_argument);
        tp.setBackend(CollectiveBackendType::NCCL);
        EXPECT_THROW(TPColumnReduceScatterStage{params}, std::invalid_argument);
    }

    TEST_F(ColumnReduceScatterContract, RejectsPhysicalOverlapDespiteSeparateTensorNames)
    {
        GpuTensorView a(address(4096), 7, 16, TensorType::FP32, params.device_id);
        GpuTensorView b(address(4096 + 444), 7, 16, TensorType::FP32, params.device_id);
        params.tensor = &a;
        params.packing = &b;
        EXPECT_THROW(TPColumnReduceScatterStage{params}, std::invalid_argument);
        b.update_view(address(4096 + 448), 7);
        EXPECT_NO_THROW(TPColumnReduceScatterStage{params});
    }

    TEST_F(ColumnReduceScatterContract, OverlapHasOnePairedJoinAndPreservesBothBanks)
    {
        ComputeGraph graph;
        for (const auto *name : {"producer", "independent"})
            graph.addNode(name, std::make_unique<llaminar2::testing::MockComputeStage>(
                ComputeStageType::GEMM, name, params.device_id), params.device_id);
        graph.addNode(params.stage_name, std::make_unique<TPColumnReduceScatterStage>(params), params.device_id);
        graph.addDependency(params.stage_name, "producer");
        overlapTPLocalColumnReduceScatter(graph, params.stage_name, "independent");
        EXPECT_EQ(graph.getExecutionOrder(), (std::vector<std::string>{
            "producer", "column_sum_submit", "independent", "column_sum"}));
        const auto contract = graph.getNode(params.stage_name)->stage->bufferContract();
        ASSERT_EQ(contract.inouts.size(), 1u);
        ASSERT_EQ(contract.outputs.size(), 1u);
        EXPECT_EQ(contract.outputs[0].id, *params.packing_buffer_id);
        EXPECT_THROW(overlapTPLocalColumnReduceScatter(graph, params.stage_name, "independent"), std::invalid_argument);
    }

    TEST(ColumnReduceScatterPrecision, OriginalRowWidthOwnsTheDecisionForEveryReceiverCount)
    {
        for (std::size_t rows : {1u, 7u, 64u, 512u})
            for (std::size_t width : {2048u, 8192u, 16384u})
            {
                EXPECT_EQ(fp32SumUsesFP16Transport("fp16", rows * width, width, 8192), width >= 8192);
                EXPECT_FALSE(fp32SumUsesFP16Transport("fp32", rows * width, width, 8192));
            }
    }
}
