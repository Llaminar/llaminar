#include <gtest/gtest.h>

#include "Perf__MoELLEPDeterminismCommon.h"

#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "tensors/Tensors.h"

#include "../../../utils/TestTensorFactory.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

namespace
{
    using KernelFactory = llaminar::v2::kernels::KernelFactory;
    using namespace llaminar2;
    using namespace llaminar2::test::moe_llep_perf;

#ifdef HAVE_CUDA
    bool hasCudaDevice()
    {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }

    struct CudaEvents
    {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;

        CudaEvents()
        {
            EXPECT_EQ(cudaEventCreate(&start), cudaSuccess);
            EXPECT_EQ(cudaEventCreate(&stop), cudaSuccess);
        }

        ~CudaEvents()
        {
            if (stop)
                cudaEventDestroy(stop);
            if (start)
                cudaEventDestroy(start);
        }
    };

    class CudaHarness
    {
    public:
        explicit CudaHarness(const Shape &shape)
            : shape_(shape),
              device_(DeviceId::cuda(0))
        {
            EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
            EXPECT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess);

            kernel_ = KernelFactory::getOrCreateMoEKernel(device_);
            EXPECT_NE(kernel_, nullptr);
            kernel_->setGPUStream(stream_);

            auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel_);
            EXPECT_NE(workspace_consumer, nullptr);
            auto reqs = MoEWorkspaceBuffers::cudaMoE(
                shape_.seq_len,
                shape_.d_model,
                shape_.intermediate,
                shape_.num_experts,
                shape_.top_k);
            workspace_ = std::make_unique<DeviceWorkspaceManager>(
                device_,
                reqs.total_bytes_with_alignment() + 8 * 1024 * 1024);
            EXPECT_TRUE(workspace_->allocate(reqs));
            workspace_consumer->bindWorkspace(workspace_.get());
        }

        ~CudaHarness()
        {
            if (kernel_)
            {
                if (auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel_))
                    workspace_consumer->unbindWorkspace();
            }
            workspace_.reset();
            if (stream_)
                cudaStreamDestroy(stream_);
        }

        void prepare(bool all_participants_resident,
                     const std::vector<float> &route_experts,
                     const std::vector<float> &route_weights)
        {
            DeviceMoERuntimeTable::Config runtime_config;
            runtime_config.device_id = device_;
            runtime_config.num_layers = 1;
            runtime_config.num_experts = shape_.num_experts;
            runtime_config.top_k = shape_.top_k;
            runtime_config.mirror_to_device = true;
            runtime_config.prefill_token_capacity = shape_.seq_len;
            runtime_table_ = std::make_unique<MoERuntimeTable>(runtime_config);

            auto runtime = runtime_table_->hostLayerState(0);
            configureRuntimeLayer(runtime, shape_, all_participants_resident);
            ASSERT_EQ(cudaMemcpyAsync(runtime_table_->deviceLayerState(0),
                                      &runtime,
                                      sizeof(runtime),
                                      cudaMemcpyHostToDevice,
                                      stream_),
                      cudaSuccess);

            route_indices_tensor_ = llaminar2::test::TestTensorFactory::createFP32(
                {static_cast<size_t>(shape_.seq_len), static_cast<size_t>(shape_.top_k)});
            route_weights_tensor_ = llaminar2::test::TestTensorFactory::createFP32(
                {static_cast<size_t>(shape_.seq_len), static_cast<size_t>(shape_.top_k)});
            std::copy(route_experts.begin(), route_experts.end(),
                      route_indices_tensor_->mutable_data());
            std::copy(route_weights.begin(), route_weights.end(),
                      route_weights_tensor_->mutable_data());
            ASSERT_TRUE(route_indices_tensor_->ensureOnDevice(device_, stream_));
            ASSERT_TRUE(route_weights_tensor_->ensureOnDevice(device_, stream_));

            ASSERT_TRUE(kernel_->groupPrefillRoutes(
                runtime_table_->deviceLayerState(0),
                route_indices_tensor_.get(),
                route_weights_tensor_.get(),
                shape_.seq_len,
                shape_.seq_len,
                shape_.num_experts,
                shape_.top_k));
            ASSERT_TRUE(kernel_->planPrefillRoutesLeastLoadedCurrentBatch(
                runtime_table_->deviceLayerState(0),
                shape_.seq_len,
                shape_.seq_len,
                shape_.num_experts,
                shape_.top_k,
                assignmentConfig(shape_)));
            ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        }

        DeviceMoELayerRuntime copyRuntime() const
        {
            DeviceMoELayerRuntime runtime{};
            EXPECT_EQ(cudaMemcpy(&runtime,
                                 runtime_table_->deviceLayerState(0),
                                 sizeof(runtime),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
            return runtime;
        }

        std::vector<int32_t> copyRouteParticipants(const DeviceMoELayerRuntime &runtime) const
        {
            std::vector<int32_t> participants(static_cast<size_t>(shape_.seq_len * shape_.top_k));
            EXPECT_EQ(cudaMemcpy(participants.data(),
                                 runtime.route_participant_ids,
                                 participants.size() * sizeof(int32_t),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
            return participants;
        }

        std::vector<DeviceMoERebalancePlanEntry> copyPlan(DeviceMoERebalancePlanEntry *d_plan,
                                                          uint32_t count) const
        {
            std::vector<DeviceMoERebalancePlanEntry> plan(count);
            if (count > 0)
            {
                EXPECT_EQ(cudaMemcpy(plan.data(),
                                     d_plan,
                                     plan.size() * sizeof(DeviceMoERebalancePlanEntry),
                                     cudaMemcpyDeviceToHost),
                          cudaSuccess);
            }
            return plan;
        }

        Shape shape_;
        DeviceId device_;
        cudaStream_t stream_ = nullptr;
        IMoEKernel *kernel_ = nullptr;
        std::unique_ptr<DeviceWorkspaceManager> workspace_;
        std::unique_ptr<MoERuntimeTable> runtime_table_;
        std::unique_ptr<FP32Tensor> route_indices_tensor_;
        std::unique_ptr<FP32Tensor> route_weights_tensor_;
    };
#endif
} // namespace

TEST(Perf__MoELLEPDeterminism, CUDA_CurrentBatchSpanAssignmentDeterministic)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    const Shape shape{};
    const int warmups = envInt("LLAMINAR_MOE_LLEP_ASSIGN_WARMUPS", 10);
    const int iterations = envInt("LLAMINAR_MOE_LLEP_ASSIGN_ITERS", 100);
    CudaHarness harness(shape);
    harness.prepare(/*all_participants_resident=*/true,
                    makeResidentAssignmentRouteExperts(shape),
                    makeRouteWeights(shape));

    auto runtime = harness.copyRuntime();
    ASSERT_GT(runtime.reserved_u64[2], 0u);
    ASSERT_EQ(runtime.reserved_u64[3], 0u);

    for (int i = 0; i < warmups; ++i)
    {
        ASSERT_TRUE(harness.kernel_->assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers(
            harness.runtime_table_->deviceLayerState(0),
            shape.seq_len,
            shape.seq_len,
            shape.num_experts,
            shape.top_k));
    }
    ASSERT_EQ(cudaStreamSynchronize(harness.stream_), cudaSuccess);

    CudaEvents events;
    ASSERT_EQ(cudaEventRecord(events.start, harness.stream_), cudaSuccess);
    for (int i = 0; i < iterations; ++i)
    {
        ASSERT_TRUE(harness.kernel_->assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers(
            harness.runtime_table_->deviceLayerState(0),
            shape.seq_len,
            shape.seq_len,
            shape.num_experts,
            shape.top_k));
    }
    ASSERT_EQ(cudaEventRecord(events.stop, harness.stream_), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(events.stop), cudaSuccess);
    float elapsed_ms = 0.0f;
    ASSERT_EQ(cudaEventElapsedTime(&elapsed_ms, events.start, events.stop), cudaSuccess);

    runtime = harness.copyRuntime();
    const auto first = harness.copyRouteParticipants(runtime);
    const uint64_t route_hash = fnv1a64(first);
    for (int repeat = 0; repeat < 8; ++repeat)
    {
        ASSERT_TRUE(harness.kernel_->assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers(
            harness.runtime_table_->deviceLayerState(0),
            shape.seq_len,
            shape.seq_len,
            shape.num_experts,
            shape.top_k));
        ASSERT_EQ(cudaStreamSynchronize(harness.stream_), cudaSuccess);
        EXPECT_EQ(harness.copyRouteParticipants(runtime), first);
    }

    std::array<int, kDeviceMoEMaxParticipants> used{};
    for (const int32_t participant : first)
    {
        ASSERT_GE(participant, 0);
        ASSERT_LT(participant, shape.participant_count);
        used[static_cast<size_t>(participant)] = 1;
    }
    EXPECT_GT(std::accumulate(used.begin(), used.begin() + shape.participant_count, 0), 1);
    printTiming("cuda",
                "current_batch_span_assignment",
                shape,
                iterations,
                elapsed_ms * 1000.0f / static_cast<float>(iterations),
                route_hash,
                static_cast<uint32_t>(runtime.reserved_u64[2]),
                static_cast<uint32_t>(runtime.reserved_u64[3]));
#endif
}

TEST(Perf__MoELLEPDeterminism, CUDA_TransferCommandMaterializationDeterministic)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    const Shape shape{};
    const int warmups = envInt("LLAMINAR_MOE_LLEP_MATERIALIZE_WARMUPS", 10);
    const int iterations = envInt("LLAMINAR_MOE_LLEP_MATERIALIZE_ITERS", 100);
    CudaHarness harness(shape);
    harness.prepare(/*all_participants_resident=*/false,
                    makeSourceZeroTransferRouteExperts(shape),
                    makeRouteWeights(shape));

    auto runtime = harness.copyRuntime();
    ASSERT_GT(runtime.reserved_u64[2], 0u);
    ASSERT_GT(runtime.reserved_u64[3], 0u);

    constexpr uint32_t plan_capacity = kDeviceMoEMaxExperts * kDeviceMoEMaxParticipants;
    DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_header = nullptr;
    DeviceMoERebalanceStatus *d_status = nullptr;
    ASSERT_EQ(cudaMalloc(&d_plan, plan_capacity * sizeof(DeviceMoERebalancePlanEntry)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_plan_count, sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_header, sizeof(DeviceMoERebalanceCommandBufferHeader)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_status, sizeof(DeviceMoERebalanceStatus)), cudaSuccess);

    const auto cleanup = [&]()
    {
        if (d_status)
            cudaFree(d_status);
        if (d_header)
            cudaFree(d_header);
        if (d_plan_count)
            cudaFree(d_plan_count);
        if (d_plan)
            cudaFree(d_plan);
    };

    for (int i = 0; i < warmups; ++i)
    {
        ASSERT_TRUE(harness.kernel_->materializePrefillLeastLoadedTransferCommands(
            harness.runtime_table_->deviceLayerState(0),
            d_plan,
            d_plan_count,
            plan_capacity,
            d_header,
            d_status,
            rebalanceConfig(shape),
            plan_capacity,
            0));
    }
    ASSERT_EQ(cudaStreamSynchronize(harness.stream_), cudaSuccess);

    CudaEvents events;
    ASSERT_EQ(cudaEventRecord(events.start, harness.stream_), cudaSuccess);
    for (int i = 0; i < iterations; ++i)
    {
        ASSERT_TRUE(harness.kernel_->materializePrefillLeastLoadedTransferCommands(
            harness.runtime_table_->deviceLayerState(0),
            d_plan,
            d_plan_count,
            plan_capacity,
            d_header,
            d_status,
            rebalanceConfig(shape),
            plan_capacity,
            0));
    }
    ASSERT_EQ(cudaEventRecord(events.stop, harness.stream_), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(events.stop), cudaSuccess);
    float elapsed_ms = 0.0f;
    ASSERT_EQ(cudaEventElapsedTime(&elapsed_ms, events.start, events.stop), cudaSuccess);

    uint32_t count = 0;
    DeviceMoERebalanceStatus status{};
    ASSERT_EQ(cudaMemcpy(&count, d_plan_count, sizeof(count), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&status, d_status, sizeof(status), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_GT(count, 0u);
    ASSERT_EQ(status.plan_overflow, 0u);
    ASSERT_EQ(status.payload_bucket_overflow, 0u);
    ASSERT_EQ(status.planned_arrivals, count);
    ASSERT_EQ(status.payload_bucket_requested_slots, count);
    ASSERT_GE(status.payload_bucket_slots, count);
    ASSERT_LT(status.payload_bucket_slots, plan_capacity);

    const auto first = harness.copyPlan(d_plan, count);
    const uint64_t plan_hash = fnv1a64Plan(first.data(), first.size());
    std::array<uint32_t, kDeviceMoEMaxParticipants> destination_slots{};
    for (uint32_t i = 0; i < count; ++i)
    {
        const auto &entry = first[static_cast<size_t>(i)];
        ASSERT_EQ(entry.op, static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival));
        ASSERT_EQ(entry.source_participant, 0u);
        ASSERT_GT(entry.destination_participant, 0u);
        ASSERT_LT(entry.destination_participant, static_cast<uint32_t>(shape.participant_count));
        EXPECT_EQ(entry.payload_slot, i);
        EXPECT_LT(entry.payload_slot, status.payload_bucket_slots);
        EXPECT_EQ(entry.destination_slot, destination_slots[entry.destination_participant]++);
    }

    for (int repeat = 0; repeat < 8; ++repeat)
    {
        ASSERT_TRUE(harness.kernel_->materializePrefillLeastLoadedTransferCommands(
            harness.runtime_table_->deviceLayerState(0),
            d_plan,
            d_plan_count,
            plan_capacity,
            d_header,
            d_status,
            rebalanceConfig(shape),
            plan_capacity,
            0));
        ASSERT_EQ(cudaStreamSynchronize(harness.stream_), cudaSuccess);
        uint32_t repeat_count = 0;
        ASSERT_EQ(cudaMemcpy(&repeat_count, d_plan_count, sizeof(repeat_count), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(repeat_count, count);
        const auto repeat_plan = harness.copyPlan(d_plan, repeat_count);
        EXPECT_EQ(fnv1a64Plan(repeat_plan.data(), repeat_plan.size()), plan_hash);
    }

    printTiming("cuda",
                "transfer_command_materialization",
                shape,
                iterations,
                elapsed_ms * 1000.0f / static_cast<float>(iterations),
                plan_hash,
                static_cast<uint32_t>(runtime.reserved_u64[2]),
                count);
    cleanup();
#endif
}
