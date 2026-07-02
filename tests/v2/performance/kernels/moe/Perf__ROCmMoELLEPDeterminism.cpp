#include <gtest/gtest.h>

#include "Perf__MoELLEPDeterminismCommon.h"

#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "tensors/Tensors.h"

#include "../../../utils/TestTensorFactory.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
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

#ifdef HAVE_ROCM
    bool hasROCmDevice()
    {
        int count = 0;
        return hipGetDeviceCount(&count) == hipSuccess && count > 0;
    }

    struct HipEvents
    {
        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;

        HipEvents()
        {
            EXPECT_EQ(hipEventCreate(&start), hipSuccess);
            EXPECT_EQ(hipEventCreate(&stop), hipSuccess);
        }

        ~HipEvents()
        {
            if (stop)
                (void)hipEventDestroy(stop);
            if (start)
                (void)hipEventDestroy(start);
        }
    };

    class ROCmHarness
    {
    public:
        explicit ROCmHarness(const Shape &shape)
            : shape_(shape),
              device_(DeviceId::rocm(0))
        {
            EXPECT_EQ(hipSetDevice(0), hipSuccess);
            EXPECT_EQ(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), hipSuccess);

            kernel_ = KernelFactory::getOrCreateMoEKernel(device_);
            EXPECT_NE(kernel_, nullptr);
            kernel_->setGPUStream(stream_);

            auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel_);
            EXPECT_NE(workspace_consumer, nullptr);
            auto reqs = MoEWorkspaceBuffers::rocmMoE(
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

        ~ROCmHarness()
        {
            if (kernel_)
            {
                if (auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel_))
                    workspace_consumer->unbindWorkspace();
            }
            workspace_.reset();
            if (stream_)
                (void)hipStreamDestroy(stream_);
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
            ASSERT_EQ(hipMemcpyAsync(runtime_table_->deviceLayerState(0),
                                     &runtime,
                                     sizeof(runtime),
                                     hipMemcpyHostToDevice,
                                     stream_),
                      hipSuccess);

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
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        }

        DeviceMoELayerRuntime copyRuntime() const
        {
            DeviceMoELayerRuntime runtime{};
            EXPECT_EQ(hipMemcpy(&runtime,
                                runtime_table_->deviceLayerState(0),
                                sizeof(runtime),
                                hipMemcpyDeviceToHost),
                      hipSuccess);
            return runtime;
        }

        std::vector<int32_t> copyRouteParticipants(const DeviceMoELayerRuntime &runtime) const
        {
            std::vector<int32_t> participants(static_cast<size_t>(shape_.seq_len * shape_.top_k));
            EXPECT_EQ(hipMemcpy(participants.data(),
                                runtime.route_participant_ids,
                                participants.size() * sizeof(int32_t),
                                hipMemcpyDeviceToHost),
                      hipSuccess);
            return participants;
        }

        std::vector<DeviceMoERebalancePlanEntry> copyPlan(DeviceMoERebalancePlanEntry *d_plan,
                                                          uint32_t count) const
        {
            std::vector<DeviceMoERebalancePlanEntry> plan(count);
            if (count > 0)
            {
                EXPECT_EQ(hipMemcpy(plan.data(),
                                    d_plan,
                                    plan.size() * sizeof(DeviceMoERebalancePlanEntry),
                                    hipMemcpyDeviceToHost),
                          hipSuccess);
            }
            return plan;
        }

        Shape shape_;
        DeviceId device_;
        hipStream_t stream_ = nullptr;
        IMoEKernel *kernel_ = nullptr;
        std::unique_ptr<DeviceWorkspaceManager> workspace_;
        std::unique_ptr<MoERuntimeTable> runtime_table_;
        std::unique_ptr<FP32Tensor> route_indices_tensor_;
        std::unique_ptr<FP32Tensor> route_weights_tensor_;
    };
#endif
} // namespace

TEST(Perf__MoELLEPDeterminism, ROCm_CurrentBatchSpanAssignmentDeterministic)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const Shape shape{};
    const int warmups = envInt("LLAMINAR_MOE_LLEP_ASSIGN_WARMUPS", 10);
    const int iterations = envInt("LLAMINAR_MOE_LLEP_ASSIGN_ITERS", 100);
    ROCmHarness harness(shape);
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
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);

    HipEvents events;
    ASSERT_EQ(hipEventRecord(events.start, harness.stream_), hipSuccess);
    for (int i = 0; i < iterations; ++i)
    {
        ASSERT_TRUE(harness.kernel_->assignPrefillRoutesFromLeastLoadedCurrentBatchPlanNoTransfers(
            harness.runtime_table_->deviceLayerState(0),
            shape.seq_len,
            shape.seq_len,
            shape.num_experts,
            shape.top_k));
    }
    ASSERT_EQ(hipEventRecord(events.stop, harness.stream_), hipSuccess);
    ASSERT_EQ(hipEventSynchronize(events.stop), hipSuccess);
    float elapsed_ms = 0.0f;
    ASSERT_EQ(hipEventElapsedTime(&elapsed_ms, events.start, events.stop), hipSuccess);

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
        ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);
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
    printTiming("rocm",
                "current_batch_span_assignment",
                shape,
                iterations,
                elapsed_ms * 1000.0f / static_cast<float>(iterations),
                route_hash,
                static_cast<uint32_t>(runtime.reserved_u64[2]),
                static_cast<uint32_t>(runtime.reserved_u64[3]));
#endif
}

TEST(Perf__MoELLEPDeterminism, ROCm_TransferCommandMaterializationDeterministic)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const Shape shape{};
    const int warmups = envInt("LLAMINAR_MOE_LLEP_MATERIALIZE_WARMUPS", 10);
    const int iterations = envInt("LLAMINAR_MOE_LLEP_MATERIALIZE_ITERS", 100);
    ROCmHarness harness(shape);
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
    ASSERT_EQ(hipMalloc(&d_plan, plan_capacity * sizeof(DeviceMoERebalancePlanEntry)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_plan_count, sizeof(uint32_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_header, sizeof(DeviceMoERebalanceCommandBufferHeader)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_status, sizeof(DeviceMoERebalanceStatus)), hipSuccess);

    const auto cleanup = [&]()
    {
        if (d_status)
            (void)hipFree(d_status);
        if (d_header)
            (void)hipFree(d_header);
        if (d_plan_count)
            (void)hipFree(d_plan_count);
        if (d_plan)
            (void)hipFree(d_plan);
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
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);

    HipEvents events;
    ASSERT_EQ(hipEventRecord(events.start, harness.stream_), hipSuccess);
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
    ASSERT_EQ(hipEventRecord(events.stop, harness.stream_), hipSuccess);
    ASSERT_EQ(hipEventSynchronize(events.stop), hipSuccess);
    float elapsed_ms = 0.0f;
    ASSERT_EQ(hipEventElapsedTime(&elapsed_ms, events.start, events.stop), hipSuccess);

    uint32_t count = 0;
    DeviceMoERebalanceStatus status{};
    ASSERT_EQ(hipMemcpy(&count, d_plan_count, sizeof(count), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(&status, d_status, sizeof(status), hipMemcpyDeviceToHost), hipSuccess);
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
        ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);
        uint32_t repeat_count = 0;
        ASSERT_EQ(hipMemcpy(&repeat_count, d_plan_count, sizeof(repeat_count), hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_EQ(repeat_count, count);
        const auto repeat_plan = harness.copyPlan(d_plan, repeat_count);
        EXPECT_EQ(fnv1a64Plan(repeat_plan.data(), repeat_plan.size()), plan_hash);
    }

    printTiming("rocm",
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
