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

            owned_kernel_ = KernelFactory::createMoEKernel(device_);
            kernel_ = owned_kernel_.get();
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
            if (resident_expert_slab_)
                cudaFree(resident_expert_slab_);
            if (stream_)
                cudaStreamDestroy(stream_);
        }

        /**
         * @brief Describe the immutable launch resources owned by this harness.
         *
         * Rebalance and current-batch LLEP methods consume this context directly
         * instead of reading the mutable stream/workspace fields inherited by
         * the process-wide kernel singleton. Keeping construction here makes
         * every benchmark launch explicit without duplicating resource wiring.
         */
        [[nodiscard]] MoEKernelLaunchContext launchContext() const noexcept
        {
            return {
                .stream = stream_,
                .workspace = workspace_.get(),
            };
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
            if (all_participants_resident)
            {
                const SyntheticPayloadSpec payload_spec{};
                const uint64_t bytes_per_expert =
                    syntheticExpertDataBytes(payload_spec);
                ASSERT_GT(bytes_per_expert, 0u);
                ASSERT_EQ(
                    cudaMalloc(
                        &resident_expert_slab_,
                        static_cast<size_t>(bytes_per_expert) *
                            static_cast<size_t>(shape_.num_experts)),
                    cudaSuccess);
                installSyntheticAllLocalExpertDescriptors(
                    runtime,
                    shape_,
                    payload_spec,
                    /*participant_id=*/0,
                    resident_expert_slab_,
                    bytes_per_expert);
            }
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
                launchContext(),
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

        void uploadRuntime(const DeviceMoELayerRuntime &runtime) const
        {
            ASSERT_EQ(cudaMemcpyAsync(runtime_table_->deviceLayerState(0),
                                      &runtime,
                                      sizeof(runtime),
                                      cudaMemcpyHostToDevice,
                                      stream_),
                      cudaSuccess);
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
        std::unique_ptr<IMoEKernel> owned_kernel_;
        IMoEKernel *kernel_ = nullptr;
        std::unique_ptr<DeviceWorkspaceManager> workspace_;
        uint8_t *resident_expert_slab_ = nullptr;
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
            harness.launchContext(),
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
            harness.launchContext(),
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
            harness.launchContext(),
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

TEST(Perf__MoELLEPDeterminism, CUDA_GroupPrefillRoutesDeterministic)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    const Shape shape{};
    const int warmups = envInt("LLAMINAR_MOE_GROUP_ROUTES_WARMUPS", 10);
    const int iterations = envInt("LLAMINAR_MOE_GROUP_ROUTES_ITERS", 100);
    CudaHarness harness(shape);
    harness.prepare(/*all_participants_resident=*/false,
                    makeSourceZeroTransferRouteExperts(shape),
                    makeRouteWeights(shape));

    for (int i = 0; i < warmups; ++i)
    {
        ASSERT_TRUE(harness.kernel_->groupPrefillRoutes(
            harness.runtime_table_->deviceLayerState(0),
            harness.route_indices_tensor_.get(),
            harness.route_weights_tensor_.get(),
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
        ASSERT_TRUE(harness.kernel_->groupPrefillRoutes(
            harness.runtime_table_->deviceLayerState(0),
            harness.route_indices_tensor_.get(),
            harness.route_weights_tensor_.get(),
            shape.seq_len,
            shape.seq_len,
            shape.num_experts,
            shape.top_k));
    }
    ASSERT_EQ(cudaEventRecord(events.stop, harness.stream_), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(events.stop), cudaSuccess);
    float elapsed_ms = 0.0f;
    ASSERT_EQ(cudaEventElapsedTime(&elapsed_ms, events.start, events.stop), cudaSuccess);

    const auto runtime = harness.copyRuntime();
    std::vector<int32_t> counts(static_cast<size_t>(shape.num_experts));
    ASSERT_EQ(cudaMemcpyAsync(counts.data(),
                              runtime.expert_counts,
                              counts.size() * sizeof(int32_t),
                              cudaMemcpyDeviceToHost,
                              harness.stream_),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(harness.stream_), cudaSuccess);
    const int total_routes =
        std::accumulate(counts.begin(), counts.end(), 0);
    ASSERT_EQ(total_routes, shape.seq_len * shape.top_k);
    printTiming("cuda",
                "group_prefill_routes",
                shape,
                iterations,
                elapsed_ms * 1000.0f / static_cast<float>(iterations),
                fnv1a64(counts),
                static_cast<uint32_t>(total_routes),
                0);
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
            harness.launchContext(),
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
            harness.launchContext(),
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
    for (uint32_t i = 0; i < count; ++i)
    {
        const auto &entry = first[static_cast<size_t>(i)];
        ASSERT_EQ(entry.op, static_cast<uint32_t>(DeviceMoERebalancePlanOp::ExpertPayloadArrival));
        ASSERT_EQ(entry.source_participant, 0u);
        ASSERT_GT(entry.destination_participant, 0u);
        ASSERT_LT(entry.destination_participant, static_cast<uint32_t>(shape.participant_count));
        EXPECT_EQ(entry.payload_slot, i);
        EXPECT_LT(entry.payload_slot, status.payload_bucket_slots);
        EXPECT_EQ(entry.destination_slot, kDeviceMoEInvalidSlot);
    }

    for (int repeat = 0; repeat < 8; ++repeat)
    {
        ASSERT_TRUE(harness.kernel_->materializePrefillLeastLoadedTransferCommands(
            harness.launchContext(),
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

TEST(Perf__MoELLEPDeterminism, CUDA_DynamicMaintenancePackAndControllerDeterministic)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    const Shape shape{};
    const DeviceMoERebalanceConfig config = dynamicMaintenanceConfig(shape);
    const uint32_t wave_layers = std::max(1u, config.layer_wave_count);
    const uint32_t local_histogram_count = wave_layers * config.num_experts;
    const uint32_t gathered_histogram_count =
        config.participant_count * local_histogram_count;
    const int warmups = envInt("LLAMINAR_MOE_DYNAMIC_MAINT_WARMUPS", 10);
    const int iterations = envInt("LLAMINAR_MOE_DYNAMIC_MAINT_ITERS", 100);

    CudaHarness harness(shape);
    harness.prepare(/*all_participants_resident=*/false,
                    makeSourceZeroTransferRouteExperts(shape),
                    makeRouteWeights(shape));

    auto runtime = harness.copyRuntime();
    configureRuntimeLayer(runtime, shape, /*all_participants_resident=*/false, 0);
    installSkewedDynamicHistogram(runtime, shape);
    harness.uploadRuntime(runtime);

    constexpr uint32_t plan_capacity = kDeviceMoEMaxExperts * kDeviceMoEMaxParticipants;
    uint64_t *d_local_histograms = nullptr;
    uint64_t *d_gathered_histograms = nullptr;
    DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    DeviceMoERebalanceStatus *d_status = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_header = nullptr;
    DeviceMoERebalanceWaveState *d_wave_state = nullptr;
    DeviceMoERebalanceGraphControllerState *d_controller_states = nullptr;
    const auto controller_states = makeDueControllerTransactions(
        config, warmups + iterations);
    ASSERT_EQ(cudaMalloc(&d_local_histograms,
                         static_cast<size_t>(local_histogram_count) * sizeof(uint64_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_gathered_histograms,
                         static_cast<size_t>(gathered_histogram_count) * sizeof(uint64_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_plan, plan_capacity * sizeof(DeviceMoERebalancePlanEntry)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_plan_count, sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_status, sizeof(DeviceMoERebalanceStatus)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_header, sizeof(DeviceMoERebalanceCommandBufferHeader)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_wave_state, sizeof(DeviceMoERebalanceWaveState)), cudaSuccess);
    ASSERT_EQ(
        cudaMalloc(
            &d_controller_states,
            controller_states.size() *
                sizeof(DeviceMoERebalanceGraphControllerState)),
        cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(d_gathered_histograms,
                              0,
                              static_cast<size_t>(gathered_histogram_count) * sizeof(uint64_t),
                              harness.stream_),
              cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(
            d_controller_states,
            controller_states.data(),
            controller_states.size() *
                sizeof(DeviceMoERebalanceGraphControllerState),
            cudaMemcpyHostToDevice,
            harness.stream_),
        cudaSuccess);

    auto run_maintenance = [&](int transaction_index)
    {
        ASSERT_TRUE(harness.kernel_->packDeviceRebalanceHistograms(
            harness.launchContext(),
            harness.runtime_table_->deviceLayerState(0),
            d_local_histograms,
            config));
        ASSERT_EQ(cudaMemcpyAsync(d_gathered_histograms,
                                  d_local_histograms,
                                  static_cast<size_t>(local_histogram_count) * sizeof(uint64_t),
                                  cudaMemcpyDeviceToDevice,
                                  harness.stream_),
                  cudaSuccess);
        ASSERT_TRUE(harness.kernel_->runDeviceRebalanceController(
            harness.launchContext(),
            harness.runtime_table_->deviceLayerState(0),
            d_gathered_histograms,
            d_status,
            config,
            d_plan,
            d_plan_count,
            plan_capacity,
            plan_capacity,
            d_header,
            d_wave_state,
            d_controller_states + transaction_index));
    };

    for (int i = 0; i < warmups; ++i)
        run_maintenance(i);
    ASSERT_EQ(cudaStreamSynchronize(harness.stream_), cudaSuccess);

    CudaEvents events;
    ASSERT_EQ(cudaEventRecord(events.start, harness.stream_), cudaSuccess);
    for (int i = 0; i < iterations; ++i)
        run_maintenance(warmups + i);
    ASSERT_EQ(cudaEventRecord(events.stop, harness.stream_), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(events.stop), cudaSuccess);
    float elapsed_ms = 0.0f;
    ASSERT_EQ(cudaEventElapsedTime(&elapsed_ms, events.start, events.stop), cudaSuccess);

    uint32_t plan_count = 0;
    DeviceMoERebalanceStatus status{};
    ASSERT_EQ(cudaMemcpyAsync(&plan_count,
                              d_plan_count,
                              sizeof(plan_count),
                              cudaMemcpyDeviceToHost,
                              harness.stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(&status,
                              d_status,
                              sizeof(status),
                              cudaMemcpyDeviceToHost,
                              harness.stream_),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(harness.stream_), cudaSuccess);
    ASSERT_EQ(status.status_code, static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
    ASSERT_GT(plan_count, 0u);
    ASSERT_EQ(status.plan_overflow, 0u);
    ASSERT_EQ(status.payload_bucket_overflow, 0u);
    ASSERT_GT(status.dynamic_ownership_swap_accepts, 0u);
    ASSERT_GT(status.accepted_load_spread_improvement_total, 0u);
    ASSERT_GT(status.pre_policy_imbalance_numerator,
              status.post_policy_imbalance_numerator);
    ASSERT_GE(status.payload_bucket_slots, status.payload_bucket_requested_slots);

    const auto plan = harness.copyPlan(d_plan, plan_count);
    const uint64_t plan_hash = fnv1a64Plan(plan.data(), plan.size());
    printTiming("cuda",
                "dynamic_maintenance_pack_controller",
                shape,
                iterations,
                elapsed_ms * 1000.0f / static_cast<float>(iterations),
                plan_hash,
                status.dynamic_ownership_swap_accepts,
                plan_count);

    if (d_wave_state)
        cudaFree(d_wave_state);
    if (d_controller_states)
        cudaFree(d_controller_states);
    if (d_header)
        cudaFree(d_header);
    if (d_status)
        cudaFree(d_status);
    if (d_plan_count)
        cudaFree(d_plan_count);
    if (d_plan)
        cudaFree(d_plan);
    if (d_gathered_histograms)
        cudaFree(d_gathered_histograms);
    if (d_local_histograms)
        cudaFree(d_local_histograms);
#endif
}

/**
 * @brief Time the exact Qwen3.6 LLEP decode-maintenance controller geometry.
 *
 * Runtime state, histogram evidence, controller transactions, and output
 * buffers are materialized before timing. Each launch receives an independent
 * request transaction that is due on its next decode edge, so the timed region
 * measures the production LLEP planner rather than a busy-wave or invalid-state
 * guard. One event synchronization follows the complete launch batch.
 */
TEST(Perf__MoELLEPDeterminism, CUDA_Qwen36LLEPMaintenanceControllerEconomy)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    Shape shape{};
    shape.participant_count = 2;
    constexpr uint32_t num_layers = 40u;
    constexpr uint32_t plan_capacity = 42u;
    DeviceMoELLEPLayerPlanScratch *d_llep_layer_plans = nullptr;
    const uint32_t layer_wave_count = static_cast<uint32_t>(
        std::min(
            4,
            envInt("LLAMINAR_MOE_LLEP_MAINT_LAYER_WAVE", 4)));
    const DeviceMoERebalanceConfig config =
        llepMaintenanceConfig(shape, num_layers, layer_wave_count);
    const uint32_t gathered_histogram_count =
        config.participant_count * config.layer_wave_count * config.num_experts;
    const int warmups =
        envInt("LLAMINAR_MOE_LLEP_MAINT_WARMUPS", 2);
    const int iterations =
        envInt("LLAMINAR_MOE_LLEP_MAINT_ITERS", 20);

    CudaHarness harness(shape);
    harness.prepare(/*all_participants_resident=*/false,
                    makeSourceZeroTransferRouteExperts(shape),
                    makeRouteWeights(shape));
    ASSERT_EQ(
        cudaMalloc(
            &d_llep_layer_plans,
            static_cast<size_t>(
                std::max(1u, config.layer_window_count)) *
                sizeof(DeviceMoELLEPLayerPlanScratch)),
        cudaSuccess);

    DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = harness.device_;
    runtime_config.num_layers = static_cast<int>(num_layers);
    runtime_config.num_experts = shape.num_experts;
    runtime_config.top_k = shape.top_k;
    runtime_config.mirror_to_device = true;
    MoERuntimeTable runtime_table(runtime_config);
    for (uint32_t layer = 0; layer < num_layers; ++layer)
    {
        auto runtime = runtime_table.hostLayerState(static_cast<int>(layer));
        configureRuntimeLayer(
            runtime,
            shape,
            /*all_participants_resident=*/false,
            /*participant_id=*/0);
        ASSERT_EQ(
            cudaMemcpyAsync(
                runtime_table.deviceLayerState(static_cast<int>(layer)),
                &runtime,
                sizeof(runtime),
                cudaMemcpyHostToDevice,
                harness.stream_),
            cudaSuccess);
    }

    std::vector<uint64_t> gathered(
        static_cast<size_t>(gathered_histogram_count),
        0ULL);
    const uint32_t participant_stride =
        config.layer_wave_count * config.num_experts;
    for (uint32_t participant = 0;
         participant < config.participant_count;
         ++participant)
    {
        for (uint32_t wave_layer = 0;
             wave_layer < config.layer_wave_count;
             ++wave_layer)
        {
            for (uint32_t expert = 0; expert < config.num_experts; ++expert)
            {
                const uint32_t owner =
                    expert % config.participant_count;
                const uint64_t count =
                    owner == 0u
                        ? (expert == 0u ? 4096ULL : 64ULL)
                        : 1ULL;
                const size_t index =
                    static_cast<size_t>(participant) * participant_stride +
                    static_cast<size_t>(wave_layer) * config.num_experts +
                    expert;
                gathered[index] =
                    moe_rebalance_policy::packCollectedState(
                        participant == owner ? count : 0ULL,
                        /*active_transfer_slots=*/0u,
                        /*physically_resident=*/participant == owner,
                        /*transfer_backed=*/false);
            }
        }
    }

    uint64_t *d_gathered_histograms = nullptr;
    DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    DeviceMoERebalanceStatus *d_status = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_header = nullptr;
    DeviceMoERebalanceWaveState *d_wave_state = nullptr;
    DeviceMoERebalanceGraphControllerState *d_controller_states = nullptr;
    const auto controller_states = makeDueControllerTransactions(
        config, warmups + iterations);
    ASSERT_EQ(
        cudaMalloc(
            &d_gathered_histograms,
            static_cast<size_t>(gathered_histogram_count) * sizeof(uint64_t)),
        cudaSuccess);
    ASSERT_EQ(
        cudaMalloc(
            &d_plan,
            static_cast<size_t>(plan_capacity) *
                sizeof(DeviceMoERebalancePlanEntry)),
        cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_plan_count, sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_status, sizeof(DeviceMoERebalanceStatus)), cudaSuccess);
    ASSERT_EQ(
        cudaMalloc(&d_header, sizeof(DeviceMoERebalanceCommandBufferHeader)),
        cudaSuccess);
    ASSERT_EQ(
        cudaMalloc(&d_wave_state, sizeof(DeviceMoERebalanceWaveState)),
        cudaSuccess);
    ASSERT_EQ(
        cudaMalloc(
            &d_controller_states,
            controller_states.size() *
                sizeof(DeviceMoERebalanceGraphControllerState)),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(
            d_gathered_histograms,
            gathered.data(),
            gathered.size() * sizeof(uint64_t),
            cudaMemcpyHostToDevice,
            harness.stream_),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemsetAsync(
            d_wave_state,
            0,
            sizeof(*d_wave_state),
            harness.stream_),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(
            d_controller_states,
            controller_states.data(),
            controller_states.size() *
                sizeof(DeviceMoERebalanceGraphControllerState),
            cudaMemcpyHostToDevice,
            harness.stream_),
        cudaSuccess);

    auto run_controller = [&](int transaction_index)
    {
        ASSERT_TRUE(harness.kernel_->runDeviceRebalanceController(
            harness.launchContext(),
            runtime_table.deviceLayerState(0),
            d_gathered_histograms,
            d_status,
            config,
            d_plan,
            d_plan_count,
            plan_capacity,
            plan_capacity,
            d_header,
            d_wave_state,
            d_controller_states + transaction_index,
            1u,
            nullptr,
            0u,
            d_llep_layer_plans));
    };

    for (int i = 0; i < warmups; ++i)
        run_controller(i);
    ASSERT_EQ(cudaStreamSynchronize(harness.stream_), cudaSuccess);

    CudaEvents events;
    ASSERT_EQ(cudaEventRecord(events.start, harness.stream_), cudaSuccess);
    for (int i = 0; i < iterations; ++i)
        run_controller(warmups + i);
    ASSERT_EQ(cudaEventRecord(events.stop, harness.stream_), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(events.stop), cudaSuccess);
    float elapsed_ms = 0.0f;
    ASSERT_EQ(
        cudaEventElapsedTime(&elapsed_ms, events.start, events.stop),
        cudaSuccess);

    uint32_t plan_count = 0;
    DeviceMoERebalanceStatus status{};
    ASSERT_EQ(
        cudaMemcpyAsync(
            &plan_count,
            d_plan_count,
            sizeof(plan_count),
            cudaMemcpyDeviceToHost,
            harness.stream_),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(
            &status,
            d_status,
            sizeof(status),
            cudaMemcpyDeviceToHost,
            harness.stream_),
        cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(harness.stream_), cudaSuccess);
    ASSERT_EQ(
        status.status_code,
        static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
    ASSERT_LE(plan_count, plan_capacity);
    ASSERT_EQ(status.payload_bucket_overflow, 0u);

    const auto plan = harness.copyPlan(d_plan, plan_count);
    printTiming(
        "cuda",
        "qwen36_llep_maintenance_controller",
        shape,
        iterations,
        elapsed_ms * 1000.0f / static_cast<float>(iterations),
        fnv1a64Plan(plan.data(), plan.size()),
        status.llep_assignment_span_count,
        plan_count);

    if (d_llep_layer_plans)
        cudaFree(d_llep_layer_plans);
    if (d_controller_states)
        cudaFree(d_controller_states);
    if (d_wave_state)
        cudaFree(d_wave_state);
    if (d_header)
        cudaFree(d_header);
    if (d_status)
        cudaFree(d_status);
    if (d_plan_count)
        cudaFree(d_plan_count);
    if (d_plan)
        cudaFree(d_plan);
    if (d_gathered_histograms)
        cudaFree(d_gathered_histograms);
#endif
}

TEST(Perf__MoELLEPDeterminism, CUDA_PayloadMovementAndApplyDeterministic)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    if (!hasCudaDevice())
        GTEST_SKIP() << "No CUDA device available";

    const Shape shape{};
    const SyntheticPayloadSpec payload_spec{};
    const uint64_t expert_bytes = syntheticExpertDataBytes(payload_spec);
    const uint64_t payload_slot_bytes = syntheticPayloadSlotBytes(payload_spec);
    ASSERT_GT(expert_bytes, 0u);
    ASSERT_GT(payload_slot_bytes, sizeof(DeviceMoEExpertDirectoryEntry));

    const int warmups = envInt("LLAMINAR_MOE_PAYLOAD_WARMUPS", 5);
    const int iterations = envInt("LLAMINAR_MOE_PAYLOAD_ITERS", 50);
    CudaHarness harness(shape);
    harness.prepare(/*all_participants_resident=*/false,
                    makeSourceZeroTransferRouteExperts(shape),
                    makeRouteWeights(shape));

    constexpr uint32_t plan_capacity = kDeviceMoEMaxExperts * kDeviceMoEMaxParticipants;
    DeviceMoERebalancePlanEntry *d_plan = nullptr;
    uint32_t *d_plan_count = nullptr;
    DeviceMoERebalanceCommandBufferHeader *d_header = nullptr;
    DeviceMoERebalanceStatus *d_plan_status = nullptr;
    DeviceMoEExpertDirectoryEntry *d_source_descriptors = nullptr;
    DeviceMoERebalanceApplyStatus *d_pack_status = nullptr;
    DeviceMoERebalanceApplyStatus *d_unpack_status = nullptr;
    DeviceMoERebalanceApplyStatus *d_apply_status = nullptr;
    uint8_t *d_source_slab = nullptr;
    uint8_t *d_destination_slab = nullptr;
    uint8_t *d_local_payload = nullptr;
    uint8_t *d_gathered_payload = nullptr;
    DeviceMoEExpertDirectoryEntry *d_transfer_slots = nullptr;

    const uint32_t source_experts =
        static_cast<uint32_t>((shape.num_experts + shape.participant_count - 1) /
                              shape.participant_count);
    ASSERT_EQ(cudaMalloc(&d_source_slab,
                         static_cast<size_t>(source_experts) *
                             static_cast<size_t>(expert_bytes)),
              cudaSuccess);
    for (uint32_t ordinal = 0; ordinal < source_experts; ++ordinal)
    {
        ASSERT_EQ(cudaMemsetAsync(d_source_slab + static_cast<uint64_t>(ordinal) * expert_bytes,
                                  static_cast<int>((ordinal * 37u + 11u) & 0xffu),
                                  static_cast<size_t>(expert_bytes),
                                  harness.stream_),
                  cudaSuccess);
    }

    auto source_runtime = harness.copyRuntime();
    installSyntheticLocalExpertDescriptors(
        source_runtime, shape, payload_spec, 0, d_source_slab, expert_bytes);
    harness.uploadRuntime(source_runtime);

    ASSERT_EQ(cudaMalloc(&d_plan, plan_capacity * sizeof(DeviceMoERebalancePlanEntry)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_plan_count, sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_header, sizeof(DeviceMoERebalanceCommandBufferHeader)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_plan_status, sizeof(DeviceMoERebalanceStatus)), cudaSuccess);
    ASSERT_TRUE(harness.kernel_->materializePrefillLeastLoadedTransferCommands(
        harness.launchContext(),
        harness.runtime_table_->deviceLayerState(0),
        d_plan,
        d_plan_count,
        plan_capacity,
        d_header,
        d_plan_status,
        rebalanceConfig(shape),
        plan_capacity,
        0));
    ASSERT_EQ(cudaStreamSynchronize(harness.stream_), cudaSuccess);

    uint32_t plan_count = 0;
    DeviceMoERebalanceStatus plan_status{};
    ASSERT_EQ(cudaMemcpyAsync(&plan_count,
                              d_plan_count,
                              sizeof(plan_count),
                              cudaMemcpyDeviceToHost,
                              harness.stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(&plan_status,
                              d_plan_status,
                              sizeof(plan_status),
                              cudaMemcpyDeviceToHost,
                              harness.stream_),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(harness.stream_), cudaSuccess);
    ASSERT_GT(plan_count, 0u);
    ASSERT_EQ(plan_status.plan_overflow, 0u);
    ASSERT_EQ(plan_status.payload_bucket_overflow, 0u);

    const auto plan = harness.copyPlan(d_plan, plan_count);
    uint32_t destination_one_arrivals = 0;
    for (const auto &entry : plan)
    {
        if (entry.destination_participant == 1u)
            ++destination_one_arrivals;
    }
    ASSERT_GT(destination_one_arrivals, 0u);
    const uint32_t payload_slot_count = plan_status.payload_bucket_slots;
    ASSERT_GE(payload_slot_count, plan_count);

    ASSERT_EQ(cudaMalloc(&d_source_descriptors,
                         static_cast<size_t>(shape.participant_count) *
                             plan_capacity * sizeof(DeviceMoEExpertDirectoryEntry)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_pack_status, sizeof(DeviceMoERebalanceApplyStatus)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_unpack_status, sizeof(DeviceMoERebalanceApplyStatus)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_apply_status, sizeof(DeviceMoERebalanceApplyStatus)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_local_payload,
                         static_cast<size_t>(payload_slot_count) *
                             static_cast<size_t>(payload_slot_bytes)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_gathered_payload,
                         static_cast<size_t>(shape.participant_count) *
                             static_cast<size_t>(payload_slot_count) *
                             static_cast<size_t>(payload_slot_bytes)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_destination_slab,
                         static_cast<size_t>(plan_count) *
                             static_cast<size_t>(expert_bytes)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_transfer_slots,
                         static_cast<size_t>(plan_count) *
                             sizeof(DeviceMoEExpertDirectoryEntry)),
              cudaSuccess);

    std::vector<DeviceMoEExpertDirectoryEntry> transfer_slots(plan_count);
    for (uint32_t slot = 0; slot < plan_count; ++slot)
    {
        transfer_slots[slot] = makeSyntheticTransferSlot(
            d_destination_slab + static_cast<uint64_t>(slot) * expert_bytes,
            payload_spec,
            1u,
            slot);
    }
    ASSERT_EQ(cudaMemcpyAsync(d_transfer_slots,
                              transfer_slots.data(),
                              transfer_slots.size() * sizeof(DeviceMoEExpertDirectoryEntry),
                              cudaMemcpyHostToDevice,
                              harness.stream_),
              cudaSuccess);

    ASSERT_TRUE(harness.kernel_->packDeviceRebalanceSourceDescriptors(
        harness.launchContext(),
        harness.runtime_table_->deviceLayerState(0),
        d_plan,
        d_header,
        plan_capacity,
        d_source_descriptors,
        rebalanceConfig(shape),
        nullptr));
    ASSERT_EQ(cudaStreamSynchronize(harness.stream_), cudaSuccess);

    auto destination_runtime = harness.copyRuntime();
    configureRuntimeLayer(destination_runtime, shape, /*all_participants_resident=*/false, 1);
    harness.uploadRuntime(destination_runtime);

    auto run_payload_wave = [&]()
    {
        const DeviceMoERebalanceConfig source_config = rebalanceConfig(shape);
        DeviceMoERebalanceConfig destination_config = rebalanceConfig(shape);
        destination_config.participant_id = 1u;

        auto pack_payloads = [&]()
        {
            ASSERT_TRUE(harness.kernel_->packDeviceRebalanceCompactPayloads(
                harness.launchContext(),
                d_plan,
                d_header,
                plan_capacity,
                d_source_descriptors,
                d_local_payload,
                payload_slot_count,
                payload_slot_bytes,
                source_config,
                d_pack_status));
        };
        auto copy_payload_bucket = [&]()
        {
            ASSERT_EQ(cudaMemcpyAsync(d_gathered_payload,
                                      d_local_payload,
                                      static_cast<size_t>(payload_slot_count) *
                                          static_cast<size_t>(payload_slot_bytes),
                                      cudaMemcpyDeviceToDevice,
                                      harness.stream_),
                      cudaSuccess);
        };
        auto unpack_payloads = [&]()
        {
            ASSERT_TRUE(harness.kernel_->unpackDeviceRebalanceCollectivePayloads(
                harness.launchContext(),
                d_plan,
                d_plan_count,
                plan_capacity,
                nullptr,
                d_gathered_payload,
                payload_slot_count,
                payload_slot_bytes,
                d_transfer_slots,
                plan_count,
                destination_config,
                d_unpack_status));
        };
        auto apply_arrivals = [&]()
        {
            ASSERT_TRUE(harness.kernel_->applyDeviceRebalanceArrivals(
                harness.launchContext(),
                harness.runtime_table_->deviceLayerState(0),
                d_plan,
                d_plan_count,
                plan_capacity,
                d_transfer_slots,
                plan_count,
                destination_config,
                d_apply_status,
                nullptr,
                -1));
        };

        pack_payloads();
        copy_payload_bucket();
        unpack_payloads();
        apply_arrivals();
    };
    auto pack_payloads_once = [&]()
    {
        ASSERT_TRUE(harness.kernel_->packDeviceRebalanceCompactPayloads(
            harness.launchContext(),
            d_plan,
            d_header,
            plan_capacity,
            d_source_descriptors,
            d_local_payload,
            payload_slot_count,
            payload_slot_bytes,
            rebalanceConfig(shape),
            d_pack_status));
    };
    auto copy_payload_bucket_once = [&]()
    {
        ASSERT_EQ(cudaMemcpyAsync(d_gathered_payload,
                                  d_local_payload,
                                  static_cast<size_t>(payload_slot_count) *
                                      static_cast<size_t>(payload_slot_bytes),
                                  cudaMemcpyDeviceToDevice,
                                  harness.stream_),
                  cudaSuccess);
    };
    auto unpack_payloads_once = [&]()
    {
        DeviceMoERebalanceConfig destination_config = rebalanceConfig(shape);
        destination_config.participant_id = 1u;
        ASSERT_TRUE(harness.kernel_->unpackDeviceRebalanceCollectivePayloads(
            harness.launchContext(),
            d_plan,
            d_plan_count,
            plan_capacity,
            nullptr,
            d_gathered_payload,
            payload_slot_count,
            payload_slot_bytes,
            d_transfer_slots,
            plan_count,
            destination_config,
            d_unpack_status));
    };
    auto apply_arrivals_once = [&]()
    {
        DeviceMoERebalanceConfig destination_config = rebalanceConfig(shape);
        destination_config.participant_id = 1u;
        ASSERT_TRUE(harness.kernel_->applyDeviceRebalanceArrivals(
            harness.launchContext(),
            harness.runtime_table_->deviceLayerState(0),
            d_plan,
            d_plan_count,
            plan_capacity,
            d_transfer_slots,
            plan_count,
            destination_config,
            d_apply_status,
            nullptr,
            -1));
    };
    auto time_component_us = [&](auto &&component)
    {
        CudaEvents component_events;
        EXPECT_EQ(cudaEventRecord(component_events.start, harness.stream_), cudaSuccess);
        for (int i = 0; i < iterations; ++i)
            component();
        EXPECT_EQ(cudaEventRecord(component_events.stop, harness.stream_), cudaSuccess);
        EXPECT_EQ(cudaEventSynchronize(component_events.stop), cudaSuccess);
        float component_elapsed_ms = 0.0f;
        EXPECT_EQ(cudaEventElapsedTime(&component_elapsed_ms,
                                       component_events.start,
                                       component_events.stop),
                  cudaSuccess);
        return component_elapsed_ms * 1000.0f / static_cast<float>(iterations);
    };

    for (int i = 0; i < warmups; ++i)
        run_payload_wave();
    ASSERT_EQ(cudaStreamSynchronize(harness.stream_), cudaSuccess);

    CudaEvents events;
    ASSERT_EQ(cudaEventRecord(events.start, harness.stream_), cudaSuccess);
    for (int i = 0; i < iterations; ++i)
        run_payload_wave();
    ASSERT_EQ(cudaEventRecord(events.stop, harness.stream_), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(events.stop), cudaSuccess);
    float elapsed_ms = 0.0f;
    ASSERT_EQ(cudaEventElapsedTime(&elapsed_ms, events.start, events.stop), cudaSuccess);
    const double pack_us = time_component_us(pack_payloads_once);
    const double bucket_copy_us = time_component_us(copy_payload_bucket_once);
    const double unpack_us = time_component_us(unpack_payloads_once);
    const double apply_us = time_component_us(apply_arrivals_once);

    /*
     * The unpack-only timing loop intentionally reuses an append-only status
     * record so the measured interval contains only the kernel under test.
     * Clear that diagnostic record and execute one untimed production-shaped
     * wave before validating counters; otherwise the assertion would compare
     * all microbenchmark repetitions against one wave's expected arrivals.
     */
    ASSERT_EQ(cudaMemsetAsync(
                  d_unpack_status,
                  0,
                  sizeof(DeviceMoERebalanceApplyStatus),
                  harness.stream_),
              cudaSuccess);
    run_payload_wave();
    ASSERT_EQ(cudaStreamSynchronize(harness.stream_), cudaSuccess);

    DeviceMoERebalanceApplyStatus pack_status{};
    DeviceMoERebalanceApplyStatus unpack_status{};
    DeviceMoERebalanceApplyStatus apply_status{};
    ASSERT_EQ(cudaMemcpyAsync(&pack_status,
                              d_pack_status,
                              sizeof(pack_status),
                              cudaMemcpyDeviceToHost,
                              harness.stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(&unpack_status,
                              d_unpack_status,
                              sizeof(unpack_status),
                              cudaMemcpyDeviceToHost,
                              harness.stream_),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(&apply_status,
                              d_apply_status,
                              sizeof(apply_status),
                              cudaMemcpyDeviceToHost,
                              harness.stream_),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(harness.stream_), cudaSuccess);

    EXPECT_EQ(pack_status.missing_source_descriptors, 0u);
    EXPECT_EQ(pack_status.descriptor_mismatches, 0u);
    EXPECT_EQ(unpack_status.copied_arrivals, destination_one_arrivals);
    EXPECT_EQ(unpack_status.missing_source_descriptors, 0u);
    EXPECT_EQ(unpack_status.missing_destination_slots, 0u);
    EXPECT_EQ(unpack_status.descriptor_mismatches, 0u);
    EXPECT_EQ(apply_status.applied_arrivals, destination_one_arrivals);
    EXPECT_EQ(apply_status.copy_incomplete, 0u);
    EXPECT_EQ(apply_status.changed_layers, 1u);

    const auto final_runtime = harness.copyRuntime();
    const auto &active_bank = final_runtime.banks[final_runtime.active_bank];
    const uint64_t mask_hash = fnv1a64Bytes(
        active_bank.resident_participant_mask,
        static_cast<size_t>(shape.num_experts) * sizeof(uint32_t));
    printTiming("cuda",
                "payload_movement_apply",
                shape,
                iterations,
                elapsed_ms * 1000.0f / static_cast<float>(iterations),
                mask_hash,
                payload_slot_count,
                destination_one_arrivals);
    printTiming("cuda",
                "payload_pack_compact",
                shape,
                iterations,
                pack_us,
                mask_hash,
                payload_slot_count,
                destination_one_arrivals);
    printTiming("cuda",
                "payload_bucket_copy",
                shape,
                iterations,
                bucket_copy_us,
                mask_hash,
                payload_slot_count,
                destination_one_arrivals);
    printTiming("cuda",
                "payload_unpack_collective",
                shape,
                iterations,
                unpack_us,
                mask_hash,
                payload_slot_count,
                destination_one_arrivals);
    printTiming("cuda",
                "payload_apply_arrivals",
                shape,
                iterations,
                apply_us,
                mask_hash,
                payload_slot_count,
                destination_one_arrivals);

    if (d_transfer_slots)
        cudaFree(d_transfer_slots);
    if (d_destination_slab)
        cudaFree(d_destination_slab);
    if (d_gathered_payload)
        cudaFree(d_gathered_payload);
    if (d_local_payload)
        cudaFree(d_local_payload);
    if (d_apply_status)
        cudaFree(d_apply_status);
    if (d_unpack_status)
        cudaFree(d_unpack_status);
    if (d_pack_status)
        cudaFree(d_pack_status);
    if (d_source_descriptors)
        cudaFree(d_source_descriptors);
    if (d_plan_status)
        cudaFree(d_plan_status);
    if (d_header)
        cudaFree(d_header);
    if (d_plan_count)
        cudaFree(d_plan_count);
    if (d_plan)
        cudaFree(d_plan);
    if (d_source_slab)
        cudaFree(d_source_slab);
#endif
}
