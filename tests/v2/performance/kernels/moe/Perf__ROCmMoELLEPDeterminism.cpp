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
    extern "C" bool hipMoE_group_prefill_routes_runtime(
        const float *routing_indices,
        const float *routing_weights,
        void *runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int filter_to_local_runtime_experts,
        int retain_routes_for_deferred_commit,
        int device_idx,
        void *stream);

    extern "C" bool hipMoE_build_runtime_original_to_grouped(
        const void *runtime,
        int *original_to_grouped,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int device_idx,
        void *stream);

    extern "C" bool hipMoE_materialize_runtime_prefill_plan(
        const void *runtime,
        llaminar2::DeviceNativeVNNIMatrixDesc *gate_descs,
        llaminar2::DeviceNativeVNNIMatrixDesc *up_descs,
        llaminar2::DeviceNativeVNNIMatrixDesc *down_descs,
        int *active_expert_ids,
        int num_experts,
        int max_active_experts,
        int device_idx,
        void *stream);

    extern "C" bool hipMoE_group_prefill_routes_and_materialize_plan_runtime(
        const float *routing_indices,
        const float *routing_weights,
        void *runtime,
        int *original_to_grouped,
        llaminar2::DeviceNativeVNNIMatrixDesc *gate_descs,
        llaminar2::DeviceNativeVNNIMatrixDesc *up_descs,
        llaminar2::DeviceNativeVNNIMatrixDesc *down_descs,
        int *active_expert_ids,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int max_active_experts,
        int filter_to_local_runtime_experts,
        int retain_routes_for_deferred_commit,
        int device_idx,
        void *stream);

    extern "C" bool hipMoE_regroup_prefill_routes_runtime_assignments(
        void *runtime,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int retain_routes_for_deferred_commit,
        int device_idx,
        void *stream);

    extern "C" bool hipMoE_regroup_prefill_routes_and_materialize_plan_runtime(
        void *runtime,
        int *original_to_grouped,
        llaminar2::DeviceNativeVNNIMatrixDesc *gate_descs,
        llaminar2::DeviceNativeVNNIMatrixDesc *up_descs,
        llaminar2::DeviceNativeVNNIMatrixDesc *down_descs,
        int *active_expert_ids,
        int current_slots,
        int max_slots,
        int num_experts,
        int top_k,
        int max_active_experts,
        int retain_routes_for_deferred_commit,
        int device_idx,
        void *stream);

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

            owned_kernel_ = KernelFactory::createMoEKernel(device_);
            kernel_ = owned_kernel_.get();
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
            if (resident_expert_slab_)
                (void)hipFree(resident_expert_slab_);
            if (stream_)
                (void)hipStreamDestroy(stream_);
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
                    hipMalloc(
                        &resident_expert_slab_,
                        static_cast<size_t>(bytes_per_expert) *
                            static_cast<size_t>(shape_.num_experts)),
                    hipSuccess);
                installSyntheticAllLocalExpertDescriptors(
                    runtime,
                    shape_,
                    payload_spec,
                    /*participant_id=*/0,
                    resident_expert_slab_,
                    bytes_per_expert);
            }
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
                launchContext(),
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

        void uploadRuntime(const DeviceMoELayerRuntime &runtime) const
        {
            ASSERT_EQ(hipMemcpyAsync(runtime_table_->deviceLayerState(0),
                                     &runtime,
                                     sizeof(runtime),
                                     hipMemcpyHostToDevice,
                                     stream_),
                      hipSuccess);
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
            harness.launchContext(),
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
            harness.launchContext(),
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
            harness.launchContext(),
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

/**
 * @brief Time fused HIP runtime descriptor and active-expert plan publication.
 *
 * Setup publishes realistic resident descriptors before timing. The measured
 * interval contains only fused descriptor materialization and deterministic
 * active-list publication: allocation, transfer, and per-launch
 * synchronization remain outside the sample, and one event pair brackets the
 * complete sequence on the explicit stream.
 */
TEST(Perf__MoELLEPDeterminism, ROCm_RuntimePrefillPlanPublication)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const Shape shape{};
    const int warmups = envInt("LLAMINAR_MOE_ACTIVE_LIST_WARMUPS", 20);
    const int iterations = envInt("LLAMINAR_MOE_ACTIVE_LIST_ITERS", 1000);
    const float max_avg_us = static_cast<float>(
        envInt("LLAMINAR_MOE_ACTIVE_LIST_MAX_US", 25));
    ROCmHarness harness(shape);
    harness.prepare(
        /*all_participants_resident=*/true,
        makeResidentAssignmentRouteExperts(shape),
        makeRouteWeights(shape));

    DeviceNativeVNNIMatrixDesc *gate_descs = nullptr;
    DeviceNativeVNNIMatrixDesc *up_descs = nullptr;
    DeviceNativeVNNIMatrixDesc *down_descs = nullptr;
    int *active_expert_ids = nullptr;
    const std::size_t descriptor_bytes =
        static_cast<std::size_t>(shape.num_experts) *
        sizeof(DeviceNativeVNNIMatrixDesc);
    ASSERT_EQ(hipMalloc(&gate_descs, descriptor_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&up_descs, descriptor_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&down_descs, descriptor_bytes), hipSuccess);
    ASSERT_EQ(
        hipMalloc(
            &active_expert_ids,
            static_cast<std::size_t>(shape.num_experts) * sizeof(int)),
        hipSuccess);

    for (int i = 0; i < warmups; ++i)
    {
        ASSERT_TRUE(hipMoE_materialize_runtime_prefill_plan(
            harness.runtime_table_->deviceLayerState(0),
            gate_descs,
            up_descs,
            down_descs,
            active_expert_ids,
            shape.num_experts,
            shape.num_experts,
            /*device_idx=*/0,
            harness.stream_));
    }
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);

    HipEvents events;
    ASSERT_EQ(hipEventRecord(events.start, harness.stream_), hipSuccess);
    for (int i = 0; i < iterations; ++i)
    {
        ASSERT_TRUE(hipMoE_materialize_runtime_prefill_plan(
            harness.runtime_table_->deviceLayerState(0),
            gate_descs,
            up_descs,
            down_descs,
            active_expert_ids,
            shape.num_experts,
            shape.num_experts,
            /*device_idx=*/0,
            harness.stream_));
    }
    ASSERT_EQ(hipEventRecord(events.stop, harness.stream_), hipSuccess);
    ASSERT_EQ(hipEventSynchronize(events.stop), hipSuccess);
    float elapsed_ms = 0.0f;
    ASSERT_EQ(
        hipEventElapsedTime(&elapsed_ms, events.start, events.stop),
        hipSuccess);

    const auto runtime = harness.copyRuntime();
    std::vector<int32_t> counts(static_cast<std::size_t>(shape.num_experts));
    std::vector<int32_t> actual(static_cast<std::size_t>(shape.num_experts));
    ASSERT_EQ(
        hipMemcpyAsync(
            counts.data(),
            runtime.expert_counts,
            counts.size() * sizeof(int32_t),
            hipMemcpyDeviceToHost,
            harness.stream_),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(
            actual.data(),
            active_expert_ids,
            actual.size() * sizeof(int32_t),
            hipMemcpyDeviceToHost,
            harness.stream_),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);

    std::vector<int32_t> expected;
    expected.reserve(static_cast<std::size_t>(shape.num_experts));
    for (int expert = 0; expert < shape.num_experts; ++expert)
    {
        if (counts[static_cast<std::size_t>(expert)] > 0)
            expected.push_back(expert);
    }
    const uint32_t active_count = static_cast<uint32_t>(expected.size());
    expected.resize(static_cast<std::size_t>(shape.num_experts), -1);
    EXPECT_EQ(actual, expected);

    const float avg_us =
        elapsed_ms * 1000.0f / static_cast<float>(iterations);
    printTiming(
        "rocm",
        "runtime_prefill_plan_publication",
        shape,
        iterations,
        avg_us,
        fnv1a64(actual),
        active_count,
        0);
    EXPECT_LT(avg_us, max_avg_us)
        << "fused runtime prefill publication regressed toward serial execution";

    ASSERT_EQ(hipFree(down_descs), hipSuccess);
    ASSERT_EQ(hipFree(up_descs), hipSuccess);
    ASSERT_EQ(hipFree(gate_descs), hipSuccess);
    ASSERT_EQ(hipFree(active_expert_ids), hipSuccess);
#endif
}

/**
 * @brief Prove and time complete verifier-plan fusion on HIP.
 *
 * The baseline deliberately invokes the former route grouping, inverse-map
 * publication, and descriptor/active-list publications separately. The fused
 * candidate must reproduce every byte of runtime route state and every
 * compute-plan product before its timing can be considered. Allocations and
 * host copies remain outside the timed interval.
 */
TEST(Perf__MoELLEPDeterminism, ROCm_CompleteRuntimePrefillPlanFusion)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    Shape shape{};
    shape.seq_len = envInt("LLAMINAR_MOE_COMPLETE_PLAN_ROWS", 4);
    const int current_slots = shape.seq_len * shape.top_k;
    constexpr int verifier_plan_max_slots = 256;
    ASSERT_GT(current_slots, 0);
    ASSERT_LE(current_slots, verifier_plan_max_slots);
    const int max_active_experts =
        std::min(current_slots, shape.num_experts);
    const int warmups = envInt("LLAMINAR_MOE_COMPLETE_PLAN_WARMUPS", 100);
    const int iterations = envInt("LLAMINAR_MOE_COMPLETE_PLAN_ITERS", 5000);

    ROCmHarness harness(shape);
    harness.prepare(
        /*all_participants_resident=*/true,
        makeResidentAssignmentRouteExperts(shape),
        makeRouteWeights(shape));
    const auto runtime = harness.copyRuntime();
    const auto *route_indices = static_cast<const float *>(
        harness.route_indices_tensor_->gpu_data_ptr());
    const auto *route_weights = static_cast<const float *>(
        harness.route_weights_tensor_->gpu_data_ptr());
    ASSERT_NE(route_indices, nullptr);
    ASSERT_NE(route_weights, nullptr);

    const std::size_t descriptor_bytes =
        static_cast<std::size_t>(shape.num_experts) *
        sizeof(DeviceNativeVNNIMatrixDesc);
    const std::size_t route_int_bytes =
        static_cast<std::size_t>(current_slots) * sizeof(int32_t);
    const std::size_t route_float_bytes =
        static_cast<std::size_t>(current_slots) * sizeof(float);
    const std::size_t expert_int_bytes =
        static_cast<std::size_t>(shape.num_experts) * sizeof(int32_t);
    const std::size_t active_bytes =
        static_cast<std::size_t>(max_active_experts) * sizeof(int32_t);

    DeviceNativeVNNIMatrixDesc *baseline_gate = nullptr;
    DeviceNativeVNNIMatrixDesc *baseline_up = nullptr;
    DeviceNativeVNNIMatrixDesc *baseline_down = nullptr;
    DeviceNativeVNNIMatrixDesc *fused_gate = nullptr;
    DeviceNativeVNNIMatrixDesc *fused_up = nullptr;
    DeviceNativeVNNIMatrixDesc *fused_down = nullptr;
    int32_t *baseline_active = nullptr;
    int32_t *fused_active = nullptr;
    int32_t *baseline_inverse = nullptr;
    int32_t *fused_inverse = nullptr;
    ASSERT_EQ(hipMalloc(&baseline_gate, descriptor_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&baseline_up, descriptor_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&baseline_down, descriptor_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&fused_gate, descriptor_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&fused_up, descriptor_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&fused_down, descriptor_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&baseline_active, active_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&fused_active, active_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&baseline_inverse, route_int_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&fused_inverse, route_int_bytes), hipSuccess);

    const auto launch_baseline = [&]()
    {
        return hipMoE_group_prefill_routes_runtime(
                   route_indices,
                   route_weights,
                   harness.runtime_table_->deviceLayerState(0),
                   current_slots,
                   current_slots,
                   shape.num_experts,
                   shape.top_k,
                   /*filter_to_local_runtime_experts=*/0,
                   /*retain_routes_for_deferred_commit=*/0,
                   /*device_idx=*/0,
                   harness.stream_) &&
               hipMoE_build_runtime_original_to_grouped(
                   harness.runtime_table_->deviceLayerState(0),
                   baseline_inverse,
                   current_slots,
                   current_slots,
                   shape.num_experts,
                   shape.top_k,
                   /*device_idx=*/0,
                   harness.stream_) &&
               hipMoE_materialize_runtime_prefill_plan(
                   harness.runtime_table_->deviceLayerState(0),
                   baseline_gate,
                   baseline_up,
                   baseline_down,
                   baseline_active,
                   shape.num_experts,
                   max_active_experts,
                   /*device_idx=*/0,
                   harness.stream_);
    };
    const auto launch_fused = [&]()
    {
        return hipMoE_group_prefill_routes_and_materialize_plan_runtime(
            route_indices,
            route_weights,
            harness.runtime_table_->deviceLayerState(0),
            fused_inverse,
            fused_gate,
            fused_up,
            fused_down,
            fused_active,
            current_slots,
            current_slots,
            shape.num_experts,
            shape.top_k,
            max_active_experts,
            /*filter_to_local_runtime_experts=*/0,
            /*retain_routes_for_deferred_commit=*/0,
            /*device_idx=*/0,
            harness.stream_);
    };

    ASSERT_TRUE(launch_baseline());
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);
    const auto copy_bytes = [](const void *device_ptr, std::size_t bytes)
    {
        std::vector<uint8_t> result(bytes);
        EXPECT_EQ(
            hipMemcpy(
                result.data(),
                device_ptr,
                bytes,
                hipMemcpyDeviceToHost),
            hipSuccess);
        return result;
    };
    const auto baseline_gate_bytes = copy_bytes(baseline_gate, descriptor_bytes);
    const auto baseline_up_bytes = copy_bytes(baseline_up, descriptor_bytes);
    const auto baseline_down_bytes = copy_bytes(baseline_down, descriptor_bytes);
    const auto baseline_active_bytes = copy_bytes(baseline_active, active_bytes);
    const auto baseline_inverse_bytes = copy_bytes(baseline_inverse, route_int_bytes);
    const auto baseline_counts = copy_bytes(runtime.expert_counts, expert_int_bytes);
    const auto baseline_offsets = copy_bytes(runtime.expert_offsets, expert_int_bytes);
    const auto baseline_route_ids = copy_bytes(runtime.route_expert_ids, route_int_bytes);
    const auto baseline_route_weights = copy_bytes(runtime.route_weights, route_float_bytes);
    const auto baseline_participants = copy_bytes(runtime.route_participant_ids, route_int_bytes);
    const auto baseline_grouped_ids = copy_bytes(runtime.grouped_token_ids, route_int_bytes);
    const auto baseline_grouped_weights =
        copy_bytes(runtime.grouped_route_weights, route_float_bytes);

    ASSERT_TRUE(launch_fused());
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);
    EXPECT_EQ(copy_bytes(fused_gate, descriptor_bytes), baseline_gate_bytes);
    EXPECT_EQ(copy_bytes(fused_up, descriptor_bytes), baseline_up_bytes);
    EXPECT_EQ(copy_bytes(fused_down, descriptor_bytes), baseline_down_bytes);
    EXPECT_EQ(copy_bytes(fused_active, active_bytes), baseline_active_bytes);
    EXPECT_EQ(copy_bytes(fused_inverse, route_int_bytes), baseline_inverse_bytes);
    EXPECT_EQ(copy_bytes(runtime.expert_counts, expert_int_bytes), baseline_counts);
    EXPECT_EQ(copy_bytes(runtime.expert_offsets, expert_int_bytes), baseline_offsets);
    EXPECT_EQ(copy_bytes(runtime.route_expert_ids, route_int_bytes), baseline_route_ids);
    EXPECT_EQ(copy_bytes(runtime.route_weights, route_float_bytes), baseline_route_weights);
    EXPECT_EQ(copy_bytes(runtime.route_participant_ids, route_int_bytes), baseline_participants);
    EXPECT_EQ(copy_bytes(runtime.grouped_token_ids, route_int_bytes), baseline_grouped_ids);
    EXPECT_EQ(
        copy_bytes(runtime.grouped_route_weights, route_float_bytes),
        baseline_grouped_weights);

    for (int i = 0; i < warmups; ++i)
    {
        ASSERT_TRUE(launch_baseline());
        ASSERT_TRUE(launch_fused());
    }
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);

    const auto measure = [&](const auto &launch)
    {
        HipEvents events;
        EXPECT_EQ(hipEventRecord(events.start, harness.stream_), hipSuccess);
        for (int i = 0; i < iterations; ++i)
            EXPECT_TRUE(launch());
        EXPECT_EQ(hipEventRecord(events.stop, harness.stream_), hipSuccess);
        EXPECT_EQ(hipEventSynchronize(events.stop), hipSuccess);
        float elapsed_ms = 0.0f;
        EXPECT_EQ(
            hipEventElapsedTime(&elapsed_ms, events.start, events.stop),
            hipSuccess);
        return elapsed_ms * 1000.0f / static_cast<float>(iterations);
    };

    const float baseline_us = measure(launch_baseline);
    const float fused_us = measure(launch_fused);
    std::cout << "backend,case,rows,slots,iters,baseline_us,fused_us,speedup\n"
              << "rocm,complete_runtime_prefill_plan_fusion,"
              << shape.seq_len << ',' << current_slots << ',' << iterations << ','
              << baseline_us << ',' << fused_us << ','
              << baseline_us / fused_us << '\n';
    EXPECT_LT(fused_us, baseline_us)
        << "one-workgroup complete plan publication must beat separate launches";

    /*
     * Model the post-planner LLEP ledger with both local and remote routes.
     * The publication kernel must preserve all assignments while compacting
     * only participant zero's rows into the local grouped compute plan.
     */
    std::vector<int32_t> assigned_participants(
        static_cast<std::size_t>(current_slots));
    for (int route_slot = 0; route_slot < current_slots; ++route_slot)
    {
        assigned_participants[static_cast<std::size_t>(route_slot)] =
            route_slot % 2;
    }
    ASSERT_GT(
        std::count(assigned_participants.begin(),
                   assigned_participants.end(),
                   0),
        0);
    ASSERT_GT(
        std::count(assigned_participants.begin(),
                   assigned_participants.end(),
                   1),
        0);
    ASSERT_EQ(
        hipMemcpyAsync(
            runtime.route_participant_ids,
            assigned_participants.data(),
            route_int_bytes,
            hipMemcpyHostToDevice,
            harness.stream_),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);

    const auto launch_assigned_baseline = [&]()
    {
        return hipMoE_regroup_prefill_routes_runtime_assignments(
                   harness.runtime_table_->deviceLayerState(0),
                   current_slots,
                   current_slots,
                   shape.num_experts,
                   shape.top_k,
                   /*retain_routes_for_deferred_commit=*/0,
                   /*device_idx=*/0,
                   harness.stream_) &&
               hipMoE_build_runtime_original_to_grouped(
                   harness.runtime_table_->deviceLayerState(0),
                   baseline_inverse,
                   current_slots,
                   current_slots,
                   shape.num_experts,
                   shape.top_k,
                   /*device_idx=*/0,
                   harness.stream_) &&
               hipMoE_materialize_runtime_prefill_plan(
                   harness.runtime_table_->deviceLayerState(0),
                   baseline_gate,
                   baseline_up,
                   baseline_down,
                   baseline_active,
                   shape.num_experts,
                   max_active_experts,
                   /*device_idx=*/0,
                   harness.stream_);
    };
    const auto launch_assigned_fused = [&]()
    {
        return hipMoE_regroup_prefill_routes_and_materialize_plan_runtime(
            harness.runtime_table_->deviceLayerState(0),
            fused_inverse,
            fused_gate,
            fused_up,
            fused_down,
            fused_active,
            current_slots,
            current_slots,
            shape.num_experts,
            shape.top_k,
            max_active_experts,
            /*retain_routes_for_deferred_commit=*/0,
            /*device_idx=*/0,
            harness.stream_);
    };

    ASSERT_TRUE(launch_assigned_baseline());
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);
    const auto assigned_baseline_gate_bytes =
        copy_bytes(baseline_gate, descriptor_bytes);
    const auto assigned_baseline_up_bytes =
        copy_bytes(baseline_up, descriptor_bytes);
    const auto assigned_baseline_down_bytes =
        copy_bytes(baseline_down, descriptor_bytes);
    const auto assigned_baseline_active_bytes =
        copy_bytes(baseline_active, active_bytes);
    const auto assigned_baseline_inverse_bytes =
        copy_bytes(baseline_inverse, route_int_bytes);
    const auto assigned_baseline_counts =
        copy_bytes(runtime.expert_counts, expert_int_bytes);
    const auto assigned_baseline_offsets =
        copy_bytes(runtime.expert_offsets, expert_int_bytes);
    const auto assigned_baseline_route_ids =
        copy_bytes(runtime.route_expert_ids, route_int_bytes);
    const auto assigned_baseline_route_weights =
        copy_bytes(runtime.route_weights, route_float_bytes);
    const auto assigned_baseline_participants =
        copy_bytes(runtime.route_participant_ids, route_int_bytes);
    const auto assigned_baseline_grouped_ids =
        copy_bytes(runtime.grouped_token_ids, route_int_bytes);
    const auto assigned_baseline_grouped_weights =
        copy_bytes(runtime.grouped_route_weights, route_float_bytes);

    ASSERT_TRUE(launch_assigned_fused());
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);
    EXPECT_EQ(copy_bytes(fused_gate, descriptor_bytes),
              assigned_baseline_gate_bytes);
    EXPECT_EQ(copy_bytes(fused_up, descriptor_bytes),
              assigned_baseline_up_bytes);
    EXPECT_EQ(copy_bytes(fused_down, descriptor_bytes),
              assigned_baseline_down_bytes);
    EXPECT_EQ(copy_bytes(fused_active, active_bytes),
              assigned_baseline_active_bytes);
    EXPECT_EQ(copy_bytes(fused_inverse, route_int_bytes),
              assigned_baseline_inverse_bytes);
    EXPECT_EQ(copy_bytes(runtime.expert_counts, expert_int_bytes),
              assigned_baseline_counts);
    EXPECT_EQ(copy_bytes(runtime.expert_offsets, expert_int_bytes),
              assigned_baseline_offsets);
    EXPECT_EQ(copy_bytes(runtime.route_expert_ids, route_int_bytes),
              assigned_baseline_route_ids);
    EXPECT_EQ(copy_bytes(runtime.route_weights, route_float_bytes),
              assigned_baseline_route_weights);
    EXPECT_EQ(copy_bytes(runtime.route_participant_ids, route_int_bytes),
              assigned_baseline_participants);
    EXPECT_EQ(copy_bytes(runtime.grouped_token_ids, route_int_bytes),
              assigned_baseline_grouped_ids);
    EXPECT_EQ(copy_bytes(runtime.grouped_route_weights, route_float_bytes),
              assigned_baseline_grouped_weights);

    for (int i = 0; i < warmups; ++i)
    {
        ASSERT_TRUE(launch_assigned_baseline());
        ASSERT_TRUE(launch_assigned_fused());
    }
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);

    const float assigned_baseline_us = measure(launch_assigned_baseline);
    const float assigned_fused_us = measure(launch_assigned_fused);
    std::cout << "rocm,complete_runtime_assigned_plan_fusion,"
              << shape.seq_len << ',' << current_slots << ',' << iterations << ','
              << assigned_baseline_us << ',' << assigned_fused_us << ','
              << assigned_baseline_us / assigned_fused_us << '\n';
    EXPECT_LT(assigned_fused_us, assigned_baseline_us)
        << "one-workgroup assigned-plan publication must beat separate launches";

    EXPECT_EQ(hipFree(fused_inverse), hipSuccess);
    EXPECT_EQ(hipFree(baseline_inverse), hipSuccess);
    EXPECT_EQ(hipFree(fused_active), hipSuccess);
    EXPECT_EQ(hipFree(baseline_active), hipSuccess);
    EXPECT_EQ(hipFree(fused_down), hipSuccess);
    EXPECT_EQ(hipFree(fused_up), hipSuccess);
    EXPECT_EQ(hipFree(fused_gate), hipSuccess);
    EXPECT_EQ(hipFree(baseline_down), hipSuccess);
    EXPECT_EQ(hipFree(baseline_up), hipSuccess);
    EXPECT_EQ(hipFree(baseline_gate), hipSuccess);
#endif
}

TEST(Perf__MoELLEPDeterminism, ROCm_GroupPrefillRoutesDeterministic)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const Shape shape{};
    const int warmups = envInt("LLAMINAR_MOE_GROUP_ROUTES_WARMUPS", 10);
    const int iterations = envInt("LLAMINAR_MOE_GROUP_ROUTES_ITERS", 100);
    ROCmHarness harness(shape);
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
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);

    HipEvents events;
    ASSERT_EQ(hipEventRecord(events.start, harness.stream_), hipSuccess);
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
    ASSERT_EQ(hipEventRecord(events.stop, harness.stream_), hipSuccess);
    ASSERT_EQ(hipEventSynchronize(events.stop), hipSuccess);
    float elapsed_ms = 0.0f;
    ASSERT_EQ(hipEventElapsedTime(&elapsed_ms, events.start, events.stop), hipSuccess);

    const auto runtime = harness.copyRuntime();
    std::vector<int32_t> counts(static_cast<size_t>(shape.num_experts));
    ASSERT_EQ(hipMemcpyAsync(counts.data(),
                             runtime.expert_counts,
                             counts.size() * sizeof(int32_t),
                             hipMemcpyDeviceToHost,
                             harness.stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);
    const int total_routes =
        std::accumulate(counts.begin(), counts.end(), 0);
    ASSERT_EQ(total_routes, shape.seq_len * shape.top_k);
    printTiming("rocm",
                "group_prefill_routes",
                shape,
                iterations,
                elapsed_ms * 1000.0f / static_cast<float>(iterations),
                fnv1a64(counts),
                static_cast<uint32_t>(total_routes),
                0);
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
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);

    HipEvents events;
    ASSERT_EQ(hipEventRecord(events.start, harness.stream_), hipSuccess);
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

TEST(Perf__MoELLEPDeterminism, ROCm_DynamicMaintenancePackAndControllerDeterministic)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const Shape shape{};
    const DeviceMoERebalanceConfig config = dynamicMaintenanceConfig(shape);
    const uint32_t wave_layers = std::max(1u, config.layer_wave_count);
    const uint32_t local_histogram_count = wave_layers * config.num_experts;
    const uint32_t gathered_histogram_count =
        config.participant_count * local_histogram_count;
    const int warmups = envInt("LLAMINAR_MOE_DYNAMIC_MAINT_WARMUPS", 10);
    const int iterations = envInt("LLAMINAR_MOE_DYNAMIC_MAINT_ITERS", 100);

    ROCmHarness harness(shape);
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
    ASSERT_EQ(hipMalloc(&d_local_histograms,
                        static_cast<size_t>(local_histogram_count) * sizeof(uint64_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(&d_gathered_histograms,
                        static_cast<size_t>(gathered_histogram_count) * sizeof(uint64_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(&d_plan, plan_capacity * sizeof(DeviceMoERebalancePlanEntry)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_plan_count, sizeof(uint32_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_status, sizeof(DeviceMoERebalanceStatus)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_header, sizeof(DeviceMoERebalanceCommandBufferHeader)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_wave_state, sizeof(DeviceMoERebalanceWaveState)), hipSuccess);
    ASSERT_EQ(
        hipMalloc(
            &d_controller_states,
            controller_states.size() *
                sizeof(DeviceMoERebalanceGraphControllerState)),
        hipSuccess);
    ASSERT_EQ(hipMemsetAsync(d_gathered_histograms,
                             0,
                             static_cast<size_t>(gathered_histogram_count) * sizeof(uint64_t),
                             harness.stream_),
              hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(
            d_controller_states,
            controller_states.data(),
            controller_states.size() *
                sizeof(DeviceMoERebalanceGraphControllerState),
            hipMemcpyHostToDevice,
            harness.stream_),
        hipSuccess);

    auto run_maintenance = [&](int transaction_index)
    {
        ASSERT_TRUE(harness.kernel_->packDeviceRebalanceHistograms(
            harness.launchContext(),
            harness.runtime_table_->deviceLayerState(0),
            d_local_histograms,
            config));
        ASSERT_EQ(hipMemcpyAsync(d_gathered_histograms,
                                 d_local_histograms,
                                 static_cast<size_t>(local_histogram_count) * sizeof(uint64_t),
                                 hipMemcpyDeviceToDevice,
                                 harness.stream_),
                  hipSuccess);
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
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);

    HipEvents events;
    ASSERT_EQ(hipEventRecord(events.start, harness.stream_), hipSuccess);
    for (int i = 0; i < iterations; ++i)
        run_maintenance(warmups + i);
    ASSERT_EQ(hipEventRecord(events.stop, harness.stream_), hipSuccess);
    ASSERT_EQ(hipEventSynchronize(events.stop), hipSuccess);
    float elapsed_ms = 0.0f;
    ASSERT_EQ(hipEventElapsedTime(&elapsed_ms, events.start, events.stop), hipSuccess);

    uint32_t plan_count = 0;
    DeviceMoERebalanceStatus status{};
    ASSERT_EQ(hipMemcpyAsync(&plan_count,
                             d_plan_count,
                             sizeof(plan_count),
                             hipMemcpyDeviceToHost,
                             harness.stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(&status,
                             d_status,
                             sizeof(status),
                             hipMemcpyDeviceToHost,
                             harness.stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);
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
    printTiming("rocm",
                "dynamic_maintenance_pack_controller",
                shape,
                iterations,
                elapsed_ms * 1000.0f / static_cast<float>(iterations),
                plan_hash,
                status.dynamic_ownership_swap_accepts,
                plan_count);

    if (d_wave_state)
        (void)hipFree(d_wave_state);
    if (d_controller_states)
        (void)hipFree(d_controller_states);
    if (d_header)
        (void)hipFree(d_header);
    if (d_status)
        (void)hipFree(d_status);
    if (d_plan_count)
        (void)hipFree(d_plan_count);
    if (d_plan)
        (void)hipFree(d_plan);
    if (d_gathered_histograms)
        (void)hipFree(d_gathered_histograms);
    if (d_local_histograms)
        (void)hipFree(d_local_histograms);
#endif
}

/**
 * @brief Time Qwen3.6 Dynamic maintenance with least-loaded assignment.
 *
 * This regression deliberately isolates the optional
 * Dynamic-plus-LeastLoadedResident controller from transfer payload movement.
 * Canonical current-batch LLEP keeps durable maintenance Off. The Dynamic lane
 * replays the controller on a
 * dedicated stream while forward graphs execute elsewhere; an accidentally
 * serialized multi-millisecond planner therefore slows every forward stage,
 * even when the grouped expert kernels themselves are economical.
 *
 * All runtime state, gathered evidence, and output buffers are allocated and
 * uploaded before the timed region.  The timed region contains controller
 * launches only, with one event synchronization after the complete batch.
 */
TEST(Perf__MoELLEPDeterminism, ROCm_Qwen36DynamicLeastLoadedMaintenanceControllerEconomy)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    Shape shape{};
    shape.participant_count = 2;
    constexpr uint32_t num_layers = 40u;
    constexpr uint32_t plan_capacity = 42u;
    DeviceMoELLEPLayerPlanScratch *d_llep_layer_plans = nullptr;
    const uint32_t layer_wave_count = static_cast<uint32_t>(
        std::min(
            4,
            envInt("LLAMINAR_MOE_DYNAMIC_LEAST_LOADED_MAINT_LAYER_WAVE", 4)));
    const DeviceMoERebalanceConfig config =
        dynamicLeastLoadedMaintenanceConfig(shape, num_layers, layer_wave_count);
    const uint32_t gathered_histogram_count =
        config.participant_count * config.layer_wave_count * config.num_experts;
    const int warmups =
        envInt("LLAMINAR_MOE_DYNAMIC_LEAST_LOADED_MAINT_WARMUPS", 2);
    const int iterations =
        envInt("LLAMINAR_MOE_DYNAMIC_LEAST_LOADED_MAINT_ITERS", 20);

    ROCmHarness harness(shape);
    harness.prepare(/*all_participants_resident=*/false,
                    makeSourceZeroTransferRouteExperts(shape),
                    makeRouteWeights(shape));
    ASSERT_EQ(
        hipMalloc(
            &d_llep_layer_plans,
            static_cast<size_t>(
                std::max(1u, config.layer_window_count)) *
                sizeof(DeviceMoELLEPLayerPlanScratch)),
        hipSuccess);

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
            hipMemcpyAsync(
                runtime_table.deviceLayerState(static_cast<int>(layer)),
                &runtime,
                sizeof(runtime),
                hipMemcpyHostToDevice,
                harness.stream_),
            hipSuccess);
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
        hipMalloc(
            &d_gathered_histograms,
            static_cast<size_t>(gathered_histogram_count) * sizeof(uint64_t)),
        hipSuccess);
    ASSERT_EQ(
        hipMalloc(
            &d_plan,
            static_cast<size_t>(plan_capacity) *
                sizeof(DeviceMoERebalancePlanEntry)),
        hipSuccess);
    ASSERT_EQ(hipMalloc(&d_plan_count, sizeof(uint32_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_status, sizeof(DeviceMoERebalanceStatus)), hipSuccess);
    ASSERT_EQ(
        hipMalloc(&d_header, sizeof(DeviceMoERebalanceCommandBufferHeader)),
        hipSuccess);
    ASSERT_EQ(
        hipMalloc(&d_wave_state, sizeof(DeviceMoERebalanceWaveState)),
        hipSuccess);
    ASSERT_EQ(
        hipMalloc(
            &d_controller_states,
            controller_states.size() *
                sizeof(DeviceMoERebalanceGraphControllerState)),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(
            d_gathered_histograms,
            gathered.data(),
            gathered.size() * sizeof(uint64_t),
            hipMemcpyHostToDevice,
            harness.stream_),
        hipSuccess);
    ASSERT_EQ(
        hipMemsetAsync(d_wave_state, 0, sizeof(*d_wave_state), harness.stream_),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(
            d_controller_states,
            controller_states.data(),
            controller_states.size() *
                sizeof(DeviceMoERebalanceGraphControllerState),
            hipMemcpyHostToDevice,
            harness.stream_),
        hipSuccess);

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
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);

    HipEvents events;
    ASSERT_EQ(hipEventRecord(events.start, harness.stream_), hipSuccess);
    for (int i = 0; i < iterations; ++i)
        run_controller(warmups + i);
    ASSERT_EQ(hipEventRecord(events.stop, harness.stream_), hipSuccess);
    ASSERT_EQ(hipEventSynchronize(events.stop), hipSuccess);
    float elapsed_ms = 0.0f;
    ASSERT_EQ(
        hipEventElapsedTime(&elapsed_ms, events.start, events.stop),
        hipSuccess);

    uint32_t plan_count = 0;
    DeviceMoERebalanceStatus status{};
    ASSERT_EQ(
        hipMemcpyAsync(
            &plan_count,
            d_plan_count,
            sizeof(plan_count),
            hipMemcpyDeviceToHost,
            harness.stream_),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(
            &status,
            d_status,
            sizeof(status),
            hipMemcpyDeviceToHost,
            harness.stream_),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);
    ASSERT_EQ(
        status.status_code,
        static_cast<uint32_t>(DeviceMoERebalanceStatusCode::Ok));
    ASSERT_LE(plan_count, plan_capacity);
    ASSERT_EQ(status.payload_bucket_overflow, 0u);

    const auto plan = harness.copyPlan(d_plan, plan_count);
    printTiming(
        "rocm",
        "qwen36_dynamic_least_loaded_maintenance_controller",
        shape,
        iterations,
        elapsed_ms * 1000.0f / static_cast<float>(iterations),
        fnv1a64Plan(plan.data(), plan.size()),
        status.llep_assignment_span_count,
        plan_count);

    if (d_llep_layer_plans)
        (void)hipFree(d_llep_layer_plans);
    if (d_controller_states)
        (void)hipFree(d_controller_states);
    if (d_wave_state)
        (void)hipFree(d_wave_state);
    if (d_header)
        (void)hipFree(d_header);
    if (d_status)
        (void)hipFree(d_status);
    if (d_plan_count)
        (void)hipFree(d_plan_count);
    if (d_plan)
        (void)hipFree(d_plan);
    if (d_gathered_histograms)
        (void)hipFree(d_gathered_histograms);
#endif
}

TEST(Perf__MoELLEPDeterminism, ROCm_PayloadMovementAndApplyDeterministic)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    if (!hasROCmDevice())
        GTEST_SKIP() << "No ROCm device available";

    const Shape shape{};
    const SyntheticPayloadSpec payload_spec{};
    const uint64_t expert_bytes = syntheticExpertDataBytes(payload_spec);
    const uint64_t payload_slot_bytes = syntheticPayloadSlotBytes(payload_spec);
    ASSERT_GT(expert_bytes, 0u);
    ASSERT_GT(payload_slot_bytes, sizeof(DeviceMoEExpertDirectoryEntry));

    const int warmups = envInt("LLAMINAR_MOE_PAYLOAD_WARMUPS", 5);
    const int iterations = envInt("LLAMINAR_MOE_PAYLOAD_ITERS", 50);
    ROCmHarness harness(shape);
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
    ASSERT_EQ(hipMalloc(&d_source_slab,
                        static_cast<size_t>(source_experts) *
                            static_cast<size_t>(expert_bytes)),
              hipSuccess);
    for (uint32_t ordinal = 0; ordinal < source_experts; ++ordinal)
    {
        ASSERT_EQ(hipMemsetAsync(d_source_slab + static_cast<uint64_t>(ordinal) * expert_bytes,
                                 static_cast<int>((ordinal * 37u + 11u) & 0xffu),
                                 static_cast<size_t>(expert_bytes),
                                 harness.stream_),
                  hipSuccess);
    }

    auto source_runtime = harness.copyRuntime();
    installSyntheticLocalExpertDescriptors(
        source_runtime, shape, payload_spec, 0, d_source_slab, expert_bytes);
    harness.uploadRuntime(source_runtime);

    ASSERT_EQ(hipMalloc(&d_plan, plan_capacity * sizeof(DeviceMoERebalancePlanEntry)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_plan_count, sizeof(uint32_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_header, sizeof(DeviceMoERebalanceCommandBufferHeader)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_plan_status, sizeof(DeviceMoERebalanceStatus)), hipSuccess);
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
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);

    uint32_t plan_count = 0;
    DeviceMoERebalanceStatus plan_status{};
    ASSERT_EQ(hipMemcpyAsync(&plan_count,
                             d_plan_count,
                             sizeof(plan_count),
                             hipMemcpyDeviceToHost,
                             harness.stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(&plan_status,
                             d_plan_status,
                             sizeof(plan_status),
                             hipMemcpyDeviceToHost,
                             harness.stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);
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

    ASSERT_EQ(hipMalloc(&d_source_descriptors,
                        static_cast<size_t>(shape.participant_count) *
                            plan_capacity * sizeof(DeviceMoEExpertDirectoryEntry)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(&d_pack_status, sizeof(DeviceMoERebalanceApplyStatus)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_unpack_status, sizeof(DeviceMoERebalanceApplyStatus)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_apply_status, sizeof(DeviceMoERebalanceApplyStatus)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_local_payload,
                        static_cast<size_t>(payload_slot_count) *
                            static_cast<size_t>(payload_slot_bytes)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(&d_gathered_payload,
                        static_cast<size_t>(shape.participant_count) *
                            static_cast<size_t>(payload_slot_count) *
                            static_cast<size_t>(payload_slot_bytes)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(&d_destination_slab,
                        static_cast<size_t>(plan_count) *
                            static_cast<size_t>(expert_bytes)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(&d_transfer_slots,
                        static_cast<size_t>(plan_count) *
                            sizeof(DeviceMoEExpertDirectoryEntry)),
              hipSuccess);

    std::vector<DeviceMoEExpertDirectoryEntry> transfer_slots(plan_count);
    for (uint32_t slot = 0; slot < plan_count; ++slot)
    {
        transfer_slots[slot] = makeSyntheticTransferSlot(
            d_destination_slab + static_cast<uint64_t>(slot) * expert_bytes,
            payload_spec,
            1u,
            slot);
    }
    ASSERT_EQ(hipMemcpyAsync(d_transfer_slots,
                             transfer_slots.data(),
                             transfer_slots.size() * sizeof(DeviceMoEExpertDirectoryEntry),
                             hipMemcpyHostToDevice,
                             harness.stream_),
              hipSuccess);

    ASSERT_TRUE(harness.kernel_->packDeviceRebalanceSourceDescriptors(
        harness.launchContext(),
        harness.runtime_table_->deviceLayerState(0),
        d_plan,
        d_header,
        plan_capacity,
        d_source_descriptors,
        rebalanceConfig(shape),
        nullptr));
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);

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
            ASSERT_EQ(hipMemcpyAsync(d_gathered_payload,
                                     d_local_payload,
                                     static_cast<size_t>(payload_slot_count) *
                                         static_cast<size_t>(payload_slot_bytes),
                                     hipMemcpyDeviceToDevice,
                                     harness.stream_),
                      hipSuccess);
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
        ASSERT_EQ(hipMemcpyAsync(d_gathered_payload,
                                 d_local_payload,
                                 static_cast<size_t>(payload_slot_count) *
                                     static_cast<size_t>(payload_slot_bytes),
                                 hipMemcpyDeviceToDevice,
                                 harness.stream_),
                  hipSuccess);
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
        HipEvents component_events;
        EXPECT_EQ(hipEventRecord(component_events.start, harness.stream_), hipSuccess);
        for (int i = 0; i < iterations; ++i)
            component();
        EXPECT_EQ(hipEventRecord(component_events.stop, harness.stream_), hipSuccess);
        EXPECT_EQ(hipEventSynchronize(component_events.stop), hipSuccess);
        float component_elapsed_ms = 0.0f;
        EXPECT_EQ(hipEventElapsedTime(&component_elapsed_ms,
                                      component_events.start,
                                      component_events.stop),
                  hipSuccess);
        return component_elapsed_ms * 1000.0f / static_cast<float>(iterations);
    };

    for (int i = 0; i < warmups; ++i)
        run_payload_wave();
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);

    HipEvents events;
    ASSERT_EQ(hipEventRecord(events.start, harness.stream_), hipSuccess);
    for (int i = 0; i < iterations; ++i)
        run_payload_wave();
    ASSERT_EQ(hipEventRecord(events.stop, harness.stream_), hipSuccess);
    ASSERT_EQ(hipEventSynchronize(events.stop), hipSuccess);
    float elapsed_ms = 0.0f;
    ASSERT_EQ(hipEventElapsedTime(&elapsed_ms, events.start, events.stop), hipSuccess);
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
    ASSERT_EQ(hipMemsetAsync(
                  d_unpack_status,
                  0,
                  sizeof(DeviceMoERebalanceApplyStatus),
                  harness.stream_),
              hipSuccess);
    run_payload_wave();
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);

    DeviceMoERebalanceApplyStatus pack_status{};
    DeviceMoERebalanceApplyStatus unpack_status{};
    DeviceMoERebalanceApplyStatus apply_status{};
    ASSERT_EQ(hipMemcpyAsync(&pack_status,
                             d_pack_status,
                             sizeof(pack_status),
                             hipMemcpyDeviceToHost,
                             harness.stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(&unpack_status,
                             d_unpack_status,
                             sizeof(unpack_status),
                             hipMemcpyDeviceToHost,
                             harness.stream_),
              hipSuccess);
    ASSERT_EQ(hipMemcpyAsync(&apply_status,
                             d_apply_status,
                             sizeof(apply_status),
                             hipMemcpyDeviceToHost,
                             harness.stream_),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(harness.stream_), hipSuccess);

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
    printTiming("rocm",
                "payload_movement_apply",
                shape,
                iterations,
                elapsed_ms * 1000.0f / static_cast<float>(iterations),
                mask_hash,
                payload_slot_count,
                destination_one_arrivals);
    printTiming("rocm",
                "payload_pack_compact",
                shape,
                iterations,
                pack_us,
                mask_hash,
                payload_slot_count,
                destination_one_arrivals);
    printTiming("rocm",
                "payload_bucket_copy",
                shape,
                iterations,
                bucket_copy_us,
                mask_hash,
                payload_slot_count,
                destination_one_arrivals);
    printTiming("rocm",
                "payload_unpack_collective",
                shape,
                iterations,
                unpack_us,
                mask_hash,
                payload_slot_count,
                destination_one_arrivals);
    printTiming("rocm",
                "payload_apply_arrivals",
                shape,
                iterations,
                apply_us,
                mask_hash,
                payload_slot_count,
                destination_one_arrivals);

    if (d_transfer_slots)
        (void)hipFree(d_transfer_slots);
    if (d_destination_slab)
        (void)hipFree(d_destination_slab);
    if (d_gathered_payload)
        (void)hipFree(d_gathered_payload);
    if (d_local_payload)
        (void)hipFree(d_local_payload);
    if (d_apply_status)
        (void)hipFree(d_apply_status);
    if (d_unpack_status)
        (void)hipFree(d_unpack_status);
    if (d_pack_status)
        (void)hipFree(d_pack_status);
    if (d_source_descriptors)
        (void)hipFree(d_source_descriptors);
    if (d_plan_status)
        (void)hipFree(d_plan_status);
    if (d_header)
        (void)hipFree(d_header);
    if (d_plan_count)
        (void)hipFree(d_plan_count);
    if (d_plan)
        (void)hipFree(d_plan);
    if (d_source_slab)
        (void)hipFree(d_source_slab);
#endif
}
