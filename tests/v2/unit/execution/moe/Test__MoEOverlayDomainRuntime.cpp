#include "execution/moe/MoEOverlayDomainRuntime.h"
#include "mocks/MockComputeStage.h"
#include "tensors/Tensors.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        ExpertComputeDomain continuationDomain()
        {
            ExpertComputeDomain domain;
            domain.name = "cuda_fast";
            domain.kind = ExpertDomainKind::SingleDevice;
            domain.backend = CollectiveBackendType::NCCL;
            domain.participants = {GlobalDeviceAddress::cuda(0)};
            domain.world_ranks = {0};
            domain.owner_rank = 0;
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertComputeDomain cpuFallbackDomain()
        {
            ExpertComputeDomain domain;
            domain.name = "cpu_cold";
            domain.kind = ExpertDomainKind::NodeLocalTP;
            domain.backend = CollectiveBackendType::UPI;
            domain.participants = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)};
            domain.world_ranks = {0, 1};
            domain.owner_rank = 0;
            domain.compute_kind = ExpertDomainComputeKind::TensorParallelExperts;
            return domain;
        }

        MoEOverlayDomainParticipant participant(
            GlobalDeviceAddress address,
            int participant_index,
            int world_rank,
            bool owned_by_current_rank,
            DeviceId local_device)
        {
            MoEOverlayDomainParticipant result;
            result.address = std::move(address);
            result.participant_index = participant_index;
            result.world_rank = world_rank;
            result.world_rank_known = true;
            result.owned_by_current_rank = owned_by_current_rank;
            result.locally_addressable = local_device.is_valid();
            result.local_device = local_device;
            return result;
        }

        MoEOverlayRuntimeDomain continuationRuntimeDomain()
        {
            MoEOverlayRuntimeDomain domain;
            domain.name = "cuda_fast";
            domain.kind = ExpertDomainKind::SingleDevice;
            domain.backend = CollectiveBackendType::NCCL;
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            domain.participants = {participant(GlobalDeviceAddress::cuda(0), 0, 0, true, DeviceId::cuda(0))};
            domain.primary_participant = GlobalDeviceAddress::cuda(0);
            domain.primary_device = DeviceId::cuda(0);
            domain.primary_world_rank = 0;
            domain.primary_world_rank_known = true;
            domain.owner_rank = 0;
            domain.primary_is_local = true;
            domain.primary_owned_by_current_rank = true;
            domain.local_reachable_for_mvp = true;
            return domain;
        }

        MoEOverlayRuntimeDomain cpuFallbackRuntimeDomain()
        {
            MoEOverlayRuntimeDomain domain;
            domain.name = "cpu_cold";
            domain.kind = ExpertDomainKind::NodeLocalTP;
            domain.backend = CollectiveBackendType::UPI;
            domain.compute_kind = ExpertDomainComputeKind::TensorParallelExperts;
            domain.participants = {
                participant(GlobalDeviceAddress::cpu(0), 0, 0, true, DeviceId::cpu()),
                participant(GlobalDeviceAddress::cpu(1), 1, 1, false, DeviceId::invalid()),
            };
            domain.primary_participant = GlobalDeviceAddress::cpu(0);
            domain.primary_device = DeviceId::cpu();
            domain.primary_world_rank = 0;
            domain.primary_world_rank_known = true;
            domain.owner_rank = 0;
            domain.primary_is_local = true;
            domain.primary_owned_by_current_rank = true;
            domain.local_reachable_for_mvp = true;
            domain.requires_domain_scoped_collective_context = true;
            domain.domain_scoped_collective_context_ready = true;
            return domain;
        }

        std::shared_ptr<MoEExpertOverlayExecutionPlan> executionPlan()
        {
            auto plan = std::make_shared<MoEExpertOverlayExecutionPlan>();
            plan->world_size = 2;
            plan->continuation_domain = "cuda_fast";
            plan->shared_expert_domain = "cuda_fast";
            plan->continuation_root_rank = 0;
            plan->current_rank.world_rank = 0;
            plan->current_rank.role = OverlayRankRole::ContinuationRoot;
            plan->current_rank.roles = {OverlayRankRole::ContinuationRoot};
            plan->current_rank.owned_domains = {"cuda_fast", "cpu_cold"};
            plan->current_rank.builds_root_graph = true;
            return plan;
        }

        std::shared_ptr<MoEExpertDispatchOutput> dispatchOutput(
            const std::string &domain,
            MoEExpertTransferMode mode = MoEExpertTransferMode::SparseTokenRows)
        {
            auto output = std::make_shared<MoEExpertDispatchOutput>();
            output->seq_len = 3;
            output->top_k = 2;
            output->d_model = 4;
            output->continuation_domain = "cuda_fast";

            MoEExpertTierDispatch tier;
            tier.tier_index = 0;
            tier.tier_name = "cold";
            tier.domain = domain;
            tier.fallback = true;
            tier.transfer_required = true;
            tier.transfer_mode = mode;
            tier.token_rows = {1, 2};
            tier.entries = {
                {.token_row = 1, .route_slot = 0, .expert_id = 4, .route_weight = 0.75f},
                {.token_row = 2, .route_slot = 1, .expert_id = 5, .route_weight = 0.25f},
            };
            tier.transfer_volume.mode = mode;
            tier.transfer_volume.dense_rows = 3;
            tier.transfer_volume.selected_rows = 2;
            tier.transfer_volume.hidden_row_bytes = 4 * sizeof(float);
            tier.transfer_volume.routing_row_bytes = 2 * sizeof(float);
            tier.transfer_volume.sparse_outbound_bytes = 2 * (tier.transfer_volume.hidden_row_bytes + tier.transfer_volume.routing_row_bytes);
            tier.transfer_volume.sparse_return_bytes = 2 * tier.transfer_volume.hidden_row_bytes;
            tier.transfer_volume.dense_outbound_bytes = 3 * (tier.transfer_volume.hidden_row_bytes + tier.transfer_volume.routing_row_bytes);
            tier.transfer_volume.dense_return_bytes = 3 * tier.transfer_volume.hidden_row_bytes;
            tier.transfer_volume.outbound_bytes = tier.transfer_volume.sparse_outbound_bytes;
            tier.transfer_volume.return_bytes = tier.transfer_volume.sparse_return_bytes;
            output->tiers.push_back(std::move(tier));
            return output;
        }

        MoEOverlayDispatchGroup dispatchGroupForTest()
        {
            MoEOverlayDispatchGroup group;
            group.domain_id = 17;
            group.layer_id = 2;
            group.dispatch_group_id = 31;
            group.participant_count = 2;
            group.participant_index = 0;
            group.owner_participant_index = 0;
            group.executor_participant_index = 0;
            group.stage_sequence = 41;
            group.microbatch_id = 0;
            group.decode_sequence = 53;
            return group;
        }

        MoEOverlayDomainWorkRequest requestFor(
            const ExpertComputeDomain &source_domain,
            const MoEOverlayRuntimeDomain &runtime_domain,
            const std::shared_ptr<MoEExpertOverlayExecutionPlan> &plan,
            TensorBase *partial)
        {
            auto dispatch = dispatchOutput(source_domain.name);
            MoEOverlayDomainWorkRequest request;
            request.layer_idx = 2;
            request.tier_index = 0;
            request.tier_name = "cold";
            request.continuation_domain = "cuda_fast";
            request.source_domain = source_domain;
            request.runtime_domain = runtime_domain;
            request.execution_plan = plan;
            request.dispatch_output = dispatch.get();
            request.dispatch_output_lifetime = dispatch;
            request.dispatch_tier_index = 0;
            request.dispatch_group = dispatchGroupForTest();
            request.output = partial;
            request.output_device = runtime_domain.primary_device;
            return request;
        }

        void attachCpuFallbackGraphDispatchParams(
            MoEOverlayDomainWorkRequest &request,
            TensorBase *input,
            TensorBase *routing_indices,
            TensorBase *routing_weights)
        {
            request.has_cpu_fallback_params = true;
            request.cpu_fallback_params.device_id = DeviceId::cpu();
            request.cpu_fallback_params.domain = request.source_domain;
            request.cpu_fallback_params.root_world_rank = 0;
            request.cpu_fallback_params.domain_id = request.dispatch_group.domain_id;
            request.cpu_fallback_params.input = input;
            request.cpu_fallback_params.routing_indices = routing_indices;
            request.cpu_fallback_params.routing_weights = routing_weights;
            request.cpu_fallback_params.output = request.output;
            request.cpu_fallback_params.seq_len = 3;
            request.cpu_fallback_params.d_model = 4;
            request.cpu_fallback_params.num_experts = 6;
            request.cpu_fallback_params.top_k = 2;
            request.cpu_fallback_params.expert_intermediate = 4;
            request.cpu_fallback_params.layer_idx = request.layer_idx;
            request.cpu_fallback_params.expert_mask = {false, false, false, false, true, true};
            request.cpu_fallback_params.transfer_mode = MoEExpertTransferMode::SparseTokenRows;
        }

        class RecordingDispatchBackend final : public IMoEOverlayDispatchBackend
        {
        public:
            MoEOverlayDispatchResult dispatch(
                const MoEOverlayDispatchGroup &group,
                const MoEOverlayDispatchRequest &request,
                IDeviceContext *ctx) override
            {
                ++calls;
                last_group = group;
                last_kind = request.kind;
                last_layer_idx = request.layer_idx;
                last_tier_index = request.tier_index;
                last_selected_rows.clear();
                if (request.selected_rows && request.selected_row_count > 0)
                {
                    last_selected_rows.assign(
                        request.selected_rows,
                        request.selected_rows + request.selected_row_count);
                }
                last_selected_row_count = request.selected_row_count;
                last_routed_entry_count = request.routed_entry_count;
                last_transfer_bytes = request.transfer_bytes;
                last_input = request.input;
                last_output = request.output;
                last_ctx = ctx;

                auto result = next_result;
                result.request_kind = request.kind;
                result.group = group;
                if (!result.partial_output)
                    result.partial_output = request.output;
                return result;
            }

            int calls = 0;
            MoEOverlayDispatchGroup last_group;
            MoEOverlayDispatchRequestKind last_kind = MoEOverlayDispatchRequestKind::NoOp;
            int last_layer_idx = -1;
            int last_tier_index = -1;
            std::vector<int> last_selected_rows;
            size_t last_selected_row_count = 0;
            size_t last_routed_entry_count = 0;
            size_t last_transfer_bytes = 0;
            TensorBase *last_input = nullptr;
            TensorBase *last_output = nullptr;
            IDeviceContext *last_ctx = nullptr;
            MoEOverlayDispatchResult next_result;
        };

    } // namespace

    TEST(Test__MoEOverlayDomainRuntime, ContinuationLocalDomainDescribesLocalWork)
    {
        auto plan = executionPlan();
        FP32Tensor partial({3, 4});
        auto request = requestFor(continuationDomain(), continuationRuntimeDomain(), plan, &partial);

        MoEOverlayDomainRuntime::Config config;
        config.execution_plan = plan;
        MoEOverlayDomainRuntime runtime(std::move(config));

        const auto descriptor = runtime.describeWork(request);

        EXPECT_EQ(descriptor.dispatch_kind, MoEOverlayDomainRuntimeDispatchKind::ContinuationLocal);
        EXPECT_FALSE(descriptor.remote_placeholder);
        EXPECT_FALSE(descriptor.compatibility_fallback_available);
        EXPECT_EQ(descriptor.expected_partial_layout, MoEOverlayDomainPartialLayout::DenseFullSequence);
        EXPECT_EQ(descriptor.partial_info.name, "cold");
        EXPECT_EQ(descriptor.partial_info.source_domain, "cuda_fast");
        EXPECT_EQ(descriptor.partial_info.source_device, DeviceId::cuda(0));
    }

    TEST(Test__MoEOverlayDomainRuntime, RemoteCpuFallbackDomainProducesExplicitPlaceholderDescriptor)
    {
        auto plan = executionPlan();
        FP32Tensor input({3, 4});
        FP32Tensor routing_indices({3, 2});
        FP32Tensor routing_weights({3, 2});
        FP32Tensor partial({3, 4});
        auto request = requestFor(cpuFallbackDomain(), cpuFallbackRuntimeDomain(), plan, &partial);
        attachCpuFallbackGraphDispatchParams(request, &input, &routing_indices, &routing_weights);

        MoEOverlayDomainRuntime::Config config;
        config.execution_plan = plan;
        MoEOverlayDomainRuntime runtime(std::move(config));

        const auto descriptor = runtime.describeWork(request);

        EXPECT_EQ(descriptor.dispatch_kind, MoEOverlayDomainRuntimeDispatchKind::GraphDispatchCollective);
        EXPECT_FALSE(descriptor.remote_placeholder);
        EXPECT_TRUE(descriptor.compatibility_fallback_available);
        EXPECT_EQ(descriptor.domain_name, "cpu_cold");
        EXPECT_EQ(descriptor.transfer_mode, MoEExpertTransferMode::SparseTokenRows);
        EXPECT_EQ(descriptor.expected_partial_layout, MoEOverlayDomainPartialLayout::SparseTokenRows);
        EXPECT_EQ(descriptor.selected_rows, (std::vector<int>{1, 2}));
        EXPECT_EQ(descriptor.routed_entries, 2u);
        EXPECT_EQ(descriptor.partial_info.selected_rows, (std::vector<int>{1, 2}));
    }

    TEST(Test__MoEOverlayDomainRuntime, GraphDispatchCollectiveInvokesConfiguredBackendWithSparseRequest)
    {
        auto plan = executionPlan();
        FP32Tensor input({3, 4});
        FP32Tensor routing_indices({3, 2});
        FP32Tensor routing_weights({3, 2});
        FP32Tensor partial({3, 4});
        auto request = requestFor(cpuFallbackDomain(), cpuFallbackRuntimeDomain(), plan, &partial);
        attachCpuFallbackGraphDispatchParams(request, &input, &routing_indices, &routing_weights);

        auto backend = std::make_shared<RecordingDispatchBackend>();
        backend->next_result.ok = true;
        backend->next_result.metrics.remote_endpoint_work_count = 1;
        backend->next_result.metrics.wait_ns = 123;

        MoEOverlayDomainRuntime::Config config;
        config.execution_plan = plan;
        config.dispatch_backend = backend;
        MoEOverlayDomainRuntime runtime(std::move(config));
        llaminar2::testing::MockDeviceContext ctx(DeviceId::cpu(), ComputeBackendType::CPU);

        const auto result = runtime.submit(request, &ctx);

        ASSERT_TRUE(result.ok) << result.error;
        EXPECT_EQ(result.dispatch_kind, MoEOverlayDomainRuntimeDispatchKind::GraphDispatchCollective);
        EXPECT_FALSE(result.compatibility_fallback_used);
        EXPECT_EQ(result.partial_tensor, &partial);
        EXPECT_EQ(result.dispatch_metrics.routed_request_count, 1u);
        EXPECT_EQ(result.dispatch_metrics.remote_endpoint_work_count, 1u);
        EXPECT_EQ(result.dispatch_metrics.wait_ns, 123u);

        EXPECT_EQ(backend->calls, 1);
        EXPECT_EQ(backend->last_group.domain_id, request.dispatch_group.domain_id);
        EXPECT_EQ(backend->last_group.dispatch_group_id, request.dispatch_group.dispatch_group_id);
        EXPECT_EQ(backend->last_kind, MoEOverlayDispatchRequestKind::RoutedWork);
        EXPECT_EQ(backend->last_layer_idx, 2);
        EXPECT_EQ(backend->last_tier_index, 0);
        EXPECT_EQ(backend->last_selected_rows, (std::vector<int>{1, 2}));
        EXPECT_EQ(backend->last_selected_row_count, 2u);
        EXPECT_EQ(backend->last_routed_entry_count, 2u);
        EXPECT_GT(backend->last_transfer_bytes, 0u);
        EXPECT_EQ(backend->last_input, &input);
        EXPECT_EQ(backend->last_output, &partial);
        EXPECT_EQ(backend->last_ctx, &ctx);
    }

    TEST(Test__MoEOverlayDomainRuntime, GraphDispatchCollectivePreservesNoOpRequestIdentity)
    {
        auto plan = executionPlan();
        FP32Tensor input({3, 4});
        FP32Tensor routing_indices({3, 2});
        FP32Tensor routing_weights({3, 2});
        FP32Tensor partial({3, 4});
        auto request = requestFor(cpuFallbackDomain(), cpuFallbackRuntimeDomain(), plan, &partial);
        request.dispatch_output_lifetime->tiers[0].entries.clear();
        request.dispatch_output_lifetime->tiers[0].token_rows.clear();
        attachCpuFallbackGraphDispatchParams(request, &input, &routing_indices, &routing_weights);

        auto backend = std::make_shared<RecordingDispatchBackend>();
        MoEOverlayDomainRuntime::Config config;
        config.execution_plan = plan;
        config.dispatch_backend = backend;
        MoEOverlayDomainRuntime runtime(std::move(config));
        llaminar2::testing::MockDeviceContext ctx(DeviceId::cpu(), ComputeBackendType::CPU);

        const auto result = runtime.submit(request, &ctx);

        ASSERT_TRUE(result.ok) << result.error;
        EXPECT_EQ(result.request_kind, MoEOverlayDispatchRequestKind::NoOp);
        EXPECT_EQ(result.dispatch_metrics.no_op_count, 1u);
        EXPECT_EQ(result.dispatch_metrics.routed_request_count, 0u);
        EXPECT_EQ(backend->calls, 1);
        EXPECT_EQ(backend->last_kind, MoEOverlayDispatchRequestKind::NoOp);
        EXPECT_EQ(backend->last_group.stage_sequence, request.dispatch_group.stage_sequence);
        EXPECT_EQ(backend->last_selected_row_count, 0u);
        EXPECT_TRUE(backend->last_selected_rows.empty());
    }

    TEST(Test__MoEOverlayDomainRuntime, GraphDispatchBackendFailureSurfacesAtCollectiveBoundary)
    {
        auto plan = executionPlan();
        FP32Tensor partial({3, 4});
        auto request = requestFor(cpuFallbackDomain(), cpuFallbackRuntimeDomain(), plan, &partial);

        auto backend = std::make_shared<RecordingDispatchBackend>();
        backend->next_result.ok = false;
        backend->next_result.error_code = 42;
        backend->next_result.error = "synthetic dispatch failure";

        MoEOverlayDomainRuntime::Config config;
        config.execution_plan = plan;
        config.dispatch_backend = backend;
        config.enable_compatibility_fallback = false;
        MoEOverlayDomainRuntime runtime(std::move(config));
        llaminar2::testing::MockDeviceContext ctx(DeviceId::cpu(), ComputeBackendType::CPU);

        const auto result = runtime.submit(request, &ctx);

        EXPECT_FALSE(result.ok);
        EXPECT_EQ(result.dispatch_kind, MoEOverlayDomainRuntimeDispatchKind::GraphDispatchCollective);
        EXPECT_EQ(backend->calls, 1);
        EXPECT_NE(result.error.find("synthetic dispatch failure"), std::string::npos) << result.error;
    }

    TEST(Test__MoEOverlayDomainRuntime, CpuFallbackGraphDispatchWithoutGenericBackendFailsBeforeNativeDomainContextPath)
    {
        auto plan = executionPlan();
        FP32Tensor input({3, 4});
        FP32Tensor routing_indices({3, 2});
        FP32Tensor routing_weights({3, 2});
        FP32Tensor partial({3, 4});
        auto request = requestFor(cpuFallbackDomain(), cpuFallbackRuntimeDomain(), plan, &partial);
        attachCpuFallbackGraphDispatchParams(request, &input, &routing_indices, &routing_weights);

        MoEOverlayDomainRuntime::Config config;
        config.execution_plan = plan;
        config.enable_compatibility_fallback = false;
        MoEOverlayDomainRuntime runtime(std::move(config));
        llaminar2::testing::MockDeviceContext ctx(DeviceId::cpu(), ComputeBackendType::CPU);

        const auto result = runtime.submit(request, &ctx);

        EXPECT_FALSE(result.ok);
        EXPECT_EQ(result.dispatch_kind, MoEOverlayDomainRuntimeDispatchKind::GraphDispatchCollective);
        EXPECT_NE(result.error.find("requires a dispatch backend"), std::string::npos) << result.error;
        EXPECT_NE(result.error.find("refusing native CPU fallback compatibility path"), std::string::npos) << result.error;
    }

    TEST(Test__MoEOverlayDomainRuntime, CpuFallbackGraphDispatchWithEmptyExpertMaskSkipsNativeDomainContextPath)
    {
        auto plan = executionPlan();
        FP32Tensor input({3, 4});
        FP32Tensor routing_indices({3, 2});
        FP32Tensor routing_weights({3, 2});
        FP32Tensor partial({3, 4});
        std::fill_n(partial.mutable_data(), partial.numel(), 1.0f);
        auto request = requestFor(cpuFallbackDomain(), cpuFallbackRuntimeDomain(), plan, &partial);
        attachCpuFallbackGraphDispatchParams(request, &input, &routing_indices, &routing_weights);
        request.cpu_fallback_params.expert_mask.assign(6, false);

        MoEOverlayDomainRuntime::Config config;
        config.execution_plan = plan;
        config.enable_compatibility_fallback = false;
        MoEOverlayDomainRuntime runtime(std::move(config));
        llaminar2::testing::MockDeviceContext ctx(DeviceId::cpu(), ComputeBackendType::CPU);

        const auto result = runtime.submit(request, &ctx);

        ASSERT_TRUE(result.ok) << result.error;
        EXPECT_EQ(result.dispatch_kind, MoEOverlayDomainRuntimeDispatchKind::GraphDispatchCollective);
        EXPECT_EQ(result.request_kind, MoEOverlayDispatchRequestKind::NoOp);
        EXPECT_FALSE(result.compatibility_fallback_used);
        EXPECT_EQ(result.partial_tensor, &partial);
        EXPECT_TRUE(result.partial_info.selected_rows.empty());
        EXPECT_EQ(result.dispatch_metrics.no_op_count, 1u);
        EXPECT_EQ(result.dispatch_metrics.routed_request_count, 0u);
        EXPECT_EQ(result.dispatch_metrics.selected_row_count, 0u);
        EXPECT_EQ(result.dispatch_metrics.routed_entry_count, 0u);
        EXPECT_EQ(result.dispatch_metrics.transfer_bytes, 0u);
        const float *partial_data = partial.data();
        ASSERT_NE(partial_data, nullptr);
        EXPECT_TRUE(std::all_of(partial_data, partial_data + partial.numel(), [](float value)
                                { return value == 0.0f; }));
    }

} // namespace llaminar2::test
