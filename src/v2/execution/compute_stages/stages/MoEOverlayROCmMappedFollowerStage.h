/**
 * @file MoEOverlayROCmMappedFollowerStage.h
 * @brief Single-node resident ROCm mapped-overlay follower stage for single-row decode.
 *
 * Replaces the three-stage (10-kernel) per-layer follower chain
 * (`DispatchConsume -> MoEExpertCompute -> ReturnPack`) with ONE resident
 * cooperative ROCm megakernel (`ROCmMappedFollowerMoE`), collapsing a 48-layer
 * MI50 follower graph from 504 nodes down to 51 nodes while preserving exact
 * mapped-activation packet validation, direct-to-mapped preweighted canonical
 * route publication, dynamic RCU bank flips, histogram updates, and device
 * service telemetry.
 */
#pragma once

#ifdef HAVE_ROCM

#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "execution/compute_stages/IComputeStage.h"
#include "execution/moe/DeviceMoEOverlayServiceTelemetry.h"
#include "execution/moe/MoEOverlayNodeLocalRankBatchTransport.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "kernels/rocm/moe/ROCmMappedFollowerMoE.h"
#include "tensors/TensorClasses.h"
#include "transfer/TransferEngine.h"
#include "utils/FNV1a.h"
#include "utils/Logger.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace llaminar2
{
    /** @brief One fused resident ROCm follower layer (acquire + compute + return release). */
    class MoEOverlayROCmMappedFollowerStage final : public IComputeStage
    {
    public:
        /** @brief Immutable cold setup bindings for one ROCm mapped follower layer. */
        struct Params
        {
            DeviceId device_id = DeviceId::invalid();
            MoEOverlayMappedActivationDeviceLane lane{};
            MoEOverlayRoutePlacementDeviceBinding placement{};
            DeviceMoELayerRuntime *runtime_layer = nullptr;
            DeviceMoEOverlayServiceTelemetryBinding service_telemetry{};
            const MoEOverlayInferenceGraphRole *service_graph_role = nullptr;
            std::int32_t *active_row_count_device = nullptr;
            TensorBase *scratch_buffer = nullptr;
            TensorBase *hidden = nullptr;
            TensorBase *routing_indices = nullptr;
            TensorBase *routing_weights = nullptr;
            TensorBase *local_canonical_route_contributions = nullptr;
            int d_model = 0;
            int expert_intermediate = 0;
            int num_experts = 0;
            int top_k = 0;
            int model_layer_index = -1;
            std::uint32_t stage_ordinal = 0u;
            std::uint32_t gate_execution_mask = 0u;
            std::uint32_t down_execution_mask = 0u;
            bool observe_histogram = true;
            std::string stage_name;
        };

        /** @brief Construct and validate immutable layer geometry and mapped lane bindings. */
        explicit MoEOverlayROCmMappedFollowerStage(Params params)
            : IComputeStage(params.device_id), params_(std::move(params))
        {
            if (!params_.device_id.is_rocm() || !params_.lane.valid() || !params_.runtime_layer ||
                !params_.active_row_count_device || !params_.scratch_buffer ||
                !params_.hidden || !params_.routing_indices || !params_.routing_weights ||
                !params_.local_canonical_route_contributions ||
                params_.d_model <= 0 || params_.expert_intermediate <= 0 ||
                params_.num_experts <= 0 || params_.top_k <= 0 || params_.top_k > 16 ||
                params_.model_layer_index < 0 ||
                !rocm::fastMappedFollowerMoEKernelFunction(params_.gate_execution_mask, params_.down_execution_mask))
            {
                throw std::invalid_argument("MoEOverlayROCmMappedFollowerStage received invalid cold bindings");
            }
            if (!params_.service_telemetry.empty() && !params_.service_telemetry.valid())
                throw std::invalid_argument("MoEOverlayROCmMappedFollowerStage received partial service telemetry");
            if (isOriginalFinalStage())
            {
                moe_kernel_ = ::llaminar::v2::kernels::KernelFactory::createMoEKernel(params_.device_id);
                if (!moe_kernel_)
                    throw std::runtime_error("MoEOverlayROCmMappedFollowerStage could not create final-stage MoE kernel");
            }
        }

        ComputeStageType type() const override { return ComputeStageType::MOE_EXPERT_FFN; }
        bool isCollectiveStage() const override { return true; }
        std::string name() const override { return params_.stage_name; }
        bool supportsBackend(ComputeBackendType backend) const override { return backend == ComputeBackendType::GPU_ROCM; }
        bool isGraphCapturable() const override { return program_.has_value(); }
        bool supportsGraphCaptureAfterLaunchPreparation() const override { return true; }
        bool supportsPaddedPrefillGraphCapturePreflight() const override { return false; }
        bool supportsLazyPrefillGraphCapturePreflight() const override { return false; }
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
        {
            return GraphLaunchPreparationPolicy::CaptureOnly;
        }
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
        CollectiveGraphCaptureAuthority collectiveGraphCaptureAuthority() const noexcept override
        {
            return CollectiveGraphCaptureAuthority::ParticipantLocalCapturedTransfer;
        }

        StageBufferContract bufferContract() const override { return StageBufferContract::build(); }
        StageBufferRequirements getBufferRequirements() const override { return {}; }

        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override
        {
            if (!ctx || ctx->deviceId() != params_.device_id || !stream)
                return false;
            if (program_)
            {
                return prepared_stream_ == stream &&
                       prepared_scratch_ptr_ == params_.scratch_buffer->gpu_data_ptr();
            }

            if (params_.stage_ordinal == 0u)
            {
                IBackend *const backend = getBackendFor(params_.device_id);
                if (!backend || !params_.lane.grant_initialization_event ||
                    !backend->streamWaitEvent(
                        stream, params_.lane.grant_initialization_event, params_.device_id.gpu_ordinal()))
                {
                    LOG_ERROR("[MoEOverlayROCmMappedFollowerStage] Failed to join grant initialization on stage 0");
                    return false;
                }
            }

            const auto payload = params_.lane.dispatchPayload(/*physical_rows=*/1);
            if (!payload.valid() || payload.selection.physical_rows != 1 || !payload.selection.usesCompactRows())
                return false;

            auto *scratch_base = static_cast<std::uint8_t *>(params_.scratch_buffer->gpu_data_ptr());
            const std::size_t required_bytes = rocm::MappedFollowerScratchLayout::totalBytes(
                params_.d_model, params_.top_k, params_.expert_intermediate);
            if (!scratch_base || params_.scratch_buffer->size_bytes() < required_bytes ||
                !params_.hidden->gpu_data_ptr() || !params_.routing_indices->gpu_data_ptr() ||
                !params_.routing_weights->gpu_data_ptr() ||
                !params_.local_canonical_route_contributions->gpu_data_ptr())
            {
                LOG_ERROR("[MoEOverlayROCmMappedFollowerStage] Device buffers not materialized for layer "
                          << params_.model_layer_index);
                return false;
            }

            const std::uint32_t bank = moeOverlayActivationBufferIndex(params_.stage_ordinal);
            const std::uint64_t timeline = moeOverlayActivationLeasedTimelineValue(
                moeOverlayActivationBufferVisit(params_.stage_ordinal));

            auto bind_wait = [&](std::size_t offset, std::uint64_t value,
                                 MoEOverlayActivationTimelineWaitDeviceBinding &out) -> bool {
                try
                {
                    const auto resolved = TransferEngine::instance().bindMappedTimelineKernelWait64(
                        *params_.lane.mapped_region, offset, value, params_.device_id);
                    out = {.signal = resolved.deviceSignal(), .value = resolved.value()};
                    return out.valid();
                }
                catch (const std::exception &e)
                {
                    LOG_ERROR("[MoEOverlayROCmMappedFollowerStage] Timeline wait bind failed: " << e.what());
                    return false;
                }
            };
            auto bind_pub = [&](std::size_t offset, std::uint64_t value,
                                MoEOverlayActivationTimelinePublishDeviceBinding &out) -> bool {
                try
                {
                    const auto resolved = TransferEngine::instance().bindMappedTimelineKernelPublish64(
                        *params_.lane.mapped_region, offset, value, params_.device_id);
                    out = {.signal = resolved.deviceSignal(), .value = resolved.value()};
                    return out.valid();
                }
                catch (const std::exception &e)
                {
                    LOG_ERROR("[MoEOverlayROCmMappedFollowerStage] Timeline publish bind failed: " << e.what());
                    return false;
                }
            };

            MoEOverlayActivationTimelineWaitDeviceBinding dispatch_acquire{};
            MoEOverlayActivationTimelinePublishDeviceBinding return_pub{};
            MoEOverlayActivationTimelineWaitDeviceBinding admission_acquire{};
            if (!bind_wait(params_.lane.dispatch_signal_offsets[bank], timeline, dispatch_acquire) ||
                !bind_pub(params_.lane.return_signal_offsets[bank], timeline, return_pub))
                return false;
            if (params_.stage_ordinal == 0u &&
                !bind_wait(params_.lane.admission_signal_offset,
                           kMoEOverlayActivationAdmissionTimeline, admission_acquire))
                return false;

            auto dispatch_view = payload.packet;
            if (usesSharedSingleRowDispatchHidden())
                dispatch_view.hidden_rows_fp32 = params_.lane.shared_dispatch_hidden_rows_fp32;
            const MoEOverlayActivationDispatchConsumeLaunch dispatch_pkt{
                .packet = dispatch_view,
                .placement = params_.placement,
                .hidden_payload_layout = payload.selection.layout,
                .control = params_.lane.control_device,
                .grant = params_.lane.grant_device,
                .hidden_rows_fp32 = static_cast<float *>(params_.hidden->gpu_data_ptr()),
                .routing_indices_fp32 = static_cast<float *>(params_.routing_indices->gpu_data_ptr()),
                .routing_weights_fp32 = static_cast<float *>(params_.routing_weights->gpu_data_ptr()),
                .active_row_count_device = params_.active_row_count_device,
                .physical_rows = 1,
                .stage_ordinal = params_.stage_ordinal,
                .model_layer_index = params_.model_layer_index,
            };
            const MoEOverlayActivationReturnPackLaunch return_pkt{
                .local_canonical_route_contributions_fp32 =
                    static_cast<const float *>(params_.local_canonical_route_contributions->gpu_data_ptr()),
                .dispatch = payload.packet,
                .returned = params_.lane.returned,
                .control = params_.lane.control_device,
                .grant = params_.lane.grant_device,
                .original_invocations = params_.lane.original_invocations,
                .retained_stage_count = params_.lane.retained_stage_count,
                .physical_rows = 1,
                .stage_ordinal = params_.stage_ordinal,
                .model_layer_index = params_.model_layer_index,
            };

            args_ = rocm::MappedFollowerMoEArgs{
                .barrier_state = reinterpret_cast<std::uint32_t *>(
                    scratch_base + rocm::MappedFollowerScratchLayout::kBarrierOffset),
                .scratch_header = reinterpret_cast<rocm::MappedFollowerScratchHeader *>(
                    scratch_base + rocm::MappedFollowerScratchLayout::kHeaderOffset),
                .scratch_hidden_q8 = scratch_base + rocm::MappedFollowerScratchLayout::kHiddenQ8Offset,
                .scratch_gate_partials = reinterpret_cast<float *>(
                    scratch_base + rocm::MappedFollowerScratchLayout::gatePartialsOffset(params_.d_model)),
                .dispatch_launch = {.packet = dispatch_pkt, .acquire = dispatch_acquire},
                .return_launch = {.packet = return_pkt, .publication = return_pub},
                .admission_acquire = admission_acquire,
                .runtime = params_.runtime_layer,
                .service_telemetry_layer = params_.service_telemetry.layer_telemetry,
                .service_telemetry_sample = params_.service_telemetry.sample,
                .service_graph_role = params_.service_graph_role,
                .wall_clock_ns_per_tick_q32 = 0u,
                .hidden_columns = params_.d_model,
                .routed_columns = params_.expert_intermediate,
                .experts = params_.num_experts,
                .top_k = params_.top_k,
                .routed_gate_policy = {},
                .routed_down_policy = {},
                .gate_execution_mask = params_.gate_execution_mask,
                .down_execution_mask = params_.down_execution_mask,
                .has_admission_acquire = params_.stage_ordinal == 0u ? std::uint8_t{1u} : std::uint8_t{0u},
                .observe_histogram = params_.observe_histogram ? std::uint8_t{1u} : std::uint8_t{0u},
            };

            const std::uint64_t generation = reinterpret_cast<std::uintptr_t>(scratch_base);
            try
            {
                program_ = rocm::prepareMappedFollowerMoE(params_.device_id, args_, generation);
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[MoEOverlayROCmMappedFollowerStage] Cold admission failed for layer "
                          << params_.model_layer_index << ": " << e.what());
                return false;
            }
            prepared_stream_ = stream;
            prepared_scratch_ptr_ = scratch_base;
            prepared_generation_ = generation;
            setGPUStream(stream);
            return true;
        }

        bool execute(IDeviceContext *ctx) override
        {
            if (!ctx || ctx->deviceId() != params_.device_id || !program_)
                return false;
            void *const stream = requireGPUStream();
            if (!stream || stream != prepared_stream_ ||
                params_.scratch_buffer->gpu_data_ptr() != prepared_scratch_ptr_)
            {
                LOG_ERROR("[MoEOverlayROCmMappedFollowerStage] Stream or scratch binding changed");
                return false;
            }
            try
            {
                rocm::launchMappedFollowerMoE(*program_, args_, stream, prepared_generation_);
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[MoEOverlayROCmMappedFollowerStage] Launch failed for layer "
                          << params_.model_layer_index << ": " << e.what());
                return false;
            }
            if (isOriginalFinalStage())
            {
                const MoEKernelLaunchContext launch_ctx{
                    .stream = stream,
                    .workspace = nullptr,
                };
                if (!moe_kernel_ || !moe_kernel_->retireOriginalMoEOverlayInvocation(
                        launch_ctx,
                        MoEOverlayOriginalNativeRetireLaunch{
                            .original_invocations = params_.lane.original_invocations,
                            .control = params_.lane.control_device,
                            .grant = params_.lane.grant_device,
                            .endpoint = MoEOverlayActivationEndpoint::Follower,
                            .physical_rows = 1,
                            .stage_ordinal = params_.stage_ordinal,
                            .retained_stage_count = params_.lane.retained_stage_count,
                        }))
                {
                    LOG_ERROR("[MoEOverlayROCmMappedFollowerStage] Final-stage retirement launch failed");
                    return false;
                }
            }
            return true;
        }

        std::uint64_t graphCaptureVariantSignature() const override
        {
            std::uint64_t sig = kFNV1a64OffsetBasis;
            const auto mix = [&](std::uint64_t v) { sig = (sig ^ v) * kFNV1a64Prime; };
            mix(static_cast<std::uint64_t>(params_.device_id.ordinal));
            mix(static_cast<std::uint64_t>(params_.model_layer_index));
            mix(static_cast<std::uint64_t>(params_.stage_ordinal));
            mix(static_cast<std::uint64_t>(params_.d_model));
            mix(static_cast<std::uint64_t>(params_.expert_intermediate));
            mix(static_cast<std::uint64_t>(params_.num_experts));
            mix(static_cast<std::uint64_t>(params_.top_k));
            mix(params_.gate_execution_mask);
            mix(params_.down_execution_mask);
            mix(reinterpret_cast<std::uintptr_t>(params_.runtime_layer));
            mix(reinterpret_cast<std::uintptr_t>(params_.lane.control_device));
            mix(reinterpret_cast<std::uintptr_t>(params_.lane.grant_device));
            mix(static_cast<std::uint64_t>(usesSharedSingleRowDispatchHidden() ? 1u : 0u));
            mix(reinterpret_cast<std::uintptr_t>(
                usesSharedSingleRowDispatchHidden()
                    ? params_.lane.shared_dispatch_hidden_rows_fp32
                    : params_.lane.dispatch.hidden_rows_fp32));
            mix(reinterpret_cast<std::uintptr_t>(params_.scratch_buffer->gpu_data_ptr()));
            return sig ? sig : 1u;
        }

    protected:
        StageDumpInfo buildDumpInfoImpl() const override
        {
            StageDumpInfo info;
            if (params_.hidden)
                info.addInput("hidden", params_.hidden, 1u, static_cast<std::size_t>(params_.d_model));
            return info;
        }

    private:
        /** @return Whether the paired GPU continuation packs one shared 10 KB single-row hidden vector. */
        bool usesSharedSingleRowDispatchHidden() const noexcept
        {
            return params_.lane.shared_dispatch_hidden_rows_fp32 != nullptr &&
                   params_.lane.shared_dispatch_consumer_count > 1u &&
                   params_.lane.shared_dispatch_consumer_count <= 32u &&
                   params_.lane.control_host != nullptr &&
                   params_.lane.control_host->channel.prefill_wire_policy ==
                       static_cast<std::uint32_t>(
                           MoEOverlayActivationPrefillWirePolicy::FixedPoint32Adaptive16);
        }

        bool isOriginalFinalStage() const noexcept
        {
            return params_.lane.original_invocations.deviceOwned() &&
                   params_.lane.retained_stage_count != 0u &&
                   params_.stage_ordinal == params_.lane.retained_stage_count - 1u;
        }

        Params params_;
        std::unique_ptr<IMoEKernel> moe_kernel_;
        rocm::MappedFollowerMoEArgs args_{};
        std::optional<rocm::PreparedMappedFollowerMoE> program_;
        void *prepared_stream_ = nullptr;
        void *prepared_scratch_ptr_ = nullptr;
        std::uint64_t prepared_generation_ = 0u;
    };

} // namespace llaminar2

#endif // HAVE_ROCM
