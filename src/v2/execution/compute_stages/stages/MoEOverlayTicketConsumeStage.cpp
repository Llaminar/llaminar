/**
 * @file MoEOverlayTicketConsumeStage.cpp
 * @brief Publishes a completed heterogeneous MoE return ticket into a GPU graph.
 *
 * A graph-native ExpertOverlay layer deliberately crosses a heterogeneous
 * boundary between captured continuation work and host-staged sparse expert
 * work.  The CPU side accumulates routed-expert rows in the ticket's pinned
 * return buffer; this stage is the sole GPU ingress for those bytes.  Besides
 * enqueuing the fixed-address H2D copy, it must publish the exact producer
 * stream through TransferEngine.  Without that publication the following GPU
 * stages can legally observe the tensor's stale host authority and overwrite
 * the just-arrived routed result during their own input preparation.
 */

#include "MoEOverlayTicketConsumeStage.h"

#include "../../../backends/BackendManager.h"
#include "../../../backends/IBackend.h"
#include "../../../kernels/IMoEKernel.h"
#include "../../../kernels/KernelFactory.h"
#include "../../local_execution/graph/GraphCaptureGuard.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

#include <sstream>
#include <utility>

namespace llaminar2
{
    using KernelFactory = llaminar::v2::kernels::KernelFactory;

    MoEOverlayTicketConsumeStage::MoEOverlayTicketConsumeStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        if (consumesCanonicalRouteTicket())
        {
            moe_kernel_ = KernelFactory::createMoEKernel(params_.device_id);
            PerfStatsCollector::addCounter(
                "moe_overlay_participant_graph",
                "materialized_rank_local_canonical_ticket_consumers",
                1.0,
                "model_setup",
                params_.device_id.toString(),
                {{"bucket_rows", std::to_string(params_.bucket_rows)},
                 {"d_model", std::to_string(params_.d_model)},
                 {"layer", std::to_string(params_.layer_idx)},
                 {"route_capacity",
                  std::to_string(
                      static_cast<size_t>(params_.bucket_rows) *
                      static_cast<size_t>(params_.top_k))},
                 {"ticket_kind", "canonical_route"}});
        }
    }

    MoEOverlayTicketConsumeStage::~MoEOverlayTicketConsumeStage() = default;

    bool MoEOverlayTicketConsumeStage::hasStaticTicketContract() const noexcept
    {
        const bool has_dense_ticket = params_.ticket_storage != nullptr;
        const bool has_canonical_ticket =
            params_.canonical_route_ticket_storage != nullptr;
        if (!params_.device_id.is_gpu() || params_.layer_idx < 0 ||
            params_.bucket_rows <= 0 || params_.d_model <= 0 ||
            !params_.output || params_.output->native_type() != TensorType::FP32 ||
            has_dense_ticket == has_canonical_ticket)
        {
            return false;
        }

        if (has_canonical_ticket)
        {
            const auto &ticket = *params_.canonical_route_ticket_storage;
            const size_t route_capacity =
                static_cast<size_t>(params_.bucket_rows) *
                static_cast<size_t>(params_.top_k);
            return params_.top_k > 0 && moe_kernel_ &&
                   params_.output->numel() >=
                       route_capacity *
                           static_cast<size_t>(params_.d_model) &&
                   ticket.hasValidBoundIdentity() &&
                   ticket.continuationDevice() == params_.device_id &&
                   ticket.layerIndex() == params_.layer_idx &&
                   ticket.routeCapacity() == route_capacity &&
                   ticket.dModel() == params_.d_model &&
                   ticket.controlDeviceAlias() &&
                   ticket.originalRouteSlotsDeviceAlias() &&
                   ticket.compactRouteSlotsDeviceAlias() &&
                   ticket.contributionRowsDeviceAlias();
        }

        if (params_.output->numel() <
                static_cast<size_t>(params_.bucket_rows) *
                    static_cast<size_t>(params_.d_model) ||
            !params_.ticket_storage->hasValidBoundIdentity() ||
            params_.ticket_storage->sourceDevice() != params_.device_id)
        {
            return false;
        }

        const auto &ticket = params_.ticket_storage->ticket();
        return ticket.header &&
               ticket.header->layer_idx == params_.layer_idx &&
               ticket.header->bucket_row_capacity == params_.bucket_rows &&
               ticket.header->d_model == params_.d_model;
    }

    bool MoEOverlayTicketConsumeStage::hasFixedTicketContract() const noexcept
    {
        return hasStaticTicketContract() && params_.output->gpu_data_ptr();
    }

    bool MoEOverlayTicketConsumeStage::execute(IDeviceContext *ctx)
    {
        if (!ctx || ctx->deviceId() != params_.device_id ||
            !params_.device_id.is_gpu())
        {
            LOG_ERROR("[MoEOverlayTicketConsumeStage] Exact GPU device context is required");
            return false;
        }
        void *const stream = requireGPUStream();
        if (!hasFixedTicketContract())
        {
            LOG_ERROR("[MoEOverlayTicketConsumeStage] Invalid fixed-capacity ticket contract: "
                      << graphCaptureReadinessDebugString());
            return false;
        }

        if (consumesCanonicalRouteTicket())
        {
            const auto &storage =
                *params_.canonical_route_ticket_storage;
            if (!isGraphCaptureActive() && !storage.payloadReady())
            {
                LOG_ERROR("[MoEOverlayTicketConsumeStage] Colocated CPU canonical route ticket is not release-published");
                return false;
            }
            MoEOverlayCanonicalRouteTicketConsumeLaunch launch{
                .control = storage.controlDeviceAlias(),
                .original_route_slots =
                    storage.originalRouteSlotsDeviceAlias(),
                .compact_route_slots =
                    storage.compactRouteSlotsDeviceAlias(),
                .compact_preweighted_contributions_fp32 =
                    storage.contributionRowsDeviceAlias(),
                .canonical_route_contributions_fp32 =
                    static_cast<float *>(params_.output->gpu_data_ptr()),
                .route_capacity = storage.routeCapacity(),
                .d_model = params_.d_model,
            };
            if (!moe_kernel_ ||
                !moe_kernel_->consumeMoEOverlayCanonicalRouteTicket(
                    MoEKernelLaunchContext{.stream = stream}, launch))
            {
                LOG_ERROR("[MoEOverlayTicketConsumeStage] Failed to enqueue canonical-route ticket materialization");
                return false;
            }
            gpuExecution().publish(params_.output);
            /* This host-side record is emitted while the immutable kernel node
             * is installed into a native graph (or during an explicit direct
             * diagnostic invocation). Replays execute wholly on device; their
             * transaction count is certified by the graph executor's retained
             * replay evidence rather than a host callback. */
            PerfStatsCollector::addCounter(
                "moe_overlay_participant_graph",
                "rank_local_canonical_ticket_consumer_launches",
                1.0,
                isGraphCaptureActive() ? "graph_capture" : "direct_execution",
                params_.device_id.toString(),
                {{"layer", std::to_string(params_.layer_idx)},
                 {"ticket_kind", "canonical_route"}});
            return true;
        }

        const auto &ticket = params_.ticket_storage->ticket();
        /*
         * Native capture records the fixed-address H2D node; it does not
         * execute that copy. Setup-only materialization therefore has no live
         * request and correctly presents an incomplete return header. The
         * admitted transaction later executes its manual CPU segment before
         * launching this captured consumer. Outside a recording window the
         * readiness check remains mandatory, so direct/manual invocation can
         * never ingest unpublished bytes.
         */
        if (!isGraphCaptureActive() && !ticket.returnPayloadReady())
        {
            LOG_ERROR("[MoEOverlayTicketConsumeStage] CPU return payload is not complete for the published logical prefix");
            return false;
        }

        const size_t bytes =
            static_cast<size_t>(params_.bucket_rows) *
            static_cast<size_t>(params_.d_model) * sizeof(float);
        IBackend *const backend = getBackendFor(params_.device_id);
        if (!backend ||
            !backend->hostToDeviceOnStream(
                params_.output->gpu_data_ptr(),
                ticket.return_rows_fp32,
                bytes,
                params_.device_id.gpu_ordinal(),
                stream))
        {
            LOG_ERROR("[MoEOverlayTicketConsumeStage] Failed to enqueue fixed-capacity return ingress");
            return false;
        }

        /*
         * The backend copy above writes the same persistent arena allocation
         * that downstream shared-expert and residual stages consume.  Record
         * its exact stream/event edge before returning so those stages cannot
         * infer a host-authoritative shadow or substitute a default stream.
         * During capture this contributes to the graph dependency ledger; on
         * eager segmented replay it records the concrete completion event.
         */
        gpuExecution().publish(params_.output);
        return true;
    }

    bool MoEOverlayTicketConsumeStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    bool MoEOverlayTicketConsumeStage::isGraphCapturable() const
    {
        return hasFixedTicketContract();
    }

    bool MoEOverlayTicketConsumeStage::supportsGraphCaptureAfterLaunchPreparation() const
    {
        return hasStaticTicketContract();
    }

    bool MoEOverlayTicketConsumeStage::supportsLazyPrefillGraphCapturePreflight() const
    {
        /* BufferArena publishes the stable output address after cold preflight. */
        return hasStaticTicketContract();
    }

    bool MoEOverlayTicketConsumeStage::supportsPaddedPrefillGraphCapturePreflight() const
    {
        return hasStaticTicketContract();
    }

    bool MoEOverlayTicketConsumeStage::prepareGraphLaunch(
        IDeviceContext *ctx,
        void *stream)
    {
        if (!ctx || ctx->deviceId() != params_.device_id || !stream)
        {
            LOG_ERROR(
                "[MoEOverlayTicketConsumeStage] Capture preparation requires "
                "the exact GPU context and non-null capture stream");
            return false;
        }
        if (!hasStaticTicketContract())
        {
            LOG_ERROR(
                "[MoEOverlayTicketConsumeStage] Capture preparation found an "
                "invalid static ticket contract: "
                << graphCaptureReadinessDebugString());
            return false;
        }

        setGPUStream(stream);
        if (!hasFixedTicketContract())
        {
            LOG_ERROR(
                "[MoEOverlayTicketConsumeStage] Graph executor did not prebind "
                "the fixed arena output before launch preparation: "
                << graphCaptureReadinessDebugString());
            return false;
        }
        return true;
    }

    std::string MoEOverlayTicketConsumeStage::graphCaptureReadinessDebugString() const
    {
        std::ostringstream out;
        out << "device=" << params_.device_id.toString()
            << " layer=" << params_.layer_idx
            << " bucket_rows=" << params_.bucket_rows
            << " top_k=" << params_.top_k
            << " d_model=" << params_.d_model
            << " ticket_kind="
            << (consumesCanonicalRouteTicket() ? "canonical_route" : "dense")
            << " ticket_bound="
            << ((params_.ticket_storage && params_.ticket_storage->isBound()) ||
                        (params_.canonical_route_ticket_storage &&
                         params_.canonical_route_ticket_storage
                             ->hasValidBoundIdentity())
                    ? "true" : "false")
            << " ticket_identity="
            << ((params_.ticket_storage &&
                 params_.ticket_storage->hasValidBoundIdentity()) ||
                        (params_.canonical_route_ticket_storage &&
                         params_.canonical_route_ticket_storage
                             ->hasValidBoundIdentity())
                    ? "valid"
                    : "invalid")
            << " output=" << (params_.output ? "true" : "false")
            << " output_device_storage="
            << (params_.output && params_.output->gpu_data_ptr()
                    ? "true"
                    : "false");
        return out.str();
    }

    StageBufferRequirements
    MoEOverlayTicketConsumeStage::getBufferRequirements() const
    {
        StageBufferRequirements requirements;
        if (params_.output)
        {
            requirements.addOutput(
                "output",
                params_.output->shape(),
                toBufferTensorType(params_.output->native_type()));
        }
        return requirements;
    }

    StageBufferContract MoEOverlayTicketConsumeStage::bufferContract() const
    {
        auto contract = StageBufferContract::build();
        if (params_.output && params_.output_buffer_id)
        {
            if (consumesCanonicalRouteTicket())
                contract.addInOut(*params_.output_buffer_id, "FP32");
            else
                contract.addOutput(*params_.output_buffer_id, "FP32");
        }
        return contract;
    }

    StageDumpInfo MoEOverlayTicketConsumeStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.output)
        {
            const size_t rows =
                consumesCanonicalRouteTicket()
                    ? static_cast<size_t>(params_.bucket_rows) *
                          static_cast<size_t>(params_.top_k)
                    : static_cast<size_t>(params_.bucket_rows);
            info.addOutput(
                "output",
                params_.output,
                rows,
                static_cast<size_t>(params_.d_model));
        }
        info.addScalarInt("layer_idx", params_.layer_idx);
        info.addScalarInt("bucket_rows", params_.bucket_rows);
        info.addScalarInt("top_k", params_.top_k);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarBool(
            "canonical_route_ticket",
            consumesCanonicalRouteTicket());
        return info;
    }

} // namespace llaminar2
