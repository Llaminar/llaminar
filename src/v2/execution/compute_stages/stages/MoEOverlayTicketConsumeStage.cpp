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
#include "../../local_execution/graph/GraphCaptureGuard.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"

#include <sstream>
#include <utility>

namespace llaminar2
{
    MoEOverlayTicketConsumeStage::MoEOverlayTicketConsumeStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool MoEOverlayTicketConsumeStage::hasStaticTicketContract() const noexcept
    {
        if (!params_.device_id.is_gpu() || params_.layer_idx < 0 ||
            params_.bucket_rows <= 0 || params_.d_model <= 0 ||
            !params_.output || params_.output->native_type() != TensorType::FP32 ||
            params_.output->numel() <
                static_cast<size_t>(params_.bucket_rows) *
                    static_cast<size_t>(params_.d_model) ||
            !params_.ticket_storage ||
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
            << " d_model=" << params_.d_model
            << " ticket_bound="
            << (params_.ticket_storage && params_.ticket_storage->isBound()
                    ? "true"
                    : "false")
            << " ticket_identity="
            << (params_.ticket_storage &&
                        params_.ticket_storage->hasValidBoundIdentity()
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
            contract.addOutput(*params_.output_buffer_id, "FP32");
        return contract;
    }

    StageDumpInfo MoEOverlayTicketConsumeStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.output)
        {
            info.addOutput(
                "output",
                params_.output,
                static_cast<size_t>(params_.bucket_rows),
                static_cast<size_t>(params_.d_model));
        }
        info.addScalarInt("layer_idx", params_.layer_idx);
        info.addScalarInt("bucket_rows", params_.bucket_rows);
        info.addScalarInt("d_model", params_.d_model);
        return info;
    }

} // namespace llaminar2
