/**
 * @file MoEOverlayTicketPublishStage.cpp
 * @brief Captured fixed-capacity GPU publication for heterogeneous MoE rows.
 */

#include "MoEOverlayTicketPublishStage.h"

#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        bool isFP32Capacity(
            const TensorBase *tensor,
            size_t required_elements)
        {
            return tensor && tensor->native_type() == TensorType::FP32 &&
                   tensor->numel() >= required_elements;
        }
    } // namespace

    MoEOverlayTicketPublishStage::MoEOverlayTicketPublishStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool MoEOverlayTicketPublishStage::hasStaticTicketContract() const noexcept
    {
        if (!params_.device_id.is_gpu() || params_.layer_idx < 0 ||
            params_.bucket_rows <= 0 || params_.top_k <= 0 ||
            params_.d_model <= 0 || !params_.ticket_storage ||
            !params_.ticket_storage->isBound() ||
            !params_.ticket_storage->hasValidBoundIdentity() ||
            params_.ticket_storage->sourceDevice() != params_.device_id)
        {
            return false;
        }

        const auto &ticket = params_.ticket_storage->ticket();
        if (!ticket.header ||
            ticket.header->layer_idx != params_.layer_idx ||
            ticket.header->bucket_row_capacity != params_.bucket_rows ||
            ticket.header->top_k != params_.top_k ||
            ticket.header->d_model != params_.d_model)
        {
            return false;
        }

        const size_t route_elements =
            static_cast<size_t>(params_.bucket_rows) *
            static_cast<size_t>(params_.top_k);
        const size_t hidden_elements =
            static_cast<size_t>(params_.bucket_rows) *
            static_cast<size_t>(params_.d_model);
        return isFP32Capacity(params_.routing_indices, route_elements) &&
               isFP32Capacity(params_.routing_weights, route_elements) &&
               isFP32Capacity(params_.hidden, hidden_elements);
    }

    bool MoEOverlayTicketPublishStage::hasFixedTicketContract() const noexcept
    {
        return hasStaticTicketContract() &&
               params_.routing_indices->gpu_data_ptr() &&
               params_.routing_weights->gpu_data_ptr() &&
               params_.hidden->gpu_data_ptr();
    }

    bool MoEOverlayTicketPublishStage::execute(IDeviceContext *ctx)
    {
        if (!ctx || ctx->deviceId() != params_.device_id ||
            !params_.device_id.is_gpu())
        {
            LOG_ERROR("[MoEOverlayTicketPublishStage] Exact GPU device context is required");
            return false;
        }
        void *const stream = requireGPUStream();
        if (!hasFixedTicketContract())
        {
            LOG_ERROR("[MoEOverlayTicketPublishStage] Invalid fixed-capacity ticket contract: "
                      << graphCaptureReadinessDebugString());
            return false;
        }

        auto &ticket = params_.ticket_storage->ticket();
        ticket.header->return_logical_row_count = 0;
        const size_t route_bytes =
            static_cast<size_t>(params_.bucket_rows) *
            static_cast<size_t>(params_.top_k) * sizeof(float);
        const size_t hidden_bytes =
            static_cast<size_t>(params_.bucket_rows) *
            static_cast<size_t>(params_.d_model) * sizeof(float);

        MoEOverlayDispatchTicketStorage::CapturedDevicePayload payload{
            .logical_row_count = params_.active_row_count_device,
            .routing_indices =
                params_.routing_indices->gpu_data_ptr(),
            .routing_weights =
                params_.routing_weights->gpu_data_ptr(),
            .route_bytes = route_bytes,
            .hidden_rows = params_.hidden->gpu_data_ptr(),
            .hidden_bytes = hidden_bytes,
        };
        std::string payload_error;
        if (!params_.ticket_storage->enqueueCapturedPayload(
                payload, stream, &payload_error))
        {
            LOG_ERROR(
                "[MoEOverlayTicketPublishStage] Failed to enqueue mapped "
                "fixed-capacity ticket payload: "
                << payload_error);
            return false;
        }
        std::string publication_error;
        if (!params_.ticket_storage->enqueueCapturedPublication(
                stream, &publication_error))
        {
            LOG_ERROR(
                "[MoEOverlayTicketPublishStage] Failed to enqueue the exact "
                "GPU-to-host ticket publication edge: "
                << publication_error);
            return false;
        }
        return true;
    }

    bool MoEOverlayTicketPublishStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    bool MoEOverlayTicketPublishStage::isGraphCapturable() const
    {
        return hasFixedTicketContract();
    }

    bool MoEOverlayTicketPublishStage::supportsGraphCaptureAfterLaunchPreparation() const
    {
        return hasStaticTicketContract();
    }

    bool MoEOverlayTicketPublishStage::supportsLazyPrefillGraphCapturePreflight() const
    {
        /*
         * Arena publication deliberately follows cold graph preflight. Only
         * immutable capture geometry is knowable here; isGraphCapturable()
         * performs the later exact-address proof after workspace binding.
         */
        return hasStaticTicketContract();
    }

    bool MoEOverlayTicketPublishStage::supportsPaddedPrefillGraphCapturePreflight() const
    {
        return hasStaticTicketContract() &&
               supportsPaddedPrefillRealLengthContract();
    }

    bool MoEOverlayTicketPublishStage::prepareGraphLaunch(
        IDeviceContext *ctx,
        void *stream)
    {
        if (!ctx || ctx->deviceId() != params_.device_id || !stream)
        {
            LOG_ERROR(
                "[MoEOverlayTicketPublishStage] Capture preparation requires "
                "the exact GPU context and non-null capture stream");
            return false;
        }
        if (!hasStaticTicketContract())
        {
            LOG_ERROR(
                "[MoEOverlayTicketPublishStage] Capture preparation found an "
                "invalid static ticket contract: "
                << graphCaptureReadinessDebugString());
            return false;
        }

        setGPUStream(stream);
        if (!hasFixedTicketContract())
        {
            LOG_ERROR(
                "[MoEOverlayTicketPublishStage] Graph executor did not prebind "
                "the fixed arena endpoints before launch preparation: "
                << graphCaptureReadinessDebugString());
            return false;
        }
        std::string publication_error;
        if (!params_.ticket_storage->armCapturedPublication(
                &publication_error))
        {
            LOG_ERROR(
                "[MoEOverlayTicketPublishStage] Could not arm the exact "
                "GPU-to-host ticket publication edge: "
                << publication_error);
            return false;
        }
        return true;
    }

    std::string MoEOverlayTicketPublishStage::graphCaptureReadinessDebugString() const
    {
        std::ostringstream out;
        out << "device=" << params_.device_id.toString()
            << " layer=" << params_.layer_idx
            << " bucket_rows=" << params_.bucket_rows
            << " top_k=" << params_.top_k
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
            << " active_row_count_device="
            << (params_.active_row_count_device ? "true" : "false")
            << " hidden=" << (params_.hidden ? "true" : "false")
            << " routing_indices="
            << (params_.routing_indices ? "true" : "false")
            << " routing_weights="
            << (params_.routing_weights ? "true" : "false")
            << " device_storage="
            << (params_.hidden && params_.routing_indices &&
                        params_.routing_weights &&
                        params_.hidden->gpu_data_ptr() &&
                        params_.routing_indices->gpu_data_ptr() &&
                        params_.routing_weights->gpu_data_ptr()
                    ? "ready"
                    : "unbound");
        return out.str();
    }

    StageBufferRequirements
    MoEOverlayTicketPublishStage::getBufferRequirements() const
    {
        StageBufferRequirements requirements;
        if (params_.hidden)
        {
            requirements.addInput(
                "hidden",
                params_.hidden->shape(),
                toBufferTensorType(params_.hidden->native_type()));
        }
        if (params_.routing_indices)
        {
            requirements.addInput(
                "routing_indices",
                params_.routing_indices->shape(),
                toBufferTensorType(params_.routing_indices->native_type()));
        }
        if (params_.routing_weights)
        {
            requirements.addInput(
                "routing_weights",
                params_.routing_weights->shape(),
                toBufferTensorType(params_.routing_weights->native_type()));
        }
        return requirements;
    }

    StageBufferContract MoEOverlayTicketPublishStage::bufferContract() const
    {
        auto contract = StageBufferContract::build();
        if (params_.hidden && params_.hidden_buffer_id)
            contract.addInput(*params_.hidden_buffer_id, "FP32");
        if (params_.routing_indices && params_.routing_indices_buffer_id)
            contract.addInput(*params_.routing_indices_buffer_id, "FP32");
        if (params_.routing_weights && params_.routing_weights_buffer_id)
            contract.addInput(*params_.routing_weights_buffer_id, "FP32");
        return contract;
    }

    StageDumpInfo MoEOverlayTicketPublishStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.hidden)
        {
            info.addInput(
                "hidden",
                params_.hidden,
                static_cast<size_t>(params_.bucket_rows),
                static_cast<size_t>(params_.d_model));
        }
        if (params_.routing_indices)
        {
            info.addInput(
                "routing_indices",
                params_.routing_indices,
                static_cast<size_t>(params_.bucket_rows),
                static_cast<size_t>(params_.top_k));
        }
        if (params_.routing_weights)
        {
            info.addInput(
                "routing_weights",
                params_.routing_weights,
                static_cast<size_t>(params_.bucket_rows),
                static_cast<size_t>(params_.top_k));
        }
        info.addScalarInt("layer_idx", params_.layer_idx);
        info.addScalarInt("bucket_rows", params_.bucket_rows);
        info.addScalarInt("top_k", params_.top_k);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarBool(
            "device_logical_rows",
            params_.active_row_count_device != nullptr);
        return info;
    }

} // namespace llaminar2
