/**
 * @file PipelineActivationExchange.cpp
 * @brief Capture-local native and heterogeneous pipeline activation boundaries.
 *
 * The leader's channel receive precedes its native broadcast on the same
 * stream. Other TP members enqueue only that broadcast, so NCCL/RCCL retains
 * intra-domain P2P and all participants consume their own arena storage.
 * TransferEngine owns channel ordering and tensor publication throughout.
 */
#include "PipelineActivationExchange.h"
#include "memory/StageBufferContract.h"
#include "tensors/TensorClasses.h"
#include <stdexcept>

namespace llaminar2
{
PipelineActivationExchange::PipelineActivationExchange(DeviceId device, TensorBase &hidden,
    size_t elements, NativePeer peer)
    : IComputeStage(device), hidden_(hidden), elements_(elements), transport_(peer),
      membership_(device, &peer.context, peer.participant)
{
    if (hidden.native_type() != TensorType::FP32 || !elements || elements > hidden.numel() ||
        (peer.direction != CollectiveP2POpKind::Send && peer.direction != CollectiveP2POpKind::Recv) ||
        (peer.direction == CollectiveP2POpKind::Send ? peer.participant + 1 >= peer.context.degree() : peer.participant == 0))
        throw std::invalid_argument("Pipeline native activation has invalid geometry or adjacent peer");
}

PipelineActivationExchange::PipelineActivationExchange(DeviceId device, std::shared_ptr<TensorBase> hidden,
    CapturedDomain domain, TransferEngine &transfer)
    : IComputeStage(device), hidden_(*(hidden ? hidden.get() : throw std::invalid_argument("Missing local activation owner"))),
      elements_(domain.message.bytes / sizeof(float)),
      transport_(DomainBinding{domain, std::move(hidden), std::nullopt, &transfer}),
      membership_(device, domain.context, domain.participant)
{
    auto &binding = std::get<DomainBinding>(transport_);
    const auto &port = binding.domain;
    if (!device.is_gpu() || !port.channel || !port.message.fits(port.channel->capacityBytes()) ||
        port.message.bytes % sizeof(float) || hidden_.native_type() != TensorType::FP32 || elements_ > hidden_.numel())
        throw std::invalid_argument("Pipeline domain activation requires an exact FP32 message and local bank");
    if (port.endpoint != CapturedTransferEndpoint::Producer && port.endpoint != CapturedTransferEndpoint::Consumer)
        throw std::invalid_argument("Pipeline domain activation has an invalid endpoint role");
    if (port.endpoint == CapturedTransferEndpoint::Producer && port.participant != 0)
        throw std::invalid_argument("Only the pipeline source domain leader may publish an activation");
    const auto leader = membership_.leader();
    if (port.channel->endpointDevice(port.endpoint) != leader ||
        port.channel->endpointDevice(CapturedTransferEndpoint::Producer).type ==
            port.channel->endpointDevice(CapturedTransferEndpoint::Consumer).type)
        throw std::invalid_argument("Pipeline channel must connect exact cross-backend domain leaders");
    if (port.participant == 0)
        binding.endpoint = transfer.bindCapturedTransfer(port.channel, port.endpoint, port.message, binding.owner);
}

void PipelineActivationExchange::validateMembership() const
{ membership_.validate(); }

bool PipelineActivationExchange::sends() const noexcept
{
    if (const auto *native = std::get_if<NativePeer>(&transport_)) return native->direction == CollectiveP2POpKind::Send;
    return std::get<DomainBinding>(transport_).domain.endpoint == CapturedTransferEndpoint::Producer;
}

bool PipelineActivationExchange::execute(IDeviceContext *context)
{
    if (!context || context->deviceId() != device()) return false;
    void *stream = requireGPUStream();
    validateMembership();
    if (!hidden_.gpu_data_ptr()) return false;
    if (const auto *native = std::get_if<NativePeer>(&transport_))
    {
        const CollectiveP2POp operation{.kind = native->direction,
            .send_buffer = sends() ? hidden_.gpu_data_ptr() : nullptr,
            .recv_buffer = sends() ? nullptr : hidden_.gpu_data_ptr(),
            .count = elements_, .dtype = CollectiveDataType::FLOAT32,
            .peer = native->participant + (sends() ? 1 : -1)};
        return native->context.groupedP2PRawOnStream({operation}, native->participant, stream, name());
    }
    const auto &binding = std::get<DomainBinding>(transport_);
    const auto &port = binding.domain;
    // Only the domain leader touches the inter-vendor slot. Its acquire/copy/
    // publication precedes the native broadcast by exact stream order.
    if (binding.endpoint) binding.transfer->enqueueCapturedTransfer(*binding.endpoint, stream);
    if (!sends() && port.context)
    {
        TransferEngine::requireDeviceOutput(&hidden_, device(), stream);
        if (!port.context->broadcastRawOnStream(hidden_.gpu_data_ptr(), hidden_.gpu_data_ptr(), elements_,
                CollectiveDataType::FLOAT32, 0, port.participant, stream, name())) return false;
        TransferEngine::publishDeviceWrite(&hidden_, device(), stream);
    }
    return true;
}

std::string PipelineActivationExchange::name() const
{ return sends() ? "pipeline_activation_send" : "pipeline_activation_receive"; }

bool PipelineActivationExchange::supportsBackend(ComputeBackendType backend) const
{
    return (device().is_cuda() && backend == ComputeBackendType::GPU_CUDA) ||
        (device().is_rocm() && backend == ComputeBackendType::GPU_ROCM);
}

StageBufferRequirements PipelineActivationExchange::getBufferRequirements() const
{
    StageBufferRequirements result;
    if (sends()) result.addInput("hidden", hidden_.shape(), BufferTensorType::FP32);
    else result.addOutput("hidden", hidden_.shape(), BufferTensorType::FP32);
    return result;
}

StageBufferContract PipelineActivationExchange::bufferContract() const
{
    auto result = StageBufferContract::build();
    if (sends()) result.addInput(BufferId::HIDDEN_STATE); else result.addOutput(BufferId::HIDDEN_STATE);
    return result;
}

StageDumpInfo PipelineActivationExchange::buildDumpInfoImpl() const
{
    StageDumpInfo result;
    if (sends()) result.addInput("hidden", &hidden_, hidden_.rows(), hidden_.cols());
    else result.addOutput("hidden", &hidden_, hidden_.rows(), hidden_.cols());
    return result;
}
}
