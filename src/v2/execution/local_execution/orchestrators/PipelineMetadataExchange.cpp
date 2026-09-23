/**
 * @file PipelineMetadataExchange.cpp
 * @brief Captured terminal-to-follower metadata without a second state authority.
 *
 * Message keys distinguish condition, verifier, commit and ordinary-token
 * traffic even when their byte counts match. Device-owned channel epochs carry
 * replay ordering; the host never inspects or resets them. A receiving TP
 * domain broadcasts the exact banks before local compute/commit consumes them.
 */
#include "PipelineMetadataExchange.h"
#include "execution/local_execution/device/WorkspaceBufferLease.h"
#include "tensors/TensorClasses.h"
#include "memory/StageBufferContract.h"
#include <limits>
#include <set>
#include <stdexcept>

namespace llaminar2
{
namespace
{
/** @brief Reject overlapping semantic fields, except the scalar position/length alias. */
void validateFields(PipelineMetadataLayout layout, std::span<const LocalTPCollectiveSidebandBuffer> fields)
{
    for (size_t i = 0; i < fields.size(); ++i)
    {
        const auto address = reinterpret_cast<uintptr_t>(fields[i].send_buffer);
        if (!address || address % alignof(int32_t))
            throw std::invalid_argument("Pipeline metadata fields require aligned non-null INT32 addresses");
        for (size_t j = 0; j < i; ++j)
        {
            const auto previous = reinterpret_cast<uintptr_t>(fields[j].send_buffer);
            const bool overlap = address >= previous
                ? address - previous < layout.message(j).bytes
                : previous - address < layout.message(i).bytes;
            const bool shared_condition_position = layout.kind() == PipelineMetadataLayout::Kind::Condition &&
                i == 2 && j == 1 && address == previous;
            if (overlap && !shared_condition_position)
                throw std::invalid_argument("Pipeline metadata fields have conflicting physical ownership");
        }
    }
}
}

PipelineMetadataLayout::PipelineMetadataLayout(Kind kind, int rows) : kind_(kind), rows_(rows)
{
    switch (kind)
    {
    case Kind::Condition: case Kind::CommittedState: case Kind::NextToken:
        if (rows == 1) return;
        break;
    case Kind::Verifier:
        if (rows >= 2) return;
        break;
    }
    throw std::invalid_argument("Pipeline metadata has an invalid semantic family or physical row extent");
}

size_t PipelineMetadataLayout::fieldCount() const noexcept
{ return kind_ == Kind::CommittedState ? 4u : kind_ == Kind::NextToken ? 1u : 3u; }

size_t PipelineMetadataLayout::elements(size_t field) const
{
    if (field >= fieldCount()) throw std::out_of_range("Pipeline metadata field is outside its schema");
    return kind_ == Kind::Verifier && field < 2 ? size_t(rows_) : 1u;
}

CapturedTransferMessage PipelineMetadataLayout::message(size_t field) const
{
    // Disjoint bit fields: family [47:40], physical rows [39:8], field [7:0].
    // Equal extents from different graph roles cannot alias a message identity.
    return {0x5050000000000000ull | (uint64_t(kind_) << 40) | (uint64_t(rows_) << 8) | (field + 1),
        elements(field) * sizeof(int32_t)};
}

const char *PipelineMetadataLayout::name() const noexcept
{
    switch (kind_)
    {
    case Kind::Condition: return "pipeline_mtp_condition_rows";
    case Kind::Verifier: return "pipeline_mtp_verifier_rows";
    case Kind::CommittedState: return "pipeline_mtp_committed_state";
    case Kind::NextToken: return "pipeline_next_token";
    }
    return "invalid_pipeline_metadata"; // Construction rejects an unknown kind.
}

void PipelineMetadataBank::validateGeometry(size_t available) const
{
    if (!elements_ || offset_ % sizeof(int32_t) ||
        elements_ > std::numeric_limits<size_t>::max() / sizeof(int32_t) ||
        offset_ > available || elements_ * sizeof(int32_t) > available - offset_)
        throw std::out_of_range("Pipeline metadata bank must be an aligned bounded INT32 region");
}

PipelineMetadataBank::PipelineMetadataBank(std::shared_ptr<TensorBase> owner, BufferId arena_id,
    size_t offset, size_t elements)
    : owner_(owner), offset_(offset), elements_(elements), device_(DeviceId::invalid()), data_(nullptr), arena_id_(arena_id)
{
    if (!owner || owner->native_type() != TensorType::INT32 || !owner->gpu_data_ptr() ||
        !owner->current_device() || !owner->current_device()->is_gpu())
        throw std::invalid_argument("Pipeline metadata requires an already-resident INT32 tensor owner");
    validateGeometry(owner->size_bytes());
    device_ = *owner->current_device();
    data_ = static_cast<unsigned char *>(owner->gpu_data_ptr()) + offset_;
}

PipelineMetadataBank::PipelineMetadataBank(std::shared_ptr<const WorkspaceBufferLease> owner,
    size_t offset, size_t elements)
    : owner_(owner), offset_(offset), elements_(elements), device_(DeviceId::invalid()), data_(nullptr)
{
    if (!owner || !owner->device().is_gpu())
        throw std::invalid_argument("Pipeline metadata requires a retained GPU workspace owner");
    validateGeometry(owner->sizeBytes());
    device_ = owner->device();
    data_ = owner->data(offset_);
    if (reinterpret_cast<uintptr_t>(data_) % alignof(int32_t))
        throw std::invalid_argument("Pipeline metadata workspace region is not INT32 aligned");
}

void *PipelineMetadataBank::data() const
{
    if (const auto *tensor = std::get_if<std::shared_ptr<TensorBase>>(&owner_))
    {
        validateGeometry((*tensor)->size_bytes());
        if ((*tensor)->native_type() != TensorType::INT32 || (*tensor)->current_device() != device_ ||
            !(*tensor)->gpu_data_ptr() || static_cast<unsigned char *>((*tensor)->gpu_data_ptr()) + offset_ != data_)
            throw std::logic_error("Pipeline metadata tensor identity changed after capture binding");
    }
    else
    {
        const auto &workspace = std::get<std::shared_ptr<const WorkspaceBufferLease>>(owner_);
        if (workspace->device() != device_ || !workspace->contains(offset_, elements_ * sizeof(int32_t)) ||
            workspace->data(offset_) != data_)
            throw std::logic_error("Pipeline metadata workspace identity changed after capture binding");
    }
    return data_;
}

CapturedTransferBinding PipelineMetadataBank::bind(TransferEngine &transfer,
    std::shared_ptr<CapturedTransferChannel> channel, CapturedTransferEndpoint endpoint,
    CapturedTransferMessage message) const
{
    (void)data();
    if (message.bytes != elements_ * sizeof(int32_t))
        throw std::invalid_argument("Pipeline metadata message does not cover its exact local bank");
    return std::visit([&](const auto &owner) {
        return transfer.bindCapturedTransfer(std::move(channel), endpoint, message, owner, offset_);
    }, owner_);
}

void PipelineMetadataBank::requireOutput(void *stream) const
{
    if (!stream) throw std::invalid_argument("Pipeline metadata requires an explicit stream");
    (void)data();
    if (const auto *tensor = std::get_if<std::shared_ptr<TensorBase>>(&owner_))
        TransferEngine::requireDeviceOutput(tensor->get(), device_, stream);
}

void PipelineMetadataBank::publishOutput(void *stream) const
{
    if (!stream) throw std::invalid_argument("Pipeline metadata publication requires an explicit stream");
    if (const auto *tensor = std::get_if<std::shared_ptr<TensorBase>>(&owner_))
        TransferEngine::publishDeviceWrite(tensor->get(), device_, stream);
}

PipelineMetadataExchange::PipelineMetadataExchange(DeviceId device, PipelineMetadataLayout layout,
    Native native, std::span<void *const> fields)
    : IComputeStage(device), layout_(layout), transport_(native),
      membership_(device, &native.context, native.participant)
{
    if (fields.size() != layout_.fieldCount())
        throw std::invalid_argument("Pipeline metadata requires every field of its declared schema");
    for (size_t i = 0; i < fields.size(); ++i)
    {
        if (!fields[i]) throw std::invalid_argument("Pipeline metadata has a missing local bank");
        fields_[i] = {.kind = LocalTPCollectiveSidebandKind::Broadcast,
            .send_buffer = fields[i], .recv_buffer = fields[i], .element_count = layout_.elements(i),
            .dtype = CollectiveDataType::INT32, .root_device_index = native.context.degree() - 1,
            .name = std::to_string(i)};
    }
    validateFields(layout_, std::span(fields_).first(layout_.fieldCount()));
}

PipelineMetadataExchange::PipelineMetadataExchange(DeviceId device, PipelineMetadataLayout layout,
    CapturedDomain domain, std::vector<PipelineMetadataBank> banks, TransferEngine &transfer)
    : IComputeStage(device), layout_(layout),
      transport_(DomainBinding{domain, std::move(banks), {}, {}, &transfer}),
      membership_(device, domain.context, domain.participant)
{
    auto &binding = std::get<DomainBinding>(transport_);
    const auto &port = binding.domain;
    if (!device.is_gpu() || (!port.inbound && !port.outbound) ||
        (port.inbound && port.inbound == port.outbound) || binding.banks.size() != layout_.fieldCount())
        throw std::invalid_argument("Pipeline metadata requires a complete schema and directed domain boundary");
    const auto leader = membership_.leader();
    for (const auto &[channel, endpoint] : std::array{
             std::pair{port.inbound, CapturedTransferEndpoint::Consumer},
             std::pair{port.outbound, CapturedTransferEndpoint::Producer}})
    {
        if (!channel) continue;
        if (channel->endpointDevice(endpoint) != leader ||
            channel->endpointDevice(CapturedTransferEndpoint::Producer).type ==
                channel->endpointDevice(CapturedTransferEndpoint::Consumer).type)
            throw std::invalid_argument("Pipeline metadata channel must connect exact cross-backend domain leaders");
        for (size_t i = 0; i < layout_.fieldCount(); ++i)
            if (!layout_.message(i).fits(channel->capacityBytes()))
                throw std::invalid_argument("Pipeline metadata exceeds admitted channel capacity");
    }
    for (size_t i = 0; i < binding.banks.size(); ++i)
    {
        const auto &bank = binding.banks[i];
        if (bank.device() != device || bank.elements() != layout_.elements(i))
            throw std::invalid_argument("Pipeline metadata has a foreign or incorrectly sized bank");
        fields_[i] = {.kind = LocalTPCollectiveSidebandKind::Broadcast,
            .send_buffer = bank.data(), .recv_buffer = bank.data(), .element_count = bank.elements(),
            .dtype = CollectiveDataType::INT32, .root_device_index = 0, .name = std::to_string(i)};
        if (port.participant == 0)
        {
            if (port.inbound) binding.inbound.push_back(bank.bind(transfer, port.inbound,
                CapturedTransferEndpoint::Consumer, layout_.message(i)));
            if (port.outbound) binding.outbound.push_back(bank.bind(transfer, port.outbound,
                CapturedTransferEndpoint::Producer, layout_.message(i)));
        }
    }
    validateFields(layout_, std::span(fields_).first(layout_.fieldCount()));
}

void PipelineMetadataExchange::validateMembership() const
{ membership_.validate(); }

bool PipelineMetadataExchange::execute(IDeviceContext *context)
{
    if (!context || context->deviceId() != device()) return false;
    void *const stream = requireGPUStream();
    validateMembership();
    const auto fields = std::span(fields_).first(layout_.fieldCount());
    if (const auto *native = std::get_if<Native>(&transport_))
        return native->context.collectiveSidebandSpanOnStream(fields, native->participant, stream, layout_.name());

    const auto &binding = std::get<DomainBinding>(transport_);
    for (const auto &bank : binding.banks) (void)bank.data();
    // The slot is reused per field, under device-owned backpressure. No field
    // can overwrite an unconsumed predecessor. Native fanout occurs after the
    // complete schema arrives and before either local use or onward publication.
    for (const auto &receive : binding.inbound) binding.transfer->enqueueCapturedTransfer(receive, stream);
    if (binding.domain.inbound && binding.domain.context)
    {
        for (const auto &bank : binding.banks) bank.requireOutput(stream);
        if (!binding.domain.context->collectiveSidebandSpanOnStream(fields,
                binding.domain.participant, stream, layout_.name())) return false;
        for (const auto &bank : binding.banks) bank.publishOutput(stream);
    }
    for (const auto &send : binding.outbound) binding.transfer->enqueueCapturedTransfer(send, stream);
    return true;
}

bool PipelineMetadataExchange::supportsBackend(ComputeBackendType backend) const
{
    return (device().is_cuda() && backend == ComputeBackendType::GPU_CUDA) ||
        (device().is_rocm() && backend == ComputeBackendType::GPU_ROCM);
}

size_t PipelineMetadataExchange::estimatedMemoryBytes() const
{
    size_t bytes = 0;
    for (size_t i = 0; i < layout_.fieldCount(); ++i) bytes += layout_.message(i).bytes;
    return bytes;
}

StageBufferContract PipelineMetadataExchange::bufferContract() const
{
    StageBufferContract contract;
    if (const auto *binding = std::get_if<DomainBinding>(&transport_))
    {
        std::set<BufferId> seen;
        for (const auto &bank : binding->banks)
        {
            const auto id = bank.arenaId();
            if (!id || !seen.insert(*id).second) continue;
            // A relay reads only after publishing its own received field. The
            // capture ledger recognizes this intra-stage dependency from the
            // declared output, without fabricating an external producer event.
            if (binding->domain.inbound) contract.addPreallocatedOutput(*id, "INT32");
            else contract.addInput(*id, "INT32");
        }
    }
    return contract;
}

StageDumpInfo PipelineMetadataExchange::buildDumpInfoImpl() const
{
    StageDumpInfo info;
    info.addScalarInt("first_field_elements", int(layout_.elements(0)));
    info.addScalarInt("field_count", int(layout_.fieldCount()));
    info.addScalarInt("source_participant", fields_[0].root_device_index);
    return info;
}
}
