/**
 * @file PooledImmutableReplicatedReaderGraph.cpp
 * @brief Participant-local compact result/ACK graph nodes around the original pooled source.
 *
 * No foreign query IDs or CPU source frames are created. Existing captured
 * channel epochs authenticate each message; the result header borrows the
 * source frame's actual interval and each reader checks its own live count.
 */
#include "PooledImmutableReplicatedReaderGraph.h"
#include "PooledImmutableReplicatedReaderContract.h"
#include "kernels/qwen4exp/PooledImmutableReplicatedReaderKernels.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
#include "execution/compute_stages/IComputeStage.h"
#include "execution/compute_stages/stages/qwen4exp/PooledImmutableSourcePhaseStage.h"
#include "execution/qwen4exp/Qwen4ExpPreparedStorage.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <array>
#include <cstddef>
#include <stdexcept>

namespace llaminar2 {
namespace {
namespace q = qwen4exp;
namespace rr = q::replicated_reader;
/** @brief Distinct channel messages; live payload never uses the header's fixed extent. */
using Wire = rr::Wire;
/** @return Stable cross-device message key from original graph shape, never pointers/live values. */
std::uint64_t messageKey(Wire wire, const q::PooledImmutableReplicatedReaderTransport& reader,
    DeviceRowRange rows, std::uint32_t elements)
{
    std::uint64_t value = 0x51464e5245414445ULL;
    const auto mix = [&](std::uint64_t word) { value = (value ^ word) * 1099511628211ULL; };
    if (reader.nativeCollective()) {
        // Same logical operation across exclusive bucket graphs. Every actual
        // stage still seals full shape/count/root identity and Header/receipt.
        return rr::nativeMessageKey(wire, {reader.coordinate(),
            static_cast<std::uint32_t>(reader.nativeResult().nativeStorage()->cohort()->degree()),
            static_cast<std::uint32_t>(rows.physicalRows()), elements}, q::pooled_native::Version);
    } else {
        const auto shape = rows.captureIdentity();
        for (const auto word : {static_cast<std::uint64_t>(wire), std::uint64_t{reader.coordinate()},
                static_cast<std::uint64_t>(shape.count != nullptr), static_cast<std::uint64_t>(shape.count_capacity),
                static_cast<std::uint64_t>(shape.first_physical_row), static_cast<std::uint64_t>(shape.physical_rows),
                static_cast<std::uint64_t>(shape.repeats_per_row), static_cast<std::uint64_t>(shape.schema_version),
                std::uint64_t{elements}, std::uint64_t{q::pooled_native::Version}}) mix(word);
    }
    return value ? value : 1;
}
/** @brief Fold every original immutable row-count producer coordinate into capture identity. */
void mixRows(std::uint64_t& value, DeviceRowRange rows)
{
    const auto identity = rows.captureIdentity();
    for (const auto word : {reinterpret_cast<std::uint64_t>(identity.count),
         static_cast<std::uint64_t>(identity.count_capacity), static_cast<std::uint64_t>(identity.first_physical_row),
         static_cast<std::uint64_t>(identity.physical_rows), static_cast<std::uint64_t>(identity.repeats_per_row),
         static_cast<std::uint64_t>(identity.schema_version)})
        value = (value ^ word) * 1099511628211ULL;
}
/** @return Exact source-versus-reader endpoint for a declared message. */
DeviceId endpoint(const q::PooledImmutableReplicatedReaderTransport& reader, Wire wire,
    CapturedTransferEndpoint side)
{
    const bool source = (wire == Wire::Ack) == (side == CapturedTransferEndpoint::Consumer);
    return source ? reader.originalSource().device() : reader.originalReader().device();
}
/** @return Exact immutable span identity; equal bytes alone never replace the original root. */
bool sameSpan(const q::PooledNativeTensorSpan& a, const q::PooledNativeTensorSpan& b)
{
    return a.root == b.root && a.buffer == b.buffer && a.offset == b.offset && a.bytes == b.bytes;
}
/** @return Complete optional original counted-row owner identity. */
bool sameCount(const std::optional<q::PooledNativeTensorSpan>& a, const std::optional<q::PooledNativeTensorSpan>& b)
{
    return bool(a) == bool(b) && (!a || sameSpan(*a, *b));
}
/** @brief Check the original root/count/control declarations before any recording or launch. */
void requireDeclarations(BufferArena& arena, const q::PooledImmutableDirectedTransport& transport,
    const q::PooledImmutableReplicatedReaderTransport& reader, DeviceId device,
    const DeviceGenerationControlBinding& control, DeviceRowRange rows, const q::PooledNativeTensorSpan& output,
    const std::optional<q::PooledNativeTensorSpan>& count)
{
    transport.requireCurrent();
    reader.requireCurrent();
    control.requireFor(device);
    if (!transport.readerGeometry() || !rows.physicalRows() || rows.physicalRows() % 16 ||
        rows.physicalRows() > transport.readerGeometry()->maximumOriginal() ||
        arena.getTensor(BufferId::MTP_GENERATION_CONTROL) != control.owner() ||
        !output.root || output.root->is_view() || output.root->transferStorageOwner() != output.root.get() ||
        output.root->native_type() != TensorType::FP32 || arena.getTensor(output.buffer) != output.root.get() ||
        output.offset % sizeof(float) || output.offset > output.root->size_bytes() ||
        output.bytes > output.root->size_bytes() - output.offset ||
        output.bytes != static_cast<std::uint64_t>(rows.physicalRows()) *
            transport.readerGeometry()->rowElements() * sizeof(float))
        throw std::invalid_argument("Shared PLE reader lost original arena/control/count/output geometry");
    if (bool(count) != bool(rows.countOwner()))
        throw std::invalid_argument("Shared PLE reader lost its original counted-row owner");
    if (count) {
        if (!count->root || count->root->is_view() || count->root->transferStorageOwner() != count->root.get() ||
            count->root->native_type() != TensorType::INT32 || count->root->current_device() != device ||
            arena.getTensor(count->buffer) != count->root.get() ||
            count->bytes != sizeof(std::int32_t) || count->offset % alignof(std::int32_t) ||
            count->offset > count->root->size_bytes() || count->bytes > count->root->size_bytes() - count->offset ||
            reinterpret_cast<const std::byte*>(rows.countOwner()) !=
                static_cast<const std::byte*>(count->root->gpu_data_ptr()) + count->offset)
            throw std::invalid_argument("Shared PLE reader replaced its original counted-row root/offset");
    }
}
class ReaderTransfer;
/** @brief One integer-only native metadata phase; it never performs a transfer or owns a source channel. */
class ReaderPhase final : public IComputeStage {
public:
    /** @brief Bind one original native participant and its canonical request/count/output roots. */
    ReaderPhase(BufferArena& arena, std::shared_ptr<const q::PooledImmutableDirectedTransport> transport,
        std::shared_ptr<const q::PooledImmutableReplicatedReaderTransport> reader,
        DeviceGenerationControlBinding control, DeviceRowRange rows, q::PooledNativeTensorSpan output,
        rr::Action action, bool source, const ReaderTransfer* payload = nullptr,
        std::optional<q::PooledNativeTensorSpan> count = {})
        : IComputeStage(source ? reader->originalSource().device() : reader->originalReader().device()),
          arena_(&arena), transport_(std::move(transport)), reader_(std::move(reader)), control_(control),
          rows_(rows), output_(std::move(output)), count_(std::move(count)), action_(action), source_(source), payload_(payload),
          prepared_(device(), std::array<TensorBase*,4>{output_.root.get(), control_.owner(),
              count_ ? count_->root.get() : output_.root.get(),
              source_ ? sourceFrame() : output_.root.get()})
    {
        requireDeclarations(*arena_, *transport_, *reader_, device(), control_, rows_, output_, count_);
        const bool sourceAction = action_ == rr::Action::PrepareResultHeader ||
            action_ == rr::Action::PrepareResult || action_ == rr::Action::ConfirmAck;
        if (sourceAction != source_ || (action_ == rr::Action::AcknowledgeResult ||
            (action_ == rr::Action::ConfirmAck && reader_->nativeCollective())) != bool(payload_))
            throw std::invalid_argument("Shared PLE reader phase changed its actual source/reader role");
        identity_ = graphCaptureVariantSignature();
    }
    /** @brief Seal the original ordinary output and original worker/stream before native capture. */
    bool prepareGraphLaunch(IDeviceContext* context, void* stream) override
    {
        if (!context || context->deviceId() != device() || !stream || reinterpret_cast<std::uintptr_t>(stream) <= 2)
            throw std::invalid_argument("Shared PLE reader metadata requires its exact participant/stream");
        requireDeclarations(*arena_, *transport_, *reader_, device(), control_, rows_, output_, count_);
        prepared_.seal();
        setGPUStream(stream);
        return true;
    }
    /** @brief Publish metadata and first error only through the original device controller and exact stream. */
    bool execute(IDeviceContext* context) override
    {
        if (!context || context->deviceId() != device())
            throw std::invalid_argument("Shared PLE reader metadata changed participant");
        requireDeclarations(*arena_, *transport_, *reader_, device(), control_, rows_, output_, count_);
        prepared_.validate();
        if (identity_ != graphCaptureVariantSignature())
            throw std::logic_error("Shared PLE reader metadata capture identity changed");
        auto execution = gpuExecution();
        execution.requirePreparedInput(control_.owner());
        if (count_) execution.requirePreparedInput(count_->root.get());
        execution.requirePreparedOutput(control_.owner());
        const auto side = source_ ? CapturedTransferEndpoint::Producer : CapturedTransferEndpoint::Consumer;
        auto* bank = static_cast<rr::Control*>(reader_->control(side)->mutableDeviceData());
        const auto& geometry = *transport_->readerGeometry();
        const q::adaptive_lookup::pooled::LookupFrame* frame = nullptr;
        if (source_) {
            const auto& coordinator = transport_->sourceBoundary()->pooledStorage()->coordinator();
            execution.requirePreparedInput(sourceFrame());
            frame = reinterpret_cast<const q::adaptive_lookup::pooled::LookupFrame*>(
                static_cast<const std::byte*>(coordinator->roots().frame->gpu_data_ptr()) +
                coordinator->plan().parameters().frame.control);
        }
        rr::NativeArgs args{device(), {reader_->coordinate(), static_cast<std::uint32_t>(geometry.consumers().size()),
                static_cast<std::uint32_t>(rows_.physicalRows()), geometry.rowElements()},
            q::pooled_native::Version, transport_->sourceBoundary()->pooledStorage()->coordinator()->plan().parameters().table, rows_, frame, &bank->incoming, &bank->outgoing,
            action_ == rr::Action::AcknowledgeResult ? &bank->payload_bytes : &bank->header_bytes,
            &bank->output_bytes, control_.healthy(), control_.error(),
            action_ == rr::Action::AcknowledgeResult && !reader_->nativeCollective() ? payloadCompletion(execution.nativeStream())
                : CapturedTransferReceiveCompletion::notRequired(),
            (action_ == rr::Action::AcknowledgeResult || action_ == rr::Action::ConfirmAck) && reader_->nativeCollective() ?
                nativePayloadCompletion() : NativeReceiveCompletionView::notRequired(),
            (action_ == rr::Action::AcceptResultHeader || action_ == rr::Action::ConfirmAck) && reader_->nativeCollective() ?
                nativeFixedHeaderAuthority() : ncclLlaminarCompletionAuthority{},
            reader_->nativeCollective() ? messageKey(Wire::Payload, *reader_, rows_, geometry.rowElements()) : 0};
        if (device().is_cuda()) {
#if defined(HAVE_CUDA)
            rr::enqueueCUDAPooledReplicatedReader(action_, args, execution.nativeStream());
#else
            throw std::runtime_error("CUDA shared PLE reader kernels were not compiled");
#endif
        } else {
#if defined(HAVE_ROCM)
            rr::enqueueROCmPooledReplicatedReader(action_, args, execution.nativeStream());
#else
            throw std::runtime_error("ROCm shared PLE reader kernels were not compiled");
#endif
        }
        execution.publish(control_.owner());
        if (action_ == rr::Action::AcknowledgeResult) execution.publish(output_.root.get());
        return true;
    }
    /** @return Actual output validation is the reader's original semantic PLE gather. */
    ComputeStageType type() const override
    { return action_ == rr::Action::AcknowledgeResult ? ComputeStageType::QWEN4EXP_NGRAM_LOOKUP : ComputeStageType::QWEN4EXP_POOLED_SOURCE_PHASE; }
    std::string name() const override { return "PooledImmutableReplicatedReaderPhase"; }
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override { return GraphLaunchPreparationPolicy::CaptureOnly; }
    bool supportsPaddedPrefillRealLengthContract() const override { return true; }
    bool supportsBackend(ComputeBackendType backend) const override
    { return device().is_cuda() ? backend == ComputeBackendType::GPU_CUDA : backend == ComputeBackendType::GPU_ROCM; }
    StageBufferRequirements getBufferRequirements() const override { return {}; }
    /** @return Full original physical/control/count/shape identity; no mutable observations enter it. */
    std::uint64_t graphCaptureVariantSignature() const override
    {
        prepared_.validateIfSealed();
        std::uint64_t value = reader_->nativeCollective() ?
            reinterpret_cast<std::uintptr_t>(reader_->nativeResult().nativeStorage().get()) : reader_->result()->identity().nonce;
        const auto mix = [&](std::uint64_t word) { value = (value ^ word) * 1099511628211ULL; };
        if (!reader_->nativeCollective()) mix(reader_->acknowledgement()->identity().nonce);
        mix(reader_->nativeCollective());
        if (reader_->nativeCollective()) {
            const auto& original = reader_->nativeResult().nativeStorage();
            const auto authority = original->authority(device()).binding();
            mix(reinterpret_cast<std::uintptr_t>(authority.abortFlag));
            mix(reinterpret_cast<std::uintptr_t>(authority.deviceCommunicator));
        }
        mix(reinterpret_cast<std::uintptr_t>(transport_.get()));
        mix(reinterpret_cast<std::uintptr_t>(output_.root.get()));
        mix(output_.offset); mix(output_.bytes); mix(static_cast<std::uint64_t>(output_.buffer));
        if(source_)mix(reinterpret_cast<std::uintptr_t>(sourceFrame()));
        mix(control_.captureIdentity()); mixRows(value, rows_);
        if (count_) { mix(reinterpret_cast<std::uintptr_t>(count_->root.get()));
            mix(count_->offset); mix(count_->bytes); mix(static_cast<std::uint64_t>(count_->buffer)); } mix(static_cast<std::uint64_t>(action_));
        mix(reinterpret_cast<std::uintptr_t>(payload_));
        return value ? value : 1;
    }
    /** @return Exact original ACK validator, used by the retirement owner instead of trusting names. */
    bool validates(const q::PooledImmutableReplicatedReaderTransport* original,
        const q::PooledImmutableDirectedTransport* transport, const DeviceGenerationControlBinding& control,
        DeviceRowRange rows, const q::PooledNativeTensorSpan& output,
        const std::optional<q::PooledNativeTensorSpan>& count, rr::Action action, bool source = true) const
    {
        prepared_.validateIfSealed();
        return reader_.get() == original && transport_.get() == transport && source_ == source && action_ == action &&
            control_.captureIdentity() == control.captureIdentity() && rows_.captureIdentity() == rows.captureIdentity() &&
            sameSpan(output_, output) && sameCount(count_, count);
    }
    /** @return Immutable actual consumer declarations, without claiming execution or native completion. */
    std::optional<PooledImmutableReplicatedConsumerCompletion> consumerDeclaration(
        const q::PooledImmutableDirectedTransport* transport, DeviceId device) const
    {
        if (source_ || action_ != rr::Action::AcknowledgeResult || transport_.get() != transport ||
            reader_->originalReader().device() != device) return {};
        return PooledImmutableReplicatedConsumerCompletion{transport_, reader_, control_, rows_, output_, count_,
            {}, {}, {}, {}, {}};
    }
    /** @return Actual received payload endpoint whose whole-launch result authorizes this ACK. */
    const ReaderTransfer* originalPayload() const noexcept { return payload_; }
    /** @return Original controller, source frame, live count and gather-result dependencies. */
    StageBufferContract bufferContract() const override
    {
        return q::detail::replicatedReaderPhaseContract(action_,output_.buffer,
            count_ ? std::optional(count_->buffer) : std::nullopt);
    }
protected:
    /** @return Standard PLE output snapshot, preserving the ordinary exact-row bank contract. */
    StageDumpInfo buildDumpInfoImpl() const override
    {
        StageDumpInfo dump;
        dump.addScalarInt("replicated_reader_phase", static_cast<unsigned>(action_));
        if (action_ == rr::Action::AcknowledgeResult) {
            const auto* base = static_cast<const std::byte*>(output_.root->raw_data());
            const auto columns = output_.root->cols();
            dump.outputs.push_back({"gathered_rows", base ? base + output_.offset : nullptr,
                output_.bytes / (columns * sizeof(float)), columns, "FP32", sizeof(float),
                output_.bytes, output_.root.get(), {}, output_.offset});
        }
        return dump;
    }
private:
    /** @return Exact original frame read by every source metadata action, never a reader-owned alias. */
    TensorBase* sourceFrame() const {
        return q::detail::checkedReplicatedReaderSourceFrame(*arena_,
            transport_->sourceBoundary()->pooledStorage()->coordinator()->roots().frame.get());
    }
    BufferArena* arena_;
    std::shared_ptr<const q::PooledImmutableDirectedTransport> transport_;
    std::shared_ptr<const q::PooledImmutableReplicatedReaderTransport> reader_;
    DeviceGenerationControlBinding control_;
    DeviceRowRange rows_;
    q::PooledNativeTensorSpan output_;
    std::optional<q::PooledNativeTensorSpan> count_;
    rr::Action action_;
    bool source_;
    const ReaderTransfer* payload_;
    /** @return Read-only original payload endpoint receipt; no GPU state is read by this host method. */
    CapturedTransferReceiveCompletion payloadCompletion(void* stream) const;
    /** @return Actual local payload receipt after its explicit whole-launch graph dependency. */
    NativeReceiveCompletionView nativePayloadCompletion() const;
    /** @return Original fixed native authority; never infers delivery from a stale byte word. */
    ncclLlaminarCompletionAuthority nativeFixedHeaderAuthority() const
    {
        const auto& original = reader_->nativeResult().nativeStorage();
        if (!original) throw std::logic_error("Native Header lost its original communicator metadata owner");
        return original->authority(device()).binding();
    }
    Qwen4ExpPreparedStorage prepared_;
    std::uint64_t identity_;
};
/** @brief One explicit native endpoint over an original source/reader channel. */
class ReaderTransfer final : public IComputeStage {
public:
    /** @brief Retain the exact original root and channel role before capture. */
    ReaderTransfer(BufferArena& arena, std::shared_ptr<const q::PooledImmutableDirectedTransport> transport,
        std::shared_ptr<const q::PooledImmutableReplicatedReaderTransport> reader,
        DeviceGenerationControlBinding control, DeviceRowRange rows, q::PooledNativeTensorSpan output,
        Wire wire, CapturedTransferEndpoint side, std::optional<q::PooledNativeTensorSpan> count = {})
        : IComputeStage(endpoint(*reader, wire, side)), arena_(&arena), transport_(std::move(transport)),
          reader_(std::move(reader)), control_(control), rows_(rows), output_(std::move(output)), count_(std::move(count)),
          wire_(wire), side_(side), prepared_(device(), std::array<TensorBase*,3>{output_.root.get(), control_.owner(),
              count_ ? count_->root.get() : output_.root.get()})
    {
        requireDeclarations(*arena_, *transport_, *reader_, device(), control_, rows_, output_, count_);
        identity_ = graphCaptureVariantSignature();
    }
    /** @brief Resolve only ordinary prepared output storage and original counted metadata, once. */
    bool prepareGraphLaunch(IDeviceContext* context, void* stream) override
    {
        if (!context || context->deviceId() != device() || !stream || reinterpret_cast<std::uintptr_t>(stream) <= 2)
            throw std::invalid_argument("Shared PLE transfer requires its exact participant/stream");
        requireDeclarations(*arena_, *transport_, *reader_, device(), control_, rows_, output_, count_);
        prepared_.seal();
        if (!binding_) {
            const bool source = device() == reader_->originalSource().device();
            const auto& bank = reader_->control(source ? CapturedTransferEndpoint::Producer : CapturedTransferEndpoint::Consumer);
            const auto& channel = wire_ == Wire::Ack ? reader_->acknowledgement() : reader_->result();
            const auto key = messageKey(wire_, *reader_, rows_, transport_->readerGeometry()->rowElements());
            auto& transfer = TransferEngine::instance();
            if (reader_->nativeCollective()) {
                const auto& channel = wire_ == Wire::Ack ? reader_->nativeAcknowledgement() : reader_->nativeResult();
                if (wire_ == Wire::Payload) {
                    auto fixed = channel.bind(side_, {key, output_.bytes}, output_.root, output_.offset);
                    const auto intervalOffset = (source ? offsetof(rr::Control, outgoing) : offsetof(rr::Control, incoming)) +
                        offsetof(rr::Header, interval);
                    auto errorOwner = arena_->getSharedTensor(BufferId::MTP_GENERATION_CONTROL);
                    const auto base = reinterpret_cast<std::uintptr_t>(errorOwner->gpu_data_ptr());
                    const auto error = reinterpret_cast<std::uintptr_t>(control_.error());
                    if (error < base) throw std::logic_error("Native reader error changed its original controller root");
                    q::PooledNativeCompletionFrame frame{
                        q::PooledNativeTransferSpan(bank, intervalOffset, sizeof(q::adaptive_lookup::pooled::Interval)),
                        q::PooledNativeTransferSpan(errorOwner, error - base, sizeof(std::int32_t)),
                        static_cast<std::int32_t>(sampling_math::DeviceGenerationError::ImmutableStorageFailure)};
                    binding_.emplace(fixed.withNativeCounted(q::PooledNativeTransferSpan(bank,
                        source ? offsetof(rr::Control, output_bytes) : offsetof(rr::Control, payload_bytes),
                        sizeof(std::uint64_t)), q::PooledNativeCountedMessage::ReaderResult, std::move(frame)));
                } else {
                    binding_.emplace(channel.bind(side_, {key, sizeof(rr::Header)}, bank,
                        side_ == CapturedTransferEndpoint::Producer ? offsetof(rr::Control, outgoing) : offsetof(rr::Control, incoming)));
                }
            } else if (wire_ == Wire::Payload) {
                binding_.emplace(transfer.bindDeviceCountedTransfer(transfer.bindCapturedTransfer(channel,
                    side_, {key, output_.bytes}, std::static_pointer_cast<ITensor>(output_.root), output_.offset),
                    bank, source ? offsetof(rr::Control, output_bytes) : offsetof(rr::Control, payload_bytes)));
            } else {
                auto fixed = transfer.bindCapturedTransfer(channel, side_, {key, sizeof(rr::Header)}, bank,
                    side_ == CapturedTransferEndpoint::Producer ? offsetof(rr::Control, outgoing) : offsetof(rr::Control, incoming));
                if (side_ == CapturedTransferEndpoint::Consumer)
                    fixed = transfer.bindDeviceCountedTransfer(std::move(fixed), bank, offsetof(rr::Control, header_bytes));
                binding_.emplace(std::move(fixed));
            }
        }
        binding_->prepare(stream);
        setGPUStream(stream);
        return true;
    }
    /** @brief Transfer only the published live prefix; terminal cancellation drains the original endpoint. */
    bool execute(IDeviceContext* context) override
    {
        if (!context || context->deviceId() != device() || !binding_)
            throw std::logic_error("Shared PLE endpoint has no exact prepared participant binding");
        requireDeclarations(*arena_, *transport_, *reader_, device(), control_, rows_, output_, count_);
        prepared_.validate();
        if (identity_ != graphCaptureVariantSignature())
            throw std::logic_error("Shared PLE endpoint capture identity changed");
        auto execution = gpuExecution();
        if (wire_ == Wire::Payload) {
            if (side_ == CapturedTransferEndpoint::Producer) execution.requirePreparedInput(output_.root.get());
            else execution.requirePreparedOutput(output_.root.get());
        }
        binding_->enqueue(execution.nativeStream());
        if (wire_ == Wire::Payload && side_ == CapturedTransferEndpoint::Consumer)
            execution.publish(output_.root.get());
        return true;
    }
    ComputeStageType type() const override { return ComputeStageType::QWEN4EXP_POOLED_DIRECTED_TRANSFER; }
    /** @return An actual native endpoint, independent of NCCL/RCCL collective capture eligibility. */
    CollectiveGraphCaptureAuthority collectiveGraphCaptureAuthority() const noexcept override
    { return reader_->nativeCollective() ? CollectiveGraphCaptureAuthority::RuntimeBackend :
        CollectiveGraphCaptureAuthority::ParticipantLocalCapturedTransfer; }
    std::string name() const override { return "PooledImmutableReplicatedReaderTransfer"; }
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override { return GraphLaunchPreparationPolicy::CaptureOnly; }
    bool supportsPaddedPrefillRealLengthContract() const override { return true; }
    bool supportsBackend(ComputeBackendType backend) const override
    { return device().is_cuda() ? backend == ComputeBackendType::GPU_CUDA : backend == ComputeBackendType::GPU_ROCM; }
    StageBufferRequirements getBufferRequirements() const override { return {}; }
    /** @return Complete original channel/arena/control/count/role identity, with retained physical seals. */
    std::uint64_t graphCaptureVariantSignature() const override
    {
        prepared_.validateIfSealed();
        std::uint64_t value = reader_->nativeCollective() ?
            reinterpret_cast<std::uintptr_t>(reader_->nativeResult().nativeStorage().get()) : reader_->result()->identity().nonce;
        const auto mix = [&](std::uint64_t word) { value = (value ^ word) * 1099511628211ULL; };
        if (!reader_->nativeCollective()) mix(reader_->acknowledgement()->identity().nonce);
        mix(reader_->nativeCollective());
        if (reader_->nativeCollective()) {
            const auto& original = reader_->nativeResult().nativeStorage();
            const auto authority = original->authority(device()).binding();
            mix(reinterpret_cast<std::uintptr_t>(authority.abortFlag));
            mix(reinterpret_cast<std::uintptr_t>(authority.deviceCommunicator));
        }
        mix(reinterpret_cast<std::uintptr_t>(output_.root.get()));
        mix(output_.offset); mix(output_.bytes); mix(static_cast<std::uint64_t>(output_.buffer));
        mix(control_.captureIdentity()); mixRows(value, rows_);
        if (count_) { mix(reinterpret_cast<std::uintptr_t>(count_->root.get()));
            mix(count_->offset); mix(count_->bytes); mix(static_cast<std::uint64_t>(count_->buffer)); }
        mix(static_cast<std::uint64_t>(wire_)); mix(static_cast<std::uint64_t>(side_));
        return value ? value : 1;
    }
    /** @return Original consumer receipt addresses after ordinary preparation, without enqueuing or reading state. */
    CapturedTransferReceiveCompletion completion(void* stream) const
    {
        if (wire_ != Wire::Payload || side_ != CapturedTransferEndpoint::Consumer || !binding_)
            throw std::logic_error("Shared PLE ACK requires its original prepared payload consumer");
        requireDeclarations(*arena_, *transport_, *reader_, device(), control_, rows_, output_, count_);
        prepared_.validate();
        return binding_->mappedCompletion(stream);
    }
    /** @return Same original native local receipt for an explicit dependent ACK/confirm phase. */
    NativeReceiveCompletionView nativeCompletion() const
    {
        if (wire_ != Wire::Payload || !binding_ || !reader_->nativeCollective())
            throw std::logic_error("Native PLE ACK requires its original prepared payload endpoint");
        prepared_.validate();
        return binding_->nativeCompletion();
    }
    /** @return Actual original endpoint identity used by cold retirement validation. */
    /** @return This actual original pure-reader first Header receive; no source publication is implied. */
    bool isDemandHeader(const RetainedGraphParticipantDefinitionDomain& domain)const {
        if(wire_!=Wire::Header || side_!=CapturedTransferEndpoint::Consumer || !transport_ ||
           !transport_->readerGeometry() || !reader_ || device()!=reader_->originalReader().device())return false;
        if(domain!=RetainedGraphParticipantDefinitionDomain{transport_->readerGeometry()->coordinator().device,transport_.get()})return false;
        transport_->requireCurrent();reader_->requireCurrent();return true;
    }
    bool matches(const q::PooledImmutableReplicatedReaderTransport* original,
        const q::PooledImmutableDirectedTransport* transport, Wire wire, CapturedTransferEndpoint side,
        const DeviceGenerationControlBinding& control, DeviceRowRange rows,
        const q::PooledNativeTensorSpan& output, const std::optional<q::PooledNativeTensorSpan>& count) const
    {
        prepared_.validateIfSealed();
        return reader_.get() == original && transport_.get() == transport && wire_ == wire && side_ == side &&
            control_.captureIdentity() == control.captureIdentity() && rows_.captureIdentity() == rows.captureIdentity() &&
            sameSpan(output_, output) && sameCount(count_, count);
    }
    /** @return Only payload endpoints touch the original graph output. */
    StageBufferContract bufferContract() const override
    {
        auto contract = StageBufferContract::build();
        if (wire_ == Wire::Payload) {
            if (side_ == CapturedTransferEndpoint::Producer) contract.addInput(output_.buffer, "FP32");
            else contract.addOutput(output_.buffer, "FP32");
        }
        if (count_) contract.addInput(count_->buffer, "INT32");
        return contract;
    }
protected:
    /** @return Exact static wire identity; device live extents remain on device. */
    StageDumpInfo buildDumpInfoImpl() const override
    {
        StageDumpInfo dump;
        dump.addScalarInt("replicated_reader_wire", static_cast<unsigned>(wire_));
        dump.addScalarInt("replicated_reader_endpoint", static_cast<unsigned>(side_));
        return dump;
    }
private:
    BufferArena* arena_;
    std::shared_ptr<const q::PooledImmutableDirectedTransport> transport_;
    std::shared_ptr<const q::PooledImmutableReplicatedReaderTransport> reader_;
    DeviceGenerationControlBinding control_;
    DeviceRowRange rows_;
    q::PooledNativeTensorSpan output_;
    std::optional<q::PooledNativeTensorSpan> count_;
    Wire wire_;
    CapturedTransferEndpoint side_;
    Qwen4ExpPreparedStorage prepared_;
    std::optional<q::PooledImmutableTransferBinding> binding_;
    std::uint64_t identity_;
};
CapturedTransferReceiveCompletion ReaderPhase::payloadCompletion(void* stream) const
{
    if (!payload_ || !payload_->matches(reader_.get(), transport_.get(), Wire::Payload, CapturedTransferEndpoint::Consumer,
            control_, rows_, output_, count_))
        throw std::logic_error("Shared PLE ACK lost its original payload completion owner");
    return payload_->completion(stream);
}
NativeReceiveCompletionView ReaderPhase::nativePayloadCompletion() const
{
    const auto side = source_ ? CapturedTransferEndpoint::Producer : CapturedTransferEndpoint::Consumer;
    if (!payload_ || !reader_->nativeCollective() ||
        !payload_->matches(reader_.get(), transport_.get(), Wire::Payload, side, control_, rows_, output_, count_))
        throw std::logic_error("Native PLE ACK/confirm lost its exact original local payload endpoint");
    return payload_->nativeCompletion();
}
/** @brief Reject partial construction/name collisions before merging participant-local nodes. */
void publish(ComputeGraph& graph, ComputeGraph fragment)
{
    for (const auto& name : fragment.getExecutionOrder())
        if (graph.getNode(name)) throw std::invalid_argument("Shared PLE reader stage collides: " + name);
    graph.merge(std::move(fragment));
}
/** @brief Require an actual graph edge, not an unrelated phase name. */
void requireDependency(const ComputeGraph& graph, const std::string& consumer, const std::string& producer)
{
    const auto* node = graph.getNode(consumer);
    if (!node || std::find(node->dependencies.begin(), node->dependencies.end(), producer) == node->dependencies.end())
        throw std::invalid_argument("Shared PLE reader lost exact dependency: " + producer + " -> " + consumer);
}
} // namespace

std::shared_ptr<const PooledImmutableReplicatedReaderCompletion> appendPooledReplicatedSourceReaders(
    ComputeGraph& graph, BufferArena& arena, std::shared_ptr<const q::PooledImmutableDirectedTransport> transport,
    DeviceGenerationControlBinding control, DeviceRowRange rows, const q::PooledNativeTensorSpan& output,
    const std::string& readOutput, const std::string& readersRetired, const std::string& prefix,
    std::optional<q::PooledNativeTensorSpan> count)
{
    if (isGraphCaptureActive() || !transport || !transport->readerGeometry() || transport->readers().empty() ||
        !graph.getNode(readOutput) || !graph.getNode(readersRetired) || prefix.empty())
        throw std::invalid_argument("Shared PLE source readers require the original cold read/retirement graph");
    auto completion = std::make_shared<PooledImmutableReplicatedReaderCompletion>(
        PooledImmutableReplicatedReaderCompletion{transport, control, rows, output, count, {}, readOutput});
    ComputeGraph fragment;
    for (const auto& reader : transport->readers()) {
        const auto stem = prefix + "/reader" + std::to_string(reader->coordinate());
        PooledImmutableReplicatedReaderCompletion::Reader nodes{reader, stem + "/header", stem + "/header_send",
            stem + "/extent", stem + "/payload_send", stem + "/ack_receive", stem + "/ack_validated"};
        const auto phase = [&](const std::string& name, rr::Action action, const ReaderTransfer* payload = nullptr) {
            fragment.addNode(name, std::make_unique<ReaderPhase>(arena, transport, reader, control, rows, output, action, true, payload, count),
                reader->originalSource().device());
        };
        const auto wire = [&](const std::string& name, Wire role, CapturedTransferEndpoint side) {
            auto transfer = std::make_unique<ReaderTransfer>(arena, transport, reader, control, rows, output, role, side, count);
            const auto* original = transfer.get();
            fragment.addNode(name, std::move(transfer), reader->originalSource().device());
            return original;
        };
        phase(nodes.header, rr::Action::PrepareResultHeader);
        wire(nodes.headerSent, Wire::Header, CapturedTransferEndpoint::Producer);
        phase(nodes.extent, rr::Action::PrepareResult);
        const auto* payload = wire(nodes.payloadSent, Wire::Payload, CapturedTransferEndpoint::Producer);
        wire(nodes.ackReceived, Wire::Ack, CapturedTransferEndpoint::Consumer);
        phase(nodes.ackValidated, rr::Action::ConfirmAck, reader->nativeCollective() ? payload : nullptr);
        fragment.addDependency(nodes.headerSent, nodes.header);
        fragment.addDependency(nodes.extent, nodes.headerSent);
        fragment.addDependency(nodes.payloadSent, nodes.extent);
        fragment.addDependency(nodes.ackReceived, nodes.payloadSent);
        fragment.addDependency(nodes.ackValidated, nodes.ackReceived);
        completion->readers.push_back(std::move(nodes));
    }
    publish(graph, std::move(fragment));
    for (const auto& reader : completion->readers) {
        graph.addDependency(reader.header, readOutput);
        graph.addDependency(readersRetired, reader.ackValidated);
    }
    requirePooledReplicatedReaderCompletion(graph, *completion, readersRetired);
    return completion;
}
PooledImmutableLookupGraphEndpoints appendPooledReplicatedReader(ComputeGraph& graph, BufferArena& arena,
    std::shared_ptr<const q::PooledImmutableDirectedTransport> transport, DeviceGenerationControlBinding control,
    DeviceRowRange rows, const q::PooledNativeTensorSpan& output, const std::string& prefix,
    const std::string& outputIdentity, std::optional<q::PooledNativeTensorSpan> count,
    PooledImmutableReplicatedConsumerCompletion* completion)
{
    if (isGraphCaptureActive() || !transport || !transport->readerGeometry() || prefix.empty() || outputIdentity.empty())
        throw std::invalid_argument("Shared PLE reader requires its original cold source/cohort declaration");
    const auto found = std::find_if(transport->readers().begin(), transport->readers().end(), [&](const auto& reader) {
        return reader->originalReader().device() == control.owner()->current_device();
    });
    if (found == transport->readers().end())
        throw std::invalid_argument("Shared PLE reader is absent from its original compute cohort");
    const auto reader = *found;
    graph.setRetainedParticipantDefinitionDomain({transport->readerGeometry()->coordinator().device,transport.get()});
    const auto device = reader->originalReader().device();
    const auto header = prefix + "/header_receive", accepted = prefix + "/header_accepted";
    const auto payload = prefix + "/payload_receive", sent = prefix + "/ack_send";
    ComputeGraph fragment;
    fragment.addNode(header, std::make_unique<ReaderTransfer>(arena, transport, reader, control, rows, output,
        Wire::Header, CapturedTransferEndpoint::Consumer, count), device);
    fragment.addNode(accepted, std::make_unique<ReaderPhase>(arena, transport, reader, control, rows, output,
        rr::Action::AcceptResultHeader, false, nullptr, count), device);
    auto receivedPayload = std::make_unique<ReaderTransfer>(arena, transport, reader, control, rows, output,
        Wire::Payload, CapturedTransferEndpoint::Consumer, count);
    const auto* originalPayload = receivedPayload.get();
    fragment.addNode(payload, std::move(receivedPayload), device);
    fragment.addNode(outputIdentity, std::make_unique<ReaderPhase>(arena, transport, reader, control, rows, output,
        rr::Action::AcknowledgeResult, false, originalPayload, count), device);
    fragment.addNode(sent, std::make_unique<ReaderTransfer>(arena, transport, reader, control, rows, output,
        Wire::Ack, CapturedTransferEndpoint::Producer, count), device);
    fragment.addDependency(accepted, header);
    fragment.addDependency(payload, accepted);
    fragment.addDependency(outputIdentity, payload);
    fragment.addDependency(sent, outputIdentity);
    const PooledImmutableReplicatedConsumerCompletion declared{transport, reader, control, rows, output, count,
        header, accepted, payload, outputIdentity, sent};
    requirePooledReplicatedConsumerCompletion(fragment, declared);
    publish(graph, std::move(fragment));
    if (completion) *completion = declared;
    return {header, sent,PooledImmutableDemandFrontier{PooledImmutableDemandRole::PureReader,{},header}};
}
PooledImmutableReplicatedConsumerCompletion findPooledReplicatedConsumerCompletion(const ComputeGraph& graph,
    const std::shared_ptr<const q::PooledImmutableDirectedTransport>& transport, DeviceId device)
{
    if (isGraphCaptureActive() || !transport)
        throw std::invalid_argument("Original consumer inventory belongs to cold actual recipient graph validation");
    std::optional<PooledImmutableReplicatedConsumerCompletion> result;
    for (const auto& name : graph.getExecutionOrder()) {
        const auto* node = graph.getNode(name);
        const auto* stage = node ? dynamic_cast<const ReaderPhase*>(node->stage.get()) : nullptr;
        auto declaration = stage ? stage->consumerDeclaration(transport.get(), device) : std::nullopt;
        if (!declaration) continue;
        if (result) throw std::logic_error("Original recipient graph has duplicate shared PLE ACK definitions");
        declaration->acknowledged = name;
        const auto predecessor = [&](const std::string& current) {
            const auto* actual = graph.getNode(current);
            if (!actual || actual->device != device || actual->dependencies.size() != 1)
                throw std::invalid_argument("Original consumer inventory lost its exact result/ACK chain");
            return actual->dependencies.front();
        };
        declaration->payload = predecessor(name);
        declaration->accepted = predecessor(declaration->payload);
        declaration->header = predecessor(declaration->accepted);
        for (const auto& candidate : graph.getExecutionOrder()) {
            const auto* after = graph.getNode(candidate);
            const auto* wire = after ? dynamic_cast<const ReaderTransfer*>(after->stage.get()) : nullptr;
            if (!wire || after->device != device ||
                !wire->matches(declaration->reader.get(), transport.get(), Wire::Ack, CapturedTransferEndpoint::Producer,
                    declaration->control, declaration->rows, declaration->output, declaration->count)) continue;
            if (!declaration->sent.empty())
                throw std::logic_error("Original consumer inventory has duplicate ACK sends");
            declaration->sent = candidate;
        }
        requirePooledReplicatedConsumerCompletion(graph, *declaration);
        result = std::move(declaration);
    }
    if (!result) throw std::invalid_argument("Actual recipient graph has no original shared PLE result/ACK definition");
    return std::move(*result);
}

void requirePooledReplicatedConsumerCompletion(const ComputeGraph& graph,
    const PooledImmutableReplicatedConsumerCompletion& completion)
{
    if (isGraphCaptureActive() || !completion.transport || !completion.reader ||
        !completion.transport->readerGeometry())
        throw std::invalid_argument("Consumer completion requires its original cold result/ACK declaration");
    completion.transport->requireCurrent();
    completion.reader->requireCurrent();
    if (std::find(completion.transport->readers().begin(), completion.transport->readers().end(),
            completion.reader) == completion.transport->readers().end())
        throw std::invalid_argument("Consumer completion replaced its original source/reader binding");
    const auto device = completion.reader->originalReader().device();
    const auto endpoint = [&](const std::string& name, Wire wire, CapturedTransferEndpoint side) {
        const auto* node = graph.getNode(name);
        const auto* stage = node ? dynamic_cast<const ReaderTransfer*>(node->stage.get()) : nullptr;
        if (!stage || node->device != device ||
            !stage->matches(completion.reader.get(), completion.transport.get(), wire, side,
                completion.control, completion.rows, completion.output, completion.count))
            throw std::invalid_argument("Consumer completion replaced its original result/ACK endpoint");
        return stage;
    };
    const auto phase = [&](const std::string& name, rr::Action action) {
        const auto* node = graph.getNode(name);
        const auto* stage = node ? dynamic_cast<const ReaderPhase*>(node->stage.get()) : nullptr;
        if (!stage || node->device != device ||
            !stage->validates(completion.reader.get(), completion.transport.get(), completion.control,
                completion.rows, completion.output, completion.count, action, false))
            throw std::invalid_argument("Consumer completion replaced its original header/ACK validator");
        return stage;
    };
    endpoint(completion.header, Wire::Header, CapturedTransferEndpoint::Consumer);
    phase(completion.accepted, rr::Action::AcceptResultHeader);
    const auto* payload = endpoint(completion.payload, Wire::Payload, CapturedTransferEndpoint::Consumer);
    const auto* ack = phase(completion.acknowledged, rr::Action::AcknowledgeResult);
    if (ack->originalPayload() != payload)
        throw std::invalid_argument("Consumer ACK lost its actual original payload receive proof");
    endpoint(completion.sent, Wire::Ack, CapturedTransferEndpoint::Producer);
    requireDependency(graph, completion.accepted, completion.header);
    requireDependency(graph, completion.payload, completion.accepted);
    requireDependency(graph, completion.acknowledged, completion.payload);
    requireDependency(graph, completion.sent, completion.acknowledged);
}

void requirePooledReplicatedReaderCompletion(const ComputeGraph& graph,
    const PooledImmutableReplicatedReaderCompletion& completion, const std::string& readersRetired)
{
    if (!completion.transport || !completion.transport->readerGeometry() ||
        completion.readers.size() != completion.transport->readers().size())
        throw std::invalid_argument("Shared PLE retirement requires every original reader");
    completion.transport->requireCurrent();
    const auto* source = graph.getNode(completion.sourceRead);
    const auto* originalRead = source ? dynamic_cast<const PooledImmutableSourcePhaseStage*>(source->stage.get()) : nullptr;
    if (!originalRead || !originalRead->matchesOriginal(completion.transport->sourceBoundary().get(),
            q::PooledSourceAction::Read, q::adaptive_lookup::None, completion.control) ||
        !originalRead->matchesReadOutput(completion.output))
        throw std::invalid_argument("Shared PLE reader transfer lost its actual original source Read");
    for (std::size_t i = 0; i < completion.readers.size(); ++i) {
        const auto& reader = completion.readers[i];
        if (reader.original != completion.transport->readers()[i])
            throw std::invalid_argument("Shared PLE retirement substituted original reader order");
        const auto phase = [&](const std::string& name, rr::Action action) {
            const auto* node = graph.getNode(name);
            const auto* actual = node ? dynamic_cast<const ReaderPhase*>(node->stage.get()) : nullptr;
            if (!actual || node->device != reader.original->originalSource().device() ||
                !actual->validates(reader.original.get(), completion.transport.get(), completion.control,
                    completion.rows, completion.output, completion.count, action))
                throw std::invalid_argument("Shared PLE retirement replaced an original result/ACK metadata phase");
        };
        const auto wire = [&](const std::string& name, Wire role, CapturedTransferEndpoint side) {
            const auto* node = graph.getNode(name);
            const auto* actual = node ? dynamic_cast<const ReaderTransfer*>(node->stage.get()) : nullptr;
            if (!actual || node->device != reader.original->originalSource().device() ||
                !actual->matches(reader.original.get(), completion.transport.get(), role, side,
                    completion.control, completion.rows, completion.output, completion.count))
                throw std::invalid_argument("Shared PLE retirement replaced an original result/ACK endpoint");
        };
        phase(reader.header, rr::Action::PrepareResultHeader);
        wire(reader.headerSent, Wire::Header, CapturedTransferEndpoint::Producer);
        phase(reader.extent, rr::Action::PrepareResult);
        wire(reader.payloadSent, Wire::Payload, CapturedTransferEndpoint::Producer);
        wire(reader.ackReceived, Wire::Ack, CapturedTransferEndpoint::Consumer);
        phase(reader.ackValidated, rr::Action::ConfirmAck);
        if (reader.original->nativeCollective()) {
            const auto* validator = dynamic_cast<const ReaderPhase*>(graph.getNode(reader.ackValidated)->stage.get());
            const auto* payload = dynamic_cast<const ReaderTransfer*>(graph.getNode(reader.payloadSent)->stage.get());
            if (!validator || !payload || validator->originalPayload() != payload)
                throw std::invalid_argument("Native source ACK replaced its exact whole-launch payload completion owner");
        }
        requireDependency(graph, reader.header, completion.sourceRead);
        requireDependency(graph, reader.headerSent, reader.header);
        requireDependency(graph, reader.extent, reader.headerSent);
        requireDependency(graph, reader.payloadSent, reader.extent);
        requireDependency(graph, reader.ackReceived, reader.payloadSent);
        requireDependency(graph, reader.ackValidated, reader.ackReceived);
        requireDependency(graph, readersRetired, reader.ackValidated);
    }
}
} // namespace llaminar2

namespace llaminar2 {
bool isPooledPureReaderDemandHeader(const ComputeNode& node,const RetainedGraphParticipantDefinitionDomain& domain){
    const auto* stage=dynamic_cast<const ReaderTransfer*>(node.stage.get());
    return stage&&stage->isDemandHeader(domain);
}
}
