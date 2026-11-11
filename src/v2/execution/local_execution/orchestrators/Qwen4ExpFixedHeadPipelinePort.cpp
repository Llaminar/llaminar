/**
 * @file Qwen4ExpFixedHeadPipelinePort.cpp
 * @brief Original fixed native n-gram ranges bound to participant-local forward graphs.
 *
 * Only the PLE compute stage owns a query. The addressed storage service runs
 * on the existing forward parent of each physical range owner. Its receive
 * count comes from the native receipt, and no host reads it during replay.
 */
#include "Qwen4ExpFixedHeadPipelinePort.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/DeviceCountedAllGather.h"
#include "execution/compute_stages/stages/NativeAllGatherStage.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/IGraphBuilder.h"
#include "execution/mtp/DeviceGenerationControlBinding.h"
#include "execution/qwen4exp/Qwen4ExpAuxiliaryState.h"
#include "execution/qwen4exp/Qwen4ExpPreparedStorage.h"
#include "interfaces/IModelContext.h"
#include "loaders/ResidentImmutableRowTable.h"
#include "loaders/ResolvedImmutableRowTablePlacement.h"
#include "memory/BufferArena.h"
#include "models/qwen4exp/Qwen4ExpReplicatedPLEQuery.h"
#include "utils/PerfStatsCollector.h"
#include <algorithm>
#include <array>
#include <functional>
#include <stdexcept>
#include <utility>

namespace llaminar2 {
namespace {
namespace q = qwen4exp;
using q::fixed_head_lookup_detail::product;

/** @brief One exact graph arena role and its local producer/consumer access. */
struct Operand {
    enum class Access { Read, Write, ReadWrite };
    BufferId role;
    std::shared_ptr<TensorBase> tensor;
    Access access;
};

/** @brief One capture-only node over original persistent roots and a typed native operation. */
class FixedHeadPipelineStage final : public IComputeStage {
public:
    /** @brief Freeze the participant-local node and the original arena contract before capture. */
    FixedHeadPipelineStage(DeviceId device, std::string name, ComputeStageType type,
        std::uint64_t identity, std::vector<Operand> operands, std::function<void(void *)> record,
        std::shared_ptr<const ResidentImmutableRowTable> table = {})
        : IComputeStage(device), name_(std::move(name)), type_(type), identity_(identity),
          operands_(std::move(operands)), record_(std::move(record)), table_(std::move(table))
    {
        if (!device.is_gpu() || name_.empty() || !identity_ || !record_)
            throw std::invalid_argument("Fixed PLE pipeline stage lost its original operation");
        std::vector<TensorBase *> roots;
        roots.reserve(operands_.size());
        for (const auto &operand : operands_) roots.push_back(operand.tensor.get());
        prepared_.emplace(device, roots);
    }

    /** @brief Acquire only this owner's immutable table on the exact capture stream. */
    bool prepareGraphLaunch(IDeviceContext *context, void *stream) override
    {
        if (!context || context->deviceId() != device() || reinterpret_cast<std::uintptr_t>(stream) <= 2 ||
            (prepared_stream_ && prepared_stream_ != stream))
            throw std::invalid_argument("Fixed PLE pipeline stage changed its original device/stream");
        prepared_->seal();
        if (table_ && !read_) read_.emplace(table_->acquire(stream, TransferEngine::instance()));
        const auto identity = captureIdentity(stream);
        if (capture_identity_ && capture_identity_ != identity)
            throw std::logic_error("Fixed PLE pipeline stage changed its original physical capture identity");
        capture_identity_ = identity;
        prepared_stream_ = stream;
        return true;
    }

    /** @brief Record exact native work after its declared local predecessors. */
    bool execute(IDeviceContext *context) override
    {
        if (!context || context->deviceId() != device() || !isGraphCaptureActive() ||
            requireGPUStream() != prepared_stream_)
            throw std::logic_error("Fixed PLE pipeline service requires its original captured forward");
        if (!capture_identity_ || capture_identity_ != captureIdentity(prepared_stream_))
            throw std::logic_error("Fixed PLE pipeline stage has stale prepared roots or native source");
        const auto gpu = gpuExecution();
        for (const auto &operand : operands_) {
            if (operand.access != Operand::Access::Write) gpu.requirePreparedInput(operand.tensor.get());
            if (operand.access != Operand::Access::Read) gpu.requirePreparedOutput(operand.tensor.get());
        }
        if (table_) {
            if (!read_) throw std::logic_error("Fixed PLE pipeline pack lost its original immutable table lease");
            TransferEngine::instance().requireImmutableNativeDeviceInput(*read_, device(), prepared_stream_);
        }
        record_(prepared_stream_);
        // The native bindings publish their own exact completed receive roots.
        if (type_ == ComputeStageType::QWEN4EXP_NGRAM_LOOKUP)
            for (const auto &operand : operands_)
                if (operand.access != Operand::Access::Read) gpu.publish(operand.tensor.get());
        return true;
    }

    /** @return The existing graph capture classification of the actual operation. */
    ComputeStageType type() const override { return type_; }
    /** @return Stable diagnostic identity of this specific producer or consumer. */
    std::string name() const override { return name_; }
    /** @return Only the original owning native backend is supported. */
    bool supportsBackend(ComputeBackendType backend) const override
    {
        return (device().is_cuda() && backend == ComputeBackendType::GPU_CUDA) ||
            (device().is_rocm() && backend == ComputeBackendType::GPU_ROCM);
    }
    /** @return This service requires exact retained row geometry for every legal prefill. */
    bool supportsPaddedPrefillRealLengthContract() const override { return false; }
    /** @return Only exact retained geometry is eligible for cold prefill capture. */
    bool supportsPaddedPrefillGraphCapturePreflight() const override { return false; }
    /** @return Native stage preparation takes place before capture only. */
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override { return GraphLaunchPreparationPolicy::CaptureOnly; }
    /** @return The exact original native graph's local input/output dependencies. */
    StageBufferContract bufferContract() const override
    {
        StageBufferContract result;
        for (const auto &operand : operands_) {
            const auto *dtype = operand.tensor->native_type() == TensorType::FP32 ? "FP32" : "INT32";
            switch (operand.access) {
            case Operand::Access::Read: result.addInput(operand.role, dtype); break;
            case Operand::Access::Write: result.addOutput(operand.role, dtype); break;
            case Operand::Access::ReadWrite: result.addPreallocatedInOut(operand.role, dtype); break;
            }
        }
        return result;
    }
    /** @return Exact logical operation, table lease and original declared operand identities. */
    std::uint64_t graphCaptureVariantSignature() const override
    {
        prepared_->validateIfSealed();
        if (prepared_stream_ && captureIdentity(prepared_stream_) != capture_identity_)
            throw std::logic_error("Fixed PLE pipeline cache changed its exact physical capture identity");
        auto signature = identity_;
        for (const auto &operand : operands_) {
            signature = (signature ^ reinterpret_cast<std::uintptr_t>(operand.tensor.get())) * 1099511628211ULL;
            signature = (signature ^ static_cast<std::uint64_t>(operand.role)) * 1099511628211ULL;
        }
        return signature ? signature : 1;
    }
    /** @return No intermediate storage or mutable count is exposed through diagnostics. */
    StageDumpInfo buildDumpInfoImpl() const override { return {}; }

private:
    /** @return The exact frozen native ABI after canonical arena preparation. */
    std::uint64_t captureIdentity(void *stream) const
    {
        auto value = (prepared_->preparedIdentity() ^ identity_) * 1099511628211ULL;
        value = (value ^ reinterpret_cast<std::uintptr_t>(stream)) * 1099511628211ULL;
        if (table_) {
            const auto range = table_->rangeView();
            for (const auto word : {std::uint64_t{reinterpret_cast<std::uintptr_t>(table_.get())},
                     std::uint64_t{reinterpret_cast<std::uintptr_t>(range.nativeData())}, range.storageGeneration(),
                     range.globalRowBegin(), range.rowCount()})
                value = (value ^ word) * 1099511628211ULL;
        }
        return value ? value : 1;
    }
    const std::string name_;
    const ComputeStageType type_;
    const std::uint64_t identity_;
    const std::vector<Operand> operands_;
    const std::function<void(void *)> record_;
    const std::shared_ptr<const ResidentImmutableRowTable> table_;
    std::optional<Qwen4ExpPreparedStorage> prepared_;
    std::uint64_t capture_identity_ = 0;
    std::optional<ImmutableNativeDeviceRead> read_;
    void *prepared_stream_ = nullptr;
};

/** @brief Resolve a prevalidated subspan only after original arena preparation. */
template<class T> T *deviceAt(const std::shared_ptr<TensorBase> &tensor, std::size_t offset = 0)
{
    if (!tensor || !tensor->gpu_data_ptr() || offset >= tensor->size_bytes())
        throw std::logic_error("Fixed PLE pipeline stage lost its prepared native subspan");
    auto *address = static_cast<std::byte *>(tensor->gpu_data_ptr()) + offset;
    if (reinterpret_cast<std::uintptr_t>(address) % alignof(T))
        throw std::logic_error("Fixed PLE pipeline native subspan is misaligned");
    return reinterpret_cast<T *>(address);
}

/** @brief Append a typed node and its existing local dependencies together. */
void appendNode(ComputeGraph &graph, std::string name, std::unique_ptr<IComputeStage> stage,
    DeviceId device, std::span<const std::string> dependencies)
{
    graph.addNode(name, std::move(stage), device);
    for (const auto &dependency : dependencies) graph.addDependency(name, dependency);
}
}

Qwen4ExpFixedHeadPipelinePort::Qwen4ExpFixedHeadPipelinePort(
    std::shared_ptr<const ResolvedImmutableRowTablePlacement> placement, Participant participant,
    int native_member, const q::NgramHashPolicy &policy)
    : placement_(std::move(placement)), projection_(placement_->fixedHeadPipeline()), participant_(participant),
      native_member_(native_member), owner_(projection_->owner({projection_->worldRank(), participant_.device}))
{
    if (!participant_.arena || !participant_.model || participant_.stage != owner_.stage_index ||
        participant_.stage_member != owner_.participant_index || reinterpret_cast<std::uintptr_t>(participant_.setup_stream) <= 2)
        throw std::invalid_argument("Fixed PLE pipeline port changed its actual stage/worker declaration");
    const auto source = participant_.model->resolvedImmutableRowTablePlacement();
    table_ = participant_.model->residentImmutableRowTable(participant_.device);
    if (!source || *source != *placement_ || !table_ || table_->device() != participant_.device ||
        table_->sourceRange() != placement_->fixedHeadSourceRangeFor(owner_.allocator))
        throw std::invalid_argument("Fixed PLE pipeline port lost its original materialized native range");
    auto &worker = GPUDeviceContextPool::instance().getContext(participant_.device);
    worker.submitAndWait([&] {
        kernels_ = q::createImmutableFixedHeadPipelineKernels(participant_.device, table_->rangeView(), policy,
            {owner_.first_head, owner_.head_count});
    });
    const auto bytes = projection_->scratchBytes(owner_.allocator, projection_->maximumRows(), placement_->extent().rowBytes());
    if (bytes) {
        if (!participant_.arena->allocateDeviceStorage(BufferId::QFN_NGRAM_SHARD_PIPELINE_NATIVE, participant_.device))
            throw std::runtime_error("Fixed PLE original pipeline scratch could not materialize from its arena");
        scratch_ = bank(BufferId::QFN_NGRAM_SHARD_PIPELINE_NATIVE);
        if (scratch_->size_bytes() != bytes)
            throw std::invalid_argument("Fixed PLE pipeline scratch differs from its exact physical BOM");
    }
    PerfStatsCollector::addCounter("immutable_lookup", "fixed_head_pipeline_storage_ranges", 1.0,
        "graph_setup", participant_.device.toString(), {{"first_head", std::to_string(owner_.first_head)},
            {"heads", std::to_string(owner_.head_count)}, {"stage", std::to_string(owner_.stage_index)},
            {"row_bytes", std::to_string(placement_->extent().rowBytes())},
            {"role", consumesPLE() ? "ple_consumer" : "storage_only"}});
}

std::vector<std::shared_ptr<const Qwen4ExpFixedHeadPipelinePort>> Qwen4ExpFixedHeadPipelinePort::prepare(
    std::shared_ptr<const ResolvedImmutableRowTablePlacement> placement, std::span<const Participant> members,
    ILocalTPContext &pipeline, const RankInventory &inventory, const std::shared_ptr<PhysicalMemoryAuthority> &memory,
    const q::NgramHashPolicy &policy)
{
    const auto *projection = placement ? placement->fixedHeadPipeline() : nullptr;
    if (!projection || !memory || inventory.rank != projection->worldRank() || memory->worldRank() != projection->worldRank() ||
        members.size() != projection->owners().size() || members.size() != pipeline.devices().size())
        throw std::invalid_argument("Fixed PLE pipeline requires its complete original native rank and PMA");
    std::vector<std::shared_ptr<Qwen4ExpFixedHeadPipelinePort>> ports;
    std::vector<void *> streams;
    for (std::size_t index = 0; index < members.size(); ++index) {
        if (members[index].device != pipeline.devices()[index].toLocalDeviceId() ||
            members[index].model != members.front().model)
            throw std::invalid_argument("Fixed PLE pipeline changed its original native/model roster");
        ports.push_back(std::shared_ptr<Qwen4ExpFixedHeadPipelinePort>(new Qwen4ExpFixedHeadPipelinePort(
            placement, members[index], static_cast<int>(index), policy)));
        streams.push_back(members[index].setup_stream);
    }
    const auto find = [&](PhysicalMemoryAllocatorIdentity identity) -> Qwen4ExpFixedHeadPipelinePort & {
        const auto port = std::find_if(ports.begin(), ports.end(), [&](const auto &actual) { return actual->owner_.allocator == identity; });
        if (port == ports.end()) throw std::invalid_argument("Fixed PLE pipeline omitted an original physical owner");
        return **port;
    };
    const auto stride = placement->extent().rowBytes();
    for (const auto &owner : projection->owners()) {
        if (owner.stage_index == projection->consumerStage()) continue;
        auto &storage = find(owner.allocator);
        auto &reader = find(projection->consumers()[owner.consumer_index]);
        const auto rows = projection->maximumRows();
        const auto storage_scratch = projection->edgeScratch(storage.owner_.allocator, owner.allocator, rows, stride);
        const auto reader_scratch = projection->edgeScratch(reader.owner_.allocator, owner.allocator, rows, stride);
        const std::array storage_coordinates{storage.native_member_};
        const std::array reader_coordinates{reader.native_member_};
        auto ids = PipelineNativeByteChannel::create(pipeline, reader.native_member_, storage_coordinates,
            storage_scratch.ids_bytes, memory, streams);
        auto response = PipelineNativeByteChannel::create(pipeline, storage.native_member_, reader_coordinates,
            storage_scratch.native_bytes, memory, streams);
        storage.edges_.push_back({owner, storage_scratch, ids, response});
        reader.edges_.push_back({owner, reader_scratch, ids, response});
        for (const bool reply : {false, true})
            PerfStatsCollector::addCounter("immutable_lookup", "fixed_head_pipeline_addressed_edges", 1.0,
                "graph_setup", (reply ? storage : reader).participant_.device.toString(),
                {{"kind", reply ? "native_rows" : "ids"}, {"source", std::to_string((reply ? storage : reader).native_member_)},
                 {"consumer", std::to_string((reply ? reader : storage).native_member_)},
                 {"row_bytes", std::to_string(owner.head_count * (reply ? stride : sizeof(std::int64_t)))},
                 {"transport", storage.participant_.device.is_cuda() ? "nccl" : "rccl"}});
    }
    std::vector<DeviceId> reader_devices;
    std::vector<void *> reader_streams;
    auto *reader_context = find(projection->consumers().front()).participant_.stage_context;
    const int maximum_heads = projection->maximumAssembledHeads();
    for (const auto &reader : projection->consumers()) {
        auto &port = find(reader);
        if (port.participant_.stage_context != reader_context)
            throw std::invalid_argument("Fixed PLE readers changed their original local context");
        reader_devices.push_back(reader.device);
        reader_streams.push_back(port.participant_.setup_stream);
        // Bank addresses are frozen before the complete serving graph family.
        for (const auto role : {BufferId::QFN_NGRAM_SHARD_LOCAL_HEADER, BufferId::QFN_NGRAM_SHARD_HEADERS,
                 BufferId::QFN_NGRAM_SHARD_TRANSPORT_ROWS, BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE,
                 BufferId::QFN_NGRAM_SHARD_GATHERED_NATIVE}) {
            if (projection->consumers().size() == 1 &&
                (role == BufferId::QFN_NGRAM_SHARD_HEADERS || role == BufferId::QFN_NGRAM_SHARD_GATHERED_NATIVE)) continue;
            if (!port.participant_.arena->allocateDeviceStorage(role, reader.device))
                throw std::runtime_error("Fixed PLE original reader bank could not materialize from its arena");
        }
    }
    if (reader_devices.size() > 1 && (!reader_context || reader_context->degree() != int(reader_devices.size()) ||
        !std::equal(reader_devices.begin(), reader_devices.end(), reader_context->devices().begin(), reader_context->devices().end(),
            [](DeviceId actual, const auto &member) { return actual == member.toLocalDeviceId(); })))
        throw std::invalid_argument("Fixed PLE consumer gather requires its actual original stage-native membership");
    if (projection->requiresVariableGather()) {
        const auto topology = DeviceCountedAllGatherTopology::resolve(inventory, reader_devices);
        const auto maximum = product(product(projection->maximumRows(), maximum_heads), stride);
        auto gather = DeviceCountedAllGather::create(*memory, *reader_context, DeviceId::cpu(), topology, reader_streams, maximum);
        for (const auto &reader : projection->consumers()) find(reader).variable_gather_ = gather;
    }
    return {ports.begin(), ports.end()};
}

int Qwen4ExpFixedHeadPipelinePort::requireConsumer(const IModelContext &model, const BufferArena &arena, DeviceId device) const
{
    if (!consumesPLE() || participant_.model != &model || participant_.arena != &arena ||
        participant_.device != device || participant_.stage_member != owner_.consumer_index)
        throw std::invalid_argument("Fixed PLE pipeline query substituted its original stage/model/arena consumer");
    return owner_.consumer_index;
}

void Qwen4ExpFixedHeadPipelinePort::requireQuery(const Qwen4ExpReplicatedPLEQuery &query, const ForwardInput &input,
    DeviceRowRange rows, const Qwen4ExpAuxiliaryState &history) const
{
    if (!consumesPLE() || query.pipeline_ != this || query.cohort_ || !query.owner_ ||
        query.participant_ != owner_.consumer_index || query.device_ != participant_.device ||
        query.device_ != input.device || input.batch_size != 1 || query.tokens_ != input.token_ids_device ||
        !query.tokens_ || query.cache_ != input.getKVCacheForDevice(input.device) || query.cache_ != &history.cache() ||
        history.device() != input.device || query.role_ != input.execution_role || query.phase_ != input.execution_phase ||
        query.rows_.physicalRows() != input.seq_len || query.rows_.countCapacity() != input.seq_len ||
        query.rows_.captureIdentity().first_physical_row != 0 || query.rows_.captureIdentity().repeats_per_row != 1 ||
        query.rows_.captureIdentity() != rows.captureIdentity() ||
        query.isWorkspaceDeclaration() != (input.graph_submission_intent == ForwardGraphSubmissionIntent::WorkspaceDeclaration))
        throw std::invalid_argument("Fixed PLE pipeline query lost its actual DGO token/count/history producer");
    (void)history.ple(0);
}

std::shared_ptr<TensorBase> Qwen4ExpFixedHeadPipelinePort::bank(BufferId role) const
{
    auto result = participant_.arena->getSharedTensor(role);
    if (!result || result.get() != participant_.arena->getTensor(role) || result->is_view())
        throw std::invalid_argument("Fixed PLE pipeline lost an original participant-local arena root");
    return result;
}

CapturedTransferMessage Qwen4ExpFixedHeadPipelinePort::wireMessage(const Edge &edge, bool response, int rows) const
{
    if (rows <= 0 || rows > projection_->maximumRows())
        throw std::invalid_argument("Fixed PLE pipeline graph exceeds its original resident row envelope");
    const auto stride = product(edge.storage.head_count, response ? placement_->extent().rowBytes() : sizeof(std::int64_t));
    constexpr std::uint64_t prefix = 0x51464e0000000000ULL;
    return {prefix | (std::uint64_t(response) << 40) | (std::uint64_t(edge.storage.first_head) << 32) | std::uint32_t(rows),
        product(rows, stride)};
}

std::string Qwen4ExpFixedHeadPipelinePort::appendStorage(ComputeGraph &graph, const ForwardInput &input,
    std::span<const std::string> dependencies) const
{
    if (consumesPLE() || input.device != participant_.device || input.batch_size != 1 ||
        edges_.size() != 1 || !scratch_ || !edges_.front().scratch.service_rows_offset)
        throw std::invalid_argument("Fixed PLE storage service lost its actual native owner or forward scope");
    const auto &edge = edges_.front();
    const auto rows_offset = *edge.scratch.service_rows_offset;
    const auto count = input.seq_len;
    const auto control = DeviceGenerationControlBinding::admit(*participant_.arena, participant_.device, 0);
    const q::FixedHeadPipelineError error{control.healthy(), control.error(), {control.row()}};
    constexpr auto scratch_role = BufferId::QFN_NGRAM_SHARD_PIPELINE_NATIVE;
    constexpr auto control_role = BufferId::MTP_GENERATION_CONTROL;
    using Access = Operand::Access;
    using Receipt = DeviceGenerationCollectiveReceipt;
    using ReceiptKind = DeviceGenerationCollectiveReceiptKind;
    const auto native_stride = placement_->extent().rowBytes();
    const auto ids = PipelineNativeByteBinding(edge.ids, native_member_, wireMessage(edge, false, count),
        scratch_, edge.scratch.ids_offset, PipelineNativeReceivedRows{scratch_, rows_offset,
            product(owner_.head_count, sizeof(std::int64_t))})
        .withTerminalReceipt(Receipt::fixedHeadPipeline(control, ReceiptKind::FixedHeadPipelineRequest,
            count, owner_.head_count, sizeof(std::int64_t)));
    const auto response = PipelineNativeByteBinding(edge.rows, native_member_, wireMessage(edge, true, count),
        scratch_, edge.scratch.native_offset, PipelineNativeRowCount{scratch_, rows_offset,
            product(owner_.head_count, native_stride)})
        .withTerminalReceipt(Receipt::fixedHeadPipeline(control, ReceiptKind::FixedHeadPipelineResponse,
            count, owner_.head_count, native_stride));
    const auto self = shared_from_this();
    const std::string receive = "qfn_fixed_pipeline_receive_ids";
    const std::string pack = "qfn_fixed_pipeline_pack_storage";
    const std::string send = "qfn_fixed_pipeline_return_rows";
    appendNode(graph, receive, std::make_unique<FixedHeadPipelineStage>(participant_.device, receive,
        ComputeStageType::PIPELINE_ACTIVATION_EXCHANGE, ids.captureIdentity(),
        std::vector<Operand>{{scratch_role, scratch_, Access::Write}, {control_role, bank(control_role), Access::ReadWrite}},
        [ids](void *stream) { ids.enqueue(stream); }), participant_.device, dependencies);
    appendNode(graph, pack, std::make_unique<FixedHeadPipelineStage>(participant_.device, pack,
        ComputeStageType::QWEN4EXP_NGRAM_LOOKUP,
        (response.captureIdentity() ^ reinterpret_cast<std::uintptr_t>(kernels_.get())) | 1u,
        std::vector<Operand>{{scratch_role, scratch_, Access::ReadWrite}, {control_role, bank(control_role), Access::ReadWrite}},
        [self, count, scratch = edge.scratch, rows_offset, error](void *stream) {
            auto *rows = deviceAt<std::int32_t>(self->scratch_, rows_offset);
            self->kernels_->pack({{count, rows}, error, deviceAt<std::int64_t>(self->scratch_, scratch.ids_offset),
                deviceAt<std::byte>(self->scratch_, scratch.native_offset), self->owner_.head_count, 0,
                self->owner_.head_count, 0}, stream);
            self->kernels_->sealResponse(rows, error, stream);
        }, table_), participant_.device, std::span{&receive, 1});
    appendNode(graph, send, std::make_unique<FixedHeadPipelineStage>(participant_.device, send,
        ComputeStageType::PIPELINE_ACTIVATION_EXCHANGE, response.captureIdentity(),
        std::vector<Operand>{{scratch_role, scratch_, Access::Read}, {control_role, bank(control_role), Access::ReadWrite}},
        [response](void *stream) { response.enqueue(stream); }), participant_.device, std::span{&pack, 1});
    return send;
}

void Qwen4ExpFixedHeadPipelinePort::appendConsumer(ComputeGraph &fragment, const std::string &prefix,
    const std::string &terminal, const Qwen4ExpReplicatedPLEQuery &query, std::optional<Qwen4ExpIntBinding> count,
    Qwen4ExpIntBinding ids, Qwen4ExpFloatBinding output) const
{
    if (!consumesPLE() || query.pipeline_ != this || query.participant() != owner_.consumer_index ||
        query.device() != participant_.device || ids.byteOffset || output.byteOffset)
        throw std::invalid_argument("Fixed PLE pipeline lookup changed its original consumer/arena identity");
    const auto capacity = query.rows().physicalRows();
    if (capacity <= 0 || capacity > projection_->maximumRows())
        throw std::invalid_argument("Fixed PLE pipeline lookup exceeds its admitted serving row envelope");
    const auto original_count = count ? bank(count->buffer) : std::shared_ptr<TensorBase>{};
    if (count.has_value() != (query.rows().countOwner() != nullptr) ||
        (count && (original_count.get() != count->tensor ||
            deviceAt<std::int32_t>(original_count, count->byteOffset) != query.rows().countOwner())))
        throw std::invalid_argument("Fixed PLE pipeline query lost its exact original row producer storage");
    const auto raw_ids = bank(ids.buffer);
    const auto output_owner = bank(output.buffer);
    if (raw_ids.get() != ids.tensor || output_owner.get() != output.tensor ||
        raw_ids->size_bytes() < product(product(capacity, q::kNgramHeads), sizeof(std::int64_t)) ||
        output_owner->size_bytes() < product(product(capacity, q::kNgramHeads), table_->rangeView().rowElements() * sizeof(float)))
        throw std::invalid_argument("Fixed PLE pipeline original ID and decoded output envelopes disagree");
    constexpr auto scratch_role = BufferId::QFN_NGRAM_SHARD_PIPELINE_NATIVE;
    constexpr auto control_role = BufferId::MTP_GENERATION_CONTROL;
    constexpr auto local_header_role = BufferId::QFN_NGRAM_SHARD_LOCAL_HEADER;
    constexpr auto headers_role = BufferId::QFN_NGRAM_SHARD_HEADERS;
    constexpr auto transport_role = BufferId::QFN_NGRAM_SHARD_TRANSPORT_ROWS;
    constexpr auto local_native_role = BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE;
    constexpr auto gathered_native_role = BufferId::QFN_NGRAM_SHARD_GATHERED_NATIVE;
    const auto control = DeviceGenerationControlBinding::admit(*participant_.arena, participant_.device, 0);
    const q::FixedHeadPipelineError error{control.healthy(), control.error(), {control.row()}};
    const auto actual_reader = owner_.consumer_index;
    const auto reader_count = static_cast<int>(projection_->consumers().size());
    const auto local_header = bank(local_header_role);
    const auto headers = reader_count > 1 ? bank(headers_role) : local_header;
    const auto transport = bank(transport_role);
    const auto local = bank(local_native_role);
    const auto gathered = reader_count > 1 ? bank(gathered_native_role) : std::shared_ptr<TensorBase>{};
    const auto native_stride = placement_->extent().rowBytes();
    const auto assembled = q::FixedHeadPipelineHeads{owner_.first_head, projection_->assembledHeadCount(actual_reader)};
    const q::FixedHeadPipelineRows original_rows{capacity, query.rows().countOwner()};
    const q::FixedHeadPipelineRows payload_rows{capacity, deviceAt<std::int32_t>(transport)};
    const auto response_offset = projection_->readerResponseRowsOffset(owner_.allocator, projection_->maximumRows(), native_stride);
    if (response_offset.has_value() != !edges_.empty())
        throw std::logic_error("Fixed PLE pipeline reader response does not match its original incident edges");
    const auto self = shared_from_this();
    using Receipt = DeviceGenerationCollectiveReceipt;
    using ReceiptKind = DeviceGenerationCollectiveReceiptKind;
    using Access = Operand::Access;
    std::string previous;
    const auto attach = [&](std::string name, std::unique_ptr<IComputeStage> stage) {
        appendNode(fragment, name, std::move(stage), participant_.device,
            previous.empty() ? std::span<const std::string>{} : std::span<const std::string>{&previous, 1});
        previous = std::move(name);
    };
    const auto add = [&](std::string name, std::vector<Operand> operands, std::function<void(void *)> record,
        ComputeStageType type = ComputeStageType::QWEN4EXP_NGRAM_LOOKUP, std::uint64_t identity = 0,
        std::shared_ptr<const ResidentImmutableRowTable> table = {}) {
        if (!identity) {
            identity = (query.captureIdentity() ^ reinterpret_cast<std::uintptr_t>(kernels_.get())) * 1099511628211ULL;
            for (const unsigned char byte : name) identity = (identity ^ byte) * 1099511628211ULL;
        }
        attach(name, std::make_unique<FixedHeadPipelineStage>(participant_.device, name, type, identity,
            std::move(operands), std::move(record), std::move(table)));
    };
    if (query.isWorkspaceDeclaration()) fragment.markWorkspaceDeclaration();
    const auto addCountOperand = [&](std::vector<Operand> operands) {
        if (count) operands.push_back({count->buffer, original_count, Access::Read});
        return operands;
    };
    const auto consensus = [=] {
        return q::FixedHeadPipelineConsensus{original_rows, error,
            deviceAt<q::FixedHeadValidationHeader>(local_header), deviceAt<q::FixedHeadValidationHeader>(headers),
            deviceAt<std::int32_t>(transport), reader_count};
    };
    add(prefix + "fixed_pipeline_prepare", addCountOperand({{transport_role, transport, Access::Write},
        {control_role, bank(control_role), Access::ReadWrite}}),
        [self, consensus](void *stream) { self->kernels_->prepareConsumer(consensus(), stream); });

    // Submit every independent request before waiting for any remote payload.
    // Local source packing can proceed while the remote owners gather rows.
    for (const auto &edge : edges_) {
        const auto name = prefix + "fixed_pipeline_request_" + std::to_string(edge.storage.first_head);
        const q::FixedHeadPipelineHeads heads{edge.storage.first_head, edge.storage.head_count};
        add(name + "_pack", {{ids.buffer, raw_ids, Access::Read}, {transport_role, transport, Access::Read},
            {scratch_role, scratch_, Access::Write}}, [self, raw_ids, payload_rows, heads, offset = edge.scratch.ids_offset](void *stream) {
                self->kernels_->projectRequest({payload_rows, heads, deviceAt<std::int64_t>(raw_ids),
                    deviceAt<std::int64_t>(self->scratch_, offset)}, stream);
            });
        const auto binding = PipelineNativeByteBinding(edge.ids, native_member_, wireMessage(edge, false, capacity), scratch_,
            edge.scratch.ids_offset, PipelineNativeRowCount{transport, 0, product(heads.count, sizeof(std::int64_t))})
            .withTerminalReceipt(Receipt::fixedHeadPipeline(control, ReceiptKind::FixedHeadPipelineRequest,
                capacity, heads.count, sizeof(std::int64_t)));
        add(name + "_send", {{scratch_role, scratch_, Access::Read}, {transport_role, transport, Access::Read},
            {control_role, bank(control_role), Access::ReadWrite}},
            [binding](void *stream) { binding.enqueue(stream); }, ComputeStageType::PIPELINE_ACTIVATION_EXCHANGE, binding.captureIdentity());
    }
    add(prefix + "fixed_pipeline_pack_local", {{ids.buffer, raw_ids, Access::Read}, {transport_role, transport, Access::Read},
        {local_native_role, local, Access::Write}, {control_role, bank(control_role), Access::ReadWrite}},
        [self, payload_rows, error, raw_ids, local, assembled](void *stream) {
            self->kernels_->pack({payload_rows, error, deviceAt<std::int64_t>(raw_ids), deviceAt<std::byte>(local),
                q::kNgramHeads, self->owner_.first_head, assembled.count, 0}, stream);
        }, ComputeStageType::QWEN4EXP_NGRAM_LOOKUP, 0, table_);
    for (const auto &edge : edges_) {
        const auto name = prefix + "fixed_pipeline_response_" + std::to_string(edge.storage.first_head);
        const auto response_rows = *response_offset;
        const auto binding = PipelineNativeByteBinding(edge.rows, native_member_, wireMessage(edge, true, capacity), scratch_,
            edge.scratch.native_offset, PipelineNativeReceivedRows{scratch_, response_rows,
                product(edge.storage.head_count, native_stride)})
            .withTerminalReceipt(Receipt::fixedHeadPipeline(control, ReceiptKind::FixedHeadPipelineResponse,
                capacity, edge.storage.head_count, native_stride));
        add(name + "_receive", {{scratch_role, scratch_, Access::Write}, {control_role, bank(control_role), Access::ReadWrite}},
            [binding](void *stream) { binding.enqueue(stream); },
            ComputeStageType::PIPELINE_ACTIVATION_EXCHANGE, binding.captureIdentity());
        add(name + "_merge", {{scratch_role, scratch_, Access::Read}, {transport_role, transport, Access::Read},
            {local_native_role, local, Access::ReadWrite}, {control_role, bank(control_role), Access::ReadWrite}},
            [self, local, error, payload_rows, response_rows, owner = edge.storage, offset = edge.scratch.native_offset,
                assembled](void *stream) {
                const auto *received = deviceAt<std::int32_t>(self->scratch_, response_rows);
                self->kernels_->validateResponse(payload_rows, received, error, stream);
                self->kernels_->merge({{payload_rows.exact, received}, error, deviceAt<std::byte>(self->scratch_, offset),
                    deviceAt<std::byte>(local), owner.head_count, assembled.count, owner.first_head - assembled.first}, stream);
            });
    }

    // Every addressed peer has now returned its actual receipt, including a
    // zero-byte response on storage error. One original stage consensus
    // withdraws payload permission for all PLE readers before the data gather.
    // Sealing consumes the prior request extent before publishing the header extent.
    add(prefix + "fixed_pipeline_status", addCountOperand({{local_header_role, local_header, Access::Write},
        {transport_role, transport, Access::ReadWrite}, {control_role, bank(control_role), Access::ReadWrite}}),
        [self, consensus](void *stream) { self->kernels_->sealConsumer(consensus(), stream); });
    if (reader_count > 1) {
        const auto name = prefix + "fixed_pipeline_header_gather";
        attach(name, std::make_unique<NativeAllGatherStage>(NativeAllGatherStage::Params{
            .device_id = participant_.device, .tp_ctx = participant_.stage_context,
            .local_input = local_header.get(), .rank_major_output = headers.get(),
            .bytes_per_participant = sizeof(q::FixedHeadValidationHeader), .participant = actual_reader, .stage_name = name,
            .input_buffer_id = local_header_role, .output_buffer_id = headers_role,
            .arena_rows = NativeAllGatherStage::ArenaRows{transport.get(), transport_role, 0, 1, sizeof(q::FixedHeadValidationHeader)}}));
    }
    add(prefix + "fixed_pipeline_consensus", addCountOperand({{reader_count == 1 ? local_header_role : headers_role,
        reader_count == 1 ? local_header : headers, Access::Read}, {transport_role, transport, Access::Write},
        {control_role, bank(control_role), Access::ReadWrite}}),
        [self, consensus](void *stream) { self->kernels_->validateConsensus(consensus(), stream); });

    const auto gather_receipt = Receipt::fixedHeadPipeline(control, ReceiptKind::FixedHeadPipelineGather,
        capacity, assembled.count, native_stride);
    std::shared_ptr<const DeviceCountedAllGatherPacketLayout> packets;
    std::optional<DeviceCountedAllGatherBinding> gather_binding;
    if (variable_gather_) {
        std::vector<std::size_t> capacities;
        for (int reader = 0; reader < reader_count; ++reader)
            capacities.push_back(product(product(capacity, projection_->assembledHeadCount(reader)), native_stride));
        packets = std::make_shared<const DeviceCountedAllGatherPacketLayout>(capacities, actual_reader);
        gather_binding = variable_gather_->bindPackets(0x51464e504c450000ULL | std::uint32_t(capacity), packets,
            local, gathered, gather_receipt);
        add(prefix + "fixed_pipeline_gather_unequal", {{local_native_role, local, Access::Read},
            {gathered_native_role, gathered, Access::Write}, {transport_role, transport, Access::Read},
            {control_role, bank(control_role), Access::ReadWrite}},
            [self, payload_rows, assembled, binding = *gather_binding, actual_reader](void *stream) {
                self->kernels_->publishExtent(payload_rows, assembled.count, binding.extent(actual_reader), stream);
                binding.enqueue(stream);
            }, ComputeStageType::PIPELINE_ACTIVATION_EXCHANGE, (gather_receipt.captureIdentity() ^ query.captureIdentity()) | 1u);
    } else if (reader_count > 1) {
        const auto name = prefix + "fixed_pipeline_gather";
        const auto stride = product(assembled.count, native_stride);
        attach(name, std::make_unique<NativeAllGatherStage>(NativeAllGatherStage::Params{
            .device_id = participant_.device, .tp_ctx = participant_.stage_context,
            .local_input = local.get(), .rank_major_output = gathered.get(), .bytes_per_participant = product(capacity, stride),
            .participant = actual_reader, .stage_name = name, .input_buffer_id = local_native_role, .output_buffer_id = gathered_native_role,
            .arena_rows = NativeAllGatherStage::ArenaRows{transport.get(), transport_role, 0, capacity, stride},
            .payload_receipt = gather_receipt}));
    }
    std::vector<Operand> decode_operands{{transport_role, transport, Access::Read}, {output.buffer, output_owner, Access::Write},
        {control_role, bank(control_role), Access::ReadWrite}};
    if (reader_count == 1 || packets) decode_operands.push_back({local_native_role, local, Access::Read});
    if (reader_count > 1) decode_operands.push_back({gathered_native_role, gathered, Access::Read});
    q::FixedHeadPipelineDecode decode{payload_rows, error, {}, reader_count, nullptr};
    for (int reader = 0; reader < reader_count; ++reader) {
        const auto &source = projection_->owner(projection_->consumers()[reader]);
        const auto heads = projection_->assembledHeadCount(reader);
        const bool owns_packet = reader_count == 1 || (packets && reader == actual_reader);
        const auto offset = owns_packet ? 0 : packets ? packets->receiveOffset(reader) : product(product(reader, capacity), product(heads, native_stride));
        decode.packets[reader] = {deviceAt<std::byte>(owns_packet ? local : gathered, offset),
            gather_binding ? gather_binding->extent(reader) : nullptr, {source.first_head, heads}};
    }
    add(terminal, std::move(decode_operands), [self, output_owner, decode](void *stream) mutable {
        decode.output = deviceAt<float>(output_owner);
        self->kernels_->decode(decode, stream);
    });
}
} // namespace llaminar2
