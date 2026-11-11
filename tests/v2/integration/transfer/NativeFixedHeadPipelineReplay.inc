/**
 * @file NativeFixedHeadPipelineReplay.inc
 * @brief Retained original GPU fixed-head request, provider and gather component replay.
 *
 * The original MPI inventory and prepared source own all workers, native
 * contexts and bytes. This component supplies fixture-authored original IDs
 * to the installed range kernels; the actual DGO token/history producer and
 * full model serving remain the responsibility of the real QFN matrix.
 */
#include "collective/DeviceCountedAllGather.h"
#include "execution/compute_stages/stages/NativeAllGatherStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/mtp/DeviceGenerationCollectiveReceipt.h"
#include "kernels/common/SamplingMath.h"
#include "kernels/qwen4exp/ImmutableFixedHeadLookupKernels.h"
#include "kernels/qwen4exp/ImmutableLookupKernels.h"
#include "loaders/ImmutableFixedHeadPipelineProjection.h"
#include "planning/ObservedPeerGrouping.h"
#include "utils/EmbeddingVerifierFormats.h"
#include <map>
#include <numeric>

namespace {
namespace fixed_replay {
namespace q = llaminar2::qwen4exp;
using Graph = test::RetainedParticipantGraphExecutableFixtureAccess;
using Tensor = std::shared_ptr<TensorBase>;
using Receipt = DeviceGenerationCollectiveReceipt;
using ReceiptKind = DeviceGenerationCollectiveReceiptKind;
using Metric = q::FixedHeadPipelineTerminalMetric;
using Controller = std::array<std::int32_t, sampling_math::kDeviceGenerationControlCount>;
constexpr int maximum_rows = 5;
constexpr int source_rows = 178;
constexpr auto workspace = PhysicalMemoryOwner::ExecutionWorkspace;
const CPUTensorStorageDeclaration controller_declaration(1, sampling_math::kDeviceGenerationControlCount,
    BufferTensorType::INT32, PhysicalMemoryOwner::ActivationArena, DeviceId::cpu());

/** @return One gap-preserving original source vocabulary shared by every range. */
q::NgramHashPolicy policy() {
    q::NgramHashPolicy result;
    result.multipliers = {1, 3, 7};
    std::uint64_t next = 0;
    for (int head = 0; head < q::kNgramHeads; ++head) {
        if (head == 8) next += 2;
        result.offsets[head] = next;
        result.moduli[head] = head + 3;
        next += result.moduli[head];
    }
    return result;
}

/** @brief One original source-format case and its precise retained graph widths. */
struct Format {
    const test::EmbeddingVerifierFormatCase* fixture;
    TensorType type;
    std::size_t stride;
    int width;
    bool repeated;
};

/** @return Fixture formats from the same canonical codec list as the existing native lookup replay. */
std::vector<Format> formats(bool all) {
    std::vector<Format> result;
    for (const auto& item : test::embeddingVerifierFormats()) {
        const bool iq4 = std::string_view(item.label) == "IQ4_NL";
        if (!all && !iq4) continue;
        const int width = iq4 ? 160 : 256;
        const auto row = item.create({1, static_cast<std::size_t>(width)}, 917);
        result.push_back({&item, row->native_type(), row->size_bytes(), width, iq4});
    }
    require(!result.empty(), "Fixed pipeline replay has no original codec");
    return result;
}

/** @return Exact retained definitions for this immutable source-format lifetime. */
std::vector<int> retainedWidths(const Format& format) {
    return format.repeated ? std::vector{maximum_rows, 1, 0} : std::vector{1};
}

/** @brief Each role's admission uses the exact typed root that will materialize it. */
CPUTensorStorageDeclaration declaration(std::size_t bytes) {
    return {1, (bytes + sizeof(std::int32_t) - 1) / sizeof(std::int32_t),
        BufferTensorType::INT32, workspace, DeviceId::cpu()};
}

/** @brief View only an admitted replay root's exact native base. */
template<class T> T* pointer(const Tensor& tensor, std::size_t offset = 0) {
    return reinterpret_cast<T*>(static_cast<std::byte*>(tensor->gpu_data_ptr()) + offset);
}

/** @brief Per-participant roots retain the independent original and transport count fields. */
struct ReaderBanks {
    Tensor ids, original, local_header, headers, transport, contribution, gathered, output;
};

/** @brief One real request/response bank on each endpoint, with disjoint row count roots. */
struct EdgeBanks {
    Tensor ids, native, request_rows, response_rows;
};

/** @brief Local logical roles share one original worker, stream and immutable table range. */
struct Participant {
    DeviceId device;
    std::shared_ptr<IWorkerGPUContext> worker;
    void* stream;
    const ImmutableFixedHeadPipelineOwner* owner;
    ReaderBanks reader;
    std::unique_ptr<BufferArena> arena;
    Tensor control_owner;
    std::optional<DeviceGenerationControlBinding> control;
    std::unique_ptr<IDeviceContext> context;
    PhysicalMemoryAllocationLease table_claim;
    std::shared_ptr<DeviceTransferBuffer> table;
    std::unique_ptr<q::IImmutableFixedHeadPipelineKernels> kernels;
};

/** @brief Distinct original stage/member coordinates and each addressed range's real consumer. */
struct Edge {
    int reader, provider;
    q::FixedHeadPipelineHeads heads;
    EdgeBanks reader_banks, provider_banks;
    std::shared_ptr<PipelineNativeByteChannel> request, response;
};

/** @brief Terminal frame evidence retains the original native channel and endpoint coordinates. */
struct Endpoint {
    std::shared_ptr<PipelineNativeByteChannel> channel;
    int consumer, participant;
    std::size_t stride;
    bool response;
    bool failing_provider;
};

/** @brief A small local test DAG retains real tensor publications while invoking actual native primitives. */
class Body final {
public:
    /** @brief Freeze one operation and its actual root dependencies before recording. */
    void add(std::string name, std::vector<Tensor> reads, std::vector<Tensor> writes,
        std::function<void()> submit, bool native_publishes = false) {
        nodes_.push_back({std::move(name), std::move(reads), std::move(writes), std::move(submit), native_publishes});
    }

    /** @return Original production capture dependency ledger for this exact local body. */
    std::unique_ptr<GraphCaptureDependencyLedger> ledger(DeviceId device, void* stream) const {
        std::map<const TensorBase*, std::size_t> producer;
        std::vector<GraphCaptureDependencyLedger::StagePlan> stages;
        for (const auto& node : nodes_) {
            GraphCaptureDependencyLedger::StagePlan stage{.stage_identity = &node, .stage_name = node.name};
            for (const auto& root : node.reads) {
                const auto prior = producer.find(root.get());
                if (prior == producer.end()) stage.external_inputs.push_back(root.get());
                else stage.internal_inputs.push_back({root.get(), prior->second});
            }
            for (const auto& root : node.writes) {
                producer.insert_or_assign(root.get(), stages.size());
                stage.outputs.push_back(root.get());
            }
            stages.push_back(std::move(stage));
        }
        return std::make_unique<GraphCaptureDependencyLedger>(device, stream, std::move(stages), "fixed native pipeline component");
    }

    /** @brief Replay the declared exact stream dependencies through the public transfer authority. */
    void record(DeviceId device, void* stream) const {
        for (const auto& node : nodes_) {
            ScopedGraphCaptureStage stage(&node);
            for (const auto& root : node.reads) TransferEngine::requireDeviceInput(root.get(), device, stream);
            for (const auto& root : node.writes) TransferEngine::requireDeviceOutput(root.get(), device, stream);
            node.submit();
            if (!node.native_publishes)
                for (const auto& root : node.writes) TransferEngine::publishDeviceWrite(root.get(), device, stream);
            stage.complete();
        }
    }

private:
    struct Node {
        std::string name;
        std::vector<Tensor> reads, writes;
        std::function<void()> submit;
        bool native_publishes;
    };
    std::vector<Node> nodes_;
};

/** @return Exact independent reader allocations, including guardable capacity behind every smaller replay. */
std::vector<std::size_t> readerSizes(const Format& format, int heads, int readers, bool unequal) {
    return {maximum_rows * q::kNgramHeads * sizeof(std::int64_t), sizeof(std::int32_t),
        sizeof(q::FixedHeadValidationHeader),
        readers>1?sizeof(q::FixedHeadValidationHeader)*readers:0, sizeof(std::int32_t),
        maximum_rows * heads * format.stride,
        readers>1?maximum_rows*(q::kNgramHeads-(unequal?heads:0))*format.stride:0,
        maximum_rows*q::kNgramHeads*format.width*sizeof(float)};
}

/** @return Each endpoint's compact request/response plus disjoint receipt-published count roots. */
std::array<std::size_t, 4> edgeSizes(const Format& format, int heads) {
    return {maximum_rows * heads * sizeof(std::int64_t), maximum_rows * heads * format.stride,
        sizeof(std::int32_t), sizeof(std::int32_t)};
}

/** @return Exact original byte allocation interval with source gaps and the trailing row tail intact. */
std::pair<std::uint64_t, std::uint64_t> originalRange(const ImmutableFixedHeadPipelineOwner& owner,
    const q::NgramHashPolicy& original) {
    const int end_head = owner.first_head + owner.head_count;
    return {owner.first_head == 0 ? 0 : original.offsets[owner.first_head],
        end_head == q::kNgramHeads ? source_rows : original.offsets[end_head]};
}

/** @brief Build one native LocalTP over original physical addresses and no additional rank namespace. */
std::unique_ptr<LocalTPContext> localContext(std::span<const DeviceId> devices, CollectiveBackendType backend) {
    std::vector<GlobalDeviceAddress> addresses;
    for (auto device : devices) addresses.push_back(GlobalDeviceAddress::fromLocalDeviceId(device));
    return std::make_unique<LocalTPContext>(addresses, std::vector<float>{}, backend);
}

/** @brief Retain one exact source format plus its lifetime, without pretending to load a complete model. */
void prepareSource(Participant& participant, const Format& format, const q::NgramHashPolicy& original,
    const TensorBase& source, const test::PreparedNativeLookupFixture& owner) {
    const auto [begin, end] = originalRange(*participant.owner, original);
    const auto bytes = (end - begin) * format.stride;
    participant.table_claim = owner.memory()->claimNewAllocation(participant.device, PhysicalMemoryOwner::ImmutableTableStorage, bytes);
    const auto generation = owner.permit(participant.device).generation();
    participant.worker->submitAndWait([&] {
        participant.table = TransferEngine::instance().allocateDeviceTransferBuffer(bytes, participant.device);
        require(participant.table&&participant.table->isBound(),"Fixed pipeline source native storage allocation failed");
        auto* backend = getBackendFor(participant.device);
        const auto* source_bytes = static_cast<const std::byte*>(source.raw_data());
        require(backend->hostToDeviceOnStream(participant.table->mutableDeviceData(), source_bytes + begin * format.stride,
            bytes, participant.device.ordinal, participant.stream) &&
            participant.worker->synchronizeStreamChecked(participant.stream), "Fixed pipeline original source upload failed");
        const auto view = q::ImmutableResidentTableRangeView::bind(participant.device, format.type,
            static_cast<const std::byte*>(participant.table->deviceData()), source_rows, begin, end - begin,
            format.width, format.stride, bytes, generation);
        participant.kernels = q::createImmutableFixedHeadPipelineKernels(participant.device, view, original,
            {participant.owner->first_head, participant.owner->head_count});
        if (participant.owner->first_head != 0)
            EXPECT_THROW((void)q::createImmutableFixedHeadPipelineKernels(participant.device, view, original,
                {0, participant.owner->head_count}), std::invalid_argument);
    });
}

/** @brief Retain the same arena-counted allgather stage used by the serving graph. */
void addGather(Body& body, const Participant& p, LocalTPContext& readers, int reader,
    std::string name, const Tensor& local, BufferId local_role, const Tensor& received,
    BufferId received_role, int capacity, std::size_t stride, std::optional<Receipt> receipt = {}) {
    auto stage = std::make_shared<NativeAllGatherStage>(NativeAllGatherStage::Params{
        .device_id = p.device, .tp_ctx = &readers, .local_input = local.get(),
        .rank_major_output = received.get(), .bytes_per_participant = capacity * stride,
        .participant = reader, .stage_name = name, .input_buffer_id = local_role,
        .output_buffer_id = received_role,
        .arena_rows = NativeAllGatherStage::ArenaRows{p.reader.transport.get(),
            BufferId::QFN_NGRAM_SHARD_TRANSPORT_ROWS, 0, capacity, stride},
        .payload_receipt = receipt});
    p.worker->submitAndWait([&] {
        stage->setGPUStream(p.stream);
        require(stage->prepareGraphLaunch(p.context.get(), p.stream),
            "Fixed pipeline native reader gather could not seal its original count");
    });
    std::vector<Tensor> reads{local,p.reader.transport};
    std::vector<Tensor> writes{received};
    if (receipt) {
        reads.push_back(p.control_owner);
        writes.push_back(p.control_owner);
    }
    body.add(std::move(name),std::move(reads),std::move(writes),[&p,stage] {
        require(stage->execute(p.context.get()),"Fixed pipeline original reader gather failed");
    },true);
}

/** @brief Record the provider's real receive, checked native pack, and completed reply. */
void recordProvider(Body& body, const Participant& p, const Edge& edge, int coordinate, int exact,
    std::size_t stride) {
    const auto b = edge.provider_banks;
    const auto request_stride = edge.heads.count * sizeof(std::int64_t);
    const auto reply_stride = edge.heads.count * stride;
    const auto wire_rows = std::max(exact, 1);
    const CapturedTransferMessage request{0x51464e1000000000ULL | std::uint32_t(exact), wire_rows * request_stride};
    const CapturedTransferMessage reply{0x51464e2000000000ULL | std::uint32_t(exact), wire_rows * reply_stride};
    const auto acquire = PipelineNativeByteBinding(edge.request, coordinate, request, b.ids, 0,
        PipelineNativeReceivedRows{b.request_rows, 0, request_stride})
        .withTerminalReceipt(Receipt::fixedHeadPipeline(*p.control,ReceiptKind::FixedHeadPipelineRequest,
            wire_rows,edge.heads.count,sizeof(std::int64_t)));
    const auto send = PipelineNativeByteBinding(edge.response, coordinate, reply, b.native, 0,
        PipelineNativeRowCount{b.request_rows, 0, reply_stride})
        .withTerminalReceipt(Receipt::fixedHeadPipeline(*p.control,ReceiptKind::FixedHeadPipelineResponse,
            wire_rows,edge.heads.count,stride));
    auto* count = pointer<std::int32_t>(b.request_rows);
    const q::FixedHeadPipelineError error{p.control->healthy(),p.control->error(),{p.control->row()}};
    body.add("provider request", {p.control_owner}, {b.ids,b.request_rows,p.control_owner},
        [acquire,stream=p.stream] { acquire.enqueue(stream); },true);
    body.add("provider pack and seal", {b.ids,b.request_rows,p.control_owner}, {b.native,b.request_rows,p.control_owner},
        [&p,b,count,error,exact,heads=edge.heads.count] {
            p.kernels->pack({{exact,count},error,pointer<std::int64_t>(b.ids),pointer<std::byte>(b.native),
                heads,0,heads,0},p.stream);
            p.kernels->sealResponse(count,error,p.stream);
        });
    body.add("provider reply", {b.native,b.request_rows,p.control_owner}, {p.control_owner},
        [send,stream=p.stream] { send.enqueue(stream); },true);
}

/** @brief Record compact actual range requests before joining and validating all provider replies. */
void recordReader(Body& body, const Participant& p, std::span<const Edge> edges, int coordinate,
    int exact, const ImmutableFixedHeadPipelineProjection& projection, std::size_t stride,
    LocalTPContext* readers, const std::shared_ptr<DeviceCountedAllGather>& variable) {
    const auto& b = p.reader;
    const int reader = p.owner->consumer_index;
    const auto population = static_cast<int>(projection.consumers().size());
    const auto assembled = projection.assembledHeadCount(reader);
    const auto wire_rows = std::max(exact, 1);
    const q::FixedHeadPipelineError error{p.control->healthy(),p.control->error(),{p.control->row()}};
    const q::FixedHeadPipelineRows payload{exact,pointer<std::int32_t>(b.transport)};
    const q::FixedHeadPipelineConsensus consensus{
        {exact,pointer<std::int32_t>(b.original)},error,pointer<q::FixedHeadValidationHeader>(b.local_header),
        pointer<q::FixedHeadValidationHeader>(readers?b.headers:b.local_header),pointer<std::int32_t>(b.transport),population};
    body.add("original count admission", {b.original,p.control_owner}, {b.transport,p.control_owner},
        [&p,consensus] { p.kernels->prepareConsumer(consensus,p.stream); });
    for (const auto& edge : edges) {
        if (edge.reader != coordinate) continue;
        const auto wire = edge.reader_banks;
        const auto request_stride = edge.heads.count * sizeof(std::int64_t);
        const CapturedTransferMessage request{0x51464e1000000000ULL | std::uint32_t(exact), wire_rows * request_stride};
        const auto send = PipelineNativeByteBinding(edge.request, coordinate, request, wire.ids, 0,
            PipelineNativeRowCount{b.transport, 0, request_stride})
            .withTerminalReceipt(Receipt::fixedHeadPipeline(*p.control,ReceiptKind::FixedHeadPipelineRequest,
                wire_rows,edge.heads.count,sizeof(std::int64_t)));
        body.add("compact original IDs", {b.ids,b.transport}, {wire.ids},
            [&p,wire,payload,heads=edge.heads] {
                p.kernels->projectRequest({payload,heads,pointer<std::int64_t>(p.reader.ids),
                    pointer<std::int64_t>(wire.ids)},p.stream);
            });
        body.add("reader request", {wire.ids,b.transport,p.control_owner}, {p.control_owner},
            [send,stream=p.stream] { send.enqueue(stream); },true);
    }
    body.add("reader local native range", {b.ids,b.transport,p.control_owner}, {b.contribution,p.control_owner},
        [&p,payload,error,assembled] {
            p.kernels->pack({payload,error,pointer<std::int64_t>(p.reader.ids),
                pointer<std::byte>(p.reader.contribution),q::kNgramHeads,p.owner->first_head,assembled,0},p.stream);
        });
    for (const auto& edge : edges) {
        if (edge.reader != coordinate) continue;
        const auto wire = edge.reader_banks;
        const auto reply_stride = edge.heads.count * stride;
        const CapturedTransferMessage reply{0x51464e2000000000ULL | std::uint32_t(exact), wire_rows * reply_stride};
        const auto receive = PipelineNativeByteBinding(edge.response, coordinate, reply, wire.native, 0,
            PipelineNativeReceivedRows{wire.response_rows, 0, reply_stride})
            .withTerminalReceipt(Receipt::fixedHeadPipeline(*p.control,ReceiptKind::FixedHeadPipelineResponse,
                wire_rows,edge.heads.count,stride));
        body.add("reader completed response", {p.control_owner}, {wire.native,wire.response_rows,p.control_owner},
            [receive,stream=p.stream] { receive.enqueue(stream); }, true);
        body.add("reader verify and merge", {wire.native,wire.response_rows,b.transport,p.control_owner,b.contribution},
            {p.control_owner,b.contribution}, [&p,wire,payload,error,assembled,heads=edge.heads] {
                const auto* received = pointer<std::int32_t>(wire.response_rows);
                p.kernels->validateResponse(payload,received,error,p.stream);
                p.kernels->merge({{payload.exact,received},error,pointer<std::byte>(wire.native),pointer<std::byte>(p.reader.contribution),
                    heads.count,assembled,heads.first-p.owner->first_head},p.stream);
            });
    }
    body.add("reader seal post-provider status", {b.original,p.control_owner,b.transport},
        {b.local_header,b.transport,p.control_owner},
        [&p,consensus] { p.kernels->sealConsumer(consensus,p.stream); });
    if (readers) addGather(body,p,*readers,reader,"fixed pipeline replay status",
        b.local_header,BufferId::QFN_NGRAM_SHARD_LOCAL_HEADER,b.headers,
        BufferId::QFN_NGRAM_SHARD_HEADERS,1,sizeof(q::FixedHeadValidationHeader));
    body.add("reader error consensus before payload", {readers?b.headers:b.local_header,b.original,p.control_owner},
        {b.transport,p.control_owner}, [&p,consensus] { p.kernels->validateConsensus(consensus,p.stream); });

    const auto gather_receipt = Receipt::fixedHeadPipeline(*p.control,ReceiptKind::FixedHeadPipelineGather,
        wire_rows,assembled,stride);
    std::optional<DeviceCountedAllGatherBinding> binding;
    std::shared_ptr<const DeviceCountedAllGatherPacketLayout> packets;
    if (variable) {
        std::vector<std::size_t> capacities;
        for (int index=0;index<population;++index)
            capacities.push_back(wire_rows * projection.assembledHeadCount(index) * stride);
        packets = std::make_shared<const DeviceCountedAllGatherPacketLayout>(capacities,reader);
        binding = variable->bindPackets(0x51464e3000000000ULL | std::uint32_t(exact),packets,b.contribution,b.gathered,gather_receipt);
        body.add("reader uneven useful native gather", {b.contribution,b.transport,p.control_owner}, {b.gathered,p.control_owner},
            [&p,payload,assembled,reader,native=*binding] {
                p.kernels->publishExtent(payload,assembled,native.extent(reader),p.stream);
                native.enqueue(p.stream);
            },true);
    } else if (readers) {
        addGather(body,p,*readers,reader,"fixed pipeline replay payload",
            b.contribution,BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE,b.gathered,BufferId::QFN_NGRAM_SHARD_GATHERED_NATIVE,
            wire_rows,assembled*stride,gather_receipt);
    }
    q::FixedHeadPipelineDecode decode{payload,error,{},population,pointer<float>(b.output)};
    std::vector<Tensor> decode_inputs{b.transport,p.control_owner};
    if (!readers || packets) decode_inputs.push_back(b.contribution);
    if (readers) decode_inputs.push_back(b.gathered);
    for (int source=0;source<population;++source) {
        const auto& owner = projection.owner(projection.consumers()[source]);
        const bool own = !readers || (packets && source==reader);
        const auto heads = projection.assembledHeadCount(source);
        const auto offset = own?0:packets?packets->receiveOffset(source):source*wire_rows*heads*stride;
        decode.packets[source]={pointer<std::byte>(own?b.contribution:b.gathered,offset),
            binding?binding->extent(source):nullptr,{owner.first_head,heads}};
    }
    body.add("reader decode original head order",std::move(decode_inputs),{p.control_owner,b.output},
        [&p,decode] { p.kernels->decode(decode,p.stream); });
}

/** @brief Upload fixture-owned canonical input bytes only at a retired request boundary. */
void upload(const Participant& p, const Tensor& root, const void* bytes, std::size_t size) {
    require(size <= root->size_bytes(), "Fixed pipeline replay upload exceeds its typed root");
    std::memcpy(root->raw_mutable_data(),bytes,size);
    TransferEngine::publishHostWrite(root.get());
    TransferEngine::prepareDeviceInput(root.get(),p.device,p.stream);
}

/** @brief Copy actual terminal evidence after all original graph submissions have joined. */
void read(const Participant& p, void* output, const void* input, std::size_t bytes) {
    auto* backend=getBackendFor(p.device);
    require(backend->deviceToHostOnStream(output,input,bytes,p.device.ordinal,p.stream) &&
        p.worker->synchronizeStreamChecked(p.stream),"Fixed pipeline replay terminal observation failed");
}

/** @brief Check fresh native receipts and primitive payload bytes separately from decoded numerics. */
void observeEndpoint(const Participant& p, const Endpoint& endpoint, LocalTPContext& pipeline,
    std::uint64_t sequence,
    int exact, std::size_t live) {
    const auto [device,frame_address]=test::PipelineNativeByteChannelReplayProbe::frame(
        *endpoint.channel,endpoint.consumer,endpoint.participant);
    const NativeCountedAllGatherLayout layout(1);
    std::vector<std::byte> image(layout.bytes());
    require(device==p.device,"Fixed pipeline terminal endpoint changed its actual device");
    read(p,image.data(),frame_address,image.size());
    NativeCountedAllGatherFrame frame{};
    ncclLlaminarByteCompletion receipt{};
    std::uint64_t extent=0,traffic=0;
    std::memcpy(&frame,image.data(),sizeof(frame));
    std::memcpy(&receipt,image.data()+layout.receipts(),sizeof(receipt));
    std::memcpy(&extent,image.data()+layout.extents(),sizeof(extent));
    std::memcpy(&traffic,image.data()+layout.counters(),sizeof(traffic));
    const auto bytes=live*endpoint.stride;
    EXPECT_EQ(frame.first_error,0);
    EXPECT_EQ(frame.sequence,sequence);
    EXPECT_EQ(extent,bytes);
    const auto authority=pipeline.nativeCompletionAuthority(endpoint.participant).binding();
    EXPECT_EQ(receipt.nativeAbortFlag,authority.abortFlag);
    EXPECT_EQ(receipt.nativeCommunicator,authority.deviceCommunicator);
    EXPECT_EQ(receipt.sequence,sequence);
    EXPECT_EQ(receipt.phase,ncclLlaminarCompletionPhase::Completed);
    EXPECT_EQ(receipt.table,(endpoint.response?0x51464e2000000000ULL:0x51464e1000000000ULL)|std::uint32_t(exact));
    EXPECT_EQ(receipt.admission,std::max(exact,1)*endpoint.stride);
    EXPECT_EQ(receipt.bytes,bytes);
    // The typed receipt routes native sends into the canonical control row.
    // The unbound frame's alternate diagnostic counter must remain untouched.
    EXPECT_EQ(traffic,0u);
}

/** @brief Inspect actual decode launches; zero rows have no data-grid node. */
void inspect(IGPUGraphCapture& definition,int exact,bool reader) {
    std::vector<GPUGraphKernelNodeInfo> nodes;
    std::string detail;
    require(definition.inspectKernelNodes(nodes,&detail),"Fixed pipeline graph kernel inspection failed");
    std::size_t decodes=0;
    for(const auto& node:nodes) {
        if(node.name.find("decodePipelinePackets")==std::string::npos)continue;
        ++decodes;
        EXPECT_EQ(node.grid_x,exact*q::kNgramHeads);
        EXPECT_EQ(node.block_x,128u);
        EXPECT_EQ(node.local_memory_bytes_per_thread,0u);
    }
    EXPECT_EQ(decodes,reader&&exact>0?1u:0u);
}

/** @brief Execute one immutable source format against all retained exact-width definitions. */
void replayFormat(const Format& format,const ImmutableFixedHeadPipelineProjection& projection,
    const RankInventory& inventory,const std::vector<DeviceId>& devices,
    LocalTPContext& pipeline, LocalTPContext* readers,bool unequal,
    const test::PreparedNativeLookupFixture& original) {
    SCOPED_TRACE(format.fixture->label);
    const auto memory=original.memory();
    const auto metadata=policy();
    const auto source=format.fixture->create({source_rows,static_cast<std::size_t>(format.width)},917);
    require(source->native_type()==format.type&&source->size_bytes()==source_rows*format.stride,
        "Fixed pipeline codec source geometry changed");
    const auto* native=static_cast<const std::byte*>(source->raw_data());
    std::vector<Participant> people;
    std::vector<void*> streams,reader_streams;
    std::vector<DeviceId> reader_devices;
    const auto allocate=[&](const Participant& p, std::size_t bytes)->Tensor {
        if(!bytes)return {};
        Tensor result=declaration(bytes).materialize(memory);
        p.worker->submitAndWait([&] {
            TransferEngine::prepareDeviceInput(result.get(),p.device,p.stream);
            TransferEngine::requireDeviceInput(result.get(),p.device,p.stream);
            require(p.worker->synchronizeStreamChecked(p.stream),"Fixed pipeline setup root did not retire its upload");
        });
        return result;
    };
    for(const auto device:devices) {
        auto worker=original.permit(device).worker();
        const auto& owner=projection.owner({inventory.rank,device});
        people.push_back({device,worker,worker->defaultStream(),&owner});
        auto& person=people.back();
        streams.push_back(person.stream);
        person.arena=std::make_unique<BufferArena>(ArenaConfig{.physical_memory_authority=memory});
        person.control_owner=controller_declaration.materialize(memory);
        person.worker->submitAndWait([&] {
            TransferEngine::prepareDeviceInput(person.control_owner.get(),device,person.stream);
            TransferEngine::requireDeviceInput(person.control_owner.get(),device,person.stream);
            require(person.arena->registerExternalBuffer(BufferId::MTP_GENERATION_CONTROL,person.control_owner),
                "Fixed pipeline replay lost its canonical control arena owner");
            person.control.emplace(DeviceGenerationControlBinding::admit(*person.arena,device,0));
            person.context=IDeviceContext::create(device,1);
            require(person.context&&person.worker->synchronizeStreamChecked(person.stream),
                "Fixed pipeline controller setup did not finish its original upload");
        });
        if(owner.stage_index==projection.consumerStage()) {
            reader_devices.push_back(device);
            reader_streams.push_back(person.stream);
            std::vector<Tensor> banks;
            for(const auto bytes:readerSizes(format,projection.assembledHeadCount(owner.consumer_index),
                static_cast<int>(projection.consumers().size()),unequal))banks.push_back(allocate(person,bytes));
            person.reader={banks[0],banks[1],banks[2],banks[3],banks[4],banks[5],banks[6],banks[7]};
        }
        prepareSource(person,format,metadata,*source,original);
    }
    const auto coordinate=[&](DeviceId device) {
        const auto found=std::find(devices.begin(),devices.end(),device);
        require(found!=devices.end(),"Fixed pipeline owner is absent from the original native domain");
        return static_cast<int>(found-devices.begin());
    };
    std::vector<Edge> edges;
    std::vector<Endpoint> endpoints;
    for(const auto& owner:projection.owners()) {
        if(owner.stage_index==projection.consumerStage())continue;
        const int reader=coordinate(projection.consumers()[owner.consumer_index].device);
        const int provider=coordinate(owner.allocator.device);
        const auto sizes=edgeSizes(format,owner.head_count);
        const auto allocate_wire=[&](int member,bool supplies) {
            const auto& p=people[member];
            return EdgeBanks{allocate(p,sizes[0]),allocate(p,sizes[1]),
                supplies?allocate(p,sizes[2]):Tensor{},supplies?Tensor{}:allocate(p,sizes[3])};
        };
        const std::array send_to{provider};
        const std::array reply_to{reader};
        auto request=PipelineNativeByteChannel::create(pipeline,reader,send_to,sizes[0],memory,streams);
        auto response=PipelineNativeByteChannel::create(pipeline,provider,reply_to,sizes[1],memory,streams);
        const bool fault_target=edges.empty();
        for(int member:{reader,provider}) {
            endpoints.push_back({request,provider,member,sizes[0]/maximum_rows,false,fault_target});
            endpoints.push_back({response,reader,member,sizes[1]/maximum_rows,true,fault_target});
        }
        edges.push_back({reader,provider,{owner.first_head,owner.head_count},allocate_wire(reader,false),
            allocate_wire(provider,true),std::move(request),std::move(response)});
    }
    require(!edges.empty(),"Fixed pipeline replay omitted every actual provider");
    const auto gather=unequal?DeviceCountedAllGather::create(*memory,*readers,DeviceId::cpu(),
        DeviceCountedAllGatherTopology::resolve(inventory,reader_devices),reader_streams,
        maximum_rows*projection.maximumAssembledHeads()*format.stride):nullptr;
    const auto await=[&] {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(
            collective_timeout_policy::kDefaultCollectiveTimeoutMs);
        for(const auto& p:people)for(;;) {
            GPUStreamExecutionState state=GPUStreamExecutionState::Pending;
            p.worker->submitAndWait([&]{state=p.worker->queryStreamExecutionState(p.stream,"fixed pipeline native terminal");});
            if(state==GPUStreamExecutionState::Complete)break;
            require(state==GPUStreamExecutionState::Pending&&std::chrono::steady_clock::now()<deadline,
                "Fixed pipeline original stream did not complete");
            std::this_thread::yield();
        }
    };
    const auto abort=[&] {
        require(pipeline.signalNativeCompletionAbort(),"Fixed pipeline original abort failed");
        if(readers)require(readers->signalNativeCompletionAbort(),"Fixed pipeline original reader abort failed");
    };
    struct Family {
        int exact;
        std::vector<Body> bodies;
        std::vector<std::shared_ptr<RetainedParticipantGraphExecutable>> graphs;
        std::uint64_t submitted=0;
    };
    std::vector<Family> families;
    NativeReplayFailureDrain drain{[&] {
        abort();
        await();
        for(auto& family:families)
            for(const auto& graph:family.graphs)
                require(Graph::observeTerminal(graph),"Fixed pipeline failed definition has no original terminal");
        families.clear();
    }};
    const auto widths=retainedWidths(format);
    for(const int exact:widths) {
        auto& family=families.emplace_back(Family{exact,std::vector<Body>(devices.size()),{}});
        std::vector<std::unique_ptr<GraphCaptureDependencyLedger>> ledgers;
        std::vector<OriginalNativeGraphRecording> jobs;
        for(std::size_t member=0;member<people.size();++member) {
            const auto& p=people[member];
            auto& body=family.bodies[member];
            if(p.reader.ids)recordReader(body,p,edges,member,exact,projection,format.stride,readers,gather);
            else {
                const auto edge=std::find_if(edges.begin(),edges.end(),[&](const auto& wire){return wire.provider==int(member);});
                require(edge!=edges.end(),"Fixed pipeline provider has no actual addressed range");
                recordProvider(body,p,*edge,member,exact,format.stride);
            }
            auto graph=Graph::original(original.permit(p.device),p.stream);
            ledgers.push_back(body.ledger(p.device,p.stream));
            jobs.push_back({p.worker,p.device,&Graph::capture(graph),ledgers.back().get(),
                [&body,&p]{body.record(p.device,p.stream);}});
            family.graphs.push_back(std::move(graph));
        }
        OriginalNativeGraphCaptureGroup::record(jobs,pipeline.borrowOriginalNativeContext(),"fixed pipeline original component");
        for(std::size_t member=0;member<people.size();++member)people[member].worker->submitAndWait([&,member] {
            auto& graph=Graph::capture(family.graphs[member]);
            inspect(graph,exact,bool(people[member].reader.ids));
            require(graph.prepareRuntimeContextStorage(memory)&&graph.instantiate(),"Fixed pipeline original definition could not instantiate");
        });
    }

    // A retained tensor lease must reject another original allocation even if
    // it is the same device, dtype, size and prepared worker generation.
    const auto& first_reader=people[edges.front().reader];
    const auto captured=TransferEngine::instance().retainCapturedDeviceStorage(*first_reader.reader.original,
        first_reader.device,sizeof(std::int32_t));
    EXPECT_THROW(TransferEngine::instance().requireCapturedDeviceStorage(*captured,*first_reader.reader.transport,
        first_reader.device,sizeof(std::int32_t)),std::logic_error);
    EXPECT_THROW(first_reader.control->requireFor(people[edges.front().provider].device),std::invalid_argument);
    Controller clean{};
    require(sampling_math::initialize_device_generation_control(0,1,
        sampling_math::DeviceGenerationPolicy::forwardOnly(),clean.data(),
        sampling_math::DeviceGenerationLeadingRowDisposition::AlreadyEmitted),
        "Fixed pipeline fixture could not create an original initial control row");
    std::vector<q::FixedHeadPipelineTerminalTotals> expected_controller(people.size());
    std::vector<std::uint64_t> controller_identities;
    for (const auto& person : people) controller_identities.push_back(person.control->captureIdentity());
    const auto replay_count=format.repeated?20u:1u;
    std::vector<std::uint64_t> variable_gather_totals(people.size());
    std::uint64_t expected_variable_gather=0;
    const std::vector<float> output_poison(maximum_rows*q::kNgramHeads*format.width,-12345.F);
    auto previous_output=output_poison;
    constexpr auto storage_failure=static_cast<int>(sampling_math::DeviceGenerationError::ImmutableStorageFailure);
    for(std::uint64_t replay=1;replay<=replay_count;++replay) {
        auto& family=families[(replay-1)%families.size()];
        const int exact=family.exact;
        const bool bad=format.repeated&&exact>0&&(replay==5||replay==13);
        SCOPED_TRACE(::testing::Message()<<"exact="<<exact<<" replay="<<replay<<" provider_failure="<<bad);
        std::vector<std::int64_t> ids(maximum_rows*q::kNgramHeads,-1);
        std::vector<std::byte> canonical(exact*q::kNgramHeads*format.stride);
        for(int row=0;row<exact;++row)for(int head=0;head<q::kNgramHeads;++head) {
            const auto index=row*q::kNgramHeads+head;
            const auto id=metadata.offsets[head]+((replay+row*7+head)%metadata.moduli[head]);
            ids[index]=id;
            std::memcpy(canonical.data()+index*format.stride,native+id*format.stride,format.stride);
        }
        if(bad) {
            const auto head=edges.front().heads.first;
            ids[(exact-1)*q::kNgramHeads+head]=metadata.offsets[head]+metadata.moduli[head];
        }
        const bool preserve_previous=exact==0&&replay==3;
        std::vector<float> expected=preserve_previous?previous_output:output_poison;
        if(exact&&!bad)q::decodeCPUImmutableLookupRows(format.type,format.width,format.stride,
            canonical.data(),exact*q::kNgramHeads,expected.data());
        // One original controller window spans changing exact graph widths.
        // Publish a fresh fixture input only on the first request and after a
        // failed request. Native transport frames continue their own sequence.
        const bool reset=replay==1||replay==6||replay==14;
        if (reset) {
            for (auto& totals : expected_controller) totals.fill(0);
            std::fill(variable_gather_totals.begin(),variable_gather_totals.end(),0);
            expected_variable_gather=0;
        }
        for(const auto& p:people)p.worker->submitAndWait([&] {
            const auto& b=p.reader;
            const auto poison=[&](const Tensor& tensor,int value) {
                if(!tensor)return;
                require(getBackendFor(p.device)->memset(tensor->gpu_data_ptr(),value,tensor->size_bytes(),
                    p.device.ordinal,p.stream),"Fixed pipeline test-only poison failed");
                TransferEngine::publishDeviceWrite(tensor.get(),p.device,p.stream);
            };
            if (reset) upload(p,p.control_owner,clean.data(),sizeof(clean));
            if(!b.ids)return;
            upload(p,b.ids,ids.data(),ids.size()*sizeof(ids[0]));
            upload(p,b.original,&exact,sizeof(exact));
            if(!preserve_previous)upload(p,b.output,output_poison.data(),output_poison.size()*sizeof(float));
            poison(b.gathered,0xa5);
            poison(b.contribution,0x6d);
            poison(b.transport,0xf1);
        });
        std::mutex failure_mutex;
        std::exception_ptr failure;
        const auto fail=[&](std::exception_ptr error) noexcept {
            {std::lock_guard lock(failure_mutex);if(!failure)failure=std::move(error);}
            try{abort();}catch(...){std::terminate();}
        };
        std::vector<std::future<void>> pending;
        const auto ordinal=++family.submitted;
        try {
            for(std::size_t member=0;member<people.size();++member)
                pending.push_back(people[member].worker->submitAsync([&,member] {
                    try {require(Graph::submit(family.graphs[member],ordinal),"Fixed pipeline retained native graph submission failed");}
                    catch(...){fail(std::current_exception());throw;}
                }));
        } catch(...){fail(std::current_exception());}
        joinOriginalNativeWorkerSubmissions(pending,fail,std::chrono::steady_clock::now()+
            std::chrono::milliseconds(collective_timeout_policy::kDefaultCollectiveTimeoutMs));
        if(failure)std::rethrow_exception(failure);
        await();
        for(const auto& graph:family.graphs)require(Graph::observeTerminal(graph),"Fixed pipeline graph terminal proof missing");
        for(std::size_t member=0;member<people.size();++member) {
            const auto& p=people[member];
            p.worker->submitAndWait([&,member] {
                const auto& b=p.reader;
                const bool reader=bool(b.ids);
                const bool failed_provider=bad&&int(member)==edges.front().provider;
                const bool failed_request=bad&&(reader||failed_provider);
                const bool rejected_local=failed_provider||(bad&&int(member)==edges.front().reader);
                auto& totals=expected_controller[member];
                const auto add=[&](Metric field,std::uint64_t delta) {
                    totals[static_cast<std::size_t>(field)]+=delta;
                };
                const auto owned_ids=rejected_local?0:exact*p.owner->head_count;
                add(Metric::Calls,1);
                add(Metric::LiveIds,owned_ids);
                add(Metric::NativeBytes,owned_ids*format.stride);
                add(Metric::Failures,failed_request?1:0);
                add(Metric::CompletedReceipts,projection.endpointCount(p.owner->allocator));
                if (reader) {
                    std::uint64_t requested_heads=0;
                    for (const auto& edge : edges)
                        if (edge.reader==int(member)) requested_heads+=edge.heads.count;
                    add(Metric::RequestBytes,exact*requested_heads*sizeof(std::int64_t));
                } else {
                    add(Metric::ResponseBytes,owned_ids*format.stride);
                }
                Controller control{};
                read(p,control.data(),p.control->row(),sizeof(control));
                constexpr auto start=q::kFixedHeadPipelineFirstWord;
                constexpr auto end=start+q::kFixedHeadPipelineCounterWords;
                for (std::size_t word=0;word<control.size();++word) {
                    if (word>=start&&word<end) continue;
                    auto expected_word=clean[word];
                    if (failed_request&&word==sampling_math::kDeviceGenerationControlOk) expected_word=0;
                    if (failed_request&&word==sampling_math::kDeviceGenerationControlErrorCode) expected_word=storage_failure;
                    EXPECT_EQ(control[word],expected_word)<<"controller word="<<word;
                }
                p.control->requireRequest(p.device,0);
                EXPECT_EQ(p.control->captureIdentity(),controller_identities[member]);
                std::array<std::int32_t,q::kFixedHeadPipelineCounterWords> observed_words{};
                std::copy_n(control.begin()+start,observed_words.size(),observed_words.begin());
                const auto observed_controller=q::readFixedHeadPipelineCounters(observed_words);
                if (!reader) {
                    EXPECT_EQ(observed_controller,totals);
                    return;
                }
                std::vector<float> actual(expected.size());
                read(p,actual.data(),b.output->gpu_data_ptr(),actual.size()*sizeof(float));
                EXPECT_EQ(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),0);
                std::int32_t live=-1;
                read(p,&live,b.transport->gpu_data_ptr(),sizeof(live));
                EXPECT_EQ(live,bad?0:exact);
                const auto reader_index=p.owner->consumer_index;
                if(gather)for(std::size_t root=0;root<reader_devices.size();++root) {
                    std::uint64_t extent=0;
                    read(p,&extent,gather->extentAddress(reader_index,root),sizeof(extent));
                    const auto useful=(bad?0:exact)*projection.assembledHeadCount(root)*format.stride;
                    EXPECT_EQ(extent,useful);
                    if(static_cast<int>(root)!=reader_index) {
                        ncclLlaminarByteCompletion receipt{};
                        read(p,&receipt,gather->completionAddress(reader_index,root),sizeof(receipt));
                        EXPECT_EQ(receipt.bytes,useful);
                        EXPECT_EQ(receipt.phase,ncclLlaminarCompletionPhase::Completed);
                        EXPECT_EQ(receipt.sequence,replay);
                    }
                    unsigned long long traffic=0;
                    read(p,&traffic,gather->payloadBytesAddress(reader_index,root),sizeof(traffic));
                    EXPECT_EQ(traffic,0u); // The typed native gather writes only the control-row observation.
                } else if(readers) {
                    add(Metric::GatherBytes,(bad?0:exact)*projection.assembledHeadCount(reader_index)*format.stride*
                        (reader_devices.size()-1));
                }
                if (gather) {
                    const auto index=static_cast<std::size_t>(Metric::GatherBytes);
                    const auto observed=observed_controller[index];
                    if (bad||exact==0) EXPECT_EQ(observed,variable_gather_totals[member]);
                    EXPECT_GE(observed,variable_gather_totals[member]);
                    variable_gather_totals[member]=observed;
                    for (std::size_t field=0;field<totals.size();++field)
                        if (field!=index) EXPECT_EQ(observed_controller[field],totals[field])<<"controller metric="<<field;
                } else {
                    EXPECT_EQ(observed_controller,totals);
                }
                if(readers) {
                    std::vector<std::byte> scratch(b.gathered->size_bytes());
                    read(p,scratch.data(),b.gathered->gpu_data_ptr(),scratch.size());
                    std::vector<std::byte> expected_native(scratch.size(),std::byte{0xa5});
                    if(!bad&&exact>0) {
                        std::size_t offset=0;
                        for(std::size_t root=0;root<reader_devices.size();++root) {
                            if(gather&&int(root)==reader_index)continue;
                            const auto heads=projection.assembledHeadCount(root);
                            const auto first=projection.owner(projection.consumers()[root]).first_head;
                            for(int row=0;row<exact;++row)
                                std::memcpy(expected_native.data()+offset+row*heads*format.stride,
                                    canonical.data()+(row*q::kNgramHeads+first)*format.stride,heads*format.stride);
                            offset+=exact*heads*format.stride;
                        }
                    }
                    EXPECT_EQ(scratch,expected_native);
                }
            });
        }
        previous_output=expected;
        if (gather) {
            // Unequal rooted broadcasts can forward another reader's bytes.
            // The exact native cohort sum is independent of that route choice.
            expected_variable_gather+=(bad?0:exact)*q::kNgramHeads*format.stride*(reader_devices.size()-1);
            EXPECT_EQ(std::accumulate(variable_gather_totals.begin(),variable_gather_totals.end(),std::uint64_t{0}),
                expected_variable_gather);
        }
        for (const auto& endpoint : endpoints) {
            const auto& p=people[endpoint.participant];
            p.worker->submitAndWait([&] {
                observeEndpoint(p,endpoint,pipeline,replay,exact,bad&&endpoint.response&&endpoint.failing_provider?0:exact);
            });
        }
    }
    families.clear();
    drain.completed=true;
}

/** @brief Admit one original shape before creating any native graph or payload bank. */
void replay(DeviceType backend, std::vector<std::size_t> stage_sizes, int ple_stage,
    const Format& format, std::string_view isa = {}) {
    const auto degree=std::accumulate(stage_sizes.begin(),stage_sizes.end(),std::size_t{0});
    originalRankLocalReplay(backend,degree,[&](const auto& selected) {
        const auto discovery=MPIContextFactory::global();
        const auto inventory=discovery->clusterInventory();
        const auto& rank=inventory->ranks.at(discovery->rank());
        const auto groups=ObservedPeerGrouping(rank).select(selected,stage_sizes);
        std::vector<std::vector<PhysicalMemoryAllocatorIdentity>> stages;
        std::vector<DeviceId> devices;
        for(const auto& group:groups) {
            auto& stage=stages.emplace_back();
            for(const auto device:group) {
                requireRankLocalGPUOwnership(rank,device,"Fixed native pipeline replay");
                devices.push_back(device);
                stage.push_back({rank.rank,device});
            }
        }
        const auto placement=ImmutableFixedHeadPipelineProjection::checked(rank.rank,ple_stage,
            q::kNgramHeads,maximum_rows,stages);
        const auto& readers=groups.at(ple_stage);
        const auto reader_count=static_cast<int>(readers.size());
        bool unequal=false;
        for(int reader=0;reader<reader_count;++reader)
            unequal|=placement.assembledHeadCount(reader)!=placement.maximumAssembledHeads();
        // General graph and ProcessRuntime commitments retain their one
        // original authority for the process. The compiled case launcher gives
        // each codec its own original process; this body admits exactly that
        // one source and proves full native retirement before process teardown.
        const auto& envelope = format;
        const auto original=test::PreparedNativeLookupFixture::prepare(devices,rank.rank);
        const auto& host=original->resource(DeviceId::cpu());
        const auto collective=backend==DeviceType::CUDA?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL;
        const auto boundary=CollectiveMemoryEstimator::nativePipelineBoundaryBytes(collective);
        const auto metadata=policy();
        PhysicalMemoryPlanBuilder plan;
        const auto quote=[&](DeviceId device, std::size_t bytes) {
            if(!bytes)return;
            const auto tensor=declaration(bytes);
            plan.add(host,workspace,tensor.physicalBytes());
            plan.add(original->resource(device),workspace,tensor.columns()*sizeof(std::int32_t));
        };
        const auto uneven_topology=unequal?std::optional{DeviceCountedAllGatherTopology::resolve(rank,readers)}:std::nullopt;
        for(const auto& owner:placement.owners()) {
            const auto device=owner.allocator.device;
            const auto& resource=original->resource(device);
            const auto [begin,end]=originalRange(owner,metadata);
            plan.add(resource,PhysicalMemoryOwner::ImmutableTableStorage,(end-begin)*envelope.stride);
            plan.add(resource,PhysicalMemoryOwner::LocalCollective,boundary);
            plan.add(host,controller_declaration.owner(),controller_declaration.physicalBytes());
            plan.add(resource,controller_declaration.owner(),controller_declaration.columns()*sizeof(std::int32_t));
            const auto graphs=RetainedParticipantGraphExecutable::generalBOM(host,resource,retainedWidths(format).size());
            for(const auto& row:graphs.resources())plan.add(row);
            if(owner.stage_index==ple_stage) {
                for(const auto bytes:readerSizes(envelope,placement.assembledHeadCount(owner.consumer_index),reader_count,unequal))
                    quote(device,bytes);
                if(reader_count>1)plan.add(resource,PhysicalMemoryOwner::LocalCollective,boundary);
                if(uneven_topology) {
                    const auto line=uneven_topology->memoryFor(owner.consumer_index,
                        maximum_rows*placement.maximumAssembledHeads()*envelope.stride);
                    require(!line.host_bytes,"Fixed pipeline native gather unexpectedly requires a host payload");
                    plan.add(resource,PhysicalMemoryOwner::ActivationTransportStaging,line.device_bytes);
                }
            } else {
                const auto reader=placement.consumers()[owner.consumer_index].device;
                const auto sizes=edgeSizes(envelope,owner.head_count);
                for(int bank=0;bank<2;++bank) {
                    quote(device,sizes[bank]);
                    quote(reader,sizes[bank]);
                }
                quote(device,sizes[2]);
                quote(reader,sizes[3]);
                for(const auto endpoint:{device,reader})
                    plan.add(original->resource(endpoint),PhysicalMemoryOwner::ActivationTransportStaging,
                        2*PipelineNativeByteChannel::endpointBytes());
            }
        }
        original->admit(plan.build());
        {
            const auto pipeline=localContext(devices,collective);
            const auto reader_context=reader_count>1?localContext(readers,collective):nullptr;
            require(pipeline->reserveGraphCaptureBoundaryResources(original->memory())&&
                (!reader_context||reader_context->reserveGraphCaptureBoundaryResources(original->memory())),
                "Fixed pipeline original capture boundaries did not materialize");
            replayFormat(format,placement,rank,devices,*pipeline,reader_context.get(),unequal,*original);
        }
        original->requirePayloadRetired();
        original->retire();
        original->requireExactRetired();
    },isa);
}
}
}

namespace {
/** @brief One native source codec owns one fresh original process authority. */
class NativeFixedHeadPipelineFormats : public ::testing::TestWithParam<fixed_replay::Format> {};
#if defined(HAVE_CUDA) && defined(HAVE_NCCL)
/** @test CUDA original singleton PLE plus provider for one canonical native codec. */
TEST_P(NativeFixedHeadPipelineFormats, CUDA2) {
    fixed_replay::replay(DeviceType::CUDA,{1,1},0,GetParam());
}
#endif
#if defined(HAVE_ROCM) && defined(HAVE_RCCL)
/** @test ROCm original singleton PLE plus provider for one canonical native codec. */
TEST_P(NativeFixedHeadPipelineFormats, ROCm2) {
    fixed_replay::replay(DeviceType::ROCm,{1,1},0,GetParam());
}
#endif
#if (defined(HAVE_CUDA) && defined(HAVE_NCCL)) || (defined(HAVE_ROCM) && defined(HAVE_RCCL))
INSTANTIATE_TEST_SUITE_P(AllNativeCodecs, NativeFixedHeadPipelineFormats,
    ::testing::ValuesIn(fixed_replay::formats(true)), [](const auto& info) { return info.param.fixture->label; });
#endif
}
#if defined(HAVE_ROCM) && defined(HAVE_RCCL)
/** @test Original unequal PLE gather receives the actual addressed ranges. */
TEST(NativeFixedHeadPipeline, ROCm4Unequal) {
    fixed_replay::replay(DeviceType::ROCm,{1,3},1,fixed_replay::formats(false).front(),"gfx906");
}
/** @test Original four MI50 readers and four MI50 storage providers. */
TEST(NativeFixedHeadPipeline, ROCm8MI50Equal) {
    fixed_replay::replay(DeviceType::ROCm,{4,4},0,fixed_replay::formats(false).front(),"gfx906");
}
/** @test Original uneven middle PLE stage talks to both surrounding physical stages. */
TEST(NativeFixedHeadPipeline, ROCm8MI50Middle) {
    fixed_replay::replay(DeviceType::ROCm,{1,3,4},1,fixed_replay::formats(false).front(),"gfx906");
}
#endif
