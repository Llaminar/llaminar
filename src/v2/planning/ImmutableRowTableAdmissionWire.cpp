/**
 * @file ImmutableRowTableAdmissionWire.cpp
 * @brief Authenticate bounded setup-only table candidate metadata with no payload reads.
 *
 * Version8 retains the resolved arithmetic and original source-authored attention schedule
 * alongside common activation geometry. Unknown values reject before narrowing; reloadable debug
 * state is never consulted when restoring an admitted remote provision.
 */
#include "ImmutableRowTableAdmissionWire.h"
#include "MemoryPlanner.h"
#include <limits>
#include <array>
#include <stdexcept>
#include <type_traits>
namespace llaminar2
{
namespace
{
    constexpr std::uint32_t Magic = 0x4e475231u, Version = 8;
    /** @brief Exact typed QFN layout words carried by this schema, without native padding. */
    constexpr std::array QfnLayoutWords{
        &Qwen4ExpMemoryGeometry::residual_width,
        &Qwen4ExpMemoryGeometry::terminal_head_width,
        &Qwen4ExpMemoryGeometry::residual_branches,
        &Qwen4ExpMemoryGeometry::residual_gate_rank,
        &Qwen4ExpMemoryGeometry::indexer_query_heads,
        &Qwen4ExpMemoryGeometry::indexer_key_heads,
        &Qwen4ExpMemoryGeometry::indexer_head_dimension,
        &Qwen4ExpMemoryGeometry::compression_ratio,
        &Qwen4ExpMemoryGeometry::selected_token_budget,
        &Qwen4ExpMemoryGeometry::maximum_selected_tokens,
        &Qwen4ExpMemoryGeometry::ple_decoder_layer,
        &Qwen4ExpMemoryGeometry::ple_heads,
        &Qwen4ExpMemoryGeometry::ple_row_elements,
        &Qwen4ExpMemoryGeometry::ple_convolution_kernel,
        &Qwen4ExpMemoryGeometry::ple_convolution_dilation,
        &Qwen4ExpMemoryGeometry::ple_convolution_history,
        &Qwen4ExpMemoryGeometry::ple_hash_history_tokens,
        &Qwen4ExpMemoryGeometry::ple_table_gguf_type,
    };

/** @brief Encode one unsigned scalar in explicit little-endian order. */
template<class T> void append(std::vector<std::uint8_t> &out,T value)
{
    static_assert(std::is_unsigned_v<T>);
    for(std::size_t i=0;i<sizeof(T);++i) out.push_back(static_cast<std::uint8_t>(value>>(i*8u)));
}
/** @brief One framed immutable sender segment with checked scalar extraction. */
class Reader
{
public:
    /** @brief Retain only the caller-owned exact wire bytes for this synchronous parse. */
    explicit Reader(std::span<const std::uint8_t> bytes):bytes_(bytes){}
    /** @return One authenticated scalar, failing before any out-of-bounds access. */
    template<class T> T read()
    {
        static_assert(std::is_unsigned_v<T>);
        if(sizeof(T)>bytes_.size()-position_) throw std::invalid_argument("Ngram admission wire is truncated");
        T value=0;
        for(std::size_t i=0;i<sizeof(T);++i) value|=static_cast<T>(bytes_[position_++])<<(i*8u);
        return value;
    }
    /** @return Bytes left in this sender's exact frame. */
    std::size_t remaining() const noexcept{return bytes_.size()-position_;}
private:
    std::span<const std::uint8_t> bytes_;std::size_t position_=0;
};
/** @return Checked fixed wire count, independent of native size_t width. */
std::uint32_t count(std::size_t n)
{
    if(n>std::numeric_limits<std::uint32_t>::max()) throw std::overflow_error("Ngram admission wire count overflows");
    return static_cast<std::uint32_t>(n);
}
/** @return A representable physical capacity, never a truncating conversion. */
std::size_t size(Reader &reader)
{
    const auto value=reader.read<std::uint64_t>();
    if(value>std::numeric_limits<std::size_t>::max()) throw std::overflow_error("Ngram admission wire capacity exceeds size_t");
    return static_cast<std::size_t>(value);
}
/** @brief Encode an exact observed physical allocator, including its sender rank. */
void resource(std::vector<std::uint8_t> &out,const PhysicalMemoryResource &r)
{
    if(!r.valid()||r.world_rank<0) throw std::invalid_argument("Ngram wire requires an exact observed resource");
    append(out,static_cast<std::uint32_t>(r.world_rank));append(out,static_cast<std::uint32_t>(r.device.type));
    append(out,static_cast<std::uint32_t>(r.device.ordinal));append(out,static_cast<std::uint64_t>(r.total_bytes));
    append(out,static_cast<std::uint64_t>(r.admission_available_bytes));
}
/** @return Exact resource authenticated to the actual communicator sender. */
PhysicalMemoryResource resource(Reader &in,int rank)
{
    const auto wire_rank=in.read<std::uint32_t>();const auto type=in.read<std::uint32_t>();const auto ordinal=in.read<std::uint32_t>();
    if(rank<0||wire_rank!=static_cast<std::uint32_t>(rank)||ordinal>static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Ngram admission wire contains a foreign or invalid allocator identity");
    if(type!=static_cast<std::uint32_t>(DeviceType::CPU) && type!=static_cast<std::uint32_t>(DeviceType::CUDA) &&
        type!=static_cast<std::uint32_t>(DeviceType::ROCm))
        throw std::invalid_argument("Ngram admission wire device type is unknown");
    PhysicalMemoryResource r{rank,DeviceId{static_cast<DeviceType>(type),static_cast<int>(ordinal)},size(in),size(in)};
    if(!r.valid()) throw std::invalid_argument("Ngram admission wire contains an invalid physical resource");
    return r;
}
/** @brief Encode exact typed owner lines; the canonical builder owns their interpretation. */
void plan(std::vector<std::uint8_t> &out,const PhysicalMemoryPlan &p)
{
    append(out,count(p.resources().size()));
    for(const auto &bom:p.resources())
    {
        resource(out,bom.resource());
        for(std::size_t i=0;i<PhysicalMemoryBOM::ownerCount();++i)
        {
            const auto owner=PhysicalMemoryBOM::ownerAt(i);
            append(out,static_cast<std::uint64_t>(bom.bytes(owner)));
            append(out,static_cast<std::uint64_t>(bom.alreadyResidentBytes(owner)));
        }
    }
}
/** @return Canonical immutable plan rebuilt from exact owner lines, with overflow checks. */
PhysicalMemoryPlan plan(Reader &in,int rank)
{
    const auto n=in.read<std::uint32_t>();
    constexpr auto record=28u+16u*PhysicalMemoryBOM::ownerCount();
    if(n>in.remaining()/record) throw std::invalid_argument("Ngram admission wire plan count exceeds its frame");
    PhysicalMemoryPlanBuilder builder;
    for(std::uint32_t j=0;j<n;++j)
    {
        const auto r=resource(in,rank);PhysicalMemoryBOMBuilder bom(r);
        for(std::size_t i=0;i<PhysicalMemoryBOM::ownerCount();++i)
        {
            const auto planned=size(in),resident=size(in);
            bom.add(PhysicalMemoryBOM::ownerAt(i),planned,resident);
        }
        builder.add(bom.build());
    }
    return builder.build();
}
}
std::vector<std::uint8_t> serializeImmutableRowTableAdmissionRequests(
    std::span<const ImmutableRowTableAdmissionRequest> requests)
{
    if(requests.size()>1u) throw std::invalid_argument("One rank may author only one PLE storage request");
    std::vector<std::uint8_t> out;
    append(out,Magic);append(out,Version);append(out,count(PhysicalMemoryBOM::ownerCount()));append(out,count(requests.size()));
    for(const auto &request:requests)
    {
        request.requested.validate();
        append(out,static_cast<std::uint32_t>(request.requested.storage));
        append(out,static_cast<std::uint32_t>(request.requested.distribution));
        append(out,static_cast<std::uint64_t>(request.requested.window_bytes));
        append(out,static_cast<std::uint64_t>(request.requested.window_count));
        const auto &a=request.requested.host_adaptive;
        append(out,a.maximum_cache_rows);append(out,static_cast<std::uint64_t>(a.cache_physical_byte_limit));
        append(out,a.policy.aging_accesses);append(out,a.policy.protected_numerator);append(out,a.policy.protected_denominator);
        const auto &g=request.requested.gpu_adaptive;
        append(out,g.maximum_gpu_cache_rows);append(out,static_cast<std::uint64_t>(g.gpu_cache_physical_byte_limit));
        append(out,g.gpu_policy.aging_accesses);append(out,g.gpu_policy.protected_numerator);append(out,g.gpu_policy.protected_denominator);
        append(out,g.host.maximum_cache_rows);append(out,static_cast<std::uint64_t>(g.host.cache_physical_byte_limit));
        append(out,g.host.policy.aging_accesses);append(out,g.host.policy.protected_numerator);append(out,g.host.policy.protected_denominator);
        const auto &e=request.input.extent();append(out,static_cast<std::uint32_t>(e.format()));
        append(out,e.rows());append(out,static_cast<std::uint64_t>(e.rowBytes()));
        resource(out,request.input.host());
        const auto node=request.input.hostPlacement().numaNode();
        append(out,static_cast<std::uint32_t>(node.has_value()));
        append(out,static_cast<std::uint32_t>(node.value_or(0))); // Reserved zero in task-policy wire arm.
        append(out,count(request.input.consumers().size()));
        for(const auto &consumer:request.input.consumers()) resource(out,consumer);
        append(out,static_cast<std::uint32_t>(request.contributions.vram.has_value()));
        if(request.contributions.vram) plan(out,*request.contributions.vram);
        plan(out,request.contributions.ram);plan(out,request.contributions.mmap);
        append(out,static_cast<std::uint32_t>(bool(request.execution_provision)));
        if(request.execution_provision)
        {
            const auto &p=*request.execution_provision;p.require(request.requested,request.input,p.requiredBase());
            // Only explicit immutable layout words are sent. The receiver
            // compares all of them with its existing authenticated source.
            const auto &source=p.source();
            for(const auto field:QfnLayoutWords)append(out,source.*field);
            append(out,source.ple_table_rows);append(out,source.ple_table_row_bytes);
            append(out,source.ple_table_native_bytes);
            for(const auto *values:{&source.ple_head_offsets,&source.ple_head_vocab_sizes})
            {
                append(out,count(values->size()));
                for(const auto value:*values)append(out,value);
            }
            append(out,count(source.sparse_decoder_layers.size()));
            for(const auto layer:source.sparse_decoder_layers)append(out,layer);
            append(out,count(p.consumers().size()));
            for(const auto &c:p.consumers())
            {
                resource(out,c.resource);
                for(const int value:{c.rows,c.batch,c.context,c.local_heads,c.first_layer,c.last_layer,c.shards,
                    c.mtp_rows,c.retained_mtp_rows,c.request_capacity})
                {
                    if(value<0)throw std::invalid_argument("Ngram provision wire requires normalized nonnegative graph geometry");
                    append(out,static_cast<std::uint32_t>(value));
                }
                if(c.tp_coordinate< -1 || c.tp_coordinate>=8)
                    throw std::invalid_argument("Ngram wire TP coordinate is out of range");
                append(out,static_cast<std::uint32_t>(c.tp_coordinate+1));
                append(out,static_cast<std::uint32_t>(c.tp_topology));
                const auto &a=c.activation;
                qwen4exp::validateQwenSparseAttentionArithmetic(a.attention_arithmetic,
                                                                c.resource.device.type);
                if (a.attention_arithmetic ==
                        qwen4exp::QwenSparseAttentionArithmetic::TileFP16WithFP32Totals33 ||
                    (a.attention_phase_schedule ==
                         qwen4exp::QwenSparseAttentionPhaseSchedule::Prefill8Decode33 &&
                     a.attention_arithmetic !=
                         qwen4exp::QwenSparseAttentionArithmetic::TileFP16WithFP32Totals))
                    throw std::invalid_argument(
                        "Ngram wire QSA policy is unresolved or conflicting");
                if(a.attention_phase_schedule!=qwen4exp::QwenSparseAttentionPhaseSchedule::Uniform &&
                   a.attention_phase_schedule!=qwen4exp::QwenSparseAttentionPhaseSchedule::Prefill8Decode33)
                    throw std::invalid_argument("Ngram wire attention phase schedule is invalid");
                for (const int value : {a.local_d_ff,
                                        a.local_n_heads,
                                        a.local_n_kv_heads,
                                        a.local_vocab,
                                        static_cast<int>(a.replicated_dense_decode),
                                        static_cast<int>(a.routed_compute_policy),
                                        static_cast<int>(a.mtp_terminal_logits_layout),
                                        static_cast<int>(a.attention_phase_schedule),
                                        static_cast<int>(a.attention_arithmetic)})
                {
                    if(value<0)throw std::invalid_argument("Ngram common activation geometry must be nonnegative");
                    append(out,static_cast<std::uint32_t>(value));
                }
            }
            plan(out,p.executionPlan());plan(out,p.requiredBase());
        }
    }
    return out;
}
std::vector<ImmutableRowTableAdmissionRequest> parseImmutableRowTableAdmissionRequests(
    std::span<const std::uint8_t> bytes,int expected_rank,const ModelMemoryProfile *canonical_source)
{
    Reader in(bytes);
    if(expected_rank<0||in.read<std::uint32_t>()!=Magic||in.read<std::uint32_t>()!=Version||
        in.read<std::uint32_t>()!=PhysicalMemoryBOM::ownerCount())
        throw std::invalid_argument("Ngram admission wire source/schema/version is invalid");
    const auto n=in.read<std::uint32_t>();
    if(n>1u) throw std::invalid_argument("Ngram admission wire duplicates rank authority");
    std::vector<ImmutableRowTableAdmissionRequest> result;
    if(n)
    {
        ImmutableRowTablePolicy policy;
        const auto preference=in.read<std::uint32_t>();
        if(preference>static_cast<std::uint32_t>(ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD))
            throw std::invalid_argument("Ngram admission wire preference is unknown");
        policy.storage=static_cast<ImmutableRowStoragePreference>(preference);
        const auto distribution=in.read<std::uint32_t>();
        if(distribution>static_cast<std::uint32_t>(ImmutableRowTableDistribution::FixedHeadSharded))
            throw std::invalid_argument("Ngram wire distribution is unknown");
        policy.distribution=static_cast<ImmutableRowTableDistribution>(distribution);
        policy.window_bytes=size(in);policy.window_count=size(in);
        policy.host_adaptive.maximum_cache_rows=in.read<std::uint64_t>();policy.host_adaptive.cache_physical_byte_limit=size(in);
        policy.host_adaptive.policy.aging_accesses=in.read<std::uint64_t>();
        policy.host_adaptive.policy.protected_numerator=in.read<std::uint32_t>();
        policy.host_adaptive.policy.protected_denominator=in.read<std::uint32_t>();
        auto &g=policy.gpu_adaptive;
        g.maximum_gpu_cache_rows=in.read<std::uint64_t>();g.gpu_cache_physical_byte_limit=size(in);
        g.gpu_policy.aging_accesses=in.read<std::uint64_t>();g.gpu_policy.protected_numerator=in.read<std::uint32_t>();
        g.gpu_policy.protected_denominator=in.read<std::uint32_t>();
        g.host.maximum_cache_rows=in.read<std::uint64_t>();g.host.cache_physical_byte_limit=size(in);
        g.host.policy.aging_accesses=in.read<std::uint64_t>();g.host.policy.protected_numerator=in.read<std::uint32_t>();
        g.host.policy.protected_denominator=in.read<std::uint32_t>();policy.validate();
        const auto format=static_cast<GGUFTensorType>(in.read<std::uint32_t>());
        const auto rows=in.read<std::uint64_t>();const auto row_bytes=size(in);
        const auto extent=ImmutableRowTableExtent::fromNative(format,rows,row_bytes);
        const auto host=resource(in,expected_rank);
        const auto bound=in.read<std::uint32_t>(),node=in.read<std::uint32_t>();
        if(bound>1u || (!bound && node) || node>static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Ngram admission wire RAM locality is invalid");
        const auto locality=bound ? ImmutableRowTableHostPlacement::onNode(static_cast<int>(node)) :
            ImmutableRowTableHostPlacement::taskPolicy();
        const auto consumers=in.read<std::uint32_t>();
        if(consumers>in.remaining()/28u) throw std::invalid_argument("Ngram admission consumer count exceeds its frame");
        std::vector<PhysicalMemoryResource> observed;
        for(std::uint32_t i=0;i<consumers;++i) observed.push_back(resource(in,expected_rank));
        auto input=ImmutableRowTablePlacementInput::forConsumers(extent,host,observed,locality);
        const auto available=in.read<std::uint32_t>();
        if(available>1u) throw std::invalid_argument("Ngram admission wire eligibility is invalid");
        std::optional<PhysicalMemoryPlan> vram;
        if(available) vram=plan(in,expected_rank);
        auto ram=plan(in,expected_rank),mmap=plan(in,expected_rank);
        const auto provision_arm=in.read<std::uint32_t>();
        if(provision_arm>1u || (canonical_source && !provision_arm))
            throw std::invalid_argument("Ngram admission wire lacks its production execution provision");
        std::shared_ptr<const ImmutableRowTableExecutionProvision> provision;
        if(provision_arm)
        {
            if(!canonical_source)throw std::invalid_argument("Ngram wire provision requires independent receiver-owned source metadata");
            Qwen4ExpMemoryGeometry layout;
            for(const auto field:QfnLayoutWords)layout.*field=in.read<std::uint32_t>();
            layout.ple_table_rows=in.read<std::uint64_t>();layout.ple_table_row_bytes=in.read<std::uint64_t>();
            layout.ple_table_native_bytes=in.read<std::uint64_t>();
            for(auto *values:{&layout.ple_head_offsets,&layout.ple_head_vocab_sizes})
            {
                const auto n=in.read<std::uint32_t>();
                if(n>16 || n>in.remaining()/8u)throw std::invalid_argument("Ngram original head metadata exceeds its source/frame");
                for(std::uint32_t h=0;h<n;++h)values->push_back(in.read<std::uint64_t>());
            }
            const auto sparse=in.read<std::uint32_t>();
            if(sparse>in.remaining()/4u)throw std::invalid_argument("Ngram wire layout layer count exceeds its frame");
            for(std::uint32_t i=0;i<sparse;++i)layout.sparse_decoder_layers.push_back(in.read<std::uint32_t>());
            if(!canonical_source->flash_next || layout!=*canonical_source->flash_next)
                throw std::invalid_argument("Ngram wire provision layout disagrees with independent receiver source metadata");
            const auto instances=in.read<std::uint32_t>();
            if (instances == 0 || instances > in.remaining() / 112u)
                throw std::invalid_argument("Ngram wire provision geometry count exceeds its exact frame");
            std::vector<ImmutableRowTableExecutionConsumer> geometry;
            for(std::uint32_t i=0;i<instances;++i)
            {
                const auto r=resource(in,expected_rank);int fields[10];
                for(int &field:fields)
                {
                    const auto value=in.read<std::uint32_t>();
                    if(value>static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
                        throw std::invalid_argument("Ngram wire provision geometry exceeds native range");
                    field=static_cast<int>(value);
                }
                const auto coordinate=in.read<std::uint32_t>(),topology=in.read<std::uint32_t>();
                if(coordinate>8 || topology>static_cast<std::uint32_t>(ImmutableRowTableTPTopology::HomogeneousFullMainRankLocal))
                    throw std::invalid_argument("Ngram wire TP provenance is invalid");
                int common[9];
                for(int &field:common)
                {
                    const auto value=in.read<std::uint32_t>();
                    if(value>static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
                        throw std::invalid_argument("Ngram common activation geometry exceeds native range");
                    field=static_cast<int>(value);
                }
                if(common[4]>1 || common[5]>static_cast<int>(RoutedExpertComputePolicy::GateUpOwnedDownColumns) ||
                    common[6]>static_cast<int>(MTPTerminalLogitsLayout::VocabularyShardPerParticipant))
                    throw std::invalid_argument("Ngram wire common activation policy is invalid");
                if(common[7]!=static_cast<int>(qwen4exp::QwenSparseAttentionPhaseSchedule::Uniform) &&
                   common[7]!=static_cast<int>(qwen4exp::QwenSparseAttentionPhaseSchedule::Prefill8Decode33))
                    throw std::invalid_argument("Ngram wire attention phase schedule is invalid");
                if (common[8] !=
                        static_cast<int>(qwen4exp::QwenSparseAttentionArithmetic::FixedFP32) &&
                    common[8] !=
                        static_cast<int>(
                            qwen4exp::QwenSparseAttentionArithmetic::TileFP16WithFP32Totals))
                    throw std::invalid_argument("Ngram wire base arithmetic is invalid");
                const auto arithmetic =
                    static_cast<qwen4exp::QwenSparseAttentionArithmetic>(common[8]);
                qwen4exp::validateQwenSparseAttentionArithmetic(arithmetic, r.device.type);
                if (arithmetic ==
                        qwen4exp::QwenSparseAttentionArithmetic::TileFP16WithFP32Totals33 ||
                    (common[7] ==
                         static_cast<int>(
                             qwen4exp::QwenSparseAttentionPhaseSchedule::Prefill8Decode33) &&
                     arithmetic != qwen4exp::QwenSparseAttentionArithmetic::TileFP16WithFP32Totals))
                    throw std::invalid_argument(
                        "Ngram wire QSA policy is unresolved or conflicting");
                ActivationGraphMemoryGeometry activation{
                    .batch_size = fields[1],
                    .resident_graph_rows = fields[0],
                    .context_tokens = fields[2],
                    .local_d_ff = common[0],
                    .local_n_heads = common[1],
                    .local_n_kv_heads = common[2],
                    .local_vocab = common[3],
                    .replicated_dense_decode = bool(common[4]),
                    .first_layer = fields[4],
                    .last_layer = fields[5],
                    .total_shards = fields[6],
                    .routed_compute_policy = static_cast<RoutedExpertComputePolicy>(common[5]),
                    .mtp_target_query_rows = fields[7],
                    .retained_mtp_candidate_rows = fields[8],
                    .mtp_terminal_logits_layout = static_cast<MTPTerminalLogitsLayout>(common[6]),
                    .generation_request_capacity = fields[9],
                    .attention_arithmetic = arithmetic,
                    .attention_phase_schedule =
                        static_cast<qwen4exp::QwenSparseAttentionPhaseSchedule>(common[7])};
                geometry.push_back({r,fields[0],fields[1],fields[2],fields[3],fields[4],fields[5],fields[6],fields[7],fields[8],fields[9],activation,
                    static_cast<int>(coordinate)-1,static_cast<ImmutableRowTableTPTopology>(topology)});
            }
            auto execution=plan(in,expected_rank),required=plan(in,expected_rank);
            provision=MemoryPlanner::restoreImmutableRowTableExecutionProvision(*canonical_source,policy,input,
                std::move(geometry),std::move(execution),std::move(required));
        }
        result.push_back({policy,std::move(input),{std::move(vram),std::move(ram),std::move(mmap)},std::move(provision)});
    }
    if(in.remaining()) throw std::invalid_argument("Ngram admission wire has trailing bytes");
    return result;
}
}
