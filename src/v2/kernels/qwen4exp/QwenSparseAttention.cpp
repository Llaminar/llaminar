/**
 * @file QwenSparseAttention.cpp
 * @brief Static, device-independent validation of sparse execution bindings.
 *
 * Device replay counts are deliberately not inspected here. Native kernels own
 * dynamic validation, making retained graph replay independent of a host shadow.
 */
#include "QwenSparseAttention.h"
#include <limits>
#include <cmath>
#include <stdexcept>

namespace llaminar2::qwen4exp
{
QwenStateTransfer qwenSparseCandidateTransfer(const QwenSparseCandidateRetention &p)
{
    if((p.work_counters.bound() && !p.work_counters.valid()) || !p.liveRows || !p.copiedBytes || p.source==p.destination)
        throw std::invalid_argument("QSA verifier retention requires a live authority, byte observation and distinct candidate bank");
    QwenStateTransfer transfer{p.source,p.destination,128*sizeof(float),p.rowCapacity,
        QwenStateRowSource::ExecutionCount,p.liveRows,0,nullptr,p.copiedBytes,
        p.work_counters.metric(QwenSparseWorkCounter::CandidatePayload)};
    validateQwenStateTransfer(transfer);
    return transfer;
}

namespace
{
/** @brief Reject enum corruption instead of interpreting it as FP32. */
void validateRepresentation(QwenSparseScalar representation)
{
    switch (representation)
    {
    case QwenSparseScalar::FP32:
    case QwenSparseScalar::FP16:
    case QwenSparseScalar::BF16: return;
    }
    throw std::invalid_argument("Qwen sparse attention: unsupported native representation");
}
/** @brief Validate the admitted complete native head without reading its payload. */
void validateBank(const QwenSparseNativeBank &bank, bool key, int heads, int tokens)
{
    std::size_t bytes = 0, alignment = 4;
    switch (bank.codec)
    {
    case TensorType::FP32: bytes = 1024; break;
    case TensorType::FP16: case TensorType::BF16: bytes = 512; alignment = 2; break;
    case TensorType::Q8_1: bytes = 8 * 36; break;
    case TensorType::Q16_1:
        if (bank.q16_block_size != 32 && bank.q16_block_size != 64 && bank.q16_block_size != 128)
            throw std::invalid_argument("Qwen sparse attention: unsupported native Q16 block size");
        bytes = std::size_t(256 / bank.q16_block_size) * (8 + 2 * bank.q16_block_size); break;
    case TensorType::AQ8:
        if (!key || !bank.anchor) throw std::invalid_argument("Qwen sparse attention: AQ8 requires an immutable key anchor");
        bytes = 260; break;
    case TensorType::TQ4: case TensorType::TQ8:
        if (!bank.centroids) throw std::invalid_argument("Qwen sparse attention: TurboQuant requires its canonical codebook");
        for (int head = 0; head < heads; ++head)
            if (!bank.rotations[head]) throw std::invalid_argument("Qwen sparse attention: TurboQuant requires each canonical KV-head rotation");
        bytes = bank.codec == TensorType::TQ4 ? 136 : 264; break;
    default: throw std::invalid_argument("Qwen sparse attention: unsupported native cache codec");
    }
    const bool token_major = bank.layout == QwenSparseNativeLayout::TokenMajor &&
        bank.head_stride_bytes >= bytes && bank.token_stride_bytes >= std::size_t(heads - 1) * bank.head_stride_bytes + bytes;
    const bool head_major = bank.layout == QwenSparseNativeLayout::HeadMajor &&
        bank.token_stride_bytes >= bytes && bank.head_stride_bytes >= std::size_t(tokens - 1) * bank.token_stride_bytes + bytes;
    if (!bank.data || (!token_major && !head_major) || bank.head_stride_bytes % alignment || bank.token_stride_bytes % alignment)
        throw std::invalid_argument("Qwen sparse attention: invalid native bank geometry");
}
}
QwenSparsePowerFrequencies makeQwenSparsePowerFrequencies(float theta,int dimension)
{
    if(!std::isfinite(theta) || theta<=0 || dimension!=64)
        throw std::invalid_argument("QFN power rotary invariants require finite positive theta and exact rotary width64");
    QwenSparsePowerFrequencies result{};
    for(int pair=0;pair<32;++pair)result[pair]=1.0f/std::pow(theta,float(2*pair)/float(dimension));
    return result;
}

void validateQwenSparsePriorSnapshot(const QwenSparsePriorSnapshot& p)
{
    if(p.work_counters.bound() && !p.work_counters.valid())
        throw std::invalid_argument("QSA terminal counters require the original aligned canonical control row");
    const auto address=[](const void* value){return reinterpret_cast<std::uintptr_t>(value);};
    const auto overlaps=[&](const void* a,std::size_t an,const void* b,std::size_t bn){
        const auto aa=address(a),bb=address(b);return aa<=bb ? bb-aa<an : aa-bb<bn;
    };
    if(!p.replay || !p.accepted_partial || !p.output || !p.traffic || p.row_capacity<=0 || p.token_capacity<=0 ||
       address(p.output)%alignof(QwenSparseTransactionPrior)!=0 || address(p.traffic)%alignof(QwenSparsePriorTraffic)!=0 ||
       overlaps(p.output,sizeof(*p.output),p.replay,sizeof(*p.replay)) ||
       overlaps(p.output,sizeof(*p.output),p.accepted_partial,4*128*sizeof(float)) ||
       overlaps(p.traffic,sizeof(*p.traffic),p.output,sizeof(*p.output)) ||
       overlaps(p.traffic,sizeof(*p.traffic),p.replay,sizeof(*p.replay)) ||
       overlaps(p.traffic,sizeof(*p.traffic),p.accepted_partial,4*128*sizeof(float)))
        throw std::invalid_argument("QSA transaction prior requires exact distinct typed arena operands and geometry");
    if(p.work_counters.bound()) {
        const auto* counter=p.work_counters.control+sampling_math::kDeviceGenerationControlQsaPriorHeaderBytesLow;
        constexpr std::size_t extent=4*sizeof(std::uint64_t);
        if(overlaps(counter,extent,p.output,sizeof(*p.output)) || overlaps(counter,extent,p.traffic,sizeof(*p.traffic)) ||
           overlaps(counter,extent,p.replay,sizeof(*p.replay)) || overlaps(counter,extent,p.accepted_partial,4*128*sizeof(float)))
            throw std::invalid_argument("QSA terminal counter words overlap immutable snapshot data");
    }
}
void validateQwenSparsePoolSelection(QwenSparsePoolSelection policy)
{
    switch (policy)
    {
    case QwenSparsePoolSelection::Installed:
    case QwenSparsePoolSelection::Bounded256_512:
    case QwenSparsePoolSelection::Bounded512_512:
    case QwenSparsePoolSelection::Bounded1024_1024: return;
    }
    throw std::invalid_argument("QSA selection policy is unknown");
}

void validateQwenSparsePoolSelection(QwenSparsePoolSelection policy, int rows)
{
    validateQwenSparsePoolSelection(policy);
    if (rows <= 0) throw std::invalid_argument("QSA selector requires positive admitted row capacity");
}

void validateQwenSparseSelection(const QwenSparseIndexer &p)
{
    if (!p.replay || !p.scores || !p.selected || !p.selected_count ||
        p.row_capacity <= 0 || p.token_capacity <= 0 ||
        p.block_capacity < (std::int64_t(p.token_capacity) + 3) / QwenSparsePool)
        throw std::invalid_argument("QSA selection requires original replay, distinct banks and admitted geometry");
    validateQwenSparsePoolSelection(p.pool_selection, p.row_capacity);
    if (p.pool_selection == QwenSparsePoolSelection::Installed && p.selection_observations)
        throw std::invalid_argument("QSA Installed selector have no bounded-only observations");
    if (p.transaction_prior && (p.replay != &p.transaction_prior->replay))
        throw std::invalid_argument("QSA selection replay is foreign to immutable transaction prior");
    struct Span { std::uintptr_t begin; std::size_t bytes; };
    const auto span = [](const void *pointer, std::size_t count, std::size_t width, std::size_t alignment)
    {
        const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
        if (count > SIZE_MAX / width || begin % alignment || begin > UINTPTR_MAX - count * width)
            throw std::invalid_argument("QSA selection operand extent or alignment overflows");
        return Span{begin, count * width};
    };
    const auto rows = std::size_t(p.row_capacity);
    if (rows > SIZE_MAX / std::size_t(p.block_capacity) || rows > SIZE_MAX / QwenSparseSelected)
        throw std::invalid_argument("QSA selection admitted row product overflows");
    const Span banks[] = {
        span(p.replay, 1, sizeof(*p.replay), alignof(QwenSparseReplay)),
        span(p.scores, rows * std::size_t(p.block_capacity), sizeof(float), alignof(float)),
        span(p.selected, rows * QwenSparseSelected, sizeof(std::int32_t), alignof(std::int32_t)),
        span(p.selected_count, rows, sizeof(std::int32_t), alignof(std::int32_t)),
        span(p.selection_observations, p.selection_observations ? rows : 0,
             sizeof(QwenSparseSelectionObservation), alignof(QwenSparseSelectionObservation))};
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = i + 1; j < 5; ++j)
            if (banks[i].bytes && banks[j].bytes &&
                banks[i].begin < banks[j].begin + banks[j].bytes &&
                banks[j].begin < banks[i].begin + banks[i].bytes)
                throw std::invalid_argument("QSA selection immutable inputs and exclusive outputs overlap");
}
QwenSparsePreparationGrid qwenSparsePreparationGrid(int rows)
{
    if(rows<=0)throw std::invalid_argument("QSA preparation requires positive row capacity");
    const std::int64_t pools=(std::int64_t(rows)+6)/4,queries=std::int64_t(rows)*4;
    if(pools+queries>std::numeric_limits<int>::max())
        throw std::invalid_argument("QSA preparation grid exceeds native index capacity");
    return {static_cast<int>(pools),static_cast<int>(queries),static_cast<int>(pools+queries),
        rows <= attention::kMaxGroupedVerifierAttentionRows
            ? QwenSparsePreparationProgram::FusedGrouped
            : QwenSparsePreparationProgram::SeparatePrefill};
}
void validateQwenSparseIndexer(const QwenSparseIndexer &p)
{
    switch(p.query_demand){
    case QwenSparseQueryDemand::AllLive:
    case QwenSparseQueryDemand::RankingOnly:break;
    default:throw std::invalid_argument("Qwen sparse indexer: unknown query demand policy");
    }
    validateQwenSparseSelection(p);
    if(p.transaction_prior && (p.replay!=&p.transaction_prior->replay || p.partial_keys!=p.transaction_prior->partial))
        throw std::invalid_argument("QSA indexer immutable prior codec does not own its replay and raw tail");
    if (!p.replay || !p.query || !p.raw_keys || !p.partial_keys || !p.pooled_keys ||
        !p.query_norm || !p.key_norm || !p.prepared_query ||
        !p.scores || !p.selected || !p.selected_count)
        throw std::invalid_argument("Qwen sparse indexer: every persistent binding is required");
    if (p.row_capacity <= 0 || p.token_capacity <= 0 ||
        p.block_capacity < (static_cast<std::int64_t>(p.token_capacity) + 3) / 4 ||
        p.position_capacity <= 0 || p.rotary_dim <= 0 || p.rotary_dim > 128 ||
        p.rotary_dim % 2 != 0 || !(p.norm_epsilon > 0.0f) ||
        !(p.norm_epsilon < std::numeric_limits<float>::infinity()))
        throw std::invalid_argument("Qwen sparse indexer: invalid admitted geometry or normalization policy");
    (void)qwenSparsePreparationGrid(p.row_capacity);
    validateRepresentation(p.representation);
    if(p.rotary_source==QwenSparseRotarySource::TablePlanes) {
        if(!p.cosine || !p.sine)throw std::invalid_argument("QSA table rotary source requires both immutable planes");
    } else if(p.rotary_source==QwenSparseRotarySource::AnalyticTheta) {
        if(p.cosine || p.sine || !std::isfinite(p.rope_theta) || p.rope_theta<=0 || p.rotary_dim!=64 || p.inverse_frequencies[0]!=1.0f)
            throw std::invalid_argument("QSA analytic source requires exact setup-derived64-wide FP32 power invariants and no table aliases");
        for(float frequency:p.inverse_frequencies)if(!std::isfinite(frequency) || frequency<=0)
            throw std::invalid_argument("QSA analytic frequency invariant is invalid");
    } else throw std::invalid_argument("QSA indexer has invalid rotary source");
    // Only newly concurrent cross-role hazards are rejected. Shared immutable
    // weights/tables and the original prior-tail codec remain legal aliases.
    struct Span { std::uintptr_t begin,end; };
    const auto span=[](const void* pointer,std::size_t count,std::size_t width)
    {
        const auto begin=reinterpret_cast<std::uintptr_t>(pointer);
        if(!pointer||count>SIZE_MAX/width||begin%alignof(float)||begin>UINTPTR_MAX-count*width)
            throw std::invalid_argument("QSA preparation span overflows or is unaligned");
        return Span{begin,begin+count*width};
    };
    const auto overlaps=[](Span a,Span b){return a.begin<b.end&&b.begin<a.end;};
    const auto rows=std::size_t(p.row_capacity);
    if(rows>SIZE_MAX/512||std::size_t(p.block_capacity)>SIZE_MAX/128||
        std::size_t(p.position_capacity)>SIZE_MAX/std::size_t(p.rotary_dim/2))
        throw std::invalid_argument("QSA preparation element-count product overflows");
    const auto queries=span(p.prepared_query,rows*512,sizeof(float));
    const auto pools=span(p.pooled_keys,std::size_t(p.block_capacity)*128,sizeof(float));
    if(overlaps(queries,pools))throw std::invalid_argument("QSA concurrent preparation outputs overlap");
    const Span query_reads[]{span(p.query,rows*512,sizeof(float)),span(p.query_norm,128,sizeof(float))};
    const Span pool_reads[]{span(p.raw_keys,rows*128,sizeof(float)),
        span(p.partial_keys,(p.transaction_prior?3u:4u)*128,sizeof(float)),span(p.key_norm,128,sizeof(float))};
    for(const auto& input:query_reads)if(overlaps(pools,input))
        throw std::invalid_argument("QSA pooled-key output overlaps concurrent query preparation input");
    for(const auto& input:pool_reads)if(overlaps(queries,input))
        throw std::invalid_argument("QSA prepared-query output overlaps concurrent pool preparation input");
    const auto protect_shared=[&](Span input)
    {
        if(overlaps(pools,input)||overlaps(queries,input))
            throw std::invalid_argument("QSA preparation output overlaps shared immutable input");
    };
    protect_shared(span(p.replay,1,sizeof(*p.replay)));
    if(p.transaction_prior)protect_shared(span(p.transaction_prior,1,sizeof(*p.transaction_prior)));
    if(p.rotary_source==QwenSparseRotarySource::TablePlanes)
    {
        const auto coordinates=std::size_t(p.position_capacity)*std::size_t(p.rotary_dim/2);
        protect_shared(span(p.cosine,coordinates,sizeof(float)));
        protect_shared(span(p.sine,coordinates,sizeof(float)));
    }

}
void validateQwenSparseAttention(const QwenSparseAttention &p)
{
    (void)qwenSparseAttentionArithmeticName(p.arithmetic);
    if (!p.replay || !p.query || !p.selected || !p.selected_count || !p.output || p.row_capacity <= 0 || p.token_capacity <= 0)
        throw std::invalid_argument("Qwen sparse attention: invalid native cache or selected-index binding");
    if (p.query_heads <= 0 || p.kv_heads <= 0 || p.query_head_start < 0 || p.kv_head_start < 0 ||
        p.query_head_start + p.query_heads > 24 || p.kv_head_start + p.kv_heads > 2 ||
        p.query_head_start / 12 < p.kv_head_start || (p.query_head_start + p.query_heads - 1) / 12 >= p.kv_head_start + p.kv_heads)
        throw std::invalid_argument("Qwen sparse attention: invalid participant-local GQA ownership");
    validateBank(p.kv.keys, true, p.kv_heads, p.token_capacity); validateBank(p.kv.values, false, p.kv_heads, p.token_capacity);
    switch (p.kv.addressing)
    {
    case QwenSparseKVAddressing::Contiguous:
        if (!p.kv.logical_view.isContiguous() || p.kv.logical_view.logical_row_origin != 0 || p.kv.row_parameters)
            throw std::invalid_argument("Qwen sparse attention: ambiguous contiguous mapping authority");
        break;
    case QwenSparseKVAddressing::FixedLogicalView:
        if (!p.kv.logical_view.validFor(p.token_capacity) || p.kv.row_parameters)
            throw std::invalid_argument("Qwen sparse attention: invalid fixed logical mapping authority");
        break;
    case QwenSparseKVAddressing::RowParameters:
        if (!p.kv.row_parameters || !p.kv.logical_view.isContiguous() || p.kv.logical_view.logical_row_origin != 0)
            throw std::invalid_argument("Qwen sparse attention: incomplete execution-owned ring mapping");
        break;
    default: throw std::invalid_argument("Qwen sparse attention: invalid native addressing policy");
    }
    switch (p.kv.key_policy.encoding)
    {
    case attention::AttentionKeyCacheEncoding::PostRotary: break;
    case attention::AttentionKeyCacheEncoding::PreRotaryDeviceTransform:
        if (p.kv.keys.codec == TensorType::TQ4 || p.kv.keys.codec == TensorType::TQ8)
            throw std::invalid_argument("Qwen sparse attention: pre-RoPE rotated TurboQuant keys require a spatial-basis codec");
        if (p.kv.key_rotary_dim <= 0 || p.kv.key_rotary_dim > 256 || p.kv.key_rotary_dim % 2 ||
            int(256 * p.kv.key_policy.partial_rotary_factor) != p.kv.key_rotary_dim)
            throw std::invalid_argument("Qwen sparse attention: incomplete selected-key RoPE bindings");
        if(p.kv.key_rotary_source==QwenSparseRotarySource::TablePlanes) {
            if(!p.kv.key_cosine || !p.kv.key_sine || p.kv.key_position_capacity<=0)
                throw std::invalid_argument("QSA selected-key table source requires exact immutable planes");
        } else if(p.kv.key_rotary_source==QwenSparseRotarySource::CanonicalEngineTheta) {
            if(p.kv.key_cosine || p.kv.key_sine || !std::isfinite(p.kv.key_policy.rope_theta) || p.kv.key_policy.rope_theta<=0 ||
                p.kv.key_negative_log_theta!=-std::log(p.kv.key_policy.rope_theta))
                throw std::invalid_argument("QSA selected-key analytic source requires finite positive theta");
        } else throw std::invalid_argument("QSA selected-key rotary source is invalid");
        break;
    default: throw std::invalid_argument("Qwen sparse attention: invalid key-cache transform policy");
    }
}
void validateQwenSparsePartialCommit(const QwenSparsePartialCommit &p)
{
    if(p.work_counters.bound() && !p.work_counters.valid())
        throw std::invalid_argument("QSA terminal counters require the original aligned canonical control row");
    if(p.transaction_prior && (p.replay!=&p.transaction_prior->replay || p.prior_partial!=p.transaction_prior->partial || p.token_capacity<=0))
        throw std::invalid_argument("QSA commit immutable prior codec does not own its replay, raw tail and cache geometry");
    if (!p.replay || !p.prior_partial || !p.raw_keys || !p.partial_keys ||
        p.acceptance.candidates().requests()!=1 ||
        p.acceptance.candidates().lengthOwner()!=&p.replay->live_rows)
        throw std::invalid_argument("Qwen sparse partial commit: incomplete acceptance binding");
    if(p.work_counters.bound()) {
        const auto c=reinterpret_cast<std::uintptr_t>(p.work_counters.metric(QwenSparseWorkCounter::AcceptedPayload).words);
        const auto overlaps=[&](const void* address,std::size_t extent) {
            const auto begin=reinterpret_cast<std::uintptr_t>(address);
            return begin<=c ? c-begin<extent : begin-c<sizeof(std::uint64_t);
        };
        if(overlaps(p.prior_partial,(p.transaction_prior?3:4)*128*sizeof(float)) || overlaps(p.partial_keys,4*128*sizeof(float)) ||
           overlaps(p.replay,sizeof(*p.replay)) || overlaps(p.raw_keys,std::size_t(p.acceptance.candidates().rowsPerRequest())*128*sizeof(float)))
            throw std::invalid_argument("QSA terminal counter words overlap accepted partial publication data");
    }
}
} // namespace llaminar2::qwen4exp
