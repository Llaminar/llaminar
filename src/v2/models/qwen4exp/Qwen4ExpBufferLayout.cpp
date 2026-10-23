/**
 * @file Qwen4ExpBufferLayout.cpp
 * @brief Shared, overflow-checked allocation declarations for QFN compute and state.
 *
 * A single buffer declaration binds capacity, scalar type, semantic lifetime
 * and BufferId. In particular, selected-attention partials are one reusable
 * scratch bank across layers; only accepted recurrent/indexer state scales
 * with model layers. Native row payloads retain their exact GGUF byte width.
 */
#include "Qwen4ExpBufferLayout.h"
#include "Qwen4ExpConfig.h"
#include "../../loaders/ResolvedImmutableRowTablePlacement.h"
#include "../GraphTypes.h"
#include "../../execution/local_execution/graph/GraphResolver.h"
#include "../../kernels/qwen4exp/QwenSparseAttention.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace llaminar2
{
namespace
{
/** @brief Form a checked nonzero physical extent before admission. */
std::size_t product(std::size_t a, std::size_t b)
{
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("Flash Next arena geometry exceeds addressable capacity");
    return a * b;
}
/** @brief The registry is immutable and created once outside graph capture. */
const std::vector<Qwen4ExpBufferDeclaration> declarations{
    {BufferId::QFN_EMBEDDING_ROWS, {"qfn_embedding_rows", {"seq_len", "d_model"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::QFN_HC_NORMALIZED, {"qfn_hc_normalized", {"seq_len", "qfn_trunk"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::QFN_HC_DOWN, {"qfn_hc_down", {"seq_len", "qfn_hc_rank"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::QFN_HC_READ_LOGITS, {"qfn_hc_read_logits", {"seq_len", "qfn_trunk"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::QFN_HC_WRITE_LOGITS, {"qfn_hc_write_logits", {"seq_len", "4"}, "fp32", BufferSemantic::Scratch}},
    {BufferId::QFN_NGRAM_ROW_IDS, {"qfn_ngram_row_ids", {"seq_len", "32"}, "int32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::PLE},
    {BufferId::QFN_NGRAM_NATIVE_ROWS, {"qfn_ngram_native_rows", {"seq_len", "qfn_native_words"}, "int32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::PLEHostStorage},
    {BufferId::QFN_NGRAM_LOOKUP_STATE, {"qfn_ngram_lookup_state", {"16"}, "int32", BufferSemantic::InOut}, Qwen4ExpBufferPlacement::PLEGPUHostStorage},
    {BufferId::QFN_NGRAM_SHARD_LOCAL_HEADER, {"qfn_ngram_shard_local_header", {"2"}, "int32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::PLEFixedHeadStorage},
    {BufferId::QFN_NGRAM_SHARD_HEADERS, {"qfn_ngram_shard_headers", {"qfn_fixed_head_participants", "2"}, "int32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::PLEFixedHeadStorage},
    {BufferId::QFN_NGRAM_SHARD_TRANSPORT_ROWS, {"qfn_ngram_shard_transport_rows", {"1"}, "int32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::PLEFixedHeadStorage},
    {BufferId::QFN_NGRAM_SHARD_LOCAL_NATIVE, {"qfn_ngram_shard_local_native", {"qfn_fixed_local_words"}, "int32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::PLEFixedHeadStorage},
    {BufferId::QFN_NGRAM_SHARD_GATHERED_NATIVE, {"qfn_ngram_shard_gathered_native", {"qfn_fixed_gathered_words"}, "int32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::PLEFixedHeadStorage},
    {BufferId::QFN_PLE_EMBEDDING, {"qfn_ple_embedding", {"seq_len", "d_model"}, "fp32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::PLE},
    {BufferId::QFN_PLE_KEY, {"qfn_ple_key", {"seq_len", "qfn_trunk"}, "fp32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::PLE},
    {BufferId::QFN_PLE_VALUE, {"qfn_ple_value", {"seq_len", "d_model"}, "fp32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::PLE},
    {BufferId::QFN_PLE_QUERY, {"qfn_ple_query", {"seq_len", "qfn_trunk"}, "fp32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::PLE},
    {BufferId::QFN_PLE_GATED, {"qfn_ple_gated", {"seq_len", "qfn_trunk"}, "fp32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::PLE},
    {BufferId::QFN_PLE_NORMALIZED, {"qfn_ple_normalized", {"seq_len", "qfn_trunk"}, "fp32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::PLE},
    {BufferId::QFN_PLE_TOKEN_HISTORY, {"qfn_ple_token_history", {"qfn_requests", "4"}, "int32", BufferSemantic::InOut}, Qwen4ExpBufferPlacement::PLE},
    {BufferId::QFN_PLE_CONV_HISTORY, {"qfn_ple_conv_history", {"qfn_requests", "18", "qfn_trunk"}, "fp32", BufferSemantic::InOut}, Qwen4ExpBufferPlacement::PLE},
    {BufferId::QFN_PLE_PRIOR_TOKEN_HISTORY, {"qfn_ple_prior_token_history", {"qfn_requests", "2"}, "int32", BufferSemantic::InOut}, Qwen4ExpBufferPlacement::PLEMTP},
    {BufferId::QFN_PLE_PRIOR_CONV_HISTORY, {"qfn_ple_prior_conv_history", {"qfn_requests", "9", "qfn_trunk"}, "fp32", BufferSemantic::InOut}, Qwen4ExpBufferPlacement::PLEMTP},
    {BufferId::QFN_PLE_TRANSACTION_LIVE, {"qfn_ple_transaction_live", {"qfn_requests"}, "int32", BufferSemantic::InOut}, Qwen4ExpBufferPlacement::PLEMTP},
    {BufferId::QFN_PLE_RETAINED_TOKENS, {"qfn_ple_retained_tokens", {"qfn_mtp_candidate_rows"}, "int32", BufferSemantic::InOut}, Qwen4ExpBufferPlacement::PLEMTP},
    {BufferId::QFN_PLE_RETAINED_NORMALIZED, {"qfn_ple_retained_normalized", {"qfn_mtp_candidate_rows", "qfn_trunk"}, "fp32", BufferSemantic::InOut}, Qwen4ExpBufferPlacement::PLEMTP},
    {BufferId::QFN_PLE_TRANSACTION_COPY_BYTES, {"qfn_ple_transaction_copy_bytes", {"qfn_requests", "4"}, "int32", BufferSemantic::InOut}, Qwen4ExpBufferPlacement::PLEMTP},
    {BufferId::QFN_QSA_QUERY, {"qfn_qsa_query", {"seq_len", "512"}, "fp32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::Sparse},
    {BufferId::QFN_QSA_CANDIDATE_RAW_KEYS, {"qfn_qsa_candidate_raw_keys", {"qfn_qsa_layers", "qfn_mtp_candidate_rows", "128"}, "fp32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::SparseMTP},
    {BufferId::QFN_QSA_CANDIDATE_COPY_BYTES, {"qfn_qsa_candidate_copy_bytes", {"qfn_qsa_layers", "qfn_requests", "2"}, "int32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::SparseMTP},
    {BufferId::QFN_QSA_TRANSACTION_PRIOR, {"qfn_qsa_transaction_prior", {"qfn_qsa_layers", "qfn_requests", "qfn_qsa_prior_words"}, "int32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::SparseMTP},
    {BufferId::QFN_QSA_TRANSACTION_PRIOR_TRAFFIC, {"qfn_qsa_transaction_prior_traffic", {"qfn_qsa_layers", "qfn_requests", "2"}, "int32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::SparseMTP},
    {BufferId::QFN_QSA_RAW_KEY, {"qfn_qsa_raw_key", {"seq_len", "128"}, "fp32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::Sparse},
    {BufferId::QFN_QSA_PREPARED_QUERY, {"qfn_qsa_prepared_query", {"seq_len", "512"}, "fp32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::Sparse},
    {BufferId::QFN_QSA_SCORES, {"qfn_qsa_scores", {"seq_len", "qfn_pools"}, "fp32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::Sparse},
    {BufferId::QFN_QSA_SELECTED, {"qfn_qsa_selected", {"seq_len", "2051"}, "int32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::Sparse},
    {BufferId::QFN_QSA_SELECTED_COUNT, {"qfn_qsa_selected_count", {"seq_len", "1"}, "int32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::Sparse},
    {BufferId::QFN_QSA_PARTIALS, {"qfn_qsa_partials", {"qfn_partial_words"}, "int32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::SparseGPU},
    {BufferId::QFN_QSA_OBSERVATIONS, {"qfn_qsa_observations", {"seq_len", "qfn_observation_words"}, "int32", BufferSemantic::Scratch}, Qwen4ExpBufferPlacement::Sparse},
    {BufferId::QFN_QSA_REPLAY, {"qfn_qsa_replay", {"qfn_replay_words"}, "int32", BufferSemantic::InOut}, Qwen4ExpBufferPlacement::Sparse},
    {BufferId::QFN_QSA_POOLED_KEYS, {"qfn_qsa_pooled_keys", {"qfn_qsa_requests", "qfn_pools", "128"}, "fp32", BufferSemantic::InOut}, Qwen4ExpBufferPlacement::Sparse},
    {BufferId::QFN_QSA_PARTIAL_KEYS, {"qfn_qsa_partial_keys", {"qfn_qsa_requests", "4", "128"}, "fp32", BufferSemantic::InOut}, Qwen4ExpBufferPlacement::Sparse},
};
} // namespace

std::span<const Qwen4ExpBufferDeclaration> qwen4ExpBufferDeclarations()
{
    return declarations;
}

namespace
{
/** @brief Validate a main-model envelope before selecting any state or scratch declaration. */
void validatePlacement(const Qwen4ExpMemoryGeometry &geometry, const Qwen4ExpBufferDimensions &dimensions)
{
    if (dimensions.main_layers <= 0 || dimensions.first_layer < 0 || dimensions.layer_count <= 0 ||
        dimensions.first_layer > dimensions.main_layers ||
        dimensions.layer_count > dimensions.main_layers - dimensions.first_layer ||
        dimensions.compute_width <= 0 || dimensions.retained_mtp_candidate_rows < 0 || !dimensions.device.is_valid() ||
        geometry.residual_branches == 0 || geometry.residual_gate_rank == 0 ||
        geometry.residual_width != product(geometry.residual_branches, dimensions.compute_width) ||
        geometry.compression_ratio == 0 || geometry.ple_heads == 0 || geometry.ple_table_row_bytes == 0)
        throw std::invalid_argument("Flash Next arena requires exact main-layer, branch, row, request and native geometry");
    qwen4exp::validateQwenSparseAttentionPhaseSchedule(dimensions.attention_phase_schedule,dimensions.device.type);
    if(dimensions.attention_phase_schedule==qwen4exp::QwenSparseAttentionPhaseSchedule::Prefill8Decode33 &&
       dimensions.request_capacity!=1)
        throw std::invalid_argument("QSA prefill8_decode33 arena requires the canonical single-request family");
    if (dimensions.ngram_lookup)
    {
        switch (*dimensions.ngram_lookup)
        {
        case ImmutableRowLookupExecution::HostProvider: break;
        case ImmutableRowLookupExecution::FixedHeadShardGPU:
        case ImmutableRowLookupExecution::ResidentGPU:
        case ImmutableRowLookupExecution::AdaptiveGPUCache:
            if (!dimensions.device.is_gpu())
                throw std::invalid_argument("Flash Next VRAM lookup requires a GPU PLE consumer");
            break;
        default: throw std::invalid_argument("Flash Next arena has an invalid ngram execution contract");
        }
    }
    const bool fixed=dimensions.ngram_lookup==ImmutableRowLookupExecution::FixedHeadShardGPU;
    if(fixed ? (dimensions.ngram_fixed_head_participants!=2 && dimensions.ngram_fixed_head_participants!=4 && dimensions.ngram_fixed_head_participants!=8)
             : dimensions.ngram_fixed_head_participants!=0)
        throw std::invalid_argument("Flash Next fixed-head execution requires its exact native cohort degree");
}

/** @brief Formula evaluation additionally requires the admitted runtime envelope. */
void validate(const Qwen4ExpMemoryGeometry &geometry, const Qwen4ExpBufferDimensions &dimensions)
{
    validatePlacement(geometry, dimensions);
    if (dimensions.context_tokens <= 0 || dimensions.request_capacity <= 0 ||
        dimensions.flattened_rows <= 0 || dimensions.local_query_heads <= 0)
        throw std::invalid_argument("Flash Next arena formulas require positive context, request, row and local-head capacity");
}

/** @brief Resolve a graph's already-admitted envelope, without deriving request slots from rows. */
Qwen4ExpBufferDimensions dimensionsFor(const GraphConfig &model, int flattened_rows)
{
    if (!model.flash_next) throw std::invalid_argument("Flash Next graph lacks immutable metadata");
    const int ple = static_cast<int>(model.flash_next->ngrams().decoder_layer);
    const bool owns_ple = model.pp_layer_offset <= ple && ple < model.pp_layer_offset + model.n_layers;
    std::optional<ImmutableRowLookupExecution> storage;
    if (owns_ple)
    {
        if (!model.ngram_table_placement)
            throw std::invalid_argument("Flash Next graph allocation requires resolved ngram placement");
        storage = model.ngram_table_placement->lookupForConsumer(
            {model.ngram_table_placement->worldRank(), model.default_device});
    }
    return {.main_layers = static_cast<int>(model.flash_next->mainLayers()),
        .first_layer = model.pp_layer_offset, .layer_count = model.n_layers, .compute_width = model.d_model,
        .context_tokens = model.max_seq_len, .request_capacity = model.max_request_count,
        .flattened_rows = flattened_rows, .local_query_heads = model.local_n_heads,
        .retained_mtp_candidate_rows = retainsMTPGraphCapacity(model.mtp)
            ? resolveMTPRetainedTargetQueryRows(model.mtp) : 0,
        .device = model.default_device, .attention_phase_schedule=model.attention_phase_schedule, .ngram_lookup = storage,
        .ngram_fixed_head_participants = storage==ImmutableRowLookupExecution::FixedHeadShardGPU
            ? static_cast<int>(model.ngram_table_placement->consumers().size()) : 0};
}

/** @brief Check every multiplication before using the generic allocator's scalar-byte view. */
std::size_t resolvedBytes(const BufferSpec &spec, const GraphResolverConfig &resolver)
{
    const auto resolved = BufferAllocator::resolve(spec, resolver);
    if (resolved.dtype != "fp32" && resolved.dtype != "float32" && resolved.dtype != "int32")
        throw std::invalid_argument("Flash Next arena declaration has an unsupported scalar byte contract: " + spec.name);
    std::size_t count = resolved.shape.empty() ? 0 : 1;
    for (const auto dimension : resolved.shape) count = product(count, dimension);
    return product(count, sizeof(std::int32_t));
}

/** @brief Sum exact additional arena owners without an anonymous reserve. */
std::size_t add(std::size_t left, std::size_t right)
{
    if (right > std::numeric_limits<std::size_t>::max() - left)
        throw std::overflow_error("Flash Next arena byte sum exceeds addressable capacity");
    return left + right;
}
} // namespace

void applyQwen4ExpBufferLayout(GraphSchema &schema, const GraphConfig &model)
{
    const auto exact = model.flash_next
        ? Qwen4ExpMemoryGeometry::fromConfig(*model.flash_next)
        : throw std::invalid_argument("Flash Next schema lacks immutable metadata");
    applyQwen4ExpBufferLayout(schema, exact, dimensionsFor(model, model.max_activation_rows));
}

void applyQwen4ExpBufferLayout(GraphSchema &schema, const Qwen4ExpMemoryGeometry &geometry,
                             const Qwen4ExpBufferDimensions &dimensions)
{
    validatePlacement(geometry, dimensions);
    const int first = dimensions.first_layer, end = first + dimensions.layer_count;
    const bool owns_ple = static_cast<int>(geometry.ple_decoder_layer) >= first &&
                          static_cast<int>(geometry.ple_decoder_layer) < end;
    const bool owns_sparse = std::any_of(geometry.sparse_decoder_layers.begin(), geometry.sparse_decoder_layers.end(),
        [first, end](auto layer) { return static_cast<int>(layer) >= first && static_cast<int>(layer) < end; });
    const bool gpu = dimensions.device.is_gpu();
    const bool host_storage = dimensions.ngram_lookup && *dimensions.ngram_lookup == ImmutableRowLookupExecution::HostProvider;
    const bool fixed_storage = dimensions.ngram_lookup==ImmutableRowLookupExecution::FixedHeadShardGPU;
    const auto selected = [owns_ple, owns_sparse, gpu, host_storage, fixed_storage, retained_rows = dimensions.retained_mtp_candidate_rows](Qwen4ExpBufferPlacement placement)
    {
        switch (placement)
        {
        case Qwen4ExpBufferPlacement::EveryParticipant: return true;
        case Qwen4ExpBufferPlacement::PLE: return owns_ple;
        case Qwen4ExpBufferPlacement::PLEHostStorage: return owns_ple && host_storage;
        case Qwen4ExpBufferPlacement::PLEGPUHostStorage: return owns_ple && gpu && host_storage;
        case Qwen4ExpBufferPlacement::PLEFixedHeadStorage: return owns_ple && gpu && fixed_storage;
        case Qwen4ExpBufferPlacement::Sparse: return owns_sparse;
        case Qwen4ExpBufferPlacement::SparseGPU: return owns_sparse && gpu;
        case Qwen4ExpBufferPlacement::SparseMTP: return owns_sparse && retained_rows > 0;
        case Qwen4ExpBufferPlacement::PLEMTP: return owns_ple && retained_rows > 0;
        }
        throw std::invalid_argument("Unknown Flash Next buffer placement");
    };
    const auto widen = [](BufferSpec &spec)
    {
        if (spec.name == "hidden" || spec.name == "residual" || spec.name == "mtp_norm_hidden" ||
            spec.name == "mtp_projected" || spec.name == "mtp_hidden")
            spec.shape.back() = "qfn_trunk";
        else if (spec.name == "mtp_concat") spec.shape.back() = "qfn_trunk * 2";
    };
    for (auto &spec : schema.model_buffers) widen(spec);
    for (auto &spec : schema.layer_buffers) widen(spec);
    for (const auto &entry : declarations)
    {
        if (!selected(entry.placement)) continue;
        const auto duplicate = [&entry](const BufferSpec &spec) { return spec.name == entry.spec.name; };
        if (std::any_of(schema.layer_buffers.begin(), schema.layer_buffers.end(), duplicate) ||
            std::any_of(schema.model_buffers.begin(), schema.model_buffers.end(), duplicate))
            throw std::invalid_argument("Flash Next buffer layout was applied twice: " + entry.spec.name);
        schema.layer_buffers.push_back(entry.spec);
    }
}

void applyQwen4ExpBufferGeometry(GraphResolverConfig &resolver, const GraphConfig &model)
{
    if (!model.flash_next) throw std::invalid_argument("Flash Next resolver lacks immutable metadata");
    auto dimensions = dimensionsFor(model, resolver.seq_len);
    dimensions.local_query_heads = resolver.local_n_heads;
    applyQwen4ExpBufferGeometry(resolver, Qwen4ExpMemoryGeometry::fromConfig(*model.flash_next), dimensions);
}

void applyQwen4ExpBufferGeometry(GraphResolverConfig &resolver, const Qwen4ExpMemoryGeometry &geometry,
                               const Qwen4ExpBufferDimensions &dimensions)
{
    validate(geometry, dimensions);
    if (resolver.seq_len != dimensions.flattened_rows || resolver.d_model != dimensions.compute_width ||
        resolver.local_n_heads != dimensions.local_query_heads)
        throw std::invalid_argument("Flash Next resolver and admitted participant envelope disagree");
    const auto context = static_cast<std::size_t>(dimensions.context_tokens);
    const auto requests = static_cast<std::size_t>(dimensions.request_capacity);
    const auto heads = static_cast<std::size_t>(dimensions.local_query_heads);
    std::size_t sparse_layers = 0;
    for (const auto layer : geometry.sparse_decoder_layers)
        sparse_layers += static_cast<int>(layer) >= dimensions.first_layer &&
            static_cast<int>(layer) < dimensions.first_layer + dimensions.layer_count;
    const auto row_bytes = product(geometry.ple_heads, geometry.ple_table_row_bytes);
    if (row_bytes > std::numeric_limits<std::size_t>::max() - sizeof(std::int32_t) + 1)
        throw std::overflow_error("Flash Next native row word rounding overflows");
    resolver.custom_formulas["qfn_trunk"] = geometry.residual_width;
    resolver.custom_formulas["qfn_hc_rank"] = geometry.residual_gate_rank;
    resolver.custom_formulas["qfn_native_words"] = (row_bytes + sizeof(std::int32_t) - 1) / sizeof(std::int32_t);
    const auto degree=static_cast<std::size_t>(dimensions.ngram_fixed_head_participants);
    resolver.custom_formulas["qfn_fixed_head_participants"]=degree;
    if(degree) {
        const auto selected=product(static_cast<std::size_t>(dimensions.flattened_rows),row_bytes);
        const auto local=product(static_cast<std::size_t>(dimensions.flattened_rows),
            product(geometry.ple_heads/degree,geometry.ple_table_row_bytes));
        if(geometry.ple_heads!=16 || geometry.ple_heads%degree || selected>std::numeric_limits<std::size_t>::max()-3 ||
            local>std::numeric_limits<std::size_t>::max()-3)
            throw std::overflow_error("Flash Next fixed-head native packet extent overflows or loses original heads");
        // Only allocation is rounded to INT32. Wire stride is exactly M*H*B;
        // no per-row, per-rank or allocation-capacity padding is communicated.
        resolver.custom_formulas["qfn_fixed_local_words"]=(local+3)/4;
        resolver.custom_formulas["qfn_fixed_gathered_words"]=(selected+3)/4;
    } else {
        resolver.custom_formulas["qfn_fixed_local_words"]=0;
        resolver.custom_formulas["qfn_fixed_gathered_words"]=0;
    }
    resolver.custom_formulas["qfn_requests"] = requests;
    resolver.custom_formulas["qfn_pools"] = context / geometry.compression_ratio +
        (context % geometry.compression_ratio != 0);
    resolver.custom_formulas["qfn_qsa_layers"] = sparse_layers;
    resolver.custom_formulas["qfn_qsa_prior_words"] = sizeof(qwen4exp::QwenSparseTransactionPrior)/sizeof(std::int32_t);
    resolver.custom_formulas["qfn_mtp_candidate_rows"] = static_cast<std::size_t>(dimensions.retained_mtp_candidate_rows);
    resolver.custom_formulas["qfn_qsa_requests"] = product(sparse_layers, requests);
    // One scalar bank is shared by mutually exclusive phase graphs and layers.
    // Keep existing8F bytes for Uniform. The explicit diagnostic adds only the
    // canonical33D alternative, not33F and not a separate allocation ledger.
    const auto prefill_records=product(static_cast<std::size_t>(dimensions.flattened_rows),std::size_t{8});
    auto records_per_head=prefill_records;
    if(dimensions.attention_phase_schedule==qwen4exp::QwenSparseAttentionPhaseSchedule::Prefill8Decode33)
        records_per_head=std::max(records_per_head,product(std::size_t{33},
            static_cast<std::size_t>(qwen4exp::qwenSparseAttentionDecodeRowCapacity(dimensions.retained_mtp_candidate_rows))));
    static_assert(sizeof(qwen4exp::QwenSparseAttentionPartial)%sizeof(std::int32_t)==0);
    resolver.custom_formulas["qfn_partial_words"] = dimensions.device.is_gpu()
        ? product(product(heads,records_per_head),sizeof(qwen4exp::QwenSparseAttentionPartial)/sizeof(std::int32_t)) : 0;
    resolver.custom_formulas["qfn_observation_words"] =
        product(heads, sizeof(qwen4exp::QwenSparseAttentionObservation) / sizeof(std::int32_t));
    // Request cursors are independent of graph row capacity. CPU kernels never
    // consume AttentionDeviceParams, so only GPU geometry retains that bank.
    const auto replay_words = product(requests, sizeof(qwen4exp::QwenSparseReplay) / sizeof(std::int32_t));
    const auto row_words = dimensions.device.is_gpu()
        ? product(static_cast<std::size_t>(dimensions.flattened_rows),
            sizeof(attention::AttentionDeviceParams) / sizeof(std::int32_t)) : 0;
    resolver.custom_formulas["qfn_replay_words"] = add(replay_words, row_words);
    for (const auto &entry : declarations)
    {
        const auto [found, inserted] = resolver.buffer_name_to_id.emplace(entry.spec.name, entry.id);
        if (!inserted && found->second != entry.id)
            throw std::invalid_argument("Flash Next buffer name is already bound to another identity: " + entry.spec.name);
    }
}

std::size_t qwen4ExpArenaExtensionBytes(const GraphSchema &common_schema, GraphResolverConfig resolver,
    const Qwen4ExpMemoryGeometry &geometry, const Qwen4ExpBufferDimensions &dimensions)
{
    applyQwen4ExpBufferGeometry(resolver, geometry, dimensions);
    GraphSchema extended = common_schema;
    applyQwen4ExpBufferLayout(extended, geometry, dimensions);
    std::size_t bytes = 0;
    const auto accumulate = [&](const auto &original, const auto &updated)
    {
        for (const auto &spec : updated)
        {
            const auto prior = std::find_if(original.begin(), original.end(),
                [&](const auto &candidate) { return candidate.name == spec.name; });
            if (prior != original.end() && prior->shape == spec.shape && prior->dtype == spec.dtype) continue;
            const auto next = resolvedBytes(spec, resolver);
            const auto before = prior == original.end() ? 0 : resolvedBytes(*prior, resolver);
            if (before > next) throw std::logic_error("Flash Next arena extension unexpectedly shrinks a shared role");
            bytes = add(bytes, next - before);
        }
    };
    accumulate(common_schema.model_buffers, extended.model_buffers);
    accumulate(common_schema.layer_buffers, extended.layer_buffers);
    return bytes;
}

std::size_t qwen4ExpLookupArenaBytes(GraphResolverConfig resolver,
    const Qwen4ExpMemoryGeometry &geometry, const Qwen4ExpBufferDimensions &dimensions)
{
    if (!dimensions.ngram_lookup)
        throw std::invalid_argument("Flash Next lookup candidate requires a ngram execution contract");
    applyQwen4ExpBufferGeometry(resolver, geometry, dimensions);
    GraphSchema candidate;
    applyQwen4ExpBufferLayout(candidate, geometry, dimensions);
    std::size_t bytes = 0;
    for (const auto &spec : candidate.layer_buffers)
    {
        const auto declaration = std::find_if(declarations.begin(), declarations.end(),
            [&](const auto &entry) { return entry.spec.name == spec.name; });
        if (declaration != declarations.end() &&
            (declaration->placement == Qwen4ExpBufferPlacement::PLEHostStorage ||
             declaration->placement == Qwen4ExpBufferPlacement::PLEGPUHostStorage ||
             declaration->placement == Qwen4ExpBufferPlacement::PLEFixedHeadStorage))
            bytes = add(bytes, resolvedBytes(spec, resolver));
    }
    return bytes;
}
} // namespace llaminar2
