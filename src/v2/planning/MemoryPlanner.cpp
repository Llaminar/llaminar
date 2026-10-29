/**
 * @file MemoryPlanner.cpp
 * @brief Exact memory admission arithmetic for production graph families.
 *
 * The implementation turns the typed lifetime declarations in
 * @ref DevicePlanConfig into checked byte counts.  In particular, sparse MoE
 * packet storage is not guessed from a model name: a serial ExpertOverlay
 * participant family has one reusable packet per participant, whereas an
 * independently retained graph family has one per participating layer.
 * CPU serial storage declarations survive admission and runtime projection.
 * Per-layer CPU packet roots retain their own exact backing allocation claims.
 */

#include "loaders/AdaptiveImmutableLookupGeometry.h"
#include "planning/MemoryPlanner.h"
#include "memory/CPUSerialExpertStorageLayout.h"
#include "memory/CPUGroupedMoEStorageLayout.h"
#include "memory/CPURouterQ8StorageLayout.h"
#include "loaders/WeightIdentity.h"
#include "loaders/ImmutableRowTable.h"
#include "loaders/CpuAdaptiveImmutableRowCache.h"
#include <map>
#include "loaders/ImmutableRowTableUploadGeometry.h"
#include "QwenSparseSnapshotMemory.h"
#include "execution/qwen4exp/ImmutableLookupService.h"
#include "collective/DeviceCountedAllGather.h"
#include "execution/moe/MoEProjectionArenaGeometry.h"
#include "planning/WeightMemoryEstimator.h"
#include "planning/KVCacheMemoryEstimator.h"
#include "planning/PersistentStateMemoryEstimator.h"
#include "planning/ActivationMemoryEstimator.h"
#include "planning/CollectiveMemoryEstimator.h"
#include "planning/WorkspaceMemoryEstimator.h"
#include "kernels/HybridGDNStateGeometry.h"
#include "planning/CapturedGraphMemoryEstimator.h"
#include "execution/local_execution/engine/PrefillBucketUtils.h"
#include "execution/local_execution/graph/SchemaFactoryRegistry.h"
#include "execution/prefix_cache/PrefixArchiveIOGeometry.h"
#include "execution/prefix_cache/PrefixPayloadLayout.h"
#include "utils/Logger.h"

#include "fort.hpp"

#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <limits>
#include <set>

namespace llaminar2
{

namespace
{

/**
 * @brief Read immutable attention visibility from the declarative model schema.
 * @param architecture Registered model architecture being admitted.
 * @return Full causal visibility only when every declared attention reader has it.
 *
 * Admission must not infer visibility from a model name or assume the runtime
 * stage's default. Named recurrent templates may contain no attention reader;
 * every real attention reader contributes its explicit causal/window policy.
 * Additional input operands exclude the unmasked Q/K/V specialization.
 */
attention::AttentionVisibilityPolicy modelAttentionVisibility(const std::string &architecture)
{
    const auto schema = SchemaFactoryRegistry::getFactory(architecture)->createSchema();
    bool seen_attention = false;
    bool full_context = true;
    const auto inspect = [&](const LayerTemplate &layer)
    {
        for (const auto &stage : layer.attention_stages)
        {
            if (stage.type != StageType::AttentionCompute) continue;
            seen_attention = true;
            full_context = full_context &&
                stage.causal.has_value() && stage.window_size.has_value() &&
                attention::attentionVisibilityPolicy(*stage.causal, *stage.window_size,
                    stage.inputs.size() != 3) == attention::AttentionVisibilityPolicy::CausalFullContext;
        }
    };
    inspect(schema.layer_template);
    for (const auto &[name, layer] : schema.named_templates) inspect(layer);
    return seen_attention && full_context
        ? attention::AttentionVisibilityPolicy::CausalFullContext
        : attention::AttentionVisibilityPolicy::General;
}

std::string formatMB(size_t bytes)
{
    double mb = static_cast<double>(bytes) / (1024.0 * 1024.0);
    std::ostringstream ss;
    if (mb >= 1024.0)
    {
        ss << std::fixed << std::setprecision(1) << (mb / 1024.0) << " GB";
    }
    else
    {
        ss << std::fixed << std::setprecision(0) << mb << " MB";
    }
    return ss.str();
}

/**
 * @brief Multiply two memory contributions without allowing size_t wraparound.
 *
 * Memory planning is an admission decision.  A wrapped contribution would make
 * an impossible graph look affordable, so reject it before it reaches the
 * device allocator.
 */
size_t checkedMultiply(
    size_t left,
    size_t right,
    const char* contribution)
{
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left)
    {
        throw std::runtime_error(
            std::string("Memory byte overflow while sizing ") + contribution);
    }
    return left * right;
}

/**
 * @brief Add two memory contributions without allowing size_t wraparound.
 *
 * The compact ExpertOverlay tensors are additive to the dense continuation
 * arena: they are separate graph-owned allocations with independent stable
 * addresses.  Keep the arithmetic checked for the same reason as
 * @ref checkedMultiply.
 */
size_t checkedAdd(
    size_t left,
    size_t right,
    const char* contribution)
{
    if (right > std::numeric_limits<size_t>::max() - left)
    {
        throw std::runtime_error(
            std::string("Memory byte overflow while sizing ") + contribution);
    }
    return left + right;
}

/**
 * @brief Merge one logical graph bill into its unique physical allocator row.
 *
 * Model topology may expose several serial logical roles on one CPU/GPU. The
 * memory plan is physical, so emitting duplicate rows would let downstream
 * callers compare each partial subtotal independently and over-admit the real
 * allocator. This helper combines typed owner lines while requiring the exact
 * same rank/device capacity observation.
 */
void appendPhysicalPlan(
    MemoryPlan &plan,
    DeviceMemoryPlan incoming)
{
    const auto &incoming_resource = incoming.bom().resource();
    const auto found = std::find_if(
        plan.devices.begin(),
        plan.devices.end(),
        [&](const DeviceMemoryPlan &candidate)
        {
            const auto &resource = candidate.bom().resource();
            return resource.world_rank == incoming_resource.world_rank &&
                   resource.device == incoming_resource.device;
        });
    if (found == plan.devices.end())
    {
        plan.devices.push_back(std::move(incoming));
        return;
    }
    if (!found->bom().resource().sameAllocator(incoming_resource))
    {
        throw std::invalid_argument(
            "Logical memory plans disagree on one physical allocator observation");
    }

    PhysicalMemoryBOMBuilder combined(found->bom());
    for (std::size_t index = 0;
         index < PhysicalMemoryBOM::ownerCount();
         ++index)
    {
        const auto owner = PhysicalMemoryBOM::ownerAt(index);
        const auto &charge = incoming.bom().charge(owner);
        combined.add(
            owner,
            charge.planned_bytes,
            charge.already_resident_bytes);
    }
    *found = DeviceMemoryPlan(
        combined.build(),
        std::max(found->max_seq_len(), incoming.max_seq_len()),
        std::max(
            found->activation_seq_len(),
            incoming.activation_seq_len()));
}

/**
 * @brief Count selected routed layers that require a compact local-expert packet.
 *
 * The owner residency map is the source of truth for whether a layer can
 * execute a local expert stage on this device.  A layer with no selected
 * experts has no compact packet and must not affect the admission result.
 */
size_t participatingRoutedLayerCount(
    const ModelMemoryProfile& profile,
    const DevicePlanConfig& config,
    int first_layer,
    int last_layer)
{
    (void)profile;
    if (!config.weight_residency.selectsRoutedExperts())
    {
        throw std::invalid_argument(
            "Routed-expert participant memory role requires selected weight residency");
    }

    size_t participating_layers = 0;
    for (int layer = first_layer; layer <= last_layer; ++layer)
    {
        if (config.weight_residency.selectedRoutedExpertsForLayer(layer) > 0)
            ++participating_layers;
    }
    if (participating_layers == 0)
        return 0;

    return participating_layers;
}

/**
 * @brief Size one compact local-expert packet at the declared graph geometry.
 *
 * The packet contains compact hidden and output vectors plus one selected
 * expert index and weight per route. CPU hidden/output storage has one row
 * per input token while routing metadata retains its independent top-k axis.
 * The common declaration owns their separate allocator rounding. GPU packet
 * admission retains its existing route-expanded geometry.
 */
size_t compactLocalExpertPacketBytes(
    const ModelMemoryProfile& profile,
    const DevicePlanConfig& config,
    int activation_rows,
    bool rows_are_flattened = false)
{
    if (!config.weight_residency.selectsRoutedExperts())
    {
        throw std::invalid_argument(
            "Routed-expert packet sizing requires selected weight residency");
    }

    const size_t batches = static_cast<size_t>(std::max(1, config.batch_size));
    const size_t rows = static_cast<size_t>(std::max(1, activation_rows));
    const size_t top_k = static_cast<size_t>(std::max(1, profile.expert_used_count));
    const size_t d_model = static_cast<size_t>(std::max(1, profile.d_model));

    size_t route_rows = rows_are_flattened
                            ? rows
                            : checkedMultiply(
                                  batches,
                                  rows,
                                  "expert participant rows");
    if(config.device.is_cpu())
        return CPUCompactPacketStorageLayout::declare(route_rows,profile.d_model,
            profile.expert_used_count,config.device).physicalBytes();
    route_rows = checkedMultiply(route_rows, top_k, "expert participant routes");
    const size_t fp32_values_per_route = 2u * d_model + 2u;
    size_t bytes = checkedMultiply(
        route_rows,
        fp32_values_per_route,
        "expert participant compact tensors");
    return checkedMultiply(bytes, sizeof(float), "expert participant FP32 tensors");
}

/**
 * @brief Price every immutable power-of-two compact family through a row bound.
 *
 * A retained GPU graph has one fixed launch geometry and tensor-coherence
 * authority per route bucket. The sum is strictly less than twice the largest
 * power-of-two family, but pricing only that largest family understates the
 * real VRAM allocation and can admit a model that later fails during capture.
 */
size_t serialCompactRouteFamilyBytes(
    const ModelMemoryProfile& profile,
    int flattened_activation_rows)
{
    if (flattened_activation_rows <= 0)
    {
        throw std::invalid_argument(
            "Serial routed-expert compact rows must be positive");
    }

    const size_t top_k =
        static_cast<size_t>(std::max(1, profile.expert_used_count));
    const size_t rows =
        static_cast<size_t>(flattened_activation_rows);
    const size_t maximum_routes = checkedMultiply(
        rows,
        top_k,
        "expert participant serial-family route capacity");
    const size_t values_per_route =
        checkedAdd(
            checkedMultiply(
                2u,
                static_cast<size_t>(std::max(1, profile.d_model)),
                "expert participant hidden/output values"),
            2u,
            "expert participant route metadata values");
    const size_t bytes_per_route = checkedMultiply(
        values_per_route,
        sizeof(float),
        "expert participant bytes per compact route");

    size_t family_capacity = 1;
    size_t total_bytes = 0;
    while (family_capacity < maximum_routes)
    {
        total_bytes = checkedAdd(
            total_bytes,
            checkedMultiply(
                family_capacity,
                bytes_per_route,
                "expert participant compact route family"),
            "expert participant compact route-family ladder");
        if (family_capacity > std::numeric_limits<size_t>::max() / 2u)
        {
            throw std::overflow_error(
                "Expert participant compact route-family ladder overflowed");
        }
        family_capacity *= 2u;
    }
    return checkedAdd(
        total_bytes,
        checkedMultiply(
            maximum_routes,
            bytes_per_route,
            "expert participant terminal compact route family"),
        "expert participant complete compact route-family ladder");
}

/**
 * @brief Size graph-lifetime compact tensors owned by local expert stages.
 *
 * @ref RoutedExpertCompactBufferLifetime is an explicit graph-family ABI:
 * layer-owned packets cannot alias, while a serial participant family has
 * explicit producer/consumer ordering and can share one fixed packet across
 * all its layers and graph roles.  Reject a serial declaration without its
 * participant count rather than silently pricing it as an arbitrary fallback.
 */
size_t routedParticipantActivationBytes(
    const ModelMemoryProfile& profile,
    const DevicePlanConfig& config,
    int first_layer,
    int last_layer,
    int activation_rows)
{
    const size_t participating_layers = participatingRoutedLayerCount(
        profile, config, first_layer, last_layer);
    switch (config.routed_expert_compact_buffer_lifetime)
    {
    case RoutedExpertCompactBufferLifetime::PerLayerGraphOwned:
        if (participating_layers == 0)
            return 0;
        return checkedMultiply(
            compactLocalExpertPacketBytes(
                profile, config, activation_rows),
            participating_layers,
            "expert participant layer-owned compact tensors");
    case RoutedExpertCompactBufferLifetime::SerialFamilyPerParticipant:
    {
        if (config.serial_routed_expert_participant_count <= 0)
        {
            if (participating_layers == 0)
                return 0;
            throw std::invalid_argument(
                "Serial routed-expert compact storage requires a positive "
                "local participant count when selected experts are resident");
        }

        /*
         * A captured sparse-collective participant keeps one stable packet
         * even when its initial live-expert quota is zero. Residency epochs
         * may change data ownership, but they cannot add graph topology or
         * allocate a packet in the hot path.
         */
        if(config.device.is_cpu())
        {
            if(!config.cpu_serial_expert_storage ||
                config.cpu_serial_expert_storage->allocator!=PhysicalMemoryAllocatorIdentity{config.world_rank,config.device} ||
                config.cpu_serial_expert_storage->participant_ids.size()!=size_t(config.serial_routed_expert_participant_count))
                throw std::invalid_argument("CPU serial planning requires the exact frozen endpoint/worker layout");
            return config.cpu_serial_expert_storage->physicalBytes();
        }
        const size_t packet_bytes =
            config.serial_routed_expert_compact_rows > 0
                ? serialCompactRouteFamilyBytes(
                      profile,
                      config.serial_routed_expert_compact_rows)
                : compactLocalExpertPacketBytes(
                      profile,
                      config,
                      activation_rows);
        return checkedMultiply(
            packet_bytes,
            static_cast<size_t>(config.serial_routed_expert_participant_count),
            "expert participant serial-family compact tensors");
    }
    }
    throw std::logic_error("Unknown routed-expert compact-buffer lifetime");
}

/** @return Whether this actual continuation owns the model's PLE decoder block. */
bool ownsNgramLookup(const ModelMemoryProfile &profile, const DevicePlanConfig &cfg)
{
    const auto last = cfg.last_layer < 0 ? profile.n_layers - 1 : cfg.last_layer;
    return profile.flash_next && cfg.execution_role == DeviceExecutionMemoryRole::ContinuationGraph &&
        cfg.first_layer <= static_cast<int>(profile.flash_next->ple_decoder_layer) &&
        last >= static_cast<int>(profile.flash_next->ple_decoder_layer);
}
/** @return Checked original source geometry without reading native table bytes. */
ImmutableRowTableExtent ngramExtent(const ModelMemoryProfile &profile)
{
    if (!profile.flash_next || profile.flash_next->ple_table_row_bytes > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Ngram planning requires representable original source metadata");
    const auto &g = *profile.flash_next;
    const auto extent = ImmutableRowTableExtent::fromNative(static_cast<GGUFTensorType>(g.ple_table_gguf_type),
        g.ple_table_rows, static_cast<std::size_t>(g.ple_table_row_bytes));
    if (extent.nativeBytes() != g.ple_table_native_bytes)
        throw std::invalid_argument("Ngram native source extent disagrees with exact model metadata");
    return extent;
}
/** @return Original source geometry only; planning never opens a file or reads native payload. */
ImmutableRowTableSource ngramNativeSource(const ModelMemoryProfile &profile)
{
    const auto extent=ngramExtent(profile);const auto &source=*profile.flash_next;
    ImmutableRowTableSource result{.rows=source.ple_table_rows,.row_elements=source.ple_row_elements,
        .format=static_cast<GGUFTensorType>(source.ple_table_gguf_type)};
    if(result.rowBytes()!=extent.rowBytes())
        throw std::invalid_argument("Adaptive original native row elements disagree with the source byte extent");
    return result;
}
/** @brief Reject unsupported topology before any mixed-domain reservation or native preparation. */
void requireGPUAdaptiveDomain(const ModelMemoryProfile &profile,const DevicePlanConfig &cfg)
{
    if(!profile.flash_next || profile.mainLayerCount()!=48 || profile.flash_next->ple_heads!=16 ||
        !cfg.device.is_gpu() || cfg.total_shards!=1 || cfg.batch_size!=1 || cfg.first_layer!=0 ||
        (cfg.last_layer>=0?cfg.last_layer:profile.mainLayerCount()-1)!=profile.mainLayerCount()-1 ||
        cfg.execution_role!=DeviceExecutionMemoryRole::ContinuationGraph || cfg.local_pipeline_backend ||
        !cfg.captured_pipeline_boundaries.empty() || cfg.ngram_tp_topology!=ImmutableRowTableTPTopology::None ||
        !cfg.associated_host_memory || !cfg.associated_host_memory->valid() ||
        !cfg.associated_host_memory->device.is_cpu() || cfg.associated_host_memory->world_rank!=cfg.world_rank)
        throw std::invalid_argument("AdaptiveVRAMRAMSSD supports one full-main rank-local GPU with an explicit local CPU source, without TP/PP");
}
/** @return Exact original loader geometry equality without reading struct padding. */
bool sameUploadGeometry(const GPUWeightLoadMemoryGeometry &a,const GPUWeightLoadMemoryGeometry &b)
{
    return a.maximum_source_bytes==b.maximum_source_bytes && a.staging_slot_bytes==b.staging_slot_bytes &&
        a.staging_slot_stride_bytes==b.staging_slot_stride_bytes && a.staging_bytes==b.staging_bytes &&
        a.host_staging_bytes==b.host_staging_bytes && a.staging_stream_count==b.staging_stream_count;
}
/** @return Concrete setup placement for an actual consumer, or the unresolved base marker. */
std::optional<ImmutableRowLookupExecution> ngramLookup(const ModelMemoryProfile &profile, const DevicePlanConfig &cfg)
{
    if (!ownsNgramLookup(profile, cfg) || !cfg.ngram_table_placement) return std::nullopt;
    if (cfg.ngram_table_placement->extent() != ngramExtent(profile))
        throw std::invalid_argument("Resolved ngram placement belongs to another native source extent");
    if(cfg.ngram_execution_policy)cfg.ngram_table_placement->requireRequestedPolicy(*cfg.ngram_execution_policy);
    if((cfg.ngram_table_placement->isHostAdaptive() || cfg.ngram_table_placement->isGPUAdaptive()) && cfg.ngram_host_placement!=cfg.ngram_table_placement->hostPlacement())
        throw std::invalid_argument("Resolved adaptive consumer locality differs from its shared provider");
    return cfg.ngram_table_placement->lookupForConsumer({cfg.world_rank, cfg.device});
}
/**
 * @brief Authenticate pre-quota fixed head execution against actual rank TP provenance.
 * @param profile Original full model source geometry.
 * @param cfg One canonical continuation participant.
 * @return Actual TP degree; zero when this execution is not fixed head.
 * @throws std::invalid_argument For unsupported source, topology, coordinate or backend.
 */
int fixedHeadParticipants(const ModelMemoryProfile &profile,const DevicePlanConfig &cfg)
{
    if(!cfg.ngram_execution_policy || !cfg.ngram_execution_policy->isFixedHeadSharded())return 0;
    cfg.ngram_execution_policy->validate();
    const int p=cfg.total_shards;
    if(!profile.flash_next || profile.mainLayerCount()!=48 || profile.flash_next->ple_heads!=16 ||
       profile.flash_next->ple_head_offsets.size()!=16 || profile.flash_next->ple_head_vocab_sizes.size()!=16 ||
       cfg.ngram_tp_topology!=ImmutableRowTableTPTopology::HomogeneousFullMainRankLocal ||
       !cfg.device.is_gpu() || (p!=2 && p!=4 && p!=8) || cfg.shard_index<0 || cfg.shard_index>=p ||
       cfg.first_layer!=0 || cfg.last_layer!=profile.mainLayerCount()-1 || cfg.batch_size!=1 ||
       cfg.execution_role!=DeviceExecutionMemoryRole::ContinuationGraph || cfg.local_pipeline_backend ||
       !cfg.captured_pipeline_boundaries.empty() || !cfg.local_tp_backend ||
       *cfg.local_tp_backend!=(cfg.device.is_cuda()?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL))
        throw std::invalid_argument("Fixed head lookup requires actual homogeneous full-main native rank-local GPU TP degree2/4/8");
    return p;
}
/** @return Same resident graph row capacity used by the complete device plan. */
int ngramActivationRows(const ModelMemoryProfile &profile, const DevicePlanConfig &cfg)
{
    const int context = cfg.max_seq_len > 0 ? cfg.max_seq_len : profile.max_seq_len;
    int rows = cfg.activation_seq_len > 0 ? cfg.activation_seq_len : context;
    if (context > 0) rows = std::min(rows, context);
    return std::max(1, rows);
}
/** @brief Exact tier-specific bytes for one retained PLE continuation lane. */
struct NgramModeMemory
{
    std::size_t arena = 0, mapped = 0, terminal = 0;
};
/** @return Source-owned tier delta, independent of table payload/placement decisions. */
NgramModeMemory ngramModeMemory(const ModelMemoryProfile &profile, const DevicePlanConfig &cfg,
    ImmutableRowLookupExecution storage)
{
    if (!ownsNgramLookup(profile, cfg)) return {};
    const int rows = ngramActivationRows(profile, cfg);
    const int heads = cfg.tensor_parallel_assignment ? cfg.tensor_parallel_assignment->head_count :
        std::max(1, profile.n_heads / std::max(1, cfg.total_shards));
    const auto arena = ActivationMemoryEstimator::immutableLookupArenaBytes(profile,
        ActivationGraphMemoryGeometry{
            .ngram_lookup = storage, .batch_size = cfg.batch_size, .resident_graph_rows = rows,
            .context_tokens = cfg.max_seq_len > 0 ? cfg.max_seq_len : profile.max_seq_len,
            .local_n_heads = heads, .first_layer = cfg.first_layer, .last_layer = cfg.last_layer,
            .total_shards = cfg.total_shards, .mtp_target_query_rows = cfg.mtp_target_query_rows,
            .retained_mtp_candidate_rows = cfg.mtp_enabled ? cfg.mtp_target_query_rows : 0,
            .generation_request_capacity = cfg.generation_request_capacity,
            .ngram_fixed_head_participants =
                storage == ImmutableRowLookupExecution::FixedHeadShardGPU
                    ? (cfg.ngram_table_placement
                           ? static_cast<int>(cfg.ngram_table_placement->consumers().size())
                           : fixedHeadParticipants(profile, cfg))
                    : 0,
            .attention_arithmetic = cfg.attention_arithmetic,
            .attention_phase_schedule = cfg.attention_phase_schedule},
        cfg.device);
    if(storage==ImmutableRowLookupExecution::AdaptiveGPUCache)
        return {arena,0u,GPUGraphMemoryContract::reservationBytesPerExecutable(cfg.device)};
    if (immutableRowLookupUsesNativeDeviceStorage(storage) || !cfg.device.is_gpu()) return {arena, 0u, 0u};
    const auto &g = *profile.flash_next;
    const auto flattened = checkedMultiply(static_cast<std::size_t>(cfg.batch_size),
        static_cast<std::size_t>(rows), "native lookup flattened graph rows");
    const auto ids = checkedMultiply(flattened, g.ple_heads, "native lookup ID capacity");
    const auto layout = qwen4exp::ImmutableLookupLayout::create(ids, g.ple_table_row_bytes);
    return {arena, qwen4exp::ImmutableLookupService::hostBytes(layout),
        GPUGraphMemoryContract::reservationBytesPerExecutable(cfg.device)};
}

/**
 * @brief Project the actual common activation descriptor shared by base admission and wire proof.
 * @param profile Canonical source geometry.
 * @param cfg One exact continuation graph instance.
 * @param storage Concrete mode when resolved; absent for the common pre-table arena.
 * @return The same typed geometry consumed by ActivationMemoryEstimator.
 */
ActivationGraphMemoryGeometry continuationActivationGeometry(const ModelMemoryProfile &profile,
    const DevicePlanConfig &cfg,std::optional<ImmutableRowLookupExecution> storage)
{
    const bool replicated=std::find(cfg.additional_weight_sets.begin(),cfg.additional_weight_sets.end(),
        AdditionalPersistentWeightSet::ReplicatedDenseDecode)!=cfg.additional_weight_sets.end();
    int heads=cfg.tensor_parallel_assignment?cfg.tensor_parallel_assignment->head_count:profile.n_heads;
    int dff=cfg.tensor_parallel_assignment?cfg.tensor_parallel_assignment->d_ff_count:profile.d_ff;
    int kv=cfg.tensor_parallel_assignment?cfg.tensor_parallel_assignment->kv_head_count:
        (cfg.local_kv_heads>0?cfg.local_kv_heads:profile.n_kv_heads);
    if(cfg.total_shards>1 && !cfg.tensor_parallel_assignment)
    {heads=std::max(1,profile.n_heads/cfg.total_shards);dff=std::max(1,profile.d_ff/cfg.total_shards);}
    if(replicated){heads=profile.n_heads;dff=profile.d_ff;kv=profile.n_kv_heads;}
    if(cfg.total_shards>1 && cfg.local_kv_heads<=0 && !cfg.tensor_parallel_assignment)
        kv=std::max(1,profile.n_kv_heads/cfg.total_shards);
    return {.ngram_lookup = storage,
            .batch_size = cfg.batch_size,
            .resident_graph_rows = ngramActivationRows(profile, cfg),
            .context_tokens = cfg.max_seq_len > 0 ? cfg.max_seq_len : profile.max_seq_len,
            .local_d_ff = dff,
            .local_n_heads = heads,
            .local_n_kv_heads = kv,
            .local_vocab = replicated ? profile.vocab_size
                                      : (cfg.tensor_parallel_assignment
                                             ? cfg.tensor_parallel_assignment->vocab_count
                                             : 0),
            .replicated_dense_decode = replicated,
            .first_layer = cfg.first_layer,
            .last_layer = cfg.last_layer >= 0 ? cfg.last_layer : profile.n_layers - 1,
            .total_shards = cfg.total_shards,
            .routed_compute_policy = cfg.routed_compute_policy,
            .mtp_target_query_rows = cfg.mtp_target_query_rows,
            .retained_mtp_candidate_rows = cfg.mtp_enabled ? cfg.mtp_target_query_rows : 0,
            .mtp_terminal_logits_layout =
                replicated ? MTPTerminalLogitsLayout::FullVocabularyPerParticipant
                           : cfg.mtp_terminal_logits_layout,
            .generation_request_capacity = cfg.generation_request_capacity,
            .ngram_fixed_head_participants =
                storage == ImmutableRowLookupExecution::FixedHeadShardGPU
                    ? (cfg.ngram_table_placement
                           ? static_cast<int>(cfg.ngram_table_placement->consumers().size())
                           : fixedHeadParticipants(profile, cfg))
                    : 0,
            .attention_arithmetic = cfg.attention_arithmetic,
            .attention_phase_schedule = cfg.attention_phase_schedule,
            .mtp_shifted_kv_head_layout = cfg.mtp_shifted_kv_head_layout};
}
/**
 * @brief Project one source-owned mandatory execution alternative.
 * @param profile Existing original source metadata.
 * @param cfg Exact retained PLE consumer geometry and observed allocators.
 * @param storage Concrete mutually exclusive execution mode.
 * @return Execution BOM only, without the original bulk table payload.
 * @throws std::invalid_argument For a hosted GPU consumer without its exact host.
 */
PhysicalMemoryPlan ngramExecutionArm(const ModelMemoryProfile &profile,const DevicePlanConfig &cfg,
    ImmutableRowStorage storage)
{
    auto execution=immutableRowLookupExecution(storage);
    if(cfg.ngram_execution_policy && cfg.ngram_execution_policy->isFixedHeadSharded())
    {
        if(storage!=ImmutableRowStorage::VRAM)
            throw std::invalid_argument("Fixed head execution cannot enter a hosted tier arm");
        (void)fixedHeadParticipants(profile,cfg);
        execution=ImmutableRowLookupExecution::FixedHeadShardGPU;
    }
    if(cfg.ngram_execution_policy && cfg.ngram_execution_policy->isGPUAdaptive())
    {
        if(storage!=ImmutableRowStorage::VRAM)throw std::invalid_argument("GPU adaptive execution cannot become whole mmap/RAM");
        requireGPUAdaptiveDomain(profile,cfg);execution=ImmutableRowLookupExecution::AdaptiveGPUCache;
    }
    const auto costs=ngramModeMemory(profile,cfg,execution);
    const PhysicalMemoryResource resource{cfg.world_rank,cfg.device,cfg.device_total_bytes,cfg.device_free_bytes};
    PhysicalMemoryPlanBuilder builder;
    builder.add(resource,PhysicalMemoryOwner::ActivationArena,costs.arena)
        .add(resource,PhysicalMemoryOwner::NativeGraphExecutable,costs.terminal);
    if(costs.mapped)
    {
        if(!cfg.associated_host_memory || !cfg.associated_host_memory->valid() ||
            !cfg.associated_host_memory->device.is_cpu() || cfg.associated_host_memory->world_rank!=cfg.world_rank)
            throw std::invalid_argument("Mandatory hosted lookup execution requires its exact associated host allocator");
        builder.add(*cfg.associated_host_memory,PhysicalMemoryOwner::ImmutableLookupStaging,costs.mapped);
    }
    return builder.build();
}
/**
 * @brief Price mandatory execution before rows or expert quotas consume capacity.
 * @param profile Existing original source metadata.
 * @param cfg Exact consumer geometry carrying the lossless requested policy.
 * @return Canonical per-owner bound of the requested exclusive execution arms.
 * @throws std::invalid_argument For malformed policy or ineligible explicit mode.
 */
PhysicalMemoryPlan ngramExecutionEnvelope(const ModelMemoryProfile &profile,const DevicePlanConfig &cfg)
{
    if(!ownsNgramLookup(profile,cfg) || !cfg.ngram_execution_policy) return {};
    cfg.ngram_execution_policy->validate();
    std::vector<PhysicalMemoryPlan> alternatives;
    switch(cfg.ngram_execution_policy->storage)
    {
    case ImmutableRowStoragePreference::Automatic:
        if(cfg.device.is_gpu())alternatives.push_back(ngramExecutionArm(profile,cfg,ImmutableRowStorage::VRAM));
        alternatives.push_back(ngramExecutionArm(profile,cfg,ImmutableRowStorage::RAM));
        alternatives.push_back(ngramExecutionArm(profile,cfg,ImmutableRowStorage::MemoryMapped));
        break;
    case ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD:
        requireGPUAdaptiveDomain(profile,cfg);
        alternatives.push_back(ngramExecutionArm(profile,cfg,ImmutableRowStorage::VRAM));break;
    case ImmutableRowStoragePreference::VRAM:
        if(!cfg.device.is_gpu())throw std::invalid_argument("Explicit VRAM lookup requires a GPU consumer");
        alternatives.push_back(ngramExecutionArm(profile,cfg,ImmutableRowStorage::VRAM));break;
    case ImmutableRowStoragePreference::RAM:
        alternatives.push_back(ngramExecutionArm(profile,cfg,ImmutableRowStorage::RAM));break;
    case ImmutableRowStoragePreference::HostAdaptiveRAMSSD:
    case ImmutableRowStoragePreference::MemoryMapped:
        alternatives.push_back(ngramExecutionArm(profile,cfg,ImmutableRowStorage::MemoryMapped));break;
    }
    // These alternatives describe one immutable table mode and cannot execute
    // concurrently. Existing PMA envelope arithmetic is the sole byte authority.
    return PhysicalMemoryPlanBuilder{}.addMutuallyExclusive(alternatives).build();
}
} // anonymous namespace

/** @brief Setup-only object identities; this is never a byte or allocation ledger. */
struct MemoryPlanner::SharedOwnerIdentities
{
    std::set<std::pair<int, std::string>> prefix_archives;
    std::set<std::pair<int, std::string>> native_table_allocators; ///< One host table or one resident copy per GPU PLE consumer.
    std::map<std::pair<int,std::string>,AdaptiveImmutableLookupGeometry> gpu_adaptive; ///< Source-owned shared domain BOM, never live capacity state.
    std::map<std::pair<int,std::string>,CpuAdaptiveImmutableRowCacheGeometry> host_adaptive; ///< Immutable shared source/request geometry, never a byte ledger.
};

/**
 * @brief Price one group's distinct owners inside a complete planning transaction.
 * @param profile Immutable complete source metadata.
 * @param device_configs This group's actual retained graph owners.
 * @param shared Archive identities already owned by earlier groups.
 * @return Fixed contribution; shared archive scratch is emitted only once.
 *
 * The set records object identity, never byte capacity or live allocation state.
 * Only MemoryPlanner owns its lifetime and no caller can reset it between groups.
 */
MemoryPlan MemoryPlanner::planOwnerGroup(
    const ModelMemoryProfile& profile,
    const std::vector<DevicePlanConfig>& device_configs,
    SharedOwnerIdentities &shared)
{
    MemoryPlan result;
    result.devices.reserve(device_configs.size());
    auto &accounted_prefix_archives = shared.prefix_archives;

    /*
     * openShared() retains one archive object, lock domain, and scratch buffer
     * for a model archive on each MPI rank.  Several local GPU participants
     * may reference that same physical host allocation, so price its archive
     * key once rather than inventing one scratch per logical device plan.
     */
    for (const auto& cfg : device_configs)
    {
        // Validate the topology-authored owner even when capture is disabled;
        // CPU placement and unknown enum arms cannot acquire a GPU parent.
        (void)OrdinaryGenerationGraphPlan::pipelineForwardOnlyExecutableCount(
            1u, cfg.pipeline_forward_only_graph_owner);
        if (cfg.pipeline_forward_only_graph_owner == PipelineForwardOnlyGraphOwner::Participant && !cfg.device.is_gpu())
            throw std::invalid_argument("Pipeline ForwardOnly parent requires its declared GPU participant");
        const auto ngram_lookup = ngramLookup(profile, cfg);
        const bool gpu_adaptive=(cfg.ngram_table_placement && cfg.ngram_table_placement->isGPUAdaptive()) ||
            (cfg.ngram_execution_policy && cfg.ngram_execution_policy->isGPUAdaptive());
        if(gpu_adaptive && ownsNgramLookup(profile,cfg))requireGPUAdaptiveDomain(profile,cfg);
        const bool native_ngram_lookup = ngram_lookup &&
            (*ngram_lookup==ImmutableRowLookupExecution::AdaptiveGPUCache || immutableRowLookupUsesNativeDeviceStorage(*ngram_lookup));
        std::size_t native_table_bytes = 0;
        bool first_ngram_physical_owner=false;
        std::size_t adaptive_request_bytes=0,adaptive_request_credit=0;
        std::size_t mixed_source_table=0,mixed_source_staging=0;
        std::size_t mixed_source_table_credit=0,mixed_source_staging_credit=0;
        const bool adaptive=(cfg.ngram_table_placement && cfg.ngram_table_placement->isHostAdaptive()) ||
            (cfg.ngram_execution_policy && cfg.ngram_execution_policy->isHostAdaptive());
        if(adaptive && ownsNgramLookup(profile,cfg) && cfg.device.is_gpu() &&
            (!cfg.associated_host_memory || !cfg.associated_host_memory->valid() ||
             !cfg.associated_host_memory->device.is_cpu() || cfg.associated_host_memory->world_rank!=cfg.world_rank))
            throw std::invalid_argument("Adaptive GPU lookup requires its exact existing rank-local host allocator");
        if (ngram_lookup)
        {
            const auto &placement = *cfg.ngram_table_placement;
            const auto allocator = native_ngram_lookup
                ? PhysicalMemoryAllocatorIdentity{cfg.world_rank, cfg.device} : placement.allocators().front();
            if (shared.native_table_allocators.emplace(allocator.world_rank, allocator.device.toString()).second)
            {
                first_ngram_physical_owner=true;
                if(placement.isGPUAdaptive())
                {
                    if(placement.gpuAdaptiveRowElements()!=profile.flash_next->ple_row_elements)
                        throw std::invalid_argument("Resolved adaptive row elements differ from the actual native source");
                    const auto geometry=AdaptiveImmutableLookupGeometry::fromLayout(placement.gpuAdaptiveLayout(),
                        ngramNativeSource(profile),placement.gpuAdaptiveRequest());
                    PhysicalMemoryPlanBuilder owned(geometry.mandatoryBOM(
                        {cfg.world_rank,cfg.device,cfg.device_total_bytes,cfg.device_free_bytes},*cfg.associated_host_memory));
                    const auto optional=geometry.cacheBOM(
                        {cfg.world_rank,cfg.device,cfg.device_total_bytes,cfg.device_free_bytes},*cfg.associated_host_memory);
                    for(const auto &bom:optional.resources())owned.add(bom);
                    const auto complete=owned.build();
                    const auto *gpu=complete.find({cfg.world_rank,cfg.device});
                    const auto *host=complete.find({cfg.world_rank,cfg.associated_host_memory->device});
                    native_table_bytes=gpu->bytes(PhysicalMemoryOwner::ImmutableTableStorage);
                    adaptive_request_bytes=gpu->bytes(PhysicalMemoryOwner::ImmutableLookupStaging);
                    mixed_source_table=host->bytes(PhysicalMemoryOwner::ImmutableTableStorage);
                    mixed_source_staging=host->bytes(PhysicalMemoryOwner::ImmutableLookupStaging);
                    adaptive_request_credit=cfg.retained_immutable_lookup_staging_bytes;
                    mixed_source_table_credit=cfg.retained_immutable_source_table_bytes;
                    mixed_source_staging_credit=cfg.retained_immutable_source_lookup_staging_bytes;
                    if((adaptive_request_credit && adaptive_request_credit!=adaptive_request_bytes) ||
                        (mixed_source_table_credit && mixed_source_table_credit!=mixed_source_table) ||
                        (mixed_source_staging_credit && mixed_source_staging_credit!=mixed_source_staging))
                        throw std::invalid_argument("Retained mixed ngram owners must attest their exact original GPU/source/channel claims");
                }
                else if(placement.isHostAdaptive())
                {
                    const auto &host=cfg.device.is_cpu()?PhysicalMemoryResource{cfg.world_rank,cfg.device,cfg.device_total_bytes,cfg.device_free_bytes}
                        :cfg.associated_host_memory.value();
                    const auto geometry=CpuAdaptiveImmutableRowCacheGeometry::fromLayout(placement.hostAdaptiveLayout(),allocator,
                        placement.hostAdaptiveMaximumLiveRows(),placement.hostAdaptivePolicy());
                    const auto owned=geometry.bom(host);const auto *lines=owned.find(allocator);
                    native_table_bytes=lines->bytes(PhysicalMemoryOwner::ImmutableTableStorage);
                    adaptive_request_bytes=lines->bytes(PhysicalMemoryOwner::ImmutableLookupStaging);
                    adaptive_request_credit=cfg.retained_immutable_lookup_staging_bytes;
                    if(adaptive_request_credit && adaptive_request_credit!=adaptive_request_bytes)
                        throw std::invalid_argument("Retained adaptive request scratch must attest its exact live lease");
                }
                else if (placement.isFixedHeadShards())
                    native_table_bytes=placement.fixedHeadSourceRangeFor(allocator).nativeBytes();
                else native_table_bytes = native_ngram_lookup ? placement.extent().nativeBytes() :
                    ImmutableRowTable::physicalBytesForNativeExtent(placement.extent().nativeBytes(), placement.storagePolicy());
            }
        }
        else if(gpu_adaptive && ownsNgramLookup(profile,cfg))
        {
            const auto key=std::pair{cfg.world_rank,cfg.device.toString()};
            if(shared.native_table_allocators.insert(key).second)
            {
                const auto mandatory=shared.gpu_adaptive.at(key).mandatoryBOM(
                    {cfg.world_rank,cfg.device,cfg.device_total_bytes,cfg.device_free_bytes},*cfg.associated_host_memory);
                const auto *gpu=mandatory.find({cfg.world_rank,cfg.device});
                const auto *host=mandatory.find({cfg.world_rank,cfg.associated_host_memory->device});
                adaptive_request_bytes=gpu->bytes(PhysicalMemoryOwner::ImmutableLookupStaging);
                mixed_source_table=host->bytes(PhysicalMemoryOwner::ImmutableTableStorage);
                mixed_source_staging=host->bytes(PhysicalMemoryOwner::ImmutableLookupStaging);
            }
        }
        else if(ownsNgramLookup(profile,cfg) && cfg.ngram_execution_policy && cfg.ngram_execution_policy->isHostAdaptive())
        {
            const auto host=cfg.device.is_cpu()?PhysicalMemoryResource{cfg.world_rank,cfg.device,cfg.device_total_bytes,cfg.device_free_bytes}
                :cfg.associated_host_memory.value();
            const auto key=std::pair{host.world_rank,host.device.toString()};
            if(shared.native_table_allocators.insert(key).second)
            {
                const auto owned=shared.host_adaptive.at(key).mandatoryBOM(host);
                const auto *lines=owned.find({host.world_rank,host.device});
                native_table_bytes=lines->bytes(PhysicalMemoryOwner::ImmutableTableStorage);
                adaptive_request_bytes=lines->bytes(PhysicalMemoryOwner::ImmutableLookupStaging);
            }
        }
        if(cfg.retained_immutable_lookup_staging_bytes &&
            (!cfg.ngram_table_placement || (!cfg.ngram_table_placement->isHostAdaptive() && !cfg.ngram_table_placement->isGPUAdaptive())))
            throw std::invalid_argument("Retained adaptive scratch credit requires its exact resolved host owner");
        if((cfg.retained_immutable_source_table_bytes || cfg.retained_immutable_source_lookup_staging_bytes) &&
            (!cfg.ngram_table_placement || !cfg.ngram_table_placement->isGPUAdaptive()))
            throw std::invalid_argument("Retained subordinate source claims require the exact mixed placement arm");
        if(cfg.retained_immutable_table_bytes && !ngram_lookup)
            throw std::invalid_argument("Retained native table credit requires its exact sealed concrete placement");
        if(gpu_adaptive && first_ngram_physical_owner && cfg.retained_immutable_table_bytes &&
            cfg.retained_immutable_table_bytes!=native_table_bytes)
            throw std::invalid_argument("Retained C0/nonzero GPU cache credit must match its exact actual optional owner");
        const auto native_table_credit=native_table_bytes ? cfg.retained_immutable_table_bytes : 0u;
        if(native_table_credit && native_table_credit!=native_table_bytes)
            throw std::invalid_argument("Retained immutable table credit must attest the complete original allocation");
        if (cfg.concurrent_workspace_owners == 0u)
            throw std::invalid_argument(
                "Memory planning requires a positive workspace-owner count");
        if (cfg.weight_load_staging.enabled() && !cfg.device.is_gpu())
        {
            throw std::invalid_argument(
                "Weight-load staging can only be attached to a GPU device plan");
        }
        if ((cfg.weight_load_staging.device_bytes == 0u) !=
            (cfg.weight_load_staging.host_bytes == 0u))
        {
            throw std::invalid_argument(
                "GPU weight-load staging requires both device and pinned-host byte authorities");
        }
        std::size_t captured_graph_bytes = 0u;
        std::size_t immutable_lookup_executable_bytes = 0u;
        std::size_t graph_snapshot_bytes = 0u;
        std::size_t primary_weight_bytes = 0u;
        // Validate the descriptive authority even when no auxiliary set is used.
        const bool mtp_replica_owns_routed_parents =
            mtpParticipantReplicaOwnsRoutedParents(
                cfg.mtp_routed_expert_weight_authority);
        std::size_t additional_weight_bytes = 0u;
        std::size_t retained_primary_weight_bytes = 0u;
        std::size_t retained_additional_weight_bytes = 0u;
        std::size_t kv_cache_bytes = 0u;
        std::size_t live_recurrent_state_bytes = 0u;
        std::size_t checkpoint_state_bytes = 0u;
        std::size_t sequence_metadata_bytes = 0u;
        std::size_t prefix_cache_staging_bytes = 0u;
        std::size_t prefix_cache_host_staging_bytes = 0u;
        std::size_t prefix_cache_device_hot_bytes = 0u;
        std::size_t prefix_cache_host_tier_bytes = 0u;
        std::size_t collective_bytes = 0u;
        std::size_t activation_bytes = 0u;
        std::size_t workspace_bytes = 0u;
        std::size_t retained_workspace_bytes = 0u;

        int last_layer = cfg.last_layer >= 0 ? cfg.last_layer : profile.n_layers - 1;
        int max_seq = cfg.max_seq_len > 0 ? cfg.max_seq_len : profile.max_seq_len;
        int activation_seq = cfg.activation_seq_len > 0 ? cfg.activation_seq_len : max_seq;
        if (max_seq > 0)
            activation_seq = std::min(activation_seq, max_seq);
        activation_seq = std::max(1, activation_seq);
        std::size_t immutable_lookup_host_bytes = 0;
        std::size_t immutable_lookup_arena_bytes = 0;
        if (!ngram_lookup && cfg.ngram_execution_policy && ownsNgramLookup(profile,cfg))
        {
            const auto provision=ngramExecutionEnvelope(profile,cfg);
            if(const auto *device=provision.find({cfg.world_rank,cfg.device}))
            {
                immutable_lookup_arena_bytes=device->bytes(PhysicalMemoryOwner::ActivationArena);
                immutable_lookup_executable_bytes=device->bytes(PhysicalMemoryOwner::NativeGraphExecutable);
            }
            if(cfg.associated_host_memory)
                if(const auto *host=provision.find({cfg.world_rank,cfg.associated_host_memory->device}))
                    immutable_lookup_host_bytes=host->bytes(PhysicalMemoryOwner::ImmutableLookupStaging);
        }
        if (ngram_lookup)
        {
            const auto lookup = ngramModeMemory(profile, cfg, *ngram_lookup);
            immutable_lookup_host_bytes = lookup.mapped;
            immutable_lookup_executable_bytes = lookup.terminal;
        }
        size_t pipeline_device_bytes = 0u;
        size_t pipeline_host_bytes = 0u;
        if (cfg.projection_peer_access == PeerAccessCoverage::None)
        {
            if (!cfg.device.is_gpu() || cfg.routed_compute_policy != RoutedExpertComputePolicy::GateUpOwnedDownColumns ||
                cfg.execution_role != DeviceExecutionMemoryRole::ContinuationGraph || !cfg.local_tp_backend)
                throw std::invalid_argument("Counted projection staging requires a rank-local GPU continuation domain");
            const auto rows = std::max(activation_seq, cfg.mtp_enabled ? std::max(1, cfg.mtp_target_query_rows) : 1);
            const auto arena = MoEProjectionArenaGeometry::resolve(profile, cfg.total_shards);
            const auto staging = DeviceCountedAllGather::memoryFor(cfg.total_shards, arena.packetCapacityBytes(rows));
            // Each producer contributes its outgoing mapped slots once. Every
            // GPU owns both endpoint cursors and its exact extent bank. This is
            // a typed BOM, never an independent capacity subtraction/reserve.
            pipeline_device_bytes = staging.device_bytes;
            pipeline_host_bytes = staging.host_bytes;
        }
        if (!cfg.captured_pipeline_boundaries.empty())
        {
            if (!cfg.device.is_gpu() || cfg.shard_index != 0 || cfg.local_pipeline_backend ||
                cfg.execution_role != DeviceExecutionMemoryRole::ContinuationGraph ||
                cfg.captured_pipeline_boundaries.size() > 2 || profile.d_model <= 0)
                throw std::invalid_argument("Captured pipeline channels require a GPU domain leader and no native PP boundary");
            const auto transport_width = profile.flash_next ? profile.flash_next->residual_width : profile.d_model;
            const auto geometry = PipelineTransferMemory::forRows(transport_width, activation_seq,
                cfg.mtp_enabled ? std::max(1, cfg.mtp_target_query_rows) : 1);
            for (size_t index = 0; index < cfg.captured_pipeline_boundaries.size(); ++index)
            {
                const auto side = cfg.captured_pipeline_boundaries[index];
                if ((side != PipelineBoundarySide::EarlierDomain && side != PipelineBoundarySide::LaterDomain) ||
                    std::find(cfg.captured_pipeline_boundaries.begin(),
                        cfg.captured_pipeline_boundaries.begin() + index, side) !=
                        cfg.captured_pipeline_boundaries.begin() + index)
                    throw std::invalid_argument("Captured pipeline BOM repeats or invents an adjacent boundary");
                pipeline_device_bytes = checkedAdd(pipeline_device_bytes,
                    geometry.device_bytes_per_boundary, "pipeline channel device cursors");
                // Both directional slots are charged once by the earlier
                // layer domain. The later endpoint owns only its GPU cursors.
                if (side == PipelineBoundarySide::EarlierDomain)
                    pipeline_host_bytes = checkedAdd(pipeline_host_bytes,
                        geometry.host_bytes_per_boundary, "pipeline channel host mappings");
            }
        }
        const bool replicated_dense_decode =
            std::find(
                cfg.additional_weight_sets.begin(),
                cfg.additional_weight_sets.end(),
                AdditionalPersistentWeightSet::ReplicatedDenseDecode) !=
            cfg.additional_weight_sets.end();
        if (cfg.tensor_parallel_assignment.has_value())
        {
            const auto &assignment = *cfg.tensor_parallel_assignment;
            if (cfg.total_shards <= 1 ||
                assignment.local_rank != cfg.shard_index ||
                assignment.device != cfg.device ||
                !assignment.isValid() ||
                assignment.kv_head_count <= 0 ||
                assignment.d_ff_count <= 0 ||
                assignment.vocab_count <= 0)
            {
                throw std::invalid_argument(
                    "Device memory plan contains a stale or mismatched tensor-parallel assignment");
            }
        }
        int local_kv_heads = cfg.tensor_parallel_assignment.has_value()
            ? cfg.tensor_parallel_assignment->kv_head_count
            : (cfg.local_kv_heads > 0
                   ? cfg.local_kv_heads
                   : profile.n_kv_heads);
        if (replicated_dense_decode)
        {
            // The phase-split graph stores complete attention K/V on every
            // participant. The TP assignment still determines GDN state and
            // prefill projection ownership; only the main KV bank is full.
            local_kv_heads = profile.n_kv_heads;
        }
        const int local_query_heads =
            cfg.tensor_parallel_assignment.has_value()
                ? cfg.tensor_parallel_assignment->head_count
                : std::max(
                      1,
                      profile.n_heads /
                          std::max(1, cfg.total_shards));
        const int local_query_head_start =
            cfg.tensor_parallel_assignment.has_value()
                ? cfg.tensor_parallel_assignment->head_start
                : (cfg.total_shards > 1
                       ? cfg.shard_index * local_query_heads
                       : 0);

        if (cfg.device.is_gpu() && cfg.captured_serving_graphs.enabled())
        {
            if (!cfg.captured_serving_graphs.generation_domain)
                throw std::invalid_argument("Captured serving inventory has no declared generation domain");
            const auto generation_execution = declaredDeviceGenerationExecutionPolicy(
                cfg.device.type, *cfg.captured_serving_graphs.generation_domain);
            const auto expected_family = resolveMTPVerifierGraphFamily(mtpVerifierExecutionRole(generation_execution));
            if (cfg.captured_serving_graphs.verifier_graph_family != expected_family)
                throw std::invalid_argument("Captured serving verifier family contradicts this GPU's declared generation domain");
            const std::size_t prefill_executable_count =
                prefillGraphBucketsAtOrBelowCapacity(
                    cfg.captured_serving_graphs.prefill_bucket_rows,
                    activation_seq)
                    .size();
            const std::size_t executable_count = checkedAdd(
                prefill_executable_count,
                cfg.captured_serving_graphs.fixed_executable_count,
                "captured serving graph executable count");
            captured_graph_bytes =
                estimateCapturedGraphExecutableBytes(
                    cfg.device,
                    CapturedGraphExecutableInventory{
                        .model_graph_identity_count = executable_count,
                        .model_graph_topology_variant_count = 1u,
                        .auxiliary_executable_count =
                            (cfg.captured_serving_graphs.ordinary_generation
                                ? OrdinaryGenerationGraphPlan::additionalExecutableCount(
                                    cfg.device,
                                    cfg.captured_serving_graphs.mtp_graph_owners.retainsGraphCapacity(),
                                    generation_execution,
                                    cfg.captured_serving_graphs.generation_parent_role,
                                    cfg.captured_serving_graphs.sparse_query_program_count) +
                                    OrdinaryGenerationGraphPlan::pipelineForwardOnlyExecutableCount(
                                        cfg.captured_serving_graphs.sparse_query_program_count,
                                        cfg.pipeline_forward_only_graph_owner)
                                : 0u) +
                            cfg.captured_serving_graphs
                                .mtp_graph_owners
                                .generalAuxiliaryExecutableSlotCount(),
                        .bounded_helper_executable_count =
                            cfg.captured_serving_graphs
                                .mtp_graph_owners
                                .boundedHelperExecutableSlotCount(),
                    });
        }

        if (cfg.device.is_gpu())
        {
            const auto &snapshots = cfg.graph_snapshot_memory;
            if (snapshots.main_phase_payload)
            {
                if (!snapshots.serving_arenas)
                    throw std::invalid_argument("Phase snapshot payload requires the authored serving-arena inventory");
                graph_snapshot_bytes = snapshots.main_phase_payload->bytesFor(*snapshots.serving_arenas);
            }
            else graph_snapshot_bytes = snapshots.per_accelerator_bytes;
        }

        // If TP sharded, divide KV heads
        if (cfg.total_shards > 1 && cfg.local_kv_heads <= 0 &&
            !cfg.tensor_parallel_assignment.has_value())
        {
            local_kv_heads = std::max(1, profile.n_kv_heads / cfg.total_shards);
        }
        const int mtp_local_kv_heads =
            cfg.mtp_enabled
                ? resolveMTPShiftedKVLocalHeadCount(
                      cfg.mtp_shifted_kv_head_layout,
                      profile.n_kv_heads,
                      local_kv_heads)
                : local_kv_heads;

        /*
         * The execution plan's layer interval names main-model blocks. MTP
         * predictors are trailing model-file blocks owned by the participant
         * that owns the terminal main layer, so include them explicitly in
         * that participant's persistent weight inventory.
         */
        const int main_layer_count =
            profile.mainLayerCount();
        const bool owns_terminal_main_layer =
            main_layer_count > 0 &&
            cfg.first_layer <= main_layer_count - 1 &&
            last_layer >= main_layer_count - 1;
        const auto mtp_state_role = resolveMTPStateRole(
            cfg.mtp_enabled, owns_terminal_main_layer);
        const auto terminal_logits_collective =
            cfg.terminalLogitsCollective(owns_terminal_main_layer);
        if (owns_terminal_main_layer && cfg.execution_role == DeviceExecutionMemoryRole::ContinuationGraph)
            (void)cfg.ordinary_terminal_projection.resolveForSequence(cfg.batch_size, activation_seq);
        if (cfg.device.is_gpu() && cfg.graph_snapshot_memory.effective_kv &&
            cfg.execution_role == DeviceExecutionMemoryRole::ContinuationGraph)
        {
            const auto &diagnostic = *cfg.graph_snapshot_memory.effective_kv;
            if (!cfg.graph_snapshot_memory.valid() || diagnostic.retained_arena_count == 0 ||
                cfg.batch_size <= 0 || max_seq <= 0 || profile.head_dim <= 0 ||
                (diagnostic.layer && (*diagnostic.layer < 0 || *diagnostic.layer >= profile.n_layers)))
                throw std::invalid_argument("Effective-KV snapshot admission requires complete layer, request, context and retained-arena geometry");

            std::size_t per_arena_bytes = 0;
            for (int layer = 0; layer < profile.n_layers; ++layer)
            {
                const bool sidecar = layer >= main_layer_count;
                const bool resident = sidecar ? mtp_state_role == MTPStateRole::PredictorOwner :
                    (layer >= cfg.first_layer && layer <= last_layer);
                if (!resident || (diagnostic.layer && layer != *diagnostic.layer) ||
                    !PersistentStateMemoryEstimator::isFullAttentionLayer(profile, layer))
                    continue;
                const int heads = sidecar ? mtp_local_kv_heads : local_kv_heads;
                if (heads <= 0)
                    throw std::invalid_argument("Effective-KV snapshot admission requires positive local KV heads");

                // Each output slot holds raw native bytes. FP32 is the maximum
                // supported effective-read element size (native or converted),
                // so the explicit envelope covers every codec without guessing
                // the model's attention read policy. No conversion is performed.
                const auto elements = checkedMultiply(
                    checkedMultiply(static_cast<std::size_t>(cfg.batch_size), max_seq,
                                    "effective-KV request/context rows"),
                    checkedMultiply(static_cast<std::size_t>(heads), profile.head_dim,
                                    "effective-KV local head width"),
                    "effective-KV bank elements");
                const auto payload = checkedMultiply(elements, 2u * sizeof(float),
                                                     "effective-K/V bank bytes");
                const auto metadata = checkedMultiply(static_cast<std::size_t>(cfg.batch_size),
                    2u * sizeof(std::int32_t), "effective-KV device count/head snapshots");
                per_arena_bytes = checkedAdd(per_arena_bytes, checkedAdd(payload, metadata,
                    "effective-KV layer snapshot"), "effective-KV layer inventory");
            }
            graph_snapshot_bytes = checkedAdd(graph_snapshot_bytes,
                checkedMultiply(per_arena_bytes, diagnostic.retained_arena_count,
                                "retained effective-KV snapshot arenas"),
                "reference and effective-KV snapshots");
        }
        if (cfg.device.is_gpu() &&
            cfg.execution_role == DeviceExecutionMemoryRole::ContinuationGraph &&
            !cfg.graph_snapshot_memory.authored_source_outputs.empty())
        {
            if (!profile.flash_next)
                throw std::invalid_argument("Source-owned QSA snapshots require authenticated Flash Next metadata");
            auto contribution = qwenSparseSnapshotMemory(*profile.flash_next, profile.mainLayerCount(), cfg.first_layer, last_layer,
                max_seq, cfg.batch_size, activation_seq,
                cfg.graph_snapshot_memory.retained_main_forward_arena_count,
                cfg.graph_snapshot_memory.authored_source_outputs);
            if (cfg.mtp_enabled)
            {
                if (cfg.generation_request_capacity <= 0 || cfg.mtp_target_query_rows <= 0 ||
                    cfg.mtp_target_query_rows % cfg.generation_request_capacity != 0)
                    throw std::invalid_argument("QSA verifier snapshots require exact request-flattened retained row geometry");
                const auto verifier = qwenSparseSnapshotMemory(*profile.flash_next, profile.mainLayerCount(),
                    cfg.first_layer, last_layer, max_seq, cfg.generation_request_capacity,
                    cfg.mtp_target_query_rows / cfg.generation_request_capacity,
                    cfg.graph_snapshot_memory.retained_main_forward_arena_count,
                    cfg.graph_snapshot_memory.authored_source_outputs);
                contribution.native_state_payload_bytes = std::max(contribution.native_state_payload_bytes, verifier.native_state_payload_bytes);
                contribution.authored_metadata_bytes = std::max(contribution.authored_metadata_bytes, verifier.authored_metadata_bytes);
            }
            graph_snapshot_bytes = checkedAdd(graph_snapshot_bytes, contribution.totalBytes(),
                "selected native-state and authored snapshot storage");
        }
        const int weight_last_layer =
            cfg.mtp_enabled &&
                    owns_terminal_main_layer &&
                    profile.mtp_layer_count > 0
                ? profile.totalLayerCount() - 1
                : last_layer;

        // Pipeline intervals restrict transformer weights, not vocabulary
        // ownership. Resolve the exact global uses once and pass that same
        // scope to primary and mirrored contributions. A terminal predictor
        // embeds its draft rows locally even though it is not the main entry.
        const bool retains_embedding_weights =
            (cfg.owns_embedding && cfg.first_layer == 0) ||
            (cfg.mtp_enabled && owns_terminal_main_layer);
        const WeightComponentScope weight_components =
            retains_embedding_weights
                ? (owns_terminal_main_layer ? WeightComponentScope::EmbeddingAndTerminal
                                            : WeightComponentScope::Embedding)
                : (owns_terminal_main_layer ? WeightComponentScope::Terminal
                                            : WeightComponentScope::Intermediate);

        // Weight estimation
        auto weight_est = WeightMemoryEstimator::estimate(
            profile, cfg.device,
            cfg.shard_index, cfg.total_shards,
            cfg.first_layer, weight_last_layer,
            cfg.weight_residency,
            cfg.tensor_parallel_assignment,
            weight_components);
        primary_weight_bytes = weight_est.device_bytes;

        for (std::size_t set_index = 0;
             set_index < cfg.additional_weight_sets.size();
             ++set_index)
        {
            const auto additional_set =
                cfg.additional_weight_sets[set_index];
            if (std::find(
                    cfg.additional_weight_sets.begin(),
                    cfg.additional_weight_sets.begin() +
                        static_cast<std::ptrdiff_t>(set_index),
                    additional_set) !=
                cfg.additional_weight_sets.begin() +
                    static_cast<std::ptrdiff_t>(set_index))
            {
                throw std::invalid_argument(
                    "Device memory plan contains a duplicate additional persistent weight set");
            }

            switch (additional_set)
            {
            case AdditionalPersistentWeightSet::ReplicatedDenseDecode:
            {
                if (cfg.total_shards <= 1)
                {
                    throw std::invalid_argument(
                        "Replicated dense decode sidecar requires a tensor-parallel primary weight view");
                }

                DeviceWeightResidency sidecar_residency;
                if (profile.expert_count > 0)
                {
                    std::vector<int> selected_routed_experts(
                        static_cast<std::size_t>(profile.totalLayerCount()), 0);
                    if (cfg.mtp_enabled && profile.mtp_layer_count > 0 &&
                        mtp_replica_owns_routed_parents)
                    {
                        // Plain MTP replicas retain their predictor parents.
                        // ExpertOverlay already prices those exact prepared
                        // residents through RoutedExpertWeights, never twice.
                        for (int layer = main_layer_count;
                             layer < profile.totalLayerCount();
                             ++layer)
                        {
                            selected_routed_experts[
                                static_cast<std::size_t>(layer)] =
                                profile.expert_count;
                        }
                    }
                    sidecar_residency =
                        DeviceWeightResidency::
                            continuationWithSelectedRoutedExperts(
                                profile.expert_count,
                                std::move(selected_routed_experts));
                }

                const auto sidecar_estimate =
                    WeightMemoryEstimator::estimate(
                        profile,
                        cfg.device,
                        /*shard_index=*/0,
                        /*total_shards=*/1,
                        cfg.first_layer,
                        weight_last_layer,
                        sidecar_residency,
                        std::nullopt,
                        weight_components);
                additional_weight_bytes = checkedAdd(
                    additional_weight_bytes,
                    sidecar_estimate.device_bytes,
                    "additional replicated dense decode weights");
                break;
            }
            case AdditionalPersistentWeightSet::MirroredDecodeEmbedding:
            case AdditionalPersistentWeightSet::MirroredMTPTerminalHead:
            {
                DeviceWeightResidency mirrored_residency;
                if (profile.expert_count > 0)
                {
                    mirrored_residency =
                        DeviceWeightResidency::
                            continuationWithSelectedRoutedExperts(
                                profile.expert_count,
                                std::vector<int>(
                                    static_cast<std::size_t>(
                                        profile.totalLayerCount()),
                                    0));
                }
                const auto mirrored_estimate =
                    WeightMemoryEstimator::estimate(
                        profile,
                        cfg.device,
                        /*shard_index=*/0,
                        /*total_shards=*/1,
                        cfg.first_layer,
                        weight_last_layer,
                        mirrored_residency,
                        std::nullopt,
                        weight_components);
                const std::size_t component_bytes =
                    additional_set ==
                            AdditionalPersistentWeightSet::
                                MirroredDecodeEmbedding
                        ? mirrored_estimate.prepared_embedding_bytes
                        : mirrored_estimate.lm_head_bytes;
                additional_weight_bytes = checkedAdd(
                    additional_weight_bytes,
                    component_bytes,
                    additional_set ==
                            AdditionalPersistentWeightSet::
                                MirroredDecodeEmbedding
                        ? "additional mirrored decode embedding"
                        : "additional mirrored MTP terminal head");
                break;
            }
            case AdditionalPersistentWeightSet::ReplicatedMTPSidecarDense:
            {
                // Earlier pipeline stages participate in grouped verification
                // of their main layers; they do not execute the tail's learned
                // predictor or materialize its replicated weight set.
                if (!owns_terminal_main_layer)
                    break;
                if (profile.mtp_layer_count <= 0 ||
                    main_layer_count >= profile.totalLayerCount())
                {
                    throw std::invalid_argument(
                        "Replicated MTP sidecar admission requires trailing predictor-layer metadata");
                }

                DeviceWeightResidency sidecar_residency;
                if (profile.expert_count > 0 &&
                    !mtp_replica_owns_routed_parents)
                {
                    sidecar_residency =
                        DeviceWeightResidency::continuationWithSelectedRoutedExperts(
                            profile.expert_count,
                            std::vector<int>(
                                static_cast<std::size_t>(profile.totalLayerCount()), 0));
                }

                // Describe precisely the extra layer uses. No globals-only
                // synthetic interval or subtractive correction is needed.
                const auto sidecar =
                    WeightMemoryEstimator::estimate(
                        profile,
                        cfg.device,
                        /*shard_index=*/0,
                        /*total_shards=*/1,
                        main_layer_count,
                        profile.totalLayerCount() - 1,
                        sidecar_residency,
                        std::nullopt,
                        WeightComponentScope::LayersOnly);
                additional_weight_bytes = checkedAdd(
                    additional_weight_bytes,
                    sidecar.device_bytes,
                    "additional replicated MTP sidecar weights");
                break;
            }
            }
        }
        if (cfg.prepared_weight_admission ==
            PreparedWeightAdmission::ReuseCertifiedCompleteSet)
        {
            /*
             * Keep the complete logical weight footprint in the BOM. The live
             * free-memory reading was taken after this allocation survived a
             * previous runner, so only the incremental admission calculation
             * subtracts it.
             */
            retained_primary_weight_bytes = primary_weight_bytes;
            retained_additional_weight_bytes = additional_weight_bytes;
        }

        // The same compact expert graph can be hosted by a continuation or a
        // follower. Resolve its flattened capacity once; dense TP and request
        // batch geometry must not reinterpret the serial packet declaration.
        std::optional<int> compact_routed_expert_token_rows;
        if (cfg.weight_residency.selectsRoutedExperts() ||
            cfg.execution_role == DeviceExecutionMemoryRole::RoutedExpertParticipant)
        {
            if (cfg.serial_routed_expert_compact_rows < 0)
                throw std::invalid_argument("Compact expert token rows cannot be negative");
            const size_t token_rows = cfg.serial_routed_expert_compact_rows > 0
                ? static_cast<size_t>(cfg.serial_routed_expert_compact_rows)
                : checkedMultiply(static_cast<size_t>(std::max(1, cfg.batch_size)),
                      static_cast<size_t>(activation_seq), "compact expert token rows");
            if (token_rows > static_cast<size_t>(std::numeric_limits<int>::max()))
                throw std::overflow_error("Compact expert token rows exceed integer geometry");
            compact_routed_expert_token_rows = static_cast<int>(token_rows);
        }

        /*
         * Cache planning follows the actual hybrid layer inventory. Full
         * attention contributes sequence-length-scaled KV storage; GDN and
         * MTP rollback state are exposed separately instead of hidden in a
         * workspace reserve.
         */
        if (cfg.execution_role == DeviceExecutionMemoryRole::ContinuationGraph)
        {
            const auto persistent_state =
                PersistentStateMemoryEstimator::estimate(
                    profile,
                    cfg.device,
                    cfg.batch_size,
                    max_seq,
                    local_kv_heads,
                    mtp_local_kv_heads,
                    local_query_head_start,
                    local_query_heads,
                    cfg.first_layer,
                    last_layer,
                    cfg.kv_precision,
                    mtp_state_role);
            kv_cache_bytes =
                persistent_state.kv_cache_bytes;
            live_recurrent_state_bytes =
                persistent_state.live_recurrent_state_bytes;
            checkpoint_state_bytes =
                persistent_state.checkpoint_state_bytes;
            sequence_metadata_bytes =
                persistent_state.sequence_metadata_bytes;

            /*
             * Prefix restore owns a second, concurrent GPU memory surface:
             * one native K/V archive slot per cache family, the serialized
             * main-model GDN state, and (when eligible) a bounded device-hot
             * LRU. These allocations survive graph capture and must be priced
             * before routed experts consume the remaining capacity.
             * An all-GDN pipeline slice archives a complete recurrent image,
             * even though it has no attention block to stage. Do not confuse
             * zero K/V layers with absence of restorable main-model state.
             */
            const bool has_prefix_main_state =
                persistent_state.main_full_attention_layers > 0 ||
                persistent_state.main_gdn_layers > 0;
            const bool prefix_enabled =
                cfg.device.is_gpu() &&
                cfg.prefix_cache.enabled &&
                cfg.prefix_cache.storage_mode !=
                    PrefixCacheStorageMode::Disabled &&
                cfg.prefix_cache.storage_mode !=
                    PrefixCacheStorageMode::Device &&
                has_prefix_main_state;
            if (prefix_enabled)
            {
                const int prefix_block_tokens =
                    std::max(1, cfg.prefix_cache.block_size);
                const auto logical_block =
                    KVCacheMemoryEstimator::estimateGPULogicalBlock(
                        persistent_state.main_kv_family,
                        prefix_block_tokens,
                        local_kv_heads,
                        profile.head_dim,
                        cfg.kv_precision,
                        cfg.device);
                const size_t one_fa_layer_bytes = checkedAdd(
                    logical_block.k_bytes,
                    logical_block.v_bytes,
                    "prefix logical K/V layer");
                const auto shifted_logical_block =
                    KVCacheMemoryEstimator::estimateGPULogicalBlock(
                        KVCacheFamily::AttentionOnly,
                        prefix_block_tokens,
                        mtp_local_kv_heads,
                        profile.head_dim,
                        cfg.kv_precision,
                        cfg.device);
                const size_t one_shifted_fa_layer_bytes = checkedAdd(
                    shifted_logical_block.k_bytes,
                    shifted_logical_block.v_bytes,
                    "prefix shifted logical K/V layer");

                prefix_cache_staging_bytes = checkedAdd(
                    persistent_state.main_full_attention_layers > 0
                        ? one_fa_layer_bytes : 0u,
                    persistent_state.prefix_hybrid_device_state_bytes,
                    "prefix main archive staging");
                if (cfg.mtp_enabled &&
                    persistent_state.mtp_full_attention_layers > 0)
                {
                    prefix_cache_staging_bytes = checkedAdd(
                        prefix_cache_staging_bytes,
                        one_shifted_fa_layer_bytes,
                        "prefix shifted-MTP archive staging");
                }

                PrefixPayloadLayout prefix_capacity_layout;
                prefix_capacity_layout.device = cfg.device;
                prefix_capacity_layout.block_size = prefix_block_tokens;
                prefix_capacity_layout.fa_layers = persistent_state.main_full_attention_layers;
                prefix_capacity_layout.gdn_layers = persistent_state.main_gdn_layers;
                prefix_capacity_layout.bytes_per_fa_layer_k = logical_block.k_bytes;
                prefix_capacity_layout.bytes_per_fa_layer_v = logical_block.v_bytes;
                prefix_capacity_layout.hybrid_device_state_bytes = persistent_state.prefix_hybrid_device_state_bytes;
                prefix_capacity_layout.hybrid_state_bytes = persistent_state.prefix_hybrid_device_state_bytes;
                prefix_capacity_layout.includes_hybrid_state = persistent_state.prefix_hybrid_device_state_bytes > 0;
                if (cfg.mtp_enabled &&
                    persistent_state.mtp_full_attention_layers > 0)
                {
                    prefix_capacity_layout.mtp_origin=profile.flash_next ? MTPShiftedCacheOrigin::VisibleZeroHiddenTokenZero : MTPShiftedCacheOrigin::SuccessorPairsOnly;
                    prefix_capacity_layout.mtp_layers = persistent_state.mtp_full_attention_layers;
                    prefix_capacity_layout.mtp_kv_bytes = checkedMultiply(
                            static_cast<size_t>(
                                persistent_state.mtp_full_attention_layers),
                                one_shifted_fa_layer_bytes,
                                "prefix shifted-MTP payload");
                    prefix_capacity_layout.includes_mtp_state = true;
                }

                /*
                 * A pipeline participant archives terminal state only when
                 * it owns the terminal main-model layer.  Runtime makes the
                 * same decision from PPStageConfig::has_lm_head.  Charging
                 * logits on an earlier PP stage changes the fixed-slot size,
                 * which can paradoxically admit fewer total arena bytes than
                 * the smaller runtime slots would allocate.  Derive both
                 * weight and prefix ownership from this one layer-boundary
                 * fact so the PhysicalMemoryAuthority certificate and the
                 * materialized device-hot slab remain byte identical.
                 */
                if (owns_terminal_main_layer &&
                    cfg.prefix_cache.terminal_state !=
                        PrefixCacheTerminalStateMode::Off)
                {
                    if (cfg.mtp_enabled)
                    {
                        prefix_capacity_layout.terminal_hidden_bytes = checkedMultiply(
                                profile.flash_next ? profile.flash_next->residual_width : static_cast<size_t>(
                                    std::max(0, profile.d_model)),
                                sizeof(float),
                                "prefix terminal hidden state");
                        prefix_capacity_layout.includes_terminal_hidden = true;
                    }

                    size_t terminal_vocab = static_cast<size_t>(
                        std::max(0, profile.vocab_size));
                    // Prefix state consumes the graph's published row. A
                    // native gather assembles the full row even though each
                    // participant projects only its physical vocabulary shard.
                    if (!replicated_dense_decode &&
                        terminal_logits_collective !=
                            MTPTerminalLogitsCollective::NativeLocalVocabularyAllGather &&
                        cfg.mtp_terminal_logits_layout ==
                        MTPTerminalLogitsLayout::VocabularyShardPerParticipant)
                    {
                        terminal_vocab =
                            cfg.tensor_parallel_assignment.has_value()
                                ? static_cast<size_t>(
                                      cfg.tensor_parallel_assignment
                                          ->vocab_count)
                                : (terminal_vocab +
                                   static_cast<size_t>(
                                       std::max(1, cfg.total_shards)) -
                                   1u) /
                                      static_cast<size_t>(
                                          std::max(1, cfg.total_shards));
                    }
                    prefix_capacity_layout.terminal_logits_bytes = checkedMultiply(
                            terminal_vocab,
                            sizeof(float),
                            "prefix terminal logits");
                    prefix_capacity_layout.includes_terminal_logits = true;
                }

                if (profile.flash_next)
                {
                    const bool owns_ple = cfg.first_layer <= static_cast<int>(profile.flash_next->ple_decoder_layer) &&
                        static_cast<int>(profile.flash_next->ple_decoder_layer) <= last_layer;
                    prefix_capacity_layout = qwen4ExpPrefixCapacityLayout(prefix_capacity_layout,
                        {.sparse_layers = static_cast<size_t>(persistent_state.main_sparse_attention_layers),
                         .ple_channels = owns_ple ? profile.flash_next->residual_width : 0u},
                        {.sparse_layers = static_cast<size_t>(persistent_state.mtp_sparse_attention_layers)}, max_seq);
                }
                const size_t prefix_block_bytes = prefix_capacity_layout.totalBytes();

                if (cfg.prefix_cache.storage_mode ==
                        PrefixCacheStorageMode::Tiered &&
                    cfg.prefix_cache.ram_budget_bytes >=
                        prefix_block_bytes &&
                    cfg.prefix_cache.device_budget_bytes >=
                        prefix_block_bytes)
                {
                    prefix_cache_device_hot_bytes =
                        prefixCacheWholeBlockReservationBytes(
                            cfg.prefix_cache.device_budget_bytes,
                        prefix_block_bytes);
                }
                prefix_cache_host_tier_bytes =
                    cfg.prefix_cache.ram_budget_bytes;

                if (cfg.prefix_cache.storage_mode ==
                        PrefixCacheStorageMode::Tiered &&
                    cfg.prefix_cache.disk_budget_bytes > 0u)
                {
                    const auto archive_key = std::make_pair(
                        cfg.world_rank,
                        std::filesystem::path(cfg.prefix_cache.disk_dir)
                            .lexically_normal()
                            .string());
                    if (accounted_prefix_archives.insert(archive_key).second)
                    {
                        prefix_cache_host_staging_bytes =
                            PrefixArchiveIOGeometry::scratchBytes();
                    }
                }
            }
            else if (
                cfg.device.is_cpu() && cfg.prefix_cache.enabled &&
                cfg.prefix_cache.storage_mode !=
                    PrefixCacheStorageMode::Disabled &&
                has_prefix_main_state)
            {
                /*
                 * CPU inference archives directly into the same host
                 * allocator as its live cache. It has no device staging/hot
                 * tier, but the bounded RAM cache is still a concurrent
                 * physical owner and must not disappear from admission.
                 */
                prefix_cache_host_tier_bytes =
                    cfg.prefix_cache.ram_budget_bytes;
                if (cfg.prefix_cache.storage_mode ==
                        PrefixCacheStorageMode::Tiered &&
                    cfg.prefix_cache.disk_budget_bytes > 0u)
                {
                    const auto archive_key = std::make_pair(
                        cfg.world_rank,
                        std::filesystem::path(cfg.prefix_cache.disk_dir)
                            .lexically_normal()
                            .string());
                    if (accounted_prefix_archives.insert(archive_key).second)
                    {
                        prefix_cache_host_staging_bytes =
                            PrefixArchiveIOGeometry::scratchBytes();
                    }
                }
            }
            // Weight sharding spans the full domain, whereas these allocations
            // belong only to an installed rank-local collective. A one-device
            // NodeTP rank must not invent LocalTP just because a peer rank owns
            // another weight shard. Absence and unresolved AUTO are distinct.
            if (cfg.local_tp_backend.has_value())
            {
                if (cfg.local_tp_backend == CollectiveBackendType::AUTO)
                {
                    throw std::invalid_argument(
                        "LocalTP memory planning requires a resolved collective backend");
                }
                collective_bytes =
                    CollectiveMemoryEstimator::localTP(
                        max_seq,
                        profile.d_model,
                        *cfg.local_tp_backend)
                        .perDeviceBytes();
            }
            if (cfg.local_pipeline_backend.has_value())
                collective_bytes += CollectiveMemoryEstimator::nativePipelineBoundaryBytes(*cfg.local_pipeline_backend);
        }

        // Activation estimation
        int local_n_heads = cfg.tensor_parallel_assignment.has_value()
            ? cfg.tensor_parallel_assignment->head_count
            : profile.n_heads;
        if (cfg.total_shards > 1 &&
            !cfg.tensor_parallel_assignment.has_value())
        {
            local_n_heads = std::max(1, profile.n_heads / cfg.total_shards);
        }
        int local_d_ff = cfg.tensor_parallel_assignment.has_value()
            ? cfg.tensor_parallel_assignment->d_ff_count
            : profile.d_ff;
        if (cfg.total_shards > 1 &&
            !cfg.tensor_parallel_assignment.has_value())
        {
            local_d_ff = std::max(1, profile.d_ff / cfg.total_shards);
        }
        if (replicated_dense_decode)
        {
            // QwenGraphBase reserves one maximum-capacity arena for both
            // phase graphs, so its resident activation widths are full even
            // while the prefill weights remain TP-sharded.
            local_n_heads = profile.n_heads;
            local_d_ff = profile.d_ff;
        }

        if (cfg.execution_role == DeviceExecutionMemoryRole::ContinuationGraph)
        {
            activation_bytes = ActivationMemoryEstimator::estimate(
                profile,
                continuationActivationGeometry(profile,cfg,ngram_lookup),
                cfg.device);

            /*
             * A continuation endpoint may also own routed experts. Its compact
             * packet tensors are outside both the dense BufferArena and the
             * grouped-MoE workspace, so they must be added independently.
             * `routedParticipantActivationBytes()` applies the typed lifetime
             * declaration: a conventional graph retains per-layer packets,
             * while the graph-native serial family retains one packet per
             * independently runnable participant.
             */
            if (cfg.weight_residency.selectsRoutedExperts())
            {
                activation_bytes = checkedAdd(
                    activation_bytes,
                    routedParticipantActivationBytes(
                        profile,
                        cfg,
                        cfg.first_layer,
                        last_layer,
                        activation_seq),
                    "continuation local-expert compact tensors");
            }

            // Only the actual phase-split native LocalTP prefill owns linked
            // state handoff stages. Preserve its original primary assignment;
            // the full maximum-width workspace descriptor is a different role.
            std::optional<gdn_live_state_workspace::Geometry> gdn_handoff;
            if (cfg.device.is_gpu() && replicated_dense_decode && cfg.batch_size > 0 &&
                checkedMultiply(static_cast<size_t>(cfg.batch_size),
                    static_cast<size_t>(activation_seq), "GDN handoff flattened rows") > 1u &&
                cfg.total_shards > 1 && cfg.local_tp_backend &&
                (*cfg.local_tp_backend == CollectiveBackendType::NCCL ||
                 *cfg.local_tp_backend == CollectiveBackendType::RCCL) &&
                profile.gdn_state_size > 0 && profile.gdn_conv_kernel_size > 1)
            {
                const auto primary = HybridGDNStateGeometry::resolve(profile.n_heads,
                    local_query_head_start, local_query_heads, profile.gdn_group_count,
                    profile.gdn_time_step_rank, profile.gdn_state_size,
                    profile.gdn_inner_size, profile.gdn_conv_kernel_size);
                const gdn_live_state_workspace::Geometry candidate{
                    .degree = cfg.total_shards,
                    .state = {.global_key_heads = primary.full_key_heads,
                        .global_value_heads = primary.full_value_heads,
                        .key_width = primary.d_k, .value_width = primary.d_v,
                        .conv_history_length = profile.gdn_conv_kernel_size - 1}};
                const auto conv = candidate.state.resolve(GDNLinkedLiveStateKind::ConvHistory, candidate.degree);
                const auto recurrence = candidate.state.resolve(GDNLinkedLiveStateKind::Recurrence, candidate.degree);
                if (conv && recurrence &&
                    primary.local_conv_state_floats == conv->local_state_floats &&
                    primary.local_recurrence_state_floats == recurrence->local_state_floats &&
                    primary.local_recurrence_state_floats != primary.full_recurrence_state_floats)
                    gdn_handoff = candidate;
            }
            // One shared source-owned candidate policy is used by this BOM
            // and Qwen35's grouped graph lowering. Ordinary/prefill stays original.
            std::optional<gdn_workspace::GPURetainedRecurrenceWorkspace> gdn_retained;
            const bool owns_gdn = std::any_of(profile.tensors.begin(), profile.tensors.end(),
                [&](const auto &tensor) { return tensor.layer_index >= cfg.first_layer &&
                    tensor.layer_index <= last_layer && tensor.name.ends_with(".ssm_out.weight"); });
            if (cfg.device.is_gpu() && cfg.mtp_enabled && owns_gdn)
            {
                const int head_start = replicated_dense_decode ? 0 :
                    cfg.tensor_parallel_assignment ? cfg.tensor_parallel_assignment->head_start : cfg.shard_index * local_n_heads;
                const auto state = HybridGDNStateGeometry::resolve(profile.n_heads,
                    head_start, local_n_heads, profile.gdn_group_count, profile.gdn_time_step_rank,
                    profile.gdn_state_size, profile.gdn_inner_size, profile.gdn_conv_kernel_size);
                gdn_retained = gdn_workspace::selectGPUVerifierWorkspace(cfg.mtp_target_query_rows,
                    cfg.generation_request_capacity, state.local_value_heads, state.d_k, state.d_v);
            }
            // Workspace estimation
            workspace_bytes = WorkspaceMemoryEstimator::estimate(
                profile,
                WorkspaceMemoryGeometry{
                    .device = cfg.device,
                    .device_compute_units = cfg.device_compute_units,
                    .cpu_execution = cfg.cpu_execution,
                    .batch_size = cfg.batch_size,
                    .ordinary_terminal_projection = cfg.ordinary_terminal_projection,
                    .resident_graph_rows = activation_seq,
                    .max_context_rows = max_seq,
                    .owns_embedding = cfg.owns_embedding,
                    .shard_index = cfg.shard_index,
                    .has_exact_tensor_parallel_assignment =
                        cfg.tensor_parallel_assignment.has_value(),
                    .local_d_ff = local_d_ff,
                    .local_d_ff_start =
                        !replicated_dense_decode &&
                                cfg.tensor_parallel_assignment.has_value()
                            ? cfg.tensor_parallel_assignment->d_ff_start
                            : 0,
                    .local_query_head_start =
                        replicated_dense_decode
                            ? 0
                            : (cfg.tensor_parallel_assignment.has_value()
                            ? cfg.tensor_parallel_assignment->head_start
                            : cfg.shard_index * local_n_heads),
                    .local_query_heads = local_n_heads,
                    .local_kv_head_start =
                        replicated_dense_decode
                            ? 0
                            : (cfg.tensor_parallel_assignment.has_value()
                            ? cfg.tensor_parallel_assignment->kv_head_start
                            : cfg.shard_index * local_kv_heads),
                    .local_kv_heads = local_kv_heads,
                    .local_vocab_start =
                        !replicated_dense_decode &&
                                cfg.tensor_parallel_assignment.has_value()
                            ? cfg.tensor_parallel_assignment->vocab_start
                            : 0,
                    .local_vocab = replicated_dense_decode
                        ? profile.vocab_size
                        : (cfg.tensor_parallel_assignment.has_value()
                            ? cfg.tensor_parallel_assignment->vocab_count
                            : 0),
                    .first_layer = cfg.first_layer,
                    .last_layer = last_layer,
                    .total_shards = cfg.total_shards,
                    .gdn_live_state_handoff = gdn_handoff,
                    .apportioned_routed_experts =
                        cfg.weight_residency.selectsRoutedExperts(),
                    .runtime_device_policy_available =
                        cfg.runtime_device_policy_available,
                    .mtp_target_query_rows =
                        cfg.mtp_enabled
                            ? cfg.mtp_target_query_rows
                            : 0,
                    .generation_request_capacity = cfg.generation_request_capacity,
                    .gpu_gdn_retention = gdn_retained,
                    .mtp_dense_layout = cfg.mtp_enabled && profile.flash_next_predictor
                        ? std::optional{cfg.mtp_shifted_kv_head_layout} : std::nullopt,
                    .mtp_terminal_logits_layout =
                        replicated_dense_decode
                            ? MTPTerminalLogitsLayout::FullVocabularyPerParticipant
                            : cfg.mtp_terminal_logits_layout,
                    .mtp_terminal_logits_collective =
                        terminal_logits_collective,
                    .compact_routed_expert_token_rows = compact_routed_expert_token_rows,
                    .attention_kv_format = KVCacheMemoryEstimator::attentionExecutionFormat(cfg.kv_precision),
                    .attention_visibility = cfg.device.is_gpu() &&
                            KVCacheMemoryEstimator::attentionExecutionFormat(cfg.kv_precision) == TensorType::FP16
                        ? modelAttentionVisibility(profile.architecture)
                        : attention::AttentionVisibilityPolicy::General,
                    .qfn_hc_up_activation = cfg.qfn_hc_up_activation,
                    .qfn_gdn_output_activation = cfg.qfn_gdn_output_activation,
                    .qfn_fa_output_activation = cfg.qfn_fa_output_activation,
                    .qfn_gdn_qkv_activation = cfg.qfn_gdn_qkv_activation,
                });
        }
        else
        {
            activation_bytes = routedParticipantActivationBytes(
                profile,
                cfg,
                cfg.first_layer,
                last_layer,
                activation_seq);
            workspace_bytes =
                WorkspaceMemoryEstimator::estimateRoutedExpertParticipant(
                    profile,
                    WorkspaceMemoryGeometry{
                    .device = cfg.device,
                    .device_compute_units = cfg.device_compute_units,
                    .cpu_execution = cfg.cpu_execution,
                    .batch_size = cfg.batch_size,
                        .resident_graph_rows = activation_seq,
                        .max_context_rows = max_seq,
                        .owns_embedding = false,
                        .shard_index = cfg.shard_index,
                        .has_exact_tensor_parallel_assignment =
                            cfg.tensor_parallel_assignment.has_value(),
                        .local_d_ff = local_d_ff,
                        .local_d_ff_start =
                            cfg.tensor_parallel_assignment.has_value()
                                ? cfg.tensor_parallel_assignment->d_ff_start
                                : 0,
                        .local_query_head_start =
                            cfg.tensor_parallel_assignment.has_value()
                                ? cfg.tensor_parallel_assignment->head_start
                                : cfg.shard_index * local_n_heads,
                        .local_query_heads = local_n_heads,
                        .local_kv_head_start =
                            cfg.tensor_parallel_assignment.has_value()
                                ? cfg.tensor_parallel_assignment->kv_head_start
                                : cfg.shard_index * local_kv_heads,
                        .local_kv_heads = local_kv_heads,
                        .local_vocab_start =
                            cfg.tensor_parallel_assignment.has_value()
                                ? cfg.tensor_parallel_assignment->vocab_start
                                : 0,
                        .local_vocab =
                            cfg.tensor_parallel_assignment.has_value()
                                ? cfg.tensor_parallel_assignment->vocab_count
                                : 0,
                        .first_layer = cfg.first_layer,
                        .last_layer = last_layer,
                        .total_shards = cfg.total_shards,
                        .apportioned_routed_experts = true,
                        .generation_request_capacity = cfg.generation_request_capacity,
                        .compact_routed_expert_token_rows = compact_routed_expert_token_rows,
                    });
        }

        bool ordinary_cpu_grouped_roots_charged = false;
        if(cfg.device.is_cpu() && cfg.cpu_serial_expert_storage)
        {
            const auto &declaration=*cfg.cpu_serial_expert_storage;
            if(cfg.concurrent_workspace_owners!=declaration.controller_devices.size())
                throw std::invalid_argument("CPU serial workspace concurrency differs from its admitted controller identities");
            const auto grouped=declaration.envelope->admitted()->workspaceBytes();
            workspace_bytes=checkedAdd(workspace_bytes,
                checkedMultiply(grouped,declaration.participant_ids.size(),"CPU serial grouped endpoint roots"),
                "CPU serial grouped endpoint roots");
        }

        else if (cfg.device.is_cpu() && cfg.routed_expert_compact_buffer_lifetime ==
                     RoutedExpertCompactBufferLifetime::PerLayerGraphOwned)
        {
            // Serial overlay topology is wholly authored by its endpoint
            // identities above. A dense-only continuation with no endpoint
            // declaration must not acquire an ordinary bundle merely because
            // the global source model includes routed weights.
            // An actual routed parent in this PP interval authors the local
            // stage. Source expert widths are whole-expert CPU widths even
            // when dense projections use TP; selected weight counts never
            // turn model top-k into the old flattened (rows * K, 1) geometry.
            const bool owns_grouped_stage = std::any_of(
                profile.tensors.begin(), profile.tensors.end(), [&](const auto &tensor)
                {
                    if (!tensor.name.ends_with(".ffn_gate_exps.weight"))
                        return false;
                    const bool ordinary_main = tensor.layer_index >= cfg.first_layer &&
                        tensor.layer_index <= last_layer && tensor.layer_index < profile.mainLayerCount() &&
                        (!cfg.weight_residency.selectsRoutedExperts() ||
                         cfg.weight_residency.selectedRoutedExpertsForLayer(tensor.layer_index) > 0);
                    const bool retained_predictor = cfg.execution_role == DeviceExecutionMemoryRole::ContinuationGraph &&
                        cfg.mtp_enabled && owns_terminal_main_layer && profile.mtp_layer_count > 0 &&
                        tensor.layer_index >= profile.mainLayerCount() && tensor.layer_index < profile.totalLayerCount();
                    return ordinary_main || retained_predictor;
                });
            if (owns_grouped_stage)
            {
                size_t grouped_rows = checkedMultiply(size_t(std::max(1, cfg.batch_size)),
                    size_t(activation_seq), "CPU grouped ordinary flattened rows");
                if (cfg.mtp_enabled)
                    grouped_rows = std::max(grouped_rows, size_t(std::max(1, cfg.mtp_target_query_rows)));
                if (compact_routed_expert_token_rows)
                    grouped_rows = std::max(grouped_rows, size_t(*compact_routed_expert_token_rows));
                const auto grouped = CPUGroupedMoEStorageLayout::declare(grouped_rows,
                    profile.d_model, profile.expert_feed_forward_length,
                    profile.expert_count, profile.expert_used_count, cfg.device);
                workspace_bytes = checkedAdd(workspace_bytes, grouped.physicalBytes(),
                    "CPU ordinary grouped expert roots");
                ordinary_cpu_grouped_roots_charged = true;
            }
        }

        if (cfg.device.is_cpu() && cfg.execution_role == DeviceExecutionMemoryRole::ContinuationGraph &&
            !ordinary_cpu_grouped_roots_charged)
        {
            // A continuation may route every row to transported experts and
            // own no local expert stage. Its graph-local router still publishes
            // Q8 hidden rows. Serial endpoint kernels have different producers
            // and their external bundle cannot supply this manager's backing.
            const bool owns_router = std::any_of(profile.tensors.begin(), profile.tensors.end(),
                [&](const auto &tensor)
                {
                    if (inferWeightRole(tensor.name) != WeightRole::MoERouter)
                        return false;
                    // Qwen's FFN builder requires a router and either a
                    // routed parent or the explicit overlay residency role.
                    if (!cfg.weight_residency.selectsRoutedExperts() &&
                        !std::any_of(profile.tensors.begin(), profile.tensors.end(), [&](const auto &parent)
                        { return parent.layer_index == tensor.layer_index &&
                                 parent.name.ends_with(".ffn_gate_exps.weight"); }))
                        return false;
                    const bool ordinary_main = tensor.layer_index >= cfg.first_layer &&
                        tensor.layer_index <= last_layer && tensor.layer_index < profile.mainLayerCount();
                    const bool retained_predictor = cfg.mtp_enabled && owns_terminal_main_layer &&
                        profile.mtp_layer_count > 0 && tensor.layer_index >= profile.mainLayerCount() &&
                        tensor.layer_index < profile.totalLayerCount();
                    return ordinary_main || retained_predictor;
                });
            if (owns_router)
            {
                size_t rows = checkedMultiply(size_t(std::max(1, cfg.batch_size)), size_t(activation_seq),
                    "CPU router ordinary flattened rows");
                if (cfg.mtp_enabled)
                    rows = std::max(rows, size_t(std::max(1, cfg.mtp_target_query_rows)));
                workspace_bytes = checkedAdd(workspace_bytes,
                    CPURouterQ8StorageLayout::declare(rows, profile.d_model, cfg.device).physicalBytes(),
                    "CPU continuation router-only Q8 publication root");
            }
        }

        // Distinct graph owners can reuse one serial family across shapes,
        // but cannot alias one physical block while they run concurrently.
        workspace_bytes = checkedMultiply(
            workspace_bytes, cfg.concurrent_workspace_owners,
            "concurrent execution workspace owners");
        retained_workspace_bytes = std::min(
            workspace_bytes, cfg.retained_workspace_bytes);

        /*
         * The builder is the only place estimates become admission bytes.
         * Every named DeviceMemoryPlan accessor below is a view over these
         * typed lines, so no later planner can edit a parallel subtotal.
         */
        PhysicalMemoryBOMBuilder bom_builder({
            .world_rank = cfg.world_rank,
            .device = cfg.device,
            .total_bytes = cfg.device_total_bytes,
            .admission_available_bytes = cfg.device_free_bytes,
        });
        bom_builder
            .add(PhysicalMemoryOwner::ImmutableLookupStaging,(cfg.device.is_cpu() || gpu_adaptive)?adaptive_request_bytes:0u,
                 (cfg.device.is_cpu() || gpu_adaptive)?adaptive_request_credit:0u)
            .add(PhysicalMemoryOwner::ImmutableTableStorage,
                 (cfg.device.is_cpu() || native_ngram_lookup) ? native_table_bytes : 0u,
                 (cfg.device.is_cpu() || native_ngram_lookup) ? native_table_credit : 0u)
            .add(
                PhysicalMemoryOwner::PrimaryModelWeights,
                primary_weight_bytes,
                retained_primary_weight_bytes)
            .add(
                PhysicalMemoryOwner::AdditionalModelWeights,
                additional_weight_bytes,
                retained_additional_weight_bytes)
            .add(PhysicalMemoryOwner::KVCache, kv_cache_bytes)
            .add(
                PhysicalMemoryOwner::RecurrentLiveState,
                live_recurrent_state_bytes)
            .add(
                PhysicalMemoryOwner::RecurrentCheckpointState,
                checkpoint_state_bytes)
            .add(
                PhysicalMemoryOwner::SequenceMetadata,
                sequence_metadata_bytes)
            .add(
                PhysicalMemoryOwner::PrefixArchiveStaging,
                cfg.device.is_cpu()
                    ? prefix_cache_host_staging_bytes
                    : prefix_cache_staging_bytes)
            .add(
                PhysicalMemoryOwner::PrefixDeviceTier,
                prefix_cache_device_hot_bytes)
            .add(
                PhysicalMemoryOwner::PrefixHostTier,
                cfg.device.is_cpu()
                    ? prefix_cache_host_tier_bytes
                    : 0u)
            .add(
                PhysicalMemoryOwner::NativeGraphExecutable,
                checkedAdd(captured_graph_bytes, immutable_lookup_executable_bytes,
                    "native lookup terminal close executable"))
            .add(
                PhysicalMemoryOwner::GraphSnapshotArena,
                graph_snapshot_bytes)
            .add(PhysicalMemoryOwner::LocalCollective, collective_bytes)
            .add(PhysicalMemoryOwner::ActivationTransportStaging, pipeline_device_bytes)
            .add(PhysicalMemoryOwner::ActivationArena, checkedAdd(activation_bytes,
                immutable_lookup_arena_bytes,"mandatory lookup execution arena"))
            .add(
                PhysicalMemoryOwner::ExecutionWorkspace,
                workspace_bytes,
                retained_workspace_bytes)
            .add(
                PhysicalMemoryOwner::WeightLoadStaging,
                cfg.device.is_gpu()
                    ? cfg.weight_load_staging.device_bytes
                    : 0u);
        DeviceMemoryPlan dev_plan(
            bom_builder.build(), max_seq, activation_seq);

        appendPhysicalPlan(result, std::move(dev_plan));

        if (cfg.device.is_gpu() &&
            (prefix_cache_host_tier_bytes != 0u ||
             prefix_cache_host_staging_bytes != 0u ||
             cfg.weight_load_staging.host_bytes != 0u || pipeline_host_bytes != 0u ||
             (native_table_bytes != 0u && !native_ngram_lookup) || immutable_lookup_host_bytes != 0u ||
             mixed_source_table || mixed_source_staging))
        {
            if (!cfg.associated_host_memory.has_value() ||
                !cfg.associated_host_memory->valid() ||
                !cfg.associated_host_memory->device.is_cpu() ||
                cfg.associated_host_memory->world_rank != cfg.world_rank)
            {
                throw std::invalid_argument(
                    "GPU memory admission requires the exact rank-local host-memory resource for its CPU-side allocations");
            }
            if (native_table_bytes && !native_ngram_lookup && cfg.ngram_table_placement &&
                !cfg.ngram_table_placement->containsAllocator({cfg.world_rank, cfg.associated_host_memory->device}))
                throw std::invalid_argument("Ngram host BOM does not match the sealed CPU/NUMA allocator");
            PhysicalMemoryBOMBuilder host_builder(
                *cfg.associated_host_memory);
            host_builder.add(PhysicalMemoryOwner::ImmutableTableStorage,
                gpu_adaptive?mixed_source_table:(native_ngram_lookup ? 0u : native_table_bytes),
                gpu_adaptive?mixed_source_table_credit:(native_ngram_lookup ? 0u : native_table_credit));
            host_builder.add(PhysicalMemoryOwner::ImmutableLookupStaging,
                gpu_adaptive?mixed_source_staging:checkedAdd(immutable_lookup_host_bytes,adaptive_request_bytes,"hosted channel and shared adaptive request scratch"),
                gpu_adaptive?mixed_source_staging_credit:adaptive_request_credit);
            host_builder.add(
                PhysicalMemoryOwner::PrefixHostTier,
                prefix_cache_host_tier_bytes);
            host_builder.add(
                PhysicalMemoryOwner::PrefixArchiveStaging,
                prefix_cache_host_staging_bytes);
            host_builder.add(
                PhysicalMemoryOwner::WeightLoadStaging,
                cfg.weight_load_staging.host_bytes);
            host_builder.add(PhysicalMemoryOwner::ActivationTransportStaging, pipeline_host_bytes);
            appendPhysicalPlan(
                result,
                DeviceMemoryPlan(
                    host_builder.build(), max_seq, activation_seq));
        }
    }


    /* Seal one aggregate before any fit decision. Per-device rows below are
     * presentation only; allocation authority comes from MemoryPlan::admit(). */
    result.sealPhysicalPlan();

    /* Diagnose this group's physical contributions. A multi-group caller must
     * admit the complete aggregate: an individually fitting stage does not
     * prove that the other stages can share the same allocator. */
    for (const auto &device_plan : result.devices)
    {
        if (!device_plan.fits())
        {
            std::ostringstream msg;
            msg << device_plan.bom().resource().id() << ": need "
                << formatMB(device_plan.incremental_bytes())
                << " of new allocation but only "
                << formatMB(device_plan.device_free_bytes()) << " available"
                << " (deficit: " << formatMB(device_plan.deficit()) << ")";
            result.diagnostics.push_back(msg.str());
        }
        else if (device_plan.remaining() < 256ULL * 1024 * 1024)
        {
            std::ostringstream msg;
            msg << device_plan.bom().resource().id()
                << ": tight fit — only "
                << formatMB(device_plan.remaining()) << " unallocated";
            result.diagnostics.push_back(msg.str());
        }
    }

    return result;
}

MemoryPlan MemoryPlanner::plan(
    const ModelMemoryProfile &profile,
    const std::vector<DevicePlanConfig> &device_configs)
{
    auto groups = planGroups(profile, std::span(&device_configs, 1));
    return std::move(groups.front());
}

std::vector<MemoryPlan> MemoryPlanner::planGroups(
    const ModelMemoryProfile &profile,
    std::span<const std::vector<DevicePlanConfig>> device_groups)
{
    if (device_groups.empty())
        throw std::invalid_argument("Memory planning requires an explicit ownership-group inventory");
    SharedOwnerIdentities shared;
    std::map<std::pair<int,std::string>,std::vector<DevicePlanConfig>> adaptive_consumers;
    std::map<std::pair<int,std::string>,std::shared_ptr<const ResolvedImmutableRowTablePlacement>> resolved_host_providers;
    std::map<std::pair<int,std::string>,std::shared_ptr<const ResolvedImmutableRowTablePlacement>> resolved_ngram_consumers;
    for(const auto &group:device_groups)for(const auto &cfg:group)
    {
        if(!ownsNgramLookup(profile,cfg) || !cfg.ngram_table_placement)continue;
        const auto &placement=cfg.ngram_table_placement;
        const auto consumer_key=std::pair{cfg.world_rank,cfg.device.toString()};
        const auto [previous,first]=resolved_ngram_consumers.emplace(consumer_key,placement);
        if(!first && *previous->second!=*placement)
            throw std::invalid_argument("One original ngram consumer cannot share conflicting storage modes or resolved source layouts");
        const auto execution=placement->lookupForConsumer({cfg.world_rank,cfg.device});
        if(execution!=ImmutableRowLookupExecution::HostProvider && execution!=ImmutableRowLookupExecution::AdaptiveGPUCache)continue;
        const auto owner=placement->isGPUAdaptive()?placement->gpuAdaptiveLayout().sourceServices().front().allocator:
            placement->allocators().front();const auto key=std::pair{owner.world_rank,owner.device.toString()};
        const auto [prior,inserted]=resolved_host_providers.emplace(key,placement);
        if(!inserted && *prior->second!=*placement)
            throw std::invalid_argument("One shared host provider cannot deduplicate conflicting resolved layouts/requests/live envelopes");
        if(placement->isHostAdaptive() && immutableRowTableMaximumLiveIds(ngramActivationRows(profile,cfg),cfg.batch_size,
            profile.flash_next->ple_heads,cfg.mtp_target_query_rows,cfg.mtp_enabled?cfg.mtp_target_query_rows:0)>
                placement->hostAdaptiveMaximumLiveRows())
            throw std::invalid_argument("Resolved adaptive provider is smaller than an actual consumer's live-ID envelope");
        if(placement->isGPUAdaptive())
        {
            requireGPUAdaptiveDomain(profile,cfg);
            if(immutableRowTableMaximumLiveIds(ngramActivationRows(profile,cfg),cfg.batch_size,
                profile.flash_next->ple_heads,cfg.mtp_target_query_rows,cfg.mtp_enabled?cfg.mtp_target_query_rows:0)>
                    placement->gpuAdaptiveMaximumLiveRows())
                throw std::invalid_argument("Resolved mixed domain cannot shrink its original live-ID envelope");
        }

    }
    for(const auto &group:device_groups)for(const auto &cfg:group)
    {
        if(!ownsNgramLookup(profile,cfg) || !cfg.ngram_execution_policy || (!cfg.ngram_execution_policy->isHostAdaptive() && !cfg.ngram_execution_policy->isGPUAdaptive()) || cfg.ngram_table_placement)continue;
        const auto host=cfg.device.is_cpu()?PhysicalMemoryResource{cfg.world_rank,cfg.device,cfg.device_total_bytes,cfg.device_free_bytes}
            :cfg.associated_host_memory.value_or(PhysicalMemoryResource{});
        if(!host.valid() || !host.device.is_cpu() || host.world_rank!=cfg.world_rank)
            throw std::invalid_argument("Host adaptive consumers require their exact pre-quota CPU resource");
        const auto key=std::pair{host.world_rank,host.device.toString()};
        if(resolved_host_providers.contains(key))
            throw std::invalid_argument("Shared host provider cannot mix unresolved and resolved adaptive admission");
        adaptive_consumers[key].push_back(cfg);
    }
    for(const auto &[key,configs]:adaptive_consumers)
    {
        const auto &first=configs.front();const auto &policy=*first.ngram_execution_policy;
        const auto host=first.device.is_cpu()?PhysicalMemoryResource{first.world_rank,first.device,first.device_total_bytes,first.device_free_bytes}
            :first.associated_host_memory.value();
        std::size_t maximum=0;
        for(const auto &cfg:configs)
        {
            if(*cfg.ngram_execution_policy!=policy || cfg.ngram_host_placement!=first.ngram_host_placement)
                throw std::invalid_argument("Shared host adaptive consumers have conflicting policy/locality before admission");
            maximum=std::max(maximum,immutableRowTableMaximumLiveIds(ngramActivationRows(profile,cfg),cfg.batch_size,
                profile.flash_next->ple_heads,cfg.mtp_target_query_rows,cfg.mtp_enabled?cfg.mtp_target_query_rows:0));
        }
        const std::vector<std::vector<DevicePlanConfig>> groups{configs};
        const auto input=immutableRowTablePlacementInput(profile,groups,host,first.ngram_host_placement);
        if(policy.isGPUAdaptive())
        {
            for(const auto &cfg:configs)requireGPUAdaptiveDomain(profile,cfg);
            if(input.consumers().size()!=1 || !input.consumers().front().device.is_gpu())
                throw std::invalid_argument("Mixed source cannot be shared by multiple actual GPU PLE consumers");
            const auto requester=input.consumers().front();
            shared.gpu_adaptive.emplace(std::pair{requester.world_rank,requester.device.toString()},
                AdaptiveImmutableLookupGeometry::fromLayout(input.gpuAdaptiveLayout(policy,0,0,maximum),
                    ngramNativeSource(profile),policy.gpu_adaptive));
            continue;
        }
        shared.host_adaptive.emplace(key,CpuAdaptiveImmutableRowCacheGeometry::fromLayout(
            input.hostAdaptiveLayout(policy,0,maximum),{host.world_rank,host.device},maximum,policy.host_adaptive.policy));
    }
    std::vector<MemoryPlan> result;
    result.reserve(device_groups.size());
    for (const auto &group : device_groups)
        result.push_back(planOwnerGroup(profile, group, shared));
    return result;
}

ResidentGraphMemoryPlan MemoryPlanner::planLargestFittingResidentGraphRows(
    const ModelMemoryProfile& profile,
    const std::vector<DevicePlanConfig>& device_configs,
    const std::vector<int>& candidate_rows)
{
    ResidentGraphMemoryPlan selection;

    const auto usesResidentGraphRows = [](const DevicePlanConfig& config)
    {
        return config.device.is_gpu() ||
               config.execution_role ==
                   DeviceExecutionMemoryRole::RoutedExpertParticipant;
    };
    const bool has_bucketed_participant = std::any_of(
        device_configs.begin(),
        device_configs.end(),
        usesResidentGraphRows);
    if (!has_bucketed_participant)
    {
        selection.memory_plan = plan(profile, device_configs);
        selection.device_inputs = device_configs;
        selection.resident_graph_rows =
            device_configs.empty()
                ? 0
                : std::max(
                      1,
                      device_configs.front().activation_seq_len > 0
                          ? device_configs.front().activation_seq_len
                          : device_configs.front().max_seq_len);
        return selection;
    }

    int common_max_rows = std::numeric_limits<int>::max();
    for (const auto& config : device_configs)
    {
        if (!usesResidentGraphRows(config))
            continue;
        const int max_rows =
            config.max_seq_len > 0 ? config.max_seq_len : profile.max_seq_len;
        if (max_rows > 0)
            common_max_rows = std::min(common_max_rows, max_rows);
    }

    const std::vector<int> candidates =
        residentPrefillGraphRowCandidates(candidate_rows, common_max_rows);

    if (candidates.empty())
    {
        selection.memory_plan = plan(profile, device_configs);
        selection.device_inputs = device_configs;
        return selection;
    }

    /*
     * Descending evaluation makes the first successful plan the most
     * economical throughput choice that respects the measured memory
     * contract. Keep the smallest failed plan for a useful hard-failure
     * diagnostic when even the minimum captured graph cannot be resident.
     */
    for (const int candidate : candidates)
    {
        /*
         * A configured prefill bucket is only one member of the retained graph
         * family. MTP verification may own a wider flattened row shape even
         * when the authenticated prompt deliberately uses a tiny exact bucket.
         * Publish and price one common capacity that covers both shapes on
         * every participant; otherwise preflight can approve (for example)
         * nine hidden rows and setup later tries to materialize the retained
         * depth-fifteen sixteen-row publication family into that arena.
         */
        int retained_graph_rows = candidate;
        for (const auto& config : device_configs)
        {
            if (usesResidentGraphRows(config) && config.mtp_enabled)
            {
                retained_graph_rows = std::max(
                    retained_graph_rows,
                    std::max(1, config.mtp_target_query_rows));
            }
        }

        std::vector<DevicePlanConfig> evaluated = device_configs;
        for (auto& config : evaluated)
        {
            if (usesResidentGraphRows(config))
                config.activation_seq_len = retained_graph_rows;
        }

        MemoryPlan candidate_plan = plan(profile, evaluated);
        selection.resident_graph_rows = retained_graph_rows;
        selection.memory_plan = std::move(candidate_plan);
        selection.device_inputs = std::move(evaluated);
        if (selection.memory_plan.fits())
            return selection;
    }

    return selection;
}

std::string MemoryPlan::renderTable() const
{
    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);

    // Header
    table << fort::header
          << "Device" << "Context" << "Act.Seq" << "Weights" << "Retained"
          << "KV Cache" << "State" << "Prefix Stage" << "Prefix Hot"
          << "Graphs" << "Snapshots" << "Collect." << "Activ." << "Wkspace" << "Ret.Wksp"
          << "Total" << "New" << "Avail." << "OK"
          << fort::endr;

    // Column alignments
    table.column(0).set_cell_text_align(fort::text_align::left);
    for (int c = 1; c <= 17; ++c)
    {
        table.column(c).set_cell_text_align(fort::text_align::right);
    }
    table.column(18).set_cell_text_align(fort::text_align::center);

    // Data rows
    for (const auto& d : devices)
    {
        table << d.device().to_string()
              << d.max_seq_len()
              << d.activation_seq_len()
              << formatMB(d.weight_bytes())
              << formatMB(d.retained_weight_bytes())
              << formatMB(d.kv_cache_bytes())
              << formatMB(d.persistent_state_bytes())
              << formatMB(d.prefix_cache_staging_bytes())
              << formatMB(d.prefix_cache_device_hot_bytes())
              << formatMB(d.captured_graph_bytes())
              << formatMB(d.graph_snapshot_bytes())
              << formatMB(d.collective_bytes())
              << formatMB(d.activation_bytes())
              << formatMB(d.workspace_bytes())
              << formatMB(d.retained_workspace_bytes())
              << formatMB(d.total_bytes())
              << formatMB(d.incremental_bytes())
              << formatMB(d.device_free_bytes())
              << (d.fits() ? "\xe2\x9c\x93" : "\xe2\x9c\x97")  // ✓ / ✗
              << fort::endr;
    }

    std::string output = table.to_string();

    // Append diagnostics
    if (!diagnostics.empty())
    {
        output += "\n";
        for (const auto& diag : diagnostics)
        {
            output += "  " + diag + "\n";
        }
    }

    return output;
}

ImmutableRowTablePlacementInput MemoryPlanner::immutableRowTablePlacementInput(
    const ModelMemoryProfile &profile, std::span<const std::vector<DevicePlanConfig>> groups,
    PhysicalMemoryResource host, ImmutableRowTableHostPlacement host_placement)
{
    auto input=immutableRowTablePlacementInputIfConsumed(profile,groups,host,host_placement);
    if(!input) throw std::invalid_argument("This rank has no actual native row-table consumer");
    return *input;
}
std::optional<ImmutableRowTablePlacementInput> MemoryPlanner::immutableRowTablePlacementInputIfConsumed(
    const ModelMemoryProfile &profile,std::span<const std::vector<DevicePlanConfig>> groups,
    PhysicalMemoryResource host, ImmutableRowTableHostPlacement host_placement)
{
    std::vector<PhysicalMemoryResource> consumers;
    PhysicalMemoryPlanBuilder observations;
    for (const auto &group : groups)
        for (const auto &cfg : group)
            if (ownsNgramLookup(profile, cfg))
            {
                const PhysicalMemoryResource r{cfg.world_rank, cfg.device, cfg.device_total_bytes, cfg.device_free_bytes};
                observations.add(r, PhysicalMemoryOwner::ImmutableTableStorage, 0u);
                if (std::none_of(consumers.begin(), consumers.end(), [&](const auto &present) {
                    return present.world_rank == r.world_rank && present.device == r.device; })) consumers.push_back(r);
            }
    (void)observations.build();
    if(consumers.empty()) return std::nullopt;
    return ImmutableRowTablePlacementInput::forConsumers(ngramExtent(profile), host, consumers, host_placement);
}
std::vector<ImmutableRowTableUploadBinding> MemoryPlanner::immutableRowTableUploadBindings(
    const ModelMemoryProfile &profile,std::span<const std::vector<DevicePlanConfig>> groups)
{
    std::vector<ImmutableRowTableUploadBinding> result;
    for(const auto &group:groups)for(const auto &cfg:group)
    {
        if(!ownsNgramLookup(profile,cfg)||!cfg.device.is_gpu()||!cfg.ordinary_weight_load_geometry)continue;
        const PhysicalMemoryAllocatorIdentity identity{cfg.world_rank,cfg.device};
        const auto found=std::find_if(result.begin(),result.end(),[&](const auto &binding){return binding.consumer==identity;});
        if(found==result.end())result.push_back({identity,*cfg.ordinary_weight_load_geometry});
        else
        {
            const auto &a=found->ordinary,&b=*cfg.ordinary_weight_load_geometry;
            if(!sameUploadGeometry(a,b))
                throw std::invalid_argument("One PLE consumer has conflicting original loader geometry");
        }
    }
    return result;
}
ImmutableRowTableModeContributions MemoryPlanner::immutableRowTableModeContributions(
    const ModelMemoryProfile &profile, std::span<const std::vector<DevicePlanConfig>> groups,
    std::span<const ImmutableRowTableUploadBinding> ordinary_upload)
{
    const auto extent=ngramExtent(profile);
    const auto hosted = [&](ImmutableRowStorage storage) {
        PhysicalMemoryPlanBuilder contribution;
        for (const auto &group : groups)
            for (const auto &cfg : group)
            {
                if (!ownsNgramLookup(profile, cfg)) continue;
                if (cfg.ngram_table_placement)
                    throw std::invalid_argument("Ngram candidates require unresolved inputs after higher-priority geometry is frozen");
                const auto costs = ngramModeMemory(profile, cfg, immutableRowLookupExecution(storage));
                const PhysicalMemoryResource resource{cfg.world_rank, cfg.device, cfg.device_total_bytes, cfg.device_free_bytes};
                contribution.add(resource, PhysicalMemoryOwner::ActivationArena, costs.arena)
                    .add(resource, PhysicalMemoryOwner::NativeGraphExecutable, costs.terminal);
                if (costs.mapped)
                {
                    if (!cfg.associated_host_memory || !cfg.associated_host_memory->valid() ||
                        !cfg.associated_host_memory->device.is_cpu() || cfg.associated_host_memory->world_rank != cfg.world_rank)
                        throw std::invalid_argument("Hosted ngram candidate requires exact rank-local CPU/NUMA authority");
                    contribution.add(*cfg.associated_host_memory, PhysicalMemoryOwner::ImmutableLookupStaging, costs.mapped);
                }
            }
        return contribution.build();
    };
    bool eligible=true;
    std::set<std::pair<int,std::string>> bindings;
    for(const auto &binding:ordinary_upload)
    {
        if(!binding.consumer.valid() || binding.consumer.world_rank<0 || !binding.consumer.device.is_gpu() ||
            !bindings.emplace(binding.consumer.world_rank,binding.consumer.device.toString()).second)
            throw std::invalid_argument("Resident ngram upload has invalid or duplicate consumer binding");
        bool actual=false;
        for(const auto &group:groups)
            for(const auto &cfg:group)
                actual=actual || (ownsNgramLookup(profile,cfg) &&
                    binding.consumer==PhysicalMemoryAllocatorIdentity{cfg.world_rank,cfg.device});
        if(!actual) throw std::invalid_argument("Resident ngram upload binding names a non-PLE participant");
    }
    for(const auto &group:groups)
        for(const auto &cfg:group)
            if(ownsNgramLookup(profile,cfg) && cfg.device.is_gpu())
            {
                const auto found=std::find_if(ordinary_upload.begin(),ordinary_upload.end(),[&](const auto &binding) {
                    return binding.consumer==PhysicalMemoryAllocatorIdentity{cfg.world_rank,cfg.device}; });
                if(found==ordinary_upload.end()) { eligible=false; continue; }
                const auto &ordinary=found->ordinary;
                if(!cfg.ordinary_weight_load_geometry||!sameUploadGeometry(ordinary,*cfg.ordinary_weight_load_geometry))
                    throw std::invalid_argument("Resident ngram binding has no matching original loader geometry owner");
                if(cfg.weight_load_staging.host_bytes!=ordinary.host_staging_bytes ||
                    cfg.weight_load_staging.device_bytes!=ordinary.staging_bytes)
                    throw std::invalid_argument("Resident ngram candidate does not match its original ordinary-loader staging geometry");
                eligible=residentImmutableRowUploadFits(extent.rowBytes(),ordinary) && eligible;
            }
    return {eligible ? std::optional{residentImmutableRowUploadIncrementalBOM()} : std::nullopt,
        hosted(ImmutableRowStorage::RAM),hosted(ImmutableRowStorage::MemoryMapped)};
}

std::optional<ImmutableRowTableAdmissionRequest> MemoryPlanner::immutableRowTableAdmissionRequest(
    const ModelMemoryProfile &profile, const ImmutableRowTablePolicy &requested,
    std::span<const std::vector<DevicePlanConfig>> groups,
    std::span<const ImmutableRowTableUploadBinding> ordinary_upload,
    ImmutableRowTableHostPlacement host_placement)
{
    std::optional<PhysicalMemoryResource> host;
    PhysicalMemoryPlanBuilder observations;
    for (const auto &group : groups)
        for (const auto &cfg : group)
        {
            if (!ownsNgramLookup(profile, cfg)) continue;
            const auto resource = cfg.device.is_cpu()
                ? std::optional{PhysicalMemoryResource{cfg.world_rank, cfg.device,
                    cfg.device_total_bytes, cfg.device_free_bytes}}
                : cfg.associated_host_memory;
            if (!resource || !resource->valid() || !resource->device.is_cpu() ||
                resource->world_rank != cfg.world_rank)
                throw std::invalid_argument("Ngram consumer has no exact frozen host allocator");
            const PhysicalMemoryAllocatorIdentity identity{resource->world_rank, resource->device};
            if (host && PhysicalMemoryAllocatorIdentity{host->world_rank, host->device} != identity)
                throw std::invalid_argument("Ngram consumers disagree on their frozen host allocator");
            host = resource;
            // The canonical BOM builder checks repeated capacity observations;
            // there is no second budget or inferred NUMA-to-device conversion.
            observations.add(*resource, PhysicalMemoryOwner::ImmutableTableStorage, 0u);
        }
    if (!host) return std::nullopt;
    (void)observations.build();
    auto input=immutableRowTablePlacementInput(profile,groups,*host,host_placement);
    auto contributions=requested.isGPUAdaptive()?ImmutableRowTableModeContributions{}:
        immutableRowTableModeContributions(profile,groups,ordinary_upload);
    PhysicalMemoryPlanBuilder execution,required;
    std::vector<ImmutableRowTableExecutionConsumer> consumers;
    for(const auto &group:groups)for(const auto &cfg:group)
    {
        if(!ownsNgramLookup(profile,cfg))continue;
        if(!cfg.ngram_execution_policy || *cfg.ngram_execution_policy!=requested || cfg.ngram_table_placement)
            throw std::invalid_argument("Final ngram request requires the unchanged pre-quota execution policy on every actual consumer");
        const auto envelope=ngramExecutionEnvelope(profile,cfg);
        for(const auto &bom:envelope.resources())execution.add(bom);
        consumers.push_back({{cfg.world_rank,cfg.device,cfg.device_total_bytes,cfg.device_free_bytes},
            ngramActivationRows(profile,cfg),cfg.batch_size,cfg.max_seq_len>0?cfg.max_seq_len:profile.max_seq_len,
            cfg.tensor_parallel_assignment?cfg.tensor_parallel_assignment->head_count:
                std::max(1,profile.n_heads/std::max(1,cfg.total_shards)),
            cfg.first_layer,cfg.last_layer>=0?cfg.last_layer:profile.n_layers-1,cfg.total_shards,cfg.mtp_target_query_rows,
            cfg.mtp_enabled?cfg.mtp_target_query_rows:0,cfg.generation_request_capacity,
            continuationActivationGeometry(profile,cfg,std::nullopt),
            requested.isFixedHeadSharded()?cfg.shard_index:-1,
            requested.isFixedHeadSharded()?cfg.ngram_tp_topology:ImmutableRowTableTPTopology::None});
    }
    if(requested.isHostAdaptive() || requested.isGPUAdaptive())
    {
        std::size_t maximum=0;
        for(const auto &c:consumers)
            maximum=std::max(maximum,immutableRowTableMaximumLiveIds(c.rows,c.batch,profile.flash_next->ple_heads,
                c.mtp_rows,c.retained_mtp_rows));
        for(const auto &group:groups)for(const auto &cfg:group)
            if(ownsNgramLookup(profile,cfg) && cfg.ngram_host_placement!=host_placement)
                throw std::invalid_argument("Final adaptive locality differs from its pre-quota consumer contract");
        PhysicalMemoryPlan mandatory;
        if(requested.isGPUAdaptive())
        {
            const auto geometry=AdaptiveImmutableLookupGeometry::fromLayout(input.gpuAdaptiveLayout(requested,0,0,maximum),
                ngramNativeSource(profile),requested.gpu_adaptive);
            mandatory=geometry.mandatoryBOM(input.consumers().front(),*host);
        }
        else
            mandatory=CpuAdaptiveImmutableRowCacheGeometry::fromLayout(input.hostAdaptiveLayout(requested,0,maximum),
                {host->world_rank,host->device},maximum,requested.host_adaptive.policy).mandatoryBOM(*host);
        for(const auto &bom:mandatory.resources())execution.add(bom);
    }
    // Reuse the same complete fixed owner construction. Comparing this required
    // footprint prevents common arena bytes from disguising an omitted provision.
    std::vector<std::vector<DevicePlanConfig>> consumer_groups;
    for(const auto &group:groups)
    {
        std::vector<DevicePlanConfig> actual;
        for(const auto &cfg:group)if(ownsNgramLookup(profile,cfg))actual.push_back(cfg);
        if(!actual.empty())consumer_groups.push_back(std::move(actual));
    }
    const auto fixed=planGroups(profile,consumer_groups);
    for(const auto &group:fixed)for(const auto &bom:group.physicalPlan().resources())required.add(bom);
    auto provision=std::shared_ptr<const ImmutableRowTableExecutionProvision>(new ImmutableRowTableExecutionProvision(
        requested,*profile.flash_next,host_placement,std::move(consumers),execution.build(),required.build()));
    if(requested.isFixedHeadSharded())provision->require(requested,input,provision->requiredBase());
    // Execution is already in the frozen base; only the original bulk table
    // and genuinely incremental resident upload remain lowest priority.
    contributions.ram={};contributions.mmap={};
    return ImmutableRowTableAdmissionRequest{requested,std::move(input),std::move(contributions),std::move(provision)};
}

std::shared_ptr<const ImmutableRowTableExecutionProvision> MemoryPlanner::restoreImmutableRowTableExecutionProvision(
    const ModelMemoryProfile &profile,const ImmutableRowTablePolicy &requested,
    const ImmutableRowTablePlacementInput &input,std::vector<ImmutableRowTableExecutionConsumer> consumers,
    PhysicalMemoryPlan execution,PhysicalMemoryPlan required_base)
{
    if(!profile.flash_next || ngramExtent(profile)!=input.extent() || consumers.empty())
        throw std::invalid_argument("Remote ngram provision requires the receiver's exact original source metadata");
    PhysicalMemoryPlanBuilder independently_derived;
    for(const auto &identity:consumers)
    {
        if(identity.rows<=0 || identity.batch<=0 || identity.context<identity.rows || identity.local_heads<=0 ||
            identity.local_heads>profile.n_heads || identity.shards<=0 || identity.first_layer<0 ||
            identity.last_layer<identity.first_layer || identity.last_layer>=profile.n_layers ||
            identity.mtp_rows<0 || identity.retained_mtp_rows<0 || identity.request_capacity<=0 ||
            (identity.retained_mtp_rows!=0 && identity.retained_mtp_rows!=identity.mtp_rows) ||
            std::none_of(input.consumers().begin(),input.consumers().end(),[&](const auto &c) {
                return c.sameAllocator(identity.resource); }))
            throw std::invalid_argument("Remote ngram provision has invalid or foreign graph geometry");
        const auto &a=identity.activation;
        qwen4exp::validateQwenSparseAttentionArithmetic(a.attention_arithmetic,
                                                        identity.resource.device.type);
        if (a.attention_arithmetic ==
                qwen4exp::QwenSparseAttentionArithmetic::TileFP16WithFP32Totals33 ||
            (a.attention_phase_schedule ==
                 qwen4exp::QwenSparseAttentionPhaseSchedule::Prefill8Decode33 &&
             a.attention_arithmetic !=
                 qwen4exp::QwenSparseAttentionArithmetic::TileFP16WithFP32Totals))
            throw std::invalid_argument("Remote QSA setup policy is unresolved or conflicting");
        if(a.attention_phase_schedule!=qwen4exp::QwenSparseAttentionPhaseSchedule::Uniform &&
           a.attention_phase_schedule!=qwen4exp::QwenSparseAttentionPhaseSchedule::Prefill8Decode33)
            throw std::invalid_argument("Remote ngram attention phase schedule is invalid");
        if(a.ngram_lookup || a.ngram_fixed_head_participants!=0 || a.batch_size!=identity.batch || a.resident_graph_rows!=identity.rows ||
            a.context_tokens!=identity.context || a.first_layer!=identity.first_layer || a.last_layer!=identity.last_layer ||
            a.total_shards!=identity.shards || a.mtp_target_query_rows!=identity.mtp_rows ||
            a.retained_mtp_candidate_rows!=identity.retained_mtp_rows || a.generation_request_capacity!=identity.request_capacity ||
            a.local_d_ff<=0 || a.local_n_heads<=0 || a.local_n_heads>profile.n_heads ||
            a.local_n_kv_heads<=0 || a.local_n_kv_heads>profile.n_kv_heads || a.local_vocab<0 || a.local_vocab>profile.vocab_size ||
            (a.routed_compute_policy!=RoutedExpertComputePolicy::Replicated &&
             a.routed_compute_policy!=RoutedExpertComputePolicy::Apportioned &&
             a.routed_compute_policy!=RoutedExpertComputePolicy::TensorSharded &&
             a.routed_compute_policy!=RoutedExpertComputePolicy::GateUpOwnedDownColumns) ||
            (a.mtp_terminal_logits_layout!=MTPTerminalLogitsLayout::FullVocabularyPerParticipant &&
             a.mtp_terminal_logits_layout!=MTPTerminalLogitsLayout::VocabularyShardPerParticipant) ||
            (a.mtp_shifted_kv_head_layout!=MTPShiftedKVHeadLayout::PrimaryTensorParallelShard &&
             a.mtp_shifted_kv_head_layout!=MTPShiftedKVHeadLayout::FullModelPerParticipant))
            throw std::invalid_argument("Remote ngram provision has inconsistent common activation geometry");
        // A canonical unsharded descriptor cannot invent a smaller vocabulary
        // or projection. Unequal TP slices remain explicit sender-owned shape
        // metadata; the startup rank is their existing compilation authority.
        if((identity.shards==1 && (a.local_d_ff!=profile.d_ff || a.local_n_heads!=profile.n_heads ||
                a.local_n_kv_heads!=profile.n_kv_heads || a.local_vocab!=(a.replicated_dense_decode?profile.vocab_size:0))) ||
            (a.replicated_dense_decode && (a.local_d_ff!=profile.d_ff || a.local_n_heads!=profile.n_heads ||
                a.local_vocab!=profile.vocab_size || a.mtp_terminal_logits_layout!=MTPTerminalLogitsLayout::FullVocabularyPerParticipant)) ||
            (!a.replicated_dense_decode && a.local_n_heads!=identity.local_heads) || a.local_d_ff>profile.d_ff)
            throw std::invalid_argument("Remote ngram common descriptor contradicts canonical whole-storage/sharding geometry");
        DevicePlanConfig cfg;cfg.world_rank=identity.resource.world_rank;cfg.device=identity.resource.device;
        cfg.device_total_bytes=identity.resource.total_bytes;cfg.device_free_bytes=identity.resource.admission_available_bytes;
        cfg.ngram_execution_policy=requested;cfg.associated_host_memory=input.host();
        cfg.activation_seq_len=identity.rows;cfg.batch_size=identity.batch;cfg.max_seq_len=identity.context;
        cfg.first_layer=identity.first_layer;cfg.last_layer=identity.last_layer;cfg.total_shards=identity.shards;
        if(requested.isFixedHeadSharded())
        {
            cfg.shard_index=identity.tp_coordinate;cfg.ngram_tp_topology=identity.tp_topology;
            cfg.local_tp_backend=cfg.device.is_cuda()?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL;
        }
        cfg.tensor_parallel_assignment=DeviceShardingAssignment{.device=cfg.device,.head_count=identity.local_heads};
        cfg.mtp_target_query_rows=identity.mtp_rows;cfg.mtp_enabled=identity.retained_mtp_rows!=0;
        cfg.generation_request_capacity=identity.request_capacity;
        if(!ownsNgramLookup(profile,cfg))throw std::invalid_argument("Remote ngram geometry names a non-PLE interval");
        const auto envelope=ngramExecutionEnvelope(profile,cfg);
        for(const auto &bom:envelope.resources())independently_derived.add(bom);
    }
    if(requested.isHostAdaptive() || requested.isGPUAdaptive())
    {
        std::size_t maximum=0;
        for(const auto &c:consumers)
            maximum=std::max(maximum,immutableRowTableMaximumLiveIds(c.rows,c.batch,profile.flash_next->ple_heads,
                c.mtp_rows,c.retained_mtp_rows));
        PhysicalMemoryPlan mandatory;
        if(requested.isGPUAdaptive())
            mandatory=AdaptiveImmutableLookupGeometry::fromLayout(input.gpuAdaptiveLayout(requested,0,0,maximum),
                ngramNativeSource(profile),requested.gpu_adaptive).mandatoryBOM(input.consumers().front(),input.host());
        else
            mandatory=CpuAdaptiveImmutableRowCacheGeometry::fromLayout(input.hostAdaptiveLayout(requested,0,maximum),
                {input.host().world_rank,input.host().device},maximum,requested.host_adaptive.policy).mandatoryBOM(input.host());
        for(const auto &bom:mandatory.resources())independently_derived.add(bom);
    }
    const auto derived=independently_derived.build();
    if(derived.requiredFootprintMismatch(execution) || execution.requiredFootprintMismatch(derived))
        throw std::invalid_argument("Remote ngram provision execution bytes disagree with canonical source/geometry");
    PhysicalMemoryPlanBuilder mandatory_floor(derived);
    for(const auto &identity:consumers)
        mandatory_floor.add(identity.resource,PhysicalMemoryOwner::ActivationArena,
            ActivationMemoryEstimator::estimate(profile,identity.activation,identity.resource.device));
    if(required_base.requiredFootprintMismatch(mandatory_floor.build()))
        throw std::invalid_argument("Remote ngram provision fixed base omits common activation plus mandatory execution");
    auto result=std::shared_ptr<const ImmutableRowTableExecutionProvision>(new ImmutableRowTableExecutionProvision(
        requested,*profile.flash_next,input.hostPlacement(),std::move(consumers),std::move(execution),std::move(required_base)));
    result->require(requested,input,result->requiredBase());return result;
}

std::vector<std::shared_ptr<const CPUSerialExpertStorageAdmission>>
MemoryPlanner::cpuSerialExpertStorageAdmissions(std::span<const DevicePlanConfig> configs)
{
    std::vector<std::shared_ptr<const CPUSerialExpertStorageAdmission>> result;
    for(const auto &config:configs)
        if(config.cpu_serial_expert_storage)
        {
            if(config.cpu_serial_expert_storage->allocator!=PhysicalMemoryAllocatorIdentity{config.world_rank,config.device})
                throw std::invalid_argument("CPU serial declaration differs from its admitted allocator");
            result.push_back(config.cpu_serial_expert_storage);
        }
    return result;
}

} // namespace llaminar2
