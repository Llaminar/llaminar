/**
 * @file MemoryPlanner.cpp
 * @brief Exact memory admission arithmetic for production graph families.
 *
 * The implementation turns the typed lifetime declarations in
 * @ref DevicePlanConfig into checked byte counts.  In particular, sparse MoE
 * packet storage is not guessed from a model name: a serial ExpertOverlay
 * participant family has one reusable packet per participant, whereas an
 * independently retained graph family has one per participating layer.
 */

#include "planning/MemoryPlanner.h"
#include "planning/WeightMemoryEstimator.h"
#include "planning/KVCacheMemoryEstimator.h"
#include "planning/PersistentStateMemoryEstimator.h"
#include "planning/ActivationMemoryEstimator.h"
#include "planning/CollectiveMemoryEstimator.h"
#include "planning/WorkspaceMemoryEstimator.h"
#include "planning/CapturedGraphMemoryEstimator.h"
#include "execution/local_execution/engine/PrefillBucketUtils.h"
#include "execution/prefix_cache/PrefixArchiveIOGeometry.h"
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
 * expert index and weight per route.  Local compaction produces a route for
 * each router top-k choice, so the flattened request rows are expanded before
 * their FP32 storage is counted.
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

} // anonymous namespace

MemoryPlan MemoryPlanner::plan(
    const ModelMemoryProfile& profile,
    const std::vector<DevicePlanConfig>& device_configs)
{
    MemoryPlan result;
    result.devices.reserve(device_configs.size());

    /*
     * openShared() retains one archive object, lock domain, and scratch buffer
     * for a model archive on each MPI rank.  Several local GPU participants
     * may reference that same physical host allocation, so price its archive
     * key once rather than inventing one scratch per logical device plan.
     */
    std::set<std::pair<int, std::string>> accounted_prefix_archives;

    for (const auto& cfg : device_configs)
    {
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
        std::size_t graph_snapshot_bytes = 0u;
        std::size_t primary_weight_bytes = 0u;
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
                            cfg.captured_serving_graphs
                                .mtp_graph_owners
                                .auxiliaryExecutableSlotCount(),
                    });
        }

        if (cfg.device.is_gpu())
        {
            graph_snapshot_bytes =
                cfg.graph_snapshot_memory.per_accelerator_bytes;
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
            std::max(0, profile.n_layers - profile.mtp_layer_count);
        const bool owns_terminal_main_layer =
            main_layer_count > 0 &&
            cfg.first_layer <= main_layer_count - 1 &&
            last_layer >= main_layer_count - 1;
        const int weight_last_layer =
            cfg.mtp_enabled &&
                    owns_terminal_main_layer &&
                    profile.mtp_layer_count > 0
                ? profile.n_layers - 1
                : last_layer;

        // Weight estimation
        auto weight_est = WeightMemoryEstimator::estimate(
            profile, cfg.device,
            cfg.shard_index, cfg.total_shards,
            cfg.first_layer, weight_last_layer,
            cfg.weight_residency,
            cfg.tensor_parallel_assignment);
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
                        static_cast<std::size_t>(profile.n_layers), 0);
                    if (cfg.mtp_enabled && profile.mtp_layer_count > 0)
                    {
                        /*
                         * Trailing MTP blocks are part of the replicated
                         * verifier sidecar even when their GGUF names use the
                         * routed-expert suffix. Ordinary base-model routed
                         * experts remain owned exclusively by ExpertOverlay.
                         */
                        for (int layer = main_layer_count;
                             layer < profile.n_layers;
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
                        sidecar_residency);
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
                                        profile.n_layers),
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
                        mirrored_residency);
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
                if (profile.mtp_layer_count <= 0 ||
                    main_layer_count >= profile.n_layers)
                {
                    throw std::invalid_argument(
                        "Replicated MTP sidecar admission requires trailing predictor-layer metadata");
                }

                DeviceWeightResidency sidecar_residency;
                if (profile.expert_count > 0)
                {
                    sidecar_residency =
                        DeviceWeightResidency::
                            continuationWithSelectedRoutedExperts(
                                profile.expert_count,
                                std::vector<int>(
                                    static_cast<std::size_t>(
                                        profile.n_layers),
                                    0));
                }

                /*
                 * WeightMemoryEstimator includes non-layer tensors for any
                 * layer interval because PP endpoints may own them. Price the
                 * trailing predictor interval and subtract the exact same
                 * estimator's globals-only view. The remainder is therefore
                 * precisely the dense/shared predictor block materialized by
                 * participantReplicaNames(ExpertOverlay), with routed parents
                 * excluded by the zero-expert residency contract.
                 */
                const auto sidecar_and_globals =
                    WeightMemoryEstimator::estimate(
                        profile,
                        cfg.device,
                        /*shard_index=*/0,
                        /*total_shards=*/1,
                        main_layer_count,
                        profile.n_layers - 1,
                        sidecar_residency);
                const auto globals_only =
                    WeightMemoryEstimator::estimate(
                        profile,
                        cfg.device,
                        /*shard_index=*/0,
                        /*total_shards=*/1,
                        profile.n_layers,
                        profile.n_layers - 1,
                        sidecar_residency);
                if (sidecar_and_globals.device_bytes <
                    globals_only.device_bytes)
                {
                    throw std::logic_error(
                        "Replicated MTP sidecar estimate underflowed its globals-only view");
                }
                additional_weight_bytes = checkedAdd(
                    additional_weight_bytes,
                    sidecar_and_globals.device_bytes -
                        globals_only.device_bytes,
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
                    cfg.mtp_enabled);
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
             */
            const bool prefix_enabled =
                cfg.device.is_gpu() &&
                cfg.prefix_cache.enabled &&
                cfg.prefix_cache.storage_mode !=
                    PrefixCacheStorageMode::Disabled &&
                cfg.prefix_cache.storage_mode !=
                    PrefixCacheStorageMode::Device &&
                persistent_state.main_full_attention_layers > 0;
            if (prefix_enabled)
            {
                const int prefix_block_tokens =
                    std::max(1, cfg.prefix_cache.block_size);
                const auto logical_block =
                    KVCacheMemoryEstimator::estimateGPULogicalBlock(
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
                    one_fa_layer_bytes,
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

                size_t prefix_block_bytes = checkedAdd(
                    checkedMultiply(
                        static_cast<size_t>(
                            persistent_state.main_full_attention_layers),
                        one_fa_layer_bytes,
                        "prefix main full-attention payload"),
                    persistent_state.prefix_hybrid_device_state_bytes,
                    "prefix main KV and hybrid payload");
                if (cfg.mtp_enabled &&
                    persistent_state.mtp_full_attention_layers > 0)
                {
                    prefix_block_bytes = checkedAdd(
                        prefix_block_bytes,
                        checkedMultiply(
                            static_cast<size_t>(
                                persistent_state.mtp_full_attention_layers),
                                one_shifted_fa_layer_bytes,
                                "prefix shifted-MTP payload"),
                        "prefix main and shifted-MTP payload");
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
                        prefix_block_bytes = checkedAdd(
                            prefix_block_bytes,
                            checkedMultiply(
                                static_cast<size_t>(
                                    std::max(0, profile.d_model)),
                                sizeof(float),
                                "prefix terminal hidden state"),
                            "prefix payload plus terminal hidden state");
                    }

                    size_t terminal_vocab = static_cast<size_t>(
                        std::max(0, profile.vocab_size));
                    if (cfg.mtp_terminal_logits_layout ==
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
                    prefix_block_bytes = checkedAdd(
                        prefix_block_bytes,
                        checkedMultiply(
                            terminal_vocab,
                            sizeof(float),
                            "prefix terminal logits"),
                        "prefix payload plus terminal logits");
                }

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
                persistent_state.main_full_attention_layers > 0)
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
            if (cfg.total_shards > 1)
            {
                if (cfg.local_tp_backend == CollectiveBackendType::AUTO)
                {
                    throw std::invalid_argument(
                        "Multi-shard memory planning requires a resolved LocalTP collective backend");
                }
                collective_bytes =
                    CollectiveMemoryEstimator::localTP(
                        max_seq,
                        profile.d_model,
                        cfg.local_tp_backend)
                        .perDeviceBytes();
            }
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

        if (cfg.execution_role == DeviceExecutionMemoryRole::ContinuationGraph)
        {
            activation_bytes = ActivationMemoryEstimator::estimate(
                profile,
                ActivationGraphMemoryGeometry{
                    .batch_size = cfg.batch_size,
                    .resident_graph_rows = activation_seq,
                    .local_d_ff = local_d_ff,
                    .local_n_heads = local_n_heads,
                    .local_n_kv_heads = local_kv_heads,
                    .local_vocab = cfg.tensor_parallel_assignment.has_value()
                        ? cfg.tensor_parallel_assignment->vocab_count
                        : 0,
                    .first_layer = cfg.first_layer,
                    .last_layer = last_layer,
                    .total_shards = cfg.total_shards,
                    .mtp_target_query_rows = cfg.mtp_target_query_rows,
                    .mtp_terminal_logits_layout =
                        cfg.mtp_terminal_logits_layout,
                },
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

            // Workspace estimation
            workspace_bytes = WorkspaceMemoryEstimator::estimate(
                profile,
                WorkspaceMemoryGeometry{
                    .device = cfg.device,
                    .device_compute_units = cfg.device_compute_units,
                    .batch_size = cfg.batch_size,
                    .resident_graph_rows = activation_seq,
                    .max_context_rows = max_seq,
                    .owns_embedding = cfg.owns_embedding,
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
                    .apportioned_routed_experts =
                        cfg.weight_residency.selectsRoutedExperts(),
                    .mtp_target_query_rows =
                        cfg.mtp_enabled
                            ? cfg.mtp_target_query_rows
                            : 0,
                    .mtp_terminal_logits_layout =
                        cfg.mtp_terminal_logits_layout,
                });
            retained_workspace_bytes = std::min(
                workspace_bytes,
                cfg.retained_workspace_bytes);
        }
        else
        {
            const int participant_graph_rows =
                cfg.serial_routed_expert_compact_rows > 0
                    ? cfg.serial_routed_expert_compact_rows
                    : activation_seq;
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
                        .batch_size = cfg.batch_size,
                        .resident_graph_rows = participant_graph_rows,
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
                    });
            retained_workspace_bytes = std::min(
                workspace_bytes,
                cfg.retained_workspace_bytes);
        }

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
                captured_graph_bytes)
            .add(
                PhysicalMemoryOwner::GraphSnapshotArena,
                graph_snapshot_bytes)
            .add(PhysicalMemoryOwner::LocalCollective, collective_bytes)
            .add(PhysicalMemoryOwner::ActivationArena, activation_bytes)
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
             cfg.weight_load_staging.host_bytes != 0u))
        {
            if (!cfg.associated_host_memory.has_value() ||
                !cfg.associated_host_memory->valid() ||
                !cfg.associated_host_memory->device.is_cpu() ||
                cfg.associated_host_memory->world_rank != cfg.world_rank)
            {
                throw std::invalid_argument(
                    "GPU memory admission requires the exact rank-local host-memory resource for its CPU-side allocations");
            }
            PhysicalMemoryBOMBuilder host_builder(
                *cfg.associated_host_memory);
            host_builder.add(
                PhysicalMemoryOwner::PrefixHostTier,
                prefix_cache_host_tier_bytes);
            host_builder.add(
                PhysicalMemoryOwner::PrefixArchiveStaging,
                prefix_cache_host_staging_bytes);
            host_builder.add(
                PhysicalMemoryOwner::WeightLoadStaging,
                cfg.weight_load_staging.host_bytes);
            appendPhysicalPlan(
                result,
                DeviceMemoryPlan(
                    host_builder.build(), max_seq, activation_seq));
        }
    }


    /* Seal one aggregate before any fit decision. Per-device rows below are
     * presentation only; allocation authority comes from MemoryPlan::admit(). */
    result.sealPhysicalPlan();

    /* Diagnose only complete physical rows. A logical sub-plan can fit while
     * the aggregate allocator does not, which is precisely the split-brain
     * failure this plan representation is intended to make impossible. */
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

} // namespace llaminar2
