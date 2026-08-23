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
#include "execution/local_execution/engine/PrefillBucketUtils.h"
#include "utils/Logger.h"

#include "fort.hpp"

#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <limits>

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

    for (const auto& cfg : device_configs)
    {
        DeviceMemoryPlan dev_plan;
        dev_plan.device = cfg.device;
        dev_plan.device_total_bytes = cfg.device_total_bytes;
        dev_plan.device_free_bytes = cfg.device_free_bytes;
        dev_plan.headroom_bytes = cfg.headroom_bytes;

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

        dev_plan.max_seq_len = max_seq;
        dev_plan.activation_seq_len = activation_seq;

        // If TP sharded, divide KV heads
        if (cfg.total_shards > 1 && cfg.local_kv_heads <= 0 &&
            !cfg.tensor_parallel_assignment.has_value())
        {
            local_kv_heads = std::max(1, profile.n_kv_heads / cfg.total_shards);
        }

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
            cfg.weight_residency);
        dev_plan.weight_bytes = weight_est.device_bytes;

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
                dev_plan.additional_weight_bytes = checkedAdd(
                    dev_plan.additional_weight_bytes,
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
                dev_plan.additional_weight_bytes = checkedAdd(
                    dev_plan.additional_weight_bytes,
                    component_bytes,
                    additional_set ==
                            AdditionalPersistentWeightSet::
                                MirroredDecodeEmbedding
                        ? "additional mirrored decode embedding"
                        : "additional mirrored MTP terminal head");
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
            dev_plan.retained_weight_bytes =
                dev_plan.total_weight_bytes();
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
                    cfg.tensor_parallel_assignment.has_value()
                        ? cfg.tensor_parallel_assignment->head_count
                        : std::max(
                              1,
                              profile.n_heads /
                                  std::max(1, cfg.total_shards)),
                    cfg.total_shards,
                    cfg.first_layer,
                    last_layer,
                    cfg.kv_precision,
                    cfg.mtp_enabled);
            dev_plan.kv_cache_bytes =
                persistent_state.kv_cache_bytes;
            dev_plan.live_recurrent_state_bytes =
                persistent_state.live_recurrent_state_bytes;
            dev_plan.checkpoint_state_bytes =
                persistent_state.checkpoint_state_bytes;
            dev_plan.persistent_state_bytes =
                persistent_state.stateBytes();
            if (cfg.total_shards > 1)
            {
                dev_plan.collective_bytes =
                    CollectiveMemoryEstimator::localTP(
                        max_seq,
                        profile.d_model)
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
            dev_plan.activation_bytes = ActivationMemoryEstimator::estimate(
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
                dev_plan.activation_bytes = checkedAdd(
                    dev_plan.activation_bytes,
                    routedParticipantActivationBytes(
                        profile,
                        cfg,
                        cfg.first_layer,
                        last_layer,
                        activation_seq),
                    "continuation local-expert compact tensors");
            }

            // Workspace estimation
            dev_plan.workspace_bytes = WorkspaceMemoryEstimator::estimate(
                profile,
                WorkspaceMemoryGeometry{
                    .device = cfg.device,
                    .device_compute_units = cfg.device_compute_units,
                    .batch_size = cfg.batch_size,
                    .resident_graph_rows = activation_seq,
                    .max_context_rows = max_seq,
                    .local_d_ff = local_d_ff,
                    .local_query_heads = local_n_heads,
                    .first_layer = cfg.first_layer,
                    .last_layer = last_layer,
                    .total_shards = cfg.total_shards,
                    .apportioned_routed_experts =
                        cfg.weight_residency.selectsRoutedExperts(),
                    .mtp_target_query_rows =
                        cfg.mtp_enabled
                            ? cfg.mtp_target_query_rows
                            : 0,
                });
        }
        else
        {
            const int participant_graph_rows =
                cfg.serial_routed_expert_compact_rows > 0
                    ? cfg.serial_routed_expert_compact_rows
                    : activation_seq;
            dev_plan.activation_bytes = routedParticipantActivationBytes(
                profile,
                cfg,
                cfg.first_layer,
                last_layer,
                activation_seq);
            dev_plan.workspace_bytes =
                WorkspaceMemoryEstimator::estimateRoutedExpertParticipant(
                    profile,
                    WorkspaceMemoryGeometry{
                        .device = cfg.device,
                        .device_compute_units = cfg.device_compute_units,
                        .batch_size = cfg.batch_size,
                        .resident_graph_rows = participant_graph_rows,
                        .max_context_rows = max_seq,
                        .local_d_ff = local_d_ff,
                        .local_query_heads = local_n_heads,
                        .first_layer = cfg.first_layer,
                        .last_layer = last_layer,
                        .total_shards = cfg.total_shards,
                        .apportioned_routed_experts = true,
                    });
        }

        // Diagnostics
        if (!dev_plan.fits())
        {
            std::ostringstream msg;
            msg << cfg.device.to_string() << ": need "
                << formatMB(dev_plan.incremental_bytes() + dev_plan.headroom_bytes)
                << " of new allocation but only "
                << formatMB(dev_plan.device_free_bytes) << " available"
                << " (deficit: " << formatMB(dev_plan.deficit()) << ")";
            result.diagnostics.push_back(msg.str());
        }
        else if (dev_plan.remaining() < 256ULL * 1024 * 1024)
        {
            std::ostringstream msg;
            msg << cfg.device.to_string() << ": tight fit — only "
                << formatMB(dev_plan.remaining()) << " remaining after headroom";
            result.diagnostics.push_back(msg.str());
        }

        result.devices.push_back(std::move(dev_plan));
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
          << "KV Cache" << "State" << "Collect." << "Activ." << "Wkspace" << "Total"
          << "New" << "Avail." << "OK"
          << fort::endr;

    // Column alignments
    table.column(0).set_cell_text_align(fort::text_align::left);
    for (int c = 1; c <= 12; ++c)
    {
        table.column(c).set_cell_text_align(fort::text_align::right);
    }
    table.column(13).set_cell_text_align(fort::text_align::center);

    // Data rows
    for (const auto& d : devices)
    {
        table << d.device.to_string()
              << d.max_seq_len
              << d.activation_seq_len
              << formatMB(d.weight_bytes)
              << formatMB(d.retained_weight_bytes)
              << formatMB(d.kv_cache_bytes)
              << formatMB(d.persistent_state_bytes)
              << formatMB(d.collective_bytes)
              << formatMB(d.activation_bytes)
              << formatMB(d.workspace_bytes)
              << formatMB(d.total_bytes())
              << formatMB(d.incremental_bytes())
              << formatMB(d.device_free_bytes)
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
