/**
 * @file OrchestrationConfigDocument.cpp
 * @brief One strict read/write field inventory for reusable orchestration input.
 *
 * Field visitors use structured bindings as compile-time exhaustiveness checks:
 * adding a member to a configuration aggregate requires updating its document
 * contract. Reads and writes use the same member list, including explicitness
 * and optional policy state. This is setup-only value serialization, not a live
 * device-state mirror or a competing physical-memory ledger.
 */
#include "OrchestrationConfigDocument.h"

#include <nlohmann/json.hpp>
#include <array>
#include <cmath>
#include <concepts>
#include <limits>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace llaminar2
{
namespace
{
    using Json = nlohmann::json;

    /** @brief Report a structural input failure at its exact document location. */
    [[noreturn]] void invalid(const std::string &path, const std::string &reason)
    {
        throw std::invalid_argument("Orchestration configuration " + path + ": " + reason);
    }

    // Symbolic enum names make the document independent of enum integer ABI.
    // No unknown value is coerced to an automatic/default execution policy.
#define VALUE(name) std::pair{std::string_view{#name}, E::name}
#define ENUM(Type, ...) \
    /** @brief Complete legal symbolic values for this document enum. */ \
    constexpr auto enumValues(Type *) { using E = Type; return std::array{__VA_ARGS__}; }
    ENUM(DeviceType, VALUE(CPU), VALUE(CUDA), VALUE(ROCm), VALUE(Vulkan), VALUE(Metal))
    ENUM(OrchestrationPlanningMode, VALUE(InferFromPlacement), VALUE(Automatic), VALUE(Apply))
    ENUM(OrchestrationStrategy, VALUE(SingleDevice), VALUE(TensorParallel), VALUE(PipelineParallel), VALUE(ExpertOverlay))
    ENUM(AutomaticHostParticipation, VALUE(BestSubset), VALUE(AllDiscovered))
    ENUM(DeviceAssignmentMode, VALUE(AUTO), VALUE(LOCAL_GPU), VALUE(ROUND_ROBIN), VALUE(EXPLICIT))
    ENUM(TPScope, VALUE(AUTO), VALUE(RANK_LOCAL), VALUE(NODE_LOCAL), VALUE(GLOBAL), VALUE(HYBRID))
    ENUM(PPSplitMode, VALUE(EQUAL), VALUE(WEIGHTED), VALUE(MANUAL))
    ENUM(MPIProfile, VALUE(AUTO), VALUE(TUNED))
    ENUM(CollectiveBackendType, VALUE(AUTO), VALUE(NCCL), VALUE(RCCL), VALUE(HETEROGENEOUS), VALUE(UPI), VALUE(MPI), VALUE(HOST))
    ENUM(ExecutionDomainScope, VALUE(AUTO), VALUE(SINGLE), VALUE(RANK_LOCAL), VALUE(NODE_LOCAL), VALUE(GLOBAL))
    ENUM(FusedAttentionBackend, VALUE(JIT), VALUE(REFERENCE), VALUE(TILED), VALUE(Q16_INTEGER))
    ENUM(RoutedExpertComputePolicy, VALUE(Unspecified), VALUE(Replicated), VALUE(Apportioned), VALUE(TensorSharded))
    ENUM(RoutedExpertPhasePolicy, VALUE(Unspecified), VALUE(Uniform), VALUE(PrefillApportionedDecodeReplicated))
    ENUM(RoutedExpertAssignmentPolicy, VALUE(Unspecified), VALUE(StaticOwner), VALUE(LeastLoadedResident))
    ENUM(RoutedExpertOwnerOrder, VALUE(Ordinal), VALUE(Random))
    ENUM(MoEHotExpertCacheConfig::Kind, VALUE(Percent), VALUE(Count), VALUE(Off))
    ENUM(MoERebalanceRuntimeMode, VALUE(Off), VALUE(Observe), VALUE(Dynamic))
    ENUM(PrefixCacheStorageMode, VALUE(Disabled), VALUE(Ram), VALUE(Device), VALUE(Tiered))
    ENUM(PrefixCacheTerminalStateMode, VALUE(Off), VALUE(Auto), VALUE(Always))
    ENUM(PrefixCacheMoEPolicy, VALUE(Disabled), VALUE(PlacementFingerprint), VALUE(InvalidateOnRebalance))
    ENUM(MTPVerifyMode, VALUE(Greedy), VALUE(SpeculativeSampling))
    ENUM(MTPDepthPolicyMode, VALUE(Fixed), VALUE(Observe), VALUE(Dynamic))
    ENUM(MTPDepthPolicyBackend, VALUE(Any), VALUE(CPU), VALUE(CUDA), VALUE(ROCm))
    ENUM(MTPDepthPolicyModelClass, VALUE(Any), VALUE(Dense), VALUE(MoE))
    ENUM(MTPDepthDefaultsProfile, VALUE(Portable), VALUE(CUDARTX3090), VALUE(ROCmMI50))
    ENUM(MTPSidecarDensePolicy, VALUE(TensorParallel), VALUE(ReplicatedPerParticipant))
    ENUM(MTPTerminalHeadPolicy, VALUE(VocabularySharded), VALUE(MirroredFullVocabulary))
    ENUM(RoutedExpertPlacementTopology, VALUE(SingleDomain), VALUE(TieredOverlay))
    ENUM(RoutedExpertResidencyPolicy, VALUE(Disabled), VALUE(StaticById), VALUE(HistogramTieredCache), VALUE(ExplicitMasks), VALUE(RoutedTierRebalanced))
    ENUM(MoEOverlayAuthorityExecutionKind, VALUE(Unresolved), VALUE(HostResident), VALUE(DeviceResident))
    ENUM(MoEContinuationDensePolicyIntent, VALUE(Resolved), VALUE(Automatic), VALUE(Explicit))
    ENUM(MoEContinuationActivationLayout, VALUE(ReplicatedHidden), VALUE(RootOnlyHidden), VALUE(ShardedHiddenRequiresGather))
    ENUM(DenseParallelPolicy, VALUE(Replicated), VALUE(TensorParallel), VALUE(TensorParallelDecodeMirroredEmbedding), VALUE(PrefillTensorParallelDecodeReplicated))
#undef ENUM
#undef VALUE

    // The first field is separate to avoid a trailing comma in the structured
    // binding. Bindings check member count; named accesses make reordering safe.
#define COMMA(name) , name
#define VISIT(name) (void)name; visitor(#name, value.name);
#define RECORD(Type, first, rest) \
    /** @brief Visit this entire aggregate with the shared read/write archive. */ \
    template<class T, class Archive> requires std::same_as<std::remove_const_t<T>, Type> \
    void fields(T &value, Archive &visitor) \
    { \
        auto &[first rest(COMMA)] = value; \
        VISIT(first) rest(VISIT) \
    }

#define MEMBERS(X) X(numa_node) X(device_type) X(device_ordinal)
    RECORD(GlobalDeviceAddress, hostname, MEMBERS)
#undef MEMBERS
#define MEMBERS(X) X(count)
    RECORD(AutomaticBackendDeviceCount, backend, MEMBERS)
#undef MEMBERS
#define MEMBERS(X) X(only_strategies) X(prefer_backend) X(prefer_strategy) X(host_participation) X(workload) X(device_counts)
    RECORD(AutomaticOrchestrationOptions, only_backends, MEMBERS)
#undef MEMBERS
#define MEMBERS(X) X(devices) X(weights) X(backend) X(routed_compute_policy) \
    X(routed_phase_policy) X(routed_decode_assignment_policy) X(routed_prefill_assignment_policy) \
    X(scope) X(owner_rank) X(explicit_ranks)
    RECORD(DomainDefinition, name, MEMBERS)
#undef MEMBERS
#define MEMBERS(X) X(participants) X(weights) X(backend) X(scope) X(owner_rank) X(ranks) \
    X(routed_compute_policy) X(routed_phase_policy) X(routed_decode_assignment_policy) X(routed_prefill_assignment_policy)
    RECORD(ExecutionDomainDefinition, name, MEMBERS)
#undef MEMBERS
#define MEMBERS(X) X(domain_name) X(first_layer) X(last_layer)
    RECORD(PPStageDefinition, stage_id, MEMBERS)
#undef MEMBERS
#define MEMBERS(X) X(count) X(percent)
    RECORD(MoEHotExpertCacheConfig, kind, MEMBERS)
#undef MEMBERS
#define MEMBERS(X) X(overlay_segment_rows) X(least_loaded_min_routed_rows) X(llep_alpha_numerator) \
    X(llep_alpha_denominator) X(llep_lambda_numerator) X(llep_lambda_denominator) X(llep_enable_balanced_skip)
    RECORD(RoutedExpertPrefillRuntimeConfig, assignment_window_tokens, MEMBERS)
#undef MEMBERS
#define MEMBERS(X) X(window_size) X(max_window_size) X(window_growth_factor) \
    X(migration_payoff_horizon_tokens) X(migration_transfer_slots) X(migration_execution_streams) \
    X(migration_cycles_per_wave) X(dynamic_imbalance_threshold_per_mille) X(dynamic_min_improvement_per_mille) \
    X(dynamic_max_swaps_per_layer) X(dynamic_max_plan_entries_per_wave) X(dynamic_min_window_activations) \
    X(device_min_load_spread_improvement) X(device_min_load_spread_improvement_divisor) \
    X(device_min_wave_spread_improvement_per_payload_slot) X(device_min_foreign_rows_per_critical_path_payload_slot) \
    X(device_min_router_spread_improvement_per_payload_slot) X(device_max_post_wave_load_spread_per_mille) \
    X(device_maintenance_slack_tokens) X(device_min_maintenance_period_tokens) \
    X(device_initial_maintenance_period_tokens) X(release_raw_expert_weights)
    RECORD(MoERebalanceRuntimeConfig, mode, MEMBERS)
#undef MEMBERS
#define MEMBERS(X) X(storage_mode) X(block_size) X(ram_budget_bytes) X(device_budget_bytes) \
    X(disk_budget_bytes) X(disk_dir) X(terminal_state) X(moe_policy)
    RECORD(PrefixCacheRuntimeConfig, enabled, MEMBERS)
#undef MEMBERS
#define MEMBERS(X) X(backend) X(model_class) X(min_depth) X(max_depth) X(initial_depth) \
    X(window_size) X(min_samples) X(cooldown_steps) X(promote_consecutive_windows) \
    X(use_generated_policy) X(promote_full_accept_rate) X(demote_zero_accept_rate) X(demote_acceptance_rate)
    RECORD(MTPDepthPolicyConfig, mode, MEMBERS)
#undef MEMBERS
#define MEMBERS(X) X(draft_tokens) X(depth_defaults_profile) X(graph_capacity_draft_tokens) \
    X(max_request_batch) X(verify_mode) X(sidecar_dense_policy) X(terminal_head_policy) \
    X(require_terminal_hidden_for_full_hit) X(depth_policy)
    RECORD(MTPRuntimeConfig, enabled, MEMBERS)
#undef MEMBERS
#define MEMBERS(X) X(scope) X(backend) X(participants) X(world_ranks) X(owner_rank) \
    X(routed_compute_policy) X(routed_phase_policy) X(routed_decode_assignment_policy) \
    X(routed_prefill_assignment_policy) X(weights)
    RECORD(RoutedExpertDomain, name, MEMBERS)
#undef MEMBERS
#define MEMBERS(X) X(domain) X(priority) X(max_experts_per_layer) X(memory_budget_bytes) \
    X(fallback) X(resolved_live_experts_per_layer)
    RECORD(RoutedExpertTier, name, MEMBERS)
#undef MEMBERS
#define MEMBERS(X) X(logical_root_participant) X(dense_tp_enabled) X(dense_decode_replicated) \
    X(dense_decode_mirrored_embedding) X(dense_policy) X(hidden_layout) X(shared_expert_uses_dense_tp)
    RECORD(MoEContinuationDomainSpec, domain, MEMBERS)
#undef MEMBERS
#define MEMBERS(X) X(routed_expert_tier)
    RECORD(RoutedExpertLayerPlacement, layer, MEMBERS)
#undef MEMBERS
#define MEMBERS(X) X(expert_ids)
    RECORD(RoutedExpertInitialLayerOrder, layer, MEMBERS)
#undef MEMBERS
#define MEMBERS(X) X(topology) X(continuation_domain) X(base_model_domain) X(shared_expert_domain) \
    X(continuation_domain_spec) X(residency_policy) X(authority_execution) X(replica_cache_capacity) \
    X(owner_order) X(dense_domains) X(domains) X(routed_tiers) X(initial_layer_order_overrides) \
    X(placements) X(continuation_dense_policy_intent)
    RECORD(MoERoutedExpertPlacementPlan, enabled, MEMBERS)
#undef MEMBERS
#define MEMBERS(X) X(automatic_planning) X(execution_rank_selection) X(dry_run) X(explain_placement) X(show_topology) X(show_numa) \
    X(validate_only) X(device_mode) X(device_for_this_rank) X(device_for_this_rank_numa_explicit) \
    X(cpu_global_tp_all_local) X(device_map) X(device_map_numa_explicit) X(domain_definitions) \
    X(pp_stage_definitions) X(tp_degree) X(tp_scope) X(tp_devices) X(tp_weights) X(tp_local_degree) \
    X(tp_global_degree) X(pp_degree) X(pp_split) X(cpu_layers) X(cpu_layers_first) X(default_backend) \
    X(config_file_path) X(topology_string) X(topology_file_path) X(topology_tree) X(model_path) \
    X(max_seq_len) X(prefill_max_bucket_size) X(use_mmap) X(prompt) X(prompt_was_explicitly_provided) X(n_predict) X(batch_size) \
    X(n_threads) X(seed) X(temperature) X(top_k) X(top_p) X(deterministic) X(chat_mode) \
    X(single_shot_chat) X(system_prompt) X(chat_template_override) X(benchmark_mode) \
    X(benchmark_json_output_path) X(benchmark_prompt_file_path) X(benchmark_prompt_file_was_provided) \
    X(serve_mode) X(serve_port) X(serve_host) X(use_fused_attention) X(fused_attention_backend) \
    X(mpi_procs) X(hostfile) X(mpi_dry_run) X(mpi_verbose) X(mpi_no_bootstrap) X(mpi_oversubscribe) \
    X(mpi_profile) X(verbose_level) X(list_devices) X(show_help) X(max_gpu_memory_mb) \
    X(max_cpu_memory_mb) X(moe_shared_experts_gpu) X(moe_sparse_experts_cpu) X(routed_expert_compute_policy) \
    X(routed_expert_owner_order) X(moe_hot_expert_cache) X(moe_routed_prefill) X(moe_rebalance) \
    X(moe_routed_expert_plan) X(activation_precision) X(kv_cache_precision) X(tp_allreduce_precision_override) \
    X(prefix_cache) X(mtp) X(shard_weights) X(disable_weight_sharding) X(heterogeneous_mode) \
    X(cpu_compute_fraction) X(disable_gpu_tp) X(disable_cpu_tp) X(min_layers_per_domain)
    RECORD(OrchestrationConfig, planning_mode, MEMBERS)
#undef MEMBERS
#undef RECORD
#undef VISIT
#undef COMMA

    /** @brief Identify supported container shapes without implicit JSON coercion. */
    template<class T> struct Shape { static constexpr int kind = 0; };
    template<class T> struct Shape<std::optional<T>> { static constexpr int kind = 1; using Value = T; };
    template<class T> struct Shape<std::shared_ptr<T>> { static constexpr int kind = 2; using Value = T; };
    template<class T> struct Shape<std::vector<T>> { static constexpr int kind = 3; using Value = T; };
    template<class A, class B> struct Shape<std::pair<A, B>> { static constexpr int kind = 4; using First = A; using Second = B; };

    template<class T> Json encode(const T &value, const std::string &path);
    template<class T> T decode(const Json &json, const std::string &path);

    /** @brief Emit each field of an aggregate through the same typed recursion. */
    struct Writer
    {
        Json json = Json::object();
        std::string path;
        /** @brief Encode the named field; never consult current defaults. */
        template<class T> void operator()(const char *name, const T &value)
        {
            json[name] = encode(value, path + "." + name);
        }
    };

    /** @brief Require a complete exact-key object before publishing any value. */
    struct Reader
    {
        const Json &json;
        std::string path;
        std::set<std::string> consumed;
        /** @brief Decode one required field into private setup-only storage. */
        template<class T> void operator()(const char *name, T &value)
        {
            const auto field_path = path + "." + name;
            if (!json.is_object() || !json.contains(name)) invalid(field_path, "missing field");
            value = decode<T>(json.at(name), field_path);
            consumed.emplace(name);
        }
        /** @brief Reject typos or fields from another schema instead of ignoring them. */
        void finish() const
        {
            if (!json.is_object()) invalid(path, "expected object");
            for (const auto &[key, value] : json.items())
                if (!consumed.contains(key)) invalid(path + "." + key, "unknown field");
        }
    };

    /** @brief Recursively encode only types owned by this configuration contract. */
    template<class T> Json encode(const T &value, const std::string &path)
    {
        if constexpr (std::is_enum_v<T>)
        {
            for (const auto &[name, known] : enumValues(static_cast<T *>(nullptr)))
                if (known == value) return name;
            invalid(path, "invalid enum value");
        }
        else if constexpr (std::is_arithmetic_v<T>)
        {
            if constexpr (std::is_floating_point_v<T>)
                if (!std::isfinite(value)) invalid(path, "nonfinite number");
            return value;
        }
        else if constexpr (std::same_as<T, std::string>) return value;
        else if constexpr (std::same_as<T, ExecutionRankSelection>)
            return encode(value.discoveryRanks(), path);
        else if constexpr (std::same_as<T, OrchestrationPlanningWorkload>)
            return Json{{"prompt_tokens", value.promptTokens()}, {"generation_tokens", value.generationTokens()}};
        else if constexpr (Shape<T>::kind == 1 || Shape<T>::kind == 2)
            return value ? encode(*value, path) : Json(nullptr);
        else if constexpr (Shape<T>::kind == 3)
        {
            Json result = Json::array();
            for (size_t i = 0; i < value.size(); ++i)
                result.push_back(encode(value[i], path + "[" + std::to_string(i) + "]"));
            return result;
        }
        else if constexpr (Shape<T>::kind == 4)
            return Json::array({encode(value.first, path + "[0]"), encode(value.second, path + "[1]")});
        else if constexpr (std::same_as<T, MoEOverlayReplicaCacheCapacity>)
            return Json{{"requested", value.requested()}, {"admitted", value.admitted()}};
        else if constexpr (std::same_as<T, ParallelismTree>)
            invalid(path, "recursive topology-tree execution is unimplemented");
        else
        {
            Writer writer{.path = path};
            fields(value, writer);
            return std::move(writer.json);
        }
    }

    /** @brief Decode exact types, checking numeric range before any narrowing cast. */
    template<class T> T decode(const Json &json, const std::string &path)
    {
        if constexpr (std::is_enum_v<T>)
        {
            const auto name = decode<std::string>(json, path);
            for (const auto &[known, value] : enumValues(static_cast<T *>(nullptr)))
                if (known == name) return value;
            invalid(path, "unknown enum '" + name + "'");
        }
        else if constexpr (std::same_as<T, bool>)
        {
            if (!json.is_boolean()) invalid(path, "expected boolean");
            return json.get<bool>();
        }
        else if constexpr (std::is_integral_v<T>)
        {
            if (!json.is_number_integer()) invalid(path, "expected integer");
            if (json.is_number_unsigned())
            {
                const auto value = json.get<uint64_t>();
                if (!std::in_range<T>(value)) invalid(path, "integer out of range");
                return static_cast<T>(value);
            }
            const auto value = json.get<int64_t>();
            if (!std::in_range<T>(value)) invalid(path, "integer out of range");
            return static_cast<T>(value);
        }
        else if constexpr (std::is_floating_point_v<T>)
        {
            if (!json.is_number()) invalid(path, "expected number");
            const double value = json.get<double>();
            if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<T>::max())
                invalid(path, "number out of range");
            return static_cast<T>(value);
        }
        else if constexpr (std::same_as<T, std::string>)
        {
            if (!json.is_string()) invalid(path, "expected string");
            return json.get<std::string>();
        }
        else if constexpr (std::same_as<T, ExecutionRankSelection>)
            return ExecutionRankSelection(decode<std::vector<int>>(json, path));
        else if constexpr (std::same_as<T, OrchestrationPlanningWorkload>)
        {
            Reader reader{.json = json, .path = path};
            int prompt = 0, generation = 0;
            reader("prompt_tokens", prompt);
            reader("generation_tokens", generation);
            reader.finish();
            return OrchestrationPlanningWorkload(prompt, generation);
        }
        else if constexpr (Shape<T>::kind == 1)
        {
            if (json.is_null()) return std::nullopt;
            return decode<typename Shape<T>::Value>(json, path);
        }
        else if constexpr (Shape<T>::kind == 2)
        {
            if (json.is_null()) return nullptr;
            return std::make_shared<typename Shape<T>::Value>(decode<typename Shape<T>::Value>(json, path));
        }
        else if constexpr (Shape<T>::kind == 3)
        {
            if (!json.is_array()) invalid(path, "expected array");
            T result;
            result.reserve(json.size());
            for (size_t i = 0; i < json.size(); ++i)
                result.push_back(decode<typename Shape<T>::Value>(json[i], path + "[" + std::to_string(i) + "]"));
            return result;
        }
        else if constexpr (Shape<T>::kind == 4)
        {
            if (!json.is_array() || json.size() != 2) invalid(path, "expected two-element pair");
            return {decode<typename Shape<T>::First>(json[0], path + "[0]"),
                    decode<typename Shape<T>::Second>(json[1], path + "[1]")};
        }
        else if constexpr (std::same_as<T, MoEOverlayReplicaCacheCapacity>)
        {
            Reader reader{.json = json, .path = path};
            int requested = 0, admitted = 0;
            reader("requested", requested);
            reader("admitted", admitted);
            reader.finish();
            return MoEOverlayReplicaCacheCapacity(requested, admitted);
        }
        else if constexpr (std::same_as<T, ParallelismTree>)
            invalid(path, "recursive topology-tree execution is unimplemented");
        else
        {
            T result{};
            Reader reader{.json = json, .path = path};
            fields(result, reader);
            reader.finish();
            return result;
        }
    }
}

std::string serializeOrchestrationConfig(const OrchestrationConfig &config)
{
    // A tree string is just as unsupported as its parsed form. Do not emit a
    // document that quietly drops this execution intent on the receiving rank.
    if (!config.topology_string.empty() || !config.topology_file_path.empty())
        invalid("$.configuration.topology", "recursive topology-tree execution is unimplemented");
    return Json{{"kind", "llaminar.orchestration-config"}, {"schema_version", kOrchestrationConfigDocumentSchemaVersion},
                {"configuration", encode(config, "$.configuration")}}.dump(2) + "\n";
}

OrchestrationConfig deserializeOrchestrationConfig(std::string_view document)
{
    try
    {
        // JSON's usual last-key-wins behavior is unsuitable for hard filters or
        // placement. Reject duplicate keys at every object nesting level.
        std::vector<std::set<std::string>> object_keys;
        auto callback = [&](int, Json::parse_event_t event, Json &value)
        {
            if (event == Json::parse_event_t::object_start) object_keys.emplace_back();
            else if (event == Json::parse_event_t::object_end) object_keys.pop_back();
            else if (event == Json::parse_event_t::key &&
                     !object_keys.back().insert(value.get<std::string>()).second)
                invalid("$", "duplicate field '" + value.get<std::string>() + "'");
            return true;
        };
        const auto json = Json::parse(document.begin(), document.end(), callback);
        Reader reader{.json = json, .path = "$"};
        std::string kind;
        int version = 0;
        reader("kind", kind);
        reader("schema_version", version);
        if (kind != "llaminar.orchestration-config" || version != kOrchestrationConfigDocumentSchemaVersion)
            invalid("$", "unsupported document kind or schema version");
        OrchestrationConfig result;
        reader("configuration", result);
        reader.finish();
        if (!result.topology_string.empty() || !result.topology_file_path.empty())
            invalid("$.configuration.topology", "recursive topology-tree execution is unimplemented");
        return result;
    }
    catch (const Json::exception &error)
    {
        invalid("$", error.what());
    }
}
}
