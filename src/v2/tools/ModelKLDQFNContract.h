/**
 * @file ModelKLDQFNContract.h
 * @brief Checked diagnostic requests for the original QFN production runner.
 *
 * The versioned request carries the existing lossless production config and
 * original file descriptors. It is a measurement contract, never a placement,
 * capacity grant, runtime topology registry or numerical certificate. Actual
 * model/PMA/prepared owners are authenticated separately after RuntimeInit.
 */
#pragma once
#include "config/OrchestrationConfigDocument.h"
#include "config/GDNProjectionArithmeticPolicy.h"
#include "config/MoERouterArithmeticPolicy.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "execution/moe/MoERoutedExpertPlacementPlan.h"
#include "execution/local_execution/graph/IGraphExecutor.h"
#include "kernels/common/QuantizedActivationOperandPrecision.h"
#include "utils/Sha256.h"
#include "utils/PerfStatsCollector.h"
#include "utils/DebugEnv.h"
#include "utils/PrefillGraphBucketDefaults.h"
// The diagnostic tool and canonical parity deliberately share one pure path policy.
#include "../../../tests/v2/utils/ProductionParityEvidence.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <set>
#include <span>
#include <vector>
#include <sys/stat.h>
#include <stdexcept>

namespace llaminar2::accuracy
{
/** @brief Existing supported public-runner execution surfaces; no serving selector. */
enum class QFNObservationSurface { CPU, GPUCPUOverlay, CPUTailPipeline, HomogeneousLocalTP, GPUCPULocalTP, CPUExpertFollower };

/** @brief Reject invalid diagnostic input without device work or partial output. */
inline void requireQFN(bool condition, const char *message)
{
    if (!condition) throw std::invalid_argument(message);
}

/**
 * @brief Authenticate the one declared distributed Static CPU fragment observation surface.
 * @param config Original complete parser value, still revalidated by production admission.
 * @return False for historical single-rank requests; true only for an exact GPU authority plus CPU joint cohort.
 * @throws std::invalid_argument For unresolved, foreign, apportioned or active/migrating distributed intent.
 * @note This grants no source, weight, collective or native readiness. Production retains those authorities.
 */
inline bool qfnDistributedCPUJointIntent(const OrchestrationConfig& config) {
    if(config.mpi_procs<=1)return false;
    requireQFN(config.mpi_procs>=2 && config.mpi_procs<=8 && config.hostfile.empty() &&
        !config.execution_rank_selection && config.moe_routed_expert_plan &&
        config.moe_routed_expert_plan->usesExpertOverlayAuthority(),
        "QFN distributed observer requires one rank per CPU socket with rank-zero GPU continuation");
    const auto& plan=*config.moe_routed_expert_plan;
    requireQFN(plan.domains.size()==2 && plan.continuation_domain==plan.base_model_domain &&
        plan.continuation_domain==plan.shared_expert_domain && config.moe_rebalance.mode==MoERebalanceRuntimeMode::Off &&
        plan.residency_policy==RoutedExpertResidencyPolicy::StaticById && config.pp_stage_definitions.empty(),
        "QFN distributed observer supports only Static single-continuation CPU joint admission");
    const RoutedExpertDomain* source=nullptr;const RoutedExpertDomain* joint=nullptr;
    for(const auto& domain:plan.domains) {
        if(domain.name==plan.continuation_domain)source=&domain;else joint=&domain;
    }
    requireQFN(source && joint && source->participants.size()==1 && source->participants.front().isGPU() &&
        (source->scope==ExecutionDomainScope::SINGLE || source->scope==ExecutionDomainScope::RANK_LOCAL) &&
        source->owner_rank==0 && (source->world_ranks.empty() || source->world_ranks==std::vector<int>{0}),
        "QFN distributed observer lost its exact rank-zero GPU continuation");
    requireQFN(joint->scope==ExecutionDomainScope::NODE_LOCAL &&
        (joint->backend==CollectiveBackendType::UPI || joint->backend==CollectiveBackendType::MPI) &&
        joint->routed_compute_policy==RoutedExpertComputePolicy::TensorSharded &&
        joint->participants.size()==size_t(config.mpi_procs) && joint->world_ranks.size()==joint->participants.size() &&
        joint->participants.size()>=2 && joint->participants.size()<=8 &&
        joint->routed_phase_policy==RoutedExpertPhasePolicy::Uniform &&
        joint->routed_decode_assignment_policy==RoutedExpertAssignmentPolicy::StaticOwner &&
        joint->routed_prefill_assignment_policy==RoutedExpertAssignmentPolicy::StaticOwner,
        "QFN distributed observer requires the original uniform CPU tensor-shard cohort");
    std::set<int> nodes,ranks;
    for(size_t i=0;i<joint->participants.size();++i) {
        const auto& endpoint=joint->participants[i];
        requireQFN(endpoint.isCPU() && endpoint.device_ordinal==0 && endpoint.numa_node>=0 &&
            nodes.insert(endpoint.numa_node).second && joint->world_ranks[i]>=0 &&
            joint->world_ranks[i]<config.mpi_procs && ranks.insert(joint->world_ranks[i]).second,
            "QFN distributed observer has missing/duplicate/foreign CPU rank or NUMA identity");
    }
    return true;
}

/**
 * @brief Project canonical placement/math intent from the actual public parsed config.
 * @param config Original lossless parser value; no model or device is initialized.
 * @return Existing replay intent, including the production-resolved serving ladder.
 * @note Prompt, prefix, context, workers, observation budget and inactive paired
 * source declaration remain explicit diagnostic controls, not placement fields.
 * This is the single projection consumed by replay and paired corpus admission.
 */
inline nlohmann::json qfnCanonicalConfigurationIntent(const OrchestrationConfig &config)
{
    const auto document = nlohmann::json::parse(serializeOrchestrationConfig(config)).at("configuration");
    nlohmann::json intent = nlohmann::json::object();
    for (const char *field : {"planning_mode", "automatic_planning", "execution_rank_selection",
                 "device_mode", "device_for_this_rank", "device_for_this_rank_numa_explicit", "device_map",
                 "device_map_numa_explicit", "cpu_global_tp_all_local", "topology_string", "topology_file_path", "topology_tree",
                 "domain_definitions", "pp_stage_definitions", "tp_degree", "tp_scope", "tp_devices", "tp_weights",
                 "tp_local_degree", "tp_global_degree", "pp_degree", "pp_split", "cpu_layers", "cpu_layers_first",
                 "default_backend", "mpi_procs", "hostfile", "max_gpu_memory_mb", "max_cpu_memory_mb",
                 "moe_shared_experts_gpu", "moe_sparse_experts_cpu", "routed_expert_compute_policy",
                 "routed_expert_owner_order", "moe_hot_expert_cache", "moe_routed_prefill", "moe_rebalance",
                 "moe_routed_expert_plan", "activation_precision", "kv_cache_precision", "tp_allreduce_precision_override",
                 "ngram_table_policy", "mtp_activation_policy", "mtp", "shard_weights", "disable_weight_sharding",
                 "heterogeneous_mode", "cpu_compute_fraction", "disable_gpu_tp", "disable_cpu_tp", "min_layers_per_domain",
                 "use_fused_attention", "fused_attention_backend"})
        intent[field] = document.at(field);
    const int maximum = config.prefill_max_bucket_size.value_or(kDefaultPrefillGraphMaxBucketSize);
    intent["prefill_max_bucket_size"] = maximum;
    intent["prefill_graph_buckets"] = prefillGraphBucketSizes(maximum);
    return intent;
}

/** @return The six actual source-owned arithmetic selectors; no caller-authored path proof. */
inline nlohmann::json qfnActualArithmeticIntent()
{
    const auto &gemm = debugEnv().gemm;
    return {
        {"gdn_projection", std::string(gdnProjectionArithmeticName(gemm.gdn_projection_arithmetic))},
        {"moe_router", std::string(moeRouterArithmeticName(resolveMoERouterArithmetic(gemm.moe_router_arithmetic, "qwen4exp")))},
        {"hc_up", std::string(quantizedActivationOperandPrecisionName(gemm.qfn_hc_up_activation))},
        {"gdn_output", std::string(quantizedActivationOperandPrecisionName(gemm.qfn_gdn_output_activation))},
        {"fa_output", std::string(quantizedActivationOperandPrecisionName(gemm.qfn_fa_output_activation))},
        {"gdn_qkv", std::string(quantizedActivationOperandPrecisionName(gemm.qfn_gdn_qkv_activation))}};
}

/**
 * @brief Reject a foreign placement/math projection before diagnostic admission.
 * @param canonical Actual original canonical parsed configuration.
 * @param observed Explicit diagnostic configuration, retaining its own full identity.
 * @param arithmetic Frozen six-field corpus arithmetic to compare with actual selectors.
 * @throws std::invalid_argument For the first mismatched intent field or selector.
 */
inline void qfnRequireCanonicalIntent(const OrchestrationConfig &canonical,
    const OrchestrationConfig &observed, const nlohmann::json &arithmetic)
{
    const auto actual = qfnCanonicalConfigurationIntent(canonical);
    const auto expected = qfnCanonicalConfigurationIntent(observed);
    // Compare the resolved cap before its derived ladder, but report the
    // authored values. This preserves the distinction between a default/null
    // declaration and an explicit override in a rejected observation.
    if (actual.at("prefill_max_bucket_size") != expected.at("prefill_max_bucket_size"))
    {
        const auto authored_cap = [](const std::optional<int> &cap) -> nlohmann::json
        {
            return cap ? nlohmann::json(*cap) : nlohmann::json(nullptr);
        };
        const std::string message =
            "Frozen history differs from original canonical placement/math intent field=prefill_max_bucket_size canonical=" +
            authored_cap(canonical.prefill_max_bucket_size).dump() + " observer=" +
            authored_cap(observed.prefill_max_bucket_size).dump();
        requireQFN(false, message.c_str());
    }
    for (const auto &[field, value] : actual.items())
    {
        const std::string message = "Frozen history differs from original canonical placement/math intent field=" +
            field + " canonical=" + value.dump() + " observer=" + expected.at(field).dump();
        requireQFN(value == expected.at(field), message.c_str());
    }
    const auto actual_arithmetic = qfnActualArithmeticIntent();
    const std::string message = "Frozen history arithmetic differs from actual canonical selectors field=arithmetic canonical=" +
        actual_arithmetic.dump() + " observer=" + arithmetic.dump();
    requireQFN(actual_arithmetic == arithmetic, message.c_str());
}

/**
 * @brief Scope one QFN observer call's passive graph-execution phase metadata.
 *
 * This follows BenchmarkRunner's existing phase attribution, not model state or
 * execution policy. RankOrchestrator propagates the current phase into worker
 * submissions. The observer scopes only the actual public runner call, so no
 * completion record is invented for publication-only, failed or empty work.
 */
class ScopedQFNObservationPhase final
{
public:
    /**
     * @brief Attribute the enclosed real prefill or forced-decode call.
     * @param phase Exact observed phase, PREFILL or DECODE; COMBINED is ambiguous.
     * @throws std::invalid_argument For COMBINED or an unknown enum, before mutation.
     */
    explicit ScopedQFNObservationPhase(ExecutionPhase phase)
        : previous_(GraphExecutorStats::currentPhase())
    {
        requireQFN(phase == ExecutionPhase::PREFILL || phase == ExecutionPhase::DECODE,
                   "QFN observer requires explicit prefill/decode phase metadata");
        GraphExecutorStats::setCurrentPhase(phase);
    }
    /** @brief Restore the caller's original phase on ordinary and exception exits. */
    ~ScopedQFNObservationPhase() noexcept
    {
        GraphExecutorStats::setCurrentPhase(previous_);
    }
    ScopedQFNObservationPhase(const ScopedQFNObservationPhase&) = delete;
    ScopedQFNObservationPhase& operator=(const ScopedQFNObservationPhase&) = delete;
    ScopedQFNObservationPhase(ScopedQFNObservationPhase&&) = delete;
    ScopedQFNObservationPhase& operator=(ScopedQFNObservationPhase&&) = delete;
private:
    ExecutionPhase previous_; ///< Exact thread-local observation value, not a live model shadow.
};

/**
 * @brief Require the retained rank plan to belong to the actual original MPI participant.
 * @param plan Original runner plan; never copied or reconstructed for observation.
 * @param actual_rank Rank from the application's existing MPI context.
 * @throws std::invalid_argument If the observer borrowed another rank's physical plan.
 */
inline void requireQFNOriginalRank(const RankExecutionPlan& plan, int actual_rank)
{
    requireQFN(actual_rank >= 0 && plan.rank == actual_rank,
               "QFN observer rank plan does not belong to its original MPI rank");
}

/**
 * @brief Authenticate a physical address against its original rank plan's canonical host.
 * @param plan Original inventory-resolved rank plan, authenticated against the MPI rank by the caller.
 * @param address Original physical address declared by that plan or its prepared expert domain.
 * @return True for local aliases or the exact retained canonical hostname; no host discovery occurs.
 */
inline bool qfnAddressOwnedByRank(const RankExecutionPlan& plan, const GlobalDeviceAddress& address)
{
    return address.isLocal() || (!plan.hostname.empty() && address.hostname == plan.hostname);
}

/** @brief Passive original source dimensions and participant logit capability; no readiness authority. */
struct QFNSourceGeometry {
    std::string runner_architecture;
    std::string model_architecture;
    int main_layers;
    int source_vocabulary;
    int experts;
    int runner_vocabulary;
};

/**
 * @brief Check original source geometry separately from participant logit capability.
 * @param actual Passive values read from the original loader/model and real runner.
 * @param config Original parsed serving configuration.
 * @param plan Original rank plan, already authenticated against the MPI rank.
 * @param joint_follower True only for the worker-loop observation callback.
 * @throws std::invalid_argument With exact observed field values on disagreement.
 * @note Expert-only followers expose zero logits; this never changes source geometry.
 */
inline void requireQFNSourceGeometry(const QFNSourceGeometry& actual,
    const OrchestrationConfig& config, const RankExecutionPlan& plan, bool joint_follower)
{
    nlohmann::json compared={{"rank",plan.rank},{"primary_hostname",plan.primary_device.hostname},
        {"primary_device_ordinal",plan.primary_device.device_ordinal},{"primary_numa",plan.primary_device.numa_node}};
    if(joint_follower && config.moe_routed_expert_plan) {
        compared["original_cohort_endpoints"]=nlohmann::json::array();
        for(const auto& domain:config.moe_routed_expert_plan->domains) {
            if(domain.name==config.moe_routed_expert_plan->continuation_domain)continue;
            for(size_t i=0;i<domain.participants.size();++i) {
                const auto& endpoint=domain.participants[i];
                compared["original_cohort_endpoints"].push_back({{"domain",domain.name},
                    {"world_rank",i<domain.world_ranks.size()?domain.world_ranks[i]:-1},
                    {"hostname",endpoint.hostname},{"device_ordinal",endpoint.device_ordinal},{"numa",endpoint.numa_node}});
            }
        }
    }
    const auto check=[&](bool valid,const char* field) {
        if(!valid) throw std::invalid_argument(std::string("QFN observer actual original source geometry differs: ")+field+
            "; observed="+nlohmann::json({{"runner_architecture",actual.runner_architecture},
                {"model_architecture",actual.model_architecture},{"main_layers",actual.main_layers},
                {"source_vocabulary",actual.source_vocabulary},{"experts",actual.experts},
                {"runner_vocabulary",actual.runner_vocabulary}}).dump()+"; compared="+compared.dump());
    };
    check(actual.runner_architecture=="qwen4exp","runner_architecture");
    check(actual.model_architecture=="qwen4exp","model_architecture");
    check(actual.main_layers==48,"main_layers");
    check(actual.source_vocabulary==248320,"source_vocabulary");
    check(actual.experts==512,"experts");
    if(joint_follower) {
        check(qfnDistributedCPUJointIntent(config) && plan.rank>0 && plan.rank<config.mpi_procs &&
            plan.primary_device.isCPU() && qfnAddressOwnedByRank(plan,plan.primary_device),"follower_rank");
        bool member=false;
        for(const auto& domain:config.moe_routed_expert_plan->domains) {
            if(domain.name==config.moe_routed_expert_plan->continuation_domain) continue;
            for(size_t i=0;i<domain.participants.size();++i) {
                const auto& endpoint=domain.participants[i];
                member|=domain.world_ranks[i]==plan.rank && qfnAddressOwnedByRank(plan,endpoint) &&
                    endpoint.isCPU() && endpoint.device_ordinal==0 && endpoint.numa_node>=0;
            }
        }
        check(member,"original_cohort_rank");
    }
    check(actual.runner_vocabulary== (joint_follower?0:248320),"runner_vocabulary");
}

/**
 * @brief Classify the actual already-admitted rank plan, without constructing one.
 * @param plan Existing production runner's original rank-local plan.
 * @return Physical host-logit/forced-row shape only; actual prepared expert ownership refines LocalTP before graph certification.
 * @throws std::invalid_argument For cross-rank or unsupported physical execution.
 */
inline QFNObservationSurface qfnObservationSurface(const RankExecutionPlan &plan)
{
    requireQFN(!plan.usesPipelineParallel() && !plan.usesGlobalTP(), "QFN observer requires exactly one rank-local execution authority");
    if (plan.usesLocalPP())
    {
        bool cuda = false, rocm = false, cpu = false;
        std::set<DeviceId> devices;
        for (const auto &address : plan.local_pp_devices)
        {
            requireQFN(qfnAddressOwnedByRank(plan, address) && devices.insert(address.toLocalDeviceId()).second,
                       "QFN observer pipeline has foreign or duplicate physical devices");
            cuda |= address.device_type == DeviceType::CUDA;
            rocm |= address.device_type == DeviceType::ROCm;
            cpu |= address.device_type == DeviceType::CPU;
        }
        requireQFN(cpu, "QFN observer GPU-only PP unsupported: pipeline ordinary resident-condition control is not implemented");
        requireQFN(cuda && rocm && plan.local_pp_devices.size() == 3 &&
                   plan.local_pp_devices.back().device_type == DeviceType::CPU,
                   "QFN observer supports only the declared CUDA/ROCm/CPU-tail pipeline");
        return QFNObservationSurface::CPUTailPipeline;
    }
    if (plan.usesLocalTP())
    {
        requireQFN(plan.local_tp_devices.size()==2 || plan.local_tp_devices.size()==4 || plan.local_tp_devices.size()==8,
                   "QFN observer requires an explicit homogeneous GPU LocalTP degree 2/4/8");
        const auto backend=plan.local_tp_devices.front().device_type;
        requireQFN(backend==DeviceType::CUDA || backend==DeviceType::ROCm,"QFN LocalTP requires actual GPU compute participants");
        std::set<DeviceId> devices;
        for (const auto &address : plan.local_tp_devices)
            requireQFN(qfnAddressOwnedByRank(plan, address) && address.device_type == backend &&
                       devices.insert(address.toLocalDeviceId()).second,
                       "QFN observer LocalTP has foreign, mixed or duplicate physical devices");
        requireQFN(qfnAddressOwnedByRank(plan, plan.primary_device) &&
            std::find(plan.local_tp_devices.begin(),plan.local_tp_devices.end(),plan.primary_device)!=plan.local_tp_devices.end(),
            "QFN LocalTP primary is outside its original physical cohort");
        return QFNObservationSurface::HomogeneousLocalTP;
    }
    requireQFN(qfnAddressOwnedByRank(plan, plan.primary_device), "QFN observer requires local physical ownership");
    if (plan.primary_device.device_type == DeviceType::CPU) return QFNObservationSurface::CPU;
    requireQFN(plan.primary_device.device_type == DeviceType::CUDA || plan.primary_device.device_type == DeviceType::ROCm,
               "QFN observer unsupported primary backend");
    return QFNObservationSurface::GPUCPUOverlay;
}

/**
 * @brief Classify an expert-only follower against its original prepared CPU domain.
 * @param config Actual parsed distributed Static source intent.
 * @param plan Actual runner plan, already checked against the original MPI rank.
 * @param original_plan Retained source preparation rank plan from the same owner.
 * @param prepared Original retained routed source placement; never reconstructed.
 * @return Expert-only surface with no full-model/logit capability.
 * @throws std::invalid_argument For foreign rank, domain, source or unsupported topology.
 * @note This passive geometry check grants no readiness. The caller must still
 * authenticate the actual model/Store/PMA and every original prepared cohort lease.
 */
inline QFNObservationSurface qfnCPUJointFollowerSurface(const OrchestrationConfig &config,
    const RankExecutionPlan &plan, const RankExecutionPlan &original_plan,
    const MoERoutedExpertPlacementPlan &prepared)
{
    const auto reject = [&](bool valid, const char *field) {
        if (!valid) throw std::invalid_argument(std::string("QFN original CPU follower domain differs: ")+field+
            "; rank="+std::to_string(plan.rank)+" prepared_rank="+std::to_string(original_plan.rank)+
            " domain="+std::to_string(plan.global_tp_domain_id.value_or(-1))+
            " member="+std::to_string(plan.global_tp_rank_in_domain)+
            " population="+std::to_string(plan.global_tp_domain_size));
    };
    reject(qfnDistributedCPUJointIntent(config), "distributed_intent");
    reject(plan.rank>0 && plan.rank<config.mpi_procs && original_plan.rank==plan.rank,
           "original_rank");
    reject(!plan.usesPipelineParallel() && !plan.usesLocalPP() && !plan.usesLocalTP() &&
        plan.local_pp_devices.empty() &&
        std::all_of(plan.local_tp_devices.begin(),plan.local_tp_devices.end(),[&](const auto &address) {
            return address.isCPU() && qfnAddressOwnedByRank(plan,address);
        }) && plan.primary_device.isCPU() && qfnAddressOwnedByRank(plan,plan.primary_device), "expert_only_topology");
    reject(plan.usesGlobalTP() && plan.global_tp_domain_id==original_plan.global_tp_domain_id &&
        plan.global_tp_rank_in_domain==original_plan.global_tp_rank_in_domain &&
        plan.global_tp_domain_size==original_plan.global_tp_domain_size, "retained_global_domain");
    const auto &intent=*config.moe_routed_expert_plan;
    const auto requested=std::find_if(intent.domains.begin(),intent.domains.end(),[&](const auto &domain) {
        return domain.name!=intent.continuation_domain;
    });
    const auto source=std::find_if(prepared.domains.begin(),prepared.domains.end(),[&](const auto &domain) {
        return requested!=intent.domains.end() && domain.name==requested->name;
    });
    reject(source!=prepared.domains.end() && requested!=intent.domains.end(), "original_source_domain");
    reject(prepared.usesExpertOverlayAuthority() && prepared.domains.size()==intent.domains.size() &&
        prepared.continuation_domain==intent.continuation_domain &&
        prepared.residency_policy==intent.residency_policy &&
        source->participants.size()==requested->participants.size() && source->world_ranks==requested->world_ranks &&
        source->backend==requested->backend && source->scope==requested->scope &&
        source->routed_compute_policy==RoutedExpertComputePolicy::TensorSharded &&
        source->routed_phase_policy==requested->routed_phase_policy &&
        source->routed_decode_assignment_policy==requested->routed_decode_assignment_policy &&
        source->routed_prefill_assignment_policy==requested->routed_prefill_assignment_policy,
        "original_source_cohort");
    const auto participation=std::find_if(plan.my_domains.begin(),plan.my_domains.end(),[&](const auto &domain) {
        return domain.domain_id==*plan.global_tp_domain_id;
    });
    const auto retained=std::find_if(original_plan.my_domains.begin(),original_plan.my_domains.end(),[&](const auto &domain) {
        return domain.domain_id==*plan.global_tp_domain_id;
    });
    reject(participation!=plan.my_domains.end() && retained!=original_plan.my_domains.end() &&
        plan.my_domains.size()==1 && original_plan.my_domains.size()==1, "original_participation");
    reject(participation->domain_name==source->name && participation->backend==source->backend &&
        participation->devices==retained->devices && participation->domain_name==retained->domain_name &&
        participation->backend==retained->backend && participation->my_index_in_domain==retained->my_index_in_domain &&
        participation->my_index_in_domain==plan.global_tp_rank_in_domain &&
        plan.global_tp_domain_size==int(source->participants.size()) &&
        participation->devices.size()==source->participants.size(), "original_domain_geometry");
    auto ranks=source->world_ranks;std::sort(ranks.begin(),ranks.end());
    reject(plan.global_tp_rank_in_domain>=0 && plan.global_tp_rank_in_domain<int(ranks.size()) &&
        ranks[size_t(plan.global_tp_rank_in_domain)]==plan.rank, "original_member_rank");
    for(size_t i=0;i<source->participants.size();++i) {
        const auto &actual=participation->devices[i];const auto &expected=source->participants[i];
        const auto &authored=requested->participants[i];
        reject(expected.isCPU() && expected.device_ordinal==authored.device_ordinal &&
            expected.numa_node==authored.numa_node && qfnAddressOwnedByRank(plan,authored), "authored_member_endpoint");
        reject(actual.isCPU() && actual.device_ordinal==expected.device_ordinal && actual.numa_node==expected.numa_node &&
            qfnAddressOwnedByRank(plan,actual) && qfnAddressOwnedByRank(plan,expected), "original_member_endpoint");
    }
    reject(std::any_of(prepared.routed_tiers.begin(),prepared.routed_tiers.end(),[&](const auto &tier) {
        return tier.domain==source->name;
    }) && !prepared.placements.empty(), "prepared_source_membership");
    return QFNObservationSurface::CPUExpertFollower;
}

/**
 * @brief Select graph topology from actual nonempty prepared expert ownership, never unused declarations.
 * @param physical Original validated homogeneous TP compute devices.
 * @param prepared Original admitted and validated routed expert plan.
 * @pre Caller retains and authenticates the actual prepared/store/PMA owner before this projection.
 * @return Heterogeneous TP only when an actually selected expert tier owns CPU work.
 * @throws std::invalid_argument For invalid selected tier/domain or foreign expert GPU/address.
 */
inline QFNObservationSurface qfnPreparedLocalTPSurface(const RankExecutionPlan& physical,
    const MoERoutedExpertPlacementPlan& prepared)
{
    requireQFN(qfnObservationSurface(physical)==QFNObservationSurface::HomogeneousLocalTP,
        "QFN prepared TP classification requires original homogeneous compute membership");
    bool cpu=false;
    std::set<std::string> selected;
    for(const auto& layer:prepared.placements)for(const auto tier:layer.routed_expert_tier){
        requireQFN(tier>=0 && std::size_t(tier)<prepared.routed_tiers.size(),"QFN prepared TP has an invalid selected expert tier");
        selected.insert(prepared.routed_tiers[tier].domain);
    }
    requireQFN(!selected.empty(),"QFN prepared TP has no original selected experts");
    for(const auto& name:selected){
        const auto domain=std::find_if(prepared.domains.begin(),prepared.domains.end(),[&](const auto& value){return value.name==name;});
        requireQFN(domain!=prepared.domains.end()&&!domain->participants.empty(),"QFN selected expert tier has no original domain");
        std::set<DeviceId> unique;
        for(const auto& address:domain->participants){
            requireQFN(qfnAddressOwnedByRank(physical, address)&&unique.insert(address.toLocalDeviceId()).second,"QFN expert domain has foreign or duplicate ownership");
            if(address.device_type==DeviceType::CPU){cpu=true;continue;}
            requireQFN(std::any_of(physical.local_tp_devices.begin(),physical.local_tp_devices.end(),[&](const auto& member){
                    return member.toLocalDeviceId()==address.toLocalDeviceId();
                }),
                "QFN expert GPU is outside its original compute cohort");
        }
    }
    return cpu?QFNObservationSurface::GPUCPULocalTP:QFNObservationSurface::HomogeneousLocalTP;
}

/**
 * @brief Hash canonical small metadata with the existing SHA256 implementation.
 * @param value Metadata serialized with sorted object keys and escaped Unicode.
 * @return Lowercase SHA256, matching the shared Python metadata convention.
 * @throws std::invalid_argument If the existing hash implementation fails.
 */
inline std::string qfnMetadataDigest(const nlohmann::json &value)
{
    const auto digest = sha256BytesHex(value.dump(-1, ' ', true));
    requireQFN(bool(digest), "QFN metadata digest failed");
    return *digest;
}

/**
 * @brief Apply the existing canonical graph evidence policy to actual QFN owners.
 * @param records Passive original forward_graph records from the current runner.
 * @param surface Actual source/prepared-owner-authenticated execution surface.
 * @param gpu_devices Exact actual GPU addresses retained by that runner.
 * @throws std::invalid_argument For incomplete replay or a topology-policy mismatch.
 *
 * The caller must first authenticate the CPU expert domain or exact CPU-tail
 * pipeline that makes a heterogeneous contract legitimate. A requested label
 * alone cannot grant segmentation. Homogeneous GPU LocalTP retains per-device strict
 * canonical capture/replay requirements. CPU and CPU-tail surfaces additionally
 * require successful nonempty runtime graph completions in both observed phases,
 * never optional profiling totals or cache-hit/submission counters. Host records
 * name the existing canonical CPU device string; original physical/NUMA owners
 * remain authenticated separately. No policy or counters are synthesized here.
 */
inline void qfnRequireGraphPathRecords(const std::vector<PerfStatRecord> &records,
                                      QFNObservationSurface surface,
                                      const std::set<std::string> &gpu_devices)
{
    using namespace llaminar2::test::parity;
    if (surface == QFNObservationSurface::CPU || surface == QFNObservationSurface::CPUTailPipeline)
    {
        bool prefill = false, decode = false;
        for (const auto &record : records)
        {
            if (record.domain != "forward_graph" || record.name != "host_graph_completed_calls" ||
                record.device != DeviceId::cpu().toString() || record.kind != PerfStatRecord::Kind::Counter ||
                !record.count || !std::isfinite(record.value) || record.value != static_cast<double>(record.count))
                continue;
            const auto source = record.tags.find("source");
            const auto authority = record.tags.find("authority");
            const auto stages = record.tags.find("stages");
            if (source == record.tags.end() || source->second != "device_graph_executor" ||
                authority == record.tags.end() || authority->second != "runtime_execution" || stages == record.tags.end())
                continue;
            std::uint64_t stageCount = 0;
            const auto parsed = std::from_chars(stages->second.data(),
                stages->second.data() + stages->second.size(), stageCount);
            if (parsed.ec != std::errc{} || parsed.ptr != stages->second.data() + stages->second.size() ||
                !stageCount || std::to_string(stageCount) != stages->second)
                continue;
            prefill |= record.phase == "prefill";
            decode |= record.phase == "decode";
        }
        requireQFN(prefill && decode,
                   "QFN observer lacks successful nonempty CPU runtime graph completion for prefill/decode");
    }
    const auto topology = surface == QFNObservationSurface::CPU
        ? ProductionParityExecutionTopology::CPUOnly
        : surface == QFNObservationSurface::HomogeneousLocalTP
            ? ProductionParityExecutionTopology::HomogeneousGPU
            : ProductionParityExecutionTopology::HeterogeneousAccelerator;
    const auto contract = resolveProductionParityGraphContract(topology, true, !gpu_devices.empty());
    const auto evidence = collectProductionParityEvidence(records, true, topology, contract, false, 0.0, 4500.0);
    requireQFN(certifyProductionParityGraphExecution(evidence) == ProductionParityGraphCertification::Certified,
               "QFN observer lacks the canonical topology-specific graph execution proof");
    for (const auto &device : gpu_devices)
    {
        std::vector<PerfStatRecord> local;
        bool prefill = false;
        for (const auto &record : records)
        {
            if (record.device != device || record.domain != "forward_graph" || !(record.value > 0.0)) continue;
            requireQFN(!record.name.starts_with("eager_"), "QFN observer found forbidden eager GPU execution");
            local.push_back(record);
            const auto phase = record.tags.find("capture_phase");
            if (record.name == "prefill_graph_phase" && phase != record.tags.end() &&
                (phase->second == "capture" || phase->second == "replay")) prefill = true;
        }
        requireQFN(prefill && (parityForwardGraphHasDecodePhase(local, "replay") ||
                              productionParityHasRetainedParentDecodeReplay(local)),
                   "QFN observer lacks actual per-device prefill/decode replay");
        if (surface == QFNObservationSurface::HomogeneousLocalTP)
        {
            const auto homogeneous = collectProductionParityEvidence(local, true, topology, contract, false, 0.0, 4500.0);
            requireQFN(certifyProductionParityGraphExecution(homogeneous) == ProductionParityGraphCertification::Certified,
                       "QFN homogeneous LocalTP requires each original GPU's unsegmented captured path");
        }
    }
}

/**
 * @brief Observe the exact original source stat fields used by the reference.
 * @param path Original regular source file, never a model payload read.
 * @return Canonical path/device/inode/extent/nanosecond timestamp descriptor.
 * @throws std::invalid_argument For missing, empty or nonregular source backing.
 */
inline nlohmann::json qfnOriginalSourceStat(const std::filesystem::path &path)
{
    const auto canonical = std::filesystem::canonical(path);
    struct stat metadata{};
    requireQFN(::stat(canonical.c_str(), &metadata) == 0 && S_ISREG(metadata.st_mode) && metadata.st_size > 0,
               "QFN original source stat is missing or not a nonempty regular file");
    return {{"path", canonical.string()}, {"device", static_cast<std::uint64_t>(metadata.st_dev)},
            {"inode", static_cast<std::uint64_t>(metadata.st_ino)}, {"bytes", metadata.st_size},
            {"mtime_ns", static_cast<std::int64_t>(metadata.st_mtim.tv_sec) * 1000000000LL + metadata.st_mtim.tv_nsec},
            {"ctime_ns", static_cast<std::int64_t>(metadata.st_ctim.tv_sec) * 1000000000LL + metadata.st_ctim.tv_nsec}};
}

/**
 * @brief Bind an actual source owner to its frozen reference metadata.
 * @param descriptor Original shard or inactive predictor descriptor.
 * @param path Actual ordered ModelContext source path.
 * @return Complete current stat descriptor after persistent reference metadata validation.
 * @throws std::invalid_argument For malformed or replaced original source metadata.
 *
 * object_identity remains the shared schema's JSON-serialized string. Comparing
 * path/inode/extent/timestamps prevents a same-size pre-admission replacement
 * from borrowing a reference source identity. Linux may renumber st_dev across
 * boots; its original value is observational, while the complete returned current
 * descriptor remains strict for before/after runtime checks. No payload is hashed.
 */
inline nlohmann::json qfnRequireOriginalSourceStat(const nlohmann::json &descriptor,
                                                const std::filesystem::path &path)
{
    const auto original = nlohmann::json::parse(descriptor.at("object_identity").get<std::string>());
    const auto actual = qfnOriginalSourceStat(path);
    requireQFN(original.is_object() && original.size() == actual.size() &&
               original.contains("path") && original.at("path").is_string(),
               "QFN original source stat descriptor is malformed");
    for (const char *field : {"device", "inode", "bytes", "mtime_ns", "ctime_ns"})
        requireQFN(original.contains(field) && original.at(field).is_number_integer(),
                   "QFN original source stat fields must retain integer precision");
    requireQFN(original.at("device") >= 0, "QFN original device number is invalid");
    for (const char *field : {"path", "inode", "bytes", "mtime_ns", "ctime_ns"})
        requireQFN(original.at(field) == actual.at(field),
                   "QFN original source stat identity changed since reference publication");
    requireQFN(descriptor.at("path") == actual.at("path") && descriptor.at("bytes") == actual.at("bytes"),
               "QFN original source stat identity changed since reference publication");
    return actual;
}

/**
 * @brief Validate only original ordinary LocalTP host-publication records.
 * @param records Passive sampling snapshot authored by the actual rank owner.
 * @param participants Exact original admitted TP compute extent.
 * @return Bounded cumulative successful one-row publication count.
 * @throws std::invalid_argument For changed operation, extent or counter schema.
 *
 * At most 4096 scored rows plus one 257-row reset are admitted by the request.
 * The larger 8192 bound also permits a before/after window without arithmetic
 * wrap. This helper neither manufactures a record nor probes live GPU state.
 */
inline std::uint64_t qfnTPHostPublicationCount(std::span<const PerfStatRecord> records,std::size_t participants)
{
    requireQFN(participants==2||participants==4||participants==8,"QFN host publication requires original TP degree 2/4/8");
    std::uint64_t count = 0;
    for (const auto &record : records)
    {
        if (record.name != "rank_resident_condition_host_logits_publications") continue;
        const auto tag = [&](const char *name) {
            const auto found = record.tags.find(name);
            return found == record.tags.end() ? std::string{} : found->second;
        };
        requireQFN(record.domain == "sampling" && record.kind == PerfStatRecord::Kind::Counter &&
                   record.device == "rank" && record.phase == "decode" &&
                   tag("operation") == "advanceOrdinaryMainConditionFromDeviceResidentLogicalState" &&
                   tag("participants") == std::to_string(participants) && tag("rows") == "1" &&
                   record.value == static_cast<double>(record.count) && record.count <= 8192 &&
                   count <= 8192 - record.count,
                   "QFN LocalTP host publication receipt has wrong original operation/extent");
        count += record.count;
    }
    return count;
}

/** @brief Immutable checked request; no caller can create a runtime owner through it. */
class QFNObserverRequest final
{
public:
    /**
     * @brief Authenticate the shared corpus-v2 request against the actual public config.
     * @param workload Versioned descriptor embedding one approved manifest and topology ID.
     * @param config Actual engine CLI parser result, before RuntimeInit normalization.
     * @return Checked metadata-only request; native/source/plan proof remains mandatory.
     * @throws std::invalid_argument For foreign schema/config/source/trajectory.
     */
    static QFNObserverRequest parse(const nlohmann::json &workload, const OrchestrationConfig &config)
    {
        requireQFN(workload.at("schema") == "llaminar.model-kld-workload.v2", "Unknown QFN workload schema");
        const auto &corpus = workload.at("corpus");
        requireQFN(corpus.at("schema") == "llaminar.model-kld-corpus.v2" && corpus.at("temperature") == 1 &&
                   corpus.at("metric") == "full_vocabulary_fp32_kl_reference_candidate_temperature1_v1" &&
                   corpus.at("vocabulary") == 248320, "Foreign QFN corpus metric/vocabulary");
        const auto &source = corpus.at("source");
        requireQFN(source.at("architecture") == "qwen4exp" && source.at("main_layers") == 48 &&
                   source.at("expert_count") == 512 && source.at("vocabulary") == 248320 &&
                   source.at("reference_arithmetic") == "cpu_fp32" && source.at("reference_execution") == "incremental_cached",
                   "QFN observer requires its original complete model/reference geometry");
        const auto &topologies = corpus.at("topologies");
        requireQFN(topologies.is_array() && !topologies.empty() && topologies.size() <= 128,
                   "Invalid source-owned topology inventory");
        nlohmann::json selected;
        std::set<std::string> topology_ids;
        for (const auto &topology : topologies)
        {
            requireQFN(topology_ids.insert(topology.at("id").get<std::string>()).second, "Duplicate topology ID");
            if (topology.at("id") == workload.at("topology_id")) selected = topology;
        }
        requireQFN(!selected.is_null(), "Missing selected original topology");
        const auto expected = deserializeOrchestrationConfig(selected.at("expected_config").dump());
        requireQFN(serializeOrchestrationConfig(expected) == serializeOrchestrationConfig(config) &&
                   selected.at("runtime_config_digest") == qfnMetadataDigest(selected.at("expected_config")),
                   "QFN observer CLI differs from its source-owned production configuration");
        const bool distributed_joint=qfnDistributedCPUJointIntent(config);
        requireQFN(config.planning_mode != OrchestrationPlanningMode::Automatic && (config.tp_scope == TPScope::AUTO || config.tp_scope == TPScope::RANK_LOCAL),
                   "QFN observer requires source-owned explicit rank-local placement");
        std::set<std::string> observed_domains;
        for (const auto &domain : config.domain_definitions)
        {
            if (!distributed_joint)
            {
                requireQFN((domain.scope == TPScope::AUTO || domain.scope == TPScope::RANK_LOCAL) &&
                           (!domain.owner_rank || *domain.owner_rank == 0) && domain.explicit_ranks.empty(),
                           "QFN observer rejects cross-rank domain intent");
                continue;
            }
            // This is the parser's original execution inventory, not another
            // placement authority. Require every authored domain to describe
            // exactly the already-approved source/cohort, including policies.
            const auto &approved = config.moe_routed_expert_plan->domains;
            const auto found = std::find_if(approved.begin(), approved.end(),
                [&](const auto &candidate) { return candidate.name == domain.name; });
            requireQFN(found != approved.end() && observed_domains.insert(domain.name).second,
                       "QFN observer has foreign or duplicate distributed domain intent");
            const auto expected_domain = DomainDefinition::fromExecutionDomainDefinition(
                found->toExecutionDomainDefinition());
            requireQFN(domain.devices == expected_domain.devices && domain.weights == expected_domain.weights &&
                       domain.backend == expected_domain.backend && domain.scope == expected_domain.scope &&
                       domain.owner_rank == expected_domain.owner_rank && domain.explicit_ranks == expected_domain.explicit_ranks &&
                       domain.routed_compute_policy == expected_domain.routed_compute_policy &&
                       domain.routed_phase_policy == expected_domain.routed_phase_policy &&
                       domain.routed_decode_assignment_policy == expected_domain.routed_decode_assignment_policy &&
                       domain.routed_prefill_assignment_policy == expected_domain.routed_prefill_assignment_policy,
                       "QFN observer distributed inventory differs from original source/cohort");
        }
        requireQFN(!distributed_joint || observed_domains.size() == config.moe_routed_expert_plan->domains.size(),
                   "QFN observer is missing an original distributed domain");
        requireQFN(config.mtp_activation_policy == MTPActivationPolicy::Disabled && !config.mtp.enabled &&
                   !config.prefix_cache.enabled && config.activation_precision == "fp32" &&
                   config.kv_cache_precision == "fp16" && config.prefill_max_bucket_size == 512 && config.max_seq_len > 0,
                   "QFN observer requires MTP/prefix off, FP32 activation, FP16 KV and M512");
        requireQFN(source.at("shards").is_array() && source.at("shards").size() == 4,
                   "QFN observer needs all four original source descriptors");
        std::set<std::string> source_names;
        const auto validate_descriptor = [&](const nlohmann::json &descriptor) {
            requireQFN(descriptor.at("name").is_string() && !descriptor.at("name").get<std::string>().empty() &&
                       descriptor.at("path").is_string() &&
                       !descriptor.at("path").get<std::string>().empty() &&
                       std::filesystem::path(descriptor.at("path").get<std::string>()).filename().string() == descriptor.at("name").get<std::string>() &&
                       source_names.insert(descriptor.at("name").get<std::string>()).second &&
                       descriptor.at("bytes").is_number_integer() && !descriptor.at("bytes").is_boolean() &&
                       descriptor.at("bytes").get<std::int64_t>() > 0 &&
                       descriptor.at("object_identity").is_string() && !descriptor.at("object_identity").get<std::string>().empty(),
                       "Missing or duplicate original QFN source metadata");
        };
        for (const auto &descriptor : source.at("shards")) validate_descriptor(descriptor);
        if (source.contains("shared_predictor")) validate_descriptor(source.at("shared_predictor"));
        requireQFN(source.at("tokenizer_revision").is_string() && !source.at("tokenizer_revision").get<std::string>().empty(),
                   "Missing original QFN tokenizer identity");
        const auto &arithmetic = selected.at("arithmetic");
        requireQFN(arithmetic.is_object() && arithmetic.size() == 6,
                   "QFN observer requires all six explicit arithmetic policies");
        // The corpus freezes the selected policy; it need not be the default.
        // Reuse production parsers so a paired diagnostic cannot silently
        // accept a typo, unresolved selection or a test-only arithmetic mode.
        const auto policy = [&](const char *name) {
            requireQFN(arithmetic.contains(name) && arithmetic.at(name).is_string(),
                       "QFN observer arithmetic policy must be an explicit string");
            return arithmetic.at(name).get<std::string>();
        };
        (void)parseGDNProjectionArithmetic(policy("gdn_projection"));
        (void)parseMoERouterArithmetic(policy("moe_router"));
        for (const char *name : {"hc_up", "gdn_output", "fa_output", "gdn_qkv"})
            (void)parseQuantizedActivationOperandPrecision(policy(name));
        if (!config.pp_stage_definitions.empty())
        {
            bool cpu = false;
            for (const auto &domain : config.domain_definitions)
                for (const auto &device : domain.devices) cpu |= device.device_type == DeviceType::CPU;
            requireQFN(cpu, "QFN observer GPU-only PP unsupported: pipeline ordinary resident-condition control is not implemented");
        }
        if (config.moe_routed_expert_plan)
            for (const auto &domain : config.moe_routed_expert_plan->domains)
                requireQFN(distributed_joint || ((domain.scope == ExecutionDomainScope::SINGLE || domain.scope == ExecutionDomainScope::RANK_LOCAL) &&
                           (domain.owner_rank == -1 || domain.owner_rank == 0) &&
                           std::all_of(domain.world_ranks.begin(), domain.world_ranks.end(),
                                       [](int rank) { return rank == 0; })),
                           "QFN observer rejects cross-rank routed authority");
        requireQFN(config.shared_predictor.has_value() == source.contains("shared_predictor"),
                   "QFN paired source declaration/descriptor disagree");
        if (config.shared_predictor)
            requireQFN(source.at("shared_predictor").at("path") == config.shared_predictor->path(),
                       "QFN paired source path differs from the original declaration");
        const auto &histories = corpus.at("histories");
        requireQFN(histories.is_array() && !histories.empty() && histories.size() <= 128, "Invalid bounded QFN history inventory");
        std::set<std::string> names;
        std::set<std::vector<std::int64_t>> original_prompts;
        std::size_t rows = 0;
        for (const auto &history : histories)
        {
            const auto name = history.at("id").get<std::string>();
            requireQFN(safeName(name) && names.insert(name).second, "Unsafe or duplicate QFN history ID");
            requireQFN(safeName(history.at("context_group").get<std::string>()), "Unsafe QFN context group");
            const auto &prompt = history.at("prompt_ids");
            const auto &continuation = history.at("continuation_ids");
            const auto &count = history.at("decode_rows");
            requireQFN(count.is_number_integer() && !count.is_boolean(), "QFN decode count must be integer");
            const int steps = count.get<int>();
            requireQFN(steps > 0 && steps <= 256 && prompt.is_array() && !prompt.empty() && continuation.is_array() &&
                       continuation.size() == std::size_t(steps + 1) &&
                       prompt.size() + std::size_t(steps) <= std::size_t(config.max_seq_len),
                       "QFN history needs complete prompt and D+1 original Pending teacher tokens");
            for (const auto *ids : {&prompt, &continuation})
                for (const auto &id : *ids)
                    requireQFN(id.is_number_integer() && !id.is_boolean() && id.get<std::int64_t>() >= 0 &&
                               id.get<std::int64_t>() < 248320, "Invalid original vocabulary ID");
            requireQFN(original_prompts.insert(prompt.get<std::vector<std::int64_t>>()).second,
                       "Duplicate original QFN prompt history");
            requireQFN(history.at("prompt").is_string() && !history.at("prompt").get<std::string>().empty() &&
                       history.at("reference_directory").is_string() &&
                       !history.at("reference_directory").get<std::string>().empty() &&
                       (history.at("reference_scope") == "full_stages" || history.at("reference_scope") == "final_logits_only"),
                       "QFN history requires original prompt/reference scope");
            rows += std::size_t(steps + 1);
        }
        requireQFN(rows <= 4096, "QFN observation row inventory is oversized");
        const auto &reference = corpus.at("reference");
        const auto report_digest = reference.at("report_digest").get<std::string>();
        requireQFN(reference.at("format") == "float32_le_row_major" && reference.at("report").is_string() &&
                   !reference.at("report").get<std::string>().empty() && report_digest.size() == 64 &&
                   report_digest.find_first_not_of("0123456789abcdef") == std::string::npos,
                   "QFN observer requires the approved complete reference metadata identity");
        const auto &reset = corpus.at("reset_control");
        requireQFN(names.contains(reset.at("history_id").get<std::string>()) &&
                   safeName(reset.at("replay_id").get<std::string>()) && !names.contains(reset.at("replay_id").get<std::string>()),
                   "QFN reset control must name an original and a distinct replay bank");
        return QFNObserverRequest(corpus, selected);
    }

    /** @return Exact immutable shared corpus metadata; no payload ownership. */
    const nlohmann::json &corpus() const noexcept { return corpus_; }
    /** @return Selected source-owned canonical topology, including complete expected config. */
    const nlohmann::json &topology() const noexcept { return topology_; }
    /**
     * @brief Keep names bounded and inside their original artifact owner.
     * @param name Original history, context or topology identifier.
     * @return True for a bounded ASCII filename component without traversal.
     */
    static bool safeName(const std::string &name)
    {
        return !name.empty() && name.size() <= 128 && name != "." && name != ".." &&
            name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") == std::string::npos;
    }
private:
    /** @brief Only checked complete metadata may construct this passive measurement request. */
    QFNObserverRequest(nlohmann::json corpus, nlohmann::json topology)
        : corpus_(std::move(corpus)), topology_(std::move(topology)) {}
    nlohmann::json corpus_;
    nlohmann::json topology_;
};
} // namespace llaminar2::accuracy
