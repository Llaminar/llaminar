/**
 * @file QueryKVGeometryObservation.h
 * @brief Passive setup observations of actual source, assignment, cache and PMA owners.
 *
 * These records copy immutable metadata at existing successful setup boundaries.
 * They neither admit memory nor retain a parallel assignment/byte ledger. A
 * construction observation is never represented as per-instance replay proof.
 */
#pragma once
#include "execution/config/KVHeadConsumerGeometry.h"
#include "execution/prefix_cache/PrefixPayloadLayout.h"
#include "kernels/IKVCache.h"
#include "kernels/common/OriginalOutputSlice.h"
#include "loaders/WeightPlan.h"
#include "planning/MemoryPlanner.h"
#include "planning/ModelMemoryProfile.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "utils/PerfStatsCollector.h"
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace llaminar2::query_kv_geometry
{
/** @brief Original fused wiring role; source names remain opaque outside artifact qualification. */
enum class ProjectionRole { Query = 0, Key = 1, Value = 2 };
/** @brief Name one actual wiring role. @param role Closed authored role. @return Checked label, never inferred from width. */
inline const char *roleName(ProjectionRole role)
{
    switch (role) { case ProjectionRole::Query:return "Q"; case ProjectionRole::Key:return "K";
        case ProjectionRole::Value:return "V"; }
    throw std::invalid_argument("unknown projection observation role");
}
/** @brief Project an existing certificate identity. @param admission Actual certificate or null. @return Process-local metadata, never a payload hash. */
inline std::string admissionIdentity(const PhysicalMemoryPlanAdmissionCertificate *admission)
{ return std::to_string(reinterpret_cast<std::uintptr_t>(admission)); }
/** @brief Describe a successful setup event. @param kind Exact publication boundary. @return Construction-only schema tags. */
inline PerfStatsCollector::Tags baseTags(const char *kind)
{
    return {{"schema","query_kv_geometry_v1"},{"kind",kind},
        {"construction_only","true"},{"per_instance_replay_proven","false"}};
}
/**
 * @brief Project the actual retained engine/source fact after cold stage validation.
 * @param fact Store-authenticated immutable owners and original output interval.
 * @param binding Actual current frozen publication authenticated by the stage.
 * @param role Actual fused wiring role, independent of canonical source naming.
 * @param stage_identity Existing full original-slice variant signature.
 * @param authority Same prepared store's physical owner, or null for device-free components.
 * @return Metadata only; unprepared/moved/foreign facts reject before publication.
 */
inline PerfStatsCollector::Tags projectionTags(const OriginalOutputSlice &fact,
    const WeightBinding &binding, ProjectionRole role, std::uint64_t stage_identity,
    const PhysicalMemoryAuthority *authority)
{
    fact.require(fact.retainedEngine(),fact.physicalColumns(),fact.reductionColumns());
    if (!binding.immutable || binding.binding_id != fact.bindingId() || !stage_identity ||
        !binding.tensor_owner || binding.tensor_owner.get()!=binding.tensor ||
        binding.slice.source_rows!=std::size_t(fact.policyColumns()) ||
        binding.slice.row_start!=std::size_t(fact.firstColumn()) ||
        binding.slice.row_count!=std::size_t(fact.physicalColumns()) ||
        binding.slice.source_cols!=std::size_t(fact.reductionColumns()) ||
        binding.slice.col_start!=0 || binding.slice.col_count!=std::size_t(fact.reductionColumns()))
        throw std::invalid_argument("projection observation lost its actual frozen binding");
    auto tags=baseTags("prepared_projection");
    tags.insert({{"role",roleName(role)},{"source_name",binding.identity.canonical_name},
        {"source_layer",std::to_string(binding.identity.layer)},
        {"binding_id",std::to_string(fact.bindingId())},{"model_id",std::to_string(fact.modelId())},
        {"physical_n",std::to_string(fact.physicalColumns())},{"policy_n",std::to_string(fact.policyColumns())},
        {"first_column",std::to_string(fact.firstColumn())},{"k",std::to_string(fact.reductionColumns())},
        {"source_codebook",std::to_string(fact.sourceIdentity().codebook_id)},
        {"source_superblock",fact.sourceIdentity().is_superblock?"true":"false"},
        {"stage_variant_identity",std::to_string(stage_identity)},
        {"admission_identity",admissionIdentity(authority?authority->admission().get():nullptr)}});
    return tags;
}
/** @copydoc projectionTags
 * @details Publishes the returned metadata into the existing setup domain; no native query or payload inspection. */
inline void recordProjection(const OriginalOutputSlice &fact,const WeightBinding &binding,
    ProjectionRole role,std::uint64_t stage_identity,const PhysicalMemoryAuthority *authority)
{
    PerfStatsCollector::addCounter("gpu_graph_inventory","query_kv_geometry",1.0,"cold_binding",
        fact.device().toString(),projectionTags(fact,binding,role,stage_identity,authority));
}
/**
 * @brief Describe an existing canonical assignment without distributing it again.
 * @param a Actual assignment retained by its physical owner.
 * @param degree Actual domain cardinality.
 * @param queries Original complete Q count.
 * @param kv Original complete KV count.
 * @param width Original head dimension.
 * @return Exact original coordinates from that assignment.
 */
inline PerfStatsCollector::Tags assignmentTags(const DeviceShardingAssignment &a,int degree,
    int queries,int kv,int width)
{
    if (degree<=1 || queries<=0 || kv<=0 || width<=0 || !a.isValid() ||
        a.local_rank<0 || a.local_rank>=degree || a.kv_head_count<=0)
        throw std::invalid_argument("query KV observation lacks an actual assignment");
    const auto cover=resolveQueryKVHeadCoverage({queries,kv},a.head_start,a.head_count);
    if(a.kv_head_start!=cover.first || a.kv_head_count!=cover.count)
        throw std::invalid_argument("observed assignment contradicts canonical original coverage");
    return {{"coverage","QueryInterval"},{"participant",std::to_string(a.local_rank)},
        {"degree",std::to_string(degree)},{"original_q",std::to_string(queries)},
        {"original_kv",std::to_string(kv)},{"head_dim",std::to_string(width)},
        {"q_first",std::to_string(a.head_start)},{"q_count",std::to_string(a.head_count)},
        {"kv_first",std::to_string(a.kv_head_start)},{"kv_count",std::to_string(a.kv_head_count)}};
}
/**
 * @brief Observe the selected successful admission's original device input.
 * @param profile Original immutable source dimensions, not a local tensor view.
 * @param cfg Actual selected DevicePlanConfig consumed by the admitted BOM.
 * @param admission Successfully installed final certificate.
 * @return Exact tuple and admitted byte-owner line; no amount is recomputed.
 */
inline PerfStatsCollector::Tags installedAssignmentTags(const ModelMemoryProfile &profile,
    const DevicePlanConfig &cfg,const PhysicalMemoryPlanAdmissionCertificate &admission)
{
    if (profile.d_model<=0 || !cfg.tensor_parallel_assignment || cfg.world_rank<0 ||
        cfg.shard_index!=cfg.tensor_parallel_assignment->local_rank ||
        cfg.device!=cfg.tensor_parallel_assignment->device)
        throw std::invalid_argument("selected admission lost its actual TP input");
    const auto *bom=admission.plan().find({cfg.world_rank,cfg.device});
    if (!bom)throw std::invalid_argument("selected assignment has no final admitted allocator");
    auto tags=baseTags("selected_admission_input");
    tags.merge(assignmentTags(*cfg.tensor_parallel_assignment,cfg.total_shards,
        profile.n_heads,profile.n_kv_heads,profile.head_dim));
    tags.insert({{"k",std::to_string(profile.d_model)},{"world_rank",std::to_string(cfg.world_rank)},
        {"admission_identity",admissionIdentity(&admission)},
        {"kv_planned_bytes",std::to_string(bom->bytes(PhysicalMemoryOwner::KVCache))},
        {"kv_resident_bytes",std::to_string(bom->alreadyResidentBytes(PhysicalMemoryOwner::KVCache))}});
    return tags;
}
/**
 * @brief Publish only real selected final inputs, never candidate-loop observations.
 * @param profile Original source metadata used by admission.
 * @param configs Actual selected successful BOM inputs, still owned by setup.
 * @param admission Exact installed final immutable certificate.
 */
inline void recordInstalledAssignments(const ModelMemoryProfile &profile,
    std::span<const DevicePlanConfig> configs,const PhysicalMemoryPlanAdmissionCertificate &admission)
{
    for (const auto &cfg:configs)
        if (cfg.device.is_gpu() && cfg.tensor_parallel_assignment && cfg.total_shards>1)
            PerfStatsCollector::addCounter("gpu_graph_inventory","query_kv_geometry",1.0,"admission_installed",
                cfg.device.toString(),installedAssignmentTags(profile,cfg,admission));
}
/**
 * @brief Project the actual constructed cache beside its graph assignment and PMA.
 * @param config Actual retained graph configuration validated before cache construction.
 * @param device Actual participant endpoint.
 * @param cache Successfully constructed, ownership/lease-bound production cache.
 * @param authority Exact graph/cache physical authority.
 * @return Passive actual geometry and coherent PMA owner observation.
 *
 * The admission-input record is separate: graph metadata cannot retroactively
 * prove which assignment authored a prior certificate. No live token count is
 * read. The one-token native block-layout request inspects geometry only.
 */
inline PerfStatsCollector::Tags cacheTags(const GraphConfig &config,DeviceId device,
    const IKVCache &cache,const PhysicalMemoryAuthority &authority)
{
    const auto interval=resolveQueryIntervalCacheHeads(config,device,KVHeadConsumerDemand::AssignedQueryInterval);
    if (!interval || !cache.hasPhysicalMemoryLease() || cache.kv_head_start()!=interval->first ||
        cache.local_n_kv_heads()!=interval->count || !authority.contains(device))
        throw std::invalid_argument("constructed query cache differs from its actual assignment/lease");
    const int layer=firstRestorablePrefixLayer(cache);
    if(layer<0)throw std::invalid_argument("actual query cache has no original attention layer");
    const auto layout=cache.logicalBlockLayout(layer,1);
    if (layout.kv_head_start!=interval->first || layout.local_kv_heads!=interval->count ||
        layout.head_dim!=config.head_dim)
        throw std::invalid_argument("actual cache payload layout differs from its query assignment");
    auto tags=baseTags("graph_cache_pma");
    tags.merge(assignmentTags(config.tp_config->forRank(config.tp_device_idx),config.tp_config->worldSize(),
        config.n_heads,config.n_kv_heads,config.head_dim));
    bool found=false;
    for (const auto &line:authority.rankAttestation())
        if(line.identity.world_rank==authority.worldRank() && line.identity.device==device &&
            line.owner==PhysicalMemoryOwner::KVCache)
        {
            tags.insert({{"kv_planned_bytes",std::to_string(line.planned_new_bytes+line.planned_resident_bytes)},
                {"kv_resident_bytes",std::to_string(line.planned_resident_bytes)},
                {"kv_materialized_bytes",std::to_string(line.materialized_new_bytes+line.adopted_resident_bytes)},
                {"kv_committed_new_bytes",std::to_string(line.committed_new_bytes)}});
            found=true;break;
        }
    if(!found)throw std::invalid_argument("actual cache PMA observation lacks its KV owner");
    tags.insert({{"k",std::to_string(config.d_model)},{"world_rank",std::to_string(authority.worldRank())},
        {"cache_first",std::to_string(cache.kv_head_start())},{"cache_count",std::to_string(cache.local_n_kv_heads())},
        {"payload_first",std::to_string(layout.kv_head_start)},{"payload_count",std::to_string(layout.local_kv_heads)},
        {"admission_identity",admissionIdentity(authority.admission().get())}});
    return tags;
}
/** @copydoc cacheTags
 * @details Publishes actual successful main-cache setup; replay never returns through this helper. */
inline void recordCache(const GraphConfig &config,DeviceId device,const IKVCache &cache,
    const PhysicalMemoryAuthority &authority)
{
    PerfStatsCollector::addCounter("gpu_graph_inventory","query_kv_geometry",1.0,"cache_constructed",
        device.toString(),cacheTags(config,device,cache,authority));
}
} // namespace llaminar2::query_kv_geometry
