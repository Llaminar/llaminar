/**
 * @file MoEOverlayPhysicalLaneDemand.h
 * @brief Passive original-piece demand shared by staging BOM and physical lane setup.
 *
 * A canonical closed cycle visits each complete placement target once. A CPU
 * cohort is one target with several real projection endpoints. Bounds count
 * the actual nonempty intersections, never allocated columns or a scalar -1
 * owner. This is immutable BOM geometry; PMA remains the only byte authority.
 */
#pragma once
#include "MoEOverlayCapacityAdmission.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include "MoEOverlayPhysicalProjectionPieces.h"
#include "MoEOverlayMPIRemoteProjectionTransport.h"
#include <optional>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <tuple>

namespace llaminar2 {
/**
 * @brief Original process-local or communicator lane topology, without rank inference.
 * No transport means an exact PMA rank-local scope, not necessarily world rank0.
 */
class MoEOverlayPhysicalLaneScope final {
public:
    /** @param size Exact original communicator size.
     * @return Checked communicator scope; no rank or device is probed. */
    static MoEOverlayPhysicalLaneScope communicator(int size) {
        if(size<=0)throw std::invalid_argument("Physical lane scope requires a positive communicator");
        return {true,size};
    }
    /** @param rank Exact process rank retained by its original PMA.
     * @return Checked local-only scope, with no fabricated communicator. */
    static MoEOverlayPhysicalLaneScope rankLocal(int rank) {
        if(rank<0)throw std::invalid_argument("Physical lane scope requires its original process rank");
        return {false,rank};
    }
    /** @return Whether existing distributed ports are actually declared. */
    bool distributed()const noexcept{return communicator_ && coordinate_>1;}
    /** @return Whether a real communicator bound is available. */
    bool hasCommunicator()const noexcept{return communicator_;}
    /** @param rank Unchanged resolved original endpoint rank.
     * @return Whether this setup scope materializes that process's local pools. */
    bool ownsRank(int rank)const noexcept{return communicator_ || rank==coordinate_;}
    /** @param rank Original endpoint rank, or unbound local metadata.
     * @return Same original rank, resolving only an explicitly local scope.
     * @throws std::invalid_argument For remote endpoints without their communicator. */
    int requireRank(int rank)const {
        if(!communicator_) {
            if(rank<0)return coordinate_;
            return rank; // Remote declarations remain passive metadata, never renumbered.
        }
        if(rank<0 && coordinate_==1)return 0;
        if(rank<0 || rank>=coordinate_)throw std::invalid_argument("Physical endpoint is outside its original communicator");
        return rank;
    }
private:
    /** @brief Retain only checked original scope coordinates. */
    MoEOverlayPhysicalLaneScope(bool communicator,int coordinate):communicator_(communicator),coordinate_(coordinate){}
    bool communicator_;int coordinate_;
};

/** @brief Existing independently-owned staging or CPU worker pool family. */
enum class MoEOverlayPhysicalLaneKind { LocalDevice, LocalCPUWorker, RemoteSource, RemoteDestination, RemoteProjectionDirectory };
/** @brief Exact existing pool identity, independent of live allocation/accounting state. */
struct MoEOverlayPhysicalLaneKey {
    MoEOverlayPhysicalLaneKind kind;
    int world_rank=-1;
    DeviceId source=DeviceId::invalid(),destination=DeviceId::invalid();
    GlobalDeviceAddress source_address,destination_address;
    ExpertTierWeightProjection projection=ExpertTierWeightProjection::Gate;
    /** @return Equality of every original endpoint and role coordinate. */
    bool operator==(const MoEOverlayPhysicalLaneKey&) const = default;
    /** @param other Complete pool key.
     * @return Stable full-key ordering, including actual CPU NUMA addresses. */
    bool operator<(const MoEOverlayPhysicalLaneKey& other) const noexcept {
        return std::tie(kind,world_rank,source,destination,source_address,destination_address,projection)<
               std::tie(other.kind,other.world_rank,other.source,other.destination,other.source_address,other.destination_address,other.projection);
    }
};
using MoEOverlayPhysicalLaneCounts=std::map<MoEOverlayPhysicalLaneKey,size_t>;
/**
 * @brief One immutable setup plan, separating real piece concurrency and retained ports.
 *
 * Existing remote GPU source/destination ports and the global MPI ordinal
 * directory are declared setup resources even when an initial inventory has
 * no cross-rank edge. They are not fabricated migration or traffic witnesses.
 */
struct MoEOverlayPhysicalLaneDemandPlan {
    size_t maximum_concurrent_cycles=0;
    int original_rank=-1; ///< Original owner of the immutable local GPU classification.
    std::map<DeviceId,size_t> original_execution_streams; ///< SAME admitted shared pool population, not per-layer/event counts.
    std::map<std::pair<DeviceId,DeviceId>,bool> direct_gpu_edges; ///< Original source/destination choices, including false.
    MoEOverlayPhysicalLaneCounts migration_pieces;
    MoEOverlayPhysicalLaneCounts setup_ports;
    /** @return Required capacity of each single existing pool, without double charging shared ports. */
    MoEOverlayPhysicalLaneCounts materializedCounts() const {
        auto counts=migration_pieces;
        for(const auto& [key,value]:setup_ports)counts[key]=std::max(counts[key],value);
        return counts;
    }
};
/** @brief Checked passive lane count arithmetic; no allocation decision is made here. */
inline size_t checkedMoEOverlayLaneAdd(size_t a,size_t b) {
    if(b>std::numeric_limits<size_t>::max()-a)throw std::overflow_error("Physical projection lane count overflow");
    return a+b;
}
/**
 * @brief Reduce actual nonempty pieces into existing pool identities.
 * @param pieces Original-coordinate intersections from the shared enumerator.
 * @param source Complete ordered original source target.
 * @param destination Complete ordered original destination target.
 * @param scope Exact original communicator or rank-local process scope.
 * @return Actual operation/worker counts, with no CPU-to-CPU staging demand.
 */
inline MoEOverlayPhysicalLaneCounts physicalMoEOverlayLaneCounts(
    std::span<const MoEOverlayPhysicalProjectionPiece> pieces,
    const MoEOverlayPhysicalProjectionTarget& source,const MoEOverlayPhysicalProjectionTarget& destination,
    MoEOverlayPhysicalLaneScope scope)
{
    MoEOverlayPhysicalLaneCounts result;
    const auto rank=[&](int value){return scope.requireRank(value);};
    const auto add=[&](MoEOverlayPhysicalLaneKey key){auto& count=result[key];count=checkedMoEOverlayLaneAdd(count,1);};
    std::optional<std::tuple<size_t,size_t,size_t>> previous;
    for(const auto& piece:pieces) {
        const auto role=size_t(piece.projection);
        const auto order=std::tuple{role,piece.destination_member,piece.source_member};
        if(role>=3 || (previous && !(*previous<order)))
            throw std::invalid_argument("Physical lane demand requires unique canonical piece order");
        previous=order;
        const auto& from=source.members.at(piece.source_member);const auto& to=destination.members.at(piece.destination_member);
        if(piece.intersection.empty() || from.participant_id!=piece.source_participant || to.participant_id!=piece.destination_participant)
            throw std::invalid_argument("Physical lane demand changed its original live piece endpoints");
        if(piece.intersection!=ExpertTierWeightProjectionIntersection::between(from.projections[role],to.projections[role]))
            throw std::invalid_argument("Physical lane demand changed original intersection geometry");
        const int sr=rank(from.world_rank),dr=rank(to.world_rank);
        if(!scope.hasCommunicator() && sr!=dr)
            throw std::invalid_argument("Physical cross-rank pieces require their original communicator");
        if(sr==dr && !scope.ownsRank(sr))continue;
        if(sr==dr) {
            if(from.device.is_cpu() && to.device.is_cpu())
                add({.kind=MoEOverlayPhysicalLaneKind::LocalCPUWorker,.world_rank=sr,
                    .source=from.device,.destination=to.device,.source_address=from.address,.destination_address=to.address,.projection=piece.projection});
            else add({.kind=MoEOverlayPhysicalLaneKind::LocalDevice,.world_rank=sr,
                    .source=from.device,.destination=to.device,.projection=piece.projection});
        } else {
            add({.kind=MoEOverlayPhysicalLaneKind::RemoteProjectionDirectory,.projection=piece.projection});
            if(from.device.is_gpu())add({.kind=MoEOverlayPhysicalLaneKind::RemoteSource,.world_rank=sr,.source=from.device,.projection=piece.projection});
            if(to.device.is_gpu())add({.kind=MoEOverlayPhysicalLaneKind::RemoteDestination,.world_rank=dr,.destination=to.device,.projection=piece.projection});
        }
    }
    return result;
}

/** @brief Communicator convenience overload retaining its exact original size.
 * @param pieces Original nonempty intersections.
 * @param source Original ordered source layout.
 * @param destination Original ordered destination layout.
 * @param world_size Exact original communicator size.
 * @return Same typed pool counts; no local rank is inferred. */
inline MoEOverlayPhysicalLaneCounts physicalMoEOverlayLaneCounts(
    std::span<const MoEOverlayPhysicalProjectionPiece> pieces,const MoEOverlayPhysicalProjectionTarget& source,
    const MoEOverlayPhysicalProjectionTarget& destination,int world_size)
{return physicalMoEOverlayLaneCounts(pieces,source,destination,MoEOverlayPhysicalLaneScope::communicator(world_size));}

/**
 * @brief Reduce an already normalized runtime edge through the original checked layouts.
 * @param pieces Actual original-coordinate physical pieces, not allocation capacity.
 * @param migration Original immutable logical edge.
 * @param manifest Full original native source manifest.
 * @param experts Original expert population.
 * @param topology Original endpoint inventory retained by the same snapshot.
 * @param scope Exact original communicator or process-local rank.
 * @return Same typed pool counts used by configured setup planning.
 */
inline MoEOverlayPhysicalLaneCounts physicalMoEOverlayLaneCounts(
    std::span<const MoEOverlayPhysicalProjectionPiece> pieces,const MoEOverlayTierMigration& migration,
    const MoEOverlayLayerWeightManifest& manifest,int experts,const MoEExpertOwnerMap& topology,
    MoEOverlayPhysicalLaneScope scope)
{
    const auto [from,to]=physicalMoEOverlayProjectionLayouts(migration,manifest,experts,topology);
    return physicalMoEOverlayLaneCounts(pieces,from,to,scope);
}
/** @brief Communicator overload for the original checked runtime edge.
 * @param pieces Actual normalized intersections.
 * @param migration Original logical edge.
 * @param manifest Full original source metadata.
 * @param experts Original full E.
 * @param topology Original endpoint inventory.
 * @param world_size Exact original communicator size.
 * @return Same pool counts through the checked scope producer. */
inline MoEOverlayPhysicalLaneCounts physicalMoEOverlayLaneCounts(
    std::span<const MoEOverlayPhysicalProjectionPiece> pieces,const MoEOverlayTierMigration& migration,
    const MoEOverlayLayerWeightManifest& manifest,int experts,const MoEExpertOwnerMap& topology,int world_size)
{return physicalMoEOverlayLaneCounts(pieces,migration,manifest,experts,topology,MoEOverlayPhysicalLaneScope::communicator(world_size));}

/**
 * @brief Validate the canonical simple-target cycle bound before lane submission.
 * @param migrations Complete immutable logical movement set.
 * @param cycles Actual canonical cycle decomposition.
 * @throws std::invalid_argument For repeated targets, torn cycles or incomplete edge coverage.
 * @note The production cycle builder uses a visited-target return path. A
 *       cohort is one vertex; its members are not independent capacity cycles.
 */
inline void requireSimpleMoEOverlayPhysicalCycles(const std::vector<MoEOverlayTierMigration>& migrations,
    const std::vector<MoEOverlayTierMigrationCycle>& cycles)
{
    std::vector<bool> covered(migrations.size());
    for(const auto& cycle:cycles) {
        if(!cycle.valid(migrations))throw std::invalid_argument("Physical lane demand requires complete canonical cycles");
        std::vector<MoEExpertPlacementTarget> vertices;
        for(size_t index:cycle.migration_indices) {
            const auto source=MoEExpertPlacementTarget::fromPlacement(migrations.at(index).source);
            if(!source || covered[index] || std::find(vertices.begin(),vertices.end(),*source)!=vertices.end())
                throw std::invalid_argument("Physical lane cycle repeats an original placement target or edge");
            vertices.push_back(*source);covered[index]=true;
        }
    }
    if(std::find(covered.begin(),covered.end(),false)!=covered.end())
        throw std::invalid_argument("Physical lane cycles omit original migration edges");
}
/**
 * @brief Require actual immutable wave demand to fit the one materialized plan.
 * @param plan Setup-owned piece/declared-port capacity, not a live ledger.
 * @param actual Actual normalized wave pool counts.
 * @throws std::invalid_argument Before enqueue if any exact pool is absent or exhausted.
 */
inline void requireFitsMoEOverlayLaneDemand(const MoEOverlayPhysicalLaneDemandPlan& plan,
    const MoEOverlayPhysicalLaneCounts& actual)
{
    const auto capacity=plan.materializedCounts();
    for(const auto& [key,count]:actual) {
        const auto found=capacity.find(key);
        if(found==capacity.end() || count>found->second)
            throw std::invalid_argument("Physical projection wave exceeds its original typed lane-demand plan");
    }
}

/**
 * @brief Derive passive configured targets from original source and declared domains.
 * @param plan Bound original topology, including tiers that initially own no experts.
 * @param source Full original native layer manifest.
 * @param experts Full original expert axis.
 * @return Whole participant targets or one complete fixed CPU cohort per tier.
 * @note No owner map, selected expert IDs, source lease or execution token is constructed.
 */
inline std::vector<MoEOverlayPhysicalProjectionTarget> configuredMoEOverlayProjectionTargets(
    const MoERoutedExpertPlacementPlan& plan,const MoEOverlayLayerWeightManifest& source,int experts)
{
    if(!source.valid() || experts<=0)throw std::invalid_argument("Configured projection demand requires original full source geometry");
    const auto bound=MoEOverlayCapacityAdmission::boundParticipants(plan);
    std::vector<MoEOverlayPhysicalProjectionTarget> result;
    for(size_t tier=0;tier<plan.routed_tiers.size();++tier) {
        const auto& definition=plan.routed_tiers[tier];
        const auto domain=std::find_if(plan.domains.begin(),plan.domains.end(),[&](const auto& value){return value.name==definition.domain;});
        if(domain==plan.domains.end())throw std::invalid_argument("Configured projection tier lost its original domain");
        std::vector<MoEOverlayPhysicalProjectionMember> members;
        for(const auto& entry:bound)if(entry.tier_index==int(tier))members.push_back({
            .participant_id=entry.participant_id,.world_rank=entry.world_rank,.device=entry.device,
            .address=domain->participants.at(size_t(entry.domain_participant_index))});
        if(members.empty())throw std::invalid_argument("Configured projection tier has no original members");
        if(domain->routed_compute_policy==RoutedExpertComputePolicy::TensorSharded && members.size()>1) {
            std::vector<MoEExpertShardCohort::Participant> declared;
            for(const auto& member:members) {
                if(!member.device.is_cpu())throw std::invalid_argument("Configured physical GPU cohort demand is not implemented");
                declared.push_back({member.participant_id,member.world_rank,member.address});
            }
            const auto cohort=MoEExpertShardCohort::checked(source,experts,std::move(declared),
                MoEExpertShardCohort::balancedIntervals(source,int(members.size())));
            for(size_t member=0;member<members.size();++member)
                for(const auto& projection:source.projections) {
                    const auto role=projection.projection==ExpertTierWeightProjection::Gate?WeightRole::MoEExpertGate:
                        projection.projection==ExpertTierWeightProjection::Up?WeightRole::MoEExpertUp:WeightRole::MoEExpertDown;
                    members[member].projections[size_t(projection.projection)]=cohort.ownership(member).projection(role);
                }
            result.push_back({int(tier),std::move(members)});
        } else {
            for(auto& member:members) {
                for(const auto& projection:source.projections)member.projections[size_t(projection.projection)]={
                    MoEProjectionResidence::ExpertOwner,projection.N,projection.K,0,projection.N,0,projection.K};
                result.push_back({int(tier),{std::move(member)}});
            }
        }
    }
    return result;
}

namespace detail {
/**
 * @brief Bound one pool's cycle demand by one incoming/outgoing edge per target.
 * @param weights Exact per-target-pair live piece counts for this pool.
 * @return Maximum assignment weight, bounding every canonical simple cycle.
 * @note This count-only augmenting assignment is O(T^3); it has no scheduling,
 *       cost-model, live ledger or admission authority. Zero-weight self edges
 *       leave unused targets outside the cycle. Whole uniform edges reduce to
 *       the existing min(source multiplicity,destination multiplicity) bound.
 */
inline size_t maximumMoEOverlayLaneIncidence(const std::vector<std::vector<size_t>>& weights) {
    const size_t n=weights.size();if(!n)return 0;
    size_t maximum=0;for(const auto& row:weights){if(row.size()!=n)throw std::invalid_argument("Projection incidence must be square");
        for(size_t value:row)maximum=std::max(maximum,value);}
    if(maximum && n>size_t(std::numeric_limits<int64_t>::max()/4)/maximum)
        throw std::overflow_error("Projection incidence exceeds checked assignment arithmetic");
    std::vector<int64_t> u(n+1),v(n+1);std::vector<size_t> p(n+1),way(n+1);
    for(size_t i=1;i<=n;++i) {
        p[0]=i;size_t j0=0;std::vector<int64_t> minv(n+1,std::numeric_limits<int64_t>::max()/2);std::vector<bool> used(n+1);
        do {
            used[j0]=true;const size_t i0=p[j0];int64_t delta=std::numeric_limits<int64_t>::max()/2;size_t j1=0;
            for(size_t j=1;j<=n;++j)if(!used[j]) {
                const int64_t current=int64_t(maximum-weights[i0-1][j-1])-u[i0]-v[j];
                if(current<minv[j]){minv[j]=current;way[j]=j0;}
                if(minv[j]<delta){delta=minv[j];j1=j;}
            }
            for(size_t j=0;j<=n;++j)if(used[j]){u[p[j]]+=delta;v[j]-=delta;}else minv[j]-=delta;
            j0=j1;
        }while(p[j0]);
        do {const size_t previous=way[j0];p[j0]=p[previous];j0=previous;}while(j0);
    }
    size_t total=0;for(size_t j=1;j<=n;++j)total=checkedMoEOverlayLaneAdd(total,weights[p[j]-1][j-1]);return total;
}
}
/**
 * @brief Quote every configured physical pool from actual source cuts and cycle policy.
 * @param plan Original bound topology.
 * @param manifests Original full native layer manifests.
 * @param experts Full expert axis.
 * @param scope Exact original communicator or process-local rank.
 * @param cycles Admitted simultaneous simple cycles.
 * @param cycles_per_layer Existing maximum cycles from one model layer in that wave.
 * @return One immutable count plan used by both BOM and lane materialization.
 */
inline MoEOverlayPhysicalLaneDemandPlan configuredMoEOverlayLaneDemand(
    const MoERoutedExpertPlacementPlan& plan,std::span<const MoEOverlayLayerWeightManifest> manifests,
    int experts,MoEOverlayPhysicalLaneScope scope,size_t cycles,size_t cycles_per_layer)
{
    if(manifests.empty() || !cycles || !cycles_per_layer)throw std::invalid_argument("Configured lane demand requires original manifests and positive cycle count");
    std::map<MoEOverlayPhysicalLaneKey,std::vector<size_t>> layer_bounds;
    std::vector<int> layer_ids;
    for(const auto& manifest:manifests) {
        if(manifest.layer_idx<0 || std::find(layer_ids.begin(),layer_ids.end(),manifest.layer_idx)!=layer_ids.end())
            throw std::invalid_argument("Configured physical lane profiles repeat an original model layer");
        layer_ids.push_back(manifest.layer_idx);
        const auto targets=configuredMoEOverlayProjectionTargets(plan,manifest,experts);const size_t n=targets.size();
        std::map<MoEOverlayPhysicalLaneKey,std::vector<std::vector<size_t>>> incidence;
        for(size_t src=0;src<n;++src)for(size_t dst=0;dst<n;++dst) {
            auto pieces=physicalMoEOverlayProjectionPieces(targets[src],targets[dst]);
            if(!scope.hasCommunicator()) {
                // Setup may retain global source metadata without owning those
                // processes or an MPI data plane. Quote only local endpoints;
                // actual cross-rank submissions are rejected by the reducer.
                pieces.erase(std::remove_if(pieces.begin(),pieces.end(),[&](const auto& piece){
                    return !scope.ownsRank(scope.requireRank(targets[src].members[piece.source_member].world_rank)) ||
                           !scope.ownsRank(scope.requireRank(targets[dst].members[piece.destination_member].world_rank));}),pieces.end());
            }
            for(const auto& [key,count]:physicalMoEOverlayLaneCounts(pieces,targets[src],targets[dst],scope)) {
                auto [it,inserted]=incidence.try_emplace(key,n,std::vector<size_t>(n));
                it->second[src][dst]=count;
            }
        }
        for(const auto& [key,weights]:incidence)layer_bounds[key].push_back(detail::maximumMoEOverlayLaneIncidence(weights));
    }
    MoEOverlayPhysicalLaneCounts maximum;
    const size_t layer_cap=std::min(cycles,cycles_per_layer);
    // Saturate at the global cap instead of multiplying an unbounded layer
    // policy. Each layer offers at most its declared number of cycle slots.
    size_t feasible_cycles=0;
    for(size_t layer=0;layer<manifests.size() && feasible_cycles<cycles;++layer)
        feasible_cycles+=std::min(layer_cap,cycles-feasible_cycles);
    for(auto& [key,bounds]:layer_bounds) {
        std::sort(bounds.begin(),bounds.end(),std::greater<size_t>());
        size_t remaining=cycles,total=0;
        for(size_t bound:bounds) {
            const size_t take=std::min(layer_cap,remaining);
            if(bound && take>std::numeric_limits<size_t>::max()/bound)
                throw std::overflow_error("Configured per-layer projection cycle demand overflow");
            total=checkedMoEOverlayLaneAdd(total,bound*take);remaining-=take;
            if(!remaining)break;
        }
        maximum[key]=total;
    }
    MoEOverlayPhysicalLaneDemandPlan result{.maximum_concurrent_cycles=cycles,.migration_pieces=std::move(maximum)};
    const auto ports=MoEOverlayCapacityAdmission::boundParticipants(plan);
    const auto add_port=[&](MoEOverlayPhysicalLaneKey key) {
        auto& value=result.setup_ports[key];value=checkedMoEOverlayLaneAdd(value,feasible_cycles);
    };
    if(scope.distributed())for(const auto& port:ports)for(size_t role=0;role<3;++role) {
        const auto projection=static_cast<ExpertTierWeightProjection>(role);
        add_port({.kind=MoEOverlayPhysicalLaneKind::RemoteProjectionDirectory,.projection=projection});
        if(port.device.is_gpu()) {
            add_port({.kind=MoEOverlayPhysicalLaneKind::RemoteSource,.world_rank=port.world_rank,.source=port.device,.projection=projection});
            add_port({.kind=MoEOverlayPhysicalLaneKind::RemoteDestination,.world_rank=port.world_rank,.destination=port.device,.projection=projection});
        }
    }
    return result;
}
/** @brief Quote configured demand for an exact communicator through the same scoped producer.
 * @param plan Bound original topology.
 * @param manifests Full original source manifests.
 * @param experts Original E.
 * @param world_size Exact original communicator size.
 * @param cycles Admitted global cycle cap.
 * @param cycles_per_layer Admitted layer-local cap.
 * @return Same immutable piece/port plan. */
inline MoEOverlayPhysicalLaneDemandPlan configuredMoEOverlayLaneDemand(
    const MoERoutedExpertPlacementPlan& plan,std::span<const MoEOverlayLayerWeightManifest> manifests,
    int experts,int world_size,size_t cycles,size_t cycles_per_layer)
{return configuredMoEOverlayLaneDemand(plan,manifests,experts,MoEOverlayPhysicalLaneScope::communicator(world_size),cycles,cycles_per_layer);}

/**
 * @brief Bind the existing MPI directory to the same per-role physical piece plan.
 * @param plan Original configured demand, including explicit retained directory ports.
 * @return Checked existing budget; actual traffic remains live remote pieces only.
 * @throws std::invalid_argument For an absent/inconsistent three-role directory.
 */
inline MoEOverlayRemoteProjectionLaneBudget remoteMoEOverlayProjectionLaneBudget(
    const MoEOverlayPhysicalLaneDemandPlan& plan)
{
    if(!plan.maximum_concurrent_cycles)throw std::invalid_argument("Remote lane budget requires its original cycle policy");
    const auto counts=plan.materializedCounts();size_t per_role=0;
    for(size_t role=0;role<3;++role) {
        const auto found=counts.find({.kind=MoEOverlayPhysicalLaneKind::RemoteProjectionDirectory,
            .projection=static_cast<ExpertTierWeightProjection>(role)});
        if(found==counts.end() || !found->second || (role && found->second!=per_role))
            throw std::invalid_argument("Remote lane directory requires its complete original role-piece bound");
        per_role=found->second;
    }
    MoEOverlayRemoteProjectionLaneBudget result{.maximum_pieces_per_projection_wave=per_role,
        .maximum_concurrent_cycles=plan.maximum_concurrent_cycles,.projections_per_expert=3};
    if(!result.projectionOperationCount())throw std::overflow_error("Remote physical piece budget overflow");
    return result;
}
/**
 * @brief Read one original discovery-position peer fact without probing a backend.
 * @param inventory Original rank's immutable published GPU rows and directed matrices.
 * @param accessor Device that would address the allocation.
 * @param allocation Original allocation owner.
 * @return Exact directed fact, or unavailable for missing/partial/ambiguous observations.
 */
inline std::optional<bool> originalMoEOverlayDirectedPeerAccess(const RankInventory& inventory,DeviceId accessor,DeviceId allocation)
{
    if(!accessor.is_gpu() || accessor.type!=allocation.type)return std::nullopt;
    const auto& matrix=accessor.is_cuda()?inventory.p2p_cuda:inventory.p2p_rocm;
    const auto count=accessor.is_cuda()?inventory.p2p_cuda_count:inventory.p2p_rocm_count;
    int a=-1,b=-1,rows=0;
    for(const auto& gpu:inventory.gpus)if(gpu.type==accessor.type){
        if(gpu.local_device_id==accessor.ordinal){if(a>=0)return std::nullopt;a=rows;}
        if(gpu.local_device_id==allocation.ordinal){if(b>=0)return std::nullopt;b=rows;}
        ++rows;
    }
    if(count<=0 || count!=rows || matrix.size()!=size_t(count)*size_t(count) || a<0 || b<0)return std::nullopt;
    return matrix[size_t(a)*size_t(count)+size_t(b)];
}

/**
 * @brief Classify the exact direct GPU lane with one original directed access observation.
 * @param source Original allocation owner.
 * @param destination Original executor/consumer.
 * @param query Passive destination-to-source capability lookup; it must not probe hardware.
 * @return True only for same-device or explicitly observed same-backend access.
 * @throws std::runtime_error For missing same-backend directed capability evidence.
 */
template<class Query> bool observedMoEOverlayDirectGpuPeer(DeviceId source,DeviceId destination,Query&& query)
{
    if(!source.is_gpu() || !destination.is_gpu() || source.type!=destination.type)return false;
    if(source==destination)return true;
    const auto observed=query(destination,source);
    if(!observed.has_value())throw std::runtime_error("Original physical GPU lane has no directed peer observation");
    return *observed;
}
/**
 * @brief Freeze original local GPU choices into an existing exact physical lane declaration.
 * @param plan Pure configured piece/port declaration, not a live allocator.
 * @param rank Original local owner.
 * @param query Original immutable directed capability lookup.
 * @throws std::exception For duplicate publication, missing evidence or invalid rank.
 */
template<class Query> void bindOriginalMoEOverlayGpuChoices(MoEOverlayPhysicalLaneDemandPlan& plan,int rank,Query&& query)
{
    if(rank<0 || plan.original_rank!=-1 || !plan.direct_gpu_edges.empty())
        throw std::invalid_argument("Physical GPU lane declaration requires one original rank publication");
    plan.original_rank=rank;
    for(const auto& [key,count]:plan.materializedCounts())
        if(count && key.kind==MoEOverlayPhysicalLaneKind::LocalDevice && key.world_rank==rank &&
            key.source.is_gpu() && key.destination.is_gpu())
            plan.direct_gpu_edges.try_emplace({key.source,key.destination},
                observedMoEOverlayDirectGpuPeer(key.source,key.destination,query));
}
/**
 * @brief Authenticate retained original geometry and peer choices before any native creator.
 * @param admitted SAME source-owned immutable declaration retained by capacity admission.
 * @param actual Actual original model/participant piece geometry, with no live ownership decisions.
 * @param rank Original materializing owner.
 * @param query Existing cached runtime facts, used only to validate the original choice.
 * @throws std::invalid_argument For missing/foreign geometry or a changed original classification.
 * @throws std::runtime_error For absent directed observations.
 */
template<class Query> void requireOriginalMoEOverlayGpuDemand(const MoEOverlayPhysicalLaneDemandPlan* admitted,
    const MoEOverlayPhysicalLaneDemandPlan& actual,int rank,Query&& query)
{
    if(!admitted || admitted->original_rank!=rank || admitted->maximum_concurrent_cycles!=actual.maximum_concurrent_cycles ||
        admitted->migration_pieces!=actual.migration_pieces || admitted->setup_ports!=actual.setup_ports ||
        admitted->original_execution_streams!=actual.original_execution_streams)
        throw std::invalid_argument("Physical GPU lane materialization differs from its original admitted declaration");
    auto observed=actual;
    bindOriginalMoEOverlayGpuChoices(observed,rank,query);
    if(observed.direct_gpu_edges!=admitted->direct_gpu_edges)
        throw std::invalid_argument("Physical GPU lane peer choice changed after original admission");
}
/** @param plan Retained original declaration.
 * @param source Exact allocation owner.
 * @param destination Exact executor.
 * @return Original choice; cached runtime facts cannot reselect it.
 * @throws std::logic_error For an edge absent from original physical incidence. */
inline bool admittedMoEOverlayDirectGpuPeer(const MoEOverlayPhysicalLaneDemandPlan& plan,DeviceId source,DeviceId destination)
{
    const auto edge=plan.direct_gpu_edges.find({source,destination});
    if(edge==plan.direct_gpu_edges.end())throw std::logic_error("Physical GPU edge was not admitted by its original source");
    return edge->second;
}
/** @param plan Original rank-bound piece/port declaration and peer choices.
 * @return Exact peer-transfer event-owner population per actual destination, independent of stream count.
 * @throws std::exception For missing original choices or count overflow. */
inline std::map<DeviceId,size_t> originalMoEOverlayPeerEventCounts(const MoEOverlayPhysicalLaneDemandPlan& plan)
{
    if(plan.original_rank<0)throw std::invalid_argument("Peer event incidence requires its original rank");
    std::map<DeviceId,size_t> result;
    for(const auto& [key,count]:plan.materializedCounts())
        if(count && key.kind==MoEOverlayPhysicalLaneKind::LocalDevice && key.world_rank==plan.original_rank &&
            key.source.is_gpu() && key.destination.is_gpu() && admittedMoEOverlayDirectGpuPeer(plan,key.source,key.destination))
            result[key.destination]=checkedMoEOverlayLaneAdd(result[key.destination],count);
    return result;
}

}
