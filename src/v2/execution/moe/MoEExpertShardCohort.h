/**
 * @file MoEExpertShardCohort.h
 * @brief Immutable source-native geometry of jointly owned expert fragments.
 *
 * One logical resident is present only when every original fragment is present.
 * This value owns no allocation, quota or execution state. The existing owner
 * map retains it; PMA receives exact local manifests through its ordinary byte
 * resolver. Original source coordinates survive compact materialization, so a
 * down-K slice cannot masquerade as a smaller-stride view of its full parent.
 */
#pragma once
#include "MoEExpertProjectionOwnership.h"
#include "MoEOverlayPhysicalResidencyFabric.h"
#include "backends/GlobalDeviceAddress.h"
#include "loaders/WeightIdentity.h"
#include "tensors/BlockStructures.h"
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <set>
#include <span>
#include <vector>
#include <limits>
#include <numeric>

namespace llaminar2 {
/** @brief Source-authenticated, immutable joint ownership of one layer's expert layout. */
class MoEExpertShardCohort final {
public:
    /** @brief Original canonical owner-map participant, including rank and physical locality. */
    struct Participant {
        int participant_id;
        int world_rank;
        GlobalDeviceAddress address;
        /** @param other Original participant metadata to compare.
         * @return Equality includes rank and NUMA, never a fabricated CPU ordinal. */
        bool operator==(const Participant& other) const = default;
    };
    using Interval=MoEExpertProjectionOwnership::IntermediateInterval;
    /**
     * @brief Partition original F in complete native source blocks, in participant order.
     * @param layer Original gate/up/down source metadata.
     * @param participants Positive joint CPU width, at least two.
     * @return Positive complete intervals, differing by at most one source block.
     * @throws std::invalid_argument For incompatible roles/shapes or insufficient/alignment-breaking F.
     */
    static std::vector<Interval> balancedIntervals(const MoEOverlayLayerWeightManifest& layer,int participants) {
        const auto axes=sourceAxes(layer);
        if(participants<2)throw std::invalid_argument("CPU expert shard cohort requires multiple original participants");
        const auto& down=projection(layer,ExpertTierWeightProjection::Down);
        const int unit=sourceBlock(down.format);
        if(axes.second%unit || axes.second/unit<participants)
            throw std::invalid_argument("CPU expert F cannot form positive source-native shard intervals");
        const int blocks=axes.second/unit;
        std::vector<Interval> result;int frontier=0;
        for(int p=0;p<participants;++p){const int width=(blocks/participants+(p<blocks%participants))*unit;result.push_back({frontier,width});frontier+=width;}
        return result;
    }
    /**
     * @brief Check whether original down-K admits a positive shard for each member.
     * @param intermediate_columns Full reduction extent of the original down projection.
     * @param down_format Original down source identity before device packing.
     * @param participants Original cohort degree.
     * @param gpu Whether slices must also retain complete Q8 activation groups.
     * @return True if an unpadded native partition exists independently of preferred tile size.
     */
    static bool supportsIntermediateDegree(std::size_t intermediate_columns,
        const ExpertWeightFormat &down_format, int participants, bool gpu)
    {
        if (!down_format.valid() || participants < 2 ||
            intermediate_columns > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            return false;
        const auto block = sourceBlock(down_format);
        const auto grain = static_cast<std::size_t>(gpu ? std::lcm(block, 32) : block);
        return intermediate_columns % grain == 0 &&
            intermediate_columns / grain >= static_cast<std::size_t>(participants);
    }

    /**
     * @brief Seal original source and exact ordered CPU fragment membership.
     * @param layer Original, unsliced source layer metadata.
     * @param experts Original complete expert-axis extent, not a tier quota.
     * @param participants Canonical participant IDs, real MPI ranks and explicit CPU NUMA endpoints.
     * @param intervals Complete positive native-block partition of the original F axis.
     * @return Immutable source layout; no execution or admission permission.
     * @throws std::invalid_argument For missing/foreign/duplicate endpoints or incompatible source cuts.
     */
    static MoEExpertShardCohort checked(MoEOverlayLayerWeightManifest layer,int experts,
        std::vector<Participant> participants,std::vector<Interval> intervals) {
        if(participants.size()<2 || participants.size()>static_cast<size_t>(std::numeric_limits<int>::max()) || intervals.size()!=participants.size())
            throw std::invalid_argument("CPU expert shard cohort requires complete ordered membership and intervals");
        // NODE_LOCAL inventory binding belongs to the caller's original topology.
        // Hostname spellings are retained identity labels, never a locality proof.
        std::set<int> ids,ranks,numa;
        for(std::size_t p=0;p<participants.size();++p){
            const auto& member=participants[p];
            if(member.participant_id<0 || member.world_rank<0 || !member.address.isCPU() ||
               !member.address.hasValidNuma() || member.address.device_ordinal!=0 ||
               !ids.insert(member.participant_id).second || !ranks.insert(member.world_rank).second ||
               !numa.insert(member.address.numa_node).second)
                throw std::invalid_argument("CPU expert shard cohort requires distinct original rank/NUMA endpoints");
        }
        auto fragments = sourceFragments(layer, experts, intervals);
        return {std::move(layer),std::move(participants),std::move(fragments)};
    }
    /**
     * @brief Partition GPU F with native blocks, activation groups and economical tiles.
     * @param layer Original unsliced source metadata, including the down codebook.
     * @param participants Original domain width; no hardware-count specialization.
     * @param rotation Layer coordinate rotating the larger fragments.
     * @param preferred_grain Bound source partition preference, always subordinate to source blocks.
     * @return Complete positive intervals, with no padding or requantization.
     * @throws std::invalid_argument For source geometry which cannot form legal fragments.
     */
    static std::vector<Interval> gpuIntervals(const MoEOverlayLayerWeightManifest &layer,
        int participants, int rotation, int preferred_grain = qfnGPUExpertPreferredIntermediateGrain())
    {
        const auto axes = sourceAxes(layer);
        return MoEExpertProjectionOwnership::nativeIntermediateIntervals(axes.second, participants,
            sourceBlock(projection(layer, ExpertTierWeightProjection::Down).format), rotation, preferred_grain);
    }
    /**
     * @brief Seal homogeneous GPU fragments owned by one original MPI rank.
     * @param layer Original unsliced source layer metadata.
     * @param experts Original complete expert-axis extent.
     * @param participants Canonical owner-map members, in collective reduction order.
     * @param intervals Complete native F partition in that same order.
     * @return Source-bound joint ownership; this grants no execution capability.
     * @throws std::invalid_argument For mixed backends/ranks, duplicate GPUs or illegal source cuts.
     * @note The original inventory-bound domain proves physical locality. Ordinals
     *       identify members, never a shard coordinate or an alternate MPI rank.
     */
    static MoEExpertShardCohort checkedRankLocalGPU(MoEOverlayLayerWeightManifest layer,
        int experts, std::vector<Participant> participants, std::vector<Interval> intervals)
    {
        if (participants.empty() || participants.size() > size_t(std::numeric_limits<int>::max()) ||
            intervals.size() != participants.size())
            throw std::invalid_argument("GPU expert shard cohort requires complete ordered membership and intervals");
        const auto &first = participants.front();
        std::set<int> ids, devices;
        for (const auto &member : participants)
            if (member.participant_id < 0 || member.world_rank < 0 || !member.address.isGPU() ||
                member.world_rank != first.world_rank || member.address.device_type != first.address.device_type ||
                member.address.device_ordinal < 0 || !ids.insert(member.participant_id).second ||
                !devices.insert(member.address.device_ordinal).second)
                throw std::invalid_argument("GPU expert shard cohort requires unique homogeneous devices on one original rank");
        auto fragments = sourceFragments(layer, experts, intervals);
        return {std::move(layer), std::move(participants), std::move(fragments)};
    }
    /** @param other Immutable cohort definition to compare, independent of its wrapper address.
     * @return Equality covers original source formats, endpoint identity and every F interval. */
    bool operator==(const MoEExpertShardCohort& other) const = default;
    /** @return Original layer/format authority, before any slicing. */
    const MoEOverlayLayerWeightManifest& source() const noexcept{return source_;}
    /** @return Original canonical membership in fixed reduction order. */
    std::span<const Participant> participants() const noexcept{return participants_;}
    /**
     * @param participant Original domain coordinate.
     * @return Exact existing projection ownership for this physical fragment.
     * @throws std::out_of_range For a foreign coordinate.
     */
    const MoEExpertProjectionOwnership& ownership(std::size_t participant) const{return fragments_.at(participant);}
    /**
     * @param participant Original domain coordinate.
     * @return Original-format local N/K metadata consumed by the existing prepared-byte calculator.
     * @throws std::out_of_range For a foreign coordinate.
     */
    MoEOverlayLayerWeightManifest localManifest(std::size_t participant) const {
        auto local=source_;for(auto& p:local.projections){auto view=ownership(participant).projection(weightRole(p.projection));p.N=view.rows;p.K=view.columns;}return local;
    }
    /**
     * @param participant Original domain coordinate.
     * @param role Original semantic projection role.
     * @param expert_ids Exact sorted original experts in the compact source parent.
     * @return Original GGUF [K,N,E] identity for a compact native-format fragment.
     * @throws std::invalid_argument For duplicate/foreign/empty original experts or an unknown role.
     * @throws std::out_of_range For a foreign participant coordinate.
     * @note A down-K fragment must actually be compacted by native block copies;
     *       this declaration grants no permission to reinterpret parent strides.
     */
    WeightSliceSpec sourceSlice(std::size_t participant,WeightRole role,std::vector<int> expert_ids) const {
        const auto& layout=ownership(participant);const auto view=layout.projection(role);
        if(expert_ids.empty() || !std::is_sorted(expert_ids.begin(),expert_ids.end()) ||
           std::adjacent_find(expert_ids.begin(),expert_ids.end())!=expert_ids.end() ||
           expert_ids.front()<0 || expert_ids.back()>=layout.geometry().experts)
            throw std::invalid_argument("Expert shard source requires exact unique original expert IDs");
        WeightSliceSpec result{.source_rows=static_cast<std::size_t>(view.source_columns),
            .source_cols=static_cast<std::size_t>(view.source_rows),
            .row_start=static_cast<std::size_t>(view.first_column),.row_count=static_cast<std::size_t>(view.columns),
            .col_start=static_cast<std::size_t>(view.first_row),.col_count=static_cast<std::size_t>(view.rows),
            .expert_count=expert_ids.size(),.expert_ids=std::move(expert_ids),.inner_is_presliced=true};
        return result;
    }
    /**
     * @brief Request original native blocks before compact materialization.
     * @param participant Original domain coordinate.
     * @param role Original semantic projection role.
     * @param expert_ids Exact sorted original experts requested from the loader.
     * @return The same original GGUF rectangle, marked as an unsliced source-load request.
     * @throws std::invalid_argument For the same foreign/duplicate expert or role errors as sourceSlice.
     * @throws std::out_of_range For a foreign participant coordinate.
     * @note Use sourceSlice only after successful compact native materialization;
     *       the original loader correctly rejects a presliced input request.
     */
    WeightSliceSpec sourceLoadSlice(std::size_t participant,WeightRole role,std::vector<int> expert_ids) const {
        auto result=sourceSlice(participant,role,std::move(expert_ids));result.inner_is_presliced=false;return result;
    }
private:
    /**
     * @brief Validate native source cuts once for both CPU and GPU membership contracts.
     * @param layer Original unsliced role/format manifest.
     * @param experts Original complete expert-axis extent.
     * @param intervals Ordered, complete F partition already matched to membership.
     * @return Exact role rectangles for every member, without copied weight payloads.
     * @throws std::invalid_argument For inconsistent source axes or split native blocks.
     */
    static std::vector<MoEExpertProjectionOwnership> sourceFragments(
        const MoEOverlayLayerWeightManifest &layer, int experts, const std::vector<Interval> &intervals)
    {
        const auto geometry = sourceGeometry(layer, experts);
        std::vector<MoEExpertProjectionOwnership> fragments;
        fragments.reserve(intervals.size());
        for (size_t p = 0; p < intervals.size(); ++p)
        {
            auto ownership = MoEExpertProjectionOwnership::withinExpertIntermediateSlices(
                geometry, static_cast<int>(p), intervals);
            for (const auto &source : layer.projections)
            {
                const auto local = ownership.projection(weightRole(source.projection));
                const int block = sourceBlock(source.format);
                if (source.K % block || local.first_column % block || local.columns % block)
                    throw std::invalid_argument("Expert shard source K cut requires unsupported native codebook repacking");
            }
            fragments.push_back(std::move(ownership));
        }
        return fragments;
    }
    /**
     * @brief Retain only fully checked source metadata and original participants.
     * @param source Validated original model-global layer and native formats.
     * @param participants Exact original canonical fragment membership.
     * @param fragments Complete source-bound ownership values in that membership order.
     */
    MoEExpertShardCohort(MoEOverlayLayerWeightManifest source,std::vector<Participant> participants,
        std::vector<MoEExpertProjectionOwnership> fragments):source_(std::move(source)),participants_(std::move(participants)),fragments_(std::move(fragments)){}
    /**
     * @param role Existing semantic manifest projection role.
     * @return Original semantic stage role, never its incidental array index.
     * @throws std::invalid_argument For an unknown projection role.
     */
    static WeightRole weightRole(ExpertTierWeightProjection role){switch(role){case ExpertTierWeightProjection::Gate:return WeightRole::MoEExpertGate;case ExpertTierWeightProjection::Up:return WeightRole::MoEExpertUp;case ExpertTierWeightProjection::Down:return WeightRole::MoEExpertDown;}throw std::invalid_argument("Unknown expert projection role");}
    /**
     * @param layer Original layer whose completeness was authenticated before lookup.
     * @param role Existing semantic manifest projection role.
     * @return Exact original projection metadata.
     * @throws std::invalid_argument If that original role is absent.
     */
    static const MoEOverlayProjectionWeightManifest& projection(const MoEOverlayLayerWeightManifest& layer,ExpertTierWeightProjection role){auto p=std::find_if(layer.projections.begin(),layer.projections.end(),[&](const auto& x){return x.projection==role;});if(p==layer.projections.end())throw std::invalid_argument("Missing expert shard projection");return *p;}
    /**
     * @param format Original immutable source format identity.
     * @return Original source block width from its canonical family trait.
     * @throws std::invalid_argument For an absent or uncatalogued source format.
     */
    static int sourceBlock(const ExpertWeightFormat& format){
        if(!format.valid())throw std::invalid_argument("Unknown expert shard source format");
        if(format.isFloating())return 1; // Ordinary IEEE scalars have no encoded group dependency.
        const auto* original=native_quantised_formats::forSourceIdentity(format.native_quantised.codebook_id,format.native_quantised.is_superblock);
        if(!original)throw std::invalid_argument("Unknown original expert source codebook");
        // The canonical source catalog declares two block families. Their POD
        // source traits own the widths; no shard codec/size table is maintained.
        return original->is_superblock?Q4_KBlock::BLOCK_SIZE:Q4_0Block::BLOCK_SIZE;
    }
    /**
     * @param layer Original complete gate/up/down source metadata.
     * @param experts Original model expert-axis extent, independent of admitted tier quota.
     * @return Original E/H/F geometry after cross-role authentication.
     * @throws std::invalid_argument For missing/duplicate roles or inconsistent N/K/E geometry.
     */
    static MoEExpertProjectionOwnership::Geometry sourceGeometry(const MoEOverlayLayerWeightManifest& layer,int experts){
        if(experts<=0)throw std::invalid_argument("Expert shard cohort requires the original expert axis");
        const auto axes=sourceAxes(layer);return {experts,axes.first,axes.second};
    }
    /**
     * @param layer Original complete semantic gate/up/down manifest.
     * @return Original H/F axes, independent of any tier quota or expert selection.
     * @throws std::invalid_argument For incomplete/duplicate roles or disagreeing projection shapes.
     */
    static std::pair<int,int> sourceAxes(const MoEOverlayLayerWeightManifest& layer){
        if(!layer.valid())throw std::invalid_argument("Expert shard cohort requires original valid layer metadata");
        const auto& gate=projection(layer,ExpertTierWeightProjection::Gate);const auto& up=projection(layer,ExpertTierWeightProjection::Up);const auto& down=projection(layer,ExpertTierWeightProjection::Down);
        if(gate.N!=up.N || gate.K!=up.K || down.N!=gate.K || down.K!=gate.N)
            throw std::invalid_argument("Expert shard gate/up/down source geometry disagrees");
        return {gate.K,gate.N};
    }
    MoEOverlayLayerWeightManifest source_;
    std::vector<Participant> participants_;
    std::vector<MoEExpertProjectionOwnership> fragments_;
};
} // namespace llaminar2
