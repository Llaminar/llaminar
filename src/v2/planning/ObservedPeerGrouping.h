/**
 * @file ObservedPeerGrouping.h
 * @brief Pure grouping preferences over original rank-owned peer observations.
 *
 * This helper never probes, enables links, reserves memory or chooses a native
 * transport. Missing/asymmetric facts grant no mutual edge. Callers first apply
 * model/capacity eligibility; NCCL/RCCL retains all transport authority, including
 * valid domains without direct peer access. Exact partition search has no
 * machine-specific participant limit and is cold planning work only. Uniform
 * and per-stage eligibility share the same exhaustive search and score order;
 * typed infeasibility never disguises malformed original identity.
 */
#pragma once
#include "planning/RankHardwareOwnership.h"
#include "backends/PeerAccessCoverage.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <compare>
#include <span>
#include <tuple>
#include <limits>
#include <vector>
#include <utility>

namespace llaminar2 {
/** @brief Lexicographic topology preference, never a measured communication cost. */
struct ObservedPeerGroupQuality {
    std::size_t fully_connected_pairs = 0; ///< Pairs belonging to complete eligible cliques.
    std::size_t mutual_pairs = 0; ///< All driver-observed bidirectional edges.
    std::size_t located_pairs = 0; ///< Mutual pairs with an observed common PCI ancestor.
    std::int64_t negative_hops = 0; ///< Prefer shorter observed paths after capability ties.
    auto operator<=>(const ObservedPeerGroupQuality&) const = default;
    /** @brief Accumulate independent actual domain groups without combining their links. */
    ObservedPeerGroupQuality& operator+=(ObservedPeerGroupQuality other) {
        fully_connected_pairs += other.fully_connected_pairs;
        mutual_pairs += other.mutual_pairs;
        located_pairs += other.located_pairs;
        negative_hops += other.negative_hops;
        return *this;
    }
};
/** @brief One stage's caller-owned eligibility and exact positive cardinality. */
struct ObservedPeerStageRequest {
    std::size_t size; ///< Actual participant count; never allocation capacity or a hardware-specific cap.
    std::span<const DeviceId> eligible; ///< Original candidates for this stage, including legal overlap with other stages.
};
/** @brief Immutable view of one original rank's published GPU facts. */
class ObservedPeerGrouping final {
public:
    /** @brief Borrow the canonical rank observation for this cold operation. */
    explicit ObservedPeerGrouping(const RankInventory& original): original_(original) {}
    /** @return Original uniquely named GPU; never infer identity from ordinal adjacency. */
    const DeviceInfo& device(DeviceId id) const {
        const DeviceInfo* result=nullptr;
        for(const auto& gpu:original_.gpus) if(gpu.type==id.type && gpu.local_device_id==id.ordinal) {
            if(result) throw std::invalid_argument("Duplicate original peer-group endpoint");
            result=&gpu;
        }
        if(!result || result->uuid.empty()) throw std::invalid_argument("Peer grouping requires original GPU identity");
        return *result;
    }
    /** @return True only for a legal original rank-local homogeneous NUMA group. */
    bool sameDomain(std::span<const DeviceId> group) const {
        if(group.empty()) return false;
        const auto& first=device(group.front());
        for(std::size_t i=0;i<group.size();++i) {
            const auto& gpu=device(group[i]);
            if(!rankOwnsGPUDevice(original_,gpu) || gpu.type!=first.type || gpu.numa_node!=first.numa_node)
                return false;
            if(std::find(group.begin(),group.begin()+i,group[i])!=group.begin()+i)
                throw std::invalid_argument("Repeated original peer-group endpoint");
        }
        return true;
    }
    /**
     * @brief Query two actual same-domain endpoints through discovery-position matrices.
     * @return False for absent, partial, malformed or asymmetric observations.
     */
    bool mutuallyAccessible(DeviceId a,DeviceId b) const {
        const std::array pair{a,b};
        if(a==b || !sameDomain(pair)) return false;
        const auto& matrix=a.type==DeviceType::CUDA?original_.p2p_cuda:original_.p2p_rocm;
        const int count=a.type==DeviceType::CUDA?original_.p2p_cuda_count:original_.p2p_rocm_count;
        int ai=-1,bi=-1,index=0;
        for(const auto& gpu:original_.gpus) if(gpu.type==a.type) {
            if(gpu.local_device_id==a.ordinal) ai=index;
            if(gpu.local_device_id==b.ordinal) bi=index;
            ++index;
        }
        if(count<=0 || count!=index || matrix.size()!=std::size_t(count)*std::size_t(count) || ai<0 || bi<0)
            return false;
        return matrix[std::size_t(ai)*count+bi] && matrix[std::size_t(bi)*count+ai];
    }
    /**
     * @brief Classify the exact original directed peer matrix without a runtime probe.
     * @param group Actual homogeneous rank-local endpoints, including sparse ordinals.
     * @return None, Partial or Complete over all directed non-self edges.
     * @throws std::invalid_argument for absent/incomplete observations or nonlocal geometry.
     * A one-way opportunity is Partial, preserving native-library transport authority.
     */
    PeerAccessCoverage coverage(std::span<const DeviceId> group) const {
        if (group.size() < 2 || !sameDomain(group))
            throw std::invalid_argument("Observed route policy requires an original homogeneous local GPU cohort");
        const auto type=group.front().type;
        const auto& matrix=type==DeviceType::CUDA?original_.p2p_cuda:original_.p2p_rocm;
        const int count=type==DeviceType::CUDA?original_.p2p_cuda_count:original_.p2p_rocm_count;
        std::vector<DeviceId> discovered;
        for (const auto& gpu:original_.gpus) if (gpu.type==type)
            discovered.emplace_back(type,gpu.local_device_id);
        if (count<=0 || std::size_t(count)!=discovered.size() ||
            matrix.size()!=std::size_t(count)*std::size_t(count))
            throw std::invalid_argument("Observed route policy requires the complete original directed peer matrix");
        bool any=false, all=true;
        for (auto from:group) for (auto to:group) if (from!=to) {
            const auto a=std::find(discovered.begin(),discovered.end(),from)-discovered.begin();
            const auto b=std::find(discovered.begin(),discovered.end(),to)-discovered.begin();
            const bool edge=matrix[std::size_t(a)*count+std::size_t(b)]!=0;
            any|=edge;all&=edge;
        }
        return all?PeerAccessCoverage::Complete:(any?PeerAccessCoverage::Partial:PeerAccessCoverage::None);
    }
    /** @return Observed bridge-edge distance; endpoint BDF prefixes never establish locality. */
    std::optional<std::size_t> distance(DeviceId a,DeviceId b) const {
        const auto& left=device(a).pcie_upstream_bdfs;
        const auto& right=device(b).pcie_upstream_bdfs;
        std::optional<std::size_t> result;
        for(std::size_t i=0;i<left.size();++i) for(std::size_t j=0;j<right.size();++j)
            if(!left[i].empty() && left[i]==right[j] && (!result || i+j+2<*result)) result=i+j+2;
        return result;
    }
    /** @return Actual group preference; a nonlocal/mixed domain gains no peer preference. */
    ObservedPeerGroupQuality quality(std::span<const DeviceId> group) const {
        if(!sameDomain(group)) return {};
        std::size_t edges=0;
        for(std::size_t i=0;i<group.size();++i) for(std::size_t j=i+1;j<group.size();++j)
            edges+=mutuallyAccessible(group[i],group[j]);
        const auto pairs=group.size()*(group.size()-1)/2;
        ObservedPeerGroupQuality result{edges==pairs?pairs:0,edges};
        for(std::size_t i=0;i<group.size();++i) for(std::size_t j=i+1;j<group.size();++j)
            if(mutuallyAccessible(group[i],group[j])) if(const auto hops=distance(group[i],group[j])) {
                ++result.located_pairs;result.negative_hops-=static_cast<std::int64_t>(*hops);
            }
        return result;
    }
    /**
     * @brief Choose disjoint actual groups with caller-authored positive sizes.
     * @param eligible Original endpoints already filtered for architecture/model/capacity.
     * @param sizes Exact semantic group sizes, retaining their authored order.
     * @return Best clique/island partition; deterministic UUID/ordinal ties.
     * @throws std::invalid_argument if no same-NUMA homogeneous partition exists.
     * No-P2P groups remain legal. Unselected devices remain in the inventory.
     */
    std::vector<std::vector<DeviceId>> select(std::span<const DeviceId> eligible,
        std::span<const std::size_t> sizes) const {
        std::vector<ObservedPeerStageRequest> request;
        for(const auto size:sizes)request.push_back({size,eligible});
        auto groups=select(request);
        if(!groups)throw std::invalid_argument("No original NUMA-owned homogeneous peer grouping satisfies requested sizes");
        return std::move(*groups);
    }
    /**
     * @brief Choose the best complete disjoint partition with stage-specific eligibility.
     * @param request Positive stage cardinalities and original candidate spans in authored order.
     * @return Best partition, or no value when the valid geometry has no complete assignment.
     * @throws std::invalid_argument For malformed cardinalities or contradictory original identities.
     *
     * Eligibility is metadata, not a second inventory. All endpoints resolve
     * through the same original owner, and overlapping candidate spans may not
     * reuse a selected physical GPU. Scoring and immutable UUID tie order are
     * identical to the uniform-eligibility API. Empty/partial peer observations
     * remain preferences rather than transport admission failures.
     */
    std::optional<std::vector<std::vector<DeviceId>>> select(
        std::span<const ObservedPeerStageRequest> request) const {
        if(request.empty())throw std::invalid_argument("Empty peer-group request");
        std::size_t required=0;
        std::vector<DeviceId> ordered;
        for(const auto& stage:request) {
            if(!stage.size || stage.size>std::numeric_limits<std::size_t>::max()-required)
                throw std::invalid_argument("Invalid peer-group cardinality");
            required+=stage.size;
            for(std::size_t i=0;i<stage.eligible.size();++i) {
                const auto id=stage.eligible[i];const auto& gpu=device(id);
                for(std::size_t j=0;j<i;++j)
                    if(id==stage.eligible[j] || gpu.uuid==device(stage.eligible[j]).uuid)
                        throw std::invalid_argument("Repeated original peer-group identity");
                if(std::find(ordered.begin(),ordered.end(),id)==ordered.end())ordered.push_back(id);
            }
        }
        std::sort(ordered.begin(),ordered.end(),[&](auto a,auto b){
            return std::tie(device(a).uuid,a.type,a.ordinal)<std::tie(device(b).uuid,b.type,b.ordinal);
        });
        for(std::size_t i=0;i<ordered.size();++i)for(std::size_t j=0;j<i;++j)
            if(device(ordered[i]).uuid==device(ordered[j]).uuid)
                throw std::invalid_argument("Repeated original peer-group identity");
        if(required>ordered.size())return std::nullopt;
        std::vector<std::vector<std::size_t>> candidates;
        for(const auto& stage:request) {
            if(stage.size>stage.eligible.size())return std::nullopt;
            std::vector<std::size_t> indices;
            for(std::size_t index=0;index<ordered.size();++index)
                if(std::find(stage.eligible.begin(),stage.eligible.end(),ordered[index])!=stage.eligible.end())
                    indices.push_back(index);
            candidates.push_back(std::move(indices));
        }
        std::vector<bool> used(ordered.size());
        std::vector<std::vector<DeviceId>> current,best;
        ObservedPeerGroupQuality bestQuality{};
        bool found=false;
        const auto visit=[&](const auto& self,std::size_t stage,ObservedPeerGroupQuality total)->void {
            if(stage==request.size()) {
                // Enumeration is lexicographic in immutable UUID order. Keep
                // the first exact tie instead of ordinal/discovery-order drift.
                if(!found || total>bestQuality) {found=true;bestQuality=total;best=current;}
                return;
            }
            std::vector<DeviceId> group;
            const auto& pool=candidates[stage];
            const auto choose=[&](const auto& chooseSelf,std::size_t next)->void {
                if(group.size()==request[stage].size) {
                    if(!sameDomain(group))return;
                    current.push_back(group);
                    auto score=total;score+=quality(group);self(self,stage+1,score);
                    current.pop_back();return;
                }
                // Reject only branches whose remaining candidate suffix cannot
                // fill this exact stage. This preserves exhaustive scoring/order
                // while avoiding impossible subsets for large exact cohorts.
                const auto remaining=request[stage].size-group.size();
                for(std::size_t i=next;i<pool.size() && pool.size()-i>=remaining;++i) {
                    const auto index=pool[i];if(used[index])continue;
                    used[index]=true;group.push_back(ordered[index]);
                    if(sameDomain(group))chooseSelf(chooseSelf,i+1);
                    group.pop_back();used[index]=false;
                }
            };
            choose(choose,0);
        };
        visit(visit,0,{});
        if(!found)return std::nullopt;
        return std::optional{std::move(best)};
    }
private:
    const RankInventory& original_; ///< Borrowed immutable observation, never a second hardware cache.
};
}
