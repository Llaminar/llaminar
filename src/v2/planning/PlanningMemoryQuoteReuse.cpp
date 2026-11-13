/**
 * @file PlanningMemoryQuoteReuse.cpp
 * @brief Lexically owned exact-input workspace memoization, without physical admission reuse.
 */
#include "planning/PlanningMemoryQuoteReuse.h"
#include <unordered_map>
#include <map>
#include <stdexcept>
namespace llaminar2
{
    /** @brief A partial hash only partitions buckets; complete default geometry equality rejects every mismatch. */
    struct WorkspaceQuoteHash
    {
        /** @return Cheap typed bucket selector, never a source/blob hash or identity substitute. */
        size_t operator()(const WorkspaceMemoryGeometry &geometry) const noexcept
        {
            size_t value=static_cast<size_t>(geometry.device.type);
            for(const int field:{geometry.device.ordinal,geometry.first_layer,geometry.last_layer,
                geometry.shard_index,geometry.total_shards,geometry.resident_graph_rows,geometry.max_context_rows})
                value=value*131u+static_cast<size_t>(field);
            return value;
        }
    };
    /** @brief Exact activation declaration; partial hash is only bucket selection. */
    struct ActivationQuoteKey
    {
        ActivationGraphMemoryGeometry geometry;
        DeviceId device;
        bool operator==(const ActivationQuoteKey &) const = default;
    };
    /** @brief Complete equality, rather than this inexpensive hash, authenticates policy and geometry. */
    struct ActivationQuoteHash
    {
        size_t operator()(const ActivationQuoteKey &key) const noexcept
        {
            size_t value=static_cast<size_t>(key.device.type);
            for(const int field:{key.device.ordinal,key.geometry.total_shards,key.geometry.first_layer,
                key.geometry.last_layer,key.geometry.resident_graph_rows,key.geometry.context_tokens})
                value=value*131u+static_cast<size_t>(field);
            return value;
        }
    };
    /** @brief One immutable source reference and pure quote population. */
    struct PlanningMemoryQuoteReuse::Impl
    {
        std::shared_ptr<const PlanningTensorIndex> source;
        std::unordered_map<WorkspaceMemoryGeometry,size_t,WorkspaceQuoteHash> workspace;
        std::unordered_map<WorkspaceMemoryGeometry,size_t,WorkspaceQuoteHash> prepared_service;
        std::map<int,MoEProjectionArenaGeometry> projection_arena;
        std::unordered_map<ActivationQuoteKey,size_t,ActivationQuoteHash> activation;
        PlanningMemoryQuoteReuseCounts counts;
        explicit Impl(std::shared_ptr<const PlanningTensorIndex> original):source(std::move(original)) {}
    };
    thread_local PlanningMemoryQuoteReuse *PlanningMemoryQuoteReuse::active_=nullptr;
    PlanningMemoryQuoteReuse::PlanningMemoryQuoteReuse(std::shared_ptr<const PlanningTensorIndex> source)
    {
        if(!source || active_)throw std::invalid_argument("Pure memory quotes require one nonnested original source scope");
        impl_=std::make_unique<Impl>(std::move(source));
        active_=this;
    }
    PlanningMemoryQuoteReuse::~PlanningMemoryQuoteReuse()
    {
        if(active_!=this)std::terminate();
        active_=nullptr;
    }
    PlanningMemoryQuoteReuseCounts PlanningMemoryQuoteReuse::counts() const noexcept {return impl_->counts;}
    std::shared_ptr<const PlanningTensorIndex> PlanningMemoryQuoteReuse::sourceIndex(const ModelMemoryProfile &profile)
    {
        if(!active_)return {};
        active_->impl_->source->requireProfile(profile);
        return active_->impl_->source;
    }
    size_t PlanningMemoryQuoteReuse::workspace(const ModelMemoryProfile &profile,const WorkspaceMemoryGeometry &geometry)
    {
        if(!active_)return WorkspaceMemoryEstimator::estimate(profile,geometry);
        auto &owner=*active_->impl_;
        owner.source->requireProfile(profile);
        if (geometry.source_tensor_index && geometry.source_tensor_index != owner.source)
            throw std::invalid_argument("Workspace quote geometry carries another original source index");
        ++owner.counts.workspace_requests;
        const auto found=owner.workspace.find(geometry);
        if(found!=owner.workspace.end()){++owner.counts.workspace_hits;return found->second;}
        const auto bytes=WorkspaceMemoryEstimator::estimate(profile,geometry);
        owner.workspace.emplace(geometry,bytes);
        ++owner.counts.workspace_builds;
        return bytes;
    }
    MoEProjectionArenaGeometry PlanningMemoryQuoteReuse::projectionArena(const ModelMemoryProfile &profile,int participants)
    {
        if(!active_)return MoEProjectionArenaGeometry::resolve(profile,participants);
        auto &owner=*active_->impl_;
        owner.source->requireProfile(profile);
        ++owner.counts.projection_arena_requests;
        const auto found=owner.projection_arena.find(participants);
        if(found!=owner.projection_arena.end()){
            ++owner.counts.projection_arena_hits;
            return found->second;
        }
        auto value=MoEProjectionArenaGeometry::resolve(profile,participants);
        owner.projection_arena.emplace(participants,value);
        ++owner.counts.projection_arena_builds;
        return value;
    }
    size_t PlanningMemoryQuoteReuse::preparedExpertService(const ModelMemoryProfile &profile,const WorkspaceMemoryGeometry &geometry)
    {
        if(!active_)return WorkspaceMemoryEstimator::estimatePreparedExpertService(profile,geometry);
        auto &owner=*active_->impl_;
        owner.source->requireProfile(profile);
        if (geometry.source_tensor_index && geometry.source_tensor_index != owner.source)
            throw std::invalid_argument("Prepared service quote geometry carries another original source index");
        ++owner.counts.prepared_service_requests;
        const auto found=owner.prepared_service.find(geometry);
        if(found!=owner.prepared_service.end()){
            ++owner.counts.prepared_service_hits;
            return found->second;
        }
        const auto bytes=WorkspaceMemoryEstimator::estimatePreparedExpertService(profile,geometry);
        owner.prepared_service.emplace(geometry,bytes);
        ++owner.counts.prepared_service_builds;
        return bytes;
    }

    size_t PlanningMemoryQuoteReuse::activation(const ModelMemoryProfile &profile,
        const ActivationGraphMemoryGeometry &geometry,DeviceId device)
    {
        if(!active_)return ActivationMemoryEstimator::estimate(profile,geometry,device);
        auto &owner=*active_->impl_;
        owner.source->requireProfile(profile);
        ++owner.counts.activation_requests;
        const ActivationQuoteKey key{geometry,device};
        const auto found=owner.activation.find(key);
        if(found!=owner.activation.end()){
            ++owner.counts.activation_hits;
            return found->second;
        }
        const auto bytes=ActivationMemoryEstimator::estimate(profile,geometry,device);
        owner.activation.emplace(key,bytes);
        ++owner.counts.activation_builds;
        return bytes;
    }

}
