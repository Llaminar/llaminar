/**
 * @file PooledImmutableNativeMembership.h
 * @brief Derive native transport membership without changing model-reader ownership.
 *
 * The original reader order remains intact; same-vendor storage-only owners
 * append once. Cross-vendor owners retain their explicit TransferEngine edges.
 * This immutable declaration owns no live count, allocator balance or runtime
 * discovery. Planning and materialization consume the identical union law.
 */
#pragma once
#include "PooledImmutableNativeTransfer.h"
#include "planning/PhysicalMemoryBOM.h"
#include <algorithm>

namespace llaminar2::qwen4exp {
/** @brief One authored physical owner, including zero-capacity membership. */
struct PooledNativeStorageMember final {
    PhysicalMemoryAllocatorIdentity allocator;
    bool positive;
};
/** @brief Actual homogeneous source/reader/storage union in one original rank. */
class PooledImmutableNativeMembership final {
public:
    /**
     * @brief Freeze the original reader order and append missing native owners.
     * @param source Original coordinator and query participant.
     * @param owners All physical storage owners, including cross-vendor and C0.
     * @param readers Actual compute-reader order; empty denotes the source alone.
     * @throws std::invalid_argument For foreign ranks, duplicated readers or a missing source.
     */
    PooledImmutableNativeMembership(PhysicalMemoryAllocatorIdentity source,
        std::span<const PooledNativeStorageMember> owners,
        std::span<const PhysicalMemoryAllocatorIdentity> readers={})
    {
        if(!source.valid() || !source.device.is_gpu())
            throw std::invalid_argument("Native pool membership requires its original GPU coordinator");
        if(readers.empty())members_.push_back(source);
        else for(const auto reader:readers) {
            if(!reader.valid() || reader.world_rank!=source.world_rank || reader.device.type!=source.device.type ||
                std::find(members_.begin(),members_.end(),reader)!=members_.end())
                throw std::invalid_argument("Native pool reader changed original rank/backend/unique membership");
            members_.push_back(reader);
        }
        const auto found=std::find(members_.begin(),members_.end(),source);
        if(found==members_.end())throw std::invalid_argument("Native pool transport omitted its original source reader");
        coordinator_=std::size_t(found-members_.begin());
        readers_.assign(members_.size(),1);positive_.assign(members_.size(),0);
        for(const auto& owner:owners) {
            if(!owner.allocator.valid() || !owner.allocator.device.is_gpu() || owner.allocator.world_rank!=source.world_rank)
                throw std::invalid_argument("Native pool storage cannot bypass original MPI rank ownership");
            if(owner.allocator.device.type!=source.device.type || owner.allocator==source)continue;
            auto member=std::find(members_.begin(),members_.end(),owner.allocator);
            const auto coordinate=std::size_t(member-members_.begin());
            if(member==members_.end()) {
                members_.push_back(owner.allocator);positive_.push_back(0);readers_.push_back(0);
            }
            positive_[coordinate]=owner.positive;
        }
    }
    /** @return Full native roster; it never becomes model compute membership. */
    std::span<const PhysicalMemoryAllocatorIdentity> members()const noexcept{return members_;}
    /** @return Whether any actual remote native endpoint exists. */
    bool remote()const noexcept{return members_.size()>1;}
    /** @return Exact source coordinate in the native roster. */
    std::size_t coordinator()const noexcept{return coordinator_;}
    /** @brief Quote only original owner and result message slots, with a distinct C0 floor. */
    PooledNativeCompletionLayout layout(PooledNativeReaderMessageProfile profile,bool floor=false)const {
        if(!remote())throw std::invalid_argument("A local-only pool has no native completion bank");
        if(floor)return {members_.size(),coordinator_,std::vector<std::uint8_t>(members_.size(),0),profile,readers_};
        return {members_.size(),coordinator_,positive_,profile,readers_};
    }
private:
    std::vector<PhysicalMemoryAllocatorIdentity> members_;
    std::vector<std::uint8_t> positive_,readers_;
    std::size_t coordinator_;
};
} // namespace llaminar2::qwen4exp
