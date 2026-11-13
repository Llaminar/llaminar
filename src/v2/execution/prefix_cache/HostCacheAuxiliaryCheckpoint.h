/**
 * @file HostCacheAuxiliaryCheckpoint.h
 * @brief Preadmitted CPU rollback slots for cache-owned model endpoint state.
 *
 * These slots contain accepted tails and lexical histories, never compressed
 * pool banks. Shared snapshot ownership prevents reuse until every reader has
 * retired. Physical memory admission remains with the orchestrator's PMA lease;
 * this pool neither allocates payload storage nor makes capacity decisions.
 */
#pragma once
#include "kernels/kvcache/CacheAuxiliaryState.h"
#include <memory>
#include <stdexcept>
#include <vector>

namespace llaminar2
{
/** @brief Immutable CPU endpoint checkpoint, distinct from a compact prefix archive. */
struct HostCacheAuxiliaryCheckpoint
{
    int cache_depth = -1; ///< Main cache is -1; predictor caches use their exact depth.
    int sequence_index = -1;
    CacheAuxiliaryStateKind kind = CacheAuxiliaryStateKind::QwenFlashNext;
    const ICacheAuxiliaryState *owner_identity = nullptr;
    std::size_t bytes = 0;
    std::shared_ptr<void> physical_memory_lease;
    std::shared_ptr<std::vector<std::uint8_t>> storage;

    /** @return Exact checkpoint metadata/storage identity; no endpoint data is read. */
    bool operator==(const HostCacheAuxiliaryCheckpoint &) const = default;

    /** @brief Reject foreign owners, request slots and incompatible endpoint layouts. */
    void validate(const ICacheAuxiliaryState &owner,int sequence) const
    {
        if(owner.device().is_gpu() || owner_identity!=&owner || sequence_index!=sequence ||
           kind!=owner.kind() || !physical_memory_lease || !storage || !bytes ||
           bytes!=owner.checkpointCapacity(sequence) || storage->size()<bytes)
            throw std::invalid_argument("CPU auxiliary rollback checkpoint has a foreign owner or incompatible layout");
    }
    /** @brief Restore accepted endpoint bytes after canonical logical frontiers are restored. */
    void restore(ICacheAuxiliaryState &owner,int sequence) const
    {
        validate(owner,sequence);
        owner.restoreCheckpoint(sequence,{storage->data(),bytes,CacheAuxiliaryMemoryDomain::Host},nullptr);
    }
};

/** @brief Setup-sealed borrower of canonical CPU rollback allocations. */
class HostCacheAuxiliaryCheckpointPool
{
    struct Slot
    {
        const ICacheAuxiliaryState *owner;
        std::shared_ptr<void> lease;
        std::shared_ptr<std::vector<std::uint8_t>> storage;
    };
    std::vector<Slot> slots_;
    bool sealed_ = false;
public:
    /** @brief Bind one already-admitted allocation before any checkpoint can execute. */
    void add(const ICacheAuxiliaryState &owner,std::shared_ptr<void> lease,
        std::shared_ptr<std::vector<std::uint8_t>> storage)
    {
        if(sealed_ || owner.device().is_gpu() || !lease || !storage || storage->empty())
            throw std::invalid_argument("CPU auxiliary rollback storage must be admitted before pool sealing");
        for(const auto &slot:slots_)if(slot.storage.get()==storage.get())
            throw std::invalid_argument("CPU auxiliary rollback allocation is already bound");
        slots_.push_back({&owner,std::move(lease),std::move(storage)});
    }
    /** @brief Finish setup; a full pool is a fatal admission error rather than a hot allocation. */
    void seal() noexcept { sealed_=true; }
    /** @brief Capture into an unborrowed admitted slot without resizing or allocating payload bytes. */
    HostCacheAuxiliaryCheckpoint capture(int depth,int sequence,const ICacheAuxiliaryState &owner) const
    {
        if(!sealed_ || owner.device().is_gpu())
            throw std::logic_error("CPU auxiliary rollback requires a sealed CPU storage pool");
        const auto bytes=owner.checkpointCapacity(sequence);
        if(!bytes)throw std::invalid_argument("CPU auxiliary rollback owner has no endpoint state");
        for(const auto &slot:slots_)if(slot.owner==&owner && slot.storage.use_count()==1 && slot.storage->size()>=bytes) {
            HostCacheAuxiliaryCheckpoint checkpoint{depth,sequence,owner.kind(),&owner,bytes,slot.lease,slot.storage};
            owner.captureCheckpoint(sequence,{checkpoint.storage->data(),bytes,CacheAuxiliaryMemoryDomain::Host},nullptr);
            return checkpoint;
        }
        throw std::runtime_error("CPU auxiliary rollback has no unborrowed preadmitted storage slot");
    }
};
}
