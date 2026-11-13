/**
 * @file GPUCaptureDependencyRoster.h
 * @brief Context-owned, geometry-sized internal capture-edge event roster.
 *
 * Default event record/wait captures store dependency edges, not external event
 * nodes. A serial recording interval borrows these handles only until all
 * matching waits have been recorded. Retained graphs own their DAG edges and
 * keep no active lease. External publication and terminal events never enter
 * the internal pool. One separate cold upload event is reused only after its
 * actual copy completion; it is never an internal edge or retained terminal.
 * Original context destruction retires actual native handles before
 * completing the same PMA obligations.
 */
#pragma once
#include "backends/DeviceId.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <memory>
#include <mutex>
#include <atomic>
#include <utility>
#include <vector>
namespace llaminar2 {
class IBackend;
class IWorkerGPUContext;
class NativeOrderingEventGroup;
class PreparedGPUContextUsePermit;
class TransferEngine;
/** @brief One context's actual peak Batch fork/join event geometry. */
class GPUCaptureDependencyRoster final {
public:
    /** @brief Construct/grow only from the original admitted source before capture. */
    static std::shared_ptr<GPUCaptureDependencyRoster> create(const PreparedGPUContextUsePermit&,std::size_t lanes,std::size_t shared_groups);
    /** @brief Append new roles cold; existing graph-observed addresses never move. */
    void grow(const PreparedGPUContextUsePermit&,std::size_t lanes,std::size_t shared_groups);
    /** @brief Destroy each original event before its certified obligation closes. */
    ~GPUCaptureDependencyRoster()noexcept;
    /** @brief Move-only recording exclusion; it owns no native handle or GPU state. */
    class CaptureBorrow final {
    public:
        CaptureBorrow(const CaptureBorrow&)=delete;
        CaptureBorrow& operator=(const CaptureBorrow&)=delete;
        CaptureBorrow(CaptureBorrow&& other)noexcept:active_(std::exchange(other.active_,nullptr)){}
        CaptureBorrow& operator=(CaptureBorrow&&)=delete;
        /** @brief Release internal event reuse only after every recording-side wait was submitted. */
        ~CaptureBorrow(){if(active_)active_->store(false,std::memory_order_release);}
    private:
        friend class GPUCaptureDependencyRoster;
        explicit CaptureBorrow(std::atomic<bool>& active):active_(&active){}
        std::atomic<bool>* active_;
    };
    /** @return Exclusive construction interval; caller also holds the original whole-capture submission borrow. */
    CaptureBorrow borrow(std::size_t lanes,std::size_t shared_groups);
    /** @return Separately owned cold publication event, reused only after exact completion. */
    void* coldPublicationEvent()const noexcept{return cold_publication_;}
    /** @brief Reject an independent authority before allocating/uploading metadata. */
    void requireOriginalMemory(const std::shared_ptr<PhysicalMemoryAuthority>&)const;
    /** @return Context-owned main-to-lane capture edge event. */
    void* producer()const noexcept{return producer_;}
    /** @return Context-owned metadata-admission capture edge for this actual lane. */
    void* admission(std::size_t lane)const{return admissions_.at(lane);}
    /** @return Context-owned publication edge for this actual physical payload group. */
    void* shared(std::size_t group)const{return shared_.at(group);}
private:
    friend class TransferEngine;
    explicit GPUCaptureDependencyRoster(const PreparedGPUContextUsePermit&);
    /** @return One nonblocking exclusion token; reentrant/concurrent recording rejects immediately. */
    CaptureBorrow acquire();
    /** @return One original native event after its exact PMA unit has been claimed. */
    void* append(const PreparedGPUContextUsePermit&);
    DeviceId device_;std::uint64_t generation_;IBackend* backend_;
    const void* memory_;IWorkerGPUContext* worker_;
    void* producer_=nullptr;
    void* cold_publication_=nullptr; ///< External cold-copy completion, never borrowed as a capture edge.
    std::mutex cold_publication_mutex_; ///< Reuse follows actual copy completion, not graph recording.
    std::vector<void*> admissions_,shared_;
    std::vector<std::unique_ptr<NativeOrderingEventGroup>> ownership_;
    std::atomic<bool> recording_{false};
};
}
