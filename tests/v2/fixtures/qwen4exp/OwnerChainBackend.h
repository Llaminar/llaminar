/**
 * @file OwnerChainBackend.h
 * @brief Hardware-free backend API double that records exact allocator ownership.
 *
 * Host allocations stand in for device APIs so ownership chains can be proven
 * without a vendor runtime. This header defines only the double; executables
 * that replace the process-wide backend-resolution boundary do so separately
 * (see PooledImmutableReadGraphFixture.h), so ordinary test executables that
 * link the static core can reuse the double without duplicate symbols.
 */
#pragma once
#include "backends/BackendManager.h"
#include "mocks/MockBackend.h"
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
namespace llaminar2::test {
/** @brief Actual allocator records around aligned host API-double allocations. */
class OwnerChainBackend final:public MockBackend {
public:
    /** @brief Install a family without probing hardware or initializing a vendor runtime. */
    explicit OwnerChainBackend(DeviceType type):MockBackend(type),type_(type){setDeviceCount(8);}
    /** @brief No mocked physical owner may outlive its exact backend record. */
    ~OwnerChainBackend()override{if(!allocations_.empty() || !registrations_.empty() || !mapped_.empty())std::terminate();}
    /** @brief Allocate exactly the requested bytes; each live record has one original capture token. */
    void* allocate(size_t bytes,int ordinal)override{++allocationCalls;if(failAllocationCall && allocationCalls==*failAllocationCall)return nullptr;void* p=nullptr;if(posix_memalign(&p,64,bytes))return nullptr;allocations_.emplace(p,Record{bytes,ordinal,std::make_shared<const NativeAllocationCaptureIdentity>()});return p;}
    /** @brief Reject borrowed reclamation before freeing the actual host-double allocation. */
    void free(void* p,int ordinal)override{if(!p)return;auto it=allocations_.find(p);if(it==allocations_.end() || it->second.ordinal!=ordinal)std::terminate();NativeAllocationCaptureLease::requireUnborrowed(it->second.identity);allocations_.erase(it);std::free(p);}
    /** @brief Borrow the exact actual allocator record, including legal interior spans. */
    NativeAllocationCaptureLease retainNativeAllocation(const void* p,size_t bytes,int ordinal)override{
        const auto start=reinterpret_cast<uintptr_t>(p);for(const auto& [base,r]:allocations_){const auto first=reinterpret_cast<uintptr_t>(base);if(start>=first && start-first<r.bytes)return NativeAllocationCaptureLease::acquire(r.identity,base,r.bytes,DeviceId{type_,r.ordinal},p,bytes,DeviceId{type_,ordinal});}throw std::invalid_argument("Owner chain mock allocation record is foreign");
    }
    /** @brief Actual service mapped-allocation API double, preserving original64-byte protocol alignment. */
    void* allocateMapped(size_t bytes,int,void** alias)override{void* p=std::aligned_alloc(64,(bytes+63)/64*64);if(p)mapped_.insert(p);if(alias)*alias=p;return p;}
    /** @brief Final original source backing retires only after all actual worker/binding owners release it. */
    void freeMapped(void* p,int)override{if(mapped_.erase(p)!=1)std::terminate();std::free(p);}
    /** @brief Register only the original host mapping; the test never emulates a native peer route. */
    bool registerExternalMappedHostMemory(void* p,size_t,int,MappedHostRegistrationScope)override{return registrations_.insert(p).second;}
    /** @brief Expose the same host-double mapping to the matching backend. */
    bool externalMappedHostDevicePointer(void* p,int,void** alias)override{if(!alias || !registrations_.contains(p))return false;*alias=p;return true;}
    /** @brief Remove registration before its original host mapping is retired. */
    bool unregisterExternalMappedHostMemory(void* p,int)override{return registrations_.erase(p)==1;}
    /** @return Stable API-double runtime generation for this exact backend lifetime, never a vendor observation. */
    uint64_t deviceRuntimeGeneration(int ordinal)const override{return ordinal>=0 && ordinal<8?generation_:0;}
    /** @brief Inject one actual setup join failure; record exact ordinary backend calls otherwise. */
    bool streamWaitEvent(void* stream,void* event,int ordinal)override {
        if(failJoinOrdinal && *failJoinOrdinal==ordinal){failJoinOrdinal.reset();return false;}
        return MockBackend::streamWaitEvent(stream,event,ordinal);
    }
    std::size_t allocationCalls=0;
    std::optional<std::size_t> failAllocationCall;
    std::optional<int> failJoinOrdinal,failEventCreationOrdinal;
    /** @brief One failed native event creation is observed before any recording work. */
    void* createEvent(int ordinal)override {
        ++eventCreates;if(failEventCreationCall && eventCreates==*failEventCreationCall)return nullptr;
        if(failEventCreationOrdinal && *failEventCreationOrdinal==ordinal){failEventCreationOrdinal.reset();return nullptr;}
        return MockBackend::createEvent(ordinal);
    }
    std::size_t eventCreates=0;
    std::optional<std::size_t> failEventCreationCall;
    /** @return Actual live physical allocations, not a duplicate byte ledger. */
    std::size_t liveAllocations()const{return allocations_.size();}
    std::function<void()> beforeRetirementWait;
    bool waitForEvent(void* event,int ordinal)override{if(beforeRetirementWait)beforeRetirementWait();return MockBackend::waitForEvent(event,ordinal);}
    /** @return Transfer calls recorded across this backend's joined capture workers. */
    unsigned capturedTransferCount() const noexcept
    {
        return captured_transfers_.load(std::memory_order_relaxed);
    }
    /** @brief Record the actual original transfer entry under the executor's capture guard, without executing a wire. */
    bool enqueueCapturedTransferChannel(const CapturedTransferChannelDeviceBinding&,void*,const void*,int,void* stream)override {
        if (!isGraphCaptureActive() || !stream) return false;
        // Several original device workers share this one backend double.
        captured_transfers_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    /** @brief Cold protocol geometry only; no kernel/vendor API is submitted. */
    bool prepareCapturedTransferChannelKernels(int,int timeout,uint64_t* ticks)override{if(timeout!=30000 || !ticks)return false;*ticks=42;return true;}
    /** @brief Materialize only the original cursor setup bytes using the explicit host API double. */
    bool copyDeviceVisibleRegionByKernelOnStream(void* dst,const void* src,size_t bytes,int,void* stream)override{if(!stream || reinterpret_cast<uintptr_t>(stream)<=2)return false;std::memcpy(dst,src,bytes);return true;}
    /** @brief Observe only cold setup-event readiness. */
    bool queryEvent(void*,int,bool* ready)override{*ready=true;return true;}
private:
    std::atomic<unsigned> captured_transfers_{0};
    struct Record{size_t bytes;int ordinal;std::shared_ptr<const NativeAllocationCaptureIdentity> identity;};
    inline static std::atomic<uint64_t> nextGeneration_{1};
    const uint64_t generation_=nextGeneration_.fetch_add(1);
    DeviceType type_;std::map<void*,Record> allocations_;std::set<void*> registrations_,mapped_;
};
} // namespace llaminar2::test
