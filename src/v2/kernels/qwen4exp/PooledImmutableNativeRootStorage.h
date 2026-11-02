/**
 * @file PooledImmutableNativeRootStorage.h
 * @brief Internal one-authority materialization and immutable native allocation seals.
 *
 * Only prepared owner/coordinator factories construct this holder. It stores
 * immutable root geometry and actual leases, never live cache/model counters.
 * The setup state is submission ownership, not a host execution-state mirror.
 * Root aliases retain the original group commitment through final Tensor event
 * destruction; handle teardown alone does not invent physical reset retirement.
 */
#pragma once
#include "PooledImmutableNativeStorage.h"
#include "backends/NativeOrderingEventGroup.h"
#include "loaders/ImmutableLookupNativeOwnership.h"
#include "execution/qwen4exp/Qwen4ExpPreparedStorage.h"
#include "transfer/TransferEngine.h"
#include <array>
#include <functional>
#include <mutex>
#include <optional>
namespace llaminar2::qwen4exp {class PooledImmutableSourcePreparedStorage;}
namespace llaminar2::test {struct PooledImmutableNativeRootStorageReplayAccess;}
namespace llaminar2::qwen4exp::detail
{
/** @brief One exact physical allocation family, shared through aliasing original Tensor handles. */
class PooledNativeRootStorage final:public std::enable_shared_from_this<PooledNativeRootStorage>
{
public:
    /**
     * @brief Claim and materialize exactly three declared roots, omitting only the optional zero-byte cache.
     * @param allocator Exact original rank/device.
     * @param bytes Actual canonical allocation requests, not merely final live subrange ends.
     * @param identity Complete immutable geometry identity.
     * @param memory Sole original admitted PMA.
     * @param stream Exact non-default ordinary setup stream.
     * @param ordering_program Closed actual root initialization program from the original native plan.
     * @return Actual lease owner; no metadata kernels or admission values are initialized here.
     */
    static std::shared_ptr<PooledNativeRootStorage> create(PhysicalMemoryAllocatorIdentity allocator,
        std::array<std::size_t,3> bytes,std::uint64_t identity,std::shared_ptr<PhysicalMemoryAuthority> memory,void* stream,
        NativeOrderingEventProgram ordering_program);
    /** @brief Drain setup, destroy all canonical publication events, then close original group use before claims retire. */
    ~PooledNativeRootStorage()noexcept;
    /** @return Original retained Tensor aliases. */
    PooledImmutableNativeRoots roots()const;
    /** @brief Require exact still-live native storage without GPU queries, allocation or mutable reads. */
    void require()const;
    /** @brief Reject another PMA even when its immutable certificate has equal bytes.
     * @param memory Actual original authority retained by the model/source owner. */
    void requireAuthority(const std::shared_ptr<const PhysicalMemoryAuthority>& memory)const;
    /** @return Exact actual still-retained root claims, without native state reads. */
    ImmutableLookupNativeOwnerObservation retainedOwnerObservation()const;
    /** @return Complete immutable root/type/backend/runtime/claim identity. */
    std::uint64_t identity()const;
    /** @brief Join the exact original setup event on a consumer stream; never wait on the host. */
    void acquire(void* stream)const;
    /**
     * @brief Submit once-only initialization and publish exact producer streams.
     * @param backend Expected native backend, checked before invoking code.
     * @param stream Actual non-default setup producer.
     * @param submit Fixed metadata submission, invoked only outside capture.
     * @note Failed submission is sticky and its partially written roots remain retained through setup retirement.
     */
    void initialize(DeviceType backend,void* stream,const std::function<void()>& submit);
    /** @brief Require a successfully submitted setup; completion belongs to event ordering. */
    void requireInitialized()const;
    /** @return Existing native address with exact immutable extent; no coherence transition occurs. */
    std::byte* address(std::size_t index)const;
private:
    friend struct ::llaminar2::test::PooledImmutableNativeRootStorageReplayAccess;
    friend class ::llaminar2::qwen4exp::PooledImmutableOwnerPreparedStorage;
    friend class ::llaminar2::qwen4exp::PooledImmutableCoordinatorPreparedStorage;
    friend class ::llaminar2::qwen4exp::PooledImmutableSourcePreparedStorage;
    /**
     * @brief Shared actual materialization body consuming a preclaimed closed group.
     * @param allocator Original rank/device. @param bytes Exact canonical root extents.
     * @param identity Complete immutable root geometry identity. @param memory Original same-PMA owner.
     * @param stream Exact setup producer. @param program Closed matching root initialization program.
     * @param group Original sealed claim, authenticated before any root allocation.
     * @return The same root owner used by production, with all original allocations/borrows.
     * @throws std::exception For foreign claims, stale provenance or allocation failure.
     */
    static std::shared_ptr<PooledNativeRootStorage> createWithOrderingGroup(PhysicalMemoryAllocatorIdentity allocator,
        std::array<std::size_t,3> bytes,std::uint64_t identity,std::shared_ptr<PhysicalMemoryAuthority> memory,void* stream,
        NativeOrderingEventProgram program,std::unique_ptr<NativeOrderingEventGroup> group);
    /** @brief Factories construct one immutable original geometry/authority association. */
    PooledNativeRootStorage(PhysicalMemoryAllocatorIdentity allocator,std::array<std::size_t,3> bytes,
        std::uint64_t identity,std::shared_ptr<PhysicalMemoryAuthority> memory,NativeOrderingEventProgram program)
        :allocator_(allocator),bytes_(bytes),geometry_(identity),memory_(std::move(memory)),ordering_program_(program){}
    /** @brief Reject default/sentinel handles before native work. */
    static void requireStream(void* stream);
    PhysicalMemoryAllocatorIdentity allocator_;
    std::array<std::size_t,3> bytes_;
    std::uint64_t geometry_;
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    NativeOrderingEventProgram ordering_program_;
    /** @brief Original opaque commitment outlives every canonical Tensor publication event. */
    std::unique_ptr<NativeOrderingEventGroup> ordering_group_;
    std::array<PhysicalMemoryAllocationLease,3> claims_;
    std::array<std::unique_ptr<TensorBase>,3> tensors_;
    std::optional<Qwen4ExpPreparedStorage> prepared_;
    std::array<std::shared_ptr<const CapturedDeviceStorageLease>,3> captured_;
    IBackend* backend_=nullptr;
    void* setup_event_=nullptr;
    void* setup_stream_=nullptr;
    bool setup_recorded_=false;
    /** @brief Submission ownership only; live native state stays entirely in native roots. */
    enum class Setup {Declared,Submitted,Failed};
    Setup setup_=Setup::Declared;
    mutable std::mutex mutex_;
};
}
