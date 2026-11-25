/**
 * @file PhysicalPagePool.h
 * @brief PMA-issued CPU page backing and bounded unpublished growth grants.
 *
 * Private append plans and completed CPU byte publications are distinct.
 * These are not yet logical KV-table or operation-readable native views;
 * no kernel/transfer may consume them as such. Physical bytes are claimed by the
 * existing CPUStorageRoot path, not by a second free-byte ledger. Logical
 * commitments protect completion growth without assigning private frame ranges.
 */
#pragma once
#include "memory/CPUStorageRoot.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace llaminar2
{
/** @brief Exact immutable pool dimensions; native format factories supply frame bytes. */
struct PhysicalPagePoolGeometry
{
    std::size_t frames = 0;
    std::size_t frame_bytes = 0;
    std::size_t request_records = 0;
    std::size_t max_pages_per_request = 0;
    /** @brief Reject empty, overflowing or internally inconsistent declarations. */
    void validate() const;
};

/**
 * @brief Single-thread CPU suballocation authority issued only by PMA.
 *
 * Host CPU ownership is intentional. This is not the GPU free-list authority:
 * a future GPU adapter retains immutable host geometry and moves all mutable
 * ownership below to device storage. No CPU mirror may drive GPU decisions.
 */
class PhysicalPagePool final : public std::enable_shared_from_this<PhysicalPagePool>
{
public:
    /** @brief Non-reused incarnation of one privately granted frame. */
    struct FrameHandle
    {
        std::uint64_t pool_identity = 0;
        std::size_t index = 0;
        std::uint64_t generation = 0;
        bool operator==(const FrameHandle &) const = default;
    };
    /** @brief Retain an assigned private incarnation through a bounded setup borrower; no readability implied. */
    class FramePin final
    {
    public:
        FramePin() = default;
        ~FramePin();
        FramePin(FramePin &&other) noexcept;
        FramePin &operator=(FramePin &&other) noexcept;
        FramePin(const FramePin &) = delete;
        FramePin &operator=(const FramePin &) = delete;
        /** @return Exact pinned incarnation; not a native readable view. */
        FrameHandle identity() const noexcept { return handle_; }
    private:
        friend class PhysicalPagePool;
        FramePin(std::shared_ptr<PhysicalPagePool> pool, FrameHandle handle);
        void release() noexcept;
        std::shared_ptr<PhysicalPagePool> pool_;
        FrameHandle handle_;
    };

    /** @brief Registered synchronous CPU consumer; registration itself pins the old content. */
    class ReadAccess final
    {
    public:
        ReadAccess() = default;
        ~ReadAccess();
        ReadAccess(ReadAccess &&other) noexcept;
        ReadAccess &operator=(ReadAccess &&other) noexcept;
        ReadAccess(const ReadAccess &) = delete;
        ReadAccess &operator=(const ReadAccess &) = delete;
        /** @brief Start the original CPU access; it must complete before destruction. */
        std::span<const std::byte> beginCPU();
        /** @brief Retire an actually completed synchronous CPU consumer, never a queued GPU operation. */
        void completeCPU();
    private:
        friend class PhysicalPagePool;
        enum class Phase : std::uint8_t { Registered, Executing, Completed };
        ReadAccess(std::shared_ptr<PhysicalPagePool> pool, FrameHandle handle,
            std::uint64_t version, std::size_t bytes);
        void release() noexcept;
        std::shared_ptr<PhysicalPagePool> pool_;
        FrameHandle handle_;
        std::uint64_t version_ = 0;
        std::size_t bytes_ = 0;
        Phase phase_ = Phase::Registered;
    };

    /** @brief Immutable completed CPU byte publication, retaining its exact frame incarnation. */
    class CommittedPage final
    {
    public:
        CommittedPage() = default;
        ~CommittedPage();
        CommittedPage(CommittedPage &&other) noexcept;
        CommittedPage &operator=(CommittedPage &&other) noexcept;
        CommittedPage(const CommittedPage &) = delete;
        CommittedPage &operator=(const CommittedPage &) = delete;
        /** @return Only the completed valid byte extent, never unwritten capacity. */
        std::span<const std::byte> read() const;
        /** @brief Register one new consumer before view closure or request retirement. */
        ReadAccess registerRead() const;
        /** @return The immutable content version; aborted write versions are never reused. */
        std::uint64_t version() const noexcept { return version_; }
    private:
        friend class PhysicalPagePool;
        CommittedPage(std::shared_ptr<PhysicalPagePool> pool, FrameHandle handle,
            std::uint64_t version, std::size_t bytes);
        void release() noexcept;
        std::shared_ptr<PhysicalPagePool> pool_;
        FrameHandle handle_;
        std::uint64_t version_ = 0;
        std::size_t bytes_ = 0;
    };
    /** @brief Exclusive CPU append destination; not a readable page or committed history. */
    class AppendPlan final
    {
    public:
        AppendPlan() = default;
        ~AppendPlan();
        AppendPlan(AppendPlan &&other) noexcept;
        AppendPlan &operator=(AppendPlan &&other) noexcept;
        AppendPlan(const AppendPlan &) = delete;
        AppendPlan &operator=(const AppendPlan &) = delete;
        /** @brief Copy exactly the granted new byte extent; old committed bytes are never overwritten. */
        void copyFrom(std::span<const std::byte> source);
        /** @brief Publish only after the exact CPU copy completed; cancellation cannot publish. */
        CommittedPage commit();
    private:
        friend class PhysicalPagePool;
        AppendPlan(std::shared_ptr<PhysicalPagePool> pool, FrameHandle handle,
            std::uint64_t version, std::size_t start, std::size_t count);
        void release() noexcept;
        std::shared_ptr<PhysicalPagePool> pool_;
        FrameHandle handle_;
        std::uint64_t version_ = 0;
        std::size_t start_ = 0;
        std::size_t count_ = 0;
        bool copied_ = false;
    };

    /** @brief Move-only unpublished request commitment; destruction rolls it back. */
    class Grant final
    {
    public:
        Grant() = default;
        ~Grant();
        Grant(Grant &&other) noexcept;
        Grant &operator=(Grant &&other) noexcept;
        Grant(const Grant &) = delete;
        Grant &operator=(const Grant &) = delete;
        /** @return False for a moved-from or refused grant. */
        bool valid() const noexcept { return pool_ != nullptr; }
        /** @brief Convert protected growth into one private frame without a physical allocation. */
        FrameHandle assignFrame();
        /** @brief Retain an original private frame without exposing payload or readiness. */
        FramePin pinFrame(FrameHandle handle);
        /** @brief Prepare an exact private append, or defer while an existing borrower pins the frame. */
        std::optional<AppendPlan> prepareAppend(FrameHandle handle, std::size_t bytes);
        /** @brief Freeze admission of new readers before joining the existing access frontier. */
        void closeReaders(FrameHandle handle);
        /** @return Assigned frame identities in logical append order, never physical stride order. */
        std::span<const FrameHandle> assignedFrames() const;
        /** @return Remaining guaranteed private growth for this unpublished request. */
        std::size_t remainingCommitment() const;
    private:
        friend class PhysicalPagePool;
        Grant(std::shared_ptr<PhysicalPagePool> pool, std::size_t record, std::uint64_t generation);
        /** @brief Release exactly this unpublished incarnation, once. */
        void release() noexcept;
        std::shared_ptr<PhysicalPagePool> pool_;
        std::size_t record_ = 0;
        std::uint64_t generation_ = 0;
    };
    /** @brief One storage-class demand in an unpublished same-authority transaction. */
    struct Demand
    {
        std::shared_ptr<PhysicalPagePool> pool;
        std::size_t pages = 0;
    };
    /**
     * @brief Acquire every unpublished growth grant or roll all new grants back.
     * @param demands Real same-PMA storage owners, in caller-declared order.
     * @param results Preallocated empty output cells; no allocation occurs here.
     * @return False on ordinary pressure, with all cells empty and original owners unchanged.
     * @throws std::invalid_argument For foreign authority, duplicate pool, or invalid demand/output.
     * @note Issued generation numbers are never restored on abort.
     */
    static bool tryAdmitTogether(std::span<const Demand> demands,
        std::span<std::optional<Grant>> results);

    /** @return Exact materialized physical BOM for this geometry and placement. */
    static std::size_t physicalBytes(PhysicalPagePoolGeometry geometry, DeviceId device,
        CPUStoragePlacement placement = CPUStoragePlacement::local());
    /**
     * @brief Atomically protect maximum private growth without assigning frames.
     * @param pages Complete declared maximum, including caller-declared transients.
     * @return Empty on ordinary request-record or commitment pressure; no mutation.
     * @throws std::invalid_argument For zero/excess authored demand.
     */
    std::optional<Grant> tryAdmit(std::size_t pages);
    /** @return Immutable dimensions; not an observation of device state. */
    const PhysicalPagePoolGeometry &geometry() const noexcept { return geometry_; }

private:
    friend class PhysicalMemoryAuthority;
    /** @brief One CPU frame owner; no public raw transition or payload pointer. */
    struct Frame
    {
        std::uint64_t generation = 0;
        std::size_t record = 0;
        std::size_t pins = 0;
        std::uint64_t next_content_version = 0;
        std::uint64_t committed_version = 0;
        std::size_t valid_bytes = 0;
        bool writer = false;
        bool readers_closed = false;
        bool owned = false;
        bool retiring = false;
    };
    /** @brief One bounded request commitment, distinct from physical frame capacity. */
    struct Request { std::uint64_t generation = 0; std::size_t remaining = 0; std::size_t assigned = 0; bool live = false; };
    PhysicalPagePool(std::shared_ptr<PhysicalMemoryAuthority> authority, DeviceId device,
        PhysicalPagePoolGeometry geometry, CPUStoragePlacement placement, std::uint64_t pool_identity);
    /** @brief Validate one original grant incarnation before observing its metadata. */
    const Request &require(const Grant &grant) const;
    /** @brief Retire only unpublished private ownership; no native access API exists yet. */
    void release(const Grant &grant) noexcept;
    FrameHandle assign(const Grant &grant);
    FramePin pin(const Grant &grant, FrameHandle handle);
    void unpin(FrameHandle handle) noexcept;
    std::optional<AppendPlan> prepareAppend(const Grant &grant, FrameHandle handle, std::size_t bytes);
    Frame &requireFrame(FrameHandle handle);
    ReadAccess registerRead(const CommittedPage &page);
    std::shared_ptr<PhysicalMemoryAuthority> authority_; ///< Exact issuer, never an independent capacity observation.
    const std::uint64_t pool_identity_; ///< Non-reused process incarnation, issued at setup only.
    PhysicalPagePoolGeometry geometry_;
    AlignedVector<std::byte> backing_;
    AlignedVector<Frame> frames_;
    AlignedVector<Request> requests_;
    AlignedVector<FrameHandle> request_frames_;
    std::size_t protected_pages_ = 0; ///< Assigned plus remaining commitments, not physical bytes.
};
}
