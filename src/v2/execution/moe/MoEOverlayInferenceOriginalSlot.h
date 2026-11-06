/**
 * @file MoEOverlayInferenceOriginalSlot.h
 * @brief Immutable native resource bindings minted by the original ticket protocol.
 *
 * Native terminal events and mapped invocation metadata use the SAME original
 * protocol slot roster. These types add no allocator, admission policy or free
 * list. Only a protocol that already accepted and submitted the exact ticket
 * can name its native slot; resource consumers retain that immutable identity
 * until aggregate graph retirement. Device execution state is not represented
 * by this host-side ticket/resource binding.
 */
#pragma once
#include "MoEOverlayInferenceTransaction.h"
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    /**
     * @brief Cold native resource geometry derived only from the canonical retained-depth rule.
     * This is a BOM declaration, not permission to submit or reuse a protocol slot.
     */
    class MoEOverlayInferenceOriginalSlotGeometry final
    {
    public:
        /**
         * @brief Declare the same fixed roster that the production protocol will own.
         * @param depth Canonical retained draft capacity, not the current dynamic depth.
         * @return Exact original roster geometry for cold PMA/resource admission.
         * @throws std::invalid_argument For a negative retained capacity.
         */
        [[nodiscard]] static MoEOverlayInferenceOriginalSlotGeometry forRetainedDraftCapacity(int depth)
        {
            if (depth < 0) throw std::invalid_argument("Original native slots require nonnegative retained draft capacity");
            return MoEOverlayInferenceOriginalSlotGeometry(depth);
        }
        /** @return Sole canonical original protocol capacity, without duplicated arithmetic. */
        [[nodiscard]] std::size_t slotCount() const noexcept
        { return moeOverlayInferenceTransactionSlotCount(depth_); }
        /** @return Original retained ceiling authenticated when runtime binds the roster. */
        [[nodiscard]] int retainedDraftCapacity() const noexcept { return depth_; }
        bool operator==(const MoEOverlayInferenceOriginalSlotGeometry &) const = default;
    private:
        /** @brief Retain only an already-checked canonical depth input. */
        explicit MoEOverlayInferenceOriginalSlotGeometry(int depth) : depth_(depth) {}
        int depth_;
    };

    /** Original fixed roster retained by native resources, never a second ring. */
    class MoEOverlayInferenceOriginalSlotDomain final
    {
        friend class MoEOverlayInferenceTransactionProtocol;
    public:
        MoEOverlayInferenceOriginalSlotDomain(const MoEOverlayInferenceOriginalSlotDomain &) = delete;
        MoEOverlayInferenceOriginalSlotDomain &operator=(const MoEOverlayInferenceOriginalSlotDomain &) = delete;
        /**
         * @brief Check retained original slot ownership before native resource publication.
         * @param slot Checked original publication, including the complete accepted ticket.
         * @throws std::logic_error For a foreign, retired, replaced or withdrawn original slot.
         * @note Aborted submitted work retains its slot for native drain; this
         *       ownership check never certifies semantic execution success.
         */
        void requireCurrent(const MoEOverlayInferenceOriginalSlot &slot) const;
        /**
         * @brief Authorize new immutable native admission only while its original slot is Submitted and Active.
         * @param slot Same checked original publication, never a raw ordinal.
         * @throws std::logic_error For aborted, terminal, replaced or withdrawn admission.
         * This is separate from retained resource ownership needed after abort.
         */
        void requireActiveAdmission(const MoEOverlayInferenceOriginalSlot &slot) const;
        /**
         * @brief Serialize a short immutable metadata publication with original Abort/retire.
         * @param slot Same checked Active/Submitted original token.
         * @param publish CPU-only immutable metadata writes; no native wait/SDK operation.
         * @return Callback result after publication completes under original ownership.
         * @throws std::logic_error If original admission is no longer active/current.
         */
        template<class Publish> decltype(auto) withActiveAdmission(
            const MoEOverlayInferenceOriginalSlot& slot,Publish&& publish) const;
        /** @return Exact admitted slot count from the original protocol instance. */
        [[nodiscard]] std::size_t slotCount() const noexcept { return count_; }
        /**
         * @brief Authenticate runtime capacity against the same cold BOM declaration.
         * @return Canonical geometry after matching this actual protocol slot roster.
         * @throws std::logic_error If a legacy custom test roster is not the native retained roster.
         */
        [[nodiscard]] MoEOverlayInferenceOriginalSlotGeometry geometry() const
        {
            const auto result=MoEOverlayInferenceOriginalSlotGeometry::forRetainedDraftCapacity(depth_);
            if(result.slotCount()!=count_)
                throw std::logic_error("Native resource geometry differs from the original protocol roster");
            return result;
        }
        /** @return Original frozen source/target/workspace topology. */
        [[nodiscard]] const MoEOverlayInferenceTopologyIdentity &topologyIdentity() const noexcept
        { return topology_; }
    private:
        /** @brief Bind a cold roster to its sole original protocol owner. */
        MoEOverlayInferenceOriginalSlotDomain(
            const MoEOverlayInferenceTransactionProtocol *owner,
            std::size_t count, int depth, MoEOverlayInferenceTopologyIdentity topology)
            : owner_(owner), count_(count), depth_(depth), topology_(std::move(topology)) {}
        /** @brief Check original retained ownership while ownership_mutex_ is held. */
        const MoEOverlayInferenceTransactionProtocol &requireRetainedOwner(
            const MoEOverlayInferenceOriginalSlot &slot) const;
        /** @brief Require exact active Submitted state while ownership_mutex_ is held. */
        void requireActiveOwnerLocked(const MoEOverlayInferenceOriginalSlot& slot) const;
        mutable std::mutex ownership_mutex_; ///< Orders original slot mutation and ownership withdrawal only.
        mutable const MoEOverlayInferenceTransactionProtocol *owner_;
        const std::size_t count_;
        const int depth_; ///< Original configuration ceiling, not mutable selected depth.
        const MoEOverlayInferenceTopologyIdentity topology_;
    };

    /** Exact already-submitted original slot; raw counts/indices cannot mint it. */
    class MoEOverlayInferenceOriginalSlot final
    {
        friend class MoEOverlayInferenceTransactionProtocol;
    public:
        /** @return Shared lifetime and identity of the original protocol roster. */
        [[nodiscard]] const std::shared_ptr<const MoEOverlayInferenceOriginalSlotDomain> &domain() const noexcept
        { return domain_; }
        /** @return Index authenticated by the actual original protocol slot contents. */
        [[nodiscard]] std::size_t index() const noexcept { return index_; }
        /** @return Entire immutable accepted ticket; no model execution state. */
        [[nodiscard]] const MoEOverlayInferenceTransactionTicket &ticket() const noexcept
        { return ticket_; }
        /** @return Equality of the exact original roster, slot and publication. */
        bool operator==(const MoEOverlayInferenceOriginalSlot &) const = default;
    private:
        /** @brief Retain only a slot checked against the original submitted record. */
        MoEOverlayInferenceOriginalSlot(
            std::shared_ptr<const MoEOverlayInferenceOriginalSlotDomain> domain,
            std::size_t index, MoEOverlayInferenceTransactionTicket ticket)
            : domain_(std::move(domain)), index_(index), ticket_(std::move(ticket)) {}
        std::shared_ptr<const MoEOverlayInferenceOriginalSlotDomain> domain_;
        std::size_t index_;
        MoEOverlayInferenceTransactionTicket ticket_;
    };

    template<class Publish> decltype(auto)
    MoEOverlayInferenceOriginalSlotDomain::withActiveAdmission(
        const MoEOverlayInferenceOriginalSlot& slot,Publish&& publish) const
    {
        std::lock_guard lock(ownership_mutex_);
        requireActiveOwnerLocked(slot);
        return std::forward<Publish>(publish)();
    }
}
