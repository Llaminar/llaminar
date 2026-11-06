/**
 * @file OriginalSlotCaptureTerminalState.h
 * @brief Exact capture-terminal metadata indexed by the original admitted protocol slots.
 *
 * This owns host submission/observation metadata only. It does not allocate slots,
 * select device epochs, or mirror native completion. Only the original protocol
 * can mint a publication binding; an event observation must prove each receipt
 * before its original publication slot may be released.
 */
#pragma once
#include "execution/moe/MoEOverlayInferenceOriginalSlot.h"
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>
namespace llaminar2 {
class OriginalSlotCaptureTerminalState;
/** @brief Unforgeable host receipt naming one exact original-slot event publication. */
class OriginalSlotCaptureTerminalReceipt final {
    friend class OriginalSlotCaptureTerminalState;
public:
    /** @return Original protocol publication retained until native observation. */
    const MoEOverlayInferenceOriginalSlot& originalSlot()const noexcept{return slot_;}
    /** @return Cache-local monotonic publication generation, never device progress. */
    std::uint64_t generation()const noexcept{return generation_;}
private:
    struct Identity final {};
    /** @brief Mint only after the actual event record accepted this original publication. */
    OriginalSlotCaptureTerminalReceipt(std::shared_ptr<const Identity> identity,
        MoEOverlayInferenceOriginalSlot slot,std::uint64_t generation)
        :identity_(std::move(identity)),slot_(std::move(slot)),generation_(generation){}
    std::shared_ptr<const Identity> identity_;
    MoEOverlayInferenceOriginalSlot slot_;
    std::uint64_t generation_;
};
/** @brief Capture-owner bookkeeping over the SAME immutable original slot roster. */
class OriginalSlotCaptureTerminalState final {
public:
    /** @brief Bind to an existing original domain; this cannot mint a roster or capacity. */
    explicit OriginalSlotCaptureTerminalState(std::shared_ptr<const MoEOverlayInferenceOriginalSlotDomain> domain)
        :domain_(std::move(domain)),identity_(std::make_shared<OriginalSlotCaptureTerminalReceipt::Identity>()) {
        if(!domain_ || !domain_->slotCount())throw std::invalid_argument("Capture terminal requires original admitted slots");
        slots_.resize(domain_->slotCount());
    }
    OriginalSlotCaptureTerminalState(const OriginalSlotCaptureTerminalState&)=delete;
    OriginalSlotCaptureTerminalState&operator=(const OriginalSlotCaptureTerminalState&)=delete;
    /** @return Same original count; never a separately chosen terminal-ring capacity. */
    std::size_t slotCount()const noexcept{return domain_->slotCount();}
    /** @return Same immutable original roster identity and shared control block. */
    const std::shared_ptr<const MoEOverlayInferenceOriginalSlotDomain>&domain()const noexcept{return domain_;}
    /**
     * @brief Authenticate before recording a native event; no metadata is changed.
     * @throws std::logic_error For retired/foreign/duplicate/out-of-order or busy slot publications.
     */
    void requirePublication(const MoEOverlayInferenceOriginalSlot& slot)const {
        std::lock_guard lock(mutex_);requirePublicationLocked(slot);
    }
private:
    /** @brief Caller holds receipt metadata only; protocol admission is always original-owned. */
    void requirePublicationLocked(const MoEOverlayInferenceOriginalSlot& slot)const {
        requireDomain(slot);
        domain_->requireCurrent(slot);
        if(slot.index()>=slots_.size())throw std::logic_error("Capture terminal original slot is out of range");
        const auto& prior=slots_[slot.index()];
        if(prior && (prior->generation_>observed_ || prior->slot_==slot))
            throw std::logic_error("Capture terminal would overwrite an unobserved or duplicate original publication");
        if(last_ && !(order(last_->ticket())<order(slot.ticket())))
            throw std::logic_error("Capture terminal original publication order regressed");
        if(published_==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Capture terminal publication namespace exhausted");
    }
public:
    /**
     * @brief Record metadata only after its exact event record succeeds.
     * @return Exact receipt, with no allocation on this publication path.
     */
    OriginalSlotCaptureTerminalReceipt publish(const MoEOverlayInferenceOriginalSlot& slot) {
        std::lock_guard lock(mutex_);requirePublicationLocked(slot);
        OriginalSlotCaptureTerminalReceipt receipt(identity_,slot,++published_);
        slots_[slot.index()]=receipt;last_=slot;return receipt;
    }
    /** @brief Authenticate the exact oldest outstanding receipt before native observation. */
    void requireObservation(const OriginalSlotCaptureTerminalReceipt& receipt)const {
        std::lock_guard lock(mutex_);requireObservationLocked(receipt);
    }
    /** @brief Validate a current original device edge even after host observation; never consume proof. */
    void requireDeviceDependency(const OriginalSlotCaptureTerminalReceipt& receipt)const {
        std::lock_guard lock(mutex_);requireReceiptLocked(receipt);
    }
private:
    /** @brief Current slot identity is checked independently of host observation order. */
    void requireReceiptLocked(const OriginalSlotCaptureTerminalReceipt& receipt)const {
        if(receipt.identity_.get()!=identity_.get() || receipt.identity_.owner_before(identity_) ||
            identity_.owner_before(receipt.identity_))throw std::logic_error("Capture terminal belongs to another cache lifetime");
        requireDomain(receipt.slot_);domain_->requireCurrent(receipt.slot_);
        if(!receipt.generation_ || receipt.generation_>published_ ||
            !slots_.at(receipt.slot_.index()) ||
            slots_[receipt.slot_.index()]->generation_!=receipt.generation_ ||
            !(slots_[receipt.slot_.index()]->slot_==receipt.slot_))
            throw std::logic_error("Capture terminal receipt is stale or belongs to another original publication");
    }
    /** @brief Exact native observation must advance the oldest unconsumed receipt, once. */
    void requireObservationLocked(const OriginalSlotCaptureTerminalReceipt& receipt)const {
        requireReceiptLocked(receipt);
        if(receipt.generation_!=observed_+1)
            throw std::logic_error("Capture terminal observation is duplicated or out of order");
    }
public:
    /** @brief Mark an authenticated receipt observed only after the exact native event proves it. */
    void observed(const OriginalSlotCaptureTerminalReceipt& receipt) {
        std::lock_guard lock(mutex_);requireObservationLocked(receipt);observed_=receipt.generation_;
    }
    /** @return Oldest outstanding receipt for original cache-owner teardown, without dropping proof. */
    std::optional<OriginalSlotCaptureTerminalReceipt> nextOutstanding()const {
        std::lock_guard lock(mutex_);
        if(observed_==published_)return std::nullopt;
        for(const auto& slot:slots_)if(slot && slot->generation_==observed_+1)return slot;
        throw std::logic_error("Capture terminal lost an original outstanding receipt");
    }
private:
    /** @brief Compare immutable scheduler publication order, not device execution state. */
    static std::tuple<std::uint64_t,std::uint64_t,std::uint64_t> order(
        const MoEOverlayInferenceTransactionTicket& ticket)noexcept {
        return std::tuple{ticket.request_generation,ticket.command_id,ticket.transaction_ordinal};
    }
    /** @brief Reject pointer aliases with foreign shared ownership before consulting protocol state. */
    void requireDomain(const MoEOverlayInferenceOriginalSlot& slot)const {
        if(slot.domain().get()!=domain_.get() || slot.domain().owner_before(domain_) ||
            domain_.owner_before(slot.domain()))throw std::logic_error("Capture terminal has foreign original slot authority");
    }
    std::shared_ptr<const MoEOverlayInferenceOriginalSlotDomain> domain_;
    std::shared_ptr<const OriginalSlotCaptureTerminalReceipt::Identity> identity_;
    std::vector<std::optional<OriginalSlotCaptureTerminalReceipt>> slots_;
    std::optional<MoEOverlayInferenceOriginalSlot> last_;
    // Worker publication and aggregate host observation may overlap, but no
    // native event operation or wait runs while this metadata lock is held.
    mutable std::mutex mutex_;
    std::uint64_t published_=0,observed_=0;
};
}
