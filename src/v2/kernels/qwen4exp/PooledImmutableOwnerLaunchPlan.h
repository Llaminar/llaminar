/**
 * @file PooledImmutableOwnerLaunchPlan.h
 * @brief Pure owner dispatch geometry shared by actual CUDA/HIP launchers and closed graph declarations.
 *
 * Counts describe native kernel invocations, not compute stages or live rows.
 * Reply capacity is immutable admitted geometry; no device count is read here.
 */
#pragma once
#include "PooledImmutableNativeTypes.h"
#include <stdexcept>
namespace llaminar2::qwen4exp::pooled_native {
/** @brief Compile-time launch roles for one exact native owner specialization. */
template<OwnerAction Action> struct OwnerLaunchPlan final {
    static constexpr bool publishesReplies = Action==OwnerAction::Pin || Action==OwnerAction::ExecuteFree ||
        Action==OwnerAction::ExecuteVictim || Action==OwnerAction::CancelFree || Action==OwnerAction::CancelVictim;
    static constexpr bool readsRows = Action==OwnerAction::ReadRows;
    static constexpr bool appliesSource = Action==OwnerAction::ApplySourceDelivery;
    /** @return Exact native invocations for immutable zero/positive reply geometry. */
    static constexpr unsigned kernelCount(std::uint32_t replyCapacity) noexcept {
        return 1u + (publishesReplies && replyCapacity ? 1u : 0u) +
            ((readsRows || appliesSource) ? 1u + (replyCapacity ? 1u : 0u) : 0u);
    }
};
/** @return Exact dispatch count; reject enum holes rather than treating them as a phase. */
constexpr unsigned ownerKernelCount(OwnerAction action, std::uint32_t replyCapacity) {
    switch(action) {
#define QFN_OWNER_COUNT(Name) case OwnerAction::Name: return OwnerLaunchPlan<OwnerAction::Name>::kernelCount(replyCapacity);
        QFN_OWNER_COUNT(Begin) QFN_OWNER_COUNT(Pin) QFN_OWNER_COUNT(SealPins) QFN_OWNER_COUNT(AcquirePins)
        QFN_OWNER_COUNT(Availability) QFN_OWNER_COUNT(AcceptFree) QFN_OWNER_COUNT(ExecuteFree)
        QFN_OWNER_COUNT(FreeReceipt) QFN_OWNER_COUNT(AcquireFree) QFN_OWNER_COUNT(AcceptVictim)
        QFN_OWNER_COUNT(ExecuteVictim) QFN_OWNER_COUNT(VictimReceipt) QFN_OWNER_COUNT(SealSource)
        QFN_OWNER_COUNT(ApplySourceDelivery) QFN_OWNER_COUNT(ValidateFills) QFN_OWNER_COUNT(AcquirePublication)
        QFN_OWNER_COUNT(MakeReady) QFN_OWNER_COUNT(ReadyReceipt) QFN_OWNER_COUNT(Release)
        QFN_OWNER_COUNT(CancelFree) QFN_OWNER_COUNT(CancelVictim) QFN_OWNER_COUNT(CancelSource)
        QFN_OWNER_COUNT(Retire) QFN_OWNER_COUNT(DrainSourceFailure) QFN_OWNER_COUNT(PublishWire)
        QFN_OWNER_COUNT(ReadyWire) QFN_OWNER_COUNT(PublicationAcquiredWire) QFN_OWNER_COUNT(RetiredWire)
        QFN_OWNER_COUNT(AcquireReaderRetirement) QFN_OWNER_COUNT(ReadRows)
#undef QFN_OWNER_COUNT
    }
    throw std::invalid_argument("Native owner launch plan rejects an undeclared action");
}
}
