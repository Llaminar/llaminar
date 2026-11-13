/**
 * @file AdaptiveImmutablePhysicalSlot.h
 * @brief Shared physical row-lifetime mutations, independent of replacement policy.
 *
 * The caller is the sole physical metadata writer and validates its frame,
 * generation, bounds and ordering before these leaf mutations. These helpers
 * allocate nothing, choose no victim and issue no communication. Existing
 * single-GPU policy and the pooled-owner prerequisite use exactly these writes.
 */
#pragma once
#include "AdaptiveImmutableRowCacheState.h"

namespace llaminar2::qwen4exp::adaptive_lookup
{
/** @brief Stateless leaf operations on the original canonical slot records. */
template <class Slot> class PhysicalSlot final
{
public:
    /** @return Native readability bits without interpreting policy membership. */
    QFN_ADAPTIVE_HD static SlotState state(const Slot &slot)
    {
        return static_cast<SlotState>(slot.flags & StateMask);
    }

    /** @return Whether the one active owner frame retains native backing. */
    QFN_ADAPTIVE_HD static bool pinned(const Slot &slot)
    {
        return (slot.flags & FramePin) != 0;
    }

    /** @brief Initialize one virgin slot; setup alone may call this operation. */
    QFN_ADAPTIVE_HD static void initialize(Slot &slot)
    {
        slot = Slot{};
        slot.previous = slot.next = None;
        slot.flags = static_cast<std::uint32_t>(Stream::None) << 2;
    }

    /** @brief Retain one validated unique initial hit, without changing its policy region. */
    QFN_ADAPTIVE_HD static void pin(Slot &slot)
    {
        slot.flags |= FramePin;
    }

    /**
     * @brief Bind a newly issued full generation and one frame pin to unreadable storage.
     * @param slot Exact owner-local slot, already removed from any old policy/directory.
     * @param row Checked complete original source key; row32 callers prove representability.
     * @param generation Nonzero monotonically owner-issued generation, never a packed token.
     * @param touched Original policy clock; physical-only owners pass zero.
     * @note NewProbation's bits are zero; the single-GPU caller appends its policy next.
     */
    QFN_ADAPTIVE_HD static void reserve(Slot &slot, std::uint64_t row,
                                       std::uint64_t generation, std::uint64_t touched)
    {
        slot = Slot{};
        slot.row = static_cast<decltype(slot.row)>(row);
        slot.generation = generation;
        slot.touched = touched;
        slot.previous = slot.next = None;
        slot.flags = static_cast<std::uint32_t>(SlotState::Reserved) | FramePin;
    }

    /** @brief Publish already-filled bytes, preserving the actual pin and policy region. */
    QFN_ADAPTIVE_HD static void ready(Slot &slot)
    {
        slot.flags = (slot.flags & ~StateMask) | static_cast<std::uint32_t>(SlotState::Ready);
    }

    /** @brief Roll back only readability when the existing directory insertion rejects. */
    QFN_ADAPTIVE_HD static void reserved(Slot &slot)
    {
        slot.flags = (slot.flags & ~StateMask) | static_cast<std::uint32_t>(SlotState::Reserved);
    }

    /** @brief Drop a validated Ready-row reader after its last native use. */
    QFN_ADAPTIVE_HD static void unpin(Slot &slot)
    {
        slot.flags &= ~FramePin;
    }

    /**
     * @brief Recycle unreadable Reserved backing into the canonical owner's free chain.
     * @param slot Already unlinked from any policy queue; generation/row remain diagnostic.
     * @param freeHead Current owner-local free head. The caller then publishes this slot as head.
     */
    QFN_ADAPTIVE_HD static void recycle(Slot &slot, std::uint32_t freeHead)
    {
        slot.flags = static_cast<std::uint32_t>(Stream::None) << 2;
        slot.previous = freeHead;
        slot.next = None;
    }
};
} // namespace llaminar2::qwen4exp::adaptive_lookup
