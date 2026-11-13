/**
 * @file AdaptiveImmutableRowCacheControlBorrow.h
 * @brief Lexical exclusive control borrow for one ordered native metadata writer.
 *
 * Frame/owner records remain the sole persistent authorities. One kernel copies
 * them into caller-provided CTA-shared temporaries, runs unchanged Operations on
 * those records, then publishes before another kernel can observe them. Slots,
 * directory, frequency cells and protocol remain their actual original arrays.
 * This helper allocates nothing and creates no persistent residency/state owner.
 */
#pragma once
#include "AdaptiveImmutableRowCacheState.h"
namespace llaminar2::qwen4exp::adaptive_lookup
{
/**
 * @brief Noncopyable temporary borrow; caller owns the exclusive writer interval.
 * @tparam Slot Actual checked row32/row64 slot type.
 *
 * Both temporaries must outlive this lexical scope and must be disjoint from
 * the published records. Native callers supply exactly128 CTA-shared bytes.
 * Scope exit publishes on every normal/early-return path. Before calling any
 * operation that mutates the original frame status, publish(), perform that
 * canonical failure operation, then refreshStatus() so scope exit cannot erase
 * the sticky error. Publication also preserves any already non-Success status.
 */
template<class Slot> class OrderedControlBorrow final
{
public:
    /** @brief Copy the actual records into exclusive caller-provided temporaries.
     * @param original Actual metadata roots, already validated by the native plan.
     * @param frame Temporary FrameControl (CTA-shared in native code).
     * @param owner Temporary OwnerControl; never read when capacity is zero.
     */
    QFN_ADAPTIVE_HD OrderedControlBorrow(Metadata<Slot> original, FrameControl &frame, OwnerControl &owner)
        : original_(original), working_(original)
    {
        frame = *original.frame;
        working_.frame = &frame;
        if (original.capacity)
        {
            owner = *original.owner;
            working_.owner = &owner;
        }
    }
    /** @brief Publish before the exclusive lexical owner retires. */
    QFN_ADAPTIVE_HD ~OrderedControlBorrow() { publish(); }
    OrderedControlBorrow(const OrderedControlBorrow &) = delete;
    OrderedControlBorrow &operator=(const OrderedControlBorrow &) = delete;
    OrderedControlBorrow(OrderedControlBorrow &&) = delete;
    OrderedControlBorrow &operator=(OrderedControlBorrow &&) = delete;
    /** @return Same actual arrays, with only the small control records borrowed. */
    QFN_ADAPTIVE_HD Metadata<Slot> metadata() const { return working_; }
    /** @return The one lexical frame used by both the body and Operations. */
    QFN_ADAPTIVE_HD FrameControl &frame() const { return *working_.frame; }
    /**
     * @brief Publish ordered controls, retaining a previously published failure.
     * @note There is no other valid writer during this borrow. The status merge
     * additionally prevents an existing failure publication from being erased.
     */
    QFN_ADAPTIVE_HD void publish()
    {
        const auto prior = original_.frame->status;
        if (original_.capacity) *original_.owner = *working_.owner;
        *original_.frame = *working_.frame;
        if (prior != Status::Success) original_.frame->status = prior;
        working_.frame->status = original_.frame->status;
    }
    /** @brief Adopt a canonical frame-status publication after publish()+failure(). */
    QFN_ADAPTIVE_HD void refreshStatus() { working_.frame->status = original_.frame->status; }
private:
    Metadata<Slot> original_, working_;
};
} // namespace llaminar2::qwen4exp::adaptive_lookup
