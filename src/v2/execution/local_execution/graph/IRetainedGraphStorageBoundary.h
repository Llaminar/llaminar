/**
 * @file IRetainedGraphStorageBoundary.h
 * @brief Explicit immutable I/O lifecycle surrounding a complete retained GPU parent.
 *
 * This service owns storage protocol metadata, never token/controller decisions.
 * A graph declares and retains its immutable channel before capture. Admission
 * arms an existing worker before GPU submission; captured terminal publication
 * releases it after the complete generation. A standalone prefill instead
 * retires at the next request/reset terminal after all its consumers join.
 * Await occurs only after the exact terminal event. The interface cannot submit
 * model graphs or choose branches.
 */
#pragma once
#include "../../../backends/DeviceId.h"
#include "RetainedStorageStatistics.h"
#include <cstdint>

namespace llaminar2
{
/** @brief Immutable addresses of canonical generation failure publication words. */
struct RetainedStorageGenerationControl
{
    int *healthy=nullptr; ///< Owning generation controller's health flag.
    int *error=nullptr; ///< Owning generation controller's terminal error code.
};
/** @brief Participant-local retained storage lifetime and exact ordering contract. */
class IRetainedGraphStorageBoundary
{
public:
    /** @brief Retire only after every owning graph and native terminal observer. */
    virtual ~IRetainedGraphStorageBoundary()=default;
    /** @brief Exact participant owning the immutable mapped channel aliases. */
    virtual DeviceId device() const noexcept=0;
    /** @brief Stable complete binding identity; immutable for this object's lifetime. */
    virtual std::uint64_t captureIdentity() const noexcept=0;
    /** @brief Bind canonical generation failure addresses once before stage construction. */
    virtual void bindGenerationControl(RetainedStorageGenerationControl control)=0;
    /** @brief Return already-bound canonical addresses without reading their device values. */
    virtual RetainedStorageGenerationControl generationControl() const=0;
    /** @brief Arm persistent I/O and publish immutable admission identity before any model launch.
     * @param generation Monotonic API session identity, never a device loop counter.
     * @param stream Exact request admission producer stream.
     */
    virtual void beginAdmission(std::uint64_t generation,void *stream)=0;
    /** @brief Whether this CPU-owned I/O interval still owes its terminal worker join.
     * @return The existing admission authority, without reading device execution state.
     * @note The owning DGO serializes request admission, terminal observation and reset.
     */
    virtual bool hasActiveInterval() const noexcept=0;
    /** @brief Fail if a graph is submitted outside its admitted storage lifetime. */
    virtual void requireArmedForLaunch() const=0;
    /** @brief Record native terminal protocol release after every lookup consumer.
     * @param stream Exact complete-parent stream, or reset stream after prior-producer joins.
     * @param healthy Optional canonical health word; null means unconditional terminal.
     * @param complete Optional canonical completion word paired with healthy.
     * @details With both predicates, the kernel publishes close only on unhealthy
     * or complete state. This lets HIP's retained ticket publisher close the
     * same I/O interval at the authenticated terminal, without a host decision.
     */
    virtual void recordTerminalClose(void *stream,const int *healthy=nullptr,const int *complete=nullptr) const=0;
    /** @brief Join I/O only after the complete native terminal event has completed. */
    virtual void awaitTerminal()=0;
    /** @brief Return this joined interval's immutable I/O evidence; reject a live/unobserved interval. */
    virtual RetainedStorageTerminalObservation terminalObservation() const=0;
};
} // namespace llaminar2
