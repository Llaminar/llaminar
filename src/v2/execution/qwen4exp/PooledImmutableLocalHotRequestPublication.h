/**
 * @file PooledImmutableLocalHotRequestPublication.h
 * @brief Once-per-request host submission ownership for the original local-cache admission.
 *
 * The canonical DGO request epoch owns submission, not cache contents. Native
 * source invocation, hits, pins and policy remain device-owned. A repeated
 * forward joins the existing publication without sending another source header
 * or resetting the local frame; a failed partial submission is never retried.
 */
#pragma once
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace llaminar2::qwen4exp {
/** @brief Request-boundary submission state, with no device-state shadow. */
class PooledImmutableLocalHotRequestPublication final {
public:
    /** @brief Start one original request-submission lifetime. */
    PooledImmutableLocalHotRequestPublication() = default;
    PooledImmutableLocalHotRequestPublication(const PooledImmutableLocalHotRequestPublication&) = delete;
    PooledImmutableLocalHotRequestPublication& operator=(const PooledImmutableLocalHotRequestPublication&) = delete;
    /**
     * @brief Submit one new request or acquire the same request's exact producer.
     * @param epoch Original positive DGO session epoch, never a fabricated device invocation.
     * @param acquire_controller Join the original controller initialization/reset producer before reading it.
     * @param publish Enqueue original native admission/reset and publish its exact stream event.
     * @param acquire Join that existing producer on the actual caller's stream without a host wait.
     * @throws std::exception For stale epochs, recursive entry or any failed producer/acquire.
     * @note A thrown callback is sticky because it may have enqueued native work.
     * This owner is used only at ordinary request admission, outside capture.
     */
    template<class AcquireController, class Publish, class Acquire>
    void enter(std::uint64_t epoch, AcquireController&& acquire_controller, Publish&& publish, Acquire&& acquire)
    {
        if (!epoch || phase_ == Phase::Submitting || phase_ == Phase::Failed || epoch < epoch_)
            throw std::logic_error("Local-cache request publication requires a current, unfailed original epoch");
        const bool fresh = phase_ == Phase::Empty || phase_ == Phase::Retired || epoch != epoch_;
        phase_ = Phase::Submitting;
        try {
            std::forward<AcquireController>(acquire_controller)();
            if (fresh) std::forward<Publish>(publish)();
            std::forward<Acquire>(acquire)();
            epoch_ = epoch;
            phase_ = Phase::Published;
        } catch (...) {
            phase_ = Phase::Failed;
            throw;
        }
    }
    /**
     * @brief Retire enqueue ownership after the original whole-request terminal has completed.
     * @param epoch The same canonical DGO epoch that submitted the interval.
     * @note A continuing response may open a new source interval in the same
     * session. Only this actual terminal edge allows another source grant/reset;
     * local roots, warmed rows, policy and captured topology remain untouched.
     * Failed submission remains sticky even when cleanup has drained its work.
     * @throws std::logic_error For a foreign terminal or retirement during submission.
     */
    void retireAfterTerminal(std::uint64_t epoch)
    {
        if (phase_ == Phase::Empty || phase_ == Phase::Failed) return;
        if (phase_ == Phase::Submitting || epoch != epoch_)
            throw std::logic_error("Local-cache terminal changed its original request-publication owner");
        phase_ = Phase::Retired;
    }
private:
    /** @brief CPU enqueue ownership only; Published does not claim GPU completion. */
    enum class Phase { Empty, Submitting, Published, Retired, Failed };
    Phase phase_ = Phase::Empty;
    std::uint64_t epoch_ = 0;
};
} // namespace llaminar2::qwen4exp
