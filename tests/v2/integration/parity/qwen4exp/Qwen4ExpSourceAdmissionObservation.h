/**
 * @file Qwen4ExpSourceAdmissionObservation.h
 * @brief Keep admission outcome and final-owner retirement independently observable.
 *
 * A fatal GTest assertion returns from the observation callback; it must not
 * bypass explicit retirement or its evidence. Exceptions likewise unwind all
 * callback-local aliases before retirement and retain their original failure.
 * This test-only sequencing owns no model, allocation or alternate cleanup path.
 */
#pragma once
#include <exception>
#include <utility>

namespace llaminar2::test::parity::qwen4exp
{
/**
 * @brief Retire after every observation outcome, preserving failures and alias order.
 * @param observe Callback whose local ownership views expire before retirement.
 * @param retire Existing lifetime's explicit retirement and evidence publication.
 * @throws Any observation exception, after successful retirement; retirement
 *         exceptions escape without publishing a false completion.
 * @note GTest failures remain recorded by GTest. Neither callback is retried,
 *       and retirement success never changes the admission result to a pass.
 */
template <class Observe, class Retire>
void observeQwen4ExpAdmissionThenRetire(Observe &&observe, Retire &&retire)
{
    std::exception_ptr observation_error;
    try
    {
        std::forward<Observe>(observe)();
    }
    catch (...)
    {
        observation_error = std::current_exception();
    }
    std::forward<Retire>(retire)();
    if (observation_error)
        std::rethrow_exception(observation_error);
}
} // namespace llaminar2::test::parity::qwen4exp
