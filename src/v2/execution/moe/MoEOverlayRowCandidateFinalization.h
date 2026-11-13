/**
 * @file MoEOverlayRowCandidateFinalization.h
 * @brief Typed table-finalization ordering for complete overlay row candidates.
 *
 * Explicit fixed-head ranges are mandatory physical storage at row selection.
 * Automatic table placement instead runs after rows and expert grants freeze.
 * This setup-only decision owns metadata results, never a live memory ledger.
 */
#pragma once
#include "loaders/ImmutableRowTablePolicy.h"
#include "planning/PhysicalMemoryCapacityExhausted.h"
#include <functional>
#include <variant>
#include <string>
#include <type_traits>
#include <utility>

namespace llaminar2
{
/** @brief Lower-priority table placement waits until rows and grants freeze. */
struct MoEOverlayRowDeferred {};

/** @brief A valid complete candidate exceeds its canonical physical capacity. */
struct MoEOverlayRowCapacityRejected
{
    std::string diagnostic;
};

/** @brief Final admission is present exactly when this alternative is held. */
template<class Result>
struct MoEOverlayRowAdmitted
{
    Result value;
};

/** @brief Exclusive setup outcomes; admitted metadata cannot be absent. */
template<class Result>
using MoEOverlayRowFinalization = std::variant<MoEOverlayRowDeferred,
    MoEOverlayRowCapacityRejected, MoEOverlayRowAdmitted<Result>>;

/**
 * @brief Finalize mandatory fixed-head storage before selecting a row candidate.
 * @param policy Unmodified declared table policy; Automatic retains lower priority.
 * @param finalize Complete canonical admission operation preserving this candidate's grants.
 * @return Deferred without invoking finalize for other policies, rejected only for
 * typed capacity exhaustion, or the one owned final admission for reuse.
 * @throws Any structural, arithmetic, identity or lifecycle error from finalize.
 * @note Failed candidates create no backing allocations or separate live ledger.
 */
template<class Finalize>
[[nodiscard]] auto finalizeMoEOverlayRowCandidate(
    const ImmutableRowTablePolicy &policy, Finalize &&finalize)
    -> MoEOverlayRowFinalization<std::invoke_result_t<Finalize>>
{
    using Result = std::invoke_result_t<Finalize>;
    if (!policy.isFixedHeadSharded()) return MoEOverlayRowDeferred{};
    try
    {
        return MoEOverlayRowAdmitted<Result>{std::invoke(std::forward<Finalize>(finalize))};
    }
    catch (const PhysicalMemoryCapacityExhausted &error)
    {
        return MoEOverlayRowCapacityRejected{error.what()};
    }
}
} // namespace llaminar2
