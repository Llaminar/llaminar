/**
 * @file MTPMainForwardPolicy.h
 * @brief Immutable main-model topology shared by MTP setup and pipeline replay.
 *
 * A condition declares commit ownership; a grouped verifier declares its
 * outcome topology and exact retained physical width. These alternatives never confer sampler authority on a
 * pipeline follower. Mutable token/position/length rows stay device-owned.
 */
#pragma once
#include "MTPConditionForwardPurpose.h"
#include "MTPVerifierOutcomeGraph.h"
#include "MTPServingForwardCaptureGeometry.h"
#include <variant>

namespace llaminar2
{
/** @brief A checked member of the admitted verifier graph family, never live row data. */
class MTPVerifierForwardPolicy final
{
public:
    /**
     * @brief Authenticate one outcome and physical bucket against retained capacity.
     * @param outcome Immutable terminal reduction topology.
     * @param physical_rows Exact graph width selected by the request width policy.
     * @param retained Widest admitted setup geometry.
     * @throws std::invalid_argument for an unknown outcome or non-member width.
     */
    MTPVerifierForwardPolicy(MTPVerifierOutcomeGraphMode outcome, int physical_rows,
                             const MTPServingForwardCaptureGeometry &retained)
        : outcome_(outcome), physical_rows_(physical_rows)
    {
        if (!retained.enabled || !retained.valid() || physical_rows < 2 || physical_rows > retained.verifier_rows ||
            (outcome != MTPVerifierOutcomeGraphMode::Greedy && outcome != MTPVerifierOutcomeGraphMode::Disabled) ||
            mtpVerifierPhysicalRowBucket(physical_rows, retained.verifier_rows) != physical_rows)
            throw std::invalid_argument("MTP main-forward policy is not a retained verifier graph member");
    }
    /** @return Immutable terminal reduction topology; followers retain Disabled. */
    [[nodiscard]] MTPVerifierOutcomeGraphMode outcome() const noexcept { return outcome_; }
    /** @return Graph width only; device state remains the authority for active rows. */
    [[nodiscard]] int physicalRows() const noexcept { return physical_rows_; }
private:
    MTPVerifierOutcomeGraphMode outcome_;
    int physical_rows_;
};

/** @brief Exactly one condition or grouped-verifier main-model invocation. */
using MTPMainForwardPolicy = std::variant<MTPConditionForwardPurpose, MTPVerifierForwardPolicy>;
}
