/**
 * @file Qwen4ExpObservedPipelineTerminalEvidence.h
 * @brief Original discovery retirement and numerical outcome remain independent.
 *
 * A failed numerical request still owns its authenticated runtime configuration.
 * Export is legal only after the original discovery join proves every selected
 * communicator borrower retired. It never changes a failed work verdict, rank
 * namespace, physical participant identity, or inactive rank's obligations.
 */
#pragma once
#include "config/OrchestrationPlanningPolicy.h"
#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace llaminar2::test::parity::qwen4exp {
/** @brief Closed observations transmitted by the original terminal allgather. */
enum class ObservedPipelineRetirement : std::int32_t { Live, Retired };
/** @brief Numerical/setup/teardown outcome, independent of communicator retirement. */
enum class ObservedPipelineWorkOutcome : std::int32_t { Failed, Succeeded };
/** @brief One original discovery rank's actual post-release observations. */
struct ObservedPipelineRankTerminalOutcome {
    ObservedPipelineRetirement retirement;
    ObservedPipelineWorkOutcome work;
};
static_assert(std::is_trivially_copyable_v<ObservedPipelineRankTerminalOutcome> &&
    sizeof(ObservedPipelineRankTerminalOutcome)==2*sizeof(std::int32_t),
    "Original terminal metadata must communicate exactly its two live outcomes");
/** @brief Immutable evidence available only after the original terminal join. */
class Qwen4ExpObservedPipelineTerminalEvidence final {
public:
    /**
     * @brief Authenticate the completed original discovery allgather.
     * @param outcomes Rank-indexed original observations, including inactive ranks.
     * @throws std::invalid_argument For missing or noncanonical terminal observations.
     */
    explicit Qwen4ExpObservedPipelineTerminalEvidence(std::span<const ObservedPipelineRankTerminalOutcome> outcomes)
        :discovery_size_(outcomes.size()) {
        if(outcomes.empty())throw std::invalid_argument("Observed pipeline terminal join has no original ranks");
        for(const auto& outcome:outcomes) {
            if((outcome.retirement!=ObservedPipelineRetirement::Live && outcome.retirement!=ObservedPipelineRetirement::Retired) ||
               (outcome.work!=ObservedPipelineWorkOutcome::Failed && outcome.work!=ObservedPipelineWorkOutcome::Succeeded))
                throw std::invalid_argument("Observed pipeline terminal join has invalid original observations");
            all_retired_ &= outcome.retirement==ObservedPipelineRetirement::Retired;
            all_succeeded_ &= outcome.work==ObservedPipelineWorkOutcome::Succeeded;
        }
    }
    /** @return Whether all actual original communicator borrowers retired. */
    bool allRetired() const noexcept {return all_retired_;}
    /** @return Existing complete-work verdict; failed numerics never become successful. */
    bool succeeded() const noexcept {return all_retired_ && all_succeeded_;}
    /**
     * @brief Permit only the original selected active owner to export its bound configuration.
     * @param selection Actual already-authenticated original execution selection, absent when unbound.
     * @param discoveryRank Original discovery identity, never compact execution rank zero.
     * @return Whether the bound original owner is retired; work success is independent.
     */
    bool mayExportConfiguration(const std::optional<ExecutionRankSelection>& selection,int discoveryRank) const noexcept {
        if(!all_retired_ || !selection || selection->size()!=1 || discoveryRank<0 ||
           static_cast<std::size_t>(discoveryRank)>=discovery_size_)return false;
        return selection->discoveryRanks().front()==discoveryRank;
    }
private:
    std::size_t discovery_size_;
    bool all_retired_=true;
    bool all_succeeded_=true;
};
}
