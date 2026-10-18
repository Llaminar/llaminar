/**
 * @file PrefixMovementJson.h
 * @brief Export completed prefix movement observations without collapsing PP epochs.
 *
 * HTTP and benchmark summaries share the same stage identities and validation.
 * This serializer consumes retained metadata only; it never queries a device,
 * fingerprints a payload, or advances a cache or maintenance lifecycle.
 */
#pragma once

#include "MoEMovementLedgerJson.h"
#include "execution/prefix_cache/PrefixCacheStats.h"

namespace llaminar2
{
    /**
     * @param stages Complete sealed PP request observations.
     * @return Versioned stage-local epochs; empty input is a programming error.
     * @throws std::logic_error for an incomplete or regressed observation.
     */
    inline nlohmann::json prefixMovementStagesJson(
        const MoEOptimizationStages<PrefixMovementEpochObservation> &stages)
    {
        if (stages.empty())
            throw std::logic_error("Pipeline prefix publication requires stage observations");
        auto result = nlohmann::json{{"schema", 1}, {"stages", nlohmann::json::array()}};
        for (const auto &stage : stages.entries())
        {
            (void)stage.value.crossed();
            result["stages"].push_back({{"identity", movement_json_detail::stageIdentityJson(stage.identity)},
                {"epochs", {{"admission_epoch_earliest", stage.value.admission.earliest()},
                    {"admission_epoch_latest", stage.value.admission.latest()},
                    {"completion_movement_epoch", stage.value.completion}}}});
        }
        return result;
    }
}
