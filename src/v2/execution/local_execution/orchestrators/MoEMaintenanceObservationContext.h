/**
 * @file MoEMaintenanceObservationContext.h
 * @brief Keep maintenance replay attached to its original captured observation family.
 *
 * Source materialization and atomic first-use capture may author different
 * contexts. Replaying either owner preserves that original label and family;
 * this helper creates no alias, execution policy or second cache authority.
 */
#pragma once
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include <stdexcept>

namespace llaminar2::dgo_detail
{
/**
 * @brief Bind the atomic default only to an unbound maintenance cache.
 * @param cache Original source-materialized or still-unbound atomic cache.
 * @throws std::logic_error If a bound family lost its context or names another owner.
 * Replay cannot rename a captured family; request reset changes no observation identity.
 * The singleton geometry describes one controller transaction, not expert payload
 * rows. It must be present before capture so the executor authors the real
 * transaction-zero phase through the same lifecycle used for replay.
 */
inline void bindMoEMaintenanceReplayContext(DeviceGraphExecutor::GraphSegmentCache &cache)
{
    const auto &workload = cache.replay_workload;
    if (workload.valid() &&
        (workload.seq_len != 1 || workload.batch_size != 1 || workload.m != 1 ||
         !workload.decode || workload.all_position_rows != 0 || workload.verifier_outcome_mode != 0 ||
         workload.position_policy != 0 || workload.moe_placement_epoch != 0 ||
         workload.all_position_logits || workload.live_mtp_request_batch_condition))
        throw std::logic_error("MoE maintenance family carries a foreign transaction workload");
    if (!workload.valid() && (cache.initialized || cache.evidence_family != 0))
        throw std::logic_error("Materialized MoE maintenance family lost its original transaction workload");
    if (cache.perf_context.empty())
    {
        if (cache.initialized || cache.evidence_family != 0)
            throw std::logic_error("Materialized MoE maintenance family lost its original observation context");
        cache.perf_context = "moe_rebalance_maintenance_atomic";
    }
    else if (cache.perf_context != "moe_rebalance_maintenance_atomic" &&
             cache.perf_context != "pipeline_original_stage_maintenance_source")
        throw std::logic_error("MoE maintenance cache carries a foreign observation context: " + cache.perf_context);
    cache.replay_workload = {.seq_len = 1, .batch_size = 1, .m = 1, .decode = true};
}
}
