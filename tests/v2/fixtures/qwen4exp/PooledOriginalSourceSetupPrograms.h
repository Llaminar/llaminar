/** @file PooledOriginalSourceSetupPrograms.h
 * @brief Exact original source/setup event programs shared by native component fixtures.
 *
 * Each entry names an actual original creator: owner root setup, coordinator
 * root setup, source root setup and the boundary submission-retirement event.
 * This is a cold typed program roster, not a duplicated live memory ledger.
 */
#pragma once
#include "kernels/qwen4exp/PooledImmutableSourceGeometry.h"
#include "backends/NativeOrderingEventProgram.h"
#include <array>
namespace llaminar2::test {
/** @return The original one-owner source setup's four actual native event programs. */
inline auto originalPooledSourceSetupOrderingPrograms(
    const qwen4exp::PooledImmutablePhysicalOwnerGeometry& physical,
    const qwen4exp::PooledImmutableCoordinatorGeometry& coordinator,
    const ImmutableRowTableSource& source)
{
    return std::array{
        qwen4exp::PooledImmutableOwnerNativePlan::orderingProgram(physical,coordinator,0),
        qwen4exp::PooledImmutableCoordinatorNativePlan::orderingProgram(physical,coordinator),
        qwen4exp::PooledImmutableSourceGeometry::orderingProgram(physical,coordinator,source),
        NativeOrderingEventProgram::lookupSubmissionRetirement()};
}
}
