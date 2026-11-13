/**
 * @file ROCmGDNHistoricalRecurrentOracle.h
 * @brief Test-only explicit inputs for a literal historical HIP recurrence.
 *
 * Every pointer is fixture-owned and captured on its exact non-default stream.
 * This oracle owns no allocation/state/publication policy and includes no
 * production normalization or recurrence implementation.
 */
#pragma once
namespace gdn_historical_native
{
/** @brief Immutable native launch tuple, including independent request and snapshot geometry. */
struct Arguments final
{
    const float *q, *k, *v; ///< Original split FP32 values, never pre-normalized.
    const float *alpha, *beta, *a_log, *dt_bias; ///< Raw projected gates and immutable gains.
    float *output; ///< Exclusive physical output rows; inactive rows become zero.
    const float *initial; ///< Immutable request-local initial state; may equal updated for serial replay.
    float *updated; ///< Exclusive terminal state, or unused when complete snapshots own publication.
    int requests, rows, heads, key_width, value_width; ///< Frozen admitted launch coordinates.
    bool normalize; ///< Historical Q/K arithmetic switch, not a test replacement path.
    const int *lengths; ///< Borrowed device-owned live count per physical request.
    int row_offset; ///< Original causal row for a captured one-row oracle node.
    float *snapshots; ///< Optional complete row-state bank with physical indexing.
    int snapshot_stride, snapshot_rows; ///< Exact snapshot envelope; no allocation is performed.
    void *stream; ///< Exact non-default stream retained by the fixture.
};
/**
 * @brief Launch the frozen historical device kernel without synchronization or allocation.
 * @param args Fixture-owned explicit native bindings and complete geometry.
 * @return False on invalid static admission or actual HIP launch failure.
 */
bool launch(const Arguments &args);
}
