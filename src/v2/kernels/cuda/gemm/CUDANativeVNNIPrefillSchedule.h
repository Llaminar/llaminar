/**
 * @file CUDANativeVNNIPrefillSchedule.h
 * @brief Host/device-independent identity and admission of BK64 operand staging.
 *
 * A staging schedule changes operand delivery, not the stored weight format or
 * FP32 arithmetic. Keep it separate from output tiles and K-partition policy so
 * launchers, resource queries, and evidence cannot confuse distinct symbols.
 * Structural admission does not certify compiler resources or performance.
 */
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define LLAMINAR_PREFILL_SCHEDULE_HD __host__ __device__
#else
#define LLAMINAR_PREFILL_SCHEDULE_HD
#endif

namespace llaminar2::cuda::prefill
{
/** @brief Physical operand delivery schedule; values are evidence identities. */
enum class PrefillStagingSchedule : std::uint8_t
{
    RegisterDecode = 0,       ///< Load packed bytes into registers, then decode.
    AsyncPayload = 1,         ///< Copy packed bytes into existing decoded slots.
    AsyncWeightOperands = 2,  ///< Also stage native scale/offset metadata.
    AsyncAllOperands = 3,     ///< Also stage FP32 activation scales.
};

/** @brief Immutable arithmetic boundaries supplied by public serial-M1 policy.
 * Count identifies the exact ordered reduction tree; the precomputed span is
 * expressed in 32-value blocks. This plain launch argument is shared with
 * diagnostic capture, which must use the production ABI without a device-body
 * recompile or a second definition of its parameter layout.
 */
struct CanonicalM1PartitionGeometry
{
    int count;                 ///< Original serial partition count.
    int blocks_per_partition;  ///< Ceiling of the logical block count / count.
};

/** @brief Reject unknown schedule values at every host-facing boundary.
 * @param schedule Typed value, possibly received from serialized evidence.
 * @return Whether the value names one implemented schedule.
 */
LLAMINAR_PREFILL_SCHEDULE_HD constexpr bool
isKnownPrefillStagingSchedule(PrefillStagingSchedule schedule)
{
    switch (schedule)
    {
    case PrefillStagingSchedule::RegisterDecode:
    case PrefillStagingSchedule::AsyncPayload:
    case PrefillStagingSchedule::AsyncWeightOperands:
    case PrefillStagingSchedule::AsyncAllOperands:
        return true;
    }
    return false;
}

/** @brief Check the copy/decode ownership geometry without a CUDA runtime.
 * @param schedule Operand delivery identity, independent of K reduction.
 * @param payload_bytes Actual native bytes per 32-value weight block.
 * @param slots Shared-memory pipeline slots of the compiled kernel.
 * @param columns Columns in one output tile.
 * @param threads Threads in that tile's CTA.
 * @return Whether this structural specialization is implemented. Exact-symbol
 *         occupancy/spill checks are a separate mandatory setup-time gate.
 */
LLAMINAR_PREFILL_SCHEDULE_HD constexpr bool supportsPrefillStaging(
    PrefillStagingSchedule schedule,
    int payload_bytes,
    int slots,
    int columns,
    int threads)
{
    if (!isKnownPrefillStagingSchedule(schedule) || payload_bytes <= 0 ||
        slots <= 0 || columns <= 0 || threads <= 0)
        return false;
    if (schedule == PrefillStagingSchedule::RegisterDecode)
        return true;
    // Each half-block needs its own copy/decode owner. Division avoids overflow
    // for malformed dimensions; this predicate must not instantiate a kernel.
    return slots == 2 && columns <= threads / 2 &&
           (payload_bytes == 16 || payload_bytes == 20);
}
} // namespace llaminar2::cuda::prefill

#undef LLAMINAR_PREFILL_SCHEDULE_HD
