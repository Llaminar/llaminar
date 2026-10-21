/**
 * @file QwenAdaptiveTerminalCounters.h
 * @brief Exact owner-authored adaptive work in the existing canonical terminal row.
 *
 * The single active adaptive frame owns these reporting words. They decide no
 * residency or inference state and allocate nothing. Direct probe/Ready/erase
 * operations author their own observations, including work before cancellation;
 * no generation/occupancy differences or CPU request absence infer GPU work.
 */
#pragma once
#include "kernels/common/DeviceGenerationControlLayout.h"
#include "kernels/common/NativeCounter64.h"
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_ADAPTIVE_COUNTER_HD __host__ __device__ inline
#else
#define LLAMINAR_ADAPTIVE_COUNTER_HD inline
#endif
namespace llaminar2::qwen4exp
{
/** @brief Immutable reporting tuple, with original duplicate IDs separate from unique probe rows. */
struct QwenAdaptiveCounterTotals
{
    std::uint64_t calls = 0, completed_live_ids = 0, hit_rows = 0, miss_rows = 0;
    std::uint64_t source_rows = 0, source_bytes = 0, promotions = 0, evictions = 0, failed_frames = 0;
    /** @return Complete observation identity, with no execution-state ownership. */
    bool operator==(const QwenAdaptiveCounterTotals &) const = default;
};
/** @brief Fixed host/device value projection, with no standard-library device dispatch. */
struct QwenAdaptiveCounterValues
{
    std::uint64_t values[9];
    /** @return One value in the canonical metric order; caller supplies an index below9. */
    LLAMINAR_ADAPTIVE_COUNTER_HD std::uint64_t operator[](unsigned i) const
    {
        return values[i];
    }
};
/** @return Fixed ordered tuple; no payload or allocation is introduced. */
LLAMINAR_ADAPTIVE_COUNTER_HD QwenAdaptiveCounterValues
adaptiveCounterValues(const QwenAdaptiveCounterTotals &v)
{
    return {{v.calls, v.completed_live_ids, v.hit_rows, v.miss_rows, v.source_rows, v.source_bytes,
             v.promotions, v.evictions, v.failed_frames}};
}
/** @brief Decode only the closed appended words after native completion or under the sole native writer.
 * @param row Original canonical controller. @param words Complete canonical extent.
 * @param out Updated only after structural validation.
 * @return False for absent, misaligned or truncated controller storage.
 */
LLAMINAR_ADAPTIVE_COUNTER_HD bool readQwenAdaptiveCounters(const std::int32_t *row, int words,
                                                           QwenAdaptiveCounterTotals &out)
{
    if (!row || words != sampling_math::kDeviceGenerationControlCount ||
        reinterpret_cast<std::uintptr_t>(row) % alignof(std::int32_t))
        return false;
    const auto *begin = row + sampling_math::kDeviceGenerationControlAdaptiveCallsLow;
    out = {readNativeCounter64(begin),      readNativeCounter64(begin + 2),  readNativeCounter64(begin + 4),
           readNativeCounter64(begin + 6),  readNativeCounter64(begin + 8),  readNativeCounter64(begin + 10),
           readNativeCounter64(begin + 12), readNativeCounter64(begin + 14), readNativeCounter64(begin + 16)};
    return true;
}
/**
 * @brief Add the actual frame tuple under its single-writer controller InOut contract.
 * @param row Original canonical request row. @param words Exact complete extent.
 * @param delta Actual completed operation counts; no inferred cache occupancy.
 * @return False before any reporting-word mutation if a field overflows or storage is malformed.
 * @note One source domain has one active frame, whose existing finish kernel is
 * the sole writer of these pairs. Unrelated controller fields remain untouched.
 */
LLAMINAR_ADAPTIVE_COUNTER_HD bool accumulateQwenAdaptiveCounters(std::int32_t *row, int words,
                                                                 const QwenAdaptiveCounterTotals &delta)
{
    QwenAdaptiveCounterTotals old;
    if (!readQwenAdaptiveCounters(row, words, old))
        return false;
    const auto before = adaptiveCounterValues(old), add = adaptiveCounterValues(delta);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
    for (unsigned i = 0; i < 9; ++i)
        if (add[i] > UINT64_MAX - before[i])
            return false;
    auto *begin = row + sampling_math::kDeviceGenerationControlAdaptiveCallsLow;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
    for (unsigned i = 0; i < 9; ++i)
    {
        const auto value = before[i] + add[i];
        begin[2 * i] = static_cast<std::int32_t>(static_cast<std::uint32_t>(value));
        begin[2 * i + 1] = static_cast<std::int32_t>(static_cast<std::uint32_t>(value >> 32));
    }
    return true;
}
/**
 * @brief Validate completed terminal observations against the exact original native source/frame envelope.
 * @param v Actual terminal tuple. @param rowBytes Original native stride, never decoded width.
 * @param maximumIds Admitted maximum original IDs per frame, not communication extent.
 * @return Whether all overflow-safe successful/failed-work relations hold.
 * @details Failed frames may have probes, evictions, validated source bytes or
 * Ready transitions even though no complete output is claimed. Successful-only
 * windows require every miss served and every eviction followed by a promotion.
 */
inline bool validQwenAdaptiveCounters(const QwenAdaptiveCounterTotals &v, std::size_t rowBytes,
                                      std::uint64_t maximumIds)
{
    if (!rowBytes || !maximumIds || maximumIds % 16 || v.completed_live_ids % 16 ||
        v.failed_frames > v.calls || v.hit_rows > UINT64_MAX - v.miss_rows ||
        v.source_rows > UINT64_MAX / rowBytes || v.source_bytes != v.source_rows * rowBytes ||
        v.source_rows > v.miss_rows || v.promotions > v.source_rows || v.evictions > v.miss_rows)
        return false;
    const auto probes = v.hit_rows + v.miss_rows;
    if (v.calls <= UINT64_MAX / maximumIds)
    {
        const auto limit = v.calls * maximumIds;
        if (probes > limit || v.completed_live_ids > limit)
            return false;
    }
    if (!v.calls)
        return v == QwenAdaptiveCounterTotals{};
    return v.failed_frames ||
           (probes <= v.completed_live_ids && v.source_rows == v.miss_rows && v.evictions <= v.promotions);
}
static_assert(sampling_math::kDeviceGenerationControlAdaptiveCallsLow == 72);
static_assert(sampling_math::kDeviceGenerationControlAdaptiveFailedFramesHigh + 1 ==
              sampling_math::kDeviceGenerationControlCount);
} // namespace llaminar2::qwen4exp
#undef LLAMINAR_ADAPTIVE_COUNTER_HD
