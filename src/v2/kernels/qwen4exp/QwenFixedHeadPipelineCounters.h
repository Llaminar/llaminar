/**
 * @file QwenFixedHeadPipelineCounters.h
 * @brief Existing generation-row observations for original fixed native PLE ranges.
 *
 * Kernels publish committed source work and checked endpoint completions. The
 * native library accumulates its actual useful sends into the same request row.
 * The request reset and terminal bridge are the only reset/readout owners.
 */
#pragma once
#include "kernels/common/DeviceGenerationControlLayout.h"
#include "kernels/common/NativeCounter64.h"
#include <array>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_QFN_PIPELINE_COUNTER_HD __host__ __device__ inline
#else
#define LLAMINAR_QFN_PIPELINE_COUNTER_HD inline
#endif
namespace llaminar2::qwen4exp {
/** @brief Closed order of actual fixed pipeline owner observations. */
enum class FixedHeadPipelineTerminalMetric : std::size_t {
    Calls, LiveIds, NativeBytes, Failures, RequestBytes, ResponseBytes, GatherBytes, CompletedReceipts, Count
};
inline constexpr auto kFixedHeadPipelineMetricCount = static_cast<std::size_t>(FixedHeadPipelineTerminalMetric::Count);
inline constexpr auto kFixedHeadPipelineCounterWords = 2 * kFixedHeadPipelineMetricCount;
// These alternatives are exclusive for one actual resolved table placement.
// Reuse the existing immutable lookup interval without changing controller ABI
// or the original request reset and pre-response publication rules.
inline constexpr auto kFixedHeadPipelineFirstWord = sampling_math::kDeviceGenerationControlFixedHeadCallsLow;
static_assert(kFixedHeadPipelineFirstWord % 2 == 0);
static_assert(kFixedHeadPipelineFirstWord + kFixedHeadPipelineCounterWords <=
    sampling_math::kDeviceGenerationControlAdaptiveFailedFramesHigh + 1);
using FixedHeadPipelineTerminalTotals = std::array<std::uint64_t, kFixedHeadPipelineMetricCount>;

/** @brief Device-only borrow of the fixed pipeline interval of the original request row. */
struct FixedHeadPipelineCounterRow {
    std::int32_t *control = nullptr;
    /** @return Exact aligned native pair for this closed metric, never a value. */
    LLAMINAR_QFN_PIPELINE_COUNTER_HD NativeCounter64 metric(FixedHeadPipelineTerminalMetric field) const {
        const auto offset = static_cast<std::size_t>(field);
        if (!control || offset >= kFixedHeadPipelineMetricCount) return {};
        return {control + kFixedHeadPipelineFirstWord + 2 * offset};
    }
    /** @brief Add performed work if this invocation retained terminal observations. */
    LLAMINAR_QFN_PIPELINE_COUNTER_HD bool add(FixedHeadPipelineTerminalMetric field, std::uint64_t value) const {
        return !control || addNativeCounter64(metric(field), value);
    }
};

/** @brief Decode the complete immutable terminal observation after its original event. */
inline FixedHeadPipelineTerminalTotals readFixedHeadPipelineCounters(
    const std::array<std::int32_t, kFixedHeadPipelineCounterWords> &words) {
    FixedHeadPipelineTerminalTotals result{};
    for (std::size_t i = 0; i < result.size(); ++i) result[i] = readNativeCounter64(words.data() + 2 * i);
    return result;
}
} // namespace llaminar2::qwen4exp
#undef LLAMINAR_QFN_PIPELINE_COUNTER_HD
