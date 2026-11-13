/**
 * @file Qwen4ExpPredictorRows.h
 * @brief Explicit single-request predictor row admission with one native count authority.
 *
 * The same token-row descriptor feeds normalization, branch concat, native
 * attention and FC's checked fourfold physical expansion. An absent count means
 * an explicitly admitted fully-active matrix; a present count is never dropped
 * or downloaded. Request-hole layouts are outside this initial composition.
 */
#pragma once
#include "Qwen4ExpPrimitiveStage.h"
#include "../../../../kernels/common/DeviceRowRange.h"
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>
namespace llaminar2
{
/** @brief Sealed original token rows; branch expansion borrows the same scalar. */
class Qwen4ExpPredictorRows final
{
public:
    /**
     * @brief Admit canonical MTP query-row metadata without observing live count contents.
     * @param requests Only one independent request is supported by this composer.
     * @param capacity Physical query rows in this captured invocation, never cache length.
     * @param device Exact execution/count owner.
     * @param host_counts CPU-owned single request count, or nullptr for fully-active rows.
     * @param device_count GPU-owned INT32 count, or nullptr for fully-active rows.
     * @return Original immutable descriptor with exactly one count owner or explicit full activity.
     * @throws std::invalid_argument For unsupported requests, geometry, owner or count alignment.
     * @throws std::overflow_error When four branch rows per token exceed native INT32 geometry.
     */
    static Qwen4ExpPredictorRows admit(int requests, int capacity, DeviceId device,
        const std::vector<int> *host_counts, const std::int32_t *device_count)
    {
        static_assert(std::is_same_v<int, std::int32_t>);
        if (requests != 1 || capacity <= 0 || !device.is_valid() || (!device.is_cpu() && !device.is_gpu()))
            throw std::invalid_argument("Dense0 predictor requires one request and positive admitted query rows");
        if (capacity > std::numeric_limits<int>::max() / qwen4exp::kResidualBranches)
            throw std::overflow_error("Dense0 branch rows exceed the native INT32 geometry");
        const std::int32_t *count = nullptr;
        if (device.is_gpu()) {
            if (host_counts || (device_count && reinterpret_cast<std::uintptr_t>(device_count) % alignof(std::int32_t)))
                throw std::invalid_argument("GPU predictor row count must be one aligned native device authority");
            count = device_count;
        } else {
            if (device_count || (host_counts && host_counts->size() != 1))
                throw std::invalid_argument("CPU predictor row count must name exactly its one host request");
            count = host_counts ? host_counts->data() : nullptr;
        }
        return {device, count ? DeviceRowRange::deviceCounted(capacity, count)
                              : DeviceRowRange::fullyActive(capacity)};
    }
    /**
     * @brief Authenticate exact compact CPU query extent without replacing its host count authority.
     * @throws std::invalid_argument If the authoritative CPU count differs from admitted physical rows.
     *
     * CPU KV/attention consume exact host-owned seq_len. This initial R1 composer
     * therefore admits compact prefill/decode only. GPU counts are never read by
     * this setup check and retain their native variable-prefix semantics.
     */
    void requireCompactCPUQuery() const
    {
        if (device_.is_cpu() && rows_.countOwner() && *rows_.countOwner()!=rows_.physicalRows())
            throw std::invalid_argument("Dense0 CPU requires exact compact host-owned query rows; padded/empty prefix descriptors are unsupported");
    }
    /** @return Original query-row descriptor for concat, FC policy and native reader wiring. */
    const DeviceRowRange &tokens() const noexcept { return rows_; }
    /** @return Four physical branch rows per live token, borrowing exactly the same count. */
    DeviceRowRange branches() const { return rows_.repeatEachRow(qwen4exp::kResidualBranches); }
    /** @return Primitive geometry derived from the same original count; no second scalar exists. */
    Qwen4ExpStageRows primitives() const
    {
        Qwen4ExpStageRows result{rows_.physicalRows(), std::nullopt};
        if (rows_.countOwner())
            result.liveRows = Qwen4ExpNativeIntInput{device_, rows_.countOwner(), 1};
        return result;
    }
    /** @return Exact immutable execution participant. */
    DeviceId device() const noexcept { return device_; }
private:
    /** @brief Only canonical admission constructs original token geometry. */
    Qwen4ExpPredictorRows(DeviceId device, DeviceRowRange rows) : device_(device), rows_(rows) {}
    DeviceId device_;
    DeviceRowRange rows_;
};
} // namespace llaminar2
