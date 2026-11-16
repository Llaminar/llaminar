/**
 * @file MoEGroupingPackedDirectory.h
 * @brief Immutable bindings for a work directory published by ordinary grouping.
 *
 * Counts and offsets stay device-owned. The combined grouping producer can use
 * each expert's original offset interval for packed WideFloor8 descriptors,
 * retaining interior sentinel holes without another prefix scan. This value
 * borrows existing workspace storage; it neither allocates nor owns live bytes.
 * MoEPrefillGroupingProvenance owns publication and revokes stale borrowers.
 */
#pragma once

#include "MoEGroupedDirectory.h"
#include "../../backends/DeviceId.h"
#include "../../backends/ExplicitGPUStream.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
static_assert(sizeof(int) == sizeof(uint32_t));
static_assert(std::bit_cast<uint32_t>(int{-1}) == UINT32_MAX);
/** @brief Exact capture-time storage and geometry for the offset-slot layout. */
class MoEGroupingPackedDirectory final
{
public:
    /** @brief Bind the existing count/offset/active-ID/work banks on their producer stream.
     * @param counts Original device expert counts, never read by this factory.
     * @param offsets Original exclusive expert offsets.
     * @param active_ids Separate ascending raw active-ID publication.
     * @param words Existing packed-directory workspace, distinct from raw IDs.
     * @param capacity_words Actual borrowed work-bank capacity in uint32 words.
     * @param rows Physical token rows retained by grouping.
     * @param top_k Original route slots per token; duplicate experts remain legal.
     * @param experts Original count/offset extent and descriptor domain.
     * @param device Original GPU owner. @param workspace Exact workspace binding owner.
     * @param stream Exact nondefault producer/consumer stream.
     * @return Validated immutable bindings, without publishing successful execution.
     * @throws std::invalid_argument For missing, aliased or unrepresentable bindings. */
    static MoEGroupingPackedDirectory wideFloor8(const int *counts, const int *offsets,
        const int *active_ids, uint32_t *words, size_t capacity_words,
        int rows, int top_k, int experts, DeviceId device,
        const void *workspace, ExplicitGPUStream stream)
    {
        if (!counts || !offsets || !active_ids || !words || !workspace ||
            !device.is_gpu() || reinterpret_cast<uintptr_t>(stream.get()) <= 2u ||
            rows <= 0 || top_k <= 0 || experts <= 0 ||
            static_cast<uint32_t>(experts) > moe_grouped_directory::kMaximumExperts ||
            rows > std::numeric_limits<int>::max() / top_k)
            throw std::invalid_argument("grouped packed directory requires original GPU bindings and representable geometry");
        const int slots = rows * top_k;
        if (static_cast<uint32_t>(slots) >= moe_grouped_directory::kMaximumRows ||
            capacity_words < static_cast<size_t>(slots))
            throw std::invalid_argument("grouped packed directory banks must be distinct and cover every physical route slot");
        const std::array<const void *,4> banks{counts, offsets, active_ids, words};
        const std::array<size_t,4> lengths{size_t(experts), size_t(experts),
            size_t(std::min(slots,experts)), size_t(slots)};
        for (size_t i = 0; i < banks.size(); ++i)
        {
            const auto begin = reinterpret_cast<uintptr_t>(banks[i]);
            const auto bytes = lengths[i] * sizeof(uint32_t);
            if (begin % alignof(uint32_t) || begin > UINTPTR_MAX - bytes)
                throw std::invalid_argument("grouped packed directory bank extent is unaligned or overflows");
            for (size_t j = 0; j < i; ++j)
            {
                const auto prior = reinterpret_cast<uintptr_t>(banks[j]);
                if (begin < prior + lengths[j] * sizeof(uint32_t) && prior < begin + bytes)
                    throw std::invalid_argument("grouped packed directory live banks overlap");
            }
        }
        return {counts, offsets, active_ids, words, capacity_words, rows, top_k, experts, device, workspace, stream};
    }

    /** @return Original authoritative count bank. */
    [[nodiscard]] const int *counts() const noexcept { return counts_; }
    /** @return Original immutable exclusive-offset bank. */
    [[nodiscard]] const int *offsets() const noexcept { return offsets_; }
    /** @return Separate ascending raw active IDs, never reinterpreted as packed work. */
    [[nodiscard]] const int *activeIds() const noexcept { return active_ids_; }
    /** @return Borrowed packed work-bank address. */
    [[nodiscard]] uint32_t *words() const noexcept { return words_; }
    /** @return Original borrowed allocation capacity, not the submitted work extent. */
    [[nodiscard]] size_t capacityWords() const noexcept { return capacity_words_; }
    /** @return Physical work span; all holes are explicit invalid descriptors. */
    [[nodiscard]] int totalSlots() const noexcept { return rows_ * top_k_; }
    /** @return Original retained token geometry. */
    [[nodiscard]] int rows() const noexcept { return rows_; }
    /** @return Original route multiplicity, including repeated experts. */
    [[nodiscard]] int topK() const noexcept { return top_k_; }
    /** @return Original descriptor/count extent. */
    [[nodiscard]] int experts() const noexcept { return experts_; }
    /** @return Exact device owning all borrowed storage. */
    [[nodiscard]] DeviceId device() const noexcept { return device_; }
    /** @return Workspace identity whose replacement revokes publication. */
    [[nodiscard]] const void *workspace() const noexcept { return workspace_; }
    /** @return Exact stream ordering grouping before both projection roles. */
    [[nodiscard]] void *stream() const noexcept { return stream_.get(); }
    /** @return Whether all original bindings and geometry match one consuming transaction. */
    [[nodiscard]] bool operator==(const MoEGroupingPackedDirectory &other) const noexcept
    {
        return counts_ == other.counts_ && offsets_ == other.offsets_ && active_ids_ == other.active_ids_ &&
            words_ == other.words_ && capacity_words_ == other.capacity_words_ && rows_ == other.rows_ &&
            top_k_ == other.top_k_ && experts_ == other.experts_ && device_ == other.device_ &&
            workspace_ == other.workspace_ && stream_.get() == other.stream_.get();
    }

private:
    /** @brief Only the validating layout factory can construct a binding. */
    MoEGroupingPackedDirectory(const int *counts, const int *offsets, const int *active_ids,
        uint32_t *words, size_t capacity_words, int rows, int top_k, int experts,
        DeviceId device, const void *workspace, ExplicitGPUStream stream)
        : counts_(counts), offsets_(offsets), active_ids_(active_ids), words_(words),
          capacity_words_(capacity_words), rows_(rows), top_k_(top_k), experts_(experts),
          device_(device), workspace_(workspace), stream_(stream) {}
    const int *counts_, *offsets_, *active_ids_;
    uint32_t *words_;
    size_t capacity_words_;
    int rows_, top_k_, experts_;
    DeviceId device_;
    const void *workspace_;
    ExplicitGPUStream stream_;
};
}
