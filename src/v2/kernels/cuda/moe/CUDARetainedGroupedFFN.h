/**
 * @file CUDARetainedGroupedFFN.h
 * @brief Prepared operand contract for CUDA's ordered grouped Gate/Up and Down.
 *
 * A router-owned device plan compacts the original route and shared work,
 * validates the physical descriptors, and associates each live item with its
 * canonical local output row. Arithmetic consumes only those prepared rows.
 */
#pragma once

#include "CUDARetainedHCFFN.h"

namespace llaminar2::cuda::moe
{
    /** @brief Complete direct operands for one compact live routed or shared FFN row. */
    struct RetainedGroupedFFNRow
    {
        RetainedM1Expert weights;
        const int8_t* input_codes;
        const float* input_scales;
        int8_t* down_codes; ///< Q8 output of the fused Gate/Up epilogue.
        float* down_scales;
        float* contribution; ///< Original local output in canonical order.
    };
    static_assert(std::is_trivially_copyable_v<RetainedGroupedFFNRow>);

    /** @brief Original publication arithmetic requested by this physical decoder range. */
    enum class RetainedGroupedFFNRole : uint8_t { Routed, Shared };

    /** @brief Named decoder and original live count for one homogeneous arithmetic range. */
    struct RetainedGroupedFFNRange
    {
        const RetainedGroupedFFNRow* rows;
        const int32_t* live_rows;
        int32_t capacity;
        int32_t width;
        uint8_t gate_up_codebook;
        uint8_t down_codebook;
        RetainedGroupedFFNRole role;
    };

    /** @brief Cold-admitted grouped native arithmetic on one original CUDA participant. */
    class RetainedGroupedFFNProgram final
    {
    public:
        /**
         * @brief Admit an exact physical decoder range and check both loaded arithmetic entries.
         * @param device Original owning CUDA participant.
         * @param range Precise immutable operands and device-owned live count.
         * @param stream Exact prepared participant stream.
         * @param generation Original workspace generation.
         * @return A sealed range ready to record on its original device and stream.
         */
        static RetainedGroupedFFNProgram prepare(DeviceId device, RetainedGroupedFFNRange range,
            void* stream, uint64_t generation);

        /**
         * @brief Append the plan and both arithmetic phases on the final native timeline.
         * @param stream Same prepared original capture stream.
         * @param generation Current original workspace generation.
         * @pre The preceding router plan compacted and authenticated every live operand.
         */
        void record(void* stream, uint64_t generation) const;

    private:
        /** @brief Freeze native arguments, exact functions and the owner that admitted them. */
        RetainedGroupedFFNProgram(RetainedGroupedFFNRange range, const void* gate, const void* down,
            void* stream, uint64_t generation)
            : range_(range), gate_(gate), down_(down), stream_(stream), generation_(generation) {}

        RetainedGroupedFFNRange range_;
        const void* gate_;
        const void* down_;
        void* stream_;
        uint64_t generation_;
    };
}
