/**
 * @file DeviceMoEIntermediateShardBinding.h
 * @brief Capture-stable original GPU fragment geometry, independent of live placement.
 *
 * Only an authenticated preparation family can create this binding. It retains
 * no engines, allocation, selected expert set or epoch; the acquired placement
 * bank alone decides current residency. Whole-owner replica masks are not a
 * representation of a shard cohort and never participate in this contract.
 */
#pragma once

#include "MoEOverlayJointRouteTarget.h"
#include "DeviceMoEExpertFlags.h"
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_FRAGMENT_HD __host__ __device__
#else
#define LLAMINAR_MOE_FRAGMENT_HD
#endif

namespace llaminar2
{
    class MoEOverlayFragmentFamily;

    /** @brief Immutable local F interval derived from one original source family. */
    class DeviceMoEIntermediateShardBinding final
    {
    public:
        /** @brief Ordinary complete-expert lane with no fragment authority. */
        LLAMINAR_MOE_FRAGMENT_HD constexpr DeviceMoEIntermediateShardBinding() noexcept = default;

        /**
         * @brief Bind the exact original GPU cohort without retaining an epoch reader.
         * @param family Closed registry-authenticated source/preparation/member family.
         * @return Its original route target and unsliced/local projection geometry.
         * @throws std::invalid_argument For a foreign, non-GPU or incomplete source family.
         */
        static DeviceMoEIntermediateShardBinding fromOriginalFamily(
            const MoEOverlayFragmentFamily &family);

        /** @return Whether this graph lane owns an original local fragment. */
        LLAMINAR_MOE_FRAGMENT_HD constexpr bool enabled() const noexcept
        { return target_.enabled(); }
        /** @return Original global member and cohort target; no live ownership is cached. */
        LLAMINAR_MOE_FRAGMENT_HD constexpr const MoEOverlayJointRouteTarget &target() const noexcept
        { return target_; }
        /** @return Complete unchanged model input/output width H. */
        LLAMINAR_MOE_FRAGMENT_HD constexpr std::uint32_t modelColumns() const noexcept
        { return model_columns_; }
        /** @return Original unsliced intermediate width F. */
        LLAMINAR_MOE_FRAGMENT_HD constexpr std::uint32_t intermediateColumns() const noexcept
        { return intermediate_columns_; }
        /** @return Source-native origin of the local F interval. */
        LLAMINAR_MOE_FRAGMENT_HD constexpr std::uint32_t firstIntermediateColumn() const noexcept
        { return first_intermediate_column_; }
        /** @return Complete local gate/up N and down K extent. */
        LLAMINAR_MOE_FRAGMENT_HD constexpr std::uint32_t localIntermediateColumns() const noexcept
        { return local_intermediate_columns_; }
        /**
         * @return Complete immutable geometry equality, including original member identity.
         * @details HIP must inline these field checks even in large decoder
         * families. An out-of-line generic reference forces the enclosing
         * by-value launch record into per-thread private storage.
         */
#if defined(__HIP_DEVICE_COMPILE__)
        __attribute__((always_inline))
#endif
        LLAMINAR_MOE_FRAGMENT_HD constexpr bool sameIdentity(const DeviceMoEIntermediateShardBinding &other) const noexcept
        {
            return target_.sameIdentity(other.target_) && model_columns_ == other.model_columns_ &&
                intermediate_columns_ == other.intermediate_columns_ &&
                first_intermediate_column_ == other.first_intermediate_column_ &&
                local_intermediate_columns_ == other.local_intermediate_columns_;
        }

    private:
        MoEOverlayJointRouteTarget target_;
        std::uint32_t model_columns_ = 0;
        std::uint32_t intermediate_columns_ = 0;
        std::uint32_t first_intermediate_column_ = 0;
        std::uint32_t local_intermediate_columns_ = 0;
    };
    static_assert(sizeof(DeviceMoEIntermediateShardBinding) == 32);
    static_assert(std::is_trivially_copyable_v<DeviceMoEIntermediateShardBinding>);
} // namespace llaminar2
#undef LLAMINAR_MOE_FRAGMENT_HD
