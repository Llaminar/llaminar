/**
 * @file DeviceMoERebalanceCadenceRecipe.h
 * @brief Declare the native cadence frontier without lending a conditional graph.
 *
 * The final executable owns its IF and imports only ordinary capture definitions.
 * CUDA cannot import a graph that already contains a native conditional node.
 */
#pragma once

#include "backends/IGPUGraphCapture.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Participant-local publication, guarded maintenance and acknowledgement. */
    class DeviceMoERebalanceCadenceRecipe final
    {
    public:
        static constexpr std::size_t fragment_count = 3;
        using Fragments = std::array<DeviceControlledLoopFragment, fragment_count>;

        /**
         * @brief Bind the three reusable native sources and their exact persistent predicate.
         * @throws std::invalid_argument If the guarded branch lacks its device predicate.
         */
        DeviceMoERebalanceCadenceRecipe(const IGPUGraphCapture &publish,
            const IGPUGraphCapture &maintenance, const IGPUGraphCapture &acknowledge,
            const uint32_t *due)
            : fragments_{{
                {.name = "decode boundary publication", .capture = &publish},
                {.name = "conditional device MoE maintenance", .capture = &maintenance,
                 .execution = DeviceControlledLoopFragmentExecution::IfDeviceWordNonZero,
                 .condition_word_device = due},
                {.name = "decode boundary acknowledgement", .capture = &acknowledge},
            }}
        {
            if (!due)
                throw std::invalid_argument("Native MoE cadence requires its original device predicate");
        }

        /** @return Ordered borrows that must be composed directly in the final executable. */
        [[nodiscard]] const Fragments &fragments() const noexcept { return fragments_; }

        /** @return Whether both parents would embed exactly the same cadence inputs. */
        [[nodiscard]] bool operator==(const DeviceMoERebalanceCadenceRecipe &other) const noexcept
        {
            return std::equal(fragments_.begin(), fragments_.end(), other.fragments_.begin(),
                [](const auto &left, const auto &right) { return left.hasSameExecutionIdentity(right); });
        }

    private:
        Fragments fragments_;
    };
}
