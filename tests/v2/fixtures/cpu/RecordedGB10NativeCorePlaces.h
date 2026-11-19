/**
 * @file RecordedGB10NativeCorePlaces.h
 * @brief Recorded kernel inputs and expected capacity order for native-place replay.
 *
 * Distinct cluster capacities and the highest-capacity CPU19 discriminate this
 * observation from a synthetic two-class or CPU-id ordering. These are host
 * sysfs observations, not a claim about a container's live OpenMP worker state.
 */
#pragma once
#include <array>
#include <string_view>

namespace llaminar2::test
{
    /** @brief One immutable CPU observation relevant to the place-order policy. */
    struct RecordedCoreCapacity
    {
        int cpu;
        int physical_sibling;
        long capacity;
    };
    inline constexpr int kRecordedCorePlacePolicyVersion=1;
    inline constexpr std::string_view kRecordedCorePlaceSource="192.168.178.79 host CPU sysfs";
    inline constexpr std::string_view kRecordedCorePlaceCaptureDate="2026-11-11";
    inline constexpr std::array<RecordedCoreCapacity,20> kRecordedGB10CoreCapacities{{
        {0,0,718},{1,1,718},{2,2,718},{3,3,718},{4,4,718},
        {5,5,997},{6,6,997},{7,7,997},{8,8,997},{9,9,997},
        {10,10,731},{11,11,731},{12,12,731},{13,13,731},{14,14,731},
        {15,15,1017},{16,16,1017},{17,17,1017},{18,18,1017},{19,19,1024}
    }};
    inline constexpr std::array<int,20> kRecordedGB10CorePlaceOrder{
        19,15,16,17,18,5,6,7,8,9,10,11,12,13,14,0,1,2,3,4};
}
