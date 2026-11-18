/**
 * @file Qwen4ExpRetainedGDNPreAttentionPolicy.h
 * @brief Cold experimental selection shared by participant graph and workspace admission.
 */
#pragma once
#include "models/qwen4exp/Qwen4ExpRetainedM1HCFFNPolicy.h"
#include "execution/config/RoutedExpertPolicy.h"
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace llaminar2 {
/**
 * @brief Select the participant and invocation for the measured GDN projection.
 * @param selected Explicit cold experimental policy.
 * @param device Actual local participant.
 * @param placement Routed expert placement for this participant.
 * @param phase Original execution phase.
 * @param role Original forward role, including MTP verifier and condition.
 * @param rows Admitted sequence width.
 * @param requests Number of concurrent requests in the graph.
 * @return True when this graph may consume the retained workspace and lowering.
 */
inline bool usesRetainedGDNPreAttention(bool selected, DeviceId device, RoutedExpertComputePolicy placement,
    ForwardExecutionPhase phase, ForwardExecutionRole role, int rows, int requests)
{
    return device.is_rocm() && placement == RoutedExpertComputePolicy::TensorSharded &&
        useQwen4ExpRetainedM1HCFFN(selected, phase, role, rows, requests);
}

/** @return Immutable request to evaluate the retained GDN projection on supported participants. */
inline bool qfnRetainedGDNPreAttentionOptIn()
{
    static const bool selected = [] {
        const char *value = std::getenv("LLAMINAR_QFN_RETAINED_GDN_PREATTENTION");
        if (!value || std::string_view(value) == "0") return false;
        if (std::string_view(value) == "1") return true;
        throw std::invalid_argument("LLAMINAR_QFN_RETAINED_GDN_PREATTENTION requires 0 or 1");
    }();
    return selected;
}
} // namespace llaminar2
