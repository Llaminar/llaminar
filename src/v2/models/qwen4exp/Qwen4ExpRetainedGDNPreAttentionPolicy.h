/**
 * @file Qwen4ExpRetainedGDNPreAttentionPolicy.h
 * @brief Participant-scoped selection shared by participant graph and workspace admission.
 */
#pragma once
#include "utils/DebugEnv.h"
#include "models/qwen4exp/Qwen4ExpRetainedM1HCFFNPolicy.h"
#include "execution/config/RoutedExpertPolicy.h"
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace llaminar2 {
/**
 * @brief Select the participant and invocation for the measured GDN projection.
 * @param selected Resolved cold participant policy.
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

/**
 * @brief Resolve retained ordinary GDN at the graph and PMA boundaries.
 * @param qualified_domain Whether this participant belongs to the measured MI50 TP4 placement.
 * @return The explicit diagnostic selection or the placement's ordinary default.
 */
inline bool qfnRetainedGDNPreAttentionEnabled(bool qualified_domain)
{
    static const std::optional<bool> selected = []() -> std::optional<bool> {
        const char *value = DebugEnv::envValue("LLAMINAR_QFN_RETAINED_GDN_PREATTENTION");
        if (!value) return std::nullopt;
        if (std::string_view(value) == "0") return false;
        if (std::string_view(value) == "1") return true;
        throw std::invalid_argument("LLAMINAR_QFN_RETAINED_GDN_PREATTENTION requires 0 or 1");
    }();
    return selected.value_or(qualified_domain);
}
} // namespace llaminar2
