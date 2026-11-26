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
 * @param qualified Whether the participant placement and original projection formats qualify.
 * @return The explicit diagnostic selection or the qualified ordinary default.
 */
inline bool qfnRetainedGDNPreAttentionEnabled(bool qualified)
{
    static const std::optional<bool> selected = []() -> std::optional<bool> {
        const char *value = DebugEnv::envValue("LLAMINAR_QFN_RETAINED_GDN_PREATTENTION");
        if (!value) return std::nullopt;
        if (std::string_view(value) == "0") return false;
        if (std::string_view(value) == "1") return true;
        throw std::invalid_argument("LLAMINAR_QFN_RETAINED_GDN_PREATTENTION requires 0 or 1");
    }();
    return selected.value_or(qualified);
}

/**
 * @brief Select the CUDA full-GDN attention megakernel across all CUDA topologies and MoE placements.
 * @param selected Resolved cold participant policy.
 * @param device Actual local CUDA participant.
 * @param phase Original execution phase.
 * @param role Original forward role.
 * @param rows Admitted sequence width.
 * @param requests Number of concurrent requests in the graph.
 * @return True when this graph may consume the CUDA retained GDN workspace and lowering.
 */
inline bool usesRetainedCUDAGDN(bool selected, DeviceId device,
    ForwardExecutionPhase phase, ForwardExecutionRole role, int rows, int requests)
{
    return device.is_cuda() &&
        useQwen4ExpRetainedM1HCFFN(selected, phase, role, rows, requests);
}

/**
 * @brief Resolve retained CUDA full-GDN at the graph and PMA boundaries.
 * @param qualified Whether the original GDN layer projection formats qualify.
 * @return Explicit diagnostic override or the qualified ordinary default.
 */
inline bool qfnRetainedCUDAGDNEnabled(bool qualified)
{
    static const std::optional<bool> selected = []() -> std::optional<bool> {
        const char *value = DebugEnv::envValue("LLAMINAR_QFN_RETAINED_CUDA_GDN");
        if (!value) return std::nullopt;
        if (std::string_view(value) == "0") return false;
        if (std::string_view(value) == "1") return true;
        throw std::invalid_argument("LLAMINAR_QFN_RETAINED_CUDA_GDN requires 0 or 1");
    }();
    return selected.value_or(qualified);
}

/** @brief Select the CUDA retained QSA Pre-/Post-Attention megakernels across all CUDA topologies. */
inline bool usesRetainedCUDAQSAAttention(bool selected, DeviceId device,
    ForwardExecutionPhase phase, ForwardExecutionRole role, int rows, int requests)
{
    return device.is_cuda() &&
        useQwen4ExpRetainedM1HCFFN(selected, phase, role, rows, requests);
}

/** @brief Resolve retained CUDA QSA megakernel selection at the graph and PMA boundaries. */
inline bool qfnRetainedCUDAQSAAttentionEnabled(bool qualified)
{
    static const std::optional<bool> selected = []() -> std::optional<bool> {
        const char *value = DebugEnv::envValue("LLAMINAR_QFN_RETAINED_CUDA_QSA");
        if (!value) return std::nullopt;
        if (std::string_view(value) == "0") return false;
        if (std::string_view(value) == "1") return true;
        throw std::invalid_argument("LLAMINAR_QFN_RETAINED_CUDA_QSA requires 0 or 1");
    }();
    return selected.value_or(qualified);
}

/** @brief Select the ROCm retained QSA Pre-/Post-Attention megakernels across all ROCm topologies. */
inline bool usesRetainedROCmQSAAttention(bool selected, DeviceId device,
    ForwardExecutionPhase phase, ForwardExecutionRole role, int rows, int requests)
{
    return device.is_rocm() &&
        useQwen4ExpRetainedM1HCFFN(selected, phase, role, rows, requests);
}

/** @brief Resolve retained ROCm QSA megakernel selection at the graph and PMA boundaries. */
inline bool qfnRetainedROCmQSAAttentionEnabled(bool qualified)
{
    static const std::optional<bool> selected = []() -> std::optional<bool> {
        const char *value = DebugEnv::envValue("LLAMINAR_QFN_RETAINED_ROCM_QSA");
        if (!value) return std::nullopt;
        if (std::string_view(value) == "0") return false;
        if (std::string_view(value) == "1") return true;
        throw std::invalid_argument("LLAMINAR_QFN_RETAINED_ROCM_QSA requires 0 or 1");
    }();
    return selected.value_or(qualified);
}

/**
 * @brief Select the CPU single-OMP two-megakernel-per-layer lowering across all CPU topologies.
 * @param selected Resolved cold participant policy.
 * @param device Actual local CPU participant.
 * @param phase Original execution phase.
 * @param role Original forward role.
 * @param rows Admitted sequence width.
 * @param requests Number of concurrent requests in the graph.
 * @return True when this CPU graph may contract local Attention and FFN regions into single-OMP megakernels.
 */
inline bool usesRetainedCPUMegakernels(bool selected, DeviceId device,
    ForwardExecutionPhase phase, ForwardExecutionRole role, int rows, int requests)
{
    return device.is_cpu() &&
        useQwen4ExpRetainedM1HCFFN(selected, phase, role, rows, requests);
}

/**
 * @brief Resolve CPU two-megakernel-per-layer selection from diagnostic override or qualified default.
 * @param qualified Whether the local CPU layer stages qualify for single-OMP contraction.
 * @return Explicit LLAMINAR_QFN_RETAINED_CPU_MEGAKERNELS override or qualified default.
 */
inline bool qfnRetainedCPUMegakernelsEnabled(bool qualified)
{
    static const std::optional<bool> selected = []() -> std::optional<bool> {
        const char *value = DebugEnv::envValue("LLAMINAR_QFN_RETAINED_CPU_MEGAKERNELS");
        if (!value) return std::nullopt;
        if (std::string_view(value) == "0") return false;
        if (std::string_view(value) == "1") return true;
        throw std::invalid_argument("LLAMINAR_QFN_RETAINED_CPU_MEGAKERNELS requires 0 or 1");
    }();
    return selected.value_or(qualified);
}
} // namespace llaminar2
