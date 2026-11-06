/**
 * @file HostedDeviceGenerationFragment.h
 * @brief Immutable source identity of an original retained hosted generation branch.
 *
 * Mechanically shared from DGO's original fragment value. A semantic verifier
 * names its exact retained forward signature rather than a native capture or
 * semantic graph pointer. Identity membership grants no execution authority;
 * original engine ownership, generation and complete-parent checks stay at the
 * caller. No request cursor, native resource or live execution state is owned.
 */
#pragma once
#include "execution/local_execution/engine/ForwardGraphTypes.h"
#include "backends/IGPUGraphCapture.h"
namespace llaminar2 {
struct HostedDeviceGenerationFragment
{
    enum class Kind : uint8_t
    {
        CapturedLocal = 0,
        /** Retained MainDecode plus sampler under one overlay scope. */
        OrdinaryExpertOverlayReplay,
        MTPFullSidecarReplay,
        MTPChainedSidecarReplay,
        MTPGroupedVerifierReplay,
        /** Entry boundary owned by the hosted scheduler stream. */
        MoEOverlayEpochAcquire,
        /** Terminal boundary that closes the hosted parent's reader. */
        MoEOverlayEpochRelease,
    };

    const char *name = nullptr;
    Kind kind = Kind::CapturedLocal;
    const IGPUGraphCapture *capture = nullptr;
    const ComputeGraph *semantic_graph = nullptr;
    ForwardGraphSignature forward_signature{};
    DeviceControlledLoopFragmentExecution execution =
        DeviceControlledLoopFragmentExecution::Always;
    const uint32_t *condition_word_device = nullptr;
    int minimum_selector = -1;

    /** @return Whether two plans embed the same immutable executable identity. */
    [[nodiscard]] bool hasSameExecutionIdentity(
        const HostedDeviceGenerationFragment &other) const noexcept
    {
        return kind == other.kind && capture == other.capture &&
               semantic_graph == other.semantic_graph &&
               forward_signature == other.forward_signature &&
               execution == other.execution &&
               condition_word_device == other.condition_word_device &&
               minimum_selector == other.minimum_selector;
    }
    /**
     * @brief Match only the original semantic hosted verifier arm's complete key.
     * @param retained Exact key already resolved through the original retained
     * forward owner and workspace generation by the production caller.
     * @return Cold identity membership, never capture, readiness or launch authority.
     */
    [[nodiscard]] bool embedsVerifierForward(const ForwardGraphSignature &retained) const noexcept
    {
        return kind == Kind::MTPGroupedVerifierReplay && !capture && !semantic_graph &&
            retained.execution_role == ForwardExecutionRole::GroupedMTPVerifier && retained.decode &&
            retained.device.is_gpu() && retained.seq_len > 0 && retained.batch_size > 0 &&
            retained.usesDeviceTokenIds() && retained.usesDevicePositionIds() && retained.usesDeviceSequenceLengths() &&
            forward_signature == retained;
    }
};
} // namespace llaminar2
