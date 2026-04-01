/**
 * @file StageVerifier.h
 * @brief Stage input/output verification (Debug/Integration builds only)
 * @author David Sanftenberg
 * @date December 2025
 *
 * Extracted from DeviceGraphExecutor to allow verification logic to be
 * tested independently and reused by other executors.
 *
 * All functions are only available when LLAMINAR_ASSERTIONS_ACTIVE is true
 * (Debug and Integration builds). They compile to nothing in Release.
 */

#pragma once

#include "../../../utils/DebugEnv.h" // For LLAMINAR_ASSERTIONS_ACTIVE

#if LLAMINAR_ASSERTIONS_ACTIVE

#include "ComputeGraph.h"

namespace llaminar2
{

    /**
     * @brief Verify stage inputs before execution (ENTRY validation)
     *
     * Checks all INPUT buffers for null, NaN, or Inf data.
     * Also validates buffer layouts if the stage declares LayoutExpectation.
     *
     * Can be disabled at runtime with LLAMINAR_VALIDATE_INPUTS=0.
     * Throws VerificationFailure on validation failure with full context.
     *
     * @param node The node whose inputs should be validated
     * @param layer_idx Current layer index for error reporting
     * @throws verification::VerificationFailure if validation fails
     */
    void verifyStageEntry(const ComputeNode &node, int layer_idx);

    /**
     * @brief Verify stage outputs after execution (EXIT validation)
     *
     * Checks all OUTPUT buffers for null, NaN, Inf, or all-zero data.
     * Also validates buffer layouts if the stage declares LayoutExpectation.
     *
     * Can be disabled at runtime with LLAMINAR_VALIDATE_BUFFERS=0.
     * Throws VerificationFailure on validation failure with full context.
     *
     * @param node The node whose outputs should be validated
     * @param layer_idx Current layer index for error reporting
     * @throws verification::VerificationFailure if validation fails
     */
    void verifyStageExit(const ComputeNode &node, int layer_idx);

    /**
     * @brief Validate stage outputs for zero/NaN tensors (legacy)
     *
     * Checks all OUTPUT buffers for uninitialized (zero) or corrupted (NaN/Inf) data.
     * Uses GPU-side validation when available, falls back to host-side.
     *
     * Can be disabled at runtime with LLAMINAR_VALIDATE_BUFFERS=0.
     *
     * @param node The node whose outputs should be validated
     * @return true if validation passes, false if errors detected
     *
     * @deprecated Use verifyStageExit() instead for exception-based validation
     */
    bool validateStageOutputs(const ComputeNode &node);

} // namespace llaminar2

#endif // LLAMINAR_ASSERTIONS_ACTIVE
