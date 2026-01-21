/**
 * @file StageCoherence.h
 * @brief Automatic device coherence management at stage boundaries
 * @author David Sanftenberg
 * @date January 2026
 *
 * Provides infrastructure for automatic tensor coherence at stage entry/exit.
 * This eliminates manual ensureOnDevice()/mark_device_dirty() calls in stages.
 *
 * The coherence system operates at stage boundaries:
 * - ENTRY: Ensures all input tensors are on the target device before execution
 * - EXIT: Marks all output tensors as device-dirty after execution
 *
 * Stages can declare their coherence policy to control automatic behavior:
 * - NONE: No automatic coherence (for MPI stages, custom management)
 * - INPUT: Only cohere inputs (outputs managed manually)
 * - OUTPUT: Only mark outputs dirty (assume inputs are ready)
 * - FULL: Both inputs and outputs (default for most stages)
 *
 * @note Automatic coherence is always enabled. The GraphExecutor handles
 *       coherence automatically at stage boundaries based on each stage's
 *       coherencePolicy().
 */

#pragma once

#include "../backends/DeviceId.h"
#include <vector>
#include <string>

namespace llaminar2
{

    // Forward declarations
    class ITensor;
    class TensorBase;
    struct StageDumpInfo;

    /**
     * @brief Coherence policy for automatic stage boundary management
     *
     * Controls how the GraphExecutor handles tensor coherence at stage entry/exit.
     */
    enum class CoherencePolicy
    {
        NONE,   ///< No automatic coherence (for MPI stages, custom management)
        INPUT,  ///< Only cohere inputs (mark outputs manually if needed)
        OUTPUT, ///< Only mark outputs dirty (assume inputs are ready)
        FULL,   ///< Both inputs and outputs (default for most stages)
    };

    /**
     * @brief Buffer descriptor for coherence management
     *
     * Holds tensor pointer and metadata for coherence operations.
     * Built from StageDumpInfo input/output buffers.
     */
    struct CoherenceBuffer
    {
        ITensor *tensor = nullptr;   ///< Tensor pointer (may be nullptr if not available)
        const char *name = nullptr;  ///< Buffer name for logging
        const void *data = nullptr;  ///< Raw data pointer (for identifying tensor)
        size_t rows = 0;             ///< Number of rows
        size_t cols = 0;             ///< Number of columns
        const char *dtype = nullptr; ///< Data type string
        bool is_inout = false;       ///< If true, treat as both input and output
    };

    /**
     * @brief Ensure all input tensors are on the target device
     *
     * Calls ensureOnDevice() on each input tensor that supports device coherence.
     * Tensors must be TensorBase (or derived) to support coherence operations.
     *
     * @param inputs Vector of input buffers with tensor pointers
     * @param target_device Target device for execution
     * @return true if all inputs are now on device, false on error
     *
     * @note Skips nullptr tensors gracefully
     * @note Logs at DEBUG level when coherence operations are performed
     */
    bool cohereInputs(const std::vector<CoherenceBuffer> &inputs, DeviceId target_device);

    /**
     * @brief Allocate GPU buffers for output tensors before execution
     *
     * For GPU execution, output tensors need their GPU buffers allocated
     * BEFORE the kernel runs so it has somewhere to write. This function
     * calls ensureOnDevice() on outputs to allocate the buffers without
     * uploading data (the kernel will write to them).
     *
     * @param outputs Vector of output buffers with tensor pointers
     * @param target_device Target device for execution
     * @return true if all outputs have GPU buffers allocated, false on error
     *
     * @note Only needed for GPU targets - CPU execution doesn't need this
     * @note Skips nullptr tensors gracefully
     */
    bool cohereOutputs(const std::vector<CoherenceBuffer> &outputs, DeviceId target_device);

    /**
     * @brief Mark all output tensors as device-dirty
     *
     * Calls mark_device_dirty() on each output tensor that supports coherence.
     * This indicates that the GPU copy is now authoritative and host is stale.
     *
     * @param outputs Vector of output buffers with tensor pointers
     *
     * @note Skips nullptr tensors gracefully
     * @note Logs at DEBUG level when coherence operations are performed
     */
    void markOutputsDirty(const std::vector<CoherenceBuffer> &outputs);

    /**
     * @brief Extract CoherenceBuffer list from StageDumpInfo inputs
     *
     * Converts StageDumpInfo::InputBuffer entries to CoherenceBuffer format.
     * Uses the tensor pointer if available in the InputBuffer.
     *
     * @param info StageDumpInfo from stage->getDumpInfo()
     * @return Vector of CoherenceBuffer for input tensors
     */
    std::vector<CoherenceBuffer> extractInputBuffers(const StageDumpInfo &info);

    /**
     * @brief Extract CoherenceBuffer list from StageDumpInfo weights
     *
     * Converts StageDumpInfo::WeightBuffer entries to CoherenceBuffer format.
     * Weights are read-only during execution but may need GPU upload for GPU stages.
     *
     * @param info StageDumpInfo from stage->getDumpInfo()
     * @return Vector of CoherenceBuffer for weight tensors
     */
    std::vector<CoherenceBuffer> extractWeightBuffers(const StageDumpInfo &info);

    /**
     * @brief Extract CoherenceBuffer list from StageDumpInfo outputs
     *
     * Converts StageDumpInfo::OutputBuffer entries to CoherenceBuffer format.
     * Uses the tensor pointer if available in the OutputBuffer.
     *
     * @param info StageDumpInfo from stage->getDumpInfo()
     * @return Vector of CoherenceBuffer for output tensors
     */
    std::vector<CoherenceBuffer> extractOutputBuffers(const StageDumpInfo &info);

    /**
     * @brief Convert CoherencePolicy enum to string for logging
     *
     * @param policy The coherence policy
     * @return C-string representation
     */
    const char *toString(CoherencePolicy policy);

} // namespace llaminar2
