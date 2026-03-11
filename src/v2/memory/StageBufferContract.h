/**
 * @file StageBufferContract.h
 * @brief Declarative I/O specification for compute stages
 *
 * A StageBufferContract declares which buffers a stage reads, writes,
 * and uses as workspace. The GraphExecutor uses this contract to:
 *   1. Cohere inputs before execution
 *   2. Allocate output storage
 *   3. Mark outputs dirty after execution
 *   4. Validate borrow safety (no aliased write conflicts)
 *
 * This replaces StageDumpInfo as the coherence driver. StageDumpInfo
 * remains available for debugging/dumping but is no longer load-bearing.
 */

#pragma once

#include <cstddef>
#include <vector>
#include "BufferId.h"
#include "BufferAccess.h"

namespace llaminar2
{
    class ITensor;
}

namespace llaminar2
{

    /**
     * @brief Describes one buffer binding for a stage.
     */
    struct BufferBinding
    {
        BufferId id;                 ///< Which buffer
        BufferAccess access;         ///< READ, WRITE, or READWRITE
        const char *dtype = nullptr; ///< Expected dtype string for validation (optional)
    };

    /**
     * @brief Describes a workspace / scratch buffer need.
     */
    struct WorkspaceDesc
    {
        const char *name = nullptr; ///< Lookup key (e.g., "attn_scores")
        size_t size_bytes = 0;      ///< Required size
        size_t alignment = 64;      ///< Alignment requirement
        bool required = true;       ///< Fail if cannot allocate?
    };

    /**
     * @brief Complete I/O contract for a compute stage.
     *
     * Built via fluent builder methods in stage constructors:
     * @code
     * return StageBufferContract::build()
     *     .addInput(BufferId::HIDDEN_STATE, "FP32")
     *     .addWeight(BufferId::Q_PROJ, "IQ4_NL")
     *     .addOutput(BufferId::Q_PROJ, "FP32");
     * @endcode
     */
    struct StageBufferContract
    {
        std::vector<BufferBinding> inputs;     ///< Read-only activation buffers (arena-managed)
        std::vector<BufferBinding> outputs;    ///< Write-only output buffers (arena-managed)
        std::vector<ITensor *> weight_tensors; ///< Read-only model weights (external, NOT in arena)
        std::vector<BufferBinding> inouts;     ///< Read-write (e.g., allreduce in-place, arena-managed)
        std::vector<WorkspaceDesc> workspaces; ///< Scratch buffers

        /// Create an empty contract (builder entry point)
        static StageBufferContract build() { return {}; }

        /// True if this contract has any bindings at all
        bool empty() const
        {
            return inputs.empty() && outputs.empty() && weight_tensors.empty() && inouts.empty();
        }

        /// Total number of buffer bindings (excluding workspaces)
        size_t bindingCount() const
        {
            return inputs.size() + outputs.size() + weight_tensors.size() + inouts.size();
        }

        // ── Fluent builder methods ──────────────────────────────────────────

        StageBufferContract &addInput(BufferId id, const char *dtype = nullptr)
        {
            inputs.push_back({id, BufferAccess::READ, dtype});
            return *this;
        }

        StageBufferContract &addOutput(BufferId id, const char *dtype = nullptr)
        {
            outputs.push_back({id, BufferAccess::WRITE, dtype});
            return *this;
        }

        StageBufferContract &addWeight(ITensor *tensor)
        {
            if (tensor)
                weight_tensors.push_back(tensor);
            return *this;
        }

        StageBufferContract &addInOut(BufferId id, const char *dtype = nullptr)
        {
            inouts.push_back({id, BufferAccess::READWRITE, dtype});
            return *this;
        }

        StageBufferContract &addWorkspace(const char *name, size_t size,
                                          size_t align = 64, bool required = true)
        {
            workspaces.push_back({name, size, align, required});
            return *this;
        }

        // ── Query helpers ───────────────────────────────────────────────────

        /// Find all arena-managed bindings that require READ access (inputs + inouts)
        /// Note: weights are NOT included — they're external ITensor* pointers
        std::vector<BufferBinding> allArenaReads() const
        {
            std::vector<BufferBinding> result;
            result.reserve(inputs.size() + inouts.size());
            result.insert(result.end(), inputs.begin(), inputs.end());
            result.insert(result.end(), inouts.begin(), inouts.end());
            return result;
        }

        /// Find all bindings that require WRITE access (outputs + inouts)
        std::vector<BufferBinding> allWrites() const
        {
            std::vector<BufferBinding> result;
            result.reserve(outputs.size() + inouts.size());
            result.insert(result.end(), outputs.begin(), outputs.end());
            result.insert(result.end(), inouts.begin(), inouts.end());
            return result;
        }
    };

} // namespace llaminar2
