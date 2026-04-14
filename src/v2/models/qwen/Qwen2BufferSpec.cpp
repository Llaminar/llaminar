/**
 * @file Qwen2BufferSpec.cpp
 * @brief Implementation of buffer allocation specifications for Qwen2 models
 * @author David Sanftenberg
 * @date January 2026
 */

#include "Qwen2BufferSpec.h"
#include "../../utils/Logger.h"

namespace llaminar2
{

    // =========================================================================
    // Row-Parallel Output Buffer Names
    // =========================================================================
    // These buffers hold the output of row-parallel matrix multiplications
    // (Wo projection, FFN down projection) and need allreduce for TP.
    //
    // When using HOST backend with heterogeneous GPUs, these are
    // communicated via host-staged transfers.

    static const std::unordered_set<std::string> ROW_PARALLEL_OUTPUT_SUFFIXES = {
        // FFN down projection outputs
        "ffn_down_output",
        "ffn_down_allreduce",
        "ffn_output", // DeviceGraphBufferManager may use this name

        // Attention Wo projection outputs
        "attention_wo_output",
        "attn_wo_output",
        "attn_wo_allreduce",
        "wo_output",
        "attn_proj" // DeviceGraphBufferManager may use this name
    };

    // =========================================================================
    // Public Methods
    // =========================================================================

    const std::unordered_set<std::string> &Qwen2BufferSpec::getRowParallelOutputSuffixes()
    {
        return ROW_PARALLEL_OUTPUT_SUFFIXES;
    }

    bool Qwen2BufferSpec::matchesRowParallelOutput(const std::string &buffer_name)
    {
        // Empty name never matches
        if (buffer_name.empty())
        {
            return false;
        }

        // Check exact match first (most common case)
        if (ROW_PARALLEL_OUTPUT_SUFFIXES.count(buffer_name))
        {
            return true;
        }

        // Check suffix match for layer-prefixed names (e.g., "layer0_ffn_down_output")
        for (const auto &suffix : ROW_PARALLEL_OUTPUT_SUFFIXES)
        {
            // Buffer name must be longer than suffix (has a prefix)
            // and end with the suffix
            if (buffer_name.size() > suffix.size())
            {
                // Check if buffer_name ends with suffix
                size_t start_pos = buffer_name.size() - suffix.size();
                if (buffer_name.compare(start_pos, suffix.size(), suffix) == 0)
                {
                    // Optionally verify there's a separator (underscore) before suffix
                    // This prevents false matches like "my_ffn_output" matching "ffn_output"
                    // but allows "layer0_ffn_output"
                    if (start_pos > 0)
                    {
                        char separator = buffer_name[start_pos - 1];
                        if (separator == '_' || separator == '.')
                        {
                            return true;
                        }
                    }
                }
            }
        }

        return false;
    }

    bool Qwen2BufferSpec::requiresBARBacked(
        const std::string &buffer_name,
        CollectiveBackendType backend_type,
        int tp_degree)
    {
        // Legacy stub — always returns false
        (void)buffer_name;
        (void)backend_type;
        (void)tp_degree;
        return false;
    }

    AllocationStrategy Qwen2BufferSpec::getAllocationStrategy(
        const std::string &buffer_name,
        CollectiveBackendType backend_type,
        int tp_degree)
    {
        (void)buffer_name;
        (void)backend_type;
        (void)tp_degree;
        return AllocationStrategy::STANDARD;
    }

} // namespace llaminar2
