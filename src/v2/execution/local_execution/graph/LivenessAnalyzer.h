/**
 * @file LivenessAnalyzer.h
 * @brief Buffer liveness analysis for memory optimization
 * @author David Sanftenberg
 * @date December 2025
 *
 * LivenessAnalyzer computes buffer lifetimes across a compute graph and
 * determines which SCRATCH buffers can share memory (alias) based on
 * non-overlapping lifetimes.
 *
 * This enables significant memory savings by reusing buffers that are
 * not simultaneously live. For example, attention Q/K/V buffers can be
 * reused for FFN gate/up/down buffers since attention completes before FFN.
 *
 * Algorithm:
 * 1. Get topological execution order from ComputeGraph
 * 2. For each stage, query getBufferRequirements()
 * 3. Track first-use and last-use stage index for each buffer
 * 4. Group non-overlapping SCRATCH buffers by interval graph coloring
 * 5. Within each group, allocate single buffer sized to maximum
 */

#pragma once

#include "../../debug/BufferRole.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstddef>

namespace llaminar2
{

    // Forward declarations
    class ComputeGraph;

    /**
     * @brief Tracks the lifetime of a single buffer across graph execution
     */
    struct BufferLiveness
    {
        std::string buffer_name;      ///< Unique buffer identifier (stage_name::buffer_name)
        std::string stage_name;       ///< Stage that owns this buffer
        size_t first_use_idx = 0;     ///< Stage index where buffer is first accessed
        size_t last_use_idx = 0;      ///< Stage index where buffer is last accessed
        BufferRole role;              ///< Buffer role (only SCRATCH eligible for aliasing)
        BufferTensorType tensor_type; ///< Tensor type for compatibility checking
        std::vector<size_t> shape;    ///< Buffer shape
        size_t size_bytes = 0;        ///< Total size in bytes

        /**
         * @brief Check if this buffer's lifetime overlaps with another
         * @param other The other buffer to check
         * @return true if lifetimes overlap (cannot alias)
         */
        bool overlaps(const BufferLiveness &other) const
        {
            return !(last_use_idx < other.first_use_idx ||
                     other.last_use_idx < first_use_idx);
        }
    };

    /**
     * @brief Represents a group of buffers that can share memory
     *
     * All buffers in a group have non-overlapping lifetimes and compatible
     * tensor types, allowing them to alias to a single physical buffer.
     */
    struct AliasingGroup
    {
        std::vector<std::string> buffer_names; ///< Buffers in this group
        size_t max_size_bytes = 0;             ///< Size of largest buffer (allocation size)
        BufferTensorType tensor_type;          ///< Common tensor type
    };

    /**
     * @brief Analyzes buffer lifetimes and computes aliasing opportunities
     *
     * Usage:
     * @code
     * LivenessAnalyzer analyzer;
     *
     * // Analyze buffer lifetimes in graph
     * auto lifetimes = analyzer.analyze(graph);
     *
     * // Compute aliasing groups
     * auto groups = analyzer.computeAliasingGroups(lifetimes);
     *
     * // Get memory savings
     * auto [original, optimized] = analyzer.computeMemoryUsage(lifetimes, groups);
     * LOG_INFO("Memory savings: " << (original - optimized) << " bytes ("
     *          << (100.0 * (original - optimized) / original) << "%)");
     * @endcode
     */
    class LivenessAnalyzer
    {
    public:
        LivenessAnalyzer() = default;
        ~LivenessAnalyzer() = default;

        /**
         * @brief Analyze buffer lifetimes across a compute graph
         *
         * Traverses the graph in topological order, querying each stage's
         * buffer requirements and tracking first/last use indices.
         *
         * @param graph The compute graph to analyze
         * @return Vector of buffer lifetimes
         */
        std::vector<BufferLiveness> analyze(const ComputeGraph &graph) const;

        /**
         * @brief Compute groups of buffers that can share memory
         *
         * Uses interval graph coloring to find optimal aliasing:
         * - Only SCRATCH buffers are eligible for aliasing
         * - Buffers with overlapping lifetimes cannot alias
         * - Buffers must have compatible tensor types
         *
         * @param lifetimes Buffer lifetimes from analyze()
         * @return Vector of aliasing groups
         */
        std::vector<AliasingGroup> computeAliasingGroups(
            const std::vector<BufferLiveness> &lifetimes) const;

        /**
         * @brief Compute memory usage with and without aliasing
         *
         * @param lifetimes Buffer lifetimes
         * @param groups Aliasing groups from computeAliasingGroups()
         * @return Pair of (original_bytes, optimized_bytes)
         */
        std::pair<size_t, size_t> computeMemoryUsage(
            const std::vector<BufferLiveness> &lifetimes,
            const std::vector<AliasingGroup> &groups) const;

        /**
         * @brief Get memory savings as a percentage
         *
         * @param lifetimes Buffer lifetimes
         * @param groups Aliasing groups
         * @return Percentage savings (0.0 to 100.0)
         */
        double computeSavingsPercent(
            const std::vector<BufferLiveness> &lifetimes,
            const std::vector<AliasingGroup> &groups) const;

        /**
         * @brief Filter lifetimes to only SCRATCH buffers
         *
         * @param lifetimes All buffer lifetimes
         * @return Only SCRATCH buffer lifetimes
         */
        static std::vector<BufferLiveness> filterScratchBuffers(
            const std::vector<BufferLiveness> &lifetimes);

        /**
         * @brief Check if two buffers can potentially alias
         *
         * Buffers can alias if:
         * 1. Both are SCRATCH buffers
         * 2. Their lifetimes don't overlap
         * 3. They have compatible tensor types (same type or both floating point)
         *
         * @param a First buffer
         * @param b Second buffer
         * @return true if buffers can share memory
         */
        static bool canAlias(const BufferLiveness &a, const BufferLiveness &b);

    private:
        /**
         * @brief Compute size in bytes for a buffer descriptor
         */
        static size_t computeBufferSize(const BufferDescriptor &desc);

        /**
         * @brief Check if two tensor types are compatible for aliasing
         */
        static bool areTypesCompatible(BufferTensorType a, BufferTensorType b);
    };

} // namespace llaminar2
