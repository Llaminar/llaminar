/**
 * @file DecodeAttentionGraphAssertions.h
 * @brief Shared captured-decode invariants for CUDA and HIP attention tests.
 *
 * Numerical equality alone cannot detect a deterministic launch policy that
 * serializes the whole KV range. Inspect immutable native graph metadata to
 * prove split parallelism independently of timings or optional PerfStats.
 * The scoped policy override is test-only and restores the process snapshot.
 */
#pragma once

#include "backends/IGPUGraphCapture.h"
#include "utils/DebugEnv.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace llaminar2::test
{
    /** @brief Select deterministic kernel policy for one isolated test scope. */
    class ScopedDeterministicKernelPolicy final
    {
    public:
        /**
         * @brief Save the cached policy and select the requested test setting.
         * @param enabled Whether this scope exercises deterministic mode.
         */
        explicit ScopedDeterministicKernelPolicy(bool enabled = true)
            : original_(mutableDebugEnv().gemm.deterministic)
        {
            mutableDebugEnv().gemm.deterministic = enabled;
        }

        /** @brief Restore the policy before the next test constructs a graph. */
        ~ScopedDeterministicKernelPolicy()
        {
            mutableDebugEnv().gemm.deterministic = original_;
        }

        ScopedDeterministicKernelPolicy(const ScopedDeterministicKernelPolicy &) = delete;
        ScopedDeterministicKernelPolicy &operator=(const ScopedDeterministicKernelPolicy &) = delete;

    private:
        bool original_; ///< Borrowed process policy, never a production mirror.
    };

    /**
     * @brief Prove a captured producer distributes each row over KV partitions.
     * @param graph Completed native graph, inspected without executing it.
     * @param producer_name Unique backend kernel-name fragment, excluding its reducer.
     * @param heads Expected participant-local query heads.
     * @param rows Expected independent serial-equivalent query rows.
     *
     * Call only with a geometry containing enough KV work for multiple splits.
     * Neither a host-side launch estimate nor a numerical-only check proves
     * that the actual captured producer retained its parallel grid.
     */
    inline void expectParallelDecodeProducer(
        const IGPUGraphCapture &graph,
        const char *producer_name,
        int heads,
        int rows)
    {
        std::vector<GPUGraphKernelNodeInfo> nodes;
        std::string error;
        ASSERT_TRUE(graph.inspectKernelNodes(nodes, &error)) << error;
        size_t producers = 0;
        for (const auto &node : nodes)
        {
            if (node.name.find(producer_name) == std::string::npos)
                continue;
            ++producers;
            EXPECT_EQ(node.grid_x, static_cast<uint32_t>(heads));
            EXPECT_EQ(node.grid_z, static_cast<uint32_t>(rows));
            EXPECT_GT(node.grid_y, 1u)
                << "Determinism must retain ordered KV-split parallelism: " << node.name;
        }
        EXPECT_EQ(producers, 1u) << "Missing or duplicated decode producer: " << producer_name;
    }
}
