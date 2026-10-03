/**
 * @file CapturedChannelGraphProof.h
 * @brief Native graph proof for complete fused and parallel transfer operations.
 *
 * Functional byte checks alone cannot detect an accidentally retained three-
 * launch protocol. Inspect the actual capture without changing execution and
 * require exactly the selected implementation for every message endpoint.
 */
#pragma once
#include "backends/IGPUGraphCapture.h"
#include "transfer/CapturedTransferKernelPlan.h"
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2::counted_channel_test
{
    /**
     * @brief Reject omitted phases, duplicate legacy nodes or a lost fusion.
     * @param graph Complete recorded participant graph, not a synthetic node list.
     * @param maximum_bytes Positive immutable message capacity used at capture.
     * @param messages Number of complete transfer operations in this participant.
     * @throws std::runtime_error on failed inspection or any lowering mismatch.
     */
    inline void verifyCapturedChannelGraph(const IGPUGraphCapture &graph,
        std::uint64_t maximum_bytes, size_t messages)
    {
        std::vector<GPUGraphKernelNodeInfo> kernels;
        std::string error;
        if (!graph.inspectKernelNodes(kernels, &error))
            throw std::runtime_error("Captured-channel graph inspection failed: " + error);
        size_t fused = 0, acquire = 0, copy_publish = 0, total = 0;
        for (const auto &kernel : kernels)
        {
            if (kernel.name.find("capturedTransfer") == std::string::npos) continue;
            if (!kernel.name_resolved || !kernel.valid())
                throw std::runtime_error("Captured-channel kernel metadata is incomplete");
            ++total;
            if (kernel.name.find("capturedTransferFusedKernel") != std::string::npos)
            {
                ++fused;
                if (kernel.grid_x != 1 || kernel.grid_y != 1 || kernel.grid_z != 1)
                    throw std::runtime_error("Fused channel must use exactly one cooperative block");
            }
            if (kernel.name.find("capturedTransferAcquireKernel") != std::string::npos) ++acquire;
            if (kernel.name.find("capturedTransferCopyPublishKernel") != std::string::npos) ++copy_publish;
        }
        const bool single = capturedTransferKernelPlan(maximum_bytes) == CapturedTransferKernelPlan::SingleBlock;
        if (fused != (single ? messages : 0) || acquire != (single ? 0 : messages) ||
            copy_publish != (single ? 0 : messages) || total != messages * (single ? 1 : 2))
            throw std::runtime_error("Captured-channel graph did not record exactly the complete selected lowering");
    }
}
