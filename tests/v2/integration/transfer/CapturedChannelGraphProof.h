/**
 * @file CapturedChannelGraphProof.h
 * @brief Native graph proof for complete fused and parallel transfer operations.
 *
 * Functional byte checks alone cannot detect an accidentally retained three-
 * launch protocol. Inspect the actual capture without changing execution and
 * require exactly the selected implementation for every message endpoint.
 * Mixed argument packets additionally authenticate each packet's exact native
 * grid and the separately fused final partial packet, independent of node order.
 */
#pragma once
#include "backends/IGPUGraphCapture.h"
#include "transfer/CapturedTransferKernelPlan.h"
#include "transfer/CapturedTransferChannelBatch.h"
#include <algorithm>
#include <span>
#include <tuple>
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
    /**
     * @brief Inspect actual packet grid/cardinality, including a final partial packet.
     * @param graph Original retained participant graph with both transfer phases.
     * @param maximum_bytes Positive immutable endpoint capacity.
     * @param endpoints Independent original channels in each phase.
     * @param phases Producer/consumer batches recorded by this participant.
     */
    inline void verifyCapturedChannelBatchGraph(const IGPUGraphCapture &graph,
        std::uint64_t maximum_bytes, size_t endpoints, size_t phases)
    {
        std::vector<GPUGraphKernelNodeInfo> kernels;
        std::string error;
        if (!graph.inspectKernelNodes(kernels, &error))
            throw std::runtime_error("Captured batch inspection failed: " + error);
        const bool single = capturedTransferBatchKernelPlan(maximum_bytes) == CapturedTransferKernelPlan::SingleBlock;
        size_t fused = 0, acquire = 0, copy = 0, total = 0, endpoint_ctas = 0;
        for (const auto &kernel : kernels)
        {
            if (kernel.name.find("capturedTransfer") == std::string::npos) continue;
            if (!kernel.valid() || !kernel.name_resolved || kernel.grid_z != 1 ||
                !kernel.grid_y || kernel.grid_y > CapturedTransferChannelBatch::kCapacity)
                throw std::runtime_error("Captured batch endpoint grid is malformed");
            ++total;
            if (kernel.name.find("capturedTransferBatchFusedKernel") != std::string::npos)
            {
                ++fused; endpoint_ctas += kernel.grid_y;
                if (kernel.grid_x != 1 || kernel.block_x != 256)
                    throw std::runtime_error("Each fused batch endpoint needs its own one-CTA copy coordinates");
            }
            else if (kernel.name.find("capturedTransferBatchAcquireKernel") != std::string::npos)
            {
                ++acquire; endpoint_ctas += kernel.grid_y;
                if (kernel.grid_x != 1) throw std::runtime_error("Batch acquire must be one CTA per endpoint");
            }
            else if (kernel.name.find("capturedTransferBatchCopyPublishKernel") != std::string::npos)
            {
                ++copy;
                if (kernel.block_x != 256) throw std::runtime_error("Parallel endpoint copy changed block geometry");
            }
            else throw std::runtime_error("Captured exchange retained an individual endpoint lowering");
        }
        const size_t packets = (endpoints + CapturedTransferChannelBatch::kCapacity - 1) /
            CapturedTransferChannelBatch::kCapacity * phases;
        if (fused != (single ? packets : 0) || acquire != (single ? 0 : packets) ||
            copy != (single ? 0 : packets) || total != packets * (single ? 1 : 2) ||
            endpoint_ctas != endpoints * phases)
            throw std::runtime_error("Captured exchange omitted/duplicated a packet or endpoint");
    }

    /** @brief Test-authored geometry for one original argument packet. */
    struct CapturedChannelBatchPacketProof
    {
        size_t endpoints = 0; ///< Original independent endpoints in this packet.
        CapturedTransferKernelPlan lowering = CapturedTransferKernelPlan::SingleBlock; ///< Expected original packet lowering.
        std::uint32_t copy_grid_x = 0; ///< Explicit expected bounded payload-grid maximum.
    };

    /**
     * @brief Verify exact mixed-packet families and grids without assuming node order.
     * @param graph Actual retained participant graph, not a synthetic packet list.
     * @param packets Fixture-authored expected original packet geometries.
     * @param phases Number of producer/consumer sets in this participant graph.
     * @param acquire_threads Native CUDA warp32 or HIP wave64 acquire block.
     * @throws std::runtime_error For any missing/duplicate/legacy node or grid mismatch.
     *
     * Uniform callers retain their original overload. This overload authenticates
     * each mixed packet rather than classifying the whole cohort by one maximum.
     * The explicit expected grid is a test oracle, not a new production policy.
     * Per-endpoint copy coordinates/counts are proved by live-count/poison replay.
     */
    inline void verifyCapturedChannelBatchGraph(const IGPUGraphCapture &graph,
        std::span<const CapturedChannelBatchPacketProof> packets, size_t phases,
        std::uint32_t acquire_threads)
    {
        if (packets.empty() || phases == 0 || (acquire_threads != 32 && acquire_threads != 64))
            throw std::runtime_error("Mixed channel graph proof requires exact packets/phases/native acquire block");
        // Family, gridX, gridY, blockX: source-node enumeration order is irrelevant.
        using Geometry = std::tuple<int, std::uint32_t, std::uint32_t, std::uint32_t>;
        std::vector<Geometry> expected, observed;
        for (size_t phase = 0; phase < phases; ++phase)
            for (const auto &packet : packets)
            {
                if (packet.endpoints == 0 || packet.endpoints > CapturedTransferChannelBatch::kCapacity ||
                    packet.copy_grid_x == 0)
                    throw std::runtime_error("Mixed channel packet proof has invalid endpoint/grid geometry");
                const auto endpoints = static_cast<std::uint32_t>(packet.endpoints);
                if (packet.lowering == CapturedTransferKernelPlan::SingleBlock)
                {
                    if (packet.copy_grid_x != 1)
                        throw std::runtime_error("Fused partial packet requires exactly one copy block per endpoint");
                    expected.emplace_back(0, 1, endpoints, 256);
                }
                else if (packet.lowering == CapturedTransferKernelPlan::ParallelCopy)
                {
                    expected.emplace_back(1, 1, endpoints, acquire_threads);
                    expected.emplace_back(2, packet.copy_grid_x, endpoints, 256);
                }
                else throw std::runtime_error("Mixed channel packet proof has unknown lowering");
            }
        std::vector<GPUGraphKernelNodeInfo> kernels;
        std::string error;
        if (!graph.inspectKernelNodes(kernels, &error))
            throw std::runtime_error("Mixed captured batch inspection failed: " + error);
        for (const auto &kernel : kernels)
        {
            if (kernel.name.find("capturedTransfer") == std::string::npos) continue;
            if (!kernel.valid() || !kernel.name_resolved || kernel.grid_z != 1 ||
                kernel.block_y != 1 || kernel.block_z != 1)
                throw std::runtime_error("Mixed captured batch kernel has incomplete geometry");
            int family;
            if (kernel.name.find("capturedTransferBatchFusedKernel") != std::string::npos) family = 0;
            else if (kernel.name.find("capturedTransferBatchAcquireKernel") != std::string::npos) family = 1;
            else if (kernel.name.find("capturedTransferBatchCopyPublishKernel") != std::string::npos) family = 2;
            else throw std::runtime_error("Mixed captured batch retained an individual endpoint lowering");
            observed.emplace_back(family, kernel.grid_x, kernel.grid_y, kernel.block_x);
        }
        std::sort(expected.begin(), expected.end());
        std::sort(observed.begin(), observed.end());
        if (observed != expected)
            throw std::runtime_error("Mixed captured batch omitted/duplicated a packet or changed its exact native grid");
    }

}
