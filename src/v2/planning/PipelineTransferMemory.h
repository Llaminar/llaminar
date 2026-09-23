/**
 * @file PipelineTransferMemory.h
 * @brief Shared physical geometry for captured cross-backend pipeline channels.
 *
 * An adjacent boundary has one forward activation slot and one reverse metadata
 * slot. Each GPU leader owns two private cursors; the earlier domain contributes
 * the two mapped host allocations exactly once. This is a typed BOM input, not
 * an admission authority or live allocation ledger.
 */
#pragma once
#include <cstddef>
#include <cstdint>

namespace llaminar2
{
/** @brief Which layer interval owns one endpoint of an adjacent boundary. */
enum class PipelineBoundarySide : std::uint8_t
{
    EarlierDomain, ///< Publishes activation and owns the shared host mappings.
    LaterDomain,   ///< Publishes metadata; does not charge the mappings again.
};

/** @brief Exact geometry consumed by both memory admission and materialization. */
struct PipelineTransferMemory
{
    std::size_t activation_capacity_bytes;
    std::size_t metadata_capacity_bytes;
    std::size_t device_bytes_per_boundary;
    std::size_t host_bytes_per_boundary;

    /** @brief Size both directions from the actual retained graph envelope.
     * @param width Full replicated FP32 activation row width, not its TP shard.
     * @param prefill_rows Largest retained physical prefill graph.
     * @param verifier_rows Retained grouped verifier rows; one with MTP off.
     * @return TransferEngine's exact page-rounded mappings and private cursors.
     * @throws std::invalid_argument for a zero dimension.
     * @throws std::overflow_error for unrepresentable physical allocation sizes. */
    static PipelineTransferMemory forRows(std::size_t width, std::size_t prefill_rows,
        std::size_t verifier_rows);
};
}
