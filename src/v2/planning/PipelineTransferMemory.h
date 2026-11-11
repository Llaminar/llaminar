/**
 * @file PipelineTransferMemory.h
 * @brief Useful pipeline message geometry and exact cross-backend mapped storage BOM.
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
    /** @brief Resolve useful native message maxima without quoting mapped payload/control storage.
     * @param width Actual full residual row width. @param prefill_rows Admitted prefill row maximum.
     * @param verifier_rows Admitted exact verifier maximum. @return Useful maxima and zero mapped storage.
     */
    static PipelineTransferMemory nativeForRows(std::size_t width, std::size_t prefill_rows,
        std::size_t verifier_rows);
    /** @return Exact participant-private device receipt/header/error bytes for an incoming native TP fanout. */
    static std::size_t fanoutDeviceBytes() noexcept;
    /** @return Exact page-rounded immutable native fanout operand backing per receiving TP member. */
    static std::size_t fanoutHostBytes();
};
}
