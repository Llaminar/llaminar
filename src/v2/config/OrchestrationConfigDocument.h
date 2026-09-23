/**
 * @file OrchestrationConfigDocument.h
 * @brief Lossless, versioned configuration exchange for plan/apply and MPI.
 *
 * One codec carries the existing typed configuration; it neither selects a
 * strategy nor admits memory. In particular, serialized capacities are inputs
 * to fresh PhysicalMemoryAuthority admission, never transferable live grants.
 * Optional policy values remain optional instead of becoming today's defaults.
 * The current version carries ordered discovery-rank selection, explicit
 * host-participation intent, optional planning workload and exact physical
 * backend-device counts. Older documents are rejected rather
 * than silently assigning newly introduced policy defaults.
 */
#pragma once

#include "OrchestrationConfig.h"
#include <string>
#include <string_view>

namespace llaminar2
{
    /** @brief Exact field schema shared by the writer and fail-closed reader. */
    inline constexpr int kOrchestrationConfigDocumentSchemaVersion = 5;

    /**
     * @brief Encode every supported configuration field without normalization.
     * @param config Declarative or resolved configuration; no runtime resources.
     * @return A self-contained JSON document accepted by the public config parser.
     * @throws std::invalid_argument for invalid enums, nonfinite values, or the
     *         unsupported recursive topology-tree execution interface.
     */
    [[nodiscard]] std::string serializeOrchestrationConfig(const OrchestrationConfig &config);

    /**
     * @brief Decode exactly one complete document without supplying defaults.
     * @param document Versioned JSON produced by the shared writer.
     * @return Independently owned configuration, including expert placement.
     * @throws std::invalid_argument for unknown/missing/duplicate fields, wrong
     *         types, numeric overflow, unknown schema/enums, or unsupported trees.
     *
     * Semantic model, topology, and memory validation remains in the same
     * production admission path as CLI input; decoding is not certification.
     */
    [[nodiscard]] OrchestrationConfig deserializeOrchestrationConfig(std::string_view document);
}
