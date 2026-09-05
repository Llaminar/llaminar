/**
 * @file CapturedGraphMemoryEstimator.h
 * @brief Canonical admission contract for opaque CUDA/HIP graph storage.
 *
 * Native graph drivers retain internal node and executable metadata outside
 * Llaminar's tensor/workspace arenas. The bytes are not allocator-addressable,
 * but they are physically resident and scale with model layers and retained
 * identities. This estimator makes that otherwise invisible storage an
 * explicit capacity term before expert placement fills the device. Capture
 * compilation fragments are deliberately absent: a retained parent imports
 * those fragments but is one physical executable and therefore pays one
 * resident charge.
 */

#pragma once

#include "backends/GPUGraphMemoryContract.h"

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Physical native-executable inventory retained by one GPU runner.
     *
     * A heterogeneous forward may be recorded as many child compilation
     * fragments, but @c retained_parent composition exposes one executable to
     * inference. Only executable owners belong here. Semantic MTP branch
     * descriptors, graph-recording children, and host ticket programs are not
     * resident executables and cannot be smuggled into either population.
     */
    struct CapturedGraphExecutableInventory
    {
        /** Complete forward identities retained in each topology variant. */
        std::size_t model_graph_identity_count = 0u;
        /** Simultaneously resident snapshot/launch-topology variants. */
        std::size_t model_graph_topology_variant_count = 0u;
        /** Retained helper/control executables outside complete forwards. */
        std::size_t auxiliary_executable_count = 0u;

        /**
         * @brief Return whether the declaration has one unambiguous shape.
         *
         * Complete graph identities and topology variants must either both be
         * zero or both be positive. Auxiliary-only inventories remain valid
         * for controller-only runners.
         */
        [[nodiscard]] bool valid() const noexcept
        {
            return (model_graph_identity_count == 0u) ==
                   (model_graph_topology_variant_count == 0u);
        }

        /** @brief Construct the conventional one-executable-per-graph layout. */
        [[nodiscard]] static CapturedGraphExecutableInventory monolithic(
            std::size_t model_graph_identities,
            std::size_t auxiliary_native_executables = 0u) noexcept
        {
            return {
                .model_graph_identity_count = model_graph_identities,
                .model_graph_topology_variant_count =
                    model_graph_identities == 0u ? 0u : 1u,
                .auxiliary_executable_count =
                    auxiliary_native_executables,
            };
        }

        /**
         * @brief Return the checked number of physical executable owners.
         * @throws std::overflow_error when the declared inventory cannot be
         *         represented by @c size_t.
         */
        [[nodiscard]] std::size_t residentExecutableCount() const
        {
            if (!valid())
            {
                throw std::invalid_argument(
                    "Captured graph executable inventory has mismatched graph identities and topology variants");
            }
            if (model_graph_identity_count != 0u &&
                model_graph_topology_variant_count >
                    std::numeric_limits<std::size_t>::max() /
                        model_graph_identity_count)
            {
                throw std::overflow_error(
                    "Captured graph model topology variant count overflows size_t");
            }
            const std::size_t model_executables =
                model_graph_identity_count *
                model_graph_topology_variant_count;
            if (auxiliary_executable_count >
                std::numeric_limits<std::size_t>::max() -
                    model_executables)
            {
                throw std::overflow_error(
                    "Captured graph total resident executable count overflows size_t");
            }
            return model_executables + auxiliary_executable_count;
        }
    };

    /**
     * @brief Diagnostic-only capture compilation topology.
     *
     * This type intentionally has no byte-estimation API. It lets setup and
     * tests describe why a heterogeneous graph takes longer to record without
     * allowing compilation fragments to become a second VRAM authority.
     */
    struct CapturedGraphCompilationInventory
    {
        /** Logical complete graph identities compiled per topology variant. */
        std::size_t model_graph_identity_count = 0u;
        /** Simultaneously retained topology variants. */
        std::size_t model_graph_topology_variant_count = 0u;
        /** Capturable child units recorded before composing each parent. */
        std::size_t compilation_units_per_model_graph = 0u;

        /** @return Whether every zero/non-zero relationship is unambiguous. */
        [[nodiscard]] bool valid() const noexcept
        {
            const bool empty = model_graph_identity_count == 0u;
            return empty
                       ? model_graph_topology_variant_count == 0u &&
                             compilation_units_per_model_graph == 0u
                       : model_graph_topology_variant_count != 0u &&
                             compilation_units_per_model_graph != 0u;
        }
    };

    namespace captured_graph_memory_detail
    {
        /** @brief Checked product used by the opaque-driver memory BOM. */
        [[nodiscard]] inline std::size_t checkedMultiply(
            std::size_t lhs,
            std::size_t rhs,
            const char *description)
        {
            if (lhs != 0u &&
                rhs > std::numeric_limits<std::size_t>::max() / lhs)
            {
                throw std::overflow_error(description);
            }
            return lhs * rhs;
        }

    } // namespace captured_graph_memory_detail

    /**
     * @brief Reserve driver bytes for an exact physical executable inventory.
     * @param device Exact CUDA or ROCm device owning the executables.
     * @param inventory Complete resident executable family.
     * @return Checked backend-certified byte reservation.
     * @throws std::invalid_argument for a non-GPU device or invalid inventory.
     * @throws std::overflow_error when the result exceeds size_t.
     *
     * The driver owns opaque pooled storage, so Llaminar cannot allocate or
     * attribute exact bytes to one graph object. The backend-certified slot
     * unit and the exact resident slot inventory therefore produce one family
     * reservation. Runtime capture records each positive pool-growth event for
     * evidence, and certification compares their aggregate with this result.
     */
    [[nodiscard]] inline std::size_t
    estimateCapturedGraphExecutableBytes(
        DeviceId device,
        const CapturedGraphExecutableInventory &inventory)
    {
        if (!inventory.valid())
        {
            throw std::invalid_argument(
                "Captured graph executable inventory has mismatched graph identities and topology variants");
        }
        const std::size_t bytes_per_executable =
            GPUGraphMemoryContract::reservationBytesPerExecutable(device);
        return captured_graph_memory_detail::checkedMultiply(
            inventory.residentExecutableCount(),
            bytes_per_executable,
            "Captured graph resident memory reservation overflows size_t");
    }

    /**
     * @brief Estimate persistent native-driver bytes for retained executables.
     * @param device Exact CUDA or ROCm device owning the executables.
     * @param executable_count Simultaneously retained graph identities.
     * @return Checked conservative byte envelope.
     * @throws std::invalid_argument for non-positive model geometry.
     * @throws std::overflow_error when the result exceeds size_t.
     */
    [[nodiscard]] inline std::size_t
    estimateCapturedGraphExecutableBytes(
        DeviceId device,
        std::size_t executable_count)
    {
        return estimateCapturedGraphExecutableBytes(
            device,
            CapturedGraphExecutableInventory::monolithic(executable_count));
    }
} // namespace llaminar2
