/**
 * @file CapturedGraphMemoryEstimator.h
 * @brief Admission envelope for opaque CUDA/HIP graph-executable storage.
 *
 * Native graph drivers retain internal node and executable metadata outside
 * Llaminar's tensor/workspace arenas. The bytes are not allocator-addressable,
 * but they are physically resident and scale with model layers and retained
 * identities. This estimator makes that otherwise invisible storage an
 * explicit capacity term before expert placement fills the device.
 */

#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    /** Base driver allocation observed for one complete executable identity. */
    inline constexpr std::size_t kCapturedGraphBaseBytesPerExecutable =
        2ULL * 1024ULL * 1024ULL;

    /**
     * Per-layer envelope covering captured nodes and backend executable state.
     *
     * Production CUDA/ROCm graph families contain several kernel/memcpy/event
     * nodes per transformer layer. Half a MiB per layer bounds the measured
     * complete Qwen35 MoE graph cost while remaining proportional for smaller
     * and larger model graphs.
     */
    inline constexpr std::size_t kCapturedGraphBytesPerModelLayer =
        512ULL * 1024ULL;

    /**
     * @brief Physical native-executable inventory retained by one GPU runner.
     *
     * A logical forward graph is not necessarily one CUDA/HIP executable. An
     * explicitly heterogeneous ExpertOverlay transaction is cut at every
     * authenticated host boundary, so one logical graph identity owns several
     * native captured segments. MTP control, verifier, sampler, and publication
     * fragments are native executables too, but are not complete model graphs.
     * Keeping those populations separate prevents capacity admission from
     * multiplying an ambiguous scalar by the model layer count.
     */
    struct CapturedGraphExecutableInventory
    {
        /** Number of independently retained complete forward-graph identities. */
        std::size_t model_graph_identity_count = 0u;
        /** Native captured segments owned by each complete forward identity. */
        std::size_t native_segments_per_model_graph = 0u;
        /** Retained native helper/control executables outside complete graphs. */
        std::size_t auxiliary_native_executable_count = 0u;

        /**
         * @brief Return whether the declaration has one unambiguous shape.
         *
         * An empty complete-graph population must name zero segments; a live
         * population must name at least one. Auxiliary-only inventories remain
         * representable for controller-only runners.
         */
        [[nodiscard]] bool valid() const noexcept
        {
            return model_graph_identity_count == 0u
                       ? native_segments_per_model_graph == 0u
                       : native_segments_per_model_graph != 0u;
        }

        /** @brief Construct the conventional one-executable-per-graph layout. */
        [[nodiscard]] static CapturedGraphExecutableInventory monolithic(
            std::size_t model_graph_identities,
            std::size_t auxiliary_native_executables = 0u) noexcept
        {
            return {
                .model_graph_identity_count = model_graph_identities,
                .native_segments_per_model_graph =
                    model_graph_identities == 0u ? 0u : 1u,
                .auxiliary_native_executable_count =
                    auxiliary_native_executables,
            };
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

        /** @brief Checked sum used by the opaque-driver memory BOM. */
        [[nodiscard]] inline std::size_t checkedAdd(
            std::size_t lhs,
            std::size_t rhs,
            const char *description)
        {
            if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
                throw std::overflow_error(description);
            return lhs + rhs;
        }
    } // namespace captured_graph_memory_detail

    /**
     * @brief Estimate driver bytes for an exact physical executable inventory.
     * @param model_layer_count Number of layers represented by each complete
     *        forward-graph identity.
     * @param inventory Complete model-graph segmentation and auxiliary family.
     * @return Checked conservative byte envelope.
     * @throws std::invalid_argument for invalid model geometry or inventory.
     * @throws std::overflow_error when the result exceeds size_t.
     *
     * Every native executable pays the measured CUDA/HIP base allocation.
     * Complete model graphs additionally pay one layer-metadata allowance per
     * logical layer irrespective of segmentation. Auxiliary executables pay
     * one metadata allowance each because even a small retained fragment owns
     * kernel parameters, events, and launch metadata. For a monolithic family
     * with no auxiliaries this is byte-identical to the historical estimate.
     */
    [[nodiscard]] inline std::size_t
    estimateCapturedGraphExecutableBytes(
        int model_layer_count,
        const CapturedGraphExecutableInventory &inventory)
    {
        if (model_layer_count <= 0)
        {
            throw std::invalid_argument(
                "Captured graph memory estimation requires positive model layers");
        }
        if (!inventory.valid())
        {
            throw std::invalid_argument(
                "Captured graph executable inventory has an ambiguous model-graph segmentation");
        }

        using namespace captured_graph_memory_detail;
        const std::size_t native_model_executables = checkedMultiply(
            inventory.model_graph_identity_count,
            inventory.native_segments_per_model_graph,
            "Captured graph native model executable count overflows size_t");
        const std::size_t native_executables = checkedAdd(
            native_model_executables,
            inventory.auxiliary_native_executable_count,
            "Captured graph total native executable count overflows size_t");
        const std::size_t executable_base_bytes = checkedMultiply(
            native_executables,
            kCapturedGraphBaseBytesPerExecutable,
            "Captured graph executable base memory overflows size_t");
        const std::size_t model_layer_instances = checkedMultiply(
            inventory.model_graph_identity_count,
            static_cast<std::size_t>(model_layer_count),
            "Captured graph model-layer instance count overflows size_t");
        const std::size_t metadata_units = checkedAdd(
            model_layer_instances,
            inventory.auxiliary_native_executable_count,
            "Captured graph metadata unit count overflows size_t");
        const std::size_t metadata_bytes = checkedMultiply(
            metadata_units,
            kCapturedGraphBytesPerModelLayer,
            "Captured graph metadata memory overflows size_t");
        return checkedAdd(
            executable_base_bytes,
            metadata_bytes,
            "Captured graph complete memory envelope overflows size_t");
    }

    /**
     * @brief Estimate persistent native-driver bytes for retained executables.
     * @param model_layer_count Number of raw model graph layers.
     * @param executable_count Simultaneously retained graph identities.
     * @return Checked conservative byte envelope.
     * @throws std::invalid_argument for non-positive model geometry.
     * @throws std::overflow_error when the result exceeds size_t.
     */
    [[nodiscard]] inline std::size_t
    estimateCapturedGraphExecutableBytes(
        int model_layer_count,
        std::size_t executable_count)
    {
        return estimateCapturedGraphExecutableBytes(
            model_layer_count,
            CapturedGraphExecutableInventory::monolithic(executable_count));
    }
} // namespace llaminar2
