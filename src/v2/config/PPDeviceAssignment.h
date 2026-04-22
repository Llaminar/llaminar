/**
 * @file PPDeviceAssignment.h
 * @brief Assignment of layers to a device in Pipeline Parallelism mode
 *
 * Part of the Unified Multi-Device Orchestration Architecture (Phase 3).
 * Describes which layers, embedding, and LM head a specific device handles.
 *
 * @see docs/v2/UNIFIED_MULTI_DEVICE_ORCHESTRATION_DESIGN.md
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "../../backends/DeviceId.h"
#include <string>

namespace llaminar2
{

    /**
     * @brief Assignment of layers to a device in PP mode
     *
     * Describes what portion of the model a specific device handles:
     * - Layer range (e.g., layers 0-11)
     * - Whether it handles embedding (first stage)
     * - Whether it handles LM head (last stage)
     *
     * This is used by:
     * - DeviceGraphOrchestrator to build the correct graph
     * - WeightManager to load only relevant weights
     * - KV cache creation to allocate for correct layer count
     *
     * ## Example
     *
     * For a 24-layer model with 2-stage PP:
     *
     * ```cpp
     * // Stage 0 (embedding stage)
     * PPDeviceAssignment stage0 {
     *     .device_index = 0,
     *     .device_id = DeviceId::cuda(0),
     *     .first_layer = 0,
     *     .last_layer = 12,
     *     .has_embedding = true,
     *     .has_lm_head = false,
     * };
     *
     * // Stage 1 (LM head stage)
     * PPDeviceAssignment stage1 {
     *     .device_index = 1,
     *     .device_id = DeviceId::cuda(1),
     *     .first_layer = 12,
     *     .last_layer = 24,
     *     .has_embedding = false,
     *     .has_lm_head = true,
     * };
     * ```
     */
    struct PPDeviceAssignment
    {
        /// Index of this device in RankOrchestrator's device list
        int device_index = -1;

        /// The device ID
        DeviceId device_id = DeviceId::cpu();

        /// First layer index (inclusive) this device handles
        int first_layer = 0;

        /// Last layer index (exclusive) this device handles
        int last_layer = 0;

        /// Does this device handle the embedding lookup?
        /// True for the first PP stage (stage with first_layer == 0)
        bool has_embedding = false;

        /// Does this device handle the LM head projection?
        /// True for the last PP stage (stage with last_layer == total_layers)
        bool has_lm_head = false;

        // =====================================================================
        // Computed Properties
        // =====================================================================

        /**
         * @brief Get number of layers this device handles
         * @return Layer count (last_layer - first_layer)
         */
        int layerCount() const { return last_layer - first_layer; }

        /**
         * @brief Check if a layer is handled by this device
         * @param layer_idx Layer index to check
         * @return true if layer_idx is in [first_layer, last_layer)
         */
        bool ownsLayer(int layer_idx) const
        {
            return layer_idx >= first_layer && layer_idx < last_layer;
        }

        /**
         * @brief Convert global layer index to local index
         *
         * Local index is used for KV cache indexing within this device's cache.
         *
         * @param global_layer_idx Global layer index
         * @return Local layer index (0-based within this device)
         * @throws std::out_of_range if layer not owned by this device
         */
        int toLocalLayerIndex(int global_layer_idx) const
        {
            if (!ownsLayer(global_layer_idx))
            {
                throw std::out_of_range(
                    "Layer " + std::to_string(global_layer_idx) +
                    " not owned by device " + std::to_string(device_index) +
                    " (range: [" + std::to_string(first_layer) + ", " +
                    std::to_string(last_layer) + ")");
            }
            return global_layer_idx - first_layer;
        }

        /**
         * @brief Convert local layer index to global index
         *
         * @param local_layer_idx Local layer index (0-based within this device)
         * @return Global layer index
         */
        int toGlobalLayerIndex(int local_layer_idx) const
        {
            return first_layer + local_layer_idx;
        }

        /**
         * @brief Check if assignment is valid
         *
         * Validates:
         * - device_index >= 0
         * - first_layer >= 0
         * - last_layer > first_layer
         *
         * @return true if valid
         */
        bool isValid() const
        {
            return device_index >= 0 &&
                   first_layer >= 0 &&
                   last_layer > first_layer;
        }

        /**
         * @brief Create string representation for debugging
         */
        std::string toString() const
        {
            std::string s = "PPDeviceAssignment{";
            s += "device_index=" + std::to_string(device_index);
            s += ", device=" + device_id.to_string();
            s += ", layers=[" + std::to_string(first_layer) + "," + std::to_string(last_layer) + ")";
            if (has_embedding)
                s += ", has_embedding";
            if (has_lm_head)
                s += ", has_lm_head";
            s += "}";
            return s;
        }

        // =====================================================================
        // Factory Methods
        // =====================================================================

        /**
         * @brief Create assignment for first PP stage (has embedding)
         *
         * @param device_index Index in orchestrator's device list
         * @param device_id The device
         * @param first_layer Should be 0 for first stage
         * @param last_layer Last layer (exclusive)
         */
        static PPDeviceAssignment firstStage(
            int device_index,
            DeviceId device_id,
            int first_layer,
            int last_layer)
        {
            return {
                .device_index = device_index,
                .device_id = device_id,
                .first_layer = first_layer,
                .last_layer = last_layer,
                .has_embedding = true,
                .has_lm_head = false,
            };
        }

        /**
         * @brief Create assignment for last PP stage (has LM head)
         *
         * @param device_index Index in orchestrator's device list
         * @param device_id The device
         * @param first_layer First layer index
         * @param last_layer Should be total_layers for last stage
         */
        static PPDeviceAssignment lastStage(
            int device_index,
            DeviceId device_id,
            int first_layer,
            int last_layer)
        {
            return {
                .device_index = device_index,
                .device_id = device_id,
                .first_layer = first_layer,
                .last_layer = last_layer,
                .has_embedding = false,
                .has_lm_head = true,
            };
        }

        /**
         * @brief Create assignment for middle PP stage
         *
         * @param device_index Index in orchestrator's device list
         * @param device_id The device
         * @param first_layer First layer index
         * @param last_layer Last layer (exclusive)
         */
        static PPDeviceAssignment middleStage(
            int device_index,
            DeviceId device_id,
            int first_layer,
            int last_layer)
        {
            return {
                .device_index = device_index,
                .device_id = device_id,
                .first_layer = first_layer,
                .last_layer = last_layer,
                .has_embedding = false,
                .has_lm_head = false,
            };
        }

        /**
         * @brief Create assignment for single-device (no PP)
         *
         * Full model on one device.
         *
         * @param device_id The device
         * @param total_layers Total layers in the model
         */
        static PPDeviceAssignment singleDevice(
            DeviceId device_id,
            int total_layers)
        {
            return {
                .device_index = 0,
                .device_id = device_id,
                .first_layer = 0,
                .last_layer = total_layers,
                .has_embedding = true,
                .has_lm_head = true,
            };
        }
    };

} // namespace llaminar2
