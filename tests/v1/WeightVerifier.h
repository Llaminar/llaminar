/**
 * @file WeightVerifier.h
 * @brief Utility for verifying Llaminar weights against PyTorch reference snapshots
 * @author David Sanftenberg
 *
 * Standalone verification utility that compares loaded Llaminar weights with
 * PyTorch reference weights, accounting for MPI slicing and numerical precision.
 *
 * Design:
 * - Accepts ModelWeightsProvider for MPI-aware weight access
 * - Loads PyTorch .npy reference weights via NpzLoader
 * - Handles sliced weights by comparing only the local rank's slice
 * - Returns structured results with detailed statistics
 *
 * Usage:
 * @code
 * WeightVerifier verifier(provider.get(), "pytorch_snapshots_mapped/weights");
 *
 * // Verify single weight
 * auto result = verifier.verifyKeyWeight(0);
 * if (!result.passed) {
 *     LOG_ERROR("K weight mismatch: " << result.details);
 * }
 *
 * // Verify all weights for a layer
 * auto layer_result = verifier.verifyLayerWeights(0);
 * ASSERT_TRUE(layer_result.passed);
 * @endcode
 */

#pragma once

#include "NpzLoader.h"
#include "ModelWeightsProvider.h"
#include <string>
#include <memory>
#include <vector>

namespace llaminar
{
    // Import NpyArray from parity namespace for convenience
    using llaminar::parity::NpyArray;
    using llaminar::parity::NpzLoader;

    /**
     * @struct WeightVerificationResult
     * @brief Result of weight verification with detailed statistics
     */
    struct WeightVerificationResult
    {
        bool passed;           ///< True if verification passed within tolerance
        float max_abs_diff;    ///< Maximum absolute difference between weights
        float mean_abs_diff;   ///< Mean absolute difference
        float rel_l2_error;    ///< Relative L2 error: ||diff||_2 / ||pytorch||_2
        size_t total_elements; ///< Number of elements compared
        std::string details;   ///< Human-readable description

        /**
         * @brief Constructor for passed verification
         */
        WeightVerificationResult()
            : passed(true), max_abs_diff(0.0f), mean_abs_diff(0.0f),
              rel_l2_error(0.0f), total_elements(0), details("OK") {}

        /**
         * @brief Constructor for failed verification
         */
        static WeightVerificationResult failure(const std::string &reason)
        {
            WeightVerificationResult result;
            result.passed = false;
            result.details = reason;
            return result;
        }

        /**
         * @brief Format as string for logging
         */
        std::string toString() const;
    };

    /**
     * @class WeightVerifier
     * @brief Verify Llaminar weights against PyTorch reference snapshots
     *
     * This utility compares Llaminar's loaded weights with PyTorch reference weights,
     * accounting for MPI slicing and numerical precision tolerances.
     *
     * **Tolerance Policy:**
     * - `abs_tol`: Absolute difference threshold (default 1e-5)
     * - `rel_tol`: Relative error threshold (default 1e-4)
     * - Pass if: max_abs_diff < abs_tol AND rel_l2_error < rel_tol
     *
     * **MPI Slicing Handling:**
     * - Replicated weights: Compare full tensors
     * - Sliced weights: Extract rank's slice from PyTorch weight, compare with local
     *
     * **PyTorch Weight Format:**
     * - Files: `{snapshot_dir}/layer{N}_{Q,K,V,O}_WEIGHT.npy`
     * - Dtype: float32
     * - Layout: [out_features, in_features] (PyTorch Linear convention)
     */
    class WeightVerifier
    {
    public:
        /**
         * @brief Construct verifier with weights provider and snapshot directory
         * @param provider MPI-aware weights provider (not owned)
         * @param snapshot_dir Directory containing PyTorch .npy weight files
         * @param abs_tol Absolute difference tolerance (default 1e-5)
         * @param rel_tol Relative L2 error tolerance (default 1e-4)
         *
         * @throws std::invalid_argument if provider is nullptr
         * @throws std::runtime_error if snapshot_dir doesn't exist
         */
        WeightVerifier(QwenModelWeightsProvider *provider,
                       const std::string &snapshot_dir,
                       float abs_tol = 1e-5f,
                       float rel_tol = 1e-4f);

        /**
         * @brief Verify query projection weight for a layer
         * @param layer Layer index [0, n_layers)
         * @return Verification result with statistics
         */
        WeightVerificationResult verifyQueryWeight(int layer);

        /**
         * @brief Verify key projection weight for a layer
         * @param layer Layer index [0, n_layers)
         * @return Verification result with statistics
         */
        WeightVerificationResult verifyKeyWeight(int layer);

        /**
         * @brief Verify value projection weight for a layer
         * @param layer Layer index [0, n_layers)
         * @return Verification result with statistics
         */
        WeightVerificationResult verifyValueWeight(int layer);

        /**
         * @brief Verify output projection weight for a layer
         * @param layer Layer index [0, n_layers)
         * @return Verification result with statistics
         */
        WeightVerificationResult verifyOutputWeight(int layer);

        /**
         * @brief Verify all attention weights for a layer (Q, K, V, O)
         * @param layer Layer index [0, n_layers)
         * @return Verification result (passes only if all weights pass)
         *
         * If any weight fails, returns the first failure encountered.
         */
        WeightVerificationResult verifyLayerWeights(int layer);

        /**
         * @brief Verify all weights across all layers
         * @return Verification result (passes only if all layers pass)
         *
         * Useful for comprehensive parity testing.
         */
        WeightVerificationResult verifyAllWeights();

        /**
         * @brief Set verbosity for detailed logging
         * @param verbose If true, logs comparison details for each weight
         */
        void setVerbose(bool verbose) { verbose_ = verbose; }

    private:
        QwenModelWeightsProvider *provider_; ///< Weights provider (not owned)
        std::string snapshot_dir_;           ///< PyTorch snapshot directory
        float abs_tol_;                      ///< Absolute difference tolerance
        float rel_tol_;                      ///< Relative L2 error tolerance
        bool verbose_;                       ///< Enable detailed logging

        /**
         * @brief Load PyTorch reference weight from .npy file
         * @param layer Layer index
         * @param weight_type "Q", "K", "V", or "O"
         * @param out_array Output array to populate
         * @return true if successfully loaded
         */
        bool loadPyTorchWeight(int layer, const std::string &weight_type,
                               NpyArray &out_array);

        /**
         * @brief Compare replicated (non-sliced) weight
         * @param pytorch_weight Full PyTorch weight
         * @param llaminar_weight Llaminar's weight (should match full PyTorch)
         * @param weight_name Descriptive name for logging
         * @return Verification result
         */
        WeightVerificationResult compareReplicatedWeight(
            const NpyArray &pytorch_weight,
            std::shared_ptr<TensorBase> llaminar_weight,
            const std::string &weight_name);

        /**
         * @brief Compare sliced weight (e.g., K/V weights sliced by heads)
         * @param pytorch_full_weight Full PyTorch weight
         * @param llaminar_local_weight Llaminar's local slice
         * @param weight_type "Q", "K", "V", etc. (for slice metadata)
         * @param weight_name Descriptive name for logging
         * @return Verification result
         */
        WeightVerificationResult compareSlicedWeight(
            const NpyArray &pytorch_full_weight,
            std::shared_ptr<TensorBase> llaminar_local_weight,
            const std::string &weight_type,
            const std::string &weight_name);

        /**
         * @brief Compute verification statistics
         * @param pytorch_data PyTorch reference data
         * @param llaminar_data Llaminar weight data
         * @param count Number of elements to compare
         * @param weight_name Descriptive name for logging
         * @return Verification result with statistics
         */
        WeightVerificationResult computeStatistics(
            const float *pytorch_data,
            const float *llaminar_data,
            size_t count,
            const std::string &weight_name);

        /**
         * @brief Extract local slice from full PyTorch weight
         * @param pytorch_full Full PyTorch weight [out_features, in_features]
         * @param weight_type Weight type ("Q", "K", "V", etc.)
         * @param out_local Output vector for local slice
         * @return true if extraction succeeded
         *
         * For row-sliced weights (K, V, Q), extracts rows corresponding to
         * this rank's head range.
         */
        bool extractLocalSlice(const NpyArray &pytorch_full,
                               const std::string &weight_type,
                               std::vector<float> &out_local);
    };

} // namespace llaminar
