/**
 * @file CudaGemmNeuralNetwork.h
 * @brief ONNX Runtime neural network heuristic for CUDA GEMM config selection
 *
 * This module wraps an ONNX neural network trained to predict CUDA GEMM performance.
 * The NN provides better generalization to unseen shapes compared to linear regression.
 *
 * Architecture (IMPROVED Nov 2, 2025): 73 features → 128 → 64 → 32 → 16 → 1 (GFLOPS)
 * Training:  Test R² = 0.96+ (expected, was 0.9569)
 * Model Size: ~35 KB (.onnx file, was 26 KB)
 * Inference: ~120μs per prediction
 *
 * IMPROVEMENTS:
 * - 73 features (was 57): +16 hardware-aware + estimated profiler features
 * - Deeper network: 128→64→32→16 (was 64→32→16)
 * - RobustScaler: Better outlier handling (median/IQR vs mean/std)
 * - 1.5B training data: Includes canary test size!
 * - More iterations: 2000 max (was 500)
 * - PHASE 2: 8 estimated profiler features (bank conflicts, coalescing, etc.) - ZERO runtime cost!
 *
 * Expected Performance:
 * - Canary Top-20: 40-60% (was 0%)
 * - Better ranking: #100-200 (was #379)
 *
 * Feature Categories (73 total):
 * - 13 raw config/problem features
 * - 19 base engineered features
 * - 41 enhanced features:
 *   - 8 hardware-aware (warps, bank conflicts, coalescing)
 *   - 8 estimated profiler metrics (ZERO runtime cost!)
 *   - 25 other (batch-aware, alignment, efficiency, tile coverage, interactions)
 *
 * Usage:
 * ```cpp
 * auto &nn = CudaGemmNeuralNetwork::instance();
 * double predicted_gflops = nn.predict(config, m, n, k);
 * ```
 *
 * @author David Sanftenberg
 * @date November 2, 2025 (Updated with Phase 2 estimated profiler features)
 */

#pragma once

#ifdef HAVE_ONNX_RUNTIME

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <array>
#include <string>
#include <memory>
#include "CudaGemmConfig.h"

namespace llaminar2
{
    namespace cuda
    {

        /**
         * @brief ONNX-based neural network for CUDA GEMM performance prediction
         *
         * Singleton class that loads an ONNX model and provides predictions.
         */
        class CudaGemmNeuralNetwork
        {
        public:
            /**
             * @brief Get singleton instance
             */
            static CudaGemmNeuralNetwork &instance();

            /**
             * @brief Predict GFLOPS for a configuration
             *
             * @param config CUDA GEMM configuration
             * @param m Number of rows (batch size)
             * @param n Number of columns
             * @param k Inner dimension
             * @return Predicted GFLOPS (higher is better)
             */
            double predict(const CudaGemmConfig &config, int m, int n, int k);

            /**
             * @brief Check if neural network is initialized
             */
            bool isInitialized() const { return initialized_; }

            /**
             * @brief Get model path
             */
            std::string getModelPath() const { return model_path_; }

        private:
            CudaGemmNeuralNetwork();
            ~CudaGemmNeuralNetwork() = default;

            // Non-copyable, non-movable (singleton)
            CudaGemmNeuralNetwork(const CudaGemmNeuralNetwork &) = delete;
            CudaGemmNeuralNetwork &operator=(const CudaGemmNeuralNetwork &) = delete;

            /**
             * @brief Initialize ONNX Runtime and load model
             */
            void initialize();

            /**
             * @brief Extract features from config and problem size
             *
             * CRITICAL: Feature extraction MUST match Python training exactly!
             * See: src/v2/kernels/cuda/python/train_cuda_neural_network.py
             *
             * @return 73-element feature vector
             */
            std::array<float, 73> extractFeatures(const CudaGemmConfig &config, int m, int n, int k);

            /**
             * @brief Load scaler parameters (mean, scale) from .txt file
             * @return true if successful, false otherwise
             */
            bool loadScalerParameters();

            bool initialized_ = false;
            std::string model_path_;
            std::string scaler_path_;

            // ONNX Runtime components
            Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "CudaGemmNN"};
            Ort::SessionOptions session_options_;
            std::unique_ptr<Ort::Session> session_;

            // Feature scaler parameters (from RobustScaler in Python)
            std::array<float, 73> feature_mean_;  // Median of each feature
            std::array<float, 73> feature_scale_; // IQR (interquartile range) of each feature

            // Input/output tensor info (store as strings to keep memory alive)
            std::string input_name_storage_;
            std::string output_name_storage_;
            std::vector<const char *> input_names_;
            std::vector<const char *> output_names_;
        };

    } // namespace cuda
} // namespace llaminar2

#endif // HAVE_ONNX_RUNTIME
