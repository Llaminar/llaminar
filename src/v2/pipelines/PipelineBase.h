/**
 * @file PipelineBase.h
 * @brief Base class for transformer pipelines
 *
 * Provides common infrastructure for all model architectures:
 * - MPI context management
 * - Device placement
 * - Weight and activation management
 * - Common pipeline operations
 *
 * Derived classes implement architecture-specific details:
 * - Qwen2Pipeline, Qwen3Pipeline, Qwen3MoEPipeline, etc.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../utils/MPIContext.h"
#include "../backends/ComputeBackend.h"
#include "../loaders/ModelContext.h"
#include "../tensors/Tensors.h"
#include "../tensors/TensorKernels.h"
#include <vector>
#include <memory>
#include <string>

namespace llaminar2
{

    /**
     * @brief Base class for transformer pipelines
     *
     * Provides common infrastructure for model execution.
     * Derived classes implement architecture-specific logic.
     */
    class PipelineBase
    {
    public:
        /**
         * @brief Construct pipeline base
         *
         * @param model_ctx Model context with GGUF metadata and loader
         * @param mpi_ctx MPI context for distributed execution (nullptr = single node)
         * @param device_idx Default device for tensors (-1 = CPU, ≥0 = GPU device)
         */
        PipelineBase(std::shared_ptr<ModelContext> model_ctx,
                     std::shared_ptr<MPIContext> mpi_ctx = nullptr,
                     int device_idx = -1);

        virtual ~PipelineBase() = default;

        /**
         * @brief Forward pass (prefill or decode)
         *
         * @param tokens Token IDs [seq_len]
         * @param seq_len Number of tokens
         * @return true on success, false on error
         */
        virtual bool forward(const int *tokens, int seq_len) = 0;

        /**
         * @brief Get output logits (FP32)
         *
         * @return Logits tensor [seq_len, vocab_size], or nullptr if not available
         */
        virtual const float *logits() const = 0;

        /**
         * @brief Get model architecture name
         *
         * @return Architecture string (e.g., "qwen2", "qwen3", "qwen3-moe")
         */
        virtual const char *architecture() const = 0;

        /**
         * @brief Get model context
         *
         * @return Model context with metadata and loader
         */
        std::shared_ptr<ModelContext> model_context() const { return model_ctx_; }

        /**
         * @brief Get MPI context
         *
         * @return MPI context pointer, or nullptr if not using MPI
         */
        std::shared_ptr<MPIContext> mpi_context() const { return mpi_ctx_; }

        /**
         * @brief Get default device index
         *
         * @return Device index (-1 = CPU, ≥0 = GPU device)
         */
        int device_index() const { return device_idx_; }

    protected:
        // Context management
        std::shared_ptr<ModelContext> model_ctx_;
        std::shared_ptr<MPIContext> mpi_ctx_;
        int device_idx_; // Default device (-1 = CPU)

        // Model path for convenience (from model_ctx_)
        std::string model_path_;

        // Common model parameters (set by derived classes)
        int n_layers_ = 0;
        int d_model_ = 0;
        int vocab_size_ = 0;

        /**
         * @brief Load model weights from GGUF file
         *
         * To be implemented by derived classes based on architecture.
         *
         * @param model_path Path to GGUF model file
         * @return true on success, false on error
         */
        virtual bool load_weights(const std::string &model_path) = 0;

        /**
         * @brief Process a single transformer layer
         *
         * To be implemented by derived classes.
         *
         * @param layer_idx Layer index (0-indexed)
         * @param seq_len Sequence length
         * @return true on success, false on error
         */
        virtual bool transformer_layer(int layer_idx, int seq_len) = 0;
    };

} // namespace llaminar2
