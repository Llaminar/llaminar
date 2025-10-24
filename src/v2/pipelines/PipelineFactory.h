/**
 * @file PipelineFactory.h
 * @brief Factory for creating architecture-specific transformer pipelines
 * @author David Sanftenberg
 *
 * Provides automatic pipeline selection based on model architecture detected
 * from GGUF metadata. Pipelines register themselves at startup via static
 * initialization.
 *
 * Usage:
 *   auto pipeline = PipelineFactory::instance().create(
 *       "qwen2", model_path, mpi_ctx, device_idx);
 *
 * Adding new architectures:
 *   1. Implement a PipelineBase-derived class
 *   2. Create a static creator function
 *   3. Register via __attribute__((constructor))
 *
 * Example:
 *   static std::unique_ptr<PipelineBase> createLlama(...) {
 *       return std::make_unique<LlamaPipeline>(...);
 *   }
 *
 *   __attribute__((constructor)) static void init() {
 *       PipelineFactory::instance().registerCreator("llama", &createLlama);
 *   }
 */

#pragma once

#include "PipelineBase.h"
#include "../loaders/ModelContext.h"
#include <map>
#include <memory>
#include <string>
#include <functional>

namespace llaminar2
{

    /**
     * @brief Factory for creating architecture-specific pipelines
     *
     * Singleton pattern with static registration. Each pipeline implementation
     * registers itself at startup using __attribute__((constructor)).
     */
    class PipelineFactory
    {
    public:
        /**
         * @brief Pipeline creator function type
         *
         * @param model_ctx Model context with GGUF metadata and loader
         * @param mpi_ctx MPI context for distributed execution (nullptr = single node)
         * @param device_idx Default device for tensors (-1 = CPU, ≥0 = GPU device)
         * @return Unique pointer to created pipeline, or nullptr on error
         */
        using CreateFn = std::function<std::unique_ptr<PipelineBase>(
            std::shared_ptr<ModelContext> model_ctx,
            std::shared_ptr<MPIContext> mpi_ctx,
            int device_idx)>;

        /**
         * @brief Get singleton instance
         *
         * @return Reference to global PipelineFactory
         */
        static PipelineFactory &instance();

        /**
         * @brief Register a pipeline creator for an architecture
         *
         * @param architecture Architecture name (e.g., "qwen2", "llama")
         * @param creator Creator function
         *
         * @note If architecture already registered, logs warning and ignores
         * @note Thread-safe (registration happens during static initialization)
         */
        void registerCreator(const std::string &architecture, CreateFn creator);

        /**
         * @brief Create pipeline for specified architecture
         *
         * @param architecture Architecture name from GGUF metadata
         * @param model_ctx Model context with GGUF metadata and loader
         * @param mpi_ctx MPI context for distributed execution (nullptr = single node)
         * @param device_idx Default device for tensors (-1 = CPU, ≥0 = GPU device)
         * @return Unique pointer to created pipeline, or nullptr if architecture not supported
         *
         * @note Returns nullptr if architecture not registered
         */
        std::unique_ptr<PipelineBase> create(
            const std::string &architecture,
            std::shared_ptr<ModelContext> model_ctx,
            std::shared_ptr<MPIContext> mpi_ctx = nullptr,
            int device_idx = -1) const;

        /**
         * @brief Check if architecture is supported
         *
         * @param architecture Architecture name to check
         * @return true if registered, false otherwise
         */
        bool isSupported(const std::string &architecture) const;

        /**
         * @brief Get list of supported architectures
         *
         * @return Vector of registered architecture names
         */
        std::vector<std::string> supportedArchitectures() const;

        /**
         * @brief Get number of registered architectures
         *
         * @return Count of registered creators
         */
        size_t registeredCount() const { return creators_.size(); }

    private:
        PipelineFactory() = default;
        ~PipelineFactory() = default;

        // Delete copy/move to enforce singleton
        PipelineFactory(const PipelineFactory &) = delete;
        PipelineFactory &operator=(const PipelineFactory &) = delete;
        PipelineFactory(PipelineFactory &&) = delete;
        PipelineFactory &operator=(PipelineFactory &&) = delete;

        std::map<std::string, CreateFn> creators_;
    };

} // namespace llaminar2
