/**
 * @file MultiDomainOrchestrator.h
 * @brief Multi-domain tensor parallelism wrapper for GraphOrchestrator
 * @author David Sanftenberg
 * @date January 2026
 *
 * MultiDomainOrchestrator wraps GraphOrchestrator to add multi-domain tensor
 * parallelism support. It manages MultiDomainTPConfig and delegates inference
 * operations to the inner orchestrator.
 *
 * Design:
 * - Wrapper pattern: Delegates IInferenceRunner methods to inner GraphOrchestrator
 * - Domain management: Owns MultiDomainTPConfig and NodeTopology
 * - Stats tracking: Collects domain-level collective operation statistics
 * - Testability: createForTest() allows injecting mock orchestrators
 */

#pragma once

#include "IInferenceRunner.h"
#include "../config/TPDomain.h"
#include "../utils/NodeTopology.h"
#include "../backends/DeviceId.h"
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class GraphOrchestrator;
    class ICollectiveContext;
    class MPIContext;

    /**
     * @brief Configuration for MultiDomainOrchestrator initialization
     */
    struct MultiDomainOrchestratorConfig
    {
        /// Path to GGUF model file
        std::string model_path;

        /// Maximum sequence length for inference
        int max_seq_len = 2048;

        /// Batch size for inference
        int batch_size = 1;

        /// Pre-configured TP config (optional - auto-detected if nullptr)
        std::unique_ptr<MultiDomainTPConfig> tp_config = nullptr;

        /// Preferred devices for execution (empty = auto-detect)
        std::vector<DeviceId> preferred_devices;

        /// Enable CPU layers for heterogeneous execution
        bool enable_cpu_layers = false;

        /// Number of layers to place on CPU (if enable_cpu_layers is true)
        int num_cpu_layers = 0;
    };

    /**
     * @brief Multi-domain tensor parallelism wrapper for GraphOrchestrator
     *
     * This class wraps GraphOrchestrator to add multi-domain TP support:
     * - GPU_INTRA_RANK domain for attention head parallelism via PCIeBAR
     * - CPU_CROSS_RANK domain for FFN parallelism via MPI/UPI
     *
     * The orchestrator manages:
     * - MultiDomainTPConfig with per-domain MPI communicators
     * - NodeTopology for NUMA-aware domain creation
     * - Domain-level statistics for profiling
     *
     * Usage:
     * @code
     * MultiDomainOrchestratorConfig config;
     * config.model_path = "models/qwen2.5-7b.gguf";
     * auto orchestrator = MultiDomainOrchestrator::create(std::move(config));
     *
     * if (orchestrator && orchestrator->isReady()) {
     *     orchestrator->forward(tokens, seq_len);
     *     const float* output = orchestrator->logits();
     * }
     * @endcode
     */
    class MultiDomainOrchestrator : public IInferenceRunner
    {
    public:
        // =========================================================================
        // Factory Methods
        // =========================================================================

        /**
         * @brief Create orchestrator with full configuration
         *
         * Creates the inner GraphOrchestrator and configures multi-domain TP
         * based on system topology and MPI context.
         *
         * @param config Configuration options
         * @param mpi_ctx MPI context for distributed execution (nullptr for single-rank)
         * @return Unique pointer to initialized orchestrator, or nullptr on failure
         */
        static std::unique_ptr<MultiDomainOrchestrator> create(
            MultiDomainOrchestratorConfig config,
            MPIContext *mpi_ctx = nullptr);

        /**
         * @brief Create orchestrator for testing with injected dependencies
         *
         * Allows unit tests to inject a mock or stub IInferenceRunner
         * without loading actual models.
         *
         * @param inner Pre-configured inner orchestrator (IInferenceRunner implementation)
         * @param tp_config Pre-configured TP domains
         * @return Unique pointer to test orchestrator
         */
        static std::unique_ptr<MultiDomainOrchestrator> createForTest(
            std::unique_ptr<IInferenceRunner> inner,
            std::unique_ptr<MultiDomainTPConfig> tp_config);

        /// Destructor
        ~MultiDomainOrchestrator() override;

        // Non-copyable, non-movable (owns MPI communicators via tp_config_)
        MultiDomainOrchestrator(const MultiDomainOrchestrator &) = delete;
        MultiDomainOrchestrator &operator=(const MultiDomainOrchestrator &) = delete;
        MultiDomainOrchestrator(MultiDomainOrchestrator &&) = delete;
        MultiDomainOrchestrator &operator=(MultiDomainOrchestrator &&) = delete;

        // =========================================================================
        // IInferenceRunner Interface Implementation
        // =========================================================================

        /**
         * @brief Run forward pass (delegates to inner orchestrator)
         * @param tokens Token IDs
         * @param seq_len Sequence length
         * @return true if forward succeeded
         */
        bool forward(const int *tokens, int seq_len) override;

        /**
         * @brief Get logits from last forward pass
         * @return Pointer to logits [vocab_size], or nullptr if unavailable
         */
        const float *logits() const override;

        /**
         * @brief Get vocabulary size
         * @return Vocabulary size
         */
        int vocab_size() const override;

        /**
         * @brief Clear KV cache (reset for new sequence)
         */
        void clear_cache() override;

        /**
         * @brief Get current position in cache
         * @return Current position
         */
        int get_position() const override;

        /**
         * @brief Get execution path being used
         * @return Always returns ExecutionPath::GRAPH
         */
        ExecutionPath executionPath() const override;

        /**
         * @brief Get architecture name
         * @return Architecture string (e.g., "qwen2")
         */
        const char *architecture() const override;

        // =========================================================================
        // Domain Access
        // =========================================================================

        /**
         * @brief Get the multi-domain TP configuration
         * @return Pointer to TP config, or nullptr if not configured
         */
        const MultiDomainTPConfig *getTPConfig() const;

        /**
         * @brief Get the GPU tensor parallel domain
         * @return Pointer to GPU domain, or nullptr if no GPU TP
         */
        const TPDomain *getGPUDomain() const;

        /**
         * @brief Get the CPU tensor parallel domain
         * @return Pointer to CPU domain, or nullptr if no CPU TP
         */
        const TPDomain *getCPUDomain() const;

        /**
         * @brief Get the system topology
         * @return Pointer to topology, or nullptr if not detected
         */
        const NodeTopology *getTopology() const;

        /**
         * @brief Get the inner IInferenceRunner
         * @return Pointer to inner runner (never null after successful init)
         */
        IInferenceRunner *getInnerRunner() const;

        /**
         * @brief Get the inner GraphOrchestrator (production use only)
         * @return Pointer to inner orchestrator, or nullptr if inner is not GraphOrchestrator
         */
        GraphOrchestrator *getInnerOrchestrator() const;

        // =========================================================================
        // Statistics
        // =========================================================================

        /**
         * @brief Statistics about domain operations
         */
        struct DomainStats
        {
            int gpu_domain_collective_count = 0; ///< GPU domain collective ops
            int cpu_domain_collective_count = 0; ///< CPU domain collective ops
        };

        /**
         * @brief Get current domain statistics
         * @return Copy of current stats
         */
        DomainStats getDomainStats() const;

        /**
         * @brief Reset domain statistics to zero
         */
        void resetStats();

        // =========================================================================
        // Status
        // =========================================================================

        /**
         * @brief Check if orchestrator is ready for inference
         * @return true if initialized and inner orchestrator is ready
         */
        bool isReady() const;

        /**
         * @brief Get model information string
         * @return Human-readable model info from inner orchestrator
         */
        std::string getModelInfo() const;

    private:
        /// Private constructor - use factory methods
        MultiDomainOrchestrator();

        /**
         * @brief Initialize with configuration
         * @param config Configuration options
         * @param mpi_ctx MPI context (may be nullptr)
         * @return true if initialization succeeded
         */
        bool initialize(MultiDomainOrchestratorConfig config, MPIContext *mpi_ctx);

        /// Inner orchestrator (delegates inference operations)
        /// Stored as IInferenceRunner for testability (can inject mocks)
        std::unique_ptr<IInferenceRunner> inner_runner_;

        /// Multi-domain TP configuration (owns MPI communicators)
        std::unique_ptr<MultiDomainTPConfig> tp_config_;

        /// System topology for NUMA-aware domain creation
        std::unique_ptr<NodeTopology> topology_;

        /// Original configuration (for getModelInfo)
        MultiDomainOrchestratorConfig config_;

        /// Domain operation statistics (mutable for const accessors)
        mutable DomainStats stats_;

        /// Initialization status
        bool initialized_ = false;
    };

} // namespace llaminar2
