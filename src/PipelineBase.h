#pragma once

#include "MpiKernelBase.h"
#include "tensors/TensorFactory.h"
#include "PipelineStages.h"
#include "AbstractPipeline.h"
#include <unordered_map>
#include <string>
#include <memory>
#include <vector>

namespace llaminar
{

    /**
     * @brief Base class for MPI-enabled pipelines that compose kernels
     *
     * PipelineBase provides infrastructure for building pipelines that orchestrate
     * multiple MPI kernels instead of implementing operations directly. This ensures
     * proper separation of concerns, reusability, and testability.
     *
     * Key Features:
     * - Kernel registry for composition-based architecture
     * - MPI-aware tensor creation utilities
     * - Synchronization and communication utilities
     * - Error handling and validation chains
     * - Performance monitoring and profiling hooks
     *
     * Design Philosophy:
     * - Pipelines orchestrate, kernels compute
     * - All operations through registered kernels
     * - Centralized MPI communication patterns
     * - NUMA-aware tensor allocation
     */
    class PipelineBase : public MPIOperatorBase
    {
    public:
        /**
         * @brief Construct PipelineBase (legacy - queries MPI context)
         */
        PipelineBase();

        /**
         * @brief Construct PipelineBase with explicit MPI context (preferred)
         * @param ctx MPI context with rank, size, and communicator
         */
        explicit PipelineBase(const MPIContext &ctx);

        /**
         * @brief Virtual destructor
         */
        virtual ~PipelineBase() = default;

        /**
         * @brief Pure virtual execute method - must be implemented by derived pipelines
         * @param inputs Input tensors for pipeline
         * @param outputs Output tensors from pipeline
         * @return true if execution successful, false otherwise
         */
        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs) override = 0;

        /**
         * @brief Get pipeline name for debugging and profiling
         * @return Pipeline name string
         */
        std::string getName() const { return "PipelineBase"; }

    protected:
        // === Kernel Management ===

        /**
         * @brief Register a kernel for use in pipeline composition
         * @param name Unique name for the kernel
         * @param kernel Kernel instance to register
         * @return true if registration successful, false if name already exists
         */
        bool registerOperator(const std::string &name, std::unique_ptr<MPIOperatorBase> kernel);

        /**
         * @brief Get registered kernel by name
         * @param name Kernel name to retrieve
         * @return Pointer to kernel, or nullptr if not found
         */
        MPIOperatorBase *getKernel(const std::string &name);

        /**
         * @brief Check if kernel is registered
         * @param name Kernel name to check
         * @return true if kernel exists, false otherwise
         */
        bool hasKernel(const std::string &name) const;

        /**
         * @brief Get list of all registered kernel names
         * @return Vector of kernel names
         */
        std::vector<std::string> getKernelNames() const;

        // === Tensor Factory Methods ===

        /**
         * @brief Create tensor suitable for MPI broadcasting
         * @param shape Tensor shape
         * @return Shared pointer to broadcast-ready tensor
         */
        std::shared_ptr<TensorBase> createBroadcastTensor(const std::vector<int> &shape);

        /**
         * @brief Create tensor for local computation on current rank
         * @param shape Tensor shape
         * @return Shared pointer to local tensor
         */
        std::shared_ptr<TensorBase> createLocalTensor(const std::vector<int> &shape, bool use_bf16 = false);

        /**
         * @brief Create tensor optimized for distributed computation
         * @param shape Tensor shape
         * @param operation_name Operation name for COSMA optimization
         * @return Shared pointer to distributed tensor
         */
        std::shared_ptr<TensorBase> createDistributedTensor(const std::vector<int> &shape,
                                                            const std::string &operation_name = "");

        /**
         * @brief Create tensor with automatic backend selection based on size and MPI context
         * @param shape Tensor shape
         * @return Shared pointer to optimally-configured tensor
         */
        std::shared_ptr<TensorBase> createAutoTensor(const std::vector<int> &shape);

        // === MPI Communication Utilities ===

        /**
         * @brief Synchronize all ranks before proceeding
         * @param operation_name Operation name for debugging
         * @param timeout_seconds Timeout in seconds (default: 30.0)
         */
        void syncAllRanks(const std::string &operation_name, double timeout_seconds = 30.0);

        /**
         * @brief Broadcast tensor from root rank to all other ranks
         * @param tensor Tensor to broadcast (modified in-place on non-root ranks)
         * @param root_rank Root rank to broadcast from (default: 0)
         * @return true if broadcast successful, false otherwise
         */
        bool broadcastTensor(std::shared_ptr<TensorBase> &tensor, int root_rank = 0);

        /**
         * @brief Gather tensors from all ranks to root rank
         * @param local_tensor Local tensor from this rank
         * @param gathered_tensors Output vector of tensors (only valid on root rank)
         * @param root_rank Root rank to gather to (default: 0)
         * @return true if gather successful, false otherwise
         */
        bool gatherTensors(const std::shared_ptr<TensorBase> &local_tensor,
                           std::vector<std::shared_ptr<TensorBase>> &gathered_tensors,
                           int root_rank = 0);

        /**
         * @brief All-gather tensors from all ranks to all ranks
         * @param local_tensor Local tensor from this rank
         * @param all_tensors Output vector of tensors from all ranks
         * @return true if all-gather successful, false otherwise
         */
        bool allGatherTensors(const std::shared_ptr<TensorBase> &local_tensor,
                              std::vector<std::shared_ptr<TensorBase>> &all_tensors);

        /**
         * @brief Reduce tensors across all ranks with specified operation
         * @param local_tensor Local tensor from this rank
         * @param result_tensor Output tensor (only valid on root rank)
         * @param op MPI reduction operation (MPI_SUM, MPI_MAX, etc.)
         * @param root_rank Root rank to reduce to (default: 0)
         * @return true if reduction successful, false otherwise
         */
        bool reduceTensors(const std::shared_ptr<TensorBase> &local_tensor,
                           std::shared_ptr<TensorBase> &result_tensor,
                           MPI_Op op = MPI_SUM, int root_rank = 0);

        /**
         * @brief All-reduce tensors across all ranks with specified operation
         * @param local_tensor Local tensor from this rank (modified in-place)
         * @param op MPI reduction operation (MPI_SUM, MPI_MAX, etc.)
         * @return true if all-reduction successful, false otherwise
         */
        bool allReduceTensors(std::shared_ptr<TensorBase> &local_tensor, MPI_Op op = MPI_SUM);

        // === Kernel Execution Utilities ===

        /**
         * @brief Execute kernel with error handling and validation
         * @param kernel_name Name of registered kernel to execute
         * @param inputs Input tensors for kernel
         * @param outputs Output tensors from kernel
         * @return true if execution successful, false otherwise
         */
        bool executeKernel(const std::string &kernel_name,
                           const std::vector<std::shared_ptr<TensorBase>> &inputs,
                           std::vector<std::shared_ptr<TensorBase>> &outputs);

        /**
         * @brief Chain multiple kernel executions with tensor flow
         * @param kernel_chain Vector of kernel names to execute in sequence
         * @param initial_inputs Initial input tensors
         * @param final_outputs Final output tensors
         * @param intermediate_tensors Optional intermediate tensor storage
         * @return true if entire chain successful, false otherwise
         */
        bool executeKernelChain(const std::vector<std::string> &kernel_chain,
                                const std::vector<std::shared_ptr<TensorBase>> &initial_inputs,
                                std::vector<std::shared_ptr<TensorBase>> &final_outputs,
                                std::vector<std::vector<std::shared_ptr<TensorBase>>> *intermediate_tensors = nullptr);

        // === Validation and Error Handling ===

        /**
         * @brief Validate tensor shapes for pipeline operations
         * @param tensors Tensors to validate
         * @param expected_shapes Expected shapes for each tensor
         * @param operation_name Operation name for error reporting
         * @return true if validation passes, false otherwise
         */
        bool validateTensorShapes(const std::vector<std::shared_ptr<TensorBase>> &tensors,
                                  const std::vector<std::vector<size_t>> &expected_shapes,
                                  const std::string &operation_name);

        /**
         * @brief Check tensors for NaN or infinite values
         * @param tensors Tensors to check
         * @param operation_name Operation name for error reporting
         * @return true if all tensors are valid, false if NaN/inf detected
         */
        bool validateTensorContents(const std::vector<std::shared_ptr<TensorBase>> &tensors,
                                    const std::string &operation_name);

        /**
         * @brief Log pipeline execution statistics
         * @param operation_name Operation name
         * @param execution_time_ms Execution time in milliseconds
         * @param tensor_shapes Input/output tensor shapes
         */
        void logExecutionStats(const std::string &operation_name,
                               double execution_time_ms,
                               const std::vector<std::vector<size_t>> &tensor_shapes);

        // === Parity Testing Helpers ===

        /**
         * @brief Capture tensor snapshot if parity testing is enabled
         *
         * Convenience method that handles:
         * - Checking if parity is enabled (early exit if not)
         * - Filtering to rank 0 only (avoids MPI duplication)
         * - Extracting tensor shape dimensions
         * - Delegating to captureStageSnapshot()
         *
         * @param stage Pipeline stage identifier
         * @param layer_index Layer index (-1 for non-layer stages)
         * @param tensor Tensor to capture
         *
         * Usage example:
         * @code
         *   auto embedding_output = embedKernel->execute(tokens);
         *   captureIfEnabled(PipelineStage::EMBEDDING, -1, embedding_output);
         * @endcode
         */
        void captureIfEnabled(
            PipelineStage stage,
            int layer_index,
            const std::shared_ptr<TensorBase> &tensor);

        /**
         * @brief Capture tensor snapshot with custom stage name
         *
         * Overload for custom or architecture-specific stages that don't
         * map to the standard PipelineStage enum.
         *
         * @param stage_name Custom stage name (for logging/comparison)
         * @param layer_index Layer index (-1 for non-layer stages)
         * @param tensor Tensor to capture
         */
        void captureIfEnabled(
            const std::string &stage_name,
            int layer_index,
            const std::shared_ptr<TensorBase> &tensor);

        /**
         * @brief Default implementation of parity snapshot capture
         *
         * Pipelines can override this to provide custom capture logic.
         * The default implementation delegates to parity::LlaminarSnapshotHook.
         *
         * @param stage Pipeline stage
         * @param layer_index Layer index
         * @param data Tensor data pointer
         * @param seq_len Sequence length
         * @param feature_dim Feature dimension
         * @param source Source identifier (default: "llaminar")
         *
         * @note This is virtual so derived classes that inherit from both
         *       PipelineBase and AbstractPipeline can provide one implementation
         */
        virtual void captureStageSnapshot(
            PipelineStage stage,
            int layer_index,
            const float *data,
            int seq_len,
            int feature_dim,
            const std::string &source = "llaminar");

        /**
         * @brief Check if parity testing is enabled
         *
         * Default implementation checks parity::LlaminarSnapshotHook::is_enabled().
         *
         * @return true if parity capture is enabled
         *
         * @note This is virtual so derived classes that inherit from both
         *       PipelineBase and AbstractPipeline can provide one implementation
         */
        virtual bool isParityEnabled() const;

        /**
         * @brief Set the snapshot source identifier for this pipeline
         * @param source Source identifier (e.g., "llaminar", "batch", "pytorch")
         */
        void setSnapshotSource(const std::string &source) { snapshot_source_ = source; }

        /**
         * @brief Get the snapshot source identifier
         * @return Current snapshot source
         */
        const std::string &getSnapshotSource() const { return snapshot_source_; }

    private:
        // Kernel registry
        std::unordered_map<std::string, std::unique_ptr<MPIOperatorBase>> kernels_;

        // Performance tracking
        std::unordered_map<std::string, double> kernel_execution_times_;
        std::unordered_map<std::string, size_t> kernel_execution_counts_;

    protected:
        // Snapshot configuration (protected so derived classes can access)
        std::string snapshot_source_ = "llaminar"; // Default source for snapshots
        int current_decode_step_ = -1;             // Active decode step index (-1 when not decoding or unspecified)
    public:
        void setDecodeStep(int step) { current_decode_step_ = step; }
        int currentDecodeStep() const { return current_decode_step_; }

    private:
        /**
         * @brief Initialize pipeline-specific configurations
         */
        void initializePipeline();

        /**
         * @brief Validate MPI environment and tensor factory
         * @return true if environment is valid, false otherwise
         */
        bool validateMPIEnvironment();

        /**
         * @brief Configure optimal tensor allocation strategies
         */
        void configureTensorAllocation();
    };

} // namespace llaminar