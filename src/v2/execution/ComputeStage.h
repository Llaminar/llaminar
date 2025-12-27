/**
 * @file ComputeStage.h
 * @brief Compute stage abstraction for device-agnostic kernel dispatch
 * @author David Sanftenberg
 * @date December 2025
 *
 * ComputeStage represents a single parallelizable operation that can execute
 * on any device (CPU, GPU). Stages are the unit of work for layer-level
 * parallelism and enable clean separation of serial setup from parallel compute.
 *
 * Key benefits:
 * 1. Device-agnostic interface - same API for CPU and GPU kernels
 * 2. Composable - build compute graphs from atomic operations
 * 3. Introspectable - stages know their FLOP counts, memory needs
 * 4. MoE-ready - expert FFN stages for parallel expert execution
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "DeviceContext.h"
#include "BufferRole.h"                 // For buffer requirements
#include "RuntimeConfig.h"              // For ActivationPrecision, FusedAttentionBackend
#include "../tensors/BlockStructures.h" // For Q8_1Block in StageDumpInfo
#include "../tensors/TensorKernels.h"   // For AttentionMode enum
#include "../utils/MPITopology.h"       // For WorkRange (tensor parallelism)

namespace llaminar2
{

    // Forward declarations (needed for StageDumpInfo before full TensorBase definition)
    class TensorBase;

    /**
     * @brief Detailed dump info for stage debugging
     *
     * Contains all input, output, and parameter buffers needed to fully
     * reproduce and debug a stage execution. Used by StageDumper for
     * first-class debugging support.
     */
    struct StageDumpInfo
    {
        // Input buffers (read during execute)
        struct InputBuffer
        {
            const char *name = nullptr; ///< Buffer name (e.g., "A", "input", "gate")
            const void *data = nullptr; ///< Pointer to data
            size_t rows = 0;
            size_t cols = 0;
            const char *dtype = "FP32";          ///< "FP32", "Q8_1", "IQ4_NL", etc.
            size_t element_size = sizeof(float); ///< Bytes per element (or block)
        };

        // Output buffers (written during execute)
        struct OutputBuffer
        {
            const char *name = nullptr; ///< Buffer name (e.g., "C", "output")
            const void *data = nullptr; ///< Pointer to data
            size_t rows = 0;
            size_t cols = 0;
            const char *dtype = "FP32";
            size_t element_size = sizeof(float);
        };

        // Weight/parameter buffers (read-only during execute)
        struct WeightBuffer
        {
            const char *name = nullptr;         ///< Weight name (e.g., "B", "gamma", "wq")
            const TensorBase *tensor = nullptr; ///< Original quantized tensor
            const void *raw_data = nullptr;     ///< Raw quantized data
            size_t raw_size = 0;                ///< Raw data size in bytes
            size_t rows = 0;
            size_t cols = 0;
            const char *dtype = nullptr; ///< Quantization type
        };

        // Scalar parameters
        struct ScalarParam
        {
            const char *name = nullptr;
            double value = 0;
            const char *dtype = "float"; ///< "int", "float", "bool"
        };

        std::vector<InputBuffer> inputs;
        std::vector<OutputBuffer> outputs;
        std::vector<WeightBuffer> weights;
        std::vector<ScalarParam> scalars;

        // Convenience methods for building dump info
        StageDumpInfo &addInput(const char *name, const float *data, size_t rows, size_t cols)
        {
            inputs.push_back({name, data, rows, cols, "FP32", sizeof(float)});
            return *this;
        }

        /**
         * @brief Add input tensor (implementation in ComputeStage.cpp to avoid header deps)
         */
        StageDumpInfo &addInput(const char *name, const TensorBase *tensor, size_t rows, size_t cols);

        StageDumpInfo &addInputQ8_1(const char *name, const void *data, size_t rows, size_t cols)
        {
            // For Q8_1, cols should be in blocks (cols/32 blocks per row)
            inputs.push_back({name, data, rows, cols, "Q8_1", sizeof(Q8_1Block)});
            return *this;
        }

        StageDumpInfo &addOutput(const char *name, const float *data, size_t rows, size_t cols)
        {
            outputs.push_back({name, data, rows, cols, "FP32", sizeof(float)});
            return *this;
        }

        /**
         * @brief Add output tensor (implementation in ComputeStage.cpp to avoid header deps)
         */
        StageDumpInfo &addOutput(const char *name, const TensorBase *tensor, size_t rows, size_t cols);

        /**
         * @brief Add weight tensor with explicit dimensions (avoids header dependency)
         */
        StageDumpInfo &addWeight(const char *name, const TensorBase *tensor,
                                 size_t rows, size_t cols, const char *dtype)
        {
            weights.push_back({name, tensor, nullptr, 0, rows, cols, dtype});
            return *this;
        }

        /**
         * @brief Add weight tensor (implementation in ComputeStage.cpp to avoid header deps)
         */
        StageDumpInfo &addWeight(const char *name, const TensorBase *tensor);

        StageDumpInfo &addScalar(const char *name, double value, const char *dtype = "float")
        {
            scalars.push_back({name, value, dtype});
            return *this;
        }

        StageDumpInfo &addScalarInt(const char *name, int value)
        {
            scalars.push_back({name, static_cast<double>(value), "int"});
            return *this;
        }

        StageDumpInfo &addScalarBool(const char *name, bool value)
        {
            scalars.push_back({name, value ? 1.0 : 0.0, "bool"});
            return *this;
        }
    };

    // Forward declarations (TensorBase already declared above)
    class IKVCache;
    class IUnifiedKVCache;
    class MPIContext;
    class FusedAttentionWoKernel; // Forward declaration for FusedAttentionWoStage

    /**
     * @brief Types of compute operations
     */
    enum class ComputeStageType
    {
        // Matrix operations
        GEMM,               ///< General matrix multiplication
        GEMM_BIAS,          ///< GEMM with bias addition
        GEMM_FUSED_QKV,     ///< Fused Q/K/V projection
        GEMM_FUSED_GATE_UP, ///< Fused Gate/Up projection (FFN)

        // Normalization
        RMS_NORM,   ///< RMS normalization
        LAYER_NORM, ///< Layer normalization (future)

        // Activations
        SWIGLU, ///< SwiGLU activation (FFN)
        GELU,   ///< GELU activation (future)
        SILU,   ///< SiLU activation

        // Attention
        ROPE,               ///< Rotary position encoding
        ATTENTION,          ///< Full attention (Q*K^T, softmax, *V)
        ATTENTION_QK,       ///< Q*K^T only
        ATTENTION_SOFTMAX,  ///< Softmax only
        ATTENTION_V,        ///< Softmax @ V only
        FUSED_ATTENTION_WO, ///< Fused attention + Wo projection (FA2 JIT)

        // Element-wise
        ADD_RESIDUAL, ///< Element-wise addition (residual connection)
        SCALE,        ///< Element-wise scaling

        // MoE specific
        MOE_ROUTER,     ///< Expert routing (softmax + top-k)
        MOE_EXPERT_FFN, ///< Single expert FFN execution
        MOE_COMBINE,    ///< Combine expert outputs with weights

        // Collective
        ALLREDUCE, ///< MPI allreduce (sum)
        ALLGATHER, ///< MPI allgather

        // Utility
        COPY,       ///< Memory copy
        QUANTIZE,   ///< Quantization (FP32 → Q8_1, etc.)
        DEQUANTIZE, ///< Dequantization

        // Model-level operations (ModelExecutor)
        EMBEDDING,  ///< Token embedding lookup
        LM_HEAD,    ///< Language model head projection
        FINAL_NORM, ///< Final layer normalization
    };

    /**
     * @brief Convert stage type to string for logging
     */
    const char *computeStageTypeName(ComputeStageType type);

    /**
     * @brief Base class for all compute stages
     *
     * Derived classes implement device-specific kernels while maintaining
     * a common interface for orchestration.
     */
    class IComputeStage
    {
    public:
        virtual ~IComputeStage() = default;

        // =========================================================================
        // Execution
        // =========================================================================

        /**
         * @brief Execute this stage on the given device context
         *
         * The stage must be compatible with the device type (CPU stages on CPU, etc.)
         * GPU stages may enqueue work asynchronously - call ctx->synchronize() if
         * you need completion.
         *
         * @param ctx Device context to execute on
         * @return true on success, false on error
         */
        virtual bool execute(IDeviceContext *ctx) = 0;

        // =========================================================================
        // Introspection
        // =========================================================================

        /**
         * @brief Get the operation type
         */
        virtual ComputeStageType type() const = 0;

        /**
         * @brief Human-readable name (for profiling/logging)
         */
        virtual std::string name() const
        {
            return computeStageTypeName(type());
        }

        /**
         * @brief Estimated floating-point operations
         *
         * Used for load balancing and performance modeling.
         * Returns 0 if not applicable (e.g., memory ops).
         */
        virtual size_t estimatedFlops() const { return 0; }

        /**
         * @brief Estimated memory traffic in bytes
         *
         * Includes reads and writes. Used for bandwidth estimation.
         */
        virtual size_t estimatedMemoryBytes() const { return 0; }

        /**
         * @brief Does this stage require MPI synchronization after?
         *
         * True for stages like row-parallel GEMM that need allreduce.
         */
        virtual bool requiresAllreduce() const { return false; }

        /**
         * @brief Can this stage execute on the given backend?
         */
        virtual bool supportsBackend(ComputeBackendType backend) const = 0;

        /**
         * @brief Get detailed dump info for debugging/snapshotting
         *
         * Returns comprehensive information about all inputs, outputs, weights,
         * and parameters needed to reproduce this stage execution.
         * Used by StageDumper for first-class debugging support.
         *
         * @return Dump info with all buffers and parameters
         */
        virtual StageDumpInfo getDumpInfo() const
        {
            return StageDumpInfo{}; // Default: empty info
        }

        // =========================================================================
        // Buffer Management (Phase 1: GraphExecutor buffer management)
        // =========================================================================

        /**
         * @brief Get buffer requirements for this stage
         *
         * Returns descriptors for all buffers this stage needs:
         * - INPUT: Read-only data consumed by the stage
         * - OUTPUT: Data produced by the stage
         * - INOUT: Read-modify-write buffers (e.g., residual accumulation)
         * - SCRATCH: Temporary workspace (eligible for aliasing)
         * - WEIGHT: Read-only model parameters
         *
         * GraphBufferManager uses this for:
         * - Pre-allocating all buffers before graph execution
         * - Liveness analysis for SCRATCH buffer reuse
         * - Memory budget estimation
         *
         * CRITICAL: Stages MUST NOT allocate buffers internally in execute().
         * All buffers should come from GraphExecutor's managed pool.
         *
         * Default implementation returns empty requirements (for backward compat).
         * Stages should override to declare their actual requirements.
         *
         * @return Buffer requirements for this stage
         * @see BufferRole for role semantics
         * @see GraphBufferManager for buffer allocation
         */
        virtual StageBufferRequirements getBufferRequirements() const
        {
            return StageBufferRequirements{}; // Default: no requirements declared
        }

        // =========================================================================
        // Stage Output Manifest (Phase 3: Buffer Contract Validation)
        // =========================================================================

        /**
         * @brief Get declared outputs this stage will produce
         *
         * Returns descriptors for outputs that this stage commits to producing.
         * Unlike getBufferRequirements() which describes all buffers needed,
         * this method focuses on OUTPUT buffers with producer contracts.
         *
         * Key difference from getBufferRequirements():
         * - getBufferRequirements(): "What buffers do I need to run?"
         * - getDeclaredOutputs(): "What buffers do I promise to populate?"
         *
         * Use cases:
         * - GraphValidator checks that declared outputs actually get populated
         * - Debug builds verify buffers are non-zero after stage executes
         * - Documentation of stage contracts for downstream consumers
         *
         * Stages that produce outputs conditionally (e.g., V_dequant in hybrid mode)
         * should only include those outputs when they will actually be produced.
         *
         * @return Vector of output buffer descriptors with producer contracts
         * @see BufferDescriptor::withProducer() for producer declaration
         * @see BufferDescriptor::validatePopulated() for zero-detection
         */
        virtual std::vector<BufferDescriptor> getDeclaredOutputs() const
        {
            return {}; // Default: no declared outputs (derive from getBufferRequirements)
        }

        // =========================================================================
        // Dynamic Parameter Update (Phase 10: Graph Caching Optimization)
        // =========================================================================

        /**
         * @brief Update dynamic parameters for cached graph reuse
         *
         * When graphs are cached and reused across executions, some parameters
         * change between runs (e.g., position offset for RoPE, seq_len).
         * This method allows efficient updates without rebuilding the graph.
         *
         * Stages that support dynamic params should override this.
         * Default implementation does nothing (parameters are static).
         *
         * @param pos_offset New position offset (for RoPE stages)
         * @param seq_len New sequence length (for dimension-dependent stages)
         */
        virtual void updateDynamicParams(int pos_offset, int seq_len)
        {
            (void)pos_offset;
            (void)seq_len;
            // Default: no dynamic params to update
        }

    protected:
        // =========================================================================
        // Stage Tracing Infrastructure (Task 3: Debugging)
        // =========================================================================

        /**
         * @brief Check if tracing is enabled for this stage.
         *
         * Considers both global trace_stages flag and per-stage filter.
         */
        bool shouldTrace() const;

        /**
         * @brief Trace input tensor values (only if tracing enabled).
         *
         * @param name Tensor name (e.g., "A", "hidden_states")
         * @param tensor Tensor to trace
         */
        void traceInput(const std::string &name, const TensorBase *tensor) const;

        /**
         * @brief Trace output tensor values (only if tracing enabled).
         *
         * @param name Tensor name (e.g., "C", "output")
         * @param tensor Tensor to trace
         */
        void traceOutput(const std::string &name, const TensorBase *tensor) const;

        /**
         * @brief Trace intermediate float array values.
         *
         * @param name Array name
         * @param data Pointer to float data
         * @param count Total element count
         */
        void traceIntermediate(const std::string &name, const float *data, size_t count) const;

        /**
         * @brief Compute checksum for tensor data (for divergence detection).
         *
         * @param data Float data pointer
         * @param count Element count
         * @return Simple float sum (not cryptographic, just for comparison)
         */
        static float computeChecksum(const float *data, size_t count);

        /**
         * @brief Format float array for logging (first N elements).
         */
        static std::string formatFloatArray(const float *data, size_t count);

        // =========================================================================
        // Shape Validation Infrastructure (Task 7: Debugging)
        // =========================================================================

        /**
         * @brief Contract for a single tensor's expected shape.
         *
         * Used by validateShapes() to verify tensor dimensions match expectations.
         */
        struct TensorShapeContract
        {
            std::string name;             ///< Tensor name for error messages
            const TensorBase *tensor;     ///< Tensor to validate
            std::vector<size_t> expected; ///< Expected shape dimensions
            bool allow_broadcast = false; ///< If true, trailing dims of 1 are OK
        };

        /**
         * @brief Validate tensor shapes match expected contracts.
         *
         * @param contracts List of tensor/shape contracts to validate
         * @throws std::runtime_error if any shape mismatches
         *
         * Example usage:
         * @code
         *   validateShapes({
         *       {"input", input_, {batch_size_, d_model_}},
         *       {"wq", wq_, {n_heads_ * head_dim_, d_model_}},
         *   });
         * @endcode
         */
        void validateShapes(std::initializer_list<TensorShapeContract> contracts) const;

        /**
         * @brief Validate a single tensor's shape.
         *
         * @param tensor_name Name for error messages
         * @param tensor Tensor to validate
         * @param expected Expected shape
         */
        void validateShape(const std::string &tensor_name, const TensorBase *tensor,
                           const std::vector<size_t> &expected) const;

        /**
         * @brief Validate two tensors have compatible shapes for matmul.
         *
         * For A @ B where A is (m, k) and B is (k, n), validates k dimensions match.
         *
         * @param a_name Name of first tensor (for error messages)
         * @param a First tensor (left operand)
         * @param b_name Name of second tensor (for error messages)
         * @param b Second tensor (right operand)
         * @throws std::runtime_error if shapes incompatible for matmul
         */
        void validateMatmulShapes(const std::string &a_name, const TensorBase *a,
                                  const std::string &b_name, const TensorBase *b) const;

    private:
        /**
         * @brief Check if actual shape matches expected, with optional broadcast.
         */
        static bool shapesMatch(const std::vector<size_t> &actual,
                                const std::vector<size_t> &expected,
                                bool allow_broadcast);

        /**
         * @brief Format shape vector as string for error messages.
         */
        static std::string shapeToString(const std::vector<size_t> &shape);
    };

    // =============================================================================
    // Concrete Stage Implementations (CPU)
    // =============================================================================

    /**
     * @brief GEMM stage: C = alpha * A * B + beta * C
     *
     * Takes FP32 activations and quantized/FP32 weights, handles quantization internally.
     * For multi-projection patterns (Q/K/V or gate/up), use FusedQKVGEMMStage or
     * FusedGateUpGEMMStage which efficiently share quantization across projections.
     *
     * **Tensor Parallelism Support (Phase 2)**:
     * When `output_range` is set, executes a row-sliced GEMM for tensor parallelism:
     * - Uses `KernelFactory::getOrCreateGemmSliced()` to create sliced kernel
     * - Only computes output rows in [output_range.start, output_range.end)
     * - Caller is responsible for MPI AllReduce after execution if needed
     */
    class GEMMStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Type-safe tensor pointers (required)
            const TensorBase *A = nullptr; ///< Input activation tensor [m, k]
            const TensorBase *B = nullptr; ///< Weight tensor [k, n] (may be quantized)
            TensorBase *C = nullptr;       ///< Output tensor [m, n]
            int m = 0, n = 0, k = 0;       ///< Matrix dimensions
            float alpha = 1.0f;            ///< Scale factor for A*B
            float beta = 0.0f;             ///< Scale factor for existing C
            bool transpose_B = false;      ///< Whether B is transposed (n × k)

            // =================================================================
            // Bias Configuration (Enhanced for Tensor Parallelism)
            // =================================================================

            /**
             * @brief Raw bias pointer (legacy interface)
             *
             * @deprecated Use bias_tensor instead for TensorSlice/TP compatibility.
             */
            const float *bias = nullptr;

            /**
             * @brief Bias tensor (preferred interface)
             *
             * Using TensorBase* allows correct handling of TensorSlice from
             * tensor parallelism weight sharding. The data is extracted via
             * is_fp32_backed() + data() interface.
             */
            const TensorBase *bias_tensor = nullptr;

            /**
             * @brief Whether bias is required for this operation
             *
             * If true, validate() will fail if neither bias nor bias_tensor
             * provides valid bias data. This enables fail-fast debugging.
             */
            bool bias_required = false;

            /**
             * @brief Get bias data pointer from either source
             *
             * Prefers bias_tensor (via is_fp32_backed() interface) over raw bias.
             * Returns nullptr if no bias provided.
             *
             * @return const float* Bias data [n], or nullptr if no bias
             */
            const float *getBiasData() const;

            /**
             * @brief Validate params and throw on configuration error
             *
             * Checks:
             * - Required tensors (A, B, C) are non-null
             * - Dimensions (m, n, k) are positive
             * - If bias_required, bias data is available
             *
             * @throws std::runtime_error on validation failure
             */
            void validate(const std::string &stage_name = "GEMMStage") const;

            // SwiGLU fusion (extended fields)
            const TensorBase *gate_input = nullptr; ///< SwiGLU gate input tensor [m, n] (nullptr if not SwiGLU)
            bool do_swiglu = false;                 ///< Whether to apply SwiGLU fusion

            // MPI context for tensor-parallel execution (optional)
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;

            // =================================================================
            // Tensor Parallelism Parameters (Phase 2)
            // =================================================================

            /**
             * @brief Output row range for row-parallel GEMM
             *
             * When set (non-empty), uses sliced kernel to compute only rows
             * [output_range.start, output_range.end) of the output matrix.
             * The weight tensor B is sliced accordingly.
             *
             * For row-parallel GEMM (e.g., attention output projection):
             * - Each rank computes a slice of the output
             * - Results are combined via AllReduce
             *
             * When empty (default), computes full GEMM.
             */
            WorkRange output_range;

            /**
             * @brief Whether this GEMM requires AllReduce after execution
             *
             * Set to true for row-parallel GEMM where output is distributed.
             * The AllReduce is NOT performed by this stage - caller must handle it.
             * This flag is informational for graph analysis.
             */
            bool needs_allreduce = false;
        };

        explicit GEMMStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::GEMM; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;
        bool requiresAllreduce() const override { return params_.needs_allreduce; }

        /// Get params for testing/introspection
        const Params &getParams() const { return params_; }

    private:
        Params params_;
    };

    /**
     * @brief Fused Q/K/V projection stage
     *
     * Efficiently computes multiple linear projections (Q, K, V) from a shared
     * input by quantizing the input once and reusing the Q8_1 buffer for all
     * projections. This avoids redundant quantization and improves cache locality.
     *
     * This stage delegates to QuantisedGemmKernel::multiply_fused_multi(), which
     * handles the quantization and multi-projection execution internally.
     */
    class FusedQKVGEMMStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Type-safe tensor pointers (required)
            const TensorBase *input = nullptr; ///< Input activation tensor [m, k]
            int m = 0;                         ///< Batch size * seq_len
            int k = 0;                         ///< Input dimension (d_model)

            // Q projection
            const TensorBase *wq = nullptr;
            TensorBase *output_q = nullptr;
            int n_q = 0;
            const float *bias_q = nullptr; ///< Optional bias for Q [n_q]

            // K projection
            const TensorBase *wk = nullptr;
            TensorBase *output_k = nullptr;
            int n_k = 0;
            const float *bias_k = nullptr; ///< Optional bias for K [n_k]

            // V projection
            const TensorBase *wv = nullptr;
            TensorBase *output_v = nullptr;
            int n_v = 0;
            const float *bias_v = nullptr; ///< Optional bias for V [n_v]
        };

        explicit FusedQKVGEMMStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::GEMM_FUSED_QKV; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

    /**
     * @brief Fused Gate/Up projection stage for FFN
     *
     * Efficiently computes both gate and up projections from a shared input
     * by quantizing the input once and reusing the Q8_1 buffer for both
     * projections. This avoids redundant quantization and improves cache locality.
     *
     * Pattern: input → [gate_proj, up_proj] → SwiGLU → down_proj
     *
     * This stage delegates to QuantisedGemmKernel::multiply_fused_multi(), which
     * handles the quantization and multi-projection execution internally.
     */
    class FusedGateUpGEMMStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Type-safe tensor pointers (required)
            const TensorBase *input = nullptr; ///< Input activation tensor [m, k]
            int m = 0;                         ///< Batch size * seq_len
            int k = 0;                         ///< Input dimension (d_model)

            // Gate projection
            const TensorBase *w_gate = nullptr;
            TensorBase *output_gate = nullptr;
            int n_gate = 0;                   ///< Intermediate dimension (d_ff)
            const float *bias_gate = nullptr; ///< Optional bias for gate [n_gate]

            // Up projection
            const TensorBase *w_up = nullptr;
            TensorBase *output_up = nullptr;
            int n_up = 0;                   ///< Intermediate dimension (d_ff, same as n_gate)
            const float *bias_up = nullptr; ///< Optional bias for up [n_up]

            // Optional MPI context for distributed execution (needed for Q8_1 path)
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;
        };

        explicit FusedGateUpGEMMStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::GEMM_FUSED_GATE_UP; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

    /**
     * @brief RMS normalization stage
     *
     * Type-safe implementation using TensorBase* instead of void*.
     * The tensor's native_type() determines precision dispatch.
     * Uses IActivationTensor::applyRMSNorm() for polymorphic device dispatch.
     *
     * Device-agnostic: IActivationTensor handles CPU/GPU dispatch internally.
     */
    class RMSNormStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Type-safe tensor pointers (required)
            TensorBase *input = nullptr;       ///< Input activation tensor (IActivationTensor*)
            TensorBase *output = nullptr;      ///< Output tensor (can be same as input for in-place)
            const TensorBase *gamma = nullptr; ///< Gamma weights tensor

            float eps = 1e-6f; ///< Epsilon for numerical stability
            // Optional MPI context for distributed execution
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;

            // Explicit seq_len override for pre-allocated buffers
            // If 0, derives from input tensor dimensions
            // CRITICAL: Must be set during decode when using pre-allocated buffers
            int seq_len = 0;
        };

        explicit RMSNormStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::RMS_NORM; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

        /// Get params for testing
        const Params &getParams() const { return params_; }

    private:
        Params params_;
    };

    /**
     * @brief Rotary position encoding stage
     *
     * Type-safe implementation using TensorBase* instead of void*.
     * The tensor's native_type() determines precision dispatch.
     * Uses IActivationTensor::applyRoPE() for polymorphic device dispatch.
     */
    class RoPEStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Type-safe tensor pointers (required)
            TensorBase *Q = nullptr; ///< Query tensor (IActivationTensor*, modified in-place)
            TensorBase *K = nullptr; ///< Key tensor (IActivationTensor*, modified in-place, optional)

            // Hybrid mode output buffers (optional - when set, output goes here instead of in-place)
            TensorBase *Q_out = nullptr; ///< FP32 output for Q after RoPE (Hybrid mode)
            TensorBase *K_out = nullptr; ///< FP32 output for K after RoPE (Hybrid mode)

            // Configuration
            int n_heads = 0;             ///< Number of query heads
            int n_kv_heads = 0;          ///< Number of KV heads (for GQA)
            int head_dim = 0;            ///< Dimension per head
            int pos_offset = 0;          ///< Position offset (for KV cache)
            float theta_base = 10000.0f; ///< RoPE theta base
            int seq_len = 0;             ///< Explicit sequence length (for pre-allocated buffers)
            // Optional MPI context for distributed execution
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;
        };

        explicit RoPEStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ROPE; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

        /// Update position offset for cached graph reuse
        void updateDynamicParams(int pos_offset, int seq_len) override
        {
            params_.pos_offset = pos_offset;
            params_.seq_len = seq_len;
        }

        /// Get params for testing
        const Params &getParams() const { return params_; }

    private:
        Params params_;
    };

    // Forward declarations for attention infrastructure
    class IUnifiedKVCache;
    class MPIContext;

    /**
     * @brief Production attention stage with KV cache and MPI support
     *
     * This is the full-featured attention implementation that integrates with:
     * - **KV Cache**: Automatic append during prefill, retrieval during decode
     * - **MPI Parallelism**: Tensor-parallel attention via KernelFactory + compute_tensor()
     * - **Batched Execution**: Multiple sequences with padding masks
     * - **Asymmetric Lengths**: Different Q and KV lengths (prefill vs decode)
     * - **GQA/MHA/MQA**: All attention variants supported
     *
     * Execution Modes:
     * 1. **Prefill**: Process all tokens, append to KV cache, full attention
     * 2. **Decode**: Single token query, use cached K/V, efficient O(n) per token
     * 3. **Batched**: Multiple sequences, padding-aware masking
     *
     * Usage in LayerExecutor:
     * @code
     * AttentionWithKVCacheStage::Params params{
     *     .Q = buffers.Q.get(),
     *     .K = buffers.K.get(),
     *     .V = buffers.V.get(),
     *     .output = buffers.attn_output.get(),
     *     .kv_cache = kv_cache_.get(),
     *     .layer_idx = layer_idx,
     *     // ... configuration
     * };
     * graph.addNode("attention", ComputeStageFactory::createAttentionWithKVCache(params), device);
     * @endcode
     *
     * MoE Compatibility:
     * For MoE models, attention is typically shared across experts (not expert-specific).
     * The KV cache is shared, and this stage can be used directly in MoE pipelines.
     * Expert parallelism affects FFN routing, not attention computation.
     *
     * @see MpiAttentionOrchestrator for the underlying attention computation
     * @see KVCacheAppendStage for explicit cache operations (advanced pipelining)
     */
    class AttentionWithKVCacheStage : public IComputeStage
    {
    public:
        /**
         * @brief Legacy mode enum for backward compatibility
         * @deprecated Use AttentionMode from TensorKernels.h for new code
         */
        enum class Mode
        {
            AUTO,    ///< Detect from seq_len and KV cache state
            PREFILL, ///< Process multiple tokens, populate cache
            DECODE,  ///< Single token, use cached K/V
            BATCHED  ///< Multiple sequences with different lengths
        };

        struct Params
        {
            // Input/output tensors (already projected)
            TensorBase *Q = nullptr;      ///< Query [batch_size * seq_len, n_heads * head_dim]
            TensorBase *K = nullptr;      ///< Key [batch_size * seq_len, n_kv_heads * head_dim]
            TensorBase *V = nullptr;      ///< Value [batch_size * seq_len, n_kv_heads * head_dim]
            TensorBase *output = nullptr; ///< Output [batch_size * seq_len, n_heads * head_dim]

            // KV cache integration
            IUnifiedKVCache *kv_cache = nullptr; ///< KV cache (nullptr = no caching)
            int layer_idx = 0;                   ///< Layer index for cache access

            // Execution mode (legacy - prefer attention_mode)
            Mode mode = Mode::AUTO;

            // Attention configuration
            int batch_size = 1;
            int seq_len = 0;      ///< Query sequence length
            int n_heads = 0;      ///< Number of query heads
            int n_kv_heads = 0;   ///< Number of KV heads (GQA: n_kv_heads < n_heads)
            int head_dim = 0;     ///< Dimension per head
            bool causal = true;   ///< Apply causal mask
            int window_size = -1; ///< Sliding window attention (-1 = full)

            // MPI configuration (for tensor-parallel execution)
            std::shared_ptr<MPIContext> mpi_ctx;
            int mpi_strategy = 0; ///< MPIStrategy cast to int (avoid header dep)

            // Workspace buffers (pre-allocated by pipeline for zero hot-path allocation)
            // If nullptr, stage will allocate internally (slower)
            TensorBase *workspace_scores = nullptr;  ///< [n_heads * max_seq, max_seq]
            TensorBase *workspace_context = nullptr; ///< [max_seq, head_dim]
            TensorBase *workspace_mask = nullptr;    ///< [max_seq, max_seq]

            // Per-sequence lengths for batched attention (nullptr = all same length)
            const std::vector<int> *sequence_lengths = nullptr;

            // Position offset for decode mode (current position in sequence)
            int position_offset = 0;

            // Device placement (-1 = CPU, >=0 = GPU device index)
            int device_idx = -1;

            // =================================================================
            // Tensor Parallelism Parameters (Phase 2.4)
            // =================================================================

            /**
             * @brief First query head for this rank (0-indexed)
             *
             * Default 0 means start from head 0 (no head-parallel slicing).
             * With TP: each rank gets a contiguous range of heads.
             *
             * Example: 14 heads, 2 ranks
             *   Rank 0: head_start=0, local_n_heads=7
             *   Rank 1: head_start=7, local_n_heads=7
             */
            int head_start = 0;

            /**
             * @brief Number of query heads for this rank
             *
             * Default -1 means use full n_heads (no slicing).
             * With TP: set to n_heads / world_size (approximately).
             */
            int local_n_heads = -1;

            /**
             * @brief Number of KV heads for this rank
             *
             * Default -1 means use full n_kv_heads (no slicing).
             * For GQA: may equal local_n_heads / gqa_ratio.
             *
             * Note: If n_kv_heads < world_size, some ranks may have 0 KV heads
             * and must skip attention entirely or participate in collective only.
             */
            int local_n_kv_heads = -1;
        };

        explicit AttentionWithKVCacheStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ATTENTION; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;

        /**
         * @brief Get the actual execution mode that will be used
         *
         * If mode is AUTO, this detects based on seq_len and cache state.
         */
        Mode effectiveMode() const;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;

        /// Prefill: Process all tokens, populate KV cache
        bool executePrefill(IDeviceContext *ctx);

        /// Decode: Single token against cached K/V
        bool executeDecode(IDeviceContext *ctx);

        /// Batched: Multiple sequences with padding
        bool executeBatched(IDeviceContext *ctx);
    };

    /**
     * @brief Explicit KV cache append stage
     *
     * Separates cache operations from attention computation, enabling:
     * - Pipelined execution: Append on one device while attending on another
     * - Explicit control: Manual cache management for advanced use cases
     * - Cross-device caches: Cache on GPU while computing on CPU
     *
     * For most use cases, prefer AttentionWithKVCacheStage which handles
     * cache operations internally. Use this stage when you need fine-grained
     * control over cache timing.
     *
     * @see AttentionWithKVCacheStage for integrated attention + cache
     */
    class KVCacheAppendStage : public IComputeStage
    {
    public:
        struct Params
        {
            const TensorBase *K = nullptr; ///< Key to append [batch_size * seq_len, n_kv_heads * head_dim]
            const TensorBase *V = nullptr; ///< Value to append [batch_size * seq_len, n_kv_heads * head_dim]
            IUnifiedKVCache *kv_cache = nullptr;
            int layer_idx = 0;
            int seq_idx = 0;    ///< Starting sequence index (for batched caches)
            int num_tokens = 0; ///< Total tokens to append (0 = full tensor)
            int batch_size = 1; ///< Number of sequences in batch (for per-seq append)
            int seq_len = 0;    ///< Tokens per sequence (for slicing K/V per-seq)

            /// [Hybrid mode] Optional output for dequantized V (FP32)
            /// When provided and V is Q8_1, the stage will dequantize V to this buffer
            /// for use by downstream attention stages
            TensorBase *V_dequant_out = nullptr;
        };

        explicit KVCacheAppendStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::COPY; } // Cache ops are data movement
        bool supportsBackend(ComputeBackendType backend) const override { return true; }
        StageBufferRequirements getBufferRequirements() const override;
        std::vector<BufferDescriptor> getDeclaredOutputs() const override;

        /// Check if this stage will produce V_dequant output
        bool producesVDequant() const { return params_.V_dequant_out != nullptr; }

    private:
        Params params_;
    };

    /**
     * @brief Gather K/V from multiple cache slots into batched output tensors
     *
     * For batched decode with KV cache history, each sequence has K/V stored
     * at a separate seq_idx in the cache. This stage gathers them into contiguous
     * tensors suitable for batched attention computation.
     *
     * Output layout: [batch_size * max_kv_len, kv_dim]
     * where each sequence occupies [seq_idx * max_kv_len, (seq_idx+1) * max_kv_len)
     *
     * The stage outputs:
     * - out_K: Gathered key tensor
     * - out_V: Gathered value tensor
     * - max_kv_len: Maximum sequence length across all sequences (for masking)
     * - per_seq_kv_lens: Vector of actual kv_len per sequence (for variable-length masking)
     *
     * DAG usage pattern (batched decode):
     * @code
     * // Stage 1: Append new tokens to cache
     * graph.addNode("kv_append", createKVCacheAppend({...}), device);
     *
     * // Stage 2: Gather K/V from all batch slots
     * graph.addNode("kv_gather", createKVCacheGather({
     *     .kv_cache = cache, .layer_idx = layer,
     *     .batch_size = batch_size,
     *     .out_K = gathered_K, .out_V = gathered_V,
     * }), device);
     * graph.addDependency("kv_append", "kv_gather");
     *
     * // Stage 3: Batched attention with gathered K/V
     * graph.addNode("attention", createAttentionCompute({
     *     .Q = Q, .K = gathered_K, .V = gathered_V,
     *     .batch_size = batch_size, .kv_len = max_kv_len, ...
     * }), device);
     * graph.addDependency("kv_gather", "attention");
     * @endcode
     *
     * @see KVCacheAppendStage for appending tokens to cache
     * @see AttentionComputeStage for attention computation
     */
    class KVCacheGatherStage : public IComputeStage
    {
    public:
        struct Params
        {
            IUnifiedKVCache *kv_cache = nullptr;
            int layer_idx = 0;
            int batch_size = 1; ///< Number of sequences to gather

            TensorBase *out_K = nullptr; ///< Output K [batch_size * max_kv_len, kv_dim]
            TensorBase *out_V = nullptr; ///< Output V [batch_size * max_kv_len, kv_dim]

            /// After execute(), contains the max kv_len across all sequences
            int *out_max_kv_len = nullptr;

            /// After execute(), contains per-sequence kv_lens (size = batch_size)
            std::vector<int> *out_per_seq_kv_lens = nullptr;
        };

        explicit KVCacheGatherStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::COPY; } // Data movement
        bool supportsBackend(ComputeBackendType backend) const override { return true; }
        StageBufferRequirements getBufferRequirements() const override;

        /// Get the max kv_len from most recent execute() call
        int getMaxKVLen() const { return last_max_kv_len_; }

        /// Get per-sequence kv_lens from most recent execute() call
        const std::vector<int> &getPerSeqKVLens() const { return last_per_seq_kv_lens_; }

    private:
        Params params_;
        int last_max_kv_len_ = 0;
        std::vector<int> last_per_seq_kv_lens_;
    };

    /**
     * @brief Pure attention compute stage (no KV cache management)
     *
     * Delegates to KernelFactory::createAttention() for device-appropriate kernel
     * selection. This stage handles ONLY the attention computation:
     *   output = softmax(Q @ K^T / sqrt(head_dim) + mask) @ V
     *
     * For KV cache management, use KVCacheAppendStage separately in the DAG.
     * For integrated cache+attention, use AttentionWithKVCacheStage.
     *
     * Benefits of separation:
     * - **Composable**: Cache append and attention can run on different devices
     * - **Testable**: Pure attention kernel can be tested without cache
     * - **GPU-ready**: KernelFactory handles CPU/GPU dispatch transparently
     *
     * DAG usage pattern:
     * @code
     * // Stage 1: Append K/V to cache
     * graph.addNode("kv_append", createKVCacheAppend(...), device);
     *
     * // Stage 2: Get cached K/V (done in graph builder, not a stage)
     * TensorBase* K_cached = kv_cache->get_k_base(layer_idx, 0);
     * TensorBase* V_cached = kv_cache->get_v_base(layer_idx, 0);
     *
     * // Stage 3: Pure attention compute
     * graph.addNode("attention", createAttentionCompute({
     *     .Q = Q, .K = K_cached, .V = V_cached, .output = output,
     *     .batch_size = 1, .seq_len = 1, .kv_len = cached_tokens,
     *     .n_heads = n_heads, .n_kv_heads = n_kv_heads, .head_dim = head_dim,
     *     .causal = true
     * }), device);
     * graph.addDependency("kv_append", "attention");
     * @endcode
     *
     * @see KVCacheAppendStage for cache operations
     * @see AttentionWithKVCacheStage for integrated cache+attention
     * @see KernelFactory::createAttention() for kernel dispatch
     */
    class AttentionComputeStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Input/output tensors
            TensorBase *Q = nullptr;      ///< Query [batch_size * seq_len, n_heads * head_dim]
            TensorBase *K = nullptr;      ///< Key [batch_size * kv_len, n_kv_heads * head_dim]
            TensorBase *V = nullptr;      ///< Value [batch_size * kv_len, n_kv_heads * head_dim]
            TensorBase *output = nullptr; ///< Output [batch_size * seq_len, n_heads * head_dim]

            // Dimensions
            int batch_size = 1; ///< Number of sequences (1 for single sequence)
            int seq_len = 0;    ///< Query sequence length
            int kv_len = 0;     ///< Key/Value length (may differ from seq_len in decode)
            int n_heads = 0;    ///< Number of query heads
            int n_kv_heads = 0; ///< Number of KV heads (GQA: n_heads % n_kv_heads == 0)
            int head_dim = 0;   ///< Dimension per head

            // Attention configuration
            bool causal = true;   ///< Apply causal (lower-triangular) mask
            int window_size = -1; ///< Sliding window size (-1 = full attention)

            /// Execution mode (PREFILL, DECODE, BATCHED_DECODE, CHUNKED_PREFILL)
            /// Default: AUTO detection from batch_size, seq_len, kv_len
            AttentionMode attention_mode = AttentionMode::PREFILL;
            bool auto_detect_mode = true; ///< If true, detect mode from dimensions

            // Workspace buffers (pre-allocated for zero hot-path allocation)
            TensorBase *workspace_scores = nullptr;  ///< [n_heads * seq_len, kv_len]
            TensorBase *workspace_context = nullptr; ///< [seq_len, head_dim]
            TensorBase *workspace_mask = nullptr;    ///< [seq_len, kv_len] or nullptr

            // KV cache for dynamic length query at execution time
            // When set, kv_len is queried from cache instead of using the static value
            // This enables declarative graph construction where the actual kv_len
            // depends on prior KVCacheAppendStage execution
            IUnifiedKVCache *kv_cache = nullptr;
            int layer_idx = -1; ///< Layer index for KV cache query

            // Position offset for decode mode causal masking
            // In decode mode (seq_len < kv_len), this indicates the query's global position
            // For proper causal masking, query at position_offset can attend to [0, position_offset]
            // If 0 (default), will be auto-computed as (kv_len - seq_len) for decode mode
            int position_offset = 0;

            // Optional MPI context (for distributed logging, not tensor-parallel here)
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1; ///< Target device (-1 = CPU, ≥0 = GPU)
        };

        explicit AttentionComputeStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ATTENTION; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

        /// Update position offset for cached graph reuse
        void updateDynamicParams(int pos_offset, int seq_len) override
        {
            params_.position_offset = pos_offset;
            (void)seq_len; // Attention doesn't need seq_len update (kv_len queried dynamically)
        }

        /// Get params for testing
        const Params &getParams() const { return params_; }

    private:
        Params params_;
    };

    /**
     * @brief Fused attention + Wo projection stage
     *
     * Combines attention computation and output projection (Wo) into a single
     * kernel for improved cache locality and reduced memory traffic.
     *
     * Benefits over separate AttentionComputeStage + GEMMStage:
     * - Eliminates attention context quantization round-trip (FP32 → Q8_1 → FP32)
     * - Context stays in registers through Wo projection
     * - Single pass over V and Wo weights
     *
     * Uses FusedAttentionWoKernel internally, which supports:
     * - JIT backend: AVX-512 VNNI optimized (default, fastest)
     * - TILED backend: Cache-blocked for correctness/fallback
     * - REFERENCE backend: Pure C++ for testing
     *
     * Output goes directly to attn_proj buffer (skip attn_output intermediate).
     *
     * DAG usage pattern:
     * @code
     * // Replace: attention → attn_output → wo_proj → attn_proj
     * // With:    fused_attention_wo → attn_proj
     * graph.addNode("fused_attn_wo",
     *     ComputeStageFactory::createFusedAttentionWo({
     *         .Q = Q, .K = K, .V = V, .Wo = layer.wo,
     *         .output = attn_proj,  // Direct to projection output
     *         ...
     *     }), device);
     * @endcode
     *
     * @see FusedAttentionWoKernel for kernel implementation
     * @see AttentionComputeStage for non-fused attention
     */
    class FusedAttentionWoStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Input tensors (Q8_1 format)
            TensorBase *Q = nullptr; ///< Query [batch_size * seq_len, n_heads * head_dim]
            TensorBase *K = nullptr; ///< Key [batch_size * kv_len, n_kv_heads * head_dim]
            TensorBase *V = nullptr; ///< Value [batch_size * kv_len, n_kv_heads * head_dim]

            // Weight tensor (Q8_1 VNNI-packed or FP32)
            TensorBase *Wo = nullptr; ///< Output projection [d_model, n_heads * head_dim]

            // Output tensor (FP32)
            TensorBase *output = nullptr; ///< Output [batch_size * seq_len, d_model]

            // Dimensions
            int batch_size = 1; ///< Number of sequences
            int seq_len = 0;    ///< Query sequence length
            int kv_len = 0;     ///< Key/Value length (may differ in decode mode)
            int n_heads = 0;    ///< Number of query heads
            int n_kv_heads = 0; ///< Number of KV heads (GQA support)
            int head_dim = 0;   ///< Dimension per head
            int d_model = 0;    ///< Model dimension (n_heads * head_dim)

            // Attention configuration
            bool causal = true;      ///< Apply causal mask
            int position_offset = 0; ///< Position offset for decode mode

            // Backend selection (JIT, TILED, REFERENCE)
            FusedAttentionBackend backend = FusedAttentionBackend::JIT;

            // KV cache for dynamic kv_len query at execution time
            IUnifiedKVCache *kv_cache = nullptr;
            int layer_idx = -1;

            // Optional MPI context
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;

            // Optional context snapshot buffer for debugging/parity testing
            // When provided, the kernel will write the pre-Wo attention context to this buffer
            // Shape: [batch_size * seq_len, n_heads * head_dim] (FP32)
            TensorBase *context_snapshot = nullptr;

            // Hybrid mode: use streaming dequantization for Wo projection
            // When true, quantized Wo weights are dequantized to FP32 row-by-row
            // during GEMM computation, giving FP32-equivalent precision.
            // When false, uses faster Q8_1 VNNI path with quantized activations.
            bool use_hybrid_wo = false;

            // ═══════════════════════════════════════════════════════════════════
            // Q16_1 RESIDUAL FUSION (HybridQ16 mode)
            // ═══════════════════════════════════════════════════════════════════
            // When enabled, the JIT kernel fuses residual addition after Wo projection:
            // - Wo output stays in registers as FP32 (no intermediate store)
            // - Residual is loaded (Q16_1), dequantized, added
            // - Result is quantized to Q16_1 and stored
            // This eliminates FP32 intermediate memory traffic (2.8× reduction).
            //
            // NOTE: When fuse_residual_add=true, the `output` field should point
            // to the Q16_1 residual buffer (not FP32). The kernel will read the
            // current residual value, add the Wo projection, and write back.
            // ═══════════════════════════════════════════════════════════════════
            bool fuse_residual_add = false;
        };

        explicit FusedAttentionWoStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::FUSED_ATTENTION_WO; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

        /// Update position offset for cached graph reuse
        void updateDynamicParams(int pos_offset, int seq_len) override
        {
            params_.position_offset = pos_offset;
            (void)seq_len;
        }

        /// Get params for testing
        const Params &getParams() const { return params_; }

    private:
        Params params_;
        std::unique_ptr<FusedAttentionWoKernel> kernel_;
    };

    /**
     * @brief SwiGLU activation stage
     *
     * Type-safe implementation using TensorBase* instead of void*.
     * The tensor's native_type() determines precision dispatch.
     * Uses typed kernel dispatch based on tensor precision.
     */
    class SwiGLUStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Type-safe tensor pointers (required)
            const TensorBase *gate = nullptr; ///< Gate tensor [seq_len, intermediate_dim]
            const TensorBase *up = nullptr;   ///< Up tensor [seq_len, intermediate_dim]
            TensorBase *output = nullptr;     ///< Output tensor [seq_len, intermediate_dim]

            // Explicit seq_len override (for pre-allocated buffers)
            // If 0, derives from tensor dimensions
            int seq_len = 0;

            // Optional MPI context for distributed execution
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;
        };

        explicit SwiGLUStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::SWIGLU; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

    /**
     * @brief Residual addition stage: output = input + residual
     *
     * Precision-aware implementation:
     * - FP32: Simple float addition
     * - BF16/FP16: Convert to FP32, add, convert back
     * - Q8_1: Native block addition via simd::q8_1_add_q8_1()
     *
     * The Q8_1 path performs fused dequant-add-requant in registers,
     * avoiding memory traffic for intermediate FP32 values.
     */
    class ResidualAddStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Type-safe tensor pointers (required)
            const TensorBase *input = nullptr;    ///< Input tensor (projection output)
            const TensorBase *residual = nullptr; ///< Residual tensor (previous hidden state)
            TensorBase *output = nullptr;         ///< Output tensor (can be same as residual for in-place)

            // Number of elements to process (0 = use input->numel())
            // IMPORTANT: For decode mode with pre-allocated buffers, this must be set to
            // seq_len * hidden_dim to avoid processing garbage data beyond the actual sequence.
            size_t num_elements = 0;

            // Optional MPI context for distributed execution
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;
        };

        explicit ResidualAddStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ADD_RESIDUAL; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;

        // Type-specific implementations (dispatch based on tensor native_type)
        // num_elements parameter avoids out-of-bounds access with pre-allocated buffers
        bool executeFP32(IDeviceContext *ctx, size_t num_elements);
        bool executeBF16(IDeviceContext *ctx, size_t num_elements);
        bool executeFP16(IDeviceContext *ctx, size_t num_elements);
        bool executeQ8_1(IDeviceContext *ctx, size_t num_elements);

        // Mixed-type implementations for hybrid precision modes
        // FP32 input + Q8_1 residual → Q8_1 output (used when fused attention outputs FP32)
        bool executeFP32_Q8_1_to_Q8_1(IDeviceContext *ctx, size_t num_elements);

        // Q8_1 input + Q16_1 residual → Q16_1 output (HybridQ16 mode)
        // Uses SIMD-optimized q16_1_add_q8_1 for in-place accumulation
        bool executeQ8_1_Q16_1_to_Q16_1(IDeviceContext *ctx, size_t num_elements);

        // FP32 input + Q16_1 residual → Q16_1 output (HybridQ16 mode FFN residual)
        // Uses SIMD-optimized q16_1_add_fp32 for adding FP32 GEMM output to Q16_1 stream
        bool executeFP32_Q16_1_to_Q16_1(IDeviceContext *ctx, size_t num_elements);

        // Q8_1 input + FP32 residual → FP32 output (HybridQ16 mode when residual starts as FP32)
        // Dequantizes Q8_1 delta and adds to FP32 residual
        bool executeQ8_1_FP32_to_FP32(IDeviceContext *ctx, size_t num_elements);
    };

    // =============================================================================
    // Model-Level Stages (for ModelExecutor)
    // =============================================================================

    /**
     * @brief Embedding lookup stage
     *
     * Performs token → embedding lookup for transformer input processing.
     * Supports output to various precision formats (FP32, BF16, Q8_1).
     *
     * The embedding table is typically FP32 (stored in model weights).
     * Output format is determined by the output tensor's native_type().
     */
    class EmbeddingStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Input/output tensors
            const TensorBase *embed_table = nullptr; ///< Embedding table [vocab_size, d_model]
            const int *token_ids = nullptr;          ///< Token IDs to look up
            TensorBase *output = nullptr;            ///< Output [num_tokens, d_model]

            // Dimensions
            int num_tokens = 0; ///< Number of tokens to look up
            int d_model = 0;    ///< Embedding dimension
            int vocab_size = 0; ///< Vocabulary size (for bounds checking)

            // Batched input (alternative to token_ids)
            const std::vector<std::vector<int>> *token_batches = nullptr;
            int padded_seq_len = 0; ///< Padded sequence length for batched

            // Device placement
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;
        };

        explicit EmbeddingStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::EMBEDDING; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;

        /// Execute Q16_1 output path: FP32 lookup + quantization
        bool executeQ16_1Output();

        /// Zero out a row of the output tensor (for padding)
        static void zero_output_row(TensorBase *output, int row_idx, int d_model);
    };

    /**
     * @brief Quantize FP32 tensor to Q16_1 format
     *
     * Used to initialize the Q16_1 residual stream from FP32 embedding output
     * in HybridQ16 mode. This stage copies and quantizes FP32 → Q16_1.
     *
     * Typically used once at the start of inference to initialize the
     * typed residual stream before layer 0 runs.
     */
    class QuantizeToQ16_1Stage : public IComputeStage
    {
    public:
        struct Params
        {
            const TensorBase *input = nullptr; ///< FP32 input [num_elements]
            TensorBase *output = nullptr;      ///< Q16_1 output [num_elements]
            size_t num_elements = 0;           ///< Number of elements to quantize
        };

        explicit QuantizeToQ16_1Stage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::QUANTIZE; }
        size_t estimatedFlops() const override { return params_.num_elements * 4; }
        size_t estimatedMemoryBytes() const override { return params_.num_elements * 6; }
        bool supportsBackend(ComputeBackendType backend) const override { return backend == ComputeBackendType::CPU; }
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

    /**
     * @brief Language model head projection stage
     *
     * Projects hidden states to vocabulary logits for token prediction.
     * Typically the final stage in a forward pass before sampling.
     *
     * This stage wraps a GEMM operation but provides semantic clarity
     * in compute graphs and enables LM head-specific optimizations.
     */
    class LMHeadStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Input/output tensors
            const TensorBase *hidden_states = nullptr;  ///< Hidden states [seq_len, d_model]
            const TensorBase *lm_head_weight = nullptr; ///< LM head weights [vocab_size, d_model]
            TensorBase *logits = nullptr;               ///< Output logits [seq_len, vocab_size]

            // Dimensions
            int seq_len = 0;    ///< Sequence length
            int d_model = 0;    ///< Model dimension
            int vocab_size = 0; ///< Vocabulary size

            // Optional bias (some models have LM head bias)
            const float *bias = nullptr; ///< LM head bias [vocab_size] (nullable)

            // Device placement
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;
        };

        explicit LMHeadStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::LM_HEAD; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

    /**
     * @brief MPI Allreduce stage
     */
    class AllreduceStage : public IComputeStage
    {
    public:
        struct Params
        {
            TensorBase *buffer = nullptr; ///< Buffer to allreduce (in-place)
            void *mpi_comm = nullptr;     ///< MPI communicator (cast to MPI_Comm)
            size_t count = 0;             ///< Number of elements to reduce (0 = use buffer->numel())
        };

        explicit AllreduceStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ALLREDUCE; }
        bool requiresAllreduce() const override { return true; }
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

    /**
     * @brief MPI AllGather stage for collecting distributed tensor slices
     *
     * Used after column-parallel GEMM to reconstruct full output tensor.
     * For example, after LM head projection where each rank computes logits
     * for a slice of the vocabulary, AllGather combines them into full logits.
     *
     * Input: local_input [seq_len, vocab_local] on each rank
     * Output: full_output [seq_len, vocab_size] on ALL ranks (same data)
     */
    class AllGatherStage : public IComputeStage
    {
    public:
        struct Params
        {
            TensorBase *local_input = nullptr; ///< This rank's local slice (input)
            TensorBase *full_output = nullptr; ///< Full tensor on all ranks (output)
            void *mpi_comm = nullptr;          ///< MPI communicator (cast to MPI_Comm)
            int world_size = 1;                ///< Number of ranks for validation
            size_t actual_seq_len = 0;         ///< Actual sequence length (0 = use buffer shape)
        };

        explicit AllGatherStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ALLGATHER; }
        bool requiresAllreduce() const override { return true; } // MPI collective
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

    // =============================================================================
    // MoE Stages
    // =============================================================================

    /**
     * @brief MoE router stage: compute expert selection
     */
    class MoERouterStage : public IComputeStage
    {
    public:
        struct Params
        {
            const TensorBase *hidden = nullptr;       ///< Hidden states [seq_len, d_model]
            const TensorBase *gate_weights = nullptr; ///< Router weights [d_model, num_experts]
            TensorBase *router_logits = nullptr;      ///< Output: router logits [seq_len, num_experts]
            int seq_len = 0;
            int d_model = 0;
            int num_experts = 0;
        };

        explicit MoERouterStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_ROUTER; }
        size_t estimatedFlops() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

    /**
     * @brief Single expert FFN execution
     *
     * This stage handles tokens routed to one specific expert.
     * Multiple MoEExpertStages can execute in parallel on different devices.
     */
    class MoEExpertStage : public IComputeStage
    {
    public:
        struct Params
        {
            int expert_id = 0;                               ///< Which expert this is
            const TensorBase *input = nullptr;               ///< Input tokens for this expert
            TensorBase *output = nullptr;                    ///< Output buffer
            const TensorBase *gate_w = nullptr;              ///< Expert gate weights
            const TensorBase *up_w = nullptr;                ///< Expert up weights
            const TensorBase *down_w = nullptr;              ///< Expert down weights
            const std::vector<int> *token_indices = nullptr; ///< Which tokens to process
            int d_model = 0;
            int intermediate_dim = 0;
        };

        explicit MoEExpertStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_EXPERT_FFN; }
        std::string name() const override;
        size_t estimatedFlops() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

    /**
     * @brief Combine expert outputs with router weights
     */
    class MoECombineStage : public IComputeStage
    {
    public:
        struct Params
        {
            const std::vector<const TensorBase *> *expert_outputs = nullptr; ///< Outputs from each expert
            const std::vector<float> *expert_weights = nullptr;              ///< Router weights per token-expert
            const std::vector<int> *token_expert_map = nullptr;              ///< Which experts each token used
            TensorBase *output = nullptr;                                    ///< Combined output [seq_len, d_model]
            int seq_len = 0;
            int d_model = 0;
            int top_k = 0; ///< Experts per token
        };

        explicit MoECombineStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_COMBINE; }
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

    // =============================================================================
    // Stage Factory
    // =============================================================================

    /**
     * @brief Factory for creating device-appropriate compute stages
     */
    class ComputeStageFactory
    {
    public:
        /**
         * @brief Create a GEMM stage
         *
         * The stage uses KernelFactory at execute-time to select the appropriate
         * kernel based on IDeviceContext::deviceType(). No backend selection needed
         * at construction time.
         */
        static std::unique_ptr<IComputeStage> createGEMM(
            const GEMMStage::Params &params);

        /**
         * @brief Create a fused QKV GEMM stage
         *
         * This stage quantizes input once and runs Q, K, V projections using a
         * shared Q8_1 buffer. More efficient than separate Quantize + 3x GEMM stages.
         */
        static std::unique_ptr<IComputeStage> createFusedQKVGEMM(
            const FusedQKVGEMMStage::Params &params);

        /**
         * @brief Create a fused Gate/Up GEMM stage for FFN
         *
         * This stage quantizes input once and runs gate + up projections using a
         * shared Q8_1 buffer. More efficient than separate Quantize + 2x GEMM stages.
         */
        static std::unique_ptr<IComputeStage> createFusedGateUpGEMM(
            const FusedGateUpGEMMStage::Params &params);

        /**
         * @brief Create an RMSNorm stage
         */
        static std::unique_ptr<IComputeStage> createRMSNorm(
            const RMSNormStage::Params &params);

        /**
         * @brief Create a RoPE stage
         */
        static std::unique_ptr<IComputeStage> createRoPE(
            const RoPEStage::Params &params);

        /**
         * @brief Create a SwiGLU stage
         */
        static std::unique_ptr<IComputeStage> createSwiGLU(
            const SwiGLUStage::Params &params);

        /**
         * @brief Create a residual add stage
         */
        static std::unique_ptr<IComputeStage> createResidualAdd(
            const ResidualAddStage::Params &params);

        /**
         * @brief Create a MoE router stage for expert selection
         */
        static std::unique_ptr<IComputeStage> createMoERouter(
            const MoERouterStage::Params &params);

        /**
         * @brief Create an expert FFN stage for MoE
         */
        static std::unique_ptr<IComputeStage> createMoEExpert(
            const MoEExpertStage::Params &params);

        /**
         * @brief Create a MoE combine stage for weighted expert output combination
         */
        static std::unique_ptr<IComputeStage> createMoECombine(
            const MoECombineStage::Params &params);

        /**
         * @brief Create an Allreduce stage for MPI collective sum
         *
         * Used after row-parallel GEMM to combine partial results across ranks.
         */
        static std::unique_ptr<IComputeStage> createAllreduce(
            const AllreduceStage::Params &params);

        /**
         * @brief Create an AllGather stage for MPI collective gather
         *
         * Used after column-parallel GEMM (e.g., LM head) to collect distributed
         * output slices into a full tensor on all ranks. Each rank contributes
         * its local slice, and all ranks receive the complete result.
         *
         * Input: local_input [seq_len, dim_local] - this rank's portion
         * Output: full_output [seq_len, dim_full] - complete tensor on ALL ranks
         */
        static std::unique_ptr<IComputeStage> createAllGather(
            const AllGatherStage::Params &params);

        /**
         * @brief Create a production attention stage with KV cache and MPI support
         *
         * This is the recommended factory for attention in production pipelines.
         * For simple testing, use createAttention() instead.
         *
         * @param params Attention parameters including KV cache and MPI config
         * @return AttentionWithKVCacheStage instance
         */
        static std::unique_ptr<IComputeStage> createAttentionWithKVCache(
            const AttentionWithKVCacheStage::Params &params);

        /**
         * @brief Create a KV cache append stage for explicit cache management
         *
         * For advanced use cases where cache operations need to be pipelined
         * separately from attention computation.
         */
        static std::unique_ptr<IComputeStage> createKVCacheAppend(
            const KVCacheAppendStage::Params &params);

        /**
         * @brief Create a KV cache gather stage for batched decode
         *
         * Gathers K/V from multiple cache slots into batched output tensors.
         * Use after KVCacheAppendStage and before AttentionComputeStage for
         * batched decode with KV cache history.
         *
         * @param params Gather parameters (cache, batch_size, output tensors)
         * @return KVCacheGatherStage instance
         */
        static std::unique_ptr<IComputeStage> createKVCacheGather(
            const KVCacheGatherStage::Params &params);

        /**
         * @brief Create a pure attention compute stage using KernelFactory
         *
         * This is the new architecture for attention that:
         * - Uses TensorBase* for type-safe tensor handling
         * - Delegates to KernelFactory::createAttention() for kernel dispatch
         * - Supports CPU and GPU backends transparently
         *
         * Use this with KVCacheAppendStage for composable DAG execution.
         * For legacy integrated cache+attention, use createAttentionWithKVCache().
         *
         * @param params Attention compute parameters (Q, K, V, output, dimensions)
         * @return AttentionComputeStage instance
         */
        static std::unique_ptr<IComputeStage> createAttentionCompute(
            const AttentionComputeStage::Params &params);

        /**
         * @brief Create a fused attention + Wo projection stage
         *
         * Combines attention computation with output projection (Wo) for:
         * - Better cache locality (context stays in registers)
         * - Eliminated context quantization round-trip
         * - Reduced memory bandwidth
         *
         * Replaces separate AttentionComputeStage + GEMMStage (wo_proj) in the DAG.
         * Output goes directly to attn_proj buffer.
         *
         * @param params Fused attention parameters including Q, K, V, Wo, output
         * @return FusedAttentionWoStage instance
         */
        static std::unique_ptr<IComputeStage> createFusedAttentionWo(
            const FusedAttentionWoStage::Params &params);

        // =====================================================================
        // Model-Level Stage Factories (for ModelExecutor)
        // =====================================================================

        /**
         * @brief Create an embedding lookup stage
         *
         * Used at the start of forward pass to convert token IDs to embeddings.
         *
         * @param params Embedding parameters including token IDs and output tensor
         * @return EmbeddingStage instance
         */
        static std::unique_ptr<IComputeStage> createEmbedding(
            const EmbeddingStage::Params &params);

        /**
         * @brief Create an LM head projection stage
         *
         * Used at the end of forward pass to project hidden states to logits.
         *
         * @param params LM head parameters including hidden states and output logits
         * @return LMHeadStage instance
         */
        static std::unique_ptr<IComputeStage> createLMHead(
            const LMHeadStage::Params &params);
    };

} // namespace llaminar2
