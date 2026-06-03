/**
 * @file IComputeStage.h
 * @brief Base interface for compute stages and supporting types
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
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "../local_execution/device/DeviceContext.h"
#include "../debug/BufferRole.h"
#include "../config/RuntimeConfig.h"
#include "../local_execution/coherence/StageCoherence.h"
#include "ComputeStageUtils.h"
#include "../../tensors/BlockStructures.h"
#include "../../tensors/TensorKernels.h"
#include "../../utils/MPITopology.h"
#include "../../memory/StageBufferContract.h"

namespace llaminar2
{

    // Forward declarations
    namespace verification
    {
        struct LayoutExpectation;
    }
    using verification::LayoutExpectation;
    class ITensor;
    class TensorBase;
    class IKVCache;
    class ICPUKVCache;
    class IMPIContext;

    /**
     * @brief Compute byte size for a tensor region given dtype and dimensions
     *
     * For quantized formats, computes block-aligned storage:
     * - Q8_0: 34 bytes per 32 elements
     * - Q8_1: 36 bytes per 32 elements
     * - Q16_1: 72 bytes per 32 elements
     * - IQ4_NL: 18 bytes per 32 elements
     * - FP32: 4 bytes per element
     * - FP16/BF16: 2 bytes per element
     *
     * @param dtype Type string ("FP32", "Q8_1", "Q16_1", etc.)
     * @param rows Number of rows
     * @param cols Number of columns
     * @return Byte size for the specified region
     */
    inline size_t computeByteSizeForDtype(const char *dtype, size_t rows, size_t cols)
    {
        if (!dtype)
            return rows * cols * sizeof(float);

        // Handle quantized block formats
        constexpr size_t BLOCK_SIZE = 32;

        if (std::strcmp(dtype, "Q8_1") == 0)
        {
            // Q8_1: 36 bytes per 32-element block
            size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            return rows * blocks_per_row * sizeof(Q8_1Block);
        }
        else if (std::strcmp(dtype, "Q16_1") == 0 || std::strcmp(dtype, "Q16_1_32") == 0)
        {
            // Q16_1/Q16_1_32: 72 bytes per 32-element block
            size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            return rows * blocks_per_row * sizeof(Q16_1Block);
        }
        else if (std::strcmp(dtype, "Q16_1_64") == 0)
        {
            // Q16_1_64: 136 bytes per 64-element block
            constexpr size_t BLOCK_SIZE_64 = 64;
            size_t blocks_per_row = (cols + BLOCK_SIZE_64 - 1) / BLOCK_SIZE_64;
            return rows * blocks_per_row * sizeof(Q16_1Block_64);
        }
        else if (std::strcmp(dtype, "Q16_1_128") == 0)
        {
            // Q16_1_128: 264 bytes per 128-element block
            constexpr size_t BLOCK_SIZE_128 = 128;
            size_t blocks_per_row = (cols + BLOCK_SIZE_128 - 1) / BLOCK_SIZE_128;
            return rows * blocks_per_row * sizeof(Q16_1Block_128);
        }
        else if (std::strcmp(dtype, "Q8_0") == 0)
        {
            // Q8_0: 34 bytes per 32-element block
            size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            return rows * blocks_per_row * sizeof(Q8_0Block);
        }
        else if (std::strcmp(dtype, "IQ4_NL") == 0)
        {
            // IQ4_NL: 18 bytes per 32-element block
            size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            return rows * blocks_per_row * sizeof(IQ4_NLBlock);
        }
        else if (std::strcmp(dtype, "FP32") == 0)
        {
            return rows * cols * sizeof(float);
        }
        else if (std::strcmp(dtype, "FP16") == 0 || std::strcmp(dtype, "BF16") == 0)
        {
            return rows * cols * sizeof(uint16_t);
        }
        else if (std::strcmp(dtype, "INT8") == 0)
        {
            return rows * cols * sizeof(int8_t);
        }
        else if (std::strcmp(dtype, "INT32") == 0)
        {
            return rows * cols * sizeof(int32_t);
        }

        // Default to FP32
        return rows * cols * sizeof(float);
    }

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
            const char *name = nullptr;
            const void *data = nullptr;
            size_t rows = 0;
            size_t cols = 0;
            const char *dtype = "FP32";
            size_t element_size = sizeof(float);
            size_t byte_size = 0;      ///< Total byte size for native format (0 = use rows*cols*element_size)
            ITensor *tensor = nullptr; ///< Optional tensor pointer for coherence management
        };

        // Output buffers (written during execute)
        struct OutputBuffer
        {
            const char *name = nullptr;
            const void *data = nullptr;
            size_t rows = 0;
            size_t cols = 0;
            const char *dtype = "FP32";
            size_t element_size = sizeof(float);
            size_t byte_size = 0;      ///< Total byte size for native format (0 = use rows*cols*element_size)
            ITensor *tensor = nullptr; ///< Optional tensor pointer for coherence management
        };

        // Weight/parameter buffers (read-only during execute)
        struct WeightBuffer
        {
            const char *name = nullptr;
            const ITensor *tensor = nullptr;
            const void *raw_data = nullptr;
            size_t raw_size = 0;
            size_t rows = 0;
            size_t cols = 0;
            const char *dtype = nullptr;
        };

        // Scalar parameters
        struct ScalarParam
        {
            const char *name = nullptr;
            double value = 0;
            const char *dtype = "float";
        };

        std::vector<InputBuffer> inputs;
        mutable std::vector<OutputBuffer> outputs; // mutable for ensureOutputsOnHost() const
        std::vector<WeightBuffer> weights;
        std::vector<ScalarParam> scalars;

        // Convenience methods for building dump info
        StageDumpInfo &addInput(const char *name, const float *data, size_t rows, size_t cols)
        {
            inputs.push_back({name, data, rows, cols, "FP32", sizeof(float)});
            return *this;
        }

        StageDumpInfo &addInput(const char *name, const ITensor *tensor, size_t rows, size_t cols);

        StageDumpInfo &addInputQ8_1(const char *name, const void *data, size_t rows, size_t cols)
        {
            inputs.push_back({name, data, rows, cols, "Q8_1", sizeof(Q8_1Block)});
            return *this;
        }

        StageDumpInfo &addOutput(const char *name, const float *data, size_t rows, size_t cols)
        {
            outputs.push_back({name, data, rows, cols, "FP32", sizeof(float)});
            return *this;
        }

        StageDumpInfo &addOutput(const char *name, const ITensor *tensor, size_t rows, size_t cols);

        StageDumpInfo &addWeight(const char *name, const ITensor *tensor,
                                 size_t rows, size_t cols, const char *dtype)
        {
            weights.push_back({name, tensor, nullptr, 0, rows, cols, dtype});
            return *this;
        }

        StageDumpInfo &addWeight(const char *name, const ITensor *tensor);

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

        /**
         * @brief Ensure all output tensors are synced from GPU to host
         *
         * Call this BEFORE reading output.data for verification/dumping.
         * This is a deferred sync - outputs are NOT synced in addOutput().
         * This allows GPU kernels to run async without blocking.
         */
        void ensureOutputsOnHost() const;
    };

    /**
     * @brief Types of compute operations
     */
    enum class ComputeStageType
    {
        // Matrix operations
        GEMM,
        GEMM_BIAS,
        GEMM_FUSED_QKV,
        GEMM_FUSED_GATE_UP,

        // Normalization
        RMS_NORM,
        LAYER_NORM,

        // Activations
        SWIGLU,
        GELU,
        SILU,

        // Attention
        ROPE,
        ATTENTION,
        ATTENTION_QK,
        ATTENTION_SOFTMAX,
        ATTENTION_V,

        // Element-wise
        ADD_RESIDUAL,
        SCALE,

        // MoE specific
        MOE_ROUTER,
        MOE_EXPERT_FFN,
        MOE_SHARED_EXPERT_FFN,      ///< Shared expert FFN (distinct from per-expert MOE_EXPERT_FFN)
        MOE_SHARED_EXPERT_GATE,     ///< Shared expert sigmoid gate
        MOE_EXPERT_DISPATCH,        ///< Host-side dispatch descriptor builder for expert-parallel tiers
        MOE_EXPERT_PARALLEL_REDUCE, ///< Host-side dense partial reduction for expert-parallel tiers
        MOE_SPARSE_DISPATCH,        ///< Graph-native sparse MoE payload dispatch
        MOE_LOCAL_EXPERT,           ///< Participant-local sparse MoE expert compute
        MOE_SPARSE_RETURN_REDUCE,   ///< Graph-native sparse MoE return reduce

        // Collective
        ALLREDUCE,
        ALLGATHER,
        ALLGATHER_V, ///< Variable-count allgather for heterogeneous TP

        // Point-to-Point (Pipeline Parallelism)
        SEND_ACTIVATIONS,
        RECV_ACTIVATIONS,
        LOCAL_PP_TRANSFER,  ///< Local PP activation transfer (intra-node GPU-to-GPU)
        GLOBAL_PP_TRANSFER, ///< Global PP activation transfer (cross-rank MPI send/recv)

        // Utility
        COPY,
        ROW_SELECT, ///< Copy one dynamically selected source row into a stable scratch row.
        QUANTIZE,
        DEQUANTIZE,

        // Model-level operations
        EMBEDDING,
        LM_HEAD,
        FINAL_NORM,

        // KV Cache operations
        KV_CACHE_APPEND,
        KV_CACHE_GATHER,
        ATTENTION_COMPUTE,

        // Quantization
        QUANTIZE_Q16_1,

        // Per-head normalization (Qwen3)
        QK_NORM,

        // Fused operations (GPU optimization)
        FUSED_RESIDUAL_NORM,
        FUSED_ADD_ALLREDUCE, ///< Fused residual-add + allreduce (MoE output combine + TP reduce)

        // GDN (Gated Delta Network) stages
        ATTENTION_OUTPUT_GATE, ///< Sigmoid gate on attention output
        GATED_RMS_NORM,        ///< RMSNorm with learned multiplicative gate
        GDN_PROJECTION,        ///< 4 separate GEMMs: in_proj_qkv, in_proj_z, in_proj_a, in_proj_b
        SHORT_CONV1D,          ///< Causal depthwise conv1d (kernel=4) + SiLU
        GDN_RECURRENCE,        ///< Delta rule recurrence (chunk prefill, single-step decode)

        // Qwen 3.5 FA-specific
        Q_GATE_SPLIT, ///< Split interleaved Q+gate GEMM output into separate buffers

        // MTP sidecar
        MTP_CONCAT, ///< Concatenate normalized draft embedding and terminal hidden rows
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
        struct RequiredPointer
        {
            const char *name = nullptr;
            const void *ptr = nullptr;
        };

        /**
         * @brief Dynamic bookkeeping for fixed-bucket prefill graph replay.
         *
         * Bucketed prefill graphs may execute a padded, fixed topology while
         * only a prefix of that bucket is real prompt data. Stages that update
         * host-side sequence state after graph replay use this metadata to keep
         * KV heads, recurrent state, and future row-selection logic aligned to
         * the real token count rather than the padded execution length.
         */
        struct PrefillReplayParams
        {
            int real_seq_len = 0;   ///< Real, non-padding token count in this replay.
            int bucket_seq_len = 0; ///< Fixed graph execution length for this replay.
            int token_offset = 0;   ///< Offset of this chunk within the original prompt.
        };

        /**
         * @brief Construct a stage with required device assignment
         *
         * Every derived stage MUST call this in its initializer list:
         *   MyStage(Params p) : IComputeStage(p.device_id), params_(std::move(p)) {}
         *
         * This ensures device assignment is compiler-enforced - forgetting to
         * pass device_id causes a compilation error (no default constructor).
         */
        explicit IComputeStage(DeviceId device) : device_id_(device) {}

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

        /**
         * @brief Validate prepared model-weight references before execution.
         *
         * Non-weight stages return true. Model-weight-backed GEMM stages override
         * this to ensure graph construction provided PreparedWeightStore refs and
         * that the store still contains those refs.
         */
        virtual bool validatePreparedWeights(std::string *error) const
        {
            if (error)
                error->clear();
            return true;
        }

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
         */
        virtual size_t estimatedFlops() const { return 0; }

        /**
         * @brief Estimated memory bandwidth (bytes read + written)
         */
        virtual size_t estimatedMemoryBytes() const { return 0; }

        /**
         * @brief Check if this stage supports a specific backend
         */
        virtual bool supportsBackend(ComputeBackendType backend) const = 0;

        /**
         * @brief Get detailed dump info for debugging (cached)
         *
         * Returns comprehensive information about all buffers and parameters
         * used by this stage, enabling full reproducibility of execution.
         *
         * This method caches the result after first call since tensor pointers
         * don't change after graph construction. Call invalidateDumpInfoCache()
         * if stage parameters change (rare - only for dynamic reconfiguration).
         *
         * @note Performance: First call builds info, subsequent calls return cached.
         */
        const StageDumpInfo &getDumpInfo() const
        {
            if (!dump_info_cached_)
            {
                cached_dump_info_ = buildDumpInfoImpl();
                dump_info_cached_ = true;
            }
            return cached_dump_info_;
        }

        /**
         * @brief Invalidate cached dump info (for dynamic reconfiguration)
         *
         * Call this if stage parameters change after construction (rare).
         */
        void invalidateDumpInfoCache() const { dump_info_cached_ = false; }

        /**
         * @brief Get buffer requirements for this stage
         *
         * Used by DeviceGraphBufferManager for intelligent buffer allocation and reuse.
         * Stages should declare all buffers they read, write, or allocate.
         */
        virtual StageBufferRequirements getBufferRequirements() const
        {
            return StageBufferRequirements{};
        }

        /**
         * @brief Get the declarative buffer contract for this stage.
         *
         * Returns which arena-managed buffers this stage reads, writes, and
         * uses in-place, plus direct weight tensor pointers. The executor
         * uses this contract (instead of StageDumpInfo) to drive coherence:
         *
         *   1. For each input binding: arena.prepareForRead(id, device)
         *   2. For each weight tensor: tensor->ensureOnDevice(device)
         *   3. For each output binding: arena.prepareForWrite(id, device)
         *   4. stage->execute(ctx)
         *   5. For each output/inout: arena.markWritten(id, device, stream)
         *
         * Default returns empty contract (not yet migrated). Stages opt-in
         * by overriding this and returning a non-empty contract.
         *
         * Design notes:
         *   - Activations use BufferId keys (arena-managed)
         *   - Weights use direct ITensor* (external, read-only)
         *   - KV caches are out of scope (managed by IKVCache)
         *   - Effective dimensions (M in GEMM) are stage-internal
         *   - updateDynamicParams/onGraphReplayed remain orthogonal
         *
         * @return StageBufferContract (empty if not migrated)
         */
        virtual StageBufferContract bufferContract() const
        {
            return StageBufferContract{};
        }

        /**
         * @brief Get layout expectation for automatic validation
         *
         * If a stage returns a non-empty LayoutExpectation, the DeviceGraphExecutor
         * will automatically validate all input/output tensors with declared
         * layouts (via getBufferRequirements().withLayout()) against this
         * expectation at stage entry and exit.
         *
         * This enables declarative layout validation:
         * 1. Stage declares expected layouts in getBufferRequirements():
         *    @code
         *    reqs.addInput("Q", ...).withLayout(TensorLayout::Q_SEQ_HEAD_DIM);
         *    @endcode
         * 2. Stage returns model dimensions in getLayoutExpectation():
         *    @code
         *    return LayoutExpectation::forAttention(
         *        params_.head_dim, params_.n_heads, params_.n_kv_heads,
         *        local_heads, local_kv_heads);
         *    @endcode
         * 3. DeviceGraphExecutor automatically validates on each execute()
         *
         * @return LayoutExpectation with model dimensions, or empty (is_set()==false)
         *         if no automatic validation is desired.
         */
        virtual LayoutExpectation getLayoutExpectation() const;

        /**
         * @brief Get output buffer descriptors for this stage
         *
         * Returns descriptors for all output buffers this stage produces.
         * Used by graph analysis for liveness tracking and buffer reuse.
         */
        virtual std::vector<BufferDescriptor> getDeclaredOutputs() const
        {
            return {};
        }

        /**
         * @brief Whether this stage requires MPI allreduce after execution
         *
         * Used for dependency analysis in tensor-parallel execution.
         */
        virtual bool requiresAllreduce() const { return false; }

        /**
         * @brief Whether this stage can be captured inside a GPU graph
         *
         * Returns true by default. Override to return false for stages whose
         * kernel launch parameters change between graph replays (e.g., attention
         * and KV cache stages where kv_len grows each decode step). Non-capturable
         * stages are executed manually between graph segments.
         *
         * @return true if this stage's kernel launches have stable grid dimensions
         *         and parameters across decode steps
         */
        virtual bool isGraphCapturable() const { return true; }

        /**
         * @brief Whether warmup can make a cold stage graph-capturable.
         *
         * Stages return true here when their backend and shape support graph
         * capture in principle, but their cold isGraphCapturable() answer may
         * remain false until the first warmup execution creates runtime tables,
         * descriptor banks, kernels, or scratch. The segmented planner may place
         * these stages in capturable segments before warmup; the capture phase
         * then hard-fails if isGraphCapturable() is still false.
         */
        virtual bool supportsWarmupDependentGraphCapture() const { return false; }

        /**
         * @brief Whether segmented capture should rebuild its segment plan after warmup.
         *
         * This is a legacy escape hatch for stages whose cold plan cannot be
         * predicted safely. Prefer supportsWarmupDependentGraphCapture() for
         * fixed-topology stages so warmup and capture use one segment plan.
         */
        virtual bool requiresPostWarmupGraphSegmentRebuild() const { return false; }

        /**
         * @brief Whether this manual stage must complete before a following graph segment may run.
         *
         * Segmented GPU graph execution runs non-capturable stages between
         * captured graph segments. Sparse MoE dispatch/return stages are manual
         * collective boundaries: a later captured continuation segment must not
         * consume their outputs until every participant has completed the same
         * collective key.
         */
        virtual bool isManualGraphBoundary() const { return false; }

        /**
         * @brief True when the last manual boundary execution completed globally.
         *
         * Only meaningful when isManualGraphBoundary() is true. Direct normal
         * execution may still accept an incomplete nonblocking collective result,
         * but segmented graph replay uses this to stop before launching the next
         * captured segment.
         */
        virtual bool manualGraphBoundaryComplete() const { return true; }

        /**
         * @brief True when the stage captured mutable verifier-row state.
         *
         * MTP verifier forwards may compute multiple candidate rows in one
         * graph. Stages with recurrent model state can snapshot their state
         * after each row so rollback can restore the accepted prefix without
         * replaying the main graph.
         */
        virtual bool hasVerifierStateCapture() const { return false; }

        /**
         * @brief Restore mutable model state captured after a verifier row.
         *
         * The row is zero-based within the most recent all-position verifier
         * forward. Implementations should restore only stage-owned mutable
         * model state; KV truncation and runner bookkeeping are handled above.
         */
        virtual bool restoreVerifierStateCaptureRow(int row, void *stream = nullptr)
        {
            (void)row;
            (void)stream;
            return false;
        }

        /**
         * @brief Whether this stage allows all-zero output tensors
         *
         * By default, all-zero outputs are treated as bugs (likely uninitialized
         * buffers or upstream computation failures). However, some stages may
         * legitimately produce all-zero outputs in specific scenarios:
         *
         * - KVCacheGatherStage: Before first token, cache may be empty
         * - SwiGLU: Theoretically possible with extreme gate values (rare)
         *
         * Override this to return true if your stage has legitimate all-zero
         * output scenarios. The DeviceGraphExecutor will skip zero-check validation
         * for stages that return true.
         *
         * @return false by default (all-zero outputs are bugs)
         */
        virtual bool allowsZeroOutput() const { return false; }

        // =========================================================================
        // Device Coherence (Phase 2: Automatic Stage Boundary Coherence)
        // =========================================================================

        /**
         * @brief Get coherence policy for this stage
         *
         * Controls automatic device coherence at stage boundaries:
         * - NONE: No automatic coherence (for MPI stages, custom management)
         * - INPUT: Only cohere inputs (outputs managed manually)
         * - OUTPUT: Only mark outputs dirty (assume inputs are ready)
         * - FULL: Both inputs and outputs (default for most stages)
         *
         * Override for stages that manage their own coherence (e.g., MPI stages
         * that coordinate data movement across ranks).
         *
         * @return FULL by default for automatic input coherence and output marking
         */
        virtual CoherencePolicy coherencePolicy() const { return CoherencePolicy::FULL; }

        /**
         * @brief Get the device for execution (non-virtual, authoritative)
         *
         * Returns the DeviceId where this stage will execute.
         * Set by each stage's constructor via setDevice(params.device_id).
         *
         * This is NOT a "preference" - it's the authoritative device assignment.
         * Stages MUST call setDevice() in their constructor to set this.
         *
         * @return The device this stage executes on (CPU by default if not set)
         */
        DeviceId device() const { return device_id_; }

        /**
         * @brief Set GPU stream for kernel dispatch
         *
         * When GPU graph capture is active, the executor sets this to the
         * capture stream so all GPU kernels submit work on the correct stream.
         * The pointer is backend-agnostic: cast to hipStream_t (ROCm) or
         * cudaStream_t (CUDA) as needed. nullptr means "use default/legacy
         * stream" (backward compatible).
         *
         * @param stream Opaque GPU stream pointer (hipStream_t / cudaStream_t as void*)
         */
        void setGPUStream(void *stream) { gpu_stream_ = stream; }

        /**
         * @brief Get the GPU stream for kernel dispatch
         *
         * Returns the stream assigned by the executor, or nullptr if none was set.
         * GPU kernel wrapper functions should pass this to their launch calls.
         *
         * @return Opaque GPU stream pointer (nullptr = use default stream)
         */
        void *gpuStream() const { return gpu_stream_; }

        /**
         * @brief Update dynamic parameters for graph reuse
         *
         * Allows updating position-dependent parameters (like RoPE position offset)
         * without rebuilding the entire compute graph. Called by DeviceGraphExecutor
         * between decode steps.
         *
         * @param pos_offset Current position in sequence (for RoPE, causal mask)
         * @param seq_len Current sequence length being processed
         */
        virtual void updateDynamicParams(int pos_offset, int seq_len)
        {
            (void)pos_offset;
            (void)seq_len;
        }

        /**
         * @brief Returns true if this stage overrides updateDynamicParams().
         *
         * Used by DeviceGraphOrchestrator to precompute a list of stages
         * needing per-step parameter updates, avoiding iteration over all
         * ~339 stages with hash lookups on every decode step.
         */
        virtual bool hasDynamicParams() const { return false; }

        /**
         * @brief Reset request-scoped stage state without discarding the graph.
         *
         * Cached ComputeGraphs persist across prompt boundaries, so any stage
         * state that depends on the previous request must be cleared when the
         * orchestrator runs clear_cache(). Topology, tensor bindings, workspace
         * bindings, and model weights must remain intact so graph reuse still
         * works. Derived stages should reset only dynamic host metadata and
         * kernel stream bindings here.
         */
        virtual void resetSessionState()
        {
            gpu_stream_ = nullptr;
        }

        /**
         * @brief Update prefill replay bookkeeping before a captured graph launch.
         *
         * The executor calls this on cached prefill graph hits after normal
         * dynamic params are refreshed and before capture/replay callbacks can
         * run. Decode graph replay continues to use updateDynamicParams() only.
         * Stages should ignore this unless their host-side replay callback must
         * distinguish real tokens from padded bucket rows.
         *
         * @param params Real-token and bucket metadata for the upcoming prefill replay.
         */
        virtual void updatePrefillReplayParams(const PrefillReplayParams &params)
        {
            (void)params;
        }

        /**
         * @brief Returns true if this stage consumes updatePrefillReplayParams().
         *
         * Used by the forward graph cache to precompute a small stage list and
         * avoid scanning every graph node before each cached prefill launch.
         */
        virtual bool hasPrefillReplayParams() const { return false; }

        /**
         * @brief Whether this stage can safely execute padded prefill buckets.
         *
         * Stateful prefill stages such as GDN recurrence and short convolution
         * may run fixed bucket-shaped kernels, but their recurrent state must
         * commit as though only the real prompt prefix executed. Stages return
         * true here only when their backend implements that real-length
         * contract for graph replay.
         */
        virtual bool supportsPaddedPrefillRealLengthContract() const { return false; }

        /**
         * @brief Whether cold padded-prefill graph preflight may allow this stage.
         *
         * Padded bucket preflight can run before the first normal warmup pass,
         * while some stages intentionally allocate kernels, descriptor tables,
         * or scratch buffers during that warmup. Such stages should return true
         * here when their backend and shape support fixed-bucket prefill capture
         * in principle, and keep isGraphCapturable() as the stricter
         * capture-time readiness check.
         *
         * The default preserves legacy behavior for existing stages: if a stage
         * has no separate cold-support contract, padded preflight still requires
         * normal graph-capture readiness.
         */
        virtual bool supportsPaddedPrefillGraphCapturePreflight() const { return isGraphCapturable(); }

        /**
         * @brief Called after a captured GPU graph segment is replayed.
         *
         * This method is invoked by DeviceGraphExecutor after launching a graph segment
         * containing this stage (Phase 3 replay). It allows stages to perform
         * host-side bookkeeping that would normally happen inside execute().
         *
         * Primary use case: KVCacheAppendStage advances the ring buffer head
         * position and count after the replayed graph performs the actual GPU append.
         * This MUST happen AFTER the graph replay (not before in updateDynamicParams)
         * to preserve the invariant that get_cached_tokens() returns the PREVIOUS
         * step's count during updateDynamicParams.
         */
        virtual void onGraphReplayed() {}

        /**
         * @brief Returns true if this stage overrides onGraphReplayed().
         *
         * Used by DeviceGraphExecutor to precompute a list of stages needing
         * post-replay callbacks, avoiding per-step hash map lookups.
         */
        virtual bool needsOnGraphReplayed() const { return false; }

    protected:
        /**
         * @brief Build dump info (implemented by derived classes)
         *
         * This is the method stages implement to describe their buffers.
         * Called once by getDumpInfo() and cached thereafter.
         *
         * @note This is a REQUIRED method - all stages must implement it for
         *       the TensorVerification framework to work correctly.
         */
        virtual StageDumpInfo buildDumpInfoImpl() const = 0;

        /**
         * @brief Build partial StageDumpInfo from this stage's bufferContract().
         *
         * Populates the weight entries from the contract's weight_tensors list.
         * Stages that implement bufferContract() can call this in their
         * buildDumpInfoImpl() and then append inputs/outputs with dynamic dims:
         *
         * @code
         * StageDumpInfo buildDumpInfoImpl() const override {
         *     auto info = buildDumpInfoFromContract();
         *     info.addInput("hidden_state", hidden_state_, M, K);
         *     info.addOutput("output", output_, M, N);
         *     return info;
         * }
         * @endcode
         */
        StageDumpInfo buildDumpInfoFromContract() const
        {
            StageDumpInfo info;
            const auto contract = bufferContract();
            for (size_t i = 0; i < contract.weight_tensors.size(); ++i)
            {
                const ITensor *w = contract.weight_tensors[i];
                if (w)
                    info.addWeight("weight", w);
            }
            return info;
        }

        // =========================================================================
        // Tracing Infrastructure (for debugging/profiling)
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
        void traceInput(const std::string &name, const ITensor *tensor) const;

        /**
         * @brief Trace output tensor values (only if tracing enabled).
         *
         * @param name Tensor name (e.g., "C", "output")
         * @param tensor Tensor to trace
         */
        void traceOutput(const std::string &name, const ITensor *tensor) const;

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

        /**
         * @brief Get or refresh a stage-cached kernel pointer by tensor dtype variant
         *
         * Many compute stages cache non-owning kernel pointers sourced from
         * KernelFactory device-scoped caches. This helper centralizes the
         * repeated refresh pattern:
         * - First use: create/resolve kernel
         * - DType change: refresh kernel pointer for new tensor variant
         * - Same dtype: reuse existing pointer
         *
         * @tparam KernelT Cached kernel interface type (e.g., ITensorRoPE)
         * @tparam TensorLike Tensor type exposing native_type()
         * @tparam FactoryFn Callable returning KernelT*
         * @param cached_kernel Stage-local non-owning kernel pointer
         * @param cached_tensor_type Stage-local cached tensor type discriminator
         * @param tensor Tensor whose dtype determines kernel variant
         * @param factory Callable to create/resolve kernel pointer
         * @return Kernel pointer (may be nullptr if factory fails)
         */
        template <typename KernelT, typename TensorLike, typename FactoryFn>
        KernelT *getOrRefreshKernelByTensorType(
            KernelT *&cached_kernel,
            int &cached_tensor_type,
            const TensorLike *tensor,
            FactoryFn factory) const
        {
            if (!tensor)
            {
                return nullptr;
            }

            const int tensor_type = static_cast<int>(tensor->native_type());
            if (!cached_kernel || cached_tensor_type != tensor_type)
            {
                cached_kernel = factory();
                cached_tensor_type = tensor_type;
            }

            return cached_kernel;
        }

        /**
         * @brief Validate stage execution context pointer and emit consistent error log
         *
         * @param ctx Device context pointer passed to execute()
         * @param stage_name Human-readable stage tag for logs (e.g., "RMSNormStage")
         * @return true if ctx is valid, false otherwise
         */
        bool ensureContext(const IDeviceContext *ctx, const char *stage_name) const
        {
            if (ctx)
            {
                return true;
            }

            LOG_ERROR("[" << (stage_name ? stage_name : "ComputeStage") << "] Null device context");
            return false;
        }

        /**
         * @brief Bind this stage's GPU stream to a kernel when available
         *
         * Many stages repeat `kernel->setGPUStream(gpuStream())`. This helper
         * centralizes that pattern while keeping behavior unchanged.
         *
         * @tparam KernelT Kernel type exposing setGPUStream(void*)
         * @param kernel Kernel pointer (may be nullptr)
         * @return The same kernel pointer for fluent usage
         */
        template <typename KernelT>
        KernelT *bindStageStream(KernelT *kernel) const
        {
            if (kernel)
            {
                kernel->setGPUStream(gpuStream());
            }
            return kernel;
        }

        /**
         * @brief Shared stage wrapper for optional ITensor->TensorBase cast (mutable)
         */
        TensorBase *asTensorBasePtr(ITensor *tensor, const char *name = nullptr) const
        {
            return llaminar2::asTensorBase(tensor, name);
        }

        /**
         * @brief Shared stage wrapper for optional ITensor->TensorBase cast (const)
         */
        const TensorBase *asTensorBasePtr(const ITensor *tensor, const char *name = nullptr) const
        {
            return llaminar2::asTensorBase(tensor, name);
        }

        /**
         * @brief Shared stage wrapper for required ITensor->TensorBase cast (mutable)
         */
        TensorBase *requireTensorBasePtr(ITensor *tensor, const char *name) const
        {
            return llaminar2::requireTensorBase(tensor, name);
        }

        /**
         * @brief Shared stage wrapper for required ITensor->TensorBase cast (const)
         */
        const TensorBase *requireTensorBasePtr(const ITensor *tensor, const char *name) const
        {
            return llaminar2::requireTensorBase(tensor, name);
        }

        /**
         * @brief Validate required pointers and emit consistent error log
         *
         * Useful for stage preflight checks where multiple required pointers
         * must be present before execution.
         *
         * @param stage_name Human-readable stage tag for logs
         * @param required List of pointers with names
         * @return true if all pointers are non-null, false otherwise
         */
        bool ensureRequiredPointers(
            const char *stage_name,
            std::initializer_list<RequiredPointer> required) const
        {
            std::string missing;
            for (const auto &item : required)
            {
                if (!item.ptr)
                {
                    if (!missing.empty())
                    {
                        missing += ", ";
                    }
                    missing += (item.name ? item.name : "<unnamed>");
                }
            }

            if (missing.empty())
            {
                return true;
            }

            LOG_ERROR("[" << (stage_name ? stage_name : "ComputeStage")
                          << "] Missing required pointer(s): " << missing);
            return false;
        }

        // =========================================================================
        // Shape Validation Infrastructure (Task 7: Debugging)
        // =========================================================================

    public:
        /**
         * @brief Contract for a single tensor's expected shape.
         *
         * Used by validateShapes() to verify tensor dimensions match expectations.
         */
        struct TensorShapeContract
        {
            std::string name;             ///< Tensor name for error messages
            const ITensor *tensor;        ///< Tensor to validate
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
        void validateShape(const std::string &tensor_name, const ITensor *tensor,
                           const std::vector<size_t> &expected) const;

        /**
         * @brief Validate that two tensors have compatible shapes for matrix multiply
         */
        void validateMatmulShapes(const std::string &a_name, const ITensor *a,
                                  const std::string &b_name, const ITensor *b) const;

    private:
        DeviceId device_id_;         ///< Authoritative device (set via constructor, no default)
        void *gpu_stream_ = nullptr; ///< GPU stream for kernel dispatch (nullptr = default stream)

        // Cached dump info (built once, reused for all subsequent calls)
        mutable StageDumpInfo cached_dump_info_;
        mutable bool dump_info_cached_ = false;

        static bool shapesMatch(const std::vector<size_t> &actual,
                                const std::vector<size_t> &expected,
                                bool allow_broadcast);
        static std::string shapeToString(const std::vector<size_t> &shape);
    };

} // namespace llaminar2
