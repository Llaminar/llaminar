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
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <functional>
#include "../local_execution/device/DeviceContext.h"
#include "../debug/BufferRole.h"
#include "../config/RuntimeConfig.h"
#include "../local_execution/coherence/CoherencePolicy.h"
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

        /**
         * @brief Describe immutable raw parameter storage without calling it an activation.
         *
         * Legacy stage APIs may still expose a read-only parameter as an unowned
         * host pointer. Keeping that storage in the weight collection prevents
         * per-execution activation validation while preserving diagnostic shape
         * and type metadata.
         *
         * @param name Stable diagnostic name.
         * @param data Immutable raw parameter bytes.
         * @param bytes Exact byte count available at @p data.
         * @param rows Logical row count.
         * @param cols Logical column count.
         * @param dtype Native element type label.
         */
        StageDumpInfo &addRawWeight(
            const char *name,
            const void *data,
            size_t bytes,
            size_t rows,
            size_t cols,
            const char *dtype)
        {
            weights.push_back({name, nullptr, data, bytes, rows, cols, dtype});
            return *this;
        }

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
         *
         * GPU-backed outputs that need a device-to-host publication require the
         * producer stream. Passing null for those outputs is a programming error:
         * the graph executor must thread the stage stream through explicitly.
         */
        void ensureOutputsOnHost(void *stream = nullptr) const;
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
        GEMM_FUSED_KV,
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
        MOE_CANONICAL_ROUTE_REDUCE, ///< Router-ordered LocalTP contribution reduction
        MOE_SHARED_RANK_BANK_PUBLISH, ///< Publish one shared partial into its canonical participant bank
        MOE_CANONICAL_PUBLICATION_FINALIZE, ///< Root-only fixed-order routed/shared finalizer
        MOE_OVERLAY_TICKET_PUBLISH,  ///< Captured fixed-capacity heterogeneous dispatch ticket
        MOE_OVERLAY_TICKET_CONSUME,  ///< Captured fixed-capacity heterogeneous return ingress
        MOE_OVERLAY_ACTIVATION_DISPATCH_PACK, ///< Device pack into a mapped node-local activation lane
        MOE_OVERLAY_ACTIVATION_DISPATCH_CONSUME, ///< Device consume from a mapped node-local activation lane
        MOE_OVERLAY_ACTIVATION_RETURN_PACK, ///< Device pack of follower output into a mapped lane
        MOE_OVERLAY_ACTIVATION_RETURN_CONSUME, ///< Device deterministic fold of one mapped return lane
        MOE_EXPERT_DISPATCH,        ///< Routed-row dispatch descriptor builder
        MOE_SPARSE_DISPATCH,        ///< Graph-native sparse MoE payload dispatch
        MOE_RANK_BATCH_DISPATCH,   ///< One sparse dispatch envelope for all participants on a remote rank
        MOE_LOCAL_EXPERT,           ///< Participant-local sparse MoE expert compute
        MOE_LOCAL_EXPERT_INPUT_PUBLISH, ///< Captured pinned-host packet publication into participant tensors
        MOE_LOCAL_EXPERT_OUTPUT_PUBLISH, ///< Captured participant output publication into pinned host storage
        MOE_LOCAL_EXPERT_COMPLETION, ///< Explicit completion of one submitted remote GPU expert packet
        MOE_SPARSE_RETURN_REDUCE,   ///< Graph-native sparse MoE return reduce
        MOE_RANK_BATCH_RETURN_REDUCE, ///< One sparse return envelope for all participants on a remote rank
        MOE_DEVICE_REBALANCE,       ///< Graph-captured device-side MoE rebalance publish/apply
        MOE_GPU_CURRENT_BATCH_LLEP, ///< Explicit GPU LLEP plan/sideband/apply transaction phase
        MOE_DEVICE_DECODE_COMMIT_BOUNDARY, ///< Captured HIP serial-decode cadence publish/ack
        MOE_CPU_CURRENT_BATCH_LLEP, ///< CPU transient LLEP plan/transfer/restore transaction

        // Collective
        ALLREDUCE,
        ROOTED_COLLECTIVE, ///< LocalTP reduce-to-root or root broadcast
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
        TP_KV_CACHE_STATE_ALLGATHER, ///< Gather TP-local prefill K/V rows into full replicated decode KV rows
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
        GDN_LIVE_STATE_LOCALIZE,  ///< Slice mirrored GDN state into TP-local verifier state
        GDN_LIVE_STATE_ALLGATHER, ///< Gather TP-local GDN state into mirrored decode state

        /** Captured device publication of the current long-prefill request bucket. */
        PREFILL_CHUNK_MATERIALIZATION,

        // Qwen 3.5 FA-specific
        Q_GATE_SPLIT, ///< Split interleaved Q+gate GEMM output into separate buffers

        // MTP sidecar
        MTP_CONCAT, ///< Concatenate normalized draft embedding and terminal hidden rows

        /**
         * Captured proposal transaction: optional serial-equivalent branch
         * penalties followed by deterministic device-slot argmax publication.
         */
        MTP_DRAFT_TOKEN_PUBLICATION,
        MOE_OVERLAY_EPOCH_BOUNDARY,

        /**
         * Captured verifier prelude: resident token composition, device geometry,
         * base-count snapshot, and opaque pre-verifier KV checkpoint capture.
         */
        MTP_VERIFIER_PREPARATION,

        /**
         * Captured target-verifier transaction: optional device-history
         * penalties followed by compact top-k/top-p row construction.
         */
        MTP_STOCHASTIC_TARGET_DISTRIBUTION,

        /**
         * Terminal grouped-verifier transaction: device argmax/distribution
         * reduction plus optional mirrored LocalTP outcome publication.
         */
        MTP_VERIFIER_OUTCOME,

        /**
         * Captured seeded stochastic transaction: sample compact target rows
         * and reduce them against the materialized verifier input sequence.
         */
        MTP_STOCHASTIC_SERIAL_OUTCOME,

        /**
         * Captured accepted-state transaction: response/controller commit,
         * main and shifted KV publication, MoE history, penalty history, and
         * recurrent verifier-row restoration.
         */
        MTP_SPEC_STATE_PUBLICATION,
    };

    /**
     * @brief Identify stage types whose execution includes a domain collective.
     *
     * Capture policy, graph partitioning, fast scheduling, and profiling must
     * agree on this classification. Keeping the mapping beside the enum avoids
     * duplicated local lists that can silently omit a specialized collective
     * such as TP KV-state or GDN-state all-gather.
     *
     * @param type Stage operation type.
     * @return true when the stage necessarily participates in a collective.
     */
    [[nodiscard]] constexpr bool isCollectiveComputeStageType(
        ComputeStageType type) noexcept
    {
        switch (type)
        {
        case ComputeStageType::ALLREDUCE:
        case ComputeStageType::ROOTED_COLLECTIVE:
        case ComputeStageType::ALLGATHER:
        case ComputeStageType::ALLGATHER_V:
        case ComputeStageType::TP_KV_CACHE_STATE_ALLGATHER:
        case ComputeStageType::GDN_LIVE_STATE_ALLGATHER:
        case ComputeStageType::FUSED_ADD_ALLREDUCE:
            return true;
        default:
            return false;
        }
    }

    /**
     * @brief Convert stage type to string for logging
     */
    const char *computeStageTypeName(ComputeStageType type);

    class IComputeStage;

    /**
     * @brief Immutable GPU execution authority for one bound compute stage.
     *
     * A GPU write is correctly published only when its completion event is
     * recorded on the exact stream that enqueued the write. Passing a raw
     * `DeviceId` and `void *` independently to launch and publication APIs made
     * it possible for those values to drift apart while remaining non-null.
     *
     * This token binds the stage's authoritative device and executor-selected
     * stream into one value. Its constructor is private, so production code
     * cannot manufacture a token from ambient tensor state or a backend default
     * stream. Stages use the same value to bind kernels, obtain the native
     * launch stream, prepare tensor inputs and outputs, and publish resulting
     * tensor writes.
     */
    class StageGPUExecution final
    {
    public:
        /**
         * @brief Return the GPU selected by the graph executor for this stage.
         */
        [[nodiscard]] DeviceId device() const noexcept { return device_; }

        /**
         * @brief Return the exact non-null native stream for backend launches.
         *
         * The returned opaque pointer is a `cudaStream_t` or `hipStream_t`
         * according to device(). It is never a default-stream sentinel.
         */
        [[nodiscard]] void *nativeStream() const noexcept { return stream_; }

        /**
         * @brief Bind a tensor kernel to this execution's producer stream.
         *
         * Returning the kernel preserves the established fluent stage setup
         * pattern while ensuring binding and later publication are sourced from
         * the same immutable token.
         */
        template <typename KernelT>
        KernelT *bind(KernelT *kernel) const
        {
            if (kernel)
                kernel->setGPUStream(stream_);
            return kernel;
        }

        /**
         * @brief Join an input tensor to this execution's consumer stream.
         *
         * This placement-capable operation is reserved for the explicit
         * heterogeneous host-packet transport stage. Ordinary GPU graph stages
         * declare arena inputs and use requirePreparedInput(), which cannot
         * allocate or upload. No free device or stream arguments are exposed.
         */
        void prepareInput(ITensor *tensor) const;

        /**
         * @brief Prepare output storage for this execution's producer stream.
         *
         * This operation carries the same immutable device/stream identity as
         * bind() and publish(), preventing output preparation from drifting to
         * a backend default or an unrelated ambient stream.
         */
        void prepareOutput(ITensor *tensor) const;

        /**
         * @brief Require an executor-prepared input without moving or allocating it.
         *
         * Graph stages call this after DeviceGraphExecutor has applied their
         * StageBufferContract. Unlike prepareInput(), this method cannot upload
         * host bytes, allocate device storage, or change tensor coherence. It
         * joins an existing producer event to nativeStream() and fails
         * immediately when the declared input is not valid on the exact device
         * bound into this execution token.
         *
         * @param tensor Declared stage input that the executor prepared.
         * @throws std::invalid_argument when @p tensor is null.
         * @throws std::runtime_error when the tensor is not valid on device().
         */
        void requirePreparedInput(ITensor *tensor) const;

        /**
         * @brief Require pre-existing output storage on the executor-bound GPU.
         *
         * Output bytes need not be valid before a producer overwrites them, but
         * the storage must already exist on the exact stage device. This method
         * never calls an allocator or imports host contents.
         *
         * @param tensor Declared stage output whose storage was prepared.
         * @throws std::invalid_argument when @p tensor is null.
         * @throws std::runtime_error when exact-device storage is absent.
         */
        void requirePreparedOutput(ITensor *tensor) const;

        /**
         * @brief Publish a tensor written by work enqueued through this token.
         *
         * During eager execution, TransferEngine records the completion event
         * on nativeStream() and makes the producer device authoritative. During
         * graph capture, recording is not execution: publication validates and
         * records the internal producer edge in the capture dependency ledger,
         * while the graph-launch boundary publishes live tensor authority after
         * replay. Callers must therefore treat successful return as the complete
         * publication contract and must never inspect or mutate raw tensor
         * coherence afterward.
         *
         * No device or stream argument is accepted here, so a stage cannot
         * publish the write against a different execution context by accident.
         *
         * @throws std::invalid_argument for a null tensor.
         * @throws std::runtime_error if completion-event publication fails.
         */
        void publish(ITensor *tensor) const;

    private:
        friend class IComputeStage;

        StageGPUExecution(DeviceId device, void *stream);

        DeviceId device_;
        void *stream_;
    };

    /**
     * @brief Base class for all compute stages
     *
     * Derived classes implement device-specific kernels while maintaining
     * a common interface for orchestration.
     */
    /**
     * @brief Declares when a stage needs work outside its captured graph body.
     *
     * Graph launch preparation is deliberately a typed lifecycle contract. A
     * capture-only stage may initialize persistent streams, import an existing
     * producer event, or bind a stable device pointer before native capture.
     * Once captured, that work is represented by the graph and does not have to
     * be repeated by its launcher. A capture-and-replay stage, by contrast,
     * publishes mutable metadata from outside the graph before every replay and
     * therefore cannot be cloned into a fully device-controlled parent loop.
     */
    enum class GraphLaunchPreparationPolicy : uint8_t
    {
        None,
        CaptureOnly,
        CaptureAndReplay,
    };

    /**
     * @brief Identifies the lifecycle boundary requesting graph preparation.
     */
    enum class GraphLaunchPreparationPhase : uint8_t
    {
        Capture,
        Replay,
    };

    /**
     * @brief Lifetime over which one successful prepared-weight proof remains valid.
     *
     * PerExecution is the conservative default for stages whose engine bindings
     * can change without another typed authority validating the replacement.
     * StageLifetime may be selected only when construction/publication owns an
     * immutable prepared representation and every mutable runtime bank performs
     * its own generation/epoch proof before use.
     */
    enum class PreparedWeightValidationLifetime : uint8_t
    {
        PerExecution = 0, ///< Revalidate prepared bindings before every stage invocation.
        StageLifetime,   ///< One successful setup proof covers this stage object's lifetime.
    };

    /**
     * @brief Return whether @p policy requires work at @p phase.
     */
    [[nodiscard]] constexpr bool requiresGraphLaunchPreparation(
        GraphLaunchPreparationPolicy policy,
        GraphLaunchPreparationPhase phase) noexcept
    {
        return policy == GraphLaunchPreparationPolicy::CaptureAndReplay ||
               (policy == GraphLaunchPreparationPolicy::CaptureOnly &&
                phase == GraphLaunchPreparationPhase::Capture);
    }

    /**
     * @brief Outcome of planning one CPU verifier-state snapshot commit.
     *
     * CPU MTP publication validates every recurrent-state copy before any live
     * state changes.  The publisher can then execute all independent layer
     * copies in one OpenMP team instead of serializing one `memcpy` per stage.
     * A typed status keeps an unsupported stage distinct from a valid no-op
     * request whose accepted row is negative.
     */
    enum class CPUVerifierStateRestorePlanStatus : uint8_t
    {
        Unsupported,
        NoOp,
        Ready,
        Invalid,
    };

    /**
     * @brief Immutable byte-copy plan for one CPU-owned verifier-state stage.
     *
     * `source` points at the selected post-verifier snapshot and `destination`
     * points at that stage's live CPU state. Both allocations remain owned by
     * the stage/workspace that produced the plan and must stay valid until the
     * enclosing publication transaction finishes. Publication copies these
     * bytes verbatim; it never replays recurrence math.
     */
    struct CPUVerifierStateRestorePlan
    {
        CPUVerifierStateRestorePlanStatus status =
            CPUVerifierStateRestorePlanStatus::Unsupported;
        void *destination = nullptr;
        const void *source = nullptr;
        size_t bytes = 0;

        /** @brief Return true when this plan describes a concrete byte copy. */
        [[nodiscard]] bool ready() const noexcept
        {
            return status == CPUVerifierStateRestorePlanStatus::Ready &&
                   destination != nullptr && source != nullptr && bytes > 0;
        }
    };

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
         *
         * `token_offset` is the absolute logical request offset for the first
         * real row in this replay.  It must stay synchronized with position-id
         * generation and restored-prefix suffix prefill so stateful stages do
         * not accidentally append or interpret rows at prompt offset zero.
         */
        struct PrefillReplayParams
        {
            int real_seq_len = 0;   ///< Real, non-padding token count in this replay.
            int bucket_seq_len = 0; ///< Fixed graph execution length for this replay.
            int token_offset = 0;   ///< Absolute offset of the first real replay token.
        };

        /**
         * @brief Root-authoritative identity for one graph-native MoE collective transaction.
         *
         * Graph capture can materialize the continuation and expert-only
         * participant graphs at different times.  A stage-local execution
         * counter therefore cannot name a distributed transaction: capture,
         * recapture, and graph-cache eviction would make independently built
         * stages disagree even though they are servicing the same request
         * chunk.  The runner stamps this immutable pair before graph
         * execution, and every sparse dispatch/return stage uses it to derive
         * its wire key.
         *
         * The values are host-side control metadata for explicit manual
         * sparse-collective boundaries. They do not create a host mirror of a
         * device tensor and they do not change captured GPU topology.
         */
        struct MoEOverlayCollectiveRuntimeParams
        {
            /** Mathematical phase carried across an explicit sparse boundary. */
            enum class ExecutionSemantics : uint8_t
            {
                Unspecified,
                Decode,
                Prefill,
                MTPDraft,
                GroupedVerifier,
            };

            uint64_t generation_id = 0; ///< Monotonic request-generation authority; zero is invalid.
            uint64_t step_id = 0;       ///< Absolute logical operation offset within that generation.
            /** Typed phase; a one-row prefill remains Prefill. */
            ExecutionSemantics execution_semantics =
                ExecutionSemantics::Unspecified;
            /**
             * MTP graph namespace depth selected by the controller.
             *
             * Main decode/prefill use -1. A NextN sidecar uses its declared
             * sidecar graph depth (currently zero), while a grouped verifier
             * uses the admitted speculative draft depth. This value is never
             * inferred from a physical graph bucket.
             */
            int mtp_depth = -1;

            /** @brief Return true when the runner supplied a usable protocol identity. */
            [[nodiscard]] bool valid() const noexcept { return generation_id != 0; }
            /** @return Whether the mathematical phase was supplied explicitly. */
            [[nodiscard]] bool hasExecutionSemantics() const noexcept
            {
                return execution_semantics !=
                       ExecutionSemantics::Unspecified;
            }
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
         * @brief Report whether executing this stage participates in a collective.
         *
         * Stages with unconditional collective semantics inherit the canonical
         * enum mapping. A future stage whose collective behavior depends on its
         * parameters can override this method, keeping capture policy tied to
         * the actual stage instance rather than another orchestration-side list.
         */
        virtual bool isCollectiveStage() const
        {
            return isCollectiveComputeStageType(type());
        }

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
            std::lock_guard<std::mutex> lock(dump_info_mutex_);
            if (!dump_info_cached_)
            {
                cached_dump_info_ = buildDumpInfoImpl();
                dump_info_cached_ = true;
            }
            return cached_dump_info_;
        }

        /**
         * @brief Return a stable copy of cached dump info.
         *
         * Use this in executor/debug paths that pass StageDumpInfo across
         * callbacks, stream waits, async dump queues, or other code that should
         * not observe a concurrent cache refresh.
         */
        StageDumpInfo getDumpInfoSnapshot() const
        {
            std::lock_guard<std::mutex> lock(dump_info_mutex_);
            if (!dump_info_cached_)
            {
                cached_dump_info_ = buildDumpInfoImpl();
                dump_info_cached_ = true;
            }
            return cached_dump_info_;
        }

        /**
         * @brief Rebuild dump info under the cache lock and return a stable copy.
         *
         * Post-execute debug consumers use this when stages may have populated
         * outputs or diagnostic tensors during execute().
         */
        StageDumpInfo refreshDumpInfoSnapshot() const
        {
            std::lock_guard<std::mutex> lock(dump_info_mutex_);
            cached_dump_info_ = buildDumpInfoImpl();
            dump_info_cached_ = true;
            return cached_dump_info_;
        }

        /**
         * @brief Invalidate cached dump info (for dynamic reconfiguration)
         *
         * Call this if stage parameters change after construction (rare).
         */
        void invalidateDumpInfoCache() const
        {
            std::lock_guard<std::mutex> lock(dump_info_mutex_);
            dump_info_cached_ = false;
        }

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
         *   2. For each weight tensor: TransferEngine prepares the exact
         *      device allocation on the stage's explicit stream
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
         *   - updateDynamicParams remains orthogonal
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
         * @brief Whether this immutable graph role performs no execution work.
         *
         * Returning true permits a graph-capture wave contract to mark this
         * node as a passive participant. The stage must then be a complete
         * no-op for its graph-bound role: execute() may not enqueue device
         * work, mutate host state, publish coherence, or allocate storage.
         * Generic orchestration uses this opt-in to omit the node from the
         * local executable while still joining a sibling's begin/end capture
         * rendezvous. Shape-dependent or runtime-dependent no-op decisions
         * must return false because capture topology cannot depend on live
         * tensor values.
         */
        virtual bool isPassiveGraphCaptureNoOp() const { return false; }

        /**
         * @brief Variant signature for graph-captured launch topology.
         *
         * Most stages have a single stable graph-capture topology and return 0.
         * Stages whose kernel launch shape is bucketed by runtime values may
         * return a nonzero signature. Segmented graph replay uses this to
         * intentionally warm/capture a new graph variant before replaying a
         * graph whose baked launch topology no longer matches the current step.
         *
         * This is deliberately about launch topology, not mathematical inputs:
         * dynamic scalar values that are read from device-side params should not
         * change this signature unless they also alter grid/block/smem shape.
         */
        virtual uint64_t graphCaptureVariantSignature() const { return 0; }

        /**
         * @brief Declare how long validatePreparedWeights() remains authoritative.
         *
         * The default deliberately repeats validation. A stage opting into
         * StageLifetime must document the immutable owner that prevents its
         * prepared bindings from changing behind a retained execution plan.
         */
        virtual PreparedWeightValidationLifetime
        preparedWeightValidationLifetime() const noexcept
        {
            return PreparedWeightValidationLifetime::PerExecution;
        }

        /**
         * @brief Whether explicit launch preparation can make this stage capture-ready.
         *
         * A stage returns true when its immutable graph topology is supported but
         * isGraphCapturable() remains false until prepareGraphLaunch() binds
         * persistent kernels, descriptor tables, scratch pointers, or events.
         * Preparation must never execute model arithmetic. The planner may admit
         * such a stage provisionally, then requires preparation to make the
         * stricter isGraphCapturable() predicate true before beginCapture().
         */
        virtual bool supportsGraphCaptureAfterLaunchPreparation() const
        {
            return false;
        }

        /**
         * @brief Human-readable readiness details when launch preparation fails.
         *
         * Returned text is diagnostic only. It must not allocate device memory or
         * mutate stage state because callers invoke it on fatal capture failures.
         */
        virtual std::string graphCaptureReadinessDebugString() const { return {}; }

        /**
         * @brief Whether a capturable stage must start a fresh graph segment.
         *
         * Return true for stages that are graph-capturable on their own but
         * cannot safely be fused after earlier captured work on a backend.
         */
        virtual bool requiresGraphCaptureSegmentBoundaryBefore() const { return false; }

        /**
         * @brief Whether a capturable stage must terminate its graph segment.
         *
         * Most capturable stages can be coalesced freely. Some stages are
         * graph-capturable on their own but carry backend replay state, captured
         * H2D parameter nodes, or callbacks that make fusing a following stage
         * into the same GPU graph unsafe on a backend. Such stages should return
         * true here so the planner starts a fresh segment for the next stage
         * while still graph-capturing this stage.
         */
        virtual bool requiresGraphCaptureSegmentBoundaryAfter() const { return false; }

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
         * @brief Whether host execution must observe the preceding captured ticket.
         *
         * A heterogeneous graph may end a captured device segment with an
         * asynchronous D2H copy into fixed pinned storage.  The first manual
         * consumer of that storage declares this contract so the replay
         * controller records and waits for one preallocated completion event
         * at the explicit device/host boundary.  This is not a general manual
         * stage synchronization switch and is invalid unless
         * @ref isManualGraphBoundary also returns true.
         */
        virtual bool requiresHostGraphTicketFence() const { return false; }

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
         * @brief True when missing verifier state capture makes publication unsafe.
         *
         * Some stages own mutable recurrent state that must be restored from
         * an accepted verifier row for MTP state publication to be
         * decode-equivalent.  Returning true here turns a missing capture slot
         * into a hard publication error when the caller requires captured
         * stage state, instead of silently skipping the stage and allowing a
         * later continuation token to drift.
         */
        virtual bool requiresVerifierStateCaptureForPublication() const { return false; }

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
         * @brief Restore one host-selected verifier row per CPU request.
         *
         * `host_row_indices[request]` contains a flat verifier snapshot row;
         * negative values leave that request's live state unchanged.  Native
         * CPU grouped stages must restore the complete vector in one call so
         * scalar publication cannot overwrite one shared layer state or clear
         * capture bindings between requests.
         */
        virtual bool restoreVerifierStateCaptureRows(
            const int *host_row_indices,
            int request_count,
            void *stream = nullptr)
        {
            (void)host_row_indices;
            (void)request_count;
            (void)stream;
            return false;
        }

        /**
         * @brief Plan a byte-exact single-request CPU state publication.
         *
         * The method validates @p row and exposes the already-computed snapshot
         * span without mutating live state. The MTP publisher first gathers and
         * validates plans from every captured stage, then copies all `Ready`
         * spans in one persistent OpenMP region. This two-phase contract makes
         * malformed publication atomic and removes serial per-layer memory
         * bandwidth from the decode hot path.
         *
         * A negative row is a valid `NoOp`: that request accepted no verifier
         * row and retains its pre-transaction live state. GPU stages and CPU
         * stages without native snapshot spans return `Unsupported`; a required
         * captured CPU stage returning that status is a fatal publication
         * contract violation, not permission to call a slower path.
         *
         * @param row Flat verifier snapshot row selected for the request.
         * @return Typed immutable restore plan whose storage remains stage-owned.
         */
        virtual CPUVerifierStateRestorePlan planCPUVerifierStateRestoreRow(int row)
        {
            (void)row;
            return {};
        }

        /**
         * @brief Restore mutable model state captured after a verifier row chosen on device.
         *
         * Device-resident stochastic MTP publication receives accepted-count
         * metadata in GPU memory.  Stages that implement this method must read
         * @p device_row_index on @p stream and restore the corresponding
         * captured row without synchronizing that index or mutable stage state
         * to the host. This is the device-owned hot path used by resident MTP
         * publication; host mirror refresh must be a separate explicit action.
         */
        virtual bool restoreVerifierStateCaptureRowFromDeviceIndex(
            const int *device_row_index,
            void *stream)
        {
            (void)device_row_index;
            (void)stream;
            return false;
        }

        /**
         * @brief Restore one captured verifier row for each request in a batch.
         *
         * Batched device-resident MTP publication receives a device array of
         * accepted verifier rows. Implementations that override this method
         * must read `device_row_indices[request * row_index_stride]` on
         * @p stream and restore that request's mutable model state into a
         * request-owned live-state slot. A negative row index means that
         * request accepted no verifier row and its live state must be left at
         * the pre-verifier value. This is not equivalent to looping
         * restoreVerifierStateCaptureRowFromDeviceIndex(): scalar restore would
         * overwrite one layer-owned state buffer and leak state between
         * requests.
         *
         * The default is a hard failure so backends cannot quietly opt into
         * request batching before their capture layout and live-state ownership
         * are request-aware.
         */
        virtual bool restoreVerifierStateCaptureRowsFromDeviceIndices(
            const int *device_row_indices,
            int request_count,
            int row_index_stride,
            void *stream)
        {
            (void)device_row_indices;
            (void)request_count;
            (void)row_index_stride;
            (void)stream;
            return false;
        }

        /**
         * @brief Publish request-local terminal states from device real lengths.
         *
         * Padded request-batched prefill owns one captured state row per flat
         * `(request,row)` coordinate. Implementations derive each terminal row
         * as `request * request_row_width + real_length - 1` on @p stream and
         * restore all request-owned live states without exposing row indices to
         * the host. The default is a hard failure.
         */
        virtual bool restoreVerifierStateCaptureRequestTerminalRows(
            const int *device_request_seq_lens,
            int request_count,
            int request_row_width,
            void *stream)
        {
            (void)device_request_seq_lens;
            (void)request_count;
            (void)request_row_width;
            (void)stream;
            return false;
        }

        /**
         * @brief Report that grouped request execution already committed live state.
         *
         * Long request-batched prefill may reserve only a small MTP verifier
         * snapshot window. When that window cannot cover the complete flattened
         * request matrix, conforming grouped kernels execute directly against
         * request-owned live state banks and need no post-graph restore. Stages
         * must return true only for that exact backend policy and geometry.
         */
        virtual bool requestBatchedTerminalStateCommittedDuringExecution(
            int request_count,
            int request_row_width) const
        {
            (void)request_count;
            (void)request_row_width;
            return false;
        }

        /**
         * @brief Detach verifier-row scratch from a shared kernel after publication.
         *
         * Some GPU recurrent kernels are shared between the all-position verifier
         * graph and the ordinary one-token decode graph.  The verifier graph binds
         * speculative capture/work buffers so it can snapshot every candidate row
         * without mutating live state.  After accepted-state publication restores
         * the chosen row into live device state, the next ordinary decode must no
         * longer see those speculative buffers as active.  Stages that multiplex a
         * shared backend kernel override this hook to leave the live state resident
         * while clearing only the verifier-capture binding.
         */
        virtual void clearVerifierStateCaptureBindingAfterPublication() {}

        /**
         * @brief True when this stage must publish derived live state after verifier-row restore.
         *
         * Not every live-state mutation owns verifier capture slots directly.  A
         * graph may restore TP-local recurrent state from captured rows and then
         * run a later handoff stage that derives the mirrored decode state consumed
         * by the next token.  Such stages return true here so the MTP publisher
         * invokes publishPostVerifierStateRestore() in graph order instead of
         * treating the stage as an irrelevant non-capturing node.
         *
         * This hook is intentionally separate from hasVerifierStateCapture(): it
         * represents derived publication from already-restored state, not another
         * verifier row snapshot owner.
         */
        virtual bool requiresPostVerifierStatePublication() const { return false; }

        /**
         * @brief Publish derived live state after accepted verifier rows are restored.
         *
         * The MTP publisher calls this on the same stream used for row restoration
         * and before the runner records publication readiness.  Implementations
         * must enqueue only deterministic state handoffs derived from the accepted
         * restored state; they must not perform host fallback replay or mutate
         * unrelated graph outputs.
         *
         * @param stream Explicit GPU stream for GPU stages, or nullptr for CPU-only stages.
         * @return true when the derived live state was published successfully.
         */
        virtual bool publishPostVerifierStateRestore(void *stream = nullptr)
        {
            (void)stream;
            return true;
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
         * cudaStream_t (CUDA) as needed. New GPU stages must treat nullptr as
         * "no explicit stream was assigned" and fail before launching GPU work
         * that requires stream ordering; the device-default stream is not a
         * valid fallback for graph-capturable execution.
         *
         * @param stream Opaque GPU stream pointer (hipStream_t / cudaStream_t as void*)
         */
        void setGPUStream(void *stream)
        {
            if (device_id_.is_gpu() && !stream)
            {
                throw std::invalid_argument(
                    "IComputeStage::setGPUStream refuses a null stream for GPU stage on " +
                    device_id_.toString());
            }
            gpu_stream_ = stream;
        }

        /**
         * @brief Report whether the executor has bound an explicit GPU stream.
         *
         * This is the only non-throwing stream-state query. Schedulers, graph
         * binders, and hardware-free tests may use it before execution to decide
         * whether a stage still needs a binding. Compute code must use
         * gpuStream() or requireGPUStream() so a missing GPU stream cannot flow
         * into a kernel launch, transfer, or completion publication as nullptr.
         *
         * @return true when an explicit stream is currently bound.
         */
        bool hasGPUStream() const noexcept { return gpu_stream_ != nullptr; }

        /**
         * @brief Test whether two GPU stages share one exact execution stream.
         *
         * Paired producer/completion nodes use this typed relation instead of
         * retrieving two opaque pointers and accidentally treating null as a
         * valid shared stream. CPU stages and either unbound GPU stage always
         * return false.
         */
        bool sharesExactGPUStreamWith(
            const IComputeStage &other) const noexcept
        {
            return device_id_.is_gpu() && other.device_id_.is_gpu() &&
                   gpu_stream_ && other.gpu_stream_ &&
                   gpu_stream_ == other.gpu_stream_;
        }

        /**
         * @brief Get the stream used by this stage's current execution.
         *
         * CPU stages have no GPU stream and return nullptr. GPU stages must have
         * been bound by the executor; retrieving an unbound GPU stream is a
         * fatal programming error. This contract deliberately makes the CUDA or
         * HIP default stream unavailable as an implicit fallback.
         *
         * @return Exact GPU stream, or nullptr only for a CPU stage.
         * @throws std::logic_error when a GPU stage has not been bound.
         */
        void *gpuStream() const
        {
            if (device_id_.is_gpu() && !gpu_stream_)
            {
                throw std::logic_error(
                    "IComputeStage::gpuStream found no explicit stream for GPU stage on " +
                    device_id_.toString());
            }
            return gpu_stream_;
        }

        /**
         * @brief Return the exact stream bound to this GPU stage.
         *
         * GPU execution and publication code may use this accessor when it also
         * wants to reject accidental use from a CPU stage. Both this method and
         * gpuStream() reject an unbound GPU stage; requireGPUStream() additionally
         * rejects CPU callers.
         *
         * @return Exact non-null stream assigned by the graph executor.
         * @throws std::logic_error when called for a CPU stage or before the
         *         executor has bound the GPU stage to its execution stream.
         */
        void *requireGPUStream() const
        {
            if (!device_id_.is_gpu())
            {
                throw std::logic_error(
                    "IComputeStage::requireGPUStream called for non-GPU stage on " +
                    device_id_.toString());
            }
            if (!gpu_stream_)
            {
                throw std::logic_error(
                    "IComputeStage::requireGPUStream found no explicit stream for GPU stage on " +
                    device_id_.toString());
            }
            return gpu_stream_;
        }

        /**
         * @brief Return the immutable execution authority for this GPU stage.
         *
         * New stage code should retain this value for the duration of an
         * execution method and use it for both kernel binding/launch and output
         * publication. Unlike separate calls to device() and gpuStream(), the
         * token cannot be assembled from unrelated values.
         *
         * @throws std::logic_error for CPU stages or an unbound GPU stage.
         */
        [[nodiscard]] StageGPUExecution gpuExecution() const
        {
            return StageGPUExecution(device_id_, requireGPUStream());
        }

        /**
         * @brief Publish a stage output when this stage executes on a GPU.
         *
         * CPU kernels write host-authoritative tensor storage directly and
         * therefore require no device publication. GPU kernels, by contrast,
         * must publish through the executor-bound stream so downstream
         * consumers inherit the exact producer event. Keeping that distinction
         * in this shared helper prevents individual dual-backend stages from
         * accidentally constructing a GPU execution token on their CPU path.
         *
         * This is intentionally not a permissive GPU fallback: a GPU stage
         * without an executor-bound stream still fails through gpuExecution().
         *
         * @param tensor Tensor written by the stage kernel.
         */
        void publishStageOutput(ITensor *tensor) const
        {
            if (device_id_.is_gpu())
                gpuExecution().publish(tensor);
        }

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
         * @brief Refresh explicit host position rows before cached graph replay.
         *
         * The forward graph cache reuses stage objects while token and position
         * inputs change every decode step.  Most stages ignore explicit
         * position rows; RoPE consumes them to preserve request-batched or
         * otherwise non-contiguous absolute positions without rebuilding the
         * graph.
         */
        virtual void updateDynamicPositionIds(const int *position_ids, int seq_len)
        {
            (void)position_ids;
            (void)seq_len;
        }

        /**
         * @brief Refresh device-resident position rows before cached graph replay.
         *
         * GPU MTP publication can keep logical continuation positions in device
         * workspace memory.  Stages that understand this contract must bind the
         * pointer on their explicit graph stream and must not copy it through
         * host memory.  The default implementation is a no-op for stages that
         * do not consume RoPE-style positions.
         */
        virtual void updateDynamicDevicePositionIds(const void *position_ids_device, int seq_len)
        {
            (void)position_ids_device;
            (void)seq_len;
        }

        /**
         * @brief Return whether replay can use device-resident position rows without a host scalar.
         *
         * Phase 10 MTP publication keeps accepted logical positions in a
         * runner-owned device mailbox.  A stage that returns true here promises
         * that, after updateDynamicDevicePositionIds() is called, any later
         * updateDynamicParams() call either ignores its scalar position
         * argument or derives equivalent metadata from device-resident state on
         * the explicit stage stream.  Stages that still need a host position
         * must keep the default false result so resident replay hard-fails
         * instead of silently reading stale host shadows.
         */
        virtual bool supportsDeviceResidentDynamicPositionReplay() const
        {
            return false;
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
         * @brief Reset request-scoped state while preserving captured replay metadata.
         *
         * This hook is used only when the caller keeps an already instantiated
         * GPU graph executable alive across a request boundary. Derived stages
         * must clear request-local host bookkeeping without invalidating stream
         * ownership, descriptor tables, pointer slots, or other persistent
         * device metadata read by the preserved graph. A request reset must not
         * force eager initialization or recapture.
         */
        virtual void resetSessionStatePreservingCapturedReplay()
        {
            resetSessionState();
        }

        /**
         * @brief Reset request-scoped state while preserving lazy initialization.
         *
         * This is used for prefill buckets that warmed lazy kernels, descriptor
         * banks, or workspace-backed helper allocations, but did not produce a
         * graph executable before a request boundary. Implementations should
         * clear host mirrors, dynamic stream ownership, per-request scalar
         * values, and capture-arming flags while keeping model-weight
         * preparations and backend objects that are safe to reuse before a
         * fresh strict capture-readiness preflight.
         */
        virtual void resetSessionStatePreservingLazyInitialization()
        {
            resetSessionState();
        }

        /**
         * @brief Invalidate handles into backend-owned kernel-dynamic state.
         *
         * Kernel-dynamic state is separate from request-scoped model state and
         * separate from immutable model weights. It includes backend-owned
         * pointer tables, descriptor table IDs, dynamic argument buffers, and
         * other launch metadata that kernels populate lazily for eager or graph
         * replay execution. When the orchestrator calls
         * KernelFactory::resetAllDynamicState(), cached ComputeGraphs may keep
         * their stage objects, but any stage-local handles into the reset kernel
         * metadata must be treated as stale.
         *
         * Implementations must not clear KV/GDN/MTP model state, graph topology,
         * tensor bindings, workspace ownership, or prepared model weights here.
         * They should only mark kernel-dynamic handles cold so the next eager
         * warmup can rebuild them, and they should fail hard if asked to rebuild
         * while GPU graph capture is already active.
         */
        virtual void invalidateKernelDynamicState() {}

        /**
         * @brief Update prefill replay bookkeeping before a captured graph launch.
         *
         * The executor calls this on cached prefill graph hits before normal
         * dynamic params are refreshed. Decode graph replay continues to use
         * updateDynamicParams() only. Stages should ignore this unless their
         * dynamic device metadata must distinguish real tokens from padded
         * bucket rows.
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
         * @brief Stamp the next graph-native MoE sparse-collective transaction.
         *
         * The forward engine invokes this before either a cold graph execution
         * or a cached graph replay. Implementations must retain only this
         * scalar protocol identity; they must not allocate, synchronize, or
         * upload request state from this hook.
         *
         * @param params Root-authoritative generation and logical step.
         */
        virtual void updateMoEOverlayCollectiveRuntimeParams(
            const MoEOverlayCollectiveRuntimeParams &params)
        {
            (void)params;
        }

        /**
         * @brief Return true when this stage requires explicit MoE protocol identity.
         *
         * The forward graph cache uses this opt-in query to update only sparse
         * collective stages rather than scanning unrelated model operations on
         * every execution.
         */
        virtual bool hasMoEOverlayCollectiveRuntimeParams() const { return false; }

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
         * @brief Whether cold prefill graph preflight may allow this stage.
         *
         * Cold preflight runs before the first normal warmup pass, while some
         * stages intentionally allocate backend state, descriptor tables, or
         * scratch buffers during that warmup. Such stages should return true
         * here when their backend and shape support prefill capture in
         * principle, and keep isGraphCapturable() as the stricter capture-time
         * readiness check.
         *
         * The default preserves legacy behavior for existing stages: if a stage
         * has no separate cold-support contract, preflight still requires normal
         * graph-capture readiness.
         */
        virtual bool supportsLazyPrefillGraphCapturePreflight() const { return isGraphCapturable(); }

        /**
         * @brief Whether cold padded-prefill graph preflight may allow this stage.
         *
         * Padded buckets also require supportsPaddedPrefillRealLengthContract()
         * to ensure recurrent state commits only the real prompt prefix. Stages
         * may override this when the padded fixed-bucket contract differs from
         * exact-shape prefill support.
         */
        virtual bool supportsPaddedPrefillGraphCapturePreflight() const
        {
            return supportsLazyPrefillGraphCapturePreflight();
        }

        /**
         * @brief Prepare stage state outside a captured graph launch.
         *
         * The executor calls this only at the lifecycle boundaries selected by
         * graphLaunchPreparationPolicy(), after rebinding workspace ownership
         * and assigning an explicit stream. Implementations may publish into
         * persistent workspace or establish event ordering on @p stream, but
         * must not allocate ad-hoc device memory or synchronize the device.
         *
         * @param ctx Device context for the launch.
         * @param stream Explicit backend stream used for the upcoming launch.
         * @return true on success.
         */
        virtual bool prepareGraphLaunch(IDeviceContext *ctx, void *stream)
        {
            (void)ctx;
            if (stream)
                setGPUStream(stream);
            return true;
        }

        /**
         * @brief Declare exactly when prepareGraphLaunch() is required.
         *
         * The default is fully self-contained. Stages that return
         * CaptureAndReplay are ineligible for device-controlled parent graph
         * composition until their mutable launch state becomes device-owned.
         */
        virtual GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const
        {
            return GraphLaunchPreparationPolicy::None;
        }

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
         * Populates weight entries from both raw and prepared contract lists.
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
            for (const auto &prepared : contract.prepared_weights)
            {
                if (prepared.source_tensor)
                    info.addWeight("prepared_weight", prepared.source_tensor);
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
         * @brief Apply this stage's device-specific stream lifecycle to a kernel
         *
         * GPU stages must bind the exact non-null producer stream assigned by
         * the graph executor. CPU stages explicitly clear any stale GPU
         * binding that may remain on a reused prepared kernel. Keeping both
         * transitions behind this boundary prevents callers from representing
         * GPU execution with a nullable raw stream.
         *
         * @tparam KernelT Kernel type exposing the explicit GPU stream lifecycle
         * @param kernel Kernel pointer (may be nullptr)
         * @return The same kernel pointer for fluent usage
         */
        template <typename KernelT>
        KernelT *bindStageStream(KernelT *kernel) const
        {
            if (kernel)
            {
                if (device_id_.is_gpu())
                    gpuExecution().bind(kernel);
                else
                    kernel->clearGPUStreamBinding();
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
        void *gpu_stream_ = nullptr; ///< Explicit GPU stream; null means unbound, never a default stream.

        // Cached dump info (built once, reused for all subsequent calls)
        mutable StageDumpInfo cached_dump_info_;
        mutable bool dump_info_cached_ = false;
        mutable std::mutex dump_info_mutex_;

        static bool shapesMatch(const std::vector<size_t> &actual,
                                const std::vector<size_t> &expected,
                                bool allow_broadcast);
        static std::string shapeToString(const std::vector<size_t> &shape);
    };

} // namespace llaminar2
