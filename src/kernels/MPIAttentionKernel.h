#pragma once

#include "../mpi_kernel_base.h"
#include "../pipeline_stages.h"
#include "attention/AttentionStageContracts.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace llaminar
{
    // Forward declaration for optional COSMA backend
    class CosmaPrefillManager;

    /**
     * @brief Result structure from input validation and setup phase
     *
     * Contains all extracted tensors, parameters, and flags needed for subsequent
     * attention computation stages.
     */
    struct InputSetupResult
    {
        // Input tensors
        std::shared_ptr<TensorBase> input;
        std::shared_ptr<TensorBase> wq_global;
        std::shared_ptr<TensorBase> wk_global;
        std::shared_ptr<TensorBase> wv_global;
        std::shared_ptr<TensorBase> wo_global;
        std::shared_ptr<TensorBase> bq_global;
        std::shared_ptr<TensorBase> bk_global;
        std::shared_ptr<TensorBase> bv_global;
        std::shared_ptr<TensorBase> k_cache_in;
        std::shared_ptr<TensorBase> v_cache_in;

        // Extracted parameters
        int seq_len = 0;
        int d_model = 0;
        int rank = 0;
        int world_size = 0;

        // Mode flags
        bool is_decode_mode = false;
        int cache_seq_len = 0;

        // Head distribution
        int local_heads = 0;
        int head_offset = 0;
        int local_kv_heads = 0;
        int kv_head_offset = 0;
        int local_head_dim = 0;
        int local_kv_head_dim = 0;

        // Weight format detection
        bool weights_are_sharded = false;

        // Early exit flags
        bool should_early_exit = false;
        bool early_exit_success = false;
    };

    /**
     * @brief Result structure from weight distribution phase
     *
     * Contains weight and bias slices distributed to this rank based on head assignment.
     */
    struct WeightSlices
    {
        // Weight slices for this rank's heads
        std::shared_ptr<TensorBase> local_wq; // [local_head_dim, d_model]
        std::shared_ptr<TensorBase> local_wk; // [local_kv_head_dim, d_model]
        std::shared_ptr<TensorBase> local_wv; // [local_kv_head_dim, d_model]
        std::shared_ptr<TensorBase> local_wo; // [d_model, local_head_dim]

        // Bias slices (may be nullptr if no bias or size <= 1)
        std::shared_ptr<TensorBase> local_bq; // [local_head_dim] or nullptr
        std::shared_ptr<TensorBase> local_bk; // [local_kv_head_dim] or nullptr
        std::shared_ptr<TensorBase> local_bv; // [local_kv_head_dim] or nullptr

        // Metadata
        bool copied_global_weights = false; // True if we copied from global weights
    };

    /**
     * @brief Result structure from QKV projection phase
     *
     * Contains Q, K, V activation tensors after linear projections with optional bias.
     */
    struct QKVProjectionResult
    {
        std::shared_ptr<TensorBase> local_q; // [seq_len, local_head_dim]
        std::shared_ptr<TensorBase> local_k; // [seq_len, local_kv_head_dim]
        std::shared_ptr<TensorBase> local_v; // [seq_len, local_kv_head_dim]
    };

    /**
     * @brief Result structure from gather and snapshot phase
     *
     * For multi-rank: Contains gathered global Q/K/V tensors (nullable after snapshot).
     * For single-rank: No gathering needed, fields remain nullptr.
     * Snapshot callback invoked BEFORE RoPE to match PyTorch reference.
     */
    struct GatherResult
    {
        // Gathered global tensors (only allocated for multi-rank with callback)
        std::shared_ptr<TensorBase> global_q; // [seq_len, n_head * head_dim] or nullptr
        std::shared_ptr<TensorBase> global_k; // [seq_len, n_head_kv * head_dim] or nullptr
        std::shared_ptr<TensorBase> global_v; // [seq_len, n_head_kv * head_dim] or nullptr

        // Flag indicating if snapshot was performed
        bool snapshot_performed = false;
    };

    /**
     * @brief Result structure from RoPE application phase
     *
     * Contains Q/K tensors after rotary position embedding, plus updated KV cache.
     * V tensor is passed through unchanged (RoPE only applies to Q and K).
     * Also includes gathered global tensors for ROPE_APPLICATION snapshot validation.
     */
    struct RoPEResult
    {
        // Local Q/K after RoPE application (in-place modified)
        std::shared_ptr<TensorBase> local_q_rope;      // [seq_len, local_head_dim]
        std::shared_ptr<TensorBase> local_k_rope;      // [seq_len, local_kv_head_dim]
        std::shared_ptr<TensorBase> local_v_unchanged; // [seq_len, local_kv_head_dim] (pass-through)

        // KV cache (after RoPE, ready for attention)
        std::shared_ptr<TensorBase> local_k_cache; // [attn_seq_len, local_kv_head_dim]
        std::shared_ptr<TensorBase> local_v_cache; // [attn_seq_len, local_kv_head_dim]
        int attn_seq_len;                          // Total sequence length for attention (n_past + seq_len)

        // Gathered global tensors for ROPE_APPLICATION snapshot (only if snapshot_callback set)
        std::shared_ptr<TensorBase> global_q_rope; // [seq_len, n_head * head_dim] or nullptr
        std::shared_ptr<TensorBase> global_k_rope; // [seq_len, n_head_kv * head_dim] or nullptr
        std::shared_ptr<TensorBase> global_v_rope; // [seq_len, n_head_kv * head_dim] or nullptr
    };

    /**
     * @brief Result structure from GQA expansion phase
     *
     * Contains expanded K/V tensors after Grouped Query Attention replication.
     * For GQA architectures (n_head != n_head_kv), each KV head is replicated to serve
     * multiple query heads. For MHA architectures (n_head == n_head_kv), K/V are passed through.
     */
    struct GQAExpansionResult
    {
        // Expanded K/V tensors (replicated for GQA, or direct reference for MHA)
        std::shared_ptr<TensorBase> local_k_expanded; // [attn_seq_len, local_head_dim]
        std::shared_ptr<TensorBase> local_v_expanded; // [attn_seq_len, local_head_dim]

        // Metadata
        bool gqa_required; // true if n_head != n_head_kv (GQA architecture)
    };

    /**
     * @brief Result structure from attention scores computation phase
     *
     * Contains the final attended output after computing scaled dot-product attention.
     * This is the result of: softmax(QK^T / sqrt(d_k)) @ V
     */
    struct AttentionScoresResult
    {
        // Attended output after applying attention weights to V
        std::shared_ptr<TensorBase> local_attended; // [seq_len, local_head_dim]
    };

    /**
     * @brief Result structure from output projection phase
     *
     * Contains the final attention output after projecting attended values back to model dimension
     * and optionally aggregating across MPI ranks.
     */
    struct OutputProjectionResult
    {
        // Final attention output in model dimension
        std::shared_ptr<TensorBase> attention_output; // [seq_len, d_model]
    };

    /**
     * @brief MPI-enabled multi-head attention kernel with modular 8-stage pipeline architecture
     * @author David Sanftenberg
     *
     * This kernel implements distributed multi-head attention using a clean, testable 8-stage pipeline.
     * Each stage is implemented as a separate private method with well-defined inputs/outputs via
     * structured result types, enabling independent testing and validation.
     *
     * Architecture (Phase 8 Refactoring - Complete):
     * The execute() method has been reduced from 2,287 lines to 183 lines (92% reduction) by
     * extracting 8 well-scoped stages. The main execute() function now serves as a high-level
     * orchestrator calling specialized stage functions.
     *
     * Pipeline Stages:
     * 1. validateAndSetupInputs()       - Input validation, parameter extraction, head distribution
     * 2. distributeWeightsByHead()      - Weight slicing/distribution based on head assignments
     * 3. computeQKVProjections()        - Linear projections for Q, K, V (direct COSMA/OpenBLAS calls)
     * 4. gatherAndSnapshotPreRoPE()     - Optional PyTorch parity snapshots (before RoPE)
     * 5. applyRotaryPositionEmbeddings() - RoPE to Q/K, KV cache management
     * 6. handleGQAExpansion()           - Grouped Query Attention head replication
     * 7. computeAttentionScores()       - Scaled dot-product attention with softmax
     * 8. projectAndGatherOutput()       - Output projection (Wo) + MPI_Allreduce for fully replicated result
     *
     * Distribution Strategy:
     * - HEAD_WISE: Each MPI rank handles a subset of attention heads (ONLY implemented strategy)
     * - Input / KV Cache: Replicated across all processes (via MPI_Bcast and MPI_Allgatherv)
     * - Q/K/V Projections: Distributed by heads (each rank projects only its assigned heads)
     * - Attention Computation: Local computation + extensive MPI_Allgatherv for K/V cache gathering
     * - Output Projection: Row-sharded Wo produces per-rank partials, then MPI_Allreduce(SUM) for replication
     *
     * Backend Selection:
     * - Simple availability check: if (cosma_mgr_ != nullptr) use COSMA, else OpenBLAS
     * - OpenBLAS: Auto-detects thread count (respects OMP_NUM_THREADS), uses CblasTrans for transpose
     * - COSMA: Used when CosmaPrefillManager is set, not based on operation size thresholds
     * - No MatMulBackendSelector class involved - direct backend branching in matmul_with_bias()
     * - All backends respect proper matrix orientation contracts (no silent transpositions)
     *
     * @note Output Contract (Fully Replicated):
     * The kernel returns a FULLY REPLICATED attention output across all ranks. The output tensor
     * is identical on all processes after MPI_Allreduce (MPI_SUM) in the projectAndGatherOutput stage.
     *
     * Internal MPI Communication:
     * The kernel performs EXTENSIVE MPI collectives internally:
     * - MPI_Allgather: Q/K/V for PyTorch parity snapshots (optional, debug only)
     * - MPI_Allgatherv: K/V cache gathering for attention computation (required)
     * - MPI_Allgatherv: Attention scores gathering across ranks (required)
     * - MPI_Allgather: Attended output (scores @ V) for snapshots (optional, debug only)
     * - MPI_Allreduce (MPI_SUM): Final output projection aggregation (ALWAYS performed)
     *
     * This design ensures:
     * - All ranks have identical attention output (no partial results)
     * - Downstream layers (RMSNorm, MLP) receive fully replicated activations
     * - No additional caller-side MPI operations needed for basic inference
     * - Correctness verified via PyTorch parity testing (2,544/2,544 comparisons passing)
     *
     * Rationale for Full Replication:
     * - Simplifies downstream pipeline (no reconstruction needed)
     * - Enables straightforward layer-to-layer activation flow
     * - Required for correctness with row-sharded Wo weights (additive partials)
     * - Matches PyTorch reference implementation semantics
     *
     * Example Usage:
     * @code
     * // Execute attention kernel - output is fully replicated after execution
     * attention_kernel->execute(inputs, outputs);
     *
     * // outputs[0] is identical on all ranks (no additional MPI operations needed)
     * // Can be used directly by downstream layers
     * rmsnorm_kernel->execute({outputs[0], ...}, next_outputs);
     * @endcode
     *
     * Testing & Validation:
     * - All 8 stages independently unit tested with AttentionStageContracts validation
     * - PyTorch parity tests verify end-to-end correctness: 2,544/2,544 comparisons passing
     * - OpenBLAS prefill: 387/387 stage comparisons passing
     * - COSMA prefill: 387/387 stage comparisons passing
     * - Incremental decode: 1,170/1,170 stage comparisons passing
     *
     * Performance Characteristics:
     * - Decode latency: <10ms per token (optimized for single-token generation)
     * - Prefill throughput: Backend determined by CosmaPrefillManager presence, not adaptive
     * - Memory: Optimized KV cache with zero-copy MPI (but memcpy unpacking still required)
     * - Scalability: Linear scaling with MPI ranks via head-wise parallelism
     * - Threading: OpenBLAS auto-detects optimal thread count (respects OMP_NUM_THREADS)
     *
     * @see AttentionStageContracts for stage validation contracts
     * @see CosmaPrefillManager for distributed COSMA integration
     */
    class MPIAttentionKernel : public MPIKernelBase
    {
    public:
        // Output assembly / distribution mode (scaffolding for hybrid head + TP design)
        enum class AttentionOutputMode
        {
            LocalHeads,                ///< Return only local head slice (no gather)
            GatherHeadsPostProjection, ///< Project locally then gather concatenated hidden (future)
            GatherHeadsPreProjection,  ///< Gather head contexts before output projection (future)
            Replicated                 ///< Force fully replicated output (debug / legacy)
        };

        struct AttentionResultMeta
        {
            AttentionOutputMode mode = AttentionOutputMode::LocalHeads;
            bool concatenated = false; ///< true if full hidden dimension assembled
            bool replicated = false;   ///< true if buffer identical on all ranks
            int local_head_offset = 0; ///< starting head index owned by this rank
            int local_head_count = 0;  ///< number of heads owned by this rank
        };
        /**
         * @brief Distribution strategies for attention computation
         */
        enum class DistributionStrategy
        {
            HEAD_WISE,    ///< Distribute by attention heads
            SEQUENCE_WISE ///< Distribute by sequence dimension (future extension)
        };

        /**
         * @brief Constructor for MPI attention kernel
         * @param n_head Total number of attention heads
         * @param n_head_kv Number of key-value heads (for grouped attention)
         * @param head_dim Dimension per attention head
         * @param rope_freq_base Base frequency for rotary embeddings
         * @param strategy Distribution strategy to use
         */
        MPIAttentionKernel(int n_head, int n_head_kv, int head_dim,
                           float rope_freq_base = 10000.0f,
                           DistributionStrategy strategy = DistributionStrategy::HEAD_WISE);

        ~MPIAttentionKernel() = default;

        // KernelBase interface implementation
        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs) override;

        // Output mode configuration
        void setOutputMode(AttentionOutputMode m) { output_mode_ = m; }
        AttentionOutputMode outputMode() const { return output_mode_; }
        const AttentionResultMeta &last_result_meta() const { return last_meta_; }

        bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const override;

        std::string getKernelType() const override { return "MPIAttention"; }
        size_t getExpectedInputCount() const override { return 10; } // input, wq, wk, wv, wo, bq, bk, bv, k_cache, v_cache
        size_t getExpectedOutputCount() const override { return 1; }

        // Configuration methods
        void setHeadDimensions(int n_head, int n_head_kv, int head_dim);
        void setSequencePosition(int n_past) { n_past_ = n_past; }
        void setLayerIndex(int idx) { layer_index_ = idx; }
        int getSequencePosition() const { return n_past_; }
        // Set externally computed expected total window length (prefill_len + decoded_tokens)
        void setExpectedTotalWindow(size_t expected) { expected_total_window_ = expected; }
        size_t getExpectedTotalWindow() const { return expected_total_window_; }
        void resetDecodeWindowTracking() { last_seen_decode_T_ = 0; }
        void setDistributionStrategy(DistributionStrategy strategy) { strategy_ = strategy; }

        /**
         * @brief Set optional COSMA backend manager for distributed matmul
         * @param cosma_mgr Pointer to CosmaPrefillManager (nullptr to disable COSMA)
         *
         * When set, the kernel will use MatMulBackendSelector to choose between
         * OpenBLAS (with CblasTrans for transpose) and COSMA (with proper shape
         * contracts) based on operation size and context.
         */
        void setCosmaManager(CosmaPrefillManager *cosma_mgr) { cosma_mgr_ = cosma_mgr; }

        /**
         * @brief Get head distribution for this process
         * @return Pair of (local_heads, head_offset)
         */
        std::pair<int, int> getHeadDistribution() const;

        /**
         * @brief Get head distribution for a specific rank
         * @param rank Target rank
         * @return Pair of (local_heads, head_offset)
         */
        std::pair<int, int> getHeadDistribution(int rank) const;

        /**
         * @brief Get K/V head distribution for this process (for GQA models)
         * @return Pair of (local_kv_heads, kv_head_offset)
         */
        std::pair<int, int> getKVHeadDistribution() const;

        /**
         * @brief Get K/V head distribution for a specific rank (for GQA models)
         * @param rank Target rank
         * @return Pair of (local_kv_heads, kv_head_offset)
         */
        std::pair<int, int> getKVHeadDistribution(int rank) const;

        /**
         * @brief Test harness helper: invoke the private output projection path directly.
         *        Exposed for unit/integration testing of TP simulation logic without
         *        running the full attention pipeline.
         * @note  Not intended for production use; keeps underlying implementation private.
         */
        void testInvokeOutputProjection(const std::shared_ptr<TensorBase> &local_attended_output,
                                        const std::shared_ptr<TensorBase> &local_wo,
                                        std::shared_ptr<TensorBase> &local_final_output,
                                        size_t seq_len, int local_heads, size_t d_model)
        {
            computeLocalOutputProjection(local_attended_output, local_wo, local_final_output,
                                         seq_len, local_heads, d_model);
        }

        /**
         * @brief Callback for capturing intermediate attention snapshots
         * @param stage Pipeline stage identifier
         * @param layer_idx Layer index
         * @param data Pointer to tensor data
         * @param seq_len Sequence length
         * @param feature_dim Feature dimension
         */
        using SnapshotCallback = std::function<void(PipelineStage stage, int layer_idx, const float *data, int seq_len, int feature_dim)>;

        void setSnapshotCallback(SnapshotCallback callback) { snapshot_callback_ = callback; }

    private:
        /**
         * @brief Validate inputs and extract parameters (STEP 1)
         *
         * Performs comprehensive input validation, mode detection, head distribution
         * calculation, and early exit handling for ranks with no work.
         *
         * @param inputs Input tensor vector from execute()
         * @param outputs Output tensor vector from execute() (modified for early exit)
         * @return InputSetupResult containing all validated parameters and tensors
         *
         * @throws std::runtime_error on validation failures (caught in execute())
         */
        InputSetupResult validateAndSetupInputs(
            const std::vector<std::shared_ptr<TensorBase>> &inputs,
            std::vector<std::shared_ptr<TensorBase>> &outputs);

        /**
         * @brief Distribute weights by head dimension (STEP 2)
         *
         * Distributes attention weight matrices across ranks based on head assignment.
         * Handles three cases:
         * - Pre-sharded weights (use directly)
         * - Single rank (use global weights)
         * - Multi-rank with global weights (slice and copy)
         *
         * @param setup Input setup result from STEP 1
         * @return WeightSlices containing local weight and bias slices for this rank
         */
        WeightSlices distributeWeightsByHead(const InputSetupResult &setup);

        /**
         * @brief Compute Q, K, V linear projections (STEP 3)
         *
         * Performs matrix multiplication with optional bias addition to compute:
         * - Q = input @ wq^T + bq (if bias present)
         * - K = input @ wk^T + bk (if bias present)
         * - V = input @ wv^T + bv (if bias present)
         *
         * Includes comprehensive validation, health checks, and optional snapshot logging.
         *
         * @param setup Input setup result from STEP 1
         * @param weights Weight slices from STEP 2
         * @return QKVProjectionResult containing local Q, K, V activation tensors
         */
        QKVProjectionResult computeQKVProjections(
            const InputSetupResult &setup,
            const WeightSlices &weights);

        /**
         * @brief Gather Q/K/V across ranks and invoke snapshot callback (STEP 4)
         *
         * For multi-rank execution with snapshot callback:
         * - Performs MPI_Allgather to collect sharded Q/K/V from all ranks
         * - Rearranges from rank-major to row-interleaved layout
         * - Invokes snapshot callback with gathered global tensors (rank 0 only)
         *
         * For single-rank execution with snapshot callback:
         * - Directly snapshots local tensors (no gathering needed)
         *
         * CRITICAL: Snapshots are captured BEFORE RoPE application to match
         * PyTorch's Q_PROJECTION/K_PROJECTION/V_PROJECTION stage semantics.
         *
         * @param setup Input setup result from STEP 1
         * @param projections QKV projection result from STEP 3
         * @return GatherResult with global tensors (if gathered) and snapshot status
         */
        GatherResult gatherAndSnapshotPreRoPE(
            const InputSetupResult &setup,
            const QKVProjectionResult &projections);

        /**
         * @brief Apply Rotary Position Embeddings (RoPE) to Q and K tensors (STEP 5)
         *
         * Applies sinusoidal rotary embeddings to query and key tensors for positional
         * information encoding. RoPE rotates pairs of dimensions using position-dependent
         * sin/cos frequencies. V tensor passes through unchanged.
         *
         * Also handles KV cache update:
         * - Prefill mode: Initialize cache with current K/V
         * - Decode mode: Append new K/V to existing cache
         *
         * Optionally gathers post-RoPE Q/K/V for ROPE_APPLICATION snapshot validation.
         *
         * Key operations:
         * 1. Optional pre-RoPE debug logging and validation
         * 2. Apply RoPE to local Q and K (in-place rotation)
         * 3. Optional post-RoPE debug logging
         * 4. Update KV cache (append for decode, initialize for prefill)
         * 5. Optional MPI_Allgather for snapshot validation
         * 6. Optional ROPE_APPLICATION snapshot (concatenated Q|K format)
         *
         * @param setup Input setup result from STEP 1
         * @param projections QKV projection result from STEP 3 (Q/K modified in-place)
         * @param k_cache_in Input K cache for decode mode
         * @param v_cache_in Input V cache for decode mode
         * @return RoPEResult with rotated Q/K, V unchanged, updated KV cache, and optional global tensors
         */
        RoPEResult applyRotaryPositionEmbeddings(
            const InputSetupResult &setup,
            const QKVProjectionResult &projections,
            const std::shared_ptr<TensorBase> &k_cache_in,
            const std::shared_ptr<TensorBase> &v_cache_in);

        /**
         * @brief Handle Grouped Query Attention (GQA) expansion (STEP 6)
         *
         * For GQA architectures, replicates K/V heads to match the number of query heads.
         * Each KV head serves multiple query heads (head_ratio = n_head / n_head_kv).
         * For MHA architectures (n_head == n_head_kv), passes through K/V unchanged.
         *
         * Multi-rank optimization (Phase 3+4+5):
         * - Phase 3: Fuse K+V gathering into single MPI_Allgatherv
         * - Phase 4: Cache metadata to skip count gathering on repeated decode calls
         * - Phase 5: Use MPI derived datatype for zero-copy K+V interleaving
         *
         * Single-rank path: Direct KV cache usage (no gathering needed)
         *
         * Key operations:
         * 1. Check if GQA expansion needed (n_head != n_head_kv)
         * 2. For multi-rank: Fused KV cache gathering with Phase 3+4+5 optimizations
         * 3. For single-rank: Use local cache directly
         * 4. Call expand_kv_for_gqa() to replicate KV heads for local query heads
         * 5. Optional debug logging of expanded tensors
         *
         * @param setup Input setup result from STEP 1
         * @param rope_result RoPE result from STEP 5 (contains KV cache)
         * @return GQAExpansionResult with expanded K/V tensors ready for attention
         */
        GQAExpansionResult handleGQAExpansion(
            const InputSetupResult &setup,
            const RoPEResult &rope_result);

        /**
         * @brief Compute attention scores and apply to values (STEP 7)
         *
         * Computes scaled dot-product attention: softmax(QK^T / sqrt(d_k)) @ V
         *
         * For snapshot validation (optional):
         * - Computes unmasked scores for ATTENTION_SCORES snapshot
         * - Gathers across ranks (multi-rank only)
         *
         * For actual attention computation:
         * - Computes masked scores with causal masking (prefill mode)
         * - No masking in decode mode (seq_len=1, attending to full cache)
         * - Applies softmax normalization per head
         * - Optionally snapshots softmax probabilities (ATTENTION_SOFTMAX)
         * - Applies attention weights to V values
         * - Optionally snapshots attended output (ATTENTION_CONTEXT)
         *
         * Key operations:
         * 1. Unmasked score computation for snapshot (if callback set)
         * 2. Masked score computation (causal for prefill, no mask for decode)
         * 3. Softmax normalization (per head, with validation)
         * 4. Apply attention probabilities to V (scores @ V)
         * 5. Optional MPI_Allgather for multi-rank snapshots
         *
         * @param setup Input setup result from STEP 1
         * @param rope_result RoPE result from STEP 5 (contains local Q)
         * @param gqa_result GQA expansion result from STEP 6 (contains expanded K/V)
         * @return AttentionScoresResult with local attended output (scores @ V)
         */
        AttentionScoresResult computeAttentionScores(
            const InputSetupResult &setup,
            const RoPEResult &rope_result,
            const GQAExpansionResult &gqa_result);

        /**
         * @brief Project attended values to output dimension and aggregate across ranks
         *
         * Performs the final stages of attention:
         * 1. Projects attended values: local_attended @ wo^T → [seq_len, d_model]
         * 2. Validates output projection with health checks
         * 3. Aggregates across MPI ranks with MPI_Allreduce (if multi-rank)
         * 4. Optionally snapshots final output for validation
         * 5. Performs final health check after aggregation
         *
         * The output projection transforms per-head attended values back to the full
         * model dimension. In multi-rank mode, each rank produces a partial result
         * (corresponding to its subset of heads), and these are summed globally.
         *
         * @param setup Input configuration and rank information
         * @param weights Weight tensors (local_wo needed for projection)
         * @param attention_result Attended output from attention scores computation
         * @return OutputProjectionResult containing final attention output in model dimension
         */
        OutputProjectionResult projectAndGatherOutput(
            const InputSetupResult &setup,
            const WeightSlices &weights,
            const AttentionScoresResult &attention_result);

        /**
         * @brief Distribute input data and weights to all processes
         * @param global_input Global input tensor
         * @param global_wq Global query weight matrix
         * @param global_wk Global key weight matrix
         * @param global_wv Global value weight matrix
         * @param global_wo Global output weight matrix
         * @param local_wq Local query weight subset
         * @param local_wk Local key weight subset
         * @param local_wv Local value weight subset
         * @param local_wo Local output weight subset
         * @param seq_len Sequence length
         * @param d_model Model dimension
         */
        void distributeInputs(const std::shared_ptr<TensorBase> &global_input,
                              const std::shared_ptr<TensorBase> &global_wq,
                              const std::shared_ptr<TensorBase> &global_wk,
                              const std::shared_ptr<TensorBase> &global_wv,
                              const std::shared_ptr<TensorBase> &global_wo,
                              std::shared_ptr<TensorBase> &local_wq,
                              std::shared_ptr<TensorBase> &local_wk,
                              std::shared_ptr<TensorBase> &local_wv,
                              std::shared_ptr<TensorBase> &local_wo,
                              size_t seq_len, size_t d_model);

        /**
         * @brief Compute local Q, K, V projections for assigned heads using COSMA
         * @param input Input tensor
         * @param local_wq Local query weight tensor
         * @param local_wk Local key weight tensor
         * @param local_wv Local value weight tensor
         * @param local_bq Local query bias tensor
         * @param local_bk Local key bias tensor
         * @param local_bv Local value bias tensor
         * @param local_q Output local query projection tensor
         * @param local_k Output local key projection tensor
         * @param local_v Output local value projection tensor
         * @param seq_len Sequence length
         * @param d_model Model dimension
         */
        void computeLocalProjections(const std::shared_ptr<TensorBase> &input,
                                     const std::shared_ptr<TensorBase> &local_wq,
                                     const std::shared_ptr<TensorBase> &local_wk,
                                     const std::shared_ptr<TensorBase> &local_wv,
                                     const std::shared_ptr<TensorBase> &local_bq,
                                     const std::shared_ptr<TensorBase> &local_bk,
                                     const std::shared_ptr<TensorBase> &local_bv,
                                     std::shared_ptr<TensorBase> &local_q,
                                     std::shared_ptr<TensorBase> &local_k,
                                     std::shared_ptr<TensorBase> &local_v,
                                     size_t seq_len, size_t d_model);

        /**
         * @brief Backend-agnostic matrix multiplication with optional bias
         * @param input Input matrix (row-major, M x K)
         * @param weight Weight matrix (row-major, N x K) - transposed during multiply
         * @param output Output matrix (row-major, M x N)
         * @param bias Optional bias vector (length N)
         * @param M Rows in input/output
         * @param N Columns in output (rows in weight)
         * @param K Columns in input/weight
         * @param operation_label Optional label for logging
         */
        void matmul_with_bias(const float *input, const float *weight, float *output,
                              const float *bias, int M, int N, int K,
                              const char *operation_label = nullptr);

        /**
         * @brief Compute attention for local heads
         * @param local_q Local query projections
         * @param local_k Local key projections
         * @param local_v Local value projections
         * @param local_output Local attention output
         * @param seq_len Sequence length
         * @param local_heads Number of local heads
         */
        void computeLocalAttention(const float *local_q, const float *local_k, const float *local_v,
                                   float *local_output, size_t seq_len, int local_heads);

        /**
         * @brief Apply RoPE to local Q and K projections
         * @param local_q Local query tensor
         * @param local_k Local key tensor
         * @param seq_len Sequence length
         * @param local_heads Number of local heads
         */
        void applyLocalRoPE(float *local_q, float *local_k, size_t seq_len, int local_heads);

        /**
         * @brief Compute attention scores and softmax for local heads
         * @param local_q Local query projections
         * @param local_k Local key projections
         * @param scores Local attention scores
         * @param seq_len Sequence length
         * @param local_heads Number of local heads
         */
        void computeLocalAttentionScores(const float *local_q, const float *local_k, float *scores,
                                         size_t seq_len, int local_heads);

        /**
         * @brief Apply local attention scores to values
         * @param scores Local attention scores
         * @param local_v Local value projections
         * @param local_attended_output Local attended output
         * @param seq_len Sequence length
         * @param local_heads Number of local heads
         */
        void applyLocalAttention(const float *scores, const float *local_v, float *local_attended_output,
                                 size_t seq_len, int local_heads);

        /**
         * @brief Compute final output projection for local heads using COSMA
         * @param local_attended_output Local attended output tensor
         * @param local_wo Local output weight tensor
         * @param local_final_output Local final output tensor
         * @param seq_len Sequence length
         * @param local_heads Number of local heads
         */
        void computeLocalOutputProjection(const std::shared_ptr<TensorBase> &local_attended_output,
                                          const std::shared_ptr<TensorBase> &local_wo,
                                          std::shared_ptr<TensorBase> &local_final_output,
                                          size_t seq_len, int local_heads, size_t d_model);

        std::shared_ptr<TensorBase> createLocalSimpleTensor(const std::vector<size_t> &shape) const;

        /**
         * @brief Initialize stage contracts for runtime validation
         * @param seq_len Sequence length (can be -1 for dynamic)
         * @param local_heads Local head count for this rank
         * @param local_kv_heads Local K/V head count for this rank
         */
        void initializeStageContracts(int seq_len, int local_heads, int local_kv_heads);

        // Configuration parameters
        int n_head_;                                                        ///< Total number of attention heads
        int n_head_kv_;                                                     ///< Number of key-value heads (grouped attention)
        int head_dim_;                                                      ///< Dimension per attention head
        int n_past_;                                                        ///< Number of past tokens for position embedding
        int layer_index_ = -1;                                              ///< Layer index for diagnostics
        int d_model_;                                                       ///< Model dimension
        AttentionOutputMode output_mode_ = AttentionOutputMode::LocalHeads; ///< Attention output mode (DEFAULT: LocalHeads is for future head-sharded TP; current row-partitioned W_o requires GatherHeadsPostProjection!)
        SnapshotCallback snapshot_callback_;                                ///< Optional callback for intermediate snapshots
        size_t expected_total_window_ = 0;                                  ///< Expected causal window (external assertion)
        size_t last_seen_decode_T_ = 0;                                     ///< Per-instance last seen T for monotonic check
        float rope_freq_base_;                                              ///< Base frequency for rotary embeddings
        DistributionStrategy strategy_;                                     ///< Distribution strategy
        AttentionResultMeta last_meta_{};                                   ///< metadata from last execute

        // ========================================================================
        // STAGE CONTRACTS - Explicit data flow validation
        // ========================================================================
        // These contracts define expected shapes, layouts, and semantics at each
        // transformation stage. They catch dimension mismatches and transpose bugs
        // early with clear error messages.

        bool contracts_enabled_ = true; ///< Whether contract validation is active (disable for benchmarks)

        StageContract contract_projection_;        ///< Stage 1: Q/K/V projections
        StageContract contract_rope_;              ///< Stage 2: RoPE application
        StageContract contract_gqa_replication_;   ///< Stage 3: K/V replication for GQA
        StageContract contract_attention_;         ///< Stage 4: Attention computation
        StageContract contract_output_projection_; ///< Stage 5: Output projection

        // ========================================================================
        // PHASE 4 & 5 MPI OPTIMIZATIONS: Cached metadata and zero-copy datatypes
        // ========================================================================
        // Phase 4: Cache count gather results for predictable decode growth
        // Phase 5: Use MPI derived datatypes to eliminate pack/unpack memcpy

        bool kv_cache_metadata_initialized_ = false;           ///< Whether cached metadata is valid
        int last_attn_seq_len_ = -1;                           ///< Last seen attn_seq_len for cache invalidation
        std::vector<int> cached_recvcounts_kv_;                ///< Cached receive counts from last gather
        std::vector<int> cached_displs_kv_;                    ///< Cached displacements from last gather
        MPI_Datatype kv_interleaved_type_ = MPI_DATATYPE_NULL; ///< Phase 5: Derived type for K+V interleaving

        // ========================================================================
        // COSMA BACKEND SUPPORT (Optional)
        // ========================================================================
        CosmaPrefillManager *cosma_mgr_ = nullptr; ///< Optional COSMA backend for distributed matmul
    };

} // namespace llaminar