/**
 * @file MPIAttentionKernel.cpp
 * @brief Clean, minimal implementation of MPI attention kernel
 * @author David Sanftenberg
 *
 * This is a simplified rewrite focusing on clarity and correctness.
 * Key improvements:
 * - No lambdas - all proper member functions
 * - Single execution path - no conditional debug branches
 * - Uses TensorFactory throughout - no type conversion issues
 * - Direct BLAS calls - no backend abstraction layers
 * - Optimized primitives for softmax, RoPE, attention
 * - Optional validation (zero overhead when disabled)
 *
 * Architecture: 7 sequential steps
 * 1. Extract inputs & validate count
 * 2. Distribute weights (single-rank: use directly, multi-rank: slice by heads)
 * 3. Compute Q/K/V projections with optional bias
 * 4. Apply RoPE to Q and K
 * 5. Expand K/V for GQA if needed
 * 6. Compute attention: QK^T → softmax → @V
 * 7. Output projection + MPI gather
 */

#include "MPIAttentionKernel.h"
#include "../logger.h"
#include "../tensors/tensor_factory.h"
#include "../matmul_backend_selection.h"
#include "../cosma_prefill_manager.h"
#include "common/attention_primitives.h"
#include "common/softmax_core.h"
#include "attention/AttentionStageContracts.h"
#include "attention/AttentionValidator.h"
#include "../utils/debug_env.h"
#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <mpi.h>

namespace llaminar
{

    // ============================================================================
    // Helper: Granular tensor validation (detects NaN, Inf, uninitialized data)
    // ============================================================================
    struct TensorHealthCheck
    {
        std::string name;
        int nan_count = 0;
        int inf_count = 0;
        int zero_count = 0;
        int normal_count = 0;
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        float abs_sum = 0.0f;

        TensorHealthCheck(const std::string &n) : name(n) {}

        void check(const float *data, size_t size)
        {
            for (size_t i = 0; i < size; ++i)
            {
                float val = data[i];
                if (std::isnan(val))
                {
                    nan_count++;
                }
                else if (std::isinf(val))
                {
                    inf_count++;
                }
                else if (val == 0.0f)
                {
                    zero_count++;
                }
                else
                {
                    normal_count++;
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                    abs_sum += std::abs(val);
                }
            }
        }

        bool is_healthy() const
        {
            return nan_count == 0 && inf_count == 0 && normal_count > 0;
        }

        bool is_uninitialized() const
        {
            // Heuristic: if min/max are huge or all zeros, likely uninitialized
            return (normal_count > 0 && (std::abs(min_val) > 1e10f || std::abs(max_val) > 1e10f)) ||
                   (normal_count == 0 && zero_count > 0);
        }

        void log(int rank = -1) const
        {
            std::string prefix = (rank >= 0) ? "[Rank " + std::to_string(rank) + "] " : "";
            if (!is_healthy())
            {
                LOG_ERROR(prefix << "UNHEALTHY " << name << ": NaN=" << nan_count
                                 << " Inf=" << inf_count << " Zero=" << zero_count
                                 << " Normal=" << normal_count);
                if (normal_count > 0)
                {
                    LOG_ERROR(prefix << "  Range: [" << min_val << ", " << max_val << "], Sum=" << abs_sum);
                }
            }
            else if (is_uninitialized())
            {
                LOG_WARN(prefix << "SUSPICIOUS " << name << ": Range [" << min_val << ", " << max_val
                                << "] suggests uninitialized data");
            }
            else
            {
                LOG_DEBUG(prefix << "HEALTHY " << name << ": " << normal_count << " values in ["
                                 << min_val << ", " << max_val << "], sum=" << abs_sum);
            }
        }
    };

    // ============================================================================
    // Constructor
    // ============================================================================
    MPIAttentionKernel::MPIAttentionKernel(int n_head, int n_head_kv, int head_dim,
                                           float rope_freq_base, DistributionStrategy strategy)
        : n_head_(n_head),
          n_head_kv_(n_head_kv),
          head_dim_(head_dim),
          n_past_(0),
          d_model_(n_head * head_dim),
          rope_freq_base_(rope_freq_base),
          strategy_(strategy)
    {
        if (head_dim_ <= 0 || n_head_ <= 0 || n_head_kv_ <= 0)
        {
            throw std::invalid_argument("MPIAttentionKernel: invalid constructor parameters");
        }
        if (n_head_kv_ > n_head_)
        {
            throw std::invalid_argument("MPIAttentionKernel: n_head_kv cannot exceed n_head");
        }

        // Check for excess MPI ranks (more ranks than heads to distribute)
        const int world_size = getSize();
        if (world_size > n_head_)
        {
            LOG_WARN("MPIAttentionKernel: More ranks (" << world_size
                                                        << ") than Q heads (" << n_head_ << "). "
                                                        << (world_size - n_head_) << " rank(s) will have no work (local_heads=0).");
        }
        if (world_size > n_head_kv_)
        {
            LOG_WARN("MPIAttentionKernel: More ranks (" << world_size
                                                        << ") than KV heads (" << n_head_kv_ << "). "
                                                        << (world_size - n_head_kv_) << " rank(s) will have no KV work (local_kv_heads=0).");
        }
    }

    // ============================================================================
    // Helper: Head distribution methods
    // ============================================================================
    std::pair<int, int> MPIAttentionKernel::getHeadDistribution() const
    {
        return getHeadDistribution(getRank());
    }

    std::pair<int, int> MPIAttentionKernel::getHeadDistribution(int rank) const
    {
        int heads_per_rank = n_head_ / getSize();
        int remainder = n_head_ % getSize();

        int local_heads = heads_per_rank + (rank < remainder ? 1 : 0);
        int head_offset = rank * heads_per_rank + std::min(rank, remainder);

        return {local_heads, head_offset};
    }

    std::pair<int, int> MPIAttentionKernel::getKVHeadDistribution() const
    {
        return getKVHeadDistribution(getRank());
    }

    std::pair<int, int> MPIAttentionKernel::getKVHeadDistribution(int rank) const
    {
        int heads_per_rank = n_head_kv_ / getSize();
        int remainder = n_head_kv_ % getSize();

        int local_heads = heads_per_rank + (rank < remainder ? 1 : 0);
        int head_offset = rank * heads_per_rank + std::min(rank, remainder);

        return {local_heads, head_offset};
    }

    // ============================================================================
    // Validation
    // ============================================================================
    bool MPIAttentionKernel::validate(
        const std::vector<std::shared_ptr<TensorBase>> &inputs,
        const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        if (inputs.size() != 10)
        {
            LOG_ERROR("MPIAttentionKernel: Expected 10 inputs, got " << inputs.size());
            return false;
        }

        if (outputs.size() != 1)
        {
            LOG_ERROR("MPIAttentionKernel: Expected 1 output, got " << outputs.size());
            return false;
        }

        auto input = inputs[0];
        if (input->shape().size() != 2)
        {
            LOG_ERROR("MPIAttentionKernel: Input must be 2D [seq_len, d_model], got shape size "
                      << input->shape().size());
            return false;
        }

        return true;
    }

    // ============================================================================
    // Helper: Backend-agnostic matrix multiplication with optional bias
    // ============================================================================
    /**
     * @brief Perform matrix multiplication with optional bias using selected backend
     *
     * @param input Input matrix (row-major, M x K)
     * @param weight Weight matrix (row-major, N x K) - will be transposed during multiplication
     * @param output Output matrix (row-major, M x N)
     * @param bias Optional bias vector (length N)
     * @param M Number of rows in input and output
     * @param N Number of columns in output (rows in weight)
     * @param K Number of columns in input (columns in weight)
     * @param operation_label Human-readable operation name for logging
     *
     * @note Weight layout: Stored as [N, K] in row-major format
     *       OpenBLAS: Uses CblasTrans flag to transpose [N,K] -> [K,N] during multiplication
     *       COSMA: Requires explicit pre-transposed [K,N] layout (handled internally)
     *
     * @note This method chooses between OpenBLAS and COSMA based on:
     *       - Operation size (M*N*K)
     *       - CosmaPrefillManager availability (cosma_mgr_ != nullptr)
     *       - MatMulBackendSelector decision logic
     */
    void MPIAttentionKernel::matmul_with_bias(
        const float *input, const float *weight, float *output,
        const float *bias, int M, int N, int K,
        const char *operation_label)
    {
        const int rank = getRank();
        const int world_size = getSize();

        // DEBUG: Trace bias status at function entry
        if (layer_index_ == 0 && operation_label)
        {
            LOG_INFO("[MATMUL_BIAS] Layer " << layer_index_ << " Rank " << rank
                                            << " Operation: " << operation_label);
            if (bias)
            {
                LOG_INFO("[MATMUL_BIAS] BIAS PRESENT - will apply bias after matmul");
                LOG_INFO("[MATMUL_BIAS] bias[0:3]: [" << bias[0] << ", " << bias[1] << ", " << bias[2] << "]");
            }
            else
            {
                LOG_INFO("[MATMUL_BIAS] NO BIAS - bias pointer is NULL");
            }
        }

        // Backend selection: Use COSMA when manager is available
        // NOTE: For parity testing, we want COSMA to be used when available
        // In production, you might want stricter thresholds (e.g., M >= 8192)
        const bool use_cosma = (cosma_mgr_ != nullptr);
        // Execute based on selected backend
        if (use_cosma)
        {
            // COSMA path: Use transposeW=true to handle [N,K] -> [K,N] transpose
            // OpenBLAS uses CblasTrans for the same purpose

            // Step 1: Create WeightDescriptor with original [N,K] layout
            // Step 2: Create WeightDescriptor with ORIGINAL dimensions (not transposed)
            // COSMA will handle the transpose internally when transposeW=true
            WeightDescriptor weight_desc{
                operation_label ? operation_label : "weight", // id
                N,                                            // rows (n = output_dim, original layout)
                K,                                            // cols (k = input_dim, original layout)
                static_cast<int64_t>(K),                      // row_stride (K elements per row in [N,K] layout)
                1,                                            // col_stride
                0,                                            // quant_type (0 = float32)
                weight,                                       // base_ptr (use original, not transposed!)
                0                                             // quant_block_size
            };

            // Step 3: Convert input to COSMA view
            CosmaView input_view = cosma_mgr_->convert_activation_in(input, M, K);
            if (!input_view.mat && !input_view.host_owned)
            {
                LOG_ERROR("[MPIAttentionKernel] Failed to convert input for COSMA");
                // Fall back to OpenBLAS
                goto openblas_fallback;
            }

            // Step 4: Load weight
            auto weight_handle = cosma_mgr_->load_weight(weight_desc);
            if (!weight_handle.view.mat)
            {
                LOG_ERROR("[MPIAttentionKernel] Failed to load weight for COSMA");
                goto openblas_fallback;
            }

            // Step 5: Perform COSMA matmul with transposeW=true
            CosmaView result_view = cosma_mgr_->matmul(input_view, weight_handle, M, K, N, true);
            if (!result_view.mat && !result_view.host_owned)
            {
                LOG_ERROR("[MPIAttentionKernel] COSMA matmul failed");
                goto openblas_fallback;
            }

            // Step 6: Reconstruct result to output buffer
            cosma_mgr_->reconstruct_matrix(result_view, output, true);

            // Step 7: Add bias if provided
            if (bias)
            {
                for (int m = 0; m < M; ++m)
                {
                    for (int n = 0; n < N; ++n)
                    {
                        output[m * N + n] += bias[n];
                    }
                }
            }

            // Step 8: Release COSMA resources
            cosma_mgr_->release_weight(std::move(weight_handle));

            return; // COSMA path complete
        }

    openblas_fallback:
        // OpenBLAS path (default and fallback)
        // Force single-threaded execution to avoid threading bugs
        openblas_set_num_threads(1);

        // DEBUG: Log cblas_sgemm parameters for layer 0
        if (operation_label && std::string(operation_label) == "Q_projection")
        {
            LOG_INFO("[CBLAS_DEBUG] Rank " << rank << " cblas_sgemm call:");
            LOG_INFO("  Operation: " << (operation_label ? operation_label : "unknown"));
            LOG_INFO("  M=" << M << " N=" << N << " K=" << K);
            LOG_INFO("  input ptr=" << (void *)input << " leading dim=" << K);
            LOG_INFO("  weight ptr=" << (void *)weight << " leading dim=" << K << " (will be transposed)");
            LOG_INFO("  output ptr=" << (void *)output << " leading dim=" << N);
            LOG_INFO("  input[0:5]: [" << input[0] << ", " << input[1] << ", " << input[2] << ", " << input[3] << ", " << input[4] << "]");
            LOG_INFO("  weight[0:5]: [" << weight[0] << ", " << weight[1] << ", " << weight[2] << ", " << weight[3] << ", " << weight[4] << "]");
        }

        // output = input @ weight^T  (input is MxK, weight is NxK, output is MxN)
        // CblasTrans flag transposes weight from [N,K] to [K,N] during multiplication
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, N, K,
                    1.0f, input, K, weight, K,
                    0.0f, output, N);

        // Add bias if provided
        if (bias)
        {
            for (int m = 0; m < M; ++m)
            {
                for (int n = 0; n < N; ++n)
                {
                    output[m * N + n] += bias[n];
                }
            }
        }
    }

    // ============================================================================
    // MAIN EXECUTE FUNCTION
    // ============================================================================
    bool MPIAttentionKernel::execute(
        const std::vector<std::shared_ptr<TensorBase>> &inputs,
        std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        const int rank = getRank();
        const int world_size = getSize();

        // DEBUG: Confirm execution
        if (rank == 0)
        {
            LOG_DEBUG("[EXECUTE] MPIAttentionKernel::execute() called"
                      << " layer=" << layer_index_
                      << " cosma_mgr=" << (void *)cosma_mgr_
                      << " snapshot_cb=" << (snapshot_callback_ ? "SET" : "NULL"));
        }

        const auto &debug_snapshot = debugEnv();
        const bool enable_validation = debug_snapshot.attention.validate_output;
        const bool validate_projections = debug_snapshot.attention.validate_proj;
        const bool trace_weight_slice = debug_snapshot.attention.trace_weight_slicing;
        const bool trace_k_projection = debug_snapshot.attention.trace_k_projection;

        // ========================================================================
        // STEP 1: Validate inputs and extract parameters
        // ========================================================================
        if (inputs.size() < 10)
        {
            LOG_ERROR("MPIAttentionKernel: Expected 10 inputs (input, wq, wk, wv, wo, bq, bk, bv, k_cache, v_cache), got " << inputs.size());
            return false;
        }

        // ========================================================================
        // TENSOR DIMENSIONALITY REFERENCE
        // ========================================================================
        // Extract inputs with explicit dimensions:
        //   input:      [seq_len, d_model]              - Input activations
        //   wq_global:  [n_head*head_dim, d_model] OR [local_head_dim, d_model] if sharded
        //   wk_global:  [n_head_kv*head_dim, d_model] OR [local_kv_head_dim, d_model] if sharded
        //   wv_global:  [n_head_kv*head_dim, d_model] OR [local_kv_head_dim, d_model] if sharded
        //   wo_global:  [d_model, n_head*head_dim] OR [d_model, local_head_dim] if sharded
        //   bq_global:  [n_head*head_dim] OR [local_head_dim] if sharded (optional, may be nullptr)
        //   bk_global:  [n_head_kv*head_dim] OR [local_kv_head_dim] if sharded (optional, may be nullptr)
        //   bv_global:  [n_head_kv*head_dim] OR [local_kv_head_dim] if sharded (optional, may be nullptr)
        //   k_cache_in: [cache_capacity, local_kv_head_dim] - KV cache from previous step (or empty for prefill)
        //   v_cache_in: [cache_capacity, local_kv_head_dim] - KV cache from previous step (or empty for prefill)
        //
        // Intermediate tensors created during execution:
        //   local_q:    [seq_len, local_head_dim]       - Q projection for this rank's heads
        //   local_k:    [seq_len, local_kv_head_dim]    - K projection for this rank's KV heads
        //   local_v:    [seq_len, local_kv_head_dim]    - V projection for this rank's KV heads
        //   global_q:   [seq_len, n_head*head_dim]      - Q after MPI_Allgather (all heads)
        //   global_k:   [seq_len, n_head_kv*head_dim]   - K after MPI_Allgather (all KV heads)
        //   global_v:   [seq_len, n_head_kv*head_dim]   - V after MPI_Allgather (all KV heads)
        //   attention_scores: [seq_len, n_head, attn_seq_len] - Attention scores after softmax
        //   attended:   [seq_len, local_head_dim]       - Attended values for this rank's heads
        //   output:     [seq_len, d_model]              - Final output after output projection
        // ========================================================================

        auto input = inputs[0];
        auto wq_global = inputs[1];
        auto wk_global = inputs[2];
        auto wv_global = inputs[3];
        auto wo_global = inputs[4];
        auto bq_global = inputs[5];
        auto bk_global = inputs[6];
        auto bv_global = inputs[7];
        auto k_cache_in = inputs[8]; // KV cache from previous step (or empty for prefill)
        auto v_cache_in = inputs[9]; // KV cache from previous step (or empty for prefill)

        const int seq_len = static_cast<int>(input->shape()[0]);
        const int d_model = static_cast<int>(input->shape()[1]);

        // DEBUG: Trace bias flow from input extraction
        if (layer_index_ == 0)
        {
            LOG_INFO("[BIAS_FLOW] Layer " << layer_index_ << " Rank " << rank
                                          << " bq_global=" << (bq_global ? "PRESENT" : "null")
                                          << " size=" << (bq_global ? bq_global->size() : 0)
                                          << " first_val=" << (bq_global && bq_global->size() > 0 ? bq_global->data()[0] : 0.0f));
            LOG_INFO("[BIAS_FLOW] Layer " << layer_index_ << " Rank " << rank
                                          << " bk_global=" << (bk_global ? "PRESENT" : "null")
                                          << " size=" << (bk_global ? bk_global->size() : 0));
            LOG_INFO("[BIAS_FLOW] Layer " << layer_index_ << " Rank " << rank
                                          << " bv_global=" << (bv_global ? "PRESENT" : "null")
                                          << " size=" << (bv_global ? bv_global->size() : 0));
        }

        // Determine mode based on n_past (authoritative signal)
        // Prefill: n_past=0, use full input sequence for Q/K/V
        // Decode: n_past>0, use cache for K/V history, input for new Q
        const bool is_decode_mode = (n_past_ > 0);
        const int cache_seq_len = is_decode_mode ? n_past_ : 0;

        // Validate cache presence in decode mode
        if (is_decode_mode)
        {
            if (!k_cache_in || !v_cache_in)
            {
                LOG_ERROR("Decode mode (n_past=" << n_past_ << ") requires KV cache inputs");
                return false;
            }
            // Cache capacity must be >= n_past (used length)
            const int k_cache_capacity = static_cast<int>(k_cache_in->shape()[0]);
            const int v_cache_capacity = static_cast<int>(v_cache_in->shape()[0]);
            if (k_cache_capacity < n_past_)
            {
                LOG_ERROR("KV cache insufficient: k_cache capacity " << k_cache_capacity
                                                                     << " < n_past=" << n_past_);
                return false;
            }
            if (v_cache_capacity < n_past_)
            {
                LOG_ERROR("KV cache insufficient: v_cache capacity " << v_cache_capacity
                                                                     << " < n_past=" << n_past_);
                return false;
            }
            if (rank == 0)
            {
                LOG_DEBUG("[KV_CACHE] Decode mode: n_past=" << n_past_
                                                            << ", cache_len=" << cache_seq_len
                                                            << ", new_seq_len=" << seq_len);
            }
        }
        else
        {
            if (rank == 0)
            {
                LOG_DEBUG("[KV_CACHE] Prefill mode: seq_len=" << seq_len
                                                              << ", will initialize cache");
            }
        }

        // Get head distribution
        auto [local_heads, head_offset] = getHeadDistribution();
        auto [local_kv_heads, kv_head_offset] = getKVHeadDistribution();
        const int local_head_dim = local_heads * head_dim_;
        const int local_kv_head_dim = local_kv_heads * head_dim_;

        // ========================================================================
        // EARLY EXIT: Rank has no work to do (more ranks than heads)
        // ========================================================================
        if (local_heads == 0)
        {
            if (rank == 0)
            {
                LOG_INFO("Rank " << rank << " has no Q heads to process (local_heads=0). "
                                 << "Creating zero-filled output and participating in collectives only.");
            }

            // Create zero-filled output tensor for this rank
            // Output shape should match expected partial output: [seq_len, 0] effectively
            // But we still need to participate in MPI collectives, so create minimal tensor
            auto zero_output = TensorFactory::create_simple({seq_len, 0});
            outputs[0] = zero_output;

            // Note: This rank will still participate in MPI_Allgather/Allreduce operations
            // with sendcount=0, which is valid and necessary for collective completion
            return true;
        }

        if (local_kv_heads == 0)
        {
            LOG_WARN("Rank " << rank << " has no KV heads (local_kv_heads=0). "
                             << "This may indicate misconfiguration (more ranks than KV heads).");
            // Continue execution - Q heads can still be processed using gathered KV
        }

        // VALIDATION: Bias tensor sizes must match projection output dimensions
        // Bias is broadcast across batch/sequence dimension, so should have size = output_features
        // Note: We allow size <= 1 biases (nullptr or dummy single-element tensors) to be skipped
        if (bq_global && bq_global->data() && bq_global->size() > 1)
        {
            if (bq_global->size() != local_head_dim && bq_global->size() != (n_head_ * head_dim_))
            {
                LOG_ERROR("Q bias size mismatch: got " << bq_global->size()
                                                       << ", expected " << local_head_dim << " (local) or "
                                                       << (n_head_ * head_dim_) << " (global). "
                                                       << "Bias must match output feature dimension or be nullptr/size-1 to skip.");
                return false;
            }
        }
        if (bk_global && bk_global->data() && bk_global->size() > 1)
        {
            if (bk_global->size() != local_kv_head_dim && bk_global->size() != (n_head_kv_ * head_dim_))
            {
                LOG_ERROR("K bias size mismatch: got " << bk_global->size()
                                                       << ", expected " << local_kv_head_dim << " (local) or "
                                                       << (n_head_kv_ * head_dim_) << " (global). "
                                                       << "Bias must match output feature dimension or be nullptr/size-1 to skip.");
                return false;
            }
        }
        if (bv_global && bv_global->data() && bv_global->size() > 1)
        {
            if (bv_global->size() != local_kv_head_dim && bv_global->size() != (n_head_kv_ * head_dim_))
            {
                LOG_ERROR("V bias size mismatch: got " << bv_global->size()
                                                       << ", expected " << local_kv_head_dim << " (local) or "
                                                       << (n_head_kv_ * head_dim_) << " (global). "
                                                       << "Bias must match output feature dimension or be nullptr/size-1 to skip.");
                return false;
            }
        }

        // WEIGHT FORMAT CONTRACT:
        // All weights are guaranteed by QwenModelWeights to be in canonical GGUF format:
        // - wq, wk, wv: [out_features, in_features] = [n_head*head_dim, d_model] or [local_head_dim, d_model] if sharded
        // - wo: [in_features, out_features] = [d_model, n_head*head_dim] or [d_model, local_head_dim] if sharded
        // Sharding detection: check if first dimension matches local vs global head dimension

        const int wq_rows = static_cast<int>(wq_global->shape()[0]);
        const bool weights_are_sharded = (wq_rows == local_head_dim);

        // Expected shapes based on sharding status
        const int expected_wq_rows = weights_are_sharded ? local_head_dim : (n_head_ * head_dim_);
        const int expected_wk_rows = weights_are_sharded ? local_kv_head_dim : (n_head_kv_ * head_dim_);
        const int expected_wv_rows = weights_are_sharded ? local_kv_head_dim : (n_head_kv_ * head_dim_);
        const int expected_wo_cols = weights_are_sharded ? local_head_dim : (n_head_ * head_dim_);

        // CRITICAL: Always validate weight dimensions before matmul to prevent crashes
        // (regardless of enable_validation flag)
        const int wq_cols = static_cast<int>(wq_global->shape()[1]);
        const int wk_rows = static_cast<int>(wk_global->shape()[0]);
        const int wk_cols = static_cast<int>(wk_global->shape()[1]);
        const int wv_rows = static_cast<int>(wv_global->shape()[0]);
        const int wv_cols = static_cast<int>(wv_global->shape()[1]);
        const int wo_rows = static_cast<int>(wo_global->shape()[0]);
        const int wo_cols = static_cast<int>(wo_global->shape()[1]);

        if (wq_rows != expected_wq_rows || wq_cols != d_model)
        {
            LOG_ERROR("wq dimension mismatch: got [" << wq_rows << "," << wq_cols << "], expected ["
                                                     << expected_wq_rows << "," << d_model << "]");
            return false;
        }
        if (wk_rows != expected_wk_rows || wk_cols != d_model)
        {
            LOG_ERROR("wk dimension mismatch: got [" << wk_rows << "," << wk_cols << "], expected ["
                                                     << expected_wk_rows << "," << d_model << "]");
            return false;
        }
        if (wv_rows != expected_wv_rows || wv_cols != d_model)
        {
            LOG_ERROR("wv dimension mismatch: got [" << wv_rows << "," << wv_cols << "], expected ["
                                                     << expected_wv_rows << "," << d_model << "]");
            return false;
        }
        if (wo_rows != d_model || wo_cols != expected_wo_cols)
        {
            LOG_ERROR("wo dimension mismatch: got [" << wo_rows << "," << wo_cols << "], expected ["
                                                     << d_model << "," << expected_wo_cols << "]");
            return false;
        }

        if (rank == 0)
        {
            LOG_INFO("Attention layer " << layer_index_ << ": seq_len=" << seq_len
                                        << ", d_model=" << d_model << ", local_heads=" << local_heads
                                        << "/" << n_head_ << ", local_kv_heads=" << local_kv_heads << "/" << n_head_kv_
                                        << ", weights_sharded=" << (weights_are_sharded ? "yes" : "no"));
        }

        // CONTRACT: Input validation (handle both sharded and full weights)
        if (enable_validation)
        {
            StageContract input_contract("InputValidation");
            input_contract.inputs = {
                TensorContract("input", ShapeSpec({seq_len, d_model}, {"seq_len", "d_model"}),
                               TensorLayout::RowMajor, TensorSemantic::Activation),
                TensorContract("wq", ShapeSpec({expected_wq_rows, d_model}),
                               TensorLayout::RowMajor, TensorSemantic::Weight),
                TensorContract("wk", ShapeSpec({expected_wk_rows, d_model}),
                               TensorLayout::RowMajor, TensorSemantic::Weight),
                TensorContract("wv", ShapeSpec({expected_wv_rows, d_model}),
                               TensorLayout::RowMajor, TensorSemantic::Weight),
                TensorContract("wo", ShapeSpec({d_model, expected_wo_cols}),
                               TensorLayout::RowMajor, TensorSemantic::Weight)};

            try
            {
                input_contract.validate_inputs({input, wq_global, wk_global, wv_global, wo_global});
                if (rank == 0)
                    LOG_DEBUG("✓ Input shape contracts validated (sharded=" << weights_are_sharded << ")");
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Input contract violation: " << e.what());
                return false;
            }

            // HEALTH CHECK: Detect uninitialized/corrupted input data
            if (rank == 0)
            {
                TensorHealthCheck checks[] = {
                    TensorHealthCheck("input"),
                    TensorHealthCheck("wq_global"),
                    TensorHealthCheck("wk_global"),
                    TensorHealthCheck("wv_global"),
                    TensorHealthCheck("wo_global")};
                const float *data_ptrs[] = {
                    input->data(), wq_global->data(), wk_global->data(),
                    wv_global->data(), wo_global->data()};
                size_t sizes[] = {
                    input->size(), wq_global->size(), wk_global->size(),
                    wv_global->size(), wo_global->size()};

                bool all_healthy = true;
                for (int i = 0; i < 5; ++i)
                {
                    checks[i].check(data_ptrs[i], sizes[i]);
                    checks[i].log(rank);
                    if (!checks[i].is_healthy())
                    {
                        all_healthy = false;
                    }
                }

                if (!all_healthy)
                {
                    LOG_ERROR("❌ Input tensors contain NaN/Inf - aborting execution");
                    return false;
                }

                // Add detailed input statistics for debugging
                float input_min = *std::min_element(input->data(), input->data() + input->size());
                float input_max = *std::max_element(input->data(), input->data() + input->size());
                float input_sum = std::accumulate(input->data(), input->data() + input->size(), 0.0f);
                float input_mean = input_sum / input->size();

                float wq_min = *std::min_element(wq_global->data(), wq_global->data() + wq_global->size());
                float wq_max = *std::max_element(wq_global->data(), wq_global->data() + wq_global->size());
                float wq_sum = std::accumulate(wq_global->data(), wq_global->data() + wq_global->size(), 0.0f);
                float wq_mean = wq_sum / wq_global->size();

                LOG_DEBUG("[INPUT_DEBUG] Layer " << layer_index_ << " Input stats: "
                                                 << "size=" << input->size() << " shape=[" << seq_len << "," << d_model << "] "
                                                 << "min=" << input_min << " max=" << input_max << " mean=" << input_mean << " "
                                                 << "sample[0:5]=[" << input->data()[0] << "," << input->data()[1] << ","
                                                 << input->data()[2] << "," << input->data()[3] << "," << input->data()[4] << "]");

                LOG_DEBUG("[INPUT_DEBUG] Layer " << layer_index_ << " WQ stats: "
                                                 << "size=" << wq_global->size() << " shape=[" << expected_wq_rows << "," << d_model << "] "
                                                 << "min=" << wq_min << " max=" << wq_max << " mean=" << wq_mean << " "
                                                 << "sample[0:5]=[" << wq_global->data()[0] << "," << wq_global->data()[1] << ","
                                                 << wq_global->data()[2] << "," << wq_global->data()[3] << "," << wq_global->data()[4] << "]");
            }
        }

        // Broadcast input to all ranks
        if (world_size > 1)
        {
            MPI_Bcast(input->data(), input->size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
        }

        // ========================================================================
        // STEP 2: Distribute weights by head dimension
        // ========================================================================
        std::shared_ptr<TensorBase> local_wq, local_wk, local_wv, local_wo;
        std::shared_ptr<TensorBase> local_bq, local_bk, local_bv;

        const size_t wq_base_offset = static_cast<size_t>(head_offset) * head_dim_ * d_model;
        const size_t kv_base_offset = static_cast<size_t>(kv_head_offset) * head_dim_ * d_model;
        bool copied_global_weights = false;

        if (weights_are_sharded)
        {
            // Weights are already sharded - use them directly
            local_wq = wq_global;
            local_wk = wk_global;
            local_wv = wv_global;
            local_wo = wo_global;

            // Biases are PRE-SLICED during weight loading (in QwenPipeline)
            // Just use them directly - no hot-path slicing needed
            local_bq = (bq_global && bq_global->size() > 1) ? bq_global : nullptr;
            local_bk = (bk_global && bk_global->size() > 1) ? bk_global : nullptr;
            local_bv = (bv_global && bv_global->size() > 1) ? bv_global : nullptr;

            // DEBUG: Verify bias assignment (sharded path)
            if (layer_index_ == 0)
            {
                LOG_INFO("[BIAS_FLOW] Layer " << layer_index_ << " Rank " << rank
                                              << " SHARDED PATH: weights_are_sharded=true");
                LOG_INFO("[BIAS_FLOW] Layer " << layer_index_ << " Rank " << rank
                                              << " local_bq=" << (local_bq ? "PRESENT" : "nullptr")
                                              << " size=" << (local_bq ? local_bq->size() : 0));
                LOG_INFO("[BIAS_FLOW] Layer " << layer_index_ << " Rank " << rank
                                              << " local_bk=" << (local_bk ? "PRESENT" : "nullptr")
                                              << " size=" << (local_bk ? local_bk->size() : 0));
                LOG_INFO("[BIAS_FLOW] Layer " << layer_index_ << " Rank " << rank
                                              << " local_bv=" << (local_bv ? "PRESENT" : "nullptr")
                                              << " size=" << (local_bv ? local_bv->size() : 0));
            }

            // DEBUG: Verify pointer assignment
            if (layer_index_ == 0)
            {
                LOG_INFO("[PTR_DEBUG] Layer " << layer_index_ << " Rank " << rank << " weight pointer assignment:");
                LOG_INFO("  wq_global ptr: " << (void *)wq_global->data() << " first 5: ["
                                             << wq_global->data()[0] << ", " << wq_global->data()[1] << ", " << wq_global->data()[2] << ", "
                                             << wq_global->data()[3] << ", " << wq_global->data()[4] << "]");
                LOG_INFO("  local_wq ptr: " << (void *)local_wq->data() << " first 5: ["
                                            << local_wq->data()[0] << ", " << local_wq->data()[1] << ", " << local_wq->data()[2] << ", "
                                            << local_wq->data()[3] << ", " << local_wq->data()[4] << "]");
                if (wq_global->data() != local_wq->data())
                {
                    LOG_ERROR("  ❌ POINTER MISMATCH! wq_global and local_wq point to different memory!");
                }
                else
                {
                    LOG_INFO("  ✅ Pointers match");
                }
            }

            // DEBUG: Verify wq_global content for both ranks
            if (layer_index_ == 0)
            {
                LOG_INFO("[WQ_VERIFY] Layer " << layer_index_ << " Rank " << rank << " wq_global stats:");
                float wq_min = *std::min_element(wq_global->data(), wq_global->data() + wq_global->size());
                float wq_max = *std::max_element(wq_global->data(), wq_global->data() + wq_global->size());
                float wq_sum = std::accumulate(wq_global->data(), wq_global->data() + wq_global->size(), 0.0f);
                float wq_mean = wq_sum / wq_global->size();
                LOG_INFO("  wq_global: size=" << wq_global->size() << " min=" << wq_min << " max=" << wq_max << " mean=" << wq_mean);
                LOG_INFO("  wq_global[0:10]: [" << wq_global->data()[0] << ", " << wq_global->data()[1] << ", "
                                                << wq_global->data()[2] << ", " << wq_global->data()[3] << ", " << wq_global->data()[4] << ", "
                                                << wq_global->data()[5] << ", " << wq_global->data()[6] << ", " << wq_global->data()[7] << ", "
                                                << wq_global->data()[8] << ", " << wq_global->data()[9] << "]");
                LOG_INFO("  wq_global[400000:400010]: [" << wq_global->data()[400000] << ", " << wq_global->data()[400001] << ", "
                                                         << wq_global->data()[400002] << ", " << wq_global->data()[400003] << ", " << wq_global->data()[400004] << ", "
                                                         << wq_global->data()[400005] << ", " << wq_global->data()[400006] << ", " << wq_global->data()[400007] << ", "
                                                         << wq_global->data()[400008] << ", " << wq_global->data()[400009] << "]");

                // Ensure rank ordering before saving files
                MPI_Barrier(MPI_COMM_WORLD);

                // Save to file for detailed comparison
                std::string filename = "/tmp/llaminar_wq_layer0_rank" + std::to_string(rank) + ".bin";
                std::ofstream outfile(filename, std::ios::binary);
                outfile.write(reinterpret_cast<const char *>(wq_global->data()), wq_global->size() * sizeof(float));
                outfile.close();
                LOG_INFO("  Saved wq_global to " << filename);

                // Ensure all ranks finish saving before proceeding
                MPI_Barrier(MPI_COMM_WORLD);
            }

            if (rank == 0)
            {
                LOG_DEBUG("Using pre-sharded weights directly (local_head_dim=" << local_head_dim << ")");
                fprintf(stderr, "[kernel-test] rank %d local_wo shape: [%d, %d]\n",
                        rank, (int)local_wo->shape()[0], (int)local_wo->shape()[1]);
                fprintf(stderr, "[kernel-test] rank %d local_wo[0,0:4]: %.6f, %.6f, %.6f, %.6f\n",
                        rank, local_wo->data()[0], local_wo->data()[1],
                        local_wo->data()[2], local_wo->data()[3]);
                fflush(stderr);
            }
        }
        else if (world_size == 1)
        {
            // Single-rank: use global weights directly
            local_wq = wq_global;
            local_wk = wk_global;
            local_wv = wv_global;
            local_wo = wo_global;
            // Only use bias if it's a real bias (size > 1), not a dummy placeholder
            local_bq = (bq_global && bq_global->size() > 1) ? bq_global : nullptr;
            local_bk = (bk_global && bk_global->size() > 1) ? bk_global : nullptr;
            local_bv = (bv_global && bv_global->size() > 1) ? bv_global : nullptr;

            // DEBUG: Verify bias assignment (single-rank path)
            if (layer_index_ == 0)
            {
                LOG_INFO("[BIAS_FLOW] Layer " << layer_index_ << " Rank " << rank
                                              << " SINGLE-RANK PATH: weights_are_sharded=false, world_size=1");
                LOG_INFO("[BIAS_FLOW] Layer " << layer_index_ << " Rank " << rank
                                              << " local_bq=" << (local_bq ? "PRESENT" : "nullptr")
                                              << " size=" << (local_bq ? local_bq->size() : 0));
            }
        }
        else
        {
            // Multi-rank with global weights: slice weights by head dimension
            local_wq = TensorFactory::create_simple({local_head_dim, d_model});
            local_wk = TensorFactory::create_simple({local_kv_head_dim, d_model});
            local_wv = TensorFactory::create_simple({local_kv_head_dim, d_model});
            local_wo = TensorFactory::create_simple({d_model, local_head_dim});

            // Copy weight slices (row-wise slicing for wq/wk/wv, column-wise for wo)
            const size_t local_q_elements = static_cast<size_t>(local_head_dim) * static_cast<size_t>(d_model);
            const size_t local_kv_elements = static_cast<size_t>(local_kv_head_dim) * static_cast<size_t>(d_model);

            memcpy(local_wq->data(), wq_global->data() + wq_base_offset, local_q_elements * sizeof(float));
            memcpy(local_wk->data(), wk_global->data() + kv_base_offset, local_kv_elements * sizeof(float));
            memcpy(local_wv->data(), wv_global->data() + kv_base_offset, local_kv_elements * sizeof(float));
            copied_global_weights = true;

            // Copy output weight (column slice)
            for (int row = 0; row < d_model; ++row)
            {
                const float *src = wo_global->data() + row * (n_head_ * head_dim_) + (head_offset * head_dim_);
                float *dst = local_wo->data() + row * local_head_dim;
                memcpy(dst, src, local_head_dim * sizeof(float));
            }

            // Biases are PRE-SLICED during weight loading (in QwenPipeline)
            // Just use them directly - no hot-path slicing needed
            local_bq = (bq_global && bq_global->size() > 1) ? bq_global : nullptr;
            local_bk = (bk_global && bk_global->size() > 1) ? bk_global : nullptr;
            local_bv = (bv_global && bv_global->size() > 1) ? bv_global : nullptr;

            // DEBUG: Verify bias assignment (multi-rank global weights path)
            if (layer_index_ == 0)
            {
                LOG_INFO("[BIAS_FLOW] Layer " << layer_index_ << " Rank " << rank
                                              << " MULTI-RANK PATH: weights_are_sharded=false, world_size>1");
                LOG_INFO("[BIAS_FLOW] Layer " << layer_index_ << " Rank " << rank
                                              << " local_bq=" << (local_bq ? "PRESENT" : "nullptr")
                                              << " size=" << (local_bq ? local_bq->size() : 0)
                                              << " first_val=" << (local_bq && local_bq->size() > 0 ? local_bq->data()[0] : 0.0f));
                LOG_INFO("[BIAS_FLOW] Layer " << layer_index_ << " Rank " << rank
                                              << " local_bk=" << (local_bk ? "PRESENT" : "nullptr")
                                              << " size=" << (local_bk ? local_bk->size() : 0));
                LOG_INFO("[BIAS_FLOW] Layer " << layer_index_ << " Rank " << rank
                                              << " local_bv=" << (local_bv ? "PRESENT" : "nullptr")
                                              << " size=" << (local_bv ? local_bv->size() : 0));
            }
        }

        if (trace_weight_slice && local_wk && wk_global)
        {
            const size_t stride = static_cast<size_t>(d_model);

            if (copied_global_weights)
            {
                float max_abs_diff = 0.0f;
                const size_t elems = static_cast<size_t>(local_kv_head_dim) * stride;
                const float *global_ptr = wk_global->data() + kv_base_offset;
                const float *local_ptr = local_wk->data();
                for (size_t idx = 0; idx < elems; ++idx)
                {
                    max_abs_diff = std::max(max_abs_diff, std::fabs(local_ptr[idx] - global_ptr[idx]));
                }
                LOG_INFO("[ATTN_WEIGHT_TRACE] rank " << rank
                                                     << " layer=" << layer_index_
                                                     << " kv_head_offset=" << kv_head_offset
                                                     << " local_kv_heads=" << local_kv_heads
                                                     << " head_dim=" << head_dim_
                                                     << " weights_sharded=no"
                                                     << " max_abs_copy_diff=" << max_abs_diff);
            }
            else
            {
                LOG_INFO("[ATTN_WEIGHT_TRACE] rank " << rank
                                                     << " layer=" << layer_index_
                                                     << " kv_head_offset=" << kv_head_offset
                                                     << " local_kv_heads=" << local_kv_heads
                                                     << " head_dim=" << head_dim_
                                                     << " weights_sharded=" << (weights_are_sharded ? "yes" : "no"));
            }

            const bool has_extra_cols = d_model > 33;

            for (int kv = 0; kv < local_kv_heads; ++kv)
            {
                const int global_kv_head = kv_head_offset + kv;
                const int local_row_base = kv * head_dim_;
                const float *local_row0 = local_wk->data() + static_cast<size_t>(local_row_base) * stride;

                LOG_INFO("[ATTN_WEIGHT_TRACE] rank " << rank
                                                     << " layer=" << layer_index_
                                                     << " kv_head_global=" << global_kv_head
                                                     << " row0_cols0-3={" << local_row0[0] << ", " << local_row0[1]
                                                     << ", " << local_row0[2] << ", " << local_row0[3] << "}");
                if (has_extra_cols)
                {
                    LOG_INFO("[ATTN_WEIGHT_TRACE] rank " << rank
                                                         << " layer=" << layer_index_
                                                         << " kv_head_global=" << global_kv_head
                                                         << " row0_cols32-33={" << local_row0[32] << ", " << local_row0[33] << "}");
                }

                if (copied_global_weights)
                {
                    const size_t global_head_row_base = static_cast<size_t>(global_kv_head) * static_cast<size_t>(head_dim_);
                    const float *global_row0 = wk_global->data() + global_head_row_base * stride;
                    LOG_INFO("[ATTN_WEIGHT_TRACE] rank " << rank
                                                         << " layer=" << layer_index_
                                                         << " kv_head_global=" << global_kv_head
                                                         << " GLOBAL_row0_cols0-3={" << global_row0[0] << ", " << global_row0[1]
                                                         << ", " << global_row0[2] << ", " << global_row0[3] << "}");
                    if (has_extra_cols)
                    {
                        LOG_INFO("[ATTN_WEIGHT_TRACE] rank " << rank
                                                             << " layer=" << layer_index_
                                                             << " kv_head_global=" << global_kv_head
                                                             << " GLOBAL_row0_cols32-33={" << global_row0[32] << ", " << global_row0[33] << "}");
                    }
                }

                if (head_dim_ > 32)
                {
                    const float *local_row32 = local_wk->data() + static_cast<size_t>(local_row_base + 32) * stride;
                    LOG_INFO("[ATTN_WEIGHT_TRACE] rank " << rank
                                                         << " layer=" << layer_index_
                                                         << " kv_head_global=" << global_kv_head
                                                         << " row32_cols0-3={" << local_row32[0] << ", " << local_row32[1]
                                                         << ", " << local_row32[2] << ", " << local_row32[3] << "}");
                    if (has_extra_cols)
                    {
                        LOG_INFO("[ATTN_WEIGHT_TRACE] rank " << rank
                                                             << " layer=" << layer_index_
                                                             << " kv_head_global=" << global_kv_head
                                                             << " row32_cols32-33={" << local_row32[32] << ", " << local_row32[33] << "}");
                    }

                    if (copied_global_weights)
                    {
                        const size_t global_row32_index = static_cast<size_t>(global_kv_head) * static_cast<size_t>(head_dim_) + 32;
                        const float *global_row32 = wk_global->data() + global_row32_index * stride;
                        LOG_INFO("[ATTN_WEIGHT_TRACE] rank " << rank
                                                             << " layer=" << layer_index_
                                                             << " kv_head_global=" << global_kv_head
                                                             << " GLOBAL_row32_cols0-3={" << global_row32[0] << ", " << global_row32[1]
                                                             << ", " << global_row32[2] << ", " << global_row32[3] << "}");
                        if (has_extra_cols)
                        {
                            LOG_INFO("[ATTN_WEIGHT_TRACE] rank " << rank
                                                                 << " layer=" << layer_index_
                                                                 << " kv_head_global=" << global_kv_head
                                                                 << " GLOBAL_row32_cols32-33={" << global_row32[32] << ", " << global_row32[33] << "}");
                        }
                    }
                }
            }
        }

        // ========================================================================
        // STEP 3: Compute Q, K, V projections
        // ========================================================================
        auto local_q = TensorFactory::create_simple({seq_len, local_head_dim});
        auto local_k = TensorFactory::create_simple({seq_len, local_kv_head_dim});
        auto local_v = TensorFactory::create_simple({seq_len, local_kv_head_dim});

        // DEBUG: Log input and weight stats before Q projection
        if (layer_index_ == 0)
        {
            LOG_INFO("[Q_PROJ_DEBUG] Layer " << layer_index_ << " Rank " << rank << " BEFORE Q projection:");
            LOG_INFO("  input shape: [" << seq_len << ", " << d_model << "]");
            LOG_INFO("  local_wq expected shape (PyTorch convention): [" << local_head_dim << ", " << d_model << "]");
            LOG_INFO("  Expected output shape: [" << seq_len << ", " << local_head_dim << "]");

            // Input stats
            float input_min = *std::min_element(input->data(), input->data() + input->size());
            float input_max = *std::max_element(input->data(), input->data() + input->size());
            float input_sum = std::accumulate(input->data(), input->data() + input->size(), 0.0f);
            float input_mean = input_sum / input->size();
            LOG_INFO("  input stats: min=" << input_min << " max=" << input_max << " mean=" << input_mean);
            LOG_INFO("  input[0, 0:5]: [" << input->data()[0] << ", " << input->data()[1] << ", "
                                          << input->data()[2] << ", " << input->data()[3] << ", " << input->data()[4] << "]");

            // Weight stats
            float wq_min = *std::min_element(local_wq->data(), local_wq->data() + local_wq->size());
            float wq_max = *std::max_element(local_wq->data(), local_wq->data() + local_wq->size());
            float wq_sum = std::accumulate(local_wq->data(), local_wq->data() + local_wq->size(), 0.0f);
            float wq_mean = wq_sum / local_wq->size();
            LOG_INFO("  local_wq stats: min=" << wq_min << " max=" << wq_max << " mean=" << wq_mean);
            LOG_INFO("  local_wq[0:5]: [" << local_wq->data()[0] << ", " << local_wq->data()[1] << ", "
                                          << local_wq->data()[2] << ", " << local_wq->data()[3] << ", " << local_wq->data()[4] << "]");
        }

        // Q projection: [seq_len, d_model] @ [d_model, local_head_dim] = [seq_len, local_head_dim]
        if (layer_index_ == 0)
        {
            LOG_INFO("[MATMUL_DEBUG] Layer " << layer_index_ << " Rank " << rank << " Q projection matmul parameters:");
            LOG_INFO("  M=" << seq_len << " N=" << local_head_dim << " K=" << d_model);
            LOG_INFO("  input ptr=" << (void *)input->data() << " weight ptr=" << (void *)local_wq->data() << " output ptr=" << (void *)local_q->data());
            LOG_INFO("  local_wq shape: [" << local_wq->shape()[0] << ", " << local_wq->shape()[1] << "]");
            LOG_INFO("  local_wq first 5 values: [" << local_wq->data()[0] << ", " << local_wq->data()[1] << ", " << local_wq->data()[2] << ", " << local_wq->data()[3] << ", " << local_wq->data()[4] << "]");
            LOG_INFO("  local_wq offset 896 values (2nd row if [448,896]): [" << local_wq->data()[896] << ", " << local_wq->data()[897] << ", " << local_wq->data()[898] << ", " << local_wq->data()[899] << ", " << local_wq->data()[900] << "]");
            LOG_INFO("  input first 5 values: [" << input->data()[0] << ", " << input->data()[1] << ", " << input->data()[2] << ", " << input->data()[3] << ", " << input->data()[4] << "]");
            LOG_INFO("  local_wq size: " << local_wq->size());
        }

        // DEBUG: Verify local_bq status before Q projection
        if (layer_index_ == 0)
        {
            LOG_INFO("[BIAS_FLOW] Layer " << layer_index_ << " Rank " << rank
                                          << " BEFORE Q_projection call:");
            LOG_INFO("[BIAS_FLOW] local_bq=" << (local_bq ? "PRESENT" : "nullptr")
                                             << " size=" << (local_bq ? local_bq->size() : 0)
                                             << " will_pass=" << (local_bq ? "local_bq->data()" : "nullptr"));
            if (local_bq && local_bq->size() > 0)
            {
                LOG_INFO("[BIAS_FLOW] local_bq[0:3]: [" << local_bq->data()[0] << ", "
                                                        << local_bq->data()[1] << ", " << local_bq->data()[2] << "]");
            }
        }

        matmul_with_bias(input->data(), local_wq->data(), local_q->data(),
                         local_bq ? local_bq->data() : nullptr,
                         seq_len, local_head_dim, d_model, "Q_projection");

        // DEBUG: Log Q projection output
        if (layer_index_ == 0)
        {
            LOG_INFO("[Q_PROJ_DEBUG] Layer " << layer_index_ << " Rank " << rank << " AFTER Q projection:");
            float q_min = *std::min_element(local_q->data(), local_q->data() + local_q->size());
            float q_max = *std::max_element(local_q->data(), local_q->data() + local_q->size());
            float q_sum = std::accumulate(local_q->data(), local_q->data() + local_q->size(), 0.0f);
            float q_mean = q_sum / local_q->size();
            LOG_INFO("  local_q stats: min=" << q_min << " max=" << q_max << " mean=" << q_mean);
            LOG_INFO("  local_q[0, 0:10]: [" << local_q->data()[0] << ", " << local_q->data()[1] << ", "
                                             << local_q->data()[2] << ", " << local_q->data()[3] << ", " << local_q->data()[4] << ", "
                                             << local_q->data()[5] << ", " << local_q->data()[6] << ", " << local_q->data()[7] << ", "
                                             << local_q->data()[8] << ", " << local_q->data()[9] << "]");
        }

        // K projection: [seq_len, d_model] @ [d_model, local_kv_head_dim] = [seq_len, local_kv_head_dim]
        matmul_with_bias(input->data(), local_wk->data(), local_k->data(),
                         local_bk ? local_bk->data() : nullptr,
                         seq_len, local_kv_head_dim, d_model, "K_projection");

        // V projection: [seq_len, d_model] @ [d_model, local_kv_head_dim] = [seq_len, local_kv_head_dim]
        matmul_with_bias(input->data(), local_wv->data(), local_v->data(),
                         local_bv ? local_bv->data() : nullptr,
                         seq_len, local_kv_head_dim, d_model, "V_projection");

        // CONTRACT: QKV projections
        if (enable_validation)
        {
            // CHECK: Q before contract validation
            {
                float *q_ptr = local_q->data();
                bool q_corrupted = false;
                for (int i = 0; i < local_q->size(); ++i)
                {
                    if (std::abs(q_ptr[i]) > 1e10f)
                    {
                        std::cerr << "[CHECK-BEFORE-CONTRACT] ⚠️ Q CORRUPTED before contract! Garbage at index " << i << ": " << q_ptr[i] << std::endl;
                        q_corrupted = true;
                        break;
                    }
                }
                if (!q_corrupted)
                {
                    std::cerr << "[CHECK-BEFORE-CONTRACT] Q clean before contract validation" << std::endl;
                }
            }

            StageContract qkv_contract("QKV_Projections");
            qkv_contract.outputs = {
                TensorContract("local_q", ShapeSpec({seq_len, local_head_dim}),
                               TensorLayout::HeadInterleaved, TensorSemantic::Activation),
                TensorContract("local_k", ShapeSpec({seq_len, local_kv_head_dim}),
                               TensorLayout::HeadInterleaved, TensorSemantic::Activation),
                TensorContract("local_v", ShapeSpec({seq_len, local_kv_head_dim}),
                               TensorLayout::HeadInterleaved, TensorSemantic::Activation)};

            try
            {
                qkv_contract.validate_outputs({local_q, local_k, local_v});

                // CHECK: Q after contract validation
                {
                    float *q_ptr = local_q->data();
                    bool q_corrupted = false;
                    for (int i = 0; i < local_q->size(); ++i)
                    {
                        if (std::abs(q_ptr[i]) > 1e10f)
                        {
                            std::cerr << "[CHECK-AFTER-CONTRACT] ⚠️ Q CORRUPTED after contract! Garbage at index " << i << ": " << q_ptr[i] << std::endl;
                            q_corrupted = true;
                            break;
                        }
                    }
                    if (!q_corrupted)
                    {
                        std::cerr << "[CHECK-AFTER-CONTRACT] Q clean after contract validation" << std::endl;
                    }
                }
                if (rank == 0)
                    LOG_DEBUG("✓ QKV projection shape contracts validated");
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("QKV projection contract violation: " << e.what());
                return false;
            }

            // HEALTH CHECK: Verify projections produced valid values
            if (rank == 0)
            {
                TensorHealthCheck q_health("local_q");
                q_health.check(local_q->data(), local_q->size());
                q_health.log(rank);

                TensorHealthCheck k_health("local_k");
                k_health.check(local_k->data(), local_k->size());
                k_health.log(rank);

                TensorHealthCheck v_health("local_v");
                v_health.check(local_v->data(), local_v->size());
                v_health.log(rank);

                if (!q_health.is_healthy() || !k_health.is_healthy() || !v_health.is_healthy())
                {
                    LOG_ERROR("❌ QKV projections produced NaN/Inf - matrix multiplication failed!");
                    return false;
                }
            }

            // OPTIONAL: Validate against scalar reference implementation
            if (validate_projections && rank == 0)
            {
                auto q_result = llaminar::attention::AttentionValidator::validateProjection(
                    input->data(), local_wq->data(), local_q->data(),
                    seq_len, local_head_dim, d_model, true);

                if (!llaminar::attention::AttentionValidator::isWithinTolerance(q_result, 1e-4, 1e-4))
                {
                    LOG_WARN("Q projection divergence: max_abs=" << q_result.max_abs
                                                                 << ", rel_l2=" << q_result.rel_l2);
                }
                else
                {
                    LOG_DEBUG("✓ Q projection validated against scalar reference");
                }
            }
        }

        // Snapshot Q/K/V projections if callback is set
        if (snapshot_callback_)
        {
            // Add instrumentation to verify values before snapshotting
            if (rank == 0)
            {
                float q_min = *std::min_element(local_q->data(), local_q->data() + local_q->size());
                float q_max = *std::max_element(local_q->data(), local_q->data() + local_q->size());
                float q_sum = std::accumulate(local_q->data(), local_q->data() + local_q->size(), 0.0f);
                float q_mean = q_sum / local_q->size();

                float k_min = *std::min_element(local_k->data(), local_k->data() + local_k->size());
                float k_max = *std::max_element(local_k->data(), local_k->data() + local_k->size());
                float k_sum = std::accumulate(local_k->data(), local_k->data() + local_k->size(), 0.0f);
                float k_mean = k_sum / local_k->size();

                float v_min = *std::min_element(local_v->data(), local_v->data() + local_v->size());
                float v_max = *std::max_element(local_v->data(), local_v->data() + local_v->size());
                float v_sum = std::accumulate(local_v->data(), local_v->data() + local_v->size(), 0.0f);
                float v_mean = v_sum / local_v->size();

                LOG_DEBUG("[SNAPSHOT_DEBUG] Layer " << layer_index_ << " Q projection stats: "
                                                    << "size=" << local_q->size() << " "
                                                    << "min=" << q_min << " max=" << q_max << " mean=" << q_mean << " "
                                                    << "sample[0:5]=[" << local_q->data()[0] << "," << local_q->data()[1] << ","
                                                    << local_q->data()[2] << "," << local_q->data()[3] << "," << local_q->data()[4] << "]");

                LOG_DEBUG("[SNAPSHOT_DEBUG] Layer " << layer_index_ << " K projection stats: "
                                                    << "size=" << local_k->size() << " "
                                                    << "min=" << k_min << " max=" << k_max << " mean=" << k_mean << " "
                                                    << "sample[0:5]=[" << local_k->data()[0] << "," << local_k->data()[1] << ","
                                                    << local_k->data()[2] << "," << local_k->data()[3] << "," << local_k->data()[4] << "]");

                LOG_DEBUG("[SNAPSHOT_DEBUG] Layer " << layer_index_ << " V projection stats: "
                                                    << "size=" << local_v->size() << " "
                                                    << "min=" << v_min << " max=" << v_max << " mean=" << v_mean << " "
                                                    << "sample[0:5]=[" << local_v->data()[0] << "," << local_v->data()[1] << ","
                                                    << local_v->data()[2] << "," << local_v->data()[3] << "," << local_v->data()[4] << "]");
            }
        }
        else
        {
            if (rank == 0)
            {
                LOG_WARN("[SNAPSHOT_DEBUG] Layer " << layer_index_ << " snapshot_callback_ is NULL - snapshots NOT captured!");
            }
        }

        // ========================================================================
        // STEP 4: Gather Q/K/V for snapshotting (BEFORE RoPE!)
        // ========================================================================
        // For multi-rank execution, we need to gather Q/K/V across all ranks before snapshotting
        // This is because PyTorch reference has the full Q/K/V, not sharded versions
        //
        // CRITICAL: PyTorch captures Q_PROJECTION/K_PROJECTION BEFORE RoPE is applied,
        // so we must gather and snapshot BEFORE applying RoPE to match the reference.
        //
        // We gather row-by-row (per token) to maintain proper layout
        // Each rank has [seq_len, local_head_dim] in row-major order
        // We need to concatenate the head dimension per row to get [seq_len, total_head_dim]

        LOG_DEBUG("[MPIAttentionKernel] Layer " << layer_index_ << ": snapshot_callback_="
                                                << (snapshot_callback_ ? "SET" : "NULL") << ", world_size=" << world_size);

        if (snapshot_callback_ && world_size > 1)
        {
            // Allocate global tensors
            auto global_q = TensorFactory::create_simple({seq_len, n_head_ * head_dim_});
            auto global_k = TensorFactory::create_simple({seq_len, n_head_kv_ * head_dim_});
            auto global_v = TensorFactory::create_simple({seq_len, n_head_kv_ * head_dim_});

            // DEBUG: Log local Q before gather
            if (layer_index_ == 0)
            {
                LOG_INFO("[Q_GATHER_DEBUG] Layer " << layer_index_ << " Rank " << rank << " BEFORE gather:");
                LOG_INFO("  local_q shape: [" << seq_len << ", " << local_head_dim << "]");
                LOG_INFO("  local_head_dim: " << local_head_dim << " (= " << local_heads << " heads * " << head_dim_ << " dims)");
                LOG_INFO("  Expected rank " << rank << " heads: [" << (rank * local_heads) << ", " << ((rank + 1) * local_heads) << ")");
                LOG_INFO("  local_q[t=0, first 10]: ["
                         << local_q->data()[0] << ", " << local_q->data()[1] << ", " << local_q->data()[2] << ", "
                         << local_q->data()[3] << ", " << local_q->data()[4] << ", " << local_q->data()[5] << ", "
                         << local_q->data()[6] << ", " << local_q->data()[7] << ", " << local_q->data()[8] << ", "
                         << local_q->data()[9] << "]");

                // Check head boundaries (first value of each head)
                for (int h = 0; h < local_heads && h < 3; ++h)
                {
                    int global_head_idx = rank * local_heads + h;
                    int offset = h * head_dim_;
                    LOG_INFO("  Head " << global_head_idx << " (local head " << h << ") first 5 dims: ["
                                       << local_q->data()[offset] << ", " << local_q->data()[offset + 1] << ", "
                                       << local_q->data()[offset + 2] << ", " << local_q->data()[offset + 3] << ", "
                                       << local_q->data()[offset + 4] << "]");
                }
            }

            // Gather row-by-row to maintain proper layout
            // For Q: gather local_head_dim elements per row from each rank
            for (int t = 0; t < seq_len; ++t)
            {
                const float *local_q_row = local_q->data() + t * local_head_dim;
                float *global_q_row = global_q->data() + t * (n_head_ * head_dim_);

                // Gather this row from all ranks, placing each rank's data at the correct offset
                MPI_Allgather(local_q_row, local_head_dim, MPI_FLOAT,
                              global_q_row, local_head_dim, MPI_FLOAT,
                              MPI_COMM_WORLD);
            }

            // DEBUG: Log global Q after gather
            if (layer_index_ == 0)
            {
                LOG_INFO("[Q_GATHER_DEBUG] Layer " << layer_index_ << " Rank " << rank << " AFTER gather:");
                LOG_INFO("  global_q shape: [" << seq_len << ", " << (n_head_ * head_dim_) << "]");
                LOG_INFO("  global_q[t=0, first 10]: ["
                         << global_q->data()[0] << ", " << global_q->data()[1] << ", " << global_q->data()[2] << ", "
                         << global_q->data()[3] << ", " << global_q->data()[4] << ", " << global_q->data()[5] << ", "
                         << global_q->data()[6] << ", " << global_q->data()[7] << ", " << global_q->data()[8] << ", "
                         << global_q->data()[9] << "]");

                // Check each head's contribution in global tensor
                for (int h = 0; h < n_head_ && h < 10; ++h)
                {
                    int offset = h * head_dim_;
                    int source_rank = h / local_heads;
                    LOG_INFO("  Global head " << h << " (from rank " << source_rank << ") first 5 dims: ["
                                              << global_q->data()[offset] << ", " << global_q->data()[offset + 1] << ", "
                                              << global_q->data()[offset + 2] << ", " << global_q->data()[offset + 3] << ", "
                                              << global_q->data()[offset + 4] << "]");
                }

                // Critical check: verify heads 7 and 8 (boundary between ranks)
                LOG_INFO("  CRITICAL CHECK - Head boundary (rank 0 last head vs rank 1 first head):");
                int h7_offset = 7 * head_dim_;
                int h8_offset = 8 * head_dim_;
                LOG_INFO("    Head 7 (rank 0): [" << global_q->data()[h7_offset] << ", " << global_q->data()[h7_offset + 1] << ", " << global_q->data()[h7_offset + 2] << "]");
                LOG_INFO("    Head 8 (rank 1): [" << global_q->data()[h8_offset] << ", " << global_q->data()[h8_offset + 1] << ", " << global_q->data()[h8_offset + 2] << "]");
            }

            // Same for K
            for (int t = 0; t < seq_len; ++t)
            {
                const float *local_k_row = local_k->data() + t * local_kv_head_dim;
                float *global_k_row = global_k->data() + t * (n_head_kv_ * head_dim_);

                MPI_Allgather(local_k_row, local_kv_head_dim, MPI_FLOAT,
                              global_k_row, local_kv_head_dim, MPI_FLOAT,
                              MPI_COMM_WORLD);
            }

            // Same for V
            for (int t = 0; t < seq_len; ++t)
            {
                const float *local_v_row = local_v->data() + t * local_kv_head_dim;
                float *global_v_row = global_v->data() + t * (n_head_kv_ * head_dim_);

                MPI_Allgather(local_v_row, local_kv_head_dim, MPI_FLOAT,
                              global_v_row, local_kv_head_dim, MPI_FLOAT,
                              MPI_COMM_WORLD);
            }

            // Snapshot the gathered global tensors (only rank 0 needs to snapshot)
            // CRITICAL: These are Q/K/V BEFORE RoPE to match PyTorch's Q_PROJECTION stage
            if (rank == 0)
            {
                snapshot_callback_(PipelineStage::Q_PROJECTION, layer_index_, global_q->data(), seq_len, n_head_ * head_dim_);
                snapshot_callback_(PipelineStage::K_PROJECTION, layer_index_, global_k->data(), seq_len, n_head_kv_ * head_dim_);
                snapshot_callback_(PipelineStage::V_PROJECTION, layer_index_, global_v->data(), seq_len, n_head_kv_ * head_dim_);
            }
        }
        else if (snapshot_callback_)
        {
            // Single rank: just snapshot the local tensors directly (also before RoPE)
            snapshot_callback_(PipelineStage::Q_PROJECTION, layer_index_, local_q->data(), seq_len, local_head_dim);
            snapshot_callback_(PipelineStage::K_PROJECTION, layer_index_, local_k->data(), seq_len, local_kv_head_dim);
            snapshot_callback_(PipelineStage::V_PROJECTION, layer_index_, local_v->data(), seq_len, local_kv_head_dim);
        }

        // ========================================================================
        // STEP 5: Apply RoPE to Q and K (AFTER snapshotting but BEFORE attention!)
        // ========================================================================
        // PyTorch applies RoPE AFTER capturing Q_PROJECTION but BEFORE computing attention.
        // We do the same: snapshot first (above), then apply RoPE to our local shard.
        // This ensures attention is computed from RoPE-rotated tensors while Q_PROJECTION
        // snapshots match PyTorch's pre-RoPE values.

        if (trace_k_projection && layer_index_ == 0)
        {
            const int log_layer = layer_index_;
            const auto log_sample = [&](const std::string &label, const float *ptr, int cols)
            {
                std::ostringstream oss;
                oss << label << "{";
                const int preview = std::min(cols, 10);
                for (int i = 0; i < preview; ++i)
                {
                    if (i > 0)
                        oss << ", ";
                    oss << ptr[i];
                }
                if (cols > 10)
                {
                    oss << ", ...";
                }
                oss << "}";
                LOG_INFO(oss.str());
            };

            LOG_INFO("[ATTN_K_TRACE] rank=" << rank
                                            << " layer=" << log_layer
                                            << " local_heads=" << local_heads
                                            << " local_kv_heads=" << local_kv_heads
                                            << " seq_len=" << seq_len
                                            << " head_dim=" << head_dim_);

            // Token 0 preview (first local head)
            log_sample("  local_q[token0_head0]=", local_q->data(), head_dim_);
            log_sample("  local_k[token0_head0]=", local_k->data(), head_dim_);

            if (seq_len > 1)
            {
                log_sample("  local_q[token1_head0]=", local_q->data() + local_head_dim, head_dim_);
                log_sample("  local_k[token1_head0]=", local_k->data() + local_kv_head_dim, head_dim_);
            }

            if (local_q->size() >= head_dim_ * 2)
            {
                const int mid = static_cast<int>(local_q->size() / 2);
                const int tail = static_cast<int>(local_q->size() - std::min(local_head_dim, head_dim_));
                log_sample("  local_q[mid_segment]=", local_q->data() + mid, head_dim_);
                log_sample("  local_q[tail_segment]=", local_q->data() + tail, head_dim_);
            }

            if (local_k->size() >= head_dim_ * 2)
            {
                const int mid = static_cast<int>(local_k->size() / 2);
                const int tail = static_cast<int>(local_k->size() - std::min(local_kv_head_dim, head_dim_));
                log_sample("  local_k[mid_segment]=", local_k->data() + mid, head_dim_);
                log_sample("  local_k[tail_segment]=", local_k->data() + tail, head_dim_);
            }
        }

        // =======================================================================
        // ROPE PARAMETER VALIDATION AND PRE-ROPE VALUE LOGGING
        // =======================================================================
        if (rank == 0 && layer_index_ == 0)
        {
            LOG_DEBUG("========== ROPE_APPLICATION DEBUG STEP 1: PRE-ROPE VALIDATION ==========");
            LOG_DEBUG("[ROPE_PARAMS] Parameters being passed to apply_rope():");
            LOG_INFO("  seq_len: " << seq_len << " (expected: 5 for test prompt)");
            LOG_INFO("  head_dim: " << head_dim_ << " (expected: 64 for Qwen-0.5B)");
            LOG_INFO("  local_heads (q_heads param): " << local_heads << " (expected: 7 per rank for 14 total)");
            LOG_INFO("  local_kv_heads (k_heads param): " << local_kv_heads << " (expected: 1 per rank for 2 total)");
            LOG_INFO("  n_past: " << n_past_ << " (KV cache position)");
            LOG_INFO("  rope_freq_base: " << rope_freq_base_ << " (expected: 10000)");

            LOG_DEBUG("[ROPE_TENSORS] Tensor shapes before RoPE:");
            LOG_INFO("  local_q size: " << local_q->size() << " shape: [" << seq_len << ", " << local_head_dim << "]");
            LOG_INFO("  local_k size: " << local_k->size() << " shape: [" << seq_len << ", " << local_kv_head_dim << "]");
            LOG_INFO("  local_head_dim: " << local_head_dim << " = " << local_heads << " heads * " << head_dim_ << " dims");
            LOG_INFO("  local_kv_head_dim: " << local_kv_head_dim << " = " << local_kv_heads << " heads * " << head_dim_ << " dims");

            LOG_DEBUG("[PRE_ROPE_Q] Token 0, head 0, first 10 dims: ["
                      << local_q->data()[0] << ", " << local_q->data()[1] << ", "
                      << local_q->data()[2] << ", " << local_q->data()[3] << ", "
                      << local_q->data()[4] << ", " << local_q->data()[5] << ", "
                      << local_q->data()[6] << ", " << local_q->data()[7] << ", "
                      << local_q->data()[8] << ", " << local_q->data()[9] << "]");

            // Check position 0 values - these should be close to Q_PROJECTION outputs
            LOG_DEBUG("[PRE_ROPE_Q] Token 0, head 0, dims [2,3,4] (key for comparison): "
                      << local_q->data()[2] << ", " << local_q->data()[3] << ", " << local_q->data()[4]);

            LOG_DEBUG("[PRE_ROPE_K] Token 0, head 0, first 10 dims: ["
                      << local_k->data()[0] << ", " << local_k->data()[1] << ", "
                      << local_k->data()[2] << ", " << local_k->data()[3] << ", "
                      << local_k->data()[4] << ", " << local_k->data()[5] << ", "
                      << local_k->data()[6] << ", " << local_k->data()[7] << ", "
                      << local_k->data()[8] << ", " << local_k->data()[9] << "]");
        }

        llaminar::attn::apply_rope(local_q->data(), local_k->data(),
                                   seq_len, head_dim_, local_heads, local_kv_heads,
                                   n_past_, rope_freq_base_);

        // =======================================================================
        // POST-ROPE VALUE LOGGING
        // =======================================================================
        if (rank == 0 && layer_index_ == 0)
        {
            LOG_DEBUG("========== ROPE_APPLICATION DEBUG STEP 2: POST-ROPE VALUES ==========");
            LOG_DEBUG("[POST_ROPE_Q] Token 0, head 0, first 10 dims: ["
                      << local_q->data()[0] << ", " << local_q->data()[1] << ", "
                      << local_q->data()[2] << ", " << local_q->data()[3] << ", "
                      << local_q->data()[4] << ", " << local_q->data()[5] << ", "
                      << local_q->data()[6] << ", " << local_q->data()[7] << ", "
                      << local_q->data()[8] << ", " << local_q->data()[9] << "]");

            // At position 0, RoPE should be identity (angle=0, cos=1, sin=0)
            // So post-RoPE values should match pre-RoPE values
            LOG_DEBUG("[POST_ROPE_Q] Token 0 CHECK: dims [2,3,4] should match pre-RoPE: "
                      << local_q->data()[2] << ", " << local_q->data()[3] << ", " << local_q->data()[4]);

            LOG_DEBUG("[POST_ROPE_K] Token 0, head 0, first 10 dims: ["
                      << local_k->data()[0] << ", " << local_k->data()[1] << ", "
                      << local_k->data()[2] << ", " << local_k->data()[3] << ", "
                      << local_k->data()[4] << ", " << local_k->data()[5] << ", "
                      << local_k->data()[6] << ", " << local_k->data()[7] << ", "
                      << local_k->data()[8] << ", " << local_k->data()[9] << "]");

            // Check token 1 to verify rotation happened
            int token1_offset = local_head_dim;
            LOG_DEBUG("[POST_ROPE_Q] Token 1, head 0, first 10 dims (should differ from Token 0): ["
                      << local_q->data()[token1_offset + 0] << ", " << local_q->data()[token1_offset + 1] << ", "
                      << local_q->data()[token1_offset + 2] << ", " << local_q->data()[token1_offset + 3] << ", "
                      << local_q->data()[token1_offset + 4] << ", " << local_q->data()[token1_offset + 5] << ", "
                      << local_q->data()[token1_offset + 6] << ", " << local_q->data()[token1_offset + 7] << ", "
                      << local_q->data()[token1_offset + 8] << ", " << local_q->data()[token1_offset + 9] << "]");
        }

        if (layer_index_ == 0)
        {
            LOG_DEBUG("[RANK=" << rank << "] AFTER RoPE:");
            LOG_INFO("  local_q[token=0, head=0, dim=0:10]: ["
                     << local_q->data()[0] << ", " << local_q->data()[1] << ", "
                     << local_q->data()[2] << ", " << local_q->data()[3] << ", "
                     << local_q->data()[4] << ", " << local_q->data()[5] << ", "
                     << local_q->data()[6] << ", " << local_q->data()[7] << ", "
                     << local_q->data()[8] << ", " << local_q->data()[9] << "]");
            // Check token 1 (t=1) which SHOULD be rotated (pos=1, non-zero angle)
            int token1_offset = local_head_dim; // Start of second token
            LOG_INFO("  local_q[token=1, head=0, dim=0:10]: ["
                     << local_q->data()[token1_offset + 0] << ", " << local_q->data()[token1_offset + 1] << ", "
                     << local_q->data()[token1_offset + 2] << ", " << local_q->data()[token1_offset + 3] << ", "
                     << local_q->data()[token1_offset + 4] << ", " << local_q->data()[token1_offset + 5] << ", "
                     << local_q->data()[token1_offset + 6] << ", " << local_q->data()[token1_offset + 7] << ", "
                     << local_q->data()[token1_offset + 8] << ", " << local_q->data()[token1_offset + 9] << "]");
            LOG_INFO("  local_k[token=0, head=0, dim=0:10]: ["
                     << local_k->data()[0] << ", " << local_k->data()[1] << ", "
                     << local_k->data()[2] << ", " << local_k->data()[3] << ", "
                     << local_k->data()[4] << ", " << local_k->data()[5] << ", "
                     << local_k->data()[6] << ", " << local_k->data()[7] << ", "
                     << local_k->data()[8] << ", " << local_k->data()[9] << "]");
            int k_token1_offset = local_kv_head_dim;
            LOG_INFO("  local_k[token=1, head=0, dim=0:10]: ["
                     << local_k->data()[k_token1_offset + 0] << ", " << local_k->data()[k_token1_offset + 1] << ", "
                     << local_k->data()[k_token1_offset + 2] << ", " << local_k->data()[k_token1_offset + 3] << ", "
                     << local_k->data()[k_token1_offset + 4] << ", " << local_k->data()[k_token1_offset + 5] << ", "
                     << local_k->data()[k_token1_offset + 6] << ", " << local_k->data()[k_token1_offset + 7] << ", "
                     << local_k->data()[k_token1_offset + 8] << ", " << local_k->data()[k_token1_offset + 9] << "]");
        }

        // CONTRACT: RoPE application
        if (enable_validation && rank == 0)
        {
            TensorHealthCheck q_rope_health("Q_after_RoPE");
            q_rope_health.check(local_q->data(), local_q->size());
            q_rope_health.log(rank);

            TensorHealthCheck k_rope_health("K_after_RoPE");
            k_rope_health.check(local_k->data(), local_k->size());
            k_rope_health.log(rank);
        }

        // ========================================================================
        // STEP 5.1: UPDATE KV CACHE (after RoPE, before attention)
        // ========================================================================
        // For decode: append new K/V to existing cache
        // For prefill: initialize cache with current K/V
        std::shared_ptr<TensorBase> local_k_cache, local_v_cache;
        int attn_seq_len; // Total sequence length for attention (n_past + seq_len)

        if (is_decode_mode)
        {
            // DECODE MODE: Append new K/V to existing cache
            attn_seq_len = cache_seq_len + seq_len; // n_past + 1

            // DEBUG: Check input cache
            if (rank == 0 && layer_index_ == 0)
            {
                LOG_INFO("[CACHE_DEBUG] Input K cache (first 10 of first row): "
                         << k_cache_in->data()[0] << " " << k_cache_in->data()[1] << " "
                         << k_cache_in->data()[2] << " " << k_cache_in->data()[3] << " "
                         << k_cache_in->data()[4] << " " << k_cache_in->data()[5] << " "
                         << k_cache_in->data()[6] << " " << k_cache_in->data()[7] << " "
                         << k_cache_in->data()[8] << " " << k_cache_in->data()[9]);
                LOG_INFO("  Cache shape: [" << k_cache_in->shape()[0] << ", " << k_cache_in->shape()[1] << "]");
            }

            local_k_cache = TensorFactory::create_simple({attn_seq_len, local_kv_head_dim});
            local_v_cache = TensorFactory::create_simple({attn_seq_len, local_kv_head_dim});

            // Copy existing cache
            std::memcpy(local_k_cache->data(), k_cache_in->data(),
                        cache_seq_len * local_kv_head_dim * sizeof(float));
            std::memcpy(local_v_cache->data(), v_cache_in->data(),
                        cache_seq_len * local_kv_head_dim * sizeof(float));

            // Append new K/V (after RoPE rotation)
            std::memcpy(local_k_cache->data() + cache_seq_len * local_kv_head_dim,
                        local_k->data(),
                        seq_len * local_kv_head_dim * sizeof(float));
            std::memcpy(local_v_cache->data() + cache_seq_len * local_kv_head_dim,
                        local_v->data(),
                        seq_len * local_kv_head_dim * sizeof(float));

            if (rank == 0 && layer_index_ == 0)
            {
                LOG_INFO("[KV_CACHE] Decode: appended " << seq_len << " new tokens to cache");
                LOG_INFO("  Old cache size: " << cache_seq_len << ", New cache size: " << attn_seq_len);
                LOG_INFO("  Cache shape: [" << attn_seq_len << ", " << local_kv_head_dim << "]");

                LOG_INFO("[CACHE_DEBUG] Updated K cache (first 10 of first 3 rows):");
                for (int t = 0; t < std::min(3, attn_seq_len); ++t)
                {
                    int offset = t * local_kv_head_dim;
                    LOG_INFO("  Row " << t << ": "
                                      << local_k_cache->data()[offset + 0] << " " << local_k_cache->data()[offset + 1] << " "
                                      << local_k_cache->data()[offset + 2] << " " << local_k_cache->data()[offset + 3] << " "
                                      << local_k_cache->data()[offset + 4] << " " << local_k_cache->data()[offset + 5] << " "
                                      << local_k_cache->data()[offset + 6] << " " << local_k_cache->data()[offset + 7] << " "
                                      << local_k_cache->data()[offset + 8] << " " << local_k_cache->data()[offset + 9]);
                }
            }
        }
        else
        {
            // PREFILL MODE: Initialize cache with current K/V
            attn_seq_len = seq_len;
            local_k_cache = local_k; // Share the tensor (no copy needed)
            local_v_cache = local_v;

            if (rank == 0 && layer_index_ == 0)
            {
                LOG_INFO("[KV_CACHE] Prefill: initialized cache with " << seq_len << " tokens");
                LOG_INFO("  Cache shape: [" << attn_seq_len << ", " << local_kv_head_dim << "]");

                LOG_INFO("[CACHE_DEBUG] Prefill K cache (first 10 of first 3 rows):");
                for (int t = 0; t < std::min(3, attn_seq_len); ++t)
                {
                    int offset = t * local_kv_head_dim;
                    LOG_INFO("  Row " << t << ": "
                                      << local_k_cache->data()[offset + 0] << " " << local_k_cache->data()[offset + 1] << " "
                                      << local_k_cache->data()[offset + 2] << " " << local_k_cache->data()[offset + 3] << " "
                                      << local_k_cache->data()[offset + 4] << " " << local_k_cache->data()[offset + 5] << " "
                                      << local_k_cache->data()[offset + 6] << " " << local_k_cache->data()[offset + 7] << " "
                                      << local_k_cache->data()[offset + 8] << " " << local_k_cache->data()[offset + 9]);
                }
            }
        }

        // STEP 5.5: Capture ROPE_APPLICATION snapshot (post-RoPE Q and K)
        // Gather Q, K, and V after RoPE application for comparison with PyTorch
        // IMPORTANT: We also need to gather K/V for GQA expansion later
        std::shared_ptr<TensorBase> global_q_rope, global_k_rope, global_v_rope;

        if (snapshot_callback_ || n_head_ != n_head_kv_)
        {
            // Calculate dimensions
            const int k_v_dim = n_head_kv_ * head_dim_;

            if (rank == 0 && layer_index_ == 0)
            {
                LOG_DEBUG("[ROPE_SNAPSHOT_DEBUG] Gathering ROPE_APPLICATION:");
                LOG_INFO("  world_size=" << world_size << " local_heads=" << local_heads
                                         << " local_kv_heads=" << local_kv_heads);
                LOG_INFO("  n_head_=" << n_head_ << " n_head_kv_=" << n_head_kv_ << " head_dim_=" << head_dim_);
                LOG_INFO("  local_head_dim=" << local_head_dim << " local_kv_head_dim=" << local_kv_head_dim);
                LOG_INFO("  d_model_=" << d_model_ << " k_v_dim=" << k_v_dim);
                LOG_INFO("  Expected global Q shape: [" << seq_len << ", " << d_model_ << "]");
                LOG_INFO("  Expected global K shape: [" << seq_len << ", " << k_v_dim << "]");
                LOG_INFO("  local_q shape: [" << seq_len << ", " << local_head_dim << "]");
                LOG_INFO("  local_k shape: [" << seq_len << ", " << local_kv_head_dim << "]");
            }

            // Gather Q, K, and V (post-RoPE) from all ranks
            if (world_size > 1)
            {
                global_q_rope = TensorFactory::create_simple({seq_len, d_model_});
                global_k_rope = TensorFactory::create_simple({seq_len, k_v_dim});
                global_v_rope = TensorFactory::create_simple({seq_len, k_v_dim});

                // DEBUG: Check local Q values before gather
                if (rank == 0 && layer_index_ == 0)
                {
                    std::ostringstream oss;
                    oss << "[ROPE_SNAPSHOT_DEBUG] Rank 0 local_q[t=0] first 10 values: ";
                    for (int i = 0; i < 10 && i < local_head_dim; ++i)
                    {
                        oss << local_q->data()[i] << " ";
                    }
                    LOG_DEBUG(oss.str());
                }

                // DEBUG: Log local_k before gathering (layer 0 only)
                if (layer_index_ == 0)
                {
                    LOG_INFO("[RANK=" << rank << "] Before gather, local_k[t=0, h=0] first 5: "
                                      << local_k->data()[0] << ", " << local_k->data()[1] << ", "
                                      << local_k->data()[2] << ", " << local_k->data()[3] << ", " << local_k->data()[4]);
                }

                // Gather row-by-row to maintain proper layout
                for (int t = 0; t < seq_len; ++t)
                {
                    const float *local_q_row = local_q->data() + t * local_head_dim;
                    const float *local_k_row = local_k->data() + t * local_kv_head_dim;
                    const float *local_v_row = local_v->data() + t * local_kv_head_dim;
                    float *global_q_row = global_q_rope->data() + t * d_model_;
                    float *global_k_row = global_k_rope->data() + t * k_v_dim;
                    float *global_v_row = global_v_rope->data() + t * k_v_dim;

                    MPI_Allgather(local_q_row, local_head_dim, MPI_FLOAT,
                                  global_q_row, local_head_dim, MPI_FLOAT,
                                  MPI_COMM_WORLD);

                    MPI_Allgather(local_k_row, local_kv_head_dim, MPI_FLOAT,
                                  global_k_row, local_kv_head_dim, MPI_FLOAT,
                                  MPI_COMM_WORLD);

                    MPI_Allgather(local_v_row, local_kv_head_dim, MPI_FLOAT,
                                  global_v_row, local_kv_head_dim, MPI_FLOAT,
                                  MPI_COMM_WORLD);
                }

                // DEBUG: Log global_k after gathering (layer 0, rank 0 only)
                if (rank == 0 && layer_index_ == 0)
                {
                    LOG_INFO("[RANK=0] After gather, global_k_rope[t=0]:");
                    LOG_INFO("  offset[0..4] (from rank 0): " << global_k_rope->data()[0] << ", "
                                                              << global_k_rope->data()[1] << ", " << global_k_rope->data()[2] << ", "
                                                              << global_k_rope->data()[3] << ", " << global_k_rope->data()[4]);
                    LOG_INFO("  offset[64..68] (from rank 1): " << global_k_rope->data()[64] << ", "
                                                                << global_k_rope->data()[65] << ", " << global_k_rope->data()[66] << ", "
                                                                << global_k_rope->data()[67] << ", " << global_k_rope->data()[68]);

                    // Check the failing position: token=3, kv_head=1, dim_in_head=32
                    // global_k_rope layout: [token][kv_head_0_64_dims, kv_head_1_64_dims]
                    // offset = token * k_v_dim + kv_head * head_dim + dim_in_head
                    //        = 3 * 128 + 1 * 64 + 32
                    //        = 384 + 64 + 32 = 480
                    const int failing_offset_in_k = 3 * k_v_dim + 1 * head_dim_ + 32;
                    LOG_INFO("  CRITICAL CHECK: global_k_rope[token=3, kv_head=1, dim=32] (offset=" << failing_offset_in_k << "): "
                                                                                                    << global_k_rope->data()[failing_offset_in_k]);
                }

                // DEBUG: Check gathered values
                if (rank == 0 && layer_index_ == 0)
                {
                    LOG_DEBUG("[ROPE_SNAPSHOT_DEBUG] After MPI_Allgather, global_q[t=0]:");
                    std::ostringstream oss1;
                    oss1 << "  First 10 (from rank 0): ";
                    for (int i = 0; i < 10 && i < d_model_; ++i)
                    {
                        oss1 << global_q_rope->data()[i] << " ";
                    }
                    LOG_DEBUG(oss1.str());
                    std::ostringstream oss2;
                    oss2 << "  Elements [" << local_head_dim << ".." << (local_head_dim + 10) << "] (from rank 1): ";
                    for (int i = local_head_dim; i < local_head_dim + 10 && i < d_model_; ++i)
                    {
                        oss2 << global_q_rope->data()[i] << " ";
                    }
                    LOG_DEBUG(oss2.str());
                }
            }
            else
            {
                // Single rank: use local tensors directly
                global_q_rope = local_q;
                global_k_rope = local_k;
                global_v_rope = local_v;
            }

            // Concatenate Q and K along feature dimension for snapshot: [Q | K]
            // This matches PyTorch's ROPE_APPLICATION which includes both
            if (snapshot_callback_)
            {
                std::vector<int> rope_shape = {seq_len, d_model_ + k_v_dim};
                auto rope_combined = TensorFactory::create_simple(rope_shape);
                float *dst = rope_combined->data();

                for (int t = 0; t < seq_len; ++t)
                {
                    const float *q_row = global_q_rope->data() + t * d_model_;
                    const float *k_row = global_k_rope->data() + t * k_v_dim;

                    // Copy Q first
                    std::memcpy(dst, q_row, d_model_ * sizeof(float));
                    dst += d_model_;

                    // Then K
                    std::memcpy(dst, k_row, k_v_dim * sizeof(float));
                    dst += k_v_dim;
                }

                // DEBUG: Show final concatenated values
                if (rank == 0 && layer_index_ == 0)
                {
                    LOG_DEBUG("[ROPE_SNAPSHOT_DEBUG] Final rope_combined[t=0]:");
                    std::ostringstream oss1;
                    oss1 << "  First 10 (Q): ";
                    for (int i = 0; i < 10; ++i)
                    {
                        oss1 << rope_combined->data()[i] << " ";
                    }
                    LOG_DEBUG(oss1.str());
                    std::ostringstream oss2;
                    oss2 << "  Elements [" << d_model_ << ".." << (d_model_ + 10) << "] (K start): ";
                    for (int i = d_model_; i < d_model_ + 10; ++i)
                    {
                        oss2 << rope_combined->data()[i] << " ";
                    }
                    LOG_DEBUG(oss2.str());
                    LOG_DEBUG("  Total rope_combined size: " << rope_combined->size()
                                                             << " expected: " << (seq_len * (d_model_ + k_v_dim)));

                    // CRITICAL DEBUG: Check position [3,992] which is the failing position
                    // Token 3, position 992, which is dim 96 of K (992 - 896 = 96)
                    // NOTE: Only valid for production Qwen-0.5B dimensions (d_model=896, seq_len>=4)
                    if (d_model_ >= 896 && seq_len >= 4)
                    {
                        const int failing_token = 3;
                        const int failing_pos = 992;
                        const int row_size = d_model_ + k_v_dim;
                        const int failing_offset = failing_token * row_size + failing_pos;
                        LOG_INFO("  CRITICAL: rope_combined[token=" << failing_token << ", pos=" << failing_pos << "] (offset=" << failing_offset << "): "
                                                                    << rope_combined->data()[failing_offset]);
                        LOG_INFO("  This should be K[token=3, dim=96] = K[token=3, kv_head=1, dim_in_head=32]");
                    }
                }

                // Only rank 0 needs to snapshot
                if (rank == 0)
                {
                    snapshot_callback_(PipelineStage::ROPE_APPLICATION, layer_index_,
                                       rope_combined->data(), seq_len, d_model_ + k_v_dim);
                }
            }
        }

        // ========================================================================
        // STEP 6: Handle GQA - replicate K/V heads if needed
        // ========================================================================
        // IMPORTANT: For GQA, we need ALL KV heads from ALL ranks to replicate them
        // to the query heads. The global_k_rope and global_v_rope gathered above
        // contain all KV heads concatenated across ranks.
        // NOW USING KV CACHE INSTEAD OF JUST CURRENT K/V!

        std::shared_ptr<TensorBase> local_k_expanded, local_v_expanded;

        if (n_head_ != n_head_kv_)
        {
            // GQA: replicate K/V heads from CACHE to match Q head count
            // Use cache (which contains all past tokens + current token)
            local_k_expanded = TensorFactory::create_simple({attn_seq_len, local_head_dim});
            local_v_expanded = TensorFactory::create_simple({attn_seq_len, local_head_dim});

            // First gather the full cache from all ranks if needed
            std::shared_ptr<TensorBase> global_k_cache, global_v_cache;

            if (world_size > 1)
            {
                // Need to gather full cache from all ranks for GQA expansion
                const int k_v_dim = n_head_kv_ * head_dim_;
                global_k_cache = TensorFactory::create_simple({attn_seq_len, k_v_dim});
                global_v_cache = TensorFactory::create_simple({attn_seq_len, k_v_dim});

                std::vector<int> recvcounts_k(world_size);
                std::vector<int> displs_k(world_size);
                int sendcount_k = attn_seq_len * local_kv_head_dim;

                MPI_Allgather(&sendcount_k, 1, MPI_INT, recvcounts_k.data(), 1, MPI_INT, MPI_COMM_WORLD);

                int offset_k = 0;
                for (int r = 0; r < world_size; ++r)
                {
                    displs_k[r] = offset_k;
                    offset_k += recvcounts_k[r];
                }

                MPI_Allgatherv(local_k_cache->data(), sendcount_k, MPI_FLOAT,
                               global_k_cache->data(), recvcounts_k.data(), displs_k.data(),
                               MPI_FLOAT, MPI_COMM_WORLD);
                MPI_Allgatherv(local_v_cache->data(), sendcount_k, MPI_FLOAT,
                               global_v_cache->data(), recvcounts_k.data(), displs_k.data(),
                               MPI_FLOAT, MPI_COMM_WORLD);
            }
            else
            {
                global_k_cache = local_k_cache;
                global_v_cache = local_v_cache;
            }

            // CRITICAL FIX: Pass head_offset and total Q heads so GQA expansion uses correct mapping
            // For Qwen: 14 Q heads, 2 KV heads → group_size=7
            //   Q heads [0-6] (rank 0) → KV head 0
            //   Q heads [7-13] (rank 1) → KV head 1
            // Formula: kv_head = (local_h + head_offset) / (total_q_heads / n_kv_heads)
            llaminar::attn::expand_kv_for_gqa(
                global_k_cache->data(), global_v_cache->data(),
                local_k_expanded->data(), local_v_expanded->data(),
                attn_seq_len, head_dim_, local_heads, n_head_kv_, head_offset, n_head_); // Use cache length!

            // DEBUG: Log after GQA expansion (layer 0 only)
            if (layer_index_ == 0)
            {
                LOG_DEBUG("[RANK=" << rank << "] After GQA expansion (using cache):");
                LOG_INFO("  attn_seq_len=" << attn_seq_len << " (n_past + seq_len)");
                LOG_INFO("  K_expanded shape: [" << attn_seq_len << ", " << local_head_dim << "]");
                LOG_INFO("  K_expanded[0,0:5]: " << local_k_expanded->data()[0] << " " << local_k_expanded->data()[1] << " "
                                                 << local_k_expanded->data()[2] << " " << local_k_expanded->data()[3] << " " << local_k_expanded->data()[4]);

                float k_exp_min = *std::min_element(local_k_expanded->data(), local_k_expanded->data() + local_k_expanded->size());
                float k_exp_max = *std::max_element(local_k_expanded->data(), local_k_expanded->data() + local_k_expanded->size());
                LOG_INFO("  K_expanded range: [" << k_exp_min << ", " << k_exp_max << "]");
            }
        }
        else
        {
            // MHA: no replication needed, use cache directly
            local_k_expanded = local_k_cache;
            local_v_expanded = local_v_cache;
        }

        // ========================================================================
        // STEP 7: Compute attention scores and apply softmax
        // ========================================================================
        // CRITICAL: Now using full cache length for K dimension!
        const int scores_size = local_heads * seq_len * attn_seq_len;
        std::vector<float> scores(scores_size);

        // IMPORTANT: For parity testing, we need to capture scores BEFORE causal masking
        // PyTorch captures unmasked scores, then applies mask separately
        if (snapshot_callback_)
        {
            // DEBUG: Log Q and K before computing scores (layer 0 only)
            if (layer_index_ == 0)
            {
                LOG_INFO("[RANK=" << rank << "] Before compute_qk_scores for snapshot:");
                LOG_INFO("  local_q size=" << (local_heads * seq_len * head_dim_)
                                           << " shape=[" << seq_len << ", " << (local_heads * head_dim_) << "]");
                LOG_INFO("  local_k_expanded size=" << (local_heads * attn_seq_len * head_dim_)
                                                    << " shape=[" << attn_seq_len << ", " << (local_heads * head_dim_) << "]");
                LOG_INFO("  scores shape will be: [" << local_heads << ", " << seq_len << ", " << attn_seq_len << "]");
                LOG_INFO("  scores total elements: " << scores_size);

                // CRITICAL: Verify memory layout expectations
                LOG_DEBUG("[MEMORY_LAYOUT_DEBUG] Q tensor layout check:");
                LOG_INFO("  Expected by compute_qk_scores: Q[token, head, dim] flattened");
                LOG_INFO("  Index formula: q[i, h, d] = q[(i * heads * head_dim) + (h * head_dim) + d]");
                LOG_INFO("  For token i=0, head h=0: offset = (0 * " << local_heads << " * " << head_dim_ << ") + (0 * " << head_dim_ << ") + d = d");
                LOG_INFO("  For token i=0, head h=1: offset = (0 * " << local_heads << " * " << head_dim_ << ") + (1 * " << head_dim_ << ") + d = " << head_dim_ << " + d");
                LOG_INFO("  For token i=1, head h=0: offset = (1 * " << local_heads << " * " << head_dim_ << ") + (0 * " << head_dim_ << ") + d = " << (local_heads * head_dim_) << " + d");

                LOG_DEBUG("  Actual Q memory layout after projection:");
                LOG_INFO("    Q[t=0, h=0, d=0:5]: "
                         << local_q->data()[0] << ", " << local_q->data()[1] << ", "
                         << local_q->data()[2] << ", " << local_q->data()[3] << ", "
                         << local_q->data()[4]);

                int offset_t0_h1 = head_dim_;
                LOG_INFO("    Q[t=0, h=1, d=0:5]: "
                         << local_q->data()[offset_t0_h1 + 0] << ", " << local_q->data()[offset_t0_h1 + 1] << ", "
                         << local_q->data()[offset_t0_h1 + 2] << ", " << local_q->data()[offset_t0_h1 + 3] << ", "
                         << local_q->data()[offset_t0_h1 + 4]);

                int offset_t1_h0 = local_heads * head_dim_;
                LOG_INFO("    Q[t=1, h=0, d=0:5]: "
                         << local_q->data()[offset_t1_h0 + 0] << ", " << local_q->data()[offset_t1_h0 + 1] << ", "
                         << local_q->data()[offset_t1_h0 + 2] << ", " << local_q->data()[offset_t1_h0 + 3] << ", "
                         << local_q->data()[offset_t1_h0 + 4]);

                LOG_DEBUG("[MEMORY_LAYOUT_DEBUG] K_expanded tensor layout check:");
                LOG_INFO("  K[0,0:5]: " << local_k_expanded->data()[0] << " " << local_k_expanded->data()[1] << " "
                                        << local_k_expanded->data()[2] << " " << local_k_expanded->data()[3] << " " << local_k_expanded->data()[4]);
                LOG_INFO("    K[t=0, h=0, d=0:5]: "
                         << local_k_expanded->data()[0] << ", " << local_k_expanded->data()[1] << ", "
                         << local_k_expanded->data()[2] << ", " << local_k_expanded->data()[3] << ", "
                         << local_k_expanded->data()[4]);
                LOG_INFO("    K[t=0, h=1, d=0:5]: "
                         << local_k_expanded->data()[offset_t0_h1 + 0] << ", " << local_k_expanded->data()[offset_t0_h1 + 1] << ", "
                         << local_k_expanded->data()[offset_t0_h1 + 2] << ", " << local_k_expanded->data()[offset_t0_h1 + 3] << ", "
                         << local_k_expanded->data()[offset_t0_h1 + 4]);
                LOG_INFO("    K[t=1, h=0, d=0:5]: "
                         << local_k_expanded->data()[offset_t1_h0 + 0] << ", " << local_k_expanded->data()[offset_t1_h0 + 1] << ", "
                         << local_k_expanded->data()[offset_t1_h0 + 2] << ", " << local_k_expanded->data()[offset_t1_h0 + 3] << ", "
                         << local_k_expanded->data()[offset_t1_h0 + 4]);

                // Compute what the first score should be
                double test_dot = 0.0;
                for (int d = 0; d < head_dim_; ++d)
                {
                    test_dot += local_q->data()[d] * local_k_expanded->data()[d];
                }
                float scale = 1.0f / std::sqrt((float)head_dim_);
                LOG_INFO("  Expected scores[0,0] = Q[0]·K[0]/sqrt(" << head_dim_ << ") = "
                                                                    << test_dot << " * " << scale << " = " << (test_dot * scale));
            }

            // Compute unmasked scores for snapshot
            std::vector<float> unmasked_scores(scores_size);
            llaminar::attn::compute_qk_scores(local_q->data(), local_k_expanded->data(),
                                              unmasked_scores.data(),
                                              seq_len, attn_seq_len, // q_seq_len, k_seq_len
                                              head_dim_, local_heads,
                                              false, false); // causal=FALSE for snapshot

            // DEBUG: Log computed scores (layer 0 only)
            if (layer_index_ == 0)
            {
                LOG_INFO("[RANK=" << rank << "] After compute_qk_scores:");
                LOG_INFO("  unmasked_scores[0:5]: " << unmasked_scores[0] << " " << unmasked_scores[1] << " "
                                                    << unmasked_scores[2] << " " << unmasked_scores[3] << " " << unmasked_scores[4]);
            }

            // Gather and snapshot unmasked scores
            if (world_size > 1)
            {
                auto global_scores = std::vector<float>(n_head_ * seq_len * attn_seq_len, 0.0f);

                std::vector<int> recvcounts(world_size);
                std::vector<int> displs(world_size);

                int sendcount = local_heads * seq_len * attn_seq_len;
                MPI_Allgather(&sendcount, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

                int offset = 0;
                for (int r = 0; r < world_size; ++r)
                {
                    displs[r] = offset;
                    offset += recvcounts[r];
                }

                MPI_Allgatherv(unmasked_scores.data(), sendcount, MPI_FLOAT,
                               global_scores.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                               MPI_COMM_WORLD);

                // DEBUG: Log gathered scores structure (layer 0 only)
                if (layer_index_ == 0 && rank == 0)
                {
                    LOG_INFO("[GATHERED_SCORES_DEBUG] After MPI_Allgatherv:");
                    LOG_INFO("  global_scores size: " << global_scores.size() << " (expected " << (n_head_ * seq_len * attn_seq_len) << ")");
                    LOG_INFO("  Will snapshot as [" << (n_head_ * seq_len) << " x " << attn_seq_len << "]");
                    LOG_INFO("  Rank 0 contributed: " << recvcounts[0] << " elements (heads 0-" << (local_heads - 1) << ")");
                    LOG_INFO("  Rank 1 contributed: " << recvcounts[1] << " elements (heads " << local_heads << "-" << (n_head_ - 1) << ")");
                    LOG_INFO("  Rank 0 offset: " << displs[0]);
                    LOG_INFO("  Rank 1 offset: " << displs[1]);
                    LOG_INFO("  First 10 elements: ");
                    for (int i = 0; i < std::min(10, (int)global_scores.size()); ++i)
                    {
                        LOG_INFO("    global_scores[" << i << "] = " << global_scores[i]);
                    }

                    // Interpret as 2D: [n_head * seq_len, attn_seq_len]
                    // Row 0 = head 0, token 0 (first attn_seq_len elements)
                    LOG_INFO("  Row 0 (head 0, token 0): " << global_scores[0] << " " << global_scores[1] << " "
                                                           << global_scores[2] << " " << global_scores[3] << " " << global_scores[4]);
                    // Row 1 = head 0, token 1
                    LOG_INFO("  Row 1 (head 0, token 1): " << global_scores[5] << " " << global_scores[6] << " "
                                                           << global_scores[7] << " " << global_scores[8] << " " << global_scores[9]);

                    // Also show what we EXPECT PyTorch to have:
                    LOG_INFO("  Expected PyTorch row 0 should match our row 0");
                    LOG_INFO("  Expected PyTorch row 1 should match our row 1");
                }

                if (rank == 0)
                {
                    snapshot_callback_(PipelineStage::ATTENTION_SCORES, layer_index_, global_scores.data(),
                                       n_head_ * seq_len, attn_seq_len); // rows, cols
                }
            }
            else
            {
                snapshot_callback_(PipelineStage::ATTENTION_SCORES, layer_index_, unmasked_scores.data(),
                                   local_heads * seq_len, attn_seq_len); // rows, cols
            }
        } // Now compute MASKED scores for actual attention (causal masking, scaling, NO softmax yet)
        llaminar::attn::compute_qk_scores(local_q->data(), local_k_expanded->data(),
                                          scores.data(), seq_len, attn_seq_len,
                                          head_dim_, local_heads,
                                          true, false); // causal=TRUE for actual computation

        // DEBUG: Log masked scores for layer 0
        if (layer_index_ == 0 && rank == 0)
        {
            LOG_INFO("[MASKED_SCORES_DEBUG] Layer 0, AFTER compute_qk_scores with causal=TRUE:");
            LOG_INFO("  Head 0, scores[0:6]: " << scores[0] << " " << scores[1] << " "
                                               << scores[2] << " " << scores[3] << " " << scores[4] << " " << scores[5]);
        }

        // Contract: Validate attention scores (raw QK^T)
        if (enable_validation && rank == 0)
        {
            TensorHealthCheck scores_health("scores_pre_softmax");
            scores_health.check(scores.data(), scores.size());
            scores_health.log(rank);

            // Note: -inf values are EXPECTED in causal positions
            if (scores_health.nan_count > 0)
            {
                LOG_ERROR("❌ Attention scores contain NaN (unexpected)");
                return false;
            }
            if (scores_health.inf_count == 0 && seq_len > 1)
            {
                LOG_WARN("⚠️ Expected -inf in causal mask positions but found none");
            }
            LOG_DEBUG("✓ Attention scores validated (inf_count=" << scores_health.inf_count << " expected for causal masking)");
        }

        // Apply softmax to each head (sequential loop - softmax_row_major parallelizes internally)
        // CRITICAL FIX: In incremental decode (seq_len=1), disable causal masking because
        // the query token is attending to the full cache which only contains PAST tokens.
        // Causal masking with rows=1 would incorrectly mask based on relative position (row 0 -> mask all c > 0).
        const bool use_causal_mask = (seq_len > 1); // Only use causal mask in prefill/batch mode

        for (int h = 0; h < local_heads; ++h)
        {
            llaminar::kernels::SoftmaxRowArgs args;
            args.scores = scores.data() + static_cast<size_t>(h) * seq_len * attn_seq_len;
            args.rows = seq_len;
            args.cols = attn_seq_len;
            args.causal = use_causal_mask; // FIX: Disable for incremental decode
            args.scale = 1.0f;

            // DEBUG: Log scores before softmax for layer 0, head 0
            if (layer_index_ == 0 && h == 0 && rank == 0)
            {
                LOG_INFO("[SOFTMAX_DEBUG] Layer 0, Head 0, BEFORE softmax:");
                LOG_INFO("  seq_len=" << seq_len << " attn_seq_len=" << attn_seq_len << " use_causal=" << use_causal_mask);
                LOG_INFO("  scores[0:6]: " << args.scores[0] << " " << args.scores[1] << " "
                                           << args.scores[2] << " " << args.scores[3] << " "
                                           << args.scores[4] << " " << args.scores[5]);
            }

            llaminar::kernels::softmax_row_major(args);

            // DEBUG: Log scores after softmax for layer 0, head 0
            if (layer_index_ == 0 && h == 0 && rank == 0)
            {
                LOG_INFO("[SOFTMAX_DEBUG] Layer 0, Head 0, AFTER softmax:");
                LOG_INFO("  scores[0:6]: " << args.scores[0] << " " << args.scores[1] << " "
                                           << args.scores[2] << " " << args.scores[3] << " "
                                           << args.scores[4] << " " << args.scores[5]);
                float sum = 0.0f;
                for (int i = 0; i < attn_seq_len; ++i)
                    sum += args.scores[i];
                LOG_INFO("  Sum (should be ~1.0): " << sum);
            }
        }

        // Snapshot scores AFTER softmax
        if (snapshot_callback_)
        {
            if (world_size > 1)
            {
                // Gather softmax scores across ranks: [local_heads, seq_len, attn_seq_len] -> [n_head, seq_len, attn_seq_len]
                auto global_softmax = std::vector<float>(n_head_ * seq_len * attn_seq_len, 0.0f);

                std::vector<int> recvcounts(world_size);
                std::vector<int> displs(world_size);

                int sendcount = local_heads * seq_len * attn_seq_len;
                MPI_Allgather(&sendcount, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

                int offset = 0;
                for (int r = 0; r < world_size; ++r)
                {
                    displs[r] = offset;
                    offset += recvcounts[r];
                }

                MPI_Allgatherv(scores.data(), sendcount, MPI_FLOAT,
                               global_softmax.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                               MPI_COMM_WORLD);

                if (rank == 0)
                {
                    snapshot_callback_(PipelineStage::ATTENTION_SOFTMAX, layer_index_, global_softmax.data(),
                                       n_head_ * seq_len, attn_seq_len);
                }
            }
            else
            {
                snapshot_callback_(PipelineStage::ATTENTION_SOFTMAX, layer_index_, scores.data(),
                                   local_heads * seq_len, attn_seq_len);
            }
        }

        // Contract: Validate attention probabilities (after softmax)
        if (enable_validation && rank == 0)
        {
            TensorHealthCheck probs_health("attention_probs");
            probs_health.check(scores.data(), scores.size());
            probs_health.log(rank);

            if (!probs_health.is_healthy())
            {
                LOG_ERROR("❌ Softmax produced NaN/Inf probabilities!");
                return false;
            }

            // Verify probability constraints: values in [0, 1], rows sum to ~1.0
            bool valid_probs = true;
            for (int h = 0; h < local_heads; ++h)
            {
                for (int r = 0; r < seq_len; ++r)
                {
                    float row_sum = 0.0f;
                    const float *row = scores.data() + h * seq_len * attn_seq_len + r * attn_seq_len;
                    for (int c = 0; c < attn_seq_len; ++c)
                    {
                        if (row[c] < 0.0f || row[c] > 1.0f)
                        {
                            LOG_ERROR("Invalid probability at head=" << h << " row=" << r << " col=" << c << ": " << row[c]);
                            valid_probs = false;
                        }
                        row_sum += row[c];
                    }
                    // Allow slight numerical error in sum
                    if (std::abs(row_sum - 1.0f) > 1e-4f && r < seq_len)
                    { // Only check non-masked rows
                        LOG_WARN("Row sum deviation at head=" << h << " row=" << r << ": sum=" << row_sum);
                    }
                }
            }

            if (!valid_probs)
            {
                LOG_ERROR("❌ Probability constraints violated!");
                return false;
            }
            LOG_DEBUG("✓ Attention probabilities validated (all in [0,1], rows sum to 1.0)");
        }

        // ========================================================================
        // STEP 6b: Apply attention scores to V
        // ========================================================================
        auto local_attended = TensorFactory::create_simple({seq_len, local_head_dim});

        llaminar::attn::apply_scores_to_v(scores.data(), local_v_expanded->data(),
                                          local_attended->data(), seq_len, attn_seq_len,
                                          head_dim_, local_heads);

        // Contract: Validate attended output (scores @ V)
        if (enable_validation && rank == 0)
        {
            StageContract attended_contract("AttendedOutput");
            attended_contract.outputs = {
                TensorContract("attended",
                               ShapeSpec({seq_len, local_head_dim}),
                               TensorLayout::RowMajor,
                               TensorSemantic::Activation)};
            attended_contract.validate_outputs({local_attended});

            TensorHealthCheck attended_health("attended_output");
            attended_health.check(local_attended->data(), local_attended->size());
            attended_health.log(rank);

            if (!attended_health.is_healthy())
            {
                LOG_ERROR("❌ Attended output (scores @ V) contains NaN/Inf!");
                return false;
            }
            LOG_DEBUG("✓ Attended output validated");
        }

        // Snapshot attended values (ATTENTION_CONTEXT: scores @ V before output projection)
        if (snapshot_callback_)
        {
            if (world_size > 1)
            {
                // Gather attended output across ranks
                // Each rank has [seq_len, local_head_dim], need [seq_len, n_head * head_dim]
                auto global_attended = TensorFactory::create_simple({seq_len, n_head_ * head_dim_});

                // Gather row-by-row to maintain proper layout (same as Q/K/V)
                for (int t = 0; t < seq_len; ++t)
                {
                    const float *local_attended_row = local_attended->data() + t * local_head_dim;
                    float *global_attended_row = global_attended->data() + t * (n_head_ * head_dim_);

                    MPI_Allgather(local_attended_row, local_head_dim, MPI_FLOAT,
                                  global_attended_row, local_head_dim, MPI_FLOAT,
                                  MPI_COMM_WORLD);
                }

                if (rank == 0)
                {
                    snapshot_callback_(PipelineStage::ATTENTION_CONTEXT, layer_index_, global_attended->data(),
                                       seq_len, n_head_ * head_dim_);
                }
            }
            else
            {
                snapshot_callback_(PipelineStage::ATTENTION_CONTEXT, layer_index_, local_attended->data(),
                                   seq_len, local_head_dim);
            }
        }

        if (rank == 0)
        {
            LOG_DEBUG("[kernel-test] rank " << rank << " local_attended[0,0:4]: "
                                            << local_attended->data()[0] << ", " << local_attended->data()[1] << ", "
                                            << local_attended->data()[2] << ", " << local_attended->data()[3]);
        }

        // ========================================================================
        // STEP 7: Output projection + MPI gather
        // ========================================================================
        auto local_output = TensorFactory::create_simple({seq_len, d_model});

        // Output projection: local_attended @ wo^T
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, d_model, local_head_dim,
                    1.0f, local_attended->data(), local_head_dim,
                    local_wo->data(), local_head_dim,
                    0.0f, local_output->data(), d_model);

        // Contract: Validate output projection
        if (enable_validation && rank == 0)
        {
            StageContract output_contract("OutputProjection");
            output_contract.outputs = {
                TensorContract("local_output",
                               ShapeSpec({seq_len, d_model}),
                               TensorLayout::RowMajor,
                               TensorSemantic::Activation)};
            output_contract.validate_outputs({local_output});

            TensorHealthCheck output_health("output_projection");
            output_health.check(local_output->data(), local_output->size());
            output_health.log(rank);

            if (!output_health.is_healthy())
            {
                LOG_ERROR("❌ Output projection contains NaN/Inf!");
                return false;
            }

            // Optional: Deep validation against scalar reference
            if (validate_projections)
            {
                auto result = llaminar::attention::AttentionValidator::validateProjection(
                    local_attended->data(), local_wo->data(), local_output->data(),
                    seq_len, d_model, local_head_dim,
                    false // wo is already transposed for gemm
                );

                if (!llaminar::attention::AttentionValidator::isWithinTolerance(result, 1e-4f, 1e-4f))
                {
                    LOG_WARN("⚠️ Output projection divergence: max_abs=" << result.max_abs
                                                                        << " rel_l2=" << result.rel_l2);
                }
                else
                {
                    LOG_DEBUG("✓ Output projection matches scalar reference");
                }
            }

            LOG_DEBUG("✓ Output projection validated");
        }

        // Aggregate across ranks if multi-rank
        if (world_size > 1)
        {
            MPI_Allreduce(MPI_IN_PLACE, local_output->data(), local_output->size(),
                          MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        }

        // Snapshot final attention output (after output projection and MPI reduction)
        if (snapshot_callback_ && rank == 0)
        {
            snapshot_callback_(PipelineStage::ATTENTION_OUTPUT, layer_index_, local_output->data(),
                               seq_len, d_model);
        }

        if (rank == 0)
        {
            LOG_DEBUG("[kernel-test] rank " << rank << " final_output[0,0:4]: "
                                            << local_output->data()[0] << ", " << local_output->data()[1] << ", "
                                            << local_output->data()[2] << ", " << local_output->data()[3]);
        }

        // Contract: Validate final output after MPI aggregation
        if (enable_validation && rank == 0)
        {
            StageContract final_contract("FinalOutput");
            final_contract.outputs = {
                TensorContract("final_output",
                               ShapeSpec({seq_len, d_model}),
                               TensorLayout::RowMajor,
                               TensorSemantic::Activation)};
            final_contract.validate_outputs({local_output});

            TensorHealthCheck final_health("final_output");
            final_health.check(local_output->data(), local_output->size());
            final_health.log(rank);

            if (!final_health.is_healthy())
            {
                LOG_ERROR("❌ Final output after MPI aggregation contains NaN/Inf!");
                return false;
            }

            LOG_DEBUG("✓ Final output validated - attention kernel complete");
        }

        // Copy to output tensor
        if (outputs.empty())
        {
            outputs.push_back(TensorFactory::create_simple({seq_len, d_model}));
            outputs.push_back(TensorFactory::create_simple(local_k_cache->shape()));
            outputs.push_back(TensorFactory::create_simple(local_v_cache->shape()));
        }
        else if (outputs.size() == 1)
        {
            // Legacy: only attention output was expected, add cache outputs
            outputs.push_back(TensorFactory::create_simple(local_k_cache->shape()));
            outputs.push_back(TensorFactory::create_simple(local_v_cache->shape()));
        }
        else if (outputs.size() >= 3)
        {
            // Ensure cache output tensors are created if nullptr
            if (!outputs[1])
            {
                outputs[1] = TensorFactory::create_simple(local_k_cache->shape());
            }
            if (!outputs[2])
            {
                outputs[2] = TensorFactory::create_simple(local_v_cache->shape());
            }
        }

        // outputs[0] = attention output
        memcpy(outputs[0]->data(), local_output->data(), seq_len * d_model * sizeof(float));

        if (rank == 0 && layer_index_ == 0)
        {
            LOG_INFO("[CACHE_RETURN_DEBUG] Before copying to outputs:");
            LOG_INFO("  local_k_cache shape: [" << local_k_cache->shape()[0] << ", " << local_k_cache->shape()[1] << "]");
            LOG_INFO("  local_k_cache first 10: "
                     << local_k_cache->data()[0] << " " << local_k_cache->data()[1] << " "
                     << local_k_cache->data()[2] << " " << local_k_cache->data()[3] << " "
                     << local_k_cache->data()[4] << " " << local_k_cache->data()[5] << " "
                     << local_k_cache->data()[6] << " " << local_k_cache->data()[7] << " "
                     << local_k_cache->data()[8] << " " << local_k_cache->data()[9]);
            LOG_INFO("  outputs[1] shape: [" << outputs[1]->shape()[0] << ", " << outputs[1]->shape()[1] << "]");
            LOG_INFO("  outputs[1] before copy: "
                     << outputs[1]->data()[0] << " " << outputs[1]->data()[1] << " "
                     << outputs[1]->data()[2] << " " << outputs[1]->data()[3] << " "
                     << outputs[1]->data()[4] << " " << outputs[1]->data()[5] << " "
                     << outputs[1]->data()[6] << " " << outputs[1]->data()[7] << " "
                     << outputs[1]->data()[8] << " " << outputs[1]->data()[9]);
        }

        // outputs[1] = updated K cache (local portion for this rank's KV heads)
        memcpy(outputs[1]->data(), local_k_cache->data(), local_k_cache->size() * sizeof(float));

        // outputs[2] = updated V cache (local portion for this rank's KV heads)
        memcpy(outputs[2]->data(), local_v_cache->data(), local_v_cache->size() * sizeof(float));

        if (rank == 0 && layer_index_ == 0)
        {
            LOG_INFO("[CACHE_RETURN_DEBUG] After copying to outputs:");
            LOG_INFO("  outputs[1] after copy: "
                     << outputs[1]->data()[0] << " " << outputs[1]->data()[1] << " "
                     << outputs[1]->data()[2] << " " << outputs[1]->data()[3] << " "
                     << outputs[1]->data()[4] << " " << outputs[1]->data()[5] << " "
                     << outputs[1]->data()[6] << " " << outputs[1]->data()[7] << " "
                     << outputs[1]->data()[8] << " " << outputs[1]->data()[9]);
        }

        if (rank == 0 && debugEnv().attention.micro_trace)
        {
            LOG_DEBUG("[KVCacheReturn] layer=" << layer_index_
                                               << " k_cache_shape=[" << local_k_cache->shape()[0] << "," << local_k_cache->shape()[1] << "]"
                                               << " v_cache_shape=[" << local_v_cache->shape()[0] << "," << local_v_cache->shape()[1] << "]"
                                               << " attn_seq_len=" << attn_seq_len);
        }

        return true;
    }

} // namespace llaminar
