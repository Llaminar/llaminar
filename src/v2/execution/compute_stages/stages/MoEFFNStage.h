/**
 * @file MoEFFNStage.h
 * @brief Unified MoE FFN stage: route → expert SwiGLU → combine
 *
 * Implements the full MoE feed-forward block as a single stage:
 * 1. Router: hidden × gate_weights → softmax → top-k selection
 * 2. Expert FFN: per-expert SwiGLU (gate, up, down) with gather/scatter
 * 3. Combine: weighted sum of expert outputs
 *
 * This is implemented as a single stage because routing creates dynamic
 * control flow that cannot be expressed as a static compute graph.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"

#include <memory>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class ITensorGemm;
    class FP32Tensor;
    class IMoEKernel;

    /**
     * @brief Unified MoE FFN stage (router + expert execution + combine)
     *
     * Supports CPU, CUDA, and ROCm backends:
     * - CPU: Inline dequantization + scalar dot products (original path)
     * - GPU: Per-expert 2D tensor views → KernelFactory GEMM dispatch
     *
     * Expert views are pre-extracted at graph build time to avoid
     * runtime 3D tensor slicing overhead.
     */
    class MoEFFNStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            // Input
            TensorBase *input = nullptr; ///< Normalized hidden [seq_len, d_model]
            int seq_len = 0;
            int d_model = 0;

            // Router
            TensorBase *gate_weights = nullptr; ///< Router gate [num_experts, d_model]
            int num_experts = 0;
            int top_k = 0;
            bool norm_topk_prob = true;

            // Expert weights (3D packed tensors) — used by CPU path
            TensorBase *gate_exps = nullptr; ///< [num_experts, intermediate, d_model]
            TensorBase *up_exps = nullptr;   ///< [num_experts, intermediate, d_model]
            TensorBase *down_exps = nullptr; ///< [num_experts, d_model, intermediate]
            int expert_intermediate = 0;

            // Expert Parallelism (EP): partition experts across TP ranks.
            // When active, this rank only computes experts in
            // [local_expert_start, local_expert_start + local_expert_count).
            // Set by graph builder when TP degree > 1.
            // -1 means all experts (no EP, single-device mode).
            int local_expert_start = 0;
            int local_expert_count = -1;

            // Per-expert 2D tensor views — used by GPU path
            // Each vector has num_experts entries; each entry is a 2D view
            // into the corresponding 3D packed tensor.
            // Set by graph builder via extractExpertViews().
            std::vector<std::shared_ptr<TensorBase>> expert_gate_views; ///< [intermediate, d_model] per expert
            std::vector<std::shared_ptr<TensorBase>> expert_up_views;   ///< [intermediate, d_model] per expert
            std::vector<std::shared_ptr<TensorBase>> expert_down_views; ///< [d_model, intermediate] per expert

            // Pre-resolved GEMM engines per expert — set by prepareExpertGemmEngines()
            // at graph build time so that execute() never triggers weight repacking.
            std::vector<ITensorGemm *> prepared_gate_gemm; ///< [num_experts] GEMM engines
            std::vector<ITensorGemm *> prepared_up_gemm;   ///< [num_experts] GEMM engines
            std::vector<ITensorGemm *> prepared_down_gemm;  ///< [num_experts] GEMM engines

            // MoE batch-packed GPU lifetime management:
            // owned_kernels keeps MoE batch-constructed kernels alive,
            // packed_*_lifetime keeps the shared GPU allocation alive.
            std::vector<std::shared_ptr<ITensorGemm>> moe_owned_kernels;
            std::shared_ptr<void> moe_packed_gate_lifetime;
            std::shared_ptr<void> moe_packed_up_lifetime;
            std::shared_ptr<void> moe_packed_down_lifetime;

            // Scratch buffers for GPU expert execution
            TensorBase *gate_scratch = nullptr; ///< [seq_len, intermediate] FP32 scratch
            TensorBase *up_scratch = nullptr;   ///< [seq_len, intermediate] FP32 scratch

            // Output
            TensorBase *output = nullptr; ///< Combined output [seq_len, d_model]

            // Buffer IDs for coherence
            BufferId input_buffer_id = BufferId::NORMALIZED;
            BufferId output_buffer_id = BufferId::MOE_COMBINED_OUTPUT;
        };

        explicit MoEFFNStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_EXPERT_FFN; }
        std::string name() const override { return "moe_ffn"; }
        size_t estimatedFlops() const override;

        /// In expert-parallel mode, a rank's MoE FFN output can be all zeros
        /// when no selected experts fall in its local range. The downstream
        /// AllReduce combines partial results across ranks.
        bool allowsZeroOutput() const override
        {
            return params_.local_expert_count >= 0;
        }
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

        /// Extract 2D expert views from 3D packed tensors.
        /// Call once at graph-build time. Views are stored in params.
        static bool extractExpertViews(Params &params);

        /// Prepare GEMM engines for all expert views at graph-build time.
        /// Must be called after extractExpertViews(). Triggers VNNI repacking
        /// during model loading rather than on first inference call.
        static bool prepareExpertGemmEngines(Params &params);

    private:
        Params params_;

#ifdef HAVE_CUDA
        static bool prepareExpertGemmEnginesCUDA(Params &params);
#endif
#ifdef HAVE_ROCM
        static bool prepareExpertGemmEnginesROCm(Params &params);
#endif

        /// Stashed routing results from last execute() — for snapshot capture.
        /// Stored as FP32 [seq_len, top_k] so buildDumpInfoImpl() can expose them.
        mutable std::vector<float> routing_indices_f32_; ///< Expert IDs cast to float
        mutable std::vector<float> routing_weights_;     ///< Normalized top-k weights
        mutable std::vector<float> router_logits_;       ///< Raw router logits [seq_len, num_experts]

        /// Cached GEMM engines per expert (resolved on first execute)
        mutable std::vector<ITensorGemm *> cached_gate_gemm_;
        mutable std::vector<ITensorGemm *> cached_up_gemm_;
        mutable std::vector<ITensorGemm *> cached_down_gemm_;

        /// Reusable scratch tensors (allocated on first use, grown if needed)
        mutable std::shared_ptr<FP32Tensor> scratch_batch_;
        mutable std::shared_ptr<FP32Tensor> scratch_gate_;
        mutable std::shared_ptr<FP32Tensor> scratch_up_;
        mutable std::shared_ptr<FP32Tensor> scratch_out_;
        mutable int scratch_capacity_ = 0;

        /// Batched gate+up scratch buffers for M=1 decode (one per top-k expert).
        /// Enables fusing all experts' gate+up into a single OMP region.
        mutable std::vector<std::shared_ptr<FP32Tensor>> scratch_gate_batch_;
        mutable std::vector<std::shared_ptr<FP32Tensor>> scratch_up_batch_;

        /// Reusable projection descriptor vector (avoids per-call heap alloc)
        mutable std::vector<ITensorGemm::TensorProjectionDesc> batch_projections_;

        /// Cached MoE kernel (routing, gather/scatter, SwiGLU fallback)
        mutable IMoEKernel *moe_kernel_ = nullptr;

        /// Fast path for decode (seq_len=1): avoids token grouping, gather/scatter,
        /// and per-expert heap allocations. Uses routing results directly.
        bool executeSingleToken(IDeviceContext *ctx);


        void ensureGemmEnginesCached() const;
        void ensureScratchBuffers(int max_batch) const;
        IMoEKernel *ensureMoEKernel() const;

        /// Stash routing results for snapshot capture
        void stashRoutingResults(
            const std::vector<int> &expert_indices,
            const std::vector<float> &expert_weights,
            int seq_len, int top_k) const;
    };

    /**
     * @brief Shared expert FFN stage: always-active dense SwiGLU
     *
     * Runs standard SwiGLU (gate_proj → up_proj → silu(gate)*up → down_proj)
     * using the shared expert weights. Executes for ALL tokens unconditionally.
     */
    class SharedExpertFFNStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            TensorBase *input = nullptr;  ///< Normalized hidden [seq_len, d_model]
            TensorBase *gate_w = nullptr; ///< Shared expert gate [intermediate, d_model]
            TensorBase *up_w = nullptr;   ///< Shared expert up [intermediate, d_model]
            TensorBase *down_w = nullptr; ///< Shared expert down [d_model, intermediate]
            TensorBase *output = nullptr; ///< Output [seq_len, d_model]
            int seq_len = 0;
            int d_model = 0;
            int intermediate = 0;

            BufferId input_buffer_id = BufferId::NORMALIZED;
            BufferId output_buffer_id = BufferId::MOE_SHARED_EXPERT_OUTPUT;
        };

        explicit SharedExpertFFNStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_SHARED_EXPERT_FFN; }
        std::string name() const override { return "shared_expert_ffn"; }
        size_t estimatedFlops() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

    private:
        Params params_;

        mutable ITensorGemm *cached_gate_gemm_ = nullptr;
        mutable ITensorGemm *cached_up_gemm_ = nullptr;
        mutable ITensorGemm *cached_down_gemm_ = nullptr;
        mutable std::shared_ptr<FP32Tensor> scratch_gate_;
        mutable std::shared_ptr<FP32Tensor> scratch_up_;
        mutable int scratch_seq_len_ = 0;

        void ensureGemmEnginesCached() const;

        /// Cached MoE kernel for SwiGLU fallback
        mutable IMoEKernel *moe_kernel_ = nullptr;
        IMoEKernel *ensureMoEKernel() const;
    };

    /**
     * @brief Shared expert sigmoid gate stage
     *
     * Computes: output = sigmoid(gate_inp · input) ⊙ shared_expert_output
     * - gate_inp: [d_model] vector
     * - input: [seq_len, d_model]
     * - shared_expert_output: [seq_len, d_model] (modified in-place)
     */
    class SharedExpertGateStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            TensorBase *input = nullptr;         ///< Normalized hidden [seq_len, d_model]
            TensorBase *gate_inp = nullptr;      ///< Gate vector [d_model]
            TensorBase *shared_output = nullptr; ///< Shared expert output (in-place) [seq_len, d_model]
            int seq_len = 0;
            int d_model = 0;

            BufferId input_buffer_id = BufferId::NORMALIZED;
            BufferId output_buffer_id = BufferId::MOE_SHARED_EXPERT_OUTPUT;
        };

        explicit SharedExpertGateStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_SHARED_EXPERT_GATE; }
        std::string name() const override { return "shared_expert_gate"; }
        size_t estimatedFlops() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

    private:
        Params params_;

        /// Cached MoE kernel for sigmoid gating
        mutable IMoEKernel *moe_kernel_ = nullptr;
        IMoEKernel *ensureMoEKernel() const;
    };

} // namespace llaminar2
