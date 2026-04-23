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

    // Forward declarations for GPU GEMM support
    class ITensorGemm;

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

            // Per-expert 2D tensor views — used by GPU path
            // Each vector has num_experts entries; each entry is a 2D view
            // into the corresponding 3D packed tensor.
            // Set by graph builder via extractExpertViews().
            std::vector<std::shared_ptr<TensorBase>> expert_gate_views; ///< [intermediate, d_model] per expert
            std::vector<std::shared_ptr<TensorBase>> expert_up_views;   ///< [intermediate, d_model] per expert
            std::vector<std::shared_ptr<TensorBase>> expert_down_views; ///< [d_model, intermediate] per expert

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
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

        /// Extract 2D expert views from 3D packed tensors.
        /// Call once at graph-build time. Views are stored in params.
        static bool extractExpertViews(Params &params);

    private:
        Params params_;

        /// Stashed routing results from last execute() — for snapshot capture.
        /// Stored as FP32 [seq_len, top_k] so buildDumpInfoImpl() can expose them.
        mutable std::vector<float> routing_indices_f32_;  ///< Expert IDs cast to float
        mutable std::vector<float> routing_weights_;      ///< Normalized top-k weights
        mutable std::vector<float> router_logits_;        ///< Raw router logits [seq_len, num_experts]

        /// CPU execution path: inline dequantization + scalar GEMV
        bool executeCPU(IDeviceContext *ctx);

        /// GPU execution path: KernelFactory GEMM per active expert
        bool executeGPU(IDeviceContext *ctx);

        /// Execute routing: softmax top-k selection (CPU always)
        bool executeRouting(
            const float *hidden, int seq_len, int d_model,
            const float *gate_w, int num_experts, int top_k,
            std::vector<int> &expert_indices,
            std::vector<float> &expert_weights) const;

        /// Stash routing results for snapshot capture
        void stashRoutingResults(
            const std::vector<int> &expert_indices,
            const std::vector<float> &expert_weights,
            int seq_len, int top_k) const;

        /// Execute SwiGLU FFN for a single expert on gathered tokens (CPU path)
        bool executeExpertFFN(
            const float *input_tokens, int num_tokens, int d_model,
            const float *gate_w, const float *up_w, const float *down_w,
            int intermediate, float *output) const;
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

            TensorBase *input = nullptr;    ///< Normalized hidden [seq_len, d_model]
            TensorBase *gate_w = nullptr;   ///< Shared expert gate [intermediate, d_model]
            TensorBase *up_w = nullptr;     ///< Shared expert up [intermediate, d_model]
            TensorBase *down_w = nullptr;   ///< Shared expert down [d_model, intermediate]
            TensorBase *output = nullptr;   ///< Output [seq_len, d_model]
            int seq_len = 0;
            int d_model = 0;
            int intermediate = 0;

            BufferId input_buffer_id = BufferId::NORMALIZED;
            BufferId output_buffer_id = BufferId::MOE_SHARED_EXPERT_OUTPUT;
        };

        explicit SharedExpertFFNStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_EXPERT_FFN; }
        std::string name() const override { return "shared_expert_ffn"; }
        size_t estimatedFlops() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

    private:
        Params params_;

        bool executeCPU_SharedExpert(IDeviceContext *ctx);
        bool executeGPU_SharedExpert(IDeviceContext *ctx);
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

            TensorBase *input = nullptr;          ///< Normalized hidden [seq_len, d_model]
            TensorBase *gate_inp = nullptr;        ///< Gate vector [d_model]
            TensorBase *shared_output = nullptr;   ///< Shared expert output (in-place) [seq_len, d_model]
            int seq_len = 0;
            int d_model = 0;

            BufferId input_buffer_id = BufferId::NORMALIZED;
            BufferId output_buffer_id = BufferId::MOE_SHARED_EXPERT_OUTPUT;
        };

        explicit SharedExpertGateStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_COMBINE; }
        std::string name() const override { return "shared_expert_gate"; }
        size_t estimatedFlops() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

    private:
        Params params_;
    };

} // namespace llaminar2
