/**
 * @file QGateSplitStage.h
 * @brief Splits interleaved Q+gate GEMM output into separate Q and gate buffers
 *
 * Qwen 3.5 FA layers project Q to 2× normal width: [seq, n_heads * head_dim * 2].
 * The HuggingFace implementation reshapes to [seq, n_heads, head_dim * 2] then
 * splits along the last dimension via torch.chunk(..., 2, dim=-1), yielding
 * separate query [seq, n_heads, head_dim] and gate [seq, n_heads, head_dim]
 * tensors. The gate is later applied as sigmoid(gate) * attn_output before Wo.
 *
 * This stage performs the equivalent deinterleave:
 *   input:  [seq, n_heads * head_dim * 2]  (contiguous row-major)
 *   out_q:  [seq, n_heads * head_dim]      (first half of each head's 2*head_dim block)
 *   out_gate: [seq, n_heads * head_dim]    (second half of each head's 2*head_dim block)
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"

#include <optional>

namespace llaminar2
{

    class QGateSplitStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            ITensor *input = nullptr;       ///< Q GEMM output [seq, n_heads * head_dim * 2]
            ITensor *output_q = nullptr;    ///< Query output [seq, n_heads * head_dim]
            ITensor *output_gate = nullptr; ///< Gate output [seq, n_heads * head_dim]

            int seq_len = 0;  ///< Number of tokens (rows)
            int n_heads = 0;  ///< Number of attention heads
            int head_dim = 0; ///< Dimension per head

            std::optional<BufferId> input_buffer_id;
            std::optional<BufferId> output_q_buffer_id;
            std::optional<BufferId> output_gate_buffer_id;
        };

        static_assert(StageParamsRequired<Params>);

        explicit QGateSplitStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::Q_GATE_SPLIT; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;

        const Params &getParams() const { return params_; }

    private:
        Params params_;
    };

} // namespace llaminar2
