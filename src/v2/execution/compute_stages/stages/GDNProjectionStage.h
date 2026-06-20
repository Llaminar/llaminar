/**
 * @file GDNProjectionStage.h
 * @brief GDN projection stage: 4 separate GEMMs for Gated Delta Net layers
 *
 * Computes 4 independent linear projections from the same input:
 *   - in_proj_qkv: hidden → mixed QKV [seq_len, qkv_dim]
 *   - in_proj_z:   hidden → gate Z      [seq_len, n_heads * d_v]
 *   - in_proj_a:   hidden → alpha A      [seq_len, n_heads]
 *   - in_proj_b:   hidden → beta B       [seq_len, n_heads]
 *
 * All projections share the same input tensor (activation).
 * Used by Qwen 3.5 GDN (Gated Delta Network) layers.
 */

#pragma once

#include "../IComputeStage.h"
#include "../IWorkspaceConsumerStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"
#include "../../../loaders/WeightPlan.h"

#include <memory>
#include <optional>

namespace llaminar2
{

    class ITensorGemm;
    class FP32Tensor;
    class TensorBase;
    class PreparedWeightStore;

    /**
     * @brief 4-projection stage for GDN layers
     *
     * Performs: QKV = input × W_qkv, Z = input × W_z,
     *          A = input × W_a, B = input × W_b
     */
    class GDNProjectionStage : public IComputeStage, public IWorkspaceConsumerStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            const ITensor *input = nullptr; ///< Input activation [seq_len, d_model]
            int m = 0;                      ///< seq_len (rows of input)
            int k = 0;                      ///< d_model (cols of input)

            // QKV projection
            const ITensor *w_qkv = nullptr; ///< Weight [d_model, qkv_dim]
            ITensor *output_qkv = nullptr;  ///< Output [seq_len, qkv_dim]
            int n_qkv = 0;                  ///< qkv_dim = 2*n_heads*d_k + n_heads*d_v

            // Z (gate) projection
            const ITensor *w_z = nullptr; ///< Weight [d_model, n_heads * d_v]
            ITensor *output_z = nullptr;  ///< Output [seq_len, n_heads * d_v]
            int n_z = 0;                  ///< n_heads * d_v

            // A (alpha / dt) projection
            const ITensor *w_a = nullptr; ///< Weight [d_model, n_heads]
            ITensor *output_a = nullptr;  ///< Output [seq_len, n_heads]
            int n_a = 0;                  ///< n_heads

            // B (beta) projection
            const ITensor *w_b = nullptr; ///< Weight [d_model, n_heads]
            ITensor *output_b = nullptr;  ///< Output [seq_len, n_heads]
            int n_b = 0;                  ///< n_heads

            // Cached GEMM kernels (set during graph construction)
            ITensorGemm *gemm_qkv = nullptr;
            ITensorGemm *gemm_z = nullptr;
            ITensorGemm *gemm_a = nullptr;
            ITensorGemm *gemm_b = nullptr;

            // Optional BufferIds
            std::optional<BufferId> input_buffer_id;
            std::optional<BufferId> output_qkv_buffer_id;
            std::optional<BufferId> output_z_buffer_id;
            std::optional<BufferId> output_a_buffer_id;
            std::optional<BufferId> output_b_buffer_id;

            // =================================================================
            // Phase 7: PreparedWeightRef for direct kernel resolution
            // =================================================================
            std::optional<PreparedWeightRef> prepared_ref_qkv;
            std::optional<PreparedWeightRef> prepared_ref_z;
            std::optional<PreparedWeightRef> prepared_ref_a;
            std::optional<PreparedWeightRef> prepared_ref_b;
            PreparedWeightStore *prepared_store = nullptr;

            /**
             * @brief Use grouped decode-equivalent kernels for tiny MTP verifier batches.
             *
             * GDN projection feeds recurrent state.  Publication-capable verifier
             * graphs therefore require M=2..4 grouped kernels that preserve the
             * same per-row quantization and accumulation contract as M=1 serial
             * decode.  Unsupported backends must fail loudly instead of falling
             * back to hidden row-wise replay.
             */
            bool force_decode_equivalent_verifier_prefill = false;
        };

        static_assert(StageParamsRequired<Params>);

        explicit GDNProjectionStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        bool validatePreparedWeights(std::string *error) const override;
        ComputeStageType type() const override { return ComputeStageType::GDN_PROJECTION; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;

        const Params &getParams() const { return params_; }

        // IWorkspaceConsumerStage - multi-kernel pattern (4 GEMM kernels)
        IWorkspaceConsumer *getKernelAsWorkspaceConsumer() override;
        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        void unbindWorkspace() override;

        // GDN projection uses standard GEMM path — graph-capturable
        bool isGraphCapturable() const override { return true; }

    private:
        /// Lazily resolve a GEMM kernel from a prepared weight ref.
        /// Caches the result in @p cached for subsequent calls (like GEMMStage).
        ITensorGemm *resolveGemm(
            const ITensor *weight, ITensorGemm *&cached, const char *name);

        Params params_;
    };

} // namespace llaminar2
