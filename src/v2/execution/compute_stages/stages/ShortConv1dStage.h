/**
 * @file ShortConv1dStage.h
 * @brief Short causal depthwise conv1d stage for GDN layers
 *
 * Applies a causal depthwise convolution (kernel_size=4 typically) followed
 * by SiLU activation on the mixed QKV projection output. Maintains a small
 * conv_state for incremental decode (kernel_size - 1 history frames).
 *
 * Prefill: conv1d over the full sequence, stores tail in conv_state
 * Decode:  conv1d_update using conv_state, outputs single timestep
 *
 * Delegates to ITensorShortConvolution kernel interface for the actual conv
 * computation, enabling device-specific implementations (CPU, CUDA).
 *
 * Reference: HuggingFace Qwen3_5GatedDeltaNet.forward() conv1d path
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"
#include "../../../interfaces/IWorkspaceConsumer.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace llaminar2
{

    class ITensorShortConvolution;

    /**
     * @brief Causal depthwise conv1d + SiLU for GDN QKV preprocessing
     *
     * Computes: output = SiLU(DepthwiseConv1D(input, weight, bias))
     * with causal padding and conv_state management for decode.
     *
     * Delegates to ITensorShortConvolution* kernel for the actual computation.
     */
    class ShortConv1dStage : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        static constexpr const char *WS_INPLACE_PREFILL_SCRATCH = "gdn_shortconv_inplace_scratch";
        static constexpr const char *WS_EFFECTIVE_SEQ_LEN_SCALAR = "gdn_shortconv_effective_seq_len_scalar";
        static constexpr const char *WS_VERIFIER_STATE_CAPTURE = "gdn_shortconv_verifier_state_capture";
        static constexpr const char *WS_SPECULATIVE_STATE_SLOTS = "gdn_shortconv_verifier_state_capture";
        static constexpr const char *WS_SPECULATIVE_STATE_WORK = "gdn_shortconv_speculative_state_work";

        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            ITensor *input = nullptr;        ///< Input [seq_len, channels] (modified in-place for decode)
            ITensor *output = nullptr;       ///< Output [seq_len, channels]
            const ITensor *weight = nullptr; ///< Conv weight [channels, kernel_size] (squeezed from [channels, 1, kernel_size])
            const ITensor *bias = nullptr;   ///< Optional conv bias [channels]

            float *conv_state = nullptr; ///< Conv state buffer [channels, kernel_size-1] (from GDNLayerState)
            int seq_len = 0;             ///< Sequence length
            int channels = 0;            ///< Number of channels (= QKV dim)
            int kernel_size = 4;         ///< Convolution kernel width
            int layer_idx = -1;          ///< Logical model layer for stable graph workspace naming.
            int verifier_state_capture_rows = 0; ///< Compatibility spelling for speculative state slots.
            int speculative_state_slot_rows = 0; ///< Phase 13.8 temporary state slots for MTP verifier rows.

            /// Kernel implementation (set during graph construction)
            ITensorShortConvolution *kernel = nullptr;

            // Optional BufferIds
            std::optional<BufferId> input_buffer_id;
            std::optional<BufferId> output_buffer_id;
        };

        static_assert(StageParamsRequired<Params>);

        explicit ShortConv1dStage(Params params);
        ~ShortConv1dStage() override;

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::SHORT_CONV1D; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;

        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        void unbindWorkspace() override;
        bool hasWorkspace() const override { return bound_workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return bound_workspace_; }

        void updateDynamicParams(int pos_offset, int seq_len) override
        {
            (void)pos_offset; // Conv1d doesn't use position offsets
            params_.seq_len = seq_len;
        }
        bool hasDynamicParams() const override { return true; }
        void resetSessionState() override
        {
            IComputeStage::resetSessionState();
            prefill_effective_seq_len_ = 0;
            prefill_bucket_seq_len_ = 0;
            prefill_replay_params_set_ = false;
            if (params_.kernel)
                params_.kernel->setGPUStream(nullptr);
        }

        bool hasPrefillReplayParams() const override { return true; }
        void updatePrefillReplayParams(const PrefillReplayParams &replay) override;
        bool supportsPaddedPrefillRealLengthContract() const override;
        bool hasVerifierStateCapture() const override;
        bool restoreVerifierStateCaptureRow(int row, void *stream = nullptr) override;
        bool publishAcceptedSpeculativeStateFromDeviceMetadata(
            const int32_t *device_accepted_state_slot_indices,
            int request_index,
            void *stream) override;

        // Short conv1d operates fully on-device when GPU is active — graph-capturable
        bool isGraphCapturable() const override { return true; }

        const Params &getParams() const { return params_; }

    private:
        struct GpuEffectiveSeqLenState;

        Params params_;
        int prefill_effective_seq_len_ = 0;
        int prefill_bucket_seq_len_ = 0;
        bool prefill_replay_params_set_ = false;
        std::unique_ptr<GpuEffectiveSeqLenState> gpu_effective_seq_len_state_;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
        uint32_t workspace_slice_id_ = 0;
        bool verifier_capture_workspace_bound_ = false;
        bool speculative_state_work_bound_ = false;
        int verifier_capture_rows_bound_ = 0;
        int verifier_capture_state_size_bound_ = 0;

        int effectivePrefillSeqLen() const;
        bool shouldUseRealLengthContract() const;
        std::string workspaceStableId() const;
        std::string effectiveSeqLenScalarBufferName() const;
        std::string verifierStateCaptureBufferName() const;
        std::string speculativeStateWorkBufferName() const;
        int requestedSpeculativeStateSlotRows() const;
        bool verifierStateCaptureWorkspaceRequired() const;
        bool ensureVerifierStateCaptureWorkspaceBound() const;
        bool ensureGpuEffectiveSeqLenStateInitialized();
        bool uploadGpuEffectiveSeqLen();
        void refreshPinnedEffectiveSeqLen();
        void releaseGpuEffectiveSeqLenState();
        void bindKernelWorkspace();
    };

} // namespace llaminar2
