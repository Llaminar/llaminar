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
#include <vector>

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
        static constexpr const char *WS_SPECULATIVE_STATE_SLOTS = "gdn_shortconv_speculative_state_slots";
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
            int request_count = 1;       ///< Number of independent requests in the flattened verifier tensor.
            int request_seq_len = 0;     ///< Per-request rows before flattening; 0 means seq_len for legacy graphs.
            /**
             * @brief Host-owned real row counts for CPU request batching.
             *
             * The vector is orchestration-state storage whose address remains
             * valid for the graph lifetime. CPU grouped kernels read it
             * directly; GPU kernels use the resident pointer below.
             */
            const std::vector<int> *request_seq_lens_host = nullptr;
            /**
             * @brief Device-owned real row count for each request.
             *
             * Non-null only for GPU request batches whose flattened rows may
             * contain padding. The pointer is arena-owned and stable across
             * graph capture/replay; kernels clamp each value to
             * `request_seq_len` and never read the host sequence-length vector.
             */
            const int32_t *request_seq_lens_device = nullptr;
            int channels = 0;            ///< Number of channels (= QKV dim)
            int kernel_size = 4;         ///< Convolution kernel width
            int layer_idx = -1;          ///< Logical model layer for stable graph workspace naming.
            /**
             * @brief Stable graph/workspace namespace for capture-sensitive buffers.
             *
             * Main inference, grouped verifier, and live request-batch graphs
             * may execute independently for the same logical layer. The
             * namespace keeps both short-conv verifier-state snapshots and
             * mutable prefill scratch graph-role local. An empty namespace is
             * reserved for the main inference graph.
             */
            std::string workspace_namespace;
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
        ~ShortConv1dStage() override = default;

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

        /**
         * @brief Refresh the logical row geometry for the next graph execution.
         *
         * @param pos_offset Unused by short convolution because its history is
         *        carried by the request-local convolution state.
         * @param seq_len Number of rows contributed by one request.  For a
         *        request-batched graph, the stage kernel still receives one
         *        flattened tensor containing `request_count * seq_len` rows.
         *
         * The execution engine expresses dynamic sequence length in the same
         * per-request domain used to build a request-batched graph.  Keeping
         * `Params::seq_len` in that domain would make the flattened geometry
         * internally inconsistent and would either reject the grouped kernel
         * or let one request consume another request's rows.  This method keeps
         * both representations synchronized whenever a cached graph is reused.
         */
        void updateDynamicParams(int pos_offset, int seq_len) override;
        bool hasDynamicParams() const override { return true; }
        bool supportsDeviceResidentDynamicPositionReplay() const override
        {
            return true;
        }
        void resetSessionState() override
        {
            IComputeStage::resetSessionState();
            prefill_effective_seq_len_ = 0;
            prefill_bucket_seq_len_ = 0;
            prefill_replay_params_set_ = false;
            if (params_.kernel)
            {
                params_.kernel->setGPUStream(nullptr);
                clearKernelVerifierStateWorkspace();
            }
        }

        /**
         * @brief Reset request-local short-conv metadata while preserving capture slots.
         *
         * Preserved prefill graphs replay over the same verifier-state capture
         * workspace and stable device request-length allocation. Request reset
         * therefore clears stream ownership without introducing a host scalar
         * mirror into graph replay.
         */
        void resetSessionStatePreservingCapturedReplay() override
        {
            IComputeStage::resetSessionState();
            prefill_effective_seq_len_ = 0;
            prefill_bucket_seq_len_ = 0;
            prefill_replay_params_set_ = false;
            if (params_.kernel)
                params_.kernel->setGPUStream(nullptr);
        }

        /**
         * @brief Preserve warmed short-conv workspaces for capture-from-Initialized.
         */
        void resetSessionStatePreservingLazyInitialization() override
        {
            resetSessionStatePreservingCapturedReplay();
        }

        bool hasPrefillReplayParams() const override { return true; }
        void updatePrefillReplayParams(const PrefillReplayParams &replay) override;
        bool supportsPaddedPrefillRealLengthContract() const override;
        bool hasVerifierStateCapture() const override;
        bool requiresVerifierStateCaptureForPublication() const override
        {
            return verifierStateCaptureWorkspaceRequired();
        }
        bool restoreVerifierStateCaptureRow(int row, void *stream = nullptr) override;
        bool restoreVerifierStateCaptureRows(
            const int *host_row_indices,
            int request_count,
            void *stream = nullptr) override;
        bool restoreVerifierStateCaptureRowFromDeviceIndex(
            const int *device_row_index,
            void *stream) override;
        /**
         * @brief Restore one captured short-conv state row per request.
         *
         * Batched publication is only correct when the backend owns a separate
         * live conv-state slot per request.  This hook exists so CUDA, ROCm,
         * and CPU can share the same publication contract once those state
         * banks are implemented; it must not be replaced by a scalar loop.
         */
        bool restoreVerifierStateCaptureRowsFromDeviceIndices(
            const int *device_row_indices,
            int request_count,
            int row_index_stride,
            void *stream) override;
        /**
         * @brief Publish every request's real terminal conv state on device.
         *
         * The backend derives flat capture rows from resident request lengths
         * and writes the request-owned live state bank in one grouped launch.
         */
        bool restoreVerifierStateCaptureRequestTerminalRows(
            const int *device_request_seq_lens,
            int request_count,
            int request_row_width,
            void *stream) override;
        /**
         * @brief Report direct grouped publication into request-owned live banks.
         *
         * When a CPU or GPU request matrix is larger than the bounded verifier
         * snapshot window, the grouped short-convolution kernel advances each
         * request's dedicated live bank in place. The executor must recognize
         * that completed transaction instead of attempting a nonexistent
         * terminal-row restore.
         */
        bool requestBatchedTerminalStateCommittedDuringExecution(
            int request_count,
            int request_row_width) const override;
        void clearVerifierStateCaptureBindingAfterPublication() override;
        void onGraphReplayed() override;
        bool needsOnGraphReplayed() const override { return params_.kernel != nullptr; }
        // Short conv1d operates fully on-device when GPU is active — graph-capturable
        bool isGraphCapturable() const override { return true; }

        const Params &getParams() const { return params_; }

    private:
        Params params_;
        int prefill_effective_seq_len_ = 0;
        int prefill_bucket_seq_len_ = 0;
        bool prefill_replay_params_set_ = false;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
        uint32_t workspace_slice_id_ = 0;
        bool verifier_capture_workspace_bound_ = false;
        bool speculative_state_work_bound_ = false;
        int verifier_capture_rows_bound_ = 0;
        int verifier_capture_state_size_bound_ = 0;
        std::vector<float> host_verifier_state_slots_;

        int effectivePrefillSeqLen() const;
        bool shouldUseRealLengthContract() const;
        std::string workspaceStableId() const;
        std::string inplacePrefillScratchBufferName() const;
        std::string speculativeStateSlotsBufferName() const;
        std::string speculativeStateWorkBufferName() const;
        int requestedSpeculativeStateSlotRows() const;
        bool verifierStateCaptureWorkspaceRequired() const;
        bool ensureVerifierStateCaptureWorkspaceBound() const;
        void bindKernelWorkspace();
        void clearKernelVerifierStateWorkspace();
        const float *cpuVerifierStateCaptureSource() const;
        bool restoreCPUVerifierStateCaptureRowDirect(int row);
    };

} // namespace llaminar2
