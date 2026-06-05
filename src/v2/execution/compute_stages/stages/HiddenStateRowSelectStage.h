/**
 * @file HiddenStateRowSelectStage.h
 * @brief Graph-capturable final hidden-state row selection stage.
 *
 * Copies one FP32 hidden-state row from a padded prefill bucket into a stable
 * one-row scratch tensor. The LM head can then always GEMM from row zero of the
 * scratch tensor, avoiding captured GEMM source offsets that would otherwise be
 * fixed to the first real length used by a bucket graph.
 *
 * Thread-safety: stage instances are not thread-safe; they are owned by one
 * ComputeGraph and updated by the forward execution engine on that graph's
 * execution thread.
 *
 * Lifecycle: the device scalar is a declared graph workspace buffer. The stage
 * owns only a pinned host scalar used to upload the selected row before capture
 * or graph replay. Captured execution reads the already-resident device scalar
 * and must not record a host-to-device scalar copy.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../memory/BufferId.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace llaminar2
{

    /**
     * @brief Selects the last real prefill row into a stable one-row buffer.
     *
     * CPU execution performs a direct memcpy. GPU execution uploads the selected
     * row scalar before capture/replay, then records only a fixed-grid row-copy
     * kernel that reads the graph-workspace device scalar.
     */
    class HiddenStateRowSelectStage : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        static constexpr const char *WS_SELECTED_ROW_SCALAR = "hidden_row_select_selected_row_scalar";

        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            const ITensor *input = nullptr; ///< Source hidden states [seq_len, d_model], FP32.
            ITensor *output = nullptr;      ///< Destination scratch row [1, d_model], FP32.
            int seq_len = 0;                ///< Fixed bucket sequence length.
            int d_model = 0;                ///< Hidden-state width.
            int selected_row_idx = -1;      ///< Initial selected row; -1 means seq_len - 1.

            std::optional<BufferId> input_buffer_id;  ///< Arena input binding, usually NORMALIZED.
            std::optional<BufferId> output_buffer_id; ///< Arena output binding, usually LM_HEAD_INPUT_ROW.
            std::string workspace_buffer_name;        ///< Optional stable selected-row scalar buffer name.
        };

        explicit HiddenStateRowSelectStage(Params params);
        ~HiddenStateRowSelectStage() override;

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ROW_SELECT; }
        size_t estimatedFlops() const override { return 0; }
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        CoherencePolicy coherencePolicy() const override
        {
            return (params_.input_buffer_id && params_.output_buffer_id)
                       ? CoherencePolicy::FULL
                       : CoherencePolicy::NONE;
        }
        bool hasPrefillReplayParams() const override { return true; }
        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        void unbindWorkspace() override;
        bool hasWorkspace() const override { return bound_workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return bound_workspace_; }

        /**
         * @brief Update selected source row for fixed-bucket prefill replay.
         *
         * @param replay Real-token and bucket metadata for the next prefill graph launch.
         */
        void updatePrefillReplayParams(const PrefillReplayParams &replay) override;

        /// @brief Return the currently selected source row, mainly for tests.
        int selectedRowForTesting() const { return selected_row_idx_; }

        /**
         * @brief Update the selected source row for direct graph replay users.
         *
         * The workspace buffer name and bound workspace remain stable; only the
         * pinned scalar value consumed by the captured GPU row-select path changes.
         */
        void setSelectedRowForReplay(int selected_row_idx);

    private:
        struct GpuParamState;

        Params params_;
        int selected_row_idx_ = 0;
        std::unique_ptr<GpuParamState> gpu_state_;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
        uint32_t workspace_slice_id_ = 0;

        int normalizeSelectedRow(int requested_row) const;
        std::string selectedRowScalarBufferName() const;
        bool validateCommon(TensorBase **input_base, TensorBase **output_base);
        bool executeCPU(TensorBase *input_base, TensorBase *output_base);
        bool executeGPU(TensorBase *input_base, TensorBase *output_base);
        bool ensureGpuParamStateInitialized();
        bool uploadGpuSelectedRow();
        void refreshPinnedSelectedRow();
        void releaseGpuParamState();
    };

} // namespace llaminar2
