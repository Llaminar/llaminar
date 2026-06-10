/**
 * @file HiddenStateRowsSelectStage.h
 * @brief Graph-capturable compact hidden-state multi-row selection stage.
 *
 * Packs a small fixed number of FP32 hidden-state rows into a dense scratch
 * tensor. This is used by MTP verifier paths to feed one batched LM-head GEMM
 * instead of either projecting every verifier row or looping one-row helpers.
 *
 * Lifecycle: replay setters update host-side intent only. GPU row-index uploads
 * happen from executeGPU(), after DeviceGraphExecutor has rebound the current
 * workspace manager and an explicit non-null stream. This avoids stale workspace
 * handoff bugs when cached graph objects outlive a previous workspace manager.
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
#include <vector>

namespace llaminar2
{

    /**
     * @brief Selects several source rows into a compact scratch tensor.
     *
     * CPU execution copies each selected contiguous row. GPU execution uploads
     * the selected row indices only while the stage is executed under executor
     * ownership, then runs one fixed-shape row-packing kernel on the explicit
     * stage stream.
     */
    class HiddenStateRowsSelectStage : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        static constexpr const char *WS_SELECTED_ROWS_ARRAY = "hidden_rows_select_selected_rows_array";

        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            const ITensor *input = nullptr; ///< Source hidden states [seq_len, d_model], FP32.
            ITensor *output = nullptr;      ///< Destination scratch rows [selected_row_count, d_model], FP32.
            int seq_len = 0;                ///< Fixed source sequence length.
            int d_model = 0;                ///< Hidden-state width.
            int selected_row_count = 0;     ///< Fixed number of rows packed by this graph stage.
            std::vector<int> selected_row_indices; ///< Initial selected source rows.

            std::optional<BufferId> input_buffer_id;
            std::optional<BufferId> output_buffer_id;
            std::string workspace_buffer_name;
            bool declare_selected_rows_workspace = true; ///< Declare row-index workspace when this stage owns it.
            bool upload_selected_rows_to_workspace = true; ///< Upload host row indices; false means an external metadata producer owns contents.
        };

        explicit HiddenStateRowsSelectStage(Params params);
        ~HiddenStateRowsSelectStage() override;

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
        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        void unbindWorkspace() override;
        bool hasWorkspace() const override { return bound_workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return bound_workspace_; }
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        bool needsGraphLaunchPreparation() const override { return params_.device_id.is_gpu(); }

        /**
         * @brief Update selected source rows for direct graph replay users.
         *
         * The selected-row count is fixed by construction. The new vector must
         * have exactly that size; the method returns false and leaves previous
         * indices intact when the shape would change. This method never
         * dereferences a bound workspace; executeGPU() performs the upload after
         * executor-managed workspace and stream binding.
         */
        bool setSelectedRowsForReplay(const std::vector<int> &selected_row_indices);

        const std::vector<int> &selectedRowsForTesting() const { return selected_rows_; }
        int selectedRowCountForTesting() const { return selected_row_count_; }

    private:
        struct GpuParamState;

        Params params_;
        int selected_row_count_ = 0;
        std::vector<int> selected_rows_;
        std::unique_ptr<GpuParamState> gpu_state_;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
        uint32_t workspace_slice_id_ = 0;

        int normalizeSelectedRow(int requested_row) const;
        std::vector<int> defaultSelectedRows() const;
        std::string selectedRowsBufferName() const;
        bool validateCommon(TensorBase **input_base, TensorBase **output_base);
        bool executeCPU(TensorBase *input_base, TensorBase *output_base);
        bool executeGPU(TensorBase *input_base, TensorBase *output_base);
        bool ensureGpuParamStateInitialized();
        bool uploadGpuSelectedRows();
        void refreshPinnedSelectedRows();
        void releaseGpuParamState();
    };

} // namespace llaminar2
