/**
 * @file HiddenStateRowsSelectStage.h
 * @brief Graph-capturable compact hidden-state multi-row selection stage.
 *
 * Packs a small fixed number of FP32 hidden-state rows into a dense scratch
 * tensor. This is used by MTP verifier paths to feed one batched LM-head GEMM
 * instead of either projecting every verifier row or looping one-row helpers.
 *
 * GPU row ownership is explicit. Ordinary compact graphs upload a stage-owned
 * row plan, verifier graphs consume an externally produced device plan, and
 * request-batched prefill derives terminal rows directly from resident request
 * lengths. Every mode uses a stable device address and an explicit stream, so a
 * captured graph never adopts mutable host row state.
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

        /**
         * @brief Selects the authoritative source of GPU row indices.
         *
         * The source is part of the graph contract rather than a runtime
         * fallback. `StageOwnedIndices` is a CPU/direct-fixture policy and is
         * forbidden for production GPU execution because it requires a pinned
         * host upload. `FixedContiguousRange` encodes an immutable verifier
         * suffix directly in a captured D2D node. `ExternalDeviceIndices` lets a
         * preceding device metadata kernel publish arbitrary verifier rows.
         * `RequestTerminalLengths` computes one terminal row per padded request
         * directly in the copy kernel. `ShiftedPrefillKVProgress` computes the
         * next contiguous shifted-prefill range from canonical device KV counts
         * and the same resident request geometry. Both policies preserve full
         * device ownership and remain stable across captured graph replay.
         */
        enum class DeviceRowIndexSource
        {
            StageOwnedIndices,
            FixedContiguousRange,
            ExternalDeviceIndices,
            RequestTerminalLengths,
            ShiftedPrefillKVProgress,
        };

        /**
         * @brief Selects the authoritative padded-row-stride owner.
         *
         * Main forward graphs have one immutable shape and therefore encode
         * their stride in graph topology. Reusable MTP publication graphs are
         * total over prompt widths and read the stride from the same
         * event-published device record as the request lengths. The two modes
         * are explicit policies; execution never probes one and falls back to
         * the other.
         */
        enum class RequestRowStrideSource
        {
            StaticGraphGeometry,
            ExternalDeviceScalar,
        };

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
            DeviceRowIndexSource device_row_index_source =
                DeviceRowIndexSource::StageOwnedIndices; ///< Authoritative GPU row-index policy.
            int fixed_contiguous_row_start = 0; ///< First immutable source row for FixedContiguousRange.
            std::string workspace_buffer_name; ///< Stable row-index workspace for stage-owned or external-device plans.
            const int32_t *request_sequence_lengths_device = nullptr; ///< Resident request lengths for RequestTerminalLengths.
            int request_row_stride = 0; ///< Padded source-row stride between requests.
            RequestRowStrideSource request_row_stride_source =
                RequestRowStrideSource::StaticGraphGeometry; ///< Typed owner of request_row_stride.
            const int32_t *request_row_stride_device = nullptr; ///< Stable device scalar for ExternalDeviceScalar.
            const int32_t *main_cached_tokens_device = nullptr; ///< Canonical main-KV count for ShiftedPrefillKVProgress.
            const int32_t *shifted_cached_tokens_device = nullptr; ///< Canonical shifted-MTP KV count for ShiftedPrefillKVProgress.
            int request_index = -1; ///< Immutable request row selected by ShiftedPrefillKVProgress.
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
