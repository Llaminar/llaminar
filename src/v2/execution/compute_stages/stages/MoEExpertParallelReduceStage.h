/**
 * @file MoEExpertParallelReduceStage.h
 * @brief Cross-domain partial reduction for MoE expert-parallel tiers.
 *
 * Supports two partial layouts:
 *  - Dense [rows, cols]: the tier stage scatter-adds internally and returns the
 *    full sequence length tensor. This is the current production path.
 *  - Sparse [selected_rows.size(), cols]: the tier stage returns only the rows
 *    it processed. The reduce stage scatter-adds them into the output.
 *    Bridge Phase 7A graph-ready interface: tier stages still produce dense
 *    partials; optimized sparse mode requires caller-provided expansion buffers
 *    until a GPU-native scatter-add kernel lands.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../backends/DeviceId.h"
#include "../../../memory/BufferId.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    class TensorBase;
    struct MoEOverlayDomainWorkResult;

    enum class MoEExpertParallelReduceMode
    {
        HostStagedCorrectness,
        ContinuationDeviceOptimized,
    };

    const char *toString(MoEExpertParallelReduceMode mode);

    enum class MoEExpertParallelReducePartialAccumulationPath
    {
        ContinuationDeviceAccumulated,
        HostStagedThenDeviceAccumulated,
        HostSummedCorrectnessFallback,
    };

    const char *toString(MoEExpertParallelReducePartialAccumulationPath path);

    struct MoEExpertParallelReducePartialInfo
    {
        std::string name;
        std::string source_domain;
        DeviceId source_device = DeviceId::invalid();
        /// If non-empty, this partial is a compact sparse tensor with shape
        /// [selected_rows.size(), cols]. The reduce stage scatter-adds these rows
        /// into the output at the specified row indices.
        /// If empty, the partial is a dense [rows, cols] tensor and is added
        /// element-wise across all live rows.
        std::vector<int> selected_rows;
    };

    struct MoEExpertParallelReducePartialDiagnostics
    {
        std::string name;
        std::string source_domain;
        DeviceId source_device = DeviceId::invalid();
        size_t bytes = 0;
        bool host_sync_required = false;
        bool source_is_continuation = false;
        bool is_sparse = false;           ///< True when partial uses compact selected_rows layout
        size_t sparse_row_count = 0;      ///< Number of selected rows when is_sparse=true
        MoEExpertParallelReducePartialAccumulationPath accumulation_path =
            MoEExpertParallelReducePartialAccumulationPath::HostSummedCorrectnessFallback;
    };

    struct MoEExpertParallelReduceDiagnostics
    {
        MoEExpertParallelReduceMode mode = MoEExpertParallelReduceMode::HostStagedCorrectness;
        std::string continuation_domain;
        DeviceId continuation_device = DeviceId::invalid();
        bool host_staged = true;
        bool output_resident_on_continuation = false;
        size_t partial_count = 0;
        size_t sparse_partial_count = 0;  ///< Number of compact sparse partials (Bridge Phase 7A interface; currently 0 in production)
        size_t input_bytes = 0;
        size_t host_staged_read_bytes = 0;
        size_t device_to_host_bytes = 0;
        size_t host_to_device_bytes = 0;
        size_t total_transfer_bytes = 0;
        double reduce_ms = 0.0;
        std::vector<MoEExpertParallelReducePartialDiagnostics> partials;

        void clear();
    };

    class MoEExpertParallelReduceStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            std::vector<const ITensor *> partials; ///< FP32 partial tensors: dense [rows,cols] or sparse [selected_rows.size(),cols]
            std::vector<std::shared_ptr<TensorBase>> partial_lifetimes; ///< Graph-owned partial tensor storage
            std::vector<MoEExpertParallelReducePartialInfo> partial_infos; ///< Optional per-partial source metadata
            std::vector<std::shared_ptr<MoEOverlayDomainWorkResult>> runtime_partial_results; ///< Optional runtime-service results aligned with partials
            std::vector<TensorBase *> sparse_expansion_scratch; ///< Dense [rows,cols] FP32 scratch for optimized sparse partials
            std::vector<std::shared_ptr<TensorBase>> sparse_expansion_scratch_lifetimes; ///< Optional owner for sparse_expansion_scratch
            ITensor *output = nullptr;             ///< Dense FP32 continuation output [rows, cols]
            BufferId output_buffer_id = BufferId::_COUNT; ///< Arena output id when this reduce writes a managed buffer
            size_t rows = 0;                       ///< Optional expected rows (0 = infer from output)
            size_t cols = 0;                       ///< Optional expected cols (0 = infer from output)
            int layer_idx = -1;

            MoEExpertParallelReduceMode mode = MoEExpertParallelReduceMode::HostStagedCorrectness;
            std::string continuation_domain;
            DeviceId continuation_device = DeviceId::cpu();
            MoEExpertParallelReduceDiagnostics *diagnostics = nullptr;
            std::shared_ptr<MoEExpertParallelReduceDiagnostics> diagnostics_lifetime;
        };

        static_assert(StageParamsRequired<Params>);

        explicit MoEExpertParallelReduceStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_EXPERT_PARALLEL_REDUCE; }
        std::string name() const override { return "moe_expert_parallel_reduce"; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override { return false; }
        bool allowsZeroOutput() const override { return true; }
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
        StageBufferContract bufferContract() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

        const Params &params() const { return params_; }

    private:
        Params params_;
    };

} // namespace llaminar2
