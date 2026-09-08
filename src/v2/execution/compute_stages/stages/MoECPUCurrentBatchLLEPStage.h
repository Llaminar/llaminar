/**
 * @file MoECPUCurrentBatchLLEPStage.h
 * @brief Explicit CPU current-batch LLEP plan/transfer/restore graph stage.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../moe/CPUCurrentBatchLLEP.h"
#include "../../../memory/BufferId.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
    class ITPContext;

    /**
     * @brief Plans/materializes or restores one transient CPU LLEP transaction.
     *
     * Begin is an explicit cross-rank graph collective. Restore is participant-
     * local but remains a graph node so transient residency cannot leak into a
     * subsequent layer or decode graph.
     */
    class MoECPUCurrentBatchLLEPStage final : public IComputeStage
    {
    public:
        /** @brief Immutable graph bindings for one transaction phase. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            CPUCurrentBatchLLEPPhase phase =
                CPUCurrentBatchLLEPPhase::Begin;
            /** Physical packed-weight executor; never a placement authority. */
            ICPUCurrentBatchLLEPPhysicalExecutor *physical_executor = nullptr;
            /** Sole durable authority that lends this child its parent epoch. */
            std::shared_ptr<MoEOverlayResidencyAuthority>
                residency_authority;
            /** Exact routed domain used to translate global to local owners. */
            std::string residency_domain;
            ITPContext *tp_ctx = nullptr;
            ICPUCurrentBatchLLEPExpertConsumer *expert_consumer = nullptr;
            std::shared_ptr<CPUCurrentBatchLLEPTransactionState> state;

            const ITensor *routing_indices = nullptr;
            const ITensor *routing_weights = nullptr;
            std::optional<BufferId> routing_indices_buffer_id;
            std::optional<BufferId> routing_weights_buffer_id;

            int layer_idx = -1;
            int seq_len = 0;
            int top_k = 0;
            int num_experts = 0;
            int participant_count = 0;
            int participant_id = -1;
            least_loaded_ep::LeastLoadedExpertAssignmentConfig planner_config;
            std::string stage_name;
        };

        static_assert(StageParamsRequired<Params>);

        /** @brief Construct and pre-size every transaction-owned host buffer. */
        explicit MoECPUCurrentBatchLLEPStage(Params params);

        /** @brief Execute the configured begin or restore transaction phase. */
        bool execute(IDeviceContext *ctx) override;

        /** @return Stable operation type used by graph policy and diagnostics. */
        ComputeStageType type() const override
        {
            return ComputeStageType::MOE_CPU_CURRENT_BATCH_LLEP;
        }

        /** @return Graph-provided stage identity. */
        std::string name() const override { return params_.stage_name; }

        /** @return true only for the CPU backend. */
        bool supportsBackend(ComputeBackendType backend) const override;

        /** @return false because this host MPI transaction is CPU-only. */
        bool isGraphCapturable() const override { return false; }

        /** @return true for begin, which owns the packed-weight collective. */
        bool isCollectiveStage() const override
        {
            return params_.phase == CPUCurrentBatchLLEPPhase::Begin;
        }

        /** @return Tensor requirements read by this phase. */
        StageBufferRequirements getBufferRequirements() const override;

        /** @return Arena coherence contract for routing inputs. */
        StageBufferContract bufferContract() const override;

        /** @return Snapshot metadata for route planning diagnostics. */
        StageDumpInfo buildDumpInfoImpl() const override;

        /** @return Shared graph-owned transaction publication. */
        const std::shared_ptr<CPUCurrentBatchLLEPTransactionState> &state() const
        {
            return params_.state;
        }

    private:
        bool executeBegin();
        bool executeRestore();
        bool validateBindings() const;
        bool verifyHashAgreement(uint64_t local_hash,
                                 std::vector<uint64_t> &gathered,
                                 const char *publication) const;

        Params params_;
    };

} // namespace llaminar2
