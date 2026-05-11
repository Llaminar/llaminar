/**
 * @file MoEExpertOverlayLocalTPExecutor.h
 * @brief Domain-scoped LocalTP TensorParallelExperts helper for MoE expert overlays.
 */

#pragma once

#include "MoEExpertOverlayRuntimePlan.h"
#include "MoEExpertTokenRowTransfer.h"
#include "collective/ILocalTPContext.h"

#include <string>
#include <vector>
#include <memory>

namespace llaminar2
{
    class FP32Tensor;
    class ITensorGemm;
    class TensorBase;
    struct MoEExpertDispatchEntry;

    struct MoEExpertOverlayLocalTPPreparedParticipant
    {
        int participant_index = -1;
        DeviceId device = DeviceId::invalid();
        std::vector<ITensorGemm *> gate_gemm;
        std::vector<ITensorGemm *> up_gemm;
        std::vector<ITensorGemm *> down_gemm;

        std::shared_ptr<FP32Tensor> input_scratch;
        std::shared_ptr<FP32Tensor> batch_scratch;
        std::shared_ptr<FP32Tensor> gate_scratch;
        std::shared_ptr<FP32Tensor> up_scratch;
        std::shared_ptr<FP32Tensor> partial_scratch;

        std::vector<int> token_indices;
        std::vector<float> token_weights;
    };

    struct MoEExpertOverlayLocalTPRunParams
    {
        TensorBase *input = nullptr;
        TensorBase *routing_indices = nullptr;
        TensorBase *routing_weights = nullptr;
        TensorBase *gate_exps = nullptr;
        TensorBase *up_exps = nullptr;
        TensorBase *down_exps = nullptr;
        TensorBase *output = nullptr;

        int seq_len = 0;
        int d_model = 0;
        int num_experts = 0;
        int top_k = 0;
        int expert_intermediate = 0;
        int layer_idx = -1;

        std::vector<bool> expert_mask;
        std::string stage_name_prefix = "moe_localtp_tensor_parallel_expert";
        bool upload_partials_to_participant_devices = true;
        std::vector<MoEExpertOverlayLocalTPPreparedParticipant> *prepared_participants = nullptr;
        std::vector<FP32Tensor *> *prepared_partial_views = nullptr;
        const std::vector<MoEExpertDispatchEntry> *dispatch_entries = nullptr;
        const std::vector<int> *dispatch_token_rows = nullptr;
    };

    struct MoEExpertOverlayLocalTPParticipantDiagnostics
    {
        int participant_index = -1;
        DeviceId device = DeviceId::invalid();
        int intermediate_start = 0;
        int intermediate_end = 0;
        int routed_entry_count = 0;
        int expert_allreduce_count = 0;
        std::vector<int> executed_expert_ids;
    };

    struct MoEExpertOverlayLocalTPDiagnostics
    {
        std::string domain_name;
        CollectiveBackendType backend = CollectiveBackendType::AUTO;
        int degree = 0;
        int total_routed_entries = 0;
        MoEExpertTransferMode transfer_mode = MoEExpertTransferMode::None;
        MoEExpertTransferVolume transfer_volume;
        std::vector<int> selected_token_rows;
        std::vector<MoEExpertOverlayLocalTPParticipantDiagnostics> participants;
        double compute_ms = 0.0;
        double domain_reduce_ms = 0.0;

        void clear()
        {
            domain_name.clear();
            backend = CollectiveBackendType::AUTO;
            degree = 0;
            total_routed_entries = 0;
            transfer_mode = MoEExpertTransferMode::None;
            transfer_volume = MoEExpertTransferVolume{};
            selected_token_rows.clear();
            participants.clear();
            compute_ms = 0.0;
            domain_reduce_ms = 0.0;
        }
    };

    class MoEExpertOverlayLocalTPExecutor
    {
    public:
        static bool canExecute(
            const MoEOverlayRuntimeDomain &domain,
            const ILocalTPContext &local_tp_context,
            std::string *reason = nullptr);

        static bool runTensorParallelExperts(
            const MoEOverlayRuntimeDomain &domain,
            ILocalTPContext &local_tp_context,
            const MoEExpertOverlayLocalTPRunParams &params,
            MoEExpertOverlayLocalTPDiagnostics *diagnostics = nullptr);
    };

} // namespace llaminar2