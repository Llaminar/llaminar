/**
 * @file MoEExpertOverlayCPUFallback.h
 * @brief Domain-scoped CPU fallback execution helper for MoE expert overlays.
 */

#pragma once

#include "MoEExpertParallelPlan.h"
#include "MoEExpertTokenRowTransfer.h"

#include "collective/IGlobalTPContext.h"

#include <mpi.h>

#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{
    class IDeviceContext;
    class PreparedWeightStore;
    class TensorBase;

    struct MoECPUFallbackDomainConfig
    {
        ExpertComputeDomain domain;
        MPI_Comm world_comm = MPI_COMM_WORLD;
        int root_world_rank = 0;
        int domain_id = -1;
        std::string hostfile_path;
    };

    struct MoECPUFallbackDomainContext
    {
        std::shared_ptr<IGlobalTPContext> tp_context;
        std::vector<int> world_ranks;
        MPI_Comm world_comm = MPI_COMM_WORLD;
        int world_rank = -1;
        int world_size = 0;
        int domain_id = -1;
        int root_world_rank = 0;
        int root_domain_index = -1;
        std::string domain_name;
        ExpertDomainComputeKind compute_kind = ExpertDomainComputeKind::ReplicatedExperts;

        bool participates() const { return tp_context != nullptr; }
        int myDomainIndex() const { return tp_context ? tp_context->myIndex() : -1; }
        int degree() const { return static_cast<int>(world_ranks.size()); }
    };

    struct MoECPUFallbackTensorParallelStats
    {
        int domain_index = -1;
        int domain_degree = 0;
        int intermediate_start = 0;
        int intermediate_end = 0;
        int expert_allreduce_count = 0;
        std::vector<int> processed_expert_ids;
    };

    struct MoECPUFallbackTransferStats
    {
        MoEExpertTransferMode mode = MoEExpertTransferMode::DenseFullSequence;
        std::vector<int> token_rows;
        MoEExpertTransferVolume volume;
    };

    struct MoECPUFallbackRunParams
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

        std::vector<bool> fallback_expert_mask;
        MoEExpertTransferMode transfer_mode = MoEExpertTransferMode::Auto;
        std::vector<int> transfer_token_rows;
        PreparedWeightStore *prepared_store = nullptr;
        MoECPUFallbackTensorParallelStats *tensor_parallel_stats = nullptr;
        MoECPUFallbackTransferStats *transfer_stats = nullptr;
    };

    class MoEExpertOverlayCPUFallback
    {
    public:
        static int stableDomainId(const std::string &domain_name);

        static std::vector<int> domainWorldRanks(const ExpertComputeDomain &domain);

        static MoECPUFallbackDomainContext createNodeLocalTPDomain(
            const MoECPUFallbackDomainConfig &config);

        static std::vector<bool> participantExpertMask(
            const std::vector<bool> &fallback_expert_mask,
            int num_experts,
            int domain_degree,
            int domain_index);

        static bool transferInputsToFallbackDomain(
            const MoECPUFallbackDomainContext &domain,
            const std::vector<TensorBase *> &tensors,
            int tag_base = 41000);

        static bool reduceFallbackPartialToRoot(
            const MoECPUFallbackDomainContext &domain,
            TensorBase *partial,
            int tag = 42000);

        static bool runReplicatedExpertFallback(
            const MoECPUFallbackDomainContext &domain,
            const MoECPUFallbackRunParams &params,
            IDeviceContext *device_context);

        static bool runTensorParallelExpertFallback(
            const MoECPUFallbackDomainContext &domain,
            const MoECPUFallbackRunParams &params);

        static bool runExpertFallback(
            const MoECPUFallbackDomainContext &domain,
            const MoECPUFallbackRunParams &params,
            IDeviceContext *device_context);
    };

} // namespace llaminar2