/**
 * @file IOverlayDomainRuntime.h
 * @brief Runtime service contract for MoE expert overlay domain work.
 */

#pragma once

#include "MoEExpertOverlayExecutionPlan.h"
#include "MoEExpertOverlayRuntimePlan.h"
#include "MoEOverlayDispatchCollective.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "execution/compute_stages/stages/MoEExpertOverlayCPUFallbackStage.h"
#include "execution/compute_stages/stages/MoEExpertOverlayLocalTPStage.h"
#include "execution/compute_stages/stages/MoEExpertParallelReduceStage.h"

#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{
    class IDeviceContext;

    enum class MoEOverlayDomainRuntimeDispatchKind
    {
        ContinuationLocal,
        GraphDispatchCollective,
        CompatibilitySingleDeviceStage,
        CompatibilityLocalTPStage,
        CompatibilityCPUFallbackStage,
        Unsupported,
    };

    const char *toString(MoEOverlayDomainRuntimeDispatchKind kind);

    enum class MoEOverlayDomainPartialLayout
    {
        DenseFullSequence,
        SparseTokenRows,
    };

    const char *toString(MoEOverlayDomainPartialLayout layout);

    struct MoEOverlayDomainWorkDescriptor
    {
        int layer_idx = -1;
        int tier_index = -1;
        std::string tier_name;
        std::string domain_name;
        std::string continuation_domain;
        MoEOverlayDomainRuntimeDispatchKind dispatch_kind =
            MoEOverlayDomainRuntimeDispatchKind::Unsupported;
        MoEOverlayDispatchGroup dispatch_group;
        MoEOverlayDispatchRequestKind request_kind =
            MoEOverlayDispatchRequestKind::RoutedWork;
        MoEOverlayDispatchMetrics dispatch_metrics;
        MoEOverlayDomainPartialLayout expected_partial_layout =
            MoEOverlayDomainPartialLayout::DenseFullSequence;
        MoEExpertTransferMode transfer_mode = MoEExpertTransferMode::Auto;
        std::vector<int> selected_rows;
        size_t routed_entries = 0;
        MoEExpertTransferVolume transfer_volume;
        MoEExpertParallelReducePartialInfo partial_info;
        bool remote_placeholder = false;
        bool compatibility_fallback_available = false;
    };

    struct MoEOverlayDomainWorkRequest
    {
        int layer_idx = -1;
        int tier_index = -1;
        std::string tier_name;
        std::string continuation_domain;

        ExpertComputeDomain source_domain;
        MoEOverlayRuntimeDomain runtime_domain;
        std::shared_ptr<const MoEExpertOverlayRuntimePlan> runtime_plan;
        std::shared_ptr<const MoEExpertOverlayExecutionPlan> execution_plan;
        MoEOverlayDispatchGroup dispatch_group;

        const MoEExpertDispatchOutput *dispatch_output = nullptr;
        std::shared_ptr<MoEExpertDispatchOutput> dispatch_output_lifetime;
        int dispatch_tier_index = -1;

        TensorBase *output = nullptr;
        DeviceId output_device = DeviceId::invalid();

        bool has_compute_params = false;
        MoEExpertComputeStage::Params compute_params;

        bool has_local_tp_params = false;
        MoEExpertOverlayLocalTPStage::Params local_tp_params;

        bool has_cpu_fallback_params = false;
        MoEExpertOverlayCPUFallbackStage::Params cpu_fallback_params;
    };

    struct MoEOverlayDomainWorkResult
    {
        bool ok = false;
        std::string error;
        MoEOverlayDomainWorkDescriptor descriptor;
        MoEOverlayDomainRuntimeDispatchKind dispatch_kind =
            MoEOverlayDomainRuntimeDispatchKind::Unsupported;
        MoEOverlayDispatchRequestKind request_kind =
            MoEOverlayDispatchRequestKind::RoutedWork;
        MoEOverlayDispatchMetrics dispatch_metrics;
        bool remote_placeholder = false;
        bool compatibility_fallback_used = false;
        TensorBase *partial_tensor = nullptr;
        MoEExpertParallelReducePartialInfo partial_info;
        MoEExpertOverlayLocalTPDiagnostics local_tp_diagnostics;
        MoECPUFallbackTransferStats cpu_transfer_stats;
        MoECPUFallbackTensorParallelStats cpu_tensor_parallel_stats;
    };

    class IOverlayDomainRuntime
    {
    public:
        virtual ~IOverlayDomainRuntime() = default;

        virtual MoEOverlayDomainWorkDescriptor describeWork(
            const MoEOverlayDomainWorkRequest &request) const = 0;

        virtual MoEOverlayDomainWorkResult submit(
            const MoEOverlayDomainWorkRequest &request,
            IDeviceContext *ctx) = 0;
    };

} // namespace llaminar2
