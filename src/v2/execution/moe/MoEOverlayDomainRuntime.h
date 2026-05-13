/**
 * @file MoEOverlayDomainRuntime.h
 * @brief Default runtime service for MoE expert overlay graph stages.
 */

#pragma once

#include "IOverlayDomainRuntime.h"

#include <condition_variable>
#include <map>
#include <mutex>

namespace llaminar2
{
    class MoEOverlayDomainRuntime final : public IOverlayDomainRuntime
    {
    public:
        struct Config
        {
            std::shared_ptr<const MoEExpertOverlayRuntimePlan> runtime_plan;
            std::shared_ptr<const MoEExpertOverlayExecutionPlan> execution_plan;
            std::shared_ptr<IMoEOverlayDispatchBackend> dispatch_backend;
            bool enable_compatibility_fallback = true;
        };

        explicit MoEOverlayDomainRuntime(Config config);

        MoEOverlayDomainWorkDescriptor describeWork(
            const MoEOverlayDomainWorkRequest &request) const override;

        MoEOverlayDomainWorkResult submit(
            const MoEOverlayDomainWorkRequest &request,
            IDeviceContext *ctx) override;

    private:
        struct CPUFallbackCompletion
        {
            bool complete = false;
            MoEOverlayDomainWorkResult result;
            std::vector<float> output_data;
            int remaining_readers = 0;
        };

        const MoEExpertOverlayExecutionPlan *executionPlanFor(
            const MoEOverlayDomainWorkRequest &request) const;
        bool hasRemoteParticipant(const MoEOverlayDomainWorkRequest &request) const;
        bool isContinuationLocal(const MoEOverlayDomainWorkRequest &request) const;
        MoEOverlayDispatchRequest buildDispatchRequest(
            const MoEOverlayDomainWorkRequest &request,
            const MoEOverlayDomainWorkDescriptor &descriptor) const;
        MoEOverlayDomainWorkResult runDispatchBackend(
            const MoEOverlayDomainWorkRequest &request,
            const MoEOverlayDomainWorkDescriptor &descriptor,
            IDeviceContext *ctx);
        MoEOverlayDomainWorkResult runCPUFallbackGraphDispatch(
            const MoEOverlayDomainWorkRequest &request,
            const MoEOverlayDomainWorkDescriptor &descriptor,
            IDeviceContext *ctx);
        MoEOverlayDomainWorkResult runCompatibilityPath(
            const MoEOverlayDomainWorkRequest &request,
            const MoEOverlayDomainWorkDescriptor &descriptor,
            IDeviceContext *ctx,
            bool compatibility_bridge_used);
        std::shared_ptr<MoECPUFallbackDomainContext> cpuFallbackDomainContextFor(
            const MoEExpertOverlayCPUFallbackStage::Params &params);
        void publishCPUFallbackCompletion(
            const MoEOverlayDispatchGroup &group,
            const MoEOverlayDomainWorkResult &result);
        MoEOverlayDomainWorkResult waitForCPUFallbackCompletion(
            const MoEOverlayDomainWorkRequest &request,
            const MoEOverlayDomainWorkDescriptor &descriptor);

        Config config_;
        std::mutex cpu_domain_context_mutex_;
        std::map<std::string, std::shared_ptr<MoECPUFallbackDomainContext>> cpu_domain_contexts_;
        static std::mutex cpu_fallback_completion_mutex_;
        static std::condition_variable cpu_fallback_completion_cv_;
        static std::map<std::string, std::shared_ptr<CPUFallbackCompletion>> cpu_fallback_completions_;
    };

    std::shared_ptr<IOverlayDomainRuntime> makeMoEOverlayDomainRuntime(
        MoEOverlayDomainRuntime::Config config);

} // namespace llaminar2
