/**
 * @file MoEOverlayFinitePlacementGraphs.h
 * @brief Two cold captured lifecycle helpers owned by an original GPU executor.
 *
 * These graphs contain only the original epoch boundary stage. They retain no
 * host epoch, invoke no policy, and cannot materialize during request admission.
 * Actual Arm completion is the original capture-stream terminal event, rather
 * than the SDK enqueue future. End reuses the original readiness epilogue.
 */
#pragma once

#include "../compute_stages/stages/MoEOverlayEpochBoundaryStage.h"
#include "../local_execution/graph/ComputeGraph.h"
#include "../local_execution/graph/DeviceGraphExecutor.h"
#include "../../backends/IWorkerGPUContext.h"

#include <array>
#include <stdexcept>

namespace llaminar2
{
    class DeviceGraphOrchestrator;
    class MoEOverlayParticipantGraphRunner;

    /** Cold graph resources, accessible only through the original inference owners. */
    class MoEOverlayFinitePlacementGraphs final
    {
        friend class DeviceGraphOrchestrator;
        friend class MoEOverlayParticipantGraphRunner;
        using Operation=MoEOverlayEpochBoundaryStage::Operation;
        using Cache=DeviceGraphExecutor::GraphSegmentCache;

        /** Exact source executor owns capture and all embedded arena addresses. */
        struct Entry
        {
            std::unique_ptr<ComputeGraph> graph;
            Cache cache;
            MoEOverlayEpochBoundaryStage *stage=nullptr;
        };
        std::shared_ptr<DeviceMoEOverlayEpochArena> arena_;
        IWorkerGPUContext *worker_=nullptr;
        IDeviceContext *context_=nullptr;
        DeviceGraphExecutor *executor_=nullptr;
        std::uint64_t workspace_generation_=0u;
        // Declared last: native graphs/events retire before their arena/PMA.
        std::array<Entry,2> entries_;

        /** @return Cache ordinal of one closed finite lifecycle action. */
        static std::size_t index(DeviceMoEOverlayEpochParentAction action)
        {
            if(action==DeviceMoEOverlayEpochParentAction::Arm)return 0u;
            if(action==DeviceMoEOverlayEpochParentAction::Close)return 1u;
            throw std::invalid_argument("Finite placement graph received an invalid action");
        }

        /**
         * @brief Retire original native caches before context/arena withdrawal.
         * Existing cache fences preserve any published terminal obligation.
         * Request reset never calls this cold/terminal topology operation.
         */
        void invalidate()
        {
            for(auto &entry:entries_) {
                entry.cache.reset(Cache::StreamResetPolicy::Destroy);
                entry.graph.reset();entry.stage=nullptr;
            }
            arena_.reset();worker_=nullptr;context_=nullptr;executor_=nullptr;
            workspace_generation_=0u;
        }

        /**
         * @brief Capture both bounded helpers without submitting their kernels.
         * @param executor Original graph owner; no alternate execution engine.
         * @param context Original participant context.
         * @param worker Exact native worker on which this method is executing.
         * @param arena Original PMA-owned epoch records.
         * @param controller Exact participant readiness lane.
         * @param generation Original arena workspace generation.
         * @param pool_context Whether the original worker belongs to the process pool.
         * @throws std::logic_error On incomplete, foreign or changed identity.
         */
        void prepare(DeviceGraphExecutor &executor,IDeviceContext &context,
            IWorkerGPUContext &worker,std::shared_ptr<DeviceMoEOverlayEpochArena> arena,
            const MoEOverlayDeviceControllerParticipantBinding &controller,
            std::uint64_t generation,bool pool_context)
        {
            if(!worker.ownsCurrentThread() || !arena || !generation ||
               !controller.valid() || controller.device!=arena->deviceId() ||
               worker.deviceOrdinal()!=arena->deviceId().toKernelDeviceIndex())
                throw std::logic_error("Finite placement capture requires its original owner worker/arena/controller");
            if(arena_) {
                if(arena_!=arena || worker_!=&worker || context_!=&context ||
                   executor_!=&executor || workspace_generation_!=generation)
                    throw std::logic_error("Finite placement capture changed its original owner identity");
                for(const auto &entry:entries_)
                    if(!entry.graph || !entry.stage || !entry.cache.initialized ||
                       !entry.cache.deviceLoopGraphTemplate(*entry.graph))
                        throw std::logic_error("Finite placement capture cannot reuse partial cold materialization");
                auto expected=entries_[1].stage->getParams();
                expected.retirement_readiness_controller=controller;
                if(!entries_[1].stage->hasSameCaptureIdentity(expected))
                    throw std::logic_error("Finite placement capture changed its original controller/source lane");
                return;
            }
            // Retain original storage before partial construction, so failed
            // setup destroys captures before their arena/PMA allocation.
            arena_=std::move(arena);worker_=&worker;context_=&context;
            executor_=&executor;workspace_generation_=generation;
            for(std::size_t i=0u;i<entries_.size();++i) {
                auto &entry=entries_[i];
                MoEOverlayEpochBoundaryStage::Params params;
                params.device_id=arena_->deviceId();params.arena=arena_;
                params.operation=i==0u?Operation::ArmFiniteParent:Operation::CloseFiniteParent;
                if(i!=0u)params.retirement_readiness_controller=controller;
                params.stage_name=i==0u?"moe_finite_parent_arm":"moe_finite_parent_end";
                entry.graph=std::make_unique<ComputeGraph>();
                entry.graph->setExecutableMemoryClass(GPUGraphExecutableClass::BoundedFlatHelper);
                auto stage=std::make_unique<MoEOverlayEpochBoundaryStage>(params);
                entry.stage=stage.get();
                entry.graph->addNode(params.stage_name,std::move(stage),params.device_id);
                if(!entry.cache.ensureCaptureStream(&worker,params.device_id,pool_context) ||
                   !entry.cache.prepareCaptureStreamTerminal(&worker))
                    throw std::logic_error("Finite placement capture has no original stream/terminal event");
                entry.cache.perf_context=params.stage_name;
                entry.cache.replay_workload={.seq_len=1,.batch_size=1,.m=1,.decode=true};
                DeviceGraphExecutor::DecodeCapturePolicy policy;
                policy.allow_cached_graph_replay=true;policy.defer_final_sync=true;
                policy.graph_replay_plan_policy=DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph;
                bool replay=false;
                if(!executor.executeDecodeWithCapturePolicy(*entry.graph,&context,&entry.cache,
                        entry.cache.capture_stream,&worker,nullptr,policy,&replay,
                        DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch) ||
                   !entry.cache.deviceLoopGraphTemplate(*entry.graph))
                    throw std::logic_error("Finite placement helper did not cold-seal an original complete native graph");
            }
        }

        /**
         * @brief Submit one already captured local operation and observe its exact terminal.
         * @param action Arm or End only; contains no host-selected epoch.
         * @param join_original_terminal Original owner's event/receipt join on the helper stream.
         * @throws std::logic_error On changed/partial ownership or failed native submission.
         * @note Called off-worker at one finite admission/closure, never per child.
         */
        void run(DeviceMoEOverlayEpochParentAction action,
            const std::function<bool(void*)> &join_original_terminal)
        {
            auto &entry=entries_.at(index(action));
            if(!arena_ || !worker_ || !context_ || !executor_ || !entry.graph ||
               !entry.cache.initialized || !entry.cache.deviceLoopGraphTemplate(*entry.graph) ||
               worker_->ownsCurrentThread() || !join_original_terminal)
                throw std::logic_error("Finite placement operation requires its cold original graph and off-worker owner");
            Cache::CaptureStreamTerminalTicket terminal;
            worker_->submitAndWait([&] {
                if(!worker_->ownsCurrentThread())
                    throw std::logic_error("Finite placement dispatch did not enter its original GPU worker");
                // The owner consumes its existing producer/terminal authority,
                // never a new event recorded after unrelated stream work.
                if(!join_original_terminal(entry.cache.capture_stream))
                    throw std::logic_error("Finite placement has no completed original inference boundary");
                DeviceGraphExecutor::DecodeCapturePolicy policy;
                policy.allow_cached_graph_replay=true;policy.defer_final_sync=true;
                policy.graph_replay_plan_policy=DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph;
                bool replay=false;
                entry.graph->reset();
                if(!executor_->executeDecodeWithCapturePolicy(*entry.graph,context_,&entry.cache,
                        entry.cache.capture_stream,worker_,nullptr,policy,&replay) || !replay)
                    throw std::logic_error("Finite placement operation lost its original captured executable");
                terminal=entry.cache.publishCaptureStreamTerminal();
            });
            // A completed SDK callback is not Arm completion. Observe only
            // this exact recorded event before permitting any new child.
            entry.cache.waitForPublishedCaptureStreamTerminal(
                terminal,Cache::HostFenceWaitPolicy::ActiveProgress);
        }
    };
}
