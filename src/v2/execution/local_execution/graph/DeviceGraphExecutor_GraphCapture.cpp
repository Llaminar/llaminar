/**
 * @file DeviceGraphExecutor_GraphCapture.cpp
 * @brief GPU graph capture/replay implementation for DeviceGraphExecutor
 *
 * Split from DeviceGraphExecutor.cpp to isolate the GPU graph capture subsystem.
 * Contains:
 * - GraphSegmentCache resource management (streams, events)
 * - executeWithGraphCapture (single-graph capture/replay)
 * - executeDecodeWithCapturePolicy (policy-based mode selection)
 * - executeWithCachedGraphReplay (cached graph replay)
 *
 * Retained-parent evidence comes from its sealed physical unit inventory.
 * Successful submission records follow both native launch and CPU ticket-service
 * retirement; they do not imply a host wait for GPU completion. This lets an
 * HTTP observer prove a rank-local heterogeneous boundary without pretending
 * that the graph contains a cross-rank coordinator or TP collective. Export
 * validates main-unit coverage separately from full declarative stage identity;
 * auxiliary work must retain its original attached-body and parent authority.
 */

#include "DeviceGraphExecutor.h"
#include "backends/NativeOrderingEventGroup.h"
#include "../../../backends/ReadAheadNativeResourceFamily.h"
#include "../../../backends/NativeContextPreparation.h"
#include "DeviceGraphCaptureController.h"
#include "GraphCaptureGuard.h"
#include "GraphCaptureStageActivity.h"
#include "RetainedParentTicketServiceWorker.h"
#include "RetainedHostServiceProgram.h"
#include "RetainedParentPerfEvidence.h"
#include "RetainedParentTemplateStageCoverage.h"
#include "../coherence/CoherencePolicy.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../utils/Logger.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../../../memory/BufferArena.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../backends/IGPUGraphCapture.h"
#include "../../../backends/IWorkerGPUContext.h"
#include "../../../collective/CollectiveTimeoutPolicy.h"
#include "../../../transfer/TransferEngine.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <sstream>
#include <thread>
#include <unordered_set>

#include "PreparedFollowerCaptureTerminals.h"
#include "memory/CPUStorageRoot.h"
namespace llaminar2
{
    namespace {
    /** @return The existing ordinary ordering-event creator on its exact owner. */
    void* createCaptureTerminalEvent(IWorkerGPUContext& worker) {
        return worker.createEvent(GPUEventPurpose::Ordering);
    }
    /** @brief Retire exactly one accepted ordinary event on its original worker. */
    void destroyCaptureTerminalEvent(IWorkerGPUContext& worker,void*& event){
        if(event){worker.destroyEvent(event);event=nullptr;}
    }
    }
    struct PreparedFollowerCaptureTerminalPopulation::State final {
        PreparedFollowerCaptureTerminalKey key;
        PreparedGPUContextUsePermit original;
        void* stream;
        MoEOverlayInferenceOriginalSlotGeometry geometry;
        std::unique_ptr<NativeOrderingEventGroup> singleton,extra;
        std::shared_ptr<PreparedFollowerCaptureTerminalPopulation> next;
        bool singleton_attempted=false,extra_attempted=false;
        /** @brief Retain immutable source identity before any native creator. */
        State(PreparedFollowerCaptureTerminalKey k,PreparedGPUContextUsePermit p,void* s,
            MoEOverlayInferenceOriginalSlotGeometry g):key(k),original(std::move(p)),stream(s),geometry(g){}
        /** @return Fixed trailing actual handle coordinates, not a slot allocator. */
        void** events(){return reinterpret_cast<void**>(this+1);}
    };
    struct PreparedFollowerCaptureTerminalOwner::State final {
        std::mutex mutex;
        std::shared_ptr<PhysicalMemoryAuthority> memory;
        MoEOverlayInferenceOriginalSlotGeometry geometry;
        std::shared_ptr<PreparedFollowerCaptureTerminalPopulation> first;
        bool sealed=false;
        std::exception_ptr failure;
        /** @brief Retain the same model admission and original slot geometry. */
        State(std::shared_ptr<PhysicalMemoryAuthority> m,MoEOverlayInferenceOriginalSlotGeometry g)
            :memory(std::move(m)),geometry(g){}
    };
    std::size_t PreparedFollowerCaptureTerminalPopulation::physicalBytes(const MoEOverlayInferenceOriginalSlotGeometry& g){
        if(g.slotCount()>(std::numeric_limits<std::size_t>::max()-sizeof(PreparedFollowerCaptureTerminalPopulation)-sizeof(State))/sizeof(void*))
            throw std::overflow_error("Prepared follower terminal metadata overflow");
        return CPUStorageRoot<std::byte>(sizeof(PreparedFollowerCaptureTerminalPopulation)+sizeof(State)+g.slotCount()*sizeof(void*),
            PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent,DeviceId::cpu()).physicalBytes();
    }
    std::size_t PreparedFollowerCaptureTerminalOwner::physicalBytes(){
        return CPUStorageRoot<std::byte>(sizeof(PreparedFollowerCaptureTerminalOwner)+sizeof(State),
            PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent,DeviceId::cpu()).physicalBytes();
    }
    std::shared_ptr<PreparedFollowerCaptureTerminalOwner> PreparedFollowerCaptureTerminalOwner::create(
        std::shared_ptr<PhysicalMemoryAuthority> memory,const MoEOverlayInferenceOriginalSlotGeometry& geometry){
        if(!memory)throw std::invalid_argument("Prepared follower terminals require original model admission");
        auto storage=CPUStorageRoot<std::byte>(physicalBytes(),PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent,
            DeviceId::cpu(),CPUStoragePlacement::local(),CPUStorageInitialization::Uninitialized).allocate(memory);
        auto* state=new(storage.data()+sizeof(PreparedFollowerCaptureTerminalOwner))State(memory,geometry);
        auto* owner=new(storage.data())PreparedFollowerCaptureTerminalOwner(state);
        return {owner,[storage=std::move(storage)](auto* value)mutable{value->~PreparedFollowerCaptureTerminalOwner();storage={};}};
    }
    PreparedFollowerCaptureTerminalOwner::~PreparedFollowerCaptureTerminalOwner(){std::destroy_at(state_);}
    void PreparedFollowerCaptureTerminalPopulation::requireOriginal(const PreparedGPUContextUsePermit& original,
        void* stream,const MoEOverlayInferenceOriginalSlotGeometry& geometry)const {
        const auto& held=state_->original;
        if(!stream || stream!=state_->stream || geometry!=state_->geometry || original.device()!=held.device() ||
            original.generation()!=held.generation() || original.memory().get()!=held.memory().get() ||
            original.memory().owner_before(held.memory()) || held.memory().owner_before(original.memory()) ||
            original.worker()!=held.worker() || original.worker().owner_before(held.worker()) || held.worker().owner_before(original.worker()))
            throw std::invalid_argument("Prepared follower terminal changed original source/geometry/worker/stream");
        held.provenance().requireWorker(original.worker());
    }
    void* PreparedFollowerCaptureTerminalPopulation::event(std::size_t index)const {
        if(index>=state_->geometry.slotCount())throw std::out_of_range("Prepared follower terminal slot is outside original geometry");
        return state_->events()[index];
    }
    PreparedFollowerCaptureTerminalPopulation::~PreparedFollowerCaptureTerminalPopulation()noexcept {
        try {
            auto destroy=[&]{for(std::size_t i=0;i<state_->geometry.slotCount();++i)
                destroyCaptureTerminalEvent(*state_->original.worker(),state_->events()[i]);};
            if(state_->original.worker()->ownsCurrentThread())destroy();else state_->original.worker()->submitAndWait(destroy);
            if(state_->singleton_attempted)state_->singleton->completeNativeHandles();
            if(state_->extra_attempted)state_->extra->completeNativeHandles();
            std::destroy_at(state_);
        }catch(...){std::terminate();}
    }
    std::shared_ptr<PreparedFollowerCaptureTerminalPopulation> PreparedFollowerCaptureTerminalOwner::acquire(
        PreparedFollowerCaptureTerminalKey key,const PreparedGPUContextUsePermit& original,void* stream){
        if(!stream || key.participant_id<0 || key.physical_rows<=0 || key.device!=original.device() || !key.device.is_gpu() ||
            original.memory().get()!=state_->memory.get() || original.memory().owner_before(state_->memory) || state_->memory.owner_before(original.memory()))
            throw std::invalid_argument("Prepared follower terminal changed original cache/model coordinate");
        std::lock_guard lock(state_->mutex);
        if(state_->failure)std::rethrow_exception(state_->failure);
        for(auto* link=&state_->first;*link;link=&(*link)->state_->next){
            if((*link)->state_->key!=key)continue;
            if(link->use_count()!=1)throw std::logic_error("Prepared follower terminal population still has a live borrower");
            (*link)->requireOriginal(original,stream,state_->geometry);
            return *link;
        }
        if(state_->sealed)throw std::invalid_argument("Prepared follower terminal changed the sealed original cache roster");
        try {
            auto storage=CPUStorageRoot<std::byte>(PreparedFollowerCaptureTerminalPopulation::physicalBytes(state_->geometry),
                PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent,DeviceId::cpu(),CPUStoragePlacement::local(),
                CPUStorageInitialization::Uninitialized).allocate(state_->memory);
            auto* metadata=new(storage.data()+sizeof(PreparedFollowerCaptureTerminalPopulation))PreparedFollowerCaptureTerminalPopulation::State(key,original,stream,state_->geometry);
            std::fill_n(metadata->events(),state_->geometry.slotCount(),nullptr);
            auto* value=new(storage.data())PreparedFollowerCaptureTerminalPopulation(metadata);
            std::shared_ptr<PreparedFollowerCaptureTerminalPopulation> population(value,[storage=std::move(storage)](auto* p)mutable{
                p->~PreparedFollowerCaptureTerminalPopulation();storage={};});
            metadata->singleton=NativeOrderingEventGroup::claim(state_->memory,original.provenance(),NativeOrderingEventProgram::captureSubmissionRetirement());
            metadata->extra=NativeOrderingEventGroup::claim(state_->memory,original.provenance(),NativeOrderingEventProgram::retainedCaptureTerminals(state_->geometry));
            metadata->singleton->beginNativeAttempt();metadata->singleton_attempted=true;
            metadata->extra->beginNativeAttempt();metadata->extra_attempted=true;
            if(!original.worker()->ownsCurrentThread())throw std::logic_error("Prepared follower terminal creators require exact worker");
            for(std::size_t slot=0;slot<state_->geometry.slotCount();++slot){
                metadata->events()[slot]=createCaptureTerminalEvent(*original.worker());
                if(!metadata->events()[slot])throw std::runtime_error("Original capture terminal event creation failed");
            }
            metadata->next=std::move(state_->first);state_->first=population;
            return population;
        }catch(...){state_->failure=std::current_exception();throw;}
    }
    void PreparedFollowerCaptureTerminalOwner::seal(){
        std::lock_guard lock(state_->mutex);
        if(state_->failure)std::rethrow_exception(state_->failure);
        if(!state_->first)throw std::logic_error("Prepared follower terminal roster is empty");
        for(auto* link=&state_->first;*link;link=&(*link)->state_->next)
            if(link->use_count()!=2)throw std::logic_error("Prepared follower terminal roster omitted or duplicated an original borrower");
        state_->sealed=true;
    }

    struct OriginalSlotCaptureTerminals::State final {
        OriginalSlotCaptureTerminalState receipts;
        std::shared_ptr<IWorkerGPUContext> worker;
        GPUContextProvenance original;
        std::shared_ptr<PhysicalMemoryAuthority> memory;
        DeviceId device;
        void* stream;
        std::vector<void*> events;
        std::unique_ptr<NativeOrderingEventGroup> extra_obligations;
        std::shared_ptr<PreparedFollowerCaptureTerminalPopulation> prepared_population;
        /** @brief Retain the original roster/source before any additional native creator. */
        State(std::shared_ptr<const MoEOverlayInferenceOriginalSlotDomain> domain,
            std::shared_ptr<IWorkerGPUContext> w,GPUContextProvenance source,
            std::shared_ptr<PhysicalMemoryAuthority> m,DeviceId d,void* st,void* singleton)
            :receipts(std::move(domain)),worker(std::move(w)),original(std::move(source)),
             memory(std::move(m)),device(d),stream(st),events(receipts.slotCount(),nullptr) {
            events[0]=singleton;
        }
    };
    PhysicalMemoryPlan OriginalSlotCaptureTerminals::extraEventBOM(
        const MoEOverlayInferenceOriginalSlotGeometry& geometry,PhysicalMemoryResource host,PhysicalMemoryResource gpu)
    {
        return NativeOrderingEventGroup::bom(
            NativeOrderingEventProgram::retainedCaptureTerminals(geometry),host,gpu);
    }
    OriginalSlotCaptureTerminals::OriginalSlotCaptureTerminals(std::unique_ptr<State> state):state_(std::move(state)){}
    std::unique_ptr<OriginalSlotCaptureTerminals> OriginalSlotCaptureTerminals::create(
        std::shared_ptr<const MoEOverlayInferenceOriginalSlotDomain> domain,
        std::shared_ptr<IWorkerGPUContext> worker,std::shared_ptr<PhysicalMemoryAuthority> memory,
        DeviceId device,void* stream,void* singleton)
    {
        if(!domain || !memory || !worker || !device.is_gpu() || !stream || !singleton || isGraphCaptureActive())
            throw std::invalid_argument("Original capture terminals require cold original source/stream/slot owners");
        const auto prepared=GPUDeviceContextPool::instance().retainExistingPreparedContext(device,memory);
        prepared.provenance().requireWorker(worker);
        if(!worker->ownsCurrentThread())throw std::logic_error("Original capture terminals require their exact source worker");
        // Claim the SAME opaque-event population before constructing its
        // proportional receipt/event metadata, not only before SDK creators.
        auto obligations=NativeOrderingEventGroup::claim(memory,prepared.provenance(),
            NativeOrderingEventProgram::retainedCaptureTerminals(*domain));
        auto state=std::make_unique<State>(std::move(domain),worker,prepared.provenance(),std::move(memory),device,stream,singleton);
        state->extra_obligations=std::move(obligations);
        state->extra_obligations->beginNativeAttempt();
        try {
            for(std::size_t slot=1;slot<state->events.size();++slot) {
                state->events[slot]=createCaptureTerminalEvent(*worker);
                if(!state->events[slot])throw std::runtime_error("Original capture terminal event creation failed");
            }
        } catch(...) {
            // No graph can use these cold handles yet. Retire every accepted
            // creator before completing original physical-lifetime ownership.
            try {for(std::size_t slot=1;slot<state->events.size();++slot)
                if(state->events[slot])worker->destroyEvent(state->events[slot]);}
            catch(...) {std::terminate();}
            state->extra_obligations->completeNativeHandles();
            throw;
        }
        return std::unique_ptr<OriginalSlotCaptureTerminals>(new OriginalSlotCaptureTerminals(std::move(state)));
    }
    std::unique_ptr<OriginalSlotCaptureTerminals> OriginalSlotCaptureTerminals::createPrepared(
        std::shared_ptr<const MoEOverlayInferenceOriginalSlotDomain> domain,
        std::shared_ptr<PreparedFollowerCaptureTerminalPopulation> population,
        std::shared_ptr<IWorkerGPUContext> worker,std::shared_ptr<PhysicalMemoryAuthority> memory,DeviceId device,void* stream){
        if(!domain || !population)throw std::invalid_argument("Prepared terminal borrow lacks actual original slot domain");
        const auto original=GPUDeviceContextPool::instance().retainExistingPreparedContext(device,memory);
        population->requireOriginal(original,stream,domain->geometry());
        auto state=std::make_unique<State>(domain,worker,original.provenance(),memory,device,stream,population->event(0));
        for(std::size_t slot=1;slot<state->events.size();++slot)state->events[slot]=population->event(slot);
        state->prepared_population=std::move(population);
        return std::unique_ptr<OriginalSlotCaptureTerminals>(new OriginalSlotCaptureTerminals(std::move(state)));
    }
    OriginalSlotCaptureTerminals::~OriginalSlotCaptureTerminals()noexcept {
        try {
            if(!state_)return;
            if(state_->receipts.nextOutstanding())throw std::logic_error("Original capture terminal has unobserved native receipts");
            if(state_->prepared_population)return;
            auto destroy=[&] {
                if(isGraphCaptureActive())throw std::logic_error("Original capture terminal destruction entered graph capture");
                for(std::size_t slot=1;slot<state_->events.size();++slot)
                    destroyCaptureTerminalEvent(*state_->worker,state_->events[slot]);
            };
            if(state_->worker->ownsCurrentThread())destroy();
            else state_->worker->submitAndWait(destroy);
            state_->extra_obligations->completeNativeHandles();
        }
        catch(...) {LOG_ERROR("Original capture terminal destruction lacks exact native closure");std::terminate();}
    }
    void OriginalSlotCaptureTerminals::requireOwner(IWorkerGPUContext* worker,void* stream)const {
        if(!state_ || worker!=state_->worker.get() || stream!=state_->stream || !stream)
            throw std::logic_error("Original capture terminal changed exact worker/stream");
        state_->original.requireWorker(state_->worker);
    }
    void* OriginalSlotCaptureTerminals::eventForPublication(const MoEOverlayInferenceOriginalSlot& slot,
        IWorkerGPUContext* worker,void* stream)const {
        requireOwner(worker,stream);
        if(!worker->ownsCurrentThread())throw std::logic_error("Original terminal publication requires its exact worker thread");
        state_->receipts.requirePublication(slot);return state_->events.at(slot.index());
    }
    OriginalSlotCaptureTerminals::Receipt OriginalSlotCaptureTerminals::published(const MoEOverlayInferenceOriginalSlot& slot) {
        return state_->receipts.publish(slot);
    }
    void* OriginalSlotCaptureTerminals::eventForObservation(const Receipt& receipt,IWorkerGPUContext* worker,void* stream)const {
        requireOwner(worker,stream);state_->receipts.requireObservation(receipt);return state_->events.at(receipt.originalSlot().index());
    }
    void* OriginalSlotCaptureTerminals::eventForDeviceDependency(const Receipt& receipt,IWorkerGPUContext* worker,void* stream)const {
        requireOwner(worker,stream);state_->receipts.requireDeviceDependency(receipt);
        return state_->events.at(receipt.originalSlot().index());
    }
    void OriginalSlotCaptureTerminals::observed(const Receipt& receipt){state_->receipts.observed(receipt);}
    std::optional<OriginalSlotCaptureTerminals::Receipt> OriginalSlotCaptureTerminals::nextOutstanding()const{return state_->receipts.nextOutstanding();}
    const std::shared_ptr<const MoEOverlayInferenceOriginalSlotDomain>&OriginalSlotCaptureTerminals::domain()const noexcept{return state_->receipts.domain();}

    namespace
    {
        /** @return Stable diagnostic label for one ordered-timeline step kind. */
        const char *orderedTimelineStepKindName(
            GPUOrderedTimelineStepKind kind) noexcept
        {
            switch (kind)
            {
            case GPUOrderedTimelineStepKind::CapturedFragment:
                return "captured_fragment";
            case GPUOrderedTimelineStepKind::WaitValue64:
                return "wait_value_64";
            case GPUOrderedTimelineStepKind::PublishValue64:
                return "publish_value_64";
            }
            return "unknown";
        }

        /**
         * @brief Non-blockingly publish the previous retained-parent GPU timeline.
         *
         * Event nodes are present only under the explicit heavy GPU-stage timing
         * gate. Collection happens before the next replay, when the request has
         * ordinarily consumed the prior terminal result. A still-pending sample
         * is counted and left alone; profiling never inserts a host wait into
         * inference. Backend query failures remain fatal because they may expose
         * an asynchronous execution error rather than missing diagnostics.
         */
        bool collectOrderedTimelineTiming(
            IGPUGraphCapture &parent,
            const std::string &context,
            const std::string &device)
        {
            GPUOrderedTimelineTimingSnapshot snapshot =
                parent.consumeOrderedTimelineTiming();
            switch (snapshot.state)
            {
            case GPUOrderedTimelineTimingState::Disabled:
            case GPUOrderedTimelineTimingState::AwaitingLaunch:
                return true;
            case GPUOrderedTimelineTimingState::Pending:
                PerfStatsCollector::addCounter(
                    "stage_gpu",
                    "ordered_timeline_pending_samples",
                    1.0,
                    "decode",
                    device,
                    {{"context", context},
                     {"sampling_policy", "nonblocking_previous_replay"}});
                return true;
            case GPUOrderedTimelineTimingState::Failed:
                LOG_ERROR(
                    "[DeviceGraphExecutor] Ordered-timeline GPU timing failed: "
                    << snapshot.error);
                return false;
            case GPUOrderedTimelineTimingState::Complete:
                break;
            }

            std::uint64_t total_ns = 0u;
            for (const auto &sample : snapshot.samples)
            {
                const std::uint64_t elapsed_ns =
                    static_cast<std::uint64_t>(std::max(
                        1.0,
                        sample.elapsed_ms * 1.0e6));
                total_ns += elapsed_ns;
                PerfStatsCollector::recordTimingNs(
                    "stage_gpu",
                    "graph_replay.ordered_timeline_step",
                    elapsed_ns,
                    "decode",
                    device,
                    {{"attribution", "gpu_event"},
                     {"context", context},
                     {"graph_capture_scope", "retained_ordered_timeline"},
                     {"step", sample.name},
                     {"step_kind",
                      orderedTimelineStepKindName(sample.kind)},
                     {"sync_scope", "nonblocking_previous_replay"}});
            }
            PerfStatsCollector::recordTimingNs(
                "stage_gpu",
                "graph_replay.ordered_timeline_total",
                std::max<std::uint64_t>(1u, total_ns),
                "decode",
                device,
                {{"attribution", "gpu_event"},
                 {"context", context},
                 {"graph_capture_scope", "retained_ordered_timeline"},
                 {"step_count", std::to_string(snapshot.samples.size())},
                 {"sync_scope", "nonblocking_previous_replay"}});
            return true;
        }
    } // namespace

    bool DeviceGraphExecutor::prepareRetainedParentTicketServiceWorker()
    {
        if (!retained_parent_ticket_service_worker_)
        {
            LOG_ERROR("[DeviceGraphExecutor] Missing original pre-quota declared/admitted executor-ticket worker");
            return false;
        }
        if (!retained_parent_ticket_service_worker_->worker().idle())
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Retained-parent ticket-service worker is not idle before transaction admission");
            return false;
        }
        return true;
    }

    bool DeviceGraphExecutor::submitRetainedParentWithConcurrentTicketService(
        IGPUGraphCapture &parent,
        ComputeGraph &graph,
        GraphSegmentCache &segment_cache,
        IDeviceContext *ctx,
        IWorkerGPUContext *gpu_ctx,
        uint64_t current_step,
        std::span<const size_t> segment_indices)
    {
        if (!ctx || !gpu_ctx || !segment_cache.capture_stream ||
            segment_indices.empty() ||
            parent.executionStream() != segment_cache.capture_stream ||
            !parent.hasExecutable() ||
            !prepareRetainedParentTicketServiceWorker())
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Retained-parent concurrent submission has incomplete graph, stream, service, or worker ownership");
            return false;
        }

        using Clock = std::chrono::steady_clock;
        const bool collect_transaction_timing =
            PerfStatsCollector::isDomainEnabled(
                "retained_parent_transaction");
        const auto transaction_begin = collect_transaction_timing
                                           ? Clock::now()
                                           : Clock::time_point{};

        struct Invocation
        {
            DeviceGraphExecutor *executor = nullptr;
            ComputeGraph *graph = nullptr;
            GraphSegmentCache *cache = nullptr;
            IDeviceContext *context = nullptr;
            IWorkerGPUContext *gpu_context = nullptr;
            uint64_t step = 0u;
            std::span<const size_t> segments;
            bool collect_timing = false;
            std::uint64_t service_elapsed_ns = 0u;
        } invocation{
            .executor = this,
            .graph = &graph,
            .cache = &segment_cache,
            .context = ctx,
            .gpu_context = gpu_ctx,
            .step = current_step,
            .segments = segment_indices,
            .collect_timing = collect_transaction_timing,
        };

        RetainedParentTicketServiceWorker::Ticket service_ticket;
        try
        {
            service_ticket = retained_parent_ticket_service_worker_->worker().dispatch(
                [](void *opaque) -> bool
                {
                    auto &job = *static_cast<Invocation *>(opaque);
                    const auto service_begin = job.collect_timing
                                                   ? Clock::now()
                                                   : Clock::time_point{};
                    const bool service_ok = DeviceGraphCaptureController::
                        executeConcurrentTicketService(
                            *job.graph,
                            *job.cache,
                            job.context,
                            job.gpu_context,
                            job.step,
                            job.segments,
                            [&](ComputeNode &node)
                            {
                                return job.executor->executeNode(
                                    node, job.context);
                            });
                    if (job.collect_timing)
                    {
                        job.service_elapsed_ns = static_cast<std::uint64_t>(
                            std::max<std::int64_t>(
                                1,
                                std::chrono::duration_cast<
                                    std::chrono::nanoseconds>(
                                    Clock::now() - service_begin)
                                    .count()));
                    }
                    return service_ok;
                },
                &invocation);
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Could not arm retained-parent ticket service: "
                << error.what());
            return false;
        }
        const auto service_dispatch_end = collect_transaction_timing
                                              ? Clock::now()
                                              : Clock::time_point{};

        /*
         * The service is live before this call. HIP may apply submission
         * backpressure after it has issued the first mapped-wait kernel; the
         * persistent worker can now consume and publish every CPU ticket while
         * this thread remains inside the backend submission routine.
         */
        const bool launch_ok = parent.launch();
        const auto parent_launch_end = collect_transaction_timing
                                           ? Clock::now()
                                           : Clock::time_point{};
        bool service_ok = false;
        try
        {
            service_ok =
                retained_parent_ticket_service_worker_->worker().await(service_ticket);
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Could not collect retained-parent ticket service: "
                << error.what());
        }
        const auto transaction_end = collect_transaction_timing
                                         ? Clock::now()
                                         : Clock::time_point{};

        if (collect_transaction_timing)
        {
            const auto elapsed_ns = [](Clock::time_point begin,
                                       Clock::time_point end)
            {
                return static_cast<std::uint64_t>(
                    std::max<std::int64_t>(
                        1,
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            end - begin)
                            .count()));
            };
            const PerfStatsCollector::Tags tags{
                {"context", segment_cache.perf_context},
                {"graph_nodes", std::to_string(parent.nodeCount())},
                {"service_segments", std::to_string(segment_indices.size())},
                {"timing_semantics", "overlapping_host_intervals"},
            };
            const auto record = [&](const char *name, std::uint64_t ns)
            {
                PerfStatsCollector::recordTimingNs(
                    "retained_parent_transaction",
                    name,
                    ns,
                    "decode",
                    ctx->deviceId().toString(),
                    tags);
            };
            record(
                "transaction_host_envelope",
                elapsed_ns(transaction_begin, transaction_end));
            record(
                "ticket_service_dispatch",
                elapsed_ns(transaction_begin, service_dispatch_end));
            record(
                "parent_launch_submission",
                elapsed_ns(service_dispatch_end, parent_launch_end));
            record(
                "post_launch_service_join",
                elapsed_ns(parent_launch_end, transaction_end));
            record(
                "concurrent_ticket_service",
                std::max<std::uint64_t>(1u, invocation.service_elapsed_ns));
        }

        if (!launch_ok || !service_ok)
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Retained-parent concurrent transaction failed"
                << " launch_ok=" << launch_ok
                << " service_ok=" << service_ok);
            /*
             * The CPU ticket worker has retired, so this thread may inspect the
             * exact producer stream without racing borrowed invocation state.
             * A poisoned CUDA/HIP stream throws here with backend and device
             * context. Pending or complete execution is still a failed atomic
             * transaction: do not hide it behind a host-blocking stream drain
             * and a recoverable false return. The owner must tear down the
             * failed request/runtime generation before any further submission.
             */
            const GPUStreamExecutionState stream_state =
                gpu_ctx->queryStreamExecutionState(
                    segment_cache.capture_stream,
                    "failed retained-parent ticket transaction");
            std::ostringstream error;
            error << "Retained-parent ticket transaction failed"
                  << " launch_ok=" << launch_ok
                  << " service_ok=" << service_ok
                  << " stream_state="
                  << (stream_state == GPUStreamExecutionState::Complete
                          ? "complete"
                          : "pending");
            throw std::runtime_error(error.str());
        }
        return true;
    }

    std::optional<
        DeviceGraphExecutor::GraphSegmentCache::DeviceLoopGraphTemplateView>
    DeviceGraphExecutor::GraphSegmentCache::deviceLoopGraphTemplate(
        const ComputeGraph &graph,
        std::string *error) const
    {
        auto reject = [&](const std::string &reason)
            -> std::optional<DeviceLoopGraphTemplateView>
        {
            if (error)
                *error = reason;
            return std::nullopt;
        };
        if (error)
            error->clear();
        const auto &execution_order = graph.getExecutionOrder();
        if (execution_order.empty())
            return reject("source graph has no stages");
        if (!initialized || needs_capture)
            return reject("graph cache is not replay-ready");
        if (DeviceGraphExecutor::isRetainedParentPlanPolicy(graph_replay_plan_policy))
        {
            // A composed parent is the only executable. Exporting its graph-only
            // source child would silently omit the fixed native exchange that
            // the topology compiler inserted around local computation.
            const auto &plan = retained_composed_parent_replay;
            if (DeviceGraphExecutor::hasConcurrentTicketService(graph_replay_plan_policy) ||
                !plan.concurrent_ticket_service_segment_indices.empty())
                return reject("device-loop composition cannot hide host-serviced boundaries");
            if (!plan.valid() || plan.graph != &graph ||
                plan.topology_generation != graph.topologyGeneration() ||
                plan.snapshot_configuration_epoch != snapshot_configuration_epoch ||
                plan.capture_variant_signature != capture_variant_signature ||
                plan.child_unit_count != segments.size() ||
                plan.stages.size() != execution_order.size() ||
                !retained_parent_capture || !retained_parent_capture->hasExecutable() ||
                retained_parent_capture->nodeCount() == 0 || !capture_stream ||
                retained_parent_capture->executionStream() != capture_stream)
                return reject("composed parent has incomplete or stale native capture identity");
            for (const auto& segment : segments)
                if (!segment.capturable || !segment.capture || segment.capture->hasExecutable())
                    return reject("composed parent contains a manual or independently executable child");
            if (!validateRetainedParentTemplateStageCoverage(graph, segments, plan, error))
                return std::nullopt;
            if (graph.retainedParticipantAuxiliaryRange())
            {
                // Main coverage alone cannot prove the separately recorded
                // body belongs to this executable. These original borrows are
                // created only by the canonical executor/source-plan attachment.
                if (!participant_auxiliary_preparation || !original_storage_attachment)
                    return reject("composed parent has no original attached participant auxiliary body");
                const auto& definition = participant_auxiliary_preparation->definition();
                const auto& range = *graph.retainedParticipantAuxiliaryRange();
                if (definition.graph != &graph ||
                    definition.topology_generation != graph.topologyGeneration() ||
                    definition.variant != capture_variant_signature ||
                    definition.range_identity != range.identity ||
                    definition.participant != range.participant ||
                    participant_auxiliary_preparation->stream() != capture_stream)
                    return reject("composed parent changed its original participant auxiliary identity");
                try
                {
                    // Both APIs validate retained immutable identity/liveness;
                    // neither requires worker-thread execution or native work.
                    participant_auxiliary_preparation->requireCurrentDefinition();
                    original_storage_attachment->requireCurrent();
                }
                catch (const std::exception& ex)
                {
                    return reject(std::string("composed parent lost its original participant auxiliary attachment: ") + ex.what());
                }
            }
            return DeviceLoopGraphTemplateView{.capture = retained_parent_capture.get(),
                .stream = capture_stream, .stage_count = execution_order.size(),
                .captured_node_count = retained_parent_capture->nodeCount(),.storage_boundary=graph.retainedStorageBoundary(),
                .read_ahead_resources=graph.readAheadBranch()?graph.readAheadBranch()->resources:nullptr};
        }
        if (segments.size() != 1)
        {
            return reject(
                "graph is segmented: replay_units=" +
                std::to_string(segments.size()));
        }

        const GraphSegment &segment = segments.front();
        if (!segment.capturable || !segment.capture ||
            !segment.capture->hasExecutable() ||
            segment.capture->nodeCount() == 0)
        {
            return reject("monolithic replay unit has no executable capture");
        }
        if (!capture_stream ||
            segment.capture->executionStream() != capture_stream)
        {
            return reject(
                "captured replay unit has ambiguous producer-stream ownership");
        }
        if (segment.stage_names != execution_order)
        {
            return reject(
                "captured replay unit does not cover the complete source graph");
        }
        for (const auto &stage_name : execution_order)
        {
            const ComputeNode *node = graph.getNode(stage_name);
            if (!node || !node->stage)
            {
                return reject(
                    "source graph contains an unresolved stage: " + stage_name);
            }
            if (node->stage->graphLaunchPreparationPolicy() ==
                GraphLaunchPreparationPolicy::CaptureAndReplay)
            {
                return reject(
                    "stage requires external preparation before every replay: " +
                    stage_name);
            }
        }

        return DeviceLoopGraphTemplateView{
            .capture = segment.capture.get(),
            .stream = capture_stream,
            .stage_count = segment.stage_names.size(),
            .captured_node_count = segment.capture->nodeCount(),
            .storage_boundary=graph.retainedStorageBoundary(),
            .read_ahead_resources=graph.readAheadBranch()?graph.readAheadBranch()->resources:nullptr,
        };
    }

    namespace
    {
        using SegmentCache = DeviceGraphExecutor::GraphSegmentCache;
        using UnitInspection =
            SegmentCache::RetainedCaptureUnitInspection;
        using UnitView = SegmentCache::RetainedCaptureUnitTemplateView;

        /** @brief Validate and export child captures for one exact lifecycle. */
        std::optional<std::vector<UnitView>> inspectRetainedCaptureUnits(
            const SegmentCache &cache,
            const ComputeGraph &graph,
            UnitInspection inspection,
            std::string *error)
        {
        auto reject = [&](const std::string &reason)
            -> std::optional<std::vector<UnitView>>
        {
            if (error)
                *error = reason;
            return std::nullopt;
        };
        if (error)
            error->clear();

        const auto &execution_order = graph.getExecutionOrder();
        if (execution_order.empty())
            return reject("source graph has no stages");
        const bool concurrent_ticket_service =
            DeviceGraphExecutor::hasConcurrentTicketService(
                cache.graph_replay_plan_policy);
        const bool has_auxiliary_body = graph.retainedParticipantAuxiliaryRange().has_value();
        if (has_auxiliary_body)
        {
            // Main compilation units omit the separately recorded finite body.
            // Validate that original declaration again before projecting it out;
            // malformed or foreign body stages must never disappear from proof.
            try
            {
                graph.requireRetainedParticipantAuxiliaryRange();
            }
            catch (const std::exception &ex)
            {
                return reject(std::string("invalid retained participant auxiliary declaration: ") + ex.what());
            }
            if (inspection != UnitInspection::ParentComposition)
                return reject("participant auxiliary children require their complete retained parent");
        }
        std::vector<std::string_view> main_execution_order;
        main_execution_order.reserve(execution_order.size());
        for (const auto &name : execution_order)
            if (!graph.isRetainedParticipantAuxiliaryNode(name))
                main_execution_order.emplace_back(name);
        switch (inspection)
        {
        case UnitInspection::ReplayReady:
            if (!cache.initialized || cache.needs_capture)
                return reject("graph cache is not replay-ready");
            break;
        case UnitInspection::ParentComposition:
            if (cache.initialized || !cache.needs_capture ||
                !DeviceGraphExecutor::isRetainedParentPlanPolicy(
                    cache.graph_replay_plan_policy) ||
                cache.retained_parent_capture)
            {
                return reject(
                    "graph cache is not in graph-only parent composition");
            }
            break;
        }
        if (!cache.capture_stream || cache.segments.empty())
        {
            return reject(
                "retained capture plan has no exact stream or replay units");
        }

        std::vector<UnitView> views;
        views.reserve(cache.segments.size());
        std::vector<std::string_view> represented_stages;
        represented_stages.reserve(execution_order.size());
        std::vector<std::string_view> captured_stages;
        std::vector<std::string_view> manual_stages;
        captured_stages.reserve(execution_order.size());
        manual_stages.reserve(execution_order.size());

        for (size_t segment_index = 0u;
             segment_index < cache.segments.size();
             ++segment_index)
        {
            const DeviceGraphExecutor::GraphSegment &segment =
                cache.segments[segment_index];
            if (!segment.capturable)
            {
                if (!concurrent_ticket_service ||
                    inspection != UnitInspection::ParentComposition)
                {
                    return reject(
                        "retained capture plan contains a manual replay unit at index " +
                        std::to_string(segment_index));
                }
                if (segment.stage_names.empty() ||
                    !segment.passive_capture_waves_after.empty())
                {
                    return reject(
                        "concurrent ticket-service plan has an empty or wave-bearing manual unit at index " +
                        std::to_string(segment_index));
                }
                for (const std::string &stage_name : segment.stage_names)
                {
                    const ComputeNode *node = graph.getNode(stage_name);
                    if (!node || !node->stage ||
                        !node->stage->isManualGraphBoundary() ||
                        node->stage->manualGraphBoundaryScheduling() !=
                            ManualGraphBoundaryScheduling::
                                ConcurrentTicketService)
                    {
                        return reject(
                            "concurrent ticket-service plan contains an uncertified manual stage: " +
                            stage_name);
                    }
                    represented_stages.emplace_back(stage_name);
                    manual_stages.emplace_back(stage_name);
                }
                continue;
            }
            if (segment.stage_names.empty() || !segment.capture ||
                segment.capture->nodeCount() == 0u)
            {
                return reject(
                    "retained capture plan has an incomplete native replay unit at index " +
                    std::to_string(segment_index));
            }
            const bool executable_state_valid =
                inspection == UnitInspection::ReplayReady
                    ? segment.capture->hasExecutable()
                    : !segment.capture->hasExecutable();
            if (!executable_state_valid)
            {
                return reject(
                    inspection == UnitInspection::ReplayReady
                        ? "retained replay unit has no executable at index " +
                              std::to_string(segment_index)
                        : "retained parent child was instantiated before composition at index " +
                              std::to_string(segment_index));
            }
            if (segment.capture->executionStream() != cache.capture_stream)
            {
                return reject(
                    "retained capture plan has ambiguous producer-stream ownership at index " +
                    std::to_string(segment_index));
            }

            for (const std::string &stage_name : segment.stage_names)
            {
                const ComputeNode *node = graph.getNode(stage_name);
                if (!node || !node->stage)
                {
                    return reject(
                        "retained capture plan contains an unresolved stage: " +
                        stage_name);
                }
                if (node->stage->graphLaunchPreparationPolicy() ==
                        GraphLaunchPreparationPolicy::CaptureAndReplay &&
                    !concurrent_ticket_service)
                {
                    return reject(
                        "stage requires external preparation before every replay: " +
                        stage_name);
                }
                represented_stages.emplace_back(stage_name);
                captured_stages.emplace_back(stage_name);
            }

            /*
             * Passive nodes are declarative rendezvous markers. They must occur
             * in the exact source-graph position, but importing an empty native
             * graph for them would invent device work and obscure authority.
             */
            for (const DeviceGraphExecutor::GraphSegment::PassiveCaptureWave &wave :
                 segment.passive_capture_waves_after)
            {
                for (const std::string &stage_name :
                     wave.declarative_noop_stage_names)
                {
                    const ComputeNode *node = graph.getNode(stage_name);
                    if (!node || !node->stage ||
                        !node->stage->isPassiveGraphCaptureNoOp())
                    {
                        return reject(
                            "retained capture plan has an invalid passive stage: " +
                            stage_name);
                    }
                    represented_stages.emplace_back(stage_name);
                }
            }

            views.push_back(UnitView{
                .capture = segment.capture.get(),
                .stream = cache.capture_stream,
                .stage_names = std::span<const std::string>(
                    segment.stage_names.data(), segment.stage_names.size()),
                .capture_wave_ordinal = segment.capture_wave_ordinal,
                .capture_wave_identity = segment.capture_wave_identity,
                .captured_node_count = segment.capture->nodeCount(),
            });
        }

        bool exact_stage_coverage = false;
        if (concurrent_ticket_service)
        {
            /*
             * A concurrent ticket plan is stored as two programs rather than
             * an alternating host schedule. Compare each program against its
             * projection of the declarative order; mapped device waits retain
             * the cross-program ordering at execution time.
             */
            std::vector<std::string_view> expected_captured_stages;
            std::vector<std::string_view> expected_manual_stages;
            expected_captured_stages.reserve(execution_order.size());
            expected_manual_stages.reserve(execution_order.size());
            for (const std::string &stage_name : execution_order)
            {
                if (graph.isRetainedParticipantAuxiliaryNode(stage_name))
                    continue;
                const ComputeNode *const node = graph.getNode(stage_name);
                if (!node || !node->stage)
                    return reject("source graph contains an unresolved stage");
                (node->stage->isManualGraphBoundary()
                     ? expected_manual_stages
                     : expected_captured_stages)
                    .emplace_back(stage_name);
            }
            exact_stage_coverage =
                captured_stages == expected_captured_stages &&
                manual_stages == expected_manual_stages;
        }
        else
        {
            exact_stage_coverage = represented_stages == main_execution_order;
        }
        if (!exact_stage_coverage)
        {
            return reject(
                "retained capture units do not cover the complete source graph in execution order");
        }
        if (views.empty())
            return reject("retained capture plan contains no native child graphs");
        return views;
        }
    } // namespace

    std::optional<std::vector<
        DeviceGraphExecutor::GraphSegmentCache::
            RetainedCaptureUnitTemplateView>>
    DeviceGraphExecutor::GraphSegmentCache::retainedCaptureUnitTemplates(
        const ComputeGraph &graph,
        std::string *error) const
    {
        return inspectRetainedCaptureUnits(
            *this,
            graph,
            RetainedCaptureUnitInspection::ReplayReady,
            error);
    }

    std::optional<std::vector<
        DeviceGraphExecutor::GraphSegmentCache::
            RetainedCaptureUnitTemplateView>>
    DeviceGraphExecutor::GraphSegmentCache::
        retainedCaptureUnitTemplatesForParentComposition(
            const ComputeGraph &graph,
            std::string *error) const
    {
        return inspectRetainedCaptureUnits(
            *this,
            graph,
            RetainedCaptureUnitInspection::ParentComposition,
            error);
    }

    namespace
    {
        /**
         * @brief Rebind every GPU stage for an explicitly selected eager pass.
         *
         * Capture and replay phases deliberately point stages at a graph-owned
         * stream. A caller that explicitly disables cached graph replay must
         * therefore establish one worker-owned stream for the complete eager
         * graph before execution. This helper is not an error-recovery path:
         * once replay is selected, replay failure is a hard execution failure.
         *
         * @param graph Graph whose stages will execute eagerly by policy.
         * @param gpu_stream Worker-owned stream supplied by the capture policy.
         * CPU-only graphs legitimately have no GPU stream. The helper first
         * validates the complete graph and determines whether any stage is
         * device-backed; only then does it require and publish the explicit
         * stream. This keeps CPU decode usable without weakening the GPU
         * ownership contract.
         *
         * @return true when every GPU graph stage was rebound, or when the
         *         graph is entirely CPU-owned.
         */
        bool bindGraphStagesForEagerExecution(ComputeGraph &graph, void *gpu_stream)
        {
            if(graph.readAheadBranch())throw std::logic_error("Read-ahead cannot enter eager execution");
            bool has_gpu_stage = false;

            for (const auto &name : graph.getExecutionOrder())
            {
                ComputeNode *node = graph.getNode(name);
                if (!node || !node->stage)
                {
                    LOG_ERROR("[DeviceGraphExecutor] Capture-policy eager execution cannot bind missing stage '"
                              << name << "'");
                    return false;
                }
                const DeviceId stage_device =
                    node->stage->device().is_valid()
                        ? node->stage->device()
                        : node->device;
                has_gpu_stage = has_gpu_stage || stage_device.is_gpu();
            }

            if (has_gpu_stage && !gpu_stream)
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Capture-policy eager GPU execution "
                    "requires an explicit non-null stream");
                return false;
            }

            if (has_gpu_stage)
            {
                for (const auto &name : graph.getExecutionOrder())
                {
                    ComputeNode *node = graph.getNode(name);
                    node->stage->setGPUStream(gpu_stream);
                }
            }
            return true;
        }

        /**
         * @brief Terminate after a graph-cache resource loses a valid lifecycle.
         *
         * A non-null stream or event is an owned backend resource. Once its
         * owner cannot be resolved, synchronization fails, or destruction
         * throws, neither dropping the handle nor continuing inference is
         * correct: queued work may still reference graph state that the host is
         * about to release. Centralizing the fatal policy here prevents future
         * lifecycle call sites from quietly reintroducing log-and-continue
         * behavior.
         *
         * @param detail Exact ownership or backend operation that failed.
         */
        [[noreturn]] void terminateGraphSegmentCacheLifecycle(
            const std::string &detail) noexcept
        {
            LOG_ERROR(
                "[GraphSegmentCache] Fatal GPU graph resource lifecycle "
                "violation: "
                << detail);
            std::terminate();
        }
    }

    // =========================================================================
    // GraphSegmentCache — capture stream management
    // =========================================================================

    IWorkerGPUContext *DeviceGraphExecutor::GraphSegmentCache::requireLifecycleContext(
        const char *operation)
    {
        if (capture_device.is_gpu() && capture_context_from_pool)
        {
            try
            {
                IWorkerGPUContext &resolved =
                    GPUDeviceContextPool::instance().getContext(capture_device);
                gpu_ctx_ref = &resolved;
                return &resolved;
            }
            catch (const std::exception &e)
            {
                terminateGraphSegmentCacheLifecycle(
                    std::string("failed to resolve the owning GPU context for ") +
                    operation + " on " + capture_device.toString() + ": " +
                    e.what());
            }
        }

        if (!gpu_ctx_ref)
        {
            terminateGraphSegmentCacheLifecycle(
                std::string("owned resource has no live GPU context during ") +
                operation);
        }
        return gpu_ctx_ref;
    }

    bool DeviceGraphExecutor::GraphSegmentCache::ensureCaptureStream(
        IWorkerGPUContext *ctx,
        DeviceId device,
        bool context_from_process_pool)
    {
        if (capture_stream)
        {
            if (capture_stream_ownership == CaptureStreamOwnership::None)
            {
                terminateGraphSegmentCacheLifecycle(
                    "capture stream has no typed ownership contract");
            }
            if (capture_device.is_valid() &&
                device.is_valid() &&
                !(capture_device == device))
            {
                LOG_ERROR("[GraphSegmentCache] Capture stream already belongs to "
                          << capture_device.toString() << ", cannot reuse for "
                          << device.toString());
                return false;
            }
            if (!capture_device.is_valid() && device.is_gpu())
            {
                capture_device = device;
            }
            capture_context_from_pool =
                capture_context_from_pool || context_from_process_pool;
            if (!gpu_ctx_ref)
                gpu_ctx_ref = ctx;
            if (!gpu_ctx_ref &&
                !(capture_device.is_gpu() && capture_context_from_pool))
            {
                terminateGraphSegmentCacheLifecycle(
                    "existing capture stream has no resolvable owner");
            }
            return true;
        }
        if (!ctx)
        {
            LOG_ERROR("[GraphSegmentCache] No GPU context for stream creation");
            return false;
        }
        capture_stream = ctx->createStream();
        if (!capture_stream)
        {
            LOG_ERROR("[GraphSegmentCache] Failed to create capture stream");
            return false;
        }
        capture_stream_ownership = CaptureStreamOwnership::Owned;
        gpu_ctx_ref = ctx;
        capture_device = device.is_gpu() ? device : DeviceId::invalid();
        capture_context_from_pool = context_from_process_pool;
        LOG_DEBUG("[GraphSegmentCache] Created local capture stream"
                  << (capture_device.is_valid()
                          ? std::string(" for ") + capture_device.toString()
                          : std::string{}));
        return true;
    }

    bool DeviceGraphExecutor::GraphSegmentCache::bindBorrowedCaptureStream(
        IWorkerGPUContext *ctx,
        void *stream,
        DeviceId device,
        bool context_from_process_pool)
    {
        if (!ctx || !stream || !device.is_gpu())
        {
            LOG_ERROR(
                "[GraphSegmentCache] Borrowed capture stream requires one live context, exact non-null stream, and GPU device");
            return false;
        }
        if (capture_stream)
        {
            if (capture_stream_ownership != CaptureStreamOwnership::Borrowed ||
                capture_stream != stream ||
                (capture_device.is_valid() && !(capture_device == device)) ||
                (gpu_ctx_ref && gpu_ctx_ref != ctx &&
                 !capture_context_from_pool))
            {
                LOG_ERROR(
                    "[GraphSegmentCache] Refusing to replace an established capture-stream ownership contract");
                return false;
            }
            capture_device = device;
            capture_context_from_pool =
                capture_context_from_pool || context_from_process_pool;
            if (!gpu_ctx_ref)
                gpu_ctx_ref = ctx;
            return true;
        }
        if (initialized || needs_capture || !segments.empty())
        {
            LOG_ERROR(
                "[GraphSegmentCache] Borrowed capture stream must be bound before graph materialization");
            return false;
        }

        capture_stream = stream;
        capture_stream_ownership = CaptureStreamOwnership::Borrowed;
        gpu_ctx_ref = ctx;
        capture_device = device;
        capture_context_from_pool = context_from_process_pool;
        return true;
    }

    void DeviceGraphExecutor::GraphSegmentCache::bindPreparedFollowerCaptureTerminals(
        std::shared_ptr<const MoEOverlayInferenceOriginalSlotDomain> domain,
        std::shared_ptr<PreparedFollowerCaptureTerminalPopulation> population,
        const std::shared_ptr<PhysicalMemoryAuthority>& memory){
        if(!memory || !domain || !population || !capture_stream || !capture_device.is_gpu() ||
            initialized || !segments.empty() || original_slot_terminals || terminal_event || terminal_event_obligation || isGraphCaptureActive())
            throw std::logic_error("Prepared follower terminals bind once before native materialization");
        auto* worker=requireLifecycleContext("prepared follower terminal binding");
        const auto original=GPUDeviceContextPool::instance().retainExistingPreparedContext(capture_device,memory);
        original.provenance().requireWorker(original.worker());
        if(original.worker().get()!=worker || !worker->ownsCurrentThread())throw std::logic_error("Prepared follower terminal binding requires original worker");
        auto receipts=OriginalSlotCaptureTerminals::createPrepared(domain,population,original.worker(),memory,capture_device,capture_stream);
        auto identity=std::make_shared<const std::uint8_t>(0);
        terminal_event=population->event(0);terminal_receipt_identity=std::move(identity);
        terminal_event_is_prepared_borrow=true;original_slot_terminals=std::move(receipts);
    }

    bool DeviceGraphExecutor::GraphSegmentCache::prepareCaptureStreamTerminal(
        IWorkerGPUContext *ctx)
    {
        if (!ctx || !capture_stream)
        {
            LOG_ERROR(
                "[GraphSegmentCache] Exact terminal preparation requires one "
                "bound capture stream and its live worker context");
            return false;
        }
        if (gpu_ctx_ref && gpu_ctx_ref != ctx && !capture_context_from_pool)
        {
            terminateGraphSegmentCacheLifecycle(
                "exact terminal event context differs from the capture stream owner");
        }
        if (terminal_event)
            return true;

        // Allocate immutable receipt identity only at this existing cold creator.
        // Copies of an issued ticket keep it unique even after event destruction.
        auto receipt_identity = std::make_shared<const std::uint8_t>(0);
        terminal_event = createCaptureTerminalEvent(*ctx);
        if (!terminal_event)
        {
            LOG_ERROR(
                "[GraphSegmentCache] Failed to create the exact terminal event");
            return false;
        }
        terminal_receipt_identity = std::move(receipt_identity);
        if (!gpu_ctx_ref)
            gpu_ctx_ref = ctx;
        return true;
    }

    void DeviceGraphExecutor::GraphSegmentCache::bindOriginalCaptureTerminalSlots(
        std::shared_ptr<const MoEOverlayInferenceOriginalSlotDomain> domain,
        std::shared_ptr<PhysicalMemoryAuthority> memory,
        const MoEOverlayInferenceOriginalSlotGeometry& original_geometry)
    {
        if(!domain || domain->geometry()!=original_geometry || !terminal_event_obligation)
            throw std::logic_error("Original capture terminal roster differs from its admitted retained geometry");
        if(initialized || !segments.empty() || original_slot_terminals ||
            terminal_fence_published_generation || terminal_fence_observed_generation || !terminal_event)
            throw std::logic_error("Original capture terminal slots bind exactly once before materialization");
        auto* context=requireLifecycleContext("original capture terminal slot binding");
        const auto original=GPUDeviceContextPool::instance().retainExistingPreparedContext(capture_device,memory);
        original.provenance().requireWorker(original.worker());
        if(original.worker().get()!=context)throw std::logic_error("Original capture terminal source differs from cache worker");
        original_slot_terminals=OriginalSlotCaptureTerminals::create(std::move(domain),original.worker(),
            std::move(memory),capture_device,capture_stream,terminal_event);
    }
    DeviceGraphExecutor::GraphSegmentCache::CaptureStreamTerminalTicket
    DeviceGraphExecutor::GraphSegmentCache::publishCaptureStreamTerminal(const MoEOverlayInferenceOriginalSlot& slot)
    {
        if(!original_slot_terminals)throw std::logic_error("Original capture terminal slots were not cold-bound");
        auto* context=requireLifecycleContext("original-slot terminal event publication");
        void* event=original_slot_terminals->eventForPublication(slot,context,capture_stream);
        if(!context->recordEventChecked(event,capture_stream))
            terminateGraphSegmentCacheLifecycle("Original-slot terminal event publication was rejected");
        auto receipt=original_slot_terminals->published(slot);
        return CaptureStreamTerminalTicket{receipt.generation(),terminal_receipt_identity,context,event,capture_stream,std::move(receipt)};
    }

    bool DeviceGraphExecutor::GraphSegmentCache::prepareCaptureStreamTerminal(
        IWorkerGPUContext* ctx,const std::shared_ptr<PhysicalMemoryAuthority>& memory)
    {
        if(!ctx || !memory || !capture_stream || !capture_device.is_gpu() ||
            initialized || !segments.empty() || isGraphCaptureActive())
            throw std::logic_error("Source-bound capture terminal requires its original cold cache/stream/PMA");
        const auto original=GPUDeviceContextPool::instance().retainExistingPreparedContext(capture_device,memory);
        original.provenance().requireWorker(original.worker());
        if(original.worker().get()!=ctx || !ctx->ownsCurrentThread())
            throw std::logic_error("Source-bound capture terminal changed its original prepared worker");
        const auto program=NativeOrderingEventProgram::captureSubmissionRetirement();
        if(terminal_event) {
            if(!terminal_event_obligation)
                throw std::logic_error("Source-bound capture terminal cannot retrofit an unpriced native creator");
            terminal_event_obligation->requireOriginal(memory,original.provenance(),program);
            return prepareCaptureStreamTerminal(ctx);
        }
        terminal_event_obligation=NativeOrderingEventGroup::claim(memory,original.provenance(),program);
        terminal_event_obligation->beginNativeAttempt();
        try {
            // The original exact creator remains single-source: same event,
            // immutable receipt identity and checked original capture stream.
            const bool prepared=prepareCaptureStreamTerminal(ctx);
            if(!prepared) {
                terminal_event_obligation->completeNativeHandles();
                terminal_event_obligation.reset();
            }
            return prepared;
        } catch(...) {
            if(terminal_event)throw; // Retain actual accepted handle/obligation for original cache teardown.
            terminal_event_obligation->completeNativeHandles();
            terminal_event_obligation.reset();
            throw;
        }
    }

    DeviceGraphExecutor::GraphSegmentCache::CaptureStreamTerminalTicket
    DeviceGraphExecutor::GraphSegmentCache::publishCaptureStreamTerminal()
    {
        if (original_slot_terminals)
            terminateGraphSegmentCacheLifecycle("Original-slot terminal publication requires its checked original ticket");
        if (!capture_stream || !terminal_event)
        {
            terminateGraphSegmentCacheLifecycle(
                "exact terminal publication was not prepared before inference");
        }
        if (terminal_fence_published_generation !=
            terminal_fence_observed_generation)
        {
            terminateGraphSegmentCacheLifecycle(
                "exact terminal event would overwrite an unobserved submission");
        }
        if (terminal_fence_published_generation ==
            std::numeric_limits<uint64_t>::max())
        {
            terminateGraphSegmentCacheLifecycle(
                "exact terminal generation namespace was exhausted");
        }

        IWorkerGPUContext *ctx =
            requireLifecycleContext("exact terminal event publication");
        if (!ctx->recordEventChecked(terminal_event, capture_stream))
        {
            terminateGraphSegmentCacheLifecycle(
                "exact terminal event publication was rejected by the backend");
        }
        ++terminal_fence_published_generation;
        return CaptureStreamTerminalTicket{
            terminal_fence_published_generation,terminal_receipt_identity,
            ctx,terminal_event,capture_stream};
    }

    void DeviceGraphExecutor::GraphSegmentCache::enqueuePublishedCaptureStreamTerminalWait(
        CaptureStreamTerminalTicket ticket,void* consumer_stream,IWorkerGPUContext* worker)
    {
        void* dependency_event=terminal_event;
        if(ticket.original_slot) {
            if(!original_slot_terminals || ticket.generation!=ticket.original_slot->generation())
                throw std::logic_error("Original-slot dependency lost its exact cache receipt");
            dependency_event=original_slot_terminals->eventForDeviceDependency(*ticket.original_slot,worker,capture_stream);
        }
        if(!worker || !consumer_stream || !ticket.valid() || !capture_stream || !dependency_event ||
            worker!=requireLifecycleContext("exact terminal device handoff") || !worker->ownsCurrentThread() ||
            ticket.worker_!=worker || ticket.event_!=dependency_event || ticket.stream_!=capture_stream ||
            ticket.identity_.get()!=terminal_receipt_identity.get() ||
            ticket.identity_.owner_before(terminal_receipt_identity) || terminal_receipt_identity.owner_before(ticket.identity_) ||
            (!ticket.original_slot && (original_slot_terminals || ticket.generation!=terminal_fence_published_generation)))
            throw std::logic_error("Exact terminal device handoff requires its original current receipt/worker/streams");
        if(!worker->waitEventChecked(dependency_event,consumer_stream))
            throw std::runtime_error("Exact terminal device handoff event wait was rejected");
    }

    void DeviceGraphExecutor::GraphSegmentCache::
        waitForPublishedCaptureStreamTerminal(
            CaptureStreamTerminalTicket ticket,
            HostFenceWaitPolicy wait_policy,
            std::function<std::string()> active_timeout_diagnostic)
    {
        void* observed_event=terminal_event;
        if(ticket.original_slot) {
            if(!original_slot_terminals || ticket.generation!=ticket.original_slot->generation())
                terminateGraphSegmentCacheLifecycle("Original-slot terminal receipt lost its original cache owner");
            observed_event=original_slot_terminals->eventForObservation(*ticket.original_slot,
                requireLifecycleContext("original-slot terminal observation"),capture_stream);
        }
        if (!ticket.valid() || !observed_event || !capture_stream ||
            ticket.identity_.get()!=terminal_receipt_identity.get() ||
            ticket.identity_.owner_before(terminal_receipt_identity) || terminal_receipt_identity.owner_before(ticket.identity_) ||
            ticket.event_!=observed_event || ticket.stream_!=capture_stream ||
            ticket.worker_!=requireLifecycleContext("exact terminal receipt ownership") ||
            (!ticket.original_slot && (original_slot_terminals ||
             ticket.generation != terminal_fence_published_generation ||
             ticket.generation <= terminal_fence_observed_generation)))
        {
            terminateGraphSegmentCacheLifecycle(
                "exact terminal observation received a missing, stale, or out-of-order ticket");
        }

        IWorkerGPUContext *ctx =
            requireLifecycleContext("exact terminal event observation");
        try
        {
            if (wait_policy == HostFenceWaitPolicy::ActiveProgress)
            {
                const auto deadline =
                    std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(
                        collective_timeout_policy::kDefaultCollectiveTimeoutMs);
                std::size_t poll_count = 0;
                for (;;)
                {
                    bool ready = false;
                    if (!ctx->queryEventChecked(observed_event, ready))
                    {
                        terminateGraphSegmentCacheLifecycle(
                            "exact terminal active query was rejected by the backend");
                    }
                    if (ready)
                        break;

                    ++poll_count;
                    if (std::chrono::steady_clock::now() >= deadline)
                    {
                        std::string diagnostic;
                        if (active_timeout_diagnostic)
                            diagnostic = active_timeout_diagnostic();
                        terminateGraphSegmentCacheLifecycle(
                            "exact capture-stream terminal exceeded the canonical "
                            "30-second deadline" +
                            (diagnostic.empty()
                                 ? std::string{}
                                 : ": " + diagnostic));
                    }
                    if ((poll_count & 1023u) == 0u)
                        std::this_thread::yield();
                }
            }
            else if (!ctx->synchronizeEventChecked(observed_event))
            {
                terminateGraphSegmentCacheLifecycle(
                    "exact terminal event wait was rejected by the backend");
            }

            /*
             * Event completion proves the submitted prefix retired, but a
             * backend may surface an asynchronous execution fault only when
             * its exact stream is queried. This observation never blocks and
             * treats both Pending (later queued work) and Complete as valid.
             */
            (void)ctx->queryStreamExecutionState(
                capture_stream,
                "retained graph exact terminal observation");
        }
        catch (const std::exception &e)
        {
            terminateGraphSegmentCacheLifecycle(
                std::string("exact terminal event observation threw: ") +
                e.what());
        }
        catch (...)
        {
            terminateGraphSegmentCacheLifecycle(
                "exact terminal event observation threw an unknown exception");
        }

        if(ticket.original_slot)original_slot_terminals->observed(*ticket.original_slot);
        else terminal_fence_observed_generation = ticket.generation;
    }

    void DeviceGraphExecutor::GraphSegmentCache::waitForCaptureStreamFence(
        HostFenceWaitPolicy wait_policy,
        std::function<std::string()> active_timeout_diagnostic)
    {
        if (!capture_stream)
            return;

        IWorkerGPUContext *ctx =
            requireLifecycleContext("capture stream event fence");

        try
        {
            if (!ensureCaptureOutputEvent(ctx))
            {
                terminateGraphSegmentCacheLifecycle(
                    "capture stream event fence allocation was rejected by the backend");
            }
            if (!ctx->recordEventChecked(sync_event, capture_stream))
            {
                terminateGraphSegmentCacheLifecycle(
                    "capture stream event fence publication was rejected by the backend");
            }
            if (wait_policy == HostFenceWaitPolicy::ActiveProgress)
            {
                const auto deadline =
                    std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(
                        collective_timeout_policy::kDefaultCollectiveTimeoutMs);
                std::size_t poll_count = 0;
                for (;;)
                {
                    bool ready = false;
                    if (!ctx->queryEventChecked(sync_event, ready))
                    {
                        terminateGraphSegmentCacheLifecycle(
                            "capture stream active event query was rejected by the backend");
                    }
                    if (ready)
                        break;

                    ++poll_count;
                    /*
                     * A backend event query is not guaranteed to be cheap
                     * while another queue has a resident wait kernel. Check
                     * wall time after every returned query: batching the check
                     * by poll count allowed 64 slow HIP queries to inflate the
                     * canonical 30-second deadline by several minutes.
                     */
                    if (std::chrono::steady_clock::now() >= deadline)
                    {
                        std::string diagnostic;
                        if (active_timeout_diagnostic)
                        {
                            diagnostic = active_timeout_diagnostic();
                        }
                        terminateGraphSegmentCacheLifecycle(
                            "capture stream active event fence exceeded the "
                            "canonical 30-second deadline" +
                            (diagnostic.empty()
                                 ? std::string{}
                                 : ": " + diagnostic));
                    }
                    /*
                     * Backend event queries already provide progress. Yield
                     * only occasionally so a pathological long-running kernel
                     * cannot monopolize its host CPU while preserving the
                     * sub-millisecond wake-up required by sparse transactions.
                     */
                    if ((poll_count & 1023u) == 0u)
                        std::this_thread::yield();
                }
            }
            else if (!ctx->synchronizeEventChecked(sync_event))
            {
                terminateGraphSegmentCacheLifecycle(
                    "capture stream event fence wait was rejected by the backend");
            }


            /* Surface late CUDA/HIP execution faults at the exact stream
             * boundary instead of allowing stale graph outputs to escape. */
            (void)ctx->queryStreamExecutionState(
                capture_stream,
                "graph capture stream fence observation");
        }
        catch (const std::exception &e)
        {
            terminateGraphSegmentCacheLifecycle(
                std::string("capture stream event fence threw: ") +
                e.what());
        }
        catch (...)
        {
            terminateGraphSegmentCacheLifecycle(
                "capture stream event fence threw an unknown exception");
        }
    }

    void DeviceGraphExecutor::GraphSegmentCache::destroyCaptureStream()
    {
        if (!capture_stream)
        {
            capture_stream_ownership = CaptureStreamOwnership::None;
            gpu_ctx_ref = nullptr;
            capture_device = DeviceId::invalid();
            capture_context_from_pool = false;
            return;
        }

        if (capture_stream_ownership == CaptureStreamOwnership::Owned)
        {
            IWorkerGPUContext *ctx =
                requireLifecycleContext("capture stream destruction");
            try
            {
                ctx->destroyStream(capture_stream);
            }
            catch (const std::exception &e)
            {
                terminateGraphSegmentCacheLifecycle(
                    std::string("capture stream destruction threw: ") + e.what());
            }
            catch (...)
            {
                terminateGraphSegmentCacheLifecycle(
                    "capture stream destruction threw an unknown exception");
            }
        }
        else if (capture_stream_ownership != CaptureStreamOwnership::Borrowed)
        {
            terminateGraphSegmentCacheLifecycle(
                "capture stream destruction found an untyped ownership state");
        }
        capture_stream = nullptr;
        capture_stream_ownership = CaptureStreamOwnership::None;
        gpu_ctx_ref = nullptr;
        capture_device = DeviceId::invalid();
        capture_context_from_pool = false;
    }

    bool DeviceGraphExecutor::GraphSegmentCache::ensureCaptureInputEvent(
        IWorkerGPUContext *ctx)
    {
        if (capture_input_event)
        {
            if (!gpu_ctx_ref &&
                !(capture_device.is_gpu() && capture_context_from_pool))
            {
                terminateGraphSegmentCacheLifecycle(
                    "existing capture-stream input event has no resolvable owner");
            }
            return true;
        }
        if (!ctx)
            return false;
        if (gpu_ctx_ref && gpu_ctx_ref != ctx && !capture_context_from_pool)
        {
            terminateGraphSegmentCacheLifecycle(
                "capture-stream handoff event context differs from the capture "
                "stream owner");
        }
        capture_input_event = ctx->createEvent();
        if (!capture_input_event)
        {
            LOG_ERROR(
                "[GraphSegmentCache] Failed to create the capture-stream input event");
            return false;
        }
        if (!gpu_ctx_ref)
            gpu_ctx_ref = ctx;
        return true;
    }

    bool DeviceGraphExecutor::GraphSegmentCache::ensureCaptureOutputEvent(
        IWorkerGPUContext *ctx)
    {
        if (sync_event)
        {
            if (!gpu_ctx_ref &&
                !(capture_device.is_gpu() && capture_context_from_pool))
            {
                terminateGraphSegmentCacheLifecycle(
                    "existing capture-stream output event has no resolvable owner");
            }
            return true;
        }
        if (!ctx)
            return false;
        if (gpu_ctx_ref && gpu_ctx_ref != ctx && !capture_context_from_pool)
        {
            terminateGraphSegmentCacheLifecycle(
                "capture-stream output event context differs from the capture stream owner");
        }
        sync_event = ctx->createEvent();
        if (!sync_event)
        {
            LOG_ERROR(
                "[GraphSegmentCache] Failed to create the capture-stream output event");
            return false;
        }
        if (!gpu_ctx_ref)
            gpu_ctx_ref = ctx;
        return true;
    }

    bool DeviceGraphExecutor::GraphSegmentCache::orderCaptureStreamAfter(
        IWorkerGPUContext *ctx,
        void *producer_stream)
    {
        if (!ctx || !producer_stream || !capture_stream)
        {
            LOG_ERROR(
                "[GraphSegmentCache] Capture-stream handoff requires one "
                "context and two explicit streams");
            return false;
        }
        if (producer_stream == capture_stream)
            return true;
        if (!ensureCaptureInputEvent(ctx))
            return false;
        if (!ctx->recordEventChecked(capture_input_event, producer_stream))
        {
            LOG_ERROR(
                "[GraphSegmentCache] Failed to record capture-stream handoff "
                "event on the producer stream");
            return false;
        }
        if (!ctx->waitEventChecked(capture_input_event, capture_stream))
        {
            LOG_ERROR(
                "[GraphSegmentCache] Failed to queue capture-stream wait for "
                "the producer handoff event");
            return false;
        }
        return true;
    }

    bool DeviceGraphExecutor::GraphSegmentCache::orderStreamAfterCapture(
        IWorkerGPUContext *ctx,
        void *consumer_stream)
    {
        if (!ctx || !consumer_stream || !capture_stream)
        {
            LOG_ERROR(
                "[GraphSegmentCache] Capture-stream publication requires one "
                "context and two explicit streams");
            return false;
        }
        if (consumer_stream == capture_stream)
            return true;
        if (!ensureCaptureOutputEvent(ctx))
            return false;
        if (!ctx->recordEventChecked(sync_event, capture_stream))
        {
            LOG_ERROR(
                "[GraphSegmentCache] Failed to record capture-stream output "
                "handoff event");
            return false;
        }
        if (!ctx->waitEventChecked(sync_event, consumer_stream))
        {
            LOG_ERROR(
                "[GraphSegmentCache] Failed to queue captured-output wait on "
                "the consumer stream");
            return false;
        }
        return true;
    }

    void DeviceGraphExecutor::GraphSegmentCache::destroySyncEvent()
    {
        if (!sync_event && !capture_input_event)
            return;
        IWorkerGPUContext *ctx =
            requireLifecycleContext("sync event destruction");
        try
        {
            if (sync_event)
                ctx->destroyEvent(sync_event);
            if (capture_input_event)
                ctx->destroyEvent(capture_input_event);
        }
        catch (const std::exception &e)
        {
            terminateGraphSegmentCacheLifecycle(
                std::string("capture-stream handoff event destruction threw: ") +
                e.what());
        }
        catch (...)
        {
            terminateGraphSegmentCacheLifecycle(
                "capture-stream handoff event destruction threw an unknown "
                "exception");
        }
        sync_event = nullptr;
        capture_input_event = nullptr;
    }

    void DeviceGraphExecutor::GraphSegmentCache::destroyTerminalEvent()
    {
        // The original cache reset observed every receipt before any graph,
        // source, stream or exact event owner can be withdrawn.
        original_slot_terminals.reset();
        if(terminal_event_is_prepared_borrow){
            if(terminal_fence_published_generation!=terminal_fence_observed_generation)
                terminateGraphSegmentCacheLifecycle("Prepared terminal return lacks original native observation");
            terminal_event_is_prepared_borrow=false;terminal_event=nullptr;terminal_receipt_identity.reset();return;
        }
        if (!terminal_event)
            return;
        if (terminal_fence_published_generation !=
            terminal_fence_observed_generation)
        {
            terminateGraphSegmentCacheLifecycle(
                "exact terminal event destruction found an unobserved submission");
        }

        IWorkerGPUContext *ctx =
            requireLifecycleContext("exact terminal event destruction");
        try
        {
            destroyCaptureTerminalEvent(*ctx,terminal_event);
        }
        catch (const std::exception &e)
        {
            terminateGraphSegmentCacheLifecycle(
                std::string("exact terminal event destruction threw: ") +
                e.what());
        }
        catch (...)
        {
            terminateGraphSegmentCacheLifecycle(
                "exact terminal event destruction threw an unknown exception");
        }
        terminal_event = nullptr;
        if(terminal_event_obligation) {
            terminal_event_obligation->completeNativeHandles();
            terminal_event_obligation.reset();
        }
        terminal_receipt_identity.reset();
    }

    void DeviceGraphExecutor::GraphSegmentCache::destroyReplayGpuTimingEvents()
    {
        if (replay_gpu_timing_slots.empty())
        {
            replay_gpu_timing_device_name.clear();
            replay_gpu_timing_busy_samples = 0;
            return;
        }

        IWorkerGPUContext *ctx =
            requireLifecycleContext("replay GPU timing event destruction");
        try
        {
            for (auto &slot : replay_gpu_timing_slots)
            {
                if (slot.start_event)
                    ctx->destroyEvent(slot.start_event);
                if (slot.stop_event)
                    ctx->destroyEvent(slot.stop_event);
                slot = {};
            }
        }
        catch (const std::exception &e)
        {
            terminateGraphSegmentCacheLifecycle(
                std::string("replay GPU timing event destruction threw: ") +
                e.what());
        }
        catch (...)
        {
            terminateGraphSegmentCacheLifecycle(
                "replay GPU timing event destruction threw an unknown exception");
        }

        replay_gpu_timing_slots.clear();
        replay_gpu_timing_device_name.clear();
        replay_gpu_timing_busy_samples = 0;
    }

    // =========================================================================
    // Single-Graph Capture/Replay
    // =========================================================================

    std::unique_ptr<GraphCaptureDependencyLedger>
    DeviceGraphExecutor::planGraphCaptureDependencies(
        ComputeGraph &graph,
        std::span<const std::string> stage_names,
        DeviceId capture_device,
        void *capture_stream,
        const char *context,
        std::span<const BufferId> retained_parent_input_ids,
        GraphCaptureDependencyLedger::ExternalInputAuthority
            external_input_authority)
    {
        if (!capture_device.is_gpu() || !capture_stream)
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Capture dependency planning requires an "
                "exact GPU and non-null stream");
            return nullptr;
        }

        const std::string capture_context =
            context ? context : "graph_capture_dependencies";
        graph.requireReadAheadRecordingSealed(capture_stream);
        std::vector<GraphCaptureDependencyLedger::StagePlan> stage_plans;
        stage_plans.reserve(stage_names.size());

        /*
         * BufferId, rather than a tensor's current coherence flag, defines the
         * topological producer.  A tensor may still be HOST_AUTHORITATIVE while
         * its producer kernel is merely being recorded.  The canonical storage
         * owner is retained only after that BufferId decision has been made.
         */
        std::unordered_map<BufferId, size_t> latest_producer;
        const std::unordered_set<BufferId> retained_parent_inputs(
            retained_parent_input_ids.begin(),
            retained_parent_input_ids.end());

        auto canonical_owner = [&](BufferId id,
                                   const std::string &stage_name,
                                   const char *access) -> const TensorBase *
        {
            if (!arena_)
            {
                LOG_ERROR("[DeviceGraphExecutor] Capture dependency stage '"
                          << stage_name << "' declares arena " << access
                          << " " << bufferIdName(id) << " without a BufferArena ("
                          << capture_context << ")");
                return nullptr;
            }
            ITensor *tensor = arena_->getTensor(id);
            auto *base = dynamic_cast<TensorBase *>(tensor);
            const TensorBase *owner = base ? base->transferStorageOwner() : nullptr;
            if (!owner)
            {
                LOG_ERROR("[DeviceGraphExecutor] Capture dependency stage '"
                          << stage_name << "' has no canonical tensor owner for "
                          << access << " " << bufferIdName(id) << " ("
                          << capture_context << ")");
            }
            return owner;
        };

        for (size_t stage_index = 0; stage_index < stage_names.size(); ++stage_index)
        {
            const std::string &stage_name = stage_names[stage_index];
            ComputeNode *node = graph.getNode(stage_name);
            if (!node || !node->stage)
            {
                LOG_ERROR("[DeviceGraphExecutor] Capture dependency plan is missing stage '"
                          << stage_name << "' (" << capture_context << ")");
                return nullptr;
            }

            DeviceId stage_device =
                node->device.is_valid() ? node->device : node->stage->device();
            if (!stage_device.is_valid())
                stage_device = capture_device;
            if (stage_device != capture_device)
            {
                LOG_ERROR("[DeviceGraphExecutor] Capture dependency stage '"
                          << stage_name << "' belongs to "
                          << stage_device.toString() << " instead of "
                          << capture_device.toString() << " ("
                          << capture_context << ")");
                return nullptr;
            }

            GraphCaptureDependencyLedger::StagePlan plan;
            plan.stage_identity = node->stage.get();
            plan.stage_name = stage_name;
            plan.recording_stream = graph.executionStreamForNode(stage_name,capture_stream);
            const StageBufferContract contract = node->stage->bufferContract();

            std::unordered_map<const TensorBase *, size_t> internal_owners;
            std::unordered_set<const TensorBase *> retained_parent_owners;
            std::unordered_set<const TensorBase *> external_owners;
            for (const auto &binding : contract.allArenaReads())
            {
                const TensorBase *owner =
                    canonical_owner(binding.id, stage_name, "read");
                if (!owner)
                    return nullptr;

                const auto producer = latest_producer.find(binding.id);
                if (producer == latest_producer.end())
                {
                    if (retained_parent_inputs.contains(binding.id))
                    {
                        external_owners.erase(owner);
                        if (internal_owners.count(owner) == 0)
                            retained_parent_owners.insert(owner);
                    }
                    else if (internal_owners.count(owner) == 0 &&
                             retained_parent_owners.count(owner) == 0)
                    {
                        external_owners.insert(owner);
                    }
                    continue;
                }

                external_owners.erase(owner);
                retained_parent_owners.erase(owner);
                auto [it, inserted] =
                    internal_owners.emplace(owner, producer->second);
                if (!inserted)
                    it->second = std::max(it->second, producer->second);
            }

            plan.external_inputs.assign(
                external_owners.begin(), external_owners.end());
            plan.retained_parent_inputs.assign(
                retained_parent_owners.begin(), retained_parent_owners.end());
            plan.internal_inputs.reserve(internal_owners.size());
            for (const auto &[owner, producer_stage_index] : internal_owners)
            {
                plan.internal_inputs.push_back({
                    .tensor = owner,
                    .producer_stage_index = producer_stage_index,
                });
            }

            std::unordered_set<const TensorBase *> output_owners;
            for (const auto &binding : contract.allWrites())
            {
                const TensorBase *owner =
                    canonical_owner(binding.id, stage_name, "write");
                if (!owner)
                    return nullptr;
                if (output_owners.insert(owner).second)
                    plan.outputs.push_back(owner);
                latest_producer[binding.id] = stage_index;
            }

            stage_plans.push_back(std::move(plan));
        }

        IWorkerGPUContext *event_worker=nullptr;
        if(const auto& declaration=graph.readAheadBranch())
        {
            const auto& branch=*declaration;
            auto index=[&](const std::string& name) -> std::optional<size_t> {
                for(size_t i=0;i<stage_names.size();++i)if(stage_names[i]==name)return i;
                return std::nullopt;
            };
            const auto fork=index(branch.fork_node),producer=index(branch.producer_node),join=index(branch.join_node);
            if(fork || producer || join)
            {
                if(!fork || !producer || !join)
                    throw std::logic_error("Read-ahead fork/producer/planned join crosses an unsupported native unit");
                if(stage_plans[*producer].recording_stream!=branch.resources->stream() ||
                   stage_plans[*fork].recording_stream!=capture_stream ||
                   stage_plans[*join].recording_stream!=capture_stream)
                    throw std::logic_error("Read-ahead native unit does not bind the declared primary/side fork and join lanes");
                stage_plans[*fork].completion_event=branch.resources->forkEvent();
                stage_plans[*producer].wait_for_stage=*fork;
                stage_plans[*producer].completion_event=branch.resources->joinEvent();
                stage_plans[*join].wait_for_stage=*producer;
                event_worker=&branch.resources->workerContext();
            }
            // Later PLE units import the original embedding as a typed retained
            // parent input. Their existing native child edge follows this joined
            // unit; no second gather, host readback or fabricated authority exists.
        }

        try
        {
            auto ledger=std::make_unique<GraphCaptureDependencyLedger>(
                capture_device,
                capture_stream,
                std::move(stage_plans),
                capture_context,
                external_input_authority,
                nullptr,event_worker);
            if(event_worker)
                PerfStatsCollector::addCounter("forward_graph","read_ahead_dag_materializations",1.0,
                    "graph_setup",capture_device.toString(),
                    {{"side_lanes","1"},{"event_operations","4"},
                     {"capture_identity",std::to_string(graph.readAheadCaptureIdentity())}});
            return ledger;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[DeviceGraphExecutor] Capture dependency planning failed ("
                      << capture_context << "): " << e.what());
            return nullptr;
        }
    }

    bool DeviceGraphExecutor::recordGraphCaptureBody(
        ComputeGraph &graph,
        IDeviceContext *ctx,
        IGPUGraphCapture *capture,
        GraphCaptureRecordPurpose purpose,
        const char *context,
        IWorkerGPUContext *setup_worker)
    {
        graph.requireExecutable();
        const bool setup_only = purpose == GraphCaptureRecordPurpose::SetupOnly;
        const bool retained = purpose != GraphCaptureRecordPurpose::ImmediateReplay;
        if (setup_only && (!setup_worker || !setup_worker->ownsCurrentThread()))
            throw std::logic_error("Setup recording requires the exact original worker thread");
        const auto input_authority = setup_only
            ? GraphCaptureDependencyLedger::ExternalInputAuthority::BindDeclaredAddressesOnly
            : GraphCaptureDependencyLedger::ExternalInputAuthority::RequireReadyBytes;
        const char *const capture_context =
            context && context[0] != '\0'
                ? context
                : (purpose == GraphCaptureRecordPurpose::RetainedComposition
                       ? "retained_graph_fragment"
                       : "single_graph_capture");
        if (!capture)
        {
            LOG_ERROR("[DeviceGraphExecutor] " << capture_context
                                                << " requires a non-null capture owner");
            return false;
        }
        if (!ctx || !ctx->deviceId().is_gpu())
        {
            LOG_ERROR("[DeviceGraphExecutor] " << capture_context
                                                << " requires an exact GPU execution context");
            return false;
        }

        void *const gpu_stream = capture->executionStream();
        if (!gpu_stream)
        {
            LOG_ERROR("[DeviceGraphExecutor] " << capture_context
                                                << " capture owner has no explicit execution stream");
            return false;
        }
        if (retained &&
            (capture->hasExecutable() || capture->nodeCount() != 0u))
        {
            LOG_ERROR("[DeviceGraphExecutor] " << capture_context
                                                << " requires a pristine retained-child capture owner");
            return false;
        }

        const auto &order = graph.getExecutionOrder();
        if (order.empty())
        {
            LOG_ERROR("[DeviceGraphExecutor] " << capture_context
                                                << " cannot record an empty graph");
            return false;
        }

        /*
         * A retained child is a complete participant-local native unit. A bound
         * captured-transfer endpoint remains a collective, but performs no
         * hidden host/rank coordination. Runtime collectives stay outside. Device
         * selection belongs to topology planning, so this recorder accepts any
         * GPU backend but requires every node and stage to agree with the exact
         * selected context. Cross-participant synchronization belongs to the
         * parent transaction and may not be hidden inside the child.
         */
        for (const auto &name : order)
        {
            auto *node = graph.getNode(name);
            if (!node || !node->stage)
            {
                LOG_ERROR("[DeviceGraphExecutor] " << capture_context
                                                    << " contains a missing stage: " << name);
                return false;
            }
            const DeviceId stage_device =
                node->device.is_valid() ? node->device : node->stage->device();
            if (stage_device != ctx->deviceId() ||
                node->stage->device() != ctx->deviceId())
            {
                LOG_ERROR("[DeviceGraphExecutor] " << capture_context
                                                    << " stage '" << name
                                                    << "' does not belong to execution device "
                                                    << ctx->deviceId().toString());
                return false;
            }
            if (!node->stage->supportsBackend(ctx->backendType()))
            {
                LOG_ERROR("[DeviceGraphExecutor] " << capture_context
                                                    << " stage '" << name
                                                    << "' rejects selected backend");
                return false;
            }
            if ((node->stage->isCollectiveStage() && !node->stage->isParticipantLocalCapturedTransfer()) ||
                node->stage->isManualGraphBoundary())
            {
                LOG_ERROR("[DeviceGraphExecutor] " << capture_context
                                                    << " stage '" << name
                                                    << "' crosses the participant-local capture boundary");
                return false;
            }
            if (retained &&
                node->graph_capture_wave.has_value())
            {
                LOG_ERROR("[DeviceGraphExecutor] " << capture_context
                                                    << " stage '" << name
                                                    << "' embeds a domain capture wave owned by its parent transaction");
                return false;
            }
            if (!node->stage->isGraphCapturable() &&
                !node->stage->supportsGraphCaptureAfterLaunchPreparation())
            {
                LOG_ERROR("[DeviceGraphExecutor] " << capture_context
                                                    << " stage '" << name
                                                    << "' has no graph-capture implementation");
                return false;
            }
        }

        graph.planReadAheadRecordingStream(capture->executionStream());

        // Assign the exact capture stream before metadata preparation. Stages
        // may create immutable pointer tables here, never during native capture.
        for (const auto &name : order)
        {
            graph.getNode(name)->stage->setGPUStream(graph.executionStreamForNode(name,gpu_stream));
        }

        /*
         * Cold capture has no eager execution to allocate arena scratch before
         * stages build pointer-bearing launch descriptors. Bind every stable
         * address first without transferring bytes or publishing authority.
         */
        if (!allocateGraphStorageForCapture(graph, ctx, capture_context))
        {
            LOG_ERROR("[DeviceGraphExecutor] " << capture_context
                                                << " storage prebind failed");
            return false;
        }

        for (const auto &name : order)
        {
            auto *node = graph.getNode(name);
            if (!requiresGraphLaunchPreparation(
                    node->stage->graphLaunchPreparationPolicy(),
                    GraphLaunchPreparationPhase::Capture))
            {
                continue;
            }
            if (!node->stage->prepareGraphLaunch(ctx, node->stage->gpuStream()))
            {
                LOG_ERROR("[DeviceGraphExecutor] " << capture_context
                                                    << " launch preparation failed for stage '"
                                                    << name << "'");
                return false;
            }
        }

        /*
         * Preparation-dependent stages must now expose their strict readiness.
         * A retained child also rejects per-replay preparation: its parent is
         * device-owned and cannot safely re-enter stage C++ between epochs.
         */
        for (const auto &name : order)
        {
            auto *stage = graph.getNode(name)->stage.get();
            if (!stage->isGraphCapturable())
            {
                const std::string readiness =
                    stage->graphCaptureReadinessDebugString();
                LOG_ERROR("[DeviceGraphExecutor] " << capture_context
                                                    << " prepared stage '" << name
                                                    << "' is not graph-capturable"
                                                    << (readiness.empty() ? "" : "; ")
                                                    << readiness);
                return false;
            }
            if (retained &&
                stage->graphLaunchPreparationPolicy() ==
                    GraphLaunchPreparationPolicy::CaptureAndReplay)
            {
                LOG_ERROR("[DeviceGraphExecutor] " << capture_context
                                                    << " stage '" << name
                                                    << "' requires host launch preparation on every replay");
                return false;
            }
        }

        /*
         * Join all eager metadata/input producers before beginCapture(). Once
         * recording starts, every dependency must be internal to this child.
         */
        if (!prepareGraphStorageForCapture(
                graph, ctx, gpu_stream, capture_context, input_authority))
        {
            LOG_ERROR("[DeviceGraphExecutor] " << capture_context
                                                << " storage preflight failed");
            return false;
        }

        // Descriptor/storage admission is cold. The stage executor records each
        // immutable copy on its producer lane before completing the event edge.
        if (graph.readAheadBranch() &&
            !prepareSnapshotsForGraphCapture(graph,ctx,gpu_stream,capture_context))
            return false;
        graph.sealReadAheadRecordingStream(capture->executionStream());
        auto dependency_ledger = planGraphCaptureDependencies(
            graph,
            std::span<const std::string>(order.data(), order.size()),
            ctx->deviceId(),
            gpu_stream,
            capture_context,
            {}, input_authority);
        if (!dependency_ledger)
        {
            LOG_ERROR("[DeviceGraphExecutor] " << capture_context
                                                << " dependency planning failed");
            return false;
        }

        std::vector<IComputeStage *> capture_stages;
        capture_stages.reserve(order.size());
        for (const auto &name : order)
        {
            ComputeNode *node = graph.getNode(name);
            capture_stages.push_back(
                node && node->stage ? node->stage.get() : nullptr);
        }

        ScopedGraphCaptureStageActivity stage_capture_activity(
            capture_stages,
            ctx,
            gpu_stream,
            capture_context);
        // Construct the same nonmovable guard directly; only setup owns a
        // worker software-capture flag. Prvalue construction adds no heap owner.
        ScopedBackendGraphCapture capture_transaction(
            [&]() -> ScopedBackendGraphCapture {
                if (setup_only)
                    return ScopedBackendGraphCapture(*setup_worker, *capture,
                        capture_context, dependency_ledger.get());
                return ScopedBackendGraphCapture(*capture, capture_context,
                    dependency_ledger.get());
            }());
        if (!stage_capture_activity.begin())
        {
            LOG_ERROR("[DeviceGraphExecutor] " << capture_context
                                                << " stage capture activity admission failed");
            return false;
        }
        if (!capture_transaction.begin())
        {
            (void)stage_capture_activity.finish(
                GraphCaptureActivityTransition::Aborted);
            LOG_ERROR("[DeviceGraphExecutor] " << capture_context
                                                << " beginCapture failed");
            return false;
        }

        DeviceGraphCaptureController::prepareDeviceForGraphCapture(ctx);
        bool exec_success = true;
        const StageRunPolicy capture_policy = setup_only
            ? StageRunPolicy::setupGraphMaterialization() : StageRunPolicy::capturePhase();
        for (const auto &name : order)
        {
            auto *node = graph.getNode(name);
            if (!runStage(
                    *node,
                    ctx,
                    capture_policy,
                    /*is_collective=*/false))
            {
                LOG_ERROR("[DeviceGraphExecutor] Stage failed during "
                          << capture_context << ": " << name);
                exec_success = false;
                break;
            }
            if (!setup_only) graph.markCompleted(name);
        }

        // End capture unconditionally before inspecting failure or graph state.
        capture_transaction.finish();
        const bool usable_capture =
            exec_success && capture->nodeCount() != 0u;
        if (!stage_capture_activity.finish(
                usable_capture
                    ? GraphCaptureActivityTransition::Completed
                    : GraphCaptureActivityTransition::Aborted))
        {
            LOG_ERROR("[DeviceGraphExecutor] " << capture_context
                                                << " stage capture activity terminal publication failed");
            capture->reset();
            return false;
        }
        if (!exec_success)
        {
            LOG_ERROR("[DeviceGraphExecutor] A stage failed during mandatory "
                      << capture_context << "; refusing eager recovery");
            capture->reset();
            return false;
        }
        if (!usable_capture)
        {
            LOG_ERROR("[DeviceGraphExecutor] Mandatory " << capture_context
                                                          << " produced zero nodes; refusing capture-time eager execution");
            capture->reset();
            return false;
        }

        LOG_DEBUG("[DeviceGraphExecutor] Recorded " << capture->nodeCount()
                                                     << " nodes for " << capture_context);
        return true;
    }

    bool DeviceGraphExecutor::captureRetainedGraphFragment(
        ComputeGraph &graph,
        IDeviceContext *ctx,
        IGPUGraphCapture *capture,
        const char *context)
    {
        graph.requireExecutable();
        return recordGraphCaptureBody(
            graph,
            ctx,
            capture,
            GraphCaptureRecordPurpose::RetainedComposition,
            context);
    }

    bool DeviceGraphExecutor::captureSetupGraphFragment(
        ComputeGraph &graph, IDeviceContext *ctx, IGPUGraphCapture *capture,
        const PreparedGPUContextUsePermit &original, const char *context)
    {
        original.requireWorker(original.worker());
        if (!ctx || ctx->deviceId() != original.device() || !capture ||
            reinterpret_cast<std::uintptr_t>(capture->executionStream()) <= 2u)
            throw std::invalid_argument("Setup recording requires its original device and exact capture stream");
        return recordGraphCaptureBody(graph, ctx, capture,
            GraphCaptureRecordPurpose::SetupOnly, context, original.worker().get());
    }

    bool DeviceGraphExecutor::executeWithGraphCapture(ComputeGraph &graph, IDeviceContext *ctx,
                                                      IGPUGraphCapture *capture,
                                                      std::span<ITensor *const> externally_visible_outputs,
                                                      const std::unordered_set<std::string> *collective_nodes)
    {
        graph.requireExecutable();
        if (!capture)
        {
            LOG_ERROR("[DeviceGraphExecutor] GPU graph capture requires a non-null capture owner");
            return false;
        }

        // TP>1 with collectives cannot be captured in a single-device graph
        if (collective_nodes && !collective_nodes->empty())
        {
            LOG_ERROR("[DeviceGraphExecutor] Single-device graph capture cannot own collective nodes");
            return false;
        }

        if (!ctx || !ctx->deviceId().is_gpu())
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Single-graph capture requires an explicit "
                "GPU execution context");
            return false;
        }

        void *const gpu_stream = capture->executionStream();
        if (!gpu_stream)
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Single-graph capture owner has no explicit "
                "execution stream");
            return false;
        }

        /*
         * Capture-time tensor writes intentionally cannot publish host-waitable
         * completion events: those event records would become graph nodes and
         * describe recording rather than replay completion. The graph boundary
         * must therefore declare every tensor that may escape this call.
         * Validate the complete declaration before capture begins so an invalid
         * output cannot leave a partially published graph launch behind.
         */
        if (externally_visible_outputs.empty())
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Single-graph capture requires at least "
                "one externally visible output publication");
            return false;
        }

        std::unordered_set<ITensor *> unique_outputs;
        unique_outputs.reserve(externally_visible_outputs.size());
        for (ITensor *output : externally_visible_outputs)
        {
            if (!output)
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Single-graph capture output "
                    "publication contains a null tensor");
                return false;
            }
            if (!unique_outputs.insert(output).second)
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Single-graph capture output "
                    "publication contains a duplicate tensor");
                return false;
            }

            auto *base = dynamic_cast<TensorBase *>(output);
            if (!base || !base->gpu_data_ptr() ||
                !base->current_device().has_value() ||
                *base->current_device() != ctx->deviceId())
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Single-graph capture output is not "
                    "resident on the execution device "
                    << ctx->deviceId().toString());
                return false;
            }
        }

        if (!recordGraphCaptureBody(
                graph,
                ctx,
                capture,
                GraphCaptureRecordPurpose::ImmediateReplay,
                "single_graph_capture"))
        {
            return false;
        }

        // Step 4: Select the one executable-publication operation supported by
        // the capture backend, then launch. This capability branch is static:
        // a backend that cannot update in place directly replaces its old
        // executable, without issuing a doomed update call and attempting to
        // recover from the resulting runtime error.
        LOG_DEBUG("[DeviceGraphExecutor] GPU graph captured " << capture->nodeCount()
                                                              << " nodes, hasExecutable=" << capture->hasExecutable());
        const bool can_update_existing_executable =
            capture->hasExecutable() && capture->supportsExecutableUpdate();
        if (can_update_existing_executable)
        {
            const GraphUpdateResult result = capture->tryUpdate();
            LOG_DEBUG("[DeviceGraphExecutor] tryUpdate result=" << static_cast<int>(result));
            if (result == GraphUpdateResult::Success)
            {
                LOG_TRACE("[DeviceGraphExecutor] GPU graph executable updated in place ("
                          << capture->nodeCount() << " nodes)");
            }
            else if (result == GraphUpdateResult::NeedsReinstantiate)
            {
                LOG_DEBUG("[DeviceGraphExecutor] NeedsReinstantiate; instantiating the captured topology");
                if (!capture->instantiate())
                {
                    LOG_ERROR("[DeviceGraphExecutor] GPU graph reinstantiation failed");
                    return false;
                }
            }
            else
            {
                // Update failed
                LOG_ERROR("[DeviceGraphExecutor] GPU graph update failed (result="
                          << static_cast<int>(result) << ")");
                return false;
            }
        }
        else
        {
            LOG_DEBUG("[DeviceGraphExecutor] Instantiating captured graph "
                      << (capture->hasExecutable() ? "as an executable replacement" : "for the first time")
                      << " (" << capture->nodeCount() << " nodes)");
            if (!capture->instantiate())
            {
                LOG_ERROR("[DeviceGraphExecutor] GPU graph instantiation failed");
                return false;
            }
            LOG_DEBUG("[DeviceGraphExecutor] GPU graph instantiated with " << capture->nodeCount()
                                                                           << " nodes (" << capture->backendName() << ")");
        }

        // Launch exactly once whether publication updated or replaced the
        // executable. A failed publication never reaches this point.
        if (!capture->launch())
        {
            LOG_ERROR("[DeviceGraphExecutor] GPU graph launch failed");
            return false;
        }

        /*
         * This is the sole externally visible publication boundary for the
         * direct single-graph API. Every event is recorded after launch on the
         * capture-owned stream, so a host read or another device stream consumes
         * the exact replay that produced the declared output. Event
         * creation/record failures throw from TransferEngine and intentionally
         * stop inference; there is no stream-synchronization fallback.
         */
        for (ITensor *output : externally_visible_outputs)
        {
            TransferEngine::publishDeviceWrite(
                output,
                ctx->deviceId(),
                gpu_stream);
        }

        return true;
    }

    // =========================================================================
    // Decode Capture Policy
    // =========================================================================

    bool DeviceGraphExecutor::executeDecodeWithCapturePolicy(
        ComputeGraph &graph,
        IDeviceContext *ctx,
        GraphSegmentCache *segment_cache,
        void *gpu_stream,
        IWorkerGPUContext *gpu_ctx,
        const std::unordered_set<std::string> *collective_nodes,
        const DecodeCapturePolicy &policy,
        bool *used_graph_replay,
        GraphInitialSubmissionPolicy initial_submission)
    {
        graph.requireExecutable();
        if(graph.retainedStorageBoundary()&&(!policy.allow_fast_decode||!policy.allow_cached_graph_replay))
            throw std::logic_error("Immutable storage graph requires complete cached native execution");
        if (used_graph_replay)
        {
            *used_graph_replay = false;
        }

        if (!policy.allow_fast_decode)
        {
            return execute(graph, ctx);
        }

        if (policy.allow_cached_graph_replay)
        {
            if (!segment_cache || !gpu_stream || !gpu_ctx)
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Cached graph replay policy requires "
                    "a segment cache, explicit stream, and live GPU context");
                return false;
            }
            bool success = executeWithCachedGraphReplay(
                graph,
                ctx,
                *segment_cache,
                gpu_stream,
                gpu_ctx,
                collective_nodes,
                policy.collectives_graph_capturable,
                policy.force_recapture,
                policy.defer_final_sync,
                policy.capture_boundary,
                policy.graph_replay_plan_policy,
                policy.launch_dependency,
                {},
                policy.retained_parent_composer,
                initial_submission,
                policy.auxiliary_branch_factory);

            if (success)
            {
                if (used_graph_replay)
                {
                    *used_graph_replay = true;
                }
                if (initial_submission ==
                    GraphInitialSubmissionPolicy::MaterializeWithoutLaunch)
                {
                    return true;
                }
                return publishSnapshotsAfterGraphExecution(
                    graph,
                    segment_cache->capture_stream,
                    "decode_graph_replay",
                    &segment_cache->snapshot_manifest);
            }

            const char *mode_name =
                segment_cache->initialized
                    ? DeviceGraphCaptureController::replayModeName(*segment_cache)
                    : "graph_replay";
            LOG_ERROR("[DeviceGraphExecutor] " << mode_name
                                               << " failed under mandatory cached graph replay policy");
            return false;
        }

        if (!bindGraphStagesForEagerExecution(graph, gpu_stream))
            return false;
        if (!executeFastDecode(
                graph,
                ctx,
                collective_nodes))
            return false;
        /*
         * executeFastDecode() publishes eager snapshots at each producer.  A
         * second post-graph pass is both redundant and incorrect for aliased
         * CPU buffers.  Post-graph publication remains mandatory above for a
         * real captured replay because only its recorded D2D copies preserve
         * every intermediate until host materialization.
         */
        return true;
    }

    // =========================================================================
    // Cached GPU Graph Replay
    // =========================================================================

    bool DeviceGraphExecutor::attachRetainedParticipantStorageParent(ComputeGraph& graph,GraphSegmentCache& cache,
        DeviceId receiver,IGPUGraphCapture& parent){
        const auto& domain=graph.retainedParticipantDefinitionDomain();
        const auto& definition=cache.recorded_source_definition;
        if(domain || definition){
            if(!domain || !definition || definition->definition().domain!=*domain || cache.capture_device!=receiver ||
               parent.hasExecutable() || !parent.nodeCount() || reinterpret_cast<std::uintptr_t>(cache.capture_stream)<=2 ||
               parent.executionStream()!=cache.capture_stream)
                throw std::logic_error("Original finite storage attachment has a foreign source/domain/parent/stream");
            definition->requireDefinition();
            if(cache.original_storage_attachment)throw std::logic_error("Original finite storage body already attached to this cache");
            cache.original_storage_attachment=definition->definition().plan->prepareOriginalStorageAttachment(
                receiver,parent,cache.auxiliary_branch.get(),definition,graph);
            if(cache.original_storage_attachment)return cache.original_storage_attachment->attachTo(parent,cache.auxiliary_branch.get());
        }
        return !cache.auxiliary_branch || cache.auxiliary_branch->attach(parent);
    }

    void DeviceGraphExecutor::bindRetainedParticipantPreparationStream(
        ComputeGraph& graph,GraphSegmentCache& cache,IWorkerGPUContext* worker){
        const auto source=graph.retainedParticipantSourcePlan();
        if(!source)return;
        const auto& original=source->originalSource();
        if(!worker)throw std::invalid_argument("Original source preparation is missing its worker");
        const auto original_worker=original.worker();
        if(original_worker.get()!=worker)
            throw std::invalid_argument("Original source preparation worker differs from its admitted source");
        if(!cache.capture_stream || reinterpret_cast<std::uintptr_t>(cache.capture_stream)<=2)
            throw std::invalid_argument("Original source preparation requires its exact non-default cache stream");
        original.requireWorker(original_worker);
        const auto bind_on_original_worker=[&]{
            if(!worker->ownsCurrentThread())
                throw std::logic_error("Original source preparation dispatch did not enter its actual worker");
            // Early input publication authenticates this stream before the cache
            // selects its capture variant. Derive only immutable graph identity,
            // never live request counts or device values, from the actual graph.
            // Initialized replay uses its sealed variant and constant-time
            // provenance checks; it must not rescan stage launch descriptors.
            const auto variant=cache.initialized ? cache.capture_variant_signature :
                DeviceGraphCaptureController::computeCaptureVariantSignature(graph);
            if(!arena_ || !variant || (cache.initialized && !cache.participant_preparation_stream) ||
               (cache.capture_variant_signature && cache.capture_variant_signature!=variant))
                throw std::logic_error("Original source preparation has missing arena or changed selected graph variant");
            if(!cache.participant_preparation_stream)
                cache.participant_preparation_stream.reset(new RetainedGraphParticipantPreparationStream(original,cache.capture_stream,
                    RetainedGraphParticipantPreparationStream::Definition{&graph,arena_,graph.topologyGeneration(),
                        variant,source->definitionDomain(),source}));
            const auto& definition=cache.participant_preparation_stream->definition();
            if(definition.graph!=&graph || definition.arena!=arena_ ||
               definition.topology_generation!=graph.topologyGeneration() || definition.variant!=variant ||
               definition.domain!=source->definitionDomain() || definition.plan!=source ||
               definition.plan.owner_before(source) || source.owner_before(definition.plan))
                throw std::logic_error("Original source preparation changed its immutable graph/arena/topology/variant/plan");
            cache.participant_preparation_stream->requireLive(original);
            if(cache.participant_preparation_stream->stream()!=cache.capture_stream)
                throw std::logic_error("Original source cache replaced its preparation stream without retirement");
            if(cache.initialized)cache.participant_preparation_stream->requireCurrentDefinition();
            else source->authenticatePreparationStream(cache.participant_preparation_stream);
        };
        // Early forward input publication and later capture both use this entry.
        // Preserve the existing exact-worker authority for control-thread callers;
        // ready caches validate their sealed borrow without repeating cold setup.
        if(worker->ownsCurrentThread())bind_on_original_worker();
        else worker->submitAndWait(bind_on_original_worker);
    }

    void DeviceGraphExecutor::authenticateRetainedParticipantRecordingOwner(
        ComputeGraph& graph,GraphSegmentCache& cache,IWorkerGPUContext* worker){
        const auto source=graph.retainedParticipantSourcePlan();
        if(!source)return;
        const auto& original=source->originalSource();
        const auto original_worker=original.worker();
        if(!worker || original_worker.get()!=worker)
            throw std::invalid_argument("Original source recording authentication requires its admitted worker");
        if(!cache.retained_parent_capture || !cache.retained_parent_capture->hasExecutable() ||
           !cache.capture_stream || reinterpret_cast<std::uintptr_t>(cache.capture_stream)<=2 ||
           cache.retained_parent_capture->executionStream()!=cache.capture_stream)
            throw std::invalid_argument("Original source recording authentication requires its exact native cache capture stream");
        original.requireWorker(original_worker);
        const auto authenticate_on_original_worker=[&]{
            if(!worker->ownsCurrentThread())
                throw std::logic_error("Original source recording authentication dispatch did not enter its actual worker");
            source->authenticateRecordingOwner(*cache.retained_parent_capture);
        };
        // Parent sealing is cold control-thread bookkeeping. Authenticate the
        // existing native definition on its original worker without launching,
        // joining GPU work or substituting the parent submission stream.
        if(worker->ownsCurrentThread())authenticate_on_original_worker();
        else worker->submitAndWait(authenticate_on_original_worker);
    }

    bool DeviceGraphExecutor::executeWithCachedGraphReplay(ComputeGraph &graph, IDeviceContext *ctx,
                                                               GraphSegmentCache &segment_cache,
                                                               void *gpu_stream,
                                                               IWorkerGPUContext *gpu_ctx,
                                                               const std::unordered_set<std::string> *collective_nodes,
                                                               bool collectives_graph_capturable,
                                                               bool force_recapture,
                                                               bool defer_final_sync,
                                                               GraphCaptureBoundaryHook capture_boundary,
                                                               GraphReplayPlanPolicy plan_policy,
                                                               GraphLaunchDependencyHook launch_dependency,
                                                               std::span<const BufferId>
                                                                   event_published_outputs,
                                                               RetainedParentCompositionHook
                                                                   retained_parent_composer,
                                                               GraphInitialSubmissionPolicy
                                                                   initial_submission,
                                                               GraphCaptureAuxiliaryBranchFactory
                                                                   auxiliary_branch_factory)
    {
        graph.requireExecutable();
        if (!ctx || !gpu_stream || !gpu_ctx)
        {
            LOG_ERROR("[DeviceGraphExecutor] GPU graph capture/replay requires an exact device context, explicit stream, and live GPU worker context");
            return false;
        }

        if(const auto& bound=segment_cache.participant_source_binding){
            const auto original=graph.retainedParticipantSourcePlan();
            if(!original || original->captureIdentity()!=bound->plan_identity ||
               bound->seal.identity().graph!=&graph ||
               bound->seal.identity().topology_generation!=graph.topologyGeneration())
                throw std::logic_error("Recurring source identity changed after its original transaction was sealed");
        }
        if(graph.retainedParticipantSourcePlan() && force_recapture)
            throw std::invalid_argument("Recurring source topology cannot be recaptured");
        if(graph.retainedParticipantSourcePlan() && !segment_cache.initialized &&
           initial_submission!=GraphInitialSubmissionPolicy::MaterializeWithoutLaunch){
            // Cold construction publishes no arithmetic. First instantiate the
            // exact original parent, then submit transaction zero through the
            // same sealed enqueue path used by every later invocation.
            if(!executeWithCachedGraphReplay(graph,ctx,segment_cache,gpu_stream,gpu_ctx,
                collective_nodes,collectives_graph_capturable,false,defer_final_sync,
                capture_boundary,plan_policy,launch_dependency,event_published_outputs,
                retained_parent_composer,GraphInitialSubmissionPolicy::MaterializeWithoutLaunch,
                auxiliary_branch_factory))return false;
            return executeWithCachedGraphReplay(graph,ctx,segment_cache,gpu_stream,gpu_ctx,
                collective_nodes,collectives_graph_capturable,false,defer_final_sync,
                capture_boundary,plan_policy,launch_dependency,event_published_outputs,
                retained_parent_composer,initial_submission,auxiliary_branch_factory);
        }

        if(segment_cache.original_storage_attachment)segment_cache.original_storage_attachment->requireCurrent();

        const bool has_collective_nodes = (collective_nodes && !collective_nodes->empty());
        const bool retained_parent_requested =
            isRetainedParentPlanPolicy(plan_policy);
        const bool retained_parent_concurrent_ticket_service =
            hasConcurrentTicketService(plan_policy);
        const bool retained_full_graph_requested =
            segment_cache.steady_replay_host_policy ==
            GraphSegmentCache::SteadyReplayHostPolicy::RetainedFullGraph;
        const bool materialize_without_launch =
            initial_submission ==
            GraphInitialSubmissionPolicy::MaterializeWithoutLaunch;

        if(const auto& branch=graph.readAheadBranch())
        {
            if(&branch->resources->workerContext()!=gpu_ctx)
                throw std::logic_error("Read-ahead capture uses a different original worker/context owner");
            graph.planReadAheadRecordingStream(segment_cache.capture_stream);
            if(branch->resources->device()!=ctx->deviceId() || has_collective_nodes ||
                (plan_policy!=GraphReplayPlanPolicy::RequireFullGraph && !isRetainedParentPlanPolicy(plan_policy)) || force_recapture)
                throw std::logic_error("Resident read-ahead requires one unchanged participant-local retained native parent; unsupported segmentation/TP/PP/MTP/recapture is fatal");
        }

        if(const auto &storage=graph.retainedStorageBoundary())
        {
            if(storage->device()!=ctx->deviceId()||plan_policy==GraphReplayPlanPolicy::AllowHeterogeneousBoundarySegmentation)
                throw std::logic_error("Immutable storage requires its exact participant and a complete retained native parent");
            if(!materialize_without_launch)storage->requireArmedForLaunch();
        }

        /*
         * Create the persistent host lane before capture/materialization. It
         * must never be born after an inference parent has already entered the
         * backend submission call, and steady replay must perform no thread
         * creation or task allocation.
         */
        if (retained_parent_concurrent_ticket_service &&
            !prepareRetainedParentTicketServiceWorker())
        {
            return false;
        }

        if (!auxiliary_branch_factory.empty() &&
            !auxiliary_branch_factory.valid())
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Graph auxiliary-branch factory has partial identity");
            return false;
        }
        if (auxiliary_branch_factory.valid() &&
            (auxiliary_branch_factory.device != ctx->deviceId() ||
             (plan_policy != GraphReplayPlanPolicy::RequireFullGraph &&
              !retained_parent_requested)))
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] A graph-owned auxiliary branch requires one final native executable on its exact device"
                << " branch_device="
                << auxiliary_branch_factory.device.toString()
                << " graph_device=" << ctx->deviceId().toString()
                << " plan_policy=" << static_cast<int>(plan_policy));
            return false;
        }

        const bool cache_state_consistent =
            (!segment_cache.initialized &&
             segment_cache.executable_submission_state ==
                 GraphSegmentCache::ExecutableSubmissionState::Empty) ||
            (segment_cache.initialized &&
             segment_cache.executable_submission_state !=
                 GraphSegmentCache::ExecutableSubmissionState::Empty);
        if (!cache_state_consistent)
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Graph cache materialization and executable submission states disagree");
            return false;
        }

        /*
         * Resolve diagnostic topology once per concrete graph/configuration.
         * The executor-wide filter may select the forward graph while adding no
         * node at all to an auxiliary helper. Such helpers retain identity one
         * and therefore keep their already-instantiated native executables.
         */
        const uint64_t graph_snapshot_configuration_identity =
            resolveGraphSnapshotConfigurationIdentity(graph, segment_cache);

        if (retained_parent_requested !=
            static_cast<bool>(retained_parent_composer))
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Retained parent policy and topology composer must be supplied together");
            return false;
        }
        if (retained_parent_requested &&
            (!ctx || !defer_final_sync || retained_full_graph_requested))
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Retained parent replay requires an exact device context, deferred completion ownership, and no competing retained-full-graph policy");
            return false;
        }
        if (materialize_without_launch &&
            (segment_cache.initialized || launch_dependency ||
             !event_published_outputs.empty()))
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Setup-only materialization requires one "
                "pristine cache and cannot publish launch or output "
                "side effects");
            return false;
        }

        /*
         * A snapshot callback requests persistent D2D copy nodes and storage;
         * it is graph topology, not a setup-time publication side effect. The
         * setup capture policy disables host callbacks and the controller does
         * not launch the executable, so the manifest may be sealed here while
         * its first values remain unpublished until transaction zero executes.
         */
        if (segment_cache.initialized &&
            segment_cache.graph_replay_plan_policy != plan_policy)
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Graph replay plan policy changed behind an initialized executable");
            return false;
        }
        if (segment_cache.initialized &&
            (static_cast<bool>(segment_cache.auxiliary_branch) !=
                 auxiliary_branch_factory.valid() ||
             (auxiliary_branch_factory.valid() &&
              (segment_cache.auxiliary_branch_authority !=
                   auxiliary_branch_factory.authority_identity ||
               segment_cache.auxiliary_branch->device() !=
                   auxiliary_branch_factory.device))))
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Graph auxiliary-branch presence or authority changed behind an initialized executable");
            return false;
        }
        if (!segment_cache.initialized &&
            (segment_cache.auxiliary_branch ||
             segment_cache.auxiliary_branch_authority != nullptr))
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Pristine graph cache already owns an auxiliary branch");
            return false;
        }

        const bool retained_full_requires_runtime_collectives = retained_full_graph_requested &&
            (segment_cache.retained_full_graph_replay.valid() &&
             segment_cache.initialized && !segment_cache.needs_capture
                ? segment_cache.runtime_collective_capture_requirement ==
                    GraphSegmentCache::RuntimeCollectiveCaptureRequirement::Required
                : graph.requiresCollectiveRuntimeCapture(collective_nodes));
        if (retained_full_graph_requested &&
            (!ctx || retained_full_requires_runtime_collectives || collectives_graph_capturable ||
             plan_policy != GraphReplayPlanPolicy::RequireFullGraph ||
             !defer_final_sync))
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Retained full-graph host replay requires "
                "one executable without runtime collective capture, an exact device context, and "
                "deferred completion ownership");
            return false;
        }

        auto seal_retained_full_graph_plan = [&]() -> bool
        {
            if (!retained_full_graph_requested)
                return true;
            if (!segment_cache.initialized || segment_cache.needs_capture ||
                segment_cache.segments.size() != 1u ||
                !segment_cache.capture_stream)
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Cannot seal an incomplete retained full-graph replay cache");
                return false;
            }

            auto &segment = segment_cache.segments.front();
            if (!segment.capturable || !segment.capture ||
                !segment.capture->hasExecutable() ||
                segment.capture->nodeCount() == 0u ||
                segment.capture->executionStream() !=
                    segment_cache.capture_stream ||
                !segment.passive_capture_waves_after.empty() ||
                !segment.arena_writes_cached)
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Retained full-graph replay did not materialize one complete executable and publication plan");
                return false;
            }

            const auto &execution_order = graph.getExecutionOrder();
            const auto &execution_stages = graph.getExecutionStages();
            if (execution_order.empty() ||
                segment.stage_names != execution_order ||
                execution_stages.size() != execution_order.size() ||
                std::any_of(
                    execution_stages.begin(),
                    execution_stages.end(),
                    [](const IComputeStage *stage)
                    {
                        return stage == nullptr;
                    }))
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Retained full-graph replay cannot freeze an incomplete or partial stage order");
                return false;
            }

            auto &plan = segment_cache.retained_full_graph_replay;
            plan.clear();
            plan.graph = &graph;
            plan.topology_generation = graph.topologyGeneration();
            plan.snapshot_configuration_epoch =
                segment_cache.snapshot_configuration_epoch;
            plan.capture_variant_signature =
                segment_cache.capture_variant_signature;
            plan.stages.assign(
                execution_stages.begin(), execution_stages.end());
            plan.stage_variant_signatures.reserve(plan.stages.size());
            for (const IComputeStage *stage : plan.stages)
            {
                plan.stage_variant_signatures.push_back(
                    stage->graphCaptureVariantSignature());
            }
            if (!plan.valid())
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Retained full-graph replay produced an invalid sealed identity");
                plan.clear();
                return false;
            }
            return true;
        };

        // Remote participants accompany the original full transaction; an eager
        // first enqueue would escape its typed source/participant ownership scope.
        if((graph.retainedParticipantSourcePlan() || graph.retainedParticipantAuxiliaryRange()) &&
           (!retained_parent_requested ||
            (!segment_cache.initialized && initial_submission!=GraphInitialSubmissionPolicy::MaterializeWithoutLaunch)))
            throw std::invalid_argument("Recurring remote lookup requires its original materialized complete hosted transaction");

        auto seal_retained_parent_plan = [&]() -> bool
        {
            if (!retained_parent_requested)
                return true;
            if (!segment_cache.initialized || segment_cache.needs_capture ||
                !segment_cache.capture_stream ||
                !segment_cache.retained_parent_capture ||
                !segment_cache.retained_parent_capture->hasExecutable() ||
                segment_cache.retained_parent_capture->nodeCount() == 0u ||
                segment_cache.retained_parent_capture->executionStream() !=
                    segment_cache.capture_stream ||
                segment_cache.segments.empty())
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Cannot seal an incomplete retained parent executable");
                return false;
            }

            const auto &execution_order = graph.getExecutionOrder();
            const auto &execution_stages = graph.getExecutionStages();
            if (execution_order.empty() ||
                execution_stages.size() != execution_order.size() ||
                std::any_of(
                    execution_stages.begin(),
                    execution_stages.end(),
                    [](const IComputeStage *stage)
                    {
                        return stage == nullptr;
                    }))
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Retained parent cannot freeze an incomplete source graph");
                return false;
            }

            auto &plan = segment_cache.retained_composed_parent_replay;
            plan.clear();
            plan.graph = &graph;
            plan.topology_generation = graph.topologyGeneration();
            plan.snapshot_configuration_epoch =
                segment_cache.snapshot_configuration_epoch;
            plan.capture_variant_signature =
                segment_cache.capture_variant_signature;
            plan.child_unit_count = static_cast<size_t>(std::count_if(
                segment_cache.segments.begin(),
                segment_cache.segments.end(),
                [](const GraphSegment &segment)
                {
                    return segment.capturable;
                }));
            plan.stages.assign(
                execution_stages.begin(), execution_stages.end());
            plan.stage_variant_signatures.reserve(plan.stages.size());
            for (const IComputeStage *stage : plan.stages)
            {
                plan.stage_variant_signatures.push_back(
                    stage->graphCaptureVariantSignature());
            }

            for (size_t segment_index = 0u;
                 segment_index < segment_cache.segments.size();
                 ++segment_index)
            {
                const auto &segment = segment_cache.segments[segment_index];
                if (!segment.capturable)
                {
                    if (!retained_parent_concurrent_ticket_service ||
                        segment.capture || segment.stage_names.empty())
                    {
                        LOG_ERROR(
                            "[DeviceGraphExecutor] Retained parent contains an uncertified manual source unit before sealing");
                        plan.clear();
                        return false;
                    }
                    for (const std::string &stage_name : segment.stage_names)
                    {
                        const ComputeNode *const node = graph.getNode(stage_name);
                        if (!node || !node->stage ||
                            !node->stage->isManualGraphBoundary() ||
                            node->stage->manualGraphBoundaryScheduling() !=
                                ManualGraphBoundaryScheduling::
                                    ConcurrentTicketService)
                        {
                            LOG_ERROR(
                                "[DeviceGraphExecutor] Retained parent manual source unit changed its concurrent ticket-service contract before sealing: "
                                << stage_name);
                            plan.clear();
                            return false;
                        }
                    }
                    plan.concurrent_ticket_service_segment_indices.push_back(
                        segment_index);
                    continue;
                }
                if (!segment.capture ||
                    segment.capture->hasExecutable() ||
                    segment.capture->nodeCount() == 0u ||
                    segment.capture->executionStream() !=
                        segment_cache.capture_stream ||
                    !segment.arena_writes_cached)
                {
                    LOG_ERROR(
                        "[DeviceGraphExecutor] Retained parent source child or publication plan changed before sealing");
                    plan.clear();
                    return false;
                }
                for (const auto &write : segment.cached_arena_writes)
                {
                    const bool duplicate = std::any_of(
                        plan.arena_writes.begin(),
                        plan.arena_writes.end(),
                        [&](const GraphSegment::ArenaWriteBinding &existing)
                        {
                            return existing.id == write.id &&
                                   existing.device == write.device;
                        });
                    if (!duplicate)
                        plan.arena_writes.push_back(write);
                }
            }
            if (!plan.valid())
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Retained parent produced an invalid sealed identity");
                plan.clear();
                return false;
            }

            if(const auto source_plan=graph.retainedParticipantSourcePlan()){
                authenticateRetainedParticipantRecordingOwner(graph,segment_cache,gpu_ctx);
                RetainedGraphParticipantSourceSeal source_seal(RetainedGraphParticipantSourceIdentity{
                    .graph=&graph,.arena=arena_,.device=ctx->deviceId(),
                    .recording_stream=segment_cache.capture_stream,
                    .topology_generation=graph.topologyGeneration(),
                    .capture_variant_signature=segment_cache.capture_variant_signature,
                    .capture_generation=segment_cache.evidence_family,
                    .original_parent=segment_cache.retained_parent_capture.get()});
                auto invocation=source_plan->bindOriginalSource(source_seal);
                segment_cache.participant_source_binding=std::make_shared<GraphSegmentCache::ParticipantSourceBinding>(
                    GraphSegmentCache::ParticipantSourceBinding{std::move(source_seal),std::move(invocation),source_plan->captureIdentity()});
            }

            PerfStatsCollector::addCounter(
                "forward_graph",
                "retained_parent_executable_nodes",
                static_cast<double>(
                    segment_cache.retained_parent_capture->nodeCount()),
                "decode",
                ctx->deviceId().toString(),
                retainedParentBoundaryTags(
                    segment_cache.perf_context, plan.child_unit_count,
                    plan.concurrent_ticket_service_segment_indices));
            return true;
        };

        auto &retained_parent_plan =
            segment_cache.retained_composed_parent_replay;
        if (retained_parent_plan.valid())
        {
            if (!retained_parent_requested ||
                retained_parent_plan.graph != &graph ||
                retained_parent_plan.topology_generation !=
                    graph.topologyGeneration())
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Retained parent replay rejected a changed policy or stale graph topology generation");
                return false;
            }

            const auto &parent_exec_env = debugEnv().execution;
            if (parent_exec_env.gpu_graph_verify ||
                parent_exec_env.gpu_graph_stream_only)
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Retained parent replay cannot be replaced by child-wise verification or stream-only execution");
                return false;
            }

            bool stage_variant_changed = false;
            for (size_t index = 0;
                 index < retained_parent_plan.stages.size();
                 ++index)
            {
                IComputeStage *const stage =
                    retained_parent_plan.stages[index];
                if (!stage ||
                    stage->graphCaptureVariantSignature() !=
                        retained_parent_plan.stage_variant_signatures[index])
                {
                    stage_variant_changed = true;
                    break;
                }
            }

            const bool diagnostic_recapture =
                force_recapture || parent_exec_env.gpu_graph_recapture;
            const bool snapshot_topology_changed =
                retained_parent_plan.snapshot_configuration_epoch !=
                    graph_snapshot_configuration_identity;
            if(graph.retainedParticipantSourcePlan() && stage_variant_changed)
                throw std::logic_error("Recurring source launch topology changed after sealing");
            if (stage_variant_changed || diagnostic_recapture ||
                snapshot_topology_changed)
            {
                segment_cache.reset(
                    GraphSegmentCache::StreamResetPolicy::Preserve);
                if (stage_variant_changed)
                    ++segment_cache.variant_recapture_count;
            }
            else
            {
                if (!segment_cache.initialized || segment_cache.needs_capture ||
                    !segment_cache.retained_parent_capture ||
                    !segment_cache.retained_parent_capture->hasExecutable() ||
                    segment_cache.retained_parent_capture->executionStream() !=
                        segment_cache.capture_stream ||
                    segment_cache.capture_device != ctx->deviceId() ||
                    segment_cache.gpu_ctx_ref != gpu_ctx ||
                    retained_parent_plan.capture_variant_signature !=
                        segment_cache.capture_variant_signature)
                {
                    LOG_ERROR(
                        "[DeviceGraphExecutor] Retained parent executable or cache identity changed after sealing");
                    return false;
                }

                const auto submission_state =
                    segment_cache.executable_submission_state;
                const bool initial_launch_pending =
                    submission_state ==
                    GraphSegmentCache::ExecutableSubmissionState::
                        MaterializedUnlaunched;
                if (!initial_launch_pending &&
                    submission_state !=
                        GraphSegmentCache::ExecutableSubmissionState::
                            ReplayReady)
                {
                    LOG_ERROR(
                        "[DeviceGraphExecutor] Retained parent has an executable but no valid submission state");
                    return false;
                }

                const auto transition =
                    DeviceGraphCaptureController::beginStep(
                        segment_cache.initialized,
                        segment_cache.needs_capture,
                        segment_cache.decode_step);
                if (transition.phase !=
                    DeviceGraphCaptureController::Phase::Replay)
                {
                    LOG_ERROR(
                        "[DeviceGraphExecutor] Retained parent reached a non-replay phase after sealing");
                    return false;
                }
                if (retained_parent_concurrent_ticket_service)
                {
                    for (const auto &segment : segment_cache.segments)
                    {
                        if (!segment.capturable)
                            continue;
                        if (!DeviceGraphCaptureController::
                                prepareGraphLaunchMetadata(
                                    graph,
                                    segment,
                                    ctx,
                                    segment_cache.capture_stream,
                                    GraphLaunchPreparationPhase::Replay))
                        {
                            LOG_ERROR(
                                "[DeviceGraphExecutor] Retained parent replay metadata preparation failed");
                            return false;
                        }
                    }
                }
                if (launch_dependency &&
                    !launch_dependency(
                        initial_launch_pending
                            ? GraphExecutableLaunchPhase::InitialTransaction
                            : GraphExecutableLaunchPhase::SteadyReplay,
                        segment_cache.capture_stream))
                {
                    LOG_ERROR(
                        "[DeviceGraphExecutor] Retained parent launch dependency failed on the exact execution stream");
                    return false;
                }
                const auto &service_segments =
                    retained_parent_plan
                        .concurrent_ticket_service_segment_indices;
                if (!collectOrderedTimelineTiming(
                        *segment_cache.retained_parent_capture,
                        segment_cache.perf_context,
                        ctx->deviceId().toString()))
                {
                    return false;
                }
                std::optional<RetainedGraphParticipantSubmission> participant_submission;
                if(const auto source_plan=graph.retainedParticipantSourcePlan()){
                    const auto& bound=segment_cache.participant_source_binding;
                    if(!bound || bound->plan_identity!=source_plan->captureIdentity())
                        throw std::logic_error("Original remote source plan changed after transaction sealing");
                    bound->invocation->requireCurrent(bound->seal);
                    participant_submission.emplace(bound->invocation->enqueueOriginalParticipants());
                }
                if(segment_cache.original_storage_attachment)segment_cache.original_storage_attachment->requireCurrent();
                const bool parent_submission_ok =
                    service_segments.empty()
                        ? segment_cache.retained_parent_capture->launch()
                        : submitRetainedParentWithConcurrentTicketService(
                              *segment_cache.retained_parent_capture,
                              graph,
                              segment_cache,
                              ctx,
                              gpu_ctx,
                              transition.decode_step,
                              service_segments);
                if (!parent_submission_ok)
                {
                    LOG_ERROR(
                        "[DeviceGraphExecutor] Retained parent executable launch failed");
                    return false;
                }
                if(participant_submission)participant_submission->completeOriginalEnqueue();
                for (const auto &write : retained_parent_plan.arena_writes)
                {
                    if (!arena_)
                        break;
                    const bool requires_external_event =
                        std::find(
                            event_published_outputs.begin(),
                            event_published_outputs.end(),
                            write.id) != event_published_outputs.end();
                    if (config_.snapshot_callback || requires_external_event)
                    {
                        arena_->markWritten(
                            write.id,
                            write.device,
                            segment_cache.capture_stream);
                    }
                    else
                    {
                        arena_->markWrittenFlagsOnly(
                            write.id, write.device);
                    }
                }
                for (auto &segment : segment_cache.segments)
                    segment.last_executed_step = transition.decode_step;
                segment_cache.executable_submission_state =
                    GraphSegmentCache::ExecutableSubmissionState::ReplayReady;

                if (PerfStatsCollector::isDomainEnabled("forward_graph"))
                {
                    // Mirror the sealed inventory only after the launch and
                    // concurrent service have succeeded. These are physical
                    // programs, not a guessed count of logical MoE layers.
                    auto boundary_tags = retainedParentBoundaryTags(
                        segment_cache.perf_context,
                        retained_parent_plan.child_unit_count,
                        retained_parent_plan.concurrent_ticket_service_segment_indices);
                    boundary_tags.emplace("materialized_during_setup",
                        initial_launch_pending ? "true" : "false");
                    PerfStatsCollector::addCounter(
                        "forward_graph",
                        initial_launch_pending
                            ? "retained_parent_transaction_zero_launches"
                            : "retained_parent_replays",
                        1.0,
                        "decode",
                        ctx->deviceId().toString(),
                        std::move(boundary_tags));
                }
                segment_cache.recordSuccessfulReplay();
                return true;
            }
        }
        if (retained_parent_requested && segment_cache.initialized)
        {
            LOG_ERROR(
                "[DeviceGraphExecutor] Initialized retained-parent cache has no sealed parent replay plan; refusing child replay");
            return false;
        }

        auto &retained_plan =
            segment_cache.retained_full_graph_replay;
        if (retained_plan.valid())
        {
            /*
             * Check the owning graph and its scalar mutation generation before
             * dereferencing any borrowed stage pointer. A changed generation is
             * a stale executable identity, not permission to replay or infer a
             * vaguely compatible topology.
             */
            if (retained_plan.graph != &graph ||
                retained_plan.topology_generation !=
                    graph.topologyGeneration())
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Retained full-graph replay rejected a stale graph topology generation");
                return false;
            }

            bool stage_variant_changed = false;
            for (size_t index = 0; index < retained_plan.stages.size();
                 ++index)
            {
                IComputeStage *const stage = retained_plan.stages[index];
                if (!stage ||
                    stage->graphCaptureVariantSignature() !=
                        retained_plan.stage_variant_signatures[index])
                {
                    stage_variant_changed = true;
                    break;
                }
            }
            if (stage_variant_changed)
            {
                /*
                 * A typed launch-topology variant change intentionally returns
                 * to the capture lifecycle before submitting work. Resetting is
                 * not an eager fallback: no stale executable is launched, and
                 * the same mandatory graph mode is recaptured below.
                 */
                segment_cache.reset(
                    GraphSegmentCache::StreamResetPolicy::Preserve);
                ++segment_cache.variant_recapture_count;
            }
        }

        const auto &exec_env = debugEnv().execution;
        const bool retained_replay_instrumented =
            exec_env.gpu_graph_verify || exec_env.gpu_graph_recapture ||
            exec_env.gpu_graph_stream_only ||
            force_recapture ||
            static_cast<bool>(config_.snapshot_callback) ||
            static_cast<bool>(capture_boundary) ||
            static_cast<bool>(launch_dependency) ||
            !event_published_outputs.empty() ||
            KernelProfiler::isEnabled() ||
            PerfStatsCollector::isDomainEnabled("forward_pass") ||
            PerfStatsCollector::isDomainEnabled("forward_graph") ||
            PerfStatsCollector::gpuStageEventTimingEnabled();

        if (retained_full_graph_requested && retained_plan.valid() &&
            !retained_replay_instrumented)
        {
            if (!segment_cache.initialized || segment_cache.needs_capture ||
                segment_cache.snapshot_configuration_epoch !=
                    graph_snapshot_configuration_identity ||
                retained_plan.snapshot_configuration_epoch !=
                    segment_cache.snapshot_configuration_epoch ||
                retained_plan.capture_variant_signature !=
                    segment_cache.capture_variant_signature ||
                segment_cache.segments.size() != 1u ||
                !segment_cache.capture_stream ||
                segment_cache.capture_stream != gpu_stream ||
                segment_cache.capture_device != ctx->deviceId() ||
                segment_cache.gpu_ctx_ref != gpu_ctx ||
                !segment_cache.replay_gpu_timing_slots.empty())
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Retained full-graph replay identity or stream ownership changed after sealing");
                return false;
            }

            auto &segment = segment_cache.segments.front();
            if (!segment.capturable || !segment.capture ||
                !segment.capture->hasExecutable() ||
                segment.capture->executionStream() !=
                    segment_cache.capture_stream ||
                !segment.arena_writes_cached)
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Retained full-graph executable or publication plan was invalidated after sealing");
                return false;
            }

            bool device_activated_for_preparation = false;
            for (size_t index = 0; index < retained_plan.stages.size();
                 ++index)
            {
                IComputeStage *const stage = retained_plan.stages[index];
                if (!requiresGraphLaunchPreparation(
                        stage->graphLaunchPreparationPolicy(),
                        GraphLaunchPreparationPhase::Replay))
                {
                    continue;
                }

                if (!device_activated_for_preparation)
                {
                    ctx->activateDevice();
                    device_activated_for_preparation = true;
                }
                stage->setGPUStream(graph.executionStreamForNode(graph.getExecutionOrder()[index],segment_cache.capture_stream));
                if (!stage->prepareGraphLaunch(
                        ctx, stage->gpuStream()))
                {
                    LOG_ERROR(
                        "[DeviceGraphExecutor] Retained full-graph replay metadata preparation failed");
                    return false;
                }
                if (stage->graphCaptureVariantSignature() !=
                    retained_plan.stage_variant_signatures[index])
                {
                    LOG_ERROR(
                        "[DeviceGraphExecutor] Retained full-graph stage changed its launch topology during replay preparation");
                    return false;
                }
            }

            const auto transition = DeviceGraphCaptureController::beginStep(
                segment_cache.initialized,
                segment_cache.needs_capture,
                segment_cache.decode_step);
            if (transition.phase !=
                DeviceGraphCaptureController::Phase::Replay)
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Retained full-graph plan reached a non-replay phase after sealing");
                return false;
            }
            if (!segment.capture->launch())
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] Retained full-graph executable launch failed");
                return false;
            }

            /*
             * The executable owns every write listed by the cold publication
             * plan. No host observer exists in this policy, so authority can be
             * advanced with flags only; the caller's explicit completion event
             * remains the sole host boundary for the transaction.
             */
            if (arena_)
            {
                for (const auto &write : segment.cached_arena_writes)
                    arena_->markWrittenFlagsOnly(write.id, write.device);
            }
            segment.last_executed_step = transition.decode_step;
            segment_cache.recordSuccessfulReplay();
            return true;
        }

        /*
         * Snapshot diagnostics change graph topology: every selected GPU stage
         * gains a D2D copy node into graph-owned mapped storage. A graph warmed
         * or captured under another callback/filter configuration cannot be
         * reused merely because its compute-stage launch signature is unchanged.
         *
         * Re-enter Phase 1 explicitly. This is the ordinary capture lifecycle,
         * not eager recovery after a failed graph launch: the old executable is
         * retired before any work is submitted under the new topology. One
         * initialization transaction warms the new manifest and materializes its
         * replacement executable before returning to the caller.
         */
        if (segment_cache.snapshot_configuration_epoch !=
            graph_snapshot_configuration_identity)
        {
            const bool pristine_identity_adopted =
                segment_cache.adoptSnapshotConfigurationEpochIfPristine(
                    graph_snapshot_configuration_identity);
            if (!pristine_identity_adopted)
            {
                if(graph.readAheadBranch())throw std::logic_error("Read-ahead snapshot topology changed after materialization");
                /*
                 * A non-pristine cache may own native graph resources or
                 * snapshot destinations.  Retire those resources behind their
                 * exact completion event before rebuilding the changed topology.
                 */
                segment_cache.reset(
                    GraphSegmentCache::StreamResetPolicy::Preserve);
                segment_cache.snapshot_manifest.clear();
                segment_cache.snapshot_configuration_epoch =
                    graph_snapshot_configuration_identity;
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "decode_snapshot_configuration_rewarm",
                    1.0,
                    "decode",
                    ctx ? ctx->deviceId().toString() : std::string{},
                    {{"snapshot_epoch",
                      std::to_string(
                          graph_snapshot_configuration_identity)},
                     {"context", segment_cache.perf_context}});
            }
        }

        const uint64_t current_variant_signature =
            DeviceGraphCaptureController::computeCaptureVariantSignature(graph);
        if (segment_cache.initialized &&
            segment_cache.capture_variant_signature != current_variant_signature)
        {
            if(graph.readAheadBranch())throw std::logic_error("Read-ahead capture identity changed; stale native replay is fatal");
            LOG_DEBUG("[DeviceGraphExecutor] GPU graph launch-topology variant changed from "
                      << segment_cache.capture_variant_signature << " to "
                      << current_variant_signature << "; recapturing");
            PerfStatsCollector::addCounter(
                "forward_graph",
                "decode_graph_variant_recapture",
                1.0,
                "decode",
                ctx ? ctx->deviceId().toString() : std::string{},
                {{"old_signature", std::to_string(segment_cache.capture_variant_signature)},
                 {"new_signature", std::to_string(current_variant_signature)},
                 {"context", segment_cache.perf_context}});
            segment_cache.reset(GraphSegmentCache::StreamResetPolicy::Preserve);
            segment_cache.capture_variant_signature = current_variant_signature;
            segment_cache.variant_recapture_count++;
        }
        else if (!segment_cache.initialized)
        {
            segment_cache.capture_variant_signature = current_variant_signature;
        }

        // Monotonic step counter + phase transition selection for segmented mode.
        const auto phase_transition = DeviceGraphCaptureController::beginStep(
            segment_cache.initialized,
            segment_cache.needs_capture,
            segment_cache.decode_step);
        const uint64_t current_step = phase_transition.decode_step;
        /*
         * Tags are maps of owning strings. Constructing them before the
         * collector rejects a disabled domain turns observability into a
         * per-layer allocation tax, which is especially visible for one small
         * retained graph on each sparse endpoint. Keep the producer-side gate
         * outside argument evaluation so ordinary replay allocates nothing.
         */
        if (PerfStatsCollector::isDomainEnabled("forward_graph"))
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                "decode_graph_phase",
                1.0,
                "decode",
                ctx ? ctx->deviceId().toString() : std::string{},
                {{"context", segment_cache.perf_context},
                 {"phase", DeviceGraphCaptureController::phaseName(
                               phase_transition.phase)}});
        }

        auto mark_arena_write_dirty = [&](BufferId id, DeviceId device)
        {
            if (!arena_)
                return;

            /*
             * Steady-state graph replay normally stays fully device-owned, so a
             * flags-only transition is enough and avoids an event per stage.
             * Parity snapshot callbacks are different: they immediately read
             * intermediate outputs on the host after replay.  Record the same
             * stream event used by the non-captured execution path so
             * ensureOnHost() waits for the captured kernels that produced the
             * tensor instead of racing a stale host copy.
             */
            const bool requires_external_event =
                std::find(
                    event_published_outputs.begin(),
                    event_published_outputs.end(),
                    id) != event_published_outputs.end();
            if ((config_.snapshot_callback || requires_external_event) &&
                segment_cache.capture_stream)
                arena_->markWritten(id, device, segment_cache.capture_stream);
            else
                arena_->markWrittenFlagsOnly(id, device);
        };

        // One source-derived boundary selects both policy and callback owner.
        const auto snapshot_boundary_for_node = [&](const ComputeNode& node)
        {
            return graph.readAheadBranch() && node.name==graph.readAheadBranch()->producer_node
                ? SnapshotPublicationBoundary::NativeEventClosedProducer
                : SnapshotPublicationBoundary::ControllerStage;
        };

        auto record_graph_snapshot_copies = [&](ComputeNode &node, void *producer_stream) -> bool
        {
            // A side-lane producer must publish its immutable copy before the
            // stage completion records join. runStage owns that exact copy;
            // the controller must neither duplicate it nor use the main lane.
            if (snapshotRecordingAuthorityForBoundary(
                    StageSnapshotRecordingAuthority::GraphCaptureController,
                    snapshot_boundary_for_node(node))!=StageSnapshotRecordingAuthority::GraphCaptureController ||
                !config_.snapshot_callback || !node.stage)
                return true;

            DeviceId snapshot_device = node.device.is_valid() ? node.device : node.stage->device();
            if (!snapshot_device.is_valid() && ctx)
                snapshot_device = ctx->deviceId();

            void *stream = producer_stream ? producer_stream : node.stage->gpuStream();
            return captureGraphSnapshotCopies(
                node,
                snapshot_device,
                stream,
                segment_cache.snapshot_manifest);
        };

        auto prepare_graph_snapshot_manifest = [&]() -> bool
        {
            return prepareSnapshotsForGraphCapture(
                graph,
                ctx,
                segment_cache.capture_stream,
                "cached_decode_graph_capture",
                &segment_cache.snapshot_manifest);
        };

        const auto is_collective_node = [&](const ComputeNode &node) -> bool
        {
            if (collective_nodes &&
                collective_nodes->find(node.name) != collective_nodes->end())
            {
                return true;
            }
            return node.stage && node.stage->isCollectiveStage();
        };

        // Only the side producer changes ownership: its copy must be inside
        // the exact runStage completion scope. Later manual/ticket stages retain
        // their original controller owner and original manifest.
        const auto snapshot_policy_for_node = [&](const ComputeNode& node,StageRunPolicy policy)
        {
            policy.snapshot_recording_authority=snapshotRecordingAuthorityForBoundary(
                policy.snapshot_recording_authority,
                snapshot_boundary_for_node(node));
            return policy;
        };

        StageRunPolicy manual_replay_policy =
            StageRunPolicy::capturePhase();
        manual_replay_policy.snapshot_recording_authority =
            StageRunPolicy::SnapshotRecordingAuthority::GraphCaptureController;

        // ===== FAST PATH: Phase 3 (Replay) =====
        // During steady-state replay, only the post_launch hook is invoked
        // (cohere_inputs is skipped for capturable segments, execute_node is
        // unused). Avoid constructing unused lambdas and skip phase 1/2 checks
        // to minimize host overhead on the hot decode path.
        const bool replay_diagnostics_enabled =
            exec_env.gpu_graph_verify || exec_env.gpu_graph_recapture || force_recapture;
        const bool materialized_transaction_zero_pending =
            segment_cache.executable_submission_state ==
            GraphSegmentCache::ExecutableSubmissionState::
                MaterializedUnlaunched;
        if (phase_transition.phase == DeviceGraphCaptureController::Phase::Replay &&
            !replay_diagnostics_enabled &&
            !materialized_transaction_zero_pending)
        {
            DeviceGraphCaptureController::prepareDeviceForGraphCapture(ctx);

            const bool initial_launch_pending =
                segment_cache.executable_submission_state ==
                GraphSegmentCache::ExecutableSubmissionState::
                    MaterializedUnlaunched;
            GraphLaunchDependencyHook fast_launch_dependency =
                launch_dependency;
            if (initial_launch_pending && launch_dependency)
            {
                fast_launch_dependency =
                    [dependency = std::move(launch_dependency)](
                        GraphExecutableLaunchPhase,
                        void *stream)
                {
                    return dependency(
                        GraphExecutableLaunchPhase::InitialTransaction,
                        stream);
                };
            }

            DeviceGraphCaptureController::ReplayHooks fast_hooks{
                .cohere_inputs = nullptr,
                .execute_node = [&](ComputeNode &node) -> bool
                {
                    return runStage(
                        node,
                        ctx,
                        snapshot_policy_for_node(node,manual_replay_policy),
                        is_collective_node(node),
                        &segment_cache.snapshot_manifest);
                },
                .prepare_snapshot_manifest = nullptr,
                .record_snapshot_copies = record_graph_snapshot_copies,
                .post_launch = [&](DeviceGraphExecutor::GraphSegment &seg, void *stream)
                {
                    DeviceGraphCaptureController::postCapturedSegmentLaunch(
                        graph, seg, current_step, stream,
                        [&](BufferId id, DeviceId device)
                        {
                            mark_arena_write_dirty(id, device);
                        });
                },
                .capture_boundary = nullptr,
                .launch_dependency = std::move(fast_launch_dependency),
                .retained_parent_composer = retained_parent_composer,
                .submit_parent_with_concurrent_ticket_service =
                    [&](IGPUGraphCapture &parent,
                        std::span<const size_t> segments,
                        uint64_t step)
                    {
                        return submitRetainedParentWithConcurrentTicketService(
                            parent,
                            graph,
                            segment_cache,
                            ctx,
                            gpu_ctx,
                            step,
                            segments);
                    },
                .auxiliary_branch = segment_cache.auxiliary_branch.get(),
            };

            const auto replay_result = DeviceGraphCaptureController::executeReplayPhase(
                graph, segment_cache, ctx, gpu_ctx,
                has_collective_nodes, collectives_graph_capturable, current_step, fast_hooks,
                /*force_recapture=*/false,
                defer_final_sync);

            if (!replay_result.success)
            {
                return false;
            }

            if (initial_launch_pending)
            {
                segment_cache.executable_submission_state =
                    GraphSegmentCache::ExecutableSubmissionState::ReplayReady;
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "materialized_graph_transaction_zero_launches",
                    1.0,
                    "decode",
                    ctx ? ctx->deviceId().toString() : std::string{},
                    {{"context", segment_cache.perf_context},
                     {"executable_family", std::to_string(segment_cache.evidence_family)},
                     {"segments",
                      std::to_string(segment_cache.segments.size())}});
            }

            segment_cache.recordSuccessfulReplay();
            return true;
        }

        // ===== FIRST-CAPTURE PATH =====
        auto post_captured_segment_launch = [&](GraphSegment &seg, void *stream)
        {
            DeviceGraphCaptureController::postCapturedSegmentLaunch(
                graph,
                seg,
                current_step,
                stream,
                [&](BufferId id, DeviceId device)
                {
                    mark_arena_write_dirty(id, device);
                });
        };

        auto cohere_replay_stage = [&](
                                       ComputeNode &node,
                                       const std::vector<BufferBinding> &external_reads,
                                       std::unordered_set<ITensor *> &prepared_inputs,
                                       std::unordered_set<ITensor *> &prepared_outputs) -> bool
        {
            DeviceId target_device = node.device.is_valid() ? node.device : node.stage->device();
            const StageBufferContract contract = node.stage->bufferContract();
            void *const coherence_stream = segment_cache.capture_stream;
            if (!prepareStageArenaFrontierForCapture(
                    node,
                    external_reads,
                    ctx->deviceId(),
                    coherence_stream,
                    prepared_inputs,
                    prepared_outputs,
                    materialize_without_launch
                        ? GraphCaptureDependencyLedger::ExternalInputAuthority::
                              BindDeclaredAddressesOnly
                        : GraphCaptureDependencyLedger::ExternalInputAuthority::
                              RequireReadyBytes,
                    "cached_graph_segment"))
            {
                return false;
            }

            std::string prepared_error;
            if (!node.stage->validatePreparedWeights(&prepared_error))
            {
                LOG_ERROR("[DeviceGraphExecutor] Prepared weight validation failed for replay stage '"
                          << node.name << "': " << prepared_error);
                return false;
            }
            if (!validatePreparedWeightBindings(node, contract, target_device))
                return false;

            // Validate exact weight residency and join producer ordering.
            // Graph capture is never an allocation or lazy-upload boundary.
            if (!node.weights_cohered)
            {
                for (auto *weight : contract.weight_tensors)
                {
                    if (auto *tb = dynamic_cast<TensorBase *>(weight))
                    {
                        TransferEngine::requireDeviceInput(
                            tb, target_device, coherence_stream);
                    }
                }
                node.weights_cohered = true;
            }

            return true;
        };

        auto cohere_segment_inputs = [&](const GraphSegment &seg) -> bool
        {
            std::unordered_set<ITensor *> prepared_inputs;
            std::unordered_set<ITensor *> prepared_outputs;
            return DeviceGraphCaptureController::cohereReplaySegmentInputs(
                graph,
                seg,
                [&](ComputeNode &node,
                    const std::vector<BufferBinding> &external_reads)
                {
                    std::vector<BufferBinding> strict_external_reads;
                    strict_external_reads.reserve(external_reads.size());
                    for (const auto &binding : external_reads)
                    {
                        if (!std::binary_search(
                                seg.retained_parent_input_ids.begin(),
                                seg.retained_parent_input_ids.end(),
                                binding.id))
                        {
                            strict_external_reads.push_back(binding);
                        }
                    }

                    /*
                     * Retained-parent inputs already have device storage from
                     * prebinding, but no completed global generation exists
                     * while child templates are recorded. Only true frontier
                     * inputs may join coherence events before beginCapture().
                     */
                    return cohere_replay_stage(
                        node,
                        strict_external_reads,
                        prepared_inputs,
                        prepared_outputs);
                });
        };

        auto prebind_segment_storage = [&](const GraphSegment &seg) -> bool
        {
            std::unordered_set<ITensor *> allocated;
            for (const auto &stage_name : seg.stage_names)
            {
                ComputeNode *node = graph.getNode(stage_name);
                if (!node || !node->stage)
                    continue;
                if (!allocateStageArenaStorageForCapture(
                        *node,
                        ctx->deviceId(),
                        allocated,
                        "cached_graph_segment_prebind"))
                {
                    return false;
                }
            }
            return true;
        };

        auto plan_capture_dependencies = [&](const GraphSegment &seg)
            -> std::unique_ptr<GraphCaptureDependencyLedger>
        {
            return planGraphCaptureDependencies(
                graph,
                std::span<const std::string>(
                    seg.stage_names.data(), seg.stage_names.size()),
                ctx->deviceId(),
                segment_cache.capture_stream,
                "cached_graph_segment",
                std::span<const BufferId>(
                    seg.retained_parent_input_ids.data(),
                    seg.retained_parent_input_ids.size()),
                materialize_without_launch
                    ? GraphCaptureDependencyLedger::ExternalInputAuthority::
                          BindDeclaredAddressesOnly
                    : GraphCaptureDependencyLedger::ExternalInputAuthority::
                          RequireReadyBytes);
        };

        // The finite body retains the same declared nodes, roots and setup
        // policy as the model graph. Only native scheduling changes: ordinary
        // model units omit these nodes, and the original parent attaches their
        // one graph-only recording as a sibling before instantiation.
        std::optional<GraphSegment> participant_auxiliary_segment;
        if(const auto& range=graph.retainedParticipantAuxiliaryRange()) {
            graph.requireRetainedParticipantAuxiliaryRange();
            participant_auxiliary_segment.emplace();
            auto& body=*participant_auxiliary_segment;
            body.capturable=true;body.stage_names=range->nodes;
            std::unordered_set<BufferId> main_writes,body_writes,parent_inputs;
            for(const auto& name:graph.getExecutionOrder()) {
                if(graph.isRetainedParticipantAuxiliaryNode(name))continue;
                const auto* node=graph.getNode(name);
                const auto contract=node->stage->bufferContract();
                for(const auto& write:contract.allWrites())main_writes.insert(write.id);
            }
            for(const auto& name:body.stage_names) {
                const auto contract=graph.getNode(name)->stage->bufferContract();
                for(const auto& read:contract.allArenaReads())
                    if(main_writes.contains(read.id) && !body_writes.contains(read.id))parent_inputs.insert(read.id);
                for(const auto& write:contract.allWrites())body_writes.insert(write.id);
            }
            body.retained_parent_input_ids.assign(parent_inputs.begin(),parent_inputs.end());
            std::sort(body.retained_parent_input_ids.begin(),body.retained_parent_input_ids.end());
        }

        DeviceGraphCaptureController::prepareDeviceForGraphCapture(ctx);

        StageRunPolicy capture_phase_policy =
            materialize_without_launch
                ? StageRunPolicy::setupGraphMaterialization()
                : StageRunPolicy::capturePhase();
        capture_phase_policy.snapshot_recording_authority =
            StageRunPolicy::SnapshotRecordingAuthority::GraphCaptureController;

        DeviceGraphCaptureController::ReplayHooks replay_hooks{
            .prebind_storage = prebind_segment_storage,
            .cohere_inputs = [&](const GraphSegment &segment)
            {
                return cohere_segment_inputs(segment);
            },
            .execute_node = [&](ComputeNode &node)
            {
                /*
                 * Captured and manual replay units share one stage policy and
                 * one cache-owned snapshot manifest. A manual collective runs
                 * outside native stream capture, but it is still inside this
                 * graph transaction; sending it through executeNode() would
                 * incorrectly select the executor's transient eager manifest.
                 * The controller records the one snapshot copy after it has
                 * closed the collective-to-capture stream edge.
                 */
                return runStage(
                    node,
                    ctx,
                    snapshot_policy_for_node(node,capture_phase_policy),
                    is_collective_node(node),
                    &segment_cache.snapshot_manifest);
            },
            .prepare_snapshot_manifest = nullptr,
            .record_snapshot_copies = record_graph_snapshot_copies,
            .post_launch = [&](GraphSegment &segment, void *stream)
            {
                post_captured_segment_launch(segment, stream);
            },
            .capture_boundary = capture_boundary,
            .launch_dependency = launch_dependency,
            .retained_parent_composer = retained_parent_composer,
            .submit_parent_with_concurrent_ticket_service =
                [&](IGPUGraphCapture &parent,
                    std::span<const size_t> segments,
                    uint64_t step)
                {
                    return submitRetainedParentWithConcurrentTicketService(
                        parent,
                        graph,
                        segment_cache,
                        ctx,
                        gpu_ctx,
                        step,
                        segments);
                },
            .auxiliary_branch = segment_cache.auxiliary_branch.get(),
            .plan_capture_dependencies = plan_capture_dependencies,
        };

        const auto recorded_source_boundary = [&](const std::string& name,void* stream){
            const auto& domain=graph.retainedParticipantDefinitionDomain();
            if(!domain || name.find(":phase=capture_end:")==std::string::npos)
                return capture_boundary?capture_boundary(name,stream):true;
            if(!auxiliary_branch_factory.source_definition_boundary)
                throw std::logic_error("Graph-declared source definition has no original cold TP boundary");
            LocalTPCaptureSourceDefinition handoff{*domain,
                graph.retainedParticipantSourcePlan()?segment_cache.participant_preparation_stream:nullptr,{}};
            if(!auxiliary_branch_factory.source_definition_boundary(name,stream,handoff))return false;
            if(!handoff.result || handoff.result->definition().domain!=*domain)
                throw std::logic_error("Original recorded TP boundary returned a foreign source definition");
            handoff.result->requireDefinition();
            if(segment_cache.recorded_source_definition && segment_cache.recorded_source_definition!=handoff.result)
                throw std::logic_error("One native parent crossed original source definition owners");
            segment_cache.recorded_source_definition=std::move(handoff.result);
            if(segment_cache.auxiliary_branch)
                segment_cache.auxiliary_branch->bindOriginalSourceDefinition(segment_cache.recorded_source_definition);
            return true;
        };

        const auto attach_original_parent = [&](IGPUGraphCapture& parent){
            if(!participant_auxiliary_segment)
                return attachRetainedParticipantStorageParent(graph,segment_cache,ctx->deviceId(),parent);
            const auto& range=*graph.retainedParticipantAuxiliaryRange();
            const auto& definition=segment_cache.recorded_source_definition;
            const auto& domain=graph.retainedParticipantDefinitionDomain();
            if(!gpu_ctx || !gpu_ctx->ownsCurrentThread() || !domain || !definition ||
               definition->definition().domain!=*domain || segment_cache.capture_device!=range.participant ||
               ctx->deviceId()!=range.participant || parent.hasExecutable() || !parent.nodeCount() ||
               parent.executionStream()!=segment_cache.capture_stream ||
               segment_cache.original_storage_attachment || segment_cache.participant_auxiliary_preparation)
                throw std::logic_error("Finite participant body requires its actual untouched parent/source/worker");
            definition->requireCurrentDefinition();
            const auto plan=definition->definition().plan;
            const auto original=GPUDeviceContextPool::instance().retainExistingPreparedContext(
                range.participant,definition->originalSource().memory());
            if(original.worker().get()!=gpu_ctx)
                throw std::logic_error("Finite participant body substituted the original prepared worker");
            original.requireWorker(original.worker());
            const IGPUGraphCapture* producer=nullptr;
            for(const auto& unit:segment_cache.segments) {
                if(std::find(unit.stage_names.begin(),unit.stage_names.end(),range.probe_ready)==unit.stage_names.end())
                    continue;
                if(producer || !unit.capturable || !unit.capture ||
                   unit.stage_names.back()!=range.probe_ready || !unit.capture->nodeCount() ||
                   unit.capture->hasExecutable() || unit.capture->executionStream()!=segment_cache.capture_stream)
                    throw std::logic_error("Finite participant body requires its exact graph-only probe frontier");
                producer=unit.capture.get();
            }
            if(!producer)
                throw std::logic_error("Finite participant body has no retained current-probe producer");
            segment_cache.participant_auxiliary_preparation.reset(new RetainedGraphParticipantPreparationStream(
                original,segment_cache.capture_stream,RetainedGraphParticipantPreparationStream::Definition{
                    .graph=&graph,.arena=arena_,.topology_generation=graph.topologyGeneration(),
                    .variant=segment_cache.capture_variant_signature,.domain=*domain,.plan=plan,
                    .scope=range.role==RetainedGraphParticipantAuxiliaryRole::Source
                        ?RetainedGraphParticipantDefinitionScope::AuxiliarySource:RetainedGraphParticipantDefinitionScope::AuxiliaryReader,
                    .participant=range.participant,.range_identity=range.identity,.original_source_definition=definition}));
            segment_cache.participant_auxiliary_preparation->requireDefinition();

            auto& body=*participant_auxiliary_segment;
            if(!cohere_segment_inputs(body))return false;
            auto ledger=plan_capture_dependencies(body);
            if(!ledger)return false;
            // This is the existing cloneable graph-only child creator. There is
            // no executable, new submission lane or separate General reservation.
            std::shared_ptr<IGPUGraphCapture> native=gpu_ctx->createGraphCapture(segment_cache.capture_stream);
            if(!native || native->hasExecutable() || native->nodeCount() ||
               native->executionStream()!=segment_cache.capture_stream)
                throw std::logic_error("Finite participant body lost its original record-only capture");
            std::vector<IComputeStage*> stages;
            for(const auto& name:body.stage_names)stages.push_back(graph.getNode(name)->stage.get());
            ScopedGraphCaptureStageActivity activity(stages,ctx,segment_cache.capture_stream,"participant_auxiliary_body");
            ScopedBackendGraphCapture recording(*gpu_ctx,*native,"participant_auxiliary_body",ledger.get());
            if(!activity.begin())return false;
            if(!recording.begin()) {
                (void)activity.finish(GraphCaptureActivityTransition::Aborted);return false;
            }
            bool recorded=true;
            for(const auto& name:body.stage_names) {
                auto* node=graph.getNode(name);
                if(!runStage(*node,ctx,snapshot_policy_for_node(*node,capture_phase_policy),
                    is_collective_node(*node),&segment_cache.snapshot_manifest) ||
                   !record_graph_snapshot_copies(*node,segment_cache.capture_stream)) {
                    recorded=false;break;
                }
                graph.markCompleted(name);
            }
            recording.finish();
            const bool complete=recorded && native->nodeCount() && !native->hasExecutable();
            if(!activity.finish(complete?GraphCaptureActivityTransition::Completed:GraphCaptureActivityTransition::Aborted) || !complete)
                return false;
            RetainedGraphParticipantAuxiliaryBodySeal seal({
                .graph=&graph,.arena=arena_,.participant=range.participant,
                .enclosing_recording_stream=segment_cache.capture_stream,.enclosing_parent=&parent,
                .producer_fragment=producer,
                .topology_generation=graph.topologyGeneration(),
                .capture_variant_signature=segment_cache.capture_variant_signature,
                .capture_generation=segment_cache.evidence_family,.range_identity=range.identity},
                original,std::move(native),segment_cache.participant_auxiliary_preparation);
            segment_cache.original_storage_attachment=plan->prepareOriginalAuxiliaryAttachment(seal,segment_cache.auxiliary_branch.get());
            if(!segment_cache.original_storage_attachment)
                throw std::logic_error("Finite participant body was not retained by its original parent attachment");
            const bool attached=segment_cache.original_storage_attachment->attachTo(parent,segment_cache.auxiliary_branch.get());
            if(attached)PerfStatsCollector::addCounter("forward_graph","participant_auxiliary_body_nodes",
                static_cast<double>(seal.recordedBody()->nodeCount()),"graph_setup",range.participant.toString(),
                {{"range_identity",std::to_string(range.identity)},{"stages",std::to_string(range.nodes.size())},
                 {"standalone_executables","0"}});
            return attached;
        };

        // Capture and replay share the same post-launch lifecycle. All mutable
        // GPU state is published by captured kernels and explicit event edges.
        DeviceGraphCaptureController::ReplayHooks capture_hooks{
            .prebind_storage = replay_hooks.prebind_storage,
            .cohere_inputs = replay_hooks.cohere_inputs,
            .execute_node = [&](ComputeNode &node)
            {
                return runStage(
                    node,
                    ctx,
                    snapshot_policy_for_node(node,capture_phase_policy),
                    is_collective_node(node),
                    &segment_cache.snapshot_manifest);
            },
            .prepare_snapshot_manifest = [&] {
                if(participant_auxiliary_segment) {
                    auto& body=*participant_auxiliary_segment;
                    if(!prebind_segment_storage(body))return false;
                    for(const auto& name:body.stage_names)
                        graph.getNode(name)->stage->setGPUStream(segment_cache.capture_stream);
                    if(!DeviceGraphCaptureController::prepareGraphLaunchMetadata(graph,body,ctx,
                        segment_cache.capture_stream,GraphLaunchPreparationPhase::Capture))return false;
                    for(const auto& name:body.stage_names)
                        if(!graph.getNode(name)->stage->isGraphCapturable())
                            throw std::logic_error("Finite participant body stage failed capture preparation: "+name);
                }
                return prepare_graph_snapshot_manifest();
            },
            .record_snapshot_copies = record_graph_snapshot_copies,
            .post_launch = [&](GraphSegment &segment, void *stream)
            {
                DeviceGraphCaptureController::postCapturedSegmentLaunch(
                    graph,
                    segment,
                    current_step,
                    stream,
                    [&](BufferId id, DeviceId device)
                    {
                        mark_arena_write_dirty(id, device);
                    });
            },
            .capture_boundary = phase_transition.phase == DeviceGraphCaptureController::Phase::Capture
                ? GraphCaptureBoundaryHook{recorded_source_boundary} : GraphCaptureBoundaryHook{},
            .launch_dependency = phase_transition.phase == DeviceGraphCaptureController::Phase::Capture && graph.retainedParticipantDefinitionDomain()
                ? GraphLaunchDependencyHook{[&](GraphExecutableLaunchPhase phase,void* stream){
                    if(segment_cache.original_storage_attachment)segment_cache.original_storage_attachment->requireCurrent();
                    return !launch_dependency || launch_dependency(phase,stream);
                }} : launch_dependency,
            .retained_parent_composer = retained_parent_composer,
            .submit_parent_with_concurrent_ticket_service =
                [&](IGPUGraphCapture &parent,
                    std::span<const size_t> segments,
                    uint64_t step)
                {
                    return submitRetainedParentWithConcurrentTicketService(
                        parent,
                        graph,
                        segment_cache,
                        ctx,
                        gpu_ctx,
                        step,
                        segments);
                },
            .auxiliary_branch = segment_cache.auxiliary_branch.get(),
            .attach_original_parent = phase_transition.phase == DeviceGraphCaptureController::Phase::Capture
                ? std::function<bool(IGPUGraphCapture&)>{attach_original_parent} : std::function<bool(IGPUGraphCapture&)>{},
            .plan_capture_dependencies = plan_capture_dependencies,
        };

        if (phase_transition.phase == DeviceGraphCaptureController::Phase::Replay)
        {
            const bool initial_launch_pending =
                segment_cache.executable_submission_state ==
                GraphSegmentCache::ExecutableSubmissionState::
                    MaterializedUnlaunched;
            GraphLaunchDependencyHook replay_launch_dependency =
                launch_dependency;
            if (initial_launch_pending && launch_dependency)
            {
                replay_launch_dependency =
                    [dependency = std::move(launch_dependency)](
                        GraphExecutableLaunchPhase,
                        void *stream)
                {
                    return dependency(
                        GraphExecutableLaunchPhase::InitialTransaction,
                        stream);
                };
            }
            DeviceGraphCaptureController::ReplayHooks replay_phase_hooks =
                replay_hooks;
            replay_phase_hooks.launch_dependency =
                std::move(replay_launch_dependency);
            replay_phase_hooks.require_replay_input_preflight =
                initial_launch_pending;
            const auto replay_result = DeviceGraphCaptureController::executeReplayPhase(
                graph, segment_cache, ctx, gpu_ctx,
                has_collective_nodes, collectives_graph_capturable, current_step, replay_phase_hooks, force_recapture,
                defer_final_sync);

            if (!replay_result.success)
            {
                return false;
            }

            if (initial_launch_pending)
            {
                segment_cache.executable_submission_state =
                    GraphSegmentCache::ExecutableSubmissionState::ReplayReady;
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "materialized_graph_transaction_zero_launches",
                    1.0,
                    "decode",
                    ctx ? ctx->deviceId().toString() : std::string{},
                    {{"context", segment_cache.perf_context},
                     {"executable_family", std::to_string(segment_cache.evidence_family)},
                     {"segments",
                      std::to_string(segment_cache.segments.size())}});
            }
            segment_cache.recordSuccessfulReplay();
            return true;
        }

        // ===== First use: prepare, capture, instantiate, and launch =====
        // Launch preparation binds immutable topology only. No stage arithmetic
        // runs outside the graph, so transaction zero is the first launch of the
        // captured executable and follows the exact same path as every replay.
        if (phase_transition.phase == DeviceGraphCaptureController::Phase::Capture)
        {
            DeviceGraphCaptureController::buildCapturePlan(
                graph,
                segment_cache,
                collective_nodes,
                has_collective_nodes,
                collectives_graph_capturable,
                plan_policy);

            // The graph and every preparation operation own this exact stream.
            if (!segment_cache.ensureCaptureStream(
                    gpu_ctx,
                    ctx ? ctx->deviceId() : DeviceId::invalid(),
                    config_.worker_gpu_context_resolver
                        ? config_.worker_gpu_context_uses_process_pool
                        : true))
            {
                LOG_ERROR("[DeviceGraphExecutor] GPU graph capture could not establish its explicit stream");
                return false;
            }
            void *capture_stream = segment_cache.capture_stream;
            bindRetainedParticipantPreparationStream(graph,segment_cache,gpu_ctx);

            if (!DeviceGraphCaptureController::prepareReplayGpuTiming(
                    segment_cache,
                    gpu_ctx,
                    ctx ? ctx->deviceId().toString() : std::string{}))
            {
                LOG_ERROR(
                    "[DeviceGraphExecutor] GPU graph capture could not establish "
                    "its fixed asynchronous replay timing ring");
                return false;
            }

            /*
             * Preserve stream order entirely on device. Previous decode work,
             * including KV/GDN writes, was queued on the worker stream. Capture
             * consumes those tensors on its own stream, so record one
             * worker-stream event and make the capture stream wait for it.
             * A device synchronization here would destroy overlap and obscure
             * the producer/consumer edge represented by this event.
             */
            if (gpu_stream && gpu_stream != capture_stream)
            {
                if (!segment_cache.orderCaptureStreamAfter(
                        gpu_ctx,
                        gpu_stream))
                {
                    LOG_ERROR(
                        "[DeviceGraphExecutor] GPU graph capture could not "
                        "publish its worker-to-capture stream handoff");
                    return false;
                }
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "capture_stream_event_handoffs",
                    1.0,
                    "decode",
                    ctx ? ctx->deviceId().toString() : std::string{},
                    {{"context", segment_cache.perf_context}});
            }

            /*
             * Materialize branch-private graph sources only after every ordinary
             * stream/timing precondition has succeeded. From this point the
             * capture controller owns reset-on-failure, so a partial first-use
             * attempt cannot strand branch identity in an uninitialized cache.
             */
            if (auxiliary_branch_factory.valid())
            {
                try
                {
                    auto branch = auxiliary_branch_factory.create();
                    if (!branch ||
                        branch->authorityIdentity() !=
                            auxiliary_branch_factory.authority_identity ||
                        branch->device() != ctx->deviceId() ||
                        branch->name().empty())
                    {
                        LOG_ERROR(
                            "[DeviceGraphExecutor] Auxiliary-branch factory produced incomplete or mismatched capture identity");
                        return false;
                    }
                    segment_cache.auxiliary_branch_authority =
                        auxiliary_branch_factory.authority_identity;
                    segment_cache.auxiliary_branch = std::move(branch);

                    /*
                     * Hook objects are assembled before the cold-path factory
                     * is invoked so ordinary capture preconditions can fail
                     * without allocating branch-private sources. Refresh both
                     * hook views now that the cache owns its final branch;
                     * otherwise transaction zero captures the model alone and
                     * every later replay permanently omits maintenance work.
                     */
                    replay_hooks.auxiliary_branch =
                        segment_cache.auxiliary_branch.get();
                    capture_hooks.auxiliary_branch =
                        segment_cache.auxiliary_branch.get();
                }
                catch (const std::exception &exception)
                {
                    LOG_ERROR(
                        "[DeviceGraphExecutor] Could not materialize graph-owned auxiliary branch: "
                        << exception.what());
                    return false;
                }
                catch (...)
                {
                    LOG_ERROR(
                        "[DeviceGraphExecutor] Graph-owned auxiliary branch factory threw a non-standard exception");
                    return false;
                }
            }

            graph.reset();
            segment_cache.needs_capture = true;
            const auto capture_result =
                DeviceGraphCaptureController::executeCapturePhase(
                    graph,
                    segment_cache,
                    ctx,
                    gpu_ctx,
                    has_collective_nodes,
                    current_step,
                    capture_hooks,
                    initial_submission);
            if (capture_result.reset_cache)
            {
                segment_cache.reset(
                    GraphSegmentCache::StreamResetPolicy::Preserve);
            }
            if (!capture_result.success)
            {
                return false;
            }
            segment_cache.initialized = true;
            segment_cache.needs_capture = false;
            segment_cache.executable_submission_state =
                materialize_without_launch
                    ? GraphSegmentCache::ExecutableSubmissionState::
                          MaterializedUnlaunched
                    : GraphSegmentCache::ExecutableSubmissionState::ReplayReady;
            if (materialize_without_launch)
            {
                /*
                 * Materialization is not an inference step. Preserve transaction
                 * numbering so the authenticated first launch remains step one in
                 * lifecycle diagnostics and segment publication bookkeeping.
                 */
                segment_cache.decode_step = 0;
                graph.reset();
            }
            if (!seal_retained_full_graph_plan())
                return false;
            if (!seal_retained_parent_plan())
                return false;
            segment_cache.recordSuccessfulCapture(
                !materialize_without_launch);
            return true;
        }

        // Replay is handled by the fast path above; this point is unreachable.
        LOG_ERROR("[DeviceGraphExecutor] Unexpected phase in GPU graph capture/replay");
        return false;
    }

} // namespace llaminar2
