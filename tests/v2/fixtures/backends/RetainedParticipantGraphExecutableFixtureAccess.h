/** @file RetainedParticipantGraphExecutableFixtureAccess.h
 * @brief Original General graph owners and explicitly unqualified actual-program diagnostics.
 *
 * Both roles retain the existing General device reservation on the original
 * prepared-source PMA through ProcessRuntime retirement. Raw diagnostic access
 * mints no host certificate and is unavailable in ordinary perf/test targets.
 */
#pragma once
#include <functional>
#include "backends/RetainedParticipantGraphExecutable.h"
#include <memory>
#include "execution/local_execution/graph/IGraphCaptureAuxiliaryBranch.h"
#include "execution/local_execution/graph/RetainedHostServiceDeclarations.h"
#include "execution/qwen4exp/PooledImmutableDirectedTransport.h"
#include "fixtures/backends/PooledNativeProgramTerminalOrder.h"
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
#include "fixtures/backends/PooledNativeProgramDiagnosticAccess.h"
#include "fixtures/execution/RetainedHostServiceCalibrationAccess.h"
#include "integration/backends/NativeContextSourceProbeSupport.h"
#include "execution/compute_stages/stages/qwen4exp/PooledImmutableSourcePhaseStage.h"
#endif
namespace llaminar2::test {
/** @brief Named test-only bridge to the original closed graph lifecycle. */
struct RetainedParticipantGraphExecutableFixtureAccess final {
    /** @return Original same-PMA General owner, using the same typed allowance as production capture. */
    static std::shared_ptr<RetainedParticipantGraphExecutable> original(const PreparedGPUContextUsePermit& original,void* stream){
        return RetainedParticipantGraphExecutable::create(original,stream);
    }
    /** @return Device-free recorded original lifecycle; no native General host qualification is asserted. */
    static std::shared_ptr<RetainedParticipantGraphExecutable> recorded(const PreparedGPUContextUsePermit& original){
        return RetainedParticipantGraphExecutable::createDeviceBound(original,original.worker()->defaultStream());
    }
#ifdef LLAMINAR_POOLED_NATIVE_PROGRAM_DIAGNOSTIC
    /** @return Explicit raw diagnostic owner: General device bound only, no host qualification claim. */
    static std::shared_ptr<RetainedParticipantGraphExecutable> rawDiagnostic(const PreparedGPUContextUsePermit& original,void* stream){
        return RetainedParticipantGraphExecutable::createDeviceBound(original,stream);
    }
#endif
    /** @return Actual original capture; no replacement graph or kernel program is introduced. */
    static IGPUGraphCapture& capture(const std::shared_ptr<RetainedParticipantGraphExecutable>& owner){return owner->capture();}
    /** @brief Apply the ACK adversarial topology splice to the sole original recorded definition.
     * @param owner Original retained definition, never a replacement capture.
     * @param mutate Test-only SDK mutation which retains and verifies all original dependencies.
     * @throws std::exception Before mutation for foreign worker/source, wrong phase or stream.
     * A failed splice irreversibly fails the definition, matching recording failure semantics.
     */
    template<class Mutation>
    static void spliceRecordedACK(const std::shared_ptr<RetainedParticipantGraphExecutable>& owner,
                                  Mutation&& mutate){
        if(!owner)throw std::invalid_argument("ACK splice requires its original graph owner");
        owner->requireRecordingAuthority();
        if(owner->phase_!=RetainedParticipantGraphExecutable::Phase::Recorded ||
           !owner->capture_ || owner->capture_->executionStream()!=owner->stream_)
            throw std::logic_error("ACK splice requires the original recorded, uninstantiated definition");
        try{mutate(*owner->capture_);}
        catch(...){owner->phase_=RetainedParticipantGraphExecutable::Phase::Failed;throw;}
    }
    /**
     * @brief Replay native finite-body assembly using two original admitted General fixture definitions.
     * @param parent Original recorded native execution owner; only its ordinary submit API launches.
     * @param body Original same-PMA/native-worker recording, never an anonymous SDK graph.
     * @param progress Existing CUDA progress branch, or null for native SDMA.
     * @return Exact native assembly result. Deliberate invalid controls must prove no mutation.
     * @note This proves native composition, not a source/ModelContext witness or the remote69 recipe.
     */
    static bool composeRecordedNativeFixtureBodies(const std::shared_ptr<RetainedParticipantGraphExecutable>& parent,
        std::span<const std::shared_ptr<RetainedParticipantGraphExecutable>> bodies,IGraphCaptureAuxiliaryBranch* progress){
        if(!parent)throw std::invalid_argument("Native finite-body replay omitted its original parent");
        parent->requireRecordingAuthority();
        if(parent->phase_!=RetainedParticipantGraphExecutable::Phase::Recorded)
            throw std::logic_error("Native finite-body replay parent is not an original recording");
        std::vector<const IGPUGraphCapture*> definitions;
        for(const auto& body:bodies){
            if(!body){definitions.push_back(nullptr);continue;}
            body->requireLiveOriginal();
            const auto a=parent->memory_,b=body->memory_;
            if(a.get()!=b.get() || a.owner_before(b) || b.owner_before(a))
                throw std::logic_error("Native finite-body fixture substituted physical-memory authority");
            definitions.push_back(body->capture_.get());
        }
        return progress?progress->attachWithFiniteBody(*parent->capture_,definitions):parent->capture_->appendFiniteBodySiblings(definitions);
    }
    /** @brief Retained original graph-only units for one diagnostic ordered parent; none is executable. */
    struct NativeForkDefinitions {
        std::unique_ptr<IGPUGraphCapture> producer,consumer,auxiliary;
    };
    /**
     * @brief Exercise the actual native ordered-parent/fork builder under its original General owner.
     * @param owner Pristine original same-source General reservation and native capture.
     * @param owned Use original timeline-owned CUDA recording views, or independent import definitions.
     * @param noEdge Diagnostic-only literal old independent-root control; never a serving strategy.
     * @param record Records the three exact graph-only definitions on the original worker/stream.
     * @param composeMain Optional actual production composer over these same graph-only definitions.
     * @return Strong graph definitions retained through the original parent's native terminal.
     * The owner enters Recorded only after the actual backend assembly succeeds;
     * original instantiate/submit/observe remain unchanged and no raw launch escapes.
     */
    template<class Record>
    static NativeForkDefinitions composeNativeForkFixture(
        const std::shared_ptr<RetainedParticipantGraphExecutable>& owner,
        bool owned,bool noEdge,Record record,
        const std::function<bool(IGPUGraphCapture&, const IGPUGraphCapture&, const IGPUGraphCapture&)>& composeMain = {})
    {
        if(!owner)throw std::invalid_argument("Native fork requires its original General owner");
        owner->requireRecordingAuthority();
        if(owner->phase_!=RetainedParticipantGraphExecutable::Phase::Fresh || !owner->capture_ ||
           owner->capture_->nodeCount() || owner->capture_->hasExecutable())
            throw std::logic_error("Native fork fixture requires an untouched original definition");
        NativeForkDefinitions definitions;
        owner->phase_=RetainedParticipantGraphExecutable::Phase::Recording;
        try {
            auto& parent=*owner->capture_;
            definitions.producer=owned?parent.createOrderedTimelineFragment():owner->worker_->createGraphCapture(owner->stream_);
            definitions.consumer=owned?parent.createOrderedTimelineFragment():owner->worker_->createGraphCapture(owner->stream_);
            definitions.auxiliary=owner->worker_->createGraphCapture(owner->stream_);
            if(!definitions.producer || !definitions.consumer || !definitions.auxiliary)
                throw std::runtime_error("Native fork omitted an original graph-only definition");
            record(*definitions.producer,*definitions.consumer,*definitions.auxiliary);
            const std::array steps{
                GPUOrderedTimelineStep{.name="original_probe",.kind=GPUOrderedTimelineStepKind::CapturedFragment,.capture=definitions.producer.get()},
                GPUOrderedTimelineStep{.name="later_local_read",.kind=GPUOrderedTimelineStepKind::CapturedFragment,.capture=definitions.consumer.get()}};
            if(!(composeMain ? composeMain(parent,*definitions.producer,*definitions.consumer)
                             : parent.buildOrderedTimelineTransaction(steps)))
                throw std::runtime_error("Native fork original main timeline assembly failed");
            const auto before=parent.nodeCount();
            if(noEdge) {
                const std::array<const IGPUGraphCapture*,1> siblings{definitions.auxiliary.get()};
                if(!parent.appendFiniteBodySiblings(siblings))
                    throw std::runtime_error("Diagnostic old independent-root control failed to assemble");
            } else if(!parent.appendFiniteBodyAfterCapturedFragment(*definitions.producer,*definitions.auxiliary))
                throw std::runtime_error("Native fork producer attachment failed");
            if(parent.nodeCount()!=before+1 || definitions.auxiliary->hasExecutable() ||
               (!noEdge && parent.appendFiniteBodyAfterCapturedFragment(*definitions.producer,*definitions.auxiliary)))
                throw std::runtime_error("Native fork changed its exact parent/auxiliary topology");
            std::string why;
            if(!parent.validateExecutableMemoryClass(GPUGraphExecutableClass::General,&why))
                throw std::runtime_error("Native fork actual General shape invalid: "+why);
            owner->phase_=RetainedParticipantGraphExecutable::Phase::Recorded;
            return definitions;
        } catch(...) {
            owner->phase_=RetainedParticipantGraphExecutable::Phase::Failed;
            throw;
        }
    }
    /** @return Owner-controlled original enqueue; the recording facade cannot launch independently. */
    static bool submit(const std::shared_ptr<RetainedParticipantGraphExecutable>& owner,std::uint64_t ordinal){return owner->submit(ordinal);}
    /** @return Original native terminal observation, called only outside its owner worker. */
    static bool observeTerminal(const std::shared_ptr<RetainedParticipantGraphExecutable>& owner){return owner->observeTerminal();}
};
}
