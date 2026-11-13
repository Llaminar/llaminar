/**
 * @file LocalTPCaptureSourceDefinitionReplay.inc
 * @brief Replay the actual LocalTP cold generation exchange without native device work.
 *
 * Recorded workers authorize only this unit fixture. The actual cyclic rendezvous,
 * revocable cache token and original source/PMA provenance remain production code;
 * this does not qualify a native executable or its final source seal.
 */
namespace llaminar2::test {
namespace {
/** @brief Recorded declarative plan deliberately has no executable launch authority. */
class ColdDefinitionReplayPlan final:public IRetainedGraphParticipantSourcePlan {
public:
    explicit ColdDefinitionReplayPlan(PreparedGPUContextUsePermit original):original_(std::move(original)){}
    const PreparedGPUContextUsePermit& originalSource()const noexcept override{return original_;}
    void authenticatePreparationStream(std::shared_ptr<const RetainedGraphParticipantPreparationStream> value)const override{value->requireDefinition();}
    uint64_t captureIdentity()const noexcept override{++identityProjections;return 511;}
    void authenticateRecordingOwner(const IGPUGraphCapture&)const override{throw std::logic_error("Unit definition has no executable");}
    std::shared_ptr<IRetainedGraphParticipantInvocation> bindOriginalSource(const RetainedGraphParticipantSourceSeal&)const override{
        throw std::logic_error("Cold definition cannot mint an executable invocation");
    }
    RetainedGraphParticipantDefinitionDomain definitionDomain()const noexcept override{return {original_.device(),this};}
    void requirePreparedSourceDefinition(const RetainedGraphParticipantPreparationStream& value)const override{
        ++coldValidations;requireRecordedIdentity(value);
    }
    void requireRecordedIdentity(const RetainedGraphParticipantPreparationStream& value)const{
        const auto& d=value.definition();
        if(d.scope==RetainedGraphParticipantDefinitionScope::AuxiliaryReader){
            value.requireLive(value.originalSource());
            if(d.graph!=readerGraph || d.arena!=readerArena || d.plan.get()!=this ||
               !readerGraph || d.topology_generation!=readerGraph->topologyGeneration() ||
               d.variant!=expectedVariant || d.domain!=definitionDomain() ||
               d.original_source_definition!=readerSource.lock() || !readerGraph->retainedParticipantAuxiliaryRange() ||
               d.range_identity!=readerGraph->retainedParticipantAuxiliaryRange()->identity)
                throw std::logic_error("Recorded reader definition changed its exact original source/range");
            return;
        }
        value.requireLive(original_);
        if(d.plan.get()!=this || d.graph!=graph || d.arena!=arena || d.domain!=definitionDomain() ||
           !graph || d.topology_generation!=graph->topologyGeneration() || d.variant!=expectedVariant ||
           graph->retainedParticipantSourcePlan().get()!=this)
            throw std::logic_error("Recorded source definition changed its original graph/count/retirement identity");
    }
    void requireCurrentPreparedSourceDefinition(const RetainedGraphParticipantPreparationStream& value)const override{++warmValidations;requireRecordedIdentity(value);}
    std::unique_ptr<IRetainedGraphParticipantStorageAttachment> prepareOriginalStorageAttachment(
        DeviceId receiver,IGPUGraphCapture& parent,IGraphCaptureAuxiliaryBranch* progress,
        std::shared_ptr<const RetainedGraphParticipantPreparationStream> token,const ComputeGraph& receiverGraph)const override{
        requirePreparedSourceDefinition(*token);++attachments;
        if (!receiverGraph.retainedParticipantDefinitionDomain() ||
            *receiverGraph.retainedParticipantDefinitionDomain() != token->definition().domain)
            throw std::logic_error("Recorded source attachment changed its actual recipient graph/domain");
        if(!attach)throw std::logic_error("Recorded adapter omitted its original attachment result");
        return attach(receiver,parent,progress,std::move(token));
    }
    mutable unsigned attachments=0,coldValidations=0,warmValidations=0,identityProjections=0;
    std::function<std::unique_ptr<IRetainedGraphParticipantStorageAttachment>(DeviceId,IGPUGraphCapture&,IGraphCaptureAuxiliaryBranch*,
        std::shared_ptr<const RetainedGraphParticipantPreparationStream>)> attach;
    uint64_t expectedVariant=511;
    const ComputeGraph* graph=nullptr;
    const void* arena=nullptr;
    const ComputeGraph* readerGraph=nullptr;
    const void* readerArena=nullptr;
    std::weak_ptr<const RetainedGraphParticipantPreparationStream> readerSource;
private:PreparedGPUContextUsePermit original_;
};
/** @brief Original cache token retires before its referenced graph or source context. */
struct ColdDefinitionReplay {
    PreparedGPUContextUsePermit original;
    ComputeGraph graph;
    int arena=0;
    std::shared_ptr<ColdDefinitionReplayPlan> plan;
    std::shared_ptr<RetainedGraphParticipantPreparationStream> token;
    explicit ColdDefinitionReplay(PreparedGPUContextUsePermit permit):original(std::move(permit)){
        plan=std::make_shared<ColdDefinitionReplayPlan>(original);graph.setRetainedParticipantSourcePlan(plan);
        plan->graph=&graph;plan->arena=&arena;
        token=RetainedGraphParticipantsReplayAccess::definition(original,reinterpret_cast<void*>(0x9190),
            {&graph,&arena,graph.topologyGeneration(),511,plan->definitionDomain(),plan});
    }
    ~ColdDefinitionReplay(){RetainedGraphParticipantsReplayAccess::retire(*token);}
};
/** @return Original authored membership with the source deliberately last, never inferred from index zero. */
std::vector<GlobalDeviceAddress> coldDefinitionDevices(DeviceId source,int count){
    std::vector<GlobalDeviceAddress> devices;
    for(int i=0;i<count-1;++i)devices.push_back(GlobalDeviceAddress::fromLocalDeviceId(source.is_cuda()?DeviceId::cuda(i):DeviceId::rocm(i)));
    devices.push_back(GlobalDeviceAddress::fromLocalDeviceId(source));return devices;
}
}
/** @test The same cyclic boundary carries exact source identity for all supported metadata degrees and ordinary generations. */
TEST(RetainedGraphParticipantEnqueueGroup, ColdDefinitionUsesOriginalMembershipAndGeneration){
    using A=NativeContextPreparationReplayAccess;
    for(const auto device:{DeviceId::cuda(91),DeviceId::rocm(91)}){
        A::seed(device,[device](auto stack){return A::construct<ColdPreparationWorker>(std::move(stack),device.ordinal);});
        auto source=A::source<ColdPreparationWorker>(device);ColdPreparationRetirement retirement{device};
        ColdDefinitionReplay recorded(source->acquire(device));
        for(const int count:{1,2,4,8}){
            auto context=createLocalTPContext(coldDefinitionDevices(device,count),{},CollectiveBackendType::HOST);
            for(int epoch=0;epoch<4;++epoch){
                std::vector<std::future<bool>> participants;
                for(int i=count-1;i>=0;--i)participants.push_back(std::async(std::launch::async,[&,i]{
                    const auto name="cold:phase=capture_end:epoch="+std::to_string(epoch);
                    LocalTPCaptureSourceDefinition handoff{recorded.plan->definitionDomain(),i==count-1?recorded.token:nullptr,{}};
                    if(!context->graphCaptureBoundaryRendezvous(name,i,30000,handoff) || handoff.result!=recorded.token)return false;
                    return context->graphCaptureBoundaryRendezvous("ordinary:epoch="+std::to_string(epoch),i,30000);
                }));
                for(auto& participant:participants)EXPECT_TRUE(participant.get());
            }
        }
    }
}
/** @test Missing, duplicate, foreign, mismatched and retired publications abort the existing generation and wake its peer. */
TEST(RetainedGraphParticipantEnqueueGroup, ColdDefinitionFailureWakesOriginalMembersWithoutSeal){
    using A=NativeContextPreparationReplayAccess;const auto device=DeviceId::rocm(92);
    A::seed(device,[device](auto stack){return A::construct<ColdPreparationWorker>(std::move(stack),device.ordinal);});
    auto source=A::source<ColdPreparationWorker>(device);ColdPreparationRetirement retirement{device};
    ColdDefinitionReplay recorded(source->acquire(device));
    for(const int failure:{0,1,2,3,4,6,5}){
        auto context=createLocalTPContext(coldDefinitionDevices(device,2),{},CollectiveBackendType::HOST);
        LocalTPCaptureSourceDefinition receiver{recorded.plan->definitionDomain(),{},{}},publisher{recorded.plan->definitionDomain(),recorded.token,{}};
        std::string name="cold:phase=capture_end:original";
        if(failure==0)publisher.publication.reset();
        if(failure==1)receiver.publication=recorded.token;
        if(failure==2)publisher.domain.source=DeviceId::rocm(55);
        if(failure==3)publisher.domain.authority=&publisher;
        if(failure==4)name="cold:phase=capture_begin:original";
        if(failure==5)RetainedGraphParticipantsReplayAccess::retire(*recorded.token);
        if(failure==6)context->requestAbort();
        auto peer=std::async(std::launch::async,[&]{return context->graphCaptureBoundaryRendezvous("cold:phase=capture_end:original",0,30000,receiver);});
        EXPECT_FALSE(context->graphCaptureBoundaryRendezvous(name,1,30000,publisher));
        const auto state=peer.wait_for(std::chrono::seconds(2));
        EXPECT_EQ(state,std::future_status::ready);
        if(state!=std::future_status::ready)context->requestAbort();
        EXPECT_FALSE(peer.get());EXPECT_FALSE(receiver.result);EXPECT_FALSE(publisher.result);
    }
    // A replacement original cache cannot revive the retired token even at the same stream/address contract.
    const auto replacement=RetainedGraphParticipantsReplayAccess::definition(recorded.original,reinterpret_cast<void*>(0x9190),
        {&recorded.graph,&recorded.arena,recorded.graph.topologyGeneration(),511,recorded.plan->definitionDomain(),recorded.plan});
    EXPECT_THROW(recorded.token->requireDefinition(),std::logic_error);EXPECT_NO_THROW(replacement->requireDefinition());
    for(const auto stream:{static_cast<void*>(nullptr),reinterpret_cast<void*>(1),reinterpret_cast<void*>(2)}){
        auto singleton=createLocalTPContext(coldDefinitionDevices(device,1),{},CollectiveBackendType::HOST);
        LocalTPCaptureSourceDefinition handoff{recorded.plan->definitionDomain(),replacement,{}};
        EXPECT_FALSE(singleton->graphCaptureBoundaryOnStream("cold:phase=capture_end:original",0,stream,30000,handoff));
        EXPECT_FALSE(handoff.result);
    }
    {
        auto singleton=createLocalTPContext(coldDefinitionDevices(device,1),{},CollectiveBackendType::HOST);
        LocalTPCaptureSourceDefinition handoff{recorded.plan->definitionDomain(),replacement,{}};
        EXPECT_FALSE(singleton->graphCaptureBoundaryRendezvous("cold:phase=capture_end:original",1,30000,handoff));
        EXPECT_FALSE(handoff.result);
    }
    RetainedGraphParticipantsReplayAccess::retire(*replacement);
}
namespace {
/** @brief Only native operations are doubled; actual cache destruction owns this lifetime. */
class ColdAttachmentProbe final:public IRetainedGraphParticipantStorageAttachment {
public:
    ColdAttachmentProbe(IGPUGraphCapture& parent,std::shared_ptr<const RetainedGraphParticipantPreparationStream> source,
        bool& nativeDestroyed,unsigned& releases):parent_(parent),source_(std::move(source)),nativeDestroyed_(nativeDestroyed),releases_(releases){}
    bool attachTo(IGPUGraphCapture& parent,IGraphCaptureAuxiliaryBranch*)override{
        if(&parent!=&parent_ || attempted_)throw std::logic_error("Recorded attachment substituted/reused its native parent");
        attempted_=true;if(nativeAssemble)nativeAssemble();attached_=true;return true;
    }
    void requireCurrent()const override{
        source_->requireCurrentDefinition();
        if(!attached_ || !parent_.hasExecutable())throw std::logic_error("Original receiver native parent is not instantiated");
    }
    ~ColdAttachmentProbe()override{EXPECT_TRUE(nativeDestroyed_);++releases_;}
    std::function<void()> nativeAssemble; ///< Native failure injection, never a production recovery path.
private:
    IGPUGraphCapture& parent_;std::shared_ptr<const RetainedGraphParticipantPreparationStream> source_;
    bool& nativeDestroyed_;unsigned& releases_;bool attempted_=false,attached_=false;
};
}
/** @test Actual receiver adapter binds exact source and parent, then cache retirement destroys native clones before the body borrow. */
TEST(RetainedGraphParticipantEnqueueGroup, ColdStorageAdapterUsesExactMatchedParentAndRetiresNativeFirst){
    using A=NativeContextPreparationReplayAccess;const auto device=DeviceId::cuda(93);
    A::seed(device,[device](auto stack){return A::construct<ColdPreparationWorker>(std::move(stack),device.ordinal);});
    auto source=A::source<ColdPreparationWorker>(device);ColdPreparationRetirement retirement{device};
    ColdDefinitionReplay original(source->acquire(device));DeviceGraphExecutor executor;
    for(unsigned epoch=0;epoch<4;++epoch){
        bool nativeDestroyed=false;unsigned releases=0;
        DeviceGraphExecutor::GraphSegmentCache cache;
        ASSERT_TRUE(cache.bindBorrowedCaptureStream(original.original.worker().get(),original.token->stream(),device,false));
        ComputeGraph receiver;receiver.setRetainedParticipantDefinitionDomain(original.plan->definitionDomain());
        cache.recorded_source_definition=original.token;
        cache.retained_parent_capture=std::make_unique<testing::MockGPUGraphCapture>(cache.capture_stream,[&]{nativeDestroyed=true;});
        auto& parent=*cache.retained_parent_capture;ASSERT_TRUE(parent.beginCapture());ASSERT_TRUE(parent.endCapture());
        original.plan->attach=[&](DeviceId actual,IGPUGraphCapture& bound,IGraphCaptureAuxiliaryBranch* progress,
            std::shared_ptr<const RetainedGraphParticipantPreparationStream> token){
            EXPECT_EQ(actual,device);EXPECT_EQ(&bound,&parent);EXPECT_EQ(token,original.token);EXPECT_EQ(progress,nullptr);
            EXPECT_FALSE(bound.hasExecutable());EXPECT_EQ(bound.executionStream(),cache.capture_stream);
            return std::make_unique<ColdAttachmentProbe>(bound,std::move(token),nativeDestroyed,releases);
        };
        ASSERT_TRUE(RetainedGraphParticipantsReplayAccess::attachStorage(executor,receiver,cache,device,parent));
        ASSERT_TRUE(cache.original_storage_attachment);
        EXPECT_THROW(cache.original_storage_attachment->requireCurrent(),std::logic_error);
        ASSERT_TRUE(parent.instantiate());EXPECT_NO_THROW(cache.original_storage_attachment->requireCurrent());
        EXPECT_THROW(RetainedGraphParticipantsReplayAccess::attachStorage(executor,receiver,cache,device,parent),std::logic_error);
        cache.reset();EXPECT_TRUE(nativeDestroyed);EXPECT_EQ(releases,1u);
        EXPECT_NO_THROW(original.token->requireDefinition()); // Receiver releases, never revokes source authority.
    }
    EXPECT_EQ(original.plan->attachments,4u);
}
/** @test Missing/foreign/revoked source and failed assembly reject before launch, preserving the original error and ordinary no-payload path. */
TEST(RetainedGraphParticipantEnqueueGroup, ColdStorageAdapterRejectsForeignAndPreservesAssemblyFailure){
    using A=NativeContextPreparationReplayAccess;const auto device=DeviceId::rocm(93);
    A::seed(device,[device](auto stack){return A::construct<ColdPreparationWorker>(std::move(stack),device.ordinal);});
    auto source=A::source<ColdPreparationWorker>(device);ColdPreparationRetirement retirement{device};
    ColdDefinitionReplay original(source->acquire(device));DeviceGraphExecutor executor;
    DeviceGraphExecutor::GraphSegmentCache cache;
    ASSERT_TRUE(cache.bindBorrowedCaptureStream(original.original.worker().get(),original.token->stream(),device,false));
    ComputeGraph receiver;receiver.setRetainedParticipantDefinitionDomain(original.plan->definitionDomain());
    testing::MockGPUGraphCapture parent(cache.capture_stream);ASSERT_TRUE(parent.beginCapture());ASSERT_TRUE(parent.endCapture());
    EXPECT_THROW(RetainedGraphParticipantsReplayAccess::attachStorage(executor,receiver,cache,device,parent),std::logic_error);
    cache.recorded_source_definition=original.token;
    EXPECT_THROW(RetainedGraphParticipantsReplayAccess::attachStorage(executor,receiver,cache,DeviceId::cuda(93),parent),std::logic_error);
    testing::MockGPUGraphCapture foreign(reinterpret_cast<void*>(0x9990));ASSERT_TRUE(foreign.beginCapture());ASSERT_TRUE(foreign.endCapture());
    EXPECT_THROW(RetainedGraphParticipantsReplayAccess::attachStorage(executor,receiver,cache,device,foreign),std::logic_error);
    original.plan->attach=[](auto,auto&,auto*,auto)->std::unique_ptr<IRetainedGraphParticipantStorageAttachment>{throw std::runtime_error("original finite assembly failure");};
    try{(void)RetainedGraphParticipantsReplayAccess::attachStorage(executor,receiver,cache,device,parent);FAIL()<<"Missing original assembly failure";}
    catch(const std::runtime_error& error){EXPECT_STREQ(error.what(),"original finite assembly failure");}
    EXPECT_EQ(parent.launchCount(),0);EXPECT_FALSE(parent.hasExecutable());EXPECT_FALSE(cache.original_storage_attachment);
    EXPECT_EQ(original.plan->attachments,1u);
    RetainedGraphParticipantsReplayAccess::retire(*original.token);
    EXPECT_THROW(RetainedGraphParticipantsReplayAccess::attachStorage(executor,receiver,cache,device,parent),std::logic_error);
    EXPECT_EQ(original.plan->attachments,1u);
    cache.reset();
    ComputeGraph ordinary;EXPECT_TRUE(RetainedGraphParticipantsReplayAccess::attachStorage(executor,ordinary,cache,device,parent));
}
/** @test A native assembly failure cannot release the body before the exact failed parent is discarded. */
TEST(RetainedGraphParticipantEnqueueGroup, ColdStorageAdapterRetainsBorrowAcrossPartialNativeFailure){
    using A=NativeContextPreparationReplayAccess;const auto device=DeviceId::cuda(94);
    A::seed(device,[device](auto stack){return A::construct<ColdPreparationWorker>(std::move(stack),device.ordinal);});
    auto source=A::source<ColdPreparationWorker>(device);ColdPreparationRetirement retirement{device};
    ColdDefinitionReplay original(source->acquire(device));DeviceGraphExecutor executor;
    bool nativeDestroyed=false;unsigned releases=0,partialMutations=0;
    DeviceGraphExecutor::GraphSegmentCache cache;
    ASSERT_TRUE(cache.bindBorrowedCaptureStream(original.original.worker().get(),original.token->stream(),device,false));
    ComputeGraph receiver;receiver.setRetainedParticipantDefinitionDomain(original.plan->definitionDomain());
    cache.recorded_source_definition=original.token;
    cache.retained_parent_capture=std::make_unique<testing::MockGPUGraphCapture>(cache.capture_stream,[&]{nativeDestroyed=true;});
    auto& parent=*cache.retained_parent_capture;ASSERT_TRUE(parent.beginCapture());ASSERT_TRUE(parent.endCapture());
    original.plan->attach=[&](DeviceId,IGPUGraphCapture& actual,IGraphCaptureAuxiliaryBranch*,auto token){
        auto result=std::make_unique<ColdAttachmentProbe>(actual,std::move(token),nativeDestroyed,releases);
        result->nativeAssemble=[&]{EXPECT_TRUE(cache.original_storage_attachment);++partialMutations;throw std::runtime_error("original partial native assembly");};
        return result;
    };
    try{(void)RetainedGraphParticipantsReplayAccess::attachStorage(executor,receiver,cache,device,parent);FAIL()<<"Original partial failure disappeared";}
    catch(const std::runtime_error& failure){EXPECT_STREQ(failure.what(),"original partial native assembly");}
    EXPECT_EQ(partialMutations,1u);EXPECT_TRUE(cache.original_storage_attachment);EXPECT_EQ(releases,0u);
    EXPECT_FALSE(nativeDestroyed);EXPECT_EQ(static_cast<testing::MockGPUGraphCapture&>(parent).launchCount(),0);EXPECT_FALSE(parent.hasExecutable());
    EXPECT_THROW(cache.original_storage_attachment->requireCurrent(),std::logic_error);
    EXPECT_THROW(RetainedGraphParticipantsReplayAccess::attachStorage(executor,receiver,cache,device,parent),std::logic_error);
    EXPECT_EQ(original.plan->attachments,1u);
    cache.reset();EXPECT_TRUE(nativeDestroyed);EXPECT_EQ(releases,1u);EXPECT_NO_THROW(original.token->requireDefinition());
}
/** @test Structural authentication is cold once; warm validation cannot walk it or revive revoked/replaced graph identity. */
TEST(RetainedGraphParticipantEnqueueGroup, ColdStorageValidationSealsOnceAndRejectsReplacedGraphs){
    using A=NativeContextPreparationReplayAccess;const auto device=DeviceId::rocm(95);
    A::seed(device,[device](auto stack){return A::construct<ColdPreparationWorker>(std::move(stack),device.ordinal);});
    auto source=A::source<ColdPreparationWorker>(device);ColdPreparationRetirement retirement{device};
    ColdDefinitionReplay original(source->acquire(device));
    EXPECT_THROW(original.token->requireCurrentDefinition(),std::logic_error);
    original.token->requireDefinition();const auto cold=original.plan->coldValidations,warm=original.plan->warmValidations;
    // The private allocator-interposition replay wraps exactly this same loop;
    // canonical replay independently proves the strict cold/warm dispatch law.
    for(unsigned submission=0;submission<256;++submission)original.token->requireCurrentDefinition();
    EXPECT_EQ(original.plan->coldValidations,cold);EXPECT_EQ(original.plan->warmValidations,warm+256);
    original.graph.clear();
    EXPECT_THROW(original.token->requireCurrentDefinition(),std::logic_error);
    original.graph.setRetainedParticipantSourcePlan(original.plan);
    const auto replacement=RetainedGraphParticipantsReplayAccess::definition(original.original,original.token->stream(),
        {&original.graph,&original.arena,original.graph.topologyGeneration(),511,original.plan->definitionDomain(),original.plan});
    EXPECT_THROW(replacement->requireCurrentDefinition(),std::logic_error);
    replacement->requireDefinition();EXPECT_NO_THROW(replacement->requireCurrentDefinition());
    EXPECT_THROW(original.token->requireCurrentDefinition(),std::logic_error);
    RetainedGraphParticipantsReplayAccess::retire(*replacement);
    EXPECT_THROW(replacement->requireCurrentDefinition(),std::logic_error);
}
/** @test Incomplete original definitions name the missing field before invoking the source plan. */
TEST(RetainedGraphParticipantEnqueueGroup, ColdDefinitionNamesEveryMissingDeclaration){
    using A=NativeContextPreparationReplayAccess;const auto device=DeviceId::cuda(96);
    A::seed(device,[device](auto stack){return A::construct<ColdPreparationWorker>(std::move(stack),device.ordinal);});
    auto source=A::source<ColdPreparationWorker>(device);ColdPreparationRetirement retirement{device};
    ColdDefinitionReplay original(source->acquire(device));
    const auto complete=original.token->definition();
    using Definition=RetainedGraphParticipantPreparationStream::Definition;
    const std::pair<const char*,std::function<void(Definition&)>> cases[]={
        {"plan",[](auto& d){d.plan.reset();}}, {"graph",[](auto& d){d.graph=nullptr;}},
        {"arena",[](auto& d){d.arena=nullptr;}}, {"topology_generation",[](auto& d){d.topology_generation=0;}},
        {"variant",[](auto& d){d.variant=0;}}, {"domain.source",[](auto& d){d.domain.source=DeviceId::invalid();}},
        {"domain.authority",[](auto& d){d.domain.authority=nullptr;}}};
    for(const auto& [field,remove]:cases){
        SCOPED_TRACE(field);auto incomplete=complete;remove(incomplete);
        const auto token=RetainedGraphParticipantsReplayAccess::definition(original.original,original.token->stream(),incomplete);
        const auto before=original.plan->coldValidations;
        try{token->requireDefinition();FAIL()<<"Missing declaration accepted";}
        catch(const std::logic_error& error){EXPECT_EQ(std::string(error.what()),
            std::string("Original source preparation has no complete declarative definition: missing ")+field);}
        EXPECT_EQ(original.plan->coldValidations,before);
        RetainedGraphParticipantsReplayAccess::retire(*token);
    }
    EXPECT_NO_THROW(original.token->requireDefinition());
}
/** @test Actual early binder seals graph identity before cache selection and rejects replacement. */
TEST(RetainedGraphParticipantEnqueueGroup, EarlyBinderUsesActualImmutableGraphIdentity){
    using A=NativeContextPreparationReplayAccess;
    for(const auto device:{DeviceId::cuda(97),DeviceId::rocm(97)}){
        A::seed(device,[device](auto stack){return A::construct<ColdPreparationWorker>(std::move(stack),device.ordinal);});
        auto source=A::source<ColdPreparationWorker>(device);ColdPreparationRetirement retirement{device};
        for(int replay=0;replay<20;++replay){
            ColdDefinitionReplay original(source->acquire(device));BufferArena arena,foreignArena;
            original.plan->arena=&arena;
            original.plan->expectedVariant=DeviceGraphCaptureController::computeCaptureVariantSignature(original.graph);
            DeviceGraphExecutor executor;executor.setArena(&arena);DeviceGraphExecutor::GraphSegmentCache cache;
            auto* worker=original.original.worker().get();
            ASSERT_TRUE(cache.bindBorrowedCaptureStream(worker,original.token->stream(),device,false));
            ASSERT_EQ(cache.capture_variant_signature,0u);
            RetainedGraphParticipantsReplayAccess::bindPreparation(executor,original.graph,cache,worker);
            const auto sealed=cache.participant_preparation_stream;ASSERT_TRUE(sealed);
            EXPECT_NO_THROW(sealed->requireDefinition());
            EXPECT_EQ(sealed->definition().arena,&arena);
            EXPECT_EQ(sealed->definition().variant,original.plan->expectedVariant);
            EXPECT_EQ(cache.capture_variant_signature,0u);
            cache.capture_variant_signature=DeviceGraphCaptureController::computeCaptureVariantSignature(original.graph);
            worker->submitAndWait([&]{
                EXPECT_NO_THROW(RetainedGraphParticipantsReplayAccess::bindPreparation(executor,original.graph,cache,worker));
                EXPECT_EQ(cache.participant_preparation_stream,sealed);
            });
            // Ready replay checks the sealed original borrow without repeating
            // source structural validation or the graph launch-signature walk.
            cache.initialized=true;
            const auto cold=original.plan->coldValidations,warm=original.plan->warmValidations;
            const auto projections=original.plan->identityProjections;
            for(int request=0;request<20;++request)
                RetainedGraphParticipantsReplayAccess::bindPreparation(executor,original.graph,cache,worker);
            EXPECT_EQ(original.plan->coldValidations,cold);
            EXPECT_EQ(original.plan->warmValidations,warm+20);
            EXPECT_EQ(original.plan->identityProjections,projections);
            auto foreignControl=sealed->definition();
            foreignControl.plan=std::shared_ptr<const IRetainedGraphParticipantSourcePlan>(original.plan.get(),[](auto*){});
            const auto counterfeit=RetainedGraphParticipantsReplayAccess::definition(
                original.original,original.token->stream(),foreignControl);
            cache.participant_preparation_stream=counterfeit;
            EXPECT_THROW(RetainedGraphParticipantsReplayAccess::bindPreparation(executor,original.graph,cache,worker),std::logic_error);
            RetainedGraphParticipantsReplayAccess::retire(*counterfeit);
            cache.participant_preparation_stream=sealed;
            cache.initialized=false;
            ++cache.capture_variant_signature;
            EXPECT_THROW(RetainedGraphParticipantsReplayAccess::bindPreparation(executor,original.graph,cache,worker),std::logic_error);
            cache.capture_variant_signature=sealed->definition().variant;
            executor.setArena(&foreignArena);
            EXPECT_THROW(RetainedGraphParticipantsReplayAccess::bindPreparation(executor,original.graph,cache,worker),std::logic_error);
            executor.setArena(&arena);
            ComputeGraph foreign;foreign.setRetainedParticipantSourcePlan(original.plan);
            EXPECT_THROW(RetainedGraphParticipantsReplayAccess::bindPreparation(executor,foreign,cache,worker),std::logic_error);
            original.graph.clear();original.graph.setRetainedParticipantSourcePlan(original.plan);
            cache.capture_variant_signature=0;
            EXPECT_THROW(RetainedGraphParticipantsReplayAccess::bindPreparation(executor,original.graph,cache,worker),std::logic_error);
            cache.reset();EXPECT_THROW(sealed->requireDefinition(),std::logic_error);
        }
    }
}
/** @test Exact original finite source variants and source-free sidecars keep distinct ownership contracts. */
TEST(RetainedGraphParticipantEnqueueGroup, ExpertSourceFamilyOriginalVariantsAndSourceFreeLifecycle)
{
    using A=NativeContextPreparationReplayAccess;
    for (const auto device : {DeviceId::cuda(96),DeviceId::rocm(96)}) {
        A::seed(device,[device](auto stack){return A::construct<ColdPreparationWorker>(std::move(stack),device.ordinal);});
        auto source=A::source<ColdPreparationWorker>(device);ColdPreparationRetirement retirement{device};
        ColdDefinitionReplay recorded(source->acquire(device));recorded.token->requireDefinition();
        for (int repeat=0;repeat<20;++repeat) {
            std::vector<RetainedForwardSourceDefinition> definitions;
            for (const bool decode : {false,true}) {
                ForwardGraphSignature signature;
                signature.seq_len=3;signature.batch_size=1;signature.device=device;signature.decode=decode;
                definitions.push_back(RetainedForwardSourceDefinition::checked(signature,
                    decode?MoEOverlayInferenceGraphRole::MainDecode:MoEOverlayInferenceGraphRole::MainPrefill,
                    3,recorded.graph,recorded.plan->expectedVariant,recorded.token));
            }
            auto selected=retainedExpertSourceVariants(definitions,false,3);
            ASSERT_EQ(selected.size(),1u);EXPECT_EQ(selected.front(),recorded.token);
            ComputeGraph follower;
            follower.setRetainedParticipantSourceVariants(selected);
            EXPECT_EQ(follower.retainedParticipantSourceVariants().front(),recorded.token);
            EXPECT_EQ(*follower.retainedParticipantDefinitionDomain(),recorded.plan->definitionDomain());
            EXPECT_THROW(follower.setRetainedParticipantSourceVariants(selected),std::invalid_argument);
            follower.clear();EXPECT_TRUE(follower.retainedParticipantSourceVariants().empty());
            // Source-free projections carry an immutable cold fact, not a live
            // raw graph borrow. Consumption remains safe after declaration scope.
            std::vector<RetainedForwardSourceDefinition> sidecars;
            {
                ComputeGraph sidecar;
                sidecars.push_back(RetainedForwardSourceDefinition::checked(std::nullopt,
                    MoEOverlayInferenceGraphRole::MTPDraft,3,sidecar,73,{}));
            }
            EXPECT_TRUE(retainedExpertSourceVariants(sidecars,true,3).empty());
            EXPECT_THROW(retainedExpertSourceVariants(sidecars,false,3),std::invalid_argument);
        }
    }
}
/** @test Partial, stale, foreign and conflicting associations reject before an actual follower graph mutates. */
TEST(RetainedGraphParticipantEnqueueGroup, ExpertSourceFamilyRejectsMissingForeignAndPartialBindings)
{
    using A=NativeContextPreparationReplayAccess;const auto device=DeviceId::cuda(97);
    A::seed(device,[device](auto stack){return A::construct<ColdPreparationWorker>(std::move(stack),device.ordinal);});
    auto source=A::source<ColdPreparationWorker>(device);ColdPreparationRetirement retirement{device};
    ColdDefinitionReplay recorded(source->acquire(device));recorded.token->requireDefinition();
    ForwardGraphSignature signature;signature.seq_len=1;signature.batch_size=1;signature.device=device;signature.decode=true;
    const auto original=RetainedForwardSourceDefinition::checked(signature,MoEOverlayInferenceGraphRole::MainDecode,
        1,recorded.graph,recorded.plan->expectedVariant,recorded.token);
    ComputeGraph foreign,follower;
    EXPECT_THROW(RetainedForwardSourceDefinition::checked(signature,MoEOverlayInferenceGraphRole::MainDecode,
        1,foreign,recorded.plan->expectedVariant,recorded.token),std::invalid_argument);
    EXPECT_THROW(RetainedForwardSourceDefinition::checked(signature,MoEOverlayInferenceGraphRole::MainDecode,
        1,recorded.graph,recorded.plan->expectedVariant,{}),std::invalid_argument);
    const std::vector<RetainedForwardSourceDefinition> originals{original};
    EXPECT_THROW(retainedExpertSourceVariants(originals,false,2),std::invalid_argument);
    EXPECT_THROW(retainedExpertSourceVariants({},false,1),std::invalid_argument);
    const auto source_free=RetainedForwardSourceDefinition::checked(signature,MoEOverlayInferenceGraphRole::MainDecode,
        1,foreign,73,{});
    const std::vector<RetainedForwardSourceDefinition> mixed{original,source_free};
    EXPECT_THROW(retainedExpertSourceVariants(mixed,false,1),std::invalid_argument);
    ColdDefinitionReplay other(source->acquire(device));other.token->requireDefinition();
    const auto other_definition=RetainedForwardSourceDefinition::checked(signature,MoEOverlayInferenceGraphRole::MainDecode,
        1,other.graph,other.plan->expectedVariant,other.token);
    const std::vector<RetainedForwardSourceDefinition> foreign_domain{original,other_definition};
    EXPECT_THROW(retainedExpertSourceVariants(foreign_domain,false,1),std::invalid_argument);
    EXPECT_TRUE(follower.retainedParticipantSourceVariants().empty());EXPECT_FALSE(follower.retainedParticipantDefinitionDomain());
    RetainedGraphParticipantsReplayAccess::retire(*recorded.token);
    EXPECT_THROW(retainedExpertSourceVariants(originals,false,1),std::logic_error);
    EXPECT_TRUE(follower.retainedParticipantSourceVariants().empty());
}
/** @test A structurally plausible physical owner gains no expert role from discovery or continuation membership. */
TEST(RetainedGraphParticipantEnqueueGroup, ExpertOwnerRequiresOriginalExecutionRole)
{
    using A=NativeContextPreparationReplayAccess;
    for (const auto device : {DeviceId::cuda(98),DeviceId::rocm(98)}) {
        A::seed(device,[device](auto stack){return A::construct<ColdPreparationWorker>(std::move(stack),device.ordinal);});
        auto source=A::source<ColdPreparationWorker>(device);ColdPreparationRetirement retirement{device};
        const auto original=source->acquire(device);
        MoEExpertOverlayExecutionPlan execution;execution.continuation_domain="original_continuation";
        execution.domains.emplace_back();auto& domain=execution.domains.back();domain.name=execution.continuation_domain;
        domain.participants.emplace_back();auto& participant=domain.participants.back();
        participant.local_device=device;participant.world_rank=original.memory()->worldRank();
        participant.owned_by_current_rank=true;participant.locally_addressable=true;
        using Policy=qwen4exp::PooledImmutableParticipantExecutionPolicy;
        EXPECT_FALSE(Policy::declaresExpertOwner(execution,original));
        domain.name="actual_expert_follower";EXPECT_TRUE(Policy::declaresExpertOwner(execution,original));
        participant.owned_by_current_rank=false;EXPECT_FALSE(Policy::declaresExpertOwner(execution,original));participant.owned_by_current_rank=true;
        participant.locally_addressable=false;EXPECT_FALSE(Policy::declaresExpertOwner(execution,original));participant.locally_addressable=true;
        ++participant.world_rank;EXPECT_FALSE(Policy::declaresExpertOwner(execution,original));--participant.world_rank;
        participant.local_device=DeviceId::cpu();EXPECT_FALSE(Policy::declaresExpertOwner(execution,original));
    }
}

/** @test Partial follower-family construction releases every actual cache body before revoking the original source token. */
TEST(RetainedGraphParticipantEnqueueGroup, ExpertSourceFamilyPartialParentsRetireBeforeOriginalSource)
{
    using A=NativeContextPreparationReplayAccess;const auto device=DeviceId::rocm(99);
    A::seed(device,[device](auto stack){return A::construct<ColdPreparationWorker>(std::move(stack),device.ordinal);});
    auto source=A::source<ColdPreparationWorker>(device);ColdPreparationRetirement retirement{device};
    ColdDefinitionReplay original(source->acquire(device));original.token->requireDefinition();DeviceGraphExecutor executor;
    ForwardGraphSignature signature;signature.seq_len=1;signature.batch_size=1;signature.device=device;signature.decode=true;
    const std::array definitions{RetainedForwardSourceDefinition::checked(signature,MoEOverlayInferenceGraphRole::MainDecode,
        1,original.graph,original.plan->expectedVariant,original.token)};
    const auto variants=retainedExpertSourceVariants(definitions,false,1);
    for (unsigned repeat=0;repeat<20;++repeat) {
        std::array<bool,4> native_destroyed{};std::array<unsigned,4> releases{};
        std::array<ComputeGraph,4> followers;
        std::array<DeviceGraphExecutor::GraphSegmentCache,4> caches;
        unsigned attached=0;
        try {
            for (unsigned i=0;i<4;++i) {
                auto& cache=caches[i];auto& graph=followers[i];
                ASSERT_TRUE(cache.bindBorrowedCaptureStream(original.original.worker().get(),original.token->stream(),device,false));
                graph.setRetainedParticipantSourceVariants(variants);
                cache.recorded_source_definition=original.token;
                cache.retained_parent_capture=std::make_unique<testing::MockGPUGraphCapture>(cache.capture_stream,[&,i]{native_destroyed[i]=true;});
                auto& parent=*cache.retained_parent_capture;ASSERT_TRUE(parent.beginCapture());ASSERT_TRUE(parent.endCapture());
                original.plan->attach=[&,i](DeviceId,IGPUGraphCapture& actual,IGraphCaptureAuxiliaryBranch*,auto token) {
                    auto body=std::make_unique<ColdAttachmentProbe>(actual,std::move(token),native_destroyed[i],releases[i]);
                    body->nativeAssemble=[&,i] {
                        EXPECT_TRUE(caches[i].original_storage_attachment);
                        if (i==1) throw std::runtime_error("original second follower partial attachment");
                        return true;
                    };
                    return body;
                };
                ASSERT_TRUE(RetainedGraphParticipantsReplayAccess::attachStorage(executor,graph,cache,device,parent));
                ASSERT_TRUE(parent.instantiate());++attached;
            }
            FAIL()<<"Original partial attachment failure disappeared";
        } catch (const std::runtime_error& failure) {
            EXPECT_STREQ(failure.what(),"original second follower partial attachment");
        }
        EXPECT_EQ(attached,1u);EXPECT_EQ(releases[0],0u);EXPECT_EQ(releases[1],0u);
        EXPECT_NO_THROW(original.token->requireCurrentDefinition());
        for (unsigned i=0;i<4;++i) {
            caches[i].reset();followers[i].clear();
            EXPECT_EQ(releases[i],i<2?1u:0u);
            if (i<2) EXPECT_TRUE(native_destroyed[i]);
        }
        // Actual cache reset destroys native graph handles first, then body
        // borrows. Only this completed frontier permits source revocation.
        EXPECT_NO_THROW(original.token->requireCurrentDefinition());
    }
    RetainedGraphParticipantsReplayAccess::retire(*original.token);
    EXPECT_THROW(original.token->requireCurrentDefinition(),std::logic_error);
}

/** @test Additive cold aliases stage atomically, preserve old invocations and reject stale/foreign replacement. */
TEST(RetainedGraphParticipantEnqueueGroup, ExpertSourceFamilyAliasRefreshKeepsOriginalInvocations)
{
    using A=NativeContextPreparationReplayAccess;const auto device=DeviceId::rocm(100);
    A::seed(device,[device](auto stack){return A::construct<ColdPreparationWorker>(std::move(stack),device.ordinal);});
    auto source=A::source<ColdPreparationWorker>(device);ColdPreparationRetirement retirement{device};
    ColdDefinitionReplay original(source->acquire(device));original.token->requireDefinition();
    const auto next=RetainedGraphParticipantsReplayAccess::definition(original.original,original.token->stream(),original.token->definition());
    next->requireDefinition();
    std::vector<std::shared_ptr<const RetainedGraphParticipantPreparationStream>> prior{original.token},proposed{next};
    const auto extended=retainedSourceVariantExtension(prior,proposed);
    ASSERT_EQ(extended.size(),2u);EXPECT_EQ(extended.front(),original.token);EXPECT_EQ(extended.back(),next);
    EXPECT_EQ(prior.size(),1u);EXPECT_EQ(prior.front(),original.token);
    EXPECT_EQ(retainedSourceVariantExtension(extended,proposed),extended);
    EXPECT_NO_THROW(original.token->requireCurrentDefinition());
    EXPECT_NO_THROW(next->requireCurrentDefinition());
    ColdDefinitionReplay foreign(source->acquire(device));foreign.token->requireDefinition();
    const std::vector<std::shared_ptr<const RetainedGraphParticipantPreparationStream>> incompatible{next,foreign.token};
    EXPECT_THROW(retainedSourceVariantExtension(prior,incompatible),std::invalid_argument);
    EXPECT_EQ(prior.size(),1u);EXPECT_EQ(prior.front(),original.token); // Whole proposed set fails before publication.
    RetainedGraphParticipantsReplayAccess::retire(*next);
    EXPECT_THROW(retainedSourceVariantExtension(prior,proposed),std::logic_error);
    EXPECT_NO_THROW(original.token->requireCurrentDefinition()); // The original alias remains a valid invocation.
}

/** @brief Actual draft geometry follows request batches, independently of verifier/prefill rows. */
TEST(RetainedGraphParticipantEnqueueGroup, ExpertSourceFamilyDraftRequestManifest)
{
    const std::vector<std::size_t> backing{1,2,4,8,16,32};
    const std::vector<int> prefill{8,32};
    const auto main=MoEOverlayFollowerRowGeometry::declare(32,16,prefill,backing);
    for(int rows=1;rows<=16;++rows)ASSERT_NE(main.forRows(rows),nullptr);
    ASSERT_NE(main.forRows(32),nullptr);
    for(const int requests:{1,2,4,8})
    {
        const auto draft=MoEOverlayFollowerRowGeometry::declareDraft(requests,backing);
        ASSERT_EQ(draft.shapes().size(),static_cast<std::size_t>(requests));
        for(int count=1;count<=requests;++count)ASSERT_NE(draft.forRows(count),nullptr);
        EXPECT_EQ(draft.forRows(requests+1),nullptr);
        EXPECT_EQ(draft.forRows(32),nullptr); // KV-only shifted prefill is not a draft ticket.
    }
    EXPECT_THROW(MoEOverlayFollowerRowGeometry::declareDraft(33,backing),std::invalid_argument);
}

} // namespace llaminar2::test

namespace llaminar2::test {
namespace {
/** @brief Metadata replay of the existing attachment liveness check; grants no native execution. */
class ReaderProjectionAttachment final:public IRetainedGraphParticipantStorageAttachment {
public:
    bool attachTo(IGPUGraphCapture&,IGraphCaptureAuxiliaryBranch*)override{throw std::logic_error("Metadata replay has no native parent");}
    void requireCurrent()const override{++checks;if(!live)throw std::logic_error("Recorded parent attachment retired");}
    bool live=true;
    mutable unsigned checks=0;
};
}
/** @test An actual associated reader is authenticated without becoming source-free or a second source owner. */
TEST(RetainedGraphParticipantEnqueueGroup, ExpertSourceFamilyAssociatedReaderProjection)
{
    using A=NativeContextPreparationReplayAccess;
    for(const bool cuda:{true,false}){
        const auto owner=cuda?DeviceId::cuda(104):DeviceId::rocm(104);
        const auto reader=cuda?DeviceId::cuda(105):DeviceId::rocm(105);
        const std::array devices{owner,reader};
        for(const auto device:devices)A::seed(device,[device](auto stack){return A::construct<ColdPreparationWorker>(std::move(stack),device.ordinal);});
        const std::array resources{PhysicalMemoryResource{0,DeviceId::cpu(),256u<<20,256u<<20},
            PhysicalMemoryResource{0,owner,64u<<20,64u<<20},PhysicalMemoryResource{0,reader,64u<<20,64u<<20}};
        auto population=A::source<ColdPreparationWorker>(devices,resources);
        ColdPreparationRetirement retireOwner{owner},retireReader{reader};
        ColdDefinitionReplay source(population->acquire(owner));source.token->requireDefinition();
        ComputeGraph graph;graph.setRetainedParticipantDefinitionDomain(source.plan->definitionDomain());
        for(const auto* name:{"probe","read","publish","retire"})
            graph.addNode(name,std::make_unique<testing::MockComputeStage>(ComputeStageType::GEMM,name,reader),reader);
        graph.setRetainedParticipantAuxiliaryRange({.role=RetainedGraphParticipantAuxiliaryRole::Reader,
            .participant=reader,.nodes={"publish","retire"},.probe_ready="probe",.ready_publication="publish",
            .main_read="read",.retirement="retire",.identity=725});
        int arena=0;source.plan->readerGraph=&graph;source.plan->readerArena=&arena;source.plan->readerSource=source.token;
        auto body=RetainedGraphParticipantsReplayAccess::definition(population->acquire(reader),reinterpret_cast<void*>(0x9191),
            {&graph,&arena,graph.topologyGeneration(),511,source.plan->definitionDomain(),source.plan,
             RetainedGraphParticipantDefinitionScope::AuxiliaryReader,reader,725,source.token});
        body->requireDefinition();ReaderProjectionAttachment attachment;
        ForwardGraphSignature signature;signature.batch_size=1;signature.seq_len=3;signature.device=reader;signature.decode=true;
        const auto project=[&](std::shared_ptr<const RetainedGraphParticipantPreparationStream> received,std::shared_ptr<const RetainedGraphParticipantPreparationStream> auxiliary,const IRetainedGraphParticipantStorageAttachment* parent){
            return RetainedForwardSourceDefinition::projectOwned(signature,MoEOverlayInferenceGraphRole::MainDecode,
                3,graph,511,{},received,auxiliary,parent);
        };
        // Literal original failing boundary: a domain-bearing reader cannot be
        // represented by checked() as either owned-source or source-free.
        EXPECT_THROW(RetainedForwardSourceDefinition::checked(signature,MoEOverlayInferenceGraphRole::MainDecode,
            3,graph,511,{}),std::invalid_argument);
        for(int repeat=0;repeat<20;++repeat){
            EXPECT_FALSE(project(source.token,body,&attachment));
            auto ownerSignature=signature;ownerSignature.device=owner;
            auto owned=RetainedForwardSourceDefinition::projectOwned(ownerSignature,MoEOverlayInferenceGraphRole::MainDecode,
                3,source.graph,511,source.token,{}, {},nullptr);
            ASSERT_TRUE(owned);EXPECT_EQ(owned->source,source.token);
            const std::array originals{*owned};EXPECT_EQ(retainedExpertSourceVariants(originals,false,3).size(),1u);
        }
        EXPECT_EQ(attachment.checks,20u);
        EXPECT_THROW(project({},body,&attachment),std::invalid_argument);
        EXPECT_THROW(project(source.token,{},&attachment),std::invalid_argument);
        EXPECT_THROW(project(source.token,body,nullptr),std::invalid_argument);
        const auto relabelled=std::shared_ptr<const RetainedGraphParticipantPreparationStream>(source.token.get(),[](const auto*){});
        EXPECT_THROW(project(relabelled,body,&attachment),std::invalid_argument);
        ColdDefinitionReplay foreign(population->acquire(owner));foreign.token->requireDefinition();
        EXPECT_THROW(project(foreign.token,body,&attachment),std::invalid_argument);
        auto wrong=signature;wrong.device=owner;
        EXPECT_THROW(RetainedForwardSourceDefinition::projectOwned(wrong,MoEOverlayInferenceGraphRole::MainDecode,
            3,graph,511,{},source.token,body,&attachment),std::invalid_argument);
        EXPECT_THROW(RetainedForwardSourceDefinition::projectOwned(signature,MoEOverlayInferenceGraphRole::MainDecode,
            2,graph,511,{},source.token,body,&attachment),std::invalid_argument);
        EXPECT_THROW(RetainedForwardSourceDefinition::projectOwned(signature,MoEOverlayInferenceGraphRole::MainPrefill,
            3,graph,511,{},source.token,body,&attachment),std::invalid_argument);
        EXPECT_THROW(RetainedForwardSourceDefinition::projectOwned(signature,MoEOverlayInferenceGraphRole::MainDecode,
            3,graph,512,{},source.token,body,&attachment),std::invalid_argument);
        attachment.live=false;EXPECT_THROW(project(source.token,body,&attachment),std::logic_error);attachment.live=true;
        RetainedGraphParticipantsReplayAccess::retire(*body);
        EXPECT_THROW(project(source.token,body,&attachment),std::logic_error);
        auto replacement=RetainedGraphParticipantsReplayAccess::definition(population->acquire(reader),reinterpret_cast<void*>(0x9191),body->definition());
        replacement->requireDefinition();
        RetainedGraphParticipantsReplayAccess::retire(*source.token);
        EXPECT_THROW(project(source.token,replacement,&attachment),std::logic_error);
        RetainedGraphParticipantsReplayAccess::retire(*replacement);
        source.plan->readerSource.reset();
    }
}

} // namespace llaminar2::test

namespace llaminar2::test {
/** @test Original captured source tokens cover retained expert rows even when the first request is Off. */
TEST(RetainedGraphParticipantEnqueueGroup, ExpertSourceFamilyRetainedRowsAcrossInitialPolicies)
{
    using A=NativeContextPreparationReplayAccess;
    const auto device=DeviceId::cuda(106);
    A::seed(device,[device](auto stack){return A::construct<ColdPreparationWorker>(std::move(stack),device.ordinal);});
    auto source=A::source<ColdPreparationWorker>(device);ColdPreparationRetirement retirement{device};
    ColdDefinitionReplay recorded(source->acquire(device));recorded.token->requireDefinition();
    for(int repeat=0;repeat<20;++repeat)
    for(const int retained:{1,3,7,15})
    for(const int initial:{0,1,3,-1}) {
        if(initial>retained)continue;
        MTPRuntimeConfig mtp;mtp.enabled=initial!=0;mtp.graph_capacity_draft_tokens=retained;
        mtp.draft_tokens=initial>0?initial:initial==0?0:1;
        if(initial==-1) {
            mtp.depth_policy.mode=MTPDepthPolicyMode::Dynamic;
            mtp.depth_policy.min_depth=1;mtp.depth_policy.initial_depth=1;mtp.depth_policy.max_depth=retained;
        }
        std::vector<RetainedForwardSourceDefinition> definitions;
        ForwardGraphSignature signature;signature.device=device;signature.decode=true;signature.batch_size=1;signature.seq_len=1;
        definitions.push_back(RetainedForwardSourceDefinition::checked(signature,MoEOverlayInferenceGraphRole::MainDecode,
            1,recorded.graph,511,recorded.token));
        // Each projection consumes the SAME checked original token API as FEE.
        // This metadata replay does not claim a native verifier executable.
        for(const auto& member:resolveMTPServingForwardCaptureFamily(mtp,retained+1,MTPVerifierGraphFamily::ExactRows)) {
            signature.seq_len=member.verifier_rows;signature.execution_role=ForwardExecutionRole::GroupedMTPVerifier;
            definitions.push_back(RetainedForwardSourceDefinition::checked(signature,MoEOverlayInferenceGraphRole::MTPGroupedVerifier,
                member.verifier_rows,recorded.graph,511,recorded.token));
        }
        const std::array<std::size_t,5> backing{1,2,4,8,16};
        const auto actual=MoEOverlayFollowerRowGeometry::declare(retained+1,retained+1,{},backing);
        for(const auto& shape:actual.shapes()) {
            const auto matched=retainedExpertSourceVariants(definitions,false,shape.physical_rows);
            ASSERT_EQ(matched.size(),1u);EXPECT_EQ(matched.front(),recorded.token);
        }
        EXPECT_THROW(retainedExpertSourceVariants(definitions,true,1),std::invalid_argument);
        EXPECT_THROW(retainedExpertSourceVariants(definitions,false,retained+2),std::invalid_argument);
    }
}
} // namespace llaminar2::test
