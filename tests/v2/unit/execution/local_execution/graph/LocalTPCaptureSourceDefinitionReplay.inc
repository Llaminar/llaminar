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
    uint64_t captureIdentity()const noexcept override{return 511;}
    void authenticateRecordingOwner(const IGPUGraphCapture&)const override{throw std::logic_error("Unit definition has no executable");}
    std::shared_ptr<IRetainedGraphParticipantInvocation> bindOriginalSource(const RetainedGraphParticipantSourceSeal&)const override{
        throw std::logic_error("Cold definition cannot mint an executable invocation");
    }
    RetainedGraphParticipantDefinitionDomain definitionDomain()const noexcept override{return {original_.device(),this};}
    void requirePreparedSourceDefinition(const RetainedGraphParticipantPreparationStream& value)const override{
        ++coldValidations;requireRecordedIdentity(value);
    }
    void requireRecordedIdentity(const RetainedGraphParticipantPreparationStream& value)const{
        value.requireLive(original_);const auto& d=value.definition();
        if(d.plan.get()!=this || d.graph!=graph || d.arena!=arena || d.domain!=definitionDomain() ||
           !graph || d.topology_generation!=graph->topologyGeneration() || d.variant!=511 ||
           graph->retainedParticipantSourcePlan().get()!=this)
            throw std::logic_error("Recorded source definition changed its original graph/count/retirement identity");
    }
    void requireCurrentPreparedSourceDefinition(const RetainedGraphParticipantPreparationStream& value)const override{++warmValidations;requireRecordedIdentity(value);}
    std::unique_ptr<IRetainedGraphParticipantStorageAttachment> prepareOriginalStorageAttachment(
        DeviceId receiver,IGPUGraphCapture& parent,IGraphCaptureAuxiliaryBranch* progress,
        std::shared_ptr<const RetainedGraphParticipantPreparationStream> token)const override{
        requirePreparedSourceDefinition(*token);++attachments;
        if(!attach)throw std::logic_error("Recorded adapter omitted its original attachment result");
        return attach(receiver,parent,progress,std::move(token));
    }
    mutable unsigned attachments=0,coldValidations=0,warmValidations=0;
    std::function<std::unique_ptr<IRetainedGraphParticipantStorageAttachment>(DeviceId,IGPUGraphCapture&,IGraphCaptureAuxiliaryBranch*,
        std::shared_ptr<const RetainedGraphParticipantPreparationStream>)> attach;
    const ComputeGraph* graph=nullptr;
    const void* arena=nullptr;
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
} // namespace llaminar2::test
