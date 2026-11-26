/**
 * @file FullNativeHCFFNEpochReplay.inc
 * @brief Retained original staged/full graphs across a genuine published source-bank epoch.
 *
 * The enclosing physical-fabric fixture owns original CPU/GPU source arrivals,
 * the canonical runtime/epoch arena, and publication. This observer never writes
 * a descriptor, selector, epoch, histogram or ticket. It authors only declared
 * FP32 graph inputs and the original INT32 row owner through TransferEngine.
 */
#include "SelectedDescriptorObserver.inc"

enum class EpochNumericalBody { OriginalNative, RetainedBank, RetainedWithDescriptorReader };
class FullEpochNumericalPair final {
 static constexpr unsigned R=320,Planes=13;
 DeviceId device_;IWorkerGPUContext& worker_;std::shared_ptr<ILocalTPContext> tp_;void* stream_=nullptr;
 std::shared_ptr<PublishedFFNEpoch> initial_;
 OriginalFFNFixtureGeometry shape_;
 std::unique_ptr<IDeviceContext> context_;
 std::unique_ptr<ScopedGPUCompletionEvent> completion_;
 std::array<FullProductionHCPrefix,2> prefix_;
 std::array<std::unique_ptr<Lane>,2> lane_;
 EpochNumericalBody body_;
 std::unique_ptr<INT32Tensor> selected_descriptors_;
 /** @return Whether the retained graph includes the actual declared descriptor reader. */
 bool hasDescriptorReader() const noexcept {return body_==EpochNumericalBody::RetainedWithDescriptorReader;}
 /** @return Whether this proof instantiates the original retained production owner. */
 bool retained() const noexcept {return body_!=EpochNumericalBody::OriginalNative;}
 /** @return The descriptor ABI expected from actual compiled graph edges. */
 MoESelectedDescriptorDemand descriptorDemand() const noexcept {
  return hasDescriptorReader()?MoESelectedDescriptorDemand::ExternalTables:MoESelectedDescriptorDemand::AcquiredBankOnly;
 }
 std::array<std::vector<float>,2> injection_,paired_write_;
 std::array<std::unique_ptr<IGPUGraphCapture>,2> capture_;
 std::array<std::array<std::vector<float>,Planes>,2> values_,first_,reset_;
 std::array<std::vector<float>,2> ids_,weights_,first_ids_,first_weights_;
 std::array<std::vector<std::uint64_t>,2> demand_,local_;
 std::vector<std::uint64_t> expected_demand_,expected_local_;
 std::vector<float> before_combined_,input_post_,input_block_,input_gate_;
 std::array<std::vector<float>,6> before_prefix_;
 std::vector<DeviceMoEExpertDescriptor> acquired_;
 std::array<std::vector<DeviceNativeQuantisedMatrixDesc>,3> published_;
 const void* full_function_=nullptr;
 std::array<std::size_t,Planes> widths_;
 DeviceMoEOverlayEpochStatus status_{};int32_t observed_count_=-1;std::array<uint32_t,2> acquired_header_{};
 unsigned member_=0;
 unsigned captures_built_=0;void* terminal_event_=nullptr;bool terminal_recorded_=false;
 /** @brief Compare original public bytes, retaining signed-zero and NaN payload bits.
  * @tparam T Original downloaded scalar type. @param expected Prior/oracle bytes.
  * @param actual Completed device observation. @return Complete byte equality, including empty extent. */
 template<class T> static bool same(const std::vector<T>& expected,const std::vector<T>& actual) {
  return expected.size()==actual.size() && (expected.empty() ||
   std::memcmp(expected.data(),actual.data(),expected.size()*sizeof(T))==0);
 }
 /** @brief Observe only completed device values; this buffer is never uploaded. */
 void download(void* destination,const void* source,std::size_t bytes,const char* why) {
  require(getBackendFor(device_)->deviceToHostOnStream(destination,source,bytes,device_.ordinal,stream_),why);
 }
 /** @brief Author only original external inputs/count, explicitly publishing each host write. */
 void publish(unsigned history,bool live) {
  for(unsigned i=0;i<input_post_.size();++i)input_post_[i]=.125f+float((i+history)%19)/512.f;
  for(unsigned i=0;i<input_block_.size();++i)input_block_[i]=.0625f+float((i+history)%13)/512.f;
  for(unsigned i=0;i<4;++i)input_gate_[i]=float(int((i+history)%5)-2)/16.f;
  for(unsigned side=0;side<2;++side) {
   auto& p=prefix_[side];auto& l=*lane_[side];
   std::memcpy(p.residual->mutable_data(),input_post_.data(),input_post_.size()*sizeof(float));
   std::memcpy(p.previous->mutable_data(),input_block_.data(),input_block_.size()*sizeof(float));
   std::memcpy(p.raw->mutable_data(),input_gate_.data(),4*sizeof(float));
   TransferEngine::prepareDeviceInput(p.residual.get(),device_,stream_);
   TransferEngine::prepareDeviceInput(p.previous.get(),device_,stream_);
   TransferEngine::prepareDeviceInput(p.raw.get(),device_,stream_);
   l.active->mutable_typed_data()[0]=live?1:0;TransferEngine::publishHostWrite(l.active.get());
   TransferEngine::prepareDeviceInput(l.active.get(),device_,stream_);
  }
 }
 /** @brief Validate one actual launch against current source bank and cumulative device counters. */
 void observe(unsigned side,std::uint64_t epoch,const MoEOverlayParticipantLayerBank& source,bool live) {
  auto& p=prefix_[side];auto& l=*lane_[side];
  const std::array<const void*,Planes> ptrs{p.post->gpu_data_ptr(),p.normalized->gpu_data_ptr(),p.down->gpu_data_ptr(),
   p.activated->gpu_data_ptr(),p.logits->gpu_data_ptr(),l.hidden->gpu_data_ptr(),
   l.workspace->getBuffer(MoEWorkspaceBuffers::ROUTE_LOGITS),l.combined->gpu_data_ptr(),l.routes->gpu_data_ptr(),
   l.routed->gpu_data_ptr(),l.shared->gpu_data_ptr(),l.gate_scratch->gpu_data_ptr(),l.up_scratch->gpu_data_ptr()};
  for(unsigned plane=0;plane<Planes;++plane)download(values_[side][plane].data(),ptrs[plane],widths_[plane]*sizeof(float),"epoch full public plane");
  if(p.injection) {
   download(injection_[side].data(),p.injection->gpu_data_ptr(),4*sizeof(float),"actual original standalone FP32 injection");
   download(paired_write_[side].data(),p.next_trunk->gpu_data_ptr(),4u*shape_.hidden*sizeof(float),"actual original paired write");
  }
  download(ids_[side].data(),l.indices->gpu_data_ptr(),ids_[side].size()*sizeof(float),"epoch full public IDs");
  download(weights_[side].data(),l.weights->gpu_data_ptr(),weights_[side].size()*sizeof(float),"epoch full public weights");
  const auto* bytes=reinterpret_cast<const std::byte*>(l.runtime->deviceLayerState(0));
  download(demand_[side].data(),bytes+offsetof(DeviceMoELayerRuntime,decode_histogram),shape_.experts*sizeof(uint64_t),"epoch actual selected demand");
  download(local_[side].data(),bytes+offsetof(DeviceMoELayerRuntime,decode_local_histogram),shape_.experts*sizeof(uint64_t),"epoch actual local demand");
  unsigned long long transferred_payload=0;
  if(tp_->degree()>1)
   download(&transferred_payload,l.receipt->gpu_data_ptr(),sizeof(transferred_payload),"epoch native collective payload bytes");
  download(&status_,l.runtime->overlayEpochStatus(),sizeof(status_),"epoch captured release status");
  observed_count_=-1;download(&observed_count_,l.active->gpu_data_ptr(),sizeof(observed_count_),"epoch actual row authority");
  require(completion_->join(),"retire epoch numerical observation");
  require(status_.succeeded()&&status_.typedOperation()==DeviceMoEOverlayEpochOperation::Release&&
   status_.epoch==epoch&&status_.bank<2,"actual captured acquire/full/collective/release selects intended published epoch");
  require(observed_count_==(live?1:0),"epoch actual device count follows published active/empty/reset");
  if(tp_->degree()>1)require(transferred_payload==std::size_t(live)*shape_.hidden*sizeof(float)*(tp_->degree()-1),
   "actual original epoch collective communicates exact live extent");
  if(live)for(float id:ids_[side]) {
   require(id>=0&&id<float(shape_.experts)&&id==float(int(id)),"original router actual live expert");
   const auto expert=unsigned(id);++expected_demand_[expert];
   if(source.resident_mask.at(expert))++expected_local_[expert];
  }
  const std::array names{MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,
    MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE};
  if(side&&retained()&&!hasDescriptorReader()) {
   std::vector<std::int32_t> dispatch_ids(shape_.top_k),dispatch_participants(shape_.top_k);
   std::vector<float> dispatch_weights(shape_.top_k);
   const std::int32_t* route_participants=nullptr;
   download(dispatch_ids.data(),bytes+offsetof(DeviceMoELayerRuntime,topk_expert_ids),
    dispatch_ids.size()*sizeof(dispatch_ids[0]),"epoch internal dispatch IDs");
   download(dispatch_weights.data(),bytes+offsetof(DeviceMoELayerRuntime,topk_weights),
    dispatch_weights.size()*sizeof(dispatch_weights[0]),"epoch internal dispatch weights");
   download(&route_participants,bytes+offsetof(DeviceMoELayerRuntime,route_participant_ids),
    sizeof(route_participants),"epoch internal participant storage");
   require(completion_->join() && route_participants,"retire original dispatch observation");
   download(dispatch_participants.data(),route_participants,
    dispatch_participants.size()*sizeof(dispatch_participants[0]),"epoch internal dispatch participants");
   require(completion_->join(),"retire original participant observation");
   const auto participant=static_cast<std::int32_t>(l.runtime->hostLayerState(0).participant_id);
   for(unsigned slot=0;slot<static_cast<unsigned>(shape_.top_k);++slot) {
    const auto id=static_cast<std::int32_t>(ids_[side][slot]);
    const bool local=live&&source.resident_mask.at(id);
    const auto weight=local?weights_[side][slot]:0.0f;
    require(dispatch_ids[slot]==(local?id:-1) &&
     std::bit_cast<std::uint32_t>(dispatch_weights[slot])==std::bit_cast<std::uint32_t>(weight) &&
     dispatch_participants[slot]==(local?participant:-1),"closed runtime dispatch uses the actual selected fragment bank");
   }
   // Inspection is after retained replay on its exact stream. No graph reader
   // owns these tables in this arm, so every byte must retain its dirty value.
   std::vector<unsigned char> table_bytes(shape_.experts*sizeof(DeviceNativeQuantisedMatrixDesc));
   for(const auto name:names) {
    download(table_bytes.data(),l.workspace->getBuffer(name),table_bytes.size(),"closed replay role storage");
    require(completion_->join(),"retire closed role storage observation");
    require(std::all_of(table_bytes.begin(),table_bytes.end(),[](auto b){return b==0xa5;}),
     "closed selected role table remains dirty across live, empty and reset replay");
   }
  }
  std::vector<unsigned char> reader_output;
  if(side&&hasDescriptorReader()) {
   reader_output.resize(shape_.top_k*kSelectedRoleBytes);
   download(reader_output.data(),selected_descriptors_->gpu_data_ptr(),reader_output.size(),"capture selected descriptor reader");
   require(completion_->join(),"retire selected descriptor reader output");
   if(!live)require(std::all_of(reader_output.begin(),reader_output.end(),[](auto b){return b==0;}),
      "inactive selected reader publishes no descriptor rows");
  }
  if(live) {
   const auto* banks=l.runtime->devicePlacementBanks(0);
   const auto* selected_bank=reinterpret_cast<const std::byte*>(banks+status_.bank);
   download(acquired_header_.data(),selected_bank+offsetof(DeviceMoEPlacementBank,epoch),sizeof(acquired_header_),"epoch actual acquired bank header");
   for(unsigned slot=0;slot<unsigned(shape_.top_k);++slot) {
    const auto expert=unsigned(ids_[side][slot]);
    download(&acquired_[slot],selected_bank+offsetof(DeviceMoEPlacementBank,experts)+expert*sizeof(DeviceMoEExpertDescriptor),
     sizeof(DeviceMoEExpertDescriptor),"epoch actual selected source row");
    if(!side||!retained())for(unsigned role=0;role<3;++role) {
     const auto* table=static_cast<const DeviceNativeQuantisedMatrixDesc*>(l.workspace->getBuffer(names[role]));
     download(&published_[role][slot],table+expert,sizeof(DeviceNativeQuantisedMatrixDesc),"epoch actual selected descriptor publication");
    }
   }
   require(completion_->join(),"retire selected source epoch observations");
   require(acquired_header_[0]==epoch&&acquired_header_[1]==unsigned(shape_.experts),"actual bank/epoch source matches captured release");
   for(unsigned slot=0;slot<unsigned(shape_.top_k);++slot) {
    const auto expert=unsigned(ids_[side][slot]);const auto& row=acquired_[slot];
    require(row.logical_expert_id==int(expert),"actual selected bank logical identity");
    if(source.resident_mask.at(expert)) {
     require(row.weight_format==DeviceMoEWeightFormat::NativeQuantised,"actual local original native source");
     const std::array refs{&row.gate,&row.up,&row.down};
     if(!side||!retained())for(unsigned role=0;role<3;++role)require(std::memcmp(refs[role],&published_[role][slot],sizeof(DeviceNativeQuantisedMatrixDesc))==0,
      "full published metadata belongs to the actual acquired source epoch");
    }
    if(side&&hasDescriptorReader()) {
     const auto* selected=reader_output.data()+slot*kSelectedRoleBytes;
     if(source.resident_mask.at(expert)) {
      const std::array refs{&row.gate,&row.up,&row.down};
      for(unsigned role=0;role<3;++role)require(std::memcmp(refs[role],
          selected+role*sizeof(DeviceNativeQuantisedMatrixDesc),sizeof(DeviceNativeQuantisedMatrixDesc))==0,
          "actual outside graph reader sees exact selected descriptor bytes");
     } else require(std::all_of(selected,selected+kSelectedRoleBytes,[](auto b){return b==0;}),
         "outside graph reader skips nonlocal selected descriptor rows");
    }
   }
  }
  require(demand_[side]==expected_demand_&&local_[side]==expected_local_,
   "each actual captured launch has exact selected/local histogram delta, despite shared runtime");
 }
 /** @brief Compare immutable public values; empty rows have no live suffix scratch extent. */
 void exact(bool live) {
  for(unsigned plane:{0u,1u,2u,5u,7u})
   require(same(values_[0][plane],values_[1][plane]),"epoch exact original staged/full public plane "+std::to_string(plane));
  require(same(ids_[0],ids_[1])&&same(weights_[0],weights_[1]),"epoch exact original staged/full routing");
  require(same(injection_[0],injection_[1]),"original independent FP32 injection survives complete local lowering");
  require(same(paired_write_[0],paired_write_[1]),"original paired write consumes the correctly ordered injection and sum");
 }
public:
 /**
  * @brief Capture both genuine programs before the real source publication changes epoch.
  * @param device Original target GPU. @param tp Original singleton source collective.
  * @param source Published initial source lease, retaining actual runtime/arena.
  * @param shared Original shared weights. @param partition Their retained declaration.
  * @param store Original prepared source owner. @param shape Actual physical fabric geometry.
  */
 FullEpochNumericalPair(DeviceId device,std::shared_ptr<ILocalTPContext> tp,std::shared_ptr<PublishedFFNEpoch> source,
  const FrozenModelWeightSet& shared,const TensorParallelConfig& partition,PreparedWeightStore& store,
  OriginalFFNFixtureGeometry shape,EpochNumericalBody body=EpochNumericalBody::OriginalNative):device_(device),worker_(GPUDeviceContextPool::instance().getContext(device)),
  tp_(std::move(tp)),initial_(std::move(source)),shape_(shape),body_(body),widths_{4u*shape.hidden,4u*shape.hidden,R,R,4u*shape.hidden,
   std::size_t(shape.hidden),std::size_t(shape.experts),std::size_t(shape.hidden),std::size_t(shape.top_k)*shape.hidden,
   std::size_t(shape.hidden),std::size_t(shape.hidden),0,0} {
  require(initial_&&initial_->epoch()==1&&initial_->runtime&&initial_->arena&&tp_&&tp_->degree()>0,
   "epoch numerical replay retains actual initially published original GPU cohort");
  const auto& original_source=initial_->bank->layerForModelLayer(0);
  require(original_source.fragment_family&&original_source.fragment_family->cohort()->participants().size()==std::size_t(tp_->degree()),
   "retained epoch GPU width matches actual original cohort");
  member_=original_source.fragment_family->participantIndex();
  if(tp_->degree()==4&&shape_.contract==OriginalFFNGeometryContract::QFNTensorSharded) {
   const auto region=MoEIntermediateShardDescriptorDeclaration::fromOriginalFamily(original_source.fragment_family);
   const auto& routed=region.originalBinding();
   const auto& local_shared=partition.forRank(int(member_));
   require(routed.intermediateColumns()==640&&local_shared.d_ff_count==160&&
    (routed.localIntermediateColumns()==128||routed.localIntermediateColumns()==256),
    "actual full TP4 motion has the production local widths");
   std::osyncstream(std::cout)<<"QFN_MOTION_ACTUAL_TP4_SHARD,member="<<member_<<",device="<<device_.toString()
    <<",F="<<routed.localIntermediateColumns()<<",F_start="<<routed.firstIntermediateColumn()
    <<",S="<<local_shared.d_ff_count<<",S_start="<<local_shared.d_ff_start<<'\n';
  }
  require(member_<tp_->devices().size()&&tp_->devices()[member_].toLocalDeviceId()==device_,
   "retained epoch preserves actual original collective member order");
  const auto* gate=shared.optionalLayerForDevice(0,"ffn_gate_shexp.weight",device);
  const auto* up=shared.optionalLayerForDevice(0,"ffn_up_shexp.weight",device);
  require(gate&&gate->tensor&&up&&up->tensor,"retained original shared slice output extents");
  widths_[11]=gate->tensor->rows();widths_[12]=up->tensor->rows();
#if !defined(LLAMINAR_CASTAGNA_FAST_EPOCH)
  require(body_==EpochNumericalBody::OriginalNative,"retained bank proof is not compiled in this test owner");
#endif
  input_post_.resize(4u*shape.hidden);input_block_.resize(shape.hidden);input_gate_.resize(4);
  if(retained())for(unsigned side=0;side<2;++side) {
   injection_[side].resize(4);paired_write_[side].resize(4u*shape.hidden);
  }
  expected_demand_.resize(shape.experts);expected_local_.resize(shape.experts);
  acquired_.resize(shape.top_k);for(auto& values:published_)values.resize(shape.top_k);
  for(unsigned side=0;side<2;++side){for(unsigned p=0;p<Planes;++p)values_[side][p].resize(widths_[p]);
   ids_[side].resize(shape.top_k);weights_[side].resize(shape.top_k);demand_[side].resize(shape.experts);local_[side].resize(shape.experts);}
  try { worker_.submitAndWait([&] {
   stream_=worker_.getOrCreateAuxiliaryStream("full_native_original_epoch_numerical");
   require(stream_,"epoch full exact auxiliary stream");
   completion_=std::make_unique<ScopedGPUCompletionEvent>(getBackendFor(device_),device_,stream_);
   terminal_event_=getBackendFor(device_)->createEvent(device_.ordinal);require(terminal_event_,"epoch inference boundary event owner");
   context_=IDeviceContext::create(device_,1);
   const auto& bank=initial_->bank->layerForModelLayer(0);
   full_function_=fullProductionHCFunction(bank,shared,store,device_,0,retained()&&!hasDescriptorReader());
   for(unsigned side=0;side<2;++side) {
    auto& p=prefix_[side];const unsigned H=shape.hidden,K=4*H;
    p.residual=fullProductionHCFloat(K,device_,stream_);p.previous=fullProductionHCFloat(H,device_,stream_);
    p.raw=fullProductionHCFloat(4,device_,stream_);p.gain=fullProductionHCFloat(K,device_,stream_);
    for(unsigned i=0;i<K;++i)p.gain->mutable_data()[i]=.75f+float(i%17)/64.f;
    TransferEngine::prepareDeviceInput(p.gain.get(),device_,stream_);
    p.post=fullProductionHCFloat(K,device_,stream_);p.normalized=fullProductionHCFloat(K,device_,stream_);
    p.down=fullProductionHCFloat(R,device_,stream_);p.activated=fullProductionHCFloat(R,device_,stream_);
    p.logits=fullProductionHCFloat(K,device_,stream_);
    p.down_source=TestTensorFactory::createQ8_0Random({R,K},864411u);
    p.up_source=TestTensorFactory::createQ8_0Random({K,R},864412u);
    p.down_prepared=makeGpuPreparedGemm(p.down_source.get(),device_,"epoch.hc.down");
    p.up_prepared=makeGpuPreparedGemm(p.up_source.get(),device_,"epoch.hc.up");
    p.down_prepared.orchestrator->finalize();p.up_prepared.orchestrator->finalize();
    const bool retain_paired_injection=retained();
    if(retain_paired_injection) {
     p.injection=p.raw;
     p.next_trunk=fullProductionHCFloat(K,device_,stream_);
     p.inject_source=TestTensorFactory::createFP32Random({4,size_t(K)},-.125f,.125f,864394u);
     p.inject_prepared=makeGpuPreparedFloatingPointGemm(p.inject_source.get(),device_,"epoch.original.hc.injection");
     p.inject_prepared.orchestrator->finalize();
    }
    const bool owned_retained=side&&retained();
    lane_[side]=makeLane(1,false,device_,member_,*tp_,worker_,*context_,stream_,bank,shared,partition,store,
     side?MoEIntermediateShardFFNExecution::ResidentCandidate:MoEIntermediateShardFFNExecution::Staged,
     MoEGateUpPublicationPolicy::Staged,OriginalM1HiddenQ8FixturePolicy::CountedOriginal,{},{},
     MoEDownPublicationPolicy::Staged,CompletionSerialFamilyReplay::Disabled,shape,bool(side)&&!owned_retained,NativeRouterAliasProbe::None,
     initial_.get(),OriginalFFNCollectiveWitness::ProductionPayloadOnly,
     [&](Lane& l,std::map<BufferId,TensorBase*>& arenas){
        appendFullProductionHCPrefix(l,p,arenas,device_,
            retain_paired_injection?FullProductionHCProbe::MiddleRead:FullProductionHCProbe::FinalRead,H);
        if(owned_retained)lowerRetainedM1HCFFN(l.graph,"ab_previous_write");
        if(owned_retained&&hasDescriptorReader()) {
         selected_descriptors_=TestTensorFactory::createINT32({(shape_.top_k*kSelectedRoleBytes+3)/4});
         require(selected_descriptors_->ensureOnDevice(device_,stream_),"admit selected descriptor reader storage");
         l.graph.addNode("actual_selected_descriptor_reader",std::make_unique<SelectedDescriptorObserver>(
          device_,*l.runtime,shape_.experts,shape_.top_k,*selected_descriptors_),device_);
         l.graph.addDependency("actual_selected_descriptor_reader","shared");
         l.graph.addDependency("promoted_epoch_release","actual_selected_descriptor_reader");
        }
     },
     owned_retained?FullProductionHCObservation::DemandedOnly:FullProductionHCObservation::ObservedAll,
     PublishedFFNEpochAdmission::InitialPublished,
     owned_retained?EpochGraphLowering::ModelRetained:EpochGraphLowering::FixtureNativeLayout);
    auto& l=*lane_[side];capture_[side]=worker_.createGraphCapture(stream_);
    for(const auto& name:l.graph.getExecutionOrder()) {auto* stage=l.graph.getNode(name)->stage.get();
     if(auto* consumer=dynamic_cast<IWorkspaceConsumer*>(stage))consumer->bindWorkspace(l.workspace.get());
     require(stage->prepareGraphLaunch(context_.get(),stream_),"prepare actual epoch full/staged stage");}
   }
   const auto* bytes=reinterpret_cast<const std::byte*>(initial_->runtime->deviceLayerState(0));
   download(expected_demand_.data(),bytes+offsetof(DeviceMoELayerRuntime,decode_histogram),shape.experts*sizeof(uint64_t),"cold real selected baseline");
   download(expected_local_.data(),bytes+offsetof(DeviceMoELayerRuntime,decode_local_histogram),shape.experts*sizeof(uint64_t),"cold real local baseline");
   require(completion_->join(),"retire original epoch capture/setup");
  }); } catch(...) {retire();throw;}
 }
 /** @brief Record and instantiate every original member after the entire cohort is cold. */
 static void captureOriginalCohort(std::span<const std::shared_ptr<FullEpochNumericalPair>> pairs) {
  require(!pairs.empty()&&pairs.size()==std::size_t(pairs.front()->tp_->degree()),"complete actual numerical capture roster");
  const auto& native=pairs.front()->tp_;
  for(unsigned side=0;side<2;++side) {
   std::vector<OriginalNativeGraphRecording> recordings;
   std::vector<std::unique_ptr<GraphCaptureDependencyLedger>> dependencies;
   for(const auto& pair:pairs) {
    require(pair->tp_==native,"same original native context for each capture member");
    dependencies.push_back(std::make_unique<GraphCaptureDependencyLedger>(pair->device_,pair->stream_,
     pair->lane_[side]->plan,"original_full_epoch_numerical",
     GraphCaptureDependencyLedger::ExternalInputAuthority::BindDeclaredAddressesOnly));
    recordings.push_back({.worker=GPUDeviceContextPool::instance().retainExistingContext(pair->device_),.device=pair->device_,
     .capture=pair->capture_[side].get(),.dependencies=dependencies.back().get(),
     .record=[&member=*pair,side]{member.recordStages(side);}});
   }
   const auto context=pairs.size()>1?std::optional{native->borrowOriginalNativeContext()}:std::optional<NativeLocalTPContextBorrow>{};
   OriginalNativeGraphCaptureGroup::record(recordings,context,"original_full_epoch_numerical_arm_"+std::to_string(side));
   for(const auto& pair:pairs)pair->worker_.submitAndWait([&]{pair->instantiateArm(side);});
  }
 }
 /** @brief Replay the real initial bank and retire its setup-only host lease. */
 void replayInitial() {
  replay(*initial_,false);
  releaseInitialHostBank();
 }
private:
 /** @brief Record the original stage order on the capture group's exact worker stream. */
 void recordStages(unsigned side) {
    auto& l=*lane_[side];
    for(const auto& name:l.graph.getExecutionOrder()) {auto* stage=l.graph.getNode(name)->stage.get();ScopedGraphCaptureStage scope(stage);
     require(stage->execute(context_.get()),"capture actual epoch full/staged stage");
     scope.complete();}
 }
 /** @brief Instantiate and authenticate the definition after all native members finish recording. */
 void instantiateArm(unsigned side) {
    auto& l=*lane_[side];auto& p=prefix_[side];
    require(capture_[side]->instantiate(),"instantiate original retained epoch graph");++captures_built_;
    if(side) {
     if(retained()) {
      std::vector<GPUGraphKernelNodeInfo> nodes;std::string error;
      require(capture_[side]->inspectKernelNodes(nodes,&error),"inspect retained fast source epoch graph: "+error);
      const auto fast_count=std::count_if(nodes.begin(),nodes.end(),[&](const auto& node){
       return node.function_identity==reinterpret_cast<std::uintptr_t>(full_function_);});
      require(fast_count==1&&nodes.size()>=3,"actual canonical fast full HC/FFN kernel and original native epoch boundaries");
     }
     requireFullProductionHCBinding(*capture_[side],full_function_,p,l);
    }
 }
 /** @brief Retire partial construction and normal state through the same original worker authority. */
 void retire() noexcept {try{worker_.submitAndWait([&]{if(completion_)require(completion_->join(),"retire full epoch owner");
  for(auto& c:capture_)c.reset();for(auto& l:lane_)if(l){l->stream=nullptr;l->worker=nullptr;l.reset();}
  selected_descriptors_.reset();prefix_={};context_.reset();
  if(terminal_event_){getBackendFor(device_)->destroyEvent(terminal_event_,device_.ordinal);terminal_event_=nullptr;}
  completion_.reset();});}catch(...){std::terminate();}}
public:
 /** @brief Retire all work before any capture, tensor, source or original context is released. */
 ~FullEpochNumericalPair(){retire();}
 /**
  * @brief Order real maintenance after this pair's last committed inference terminal.
  * @param maintenance_stream Exact same-device producer supplied by the original publisher.
  * @return True only after the actual inference event wait is enqueued.
  */
 bool enqueueMaintenanceBoundary(void* maintenance_stream) const {
  return reinterpret_cast<std::uintptr_t>(maintenance_stream)>2&&terminal_recorded_&&terminal_event_&&
   getBackendFor(device_)->streamWaitEvent(maintenance_stream,terminal_event_,device_.ordinal);
 }
 /** @brief Drop the setup-only old host source borrow after real captured epoch1 readers released. */
 void releaseInitialHostBank() {
  require(terminal_recorded_&&initial_->bank&&initial_->epoch()==1,"initial source retires after actual captured readers");
  initial_->bank={};
 }
 /**
  * @brief Replay unchanged captures against the current genuine bank, with dirty/repeat/reset/empty checks.
  * @param current Real installed source lease; producer publication completed before this call.
  * @param require_change Require the actual new local route to change combined numerical bytes.
  */
 void replay(const PublishedFFNEpoch& current,bool require_change) {
  const auto epoch=current.epoch();const auto& source=current.bank->layerForModelLayer(0);
  require(current.runtime==initial_->runtime&&current.arena==initial_->arena&&captures_built_==2,
   "epoch switch retains exact runtime, arena and both original graph instances");
  terminal_recorded_=false;
  worker_.submitAndWait([&] {
   const auto submit=[&](unsigned side) {
    auto& l=*lane_[side];
    if(tp_->degree()>1)require(getBackendFor(device_)->memset(l.receipt->gpu_data_ptr(),0,l.receipt->size_bytes(),
        device_.ordinal,stream_),"reset original counted collective receipt on exact stream");
    if(side&&retained()) {
     const std::array names{MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE,
    MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_UP_DESC_TABLE,MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE};
     for(const auto name:names)require(getBackendFor(device_)->memset(l.workspace->getBuffer(name),0xa5,
         shape_.experts*sizeof(DeviceNativeQuantisedMatrixDesc),device_.ordinal,stream_),
         "poison closed selected descriptor tables before native replay");
    }
    require(capture_[side]->launch(),"replay retained original epoch graph");
   };
   for(unsigned history=0;history<20;++history) {
    for(unsigned repeat=0;repeat<2;++repeat){
     // The actual paired write consumes an injection output that reuses the
     // exact incoming gate tensor. Reauthor the original external request
     // bytes before comparing repeated replays of the same input.
     publish(history==13?0:history,true);
     for(unsigned side=0;side<2;++side){
     submit(side);observe(side,epoch,source,true);
     if(!repeat){first_[side]=values_[side];first_ids_[side]=ids_[side];first_weights_[side]=weights_[side];}
     else {for(unsigned p=0;p<Planes;++p)require(same(first_[side][p],values_[side][p]),"epoch repeat exact public bytes at plane "+std::to_string(p));
      require(same(first_ids_[side],ids_[side])&&same(first_weights_[side],weights_[side]),"epoch repeat exact routes");}}
     exact(true);}
    if(!history) {
     reset_=values_;
     if(require_change){require(!same(before_combined_,values_[1][7]),"genuine promoted route changes full numerical output");
      for(unsigned p=0;p<6;++p)require(same(before_prefix_[p],values_[1][p]),"unchanged original HC prefix across actual source epoch");}
     else {before_combined_=values_[1][7];for(unsigned p=0;p<6;++p)before_prefix_[p]=values_[1][p];}
    }
    if(history==1)require(!same(reset_[1][0],values_[1][0]),"epoch dirty input changes actual full POST bytes");
    if(history==13)for(unsigned side=0;side<2;++side)for(unsigned p=0;p<Planes;++p)
     require(same(reset_[side][p],values_[side][p]),"epoch dirty-to-original reset exact bytes");
   }
   publish(21,false);for(unsigned repeat=0;repeat<2;++repeat){for(unsigned side=0;side<2;++side){
    submit(side);observe(side,epoch,source,false);}exact(false);}
   publish(0,true);for(unsigned side=0;side<2;++side){submit(side);observe(side,epoch,source,true);
    for(unsigned p=0;p<Planes;++p)require(same(reset_[side][p],values_[side][p]),"epoch active-after-empty bytes");}exact(true);
   if(require_change) {
    bool witnessed=false;
    for(unsigned slot=0;slot<unsigned(shape_.top_k);++slot)if(std::find(current.promoted_experts.begin(),
       current.promoted_experts.end(),int(ids_[1][slot]))!=current.promoted_experts.end()) {
     const auto& row=acquired_[slot];
     if(retained()) {
      const auto declaration=MoEIntermediateShardDescriptorDeclaration::fromOriginalFamily(
       current.bank->layerForModelLayer(0).fragment_family);
      require(declaration.matchesNativeGateUp(row.gate,row.up)&&declaration.matchesNativeDown(row.down),
       "genuine GPU arrival executes actual same-source prepared descriptors");
      std::osyncstream(std::cout)<<"QFN_RETAINED_ACQUIRED_ARRIVAL,epoch="<<epoch<<",member="<<member_
       <<",expert="<<int(ids_[1][slot])<<",gate="<<unsigned(row.gate.codebook_id)
       <<",up="<<unsigned(row.up.codebook_id)<<",down="<<unsigned(row.down.codebook_id)
       <<",F="<<row.gate.n<<",down_N="<<row.down.n<<",down_K="<<row.down.k<<'\n';
     } else require(row.down.codebook_id==kNativeQuantisedExpandedInt8MinCodebook&&row.down.payload&&row.down.scales,
      "genuine CPU arrival executes its actual prepared physical23 descriptor");
     witnessed=true;
    }
    require(witnessed,"actual full selected packet includes genuine promoted expert");
   }
   require(getBackendFor(device_)->recordEvent(terminal_event_,device_.ordinal,stream_),"publish exact full inference terminal");
   terminal_recorded_=true;
  });
  std::osyncstream(std::cout)<<"FULL_NATIVE_EPOCH_REPLAY,epoch="<<epoch<<",captures=2,recaptured=0,histories=20,repeated=1,reset=1,empty=2,active_after_empty=1,planes=5,actual_release=1,histogram_delta=1,numerical_change="<<require_change<<",fast_retained="<<(retained())<<",descriptor_reader="<<hasDescriptorReader()<<",degree="<<tp_->degree()
        <<",member="<<member_<<",collective_extent_exact="<<(tp_->degree()>1)<<'\n';
 }
};
