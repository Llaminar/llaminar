/**
 * @file CastagnaFullRegionABReplay.inc
 * @brief Actual exported full Castagna POST versus original full HC/FFN graphs.
 *
 * Initial generated-local source smoke uses original GGUF blocks and the exact
 * fork-admitted F/S widths, not a claim that Llaminar's standard shared TP cut
 * is192. The existing model/WeightManager prepares canonical native weights;
 * the fork reads original source windows separately, never inverse-repacking.
 */

/** @brief Explicit test-only file action; absent variables preserve ordinary replay/timing. */
CastagnaOracleConfig castagnaOracleConfig() {
    const char* action=std::getenv("LLAMINAR_TEST_CASTAGNA_ORACLE_ACTION");
    const char* path=std::getenv("LLAMINAR_TEST_CASTAGNA_ORACLE_PATH");
    if(!action&&!path)return {};
    require(action&&path&&*path,"oracle action and path must be supplied together");
    const std::string mode=action;
    require(mode=="write-baseline"||mode=="compare-baseline","oracle action must be explicit write-baseline or compare-baseline");
    return {mode=="write-baseline"?CastagnaOracleAction::WriteBaseline:CastagnaOracleAction::CompareBaseline,path};
}

/** @brief Explicit immutable input selection; generated diagnostics remain separate. */
enum class CastagnaABSource { Generated, BoundedCentered, RealGGUF };
/** @brief Which predeclared input cohort must agree between the independent native routers. */
enum class CastagnaRouteCohort { AllReplayHistories, TimedRecords };
/** @brief Local probe input record, never a production model flag or ambient environment override. */
struct CastagnaABInput {
    CastagnaABSource source=CastagnaABSource::Generated;
    std::string path;
    unsigned layer=1;
    unsigned participant=3; ///< Logical source-split slot after removing the layer-dependent physical rotation.
    unsigned degree=4; ///< Reference split used to derive both engines' original local source windows.
    CastagnaRouteCohort route_cohort=CastagnaRouteCohort::AllReplayHistories;
    /** @return Whether the real source path was explicitly requested. */
    bool real() const noexcept {return source==CastagnaABSource::RealGGUF;}
    /** @return Canonical layer-local tensor name in the selected original source. */
    std::string name(std::string_view suffix) const {return "blk."+std::to_string(layer)+"."+std::string(suffix);}
};
/** @brief The selected original checkpoint; seeded boundary activations remain diagnostic inputs. */
CastagnaABInput realCastagnaABInput() {
    const auto coordinate = [](const char* key, unsigned initial) {
        const char* value = std::getenv(key);
        if (!value) return initial;
        const std::string_view text(value);
        unsigned result = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);
        require(error == std::errc{} && end == text.data() + text.size(), "invalid original comparator coordinate");
        return result;
    };
    const char* path = std::getenv("LLAMINAR_TEST_CASTAGNA_GGUF");
    const auto layer = coordinate("LLAMINAR_TEST_CASTAGNA_LAYER", 1);
    const auto degree = coordinate("LLAMINAR_TEST_CASTAGNA_TP", 4);
    const auto participant = coordinate("LLAMINAR_TEST_CASTAGNA_LOGICAL_SLOT", 3);
    require(degree && participant < degree, "original comparator coordinate out of range");
    const auto* timing_cohort=std::getenv("LLAMINAR_TEST_CASTAGNA_ROUTE_COHORT");
    require(!timing_cohort || std::string_view(timing_cohort)=="timed_records",
        "unknown original comparator route cohort");
    return {CastagnaABSource::RealGGUF, path && *path ? path :
        "/opt/llaminar-models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
        layer, participant, degree,
        timing_cohort ? CastagnaRouteCohort::TimedRecords : CastagnaRouteCohort::AllReplayHistories};
}

/**
 * @brief Borrow a canonical matrix through the existing prepared-engine authority.
 * @param engine Retained original HC/shared/fragment engine.
 * @param rows Original local output extent. @param columns Original local input extent.
 * @return Exact uploaded descriptor; no weight conversion or policy reconstruction.
 */
DeviceNativeQuantisedMatrixDesc castagnaABNativeMatrix(const ITensorGemm* engine,int rows,int columns) {
    DeviceNativeQuantisedMatrixDesc matrix{};
    require(engine&&engine->weights_converted()&&engine->inspectPreparedNativeQuantisedMatrixDesc(matrix)&&matrix.valid()&&
        matrix.n==rows&&matrix.k==columns&&matrix.blocks_per_row==unsigned(columns/32),
        "adapted Castagna borrows exact original prepared matrix geometry");
    return matrix;
}
/**
 * @brief Copy only immutable descriptor records through the existing PMA tensor/transfer authority.
 * @param records Persistent host record owner, retained through the caller's terminal event.
 * @param storage Persistent destination owner, assigned before the upload can throw.
 * @param device Original participant. @param stream Exact original operation stream.
 * @return Device descriptor table, never a reconstructed weight payload.
 */
const DeviceNativeQuantisedMatrixDesc* castagnaABPublishMatrices(
    const std::vector<DeviceNativeQuantisedMatrixDesc>& records,std::unique_ptr<INT32Tensor>& storage,
    DeviceId device,void* stream) {
    static_assert(sizeof(DeviceNativeQuantisedMatrixDesc)%sizeof(int32_t)==0);
    require(!records.empty(),"complete adapted descriptor table");
    storage=TestTensorFactory::createINT32({records.size()*sizeof(DeviceNativeQuantisedMatrixDesc)/sizeof(int32_t)});
    std::memcpy(storage->mutable_typed_data(),records.data(),records.size()*sizeof(DeviceNativeQuantisedMatrixDesc));
    TransferEngine::prepareDeviceInput(storage.get(),device,stream);
    return static_cast<const DeviceNativeQuantisedMatrixDesc*>(storage->gpu_data_ptr());
}
/** @brief Original raw bytes are copied only by the sealed loader into PMA-owned words. */
std::unique_ptr<INT32Tensor> castagnaABRaw(IModelLoader& loader,const std::string& name,
    DeviceId device,void* stream) {
    const auto source=loader.nativeWeightSourceGeometry(name);
    require(!source.original_shape.empty()&&source.original_shape.size()<=3,
        "fork raw source requires original K[/N/E] tensor");
    const auto K=source.original_shape[0],N=source.original_shape.size()>1?source.original_shape[1]:1;
    const auto E=source.original_shape.size()==3?source.original_shape[2]:1;
    require(K%source.native_group_elements==0,"fork raw source respects original encoded group");
    const auto row=K/source.native_group_elements*source.native_group_bytes;
    const auto bytes=E*N*row;require(bytes%4==0,"fork raw source has exact dword storage");
    auto tensor=TestTensorFactory::createINT32({bytes/4});
    auto* host=reinterpret_cast<std::byte*>(tensor->mutable_typed_data());require(host,"raw source host owner");
    for(std::size_t expert=0;expert<E;++expert) {
        NativeWeightSourceWindow window{.original_expert=expert,.selected_expert_ordinal=expert,
            .first_row=0,.rows=N,.first_column=0,.columns=K,.native_row_bytes=row};
        loader.readNativeWeightSourceWindow(name,window,{host+expert*N*row,N*row});
    }
    // Return only after the original transfer prefix retires: allocation or
    // upload failure must not release a still-borrowed local source owner.
    ScopedGPUCompletionEvent completion(getBackendFor(device),device,stream);
    TransferEngine::prepareDeviceInput(tensor.get(),device,stream);
    require(completion.join(),"cold exact source upload");return tensor;
}
/** @brief Copy existing intrinsic source bytes, not converted values, for fork HC weights. */
std::unique_ptr<INT32Tensor> castagnaABRawTensor(const TensorBase& source,DeviceId device,void* stream) {
    require(source.raw_data()&&source.size_bytes()%4==0,"intrinsic original HC raw storage");
    auto tensor=TestTensorFactory::createINT32({source.size_bytes()/4});
    std::memcpy(tensor->mutable_typed_data(),source.raw_data(),source.size_bytes());
    // Return only after the original transfer prefix retires: allocation or
    // upload failure must not release a still-borrowed local source owner.
    ScopedGPUCompletionEvent completion(getBackendFor(device),device,stream);
    TransferEngine::prepareDeviceInput(tensor.get(),device,stream);
    require(completion.join(),"cold exact source upload");return tensor;
}
/**
 * @brief Populate the local GGUF from the original fork partition at the selected TP degree.
 * @param local Destination directory with the real local dimensions.
 * @param format Original source factories; never a prepared byte reconstruction.
 * @param shape Local geometry; selected participant is found from source-owned split rules.
 * @details This is generated-source structural evidence, not model-weight parity.
 * The unchanged full source supplies every byte; both engines consume the same
 * selected native blocks. No partialK Q4_K group is cut by down selection.
 */
void writeCastagnaABOriginalWindows(PlanningGGUFFixture& local,const SourceFormat& format,
    OriginalFFNFixtureGeometry shape,const CastagnaABInput& input={}) {
    unsigned participant=input.degree;
    for(unsigned slot=0;slot<input.degree;++slot) {
        const auto routed=qfn_castagna_reference::equalWeightSourceShard(640,128,input.degree,slot);
        const auto shared=qfn_castagna_reference::equalWeightSourceShard(640,64,input.degree,slot);
        if ((!input.real() || input.participant == slot) &&
            routed.count == unsigned(shape.routed) && shared.count == unsigned(shape.shared)) participant = slot;
    }
    require(participant<input.degree,"selected local shape belongs to the original fork split");
    const auto routed=qfn_castagna_reference::equalWeightSourceShard(640,128,input.degree,participant);
    const auto shared=qfn_castagna_reference::equalWeightSourceShard(640,64,input.degree,participant);
    require(format.down_quantized, "comparison requires an independently declared original routed-down encoding");
    PlanningGGUFFixture full(true,false,PlanningGGUFFixture::sourceType(format.type),640,
        std::nullopt,8,PlanningGGUFFixture::Geometry::Directory,1,512,GGUFTensorType::Q8_0,
        std::array{PlanningGGUFFixture::sourceType(format.type), PlanningGGUFFixture::sourceType(format.type),
            PlanningGGUFFixture::sourceType(format.down_quantized->tensor_type)},
        PlanningGGUFFixture::MoEDimensions{2560,640,10});
    auto full_shape=shape;full_shape.routed=640;full_shape.shared=640;
    if(!input.real()) {
        auto authored=format;
        if(input.source==CastagnaABSource::BoundedCentered) {
            require(format.type==TensorType::Q4_K&&!format.up_quantized,
                "bounded centered source owns only original Q4_K gate/up");
            authored.create=[](const auto& shape,std::uint32_t seed) {
                return makeBoundedCenteredQ4K(shape,seed);
            };
        }
        writeOriginalSources(full,authored,full_shape);
    }
    ModelLoader loader;loader.setUseMmap(false);
    require(loader.loadModel(input.real()?input.path:full.path()),"original full source window reader");
    if(input.real())require(participant==input.participant,"real source participant matches explicit selected local shape");
    const auto original_name=[&](const std::string& local_name) {
        return input.real()?input.name(local_name.substr(std::string("blk.0.").size())):local_name;
    };
    for(unsigned role=0;role<3;++role)for(unsigned expert=0;expert<512;++expert) {
        const std::string name=kRoutedNames[role],source_name=original_name(name);
        const auto source=loader.nativeWeightSourceGeometry(source_name);
        require(source.original_type==(role==2?format.down_quantized->tensor_type:format.type)&&
            source.original_shape==std::vector<size_t>({role==2?size_t(640):size_t(2560),role==2?size_t(2560):size_t(640),512}),
            "routed source must retain the declared original encoding and complete dimensions");
        const bool down=role==2;
        NativeWeightSourceWindow window{.original_expert=expert,.selected_expert_ordinal=expert,
            .first_row=down?0:routed.first,.rows=down?2560:routed.count,
            .first_column=down?routed.first:0,.columns=down?routed.count:2560,
            .native_row_bytes=(down?routed.count:2560)/source.native_group_elements*source.native_group_bytes};
        auto bytes=TestTensorFactory::createINT32({(window.nativeBytes()+3)/4});
        auto* storage=reinterpret_cast<std::byte*>(bytes->mutable_typed_data());
        loader.readNativeWeightSourceWindow(source_name,window,{storage,window.nativeBytes()});
        local.writePayload(name,expert*window.nativeBytes(),{reinterpret_cast<const uint8_t*>(storage),window.nativeBytes()});
    }
    for(unsigned role=0;role<kSharedNames.size();++role) {
        const std::string name=kSharedNames[role],source_name=original_name(name);
        const auto source=loader.nativeWeightSourceGeometry(source_name);
        require(source.original_type==(role<3?TensorType::Q8_0:TensorType::FP32),
            "real shared source keeps original Q8_0 and FP32 scalar/router policy");
        const auto expected=role==2?std::vector<size_t>{640,2560}:role<2?std::vector<size_t>{2560,640}:
            role==3?std::vector<size_t>{2560}:std::vector<size_t>{2560,512};
        require(source.original_shape==expected ||
            (role==3&&source.original_shape==std::vector<size_t>{2560,1}),
            "real shared/router source retains exact original full geometry");
        const bool projected=role<3,down=role==2;
        const auto rows=projected?(down?size_t(2560):shared.count):(role==3?size_t(1):size_t(512));
        const auto columns=projected?(down?shared.count:size_t(2560)):size_t(2560);
        NativeWeightSourceWindow window{.first_row=projected&&!down?shared.first:0,.rows=rows,
            .first_column=down?shared.first:0,.columns=columns,
            .native_row_bytes=columns/source.native_group_elements*source.native_group_bytes};
        auto bytes=TestTensorFactory::createINT32({(window.nativeBytes()+3)/4});
        auto* storage=reinterpret_cast<std::byte*>(bytes->mutable_typed_data());
        loader.readNativeWeightSourceWindow(source_name,window,{storage,window.nativeBytes()});
        local.writePayload(name,0,{reinterpret_cast<const uint8_t*>(storage),window.nativeBytes()});
    }
    std::osyncstream(std::cout)<<"CASTAGNA_FULL_AB_SOURCE,generated="<<!input.real()<<",original_F=640,original_S=640,tp="<<input.degree<<",participant="
        <<participant<<",bounded_centered="<<(input.source==CastagnaABSource::BoundedCentered)
        <<",real_gguf="<<input.real()<<",layer="<<input.layer<<",routed_first="<<routed.first<<",routed_count="<<routed.count
        <<",shared_first="<<shared.first<<",shared_count="<<shared.count
        <<",gate="<<tensorTypeName(format.type)<<",down="<<tensorTypeName(format.down_quantized->tensor_type)<<"\n";
}

/**
 * @brief Capture full equal external boundaries using the actual existing local owner.
 * @param device Original singleton ROCm participant. @param member Native TP coordinate.
 * @param tp Original collective owner, excluded from measured local boundary.
 * @param rendezvous Existing owner lifecycle barrier. @param banks Original resident source banks.
 * @param original_member Canonical source coordinate. @param shared Original shared sources.
 * @param partition Original shared partition declaration. @param store Existing prepared store.
 * @param loader Same original GGUF reader. @param shape Admitted local source shape.
 * @param timing Explicit whole-parent economy observation; never changes numerical work.
 * @param input Original checkpoint or explicitly generated source declaration.
 * @param phase_observation Optional existing per-CTA HTS outputs, outside economy observation.
 * @param production Explicit real production full-entry lowering; default retains the existing four arms.
 */
void replayCastagnaFullRegionAB(DeviceId device,int member,ILocalTPContext& tp,std::barrier<>& rendezvous,
 MoEOverlayParticipantResidencyRegistry& banks,int original_member,const FrozenModelWeightSet& shared,
 const TensorParallelConfig& partition,PreparedWeightStore& store,IModelLoader& loader,
 OriginalFFNFixtureGeometry shape,bool timing,const CastagnaABInput& input={},
 CastagnaPhaseObservation phase_observation=CastagnaPhaseObservation::Disabled,
 FullProductionHCProbe production=FullProductionHCProbe::Disabled,
 FullProductionHCObservation observation=FullProductionHCObservation::ObservedAll) {
    require(production!=FullProductionHCProbe::MiddleRead||!timing,
        "middle-read injection is functional coverage, not equal-work fork timing");
    const bool full_production=production!=FullProductionHCProbe::Disabled;
    const bool observed_all=observation==FullProductionHCObservation::ObservedAll;
    const bool retained=observation==FullProductionHCObservation::RetainedDemandedOnly;
    const bool up_observed=observation!=FullProductionHCObservation::DemandedOnly&&!retained;
    require(!retained||(production==FullProductionHCProbe::FinalRead&&timing&&input.real()),
        "retained comparison requires the real ordinary final-read capture");
    const unsigned arm_count=retained?3u:4u;
    require(observation!=FullProductionHCObservation::OutsideHCUpReader,"outside reader uses its explicit original-source functional fixture");
    const bool observe_phases=phase_observation==CastagnaPhaseObservation::PerCTA;
    require(!observe_phases||(!timing&&input.real()),"HTS requires explicit real-weight diagnostic, never timing");
    require(device.is_rocm()&&tp.degree()==1&&member==0&&shape.hidden==2560&&shape.experts==512&&
        shape.top_k==10&&shape.routed%64==0&&shape.routed<=512&&shape.shared%64==0,
        "full Castagna comparator admits exact local original shape, no padding");
    const auto oracle_config=castagnaOracleConfig();
    require(oracle_config.action==CastagnaOracleAction::Disabled||(!full_production&&!timing&&input.real()),
        "exact B oracle requires explicit real-weight replay, never timing or generated input");
    auto& worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&] {
        auto* stream=worker.getOrCreateAuxiliaryStream("castagna_full_region_original");
        auto* backend=getBackendFor(device);require(backend,"full-region backend");
        auto context=IDeviceContext::create(device,1);
        constexpr unsigned H=2560,K=10240,R=320,guards=16;
        const float guard=-31789.25f;
        std::unique_ptr<ModelLoader> real_loader;
        if(input.real()) {
            real_loader=std::make_unique<ModelLoader>();real_loader->setUseMmap(false);
            require(real_loader->loadModel(input.path),"explicit original real GGUF source");
        }
        std::array<FullProductionHCPrefix,2> prefix;
        std::array<std::unique_ptr<Lane>,2> lanes;
        std::array<std::unique_ptr<IGPUGraphCapture>,4> captures,children,timed;
        std::vector<std::unique_ptr<INT32Tensor>> raw;
        std::array<std::shared_ptr<FP32Tensor>,5> fork_outputs,adapted_outputs;
        std::array<std::vector<DeviceNativeQuantisedMatrixDesc>,3> adapted_host_tables;
        std::array<std::unique_ptr<INT32Tensor>,3> adapted_device_tables;
        std::array<DeviceNativeQuantisedMatrixDesc,5> adapted_fixed;
        std::array<std::array<std::vector<float>,5>,4> observed,baseline;
        // Persistent failure observations precede the exact terminal guard too.
        std::array<std::vector<float>,3> prefix_diagnostics;
        prefix_diagnostics[0].resize(R);prefix_diagnostics[1].resize(R);prefix_diagnostics[2].resize(K);
        // Every async diagnostic destination exists before the terminal guard.
        std::array<std::array<std::vector<float>,5>,2> hc_observed,hc_baseline,hc_reset;
        std::array<std::vector<float>,2> hc_omitted_poison;
        hc_omitted_poison[0].resize(R);hc_omitted_poison[1].resize(K);
        for(unsigned plane=0;plane<2;++plane)for(unsigned i=0;i<hc_omitted_poison[plane].size();++i)
            hc_omitted_poison[plane][i]=-12288.f-float(32*plane)-float(i%7)/16.f;
        if(full_production)for(auto& values:hc_observed) {
            values[0].resize(R);values[1].resize(R);values[2].resize(K);
            if(production==FullProductionHCProbe::MiddleRead){values[3].resize(4);values[4].resize(K);}
        }
        // Expected authored fixture input, never a live production count mirror.
        std::array<std::int32_t,2> observed_active_count{};
        std::int32_t expected_authored_active_count=1;
        unsigned current_history=0;
        const char* current_submission="initial";
        std::array<std::vector<float>,2> route_ids,route_weights;
        std::unique_ptr<INT32Tensor> fork_ids;
        std::shared_ptr<FP32Tensor> fork_weights,adapted_weights;
        std::unique_ptr<INT32Tensor> adapted_ids;
        std::vector<int32_t> observed_fork_ids(shape.top_k);
        std::vector<float> observed_fork_weights(shape.top_k),observed_adapted_weights(shape.top_k);
        std::vector<int32_t> observed_adapted_ids(shape.top_k);
        std::array<std::array<std::vector<float>,5>,4> reset_baseline;
        std::array<std::vector<float>,5> input_words;
        input_words[0].resize(K);input_words[1].resize(H);input_words[2].resize(4);
        input_words[3].resize(K);input_words[4].resize(1);
        std::array<std::vector<float>,3> fork_failure_values;
        std::array<const qfn_castagna_reference::GlobalRegion*,3> fork_failure_regions{};
        qfn_castagna_reference::GlobalRegions globals{};
        qfn_castagna_reference::Request request{};
        qfn_castagna_reference::Admission admission{};
        qfn_castagna_native::Request adapted_request{};
        qfn_castagna_native::Admission adapted_admission{};
        qfn_castagna_reference::GlobalRegions adapted_globals{};
        decltype(banks.initialJointLayer(original_member,0)) retained_frame;
        decltype(retained_frame->acquireInitialConstruction()) retained_setup;
        // Device outputs and asynchronous readback destinations outlive completion,
        // captures and all submissions, including exception unwinding.
        constexpr std::size_t timestamp_ctas=60,timestamp_slots=16,timestamp_guards=8;
        std::array<std::unique_ptr<INT32Tensor>,2> timestamp_storage;
        std::array<std::vector<unsigned long long>,2> timestamp_host;
        ScopedGPUCompletionEvent completion(backend,device,stream);
        // All destinations/borrowers precede this guard. Lane's legacy teardown
        // is disabled below after transferring exact stream retirement here.
        struct Retire {ScopedGPUCompletionEvent& event;std::array<std::unique_ptr<Lane>,2>& lanes;
            ~Retire(){if(!event.join())std::terminate();for(auto& lane:lanes)if(lane){lane->stream=nullptr;lane->worker=nullptr;}}} retire{completion,lanes};
        for(unsigned side=0;side<2;++side) {
            auto& p=prefix[side];
            p.residual=fullProductionHCFloat(K,device,stream);p.previous=fullProductionHCFloat(H,device,stream);
            p.raw=fullProductionHCFloat(4,device,stream);p.gain=fullProductionHCFloat(K,device,stream);
            for(unsigned i=0;i<K;++i)p.gain->mutable_data()[i]=.75f+float(i%17)/64.f;
            TransferEngine::prepareDeviceInput(p.gain.get(),device,stream);
            require(completion.join(),"retire immutable gain publication");
            p.post=fullProductionHCFloat(K,device,stream);p.normalized=fullProductionHCFloat(K,device,stream);
            p.down=fullProductionHCFloat(R,device,stream);p.activated=fullProductionHCFloat(R,device,stream);
            p.logits=fullProductionHCFloat(K,device,stream);
            if(input.real()) {
                const auto down_name=input.name("hc_ffn_down.weight"),up_name=input.name("hc_ffn_up.weight");
                const auto down_geometry=real_loader->nativeWeightSourceGeometry(down_name);
                const auto up_geometry=real_loader->nativeWeightSourceGeometry(up_name);
                require(down_geometry.original_type==TensorType::Q8_0&&up_geometry.original_type==TensorType::Q8_0&&
                    down_geometry.original_shape==std::vector<size_t>{K,R}&&
                    up_geometry.original_shape==std::vector<size_t>{R,K},
                    "real HC projection source must retain exact original Q8_0 geometry");
                p.down_source=real_loader->loadTensor(down_name,DeviceId::cpu(),WeightPrecision::NATIVE);
                p.up_source=real_loader->loadTensor(up_name,DeviceId::cpu(),WeightPrecision::NATIVE);
                const auto norm_geometry=real_loader->nativeWeightSourceGeometry(input.name("hc_ffn_norm.weight"));
                require(norm_geometry.original_type==TensorType::FP32&&norm_geometry.original_shape==std::vector<size_t>{K},
                    "selected real checkpoint norm is original FP32 vector");
                auto gain=real_loader->loadTensor(input.name("hc_ffn_norm.weight"),DeviceId::cpu(),WeightPrecision::NATIVE);
                require(gain&&gain->numel()==K&&gain->native_type()==TensorType::FP32,"original real HC gain geometry/policy");
                std::memcpy(p.gain->mutable_data(),gain->raw_data(),K*sizeof(float));
                TransferEngine::prepareDeviceInput(p.gain.get(),device,stream);
                require(completion.join(),"retire original real HC gain publication");
            } else {
                p.down_source=TestTensorFactory::createQ8_0Random({R,K},864411u);
                p.up_source=TestTensorFactory::createQ8_0Random({K,R},864412u);
            }
            p.down_prepared=makeGpuPreparedGemm(p.down_source.get(),device,"ab.hc.down");
            p.up_prepared=makeGpuPreparedGemm(p.up_source.get(),device,"ab.hc.up");
            p.down_prepared.orchestrator->finalize();p.up_prepared.orchestrator->finalize();
            if(production==FullProductionHCProbe::MiddleRead) {
                p.injection=fullProductionHCFloat(4,device,stream);p.next_trunk=fullProductionHCFloat(K,device,stream);
                if(input.real()) {
                    const auto name=input.name("hc_ffn_inject.weight");
                    const auto geometry=real_loader->nativeWeightSourceGeometry(name);
                    require(geometry.original_type==TensorType::FP32&&geometry.original_shape==std::vector<size_t>{K,4},
                        "middle read requires original FP32 injection [4H,4]");
                    p.inject_source=real_loader->loadTensor(name,DeviceId::cpu(),WeightPrecision::NATIVE);
                } else p.inject_source=TestTensorFactory::createFP32Random({4,size_t(K)},-.125f,.125f,864394u);
                p.inject_prepared=makeGpuPreparedFloatingPointGemm(p.inject_source.get(),device,"ab.hc.injection");
                p.inject_prepared.orchestrator->finalize();
            }
        }
        retained_frame=banks.initialJointLayer(original_member,0);
        require(bool(retained_frame),"original complete source bank");
        retained_setup=retained_frame->acquireInitialConstruction();
        require(bool(retained_setup),"original source construction lease");
        const auto& source=retained_setup->layerForModelLayer(0);
        for(unsigned side=0;side<2;++side) {
            lanes[side]=makeLane(1,false,device,member,tp,worker,*context,stream,source,shared,partition,store,
                side?MoEIntermediateShardFFNExecution::ResidentCandidate:MoEIntermediateShardFFNExecution::Staged,
                MoEGateUpPublicationPolicy::Staged,OriginalM1HiddenQ8FixturePolicy::CountedOriginal,{},{},
                MoEDownPublicationPolicy::Staged,CompletionSerialFamilyReplay::Disabled,shape,side&&!retained,
                NativeRouterAliasProbe::None,nullptr,OriginalFFNCollectiveWitness::ProductionPayloadOnly,
                [&](Lane& lane,std::map<BufferId,TensorBase*>& arenas){
                    appendFullProductionHCPrefix(lane,prefix[side],arenas,device,
                        side&&retained?FullProductionHCProbe::Disabled:production);
                    if(side&&retained)lowerRetainedM1HCFFN(lane.graph,"ab_previous_write");
                },
                side?observation:FullProductionHCObservation::ObservedAll);
            if(full_production&&side&&!retained)requireFullProductionOutputDemandSealNegatives(*lanes[side]);
        }
        if(full_production&&!observed_all) {
            const std::array<FP32Tensor*,2> targets{prefix[1].activated.get(),prefix[1].logits.get()};
            for(unsigned plane=0;plane<2;++plane) {
                if(plane==0&&prefix[1].activated.get()==prefix[1].down.get())continue;
                if(plane==1&&up_observed)continue;
                std::memcpy(targets[plane]->mutable_data(),hc_omitted_poison[plane].data(),hc_omitted_poison[plane].size()*sizeof(float));
                TransferEngine::prepareDeviceInput(targets[plane],device,stream);
            }
            require(completion.join(),"retire omitted production mirror poison before original capture");
        }
        if(!retained) {
            // Reuse the original prepared source authorities. Only these descriptor
            // records are copied; canonical matrix payloads remain in their pools.
            adapted_fixed[0]=castagnaABNativeMatrix(prefix[0].down_prepared.store->gemmKernel(prefix[0].down_prepared.ref),R,K);
            adapted_fixed[1]=castagnaABNativeMatrix(prefix[0].up_prepared.store->gemmKernel(prefix[0].up_prepared.ref),K,R);
            // FrozenModelWeightSet indexes layer-relative suffixes, exactly as the
            // original SharedExpertFFNStage bindings above; GGUF payload names are
            // fully qualified and must not be used as this API's lookup key.
            constexpr std::array shared_suffixes{"ffn_gate_shexp.weight","ffn_up_shexp.weight","ffn_down_shexp.weight"};
            for(unsigned role=0;role<shared_suffixes.size();++role) {
                const std::string role_context="role="+std::to_string(role)+" suffix="+shared_suffixes[role]+
                    " device="+device.toString();
                const auto* binding=shared.optionalLayerForDevice(0,shared_suffixes[role],device);
                require(binding&&binding->tensor,"original shared adapted source binding: "+role_context);
                const auto ref=store.preparedRefForBinding(binding->binding_id,device);
                require(bool(ref),"original shared adapted prepared reference: "+role_context);
                adapted_fixed[role+2]=castagnaABNativeMatrix(store.gemmKernel(*ref),role==2?H:shape.shared,role==2?shape.shared:H);
                adapted_host_tables[role].reserve(shape.experts);
            }
            require(source.fragments.size()==unsigned(shape.experts),"complete original adapted fragment bank");
            for(unsigned expert=0;expert<unsigned(shape.experts);++expert) {
                const auto& fragment=source.fragments[expert];
                require(fragment.valid()&&fragment.expertId()==int(expert),"original adapted fragment identity");
                const std::array engines{fragment.gate(),fragment.up(),fragment.down()};
                for(unsigned role=0;role<3;++role)
                    adapted_host_tables[role].push_back(castagnaABNativeMatrix(engines[role].get(),
                        role==2?H:shape.routed,role==2?shape.routed:H));
            }
        }
        raw.push_back(castagnaABRawTensor(*prefix[0].down_source,device,stream));
        raw.push_back(castagnaABRawTensor(*prefix[0].up_source,device,stream));
        for(const auto* name:{"blk.0.ffn_gate_exps.weight","blk.0.ffn_up_exps.weight","blk.0.ffn_down_exps.weight",
            "blk.0.ffn_gate_shexp.weight","blk.0.ffn_up_shexp.weight","blk.0.ffn_down_shexp.weight",
            "blk.0.ffn_gate_inp_shexp.weight","blk.0.ffn_gate_inp.weight"})
            raw.push_back(castagnaABRaw(loader,name,device,stream));
        const std::array<unsigned,5> extents{K,K,H,512,H};
        for(unsigned plane=0;plane<5;++plane) {
            fork_outputs[plane]=fullProductionHCFloat(extents[plane]+guards,device,stream);
            if(!retained)adapted_outputs[plane]=fullProductionHCFloat(extents[plane]+guards,device,stream);
            for(unsigned arm=0;arm<arm_count;++arm){observed[arm][plane].resize(extents[plane]+(arm>=2?guards:0));baseline[arm][plane].resize(observed[arm][plane].size());}
        }
        for(unsigned side=0;side<2;++side){route_ids[side].resize(shape.top_k);route_weights[side].resize(shape.top_k);}
        fork_ids=TestTensorFactory::createINT32({std::size_t(shape.top_k)});
        require(fork_ids->allocateOnDevice(device,stream),"fork observer ID output");
        fork_weights=fullProductionHCFloat(shape.top_k,device,stream);
        if(!retained) {
            adapted_ids=TestTensorFactory::createINT32({std::size_t(shape.top_k)});
            require(adapted_ids->allocateOnDevice(device,stream),"adapted observer ID output");
            adapted_weights=fullProductionHCFloat(shape.top_k,device,stream);
        }
        const auto ptr=[&](unsigned index){return static_cast<const char*>(raw[index]->gpu_data_ptr());};
        request.prefix={static_cast<const float*>(prefix[0].residual->gpu_data_ptr()),
            static_cast<const float*>(prefix[0].previous->gpu_data_ptr()),static_cast<const float*>(prefix[0].raw->gpu_data_ptr()),
            static_cast<const float*>(prefix[0].gain->gpu_data_ptr()),ptr(0),ptr(1),1e-6f,.25f,.25f,.25f,2.f};
        using ForkGate = qfn_castagna_reference::GateSource;
        using ForkDown = qfn_castagna_reference::DownSource;
        const auto gate_type = loader.nativeWeightSourceGeometry(kRoutedNames[0]).original_type;
        const auto down_type = loader.nativeWeightSourceGeometry(kRoutedNames[2]).original_type;
        require((gate_type == TensorType::Q4_K || gate_type == TensorType::Q5_K) &&
            (down_type == TensorType::Q5_1 || down_type == TensorType::Q8_0),
            "original fork requests an admitted GGUF gate/up and down encoding");
        request.ffn={reinterpret_cast<const float*>(ptr(9)),ptr(5),ptr(6),ptr(7),reinterpret_cast<const float*>(ptr(8)),
            ptr(2),ptr(3),ptr(4),gate_type == TensorType::Q4_K ? ForkGate::Q4_K : ForkGate::Q5_K,
            down_type == TensorType::Q5_1 ? ForkDown::Q5_1 : ForkDown::Q8_0,
            unsigned(shape.routed),unsigned(shape.shared),unsigned(shape.top_k),1e-20f};
        if(observe_phases)for(unsigned side=0;side<2;++side) {
            timestamp_storage[side]=TestTensorFactory::createINT32({2*(timestamp_ctas*timestamp_slots+timestamp_guards)});
            require(timestamp_storage[side]->allocateOnDevice(device,stream),"HTS persistent output allocation");
            timestamp_host[side].resize(timestamp_ctas*timestamp_slots+timestamp_guards);
        }
        const auto timestampPointer=[&](unsigned side)->unsigned long long* {
            return observe_phases?static_cast<unsigned long long*>(timestamp_storage[side]->gpu_data_ptr()):nullptr;
        };
        request.outputs={static_cast<float*>(fork_outputs[0]->gpu_data_ptr()),static_cast<float*>(fork_outputs[1]->gpu_data_ptr()),
            static_cast<float*>(fork_outputs[2]->gpu_data_ptr()),static_cast<float*>(fork_outputs[3]->gpu_data_ptr()),
            static_cast<float*>(fork_outputs[4]->gpu_data_ptr()),timestampPointer(0)};
        require(qfn_castagna_t1_post_ffn_prepare(&request,device.ordinal,&admission)==hipSuccess,
            "actual exported fork full-kernel admission");
        require(admission.grid==60&&admission.threads==1024&&admission.resources.localSizeBytes==0,
            "actual fork original geometry and zero private storage");
        require(qfn_castagna_t1_post_ffn_globals(&globals)==hipSuccess,"original fork module state");
        if(!retained) {
            adapted_request.original=request;
            auto& original=adapted_request.original;
            original.prefix.hc_down_q8_0=nullptr;original.prefix.hc_up_q8_0=nullptr;
            original.ffn.shared_gate_q8_0=nullptr;original.ffn.shared_up_q8_0=nullptr;original.ffn.shared_down_q8_0=nullptr;
            original.ffn.expert_gate=nullptr;original.ffn.expert_up=nullptr;original.ffn.expert_down=nullptr;
            original.outputs={static_cast<float*>(adapted_outputs[0]->gpu_data_ptr()),static_cast<float*>(adapted_outputs[1]->gpu_data_ptr()),
                static_cast<float*>(adapted_outputs[2]->gpu_data_ptr()),static_cast<float*>(adapted_outputs[3]->gpu_data_ptr()),
                static_cast<float*>(adapted_outputs[4]->gpu_data_ptr()),timestampPointer(1)};
            adapted_request.native={adapted_fixed[0],adapted_fixed[1],adapted_fixed[2],adapted_fixed[3],adapted_fixed[4],
                castagnaABPublishMatrices(adapted_host_tables[0],adapted_device_tables[0],device,stream),
                castagnaABPublishMatrices(adapted_host_tables[1],adapted_device_tables[1],device,stream),
                castagnaABPublishMatrices(adapted_host_tables[2],adapted_device_tables[2],device,stream)};
            require(qfn_castagna_native_weight_prepare(&adapted_request,adapted_host_tables[0],adapted_host_tables[1],
                adapted_host_tables[2],device.ordinal,&adapted_admission)==hipSuccess,"adapted actual fork admission");
            require(adapted_admission.function==qfn_castagna_native_weight_kernel()&&
                adapted_admission.function!=admission.function&&adapted_admission.grid==admission.grid&&
                adapted_admission.threads==admission.threads&&adapted_admission.resources.localSizeBytes==0,
                "adapted full fork retains original geometry and zero private storage");
            require(qfn_castagna_native_weight_globals(&adapted_globals)==hipSuccess,"adapted separate module state");
            for(unsigned i=0;i<qfn_castagna_reference::global_region_count;++i) {
                require(adapted_globals.regions[i].data&&adapted_globals.regions[i].bytes==globals.regions[i].bytes&&
                    std::string_view(adapted_globals.regions[i].name)==globals.regions[i].name&&
                    adapted_globals.regions[i].data!=globals.regions[i].data,"adapted fork owns separate matching global ranges");
            }
        }
        constexpr std::array failure_names{"g_hcp_mh","g_hcp_h","g_hcp_sg"};
        for(unsigned i=0;i<failure_names.size();++i) {
            const auto found=std::find_if(std::begin(globals.regions),std::end(globals.regions),[&](const auto& region) {
                return std::string_view(region.name)==failure_names[i];
            });
            require(found!=std::end(globals.regions)&&found->bytes%sizeof(float)==0,"fork failure global has exact float storage");
            fork_failure_regions[i]=&*found;fork_failure_values[i].resize(found->bytes/sizeof(float));
        }
        require(completion.join(),"finish cold source preparation");
        // Keep the original source construction lease through every adapted
        // launch; its borrowed canonical matrix allocations remain immutable.

        for(unsigned arm=0;arm<arm_count;++arm) {
            captures[arm]=worker.createGraphCapture(stream);
            if(arm<2) {
                auto& lane=*lanes[arm];
                for(const auto& name:lane.graph.getExecutionOrder()) {
                    auto* stage=lane.graph.getNode(name)->stage.get();
                    if(auto* consumer=dynamic_cast<IWorkspaceConsumer*>(stage))consumer->bindWorkspace(lane.workspace.get());
                    require(stage->prepareGraphLaunch(context.get(),stream),"prepare exact full local capture");
                }
                auto local_plan=lane.plan;
                if(production!=FullProductionHCProbe::MiddleRead)
                    std::erase_if(local_plan,[&](const auto& step) {
                        return static_cast<const IComputeStage*>(step.stage_identity)->isCollectiveStage();
                    });
                GraphCaptureDependencyLedger ledger(device,stream,local_plan,"full_hc_local_ffn",
                    GraphCaptureDependencyLedger::ExternalInputAuthority::BindDeclaredAddressesOnly);
                ScopedBackendGraphCapture capture(worker,*captures[arm],"full_hc_local_ffn",&ledger);
                require(capture.begin(),"begin full native graph");
                for(const auto& name:lane.graph.getExecutionOrder()) {
                    auto* stage=lane.graph.getNode(name)->stage.get();
                    if(production!=FullProductionHCProbe::MiddleRead&&stage->isCollectiveStage())continue;
                    // Middle correctness retains the actual collective that
                    // publishes the following write's block input.
                    ScopedGraphCaptureStage scope(stage);require(stage->execute(context.get()),"capture actual full local operation");scope.complete();
                }
                capture.finish();
            } else {
                ScopedBackendGraphCapture capture(worker,*captures[arm],"actual_castagna_full_post");
                require(capture.begin(),"begin actual fork capture");
                const auto status=arm==2?qfn_castagna_t1_post_ffn_launch(&request,&admission,static_cast<hipStream_t>(stream)):
                    qfn_castagna_native_weight_launch(&adapted_request,&adapted_admission,static_cast<hipStream_t>(stream));
                require(status==hipSuccess,"capture original/adapted actual fork kernel");capture.finish();
            }
            require(captures[arm]->instantiate(),"instantiate complete local arm");
            std::vector<GPUGraphKernelNodeInfo> nodes;std::string error;
            require(captures[arm]->inspectKernelNodes(nodes,&error)&&!nodes.empty(),"inspect complete arm: "+error);
            if(full_production&&arm==1) {
                require(production==FullProductionHCProbe::MiddleRead?nodes.size()>=2u:nodes.size()==1u,
                    "full production final entry or complete middle collective/write graph");
                const auto* expected_fn = fullProductionHCFunction(source,shared,store,device,0,retained);
                require(expected_fn&&nodes.front().local_memory_bytes_per_thread==0&&nodes.front().block_x==1024,
                    "actual canonical closed capture and physical launch geometry");
                requireFullProductionHCBinding(*captures[arm],expected_fn,prefix[arm],*lanes[arm]);
            }
            if(arm>=2)require(nodes.size()==1&&nodes[0].function_identity==reinterpret_cast<std::uintptr_t>(
                arm==2?admission.function:adapted_admission.function),"fork arm executes exactly admitted full kernel");
            std::osyncstream(std::cout)<<"CASTAGNA_FULL_AB_SCOPE,arm="<<arm<<",F="<<shape.routed<<",S="<<shape.shared
                <<",full_production="<<(full_production&&arm==1)<<",middle_injection="<<(production==FullProductionHCProbe::MiddleRead&&arm<2)
                <<",bounded_centered="<<(input.source==CastagnaABSource::BoundedCentered)
                <<",real_gguf="<<input.real()<<",seeded_boundary_input=1,adapted_weight_readers="<<(arm==3)<<",kernels="<<nodes.size()
                <<",scope="<<(production==FullProductionHCProbe::MiddleRead&&arm<2?
                    "post+HC+router+FFN+collective+next_write,functional_only":"post+HC+router+localFFN,no_collective,no_early")<<'\n';
            children[arm]=worker.createGraphCapture(stream);
            std::array<std::string,32> names;std::vector<GPUOrderedTimelineStep> steps;
            for(unsigned i=0;i<32;++i){names[i]="full_operation_"+std::to_string(i);steps.push_back({.name=names[i].c_str(),.capture=captures[arm].get()});}
            require(children[arm]->buildOrderedTimelineTransaction(steps,GPUOrderedTimelineInstrumentation::Disabled),"build32 full operations");
            timed[arm]=worker.createGraphCapture(stream);const GPUOrderedTimelineStep group{.name="full32",.capture=children[arm].get()};
            require(timed[arm]->buildOrderedTimelineTransaction(std::span(&group,1),GPUOrderedTimelineInstrumentation::PerStepEvents)&&timed[arm]->instantiate(),
                "one event pair around32 complete operations");
            require(timed[arm]->nodeCount()==children[arm]->nodeCount()+2,"one whole-parent pair only");
        }
        const auto resetFork=[&](unsigned arm) {
            for(const auto& region:(arm==2?globals:adapted_globals).regions)
                require(region.data&&region.bytes&&backend->memset(region.data,0,region.bytes,device.ordinal,stream),
                    "reset original fork module state outside bracket");
        };
        const auto publish=[&](unsigned history) {
            expected_authored_active_count=1;
            for(unsigned i=0;i<K;++i) {
                input_words[0][i]=.2f+float(int((i*13+history*17)%127)-63)/256.f;
                input_words[3][i]=.75f+float(i%17)/64.f;
            }
            for(unsigned i=0;i<H;++i)input_words[1][i]=float(int((i*19+history*23)%193)-96)/512.f;
            for(unsigned i=0;i<4;++i)input_words[2][i]=float(int((history+i*3)%13)-6)/8.f;
            for(auto& p:prefix)for(unsigned role=0;role<3;++role) {
                const std::array tensors{p.residual,p.previous,p.raw,p.gain};
                std::memcpy(tensors[role]->mutable_data(),input_words[role].data(),input_words[role].size()*sizeof(float));
                TransferEngine::prepareDeviceInput(tensors[role].get(),device,stream);
            }
            for(auto& lane:lanes) {
                lane->active->mutable_typed_data()[0]=1;
                TransferEngine::publishHostWrite(lane->active.get());
                TransferEngine::prepareDeviceInput(lane->active.get(),device,stream);
            }
        };
        // Reset only diagnostic output bytes on the exact stream before each
        // replay. The original kernel never consumes HTS; topology stays retained.
        const auto resetTimestamps=[&](unsigned arm) {
            if(!observe_phases||arm<2)return;
            auto& storage=timestamp_storage[arm-2];
            std::fill_n(storage->mutable_typed_data(),storage->numel(),int32_t(-1));
            TransferEngine::publishHostWrite(storage.get());
            TransferEngine::prepareDeviceInput(storage.get(),device,stream);
            // Prove this particular poison reached device storage before the
            // next kernel can overwrite it. Otherwise two stale timestamp
            // observations could satisfy only a written-slot comparison.
            require(backend->deviceToHostOnStream(timestamp_host[arm-2].data(),storage->gpu_data_ptr(),
                timestamp_host[arm-2].size()*sizeof(unsigned long long),device.ordinal,stream),"observe actual prelaunch HTS poison");
            require(completion.join(),"retire HTS poison freshness observation before launch");
            require(std::all_of(timestamp_host[arm-2].begin(),timestamp_host[arm-2].end(),
                [](unsigned long long value){return value==~0ull;}),"actual device HTS bank reset to authored sentinel before every replay");
        };
        const auto observe=[&](unsigned arm,bool live=true) {
            std::array<const void*,5> pointers;
            if(arm>=2)for(unsigned i=0;i<5;++i)pointers[i]=(arm==2?fork_outputs[i]:adapted_outputs[i])->gpu_data_ptr();
            else pointers={prefix[arm].post->gpu_data_ptr(),prefix[arm].normalized->gpu_data_ptr(),
                lanes[arm]->hidden->gpu_data_ptr(),lanes[arm]->workspace->getBuffer(MoEWorkspaceBuffers::ROUTE_LOGITS),
                lanes[arm]->combined->gpu_data_ptr()};
            if(arm>=2) {
                const auto& outputs=arm==2?fork_outputs:adapted_outputs;
                auto& ids=arm==2?fork_ids:adapted_ids;
                auto& weights=arm==2?fork_weights:adapted_weights;
                auto& host_ids=arm==2?observed_fork_ids:observed_adapted_ids;
                auto& host_weights=arm==2?observed_fork_weights:observed_adapted_weights;
                for(const auto& value:outputs)TransferEngine::publishDeviceWrite(value.get(),device,stream);
                const auto status=arm==2?qfn_castagna_t1_observe_selection(&admission,static_cast<int*>(ids->gpu_data_ptr()),
                    static_cast<float*>(weights->gpu_data_ptr()),static_cast<hipStream_t>(stream)):
                    qfn_castagna_native_weight_observe_selection(&adapted_admission,static_cast<int*>(ids->gpu_data_ptr()),
                    static_cast<float*>(weights->gpu_data_ptr()),static_cast<hipStream_t>(stream));
                require(status==hipSuccess,"observe original/adapted fork topK outside measured parent");
                TransferEngine::publishDeviceWrite(ids.get(),device,stream);
                TransferEngine::publishDeviceWrite(weights.get(),device,stream);
                require(backend->deviceToHostOnStream(host_ids.data(),ids->gpu_data_ptr(),shape.top_k*sizeof(int32_t),device.ordinal,stream),"observe actual fork IDs");
                require(backend->deviceToHostOnStream(host_weights.data(),weights->gpu_data_ptr(),shape.top_k*sizeof(float),device.ordinal,stream),"observe actual fork weights");
            }
            for(unsigned i=0;i<5;++i)require(backend->deviceToHostOnStream(observed[arm][i].data(),pointers[i],
                observed[arm][i].size()*sizeof(float),device.ordinal,stream),"observe full actual arm");
            if(arm<2) {
                require(backend->deviceToHostOnStream(route_ids[arm].data(),lanes[arm]->indices->gpu_data_ptr(),shape.top_k*sizeof(float),device.ordinal,stream),"observe native original route IDs");
                require(backend->deviceToHostOnStream(route_weights[arm].data(),lanes[arm]->weights->gpu_data_ptr(),shape.top_k*sizeof(float),device.ordinal,stream),"observe native original route weights");
            }
            if(full_production&&arm<2) {
                const auto& p=prefix[arm];
                const std::array<const void*,5> hc{p.down->gpu_data_ptr(),p.activated->gpu_data_ptr(),p.logits->gpu_data_ptr(),
                    p.injection?p.injection->gpu_data_ptr():nullptr,p.next_trunk?p.next_trunk->gpu_data_ptr():nullptr};
                for(unsigned i=0;i<hc.size();++i)if(!hc_observed[arm][i].empty())
                    require(backend->deviceToHostOnStream(hc_observed[arm][i].data(),hc[i],
                        hc_observed[arm][i].size()*sizeof(float),device.ordinal,stream),"observe original HC public intermediate");
            }
            if(observe_phases&&arm>=2) {
                auto& storage=timestamp_storage[arm-2];
                TransferEngine::publishDeviceWrite(storage.get(),device,stream);
                require(backend->deviceToHostOnStream(timestamp_host[arm-2].data(),storage->gpu_data_ptr(),
                    timestamp_host[arm-2].size()*sizeof(unsigned long long),device.ordinal,stream),"observe original CTA HTS");
            }
            if(arm<2)require(backend->deviceToHostOnStream(&observed_active_count[arm],lanes[arm]->active->gpu_data_ptr(),
                sizeof(std::int32_t),device.ordinal,stream),"observe actual authored count after captured replay");
            require(completion.join(),"retire exact arm observations");
            if(full_production&&arm==1&&!observed_all) {
                if(prefix[1].activated.get()!=prefix[1].down.get())
                    require(hc_observed[1][1]==hc_omitted_poison[0],"unobserved activated mirror keeps original poison");
                if(!up_observed)require(hc_observed[1][2]==hc_omitted_poison[1],"unobserved up mirror keeps original poison");
            }
            if(arm<2)require(observed_active_count[arm]==expected_authored_active_count,
                "actual device input count follows active/empty/reset authoring, not a stale peer comparison");
            if(observe_phases&&arm>=2) {
                const auto& ticks=timestamp_host[arm-2];
                constexpr auto untouched=~0ull;
                for(std::size_t cta=0;cta<timestamp_ctas;++cta) {
                    std::osyncstream row(std::cout);
                    row<<"CASTAGNA_CTA_HTS,arm="<<arm<<",history="<<current_history
                       <<",submission="<<current_submission<<",cta="<<cta<<",unit=invariant_ticks";
                    for(std::size_t slot=0;slot<timestamp_slots;++slot) {
                        const auto value=ticks[cta*timestamp_slots+slot];
                        require(slot==13?value==untouched:value!=untouched,"original T1 HTS written-slot contract");
                        row<<",s"<<slot<<'='<<value;
                    }
                    row<<'\n';
                }
                for(std::size_t i=timestamp_ctas*timestamp_slots;i<ticks.size();++i)
                    require(ticks[i]==untouched,"HTS output suffix guard");
            }
            constexpr std::array plane_names{"post_residual","normalized","mixed","router_probabilities_or_raw_logits","ffn"};
            for(unsigned i=0;i<5;++i) {
                // Empty native rows have no consumable probability snapshot;
                // this arena may contain retired HC quantization bytes. Every
                // active snapshot and every other public plane stays checked.
                if(!live&&arm<2&&i==3)continue;
                const auto bad=std::find_if(observed[arm][i].begin(),observed[arm][i].begin()+extents[i],
                    [](float value){return !std::isfinite(value);});
                if(bad!=observed[arm][i].begin()+extents[i]) {
                    const auto index=std::size_t(bad-observed[arm][i].begin());
                    std::osyncstream(std::cerr)<<"CASTAGNA_FULL_AB_NONFINITE,arm="<<arm<<",history="<<current_history
                        <<",submission="<<current_submission<<",plane="<<plane_names[i]<<",index="<<index
                        <<",bits=0x"<<std::hex<<std::bit_cast<std::uint32_t>(*bad)<<std::dec<<'\n';
                    if(arm==2) {
                        // Failure-only inspection of original globals. No
                        // encoder, clamp, route repair or measured work changes.
                        for(unsigned state=0;state<3;++state)
                            require(backend->deviceToHostOnStream(fork_failure_values[state].data(),
                                fork_failure_regions[state]->data,fork_failure_regions[state]->bytes,
                                device.ordinal,stream),"read original fork failure global");
                        require(completion.join(),"retire original fork failure ranges");
                        const auto report=[&](std::string_view name,const std::vector<float>& values,
                            unsigned rows,unsigned width,unsigned stride,bool q81) {
                            double minimum=std::numeric_limits<double>::infinity(),maximum=-minimum,max_abs=0,max_sum=0;
                            unsigned nonfinite=0,scale_overflow=0,sum_overflow=0;
                            for(unsigned row=0;row<rows;++row)for(unsigned block=0;block<width;block+=32) {
                                double sum=0,block_abs=0;
                                for(unsigned j=block;j<std::min(width,block+32);++j) {
                                    const float value=values.at(size_t(row)*stride+j);
                                    if(!std::isfinite(value)){++nonfinite;continue;}
                                    minimum=std::min(minimum,double(value));maximum=std::max(maximum,double(value));
                                    block_abs=std::max(block_abs,std::abs(double(value)));sum+=value;
                                }
                                max_abs=std::max(max_abs,block_abs);max_sum=std::max(max_sum,std::abs(sum));
                                if(q81){scale_overflow+=block_abs/127.0>65504.0;sum_overflow+=std::abs(sum)>65504.0;}
                            }
                            std::osyncstream(std::cerr)<<"CASTAGNA_FULL_AB_RANGE,plane="<<name<<",rows="<<rows<<",width="<<width
                                <<",min="<<minimum<<",max="<<maximum<<",max_abs="<<max_abs<<",nonfinite="<<nonfinite
                                <<",max_abs_block32_sum="<<max_sum<<",half_scale_overflow_blocks="<<scale_overflow
                                <<",half_sum_overflow_blocks="<<sum_overflow<<",sum_check=diagnostic_FP64_not_original_reduction\n";
                        };
                        report("fork_post",observed[arm][0],1,K,K,false);
                        report("fork_normalized",observed[arm][1],1,K,K,false);
                        report("fork_mixed",observed[arm][2],1,H,H,true);
                        report("fork_raw_router",observed[arm][3],1,shape.experts,shape.experts,false);
                        report("g_hcp_mh_selected_slots",fork_failure_values[0],shape.top_k,shape.routed,640,true);
                        report("g_hcp_h",fork_failure_values[1],1,shape.shared,640,true);
                        report("g_hcp_sg",fork_failure_values[2],1,1,1,false);
                    }
                    if(arm<2) {
                        const std::array<const void*,3> intermediate{prefix[arm].down->gpu_data_ptr(),
                            prefix[arm].activated->gpu_data_ptr(),prefix[arm].logits->gpu_data_ptr()};
                        for(unsigned stage=0;stage<3;++stage)
                            require(backend->deviceToHostOnStream(prefix_diagnostics[stage].data(),intermediate[stage],
                                prefix_diagnostics[stage].size()*sizeof(float),device.ordinal,stream),"read prefix failure evidence");
                        require(completion.join(),"retire prefix failure evidence");
                        constexpr std::array names{"hc_down","hc_activated","hc_up_logits"};
                        for(unsigned stage=0;stage<3;++stage) {
                            const auto& values=prefix_diagnostics[stage];
                            const auto failure=std::find_if(values.begin(),values.end(),[](float value){return !std::isfinite(value);});
                            const auto position=failure==values.end()?0:std::size_t(failure-values.begin());
                            std::osyncstream(std::cerr)<<"CASTAGNA_FULL_AB_PREFIX,arm="<<arm<<",plane="<<names[stage]
                                <<",all_finite="<<(failure==values.end())<<",index="<<position
                                <<",bits=0x"<<std::hex<<std::bit_cast<std::uint32_t>(values[position])<<std::dec<<'\n';
                        }
                    }
                    require(false,"finite full-region public plane: arm="+std::to_string(arm)+" plane="+plane_names[i]+
                        " index="+std::to_string(index));
                }
                if(arm>=2)for(unsigned j=extents[i];j<observed[arm][i].size();++j)
                    require(observed[arm][i][j]==guard,"fork public plane guard");
            }
            if(arm>=2) {
                const auto& ids=arm==2?observed_fork_ids:observed_adapted_ids;
                const auto& weights=arm==2?observed_fork_weights:observed_adapted_weights;
                float sum=0.f;
                for(unsigned i=0;i<unsigned(shape.top_k);++i) {
                    require(ids[i]>=0&&ids[i]<shape.experts&&std::isfinite(weights[i])&&weights[i]>=0.f,"valid actual fork topK");
                    for(unsigned j=0;j<i;++j)require(ids[i]!=ids[j],"unique actual fork topK");
                    sum+=weights[i];
                }
                require(std::abs(sum-1.f)<1e-5f,"normalized actual fork topK weights");
            }
            if(live)require(std::any_of(observed[arm][4].begin(),observed[arm][4].begin()+H,
                [](float value){return value!=0.f;}),"actual FFN must do nonzero work");
        };
        const auto poison=[&](unsigned arm) {
            if(arm>=2)for(auto& value:(arm==2?fork_outputs:adapted_outputs)) {
                std::fill_n(value->mutable_data(),value->numel(),guard);
                TransferEngine::prepareDeviceInput(value.get(),device,stream);
            }
        };
        // Q8 HC operands and the original FP32 router are bit-identical
        // between A and B. Only later Q4_K effective coefficients may differ.
        const auto requireExactReaderBoundary=[&] {
            for(unsigned plane=0;plane<4;++plane)
                require(std::memcmp(observed[2][plane].data(),observed[3][plane].data(),
                    extents[plane]*sizeof(float))==0,
                    "weight-reader A/B exact POST/normalized/mixed/raw-router plane "+std::to_string(plane));
            require(observed_fork_ids==observed_adapted_ids,"weight-reader A/B exact ordered selected IDs");
            require(std::memcmp(observed_fork_weights.data(),observed_adapted_weights.data(),
                observed_fork_weights.size()*sizeof(float))==0,"weight-reader A/B exact selected weights");
            std::osyncstream(std::cout)<<"CASTAGNA_WEIGHT_AB_EXACT,history="<<current_history
                <<",submission="<<current_submission<<",prefix_planes=4,ordered_ids=1,weights=1\n";
        };
        CastagnaOracleBytes oracle_metadata;
        if(oracle_config.action!=CastagnaOracleAction::Disabled) {
            oracle_metadata.text("R53-fork-math-native-Q4K-effective-half-Q51-Q8-T1POST-v1");
            for(const auto value:{H,K,R,unsigned(shape.routed),unsigned(shape.shared),unsigned(shape.experts),unsigned(shape.top_k),
                                 input.layer,input.participant,input.degree,128u,64u,60u,1024u,64u})oracle_metadata.scalar(std::uint32_t(value));
            const auto routed_window=qfn_castagna_reference::equalWeightSourceShard(640,128,input.degree,input.participant);
            const auto shared_window=qfn_castagna_reference::equalWeightSourceShard(640,64,input.degree,input.participant);
            for(const auto value:{routed_window.first,routed_window.count,shared_window.first,shared_window.count})oracle_metadata.scalar(value);
            for(const auto value:{request.prefix.epsilon,request.prefix.down_scale,request.prefix.pre_scale,
                request.prefix.post_scale1,request.prefix.post_scale2,request.ffn.selected_sum_clamp})oracle_metadata.scalar(std::bit_cast<std::uint32_t>(value));
            oracle_metadata.text("input-v1:residual(13i+17h)%127;block(19i+23h)%193;inject(h+3i)%13;reset13=0");
            oracle_metadata.text(device.toString());oracle_metadata.text("gfx906-wave64-IEEE-fp32-original-fork");
            const auto paths=real_loader->artifactPaths();
            require(!paths.empty(),"real source artifact identities for exact oracle");
            oracle_metadata.scalar(std::uint32_t(paths.size()));
            for(const auto& path:paths)castagnaOracleFileIdentity(oracle_metadata,path);
            const auto append_matrix_policy=[&](const auto& matrix) {
                oracle_metadata.scalar(std::int32_t(matrix.n));oracle_metadata.scalar(std::int32_t(matrix.k));
                oracle_metadata.scalar(matrix.blocks_per_row);oracle_metadata.scalar(matrix.codebook_id);
                oracle_metadata.scalar(matrix.source_codebook_id);oracle_metadata.scalar(matrix.source_is_superblock);
            };
            for(const auto& matrix:adapted_fixed)append_matrix_policy(matrix);
            for(const auto& table:adapted_host_tables)for(const auto& matrix:table)append_matrix_policy(matrix);
            // The learned gain is immutable and represented as exact original FP32
            // source bytes; it is not copied from a device-owned value.
            oracle_metadata.field("norm_gain",{reinterpret_cast<const std::byte*>(prefix[0].gain->raw_data()),K*sizeof(float)});
        }
        CastagnaExactCandidateOracle oracle(oracle_config,oracle_metadata);
        bool identical_route_work=true;
        for(unsigned history=0;history<20;++history) {
            current_history=history;
            publish(history==13?0:history);
            for(unsigned arm=0;arm<arm_count;++arm) {
                poison(arm);if(arm>=2)resetFork(arm);
                current_submission="first";
                resetTimestamps(arm);require(captures[arm]->launch(),"launch original fullregion arm");observe(arm);baseline[arm]=observed[arm];
                if(full_production&&arm<2)hc_baseline[arm]=hc_observed[arm];
                current_submission="repeat";
                resetTimestamps(arm);require(captures[arm]->launch(),"repeat retained actual fullregion arm");observe(arm);
                for(unsigned i=0;i<5;++i)require(std::memcmp(observed[arm][i].data(),baseline[arm][i].data(),
                    observed[arm][i].size()*sizeof(float))==0,"same-engine exact repeated public plane");
                if(full_production&&arm<2)for(unsigned i=0;i<5;++i)
                    require(hc_observed[arm][i].empty()||std::memcmp(hc_observed[arm][i].data(),hc_baseline[arm][i].data(),
                        hc_observed[arm][i].size()*sizeof(float))==0,"same-engine repeated HC public intermediate bytes");
            }
            if(!retained) {
                requireExactReaderBoundary();
                if(oracle_config.action!=CastagnaOracleAction::Disabled) {
                    CastagnaOracleBytes record;
                    for(unsigned role=0;role<3;++role)record.field("input"+std::to_string(role),std::as_bytes(std::span(input_words[role])));
                    for(unsigned plane=0;plane<5;++plane)record.field("output"+std::to_string(plane),
                        std::as_bytes(std::span(observed[3][plane].data(),size_t(extents[plane]))));
                    record.field("ordered_ids",std::as_bytes(std::span(observed_adapted_ids)));
                    record.field("selected_weights",std::as_bytes(std::span(observed_adapted_weights)));
                    oracle.observe(history,record);
                }
                unsigned adapted_ordered=0,adapted_intersection=0;
                for(unsigned i=0;i<unsigned(shape.top_k);++i) {
                    adapted_ordered+=observed_fork_ids[i]==observed_adapted_ids[i];
                    for(unsigned j=0;j<unsigned(shape.top_k);++j)adapted_intersection+=observed_fork_ids[i]==observed_adapted_ids[j];
                }
                {
                    std::osyncstream row(std::cout);
                    row<<"CASTAGNA_WEIGHT_AB_ROUTES,history="<<history<<",same_ordered="<<adapted_ordered
                        <<",intersection="<<adapted_intersection<<",top_k="<<shape.top_k<<",original=";
                    for(unsigned i=0;i<unsigned(shape.top_k);++i)row<<(i?";":"")<<observed_fork_ids[i];
                    row<<",adapted=";for(unsigned i=0;i<unsigned(shape.top_k);++i)row<<(i?";":"")<<observed_adapted_ids[i];
                    row<<'\n';
                }
                for(unsigned plane=0;plane<5;++plane) {
                    double max_error=0,squared=0;unsigned equal_bits=0;
                    for(unsigned j=0;j<extents[plane];++j) {
                        const double error=double(observed[2][plane][j])-observed[3][plane][j];
                        max_error=std::max(max_error,std::abs(error));squared+=error*error;
                        equal_bits+=std::bit_cast<uint32_t>(observed[2][plane][j])==std::bit_cast<uint32_t>(observed[3][plane][j]);
                    }
                    std::osyncstream(std::cout)<<"CASTAGNA_WEIGHT_AB_NUMERIC,history="<<history<<",plane="<<plane
                        <<",max_abs="<<max_error<<",rms="<<std::sqrt(squared/extents[plane])<<",equal_bits="<<equal_bits
                        <<",elements="<<extents[plane]<<",q4k_effective_coefficients=prepared_half,cross_engine_exact_required="
                        <<(plane<4?1:0)<<'\n';
                }
            }
            for(unsigned plane=0;plane<5;++plane)if(!full_production||observed_all||plane!=3)
                require(std::memcmp(observed[0][plane].data(),observed[1][plane].data(),extents[plane]*sizeof(float))==0,
                    "NativeLayout preserves original fullprefix public bytes");
            if(full_production)for(unsigned i=0;i<5;++i)
                if(observed_all||(i!=1&&(i!=2||up_observed)&&
                    (i!=0||prefix[1].activated.get()!=prefix[1].down.get())))
                require(hc_observed[0][i].size()==hc_observed[1][i].size()&&
                    (hc_observed[0][i].empty()||std::memcmp(hc_observed[0][i].data(),hc_observed[1][i].data(),hc_observed[0][i].size()*sizeof(float))==0),
                    "full production preserves down/activated/up/injection/next-write public bytes plane "+std::to_string(i));
            require(route_ids[0]==route_ids[1]&&route_weights[0]==route_weights[1],
                "same original native router decision and probabilities");
            for(unsigned side=0;side<2;++side) {
                float weight_sum=0.f;
                for(unsigned i=0;i<unsigned(shape.top_k);++i) {
                    const float id=route_ids[side][i],weight=route_weights[side][i];
                    require(std::isfinite(id)&&id>=0&&id<shape.experts&&id==std::floor(id)&&
                        std::isfinite(weight)&&weight>=0.f,"valid actual native routing output");
                    for(unsigned j=0;j<i;++j)require(route_ids[side][j]!=id,"unique actual native topK");
                    weight_sum+=weight;
                }
                require(std::abs(weight_sum-1.f)<1e-5f,"original normalized native selected probabilities");
            }
            if(history==0){reset_baseline=observed;if(full_production)hc_reset=hc_observed;}
            if(history==1)for(unsigned arm=0;arm<arm_count;++arm)
                require(std::memcmp(observed[arm][0].data(),reset_baseline[arm][0].data(),extents[0]*sizeof(float))!=0,
                    "each actual engine consumes changed external residual/write input");
            if(history==13)for(unsigned arm=0;arm<arm_count;++arm)for(unsigned plane=0;plane<5;++plane)
                require(std::memcmp(observed[arm][plane].data(),reset_baseline[arm][plane].data(),
                    observed[arm][plane].size()*sizeof(float))==0,"each actual engine resets to original request bytes");
            if(full_production&&history==13)for(unsigned arm=0;arm<2;++arm)for(unsigned plane=0;plane<5;++plane)
                require(hc_observed[arm][plane].size()==hc_reset[arm][plane].size()&&
                    (hc_observed[arm][plane].empty()||std::memcmp(hc_observed[arm][plane].data(),hc_reset[arm][plane].data(),hc_observed[arm][plane].size()*sizeof(float))==0),
                    "full production HC public values reset without recapture");
            unsigned same_routes=0,route_intersection=0;double route_weight_error=0;
            for(unsigned i=0;i<unsigned(shape.top_k);++i) {
                if(route_ids[0][i]==float(observed_fork_ids[i]))++same_routes;
                for(unsigned j=0;j<unsigned(shape.top_k);++j)if(route_ids[0][i]==float(observed_fork_ids[j])) {
                    ++route_intersection;
                    route_weight_error=std::max(route_weight_error,std::abs(double(route_weights[0][i])-observed_fork_weights[j]));
                }
            }
            {
                std::osyncstream row(std::cout);
                row<<"CASTAGNA_FULL_AB_ROUTE_IDS,history="<<history<<",native=";
                for(unsigned i=0;i<unsigned(shape.top_k);++i)row<<(i?";":"")<<int(route_ids[0][i]);
                row<<",fork=";
                for(unsigned i=0;i<unsigned(shape.top_k);++i)row<<(i?";":"")<<observed_fork_ids[i];
                row<<",intersection="<<route_intersection<<",top_k="<<shape.top_k<<'\n';
            }
            identical_route_work=identical_route_work&&(route_intersection==unsigned(shape.top_k));
            std::osyncstream(std::cout)<<"CASTAGNA_FULL_AB_ROUTES,history="<<history<<",same_ordered_ids="<<same_routes
                <<",selected_set_intersection="<<route_intersection<<",top_k="<<shape.top_k<<",max_selected_weight_error="<<route_weight_error
                <<",native_router=fp16_original,fork_router=fp32_original,native_histogram=true,fork_histogram=absent\n";
            for(unsigned i=0;i<5;++i) {
                if(i==3)continue; // Native publication is probabilities; fork public plane is raw logits.
                double maximum=0,absolute=0;
                for(unsigned j=0;j<extents[i];++j){const double e=std::abs(double(observed[0][i][j])-observed[2][i][j]);maximum=std::max(maximum,e);absolute+=e;}
                std::osyncstream(std::cout)<<"CASTAGNA_FULL_AB_NUMERIC,history="<<history<<",plane="<<i<<",max_abs="<<maximum
                    <<",mean_abs="<<absolute/extents[i]<<",cross_engine_exact_required=0\n";
            }
        }
        if(full_production) {
            // Retire a large→empty→large sequence through the same captured
            // production entries. Empty retains each original public plane;
            // no CPU decision substitutes for the device-owned count.
            expected_authored_active_count=0;
            for(auto& lane:lanes) {
                lane->active->mutable_typed_data()[0]=0;
                TransferEngine::publishHostWrite(lane->active.get());
                TransferEngine::prepareDeviceInput(lane->active.get(),device,stream);
            }
            for(unsigned repeat=0;repeat<2;++repeat) {
                for(unsigned arm=0;arm<2;++arm) {
                    require(captures[arm]->launch(),"replay original/full production empty rows");observe(arm,false);
                }
                for(unsigned plane=0;plane<5;++plane) {
                    if(plane!=3)require(std::memcmp(observed[0][plane].data(),observed[1][plane].data(),extents[plane]*sizeof(float))==0,
                        "empty full production consumable public bytes match staged");
                    if(observed_all||(plane!=1&&(plane!=2||up_observed)&&
                        (plane!=0||prefix[1].activated.get()!=prefix[1].down.get())))
                    require(hc_observed[0][plane].empty()||std::memcmp(hc_observed[0][plane].data(),hc_observed[1][plane].data(),
                        hc_observed[0][plane].size()*sizeof(float))==0,"empty full production HC bytes match staged");
                }
                require(route_ids[0]==route_ids[1]&&route_weights[0]==route_weights[1],"empty routing bytes match staged");
                for(unsigned arm=0;arm<2;++arm) {
                    for(unsigned slot=0;slot<unsigned(shape.top_k);++slot)
                        require(route_ids[arm][slot]==-1.f&&route_weights[arm][slot]==0.f,
                            "each empty engine independently clears selected route IDs and weights");
                    require(std::all_of(observed[arm][4].begin(),observed[arm][4].begin()+extents[4],
                        [](float value){return value==0.f;}),"each empty engine authors original zero combined output");
                }
            }
            publish(0);
            for(unsigned arm=0;arm<2;++arm) {
                require(captures[arm]->launch(),"reset active row after empty without recapture");observe(arm);
                for(unsigned plane=0;plane<5;++plane) {
                    require(std::memcmp(observed[arm][plane].data(),reset_baseline[arm][plane].data(),
                        extents[plane]*sizeof(float))==0,"active reset public bytes after empty");
                    require(hc_observed[arm][plane].empty()||std::memcmp(hc_observed[arm][plane].data(),hc_reset[arm][plane].data(),
                        hc_observed[arm][plane].size()*sizeof(float))==0,"active reset HC bytes after empty");
                }
            }
            std::osyncstream(std::cout)<<"FULL_PRODUCTION_HC_REPLAY,mode="<<unsigned(production)
                <<",histories=20,repeated=1,reset=1,empty_repeated=2,active_after_empty=1,public_byte_oracle=staged\n";
        }
        std::osyncstream(std::cout)<<"CASTAGNA_FULL_AB_ROUTE_SCOPE,all_histories_same_selected_set="<<identical_route_work
            <<",cross_engine_arithmetic_equal=0\n";
        // The original checkpoint compares the independently selected experts for all
        // 20 histories. Other source windows retain all 20 staged byte checks and
        // disclose every router difference. For those cases, every actually timed
        // record must match, including priming. The schedule is independent of routing.
        if(retained && input.route_cohort==CastagnaRouteCohort::AllReplayHistories)require(identical_route_work,"retained timing requires equal selected expert sets across all 20 histories");
        if(timing) {
            // Primary original/adapted Castagna pair first; retain the three
            // existing comparisons as secondary controls. Every observation is
            // one genuine whole32-operation parent interval.
            constexpr std::array<std::array<unsigned,2>,1> production_pair{{{1u,2u}}};
            constexpr std::array diagnostic_pairs{std::array{2u,3u},std::array{0u,2u},std::array{1u,2u},std::array{0u,1u}};
            const std::span<const std::array<unsigned,2>> pairs=retained?
                std::span<const std::array<unsigned,2>>{production_pair}:diagnostic_pairs;
            for(const auto pair:pairs)
            for(unsigned round=0;round<37;++round) {
                const auto order=native_quantised_dispatch::pairedRecordTimingRound<4>(round+5);
                current_history=unsigned(order.record);current_submission="timing_prime";
                publish(unsigned(order.record));
                // Cache-hot reference for this exact record and original source
                // is outside timed parents; measured outputs are checked directly.
                const unsigned prime_count=pair==std::array{2u,3u}?4:3;
                for(unsigned arm=0;arm<prime_count;++arm) {
                    if(arm>=2)resetFork(arm);
                    require(captures[arm]->launch(),"prime exact fullregion record");
                    observe(arm);baseline[arm]=observed[arm];
                }
                const bool adapted_pair=pair==std::array{2u,3u};
                if(adapted_pair)requireExactReaderBoundary();
                unsigned record_intersection=0,record_ordered=0;
                for(unsigned i=0;i<unsigned(shape.top_k);++i) {
                    const int id=adapted_pair?observed_adapted_ids[i]:int(route_ids[0][i]);
                    record_ordered+=id==observed_fork_ids[i];
                    for(unsigned j=0;j<unsigned(shape.top_k);++j)record_intersection+=id==observed_fork_ids[j];
                }
                std::osyncstream(std::cout)<<"CASTAGNA_FULL_AB_TIMING_ROUTES,record="<<order.record
                    <<",intersection="<<record_intersection<<",same_ordered="<<record_ordered
                    <<",top_k="<<shape.top_k<<'\n';
                // Original routers may order a near-tied selected set differently.
                // Unique IDs were already proven; same set preserves expert work.
                require(record_intersection==unsigned(shape.top_k),
                    "timing requires same actually selected expert set; no forced route substitution");
                const std::array sequence=order.order==native_quantised_dispatch::PairedTimingOrder::ABBA?std::array{pair[0],pair[1],pair[1],pair[0]}:
                    std::array{pair[1],pair[0],pair[0],pair[1]};
                for(unsigned slot=0;slot<4;++slot) {
                    const auto arm=sequence[slot];
                    require(completion.join(),"cold inputs and reset outside timed parent");
                    current_submission="timed_parent";
                    require(timed[arm]->launch(),"launch whole32 fullregion timing parent");observe(arm);
                    for(unsigned plane=0;plane<5;++plane)
                        require(std::memcmp(observed[arm][plane].data(),baseline[arm][plane].data(),
                            observed[arm][plane].size()*sizeof(float))==0,"actual32-parent output matches same-engine record");
                    const auto snapshot=timed[arm]->consumeOrderedTimelineTiming();
                    require(snapshot.state==GPUOrderedTimelineTimingState::Complete&&snapshot.samples.size()==1,
                        "one completed full32 interval");
                    if(round>=5)std::osyncstream(std::cout)<<"CASTAGNA_FULL_AB_SAMPLE,pair="<<pair[0]<<pair[1]<<",arm="<<arm
                        <<",round="<<round-5<<",record="<<order.record<<",slot="<<slot<<",operations=32,total_ms="
                        <<snapshot.samples[0].elapsed_ms<<",us_per_op="<<snapshot.samples[0].elapsed_ms*1000.0/32<<'\n';
                }
            }
        }
        require(completion.join(),"retire complete fullregion experiment");
        oracle.finish();
    });
}
