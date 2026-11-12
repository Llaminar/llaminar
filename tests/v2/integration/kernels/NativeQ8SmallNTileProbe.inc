/**
 * @file NativeQ8SmallNTileProbe.inc
 * @brief Existing cooperative Q8 tiles through original prepared GEMM graphs.
 *
 * Included inside the original Q8 device fixture's anonymous namespace. Cold
 * diagnostic controls bind one physical candidate before capture and restore
 * immediately; replay reads no host selector. Fixed-M graphs share one admitted
 * serial workspace and stable tensors. The explicit NT64/MT32 launch is the
 * byte oracle for the historical comparisons; the M512 prefill comparison uses
 * forced NT128/MT32 so its baseline survives later Auto promotion. Neither is
 * an invented serial-M1 arithmetic contract. This fixture installs no policy.
 */
#ifdef HAVE_ROCM
/** @brief Installed policy and existing checked cooperative physical candidates. */
enum class NativeQ8TileProbe { Installed, NT64, NT128, NT64MT16 };
/** @brief Select a physical comparison without changing the installed policy. */
enum class NativeQ8TileComparison { Columns, Rows, Prefill };
/** @brief Exact retained row geometries; no runtime engine capacity is inferred. */
struct NativeQ8TileGeometry {
    int large_rows; ///< Fixed large graph extent, excluding its untouched guard row.
    int small_rows; ///< Fixed partial graph extent sharing the same admitted storage.
};
/** @brief Bind existing diagnostic controls only during cold graph construction. */
class NativeQ8ColdTileScope final {
    ROCmConfig saved_;
public:
    /** @param choice Exact existing physical tile, or the unforced installed policy. */
    explicit NativeQ8ColdTileScope(NativeQ8TileProbe choice):saved_(debugEnv().rocm) {
        auto& c=mutableDebugEnv().rocm;
        c.native_quantised_mt=choice==NativeQ8TileProbe::Installed?0:(choice==NativeQ8TileProbe::NT64MT16?16:32);
        c.native_quantised_unroll=choice==NativeQ8TileProbe::Installed?-1:2;
        c.native_quantised_min_blocks=choice==NativeQ8TileProbe::Installed?0:1;
        c.native_quantised_force_n64=choice==NativeQ8TileProbe::NT64||choice==NativeQ8TileProbe::NT64MT16;
        c.native_quantised_force_n128=choice==NativeQ8TileProbe::NT128;
        c.native_quantised_full_tiles=choice==NativeQ8TileProbe::Installed?-1:0;
    }
    NativeQ8ColdTileScope(const NativeQ8ColdTileScope&)=delete;
    NativeQ8ColdTileScope&operator=(const NativeQ8ColdTileScope&)=delete;
    /** @brief Restore the original frozen startup configuration before replay. */
    ~NativeQ8ColdTileScope(){mutableDebugEnv().rocm=std::move(saved_);}
};
/** @brief Retire all graphs before their stage borrows, named workspace and tensors. */
struct NativeQ8TileGraphs final {
    Storage& storage;
    std::vector<std::unique_ptr<IGPUGraphCapture>> graphs;
    std::vector<std::unique_ptr<GEMMStage>> stages;
    /** @param original Original exact worker stream and allocation lifetime.
     * @param count Number of exact shape/tile captures in this finite experiment. */
    NativeQ8TileGraphs(Storage& original,std::size_t count):storage(original),graphs(count),stages(count){}
    /** @brief Terminal fixture join, then graph release, then stage unbinding. */
    ~NativeQ8TileGraphs(){
        storage.retire();
        for(auto& graph:graphs)if(graph){graph->reset();graph.reset();}
        for(auto& stage:stages)if(stage)stage->unbindWorkspace();
    }
};
/**
 * @brief Compare installed and forced cooperative tiles at one original shape.
 * @param device Explicit visible participant; the frozen actual PCI/ISA is printed.
 * @param k Original HC-down10240 or shared gate/up2560 reduction width.
 * @param timedRows Zero for twenty large/small dirty replays; otherwise a retained row extent.
 * @param comparison Historical column/row tiles, or forced M512 prefill controls.
 * @param format Optional original Q8 registry source; null retains historical Q8_0 bytes.
 * @param geometry Exact large/small graph extents; historical callers retain448/64.
 *
 * All source packing, allocation, byte observations and resource queries are
 * outside timed/captured execution. Timing is one retained prepared projection
 * pipeline per event interval, including its original activation quantizer.
 */
void proveNativeQ8SmallNTiles(DeviceId device,int k,int timedRows=0,
    NativeQ8TileComparison comparison=NativeQ8TileComparison::Columns,
    const test::QuantizedVerifierFormatCase* format=nullptr,
    NativeQ8TileGeometry geometry={448,64}) {
    constexpr int n=320;
    const int large=geometry.large_rows,small=geometry.small_rows;
    require(comparison==NativeQ8TileComparison::Columns||comparison==NativeQ8TileComparison::Rows||
        comparison==NativeQ8TileComparison::Prefill,
        "Native Q8 tile probe received an unknown physical comparison");
    const bool rowTiles=comparison==NativeQ8TileComparison::Rows;
    const bool prefillTiles=comparison==NativeQ8TileComparison::Prefill;
    require(large>small&&small>0&&large<std::numeric_limits<int>::max(),
        "Native Q8 retained geometry requires positive ordered row extents and a guard row");
    require(device.is_rocm()&&(k==2560||k==10240)&&(timedRows==0||timedRows==small||timedRows==large),
        "Native Q8 tile probe requires its original explicit ROCm shape");
    require(!format || (format->device_execution_codebook_id==19 &&
        (format->tensor_type==TensorType::Q8_0 || format->tensor_type==TensorType::Q8_1 ||
         format->tensor_type==TensorType::Q8_K)),"Native Q8 probe requires an original Q8 source");
    require(!timedRows || !format || prefillTiles,
        "Historical Native Q8 paired events retain the original source");
    const auto& initial=debugEnv().rocm;
    require(initial.native_quantised_mt<=0&&initial.native_quantised_unroll<0&&
        initial.native_quantised_min_blocks<=0&&!initial.native_quantised_force_n64&&
        !initial.native_quantised_force_n128&&initial.native_quantised_full_tiles<0,
        "Native Q8 tile probe requires unforced original startup policy");
    auto* backend=getBackendFor(device);
    require(backend&&backend->deviceCount()>device.ordinal,"Native Q8 tile probe participant unavailable");
    auto context=IDeviceContext::create(device,1);
    auto& worker=GPUDeviceContextPool::instance().getContext(device);
    worker.submitAsync([&]{
        const auto& hardware=rocmHardwareMetadata().at(device.ordinal);
        const auto actualHardware=rocm::nativeQuantisedHardwareForDevice(device.ordinal);
        require(rocm::densePrefillExactOverlayEnabled(),"Native Q8 installed Auto lookup is disabled on its capture owner");
        const char* sourceName=format?format->label:"Q8_0";
        const auto sourceCodebook=format?format->source_codebook_id:std::uint8_t{19};
        const bool sourceSuperblock=format&&format->source_is_superblock;
        SCOPED_TRACE(::testing::Message()<<"requested_source="<<sourceName);

        std::ostringstream uuid;
        uuid<<std::hex<<std::setfill('0');for(const auto byte:hardware.uuid)uuid<<std::setw(2)<<unsigned(byte);
        std::cout<<"NATIVE_Q8_TILE_HARDWARE,"<<device.toString()<<','<<hardware.pci_address<<','
            <<hardware.isa_target<<','<<hardware.compute_units<<','<<hardware.native_wave_width<<','<<uuid.str()<<'\n';
        Storage s{device,backend,ExplicitGPUStream(worker.defaultStream()).get()};
        std::unique_ptr<TensorBase> source;
        if(format){
            test::requireQuantizedVerifierSourceShape(*format,n,k);
            source=test::q8_all_formats::source(*format,n,k);
        }else{
            auto authored=nativeWeights(n,k);
            source=std::make_unique<Q8_0Tensor>(std::vector<std::size_t>{n,std::size_t(k)},authored);
        }
        const auto raw=test::q8_all_formats::originalBytes(*source);
        auto prepared=test::makeGpuPreparedGemm(source.get(),device,
            k==10240?"blk.0.hc_attn_down.weight":"blk.0.ffn_gate_shexp.weight",ModelContextId{803120});
        auto* kernel=prepared.kernel;require(kernel,"Native Q8 prepared engine missing");
        kernel->setGPUStream(s.stream);
        DeviceNativeQuantisedMatrixDesc original;
        require(kernel->exportNativeQuantisedMatrixDesc(original),"Native Q8 original source descriptor missing");
        require(original.valid()&&original.source_identity_present&&original.codebook_id==19&&
            original.source_codebook_id==sourceCodebook&&original.source_is_superblock==sourceSuperblock&&
            original.blocks_per_row==unsigned(k/32)&&original.n==n&&original.k==k,
            "Native Q8 probe lost its independent original prepared source identity");
        // Historical Q8_0 additionally owns an exact native payload/half-bit witness.
        // Q8_1/Q8_K keep their actual original factories and preparation contracts;
        // their distinct source tuples are never relabeled as source Q8_0.
        if(sourceCodebook==19)
            authenticate(raw,s.download(static_cast<const std::uint8_t*>(original.payload),std::size_t(n)*k),
                s.download(static_cast<const std::uint16_t*>(original.scales),std::size_t(n)*k/32),n,k);
        ASSERT_FALSE(::testing::Test::HasFailure());
        const auto allocatedRows=std::size_t(large)+1;
        FP32Tensor input({allocatedRows,std::size_t(k)},DeviceId::cpu()),output({allocatedRows,n},DeviceId::cpu());
        std::array<std::vector<float>,2> inputs;
        for(int history=0;history<2;++history){
            inputs[history].resize(std::size_t(large+1)*k);
            for(int row=0;row<=large;++row)for(int column=0;column<k;++column)
                inputs[history][std::size_t(row)*k+column]=row==large?std::numeric_limits<float>::quiet_NaN():
                    column%32==0?float(1+(row+history)%7)*3.125f:float((column*7+row*11+history*19)%127-63)*.03125f;
        }
        std::copy(inputs[0].begin(),inputs[0].end(),input.mutable_data());
        std::fill_n(output.mutable_data(),output.numel(),-7312.5f);
        for(auto* tensor:std::array<TensorBase*,2>{&input,&output}){
            TransferEngine::prepareDeviceInput(tensor,device,s.stream);
            TransferEngine::requireDeviceInput(tensor,device,s.stream);
        }
        auto* consumer=dynamic_cast<IWorkspaceConsumer*>(kernel);
        require(consumer,"Native Q8 engine has no original named workspace contract");
        const auto requirements=consumer->getWorkspaceRequirements(large,n,k);
        Workspace workspace(device,requirements);
        // Declared after every captured byte owner: ASSERT/exception unwind is safe.
        const std::vector choices=prefillTiles?
            std::vector{NativeQ8TileProbe::Installed,NativeQ8TileProbe::NT128,
                NativeQ8TileProbe::NT64,NativeQ8TileProbe::NT64MT16}:
            std::vector{NativeQ8TileProbe::Installed,NativeQ8TileProbe::NT64,
                rowTiles?NativeQ8TileProbe::NT64MT16:NativeQ8TileProbe::NT128};
        const int choiceCount=static_cast<int>(choices.size());
        NativeQ8TileGraphs retained(s,2*choices.size());
        std::vector<const void*> producerFunctions(retained.graphs.size());
        for(int rowCase=0;rowCase<2;++rowCase)for(int choice=0;choice<choiceCount;++choice){
            const int m=rowCase==0?large:small,index=rowCase*choiceCount+choice;
            NativeQ8ColdTileScope cold(choices[choice]);
            retained.stages[index]=std::make_unique<GEMMStage>(GEMMStage::Params{
                .device_id=device,.A=&input,.B=source.get(),.C=&output,.m=m,.n=n,.k=k,.alpha=1,.beta=0,.transpose_B=false,
                .a_buffer_id=BufferId::NORMALIZED,.c_buffer_id=BufferId::GATE_PROJ,
                .prepared_ref=prepared.ref,.prepared_store=prepared.store.get()});
            auto& stage=*retained.stages[index];stage.setGPUStream(s.stream);stage.bindWorkspace(workspace.value.get());
            require(stage.prepareGraphLaunch(context.get(),s.stream),"Native Q8 cold stage preparation failed");
            retained.graphs[index]=worker.createGraphCapture(s.stream);
            rocm::clearVNNIPrefillLaunch();
            {ScopedBackendGraphCapture recording(worker,*retained.graphs[index],"prepared native Q8 tile probe");
                require(recording.begin(),"Native Q8 capture begin failed");
                require(stage.execute(context.get()),"Native Q8 captured stage failed");recording.finish();}
            require(retained.graphs[index]->instantiate(),"Native Q8 graph instantiate failed");
            const auto launch=rocm::lastVNNIPrefillLaunch();
            const auto exact=choice==0?rocm::selectDensePrefillExactConfig(
                19,m,n,k,rocm::DensePrefillWeightFamily::NativeQuantised,&actualHardware):
                std::optional<rocm::DensePrefillExactConfig>{};
            const auto expected=exact?*exact:rocm::DensePrefillExactConfig{
                .n_tile=choice==0?(m==large?128:64):(choices[choice]==NativeQ8TileProbe::NT128?128:64),
                .m_tile=choices[choice]==NativeQ8TileProbe::NT64MT16?16:32,
                .min_blocks=1,.unroll=2,.full_tiles=false,
                .producer=rocm::VNNIPrefillProducer::NativeCooperative};
            const int nt=expected.n_tile,mt=expected.m_tile;
            require(launch&&expected.producer==rocm::VNNIPrefillProducer::NativeCooperative&&
                launch->producer==expected.producer&&launch->execution_codebook==19&&
                launch->n_tile==nt&&launch->m_tile==mt&&launch->min_blocks==expected.min_blocks&&
                launch->unroll==expected.unroll&&launch->full_tiles==expected.full_tiles&&
                launch->function&&launch->block_threads==256,
                "Native Q8 graph did not capture the requested physical cooperative producer");
            producerFunctions[index]=launch->function;
            if(choices[choice]==NativeQ8TileProbe::NT64MT16)
                require(producerFunctions[rowCase*choiceCount+1]!=producerFunctions[index],
                    "Native Q8 MT32 control and MT16 candidate must be distinct actual kernels");
            rocm::VNNIPrefillLaunchResources resources;
            require(rocm::inspectVNNIPrefillLaunch(*launch,resources),"Native Q8 physical resources unavailable");
            require(resources.local_memory_bytes_per_thread==0&&resources.max_active_blocks_per_sm>0,
                "Native Q8 physical candidate has scratch or cannot reside");
            std::vector<GPUGraphKernelNodeInfo> nodes;std::string error;
            require(retained.graphs[index]->inspectKernelNodes(nodes,&error),"Native Q8 graph inventory unavailable");
            int producers=0;
            for(const auto& node:nodes){
                require(node.name_resolved&&node.local_memory_bytes_per_thread==0,"Native Q8 graph has unresolved/scratch kernel");
                // HIP exposes the captured runtime name without promising
                // demangling. The launch receipt owns the exact function;
                // match that identity rather than a display-name substring.
                if(node.function_identity==reinterpret_cast<std::uintptr_t>(launch->function)){
                    ++producers;require(node.grid_x==unsigned((n+nt-1)/nt)&&node.grid_y==unsigned((m+mt-1)/mt)&&
                        node.grid_z==1&&node.block_x==256&&node.block_y==1&&node.block_z==1,
                        "Native Q8 retained graph physical geometry differs from its receipt");
                }
            }
            require(producers==1,"Native Q8 retained graph must contain exactly one cooperative producer");
            std::cout<<(prefillTiles?"NATIVE_Q8_PREFILL_TILE_GRAPH,":
                rowTiles?"NATIVE_Q8_ROW_TILE_GRAPH,":"NATIVE_Q8_TILE_GRAPH,")<<device.toString()<<','<<hardware.pci_address<<','<<hardware.isa_target<<','
                <<m<<','<<n<<','<<k<<','<<choice<<','<<nt<<','<<mt<<','<<expected.min_blocks<<','
                <<expected.unroll<<','<<(expected.full_tiles?"full":"checked")<<','<<nodes.size()<<','
                <<resources.registers_per_thread<<','<<resources.static_shared_memory_bytes<<','
                <<resources.local_memory_bytes_per_thread<<','<<resources.max_active_blocks_per_sm<<','
                <<resources.max_threads_per_block<<','<<launch->dynamic_shared_memory_bytes<<','
                <<reinterpret_cast<std::uintptr_t>(launch->function)<<",requested_source="<<sourceName
                <<",source_codebook="<<unsigned(sourceCodebook)<<",source_superblock="<<sourceSuperblock
                <<",installed_exact="<<bool(exact)<<'\n';
        }
        const auto owners=s.owners->size();
        auto poison=[&](float value){
            s.upload(static_cast<float*>(output.gpu_data_ptr()),std::vector<float>(output.numel(),value));
            for(const auto* name:{GemmWorkspaceBuffers::QUANT_A,GemmWorkspaceBuffers::SCALES_A,
                    GemmWorkspaceBuffers::SCALES_A_BLOCKWISE,GemmWorkspaceBuffers::SUMS_A_BLOCKWISE}){
                auto* buffer=workspace.value->getBuffer(name);const auto bytes=workspace.value->getBufferSize(name);
                if(bytes){require(buffer,"Native Q8 declared workspace lost its original address");
                    require(backend->memset(buffer,0xa5,bytes,device.ordinal,s.stream),"Native Q8 workspace poison failed");}
            }
        };
        // Auto and every forced tile see independent dirty storage every time.
        // Large->small->large replays reset workspace/input data, never topology.
        for(int replay=0;replay<(timedRows?4:20);++replay){
            const int rowCase=(replay%4==1||replay%4==3)?1:0,m=rowCase==0?large:small;
            const float dirty=-7312.5f-float(replay);
            s.upload(static_cast<float*>(input.gpu_data_ptr()),inputs[(replay/2)%2]);
            std::vector<float> oracle;
            // Keep the explicit original MT32 kernel authoritative even when
            // installed Auto and the MT16 candidate become the same function.
            std::vector<int> replayOrder{1,0,2};
            if(prefillTiles)replayOrder.push_back(3);
            for(const int choice:replayOrder){
                poison(dirty);require(retained.graphs[rowCase*choiceCount+choice]->launch(),"Native Q8 dirty replay failed");
                auto actual=s.download(static_cast<const float*>(output.gpu_data_ptr()),output.numel());
                if(choice==1){oracle=actual;for(std::size_t i=0;i<std::size_t(m)*n;++i)ASSERT_TRUE(std::isfinite(actual[i]));}
                else ASSERT_EQ(std::memcmp(actual.data(),oracle.data(),actual.size()*sizeof(float)),0)
                    <<"M="<<m<<" K="<<k<<" replay="<<replay<<" tile="<<choice;
                for(std::size_t i=std::size_t(m)*n;i<actual.size();++i)
                    ASSERT_EQ(std::memcmp(&actual[i],&dirty,sizeof(float)),0)<<"inactive/guard="<<i;
                EXPECT_EQ(s.owners->size(),owners);
            }
        }
        DeviceNativeQuantisedMatrixDesc after;
        require(kernel->exportNativeQuantisedMatrixDesc(after),"Native Q8 source vanished after reset histories");
        EXPECT_EQ(original.payload,after.payload);EXPECT_EQ(original.scales,after.scales);
        EXPECT_EQ(original.codebook_id,after.codebook_id);EXPECT_EQ(original.source_codebook_id,after.source_codebook_id);
        EXPECT_EQ(original.source_is_superblock,after.source_is_superblock);
        EXPECT_EQ(original.source_identity_present,after.source_identity_present);
        ASSERT_EQ(std::memcmp(raw.data(),source->raw_data(),raw.size()),0);
        ASSERT_FALSE(::testing::Test::HasFailure());
        std::cout<<(prefillTiles?"NATIVE_Q8_PREFILL_TILE_BYTE_ORACLE,":
            rowTiles?"NATIVE_Q8_ROW_TILE_BYTE_ORACLE,":"NATIVE_Q8_TILE_BYTE_ORACLE,")
            <<device.toString()<<','<<k<<','<<(timedRows?4:20)
            <<(prefillTiles?",control_nt128mt32_installed_auto_nt64mt32_nt64mt16_dirty_large_small_guard":
                rowTiles?",control_nt64mt32_installed_auto_nt64mt16_dirty_large_small_guard":
                ",control_nt64_installed_auto_nt128_dirty_large_small_guard")
            <<",requested_source="<<sourceName<<",source_codebook="<<unsigned(sourceCodebook)
            <<",source_superblock="<<sourceSuperblock<<",PASS\n";
        if(timedRows){
            // Original exact-stream events observe ONE graph; setup/byte reads
            // and the other candidate cannot contaminate this event interval.
            struct Events {IBackend* backend;DeviceId device;void* stream;void* begin;void* end;
                ~Events(){if(!backend->synchronizeStream(stream,device.ordinal))std::terminate();
                    if(end)backend->destroyEvent(end,device.ordinal);if(begin)backend->destroyEvent(begin,device.ordinal);}
            } events{backend,device,s.stream,backend->createTimingEvent(device.ordinal),backend->createTimingEvent(device.ordinal)};
            require(events.begin&&events.end,"Native Q8 timing events unavailable");
            s.upload(static_cast<float*>(input.gpu_data_ptr()),inputs[0]);
            auto sample=[&](int choice){
                const int index=(timedRows==large?0:choiceCount)+choice;
                require(backend->recordEvent(events.begin,device.ordinal,s.stream),"Native Q8 timing start failed");
                require(retained.graphs[index]->launch(),"Native Q8 timed graph failed");
                require(backend->recordEvent(events.end,device.ordinal,s.stream),"Native Q8 timing stop failed");
                require(backend->waitForEvent(events.end,device.ordinal),"Native Q8 terminal timing observation failed");
                float ms=0;require(backend->eventElapsedTimeMs(events.begin,events.end,device.ordinal,&ms)&&
                    std::isfinite(ms)&&ms>0,"Native Q8 event interval invalid");return ms*1000.f;
            };
            for(int warm=0;warm<5;++warm){
                (void)sample(1);(void)sample(2);if(prefillTiles)(void)sample(3);
            }
            if(prefillTiles){
                // Each forced challenger has its own balanced AB/BA pair against
                // the independent NT128/MT32 baseline; Auto is only byte-checked.
                for(int pair=0;pair<31;++pair)for(const int challenger:{3,2}){
                    float baseline,candidate;
                    if(pair&1){candidate=sample(challenger);baseline=sample(1);}
                    else{baseline=sample(1);candidate=sample(challenger);}
                    std::cout<<"NATIVE_Q8_PREFILL_TILE_PAIR,"<<device.toString()<<','<<hardware.pci_address<<','
                        <<hardware.isa_target<<','<<timedRows<<','<<n<<','<<k<<','<<pair
                        <<",128,32,64,"<<(challenger==3?16:32)<<','<<baseline<<','<<candidate
                        <<",prepared_projection_pipeline_us,requested_source="<<sourceName
                        <<",source_codebook="<<unsigned(sourceCodebook)
                        <<",source_superblock="<<sourceSuperblock<<'\n';
                }
                return;
            }
            for(int pair=0;pair<31;++pair){float a,b;
                if(pair&1){b=sample(2);a=sample(1);}else{a=sample(1);b=sample(2);}
                if(rowTiles)
                    std::cout<<"NATIVE_Q8_ROW_TILE_PAIR,"<<device.toString()<<','<<hardware.pci_address<<','<<hardware.isa_target<<','
                        <<timedRows<<','<<n<<','<<k<<','<<pair<<",64,32,64,16,"<<a<<','<<b<<",prepared_projection_pipeline_us\n";
                else
                    std::cout<<"NATIVE_Q8_TILE_PAIR,"<<device.toString()<<','<<hardware.pci_address<<','<<hardware.isa_target<<','
                        <<timedRows<<','<<n<<','<<k<<','<<pair<<','<<a<<','<<b<<",prepared_projection_pipeline_us\n";
            }
        }
    }).get();
}

/**
 * @brief Run one original source alias at the exact HC-down prefill geometry.
 * @param sourceName Existing verifier-registry source alias, never a relabeled payload.
 * @param timedRows Zero for the byte lifecycle;512 for paired retained events.
 * @throws std::runtime_error For malformed/unavailable explicit device selection.
 * @details Only new M512 cases consume the existing test-only ordinal selector.
 * It changes no visibility, production startup policy or historical fixture.
 */
void proveNativeQ8PrefillTiles(const char* sourceName,int timedRows) {
    int ordinal=0;
    if(const char* selected=std::getenv("LLAMINAR_TEST_ROCM_DEVICE")){
        const char* end=selected+std::strlen(selected);
        const auto parsed=std::from_chars(selected,end,ordinal);
        require(parsed.ec==std::errc{}&&parsed.ptr==end&&ordinal>=0,
            "LLAMINAR_TEST_ROCM_DEVICE requires a nonnegative native ordinal");
    }
    const auto& format=test::quantizedMoEVerifierFormat(sourceName);
    proveNativeQ8SmallNTiles(DeviceId::rocm(ordinal),10240,timedRows,
        NativeQ8TileComparison::Prefill,&format,NativeQ8TileGeometry{512,64});
}
#endif
