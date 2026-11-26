/**
 * @file RetainedPreAttentionGDNReplay.inc
 * @brief Replays the original staged GDN projections against the retained gfx906 entry.
 *
 * The original prepared engines and explicit worker stream own the weights,
 * borrowed workspaces and graph execution. The device count and trunk are
 * independent input owners. Source variants and half policies capture once.
 */
#if defined(HAVE_ROCM)
namespace {
namespace probe=llaminar2::rocm::pre_attention_probe;

/** @brief Original capture source lifetime, including the legal exact POST alias. */
enum class ReplayTrunk { PreviousDisjoint, PreviousInPlace, Published };

/** @brief Replay one physical source topology with the selected original half owner.
 * @param device Authenticated rank-local native gfx906 participant.
 * @param trunk Ownership and alias layout to capture.
 * @param half_policy Whether the original half subgroup remains a separate graph node.
 */
void proveRetainedPreAttentionGDN(DeviceId device, ReplayTrunk trunk, probe::HalfGDNPolicy half_policy)
{
    const bool published=trunk==ReplayTrunk::Published;
    const bool inplace=trunk==ReplayTrunk::PreviousInPlace;
    const bool inline_half=half_policy==probe::HalfGDNPolicy::RetainedPreparedFP16RNE;
    constexpr int H=2560,K=4*H,R=320,QKV=2560,Z=1536,Heads=12;
    auto* backend=getBackendFor(device);
    require(backend&&backend->deviceCount()>0,"Exact ROCm test owner required");
    auto& worker=GPUDeviceContextPool::instance().getContext(device);
    auto context=IDeviceContext::create(device,1);
    worker.submitAndWait([&] {
        Storage storage{device,backend,ExplicitGPUStream(worker.getOrCreateAuxiliaryStream("pre_attention_q8_ab")).get()};
        const std::array shapes{std::array{R,K},std::array{K,R},std::array{QKV,H},std::array{Z,H}};
        std::array<std::vector<std::uint8_t>,4> originals;
        std::array<std::unique_ptr<Q8_0Tensor>,4> sources;
        std::array<test::GpuPreparedGemm,4> prepared;
        std::array<DeviceNativeQuantisedMatrixDesc,4> weights;
        for(unsigned role=0;role<shapes.size();++role) {
            const auto n=shapes[role][0],k=shapes[role][1];
            originals[role]=nativeWeights(n,k);
            sources[role]=std::make_unique<Q8_0Tensor>(std::vector<std::size_t>{unsigned(n),unsigned(k)},originals[role]);
            prepared[role]=test::makeGpuPreparedGemm(sources[role].get(),device,
                "pre_attention.original."+std::to_string(role),ModelContextId{880000+role});
            auto* engine=prepared[role].kernel;
            engine->setGPUStream(storage.stream);
            require(engine->exportNativeQuantisedMatrixDesc(weights[role])&&weights[role].valid()&&weights[role].codebook_id==19,
                    "Original native Q8 prepared operands");
        }
        auto* residual=storage.allocate<float>(K);
        auto* previous=storage.allocate<float>(H);
        auto* write_gate=storage.allocate<float>(4);
        auto* gain=storage.input(std::vector<float>(K,1.0f));
        const auto tensor=[&](int n) {
            auto p=std::make_unique<FP32Tensor>(std::vector<std::size_t>{1u,unsigned(n)},DeviceId::cpu());
            std::fill_n(p->mutable_data(),n,0.0f);
            TransferEngine::prepareDeviceInput(p.get(),device,storage.stream);
            TransferEngine::requireDeviceInput(p.get(),device,storage.stream);
            return p;
        };
        constexpr std::array extents{K,K,H,R,QKV,Z,Heads,Heads};
        std::array<std::array<std::unique_ptr<FP32Tensor>,8>,2> planes;
        for(auto& side:planes)for(unsigned plane=0;plane<extents.size();++plane)
            side[plane]=tensor(extents[plane]);
        const auto ptr=[&](int side,int plane) {
            if(side==1&&plane==0&&inplace)return residual;
            return static_cast<float*>(planes[side][plane]->gpu_data_ptr());
        };
        auto activated=tensor(R),up=tensor(K);
        GEMMStage down_stage({
            .device_id=device,.A=planes[0][1].get(),.B=sources[0].get(),.C=planes[0][3].get(),
            .m=1,.n=R,.k=K,.transpose_B=false,.prepared_ref=prepared[0].ref,.prepared_store=prepared[0].store.get(),
        });
        GEMMStage::Params up_parameters{
            .device_id=device,.A=planes[0][3].get(),.B=sources[1].get(),.C=up.get(),.m=1,.n=K,.k=R,
            .transpose_B=false,.force_decode_equivalent_verifier_prefill=true,
            .verifier_row_range=DeviceRowRange::fullyActive(1),.prepared_ref=prepared[1].ref,
            .prepared_store=prepared[1].store.get(),.quantized_operand_precision=QuantizedActivationOperandPrecision::TwoTermQ8,
        };
        up_parameters.hc_residual_down=GEMMStage::HCResidualDownPublication{activated.get(),BufferId::UP_PROJ};
        up_parameters.a_buffer_id=BufferId::GATE_PROJ;
        up_parameters.c_buffer_id=BufferId::ATTN_OUTPUT;
        GEMMStage up_stage(up_parameters);
        Workspace down_workspace(device,down_stage.getWorkspaceRequirements(1,R,K));
        Workspace up_workspace(device,up_stage.getWorkspaceRequirements(1,K,R));
        down_stage.setGPUStream(storage.stream);up_stage.setGPUStream(storage.stream);
        down_stage.bindWorkspace(down_workspace.value.get());up_stage.bindWorkspace(up_workspace.value.get());
        require(down_stage.prepareGraphLaunch(context.get(),storage.stream)&&up_stage.prepareGraphLaunch(context.get(),storage.stream),
                "Original staged HC preparation");
        struct StageBorrows {
            GEMMStage &down,&up;
            ~StageBorrows(){down.unbindWorkspace();up.unbindWorkspace();}
        } borrows{down_stage,up_stage};
        std::vector<ITensorGemm::TensorProjectionDesc> original_projections{
            {prepared[2].kernel,planes[0][4].get(),QKV,nullptr,"qkv"},
            {prepared[3].kernel,planes[0][5].get(),Z,nullptr,"z"}};
        auto* engine=dynamic_cast<rocm::ROCmQuantisedGemmKernel*>(prepared[2].kernel);
        require(engine!=nullptr,"Original shared native GDN Q8 engine");
        auto projection_requirements=engine->getWorkspaceRequirements(1,QKV,H);
        const std::array widths{QKV,Z};
        engine->appendFusedProjectionWorkspaceRequirements(projection_requirements,1,widths,H);
        Workspace projection_workspace(device,projection_requirements);
        auto* projected_z=dynamic_cast<rocm::ROCmQuantisedGemmKernel*>(prepared[3].kernel);
        require(projected_z!=nullptr,"Original native GDN Z engine");
        struct ProjectionBorrows {
            rocm::ROCmQuantisedGemmKernel& qkv;
            rocm::ROCmQuantisedGemmKernel& z;
            ~ProjectionBorrows(){qkv.unbindWorkspace();z.unbindWorkspace();}
        } projection_borrows{*engine,*projected_z};
        engine->bindWorkspace(projection_workspace.value.get());
        projected_z->bindWorkspace(projection_workspace.value.get());
        using HalfEngine=rocm::ROCmFloatingPointGemmKernel;
        std::array<std::unique_ptr<HalfEngine>,2> half_engines;
        std::array<const std::uint16_t*,2> prepared_half{};
        std::array<std::vector<ITensorGemm::TensorProjectionDesc>,2> half_projections;
        for(unsigned role=0;role<2;++role) {
            std::vector<std::uint16_t> packed(Heads*H);
            for(std::size_t i=0;i<packed.size();++i)
                packed[i]=fp32_to_fp16(float(int((i*17+role*13)%1021)-510)*0.000244140625f);
            prepared_half[role]=storage.input(packed);
            half_engines[role]=std::make_unique<HalfEngine>(prepared_half[role],Heads,H,
                device.ordinal,HalfEngine::Precision::FP16,storage.owners,FloatingActivationOperandPrecision::FP16RNE);
            half_engines[role]->setGPUStream(storage.stream);
            for(unsigned side=0;side<2;++side)
                half_projections[side].push_back({half_engines[role].get(),planes[side][6+role].get(),
                    Heads,nullptr,role?"beta":"alpha"});
        }
        Workspace half_workspace(device,half_engines[0]->getWorkspaceRequirements(1,Heads,H));
        struct HalfBorrow {
            std::array<std::unique_ptr<HalfEngine>,2>& engines;
            ~HalfBorrow() { for(auto& engine:engines)engine->unbindWorkspace(); }
        } half_borrow{half_engines};
        for(auto& engine:half_engines)engine->bindWorkspace(half_workspace.value.get());
        auto primitives=qwen4exp::createROCmQwen4ExpKernels();
        const qwen4exp::RowExtent extent{1,nullptr};
        auto* state=storage.input(std::vector<std::byte>(probe::scratchBytes()));
        auto* live=storage.input(std::vector<int>{1});
        const auto rows=DeviceRowRange::deviceCounted(1,live);
        auto* hc_codes=storage.allocate<std::int8_t>(K);
        auto* hc_scales=storage.allocate<float>(K/32);
        auto* hc_partials=storage.allocate<float>(16*R);
        auto* hc_res=storage.allocate<std::int8_t>(R);
        auto* hc_rs=storage.allocate<float>(R/32);
        probe::Invocation invocation{.prefix={.hc={.publication={rows,R,nullptr,hc_codes,hc_res,hc_scales,hc_rs},
            .up={.rows=rows,.primary=hc_codes,.residual=hc_res,.primaryScales=hc_scales,.residualScales=hc_rs}}}};
        invocation.geometry=probe::EarlyQ8Geometry::GDN;
        if(inline_half) {
            constexpr std::size_t bytes=Heads*H*sizeof(std::uint16_t);
            invocation.half_policy=probe::HalfGDNPolicy::RetainedPreparedFP16RNE;
            invocation.half={{prepared_half[0],bytes},{prepared_half[1],bytes},ptr(1,6),ptr(1,7),Heads};
        }
        invocation.scratch=state;invocation.scratch_bytes=probe::scratchBytes();
        invocation.prefix.output_demand=FullNativeHCFFNOutputDemand::none();
        if(published)invocation.trunk_source=probe::TrunkSource::PublishedTrunk;
        else invocation.prefix.write={.residual=residual,.block=previous,.gateLogits=write_gate,.output=ptr(1,0)};
        auto& hc=invocation.prefix.hc;
        hc.norm={{1,rows.countOwner()},H,1e-6f,published?residual:ptr(1,0),gain,ptr(1,1)};
        hc.down_weights=weights[0];hc.input_codes=hc_codes;hc.input_scales=hc_scales;hc.down_partials=hc_partials;hc.raw_down=ptr(1,3);
        hc.up.weights=weights[1];hc.up.rows=rows;
        hc.up.quantizer=Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal;
        hc.mixed=ptr(1,2);
        invocation.projections[0]={weights[2],ptr(1,4)};
        invocation.projections[1]={weights[3],ptr(1,5)};
        probe::Admission candidate{.invocation=invocation};
        require(probe::prepare(invocation,device.ordinal,candidate)==hipSuccess,"Retained pre-attention geometry");
        if(published) {
            auto stale=invocation;
            stale.prefix.write.output=ptr(1,0);
            require(probe::prepare(stale,device.ordinal,candidate)==hipErrorInvalidValue,
                    "Published trunk rejects an additional write owner");
        }
        if(inline_half) {
            auto truncated=invocation;
            truncated.half.beta.bytes-=sizeof(std::uint16_t);
            require(probe::prepare(truncated,device.ordinal,candidate)==hipErrorInvalidValue,
                    "Retained half admission rejects truncated prepared weights");
        }
        const auto stream=static_cast<hipStream_t>(storage.stream);
        const auto run=[&](int side) {
            if(side==1) {
                require(probe::launch(candidate,stream)==hipSuccess,"Submit exact retained pre-attention");
            } else {
                if(!published)
                    primitives->residualWrite({extent,H,residual,previous,write_gate,ptr(0,0)},storage.stream);
                primitives->branchRMSNorm({extent,H,1e-6f,published?residual:ptr(0,0),gain,ptr(0,1)},storage.stream);
                require(down_stage.execute(context.get())&&up_stage.execute(context.get()),"Original staged HC projections");
                primitives->residualRead({extent,H,ptr(0,1),static_cast<const float*>(up->gpu_data_ptr()),ptr(0,2)},storage.stream);
                require(engine->multiply_fused_tensor(planes[0][2].get(),original_projections,1,H,nullptr,projection_workspace.value.get()),
                        "Original shared SingleQ8 early bundle");
            }
            if(side==0||!inline_half) {
                ITensorGemm::VerifierKernelModeScope scope(rows);
                require(half_engines[0]->multiply_fused_tensor(planes[side][2].get(),half_projections[side],
                    1,H,nullptr,half_workspace.value.get()),"Original FP16RNE GDN alpha/beta bundle");
            }
        };
        std::array<std::unique_ptr<IGPUGraphCapture>,2> graphs;
        const auto capture=[&](unsigned copies) {
            for(unsigned side=0;side<graphs.size();++side) {
                graphs[side]=worker.createGraphCapture(storage.stream);
                ScopedBackendGraphCapture recording(worker,*graphs[side],"Original POST through native Q8 early boundary");
                require(recording.begin(),"Begin exact pre-attention capture");
                for(unsigned i=0;i<copies;++i)run(side);
                recording.finish();
                require(graphs[side]->instantiate(),"Instantiate original pre-attention capture");
                std::vector<GPUGraphKernelNodeInfo> nodes;
                std::string error;
                require(graphs[side]->inspectKernelNodes(nodes,&error)&&!nodes.empty(),"Authenticate captured native pre-attention nodes");
                if(side==1)require(nodes.size()==copies*(inline_half?1:3),"Exact retained GDN native phase count");
                for(const auto& node:nodes)require(node.name_resolved&&node.local_memory_bytes_per_thread==0,"Authentic and spill-free native node");
                std::cout<<"PRE_ATTENTION_CAPTURE,arm="<<side<<",nodes="<<nodes.size()<<",copies="<<copies<<'\n';
            }
        };
        const auto input=[&](int history) {
            std::vector<float> r(K),p(H),g(4);
            for(int x=0;x<K;++x)r[x]=std::sin(float(x*3+history*17))*.21f;
            for(int x=0;x<H;++x)p[x]=std::cos(float(x*7+history*11))*.13f;
            for(int x=0;x<4;++x)g[x]=float(history-x)/64.f;
            storage.upload(residual,r);storage.upload(previous,p);storage.upload(write_gate,g);
        };
        try {
            input(0);capture(1);
            for(int history=0;history<20;++history)for(unsigned repeat=0;repeat<2;++repeat) {
                if(history==13 && repeat==0)
                    require(backend->memset(state,0,probe::scratchBytes(),device.ordinal,storage.stream),
                            "Reset retained GDN scratch without capture replacement");
                input(history==13?0:history);
                const auto original_trunk=storage.download(residual,K);
                for(unsigned side=0;side<graphs.size();++side)
                    require(graphs[side]->launch(),"Original pre-attention replay");
                if(published) {
                    const auto actual=storage.download(residual,K);
                    require(std::memcmp(actual.data(),original_trunk.data(),K*sizeof(float))==0,
                            "Published trunk source bytes changed");
                }
                for(unsigned plane=published?1:0;plane<extents.size();++plane) {
                    const auto expected=storage.download(ptr(0,plane),extents[plane]);
                    const auto actual=storage.download(ptr(1,plane),extents[plane]);
                    require(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float))==0,
                            ("Pre-attention staged byte mismatch at plane "+std::to_string(plane)+
                             ", history "+std::to_string(history)).c_str());
                }
                if(history%3==0) {
                    const auto original_down=storage.download(ptr(0,3),R);
                    storage.upload(live,std::vector<int>{0});
                    // The original HC prefix's counted empty scope still
                    // publishes raw down from prior normalized bits. All
                    // inactive suffix outputs and the existing trunk retain.
                    for(unsigned plane=2;plane<extents.size();++plane)
                        storage.upload(ptr(1,plane),std::vector<float>(extents[plane],-99.125f));
                    const auto trunk_before_empty=storage.download(residual,K);
                    require(graphs[1]->launch(),"Retained count0 replay");
                    const auto after_empty=storage.download(residual,K);
                    require(std::memcmp(after_empty.data(),trunk_before_empty.data(),K*sizeof(float))==0,
                            "Empty launch touched existing trunk");
                    for(unsigned plane=1;plane<extents.size();++plane) {
                        const auto actual=storage.download(ptr(1,plane),extents[plane]);
                        if(plane==1) {
                            const auto expected=storage.download(ptr(0,plane),extents[plane]);
                            require(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float))==0,
                                    "Empty launch touched normalized output");
                        } else if(plane==3) {
                            require(std::memcmp(actual.data(),original_down.data(),R*sizeof(float))==0,
                                    "Empty launch did not republish raw down");
                        } else {
                            require(std::all_of(actual.begin(),actual.end(),[](float x){return x==-99.125f;}),
                                    "Empty launch touched an inactive mixed/projection word");
                        }
                    }
                    storage.upload(live,std::vector<int>{1});
                }
            }
            std::cout<<"PRE_ATTENTION_BYTE_ORACLE,published="<<published<<",inplace="<<inplace
                     <<",inline_half="<<inline_half<<",histories=20,repeats=2,count0=14\n";
        } catch(...) {
            require(backend->synchronizeStream(storage.stream,device.ordinal),"Retire failing pre-attention captures");
            throw;
        }
        require(backend->synchronizeStream(storage.stream,device.ordinal),"Retire completed pre-attention captures");
        for(auto& graph:graphs)graph.reset();
    });
}
} // namespace
/** @test One captured native graph per source/half policy with reset and empty histories. */
TEST(Q8ResidualCorrectionDevice, ROCmRetainedPreAttentionGDN)
{
    auto* backend=getBackendFor(DeviceId::rocm(0));
    require(backend&&backend->deviceCount()>0,"ROCm lane requires at least one visible physical device");
    const auto& metadata=rocmHardwareMetadata();
    for(std::size_t index=0;index<metadata.size();++index) {
        const auto& device=metadata.at(static_cast<int>(index));
        if(device.compute_units!=60||device.native_wave_width!=64||
           (device.isa_target!="gfx906"&&!device.isa_target.starts_with("gfx906:")))continue;
        for(auto half:{probe::HalfGDNPolicy::SeparateOriginalSubgroup,probe::HalfGDNPolicy::RetainedPreparedFP16RNE})
            for(auto trunk:{ReplayTrunk::PreviousDisjoint,ReplayTrunk::PreviousInPlace,ReplayTrunk::Published})
                proveRetainedPreAttentionGDN(DeviceId::rocm(device.ordinal),trunk,half);
        return;
    }
    GTEST_SKIP()<<"The retained pre-attention leaf requires a native 60-CU gfx906 participant";
}

/** @brief Replay the full GDN attention megakernel (HC + QKV/Z/A/B + conv + recurrence + gated_norm + ssm_out) against staged ROCm. */
static void proveRetainedFullGDNMegakernel(DeviceId device, int n_k_heads, int n_v_heads)
{
    constexpr int H=2560,K=4*H,R=320,D=128;
    const int QKV=2*n_k_heads*D+n_v_heads*D;
    const int Z=n_v_heads*D;
    const int Heads=n_v_heads;
    auto* backend=getBackendFor(device);
    require(backend&&backend->deviceCount()>0,"Exact ROCm test owner required");
    auto& worker=GPUDeviceContextPool::instance().getContext(device);
    auto context=IDeviceContext::create(device,1);
    worker.submitAndWait([&] {
        Storage storage{device,backend,ExplicitGPUStream(worker.getOrCreateAuxiliaryStream("full_gdn_mk_replay")).get()};
        const std::array shapes{std::array{R,K},std::array{K,R},std::array{QKV,H},std::array{Z,H},std::array{H,Z}};
        std::array<std::vector<std::uint8_t>,5> originals;
        std::array<std::unique_ptr<Q8_0Tensor>,5> sources;
        std::array<test::GpuPreparedGemm,5> prepared;
        std::array<DeviceNativeQuantisedMatrixDesc,5> weights;
        for(unsigned role=0;role<shapes.size();++role) {
            const auto n=shapes[role][0],k=shapes[role][1];
            originals[role]=nativeWeights(n,k);
            sources[role]=std::make_unique<Q8_0Tensor>(std::vector<std::size_t>{unsigned(n),unsigned(k)},originals[role]);
            prepared[role]=test::makeGpuPreparedGemm(sources[role].get(),device,
                "full_gdn.orig."+std::to_string(n_v_heads)+"."+std::to_string(role),
                ModelContextId{890000+unsigned(n_v_heads)*16u+role});
            auto* engine=prepared[role].kernel;
            engine->setGPUStream(storage.stream);
            require(engine->exportNativeQuantisedMatrixDesc(weights[role])&&weights[role].valid()&&weights[role].codebook_id==19,
                    "Original native Q8 prepared operands");
        }
        auto* residual=storage.allocate<float>(K);
        auto* previous=storage.allocate<float>(H);
        auto* write_gate=storage.allocate<float>(4);
        auto* gain=storage.input(std::vector<float>(K,1.0f));
        std::vector<float> h_conv_w(std::size_t(QKV)*4u), h_a_log(Heads), h_dt_bias(Heads), h_gamma(D);
        for(std::size_t i=0;i<h_conv_w.size();++i) h_conv_w[i]=std::sin(float(i*5+3))*0.15f;
        for(int i=0;i<Heads;++i) { h_a_log[i]=-0.25f-0.02f*float(i); h_dt_bias[i]=0.05f*float(i-Heads/2); }
        for(int i=0;i<D;++i) h_gamma[i]=0.85f+0.002f*float(i);
        auto* d_conv_w=storage.input(h_conv_w);
        auto* d_a_log=storage.input(h_a_log);
        auto* d_dt_bias=storage.input(h_dt_bias);
        const auto tensor=[&](int n) {
            auto p=std::make_unique<FP32Tensor>(std::vector<std::size_t>{1u,unsigned(n)},DeviceId::cpu());
            std::fill_n(p->mutable_data(),n,0.0f);
            TransferEngine::prepareDeviceInput(p.get(),device,storage.stream);
            TransferEngine::requireDeviceInput(p.get(),device,storage.stream);
            return p;
        };
        auto gamma_t=std::make_unique<FP32Tensor>(std::vector<std::size_t>{unsigned(D)},DeviceId::cpu());
        std::copy(h_gamma.begin(),h_gamma.end(),gamma_t->mutable_data());
        TransferEngine::prepareDeviceInput(gamma_t.get(),device,storage.stream);
        TransferEngine::requireDeviceInput(gamma_t.get(),device,storage.stream);

        // Planes: 0=post,1=norm,2=mixed,3=raw_down,4=qkv,5=z,6=alpha,7=beta,8=conv_out,9=attn_out,10=attn_proj
        const std::array extents{K,K,H,R,QKV,Z,Heads,Heads,QKV,Z,H};
        std::array<std::array<std::unique_ptr<FP32Tensor>,11>,2> planes;
        for(auto& side:planes)for(unsigned p=0;p<extents.size();++p) side[p]=tensor(extents[p]);
        const auto ptr=[&](int side,int plane) { return static_cast<float*>(planes[side][plane]->gpu_data_ptr()); };
        auto* d_prep_qk=storage.allocate<float>(2*n_k_heads*D);
        std::array<float*,2> conv_states{storage.allocate<float>(QKV*3), storage.allocate<float>(QKV*3)};
        std::array<float*,2> rec_states{storage.allocate<float>(Heads*D*D), storage.allocate<float>(Heads*D*D)};

        auto activated=tensor(R),up=tensor(K);
        GEMMStage down_stage({.device_id=device,.A=planes[0][1].get(),.B=sources[0].get(),.C=planes[0][3].get(),
            .m=1,.n=R,.k=K,.transpose_B=false,.prepared_ref=prepared[0].ref,.prepared_store=prepared[0].store.get()});
        GEMMStage::Params up_parameters{.device_id=device,.A=planes[0][3].get(),.B=sources[1].get(),.C=up.get(),
            .m=1,.n=K,.k=R,.transpose_B=false,.force_decode_equivalent_verifier_prefill=true,
            .verifier_row_range=DeviceRowRange::fullyActive(1),.prepared_ref=prepared[1].ref,
            .prepared_store=prepared[1].store.get(),.quantized_operand_precision=QuantizedActivationOperandPrecision::TwoTermQ8};
        up_parameters.hc_residual_down=GEMMStage::HCResidualDownPublication{activated.get(),BufferId::UP_PROJ};
        up_parameters.a_buffer_id=BufferId::GATE_PROJ; up_parameters.c_buffer_id=BufferId::ATTN_OUTPUT;
        GEMMStage up_stage(up_parameters);
        Workspace down_ws(device,down_stage.getWorkspaceRequirements(1,R,K));
        Workspace up_ws(device,up_stage.getWorkspaceRequirements(1,K,R));
        down_stage.setGPUStream(storage.stream); up_stage.setGPUStream(storage.stream);
        down_stage.bindWorkspace(down_ws.value.get()); up_stage.bindWorkspace(up_ws.value.get());
        require(down_stage.prepareGraphLaunch(context.get(),storage.stream)&&up_stage.prepareGraphLaunch(context.get(),storage.stream),
                "Original staged HC preparation");
        struct StageBorrows { GEMMStage &d,&u; ~StageBorrows(){d.unbindWorkspace();u.unbindWorkspace();} } sb{down_stage,up_stage};

        std::vector<ITensorGemm::TensorProjectionDesc> orig_proj{
            {prepared[2].kernel,planes[0][4].get(),QKV,nullptr,"qkv"},
            {prepared[3].kernel,planes[0][5].get(),Z,nullptr,"z"}};
        auto* qkv_eng=dynamic_cast<rocm::ROCmQuantisedGemmKernel*>(prepared[2].kernel);
        auto* z_eng=dynamic_cast<rocm::ROCmQuantisedGemmKernel*>(prepared[3].kernel);
        auto* ssm_eng=dynamic_cast<rocm::ROCmQuantisedGemmKernel*>(prepared[4].kernel);
        auto proj_req=qkv_eng->getWorkspaceRequirements(1,QKV,H);
        const std::array widths{QKV,Z};
        qkv_eng->appendFusedProjectionWorkspaceRequirements(proj_req,1,widths,H);
        Workspace proj_ws(device,proj_req);
        Workspace ssm_ws(device,ssm_eng->getWorkspaceRequirements(1,H,Z));
        struct ProjBorrows { rocm::ROCmQuantisedGemmKernel &a,&b,&c; ~ProjBorrows(){a.unbindWorkspace();b.unbindWorkspace();c.unbindWorkspace();} } pb{*qkv_eng,*z_eng,*ssm_eng};
        qkv_eng->bindWorkspace(proj_ws.value.get()); z_eng->bindWorkspace(proj_ws.value.get()); ssm_eng->bindWorkspace(ssm_ws.value.get());

        using HalfEngine=rocm::ROCmFloatingPointGemmKernel;
        std::array<std::unique_ptr<HalfEngine>,2> half_engines;
        std::array<const std::uint16_t*,2> prepared_half{};
        std::vector<ITensorGemm::TensorProjectionDesc> half_proj;
        for(unsigned role=0;role<2;++role) {
            std::vector<std::uint16_t> packed(std::size_t(Heads)*H);
            for(std::size_t i=0;i<packed.size();++i)
                packed[i]=fp32_to_fp16(float(int((i*17+role*13)%1021)-510)*0.000244140625f);
            prepared_half[role]=storage.input(packed);
            half_engines[role]=std::make_unique<HalfEngine>(prepared_half[role],Heads,H,
                device.ordinal,HalfEngine::Precision::FP16,storage.owners,FloatingActivationOperandPrecision::FP16RNE);
            half_engines[role]->setGPUStream(storage.stream);
            half_proj.push_back({half_engines[role].get(),planes[0][6+role].get(),Heads,nullptr,role?"beta":"alpha"});
        }
        Workspace half_ws(device,half_engines[0]->getWorkspaceRequirements(1,Heads,H));
        struct HalfBorrow { std::array<std::unique_ptr<HalfEngine>,2>& e; ~HalfBorrow(){for(auto& x:e)x->unbindWorkspace();} } hb{half_engines};
        for(auto& e:half_engines) e->bindWorkspace(half_ws.value.get());

        auto primitives=qwen4exp::createROCmQwen4ExpKernels();
        const qwen4exp::RowExtent extent{1,nullptr};
        auto* state=storage.input(std::vector<std::byte>(probe::scratchBytes()));
        auto* live=storage.input(std::vector<int>{1});
        const auto rows=DeviceRowRange::deviceCounted(1,live);

        auto* hc_codes=storage.allocate<std::int8_t>(K);
        auto* hc_scales=storage.allocate<float>(K/32);
        auto* hc_partials=storage.allocate<float>(16*R);
        auto* hc_res=storage.allocate<std::int8_t>(R);
        auto* hc_rs=storage.allocate<float>(R/32);
        probe::Invocation inv{.prefix={.hc={.publication={rows,R,nullptr,hc_codes,hc_res,hc_scales,hc_rs},
            .up={.rows=rows,.primary=hc_codes,.residual=hc_res,.primaryScales=hc_scales,.residualScales=hc_rs}}}};
        inv.geometry=probe::EarlyQ8Geometry::GDN;
        inv.half_policy=probe::HalfGDNPolicy::RetainedPreparedFP16RNE;
        const std::size_t half_bytes=std::size_t(Heads)*H*sizeof(std::uint16_t);
        inv.half={{prepared_half[0],half_bytes},{prepared_half[1],half_bytes},ptr(1,6),ptr(1,7),Heads};
        inv.scratch=state; inv.scratch_bytes=probe::scratchBytes();
        inv.prefix.output_demand=FullNativeHCFFNOutputDemand::none();
        inv.prefix.write={.residual=residual,.block=previous,.gateLogits=write_gate,.output=ptr(1,0)};
        auto& hc=inv.prefix.hc;
        hc.norm={{1,rows.countOwner()},H,1e-6f,ptr(1,0),gain,ptr(1,1)};
        hc.down_weights=weights[0]; hc.input_codes=hc_codes; hc.input_scales=hc_scales; hc.down_partials=hc_partials; hc.raw_down=ptr(1,3);
        hc.up.weights=weights[1]; hc.up.rows=rows;
        hc.up.quantizer=Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal;
        hc.mixed=ptr(1,2);
        inv.projections[0]={weights[2],ptr(1,4)};
        inv.projections[1]={weights[3],ptr(1,5)};
        auto& fg=inv.full_gdn;
        fg.enabled=true;
        fg.conv_weight=d_conv_w; fg.conv_bias=nullptr; fg.conv_state=conv_states[1];
        fg.gdn_recurrence_in=ptr(1,8);
        fg.A_log=d_a_log; fg.dt_bias=d_dt_bias; fg.recurrence_state=rec_states[1];
        fg.attn_output=ptr(1,9);
        fg.ssm_norm_gamma=static_cast<const float*>(gamma_t->gpu_data_ptr());
        fg.ssm_norm_eps=1e-6f; fg.ssm_norm_subtract_one=true;
        fg.ssm_norm_activation=GatedNormActivation::Sigmoid;
        fg.n_k_heads=n_k_heads; fg.n_v_heads=n_v_heads; fg.global_v_offset=0; fg.use_qk_l2norm=true;
        fg.ssm_out={weights[4],ptr(1,10)};
        probe::Admission cand{.invocation=inv};
        const auto prep_err = probe::prepare(inv,device.ordinal,cand);
        if (prep_err != hipSuccess) {
            std::cout<<"PREPARE_FAIL err="<<int(prep_err)<<" ("<<hipGetErrorString(prep_err)<<")"
                     <<" qkv.n="<<inv.projections[0].weights.n<<" z.n="<<inv.projections[1].weights.n
                     <<" ssm.n="<<fg.ssm_out.weights.n<<" ssm.k="<<fg.ssm_out.weights.k
                     <<" half.heads="<<inv.half.heads<<" half.bytes="<<inv.half.alpha.bytes
                     <<" rec_state_align="<<(reinterpret_cast<std::uintptr_t>(fg.recurrence_state)%16)
                     <<" rec_in_align="<<(reinterpret_cast<std::uintptr_t>(fg.gdn_recurrence_in)%16)<<'\n';
        }
        require(prep_err==hipSuccess,"Prepare Full GDN megakernel");

        const auto stream=static_cast<hipStream_t>(storage.stream);
        const auto run=[&](int side) {
            if(side==1) {
                require(probe::launch(cand,stream)==hipSuccess,"Launch Full GDN megakernel");
                return;
            }
            primitives->residualWrite({extent,H,residual,previous,write_gate,ptr(0,0)},storage.stream);
            primitives->branchRMSNorm({extent,H,1e-6f,ptr(0,0),gain,ptr(0,1)},storage.stream);
            require(down_stage.execute(context.get())&&up_stage.execute(context.get()),"Staged HC");
            primitives->residualRead({extent,H,ptr(0,1),static_cast<const float*>(up->gpu_data_ptr()),ptr(0,2)},storage.stream);
            require(qkv_eng->multiply_fused_tensor(planes[0][2].get(),orig_proj,1,H,nullptr,proj_ws.value.get()),"Staged QKV/Z");
            {
                ITensorGemm::VerifierKernelModeScope scope(rows);
                require(half_engines[0]->multiply_fused_tensor(planes[0][2].get(),half_proj,1,H,nullptr,half_ws.value.get()),"Staged A/B");
            }
            require(gdn::launchROCmScalarPreparedQKConvolution({{n_k_heads,n_v_heads,D,D,true,
                ptr(0,4),d_conv_w,nullptr,ptr(0,8),d_prep_qk},conv_states[0],conv_states[0],nullptr,0,0},storage.stream),"Staged conv");
            gdn::MergedRecurrenceInput rec_in{DeviceRequestRowRanges::fullyActiveRequests(1,1),n_k_heads,n_v_heads,D,D,0,true,
                ptr(0,8),ptr(0,6),ptr(0,7),d_a_log,d_dt_bias,ptr(0,9),rec_states[0],rec_states[0],nullptr,0,0,
                GDNRecurrenceStateLayout::ValueMajorKeyContiguous};
            require(gdn::launchROCmScalarPreparedQKRecurrence({rec_in,d_prep_qk},storage.stream),"Staged recurrence");
            GatedRMSNormProjectionInput gnorm{*planes[0][9],*planes[0][5],*gamma_t,
                {D,D,1e-6f,true,GatedNormActivation::Sigmoid}};
            require(ssm_eng->multiply_gated_rmsnorm_single_q8(gnorm,planes[0][10].get(),H,Z,ssm_ws.value.get()),"Staged gated norm+ssm_out");
        };

        std::array<std::unique_ptr<IGPUGraphCapture>,2> graphs;
        for(unsigned side=0;side<2;++side) {
            graphs[side]=worker.createGraphCapture(storage.stream);
            ScopedBackendGraphCapture rec(worker,*graphs[side],"Full GDN replay");
            require(rec.begin(),"Begin capture");
            run(int(side));
            rec.finish();
            require(graphs[side]->instantiate(),"Instantiate capture");
            std::vector<GPUGraphKernelNodeInfo> nodes; std::string err;
            require(graphs[side]->inspectKernelNodes(nodes,&err)&&!nodes.empty(),"Inspect nodes");
            if(side==1) require(nodes.size()==1,"Full GDN megakernel must be exactly 1 GPU kernel node");
            std::cout<<"FULL_GDN_CAPTURE,nv="<<n_v_heads<<",arm="<<side<<",nodes="<<nodes.size()<<'\n';
        }
        const std::array<const char*,11> pnames{"post","norm","mixed","raw_down","qkv","z","alpha","beta","conv_out","attn_out","attn_proj"};
        for(int side=0;side<2;++side) {
            require(backend->memset(conv_states[side],0,std::size_t(QKV)*3*sizeof(float),device.ordinal,storage.stream),"Zero conv state");
            require(backend->memset(rec_states[side],0,std::size_t(Heads)*D*D*sizeof(float),device.ordinal,storage.stream),"Zero rec state");
        }
        for(int step=0;step<8;++step) {
            std::vector<float> r(K),p(H),g(4);
            for(int x=0;x<K;++x) r[x]=std::sin(float(x*3+step*17))*0.21f;
            for(int x=0;x<H;++x) p[x]=std::cos(float(x*7+step*11))*0.13f;
            for(int x=0;x<4;++x) g[x]=float(step-x)/64.f;
            storage.upload(residual,r); storage.upload(previous,p); storage.upload(write_gate,g);
            for(unsigned side=0;side<2;++side) require(graphs[side]->launch(),"Replay step");
            auto checkVec=[&](const char* label, const std::vector<float>& exp, const std::vector<float>& act) {
                std::size_t diffs=0; float max_abs=0.0f; std::size_t first_i=0;
                for(std::size_t i=0;i<exp.size();++i) {
                    std::uint32_t eb, ab;
                    std::memcpy(&eb,&exp[i],4); std::memcpy(&ab,&act[i],4);
                    if(eb!=ab) {
                        if(diffs==0) first_i=i;
                        ++diffs;
                        max_abs=std::max(max_abs,std::fabs(exp[i]-act[i]));
                    }
                }
                if(diffs>0) {
                    std::uint32_t eb, ab;
                    std::memcpy(&eb,&exp[first_i],4); std::memcpy(&ab,&act[first_i],4);
                    std::cout<<"MISMATCH nv="<<n_v_heads<<" step="<<step<<" plane="<<label
                             <<" diffs="<<diffs<<"/"<<exp.size()<<" first="<<first_i
                             <<" exp="<<exp[first_i]<<"(0x"<<std::hex<<eb<<") act="<<act[first_i]
                             <<"(0x"<<ab<<std::dec<<") max_abs="<<max_abs<<'\n';
                }
                return diffs==0;
            };
            bool all_ok=true;
            for(unsigned pl=0;pl<extents.size();++pl)
                all_ok&=checkVec(pnames[pl],storage.download(ptr(0,int(pl)),extents[pl]),storage.download(ptr(1,int(pl)),extents[pl]));
            all_ok&=checkVec("conv_state",storage.download(conv_states[0],QKV*3),storage.download(conv_states[1],QKV*3));
            all_ok&=checkVec("rec_state",storage.download(rec_states[0],Heads*D*D),storage.download(rec_states[1],Heads*D*D));
            if(!all_ok) {
                require(backend->synchronizeStream(storage.stream,device.ordinal),"Sync before fail");
                for(auto& g:graphs) g.reset();
                FAIL()<<"Bit-exact mismatch in Full GDN Megakernel at step "<<step;
            }
        }
        // Hardware timing comparison (500 graph replays on MI50) + per-phase breakdown
        hipEvent_t t0{}, t1{};
        require(hipEventCreate(&t0)==hipSuccess&&hipEventCreate(&t1)==hipSuccess,"Events");
        for(unsigned side=0;side<2;++side) {
            for(int i=0;i<20;++i) require(graphs[side]->launch(),"Warmup");
            require(hipEventRecord(t0,stream)==hipSuccess,"t0");
            for(int i=0;i<500;++i) require(graphs[side]->launch(),"Timed");
            require(hipEventRecord(t1,stream)==hipSuccess&&hipEventSynchronize(t1)==hipSuccess,"t1");
            float ms=0.0f;
            require(hipEventElapsedTime(&ms,t0,t1)==hipSuccess,"elapsed");
            std::cout<<"FULL_GDN_TIMING,nv="<<n_v_heads<<",arm="<<(side==0?"staged":"megakernel")
                     <<",us_per_layer="<<(ms*1000.0f/500.0f)<<",grid="<<cand.grid<<'\n';
        }
        // Also time pre-attention-only megakernel (HC + QKV + Z + A/B) to isolate prefix vs recurrence+ssm_out
        {
            auto pre_inv = inv;
            pre_inv.full_gdn = {};
            probe::Admission pre_cand{.invocation = pre_inv};
            require(probe::prepare(pre_inv, device.ordinal, pre_cand) == hipSuccess, "Prepare pre-only");
            auto pre_graph = worker.createGraphCapture(storage.stream);
            {
                ScopedBackendGraphCapture rec(worker, *pre_graph, "Pre-only");
                require(rec.begin(), "Begin pre-only");
                require(probe::launch(pre_cand, stream) == hipSuccess, "Launch pre-only");
                rec.finish();
            }
            require(pre_graph->instantiate(), "Inst pre-only");
            for(int i=0;i<20;++i) require(pre_graph->launch(),"Warmup pre-only");
            require(hipEventRecord(t0,stream)==hipSuccess,"t0");
            for(int i=0;i<500;++i) require(pre_graph->launch(),"Timed pre-only");
            require(hipEventRecord(t1,stream)==hipSuccess&&hipEventSynchronize(t1)==hipSuccess,"t1");
            float ms=0.0f;
            require(hipEventElapsedTime(&ms,t0,t1)==hipSuccess,"elapsed");
            std::cout<<"FULL_GDN_TIMING,nv="<<n_v_heads<<",arm=mega_pre_only,us_per_layer="<<(ms*1000.0f/500.0f)<<'\n';
        }
        // Time staged sub-phases individually
        {
            auto timePhase = [&](const char* name, auto&& fn) {
                auto g = worker.createGraphCapture(storage.stream);
                {
                    ScopedBackendGraphCapture rec(worker, *g, name);
                    require(rec.begin(), "Begin");
                    fn();
                    rec.finish();
                }
                require(g->instantiate(), "Inst");
                for(int i=0;i<20;++i) require(g->launch(),"W");
                require(hipEventRecord(t0,stream)==hipSuccess,"t0");
                for(int i=0;i<500;++i) require(g->launch(),"T");
                require(hipEventRecord(t1,stream)==hipSuccess&&hipEventSynchronize(t1)==hipSuccess,"t1");
                float ms=0.0f;
                require(hipEventElapsedTime(&ms,t0,t1)==hipSuccess,"el");
                std::cout<<"FULL_GDN_PHASE,nv="<<n_v_heads<<",phase="<<name<<",us="<<(ms*1000.0f/500.0f)<<'\n';
            };
            timePhase("staged_hc", [&](){
                primitives->residualWrite({extent,H,residual,previous,write_gate,ptr(0,0)},storage.stream);
                primitives->branchRMSNorm({extent,H,1e-6f,ptr(0,0),gain,ptr(0,1)},storage.stream);
                require(down_stage.execute(context.get())&&up_stage.execute(context.get()),"Staged HC");
                primitives->residualRead({extent,H,ptr(0,1),static_cast<const float*>(up->gpu_data_ptr()),ptr(0,2)},storage.stream);
            });
            timePhase("staged_qkv_z", [&](){
                require(qkv_eng->multiply_fused_tensor(planes[0][2].get(),orig_proj,1,H,nullptr,proj_ws.value.get()),"Staged QKV/Z");
            });
            timePhase("staged_ab", [&](){
                ITensorGemm::VerifierKernelModeScope scope(rows);
                require(half_engines[0]->multiply_fused_tensor(planes[0][2].get(),half_proj,1,H,nullptr,half_ws.value.get()),"Staged A/B");
            });
            timePhase("staged_conv_rec", [&](){
                require(gdn::launchROCmScalarPreparedQKConvolution({{n_k_heads,n_v_heads,D,D,true,
                    ptr(0,4),d_conv_w,nullptr,ptr(0,8),d_prep_qk},conv_states[0],conv_states[0],nullptr,0,0},storage.stream),"Staged conv");
                gdn::MergedRecurrenceInput rec_in{DeviceRequestRowRanges::fullyActiveRequests(1,1),n_k_heads,n_v_heads,D,D,0,true,
                    ptr(0,8),ptr(0,6),ptr(0,7),d_a_log,d_dt_bias,ptr(0,9),rec_states[0],rec_states[0],nullptr,0,0,
                    GDNRecurrenceStateLayout::ValueMajorKeyContiguous};
                require(gdn::launchROCmScalarPreparedQKRecurrence({rec_in,d_prep_qk},storage.stream),"Staged recurrence");
            });
            timePhase("staged_norm_ssm_out", [&](){
                GatedRMSNormProjectionInput gnorm{*planes[0][9],*planes[0][5],*gamma_t,
                    {D,D,1e-6f,true,GatedNormActivation::Sigmoid}};
                require(ssm_eng->multiply_gated_rmsnorm_single_q8(gnorm,planes[0][10].get(),H,Z,ssm_ws.value.get()),"Staged gated norm+ssm_out");
            });
        }
        (void)hipEventDestroy(t0); (void)hipEventDestroy(t1);
        require(backend->synchronizeStream(storage.stream,device.ordinal),"Sync");
        for(auto& g:graphs) g.reset();
    });
}

TEST(Q8ResidualCorrectionDevice, ROCmRetainedFullGDNMegakernelReplay)
{
    auto* backend=getBackendFor(DeviceId::rocm(0));
    require(backend&&backend->deviceCount()>0,"ROCm lane requires at least one visible physical device");
    const auto& metadata=rocmHardwareMetadata();
    for(std::size_t index=0;index<metadata.size();++index) {
        const auto& device=metadata.at(static_cast<int>(index));
        if(device.compute_units!=60||device.native_wave_width!=64||
           (device.isa_target!="gfx906"&&!device.isa_target.starts_with("gfx906:")))continue;
        proveRetainedFullGDNMegakernel(DeviceId::rocm(device.ordinal),4,12);
        proveRetainedFullGDNMegakernel(DeviceId::rocm(device.ordinal),8,24);
        return;
    }
    GTEST_SKIP()<<"The retained full GDN megakernel requires a native 60-CU gfx906 participant";
}
#endif
