/**
 * @file ROCmRoutedVerifierTransaction.h
 * @brief Actual counted router/expert-stage replay with an independent M1 oracle.
 *
 * Included in the expert-pipeline fixture's private namespace after its common
 * source/descriptor helpers. Prepared sources remain owned by runExpertPipeline.
 * Router weights select uniform, hotset, concentrated or mixed-cohort work;
 * the real router owns every live route and inactive sentinel. Dense activation
 * dimensions outside the first Q8 block keep expert arithmetic nontrivial.
 * Each candidate retains its own stages, workspace, publication and graph.
 */

/** @brief Preserve the application's router policy outside this focused fixture. */
class RoutedVerifierRouterPolicy final
{
    bool q8_ = mutableDebugEnv().rocm.moe_router_q8;
    bool reuse_ = mutableDebugEnv().rocm.moe_reuse_router_q8_hidden;
public:
    /** @brief Select the production Q8 publication contract before worker setup. */
    RoutedVerifierRouterPolicy()
    { mutableDebugEnv().rocm.moe_router_q8 = mutableDebugEnv().rocm.moe_reuse_router_q8_hidden = true; }
    /** @brief Restore the original policy after the worker and graphs retire. */
    ~RoutedVerifierRouterPolicy()
    { mutableDebugEnv().rocm.moe_router_q8 = q8_; mutableDebugEnv().rocm.moe_reuse_router_q8_hidden = reuse_; }
    RoutedVerifierRouterPolicy(const RoutedVerifierRouterPolicy &) = delete;
    RoutedVerifierRouterPolicy &operator=(const RoutedVerifierRouterPolicy &) = delete;
};

/** @brief Retain an exact temporary omission until outstanding work has drained. */
class RoutedVerifierNodeOmission final
{
    hipGraphExec_t executable_;
    hipStream_t stream_;
    const std::vector<hipGraphNode_t> &nodes_;
    bool omitted_ = false;
public:
    /** @param executable Borrowed parent. @param stream Exact stream. @param nodes Borrowed node family. */
    RoutedVerifierNodeOmission(hipGraphExec_t executable, hipStream_t stream,
        const std::vector<hipGraphNode_t> &nodes) : executable_(executable), stream_(stream), nodes_(nodes) {}
    /** @brief Restore even after an assertion or partially applied omission. */
    ~RoutedVerifierNodeOmission() { restore(); }
    RoutedVerifierNodeOmission(const RoutedVerifierNodeOmission &) = delete;
    RoutedVerifierNodeOmission &operator=(const RoutedVerifierNodeOmission &) = delete;
    /** @brief Disable all actual parent nodes of the selected family. */
    void omit()
    {
        if (omitted_ || nodes_.empty()) throw std::logic_error("Invalid routed node omission");
        omitted_ = true;
        for (auto node : nodes_)
            if (hipGraphNodeSetEnabled(executable_, node, 0) != hipSuccess) throw std::runtime_error("Routed node omission failed");
    }
    /** @brief Join the original stream before restoring the borrowed executable. */
    void restore() noexcept
    {
        if (!omitted_) return;
        if (hipStreamSynchronize(stream_) != hipSuccess) std::terminate();
        for (auto node : nodes_)
            if (hipGraphNodeSetEnabled(executable_, node, 1) != hipSuccess) std::terminate();
        omitted_ = false;
    }
};

/** @brief Replay supplied duplicate routes through the actual grouping producer.
 * @param worker Exact owner of capture and observation. @param device Original gfx1100 device.
 * @param prepared Original prepared expert sources, shared with the actual-router fixture.
 * @param directory_source Exact offset or column producer to replay.
 * @note This separate grouping oracle does not assume the actual router emits duplicate top-k IDs. */
void runCapturedOffsetSlotGrouping(IWorkerGPUContext &worker, DeviceId device,
    const std::vector<GpuPreparedGemm> &prepared,
    rocm::MoEGroupedDirectorySource directory_source=rocm::MoEGroupedDirectorySource::GroupingOffsetSlots)
{
    constexpr int rows=16, routes=8, experts=256, slots=rows*routes;
    const bool columns=directory_source==rocm::MoEGroupedDirectorySource::GroupingColumnSlots;
    const auto require=[](bool ok,const char *what) { if(!ok) throw std::runtime_error(what); };
    auto *backend=getBackendFor(device);
    const auto stream=static_cast<hipStream_t>(worker.defaultStream());
    auto ids=TestTensorFactory::createFP32({rows,routes});
    auto weights=TestTensorFactory::createFP32({rows,routes});
    auto input=TestTensorFactory::createFP32({rows,2048}),output=TestTensorFactory::createFP32({rows,2048});
    auto serial_input=TestTensorFactory::createFP32({1,2048}),serial_output=TestTensorFactory::createFP32({1,2048});
    auto serial_ids=TestTensorFactory::createFP32({1,routes}),serial_weights=TestTensorFactory::createFP32({1,routes});
    std::fill_n(ids->mutable_data(),slots,-1.f);
    for(int i=0;i<slots;++i) weights->mutable_data()[i]=float(i+1)/256.f;
    ASSERT_TRUE(TransferEngine::instance().uploadFull(ids.get(),device,stream).success);
    ASSERT_TRUE(TransferEngine::instance().uploadFull(weights.get(),device,stream).success);
    for(size_t i=0;i<input->numel();++i) input->mutable_data()[i]=float(int((i*17+11)%61)-30)/256.f;
    for(auto *tensor:{input.get(),output.get(),serial_input.get(),serial_output.get(),serial_ids.get(),serial_weights.get()})
        ASSERT_TRUE(TransferEngine::instance().uploadFull(tensor,device,stream).success);
    auto requirements=MoEWorkspaceBuffers::rocmMoE(rows,2048,512,experts,routes);
    auto workspace=std::make_unique<DeviceWorkspaceManager>(device,requirements.total_bytes_with_alignment());
    ASSERT_TRUE(workspace->allocate(requirements));
    auto kernel=llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
    ASSERT_TRUE(kernel); kernel->setGPUStream(stream);
    auto *consumer=dynamic_cast<IWorkspaceConsumer *>(kernel.get()); ASSERT_NE(consumer,nullptr);
    consumer->bindWorkspace(workspace.get());
    auto serial_requirements=MoEWorkspaceBuffers::rocmMoE(1,2048,512,experts,routes);
    auto serial_workspace=std::make_unique<DeviceWorkspaceManager>(device,serial_requirements.total_bytes_with_alignment());
    ASSERT_TRUE(serial_workspace->allocate(serial_requirements));
    auto serial=llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
    ASSERT_TRUE(serial); serial->setGPUStream(stream);
    auto *serial_consumer=dynamic_cast<IWorkspaceConsumer *>(serial.get()); ASSERT_NE(serial_consumer,nullptr);
    serial_consumer->bindWorkspace(serial_workspace.get());
    std::vector<DeviceNativeQuantisedMatrixDesc> gates(experts),ups(experts),downs(experts);
    for(int e=0;e<experts;++e)
    {
        ASSERT_TRUE(prepared[3*e].kernel->exportNativeQuantisedMatrixDesc(gates[e]));
        ASSERT_TRUE(prepared[3*e+1].kernel->exportNativeQuantisedMatrixDesc(ups[e]));
        ASSERT_TRUE(prepared[3*e+2].kernel->exportNativeQuantisedMatrixDesc(downs[e]));
    }
    const int gu=kernel->uploadGroupedExpertGateUpDescriptorTables(gates.data(),ups.data(),experts,2048,512);
    const int down=kernel->uploadGroupedExpertDownDescriptorTable(downs.data(),experts,2048,512);
    const int serial_gu=serial->uploadGroupedExpertGateUpDescriptorTables(gates.data(),ups.data(),experts,2048,512);
    const int serial_down=serial->uploadGroupedExpertDownDescriptorTable(downs.data(),experts,2048,512);
    ASSERT_GE(gu,0); ASSERT_GE(down,0); ASSERT_GE(serial_gu,0); ASSERT_GE(serial_down,0);
    auto graph=worker.createGraphCapture(stream),full=worker.createGraphCapture(stream);
    std::vector<uint32_t> expected(size_t(rows)*2048),observed(expected.size());
    std::array<int,experts> counts{},offsets{};
    std::array<int,slots> active{},inverse{},tokens{};
    std::array<float,slots> grouped_weights{};
    std::array<uint32_t,slots> directory{};
    const auto retire=[&](void *) { if(hipStreamSynchronize(stream)!=hipSuccess) std::terminate(); };
    std::unique_ptr<void,decltype(retire)> completion(&worker,retire);
    rocm::ScopedMoEGroupedPrefillEngine selected(rocm::MoEGroupedPrefillEngine::WideDot4Wmma,
        rocm::MoEGroupedWmmaRowFloor::Eight,directory_source);
    ASSERT_TRUE(kernel->prepareExpertGroupsAsync(ids.get(),weights.get(),rows,experts,routes));
    ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
    ScopedBackendGraphCapture capture(worker,*graph,"duplicate offset-slot grouping");
    ASSERT_TRUE(capture.begin());
    ASSERT_TRUE(kernel->prepareExpertGroupsAsync(ids.get(),weights.get(),rows,experts,routes));
    capture.finish(); ASSERT_TRUE(graph->instantiate());
    ASSERT_TRUE(kernel->executeGroupedPrefillPipeline(input.get(),output.get(),gu,down,rows,2048,512,experts,routes));
    ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
    ScopedBackendGraphCapture full_capture(worker,*full,"duplicate grouped projection");
    ASSERT_TRUE(full_capture.begin());
    ASSERT_TRUE(kernel->prepareExpertGroupsAsync(ids.get(),weights.get(),rows,experts,routes));
    ASSERT_TRUE(kernel->executeGroupedPrefillPipeline(input.get(),output.get(),gu,down,rows,2048,512,experts,routes));
    full_capture.finish(); ASSERT_TRUE(full->instantiate());
    auto *native=dynamic_cast<HIPGraphCapture *>(graph.get()); ASSERT_NE(native,nullptr);
    size_t node_count=0; ASSERT_EQ(hipGraphGetNodes(native->graph(),nullptr,&node_count),hipSuccess);
    std::vector<hipGraphNode_t> nodes(node_count), publishers;
    ASSERT_EQ(hipGraphGetNodes(native->graph(),nodes.data(),&node_count),hipSuccess);
    for(auto node:nodes)
    {
        hipGraphNodeType type; ASSERT_EQ(hipGraphNodeGetType(node,&type),hipSuccess);
        if(type!=hipGraphNodeTypeKernel) continue;
        hipKernelNodeParams launch{}; ASSERT_EQ(hipGraphKernelNodeGetParams(node,&launch),hipSuccess);
        const char *name=hipKernelNameRefByPtr(launch.func,stream); ASSERT_NE(name,nullptr);
        if(std::string_view(name).find("rocm_moe_build_active_expert_list_kernel")!=std::string_view::npos)
            publishers.push_back(node);
    }
    ASSERT_EQ(publishers.size(),1u);
    const auto read=[&](auto &destination,const char *bank) {
        require(backend->deviceToHost(destination.data(),workspace->getBuffer(bank),sizeof(destination),device.ordinal,stream),"grouping observation failed");
    };
    const auto replay=[&] {
        require(hipMemsetAsync(workspace->getBuffer(MoEWorkspaceBuffers::ROCM_PREFILL_WORK_DIRECTORY),0xa5,slots*sizeof(uint32_t),stream)==hipSuccess,"directory poison failed");
        require(graph->launch(),"duplicate grouping replay failed");
        read(counts,MoEWorkspaceBuffers::GROUP_COUNTS); read(offsets,MoEWorkspaceBuffers::GROUP_OFFSETS);
        read(active,MoEWorkspaceBuffers::GROUP_ACTIVE_EXPERT_IDS); read(inverse,MoEWorkspaceBuffers::GROUP_ORIGINAL_TO_GROUPED);
        read(tokens,MoEWorkspaceBuffers::GROUP_TOKEN_INDICES); read(grouped_weights,MoEWorkspaceBuffers::GROUP_WEIGHTS);
        read(directory,MoEWorkspaceBuffers::ROCM_PREFILL_WORK_DIRECTORY);
        require(hipStreamSynchronize(stream)==hipSuccess,"duplicate grouping observation join failed");
    };
    for(int pattern:{128,17,33,5,7,0,-1,0,-1,128})
    {
        std::fill_n(ids->mutable_data(),slots,-1.f);
        std::array<int,experts> expected_counts{};
        if(pattern>=0) { expected_counts[255]=pattern; std::fill_n(ids->mutable_data(),pattern,255.f); }
        else
        {
            int original=0,expert=1;
            for(int count:{17,33,1,7,8,16,31,15})
            {
                expected_counts[expert]=count;
                for(int i=0;i<count;++i) ids->mutable_data()[original++]=float(expert);
                expert+=31;
            }
        }
        ASSERT_TRUE(backend->hostToDevice(ids->gpu_data_ptr(),ids->data(),slots*sizeof(float),device.ordinal,stream));
        replay(); ASSERT_EQ(counts,expected_counts);
        std::array<uint32_t,slots> expected_directory; expected_directory.fill(UINT32_MAX);
        int offset=0,active_count=0;
        for(int expert=0;expert<experts;++expert)
        {
            ASSERT_EQ(offsets[expert],offset);
            const int count=expected_counts[expert];
            if(count)
            {
                ASSERT_EQ(active[active_count++],expert);
                if(columns) {
                    // Independent writer oracle derives compact ordinals from
                    // actual row/column work, not the production interval helper.
                    int ordinal=0;
                    for(int row=0;row<count;row+=count<8 ? 4 : 16)
                        for(int quadrant=0;quadrant<(count<8 ? 1 : 4);++quadrant)
                            expected_directory[offset+ordinal++]=(uint32_t(row)<<10)|(uint32_t(quadrant)<<8)|uint32_t(expert);
                } else for(int row=0;row<count;row+=count<8?count:16)
                    expected_directory[offset+row]=(uint32_t(row)<<10)|uint32_t(expert);
            }
            offset+=count;
        }
        for(int i=active_count;i<slots;++i) ASSERT_EQ(active[i],-1);
        ASSERT_EQ(directory,expected_directory);
        std::array<bool,slots> written{};
        for(int original=0;original<slots;++original)
        {
            if(ids->data()[original]<0) { ASSERT_EQ(inverse[original],-1); continue; }
            const int grouped=inverse[original],expert=int(ids->data()[original]);
            ASSERT_GE(grouped,offsets[expert]); ASSERT_LT(grouped,offsets[expert]+counts[expert]);
            ASSERT_FALSE(written[grouped]); written[grouped]=true;
            ASSERT_EQ(tokens[grouped],original/routes);
            ASSERT_EQ(std::bit_cast<uint32_t>(grouped_weights[grouped]),std::bit_cast<uint32_t>(weights->data()[original]));
        }
        if(pattern!=0)
        {
            RoutedVerifierNodeOmission omission(native->executable(),stream,publishers);
            omission.omit(); replay();
            ASSERT_TRUE(std::all_of(directory.begin(),directory.end(),[](uint32_t word){return word==0xa5a5a5a5u;}));
            omission.restore(); replay(); ASSERT_EQ(directory,expected_directory);
        }
        // The independent M1 producer consumes the supplied original routes,
        // including duplicates. It never reads the candidate's grouped maps.
        for(int row=0;row<rows;++row)
        {
            ASSERT_TRUE(backend->hostToDevice(serial_input->gpu_data_ptr(),input->data()+size_t(row)*2048,2048*sizeof(float),device.ordinal,stream));
            ASSERT_TRUE(backend->hostToDevice(serial_ids->gpu_data_ptr(),ids->data()+row*routes,routes*sizeof(float),device.ordinal,stream));
            ASSERT_TRUE(backend->hostToDevice(serial_weights->gpu_data_ptr(),weights->data()+row*routes,routes*sizeof(float),device.ordinal,stream));
            serial->zeroBuffer(serial_output.get(),2048*sizeof(float));
            ASSERT_TRUE(serial->groupedExpertDecodeFromRouting(serial_input.get(),serial_ids.get(),serial_weights.get(),
                serial_gu,serial_down,routes,serial_output.get(),2048,512));
            ASSERT_TRUE(backend->deviceToHost(expected.data()+size_t(row)*2048,serial_output->gpu_data_ptr(),2048*sizeof(uint32_t),device.ordinal,stream));
        }
        ASSERT_EQ(hipMemsetAsync(output->gpu_data_ptr(),0xa5,observed.size()*sizeof(uint32_t),stream),hipSuccess);
        ASSERT_EQ(hipMemsetAsync(workspace->getBuffer(MoEWorkspaceBuffers::ROCM_PREFILL_WORK_DIRECTORY),0xff,slots*sizeof(uint32_t),stream),hipSuccess);
        ASSERT_TRUE(full->launch());
        ASSERT_TRUE(backend->deviceToHost(observed.data(),output->gpu_data_ptr(),observed.size()*sizeof(uint32_t),device.ordinal,stream));
        ASSERT_EQ(hipStreamSynchronize(stream),hipSuccess);
        ASSERT_TRUE(std::all_of(expected.begin(),expected.end(),[](uint32_t word){return std::isfinite(std::bit_cast<float>(word));}));
        ASSERT_EQ(observed,expected) << "duplicate count pattern=" << pattern;
    }
}

/** @brief Replay the complete actual router-to-expert transaction on its worker.
 * @param c Original geometry, sources and declared load distribution.
 * @param mode Replay or paired timing; both use the same retained arithmetic.
 * @param prepared Existing independent expert triplets from the common fixture.
 * @param worker Exact device worker owning setup, capture and retirement.
 */
void runCapturedRoutedVerifier(const Case &c, ExpertBenchmark mode,
    const std::vector<GpuPreparedGemm> &prepared, IWorkerGPUContext &worker)
{
    constexpr int hidden = 2048, intermediate = 512, experts = 256, top_k = 8, repeats = 8;
    using Engine = rocm::MoEGroupedPrefillEngine;
    using Floor = rocm::MoEGroupedWmmaRowFloor;
    std::array<GroupedDotCandidate, 5> requested_candidates{{
        {Engine::OrderedDot4, Floor::EveryExpert, "Dot4"},
        {Engine::WideDot4Wmma, Floor::Eight, "WideFloor8"},
        {Engine::WideDot4Wmma, Floor::Eight, "WideGroupingOffsetSlots", rocm::MoEGroupedDirectorySource::GroupingOffsetSlots},
        {Engine::WideDot4Wmma, Floor::Eight, "ColumnGroupingSlots", rocm::MoEGroupedDirectorySource::GroupingColumnSlots},
        {Engine::Automatic, Floor::EveryExpert, "Automatic"}}};
    const int automatic_path = c.rows == 16 ? 4 : c.rows > 8 ? 3 : 2;
    requested_candidates[automatic_path] = {Engine::Automatic, Floor::EveryExpert, "Automatic"};
    // Expected production policy is independent of the selector under test.
    // The last owner records Automatic, including Dot4 controls at M8/M9.
    auto candidates = requested_candidates;
    candidates[automatic_path] = c.rows == 16
        ? GroupedDotCandidate{Engine::WideDot4Wmma, Floor::Eight, "Automatic", rocm::MoEGroupedDirectorySource::GroupingColumnSlots}
        : GroupedDotCandidate{Engine::OrderedDot4, Floor::EveryExpert, "Automatic"};
    constexpr int max_paths = std::tuple_size_v<decltype(candidates)>;
    const int paths = automatic_path + 1;
    std::printf("ROCM_ROUTED_VERIFIER_CANDIDATES,%s",c.name.c_str());
    for(int path=0;path<paths;++path) std::printf(",%s",candidates[path].name.c_str());
    std::printf("\n");
    ASSERT_EQ(c.hidden, hidden); ASSERT_EQ(c.intermediate, intermediate); ASSERT_EQ(c.experts, experts);
    ASSERT_EQ(prepared.size(), experts * 3u); ASSERT_GE(c.rows, 8); ASSERT_LE(c.rows, 16);
    ASSERT_TRUE(c.verifier_distribution);
    const int rows = c.rows, ordinal = c.device.ordinal;
    if(rows==16 && *c.verifier_distribution==RoutedVerifierDistribution::Uniform) {
        ASSERT_NO_FATAL_FAILURE(runCapturedOffsetSlotGrouping(worker,c.device,prepared));
        ASSERT_NO_FATAL_FAILURE(runCapturedOffsetSlotGrouping(worker,c.device,prepared,
            rocm::MoEGroupedDirectorySource::GroupingColumnSlots));
    }
    auto *backend = getBackendFor(c.device);
    const auto stream = static_cast<hipStream_t>(worker.defaultStream());
    ASSERT_NE(backend, nullptr); ASSERT_NE(stream, nullptr);
    const auto require = [](bool ok, const char *what) { if (!ok) throw std::runtime_error(what); };
    auto tensor = [&](int m, int n) { return TestTensorFactory::createFP32({size_t(m), size_t(n)}); };
    auto input = tensor(rows, hidden), gate = tensor(experts, hidden);
    auto desired_ids = native_quantised_dispatch::makeMoERoutingIndices(
        *c.verifier_distribution == RoutedVerifierDistribution::Hotset32 ? MoERoutingProfile::Hotset : MoERoutingProfile::Uniform,
        rows, top_k, experts);
    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < hidden; ++col)
            input->mutable_data()[size_t(row) * hidden + col] = col < 32 ? float(col == row) :
                float((row * 17 + col * 13) % 61 - 30) / 256.0f;
        for (int slot = 0; slot < top_k; ++slot)
        {
            int expert = row * top_k + slot;
            switch (*c.verifier_distribution)
            {
            case RoutedVerifierDistribution::Uniform: break;
            case RoutedVerifierDistribution::Hotset32: expert = int(desired_ids[size_t(row) * top_k + slot]); break;
            case RoutedVerifierDistribution::Concentrated8: expert = slot; break;
            case RoutedVerifierDistribution::MixedCohorts:
                // Eight complete hot rows share eight experts; later rows own
                // disjoint cold experts, allowing independent survivor checks.
                expert = row < 8 ? slot : (row - 7) * top_k + slot;
                break;
            case RoutedVerifierDistribution::ColdThenHot:
                expert = row < 8 ? (row + 1) * top_k + slot : slot;
                break;
            }
            desired_ids[size_t(row) * top_k + slot] = float(expert);
            gate->mutable_data()[size_t(expert) * hidden + row] = float(top_k - slot);
        }
    }
    auto live = TestTensorFactory::createINT32({1}); live->mutable_int32_data()[0] = rows;
    auto serial_input = tensor(1, hidden), serial_output = tensor(1, hidden);
    auto serial_ids = tensor(1, top_k), serial_weights = tensor(1, top_k);
    std::array<std::unique_ptr<FP32Tensor>, max_paths> outputs, ids, weights;
    std::array<std::shared_ptr<MoERoutedPipelineKernelOwner>, max_paths> owners;
    std::array<std::unique_ptr<MoEExpertComputeStage>, max_paths> stages;
    std::array<std::unique_ptr<DeviceWorkspaceManager>, max_paths> workspaces;
    std::array<std::unique_ptr<IGPUGraphCapture>, max_paths> fragments, graphs;
    std::array<std::unique_ptr<RouterQ8CaptureProof>, max_paths> q8_proofs;
    std::array<std::vector<hipGraphNode_t>, max_paths> native_nodes, packed_publishers;
    std::array<std::optional<rocm::ROCmMoEGroupedRouteAdmission>, max_paths> native_admission;
    std::vector<uint32_t> expected(size_t(rows) * hidden), observed(expected.size());
    std::vector<uint8_t> private_q8;
    std::vector<float> expected_ids(size_t(rows) * top_k), expected_weights(expected_ids.size());
    std::vector<float> observed_ids(expected_ids.size()), observed_weights(expected_ids.size());
    int published_live = rows;
    std::array<int, 2> reset_values{std::numeric_limits<int>::max(), 0};
    auto serial = llaminar::v2::kernels::KernelFactory::createMoEKernel(c.device);
    ASSERT_NE(serial, nullptr); serial->setGPUStream(stream);
    auto serial_requirements = MoEWorkspaceBuffers::rocmMoE(1, hidden, intermediate, experts, top_k);
    auto serial_workspace = std::make_unique<DeviceWorkspaceManager>(c.device, serial_requirements.total_bytes_with_alignment());
    ASSERT_TRUE(serial_workspace->allocate(serial_requirements));
    auto *serial_consumer = dynamic_cast<IWorkspaceConsumer *>(serial.get()); ASSERT_NE(serial_consumer, nullptr);
    serial_consumer->bindWorkspace(serial_workspace.get());
    DeviceMoERuntimeTable::Config runtime_config;
    runtime_config.device_id = c.device; runtime_config.num_layers = 1; runtime_config.num_experts = experts;
    runtime_config.top_k = top_k; runtime_config.mirror_to_device = true; runtime_config.prefill_token_capacity = 1;
    DeviceMoERuntimeTable serial_runtime(runtime_config);
    // All asynchronous staging destinations and sources precede this retirement
    // guard. Prepared payloads belong to the caller and outlive this function.
    const auto retire = [&](void *) { if (hipStreamSynchronize(stream) != hipSuccess) std::terminate(); };
    std::unique_ptr<void, decltype(retire)> completion(&worker, retire);
    auto &transfer = TransferEngine::instance();
    const auto upload = [&](TensorBase *t) { require(transfer.uploadFull(t, c.device, stream).success, "Routed fixture upload failed"); };
    for (TensorBase *t : std::array<TensorBase *, 7>{input.get(), gate.get(), live.get(), serial_input.get(), serial_output.get(), serial_ids.get(), serial_weights.get()}) upload(t);
    const auto *count = static_cast<const int *>(live->gpu_data_ptr());
    std::vector<DeviceNativeQuantisedMatrixDesc> gd(experts), ud(experts), dd(experts);
    auto &runtime = serial_runtime.hostLayerState(0);
    runtime.active_bank = 0; runtime.active_epoch = 1; runtime.expert_count = experts; runtime.top_k = top_k;
    for (int e = 0; e < experts; ++e)
    {
        ASSERT_TRUE(prepared[3 * e].kernel->exportNativeQuantisedMatrixDesc(gd[e]));
        ASSERT_TRUE(prepared[3 * e + 1].kernel->exportNativeQuantisedMatrixDesc(ud[e]));
        ASSERT_TRUE(prepared[3 * e + 2].kernel->exportNativeQuantisedMatrixDesc(dd[e]));
        for (auto &bank : runtime.banks)
        {
            bank.epoch = 1; bank.expert_count = experts; bank.local_compute_mask[e] = 1;
            bank.replica_role[e] = uint8_t(DeviceMoEReplicaRole::Primary); bank.resident_participant_mask[e] = 1u;
            auto &d = bank.experts[e]; d.logical_expert_id = e; d.owner_participant = 0; d.local_slot = e;
            d.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid | DeviceMoEExpertFlags::Resident | DeviceMoEExpertFlags::LocalCompute);
            d.gate = gd[e]; d.up = ud[e]; d.down = dd[e];
        }
    }
    // Diagnostic setup publishes the managed runtime's original descriptors
    // once. Its mutable route state is subsequently written only by M1 routing.
    for(const auto &role:std::array<std::pair<const char *,const std::vector<DeviceNativeQuantisedMatrixDesc> *>,3>{{
        {"gate",&gd},{"up",&ud},{"down",&dd}}})
    {
        const auto &first=role.second->front();
        ASSERT_TRUE(first.source_identity_present);
        for(const auto &descriptor:*role.second)
        {
            ASSERT_EQ(descriptor.n,first.n); ASSERT_EQ(descriptor.k,first.k);
            ASSERT_EQ(descriptor.source_codebook_id,first.source_codebook_id);
            ASSERT_EQ(descriptor.source_is_superblock,first.source_is_superblock);
            ASSERT_EQ(descriptor.source_identity_present,first.source_identity_present);
            ASSERT_EQ(descriptor.codebook_id,first.codebook_id);
        }
        std::printf("ROCM_ROUTED_VERIFIER_SOURCE,%s,%s,%u,%u,%u,%u,%u,%u,%d\n",c.name.c_str(),role.first,
            first.n,first.k,unsigned(first.source_codebook_id),unsigned(first.source_is_superblock),
            unsigned(first.codebook_id),unsigned(first.source_identity_present),experts);
    }
    ASSERT_EQ(hipMemcpyAsync(serial_runtime.deviceLayerState(0), &runtime, sizeof(runtime), hipMemcpyHostToDevice, stream), hipSuccess);
    const int gu_table = serial->uploadGroupedExpertGateUpDescriptorTables(gd.data(), ud.data(), experts, hidden, intermediate);
    const int down_table = serial->uploadGroupedExpertDownDescriptorTable(dd.data(), experts, hidden, intermediate);
    ASSERT_GE(gu_table, 0); ASSERT_GE(down_table, 0);
    ASSERT_TRUE(serial->prepareRouteLaunch(gate.get(), {.kind=MoERouteLaunchKind::RuntimeDecode,
        .physical_rows=1, .d_model=hidden, .num_experts=experts, .top_k=top_k}));
    for (int row = 0; row < rows; ++row)
    {
        ASSERT_TRUE(backend->hostToDevice(serial_input->gpu_data_ptr(), input->data() + size_t(row) * hidden,
            hidden * sizeof(float), ordinal, stream));
        ASSERT_TRUE(serial->decodeRouteSelect(serial_runtime.deviceLayerState(0), serial_input.get(), gate.get(), hidden,
            experts, top_k, true, serial_ids.get(), serial_weights.get(), true, false, nullptr,
            RoutedExpertRowExecutionPolicy::ParticipantAssigned));
        serial->zeroBuffer(serial_output.get(), hidden * sizeof(float));
        ASSERT_TRUE(serial->groupedExpertDecodeFromRuntime(serial_runtime.deviceLayerState(0), serial_input.get(), gu_table,
            down_table, top_k, serial_output.get(), hidden, intermediate, MoEDecodeDescriptorSource::RuntimePlacementTable));
        ASSERT_TRUE(backend->deviceToHost(expected.data() + size_t(row) * hidden, serial_output->gpu_data_ptr(), hidden * sizeof(float), ordinal, stream));
        ASSERT_TRUE(backend->deviceToHost(expected_ids.data() + size_t(row) * top_k, serial_ids->gpu_data_ptr(), top_k * sizeof(float), ordinal, stream));
        ASSERT_TRUE(backend->deviceToHost(expected_weights.data() + size_t(row) * top_k, serial_weights->gpu_data_ptr(), top_k * sizeof(float), ordinal, stream));
    }
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    for (int row = 0; row < rows; ++row)
        for (int slot = 0; slot < top_k; ++slot)
            ASSERT_EQ(expected_ids[size_t(row) * top_k + slot], desired_ids[size_t(row) * top_k + slot]);
    ASSERT_TRUE(std::all_of(expected.begin(), expected.end(), [](auto word) { return std::isfinite(std::bit_cast<float>(word)); }));
    ASSERT_TRUE(std::any_of(expected.begin(), expected.end(), [](auto word) { return word != 0; }));
    llaminar2::testing::MockDeviceContext context(c.device, ComputeBackendType::GPU_ROCM);
    const auto engine_index = [](Engine engine) {
        switch (engine)
        {
        case Engine::OrderedDot4: return 0;
        case Engine::WmmaM16N32: return 1;
        case Engine::WmmaM16N64: return 2;
        case Engine::WideDot4Wmma: return 3;
        default: throw std::logic_error("Unknown routed candidate engine");
        }
    };
    const auto enqueue = [&](int path) {
        return owners[path]->kernel->routeVerifierRowsDecodeEquivalent(input.get(), gate.get(), rows, hidden, experts,
            top_k, true, ids[path].get(), weights[path].get(), count) && stages[path]->execute(&context);
    };
    const auto submissions = [&] {
        std::array<double, 5> counts{};
        for (const auto &record : PerfStatsCollector::snapshot({
                "kernel.rocm_moe_grouped_prefill_batch_invariant_calls",
                "kernel.rocm_moe_grouped_prefill_router_q8_reuse_calls"}))
        {
            if (record.tags.at("seq_len") != std::to_string(rows) || record.tags.at("top_k") != "8") continue;
            if (record.name == "rocm_moe_grouped_prefill_router_q8_reuse_calls") { counts[4] += record.value; continue; }
            // The separate duplicate-route oracle uses internal quantization;
            // its successful submissions are outside this router-owned census.
            if (record.tags.at("hidden_preparation") != "router_q8") continue;
            require(record.tags.at("counter_semantics") == "host_submissions" &&
                record.tags.at("hidden_preparation") == "router_q8" && record.tags.at("geometry_role") == "physical_capacity",
                "Routed submission counter contract changed");
            const auto &policy = record.tags.at("dot_policy");
            require(policy == "dot4" || policy == "wmma_n32" || policy == "wmma_n64" || policy == "wide_dot4_wmma", "Unknown routed arithmetic producer");
            counts[policy == "dot4" ? 0 : policy == "wmma_n32" ? 1 : policy == "wmma_n64" ? 2 : 3] += record.value;
        }
        return counts;
    };
    for (int path = 0; path < paths; ++path)
    {
        outputs[path] = tensor(rows, hidden); ids[path] = tensor(rows, top_k); weights[path] = tensor(rows, top_k);
        for (auto *t : {outputs[path].get(), ids[path].get(), weights[path].get()}) upload(t);
        owners[path] = std::make_shared<MoERoutedPipelineKernelOwner>();
        owners[path]->kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(c.device);
        ASSERT_TRUE(owners[path]->kernel); owners[path]->kernel->setGPUStream(stream);
        MoEExpertComputeStage::Params params;
        params.device_id = c.device; params.input = input.get(); params.output = outputs[path].get();
        params.seq_len = rows; params.d_model = hidden; params.expert_intermediate = intermediate;
        params.num_experts = experts; params.top_k = top_k; params.routed_pipeline_kernel_owner = owners[path];
        params.routing_indices = ids[path].get(); params.routing_weights = weights[path].get();
        params.force_decode_equivalent_verifier_prefill = true; params.active_row_count_device = count;
        for (int e = 0; e < experts; ++e)
        {
            params.prepared_gate_gemm.push_back(prepared[3 * e].kernel);
            params.prepared_up_gemm.push_back(prepared[3 * e + 1].kernel);
            params.prepared_down_gemm.push_back(prepared[3 * e + 2].kernel);
        }
        stages[path] = std::make_unique<MoEExpertComputeStage>(params); stages[path]->setGPUStream(stream);
        auto requirements = stages[path]->getWorkspaceRequirements(rows, hidden, intermediate);
        requirements.merge(MoEWorkspaceBuffers::rocmMoE(rows, hidden, intermediate, experts, top_k));
        workspaces[path] = std::make_unique<DeviceWorkspaceManager>(c.device, requirements.total_bytes_with_alignment());
        ASSERT_TRUE(workspaces[path]->allocate(requirements)); stages[path]->bindWorkspace(workspaces[path].get());
        auto *consumer = dynamic_cast<IWorkspaceConsumer *>(owners[path]->kernel.get()); ASSERT_NE(consumer, nullptr);
        consumer->bindWorkspace(workspaces[path].get());
        ASSERT_TRUE(owners[path]->kernel->bindRouterQ8HiddenPublication(owners[path]->router_q8_publication, MoERouterQ8PublicationAccess::ProducerAndConsumer));
        ASSERT_TRUE(owners[path]->kernel->prepareRouteLaunch(gate.get(), {.kind=MoERouteLaunchKind::DecodeEquivalentVerifier,
            .physical_rows=rows, .d_model=hidden, .num_experts=experts, .top_k=top_k}));
        ASSERT_TRUE(stages[path]->prepareGraphLaunch(&context, stream));
        auto expected_submissions = submissions();
        const auto &candidate = candidates[path];
        const auto &requested = requested_candidates[path];
        rocm::ScopedMoEGroupedPrefillEngine selected(requested.engine, requested.floor, requested.directory);
        ASSERT_TRUE(enqueue(path)); ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        for (auto *t : {input.get(), gate.get(), outputs[path].get(), ids[path].get(), weights[path].get()})
            TransferEngine::requireDeviceInput(t, c.device, stream);
        graphs[path] = worker.createGraphCapture(stream); fragments[path] = graphs[path]->createOrderedTimelineFragment();
        ASSERT_TRUE(fragments[path]); ScopedBackendGraphCapture capture(worker, *fragments[path], "counted routed verifier transaction");
        ASSERT_TRUE(capture.begin());
        for (int repeat = 0; repeat < repeats; ++repeat) ASSERT_TRUE(enqueue(path));
        capture.finish();
        const std::array<GPUOrderedTimelineStep, 1> steps{{{.name=candidates[path].name.c_str(), .kind=GPUOrderedTimelineStepKind::CapturedFragment, .capture=fragments[path].get()}}};
        ASSERT_TRUE(graphs[path]->buildOrderedTimelineTransaction(steps, mode == ExpertBenchmark::RoutedVerifierProfile
            ? GPUOrderedTimelineInstrumentation::Disabled : GPUOrderedTimelineInstrumentation::PerStepEvents));
        ASSERT_TRUE(graphs[path]->instantiate());
        auto *parent = dynamic_cast<HIPGraphCapture *>(graphs[path].get()); ASSERT_NE(parent, nullptr);
        q8_proofs[path] = std::make_unique<RouterQ8CaptureProof>(*parent, stream, *owners[path]->router_q8_publication, count,
            static_cast<const int *>(workspaces[path]->getBuffer(MoEWorkspaceBuffers::GROUP_TOKEN_INDICES)),
            rows, hidden, intermediate, repeats, candidate.engine != Engine::OrderedDot4 && rows > 8 ? 1 : 0);
        size_t node_count = 0; ASSERT_EQ(hipGraphGetNodes(parent->graph(), nullptr, &node_count), hipSuccess);
        std::vector<hipGraphNode_t> nodes(node_count); ASSERT_EQ(hipGraphGetNodes(parent->graph(), nodes.data(), &node_count), hipSuccess);
        std::array<int, 2> native_roles{};
        std::array<int, 2> wide_directories{};
        int route_owned_gateup = 0;
        for (auto node : nodes)
        {
            hipGraphNodeType type; ASSERT_EQ(hipGraphNodeGetType(node, &type), hipSuccess);
            if (type != hipGraphNodeTypeKernel) continue;
            hipKernelNodeParams launch{}; ASSERT_EQ(hipGraphKernelNodeGetParams(node, &launch), hipSuccess);
            const char *symbol = hipKernelNameRefByPtr(launch.func, stream); ASSERT_NE(symbol, nullptr);
            if (candidate.directory != rocm::MoEGroupedDirectorySource::RolePlanner &&
                std::string_view(symbol).find("rocm_moe_build_active_expert_list_kernel") != std::string_view::npos)
            {
                ASSERT_NE(std::string_view(symbol).find(candidate.directory==rocm::MoEGroupedDirectorySource::GroupingColumnSlots
                    ? "ILb1ELb1EE" : "ILb1ELb0EE"),std::string_view::npos);
                ASSERT_EQ(*static_cast<const int **>(launch.kernelParams[0]),workspaces[path]->getBuffer(MoEWorkspaceBuffers::GROUP_COUNTS));
                ASSERT_EQ(*static_cast<int **>(launch.kernelParams[1]),workspaces[path]->getBuffer(MoEWorkspaceBuffers::GROUP_ACTIVE_EXPERT_IDS));
                ASSERT_EQ(*static_cast<int *>(launch.kernelParams[2]),experts);
                ASSERT_EQ(*static_cast<int *>(launch.kernelParams[3]),rows*top_k);
                ASSERT_EQ(*static_cast<const int **>(launch.kernelParams[4]),workspaces[path]->getBuffer(MoEWorkspaceBuffers::GROUP_OFFSETS));
                ASSERT_EQ(*static_cast<uint32_t **>(launch.kernelParams[5]),workspaces[path]->getBuffer(MoEWorkspaceBuffers::ROCM_PREFILL_WORK_DIRECTORY));
                ASSERT_EQ(*static_cast<int *>(launch.kernelParams[6]),rows*top_k);
                ASSERT_EQ(launch.gridDim.x,1u); ASSERT_EQ(launch.blockDim.x,256u);
                packed_publishers[path].push_back(node);
            }
            if (candidate.engine == Engine::WideDot4Wmma)
            {
                // One adaptive leaf owns each tiled role. Retaining the old
                // separate tiled family would restore the rejected launch cost.
                ASSERT_EQ(std::string_view(symbol).find("moe_grouped_gate_up_batch_invariant_tiled_prefill_kernel"), std::string_view::npos);
                ASSERT_EQ(std::string_view(symbol).find("moe_grouped_down_batch_invariant_tiled_prefill_kernel"), std::string_view::npos);
                if (std::string_view(symbol).find("moe_grouped_build_prefill_tile_directory_kernel") != std::string_view::npos)
                {
                    ASSERT_EQ(*static_cast<const int **>(launch.kernelParams[0]), workspaces[path]->getBuffer(MoEWorkspaceBuffers::GROUP_COUNTS));
                    ASSERT_EQ(*static_cast<uint32_t **>(launch.kernelParams[1]), workspaces[path]->getBuffer(MoEWorkspaceBuffers::ROCM_PREFILL_WORK_DIRECTORY));
                    const auto plan = *static_cast<const rocm::MoEGroupedRowPlan *>(launch.kernelParams[5]);
                    const auto role = plan.role();
                    ASSERT_EQ(plan, rocm::MoEGroupedRowPlan::wide(role, candidate.floor));
                    ++wide_directories[role == rocm::MoEPrefillProjectionRole::GateUp ? 0 : 1];
                }
            }
            if (std::string_view(symbol).find("moe_grouped_gate_up_decode_equivalent_prefill_kernel") != std::string_view::npos)
            {
                ++route_owned_gateup;
                ASSERT_EQ(*static_cast<const int8_t **>(launch.kernelParams[0]), owners[path]->router_q8_publication->quantized_rows);
                ASSERT_EQ(*static_cast<const float **>(launch.kernelParams[1]), owners[path]->router_q8_publication->row_scales);
                ASSERT_EQ(*static_cast<const int **>(launch.kernelParams[2]), workspaces[path]->getBuffer(MoEWorkspaceBuffers::GROUP_TOKEN_INDICES));
                ASSERT_EQ(*static_cast<int *>(launch.kernelParams[3]), 0);
                ASSERT_EQ(*static_cast<float **>(launch.kernelParams[6]), workspaces[path]->getBuffer(MoEWorkspaceBuffers::PREFILL_GATE));
                ASSERT_EQ(*static_cast<float **>(launch.kernelParams[7]), workspaces[path]->getBuffer(MoEWorkspaceBuffers::PREFILL_UP));
                ASSERT_EQ(*static_cast<int *>(launch.kernelParams[9]), rows);
                ASSERT_EQ(*static_cast<int *>(launch.kernelParams[10]), top_k);
                ASSERT_EQ(*static_cast<int *>(launch.kernelParams[11]), 1);
                ASSERT_EQ(*static_cast<int *>(launch.kernelParams[12]), intermediate);
                ASSERT_EQ(*static_cast<int *>(launch.kernelParams[13]), hidden);
            }
            if (std::string_view(symbol).find("grouped_wmma_detail") == std::string_view::npos) continue;
            native_nodes[path].push_back(node);
            const auto &args = *static_cast<const rocm::MoEGroupedWmmaArguments *>(launch.kernelParams[0]);
            ASSERT_GT(path, 0);
            const bool wide = candidate.engine == Engine::WideDot4Wmma;
            const bool columns=candidate.directory==rocm::MoEGroupedDirectorySource::GroupingColumnSlots;
            const auto role = launch.gridDim.z == 2 ? rocm::MoEPrefillProjectionRole::GateUp : rocm::MoEPrefillProjectionRole::Down;
            const int tile = wide ? (launch.gridDim.z == 2 ? 64 : 32) : candidate.engine == Engine::WmmaM16N32 ? 32 : 64;
            ASSERT_EQ(args.directory_layout,columns ? MoEGroupingPackedLayout::ColumnFloor8 : MoEGroupingPackedLayout::WideFloor8);
            ASSERT_EQ(args.dot_admission.rowPlan(), columns ? rocm::MoEGroupedRowPlan::column(role) : wide ? rocm::MoEGroupedRowPlan::wide(role, candidate.floor) :
                rocm::MoEGroupedRowPlan::nativeMicro(role, candidate.floor));
            ASSERT_TRUE(args.dot_admission.binds(role, args.n, args.k));
            ASSERT_NE(std::string_view(symbol).find(wide ? "wideProjection" : "ELi" + std::to_string(tile) + "ELb"), std::string_view::npos);
            ASSERT_EQ(launch.blockDim.x, unsigned(tile * 4));
            ASSERT_EQ(launch.blockDim.y, 1u); ASSERT_EQ(launch.blockDim.z, 1u);
            ASSERT_EQ(args.rows, rows); ASSERT_EQ(args.experts, experts); ASSERT_EQ(args.top_k, top_k);
            ASSERT_EQ(args.device, ordinal); ASSERT_NE(args.descriptors, nullptr);
            ASSERT_EQ(launch.gridDim.y, unsigned(args.directory_capacity));
            ASSERT_EQ(args.branch_suppression, rocm::MoEGroupedBranchSuppression::None);
            ASSERT_EQ(args.group_counts, workspaces[path]->getBuffer(MoEWorkspaceBuffers::GROUP_COUNTS));
            ASSERT_EQ(args.group_offsets, workspaces[path]->getBuffer(MoEWorkspaceBuffers::GROUP_OFFSETS));
            ASSERT_EQ(args.directory,workspaces[path]->getBuffer(MoEWorkspaceBuffers::ROCM_PREFILL_WORK_DIRECTORY));
            if (candidate.directory != rocm::MoEGroupedDirectorySource::RolePlanner)
                ASSERT_EQ(args.directory_capacity,rows*top_k);
            if (launch.gridDim.z == 2)
            {
                ++native_roles[0];
                ASSERT_EQ(args.group_token_indices, workspaces[path]->getBuffer(MoEWorkspaceBuffers::GROUP_TOKEN_INDICES));
                ASSERT_EQ(launch.gridDim.x, unsigned(intermediate / (columns ? 256 : tile)));
                ASSERT_NE(std::string_view(symbol).find("Format13E"), std::string_view::npos);
                ASSERT_EQ(args.n, intermediate); ASSERT_EQ(args.k, hidden);
                ASSERT_EQ(args.input, rocm::MoEGroupedWmmaInput::OriginalTokens);
                ASSERT_EQ(args.activations, owners[path]->router_q8_publication->quantized_rows);
                ASSERT_EQ(args.activation_scales, owners[path]->router_q8_publication->row_scales);
                ASSERT_NE(args.second_descriptors, nullptr);
                ASSERT_EQ(args.output, workspaces[path]->getBuffer(MoEWorkspaceBuffers::PREFILL_GATE));
                ASSERT_EQ(args.second_output, workspaces[path]->getBuffer(MoEWorkspaceBuffers::PREFILL_UP));
            }
            else
            {
                ++native_roles[1]; ASSERT_EQ(launch.gridDim.z, 1u);
                ASSERT_EQ(args.group_token_indices, nullptr);
                ASSERT_EQ(args.output, workspaces[path]->getBuffer(MoEWorkspaceBuffers::PREFILL_GATE));
                ASSERT_EQ(launch.gridDim.x, unsigned(hidden / (columns ? 128 : tile)));
                ASSERT_NE(std::string_view(symbol).find("Format24E"), std::string_view::npos);
                ASSERT_EQ(args.n, hidden); ASSERT_EQ(args.k, intermediate);
                ASSERT_EQ(args.input, rocm::MoEGroupedWmmaInput::GroupedRows);
                ASSERT_EQ(args.activations, workspaces[path]->getBuffer(MoEWorkspaceBuffers::PREFILL_SWIGLU_INT8));
                ASSERT_EQ(args.activation_scales, workspaces[path]->getBuffer(MoEWorkspaceBuffers::PREFILL_SWIGLU_SCALES));
                ASSERT_EQ(args.second_descriptors, nullptr); ASSERT_EQ(args.second_output, nullptr);
                native_admission[path] = args.admission;
            }
        }
        ASSERT_EQ(native_nodes[path].size(), candidate.engine != Engine::OrderedDot4 && rows > 8 ? 2u * repeats : 0u);
        ASSERT_EQ(route_owned_gateup, repeats);
        for (int role_count : native_roles) ASSERT_EQ(role_count, candidate.engine != Engine::OrderedDot4 && rows > 8 ? repeats : 0);
        if (candidate.engine == Engine::WideDot4Wmma)
            for (int count : wide_directories) ASSERT_EQ(count,
                rows > 8 && candidate.directory == rocm::MoEGroupedDirectorySource::RolePlanner ? repeats : 0);
        ASSERT_EQ(packed_publishers[path].size(),candidate.directory != rocm::MoEGroupedDirectorySource::RolePlanner ? repeats : 0u);
        if (candidate.engine == Engine::WideDot4Wmma)
        {
            const auto roles = inspectROCmGroupedWideCapture(*parent, stream, candidate.floor,
                candidate.directory==rocm::MoEGroupedDirectorySource::GroupingColumnSlots
                    ? MoEGroupingPackedLayout::ColumnFloor8 : MoEGroupingPackedLayout::WideFloor8);
            ASSERT_EQ(roles[0], rows > 8 ? repeats : 0);
            ASSERT_EQ(roles[1], rows > 8 ? repeats : 0);
        }
        expected_submissions[engine_index(candidate.engine)] += repeats + 1;
        expected_submissions[4] += repeats + 1;
        ASSERT_EQ(submissions(), expected_submissions);
        std::vector<GPUGraphKernelNodeInfo> inventory; ASSERT_TRUE(graphs[path]->inspectKernelNodes(inventory));
        std::set<std::string> printed;
        for (const auto &node : inventory)
        {
            ASSERT_TRUE(node.valid()); ASSERT_EQ(node.local_memory_bytes_per_thread, 0u);
            if (printed.insert(node.name).second)
                std::printf("ROCM_ROUTED_VERIFIER_RESOURCE,%s,%s,%u,%zu,%u,%u,%u,%u,%s\n", c.name.c_str(), candidates[path].name.c_str(),
                    node.registers_per_thread, node.static_shared_memory_bytes, node.grid_x, node.grid_y, node.grid_z, node.block_x, node.name.c_str());
        }
    }
    const auto recorded_submissions = submissions();
    const auto poison = [&](int path) {
        for (const char *name : {MoEWorkspaceBuffers::PREFILL_GATE, MoEWorkspaceBuffers::PREFILL_UP,
                MoEWorkspaceBuffers::PREFILL_SWIGLU_INT8, MoEWorkspaceBuffers::PREFILL_SWIGLU_SCALES,
                MoEWorkspaceBuffers::GATEUP_GATE_PARTIALS, MoEWorkspaceBuffers::GATEUP_UP_PARTIALS,
                MoEWorkspaceBuffers::DOWN_PARTIALS, MoEWorkspaceBuffers::ROCM_ROUTER_Q8_HIDDEN,
                MoEWorkspaceBuffers::ROCM_ROUTER_Q8_SCALES, MoEWorkspaceBuffers::ROCM_GROUP_MAX_TOKENS,
                MoEWorkspaceBuffers::ROCM_PREFILL_WORK_DIRECTORY})
        {
            const auto bytes = workspaces[path]->getBufferSize(name);
            if (bytes) require(hipMemsetAsync(workspaces[path]->getBuffer(name), 0xff, bytes, stream) == hipSuccess, "Routed scratch poison failed");
        }
        require(hipMemsetAsync(workspaces[path]->getBuffer(MoEWorkspaceBuffers::PREFILL_A_INT8), 0x5a,
            workspaces[path]->getBufferSize(MoEWorkspaceBuffers::PREFILL_A_INT8), stream) == hipSuccess, "Routed private Q8 poison failed");
        for (auto *t : {outputs[path].get(), ids[path].get(), weights[path].get()})
            require(hipMemsetAsync(t->gpu_data_ptr(), 0xa5, t->numel() * sizeof(float), stream) == hipSuccess, "Routed output poison failed");
    };
    const auto observe = [&](int path) {
        require(backend->deviceToHost(observed.data(), outputs[path]->gpu_data_ptr(), observed.size() * sizeof(uint32_t), ordinal, stream), "Routed observation failed");
        require(backend->deviceToHost(observed_ids.data(), ids[path]->gpu_data_ptr(), observed_ids.size() * sizeof(float), ordinal, stream), "Routed ID observation failed");
        require(backend->deviceToHost(observed_weights.data(), weights[path]->gpu_data_ptr(), observed_weights.size() * sizeof(float), ordinal, stream), "Routed weight observation failed");
        require(hipStreamSynchronize(stream) == hipSuccess, "Routed observation completion failed");
    };
    const auto replay = [&](int path) {
        poison(path);
        require(graphs[path]->launch(), "Routed retained replay failed");
        observe(path);
    };
    const auto matches = [&] {
        for (size_t i = 0; i < observed.size(); ++i)
            if (observed[i] != (i < size_t(published_live) * hidden ? expected[i] : 0u)) return false;
        return true;
    };
    const auto verify_routes = [&] {
        for (size_t i = 0; i < observed_ids.size(); ++i)
        {
            const bool active = i < size_t(published_live) * top_k;
            ASSERT_EQ(observed_ids[i], active ? expected_ids[i] : -1.0f);
            ASSERT_EQ(std::bit_cast<uint32_t>(observed_weights[i]), active ? std::bit_cast<uint32_t>(expected_weights[i]) : 0u);
        }
    };
    if (mode == ExpertBenchmark::RoutedVerifierProfile)
    {
        ASSERT_EQ(rows, 16);
        replay(automatic_path); ASSERT_TRUE(matches()); ASSERT_NO_FATAL_FAILURE(verify_routes());
        for (int warm = 0; warm < 10; ++warm) ASSERT_TRUE(graphs[automatic_path]->launch());
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        poison(automatic_path);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        ASSERT_GE(roctxRangePushA("ROCmRoutedVerifierAutomatic"), 0);
        ASSERT_EQ(roctxProfilerResume(0), 0);
        const bool launched = graphs[automatic_path]->launch();
        const auto joined = hipStreamSynchronize(stream);
        const int paused = roctxProfilerPause(0), popped = roctxRangePop();
        ASSERT_TRUE(launched && joined == hipSuccess && paused == 0 && popped >= 0);
        observe(automatic_path); ASSERT_TRUE(matches()); ASSERT_NO_FATAL_FAILURE(verify_routes());
        ASSERT_EQ(submissions(), recorded_submissions);
        std::printf("ROCM_ROUTED_VERIFIER_PROFILE_OBSERVED,%s,%d,Automatic\n", c.name.c_str(), rows);
        return;
    }
    std::vector<int> live_sequence{rows, 8, std::min(rows, 9), 0, rows, rows};
    for (int n = 1; n <= rows; ++n) live_sequence.push_back(n);
    for (int n : live_sequence)
    {
        published_live = n; ASSERT_TRUE(backend->hostToDevice(live->gpu_data_ptr(), &published_live, sizeof(int), ordinal, stream));
        // This independent fixture histogram predicts omissions; production
        // still owns all routing and never downloads a live group count.
        std::array<int, experts> expert_rows{};
        for (int slot = 0; slot < n * top_k; ++slot) ++expert_rows[int(desired_ids[slot])];
        for (int path = 0; path < paths; ++path)
        {
            SCOPED_TRACE(::testing::Message() << c.name << " " << candidates[path].name.c_str() << " live=" << n);
            replay(path); ASSERT_TRUE(matches()); ASSERT_NO_FATAL_FAILURE(verify_routes());
            private_q8.resize(workspaces[path]->getBufferSize(MoEWorkspaceBuffers::PREFILL_A_INT8));
            ASSERT_TRUE(backend->deviceToHost(private_q8.data(), workspaces[path]->getBuffer(MoEWorkspaceBuffers::PREFILL_A_INT8), private_q8.size(), ordinal, stream));
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            ASSERT_TRUE(std::all_of(private_q8.begin(), private_q8.end(), [](auto byte) { return byte == 0x5a; }));
            if (!native_nodes[path].empty())
            {
                ASSERT_TRUE(native_admission[path]);
                const bool tiled = native_admission[path]->strategyForSlots(n * top_k) == rocm::ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled;
                const bool active = tiled;
                auto *parent = dynamic_cast<HIPGraphCapture *>(graphs[path].get());
                RoutedVerifierNodeOmission omission(parent->executable(), stream, native_nodes[path]);
                omission.omit(); replay(path); ASSERT_EQ(matches(), !active);
                omission.restore(); replay(path); ASSERT_TRUE(matches());
                if (!packed_publishers[path].empty())
                {
                    RoutedVerifierNodeOmission publisher(parent->executable(),stream,packed_publishers[path]);
                    publisher.omit(); replay(path); ASSERT_EQ(matches(),!active);
                    publisher.restore(); replay(path); ASSERT_TRUE(matches());
                }
                if (candidates[path].floor != Floor::EveryExpert)
                    for (auto role : {rocm::MoEPrefillProjectionRole::GateUp, rocm::MoEPrefillProjectionRole::Down})
                    {
                        ROCmGroupedBranchMutation branch(*parent, stream, role);
                        ASSERT_EQ(branch.nodeCount(), repeats);
                        for (auto suppressed : {rocm::MoEGroupedBranchSuppression::Ordered, rocm::MoEGroupedBranchSuppression::Wmma})
                        {
                            SCOPED_TRACE(::testing::Message() << "role=" << int(role) << " suppressed=" << int(suppressed));
                            branch.suppress(suppressed); replay(path);
                            for (int row = 0; row < rows; ++row)
                            {
                                bool affected = false;
                                for (int slot = 0; active && row < n && slot < top_k; ++slot)
                                {
                                    const bool native = expert_rows[int(desired_ids[size_t(row) * top_k + slot])] >= int(candidates[path].floor);
                                    affected |= native == (suppressed == rocm::MoEGroupedBranchSuppression::Wmma);
                                }
                                bool row_matches = true;
                                for (int col = 0; col < hidden; ++col)
                                {
                                    const size_t index = size_t(row) * hidden + col;
                                    row_matches &= observed[index] == (row < n ? expected[index] : 0u);
                                }
                                ASSERT_EQ(row_matches, !affected) << "row=" << row;
                            }
                            ASSERT_NO_FATAL_FAILURE(verify_routes());
                            branch.restore(); replay(path); ASSERT_TRUE(matches());
                        }
                    }
            }
            if (n)
            {
                q8_proofs[path]->omit(); replay(path); ASSERT_FALSE(matches());
                q8_proofs[path]->restore(); replay(path); ASSERT_TRUE(matches());
            }
        }
        if (!n)
        {
            ASSERT_TRUE(backend->hostToDevice(live->gpu_data_ptr(), &reset_values[0], sizeof(int), ordinal, stream));
            for (auto &stage : stages) if (stage) { stage->resetSessionStatePreservingCapturedReplay(); ASSERT_TRUE(stage->prepareGraphLaunch(&context, stream)); }
            ASSERT_TRUE(backend->deviceToHost(&reset_values[1], count, sizeof(int), ordinal, stream));
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess); ASSERT_EQ(reset_values[0], reset_values[1]);
        }
    }
    const int samples = mode == ExpertBenchmark::RoutedVerifierTournament ? 31 : 3;
    for (int n = 0; n <= rows; ++n)
    {
        published_live = n; ASSERT_TRUE(backend->hostToDevice(live->gpu_data_ptr(), &published_live, sizeof(int), ordinal, stream));
        for (int warm = 0; warm < 10; ++warm) for (auto &graph : graphs) if (graph) ASSERT_TRUE(graph->launch());
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        for (int sample = 0; sample < samples; ++sample)
        {
            std::array<double, max_paths> elapsed{};
            for (int order = 0; order < paths; ++order)
            {
                const int path = sample % 2 ? paths - 1 - order : order;
                replay(path); ASSERT_TRUE(matches()); ASSERT_NO_FATAL_FAILURE(verify_routes());
                const auto timing = graphs[path]->consumeOrderedTimelineTiming();
                ASSERT_EQ(timing.state, GPUOrderedTimelineTimingState::Complete); ASSERT_EQ(timing.samples.size(), 1u);
                ASSERT_EQ(timing.samples[0].name, candidates[path].name.c_str());
                ASSERT_EQ(timing.samples[0].kind, GPUOrderedTimelineStepKind::CapturedFragment);
                ASSERT_EQ(graphs[path]->consumeOrderedTimelineTiming().state, GPUOrderedTimelineTimingState::AwaitingLaunch);
                ASSERT_TRUE(std::isfinite(timing.samples[0].elapsed_ms) && timing.samples[0].elapsed_ms > 0);
                elapsed[path] = timing.samples[0].elapsed_ms * 1000.0 / repeats;
            }
            std::printf("ROCM_ROUTED_VERIFIER_PAIR,%s,%d,%d,%d", c.name.c_str(), rows, n, sample);
            for (int path = 0; path < paths; ++path) std::printf(",%.9f", elapsed[path]);
            std::printf("\n");
        }
    }
    ASSERT_EQ(submissions(), recorded_submissions);
}
