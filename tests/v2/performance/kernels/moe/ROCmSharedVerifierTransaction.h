/**
 * @file ROCmSharedVerifierTransaction.h
 * @brief Complete retained router/shared-FFN/gate-add evidence for grouped MTP.
 *
 * Included by the established ROCm MoE verifier benchmark inside its private
 * namespace, reusing its format/preparation helpers. Each candidate has its own
 * production stages, workspace and Q8 publication. Internal events time only
 * complete operations. Test-only node omission observes the raw FFN without
 * adding diagnostic copies to the timed graph. Independent serial M1 table
 * decode and gate/add own the expected bytes, including every physical suffix.
 */

/** @brief Separate fast lifecycle replay from paired Release collection. */
enum class SharedVerifierEvidence { Replay, Timing, ProfileN32, ProfileN64 };

/** @brief Original source formats and retained physical capacity identify a transaction. */
struct SharedVerifierCase { std::string name; const char *gateup; const char *down; int rows; };
/** @brief Independently filterable source/capacity case; live counts remain device data. */
class ROCmSharedVerifierTransaction : public ::testing::TestWithParam<SharedVerifierCase> {};
/** @return Canonical all-format M16 cases plus mixed-source M8/M9/M16 topology controls. */
std::vector<SharedVerifierCase> sharedVerifierTransactionCases()
{
    std::vector<SharedVerifierCase> cases;
    for (const auto &format : llaminar2::test::quantizedVerifierFormats())
        cases.push_back({std::string(format.label) + "_M16", format.label, format.label, 16});
    for (int rows : {8, 9, 16})
        cases.push_back({"IQ2S_IQ4XS_M" + std::to_string(rows), "IQ2_S", "IQ4_XS", rows});
    return cases;
}

/** @brief Restore omitted parent nodes before their retained graph can retire. */
class SharedVerifierNodeOmission final
{
    hipGraphExec_t executable_;
    hipStream_t stream_;
    const std::vector<hipGraphNode_t> &nodes_;
    bool omitted_ = false;
public:
    /** @param executable Actual parent executable. @param stream Exact submission stream.
     * @param nodes Actual nodes from that parent's graph; storage outlives this lease. */
    SharedVerifierNodeOmission(hipGraphExec_t executable, hipStream_t stream,
        const std::vector<hipGraphNode_t> &nodes) : executable_(executable), stream_(stream), nodes_(nodes) {}
    /** @brief Drain outstanding work and restore nodes, including on assertion exit. */
    ~SharedVerifierNodeOmission() { restore(); }
    SharedVerifierNodeOmission(const SharedVerifierNodeOmission &) = delete;
    SharedVerifierNodeOmission &operator=(const SharedVerifierNodeOmission &) = delete;
    /** @brief Omit the complete family; the live lease also restores partial mutation. */
    void omit()
    {
        requireHipBenchBody(!omitted_ && !nodes_.empty(), "shared verifier omission ownership");
        omitted_ = true;
        for (auto node : nodes_)
            requireHipBenchBody(hipGraphNodeSetEnabled(executable_, node, 0) == hipSuccess, "shared verifier omit node");
    }
    /** @brief Failed retirement/restoration is fatal because the executable is borrowed. */
    void restore() noexcept
    {
        if (!omitted_) return;
        if (hipStreamSynchronize(stream_) != hipSuccess) std::terminate();
        for (auto node : nodes_)
            if (hipGraphNodeSetEnabled(executable_, node, 1) != hipSuccess) std::terminate();
        omitted_ = false;
    }
};

/** @brief Qualify actual shared-stage ownership and complete-operation economics.
 * @param c Original source tuple and physical capture capacity.
 * @param evidence Replay, paired timing or isolated profiling; arithmetic is unchanged.
 * Geometry and mixed source formats come from the existing Qwen shared-stage
 * fixture. The residual is explicitly injected, not claimed as routed MoE work.
 */
void runROCmSharedVerifierTransaction(const SharedVerifierCase &c, SharedVerifierEvidence evidence)
{
    using namespace llaminar2;
    using Engine = rocm::MoEGroupedPrefillEngine;
    initializeROCmHardwareMetadata();
    int ordinal = -1;
    for (int i = 0; i < int(rocmHardwareMetadata().size()); ++i)
        if (rocmHardwareMetadata().at(i).isa_target.starts_with("gfx1100")) { ordinal = i; break; }
    if (ordinal < 0) GTEST_SKIP() << "Native gfx1100 is not visible";
    const bool profile = evidence == SharedVerifierEvidence::ProfileN32 || evidence == SharedVerifierEvidence::ProfileN64;
    if (profile && (::testing::UnitTest::GetInstance()->test_to_run_count() != 1 || c.rows <= 8))
        throw std::invalid_argument("Shared verifier profiling requires one native-capable source/capacity case");
    ScopedROCmBatchInvariantRouterPolicy router_policy;
    if (!DeviceManager::instance().hardware()) DeviceManager::instance().initialize(-1, false);
    const DeviceId device = DeviceId::rocm(ordinal);
    const auto &hardware = rocmHardwareMetadata().at(ordinal);
    std::printf("ROCM_SHARED_VERIFIER_DEVICE,%d,%s,%s,", ordinal, hardware.isa_target.c_str(), hardware.pci_address.c_str());
    for (auto byte : hardware.uuid) std::printf("%02x", unsigned(byte));
    std::printf("\n");
    auto &worker = GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait([&]
    {
        constexpr int hidden_width = 2048, width = 512, experts = 256, top_k = 8;
        const int rows = c.rows, repeats = profile ? 1 : 8;
        constexpr std::array engines{Engine::OrderedDot4, Engine::WmmaM16N32, Engine::WmmaM16N64};
        constexpr std::array names{"Dot4", "N32", "N64"};
        const auto stream = static_cast<hipStream_t>(worker.defaultStream());
        auto *backend = getBackendFor(device);
        ASSERT_NE(stream, nullptr);
        ASSERT_NE(backend, nullptr);
        auto &transfer = TransferEngine::instance();
        const auto &formats = test::quantizedVerifierFormats();
        const auto find_format = [&](const char *label) -> const test::QuantizedVerifierFormatCase & {
            const auto found = std::find_if(formats.begin(), formats.end(), [&](const auto &f) { return std::string_view(f.label) == label; });
            if (found == formats.end()) throw std::invalid_argument("Unknown canonical shared verifier source format");
            return *found;
        };
        const auto &format = find_format(c.gateup), &down_format = find_format(c.down);
        test::requireQuantizedVerifierSourceShape(format, width, hidden_width);
        test::requireQuantizedVerifierSourceShape(down_format, hidden_width, width);
        auto gate_w = format.create({size_t(width), size_t(hidden_width)}, 6101);
        auto up_w = format.create({size_t(width), size_t(hidden_width)}, 6102);
        auto down_w = down_format.create({size_t(hidden_width), size_t(width)}, 6103);
        ASSERT_EQ(gate_w->native_type(), format.tensor_type);
        ASSERT_EQ(up_w->native_type(), format.tensor_type);
        ASSERT_EQ(down_w->native_type(), down_format.tensor_type);
        auto prepared = test::makeGpuPreparedFFNFixture(gate_w.get(), up_w.get(), down_w.get(), device,
            "perf.rocm.shared_verifier_transaction." + c.name, ModelContextId{390010});
        const auto hidden_values = makeHiddenValues(rows, hidden_width);
        auto hidden = makeTensor({size_t(rows), size_t(hidden_width)}, hidden_values);
        auto residual = makeTensor({size_t(rows), size_t(hidden_width)}, makeHiddenValues(rows, hidden_width));
        auto gate_vector = makeTensor({size_t(hidden_width)}, makeHiddenValues(1, hidden_width));
        auto router_gate = makeTensor({size_t(experts), size_t(hidden_width)}, makeHiddenValues(experts, hidden_width));
        INT32Tensor live({size_t{1}}, std::vector<int32_t>{rows});
        auto serial_hidden = makeZeros({size_t{1}, size_t(hidden_width)});
        auto serial_gate = makeZeros({size_t{1}, size_t(width)});
        auto serial_up = makeZeros({size_t{1}, size_t(width)});
        auto serial_output = makeZeros({size_t{1}, size_t(hidden_width)});
        auto serial_residual = makeZeros({size_t{1}, size_t(hidden_width)});
        auto serial_combined = makeZeros({size_t{1}, size_t(hidden_width)});
        std::array<std::shared_ptr<FP32Tensor>, 3> output, combined, gate_scratch, up_scratch, route_ids, route_weights;
        std::array<std::unique_ptr<IMoEKernel>, 3> router, shared;
        std::array<std::shared_ptr<MoERouterQ8HiddenPublication>, 3> publication;
        std::array<std::unique_ptr<SharedExpertFFNStage>, 3> stages;
        std::array<std::unique_ptr<SharedExpertGateStage>, 3> gates;
        std::array<std::unique_ptr<DeviceWorkspaceManager>, 3> workspaces;
        std::array<std::unique_ptr<IGPUGraphCapture>, 3> fragments, graphs;
        std::array<std::vector<hipGraphNode_t>, 3> gate_nodes, native_nodes, router_nodes;
        std::array<std::vector<uint32_t>, 3> expected;
        for (auto &bank : expected) bank.resize(size_t(rows) * hidden_width);
        std::vector<uint32_t> observed(expected[0].size()), combined_observed(observed.size());
        std::vector<uint8_t> private_observed;
        int published_live = rows;
        std::array<int32_t, 2> reset_values{std::numeric_limits<int32_t>::max(), 0};
        auto serial_kernel = KernelFactory::createMoEKernel(device);
        ASSERT_NE(serial_kernel, nullptr);
        serial_kernel->setGPUStream(stream);
        auto serial_requirements = MoEWorkspaceBuffers::rocmMoE(1, hidden_width, width, experts, top_k);
        auto serial_workspace = std::make_unique<DeviceWorkspaceManager>(device, serial_requirements.total_bytes_with_alignment());
        ASSERT_TRUE(serial_workspace->allocate(serial_requirements));
        auto *serial_consumer = dynamic_cast<IWorkspaceConsumer *>(serial_kernel.get());
        ASSERT_NE(serial_consumer, nullptr);
        serial_consumer->bindWorkspace(serial_workspace.get());
        // This final guard precedes every possible release of staging, graphs,
        // workspaces or prepared sources, including an assertion during setup.
        const auto retire = [&](void *) {
            if (hipStreamSynchronize(stream) != hipSuccess) std::terminate();
        };
        std::unique_ptr<void, decltype(retire)> completion(&worker, retire);
        const auto upload = [&](TensorBase *tensor) {
            requireHipBenchBody(transfer.uploadFull(tensor, device, stream).success, "shared verifier tensor preparation");
        };
        for (TensorBase *tensor : std::array<TensorBase *, 11>{hidden.get(), residual.get(), gate_vector.get(), router_gate.get(),
                &live, serial_hidden.get(), serial_gate.get(), serial_up.get(), serial_output.get(), serial_residual.get(), serial_combined.get()})
            upload(tensor);
        const auto *count = static_cast<const int32_t *>(live.gpu_data_ptr());
        DeviceNativeQuantisedMatrixDesc gd{}, ud{}, dd{};
        ASSERT_TRUE(prepared.gate_kernel->exportNativeQuantisedMatrixDesc(gd));
        ASSERT_TRUE(prepared.up_kernel->exportNativeQuantisedMatrixDesc(ud));
        ASSERT_TRUE(prepared.down_kernel->exportNativeQuantisedMatrixDesc(dd));
        const int gu_table = serial_kernel->uploadGroupedExpertGateUpDescriptorTables(&gd, &ud, 1, hidden_width, width);
        const int down_table = serial_kernel->uploadGroupedExpertDownDescriptorTable(&dd, 1, hidden_width, width);
        ASSERT_GE(gu_table, 0);
        ASSERT_GE(down_table, 0);
        // Independent serial M1 uses original table decode, never the grouped
        // prefill candidate or an eager multi-row Dot4 snapshot as its oracle.
        const int expert = 0;
        const float weight = 1.0f;
        ITensor *gate_outputs[]{serial_gate.get()}, *up_outputs[]{serial_up.get()};
        for (int row = 0; row < rows; ++row)
        {
            const size_t offset = size_t(row) * hidden_width;
            ASSERT_TRUE(backend->hostToDevice(serial_hidden->gpu_data_ptr(), hidden_values.data() + offset,
                hidden_width * sizeof(float), ordinal, stream));
            ASSERT_TRUE(backend->hostToDevice(serial_residual->gpu_data_ptr(), hidden_values.data() + offset,
                hidden_width * sizeof(float), ordinal, stream));
            ASSERT_TRUE(serial_kernel->groupedExpertGateUpDecodeFromTable(serial_hidden.get(), &expert, gu_table, 1,
                gate_outputs, up_outputs, hidden_width, width));
            ASSERT_TRUE(serial_kernel->groupedExpertDownDecodeFromTable(gate_outputs, up_outputs, &expert, &weight,
                down_table, 1, serial_output.get(), hidden_width, width));
            ASSERT_TRUE(backend->deviceToHost(expected[0].data() + offset, serial_output->gpu_data_ptr(),
                hidden_width * sizeof(float), ordinal, stream));
            serial_kernel->sharedExpertGateAddFromTensors(serial_hidden.get(), gate_vector.get(), serial_output.get(),
                serial_residual.get(), serial_combined.get(), 1, hidden_width);
            ASSERT_TRUE(backend->deviceToHost(expected[1].data() + offset, serial_output->gpu_data_ptr(),
                hidden_width * sizeof(float), ordinal, stream));
            ASSERT_TRUE(backend->deviceToHost(expected[2].data() + offset, serial_combined->gpu_data_ptr(),
                hidden_width * sizeof(float), ordinal, stream));
        }
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        for (const auto &bank : expected)
        {
            ASSERT_TRUE(std::all_of(bank.begin(), bank.end(), [](auto word) { return std::isfinite(std::bit_cast<float>(word)); }));
            ASSERT_TRUE(std::any_of(bank.begin(), bank.end(), [](auto word) { return word != 0; }));
        }
        llaminar2::testing::MockDeviceContext context(device, ComputeBackendType::GPU_ROCM);
        const auto enqueue = [&](int path) {
            return router[path]->routeVerifierRowsDecodeEquivalent(hidden.get(), router_gate.get(), rows, hidden_width,
                experts, top_k, true, route_ids[path].get(), route_weights[path].get(), count) &&
                stages[path]->execute(&context) && gates[path]->execute(&context);
        };
        for (int path = 0; path < 3; ++path)
        {
            output[path] = makeZeros({size_t(rows), size_t(hidden_width)});
            combined[path] = makeZeros({size_t(rows), size_t(hidden_width)});
            gate_scratch[path] = makeZeros({size_t(rows), size_t(width)});
            up_scratch[path] = makeZeros({size_t(rows), size_t(width)});
            route_ids[path] = makeZeros({size_t(rows), size_t(top_k)});
            route_weights[path] = makeZeros({size_t(rows), size_t(top_k)});
            for (auto *tensor : {output[path].get(), combined[path].get(), gate_scratch[path].get(), up_scratch[path].get(),
                    route_ids[path].get(), route_weights[path].get()}) upload(tensor);
            router[path] = KernelFactory::createMoEKernel(device);
            shared[path] = KernelFactory::createMoEKernel(device);
            ASSERT_TRUE(router[path] && shared[path]);
            router[path]->setGPUStream(stream); shared[path]->setGPUStream(stream);
            publication[path] = std::make_shared<MoERouterQ8HiddenPublication>();
            ASSERT_TRUE(router[path]->bindRouterQ8HiddenPublication(publication[path], MoERouterQ8PublicationAccess::ProducerAndConsumer));
            ASSERT_TRUE(shared[path]->bindRouterQ8HiddenPublication(publication[path], MoERouterQ8PublicationAccess::RequiredConsumer));
            SharedExpertFFNStage::Params params;
            params.device_id = device; params.input = hidden.get(); params.output = output[path].get();
            params.gate_w = gate_w.get(); params.up_w = up_w.get(); params.down_w = down_w.get();
            params.gate_scratch = gate_scratch[path].get(); params.up_scratch = up_scratch[path].get();
            params.seq_len = rows; params.d_model = hidden_width; params.intermediate = width;
            params.prepared_ref_gate = prepared.gate_ref; params.prepared_ref_up = prepared.up_ref;
            params.prepared_ref_down = prepared.down_ref; params.prepared_store = prepared.store.get();
            params.force_grouped_verifier_prefill_for_decode = true;
            params.force_decode_equivalent_verifier_prefill = false;
            params.active_row_count_device = count; params.required_router_q8_publication = publication[path];
            stages[path] = std::make_unique<SharedExpertFFNStage>(params);
            stages[path]->setGPUStream(stream); stages[path]->setMoEKernelForTesting(shared[path].get());
            SharedExpertGateStage::Params epilogue;
            epilogue.device_id = device; epilogue.input = hidden.get(); epilogue.gate_inp = gate_vector.get();
            epilogue.shared_output = output[path].get(); epilogue.routed_residual = residual.get();
            epilogue.combined_output = combined[path].get(); epilogue.seq_len = rows; epilogue.d_model = hidden_width;
            epilogue.active_row_count_device = count;
            gates[path] = std::make_unique<SharedExpertGateStage>(epilogue);
            gates[path]->setGPUStream(stream);
            auto requirements = stages[path]->getWorkspaceRequirements(rows, hidden_width, width);
            requirements.merge(MoEWorkspaceBuffers::rocmMoE(rows, hidden_width, width, experts, top_k));
            workspaces[path] = std::make_unique<DeviceWorkspaceManager>(device, requirements.total_bytes_with_alignment());
            ASSERT_TRUE(workspaces[path]->allocate(requirements));
            stages[path]->bindWorkspace(workspaces[path].get()); gates[path]->bindWorkspace(workspaces[path].get());
            for (auto *kernel : {router[path].get(), shared[path].get()})
            {
                auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
                ASSERT_NE(consumer, nullptr); consumer->bindWorkspace(workspaces[path].get());
            }
            ASSERT_TRUE(router[path]->prepareRouteLaunch(router_gate.get(), {
                .kind = MoERouteLaunchKind::DecodeEquivalentVerifier, .physical_rows = rows,
                .d_model = hidden_width, .num_experts = experts, .top_k = top_k}));
            ASSERT_TRUE(stages[path]->prepareGraphLaunch(&context, stream));
            ASSERT_TRUE(gates[path]->prepareGraphLaunch(&context, stream));
            rocm::ScopedMoEGroupedPrefillEngine engine(engines[path], rocm::MoEGroupedWmmaRowFloor::EveryExpert);
            ASSERT_TRUE(enqueue(path));
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            // The executor's external-input pass joins the final warmup
            // publications before capture. In-place gate/add republishes its
            // input, so an earlier warmup read is not that final event edge.
            for (auto *tensor : {hidden.get(), residual.get(), gate_vector.get(), router_gate.get(),
                    output[path].get(), combined[path].get(), gate_scratch[path].get(), up_scratch[path].get(),
                    route_ids[path].get(), route_weights[path].get()})
                TransferEngine::requireDeviceInput(tensor, device, stream);
            graphs[path] = worker.createGraphCapture(stream);
            fragments[path] = graphs[path]->createOrderedTimelineFragment();
            ASSERT_TRUE(fragments[path]);
            ScopedBackendGraphCapture capture(worker, *fragments[path], "router shared FFN gate/add transaction");
            ASSERT_TRUE(capture.begin());
            for (int repeat = 0; repeat < repeats; ++repeat) ASSERT_TRUE(enqueue(path));
            capture.finish();
            const std::array<GPUOrderedTimelineStep, 1> steps{{{
                .name = names[path], .kind = GPUOrderedTimelineStepKind::CapturedFragment, .capture = fragments[path].get()}}};
            ASSERT_TRUE(graphs[path]->buildOrderedTimelineTransaction(steps, profile ?
                GPUOrderedTimelineInstrumentation::Disabled : GPUOrderedTimelineInstrumentation::PerStepEvents));
            ASSERT_TRUE(graphs[path]->instantiate());
            auto *parent = dynamic_cast<HIPGraphCapture *>(graphs[path].get());
            ASSERT_NE(parent, nullptr);
            size_t node_count = 0;
            ASSERT_EQ(hipGraphGetNodes(parent->graph(), nullptr, &node_count), hipSuccess);
            std::vector<hipGraphNode_t> nodes(node_count);
            ASSERT_EQ(hipGraphGetNodes(parent->graph(), nodes.data(), &node_count), hipSuccess);
            int native_gateup = 0, native_down = 0;
            for (auto node : nodes)
            {
                hipGraphNodeType type;
                ASSERT_EQ(hipGraphNodeGetType(node, &type), hipSuccess);
                if (type != hipGraphNodeTypeKernel) continue;
                hipKernelNodeParams launch{};
                ASSERT_EQ(hipGraphKernelNodeGetParams(node, &launch), hipSuccess);
                const char *name = hipKernelNameRefByPtr(launch.func, stream);
                ASSERT_NE(name, nullptr);
                const std::string_view symbol(name);
                ASSERT_EQ(symbol.find("moe_prefill_hidden_quantize_rows_kernel"), std::string_view::npos);
                if (symbol.find("rocm_moe_shared_expert_gate_add_kernel") != std::string_view::npos)
                {
                    ASSERT_EQ(*static_cast<float **>(launch.kernelParams[2]), output[path]->gpu_data_ptr());
                    ASSERT_EQ(*static_cast<const float **>(launch.kernelParams[3]), residual->gpu_data_ptr());
                    ASSERT_EQ(*static_cast<float **>(launch.kernelParams[4]), combined[path]->gpu_data_ptr());
                    ASSERT_EQ(*static_cast<int *>(launch.kernelParams[5]), rows);
                    ASSERT_EQ(*static_cast<int *>(launch.kernelParams[6]), hidden_width);
                    ASSERT_EQ(*static_cast<const int **>(launch.kernelParams[7]), count);
                    gate_nodes[path].push_back(node);
                }
                if (symbol.find("rocm_moe_router_hidden_quantize_q8_rows_kernel") != std::string_view::npos)
                {
                    ASSERT_EQ(*static_cast<const float **>(launch.kernelParams[0]), hidden->gpu_data_ptr());
                    ASSERT_EQ(*static_cast<int8_t **>(launch.kernelParams[1]), publication[path]->quantized_rows);
                    ASSERT_EQ(*static_cast<float **>(launch.kernelParams[2]), publication[path]->row_scales);
                    ASSERT_EQ(*static_cast<int *>(launch.kernelParams[3]), rows);
                    ASSERT_EQ(*static_cast<int *>(launch.kernelParams[4]), hidden_width);
                    ASSERT_EQ(*static_cast<const int **>(launch.kernelParams[5]), count);
                    router_nodes[path].push_back(node);
                }
                if (symbol.find("grouped_wmma_detail") != std::string_view::npos)
                {
                    native_nodes[path].push_back(node);
                    const auto &args = *static_cast<const rocm::MoEGroupedWmmaArguments *>(launch.kernelParams[0]);
                    ASSERT_GT(path, 0);
                    ASSERT_EQ(launch.blockDim.x, path == 1 ? 128u : 256u);
                    ASSERT_EQ(launch.blockDim.y, 1u); ASSERT_EQ(launch.blockDim.z, 1u);
                    ASSERT_EQ(args.rows, rows); ASSERT_EQ(args.experts, 1); ASSERT_EQ(args.top_k, 1);
                    if (launch.gridDim.z == 2)
                    {
                        ++native_gateup;
                        ASSERT_EQ(launch.gridDim.x, unsigned(width / (path == 1 ? 32 : 64)));
                        ASSERT_EQ(args.activations, publication[path]->quantized_rows);
                        ASSERT_EQ(args.activation_scales, publication[path]->row_scales);
                        ASSERT_EQ(args.input, rocm::MoEGroupedWmmaInput::OriginalTokens);
                        ASSERT_EQ(args.rows, rows); ASSERT_EQ(args.k, hidden_width); ASSERT_EQ(args.n, width);
                    }
                    else
                    {
                        ++native_down;
                        ASSERT_EQ(launch.gridDim.z, 1u);
                        ASSERT_EQ(launch.gridDim.x, unsigned(hidden_width / (path == 1 ? 32 : 64)));
                        ASSERT_EQ(args.n, hidden_width); ASSERT_EQ(args.k, width);
                        ASSERT_EQ(args.input, rocm::MoEGroupedWmmaInput::GroupedRows);
                        ASSERT_EQ(args.activations, workspaces[path]->getBuffer(MoEWorkspaceBuffers::PREFILL_SWIGLU_INT8));
                        ASSERT_EQ(args.activation_scales, workspaces[path]->getBuffer(MoEWorkspaceBuffers::PREFILL_SWIGLU_SCALES));
                    }
                }
            }
            ASSERT_EQ(native_gateup, path && rows > 8 ? repeats : 0);
            ASSERT_EQ(native_down, path && rows > 8 ? repeats : 0);
            ASSERT_EQ(gate_nodes[path].size(), repeats);
            ASSERT_EQ(router_nodes[path].size(), repeats);
            ASSERT_EQ(native_nodes[path].size(), path && rows > 8 ? 2 * repeats : 0);
            std::vector<GPUGraphKernelNodeInfo> inventory;
            ASSERT_TRUE(graphs[path]->inspectKernelNodes(inventory));
            std::set<std::string> printed;
            for (const auto &node : inventory)
            {
                ASSERT_TRUE(node.valid()); ASSERT_EQ(node.local_memory_bytes_per_thread, 0u) << node.name;
                if (printed.insert(node.name).second)
                    std::printf("ROCM_SHARED_VERIFIER_RESOURCE,%s,%s,%u,%zu,%u,%u,%u,%u,%s\n", c.name.c_str(), names[path],
                        node.registers_per_thread, node.static_shared_memory_bytes, node.grid_x, node.grid_y,
                        node.grid_z, node.block_x, node.name.c_str());
            }
        }
        const auto poison = [&](int path) {
            auto &workspace = *workspaces[path];
            for (const char *name : {MoEWorkspaceBuffers::PREFILL_GATE, MoEWorkspaceBuffers::PREFILL_UP,
                    MoEWorkspaceBuffers::PREFILL_SWIGLU_INT8, MoEWorkspaceBuffers::PREFILL_SWIGLU_SCALES,
                    MoEWorkspaceBuffers::GATEUP_GATE_PARTIALS, MoEWorkspaceBuffers::GATEUP_UP_PARTIALS,
                    MoEWorkspaceBuffers::DOWN_PARTIALS})
            {
                const auto bytes = workspace.getBufferSize(name);
                if (bytes) requireHipBenchBody(hipMemsetAsync(workspace.getBuffer(name), 0xff, bytes, stream) == hipSuccess,
                    "shared verifier consumed scratch poison");
            }
            requireHipBenchBody(hipMemsetAsync(workspace.getBuffer(MoEWorkspaceBuffers::PREFILL_A_INT8), 0x5a,
                workspace.getBufferSize(MoEWorkspaceBuffers::PREFILL_A_INT8), stream) == hipSuccess, "shared verifier private Q8 poison");
            requireHipBenchBody(hipMemsetAsync(const_cast<int8_t *>(publication[path]->quantized_rows), 0x5a,
                size_t(rows) * hidden_width, stream) == hipSuccess, "shared verifier router Q8 poison");
            requireHipBenchBody(hipMemsetAsync(const_cast<float *>(publication[path]->row_scales), 0xff,
                size_t(rows) * (hidden_width / 32) * sizeof(float), stream) == hipSuccess, "shared verifier router scale poison");
            for (auto *tensor : {output[path].get(), combined[path].get()})
                requireHipBenchBody(hipMemsetAsync(tensor->gpu_data_ptr(), 0xa5, observed.size() * sizeof(uint32_t), stream) == hipSuccess,
                    "shared verifier output poison");
        };
        const auto observe = [&](int path) {
            requireHipBenchBody(backend->deviceToHost(observed.data(), output[path]->gpu_data_ptr(),
                observed.size() * sizeof(uint32_t), ordinal, stream), "shared verifier output observation");
            requireHipBenchBody(backend->deviceToHost(combined_observed.data(), combined[path]->gpu_data_ptr(),
                combined_observed.size() * sizeof(uint32_t), ordinal, stream), "shared verifier combined observation");
            requireHipBenchBody(hipStreamSynchronize(stream) == hipSuccess, "shared verifier observation completion");
        };
        const auto matches = [&](const std::vector<uint32_t> &actual, int bank) {
            for (size_t i = 0; i < actual.size(); ++i)
                if (actual[i] != (i < size_t(published_live) * hidden_width ? expected[bank][i] : uint32_t{0})) return false;
            return true;
        };
        const auto replay = [&](int path) {
            poison(path);
            requireHipBenchBody(graphs[path]->launch(), "shared verifier retained replay");
            observe(path);
        };
        std::vector<int> live_sequence{rows, std::min(rows, 8), std::min(rows, 9), 0, rows, rows};
        for (int active = 1; active <= rows; ++active) live_sequence.push_back(active);
        for (const int live_rows : live_sequence)
        {
            published_live = live_rows;
            ASSERT_TRUE(backend->hostToDevice(live.gpu_data_ptr(), &published_live, sizeof(published_live), ordinal, stream));
            for (int path = 0; path < 3; ++path)
            {
                SCOPED_TRACE(::testing::Message() << names[path] << " live=" << live_rows);
                auto *parent = dynamic_cast<HIPGraphCapture *>(graphs[path].get());
                SharedVerifierNodeOmission gate_omission(parent->executable(), stream, gate_nodes[path]);
                gate_omission.omit(); replay(path);
                ASSERT_TRUE(matches(observed, 0)) << "raw FFN differs from original M1 table decode";
                ASSERT_TRUE(std::all_of(combined_observed.begin(), combined_observed.end(), [](auto word) { return word == 0xa5a5a5a5u; }));
                gate_omission.restore(); replay(path);
                ASSERT_TRUE(matches(observed, 1)) << "gated FFN differs from serial gate/add";
                ASSERT_TRUE(matches(combined_observed, 2)) << "combined output differs from serial gate/add";
                auto &workspace = *workspaces[path];
                private_observed.resize(workspace.getBufferSize(MoEWorkspaceBuffers::PREFILL_A_INT8));
                ASSERT_TRUE(backend->deviceToHost(private_observed.data(), workspace.getBuffer(MoEWorkspaceBuffers::PREFILL_A_INT8),
                    private_observed.size(), ordinal, stream));
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
                ASSERT_TRUE(std::all_of(private_observed.begin(), private_observed.end(), [](auto byte) { return byte == 0x5a; }));
                if (path && rows > 8)
                {
                    SharedVerifierNodeOmission native_omission(parent->executable(), stream, native_nodes[path]);
                    native_omission.omit(); replay(path);
                    ASSERT_EQ(matches(combined_observed, 2), live_rows <= 8) << "retained native topology is not active-family evidence";
                    native_omission.restore(); replay(path);
                    ASSERT_TRUE(matches(observed, 1) && matches(combined_observed, 2));
                }
                if (live_rows > 0)
                {
                    SharedVerifierNodeOmission router_omission(parent->executable(), stream, router_nodes[path]);
                    router_omission.omit(); replay(path);
                    ASSERT_FALSE(matches(combined_observed, 2)) << "shared path did not consume poisoned router Q8";
                    router_omission.restore(); replay(path);
                    ASSERT_TRUE(matches(observed, 1) && matches(combined_observed, 2));
                }
            }
            if (live_rows == 0)
            {
                // Reset data without replacing any retained node or count
                // binding. Cold preparation must not consume an unpublished count.
                ASSERT_TRUE(backend->hostToDevice(live.gpu_data_ptr(), &reset_values[0], sizeof(int32_t), ordinal, stream));
                for (auto &stage : stages)
                {
                    stage->resetSessionStatePreservingCapturedReplay();
                    ASSERT_TRUE(stage->prepareGraphLaunch(&context, stream));
                }
                ASSERT_TRUE(backend->deviceToHost(&reset_values[1], count, sizeof(int32_t), ordinal, stream));
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
                ASSERT_EQ(reset_values[0], reset_values[1]);
            }
        }
        if (profile)
        {
            published_live = rows;
            ASSERT_TRUE(backend->hostToDevice(live.gpu_data_ptr(), &published_live, sizeof(published_live), ordinal, stream));
            const int path = evidence == SharedVerifierEvidence::ProfileN32 ? 1 : 2;
            for (int warm = 0; warm < 10; ++warm) ASSERT_TRUE(graphs[path]->launch());
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            // Poison outside the selected region so the profiled replay must
            // independently define every consumed output and scratch value.
            poison(path);
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            requireHipBenchBody(roctxRangePushA("ROCmSharedVerifierTransaction") >= 0, "shared profile range");
            if (roctxProfilerResume(0) != 0) { (void)roctxRangePop(); throw std::runtime_error("shared profile resume"); }
            const bool launched = graphs[path]->launch();
            const bool joined = hipStreamSynchronize(stream) == hipSuccess;
            const int paused = roctxProfilerPause(0), popped = roctxRangePop();
            requireHipBenchBody(launched && joined && paused == 0 && popped >= 0, "shared profile completion");
            observe(path);
            ASSERT_TRUE(matches(observed, 1) && matches(combined_observed, 2))
                << "profiled replay differs from serial gate/add";
            std::printf("ROCM_SHARED_VERIFIER_PROFILE_OBSERVED,%s,%s,%d\n", c.name.c_str(), names[path], rows);
            return;
        }
        const int samples = evidence == SharedVerifierEvidence::Timing ? 31 : 3;
        for (int live_rows = 0; live_rows <= rows; ++live_rows)
        {
            published_live = live_rows;
            ASSERT_TRUE(backend->hostToDevice(live.gpu_data_ptr(), &published_live, sizeof(published_live), ordinal, stream));
            for (int warm = 0; warm < 10; ++warm)
                for (auto &graph : graphs) ASSERT_TRUE(graph->launch());
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            for (int sample = 0; sample < samples; ++sample)
            {
                std::array<double, 3> elapsed{};
                for (int order = 0; order < 3; ++order)
                {
                    const int path = sample % 2 ? 2 - order : order;
                    replay(path);
                    ASSERT_TRUE(matches(observed, 1) && matches(combined_observed, 2));
                    const auto timing = graphs[path]->consumeOrderedTimelineTiming();
                    ASSERT_EQ(timing.state, GPUOrderedTimelineTimingState::Complete);
                    ASSERT_EQ(timing.samples.size(), 1u);
                    ASSERT_EQ(timing.samples[0].name, names[path]);
                    ASSERT_EQ(timing.samples[0].kind, GPUOrderedTimelineStepKind::CapturedFragment);
                    ASSERT_TRUE(std::isfinite(timing.samples[0].elapsed_ms) && timing.samples[0].elapsed_ms > 0);
                    ASSERT_EQ(graphs[path]->consumeOrderedTimelineTiming().state, GPUOrderedTimelineTimingState::AwaitingLaunch);
                    elapsed[path] = timing.samples[0].elapsed_ms * 1000.0 / repeats;
                }
                std::printf("ROCM_SHARED_VERIFIER_PAIR,%s,%d,%d,%d,%.9f,%.9f,%.9f\n", c.name.c_str(), rows, live_rows, sample,
                    elapsed[0], elapsed[1], elapsed[2]);
            }
        }
    });
}
