/**
 * @file RCCLRankOrderedReplay.inc
 * @brief Captured numerical/lifetime replay and separate timing of native ordered FP32.
 *
 * The original latency fixture owns communicators, streams and admitted test
 * buffers. Both arms use those same owners, native payload receipts and exact
 * input bits. The candidate is an explicit dependency operation; no transport,
 * communicator or kernel implementation is copied into the test.
 */

/**
 * @brief Replay dirty fixed/device-counted banks, or time a retained operation chain.
 * @param mode Functional replay, paired timing, or an isolated exact profiler arm.
 * @param profileRows Positive row count for profiler modes; zero for other modes.
 * @param tinyPush Exercise the lightweight production kernel instead of the native FIFO candidate.
 *
 * Functional mode groups two independent outputs to expose descriptor aliasing.
 * In-place/out-of-place bindings, empty/partial/full/reset and twenty repeats
 * retain their captured graph. Timing uses immutable separate input/output banks
 * so repeated transactions do not multiply their own previous results. One
 * arrival transaction precedes captured timing events: the measured chain owns
 * native communication latency, without initial host graph-submission skew.
 */
void Perf__RCCLAllreduceLatency::runRankOrderedReplay(RankOrderedReplayMode mode, int profileRows, bool tinyPush)
{
    if (inactive_owner_) return; // Original discovery peer owns no selected GPU work.
    ASSERT_TRUE(initialized_);
    const bool mixedPrefill=mode==RankOrderedReplayMode::MixedPrefill;
    const bool aborting=mode==RankOrderedReplayMode::Abort;
    const bool profiling=mode==RankOrderedReplayMode::ControlProfile || mode==RankOrderedReplayMode::FusedProfile;
    ASSERT_EQ(profileRows>0,profiling);
    const bool timing=mode!=RankOrderedReplayMode::Functional && !mixedPrefill && !aborting;
    const bool candidate=mode!=RankOrderedReplayMode::ControlTiming && mode!=RankOrderedReplayMode::ControlProfile;
    const int ranks = num_devices_;
    // This is the test's terminal observation boundary, outside all captured
    // operations and event timing. A broken peer protocol must trigger the
    // original native abort flag instead of trapping fixture teardown forever.
    // Signal-only abort preserves communicator storage until GPU retirement.
    const auto awaitRetirement = [&]() -> bool {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        for (;;) {
            bool complete=true, failed=false;
            for (int rank=0; rank<(aborting?1:ranks); ++rank) {
                const auto selected=hipSetDevice(devices_[rank].ordinal);
                const auto status=selected==hipSuccess?hipEventQuery(devices_[rank].stop_event):selected;
                if (status!=hipSuccess) complete=false;
                if (status!=hipSuccess && status!=hipErrorNotReady) failed=true;
            }
            if (complete) return true;
            if (failed || std::chrono::steady_clock::now()>=deadline) {
                for (const auto comm:comms_)
                    EXPECT_EQ(rccl::signalNativeCompletionAbort(comm),rccl::ncclSuccess);
                ADD_FAILURE() << "Native ordered replay failed or exceeded the 30-second collective deadline";
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    };
    struct GraphOwner
    {
        int device = -1;
        hipStream_t stream = nullptr;
        hipGraph_t graph = nullptr;
        hipGraphExec_t executable = nullptr;
        /** @brief Retire replay before graph handles and the outer original owners. */
        ~GraphOwner()
        {
            if (device < 0) return;
            (void)hipSetDevice(device);
            if (stream) (void)hipStreamSynchronize(stream);
            if (executable) (void)hipGraphExecDestroy(executable);
            if (graph) (void)hipGraphDestroy(graph);
        }
    };
    struct Shape { int capacity; int width; bool counted = true; };
    const std::vector<Shape> shapes = aborting ? std::vector<Shape>{{1,2560}} :
        mixedPrefill ? std::vector<Shape>{{1,2560},{3,2560},{7,257}} :
        profiling ? std::vector<Shape>{{profileRows,2560}} :
        // Cover serial decode, verifier depths and the crossover toward prefill.
        // A small-row win cannot justify dispatching every larger bank identically.
        tinyPush && timing ? std::vector<Shape>{{1,33},{1,256},{1,1024},{1,2560},
            {2,2560},{3,2560},{4,2560},{8,2560},{16,2560},{24,2560},{32,2560},{40,2560},
            {48,2560},{56,2560},{63,2560},{64,2560},{65,2560},{80,2560},{96,2560},{1,4096},{3,4096},
            {8,4096},{16,4096},{32,4096},{1,8192},{3,8192},{8,8192},{16,8192},{32,8192}} :
        tinyPush ? std::vector<Shape>{{1,33},{3,257},{1,2560},{3,2560},{7,2560},
            {16,4096},{63,2560},{64,2560},{1,33},{3,257,false},{1,2560,false},{63,2560,false}} :
        timing ? std::vector<Shape>{{1,2560},{2,2560},{3,2560},{4,2560},
            {8,2560},{16,2560},{32,2560},{64,2560},{129,2560}} :
        std::vector<Shape>{{1,33},{3,257},{1,2560},{3,2560},{7,2560},{129,2560},
            {3,257,false},{129,2560,false}};
    // Exercise the actual production storage owner and PMA claims. Only the
    // passive metadata receipt below is fixture storage. The original fixture's
    // peer setup, communicator and abort owner remain authoritative throughout.
    struct PushStorage
    {
        int device = -1;
        hipStream_t stream = nullptr;
        std::unique_ptr<llaminar2::ROCmPeerPushResources> owner;
        unsigned long long *metadata = nullptr;
        /** @brief Retire retained graphs before releasing receiver-local physical storage. */
        ~PushStorage()
        {
            if (device < 0) return;
            (void)hipSetDevice(device);
            (void)hipStreamSynchronize(stream);
            owner.reset();
            if (metadata) (void)hipFree(metadata);
        }
    };
    std::shared_ptr<llaminar2::PhysicalMemoryAuthority> pushAuthority;
    std::vector<PushStorage> pushStorage(tinyPush?ranks:0);
    std::vector<llaminar2::PeerPushBinding> pushBindings(tinyPush?ranks:0);
    if (tinyPush)
    {
        std::size_t residency=SIZE_MAX;
        for (const auto &d:devices_) {
            ASSERT_EQ(hipSetDevice(d.ordinal),hipSuccess);
            std::size_t capacity=0;
            ASSERT_TRUE(llaminar2::rocmPeerPushResidentCapacity(capacity));
            residency=std::min(residency,capacity);
        }
        const auto layout=llaminar2::PeerPushLayout::create(262144,ranks,residency);
        const auto bytes=layout.inboxBytes()+layout.flagBytes()+layout.endpointBytes();
        llaminar2::PhysicalMemoryPlanBuilder plan;
        for (const auto &d:devices_) {
            ASSERT_EQ(hipSetDevice(d.ordinal),hipSuccess);
            std::size_t available=0,total=0;
            ASSERT_EQ(hipMemGetInfo(&available,&total),hipSuccess);
            plan.add({0,llaminar2::DeviceId::rocm(d.ordinal),total,available},
                llaminar2::PhysicalMemoryOwner::LocalCollective,bytes);
        }
        pushAuthority=std::make_shared<llaminar2::PhysicalMemoryAuthority>(
            std::make_shared<llaminar2::PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
        std::vector<llaminar2::PeerPushEndpoint> endpoints(ranks);
        for (int rank=0;rank<ranks;++rank)
        {
            auto &storage=pushStorage[rank];
            storage.device=devices_[rank].ordinal; storage.stream=devices_[rank].stream;
            ASSERT_EQ(hipSetDevice(storage.device),hipSuccess);
            ASSERT_EQ(hipMalloc(&storage.metadata,sizeof(unsigned long long)),hipSuccess);
            ncclLlaminarCompletionAuthority original{};
            ASSERT_EQ(rccl::nativeCompletionAuthority(comms_[rank],&original),rccl::ncclSuccess);
            const auto device=llaminar2::DeviceId::rocm(storage.device);
            if (!timing && rank==0) {
                auto tooSmall=pushAuthority->reserveNewAllocations(device,llaminar2::PhysicalMemoryOwner::LocalCollective,bytes-1);
                EXPECT_THROW(llaminar2::ROCmPeerPushResources::create(device,tooSmall,layout,rank,
                    original.abortFlag,storage.stream),std::logic_error);
                EXPECT_EQ(tooSmall.materializedBytes(),0u) << "Partial construction leaked its admitted slab";
            }
            auto reserve=pushAuthority->reserveNewAllocations(device,llaminar2::PhysicalMemoryOwner::LocalCollective,bytes);
            storage.owner=llaminar2::ROCmPeerPushResources::create(device,reserve,layout,rank,original.abortFlag,storage.stream);
            EXPECT_EQ(reserve.materializedBytes(),bytes);
            EXPECT_THROW(storage.owner->binding(),std::logic_error);
            endpoints[rank]=storage.owner->endpoint();
        }
        for (int rank=0;rank<ranks;++rank) {
            auto &storage=pushStorage[rank];
            ASSERT_EQ(hipSetDevice(storage.device),hipSuccess);
            storage.owner->publish(endpoints,storage.stream);
            pushBindings[rank]=storage.owner->binding();
            EXPECT_THROW(storage.owner->publish(endpoints,storage.stream),std::logic_error);
        }
        std::printf("PEER_PUSH_STORAGE,ranks=%d,bank_elements=%zu,blocks=%zu,resident_blocks=%zu,inbox_bytes=%zu,flag_bytes=%zu,table_bytes=%zu\n",
            ranks,layout.bank_elements,layout.blocks,residency,layout.inboxBytes(),layout.flagBytes(),layout.endpointBytes());
        EXPECT_FALSE(llaminar2::launchROCmPeerPushFP32(pushBindings[0],devices_[0].d_buffer,devices_[0].d_buffer+1,
            nullptr,1,33,nullptr,nullptr,devices_[0].stream));
        EXPECT_FALSE(llaminar2::launchROCmPeerPushFP32(pushBindings[0],devices_[0].d_buffer,devices_[0].d_buffer,
            nullptr,1,33,nullptr,nullptr,nullptr));
    }
    const int operations = timing || aborting ? 1 : 2;
    const int transactions = timing ? 97 : 1;
    const int arrivalTransactions = timing ? 1 : 0;
    const int capturedTransactions = transactions + arrivalTransactions;
    for (const auto [capacity, width, counted] : shapes)
    {
        const size_t elements = size_t(capacity) * width;
        const size_t laneElements = size_t(ranks + 2) * elements;
        const size_t countOffset = size_t(operations) * laneElements;
        const size_t receiptOffset = (countOffset + 3) & ~size_t(1);
        // Separate FP16 operands expose accidental changes to ordinary native
        // scheduling while the ordered operation borrows the same communicator.
        // Each prefill stage keeps its own submission, as in the production
        // graph. Native FP16 grouping itself may change channel partitions.
        constexpr size_t prefillElements=32768;
        const size_t prefillOffset=(receiptOffset+2)*sizeof(float);
        const size_t prefillBankBytes=prefillElements*sizeof(std::uint16_t);
        constexpr size_t addressedElements=32;
        constexpr size_t addressedBytes=addressedElements*sizeof(std::uint16_t);
        ASSERT_LE(prefillOffset+(mixedPrefill?3*prefillBankBytes+addressedBytes:0),MAX_BUFFER_BYTES);
        for (const bool inPlace : {false, true})
        {
            if ((timing || aborting) && inPlace) continue;
            // A/B/B/A includes both launch orders without changing the work.
            const std::vector<bool> arms = aborting?std::vector<bool>{true}:
                profiling?std::vector<bool>{mode==RankOrderedReplayMode::FusedProfile}:
                !candidate?std::vector<bool>{false}:
                timing?std::vector<bool>{false,true,true,false}:std::vector<bool>{false,true};
            std::vector<std::vector<std::vector<std::uint16_t>>> prefillControl;
            for (size_t arm = 0; arm < arms.size(); ++arm)
            {
                const bool fused = arms[arm];
                std::vector<GraphOwner> graphs(ranks);
                std::vector<hipError_t> status(ranks, hipSuccess);
                std::vector<rccl::ncclResult_t> native(ranks, rccl::ncclSuccess);
                std::vector<int> folded(ranks, 1);
                std::barrier entered(ranks), recorded(ranks);
                std::vector<std::thread> builders;
                for (int rank = 0; rank < ranks; ++rank)
                {
                    graphs[rank].device=devices_[rank].ordinal;
                    graphs[rank].stream=devices_[rank].stream;
                    builders.emplace_back([&,rank] {
                        auto &d=devices_[rank];
                        status[rank]=hipSetDevice(d.ordinal);
                        if (status[rank]==hipSuccess)
                            status[rank]=hipStreamBeginCapture(d.stream, hipStreamCaptureModeRelaxed);
                        const bool capturing=status[rank]==hipSuccess;
                        entered.arrive_and_wait();
                        auto *count=reinterpret_cast<std::int32_t *>(d.d_buffer+countOffset);
                        auto *receipt=reinterpret_cast<unsigned long long *>(d.d_buffer+receiptOffset);
                        const auto rows=counted ? llaminar2::NativeCollectiveRows(
                            llaminar2::DeviceRowRange::deviceCounted(capacity,count),width).withPayloadReceipt(receipt) :
                            llaminar2::NativeCollectiveRows(llaminar2::DeviceRowRange::fullyActive(capacity),width);
                        if (capturing)
                            for (int iteration=0; iteration<capturedTransactions &&
                                status[rank]==hipSuccess && native[rank]==rccl::ncclSuccess; ++iteration)
                            {
                                auto *prefill=reinterpret_cast<unsigned char*>(d.d_buffer)+prefillOffset;
                                if (mixedPrefill && native[rank]==rccl::ncclSuccess)
                                    native[rank]=rccl::ncclAllReduce(prefill,prefill+prefillBankBytes,
                                        prefillElements,rccl::ncclHalf,rccl::ncclSum,comms_[rank],d.stream);
                                if (native[rank]==rccl::ncclSuccess && !(tinyPush && fused))
                                    native[rank]=rccl::ncclGroupStart();
                                const bool groupOpened=native[rank]==rccl::ncclSuccess && !(tinyPush && fused);
                                for (int operation=0; operation<operations && native[rank]==rccl::ncclSuccess; ++operation)
                                {
                                    auto *input=d.d_buffer+size_t(operation)*laneElements;
                                    auto *banks=input+elements;
                                    auto *output=inPlace?input:banks+size_t(ranks)*elements;
                                    if (tinyPush && fused)
                                        folded[rank] &= llaminar2::launchROCmPeerPushFP32(pushBindings[rank],input,output,
                                            counted?count:nullptr,capacity,width,counted?receipt:nullptr,
                                            pushStorage[rank].metadata,d.stream);
                                    else native[rank]=
                                        rccl::nativeRows(llaminar2::NativeRowCollective::AllGather,input,banks,rows,
                                            rccl::ncclFloat,rccl::ncclSum,nullptr,comms_[rank],d.stream);
                                }
                                // Addressed native work must not be packed into the
                                // ordered kernel's full-block reduction program.
                                if (mixedPrefill && tinyPush && fused && native[rank]==rccl::ncclSuccess)
                                    native[rank]=rccl::ncclGroupStart();
                                const bool addressedGroupOpened=mixedPrefill && tinyPush && fused && native[rank]==rccl::ncclSuccess;
                                if (mixedPrefill && native[rank]==rccl::ncclSuccess)
                                    native[rank]=rccl::ncclSend(prefill,addressedElements,rccl::ncclHalf,
                                        (rank+1)%ranks,comms_[rank],d.stream);
                                if (mixedPrefill && native[rank]==rccl::ncclSuccess)
                                    native[rank]=rccl::ncclRecv(prefill+3*prefillBankBytes,addressedElements,rccl::ncclHalf,
                                        (rank+ranks-1)%ranks,comms_[rank],d.stream);
                                const auto closed=(groupOpened || addressedGroupOpened)?rccl::ncclGroupEnd():native[rank];
                                if (native[rank]==rccl::ncclSuccess) native[rank]=closed;
                                if (mixedPrefill && native[rank]==rccl::ncclSuccess)
                                    native[rank]=rccl::ncclAllReduce(prefill,prefill+2*prefillBankBytes,
                                        prefillElements,rccl::ncclHalf,rccl::ncclSum,comms_[rank],d.stream);
                                if (!fused && native[rank]==rccl::ncclSuccess)
                                    for (int operation=0; operation<operations; ++operation)
                                    {
                                        auto *input=d.d_buffer+size_t(operation)*laneElements;
                                        auto *banks=input+elements;
                                        auto *output=inPlace?input:banks+size_t(ranks)*elements;
                                        folded[rank] &= llaminar2::launchROCmTPRankOrderedSumFP32(
                                            banks,output,rows,ranks,d.ordinal,d.stream);
                                    }
                                // An external event is a real retained record
                                // node. Ordinary capture events may express only
                                // dependencies and cannot certify elapsed time.
                                if (timing && iteration+1==arrivalTransactions &&
                                    native[rank]==rccl::ncclSuccess && folded[rank])
                                    status[rank]=hipEventRecordWithFlags(d.start_event,
                                        d.stream,hipEventRecordExternal);
                            }
                        if (capturing && timing && status[rank]==hipSuccess &&
                            native[rank]==rccl::ncclSuccess && folded[rank])
                            status[rank]=hipEventRecordWithFlags(d.stop_event,
                                d.stream,hipEventRecordExternal);
                        recorded.arrive_and_wait();
                        if (capturing) {
                            const auto ended=hipStreamEndCapture(d.stream,&graphs[rank].graph);
                            if (status[rank]==hipSuccess) status[rank]=ended;
                        }
                        if (status[rank]==hipSuccess && native[rank]==rccl::ncclSuccess && folded[rank])
                            status[rank]=hipGraphInstantiate(&graphs[rank].executable,graphs[rank].graph,nullptr,nullptr,0);
                    });
                }
                for (auto &thread:builders) thread.join();
                for (int rank=0; rank<ranks; ++rank)
                {
                    ASSERT_EQ(status[rank],hipSuccess) << "rank=" << rank;
                    ASSERT_EQ(native[rank],rccl::ncclSuccess) << "fused=" << fused << " rank=" << rank;
                    ASSERT_TRUE(folded[rank]); ASSERT_NE(graphs[rank].executable,nullptr);
                    size_t nodes=0;
                    ASSERT_EQ(hipGraphGetNodes(graphs[rank].graph,nullptr,&nodes),hipSuccess);
                    if (timing) {
                        std::vector<hipGraphNode_t> retainedNodes(nodes);
                        ASSERT_EQ(hipGraphGetNodes(graphs[rank].graph,retainedNodes.data(),&nodes),hipSuccess);
                        int starts=0,stops=0;
                        for (const auto node:retainedNodes) {
                            hipGraphNodeType type;
                            ASSERT_EQ(hipGraphNodeGetType(node,&type),hipSuccess);
                            if (type!=hipGraphNodeTypeEventRecord) continue;
                            hipEvent_t event=nullptr;
                            ASSERT_EQ(hipGraphEventRecordNodeGetEvent(node,&event),hipSuccess);
                            starts+=event==devices_[rank].start_event;
                            stops+=event==devices_[rank].stop_event;
                        }
                        ASSERT_EQ(starts,1);
                        ASSERT_EQ(stops,1);
                    }
                    std::printf("ORDERED_GRAPH,fused=%d,rank=%d,M=%d,H=%d,nodes=%zu,transactions=%d,arrival_transactions=%d\n",
                        int(fused),rank,capacity,width,nodes,transactions,arrivalTransactions);
                }
                // Profiling attributes one shape/arm only. Its event times are
                // deliberately excluded from the unprofiled timing records.
                const int repeats=aborting?1:profiling?4:timing?36:20;
                if (mixedPrefill && arm==0)
                    prefillControl.resize(repeats,std::vector<std::vector<std::uint16_t>>(ranks));
                std::vector<double> samples;
                for (int repeat=0; repeat<repeats; ++repeat)
                {
                    const std::array<int,10> liveCases{capacity,1,0,std::min(3,capacity),
                        std::min(64,capacity),std::max(0,capacity-1),std::min(7,capacity),
                        0,std::min(65,capacity),capacity};
                    const int live=(timing || !counted)?capacity:liveCases[repeat%liveCases.size()];
                    const size_t liveElements=size_t(live)*width;
                    std::vector<std::vector<float>> initial(ranks,std::vector<float>(countOffset,-12345.0f));
                    std::vector<std::vector<float>> expected(operations,std::vector<float>(elements));
                    for (int rank=0; rank<ranks; ++rank)
                        for (int operation=0; operation<operations; ++operation)
                            for (size_t column=0; column<elements; ++column)
                            {
                                // Rank-zero initialization and one FP32 addition per
                                // following rank preserve cancellation and negative zero.
                                const float value=llaminar2::test::peerPushReplayValue(
                                    rank,operation,column,timing?0:repeat);
                                initial[rank][size_t(operation)*laneElements+column]=value;
                                volatile float rounded=rank==0?value:expected[operation][column]+value;
                                expected[operation][column]=rounded;
                            }
                    for (int rank=0; rank<ranks; ++rank)
                    {
                        auto &d=devices_[rank];
                        ASSERT_EQ(hipSetDevice(d.ordinal),hipSuccess);
                        ASSERT_EQ(hipMemcpyAsync(d.d_buffer,initial[rank].data(),countOffset*sizeof(float),
                            hipMemcpyHostToDevice,d.stream),hipSuccess);
                        ASSERT_EQ(hipMemcpyAsync(d.d_buffer+countOffset,&live,sizeof(live),hipMemcpyHostToDevice,d.stream),hipSuccess);
                        ASSERT_EQ(hipMemsetAsync(d.d_buffer+receiptOffset,0,sizeof(unsigned long long),d.stream),hipSuccess);
                        if (tinyPush) ASSERT_EQ(hipMemsetAsync(pushStorage[rank].metadata,0,sizeof(unsigned long long),d.stream),hipSuccess);
                        std::vector<std::uint16_t> prefillInput;
                        if (mixedPrefill) {
                            prefillInput.resize(prefillElements);
                            // Exactly represented half values with cancellation:
                            // changes in a native FP16 reduction tree can change
                            // the answer even though the ordered FP32 fold cannot.
                            constexpr std::array<std::uint16_t,4> values{0x5400,0x2400,0xd400,0x2400};
                            for (size_t i=0;i<prefillElements;++i)
                                prefillInput[i]=values[(size_t(rank)+i+repeat)%values.size()];
                            auto *prefill=reinterpret_cast<unsigned char*>(d.d_buffer)+prefillOffset;
                            ASSERT_EQ(hipMemcpyAsync(prefill,prefillInput.data(),prefillBankBytes,
                                hipMemcpyHostToDevice,d.stream),hipSuccess);
                            ASSERT_EQ(hipMemsetAsync(prefill+prefillBankBytes,0xda,2*prefillBankBytes+addressedBytes,d.stream),hipSuccess);
                        }
                        // Input publication is outside the timed interval.
                        ASSERT_EQ(hipStreamSynchronize(d.stream),hipSuccess);
                    }
                    // Production participants submit concurrently. Serial host
                    // submission charges earlier GPUs for the host's remaining
                    // graph launches, confounding collective latency with a
                    // fixture-only peer arrival delay. Context selection and
                    // worker creation are outside the timed interval.
                    std::barrier ready(ranks);
                    std::vector<std::thread> submitters;
                    for (int rank=0; rank<ranks; ++rank)
                        submitters.emplace_back([&,rank] {
                            auto &d=devices_[rank];
                            status[rank]=hipSetDevice(d.ordinal);
                            ready.arrive_and_wait();
                            // Fault injection: peers own their recorded graphs
                            // but withhold launch, so rank zero must await them.
                            if (aborting && rank!=0) return;
                            if (!timing && status[rank]==hipSuccess)
                                status[rank]=hipEventRecord(d.start_event,d.stream);
                            if (status[rank]==hipSuccess)
                                status[rank]=hipGraphLaunch(graphs[rank].executable,d.stream);
                            if (!timing && status[rank]==hipSuccess)
                                status[rank]=hipEventRecord(d.stop_event,d.stream);
                        });
                    for (auto &thread:submitters) thread.join();
                    // A partially submitted family may already be waiting on a
                    // missing peer. Abort its native owners before a fatal test
                    // assertion invokes graph/communicator teardown.
                    if (std::any_of(status.begin(),status.end(),
                        [](hipError_t value) { return value!=hipSuccess; }))
                        for (const auto comm:comms_)
                            EXPECT_EQ(rccl::signalNativeCompletionAbort(comm),rccl::ncclSuccess);
                    for (int rank=0; rank<ranks; ++rank) ASSERT_EQ(status[rank],hipSuccess);
                    if (aborting) {
                        ASSERT_EQ(hipSetDevice(devices_[0].ordinal),hipSuccess);
                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                        const auto waiting=hipEventQuery(devices_[0].stop_event);
                        // Signal before asserting so even a failed observation
                        // cannot leave an outstanding graph trapping teardown.
                        for (const auto comm:comms_)
                            EXPECT_EQ(rccl::signalNativeCompletionAbort(comm),rccl::ncclSuccess);
                        ASSERT_TRUE(awaitRetirement());
                        ASSERT_EQ(waiting,hipErrorNotReady);
                        std::vector<float> actual(countOffset);
                        ASSERT_EQ(hipSetDevice(devices_[0].ordinal),hipSuccess);
                        ASSERT_EQ(hipMemcpy(actual.data(),devices_[0].d_buffer,countOffset*sizeof(float),
                            hipMemcpyDeviceToHost),hipSuccess);
                        const size_t output=size_t(ranks+1)*elements;
                        ASSERT_EQ(std::memcmp(actual.data()+output,initial[0].data()+output,
                            elements*sizeof(float)),0) << "Aborted exchange published an incomplete result";
                        return;
                    }
                    ASSERT_TRUE(awaitRetirement());
                    double maximum=0;
                    for (int rank=0; rank<ranks; ++rank)
                    {
                        auto &d=devices_[rank]; ASSERT_EQ(hipSetDevice(d.ordinal),hipSuccess);
                        ASSERT_EQ(hipEventSynchronize(d.stop_event),hipSuccess);
                        float ms=0; ASSERT_EQ(hipEventElapsedTime(&ms,d.start_event,d.stop_event),hipSuccess);
                        maximum=std::max(maximum,double(ms)*1000/transactions/operations);
                        std::vector<float> actual(countOffset);
                        unsigned long long transferred=0;
                        ASSERT_EQ(hipMemcpy(actual.data(),d.d_buffer,countOffset*sizeof(float),hipMemcpyDeviceToHost),hipSuccess);
                        ASSERT_EQ(hipMemcpy(&transferred,d.d_buffer+receiptOffset,sizeof(transferred),hipMemcpyDeviceToHost),hipSuccess);
                        ASSERT_EQ(transferred,counted?size_t(capturedTransactions)*operations*liveElements*sizeof(float)*(ranks-1):0);
                        if (tinyPush && fused) {
                            unsigned long long metadata=0;
                            ASSERT_EQ(hipMemcpy(&metadata,pushStorage[rank].metadata,sizeof(metadata),hipMemcpyDeviceToHost),hipSuccess);
                            ASSERT_EQ(metadata,size_t(capturedTransactions)*operations*llaminar2::PeerPushLayout::blocksFor(elements)*sizeof(std::uint64_t)*(ranks-1));
                        }
                        if (mixedPrefill) {
                            std::vector<std::uint16_t> actualPrefill(2*prefillElements);
                            const auto *prefill=reinterpret_cast<const unsigned char*>(d.d_buffer)+prefillOffset;
                            ASSERT_EQ(hipMemcpy(actualPrefill.data(),prefill+prefillBankBytes,
                                2*prefillBankBytes,hipMemcpyDeviceToHost),hipSuccess);
                            ASSERT_EQ(std::memcmp(actualPrefill.data(),actualPrefill.data()+prefillElements,
                                prefillBankBytes),0) << "Interleaving changed identical native FP16 reductions";
                            std::array<std::uint16_t,addressedElements> addressed{};
                            ASSERT_EQ(hipMemcpy(addressed.data(),prefill+3*prefillBankBytes,
                                addressedBytes,hipMemcpyDeviceToHost),hipSuccess);
                            constexpr std::array<std::uint16_t,4> routedValues{0x5400,0x2400,0xd400,0x2400};
                            for (size_t i=0; i<addressed.size(); ++i)
                                EXPECT_EQ(addressed[i],routedValues[(size_t((rank+ranks-1)%ranks)+i+repeat)%routedValues.size()]);
                            if (arm==0) prefillControl[repeat][rank]=actualPrefill;
                            else ASSERT_EQ(actualPrefill,prefillControl[repeat][rank])
                                << "Ordered schedule changed native FP16 partitions rank=" << rank
                                << " repeat=" << repeat;
                        }
                        for (int operation=0; operation<operations; ++operation)
                        {
                            const size_t base=size_t(operation)*laneElements;
                            const size_t output=base+(inPlace?0:size_t(ranks+1)*elements);
                            ASSERT_EQ(std::memcmp(actual.data()+output,expected[operation].data(),liveElements*sizeof(float)),0)
                                << "fused=" << fused << " rank=" << rank << " operation=" << operation
                                << " M=" << capacity << " live=" << live << " repeat=" << repeat;
                            ASSERT_EQ(std::memcmp(actual.data()+output+liveElements,initial[rank].data()+output+liveElements,
                                (elements-liveElements)*sizeof(float)),0) << "Inactive output changed";
                            if (!inPlace)
                                ASSERT_EQ(std::memcmp(actual.data()+base,initial[rank].data()+base,elements*sizeof(float)),0);
                            for (int source=0; source<ranks; ++source)
                            {
                                const size_t bank=base+elements+size_t(source)*elements;
                                const auto *expectedBank=tinyPush && fused?initial[rank].data()+bank:initial[source].data()+base;
                                ASSERT_EQ(std::memcmp(actual.data()+bank,expectedBank,liveElements*sizeof(float)),0);
                                ASSERT_EQ(std::memcmp(actual.data()+bank+liveElements,initial[rank].data()+bank+liveElements,
                                    (elements-liveElements)*sizeof(float)),0) << "Inactive scratch changed";
                            }
                        }
                    }
                    if (timing && repeat>=5)
                    {
                        samples.push_back(maximum);
                        std::printf("ORDERED_SAMPLE,fused=%d,arm=%zu,M=%d,H=%d,sample=%d,us=%.9f\n",
                            int(fused),arm,capacity,width,repeat-5,maximum);
                    }
                }
                if (timing && !profiling)
                {
                    const auto stats=computeStats(samples,elements*sizeof(float),"native ordered paired");
                    std::printf("ORDERED_SUMMARY,fused=%d,arm=%zu,ranks=%d,M=%d,H=%d,median_us=%.9f,p5_us=%.9f,p95_us=%.9f,exact=1\n",
                        int(fused),arm,ranks,capacity,width,stats.median_us,stats.p5_us,stats.p95_us);
                }
            }
        }
    }
}

/** @test Captured small-push reset, partial/empty, changing grids and repeated exact outputs. */
TEST_F(Perf__RCCLAllreduceLatency, PeerPushCapturedReplay)
{ runRankOrderedReplay(RankOrderedReplayMode::Functional,0,true); }

/** @test Small push and ordinary native FP16/P2P work retain their stream and arithmetic contracts. */
TEST_F(Perf__RCCLAllreduceLatency, PeerPushCapturedMixedPrefill)
{ runRankOrderedReplay(RankOrderedReplayMode::MixedPrefill,0,true); }

/** @test The original communicator's abort releases a missing-peer small push without publication. */
TEST_F(Perf__RCCLAllreduceLatency, PeerPushCapturedAbort)
{ runRankOrderedReplay(RankOrderedReplayMode::Abort,0,true); }

/** @test Matched A/B/B/A row and width sweep, with exact bytes and bounded protocol receipts. */
TEST_F(Perf__RCCLAllreduceLatency, PeerPushCapturedTiming)
{ runRankOrderedReplay(RankOrderedReplayMode::PairedTiming,0,true); }

/** @test One exact small-push kernel geometry for isolated resource/timeline profiling. */
TEST_F(Perf__RCCLAllreduceLatency, PeerPushProfileDecode)
{ runRankOrderedReplay(RankOrderedReplayMode::FusedProfile,1,true); }

/** @test One grouped-row geometry for isolated resource/timeline profiling. */
TEST_F(Perf__RCCLAllreduceLatency, PeerPushProfileGrouped)
{ runRankOrderedReplay(RankOrderedReplayMode::FusedProfile,16,true); }
