/**
 * @file ExpertTierCapturedPackedTransferContract.inc
 * @brief Existing retained-capture lane proof shared by CUDA service and HIP native SDMA.
 *
 * CUDA receipts prove graph-owned packed service bytes; HIP uses its original
 * explicit native-stream completion while the inference graph remains held.
 * Remote lane release is the service completion/owner boundary, NOT MPI send
 * acknowledgement. No transport or backend execution mechanism is invented.
 * Only DeviceBuffer/Success adapt the existing owning fixtures' native storage.
 */
        /** Both CPU edge frontends must consume the canonical transfer receipt. */
        enum class CapturedPackedClient { LocalTier, RemoteProjection };

        /**
         * @brief Prove every source format progresses under a captured future-reader fanout.
         * @tparam DeviceBuffer Owning backend fixture's genuine native allocation wrapper.
         * @tparam Success Owning backend's native success status.
         * @param device Exact compiled backend endpoint.
         * @param client Production local or remote CPU edge, never a kernel-only surrogate.
         * @param rectangle Whether to transfer the original47-row intersection across unequal roots.
         *
         * Every allocation, graph and event is prepared before the peer is held.
         * The held graph makes native future readers unable to service the
         * conversion; the graph-owned CUDA worker or explicit HIP native queue
         * must finish both directions anyway. Its receipt/event, not wall time
         * or PerfStats, releases each staging slice. Three rectangle repetitions
         * reuse the same retained body; the direct dirty-replay test separately
         * proves GPU source/destination reset. No MPI send ACK is fabricated.
         */
        template<template<typename> class DeviceBuffer, auto Success>
        void proveCapturedPackedTransfer(DeviceId device,CapturedPackedClient client,bool rectangle=false)
        {
            auto &context = GPUDeviceContextPool::instance().getContext(device);
            auto *backend = getBackendFor(device);
            ASSERT_NE(backend, nullptr);
            constexpr int N = 70;
            const int K=rectangle?512:256;
            constexpr int targetN=95,targetK=256;
            using Projection=MoEExpertProjectionOwnership::Projection;
            using Residence=MoEProjectionResidence;
            // Local edges exercise a nonzero GPU K origin; remote edges also
            // exercise a nonzero CPU K origin. Both retain the original global
            // coordinate declarations and independent endpoint root strides.
            const bool cpuKOrigin=client==CapturedPackedClient::RemoteProjection;
            const Projection originalSource{Residence::ParticipantOutputSlice,200,1024,11,N,cpuKOrigin?64:32,K};
            const Projection originalTarget{Residence::ParticipantInputSlice,200,1024,34,targetN,cpuKOrigin?32:128,targetK};
            constexpr std::uint32_t chunk_units = 3u;
            constexpr std::size_t readers = 32u;
            const DeviceId devices[] = {device};
            TransferEngine transfers;
            for (const auto &format : test::quantizedVerifierFormats())
            {
                SCOPED_TRACE(format.label);
                auto tensor = format.create({std::size_t(N),std::size_t(K)},91037u);
                auto source = packProductionGpuProjection(*tensor);
                cpu::native_vnni::CPUNativeVNNIPackedWeights cpu;
                ASSERT_TRUE(cpu::native_vnni::packWeightsCPUNativeVNNI(tensor.get(), cpu));
                std::string error;
                HostGpuExpertPackedProjection promoted;
                ASSERT_TRUE(cpuToGpuExpertPackedReference(cpu, promoted, &error)) << error;
                const auto *metadata = native_vnni_formats::forSourceIdentity(
                    source.source_codebook_id, source.is_superblock);
                ASSERT_NE(metadata, nullptr);
                auto downManifest=makeGpuToCpuExpertTierWeightStreamManifest(*metadata,
                    N,K,901,2,5,ExpertTierWeightProjection::Down,chunk_units);
                if(rectangle)downManifest=downManifest.forIntersection(
                    ExpertTierWeightProjectionIntersection::between(originalSource,originalTarget));
                const auto down=downManifest.deviceLayout();
                // Same genuine CPU final root geometry as the direct rectangle
                // contract; it is not a full-parent staging buffer.
                cpu::native_vnni::CPUNativeVNNIPackedWeights destinationCPU;
                if(rectangle) {
                    destinationCPU.N=targetN;destinationCPU.K=targetK;destinationCPU.N_padded=128;
                    destinationCPU.blocks_per_row=targetK/32;destinationCPU.payload_bytes=cpu.payload_bytes;
                    destinationCPU.codebook_id=cpu.codebook_id;destinationCPU.is_superblock=cpu.is_superblock;
                    destinationCPU.is_asymmetric=cpu.is_asymmetric;destinationCPU.encoding=cpu.encoding;
                    destinationCPU.data_stride=cpu.data_stride;destinationCPU.interleaved_block_stride=cpu.interleaved_block_stride;
                    destinationCPU.native_interleaved.resize_uninitialized(down.cpuStorageBytes());
                }
                auto upManifest=makeCpuToGpuExpertTierWeightStreamManifest(rectangle?destinationCPU:cpu,
                    902,2,5,ExpertTierWeightProjection::Down,chunk_units);
                if(rectangle)upManifest=upManifest.forIntersection(
                    ExpertTierWeightProjectionIntersection::between(originalTarget,originalSource));
                const auto up=upManifest.deviceLayout();
                auto sourceCPU=down;
                if(rectangle){sourceCPU.cpu_N=N;sourceCPU.cpu_K=K;sourceCPU.cpu_first_row=down.gpu_first_row;sourceCPU.cpu_first_column=down.gpu_first_column;}
                std::vector<uint8_t> expectedCPU(down.cpuStorageBytes(),0xa5);
                std::size_t oneWayBytes=0,oneWayChunks=0;
                for(uint32_t first=0;first<down.unit_count;first+=chunk_units) {
                    const auto count=std::min(chunk_units,down.unit_count-first);
                    const auto wire=test::expertTierWireOracle(sourceCPU,cpu.native_interleaved,first,count);
                    ASSERT_EQ(wire.size(),down.chunkBytes(count,first));
                    ASSERT_TRUE(scatterExpertTierCPUWeightChunk(down,wire,first,count,expectedCPU));
                    oneWayBytes+=wire.size();++oneWayChunks;
                }
                if(rectangle) {
                    ASSERT_EQ(down.N,47); // compact tail, not a padded64-column unit
                    EXPECT_EQ(oneWayBytes,std::size_t(47*(down.K/32))*(down.cpu_block_stride/64));
                    EXPECT_NE(down.gpu_N,down.cpu_N);
                    EXPECT_NE(down.gpu_K,down.cpu_K);
                    EXPECT_EQ(down.gpu_first_row,23);
                    if(cpuKOrigin)EXPECT_EQ(down.cpu_first_column,32);
                    else EXPECT_EQ(down.gpu_first_column,96);
                }

                DeviceBuffer<std::uint8_t> payload(source.payload.size()), output(promoted.payload.size());
                DeviceBuffer<std::uint16_t> scales(source.scales.size()), mins(source.mins.size()),
                    output_scales(promoted.scales.size()), output_mins(promoted.mins.size());
                DeviceBuffer<std::uint32_t> emins(source.emins.size()), output_emins(promoted.emins.size());
                ASSERT_EQ(payload.upload(source.payload), Success);
                ASSERT_EQ(scales.upload(source.scales), Success);
                ASSERT_EQ(mins.upload(source.mins), Success);
                ASSERT_EQ(emins.upload(source.emins), Success);
                const ExpertTierGpuConstProjectionView source_view{
                    payload.data(), scales.data(), mins.data(), emins.data(),
                    payload.bytes(), scales.bytes(), mins.bytes(), emins.bytes()};
                ASSERT_EQ(output.upload(std::vector<uint8_t>(promoted.payload.size(),0xa5)),Success);
                ASSERT_EQ(output_scales.upload(std::vector<uint16_t>(promoted.scales.size(),0xa5a5)),Success);
                ASSERT_EQ(output_mins.upload(std::vector<uint16_t>(promoted.mins.size(),0xa5a5)),Success);
                ASSERT_EQ(output_emins.upload(std::vector<uint32_t>(promoted.emins.size(),0xa5a5a5a5)),Success);
                const ExpertTierGpuMutableProjectionView destination_view{
                    output.data(), output_scales.data(), output_mins.data(), output_emins.data(),
                    output.bytes(), output_scales.bytes(), output_mins.bytes(), output_emins.bytes()};
                const auto capacity = std::max(down.chunkBytes(chunk_units), up.chunkBytes(chunk_units));
                auto execution = transfers.allocatePersistentTransferExecutionLanes(
                    1u, device, "captured_packed_transfer");
                std::shared_ptr<MappedTransferProgressEpoch> epoch;
                if(device.is_cuda())epoch = MappedTransferProgressEpoch::create({
                    .device = device, .slot_capacity = 2u, .execution_lane_capacity = 1u,
                    .execution_streams = execution, .maximum_bytes = capacity,
                    .name = "captured_packed_transfer", .perf_device = device.toString()});
                auto staging = transfers.allocatePersistentTransferStagingSlices(capacity, 1u, device).front();
                std::unique_ptr<ExpertTierWeightTransferLane> local;
                std::unique_ptr<MoEOverlayGpuRemoteProjectionLane> remote;
                if (client == CapturedPackedClient::LocalTier)
                {
                    local = std::make_unique<ExpertTierWeightTransferLane>(ExpertTierWeightTransferLane::Config{
                        .device = device, .staging = staging, .execution = execution.front(),
                        .progress = epoch?BackgroundTransferProgressBinding::graphService(epoch):BackgroundTransferProgressBinding::nativeStream(),
                        .lane_name = "captured_packed_local", .perf_device = device.toString()});
                    ASSERT_TRUE(local->materialize(&error)) << error;
                }
                else
                {
                    remote = std::make_unique<MoEOverlayGpuRemoteProjectionLane>(MoEOverlayGpuRemoteProjectionLane::Config{
                        .device = device, .staging = staging, .execution = execution.front(),
                        .progress = epoch?BackgroundTransferProgressBinding::graphService(epoch):BackgroundTransferProgressBinding::nativeStream(),
                        .lane_name = "captured_packed_remote", .perf_device = device.toString()});
                    ASSERT_TRUE(remote->materialize(&error)) << error;
                }
                auto control = transfers.allocateMappedHostRegion((readers + 1u) * 64u, devices);
                std::memset(control->mutableHostData(), 0, control->sizeBytes());
                void *stream = nullptr, *terminal = nullptr;
                std::vector<void *> observers;
                std::unique_ptr<IGPUGraphCapture> graph;
                std::unique_ptr<IGraphCaptureAuxiliaryBranch> branch;
                if(epoch)branch=epoch->graphBranchFactory().create();
                context.submitAndWait([&] {
                    stream = context.getOrCreateAuxiliaryStream("captured_packed_inference");
                    terminal = context.createEvent();
                    for (std::size_t index = 0; index < readers; ++index)
                        observers.push_back(context.getOrCreateAuxiliaryStream(
                            "captured_packed_reader_" + std::to_string(index)));
                    graph = context.createGraphCapture(stream);
                    if (!stream || !terminal || !graph)
                        throw std::runtime_error("Captured packed proof has incomplete setup");
                    ScopedBackendGraphCapture capture(context, *graph, "captured packed progress");
                    if (!capture.begin()) throw std::runtime_error("Captured packed proof did not begin");
                    transfers.enqueueMappedTimelineWait64(*control, 0u, 1u, device, stream);
                    capture.finish();
                    if ((branch && !branch->attach(*graph)) || !graph->instantiate())
                        throw std::runtime_error("Captured packed service did not attach");
                });

                bool remote_acquired = false;
                // Release the peer before any teardown/drain, even on a failed
                // assertion. Pending commands retain their original leases.
                auto release = [&](void *) {
                    std::atomic_ref<std::uint64_t>(*static_cast<std::uint64_t *>(
                        control->mutableHostData())).store(1u, std::memory_order_release);
                    context.submitAndWait([&] {
                        context.synchronizeStream(stream);
                        for (auto *observer : observers) context.synchronizeStream(observer);
                    });
                    if (local && local->progress() == ExpertTierWeightTransferProgress::Pending)
                        EXPECT_EQ(pollLaneToCompletion(*local), ExpertTierWeightTransferProgress::Ready);
                    if (remote_acquired)
                    {
                        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                        while (remote->poll(remote.get(), &error) == MoEOverlayGpuRemoteLaneProgress::Pending &&
                               std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
                        EXPECT_TRUE(remote->release(remote.get(), &error)) << error;
                        remote_acquired = false;
                    }
                    context.submitAndWait([&] { graph.reset(); context.destroyEvent(terminal); });
                    branch.reset();
                };
                std::unique_ptr<void, decltype(release)> release_guard(control.get(), release);
                context.submitAndWait([&] {
                    if (!graph->launch() || !context.recordEventChecked(terminal, stream))
                        throw std::runtime_error("Captured packed proof did not launch");
                    for (std::size_t index = 0; index < readers; ++index)
                        if (!context.waitEventChecked(terminal, observers[index]) ||
                            !backend->copyDeviceVisibleRegionByKernelOnStream(
                                control->deviceAlias(device, (index + 1u) * 64u), payload.data(),
                                16u, 0, observers[index]))
                            throw std::runtime_error("Captured packed future-reader submission failed");
                });
                const unsigned repetitions=rectangle?3u:1u;
                std::vector<std::uint8_t> observed_cpu(down.cpuStorageBytes());
                for(unsigned replay=0;replay<repetitions;++replay) {
                SCOPED_TRACE(replay);
                std::fill(observed_cpu.begin(),observed_cpu.end(),0xa5);
                if (local)
                {
                    ASSERT_TRUE(local->startGpuToCpu(down, source_view, observed_cpu,
                        ExpertTierSourceReadiness::publishedResidencyBank(901), &error)) << error;
                    ASSERT_EQ(pollLaneToCompletion(*local), ExpertTierWeightTransferProgress::Ready);
                    EXPECT_EQ(observed_cpu,expectedCPU);
                    ASSERT_TRUE(local->startCpuToGpu(up, observed_cpu, destination_view, &error)) << error;
                    ASSERT_EQ(pollLaneToCompletion(*local), ExpertTierWeightTransferProgress::Ready);
                }
                else
                {
                    for (const bool to_cpu : {true, false})
                    {
                        const auto &layout = to_cpu ? down : up;
                        for (std::uint32_t first = 0; first < layout.unit_count; first += chunk_units)
                        {
                            const auto count = std::min(chunk_units, layout.unit_count - first);
                            const auto bytes = layout.chunkBytes(count,first);
                            const auto expected_wire=test::expertTierWireOracle(to_cpu?sourceCPU:up,
                                to_cpu?std::span<const uint8_t>(cpu.native_interleaved.data(),cpu.native_interleaved.size()):std::span<const uint8_t>(observed_cpu),first,count);
                            ASSERT_TRUE(remote->tryAcquire(remote.get()));
                            remote_acquired = true;
                            ASSERT_TRUE(remote->bindSourceReadiness(remote.get(),
                                ExpertTierSourceReadiness::publishedResidencyBank(901), &error)) << error;
                            if (to_cpu)
                            {
                                ASSERT_TRUE(remote->submitGpuToCpuRepack(remote.get(), layout,
                                    source_view, first, count, &error)) << error;
                            }
                            else
                            {
                                ASSERT_TRUE(remote->submitCpuToGpuRepack(remote.get(), layout,
                                    destination_view, first, count,
                                    std::span<const uint8_t>(expected_wire),
                                    &error)) << error;
                            }
                            auto state = MoEOverlayGpuRemoteLaneProgress::Pending;
                            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                            while (state == MoEOverlayGpuRemoteLaneProgress::Pending &&
                                   std::chrono::steady_clock::now() < deadline)
                            {
                                state = remote->poll(remote.get(), &error);
                                std::this_thread::yield();
                            }
                            ASSERT_EQ(state, MoEOverlayGpuRemoteLaneProgress::Ready) << error;
                            if (to_cpu)
                            {
                                const auto bytes_out = remote->pinnedOutput(remote.get(), bytes);
                                ASSERT_EQ(bytes_out.size(), bytes);
                                EXPECT_TRUE(remote->pinnedOutput(remote.get(),bytes+1u).empty());
                                EXPECT_FALSE(remote->tryAcquire(remote.get())); // completion still retains the staging owner
                                ASSERT_TRUE(scatterExpertTierCPUWeightChunk(down,bytes_out,first,count,observed_cpu));
                                EXPECT_TRUE(std::equal(bytes_out.begin(), bytes_out.end(),
                                    expected_wire.begin()));
                            }
                            ASSERT_TRUE(remote->release(remote.get(), &error)) << error;
                            remote_acquired = false;
                        }
                    }
                }
                } // repeated source/publication commands on the same retained captured body
                const auto expectedBytes=2u*repetitions*oneWayBytes;
                const auto expectedChunks=2u*repetitions*oneWayChunks;
                if(local) {
                    EXPECT_EQ(local->stats().bytes_submitted,expectedBytes);
                    EXPECT_EQ(local->stats().chunks_submitted,expectedChunks);
                    EXPECT_EQ(local->stats().transfers_completed,2u*repetitions);
                } else {
                    EXPECT_EQ(remote->stats().bytes_submitted,expectedBytes);
                    EXPECT_EQ(remote->stats().chunks_submitted,expectedChunks);
                    EXPECT_EQ(remote->stats().chunks_completed,expectedChunks);
                }
                if(epoch) {
                    EXPECT_EQ(epoch->stats().commands_published,expectedChunks);
                    EXPECT_EQ(epoch->stats().commands_completed,expectedChunks);
                    EXPECT_EQ(epoch->stats().bytes_completed,expectedBytes);
                }
                context.submitAndWait([&] {
                    bool ready=true;
                    ASSERT_TRUE(context.queryEventChecked(terminal,ready));
                    EXPECT_FALSE(ready) << "transfer must progress before the held inference terminal";
                });
                release_guard.reset();
                std::vector<std::uint8_t> actual_payload;
                std::vector<std::uint16_t> actual_scales, actual_mins;
                std::vector<std::uint32_t> actual_emins;
                ASSERT_EQ(output.download(actual_payload), Success);
                ASSERT_EQ(output_scales.download(actual_scales), Success);
                ASSERT_EQ(output_mins.download(actual_mins), Success);
                ASSERT_EQ(output_emins.download(actual_emins), Success);
                if(!rectangle) {
                    EXPECT_EQ(actual_payload,promoted.payload);EXPECT_EQ(actual_scales,promoted.scales);
                    EXPECT_EQ(actual_mins,promoted.mins);EXPECT_EQ(actual_emins,promoted.emins);
                } else {
                    EXPECT_EQ(observed_cpu,expectedCPU);
                    for(int kb=0;kb<K/32;++kb)for(int row=0;row<N;++row) {
                        const auto index=std::size_t(kb)*N+row;
                        const bool active=row>=up.gpu_first_row && row<up.gpu_first_row+up.N &&
                            kb>=up.gpu_first_column/32 && kb<(up.gpu_first_column+up.K)/32;
                        for(unsigned byte=0;byte<promoted.payload_bytes_per_block;++byte)
                            EXPECT_EQ(actual_payload[index*promoted.payload_bytes_per_block+byte],
                                active?promoted.payload[index*promoted.payload_bytes_per_block+byte]:uint8_t(0xa5));
                        EXPECT_EQ(actual_scales[index],active?promoted.scales[index]:uint16_t(0xa5a5));
                        if(!actual_mins.empty())EXPECT_EQ(actual_mins[index],active?promoted.mins[index]:uint16_t(0xa5a5));
                        if(!actual_emins.empty())EXPECT_EQ(actual_emins[index],active?promoted.emins[index]:uint32_t(0xa5a5a5a5));
                    }
                }
            }
        }
