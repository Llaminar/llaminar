/**
 * @file CUDAExpertTierRectangularWeightContract.inc
 * @brief Actual all-format rectangular native repack, independent CPU oracle and dirty replay.
 *
 * Test-only full source preparation is an independent oracle. The production
 * transfer uses bounded encoded chunks directly between original subrectangles;
 * no floating decode/requantization or CPU whole-expert assembly is involved.
 */
namespace llaminar2 { namespace {
TEST_F(CUDAExpertTierWeightKernelsTest, RectangularNativeMovementAllFormatsAndDirtyReplay)
{
    using P=MoEExpertProjectionOwnership::Projection;using R=MoEProjectionResidence;
    constexpr int N=70,K=512,targetN=95,targetK=256;
    const P originalSource{R::ParticipantOutputSlice,200,1024,11,N,32,K};
    const P originalTarget{R::ParticipantInputSlice,200,1024,34,targetN,128,targetK};
    const auto forward=ExpertTierWeightProjectionIntersection::between(originalSource,originalTarget);
    const auto backward=ExpertTierWeightProjectionIntersection::between(originalTarget,originalSource);
    TestCUDAStream transfer_stream;
    for(size_t format_index=0;format_index<test::quantizedVerifierFormats().size();++format_index) {
        const auto& format=test::quantizedVerifierFormats()[format_index];SCOPED_TRACE(format.label);
        auto source_tensor=format.create({N,K},static_cast<uint32_t>(30117+format_index));
        auto source=packProductionGpuProjection(*source_tensor);
        cpu::native_vnni::CPUNativeVNNIPackedWeights cpu;
        ASSERT_TRUE(cpu::native_vnni::packWeightsCPUNativeVNNI(source_tensor.get(),cpu));
        const auto* originalFormat=native_vnni_formats::forSourceIdentity(source.source_codebook_id,source.is_superblock);
        ASSERT_NE(originalFormat,nullptr);
        const auto demotionManifest=makeGpuToCpuExpertTierWeightStreamManifest(*originalFormat,N,K,81,0,7,ExpertTierWeightProjection::Down,2).forIntersection(forward);
        const auto down=demotionManifest.deviceLayout();
        auto cpuOracle=down;cpuOracle.cpu_N=N;cpuOracle.cpu_K=K;cpuOracle.cpu_first_row=23;cpuOracle.cpu_first_column=96;
        TestCUDABuffer<uint8_t> input(source.payload.size());TestCUDABuffer<uint16_t> scales(source.scales.size()),mins(source.mins.size());TestCUDABuffer<uint32_t> emins(source.emins.size());
        ASSERT_EQ(input.upload(source.payload),cudaSuccess);ASSERT_EQ(scales.upload(source.scales),cudaSuccess);
        ASSERT_EQ(mins.upload(source.mins),cudaSuccess);ASSERT_EQ(emins.upload(source.emins),cudaSuccess);
        const ExpertTierGpuConstProjectionView in{input.data(),scales.data(),mins.data(),emins.data(),input.bytes(),scales.bytes(),mins.bytes(),emins.bytes()};
        ASSERT_TRUE(in.validFor(down));
        cpu::native_vnni::CPUNativeVNNIPackedWeights destinationCPU;
        destinationCPU.N=targetN;destinationCPU.K=targetK;destinationCPU.N_padded=128;destinationCPU.blocks_per_row=targetK/32;
        destinationCPU.payload_bytes=cpu.payload_bytes;destinationCPU.codebook_id=cpu.codebook_id;destinationCPU.is_superblock=cpu.is_superblock;destinationCPU.is_asymmetric=cpu.is_asymmetric;
        destinationCPU.encoding=cpu.encoding;destinationCPU.data_stride=cpu.data_stride;destinationCPU.interleaved_block_stride=cpu.interleaved_block_stride;
        destinationCPU.native_interleaved.resize_uninitialized(down.cpuStorageBytes());
        std::span<uint8_t> finalCPU(destinationCPU.native_interleaved.data(),destinationCPU.native_interleaved.size());
        std::fill(finalCPU.begin(),finalCPU.end(),0xa5);
        const auto promotionManifest=makeCpuToGpuExpertTierWeightStreamManifest(destinationCPU,82,0,7,ExpertTierWeightProjection::Down,2).forIntersection(backward);
        const auto up=promotionManifest.deviceLayout();
        HostGpuExpertPackedProjection expected;
        ASSERT_TRUE(cpuToGpuExpertPackedReference(cpu,expected));
        TestCUDABuffer<uint8_t> output(expected.payload.size());TestCUDABuffer<uint16_t> outputScales(expected.scales.size()),outputMins(expected.mins.size());TestCUDABuffer<uint32_t> outputEmins(expected.emins.size());
        const ExpertTierGpuMutableProjectionView out{output.data(),outputScales.data(),outputMins.data(),outputEmins.data(),output.bytes(),outputScales.bytes(),outputMins.bytes(),outputEmins.bytes()};
        ASSERT_TRUE(out.validFor(up));
        TestCUDABuffer<uint8_t> chunk(std::max(down.chunkBytes(2),up.chunkBytes(2)));
        for(unsigned replay=0;replay<3;++replay) {
            const unsigned perChunk=replay==1?1:2;size_t bytesMoved=0;
            ASSERT_EQ(cudaMemsetAsync(output.data(),0xa5,output.bytes(),transfer_stream.native()),cudaSuccess);
            ASSERT_EQ(cudaMemsetAsync(outputScales.data(),0xa5,outputScales.bytes(),transfer_stream.native()),cudaSuccess);
            if(outputMins.bytes())ASSERT_EQ(cudaMemsetAsync(outputMins.data(),0xa5,outputMins.bytes(),transfer_stream.native()),cudaSuccess);
            if(outputEmins.bytes())ASSERT_EQ(cudaMemsetAsync(outputEmins.data(),0xa5,outputEmins.bytes(),transfer_stream.native()),cudaSuccess);
            std::fill(finalCPU.begin(),finalCPU.end(),0xa5);
            for(uint32_t first=0;first<down.unit_count;first+=perChunk) {
                const auto count=std::min(perChunk,down.unit_count-first),bytes=static_cast<uint32_t>(down.chunkBytes(count,first));
                std::vector<uint8_t> observed(bytes);
                ASSERT_TRUE(launchGpuToCpuExpertTierChunkCUDA(in,down,first,count,chunk.data(),chunk.bytes(),transfer_stream.opaque()));
                ASSERT_EQ(cudaMemcpyAsync(observed.data(),chunk.data(),bytes,cudaMemcpyDeviceToHost,transfer_stream.native()),cudaSuccess);
                ASSERT_EQ(transfer_stream.synchronize(),cudaSuccess);
                EXPECT_EQ(observed,test::expertTierWireOracle(cpuOracle,cpu.native_interleaved,first,count));
                ASSERT_TRUE(scatterExpertTierCPUWeightChunk(down,observed,first,count,finalCPU));
                // Same actual bounded bytes promote from the destination's own
                // local origins/strides, not the source's physical coordinates.
                const auto promoteBytes=test::expertTierWireOracle(up,finalCPU,first,count);
                ASSERT_EQ(cudaMemcpyAsync(chunk.data(),promoteBytes.data(),promoteBytes.size(),cudaMemcpyHostToDevice,transfer_stream.native()),cudaSuccess);
                ASSERT_TRUE(launchCpuToGpuExpertTierChunkCUDA(chunk.data(),chunk.bytes(),up,first,count,out,transfer_stream.opaque()));
                ASSERT_EQ(transfer_stream.synchronize(),cudaSuccess);bytesMoved+=bytes;
            }
            EXPECT_EQ(bytesMoved,static_cast<size_t>(47*(256/32))*(down.cpu_block_stride/64));
            std::vector<uint8_t> observed;std::vector<uint16_t> observedScales,observedMins;std::vector<uint32_t> observedEmins;
            ASSERT_EQ(output.download(observed),cudaSuccess);ASSERT_EQ(outputScales.download(observedScales),cudaSuccess);
            ASSERT_EQ(outputMins.download(observedMins),cudaSuccess);ASSERT_EQ(outputEmins.download(observedEmins),cudaSuccess);
            for(int kb=0;kb<K/32;++kb)for(int row=0;row<N;++row) {
                const size_t index=size_t(kb)*N+row;const bool active=row>=23 && kb>=3 && kb<11;
                for(unsigned byte=0;byte<expected.payload_bytes_per_block;++byte)
                    EXPECT_EQ(observed[index*expected.payload_bytes_per_block+byte],active?expected.payload[index*expected.payload_bytes_per_block+byte]:uint8_t(0xa5));
                // Compare the exact plane bytes (one byte per block for native MXFP4).
                const size_t scaleWidth=nativeVnniScalePlaneBytes(expected.codebook_id);
                for(size_t byte=0;byte<scaleWidth;++byte)
                    EXPECT_EQ(reinterpret_cast<const uint8_t*>(observedScales.data())[index*scaleWidth+byte],
                        active?reinterpret_cast<const uint8_t*>(expected.scales.data())[index*scaleWidth+byte]:uint8_t(0xa5));
                if(!observedMins.empty())EXPECT_EQ(observedMins[index],active?expected.mins[index]:uint16_t(0xa5a5));
                if(!observedEmins.empty())EXPECT_EQ(observedEmins[index],active?expected.emins[index]:uint32_t(0xa5a5a5a5));
            }
        }
    }
}
} }
