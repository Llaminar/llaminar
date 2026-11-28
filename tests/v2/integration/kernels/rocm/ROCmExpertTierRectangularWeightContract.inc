/**
 * @file ROCmExpertTierRectangularWeightContract.inc
 * @brief Actual all-format rectangular native repack, independent CPU oracle and dirty replay.
 *
 * Test-only full source preparation is an independent oracle. The production
 * transfer uses bounded encoded chunks directly between original subrectangles;
 * no floating decode/requantization or CPU whole-expert assembly is involved.
 */
namespace llaminar2 { namespace {
TEST_F(ROCmExpertTierWeightKernelsTest, RectangularNativeMovementAllFormatsAndDirtyReplay)
{
    using P=MoEExpertProjectionOwnership::Projection;using R=MoEProjectionResidence;
    constexpr int N=70,K=512,targetN=95,targetK=256;
    TestHIPStream transfer_stream;
    for(size_t format_index=0;format_index<test::quantizedVerifierFormats().size();++format_index) {
        const auto& format=test::quantizedVerifierFormats()[format_index];SCOPED_TRACE(format.label);
        auto source_tensor=format.create({N,K},static_cast<uint32_t>(30117+format_index));
        cpu::native_quantised::CPUNativeQuantisedPackedWeights cpu;
        ASSERT_TRUE(cpu::native_quantised::packWeightsCPUNativeQuantised(source_tensor.get(),cpu));
        const auto* originalFormat=native_quantised_formats::forSourceIdentity(format.source_codebook_id,format.source_is_superblock);
        ASSERT_NE(originalFormat,nullptr);
        const bool coded = cpu.superblockCoded();
        const P originalSource{R::ParticipantOutputSlice,200,1024,11,N,coded ? 0 : 32,K};
        const P originalTarget{R::ParticipantInputSlice,200,1024,34,targetN,coded ? 256 : 128,targetK};
        const auto forward=ExpertTierWeightProjectionIntersection::between(originalSource,originalTarget);
        const auto backward=ExpertTierWeightProjectionIntersection::between(originalTarget,originalSource);
        const auto demotionManifest=makeGpuToCpuExpertTierWeightStreamManifest(*originalFormat,N,K,81,0,7,ExpertTierWeightProjection::Down,2).forIntersection(forward);
        const auto down=demotionManifest.deviceLayout();
        auto cpuOracle=down;cpuOracle.cpu_N=N;cpuOracle.cpu_K=K;cpuOracle.cpu_first_row=23;
        cpuOracle.cpu_first_column=forward.sourceColumnOrigin();
        auto loaded = test::makeGpuPreparedGemm(source_tensor.get(), DeviceId::rocm(0));
        const auto in = sourceNativeTierView(*loaded.kernel, *originalFormat);
        ASSERT_TRUE(in.validFor(down));
        cpu::native_quantised::CPUNativeQuantisedPackedWeights destinationCPU;
        destinationCPU.N=targetN;destinationCPU.K=targetK;destinationCPU.N_padded=128;destinationCPU.blocks_per_row=targetK/32;
        destinationCPU.payload_bytes=cpu.payload_bytes;destinationCPU.codebook_id=cpu.codebook_id;destinationCPU.is_superblock=cpu.is_superblock;destinationCPU.is_asymmetric=cpu.is_asymmetric;
        destinationCPU.encoding=cpu.encoding;destinationCPU.data_stride=cpu.data_stride;destinationCPU.interleaved_block_stride=cpu.interleaved_block_stride;
        destinationCPU.native_interleaved.resize_uninitialized(down.cpuStorageBytes());
        auto finalCPU=destinationCPU.native_interleaved.unsealedGenerationBytes();
        std::fill(finalCPU.begin(),finalCPU.end(),0xa5);
        const auto promotionManifest=makeCpuToGpuExpertTierWeightStreamManifest(destinationCPU,82,0,7,ExpertTierWeightProjection::Down,2).forIntersection(backward);
        const auto up=promotionManifest.deviceLayout();
        HostGpuExpertPackedProjection expected;
        ASSERT_TRUE(cpuToGpuExpertPackedReference(cpu,expected));
        namespace sb = native_quantised_superblock;
        const auto primary_bytes = coded ? expected.codes.size() : expected.scaleLiveBytes();
        const auto primary_live = coded && primary_bytes ? primary_bytes - sb::kSuperblockCodePlanePadBytes : primary_bytes;
        const auto secondary_bytes = coded ? expected.superblocks.size() : expected.mins.size() * sizeof(uint16_t);
        TestHIPBuffer<uint8_t> output(expected.payload.size()), outputScales(primary_bytes), outputMins(secondary_bytes);
        TestHIPBuffer<uint32_t> outputEmins(expected.emins.size());
        const ExpertTierGpuMutableProjectionView out{output.data(), reinterpret_cast<uint16_t *>(primary_bytes ? outputScales.data() : output.data()),
            reinterpret_cast<uint16_t *>(outputMins.data()), outputEmins.data(), output.bytes(), primary_live, outputMins.bytes(),
            outputEmins.bytes(), expected.metadata_layout};
        ASSERT_TRUE(out.validFor(up));
        TestHIPBuffer<uint8_t> chunk(2 * std::max(down.cpuStorage().maximumWireUnitBytes(), up.cpuStorage().maximumWireUnitBytes()));
        for(unsigned replay=0;replay<3;++replay) {
            const unsigned perChunk=replay==1?1:2;size_t bytesMoved=0;
            ASSERT_EQ(hipMemsetAsync(output.data(),0xa5,output.bytes(),transfer_stream.native()),hipSuccess);
            if (primary_bytes)
                ASSERT_EQ(hipMemsetAsync(outputScales.data(),0xa5,outputScales.bytes(),transfer_stream.native()),hipSuccess);
            if(outputMins.bytes())ASSERT_EQ(hipMemsetAsync(outputMins.data(),0xa5,outputMins.bytes(),transfer_stream.native()),hipSuccess);
            if(outputEmins.bytes())ASSERT_EQ(hipMemsetAsync(outputEmins.data(),0xa5,outputEmins.bytes(),transfer_stream.native()),hipSuccess);
            std::fill(finalCPU.begin(),finalCPU.end(),0xa5);
            for(uint32_t first=0;first<down.unit_count;first+=perChunk) {
                const auto count=std::min(perChunk,down.unit_count-first),bytes=static_cast<uint32_t>(down.chunkBytes(count,first));
                std::vector<uint8_t> observed(bytes);
                ASSERT_TRUE(launchGpuToCpuExpertTierChunkROCm(in,down,first,count,chunk.data(),chunk.bytes(),transfer_stream.opaque()));
                ASSERT_EQ(hipMemcpyAsync(observed.data(),chunk.data(),bytes,hipMemcpyDeviceToHost,transfer_stream.native()),hipSuccess);
                ASSERT_EQ(transfer_stream.synchronize(),hipSuccess);
                EXPECT_EQ(observed,test::expertTierWireOracle(cpuOracle,cpu.native_interleaved,first,count));
                ASSERT_TRUE(scatterExpertTierCPUWeightChunk(down,observed,first,count,finalCPU));
                // Same actual bounded bytes promote from the destination's own
                // local origins/strides, not the source's physical coordinates.
                const auto promoteBytes=test::expertTierWireOracle(up,finalCPU,first,count);
                ASSERT_EQ(hipMemcpyAsync(chunk.data(),promoteBytes.data(),promoteBytes.size(),hipMemcpyHostToDevice,transfer_stream.native()),hipSuccess);
                ASSERT_TRUE(launchCpuToGpuExpertTierChunkROCm(chunk.data(),chunk.bytes(),up,first,count,out,transfer_stream.opaque()));
                ASSERT_EQ(transfer_stream.synchronize(),hipSuccess);bytesMoved+=bytes;
            }
            if (!coded)
                EXPECT_EQ(bytesMoved,static_cast<size_t>(47*(256/32))*(down.cpu_block_stride/64));
            else
            {
                const auto planes = expertTierCodedPlanes(expected.codebook_id);
                const auto header = sb::superblockFieldBytes(expected.codebook_id) * size_t(47);
                EXPECT_EQ(bytesMoved, 8 * (47 * expected.payload_bytes_per_block + planes.bytes(47)) + header);
            }
            std::vector<uint8_t> observed, observedScales, observedMins;std::vector<uint32_t> observedEmins;
            ASSERT_EQ(output.download(observed),hipSuccess);ASSERT_EQ(outputScales.download(observedScales),hipSuccess);
            ASSERT_EQ(outputMins.download(observedMins),hipSuccess);ASSERT_EQ(outputEmins.download(observedEmins),hipSuccess);
            for(int kb=0;kb<K/32;++kb)for(int row=0;row<N;++row) {
                const size_t index=size_t(kb)*N+row;const bool active=row>=23 && kb>=up.gpu_first_column/32 && kb<(up.gpu_first_column+up.K)/32;
                for(unsigned byte=0;byte<expected.payload_bytes_per_block;++byte)
                    EXPECT_EQ(observed[index*expected.payload_bytes_per_block+byte],active?expected.payload[index*expected.payload_bytes_per_block+byte]:uint8_t(0xa5));
                if (coded)
                {
                    const auto bits = sb::superblockCodePlaneBits(expected.codebook_id);
                    if (bits)
                    {
                        unsigned dirty = 0;
                        for (unsigned bit = 0; bit < bits; ++bit)
                            dirty |= ((0xa5u >> ((index * bits + bit) % 8)) & 1u) << bit;
                        EXPECT_EQ(sb::loadPackedBits(observedScales.data(), index * bits, bits), active
                            ? sb::loadPackedBits(expected.codes.data(), index * bits, bits) : dirty);
                    }
                    if (kb % 8 == 0)
                    {
                        const auto field = sb::superblockFieldBytes(expected.codebook_id);
                        const auto header = (size_t(kb / 8) * N + row) * field;
                        for (unsigned byte = 0; byte < field; ++byte)
                            EXPECT_EQ(observedMins[header + byte], active ? expected.superblocks[header + byte] : uint8_t(0xa5));
                    }
                }
                else
                {
                    const size_t width = nativeQuantisedScalePlaneBytes(expected.codebook_id);
                    for (unsigned byte = 0; byte < width; ++byte)
                        EXPECT_EQ(observedScales[index * width + byte], active
                            ? reinterpret_cast<const uint8_t *>(expected.scales.data())[index * width + byte] : uint8_t(0xa5));
                    if (!observedMins.empty())
                        for (unsigned byte = 0; byte < 2; ++byte)
                            EXPECT_EQ(observedMins[index * 2 + byte], active
                                ? reinterpret_cast<const uint8_t *>(expected.mins.data())[index * 2 + byte] : uint8_t(0xa5));
                }
                if(!observedEmins.empty())EXPECT_EQ(observedEmins[index],active?expected.emins[index]:uint32_t(0xa5a5a5a5));
            }
        }
    }
}
} }
