/**
 * @file ExpertTierRectangularRemoteContract.inc
 * @brief Original CPU root gathering and remote final-root scatter through existing MPI endpoint contracts.
 *
 * Opaque native bytes and explicit protocol fixture ownership prove wire geometry,
 * ACK lifetime and mutation bounds. This creates no MPI communicator or GPU and
 * does not claim executable prepared-engine/source admission.
 */
namespace llaminar2::test {
TEST(Test__MoEOverlayDistributedResidencyProtocol, RectangularCpuGpuSourcesUseOriginalRootAndExistingSendStaging)
{
    using P=MoEExpertProjectionOwnership::Projection;using R=MoEProjectionResidence;
    auto fixture=makeTransaction();const auto cpu=cpuProjectionBytes();
    const P originalCPU{R::ParticipantOutputSlice,128,128,5,32,32,64};
    const P originalGPU{R::ParticipantInputSlice,128,128,17,50,64,64};
    const auto upIdentity=remoteProjectionIdentity(fixture.transaction,DeviceId::cpu(),DeviceId::cuda(0));
    const auto up=makeCpuToGpuExpertTierWeightStreamManifest(cpu,upIdentity.candidate_epoch,
        upIdentity.layer_idx,upIdentity.expert_id,upIdentity.projection,1)
        .forIntersection(ExpertTierWeightProjectionIntersection::between(originalCPU,originalGPU));
    const auto outgoing=makeMoEOverlayRemoteCpuProjectionManifest(upIdentity,up,4096);
    ASSERT_TRUE(outgoing.valid());
    std::array<uint8_t,MoEOverlayRemoteProjectionManifest::kWireBytes> packet{};
    ASSERT_TRUE(encodeMoEOverlayRemoteProjectionManifest(outgoing,packet));
    MoEOverlayRemoteProjectionManifest decoded;
    ASSERT_TRUE(decodeMoEOverlayRemoteProjectionManifest(packet,&decoded));
    EXPECT_EQ(decoded,outgoing);EXPECT_EQ(decoded.source_projection,originalCPU);EXPECT_EQ(decoded.destination_projection,originalGPU);
    const std::array<std::span<const uint8_t>,4> sourceRegions{
        std::span<const uint8_t>(cpu.native_interleaved.data(),cpu.native_interleaved.size()),{},{},{}};
    MoEOverlayHostRemoteProjectionSource sender(outgoing,sourceRegions,std::make_shared<int>(1));
    std::vector<uint8_t> staging(outgoing.maximum_chunk_bytes,0xa5);const auto untouchedStaging=staging;
    MoEOverlayRemoteProjectionChunkView chunk;std::string error;
    EXPECT_EQ(sender.pollNextChunk(&chunk,&error),MoEOverlayResidencyWaveProgress::Failed);
    EXPECT_EQ(staging,untouchedStaging);
    ASSERT_EQ(sender.pollNextChunk(&chunk,&error,staging),MoEOverlayResidencyWaveProgress::Ready) << error;
    EXPECT_EQ(chunk.payload.size(),size_t(20)*(up.cpu_block_stride/64));
    EXPECT_EQ(outgoing.total_bytes,chunk.payload.size());EXPECT_TRUE(chunk.header.final_chunk);
    EXPECT_EQ(chunk.payload.data(),staging.data());
    const std::vector<uint8_t> transmitted(chunk.payload.begin(),chunk.payload.end()),heldStage=staging;
    MoEOverlayRemoteProjectionChunkView forbidden;
    EXPECT_EQ(sender.pollNextChunk(&forbidden,&error,staging),MoEOverlayResidencyWaveProgress::Failed);
    EXPECT_EQ(staging,heldStage);
    ASSERT_TRUE(sender.acknowledgeChunkSent(chunk.header,&error));

    const auto downIdentity=remoteProjectionIdentity(fixture.transaction,DeviceId::cuda(0),DeviceId::cpu());
    const auto down=makeGpuToCpuExpertTierWeightStreamManifest(native_vnni_formats::Q4_0,50,64,
        downIdentity.expected_epoch,downIdentity.layer_idx,downIdentity.expert_id,downIdentity.projection,1)
        .forIntersection(ExpertTierWeightProjectionIntersection::between(originalGPU,originalCPU));
    const auto incoming=makeMoEOverlayRemoteCpuProjectionManifest(downIdentity,down,4096);
    std::vector<uint8_t> destination(cpu.native_interleaved.size(),0xa5);
    const std::array<std::span<uint8_t>,4> destinationRegions{std::span<uint8_t>(destination),{},{},{}};
    MoEOverlayHostRemoteProjectionDestination receiver(downIdentity,destinationRegions,std::make_shared<int>(2),incoming);
    ASSERT_TRUE(receiver.beginManifest(incoming,&error)) << error;
    auto header=chunk.header;header.manifest_hash=incoming.manifest_hash;
    header.payload_hash=expertTierWeightBytesHash(transmitted);
    auto bad=header;--bad.payload_bytes;const auto before=destination;
    EXPECT_EQ(receiver.beginChunk(bad,std::span(transmitted).first(transmitted.size()-1),&error),MoEOverlayResidencyWaveProgress::Failed);
    EXPECT_EQ(destination,before);
    ASSERT_EQ(receiver.beginChunk(header,transmitted,&error),MoEOverlayResidencyWaveProgress::Ready) << error;
    std::vector<uint8_t> observed(transmitted.size());
    ASSERT_TRUE(gatherExpertTierCPUWeightChunk(down.deviceLayout(),destination,0,1,observed));
    EXPECT_EQ(observed,transmitted);
    EXPECT_EQ(std::memcmp(destination.data(),before.data(),cpu.interleaved_block_stride),0)
        << "the nonintersecting original K block must remain untouched";
    auto changed=incoming;++changed.source_projection.first_row;changed.manifest_hash=changed.computedHash();
    EXPECT_FALSE(changed.valid());
}
} // namespace llaminar2::test
namespace llaminar2::test {
TEST(Test__MoEOverlayDistributedResidencyProtocol, FloatingRectanglesMapOriginalRowsWithoutHostAssembly)
{
    using P=MoEExpertProjectionOwnership::Projection;using R=MoEProjectionResidence;
    auto fixture=makeTransaction();
    const auto identity=remoteProjectionIdentity(fixture.transaction,DeviceId::cpu(),DeviceId::cpu());
    const P sourceProjection{R::ParticipantOutputSlice,32,32,2,19,1,23};
    const P destinationProjection{R::ParticipantInputSlice,32,32,8,17,7,17};
    const auto intersection=ExpertTierWeightProjectionIntersection::between(sourceProjection,destinationProjection);
    for(const auto type:{TensorType::FP16,TensorType::BF16,TensorType::FP32}) {
        SCOPED_TRACE(static_cast<int>(type));const size_t element=type==TensorType::FP32?4:2;
        std::vector<uint8_t> source(19*23*element),destination(17*17*element,0xa5);
        for(size_t byte=0;byte<source.size();++byte)source[byte]=static_cast<uint8_t>(byte*17+3);
        const ContiguousFloatingPointWeightDescriptor descriptor{source.data(),type,19,23,source.size()};
        const auto manifest=makeMoEOverlayRemoteFloatingProjectionManifest(identity,descriptor,23).forFloatingIntersection(intersection);
        ASSERT_TRUE(manifest.valid());EXPECT_EQ(manifest.N,13);EXPECT_EQ(manifest.K,17);
        EXPECT_EQ(manifest.N_padded,manifest.N);
        auto staleRows=manifest;staleRows.N_padded=sourceProjection.rows;
        staleRows.manifest_hash=staleRows.computedHash();
        EXPECT_FALSE(staleRows.valid()) << "original allocation rows are not floating wire rows";
        const std::array<std::span<const uint8_t>,4> input{std::span<const uint8_t>(source),{},{},{}};
        const std::array<std::span<uint8_t>,4> output{std::span<uint8_t>(destination),{},{},{}};
        MoEOverlayHostRemoteProjectionSource sender(manifest,input,std::make_shared<int>(1));
        MoEOverlayHostRemoteProjectionDestination receiver(identity,output,std::make_shared<int>(2),manifest);
        std::string error;ASSERT_TRUE(receiver.beginManifest(manifest,&error));size_t transferred=0;
        while(transferred<manifest.total_bytes) {
            MoEOverlayRemoteProjectionChunkView chunk;
            ASSERT_EQ(sender.pollNextChunk(&chunk,&error),MoEOverlayResidencyWaveProgress::Ready) << error;
            EXPECT_LE(chunk.payload.size(),17*element-chunk.header.region_offset%(17*element));
            ASSERT_EQ(receiver.beginChunk(chunk.header,chunk.payload,&error),MoEOverlayResidencyWaveProgress::Ready) << error;
            transferred+=chunk.payload.size();ASSERT_TRUE(sender.acknowledgeChunkSent(chunk.header,&error));
        }
        EXPECT_EQ(transferred,size_t(13*17)*element);
        for(size_t row=0;row<17;++row)for(size_t column=0;column<17;++column)for(size_t byte=0;byte<element;++byte)
            EXPECT_EQ(destination[(row*17+column)*element+byte],row<13?source[((row+6)*23+column+6)*element+byte]:uint8_t(0xa5));
    }
}
} // namespace llaminar2::test

namespace llaminar2::test {
TEST(Test__MoEOverlayDistributedResidencyProtocol, NativeCpuRectanglesKeepStagingUntilExactAckAndScatterOnlyLiveColumns)
{
    using P=MoEExpertProjectionOwnership::Projection;using R=MoEProjectionResidence;
    auto fixture=makeTransaction();
    const auto identity=remoteProjectionIdentity(fixture.transaction,DeviceId::cpu(),DeviceId(DeviceType::CPU,1));
    const auto intersection=ExpertTierWeightProjectionIntersection::between(
        P{R::ParticipantOutputSlice,256,256,13,128,32,96},
        P{R::ParticipantInputSlice,256,256,72,160,64,128});
    for(const auto& format:native_vnni_formats::kAllSourceFormats) {
        SCOPED_TRACE(format.quant_type);
        cpu::native_vnni::CPUNativeVNNIPackedWeights packed;
        packed.N=128;packed.K=96;packed.N_padded=128;packed.blocks_per_row=3;
        packed.payload_bytes=format.metadata->payload_bytes;
        packed.codebook_id=format.metadata->codebook_id;packed.is_superblock=format.metadata->is_superblock;
        packed.is_asymmetric=format.metadata->is_asymmetric;
        packed.encoding=cpu::native_vnni::preparedEncodingForSource(*format.metadata);
        packed.data_stride=cpu::native_vnni::preparedDataStride(packed.encoding);
        packed.interleaved_block_stride=cpu::native_vnni::preparedInterleavedBlockStride(packed.encoding,packed.is_asymmetric);
        packed.native_interleaved.resize_uninitialized(size_t(6)*packed.interleaved_block_stride);
        for(size_t i=0;i<packed.native_interleaved.size();++i)packed.native_interleaved[i]=uint8_t(i*37+11);
        const auto manifest=makeMoEOverlayRemoteCpuProjectionManifest(identity,packed,73).forCPUIntersection(intersection);
        ASSERT_TRUE(manifest.valid());EXPECT_EQ(manifest.total_bytes,size_t(69*2)*(packed.interleaved_block_stride/64));
        EXPECT_LT(manifest.total_bytes,packed.native_interleaved.size());
        std::array<uint8_t,MoEOverlayRemoteProjectionManifest::kWireBytes> packet{};
        ASSERT_TRUE(encodeMoEOverlayRemoteProjectionManifest(manifest,packet));
        MoEOverlayRemoteProjectionManifest restored;ASSERT_TRUE(decodeMoEOverlayRemoteProjectionManifest(packet,&restored));
        EXPECT_EQ(restored,manifest);
        const auto sourceLayout=remoteCpuProjectionStorageLayout(manifest,MoEOverlayRemoteCPUStorageEndpoint::Source);
        const auto destinationLayout=remoteCpuProjectionStorageLayout(manifest,MoEOverlayRemoteCPUStorageEndpoint::Destination);
        EXPECT_EQ(sourceLayout.cpu_first_row,59);EXPECT_EQ(sourceLayout.cpu_first_column,32);
        EXPECT_EQ(destinationLayout.cpu_first_row,0);EXPECT_EQ(destinationLayout.cpu_first_column,0);
        std::vector<uint8_t> destination(destinationLayout.cpuStorageBytes(),0xa5),staging(manifest.maximum_chunk_bytes,0xcc);
        const auto dirty=destination;
        const std::array<std::span<const uint8_t>,4> input{std::span<const uint8_t>(packed.native_interleaved.data(),packed.native_interleaved.size()),{},{},{}};
        const std::array<std::span<uint8_t>,4> output{std::span<uint8_t>(destination),{},{},{}};
        MoEOverlayHostRemoteProjectionSource sender(manifest,input,std::make_shared<int>(1));
        MoEOverlayHostRemoteProjectionDestination receiver(identity,output,std::make_shared<int>(2),manifest);
        std::string error;ASSERT_TRUE(receiver.beginManifest(manifest,&error)) << error;
        size_t transferred=0;uint32_t first=0;
        while(transferred<manifest.total_bytes) {
            MoEOverlayRemoteProjectionChunkView chunk;
            ASSERT_EQ(sender.pollNextChunk(&chunk,&error,staging),MoEOverlayResidencyWaveProgress::Ready) << error;
            const auto held=staging;const auto originalHeader=chunk.header;
            MoEOverlayRemoteProjectionChunkView forbidden;
            EXPECT_EQ(sender.pollNextChunk(&forbidden,&error,staging),MoEOverlayResidencyWaveProgress::Failed);
            EXPECT_EQ(staging,held);
            auto wrong=chunk.header;wrong.payload_hash^=1;
            EXPECT_FALSE(sender.acknowledgeChunkSent(wrong,&error));EXPECT_EQ(staging,held);
            const auto before=destination;auto truncated=chunk.header;--truncated.payload_bytes;
            EXPECT_EQ(receiver.beginChunk(truncated,chunk.payload.first(chunk.payload.size()-1),&error),MoEOverlayResidencyWaveProgress::Failed);
            EXPECT_EQ(destination,before);
            ASSERT_EQ(receiver.beginChunk(chunk.header,chunk.payload,&error),MoEOverlayResidencyWaveProgress::Ready) << error;
            const auto end=expertTierWireUnitAt(sourceLayout,transferred+chunk.payload.size());
            std::vector<uint8_t> observed(chunk.payload.size());
            ASSERT_TRUE(gatherExpertTierCPUWeightChunk(destinationLayout,destination,first,end-first,observed));
            EXPECT_TRUE(std::equal(observed.begin(),observed.end(),chunk.payload.begin()));
            transferred+=chunk.payload.size();first=end;
            ASSERT_TRUE(sender.acknowledgeChunkSent(originalHeader,&error));
        }
        EXPECT_EQ(transferred,manifest.total_bytes);EXPECT_EQ(first,sourceLayout.unit_count);
        // Entire nonintersecting final tiles/K blocks remain dirty, including padding.
        EXPECT_EQ(std::memcmp(destination.data()+size_t(2)*destinationLayout.cpu_block_stride,
            dirty.data()+size_t(2)*destinationLayout.cpu_block_stride,size_t(2)*destinationLayout.cpu_block_stride),0);
        auto changed=manifest;changed.cpu_wire_columns_per_unit=65;changed.manifest_hash=changed.computedHash();
        EXPECT_FALSE(changed.valid());
        EXPECT_THROW((void)makeMoEOverlayRemoteCpuProjectionManifest(identity,packed,1),std::invalid_argument);
    }
}
} // namespace llaminar2::test
