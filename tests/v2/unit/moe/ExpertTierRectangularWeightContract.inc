/**
 * @file ExpertTierRectangularWeightContract.inc
 * @brief Original-coordinate and live encoded-column transport contracts, without devices.
 *
 * Seed opaque native payload/metadata bits, including Q6_K shared planes. This
 * proves representation movement and untouched neighbours, not arithmetic or
 * hardware readiness. Existing native suites own actual repack qualification.
 */
namespace llaminar2 {
namespace {
/** @brief Independently inspect one logical native column/block, excluding other columns' bits. */
template<class Layout>
std::vector<uint8_t> encodedTierColumn(std::span<const uint8_t> root,
    const Layout& layout,size_t n,size_t kb)
{
    const size_t unit=((n/64)*(layout.cpu_K/32)+kb)*layout.cpu_block_stride,column=n%64;
    const bool q6=layout.cpu_encoding==cpu::native_vnni::CPUNativeVNNIEncoding::Q6KNativeDualScale;
    std::vector<uint8_t> result;
    for(size_t byte=0;byte<(q6?16:layout.cpu_data_stride/64);++byte)
        result.push_back(root[unit+(byte/4)*256+column*4+byte%4]);
    if(q6)for(size_t group=0;group<8;++group) {
        uint64_t low,high;const size_t offset=unit+1024+group*64+(column/16)*16;
        std::memcpy(&low,root.data()+offset,8);std::memcpy(&high,root.data()+offset+8,8);
        const auto bit=(column%16)*4;
        result.push_back(static_cast<uint8_t>(((low>>bit)&15)|(((high>>bit)&15)<<4)));
    }
    const bool compact=layout.cpu_encoding==cpu::native_vnni::CPUNativeVNNIEncoding::CompactMultiScale;
    for(size_t plane=0;plane<(compact?2:(layout.cpu_block_stride-layout.cpu_data_stride)/128);++plane)
        for(size_t byte=0;byte<2;++byte)result.push_back(root[unit+layout.cpu_data_stride+plane*128+column*2+byte]);
    if(compact)for(size_t byte=0;byte<4;++byte)result.push_back(root[unit+layout.cpu_data_stride+256+column*4+byte]);
    return result;
}
TEST(ExpertTierWeightStream, OriginalProjectionIntersectionsRetainIndependentCoordinates)
{
    using P=MoEExpertProjectionOwnership::Projection;using R=MoEProjectionResidence;
    const P source{R::ParticipantOutputSlice,256,256,13,128,32,96};
    const P destination{R::ParticipantInputSlice,256,256,72,160,64,128};
    const auto overlap=ExpertTierWeightProjectionIntersection::between(source,destination);
    EXPECT_EQ(overlap.rows(),69);EXPECT_EQ(overlap.columns(),64);
    EXPECT_EQ(overlap.sourceRowOrigin(),59);EXPECT_EQ(overlap.sourceColumnOrigin(),32);
    EXPECT_EQ(overlap.destinationRowOrigin(),0);EXPECT_EQ(overlap.destinationColumnOrigin(),0);
    EXPECT_EQ(overlap.source(),source);EXPECT_EQ(overlap.destination(),destination);
    // Floating coordinates have no quantized native-block restriction.
    const P floating{R::ExpertOwner,9,17,0,9,0,17};
    EXPECT_EQ(ExpertTierWeightProjectionIntersection::between(floating,floating).columns(),17);
    auto foreign=destination;++foreign.source_columns;
    EXPECT_THROW((void)ExpertTierWeightProjectionIntersection::between(source,foreign),std::invalid_argument);
    auto invalid=source;invalid.rows=256;
    EXPECT_THROW((void)ExpertTierWeightProjectionIntersection::between(invalid,destination),std::invalid_argument);
    auto disjoint=source;disjoint.first_row=200;disjoint.rows=32;
    EXPECT_TRUE(ExpertTierWeightProjectionIntersection::between(source,disjoint).empty());
    for(int population:{1,2,4,8})for(int member=0;member<population;++member) {
        const P whole{R::ExpertOwner,256,256,0,256,0,256};
        const P shard{R::ParticipantOutputSlice,256,256,member*(256/population),256/population,0,256};
        EXPECT_EQ(ExpertTierWeightProjectionIntersection::between(whole,shard).rows(),256/population);
    }
}
TEST(ExpertTierWeightStream, RectangularEncodedColumnsAllFormatsPreserveNeighboursAndLiveBytes)
{
    using P=MoEExpertProjectionOwnership::Projection;using R=MoEProjectionResidence;
    const auto intersection=ExpertTierWeightProjectionIntersection::between(
        P{R::ParticipantOutputSlice,256,256,13,128,32,96},
        P{R::ParticipantInputSlice,256,256,72,160,64,128});
    for(const auto& format:native_vnni_formats::kAllSourceFormats) {
        SCOPED_TRACE(format.quant_type);
        const auto full=makeGpuToCpuExpertTierWeightStreamManifest(*format.metadata,128,96,17,0,9,ExpertTierWeightProjection::Down,2);
        const auto manifest=full.forIntersection(intersection);
        const auto source_layout=manifest.deviceLayout();ASSERT_TRUE(source_layout.valid());
        // In this demotion contract CPU is the destination; use the same exact
        // scalar source layout to independently exercise the inverse native copy.
        auto cpu_source=source_layout;cpu_source.cpu_N=128;cpu_source.cpu_K=96;
        cpu_source.cpu_first_row=59;cpu_source.cpu_first_column=32;
        std::vector<uint8_t> source(cpu_source.cpuStorageBytes());
        for(size_t byte=0;byte<source.size();++byte)source[byte]=static_cast<uint8_t>(byte*37+11);
        std::vector<uint8_t> destination(source_layout.cpuStorageBytes(),0xa5);const auto untouched=destination;
        size_t transferred=0;
        for(uint32_t first=0;first<source_layout.unit_count;first+=2) {
            const auto count=std::min(2u,source_layout.unit_count-first);
            const auto bytes=source_layout.chunkBytes(count,first);
            std::vector<uint8_t> wire(bytes),returned(bytes);
            ASSERT_TRUE(gatherExpertTierCPUWeightChunk(cpu_source,source,first,count,wire));
            ASSERT_TRUE(scatterExpertTierCPUWeightChunk(source_layout,wire,first,count,destination));
            ASSERT_TRUE(gatherExpertTierCPUWeightChunk(source_layout,destination,first,count,returned));
            EXPECT_EQ(returned,wire);transferred+=bytes;
            const auto before=destination;
            EXPECT_FALSE(scatterExpertTierCPUWeightChunk(source_layout,std::span(wire).first(bytes-1),first,count,destination));
            EXPECT_EQ(destination,before);
        }
        EXPECT_EQ(transferred,size_t(69*2)*(source_layout.cpu_block_stride/64));
        EXPECT_EQ(manifest.total_stream_bytes,transferred);
        EXPECT_LT(transferred,size_t(source_layout.unit_count)*source_layout.cpu_block_stride);
        for(size_t n=0;n<192;++n)for(size_t kb=0;kb<4;++kb) {
            const auto actual=encodedTierColumn(destination,source_layout,n,kb);
            if(n<69 && kb<2)EXPECT_EQ(actual,encodedTierColumn(source,cpu_source,59+n,1+kb));
            else EXPECT_EQ(actual,encodedTierColumn(untouched,source_layout,n,kb));
        }
        auto bad=manifest;bad.source_projection.first_row++;
        EXPECT_FALSE(bad.valid()); // Original source coordinates belong to identity.
    }
}
} // namespace
} // namespace llaminar2

namespace llaminar2 {
namespace {
TEST(ExpertTierWeightStream, NativeCpuRectanglesUseOnlyLivePlanesAndIndependentRootStrides)
{
    for(const auto& format:native_vnni_formats::kAllSourceFormats)for(unsigned quantum:{1u,3u,17u,64u}) {
        SCOPED_TRACE(format.quant_type);
        SCOPED_TRACE(quantum);
        const auto encoding=cpu::native_vnni::preparedEncodingForSource(*format.metadata);
        ExpertTierWeightCPUStorageLayout source{
            .N=69,.K=64,.blocks_per_row=2,
            .cpu_codebook_id=format.metadata->codebook_id,
            .cpu_is_asymmetric=uint8_t(format.metadata->is_asymmetric),
            .cpu_is_superblock=uint8_t(format.metadata->is_superblock),
            .cpu_encoding=encoding,
            .cpu_data_stride=uint32_t(cpu::native_vnni::preparedDataStride(encoding)),
            .cpu_block_stride=uint32_t(cpu::native_vnni::preparedInterleavedBlockStride(encoding,format.metadata->is_asymmetric))};
        source.cpu_N=128;source.cpu_K=96;source.cpu_first_row=59;source.cpu_first_column=32;
        source.wire_columns_per_unit=quantum;source.unit_count=((69+quantum-1)/quantum)*2;
        source.maximum_units_per_chunk=std::min(2u,source.unit_count);
        auto target=source;target.cpu_N=160;target.cpu_K=128;target.cpu_first_row=7;target.cpu_first_column=64;
        ASSERT_TRUE(source.valid());ASSERT_TRUE(target.valid());
        std::vector<uint8_t> original(source.cpuStorageBytes()),destination(target.cpuStorageBytes(),0xa5);
        for(size_t byte=0;byte<original.size();++byte)original[byte]=uint8_t(byte*37+11);
        if(quantum==64) {
            auto identity=source;identity.N=128;identity.K=96;identity.blocks_per_row=3;
            identity.cpu_first_row=0;identity.cpu_first_column=0;
            identity.unit_count=6;identity.maximum_units_per_chunk=6;
            std::vector<uint8_t> whole(original.size());
            ASSERT_TRUE(gatherExpertTierCPUWeightChunk(identity,original,0,6,whole));
            EXPECT_EQ(whole,original); // Whole aligned units preserve their exact bulk-copy representation.
        }
        const auto dirty=destination;size_t transferred=0;
        for(uint32_t first=0;first<source.unit_count;first+=source.maximum_units_per_chunk) {
            const auto count=std::min(source.maximum_units_per_chunk,source.unit_count-first);
            const auto bytes=source.chunkBytes(count,first);
            std::vector<uint8_t> wire(bytes),returned(bytes);
            ASSERT_TRUE(gatherExpertTierCPUWeightChunk(source,original,first,count,wire));
            ASSERT_TRUE(scatterExpertTierCPUWeightChunk(target,wire,first,count,destination));
            ASSERT_TRUE(gatherExpertTierCPUWeightChunk(target,destination,first,count,returned));
            EXPECT_EQ(returned,wire);transferred+=bytes;
            const auto before=destination;
            EXPECT_FALSE(scatterExpertTierCPUWeightChunk(target,std::span(wire).first(bytes-1),first,count,destination));
            EXPECT_EQ(before,destination);
        }
        EXPECT_EQ(transferred,size_t(69*2)*(source.cpu_block_stride/64));
        for(size_t n=0;n<192;++n)for(size_t kb=0;kb<4;++kb) {
            const auto observed=encodedTierColumn(destination,target,n,kb);
            if(n>=7 && n<76 && kb>=2)
                EXPECT_EQ(observed,encodedTierColumn(original,source,n-7+59,kb-2+1));
            else EXPECT_EQ(observed,encodedTierColumn(dirty,target,n,kb));
        }
    }
}
} // namespace
} // namespace llaminar2
