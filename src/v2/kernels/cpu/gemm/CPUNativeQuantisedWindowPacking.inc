/**
 * @file CPUNativeQuantisedWindowPacking.inc
 * @brief Canonical native-group packing into one final interleaved CPU root.
 *
 * Included inside cpu::native_quantised after the existing encoding helpers. Every
 * task owns one destination N chunk and source-native K group, including Q6_K
 * bitplanes, so rows never race on shared words. Original scales/coefficients
 * and the historical FP32-to-half edges are preserved exactly. Scratch is an
 * explicit disjoint source-arena subrange, never a matrix-sized side vector.
 */

/** @brief Exact simultaneous scratch for one original native group of one source row. */
struct NativePackGroupScratch {
    size_t payload=0, primary=0, secondary=0, emins=0, decoded=0, scales=0, mins=0, bytes=0;
    /** @brief Price one original codec group. @param format Canonical source format. @return Offsets consumed by the same native-group producer below. */
    static NativePackGroupScratch checked(const NativeQuantisedFormatInfo& format) {
        NativePackGroupScratch result;size_t cursor=0;
        const auto reserve=[&](size_t bytes,size_t alignment) {
            cursor=(cursor+alignment-1)&~(alignment-1);const auto offset=cursor;cursor+=bytes;return offset;
        };
        const auto encoding=preparedEncodingForSource(format);
        const size_t blocks=format.is_superblock?8:1;
        const bool expanded=encoding==CPUNativeQuantisedEncoding::ExpandedInt8 ||
            encoding==CPUNativeQuantisedEncoding::IQ3SExpandedSourceScale;
        const bool packed=!expanded || is_payload_decodable(format.codebook_id);
        if(packed) {
            result.payload=reserve(blocks*size_t(format.payload_bytes),1);
            result.primary=reserve(blocks*sizeof(uint16_t),alignof(uint16_t));
            result.secondary=reserve(blocks*sizeof(uint16_t),alignof(uint16_t));
            if(encoding==CPUNativeQuantisedEncoding::CompactMultiScale)
                result.emins=reserve(blocks*sizeof(uint32_t),alignof(uint32_t));
        }
        if(expanded || encoding==CPUNativeQuantisedEncoding::NibbleLUT ||
            encoding==CPUNativeQuantisedEncoding::NibbleLUTSourceScale)
            result.decoded=reserve((expanded?blocks:1)*32,1);
        if(expanded && !packed) {
            result.scales=reserve(blocks*sizeof(float),alignof(float));
            result.mins=reserve(blocks*sizeof(float),alignof(float));
        }
        result.bytes=(cursor+63)&~size_t(63);
        if(!result.bytes || format.payload_bytes<=0 || format.payload_bytes>32)
            throw std::invalid_argument("Native pack source group has invalid scratch geometry");
        return result;
    }
};

/**
 * @brief Initialize only the one final canonical root; no source-oriented arrays exist.
 * @param format Original canonical native format descriptor.
 * @param n Useful output rows, before existing 64-row padding.
 * @param k Useful input columns.
 * @param placement Original admitted final CPU owner/NUMA placement.
 * @param out Final root and immutable kernel geometry.
 */
inline void initializeNativeWindowOutput(const NativeQuantisedFormatInfo& format,int n,int k,
    CPUWeightStoragePlacement placement,CPUNativeQuantisedPackedWeights& out) {
    if(n<=0 || k<=0 || n>std::numeric_limits<int>::max()-63 || k>std::numeric_limits<int>::max()-31)
        throw std::invalid_argument("Native weight writer requires representable positive geometry");
    out.releaseSourcePayload();out.releaseDecodedBuffer();
    out.N=n;out.K=k;out.N_padded=(n+63)/64*64;out.blocks_per_row=(k+31)/32;
    out.payload_bytes=format.payload_bytes;out.codebook_id=format.codebook_id;
    out.is_asymmetric=format.is_asymmetric;out.is_superblock=format.is_superblock;
    out.encoding=preparedEncodingForSource(format);out.data_stride=preparedDataStride(out.encoding);
    out.interleaved_block_stride=preparedInterleavedBlockStride(out.encoding,out.is_asymmetric);
    const auto chunks=size_t(out.N_padded/64),blocks=size_t(out.blocks_per_row);
    if(blocks && chunks>std::numeric_limits<size_t>::max()/blocks/out.interleaved_block_stride)
        throw std::overflow_error("Native weight final extent overflows");
    out.native_interleaved=placement.allocate<uint8_t>(chunks*blocks*out.interleaved_block_stride);
}

/**
 * @brief Append actual source rows/native groups into their one final destination.
 * @param source Actual readable native source/window object, not whole-parent metadata.
 * @param format Same original source descriptor used to initialize output and scratch.
 * @param source_k Actual source object's K extent, independent of destination slicing.
 * @param source_row First actual source row in this append.
 * @param source_kb First actual source 32-value block in this append.
 * @param rows Actual source rows present.
 * @param blocks Actual logical K blocks present, including a legal final tensor tail.
 * @param destination_row First useful final row written by this append.
 * @param destination_kb First final logical K block written by this append.
 * @param scratch Explicit per-source-group scratch in the original admitted arena.
 * @param out Existing single final root; every source coordinate is written once by its owner.
 */
inline void appendNativeWeightGroups(const IINT8Unpackable& source,const NativeQuantisedFormatInfo& format,
    int source_k,int source_row,int source_kb,int rows,int blocks,int destination_row,int destination_kb,
    std::span<std::byte> scratch,CPUNativeQuantisedPackedWeights& out) {
    const auto* source_tensor=dynamic_cast<const TensorBase*>(&source);
    if(!source_tensor || source_tensor->shape().size()!=2 ||
       source_tensor->shape()[1]!=size_t(source_k) || source_row<0 ||
       size_t(source_row)>source_tensor->shape()[0] || rows<0 ||
       size_t(rows)>source_tensor->shape()[0]-size_t(source_row))
        throw std::invalid_argument("Native append requires exact readable source matrix bounds");
    const auto* actual_format=source.vnniFormatInfo();
    if(!actual_format || actual_format->codebook_id!=format.codebook_id ||
       actual_format->payload_bytes!=format.payload_bytes || actual_format->is_superblock!=format.is_superblock ||
       actual_format->is_asymmetric!=format.is_asymmetric || actual_format->has_emins!=format.has_emins ||
       out.codebook_id!=format.codebook_id || out.payload_bytes!=format.payload_bytes ||
       out.is_superblock!=format.is_superblock || out.is_asymmetric!=format.is_asymmetric ||
       out.encoding!=preparedEncodingForSource(format) || out.N<=0 || out.K<=0 ||
       out.N_padded!=int((size_t(out.N)+63)/64*64) ||
       out.blocks_per_row!=int((size_t(out.K)+31)/32) ||
       out.interleaved_block_stride!=preparedInterleavedBlockStride(out.encoding,out.is_asymmetric) ||
       out.native_interleaved.size()!=size_t(out.N_padded/64)*out.blocks_per_row*out.interleaved_block_stride ||
       source_k<=0 || source_k>std::numeric_limits<int>::max()-31)
        throw std::invalid_argument("Native append changed original codec or final owned geometry");
    const int group_blocks=format.is_superblock?8:1;
    const auto layout=NativePackGroupScratch::checked(format);
    const int groups=blocks>0?1+(blocks-1)/group_blocks:0;
    if(rows<=0 || blocks<=0 || source_row<0 || source_kb<0 || destination_row<0 || destination_kb<0 ||
        destination_row>out.N || rows>out.N-destination_row || destination_kb>out.blocks_per_row ||
        blocks>out.blocks_per_row-destination_kb || source_k<=0 ||
        source_kb>(source_k+31)/32 || blocks>(source_k+31)/32-source_kb ||
        source_kb%group_blocks || destination_kb%group_blocks ||
        size_t(rows)>std::numeric_limits<size_t>::max()/size_t(groups)/layout.bytes ||
        scratch.size()<size_t(rows)*groups*layout.bytes ||
        reinterpret_cast<uintptr_t>(scratch.data())%alignof(uint32_t))
        throw std::invalid_argument("Native weight append source/destination/scratch mismatch");
    const int first_chunk=destination_row/64,last_chunk=(destination_row+rows-1)/64;
#pragma omp parallel for collapse(2) schedule(static)
    for(int chunk=first_chunk;chunk<=last_chunk;++chunk)for(int group=0;group<groups;++group) {
        const int first=std::max(destination_row,chunk*64);
        const int end=std::min(destination_row+rows,(chunk+1)*64);
        const int count=std::min(group_blocks,blocks-group*group_blocks);
        const int input_first=source_kb+group*group_blocks;
        const int output_first=destination_kb+group*group_blocks;
        // A destination unit is first touched exactly when its original row0
        // arrives. Later partial windows preserve already-written sibling rows.
        if(first==chunk*64)for(int b=0;b<count;++b)
            std::memset(out.native_interleaved.data()+(size_t(chunk)*out.blocks_per_row+output_first+b)*out.interleaved_block_stride,
                0,out.interleaved_block_stride);
        for(int row=first;row<end;++row) {
            const int local_row=row-destination_row,column=row%64;
            auto* memory=reinterpret_cast<uint8_t*>(scratch.data())+(size_t(local_row)*groups+group)*layout.bytes;
            std::memset(memory,0,layout.bytes);
            auto* payload=memory+layout.payload;
            auto* primary=reinterpret_cast<uint16_t*>(memory+layout.primary);
            auto* secondary=reinterpret_cast<uint16_t*>(memory+layout.secondary);
            auto* emins=reinterpret_cast<uint32_t*>(memory+layout.emins);
            auto* decoded=reinterpret_cast<int8_t*>(memory+layout.decoded);
            auto* scales=reinterpret_cast<float*>(memory+layout.scales);
            auto* mins=reinterpret_cast<float*>(memory+layout.mins);
            const int actual_row=source_row+local_row;
            const bool expanded=out.usesExpandedInt8();
            const bool native_payload=!expanded || is_payload_decodable(format.codebook_id);
            if(native_payload) {
                const VnniPackContext context{.raw_bytes=nullptr,.N=1,.K=source_k,
                    .blocks_per_row=(source_k+31)/32,.payload_bytes=format.payload_bytes,
                    .payload_array=payload,.scales_array=primary,.mins_array=secondary,
                    .emins_array=out.usesCompactMultiScale()?emins:nullptr,
                    .destination_block_origin=input_first,.destination_block_count=count};
                for(int b=0;b<count;++b) {
                    source.packVnniBlock(context,actual_row,0,input_first+b);
                    if(expanded)decode_native_block(format.codebook_id,payload+size_t(b)*format.payload_bytes,decoded+b*32);
                }
            } else if(group_blocks==8 && count==8) {
                source.unpack_superblock_to_int8(actual_row,input_first/8,decoded,scales,mins);
            } else for(int b=0;b<count;++b) {
                source.unpack_block_to_int8(actual_row,input_first+b,decoded+b*32);
                scales[b]=source.get_block_scale(actual_row,input_first+b);
                mins[b]=source.get_block_min(actual_row,input_first+b);
            }
            for(int b=0;b<count;++b) {
                auto* target=out.native_interleaved.data()+(size_t(chunk)*out.blocks_per_row+output_first+b)*out.interleaved_block_stride;
                const auto* input=payload+size_t(b)*format.payload_bytes;
                if(out.usesCompactMultiScale()) {
                    for(int byte=0;byte<format.payload_bytes;++byte)
                        target[(byte/4)*256+column*4+byte%4]=input[byte];
                    std::memcpy(target+out.data_stride+column*2,primary+b,2);
                    std::memcpy(target+out.data_stride+128+column*2,secondary+b,2);
                    std::memcpy(target+out.data_stride+256+column*4,emins+b,4);
                } else if(out.usesQ6KNativeDualScale()) {
                    for(int g=0;g<4;++g)std::memcpy(target+g*256+column*4,input+g*4,4);
                    for(int g=0;g<8;++g) {
                        auto* planes=target+1024+g*64+(column/16)*16;
                        uint64_t low=0,high=0;std::memcpy(&low,planes,8);std::memcpy(&high,planes+8,8);
                        for(int k=0;k<4;++k) {
                            const auto bits=(input[16+g]>>(k*2))&3;
                            low|=uint64_t(bits&1)<<((column%16)*4+k);
                            high|=uint64_t(bits>>1)<<((column%16)*4+k);
                        }
                        std::memcpy(planes,&low,8);std::memcpy(planes+8,&high,8);
                    }
                    std::memcpy(target+1536+column*2,primary+b,2);
                    std::memcpy(target+1664+column*2,secondary+b,2);
                } else {
                    const bool nibble=out.usesNibbleLUT();
                    const int8_t* values=nibble?decoded:decoded+b*32;
                    if(nibble) {
                        decode_native_block(format.codebook_id,input,decoded);values=decoded;
                        for(int g=0;g<4;++g)std::memcpy(target+g*256+column*4,input+g*4,4);
                    } else for(int g=0;g<8;++g)std::memcpy(target+g*256+column*4,values+g*4,4);
                    int32_t sum=0;for(int k=0;k<32;++k)sum+=values[k];
                    const auto compensation=static_cast<int16_t>(sum);
                    std::memcpy(target+out.data_stride+column*2,&compensation,2);
                    uint16_t scale=0,minimum=0;
                    if(out.usesNibbleLUTSourceScale() || out.usesIQ3SExpandedSourceScale()) {
                        scale=primary[b];minimum=secondary[b];
                    } else if(nibble) {
                        scale=fp32_to_fp16(source.get_block_scale(actual_row,input_first+b));
                        minimum=fp32_to_fp16(source.get_block_min(actual_row,input_first+b));
                    } else if(native_payload) {
                        scale=fp32_to_fp16(fp16_to_fp32(primary[b]));
                        minimum=fp32_to_fp16(fp16_to_fp32(secondary[b]));
                    } else {scale=fp32_to_fp16(scales[b]);minimum=fp32_to_fp16(mins[b]);}
                    if(out.usesIQ3SExpandedSourceScale() && !native_quantised_iq3s_scale::valid(minimum))
                        throw std::invalid_argument("IQ3_S source nibble is invalid");
                    std::memcpy(target+out.data_stride+128+column*2,&scale,2);
                    if(out.is_asymmetric || out.usesNibbleLUTSourceScale() || out.usesIQ3SExpandedSourceScale())
                        std::memcpy(target+out.data_stride+256+column*2,&minimum,2);
                }
            }
        }
    }
}
