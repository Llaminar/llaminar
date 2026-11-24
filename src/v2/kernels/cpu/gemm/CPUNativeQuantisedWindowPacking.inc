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
    size_t payload=0, primary=0, secondary=0, codes=0, superblock=0, decoded=0, scales=0, mins=0, bytes=0;
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
        if(packed && preparedSuperblockCoded(encoding)) {
            // Coded units: exact payload, one code per block, one word per superblock.
            result.payload=reserve(blocks*size_t(format.payload_bytes),1);
            result.codes=reserve(blocks*sizeof(uint32_t),alignof(uint32_t));
            result.superblock=reserve(sizeof(uint32_t),alignof(uint32_t));
        } else if(packed) {
            result.payload=reserve(blocks*size_t(format.payload_bytes),1);
            result.primary=reserve(blocks*sizeof(uint16_t),alignof(uint16_t));
            result.secondary=reserve(blocks*sizeof(uint16_t),alignof(uint16_t));
        }
        if(expanded)
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
    out.encoding=preparedEncodingForSource(format);out.data_stride=preparedDataStride(out.encoding,out.codebook_id);
    out.interleaved_block_stride=preparedInterleavedBlockStride(out.encoding,out.is_asymmetric,out.codebook_id);
    const auto chunks=size_t(out.N_padded/64),blocks=size_t(out.blocks_per_row);
    const size_t unit_bytes=size_t(out.interleaved_block_stride)+out.superblockHeaderBytes();
    if(blocks && chunks>std::numeric_limits<size_t>::max()/blocks/unit_bytes)
        throw std::overflow_error("Native weight final extent overflows");
    out.native_interleaved.replace(placement.allocate<uint8_t>(out.preparedBytes()));
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
       out.interleaved_block_stride!=preparedInterleavedBlockStride(out.encoding,out.is_asymmetric,out.codebook_id) ||
       out.native_interleaved.size()!=out.preparedBytes() ||
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
    // Each window rewrites part of the publication; the producer seals after the last window.
    const auto interleaved=out.native_interleaved.beginRewrite();
#pragma omp parallel for collapse(2) schedule(static)
    for(int chunk=first_chunk;chunk<=last_chunk;++chunk)for(int group=0;group<groups;++group) {
        const int first=std::max(destination_row,chunk*64);
        const int end=std::min(destination_row+rows,(chunk+1)*64);
        const int count=std::min(group_blocks,blocks-group*group_blocks);
        const int input_first=source_kb+group*group_blocks;
        const int output_first=destination_kb+group*group_blocks;
        // A destination unit is first touched exactly when its original row0
        // arrives. Later partial windows preserve already-written sibling rows.
        if(first==chunk*64) {
            for(int b=0;b<count;++b)
                std::memset(interleaved.data()+out.blockOffset(chunk,output_first+b),0,out.interleaved_block_stride);
            if(out.superblockHeaderBytes())
                std::memset(interleaved.data()+out.superblockHeaderOffset(chunk,output_first),0,out.superblockHeaderBytes());
        }
        for(int row=first;row<end;++row) {
            const int local_row=row-destination_row,column=row%64;
            auto* memory=reinterpret_cast<uint8_t*>(scratch.data())+(size_t(local_row)*groups+group)*layout.bytes;
            std::memset(memory,0,layout.bytes);
            auto* payload=memory+layout.payload;
            auto* primary=reinterpret_cast<uint16_t*>(memory+layout.primary);
            auto* secondary=reinterpret_cast<uint16_t*>(memory+layout.secondary);
            auto* codes=reinterpret_cast<uint32_t*>(memory+layout.codes);
            auto* superblock=reinterpret_cast<uint32_t*>(memory+layout.superblock);
            const bool coded=preparedSuperblockCoded(out.encoding);
            auto* decoded=reinterpret_cast<int8_t*>(memory+layout.decoded);
            auto* scales=reinterpret_cast<float*>(memory+layout.scales);
            auto* mins=reinterpret_cast<float*>(memory+layout.mins);
            const int actual_row=source_row+local_row;
            const bool expanded=out.usesExpandedInt8();
            const bool native_payload=!expanded || is_payload_decodable(format.codebook_id);
            if(native_payload) {
                // Coded units receive the same codes/words the GPU repack
                // publishes (one window = one source superblock, so its word
                // lands in superblock[0]); per-block units their staged words.
                const VnniPackContext context{.raw_bytes=nullptr,.N=1,.K=source_k,
                    .blocks_per_row=(source_k+31)/32,.payload_bytes=format.payload_bytes,
                    .payload_array=payload,.scales_array=coded?nullptr:primary,.mins_array=coded?nullptr:secondary,
                    .emins_array=nullptr,
                    .destination_block_origin=input_first,.destination_block_count=count,
                    .primary_layout=out.primaryScaleLayout(),
                    .codes_array=coded?codes:nullptr,.superblocks_array=coded?superblock:nullptr};
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
                auto* target=interleaved.data()+out.blockOffset(chunk,output_first+b);
                const auto* input=payload+size_t(b)*format.payload_bytes;
                if(out.usesCompactMultiScale()) {
                    // Whole words as [word][column], the 1-2 byte tail as [column][tail].
                    const int words=int(preparedCompactPayloadWords(out.codebook_id));
                    const int tail=int(preparedCompactPayloadTailBytes(out.codebook_id));
                    if(out.codebook_id==7) {
                        for(int word=0;word<4;++word)std::memcpy(target+word*256+column*4,input+word*4,4);
                        coded_metadata::storeQ5HighBits(target+1024,column,input+16);
                    } else for(int byte=0;byte<format.payload_bytes;++byte)
                        if(byte<4*words)target[(byte/4)*256+column*4+byte%4]=input[byte];
                        else target[words*256+column*tail+(byte-4*words)]=input[byte];
                    if(preparedCompactCodeBits(out.codebook_id))
                        coded_metadata::storeCode(target+out.data_stride,out.codebook_id,column,codes[b]);
                    if(b==0)
                        coded_metadata::storeSuperblockWord(interleaved.data()+out.superblockHeaderOffset(chunk,output_first),
                            out.codebook_id,column,superblock[0]);
                } else if(out.encoding==CPUNativeQuantisedEncoding::Q4KSuperblockCoded) {
                    // Native nibble words, then the 12-bit code split into an 8-bit and a nibble plane.
                    for(int g=0;g<4;++g)std::memcpy(target+g*256+column*4,input+g*4,4);
                    coded_metadata::storeCode(target+out.data_stride,5,column,codes[b]);
                    if(b==0)
                        coded_metadata::storeSuperblockWord(interleaved.data()+out.superblockHeaderOffset(chunk,output_first),
                            5,column,superblock[0]);
                } else if(out.encoding==CPUNativeQuantisedEncoding::Q5NativeSingleScale) {
                    // Four nibble groups and one group of four paired fifth-bit bytes.
                    // Keeping every group four bytes wide per column lets the CPU
                    // rectangle stream use its unchanged native plane copier.
                    for(int g=0;g<Q5NativePayload::groups;++g) {
                        std::memcpy(target+Q5NativePayload::nibbleOffset(g,column),input+g*4,4);
                        target[Q5NativePayload::fifthOffset(g,column)]=uint8_t(((input[16+g/2]>>(4*(g%2)))&15) |
                            (((input[18+g/2]>>(4*(g%2)))&15)<<4));
                    }
                    const uint16_t scale=fp32_to_fp16(fp16_to_fp32(primary[b]));
                    std::memcpy(target+out.data_stride+column*2,&scale,2);
                    if(out.is_asymmetric) {
                        const uint16_t minimum=fp32_to_fp16(fp16_to_fp32(secondary[b]));
                        std::memcpy(target+out.data_stride+128+column*2,&minimum,2);
                    }
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
                    coded_metadata::storeCode(target+out.data_stride,out.codebook_id,column,codes[b]);
                    if(b==0)
                        coded_metadata::storeSuperblockWord(interleaved.data()+out.superblockHeaderOffset(chunk,output_first),
                            out.codebook_id,column,superblock[0]);
                } else {
                    const bool nibble=out.usesNibbleLUT();
                    const int8_t* values=nibble?nullptr:decoded+b*32;
                    if(nibble) {
                        for(int g=0;g<4;++g)std::memcpy(target+g*256+column*4,input+g*4,4);
                    } else for(int g=0;g<8;++g)std::memcpy(target+g*256+column*4,values+g*4,4);
                    if(out.scaleEncoding()==CPUNativeQuantisedScaleEncoding::E8M0) {
                        // The source packer already admitted the exponent and published
                        // its exact byte, with zero as the all-zero block marker.
                        target[out.data_stride+column]=reinterpret_cast<const uint8_t*>(primary)[b];
                        continue;
                    }
                    const int comp_bytes=out.usesInlineCompensation()?128:0;
                    if(comp_bytes) {
                        int32_t sum=0;for(int k=0;k<32;++k)sum+=values[k];
                        const auto compensation=static_cast<int16_t>(sum);
                        std::memcpy(target+out.data_stride+column*2,&compensation,2);
                    }
                    if(out.usesNibbleLUTSourceScale()) {
                        const auto codebook=superblockMetadataCodebook(out.encoding,out.codebook_id);
                        coded_metadata::storeCode(target+preparedCodedPlaneOffset(out.encoding,out.data_stride),
                            codebook,column,codes[b]);
                        if(b==0)
                            coded_metadata::storeSuperblockWord(interleaved.data()+out.superblockHeaderOffset(chunk,output_first),
                                codebook,column,superblock[0]);
                        continue;
                    }
                    uint16_t scale=0,minimum=0;
                    if(out.usesIQ3SExpandedSourceScale()) {
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
                    std::memcpy(target+out.data_stride+comp_bytes+column*2,&scale,2);
                    if(out.is_asymmetric || out.usesIQ3SExpandedSourceScale())
                        std::memcpy(target+out.data_stride+comp_bytes+128+column*2,&minimum,2);
                }
            }
        }
    }
}
