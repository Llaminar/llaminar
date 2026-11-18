/**
 * @file NativeWeightReaders.inc
 * @brief Native operand views and phase-local Q4 coefficient copying from authoritative K32-major weights.
 * Included inside the copied original anonymous namespace; no persistent weight
 * representation is created. Native Q8 keeps exact decoded code/scale words;
 * Q5_1 reconstructs the exact original register operands. Q4_K/Q5_K consume the already-rounded effective half scales.
 */
using NativeMatrix = llaminar2::DeviceNativeQuantisedMatrixDesc;

/**
 * @brief Read immutable prepared weight bytes through their global-memory space.
 * @tparam T Exact existing operand type; no conversion or repacking is performed.
 * @param pointer Address within an admitted prepared payload or metadata plane.
 * @return The identical source bytes in a register value.
 * @pre The prepared-weight owner retains globally addressable storage for this
 * launch. LDS/private pointers are not valid weight storage.
 * @details The descriptor ABI remains unchanged. Making the already-admitted
 * global address space explicit prevents generic FLAT reads from coupling weight
 * traffic to LDS completion counters. Memcpy retains the existing unaligned-read
 * contract of native Q5_1's twenty-byte blocks.
 */
template<class T>
__device__ __forceinline__ T native_global_weight_read(const void* pointer)
{
    const auto* source = reinterpret_cast<const unsigned char __attribute__((address_space(1)))*>(
        reinterpret_cast<uintptr_t>(pointer));
    T value;
    __builtin_memcpy(&value,source,sizeof(T));
    return value;
}

/**
 * @brief Borrow an immutable descriptor scalar with the original wave authority.
 * @param value Field shared by every active lane of the original weight task.
 * @return The same bits, explicitly carried as a wave-uniform value.
 * @pre Every active lane owns the same matrix; row and K coordinates are excluded.
 */
__device__ __forceinline__ unsigned native_weight_uniform(unsigned value)
{
    return static_cast<unsigned>(__builtin_amdgcn_readfirstlane(value));
}

/**
 * @brief Borrow an immutable descriptor plane without retaining vector addresses.
 * @tparam T Original pointed-to plane type, including its const qualification.
 * @param value Plane shared by every active lane of the original weight task.
 * @return The identical pointer; both halves retain their unsigned representation.
 * @pre Every active lane owns the same matrix, including under tail predicates.
 */
template<class T>
__device__ __forceinline__ T* native_weight_uniform(T* value)
{
    const uintptr_t bits = reinterpret_cast<uintptr_t>(value);
    const unsigned low = native_weight_uniform(static_cast<unsigned>(bits));
    const unsigned high = native_weight_uniform(static_cast<unsigned>(bits >> 32));
    return reinterpret_cast<T*>((uintptr_t(high) << 32) | low);
}

/** @brief SGPR result tuple for one native scalar-memory instruction. */
using NativeScalarWords4 = unsigned __attribute__((ext_vector_type(4)));

/**
 * @brief Borrow required fields of an existing kernel argument descriptor.
 * @tparam NeedMins Whether the original reader consumes the secondary plane.
 * @param matrix Original descriptor reference, possibly a by-value argument.
 * @param columns Exact original output-column count.
 * @param payload Exact original compact payload pointer.
 * @param scales Exact original primary half plane.
 * @param mins Exact secondary half plane, or null when unused.
 */
template<bool NeedMins>
__device__ __forceinline__ void native_descriptor_fields(const NativeMatrix& matrix,
    unsigned& columns,const uint8_t*& payload,const unsigned short*& scales,const unsigned short*& mins)
{
    columns=native_weight_uniform(matrix.n);
    payload=native_weight_uniform(matrix.payload);
    scales=native_weight_uniform((const unsigned short*)matrix.scales);
    if constexpr (NeedMins) mins=native_weight_uniform((const unsigned short*)matrix.mins);
    else mins=nullptr;
}

/**
 * @brief Read only required canonical fields from a retained global routed table.
 * @tparam NeedMins Whether the original reader consumes the secondary plane.
 * @param matrix Same original global table entry for all active wave lanes.
 * @param columns Original column count.
 * @param payload Original payload pointer.
 * @param scales Original scale-plane pointer.
 * @param mins Original minimum-plane pointer, or null when unused.
 * @pre The exact-stream table upload is complete before dispatch acquisition;
 * table bytes remain retained and unchanged until all launch work retires.
 * The admitted HIP dispatch must preserve its AGENT/SYSTEM cache-acquire scope.
 * @details Offsets and field types belong to NativeMatrix, not a parallel ABI.
 * Ordinary cached SMEM is valid because the dispatch acquire supplies freshness
 * and no vector/scalar writer changes these tables during the launch. The wait
 * completes the same reads before use; it is not a publication or invalidation.
 * Mutable protocol counters and runtime tables without this lifetime proof are
 * not covered by this reader. Request reset must not rewrite a live table.
 */
template<bool NeedMins>
__device__ __forceinline__ void native_descriptor_fields(const NativeMatrix* matrix,
    unsigned& columns,const uint8_t*& payload,const unsigned short*& scales,const unsigned short*& mins)
{
    static_assert(__is_standard_layout(NativeMatrix) && __is_trivially_copyable(NativeMatrix));
    static_assert(sizeof(matrix->payload)==8 && sizeof(matrix->scales)==8 && sizeof(matrix->mins)==8);
    static_assert(sizeof(void*)==8 && sizeof(matrix->n)==4);
    static_assert(__BYTE_ORDER__==__ORDER_LITTLE_ENDIAN__);
    static_assert(__builtin_offsetof(NativeMatrix,scales)==__builtin_offsetof(NativeMatrix,payload)+8);
    const auto address=reinterpret_cast<uint64_t>(native_weight_uniform(matrix));
    NativeScalarWords4 planes;
    uint64_t min_bits=0;
    if constexpr (NeedMins) {
        asm volatile("s_load_dwordx4 %0, %3, %c4\n\t"
                     "s_load_dwordx2 %1, %3, %c5\n\t"
                     "s_load_dword %2, %3, %c6\n\t"
                     "s_waitcnt lgkmcnt(0)"
            : "=&s"(planes), "=&s"(min_bits), "=&s"(columns)
            : "s"(address), "n"(__builtin_offsetof(NativeMatrix,payload)),
              "n"(__builtin_offsetof(NativeMatrix,mins)), "n"(__builtin_offsetof(NativeMatrix,n))
            : "memory");
    } else {
        asm volatile("s_load_dwordx4 %0, %2, %c3\n\t"
                     "s_load_dword %1, %2, %c4\n\t"
                     "s_waitcnt lgkmcnt(0)"
            : "=&s"(planes), "=&s"(columns)
            : "s"(address), "n"(__builtin_offsetof(NativeMatrix,payload)),
              "n"(__builtin_offsetof(NativeMatrix,n)) : "memory");
    }
    payload=reinterpret_cast<const uint8_t*>(uint64_t(planes[0])|(uint64_t(planes[1])<<32));
    scales=reinterpret_cast<const unsigned short*>(uint64_t(planes[2])|(uint64_t(planes[3])<<32));
    mins=reinterpret_cast<const unsigned short*>(min_bits);
}

/**
 * @brief Wave-task register borrow of an immutable routed matrix's required fields.
 * @details This local value exists only for one original expert task. It does not
 * publish, own or cache weights across tasks; the retained canonical table is
 * the sole descriptor authority. All K iterations borrow the same four fields.
 */
class NativeTaskFields {
    unsigned columns_;
    const uint8_t* payload_;
    const unsigned short *scales_, *mins_;
public:
    /**
     * @brief Read the original required fields once for this task.
     * @param matrix Original retained global descriptor shared by the wave.
     * @pre The original producer/epoch publication is complete and the descriptor
     * remains immutable until this launch retires.
     */
    __device__ __forceinline__ explicit NativeTaskFields(const NativeMatrix* matrix)
    {
        native_descriptor_fields<true>(matrix,columns_,payload_,scales_,mins_);
    }

    /**
     * @brief Borrow exactly the task's already-read descriptor fields.
     * @tparam NeedMins Whether the consumer uses the secondary plane.
     * @param columns Original output-column count.
     * @param payload Original payload address.
     * @param scales Original scale-plane address.
     * @param mins Original secondary plane, or null when unused.
     */
    template<bool NeedMins>
    __device__ __forceinline__ void borrow(unsigned& columns,const uint8_t*& payload,
        const unsigned short*& scales,const unsigned short*& mins) const
    {
        columns=columns_;payload=payload_;scales=scales_;
        if constexpr (NeedMins) mins=mins_; else mins=nullptr;
    }
};

/**
 * @brief Reuse task-local const register fields in the unchanged numerical reader.
 * @tparam NeedMins Whether the reader consumes the secondary metadata plane.
 * @param fields Same original descriptor's task-local register borrow.
 * @param columns Original output width.
 * @param payload Original native payload.
 * @param scales Original half scale plane.
 * @param mins Original secondary plane, or null when unused.
 */
template<bool NeedMins>
__device__ __forceinline__ void native_descriptor_fields(const NativeTaskFields& fields,
    unsigned& columns,const uint8_t*& payload,const unsigned short*& scales,const unsigned short*& mins)
{
    fields.borrow<NeedMins>(columns,payload,scales,mins);
}

/** @brief Load the same Q8 pair into seventeen native register words without byte interleaving.
 * @tparam PairHCDownScales True only for the original interleaved HC-down row pair.
 * @param matrix Exact canonical source descriptor. @param row Original output row.
 * @param block First of the original two adjacent K32 blocks. @param words Sixteen native code words and one packed two-scale word.
 */
template<bool PairHCDownScales=false,class Descriptor>
__device__ __forceinline__ void native_q8_pair(const Descriptor& matrix,int row,int block,int (&words)[17])
{
    unsigned columns;
    const uint8_t* payload;
    const unsigned short *scales,*mins;
    native_descriptor_fields<false>(matrix,columns,payload,scales,mins);
    const size_t first=size_t(block)*columns+row,second=first+columns;
    const int4 a0=native_global_weight_read<int4>(payload+first*32),a1=native_global_weight_read<int4>(payload+first*32+16);
    const int4 b0=native_global_weight_read<int4>(payload+second*32),b1=native_global_weight_read<int4>(payload+second*32+16);
    // Keep native code words native in registers: the original raw reader's
    // halfword interleave followed by alignbyte extraction would add temporary
    // dependencies with no semantic value. The seventeenth word holds both
    // unchanged source scale bits; all original consumers share this mapping.
    words[0]=a0.x;words[1]=a0.y;words[2]=a0.z;words[3]=a0.w;
    words[4]=a1.x;words[5]=a1.y;words[6]=a1.z;words[7]=a1.w;
    words[8]=b0.x;words[9]=b0.y;words[10]=b0.z;words[11]=b0.w;
    words[12]=b1.x;words[13]=b1.y;words[14]=b1.z;words[15]=b1.w;
    if constexpr (PairHCDownScales) {
        // Original HC-down XOR1 lanes own consecutive even/odd rows of the
        // same clamped K pair. Both loaded half fields belong to real rows.
        const unsigned odd=unsigned(row)&1u;
        const size_t scale_pair=(odd ? second : first)-odd;
        const unsigned local=native_global_weight_read<unsigned>(scales+scale_pair);
#if defined(__HIP_DEVICE_COMPILE__)
        static_assert(__builtin_amdgcn_is_invocable(__builtin_amdgcn_update_dpp),
            "HC-down scale pairing requires the fixed quad DPP primitive");
#endif
        const unsigned peer=__builtin_bit_cast(unsigned,__builtin_amdgcn_update_dpp(
            0,__builtin_bit_cast(int,local),0xb1,0xf,0xf,false));
        const unsigned first_pair=odd ? peer : local,second_pair=odd ? local : peer;
        const unsigned shift=16u*odd;
        words[16]=__builtin_bit_cast(int,((first_pair>>shift)&0xffffu)|(((second_pair>>shift)&0xffffu)<<16));
    } else {
    words[16]=int(unsigned(native_global_weight_read<unsigned short>(scales+first))|(unsigned(native_global_weight_read<unsigned short>(scales+second))<<16));
    }
}
/** @brief Load the original six-dword Q5_1 operand, preserving every code/metadata bit.
 * @param matrix Exact original native Q5_1 descriptor. @param row Original output row.
 * @param block Original K32 block. @param words Original6-word destination.
 */
template<class Descriptor>
__device__ __forceinline__ void native_q51_block(const Descriptor& matrix,int row,int block,int (&words)[6])
{
    unsigned columns;
    const uint8_t* payload;
    const unsigned short *scales,*mins;
    native_descriptor_fields<true>(matrix,columns,payload,scales,mins);
    const size_t linear=size_t(block)*columns+row;
    const int4 q=native_global_weight_read<int4>(payload+linear*20);
    words[0]=int(unsigned(native_global_weight_read<unsigned short>(scales+linear))|
        (unsigned(native_global_weight_read<unsigned short>(mins+linear))<<16));
    words[1]=native_global_weight_read<int>(payload+linear*20+16);
    words[2]=q.x;words[3]=q.y;words[4]=q.z;words[5]=q.w;
}
/**
 * @brief Exchange the current native word with its original half-pair peer.
 * @param word Original payload or packed coefficient bits owned by this lane.
 * @return Bit-identical word from physical lane XOR4.
 * @pre All64 lanes of the original Q4 GU task execute each exchange convergently.
 * @details The canonical column mapping puts the two original16-value halves
 * in XOR4 lanes. Immediate0x101f is LLVM's ds_swizzle(SWAP,4): AND31,OR0,XOR4
 * within each32-lane half; bit5 is preserved. No payload address is synthesized.
 */
__device__ __forceinline__ unsigned native_q4_half_peer(unsigned word)
{
#if defined(__HIP_DEVICE_COMPILE__)
    static_assert(__builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_swizzle),
        "Native Q4 half cooperation requires the fixed XOR4 lane-word primitive");
#endif
    return unsigned(__builtin_amdgcn_ds_swizzle(int(word),0x101f));
}

/**
 * @brief Existing Q4 coefficient packet and native-column wave ownership.
 * @details Eight original half fields form one int4 source read; four output
 * columns occupy one physical wave. Their least common row grain is eight.
 * Capacity remains the original64 rows, independent of the live CTA tile.
 */
struct NativeQ4PanelContract {
    static constexpr unsigned rows_per_wave=4;
    static constexpr unsigned rows_per_packet=sizeof(int4)/sizeof(unsigned short);
    static constexpr unsigned minimum_row_grain=rows_per_packet;
    static constexpr unsigned row_capacity=64;
    static_assert(rows_per_packet==8 && rows_per_packet%rows_per_wave==0);
};
/**
 * @brief Choose aligned row chunks without increasing the smallest-grain maximum.
 * @param rows Actual per-entry output extent. @param owners Actual cohort CTAs.
 * @return One or two original metadata packets per cardinal row chunk.
 * @details A two-packet grain reduces split metadata-line ownership for some
 * geometries. Use it only for complete doubled packets and when its cardinal
 * maximum is no greater than the minimum packet grain. Partial source extents
 * retain the minimum grain and a counted last packet; no topology value enters.
 */
__host__ __device__ constexpr unsigned native_gu_row_grain(unsigned rows,unsigned owners) {
    const unsigned one=NativeQ4PanelContract::minimum_row_grain,two=2*one;
    const unsigned chunks_one=(rows+one-1)/one,chunks_two=(rows+two-1)/two;
    const unsigned maximum_one=((chunks_one+owners-1)/owners)*one;
    const unsigned maximum_two=((chunks_two+owners-1)/owners)*two;
    return rows%two==0 && maximum_two<=maximum_one?two:one;
}

/**
 * @brief Locate one unchanged packed scale|min word in temporary GU LDS.
 * @param row CTA-local output row0..63. @param block Original K32 index.
 * @return Word index with each native consumer wave reading64 consecutive words.
 * @details Physical GU lane is4*(block%16)+(row%4); groups of four output
 * rows select its wave. XOR with its wave-pair spreads producer int4 packets
 * across banks while each consumer retains a fixed bank permutation. This is
 * only a phase-local copy of authoritative bits.
 */
__host__ __device__ constexpr unsigned native_q4_panel_index(unsigned row,unsigned block)
{
    return (block/16)*1024+(row/4)*64+((block%16)^((row/4)>>1))*4+row%4;
}

/**
 * @brief Copy only real half coefficients into the original reusable panel.
 * @tparam Descriptor Original immutable routed descriptor authority.
 * @param matrix Selected gate or up descriptor, identical for the whole CTA.
 * @param first_row Actual first output row, aligned to the metadata-packet grain.
 * @param rows Exact real tile rows, at most the original64-row capacity.
 * @param tid Physical worker. @param threads Original CTA size.
 * @param blocks Original hidden/32. @param panel Existing blocks*64 word bank.
 * @details Complete aligned packets use the original int4 loads and packing.
 * A partial or otherwise unaligned source packet reads only its actual ushort
 * fields. No neighboring/padded column is read. Local helper returns do not
 * bypass the caller's unconditional CTA publication and retirement joins.
 */
template<class Descriptor>
__device__ __forceinline__ void native_q4_panel_publish(const Descriptor& matrix,
    unsigned first_row,unsigned rows,unsigned tid,unsigned threads,unsigned blocks,unsigned* panel)
{
    constexpr unsigned Packet=NativeQ4PanelContract::rows_per_packet;
    const unsigned packets=(rows+Packet-1)/Packet;
    if(rows==0 || tid>=blocks*packets)return;
    unsigned columns;const uint8_t* payload;const unsigned short *scales,*mins;
    native_descriptor_fields<true>(matrix,columns,payload,scales,mins);
#pragma unroll 1
    for(unsigned task=tid;task<blocks*packets;task+=threads) {
        const unsigned block=task/packets,row=(task%packets)*Packet;
        const unsigned count=rows-row<Packet?rows-row:Packet;
        const size_t source=size_t(block)*columns+first_row+row;
        if(count==Packet && (source&(Packet-1))==0) {
            const int4 s=native_global_weight_read<int4>(scales+source);
            const int4 m=native_global_weight_read<int4>(mins+source);
            const int4 a=make_int4(
                int((unsigned(s.x)&0xffffu)|(unsigned(m.x)<<16)),
                int((unsigned(s.x)>>16)|(unsigned(m.x)&0xffff0000u)),
                int((unsigned(s.y)&0xffffu)|(unsigned(m.y)<<16)),
                int((unsigned(s.y)>>16)|(unsigned(m.y)&0xffff0000u)));
            const int4 b=make_int4(
                int((unsigned(s.z)&0xffffu)|(unsigned(m.z)<<16)),
                int((unsigned(s.z)>>16)|(unsigned(m.z)&0xffff0000u)),
                int((unsigned(s.w)&0xffffu)|(unsigned(m.w)<<16)),
                int((unsigned(s.w)>>16)|(unsigned(m.w)&0xffff0000u)));
            *reinterpret_cast<int4*>(panel+native_q4_panel_index(row,block))=a;
            *reinterpret_cast<int4*>(panel+native_q4_panel_index(row+4,block))=b;
        } else {
#pragma unroll 1
            for(unsigned offset=0;offset<count;++offset) {
                const unsigned s=native_global_weight_read<unsigned short>(scales+source+offset);
                const unsigned m=native_global_weight_read<unsigned short>(mins+source+offset);
                panel[native_q4_panel_index(row+offset,block)]=s|(m<<16);
            }
        }
    }
}

/**
 * @brief Native QK block operands and the existing effective coefficients.
 * Q4 uses q as its owned canonical block and high.x as its two half metadata
 * bits; its peer owns the other block. Q5 retains the original
 * joined nibble/high-bit interpretation. No persistent weight copy is created.
 */
struct NativeQKItem {
    int4 q,high;
    float scale_a,scale_b,min_a,min_b;
};
/** @brief Read the original lane's native paired blocks without a Q4 nibble repack.
 * @tparam Q5 Actual Q5_K source; false is original Q4_K.
 * @param matrix Canonical affine native source. @param row Original output row.
 * @param superblock Original256-element superblock. @param pair Original subblock-pair0..3.
 * @param half Original low/high16-value portion.
 * @param coefficient_panel Published temporary Q4 word plane, only when selected.
 * @param panel_first_row Original first output row of that64-row plane.
 * @return Same integer codes and prepared effective coefficients.
 * @tparam CoefficientPanel Select temporary copied coefficient bits only for Q4.
 * @details All existing unselected readers retain their original metadata source;
 * payload reads and both Q4 integer/FP consumers are shared without duplication.
 */
template<bool Q5, bool CoefficientPanel=false, class Descriptor>
__device__ __forceinline__ NativeQKItem native_qk_item(const Descriptor& matrix,int row,int superblock,int pair,int half,
    const unsigned* coefficient_panel=nullptr,unsigned panel_first_row=0)
{
    static_assert(!Q5 || !CoefficientPanel,"Only Q4 uses the temporary coefficient panel");
    constexpr size_t stride=Q5?20:16;
    unsigned columns;
    const uint8_t* payload;
    const unsigned short *scales,*mins;
    native_descriptor_fields<!CoefficientPanel>(matrix,columns,payload,scales,mins);
    const size_t a=size_t(superblock*8+pair*2)*columns+row,b=a+columns;
    if constexpr (!Q5) {
        // Each original half owns one unchanged globally stored K32 block.
        // Only the current payload word is exchanged at consumption.
        const size_t owned=half ? b : a;
        const int4 q=native_global_weight_read<int4>(payload+owned*stride);
        unsigned coefficients;
        if constexpr (CoefficientPanel) {
            // The original canonical block and row select identical copied bits.
            const unsigned block=unsigned(superblock*8+pair*2+half);
            coefficients=coefficient_panel[native_q4_panel_index(unsigned(row)-panel_first_row,block)];
        } else {
            coefficients=unsigned(native_global_weight_read<unsigned short>(scales+owned))|
                (unsigned(native_global_weight_read<unsigned short>(mins+owned))<<16);
        }
        return {q,make_int4(int(coefficients),0,0,0),0.0f,0.0f,0.0f,0.0f};
    }
    const int4 qa=native_global_weight_read<int4>(payload+a*stride);
    const int4 qb=native_global_weight_read<int4>(payload+b*stride);
    const unsigned av[4]={(unsigned)qa.x,(unsigned)qa.y,(unsigned)qa.z,(unsigned)qa.w};
    const unsigned bv[4]={(unsigned)qb.x,(unsigned)qb.y,(unsigned)qb.z,(unsigned)qb.w};
    unsigned q[4],high[4]{};
    unsigned ah=0,bh=0;
    if constexpr(Q5){ah=native_global_weight_read<unsigned>(payload+a*stride+16);bh=native_global_weight_read<unsigned>(payload+b*stride+16);}
#pragma unroll
    for(int j=0;j<4;++j) {
        q[j]=((av[j]>>(half*4))&0x0f0f0f0fu)|(((bv[j]>>(half*4))&0x0f0f0f0fu)<<4);
        if constexpr(Q5) {
            unsigned h=0;
#pragma unroll
            for(int byte=0;byte<4;++byte) {
                const int i=half*16+j*4+byte;
                h|=(((ah>>i)&1u)<<(pair*2+byte*8));
                h|=(((bh>>i)&1u)<<(pair*2+1+byte*8));
            }
            high[j]=h;
        }
    }
    return {make_int4(q[0],q[1],q[2],q[3]),make_int4(high[0],high[1],high[2],high[3]),
        __half2float(__ushort_as_half(native_global_weight_read<unsigned short>(scales+a))),__half2float(__ushort_as_half(native_global_weight_read<unsigned short>(scales+b))),
        __half2float(__ushort_as_half(native_global_weight_read<unsigned short>(mins+a))),__half2float(__ushort_as_half(native_global_weight_read<unsigned short>(mins+b)))};
}


/**
 * @brief Restore two shared output rows to independent original half32 trees.
 * @param value One unchanged original Q8 pair-dot leaf for this physical lane.
 * @param lane Physical lane0..63; all64 lanes execute this bit transfer.
 * @return Exact source bits from physical lane2*partition+row.
 * @details Each input bank owns32 successive original K64 pair leaves per row.
 * The caller applies the canonical half32_sum separately to banks0/1, then
 * appends upper+lower in the same order as the original wave64 root. Only
 * lanes31/63 are consumed; no all-lane equivalent-result promise is made.
 */
__device__ __forceinline__ float native_shared_two_row_original_lane(float value,unsigned lane)
{
    const unsigned source=((lane&31u)<<1)|(lane>>5);
    return __int_as_float(__builtin_amdgcn_ds_bpermute(source<<2,__float_as_int(value)));
}

/**
 * @brief Restore eight original output rows before the unchanged row8 tree.
 * @param value Original per-partition leaf in column-contiguous physical lanes.
 * @param lane Physical lane0..63; every lane participates.
 * @return Exact bits from physical lane8*partition+row.
 * @pre T1 Q51 down has NBK<=8 and shared PD<=8; no metadata pairing is used.
 */
__device__ __forceinline__ float native_eight_row_original_lane(float value,unsigned lane)
{
    const unsigned source=((lane&7u)<<3)|(lane>>3);
    return __int_as_float(__builtin_amdgcn_ds_bpermute(source<<2,__float_as_int(value)));
}

/**
 * @brief Retain the original row16 tree's final addition of its positive-zero half.
 * @param value Completed original row8 tree for partitions0..7.
 * @return RN(value + positive zero), including signed-zero normalization.
 * @details Original inactive partitions8..15 reduce to +0 under T1/NBK<=8,
 * PD<=8. Explicit native addition prevents removal of that required operation.
 */
__device__ __forceinline__ float native_append_zero_half(float value)
{
    float result;
    asm volatile("v_add_f32 %0, %1, 0" : "=v"(result) : "v"(value));
    return result;
}

/**
 * @brief Restore fork row-major lane ownership before its original DPP reduction.
 * @param value Partial produced with physical lane = 4*partition + output row.
 * @param lane Original physical lane in [0,63], uniform participation required.
 * @return Bit-identical partial for fork lane = 16*output row + partition.
 * @details Only register ownership changes. The subsequent original row16_sum
 * executes its unmodified four additions in the original lane topology.
 */
__device__ __forceinline__ float native_four_row_original_lane(float value, unsigned lane)
{
    const unsigned source = ((lane & 15u) << 2) | (lane >> 4);
    return __int_as_float(__builtin_amdgcn_ds_bpermute(source << 2, __float_as_int(value)));
}
