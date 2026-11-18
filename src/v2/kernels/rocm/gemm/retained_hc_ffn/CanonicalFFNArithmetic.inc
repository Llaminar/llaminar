/**
 * @file CanonicalFFNArithmetic.inc
 * @brief Current per-block and ordered-fold arithmetic inside retained native task/read structure.
 * Included after NativeWeightReaders in the copied device namespace; numerical
 * headers must be included at global scope. No descriptor, storage or policy
 * authority is created by these helpers.
 */
namespace canonical_gu {
static constexpr unsigned rocmOrderedWaveWidth=64;
__device__ __forceinline__ float absoluteWaveValue(float value,unsigned source)
{
    static_assert(rocmOrderedWaveWidth==64,"Absolute partial transfer requires the ordered wave64 program");
    if(__builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_bpermute))
    {
        const auto bits=__builtin_bit_cast(int32_t,value);
#if __has_builtin(__builtin_amdgcn_wave_shuffle)
        const auto moved=__builtin_amdgcn_wave_shuffle(bits,static_cast<int>(source));
#else
        const auto moved=__builtin_amdgcn_ds_bpermute(static_cast<int>(source<<2),bits);
#endif
        return __builtin_bit_cast(float,moved);
    }
    __builtin_trap();
}

template<unsigned Shift>
__device__ __forceinline__ float orderedRowValue(float value)
{
    static_assert(Shift==4||Shift==8||Shift==12);
    return ::llaminar2::rocm::moe_wave_contract::rowShiftNativeBits<Shift>(value);
}

template<unsigned Columns,unsigned Bank>
__device__ __forceinline__ void appendOrderedBank(float partial,unsigned parts,unsigned lane,float& accumulator)
{
    static_assert((Columns==4||Columns==8)&&Bank<4);
    constexpr unsigned first=Bank*(16/Columns);
    if(parts>first)
    {
        // No bank value survives its own ordered appends. Every input shuffle
        // reads the ORIGINAL partial, never another bank's accumulated result.
        float bank=partial;
        if constexpr(Bank!=0)
        {
            // Derive this bank's source only after the preceding original RN
            // prefix. Otherwise the optimizer can retain all bank coordinates
            // through the entire weight/decode loop, although no fold uses them
            // there. This register-only identity changes no bits or ordering.
            unsigned current_lane=lane;
#if defined(__HIP_DEVICE_COMPILE__)
            asm volatile("" : "+v"(current_lane) : "v"(accumulator));
#endif
            bank=absoluteWaveValue(partial,(current_lane+16*Bank)&63u);
        }
        accumulator=rocm_native_quantised_accumulate_rn(accumulator,bank);
        if(parts>first+1)
            accumulator=rocm_native_quantised_accumulate_rn(accumulator,orderedRowValue<Columns>(bank));
        if constexpr(Columns==4)
        {
            if(parts>first+2)
                accumulator=rocm_native_quantised_accumulate_rn(accumulator,orderedRowValue<8>(bank));
            if(parts>first+3)
                accumulator=rocm_native_quantised_accumulate_rn(accumulator,orderedRowValue<12>(bank));
        }
    }
}
/** @brief Fold the original actual P in order; only physical worker0 outputs are consumed. */
__device__ __forceinline__ float foldFourColumns(float partial,unsigned parts,unsigned lane) {
    if(parts==1)return absoluteWaveValue(partial,lane&3u);
    float value=0.f;
    appendOrderedBank<4,0>(partial,parts,lane,value);
    appendOrderedBank<4,1>(partial,parts,lane,value);
    appendOrderedBank<4,2>(partial,parts,lane,value);
    appendOrderedBank<4,3>(partial,parts,lane,value);
    return value;
}
/** @brief Fold original P from arbitrary uniform-stride lanes, without padded-zero additions. */
__device__ __forceinline__ float foldStrided(float partial,unsigned parts,unsigned column,unsigned stride) {
    if(parts==1)return absoluteWaveValue(partial,column);
    float result=0.f;
#pragma unroll 1
    for(unsigned part=0;part<parts;++part)
        result=rocm_native_quantised_accumulate_rn(result,absoluteWaveValue(partial,stride*part+column));
    return result;
}
} // namespace canonical_gu

/** @brief One immutable original router pair; no rounding or correction is cached. */
struct CanonicalRouterPacket {float2 hidden;__half2 weights;};
/** @brief Borrow one live even-column pair from original published sources. */
__device__ __forceinline__ CanonicalRouterPacket canonicalRouterPacket(const float* hidden,const __half2* weights,unsigned pair) {
 return {{hidden[2*pair],hidden[2*pair+1]},weights[pair]};
}
/** @brief Commit one original pair using the actual operand conversion and original expression.
 * @param packet Exact immutable FP32 hidden/FP16 weight pair. @param sum Original lane accumulator.
 */
__device__ __forceinline__ void canonicalRouterCommitPacket(const CanonicalRouterPacket& packet,float& sum) {
 const float2 ww=__half22float2(packet.weights);
 sum+=::llaminar2::rocm::router_native::router_fp16_operand(packet.hidden.x)*ww.x+
      ::llaminar2::rocm::router_native::router_fp16_operand(packet.hidden.y)*ww.y;
}
/** @brief Evaluate the exact current logical128 router lane before its original wave tree. */
template<bool Lookahead=false>
__device__ __forceinline__ float canonicalRouterPartial128(const float* hidden,const __half* gate_weights,
 int d_model,int num_experts,int expert,unsigned team_lane) {
 constexpr int TeamThreads=128;
 using llaminar2::rocm::router_native::router_fp16_operand;
 float sum=0.0f;
 if(expert<num_experts) {
  const __half* w=gate_weights+static_cast<size_t>(expert)*d_model;
  if((d_model&1)==0) {
   const int pair_count=d_model>>1;const __half2* w2=reinterpret_cast<const __half2*>(w);
   if constexpr(Lookahead) {
    for(int i=team_lane;i<pair_count;i+=2*TeamThreads) {
     const auto first=canonicalRouterPacket(hidden,w2,unsigned(i));
     const bool second_live=i+TeamThreads<pair_count;
     CanonicalRouterPacket second{};
     if(second_live)second=canonicalRouterPacket(hidden,w2,unsigned(i+TeamThreads));
     canonicalRouterCommitPacket(first,sum);
     if(second_live)canonicalRouterCommitPacket(second,sum);
    }
   } else {
   for(int i=team_lane;i<pair_count;i+=TeamThreads) {
    const float2 ww=__half22float2(w2[i]);const int h=i<<1;
    sum+=::llaminar2::rocm::router_native::router_fp16_operand(hidden[h])*ww.x+::llaminar2::rocm::router_native::router_fp16_operand(hidden[h+1])*ww.y;
   }
   }
  } else for(int i=team_lane;i<d_model;i+=TeamThreads)
   sum+=::llaminar2::rocm::router_native::router_fp16_operand(hidden[i])*__half2float(w[i]);
 }
 return sum;
}
/** @brief Map two original logical router waves onto one physical wave, retaining all three original trees. */
template<bool Lookahead=false>
__device__ __forceinline__ float canonicalRouter128(const float* hidden,const uint16_t* gate_weights,
 int width,int experts,int expert,unsigned lane) {
    using llaminar2::rocm::moe_wave_contract::rocm_moe_wave_reduce_sum_f32;
    float first=0.f,second=0.f;
    // One current logical-wave address site, then its completed root. Do not
    // retain both future hidden/weight address chains through the first wave.
#pragma unroll 1
    for(unsigned bank=0;bank<2;++bank) {
        float value=canonicalRouterPartial128<Lookahead>(hidden,reinterpret_cast<const __half*>(gate_weights),
            width,experts,expert,lane+64*bank);
        value=rocm_moe_wave_reduce_sum_f32(value);
        value=__builtin_bit_cast(float,__builtin_amdgcn_readlane(__builtin_bit_cast(int,value),0));
        if(bank==0)first=value;else second=value;
    }
    const float seed=lane==0?first:lane==1?second:0.f;
    return rocm_moe_wave_reduce_sum_f32(seed);
}
/** @brief Preserve the original256-logical-lane FP32 shared-gate dot through four virtual wave states. */
__device__ __forceinline__ float canonicalSharedGate256(const float* input,const float* weights,unsigned width,unsigned lane) {
    using llaminar2::rocm::moe_wave_contract::rocm_moe_wave_reduce_sum_f32;
    float first=0.f,second=0.f,third=0.f,fourth=0.f;
    // A common bank loop keeps only the current pair of float4 addresses live.
    // Each original logical thread and its original FMA sequence are unchanged.
#pragma unroll 1
    for(unsigned bank=0;bank<4;++bank) {
        float sum=0.f;
        const auto* a=reinterpret_cast<const float4*>(input);const auto* b=reinterpret_cast<const float4*>(weights);
        for(unsigned i=lane+bank*64;i<width/4;i+=256) {
            const float4 av=a[i],bv=b[i];sum+=av.x*bv.x+av.y*bv.y+av.z*bv.z+av.w*bv.w;
        }
        sum=rocm_moe_wave_reduce_sum_f32(sum);
        const float root=__builtin_bit_cast(float,__builtin_amdgcn_readlane(__builtin_bit_cast(int,sum),0));
        if(bank==0)first=root;else if(bank==1)second=root;else if(bank==2)third=root;else fourth=root;
    }
    const float seed=lane==0?first:lane==1?second:lane==2?third:lane==3?fourth:0.f;
    return rocm_moe_wave_reduce_sum_f32(seed);
}
/** @brief Append one current full-block affine gate/up pair with the sole production contribution primitives. */
__device__ __forceinline__ void canonicalQ4PairCommit(const NativeQKItem& gate,const NativeQKItem& up,
 const CanonicalFFNQ8Block& activation,float& g,float& u) {
    const unsigned gw[4]={unsigned(gate.q.x),unsigned(gate.q.y),unsigned(gate.q.z),unsigned(gate.q.w)};
    const unsigned uw[4]={unsigned(up.q.x),unsigned(up.q.y),unsigned(up.q.z),unsigned(up.q.w)};
    int gd=0,ud=0;
#pragma unroll
    for(unsigned group=0;group<8;++group) {
        const int value=activation.qs[group];
        llaminar2::rocm::q4PairedDotStep(value,gw[group&3],uw[group&3],(group>>2)*4,gd,ud);
    }
    const unsigned gc=unsigned(gate.high.x),uc=unsigned(up.high.x);
    // A corrected K32 block is one canonical publication: first round the
    // dot-plus-minimum contribution, then append it to the running partition.
    // Calling the original helper also retains its explicit FP32 boundaries.
    g=rocm_native_quantised_commit_corrected_single_scale_block(g,gd,
        __half2float(__ushort_as_half(uint16_t(gc))),activation.sum,
        __half2float(__ushort_as_half(uint16_t(gc>>16))),activation.scale);
    u=rocm_native_quantised_commit_corrected_single_scale_block(u,ud,
        __half2float(__ushort_as_half(uint16_t(uc))),activation.sum,
        __half2float(__ushort_as_half(uint16_t(uc>>16))),activation.scale);
}
/** @brief Shared Q8 role; only an intra-wave coordinate, never a source-format policy. */
enum class CanonicalSharedQ8Role : unsigned { Gate, Up };

/**
 * @brief Required original fields for one Q8 role, possibly selected per physical lane.
 * @details This view has no native_descriptor_fields/native_weight_uniform overload.
 * A selected view must not be rebroadcast; its own N/payload/scales remain authority.
 */
struct CanonicalSharedQ8Fields {
    unsigned columns; ///< This exact role's original output-column count.
    const uint8_t* payload; ///< This exact role's retained canonical Q8 payload.
    const unsigned short* scales; ///< This role's original half scale plane.
    /**
     * @brief Borrow an actual wave-uniform by-value descriptor before lane selection.
     * @param matrix Same immutable original kernarg descriptor for every active lane.
     * @return Its exact original fields; no weight representation or cache is made.
     * @pre The existing prepared owner/dispatch publication remains live through launch.
     */
    __device__ __forceinline__ static CanonicalSharedQ8Fields borrowUniform(const NativeMatrix& matrix) {
        CanonicalSharedQ8Fields fields;
        const unsigned short* unused_mins;
        native_descriptor_fields<false>(matrix,fields.columns,fields.payload,fields.scales,unused_mins);
        return fields;
    }
};

/**
 * @brief Borrow both shared role descriptors uniformly, then select only raw fields.
 * @details Gate/up may be different allocations and each keeps its own N. Both
 * descriptor borrows precede any row/P mask and use the original kernarg-reference
 * accessor. No lane-varying NativeMatrix is passed to a uniform-field helper.
 */
class CanonicalSharedQ8RolePair {
    CanonicalSharedQ8Fields gate_,up_;
public:
    /** @brief Bind the two original retained role authorities. @param gate Uniform gate. @param up Uniform up. */
    __device__ __forceinline__ CanonicalSharedQ8RolePair(const NativeMatrix& gate,const NativeMatrix& up)
        : gate_(CanonicalSharedQ8Fields::borrowUniform(gate)),up_(CanonicalSharedQ8Fields::borrowUniform(up)) {}
    /**
     * @brief Select exact per-lane fields without any readfirstlane operation.
     * @param role This lane's original gate/up role. @return The selected role's own fields.
     */
    __device__ __forceinline__ CanonicalSharedQ8Fields select(CanonicalSharedQ8Role role) const {
        const bool up=role==CanonicalSharedQ8Role::Up;
        return {up?up_.columns:gate_.columns,up?up_.payload:gate_.payload,up?up_.scales:gate_.scales};
    }
};

/**
 * @brief Execute the unchanged actual Q8 K32/P body from already-borrowed fields.
 * @param fields This lane's own N/payload/scale authority, without rebroadcast.
 * @param row Original real output row. @param part Original fixed arithmetic partition.
 * @param width Original K width. @param activation Original published canonical Q8 row.
 * @return The original ascending K32 partial, including original empty-P zero.
 */
__device__ __forceinline__ float canonicalSharedQ8Partial(const CanonicalSharedQ8Fields& fields,
 unsigned row,unsigned part,unsigned width,const CanonicalFFNQ8Block* activation) {
    const unsigned parts=llaminar2::MoEProjectionNumericalContract::orderedKPartitionsForWidth(int(width));
    if(part>=parts)return 0.f;
    const unsigned blocks=width/32,step=(blocks+parts-1)/parts;
    const unsigned columns=fields.columns;
    const uint8_t* payload=fields.payload;
    const unsigned short* scales=fields.scales;
    float result=0.f;
#pragma unroll 1
    for(unsigned block=part*step;block<blocks&&block<(part+1)*step;++block) {
        const size_t linear=size_t(block)*columns+row;
        const int4 lo=native_global_weight_read<int4>(payload+linear*32);
        const int4 hi=native_global_weight_read<int4>(payload+linear*32+16);
        const int weights[8]={lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w};
        const float scale=__half2float(__ushort_as_half(native_global_weight_read<unsigned short>(scales+linear)));
        int dot=0;
#pragma unroll
        for(unsigned group=0;group<8;++group)
            dot=llaminar2::rocm_integer_contract::signedDot4(activation[block].qs[group],weights[group],dot);
        result=llaminar2::device_fp32_contract::add(result,
            llaminar2::device_native_quantised_contract::singleScaleBlock(dot,scale,activation[block].scale));
    }
    return result;
}

/** @brief Preserve the original uniform single-role reader and empty-part guard. */
__device__ __forceinline__ float canonicalSharedQ8Partial(const NativeMatrix& matrix,unsigned row,unsigned part,
 unsigned width,const CanonicalFFNQ8Block* activation) {
    const unsigned parts=llaminar2::MoEProjectionNumericalContract::orderedKPartitionsForWidth(int(width));
    if(part>=parts)return 0.f;
    return canonicalSharedQ8Partial(CanonicalSharedQ8Fields::borrowUniform(matrix),row,part,width,activation);
}

/**
 * @brief Complete one full-wave two-row/two-role task with the original P order.
 * @param partial This lane's original row/role/P result (zero for an absent row).
 * @param parts Actual fixed arithmetic partition count, from one through sixteen.
 * @param lane Physical wave64 lane. @param paired Uniform presence of two real rows.
 * @return Gate/up roots for writer lanes0/1 (only lane0 for an odd tail).
 * @pre All64 lanes call after reconvergence, including inactive row/P producers.
 * @details The same helper is used by the full entry and its native differential.
 * It creates no publication edge; callers publish real roots before grid entry.
 */
__device__ __forceinline__ float2 canonicalSharedQ8Fold(float partial,unsigned parts,unsigned lane,bool paired) {
    if(paired) {
        const float gate=canonical_gu::foldFourColumns(partial,parts,lane);
        // Roots0/1 gate,2/3 up: both consumers stay in the first16-lane row.
        const float up=::llaminar2::rocm::moe_wave_contract::rowShiftNativeBits<2>(gate);
        return {gate,up};
    }
    // Preserve original odd-tail scalar-source READLANE machinery. Its native
    // ADD priority differs from the paired vector-source shape; the same P
    // values now occupy columns0/2 at stride4. Native NaN witnesses gate this.
    const float gate=canonical_gu::foldStrided(partial,parts,0,4);
    const float up=canonical_gu::foldStrided(partial,parts,2,4);
    return {gate,up};
}
