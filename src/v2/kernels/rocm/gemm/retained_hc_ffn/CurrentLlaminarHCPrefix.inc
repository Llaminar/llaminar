/**
 * @file CurrentLlaminarHCPrefix.inc
 * @brief Current HC arithmetic inside the tuned diagnostic full-entry layout.
 *
 * This explicitly pinned H2560/R320/P16 ordinary test implementation is not a
 * production geometry selector. It borrows the canonical write/norm, SingleQ8,
 * K32 contribution and TwoTerm publication bodies. Only native issue ownership,
 * scratch lifetime and fusion are changed; the unchanged suffix follows mixed.
 * Explicit ModuleGlobal mode owns its module state; Instance mode borrows one
 * PMA allocation sealed through exact captured retirement. Numerical helpers
 * are shared by both modes.
 */

/** @brief Complete current-prefix banks, independent from immutable source weights. */
struct CurrentHCPrefixState {
    signed char norm_codes[TMAX][K];
    float norm_scales[TMAX][QB];
    float raw_down[TMAX][LR],activated[TMAX][LR];
    signed char primary[TMAX][LR],residual[TMAX][LR];
    float primary_scales[TMAX][LR/32],residual_scales[TMAX][LR/32];
    float up_logits[TMAX][K];
};

/**
 * @brief Bind only the two existing MUL operand roles to the frozen original gfx906 program.
 * @details Both the frozen full prefix and its literal oracle emit inverse*POST,
 * then intermediate*gain. A compiler commute changed competing NaN payloads in
 * the rejected register candidate. These single-instruction products preserve
 * those actual priorities without extra arithmetic, memory fences or NaN repair.
 */
struct CurrentHCOriginalNormalizedProducts {
 /** @brief Original first MUL instruction. @param inverse Original inverse. @param post Original POST. @return RN(inverse*post) with frozen native source roles. */
 __device__ __forceinline__ float inversePost(float inverse,float post) const {
  float value;
  asm volatile("v_mul_f32 %0, %1, %2" : "=v"(value) : "v"(inverse),"v"(post));
  return value;
 }
 /** @brief Original second MUL instruction. @param value Original inner result. @param gain Original gain. @return RN(value*gain) with frozen native source roles. */
 __device__ __forceinline__ float appendGain(float value,float gain) const {
  float normalized;
  asm volatile("v_mul_f32 %0, %1, %2" : "=v"(normalized) : "v"(value),"v"(gain));
  return normalized;
 }
};
/** @brief Original instance's normalized Q8 destination, with no address held across the reduction. */
struct CurrentHCGlobalNormQ8Bank {
 unsigned token,branch;
 CurrentHCPrefixState* state; ///< Original task coordinates.
 /** @return Existing code block base for this original row/branch. */
 __device__ __forceinline__ signed char* codes() const {return state->norm_codes[token]+branch*E;}
 /** @return Existing scale block base for this original row/branch. */
 __device__ __forceinline__ float* scales() const {return state->norm_scales[token]+branch*(E/32);}
};
/**
 * @brief Adjacent-four normalized output and exact Q8 publication in registers.
 * @tparam Bank Typed destination-only binding; never a second arithmetic implementation.
 * @details This explicit pinned policy changes output ownership only. All1024
 * physical lanes enter after the original logical256 square tree. Each of640
 * live quad owners reads the COMPLETED total, evaluates the same inverse and
 * two original multiplies per channel, publishes required global normalized
 * words, then calls the unchanged eight-lane Q8 helper on its register values.
 * POST/gain LDS remains immutable through this policy and its enclosing join.
 */
template<class Bank,bool NativeHalfScale>
struct CurrentHCRegisterNormQ8Output {
 Bank bank; ///< Typed destination binding only; numerical body is shared with diagnostics.
 /**
  * @brief Consume the original completed square total and retain rounded float4 values.
  * @tparam Input Original retained POST view. @tparam Output Original public output pointer.
  * @param input Stable original POST. @param branch_gain Immutable staged gains.
  * @param output Original full normalized output. @param hidden Original branch width.
  * @param epsilon Original epsilon. @param base Original task-major channel base.
  * @param logical_sum Original per-member return; deliberately not used by wider owners.
  * @param completed_total Original256-float plane after its final all-CTA join.
  * @param lane Physical thread0..1023. @param member Original reduction membership, unchanged.
  */
 template<class Input,class Output>
 __device__ __forceinline__ void publish(const Input& input,const float* branch_gain,Output output,
  int hidden,float epsilon,std::uint64_t base,float logical_sum,const float* completed_total,
  unsigned lane,bool member) const {
  (void)logical_sum;(void)member;
  static_assert(E%32==0 && NT==1024,"Register norm Q8 owns complete original32-value cohorts");
  using namespace ::llaminar2::qwen4exp::branch_norm_native;
  for(unsigned quad=lane;quad<unsigned(hidden/4);quad+=NT) {
   // Wider owners explicitly consume the completed plane: squareSum continues
   // returning0 to nonmembers. The original inverse expression is evaluated
   //640 rather than256 times perbranch, an intentional measured work tradeoff.
   const float inverse=normalizationInverse(completed_total[0],hidden,epsilon);
   const unsigned first=4*quad;
   const float4 value{
    normalizedElement(input.retained(first),inverse,branch_gain[first],CurrentHCOriginalNormalizedProducts{}),
    normalizedElement(input.retained(first+1),inverse,branch_gain[first+1],CurrentHCOriginalNormalizedProducts{}),
    normalizedElement(input.retained(first+2),inverse,branch_gain[first+2],CurrentHCOriginalNormalizedProducts{}),
    normalizedElement(input.retained(first+3),inverse,branch_gain[first+3],CurrentHCOriginalNormalizedProducts{})};
   // Scalar original publication imposes no new alignment on public outputs.
   // Only phase-owned POST/gain LDS is staged; no normalized LDS store/reload.
   publishNormalizedValue(output,base+first,value.x);
   publishNormalizedValue(output,base+first+1,value.y);
   publishNormalizedValue(output,base+first+2,value.z);
   publishNormalizedValue(output,base+first+3,value.w);
   const unsigned block=quad/8;
   current_hc_norm_q8_four<NativeHalfScale>(value,bank.codes()+block*32,bank.scales()+block,quad%8);
  }
 }
};

/** @brief Actual bank-only leaf consumed by the canonical publication body. */
struct CurrentHCPublishValues {
    float* activated;
    signed char *primary,*residual;
    float *primaryScales,*residualScales;
};

/**
 * @brief One complete native HC-down output interval for this physical CTA.
 * @details Empty norm owners hold no down column; they still enter every
 * original local/grid join. This view is coordinates only, never a counter or
 * host mirror. Every real column retains its entire original K32/P arithmetic.
 */
struct CurrentHCDownOwnership {
    unsigned first; ///< First real output column, or unused0 for an empty owner.
    unsigned columns; ///< Exact live column count; zero must not issue a load.
};
/**
 * @brief Assign complete down columns only to CTAs without original norm work.
 * @tparam NormOwners Original T*HC norm-task extent, explicitly pinned by caller.
 * @param bid Original physical CTA index in0..NB-1.
 * @return Unique cardinal interval among the remaining NB-NormOwners producers.
 * @details The same helper controls immutable packet issue and every later
 * consume/fold/store. It introduces no storage, packet, wait or arithmetic node.
 * Coordinate division/modulo by the returned count remains inside live guards.
 */
template<unsigned NormOwners>
__device__ __forceinline__ CurrentHCDownOwnership current_hc_down_ownership(unsigned bid) {
    static_assert(NormOwners>0 && NormOwners<NB,"HC-down requires a nonempty producer cohort");
    if(bid<NormOwners)return {0,0};
    const unsigned producer=bid-NormOwners;
    const unsigned first=producer*LR/(NB-NormOwners);
    return {first,(producer+1)*LR/(NB-NormOwners)-first};
}

/** @brief One immutable actual native Q8 packet; no ready-input fence or arithmetic. */
struct CurrentHCNativeQ8Packet {
    int4 low,high;
    unsigned scale_bits;
};
/**
 * @brief Issue one real native packet through compiler-visible global loads.
 * @param weights Original immutable prepared matrix. @param column Owned output column.
 * @param block Original K32 coordinate. @return Exact eight payload words and raw half bits.
 * @details Raw coefficient bits are not converted, packed or consumed here.
 * The existing grid entry wait retires idle-CTA prefetches before arrival.
 */
__device__ __forceinline__ CurrentHCNativeQ8Packet current_hc_load_packet(
    const NativeMatrix& weights,unsigned column,unsigned block) {
    const auto linear=std::size_t(block)*unsigned(weights.n)+column;
    const auto* payload=weights.payload+linear*32;
    return {native_global_weight_read<int4>(payload),native_global_weight_read<int4>(payload+16),
        unsigned(native_global_weight_read<unsigned short>(static_cast<const unsigned short*>(weights.scales)+linear))};
}
/**
 * @brief Consume the original native words with the sole original SDOT/FP commit.
 * @param packet Actual packet, completed before any dependent use.
 * @param input Exact globally published Q8 code block. @param scale Original activation scale.
 * @return The unchanged rounded K32 contribution.
 */
__device__ __forceinline__ float current_hc_consume_packet(
    const CurrentHCNativeQ8Packet& packet,const signed char* input,float scale) {
    const int4 a=packet.low,b=packet.high;
    const int w[8]{a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
    const auto* x=reinterpret_cast<const int*>(input);
    int dot=0;
#pragma unroll
    for(unsigned word=0;word<8;++word)
        dot=__builtin_amdgcn_sdot4(x[word],w[word],dot,false);
    const float weight_scale=__half2float(__ushort_as_half(static_cast<unsigned short>(packet.scale_bits)));
    return ::llaminar2::device_native_quantised_contract::singleScaleBlock(
        dot,weight_scale,scale);
}
/**
 * @brief Retain the same original one-packet API for other actual consumers.
 * @param weights Original immutable prepared source. @param column Owned output.
 * @param block Original K32 coordinate. @param input Published activation codes.
 * @param scale Original activation scale. @return Original rounded K32 commit.
 */
__device__ __forceinline__ float current_hc_contribution(
    const NativeMatrix& weights,unsigned column,unsigned block,
    const signed char* input,float scale) {
    const auto packet=current_hc_load_packet(weights,column,block);
    return current_hc_consume_packet(packet,input,scale);
}

/**
 * @brief Two original independent K32 commits sharing one actual weight packet.
 * @param weights Prepared Q8 source. @param column Actual output column.
 * @param block Increasing original K32 index. @param primary Original Q8 bank.
 * @param residual Exact rounded residual bank. @param ps Primary scale.
 * @param rs Residual scale. @param p Receives original primary commit.
 * @param q Receives original residual commit.
 */
__device__ __forceinline__ void current_hc_two_contributions(
    const NativeMatrix& weights,unsigned column,unsigned block,
    const signed char* primary,const signed char* residual,float ps,float rs,
    float& p,float& q) {
    const auto linear=std::size_t(block)*unsigned(weights.n)+column;
    const auto* payload=weights.payload+linear*32;
    const int4 a=native_global_weight_read<int4>(payload);
    const int4 b=native_global_weight_read<int4>(payload+16);
    const int w[8]{a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
    const auto* x=reinterpret_cast<const int*>(primary);
    const auto* y=reinterpret_cast<const int*>(residual);
    int dp=0,dq=0;
#pragma unroll
    for(unsigned word=0;word<8;++word) {
        dp=__builtin_amdgcn_sdot4(x[word],w[word],dp,false);
        dq=__builtin_amdgcn_sdot4(y[word],w[word],dq,false);
    }
    const auto bits=native_global_weight_read<unsigned short>(
        static_cast<const unsigned short*>(weights.scales)+linear);
    const float ws=__half2float(__ushort_as_half(bits));
    p=::llaminar2::device_native_quantised_contract::singleScaleBlock(dp,ws,ps);
    q=::llaminar2::device_native_quantised_contract::singleScaleBlock(dq,ws,rs);
}

/**
 * @brief Pinned current HC prefix with balanced native-column issue.
 * @tparam T Exact retained diagnostic token rows, never an inferred live count.
 * @tparam PublishedTrunk Read the norm producer's published trunk without owning a POST write.
 * @tparam Precision Explicit SingleQ8 or TwoTermQ8 operation.
 * @param src Original POST input view. @param gain Original FP32 norm weights.
 * @param down Prepared Q8 down source. @param up Prepared Q8 up source.
 * @param epsilon Original RMS epsilon. @param post Original written public plane.
 * @param normalized Original normalized public plane. @param mixed Mixed output.
 * @param temporary Existing phase arena; holds 256 reduction floats, E staged
 * POST floats and E immutable gain floats during norm, then the original up
 * contribution planes. The generated caller proves capacity; lifetimes do not overlap.
 * @param up_storage Existing lo2 phase storage, reused after down publication.
 * @param gates Existing per-row/branch/owned-column LDS for final mix.
 * @param observation Complete caller-owned untimed HC output binding.
 * @tparam Observe Compile-time diagnostic publication; false emits no observation traffic.
 * @details Four explicit grid publications: norm+Q8, down partials, TwoTerm
 * banks, mixed. All physical lanes reach all CTA joins. One whole-CTA panel
 * publishes actual K32 contributions; output owners then fold both banks in registers.
 */
template<int T,::llaminar2::QuantizedActivationOperandPrecision Precision,bool Observe,bool TileEightHCUp,bool PublishedTrunk=false>
__device__ __forceinline__ void current_llaminar_hc_prefix(
    const hc_src& src,const float* gain,const NativeMatrix& down,const NativeMatrix& up,
    float epsilon,float* post,float* normalized,float* mixed,float* temporary,
    float* up_storage,float (*gates)[HC][64],const qfn_castagna_native::CanonicalPortObservation& observation,CurrentHCPrefixState& g_current_hc,auto& join,bool live=true) {
    static_assert(E==2560 && K==10240 && LR==320 && NT==1024 && NB==60);
    static_assert(T>0 && T<=TMAX);
    static_assert(Precision==::llaminar2::QuantizedActivationOperandPrecision::SingleQ8 ||
                  Precision==::llaminar2::QuantizedActivationOperandPrecision::TwoTermQ8);
    constexpr unsigned P=16,Per=(QB+P-1)/P;
    const unsigned tid=threadIdx.x,bid=blockIdx.x;
    // The private ordinary entry has at most two real packets per lane.
    // Every down producer has no norm task and issues its real packets here.
    // The four norm owners retain their source/gain path and own no down column;
    // they still participate in every original local and grid publication.
    static_assert(T==1,"Idle down prefetch is qualified only for this ordinary-row entry");
    static_assert(P*Per*((LR+(NB-T*HC)-1)/(NB-T*HC))<=2*NT,"Two register packets must cover all actual down producers");
    // Cardinal cuts of LR columns among the non-norm CTAs have just two
    // widths. Their invariant multipliers give the exact quotient of every
    // live flat index. The error of ceil(65536 / width) is <1 per product
    // term, and the bound below keeps the excess below one quotient step.
    const auto columnBlock = [](unsigned flat, unsigned columns) {
        if constexpr (!TileEightHCUp) return flat / columns;
        constexpr unsigned owners = NB - T*HC;
        constexpr unsigned narrow = LR / owners, wide = (LR + owners - 1) / owners;
        constexpr unsigned unit = 1u << 16;
        static_assert(narrow > 0 && std::uint64_t(2)*NT*wide < unit &&
                      std::uint64_t(2)*NT*(unit/narrow + 1u) <= 0xffffffffu);
        const unsigned divisor = columns == narrow ? (unit + narrow - 1)/narrow : (unit + wide - 1)/wide;
        return (flat * divisor) >> 16;
    };
    const bool norm_owner=bid<unsigned(T*HC);
    CurrentHCNativeQ8Packet first_packet{},second_packet{};
    if(!norm_owner) {
        const auto ownership=current_hc_down_ownership<T*HC>(bid);
        const unsigned owned_first=ownership.first,owned_columns=ownership.columns;
        const unsigned down_items=P*Per*owned_columns;
        if(tid<down_items)first_packet=current_hc_load_packet(down,
            owned_first+(tid-columnBlock(tid,owned_columns)*owned_columns),columnBlock(tid,owned_columns));
        if(tid+NT<down_items)second_packet=current_hc_load_packet(down,
            owned_first+((tid+NT)-columnBlock((tid+NT),owned_columns)*owned_columns),columnBlock((tid+NT),owned_columns));
    }
    // All 1,024 physical workers issue the original immutable source/gain
    // reads. Scalar channels require no stronger alignment than the original
    // FP32 inputs. PreviousWrite publishes every POST word once and keeps
    // exact in-place writes same-channel owned. PublishedTrunk only reads
    // the existing publication. Both materialize the same local norm input.
    for(unsigned task=bid;task<unsigned(T*HC);task+=NB) {
        const unsigned t=task/HC,c=task%HC;
        const auto base=std::uint64_t(task)*E;
        if(!live) {
            // The counted original keeps POST and normalized outputs at count0.
            // Down still receives a complete Q8 operand from retained normalized values.
            const CurrentHCGlobalNormQ8Bank bank{t,c,&g_current_hc};
            for(unsigned quad=tid;quad<E/4;quad+=NT) {
                const auto first=base+4u*quad;
                const float4 value{normalized[first],normalized[first+1],normalized[first+2],normalized[first+3]};
                current_hc_norm_q8_four<TileEightHCUp>(value,bank.codes()+32u*(quad/8),bank.scales()+quad/8,quad%8);
            }
            lds_sync();
            continue;
        }
        float* norm_local=temporary+256;
        float* gain_local=norm_local+E;
        if constexpr(PublishedTrunk) {
            const ::llaminar2::qwen4exp::branch_norm_native::RetainedBranchInput input{src.x+base};
            for(unsigned channel=tid;channel<unsigned(E);channel+=NT) {
                norm_local[channel]=input.publish(channel);
                gain_local[channel]=gain[c*E+channel];
            }
        } else {
            const float gate=::llaminar2::qwen4exp::branch_norm_native::residualGate(src.raw[task]);
            const ::llaminar2::qwen4exp::branch_norm_native::ResidualBranchInput input{
                src.x+base,src.out+std::uint64_t(t)*E,post+base,gate};
            for(unsigned channel=tid;channel<unsigned(E);channel+=NT) {
                const float original_gain=gain[c*E+channel];
                const float value=input.publish(channel);
                norm_local[channel]=value;
                gain_local[channel]=original_gain;
            }
        }
        // Every physical lane reaches this new publication edge. The original
        // 256 logical owners then consume their unchanged channel+256*j FMA
        // sequence and squareSum tree from the identical staged POST words.
        lds_sync();
        ::llaminar2::qwen4exp::branch_norm_native::taskFromBranchInput(
            ::llaminar2::qwen4exp::branch_norm_native::RetainedBranchInput{norm_local},
            gain_local,normalized,E,epsilon,task,temporary,tid,tid<256,
            CurrentHCRegisterNormQ8Output<CurrentHCGlobalNormQ8Bank,TileEightHCUp>{{t,c,&g_current_hc}});
        // Keep this original post-task CTA edge in the initial experiment.
        // All normalized/Q8 values now come from registers; the retained POST
        // and gain plane is retired before any following task/arena reuse.
        lds_sync();
        // A following task may reuse both norm/reduction intervals. In the
        // current T1 tuple each producer has one task, so no tail join executes.
        if(task+NB<unsigned(T*HC))lds_sync();
    }
    grid_sync_all(join);

    // Each CTA owns complete output columns, including every original P/K32
    // leaf. Cardinal intervals cover live columns once without clamped work.
    const auto ownership=current_hc_down_ownership<T*HC>(bid);
    const unsigned owned_first=ownership.first,owned_columns=ownership.columns;
    const unsigned down_items=P*Per*owned_columns;
    const bool first_live=tid<down_items,second_live=tid+NT<down_items;
    // Complete original VM0/LGKM0, arrival/poll/invalidation and both CTA
    // joins above are unchanged. Every real producer's packets were prefetched;
    // empty norm owners consume nothing and still reach both local joins below.
    for(unsigned t=0;t<unsigned(T);++t) {
        float* partition_values=temporary+P*Per*owned_columns;
        // These are precisely the two original strided items, with the same
        // live guards. Activation pointers/scales are read only after norm's
        // complete grid publication, never by the immutable prefetch stage.
        if(first_live) {
            const unsigned leaf=columnBlock(tid,owned_columns);
            temporary[tid]=current_hc_consume_packet(first_packet,
                g_current_hc.norm_codes[t]+leaf*32,g_current_hc.norm_scales[t][leaf]);
        }
        if(second_live) {
            const unsigned leaf=columnBlock((tid+NT),owned_columns);
            temporary[tid+NT]=current_hc_consume_packet(second_packet,
                g_current_hc.norm_codes[t]+leaf*32,g_current_hc.norm_scales[t][leaf]);
        }
        lds_sync();
        for(unsigned owner=tid;owner<P*owned_columns;owner+=NT) {
            const unsigned partition=columnBlock(owner,owned_columns),column_local=(owner-columnBlock(owner,owned_columns)*owned_columns);
            const float* leaves=temporary+partition*Per*owned_columns;
            float sum=0.0f;
            // Issue eight independent LDS leaves before consuming any of them.
            // The register-only dependency prevents the compiler from sinking
            // each read beneath the preceding rounded addition. This changes
            // the memory schedule, never the original increasing-K32 fold.
#pragma unroll 1
            for(unsigned group=0;group<Per/8;++group) {
                const unsigned first=group*8;
                float a=leaves[(first+0)*owned_columns+column_local];
                float b=leaves[(first+1)*owned_columns+column_local];
                float c=leaves[(first+2)*owned_columns+column_local];
                float d=leaves[(first+3)*owned_columns+column_local];
                float e=leaves[(first+4)*owned_columns+column_local];
                float f=leaves[(first+5)*owned_columns+column_local];
                float g=leaves[(first+6)*owned_columns+column_local];
                float h=leaves[(first+7)*owned_columns+column_local];
                asm volatile("" : "+v"(a), "+v"(b), "+v"(c), "+v"(d),
                    "+v"(e), "+v"(f), "+v"(g), "+v"(h));
                sum=::llaminar2::device_fp32_contract::add(sum,a);
                sum=::llaminar2::device_fp32_contract::add(sum,b);
                sum=::llaminar2::device_fp32_contract::add(sum,c);
                sum=::llaminar2::device_fp32_contract::add(sum,d);
                sum=::llaminar2::device_fp32_contract::add(sum,e);
                sum=::llaminar2::device_fp32_contract::add(sum,f);
                sum=::llaminar2::device_fp32_contract::add(sum,g);
                sum=::llaminar2::device_fp32_contract::add(sum,h);
            }
            // Keep the already-proven four-leaf tail batch when present.
            // In the current Per20 this avoids four serialized scalar reads
            // after the two eight-leaf groups. Only fewer than four leaves
            // use the final scalar tail; no unused leaf is ever loaded.
            for(unsigned first=(Per/8)*8;first+4<=Per;first+=4) {
                float a=leaves[(first+0)*owned_columns+column_local];
                float b=leaves[(first+1)*owned_columns+column_local];
                float c=leaves[(first+2)*owned_columns+column_local];
                float d=leaves[(first+3)*owned_columns+column_local];
                asm volatile("" : "+v"(a), "+v"(b), "+v"(c), "+v"(d));
                sum=::llaminar2::device_fp32_contract::add(sum,a);
                sum=::llaminar2::device_fp32_contract::add(sum,b);
                sum=::llaminar2::device_fp32_contract::add(sum,c);
                sum=::llaminar2::device_fp32_contract::add(sum,d);
            }
            for(unsigned i=(Per/4)*4;i<Per;++i)
                sum=::llaminar2::device_fp32_contract::add(sum,leaves[i*owned_columns+column_local]);
            partition_values[owner]=__fmul_rn(sum,1.0f);
        }
        lds_sync();
        if(tid<owned_columns) {
            float raw=0.0f;
            // Prefetch independent completed partitions in bounded groups. Only
            // their loads overlap; the original P0..P15 rounded adds stay serial.
    #pragma unroll 1
            for(unsigned group=0;group<P/8;++group) {
                const unsigned part=group*8;
                float a=partition_values[(part+0)*owned_columns+tid];
                float b=partition_values[(part+1)*owned_columns+tid];
                float c=partition_values[(part+2)*owned_columns+tid];
                float d=partition_values[(part+3)*owned_columns+tid];
                float e=partition_values[(part+4)*owned_columns+tid];
                float f=partition_values[(part+5)*owned_columns+tid];
                float g=partition_values[(part+6)*owned_columns+tid];
                float h=partition_values[(part+7)*owned_columns+tid];
                asm volatile("" : "+v"(a), "+v"(b), "+v"(c), "+v"(d),
                    "+v"(e), "+v"(f), "+v"(g), "+v"(h));
                raw=::llaminar2::device_fp32_contract::add(raw,a);
                raw=::llaminar2::device_fp32_contract::add(raw,b);
                raw=::llaminar2::device_fp32_contract::add(raw,c);
                raw=::llaminar2::device_fp32_contract::add(raw,d);
                raw=::llaminar2::device_fp32_contract::add(raw,e);
                raw=::llaminar2::device_fp32_contract::add(raw,f);
                raw=::llaminar2::device_fp32_contract::add(raw,g);
                raw=::llaminar2::device_fp32_contract::add(raw,h);
            }
            for(unsigned part=(P/8)*8;part<P;++part)
                raw=::llaminar2::device_fp32_contract::add(raw,
                    partition_values[part*owned_columns+tid]);
            g_current_hc.raw_down[t][owned_first+tid]=raw;
        }
        // A following token reuses the same leaf and partition intervals.
        if(t+1<unsigned(T))lds_sync();
    }
    // Completed raw outputs, rather than P partials, are the sole inter-CTA
    // publication. Every later local publisher consumes these exact bits.
    grid_sync_all(join);

    if(!live) {
        // Up and activation retain their public values. Finish the same prefix
        // publication protocol before the caller processes the inactive suffix.
        grid_sync_all(join);
        if(observation.hc_down&&bid==0&&tid<LR)
            observation.hc_down[tid]=g_current_hc.raw_down[0][tid];
        return;
    }
    const unsigned e0=bid*E/NB,n=(bid+1)*E/NB-e0;
    for(unsigned t=0;t<unsigned(T);++t) {
        auto* primary=reinterpret_cast<signed char*>(up_storage);
        auto* residual=primary+LR;
        auto* ps=reinterpret_cast<float*>(residual+LR);
        auto* rs=ps+LR/32;
        // Every CTA owns an exact local activation bank. The preceding
        // raw-down grid edge published the exact original two-level fold.
        // No CTA consumes another CTA's freshly quantized output, so only a
        // local publication edge is needed before the unchanged up issue loop.
        static_assert(LR%32==0,"HC local publication retains complete quantizer cohorts");
        float* activated_local=temporary;
        if(tid<LR) {
            const unsigned column=tid;
            const float raw=g_current_hc.raw_down[t][column];
            const CurrentHCPublishValues output{activated_local,primary,residual,ps,rs};
            // The closed gfx906 entry has the qualified native half mode.
            if constexpr(TileEightHCUp)
                ::llaminar2::hc_down_publication::publish(raw,output,column,
                    ::llaminar2::hc_down_publication::NativePositiveHalfScalePublication{});
            else
                ::llaminar2::hc_down_publication::publish(raw,output,column);
        }
        lds_sync();
        if constexpr(Observe) {
            // Retain the original diagnostic/public bank meanings with one
            // writer. Retire this local read before the up panel reuses arena.
            if(bid==0) {
                if(tid<LR) {
                    g_current_hc.activated[t][tid]=activated_local[tid];
                    g_current_hc.primary[t][tid]=primary[tid];
                    g_current_hc.residual[t][tid]=residual[tid];
                }
                if(tid<LR/32) {
                    g_current_hc.primary_scales[t][tid]=ps[tid];
                    g_current_hc.residual_scales[t][tid]=rs[tid];
                }
            }
            lds_sync();
        }
        if constexpr(TileEightHCUp) {
            // Each consecutive five-block segment keeps its original root.
            // Whole-wave transfer pairs adjacent groups of eight lanes, including
            // tail lanes, while only live columns issue projection loads.
            // Preserve accumulator-first and final primary-first native source
            // roles so distinct NaN payloads keep the original priority.
            const auto ordered_add=[] __device__ (float previous,float leaf){
                float value;
                asm("v_add_f32 %0, %1, %2" : "=v"(value) : "v"(previous), "v"(leaf));
                return value;
            };
            constexpr unsigned width=2;
            constexpr unsigned neighbors=64/width;
            constexpr unsigned groups=LR/32;
            constexpr unsigned chunk=(groups+width-1)/width;
            const unsigned columns=HC*n,physical=tid%64;
            constexpr unsigned tile=8;
            const unsigned part=(physical/tile)%2;
            const unsigned first_ci=(tid/64)*neighbors+(physical/(2*tile))*tile+physical%tile;
            for(unsigned base=0;base<columns;base+=NT/width) {
                const unsigned ci=base+first_ci;
                const bool live_ci=ci<columns;
                const unsigned c=ci/n,i=ci%n,column=c*E+e0+i;
                float cp[chunk]{},cq[chunk]{};
#pragma unroll
                for(unsigned j=0;j<chunk;++j) {
                    const unsigned block=part*chunk+j;
                    if(live_ci && block<groups) {
                        if constexpr(Precision==::llaminar2::QuantizedActivationOperandPrecision::TwoTermQ8)
                            current_hc_two_contributions(up,column,block,primary+block*32,residual+block*32,
                                ps[block],rs[block],cp[j],cq[j]);
                        else cp[j]=current_hc_contribution(up,column,block,primary+block*32,ps[block]);
                    }
                }
                float p=0.0f,q=0.0f;
#pragma unroll
                for(unsigned segment=0;segment<(groups+chunk-1)/chunk;++segment) {
                    if(segment==part) {
#pragma unroll
                        for(unsigned j=0;j<chunk && segment*chunk+j<groups;++j) {
                            p=ordered_add(p,cp[j]);
                            if constexpr(Precision==::llaminar2::QuantizedActivationOperandPrecision::TwoTermQ8)
                                q=ordered_add(q,cq[j]);
                        }
                    }
                    if constexpr(width>1) {
                        if(segment+1<(groups+chunk-1)/chunk){
                            p=::llaminar2::rocm::moe_wave_contract::xor32NativeBits<tile>(p);
                            if constexpr(Precision==::llaminar2::QuantizedActivationOperandPrecision::TwoTermQ8)
                                q=::llaminar2::rocm::moe_wave_contract::xor32NativeBits<tile>(q);
                        }
                    }
                }
                if(live_ci && part==(groups-1)/chunk) {
                    const float value=Precision==::llaminar2::QuantizedActivationOperandPrecision::TwoTermQ8?
                        ordered_add(p,q):p;
                    if constexpr(Observe)g_current_hc.up_logits[t][column]=value;
                    gates[t][c][i]=value;
                }
            }
        } else {
            // The actual local output count, not a padded power-of-two envelope,
            // determines this complete K32 contribution panel. Every wave issues
            // adjacent native output columns; each current weight packet serves
            // both activation chains. The retired Q4 arena bounds both planes.
            const unsigned columns=HC*n,groups=LR/32;
            float* primary_partials=temporary;
            float* residual_partials=temporary+groups*columns;
            for(unsigned task=tid;task<groups*columns;task+=NT) {
                const unsigned block=task/columns,ci=task%columns;
                const unsigned c=ci/n,i=ci%n,column=c*E+e0+i;
                float pc,qc;
                if constexpr(Precision==::llaminar2::QuantizedActivationOperandPrecision::TwoTermQ8)
                    current_hc_two_contributions(up,column,block,primary+block*32,residual+block*32,
                        ps[block],rs[block],pc,qc);
                else {pc=current_hc_contribution(up,column,block,primary+block*32,ps[block]);qc=0.0f;}
                primary_partials[task]=pc;
                if constexpr(Precision==::llaminar2::QuantizedActivationOperandPrecision::TwoTermQ8)
                    residual_partials[task]=qc;
            }
            // Only LDS values cross this edge. Immutable weight loads must be ready
            // before their dependent stores; no unrelated VMEM drain is required.
            lds_sync();
            for(unsigned ci=tid;ci<columns;ci+=NT) {
                const unsigned c=ci/n,i=ci%n,column=c*E+e0+i;
                float p=0.0f,q=0.0f;
                for(unsigned block=0;block<groups;++block) {
                    p=::llaminar2::device_fp32_contract::add(p,primary_partials[block*columns+ci]);
                    if constexpr(Precision==::llaminar2::QuantizedActivationOperandPrecision::TwoTermQ8)
                        q=::llaminar2::device_fp32_contract::add(q,residual_partials[block*columns+ci]);
                }
                const float value=Precision==::llaminar2::QuantizedActivationOperandPrecision::TwoTermQ8?
                    ::llaminar2::device_fp32_contract::add(p,q):p;
                if constexpr(Observe)g_current_hc.up_logits[t][column]=value;
                gates[t][c][i]=value;
            }
        }
        lds_sync();
        if(tid<n) {
            float value=0.0f;
#pragma unroll
            for(unsigned c=0;c<HC;++c) {
                const float gate=1.0f/(1.0f+expf(-gates[t][c][tid]));
                value=fmaf(gate,normalized[t*K+c*E+e0+tid],value);
            }
            mixed[t*E+e0+tid]=value*0.25f;
        }
        // The next token begins with a full LDS publication before panel reuse;
        // the final token is retired by the unchanged full-grid edge below.
    }
    grid_sync_all(join);
    // Raw down is an original mandatory public output, independent of
    // optional observation. Its completed values retired at the prefix join.
    if constexpr(!Observe) if(observation.hc_down)
        for(unsigned i=bid*NT+tid;i<unsigned(LR);i+=NB*NT)
            observation.hc_down[i]=g_current_hc.raw_down[0][i];
    if constexpr(Observe) {
        static_assert(T==1,"Current observation ABI explicitly owns one ordinary row");
        for(unsigned i=bid*NT+tid;i<unsigned(K);i+=NB*NT) {
            if(observation.norm_codes)observation.norm_codes[i]=g_current_hc.norm_codes[0][i];
            if(observation.hc_up)observation.hc_up[i]=g_current_hc.up_logits[0][i];
        }
        for(unsigned i=bid*NT+tid;i<unsigned(QB);i+=NB*NT)
            if(observation.norm_scales)observation.norm_scales[i]=g_current_hc.norm_scales[0][i];
        for(unsigned i=bid*NT+tid;i<unsigned(LR);i+=NB*NT) {
            if(observation.hc_down)observation.hc_down[i]=g_current_hc.raw_down[0][i];
            if(observation.hc_activated)observation.hc_activated[i]=g_current_hc.activated[0][i];
            if(observation.up_primary_codes)observation.up_primary_codes[i]=g_current_hc.primary[0][i];
            if(observation.up_residual_codes)observation.up_residual_codes[i]=g_current_hc.residual[0][i];
        }
        for(unsigned i=bid*NT+tid;i<unsigned(LR/32);i+=NB*NT) {
            if(observation.up_primary_scales)observation.up_primary_scales[i]=g_current_hc.primary_scales[0][i];
            if(observation.up_residual_scales)observation.up_residual_scales[i]=g_current_hc.residual_scales[0][i];
        }
    }
}
