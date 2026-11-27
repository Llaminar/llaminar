/**
 * @file CPUQwenSparseExecution.h
 * @brief Checked CPU QSA invocation with native original-storage score/value execution.
 *
 * Admission seals CPU-owned replay, original pointer roots, complete read/write
 * exclusions and every consumed selected-ring address before worksharing. Each
 * row/head then streams original codec fragments through the declared eight- or
 * sixteen-chain score program. The rotated-value path retains only eight weighted
 * coordinates at a time and applies the original ordered transpose-FMA program;
 * no decoded cache head or full weighted-value mirror exists. Produced query
 * rotation and softmax logits are distinct algorithm operands, not cache shadows.
 * PhysicalMemoryAuthority/cache owners retain capacity and leases; these checked
 * synchronous descriptors neither allocate nor establish independent accounting.
 */
#pragma once
#include "CPUQwenSparseNativeCodec.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <type_traits>
namespace llaminar2::qwen4exp::cpu_sparse
{
namespace native=::llaminar2::cpu::qsa_native;
/** @brief Authenticate the original replay interval before any caller loads or forms member addresses.
 * @param replay Original untrusted CPU replay root, with owner-retained storage/lease.
 * @throws std::invalid_argument For null, wrapping or misaligned original bytes.
 * @details This is an address-domain proof, not an allocation/capacity claim.
 */
inline void validateReplayRoot(const QwenSparseReplay*replay)
{(void)native_bank::Range::bind(replay,sizeof(QwenSparseReplay),alignof(QwenSparseReplay));}
/** @brief Validate the original CPU-owned causal replay before any publication.
 * @param replay Original transaction state; @param rows Admitted candidate rows.
 * @param tokens Original physical/logical token limit.
 * @throws std::invalid_argument For noncausal, overflowing or unrepresentable state.
 */
inline void validateReplay(const QwenSparseReplay&replay,int rows,int tokens)
{
    if(replay.prefix_tokens<0||replay.live_rows<0||replay.live_rows>rows||
        std::int64_t(replay.prefix_tokens)+replay.live_rows>tokens||replay.position_origin<0||
        std::int64_t(replay.position_origin)+replay.prefix_tokens+replay.live_rows>std::numeric_limits<int>::max())
        throw std::invalid_argument("CPU Qwen sparse replay: invalid causal prefix or live row count");
}
/** @brief Select one original native decoder outside its coordinate/selected-row loops.
 * @tparam Function Original operation accepting a stable decoder.
 * @param binding Complete original bank; @param head Original local KV head.
 * @param function Synchronous native consumer; @return Original consumer result.
 */
template<class Function>decltype(auto) withDecoder(const native_bank::Binding&binding,int head,Function&&function)noexcept
{
#define QSA_CODEC_CASE(codec) case TensorType::codec:{const auto decoder=native::Decoder<TensorType::codec>::bind(binding,head);return function(decoder);}
    switch(binding.original().codec)
    {
    QSA_CODEC_CASE(FP32);QSA_CODEC_CASE(FP16);QSA_CODEC_CASE(BF16);QSA_CODEC_CASE(Q8_1);
    QSA_CODEC_CASE(Q16_1);QSA_CODEC_CASE(AQ8);QSA_CODEC_CASE(TQ4);QSA_CODEC_CASE(TQ8);
    default:native::rejectCodec();
    }
#undef QSA_CODEC_CASE
}
#if defined(__AVX512F__)
/** @brief Join adjacent original register fragments into the declared sixteen-lane program.
 * @param low Original coordinates0..7; @param high Original coordinates8..15.
 * @return Sixteen original coordinates, without FP32 staging storage.
 */
[[gnu::always_inline]] inline __m512 join16(native::Fragment8 low,native::Fragment8 high)noexcept
{return _mm512_insertf32x8(_mm512_castps256_ps512(low),high,1);}
#endif
/** @brief Score an original selected row with the exact declared native chain tree.
 * @tparam Row Stable original physical-head borrow; @param row Original codec row.
 * @param query Original/produced query basis; @param isa Cold-admitted execution ISA.
 * @return Original finite dot, with no decoded head or whole-operation scalar replay.
 */
template<class Row>[[gnu::always_inline]] inline float score(const Row&row,const float*query,ISALevel isa)noexcept
{
#if defined(__AVX512F__)
    if(isa==ISALevel::AVX512)
    {
        auto total=_mm512_setzero_ps();
        for(int c=0;c<256;c+=16)total=_mm512_fmadd_ps(_mm512_loadu_ps(query+c),join16(row.read(c),row.read(c+8)),total);
        return native::fold16(total);
    }
#else
    (void)isa;
#endif
    auto total=native::zero8();
    for(int c=0;c<256;c+=8)total=native::fma8(total,native::load8(query+c),row.read(c));
    return native::fold8(total);
}
/** @brief Extract one already-computed weighted coordinate without a FP32 lane array.
 * @tparam Lane Original coordinate within a native eight-coordinate fragment.
 * @param values Original computed lanes; @return The exact original FP32 word.
 */
template<unsigned Lane>[[gnu::always_inline]] inline float lane(native::Fragment8 values)noexcept
{
    static_assert(Lane<8);
#if LLAMINAR_COMPILED_WITH_NEON
    if constexpr(Lane<4)return vgetq_lane_f32(values.low,Lane);
    else return vgetq_lane_f32(values.high,Lane-4);
#else
    const auto part=Lane<4?_mm256_castps256_ps128(values):_mm256_extractf128_ps(values,1);
    return _mm_cvtss_f32(_mm_shuffle_ps(part,part,_MM_SHUFFLE(Lane%4,Lane%4,Lane%4,Lane%4)));
#endif
}
/** @brief Apply one original transpose coordinate in the fixed increasing-coordinate FMA order.
 * @tparam Lane Original lane0..7; @param weighted Completed eight weighted coordinates.
 * @param matrix Original admitted row-major rotation; @param base Original coordinate base.
 * @param output Exclusive spatial-basis result; @param isa Admitted native program.
 */
template<unsigned Lane>[[gnu::always_inline]] inline void transposeCoordinate(native::Fragment8 weighted,const float*matrix,int base,float*output,ISALevel isa)noexcept
{
    const auto value=lane<Lane>(weighted);const auto*coefficients=matrix+std::size_t(base+Lane)*256;
#if defined(__AVX512F__)
    if(isa==ISALevel::AVX512)
    {
        const auto scale=_mm512_set1_ps(value);
        for(int c=0;c<256;c+=16)
        {
            const auto lo=native::load8(coefficients+c),hi=native::load8(coefficients+c+8);
            if(!native::finite8(lo)||!native::finite8(hi))native::rejectCodec();
            const auto result=_mm512_fmadd_ps(join16(lo,hi),scale,_mm512_loadu_ps(output+c));
            _mm512_storeu_ps(output+c,result);
        }
        return;
    }
#else
    (void)isa;
#endif
    const auto scale=native::broadcast8(value);
    for(int c=0;c<256;c+=8)
    {
        const auto coefficients8=native::load8(coefficients+c);if(!native::finite8(coefficients8))native::rejectCodec();
        native::store8(output+c,native::fma8(native::load8(output+c),coefficients8,scale));
    }
}
/** @brief Sequence the exact original eight adjacent transpose coordinates, not an arithmetic reduction.
 * @tparam Lanes Increasing compile-time coordinate indices; @param weighted Original weighted fragment.
 * @param matrix Original rotation; @param base Original coordinate base.
 * @param output Exclusive result; @param isa Admitted native program; @param indices Ordered index token.
 */
template<unsigned...Lanes>[[gnu::always_inline]] inline void transposeFragment(native::Fragment8 weighted,const float*matrix,int base,float*output,ISALevel isa,std::integer_sequence<unsigned,Lanes...>indices)noexcept
{
    (void)indices;(transposeCoordinate<Lanes>(weighted,matrix,base,output,isa),...);
}

/** @brief One complete checked CPU invocation; only this class authors row/head publications. */
class Invocation final
{
public:
    /** @brief Seal original replay, addresses, exclusions and native program before collective execution.
     * @param p Existing arena/cache-owned public kernel bindings.
     * @return Synchronous invocation descriptor, never a capacity/lease or host device-state shadow.
     * @throws std::invalid_argument Before stores for unsupported policy/ISA, malformed
     * replay/ring/selected indices, wrapping metadata or conflicting publications.
     */
    [[nodiscard]] static Invocation bind(const QwenSparseAttention&p)
    {
        validateQwenSparseAttentionArithmetic(p.arithmetic,DeviceType::CPU);validateQwenSparseAttention(p);
        const auto isa=activeISALevel();native::requireNativeProgram(isa);
        // Replay is an original untrusted root: its interval must be admitted
        // before even the first CPU-owned word is copied or interpreted.
        validateReplayRoot(p.replay);
        const auto replay=*p.replay;validateReplay(replay,p.row_capacity,p.token_capacity);
        if(p.kv.key_policy.encoding!=attention::AttentionKeyCacheEncoding::PostRotary)
            throw std::invalid_argument("CPU Qwen sparse attention requires canonical post-RoPE cache publication");
        const auto keys=native_bank::Binding::bind(p.kv.keys,detail::BankRole::Key,p.kv_heads,p.token_capacity);
        const auto values=native_bank::Binding::bind(p.kv.values,detail::BankRole::Value,p.kv_heads,p.token_capacity);
        const auto rowHeads=std::size_t(p.row_capacity)*p.query_heads;
        const auto output=range(p.output,rowHeads*256);exclude(p,keys,values,output,true);
        if(p.observations)
        {
            const auto observations=range(p.observations,rowHeads);exclude(p,keys,values,observations,false);
            if(observations.overlaps(output))throw std::invalid_argument("CPU QSA output and observations overlap");
        }
        for(int row=0;row<replay.live_rows;++row)
        {
            int origin=0,capacity=0;
            if(!detail::resolveMapping(p.kv,replay,row,p.token_capacity,origin,capacity))
                throw std::invalid_argument("CPU Qwen sparse attention: stale or invalid canonical ring mapping");
            const auto count=p.selected_count[row];if(count<=0||count>QwenSparseSelected)
                throw std::invalid_argument("CPU Qwen sparse attention: invalid selected live count");
            int previous=-1;for(int i=0;i<count;++i)
            {
                const auto token=p.selected[std::size_t(row)*QwenSparseSelected+i];
                if(token<=previous||token>=replay.prefix_tokens+row+1)
                    throw std::invalid_argument("CPU Qwen sparse attention: noncausal, repeated, or unsorted selected index");
                previous=token;
            }
        }
        return Invocation(p,replay,isa,keys,values);
    }
    /** @return Exact original live row/head work, including the zero-work collective case. */
    [[nodiscard]] int rowHeads()const noexcept{return replay_.live_rows*p_.query_heads;}
    /** @return Exact admitted live row count. */
    [[nodiscard]] int liveRows()const noexcept{return replay_.live_rows;}
    /** @return Exact admitted local query-head count. */
    [[nodiscard]] int queryHeads()const noexcept{return p_.query_heads;}
    /** @return Global starting query-head ordinal for GQA KV-group mapping. */
    [[nodiscard]] int queryHeadStart()const noexcept{return p_.query_head_start;}
    /** @return Cold-admitted execution ISA for register-group sizing. */
    [[nodiscard]] ISALevel isa()const noexcept{return isa_;}
    /** @return Whether either key or value bank uses a rotated basis (TQ4/TQ8). */
    [[nodiscard]] bool isRotated()const noexcept{return detail::isRotated(p_.kv.keys)||detail::isRotated(p_.kv.values);}
    /** @return Selected token count for admitted live row @p row. */
    [[nodiscard]] int selectedCount(int row)const noexcept{return p_.selected_count[row];}
    /**
     * @brief Execute up to 4 single-row GQA KV groups together in ONE 3-barrier cooperative pass.
     *
     * Fuses K-scoring across `group_count * count`, softmax+transpose across all `total_heads`
     * (e.g. 24 on 2xCPU TP2, 48 on 1xCPU), and V-accumulation across `group_count * 16` AVX-512
     * strips (`32` on 2xCPU TP2 -> all 28 cores busy; `64` on 1xCPU) while preserving exact
     * increasing-token order. Bitwise identical to @ref executeHeadGroup.
     */
    struct CooperativeGroupSpan { int begin; int count; };

    void executeSingleRowGroupsCooperativeInTeam(
        int row,
        const CooperativeGroupSpan *groups,
        int groupCount,
        float *shared_logits) const noexcept
    {
        if (row < 0 || row >= replay_.live_rows || !groups || groupCount <= 0 || groupCount > 4 || !shared_logits)
            native::rejectCodec();
        if (groupCount == 1)
        {
            executeSingleRowGroupCooperativeInTeam(row, groups[0].begin, groups[0].count, shared_logits);
            return;
        }
        bool all_twelve = true;
        int total_heads = 0;
        for (int g = 0; g < groupCount; ++g)
        {
            if (groups[g].count != 12)
                all_twelve = false;
            total_heads += groups[g].count;
        }
        if (!all_twelve)
        {
            for (int g = 0; g < groupCount; ++g)
                executeSingleRowGroupCooperativeInTeam(row, groups[g].begin, groups[g].count, shared_logits);
            return;
        }
        executeSingleRowMultiGroup12CooperativeInTeam(row, groups, groupCount, shared_logits);
    }

    /**
     * @brief Execute one single-row GQA head group cooperatively across the entire active OpenMP team.
     *
     * Must be called identically by every thread of an active team (no internal fork).
     * Parallelizes K-scoring across `0..count` (decoding each K row ONCE for all `headCount` heads),
     * softmax across heads, and V-accumulation across independent 16-float (AVX-512) or 8-float (AVX2)
     * coordinate strips while walking `i = 0..count-1` in exact original order. Bitwise identical to
     * @ref executeHeadGroup.
     */
    void executeSingleRowGroupCooperativeInTeam(int row,int headBegin,int headCount,float*shared_logits)const noexcept
    {
        if(row<0||row>=replay_.live_rows||headBegin<0||headCount<=0||headCount>12||
           headBegin+headCount>p_.query_heads||!shared_logits)native::rejectCodec();
        switch(headCount)
        {
        case 1: return executeSingleRowGroupCooperativeFixed<1>(row,headBegin,shared_logits);
        case 2: return executeSingleRowGroupCooperativeFixed<2>(row,headBegin,shared_logits);
        case 3: return executeSingleRowGroupCooperativeFixed<3>(row,headBegin,shared_logits);
        case 4: return executeSingleRowGroupCooperativeFixed<4>(row,headBegin,shared_logits);
        case 6: return executeSingleRowGroupCooperativeFixed<6>(row,headBegin,shared_logits);
        case 12: return executeSingleRowGroupCooperativeFixed<12>(row,headBegin,shared_logits);
        default:
            for(int h=0;h<headCount;++h)executeSingleRowGroupCooperativeFixed<1>(row,headBegin+h,shared_logits);
            return;
        }
    }
    /** @brief Execute one admitted contiguous GQA head group sharing a single KV head.
     * @param row Original invocation-local live row index.
     * @param headBegin Starting local query-head index within this row.
     * @param headCount Number of contiguous query heads (1..12) sharing one KV head.
     * @details Dispatches compile-time head-group sizes (1, 2, 3, 4, 6, 12) so every
     * score and value accumulator stays in hardware vector registers with zero stack
     * spills, streaming each native codec fragment once across all heads in the group.
     */
    void executeHeadGroup(int row,int headBegin,int headCount)const noexcept
    {
        if(row<0||row>=replay_.live_rows||headBegin<0||headCount<=0||headCount>12||headBegin+headCount>p_.query_heads)native::rejectCodec();
        switch(headCount)
        {
        case 1: return executeHeadGroupFixed<1>(row,headBegin);
        case 2: return executeHeadGroupFixed<2>(row,headBegin);
        case 3: return executeHeadGroupFixed<3>(row,headBegin);
        case 4: return executeHeadGroupFixed<4>(row,headBegin);
        case 6: return executeHeadGroupFixed<6>(row,headBegin);
        case 12: return executeHeadGroupFixed<12>(row,headBegin);
        default:
            for(int h=0;h<headCount;++h)executeHeadGroupFixed<1>(row,headBegin+h);
            return;
        }
    }
    /** @brief Execute one admitted independent row/head through the native original-storage program.
     * @param rowHead Original invocation-local row/head index, never an inactive capacity row.
     * @details The caller gives each index one worksharing owner and completes
     * its retained team before returning. Numeric corruption remains fatal;
     * earlier retired rows are not rolled back or repaired to a second mode.
     */
    void execute(int rowHead)const noexcept
    {
        if(rowHead<0||rowHead>=rowHeads())native::rejectCodec();
        executeHeadGroupFixed<1>(rowHead/p_.query_heads,rowHead%p_.query_heads);
    }
private:
    /** @brief Compile-time fixed-group native execution keeping all accumulators in vector registers. */
    template<int HeadCount>
    void executeHeadGroupFixed(int row,int headBegin)const noexcept
    {
        static_assert(HeadCount>=1&&HeadCount<=12);
        const int kvHead=(p_.query_head_start+headBegin)/12-p_.kv_head_start;
        if((p_.query_head_start+headBegin+HeadCount-1)/12-p_.kv_head_start!=kvHead)native::rejectCodec();
        const auto count=p_.selected_count[row];if(count<=0||count>QwenSparseSelected)native::rejectCodec();
        const auto*selected=p_.selected+std::size_t(row)*QwenSparseSelected;
        int origin=0,capacity=0;if(!detail::resolveMapping(p_.kv,replay_,row,p_.token_capacity,origin,capacity))native::rejectCodec();

        const float*queries[HeadCount];
        float*outputs[HeadCount];
        float rotatedQueries[HeadCount][256];
        const bool keyRotated=detail::isRotated(p_.kv.keys);
        for(int h=0;h<HeadCount;++h)
        {
            const int rowHead=row*p_.query_heads+headBegin+h;
            const float*q=p_.query+std::size_t(rowHead)*256;
            if(keyRotated)
            {
                for(int c=0;c<256;++c)
                {
                    rotatedQueries[h][c]=native::dot<256>(p_.kv.keys.rotations[kvHead]+c*256,q,isa_);
                    if(!std::isfinite(rotatedQueries[h][c]))native::rejectCodec();
                }
                queries[h]=rotatedQueries[h];
            }
            else queries[h]=q;
            outputs[h]=p_.output+std::size_t(rowHead)*256;
        }

        float logits[HeadCount][QwenSparseSelected];
        float maximum[HeadCount];
        for(int h=0;h<HeadCount;++h)maximum[h]=-std::numeric_limits<float>::infinity();

        withDecoder(keys_,kvHead,[&](const auto&decoder)
        {
            for(int i=0;i<count;++i)
            {
                const auto nativeRow=decoder.row(detail::physicalRow(selected[i],origin,capacity));
#if defined(__AVX512F__)
                if(isa_==ISALevel::AVX512)
                {
                    __m512 totals[HeadCount];
                    for(int h=0;h<HeadCount;++h)totals[h]=_mm512_setzero_ps();
                    for(int c=0;c<256;c+=16)
                    {
                        const __m512 k=join16(nativeRow.read(c),nativeRow.read(c+8));
                        for(int h=0;h<HeadCount;++h)
                            totals[h]=_mm512_fmadd_ps(_mm512_loadu_ps(queries[h]+c),k,totals[h]);
                    }
                    for(int h=0;h<HeadCount;++h)
                    {
                        const float s=native::fold16(totals[h])*0.0625f;
                        if(!std::isfinite(s))native::rejectCodec();
                        logits[h][i]=s;maximum[h]=std::max(maximum[h],s);
                    }
                    continue;
                }
#endif
                if constexpr(HeadCount<=6)
                {
                    native::Fragment8 totals[HeadCount];
                    for(int h=0;h<HeadCount;++h)totals[h]=native::zero8();
                    for(int c=0;c<256;c+=8)
                    {
                        const auto k=nativeRow.read(c);
                        for(int h=0;h<HeadCount;++h)
                            totals[h]=native::fma8(totals[h],native::load8(queries[h]+c),k);
                    }
                    for(int h=0;h<HeadCount;++h)
                    {
                        const float s=native::fold8(totals[h])*0.0625f;
                        if(!std::isfinite(s))native::rejectCodec();
                        logits[h][i]=s;maximum[h]=std::max(maximum[h],s);
                    }
                }
                else
                {
                    // Split into two 6-head register halves on AVX2/NEON to stay spill-free.
                    for(int half=0;half<HeadCount;half+=6)
                    {
                        native::Fragment8 totals[6];
                        for(int h=0;h<6;++h)totals[h]=native::zero8();
                        for(int c=0;c<256;c+=8)
                        {
                            const auto k=nativeRow.read(c);
                            for(int h=0;h<6;++h)
                                totals[h]=native::fma8(totals[h],native::load8(queries[half+h]+c),k);
                        }
                        for(int h=0;h<6;++h)
                        {
                            const float s=native::fold8(totals[h])*0.0625f;
                            if(!std::isfinite(s))native::rejectCodec();
                            logits[half+h][i]=s;maximum[half+h]=std::max(maximum[half+h],s);
                        }
                    }
                }
            }
        });

        for(int h=0;h<HeadCount;++h)
        {
            float denominator=0.0f;
            const float m=maximum[h];
            for(int i=0;i<count;++i){logits[h][i]=std::exp(logits[h][i]-m);denominator+=logits[h][i];}
            if(!std::isfinite(denominator)||denominator<=0.0f)native::rejectCodec();
            for(int i=0;i<count;++i)logits[h][i]/=denominator;
        }

        withDecoder(values_,kvHead,[&](const auto&decoder)
        {
            if constexpr(std::remove_cvref_t<decltype(decoder)>::rotatedBasis)
            {
                for(int h=0;h<HeadCount;++h)std::fill_n(outputs[h],256,0.0f);
                for(int base=0;base<256;base+=8)
                {
                    native::Fragment8 weighted[HeadCount];
                    for(int h=0;h<HeadCount;++h)weighted[h]=native::zero8();
                    for(int i=0;i<count;++i)
                    {
                        const auto frag=decoder.row(detail::physicalRow(selected[i],origin,capacity)).read(base);
                        for(int h=0;h<HeadCount;++h)
                            weighted[h]=native::fma8(weighted[h],frag,native::broadcast8(logits[h][i]));
                    }
                    for(int h=0;h<HeadCount;++h)
                    {
                        if(!native::finite8(weighted[h]))native::rejectCodec();
                        transposeFragment(weighted[h],p_.kv.values.rotations[kvHead],base,outputs[h],isa_,std::make_integer_sequence<unsigned,8>{});
                    }
                }
            }
            else
            {
#if defined(__AVX512F__)
                if(isa_==ISALevel::AVX512)
                {
                    if constexpr(HeadCount==1)
                    {
                        __m512 acc[16];
                        for(int t=0;t<16;++t)acc[t]=_mm512_setzero_ps();
                        for(int i=0;i<count;++i)
                        {
                            const auto source=decoder.row(detail::physicalRow(selected[i],origin,capacity));
                            const __m512 prob=_mm512_set1_ps(logits[0][i]);
                            for(int t=0;t<16;++t)
                                acc[t]=_mm512_fmadd_ps(prob,join16(source.read(t*16),source.read(t*16+8)),acc[t]);
                        }
                        for(int t=0;t<16;++t)_mm512_storeu_ps(outputs[0]+t*16,acc[t]);
                    }
                    else
                    {
                        for(int c=0;c<256;c+=32)
                        {
                            __m512 acc0[HeadCount],acc1[HeadCount];
                            for(int h=0;h<HeadCount;++h){acc0[h]=_mm512_setzero_ps();acc1[h]=_mm512_setzero_ps();}
                            for(int i=0;i<count;++i)
                            {
                                const auto source=decoder.row(detail::physicalRow(selected[i],origin,capacity));
                                const __m512 v0=join16(source.read(c),source.read(c+8));
                                const __m512 v1=join16(source.read(c+16),source.read(c+24));
                                for(int h=0;h<HeadCount;++h)
                                {
                                    const __m512 prob=_mm512_set1_ps(logits[h][i]);
                                    acc0[h]=_mm512_fmadd_ps(prob,v0,acc0[h]);
                                    acc1[h]=_mm512_fmadd_ps(prob,v1,acc1[h]);
                                }
                            }
                            for(int h=0;h<HeadCount;++h)
                            {
                                _mm512_storeu_ps(outputs[h]+c,acc0[h]);
                                _mm512_storeu_ps(outputs[h]+c+16,acc1[h]);
                            }
                        }
                    }
                    return;
                }
#endif
                if constexpr(HeadCount<=6)
                {
                    for(int c=0;c<256;c+=16)
                    {
                        native::Fragment8 acc0[HeadCount],acc1[HeadCount];
                        for(int h=0;h<HeadCount;++h){acc0[h]=native::zero8();acc1[h]=native::zero8();}
                        for(int i=0;i<count;++i)
                        {
                            const auto source=decoder.row(detail::physicalRow(selected[i],origin,capacity));
                            const auto v0=source.read(c);
                            const auto v1=source.read(c+8);
                            for(int h=0;h<HeadCount;++h)
                            {
                                const auto prob=native::broadcast8(logits[h][i]);
                                acc0[h]=native::fma8(acc0[h],prob,v0);
                                acc1[h]=native::fma8(acc1[h],prob,v1);
                            }
                        }
                        for(int h=0;h<HeadCount;++h)
                        {
                            native::store8(outputs[h]+c,acc0[h]);
                            native::store8(outputs[h]+c+8,acc1[h]);
                        }
                    }
                }
                else
                {
                    for(int half=0;half<HeadCount;half+=6)
                    {
                        for(int c=0;c<256;c+=16)
                        {
                            native::Fragment8 acc0[6],acc1[6];
                            for(int h=0;h<6;++h){acc0[h]=native::zero8();acc1[h]=native::zero8();}
                            for(int i=0;i<count;++i)
                            {
                                const auto source=decoder.row(detail::physicalRow(selected[i],origin,capacity));
                                const auto v0=source.read(c);
                                const auto v1=source.read(c+8);
                                for(int h=0;h<6;++h)
                                {
                                    const auto prob=native::broadcast8(logits[half+h][i]);
                                    acc0[h]=native::fma8(acc0[h],prob,v0);
                                    acc1[h]=native::fma8(acc1[h],prob,v1);
                                }
                            }
                            for(int h=0;h<6;++h)
                            {
                                native::store8(outputs[half+h]+c,acc0[h]);
                                native::store8(outputs[half+h]+c+8,acc1[h]);
                            }
                        }
                    }
                }
            }
        });

        for(int h=0;h<HeadCount;++h)
        {
            const int rowHead=row*p_.query_heads+headBegin+h;
            float*output=outputs[h];
            for(int c=0;c<256;++c)
            {
                if(p_.gate)
                {const auto gate=p_.gate[std::size_t(rowHead)*256+c];if(!std::isfinite(gate))native::rejectCodec();output[c]*=1.0f/(1.0f+std::exp(-gate));}
                if(!std::isfinite(output[c]))native::rejectCodec();
            }
            if(p_.observations)p_.observations[rowHead]={std::uint64_t(count)*256,std::uint64_t(count)*256,std::uint32_t(count)};
        }
    }

    /**
     * @brief Fused 3-barrier cooperative decode across `groupCount` (2..4) full 12-head GQA KV groups.
     *
     * Uses 64-byte-padded per-head contiguous logits/probs inside `shared_logits` (capacity >=
     * `4 * 12 * 2064` floats) so Phase 2 normalizes in-place with ZERO false sharing across heads.
     */
    void executeSingleRowMultiGroup12CooperativeInTeam(
        int row,
        const CooperativeGroupSpan *groups,
        int groupCount,
        float *shared_logits) const noexcept
    {
        constexpr int HeadCount = 12;
        constexpr std::size_t kMaxHeadStride = (static_cast<std::size_t>(QwenSparseSelected) + 15u) & ~std::size_t{15u};
        constexpr std::size_t kGroupStride = static_cast<std::size_t>(HeadCount) * kMaxHeadStride;
        const auto count = p_.selected_count[row];
        if (count <= 0 || count > QwenSparseSelected)
            native::rejectCodec();
        const std::size_t head_stride = (static_cast<std::size_t>(count) + 15u) & ~std::size_t{15u};
        const auto *selected = p_.selected + std::size_t(row) * QwenSparseSelected;
        int origin = 0, capacity = 0;
        if (!detail::resolveMapping(p_.kv, replay_, row, p_.token_capacity, origin, capacity))
            native::rejectCodec();

        int kvHeads[4] = {};
        for (int g = 0; g < groupCount; ++g)
        {
            const int headBegin = groups[g].begin;
            if (headBegin < 0 || headBegin + HeadCount > p_.query_heads)
                native::rejectCodec();
            kvHeads[g] = (p_.query_head_start + headBegin) / 12 - p_.kv_head_start;
            if ((p_.query_head_start + headBegin + HeadCount - 1) / 12 - p_.kv_head_start != kvHeads[g])
                native::rejectCodec();
        }

        // Phase 1: Parallel K-scoring in 16-token blocks (zero boundary false sharing) with
        // codec dispatch hoisted outside the workshare and real L1 prefetch on selected[i+2].
        const int blocks16 = (count + 15) / 16;
        const int total_k_blocks = groupCount * blocks16;
        withDecoder(keys_, kvHeads[0], [&](const auto &proto_decoder) {
            using DecoderT = std::remove_cvref_t<decltype(proto_decoder)>;
#pragma omp for schedule(static)
            for (int task = 0; task < total_k_blocks; ++task)
            {
                const int g = task / blocks16;
                const int bi = (task % blocks16) * 16;
                const int bend = std::min(bi + 16, count);
                const int headBegin = groups[g].begin;
                const auto decoder = DecoderT::bind(keys_, kvHeads[g]);
                float *g_logits = shared_logits + static_cast<std::size_t>(g) * kGroupStride;
                const float *queries[HeadCount];
                for (int h = 0; h < HeadCount; ++h)
                    queries[h] = p_.query + std::size_t(row * p_.query_heads + headBegin + h) * 256;

                for (int i = bi; i < bend; ++i)
                {
#if defined(__AVX512F__) || defined(__AVX2__)
                    if (i + 2 < bend)
                    {
                        const auto *pb = decoder.row(detail::physicalRow(selected[i + 2], origin, capacity)).rawBytes();
                        _mm_prefetch(reinterpret_cast<const char *>(pb), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char *>(pb + 128), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char *>(pb + 256), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char *>(pb + 384), _MM_HINT_T0);
                    }
#endif
                    const auto nativeRow = decoder.row(detail::physicalRow(selected[i], origin, capacity));
#if defined(__AVX512F__)
                    if (isa_ == ISALevel::AVX512)
                    {
                        __m512 totals[HeadCount];
                        for (int h = 0; h < HeadCount; ++h) totals[h] = _mm512_setzero_ps();
                        for (int c = 0; c < 256; c += 16)
                        {
                            const __m512 k = nativeRow.read16AVX512(c);
                            for (int h = 0; h < HeadCount; ++h)
                                totals[h] = _mm512_fmadd_ps(_mm512_loadu_ps(queries[h] + c), k, totals[h]);
                        }
                        for (int h = 0; h < HeadCount; ++h)
                        {
                            const float s = native::fold16(totals[h]) * 0.0625f;
                            if (!std::isfinite(s)) native::rejectCodec();
                            g_logits[static_cast<std::size_t>(h) * head_stride + i] = s;
                        }
                        continue;
                    }
#endif
                    for (int half = 0; half < HeadCount; half += 6)
                    {
                        native::Fragment8 totals[6];
                        for (int h = 0; h < 6; ++h) totals[h] = native::zero8();
                        for (int c = 0; c < 256; c += 8)
                        {
                            const auto k = nativeRow.read(c);
                            for (int h = 0; h < 6; ++h)
                                totals[h] = native::fma8(totals[h], native::load8(queries[half + h] + c), k);
                        }
                        for (int h = 0; h < 6; ++h)
                        {
                            const float s = native::fold8(totals[h]) * 0.0625f;
                            if (!std::isfinite(s)) native::rejectCodec();
                            g_logits[static_cast<std::size_t>(half + h) * head_stride + i] = s;
                        }
                    }
                }
            }
        });

        // Phase 2: In-place exact softmax across ALL groupCount * 12 heads (64B-disjoint rows, ZERO false sharing).
        const int total_heads = groupCount * HeadCount;
#pragma omp for schedule(static)
        for (int gh = 0; gh < total_heads; ++gh)
        {
            const int g = gh / HeadCount;
            const int h = gh % HeadCount;
            float *g_logits = shared_logits + static_cast<std::size_t>(g) * kGroupStride;
            float *row_logits = g_logits + static_cast<std::size_t>(h) * head_stride;
            float m = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < count; ++i) m = std::max(m, row_logits[i]);
            float denominator = 0.0f;
            for (int i = 0; i < count; ++i) { row_logits[i] = std::exp(row_logits[i] - m); denominator += row_logits[i]; }
            if (!std::isfinite(denominator) || denominator <= 0.0f) native::rejectCodec();
            for (int i = 0; i < count; ++i) row_logits[i] = row_logits[i] / denominator;
        }

        // Phase 3: Parallel V-accumulation across ALL groupCount * 16 strips * 3 4-head slices
        // (AVX-512 reads only 4 contiguous streams directly from g_logits, eliminating Phase 2b transpose & barrier!).
#if defined(__AVX512F__)
        if (isa_ == ISALevel::AVX512)
        {
            constexpr int kHeadSlices = 3;
            constexpr int kSliceHeads = HeadCount / kHeadSlices; // 4
            const int total_tasks = groupCount * 16 * kHeadSlices;
            withDecoder(values_, kvHeads[0], [&](const auto &proto_decoder) {
                using DecoderT = std::remove_cvref_t<decltype(proto_decoder)>;
                if constexpr (!DecoderT::rotatedBasis)
                {
#pragma omp for schedule(static)
                    for (int task = 0; task < total_tasks; ++task)
                    {
                        const int g = task / (16 * kHeadSlices);
                        const int rem = task % (16 * kHeadSlices);
                        const int c = (rem / kHeadSlices) * 16;
                        const int h0 = (rem % kHeadSlices) * kSliceHeads;
                        const int headBegin = groups[g].begin;
                        const auto decoder = DecoderT::bind(values_, kvHeads[g]);
                        const float *g_logits = shared_logits + static_cast<std::size_t>(g) * kGroupStride;
                        const float *p_rows[kSliceHeads];
                        float *outputs[kSliceHeads];
                        for (int h = 0; h < kSliceHeads; ++h)
                        {
                            p_rows[h] = g_logits + static_cast<std::size_t>(h0 + h) * head_stride;
                            outputs[h] = p_.output + std::size_t(row * p_.query_heads + headBegin + h0 + h) * 256;
                        }
                        __m512 acc[kSliceHeads];
                        for (int h = 0; h < kSliceHeads; ++h) acc[h] = _mm512_setzero_ps();
                        for (int i = 0; i < count; ++i)
                        {
                            if (i + 2 < count)
                            {
                                const auto *pb = decoder.row(detail::physicalRow(selected[i + 2], origin, capacity)).rawBytes();
                                _mm_prefetch(reinterpret_cast<const char *>(pb + (c * 2)), _MM_HINT_T0);
                            }
                            const auto source = decoder.row(detail::physicalRow(selected[i], origin, capacity));
                            const __m512 v = source.read16AVX512(c);
#pragma GCC unroll 4
                            for (int h = 0; h < kSliceHeads; ++h)
                                acc[h] = _mm512_fmadd_ps(_mm512_set1_ps(p_rows[h][i]), v, acc[h]);
                        }
                        const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);
                        const __m512 flt_max = _mm512_set1_ps(std::numeric_limits<float>::max());
                        for (int h = 0; h < kSliceHeads; ++h)
                        {
                            float *out = outputs[h] + c;
                            _mm512_storeu_ps(out, acc[h]);
                            if (__builtin_expect(p_.gate != nullptr, 0))
                            {
                                const int rowHead = row * p_.query_heads + headBegin + h0 + h;
                                for (int lane_i = 0; lane_i < 16; ++lane_i)
                                {
                                    const auto gv = p_.gate[std::size_t(rowHead) * 256 + c + lane_i];
                                    if (!std::isfinite(gv)) native::rejectCodec();
                                    out[lane_i] *= 1.0f / (1.0f + std::exp(-gv));
                                    if (!std::isfinite(out[lane_i])) native::rejectCodec();
                                }
                            }
                            else
                            {
                                const __m512 mag = _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(acc[h]), abs_mask));
                                if (__builtin_expect(_mm512_cmp_ps_mask(mag, flt_max, _CMP_LE_OQ) != 0xffffu, 0))
                                    native::rejectCodec();
                            }
                        }
                    }
                }
            });
        }
        else
#endif
        {
            // AVX2/NEON: transpose into probs_t for 6-head L1d locality.
            const int total_t_blocks = groupCount * blocks16;
#pragma omp for schedule(static)
            for (int task = 0; task < total_t_blocks; ++task)
            {
                const int g = task / blocks16;
                const int bi = (task % blocks16) * 16;
                const int bend = std::min(bi + 16, count);
                const float *g_logits = shared_logits + static_cast<std::size_t>(g) * kGroupStride;
                float *g_probs_t = shared_logits + static_cast<std::size_t>(4 + g) * kGroupStride;
                for (int i = bi; i < bend; ++i)
                {
                    float *dst = g_probs_t + static_cast<std::size_t>(i) * static_cast<std::size_t>(HeadCount);
#pragma GCC unroll 12
                    for (int h = 0; h < HeadCount; ++h)
                        dst[h] = g_logits[static_cast<std::size_t>(h) * head_stride + i];
                }
            }
            const int total_strips = groupCount * 32;
            withDecoder(values_, kvHeads[0], [&](const auto &proto_decoder) {
                using DecoderT = std::remove_cvref_t<decltype(proto_decoder)>;
                if constexpr (!DecoderT::rotatedBasis)
                {
#pragma omp for schedule(static)
                    for (int task = 0; task < total_strips; ++task)
                    {
                        const int g = task / 32;
                        const int c = (task % 32) * 8;
                        const int headBegin = groups[g].begin;
                        const auto decoder = DecoderT::bind(values_, kvHeads[g]);
                        const float *g_probs_t = shared_logits + static_cast<std::size_t>(4 + g) * kGroupStride;
                        float *outputs[HeadCount];
                        for (int h = 0; h < HeadCount; ++h)
                            outputs[h] = p_.output + std::size_t(row * p_.query_heads + headBegin + h) * 256;

                        for (int half = 0; half < HeadCount; half += 6)
                        {
                            native::Fragment8 acc[6];
                            for (int h = 0; h < 6; ++h) acc[h] = native::zero8();
                            for (int i = 0; i < count; ++i)
                            {
                                const auto source = decoder.row(detail::physicalRow(selected[i], origin, capacity));
                                const auto v = source.read(c);
                                const float *pi = g_probs_t + static_cast<std::size_t>(i) * static_cast<std::size_t>(HeadCount) + half;
                                for (int h = 0; h < 6; ++h)
                                    acc[h] = native::fma8(acc[h], native::broadcast8(pi[h]), v);
                            }
                            for (int h = 0; h < 6; ++h)
                            {
                                const int rowHead = row * p_.query_heads + headBegin + half + h;
                                float *out = outputs[half + h] + c;
                                native::store8(out, acc[h]);
                                for (int lane_i = 0; lane_i < 8; ++lane_i)
                                {
                                    if (p_.gate)
                                    {
                                        const auto gv = p_.gate[std::size_t(rowHead) * 256 + c + lane_i];
                                        if (!std::isfinite(gv)) native::rejectCodec();
                                        out[lane_i] *= 1.0f / (1.0f + std::exp(-gv));
                                    }
                                    if (!std::isfinite(out[lane_i])) native::rejectCodec();
                                }
                            }
                        }
                    }
                }
            });
        }

        if (p_.observations)
        {
#pragma omp single
            {
                for (int g = 0; g < groupCount; ++g)
                {
                    for (int h = 0; h < HeadCount; ++h)
                    {
                        const int rowHead = row * p_.query_heads + groups[g].begin + h;
                        p_.observations[rowHead] = {std::uint64_t(count) * 256, std::uint64_t(count) * 256, std::uint32_t(count)};
                    }
                }
            }
        }
    }

    template<int HeadCount>
    void executeSingleRowGroupCooperativeFixed(int row,int headBegin,float*shared_logits)const noexcept
    {
        static_assert(HeadCount>=1&&HeadCount<=12);
        const int kvHead=(p_.query_head_start+headBegin)/12-p_.kv_head_start;
        if((p_.query_head_start+headBegin+HeadCount-1)/12-p_.kv_head_start!=kvHead)native::rejectCodec();
        const auto count=p_.selected_count[row];if(count<=0||count>QwenSparseSelected)native::rejectCodec();
        const std::size_t head_stride=(static_cast<std::size_t>(count)+15u)&~std::size_t{15u};
        const auto*selected=p_.selected+std::size_t(row)*QwenSparseSelected;
        int origin=0,capacity=0;if(!detail::resolveMapping(p_.kv,replay_,row,p_.token_capacity,origin,capacity))native::rejectCodec();

        const float*queries[HeadCount];
        float*outputs[HeadCount];
        for(int h=0;h<HeadCount;++h)
        {
            const int rowHead=row*p_.query_heads+headBegin+h;
            queries[h]=p_.query+std::size_t(rowHead)*256;
            outputs[h]=p_.output+std::size_t(rowHead)*256;
        }

        // Phase 1: Parallel K-scoring in 16-token blocks (zero boundary false sharing on shared_logits!)
        // with real L1 prefetch on selected[i+2]; each KV row is decoded ONCE for all HeadCount heads.
        const int blocks16 = (count + 15) / 16;
        withDecoder(keys_,kvHead,[&](const auto&decoder)
        {
#pragma omp for schedule(static)
            for(int blk=0;blk<blocks16;++blk)
            {
                const int bi=blk*16;
                const int bend=std::min(bi+16,count);
                for(int i=bi;i<bend;++i)
                {
#if defined(__AVX512F__) || defined(__AVX2__)
                    if(i+2<bend)
                    {
                        const auto*pb=decoder.row(detail::physicalRow(selected[i+2],origin,capacity)).rawBytes();
                        _mm_prefetch(reinterpret_cast<const char*>(pb),_MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(pb+128),_MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(pb+256),_MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(pb+384),_MM_HINT_T0);
                    }
#endif
                    const auto nativeRow=decoder.row(detail::physicalRow(selected[i],origin,capacity));
#if defined(__AVX512F__)
                    if(isa_==ISALevel::AVX512)
                    {
                        __m512 totals[HeadCount];
                        for(int h=0;h<HeadCount;++h)totals[h]=_mm512_setzero_ps();
                        for(int c=0;c<256;c+=16)
                        {
                            const __m512 k=nativeRow.read16AVX512(c);
                            for(int h=0;h<HeadCount;++h)
                                totals[h]=_mm512_fmadd_ps(_mm512_loadu_ps(queries[h]+c),k,totals[h]);
                        }
                        for(int h=0;h<HeadCount;++h)
                        {
                            const float s=native::fold16(totals[h])*0.0625f;
                            if(!std::isfinite(s))native::rejectCodec();
                            shared_logits[static_cast<std::size_t>(h)*head_stride+i]=s;
                        }
                        continue;
                    }
#endif
                    if constexpr(HeadCount<=6)
                    {
                        native::Fragment8 totals[HeadCount];
                        for(int h=0;h<HeadCount;++h)totals[h]=native::zero8();
                        for(int c=0;c<256;c+=8)
                        {
                            const auto k=nativeRow.read(c);
                            for(int h=0;h<HeadCount;++h)
                                totals[h]=native::fma8(totals[h],native::load8(queries[h]+c),k);
                        }
                        for(int h=0;h<HeadCount;++h)
                        {
                            const float s=native::fold8(totals[h])*0.0625f;
                            if(!std::isfinite(s))native::rejectCodec();
                            shared_logits[static_cast<std::size_t>(h)*head_stride+i]=s;
                        }
                    }
                    else
                    {
                        for(int half=0;half<HeadCount;half+=6)
                        {
                            native::Fragment8 totals[6];
                            for(int h=0;h<6;++h)totals[h]=native::zero8();
                            for(int c=0;c<256;c+=8)
                            {
                                const auto k=nativeRow.read(c);
                                for(int h=0;h<6;++h)
                                    totals[h]=native::fma8(totals[h],native::load8(queries[half+h]+c),k);
                            }
                            for(int h=0;h<6;++h)
                            {
                                const float s=native::fold8(totals[h])*0.0625f;
                                if(!std::isfinite(s))native::rejectCodec();
                                shared_logits[static_cast<std::size_t>(half+h)*head_stride+i]=s;
                            }
                        }
                    }
                }
            }
        });

        // Phase 2: In-place exact softmax across heads on 64-byte-padded rows (ZERO cross-head false sharing).
#pragma omp for schedule(static)
        for(int h=0;h<HeadCount;++h)
        {
            float*row_logits=shared_logits+static_cast<std::size_t>(h)*head_stride;
            float m=-std::numeric_limits<float>::infinity();
            for(int i=0;i<count;++i)m=std::max(m,row_logits[i]);
            float denominator=0.0f;
            for(int i=0;i<count;++i){row_logits[i]=std::exp(row_logits[i]-m);denominator+=row_logits[i];}
            if(!std::isfinite(denominator)||denominator<=0.0f)native::rejectCodec();
            for(int i=0;i<count;++i)row_logits[i]=row_logits[i]/denominator;
        }

        // Phase 3 (AVX-512 HeadCount==12 fast path): 48 tasks reading ONLY 4 contiguous head streams
        // directly from shared_logits -> skips Phase 2b transpose & eliminates 1 full barrier per layer!
#if defined(__AVX512F__)
        if constexpr(HeadCount==12)
        {
            if(isa_==ISALevel::AVX512)
            {
                withDecoder(values_,kvHead,[&](const auto&decoder)
                {
                    if constexpr(!std::remove_cvref_t<decltype(decoder)>::rotatedBasis)
                    {
                        constexpr int kHeadSlices=3;
                        constexpr int kSliceHeads=4;
#pragma omp for schedule(static)
                        for(int task=0;task<16*kHeadSlices;++task)
                        {
                            const int c=(task/kHeadSlices)*16;
                            const int h0=(task%kHeadSlices)*kSliceHeads;
                            const float*p_rows[kSliceHeads];
                            for(int h=0;h<kSliceHeads;++h)
                                p_rows[h]=shared_logits+static_cast<std::size_t>(h0+h)*head_stride;
                            __m512 acc[kSliceHeads];
                            for(int h=0;h<kSliceHeads;++h)acc[h]=_mm512_setzero_ps();
                            for(int i=0;i<count;++i)
                            {
                                if(i+2<count)
                                {
                                    const auto*pb=decoder.row(detail::physicalRow(selected[i+2],origin,capacity)).rawBytes();
                                    _mm_prefetch(reinterpret_cast<const char*>(pb+(c*2)),_MM_HINT_T0);
                                }
                                const auto source=decoder.row(detail::physicalRow(selected[i],origin,capacity));
                                const __m512 v=source.read16AVX512(c);
#pragma GCC unroll 4
                                for(int h=0;h<kSliceHeads;++h)
                                    acc[h]=_mm512_fmadd_ps(_mm512_set1_ps(p_rows[h][i]),v,acc[h]);
                            }
                            const __m512i abs_mask=_mm512_set1_epi32(0x7fffffff);
                            const __m512 flt_max=_mm512_set1_ps(std::numeric_limits<float>::max());
                            for(int h=0;h<kSliceHeads;++h)
                            {
                                float*out=outputs[h0+h]+c;
                                _mm512_storeu_ps(out,acc[h]);
                                if(__builtin_expect(p_.gate!=nullptr,0))
                                {
                                    const int rowHead=row*p_.query_heads+headBegin+h0+h;
                                    for(int lane_i=0;lane_i<16;++lane_i)
                                    {
                                        const auto g=p_.gate[std::size_t(rowHead)*256+c+lane_i];
                                        if(!std::isfinite(g))native::rejectCodec();
                                        out[lane_i]*=1.0f/(1.0f+std::exp(-g));
                                        if(!std::isfinite(out[lane_i]))native::rejectCodec();
                                    }
                                }
                                else
                                {
                                    const __m512 mag=_mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(acc[h]),abs_mask));
                                    if(__builtin_expect(_mm512_cmp_ps_mask(mag,flt_max,_CMP_LE_OQ)!=0xffffu,0))
                                        native::rejectCodec();
                                }
                            }
                        }
                    }
                });
                if(p_.observations)
                {
#pragma omp single
                    {
                        for(int h=0;h<HeadCount;++h)
                        {
                            const int rowHead=row*p_.query_heads+headBegin+h;
                            p_.observations[rowHead]={std::uint64_t(count)*256,std::uint64_t(count)*256,std::uint32_t(count)};
                        }
                    }
                }
                return;
            }
        }
#endif

        // Phase 2b (AVX2 / non-12-head): Disjoint 16-token parallel transpose into contiguous probs_t[i * HeadCount + h].
        constexpr std::size_t kMaxPaddedSelected=(static_cast<std::size_t>(QwenSparseSelected)+15u)&~std::size_t{15u};
        float*probs_t=shared_logits+12u*kMaxPaddedSelected;
#pragma omp for schedule(static)
        for(int blk=0;blk<blocks16;++blk)
        {
            const int bi=blk*16;
            const int bend=std::min(bi+16,count);
            for(int i=bi;i<bend;++i)
            {
                float*dst=probs_t+static_cast<std::size_t>(i)*static_cast<std::size_t>(HeadCount);
                for(int h=0;h<HeadCount;++h)
                    dst[h]=shared_logits[static_cast<std::size_t>(h)*head_stride+i];
            }
        }

        // Phase 3: Parallel V-accumulation across independent coordinate strips, walking i=0..count-1
        // in exact original order so outputs are 100% bitwise identical to executeHeadGroupFixed.
        withDecoder(values_,kvHead,[&](const auto&decoder)
        {
            if constexpr(!std::remove_cvref_t<decltype(decoder)>::rotatedBasis)
            {
#if defined(__AVX512F__)
                if(isa_==ISALevel::AVX512)
                {
#pragma omp for schedule(static)
                    for(int strip=0;strip<16;++strip)
                    {
                        const int c=strip*16;
                        __m512 acc[HeadCount];
                        for(int h=0;h<HeadCount;++h)acc[h]=_mm512_setzero_ps();
                        for(int i=0;i<count;++i)
                        {
                            const auto source=decoder.row(detail::physicalRow(selected[i],origin,capacity));
                            const __m512 v=source.read16AVX512(c);
                            const float*pi=probs_t+static_cast<std::size_t>(i)*static_cast<std::size_t>(HeadCount);
#pragma GCC unroll 12
                            for(int h=0;h<HeadCount;++h)
                                acc[h]=_mm512_fmadd_ps(_mm512_set1_ps(pi[h]),v,acc[h]);
                        }
                        for(int h=0;h<HeadCount;++h)
                        {
                            const int rowHead=row*p_.query_heads+headBegin+h;
                            float*out=outputs[h]+c;
                            _mm512_storeu_ps(out,acc[h]);
                            for(int lane_i=0;lane_i<16;++lane_i)
                            {
                                if(p_.gate)
                                {
                                    const auto g=p_.gate[std::size_t(rowHead)*256+c+lane_i];
                                    if(!std::isfinite(g))native::rejectCodec();
                                    out[lane_i]*=1.0f/(1.0f+std::exp(-g));
                                }
                                if(!std::isfinite(out[lane_i]))native::rejectCodec();
                            }
                        }
                    }
                    return;
                }
#endif
#pragma omp for schedule(static)
                for(int strip=0;strip<32;++strip)
                {
                    const int c=strip*8;
                    if constexpr(HeadCount<=6)
                    {
                        native::Fragment8 acc[HeadCount];
                        for(int h=0;h<HeadCount;++h)acc[h]=native::zero8();
                        for(int i=0;i<count;++i)
                        {
                            const auto source=decoder.row(detail::physicalRow(selected[i],origin,capacity));
                            const auto v=source.read(c);
                            const float*pi=probs_t+static_cast<std::size_t>(i)*static_cast<std::size_t>(HeadCount);
                            for(int h=0;h<HeadCount;++h)
                                acc[h]=native::fma8(acc[h],native::broadcast8(pi[h]),v);
                        }
                        for(int h=0;h<HeadCount;++h)
                        {
                            const int rowHead=row*p_.query_heads+headBegin+h;
                            float*out=outputs[h]+c;
                            native::store8(out,acc[h]);
                            for(int lane_i=0;lane_i<8;++lane_i)
                            {
                                if(p_.gate)
                                {
                                    const auto g=p_.gate[std::size_t(rowHead)*256+c+lane_i];
                                    if(!std::isfinite(g))native::rejectCodec();
                                    out[lane_i]*=1.0f/(1.0f+std::exp(-g));
                                }
                                if(!std::isfinite(out[lane_i]))native::rejectCodec();
                            }
                        }
                    }
                    else
                    {
                        for(int half=0;half<HeadCount;half+=6)
                        {
                            native::Fragment8 acc[6];
                            for(int h=0;h<6;++h)acc[h]=native::zero8();
                            for(int i=0;i<count;++i)
                            {
                                const auto source=decoder.row(detail::physicalRow(selected[i],origin,capacity));
                                const auto v=source.read(c);
                                const float*pi=probs_t+static_cast<std::size_t>(i)*static_cast<std::size_t>(HeadCount)+half;
                                for(int h=0;h<6;++h)
                                    acc[h]=native::fma8(acc[h],native::broadcast8(pi[h]),v);
                            }
                            for(int h=0;h<6;++h)
                            {
                                const int rowHead=row*p_.query_heads+headBegin+half+h;
                                float*out=outputs[half+h]+c;
                                native::store8(out,acc[h]);
                                for(int lane_i=0;lane_i<8;++lane_i)
                                {
                                    if(p_.gate)
                                    {
                                        const auto g=p_.gate[std::size_t(rowHead)*256+c+lane_i];
                                        if(!std::isfinite(g))native::rejectCodec();
                                        out[lane_i]*=1.0f/(1.0f+std::exp(-g));
                                    }
                                    if(!std::isfinite(out[lane_i]))native::rejectCodec();
                                }
                            }
                        }
                    }
                }
            }
        });

        if(p_.observations)
        {
#pragma omp single
            {
                for(int h=0;h<HeadCount;++h)
                {
                    const int rowHead=row*p_.query_heads+headBegin+h;
                    p_.observations[rowHead]={std::uint64_t(count)*256,std::uint64_t(count)*256,std::uint32_t(count)};
                }
            }
        }
    }
private:
    /** @brief Retain sealed original roots/replay only, with no new allocation/lease authority.
     * @param p Original metadata; @param replay CPU-owned immutable transaction snapshot.
     * @param isa Selected native program; @param keys Original key binding; @param values Original value binding.
     */
    Invocation(const QwenSparseAttention&p,QwenSparseReplay replay,ISALevel isa,native_bank::Binding keys,native_bank::Binding values)noexcept:
        p_(p),replay_(replay),isa_(isa),keys_(keys),values_(values){}
    /** @brief Form a checked complete original element interval, without claiming capacity.
     * @tparam T Original element type; @param pointer Original root; @param count Declared original elements.
     * @return Checked byte interval; @throws std::invalid_argument For element-byte overflow/wrapping/alignment.
     */
    template<class T>static native_bank::Range range(const T*pointer,std::size_t count)
    {
        if(count>std::numeric_limits<std::size_t>::max()/sizeof(T))throw std::invalid_argument("CPU QSA original element extent overflow");
        return native_bank::Range::bind(pointer,count*sizeof(T),alignof(T));
    }
    /** @brief Exclude each publication from all consumed original read-only banks and metadata.
     * @param p Original caller roots; @param keys Original key authority; @param values Original value authority.
     * @param write Complete proposed publication; @param exactQueryAlias Whether full same-head query reuse is admitted.
     * @throws std::invalid_argument Before any publication for conflicting roots.
     */
    static void exclude(const QwenSparseAttention&p,const native_bank::Binding&keys,const native_bank::Binding&values,native_bank::Range write,bool exactQueryAlias)
    {
        keys.exclude(write);values.exclude(write);const auto rows=std::size_t(p.row_capacity);
        if(write.overlaps(range(p.query,rows*p.query_heads*256))&&(!exactQueryAlias||p.output!=p.query))
            throw std::invalid_argument("CPU QSA publication overlaps another original query interval");
        if(write.overlaps(range(p.replay,1))||write.overlaps(range(&p,1))||write.overlaps(range(p.selected,rows*QwenSparseSelected))||
            write.overlaps(range(p.selected_count,rows))||(p.gate&&write.overlaps(range(p.gate,rows*p.query_heads*256)))||
            (p.kv.addressing==QwenSparseKVAddressing::RowParameters&&write.overlaps(range(p.kv.row_parameters,rows))))
            throw std::invalid_argument("CPU QSA publication overlaps original immutable operands or metadata");
    }
    QwenSparseAttention p_;QwenSparseReplay replay_;ISALevel isa_;native_bank::Binding keys_,values_;
};
} // namespace llaminar2::qwen4exp::cpu_sparse
